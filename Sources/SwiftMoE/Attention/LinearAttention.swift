import Foundation
import Accelerate  // vDSP for BLAS-accelerated state updates

/// CPU-side GatedDeltaNet linear attention with BLAS acceleration.
///
/// Used by 45 of the 60 layers. Maintains persistent state (delta-net recurrence
/// matrix + conv1d history) across tokens. Uses Accelerate BLAS (cblas_sscal,
/// cblas_sgemv, cblas_sger) for the 64-head × 128×128 state update — 64% faster
/// than scalar code per the paper.
///
/// Matches `infer.m:4589-4746` (the CPU linear attention compute path).
public enum LinearAttention {

    /// Computes one step of GatedDeltaNet linear attention.
    ///
    /// Flow: conv1d → split Q/K/V → bare RMS norm → decay+beta → state update → gated RMS norm
    ///
    /// - Parameters:
    ///   - qkvOut: QKV projection output [linearConvDim].
    ///   - zOut: Z gating projection output [linearTotalValue].
    ///   - betaOut: Beta projection output [linearNumVHeads].
    ///   - alphaOut: Alpha projection output [linearNumVHeads].
    ///   - state: Persistent linear attention state (mutated).
    ///   - config: Model configuration.
    ///   - conv1dW: Conv1d weights (BF16) [convDim × kernelSize].
    ///   - aLog: A_log parameters [numVHeads].
    ///   - dtBias: dt_bias parameters (BF16) [numVHeads].
    ///   - gatedNormW: Gated RMS norm weights (BF16) [valueDim].
    ///   - output: Gated attention output [linearTotalValue], written here.
    public static func forward(
        qkvOut: UnsafePointer<Float>,
        zOut: UnsafePointer<Float>,
        betaOut: UnsafePointer<Float>,
        alphaOut: UnsafePointer<Float>,
        state: inout LinearAttentionState,
        config: ModelConfig,
        conv1dW: UnsafePointer<UInt16>?,
        aLog: UnsafePointer<Float>?,
        dtBias: UnsafePointer<UInt16>?,
        gatedNormW: UnsafePointer<UInt16>?,
        output: UnsafeMutablePointer<Float>
    ) {
        let numKHeads = config.linearNumKHeads
        let numVHeads = config.linearNumVHeads
        let keyDim = config.linearKeyDim
        let valueDim = config.linearValueDim
        let totalKey = config.linearTotalKey    // numKHeads * keyDim
        let totalValue = config.linearTotalValue // numVHeads * valueDim
        let convDim = config.linearConvDim
        let kernelSize = config.convKernelSize
        let vPerK = numVHeads / numKHeads  // value heads per key head

        // ---- Conv1d step ----
        var convOut = [Float](repeating: 0, count: convDim)
        conv1dStep(
            convState: &state.convState,
            newInput: qkvOut,
            weights: conv1dW,
            output: &convOut,
            channels: convDim,
            kernelSize: kernelSize
        )

        // ---- Split Q/K/V from conv output ----
        // Q: [0, totalKey), K: [totalKey, 2*totalKey), V: [2*totalKey, 2*totalKey+totalValue)
        let linQ = convOut.withUnsafeBufferPointer { $0.baseAddress! }
        let linK = linQ + totalKey
        let linV = linK + totalKey

        // ---- Bare RMS norm (per-head, no weights) ----
        // Q and K are normalized per-head with scaling
        var normQ = [Float](repeating: 0, count: totalKey)
        var normK = [Float](repeating: 0, count: totalKey)
        let invScale = 1.0 / sqrtf(Float(keyDim))

        for kh in 0..<numKHeads {
            // Q: normalize and scale by invScale^2
            let qOff = kh * keyDim
            var qSumSq: Float = 0
            for d in 0..<keyDim { qSumSq += linQ[qOff + d] * linQ[qOff + d] }
            let qInvRms = 1.0 / sqrtf(qSumSq / Float(keyDim) + config.rmsNormEps)
            for d in 0..<keyDim {
                normQ[qOff + d] = linQ[qOff + d] * qInvRms * invScale * invScale
            }

            // K: normalize and scale by invScale
            let kOff = kh * keyDim
            var kSumSq: Float = 0
            for d in 0..<keyDim { kSumSq += linK[kOff + d] * linK[kOff + d] }
            let kInvRms = 1.0 / sqrtf(kSumSq / Float(keyDim) + config.rmsNormEps)
            for d in 0..<keyDim {
                normK[kOff + d] = linK[kOff + d] * kInvRms * invScale
            }
        }

        // ---- Gated delta-net recurrence (BLAS-accelerated) ----
        var outValues = [Float](repeating: 0, count: totalValue)

        outValues.withUnsafeMutableBufferPointer { outBuf in
            for kh in 0..<numKHeads {
                let kH = normK.withUnsafeBufferPointer { $0.baseAddress! + kh * keyDim }

                for vi in 0..<vPerK {
                    let vh = kh * vPerK + vi
                    let vH = linV + vh * valueDim

                    // Decay: g = exp(-exp(A_log) * softplus(alpha + dt_bias))
                    var gDecay: Float = 1.0
                    if let aLogPtr = aLog, let dtBiasPtr = dtBias {
                        let a = alphaOut[vh]
                        let dt = bf16ToFloat(dtBiasPtr[vh])
                        let softplus = logf(1.0 + expf(a + dt))
                        gDecay = expf(-expf(aLogPtr[vh]) * softplus)
                    }

                    // Beta gate: sigmoid(beta)
                    let bGate = 1.0 / (1.0 + expf(-betaOut[vh]))

                    // State pointer: S[vh] is [valueDim × keyDim]
                    let stateSize = valueDim * keyDim
                    state.state.withUnsafeMutableBufferPointer { stateBuf in
                        let S = stateBuf.baseAddress! + vh * stateSize

                        // Step 1: Decay state — S *= g (vDSP scalar multiply)
                        var g = gDecay
                        vDSP_vsmul(S, 1, &g, S, 1, vDSP_Length(stateSize))

                        // Step 2: Predict — kvMem = S @ k (row-major matvec)
                        // kvMem[vi] = sum_ki(S[vi,ki] * k[ki])
                        var kvMem = [Float](repeating: 0, count: valueDim)
                        for vi in 0..<valueDim {
                            var dot: Float = 0
                            vDSP_dotpr(S + vi * keyDim, 1, kH, 1, &dot, vDSP_Length(keyDim))
                            kvMem[vi] = dot
                        }

                        // Step 3: Delta = (v - kvMem) * beta, then rank-1 update S += delta @ k^T
                        var delta = [Float](repeating: 0, count: valueDim)
                        for d in 0..<valueDim {
                            delta[d] = (vH[d] - kvMem[d]) * bGate
                        }
                        // S[vi,ki] += delta[vi] * k[ki]
                        for vi in 0..<valueDim {
                            var d = delta[vi]
                            vDSP_vsma(kH, 1, &d, S + vi * keyDim, 1, S + vi * keyDim, 1, vDSP_Length(keyDim))
                        }

                        // Step 4: Output = S @ q (row-major matvec)
                        let qH = normQ.withUnsafeBufferPointer { $0.baseAddress! + kh * keyDim }
                        let oH = outBuf.baseAddress! + vh * valueDim
                        for vi in 0..<valueDim {
                            var dot: Float = 0
                            vDSP_dotpr(S + vi * keyDim, 1, qH, 1, &dot, vDSP_Length(keyDim))
                            oH[vi] = dot
                        }
                    }
                }
            }
        }

        // ---- Gated RMS norm: output = rmsNorm(outValues) * silu(z) * weight ----
        for vh in 0..<numVHeads {
            let oH = outValues.withUnsafeBufferPointer { $0.baseAddress! + vh * valueDim }
            let zH = zOut + vh * valueDim
            let outH = output + vh * valueDim

            if let normW = gatedNormW {
                RMSNorm.gated(input: oH, z: zH, weights: normW,
                              output: outH, dim: valueDim, eps: config.rmsNormEps)
            } else {
                memcpy(outH, oH, valueDim * MemoryLayout<Float>.size)
            }
        }
    }

    // MARK: - Conv1d Step

    /// Performs one step of causal 1D convolution with SiLU activation.
    ///
    /// Maintains a rolling buffer of (kernelSize-1) previous inputs.
    /// Matches `cpu_conv1d_step` in `infer.m:867-890`.
    static func conv1dStep(
        convState: inout [Float],
        newInput: UnsafePointer<Float>,
        weights: UnsafePointer<UInt16>?,
        output: inout [Float],
        channels: Int,
        kernelSize: Int
    ) {
        if let w = weights {
            // Compute convolution: dot product of [state..., newInput] with weight per channel
            for c in 0..<channels {
                var acc: Float = 0
                for k in 0..<(kernelSize - 1) {
                    let weight = bf16ToFloat(w[c * kernelSize + k])
                    acc += convState[k * channels + c] * weight
                }
                let weight = bf16ToFloat(w[c * kernelSize + (kernelSize - 1)])
                acc += newInput[c] * weight
                output[c] = acc
            }
            // SiLU activation
            for c in 0..<channels {
                let x = output[c]
                output[c] = x / (1.0 + expf(-x))
            }
        } else {
            // No weights — pass through input directly
            memcpy(&output, newInput, channels * MemoryLayout<Float>.size)
        }

        // Update conv state: shift left, append new input
        let historyLen = (kernelSize - 1) * channels
        if kernelSize > 2 {
            for i in 0..<(historyLen - channels) {
                convState[i] = convState[i + channels]
            }
        }
        // Write new input into the last history position
        if kernelSize > 1 {
            let lastOffset = (kernelSize - 2) * channels
            for c in 0..<channels {
                convState[lastOffset + c] = newInput[c]
            }
        }
    }
}
