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

        // FP-safety: keyDim must be positive for the division below
        guard keyDim > 0 else { return }

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

        // ---- Split Q/K/V from conv output (safe array slicing, no pointer escape) ----
        let linQArr = Array(convOut[0..<totalKey])
        let linKArr = Array(convOut[totalKey..<(2 * totalKey)])
        let linVArr = Array(convOut[(2 * totalKey)..<(2 * totalKey + totalValue)])

        // ---- Bare RMS norm (per-head, no weights) ----
        // Q and K are normalized per-head with scaling
        var normQ = [Float](repeating: 0, count: totalKey)
        var normK = [Float](repeating: 0, count: totalKey)
        let sqrtKeyDim = sqrtf(Float(keyDim))
        let invScale = sqrtKeyDim > 0 ? 1.0 / sqrtKeyDim : 0.0

        for kh in 0..<numKHeads {
            // Q: normalize and scale by invScale^2
            let qOff = kh * keyDim
            var qSumSq: Float = 0
            for d in 0..<keyDim { qSumSq += linQArr[qOff + d] * linQArr[qOff + d] }
            let qRms = sqrtf(qSumSq / Float(keyDim) + config.rmsNormEps)
            let qInvRms = qRms > 0 ? 1.0 / qRms : 0.0
            for d in 0..<keyDim {
                normQ[qOff + d] = linQArr[qOff + d] * qInvRms * invScale * invScale
            }

            // K: normalize and scale by invScale
            let kOff = kh * keyDim
            var kSumSq: Float = 0
            for d in 0..<keyDim { kSumSq += linKArr[kOff + d] * linKArr[kOff + d] }
            let kRms = sqrtf(kSumSq / Float(keyDim) + config.rmsNormEps)
            let kInvRms = kRms > 0 ? 1.0 / kRms : 0.0
            for d in 0..<keyDim {
                normK[kOff + d] = linKArr[kOff + d] * kInvRms * invScale
            }
        }

        // ---- Gated delta-net recurrence (BLAS-accelerated) ----
        var outValues = [Float](repeating: 0, count: totalValue)

        // Nest all withUnsafe closures so pointers stay in scope
        normK.withUnsafeBufferPointer { normKBuf in
            normQ.withUnsafeBufferPointer { normQBuf in
                outValues.withUnsafeMutableBufferPointer { outBuf in
                    guard let normKBase = normKBuf.baseAddress,
                          let normQBase = normQBuf.baseAddress,
                          let outBase = outBuf.baseAddress else { return }

                    for kh in 0..<numKHeads {
                        let kH = normKBase + kh * keyDim

                        for vi in 0..<vPerK {
                            let vh = kh * vPerK + vi

                            // Decay: g = exp(-exp(A_log) * softplus(alpha + dt_bias))
                            var gDecay: Float = 1.0
                            if let aLogPtr = aLog, let dtBiasPtr = dtBias {
                                let a = alphaOut[vh]
                                let dt = bf16ToFloat(dtBiasPtr[vh])
                                let softplus = logf(1.0 + expf(a + dt))
                                gDecay = expf(-expf(aLogPtr[vh]) * softplus)
                            }

                            // Beta gate: sigmoid(beta) — denominator is always >= 1.0
                            let bDenom = 1.0 + expf(-betaOut[vh])
                            let bGate = bDenom > 0 ? 1.0 / bDenom : 0.5

                            let stateSize = valueDim * keyDim
                            state.state.withUnsafeMutableBufferPointer { stateBuf in
                                guard let sBase = stateBuf.baseAddress else { return }
                                let S = sBase + vh * stateSize

                                var gDecayVar = gDecay
                                vDSP_vsmul(S, 1, &gDecayVar, S, 1, vDSP_Length(stateSize))

                                var kvMem = [Float](repeating: 0, count: valueDim)
                                for vi2 in 0..<valueDim {
                                    var dot: Float = 0
                                    vDSP_dotpr(S + vi2 * keyDim, 1, kH, 1, &dot, vDSP_Length(keyDim))
                                    kvMem[vi2] = dot
                                }

                                var delta = [Float](repeating: 0, count: valueDim)
                                linVArr.withUnsafeBufferPointer { vBuf in
                                    guard let vBase = vBuf.baseAddress else { return }
                                    let vH = vBase + vh * valueDim
                                    for d in 0..<valueDim {
                                        delta[d] = (vH[d] - kvMem[d]) * bGate
                                    }
                                }
                                for vi2 in 0..<valueDim {
                                    var d = delta[vi2]
                                    vDSP_vsma(kH, 1, &d, S + vi2 * keyDim, 1, S + vi2 * keyDim, 1, vDSP_Length(keyDim))
                                }

                                let qH = normQBase + kh * keyDim
                                let oH = outBase + vh * valueDim
                                for vi2 in 0..<valueDim {
                                    var dot: Float = 0
                                    vDSP_dotpr(S + vi2 * keyDim, 1, qH, 1, &dot, vDSP_Length(keyDim))
                                    oH[vi2] = dot
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---- Gated RMS norm: output = rmsNorm(outValues) * silu(z) * weight ----
        outValues.withUnsafeBufferPointer { outValBuf in
            guard let outValBase = outValBuf.baseAddress else { return }
            for vh in 0..<numVHeads {
                let oH = outValBase + vh * valueDim
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
