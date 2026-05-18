import Foundation

/// CPU-side full scaled dot-product attention with Grouped Query Attention (GQA).
///
/// Qwen3.5 uses 32 query heads with 2 KV heads (16 query heads per KV head).
/// Each head has dimension 256. Full attention is used by 15 of the 60 layers
/// (every 4th layer starting at layer 3).
///
/// ## Pipeline Position
/// Called during Phase 2 of `fused_layer_forward`, after CMD1 has computed
/// Q/K/V projections and before CMD2 encodes o_proj + routing.
///
/// Matches `infer.m:4479-4580` (the CPU full attention compute path).
public enum FullAttention {

    /// Computes full attention for a single token.
    ///
    /// Flow: Q/K per-head norm → RoPE → KV cache update → scaled dot-product → sigmoid gate
    ///
    /// - Parameters:
    ///   - qProjOut: Q projection output [numHeads * headDim * 2] (includes sigmoid gate).
    ///   - kOut: K projection output [numKVHeads * headDim].
    ///   - vOut: V projection output [numKVHeads * headDim].
    ///   - kvCache: KV cache for this layer (mutated to append new K/V).
    ///   - position: Token position for RoPE.
    ///   - config: Model configuration.
    ///   - qNormW: Per-head Q RMS norm weights (BF16), or nil to skip.
    ///   - kNormW: Per-head K RMS norm weights (BF16), or nil to skip.
    ///   - output: Attention output [numHeads * headDim], written here.
    public static func forward(
        qProjOut: UnsafeMutablePointer<Float>,
        kOut: UnsafeMutablePointer<Float>,
        vOut: UnsafeMutablePointer<Float>,
        kvCache: inout KVCache,
        position: Int,
        config: ModelConfig,
        qNormW: UnsafePointer<UInt16>?,
        kNormW: UnsafePointer<UInt16>?,
        output: UnsafeMutablePointer<Float>
    ) {
        let numHeads = config.numAttentionHeads
        let numKVHeads = config.numKVHeads
        let headDim = config.headDim
        guard headDim > 0, numKVHeads > 0 else { return }
        let headsPerKV = numHeads / numKVHeads
        guard headsPerKV > 0 else { return }
        let kvDim = numKVHeads * headDim

        // Split Q projection into query and sigmoid gate
        // q_proj output is [numHeads * headDim * 2]: interleaved (q_head, gate_head) pairs
        let qDim = numHeads * headDim
        var q = [Float](repeating: 0, count: qDim)
        var qGate = [Float](repeating: 0, count: qDim)

        for h in 0..<numHeads {
            let src = qProjOut + h * (2 * headDim)
            for d in 0..<headDim {
                q[h * headDim + d] = src[d]
                qGate[h * headDim + d] = src[headDim + d]
            }
        }

        // Per-head Q RMS norm
        if let normW = qNormW {
            for h in 0..<numHeads {
                let offset = h * headDim
                var sumSq: Float = 0
                for d in 0..<headDim { sumSq += q[offset + d] * q[offset + d] }
                let rmsVal = sqrtf(sumSq / Float(headDim) + config.rmsNormEps)
                let invRms = rmsVal > 0 ? 1.0 / rmsVal : 0.0
                for d in 0..<headDim {
                    q[offset + d] = q[offset + d] * invRms * bf16ToFloat(normW[d])
                }
            }
        }

        // Per-head K RMS norm
        if let normW = kNormW {
            for h in 0..<numKVHeads {
                let offset = h * headDim
                var sumSq: Float = 0
                for d in 0..<headDim { sumSq += kOut[offset + d] * kOut[offset + d] }
                let rmsVal = sqrtf(sumSq / Float(headDim) + config.rmsNormEps)
                let invRms = rmsVal > 0 ? 1.0 / rmsVal : 0.0
                for d in 0..<headDim {
                    kOut[offset + d] = kOut[offset + d] * invRms * bf16ToFloat(normW[d])
                }
            }
        }

        // RoPE
        q.withUnsafeMutableBufferPointer { qBuf in
            guard let qBase = qBuf.baseAddress else { return }
            RoPE.apply(
                q: qBase, k: kOut,
                position: position,
                numHeads: numHeads, numKVHeads: numKVHeads,
                headDim: headDim, rotaryDim: config.rotaryDim,
                theta: config.ropeTheta
            )
        }

        // Update KV cache
        kvCache.append(kPtr: UnsafePointer(kOut), vPtr: UnsafePointer(vOut))
        let seqLen = kvCache.length
        let sqrtHeadDim = sqrtf(Float(headDim))
        let scale = sqrtHeadDim > 0 ? 1.0 / sqrtHeadDim : 0.0

        // Scaled dot-product attention with GQA
        memset(output, 0, qDim * MemoryLayout<Float>.size)

        kvCache.withKCache { kCache in
            kvCache.withVCache { vCache in
                for h in 0..<numHeads {
                    let kvH = h / headsPerKV

                    // Compute attention scores
                    var scores = [Float](repeating: 0, count: seqLen)
                    q.withUnsafeBufferPointer { qBuf in
                        guard let qBase = qBuf.baseAddress else { return }
                        let qH = qBase + h * headDim
                        for p in 0..<seqLen {
                            let kP = kCache + p * kvDim + kvH * headDim
                            var dot: Float = 0
                            for d in 0..<headDim { dot += qH[d] * kP[d] }
                            scores[p] = dot * scale
                        }
                    }

                    // Softmax
                    Softmax.apply(&scores, count: seqLen)

                    // Weighted sum of values
                    let oH = output + h * headDim
                    for p in 0..<seqLen {
                        let vP = vCache + p * kvDim + kvH * headDim
                        for d in 0..<headDim {
                            oH[d] += scores[p] * vP[d]
                        }
                    }
                }
            }
        }

        // Apply sigmoid gate: output *= sigmoid(qGate)
        for i in 0..<qDim {
            let denom = 1.0 + expf(-qGate[i])
            let g = denom > 0 ? 1.0 / denom : 0.5
            output[i] *= g
        }
    }
}
