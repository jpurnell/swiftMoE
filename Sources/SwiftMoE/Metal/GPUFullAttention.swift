import Metal

/// Encodes GPU full attention (scores + softmax + values + sigmoid gate) into CMD2.
///
/// Prepends 4 compute encoders to the CMD2 command buffer for full-attention layers
/// when sequence length ≥ 32. Below that threshold, CPU attention is faster because
/// GPU command encoder overhead dominates at short sequences.
///
/// After this runs, `buf_attn_out` contains the gated attention output. CMD2's o_proj
/// reads from `buf_attn_out` instead of `batchSlots[6]`.
///
/// Matches `infer.m:4819-4891` (GPU attention dispatches fused into CMD2).
public enum GPUFullAttention {

    /// Minimum sequence length for GPU attention to be worthwhile.
    /// Below this, CPU dot-product attention is faster due to encoder overhead.
    public static let minSeqLength = 32

    /// Whether GPU full attention should be used for this layer.
    public static func shouldUseGPU(
        context: MetalContext,
        config: ModelConfig,
        kvCacheLength: Int,
        layerIndex: Int
    ) -> Bool {
        let faIdx = config.fullAttentionIndex(layer: layerIndex)
        return kvCacheLength >= minSeqLength
            && kvCacheLength < AttentionBuffers.gpuKVSeqLength
            && faIdx >= 0
            && faIdx < config.numFullAttentionLayers
    }

    /// Copies Q and gate vectors to GPU buffers for attention dispatch.
    ///
    /// Must be called after CPU-side Q/K norm + RoPE + gate splitting,
    /// before the GPU attention encoders run.
    public static func copyToGPU(
        context: MetalContext,
        config: ModelConfig,
        q: UnsafePointer<Float>,
        qGate: UnsafePointer<Float>,
        k: UnsafePointer<Float>,
        v: UnsafePointer<Float>,
        faIndex: Int,
        cachePosition: Int
    ) {
        let qDim = config.numAttentionHeads * config.headDim
        let kvDim = config.kvDim

        memcpy(context.attention.query.contents(), q, qDim * MemoryLayout<Float>.size)
        memcpy(context.attention.gate.contents(), qGate, qDim * MemoryLayout<Float>.size)

        // Mirror K/V to GPU KV cache at the current position
        let kDst = context.attention.kvK[faIndex].contents()
            .assumingMemoryBound(to: Float.self) + cachePosition * kvDim
        let vDst = context.attention.kvV[faIndex].contents()
            .assumingMemoryBound(to: Float.self) + cachePosition * kvDim
        memcpy(kDst, k, kvDim * MemoryLayout<Float>.size)
        memcpy(vDst, v, kvDim * MemoryLayout<Float>.size)
    }

    /// Encodes 4 GPU attention dispatches into a command buffer.
    ///
    /// 1. `attn_scores_batched` — Q @ K^T for all heads
    /// 2. `attn_softmax_batched` — per-head softmax over scores
    /// 3. `attn_values_batched` — scores @ V weighted sum
    /// 4. `sigmoid_gate` — multiply output by sigmoid(gate)
    ///
    /// Result is in `context.attention.output` (`buf_attn_out`).
    public static func encode(
        context: MetalContext,
        config: ModelConfig,
        commandBuffer: MTLCommandBuffer,
        faIndex: Int,
        seqLen: Int
    ) {
        let numHeads = config.numAttentionHeads
        let headDim = config.headDim
        let kvDim = config.kvDim
        let headsPerKV = numHeads / config.numKVHeads

        var hd = UInt32(headDim)
        var kvd = UInt32(kvDim)
        var sl = UInt32(seqLen)
        var seqStride = UInt32(AttentionBuffers.gpuKVSeqLength)
        var scale = 1.0 / sqrtf(Float(headDim))
        var hpkv = UInt32(headsPerKV)

        // ---- Enc A1: attn_scores_batched ----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(context.shaders.attnScores)
            enc.setBuffer(context.attention.query, offset: 0, index: 0)
            enc.setBuffer(context.attention.kvK[faIndex], offset: 0, index: 1)
            enc.setBuffer(context.attention.scores, offset: 0, index: 2)
            enc.setBytes(&hd, length: 4, index: 3)
            enc.setBytes(&kvd, length: 4, index: 4)
            enc.setBytes(&sl, length: 4, index: 5)
            enc.setBytes(&seqStride, length: 4, index: 6)
            enc.setBytes(&scale, length: 4, index: 7)
            enc.setBytes(&hpkv, length: 4, index: 8)
            enc.setBytes(&sl, length: 4, index: 9)
            let totalTGs = Int(sl) * numHeads
            enc.dispatchThreadgroups(
                MTLSize(width: totalTGs, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // ---- Enc A2: attn_softmax_batched ----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(context.shaders.attnSoftmax)
            enc.setBuffer(context.attention.scores, offset: 0, index: 0)
            enc.setBytes(&sl, length: 4, index: 1)
            enc.setBytes(&seqStride, length: 4, index: 2)
            enc.dispatchThreadgroups(
                MTLSize(width: numHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // ---- Enc A3: attn_values_batched ----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(context.shaders.attnValues)
            enc.setBuffer(context.attention.scores, offset: 0, index: 0)
            enc.setBuffer(context.attention.kvV[faIndex], offset: 0, index: 1)
            enc.setBuffer(context.attention.output, offset: 0, index: 2)
            enc.setBytes(&hd, length: 4, index: 3)
            enc.setBytes(&kvd, length: 4, index: 4)
            enc.setBytes(&sl, length: 4, index: 5)
            enc.setBytes(&seqStride, length: 4, index: 6)
            enc.setBytes(&hpkv, length: 4, index: 7)
            let totalThreads = headDim * numHeads
            let tgs = (totalThreads + 255) / 256
            enc.dispatchThreadgroups(
                MTLSize(width: tgs, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // ---- Enc A4: sigmoid_gate ----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            var qdim = UInt32(numHeads * headDim)
            enc.setComputePipelineState(context.shaders.sigmoidGate)
            enc.setBuffer(context.attention.output, offset: 0, index: 0)
            enc.setBuffer(context.attention.gate, offset: 0, index: 1)
            enc.setBytes(&qdim, length: 4, index: 2)
            let tgs = Int((qdim + 255) / 256)
            enc.dispatchThreadgroups(
                MTLSize(width: tgs, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
    }
}
