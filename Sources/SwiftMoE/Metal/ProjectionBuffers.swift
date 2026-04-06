import Metal

/// Pre-allocated buffers for attention projection matrix-vector multiplies.
///
/// These are the most frequently accessed buffers — used every layer, every token.
/// `input` holds the current hidden state; `output` holds projection results;
/// `batchSlots` hold batched matmul outputs for multi-projection CMD1 encoding.
public struct ProjectionBuffers {
    static let maxBatchSlots = 8

    /// Input vector buffer (sized for the largest projection input).
    public let input: MTLBuffer

    /// Output vector buffer (sized for lm_head = VOCAB_SIZE floats).
    public let output: MTLBuffer

    /// Batched matmul output slots (sized for the largest projection output).
    public let batchSlots: [MTLBuffer]

    /// Allocates projection buffers matching the sizes from `metal_setup()`.
    init(device: MTLDevice, config: ModelConfig) throws {
        // Largest input: linear_attn out_proj or full-attn o_proj = 8192 floats
        let maxIn = max(
            config.linearNumVHeads * config.linearValueDim,
            config.numAttentionHeads * config.headDim
        ) * MemoryLayout<Float>.size

        // Largest output: lm_head = VOCAB_SIZE floats
        let maxOut = config.vocabSize * MemoryLayout<Float>.size

        guard let inputBuf = device.makeBuffer(length: maxIn, options: .storageModeShared),
              let outputBuf = device.makeBuffer(length: maxOut, options: .storageModeShared) else {
            throw FlashMoEError.bufferAllocationFailed(size: maxIn + maxOut)
        }

        self.input = inputBuf
        self.output = outputBuf

        // Batch slots: each sized for the largest projection output
        // q_proj = 16384 floats, qkv_proj = 12288, z_proj = 8192
        let linearConvDim = config.linearNumKHeads * config.linearKeyDim * 2
            + config.linearNumVHeads * config.linearValueDim  // 12288
        let slotSize = max(
            config.numAttentionHeads * config.headDim * 2,  // 16384
            linearConvDim  // 12288
        ) * MemoryLayout<Float>.size

        var slots: [MTLBuffer] = []
        for _ in 0..<Self.maxBatchSlots {
            guard let slot = device.makeBuffer(length: slotSize, options: .storageModeShared) else {
                throw FlashMoEError.bufferAllocationFailed(size: slotSize)
            }
            slots.append(slot)
        }
        self.batchSlots = slots
    }
}
