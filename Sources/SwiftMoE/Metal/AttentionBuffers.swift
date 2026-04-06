import Metal

/// Pre-allocated buffers for full attention (RoPE) layers.
///
/// Only 15 of the 60 layers use full attention (every 4th layer starting at layer 3).
/// Each needs a KV cache that grows with sequence length.
public struct AttentionBuffers {
    static let numFullAttentionLayers = 15
    static let gpuKVSeqLength = 8192  // Pre-allocated GPU KV buffer length

    /// K cache per full-attention layer.
    public let kvK: [MTLBuffer]
    /// V cache per full-attention layer.
    public let kvV: [MTLBuffer]

    /// Query projection output for all heads.
    public let query: MTLBuffer
    /// Attention scores (all heads × sequence length).
    public let scores: MTLBuffer
    /// Attention output (all heads × head_dim).
    public let output: MTLBuffer
    /// Sigmoid gate for gated attention.
    public let gate: MTLBuffer

    init(device: MTLDevice, config: ModelConfig) throws {
        let kvDim = config.numKVHeads * config.headDim  // 512
        let kvCacheSize = Self.gpuKVSeqLength * kvDim * MemoryLayout<Float>.size

        var kCaches: [MTLBuffer] = []
        var vCaches: [MTLBuffer] = []
        for _ in 0..<Self.numFullAttentionLayers {
            guard let k = device.makeBuffer(length: kvCacheSize, options: .storageModeShared),
                  let v = device.makeBuffer(length: kvCacheSize, options: .storageModeShared) else {
                throw FlashMoEError.bufferAllocationFailed(size: kvCacheSize)
            }
            kCaches.append(k)
            vCaches.append(v)
        }
        self.kvK = kCaches
        self.kvV = vCaches

        let headBytes = config.numAttentionHeads * config.headDim * MemoryLayout<Float>.size
        let scoreBytes = config.numAttentionHeads * Self.gpuKVSeqLength * MemoryLayout<Float>.size

        guard let q = device.makeBuffer(length: headBytes, options: .storageModeShared),
              let s = device.makeBuffer(length: scoreBytes, options: .storageModeShared),
              let o = device.makeBuffer(length: headBytes, options: .storageModeShared),
              let g = device.makeBuffer(length: headBytes, options: .storageModeShared) else {
            throw FlashMoEError.bufferAllocationFailed(size: headBytes)
        }
        self.query = q
        self.scores = s
        self.output = o
        self.gate = g
    }
}
