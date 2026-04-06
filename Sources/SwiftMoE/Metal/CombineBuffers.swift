import Metal

/// Buffers for CMD3's GPU-side combine: weighted expert sum + residual + RMS norm.
///
/// The combine step runs on GPU at the end of CMD3, producing the normalized
/// hidden state for the next layer's CMD1. This eliminates a CPU round-trip
/// (0.83ms/layer savings in the original optimization).
public struct CombineBuffers {

    /// Residual connection buffer [HIDDEN_DIM floats].
    public let residual: MTLBuffer
    /// Post-attention hidden state [HIDDEN_DIM floats].
    public let hMid: MTLBuffer
    /// RMS norm reduction (sum of squares) [1 float].
    public let sumSq: MTLBuffer

    /// MoE combine output (weighted expert sum + residual) [HIDDEN_DIM floats].
    public let moeHidden: MTLBuffer
    /// Expert weights + shared gate score [10 floats: weights[8] + shared_gate + pad].
    public let combineParams: MTLBuffer
    /// CMD3-specific RMS norm reduction [1 float].
    public let cmd3SumSq: MTLBuffer

    init(device: MTLDevice, config: ModelConfig) throws {
        let hiddenBytes = config.hiddenDim * MemoryLayout<Float>.size
        let floatSize = MemoryLayout<Float>.size

        guard let r = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let h = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let s = device.makeBuffer(length: floatSize, options: .storageModeShared),
              let m = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let p = device.makeBuffer(length: 10 * floatSize, options: .storageModeShared),
              let c = device.makeBuffer(length: floatSize, options: .storageModeShared) else {
            throw FlashMoEError.bufferAllocationFailed(size: hiddenBytes)
        }

        self.residual = r
        self.hMid = h
        self.sumSq = s
        self.moeHidden = m
        self.combineParams = p
        self.cmd3SumSq = c
    }
}
