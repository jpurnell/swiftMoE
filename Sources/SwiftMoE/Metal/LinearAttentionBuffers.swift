import Metal

/// GPU state and scratch buffers for GatedDeltaNet linear attention layers.
///
/// 45 of the 60 layers use linear attention. Each layer maintains persistent
/// state (delta recurrence matrix + conv1d history) that carries across tokens
/// and must be zeroed at the start of each new generation.
public struct LinearAttentionBuffers {
    // LIVE: referenced by external test and configuration code
    static let numLinearLayers = 45

    // Persistent state per layer (must be zeroed between generations)

    /// Delta-net recurrence state: [64 heads × 128 × 128] float per layer.
    public let deltaState: [MTLBuffer]
    /// Conv1d history state: [3 × 12288] float per layer.
    public let convState: [MTLBuffer]

    // Scratch buffers (reused across all layers, allocated once)

    /// Query scratch buffer for delta-net projection.
    public let deltaQ: MTLBuffer
    /// Key scratch buffer for delta-net projection.
    public let deltaK: MTLBuffer
    /// Value scratch buffer for delta-net projection.
    public let deltaV: MTLBuffer
    /// Gated decay scratch buffer per value head.
    public let deltaGDecay: MTLBuffer
    /// Beta gate scratch buffer per value head.
    public let deltaBeta: MTLBuffer
    /// Output scratch buffer for delta-net attention output.
    public let deltaOutput: MTLBuffer
    /// Conv1d input scratch buffer.
    public let convInput: MTLBuffer
    /// Conv1d output scratch buffer.
    public let convOutput: MTLBuffer

    init(device: MTLDevice, config: ModelConfig) throws {
        let numVHeads = config.linearNumVHeads
        let valueDim = config.linearValueDim
        let keyDim = config.linearKeyDim
        let convDim = config.linearConvDim
        let numLinear = config.numLinearAttentionLayers
        let convKernel = config.convKernelSize

        let deltaStateSize = numVHeads * valueDim * keyDim * MemoryLayout<Float>.size
        let convStateSize = (convKernel - 1) * convDim * MemoryLayout<Float>.size

        var dStates: [MTLBuffer] = []
        var cStates: [MTLBuffer] = []
        for _ in 0..<numLinear {
            guard let ds = device.makeBuffer(length: deltaStateSize, options: .storageModeShared),
                  let cs = device.makeBuffer(length: convStateSize, options: .storageModeShared) else {
                throw FlashMoEError.bufferAllocationFailed(size: deltaStateSize)
            }
            // Zero-initialize persistent state
            memset(ds.contents(), 0, deltaStateSize)
            memset(cs.contents(), 0, convStateSize)
            dStates.append(ds)
            cStates.append(cs)
        }
        self.deltaState = dStates
        self.convState = cStates

        // Scratch buffers
        func scratch(_ count: Int) throws -> MTLBuffer {
            let size = count * MemoryLayout<Float>.size
            guard let buf = device.makeBuffer(length: size, options: .storageModeShared) else {
                throw FlashMoEError.bufferAllocationFailed(size: size)
            }
            return buf
        }

        let totalKey = config.linearTotalKey   // numKHeads * keyDim
        let totalValue = config.linearTotalValue // numVHeads * valueDim

        self.deltaQ = try scratch(totalKey)
        self.deltaK = try scratch(totalKey)
        self.deltaV = try scratch(totalValue)
        self.deltaGDecay = try scratch(numVHeads)
        self.deltaBeta = try scratch(numVHeads)
        self.deltaOutput = try scratch(totalValue)
        self.convInput = try scratch(convDim)
        self.convOutput = try scratch(convDim)
    }

    /// Zeros all persistent state buffers. Call at the start of each new generation.
    public func resetState() {
        for i in 0..<deltaState.count {
            memset(deltaState[i].contents(), 0, deltaState[i].length)
            memset(convState[i].contents(), 0, convState[i].length)
        }
    }
}
