import Metal

/// Double-buffered expert weight slots for the CMD3 pipeline.
///
/// Each of the K=8 slots has two 2MB-aligned data buffers:
/// - **Set A**: GPU reads from these during CMD3 expert compute
/// - **Set B**: CPU preads into these while GPU processes set A
///
/// Intermediate buffers (gate, up, activation, output) only need one set
/// because the GPU uses them sequentially after the pread completes.
///
/// ## Quantization-Aware Sizing
/// The `use2Bit` parameter controls data buffer allocation:
/// - 4-bit: 8MB per slot (7,077,888 rounded to 2MB) × 8 × 2 = 128MB
/// - 2-bit: 4MB per slot (3,932,160 rounded to 2MB) × 8 × 2 = 64MB
///
/// This 64MB difference matters on constrained systems because Metal shared
/// memory competes with the OS page cache for DRAM (see "Trust the OS", Section 5.3).
public struct ExpertBuffers {
    /// Maximum number of expert slots (double-buffered).
    public static let maxK = 8

    /// Set A: GPU reads from these during CMD3.
    public let dataA: [AlignedBuffer]
    /// Set B: CPU preads into these while GPU processes set A.
    public let dataB: [AlignedBuffer]

    /// Per-expert intermediate buffers (gate projection output).
    public let gate: [MTLBuffer]
    /// Per-expert intermediate buffers (up projection output).
    public let up: [MTLBuffer]
    /// Per-expert intermediate buffers (SwiGLU output).
    public let activation: [MTLBuffer]
    /// Per-expert intermediate buffers (down projection output).
    public let output: [MTLBuffer]
    /// Shared input vector (read-only during expert dispatch).
    public let input: MTLBuffer

    // Shared expert (always-active, one per layer)
    /// Shared expert gate projection output.
    public let sharedGate: MTLBuffer
    /// Shared expert up projection output.
    public let sharedUp: MTLBuffer
    /// Shared expert SwiGLU output.
    public let sharedActivation: MTLBuffer
    /// Shared expert down projection output.
    public let sharedOutput: MTLBuffer

    /// Allocates all expert buffers.
    ///
    /// - Parameters:
    ///   - device: Metal device for buffer creation.
    ///   - config: Model configuration providing expert sizes and alignment.
    ///   - use2Bit: If true, sizes data buffers for 2-bit experts (saves ~64MB).
    public init(device: MTLDevice, config: ModelConfig, use2Bit: Bool) throws {
        let expertSize = use2Bit ? config.expertSize2Bit : config.expertSize4Bit
        let alignment = ModelConfig.dmaAlignment  // 2MB

        // Round up to alignment boundary
        let allocSize = (expertSize + alignment - 1) & ~(alignment - 1)

        var dataASlots: [AlignedBuffer] = []
        var dataBSlots: [AlignedBuffer] = []
        var gateSlots: [MTLBuffer] = []
        var upSlots: [MTLBuffer] = []
        var actSlots: [MTLBuffer] = []
        var outSlots: [MTLBuffer] = []

        let intermediateBytes = config.moeIntermediate * MemoryLayout<Float>.size
        let hiddenBytes = config.hiddenDim * MemoryLayout<Float>.size

        for _ in 0..<Self.maxK {
            dataASlots.append(try AlignedBuffer(device: device, size: allocSize, alignment: alignment))
            dataBSlots.append(try AlignedBuffer(device: device, size: allocSize, alignment: alignment))

            guard let g = device.makeBuffer(length: intermediateBytes, options: .storageModeShared),
                  let u = device.makeBuffer(length: intermediateBytes, options: .storageModeShared),
                  let a = device.makeBuffer(length: intermediateBytes, options: .storageModeShared),
                  let o = device.makeBuffer(length: hiddenBytes, options: .storageModeShared) else {
                throw FlashMoEError.bufferAllocationFailed(size: intermediateBytes)
            }
            gateSlots.append(g)
            upSlots.append(u)
            actSlots.append(a)
            outSlots.append(o)
        }

        self.dataA = dataASlots
        self.dataB = dataBSlots
        self.gate = gateSlots
        self.up = upSlots
        self.activation = actSlots
        self.output = outSlots

        guard let inputBuf = device.makeBuffer(length: hiddenBytes, options: .storageModeShared) else {
            throw FlashMoEError.bufferAllocationFailed(size: hiddenBytes)
        }
        self.input = inputBuf

        // Shared expert buffers
        let sharedBytes = config.sharedIntermediate * MemoryLayout<Float>.size
        guard let sg = device.makeBuffer(length: sharedBytes, options: .storageModeShared),
              let su = device.makeBuffer(length: sharedBytes, options: .storageModeShared),
              let sa = device.makeBuffer(length: sharedBytes, options: .storageModeShared),
              let so = device.makeBuffer(length: hiddenBytes, options: .storageModeShared) else {
            throw FlashMoEError.bufferAllocationFailed(size: sharedBytes)
        }
        self.sharedGate = sg
        self.sharedUp = su
        self.sharedActivation = sa
        self.sharedOutput = so
    }
}
