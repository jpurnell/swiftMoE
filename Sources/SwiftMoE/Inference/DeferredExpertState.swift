import Metal

/// State for the deferred GPU expert computation pipeline.
///
/// GPU expert computation (CMD3) is submitted asynchronously — the CPU moves on
/// to the next layer's attention while experts run on the GPU. This struct holds
/// the state needed to finalize the deferred computation when the next layer
/// begins (or at the end of the 60-layer pass).
///
/// ## Pipeline Flow
/// ```
/// Layer N:  ... → CMD3 [commit, NO wait] → return
/// Layer N+1: [wait CMD3] → finalize → CMD1 → ...
/// ```
///
/// ## GPU Combined Mode
/// When `isGPUCombined` is true, CMD3 also computed the combine+residual+norm
/// on GPU, placing the result directly in `buf_input`. The next layer skips
/// the CPU finalize+input_norm entirely and submits CMD1 immediately.
public struct DeferredExpertState {
    /// Whether there's a pending GPU expert computation.
    public private(set) var isActive: Bool = false

    /// Whether CMD3 includes GPU-side combine+residual+norm.
    public private(set) var isGPUCombined: Bool = false

    /// The async command buffer (committed but not waited).
    public var commandBuffer: MTLCommandBuffer?

    /// Routing weights for weighted expert accumulation.
    public private(set) var expertWeights: [Float] = []

    /// Which expert slots loaded successfully.
    public private(set) var valid: [Bool] = []

    /// Number of active experts for this deferred computation.
    public private(set) var actualK: Int = 0

    /// Shared expert sigmoid gate score.
    public private(set) var sharedGateScore: Float = 0

    /// Which layer produced this deferred state.
    public private(set) var layerIndex: Int = 0

    /// Saved h_mid for CPU-side combine (when not GPU combined).
    public var hMid: [Float] = []

    /// Pointer to hidden state (for writing the final combined result).
    public var hiddenPointer: UnsafeMutablePointer<Float>?

    /// Creates a new inactive DeferredExpertState.
    public init() {}

    /// Activates the deferred state with the current layer's expert results.
    public mutating func activate(
        expertWeights: [Float],
        valid: [Bool],
        sharedGateScore: Float,
        layerIndex: Int,
        gpuCombined: Bool
    ) {
        self.isActive = true
        self.isGPUCombined = gpuCombined
        self.expertWeights = expertWeights
        self.valid = valid
        self.sharedGateScore = sharedGateScore
        self.layerIndex = layerIndex
        self.actualK = expertWeights.count
    }

    /// Deactivates the deferred state after finalization.
    public mutating func deactivate() {
        isActive = false
        isGPUCombined = false
        commandBuffer = nil
    }

    /// Waits for the deferred GPU command buffer to complete.
    public func waitForGPU() {
        guard isActive, let cmd = commandBuffer else { return }
        cmd.waitUntilCompleted()
    }
}
