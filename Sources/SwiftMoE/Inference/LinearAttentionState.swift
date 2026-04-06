/// CPU-side persistent state for GatedDeltaNet linear attention.
///
/// Each of the 45 linear attention layers maintains:
/// - A delta-net recurrence state matrix [numVHeads × valueDim × keyDim]
/// - A conv1d history buffer [(kernelSize-1) × convDim]
///
/// These carry across tokens within a generation and must be zeroed
/// between generations (new conversations).
public struct LinearAttentionState {
    /// Delta-net recurrence state: [numVHeads × valueDim × keyDim] float.
    public var state: [Float]

    /// Conv1d history: [(kernelSize-1) × convDim] float.
    public var convState: [Float]

    private let stateSize: Int
    private let convSize: Int

    /// Creates a new linear attention state for one layer.
    public init(config: ModelConfig) {
        let numVHeads = config.linearNumVHeads
        let valueDim = config.linearValueDim
        let keyDim = config.linearKeyDim
        let convKernelSize = config.convKernelSize
        let convDim = config.linearConvDim
        self.stateSize = numVHeads * valueDim * keyDim
        self.convSize = (convKernelSize - 1) * convDim
        self.state = [Float](repeating: 0, count: stateSize)
        self.convState = [Float](repeating: 0, count: convSize)
    }

    /// Resets all state to zero (call between generations).
    public mutating func reset() {
        state = [Float](repeating: 0, count: stateSize)
        convState = [Float](repeating: 0, count: convSize)
    }
}
