/// Runtime configuration for a Mixture-of-Experts transformer model.
///
/// Describes the architecture of any MoE model: dimensions, layer counts,
/// expert configuration, attention layout, and quantization format.
/// The Flash-MoE inference engine uses this to size buffers, dispatch GPU
/// kernels, and orchestrate the pipeline.
///
/// ## Presets
/// - ``qwen397B``: Qwen3.5-397B-A17B (the original Flash-MoE target)
/// - ``tiny``: Minimal model for integration testing (hidden=64, 2 layers)
///
/// ## Hardware Constants
/// ``dmaAlignment`` is a hardware constant (Apple Silicon DMA) and lives
/// outside the model config as a static property.
public struct ModelConfig: Sendable {

    // MARK: - Core Dimensions

    public let hiddenDim: Int
    public let numLayers: Int
    public let numAttentionHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let vocabSize: Int
    public let rmsNormEps: Float

    // MARK: - Expert Configuration

    public let numExperts: Int
    public let numExpertsPerToken: Int
    public let moeIntermediate: Int
    public let sharedIntermediate: Int

    // MARK: - Expert Binary Layout

    /// Total bytes per expert at 4-bit quantization.
    public let expertSize4Bit: Int
    /// Total bytes per expert at 2-bit requantization.
    public let expertSize2Bit: Int

    // MARK: - 2-Bit Expert Offsets

    public let gateWeightsOffset2Bit: Int
    public let gateScalesOffset2Bit: Int
    public let gateBiasesOffset2Bit: Int
    public let upWeightsOffset2Bit: Int
    public let upScalesOffset2Bit: Int
    public let upBiasesOffset2Bit: Int
    public let downWeightsOffset2Bit: Int
    public let downScalesOffset2Bit: Int
    public let downBiasesOffset2Bit: Int

    // MARK: - Quantization

    public let groupSize: Int
    public let bits: Int

    // MARK: - Attention

    public let fullAttentionInterval: Int
    public let ropeTheta: Float
    public let partialRotary: Float

    // MARK: - Linear Attention (GatedDeltaNet)

    public let linearNumVHeads: Int
    public let linearNumKHeads: Int
    public let linearKeyDim: Int
    public let linearValueDim: Int
    public let convKernelSize: Int

    // MARK: - Special Tokens

    public let eosToken1: Int
    public let eosToken2: Int
    public let thinkStartToken: Int
    public let thinkEndToken: Int

    // MARK: - Hardware Constants (not model-specific)

    /// DMA-optimal alignment for pread destination buffers.
    /// 2MB alignment yields 3.6x faster reads than 16KB (see paper Section 3.6).
    public static let dmaAlignment = 2 * 1024 * 1024

    // MARK: - Computed Properties

    /// Number of full attention layers.
    public var numFullAttentionLayers: Int {
        (0..<numLayers).filter { ($0 + 1) % fullAttentionInterval == 0 }.count
    }

    /// Number of linear attention layers.
    public var numLinearAttentionLayers: Int {
        numLayers - numFullAttentionLayers
    }

    /// Key-value dimension (numKVHeads × headDim).
    public var kvDim: Int { numKVHeads * headDim }

    /// Rotary embedding dimension.
    public var rotaryDim: Int { Int(Float(headDim) * partialRotary) }

    /// Linear attention conv dimension.
    public var linearConvDim: Int {
        linearNumKHeads * linearKeyDim * 2 + linearNumVHeads * linearValueDim
    }

    /// Linear attention total value dimension.
    public var linearTotalValue: Int { linearNumVHeads * linearValueDim }

    /// Linear attention total key dimension.
    public var linearTotalKey: Int { linearNumKHeads * linearKeyDim }

    /// Whether a given layer uses full attention (vs. linear).
    public func isFullAttention(layer: Int) -> Bool {
        (layer + 1) % fullAttentionInterval == 0
    }

    /// Full attention KV cache index for a given layer.
    public func fullAttentionIndex(layer: Int) -> Int {
        (layer + 1) / fullAttentionInterval - 1
    }

    /// Linear attention state index for a given layer.
    public func linearAttentionIndex(layer: Int) -> Int {
        layer - (layer + 1) / fullAttentionInterval
    }

    /// Expert size for the given quantization mode.
    public func expertSize(use2Bit: Bool) -> Int {
        use2Bit ? expertSize2Bit : expertSize4Bit
    }
}

// MARK: - Presets

extension ModelConfig {

    /// Qwen3.5-397B-A17B — the original Flash-MoE target.
    ///
    /// 60 layers, 512 experts, hidden_dim=4096, 45 GatedDeltaNet + 15 full attention.
    public static let qwen397B = ModelConfig(
        hiddenDim: 4096, numLayers: 60, numAttentionHeads: 32, numKVHeads: 2,
        headDim: 256, vocabSize: 248_320, rmsNormEps: 1e-6,
        numExperts: 512, numExpertsPerToken: 10, moeIntermediate: 1024, sharedIntermediate: 1024,
        expertSize4Bit: 7_077_888, expertSize2Bit: 3_932_160,
        gateWeightsOffset2Bit: 0, gateScalesOffset2Bit: 1_048_576, gateBiasesOffset2Bit: 1_179_648,
        upWeightsOffset2Bit: 1_310_720, upScalesOffset2Bit: 2_359_296, upBiasesOffset2Bit: 2_490_368,
        downWeightsOffset2Bit: 2_621_440, downScalesOffset2Bit: 3_670_016, downBiasesOffset2Bit: 3_801_088,
        groupSize: 64, bits: 4,
        fullAttentionInterval: 4, ropeTheta: 10_000_000.0, partialRotary: 0.25,
        linearNumVHeads: 64, linearNumKHeads: 16, linearKeyDim: 128, linearValueDim: 128, convKernelSize: 4,
        eosToken1: 248_046, eosToken2: 248_044, thinkStartToken: 248_068, thinkEndToken: 248_069
    )

    /// Tiny model for integration testing.
    ///
    /// Runs the full pipeline with minimal memory (~1MB total).
    /// 2 layers (1 linear + 1 full attention), 4 experts, hidden=64, vocab=32.
    public static let tiny = ModelConfig(
        hiddenDim: 64, numLayers: 2, numAttentionHeads: 2, numKVHeads: 1,
        headDim: 32, vocabSize: 32, rmsNormEps: 1e-6,
        numExperts: 4, numExpertsPerToken: 2, moeIntermediate: 32, sharedIntermediate: 32,
        // Expert layout for tiny: gate[32,64] + up[32,64] + down[64,32] at 4-bit
        // packed_cols: gate/up = 64/8=8, down = 32/8=4
        // gate weights: 32*8*4 = 1024 bytes, scales: 32*(64/64)*2 = 64 bytes, biases: 64 bytes
        // up: same as gate
        // down weights: 64*4*4 = 1024 bytes, scales: 64*(32/64)*2 = 64 bytes (1 group for 32)
        // Total rough: ~3200 bytes per expert
        expertSize4Bit: 4096,  // Round up for simplicity
        expertSize2Bit: 2048,
        gateWeightsOffset2Bit: 0, gateScalesOffset2Bit: 512, gateBiasesOffset2Bit: 576,
        upWeightsOffset2Bit: 640, upScalesOffset2Bit: 1152, upBiasesOffset2Bit: 1216,
        downWeightsOffset2Bit: 1280, downScalesOffset2Bit: 1792, downBiasesOffset2Bit: 1856,
        groupSize: 64, bits: 4,
        fullAttentionInterval: 2, ropeTheta: 10000.0, partialRotary: 0.25,
        linearNumVHeads: 2, linearNumKHeads: 1, linearKeyDim: 16, linearValueDim: 16, convKernelSize: 4,
        eosToken1: 30, eosToken2: 31, thinkStartToken: 28, thinkEndToken: 29
    )
}
