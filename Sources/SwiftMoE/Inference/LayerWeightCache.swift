import Foundation

/// Pre-computed weight pointers for a single transformer layer.
///
/// Eliminates ~40 `snprintf` + hash table lookups per layer at inference time
/// by caching raw pointers into the mmap'd weight file. Built once at startup
/// via ``LayerWeightCacheBuilder``.
///
/// Replaces the C `LayerWeightCache` struct from `infer.m:3649-3678`.
public struct LayerWeightPointers {

    // MARK: - Layer Norms

    /// Input layer norm weights (BF16).
    public var inputNormW: UnsafePointer<UInt16>?
    /// Post-attention layer norm weights (BF16).
    public var postAttnNormW: UnsafePointer<UInt16>?

    // MARK: - Full Attention Projections (15 layers)

    /// Query projection weights (packed 4-bit).
    public var qW: UnsafePointer<UInt32>?
    /// Query projection scales (BF16).
    public var qS: UnsafePointer<UInt16>?
    /// Query projection biases (BF16).
    public var qB: UnsafePointer<UInt16>?
    /// Key projection weights (packed 4-bit).
    public var kW: UnsafePointer<UInt32>?
    /// Key projection scales (BF16).
    public var kS: UnsafePointer<UInt16>?
    /// Key projection biases (BF16).
    public var kB: UnsafePointer<UInt16>?
    /// Value projection weights (packed 4-bit).
    public var vW: UnsafePointer<UInt32>?
    /// Value projection scales (BF16).
    public var vS: UnsafePointer<UInt16>?
    /// Value projection biases (BF16).
    public var vB: UnsafePointer<UInt16>?
    /// Output projection weights (packed 4-bit).
    public var oW: UnsafePointer<UInt32>?
    /// Output projection scales (BF16).
    public var oS: UnsafePointer<UInt16>?
    /// Output projection biases (BF16).
    public var oB: UnsafePointer<UInt16>?
    /// Query RMS norm weights (BF16).
    public var qNormW: UnsafePointer<UInt16>?
    /// Key RMS norm weights (BF16).
    public var kNormW: UnsafePointer<UInt16>?

    // MARK: - Linear Attention Projections (45 layers)

    /// Fused QKV projection weights (packed 4-bit).
    public var qkvW: UnsafePointer<UInt32>?
    /// Fused QKV projection scales (BF16).
    public var qkvS: UnsafePointer<UInt16>?
    /// Fused QKV projection biases (BF16).
    public var qkvB: UnsafePointer<UInt16>?
    /// Z (gate) projection weights (packed 4-bit).
    public var zW: UnsafePointer<UInt32>?
    /// Z (gate) projection scales (BF16).
    public var zS: UnsafePointer<UInt16>?
    /// Z (gate) projection biases (BF16).
    public var zB: UnsafePointer<UInt16>?
    /// Beta (update gate) projection weights (packed 4-bit).
    public var betaW: UnsafePointer<UInt32>?
    /// Beta (update gate) projection scales (BF16).
    public var betaS: UnsafePointer<UInt16>?
    /// Beta (update gate) projection biases (BF16).
    public var betaB: UnsafePointer<UInt16>?
    /// Alpha (decay) projection weights (packed 4-bit).
    public var alphaW: UnsafePointer<UInt32>?
    /// Alpha (decay) projection scales (BF16).
    public var alphaS: UnsafePointer<UInt16>?
    /// Alpha (decay) projection biases (BF16).
    public var alphaB: UnsafePointer<UInt16>?
    /// Conv1d kernel weights (BF16).
    public var conv1dW: UnsafePointer<UInt16>?
    /// Logarithmic decay factors for GatedDeltaNet.
    public var aLog: UnsafePointer<Float>?
    /// Delta-net time-step bias (BF16).
    public var dtBias: UnsafePointer<UInt16>?
    /// Gated RMS norm weights for linear attention output (BF16).
    public var gatedNormW: UnsafePointer<UInt16>?
    /// Output projection weights (packed 4-bit).
    public var outProjW: UnsafePointer<UInt32>?
    /// Output projection scales (BF16).
    public var outProjS: UnsafePointer<UInt16>?
    /// Output projection biases (BF16).
    public var outProjB: UnsafePointer<UInt16>?

    // MARK: - MoE Routing + Shared Expert

    /// Router gate projection weights (packed 4-bit).
    public var gateW: UnsafePointer<UInt32>?
    /// Router gate projection scales (BF16).
    public var gateS: UnsafePointer<UInt16>?
    /// Router gate projection biases (BF16).
    public var gateB: UnsafePointer<UInt16>?

    /// Shared expert gate projection weights (packed 4-bit).
    public var sharedGateW: UnsafePointer<UInt32>?
    /// Shared expert gate projection scales (BF16).
    public var sharedGateS: UnsafePointer<UInt16>?
    /// Shared expert gate projection biases (BF16).
    public var sharedGateB: UnsafePointer<UInt16>?

    /// Shared expert up projection weights (packed 4-bit).
    public var sharedUpW: UnsafePointer<UInt32>?
    /// Shared expert up projection scales (BF16).
    public var sharedUpS: UnsafePointer<UInt16>?
    /// Shared expert up projection biases (BF16).
    public var sharedUpB: UnsafePointer<UInt16>?

    /// Shared expert sigmoid gate weights (packed 4-bit).
    public var sharedExpertGateW: UnsafePointer<UInt32>?
    /// Shared expert sigmoid gate scales (BF16).
    public var sharedExpertGateS: UnsafePointer<UInt16>?
    /// Shared expert sigmoid gate biases (BF16).
    public var sharedExpertGateB: UnsafePointer<UInt16>?

    /// Shared expert down projection weights (packed 4-bit).
    public var sharedDownW: UnsafePointer<UInt32>?
    /// Shared expert down projection scales (BF16).
    public var sharedDownS: UnsafePointer<UInt16>?
    /// Shared expert down projection biases (BF16).
    public var sharedDownB: UnsafePointer<UInt16>?

    /// Creates a new empty LayerWeightPointers.
    public init() {}
}

/// Builds the layer weight cache from a mmap'd weight file.
///
/// Replaces `build_layer_cache()` from `infer.m:3683-3804`.
/// Runs once at startup, pre-computing all tensor pointers for all 60 layers.
public enum LayerWeightCacheBuilder {

    /// Builds weight pointer caches for all 60 layers.
    ///
    /// - Parameters:
    ///   - weightFile: The mmap'd weight file.
    ///   - config: Model configuration providing layer count and attention layout.
    /// - Returns: Array of 60 `LayerWeightPointers`, one per layer.
    public static func build(from weightFile: WeightFile, config: ModelConfig) -> [LayerWeightPointers] {
        var caches = [LayerWeightPointers](repeating: LayerWeightPointers(), count: config.numLayers)

        for i in 0..<config.numLayers {
            let prefix = "model.layers.\(i)"
            let isFull = config.isFullAttention(layer: i)

            // Layer norms
            caches[i].inputNormW = weightFile.tensorPointer(
                name: "\(prefix).input_layernorm.weight", as: UInt16.self)
            caches[i].postAttnNormW = weightFile.tensorPointer(
                name: "\(prefix).post_attention_layernorm.weight", as: UInt16.self)

            if isFull {
                // Full attention projections
                let attn = "\(prefix).self_attn"
                caches[i].qW = weightFile.tensorPointer(name: "\(attn).q_proj.weight", as: UInt32.self)
                caches[i].qS = weightFile.tensorPointer(name: "\(attn).q_proj.scales", as: UInt16.self)
                caches[i].qB = weightFile.tensorPointer(name: "\(attn).q_proj.biases", as: UInt16.self)
                caches[i].kW = weightFile.tensorPointer(name: "\(attn).k_proj.weight", as: UInt32.self)
                caches[i].kS = weightFile.tensorPointer(name: "\(attn).k_proj.scales", as: UInt16.self)
                caches[i].kB = weightFile.tensorPointer(name: "\(attn).k_proj.biases", as: UInt16.self)
                caches[i].vW = weightFile.tensorPointer(name: "\(attn).v_proj.weight", as: UInt32.self)
                caches[i].vS = weightFile.tensorPointer(name: "\(attn).v_proj.scales", as: UInt16.self)
                caches[i].vB = weightFile.tensorPointer(name: "\(attn).v_proj.biases", as: UInt16.self)
                caches[i].oW = weightFile.tensorPointer(name: "\(attn).o_proj.weight", as: UInt32.self)
                caches[i].oS = weightFile.tensorPointer(name: "\(attn).o_proj.scales", as: UInt16.self)
                caches[i].oB = weightFile.tensorPointer(name: "\(attn).o_proj.biases", as: UInt16.self)
                caches[i].qNormW = weightFile.tensorPointer(name: "\(attn).q_norm.weight", as: UInt16.self)
                caches[i].kNormW = weightFile.tensorPointer(name: "\(attn).k_norm.weight", as: UInt16.self)
            } else {
                // Linear attention projections
                let attn = "\(prefix).self_attn"
                caches[i].qkvW = weightFile.tensorPointer(name: "\(attn).qkv_proj.weight", as: UInt32.self)
                caches[i].qkvS = weightFile.tensorPointer(name: "\(attn).qkv_proj.scales", as: UInt16.self)
                caches[i].qkvB = weightFile.tensorPointer(name: "\(attn).qkv_proj.biases", as: UInt16.self)
                caches[i].zW = weightFile.tensorPointer(name: "\(attn).z_proj.weight", as: UInt32.self)
                caches[i].zS = weightFile.tensorPointer(name: "\(attn).z_proj.scales", as: UInt16.self)
                caches[i].zB = weightFile.tensorPointer(name: "\(attn).z_proj.biases", as: UInt16.self)
                caches[i].betaW = weightFile.tensorPointer(name: "\(attn).beta_proj.weight", as: UInt32.self)
                caches[i].betaS = weightFile.tensorPointer(name: "\(attn).beta_proj.scales", as: UInt16.self)
                caches[i].betaB = weightFile.tensorPointer(name: "\(attn).beta_proj.biases", as: UInt16.self)
                caches[i].alphaW = weightFile.tensorPointer(name: "\(attn).alpha_proj.weight", as: UInt32.self)
                caches[i].alphaS = weightFile.tensorPointer(name: "\(attn).alpha_proj.scales", as: UInt16.self)
                caches[i].alphaB = weightFile.tensorPointer(name: "\(attn).alpha_proj.biases", as: UInt16.self)
                caches[i].conv1dW = weightFile.tensorPointer(name: "\(attn).conv1d.weight", as: UInt16.self)
                caches[i].aLog = weightFile.tensorPointer(name: "\(attn).a_log", as: Float.self)
                caches[i].dtBias = weightFile.tensorPointer(name: "\(attn).dt_bias", as: UInt16.self)
                caches[i].gatedNormW = weightFile.tensorPointer(name: "\(attn).g_norm.weight", as: UInt16.self)
                caches[i].outProjW = weightFile.tensorPointer(name: "\(attn).out_proj.weight", as: UInt32.self)
                caches[i].outProjS = weightFile.tensorPointer(name: "\(attn).out_proj.scales", as: UInt16.self)
                caches[i].outProjB = weightFile.tensorPointer(name: "\(attn).out_proj.biases", as: UInt16.self)
            }

            // MoE routing
            let moe = "\(prefix).mlp"
            caches[i].gateW = weightFile.tensorPointer(name: "\(moe).gate.weight", as: UInt32.self)
            caches[i].gateS = weightFile.tensorPointer(name: "\(moe).gate.scales", as: UInt16.self)
            caches[i].gateB = weightFile.tensorPointer(name: "\(moe).gate.biases", as: UInt16.self)

            // Shared expert
            let shared = "\(moe).shared_expert"
            caches[i].sharedGateW = weightFile.tensorPointer(name: "\(shared).gate_proj.weight", as: UInt32.self)
            caches[i].sharedGateS = weightFile.tensorPointer(name: "\(shared).gate_proj.scales", as: UInt16.self)
            caches[i].sharedGateB = weightFile.tensorPointer(name: "\(shared).gate_proj.biases", as: UInt16.self)
            caches[i].sharedUpW = weightFile.tensorPointer(name: "\(shared).up_proj.weight", as: UInt32.self)
            caches[i].sharedUpS = weightFile.tensorPointer(name: "\(shared).up_proj.scales", as: UInt16.self)
            caches[i].sharedUpB = weightFile.tensorPointer(name: "\(shared).up_proj.biases", as: UInt16.self)
            caches[i].sharedDownW = weightFile.tensorPointer(name: "\(shared).down_proj.weight", as: UInt32.self)
            caches[i].sharedDownS = weightFile.tensorPointer(name: "\(shared).down_proj.scales", as: UInt16.self)
            caches[i].sharedDownB = weightFile.tensorPointer(name: "\(shared).down_proj.biases", as: UInt16.self)

            // Shared expert gate
            caches[i].sharedExpertGateW = weightFile.tensorPointer(
                name: "\(moe).shared_expert_gate.weight", as: UInt32.self)
            caches[i].sharedExpertGateS = weightFile.tensorPointer(
                name: "\(moe).shared_expert_gate.scales", as: UInt16.self)
            caches[i].sharedExpertGateB = weightFile.tensorPointer(
                name: "\(moe).shared_expert_gate.biases", as: UInt16.self)
        }

        return caches
    }
}
