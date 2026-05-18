import Metal
import Foundation

/// Compiles Metal shaders from source and holds all compute pipeline states.
///
/// Pipeline states are immutable and thread-safe after creation, so this type
/// is `Sendable`. Created once at startup via runtime compilation of `shaders.metal`.
///
/// ## Pipeline Categories
/// - **Required:** `matvecV3`, `matvecFast`, `rmsNormSum`, `rmsNormApply`, `swiglu`, `moeCombineResidual`
/// - **Optional:** Delta-net pipelines fall back to CPU if unavailable
public struct ShaderLibrary: Sendable {

    // MARK: - Required Pipelines

    /// 4-bit dequant matvec kernel (v3, 8 rows/threadgroup, for dim <= 4096).
    public let matvecV3: MTLComputePipelineState
    /// 4-bit dequant matvec kernel (v5, alternative tiling strategy).
    public let matvecV5: MTLComputePipelineState
    /// 4-bit dequant matvec kernel (fast, 1 row/threadgroup, for large dims).
    public let matvecFast: MTLComputePipelineState
    /// 2-bit dequant matvec kernel for requantized experts.
    public let matvec2Bit: MTLComputePipelineState
    /// RMS norm pass 1: computes sum of squares reduction.
    public let rmsNormSum: MTLComputePipelineState
    /// RMS norm pass 2: applies normalization with float weights.
    public let rmsNormApply: MTLComputePipelineState
    /// RMS norm pass 2: applies normalization with BF16 weights.
    public let rmsNormApplyBf16: MTLComputePipelineState
    /// Element-wise residual addition.
    public let residualAdd: MTLComputePipelineState
    /// Fused SwiGLU activation (gate * silu(up)).
    public let swiglu: MTLComputePipelineState
    /// Batched attention: Q @ K^T score computation.
    public let attnScores: MTLComputePipelineState
    /// Batched attention: softmax over scores.
    public let attnSoftmax: MTLComputePipelineState
    /// Batched attention: scores @ V value computation.
    public let attnValues: MTLComputePipelineState
    /// Sigmoid gating for shared expert contribution.
    public let sigmoidGate: MTLComputePipelineState
    /// Fused MoE combine + residual + norm kernel.
    public let moeCombineResidual: MTLComputePipelineState

    // MARK: - Optional Pipelines (GPU linear attention — CPU fallback if nil)

    /// GatedDeltaNet recurrence step (GPU accelerated).
    public let deltaNetStep: MTLComputePipelineState?
    /// Conv1d single-step kernel for linear attention.
    public let conv1dStep: MTLComputePipelineState?
    /// Fused RMS norm for Q and K projections.
    public let rmsNormQK: MTLComputePipelineState?
    /// Computes per-head decay and beta from a_log and dt_bias.
    public let computeDecayBeta: MTLComputePipelineState?
    /// Gated RMS normalization for linear attention output.
    public let gatedRmsNorm: MTLComputePipelineState?

    /// Compiles shaders from source and creates all pipeline states.
    ///
    /// - Parameters:
    ///   - device: Metal device for pipeline creation.
    ///   - shaderPath: Path to `shaders.metal` source file.
    /// - Throws: ``FlashMoEError/fileNotFound(path:)`` or ``FlashMoEError/shaderCompilationFailed(reason:)``.
    public init(device: MTLDevice, shaderPath: String) throws {
        // silent: non-critical path — throws a more specific error below
        guard let source = try? String(contentsOfFile: shaderPath, encoding: .utf8) else {
            throw FlashMoEError.fileNotFound(path: shaderPath)
        }

        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) {
            options.mathMode = .fast
        }
        options.languageVersion = .version3_1

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: options)
        } catch {
            throw FlashMoEError.shaderCompilationFailed(reason: error.localizedDescription)
        }

        // Helper to create a pipeline state from a function name
        func makePipeline(_ name: String) -> MTLComputePipelineState? {
            guard let function = library.makeFunction(name: name) else { return nil }
            return try? device.makeComputePipelineState(function: function) // silent: optional pipelines return nil on failure
        }

        func requirePipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let pipeline = makePipeline(name) else {
                throw FlashMoEError.shaderCompilationFailed(
                    reason: "Required shader function '\(name)' not found or failed to create pipeline"
                )
            }
            return pipeline
        }

        // Required pipelines
        self.matvecV3 = try requirePipeline("dequant_matvec_4bit_v3")
        self.matvecV5 = try requirePipeline("dequant_matvec_4bit_v5")
        self.matvecFast = try requirePipeline("dequant_matvec_4bit_fast")
        self.matvec2Bit = try requirePipeline("dequant_matvec_2bit")
        self.rmsNormSum = try requirePipeline("rms_norm_sum_sq")
        self.rmsNormApply = try requirePipeline("rms_norm_apply")
        self.rmsNormApplyBf16 = try requirePipeline("rms_norm_apply_bf16")
        self.residualAdd = try requirePipeline("residual_add")
        self.swiglu = try requirePipeline("swiglu_fused")
        self.attnScores = try requirePipeline("attn_scores_batched")
        self.attnSoftmax = try requirePipeline("attn_softmax_batched")
        self.attnValues = try requirePipeline("attn_values_batched")
        self.sigmoidGate = try requirePipeline("sigmoid_gate")
        self.moeCombineResidual = try requirePipeline("moe_combine_residual")

        // Optional pipelines (GPU linear attention — CPU fallback if absent)
        self.deltaNetStep = makePipeline("gated_delta_net_step")
        self.conv1dStep = makePipeline("conv1d_step")
        self.rmsNormQK = makePipeline("rms_norm_qk")
        self.computeDecayBeta = makePipeline("compute_decay_beta")
        self.gatedRmsNorm = makePipeline("gated_rms_norm")
    }
}
