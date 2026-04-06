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

    public let matvecV3: MTLComputePipelineState
    public let matvecV5: MTLComputePipelineState
    public let matvecFast: MTLComputePipelineState
    public let matvec2Bit: MTLComputePipelineState
    public let rmsNormSum: MTLComputePipelineState
    public let rmsNormApply: MTLComputePipelineState
    public let rmsNormApplyBf16: MTLComputePipelineState
    public let residualAdd: MTLComputePipelineState
    public let swiglu: MTLComputePipelineState
    public let attnScores: MTLComputePipelineState
    public let attnSoftmax: MTLComputePipelineState
    public let attnValues: MTLComputePipelineState
    public let sigmoidGate: MTLComputePipelineState
    public let moeCombineResidual: MTLComputePipelineState

    // MARK: - Optional Pipelines (GPU linear attention — CPU fallback if nil)

    public let deltaNetStep: MTLComputePipelineState?
    public let conv1dStep: MTLComputePipelineState?
    public let rmsNormQK: MTLComputePipelineState?
    public let computeDecayBeta: MTLComputePipelineState?
    public let gatedRmsNorm: MTLComputePipelineState?

    /// Compiles shaders from source and creates all pipeline states.
    ///
    /// - Parameters:
    ///   - device: Metal device for pipeline creation.
    ///   - shaderPath: Path to `shaders.metal` source file.
    /// - Throws: ``FlashMoEError/fileNotFound(path:)`` or ``FlashMoEError/shaderCompilationFailed(reason:)``.
    public init(device: MTLDevice, shaderPath: String) throws {
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
            return try? device.makeComputePipelineState(function: function)
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
