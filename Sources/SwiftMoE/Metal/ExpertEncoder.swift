import Metal

/// GPU compute encoder for MoE expert forward passes.
///
/// Encodes gate_proj + up_proj + SwiGLU + down_proj for K experts into a
/// single Metal command buffer. Each expert gets two encoders:
/// - **Encoder A**: gate_proj + up_proj (parallel, same input)
/// - **Encoder B**: SwiGLU + down_proj (serial, depends on A)
///
/// Matches `gpu_encode_experts_batched` from `infer.m:1695-1786`.
public enum ExpertEncoder {

    /// Expert weight layout offsets within a packed expert binary.
    public struct ExpertOffsets: Sendable {
        /// Byte offset to gate projection packed weights.
        public let gateW: Int, gateS: Int, gateB: Int
        /// Byte offset to up projection packed weights.
        public let upW: Int, upS: Int, upB: Int
        /// Byte offset to down projection packed weights.
        public let downW: Int, downS: Int, downB: Int

        /// 4-bit expert offsets (from infer.m:1711-1713).
        public static let fourBit = ExpertOffsets(
            gateW: 0, gateS: 2_097_152, gateB: 2_228_224,
            upW: 2_359_296, upS: 4_456_448, upB: 4_587_520,
            downW: 4_718_592, downS: 6_815_744, downB: 6_946_816
        )

        /// 2-bit expert offsets derived from the given model config.
        public static func twoBit(config: ModelConfig) -> ExpertOffsets {
            ExpertOffsets(
                gateW: config.gateWeightsOffset2Bit,
                gateS: config.gateScalesOffset2Bit,
                gateB: config.gateBiasesOffset2Bit,
                upW: config.upWeightsOffset2Bit,
                upS: config.upScalesOffset2Bit,
                upB: config.upBiasesOffset2Bit,
                downW: config.downWeightsOffset2Bit,
                downS: config.downScalesOffset2Bit,
                downB: config.downBiasesOffset2Bit
            )
        }
    }

    /// Encodes K expert forward passes into a command buffer.
    ///
    /// Each expert: gate_proj → up_proj → SwiGLU → down_proj.
    /// Expert input must already be in `context.experts.input`.
    /// Expert weight data must already be in `expertBuffers[k]`.
    ///
    /// - Parameters:
    ///   - context: Metal context with pipeline states.
    ///   - commandBuffer: Command buffer to encode into (not committed).
    ///   - config: Model configuration providing expert dimensions and intermediate size.
    ///   - K: Number of active experts.
    ///   - valid: Which expert slots loaded successfully.
    ///   - expertBuffers: Per-expert weight data Metal buffers.
    ///   - use2Bit: Whether experts use 2-bit quantization.
    public static func encode(
        context: MetalContext,
        commandBuffer: MTLCommandBuffer,
        config: ModelConfig,
        K: Int,
        valid: [Bool],
        expertBuffers: [MTLBuffer],
        use2Bit: Bool
    ) {
        let offsets = use2Bit ? ExpertOffsets.twoBit(config: config) : ExpertOffsets.fourBit
        let expertPipe = use2Bit ? context.shaders.matvec2Bit : context.shaders.matvecV3

        var gateUpOut = UInt32(config.moeIntermediate)
        var gateUpIn = UInt32(config.hiddenDim)
        var downOut = UInt32(config.hiddenDim)
        var downIn = UInt32(config.moeIntermediate)
        var gs = UInt32(config.groupSize)
        let gateUpTGs = Int((gateUpOut + 7) / 8)
        let downTGs = Int((downOut + 7) / 8)
        let swigluTGs = Int((gateUpOut + 255) / 256)

        for k in 0..<K {
            guard valid[k] else { continue }

            // Encoder A: gate_proj + up_proj
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                // gate_proj
                enc.setComputePipelineState(expertPipe)
                enc.setBuffer(expertBuffers[k], offset: offsets.gateW, index: 0)
                enc.setBuffer(expertBuffers[k], offset: offsets.gateS, index: 1)
                enc.setBuffer(expertBuffers[k], offset: offsets.gateB, index: 2)
                enc.setBuffer(context.experts.input, offset: 0, index: 3)
                enc.setBuffer(context.experts.gate[k], offset: 0, index: 4)
                enc.setBytes(&gateUpOut, length: 4, index: 5)
                enc.setBytes(&gateUpIn, length: 4, index: 6)
                enc.setBytes(&gs, length: 4, index: 7)
                enc.dispatchThreadgroups(
                    MTLSize(width: gateUpTGs, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )

                // up_proj (same encoder, serialized)
                enc.setBuffer(expertBuffers[k], offset: offsets.upW, index: 0)
                enc.setBuffer(expertBuffers[k], offset: offsets.upS, index: 1)
                enc.setBuffer(expertBuffers[k], offset: offsets.upB, index: 2)
                enc.setBuffer(context.experts.up[k], offset: 0, index: 4)
                enc.dispatchThreadgroups(
                    MTLSize(width: gateUpTGs, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.endEncoding()
            }

            // Encoder B: SwiGLU + down_proj
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                // SwiGLU
                enc.setComputePipelineState(context.shaders.swiglu)
                enc.setBuffer(context.experts.gate[k], offset: 0, index: 0)
                enc.setBuffer(context.experts.up[k], offset: 0, index: 1)
                enc.setBuffer(context.experts.activation[k], offset: 0, index: 2)
                enc.setBytes(&gateUpOut, length: 4, index: 3)
                enc.dispatchThreadgroups(
                    MTLSize(width: swigluTGs, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )

                // down_proj (serialized after SwiGLU)
                enc.setComputePipelineState(expertPipe)
                enc.setBuffer(expertBuffers[k], offset: offsets.downW, index: 0)
                enc.setBuffer(expertBuffers[k], offset: offsets.downS, index: 1)
                enc.setBuffer(expertBuffers[k], offset: offsets.downB, index: 2)
                enc.setBuffer(context.experts.activation[k], offset: 0, index: 3)
                enc.setBuffer(context.experts.output[k], offset: 0, index: 4)
                enc.setBytes(&downOut, length: 4, index: 5)
                enc.setBytes(&downIn, length: 4, index: 6)
                enc.setBytes(&gs, length: 4, index: 7)
                enc.dispatchThreadgroups(
                    MTLSize(width: downTGs, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.endEncoding()
            }
        }
    }

    /// Encodes a dequant matvec using custom input/output buffers.
    ///
    /// Used for shared expert down_proj which reads from `buf_shared_act`
    /// instead of the standard projection input.
    ///
    /// Matches `gpu_encode_dequant_matvec_with_io_bufs` from `infer.m:1465-1497`.
    public static func encodeMatvec(
        context: MetalContext,
        commandBuffer: MTLCommandBuffer,
        weights: UnsafeRawPointer,
        scales: UnsafeRawPointer,
        biases: UnsafeRawPointer,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        outDim: UInt32,
        inDim: UInt32,
        groupSize: UInt32
    ) {
        guard let wfBuf = context.weightBuffer else { return }
        let wfBase = UInt(bitPattern: wfBuf.contents())

        let wOff = Int(UInt(bitPattern: weights) - wfBase)
        let sOff = Int(UInt(bitPattern: scales) - wfBase)
        let bOff = Int(UInt(bitPattern: biases) - wfBase)

        let useV3 = inDim <= 4096

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(useV3 ? context.shaders.matvecV3 : context.shaders.matvecFast)
        enc.setBuffer(wfBuf, offset: wOff, index: 0)
        enc.setBuffer(wfBuf, offset: sOff, index: 1)
        enc.setBuffer(wfBuf, offset: bOff, index: 2)
        enc.setBuffer(inputBuffer, offset: 0, index: 3)
        enc.setBuffer(outputBuffer, offset: 0, index: 4)
        var od = outDim, id_ = inDim, gs = groupSize
        enc.setBytes(&od, length: 4, index: 5)
        enc.setBytes(&id_, length: 4, index: 6)
        enc.setBytes(&gs, length: 4, index: 7)

        if useV3 {
            enc.dispatchThreadgroups(
                MTLSize(width: Int((outDim + 7) / 8), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
        } else {
            enc.dispatchThreadgroups(
                MTLSize(width: Int(outDim), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        enc.endEncoding()
    }

    /// Encodes the GPU-side combine + residual + RMS norm into CMD3.
    ///
    /// This makes CMD3 self-contained: it produces `buf_input` for the next layer's CMD1,
    /// eliminating the CPU deferred_wait + finalize + input_norm at layer start.
    ///
    /// Matches `infer.m:5361-5431` (Enc C1 + C2 + C3).
    public static func encodeCombine(
        context: MetalContext,
        commandBuffer: MTLCommandBuffer,
        config: ModelConfig,
        actualK: Int,
        valid: [Bool],
        expertWeights: [Float],
        sharedGateScore: Float,
        nextLayerNormW: UnsafePointer<UInt16>
    ) {
        guard let wfBuf = context.weightBuffer else { return }

        // Prepare combine params
        let paramsPtr = context.combine.combineParams.contents()
            .assumingMemoryBound(to: Float.self)
        memset(paramsPtr, 0, 10 * MemoryLayout<Float>.size)
        for k in 0..<actualK {
            paramsPtr[k] = valid[k] ? expertWeights[k] : 0.0
        }
        paramsPtr[8] = sharedGateScore

        // Enc C1: moe_combine_residual
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(context.shaders.moeCombineResidual)
            enc.setBuffer(context.combine.hMid, offset: 0, index: 0)
            enc.setBuffer(context.experts.sharedOutput, offset: 0, index: 1)
            enc.setBuffer(context.combine.moeHidden, offset: 0, index: 2)
            for k in 0..<ExpertBuffers.maxK {
                enc.setBuffer(context.experts.output[k], offset: 0, index: 3 + k)
            }
            enc.setBuffer(context.combine.combineParams, offset: 0, index: 11)
            var dim = UInt32(config.hiddenDim)
            var kVal = UInt32(actualK)
            enc.setBytes(&dim, length: 4, index: 12)
            enc.setBytes(&kVal, length: 4, index: 13)
            enc.dispatchThreadgroups(
                MTLSize(width: Int((dim + 255) / 256), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
        }

        // Enc C2: rms_norm_sum_sq
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            var dim = UInt32(config.hiddenDim)
            enc.setComputePipelineState(context.shaders.rmsNormSum)
            enc.setBuffer(context.combine.moeHidden, offset: 0, index: 0)
            enc.setBuffer(context.combine.cmd3SumSq, offset: 0, index: 1)
            enc.setBytes(&dim, length: 4, index: 2)
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
        }

        // Enc C3: rms_norm_apply_bf16 → buf_input
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            let normOff = Int(UInt(bitPattern: nextLayerNormW) - UInt(bitPattern: wfBuf.contents()))
            var dim = UInt32(config.hiddenDim)
            var eps = config.rmsNormEps
            enc.setComputePipelineState(context.shaders.rmsNormApplyBf16)
            enc.setBuffer(context.combine.moeHidden, offset: 0, index: 0)
            enc.setBuffer(wfBuf, offset: normOff, index: 1)
            enc.setBuffer(context.combine.cmd3SumSq, offset: 0, index: 2)
            enc.setBuffer(context.projections.input, offset: 0, index: 3)
            enc.setBytes(&dim, length: 4, index: 4)
            enc.setBytes(&eps, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: Int((dim + 255) / 256), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
        }
    }
}
