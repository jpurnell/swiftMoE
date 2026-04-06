import Metal

/// Encodes CMD2: the fused o_proj + residual_add + rms_norm + routing + shared expert.
///
/// This eliminates a CPU round-trip by keeping the entire post-attention pipeline
/// on GPU in a single command buffer (8-12 compute encoders, 1 commit+wait).
///
/// Buffer flow: `batchSlots[6] → buf_output → buf_h_mid → buf_input → batchSlots[0-3]`
///
/// Matches `infer.m:4789-4994` (the "FULLY FUSED CMD2" path).
public enum CMD2Encoder {

    /// Encodes the fused CMD2 into a command buffer.
    ///
    /// - Parameters:
    ///   - context: Metal context with pipeline states and buffers.
    ///   - config: Model configuration.
    ///   - weights: Layer weight pointers.
    ///   - attnOut: CPU attention output (copied to GPU batchSlots[6]).
    ///   - residual: Saved residual from before attention.
    ///   - isFull: Whether this is a full attention layer.
    ///   - oProjInDim: Input dimension for o_proj (numHeads*headDim or linearTotalValue).
    /// - Returns: Tuple of (h_mid, h_post, gateScores, sharedGateScore) read back from GPU.
    public static func encode(
        context: MetalContext,
        config: ModelConfig,
        weights: LayerWeightPointers,
        attnOut: UnsafePointer<Float>,
        residual: UnsafePointer<Float>,
        isFull: Bool,
        oProjInDim: Int
    ) -> (hMid: [Float], hPost: [Float], gateScores: [Float], sharedGateScore: Float)? {
        guard let wfBuf = context.weightBuffer else { return nil }

        let hiddenDim = config.hiddenDim
        let hiddenBytes = hiddenDim * MemoryLayout<Float>.size

        // Copy attention output and residual to GPU
        memcpy(context.projections.batchSlots[6].contents(), attnOut,
               oProjInDim * MemoryLayout<Float>.size)
        memcpy(context.combine.residual.contents(), residual, hiddenBytes)

        guard let cmd = context.queue.makeCommandBuffer() else { return nil }

        // ---- Enc 1: o_proj matvec (batchSlots[6] → buf_output) ----
        let oProjW: UnsafePointer<UInt32>?
        let oProjS: UnsafePointer<UInt16>?
        let oProjB: UnsafePointer<UInt16>?

        if isFull {
            oProjW = weights.oW; oProjS = weights.oS; oProjB = weights.oB
        } else {
            oProjW = weights.outProjW; oProjS = weights.outProjS; oProjB = weights.outProjB
        }

        if let w = oProjW, let s = oProjS, let b = oProjB {
            ExpertEncoder.encodeMatvec(
                context: context, commandBuffer: cmd,
                weights: UnsafeRawPointer(w), scales: UnsafeRawPointer(s), biases: UnsafeRawPointer(b),
                inputBuffer: context.projections.batchSlots[6],
                outputBuffer: context.projections.output,
                outDim: UInt32(hiddenDim), inDim: UInt32(oProjInDim),
                groupSize: UInt32(config.groupSize)
            )
        }

        // ---- Enc 2: residual_add (buf_output + buf_residual → buf_h_mid) ----
        if let enc = cmd.makeComputeCommandEncoder() {
            var dim = UInt32(hiddenDim)
            enc.setComputePipelineState(context.shaders.residualAdd)
            enc.setBuffer(context.combine.residual, offset: 0, index: 0)
            enc.setBuffer(context.projections.output, offset: 0, index: 1)
            enc.setBuffer(context.combine.hMid, offset: 0, index: 2)
            enc.setBytes(&dim, length: 4, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: Int((dim + 255) / 256), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // ---- Enc 3: rms_norm_sum_sq (buf_h_mid → buf_sum_sq) ----
        if let enc = cmd.makeComputeCommandEncoder() {
            var dim = UInt32(hiddenDim)
            enc.setComputePipelineState(context.shaders.rmsNormSum)
            enc.setBuffer(context.combine.hMid, offset: 0, index: 0)
            enc.setBuffer(context.combine.sumSq, offset: 0, index: 1)
            enc.setBytes(&dim, length: 4, index: 2)
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // ---- Enc 4: rms_norm_apply_bf16 (buf_h_mid + norm_w → buf_input) ----
        if let normW = weights.postAttnNormW, let enc = cmd.makeComputeCommandEncoder() {
            let normOff = Int(UInt(bitPattern: normW) - UInt(bitPattern: wfBuf.contents()))
            var dim = UInt32(hiddenDim)
            var eps = config.rmsNormEps
            enc.setComputePipelineState(context.shaders.rmsNormApplyBf16)
            enc.setBuffer(context.combine.hMid, offset: 0, index: 0)
            enc.setBuffer(wfBuf, offset: normOff, index: 1)
            enc.setBuffer(context.combine.sumSq, offset: 0, index: 2)
            enc.setBuffer(context.projections.input, offset: 0, index: 3)
            enc.setBytes(&dim, length: 4, index: 4)
            enc.setBytes(&eps, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: Int((dim + 255) / 256), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // ---- Enc 5-8: routing + shared expert (read buf_input) ----
        // Gate matvec
        if let gw = weights.gateW, let gs = weights.gateS, let gb = weights.gateB {
            ExpertEncoder.encodeMatvec(
                context: context, commandBuffer: cmd,
                weights: .init(gw), scales: .init(gs), biases: .init(gb),
                inputBuffer: context.projections.input,
                outputBuffer: context.projections.batchSlots[0],
                outDim: UInt32(config.numExperts), inDim: UInt32(hiddenDim),
                groupSize: UInt32(config.groupSize))
        }
        // Shared gate proj
        if let w = weights.sharedGateW, let s = weights.sharedGateS, let b = weights.sharedGateB {
            ExpertEncoder.encodeMatvec(
                context: context, commandBuffer: cmd,
                weights: .init(w), scales: .init(s), biases: .init(b),
                inputBuffer: context.projections.input,
                outputBuffer: context.projections.batchSlots[1],
                outDim: UInt32(config.sharedIntermediate), inDim: UInt32(hiddenDim),
                groupSize: UInt32(config.groupSize))
        }
        // Shared up proj
        if let w = weights.sharedUpW, let s = weights.sharedUpS, let b = weights.sharedUpB {
            ExpertEncoder.encodeMatvec(
                context: context, commandBuffer: cmd,
                weights: .init(w), scales: .init(s), biases: .init(b),
                inputBuffer: context.projections.input,
                outputBuffer: context.projections.batchSlots[2],
                outDim: UInt32(config.sharedIntermediate), inDim: UInt32(hiddenDim),
                groupSize: UInt32(config.groupSize))
        }
        // Shared expert gate (sigmoid)
        if let w = weights.sharedExpertGateW, let s = weights.sharedExpertGateS,
           let b = weights.sharedExpertGateB {
            ExpertEncoder.encodeMatvec(
                context: context, commandBuffer: cmd,
                weights: .init(w), scales: .init(s), biases: .init(b),
                inputBuffer: context.projections.input,
                outputBuffer: context.projections.batchSlots[3],
                outDim: 1, inDim: UInt32(hiddenDim),
                groupSize: UInt32(config.groupSize))
        }

        // ---- Single commit+wait for all encoders ----
        cmd.commit()
        cmd.waitUntilCompleted()

        // Read back results
        var hMid = [Float](repeating: 0, count: hiddenDim)
        var hPost = [Float](repeating: 0, count: hiddenDim)
        var gateScores = [Float](repeating: 0, count: config.numExperts)
        var sharedGateScore: Float = 0

        memcpy(&hMid, context.combine.hMid.contents(), hiddenBytes)
        memcpy(&hPost, context.projections.input.contents(), hiddenBytes)
        memcpy(&gateScores, context.projections.batchSlots[0].contents(),
               config.numExperts * MemoryLayout<Float>.size)

        let sharedGatePtr = context.projections.batchSlots[3].contents()
            .assumingMemoryBound(to: Float.self)
        sharedGateScore = sharedGatePtr[0]

        return (hMid: hMid, hPost: hPost, gateScores: gateScores, sharedGateScore: sharedGateScore)
    }
}
