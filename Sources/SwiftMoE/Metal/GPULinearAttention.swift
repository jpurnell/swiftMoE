import Metal

/// Encodes the full GatedDeltaNet linear attention pipeline on GPU within CMD1.
///
/// Adds 5 compute encoders to the command buffer (after the projection dispatches):
/// 1. `conv1d_step` — temporal convolution with SiLU
/// 2. `rms_norm_qk` — per-head Q/K normalization
/// 3. `compute_decay_beta` — decay and beta gate from alpha/beta projections
/// 4. `gated_delta_net_step` — the main state recurrence
/// 5. `gated_rms_norm` — z-gated output normalization → batchSlots[6]
///
/// When this runs, Phase 2 skips the CPU linear attention path entirely.
///
/// Matches `infer.m:4098-4187` (GPU linear attention encoding in CMD1).
public enum GPULinearAttention {

    /// Whether GPU linear attention can be used for a given layer.
    ///
    /// Requires all 5 optional shader pipelines + weight pointers.
    public static func isAvailable(context: MetalContext, weights: LayerWeightPointers) -> Bool {
        context.shaders.deltaNetStep != nil
            && context.shaders.conv1dStep != nil
            && context.shaders.rmsNormQK != nil
            && context.shaders.computeDecayBeta != nil
            && context.shaders.gatedRmsNorm != nil
            && context.weightBuffer != nil
            && weights.conv1dW != nil
            && weights.aLog != nil
            && weights.dtBias != nil
            && weights.gatedNormW != nil
    }

    /// Encodes the linear attention pipeline into an existing CMD1 command buffer.
    ///
    /// Assumes batch matmul outputs are already in batchSlots:
    /// - `batchSlots[0]`: QKV projection [linearConvDim]
    /// - `batchSlots[1]`: Z projection [linearTotalValue]
    /// - `batchSlots[2]`: Beta projection [linearNumVHeads]
    /// - `batchSlots[3]`: Alpha projection [linearNumVHeads]
    ///
    /// Output is written to `batchSlots[6]` for CMD2's o_proj.
    ///
    /// - Parameters:
    ///   - context: Metal context.
    ///   - config: Model configuration.
    ///   - commandBuffer: CMD1 command buffer to append encoders to.
    ///   - weights: Layer weight pointers.
    ///   - linearLayerIndex: Index among linear attention layers (0..<45).
    public static func encode(
        context: MetalContext,
        config: ModelConfig,
        commandBuffer: MTLCommandBuffer,
        weights: LayerWeightPointers,
        linearLayerIndex: Int
    ) {
        guard let wfBuf = context.weightBuffer,
              let conv1dPipe = context.shaders.conv1dStep,
              let normQKPipe = context.shaders.rmsNormQK,
              let decayBetaPipe = context.shaders.computeDecayBeta,
              let deltaNetPipe = context.shaders.deltaNetStep,
              let gatedNormPipe = context.shaders.gatedRmsNorm,
              let conv1dW = weights.conv1dW,
              let aLog = weights.aLog,
              let dtBias = weights.dtBias,
              let gatedNormW = weights.gatedNormW else {
            return
        }

        let wfBase = UInt(bitPattern: wfBuf.contents())
        let totalKey = config.linearTotalKey
        let numVHeads = config.linearNumVHeads
        let numKHeads = config.linearNumKHeads
        let keyDim = config.linearKeyDim
        let valueDim = config.linearValueDim
        var convDim = UInt32(config.linearConvDim)

        // ---- Enc L1: conv1d_step ----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            let convWOff = Int(UInt(bitPattern: conv1dW) - wfBase)
            enc.setComputePipelineState(conv1dPipe)
            enc.setBuffer(context.linearAttention.convState[linearLayerIndex], offset: 0, index: 0)
            enc.setBuffer(context.projections.batchSlots[0], offset: 0, index: 1)
            enc.setBuffer(wfBuf, offset: convWOff, index: 2)
            enc.setBuffer(context.linearAttention.convOutput, offset: 0, index: 3)
            enc.setBytes(&convDim, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: Int((convDim + 255) / 256), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // ---- Enc L2: rms_norm_qk ----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            var kd = UInt32(keyDim)
            var invScale = 1.0 / sqrtf(Float(keyDim))
            enc.setComputePipelineState(normQKPipe)
            enc.setBuffer(context.linearAttention.convOutput, offset: 0, index: 0)
            enc.setBuffer(context.linearAttention.convOutput,
                          offset: totalKey * MemoryLayout<Float>.size, index: 1)
            enc.setBytes(&kd, length: 4, index: 2)
            enc.setBytes(&invScale, length: 4, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: numKHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: keyDim, height: 1, depth: 1))
            enc.endEncoding()
        }

        // ---- Enc L3: compute_decay_beta ----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            let aLogOff = Int(UInt(bitPattern: aLog) - wfBase)
            let dtBiasOff = Int(UInt(bitPattern: dtBias) - wfBase)
            enc.setComputePipelineState(decayBetaPipe)
            enc.setBuffer(context.projections.batchSlots[3], offset: 0, index: 0)  // alpha
            enc.setBuffer(context.projections.batchSlots[2], offset: 0, index: 1)  // beta
            enc.setBuffer(wfBuf, offset: aLogOff, index: 2)
            enc.setBuffer(wfBuf, offset: dtBiasOff, index: 3)
            enc.setBuffer(context.linearAttention.deltaGDecay, offset: 0, index: 4)
            enc.setBuffer(context.linearAttention.deltaBeta, offset: 0, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: numVHeads, height: 1, depth: 1))
            enc.endEncoding()
        }

        // ---- Enc L4: gated_delta_net_step ----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            var khpv = UInt32(numVHeads / numKHeads)
            enc.setComputePipelineState(deltaNetPipe)
            enc.setBuffer(context.linearAttention.deltaState[linearLayerIndex], offset: 0, index: 0)
            enc.setBuffer(context.linearAttention.convOutput, offset: 0, index: 1)  // q
            enc.setBuffer(context.linearAttention.convOutput,
                          offset: totalKey * MemoryLayout<Float>.size, index: 2)  // k
            enc.setBuffer(context.linearAttention.convOutput,
                          offset: 2 * totalKey * MemoryLayout<Float>.size, index: 3)  // v
            enc.setBuffer(context.linearAttention.deltaGDecay, offset: 0, index: 4)
            enc.setBuffer(context.linearAttention.deltaBeta, offset: 0, index: 5)
            enc.setBuffer(context.linearAttention.deltaOutput, offset: 0, index: 6)
            enc.setBytes(&khpv, length: 4, index: 7)
            enc.dispatchThreadgroups(
                MTLSize(width: numVHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
        }

        // ---- Enc L5: gated_rms_norm → batchSlots[6] ----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            let gnormOff = Int(UInt(bitPattern: gatedNormW) - wfBase)
            var vd = UInt32(valueDim)
            var eps = config.rmsNormEps
            enc.setComputePipelineState(gatedNormPipe)
            enc.setBuffer(context.linearAttention.deltaOutput, offset: 0, index: 0)
            enc.setBuffer(context.projections.batchSlots[1], offset: 0, index: 1)  // z
            enc.setBuffer(wfBuf, offset: gnormOff, index: 2)
            enc.setBuffer(context.projections.batchSlots[6], offset: 0, index: 3)
            enc.setBytes(&vd, length: 4, index: 4)
            enc.setBytes(&eps, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: numVHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: valueDim, height: 1, depth: 1))
            enc.endEncoding()
        }
    }
}
