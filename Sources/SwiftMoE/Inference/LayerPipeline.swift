import Metal
import Foundation
import os

private let pipelineLogger = Logger(subsystem: "com.swiftmoe", category: "pipeline")

/// Orchestrates the CMD1→CMD2→CMD3 pipeline for a single transformer layer.
///
/// This is the Swift decomposition of `fused_layer_forward` from `infer.m:3998-5462`.
/// The original is a single 1,500-line function; this class breaks it into
/// documented pipeline phases that mirror the original's structure:
///
/// ## Pipeline Phases
/// ```
/// Phase 1: Deferred completion + CMD1 (attention projections)
///   - FAST PATH: prev CMD3 did GPU combine → submit CMD1 immediately
///   - SLOW PATH: wait for prev CMD3, CPU finalize, input norm, submit CMD1
///
/// Phase 2: CPU attention compute
///   - Full attention: RoPE + scaled dot-product + sigmoid gate
///   - Linear attention: conv1d + GatedDeltaNet recurrence + gated RMS norm
///
/// Phase 3: CMD2 (fused o_proj + residual + norm + routing + shared expert)
///   - 8-12 compute encoders in a single command buffer
///
/// Phase 4: Expert I/O + CMD3 (DEFERRED)
///   - Parallel pread K experts from SSD
///   - Encode expert forwards + shared SwiGLU + combine into CMD3
///   - Commit without waiting → GPU runs async
/// ```
///
/// ## Usage
/// ```swift
/// let pipeline = LayerPipeline(context: metalCtx)
/// for layer in 0..<60 {
///     pipeline.forward(layerIndex: layer, hidden: &hidden, ...)
/// }
/// pipeline.completeDeferredExperts()  // finalize last layer
/// ```
public final class LayerPipeline {

    /// The Metal context (device, shaders, buffers).
    public let context: MetalContext

    /// The model configuration.
    public let config: ModelConfig

    /// Deferred expert state from the previous layer's CMD3.
    public var deferred = DeferredExpertState()

    /// Whether timing instrumentation is enabled.
    public var timingEnabled = false

    /// Per-phase timing accumulator.
    public var timing = LayerTiming()

    /// Creates a new layer pipeline with the given Metal context and model configuration.
    public init(context: MetalContext, config: ModelConfig) {
        self.context = context
        self.config = config
    }

    /// Completes any pending deferred GPU expert computation.
    ///
    /// Call this after the last layer (layer 59) to finalize the output,
    /// or before reading the hidden state for any reason.
    public func completeDeferredExperts(hidden: UnsafeMutablePointer<Float>) {
        guard deferred.isActive else { return }
        deferred.waitForGPU()

        if deferred.isGPUCombined {
            // GPU already computed combine+residual+norm.
            // Read back hidden from buf_moe_hidden.
            memcpy(hidden, context.combine.moeHidden.contents(),
                   config.hiddenDim * MemoryLayout<Float>.size)
        } else {
            // CPU-side combine (fallback path)
            cpuFinalize(hidden: hidden)
        }

        deferred.deactivate()
    }

    /// Discards deferred expert results without reading them back.
    ///
    /// Used during prefill for intermediate tokens whose hidden state
    /// will be immediately overwritten by the next token's embedding.
    public func discardDeferredExperts() {
        guard deferred.isActive else { return }
        deferred.waitForGPU()
        deferred.deactivate()
    }

    // MARK: - Forward Pass

    /// Executes one transformer layer: attention → routing → expert I/O → expert compute.
    ///
    /// This is the Swift equivalent of `fused_layer_forward` from `infer.m:3998-5462`,
    /// decomposed into the 4 pipeline phases documented in the class header.
    public func forward(
        layerIndex: Int,
        hidden: UnsafeMutablePointer<Float>,
        weights: LayerWeightPointers,
        kvCache: inout KVCache?,
        linearState: inout LinearAttentionState?,
        position: Int,
        K: Int,
        expertFD: Int32,
        use2Bit: Bool,
        layerWeights: [LayerWeightPointers]
    ) {
        let hiddenDim = config.hiddenDim
        let hiddenBytes = hiddenDim * MemoryLayout<Float>.size
        let isFull = config.isFullAttention(layer: layerIndex)

        // Scratch buffers (reused across layers, allocated once)
        if residualScratch.isEmpty {
            residualScratch = [Float](repeating: 0, count: hiddenDim)
            normedScratch = [Float](repeating: 0, count: hiddenDim)
            hPostScratch = [Float](repeating: 0, count: hiddenDim)
            gateScoresScratch = [Float](repeating: 0, count: config.numExperts)
        }

        // Allocate attention output scratch based on layer type
        let attnOutDim: Int
        if isFull {
            attnOutDim = config.numAttentionHeads * config.headDim
        } else {
            attnOutDim = config.linearTotalValue
        }
        var attnOut = [Float](repeating: 0, count: attnOutDim)

        // ================================================================
        // PHASE 1: Deferred completion + CMD1 (attention projections)
        // ================================================================
        let prevGPUCombined = deferred.isActive && deferred.isGPUCombined

        if prevGPUCombined {
            // FAST PATH: prev CMD3 already computed combine+norm into buf_input.
            // Submit CMD1 immediately — GPU queue serializes CMD3(N-1) then CMD1(N).
            guard let cmd1 = context.queue.makeCommandBuffer() else {
                preconditionFailure("Metal command buffer allocation failed")
            }
            let specs = buildAttentionSpecs(weights: weights, isFull: isFull)
            if !specs.isEmpty {
                BatchMatvec.encode(context: context, commandBuffer: cmd1, specs: specs)
            }
            let gpuLinearAttn = !isFull && GPULinearAttention.isAvailable(context: context, weights: weights)
            if gpuLinearAttn {
                let linearIdx = config.linearAttentionIndex(layer: layerIndex)
                GPULinearAttention.encode(
                    context: context, config: config, commandBuffer: cmd1,
                    weights: weights, linearLayerIndex: linearIdx)
            }
            cmd1.commit()
            cmd1.waitUntilCompleted()

            // Now CMD3(N-1) is done. Read back hidden state.
            completeDeferredExperts(hidden: hidden)
            memcpy(&residualScratch, hidden, hiddenBytes)
        } else {
            // SLOW PATH: wait for prev CMD3, finalize, input norm, CMD1
            completeDeferredExperts(hidden: hidden)
            memcpy(&residualScratch, hidden, hiddenBytes)

            if let normW = weights.inputNormW {
                RMSNorm.apply(input: hidden, weights: normW,
                              output: &normedScratch, dim: hiddenDim)
            } else {
                memcpy(&normedScratch, hidden, hiddenBytes)
            }

            // Copy normed input to GPU and submit CMD1
            memcpy(context.projections.input.contents(), &normedScratch, hiddenBytes)

            let specs = buildAttentionSpecs(weights: weights, isFull: isFull)
            let gpuLinearAttn = !isFull && GPULinearAttention.isAvailable(context: context, weights: weights)

            if !specs.isEmpty, context.weightBuffer != nil {
                guard let cmd1 = context.queue.makeCommandBuffer() else {
                    preconditionFailure("Metal command buffer allocation failed")
                }
                BatchMatvec.encode(context: context, commandBuffer: cmd1, specs: specs)
                if gpuLinearAttn {
                    let linearIdx = config.linearAttentionIndex(layer: layerIndex)
                    GPULinearAttention.encode(
                        context: context, config: config, commandBuffer: cmd1,
                        weights: weights, linearLayerIndex: linearIdx)
                }
                cmd1.commit()
                cmd1.waitUntilCompleted()
            } else if !specs.isEmpty {
                // CPU fallback
                normedScratch.withUnsafeBufferPointer { normedBuf in
                    guard let normedBase = normedBuf.baseAddress else { return }
                    for spec in specs {
                        let outPtr = context.projections.batchSlots[spec.batchSlot].contents()
                            .assumingMemoryBound(to: Float.self)
                        Embedding.cpuDequantMatvec(
                            W: spec.weights.assumingMemoryBound(to: UInt32.self),
                            scales: spec.scales.assumingMemoryBound(to: UInt16.self),
                            biases: spec.biases.assumingMemoryBound(to: UInt16.self),
                            input: normedBase,
                            output: outPtr,
                            outDim: Int(spec.outDim), inDim: Int(spec.inDim),
                            groupSize: Int(spec.groupSize))
                    }
                }
            }
        }

        // ================================================================
        // PHASE 2: CPU attention compute (skipped when GPU linear attn ran)
        // ================================================================
        let gpuLinearAttnRan = !isFull && GPULinearAttention.isAvailable(context: context, weights: weights)

        var gpuFullAttnUsed = false
        if isFull, var kv = kvCache {
            let qDim = config.numAttentionHeads * config.headDim * 2
            let kvDim = config.kvDim
            var qProjOut = [Float](repeating: 0, count: qDim)
            var kOut = [Float](repeating: 0, count: kvDim)
            var vOut = [Float](repeating: 0, count: kvDim)
            memcpy(&qProjOut, context.projections.batchSlots[0].contents(), qDim * MemoryLayout<Float>.size)
            memcpy(&kOut, context.projections.batchSlots[1].contents(), kvDim * MemoryLayout<Float>.size)
            memcpy(&vOut, context.projections.batchSlots[2].contents(), kvDim * MemoryLayout<Float>.size)

            // Check if GPU attention should be used (seq_len >= 32)
            let useGPUAttn = GPUFullAttention.shouldUseGPU(
                context: context, config: config, kvCacheLength: kv.length + 1, layerIndex: layerIndex)

            if useGPUAttn {
                attnOut.withUnsafeMutableBufferPointer { outBuf in
                    guard let outBase = outBuf.baseAddress else { return }
                    FullAttention.forward(
                        qProjOut: &qProjOut, kOut: &kOut, vOut: &vOut,
                        kvCache: &kv, position: position, config: config,
                        qNormW: weights.qNormW, kNormW: weights.kNormW,
                        output: outBase)
                }
                memcpy(context.attention.output.contents(), &attnOut,
                       config.numAttentionHeads * config.headDim * MemoryLayout<Float>.size)
                gpuFullAttnUsed = true
            } else {
                attnOut.withUnsafeMutableBufferPointer { outBuf in
                    guard let outBase = outBuf.baseAddress else { return }
                    FullAttention.forward(
                        qProjOut: &qProjOut, kOut: &kOut, vOut: &vOut,
                        kvCache: &kv, position: position, config: config,
                        qNormW: weights.qNormW, kNormW: weights.kNormW,
                        output: outBase)
                }
            }
            kvCache = kv
        } else if !gpuLinearAttnRan, var ls = linearState {
            // CPU linear attention fallback
            let convDim = config.linearConvDim
            let totalValue = config.linearTotalValue
            let numVHeads = config.linearNumVHeads
            var qkvOut = [Float](repeating: 0, count: convDim)
            var zOut = [Float](repeating: 0, count: totalValue)
            var betaOut = [Float](repeating: 0, count: numVHeads)
            var alphaOut = [Float](repeating: 0, count: numVHeads)
            memcpy(&qkvOut, context.projections.batchSlots[0].contents(), convDim * MemoryLayout<Float>.size)
            memcpy(&zOut, context.projections.batchSlots[1].contents(), totalValue * MemoryLayout<Float>.size)
            memcpy(&betaOut, context.projections.batchSlots[2].contents(), numVHeads * MemoryLayout<Float>.size)
            memcpy(&alphaOut, context.projections.batchSlots[3].contents(), numVHeads * MemoryLayout<Float>.size)

            attnOut.withUnsafeMutableBufferPointer { outBuf in
                guard let outBase = outBuf.baseAddress else { return }
                qkvOut.withUnsafeBufferPointer { qkvBuf in
                    guard let qkvBase = qkvBuf.baseAddress else { return }
                    zOut.withUnsafeBufferPointer { zBuf in
                        guard let zBase = zBuf.baseAddress else { return }
                        betaOut.withUnsafeBufferPointer { betaBuf in
                            guard let betaBase = betaBuf.baseAddress else { return }
                            alphaOut.withUnsafeBufferPointer { alphaBuf in
                                guard let alphaBase = alphaBuf.baseAddress else { return }
                                LinearAttention.forward(
                                    qkvOut: qkvBase, zOut: zBase,
                                    betaOut: betaBase, alphaOut: alphaBase,
                                    state: &ls, config: config,
                                    conv1dW: weights.conv1dW, aLog: weights.aLog,
                                    dtBias: weights.dtBias, gatedNormW: weights.gatedNormW,
                                    output: outBase)
                            }
                        }
                    }
                }
            }
            linearState = ls
        }
        // If GPU linear attn ran, batchSlots[6] already has the output for CMD2

        // ================================================================
        // PHASE 3: CMD2 — GPU-fused o_proj + residual + norm + routing
        // ================================================================
        let oProjInDim = isFull
            ? config.numAttentionHeads * config.headDim
            : config.linearTotalValue

        var sharedGateScore: Float = 0
        let useGPUCmd2 = context.weightBuffer != nil
            && weights.postAttnNormW != nil
            && weights.gateW != nil

        if useGPUCmd2 {
            if gpuFullAttnUsed {
                let attnOutPtr = UnsafePointer(context.attention.output.contents()
                    .assumingMemoryBound(to: Float.self))
                if let result = CMD2Encoder.encode(
                    context: context, config: config, weights: weights,
                    attnOut: attnOutPtr, residual: &residualScratch,
                    isFull: isFull, oProjInDim: oProjInDim
                ) {
                    memcpy(hidden, result.hMid, hiddenBytes)
                    memcpy(&hPostScratch, result.hPost, hiddenBytes)
                    gateScoresScratch = result.gateScores
                    sharedGateScore = result.sharedGateScore
                }
            } else if gpuLinearAttnRan {
                let attnOutPtr = UnsafePointer(context.projections.batchSlots[6].contents()
                    .assumingMemoryBound(to: Float.self))
                if let result = CMD2Encoder.encode(
                    context: context, config: config, weights: weights,
                    attnOut: attnOutPtr, residual: &residualScratch,
                    isFull: isFull, oProjInDim: oProjInDim
                ) {
                    memcpy(hidden, result.hMid, hiddenBytes)
                    memcpy(&hPostScratch, result.hPost, hiddenBytes)
                    gateScoresScratch = result.gateScores
                    sharedGateScore = result.sharedGateScore
                }
            } else {
                attnOut.withUnsafeBufferPointer { attnBuf in
                    guard let attnOutPtr = attnBuf.baseAddress else { return }
                    if let result = CMD2Encoder.encode(
                        context: context, config: config, weights: weights,
                        attnOut: attnOutPtr, residual: &residualScratch,
                        isFull: isFull, oProjInDim: oProjInDim
                    ) {
                        memcpy(hidden, result.hMid, hiddenBytes)
                        memcpy(&hPostScratch, result.hPost, hiddenBytes)
                        gateScoresScratch = result.gateScores
                        sharedGateScore = result.sharedGateScore
                    }
                }
            }
        } else {
            // CPU fallback for CMD2
            var attnProjected = [Float](repeating: 0, count: hiddenDim)
            let oProjW = isFull ? weights.oW : weights.outProjW
            let oProjS = isFull ? weights.oS : weights.outProjS
            let oProjB = isFull ? weights.oB : weights.outProjB

            if let w = oProjW, let s = oProjS, let b = oProjB {
                attnOut.withUnsafeBufferPointer { attnBuf in
                    guard let attnBase = attnBuf.baseAddress else { return }
                    Embedding.cpuDequantMatvec(
                        W: UnsafePointer(OpaquePointer(w)), scales: s, biases: b,
                        input: attnBase, output: &attnProjected,
                        outDim: hiddenDim, inDim: oProjInDim, groupSize: config.groupSize)
                }
            }
            for i in 0..<hiddenDim { hidden[i] = residualScratch[i] + attnProjected[i] }
            if let normW = weights.postAttnNormW {
                RMSNorm.apply(input: hidden, weights: normW, output: &hPostScratch, dim: hiddenDim)
            } else { memcpy(&hPostScratch, hidden, hiddenBytes) }

            for i in 0..<gateScoresScratch.count { gateScoresScratch[i] = 0 }
            if let gw = weights.gateW, let gs = weights.gateS, let gb = weights.gateB {
                hPostScratch.withUnsafeBufferPointer { hBuf in
                    guard let hBase = hBuf.baseAddress else { return }
                    Embedding.cpuDequantMatvec(
                        W: UnsafePointer(OpaquePointer(gw)), scales: gs, biases: gb,
                        input: hBase, output: &gateScoresScratch,
                        outDim: config.numExperts, inDim: hiddenDim, groupSize: config.groupSize)
                }
            }
            if let segW = weights.sharedExpertGateW, let segS = weights.sharedExpertGateS,
               let segB = weights.sharedExpertGateB {
                hPostScratch.withUnsafeBufferPointer { hBuf in
                    guard let hBase = hBuf.baseAddress else { return }
                    Embedding.cpuDequantMatvec(
                        W: UnsafePointer(OpaquePointer(segW)), scales: segS, biases: segB,
                        input: hBase, output: &sharedGateScore,
                        outDim: 1, inDim: hiddenDim, groupSize: config.groupSize)
                }
            }
        }

        Softmax.apply(&gateScoresScratch, count: config.numExperts)
        let actualK = min(K, ExpertBuffers.maxK)
        let (expertIndices, expertWeightsArr) = TopK.select(scores: &gateScoresScratch, k: actualK)

        // ================================================================
        // PHASE 4: Expert I/O + CMD3 (DEFERRED)
        // ================================================================
        let expertSize = config.expertSize(use2Bit: use2Bit)

        // Parallel pread experts into Metal buffers
        var valid = [Bool](repeating: false, count: actualK)
        for k in 0..<actualK {
            let offset = off_t(expertIndices[k]) * off_t(expertSize)
            let dst = context.experts.dataA[k].pointer
            let bytesRead = pread(expertFD, dst, expertSize, offset)
            valid[k] = bytesRead == expertSize
        }

        // Copy expert input (h_post) to GPU buffer
        memcpy(context.experts.input.contents(), &hPostScratch, hiddenBytes)

        // Copy shared expert gate/up projections to GPU
        if let sgW = weights.sharedGateW, let sgS = weights.sharedGateS, let sgB = weights.sharedGateB {
            var sharedGate = [Float](repeating: 0, count: config.sharedIntermediate)
            hPostScratch.withUnsafeBufferPointer { hBuf in
                guard let hBase = hBuf.baseAddress else { return }
                Embedding.cpuDequantMatvec(
                    W: UnsafePointer(OpaquePointer(sgW)), scales: sgS, biases: sgB,
                    input: hBase, output: &sharedGate,
                    outDim: config.sharedIntermediate, inDim: hiddenDim, groupSize: config.groupSize
                )
            }
            memcpy(context.experts.sharedGate.contents(), &sharedGate,
                   config.sharedIntermediate * MemoryLayout<Float>.size)
        }
        if let suW = weights.sharedUpW, let suS = weights.sharedUpS, let suB = weights.sharedUpB {
            var sharedUp = [Float](repeating: 0, count: config.sharedIntermediate)
            hPostScratch.withUnsafeBufferPointer { hBuf in
                guard let hBase = hBuf.baseAddress else { return }
                Embedding.cpuDequantMatvec(
                    W: UnsafePointer(OpaquePointer(suW)), scales: suS, biases: suB,
                    input: hBase, output: &sharedUp,
                    outDim: config.sharedIntermediate, inDim: hiddenDim, groupSize: config.groupSize
                )
            }
            memcpy(context.experts.sharedUp.contents(), &sharedUp,
                   config.sharedIntermediate * MemoryLayout<Float>.size)
        }

        // Copy h_mid to GPU for combine
        memcpy(context.combine.hMid.contents(), hidden, hiddenBytes)

        // Encode CMD3: expert forwards + shared SwiGLU + shared down + combine
        guard let cmdExperts = context.queue.makeCommandBuffer() else {
            preconditionFailure("Metal command buffer allocation failed for experts")
        }

        let expertMetalBuffers = (0..<actualK).map { context.experts.dataA[$0].metalBuffer }
        ExpertEncoder.encode(
            context: context, commandBuffer: cmdExperts, config: config,
            K: actualK, valid: valid, expertBuffers: expertMetalBuffers, use2Bit: use2Bit
        )

        // Shared expert SwiGLU + down_proj
        if let enc = cmdExperts.makeComputeCommandEncoder() {
            enc.setComputePipelineState(context.shaders.swiglu)
            enc.setBuffer(context.experts.sharedGate, offset: 0, index: 0)
            enc.setBuffer(context.experts.sharedUp, offset: 0, index: 1)
            enc.setBuffer(context.experts.sharedActivation, offset: 0, index: 2)
            var dim = UInt32(config.sharedIntermediate)
            enc.setBytes(&dim, length: 4, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: Int((dim + 255) / 256), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // Shared down_proj
        if let sdW = weights.sharedDownW, let sdS = weights.sharedDownS, let sdB = weights.sharedDownB {
            ExpertEncoder.encodeMatvec(
                context: context, commandBuffer: cmdExperts,
                weights: UnsafeRawPointer(sdW), scales: UnsafeRawPointer(sdS), biases: UnsafeRawPointer(sdB),
                inputBuffer: context.experts.sharedActivation, outputBuffer: context.experts.sharedOutput,
                outDim: UInt32(hiddenDim), inDim: UInt32(config.sharedIntermediate),
                groupSize: UInt32(config.groupSize)
            )
        }

        // GPU-side combine (if not last layer)
        let gpuCombine = layerIndex < config.numLayers - 1
            && layerWeights[layerIndex + 1].inputNormW != nil

        if gpuCombine, let nextNormW = layerWeights[layerIndex + 1].inputNormW {
            ExpertEncoder.encodeCombine(
                context: context, commandBuffer: cmdExperts, config: config,
                actualK: actualK, valid: valid, expertWeights: expertWeightsArr,
                sharedGateScore: sharedGateScore, nextLayerNormW: nextNormW
            )
        }

        // DEFERRED commit
        cmdExperts.commit()

        deferred.activate(
            expertWeights: expertWeightsArr, valid: valid,
            sharedGateScore: sharedGateScore, layerIndex: layerIndex,
            gpuCombined: gpuCombine
        )
        deferred.commandBuffer = cmdExperts
        deferred.hiddenPointer = hidden
        if !gpuCombine {
            deferred.hMid = [Float](repeating: 0, count: hiddenDim)
            memcpy(&deferred.hMid, hidden, hiddenBytes)
        }
    }

    // MARK: - Scratch Buffers (allocated once, reused across layers)

    private var residualScratch: [Float] = []
    private var normedScratch: [Float] = []
    private var hPostScratch: [Float] = []
    private var gateScoresScratch: [Float] = []

    // MARK: - Helpers

    // Justification: dummyCPU is only used as a placeholder address for BatchMatvecSpec.outputCPU; never read or written concurrently.
    private nonisolated(unsafe) static let dummyCPU: UnsafeMutablePointer<Float> = .allocate(capacity: 1)

    /// Builds attention projection BatchMatvecSpecs for CMD1.
    ///
    /// Results go into GPU batch output slots. Phase 2 reads from `batchSlots[n].contents()`.
    /// The `outputCPU` field is a dummy — not used in this pipeline.
    private func buildAttentionSpecs(weights: LayerWeightPointers, isFull: Bool) -> [BatchMatvecSpec] {
        let dummy = LayerPipeline.dummyCPU
        var specs: [BatchMatvecSpec] = []
        let inDim = UInt32(config.hiddenDim)
        let gs = UInt32(config.groupSize)

        if isFull {
            guard let qW = weights.qW, let qS = weights.qS, let qB = weights.qB,
                  let kW = weights.kW, let kS = weights.kS, let kB = weights.kB,
                  let vW = weights.vW, let vS = weights.vS, let vB = weights.vB else {
                return []
            }
            let qDim = UInt32(config.numAttentionHeads * config.headDim * 2)
            let kvDim = UInt32(config.kvDim)

            specs.append(BatchMatvecSpec(weights: .init(qW), scales: .init(qS), biases: .init(qB),
                outputCPU: dummy, outDim: qDim, inDim: inDim, groupSize: gs, batchSlot: 0))
            specs.append(BatchMatvecSpec(weights: .init(kW), scales: .init(kS), biases: .init(kB),
                outputCPU: dummy, outDim: kvDim, inDim: inDim, groupSize: gs, batchSlot: 1))
            specs.append(BatchMatvecSpec(weights: .init(vW), scales: .init(vS), biases: .init(vB),
                outputCPU: dummy, outDim: kvDim, inDim: inDim, groupSize: gs, batchSlot: 2))
        } else {
            guard let qkvW = weights.qkvW, let qkvS = weights.qkvS, let qkvB = weights.qkvB,
                  let zW = weights.zW, let zS = weights.zS, let zB = weights.zB,
                  let bW = weights.betaW, let bS = weights.betaS, let bB = weights.betaB,
                  let aW = weights.alphaW, let aS = weights.alphaS, let aB = weights.alphaB else {
                return []
            }
            specs.append(BatchMatvecSpec(weights: .init(qkvW), scales: .init(qkvS), biases: .init(qkvB),
                outputCPU: dummy, outDim: UInt32(config.linearConvDim), inDim: inDim, groupSize: gs, batchSlot: 0))
            specs.append(BatchMatvecSpec(weights: .init(zW), scales: .init(zS), biases: .init(zB),
                outputCPU: dummy, outDim: UInt32(config.linearTotalValue), inDim: inDim, groupSize: gs, batchSlot: 1))
            specs.append(BatchMatvecSpec(weights: .init(bW), scales: .init(bS), biases: .init(bB),
                outputCPU: dummy, outDim: UInt32(config.linearNumVHeads), inDim: inDim, groupSize: gs, batchSlot: 2))
            specs.append(BatchMatvecSpec(weights: .init(aW), scales: .init(aS), biases: .init(aB),
                outputCPU: dummy, outDim: UInt32(config.linearNumVHeads), inDim: inDim, groupSize: gs, batchSlot: 3))
        }

        return specs
    }

    // MARK: - Private

    /// CPU-side combine: accumulate expert outputs + shared expert + residual.
    private func cpuFinalize(hidden: UnsafeMutablePointer<Float>) {
        let hiddenDim = config.hiddenDim

        // Accumulate weighted expert outputs
        var moeOut = [Float](repeating: 0, count: hiddenDim)
        for k in 0..<deferred.actualK {
            guard deferred.valid[k] else { continue }
            let expertResult = context.experts.output[k].contents()
                .assumingMemoryBound(to: Float.self)
            let weight = deferred.expertWeights[k]
            for i in 0..<hiddenDim {
                moeOut[i] += expertResult[i] * weight
            }
        }

        // Read shared expert result
        let sharedResult = context.experts.sharedOutput.contents()
            .assumingMemoryBound(to: Float.self)
        let expVal = 1.0 + expf(-deferred.sharedGateScore)
        let sharedWeight = expVal > 0 ? 1.0 / expVal : 0.5

        // Final combine: hidden = h_mid + moe_out + shared_out * sigmoid(gate)
        for i in 0..<hiddenDim {
            hidden[i] = deferred.hMid[i] + moeOut[i] + sharedResult[i] * sharedWeight
        }
    }
}

/// Per-phase timing accumulator for layer pipeline instrumentation.
///
/// Matches the `LayerTimingAccum` struct from `infer.m:146-160`.
public struct LayerTiming {
    /// Accumulated time waiting for deferred GPU completion (ms).
    public var deferredWait: Double = 0
    /// Accumulated CPU finalization time after deferred GPU (ms).
    public var deferredCPU: Double = 0
    /// Accumulated input RMS norm time (ms).
    public var inputNorm: Double = 0
    /// Accumulated CMD1 submission time (ms).
    public var cmd1Submit: Double = 0
    /// Accumulated CMD1 GPU wait time (ms).
    public var cmd1Wait: Double = 0
    /// Accumulated CPU attention compute time (ms).
    public var cpuAttention: Double = 0
    /// Accumulated CMD2 encoding time (ms).
    public var cmd2Encode: Double = 0
    /// Accumulated CMD2 GPU wait time (ms).
    public var cmd2Wait: Double = 0
    /// Accumulated CPU routing (softmax + topK) time (ms).
    public var routingCPU: Double = 0
    /// Accumulated expert I/O (pread) time (ms).
    public var expertIO: Double = 0
    /// Accumulated CMD3 encoding time (ms).
    public var cmd3Encode: Double = 0
    /// Accumulated total per-layer time (ms).
    public var totalLayer: Double = 0
    /// Number of layers timed.
    public var layerCount: Int = 0

    /// Resets all accumulators.
    public mutating func reset() {
        self = LayerTiming()
    }

    /// Returns a formatted summary matching the original's `timing_print()` format.
    public func formatSummary() -> String {
        guard layerCount > 0 else { return "" }
        let n = Double(layerCount)
        func fmt(_ v: Double) -> String {
            let avg = v / n
            return String(repeating: " ", count: max(0, 6 - "\(Int(avg))".count))
                + "\((avg * 1000).rounded() / 1000)"
        }
        return """

        [timing] Per-layer breakdown (avg of \(layerCount) layers, ms):
          deferred_wait:  \(fmt(deferredWait))
          deferred_cpu:   \(fmt(deferredCPU))
          input_norm:     \(fmt(inputNorm))
          cmd1_submit:    \(fmt(cmd1Submit))
          cmd1_wait:      \(fmt(cmd1Wait))
          cpu_attn:       \(fmt(cpuAttention))
          cmd2_encode:    \(fmt(cmd2Encode))
          cmd2_wait:      \(fmt(cmd2Wait))
          routing_cpu:    \(fmt(routingCPU))
          expert_io:      \(fmt(expertIO))
          cmd3_encode:    \(fmt(cmd3Encode))
          total_layer:    \(fmt(totalLayer))
        """
    }

    /// Prints a timing summary to standard output.
    public func printSummary() {
        let summary = formatSummary()
        guard !summary.isEmpty else { return }
        pipelineLogger.info("\(summary, privacy: .public)")
    }
}
