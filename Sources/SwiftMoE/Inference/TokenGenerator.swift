import Metal
import Foundation

/// Top-level inference engine: embed → 60 layers → norm → lm_head → sample.
///
/// This is the Swift equivalent of the token generation loop in `infer.m:6000+`.
/// It drives the `LayerPipeline` across all 60 transformer layers for each token.
///
/// ## Token Generation Loop
/// ```
/// for each token:
///   1. Embedding lookup → hidden[4096]
///   2. For layer 0..<60:
///      pipeline.forward(layer, hidden, kv/linearState, pos, expertFile)
///   3. pipeline.completeDeferredExperts()  // finalize layer 59
///   4. RMS norm → lm_head matmul → logits[248320]
///   5. Sample next token (top-K, temperature)
/// ```
public final class TokenGenerator {

    /// Metal context with all GPU resources.
    public let context: MetalContext

    /// The model configuration.
    public let config: ModelConfig

    /// The layer pipeline (CMD1→CMD2→CMD3 orchestration).
    public let pipeline: LayerPipeline

    /// KV caches for the 15 full attention layers.
    public var kvCaches: [KVCache]

    /// Linear attention states for the 45 GatedDeltaNet layers.
    public var linearStates: [LinearAttentionState]

    /// Active expert count per token (default K=4).
    public var activeExperts: Int

    /// Hidden state vector [HIDDEN_DIM].
    private var hidden: [Float]

    /// Positions the KV caches were allocated for: the longest sequence, prompt and completion
    /// together, that ``generate(promptTokens:maxTokens:weightFile:expertFDs:layerWeights:use2Bit:shouldContinue:onToken:)``
    /// accepts.
    public let maxSequenceLength: Int

    /// Positions recorded so far in the current sequence: the fullest KV cache's length.
    public var sequenceLength: Int {
        kvCaches.map(\.length).max() ?? 0
    }

    /// Positions the KV caches are allocated for unless a caller asks otherwise: 8192.
    ///
    /// This is `GPU_KV_SEQ` in `infer.m` — the GPU KV buffers are pre-allocated for this many
    /// positions, and ``KVCache`` throws rather than grow past its allocation. It is therefore
    /// the longest sequence, prompt and completion together, that can be generated;
    /// ``HTTPServer/Limits/maxCompletionTokens`` defaults to it.
    public static let defaultMaxSequenceLength = 8192

    /// Creates a token generator with the given Metal context.
    ///
    /// - Parameters:
    ///   - context: Initialized Metal context with compiled shaders.
    ///   - config: Model configuration providing dimensions and layer layout.
    ///   - activeExperts: Number of experts per token (default 4).
    ///   - maxSeqLen: Maximum sequence length for KV caches (default
    ///     ``defaultMaxSequenceLength``).
    public init(context: MetalContext, config: ModelConfig, activeExperts: Int = 4,
                maxSeqLen: Int = TokenGenerator.defaultMaxSequenceLength) {
        self.context = context
        self.config = config
        self.pipeline = LayerPipeline(context: context, config: config)
        self.activeExperts = activeExperts
        self.maxSequenceLength = maxSeqLen

        let kvDim = config.numKVHeads * config.headDim  // 512
        self.kvCaches = (0..<config.numFullAttentionLayers).map { _ in
            KVCache(kvDim: kvDim, maxLength: maxSeqLen)
        }
        self.linearStates = (0..<config.numLinearAttentionLayers).map { _ in
            LinearAttentionState(config: config)
        }
        self.hidden = [Float](repeating: 0, count: config.hiddenDim)
    }

    /// Resets all state for a new generation.
    ///
    /// Call this between conversations or when starting a new prompt.
    public func reset() {
        for i in 0..<kvCaches.count {
            kvCaches[i].reset()
        }
        for i in 0..<linearStates.count {
            linearStates[i].reset()
        }
        context.resetLinearAttentionState()
        pipeline.timing.reset()
    }

    /// Whether a layer uses full attention (vs. linear attention).
    ///
    /// Full attention occurs every 4th layer starting at layer 3:
    /// layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59.
    public static func isFullAttention(layer: Int, config: ModelConfig) -> Bool {
        config.isFullAttention(layer: layer)
    }

    /// Converts a layer index to its full-attention KV cache index (0..<15).
    public static func fullAttentionIndex(layer: Int, config: ModelConfig) -> Int {
        config.fullAttentionIndex(layer: layer)
    }

    /// Converts a layer index to its linear attention state index (0..<45).
    public static func linearAttentionIndex(layer: Int, config: ModelConfig) -> Int {
        config.linearAttentionIndex(layer: layer)
    }

    // MARK: - Inference

    /// Generates tokens from a prompt, calling the callback for each generated token.
    ///
    /// This is the main inference loop matching `infer.m:7018-7081`:
    /// ```
    /// embed → 60 × fused_layer_forward → complete_deferred → norm → lm_head → argmax
    /// ```
    ///
    /// - Parameters:
    ///   - promptTokens: Tokenized prompt (array of token IDs).
    ///   - maxTokens: Maximum number of tokens to generate.
    ///   - weightFile: The mmap'd non-expert weight file.
    ///   - expertFDs: Per-layer expert file descriptors (60 FDs).
    ///   - layerWeights: Pre-computed weight pointers for all 60 layers.
    ///   - use2Bit: Whether experts use 2-bit quantization (default `false`).
    ///   - shouldContinue: Asked before every token, prompt and generated alike. Return `false`
    ///     to abandon the generation — the caller has gone away, say. Prefill produces no
    ///     tokens, so without this a long prompt could not be interrupted at all.
    ///   - onToken: Callback invoked for each generated token. Return `false` to stop.
    /// - Throws: ``FlashMoEError/sequenceCapacityExceeded(capacity:required:)`` when
    ///   `promptTokens.count + maxTokens` is more than ``maxSequenceLength``. It is thrown
    ///   before any token is processed and leaves the generator's state as it was.
    ///
    /// Each call is a new sequence: the KV caches and linear-attention states are reset
    /// first, so nothing of an earlier prompt is attended to by a later one. A `maxTokens`
    /// below zero generates nothing.
    public func generate(
        promptTokens: [Int],
        maxTokens: Int,
        weightFile: WeightFile,
        expertFDs: [Int32],
        layerWeights: [LayerWeightPointers],
        use2Bit: Bool = false,
        shouldContinue: () -> Bool = { true },
        onToken: (Int) -> Bool
    ) throws {
        let completionBudget = max(0, maxTokens)
        let (required, overflow) = promptTokens.count.addingReportingOverflow(completionBudget)
        guard !overflow, required <= maxSequenceLength else {
            throw FlashMoEError.sequenceCapacityExceeded(capacity: maxSequenceLength,
                                                         required: overflow ? Int.max : required)
        }

        reset()
        // An early exit, cancelled or thrown, must not leave a command buffer in flight.
        defer { pipeline.discardDeferredExperts() }

        let hiddenDim = config.hiddenDim
        var logits = [Float](repeating: 0, count: config.vocabSize)
        var normed = [Float](repeating: 0, count: hiddenDim)
        var pos = 0

        let finalNormW = weightFile.tensorPointer(
            name: "model.norm.weight", as: UInt16.self)

        context.setWeights(weightFile.data, size: weightFile.size)

        // ---- Prefill: process all prompt tokens ----
        for (i, tokenID) in promptTokens.enumerated() {
            guard shouldContinue() else { return }
            try hidden.withUnsafeMutableBufferPointer { hiddenBuf in
                guard let hiddenPtr = hiddenBuf.baseAddress else { return }

                Embedding.lookup(weightFile: weightFile, tokenID: tokenID, config: config, output: hiddenPtr)

                try forwardAllLayers(
                    hiddenPtr: hiddenPtr,
                    layerWeights: layerWeights,
                    expertFDs: expertFDs,
                    pos: pos,
                    use2Bit: use2Bit
                )

                if i < promptTokens.count - 1 {
                    pipeline.discardDeferredExperts()
                } else {
                    pipeline.completeDeferredExperts(hidden: hiddenPtr)
                }
            }
            pos += 1
        }

        // ---- Generation: produce new tokens ----
        for _ in 0..<completionBudget {
            guard shouldContinue() else { return }
            try hidden.withUnsafeMutableBufferPointer { hiddenBuf in
                guard let hiddenPtr = hiddenBuf.baseAddress else { return }

                try forwardAllLayers(
                    hiddenPtr: hiddenPtr,
                    layerWeights: layerWeights,
                    expertFDs: expertFDs,
                    pos: pos,
                    use2Bit: use2Bit
                )

                pipeline.completeDeferredExperts(hidden: hiddenPtr)

                if let normW = finalNormW {
                    normed.withUnsafeMutableBufferPointer { normedBuf in
                        guard let normedPtr = normedBuf.baseAddress else { return }
                        RMSNorm.apply(
                            input: hiddenPtr,
                            weights: normW,
                            output: normedPtr,
                            dim: hiddenDim
                        )

                        logits.withUnsafeMutableBufferPointer { logitsBuf in
                            guard let logitsPtr = logitsBuf.baseAddress else { return }
                            Embedding.lmHead(
                                weightFile: weightFile,
                                hidden: normedPtr,
                                config: config,
                                logits: logitsPtr
                            )
                        }
                    }
                }
            }

            let nextToken = logits.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return 0 }
                return Embedding.argmax(logits: base, vocabSize: config.vocabSize)
            }

            if nextToken == config.eosToken1 || nextToken == config.eosToken2 {
                break
            }

            if !onToken(nextToken) {
                break
            }

            hidden.withUnsafeMutableBufferPointer { hiddenBuf in
                guard let hiddenPtr = hiddenBuf.baseAddress else { return }
                Embedding.lookup(weightFile: weightFile, tokenID: nextToken,
                                config: config, output: hiddenPtr)
            }
            pos += 1
        }
    }

    // MARK: - Private

    private func forwardAllLayers(
        hiddenPtr: UnsafeMutablePointer<Float>,
        layerWeights: [LayerWeightPointers],
        expertFDs: [Int32],
        pos: Int,
        use2Bit: Bool
    ) throws {
        for layer in 0..<config.numLayers {
            let isFull = config.isFullAttention(layer: layer)
            var kv: KVCache? = isFull ? kvCaches[config.fullAttentionIndex(layer: layer)] : nil
            var ls: LinearAttentionState? = isFull ? nil : linearStates[config.linearAttentionIndex(layer: layer)]

            try pipeline.forward(
                layerIndex: layer,
                hidden: hiddenPtr,
                weights: layerWeights[layer],
                kvCache: &kv,
                linearState: &ls,
                position: pos,
                K: activeExperts,
                expertFD: expertFDs[layer],
                use2Bit: use2Bit,
                layerWeights: layerWeights
            )

            if isFull, let updatedKV = kv {
                kvCaches[config.fullAttentionIndex(layer: layer)] = updatedKV
            }
            if !isFull, let updatedLS = ls {
                linearStates[config.linearAttentionIndex(layer: layer)] = updatedLS
            }
        }
    }
}
