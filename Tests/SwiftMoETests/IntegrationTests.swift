import Testing
import Foundation
import Metal
@testable import SwiftMoE

/// End-to-end integration tests using ModelConfig.tiny and synthetic fixtures.
///
/// These tests prove the full pipeline executes without crashes:
/// weight loading → embedding → layer forward → expert I/O → lm_head → argmax.
/// Output is nonsensical (zero-initialized weights), but the data flow is real.
@Suite("Integration")
struct IntegrationTests {

    @Test("Synthetic fixtures create valid weight file and manifest")
    func syntheticFixtures() throws {
        let config = ModelConfig.tiny
        let fixtures = try SyntheticFixtures.create(
            config: config, numLayers: config.numLayers, numExperts: config.numExperts
        )
        defer { SyntheticFixtures.cleanup(fixtures) }

        // Weight file should exist and be non-empty
        let weightsData = FileManager.default.contents(atPath: fixtures.weightsPath)
        #expect(weightsData != nil, "Weight file should exist")
        #expect((weightsData?.count ?? 0) > 0, "Weight file should be non-empty")

        // Manifest should be loadable
        let manifest = try WeightManifest(path: fixtures.manifestPath)
        #expect(manifest.count > 0, "Manifest should have tensors")

        // Key tensors should be present
        #expect(manifest["model.embed_tokens.weight"] != nil, "Embedding tensor should exist")
        #expect(manifest["model.norm.weight"] != nil, "Final norm tensor should exist")
        #expect(manifest["lm_head.weight"] != nil, "LM head tensor should exist")
        #expect(manifest["model.layers.0.input_layernorm.weight"] != nil, "Layer 0 norm should exist")

        // Expert files should exist
        #expect(fixtures.expertPaths.count == config.numLayers)
        for path in fixtures.expertPaths {
            #expect(FileManager.default.fileExists(atPath: path), "Expert file should exist: \(path)")
        }
    }

    @Test("WeightFile mmap and tensor lookup with tiny config")
    func weightFileLookup() throws {
        let config = ModelConfig.tiny
        let fixtures = try SyntheticFixtures.create(
            config: config, numLayers: config.numLayers, numExperts: config.numExperts
        )
        defer { SyntheticFixtures.cleanup(fixtures) }

        let wf = try WeightFile(weightsPath: fixtures.weightsPath, manifestPath: fixtures.manifestPath)

        // Lookup embedding tensor
        let embedW: UnsafePointer<UInt32>? = wf.tensorPointer(name: "model.embed_tokens.weight")
        #expect(embedW != nil, "Embedding weight pointer should be valid")

        // Lookup a layer norm
        let normW: UnsafePointer<UInt16>? = wf.tensorPointer(name: "model.layers.0.input_layernorm.weight")
        #expect(normW != nil, "Layer 0 input norm should be found")
    }

    @Test("LayerWeightCache builds from tiny config")
    func layerWeightCache() throws {
        let config = ModelConfig.tiny
        let fixtures = try SyntheticFixtures.create(
            config: config, numLayers: config.numLayers, numExperts: config.numExperts
        )
        defer { SyntheticFixtures.cleanup(fixtures) }

        let wf = try WeightFile(weightsPath: fixtures.weightsPath, manifestPath: fixtures.manifestPath)
        let caches = LayerWeightCacheBuilder.build(from: wf, config: config)

        #expect(caches.count == config.numLayers, "Should have \(config.numLayers) layer caches")

        // Layer 0 is linear attention (interval=2, so layer 1 is full)
        #expect(caches[0].inputNormW != nil, "Layer 0 should have input norm")
        #expect(caches[0].gateW != nil, "Layer 0 should have routing gate")

        // Check attention type matches config
        for i in 0..<config.numLayers {
            if config.isFullAttention(layer: i) {
                #expect(caches[i].qW != nil, "Full attention layer \(i) should have Q projection")
            } else {
                #expect(caches[i].qkvW != nil, "Linear attention layer \(i) should have QKV projection")
            }
        }
    }

    @Test("Embedding lookup produces output with tiny config")
    func embeddingLookup() throws {
        let config = ModelConfig.tiny
        let fixtures = try SyntheticFixtures.create(
            config: config, numLayers: config.numLayers, numExperts: config.numExperts
        )
        defer { SyntheticFixtures.cleanup(fixtures) }

        let wf = try WeightFile(weightsPath: fixtures.weightsPath, manifestPath: fixtures.manifestPath)
        var output = [Float](repeating: -999, count: config.hiddenDim)

        output.withUnsafeMutableBufferPointer { buf in
            Embedding.lookup(weightFile: wf, tokenID: 0, config: config, output: buf.baseAddress!)
        }

        // With zero-initialized weights, all nibbles are 0, scale=0, bias=0 → output = 0
        #expect(output[0] == 0.0, "Zero weights should produce zero embedding")
    }

    @Test("Argmax works with tiny vocab")
    func argmaxTiny() {
        let config = ModelConfig.tiny
        var logits = [Float](repeating: 0, count: config.vocabSize)
        logits[5] = 1.0  // Make token 5 the winner

        let result = logits.withUnsafeBufferPointer { buf in
            Embedding.argmax(logits: buf.baseAddress!, vocabSize: config.vocabSize)
        }
        #expect(result == 5)
    }

    @Test("CPU dequant matvec (lm_head path) executes with tiny config")
    func lmHeadTiny() throws {
        let config = ModelConfig.tiny
        let fixtures = try SyntheticFixtures.create(
            config: config, numLayers: config.numLayers, numExperts: config.numExperts
        )
        defer { SyntheticFixtures.cleanup(fixtures) }

        let wf = try WeightFile(weightsPath: fixtures.weightsPath, manifestPath: fixtures.manifestPath)
        let hidden = [Float](repeating: 1.0, count: config.hiddenDim)
        var logits = [Float](repeating: -999, count: config.vocabSize)

        hidden.withUnsafeBufferPointer { hBuf in
            logits.withUnsafeMutableBufferPointer { lBuf in
                Embedding.lmHead(
                    weightFile: wf,
                    hidden: hBuf.baseAddress!,
                    config: config,
                    logits: lBuf.baseAddress!
                )
            }
        }

        // With zero weights: all logits should be 0
        #expect(logits[0] == 0.0, "Zero weights should produce zero logits")
        #expect(logits[config.vocabSize - 1] == 0.0)
    }

    @Test("TokenGenerator.generate() runs end-to-end with tiny config")
    func generateEndToEnd() throws {
        let config = ModelConfig.tiny
        let fixtures = try SyntheticFixtures.create(
            config: config, numLayers: config.numLayers, numExperts: config.numExperts
        )
        defer { SyntheticFixtures.cleanup(fixtures) }

        let path = ShaderLibraryTests.shaderPath
        guard FileManager.default.fileExists(atPath: path) else { return }

        let ctx = try MetalContext(config: config, shaderPath: path, use2Bit: false)
        let wf = try WeightFile(weightsPath: fixtures.weightsPath, manifestPath: fixtures.manifestPath)
        let layerWeights = LayerWeightCacheBuilder.build(from: wf, config: config)

        // Open expert file descriptors
        var expertFDs: [Int32] = []
        for expertPath in fixtures.expertPaths {
            let fd = open(expertPath, O_RDONLY)
            #expect(fd >= 0, "Expert file should open: \(expertPath)")
            expertFDs.append(fd)
        }
        defer { expertFDs.forEach { close($0) } }

        // Wrap weights for GPU (even though we'll fall back to CPU for tiny config)
        ctx.setWeights(wf.data, size: wf.size)

        let gen = TokenGenerator(context: ctx, config: config, activeExperts: config.numExpertsPerToken)

        var generatedTokens: [Int] = []
        gen.generate(
            promptTokens: [0],  // Single token prompt
            maxTokens: 3,       // Generate up to 3 tokens
            weightFile: wf,
            expertFDs: expertFDs,
            layerWeights: layerWeights,
            use2Bit: false,
            onToken: { token in
                generatedTokens.append(token)
                return true  // Keep generating
            }
        )

        // With zero weights, the pipeline should execute without crashing.
        // All logits are zero → argmax returns 0, which is not EOS for tiny config
        // (eosToken1=30, eosToken2=31), so we should get 3 tokens.
        #expect(generatedTokens.count > 0,
                "Should generate at least one token (pipeline executed without crash)")
    }

    @Test("Full pipeline: embed → lm_head → argmax with tiny config")
    func fullCPUPipeline() throws {
        let config = ModelConfig.tiny
        let fixtures = try SyntheticFixtures.create(
            config: config, numLayers: config.numLayers, numExperts: config.numExperts
        )
        defer { SyntheticFixtures.cleanup(fixtures) }

        let wf = try WeightFile(weightsPath: fixtures.weightsPath, manifestPath: fixtures.manifestPath)

        // Step 1: Embed token 0
        var hidden = [Float](repeating: 0, count: config.hiddenDim)
        hidden.withUnsafeMutableBufferPointer { buf in
            Embedding.lookup(weightFile: wf, tokenID: 0, config: config, output: buf.baseAddress!)
        }

        // Step 2: Final norm (with zero weights, output is zero)
        var normed = [Float](repeating: 0, count: config.hiddenDim)
        if let normW = wf.tensorPointer(name: "model.norm.weight", as: UInt16.self) {
            normed.withUnsafeMutableBufferPointer { nBuf in
                hidden.withUnsafeBufferPointer { hBuf in
                    RMSNorm.apply(input: hBuf.baseAddress!, weights: normW,
                                  output: nBuf.baseAddress!, dim: config.hiddenDim)
                }
            }
        }

        // Step 3: LM head → logits
        var logits = [Float](repeating: 0, count: config.vocabSize)
        normed.withUnsafeBufferPointer { nBuf in
            logits.withUnsafeMutableBufferPointer { lBuf in
                Embedding.lmHead(weightFile: wf, hidden: nBuf.baseAddress!,
                                 config: config, logits: lBuf.baseAddress!)
            }
        }

        // Step 4: Argmax
        let nextToken = logits.withUnsafeBufferPointer { buf in
            Embedding.argmax(logits: buf.baseAddress!, vocabSize: config.vocabSize)
        }

        // With all-zero weights, all logits are 0, argmax returns 0
        #expect(nextToken >= 0 && nextToken < config.vocabSize,
                "Token should be in valid range")
    }
}
