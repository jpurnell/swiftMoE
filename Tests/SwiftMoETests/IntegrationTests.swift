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
        let weightsData = try #require(
            FileManager.default.contents(atPath: fixtures.weightsPath),
            "Weight file should exist"
        )
        #expect(!weightsData.isEmpty, "Weight file should be non-empty")

        // Manifest should be loadable
        let manifest = try WeightManifest(path: fixtures.manifestPath)
        #expect(manifest.count == 74, "Manifest should have 74 tensors for tiny config (2 layers)")

        // Key tensors should be present with valid offsets
        let embedTensor = try #require(manifest["model.embed_tokens.weight"], "Embedding tensor should exist")
        #expect(embedTensor.size == 1024, "Embedding tensor: vocabSize(32) * packedCols(8) * 4 = 1024")
        let normTensor = try #require(manifest["model.norm.weight"], "Final norm tensor should exist")
        #expect(normTensor.size == 128, "Final norm: hiddenDim(64) * 2 = 128")
        let lmHeadTensor = try #require(manifest["lm_head.weight"], "LM head tensor should exist")
        #expect(lmHeadTensor.size == 1024, "LM head: vocabSize(32) * packedCols(8) * 4 = 1024")
        let layer0Norm = try #require(manifest["model.layers.0.input_layernorm.weight"], "Layer 0 norm should exist")
        #expect(layer0Norm.size == 128, "Layer norm: hiddenDim(64) * 2 = 128")

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

        // Lookup embedding tensor — verify pointer is valid by reading first element
        let embedW: UnsafePointer<UInt32> = try #require(
            wf.tensorPointer(name: "model.embed_tokens.weight"),
            "Embedding weight pointer should be valid"
        )
        // Synthetic fixtures zero-fill, so first element should be 0
        #expect(embedW.pointee == 0, "Synthetic embedding data should be zero-initialized")

        // Lookup a layer norm — verify pointer is valid
        let normW: UnsafePointer<UInt16> = try #require(
            wf.tensorPointer(name: "model.layers.0.input_layernorm.weight"),
            "Layer 0 input norm should be found"
        )
        #expect(normW.pointee == 0, "Synthetic norm data should be zero-initialized")
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
        let inputNormW = try #require(caches[0].inputNormW, "Layer 0 should have input norm")
        #expect(inputNormW.pointee == 0, "Synthetic norm weights should be zero-initialized")
        let gateW = try #require(caches[0].gateW, "Layer 0 should have routing gate")
        #expect(gateW.pointee == 0, "Synthetic gate weights should be zero-initialized")

        // Check attention type matches config
        for i in 0..<config.numLayers {
            if config.isFullAttention(layer: i) {
                let qW = try #require(caches[i].qW, "Full attention layer \(i) should have Q projection")
                #expect(qW.pointee == 0)
            } else {
                let qkvW = try #require(caches[i].qkvW, "Linear attention layer \(i) should have QKV projection")
                #expect(qkvW.pointee == 0)
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
            guard let base = buf.baseAddress else { return }
            Embedding.lookup(weightFile: wf, tokenID: 0, config: config, output: base)
        }

        #expect(abs(output[0]) < 1e-6, "Zero weights should produce zero embedding")
    }

    @Test("Argmax works with tiny vocab")
    func argmaxTiny() {
        let config = ModelConfig.tiny
        var logits = [Float](repeating: 0, count: config.vocabSize)
        logits[5] = 1.0  // Make token 5 the winner

        let result = logits.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return 0 }
            return Embedding.argmax(logits: base, vocabSize: config.vocabSize)
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
            guard let hBase = hBuf.baseAddress else { return }
            logits.withUnsafeMutableBufferPointer { lBuf in
                guard let lBase = lBuf.baseAddress else { return }
                Embedding.lmHead(
                    weightFile: wf,
                    hidden: hBase,
                    config: config,
                    logits: lBase
                )
            }
        }

        #expect(abs(logits[0]) < 1e-6, "Zero weights should produce zero logits")
        #expect(abs(logits[config.vocabSize - 1]) < 1e-6)
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
        #expect(generatedTokens.count == 3,
                "Should generate exactly 3 tokens (zero weights → token 0, not EOS)")
    }

    @Test("Full pipeline: embed → lm_head → argmax with tiny config")
    func fullCPUPipeline() throws {
        let config = ModelConfig.tiny
        let fixtures = try SyntheticFixtures.create(
            config: config, numLayers: config.numLayers, numExperts: config.numExperts
        )
        defer { SyntheticFixtures.cleanup(fixtures) }

        let wf = try WeightFile(weightsPath: fixtures.weightsPath, manifestPath: fixtures.manifestPath)

        var hidden = [Float](repeating: 0, count: config.hiddenDim)
        hidden.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            Embedding.lookup(weightFile: wf, tokenID: 0, config: config, output: base)
        }

        var normed = [Float](repeating: 0, count: config.hiddenDim)
        if let normW = wf.tensorPointer(name: "model.norm.weight", as: UInt16.self) {
            normed.withUnsafeMutableBufferPointer { nBuf in
                guard let nBase = nBuf.baseAddress else { return }
                hidden.withUnsafeBufferPointer { hBuf in
                    guard let hBase = hBuf.baseAddress else { return }
                    RMSNorm.apply(input: hBase, weights: normW,
                                  output: nBase, dim: config.hiddenDim)
                }
            }
        }

        var logits = [Float](repeating: 0, count: config.vocabSize)
        normed.withUnsafeBufferPointer { nBuf in
            guard let nBase = nBuf.baseAddress else { return }
            logits.withUnsafeMutableBufferPointer { lBuf in
                guard let lBase = lBuf.baseAddress else { return }
                Embedding.lmHead(weightFile: wf, hidden: nBase,
                                 config: config, logits: lBase)
            }
        }

        let nextToken = logits.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return 0 }
            return Embedding.argmax(logits: base, vocabSize: config.vocabSize)
        }

        // With all-zero weights, all logits are 0, argmax returns 0
        #expect(nextToken >= 0 && nextToken < config.vocabSize,
                "Token should be in valid range")
    }
}
