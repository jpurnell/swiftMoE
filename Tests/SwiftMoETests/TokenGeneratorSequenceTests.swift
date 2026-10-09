import Foundation
import Testing
@testable import SwiftMoE

/// How long a sequence ``TokenGenerator`` accepts, and when it stops early.
///
/// The KV caches are allocated once and do not grow. A sequence that outruns them used to be
/// generated anyway, each further token attending to a history that had silently stopped
/// being recorded. These tests drive the real pipeline on the tiny synthetic model.
@Suite("TokenGenerator sequence capacity and cancellation",
       .enabled(if: ShaderLibraryTests.shadersAvailable, "requires metal_infer/shaders.metal"))
struct TokenGeneratorSequenceTests {

    /// A generator over the tiny synthetic model, with everything a call to `generate` needs.
    private final class Harness {
        let generator: TokenGenerator
        private let fixtures: SyntheticFixtures.FixturePaths
        private let weightFile: WeightFile
        private let layerWeights: [LayerWeightPointers]
        private let expertFDs: [Int32]

        init(maxSequenceLength: Int) throws {
            let config = ModelConfig.tiny
            fixtures = try SyntheticFixtures.create(
                config: config, numLayers: config.numLayers, numExperts: config.numExperts)
            let shaderURL = try #require(ShaderLibraryTests.shaderURL)
            let context = try MetalContext(config: config, shaderPath: shaderURL.path, use2Bit: false)
            weightFile = try WeightFile(weightsPath: fixtures.weightsPath, manifestPath: fixtures.manifestPath)
            layerWeights = LayerWeightCacheBuilder.build(from: weightFile, config: config)
            expertFDs = fixtures.expertPaths.map { open($0, O_RDONLY) }
            context.setWeights(weightFile.data, size: weightFile.size)
            generator = TokenGenerator(context: context, config: config,
                                       activeExperts: config.numExpertsPerToken,
                                       maxSeqLen: maxSequenceLength)
        }

        deinit {
            expertFDs.forEach { close($0) }
            SyntheticFixtures.cleanup(fixtures)
        }

        /// Runs one generation and returns the tokens it produced.
        ///
        /// - Parameter continueFor: How many times `shouldContinue` answers `true` before it
        ///   starts answering `false`; `nil` never cancels.
        func generate(prompt: [Int], maxTokens: Int, continueFor: Int? = nil) throws -> [Int] {
            var produced: [Int] = []
            var asked = 0
            try generator.generate(
                promptTokens: prompt, maxTokens: maxTokens, weightFile: weightFile,
                expertFDs: expertFDs, layerWeights: layerWeights, use2Bit: false,
                shouldContinue: {
                    asked += 1
                    return continueFor.map { asked <= $0 } ?? true
                },
                onToken: { token in
                    produced.append(token)
                    return true
                })
            return produced
        }
    }

    @Test("The generator reports the capacity it was created with")
    func reportsCapacity() throws {
        let harness = try Harness(maxSequenceLength: 6)
        #expect(harness.generator.maxSequenceLength == 6)
        #expect(harness.generator.sequenceLength == 0)
    }

    @Test("A prompt and completion that exactly fill the capacity are generated in full")
    func exactlyAtCapacity() throws {
        let harness = try Harness(maxSequenceLength: 4)
        let produced = try harness.generate(prompt: [0, 1], maxTokens: 2)
        #expect(produced.count == 2)
        #expect(harness.generator.sequenceLength == 4)
    }

    @Test("One position over the capacity is refused before anything is computed")
    func overCapacityIsRefused() throws {
        let harness = try Harness(maxSequenceLength: 4)
        var produced: [Int] = []
        #expect(throws: FlashMoEError.sequenceCapacityExceeded(capacity: 4, required: 5)) {
            produced = try harness.generate(prompt: [0, 1, 2], maxTokens: 2)
        }
        #expect(produced == [])
        #expect(harness.generator.sequenceLength == 0)
    }

    @Test("A prompt longer than the capacity on its own is refused too")
    func promptAloneOverCapacity() throws {
        let harness = try Harness(maxSequenceLength: 2)
        #expect(throws: FlashMoEError.sequenceCapacityExceeded(capacity: 2, required: 4)) {
            _ = try harness.generate(prompt: [0, 1, 2], maxTokens: 1)
        }
        #expect(harness.generator.sequenceLength == 0)
    }

    @Test("Each generation is a new sequence: nothing of the last one is left in the caches")
    func eachGenerationStartsEmpty() throws {
        let harness = try Harness(maxSequenceLength: 8)
        #expect(try harness.generate(prompt: [0, 1], maxTokens: 2).count == 2)
        #expect(harness.generator.sequenceLength == 4)
        #expect(try harness.generate(prompt: [2], maxTokens: 2).count == 2)
        #expect(harness.generator.sequenceLength == 3)
    }

    @Test("Cancelled during prefill: the rest of the prompt is not processed and nothing is generated")
    func cancelledDuringPrefill() throws {
        let harness = try Harness(maxSequenceLength: 8)
        let produced = try harness.generate(prompt: [0, 1, 2], maxTokens: 3, continueFor: 1)
        #expect(produced == [])
        #expect(harness.generator.sequenceLength == 1)
    }

    @Test("Cancelled during generation: it stops at the next token")
    func cancelledDuringGeneration() throws {
        let harness = try Harness(maxSequenceLength: 8)
        // One answer for the single prompt token, one for the first generated token.
        let produced = try harness.generate(prompt: [0], maxTokens: 3, continueFor: 2)
        #expect(produced.count == 1)
        #expect(harness.generator.sequenceLength == 2)
    }

    @Test("A negative token count generates nothing rather than trapping")
    func negativeTokenCount() throws {
        let harness = try Harness(maxSequenceLength: 8)
        #expect(try harness.generate(prompt: [0], maxTokens: -1) == [])
        #expect(harness.generator.sequenceLength == 1)
    }
}
