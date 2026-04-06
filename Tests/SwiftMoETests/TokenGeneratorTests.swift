import Testing
import Foundation
@testable import SwiftMoE

@Suite("TokenGenerator")
struct TokenGeneratorTests {

    @Test("Layer type classification matches Qwen3.5 architecture")
    func layerTypes() {
        let config = ModelConfig.qwen397B
        // Full attention every 4th layer: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59
        let fullLayers = (0..<60).filter { TokenGenerator.isFullAttention(layer: $0, config: config) }
        #expect(fullLayers.count == 15, "Should have exactly 15 full attention layers")
        #expect(fullLayers.first == 3, "First full attention layer should be 3")
        #expect(fullLayers.last == 59, "Last full attention layer should be 59")

        let linearLayers = (0..<60).filter { !TokenGenerator.isFullAttention(layer: $0, config: config) }
        #expect(linearLayers.count == 45, "Should have exactly 45 linear attention layers")
    }

    @Test("Full attention index mapping is correct")
    func fullAttentionIndex() {
        let config = ModelConfig.qwen397B
        // Layer 3 → FA index 0, Layer 7 → 1, ..., Layer 59 → 14
        #expect(TokenGenerator.fullAttentionIndex(layer: 3, config: config) == 0)
        #expect(TokenGenerator.fullAttentionIndex(layer: 7, config: config) == 1)
        #expect(TokenGenerator.fullAttentionIndex(layer: 59, config: config) == 14)
    }

    @Test("Linear attention index mapping is correct")
    func linearAttentionIndex() {
        let config = ModelConfig.qwen397B
        // Layer 0 → LA index 0, Layer 1 → 1, Layer 2 → 2, Layer 4 → 3 (skips layer 3)
        #expect(TokenGenerator.linearAttentionIndex(layer: 0, config: config) == 0)
        #expect(TokenGenerator.linearAttentionIndex(layer: 1, config: config) == 1)
        #expect(TokenGenerator.linearAttentionIndex(layer: 2, config: config) == 2)
        #expect(TokenGenerator.linearAttentionIndex(layer: 4, config: config) == 3)  // layer 3 is full attn
    }

    @Test("Initialization creates correct number of caches and states")
    func initialization() throws {
        let path = ShaderLibraryTests.shaderPath
        guard FileManager.default.fileExists(atPath: path) else { return }

        let config = ModelConfig.qwen397B
        let ctx = try MetalContext(config: config, shaderPath: path, use2Bit: false)
        let gen = TokenGenerator(context: ctx, config: config)

        #expect(gen.kvCaches.count == 15, "Should have 15 KV caches (one per full attention layer)")
        #expect(gen.linearStates.count == 45, "Should have 45 linear attention states")
        #expect(gen.activeExperts == 4, "Default K=4")
    }
}
