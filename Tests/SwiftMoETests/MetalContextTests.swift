import Testing
import Foundation
import Metal
@testable import SwiftMoE

/// Tests for MetalContext — top-level coordinator of all GPU resources.
@Suite("MetalContext")
struct MetalContextTests {

    @Test("Initializes with device, queue, and all buffer groups",
          .enabled(if: ShaderLibraryTests.shadersAvailable, "requires metal_infer/shaders.metal"))
    func initialization() throws {
        let shaderURL = try #require(ShaderLibraryTests.shaderURL)

        let ctx = try MetalContext(config: .qwen397B, shaderPath: shaderURL.path, use2Bit: false)

        #expect(ctx.device.name.isEmpty == false, "Device should have a name")
        #expect(ctx.projections.input.length == 32768,
                "Input buffer: max(64*128, 32*256) * 4 = 32768")
        #expect(ctx.experts.dataA.count == ExpertBuffers.maxK)
        #expect(ctx.combine.residual.length == ModelConfig.qwen397B.hiddenDim * MemoryLayout<Float>.size)
    }

    @Test("Weight buffer wrapping works with synthetic data",
          .enabled(if: ShaderLibraryTests.shadersAvailable, "requires metal_infer/shaders.metal"))
    func setWeights() throws {
        let shaderURL = try #require(ShaderLibraryTests.shaderURL)

        let ctx = try MetalContext(config: .qwen397B, shaderPath: shaderURL.path, use2Bit: false)
        #expect(ctx.weightBuffer == nil, "No weights set yet")

        // Allocate a page-aligned buffer to simulate mmap'd weights
        let size = 65536
        var ptr: UnsafeMutableRawPointer?
        let allocation = posix_memalign(&ptr, 16384, size)
        #expect(allocation == 0, "posix_memalign failed with errno \(allocation)")
        // A failed allocation is a broken fixture, not a reason to skip: returning
        // here reported a pass without ever calling setWeights.
        let aligned = try #require(ptr)
        defer { free(aligned) }

        ctx.setWeights(aligned, size: size)
        let weightBuf = try #require(ctx.weightBuffer, "Weight buffer should be set")
        #expect(weightBuf.length == size, "Weight buffer length should match input size")
    }

    @Test("2-bit mode uses smaller expert buffers",
          .enabled(if: ShaderLibraryTests.shadersAvailable, "requires metal_infer/shaders.metal"))
    func twobitSizing() throws {
        let shaderURL = try #require(ShaderLibraryTests.shaderURL)

        let ctx4 = try MetalContext(config: .qwen397B, shaderPath: shaderURL.path, use2Bit: false)
        let ctx2 = try MetalContext(config: .qwen397B, shaderPath: shaderURL.path, use2Bit: true)

        #expect(ctx2.experts.dataA[0].size < ctx4.experts.dataA[0].size,
                "2-bit should allocate smaller expert data buffers")
    }

    @Test("Reset linear attention state zeros buffers",
          .enabled(if: ShaderLibraryTests.shadersAvailable, "requires metal_infer/shaders.metal"))
    func resetState() throws {
        let shaderURL = try #require(ShaderLibraryTests.shaderURL)

        let ctx = try MetalContext(config: .qwen397B, shaderPath: shaderURL.path, use2Bit: false)

        // Write nonzero data to a delta state buffer
        if let firstState = ctx.linearAttention.deltaState.first {
            let ptr = firstState.contents().assumingMemoryBound(to: Float.self)
            ptr[0] = 42.0
        }

        ctx.resetLinearAttentionState()

        // Verify it's zeroed
        if let firstState = ctx.linearAttention.deltaState.first {
            let ptr = firstState.contents().assumingMemoryBound(to: Float.self)
            #expect(abs(ptr[0]) < 1e-6, "Delta state should be zeroed after reset")
        }
    }
}
