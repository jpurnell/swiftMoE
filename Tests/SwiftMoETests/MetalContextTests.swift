import Testing
import Foundation
import Metal
@testable import SwiftMoE

/// Tests for MetalContext — top-level coordinator of all GPU resources.
@Suite("MetalContext")
struct MetalContextTests {

    @Test("Initializes with device, queue, and all buffer groups")
    func initialization() throws {
        let path = ShaderLibraryTests.shaderPath
        guard FileManager.default.fileExists(atPath: path) else { return }

        let ctx = try MetalContext(config: .qwen397B, shaderPath: path, use2Bit: false)

        #expect(ctx.device.name.isEmpty == false, "Device should have a name")
        #expect(ctx.projections.input.length > 0)
        #expect(ctx.experts.dataA.count == ExpertBuffers.maxK)
        #expect(ctx.combine.residual.length == ModelConfig.qwen397B.hiddenDim * MemoryLayout<Float>.size)
    }

    @Test("Weight buffer wrapping works with synthetic data")
    func setWeights() throws {
        let path = ShaderLibraryTests.shaderPath
        guard FileManager.default.fileExists(atPath: path) else { return }

        let ctx = try MetalContext(config: .qwen397B, shaderPath: path, use2Bit: false)
        #expect(ctx.weightBuffer == nil, "No weights set yet")

        // Allocate a page-aligned buffer to simulate mmap'd weights
        let size = 65536
        var ptr: UnsafeMutableRawPointer?
        posix_memalign(&ptr, 16384, size)
        guard let aligned = ptr else { return }
        defer { free(aligned) }

        ctx.setWeights(aligned, size: size)
        #expect(ctx.weightBuffer != nil, "Weight buffer should be set")
    }

    @Test("2-bit mode uses smaller expert buffers")
    func twobitSizing() throws {
        let path = ShaderLibraryTests.shaderPath
        guard FileManager.default.fileExists(atPath: path) else { return }

        let ctx4 = try MetalContext(config: .qwen397B, shaderPath: path, use2Bit: false)
        let ctx2 = try MetalContext(config: .qwen397B, shaderPath: path, use2Bit: true)

        #expect(ctx2.experts.dataA[0].size < ctx4.experts.dataA[0].size,
                "2-bit should allocate smaller expert data buffers")
    }

    @Test("Reset linear attention state zeros buffers")
    func resetState() throws {
        let path = ShaderLibraryTests.shaderPath
        guard FileManager.default.fileExists(atPath: path) else { return }

        let ctx = try MetalContext(config: .qwen397B, shaderPath: path, use2Bit: false)

        // Write nonzero data to a delta state buffer
        if let firstState = ctx.linearAttention.deltaState.first {
            let ptr = firstState.contents().assumingMemoryBound(to: Float.self)
            ptr[0] = 42.0
        }

        ctx.resetLinearAttentionState()

        // Verify it's zeroed
        if let firstState = ctx.linearAttention.deltaState.first {
            let ptr = firstState.contents().assumingMemoryBound(to: Float.self)
            #expect(ptr[0] == 0.0, "Delta state should be zeroed after reset")
        }
    }
}
