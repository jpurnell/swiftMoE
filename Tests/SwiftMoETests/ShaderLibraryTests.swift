import Testing
import Foundation
import Metal
@testable import SwiftMoE

/// Tests for ShaderLibrary — runtime shader compilation and pipeline state creation.
@Suite("ShaderLibrary")
struct ShaderLibraryTests {

    /// The real `shaders.metal` in the repo, or nil when it isn't checked out.
    ///
    /// Resolved against the repo root derived from `#filePath` — not the process
    /// working directory — so the result does not depend on where tests are run.
    static var shaderURL: URL? {
        TestPaths.existingRepoFile("metal_infer/shaders.metal")
    }

    @Test("Compiles all required pipeline states from shaders.metal")
    func compilesRequiredPipelines() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        // Skip test if shader file isn't available (CI without repo checkout)
        guard let shaderURL = Self.shaderURL else { return }

        let library = try ShaderLibrary(device: device, shaderPath: shaderURL.path)

        // Required pipelines — verify they compiled with correct thread config
        #expect(library.matvecV3.maxTotalThreadsPerThreadgroup > 0)
        #expect(library.matvecFast.maxTotalThreadsPerThreadgroup > 0)
        #expect(library.rmsNormSum.maxTotalThreadsPerThreadgroup > 0)
        #expect(library.rmsNormApply.maxTotalThreadsPerThreadgroup > 0)
        #expect(library.swiglu.maxTotalThreadsPerThreadgroup > 0)
        #expect(library.moeCombineResidual.maxTotalThreadsPerThreadgroup > 0)
    }

    @Test("Compiles optional delta-net pipelines")
    func compilesDeltaNetPipelines() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        guard let shaderURL = Self.shaderURL else { return }

        let library = try ShaderLibrary(device: device, shaderPath: shaderURL.path)

        // Optional but expected on M-series hardware
        let deltaNet = try #require(library.deltaNetStep, "delta_net_step should compile on Apple Silicon")
        #expect(deltaNet.maxTotalThreadsPerThreadgroup > 0)
        let conv1d = try #require(library.conv1dStep, "conv1d_step should compile on Apple Silicon")
        #expect(conv1d.maxTotalThreadsPerThreadgroup > 0)
    }

    @Test("Throws for nonexistent shader path")
    func nonexistentPath() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        #expect(throws: FlashMoEError.self) {
            _ = try ShaderLibrary(device: device, shaderPath: "/nonexistent/shaders.metal")
        }
    }
}
