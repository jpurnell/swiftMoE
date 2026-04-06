import Testing
import Foundation
import Metal
@testable import SwiftMoE

/// Tests for ShaderLibrary — runtime shader compilation and pipeline state creation.
@Suite("ShaderLibrary")
struct ShaderLibraryTests {

    /// Path to the real shaders.metal file in the repo.
    static var shaderPath: String {
        // Walk up from the test bundle to find metal_infer/shaders.metal
        let candidates = [
            "metal_infer/shaders.metal",
            "shaders.metal",
            "../metal_infer/shaders.metal",
        ]
        for candidate in candidates {
            if FileManager.default.fileExists(atPath: candidate) {
                return candidate
            }
        }
        // Try from the repo root based on known structure
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // FlashMoETests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // repo root
        let fullPath = repoRoot.appendingPathComponent("metal_infer/shaders.metal").path
        return fullPath
    }

    @Test("Compiles all required pipeline states from shaders.metal")
    func compilesRequiredPipelines() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        let path = Self.shaderPath
        guard FileManager.default.fileExists(atPath: path) else {
            // Skip test if shader file isn't available (CI without repo checkout)
            return
        }

        let library = try ShaderLibrary(device: device, shaderPath: path)

        // Required pipelines — must be non-nil
        #expect(library.matvecV3 != nil, "matvec_v3 pipeline required")
        #expect(library.matvecFast != nil, "matvec_fast pipeline required")
        #expect(library.rmsNormSum != nil, "rms_norm_sum pipeline required")
        #expect(library.rmsNormApply != nil, "rms_norm_apply pipeline required")
        #expect(library.swiglu != nil, "swiglu pipeline required")
        #expect(library.moeCombineResidual != nil, "moe_combine_residual pipeline required")
    }

    @Test("Compiles optional delta-net pipelines")
    func compilesDeltaNetPipelines() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        let path = Self.shaderPath
        guard FileManager.default.fileExists(atPath: path) else { return }

        let library = try ShaderLibrary(device: device, shaderPath: path)

        // Optional but expected on M-series hardware
        #expect(library.deltaNetStep != nil, "delta_net_step should compile on Apple Silicon")
        #expect(library.conv1dStep != nil, "conv1d_step should compile on Apple Silicon")
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
