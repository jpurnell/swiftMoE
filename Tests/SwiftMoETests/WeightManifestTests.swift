import Testing
import Foundation
@testable import SwiftMoE

/// Tests for WeightManifest — JSON tensor manifest loader with O(1) lookup.
@Suite("WeightManifest")
struct WeightManifestTests {

    /// Creates a temporary JSON manifest fixture.
    static func createManifestFixture() throws -> String {
        let json = """
        {
            "tensors": {
                "model.embed_tokens.weight": {
                    "offset": 0,
                    "size": 4071628800,
                    "shape": [248320, 4096],
                    "dtype": "BF16"
                },
                "model.layers.0.input_layernorm.weight": {
                    "offset": 4071628800,
                    "size": 16384,
                    "shape": [4096],
                    "dtype": "F32"
                },
                "model.layers.0.self_attn.q_proj.weight": {
                    "offset": 4071645184,
                    "size": 2097152,
                    "shape": [16384, 512],
                    "dtype": "U32"
                }
            }
        }
        """
        let tempDir = FileManager.default.temporaryDirectory
        let path = tempDir.appendingPathComponent("test_manifest_\(UUID().uuidString).json").path
        try json.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    @Test("Loads manifest and reports correct tensor count")
    func loadManifest() throws {
        let path = try Self.createManifestFixture()
        defer { unlink(path) }

        let manifest = try WeightManifest(path: path)
        #expect(manifest.count == 3)
    }

    @Test("Looks up tensor by name with correct offset and size")
    func lookupByName() throws {
        let path = try Self.createManifestFixture()
        defer { unlink(path) }

        let manifest = try WeightManifest(path: path)
        let tensor = manifest["model.embed_tokens.weight"]

        #expect(tensor != nil)
        #expect(tensor?.offset == 0)
        #expect(tensor?.size == 4_071_628_800)
        #expect(tensor?.shape == [248320, 4096])
        #expect(tensor?.dtype == "BF16")
    }

    @Test("Returns nil for nonexistent tensor name")
    func missingTensor() throws {
        let path = try Self.createManifestFixture()
        defer { unlink(path) }

        let manifest = try WeightManifest(path: path)
        #expect(manifest["nonexistent.tensor"] == nil)
    }

    @Test("Throws for invalid JSON file")
    func invalidJSON() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let path = tempDir.appendingPathComponent("bad_manifest_\(UUID().uuidString).json").path
        try "not json".write(toFile: path, atomically: true, encoding: .utf8)
        defer { unlink(path) }

        #expect(throws: FlashMoEError.self) {
            _ = try WeightManifest(path: path)
        }
    }

    @Test("Throws for nonexistent file")
    func fileNotFound() {
        #expect(throws: FlashMoEError.self) {
            _ = try WeightManifest(path: "/nonexistent/manifest.json")
        }
    }
}
