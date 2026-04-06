import Foundation
@testable import SwiftMoE

/// Generates synthetic model weight files for integration testing.
///
/// Creates a minimal `model_weights.bin` + `model_weights.json` with the right
/// tensor names, shapes, and formats for the given model configuration.
/// Weights are zero-initialized (output will be nonsensical, but pipeline executes).
enum SyntheticFixtures {

    struct FixturePaths {
        let weightsPath: String
        let manifestPath: String
        let expertPaths: [String]  // one per layer (only layer 0 for testing)
        let tempDir: String
    }

    /// Creates synthetic model files for a single-layer integration test.
    ///
    /// Only generates tensors for layer 0 + embedding + final norm + lm_head.
    /// Expert files are sized for 4-bit experts with K=4 experts each.
    static func create(config: ModelConfig = .qwen397B, numLayers: Int = 1, numExperts: Int = 4) throws -> FixturePaths {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flash_moe_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        let hiddenDim = config.hiddenDim  // 4096
        let groupSize = config.groupSize  // 64
        let vocabSize = config.vocabSize
        let numGroups = hiddenDim / groupSize  // 64
        let packedCols = hiddenDim / 8  // 512

        // Build manifest and binary
        var tensors: [String: [String: Any]] = [:]
        var binaryData = Data()

        func addTensor(name: String, shape: [Int], dtype: String, elementSize: Int) {
            let numElements = shape.reduce(1, *)
            let size = numElements * elementSize
            let offset = binaryData.count

            // Align to 64 bytes (matching extract_weights.py)
            let padding = (64 - (offset % 64)) % 64
            binaryData.append(Data(repeating: 0, count: padding))
            let alignedOffset = binaryData.count

            // Zero-initialized data
            binaryData.append(Data(repeating: 0, count: size))

            tensors[name] = [
                "offset": alignedOffset,
                "size": size,
                "shape": shape,
                "dtype": dtype
            ]
        }

        // Embedding table (tiny vocab)
        addTensor(name: "model.embed_tokens.weight", shape: [vocabSize, packedCols], dtype: "U32", elementSize: 4)
        addTensor(name: "model.embed_tokens.scales", shape: [vocabSize, numGroups], dtype: "BF16", elementSize: 2)
        addTensor(name: "model.embed_tokens.biases", shape: [vocabSize, numGroups], dtype: "BF16", elementSize: 2)

        // Per-layer tensors
        for i in 0..<numLayers {
            let prefix = "model.layers.\(i)"
            let isFull = (i + 1) % config.fullAttentionInterval == 0

            // Layer norms [4096] as F32 stored as BF16
            addTensor(name: "\(prefix).input_layernorm.weight", shape: [hiddenDim], dtype: "BF16", elementSize: 2)
            addTensor(name: "\(prefix).post_attention_layernorm.weight", shape: [hiddenDim], dtype: "BF16", elementSize: 2)

            let attn = "\(prefix).self_attn"
            if isFull {
                // Full attention (Q,K,V,O projections)
                let qDim = config.numAttentionHeads * config.headDim * 2  // 16384
                let kvDim = config.numKVHeads * config.headDim  // 512
                for (proj, outDim) in [("q_proj", qDim), ("k_proj", kvDim), ("v_proj", kvDim), ("o_proj", hiddenDim)] {
                    let inDim = proj == "o_proj" ? config.numAttentionHeads * config.headDim : hiddenDim
                    addTensor(name: "\(attn).\(proj).weight", shape: [outDim, inDim / 8], dtype: "U32", elementSize: 4)
                    addTensor(name: "\(attn).\(proj).scales", shape: [outDim, inDim / groupSize], dtype: "BF16", elementSize: 2)
                    addTensor(name: "\(attn).\(proj).biases", shape: [outDim, inDim / groupSize], dtype: "BF16", elementSize: 2)
                }
                addTensor(name: "\(attn).q_norm.weight", shape: [config.headDim], dtype: "BF16", elementSize: 2)
                addTensor(name: "\(attn).k_norm.weight", shape: [config.headDim], dtype: "BF16", elementSize: 2)
            } else {
                // Linear attention (QKV, Z, Beta, Alpha, conv, out_proj)
                let linearConvDim = config.linearNumKHeads * config.linearKeyDim * 2
                    + config.linearNumVHeads * config.linearValueDim  // 12288
                let zDim = config.linearNumVHeads * config.linearValueDim  // 8192
                let headsDim = config.linearNumVHeads  // 64

                addTensor(name: "\(attn).qkv_proj.weight", shape: [linearConvDim, packedCols], dtype: "U32", elementSize: 4)
                addTensor(name: "\(attn).qkv_proj.scales", shape: [linearConvDim, numGroups], dtype: "BF16", elementSize: 2)
                addTensor(name: "\(attn).qkv_proj.biases", shape: [linearConvDim, numGroups], dtype: "BF16", elementSize: 2)
                addTensor(name: "\(attn).z_proj.weight", shape: [zDim, packedCols], dtype: "U32", elementSize: 4)
                addTensor(name: "\(attn).z_proj.scales", shape: [zDim, numGroups], dtype: "BF16", elementSize: 2)
                addTensor(name: "\(attn).z_proj.biases", shape: [zDim, numGroups], dtype: "BF16", elementSize: 2)
                for proj in ["beta_proj", "alpha_proj"] {
                    addTensor(name: "\(attn).\(proj).weight", shape: [headsDim, packedCols], dtype: "U32", elementSize: 4)
                    addTensor(name: "\(attn).\(proj).scales", shape: [headsDim, numGroups], dtype: "BF16", elementSize: 2)
                    addTensor(name: "\(attn).\(proj).biases", shape: [headsDim, numGroups], dtype: "BF16", elementSize: 2)
                }
                addTensor(name: "\(attn).conv1d.weight", shape: [linearConvDim * config.convKernelSize], dtype: "BF16", elementSize: 2)
                addTensor(name: "\(attn).a_log", shape: [headsDim], dtype: "F32", elementSize: 4)
                addTensor(name: "\(attn).dt_bias", shape: [headsDim], dtype: "BF16", elementSize: 2)
                addTensor(name: "\(attn).g_norm.weight", shape: [config.linearValueDim], dtype: "BF16", elementSize: 2)
                addTensor(name: "\(attn).out_proj.weight", shape: [hiddenDim, zDim / 8], dtype: "U32", elementSize: 4)
                addTensor(name: "\(attn).out_proj.scales", shape: [hiddenDim, zDim / groupSize], dtype: "BF16", elementSize: 2)
                addTensor(name: "\(attn).out_proj.biases", shape: [hiddenDim, zDim / groupSize], dtype: "BF16", elementSize: 2)
            }

            // MoE routing gate
            let moe = "\(prefix).mlp"
            addTensor(name: "\(moe).gate.weight", shape: [config.numExperts, packedCols], dtype: "U32", elementSize: 4)
            addTensor(name: "\(moe).gate.scales", shape: [config.numExperts, numGroups], dtype: "BF16", elementSize: 2)
            addTensor(name: "\(moe).gate.biases", shape: [config.numExperts, numGroups], dtype: "BF16", elementSize: 2)

            // Shared expert
            let shared = "\(moe).shared_expert"
            let intermediate = config.sharedIntermediate
            for (proj, outDim, inDim) in [
                ("gate_proj", intermediate, hiddenDim),
                ("up_proj", intermediate, hiddenDim),
                ("down_proj", hiddenDim, intermediate)
            ] {
                addTensor(name: "\(shared).\(proj).weight", shape: [outDim, inDim / 8], dtype: "U32", elementSize: 4)
                addTensor(name: "\(shared).\(proj).scales", shape: [outDim, inDim / groupSize], dtype: "BF16", elementSize: 2)
                addTensor(name: "\(shared).\(proj).biases", shape: [outDim, inDim / groupSize], dtype: "BF16", elementSize: 2)
            }

            // Shared expert gate
            addTensor(name: "\(moe).shared_expert_gate.weight", shape: [1, packedCols], dtype: "U32", elementSize: 4)
            addTensor(name: "\(moe).shared_expert_gate.scales", shape: [1, numGroups], dtype: "BF16", elementSize: 2)
            addTensor(name: "\(moe).shared_expert_gate.biases", shape: [1, numGroups], dtype: "BF16", elementSize: 2)
        }

        // Final norm
        addTensor(name: "model.norm.weight", shape: [hiddenDim], dtype: "BF16", elementSize: 2)

        // LM head (tiny vocab)
        addTensor(name: "lm_head.weight", shape: [vocabSize, packedCols], dtype: "U32", elementSize: 4)
        addTensor(name: "lm_head.scales", shape: [vocabSize, numGroups], dtype: "BF16", elementSize: 2)
        addTensor(name: "lm_head.biases", shape: [vocabSize, numGroups], dtype: "BF16", elementSize: 2)

        // Write binary
        let weightsPath = tempDir.appendingPathComponent("model_weights.bin").path
        try binaryData.write(to: URL(fileURLWithPath: weightsPath))

        // Write manifest
        let manifest: [String: Any] = ["tensors": tensors]
        let jsonData = try JSONSerialization.data(withJSONObject: manifest, options: .prettyPrinted)
        let manifestPath = tempDir.appendingPathComponent("model_weights.json").path
        try jsonData.write(to: URL(fileURLWithPath: manifestPath))

        // Create expert files (one per layer)
        var expertPaths: [String] = []
        let expertSize = config.expertSize4Bit
        for i in 0..<numLayers {
            let path = tempDir.appendingPathComponent("layer_\(String(format: "%02d", i)).bin").path
            let fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0o644)
            guard fd >= 0 else { continue }
            // Write enough experts (numExperts × expertSize bytes)
            let zeros = Data(repeating: 0, count: numExperts * expertSize)
            zeros.withUnsafeBytes { ptr in
                _ = Darwin.write(fd, ptr.baseAddress!, zeros.count)
            }
            close(fd)
            expertPaths.append(path)
        }

        return FixturePaths(
            weightsPath: weightsPath,
            manifestPath: manifestPath,
            expertPaths: expertPaths,
            tempDir: tempDir.path
        )
    }

    /// Removes all synthetic fixture files.
    static func cleanup(_ paths: FixturePaths) {
        try? FileManager.default.removeItem(atPath: paths.tempDir)
    }
}
