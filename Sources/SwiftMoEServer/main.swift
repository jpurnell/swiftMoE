import Foundation
import os
import SwiftMoE

// ============================================================================
// flash-moe-server — OpenAI-compatible HTTP server with SSE streaming
//
// Usage:
//   flash-moe-server --model <path> [--port 8080] [--k 4] [--2bit] [--timing]
//   flash-moe-server --demo [--port 8080]     # Run with tiny synthetic model
//
// API:
//   POST /v1/chat/completions  (OpenAI chat format, SSE response)
// ============================================================================

private let logger = Logger(subsystem: "com.swiftmoe.server", category: "main")

struct ServerConfig {
    var modelPath: String?
    var port: UInt16 = 8080
    var activeExperts: Int = 4
    var use2Bit: Bool = false
    var timing: Bool = false
    var demo: Bool = false
    var shaderPath: String = "metal_infer/shaders.metal"
}

func parseArgs() -> ServerConfig {
    var config = ServerConfig()
    var i = 1
    let args = CommandLine.arguments
    while i < args.count {
        switch args[i] {
        case "--model": i += 1; if i < args.count { config.modelPath = args[i] }
        case "--port": i += 1; if i < args.count { config.port = UInt16(args[i]) ?? 8080 }
        case "--k": i += 1; if i < args.count { config.activeExperts = Int(args[i]) ?? 4 }
        case "--2bit": config.use2Bit = true
        case "--timing": config.timing = true
        case "--demo": config.demo = true
        case "--shaders": i += 1; if i < args.count { config.shaderPath = args[i] }
        default: break
        }
        i += 1
    }
    return config
}

func main() throws {
    let serverConfig = parseArgs()

    let modelConfig: ModelConfig
    let weightFile: WeightFile
    let expertFDs: [Int32]
    let layerWeights: [LayerWeightPointers]
    var tempDir: String? = nil

    if serverConfig.demo {
        // ---- Demo mode: synthetic tiny model ----
        modelConfig = .tiny
        logger.info("[demo] Using ModelConfig.tiny (hidden=\(modelConfig.hiddenDim, privacy: .public), \(modelConfig.numLayers, privacy: .public) layers, \(modelConfig.numExperts, privacy: .public) experts)")

        // Generate synthetic fixtures
        let tempBase = FileManager.default.temporaryDirectory
            .appendingPathComponent("flash_moe_demo_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempBase, withIntermediateDirectories: true)
        tempDir = tempBase.path

        // Create weight file
        let hiddenDim = modelConfig.hiddenDim
        let groupSize = modelConfig.groupSize
        let vocabSize = modelConfig.vocabSize
        let numGroups = hiddenDim / groupSize
        let packedCols = hiddenDim / 8

        var tensors: [String: [String: Any]] = [:]
        var binaryData = Data()

        func addTensor(name: String, size: Int) {
            let padding = (64 - (binaryData.count % 64)) % 64
            binaryData.append(Data(repeating: 0, count: padding))
            let offset = binaryData.count
            binaryData.append(Data(repeating: 0, count: size))
            tensors[name] = ["offset": offset, "size": size, "shape": [size], "dtype": "U32"]
        }

        // Minimal tensors for the tiny model
        addTensor(name: "model.embed_tokens.weight", size: vocabSize * packedCols * 4)
        addTensor(name: "model.embed_tokens.scales", size: vocabSize * numGroups * 2)
        addTensor(name: "model.embed_tokens.biases", size: vocabSize * numGroups * 2)

        for i in 0..<modelConfig.numLayers {
            let prefix = "model.layers.\(i)"
            addTensor(name: "\(prefix).input_layernorm.weight", size: hiddenDim * 2)
            addTensor(name: "\(prefix).post_attention_layernorm.weight", size: hiddenDim * 2)

            let attn = "\(prefix).self_attn"
            if modelConfig.isFullAttention(layer: i) {
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    addTensor(name: "\(attn).\(proj).weight", size: 4096)
                    addTensor(name: "\(attn).\(proj).scales", size: 256)
                    addTensor(name: "\(attn).\(proj).biases", size: 256)
                }
                addTensor(name: "\(attn).q_norm.weight", size: modelConfig.headDim * 2)
                addTensor(name: "\(attn).k_norm.weight", size: modelConfig.headDim * 2)
            } else {
                for proj in ["qkv_proj", "z_proj", "beta_proj", "alpha_proj", "out_proj"] {
                    addTensor(name: "\(attn).\(proj).weight", size: 4096)
                    addTensor(name: "\(attn).\(proj).scales", size: 256)
                    addTensor(name: "\(attn).\(proj).biases", size: 256)
                }
                addTensor(name: "\(attn).conv1d.weight", size: 1024)
                addTensor(name: "\(attn).a_log", size: modelConfig.linearNumVHeads * 4)
                addTensor(name: "\(attn).dt_bias", size: modelConfig.linearNumVHeads * 2)
                addTensor(name: "\(attn).g_norm.weight", size: modelConfig.linearValueDim * 2)
            }

            let moe = "\(prefix).mlp"
            for name in ["\(moe).gate", "\(moe).shared_expert.gate_proj", "\(moe).shared_expert.up_proj",
                         "\(moe).shared_expert.down_proj", "\(moe).shared_expert_gate"] {
                addTensor(name: "\(name).weight", size: 4096)
                addTensor(name: "\(name).scales", size: 256)
                addTensor(name: "\(name).biases", size: 256)
            }
        }

        addTensor(name: "model.norm.weight", size: hiddenDim * 2)
        addTensor(name: "lm_head.weight", size: vocabSize * packedCols * 4)
        addTensor(name: "lm_head.scales", size: vocabSize * numGroups * 2)
        addTensor(name: "lm_head.biases", size: vocabSize * numGroups * 2)

        let weightsPath = tempBase.appendingPathComponent("model_weights.bin").path
        try binaryData.write(to: URL(fileURLWithPath: weightsPath))

        let manifest: [String: Any] = ["tensors": tensors]
        let jsonData = try JSONSerialization.data(withJSONObject: manifest, options: .prettyPrinted)
        let manifestPath = tempBase.appendingPathComponent("model_weights.json").path
        try jsonData.write(to: URL(fileURLWithPath: manifestPath))

        weightFile = try WeightFile(weightsPath: weightsPath, manifestPath: manifestPath)
        layerWeights = LayerWeightCacheBuilder.build(from: weightFile, config: modelConfig)

        // Create expert files
        var fds: [Int32] = []
        for i in 0..<modelConfig.numLayers {
            let path = tempBase.appendingPathComponent("layer_\(i).bin").path
            let fd = open(path, O_CREAT | O_RDWR | O_TRUNC, 0o644)
            let zeros = Data(repeating: 0, count: modelConfig.numExperts * modelConfig.expertSize4Bit)
            let writeResult = zeros.withUnsafeBytes { ptr in
                guard let base = ptr.baseAddress else { return -1 }
                return Darwin.write(fd, base, zeros.count)
            }
            _ = writeResult
            _ = lseek(fd, 0, SEEK_SET)
            fds.append(fd)
        }
        expertFDs = fds

        logger.info("[demo] Synthetic model created (\(binaryData.count, privacy: .public) bytes)")
    } else {
        throw FlashMoEError.notImplemented(feature: "--model mode not yet implemented. Use --demo for testing.")
    }

    // ---- Initialize Metal context ----
    let ctx = try MetalContext(config: modelConfig, shaderPath: serverConfig.shaderPath,
                                use2Bit: serverConfig.use2Bit)
    ctx.setWeights(weightFile.data, size: weightFile.size)

    let generator = TokenGenerator(context: ctx, config: modelConfig,
                                    activeExperts: serverConfig.activeExperts)
    if serverConfig.timing {
        generator.pipeline.timingEnabled = true
    }

    logger.info("[server] Metal context ready: \(ctx.device.name, privacy: .public)")
    logger.info("[server] Config: \(modelConfig.numLayers, privacy: .public) layers, \(modelConfig.numExperts, privacy: .public) experts, K=\(serverConfig.activeExperts, privacy: .public)")

    // ---- Start HTTP server ----
    let server = HTTPServer(port: serverConfig.port) { prompt, maxTokens, writer in
        logger.info("[request] prompt=\(prompt.prefix(80), privacy: .private)... maxTokens=\(maxTokens, privacy: .public)")

        writer.sendHeaders()

        // Tokenize (placeholder: use character codes for demo)
        let promptTokens = Array(prompt.utf8).map { Int($0) % modelConfig.vocabSize }

        generator.generate(
            promptTokens: promptTokens.isEmpty ? [0] : promptTokens,
            maxTokens: maxTokens,
            weightFile: weightFile,
            expertFDs: expertFDs,
            layerWeights: layerWeights,
            use2Bit: serverConfig.use2Bit,
            onToken: { token in
                // In demo mode, map token ID to a character for visible output
                let ch = String(UnicodeScalar(UInt8(token % 128)))
                return writer.sendDelta(token: ch)
            }
        )

        writer.sendFinish()
        writer.sendDone()
        logger.info("[request] done")
    }

    try server.start()

    // Cleanup (unreachable in normal operation)
    if let dir = tempDir {
        let tempDirURL = URL(fileURLWithPath: dir).standardized
        let tempRoot = FileManager.default.temporaryDirectory.standardized
        guard tempDirURL.path.hasPrefix(tempRoot.path) else {
            logger.error("Temp directory path escapes allowed root: \(dir, privacy: .private)")
            return
        }
        do {
            try FileManager.default.removeItem(at: tempDirURL)
        } catch {
            logger.error("Failed to clean up temp directory: \(error.localizedDescription, privacy: .public)")
        }
    }
}

try main()
