import Foundation
#if canImport(os)
import os
#endif
import SwiftHTTPService
import SwiftMoE
import SwiftMoEHTTP

// ============================================================================
// flash-moe-server — OpenAI-compatible HTTP server with SSE streaming
//
// Usage:
//   swift-moe-server --demo [--host 127.0.0.1] [--port 8080] [--api-key-file <path>]
//                    [--allow-origin <origin>]... [--no-auth]
//                    [--tls-cert <chain.pem> --tls-key <key.pem>]
//                    [--k 4] [--2bit] [--timing]
//
// Authentication:
//   Every request must carry `Authorization: Bearer <key>`. The key is read from the file
//   named by --api-key-file (which must not be accessible to group or other), or else from
//   the environment variable SWIFT_MOE_API_KEY. It is never taken from the command line,
//   where `ps` would show it, and never logged.
//
//   Without a key the server does not start. --no-auth starts it without one on a loopback
//   address only; on any other address a key is required and --no-auth is refused.
//
// Transport:
//   Plain HTTP on a loopback address exposes nothing. On any other --host the server refuses
//   to start unless it terminates TLS itself: give it --tls-cert (the PEM chain, leaf first)
//   and --tls-key (the PEM private key, mode 0600). SIGHUP reloads both without a restart.
//   The transport speaks HTTP/1.1, one request per connection, and ends a stream after two hours.
//
// Browsers:
//   No CORS headers are sent, and a request carrying an Origin is refused, unless that origin
//   was named with --allow-origin (repeatable), e.g. --allow-origin http://localhost:3000.
//
// API:
//   POST /v1/chat/completions  (OpenAI chat format, SSE response)
// ============================================================================

/// The demo model and what it needs to generate, handed to the service's generator thread.
// Justification: the service runs one generation at a time and never touches these from two threads at once; the generator is not Sendable only because it holds Metal buffers.
final class DemoModel: @unchecked Sendable {
    let generator: TokenGenerator
    let weightFile: WeightFile
    let expertFDs: [Int32]
    let layerWeights: [LayerWeightPointers]
    let use2Bit: Bool

    init(generator: TokenGenerator, weightFile: WeightFile, expertFDs: [Int32],
         layerWeights: [LayerWeightPointers], use2Bit: Bool) {
        self.generator = generator
        self.weightFile = weightFile
        self.expertFDs = expertFDs
        self.layerWeights = layerWeights
        self.use2Bit = use2Bit
    }

    func generate(_ request: ChatCompletionRequest, shouldContinue: @Sendable () -> Bool,
                  emit: @Sendable (String) -> Bool) throws {
        try generator.generate(
            promptTokens: request.promptTokens,
            maxTokens: request.maxTokens,
            weightFile: weightFile,
            expertFDs: expertFDs,
            layerWeights: layerWeights,
            use2Bit: use2Bit,
            // Asked before every token, prompt tokens included: a client that has left
            // stops costing GPU time at the next token instead of at the end.
            shouldContinue: shouldContinue,
            // In demo mode, map token ID to a character for visible output
            onToken: { token in emit(String(UnicodeScalar(UInt8(token % 128)))) }
        )
    }
}

private let logger = Logger(subsystem: "com.swiftmoe.server", category: "main")

struct ServerConfig {
    var modelPath: String?
    var host: String = "127.0.0.1"
    var port: UInt16 = 8080
    var activeExperts: Int = 4
    var use2Bit: Bool = false
    var timing: Bool = false
    var demo: Bool = false
    var shaderPath: String = "metal_infer/shaders.metal"
    var apiKeyFile: String?
    var noAuth: Bool = false
    var tlsCertificate: String?
    var tlsKey: String?
    var allowedOrigins: [String] = []
}

func parseArgs() throws -> ServerConfig {
    var config = ServerConfig()
    var i = 1
    let args = CommandLine.arguments
    while i < args.count {
        switch args[i] {
        case "--model": i += 1; if i < args.count { config.modelPath = args[i] }
        case "--host": i += 1; if i < args.count { config.host = args[i] }
        case "--port": i += 1; if i < args.count { config.port = UInt16(args[i]) ?? 8080 }
        case "--k": i += 1; if i < args.count { config.activeExperts = Int(args[i]) ?? 4 }
        case "--2bit": config.use2Bit = true
        case "--timing": config.timing = true
        case "--demo": config.demo = true
        case "--shaders": i += 1; if i < args.count { config.shaderPath = args[i] }
        case "--api-key-file", "--allow-origin", "--tls-cert", "--tls-key":
            // A security option with its value missing is an error, not an option to skip.
            let option = args[i]
            i += 1
            guard i < args.count else { throw ServiceError.missingValue(option: option) }
            switch option {
            case "--api-key-file": config.apiKeyFile = args[i]
            case "--tls-cert": config.tlsCertificate = args[i]
            case "--tls-key": config.tlsKey = args[i]
            default: config.allowedOrigins.append(args[i])
            }
        case "--no-auth": config.noAuth = true
        default: break
        }
        i += 1
    }
    return config
}

func main() async throws {
    let serverConfig = try parseArgs()

    // Decide who may call the server before anything expensive is built, so a server that
    // may not listen fails in the first millisecond rather than after the model is loaded.
    let authentication = try Authentication.resolve(
        host: serverConfig.host,
        keyFile: serverConfig.apiKeyFile,
        environment: ProcessInfo.processInfo.environment,
        noAuth: serverConfig.noAuth
    )

    // The same for the bind: a listener that would serve plain text to other machines fails now,
    // not after the model is loaded. Half a TLS identity is an error, never plain text.
    var tls: TLSFiles?
    switch (serverConfig.tlsCertificate, serverConfig.tlsKey) {
    case (let certificate?, let key?):
        tls = TLSFiles(certificateChainPath: certificate, privateKeyPath: key)
    case (nil, nil):
        break
    case (.some, nil):
        throw ServiceError.missingValue(option: "--tls-key")
    case (nil, .some):
        throw ServiceError.missingValue(option: "--tls-cert")
    }
    if tls == nil, !ChatService.isLocalBind(serverConfig.host) {
        throw ServiceError.exposedPlaintext(host: serverConfig.host)
    }

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
    // Placeholder tokenizer for the demo: one token per UTF-8 byte, and one token for an
    // empty prompt, because the generator needs something to start from.
    let vocabSize = modelConfig.vocabSize
    let tokenizer: ChatService.Tokenizer = { prompt in
        let tokens = prompt.utf8.map { Int($0) % vocabSize }
        return tokens.isEmpty ? [0] : tokens
    }

    // The sequence the service admits is the sequence this generator can hold.
    var limits = InferenceLimits()
    limits.maxSequenceTokens = generator.maxSequenceLength
    limits.maxCompletionTokens = min(limits.maxCompletionTokens, generator.maxSequenceLength)
    limits.defaultCompletionTokens = min(limits.defaultCompletionTokens, limits.maxCompletionTokens)

    let model = DemoModel(generator: generator, weightFile: weightFile, expertFDs: expertFDs,
                          layerWeights: layerWeights, use2Bit: serverConfig.use2Bit)
    let service = ChatService(
        host: serverConfig.host,
        port: serverConfig.port,
        authentication: authentication,
        allowedOrigins: serverConfig.allowedOrigins,
        limits: limits,
        tls: tls,
        tokenizer: tokenizer
    ) { request, shouldContinue, emit in
        logger.info("[request] prompt=\(request.prompt.prefix(80), privacy: .private)... promptTokens=\(request.promptTokens.count, privacy: .public) maxTokens=\(request.maxTokens, privacy: .public)")
        try model.generate(request, shouldContinue: shouldContinue, emit: emit)
        logger.info("[request] done")
    }

    // Serve until SIGINT or SIGTERM, then stop gracefully: a stream in flight gets its grace period.
    let serving = Task { try await service.run() }
    let signals = [SIGINT, SIGTERM].map { number -> DispatchSourceSignal in
        signal(number, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
        source.setEventHandler { serving.cancel() }
        source.resume()
        return source
    }
    defer { signals.forEach { $0.cancel() } }
    try await serving.value

    // Cleanup (unreachable in normal operation)
    if let dir = tempDir {
        let tempDirURL = URL(fileURLWithPath: dir).standardized
        let tempRoot = FileManager.default.temporaryDirectory.standardized
        guard PathContainment.isContained(tempDirURL, in: tempRoot) else {
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

do {
    try await main()
} catch {
    logger.error("[server] \(error.localizedDescription, privacy: .public)")
    FileHandle.standardError.write(Data("swift-moe-server: \(error.localizedDescription)\n".utf8))
    // A refusal to start is an ordinary failure with a sentence attached, not a trap.
    exit(EXIT_FAILURE)
}
