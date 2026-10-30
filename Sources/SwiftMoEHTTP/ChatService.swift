import Foundation
import HTTPTypes
import SwiftHTTPService
import SwiftHTTPServiceNIO
#if canImport(os)
import os
#endif

private let logger = Logger(subsystem: "com.swiftmoe", category: "server")

/// A chat-completions request that passed every check, as the generator receives it.
public struct ChatCompletionRequest: Equatable, Sendable {
    /// Content of the last message, or `""` when there is none.
    public let prompt: String
    /// ``prompt`` as the service's tokenizer split it. These are the tokens the sequence
    /// budget was checked against, so they are the ones to generate from.
    public let promptTokens: [Int]
    /// Tokens to generate: within ``InferenceLimits/maxCompletionTokens`` and, added to
    /// ``promptTokens``, within ``InferenceLimits/maxSequenceTokens``.
    public let maxTokens: Int

    /// Creates a request.
    public init(prompt: String, promptTokens: [Int], maxTokens: Int) {
        self.prompt = prompt
        self.promptTokens = promptTokens
        self.maxTokens = maxTokens
    }
}

/// Things that happen to a request on its way to the model, for tests to wait on instead of the clock.
enum ChatServiceEvent: Hashable, CaseIterable, Sendable {
    /// A validated request found the model busy and began to wait.
    case requestQueued
    /// A waiting request's client went away; the request was dropped without running.
    case queuedRequestAbandoned
    /// A waiting request's deadline passed; it was answered 503.
    case queuedRequestTimedOut
    /// A request gave up the model: its stream ended, failed, or was discarded.
    case slotReleased
}

/// The OpenAI-compatible `POST /v1/chat/completions` service, streaming tokens as Server-Sent Events.
///
/// It is built on `SwiftHTTPService`: the transport bounds every request and connection, and
/// this type adds what is particular to a model server.
///
/// ## Who may call it
/// Running inference is the whole of what the service does, so reaching the generator is
/// the thing to control. The checks run in this order, and nothing a request says about
/// itself is acted on until the ones before it have passed:
///
/// 1. The transport reads the head within its deadline and limits (408, 431, 400), and
///    refuses a target that is not strict origin-form (400).
/// 2. On a loopback bind, `Host` names this server (421): the defence against DNS rebinding.
/// 3. An `Origin`, if sent, is on the allow-list (403); a preflight for it is answered here,
///    before any credential check.
/// 4. The bearer credential matches (401).
/// 5. Method and path are the route (404, 405), and the declared body is within the bound (413).
/// 6. The body is a JSON object (400), the token count is a whole number in range (400), and
///    the prompt and completion fit in a sequence (400).
///
/// Only then does the request wait its turn for the model, no longer than
/// ``InferenceLimits/queueDeadline`` (503), and not at all once its client has gone.
///
/// ## Transport
/// The service speaks HTTP/1.1, one request per connection. Off loopback it refuses to start
/// without a TLS identity: the key and every prompt would cross the network readable.
/// A stream is bounded by ``InferenceLimits/maxStreamDuration`` (two hours by default).
///
/// ```swift
/// let service = ChatService(authentication: .unauthenticatedLoopback, port: 0,
///                           tokenizer: { prompt in prompt.utf8.map { Int($0) } }) { request, shouldContinue, emit in
///     for word in ["Hello", ", ", "world"] where shouldContinue() {
///         guard emit(word) else { return }
///     }
/// }
/// let address = try await service.start()
/// print(address.port)
/// await service.shutdown()
/// ```
public final class ChatService: Sendable {

    /// Splits a prompt into tokens. Deliberately without a default: the sequence budget is only
    /// as true as the count it is given.
    public typealias Tokenizer = @Sendable (_ prompt: String) -> [Int]

    /// Generates a completion, synchronously, on a thread of its own.
    ///
    /// Call `emit` with the text of each token; it returns `false` once the client has gone or
    /// the stream has ended, and the generator should stop. Ask `shouldContinue` before each
    /// unit of work that cannot be taken back, prompt tokens included. Throwing ends the
    /// stream without a finish chunk or `[DONE]`: a truncated stream is what tells a client
    /// that this is not a completed answer.
    public typealias Generate = @Sendable (
        _ request: ChatCompletionRequest,
        _ shouldContinue: @Sendable () -> Bool,
        _ emit: @Sendable (String) -> Bool
    ) throws -> Void

    /// The address the listener holds.
    public typealias BoundAddress = SwiftHTTPServiceNIO.BoundAddress

    /// The first octet of the IPv4 loopback block (RFC 1122 section 3.2.1.3).
    private static let loopbackNetwork: UInt32 = 127

    /// How callers are authenticated.
    public let authentication: Authentication
    /// Origins whose pages may call the service. Empty sends no CORS headers.
    public let allowedOrigins: [String]
    /// Bounds on a request and a connection.
    public let limits: InferenceLimits

    private let host: String
    private let port: UInt16
    private let tls: TLSFiles?
    private let tokenizer: Tokenizer
    private let generate: Generate
    private let observer: (@Sendable (ChatServiceEvent) -> Void)?
    private let accessLogSink: @Sendable (AccessLogEntry) -> Void
    private let gate = InferenceGate()
    private let hosts = HostGate()
    private let running = OSAllocatedUnfairLock<HTTPServer?>(initialState: nil)

    /// Creates a service. Nothing is opened until ``start()``.
    ///
    /// - Parameters:
    ///   - host: The address to bind (default `127.0.0.1`). Off loopback, `tls` is required.
    ///   - port: TCP port to listen on (default 8080). `0` lets the kernel choose.
    ///   - authentication: How callers are authenticated. Deliberately without a default.
    ///   - allowedOrigins: Origins, as `scheme://host[:port]`, whose pages may call the service.
    ///   - limits: Bounds on a request and a connection.
    ///   - tls: The certificate chain and key to serve, or `nil` for plain HTTP on loopback.
    ///   - tokenizer: Splits a prompt into tokens.
    ///   - accessLog: Receives one entry per request; by default one log line each.
    ///   - generate: Generates a completion for each request that passed every check, one at a time.
    public convenience init(
        host: String = "127.0.0.1",
        port: UInt16 = 8080,
        authentication: Authentication,
        allowedOrigins: [String] = [],
        limits: InferenceLimits = InferenceLimits(),
        tls: TLSFiles? = nil,
        tokenizer: @escaping Tokenizer,
        accessLog: (@Sendable (AccessLogEntry) -> Void)? = nil,
        generate: @escaping Generate
    ) {
        self.init(host: host, port: port, authentication: authentication, allowedOrigins: allowedOrigins,
                  limits: limits, tls: tls, tokenizer: tokenizer, observer: nil, accessLog: accessLog,
                  generate: generate)
    }

    /// Creates a service that reports what happens to its requests.
    init(
        host: String,
        port: UInt16,
        authentication: Authentication,
        allowedOrigins: [String],
        limits: InferenceLimits,
        tls: TLSFiles?,
        tokenizer: @escaping Tokenizer,
        observer: (@Sendable (ChatServiceEvent) -> Void)?,
        accessLog: (@Sendable (AccessLogEntry) -> Void)? = nil,
        generate: @escaping Generate
    ) {
        self.host = host
        self.port = port
        self.authentication = authentication
        self.allowedOrigins = allowedOrigins
        self.limits = limits
        self.tls = tls
        self.tokenizer = tokenizer
        self.generate = generate
        self.observer = observer
        self.accessLogSink = accessLog ?? { entry in
            logger.info("[access] \(entry.line, privacy: .public)")
        }
    }

    /// Whether `host` is a loopback address: a literal in `127.0.0.0/8`, or `::1`.
    ///
    /// Names are not resolved, so `localhost` is not one: what a name resolves to can change
    /// without this code being told. This is the test for running without a credential.
    ///
    /// - Parameter host: Text such as `"127.0.0.1"`.
    public static func isLoopback(_ host: String) -> Bool {
        var address = in_addr()
        if inet_pton(AF_INET, host, &address) == 1 {
            return UInt32(bigEndian: address.s_addr) >> 24 == loopbackNetwork
        }
        return host.trimmingCharacters(in: CharacterSet(charactersIn: "[]")) == "::1"
    }

    /// Whether the transport counts `host` as a bind only this machine can reach: a loopback
    /// address, or the name `localhost`. Such a bind needs no TLS and gets the `Host` check.
    public static func isLocalBind(_ host: String) -> Bool {
        isLoopback(host) || host.lowercased() == "localhost"
    }

    /// The route this service answers.
    public static let route = "/v1/chat/completions"

    // MARK: - Lifecycle

    /// Checks the configuration, opens the listener, and starts serving.
    ///
    /// Nothing is bound until the configuration has been checked, so a service that may not
    /// listen never holds the port.
    ///
    /// - Returns: The address the socket holds (useful with port `0`).
    /// - Throws: ``ServiceError`` for a configuration that cannot be served, a TLS identity
    ///   that is unfit, plain text off loopback, or a socket that will not open.
    public func start() async throws -> BoundAddress {
        if case .unauthenticatedLoopback = authentication, !Self.isLoopback(host) {
            throw ServiceError.credentialRequired(host: host)
        }
        try limits.validate()

        let server = try makeServer()
        let address: BoundAddress
        do {
            address = try await server.start()
        } catch let error as HTTPServerError {
            throw Self.translated(error, host: host)
        }
        if Self.isLocalBind(host) {
            hosts.allow(for: address)
        }
        running.withLock { $0 = server }
        switch authentication {
        case .bearer:
            logger.info("[server] Bearer credential required.")
        case .unauthenticatedLoopback:
            logger.warning("[server] Running WITHOUT authentication on \(address.host, privacy: .public): any local process can run inference.")
        }
        logger.info("[server] Listening on \(address.host, privacy: .public):\(address.port, privacy: .public)\(self.tls == nil ? "" : " (TLS)", privacy: .public)")
        return address
    }

    /// Stops serving, gracefully: no new connection is accepted, and a stream in flight has
    /// ``InferenceLimits/shutdownGrace`` to finish.
    public func shutdown() async {
        let server = running.withLock { state -> HTTPServer? in
            defer { state = nil }
            return state
        }
        await server?.shutdown()
    }

    /// Reads the certificate and key again from the files they were loaded from, without a
    /// restart. Connections already open are not touched; an unfit replacement leaves the old
    /// identity in service.
    ///
    /// - Returns: Whether a server with TLS was running to reload.
    @discardableResult
    public func reloadTLS() -> Bool {
        guard tls != nil, let server = running.withLock({ $0 }) else { return false }
        return server.reloadTLS() != nil
    }

    /// Starts, reports the bound address to `onBound`, and serves until the calling task is
    /// cancelled, then shuts down gracefully. With TLS, `SIGHUP` reloads the identity.
    ///
    /// - Parameter onBound: Called once the socket is open.
    /// - Throws: What ``start()`` throws.
    public func run(onBound: @Sendable (BoundAddress) -> Void = { _ in }) async throws {
        let triggers: AsyncStream<Void>? = tls == nil ? nil : TLSReloadTrigger.hangup()
        let address = try await start()
        onBound(address)
        let reloading: Task<Void, Never>? = triggers.flatMap { triggers in
            running.withLock { $0 }.map { server in
                Task {
                    await server.reloadTLS(onEach: triggers, mode: .always) { outcome in
                        logger.notice("[server] Reloaded the TLS identity: \(String(describing: outcome), privacy: .public)")
                    }
                }
            }
        }
        // A stream nobody yields to: iterating it ends when, and only when, the task is cancelled.
        let (idle, neverYielded) = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        for await _ in idle {}
        withExtendedLifetime(neverYielded) {}
        logger.info("[server] Stopping")
        reloading?.cancel()
        await shutdown()
    }

    // MARK: - Building the server

    private func makeServer() throws -> HTTPServer {
        var transport = Limits()
        transport.maximumHeaderBytes = limits.maxHeaderBytes
        transport.maximumBodyBytes = limits.maxBodyBytes
        transport.headerReadTimeout = limits.readDeadline
        transport.bodyReadTimeout = limits.readDeadline
        transport.streamWriteTimeout = limits.writeDeadline
        transport.maximumConnections = limits.maxConnections
        transport.maximumStreamDuration = limits.maxStreamDuration
        transport.shutdownGrace = limits.shutdownGrace

        let configuration: ServiceConfiguration
        do {
            configuration = try ServiceConfiguration(host: host, port: port, tls: tls, limits: transport,
                                                     protocols: .http1, connections: .closeAfterResponse)
        } catch {
            logger.error("[server] configuration refused: \(error.field, privacy: .public)")
            throw ServiceError.invalidConfiguration(field: error.field)
        }

        var router = Router(
            notFound: { _ in OpenAIError.notFound.response },
            methodNotAllowed: { _, _ in OpenAIError.methodNotAllowed.response })
        do {
            try router.post(Self.route, maximumBodyBytes: limits.maxBodyBytes) { [self] request in
                try await handle(request)
            }
        } catch {
            logger.error("[server] route refused: \(String(describing: error), privacy: .public)")
            throw ServiceError.invalidConfiguration(field: "routes")
        }

        let limits = self.limits
        let refusals = RefusalHooks(respond: { refusal in
            Self.refusal(for: refusal, limits: limits).response
        })
        return HTTPServer(configuration: configuration, router: router, middleware: try middleware(),
                          refusals: refusals)
    }

    /// The layers around the router, outermost first: the access log, the shape of every refusal,
    /// the `Host` check (loopback binds), the origin check and preflight, the credential.
    func middleware() throws -> [Middleware] {
        var layers: [Middleware] = [accessLog(sink: accessLogSink), Self.openAIShapedProblems]
        if Self.isLocalBind(host) {
            layers.append(hosts.middleware)
        }
        do {
            layers.append(try cors(allowedOrigins: allowedOrigins))
        } catch CORSError.invalidOrigin(let index) {
            throw ServiceError.invalidOrigin(allowedOrigins.indices.contains(index) ? allowedOrigins[index] : "")
        } catch {
            logger.error("[server] allowed origins refused: \(String(describing: error), privacy: .public)")
            throw ServiceError.invalidConfiguration(field: "allowedOrigins")
        }
        if case .bearer(let credential) = authentication {
            layers.append(Self.bearer(credential))
        }
        return layers
    }

    private static func bearer(_ credential: BearerCredential) -> Middleware {
        { request, next in
            guard credential.accepts(authorization: request.headers[.authorization]) else {
                return OpenAIError.unauthorized.response
            }
            return try await next(request)
        }
    }

    /// The kit's middleware refuse with RFC 9457 problems; a client of this API reads
    /// `{"error":{"message","type"}}`. Every other header the layer set (`Vary`, `Access-Control-*`) stays.
    private static let openAIShapedProblems: Middleware = { request, next in
        var response = try await next(request)
        guard response.headers[.contentType] == "application/problem+json" else { return response }
        let replacement: OpenAIError
        switch response.status.code {
        case 421: replacement = .wrongHost
        case 403: replacement = .originNotAllowed
        default: replacement = .internalError
        }
        response.body = Data(replacement.body.utf8)
        response.headers[.contentType] = "application/json"
        return response
    }

    private static func refusal(for refusal: TransportRefusal, limits: InferenceLimits) -> OpenAIError {
        switch refusal {
        case .badRequest: .malformed
        case .malformedTarget: .malformedTarget
        case .requestTimeout: .timedOut
        case .bodyTooLarge: .bodyTooLarge(limit: limits.maxBodyBytes)
        case .expectationFailed: .expectationFailed
        case .headersTooLarge: .headersTooLarge(limit: limits.maxHeaderBytes)
        case .internalError: .internalError
        }
    }

    private static func translated(_ error: HTTPServerError, host: String) -> ServiceError {
        switch error {
        case .exposedPlaintext: .exposedPlaintext(host: host)
        case .tlsIdentity(let identityError): .tlsIdentity(String(describing: identityError))
        case .bindFailed: .bindFailed(host: host)
        case .alreadyStarted: .invalidConfiguration(field: "alreadyStarted")
        }
    }

    // MARK: - The request

    func handle(_ request: Request) async throws -> Response {
        guard request.headers[.contentLength] != nil || request.headers[.transferEncoding] != nil else {
            return OpenAIError.lengthRequired.response
        }
        let chat: ChatRequest
        switch ChatRequest.parse(Array(request.body), limits: limits) {
        case .failure(let rejection):
            return rejection.refusal.response
        case .success(let parsed):
            chat = parsed
        }
        let promptTokens = tokenizer(chat.prompt)
        // `maxTokens` is at most `maxSequenceTokens`, so the subtraction cannot go below zero.
        guard promptTokens.count <= limits.maxSequenceTokens - chat.maxTokens else {
            return OpenAIError.sequenceTooLong(promptTokens: promptTokens.count, completionTokens: chat.maxTokens,
                                               limit: limits.maxSequenceTokens).response
        }
        let completion = ChatCompletionRequest(prompt: chat.prompt, promptTokens: promptTokens, maxTokens: chat.maxTokens)

        let observer = self.observer
        let slot: InferenceSlot
        do {
            slot = try await gate.admit(deadline: ContinuousClock.now + limits.queueDeadline,
                                        onQueued: { observer?(.requestQueued) })
        } catch InferenceGateError.timedOut {
            logger.info("[server] request waited \(self.limits.queueDeadline.description, privacy: .public) for the model; answered 503")
            observer?(.queuedRequestTimedOut)
            return OpenAIError.queueTimedOut(retryAfter: limits.retryAfterSeconds).response
        } catch {
            logger.info("[server] client left while queued; request dropped")
            observer?(.queuedRequestAbandoned)
            throw error
        }

        let generate = self.generate
        var response = Response.eventStream { events in
            try await Self.stream(completion, generate: generate, to: events)
        }
        response.stream = response.stream?.observing { _, _ in
            slot.release { observer?(.slotReleased) }
        }
        return response
    }

    /// Runs the generation on its own thread and writes what it produces.
    ///
    /// Returns only once the thread has left the generator, so the slot is never given to another
    /// request while this one is still using the model.
    private static func stream(_ request: ChatCompletionRequest, generate: @escaping Generate,
                               to events: EventWriter) async throws {
        let bridge = GenerationBridge { shouldContinue, emit in
            try generate(request, shouldContinue, emit)
        }
        let chunk = ChatChunk()
        bridge.start()
        do {
            try await withTaskCancellationHandler {
                for try await token in bridge.tokens {
                    try await events.send(data: chunk.delta(token: token))
                    bridge.consumed()
                }
            } onCancel: {
                bridge.cancel()
            }
        } catch {
            // The generator failed, or the client left mid-write: the stream ends truncated.
            bridge.cancel()
            await bridge.stopped()
            logger.error("[request] stream ended early: \(String(describing: error), privacy: .public)")
            throw error
        }
        await bridge.stopped()
        guard !bridge.isCancelled, !Task.isCancelled else {
            logger.info("[request] client left; generation stopped")
            throw CancellationError()
        }
        try await events.send(data: chunk.finish())
        try await events.send(data: ChatChunk.done)
    }
}

/// Holds the `Host` check until the port is known.
///
/// The service may be asked to bind port 0, so the names a `Host` header may carry are not known
/// until the socket is open. Until then every request is refused, which is the safe answer.
private final class HostGate: Sendable {
    private let check = OSAllocatedUnfairLock<Middleware?>(initialState: nil)

    func allow(for address: BoundAddress) {
        let literal = address.host.contains(":") ? "[\(address.host)]" : address.host
        let names = [literal, "localhost"].map { "\($0):\(address.port)" }
        do {
            let middleware = try allowedHosts(names)
            check.withLock { $0 = middleware }
        } catch {
            // Names are built from a bound address and a fixed word, so this is not expected; with no
            // check installed every request is refused, which is the safe answer.
            logger.error("[server] the Host check could not be built: \(String(describing: error), privacy: .public)")
        }
    }

    var middleware: Middleware {
        { [self] request, next in
            guard let check = check.withLock({ $0 }) else {
                return try Response.problem(Problem(status: HTTPResponse.Status(code: 421), code: "misdirected_request", title: "Misdirected Request"))
            }
            return try await check(request, next)
        }
    }
}
