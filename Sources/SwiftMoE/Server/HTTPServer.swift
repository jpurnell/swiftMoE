import Foundation
#if canImport(os)
import os
#endif

private let logger = Logger(subsystem: "com.swiftmoe", category: "server")

/// Minimal HTTP server for OpenAI-compatible chat completions with SSE streaming.
///
/// Listens on a TCP port and answers `POST /v1/chat/completions`, streaming tokens as
/// Server-Sent Events.
///
/// ## Who may call it
/// Running inference is the whole of what this server does, so reaching the handler is the
/// thing guarded:
///
/// - **Credential.** A server is created with an ``Authentication``; there is no default.
///   ``Authentication/bearer(_:)`` requires `Authorization: Bearer <key>` on every request and
///   answers 401 otherwise. ``Authentication/unauthenticatedLoopback`` checks nothing, and
///   ``openListener()`` refuses it for any address that is not loopback.
/// - **Origin.** No CORS header is sent unless an origin is on `allowedOrigins`, and a request
///   carrying any other `Origin` is refused. Loopback does not keep a web page out — the
///   browser is on loopback too — so this is what does.
/// - **Host.** On a loopback bind the `Host` header must be the bound address or `localhost`
///   with the bound port, which is what defeats DNS rebinding.
/// - **Limits.** ``Limits`` bounds tokens, header and body size, and how long a client may take.
///
/// ```swift
/// let key = try APIKey(String(repeating: "k", count: 32))
/// let server = HTTPServer(
///     port: 8080,
///     authentication: .bearer(BearerCredential(key: key)),
///     tokenizer: { prompt in prompt.utf8.map(Int.init) }
/// ) { request, writer in
///     writer.sendHeaders()
///     writer.sendDone()
/// }
/// let bound = try server.openListener()   // 127.0.0.1:8080
/// server.stop()
/// ```
///
/// ## Concurrency
/// Connections are read on a bounded pool, so a client that is slow or silent delays nobody
/// else; the handler is called for one request at a time.
///
/// ## Protocol
/// - **Endpoint:** `POST /v1/chat/completions`
/// - **Request body:** OpenAI chat completion format (messages array), with `Content-Length`
/// - **Response:** Server-Sent Events with token deltas, or a JSON error
///
/// Matches the server in `infer.m:5635-6500`.
public final class HTTPServer: Sendable {

    /// The loopback address, `127.0.0.1` — the default bind address.
    public static let loopbackHost = "127.0.0.1"

    /// The address a listening socket is actually bound to, read back from the socket.
    ///
    /// This is what `getsockname(2)` reports, not what the caller asked for: a request for
    /// port 0 comes back with the port the kernel assigned.
    public struct BoundAddress: Equatable, Sendable {
        /// Dotted-quad IPv4 address the socket is bound to.
        public let host: String
        /// TCP port the socket is bound to.
        public let port: UInt16

        /// Creates a bound address.
        ///
        /// - Parameters:
        ///   - host: Dotted-quad IPv4 address.
        ///   - port: TCP port.
        public init(host: String, port: UInt16) {
            self.host = host
            self.port = port
        }

        /// URL of the chat-completions endpoint at this address.
        public var endpoint: String {
            "http://\(host):\(port)/v1/chat/completions"
        }
    }

    /// IPv4 address to bind, as a dotted-quad literal. Defaults to ``loopbackHost``.
    public let host: String

    /// Port to listen on. `0` asks the kernel for a free port; see ``boundAddress``.
    public let port: UInt16

    /// A chat-completions request that passed every check, as the handler receives it.
    public struct Request: Equatable, Sendable {
        /// Content of the last message, or `""` when there is none.
        public let prompt: String
        /// ``prompt`` as the server's ``Tokenizer`` split it. These are the tokens the sequence
        /// budget was checked against, so they are the ones to generate from.
        public let promptTokens: [Int]
        /// Tokens to generate: within ``Limits/maxCompletionTokens``, and, added to
        /// ``promptTokens``, within ``Limits/maxSequenceTokens``.
        public let maxTokens: Int

        /// Creates a request.
        ///
        /// - Parameters:
        ///   - prompt: Content of the last message.
        ///   - promptTokens: The prompt's tokens.
        ///   - maxTokens: Tokens to generate.
        public init(prompt: String, promptTokens: [Int], maxTokens: Int) {
            self.prompt = prompt
            self.promptTokens = promptTokens
            self.maxTokens = maxTokens
        }
    }

    /// Splits a prompt into the tokens the model will be given.
    ///
    /// The server calls it once per request, on the connection's own thread and possibly for
    /// several requests at once, before the request waits for the model. It decides how long
    /// a prompt is, which is why it is the server's and not the handler's: a length checked
    /// with one tokenizer and generated with another is not checked.
    public typealias Tokenizer = @Sendable (_ prompt: String) -> [Int]

    /// Called for each chat request that passed every check, one call at a time.
    public typealias RequestHandler = (
        _ request: Request,
        _ sseWriter: SSEWriter
    ) -> Void

    /// How callers are authenticated.
    public let authentication: Authentication

    /// Origins whose pages may call this server. Empty — the default — sends no CORS headers.
    public let allowedOrigins: [String]

    /// Bounds on a request and a connection.
    public let limits: Limits

    /// The listening socket and the accept loop's progress, guarded together.
    private struct State {
        /// The listening socket, or -1.
        var descriptor: Int32 = -1
        /// Whether ``start()`` is inside its accept loop.
        var accepting = false
        /// Set by ``stop()``; the accept loop leaves when it sees it.
        var stopRequested = false
        /// A socket ``stop()`` retired while the loop was still polling it. The loop closes it.
        var retired: Int32 = -1
    }

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let processor: HTTPConnectionProcessor
    /// One permit per connection being served.
    private let connectionSlots: DispatchSemaphore
    /// One permit per over-limit connection being told the server is busy.
    private let refusalSlots: DispatchSemaphore

    /// How often the accept loop looks up from `poll(2)` to see whether it has been stopped.
    private static let acceptPollMilliseconds: Int32 = 100
    /// Connections the kernel queues ahead of `accept(2)`.
    private static let listenBacklog: Int32 = 16

    /// Creates an HTTP server.
    ///
    /// - Parameters:
    ///   - host: IPv4 literal to bind (default ``loopbackHost``, `127.0.0.1`). Host names are
    ///     not resolved; `"0.0.0.0"` binds every interface, and requires a bearer credential.
    ///   - port: TCP port to listen on (default 8080). `0` lets the kernel choose.
    ///   - authentication: How callers are authenticated. Deliberately without a default.
    ///   - allowedOrigins: Origins, as `scheme://host[:port]`, whose pages may call the server.
    ///   - limits: Bounds on a request and a connection.
    ///   - tokenizer: Splits a prompt into tokens. Deliberately without a default: the
    ///     sequence budget is only as true as the count it is given.
    ///   - handler: Callback invoked for each chat completion request, one at a time, on a
    ///     background thread.
    public init(
        host: String = HTTPServer.loopbackHost,
        port: UInt16 = 8080,
        authentication: Authentication,
        allowedOrigins: [String] = [],
        limits: Limits = Limits(),
        tokenizer: @escaping Tokenizer,
        handler: @escaping RequestHandler
    ) {
        self.host = host
        self.port = port
        self.authentication = authentication
        self.allowedOrigins = allowedOrigins
        self.limits = limits
        self.processor = HTTPConnectionProcessor(
            authentication: authentication,
            allowedOrigins: allowedOrigins,
            limits: limits,
            tokenizer: tokenizer,
            handler: SerializedHandler(handler)
        )
        self.connectionSlots = DispatchSemaphore(value: max(0, limits.maxConnections))
        self.refusalSlots = DispatchSemaphore(value: max(0, limits.maxConnections))
    }

    /// The address the listening socket is bound to, or `nil` when the server is not listening.
    ///
    /// Read from the socket with `getsockname(2)` on every access.
    public var boundAddress: BoundAddress? {
        state.withLock { current in
            current.descriptor >= 0 ? Self.socketName(current.descriptor) : nil
        }
    }

    /// Parses a dotted-quad IPv4 literal into a network-byte-order address.
    ///
    /// - Parameter host: Text such as `"127.0.0.1"`.
    /// - Returns: The address in network byte order, or `nil` when `host` is not an IPv4
    ///   literal. Host names are not resolved: `"localhost"` is `nil`.
    public static func ipv4Address(_ host: String) -> in_addr_t? {
        var address = in_addr()
        guard inet_pton(AF_INET, host, &address) == 1 else { return nil }
        return address.s_addr
    }

    /// Whether `host` is an IPv4 literal in the loopback block, `127.0.0.0/8`.
    ///
    /// - Parameter host: Text such as `"127.0.0.1"`.
    /// - Returns: `false` for any other address, and for anything that is not an IPv4 literal.
    public static func isLoopback(_ host: String) -> Bool {
        guard let address = ipv4Address(host) else { return false }
        return UInt32(bigEndian: address) >> 24 == Self.loopbackNetwork
    }

    /// First octet of the IPv4 loopback block (RFC 1122 §3.2.1.3).
    private static let loopbackNetwork: UInt32 = 127

    /// Creates the listening socket, binds it to ``host``:``port``, and starts listening.
    ///
    /// Idempotent: a server that is already listening returns its current address. ``start()``
    /// calls this itself; call it first to learn the address before blocking in `start()`.
    ///
    /// Nothing is bound until the configuration has been checked, so a server that may not
    /// listen never holds the port.
    ///
    /// - Returns: The address the socket is bound to, as the kernel reports it.
    /// - Throws: ``FlashMoEError/invalidBindAddress(host:)`` when ``host`` is not an IPv4
    ///   literal; ``HTTPServerError/credentialRequired(host:)`` when ``host`` is not loopback
    ///   and ``authentication`` is ``Authentication/unauthenticatedLoopback``;
    ///   ``HTTPServerError/invalidOrigin(_:)`` and ``HTTPServerError/invalidLimit(name:)`` for
    ///   a bad allowlist entry or limit; ``FlashMoEError/readFailed(errno:context:)`` when a
    ///   socket call fails.
    @discardableResult
    public func openListener() throws -> BoundAddress {
        if let existing = boundAddress { return existing }

        guard let address = Self.ipv4Address(host) else {
            throw FlashMoEError.invalidBindAddress(host: host)
        }
        if case .unauthenticatedLoopback = authentication, !Self.isLoopback(host) {
            throw HTTPServerError.credentialRequired(host: host)
        }
        for origin in allowedOrigins where !Self.isOrigin(origin) {
            throw HTTPServerError.invalidOrigin(origin)
        }
        try limits.validate()

        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw FlashMoEError.readFailed(errno: errno, context: "socket()")
        }

        // Allow port reuse
        var opt: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = address

        guard Self.bindSocket(fd, &addr) == 0 else {
            let code = errno
            close(fd)
            throw FlashMoEError.readFailed(errno: code, context: "bind(\(host):\(port))")
        }

        guard listen(fd, Self.listenBacklog) == 0 else {
            let code = errno
            close(fd)
            throw FlashMoEError.readFailed(errno: code, context: "listen()")
        }

        guard let bound = Self.socketName(fd) else {
            let code = errno
            close(fd)
            throw FlashMoEError.readFailed(errno: code, context: "getsockname()")
        }

        state.withLock { current in
            current.descriptor = fd
            current.stopRequested = false
        }
        return bound
    }

    /// Starts the server and blocks, accepting connections.
    ///
    /// This method does not return until the server is stopped or an error occurs. Each
    /// accepted connection is handed to a worker, so the loop is back in `accept` at once.
    public func start() throws {
        let bound = try openListener()
        let listener = state.withLock { current -> Int32 in
            current.accepting = true
            return current.descriptor
        }
        defer {
            let retired = state.withLock { current -> Int32 in
                current.accepting = false
                let descriptor = current.retired
                current.retired = -1
                return descriptor
            }
            if retired >= 0 { close(retired) }
        }

        logger.info("[server] Listening on \(bound.endpoint, privacy: .public)")
        switch authentication {
        case .bearer:
            logger.info("[server] Bearer credential required.")
        case .unauthenticatedLoopback:
            logger.warning("[server] Running WITHOUT authentication on \(bound.host, privacy: .public): any local process can run inference.")
        }
        if !Self.isLoopback(bound.host) {
            logger.warning("[server] Bound to \(bound.host, privacy: .public), which is not loopback: the bearer key travels in clear text unless TLS is terminated in front of this server.")
        }

        while !state.withLock({ $0.stopRequested }) {
            var readiness = pollfd(fd: listener, events: Int16(POLLIN), revents: 0)
            guard poll(&readiness, 1, Self.acceptPollMilliseconds) > 0 else { continue }

            var clientAddr = sockaddr_in()
            var addrLen = socklen_t(MemoryLayout<sockaddr_in>.size)
            let clientFD = Self.acceptSocket(listener, &clientAddr, &addrLen)
            guard clientFD >= 0 else { continue }

            dispatch(clientFD, bound: bound)
        }
    }

    /// Hands an accepted connection to a thread of its own, or refuses it when the server is full.
    ///
    /// A thread per connection, rather than a dispatch queue: a worker spends its life blocked
    /// in `poll(2)`, and a shared pool that declines to grow while its threads are blocked
    /// would turn sixteen silent clients back into a stalled server. The two semaphores bound
    /// the threads at twice ``Limits/maxConnections``.
    private func dispatch(_ clientFD: Int32, bound: BoundAddress) {
        let processor = self.processor
        if connectionSlots.wait(timeout: .now()) == .success {
            let slots = connectionSlots
            Thread.detachNewThread {
                processor.serve(clientFD, bound: bound)
                slots.signal()
            }
        } else if refusalSlots.wait(timeout: .now()) == .success {
            let slots = refusalSlots
            Thread.detachNewThread {
                processor.refuseBusy(clientFD)
                slots.signal()
            }
        } else {
            // Past both bounds there is no thread left to say why; the connection is dropped.
            close(clientFD)
        }
    }

    /// Stops the server.
    ///
    /// The listening socket stops being this server's at once — ``boundAddress`` is `nil` on
    /// return. Connections already accepted run to completion.
    public func stop() {
        let toClose = state.withLock { current -> Int32 in
            current.stopRequested = true
            let descriptor = current.descriptor
            current.descriptor = -1
            guard current.accepting else { return descriptor }
            // The accept loop is polling this socket; it closes it on the way out.
            current.retired = descriptor
            return -1
        }
        if toClose >= 0 { close(toClose) }
    }

    // MARK: - Private

    /// Binds a socket to a sockaddr_in using heap-allocated storage to avoid nested withUnsafe scopes.
    private static func bindSocket(_ fd: Int32, _ addr: inout sockaddr_in) -> Int32 {
        let raw = UnsafeMutableRawPointer.allocate(byteCount: MemoryLayout<sockaddr_in>.size,
                                                    alignment: MemoryLayout<sockaddr_in>.alignment)
        defer { raw.deallocate() }
        raw.storeBytes(of: addr, as: sockaddr_in.self)
        let sockPtr = raw.assumingMemoryBound(to: sockaddr.self)
        return bind(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
    }

    /// Reads the address a socket is bound to with `getsockname(2)`.
    private static func socketName(_ fd: Int32) -> BoundAddress? {
        let raw = UnsafeMutableRawPointer.allocate(byteCount: MemoryLayout<sockaddr_in>.size,
                                                    alignment: MemoryLayout<sockaddr_in>.alignment)
        defer { raw.deallocate() }
        raw.initializeMemory(as: UInt8.self, repeating: 0, count: MemoryLayout<sockaddr_in>.size)
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        guard getsockname(fd, raw.assumingMemoryBound(to: sockaddr.self), &length) == 0 else {
            return nil
        }
        let name = raw.load(as: sockaddr_in.self)

        var text = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
        var address = name.sin_addr
        guard inet_ntop(AF_INET, &address, &text, socklen_t(INET_ADDRSTRLEN)) != nil else {
            return nil
        }
        let bytes = text.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return BoundAddress(host: String(decoding: bytes, as: UTF8.self),
                            port: UInt16(bigEndian: name.sin_port))
    }

    /// Accepts a connection using heap-allocated storage to avoid nested withUnsafe scopes.
    private static func acceptSocket(_ fd: Int32, _ addr: inout sockaddr_in, _ addrLen: inout socklen_t) -> Int32 {
        let raw = UnsafeMutableRawPointer.allocate(byteCount: MemoryLayout<sockaddr_in>.size,
                                                    alignment: MemoryLayout<sockaddr_in>.alignment)
        defer { raw.deallocate() }
        raw.storeBytes(of: addr, as: sockaddr_in.self)
        let sockPtr = raw.assumingMemoryBound(to: sockaddr.self)
        let result = accept(fd, sockPtr, &addrLen)
        addr = raw.load(as: sockaddr_in.self)
        return result
    }
}
