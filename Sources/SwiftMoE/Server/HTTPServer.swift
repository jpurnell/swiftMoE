import Foundation
#if canImport(os)
import os
#endif

private let logger = Logger(subsystem: "com.swiftmoe", category: "server")

/// Minimal HTTP server for OpenAI-compatible chat completions with SSE streaming.
///
/// Listens on a TCP port and handles `/v1/chat/completions` POST requests.
/// Each request spawns token generation and streams results via SSE.
///
/// ## Reachability
/// The server has **no authentication**: whoever can open a connection to the port can run
/// inference. It therefore binds the loopback interface (`127.0.0.1`) unless a caller names
/// another address. Passing `host: "0.0.0.0"` — or any non-loopback address — publishes an
/// unauthenticated endpoint to every machine that can route to it, and is a decision for the
/// caller to make in writing, not a default.
///
/// ```swift
/// let server = HTTPServer(port: 8080) { prompt, maxTokens, writer in
///     writer.sendHeaders()
///     writer.sendDone()
/// }
/// let bound = try server.openListener()   // 127.0.0.1:8080
/// try server.start()
/// ```
///
/// ## Protocol
/// - **Endpoint:** `POST /v1/chat/completions`
/// - **Request body:** OpenAI chat completion format (messages array)
/// - **Response:** Server-Sent Events with token deltas
///
/// Matches the server in `infer.m:5635-6500`.
public final class HTTPServer {

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

    /// Called for each incoming chat request. Return the prompt string.
    public typealias RequestHandler = (
        _ prompt: String,
        _ maxTokens: Int,
        _ sseWriter: SSEWriter
    ) -> Void

    private let handler: RequestHandler
    private var serverFD: Int32 = -1
    private var shouldStop = false

    /// Creates an HTTP server.
    ///
    /// The server is unauthenticated, so `host` defaults to loopback. Name another address only
    /// when the endpoint is meant to be reachable from other machines.
    ///
    /// - Parameters:
    ///   - host: IPv4 literal to bind (default ``loopbackHost``, `127.0.0.1`). Host names are
    ///     not resolved; `"0.0.0.0"` binds every interface.
    ///   - port: TCP port to listen on (default 8080). `0` lets the kernel choose.
    ///   - handler: Callback invoked for each chat completion request.
    public init(host: String = HTTPServer.loopbackHost, port: UInt16 = 8080, handler: @escaping RequestHandler) {
        self.host = host
        self.port = port
        self.handler = handler
    }

    /// The address the listening socket is bound to, or `nil` when the server is not listening.
    ///
    /// Read from the socket with `getsockname(2)` on every access.
    public var boundAddress: BoundAddress? {
        guard serverFD >= 0 else { return nil }
        return Self.socketName(serverFD)
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
    /// - Returns: The address the socket is bound to, as the kernel reports it.
    /// - Throws: ``FlashMoEError/invalidBindAddress(host:)`` when ``host`` is not an IPv4
    ///   literal; ``FlashMoEError/readFailed(errno:context:)`` when a socket call fails.
    @discardableResult
    public func openListener() throws -> BoundAddress {
        if let existing = boundAddress { return existing }

        guard let address = Self.ipv4Address(host) else {
            throw FlashMoEError.invalidBindAddress(host: host)
        }

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

        guard listen(fd, 5) == 0 else {
            let code = errno
            close(fd)
            throw FlashMoEError.readFailed(errno: code, context: "listen()")
        }

        guard let bound = Self.socketName(fd) else {
            let code = errno
            close(fd)
            throw FlashMoEError.readFailed(errno: code, context: "getsockname()")
        }

        serverFD = fd
        shouldStop = false
        return bound
    }

    /// Starts the server and blocks, accepting connections.
    ///
    /// This method does not return until the server is stopped or an error occurs.
    public func start() throws {
        let bound = try openListener()

        logger.info("[server] Listening on \(bound.endpoint, privacy: .public)")
        if !Self.isLoopback(bound.host) {
            logger.warning("[server] Bound to \(bound.host, privacy: .public), which is not loopback. This server has no authentication: any machine that can reach the port can run inference.")
        }

        while !shouldStop {
            var clientAddr = sockaddr_in()
            var addrLen = socklen_t(MemoryLayout<sockaddr_in>.size)

            let clientFD = Self.acceptSocket(serverFD, &clientAddr, &addrLen)
            guard clientFD >= 0 else { continue }

            handleConnection(clientFD)
            close(clientFD)
        }
    }

    /// Stops the server.
    public func stop() {
        shouldStop = true
        if serverFD >= 0 {
            close(serverFD)
            serverFD = -1
        }
    }

    // MARK: - Private

    private func handleConnection(_ clientFD: Int32) {
        // Read HTTP request
        var buf = [UInt8](repeating: 0, count: 65536)
        var total = 0

        // Read until we find \r\n\r\n (end of headers)
        buf.withUnsafeMutableBufferPointer { bufPtr in
            guard let base = bufPtr.baseAddress else { return }
            while total < bufPtr.count - 1 {
                let n = Darwin.read(clientFD, base + total, 1)
                if n <= 0 { return }
                total += 1
                if total >= 4 &&
                    bufPtr[total-4] == 0x0D && bufPtr[total-3] == 0x0A &&
                    bufPtr[total-2] == 0x0D && bufPtr[total-1] == 0x0A {
                    break
                }
            }
        }

        let headerString = String(bytes: buf[0..<total], encoding: .utf8) ?? ""

        // Read body if Content-Length present
        if let clRange = headerString.range(of: "Content-Length:", options: .caseInsensitive) {
            let afterCL = headerString[clRange.upperBound...]
            if let contentLen = Int(afterCL.prefix(while: { $0.isNumber || $0 == " " }).trimmingCharacters(in: .whitespaces)) {
                if contentLen > 0 && total + contentLen < buf.count - 1 {
                    buf.withUnsafeMutableBufferPointer { bufPtr in
                        guard let base = bufPtr.baseAddress else { return }
                        var bodyRead = 0
                        while bodyRead < contentLen {
                            let n = Darwin.read(clientFD, base + total + bodyRead, contentLen - bodyRead)
                            if n <= 0 { break }
                            bodyRead += n
                        }
                        total += bodyRead
                    }
                }
            }
        }

        let requestString = String(bytes: buf[0..<total], encoding: .utf8) ?? ""

        // Handle CORS preflight
        if headerString.hasPrefix("OPTIONS") {
            let response = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\n\r\n"
            let responseBytes = Array(response.utf8)
            responseBytes.withUnsafeBufferPointer { respBuf in
                guard let base = respBuf.baseAddress else { return }
                _ = Darwin.write(clientFD, base, respBuf.count)
            }
            return
        }

        // Only handle POST /v1/chat/completions
        guard headerString.hasPrefix("POST") && headerString.contains("/v1/chat/completions") else {
            let response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
            let responseBytes = Array(response.utf8)
            responseBytes.withUnsafeBufferPointer { respBuf in
                guard let base = respBuf.baseAddress else { return }
                _ = Darwin.write(clientFD, base, respBuf.count)
            }
            return
        }

        // Extract prompt from OpenAI messages format
        let prompt = extractLastContent(from: requestString) ?? ""
        let maxTokens = extractMaxTokens(from: requestString, default: 100)

        let writer = SSEWriter(fileDescriptor: clientFD)
        handler(prompt, maxTokens, writer)
    }

    /// Extracts the last "content" value from an OpenAI messages array.
    private func extractLastContent(from request: String) -> String? {
        guard let bodyStart = request.range(of: "\r\n\r\n") else { return nil }
        let body = String(request[bodyStart.upperBound...])

        guard let data = body.data(using: .utf8) else { return nil }
        do {
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let messages = json["messages"] as? [[String: Any]] else {
                return nil
            }
            return messages.last?["content"] as? String
        } catch {
            logger.debug("Failed to parse chat request JSON: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

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

    /// Extracts max_tokens or max_completion_tokens from request body.
    private func extractMaxTokens(from request: String, default defaultVal: Int) -> Int {
        guard let bodyStart = request.range(of: "\r\n\r\n") else { return defaultVal }
        let body = String(request[bodyStart.upperBound...])

        guard let data = body.data(using: .utf8) else { return defaultVal }
        do {
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return defaultVal
            }
            if let max = json["max_completion_tokens"] as? Int { return max }
            if let max = json["max_tokens"] as? Int { return max }
            return defaultVal
        } catch {
            logger.debug("Failed to parse max_tokens JSON: \(error.localizedDescription, privacy: .public)")
            return defaultVal
        }
    }
}
