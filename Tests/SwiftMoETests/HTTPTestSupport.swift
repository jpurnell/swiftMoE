import Foundation
import os
import Testing
@testable import SwiftMoE

/// One call the server made to its request handler.
struct HandlerCall: Equatable, Sendable {
    let prompt: String
    let maxTokens: Int
}

/// What the stub handler does once it has recorded a call.
typealias StubResponse = @Sendable (HTTPServer.Request, SSEWriter) -> Void

/// The events a server under test reported, and a way to wait for the next one of a kind.
///
/// Tests that need the server to have reached a state — a request queued, a slot released —
/// wait here for the server to say so, not for an interval after which it probably has.
final class ServerEvents: Sendable {

    /// How long a wait lasts before the test reports that the event never came.
    private static let allowance: DispatchTimeInterval = .seconds(10)

    private let log = OSAllocatedUnfairLock<[HTTPServerEvent]>(initialState: [])
    private let signals: [HTTPServerEvent: DispatchSemaphore]

    init() {
        var signals: [HTTPServerEvent: DispatchSemaphore] = [:]
        for event in HTTPServerEvent.allCases {
            signals[event] = DispatchSemaphore(value: 0)
        }
        self.signals = signals
    }

    func record(_ event: HTTPServerEvent) {
        log.withLock { $0.append(event) }
        signals[event]?.signal()
    }

    /// Waits for one occurrence of `event` that no earlier call has consumed.
    ///
    /// - Returns: `false`, having recorded an issue, when none arrived within the allowance.
    @discardableResult
    func awaitNext(_ event: HTTPServerEvent, sourceLocation: SourceLocation = #_sourceLocation) -> Bool {
        guard let signal = signals[event], signal.wait(timeout: .now() + Self.allowance) == .success else {
            Issue.record("the server never reported \(event)", sourceLocation: sourceLocation)
            return false
        }
        return true
    }

    /// How many times `event` has been reported.
    func count(of event: HTTPServerEvent) -> Int {
        log.withLock { $0.filter { $0 == event }.count }
    }
}

/// A real ``HTTPServer`` on a loopback ephemeral port, accepting on its own thread.
///
/// The handler is a stub: it records what it was called with and answers with an empty event
/// stream, so no model is needed to see whether a request got as far as inference.
final class RunningServer: Sendable {

    /// The key the server is configured with unless a test says otherwise.
    static let key = String(repeating: "s", count: 48)
    /// A well-formed key that is not the server's.
    static let wrongKey = String(repeating: "w", count: 48)
    /// How long teardown waits for the accept loop to return before reporting it stuck.
    private static let shutdownAllowance: DispatchTimeInterval = .seconds(10)

    let server: HTTPServer
    let bound: HTTPServer.BoundAddress
    /// What the server has reported about its connections.
    let events = ServerEvents()
    private let calls = OSAllocatedUnfairLock<[HandlerCall]>(initialState: [])
    private let requests = OSAllocatedUnfairLock<[HTTPServer.Request]>(initialState: [])
    private let finished = DispatchSemaphore(value: 0)

    /// The tokenizer the server is given unless a test says otherwise: one token per UTF-8 byte.
    static let byteTokenizer: HTTPServer.Tokenizer = { prompt in prompt.utf8.map { Int($0) } }

    /// The stub's whole answer: stream headers, then `[DONE]`.
    static let emptyStream: StubResponse = { _, writer in
        writer.sendHeaders()
        writer.sendDone()
    }

    /// - Parameters:
    ///   - authenticated: Whether the server requires ``key``.
    ///   - allowedOrigins: The origin allowlist.
    ///   - limits: The server's limits.
    ///   - tokenizer: How the server counts a prompt.
    ///   - respond: What the handler does after recording the call.
    init(
        authenticated: Bool = true,
        allowedOrigins: [String] = [],
        limits: HTTPServer.Limits = HTTPServer.Limits(),
        tokenizer: @escaping HTTPServer.Tokenizer = RunningServer.byteTokenizer,
        respond: @escaping StubResponse = RunningServer.emptyStream
    ) throws {
        let authentication: HTTPServer.Authentication = authenticated
            ? .bearer(BearerCredential(key: try APIKey(Self.key)))
            : .unauthenticatedLoopback
        let calls = self.calls
        let requests = self.requests
        let events = self.events
        let server = HTTPServer(host: HTTPServer.loopbackHost, port: 0, authentication: authentication,
                                allowedOrigins: allowedOrigins, limits: limits, allowPlaintext: false,
                                tokenizer: tokenizer,
                                observer: { events.record($0) }) { request, writer in
            calls.withLock { $0.append(HandlerCall(prompt: request.prompt, maxTokens: request.maxTokens)) }
            requests.withLock { $0.append(request) }
            respond(request, writer)
        }
        self.server = server
        self.bound = try server.openListener()

        let finished = self.finished
        Thread.detachNewThread {
            do {
                try server.start()
            } catch {
                Issue.record("accept loop failed: \(error)")
            }
            finished.signal()
        }
    }

    /// Everything the handler has been called with so far, in order.
    var handlerCalls: [HandlerCall] { calls.withLock { $0 } }

    /// The requests the handler received, in order, with the tokens the server counted.
    var handledRequests: [HTTPServer.Request] { requests.withLock { $0 } }

    /// `127.0.0.1:<port>` — the authority a well-behaved local client sends as `Host`.
    var authority: String { "\(bound.host):\(bound.port)" }

    /// Stops the server and waits for its accept loop to return.
    func shutdown() {
        server.stop()
        if finished.wait(timeout: .now() + Self.shutdownAllowance) == .timedOut {
            Issue.record("accept loop did not return after stop()")
        }
    }

    // MARK: - Requests

    /// Builds a request for this server.
    ///
    /// - Parameters:
    ///   - method: Request method.
    ///   - target: Request target.
    ///   - host: `Host` header value; `nil` sends ``authority``, `""` omits the header.
    ///   - key: Bearer key to present, or `nil` for no `Authorization` header.
    ///   - origin: `Origin` header value, or `nil` to omit it.
    ///   - extraHeaders: Further header lines, without line endings.
    ///   - body: Request body; `nil` sends neither a body nor `Content-Length`.
    func request(
        _ method: String = "POST",
        _ target: String = "/v1/chat/completions",
        host: String? = nil,
        key: String? = RunningServer.key,
        origin: String? = nil,
        extraHeaders: [String] = [],
        body: String? = #"{"messages":[{"role":"user","content":"hi"}]}"#
    ) -> [UInt8] {
        var lines = ["\(method) \(target) HTTP/1.1"]
        let hostValue = host ?? authority
        if !hostValue.isEmpty { lines.append("Host: \(hostValue)") }
        if let key { lines.append("Authorization: Bearer \(key)") }
        if let origin { lines.append("Origin: \(origin)") }
        lines.append(contentsOf: extraHeaders)
        if let body {
            lines.append("Content-Type: application/json")
            lines.append("Content-Length: \(body.utf8.count)")
        }
        return Array((lines.joined(separator: "\r\n") + "\r\n\r\n" + (body ?? "")).utf8)
    }

    /// Sends `bytes` on a fresh connection and returns everything the server wrote back.
    func exchange(_ bytes: [UInt8]) throws -> String {
        let client = try ClientSocket(port: bound.port)
        defer { client.close() }
        client.send(bytes)
        return client.readToEnd()
    }
}

/// A blocking client socket for driving the server byte by byte.
struct ClientSocket {

    /// Seconds a client read may block before the test gives up on the server.
    private static let receiveAllowanceSeconds = 30
    private static let chunkBytes = 4096

    let descriptor: Int32

    /// Wraps a socket that is already connected.
    init(descriptor: Int32) {
        self.descriptor = descriptor
    }

    init(port: UInt16) throws {
        let address = try #require(HTTPServer.ipv4Address(HTTPServer.loopbackHost))
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        try #require(fd >= 0)

        var enabled: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, socklen_t(MemoryLayout<Int32>.size))
        var allowance = timeval(tv_sec: Self.receiveAllowanceSeconds, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &allowance, socklen_t(MemoryLayout<timeval>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = address

        let size = MemoryLayout<sockaddr_in>.size
        let raw = UnsafeMutableRawPointer.allocate(byteCount: size,
                                                    alignment: MemoryLayout<sockaddr_in>.alignment)
        defer { raw.deallocate() }
        raw.storeBytes(of: addr, as: sockaddr_in.self)
        let connected = connect(fd, raw.assumingMemoryBound(to: sockaddr.self), socklen_t(size))
        if connected != 0 {
            Darwin.close(fd)
        }
        try #require(connected == 0)
        self.descriptor = fd
    }

    /// Writes all of `bytes`, stopping early only if the server has gone away.
    func send(_ bytes: [UInt8]) {
        var sent = 0
        while sent < bytes.count {
            let written = bytes[sent...].withUnsafeBufferPointer { buffer -> Int in
                guard let base = buffer.baseAddress else { return -1 }
                return Darwin.write(descriptor, base, buffer.count)
            }
            if written <= 0 { return }
            sent += written
        }
    }

    /// Reads until the server closes the connection.
    func readToEnd() -> String {
        var received: [UInt8] = []
        var count = receiveOnce(into: &received)
        while count > 0 {
            count = receiveOnce(into: &received)
        }
        return String(decoding: received, as: UTF8.self)
    }

    /// Appends one read's worth of bytes; returns what `read(2)` returned.
    private func receiveOnce(into received: inout [UInt8]) -> Int {
        let chunk = [UInt8](unsafeUninitializedCapacity: Self.chunkBytes) { buffer, initialized in
            guard let base = buffer.baseAddress else { return }
            initialized = max(0, Darwin.read(descriptor, base, buffer.count))
        }
        received.append(contentsOf: chunk)
        return chunk.count
    }

    func close() {
        Darwin.close(descriptor)
    }
}

/// The exact bytes the server is expected to write, built independently of the server.
enum Expected {

    /// A refusal: status line, JSON error body, and `Connection: close`.
    ///
    /// - Parameters:
    ///   - status: Status code and reason phrase, e.g. `401 Unauthorized`.
    ///   - message: The error message.
    ///   - type: The error type; `invalid_request_error` unless stated.
    ///   - headers: Header lines that follow `Connection: close`, in order.
    static func refusal(
        _ status: String,
        _ message: String,
        type: String = "invalid_request_error",
        headers: [String] = []
    ) -> String {
        let body = #"{"error":{"message":"\#(message)","type":"\#(type)"}}"#
        var lines = [
            "HTTP/1.1 \(status)",
            "Content-Type: application/json",
            "Content-Length: \(body.utf8.count)",
            "Connection: close",
        ]
        lines.append(contentsOf: headers)
        return lines.joined(separator: "\r\n") + "\r\n\r\n" + body
    }

    /// The stub handler's whole response: stream headers, then `[DONE]`.
    ///
    /// - Parameter headers: Header lines that follow `Connection: close`, in order.
    static func stream(headers: [String] = []) -> String {
        var lines = [
            "HTTP/1.1 200 OK",
            "Content-Type: text/event-stream",
            "Cache-Control: no-cache",
            "Connection: close",
        ]
        lines.append(contentsOf: headers)
        return lines.joined(separator: "\r\n") + "\r\n\r\ndata: [DONE]\n\n"
    }

    static let unauthorized = refusal("401 Unauthorized", "A valid bearer credential is required.",
                                      type: "authentication_error",
                                      headers: ["WWW-Authenticate: Bearer"])
    static let wrongHost = refusal("421 Misdirected Request",
                                   "The Host header does not name this server.")
    static let notFound = refusal("404 Not Found", "No such route.")
    static let methodNotAllowed = refusal("405 Method Not Allowed", "Method not allowed; use POST.",
                                          headers: ["Allow: POST"])
    static let timedOut = refusal("408 Request Timeout",
                                  "The request was not received within the read deadline.")
    static let malformed = refusal("400 Bad Request", "Malformed HTTP request.")
    static let notJSON = refusal("400 Bad Request", "Request body is not a JSON object.")
    /// The connection limit's refusal, with the default limits' `Retry-After`.
    static let busy = refusal("503 Service Unavailable", "The server is at its connection limit.",
                              type: "server_error", headers: ["Retry-After: 30"])

    /// What a request is told when its turn did not come within the queue deadline.
    static func queueTimedOut(retryAfter seconds: Int) -> String {
        refusal("503 Service Unavailable",
                "The server is busy with another request and could not start this one in time. Try again later.",
                type: "server_error", headers: ["Retry-After: \(seconds)"])
    }
}
