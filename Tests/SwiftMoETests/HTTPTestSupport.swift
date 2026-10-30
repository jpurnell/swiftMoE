import Foundation
import os
import SwiftHTTPService
import Testing
@testable import SwiftMoE
@testable import SwiftMoEHTTP

/// Every test that drives a real server over a socket, run one at a time.
///
/// Swift Testing runs a synchronous test on a thread of the cooperative pool. A test that blocks
/// that thread reading from a socket needs the server's tasks to get a pool thread, so enough of
/// them at once leave the server none and every test waits for every other. One at a time leaves
/// the rest of the pool to the server.
@Suite("HTTP service", .serialized)
enum HTTPServiceTests {}

/// One call the server made to its request handler.
struct HandlerCall: Equatable, Sendable {
    let prompt: String
    let maxTokens: Int
}

/// What the stub does once the service has recorded a call: the service's own generator
/// contract, so a stub produces tokens the way the real generator does.
typealias StubResponse = ChatService.Generate

/// The events a server under test reported, and a way to wait for the next one of a kind.
///
/// Tests that need the server to have reached a state — a request queued, a slot released —
/// wait here for the server to say so, not for an interval after which it probably has.
final class ServerEvents: Sendable {

    /// How long a wait lasts before the test reports that the event never came.
    private static let allowance: DispatchTimeInterval = .seconds(10)

    private let log = OSAllocatedUnfairLock<[ChatServiceEvent]>(initialState: [])
    private let signals: [ChatServiceEvent: DispatchSemaphore]

    init() {
        var signals: [ChatServiceEvent: DispatchSemaphore] = [:]
        for event in ChatServiceEvent.allCases {
            signals[event] = DispatchSemaphore(value: 0)
        }
        self.signals = signals
    }

    func record(_ event: ChatServiceEvent) {
        log.withLock { $0.append(event) }
        signals[event]?.signal()
    }

    /// Waits for one occurrence of `event` that no earlier call has consumed.
    ///
    /// - Returns: `false`, having recorded an issue, when none arrived within the allowance.
    @discardableResult
    func awaitNext(_ event: ChatServiceEvent, sourceLocation: SourceLocation = #_sourceLocation) -> Bool {
        guard let signal = signals[event], signal.wait(timeout: .now() + Self.allowance) == .success else {
            Issue.record("the server never reported \(event)", sourceLocation: sourceLocation)
            return false
        }
        return true
    }

    /// How many times `event` has been reported.
    func count(of event: ChatServiceEvent) -> Int {
        log.withLock { $0.filter { $0 == event }.count }
    }
}

/// A real ``ChatService`` on a loopback ephemeral port.
///
/// The generator is a stub: it records what it was called with and produces the tokens a test
/// gives it, so no model is needed to see whether a request got as far as inference.
final class RunningServer: Sendable {

    /// The key the server is configured with unless a test says otherwise.
    static let key = String(repeating: "s", count: 48)
    /// A well-formed key that is not the server's.
    static let wrongKey = String(repeating: "w", count: 48)

    let service: ChatService
    let bound: ChatService.BoundAddress
    /// What the service has reported about its requests.
    let events = ServerEvents()
    private let calls = OSAllocatedUnfairLock<[HandlerCall]>(initialState: [])
    private let requests = OSAllocatedUnfairLock<[ChatCompletionRequest]>(initialState: [])

    /// The tokenizer the server is given unless a test says otherwise: one token per UTF-8 byte.
    static let byteTokenizer: ChatService.Tokenizer = { prompt in prompt.utf8.map { Int($0) } }

    /// The stub's whole answer: no tokens, so the stream is the finish chunk and `[DONE]`.
    static let emptyStream: StubResponse = { _, _, _ in }

    /// - Parameters:
    ///   - authenticated: Whether the server requires ``key``.
    ///   - allowedOrigins: The origin allowlist.
    ///   - limits: The server's limits.
    ///   - tokenizer: How the server counts a prompt.
    ///   - respond: What the generator does after recording the call.
    init(
        authenticated: Bool = true,
        allowedOrigins: [String] = [],
        limits: InferenceLimits = InferenceLimits(),
        tls: TLSFiles? = nil,
        tokenizer: @escaping ChatService.Tokenizer = RunningServer.byteTokenizer,
        respond: @escaping StubResponse = RunningServer.emptyStream
    ) throws {
        let authentication: Authentication = authenticated
            ? .bearer(BearerCredential(key: try APIKey(Self.key)))
            : .unauthenticatedLoopback
        let calls = self.calls
        let requests = self.requests
        let events = self.events
        let service = ChatService(host: "127.0.0.1", port: 0, authentication: authentication,
                                  allowedOrigins: allowedOrigins, limits: limits, tls: tls,
                                  tokenizer: tokenizer,
                                  observer: { events.record($0) },
                                  accessLog: { _ in }) { request, shouldContinue, emit in
            calls.withLock { $0.append(HandlerCall(prompt: request.prompt, maxTokens: request.maxTokens)) }
            requests.withLock { $0.append(request) }
            try respond(request, shouldContinue, emit)
        }
        self.service = service
        self.bound = try blocking { try await service.start() }
    }

    /// Everything the handler has been called with so far, in order.
    var handlerCalls: [HandlerCall] { calls.withLock { $0 } }

    /// The requests the handler received, in order, with the tokens the server counted.
    var handledRequests: [ChatCompletionRequest] { requests.withLock { $0 } }

    /// The port the service listens on.
    var port: UInt16 { UInt16(bound.port) }

    /// `127.0.0.1:<port>` — the authority a well-behaved local client sends as `Host`.
    var authority: String { "\(bound.host):\(bound.port)" }

    /// Stops the server and waits for it to finish.
    func shutdown() {
        let service = self.service
        blocking { await service.shutdown() }
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

    /// Sends `bytes` on a fresh connection and returns everything the server wrote back, in the
    /// canonical form ``Expected`` builds (see ``Canonical``).
    func exchange(_ bytes: [UInt8]) throws -> String {
        Canonical.normalise(try rawExchange(bytes))
    }

    /// Sends `bytes` on a fresh connection and returns everything the server wrote back, as written.
    func rawExchange(_ bytes: [UInt8]) throws -> String {
        let client = try ClientSocket(port: UInt16(bound.port))
        defer { client.close() }
        client.send(bytes)
        return client.readToEnd()
    }
}

/// A stub handler that holds the model, for any request whose prompt is `hold`, until told
/// to let go.
final class Hold: Sendable {
    /// Generous enough that reaching it means the test has hung.
    private static let allowance: DispatchTimeInterval = .seconds(30)

    private let started = DispatchSemaphore(value: 0)
    private let released = DispatchSemaphore(value: 0)

    var respond: StubResponse {
        { [self] request, shouldContinue, emit in
            if request.prompt == "hold" {
                started.signal()
                _ = released.wait(timeout: .now() + Self.allowance)
            }
            try RunningServer.emptyStream(request, shouldContinue, emit)
        }
    }

    /// Waits until the handler is holding the model.
    func awaitStarted() -> Bool { started.wait(timeout: .now() + Self.allowance) == .success }

    /// Lets the held request finish.
    func release() { released.signal() }
}

/// A blocking client socket for driving the server byte by byte.
struct ClientSocket {

    /// Seconds a client read may block before the test gives up on the server.
    private static let receiveAllowanceSeconds = 30
    private static let chunkBytes = 4096

    let descriptor: Int32

    /// `127.0.0.1` in network byte order.
    static var loopbackAddress: in_addr_t? {
        var address = in_addr()
        guard inet_pton(AF_INET, "127.0.0.1", &address) == 1 else { return nil }
        return address.s_addr
    }

    /// Whether a connection to `127.0.0.1:port` is accepted.
    static func canConnect(port: UInt16) -> Bool {
        guard let address = loopbackAddress else { return false }
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        defer { Darwin.close(fd) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = address
        let size = MemoryLayout<sockaddr_in>.size
        let raw = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: MemoryLayout<sockaddr_in>.alignment)
        defer { raw.deallocate() }
        raw.storeBytes(of: addr, as: sockaddr_in.self)
        return connect(fd, raw.assumingMemoryBound(to: sockaddr.self), socklen_t(size)) == 0
    }

    /// Wraps a socket that is already connected.
    init(descriptor: Int32) {
        self.descriptor = descriptor
    }

    init(port: UInt16) throws {
        let address = try #require(ClientSocket.loopbackAddress)
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

/// Where `blocking` runs its work: a pool of threads of its own.
///
/// Swift Testing runs synchronous tests on the cooperative pool, one thread each. A test that
/// blocks its thread waiting for async work that needs a pool thread deadlocks once every
/// pool thread is a waiting test, so the work runs here instead.
// Justification: the only state is an immutable dispatch queue, which is safe to share.
@available(macOS 15.0, *)
private final class BlockingExecutor: TaskExecutor, @unchecked Sendable {
    static let shared = BlockingExecutor()
    private let queue = DispatchQueue(label: "swiftmoe.tests.blocking", attributes: .concurrent)

    func enqueue(_ job: consuming ExecutorJob) {
        let unowned = UnownedJob(job)
        queue.async { unowned.runSynchronously(on: self.asUnownedTaskExecutor()) }
    }
}

private func startBlockingWork(_ work: @escaping @Sendable () async -> Void) {
    if #available(macOS 15.0, *) {
        Task(executorPreference: BlockingExecutor.shared) { await work() }
    } else {
        Task.detached { await work() }
    }
}

/// Runs `body` to completion from synchronous code. Test setup only: it blocks the calling thread.
func blocking<T: Sendable>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
    let box = OSAllocatedUnfairLock<Result<T, any Error>?>(initialState: nil)
    let done = DispatchSemaphore(value: 0)
    let work: @Sendable () async -> Void = {
        let result: Result<T, any Error>
        do { result = .success(try await body()) } catch { result = .failure(error) }
        box.withLock { $0 = result }
        done.signal()
    }
    startBlockingWork(work)
    _ = done.wait(timeout: .now() + .seconds(600))
    guard let result = box.withLock({ $0 }) else { throw CancellationError() }
    return try result.get()
}

/// Runs `body` to completion from synchronous code, for work that cannot throw.
func blocking(_ body: @escaping @Sendable () async -> Void) {
    let done = DispatchSemaphore(value: 0)
    startBlockingWork {
        await body()
        done.signal()
    }
    _ = done.wait(timeout: .now() + .seconds(600))
}

/// A response reduced to what a client sees, the same way whichever transport wrote it.
///
/// The transport owns `Date`, `Server`, and the framing of a streamed body (chunks); none of
/// them is the service's to promise. What is compared is the status line, the other header
/// fields (sorted: their order is the transport's), and the body de-chunked, with the random
/// per-stream id replaced by `req_<ID>`.
enum Canonical {

    private static let transportOwned: Set<String> = ["date", "server", "transfer-encoding"]

    static func normalise(_ raw: String) -> String {
        guard !raw.isEmpty else { return "" }
        var head = raw
        var body = ""
        if let range = raw.range(of: "\r\n\r\n") {
            head = String(raw[..<range.lowerBound])
            body = String(raw[range.upperBound...])
        }
        let lines = head.split(separator: "\r\n" as Character, omittingEmptySubsequences: false).map(String.init)
        let fields = Array(lines.dropFirst())
        let chunked = fields.contains { $0.lowercased() == "transfer-encoding: chunked" }
        if chunked { body = dechunk(body) }
        let kept = fields.filter { field in
            guard let name = field.split(separator: ":", maxSplits: 1).first else { return true }
            return !transportOwned.contains(name.lowercased())
        }
        return build(status: lines.first ?? "", headers: kept, body: body)
    }

    static func build(status: String, headers: [String], body: String) -> String {
        let sorted = headers.sorted()
        let id = body.replacingOccurrences(of: "req_[0-9A-Fa-f-]{36}", with: "req_<ID>", options: .regularExpression)
        return ([status] + sorted).joined(separator: "\r\n") + "\r\n\r\n" + id
    }

    /// The payload of a chunked body, as far as it goes: a stream that ends without its
    /// terminating chunk is returned up to the last complete chunk.
    static func dechunk(_ body: String) -> String {
        // Chunk sizes count bytes, and Swift counts a CR LF pair as one character: work on bytes.
        let bytes = Array(body.utf8)
        var out: [UInt8] = []
        var index = 0
        while let lineEnd = Self.crlf(in: bytes, from: index) {
            guard let size = Int(String(decoding: bytes[index..<lineEnd], as: UTF8.self), radix: 16), size > 0 else { break }
            let start = lineEnd + 2
            guard bytes.count - start >= size else { break }
            out.append(contentsOf: bytes[start..<(start + size)])
            index = start + size + 2
        }
        return String(decoding: out, as: UTF8.self)
    }

    private static func crlf(in bytes: [UInt8], from start: Int) -> Int? {
        var index = start
        while index + 1 < bytes.count {
            if bytes[index] == 0x0D, bytes[index + 1] == 0x0A { return index }
            index += 1
        }
        return nil
    }
}

/// The responses the server is expected to write, built independently of the server.
enum Expected {

    /// A refusal: status line, JSON error body, and `Connection: close`.
    ///
    /// - Parameters:
    ///   - status: Status code and reason phrase, e.g. `401 Unauthorized`.
    ///   - message: The error message.
    ///   - type: The error type; `invalid_request_error` unless stated.
    ///   - headers: Further header lines.
    static func refusal(
        _ status: String,
        _ message: String,
        type: String = "invalid_request_error",
        headers: [String] = []
    ) -> String {
        let body = #"{"error":{"message":"\#(message)","type":"\#(type)"}}"#
        return response(status, headers: [
            "Content-Type: application/json",
            "Content-Length: \(body.utf8.count)",
            "Connection: close",
        ] + headers, body: body)
    }

    /// Any response, in the canonical form.
    static func response(_ status: String, headers: [String], body: String) -> String {
        Canonical.build(status: "HTTP/1.1 \(status)", headers: headers, body: body)
    }

    /// The stub's whole answer: stream headers, the finish chunk, then `[DONE]`.
    ///
    /// - Parameter headers: Further header lines.
    static func stream(headers: [String] = []) -> String {
        response("200 OK", headers: [
            "Content-Type: text/event-stream",
            "Cache-Control: no-store",
            "Connection: close",
        ] + headers, body: finishAndDone)
    }

    /// The end of every completed stream.
    static let finishAndDone =
        #"data: {"id":"req_<ID>","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#
        + "\n\ndata: [DONE]\n\n"

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
