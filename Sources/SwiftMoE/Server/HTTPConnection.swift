import Foundation
#if canImport(os)
import os
#endif

private let logger = Logger(subsystem: "com.swiftmoe", category: "server.connection")

/// The request handler and the queue for the one slot it runs in.
///
/// Connections are read concurrently, but the handler drives one model on one GPU and is not
/// written to be re-entered, so it runs one request at a time, in the order requests finished
/// arriving. ``InferenceQueue`` decides whose turn it is.
// Justification: `handler` is only invoked through `run`, by the one caller `queue` has admitted, so it never runs on two threads at once; `queue` is itself Sendable.
struct SerializedHandler: @unchecked Sendable {
    private let handler: HTTPServer.RequestHandler
    /// Admission to the handler.
    let queue = InferenceQueue()

    init(_ handler: @escaping HTTPServer.RequestHandler) {
        self.handler = handler
    }

    /// Calls the handler and then gives up the slot.
    ///
    /// - Precondition: ``queue`` admitted the caller, which has not yet left.
    func run(_ request: HTTPServer.Request, writer: SSEWriter) {
        defer { queue.leave() }
        handler(request, writer)
    }
}

/// Serves one accepted connection: read, decide, and either refuse or call the handler.
///
/// The order of decisions is the point of this type. Nothing a request says about itself is
/// acted on until the checks before it have passed:
///
/// 1. The head arrives within the read deadline and the header limit (408, 431).
/// 2. It parses as a request line and header fields (400), in a supported version (505).
/// 3. On a loopback bind, `Host` names this server (421).
/// 4. An `Origin`, if sent, is on the allowlist (403); a preflight for it is answered here.
/// 5. The bearer credential matches (401).
/// 6. Method and path are the route (404, 405).
/// 7. The declared body length is present and within the limit (411, 501, 413).
/// 8. The body arrives within the same deadline (408, 400) and is a JSON object (400).
/// 9. The token count is a whole number in range (400).
/// 10. The prompt, counted by the server's tokenizer, and the token count together fit in a
///     sequence (400).
///
/// Only then does the request wait its turn for the model — no longer than
/// ``HTTPServer/Limits/queueDeadline`` (503), and not at all once its client has gone — and
/// the handler run. The body is not read until step 8, so an unauthenticated
/// caller can make the server read at most ``HTTPServer/Limits/maxHeaderBytes``.
struct HTTPConnectionProcessor: Sendable {

    /// The one route this server answers.
    static let route = "/v1/chat/completions"
    private static let supportedVersions: Set<String> = ["HTTP/1.0", "HTTP/1.1"]
    /// Most digits accepted in `Content-Length`; 18 digits always fit in an `Int`.
    private static let maxContentLengthDigits = 18
    /// Size of the scratch buffer a refused upload is discarded through.
    private static let drainChunkBytes = 16 * 1024
    private static let headTerminator: [UInt8] = [0x0D, 0x0A, 0x0D, 0x0A]

    let authentication: HTTPServer.Authentication
    let allowedOrigins: [String]
    let limits: HTTPServer.Limits
    let tokenizer: HTTPServer.Tokenizer
    let observer: HTTPServerObserver?
    let handler: SerializedHandler

    // MARK: - Entry points

    /// Serves the connection on `descriptor`, then closes it.
    ///
    /// - Parameters:
    ///   - descriptor: An accepted client socket. Ownership passes to this call.
    ///   - bound: The listener's address, for the `Host` check.
    func serve(_ descriptor: Int32, bound: HTTPServer.BoundAddress) {
        defer { close(descriptor) }
        let socket = ClientConnection(descriptor: descriptor, writeDeadline: limits.writeDeadline)
        let deadline = ContinuousClock.now + limits.readDeadline

        let received: ReceivedHead
        switch readHead(socket, deadline: deadline) {
        case .success(let head):
            received = head
        case .failure(let failure):
            refuse(socket, failure.refusal, corsHeaders: [], unread: failure.unread)
            return
        }

        switch decide(received.head, bound: bound) {
        case .refuse(let refusal, let corsHeaders):
            var unread = Unread.none
            if let declared = received.declaredLength {
                unread = .bytes(max(0, declared - received.bodyPrefix.count))
            }
            refuse(socket, refusal, corsHeaders: corsHeaders, unread: unread)
        case .preflight(let corsHeaders):
            socket.write(Self.preflightResponse(corsHeaders: corsHeaders))
        case .read(let length, let corsHeaders):
            complete(socket, received: received, length: length, deadline: deadline, corsHeaders: corsHeaders)
        }
    }

    /// Tells a connection beyond the limit that the server is busy, then closes it.
    ///
    /// - Parameter descriptor: An accepted client socket. Ownership passes to this call.
    func refuseBusy(_ descriptor: Int32) {
        defer { close(descriptor) }
        let socket = ClientConnection(descriptor: descriptor, writeDeadline: limits.writeDeadline)
        refuse(socket, .busy(retryAfter: limits.retryAfterSeconds), corsHeaders: [], unread: .unknown)
    }

    // MARK: - Reading

    /// A parsed head, the length it declared, and whatever body bytes arrived with it.
    private struct ReceivedHead {
        let head: HTTPRequestHead
        /// `Content-Length`, when the request carried a valid one.
        let declaredLength: Int?
        let bodyPrefix: [UInt8]
    }

    /// How much of a refused request may still be on its way.
    private enum Unread {
        /// Nothing: the whole request has been read.
        case none
        /// This many body bytes were declared and not yet read.
        case bytes(Int)
        /// The request could not be measured; read until the client stops.
        case unknown
    }

    private struct HeadFailure: Error {
        let refusal: HTTPRefusal
        let unread: Unread
    }

    private func readHead(_ socket: ClientConnection, deadline: ContinuousClock.Instant) -> Result<ReceivedHead, HeadFailure> {
        var buffer: [UInt8] = []
        var searchFrom = 0
        var terminator: Range<Int>?

        while terminator == nil {
            guard buffer.count < limits.maxHeaderBytes else {
                return .failure(HeadFailure(refusal: .headersTooLarge(limit: limits.maxHeaderBytes), unread: .unknown))
            }
            switch socket.read(upTo: limits.maxHeaderBytes - buffer.count, deadline: deadline) {
            case .bytes(let chunk):
                buffer.append(contentsOf: chunk)
            case .timedOut:
                return .failure(HeadFailure(refusal: .timedOut, unread: .none))
            case .closed:
                return .failure(HeadFailure(refusal: .malformed, unread: .none))
            }
            terminator = Self.firstRange(of: Self.headTerminator, in: buffer, from: searchFrom)
            searchFrom = max(0, buffer.count - (Self.headTerminator.count - 1))
        }

        guard let terminator, let head = HTTPRequestHead.parse(Array(buffer[..<terminator.lowerBound])) else {
            return .failure(HeadFailure(refusal: .malformed, unread: .unknown))
        }
        let bodyPrefix = Array(buffer[terminator.upperBound...])

        var declaredLength: Int?
        if let text = head.headers["content-length"] {
            guard let length = Self.contentLength(text) else {
                return .failure(HeadFailure(refusal: .malformed, unread: .unknown))
            }
            declaredLength = length
        }
        return .success(ReceivedHead(head: head, declaredLength: declaredLength, bodyPrefix: bodyPrefix))
    }

    /// Reads the body, validates it, and calls the handler.
    private func complete(
        _ socket: ClientConnection,
        received: ReceivedHead,
        length: Int,
        deadline: ContinuousClock.Instant,
        corsHeaders: [String]
    ) {
        var body = Array(received.bodyPrefix.prefix(length))
        while body.count < length {
            switch socket.read(upTo: length - body.count, deadline: deadline) {
            case .bytes(let chunk):
                body.append(contentsOf: chunk)
            case .timedOut:
                refuse(socket, .timedOut, corsHeaders: corsHeaders, unread: .none)
                return
            case .closed:
                refuse(socket, .truncatedBody, corsHeaders: corsHeaders, unread: .none)
                return
            }
        }

        switch ChatRequest.parse(body, limits: limits) {
        case .failure(let rejection):
            refuse(socket, rejection.refusal, corsHeaders: corsHeaders, unread: .none)
        case .success(let chat):
            let promptTokens = tokenizer(chat.prompt)
            // `maxTokens` is at most `maxSequenceTokens`, so the subtraction cannot go below zero.
            guard promptTokens.count <= limits.maxSequenceTokens - chat.maxTokens else {
                let refusal = HTTPRefusal.sequenceTooLong(promptTokens: promptTokens.count,
                                                          completionTokens: chat.maxTokens,
                                                          limit: limits.maxSequenceTokens)
                refuse(socket, refusal, corsHeaders: corsHeaders, unread: .none)
                return
            }
            let request = HTTPServer.Request(prompt: chat.prompt, promptTokens: promptTokens,
                                             maxTokens: chat.maxTokens)
            run(request, on: socket, corsHeaders: corsHeaders)
        }
    }

    /// Waits for the model, within the queue deadline, and calls the handler.
    ///
    /// The wait ends one of three ways. Admitted: the handler runs. Timed out: the client is
    /// told 503 with `Retry-After`, through the same drain as every other refusal, since a
    /// client that has waited this long may well have sent more. Abandoned: the client hung
    /// up while waiting, so there is nobody to answer and nothing is run on its behalf.
    private func run(_ request: HTTPServer.Request, on socket: ClientConnection, corsHeaders: [String]) {
        let admission = handler.queue.enter(
            deadline: ContinuousClock.now + limits.queueDeadline,
            isAbandoned: { socket.peerHasClosed() },
            onQueued: { observer?(.requestQueued) })

        switch admission {
        case .timedOut:
            observer?(.queuedRequestTimedOut)
            refuse(socket, .queueTimedOut(retryAfter: limits.retryAfterSeconds),
                   corsHeaders: corsHeaders, unread: .unknown)
        case .abandoned:
            logger.info("[server] client left while queued; request dropped")
            observer?(.queuedRequestAbandoned)
        case .admitted:
            let writer = SSEWriter(fileDescriptor: socket.descriptor, extraHeaders: corsHeaders)
            handler.run(request, writer: writer)
            // The model is free again; only this connection waits for its client to finish.
            finish(socket, unread: .unknown)
        }
    }

    // MARK: - Deciding

    private enum Decision {
        case refuse(HTTPRefusal, corsHeaders: [String])
        case preflight(corsHeaders: [String])
        /// Read `length` body bytes and go on to the handler.
        case read(length: Int, corsHeaders: [String])
    }

    private func decide(_ head: HTTPRequestHead, bound: HTTPServer.BoundAddress) -> Decision {
        guard Self.supportedVersions.contains(head.version) else {
            return .refuse(.versionNotSupported, corsHeaders: [])
        }
        if HTTPServer.isLoopback(bound.host),
           !Self.hostNamesServer(head.headers["host"], bound: bound) {
            return .refuse(.wrongHost, corsHeaders: [])
        }

        // With an allowlist the response depends on Origin, so caches are told so on every response.
        var corsHeaders = allowedOrigins.isEmpty ? [] : ["Vary: Origin"]
        if let origin = head.headers["origin"] {
            guard allowedOrigins.contains(origin) else {
                return .refuse(.originNotAllowed, corsHeaders: corsHeaders)
            }
            corsHeaders.insert("Access-Control-Allow-Origin: \(origin)", at: 0)
            if head.method == "OPTIONS" {
                // A preflight carries no credential by design, so it is answered before the key check.
                return head.path == Self.route
                    ? .preflight(corsHeaders: corsHeaders)
                    : .refuse(.notFound, corsHeaders: corsHeaders)
            }
        }

        if case .bearer(let credential) = authentication,
           !credential.accepts(authorization: head.headers["authorization"]) {
            return .refuse(.unauthorized, corsHeaders: corsHeaders)
        }

        guard head.path == Self.route else {
            return .refuse(.notFound, corsHeaders: corsHeaders)
        }
        guard head.method == "POST" else {
            return .refuse(.methodNotAllowed, corsHeaders: corsHeaders)
        }
        guard head.headers["transfer-encoding"] == nil else {
            return .refuse(.transferEncoding, corsHeaders: corsHeaders)
        }
        guard let text = head.headers["content-length"], let length = Self.contentLength(text) else {
            return .refuse(.lengthRequired, corsHeaders: corsHeaders)
        }
        guard length <= limits.maxBodyBytes else {
            return .refuse(.bodyTooLarge(limit: limits.maxBodyBytes), corsHeaders: corsHeaders)
        }
        return .read(length: length, corsHeaders: corsHeaders)
    }

    /// Whether a `Host` header names a loopback listener.
    ///
    /// A page on `evil.example` whose DNS answer is switched to `127.0.0.1` reaches this server
    /// with the browser's blessing, but it still sends `Host: evil.example`. Only the bound
    /// address and `localhost` are accepted, and only with the bound port.
    static func hostNamesServer(_ header: String?, bound: HTTPServer.BoundAddress) -> Bool {
        guard let header else { return false }
        let parts = header.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2, parts[1] == String(bound.port) else { return false }
        let name = parts[0].lowercased()
        return name == bound.host || name == "localhost"
    }

    /// Parses `Content-Length`: decimal digits only, and few enough to fit an `Int`.
    static func contentLength(_ text: String) -> Int? {
        guard !text.isEmpty, text.utf8.count <= maxContentLengthDigits,
              text.utf8.allSatisfy({ $0 >= 0x30 && $0 <= 0x39 }) else {
            return nil
        }
        return Int(text)
    }

    private static func firstRange(of pattern: [UInt8], in buffer: [UInt8], from start: Int) -> Range<Int>? {
        guard buffer.count >= pattern.count else { return nil }
        var index = start
        while index <= buffer.count - pattern.count {
            if buffer[index..<(index + pattern.count)].elementsEqual(pattern) {
                return index..<(index + pattern.count)
            }
            index += 1
        }
        return nil
    }

    // MARK: - Answering

    private static func preflightResponse(corsHeaders: [String]) -> [UInt8] {
        var lines = [
            "HTTP/1.1 204 No Content",
            "Content-Length: 0",
            "Connection: close",
            "Access-Control-Allow-Methods: POST",
            "Access-Control-Allow-Headers: Authorization, Content-Type",
        ]
        lines.append(contentsOf: corsHeaders)
        return Array((lines.joined(separator: "\r\n") + "\r\n\r\n").utf8)
    }

    /// Writes a refusal, then discards whatever the client is still sending before returning.
    ///
    /// Returning closes the socket. Closing it while the client is mid-upload makes the kernel
    /// answer the rest of the upload with a reset, and a reset discards the response the client
    /// has not read yet — so the status would be written and never seen. The write side is shut
    /// first, so the client sees the complete response and end-of-stream at once; then the
    /// upload is read into a fixed scratch buffer and dropped until it ends, the declared
    /// length has gone by, or ``HTTPServer/Limits/refusalDrainDeadline`` passes.
    private func refuse(_ socket: ClientConnection, _ refusal: HTTPRefusal, corsHeaders: [String], unread: Unread) {
        logger.info("[server] refused: \(refusal.status, privacy: .public)")
        socket.write(refusal.response(corsHeaders: corsHeaders))
        finish(socket, unread: unread)
    }

    /// Ends a response so that the client receives all of it: shuts the write side, then
    /// discards what the client is still sending, for at most
    /// ``HTTPServer/Limits/refusalDrainDeadline``.
    ///
    /// A completed stream ends this way as well as a refusal. The request ended at
    /// `Content-Length`, but a client may have sent more, and closing over unread bytes resets
    /// the connection and takes the end of the stream with it.
    private func finish(_ socket: ClientConnection, unread: Unread) {
        socket.finishWriting()

        var remaining: Int?
        switch unread {
        case .none:
            return
        case .bytes(let count):
            guard count > 0 else { return }
            remaining = count
        case .unknown:
            remaining = nil
        }

        let deadline = ContinuousClock.now + limits.refusalDrainDeadline
        while remaining.map({ $0 > 0 }) ?? true {
            let want = min(Self.drainChunkBytes, remaining ?? Self.drainChunkBytes)
            guard case .bytes(let chunk) = socket.read(upTo: want, deadline: deadline) else { return }
            remaining = remaining.map { $0 - chunk.count }
        }
    }
}

/// The parts of a chat-completions body this server acts on.
struct ChatRequest: Equatable, Sendable {

    /// Content of the last message, or `""` when there is none.
    let prompt: String
    /// Tokens to generate, already checked against the ceiling.
    let maxTokens: Int

    /// Why a body was refused.
    struct Rejection: Error, Equatable {
        let refusal: HTTPRefusal
    }

    /// The two token-count fields. Decoded as `Double` so that `64.0` is accepted, `1.5` can be
    /// refused by name, and a boolean is a type mismatch rather than a 1.
    private struct TokenFields: Decodable {
        let maxTokens: Double?
        let maxCompletionTokens: Double?

        enum CodingKeys: String, CodingKey {
            case maxTokens = "max_tokens"
            case maxCompletionTokens = "max_completion_tokens"
        }
    }

    /// Validates a request body.
    ///
    /// Both token fields are checked when both are present; `max_completion_tokens` is the one
    /// used, as before. A count outside `1...maxCompletionTokens` is refused, never clamped.
    ///
    /// - Parameters:
    ///   - body: The request body.
    ///   - limits: The server's limits.
    /// - Returns: The request, or the refusal to send.
    static func parse(_ body: [UInt8], limits: HTTPServer.Limits) -> Result<ChatRequest, Rejection> {
        let data = Data(body)
        let object: [String: Any]
        do {
            guard let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return .failure(Rejection(refusal: .notJSON))
            }
            object = parsed
        } catch {
            logger.debug("Request body is not JSON: \(error.localizedDescription, privacy: .public)")
            return .failure(Rejection(refusal: .notJSON))
        }

        let fields: TokenFields
        do {
            fields = try JSONDecoder().decode(TokenFields.self, from: data)
        } catch let error as DecodingError {
            logger.debug("Token count refused: \(error.localizedDescription, privacy: .public)")
            return .failure(Rejection(refusal: tokenRefusal(for: error, limits: limits)))
        } catch {
            logger.debug("Token count undecodable: \(error.localizedDescription, privacy: .public)")
            return .failure(Rejection(refusal: .notJSON))
        }

        var maxTokens = limits.defaultCompletionTokens
        let candidates: [(String, Double?)] = [
            (TokenFields.CodingKeys.maxTokens.rawValue, fields.maxTokens),
            (TokenFields.CodingKeys.maxCompletionTokens.rawValue, fields.maxCompletionTokens),
        ]
        for (field, value) in candidates {
            guard let value else { continue }
            guard let count = wholeNumber(value), count >= 1, count <= limits.maxCompletionTokens else {
                return .failure(Rejection(refusal: .tokensOutOfRange(field: field,
                                                                    maximum: limits.maxCompletionTokens)))
            }
            maxTokens = count
        }

        let messages = object["messages"] as? [[String: Any]]
        let prompt = messages?.last?["content"] as? String ?? ""
        return .success(ChatRequest(prompt: prompt, maxTokens: maxTokens))
    }

    /// The integer a JSON number denotes, or `nil` when it is fractional, non-finite, or too large.
    private static func wholeNumber(_ value: Double) -> Int? {
        guard value.isFinite, value == value.rounded(.towardZero) else { return nil }
        return Int(exactly: value)
    }

    private static func tokenRefusal(for error: DecodingError, limits: HTTPServer.Limits) -> HTTPRefusal {
        let path: [any CodingKey]
        switch error {
        case .typeMismatch(_, let context), .valueNotFound(_, let context), .dataCorrupted(let context):
            path = context.codingPath
        case .keyNotFound(_, let context):
            path = context.codingPath
        @unknown default:
            path = []
        }
        guard let field = path.first?.stringValue else { return .notJSON }
        return .tokensOutOfRange(field: field, maximum: limits.maxCompletionTokens)
    }
}
