import Foundation

extension HTTPServer {

    /// The bounds one request and one connection are held to.
    ///
    /// Every number a caller could otherwise choose — how many tokens to generate, how many
    /// bytes to send, how long to take sending them — has a ceiling here, and a request over a
    /// ceiling is refused with a status and a sentence rather than trimmed to fit. A caller who
    /// asked for 20,000 tokens and silently received 8,192 has been given a different answer
    /// from the one requested, with nothing to say so; a 400 that names the range can be acted on.
    ///
    /// ```swift
    /// var limits = HTTPServer.Limits()
    /// limits.maxCompletionTokens = 2048
    /// print(limits.maxBodyBytes)   // 65536
    /// ```
    public struct Limits: Sendable, Equatable {

        /// Most tokens one request may ask for. Default ``TokenGenerator/defaultMaxSequenceLength``.
        ///
        /// It may not exceed ``maxSequenceTokens``, and a request is also held to that limit
        /// with its prompt counted in, so a count this large only fits beside an empty prompt.
        /// At the measured 4.4 tokens per second the default is already half an hour of GPU
        /// time for one request.
        public var maxCompletionTokens: Int = TokenGenerator.defaultMaxSequenceLength

        /// Most positions one request may occupy, prompt and completion together. Default
        /// ``TokenGenerator/defaultMaxSequenceLength``.
        ///
        /// Set it to the capacity of the generator behind the handler
        /// (``TokenGenerator/maxSequenceLength``). The prompt is counted with the server's
        /// ``HTTPServer/Tokenizer``; a request whose prompt tokens plus requested completion
        /// tokens come to more than this is refused with a 400 that gives all three numbers.
        /// The completion is not shortened to fit: that would be a different answer from the
        /// one asked for.
        public var maxSequenceTokens: Int = TokenGenerator.defaultMaxSequenceLength

        /// Tokens generated when a request names no count. Default 100, as before.
        public var defaultCompletionTokens: Int = 100

        /// Most bytes of request line and header fields. Default 16 KiB.
        ///
        /// A chat-completions request carries a handful of short fields; 16 KiB is twice the
        /// common 8 KiB server default, which leaves room for a long bearer key or a proxy's
        /// forwarding fields without making the head a place to park data.
        public var maxHeaderBytes: Int = 16 * 1024

        /// Most bytes of request body. Default 64 KiB.
        ///
        /// The server previously read into one 64 KiB buffer and silently ignored a body that
        /// did not fit, so this is the size that already worked — now a stated limit with a
        /// 413 on the far side of it. At the placeholder tokenizer's one token per byte it is
        /// also eight times the sequence the KV caches can hold.
        public var maxBodyBytes: Int = 64 * 1024

        /// Longest a client may take to deliver its whole request, head and body. Default 10 s.
        ///
        /// It is a budget for the request, not a gap between reads, so a client that sends one
        /// byte every nine seconds is dropped at ten like one that sends nothing.
        public var readDeadline: Duration = .seconds(10)

        /// Longest one write to a client may block. Default 30 s.
        ///
        /// Generation holds the model while it streams; a client that stops reading would
        /// otherwise hold it for good. A failed write ends the stream.
        public var writeDeadline: Duration = .seconds(30)

        /// Longest the server keeps discarding an upload it has already refused. Default 2 s.
        ///
        /// Closing a socket while the client is still sending turns into a reset on the client
        /// side, and the client never sees the status. So after a refusal the rest of the upload
        /// is read and thrown away — never buffered — until it ends or this much time has gone.
        public var refusalDrainDeadline: Duration = .seconds(2)

        /// Longest a validated request waits for the model before it is turned away. Default 30 s.
        ///
        /// One request runs at a time. The rest wait their turn, each holding a connection;
        /// a request whose turn has not come by this deadline is answered 503 with
        /// `Retry-After` and its connection is released. Thirty seconds is one default-sized
        /// completion — 100 tokens at the measured 4.4 per second is 23 s — so a request behind
        /// an ordinary one is served and a request behind a half-hour one is told promptly.
        public var queueDeadline: Duration = .seconds(30)

        /// Seconds a turned-away client is told to wait: ``queueDeadline`` rounded up to a
        /// whole second, and never less than one.
        public var retryAfterSeconds: Int {
            let milliseconds = ClientConnection.milliseconds(queueDeadline)
            let (rounded, overflow) = milliseconds.addingReportingOverflow(Self.millisecondsPerSecond - 1)
            let seconds = (overflow ? milliseconds : rounded) / Self.millisecondsPerSecond
            return Int(clamping: max(1, seconds))
        }

        private static let millisecondsPerSecond: Int64 = 1000

        /// Most connections being read at once. Default 16.
        ///
        /// Inference itself is one request at a time; this bounds how many clients may be
        /// connected at all — sending, waiting for the model, or being answered. One more is
        /// answered 503 with `Retry-After`. It is also what bounds the queue for the model:
        /// every waiting request is one of these connections, so at most `maxConnections - 1`
        /// can be waiting behind the one that is running.
        public var maxConnections: Int = 16

        /// Creates the default limits.
        public init() {}

        /// Checks that every limit is positive, the default token count fits the completion
        /// ceiling, and the completion ceiling fits the sequence.
        ///
        /// - Throws: ``HTTPServerError/invalidLimit(name:)`` naming the first limit that is not
        ///   positive; ``HTTPServerError/limitAboveLimit(name:ceiling:)`` naming the first that
        ///   is larger than the limit bounding it.
        func validate() throws {
            let counts: [(String, Int)] = [
                ("maxHeaderBytes", maxHeaderBytes),
                ("maxBodyBytes", maxBodyBytes),
                ("maxSequenceTokens", maxSequenceTokens),
                ("maxCompletionTokens", maxCompletionTokens),
                ("defaultCompletionTokens", defaultCompletionTokens),
                ("maxConnections", maxConnections),
            ]
            for (name, value) in counts where value <= 0 {
                throw HTTPServerError.invalidLimit(name: name)
            }
            guard maxCompletionTokens <= maxSequenceTokens else {
                throw HTTPServerError.limitAboveLimit(name: "maxCompletionTokens", ceiling: "maxSequenceTokens")
            }
            guard defaultCompletionTokens <= maxCompletionTokens else {
                throw HTTPServerError.limitAboveLimit(name: "defaultCompletionTokens",
                                                      ceiling: "maxCompletionTokens")
            }
            let deadlines: [(String, Duration)] = [
                ("readDeadline", readDeadline),
                ("writeDeadline", writeDeadline),
                ("refusalDrainDeadline", refusalDrainDeadline),
                ("queueDeadline", queueDeadline),
            ]
            for (name, value) in deadlines where value <= .zero {
                throw HTTPServerError.invalidLimit(name: name)
            }
        }
    }

    /// Whether `origin` is a serialized origin: `scheme://host[:port]`, nothing after it.
    ///
    /// `*` and `null` are not origins, so neither can be put on the allowlist: the first would
    /// be the wildcard this server used to send, and the second is what every sandboxed frame
    /// and `file:` page calls itself.
    ///
    /// - Parameter origin: Text given as `--allow-origin`.
    /// - Returns: `true` when the text can be compared, byte for byte, with an `Origin` header.
    public static func isOrigin(_ origin: String) -> Bool {
        guard let separator = origin.range(of: "://") else { return false }
        let scheme = origin[origin.startIndex..<separator.lowerBound]
        let authority = origin[separator.upperBound...]
        guard !scheme.isEmpty, scheme.utf8.allSatisfy(isSchemeByte) else { return false }
        guard !authority.isEmpty, authority.utf8.allSatisfy(isAuthorityByte) else { return false }
        return true
    }

    private static func isSchemeByte(_ byte: UInt8) -> Bool {
        (byte >= 0x61 && byte <= 0x7A) || (byte >= 0x30 && byte <= 0x39) || byte == 0x2B || byte == 0x2D || byte == 0x2E
    }

    /// Lower-case letters, digits, and `. - : [ ]` — a host, an optional port, no path.
    private static func isAuthorityByte(_ byte: UInt8) -> Bool {
        (byte >= 0x61 && byte <= 0x7A) || (byte >= 0x30 && byte <= 0x39)
            || byte == 0x2E || byte == 0x2D || byte == 0x3A || byte == 0x5B || byte == 0x5D
    }
}
