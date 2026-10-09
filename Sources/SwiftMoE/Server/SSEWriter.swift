import Foundation

/// Server-Sent Events (SSE) writer for streaming token generation.
///
/// Implements the OpenAI-compatible `/v1/chat/completions` SSE format:
/// ```
/// data: {"id":"req_xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"token"}}]}
///
/// ```
///
/// Each token is sent as a separate SSE event. The stream ends with `data: [DONE]`.
public struct SSEWriter: Sendable {
    private let fileDescriptor: Int32
    private let requestID: String
    /// Header lines the server adds for this request — the CORS headers for an allowed origin.
    private let extraHeaders: [String]

    /// Creates an SSE writer for a client connection.
    ///
    /// - Parameters:
    ///   - fileDescriptor: The client socket fd to write to.
    ///   - requestID: Unique request ID for the SSE events.
    public init(fileDescriptor: Int32, requestID: String = UUID().uuidString) {
        self.init(fileDescriptor: fileDescriptor, requestID: requestID, extraHeaders: [])
    }

    /// Creates an SSE writer whose response carries further header lines.
    ///
    /// - Parameters:
    ///   - fileDescriptor: The client socket fd to write to.
    ///   - requestID: Unique request ID for the SSE events.
    ///   - extraHeaders: Header lines, without line endings, sent after the fixed ones.
    init(fileDescriptor: Int32, requestID: String = UUID().uuidString, extraHeaders: [String]) {
        self.fileDescriptor = fileDescriptor
        self.requestID = requestID
        self.extraHeaders = extraHeaders
    }

    /// Sends the HTTP response headers for an SSE stream.
    ///
    /// No `Access-Control-Allow-Origin` is sent unless the request came from an origin on the
    /// server's allowlist, in which case that one origin is echoed. There is no wildcard.
    public func sendHeaders() {
        var lines = [
            "HTTP/1.1 200 OK",
            "Content-Type: text/event-stream",
            "Cache-Control: no-cache",
            "Connection: close",
        ]
        lines.append(contentsOf: extraHeaders)
        writeString(lines.joined(separator: "\r\n") + "\r\n\r\n")
    }

    /// Sends a single token as an SSE delta event.
    ///
    /// - Parameter token: The token text to stream.
    /// - Returns: `true` if the write succeeded, `false` if the client disconnected.
    @discardableResult
    public func sendDelta(token: String) -> Bool {
        let escaped = jsonEscape(token)
        let chunk = """
        data: {"id":"req_\(requestID)","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"\(escaped)"},"finish_reason":null}]}\n\n
        """
        return writeString(chunk)
    }

    /// Sends the final `[DONE]` event to close the stream.
    public func sendDone() {
        writeString("data: [DONE]\n\n")
    }

    /// Sends a finish reason (stop, length, etc.) before [DONE].
    public func sendFinish(reason: String = "stop") {
        let chunk = """
        data: {"id":"req_\(requestID)","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"\(jsonEscape(reason))"}]}\n\n
        """
        writeString(chunk)
    }

    /// Whether the client has gone away: closed the connection, shut its sending side, or failed.
    ///
    /// A handler should ask before each unit of work it cannot take back — the server passes
    /// this to ``TokenGenerator`` as `shouldContinue` — because a write only fails after the
    /// fact, and prefill writes nothing at all. Asking costs one `poll(2)` and one `recv(2)`,
    /// and discards anything the client sent after its request.
    ///
    /// A client that shuts only its sending side cannot be told from one that has left, and
    /// is counted as gone. A client must keep its side open until it has its answer.
    public var clientHasDisconnected: Bool {
        clientHasDisconnected(within: .zero)
    }

    /// Whether the client has gone away, waiting up to `wait` for it to.
    ///
    /// - Parameter wait: How long to wait; zero only looks.
    /// - Returns: `true` once the client is gone, `false` if it is still there when the wait ends.
    func clientHasDisconnected(within wait: Duration) -> Bool {
        ClientConnection.peerHasClosed(descriptor: fileDescriptor, waitingUpTo: wait)
    }

    // MARK: - Private

    @discardableResult
    private func writeString(_ s: String) -> Bool {
        ClientConnection.write(Array(s.utf8), to: fileDescriptor)
    }

    /// First code point that JSON allows unescaped inside a string.
    private static let firstUnescapedScalar: UInt32 = 0x20
    private static let hexRadix = 16
    /// Hex digits in a `\uXXXX` escape.
    private static let unicodeEscapeDigits = 4

    /// Escapes text for the inside of a JSON string.
    ///
    /// Quotes, backslashes and every control character below U+0020 are escaped, as RFC 8259
    /// requires. A model emits whatever its vocabulary holds; an unescaped control character
    /// makes the event unparseable, and an unescaped line feed would end the SSE line early.
    private func jsonEscape(_ s: String) -> String {
        var result = ""
        result.reserveCapacity(s.utf8.count)
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": result += "\\\""
            case "\\": result += "\\\\"
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            case _ where scalar.value < Self.firstUnescapedScalar:
                let digits = String(scalar.value, radix: Self.hexRadix)
                result += "\\u" + String(repeating: "0", count: max(0, Self.unicodeEscapeDigits - digits.count)) + digits
            default: result.unicodeScalars.append(scalar)
            }
        }
        return result
    }
}
