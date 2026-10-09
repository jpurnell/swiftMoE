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
        data: {"id":"req_\(requestID)","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"\(reason)"}]}\n\n
        """
        writeString(chunk)
    }

    // MARK: - Private

    @discardableResult
    private func writeString(_ s: String) -> Bool {
        let data = Array(s.utf8)
        var written = 0
        while written < data.count {
            let n = data[written...].withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return -1 }
                return Darwin.write(fileDescriptor, base, buf.count)
            }
            if n <= 0 { return false }
            written += n
        }
        return true
    }

    private func jsonEscape(_ s: String) -> String {
        var result = ""
        result.reserveCapacity(s.count)
        for c in s {
            switch c {
            case "\"": result += "\\\""
            case "\\": result += "\\\\"
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            default: result.append(c)
            }
        }
        return result
    }
}
