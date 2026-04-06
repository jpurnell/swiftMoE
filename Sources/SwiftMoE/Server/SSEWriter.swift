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
public struct SSEWriter {
    private let fileDescriptor: Int32
    private let requestID: String

    /// Creates an SSE writer for a client connection.
    ///
    /// - Parameters:
    ///   - fileDescriptor: The client socket fd to write to.
    ///   - requestID: Unique request ID for the SSE events.
    public init(fileDescriptor: Int32, requestID: String = UUID().uuidString) {
        self.fileDescriptor = fileDescriptor
        self.requestID = requestID
    }

    /// Sends the HTTP response headers for an SSE stream.
    public func sendHeaders() {
        let headers = """
        HTTP/1.1 200 OK\r
        Content-Type: text/event-stream\r
        Cache-Control: no-cache\r
        Connection: close\r
        Access-Control-Allow-Origin: *\r
        \r

        """
        writeString(headers)
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
            let n = data.withUnsafeBufferPointer { buf in
                Darwin.write(fileDescriptor, buf.baseAddress! + written, data.count - written)
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
