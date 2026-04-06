import Foundation

/// Minimal HTTP server for OpenAI-compatible chat completions with SSE streaming.
///
/// Listens on a TCP port and handles `/v1/chat/completions` POST requests.
/// Each request spawns token generation and streams results via SSE.
///
/// ## Protocol
/// - **Endpoint:** `POST /v1/chat/completions`
/// - **Request body:** OpenAI chat completion format (messages array)
/// - **Response:** Server-Sent Events with token deltas
///
/// Matches the server in `infer.m:5635-6500`.
public final class HTTPServer {

    /// Port to listen on.
    public let port: UInt16

    /// Called for each incoming chat request. Return the prompt string.
    public typealias RequestHandler = (
        _ prompt: String,
        _ maxTokens: Int,
        _ sseWriter: SSEWriter
    ) -> Void

    private let handler: RequestHandler
    private var serverFD: Int32 = -1

    /// Creates an HTTP server.
    ///
    /// - Parameters:
    ///   - port: TCP port to listen on (default 8080).
    ///   - handler: Callback invoked for each chat completion request.
    public init(port: UInt16 = 8080, handler: @escaping RequestHandler) {
        self.port = port
        self.handler = handler
    }

    /// Starts the server and blocks, accepting connections.
    ///
    /// This method does not return until the server is stopped or an error occurs.
    public func start() throws {
        serverFD = socket(AF_INET, SOCK_STREAM, 0)
        guard serverFD >= 0 else {
            throw FlashMoEError.readFailed(errno: errno, context: "socket()")
        }

        // Allow port reuse
        var opt: Int32 = 1
        setsockopt(serverFD, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))

        // Bind
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = INADDR_ANY.bigEndian

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                bind(serverFD, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            close(serverFD)
            throw FlashMoEError.readFailed(errno: errno, context: "bind(port: \(port))")
        }

        // Listen
        guard listen(serverFD, 5) == 0 else {
            close(serverFD)
            throw FlashMoEError.readFailed(errno: errno, context: "listen()")
        }

        print("[server] Listening on http://localhost:\(port)/v1/chat/completions")

        // Accept loop
        while true {
            var clientAddr = sockaddr_in()
            var addrLen = socklen_t(MemoryLayout<sockaddr_in>.size)

            let clientFD = withUnsafeMutablePointer(to: &clientAddr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                    accept(serverFD, sockPtr, &addrLen)
                }
            }

            guard clientFD >= 0 else { continue }

            handleConnection(clientFD)
            close(clientFD)
        }
    }

    /// Stops the server.
    public func stop() {
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
            while total < bufPtr.count - 1 {
                let n = Darwin.read(clientFD, bufPtr.baseAddress! + total, 1)
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
                        var bodyRead = 0
                        while bodyRead < contentLen {
                            let n = Darwin.read(clientFD, bufPtr.baseAddress! + total + bodyRead, contentLen - bodyRead)
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
            _ = response.withCString { cstr in
                Darwin.write(clientFD, cstr, strlen(cstr))
            }
            return
        }

        // Only handle POST /v1/chat/completions
        guard headerString.hasPrefix("POST") && headerString.contains("/v1/chat/completions") else {
            let response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
            _ = response.withCString { cstr in
                Darwin.write(clientFD, cstr, strlen(cstr))
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
        // Find the body (after \r\n\r\n)
        guard let bodyStart = request.range(of: "\r\n\r\n") else { return nil }
        let body = String(request[bodyStart.upperBound...])

        // Parse JSON
        guard let data = body.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let messages = json["messages"] as? [[String: Any]] else {
            return nil
        }

        // Return the last message's content
        return messages.last?["content"] as? String
    }

    /// Extracts max_tokens or max_completion_tokens from request body.
    private func extractMaxTokens(from request: String, default defaultVal: Int) -> Int {
        guard let bodyStart = request.range(of: "\r\n\r\n") else { return defaultVal }
        let body = String(request[bodyStart.upperBound...])

        guard let data = body.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return defaultVal
        }

        if let max = json["max_completion_tokens"] as? Int { return max }
        if let max = json["max_tokens"] as? Int { return max }
        return defaultVal
    }
}
