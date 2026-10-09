import Foundation

/// A response that ends a request without running inference: a status and a written reason.
///
/// The body is the OpenAI error shape, `{"error":{"message":…,"type":…}}`, so a client built
/// for that API reports the sentence rather than a bare status.
struct HTTPRefusal: Equatable, Sendable {

    /// Status code and reason phrase, e.g. `401 Unauthorized`.
    let status: String
    /// What was wrong, in a sentence the caller can act on.
    let message: String
    /// Error category in the response body.
    let type: String
    /// Header lines specific to this refusal, e.g. `WWW-Authenticate: Bearer`.
    let headers: [String]

    private init(_ status: String, _ message: String, type: String = "invalid_request_error", headers: [String] = []) {
        self.status = status
        self.message = message
        self.type = type
        self.headers = headers
    }

    static let malformed = HTTPRefusal("400 Bad Request", "Malformed HTTP request.")
    static let notJSON = HTTPRefusal("400 Bad Request", "Request body is not a JSON object.")
    static let truncatedBody = HTTPRefusal("400 Bad Request",
                                           "The request body ended before Content-Length bytes arrived.")
    static let unauthorized = HTTPRefusal("401 Unauthorized", "A valid bearer credential is required.",
                                          type: "authentication_error", headers: ["WWW-Authenticate: Bearer"])
    static let originNotAllowed = HTTPRefusal("403 Forbidden", "Origin is not allowed.")
    static let notFound = HTTPRefusal("404 Not Found", "No such route.")
    static let methodNotAllowed = HTTPRefusal("405 Method Not Allowed", "Method not allowed; use POST.",
                                              headers: ["Allow: POST"])
    static let timedOut = HTTPRefusal("408 Request Timeout",
                                      "The request was not received within the read deadline.")
    static let lengthRequired = HTTPRefusal("411 Length Required", "Content-Length is required.")
    static let wrongHost = HTTPRefusal("421 Misdirected Request", "The Host header does not name this server.")
    static let transferEncoding = HTTPRefusal("501 Not Implemented",
                                              "Transfer-Encoding is not supported; send Content-Length.")
    static let busy = HTTPRefusal("503 Service Unavailable", "The server is at its connection limit.",
                                  type: "server_error")
    static let versionNotSupported = HTTPRefusal("505 HTTP Version Not Supported",
                                                 "Only HTTP/1.0 and HTTP/1.1 are supported.")

    /// A body larger than the configured limit.
    static func bodyTooLarge(limit: Int) -> HTTPRefusal {
        HTTPRefusal("413 Content Too Large", "Request body exceeds the limit of \(limit) bytes.")
    }

    /// A request head larger than the configured limit.
    static func headersTooLarge(limit: Int) -> HTTPRefusal {
        HTTPRefusal("431 Request Header Fields Too Large", "Request headers exceed the limit of \(limit) bytes.")
    }

    /// A token count outside `1...maximum`, or not a whole number at all.
    ///
    /// - Parameters:
    ///   - field: The request field at fault: `max_tokens` or `max_completion_tokens`.
    ///   - maximum: The configured ceiling.
    static func tokensOutOfRange(field: String, maximum: Int) -> HTTPRefusal {
        HTTPRefusal("400 Bad Request", "\(field) must be a whole number from 1 to \(maximum).")
    }

    /// A prompt and completion that together need more positions than a sequence holds.
    ///
    /// - Parameters:
    ///   - promptTokens: Tokens in the prompt, as the server's tokenizer counted them.
    ///   - completionTokens: Tokens the request asked to generate, or the default.
    ///   - limit: The configured sequence limit.
    static func sequenceTooLong(promptTokens: Int, completionTokens: Int, limit: Int) -> HTTPRefusal {
        let (total, overflow) = promptTokens.addingReportingOverflow(completionTokens)
        let sum = overflow ? "more than \(Int.max)" : String(total)
        return HTTPRefusal(
            "400 Bad Request",
            "Prompt (\(promptTokens) tokens) plus completion (\(completionTokens) tokens) is \(sum) tokens; "
                + "the limit for a sequence is \(limit). Shorten the prompt or lower max_tokens.")
    }

    /// The JSON error body.
    var body: String {
        "{\"error\":{\"message\":\"\(Self.escape(message))\",\"type\":\"\(Self.escape(type))\"}}"
    }

    /// The whole response.
    ///
    /// - Parameter corsHeaders: CORS header lines for this request, appended after ``headers``.
    func response(corsHeaders: [String]) -> [UInt8] {
        let payload = body
        var lines = [
            "HTTP/1.1 \(status)",
            "Content-Type: application/json",
            "Content-Length: \(payload.utf8.count)",
            "Connection: close",
        ]
        lines.append(contentsOf: headers)
        lines.append(contentsOf: corsHeaders)
        return Array((lines.joined(separator: "\r\n") + "\r\n\r\n" + payload).utf8)
    }

    private static func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
    }
}
