import Foundation
import HTTPTypes
import SwiftHTTPService

/// A response that ends a request without running inference: a status and a written reason.
///
/// The body is the OpenAI error shape, `{"error":{"message":…,"type":…}}`, so a client built
/// for that API reports the sentence rather than a bare status.
struct OpenAIError: Equatable, Sendable {

    /// Status code and reason phrase, e.g. `401 Unauthorized`.
    let status: String
    /// What was wrong, in a sentence the caller can act on.
    let message: String
    /// Error category in the response body.
    let type: String
    /// Header fields specific to this refusal, as `Name: value`.
    let headers: [String]

    private init(_ status: String, _ message: String, type: String = "invalid_request_error", headers: [String] = []) {
        self.status = status
        self.message = message
        self.type = type
        self.headers = headers
    }

    static let malformed = OpenAIError("400 Bad Request", "Malformed HTTP request.")
    static let malformedTarget = OpenAIError("400 Bad Request", "Malformed request target.")
    static let notJSON = OpenAIError("400 Bad Request", "Request body is not a JSON object.")
    static let unauthorized = OpenAIError("401 Unauthorized", "A valid bearer credential is required.",
                                          type: "authentication_error", headers: ["WWW-Authenticate: Bearer"])
    static let originNotAllowed = OpenAIError("403 Forbidden", "Origin is not allowed.")
    static let notFound = OpenAIError("404 Not Found", "No such route.")
    static let methodNotAllowed = OpenAIError("405 Method Not Allowed", "Method not allowed; use POST.",
                                              headers: ["Allow: POST"])
    static let timedOut = OpenAIError("408 Request Timeout",
                                      "The request was not received within the read deadline.")
    static let lengthRequired = OpenAIError("411 Length Required", "Content-Length is required.")
    static let wrongHost = OpenAIError("421 Misdirected Request", "The Host header does not name this server.")
    static let expectationFailed = OpenAIError("417 Expectation Failed", "The expectation cannot be met.")
    static let versionNotSupported = OpenAIError("505 HTTP Version Not Supported",
                                                 "Only HTTP/1.0 and HTTP/1.1 are supported.")
    /// One connection more than the service serves at once. The transport adds `Retry-After`.
    static let busy = OpenAIError("503 Service Unavailable", "The server is at its connection limit.",
                                  type: "server_error")
    static let internalError = OpenAIError("500 Internal Server Error", "The server could not complete the request.",
                                           type: "server_error")

    /// A request whose turn at the model did not come within the queue deadline.
    ///
    /// - Parameter seconds: What to send as `Retry-After`.
    static func queueTimedOut(retryAfter seconds: Int) -> OpenAIError {
        OpenAIError(
            "503 Service Unavailable",
            "The server is busy with another request and could not start this one in time. Try again later.",
            type: "server_error", headers: ["Retry-After: \(seconds)"])
    }

    /// A body larger than the configured limit.
    static func bodyTooLarge(limit: Int) -> OpenAIError {
        OpenAIError("413 Content Too Large", "Request body exceeds the limit of \(limit) bytes.")
    }

    /// A request head larger than the configured limit.
    static func headersTooLarge(limit: Int) -> OpenAIError {
        OpenAIError("431 Request Header Fields Too Large", "Request headers exceed the limit of \(limit) bytes.")
    }

    /// A token count outside `1...maximum`, or not a whole number at all.
    static func tokensOutOfRange(field: String, maximum: Int) -> OpenAIError {
        OpenAIError("400 Bad Request", "\(field) must be a whole number from 1 to \(maximum).")
    }

    /// A prompt and completion that together need more positions than a sequence holds.
    static func sequenceTooLong(promptTokens: Int, completionTokens: Int, limit: Int) -> OpenAIError {
        let (total, overflow) = promptTokens.addingReportingOverflow(completionTokens)
        let sum = overflow ? "more than \(Int.max)" : String(total)
        return OpenAIError(
            "400 Bad Request",
            "Prompt (\(promptTokens) tokens) plus completion (\(completionTokens) tokens) is \(sum) tokens; "
                + "the limit for a sequence is \(limit). Shorten the prompt or lower max_tokens.")
    }

    /// The JSON error body.
    var body: String {
        "{\"error\":{\"message\":\"\(Self.escape(message))\",\"type\":\"\(Self.escape(type))\"}}"
    }

    /// The numeric status.
    var code: Int { Int(status.prefix(3)) ?? 500 }

    /// The response, for a layer to return.
    var response: Response {
        var fields = HTTPFields()
        fields[.contentType] = "application/json"
        for header in headers {
            guard let colon = header.firstIndex(of: ":"),
                  let name = HTTPField.Name(String(header[..<colon])) else { continue }
            fields[name] = String(header[header.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
        }
        return Response(status: HTTPResponse.Status(code: code), headers: fields, body: Data(body.utf8))
    }

    private static func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
    }
}
