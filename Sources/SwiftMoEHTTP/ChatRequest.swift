import Foundation
#if canImport(os)
import os
#endif

private let logger = Logger(subsystem: "com.swiftmoe", category: "server.connection")

/// The parts of a chat-completions body this server acts on.
struct ChatRequest: Equatable, Sendable {

    /// Content of the last message, or `""` when there is none.
    let prompt: String
    /// Tokens to generate, already checked against the ceiling.
    let maxTokens: Int

    /// Why a body was refused.
    struct Rejection: Error, Equatable {
        let refusal: OpenAIError
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
    static func parse(_ body: [UInt8], limits: InferenceLimits) -> Result<ChatRequest, Rejection> {
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

    private static func tokenRefusal(for error: DecodingError, limits: InferenceLimits) -> OpenAIError {
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
