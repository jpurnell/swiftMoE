import Foundation

/// The JSON of the OpenAI `chat.completion.chunk` events this server streams.
///
/// Each value is one line, with every quote, backslash and control character escaped, so it can
/// be the `data:` of an event without being split.
struct ChatChunk: Sendable {
    private let requestID: String

    /// - Parameter requestID: Identifies the stream; every chunk of one stream carries the same `req_` id.
    init(requestID: String = UUID().uuidString) {
        self.requestID = requestID
    }

    /// The chunk that carries one token of text.
    func delta(token: String) -> String {
        "{\"id\":\"req_\(requestID)\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\(Self.escape(token))\"},\"finish_reason\":null}]}"
    }

    /// The chunk that ends the completion with a finish reason (`stop`, `length`).
    func finish(reason: String = "stop") -> String {
        "{\"id\":\"req_\(requestID)\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"\(Self.escape(reason))\"}]}"
    }

    /// The last event of a completed stream.
    static let done = "[DONE]"

    /// First code point that JSON allows unescaped inside a string.
    private static let firstUnescapedScalar: UInt32 = 0x20
    private static let hexRadix = 16
    /// Hex digits in a `\uXXXX` escape.
    private static let unicodeEscapeDigits = 4

    /// Escapes text for the inside of a JSON string.
    ///
    /// Quotes, backslashes and every control character below U+0020 are escaped, as RFC 8259
    /// requires. A model emits whatever its vocabulary holds; an unescaped control character
    /// makes the event unparseable.
    static func escape(_ s: String) -> String {
        var result = ""
        result.reserveCapacity(s.utf8.count)
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": result += "\\\""
            case "\\": result += "\\\\"
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            case _ where scalar.value < firstUnescapedScalar:
                let digits = String(scalar.value, radix: hexRadix)
                result += "\\u" + String(repeating: "0", count: max(0, unicodeEscapeDigits - digits.count)) + digits
            default: result.unicodeScalars.append(scalar)
            }
        }
        return result
    }
}
