import Foundation
import Testing
@testable import SwiftMoEHTTP

/// The JSON of the events ``ChatChunk`` puts in a stream.
///
/// Every event's payload has to be JSON whatever the model emitted. The demo server maps
/// token ids onto the first 128 code points, so control characters are ordinary output there.
@Suite("SSE chunks")
struct ChatChunkTests {

    /// What the stream writes for one chunk: a `data:` line and a blank line.
    private static func event(_ payload: String) -> String { "data: \(payload)\n\n" }

    /// The `content` of the delta in one `data:` line.
    private static func content(ofEvent event: String) throws -> String {
        let prefix = "data: "
        try #require(event.hasPrefix(prefix))
        let payload = Data(event.dropFirst(prefix.count).utf8)
        let object = try #require(try JSONSerialization.jsonObject(with: payload) as? [String: Any])
        let choices = try #require(object["choices"] as? [[String: Any]])
        let delta = try #require(choices.first?["delta"] as? [String: Any])
        return try #require(delta["content"] as? String)
    }

    @Test("A delta is one data line of JSON followed by a blank line")
    func deltaShape() {
        let chunk = ChatChunk(requestID: "t")
        #expect(Self.event(chunk.delta(token: "hi")) == #"data: {"id":"req_t","object":"chat.completion.chunk","choices":"#
            + #"[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"# + "\n\n")
    }

    @Test("Quotes, backslashes and every control character are escaped, by name or by number")
    func escaping() {
        let chunk = ChatChunk(requestID: "t")
        #expect(Self.event(chunk.delta(token: "\u{01}\u{08}\u{0C}\u{1F}\"\\\n\r\t")) == #"data: {"id":"req_t","object":"chat.completion.chunk","choices":"#
            + #"[{"index":0,"delta":{"content":"\u0001\u0008\u000c\u001f\"\\\n\r\t"},"finish_reason":null}]}"#
            + "\n\n")
    }

    @Test("Each of the first 128 code points arrives as the token that was sent", arguments: 0..<128)
    func everyDemoTokenIsJSON(codePoint: Int) throws {
        let token = String(UnicodeScalar(UInt8(codePoint)))
        let event = Self.event(ChatChunk(requestID: "t").delta(token: token))
        #expect(event.hasSuffix("\n\n"))
        // One event, on one line: nothing in the payload may end the line early.
        let line = String(event.dropLast(2))
        #expect(!line.contains("\n"))
        #expect(try Self.content(ofEvent: line) == token)
    }

    @Test("A finish reason is escaped like any other string")
    func finishReason() {
        let chunk = ChatChunk(requestID: "t")
        #expect(Self.event(chunk.finish(reason: "sto\"p")) + Self.event(ChatChunk.done)
            == #"data: {"id":"req_t","object":"chat.completion.chunk","choices":"#
            + #"[{"index":0,"delta":{},"finish_reason":"sto\"p"}]}"# + "\n\ndata: [DONE]\n\n")
    }
}
