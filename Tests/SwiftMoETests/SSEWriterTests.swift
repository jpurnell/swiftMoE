import Foundation
import Testing
@testable import SwiftMoE

/// The bytes ``SSEWriter`` puts on the wire.
///
/// Every event's payload has to be JSON whatever the model emitted. The demo server maps
/// token ids onto the first 128 code points, so control characters are ordinary output there.
@Suite("SSE writer")
struct SSEWriterTests {

    /// A connected pair of stream sockets: the writer's end and the end a test reads.
    private struct Pair {
        let writerEnd: Int32
        let readerEnd: Int32

        init() throws {
            var descriptors: [Int32] = [-1, -1]
            try #require(socketpair(AF_UNIX, SOCK_STREAM, 0, &descriptors) == 0)
            writerEnd = descriptors[0]
            readerEnd = descriptors[1]
        }

        /// Closes the writer's end and returns everything written to it.
        func written() -> String {
            close(writerEnd)
            defer { close(readerEnd) }
            return ClientSocket(descriptor: readerEnd).readToEnd()
        }
    }

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
    func deltaShape() throws {
        let pair = try Pair()
        let writer = SSEWriter(fileDescriptor: pair.writerEnd, requestID: "t")
        #expect(writer.sendDelta(token: "hi"))
        #expect(pair.written() == #"data: {"id":"req_t","object":"chat.completion.chunk","choices":"#
            + #"[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"# + "\n\n")
    }

    @Test("Quotes, backslashes and every control character are escaped, by name or by number")
    func escaping() throws {
        let pair = try Pair()
        let writer = SSEWriter(fileDescriptor: pair.writerEnd, requestID: "t")
        #expect(writer.sendDelta(token: "\u{01}\u{08}\u{0C}\u{1F}\"\\\n\r\t"))
        #expect(pair.written() == #"data: {"id":"req_t","object":"chat.completion.chunk","choices":"#
            + #"[{"index":0,"delta":{"content":"\u0001\u0008\u000c\u001f\"\\\n\r\t"},"finish_reason":null}]}"#
            + "\n\n")
    }

    @Test("Each of the first 128 code points arrives as the token that was sent", arguments: 0..<128)
    func everyDemoTokenIsJSON(codePoint: Int) throws {
        let pair = try Pair()
        let writer = SSEWriter(fileDescriptor: pair.writerEnd, requestID: "t")
        let token = String(UnicodeScalar(UInt8(codePoint)))
        #expect(writer.sendDelta(token: token))

        let written = pair.written()
        #expect(written.hasSuffix("\n\n"))
        // One event, on one line: nothing in the payload may end the line early.
        let event = String(written.dropLast(2))
        #expect(!event.contains("\n"))
        #expect(try Self.content(ofEvent: event) == token)
    }

    @Test("A finish reason is escaped like any other string")
    func finishReason() throws {
        let pair = try Pair()
        let writer = SSEWriter(fileDescriptor: pair.writerEnd, requestID: "t")
        writer.sendFinish(reason: "sto\"p")
        writer.sendDone()
        #expect(pair.written() == #"data: {"id":"req_t","object":"chat.completion.chunk","choices":"#
            + #"[{"index":0,"delta":{},"finish_reason":"sto\"p"}]}"# + "\n\ndata: [DONE]\n\n")
    }

    @Test("A write to a client that has gone reports failure instead of raising SIGPIPE")
    func writeAfterClose() throws {
        let pair = try Pair()
        var enabled: Int32 = 1
        setsockopt(pair.writerEnd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, socklen_t(MemoryLayout<Int32>.size))
        close(pair.readerEnd)

        let writer = SSEWriter(fileDescriptor: pair.writerEnd, requestID: "t")
        #expect(!writer.sendDelta(token: "hi"))
        #expect(writer.clientHasDisconnected)
        close(pair.writerEnd)
    }

    @Test("A writer on a connected socket reports its client as still there")
    func connectedClient() throws {
        let pair = try Pair()
        let writer = SSEWriter(fileDescriptor: pair.writerEnd, requestID: "t")
        #expect(!writer.clientHasDisconnected)
        _ = pair.written()
    }
}
