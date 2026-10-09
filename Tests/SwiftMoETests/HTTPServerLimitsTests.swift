import Foundation
import Testing
@testable import SwiftMoE

/// What a request may ask for and how long a connection may take: token counts, header and
/// body sizes, the read deadline, the connection limit, and routing.
@Suite("HTTP server limits and routing")
struct HTTPServerLimitsTests {

    private static func body(_ tokens: String) -> String {
        #"{"messages":[{"role":"user","content":"hi"}],\#(tokens)}"#
    }

    private static func tokenRefusal(_ field: String) -> String {
        Expected.refusal("400 Bad Request", "\(field) must be a whole number from 1 to 8192.")
    }

    // MARK: - Defaults

    @Test("The default limits, and where the token ceiling comes from")
    func defaults() {
        let limits = HTTPServer.Limits()
        #expect(limits.maxCompletionTokens == 8192)
        #expect(limits.maxCompletionTokens == TokenGenerator.defaultMaxSequenceLength)
        #expect(limits.defaultCompletionTokens == 100)
        #expect(limits.maxHeaderBytes == 16384)
        #expect(limits.maxBodyBytes == 65536)
        #expect(limits.maxConnections == 16)
        #expect(limits.readDeadline == .seconds(10))
        #expect(limits.writeDeadline == .seconds(30))
        #expect(limits.refusalDrainDeadline == .seconds(2))
    }

    @Test("A limit of zero stops the listener from opening",
          arguments: ["maxHeaderBytes", "maxBodyBytes", "maxCompletionTokens", "defaultCompletionTokens",
                      "maxConnections", "readDeadline", "writeDeadline", "refusalDrainDeadline"])
    func zeroLimit(name: String) {
        var limits = HTTPServer.Limits()
        switch name {
        case "maxHeaderBytes": limits.maxHeaderBytes = 0
        case "maxBodyBytes": limits.maxBodyBytes = 0
        case "maxCompletionTokens": limits.maxCompletionTokens = 0
        case "defaultCompletionTokens": limits.defaultCompletionTokens = 0
        case "maxConnections": limits.maxConnections = 0
        case "readDeadline": limits.readDeadline = .zero
        case "writeDeadline": limits.writeDeadline = .zero
        default: limits.refusalDrainDeadline = .zero
        }
        let server = HTTPServer(port: 0, authentication: .unauthenticatedLoopback, limits: limits, tokenizer: RunningServer.byteTokenizer) { _, _ in }
        #expect(throws: HTTPServerError.invalidLimit(name: name)) {
            _ = try server.openListener()
        }
        #expect(server.boundAddress == nil)
    }

    @Test("A default token count above the ceiling stops the listener from opening")
    func defaultAboveCeiling() {
        var limits = HTTPServer.Limits()
        limits.maxCompletionTokens = 50
        let server = HTTPServer(port: 0, authentication: .unauthenticatedLoopback, limits: limits, tokenizer: RunningServer.byteTokenizer) { _, _ in }
        #expect(throws: HTTPServerError.limitAboveLimit(name: "defaultCompletionTokens",
                                                        ceiling: "maxCompletionTokens")) {
            _ = try server.openListener()
        }
    }

    // MARK: - Token counts

    @Test("max_tokens at the ceiling, and at one, reaches the handler unchanged")
    func tokensAtTheBounds() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        // The ceiling is the whole sequence, so it fits only beside a prompt of no tokens.
        #expect(try running.exchange(running.request(body: #"{"messages":[],"max_tokens":8192}"#))
            == Expected.stream())
        // "hi" is two tokens to the stub tokenizer: 8190 is the most that fits beside it.
        #expect(try running.exchange(running.request(body: Self.body(#""max_tokens":8190"#))) == Expected.stream())
        #expect(try running.exchange(running.request(body: Self.body(#""max_tokens":1"#))) == Expected.stream())
        #expect(try running.exchange(running.request(body: Self.body(#""max_tokens":64.0"#))) == Expected.stream())
        #expect(running.handlerCalls == [
            HandlerCall(prompt: "", maxTokens: 8192),
            HandlerCall(prompt: "hi", maxTokens: 8190),
            HandlerCall(prompt: "hi", maxTokens: 1),
            HandlerCall(prompt: "hi", maxTokens: 64),
        ])
    }

    @Test("max_tokens within the ceiling but past the sequence, prompt included, is a 400 of its own")
    func tokensWithinCeilingPastSequence() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request(body: Self.body(#""max_tokens":8191"#)))
        #expect(response == Expected.refusal(
            "400 Bad Request",
            "Prompt (2 tokens) plus completion (8191 tokens) is 8193 tokens; "
                + "the limit for a sequence is 8192. Shorten the prompt or lower max_tokens."))
        #expect(running.handlerCalls == [])
    }

    @Test("max_tokens one over the ceiling is a 400 that says so, not a quiet 8192")
    func tokensOverTheCeiling() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request(body: Self.body(#""max_tokens":8193"#)))
        #expect(response == Expected.refusal("400 Bad Request",
                                             "max_tokens must be a whole number from 1 to 8192."))
        #expect(running.handlerCalls == [])
    }

    @Test("max_tokens that is zero, negative, fractional, enormous, or not a number is a 400",
          arguments: ["0", "-1", "-8192", "1.5", "0.9", "1e30", "9223372036854775808",
                      "\"10\"", "true", "false", "[10]", "{}"])
    func tokensRefused(value: String) throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request(body: Self.body("\"max_tokens\":\(value)")))
        #expect(response == Self.tokenRefusal("max_tokens"))
        #expect(running.handlerCalls == [])
    }

    @Test("A number too large to be a number at all makes the body unparseable, which is also a 400",
          arguments: ["1e999", "-1e999"])
    func tokensUnrepresentable(value: String) throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request(body: Self.body("\"max_tokens\":\(value)")))
        #expect(response == Expected.notJSON)
        #expect(running.handlerCalls == [])
    }

    @Test("A null max_tokens is an absent one: the default applies")
    func tokensNull() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        #expect(try running.exchange(running.request(body: Self.body(#""max_tokens":null"#))) == Expected.stream())
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 100)])
    }

    @Test("max_completion_tokens is held to the same range, and named in its own refusal")
    func completionTokens() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        #expect(try running.exchange(running.request(body: Self.body(#""max_completion_tokens":8190"#)))
            == Expected.stream())
        #expect(try running.exchange(running.request(body: Self.body(#""max_completion_tokens":8193"#)))
            == Self.tokenRefusal("max_completion_tokens"))
        #expect(try running.exchange(running.request(body: Self.body(#""max_completion_tokens":0"#)))
            == Self.tokenRefusal("max_completion_tokens"))
        #expect(try running.exchange(running.request(body: Self.body(#""max_completion_tokens":-5"#)))
            == Self.tokenRefusal("max_completion_tokens"))
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 8190)])
    }

    @Test("When both are sent, max_completion_tokens wins, and an invalid loser is still refused")
    func bothTokenFields() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        #expect(try running.exchange(running.request(
            body: Self.body(#""max_tokens":7,"max_completion_tokens":9"#))) == Expected.stream())
        #expect(try running.exchange(running.request(
            body: Self.body(#""max_tokens":0,"max_completion_tokens":9"#))) == Self.tokenRefusal("max_tokens"))
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 9)])
    }

    @Test("A configured ceiling is the one enforced and the one quoted")
    func configuredCeiling() throws {
        var limits = HTTPServer.Limits()
        limits.maxCompletionTokens = 50
        limits.defaultCompletionTokens = 20
        let running = try RunningServer(limits: limits)
        defer { running.shutdown() }

        #expect(try running.exchange(running.request(body: Self.body(#""max_tokens":50"#))) == Expected.stream())
        #expect(try running.exchange(running.request(body: Self.body(#""max_tokens":51"#)))
            == Expected.refusal("400 Bad Request", "max_tokens must be a whole number from 1 to 50."))
        #expect(try running.exchange(running.request()) == Expected.stream())
        #expect(running.handlerCalls == [
            HandlerCall(prompt: "hi", maxTokens: 50),
            HandlerCall(prompt: "hi", maxTokens: 20),
        ])
    }

    @Test("A body that is not a JSON object is a 400, not a default-length completion of nothing",
          arguments: ["", "not json", "[1,2,3]", "\"text\"", "{\"messages\":"])
    func bodyIsNotAnObject(body: String) throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        #expect(try running.exchange(running.request(body: body)) == Expected.notJSON)
        #expect(running.handlerCalls == [])
    }

    // MARK: - Body size

    @Test("A body one byte over the limit is 413; the client receives it every time",
          arguments: 0..<20)
    func refusalIsDeliveredEveryTime(attempt: Int) throws {
        var limits = HTTPServer.Limits()
        limits.maxBodyBytes = 1024
        let running = try RunningServer(limits: limits)
        defer { running.shutdown() }

        let oversized = String(repeating: "a", count: 512 * 1024)
        let response = try running.exchange(running.request(body: oversized))
        #expect(response == Expected.refusal("413 Content Too Large",
                                             "Request body exceeds the limit of 1024 bytes."))
        #expect(running.handlerCalls == [])
    }

    @Test("A body exactly at the limit is read; one byte more is refused")
    func bodyAtTheLimit() throws {
        let atLimit = #"{"messages":[{"role":"user","content":"hi"}],"pad":"\#(String(repeating: "p", count: 40))"}"#
        var limits = HTTPServer.Limits()
        limits.maxBodyBytes = atLimit.utf8.count
        let running = try RunningServer(limits: limits)
        defer { running.shutdown() }

        #expect(try running.exchange(running.request(body: atLimit)) == Expected.stream())
        #expect(try running.exchange(running.request(body: atLimit + " "))
            == Expected.refusal("413 Content Too Large",
                                "Request body exceeds the limit of \(atLimit.utf8.count) bytes."))
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 100)])
    }

    @Test("A POST must declare its length: no Content-Length is 411, chunked is 501")
    func lengthRequired() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        #expect(try running.exchange(running.request(body: nil))
            == Expected.refusal("411 Length Required", "Content-Length is required."))
        #expect(try running.exchange(running.request(extraHeaders: ["Transfer-Encoding: chunked"], body: nil))
            == Expected.refusal("501 Not Implemented",
                                "Transfer-Encoding is not supported; send Content-Length."))
        #expect(try running.exchange(running.request(extraHeaders: ["Content-Length: ten"], body: nil))
            == Expected.malformed)
        #expect(try running.exchange(running.request(extraHeaders: ["Content-Length: -1"], body: nil))
            == Expected.malformed)
        #expect(running.handlerCalls == [])
    }

    @Test("A body that ends before its declared length is a 400")
    func truncatedBody() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let client = try ClientSocket(port: running.bound.port)
        defer { client.close() }
        client.send(running.request(extraHeaders: ["Content-Length: 500"], body: nil))
        client.send(Array("{\"messages\":".utf8))
        shutdown(client.descriptor, SHUT_WR)
        #expect(client.readToEnd() == Expected.refusal(
            "400 Bad Request", "The request body ended before Content-Length bytes arrived."))
        #expect(running.handlerCalls == [])
    }

    // MARK: - Header size

    @Test("Headers over the limit are 431; headers that fit are served")
    func headerLimit() throws {
        var limits = HTTPServer.Limits()
        limits.maxHeaderBytes = 512
        let running = try RunningServer(limits: limits)
        defer { running.shutdown() }

        #expect(try running.exchange(running.request()) == Expected.stream())
        let padded = try running.exchange(running.request(
            extraHeaders: ["X-Padding: \(String(repeating: "x", count: 600))"]))
        #expect(padded == Expected.refusal("431 Request Header Fields Too Large",
                                           "Request headers exceed the limit of 512 bytes."))
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 100)])
    }

    // MARK: - Read deadline

    @Test("A client that connects and sends nothing is dropped at the read deadline")
    func silentClientIsDropped() throws {
        var limits = HTTPServer.Limits()
        limits.readDeadline = .milliseconds(200)
        let running = try RunningServer(limits: limits)
        defer { running.shutdown() }

        let silent = try ClientSocket(port: running.bound.port)
        defer { silent.close() }
        // The client never writes. The read below returns only because the server gave up.
        #expect(silent.readToEnd() == Expected.timedOut)
        #expect(running.handlerCalls == [])
    }

    @Test("The deadline covers the whole request: a client that stalls part-way is dropped too")
    func stalledClientIsDropped() throws {
        var limits = HTTPServer.Limits()
        limits.readDeadline = .milliseconds(200)
        let running = try RunningServer(limits: limits)
        defer { running.shutdown() }

        let stalledInHead = try ClientSocket(port: running.bound.port)
        defer { stalledInHead.close() }
        stalledInHead.send(Array("POST /v1/chat/completions HTTP/1.1\r\nHost: ".utf8))
        #expect(stalledInHead.readToEnd() == Expected.timedOut)

        let stalledInBody = try ClientSocket(port: running.bound.port)
        defer { stalledInBody.close() }
        stalledInBody.send(running.request(extraHeaders: ["Content-Length: 500"], body: nil))
        stalledInBody.send(Array("{\"messages\":".utf8))
        #expect(stalledInBody.readToEnd() == Expected.timedOut)
        #expect(running.handlerCalls == [])
    }

    @Test("A silent client does not hold up the next one")
    func silentClientDoesNotBlockOthers() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        // Default deadline: this connection stays open, unanswered, for the whole test.
        let silent = try ClientSocket(port: running.bound.port)
        defer { silent.close() }

        #expect(try running.exchange(running.request()) == Expected.stream())
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 100)])
    }

    @Test("A connection beyond the limit is told the server is busy")
    func connectionLimit() throws {
        var limits = HTTPServer.Limits()
        limits.maxConnections = 1
        let running = try RunningServer(limits: limits)
        defer { running.shutdown() }

        let holder = try ClientSocket(port: running.bound.port)
        defer { holder.close() }

        #expect(try running.exchange(running.request()) == Expected.busy)
        #expect(running.handlerCalls == [])
    }

    // MARK: - Routing

    @Test("GET on the route is 405 with Allow: POST")
    func wrongMethod() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        #expect(try running.exchange(running.request("GET", body: nil)) == Expected.methodNotAllowed)
        #expect(try running.exchange(running.request("PUT")) == Expected.methodNotAllowed)
        #expect(try running.exchange(running.request("OPTIONS", body: nil)) == Expected.methodNotAllowed)
        #expect(running.handlerCalls == [])
    }

    @Test("Any other path is 404, however much it resembles the route",
          arguments: ["/other", "/", "/v1/chat/completions/", "/v1/chat/completions/extra",
                      "/prefix/v1/chat/completions", "/v1/chat/completionsX", "/V1/CHAT/COMPLETIONS",
                      "/other?next=/v1/chat/completions"])
    func wrongPath(target: String) throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        #expect(try running.exchange(running.request("POST", target)) == Expected.notFound)
        #expect(running.handlerCalls == [])
    }

    @Test("The route's name in a header does not make another path the route")
    func routeNameInHeader() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request(
            "POST", "/other", extraHeaders: ["Referer: http://127.0.0.1/v1/chat/completions"]))
        #expect(response == Expected.notFound)
        #expect(running.handlerCalls == [])
    }

    @Test("A query string is split off before the path is compared")
    func queryString() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        #expect(try running.exchange(running.request("POST", "/v1/chat/completions?stream=true"))
            == Expected.stream())
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 100)])
    }

    @Test("A request line that is not method, target, version is a 400",
          arguments: ["GARBAGE", "POST /v1/chat/completions", "POST  /v1/chat/completions HTTP/1.1",
                      "POST /v1/chat/completions HTTP/1.1 extra", "post /v1/chat/completions HTTP/1.1",
                      "POST v1/chat/completions HTTP/1.1", "POST http://x/v1/chat/completions HTTP/1.1",
                      " POST /v1/chat/completions HTTP/1.1", "POST /v1/chat/completions FTP/1.1"])
    func malformedRequestLine(line: String) throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let bytes = Array("\(line)\r\nHost: \(running.authority)\r\n\r\n".utf8)
        #expect(try running.exchange(bytes) == Expected.malformed)
        #expect(running.handlerCalls == [])
    }

    @Test("An HTTP version other than 1.0 or 1.1 is 505")
    func unsupportedVersion() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let bytes = Array("POST /v1/chat/completions HTTP/2.0\r\nHost: \(running.authority)\r\n\r\n".utf8)
        #expect(try running.exchange(bytes)
            == Expected.refusal("505 HTTP Version Not Supported", "Only HTTP/1.0 and HTTP/1.1 are supported."))
    }

    // MARK: - Request head parsing

    @Test("The request head is parsed into method, path, query, version and headers")
    func headParsing() throws {
        let text = "POST /v1/chat/completions?a=1&b=2 HTTP/1.1\r\nHost: localhost:8080\r\ncontent-LENGTH:  12 \r\nAccept: a\r\nAccept: b"
        let head = try #require(HTTPRequestHead.parse(Array(text.utf8)))
        #expect(head.method == "POST")
        #expect(head.path == "/v1/chat/completions")
        #expect(head.query == "a=1&b=2")
        #expect(head.version == "HTTP/1.1")
        #expect(head.headers == ["host": "localhost:8080", "content-length": "12", "accept": "a, b"])
    }

    @Test("Header lines that are folded, nameless, or not UTF-8 make the head unparseable")
    func headRejections() {
        let requestLine = "POST / HTTP/1.1\r\n"
        #expect(HTTPRequestHead.parse(Array((requestLine + "Host: a\r\n continued").utf8)) == nil)
        #expect(HTTPRequestHead.parse(Array((requestLine + ": value").utf8)) == nil)
        #expect(HTTPRequestHead.parse(Array((requestLine + "No colon here").utf8)) == nil)
        #expect(HTTPRequestHead.parse(Array((requestLine + "Bad Name: value").utf8)) == nil)
        #expect(HTTPRequestHead.parse(Array(requestLine.utf8) + [0x48, 0x3A, 0xFF, 0xFE]) == nil)
        #expect(HTTPRequestHead.parse(Array((requestLine + "Authorization: a\r\nAuthorization: b").utf8)) == nil)
        #expect(HTTPRequestHead.parse([]) == nil)
    }
}
