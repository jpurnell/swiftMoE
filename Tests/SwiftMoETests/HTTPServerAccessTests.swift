import Foundation
import Testing
@testable import SwiftMoE
@testable import SwiftMoEHTTP

/// Who may reach the handler: the bearer credential, the origin allowlist, and the `Host` check.
///
/// Every test drives a real server over loopback and compares the whole response, byte for
/// byte, so a header that should not be there fails the test as surely as one that is missing.
extension HTTPServiceTests {
@Suite("HTTP server access control")
struct HTTPServerAccessTests {

    private static let webOrigin = "http://localhost:3000"
    private static let corsHeaders = ["Access-Control-Allow-Origin: http://localhost:3000", "Vary: Origin"]

    // MARK: - Credential

    @Test("No key is 401 with WWW-Authenticate: Bearer, and the handler is not called")
    func noKey() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request(key: nil))
        #expect(response == Expected.unauthorized)
        #expect(running.handlerCalls == [])
    }

    @Test("A wrong key is the same 401 as no key")
    func wrongKey() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request(key: RunningServer.wrongKey))
        #expect(response == Expected.unauthorized)

        let basic = try running.exchange(running.request(
            key: nil, extraHeaders: ["Authorization: Basic \(RunningServer.key)"]))
        #expect(basic == Expected.unauthorized)
        #expect(running.handlerCalls == [])
    }

    @Test("The right key reaches the handler, with the prompt and the default token count")
    func rightKey() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request())
        #expect(response == Expected.stream())
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 100)])
    }

    @Test("Without a key, every method and path answers 401: nothing is learned about routes")
    func unauthenticatedProbes() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        #expect(try running.exchange(running.request("GET", key: nil, body: nil)) == Expected.unauthorized)
        #expect(try running.exchange(running.request("POST", "/other", key: nil)) == Expected.unauthorized)
        #expect(try running.exchange(running.request("OPTIONS", key: nil, body: nil)) == Expected.unauthorized)
        #expect(running.handlerCalls == [])
    }

    @Test("A request refused for its key is refused before its body is read, and still hears why",
          arguments: 0..<20)
    func refusalBeforeBodyIsDelivered(attempt: Int) throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        // Within the body bound, a wrong key is 401 and the handler is not called.
        let body = String(repeating: "a", count: 32 * 1024)
        let response = try running.exchange(running.request(key: RunningServer.wrongKey, body: body))
        #expect(response == Expected.unauthorized)
        #expect(running.handlerCalls == [])
    }

    // DIFFERENCE (SwiftHTTPService 0.5.0): the transport checks the declared length against the
    // route's bound before any middleware runs, so a body over the bound is 413 whatever the key.
    // Either way the body is refused unread, and the client hears why.
    @Test("A body declared over the bound is refused 413 before it is read, with or without a key",
          arguments: 0..<5)
    func oversizedBodyIsRefusedUnread(attempt: Int) throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let declared = ["Content-Length: \(512 * 1024)"]
        let refusal = Expected.refusal("413 Payload Too Large", "Request body exceeds the limit of 65536 bytes.")
        #expect(try running.exchange(running.request(key: RunningServer.wrongKey, extraHeaders: declared, body: nil)) == refusal)
        #expect(try running.exchange(running.request(extraHeaders: declared, body: nil)) == refusal)
        #expect(running.handlerCalls == [])
    }

    @Test("A loopback server started without authentication serves a request that has no key")
    func explicitNoAuth() throws {
        let running = try RunningServer(authenticated: false)
        defer { running.shutdown() }

        let response = try running.exchange(running.request(key: nil))
        #expect(response == Expected.stream())
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 100)])
    }

    @Test("A listener off loopback refuses to open without a credential, and binds nothing")
    func nonLoopbackNeedsCredential() {
        for host in ["0.0.0.0", "10.0.1.114"] {
            let server = ChatService(host: host, port: 0, authentication: .unauthenticatedLoopback, tokenizer: RunningServer.byteTokenizer) { _, _, _ in }
            #expect(throws: ServiceError.credentialRequired(host: host)) {
                _ = try blocking { try await server.start() }
            }
        }
    }

    // MARK: - CORS

    @Test("By default no response carries a CORS header, with or without an Origin")
    func noCORSByDefault() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let plain = try running.exchange(running.request())
        #expect(plain == Expected.stream())
        #expect(!plain.contains("Access-Control-Allow-Origin"))

        let fromPage = try running.exchange(running.request(origin: Self.webOrigin))
        #expect(fromPage == Expected.refusal("403 Forbidden", "Origin is not allowed."))

        let preflight = try running.exchange(running.request(
            "OPTIONS", key: nil, origin: Self.webOrigin,
            extraHeaders: ["Access-Control-Request-Method: POST"], body: nil))
        #expect(preflight == Expected.refusal("403 Forbidden", "Origin is not allowed."))
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 100)])
    }

    @Test("An allowed origin is echoed exactly, with Vary: Origin")
    func allowedOriginEchoed() throws {
        let running = try RunningServer(allowedOrigins: [Self.webOrigin, "https://app.example"])
        defer { running.shutdown() }

        let response = try running.exchange(running.request(origin: Self.webOrigin))
        #expect(response == Expected.stream(headers: Self.corsHeaders))

        let second = try running.exchange(running.request(origin: "https://app.example"))
        #expect(second == Expected.stream(
            headers: ["Access-Control-Allow-Origin: https://app.example", "Vary: Origin"]))

        let noOrigin = try running.exchange(running.request())
        #expect(noOrigin == Expected.stream(headers: ["Vary: Origin"]))
    }

    @Test("A disallowed origin gets no CORS header and does not reach the handler",
          arguments: ["https://evil.example", "http://localhost:3001", "https://localhost:3000",
                      "http://localhost:3000/", "HTTP://LOCALHOST:3000", "null", "*"])
    func disallowedOrigin(origin: String) throws {
        let running = try RunningServer(allowedOrigins: [Self.webOrigin])
        defer { running.shutdown() }

        let response = try running.exchange(running.request(origin: origin))
        #expect(response == Expected.refusal("403 Forbidden", "Origin is not allowed.",
                                             headers: ["Vary: Origin"]))
        #expect(running.handlerCalls == [])
    }

    @Test("Preflight is answered for an allowed origin, without a key, and only for the route")
    func preflight() throws {
        let running = try RunningServer(allowedOrigins: [Self.webOrigin])
        defer { running.shutdown() }

        let allowed = try running.exchange(running.request(
            "OPTIONS", key: nil, origin: Self.webOrigin,
            extraHeaders: ["Access-Control-Request-Method: POST"], body: nil))
        // INTENDED (SwiftHTTPService 0.5.0): the kit's CORS layer writes no `Content-Length: 0` on a 204.
        let preflightAnswer = Expected.response("204 No Content", headers: [
            "Connection: close",
            "Access-Control-Allow-Methods: POST",
            "Access-Control-Allow-Headers: Authorization, Content-Type",
            "Access-Control-Allow-Origin: http://localhost:3000",
            "Vary: Origin",
        ], body: "")
        #expect(allowed == preflightAnswer)

        // DIFFERENCE (SwiftHTTPService 0.5.0): the kit answers a preflight from an allowed origin
        // for any path, where the old server answered 404 for a path that is not the route.
        let elsewhere = try running.exchange(running.request(
            "OPTIONS", "/other", key: nil, origin: Self.webOrigin, body: nil))
        #expect(elsewhere == preflightAnswer)

        let disallowed = try running.exchange(running.request(
            "OPTIONS", key: nil, origin: "https://evil.example", body: nil))
        #expect(disallowed == Expected.refusal("403 Forbidden", "Origin is not allowed.",
                                               headers: ["Vary: Origin"]))
        #expect(running.handlerCalls == [])
    }

    @Test("A refusal sent to an allowed origin is readable by that origin")
    func refusalToAllowedOrigin() throws {
        let running = try RunningServer(allowedOrigins: [Self.webOrigin])
        defer { running.shutdown() }

        let response = try running.exchange(running.request(key: nil, origin: Self.webOrigin))
        #expect(response == Expected.refusal(
            "401 Unauthorized", "A valid bearer credential is required.",
            type: "authentication_error",
            headers: ["WWW-Authenticate: Bearer"] + Self.corsHeaders))
    }

    @Test("An allowlist entry that is not an origin stops the listener from opening",
          arguments: ["*", "null", "localhost:3000", "http://localhost:3000/", "http://localhost:3000/app",
                      "https://", "https://a b", ""])
    func invalidAllowlistEntry(entry: String) {
        let server = ChatService(port: 0, authentication: .unauthenticatedLoopback,
                                 allowedOrigins: [entry], tokenizer: RunningServer.byteTokenizer) { _, _, _ in }
        #expect(throws: ServiceError.invalidOrigin(entry)) {
            _ = try blocking { try await server.start() }
        }
    }

    // MARK: - Host

    @Test("On loopback, a Host that is not this server is refused before the key is looked at",
          arguments: ["evil.example", "evil.example:8080", "127.0.0.1", "127.0.0.1:1", "localhost",
                      "localhost.", "10.0.1.114"])
    func badHost(host: String) throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request(host: host))
        #expect(response == Expected.wrongHost)
        #expect(running.handlerCalls == [])
    }

    // DIFFERENCE (SwiftHTTPService 0.5.0): a request with no Host header is refused by the transport
    // as malformed (400) before the Host check can say 421.
    @Test("A request with no Host header is refused, and the handler is not called")
    func missingHost() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        #expect(try running.exchange(running.request(host: "")) == Expected.malformed)
        #expect(running.handlerCalls == [])
    }

    @Test("On loopback, the bound address and localhost are accepted as Host, in any letter case")
    func goodHost() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let port = running.port
        for host in ["127.0.0.1:\(port)", "localhost:\(port)", "LOCALHOST:\(port)"] {
            #expect(try running.exchange(running.request(host: host)) == Expected.stream())
        }
        #expect(running.handlerCalls.count == 3)
    }

    @Test("Two Host headers are a malformed request")
    func duplicateHost() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request(extraHeaders: ["Host: evil.example"]))
        #expect(response == Expected.malformed)
        #expect(running.handlerCalls == [])
    }
}
}
