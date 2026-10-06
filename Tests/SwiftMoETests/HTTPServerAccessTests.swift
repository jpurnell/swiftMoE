import Foundation
import Testing
@testable import SwiftMoE

/// Who may reach the handler: the bearer credential, the origin allowlist, and the `Host` check.
///
/// Every test drives a real server over loopback and compares the whole response, byte for
/// byte, so a header that should not be there fails the test as surely as one that is missing.
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

        let body = String(repeating: "a", count: 512 * 1024)
        let response = try running.exchange(running.request(key: RunningServer.wrongKey, body: body))
        #expect(response == Expected.unauthorized)
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
            let server = HTTPServer(host: host, port: 0, authentication: .unauthenticatedLoopback) { _, _, _ in }
            #expect(throws: HTTPServerError.credentialRequired(host: host)) {
                _ = try server.openListener()
            }
            #expect(server.boundAddress == nil)
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
        #expect(allowed == [
            "HTTP/1.1 204 No Content",
            "Content-Length: 0",
            "Connection: close",
            "Access-Control-Allow-Methods: POST",
            "Access-Control-Allow-Headers: Authorization, Content-Type",
            "Access-Control-Allow-Origin: http://localhost:3000",
            "Vary: Origin",
        ].joined(separator: "\r\n") + "\r\n\r\n")

        let elsewhere = try running.exchange(running.request(
            "OPTIONS", "/other", key: nil, origin: Self.webOrigin, body: nil))
        #expect(elsewhere == Expected.refusal("404 Not Found", "No such route.", headers: Self.corsHeaders))

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
        let server = HTTPServer(port: 0, authentication: .unauthenticatedLoopback,
                                allowedOrigins: [entry]) { _, _, _ in }
        #expect(throws: HTTPServerError.invalidOrigin(entry)) {
            _ = try server.openListener()
        }
        #expect(server.boundAddress == nil)
    }

    // MARK: - Host

    @Test("On loopback, a Host that is not this server is refused before the key is looked at",
          arguments: ["evil.example", "evil.example:8080", "127.0.0.1", "127.0.0.1:1", "localhost",
                      "localhost.", "10.0.1.114", ""])
    func badHost(host: String) throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request(host: host))
        #expect(response == Expected.wrongHost)
        #expect(running.handlerCalls == [])
    }

    @Test("On loopback, the bound address and localhost are accepted as Host, in any letter case")
    func goodHost() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let port = running.bound.port
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
