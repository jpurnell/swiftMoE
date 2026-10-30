import Foundation
import os
import SwiftHTTPService
import Testing
@testable import SwiftMoE
@testable import SwiftMoEHTTP

/// What only a whole running service shows: TLS, a client that drops mid-stream, a shutdown with a
/// stream in flight, and the order of the layers around the route.
extension HTTPServiceTests {
@Suite("Chat service behaviour")
struct ServiceBehaviourTests {

    private static let webOrigin = "https://app.example"

    private static func body(_ prompt: String) -> String {
        #"{"messages":[{"role":"user","content":"\#(prompt)"}]}"#
    }

    // MARK: - TLS

    @Test("Over TLS, with a certificate the client trusts, the same stream arrives")
    func tlsStream() throws {
        let tls = try TestTLS()
        defer { tls.cleanUp() }
        let running = try RunningServer(tls: tls.files, respond: { _, _, emit in _ = emit("secure") })
        defer { running.shutdown() }

        let port = Int(running.port)
        let request = String(decoding: running.request(), as: UTF8.self)
        let raw = try blocking { try await tlsExchange(port: port, tls: tls, request: request) }
        let answer = Canonical.normalise(raw)
        #expect(answer.hasPrefix("HTTP/1.1 200 OK\r\n"))
        #expect(answer.contains(#""delta":{"content":"secure"}"#))
        #expect(answer.hasSuffix("data: [DONE]\n\n"))
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 100)])
    }

    @Test("A TLS port does not answer a plain-text request")
    func tlsPortRefusesPlaintext() throws {
        let tls = try TestTLS()
        defer { tls.cleanUp() }
        let running = try RunningServer(tls: tls.files)
        defer { running.shutdown() }

        let answer = try running.rawExchange(running.request())
        #expect(!answer.contains("HTTP/1.1 200"))
        #expect(running.handlerCalls == [])
    }

    // MARK: - A client that drops mid-stream

    @Test("After a client drops mid-stream the generation stops and the next request is served")
    func droppedClientFreesTheModel() throws {
        let probe = LeaveProbe()
        let first = OSAllocatedUnfairLock(initialState: true)
        let running = try RunningServer(respond: { request, shouldContinue, emit in
            if request.prompt == "stream", first.withLock({ let value = $0; $0 = false; return value }) {
                try probe.respond(request, shouldContinue, emit)
            }
        })
        defer { running.shutdown() }

        let client = try ClientSocket(port: running.port)
        client.send(running.request(body: Self.body("stream")))
        #expect(probe.awaitStreaming())
        client.close()
        let seen = try #require(probe.awaitFinished())
        #expect(seen.sawDisconnect)
        #expect(seen.sent < LeaveProbe.cap)
        #expect(running.events.awaitNext(.slotReleased))

        #expect(try running.exchange(running.request(body: Self.body("next"))) == Expected.stream())
    }

    // MARK: - Shutdown

    @Test("A graceful shutdown lets a stream in flight finish, then stops accepting")
    func gracefulShutdown() throws {
        let started = DispatchSemaphore(value: 0)
        let running = try RunningServer(respond: { _, shouldContinue, emit in
            for index in 0..<5 where shouldContinue() {
                guard emit("t\(index)") else { return }
                if index == 0 { started.signal() }
                Thread.sleep(forTimeInterval: 0.05)
            }
        })
        let client = try ClientSocket(port: running.port)
        defer { client.close() }
        client.send(running.request())
        #expect(started.wait(timeout: .now() + .seconds(10)) == .success)

        running.shutdown()

        let answer = Canonical.normalise(client.readToEnd())
        for index in 0..<5 { #expect(answer.contains("\"content\":\"t\(index)\"")) }
        #expect(answer.hasSuffix("data: [DONE]\n\n"))
        #expect(!ClientSocket.canConnect(port: running.port))
    }

    // MARK: - Order

    @Test("A preflight from a hostile origin gets no inference and no CORS header")
    func hostilePreflight() throws {
        let running = try RunningServer(allowedOrigins: [Self.webOrigin])
        defer { running.shutdown() }

        let answer = try running.exchange(running.request(
            "OPTIONS", key: nil, origin: "https://evil.example",
            extraHeaders: ["Access-Control-Request-Method: POST"], body: nil))
        #expect(answer == Expected.refusal("403 Forbidden", "Origin is not allowed.", headers: ["Vary: Origin"]))
        #expect(!answer.contains("Access-Control-Allow-Origin"))
        #expect(running.handlerCalls == [])
    }

    @Test("A POST from a listed origin with no key is 401, readable by that origin, and runs nothing")
    func listedOriginWithoutKey() throws {
        let running = try RunningServer(allowedOrigins: [Self.webOrigin])
        defer { running.shutdown() }

        let answer = try running.exchange(running.request(key: nil, origin: Self.webOrigin))
        #expect(answer == Expected.refusal(
            "401 Unauthorized", "A valid bearer credential is required.", type: "authentication_error",
            headers: ["WWW-Authenticate: Bearer", "Access-Control-Allow-Origin: \(Self.webOrigin)", "Vary: Origin"]))
        #expect(running.handlerCalls == [])
    }

    @Test("A preflight from a listed origin needs no key, and still runs nothing")
    func preflightBeforeAuthentication() throws {
        let running = try RunningServer(allowedOrigins: [Self.webOrigin])
        defer { running.shutdown() }

        let answer = try running.exchange(running.request("OPTIONS", key: nil, origin: Self.webOrigin, body: nil))
        #expect(answer.hasPrefix("HTTP/1.1 204 No Content\r\n"))
        #expect(answer.contains("Access-Control-Allow-Origin: \(Self.webOrigin)"))
        #expect(running.handlerCalls == [])
    }

    @Test("The Host check comes before the preflight: a rebinding page reaches neither")
    func hostBeforePreflight() throws {
        let running = try RunningServer(allowedOrigins: [Self.webOrigin])
        defer { running.shutdown() }

        let answer = try running.exchange(running.request(
            "OPTIONS", host: "evil.example", key: nil, origin: Self.webOrigin, body: nil))
        #expect(answer == Expected.wrongHost)
        #expect(!answer.contains("Access-Control-Allow-Origin"))
        #expect(running.handlerCalls == [])
    }

    @Test("Authentication comes before the route: an unknown path with no key is 401, not 404")
    func authenticationBeforeRouting() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        #expect(try running.exchange(running.request("POST", "/other", key: nil)) == Expected.unauthorized)
        #expect(try running.exchange(running.request("POST", "/other")) == Expected.notFound)
    }
}
}
