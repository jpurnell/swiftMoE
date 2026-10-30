import Foundation
import HTTPTypes
import os
import SwiftHTTPService
import Testing
@testable import SwiftMoE
@testable import SwiftMoEHTTP

/// Questions the design proposal (section 10) asked before the service was written, kept as tests
/// so the answers stay true. S1 (does the kit cancel an idle handler?) is answered by the kit's own
/// tests, which this package relies on and re-checks through its queue and mid-stream tests.
extension HTTPServiceTests {
@Suite("Design spikes S2 to S5")
struct SpikeTests {

    // MARK: - S2: the bytes of an event stream

    @Test("S2: Response.eventStream writes the bytes the old SSEWriter wrote, and a client can parse them")
    func eventStreamBytes() async throws {
        let recorded = try #require(try WireFixture.load()["sse_tokens_complete_stream"]).recorded
        let chunk = ChatChunk(requestID: "<ID>")
        let response = Response.eventStream { events in
            for token in ["Hel", "lo ", "\"w\"\n\u{1}"] { try await events.send(data: chunk.delta(token: token)) }
            try await events.send(data: chunk.finish())
            try await events.send(data: ChatChunk.done)
        }
        let written = OSAllocatedUnfairLock(initialState: Data())
        let stream = try #require(response.stream)
        try await stream.produce(ResponseWriter { data in written.withLock { $0.append(data) } })

        // The body is byte for byte what the old server recorded.
        #expect(String(decoding: written.withLock { $0 }, as: UTF8.self) == recorded.body)
        // The head differs only where proposal section 7 says: `no-store`, not `no-cache`.
        #expect(response.headers[.contentType] == "text/event-stream")
        #expect(response.headers[.cacheControl] == "no-store")

        // A client reads `data:` lines, stops at [DONE], and decodes the rest as chunk JSON.
        let body = String(decoding: written.withLock { $0 }, as: UTF8.self)
        var contents: [String] = []
        var finish: String?
        for event in body.components(separatedBy: "\n\n") where event.hasPrefix("data: ") {
            let payload = String(event.dropFirst(6))
            if payload == "[DONE]" { break }
            let json = try #require(try JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any])
            let choice = try #require((json["choices"] as? [[String: Any]])?.first)
            if let delta = choice["delta"] as? [String: Any], let content = delta["content"] as? String {
                contents.append(content)
            }
            if let reason = choice["finish_reason"] as? String { finish = reason }
        }
        #expect(contents == ["Hel", "lo ", "\"w\"\n\u{1}"])
        #expect(finish == "stop")
    }

    // MARK: - S3: a long generation on its own thread

    @Test("S3: other routes stay responsive during a long generation, with eight requests waiting")
    func longGenerationLeavesTheServerResponsive() throws {
        let started = DispatchSemaphore(value: 0)
        let running = try RunningServer(respond: { request, shouldContinue, emit in
            guard request.prompt == "long" else { return }
            started.signal()
            for _ in 0..<60 where shouldContinue() {
                guard emit("x") else { return }
                Thread.sleep(forTimeInterval: 0.05)
            }
        })
        defer { running.shutdown() }

        let long = try ClientSocket(port: running.port)
        defer { long.close() }
        long.send(running.request(body: #"{"messages":[{"role":"user","content":"long"}]}"#))
        #expect(started.wait(timeout: .now() + .seconds(10)) == .success)

        var waiting: [ClientSocket] = []
        defer { waiting.forEach { $0.close() } }
        for _ in 0..<8 {
            let client = try ClientSocket(port: running.port)
            client.send(running.request(body: #"{"messages":[{"role":"user","content":"wait"}]}"#))
            #expect(running.events.awaitNext(.requestQueued))
            waiting.append(client)
        }

        let clock = ContinuousClock()
        var slowest = Duration.zero
        for _ in 0..<5 {
            var answer = ""
            let elapsed = try clock.measure { answer = try running.exchange(running.request("POST", "/other")) }
            #expect(answer == Expected.notFound)
            slowest = max(slowest, elapsed)
        }
        #expect(slowest < .milliseconds(100))
    }

    @Test("S3: a reader that stops reading stops the generation, and the model is freed")
    func pausedReaderStopsGeneration() throws {
        let produced = OSAllocatedUnfairLock(initialState: 0)
        let stoppedBecauseCancelled = OSAllocatedUnfairLock(initialState: false)
        let big = String(repeating: "x", count: 1 << 20)
        let running = try RunningServer(
            limits: { var limits = InferenceLimits(); limits.writeDeadline = .seconds(1); return limits }(),
            respond: { request, shouldContinue, emit in
                guard request.prompt == "flood" else { return }
                for _ in 0..<1000 where shouldContinue() {
                    produced.withLock { $0 += 1 }
                    guard emit(big) else { break }
                }
                stoppedBecauseCancelled.withLock { $0 = !shouldContinue() }
            })
        defer { running.shutdown() }

        // Sends the request and then never reads: the socket buffers fill, the write waits, and the
        // transport's write deadline ends the stream.
        let client = try ClientSocket(port: running.port)
        defer { client.close() }
        client.send(running.request(body: #"{"messages":[{"role":"user","content":"flood"}]}"#))
        #expect(running.events.awaitNext(.slotReleased))

        // A reader that never read took the socket's buffers and no more.
        #expect(produced.withLock { $0 } < 100)
        #expect(stoppedBecauseCancelled.withLock { $0 })
        #expect(try running.exchange(running.request()) == Expected.stream())
    }

    // MARK: - S4: the slot is released on every exit

    @Test("S4: a response discarded before its stream starts releases the slot")
    func discardedResponseReleasesTheSlot() async throws {
        let events = ServerEvents()
        let service = ChatService(host: "127.0.0.1", port: 0, authentication: .unauthenticatedLoopback,
                                  allowedOrigins: [], limits: InferenceLimits(), tls: nil,
                                  tokenizer: RunningServer.byteTokenizer, observer: { events.record($0) },
                                  accessLog: { _ in }) { _, _, _ in }
        var headers = HTTPFields()
        headers[.contentLength] = "40"
        let request = try Request(method: .post, target: ChatService.route, headers: headers,
                                  body: Data(#"{"messages":[{"role":"user","content":"hi"}]}"#.utf8))

        let first = try await service.handle(request)
        // The connection died between the handler returning and the stream starting: the transport
        // discards the stream instead of running it.
        try #require(first.stream).discard()
        #expect(events.awaitNext(.slotReleased))

        // The next request is admitted at once, not queued behind a slot nobody holds.
        let second = try await service.handle(request)
        #expect(second.status == .ok)
        #expect(events.count(of: .requestQueued) == 0)
        try #require(second.stream).discard()
    }

    @Test("S4: clients that vanish right after sending never starve the next request", arguments: 0..<3)
    func vanishingClientsDoNotLeakTheSlot(round: Int) throws {
        let running = try RunningServer(respond: { _, _, emit in _ = emit("x") })
        defer { running.shutdown() }

        for _ in 0..<30 {
            let client = try ClientSocket(port: running.port)
            client.send(running.request())
            var linger = linger(l_onoff: 1, l_linger: 0)
            setsockopt(client.descriptor, SOL_SOCKET, SO_LINGER, &linger, socklen_t(MemoryLayout<linger>.size))
            client.close()
        }
        let answer = try running.exchange(running.request())
        #expect(answer.hasPrefix("HTTP/1.1 200 OK"))
    }

    // MARK: - S5: the cap, and a chunked upload

    @Test("S5: past the connection cap the transport closes with no response, so there is no Retry-After")
    func connectionCap() throws {
        var limits = InferenceLimits()
        limits.maxConnections = 1
        let running = try RunningServer(limits: limits)
        defer { running.shutdown() }

        let holder = try ClientSocket(port: running.port)
        defer { holder.close() }
        #expect(try running.rawExchange(running.request()) == "")
        #expect(running.handlerCalls == [])
    }

    private static func chunked(_ payload: String, running: RunningServer) -> [UInt8] {
        let size = String(payload.utf8.count, radix: 16)
        return Array(("POST /v1/chat/completions HTTP/1.1\r\nHost: \(running.authority)\r\n"
            + "Authorization: Bearer \(RunningServer.key)\r\nTransfer-Encoding: chunked\r\n\r\n"
            + "\(size)\r\n\(payload)\r\n0\r\n\r\n").utf8)
    }

    @Test("S5: a chunked upload within the bound is accepted; one past it is refused 413")
    func chunkedUploads() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let small = #"{"messages":[{"role":"user","content":"hi"}]}"#
        #expect(try running.exchange(Self.chunked(small, running: running)) == Expected.stream())
        let large = #"{"messages":[{"role":"user","content":""# + String(repeating: "a", count: 70_000) + #""}]}"#
        let answer = try running.exchange(Self.chunked(large, running: running))
        #expect(answer == Expected.refusal("413 Payload Too Large", "Request body exceeds the limit of 65536 bytes."))
        #expect(running.handlerCalls == [HandlerCall(prompt: "hi", maxTokens: 100)])
    }
}
}
