import Foundation
import os
import Testing
@testable import SwiftMoE

/// What happens to a request that arrives while the model is busy, and to the model when the
/// client it is working for leaves.
///
/// A validated request used to wait on a lock behind the running inference: no deadline, no
/// notice taken of a client that hung up, one of the connection slots held throughout.
@Suite("HTTP server queue and disconnects")
struct HTTPServerQueueTests {

    /// A stub handler that holds the model, for any request whose prompt is `hold`, until told
    /// to let go.
    private final class Hold: Sendable {
        /// Generous enough that reaching it means the test has hung.
        private static let allowance: DispatchTimeInterval = .seconds(30)

        private let started = DispatchSemaphore(value: 0)
        private let released = DispatchSemaphore(value: 0)

        var respond: StubResponse {
            { [self] request, writer in
                if request.prompt == "hold" {
                    started.signal()
                    _ = released.wait(timeout: .now() + Self.allowance)
                }
                RunningServer.emptyStream(request, writer)
            }
        }

        /// Waits until the handler is holding the model.
        func awaitStarted() -> Bool { started.wait(timeout: .now() + Self.allowance) == .success }

        /// Lets the held request finish.
        func release() { released.signal() }
    }

    private static func body(_ prompt: String) -> String {
        #"{"messages":[{"role":"user","content":"\#(prompt)"}]}"#
    }

    /// Opens a connection and sends a whole request on it, leaving the response unread.
    private static func send(_ prompt: String, to running: RunningServer) throws -> ClientSocket {
        let client = try ClientSocket(port: running.bound.port)
        client.send(running.request(body: body(prompt)))
        return client
    }

    // MARK: - The limit

    @Test("The queue deadline defaults to 30 s, and Retry-After is that in whole seconds")
    func defaults() {
        let limits = HTTPServer.Limits()
        #expect(limits.queueDeadline == .seconds(30))
        #expect(limits.retryAfterSeconds == 30)
    }

    @Test("Retry-After rounds the queue deadline up to a whole second and is never zero",
          arguments: [(Duration.milliseconds(1), 1), (.milliseconds(999), 1), (.seconds(1), 1),
                      (.milliseconds(1001), 2), (.milliseconds(2500), 3), (.seconds(120), 120)])
    func retryAfter(deadline: Duration, seconds: Int) {
        var limits = HTTPServer.Limits()
        limits.queueDeadline = deadline
        #expect(limits.retryAfterSeconds == seconds)
    }

    @Test("A queue deadline of zero stops the listener from opening")
    func zeroDeadline() {
        var limits = HTTPServer.Limits()
        limits.queueDeadline = .zero
        let server = HTTPServer(port: 0, authentication: .unauthenticatedLoopback, limits: limits,
                                tokenizer: RunningServer.byteTokenizer) { _, _ in }
        #expect(throws: HTTPServerError.invalidLimit(name: "queueDeadline")) {
            _ = try server.openListener()
        }
        #expect(server.boundAddress == nil)
    }

    // MARK: - Waiting

    @Test("A request that cannot start within the queue deadline is 503 with Retry-After, and never runs")
    func queuedRequestTimesOut() throws {
        var limits = HTTPServer.Limits()
        limits.queueDeadline = .milliseconds(100)
        let hold = Hold()
        let running = try RunningServer(limits: limits, respond: hold.respond)
        defer { running.shutdown() }

        let first = try Self.send("hold", to: running)
        defer { first.close() }
        #expect(hold.awaitStarted())

        let response = try running.exchange(running.request(body: Self.body("second")))
        #expect(response == Expected.queueTimedOut(retryAfter: 1))
        #expect(running.events.count(of: .queuedRequestTimedOut) == 1)

        hold.release()
        #expect(first.readToEnd() == Expected.stream())
        #expect(running.handlerCalls == [HandlerCall(prompt: "hold", maxTokens: 100)])
    }

    @Test("The refusal reaches a queued client that is still uploading, every time", arguments: 0..<10)
    func queueRefusalIsDelivered(attempt: Int) throws {
        var limits = HTTPServer.Limits()
        limits.queueDeadline = .milliseconds(50)
        let hold = Hold()
        let running = try RunningServer(limits: limits, respond: hold.respond)
        defer { running.shutdown() }

        let first = try Self.send("hold", to: running)
        defer { first.close() }
        #expect(hold.awaitStarted())

        // A complete request with more bytes behind it, as a pipelining client would send.
        let client = try ClientSocket(port: running.bound.port)
        defer { client.close() }
        client.send(running.request(body: Self.body("second")) + [UInt8](repeating: 0x20, count: 64 * 1024))
        #expect(client.readToEnd() == Expected.queueTimedOut(retryAfter: 1))

        hold.release()
        #expect(first.readToEnd() == Expected.stream())
    }

    @Test("A request queued behind another runs when the model is free, within its deadline")
    func queuedRequestRuns() throws {
        let hold = Hold()
        let running = try RunningServer(respond: hold.respond)
        defer { running.shutdown() }

        let first = try Self.send("hold", to: running)
        defer { first.close() }
        #expect(hold.awaitStarted())

        let second = try Self.send("second", to: running)
        defer { second.close() }
        #expect(running.events.awaitNext(.requestQueued))

        hold.release()
        #expect(first.readToEnd() == Expected.stream())
        #expect(second.readToEnd() == Expected.stream())
        #expect(running.handlerCalls == [
            HandlerCall(prompt: "hold", maxTokens: 100),
            HandlerCall(prompt: "second", maxTokens: 100),
        ])
    }

    @Test("Queued requests run in the order they arrived")
    func queueIsFirstComeFirstServed() throws {
        let hold = Hold()
        let running = try RunningServer(respond: hold.respond)
        defer { running.shutdown() }

        let first = try Self.send("hold", to: running)
        defer { first.close() }
        #expect(hold.awaitStarted())

        var waiting: [ClientSocket] = []
        defer { waiting.forEach { $0.close() } }
        for prompt in ["second", "third", "fourth"] {
            waiting.append(try Self.send(prompt, to: running))
            #expect(running.events.awaitNext(.requestQueued))
        }

        hold.release()
        #expect(first.readToEnd() == Expected.stream())
        for client in waiting {
            #expect(client.readToEnd() == Expected.stream())
        }
        #expect(running.handlerCalls.map(\.prompt) == ["hold", "second", "third", "fourth"])
    }

    @Test("A request that does not fit in a sequence is refused at once, not queued to be refused later")
    func budgetIsCheckedBeforeQueueing() throws {
        let hold = Hold()
        let running = try RunningServer(respond: hold.respond)
        defer { running.shutdown() }

        let first = try Self.send("hold", to: running)
        defer { first.close() }
        #expect(hold.awaitStarted())

        let response = try running.exchange(running.request(
            body: Self.body(String(repeating: "p", count: 9000))))
        #expect(response == Expected.refusal(
            "400 Bad Request",
            "Prompt (9000 tokens) plus completion (100 tokens) is 9100 tokens; "
                + "the limit for a sequence is 8192. Shorten the prompt or lower max_tokens."))
        #expect(running.events.count(of: .requestQueued) == 0)

        hold.release()
        #expect(first.readToEnd() == Expected.stream())
    }

    // MARK: - Clients that leave

    @Test("A client that disconnects while queued frees its connection and is never run")
    func disconnectWhileQueued() throws {
        var limits = HTTPServer.Limits()
        limits.maxConnections = 2
        let hold = Hold()
        let running = try RunningServer(limits: limits, respond: hold.respond)
        defer { running.shutdown() }

        let first = try Self.send("hold", to: running)
        defer { first.close() }
        #expect(hold.awaitStarted())

        let second = try Self.send("second", to: running)
        #expect(running.events.awaitNext(.requestQueued))
        second.close()
        #expect(running.events.awaitNext(.queuedRequestAbandoned))
        #expect(running.events.awaitNext(.connectionFinished))

        // Both connections were in use; the third is served only because the second's was freed.
        let third = try Self.send("third", to: running)
        defer { third.close() }
        #expect(running.events.awaitNext(.requestQueued))

        hold.release()
        #expect(first.readToEnd() == Expected.stream())
        #expect(third.readToEnd() == Expected.stream())
        #expect(running.handlerCalls.map(\.prompt) == ["hold", "third"])
    }

    @Test("The handler can see that its client has left mid-stream, and that one still there has not")
    func disconnectMidStream() throws {
        let observed = OSAllocatedUnfairLock<[Bool]>(initialState: [])
        let streaming = DispatchSemaphore(value: 0)
        let running = try RunningServer(respond: { _, writer in
            writer.sendHeaders()
            observed.withLock { $0.append(writer.clientHasDisconnected) }
            streaming.signal()
            // Returns as soon as the client's close arrives; the allowance only bounds a hang.
            observed.withLock { $0.append(writer.clientHasDisconnected(within: .seconds(10))) }
            observed.withLock { $0.append(writer.clientHasDisconnected) }
        })
        defer { running.shutdown() }

        let client = try Self.send("stream", to: running)
        #expect(streaming.wait(timeout: .now() + .seconds(10)) == .success)
        client.close()
        #expect(running.events.awaitNext(.connectionFinished))

        #expect(observed.withLock { $0 } == [false, true, true])
    }

    @Test("A client that only shuts its sending side counts as gone: it is indistinguishable from one that left")
    func halfCloseCountsAsGone() throws {
        let observed = OSAllocatedUnfairLock<[Bool]>(initialState: [])
        let streaming = DispatchSemaphore(value: 0)
        let running = try RunningServer(respond: { _, writer in
            writer.sendHeaders()
            streaming.signal()
            observed.withLock { $0.append(writer.clientHasDisconnected(within: .seconds(10))) }
        })
        defer { running.shutdown() }

        let client = try Self.send("stream", to: running)
        defer { client.close() }
        #expect(streaming.wait(timeout: .now() + .seconds(10)) == .success)
        shutdown(client.descriptor, SHUT_WR)
        #expect(running.events.awaitNext(.connectionFinished))

        #expect(observed.withLock { $0 } == [true])
    }

    @Test("Bytes a client sends after its request are not mistaken for a disconnect")
    func trailingBytesAreNotADisconnect() throws {
        let observed = OSAllocatedUnfairLock<[Bool]>(initialState: [])
        let streaming = DispatchSemaphore(value: 0)
        let trailingSent = DispatchSemaphore(value: 0)
        let running = try RunningServer(respond: { request, writer in
            writer.sendHeaders()
            streaming.signal()
            _ = trailingSent.wait(timeout: .now() + .seconds(10))
            // Whether or not the trailing bytes have arrived yet, the client is still there.
            observed.withLock { $0.append(writer.clientHasDisconnected) }
            observed.withLock { $0.append(writer.clientHasDisconnected) }
            writer.sendDone()
        })
        defer { running.shutdown() }

        let client = try Self.send("stream", to: running)
        defer { client.close() }
        #expect(streaming.wait(timeout: .now() + .seconds(10)) == .success)
        client.send(Array("trailing bytes".utf8))
        trailingSent.signal()

        #expect(client.readToEnd() == Expected.stream())
        #expect(observed.withLock { $0 } == [false, false])
    }

    @Test("A stream reaches a client that is still sending bytes after its request, every time",
          arguments: 0..<20)
    func streamIsDeliveredDespiteTrailingBytes(attempt: Int) throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        // The request is complete at Content-Length; half a megabyte follows it, more than the
        // socket buffers hold, so the client is still in `send` when the stream is written.
        // Closing over the unread remainder would reset the connection and lose the stream.
        let trailing = [UInt8](repeating: 0x20, count: 512 * 1024)
        let response = try running.exchange(running.request(body: Self.body("stream")) + trailing)
        #expect(response == Expected.stream())
        #expect(running.handlerCalls == [HandlerCall(prompt: "stream", maxTokens: 100)])
    }

    @Test("A client that leaves while its request is queued for the model does not get the model")
    func abandonedRequestSkipsItsTurn() throws {
        let hold = Hold()
        let running = try RunningServer(respond: hold.respond)
        defer { running.shutdown() }

        let first = try Self.send("hold", to: running)
        defer { first.close() }
        #expect(hold.awaitStarted())

        let second = try Self.send("second", to: running)
        #expect(running.events.awaitNext(.requestQueued))
        let third = try Self.send("third", to: running)
        defer { third.close() }
        #expect(running.events.awaitNext(.requestQueued))

        second.close()
        #expect(running.events.awaitNext(.queuedRequestAbandoned))

        hold.release()
        #expect(first.readToEnd() == Expected.stream())
        #expect(third.readToEnd() == Expected.stream())
        #expect(running.handlerCalls.map(\.prompt) == ["hold", "third"])
    }
}
