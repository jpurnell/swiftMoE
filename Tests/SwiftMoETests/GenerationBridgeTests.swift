import Foundation
import os
import Testing
@testable import SwiftMoEHTTP

/// The thread that runs a generation, and the async stream that carries its tokens.
@Suite("Generation bridge")
struct GenerationBridgeTests {

    /// Far enough away that a test which reaches it has hung.
    private static let allowance: DispatchTimeInterval = .seconds(10)

    /// Waits for `semaphore` off the cooperative pool.
    private static func signalled(_ semaphore: DispatchSemaphore) async -> Bool {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                continuation.resume(returning: semaphore.wait(timeout: .now() + allowance) == .success)
            }
        }
    }

    @Test("Tokens arrive in order and the stream finishes when the generator returns")
    func tokensInOrder() async throws {
        let bridge = GenerationBridge { _, emit in
            for word in ["a", "b", "c"] { _ = emit(word) }
        }
        bridge.start()
        var received: [String] = []
        for try await token in bridge.tokens {
            received.append(token)
            bridge.consumed()
        }
        #expect(received == ["a", "b", "c"])
        await bridge.stopped()
    }

    @Test("A generator that throws ends the stream with that error")
    func failureIsDelivered() async {
        struct Failure: Error, Equatable {}
        let bridge = GenerationBridge { _, emit in
            _ = emit("a")
            throw Failure()
        }
        bridge.start()
        var received: [String] = []
        do {
            for try await token in bridge.tokens {
                received.append(token)
                bridge.consumed()
            }
            Issue.record("the stream ended without the generator's error")
        } catch {
            #expect(error is Failure)
        }
        #expect(received == ["a"])
    }

    @Test("A reader that stops reading stops the generation: at most one token is produced ahead")
    func backpressure() async throws {
        let produced = OSAllocatedUnfairLock(initialState: 0)
        let reachedSecond = DispatchSemaphore(value: 0)
        let bridge = GenerationBridge { shouldContinue, emit in
            while shouldContinue() {
                produced.withLock { $0 += 1 }
                if produced.withLock({ $0 }) == 2 { reachedSecond.signal() }
                guard emit("x") else { return }
            }
        }
        bridge.start()
        var iterator = bridge.tokens.makeAsyncIterator()
        // Take one token and never return its credit: the reader is "paused" mid-write.
        _ = try await iterator.next()
        #expect(await Self.signalled(reachedSecond))
        // The generator has computed a second token and is held in emit, waiting for the credit.
        // Nothing more is produced however long it waits for one; the count stays at 2.
        try await Task.sleep(for: .milliseconds(300))
        #expect(produced.withLock { $0 } == 2)

        bridge.cancel()
        await bridge.stopped()
    }

    @Test("Cancelling is seen by shouldContinue before the next token, and stopped() waits for the thread")
    func cancellation() async throws {
        let sawCancel = OSAllocatedUnfairLock(initialState: false)
        let started = DispatchSemaphore(value: 0)
        let bridge = GenerationBridge { shouldContinue, _ in
            started.signal()
            while shouldContinue() { Thread.sleep(forTimeInterval: 0.005) }
            sawCancel.withLock { $0 = true }
        }
        bridge.start()
        #expect(await Self.signalled(started))
        bridge.cancel()
        await bridge.stopped()
        #expect(sawCancel.withLock { $0 })
        #expect(bridge.isCancelled)
    }

    @Test("emit reports false once the generation is cancelled, even with a token waiting")
    func emitAfterCancel() async {
        let results = OSAllocatedUnfairLock<[Bool]>(initialState: [])
        let first = DispatchSemaphore(value: 0)
        let bridge = GenerationBridge { _, emit in
            results.withLock { $0.append(emit("a")) }
            first.signal()
            results.withLock { $0.append(emit("b")) }
        }
        bridge.start()
        #expect(await Self.signalled(first))
        bridge.cancel()
        await bridge.stopped()
        #expect(results.withLock { $0 } == [true, false])
    }

    @Test("stopped() returns at once for a bridge that never started")
    func neverStarted() async throws {
        let bridge = GenerationBridge { _, _ in }
        bridge.abandon()
        await bridge.stopped()
        var iterator = bridge.tokens.makeAsyncIterator()
        #expect(try await iterator.next() == nil)
    }
}
