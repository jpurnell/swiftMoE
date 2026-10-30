import Foundation
import os
import Testing
@testable import SwiftMoEHTTP

/// The one inference slot: who holds it, who waits, and how a wait ends.
@Suite("Inference gate")
struct InferenceGateTests {

    /// Far enough away that a test which reaches it has hung.
    private static var distant: ContinuousClock.Instant { ContinuousClock.now + .seconds(30) }

    /// How a wait for the slot ended.
    private enum Outcome: Equatable {
        case admitted
        case timedOut
        case abandoned
    }

    /// Asks for the slot in a task of its own.
    private final class Waiter: Sendable {
        private let task: Task<Outcome, Never>
        private let queued: AsyncStream<Void>

        init(_ gate: InferenceGate, deadline: ContinuousClock.Instant) {
            let (stream, continuation) = AsyncStream<Void>.makeStream()
            queued = stream
            task = Task {
                do {
                    _ = try await gate.admit(deadline: deadline, onQueued: { continuation.yield() })
                    return .admitted
                } catch InferenceGateError.timedOut {
                    return .timedOut
                } catch {
                    return .abandoned
                }
            }
        }

        /// Marks this waiter's client as gone: the transport cancels the handler's task.
        func abandon() { task.cancel() }

        /// Waits until the waiter has joined the queue.
        func awaitQueued() async -> Bool {
            var iterator = queued.makeAsyncIterator()
            return await iterator.next() != nil
        }

        func result() async -> Outcome { await task.value }
    }

    private static func take(_ gate: InferenceGate) async throws -> InferenceSlot {
        try await gate.admit(deadline: distant)
    }

    @Test("A free slot is taken at once, without queueing")
    func freeSlot() async throws {
        let gate = InferenceGate()
        let queuedCalls = OSAllocatedUnfairLock(initialState: 0)
        _ = try await gate.admit(deadline: Self.distant, onQueued: { queuedCalls.withLock { $0 += 1 } })
        #expect(queuedCalls.withLock { $0 } == 0)
        #expect(await gate.depth == 0)
    }

    @Test("A caller that finds the slot taken times out at its deadline and leaves the queue")
    func timesOut() async throws {
        let gate = InferenceGate()
        _ = try await Self.take(gate)

        let queuedCalls = OSAllocatedUnfairLock(initialState: 0)
        await #expect(throws: InferenceGateError.timedOut) {
            _ = try await gate.admit(deadline: ContinuousClock.now + .milliseconds(50),
                                     onQueued: { queuedCalls.withLock { $0 += 1 } })
        }
        #expect(queuedCalls.withLock { $0 } == 1)
        #expect(await gate.depth == 0)
    }

    @Test("A deadline already past is a timeout, not a wait")
    func deadlineAlreadyPast() async throws {
        let gate = InferenceGate()
        _ = try await Self.take(gate)
        await #expect(throws: InferenceGateError.timedOut) {
            _ = try await gate.admit(deadline: ContinuousClock.now - .seconds(1))
        }
        #expect(await gate.depth == 0)
    }

    @Test("A waiter whose client goes away gives up its place")
    func abandoned() async throws {
        let gate = InferenceGate()
        _ = try await Self.take(gate)
        let waiter = Waiter(gate, deadline: Self.distant)
        #expect(await waiter.awaitQueued())
        #expect(await gate.depth == 1)
        waiter.abandon()
        #expect(await waiter.result() == .abandoned)
        #expect(await gate.depth == 0)
    }

    @Test("Leaving hands the slot to the waiters in the order they arrived")
    func firstComeFirstServed() async throws {
        let gate = InferenceGate()
        _ = try await Self.take(gate)

        let second = Waiter(gate, deadline: Self.distant)
        #expect(await second.awaitQueued())
        let third = Waiter(gate, deadline: Self.distant)
        #expect(await third.awaitQueued())
        #expect(await gate.depth == 2)

        await gate.release()
        let secondResult = await second.result()
        #expect(secondResult == .admitted)
        #expect(await gate.depth == 1)

        await gate.release()
        let thirdResult = await third.result()
        #expect(thirdResult == .admitted)
        #expect(await gate.depth == 0)
    }

    @Test("A waiter that gave up is skipped: the slot goes to the next one still waiting")
    func abandonedWaiterIsSkipped() async throws {
        let gate = InferenceGate()
        _ = try await Self.take(gate)

        let second = Waiter(gate, deadline: Self.distant)
        #expect(await second.awaitQueued())
        let third = Waiter(gate, deadline: Self.distant)
        #expect(await third.awaitQueued())

        second.abandon()
        #expect(await second.result() == .abandoned)
        await gate.release()
        let thirdResult = await third.result()
        #expect(thirdResult == .admitted)
    }

    @Test("A waiter handed the slot just as its client leaves passes it on instead of taking it",
          arguments: 0..<20)
    func slotHandedToADepartedClient(attempt: Int) async throws {
        let gate = InferenceGate()
        _ = try await Self.take(gate)

        let second = Waiter(gate, deadline: Self.distant)
        #expect(await second.awaitQueued())
        let third = Waiter(gate, deadline: Self.distant)
        #expect(await third.awaitQueued())

        // No wait between the two: the slot may reach the second waiter as it is cancelled.
        second.abandon()
        await gate.release()
        #expect(await second.result() == .abandoned)
        let thirdResult = await third.result()
        #expect(thirdResult == .admitted)
        #expect(await gate.depth == 0)
    }

    @Test("Once the slot has been released with nobody waiting, the next caller takes it at once")
    func slotIsReusable() async throws {
        let gate = InferenceGate()
        let slot = try await Self.take(gate)
        slot.release()
        // The release hops onto the gate; the next admit waits behind it and is handed the slot.
        let next = Waiter(gate, deadline: ContinuousClock.now + .seconds(5))
        #expect(await next.result() == .admitted)
    }

    @Test("A slot is given up once, however many times it is released")
    func releaseIsIdempotent() async throws {
        let gate = InferenceGate()
        let slot = try await Self.take(gate)
        let next = Waiter(gate, deadline: Self.distant)
        #expect(await next.awaitQueued())
        let other = Waiter(gate, deadline: ContinuousClock.now + .milliseconds(200))
        #expect(await other.awaitQueued())
        slot.release()
        slot.release()
        slot.release()
        let nextResult = await next.result()
        #expect(nextResult == .admitted)
        // The extra releases did not hand the slot to the third caller while the second holds it.
        #expect(await other.result() == .timedOut)
    }
}
