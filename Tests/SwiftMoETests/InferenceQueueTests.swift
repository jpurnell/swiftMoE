import Foundation
import os
import Testing
@testable import SwiftMoE

/// The one inference slot: who holds it, who waits, and how a wait ends.
@Suite("Inference queue")
struct InferenceQueueTests {

    /// Far enough away that a test which reaches it has hung.
    private static var distant: ContinuousClock.Instant { ContinuousClock.now + .seconds(30) }

    /// Runs `enter` on a thread of its own and reports the outcome when asked.
    private final class Waiter: Sendable {
        private let outcome = OSAllocatedUnfairLock<InferenceQueue.Admission?>(initialState: nil)
        private let queued = DispatchSemaphore(value: 0)
        private let finished = DispatchSemaphore(value: 0)
        private let abandoned = OSAllocatedUnfairLock(initialState: false)

        init(_ queue: InferenceQueue, deadline: ContinuousClock.Instant) {
            Thread.detachNewThread { [self] in
                let admission = queue.enter(deadline: deadline,
                                            isAbandoned: { self.abandoned.withLock { $0 } },
                                            onQueued: { self.queued.signal() })
                outcome.withLock { $0 = admission }
                finished.signal()
            }
        }

        /// Marks this waiter's client as gone.
        func abandon() { abandoned.withLock { $0 = true } }

        /// Waits until the waiter has joined the queue.
        func awaitQueued() -> Bool { queued.wait(timeout: .now() + .seconds(10)) == .success }

        /// Waits for `enter` to return.
        func result() -> InferenceQueue.Admission? {
            guard finished.wait(timeout: .now() + .seconds(10)) == .success else { return nil }
            return outcome.withLock { $0 }
        }
    }

    @Test("A free slot is taken at once, without queueing")
    func freeSlot() {
        let queue = InferenceQueue()
        var queuedCalls = 0
        let admission = queue.enter(deadline: Self.distant, isAbandoned: { false }, onQueued: { queuedCalls += 1 })
        #expect(admission == .admitted)
        #expect(queuedCalls == 0)
        #expect(queue.depth == 0)
    }

    @Test("A caller that finds the slot taken times out at its deadline and leaves the queue")
    func timesOut() {
        let queue = InferenceQueue()
        #expect(queue.enter(deadline: Self.distant, isAbandoned: { false }, onQueued: {}) == .admitted)

        var queuedCalls = 0
        let admission = queue.enter(deadline: ContinuousClock.now + .milliseconds(50),
                                    isAbandoned: { false }, onQueued: { queuedCalls += 1 })
        #expect(admission == .timedOut)
        #expect(queuedCalls == 1)
        #expect(queue.depth == 0)
    }

    @Test("A deadline already past is a timeout, not a wait")
    func deadlineAlreadyPast() {
        let queue = InferenceQueue()
        #expect(queue.enter(deadline: Self.distant, isAbandoned: { false }, onQueued: {}) == .admitted)
        let admission = queue.enter(deadline: ContinuousClock.now - .seconds(1),
                                    isAbandoned: { false }, onQueued: {})
        #expect(admission == .timedOut)
        #expect(queue.depth == 0)
    }

    @Test("A waiter whose client goes away gives up its place")
    func abandoned() {
        let queue = InferenceQueue()
        #expect(queue.enter(deadline: Self.distant, isAbandoned: { false }, onQueued: {}) == .admitted)

        let waiter = Waiter(queue, deadline: Self.distant)
        #expect(waiter.awaitQueued())
        #expect(queue.depth == 1)
        waiter.abandon()
        #expect(waiter.result() == .abandoned)
        #expect(queue.depth == 0)
    }

    @Test("Leaving hands the slot to the waiters in the order they arrived")
    func firstComeFirstServed() {
        let queue = InferenceQueue()
        #expect(queue.enter(deadline: Self.distant, isAbandoned: { false }, onQueued: {}) == .admitted)

        let second = Waiter(queue, deadline: Self.distant)
        #expect(second.awaitQueued())
        let third = Waiter(queue, deadline: Self.distant)
        #expect(third.awaitQueued())
        #expect(queue.depth == 2)

        queue.leave()
        #expect(second.result() == .admitted)
        #expect(queue.depth == 1)

        queue.leave()
        #expect(third.result() == .admitted)
        #expect(queue.depth == 0)
    }

    @Test("A waiter that gave up is skipped: the slot goes to the next one still waiting")
    func abandonedWaiterIsSkipped() {
        let queue = InferenceQueue()
        #expect(queue.enter(deadline: Self.distant, isAbandoned: { false }, onQueued: {}) == .admitted)

        let second = Waiter(queue, deadline: Self.distant)
        #expect(second.awaitQueued())
        let third = Waiter(queue, deadline: Self.distant)
        #expect(third.awaitQueued())

        second.abandon()
        #expect(second.result() == .abandoned)
        queue.leave()
        #expect(third.result() == .admitted)
    }

    @Test("A waiter handed the slot just as its client leaves passes it on instead of taking it",
          arguments: 0..<20)
    func slotHandedToADepartedClient(attempt: Int) {
        let queue = InferenceQueue()
        #expect(queue.enter(deadline: Self.distant, isAbandoned: { false }, onQueued: {}) == .admitted)

        let second = Waiter(queue, deadline: Self.distant)
        #expect(second.awaitQueued())
        let third = Waiter(queue, deadline: Self.distant)
        #expect(third.awaitQueued())

        // No wait between the two: the slot reaches the second waiter before it next looks.
        second.abandon()
        queue.leave()
        #expect(second.result() == .abandoned)
        #expect(third.result() == .admitted)
        #expect(queue.depth == 0)
    }

    @Test("Once the slot has been left with nobody waiting, the next caller takes it at once")
    func slotIsReusable() {
        let queue = InferenceQueue()
        #expect(queue.enter(deadline: Self.distant, isAbandoned: { false }, onQueued: {}) == .admitted)
        queue.leave()
        #expect(queue.enter(deadline: ContinuousClock.now - .seconds(1),
                            isAbandoned: { true }, onQueued: {}) == .admitted)
    }
}
