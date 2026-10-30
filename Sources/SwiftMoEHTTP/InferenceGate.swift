import Foundation
#if canImport(os)
import os
#endif

/// Why a wait for the model ended without the model.
enum InferenceGateError: Error, Equatable, Sendable {
    /// The deadline passed before the slot was handed over.
    case timedOut
}

/// The place a request holds while it runs on the model; give it up exactly once.
///
/// Releasing is idempotent and callable from any thread, because the three ways a response can
/// end (the stream finishes, the producer fails, the response is discarded before it starts)
/// all arrive through one callback that may not be alone.
final class InferenceSlot: Sendable {
    private let gate: InferenceGate
    private let released = OSAllocatedUnfairLock(initialState: false)

    fileprivate init(gate: InferenceGate) {
        self.gate = gate
    }

    /// Hands the slot to the longest-waiting request, if any. Later calls do nothing.
    func release() {
        let first = released.withLock { state -> Bool in
            defer { state = true }
            return !state
        }
        guard first else { return }
        let gate = self.gate
        Task { await gate.release() }
    }
}

/// Admission to the one inference slot: who runs now, who waits, and for how long.
///
/// The service drives one model on one GPU, so one request runs at a time. A waiter holds a
/// place in a first-come queue and gives it up when its deadline passes or its task is
/// cancelled (the transport cancels the handler's task when the client goes away), so a
/// request never waits longer than its deadline and a departed client stops holding a place.
///
/// The queue's depth has no limit of its own because it cannot exceed the connection limit:
/// every waiter is a connection, and connections are already counted.
actor InferenceGate {

    private struct Waiter {
        let id: UInt64
        let continuation: CheckedContinuation<Void, any Error>
        let deadline: Task<Void, Never>
    }

    private var running = false
    private var nextID: UInt64 = 0
    private var waiters: [Waiter] = []

    /// Requests waiting for the slot.
    var depth: Int { waiters.count }

    /// Takes the slot, waiting no later than `deadline`.
    ///
    /// The slot is handed over in arrival order. A caller whose task is cancelled while it
    /// waits leaves the queue; one that is handed the slot at the moment it is cancelled
    /// passes the slot straight on, so a client that left is never run for.
    ///
    /// - Parameters:
    ///   - deadline: The instant after which the caller gives up.
    ///   - onQueued: Called once if the slot was taken and the caller has to wait.
    /// - Returns: The slot, which the caller must ``InferenceSlot/release()``.
    /// - Throws: ``InferenceGateError/timedOut``, or `CancellationError`.
    func admit(deadline: ContinuousClock.Instant, onQueued: @Sendable () -> Void = {}) async throws -> InferenceSlot {
        try Task.checkCancellation()
        guard running else {
            running = true
            return InferenceSlot(gate: self)
        }
        let id = nextID
        nextID &+= 1
        onQueued()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                guard !Task.isCancelled else {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                let expiry = Task { [weak self] in
                    try? await Task.sleep(until: deadline, clock: .continuous)
                    await self?.expire(id)
                }
                waiters.append(Waiter(id: id, continuation: continuation, deadline: expiry))
            }
        } onCancel: {
            Task { await self.withdraw(id, throwing: CancellationError()) }
        }
        let slot = InferenceSlot(gate: self)
        // Handed the slot in the instant the client left: do not run for nobody.
        guard !Task.isCancelled else {
            slot.release()
            throw CancellationError()
        }
        return slot
    }

    /// Gives the slot to the longest-waiting caller, or frees it.
    func release() {
        guard !waiters.isEmpty else {
            running = false
            return
        }
        // The slot passes straight to the next waiter: `running` stays true.
        let next = waiters.removeFirst()
        next.deadline.cancel()
        next.continuation.resume()
    }

    private func expire(_ id: UInt64) {
        withdraw(id, throwing: InferenceGateError.timedOut)
    }

    private func withdraw(_ id: UInt64, throwing error: any Error) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.deadline.cancel()
        waiter.continuation.resume(throwing: error)
    }
}
