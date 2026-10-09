import Foundation
#if canImport(os)
import os
#endif

/// Admission to the one inference slot: who runs now, who waits, and for how long.
///
/// The handler drives one model on one GPU, so one request runs at a time. The rest used to
/// wait on a lock, which has no deadline and cannot notice that the client it is waiting for
/// has hung up. Here a waiter holds a place in a first-come queue and gives it up when its
/// deadline passes or its client goes away, so a request never waits longer than
/// ``HTTPServer/Limits/queueDeadline`` and a departed client stops holding a connection slot.
///
/// The queue's depth has no limit of its own because it cannot exceed
/// ``HTTPServer/Limits/maxConnections``: every waiter is a connection, and connections are
/// already counted.
final class InferenceQueue: Sendable {

    /// How a wait for the slot ended.
    enum Admission: Equatable, Sendable {
        /// The caller holds the slot and must call ``InferenceQueue/leave()``.
        case admitted
        /// The deadline passed first.
        case timedOut
        /// The caller's client went away first.
        case abandoned
    }

    private struct Waiter {
        let ticket: UInt64
        /// Signalled once, by ``leave()``, when the slot is handed to this waiter.
        let wake: DispatchSemaphore
    }

    private struct State {
        var running = false
        var nextTicket: UInt64 = 0
        /// Waiters in arrival order.
        var waiters: [Waiter] = []
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    /// How often a waiter looks up to see whether its client is still there.
    private static let abandonmentCheckMilliseconds = 100

    /// Takes the slot, waiting no later than `deadline`.
    ///
    /// The slot is handed over in arrival order. While waiting, `isAbandoned` is asked every
    /// tenth of a second, so a client that hangs up frees its place within that time rather
    /// than when its turn would have come; it is asked again at the moment the slot is
    /// handed over, so a caller that waited is never admitted for a client already gone.
    ///
    /// - Parameters:
    ///   - deadline: The instant after which the caller gives up.
    ///   - isAbandoned: Whether the caller's client has gone away.
    ///   - onQueued: Called once if the slot was taken and the caller has to wait.
    /// - Returns: ``Admission/admitted`` when the caller now holds the slot.
    func enter(
        deadline: ContinuousClock.Instant,
        isAbandoned: () -> Bool,
        onQueued: () -> Void
    ) -> Admission {
        let wake = DispatchSemaphore(value: 0)
        let ticket = state.withLock { current -> UInt64? in
            guard current.running else {
                current.running = true
                return nil
            }
            let ticket = current.nextTicket
            current.nextTicket &+= 1
            current.waiters.append(Waiter(ticket: ticket, wake: wake))
            return ticket
        }
        guard let ticket else { return .admitted }
        onQueued()

        var remaining = ClientConnection.milliseconds(deadline - ContinuousClock.now)
        while remaining > 0 {
            let slice = Int(min(remaining, Int64(Self.abandonmentCheckMilliseconds)))
            if wake.wait(timeout: .now() + .milliseconds(slice)) == .success {
                // Handed the slot. A client that left in the last tenth of a second has not
                // been noticed yet; look once more, and pass the slot on if there is nobody
                // to run for.
                guard isAbandoned() else { return .admitted }
                leave()
                return .abandoned
            }
            if isAbandoned() {
                // If the ticket has already gone, the slot arrived in the same instant: pass it on.
                if !withdraw(ticket) { leave() }
                return .abandoned
            }
            remaining = ClientConnection.milliseconds(deadline - ContinuousClock.now)
        }
        // Out of time. If the ticket has already gone, the slot was handed over just now.
        return withdraw(ticket) ? .timedOut : .admitted
    }

    /// Releases the slot, handing it to the longest-waiting caller if there is one.
    func leave() {
        let next = state.withLock { current -> DispatchSemaphore? in
            guard !current.waiters.isEmpty else {
                current.running = false
                return nil
            }
            // The slot passes straight to the next waiter: `running` stays true.
            return current.waiters.removeFirst().wake
        }
        next?.signal()
    }

    /// Requests waiting for the slot.
    var depth: Int { state.withLock { $0.waiters.count } }

    /// Takes `ticket` out of the queue.
    ///
    /// - Returns: `false` when the ticket was no longer queued, which means ``leave()`` handed
    ///   the slot over between the caller's last look and now: the caller holds the slot.
    private func withdraw(_ ticket: UInt64) -> Bool {
        state.withLock { current in
            guard let index = current.waiters.firstIndex(where: { $0.ticket == ticket }) else { return false }
            current.waiters.remove(at: index)
            return true
        }
    }
}
