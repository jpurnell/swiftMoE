import Foundation
#if canImport(os)
import os
#endif

private let logger = Logger(subsystem: "com.swiftmoe", category: "server.generation")

/// One generation, run on a thread of its own and delivered as a stream of token text.
///
/// `TokenGenerator.generate` is synchronous and holds its thread for as long as the completion
/// takes — up to half an hour — so it must not run on the cooperative pool or a network event
/// loop. This type owns a dedicated `Thread` for it and hands the tokens to an async consumer.
///
/// - Backpressure. The consumer returns a credit with ``consumed()`` after it has handled a
///   token, and the thread does not hand over the next token until it holds one, so at most one
///   token waits and a reader that stops reading stops the generation within a token.
/// - Cancellation. ``cancel()`` sets a flag that the generator reads through
///   `shouldContinue` before every token, prompt tokens included, and wakes a thread that is
///   waiting for a credit.
/// - Ending. ``stopped()`` returns once the thread has left the generator, so a caller can
///   release the model only when nothing is still running on it.
final class GenerationBridge: Sendable {

    /// The generator's work: the closure the service was given.
    typealias Work = @Sendable (
        _ shouldContinue: @Sendable () -> Bool,
        _ emit: @Sendable (String) -> Bool
    ) throws -> Void

    private struct State {
        var cancelled = false
        var finished = false
        var waiters: [CheckedContinuation<Void, Never>] = []
    }

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let credits = Credits(initial: 1)
    private let continuation: AsyncThrowingStream<String, any Error>.Continuation
    /// The tokens, in order. Finishes when the generator returns, and throws what it threw.
    let tokens: AsyncThrowingStream<String, any Error>
    private let work: Work

    /// Prepares a generation. Nothing runs until ``start()``.
    init(work: @escaping Work) {
        self.work = work
        let (stream, continuation) = AsyncThrowingStream<String, any Error>.makeStream(bufferingPolicy: .bufferingOldest(1))
        self.tokens = stream
        self.continuation = continuation
    }

    /// Whether generation has been asked to stop.
    var isCancelled: Bool { state.withLock { $0.cancelled } }

    /// Starts the thread.
    func start() {
        let thread = Thread { [self] in run() }
        thread.name = "swiftmoe.generation"
        thread.start()
    }

    /// Returns the credit for a token that has been handled, letting the next one through.
    func consumed() {
        credits.give()
    }

    /// Asks the generation to stop at the next token.
    func cancel() {
        state.withLock { $0.cancelled = true }
        // A thread waiting for a credit must not wait for a reader that has gone.
        credits.give()
    }

    /// Suspends until the thread has left the generator (immediately, if it never started or has finished).
    func stopped() async {
        await withCheckedContinuation { (waiter: CheckedContinuation<Void, Never>) in
            let ready = state.withLock { state -> Bool in
                guard !state.finished else { return true }
                state.waiters.append(waiter)
                return false
            }
            if ready { waiter.resume() }
        }
    }

    /// Tells a bridge that never started that nothing will run, so ``stopped()`` returns.
    func abandon() {
        markFinished()
        continuation.finish()
    }

    private func run() {
        let outcome: (any Error)?
        do {
            try work(
                { !self.isCancelled },
                { token in
                    // One token waits at most: take the credit before handing this one over.
                    self.credits.take()
                    guard !self.isCancelled else { return false }
                    if case .terminated = self.continuation.yield(token) { return false }
                    return true
                })
            outcome = nil
        } catch {
            logger.error("[generation] failed: \(String(describing: error), privacy: .public)")
            outcome = error
        }
        markFinished()
        if let outcome {
            continuation.finish(throwing: outcome)
        } else {
            continuation.finish()
        }
    }

    private func markFinished() {
        let waiters = state.withLock { state -> [CheckedContinuation<Void, Never>] in
            state.finished = true
            defer { state.waiters = [] }
            return state.waiters
        }
        waiters.forEach { $0.resume() }
    }
}

/// A count of tokens the reader is ready for, that a thread can wait on.
///
/// Not a `DispatchSemaphore`: that traps if it is destroyed while its value is below the one it
/// started with, and a generation that is cancelled mid-token ends exactly so.
// Justification: every access to `count` happens inside the condition's lock.
private final class Credits: @unchecked Sendable {
    private let condition = NSCondition()
    private var count: Int

    init(initial: Int) {
        count = initial
    }

    /// Waits for a credit and takes it.
    func take() {
        condition.lock()
        defer { condition.unlock() }
        // A generation may wait on a slow reader as long as the transport's write timeout allows; the
        // wait is taken in slices so that it is a bounded wait that is repeated, not an unbounded one.
        while count <= 0 { _ = condition.wait(until: Date(timeIntervalSinceNow: 1)) }
        count -= 1
    }

    /// Returns a credit, waking one waiter.
    func give() {
        condition.lock()
        count += 1
        condition.signal()
        condition.unlock()
    }
}
