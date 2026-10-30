import Foundation
import SwiftMoE

/// The bounds one request and one connection are held to.
///
/// Every number a caller could otherwise choose — how many tokens to generate, how many
/// bytes to send, how long to take sending them — has a ceiling here, and a request over a
/// ceiling is refused with a status and a sentence rather than trimmed to fit. A caller who
/// asked for 20,000 tokens and silently received 8,192 has been given a different answer
/// from the one requested, with nothing to say so; a 400 that names the range can be acted on.
///
/// ```swift
/// var limits = InferenceLimits()
/// limits.maxCompletionTokens = 2048
/// print(limits.maxBodyBytes)   // 65536
/// ```
public struct InferenceLimits: Sendable, Equatable {

    /// Most tokens one request may ask for. Default `TokenGenerator.defaultMaxSequenceLength`.
    ///
    /// It may not exceed ``maxSequenceTokens``, and a request is also held to that limit
    /// with its prompt counted in, so a count this large only fits beside an empty prompt.
    public var maxCompletionTokens: Int = TokenGenerator.defaultMaxSequenceLength

    /// Most positions one request may occupy, prompt and completion together. Default
    /// `TokenGenerator.defaultMaxSequenceLength`.
    ///
    /// Set it to the capacity of the generator behind the service
    /// (`TokenGenerator.maxSequenceLength`). A request whose prompt tokens plus requested
    /// completion tokens come to more than this is refused with a 400 that gives all three
    /// numbers. The completion is not shortened to fit.
    public var maxSequenceTokens: Int = TokenGenerator.defaultMaxSequenceLength

    /// Tokens generated when a request does not say. Default 100.
    public var defaultCompletionTokens: Int = 100

    /// Most bytes of request line and headers. Default 16 KiB.
    public var maxHeaderBytes: Int = 16 * 1024

    /// Most bytes of request body. Default 64 KiB. A larger declared body is refused with 413
    /// before any of it is read.
    public var maxBodyBytes: Int = 64 * 1024

    /// How long the head, and then the body, of a request may each take to arrive. Default 10 s.
    public var readDeadline: Duration = .seconds(10)

    /// How long one write of the stream may wait for the client to read before the stream is
    /// ended. Default 30 s.
    public var writeDeadline: Duration = .seconds(30)

    /// How long a validated request may wait for the model before it is answered 503 with
    /// `Retry-After`. Default 30 s.
    public var queueDeadline: Duration = .seconds(30)

    /// The longest a completion may take to deliver, from its first byte to its last. Default
    /// two hours: at the measured 4.4 tokens per second, 8,192 tokens take about 31 minutes, so
    /// this bounds the slowest configuration while still ending a stream that never would.
    public var maxStreamDuration: Duration = .seconds(2 * 3600)

    /// How long a graceful shutdown lets a stream in flight finish. Default 10 s.
    public var shutdownGrace: Duration = .seconds(10)

    /// Most connections served at once. Default 16. Further connections are refused by the
    /// transport; waiting requests are connections, so at most `maxConnections - 1` can be
    /// queued behind the one that is running.
    public var maxConnections: Int = 16

    /// Creates the default limits.
    public init() {}

    /// `Retry-After` for a refusal that asks the caller to come back: the queue deadline in
    /// whole seconds, rounded up, never zero.
    public var retryAfterSeconds: Int {
        let parts = queueDeadline.components
        guard queueDeadline > .zero else { return 1 }
        let milliseconds = parts.seconds.multipliedReportingOverflow(by: 1000)
        guard !milliseconds.overflow else { return Int(Int32.max) }
        let total = milliseconds.partialValue + parts.attoseconds / 1_000_000_000_000_000
        let (rounded, overflow) = total.addingReportingOverflow(999)
        return Int(max(1, (overflow ? total : rounded) / 1000))
    }

    /// Checks that every limit is positive, the default token count fits the completion
    /// ceiling, and the completion ceiling fits the sequence.
    ///
    /// - Throws: ``ServiceError/invalidLimit(name:)`` naming the first limit that is not
    ///   positive; ``ServiceError/limitAboveLimit(name:ceiling:)`` naming the first that
    ///   is larger than the limit bounding it.
    public func validate() throws {
        let counts: [(String, Int)] = [
            ("maxHeaderBytes", maxHeaderBytes),
            ("maxBodyBytes", maxBodyBytes),
            ("maxSequenceTokens", maxSequenceTokens),
            ("maxCompletionTokens", maxCompletionTokens),
            ("defaultCompletionTokens", defaultCompletionTokens),
            ("maxConnections", maxConnections),
        ]
        for (name, value) in counts where value <= 0 {
            throw ServiceError.invalidLimit(name: name)
        }
        guard maxCompletionTokens <= maxSequenceTokens else {
            throw ServiceError.limitAboveLimit(name: "maxCompletionTokens", ceiling: "maxSequenceTokens")
        }
        guard defaultCompletionTokens <= maxCompletionTokens else {
            throw ServiceError.limitAboveLimit(name: "defaultCompletionTokens", ceiling: "maxCompletionTokens")
        }
        let deadlines: [(String, Duration)] = [
            ("readDeadline", readDeadline),
            ("writeDeadline", writeDeadline),
            ("queueDeadline", queueDeadline),
            ("maxStreamDuration", maxStreamDuration),
        ]
        for (name, value) in deadlines where value <= .zero {
            throw ServiceError.invalidLimit(name: name)
        }
    }
}
