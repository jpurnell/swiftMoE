import Foundation

/// An accepted client socket, read against a deadline.
///
/// `read(2)` on a blocking socket waits for as long as the peer cares to stay silent. Every
/// read here is preceded by `poll(2)` for no longer than the time left before the caller's
/// deadline, so no client decides how long a thread waits.
struct ClientConnection {

    /// What one read produced.
    enum ReadOutcome {
        /// One or more bytes.
        case bytes([UInt8])
        /// The deadline passed before anything arrived.
        case timedOut
        /// The peer closed its side, or the socket failed.
        case closed
    }

    /// Largest single read, whatever the caller asks for.
    private static let maxReadBytes = 16 * 1024
    private static let millisecondsPerSecond: Int64 = 1000
    private static let attosecondsPerMillisecond: Int64 = 1_000_000_000_000_000
    private static let microsecondsPerMillisecond: Int64 = 1000

    let descriptor: Int32

    /// Wraps an accepted socket and bounds how long a write to it may block.
    ///
    /// Also sets `SO_NOSIGPIPE`: a client that hangs up mid-stream must fail the write, not
    /// deliver `SIGPIPE` and take the server down with it.
    ///
    /// - Parameters:
    ///   - descriptor: The accepted socket. Not closed by this type.
    ///   - writeDeadline: Longest one `write(2)` may block.
    init(descriptor: Int32, writeDeadline: Duration) {
        self.descriptor = descriptor

        var enabled: Int32 = 1
        setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &enabled, socklen_t(MemoryLayout<Int32>.size))

        let milliseconds = Self.milliseconds(writeDeadline)
        var timeout = timeval(
            tv_sec: Int(milliseconds / Self.millisecondsPerSecond),
            tv_usec: Int32((milliseconds % Self.millisecondsPerSecond) * Self.microsecondsPerMillisecond)
        )
        setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
    }

    /// Reads at most `count` bytes, waiting no later than `deadline`.
    ///
    /// - Parameters:
    ///   - count: Most bytes to return; capped at 16 KiB per call.
    ///   - deadline: The instant after which the read gives up.
    /// - Returns: The bytes read, or why there were none.
    func read(upTo count: Int, deadline: ContinuousClock.Instant) -> ReadOutcome {
        guard count > 0 else { return .closed }

        var remaining = Self.milliseconds(deadline - ContinuousClock.now)
        while remaining > 0 {
            var descriptors = pollfd(fd: descriptor, events: Int16(POLLIN), revents: 0)
            let ready = poll(&descriptors, 1, Int32(clamping: remaining))
            if ready == 0 { return .timedOut }
            if ready > 0 {
                let chunk = receive(upTo: min(count, Self.maxReadBytes))
                if !chunk.isEmpty { return .bytes(chunk) }
                // Readable with nothing to read is end-of-stream, unless the call was interrupted.
                guard errno == EINTR || errno == EAGAIN else { return .closed }
            } else if errno != EINTR {
                return .closed
            }
            remaining = Self.milliseconds(deadline - ContinuousClock.now)
        }
        return .timedOut
    }

    /// One `read(2)`: the bytes it returned, empty at end-of-stream or on failure.
    private func receive(upTo count: Int) -> [UInt8] {
        errno = 0
        return [UInt8](unsafeUninitializedCapacity: count) { buffer, initialized in
            guard let base = buffer.baseAddress else { return }
            initialized = max(0, Darwin.read(descriptor, base, buffer.count))
        }
    }

    /// Writes all of `bytes`, or as much as the peer accepts before failing.
    ///
    /// - Returns: `true` when every byte was written.
    @discardableResult
    func write(_ bytes: [UInt8]) -> Bool {
        var sent = 0
        while sent < bytes.count {
            let written = bytes[sent...].withUnsafeBufferPointer { buffer -> Int in
                guard let base = buffer.baseAddress else { return -1 }
                return Darwin.write(descriptor, base, buffer.count)
            }
            if written < 0, errno == EINTR { continue }
            guard written > 0 else { return false }
            sent += written
        }
        return true
    }

    /// Whether the peer has closed its side, or the socket has failed.
    ///
    /// - Parameter wait: How long to wait for that to happen; zero only looks.
    /// - Returns: `true` once the peer is gone. See ``peerHasClosed(descriptor:waitingUpTo:)``.
    func peerHasClosed(waitingUpTo wait: Duration = .zero) -> Bool {
        Self.peerHasClosed(descriptor: descriptor, waitingUpTo: wait)
    }

    /// Whether the peer of `descriptor` has closed its side, or the socket has failed.
    ///
    /// For use once a request has been read in full: nothing more is expected from the
    /// client, so end-of-stream means it has hung up, and anything else it sends is read and
    /// thrown away — at most 16 KiB a call — so that the end-of-stream behind it can be seen.
    ///
    /// A client that shuts only its sending side looks exactly like one that has closed: the
    /// kernel reports the same end-of-stream for both, and the only way to tell them apart is
    /// to write to it. It is counted as gone, as nginx and Go's `net/http` count it.
    ///
    /// - Parameters:
    ///   - descriptor: A connected socket. Anything that is not a socket is never "closed".
    ///   - wait: How long to wait for the peer to go; zero only looks.
    /// - Returns: `true` once the peer is gone, `false` if it is still there when the wait ends.
    static func peerHasClosed(descriptor: Int32, waitingUpTo wait: Duration) -> Bool {
        let deadline = ContinuousClock.now + wait
        // Always looks once; looks again only while there is time left to wait.
        var remaining = max(0, milliseconds(wait))
        var looked = false
        while !looked || remaining > 0 {
            looked = true
            var descriptors = pollfd(fd: descriptor, events: Int16(POLLIN), revents: 0)
            let ready = poll(&descriptors, 1, Int32(clamping: remaining))
            if ready > 0, discardPending(descriptor) == .endOfStream {
                return true
            }
            if ready < 0, errno != EINTR {
                return true
            }
            remaining = max(0, milliseconds(deadline - ContinuousClock.now))
        }
        return false
    }

    /// What a non-blocking read of whatever the peer has sent found.
    private enum Pending {
        /// The peer has closed, or the socket has failed.
        case endOfStream
        /// The peer is still there; any bytes it had sent were discarded.
        case stillOpen
    }

    /// Reads and drops up to 16 KiB without blocking.
    private static func discardPending(_ descriptor: Int32) -> Pending {
        var received = 0
        var failure: Int32 = 0
        _ = [UInt8](unsafeUninitializedCapacity: maxReadBytes) { buffer, initialized in
            initialized = 0
            guard let base = buffer.baseAddress else { return }
            received = recv(descriptor, base, buffer.count, MSG_DONTWAIT)
            failure = errno
        }
        if received > 0 { return .stillOpen }
        if received == 0 { return .endOfStream }
        // Nothing to read yet, an interrupted call, or not a socket at all: no evidence of a close.
        let inconclusive = [EAGAIN, EWOULDBLOCK, EINTR, ENOTSOCK]
        return inconclusive.contains(failure) ? .stillOpen : .endOfStream
    }

    /// Shuts the write side, so the peer sees the response end while this side can still read.
    func finishWriting() {
        shutdown(descriptor, SHUT_WR)
    }

    /// Whole milliseconds in `duration`, rounded up so a positive duration is never zero.
    static func milliseconds(_ duration: Duration) -> Int64 {
        let (seconds, attoseconds) = duration.components
        let (scaled, overflow) = seconds.multipliedReportingOverflow(by: millisecondsPerSecond)
        guard !overflow else { return seconds > 0 ? Int64.max : Int64.min }
        let fraction = (attoseconds + attosecondsPerMillisecond - 1) / attosecondsPerMillisecond
        let (total, sumOverflow) = scaled.addingReportingOverflow(fraction)
        return sumOverflow ? Int64.max : total
    }
}
