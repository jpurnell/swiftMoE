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
