import Foundation

/// Latency statistics from a parallel expert read operation.
public struct IOReadStats: Sendable {
    /// Total wall-clock time for the parallel read batch, in milliseconds.
    public let totalMs: Double

    /// Number of individual pread calls completed.
    public let readCount: Int

    /// Bytes read per second across all reads.
    public var throughputBytesPerSec: Double {
        guard totalMs > 0 else { return 0 }
        return Double(totalBytes) / (totalMs / 1000.0)
    }

    /// Total bytes read across all experts.
    public let totalBytes: Int
}

/// Manages parallel pread operations for expert weight loading.
///
/// Replaces the hand-rolled pthread pool from `infer.m:2970-3120` with
/// Swift structured concurrency. Each expert read becomes a `Task` within
/// a `TaskGroup`, matching the existing "read K=4 experts in parallel" pattern.
///
/// ## Latency Tracking
/// Use ``readExpertsWithTiming(from:indices:into:expertSize:)`` to get
/// per-batch latency statistics for performance monitoring.
public struct IOPool: Sendable {
    /// Maximum number of concurrent pread operations.
    public let concurrency: Int

    /// Creates an I/O pool with the given concurrency level.
    ///
    /// - Parameter concurrency: Number of concurrent reads. Default is 4,
    ///   matching the original Flash-MoE configuration (4 experts per token).
    public init(concurrency: Int = 4) {
        self.concurrency = concurrency
    }

    /// Reads K experts in parallel from the given file into destination buffers.
    ///
    /// Each expert is read via `pread`, which is position-independent and thread-safe.
    /// Reads execute concurrently within a `TaskGroup`.
    ///
    /// - Parameters:
    ///   - file: The expert file to read from.
    ///   - expertIndices: Which experts to load (e.g., top-K routing results).
    ///   - destinations: Pre-allocated buffers, one per expert index.
    ///   - expertSize: Bytes per expert.
    /// - Throws: ``FlashMoEError/readFailed(errno:context:)`` if any read fails.
    public func readExperts(
        from file: borrowing ExpertFile,
        indices expertIndices: [Int],
        into destinations: [UnsafeMutableRawPointer],
        expertSize: Int
    ) async throws {
        // Capture the fd value before entering the sendable closure
        let fd = file.fileDescriptor

        try await withThrowingTaskGroup(of: Void.self) { group in
            for (i, expertIdx) in expertIndices.enumerated() {
                // Each task writes to a unique, non-overlapping buffer.
                // The compiler can't prove this, so we opt out of the Sendable check.
                nonisolated(unsafe) let dst = destinations[i]
                group.addTask {
                    let offset = off_t(expertIdx) * off_t(expertSize)
                    let bytesRead = pread(fd, dst, expertSize, offset)
                    guard bytesRead >= 0 else {
                        throw FlashMoEError.readFailed(
                            errno: errno,
                            context: "pread expert \(expertIdx)"
                        )
                    }
                }
            }
            try await group.waitForAll()
        }
    }

    /// Reads K experts in parallel with latency instrumentation.
    ///
    /// Identical to ``readExperts(from:indices:into:expertSize:)`` but returns
    /// timing statistics for performance monitoring.
    ///
    /// - Returns: ``IOReadStats`` with wall-clock time and throughput.
    public func readExpertsWithTiming(
        from file: borrowing ExpertFile,
        indices expertIndices: [Int],
        into destinations: [UnsafeMutableRawPointer],
        expertSize: Int
    ) async throws -> IOReadStats {
        let start = DispatchTime.now()

        try await readExperts(
            from: file,
            indices: expertIndices,
            into: destinations,
            expertSize: expertSize
        )

        let end = DispatchTime.now()
        let nanos = end.uptimeNanoseconds - start.uptimeNanoseconds
        let totalMs = Double(nanos) / 1_000_000.0

        return IOReadStats(
            totalMs: totalMs,
            readCount: expertIndices.count,
            totalBytes: expertIndices.count * expertSize
        )
    }
}
