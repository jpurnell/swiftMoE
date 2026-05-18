import Testing
import Foundation
@testable import SwiftMoE

/// Tests for IOPool — parallel pread with latency tracking.
@Suite("IOPool")
struct IOPoolTests {

    @Test("Reads multiple experts in parallel and returns correct data")
    func parallelRead() async throws {
        let expertSize = 1024
        let expertCount = 8
        let path = try ExpertFileTests.createFixtureFile(
            expertCount: expertCount, expertSize: expertSize
        )
        defer { ExpertFileTests.removeFixture(path) }

        let file = try ExpertFile(path: path)
        let pool = IOPool(concurrency: 4)

        let indices = [0, 3, 5, 7]

        // Allocate stable memory for pread destinations
        let ptrs = indices.map { _ in UnsafeMutableRawPointer.allocate(byteCount: expertSize, alignment: 1) }
        defer { ptrs.forEach { $0.deallocate() } }

        try await pool.readExperts(
            from: file,
            indices: indices,
            into: ptrs,
            expertSize: expertSize
        )

        // Verify each expert has the correct byte pattern
        for (i, expertIdx) in indices.enumerated() {
            let byte0 = ptrs[i].load(as: UInt8.self)
            let byte1 = ptrs[i].load(fromByteOffset: 1, as: UInt8.self)
            #expect(byte0 == UInt8(truncatingIfNeeded: expertIdx),
                    "Expert \(expertIdx) byte 0 mismatch")
            #expect(byte1 == UInt8(truncatingIfNeeded: expertIdx &+ 1),
                    "Expert \(expertIdx) byte 1 mismatch")
        }
    }

    @Test("Reports latency statistics")
    func latencyTracking() async throws {
        let expertSize = 512
        let path = try ExpertFileTests.createFixtureFile(expertCount: 4, expertSize: expertSize)
        defer { ExpertFileTests.removeFixture(path) }

        let file = try ExpertFile(path: path)
        let pool = IOPool(concurrency: 2)

        let ptr0 = UnsafeMutableRawPointer.allocate(byteCount: expertSize, alignment: 1)
        let ptr1 = UnsafeMutableRawPointer.allocate(byteCount: expertSize, alignment: 1)
        defer { ptr0.deallocate(); ptr1.deallocate() }

        let stats = try await pool.readExpertsWithTiming(
            from: file,
            indices: [0, 1],
            into: [ptr0, ptr1],
            expertSize: expertSize
        )

        #expect(stats.totalMs.isFinite, "Latency should be a finite value")
        #expect(stats.readCount == 2, "Should report 2 reads")
    }

    @Test("Single expert read works correctly")
    func singleExpert() async throws {
        let expertSize = 256
        let path = try ExpertFileTests.createFixtureFile(expertCount: 2, expertSize: expertSize)
        defer { ExpertFileTests.removeFixture(path) }

        let file = try ExpertFile(path: path)
        let pool = IOPool(concurrency: 4)

        let ptr = UnsafeMutableRawPointer.allocate(byteCount: expertSize, alignment: 1)
        defer { ptr.deallocate() }

        try await pool.readExperts(
            from: file,
            indices: [1],
            into: [ptr],
            expertSize: expertSize
        )

        #expect(ptr.load(as: UInt8.self) == 1)  // Expert 1
    }
}
