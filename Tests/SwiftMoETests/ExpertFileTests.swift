import Testing
import Foundation
@testable import SwiftMoE

/// Tests for ExpertFile — the ~Copyable file descriptor wrapper for packed expert binaries.
@Suite("ExpertFile")
struct ExpertFileTests {

    /// Create a temporary test fixture file with a known byte pattern.
    /// Pattern: each "expert" is `expertSize` bytes where byte[i] = UInt8(expertIndex &+ i).
    static func createFixtureFile(expertCount: Int, expertSize: Int) throws -> String {
        let tempDir = FileManager.default.temporaryDirectory
        let path = tempDir.appendingPathComponent("test_experts_\(UUID().uuidString).bin").path
        let fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0o644)
        #expect(fd >= 0, "Failed to create fixture file")
        defer { close(fd) }

        var buffer = [UInt8](repeating: 0, count: expertSize)
        for expertIdx in 0..<expertCount {
            for i in 0..<expertSize {
                buffer[i] = UInt8(truncatingIfNeeded: expertIdx &+ i)
            }
            let written = Darwin.write(fd, &buffer, expertSize)
            #expect(written == expertSize)
        }
        return path
    }

    static func removeFixture(_ path: String) {
        unlink(path)
    }

    @Test("Opens a valid file and reads expert at index 0")
    func readFirstExpert() throws {
        let expertSize = 1024
        let path = try Self.createFixtureFile(expertCount: 4, expertSize: expertSize)
        defer { Self.removeFixture(path) }

        let file = try ExpertFile(path: path)
        var buffer = [UInt8](repeating: 0, count: expertSize)
        let bytesRead = try buffer.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress else { return 0 }
            return try file.readExpert(index: 0, into: base, expertSize: expertSize)
        }

        #expect(bytesRead == expertSize)
        // Expert 0: byte[i] = UInt8(0 &+ i)
        #expect(buffer[0] == 0)
        #expect(buffer[1] == 1)
        #expect(buffer[255] == 255)
    }

    @Test("Reads expert at last valid index")
    func readLastExpert() throws {
        let expertSize = 512
        let expertCount = 8
        let path = try Self.createFixtureFile(expertCount: expertCount, expertSize: expertSize)
        defer { Self.removeFixture(path) }

        let file = try ExpertFile(path: path)
        var buffer = [UInt8](repeating: 0, count: expertSize)
        let bytesRead = try buffer.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress else { return 0 }
            return try file.readExpert(index: expertCount - 1, into: base, expertSize: expertSize)
        }

        #expect(bytesRead == expertSize)
        // Expert 7: byte[i] = UInt8(7 &+ i)
        #expect(buffer[0] == 7)
        #expect(buffer[1] == 8)
    }

    @Test("Throws fileNotFound for nonexistent path")
    func fileNotFound() {
        #expect(throws: FlashMoEError.self) {
            _ = try ExpertFile(path: "/nonexistent/path/experts.bin")
        }
    }

    @Test("File descriptor is valid after init")
    func fileDescriptorValid() throws {
        let path = try Self.createFixtureFile(expertCount: 1, expertSize: 64)
        defer { Self.removeFixture(path) }

        let file = try ExpertFile(path: path)
        // fcntl with F_GETFD returns -1 if fd is invalid
        let result = fcntl(file.fileDescriptor, F_GETFD)
        #expect(result >= 0, "File descriptor should be valid")
    }

    @Test("Multiple reads from the same file are independent (pread is stateless)")
    func multipleReads() throws {
        let expertSize = 256
        let path = try Self.createFixtureFile(expertCount: 4, expertSize: expertSize)
        defer { Self.removeFixture(path) }

        let file = try ExpertFile(path: path)
        var buf0 = [UInt8](repeating: 0, count: expertSize)
        var buf2 = [UInt8](repeating: 0, count: expertSize)

        try buf2.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            try file.readExpert(index: 2, into: base, expertSize: expertSize)
        }
        try buf0.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            try file.readExpert(index: 0, into: base, expertSize: expertSize)
        }

        #expect(buf0[0] == 0)  // Expert 0
        #expect(buf2[0] == 2)  // Expert 2
    }
}
