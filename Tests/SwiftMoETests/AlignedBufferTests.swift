import Testing
import Foundation
import Metal
@testable import SwiftMoE

/// Tests for AlignedBuffer — the ~Copyable 2MB-aligned memory + MTLBuffer wrapper.
@Suite("AlignedBuffer")
struct AlignedBufferTests {

    @Test("Allocates buffer with 2MB alignment")
    func alignment() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        let buffer = try AlignedBuffer(device: device, size: 4096)
        let address = UInt(bitPattern: buffer.pointer)
        let alignment = ModelConfig.dmaAlignment  // 2MB

        #expect(address % UInt(alignment) == 0,
                "Buffer pointer must be 2MB-aligned for DMA. Got address \(address)")
    }

    @Test("Metal buffer shares the same memory as the raw pointer")
    func sharedMemory() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        let size = 4096
        let buffer = try AlignedBuffer(device: device, size: size)

        // Write via raw pointer
        let writePtr = buffer.pointer.assumingMemoryBound(to: UInt8.self)
        for i in 0..<size {
            writePtr[i] = UInt8(truncatingIfNeeded: i)
        }

        // Read via Metal buffer contents — should see the same data
        let metalContents = buffer.metalBuffer.contents().assumingMemoryBound(to: UInt8.self)
        #expect(metalContents[0] == 0)
        #expect(metalContents[1] == 1)
        #expect(metalContents[255] == 255)
    }

    @Test("Reports correct size")
    func sizeProperty() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        let size = 8192
        let buffer = try AlignedBuffer(device: device, size: size)
        #expect(buffer.size == size)
    }

    @Test("Metal buffer has shared storage mode")
    func storageMode() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        let buffer = try AlignedBuffer(device: device, size: 4096)
        #expect(buffer.metalBuffer.storageMode == .shared)
    }
}
