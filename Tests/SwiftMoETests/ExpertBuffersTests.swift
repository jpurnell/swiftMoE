import Testing
import Foundation
import Metal
@testable import SwiftMoE

/// Tests for ExpertBuffers — double-buffered expert weight slots.
@Suite("ExpertBuffers")
struct ExpertBuffersTests {

    @Test("Allocates all K=8 expert data slots with 2MB alignment (4-bit)")
    func allocation4Bit() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        let buffers = try ExpertBuffers(device: device, config: .qwen397B, use2Bit: false)

        #expect(buffers.dataA.count == ExpertBuffers.maxK)
        #expect(buffers.dataB.count == ExpertBuffers.maxK)

        // Verify 2MB alignment on all data buffers
        for k in 0..<ExpertBuffers.maxK {
            let addrA = UInt(bitPattern: buffers.dataA[k].pointer)
            let addrB = UInt(bitPattern: buffers.dataB[k].pointer)
            #expect(addrA % UInt(ModelConfig.dmaAlignment) == 0,
                    "dataA[\(k)] must be 2MB-aligned")
            #expect(addrB % UInt(ModelConfig.dmaAlignment) == 0,
                    "dataB[\(k)] must be 2MB-aligned")
        }
    }

    @Test("4-bit expert data buffers are large enough for EXPERT_SIZE")
    func size4Bit() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        let buffers = try ExpertBuffers(device: device, config: .qwen397B, use2Bit: false)
        for k in 0..<ExpertBuffers.maxK {
            #expect(buffers.dataA[k].size >= ModelConfig.qwen397B.expertSize4Bit,
                    "dataA[\(k)] must hold a 4-bit expert")
        }
    }

    @Test("2-bit expert data buffers are sized for EXPERT_SIZE_2BIT, not 4-bit")
    func size2Bit() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        let buffers2 = try ExpertBuffers(device: device, config: .qwen397B, use2Bit: true)
        let buffers4 = try ExpertBuffers(device: device, config: .qwen397B, use2Bit: false)

        // 2-bit should be smaller than 4-bit
        #expect(buffers2.dataA[0].size < buffers4.dataA[0].size,
                "2-bit allocation should be smaller than 4-bit")

        // But still large enough for 2-bit experts
        for k in 0..<ExpertBuffers.maxK {
            #expect(buffers2.dataA[k].size >= ModelConfig.qwen397B.expertSize2Bit,
                    "dataA[\(k)] must hold a 2-bit expert")
        }
    }

    @Test("Intermediate buffers have correct sizes")
    func intermediateBufferSizes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        let buffers = try ExpertBuffers(device: device, config: .qwen397B, use2Bit: false)

        for k in 0..<ExpertBuffers.maxK {
            #expect(buffers.gate[k].length == ModelConfig.qwen397B.moeIntermediate * MemoryLayout<Float>.size)
            #expect(buffers.up[k].length == ModelConfig.qwen397B.moeIntermediate * MemoryLayout<Float>.size)
            #expect(buffers.activation[k].length == ModelConfig.qwen397B.moeIntermediate * MemoryLayout<Float>.size)
            #expect(buffers.output[k].length == ModelConfig.qwen397B.hiddenDim * MemoryLayout<Float>.size)
        }
    }

    @Test("Shared expert buffers are allocated")
    func sharedExpertBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }

        let buffers = try ExpertBuffers(device: device, config: .qwen397B, use2Bit: false)

        #expect(buffers.sharedGate.length == ModelConfig.qwen397B.sharedIntermediate * MemoryLayout<Float>.size)
        #expect(buffers.sharedUp.length == ModelConfig.qwen397B.sharedIntermediate * MemoryLayout<Float>.size)
        #expect(buffers.sharedActivation.length == ModelConfig.qwen397B.sharedIntermediate * MemoryLayout<Float>.size)
        #expect(buffers.sharedOutput.length == ModelConfig.qwen397B.hiddenDim * MemoryLayout<Float>.size)
    }
}
