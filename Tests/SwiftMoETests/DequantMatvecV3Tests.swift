import Testing
import Foundation
import Metal
@testable import SwiftMoE

/// Numeric tests for `dequant_matvec_4bit_v3` against the CPU reference.
///
/// The kernel assigns one SIMD group per output row, eight rows to a
/// 256-thread threadgroup, and caches the input vector in threadgroup memory
/// via a cooperative load. An `outDim` that is not a multiple of eight leaves
/// the final threadgroup partially populated, which is where a load striding by
/// a hardcoded width — rather than the real threadgroup size — silently fills
/// `x_shared` incompletely and corrupts the rows that *are* valid.
@Suite("DequantMatvecV3")
struct DequantMatvecV3Tests {

    /// A deterministic 4-bit quantized matrix plus its input vector.
    ///
    /// `blob` is laid out as the kernel expects — packed weights, then bf16
    /// scales, then bf16 biases — with the byte offset of each later section.
    struct Fixture {
        let blob: [UInt8]
        let scalesOffset: Int
        let biasesOffset: Int
        let input: [Float]
        let outDim: Int
        let inDim: Int
        let groupSize: Int
    }

    static func makeFixture(outDim: Int, inDim: Int, groupSize: Int) -> Fixture {
        let packedCols = inDim / 8
        let numGroups = inDim / groupSize

        // Little-endian, assembled byte-by-byte to match the on-disk weight layout.
        var blob = [UInt8]()
        blob.reserveCapacity(outDim * packedCols * 4 + outDim * numGroups * 4)

        func appendU32(_ value: UInt32) {
            blob.append(UInt8(truncatingIfNeeded: value))
            blob.append(UInt8(truncatingIfNeeded: value >> 8))
            blob.append(UInt8(truncatingIfNeeded: value >> 16))
            blob.append(UInt8(truncatingIfNeeded: value >> 24))
        }
        func appendU16(_ value: UInt16) {
            blob.append(UInt8(truncatingIfNeeded: value))
            blob.append(UInt8(truncatingIfNeeded: value >> 8))
        }

        for row in 0..<outDim {
            for col in 0..<packedCols {
                // Deterministic nibbles spread across the full 0...15 range.
                var word: UInt32 = 0
                for nibble in 0..<8 {
                    let value = UInt32((row &* 7 &+ col &* 3 &+ nibble &* 5) % 16)
                    word |= value << UInt32(nibble * 4)
                }
                appendU32(word)
            }
        }

        let scalesOffset = blob.count
        for i in 0..<(outDim * numGroups) {
            appendU16(floatToBf16(0.05 + Float(i % 5) * 0.01))
        }
        let biasesOffset = blob.count
        for i in 0..<(outDim * numGroups) {
            appendU16(floatToBf16(-0.4 + Float(i % 3) * 0.1))
        }

        var input = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim {
            input[i] = Float(i % 17) * 0.03 - 0.25
        }

        return Fixture(blob: blob, scalesOffset: scalesOffset, biasesOffset: biasesOffset,
                       input: input, outDim: outDim, inDim: inDim, groupSize: groupSize)
    }

    /// Runs the fixture through the GPU kernel and the CPU reference.
    static func run(_ fixture: Fixture, context: MetalContext) throws -> (gpu: [Float], cpu: [Float]) {
        // Page-aligned so it can back an MTLBuffer with no copy, mirroring the
        // mmap'd weight file the engine uses in production.
        let pageSize = 16384
        let size = (fixture.blob.count + pageSize - 1) / pageSize * pageSize
        var raw: UnsafeMutableRawPointer?
        posix_memalign(&raw, pageSize, size)
        let base = try #require(raw, "posix_memalign should provide aligned storage")
        defer { free(base) }

        let blobBytes = base.initializeMemory(as: UInt8.self, repeating: 0, count: size)
        for (index, byte) in fixture.blob.enumerated() {
            blobBytes[index] = byte
        }

        context.setWeights(base, size: size)

        let inputBuffer = context.projections.input
        let inputGPU = inputBuffer.contents().assumingMemoryBound(to: Float.self)
        for (index, value) in fixture.input.enumerated() {
            inputGPU[index] = value
        }
        let outputBuffer = context.projections.batchSlots[0]
        let outputGPU = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<fixture.outDim {
            outputGPU[index] = 0
        }

        let cmd = try #require(context.queue.makeCommandBuffer())
        ExpertEncoder.encodeMatvec(
            context: context, commandBuffer: cmd,
            weights: UnsafeRawPointer(base),
            scales: UnsafeRawPointer(base.advanced(by: fixture.scalesOffset)),
            biases: UnsafeRawPointer(base.advanced(by: fixture.biasesOffset)),
            inputBuffer: inputBuffer, outputBuffer: outputBuffer,
            outDim: UInt32(fixture.outDim), inDim: UInt32(fixture.inDim),
            groupSize: UInt32(fixture.groupSize))
        cmd.commit()
        cmd.waitUntilCompletedChecked("test matvec v3")

        var gpu = [Float](repeating: 0, count: fixture.outDim)
        for index in 0..<fixture.outDim {
            gpu[index] = outputGPU[index]
        }

        // CPU reference runs against its own allocations so no Swift array has
        // to be borrowed as a pointer across the call.
        let inputCPU = UnsafeMutablePointer<Float>.allocate(capacity: fixture.inDim)
        defer { inputCPU.deallocate() }
        for (index, value) in fixture.input.enumerated() {
            inputCPU[index] = value
        }
        let outputCPU = UnsafeMutablePointer<Float>.allocate(capacity: fixture.outDim)
        defer { outputCPU.deallocate() }
        outputCPU.initialize(repeating: 0, count: fixture.outDim)

        Embedding.cpuDequantMatvec(
            W: base.assumingMemoryBound(to: UInt32.self),
            scales: base.advanced(by: fixture.scalesOffset).assumingMemoryBound(to: UInt16.self),
            biases: base.advanced(by: fixture.biasesOffset).assumingMemoryBound(to: UInt16.self),
            input: inputCPU, output: outputCPU,
            outDim: fixture.outDim, inDim: fixture.inDim, groupSize: fixture.groupSize)

        var cpu = [Float](repeating: 0, count: fixture.outDim)
        for index in 0..<fixture.outDim {
            cpu[index] = outputCPU[index]
        }

        return (gpu, cpu)
    }

    /// Tolerance for one row: the GPU reduces in a different order than the CPU,
    /// so exact equality is not expected, but the gap should stay at rounding scale.
    static func tolerance(for reference: Float) -> Float {
        max(1e-3, abs(reference) * 1e-3)
    }

    /// `outDim` values that are and are not multiples of the 8 rows per
    /// threadgroup. The non-multiples leave a partial final threadgroup.
    @Test("Matches CPU reference across row counts", arguments: [8, 16, 13, 5, 1, 23])
    func matchesCPUReference(outDim: Int) throws {
        guard let shaderURL = ShaderLibraryTests.shaderURL else { return }
        let context = try MetalContext(config: .qwen397B, shaderPath: shaderURL.path, use2Bit: false)

        let fixture = Self.makeFixture(outDim: outDim, inDim: 512, groupSize: 64)
        let (gpu, cpu) = try Self.run(fixture, context: context)

        for row in 0..<outDim {
            let delta = abs(gpu[row] - cpu[row])
            #expect(delta <= Self.tolerance(for: cpu[row]),
                    "row \(row) of \(outDim): GPU \(gpu[row]) vs CPU \(cpu[row]) (delta \(delta))")
        }
    }

    /// A row count that is not a multiple of 8 must still compute its tail rows.
    ///
    /// Skipping the partial threadgroup leaves them at 0, and striding the
    /// cooperative load by a hardcoded 256 leaves them NaN or wildly out of
    /// range; both are caught by requiring the tail to match the reference.
    @Test("Partial final threadgroup computes its rows")
    func partialThreadgroupComputesTailRows() throws {
        guard let shaderURL = ShaderLibraryTests.shaderURL else { return }
        let context = try MetalContext(config: .qwen397B, shaderPath: shaderURL.path, use2Bit: false)

        let outDim = 13  // one full threadgroup of 8 rows, then a partial group of 5
        let fixture = Self.makeFixture(outDim: outDim, inDim: 512, groupSize: 64)
        let (gpu, cpu) = try Self.run(fixture, context: context)

        for row in 8..<outDim {
            // The fixture is built so every reference row is well away from zero;
            // asserting that keeps the comparison below from being satisfied by
            // a kernel that simply wrote nothing.
            #expect(abs(cpu[row]) > 0.1,
                    "fixture row \(row) should be distinguishable from an unwritten 0")
            #expect(gpu[row].isFinite, "tail row \(row) should be finite, got \(gpu[row])")
            #expect(abs(gpu[row] - cpu[row]) <= Self.tolerance(for: cpu[row]),
                    "tail row \(row): GPU \(gpu[row]) vs CPU \(cpu[row])")
        }
    }
}
