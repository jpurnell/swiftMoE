import Testing
import Foundation
@testable import SwiftMoE

@Suite("Embedding")
struct EmbeddingTests {

    @Test("Argmax returns index of maximum value")
    func argmax() {
        var logits: [Float] = [0.1, 0.3, 0.9, 0.2, 0.5]
        let result = logits.withUnsafeBufferPointer { buf in
            Embedding.argmax(logits: buf.baseAddress!, vocabSize: 5)
        }
        #expect(result == 2)
    }

    @Test("Argmax with negative values")
    func argmaxNegative() {
        var logits: [Float] = [-5.0, -1.0, -3.0, -0.5]
        let result = logits.withUnsafeBufferPointer { buf in
            Embedding.argmax(logits: buf.baseAddress!, vocabSize: 4)
        }
        #expect(result == 3)  // -0.5 is the largest
    }

    @Test("CPU dequant matvec produces correct output for known values")
    func cpuDequantMatvec() {
        // 2x8 matrix (outDim=2, inDim=8, groupSize=8, 1 group)
        // Each uint32 holds 8 nibbles. We'll use nibble=1 for all → weight = 1*scale + bias
        let packed: UInt32 = 0x11111111  // all nibbles = 1
        var W: [UInt32] = [packed, packed]  // 2 rows, 1 packed uint32 each (8 values per row)
        // scale=1.0 (BF16 0x3F80), bias=0.0 (BF16 0x0000) → dequantized weight = 1.0
        var scales: [UInt16] = [0x3F80, 0x3F80]  // 1 group per row
        var biases: [UInt16] = [0x0000, 0x0000]
        var input: [Float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        var output = [Float](repeating: 0, count: 2)

        W.withUnsafeBufferPointer { wBuf in
            scales.withUnsafeBufferPointer { sBuf in
                biases.withUnsafeBufferPointer { bBuf in
                    input.withUnsafeBufferPointer { iBuf in
                        output.withUnsafeMutableBufferPointer { oBuf in
                            Embedding.cpuDequantMatvec(
                                W: wBuf.baseAddress!,
                                scales: sBuf.baseAddress!,
                                biases: bBuf.baseAddress!,
                                input: iBuf.baseAddress!,
                                output: oBuf.baseAddress!,
                                outDim: 2, inDim: 8, groupSize: 8
                            )
                        }
                    }
                }
            }
        }

        // Each row: 8 weights of 1.0 × 8 inputs of 1.0 = 8.0
        #expect(abs(output[0] - 8.0) < 1e-4, "Row 0 should be 8.0, got \(output[0])")
        #expect(abs(output[1] - 8.0) < 1e-4, "Row 1 should be 8.0, got \(output[1])")
    }
}
