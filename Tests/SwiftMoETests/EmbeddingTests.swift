import Testing
import Foundation
@testable import SwiftMoE

@Suite("Embedding")
struct EmbeddingTests {

    @Test("Argmax returns index of maximum value")
    func argmax() {
        let logits: [Float] = [0.1, 0.3, 0.9, 0.2, 0.5]
        let result = logits.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return 0 }
            return Embedding.argmax(logits: base, vocabSize: 5)
        }
        #expect(result == 2)
    }

    @Test("Argmax with negative values")
    func argmaxNegative() {
        let logits: [Float] = [-5.0, -1.0, -3.0, -0.5]
        let result = logits.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return 0 }
            return Embedding.argmax(logits: base, vocabSize: 4)
        }
        #expect(result == 3)
    }

    @Test("CPU dequant matvec produces correct output for known values")
    func cpuDequantMatvec() {
        let packed: UInt32 = 0x11111111
        let W: [UInt32] = [packed, packed]
        let scales: [UInt16] = [0x3F80, 0x3F80]
        let biases: [UInt16] = [0x0000, 0x0000]
        let input: [Float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        var output = [Float](repeating: 0, count: 2)

        W.withUnsafeBufferPointer { wBuf in
            guard let wBase = wBuf.baseAddress else { return }
            scales.withUnsafeBufferPointer { sBuf in
                guard let sBase = sBuf.baseAddress else { return }
                biases.withUnsafeBufferPointer { bBuf in
                    guard let bBase = bBuf.baseAddress else { return }
                    input.withUnsafeBufferPointer { iBuf in
                        guard let iBase = iBuf.baseAddress else { return }
                        output.withUnsafeMutableBufferPointer { oBuf in
                            guard let oBase = oBuf.baseAddress else { return }
                            Embedding.cpuDequantMatvec(
                                W: wBase,
                                scales: sBase,
                                biases: bBase,
                                input: iBase,
                                output: oBase,
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
