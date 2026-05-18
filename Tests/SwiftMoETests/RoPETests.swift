import Testing
import Foundation
@testable import SwiftMoE

@Suite("RoPE")
struct RoPETests {

    @Test("Position 0 produces identity (no rotation)")
    func positionZero() {
        // At position 0, all angles are 0 → cos=1, sin=0 → no change
        var q: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        var k: [Float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        let original = q

        RoPE.apply(
            q: &q, k: &k,
            position: 0,
            numHeads: 1,
            numKVHeads: 1,
            headDim: 8,
            rotaryDim: 4,  // Only first 4 dims get rotated
            theta: 10000.0
        )

        // First rotaryDim elements should be unchanged at position 0
        for i in 0..<4 {
            #expect(abs(q[i] - original[i]) < 1e-5,
                    "Position 0 should not change element \(i)")
        }
        // Non-rotary dims should always be unchanged
        for i in 4..<8 {
            #expect(abs(q[i] - original[i]) < 1e-6,
                    "Non-rotary dim \(i) should never change")
        }
    }

    @Test("Rotation preserves vector norm")
    func preservesNorm() {
        var q: [Float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        let normBefore = sqrt(q.reduce(0) { $0 + $1 * $1 })

        // Use a dummy k buffer
        var k: [Float] = [Float](repeating: 0, count: 8)

        RoPE.apply(
            q: &q, k: &k,
            position: 7,
            numHeads: 1,
            numKVHeads: 1,
            headDim: 8,
            rotaryDim: 4,
            theta: 10000.0
        )

        let normAfter = sqrt(q.reduce(0) { $0 + $1 * $1 })
        #expect(abs(normAfter - normBefore) < 1e-4,
                "Rotation should preserve vector norm")
    }

    @Test("Different positions produce different embeddings")
    func differentPositions() {
        var q1: [Float] = [1.0, 0.0, 0.0, 0.0]
        var q2: [Float] = [1.0, 0.0, 0.0, 0.0]
        var k = [Float](repeating: 0, count: 4)

        RoPE.apply(q: &q1, k: &k, position: 1,
                   numHeads: 1, numKVHeads: 1, headDim: 4, rotaryDim: 4, theta: 10000.0)
        RoPE.apply(q: &q2, k: &k, position: 10,
                   numHeads: 1, numKVHeads: 1, headDim: 4, rotaryDim: 4, theta: 10000.0)

        // At least one element should differ
        let differs = zip(q1, q2).contains { abs($0 - $1) > 1e-6 }
        #expect(differs, "Different positions should produce different rotations")
    }
}
