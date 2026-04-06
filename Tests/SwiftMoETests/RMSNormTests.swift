import Testing
import Foundation
@testable import SwiftMoE

@Suite("RMSNorm")
struct RMSNormTests {

    @Test("RMS norm with uniform weights produces correctly scaled output")
    func basicNorm() {
        // Input: [3.0, 4.0]
        // sum_sq = 9 + 16 = 25
        // rms = sqrt(25/2 + 1e-6) ≈ sqrt(12.500001) ≈ 3.535534
        // Without weights: [3.0/3.535534, 4.0/3.535534] ≈ [0.8485, 1.1314]
        // With weight=1.0 (BF16 0x3F80): same as without weights
        var input: [Float] = [3.0, 4.0]
        var weights: [UInt16] = [0x3F80, 0x3F80]  // [1.0, 1.0] in BF16
        var output = [Float](repeating: 0, count: 2)

        RMSNorm.apply(
            input: &input,
            weights: &weights,
            output: &output,
            dim: 2,
            eps: 1e-6
        )

        let rms = sqrtf(25.0 / 2.0 + 1e-6)
        #expect(abs(output[0] - 3.0 / rms) < 1e-4)
        #expect(abs(output[1] - 4.0 / rms) < 1e-4)
    }

    @Test("Bare RMS norm (no weights)")
    func bareNorm() {
        var input: [Float] = [3.0, 4.0]
        var output = [Float](repeating: 0, count: 2)

        RMSNorm.bare(
            input: &input,
            output: &output,
            dim: 2,
            eps: 1e-6
        )

        let rms = sqrtf(25.0 / 2.0 + 1e-6)
        #expect(abs(output[0] - 3.0 / rms) < 1e-4)
        #expect(abs(output[1] - 4.0 / rms) < 1e-4)
    }

    @Test("Gated RMS norm applies SiLU gate")
    func gatedNorm() {
        var input: [Float] = [1.0, 2.0]
        var z: [Float] = [0.0, 0.0]  // silu(0) = 0 → output should be zero
        var weights: [UInt16] = [0x3F80, 0x3F80]  // 1.0
        var output = [Float](repeating: 999, count: 2)

        RMSNorm.gated(
            input: &input,
            z: &z,
            weights: &weights,
            output: &output,
            dim: 2,
            eps: 1e-6
        )

        // silu(0) = 0 * sigmoid(0) = 0, so output should be 0
        #expect(abs(output[0]) < 1e-6)
        #expect(abs(output[1]) < 1e-6)
    }

    @Test("Zero input produces zero output")
    func zeroInput() {
        var input: [Float] = [0.0, 0.0, 0.0, 0.0]
        var weights: [UInt16] = [0x3F80, 0x3F80, 0x3F80, 0x3F80]
        var output = [Float](repeating: 999, count: 4)

        RMSNorm.apply(
            input: &input,
            weights: &weights,
            output: &output,
            dim: 4,
            eps: 1e-6
        )

        // With eps > 0, rms = sqrt(eps), output = 0 / rms = 0
        for i in 0..<4 {
            #expect(abs(output[i]) < 1e-3)
        }
    }
}
