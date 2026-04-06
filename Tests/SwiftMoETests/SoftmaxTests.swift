import Testing
import Foundation
@testable import SwiftMoE

@Suite("Softmax")
struct SoftmaxTests {

    @Test("Uniform input produces uniform output")
    func uniform() {
        var values: [Float] = [1.0, 1.0, 1.0, 1.0]
        Softmax.apply(&values, count: 4)

        for v in values {
            #expect(abs(v - 0.25) < 1e-6)
        }
    }

    @Test("Output sums to 1.0")
    func sumsToOne() {
        var values: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        Softmax.apply(&values, count: 5)

        let sum = values.reduce(0, +)
        #expect(abs(sum - 1.0) < 1e-5)
    }

    @Test("Large dominant value produces near-one-hot output")
    func dominant() {
        var values: [Float] = [0.0, 0.0, 100.0, 0.0]
        Softmax.apply(&values, count: 4)

        #expect(values[2] > 0.99)
        #expect(values[0] < 0.01)
    }

    @Test("Numerically stable with large values")
    func largeValues() {
        var values: [Float] = [1000.0, 1001.0, 1002.0]
        Softmax.apply(&values, count: 3)

        let sum = values.reduce(0, +)
        #expect(abs(sum - 1.0) < 1e-5, "Softmax should be stable with large inputs")
        // After subtracting max (1002), inputs become [-2, -1, 0]
        // exp(-2) ≈ 0.135, exp(-1) ≈ 0.368, exp(0) = 1.0
        #expect(values[2] > values[1])
        #expect(values[1] > values[0])
    }

    @Test("Single element produces 1.0")
    func singleElement() {
        var values: [Float] = [42.0]
        Softmax.apply(&values, count: 1)
        #expect(abs(values[0] - 1.0) < 1e-6)
    }
}
