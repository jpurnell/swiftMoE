import Testing
@testable import SwiftMoE

@Suite("TopK")
struct TopKTests {

    @Test("Selects correct top-K indices from scores")
    func basicTopK() {
        var scores: [Float] = [0.1, 0.5, 0.3, 0.8, 0.2]
        let (indices, weights) = TopK.select(scores: &scores, k: 3)

        #expect(indices.count == 3)
        #expect(weights.count == 3)
        // Top 3 by score: index 3 (0.8), index 1 (0.5), index 2 (0.3)
        #expect(indices[0] == 3)
        #expect(indices[1] == 1)
        #expect(indices[2] == 2)
    }

    @Test("Weights are normalized to sum to 1")
    func normalizedWeights() {
        var scores: [Float] = [0.1, 0.5, 0.3, 0.8, 0.2]
        let (_, weights) = TopK.select(scores: &scores, k: 3)

        let sum = weights.reduce(0, +)
        #expect(abs(sum - 1.0) < 1e-5, "Weights should sum to 1.0, got \(sum)")
    }

    @Test("K=1 selects the maximum")
    func singleSelection() {
        var scores: [Float] = [0.1, 0.9, 0.3]
        let (indices, weights) = TopK.select(scores: &scores, k: 1)

        #expect(indices[0] == 1)
        #expect(abs(weights[0] - 1.0) < 1e-6)
    }

    @Test("K equal to count returns all elements sorted")
    func fullSelection() {
        var scores: [Float] = [0.3, 0.1, 0.2]
        let (indices, _) = TopK.select(scores: &scores, k: 3)

        #expect(indices[0] == 0)  // 0.3
        #expect(indices[1] == 2)  // 0.2
        #expect(indices[2] == 1)  // 0.1
    }
}
