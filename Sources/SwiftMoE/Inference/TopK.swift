/// CPU top-K selection and weight normalization for expert routing.
///
/// After softmax over the 512 expert gate scores, selects the K highest-scoring
/// experts and normalizes their weights to sum to 1. Qwen3.5 defaults to K=10
/// but K=4 gives "excellent" quality with 2.6x less I/O.
public enum TopK {

    /// Selects the top-K elements by value from a score array.
    ///
    /// Returns sorted indices (highest score first) and their normalized weights.
    /// The softmax should already have been applied to the scores.
    ///
    /// - Parameters:
    ///   - scores: Array of scores (e.g., 512 expert gate logits after softmax).
    ///   - k: Number of top elements to select.
    /// - Returns: Tuple of (indices, normalizedWeights), both length K.
    public static func select(scores: inout [Float], k: Int) -> (indices: [Int], weights: [Float]) {
        let n = scores.count
        let actualK = min(k, n)

        // Partial sort: find top-K by repeatedly finding the max
        var indices = [Int](repeating: 0, count: actualK)
        var weights = [Float](repeating: 0, count: actualK)
        var used = [Bool](repeating: false, count: n)

        for i in 0..<actualK {
            var bestIdx = -1
            var bestVal: Float = -.infinity
            for j in 0..<n {
                if !used[j] && scores[j] > bestVal {
                    bestVal = scores[j]
                    bestIdx = j
                }
            }
            indices[i] = bestIdx
            weights[i] = bestVal
            if bestIdx >= 0 { used[bestIdx] = true }
        }

        // Normalize weights to sum to 1
        let sum = weights.reduce(0, +)
        if sum > 0 {
            for i in 0..<actualK {
                weights[i] /= sum
            }
        }

        return (indices, weights)
    }
}
