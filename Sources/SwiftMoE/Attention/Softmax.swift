import Foundation

/// Numerically stable softmax implementation.
///
/// Uses the max-subtraction trick to prevent overflow:
/// `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`
///
/// This is mathematically equivalent to the standard formulation but
/// avoids `exp(large_number)` → infinity.
public enum Softmax {

    /// Applies softmax in-place over the first `count` elements.
    ///
    /// Matches `cpu_softmax` in `infer.m`.
    public static func apply(_ values: UnsafeMutablePointer<Float>, count: Int) {
        guard count > 0 else { return }

        // Find max for numerical stability
        var maxVal = values[0]
        for i in 1..<count {
            if values[i] > maxVal { maxVal = values[i] }
        }

        // exp(x - max) and accumulate sum
        var sum: Float = 0
        for i in 0..<count {
            let e = expf(values[i] - maxVal)
            values[i] = e
            sum += e
        }

        // Normalize
        guard sum > 0 else { return }
        let invSum = 1.0 / sum
        for i in 0..<count {
            values[i] *= invSum
        }
    }
}
