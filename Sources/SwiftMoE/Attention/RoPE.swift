import Foundation

/// Rotary Position Embeddings (RoPE) for transformer attention.
///
/// RoPE encodes position information by rotating pairs of dimensions in Q and K
/// vectors. This allows the dot product `q · k` to naturally encode relative
/// position through the rotation angle difference.
///
/// Qwen3.5 uses partial rotary (25% of head dimensions) with theta=10,000,000.
///
/// ## Rotation Pairing
/// The pairing is non-traditional: `(x[i], x[i + half])` where `half = rotaryDim / 2`.
/// This matches the original implementation in `infer.m:2025-2059`.
public enum RoPE {

    /// Applies rotary position embeddings to Q and K tensors in-place.
    ///
    /// Only the first `rotaryDim` dimensions of each head are rotated.
    /// Remaining dimensions are left unchanged.
    ///
    /// - Parameters:
    ///   - q: Query tensor [numHeads * headDim], modified in-place.
    ///   - k: Key tensor [numKVHeads * headDim], modified in-place.
    ///   - position: Token position in the sequence (for computing rotation angles).
    ///   - numHeads: Number of query attention heads.
    ///   - numKVHeads: Number of key-value heads (GQA).
    ///   - headDim: Dimension per head.
    ///   - rotaryDim: Number of dimensions to rotate (typically headDim * partialRotary).
    ///   - theta: RoPE theta base (10,000,000 for Qwen3.5).
    public static func apply(
        q: UnsafeMutablePointer<Float>,
        k: UnsafeMutablePointer<Float>,
        position: Int,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        rotaryDim: Int,
        theta: Float
    ) {
        guard rotaryDim > 0 else { return }
        let halfDim = rotaryDim / 2
        let posFloat = Float(position)
        let rotaryDimFloat = Float(rotaryDim)
        guard rotaryDimFloat > 0, theta > 0 else { return }

        let invRotaryDim = 1.0 / rotaryDimFloat

        // Apply to query heads
        for h in 0..<numHeads {
            let base = h * headDim
            for i in 0..<halfDim {
                let exponent = Float(2 * i) * invRotaryDim
                let base10 = powf(theta, exponent)
                let freq = base10 > 0 ? 1.0 / base10 : 0.0
                let angle = posFloat * freq
                let cosA = cosf(angle)
                let sinA = sinf(angle)

                let x0 = q[base + i]
                let x1 = q[base + i + halfDim]
                q[base + i]           = x0 * cosA - x1 * sinA
                q[base + i + halfDim] = x0 * sinA + x1 * cosA
            }
        }

        // Apply to key heads
        for h in 0..<numKVHeads {
            let base = h * headDim
            for i in 0..<halfDim {
                let exponent = Float(2 * i) * invRotaryDim
                let base10 = powf(theta, exponent)
                let freq = base10 > 0 ? 1.0 / base10 : 0.0
                let angle = posFloat * freq
                let cosA = cosf(angle)
                let sinA = sinf(angle)

                let x0 = k[base + i]
                let x1 = k[base + i + halfDim]
                k[base + i]           = x0 * cosA - x1 * sinA
                k[base + i + halfDim] = x0 * sinA + x1 * cosA
            }
        }
    }
}
