import Foundation

/// RMS normalization variants used throughout the transformer.
///
/// RMS norm is simpler than LayerNorm — it normalizes by the root mean square
/// without centering. This is used for all normalization in Qwen3.5:
/// - Input layer norm (before attention)
/// - Post-attention norm (before MoE)
/// - Per-head Q/K normalization
/// - Gated output normalization (linear attention z-gating)
public enum RMSNorm {

    /// Standard RMS normalization with BF16 learnable weights.
    ///
    /// Computes: `out[i] = (x[i] / rms) * weight[i]`
    /// where `rms = sqrt(mean(x²) + eps)`
    ///
    /// Matches `cpu_rms_norm` in `infer.m:738-749`.
    public static func apply(
        input: UnsafePointer<Float>,
        weights: UnsafePointer<UInt16>,
        output: UnsafeMutablePointer<Float>,
        dim: Int,
        eps: Float = 1e-6
    ) {
        guard dim > 0 else { return }
        var sumSq: Float = 0
        for i in 0..<dim {
            sumSq += input[i] * input[i]
        }
        let rms = sqrtf(sumSq / Float(dim) + eps)
        let invRms = rms > 0 ? 1.0 / rms : 0.0

        for i in 0..<dim {
            output[i] = input[i] * invRms * bf16ToFloat(weights[i])
        }
    }

    /// Bare RMS normalization (no weights).
    ///
    /// Computes: `out[i] = x[i] / rms`
    ///
    /// Used for per-head Q/K normalization in both full and linear attention.
    /// Matches `cpu_rms_norm_bare` in `infer.m:2354-2359`.
    public static func bare(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        dim: Int,
        eps: Float = 1e-6
    ) {
        guard dim > 0 else { return }
        var sumSq: Float = 0
        for i in 0..<dim {
            sumSq += input[i] * input[i]
        }
        let rms = sqrtf(sumSq / Float(dim) + eps)
        let invRms = rms > 0 ? 1.0 / rms : 0.0

        for i in 0..<dim {
            output[i] = input[i] * invRms
        }
    }

    /// Gated RMS normalization: `out = rms_norm(x) * silu(z) * weight`
    ///
    /// Used in the GatedDeltaNet linear attention output, where `z` is the
    /// gating projection. SiLU(z) = z * sigmoid(z).
    ///
    /// Matches `cpu_rms_norm_gated` in `infer.m:2362-2372`.
    public static func gated(
        input: UnsafePointer<Float>,
        z: UnsafePointer<Float>,
        weights: UnsafePointer<UInt16>,
        output: UnsafeMutablePointer<Float>,
        dim: Int,
        eps: Float = 1e-6
    ) {
        guard dim > 0 else { return }
        var sumSq: Float = 0
        for i in 0..<dim {
            sumSq += input[i] * input[i]
        }
        let rms = sqrtf(sumSq / Float(dim) + eps)
        let invRms = rms > 0 ? 1.0 / rms : 0.0

        for i in 0..<dim {
            let normalized = input[i] * invRms
            let zi = z[i]
            let silu = zi / (1.0 + expf(-zi))  // SiLU = z * sigmoid(z)
            output[i] = normalized * silu * bf16ToFloat(weights[i])
        }
    }
}
