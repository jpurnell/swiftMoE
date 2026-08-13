# Design Proposal: Phase 3 — Attention Computation

**Status:** PROPOSAL — awaiting approval

---

## 1. Objective

Port the attention computation layer — the only phase that actually *computes* on the GPU and CPU rather than just allocating buffers. This includes RoPE, softmax, full scaled-dot-product attention (15 layers), GatedDeltaNet linear attention (45 layers), GPU batched matrix-vector dispatch, and RMS normalization.

**Master Plan Reference:** Phase 3 — Attention Computation

---

## 2. Proposed Architecture

**New Files:**

```
Sources/FlashMoE/
├── Metal/
│   └── BatchMatvec.swift          # BatchMatvecSpec + GPU encode/flush helpers
├── Attention/
│   ├── RMSNorm.swift              # cpu_rms_norm + bare + gated variants
│   ├── RoPE.swift                 # Rotary position embeddings
│   ├── Softmax.swift              # CPU softmax
│   ├── FullAttention.swift        # Full scaled-dot-product attention (15 layers)
│   ├── LinearAttention.swift      # GatedDeltaNet + conv1d step (45 layers)
│   └── BFloat16.swift             # bf16 ↔ f32 conversion helpers
```

**Scope:** CPU-side math + GPU encode helpers. The actual Metal shaders are unchanged.

---

## 3. API Surface

### BFloat16 Helpers

```swift
/// Converts a BFloat16 value (stored as UInt16) to Float32.
@inline(__always)
public func bf16ToFloat(_ bf16: UInt16) -> Float

/// Converts a Float32 value to BFloat16 (truncation, no rounding).
@inline(__always)
public func floatToBf16(_ f: Float) -> UInt16
```

### RMSNorm

```swift
public enum RMSNorm {
    /// RMS normalization with BF16 weights: out = (x / rms) * weight
    static func apply(
        input: UnsafePointer<Float>,
        weights: UnsafePointer<UInt16>,  // BF16
        output: UnsafeMutablePointer<Float>,
        dim: Int,
        eps: Float
    )

    /// Bare RMS normalization (no weights): out = x / rms
    static func bare(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        dim: Int,
        eps: Float
    )

    /// Gated RMS normalization: out = rms_norm(x) * silu(z) * weight
    static func gated(
        input: UnsafePointer<Float>,
        z: UnsafePointer<Float>,
        weights: UnsafePointer<UInt16>,  // BF16
        output: UnsafeMutablePointer<Float>,
        dim: Int,
        eps: Float
    )
}
```

### BatchMatvec (GPU Encode)

```swift
/// Specification for a single GPU dequantized matrix-vector multiply.
public struct BatchMatvecSpec {
    public let weights: UnsafeRawPointer      // into mmap'd weight file
    public let scales: UnsafeRawPointer       // BF16
    public let biases: UnsafeRawPointer       // BF16
    public var outputCPU: UnsafeMutablePointer<Float>  // CPU destination
    public let outDim: UInt32
    public let inDim: UInt32
    public let groupSize: UInt32
    public let batchSlot: Int                 // which batch_out[] slot
}

public enum BatchMatvec {
    /// Encodes multiple dequant matvec dispatches into a command buffer.
    /// Does NOT commit — caller batches and commits once.
    static func encode(
        context: MetalContext,
        commandBuffer: MTLCommandBuffer,
        specs: [BatchMatvecSpec]
    )

    /// Copies GPU results from batch_out slots back to CPU arrays.
    static func flushResults(
        context: MetalContext,
        specs: [BatchMatvecSpec]
    )
}
```

### RoPE

```swift
public enum RoPE {
    /// Applies rotary position embeddings to Q and K tensors in-place.
    static func apply(
        q: UnsafeMutablePointer<Float>,  // [numHeads * headDim]
        k: UnsafeMutablePointer<Float>,  // [numKVHeads * headDim]
        position: Int,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        rotaryDim: Int,
        theta: Float
    )
}
```

### Softmax

```swift
public enum Softmax {
    /// In-place softmax over a float array.
    static func apply(_ values: UnsafeMutablePointer<Float>, count: Int)
}
```

---

## 4. Constraints & Compliance

| Constraint | Compliance |
|------------|-----------|
| **Numerical equivalence** | Output must match Obj-C bit-for-bit (same float ops) |
| **BLAS acceleration** | GatedDeltaNet uses Accelerate: cblas_sscal, cblas_sgemv, cblas_sger |
| **No allocations in hot path** | All scratch buffers pre-allocated (use Phase 2 buffers) |
| **@inline(__always)** | BFloat16 conversion must be inlined (called millions of times) |

---

## 5. Test Strategy

### RMSNorm Tests
- **Golden path:** Known 4-element vector → hand-computed expected output
- **Bare variant:** Same without weight multiplication
- **Gated variant:** Include SiLU gating

### BFloat16 Tests
- **Round-trip:** Float → BF16 → Float preserves value (within truncation)
- **Known values:** 1.0 = 0x3F80, -1.0 = 0xBF80

### RoPE Tests
- **Position 0:** No rotation (cos=1, sin=0 for all frequencies)
- **Known rotation:** Small vector at position 1, verify against hand computation

### Softmax Tests
- **Uniform input:** All equal → all output 1/n
- **Single max:** One large value → output ≈ [0, ..., 1, ..., 0]
- **Numerical stability:** Very large values don't overflow

### BatchMatvec Tests
- **Encode + flush:** Dispatch a known matvec on GPU, verify result matches CPU reference

---

## 6. Open Questions

1. **BLAS for delta-net:** The original uses `cblas_sscal`, `cblas_sgemv`, `cblas_sger` for the 64-head × 128×128 state update (64% faster than scalar). Should we use Accelerate directly or wrap it? **Recommendation:** Call Accelerate directly — it's a system framework, zero overhead, and matching the original exactly.

---

**Created:** 2026-04-05
