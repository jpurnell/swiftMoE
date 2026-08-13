# Design Proposal: Phase 2 — Metal Context + Buffer Management

**Status:** PROPOSAL — awaiting approval

---

## 1. Objective

Port the `MetalCtx` struct (infer.m:900-990) and its initialization (`metal_setup`, lines 994-1224) to Swift. The original is a single C struct with ~60 `id<MTLBuffer>` fields and ~20 pipeline states, all initialized in one 230-line function. The Swift version decomposes this into logical groups while preserving the exact same buffer layout and sizes.

**Master Plan Reference:** Phase 2 — Metal Context + Buffer Management

---

## 2. Proposed Architecture

The monolithic `MetalCtx` decomposes into a coordinating `MetalContext` class that owns logically grouped buffer sets:

**New Files:**

```
Sources/FlashMoE/Metal/
├── MetalContext.swift          # Top-level coordinator: device, queue, library, pipelines
├── ShaderLibrary.swift         # Runtime shader compilation + pipeline state creation
├── ProjectionBuffers.swift     # buf_input, buf_output, batch_out slots
├── ExpertBuffers.swift         # Multi-expert double-buffered slots (uses AlignedBuffer)
├── AttentionBuffers.swift      # KV caches, scores, attention scratch
├── LinearAttentionBuffers.swift # Delta-net state + conv state per layer
├── CombineBuffers.swift        # CMD3 combine: residual, hidden, params, norms
```

**Why a class (not struct)?** `MetalContext` owns Metal objects (device, queue, buffers) that are reference types. It's created once at startup and lives for the process duration. A class with no copying is the natural fit — it doesn't need `~Copyable` because it's never in a hot path (it's the *container* that holds buffers, not the buffers flowing through the pipeline).

---

## 3. API Surface

### ShaderLibrary

```swift
/// Compiles Metal shaders from source and creates compute pipeline states.
public struct ShaderLibrary: Sendable {
    /// All pipeline states needed by the inference engine.
    public let matvecV3: MTLComputePipelineState
    public let matvecV5: MTLComputePipelineState
    public let matvecFast: MTLComputePipelineState
    public let matvec2Bit: MTLComputePipelineState
    public let rmsNormSum: MTLComputePipelineState
    public let rmsNormApply: MTLComputePipelineState
    public let rmsNormApplyBf16: MTLComputePipelineState
    public let residualAdd: MTLComputePipelineState
    public let swiglu: MTLComputePipelineState
    public let attnScores: MTLComputePipelineState
    public let attnSoftmax: MTLComputePipelineState
    public let attnValues: MTLComputePipelineState
    public let sigmoidGate: MTLComputePipelineState
    public let moeCombineResidual: MTLComputePipelineState

    // Optional pipelines (GPU linear attention — falls back to CPU if absent)
    public let deltaNetStep: MTLComputePipelineState?
    public let conv1dStep: MTLComputePipelineState?
    public let rmsNormQK: MTLComputePipelineState?
    public let computeDecayBeta: MTLComputePipelineState?
    public let gatedRmsNorm: MTLComputePipelineState?

    /// Compiles shaders from source file and creates all pipeline states.
    public init(device: MTLDevice, shaderPath: String) throws
}
```

### ExpertBuffers

```swift
/// Double-buffered expert weight slots for the CMD3 pipeline.
///
/// Each of the K=8 slots has two aligned data buffers (A for GPU compute,
/// B for background pread) plus intermediate buffers for gate/up/act/out.
public struct ExpertBuffers {
    public static let maxK = 8

    /// Set A: GPU reads from these during CMD3 expert compute.
    public let dataA: [AlignedBuffer]      // [maxK] — 2MB-aligned expert weight data
    /// Set B: CPU preads into these while GPU processes set A.
    public let dataB: [AlignedBuffer]      // [maxK] — double-buffer for prefetch

    public let gate: [MTLBuffer]           // [maxK] — gate projection output
    public let up: [MTLBuffer]             // [maxK] — up projection output
    public let activation: [MTLBuffer]     // [maxK] — SwiGLU output
    public let output: [MTLBuffer]         // [maxK] — down projection output
    public let input: MTLBuffer            // shared input vector (read-only during dispatch)

    // Shared expert (always-active)
    public let sharedGate: MTLBuffer
    public let sharedUp: MTLBuffer
    public let sharedActivation: MTLBuffer
    public let sharedOutput: MTLBuffer

    public init(device: MTLDevice, use2Bit: Bool) throws
}
```

### MetalContext

```swift
/// Top-level Metal context owning all GPU resources for inference.
///
/// Created once at startup. Provides access to the device, command queue,
/// shader pipelines, and all pre-allocated buffers.
public final class MetalContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let shaders: ShaderLibrary

    // Buffer groups
    public let projections: ProjectionBuffers
    public let experts: ExpertBuffers
    public let attention: AttentionBuffers
    public let linearAttention: LinearAttentionBuffers
    public let combine: CombineBuffers

    // Weight file as Metal buffer (set after mmap)
    public private(set) var weightBuffer: MTLBuffer?

    // Pipeline event for CPU-GPU sync
    public let pipelineEvent: MTLSharedEvent
    public private(set) var eventValue: UInt64 = 0

    public init(shaderPath: String, use2Bit: Bool) throws

    /// Wraps the mmap'd weight file as a Metal buffer (zero-copy).
    public func setWeights(_ data: UnsafeMutableRawPointer, size: Int)

    /// Resets all delta-net and conv state buffers (call at generation start).
    public func resetLinearAttentionState()
}
```

---

## 4. Constraints & Compliance

| Constraint | Compliance |
|------------|-----------|
| **Performance** | Identical buffer sizes, alignments, and storage modes to Obj-C |
| **AlignedBuffer reuse** | Expert data buffers use Phase 1's `AlignedBuffer` for 2MB alignment |
| **Memory footprint** | ≤ original (~200MB at 4-bit, ~136MB at 2-bit). Quantization-aware sizing. |
| **Sendable** | `ShaderLibrary` is Sendable (pipeline states are thread-safe). `MetalContext` is not — accessed from main inference thread only |
| **No force unwraps** | All `makeBuffer` calls checked, throw on failure |
| **Shader compilation** | Runtime from source file, matching original |

---

## 5. Test Strategy

### ShaderLibrary Tests
- **Golden path:** Compile shaders from the real `shaders.metal` file, verify all required pipelines are non-nil
- **Error handling:** Nonexistent shader path throws `.shaderCompilationFailed`
- **Optional pipelines:** Verify delta-net pipelines are present (they should be on any M-series Mac)

### ExpertBuffers Tests
- **Allocation:** All K=8 slots allocate successfully
- **Alignment:** `dataA` and `dataB` buffers are 2MB-aligned (via AlignedBuffer)
- **Size:** Expert data buffers are at least `expertSize4Bit` bytes (rounded up to 2MB boundary)
- **Shared memory:** Writing to `dataA[0].pointer` is visible via `dataA[0].metalBuffer.contents()`

### MetalContext Tests
- **Init:** Creates device, queue, compiles shaders, allocates all buffers
- **Weight wrapping:** `setWeights` creates a non-nil Metal buffer
- **State reset:** `resetLinearAttentionState` zeros delta-net state buffers

### ProjectionBuffers / AttentionBuffers / CombineBuffers Tests
- **Size verification:** Each buffer has the expected byte length
- **Storage mode:** All buffers are `.shared`

**Reference Truth:** Buffer sizes and counts from `metal_setup()` in `infer.m:994-1224`.

---

## 6. Open Questions

1. **Shader file path resolution:** The original searches `["shaders.metal", "metal_infer/shaders.metal"]`. For SPM, we could bundle the shader as a package resource or accept a path parameter. **Recommendation:** Accept a path parameter for now, matching the original's flexibility. Add SPM resource bundling later if needed.

2. **ExpertBuffers size for 2-bit:** ~~Should we always allocate for 4-bit (matching original)?~~ **RESOLVED:** Allocate based on active quantization mode. 4-bit allocates 128MB of Metal shared memory for expert data buffers vs. 64MB for 2-bit. On constrained systems (24GB MacBook), every byte of GPU-visible shared memory competes with the OS page cache for DRAM. The "Trust the OS" principle (Section 5.3) proved that freeing Metal shared memory directly improves throughput by giving more room to the page cache. `ExpertBuffers.init` takes a `use2Bit: Bool` parameter and sizes accordingly.

---

**Created:** 2026-04-05
