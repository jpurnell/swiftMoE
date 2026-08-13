# Session Summary: Flash-MoE Swift Modernization — Phases 1-5

| Date | Phase | Status |
| :--- | :--- | :--- |
| 2026-04-05 | Phases 1-4: Foundation through Pipeline | COMPLETED |

## 1. Core Objective

Incrementally port the Flash-MoE inference engine (7,151-line Obj-C/C monolith) to Swift, preserving the proven performance characteristics while gaining type safety, ownership semantics, structured concurrency, and testability.

## 2. Design Decisions

- **AlignedBuffer**: Changed from `~Copyable struct` to `final class` — `Array<~Copyable>` not supported in Swift. ARC overhead negligible (startup-only allocation).
- **Expert buffer sizing**: Quantization-aware (4MB for 2-bit vs 8MB for 4-bit per slot), saving 64MB of GPU-visible shared memory in 2-bit mode.
- **IOPool concurrency**: Uses `TaskGroup` with `nonisolated(unsafe)` for per-task buffer pointers that the compiler can't prove are disjoint.
- **Shader compilation**: Runtime from source file via path parameter.

## 3. Work Completed

### Phase 1: I/O Subsystem (6 source files, 17 tests)
- SPM package with CTokenizer/CLineNoise C interop targets
- `ExpertFile` (~Copyable fd wrapper), `AlignedBuffer`, `IOPool`, `WeightManifest`
- `ModelConfig` constants, `FlashMoEError` error types

### Phase 2: Metal Context (7 source files, 12 tests)
- `ShaderLibrary`: 14 required + 5 optional pipeline states from real `shaders.metal`
- `MetalContext`: coordinator owning 5 buffer groups
- Buffer decomposition: `ProjectionBuffers`, `ExpertBuffers`, `AttentionBuffers`, `LinearAttentionBuffers`, `CombineBuffers`

### Phase 3: Attention Computation (5 source files, 16 tests)
- `BFloat16`: `@inline(__always)` conversion matching C bit patterns
- `RMSNorm`: standard, bare, gated variants
- `Softmax`: numerically stable with max-subtraction
- `RoPE`: rotary position embeddings with partial rotation
- `BatchMatvec`: GPU encode/flush helpers

### Phase 4: Pipeline Orchestration (6 source files, 9 tests)
- `LayerPipeline`: CMD1→CMD2→CMD3 orchestration
- `DeferredExpertState`: async GPU expert state machine
- `KVCache`, `LinearAttentionState`: attention state management
- `TopK`: expert routing selection + weight normalization
- `TokenGenerator`: top-level inference loop structure
- `LayerTiming`: per-phase instrumentation

## 4. Quality Gate

| Check | Status |
| :--- | :--- |
| **build** | 0 warnings |
| **test** | 54/54 passing (0.390s) |
| **safety** | No force unwraps, force casts, or try! |

### Phase 5: Token Generation Loop (3 source files, 7 tests)
- `WeightFile`: mmap'd weight access with generic `tensorPointer<T>` lookup
- `LayerWeightCache`: Pre-computed weight pointers for all 60 layers (eliminates 36K snprintf+lookup)
- `Embedding`: embed_lookup (row dequant) + lm_head (full matvec) + argmax
- `TokenGenerator.generate()`: Full inference loop structure (embed → 60 layers → norm → lm_head → argmax)

## 5. Project State

- 27 Swift source files in `Sources/FlashMoE/` (2,394 LOC)
- 17 test files in `Tests/FlashMoETests/` (1,153 LOC)
- 61 tests across 17 suites — all passing (0.524s)
- Original Obj-C code preserved in `metal_infer/` as reference
- Zero compiler warnings

## 6. Next Session Handover

### Immediate Starting Point

Wire up `LayerPipeline.forward()` — the actual GPU dispatch per layer. The building blocks are all in place:
- I/O subsystem can read experts in parallel
- Metal context has all buffers and pipeline states
- Attention primitives (RMSNorm, RoPE, Softmax, BatchMatvec) are implemented
- Pipeline orchestration has the deferred expert state machine and layer pipeline

Next step is to implement the actual `LayerPipeline.forward()` method that wires all phases together for a single layer, then the `TokenGenerator.generate()` method that drives the 60-layer loop. This requires the `WeightFile` (mmap'd weight access) and `LayerWeightCache` (pre-computed weight pointers).

### Blockers

- **No model weights**: The 210GB model isn't downloaded. All testing is against synthetic fixtures and the real `shaders.metal`. End-to-end inference testing requires the model.

### Context Loss Warning

- `AlignedBuffer` is a class, not a `~Copyable struct`. This was a deliberate trade-off for `Array` compatibility. Do not refactor back to struct without solving the `Array<~Copyable>` limitation.
- The `nonisolated(unsafe)` in `IOPool` is correct — each task writes to a disjoint buffer. Don't remove it.

---

**Session Duration:** ~2 hours
**AI Model Used:** Claude Opus 4.6 (1M context)
