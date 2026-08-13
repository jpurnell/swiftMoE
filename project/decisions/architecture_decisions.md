# Architecture Decisions Log

**Purpose:** Machine-readable log of architectural decisions. Each entry is a YAML block.

> **When to add entries:**
> - Choosing between competing approaches (actor vs struct, sync vs async)
> - Establishing conventions (error handling, naming, file structure)
> - Making tradeoffs (performance vs safety, simplicity vs flexibility)
> - Rejecting a previously considered approach

---

## Decisions

```yaml
id: ADR-001
date: 2026-04-05
status: proposed
category: architecture
title: Use ~Copyable types for system resource ownership
context: |
  The original Obj-C code manages file descriptors (open/close) and aligned memory
  (posix_memalign/free) manually via C patterns. The MetalContext struct has ~40 buffer
  fields with nil deallocators, and file descriptors are managed through global arrays.
  Bugs in resource lifecycle were a source of discarded experiments.
decision: |
  ExpertFile (file descriptor) and AlignedBuffer (posix_memalign'd memory + MTLBuffer)
  use Swift's ~Copyable (move-only) types to enforce single ownership at compile time.
  Resources are freed in deinit. No ARC overhead because these are non-class types.
rationale:
  - "Eliminates use-after-close and double-free bugs by construction"
  - "Zero runtime overhead compared to manual C resource management"
  - "Better than ARC-based class wrappers (no reference counting in hot path)"
  - "Enables borrowing semantics — pread borrows ExpertFile without consuming it"
consequences: |
  + Compile-time proof that every fd is closed and every buffer is freed
  + No retain/release traffic in the 2.9ms per-layer budget
  - Requires Swift 5.9+ for ~Copyable support
  - Some API patterns need borrowing/consuming annotations
alternatives_rejected:
  - "Class with deinit: Adds ARC overhead in tight loops (retain/release on each buffer access)"
  - "Manual C-style open/close: The original approach; error-prone, not testable in isolation"
  - "Unmanaged<T>: Still requires manual retain/release, no compiler enforcement"
affected_files:
  - Sources/FlashMoE/IO/ExpertFile.swift
  - Sources/FlashMoE/IO/AlignedBuffer.swift
supersedes: null
amends: null
superseded_by: null
```

---

```yaml
id: ADR-002
date: 2026-04-05
status: proposed
category: concurrency
title: Use Swift structured concurrency for parallel I/O instead of pthreads
context: |
  The original engine uses a hand-rolled pthread pool (infer.m:2970-3120) with
  pthread_mutex/pthread_cond for parallel expert loading. This pattern works but:
  (1) is not testable in isolation, (2) has no compile-time data race prevention,
  (3) requires manual shutdown protocol, and (4) contributed to the 42% experiment
  discard rate during the autoresearch sprint (cache eviction races, buffer mapping bugs).
decision: |
  Replace pthread pool with Swift async/await and TaskGroup for parallel pread.
  Each expert read becomes a Task within a TaskGroup, matching the existing
  "read K=4 experts in parallel" pattern exactly.
rationale:
  - "Compile-time Sendable checking prevents data races"
  - "Built-in cancellation and priority propagation"
  - "No manual mutex/cond lifecycle management"
  - "TaskGroup maps directly to the read-K-experts pattern"
  - "Testable: can await results and verify in test code"
consequences: |
  + Data race prevention at compile time via Sendable
  + Cleaner shutdown (structured = automatic)
  + Testable I/O operations
  - Cooperative threading: blocking pread inside Task may exhaust thread pool
  - Mitigation: K=4 concurrent reads of 1.37ms each is within tolerance
  - If benchmarks show issues, fall back to DispatchQueue-based wrapper
alternatives_rejected:
  - "Keep pthreads: Works but no compile-time safety, not testable"
  - "GCD dispatch_apply: Already tried in original; comparable performance but no Sendable checking"
  - "DispatchIO: Tested in original (infer.m experiment log); 70% slower due to dispatch_data overhead"
  - "OperationQueue: Higher overhead than TaskGroup for this simple fan-out pattern"
affected_files:
  - Sources/FlashMoE/IO/IOPool.swift
supersedes: null
amends: null
superseded_by: null
```

---

```yaml
id: ADR-003
date: 2026-04-05
status: proposed
category: architecture
title: Incremental module extraction, not monolithic rewrite
context: |
  The original infer.m is 7,151 lines containing I/O, Metal, attention, pipeline,
  server, and CLI code. A full rewrite risks introducing regressions in the
  carefully tuned pipeline. The 90-experiment optimization history means every
  line has been benchmarked.
decision: |
  Extract modules incrementally, one subsystem at a time, with comparative
  benchmarks at each phase boundary. The original Obj-C code remains in the
  repo as reference. Each phase produces a standalone benchmark proving
  performance parity before proceeding.
rationale:
  - "Proven performance characteristics must be preserved"
  - "Each phase is independently valuable and shippable"
  - "Regressions are caught immediately via per-phase benchmarks"
  - "Original code serves as executable specification"
consequences: |
  + Risk-controlled: any phase can be reverted independently
  + Always have a working system (original) to fall back to
  - Temporary code duplication during transition
  - Need to maintain both build systems until Phase 5+
alternatives_rejected:
  - "Full rewrite: Too risky for a performance-critical system with 90 experiments of tuning"
  - "Wrapper approach (Swift calling Obj-C): Adds overhead without gaining type safety"
affected_files:
  - All Sources/FlashMoE/ files (new)
  - metal_infer/ (unchanged, reference)
supersedes: null
amends: null
superseded_by: null
```

---

```yaml
id: ADR-004
date: 2026-04-05
status: accepted
category: architecture
title: Runtime ModelConfig instead of compile-time constants
context: |
  The original C code used #define for all model constants (hidden_dim=4096,
  num_layers=60, etc.), hardcoding to Qwen3.5-397B. This made the engine
  untestable without 210GB of weights and unable to support other MoE models.
decision: |
  ModelConfig is a runtime struct with model presets (.qwen397B, .tiny),
  not a compile-time enum. Every type receives dimensions from config.
rationale:
  - "Enables testing with tiny synthetic models without 210GB download"
  - "Supports multiple MoE architectures (DeepSeek-V3, Mixtral, future models)"
  - "The paper explicitly says the approach generalizes to any MoE model"
consequences: |
  + Can test full pipeline with ModelConfig.tiny (~1MB vs 210GB)
  + New model support = new ModelConfig preset, no code changes
  - Every buffer/pipeline type takes config parameter (more verbose)
alternatives_rejected:
  - "Keep hardcoded: Works for one model but untestable without 210GB weights"
  - "Generic over config type: Over-engineered, runtime struct is simpler"
affected_files:
  - Sources/FlashMoE/Model/ModelConfig.swift
  - All buffer group, pipeline, and inference files
supersedes: null
amends: null
superseded_by: null
```

---

```yaml
id: ADR-005
date: 2026-04-05
status: accepted
category: architecture
title: No hardcoded model-specific constants — all dimensions flow from ModelConfig
context: |
  During the ModelConfig refactor, we discovered that hardcoded constants had
  leaked into buffer sizes, loop bounds, layer counts, and expert offsets
  across 18 files. Each was a hidden assumption that would silently break
  for any model other than Qwen3.5-397B.
decision: |
  POLICY: No model-specific numeric literal may appear in source code outside
  of ModelConfig presets. All dimensions, counts, sizes, offsets, token IDs,
  and architecture parameters must flow from a ModelConfig instance.

  Permitted hardcoded values:
    - Hardware constants: DMA alignment (2MB), page size (16KB), Metal threadgroup sizes
    - Math constants: epsilon values, pi, etc.
    - Protocol constants: max batch slots, max K (these are engine limits, not model properties)

  Forbidden:
    - Layer counts (60), expert counts (512), hidden dimensions (4096)
    - Attention head counts, KV head counts, head dimensions
    - Vocab sizes, token IDs (EOS, think tokens)
    - Expert binary layout offsets
    - Any value that would change for a different model

  Enforcement: Design proposal review must check for hardcoded model constants.
  Tests should use ModelConfig.tiny where possible to prove model-independence.
rationale:
  - "Hardcoded constants are the #1 barrier to model portability"
  - "Testing with .tiny caught assumptions that would have been invisible with .qwen397B"
  - "The paper's value proposition is the technique, not the specific model"
consequences: |
  + Any MoE model can be supported by adding a ModelConfig preset
  + Integration tests run in milliseconds with .tiny instead of requiring 210GB
  + New contributors can test without downloading the model
  - Slightly more verbose init signatures (config: parameter everywhere)
  - Must audit new code during review for leaked constants
alternatives_rejected:
  - "Allow hardcoded with TODO comments: TODOs never get fixed"
  - "Separate 'model-specific' module: Splits logic that belongs together"
affected_files:
  - All files — this is a project-wide policy
supersedes: null
amends: null
superseded_by: null
```

---

**Last Updated:** 2026-04-05
