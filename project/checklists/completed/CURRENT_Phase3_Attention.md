# Implementation Checklist: Phase 3 — Attention Computation

**Design Proposal:** [Phase3_Attention.md](../02_IMPLEMENTATION_PLANS/UPCOMING/Phase3_Attention.md)

---

## Current Phase: Phase 3 — Attention Computation

### Completed
- [x] `BFloat16` — bf16 ↔ f32 conversion with @inline(__always) (3 tests)
- [x] `RMSNorm` — standard, bare, and gated variants (4 tests)
- [x] `Softmax` — numerically stable with max-subtraction trick (5 tests)
- [x] `RoPE` — rotary position embeddings with partial rotation (3 tests)
- [x] `BatchMatvecSpec` — GPU dispatch specification struct (1 test)
- [x] `BatchMatvec.encode` — GPU command buffer encoding for batched matvec
- [x] `BatchMatvec.flushResults` — GPU→CPU result copy

### Deferred to Phase 4
- [ ] `FullAttention` — complete forward pass (needs pipeline orchestration)
- [ ] `LinearAttention` — GatedDeltaNet + conv1d (needs BLAS + pipeline)
- [ ] End-to-end GPU matvec validation (needs real model-sized weights)

### Quality Gate
- Build: 0 warnings
- Tests: 45/45 passing (0.428s) — 16 new tests for Phase 3
- Safety: No force unwraps, force casts, or try!

### Design Decision
- `AlignedBuffer` remains `final class` (from Phase 2 pivot)
- Accelerate BLAS will be called directly in Phase 4 for delta-net recurrence

---

**Last Updated:** 2026-04-05
