# Implementation Checklist: Phase 2 — Metal Context

**Design Proposal:** [Phase2_MetalContext.md](../02_IMPLEMENTATION_PLANS/UPCOMING/Phase2_MetalContext.md)

---

## Current Phase: Phase 2 — Metal Context + Buffer Management

### Completed
- [x] `ShaderLibrary` — runtime shader compilation, 14 required + 5 optional pipelines (3 tests)
- [x] `ProjectionBuffers` — input/output/batch slots
- [x] `ExpertBuffers` — K=8 double-buffered, quantization-aware sizing (5 tests)
- [x] `AttentionBuffers` — 15 KV caches + scratch buffers
- [x] `LinearAttentionBuffers` — 45 delta-net states + conv states + scratch
- [x] `CombineBuffers` — CMD3 residual/hidden/params/norms
- [x] `MetalContext` — top-level coordinator (4 tests)
- [x] `AlignedBuffer` pivot: struct ~Copyable → final class (Array compat)

### Design Decisions
- `AlignedBuffer` changed from `~Copyable struct` to `final class` — `Array<~Copyable>` not supported in Swift. ARC overhead negligible (startup-only allocation).
- Expert data buffers size based on quantization mode (4MB for 2-bit vs 8MB for 4-bit per slot), saving 64MB of GPU-visible shared memory in 2-bit mode.

### Quality Gate
- Build: 0 warnings
- Tests: 29/29 passing (0.554s) — 12 new tests for Phase 2
- Safety: No force unwraps, force casts, or try!

---

**Last Updated:** 2026-04-05
