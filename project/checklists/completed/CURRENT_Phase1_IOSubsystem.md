# Implementation Checklist: Phase 1 — I/O Subsystem

**Design Proposal:** [Phase1_IOSubsystem.md](../02_IMPLEMENTATION_PLANS/UPCOMING/Phase1_IOSubsystem.md)

---

## Current Phase: Phase 1 — Foundation

### In Progress
- [ ] Benchmark: pread throughput vs. Obj-C baseline

### Completed
- [x] SPM package structure with C interop targets (CTokenizer, CLineNoise)
- [x] `ModelConfig` — model constants (all #defines from infer.m)
- [x] `FlashMoEError` — error types (6 cases)
- [x] `ExpertFile` — ~Copyable file descriptor wrapper (5 tests)
- [x] `AlignedBuffer` — ~Copyable 2MB-aligned Metal buffer (4 tests)
- [x] `IOPool` — parallel pread with latency tracking via TaskGroup (3 tests)
- [x] `WeightManifest` — JSON tensor manifest loader with O(1) lookup (5 tests)

### Blocked
*(none)*

---

## Module Status

| Module | Status | Tests | Docs | Warnings |
|--------|--------|-------|------|----------|
| Model/ModelConfig | Complete | N/A (constants) | Yes | 0 |
| Model/FlashMoEError | Complete | N/A (error enum) | Yes | 0 |
| IO/ExpertFile | Complete | 5 passing | Yes | 0 |
| IO/AlignedBuffer | Complete | 4 passing | Yes | 0 |
| IO/IOPool | Complete | 3 passing | Yes | 0 |
| IO/WeightManifest | Complete | 5 passing | Yes | 0 |
| CTokenizer | Complete | N/A (C interop) | N/A | 0 |
| CLineNoise | Complete | N/A (C interop) | N/A | 0 |

### Quality Gate
- Build: 0 warnings
- Tests: 17/17 passing (0.035s)
- Safety: No force unwraps, force casts, or try!

---

**Last Updated:** 2026-04-05
