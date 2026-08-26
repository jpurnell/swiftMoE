# Session Summary — Quality Gate to 0 Errors / 0 Warnings

**Date:** 2026-08-25
**Branch:** main
**Outcome:** All 45 checkers pass with 0 errors and 0 warnings, no overrides or suppressions.

---

## Starting State

The updated gate flagged 4 pointer-escape errors in `BPETokenizerTests`. Running the full
gate with `--continue-on-failure` showed the run had been halting at `[safety]`, leaving 31
checkers unreported. The real total was 10 errors and 32 warnings across four checkers:
`safety`, `gpu-safety`, `liveness`, and `consistency`.

---

## What Was Fixed

### pointer-escape (4 errors)
`BPETokenizerTests` built its fixture with `withUnsafeBytes(of:) { data.append(contentsOf: $0) }`.
That pattern is safe — `Data.append(contentsOf:)` copies before the block exits — but rather
than suppress the auditor the helpers now assemble bytes explicitly in little-endian order.
Side benefit: the fixture's byte order no longer depends on host endianness.

### safety (3 errors, 12 warnings)
- Three `UnsafeRawPointer(bitPattern: 1)!` force unwraps in `BatchMatvecTests` replaced with a
  real allocation.
- Twelve CWE-22 path-traversal warnings came from string-path `FileManager` APIs (`atPath:`)
  fed by a `shaderPath` helper that searched relative to the process working directory. Probing
  the checker showed it accepts URL-based APIs, which is also the idiomatic form. Added
  `TestPaths`, which resolves against a root derived from `#filePath`, standardizes, and rejects
  anything that escapes the root; every call site now routes through it.

### gpu-safety (6 errors, 14 warnings)
- **Unchecked command buffers (3 errors).** `DeferredExpertState`, `LayerPipeline`, and
  `CMD2Encoder` read results after `waitUntilCompleted()` without checking `status`. A failed
  dispatch leaves the previous contents in place, so the failure would have surfaced later as
  wrong tokens with nothing pointing back at it. Added
  `MTLCommandBuffer.waitUntilCompletedChecked(_:)`, following the existing `preconditionFailure`
  idiom for unrecoverable Metal errors.
- **Unbounded thread ids (3 errors).** `swiglu_fused_vec4` and `swiglu_fused_batched` were false
  positives — they bounded `tid` against a derived local the checker could not trace — and were
  rewritten to compare against the parameter expression directly. `compute_decay_beta` was a
  genuine gap: it indexed six buffers by `idx` with no bound and took no count to bound against.
  It now takes `num_v_heads` and guards on it; all three dispatch sites (one Swift, two ObjC)
  bind the new parameter.
- **Rounded dispatches (14 warnings).** Six elementwise dispatches moved to a new
  `dispatchExactly(threadCount:threadsPerThreadgroup:device:)` helper, which uses
  `dispatchThreads` where supported and falls back to the previous rounded dispatch otherwise.

  The seventh — `dequant_matvec_4bit_v3` — could not simply switch: it reduces across SIMD
  groups and cooperatively loads the input vector into threadgroup memory, and its load loop
  strided by a hardcoded 256. The fix was to stride by `[[threads_per_threadgroup]]` instead,
  matching what four other kernels in the file already do. That makes a partially populated
  final threadgroup safe, after which dispatching `outDim * 32` threads is exact: a tail group
  is still a whole number of SIMD groups, so `simd_sum` keeps its full width. The change is
  backward compatible with the C engine, which keeps dispatching full 256-thread groups.

### liveness (1 error)
`SwiftMoEChat` waited on a `DispatchSemaphore` with no deadline, so an unresponsive server
blocked the CLI forever. The wait is now bounded by a deadline that scales with `maxTokens`
(a conservative 1 tok/s floor plus a fixed startup allowance, against a measured ~4.4 tok/s),
and cancels the request when it fires.

### consistency
Resolved as a consequence of the above — score went 0.00 → 1.00 (threshold 0.70). The last
contributing cluster was a missing privacy annotation on the log line added for the timeout.

---

## Verification

- `quality-gate --check all --no-cache` — 45/45 checkers, 0 errors, 0 warnings
- `swift test` — 84 tests in 22 suites, all passing
- `metal_infer` clean rebuild — builds, and its 19 pre-existing compiler warnings are an
  identical set before and after (diffed)

### Numeric validation of the v3 kernel change

The riskiest change touches a hot kernel in the real inference path, which cannot be exercised
without the 209GB model. Added `DequantMatvecV3Tests`, which builds a synthetic 4-bit quantized
matrix and compares GPU output against `Embedding.cpuDequantMatvec` for `outDim` of 1, 5, 8, 13,
16, and 23 — covering both full and partial final threadgroups.

The test was confirmed load-bearing: reverting only the `tg_size` stride while keeping the exact
dispatch makes it fail with `NaN` and values around `5.5e37`, which is precisely the
partially-filled `x_shared` hazard. With the fix in place all cases match within tolerance.

---

## Notes for Next Session

- `metal_infer/shaders.metal` still emits 3 pre-existing Metal compiler warnings about
  threadgroup variables (`broadcast_max`, `broadcast_sum`, `k_sum_sq`) being "used uninitialized".
  They are barrier-synchronized and outside the quality gate's scope, but they are real noise.
- `metal_infer/infer.m` has 19 pre-existing clang warnings, mostly unused static functions left
  over from discarded experiments.
- Numerical equivalence against the real Qwen3.5-397B weights remains unvalidated (unchanged
  from before this session).
