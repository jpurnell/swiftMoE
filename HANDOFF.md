# HANDOFF — SwiftMoE

**Last session:** 2026-08-25
**Branch:** `main` (commit `5fb88bc`)
**State:** Green. Quality gate 0/0, 84 tests passing, nothing in progress.

---

## Where things stand

The quality gate is clean at **0 errors / 0 warnings across all 45 checkers**, with no
overrides, suppressions, or config exclusions. Verified uncached:

```bash
quality-gate --check all --no-cache    # 45/45, 0/0
swift test                             # 84 tests, 22 suites
cd metal_infer && make                 # builds; 19 pre-existing warnings
```

Note the plain `quality-gate` invocation runs 40 of 45 checkers ("5 not selected") and
caches results. Use `--check all --no-cache` to see the real picture — a cached run will
happily report PASSED for files it never examined, which is how the gaps fixed last
session stayed hidden. Also use `--continue-on-failure`: the gate halts at the first
failing checker, and a run that stops at `[safety]` leaves 31 checkers unreported.

## What the last session did

Drove the gate from 10 errors / 32 warnings to 0/0. Full detail in
`project/summaries/2026-08-25_QualityGateZeroZero.md`. The parts worth carrying forward:

- **`compute_decay_beta` had a real out-of-bounds hazard** — it indexed six buffers by
  thread id with no bound and took no count to bound against. Now takes `num_v_heads`;
  all three dispatch sites (one Swift, two ObjC) bind it.
- **Command buffers are now checked.** `waitUntilCompletedChecked(_:)` in
  `Sources/SwiftMoE/Metal/CommandBufferWait.swift` traps on a failed dispatch. Use it
  instead of bare `waitUntilCompleted()` anywhere results are read afterwards — a failed
  buffer returns its *previous* contents, so the failure otherwise surfaces later as
  wrong tokens with nothing pointing back at it.
- **`dispatchExactly(...)`** in `Sources/SwiftMoE/Metal/ExactDispatch.swift` replaces
  rounded threadgroup dispatches. Read its doc comment before using it on a new kernel:
  it is only valid when the kernel tolerates a partially populated final threadgroup.
- **`dequant_matvec_4bit_v3` now strides its cooperative load by
  `[[threads_per_threadgroup]]`** rather than a hardcoded 256. This is load-bearing —
  reverting it while keeping the exact dispatch produces `NaN`. Four other kernels in
  `shaders.metal` already used this pattern; prefer it for any new tiled kernel.
- **`TestPaths`** (`Tests/SwiftMoETests/TestPaths.swift`) resolves test paths from
  `#filePath` with containment checks. New tests should go through it rather than calling
  string-path `FileManager` APIs, which the `safety` checker flags as CWE-22.

## Next step

Nothing is half-finished — pick up whatever is next by priority. The standing candidates,
unchanged by last session:

1. **Numerical equivalence against real Qwen3.5-397B weights.** Still unvalidated, and the
   largest open risk. Everything is currently verified against synthetic fixtures and a
   CPU reference; the 209GB model has never been run through the Swift engine.
2. **Performance benchmarking vs. the original C engine.** No Swift-side numbers exist yet.
   The C engine's baseline is 4.36 tok/s at 4-bit (see `CLAUDE.md`).
3. **DocC documentation generation.**
4. **Additional model presets** (DeepSeek-V3, Mixtral).

## Known noise (pre-existing, outside the gate)

Neither is a regression and neither blocks anything, but both are real:

- `metal_infer/shaders.metal` — 3 Metal compiler warnings about threadgroup variables
  (`broadcast_max`, `broadcast_sum`, `k_sum_sq`) "used uninitialized". They are
  barrier-synchronized, so the warnings are false positives, but they are noise.
- `metal_infer/infer.m` — 19 clang warnings, mostly unused static functions left over from
  discarded experiments (`parallel_pread_experts`, `infer_prefetch_*`, `server_save_turn`).
  Candidates for deletion; the experiment log in `CLAUDE.md` records why each was dropped.

## Conventions

- Resume artifacts: this file first, then `project/master_plan.md`, then the newest file in
  `project/summaries/`.
- `development-guidelines/` is a **cloned template repo and is gitignored** — do not write
  session summaries or project state into it. Project-local state lives in `project/`.
- Quality gate must be 0/0 before commit, fixed at the root. No override comments,
  suppression annotations, or config exclusions.
