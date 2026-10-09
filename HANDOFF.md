# HANDOFF — SwiftMoE

**Last session:** 2026-10-09 (bounded work, and preparation for the first tag — see `project/summaries/2026-10-09_BoundedWork.md`)
**Branch:** `fix/bounded-work` (PR open against `main`, not merged, not tagged)
**State:** Green. Quality gate 0/0, 226 tests passing, nothing in progress.

`swift-moe-server` now refuses a prompt that does not fit in a sequence (400), gives a queued
request 30 s to start (503 + `Retry-After`), stops generating for a client that has left, and
will not bind anything but loopback without `--allow-plaintext`. The inference layer throws
instead of dropping context.

**The next thing to do is tag 0.1.0**, once the PR is merged. The summary lists exactly what
the tagging commit has to change.

---

## Where things stand

The quality gate is clean at **0 errors / 0 warnings**, with no overrides, suppressions, or
config exclusions:

```bash
quality-gate --check all               # 49 of 49 checkers; 4 are not applicable and SKIP
swift test                             # 226 tests, 35 suites, ~2 s
cd metal_infer && make                 # builds; 19 pre-existing warnings
```

The plain `quality-gate` invocation (what the pre-commit hook runs) selects 43 of the 49;
the pre-push hook runs `--check all`. Use `--continue-on-failure` when investigating: the gate
halts at the first failing checker otherwise.

The server tests drive a real `HTTPServer` on a loopback ephemeral port
(`Tests/SwiftMoETests/HTTPTestSupport.swift`). Where a test needs the server to have reached a
state — a request queued, a connection released — it waits on `RunningServer.events`, which
the server feeds through an internal observer. Do not replace those waits with sleeps.

## Worth carrying forward

From the bounded-work session (2026-10-09):

- **`TokenGenerator.generate` is a new sequence every call.** It resets the KV caches and
  linear-attention state first. It always restarted at position 0; it used to keep the
  previous call's cache entries as well, so one HTTP caller's prompt was attended to by the
  next caller's request. Multi-turn KV reuse, if it is ever wanted, needs a real design — a
  position that continues, and a cache keyed by conversation.
- **`KVCache.append` throws at capacity.** Do not "fix" a `sequenceCapacityExceeded` by
  catching it and carrying on: the point is that a truncated history is never computed from.
- **The server owns the tokenizer.** `HTTPServer(tokenizer:)` counts the prompt and hands the
  same tokens to the handler in `HTTPServer.Request`. `--model` mode should pass the BPE
  tokenizer through that parameter, not tokenize again in the handler.
- **A handler must ask `writer.clientHasDisconnected`.** The server cannot interrupt a
  handler. `swift-moe-server` passes it to `generate(shouldContinue:)`.
- **A half-closed client counts as gone.** `poll(2)` cannot tell `shutdown(SHUT_WR)` from
  `close`. This is documented in the README and CHANGELOG as a client requirement.
- **Every response ends with a drain, streams included.** Closing a socket over unread bytes
  resets it and the client loses what it had not read yet.

From the quality-gate session (2026-08-25, `project/summaries/2026-08-25_QualityGateZeroZero.md`):

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

1. ~~**Merge the PR and tag 0.1.0.**~~ Done 2026-10-09 (`v0.1.0`). See "What the tagging commit must change" in the summary.
2. **Numerical equivalence against real Qwen3.5-397B weights.** Still unvalidated, and the
   largest open risk. Everything is currently verified against synthetic fixtures and a
   CPU reference; the 209GB model has never been run through the Swift engine.
3. **`swift-moe-server --model`.** Not implemented; the server only serves `--demo`.
4. **Performance benchmarking vs. the original C engine.** No Swift-side numbers exist yet.
   The C engine's baseline is 4.36 tok/s at 4-bit (see `CLAUDE.md`).
5. The open server items in `project/master_plan.md` under Remaining: a whole-response
   deadline, unauthenticated connection slots, a mid-stream error event.
6. **Additional model presets** (DeepSeek-V3, Mixtral).

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
