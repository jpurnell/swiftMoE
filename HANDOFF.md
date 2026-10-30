# HANDOFF — SwiftMoE

**Last session:** 2026-10-30 (the HTTP server moved onto SwiftHTTPService, v0.2.0, then kit 0.6.0, v0.2.1 — see `project/summaries/2026-10-30_HTTPOnKit.md` and `2026-10-30_KitSixZero.md`)
**Branch:** `main` at `v0.2.1`
**State:** Green. Quality gate 0/0, 255 tests in 40 suites, nothing in progress.

The raw-socket server is gone. `SwiftMoEHTTP` (new library) serves `POST /v1/chat/completions` on
`SwiftHTTPService` 0.5.0: TLS in the process (`--tls-cert`, `--tls-key`, SIGHUP reload),
`--allow-plaintext` removed, a two-hour bound on a response, graceful shutdown. `SwiftMoE` imports no
networking.

**The connection-cap regression is fixed** (0.2.1, kit 0.6.0): over the cap a plaintext client gets `503` with
`Retry-After` = the queue deadline. Six accepted differences from 0.1.0 remain (CHANGELOG 0.2.1).

**The next things** are the ones in "Next step" below; none is blocked by this release.

---

## Where things stand

The quality gate is clean at **0 errors / 0 warnings**, with no overrides, suppressions, or
config exclusions:

```bash
quality-gate --check all               # 49 of 49 checkers; 4 are not applicable and SKIP
swift test                             # 255 tests, 40 suites, ~20 s (the server tests drive real sockets)
cd metal_infer && make                 # builds; 19 pre-existing warnings
```

The plain `quality-gate` invocation (what the pre-commit hook runs) selects 43 of the 49;
the pre-push hook runs `--check all`. Use `--continue-on-failure` when investigating: the gate
halts at the first failing checker otherwise.

The server tests drive a real `ChatService` on a loopback ephemeral port
(`Tests/SwiftMoETests/HTTPTestSupport.swift`). Where a test needs the server to have reached a
state — a request queued, a slot released — it waits on `RunningServer.events`, which the service
feeds through an internal observer. Do not replace those waits with sleeps.

**Every socket test lives under `HTTPServiceTests`, which is `.serialized`.** Swift Testing runs a
synchronous test on a cooperative-pool thread; a test that blocks that thread on a socket needs the
server's tasks to get a pool thread, so enough of them at once deadlock the whole run. Put a new
socket test in an `extension HTTPServiceTests`. `blocking { }` runs async setup on a thread pool of
its own for the same reason.

`Tests/SwiftMoETests/Fixtures/exchanges.json` is the OLD server's recorded wire behaviour (85
exchanges, from v0.1.0). Never regenerate it to make a test pass. `WireCharacterizationTests`
replays it against the current server and fails on any difference that is not in its tables
(`expectedToChange`, `intended`, the stream header changes).

## Worth carrying forward

From the bounded-work session (2026-10-09):

- **`TokenGenerator.generate` is a new sequence every call.** It resets the KV caches and
  linear-attention state first. It always restarted at position 0; it used to keep the
  previous call's cache entries as well, so one HTTP caller's prompt was attended to by the
  next caller's request. Multi-turn KV reuse, if it is ever wanted, needs a real design — a
  position that continues, and a cache keyed by conversation.
- **`KVCache.append` throws at capacity.** Do not "fix" a `sequenceCapacityExceeded` by
  catching it and carrying on: the point is that a truncated history is never computed from.
- **The service owns the tokenizer.** `ChatService(tokenizer:)` counts the prompt and hands the
  same tokens to the generator in `ChatCompletionRequest`. `--model` mode should pass the BPE
  tokenizer through that parameter, not tokenize again in the generator.
- **A generator must ask `shouldContinue()` and honour `emit` returning false.** The service cannot
  interrupt it. The client leaving cancels the stream's task (the kit does that), which sets the flag
  `shouldContinue` reads before every token, prompt tokens included.
- **The slot outlives the stream by one thing: the generator thread.** The producer awaits
  `GenerationBridge.stopped()` before it returns, so the next request never starts while the model is
  still being used. Do not remove that wait.
- **A half-closed client still counts as gone** (tested): the kit cancels on FIN.

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
5. The open server items in `project/master_plan.md` under Remaining: unauthenticated connection slots, and the mid-stream error event (decided:
   keep the truncation).
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
