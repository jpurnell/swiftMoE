# Session Summary — Bounded Work, and Preparing the First Tag

**Date:** 2026-10-09
**Branch:** `fix/bounded-work` (from `origin/main` at `7031cdb`)
**Outcome:** The three things the credential-and-CORS fix left open are closed or made
explicit, the inference layer no longer drops context silently, and the repository is
prepared for its first tag (intended 0.1.0). Deployed gate 0 errors / 0 warnings; 226 tests.

---

## Starting State

`HTTPServer` bounded what a client could send (head, body, time, connections, `max_tokens`).
It did not bound what the request then cost:

- A prompt was limited only by the 64 KiB body — 65,536 tokens with the placeholder
  tokenizer, against KV caches of 8,192 positions — and `KVCache.append` returned without a
  word once full, so the answer was computed from a truncated context with nothing to say so.
- A validated request waited on an `NSLock` behind the running inference: no deadline, one of
  16 connection slots held throughout.
- A generation whose client had disconnected ran on until a write happened to fail; during
  prefill, which writes nothing, it could not be stopped at all.
- Off loopback the server required a key and then sent it in clear text, with a log warning.

## What Was Fixed

| # | Finding | Fix |
|---|---|---|
| 1 | Prompt length unbounded | `HTTPServer(tokenizer:)` counts the prompt; prompt + completion over `Limits.maxSequenceTokens` (8192) is 400 with the three numbers. Rejected, not shortened |
| 1 | Silent truncation | `KVCache.append` throws `FlashMoEError.sequenceCapacityExceeded`; `generate` checks the whole sequence before the first token |
| 2 | Queue with no deadline | `InferenceQueue`: FIFO, `Limits.queueDeadline` 30 s, then 503 + `Retry-After` through the refusal drain. Depth is bounded by `maxConnections` |
| 2 | Work for nobody | Queued: place and connection given up within 100 ms of the client leaving. Running: `SSEWriter.clientHasDisconnected` → `generate(shouldContinue:)`, asked before every token including prompt tokens |
| 3 | Plain text off loopback | Not TLS. `openListener()` refuses a non-loopback bind without `allowPlaintext` / `--allow-plaintext`; README has Caddy and nginx recipes |
| 4 | Found on re-reading | See below |

### Found on re-reading the server

- **One caller's prompt reached the next caller's answer.** `generate` restarted at position 0
  each call but kept the KV entries and linear-attention state of the last, and
  `swift-moe-server` never called `reset()`. `generate` now resets first.
- **SSE events were not always JSON.** 29 of the 32 control characters were written raw; the
  demo model emits them routinely (it emits NUL for the all-zero weights).
- **A completed stream could be reset away.** A client that had sent anything after its
  request lost the stream when the socket was closed over the unread bytes — 20 times in 20
  once there was a test for it. Streams now end with the same drain as refusals.
- `generate(maxTokens: -1)` trapped.
- The limit error for "default above ceiling" said "must be greater than zero".

## Decisions Worth Carrying Forward

- **Reject, do not reduce.** A prompt that leaves less room than `max_tokens` asks for is
  refused; the completion budget is not trimmed to fit. Same reason as for `max_tokens`
  itself: a different answer from the one requested, with nothing to say so.
- **The server owns the tokenizer.** The budget is checked with the tokens the handler then
  generates from (`HTTPServer.Request.promptTokens`), so the two cannot disagree. This is why
  the item could ship before `--model` mode, which the plan had said it should wait for.
- **Two independent guards.** The server's 400 and the generator's up-front throw both exist,
  and beneath both the cache throws. Any one would do today; none of them trusts the others.
- **A half-closed client is a departed client.** `poll(2)` reports the same end-of-stream for
  `shutdown(SHUT_WR)` and `close`. nginx and Go's `net/http` make the same call. It is a
  documented requirement on clients.
- **No queue-depth limit of its own.** Every waiter is a connection, so `maxConnections - 1`
  already bounds it; a second number would be a second thing to get wrong.
- **Semaphores, not a condition variable, in `InferenceQueue`.** One per waiter, signalled by
  direct hand-off under an unfair lock. It keeps the type `Sendable` with no mutable stored
  properties, and makes the hand-off/time-out race a single question: is my ticket still in
  the queue?
- **Tests wait on events, not on time.** `HTTPServer` has an internal observer;
  `RunningServer.events.awaitNext(.requestQueued)` is how a test knows the server got there.
  The only clocks are short configured limits (a 50–100 ms queue deadline) and generous
  allowances that bound a hang.

## What Was Not Fixed

- **TLS.** By decision: the raw-socket server does not get its own. `--allow-plaintext`
  encrypts nothing.
- **How long a response may take to deliver.** The 30 s write timeout is per `write(2)`. A
  slow but steady reader holds the model for hours. Needs a whole-stream budget and a decision
  about long legitimate generations.
- **Unauthenticated clients can occupy the 16 connection slots**, 10 s at a time.
- **A handler that ignores `clientHasDisconnected`** still runs to the end. The server cannot
  interrupt a closure; the shipped handler does ask.
- **No error event mid-stream.** A generation that throws after the headers ends the stream
  without `[DONE]`.
- `--model` mode; a `LICENSE` file.

## Verification

- `swift test` — 226 tests, 35 suites; see the PR for the consecutive-run count.
- `quality-gate --check all` — passed, 49 of 49 checkers (4 not applicable: `doc-claims`,
  `mcp-readiness`, `privacy-manifest`, `appintents-readiness`), 0 errors / 0 warnings.
- By hand against `swift-moe-server --demo`: `--host 0.0.0.0` with a key and no flag refuses
  to start with the message in the README; a 9,000-byte prompt is 400 with the three numbers;
  two requests in a row each stream; no key is 401.

## What the tagging commit must change

Nothing in `Sources/` — there is no version constant, and none should be added.

1. `CHANGELOG.md`: rename `## [Unreleased]` to `## [0.1.0] - <date>`, add a fresh empty
   `## [Unreleased]` above it, drop the paragraph that says no version has been tagged, and
   add link-reference definitions at the bottom (there are none today):
   `[Unreleased]: https://github.com/jpurnell/swiftMoE/compare/v0.1.0...HEAD` and
   `[0.1.0]: https://github.com/jpurnell/swiftMoE/releases/tag/v0.1.0`.
2. `README.md`: in **Installation**, replace `branch: "main"` with `from: "0.1.0"` and the
   sentence above it; in **Status**, replace "no version has been tagged yet".
3. `project/master_plan.md`: bump **Last Updated**.
4. `HANDOFF.md`: "Next step" item 1 is done.
5. Tag `v0.1.0` on that commit (annotated), push `main`, push the tag.

The pre-push hook refuses a branch that names an untagged version in a changelog heading or
an install snippet, which is why none of this is on the branch already.
