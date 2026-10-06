# Session Summary — Server Credential and CORS

**Date:** 2026-10-05
**Branch:** `fix/server-credential-and-cors`
**Outcome:** The six findings the listener-defaults review left open are fixed in code and
driven by tests over a real loopback socket. Deployed gate 0 errors / 0 warnings; 166 tests.

---

## Starting State

`HTTPServer` bound loopback by default (2026-10-03) and otherwise did nothing to decide who
could call it: no credential, `Access-Control-Allow-Origin: *`, an uncapped `max_tokens`, a
single-threaded accept loop reading headers one byte at a time with no deadline, and routing by
substring. The chat client logged the session id as `.public`.

## What Was Fixed

| # | Finding | Fix |
|---|---|---|
| 1 | No credential | `Authorization: Bearer`, SHA-256 digests compared in constant time; key from `SWIFT_MOE_API_KEY` or an owner-only `--api-key-file`; required off loopback, `--no-auth` on loopback only |
| 2 | CORS wildcard | No CORS headers by default; `--allow-origin` allowlist echoed exactly with `Vary: Origin`; other origins 403; `Host` check on loopback (421) |
| 3 | Unbounded tokens | Whole number 1…8192 or a 400 naming the field and range; rejected, not clamped |
| 4 | Stall | 16 KiB head (431), 64 KiB body (413), 10 s whole-request deadline (408) via `poll(2)`, 30 s write timeout, a thread per connection (16, then 503) |
| 5 | Loose routing | Request line parsed; exact method and path; 404 / 405 |
| 6 | Session id `.public` | `.private` |

## Decisions Worth Carrying Forward

- **Reject, do not clamp.** A clamped `max_tokens` returns a different answer from the one
  asked for with nothing to say so.
- **The ceiling is the KV cache allocation.** `TokenGenerator.defaultMaxSequenceLength` (8192,
  `GPU_KV_SEQ` in `infer.m`) is now the single name for it; `KVCache.append` stops recording
  past it rather than growing.
- **Every refusal drains.** The SwiftMCPServer 5.0.1 lesson applies to 401 as much as to 413:
  the body of a refused request is usually already on its way. `refuse` shuts the write side,
  then reads and drops the upload until it ends, the declared length passes, or 2 s go by.
  `refusalIsDeliveredEveryTime` and `refusalBeforeBodyIsDelivered` each run 20 times.
- **Threads, not a dispatch queue.** The first version put connections on a concurrent
  `DispatchQueue`; under the parallel test run every request went unanswered, because the pool
  did not grow while its threads sat in `poll(2)`. A worker here is blocked by design, so each
  connection gets a thread, bounded by two semaphores.
- **Credential before route.** Without a key every method and path answers 401, so an
  unauthenticated caller learns nothing about routes. Preflight is the one exception, because
  a preflight cannot carry a credential.
- **An unlisted `Origin` is refused, not merely unanswered.** Omitting the CORS header stops a
  page reading the stream; it does not stop a form post from starting inference on a
  `--no-auth` server.

## What Was Not Fixed

- **TLS.** Off loopback the key is sent in clear text.
- **Prompt length.** Only the 64 KiB body cap bounds it; see Remaining in `project/master_plan.md`.
- **`Host` is not checked off loopback** — the server cannot know its public name. The
  credential is the control there.
- **Queued requests wait without a deadline** once validated, behind the running inference.
- `--model` mode is still unimplemented, so the real-weights path has never served a request.

## Verification

- `swift test` — 166 tests, 29 suites; run six times in a row, all green.
- `quality-gate --check all --no-cache --continue-on-failure` — passed, 46/46, 0 errors / 0 warnings.
- By hand against `swift-moe-server --demo`: no key refuses to start; `--host 0.0.0.0 --no-auth`
  refuses to start; a request without a key is 401; with the key it streams; `max_tokens: 9000`
  is 400; a 300 KB body is 413.
