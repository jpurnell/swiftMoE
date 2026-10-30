# Session Summary — The HTTP Server Moves Onto SwiftHTTPService

**Date:** 2026-10-30
**Branch:** `feat/http-on-kit`, merged to `main`, tagged `v0.2.0`
**Outcome:** The raw-socket server (1,879 lines under `Sources/SwiftMoE/Server/`) is replaced by
`SwiftMoEHTTP` on SwiftHTTPService 0.5.0. Gate 0/0; 254 tests in 40 suites.
Design: `project/plans/proposals/DesignProposal_HTTPServerOnSwiftHTTPService.md` (status: implemented).

## Order of work
1. **Characterize first.** 85 exchanges of the old server recorded from v0.1.0 into
   `Fixtures/exchanges.json`, before anything was replaced; six marked `expectedToChange`.
2. **New target.** `ChatService`, `InferenceGate` (actor), `GenerationBridge` (dedicated thread,
   credit backpressure), `InferenceLimits`, `OpenAIError`, `ChatChunk`; `APIKey` and friends moved.
   Middleware: access log, OpenAI-shaped problems, `allowedHosts`, `cors`, bearer, `Router`.
3. **Replay.** The recorded exchanges run against the new server. Nine differences the proposal did
   not predict were found and accepted by the owner's delegate; they are `intended` entries with
   reasons, and the replay fails on any other difference.
4. **Spikes S2 to S5** as tests, **14 hand mutations** (all killed), TLS, mid-stream drop, graceful
   shutdown and layer-order tests.

## What surprised
- The kit answers an HTTP/1.0 request with an HTTP/1.0 status line; the proposal (and the old
  server) said 1.1.
- Past `maxConnections` the kit closes with **no response**: no 503, no `Retry-After`. A known
  regression, not worked around.
- A body over the route bound is 413 before the credential check (old: 401).
- A `DispatchSemaphore` traps if destroyed below its initial value, which a cancelled generation
  does; `GenerationBridge` uses its own credit counter.
- Socket tests deadlock Swift Testing's cooperative pool unless serialized (see HANDOFF).
- The S4 test caught that `slotReleased` was reported before the gate had taken the slot back; the
  event now fires after.

## Smoke run (real executable)
`swift-moe-server --demo` with a 0600 key file: bearer POST returns an SSE stream ending in
`data: [DONE]`; no key is 401; SIGTERM exits. With `--tls-cert/--tls-key` (throwaway openssl
certificate) and `curl -k`: same stream over TLS; plain HTTP to the TLS port gets nothing.
`--host 0.0.0.0` without TLS refuses to start with the message in the README.

## Not verified
Real weights (still demo only); an OpenAI client library against the stream (Foundation JSON only);
HTTP/2 (off by decision O4).
