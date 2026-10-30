# Design Proposal: Move the HTTP server onto SwiftHTTPService

**Status:** Implemented in v0.2.0 and corrected in v0.2.1 on kit 0.6.0 (2026-10-30): rows 1, 3, 6 and the silent-client 408 of the table at the end of section 7 no longer differ. Approved by the owner 2026-10-30, with every recommendation below accepted (section 13 records the decisions); the differences the replay found beyond section 7 were accepted the same day and are the table at the end of section 7. **Author's note:** every claim about SwiftHTTPService (the kit) below was
checked against its source at v0.4.1 (`Sources/SwiftHTTPService*`), and every claim about this server against
`Sources/SwiftMoE/Server/*` and `Sources/SwiftMoEServer/main.swift` at `main` (tag `v0.1.0`). Where a claim rests on
behaviour I could not confirm by reading, it is marked **verify** and has a spike in section 10.

## 1. Objective

Replace this package's hand-rolled HTTP listener (about 2,400 lines under `Sources/SwiftMoE/Server/`, raw `socket(2)`,
a thread per connection) with the owner's `SwiftHTTPService` kit, keeping every property the server has today:
the OpenAI-compatible `POST /v1/chat/completions`, SSE streaming, bearer authentication, the sequence budget and its
400s, one model run at a time behind a deadline-bounded queue, and cancellation when a client leaves.

**Done when:** the existing server-level tests pass against the new implementation with their assertions unchanged
(only the way a server is constructed changes), a recorded exchange of the old server replays against the new with
only written-down differences, and the raw-socket server code is deleted.

## 2. Motivation

- The owner's rule (migration decision D4): the kit covers every server's needs so that no second HTTP system is
  maintained. This is the fourth hand-rolled server to move (`ijs-dashboard-server`, SwiftPushHub and FleetWatch have).
- Four items on this package's own master plan are things the kit already does:
  TLS (open: "will not grow TLS of its own"), a whole-stream budget (open: "no bound on how long one response may take
  to deliver"), a slow-reader bound, HTTP/2 and graceful shutdown. They stop being this package's to maintain.
- The server holds the expensive resource in the workspace (a 30-minute GPU generation), so it is where a transport
  bug costs most. It deserves the transport that has been mutation-tested and soaked by three other servers.

## 3. Do we cover everything? The capability map

"Covered" means the kit provides it or it is application logic that moves unchanged. Nothing here needs the kit to
change **except the four rows marked K**.

| Capability today | Where it lives | On the kit | Verdict |
|---|---|---|---|
| Exact route `POST /v1/chat/completions`; 404 and 405 with `Allow` | `HTTPConnectionProcessor.decide` | `Router`, with `notFound` and `methodNotAllowed` replaced to the OpenAI error shape | Covered |
| Body bound 64 KiB (413), `Content-Length` required (411), `Transfer-Encoding` refused (501) | `decide`, `Limits.maxBodyBytes` | Per-route `maximumBodyBytes`; the transport refuses what is ambiguous (smuggling hygiene) | Covered; status for chunked uploads differs (section 7) |
| Header bound 16 KiB, header count | `Limits.maxHeaderBytes` | `Limits.maximumHeaderBytes`, `maximumHeaderCount` (431) | Covered |
| Read budget 10 s for the whole request | `readDeadline` | `headerReadTimeout` and `bodyReadTimeout`, each separate | Covered; worst case is the sum, not one budget (section 7) |
| Write deadline 30 s per write | `writeDeadline` | `streamWriteTimeout` (a stalled write ends the stream) | Covered |
| Whole-response budget (an **open item** today) | none | `maximumStreamDuration`, `maximumStreamBytes` | **Closed by the kit** |
| Connection cap 16, 503 with `Retry-After` beyond it | `maxConnections`, `refuseBusy` | `maximumConnections` (the kit's own refusal; the `Retry-After` value is the kit's, **verify**) | Covered; verify header |
| One request per connection, `Connection: close` | every response | `connections: .closeAfterResponse` | Covered |
| HTTP/1.0 and 1.1 accepted, 505 otherwise | `supportedVersions` | The kit parses 1.0 and 1.1 and answers a 1.0 client with an `HTTP/1.0` status line (corrected after implementation; this row said `HTTP/1.1`); an `HTTP/2.0` request line is 400, not 505 | Covered; the 505 is a difference (table in section 7) |
| SSE: `data: {chunk}\n\n`, then `finish_reason`, then `data: [DONE]` | `SSEWriter` | `Response.eventStream` and `EventWriter.send(data:)`; the JSON is one line, so no `data:` splitting | Covered |
| Stream ends truncated (no `[DONE]`) when generation fails | `main.swift` | A producer that throws makes the kit close the connection without the chunk terminator | Covered (`streamEnded(clean: false)`) |
| Bearer key: file mode check, env var, constant-time compare, never logged, required off loopback | `APIKey`, `BearerCredential`, `Authentication` | **Application logic.** The kit has no authentication, by design. Moves unchanged and becomes one `Middleware` | Covered (app) |
| Sequence budget (prompt tokens + completion within the KV capacity, 400 with three numbers) | `ChatRequest.parse`, `Limits.maxSequenceTokens` | Application logic in the handler, unchanged; the server's own tokenizer still counts | Covered (app) |
| `max_tokens` validation (whole number, in range, never clamped) | `ChatRequest.parse` | Unchanged, pure | Covered (app) |
| One inference at a time; waiters in arrival order; 503 + `Retry-After` at the queue deadline; a waiter whose client left is dropped | `InferenceQueue`, `SerializedHandler` | Application logic, rewritten as an actor (section 5.3). The kit supplies the cancellation signal: it cancels the handler's `Task` when the connection closes | Covered, **verify** the signal (K4) |
| Stop generating when the client leaves, including during prefill when nothing is written | `clientHasDisconnected` polled by `TokenGenerator` | Task cancellation (same signal) replaces polling the socket | Covered, **verify** (K4) |
| Refusals in the OpenAI error shape (`{"error":{"message","type"}}`) | `HTTPRefusal` | `RefusalHooks(respond:)` for the transport's refusals, `notFound` and `methodNotAllowed` for the router's | Covered |
| Origin allow-list, `Vary: Origin`, preflight answered before the key check, no wildcard | `decide` | **Not in the kit** | **K2** |
| `Host` must name the loopback listener (DNS rebinding) | `hostNamesServer` | **Not in the kit.** SwiftPushHub and FleetWatch each wrote their own | **K1** |
| Refuse exposed plaintext unless `--allow-plaintext` says a TLS proxy fronts it | `requirePlaintextAcknowledged` | The kit refuses exposed plaintext with **no way to acknowledge**; it expects TLS in the process | **K3** |
| Fail to start in the first millisecond if auth or bind is wrong, before the model loads | `main.swift` | `ServiceConfiguration` and the auth resolve are plain throwing calls, so the order is kept | Covered |
| Observer events (`requestQueued`, `queuedRequestAbandoned`, ...) used by tests instead of the clock | `HTTPServerObserver` | The gate keeps its own events; the access log and `RefusalHooks.record` cover the rest | Covered |
| Real TLS, HTTP/2, graceful shutdown, hot cert reload | not present | Provided by the kit | Gain |

**Answer: yes, with four small kit changes, one of them possibly unnecessary.** Everything else is either provided or
is application logic that moves unchanged.

### The kit changes (K1 to K4), as a 0.5.0

- **K1. A `Host` allow-list middleware** (`Middleware.allowedHosts(_:)`, 421). Two consumers already wrote it by
  hand; this is the third. Without it the DNS-rebinding defence regresses.
- **K2. An origin/CORS middleware** (`Middleware.cors(origins:)`): exact-match allow-list, no wildcard, `null` and
  `*` refused at construction, `Vary: Origin`, an `OPTIONS` preflight answered *ahead of* any later middleware so it
  needs no credential. New to the kit; this package is the first to need it, and any browser-facing service will.
- **K3. A plaintext acknowledgement.** Decision O2: either the process terminates TLS itself using the kit and
  SwiftTLSKit (then no kit change and `--allow-plaintext` goes away), or the reverse-proxy recipe stays and the
  kit gains an explicit `ServiceConfiguration(acknowledgingPlaintextBehindProxy: true)` that downgrades the refusal to
  a logged warning. I recommend the first; the second is needed only if the proxy recipe must survive.
- **K4. Prove (and if need be fix) that a client leaving cancels an idle handler.** `ConnectionHandler.channelInactive`
  cancels the in-flight handler `Task` and the stream producer, so the design relies on it. What I could not
  confirm by reading is that the channel *becomes inactive* when the peer sends FIN while the handler is idle: it is
  neither reading a body nor writing a response. Spike S1 settles it; if the kit does not deliver the signal it must.

## 4. Proposed architecture

### 4.1 Targets

The `SwiftMoE` library today is dependency-free and ships `HTTPServer` as public API. The kit pulls in NIO and
SwiftTLSKit, so the server must not stay inside the library.

```
Sources/SwiftMoE/            the inference engine and nothing HTTP (SessionStore stays: it is the chat TUI's)
Sources/SwiftMoEHTTP/        NEW library: the OpenAI-compatible service on the kit
Sources/SwiftMoEServer/      the executable: argument parsing and the demo model; depends on SwiftMoEHTTP
```

Moved unchanged into `SwiftMoEHTTP`, with their tests: `APIKey.swift` (303 lines), the pure parts of `ChatRequest` (the
body parse and sequence budget). Deleted: `HTTPServer.swift` (511), `HTTPConnection.swift` (489),
`ClientConnection.swift` (194), `HTTPRequestHead.swift` (130), `SSEWriter.swift` (137) and the transport halves of
`HTTPServerLimits.swift` (171). Rewritten: `InferenceQueue.swift` (127) as an actor, `HTTPRefusal.swift` (120) as the
error-shape mapping. Roughly 1,700 of the 2,400 lines go; about 600 remain and about 250 are new.

### 4.2 Request path

```
accessLog (outermost)
  -> allowedHosts   (K1; only on a loopback bind)
  -> cors           (K2; answers OPTIONS before auth)
  -> bearer         (app: Authentication.resolve result, 401 + WWW-Authenticate)
  -> Router: POST /v1/chat/completions   maximumBodyBytes = limits.maxBodyBytes
```

The chat handler: parse and validate the body (unchanged pure code; 400s), count the prompt with the server's
tokenizer and enforce the sequence budget (400), then `await gate.admit(deadline:)`, then return
`Response.eventStream { events in ... }`.

### 4.3 Admission and the slot (`InferenceGate`)

An `actor` holding the one slot and a FIFO of waiters. `admit(deadline:)` returns a `Slot`, throws
`InferenceGateError.timedOut` at the deadline (the handler answers 503 with `Retry-After`, from
`Limits.retryAfterSeconds`), and throws `CancellationError` if the handler's `Task` is cancelled (the client left)
and, in that case, hands the place on. The slot is released by `ResponseStream.observing { release }`, which the kit
documents as firing both when the stream ends and when the response is discarded because the connection died
between the handler returning and the stream starting. That second case is the one that would leak the slot, so it
gets its own test (S4) and a hand mutation.

Admission happens in the handler, *before* the response head is sent, so a queue timeout can still be a real `503`.

### 4.4 Generation: a blocking GPU loop behind an async stream

`TokenGenerator.generate` is synchronous and holds a thread for up to half an hour. It must not run on the
cooperative pool or on a NIO loop. Design:

- One dedicated `Thread` owns generation. The producer closure starts it, then consumes an `AsyncThrowingStream`
  of token strings and writes each as an SSE chunk.
- Cancellation flows one way: the producer's `Task` is cancelled (client left, duration limit, shutdown), a
  `withTaskCancellationHandler` flips an atomic flag, and `shouldContinue` reads it before every token, prefill
  included. This replaces `clientHasDisconnected`, which polled the socket.
- Backpressure: the stream's buffer is one element, so a slow reader slows generation exactly as a blocking write
  does today, and `streamWriteTimeout` ends a reader that stopped.
- `onToken` returns `false` once the flag is set or the stream's continuation is terminated.

### 4.5 Configuration mapping

| Today (`Limits` and flags) | Kit |
|---|---|
| `maxBodyBytes` | `Router.add(..., maximumBodyBytes:)` and `Limits.maximumBodyBytes` |
| `maxHeaderBytes` | `Limits.maximumHeaderBytes` |
| `readDeadline` 10 s | `headerReadTimeout` and `bodyReadTimeout` (each 10 s; see section 7) |
| `writeDeadline` 30 s | `streamWriteTimeout` 30 s |
| `maxConnections` 16 | `maximumConnections` 16 |
| (open item: response deadline) | `maximumStreamDuration`, see O5 |
| `queueDeadline`, `maxSequenceTokens`, `maxCompletionTokens`, `defaultCompletionTokens` | stay application limits (`InferenceLimits`) |
| `--host`, `--port`, `--api-key-file`, `--no-auth`, `--allow-origin`, `--allow-plaintext` | unchanged flags; `--allow-plaintext` per O2 |
| new | `--tls-cert`, `--tls-key` |

## 5. API surface

Public API **removed** from the `SwiftMoE` library: `HTTPServer`, `HTTPServer.Limits`, `SSEWriter`,
`HTTPServer.Request`, `BearerCredential`, `APIKey`, `HTTPServerError`. They reappear (renamed where the old name would
mislead) in `SwiftMoEHTTP`: `ChatService`, `InferenceLimits`, `APIKey`, `BearerCredential`, `ServiceError`. Pre-1.0 this
is a minor-version break: **0.2.0**. Nothing else in the workspace depends on the library (`grep` over every
`Package.swift` found only this one), so no migration shim is proposed (O1).

## 6. Dependencies

`SwiftMoEHTTP` adds `SwiftHTTPService` and `SwiftHTTPServiceNIO` (0.5.0 once K1 to K4 land, `.upToNextMinor`), which bring
`swift-nio`, `swift-nio-http2`, `swift-http-types`, `swift-nio-extras` and `SwiftTLSKit` into the package's resolution.
The `SwiftMoE` library target itself still imports none of them. A consumer of the library product resolves them anyway
(SwiftPM resolves per package). If that matters, option C below moves the server to its own package.

## 7. Behaviour that changes (decided now, tested later)

| Request | Today | After |
|---|---|---|
| `Transfer-Encoding: chunked` upload | `501` OpenAI-shaped | **Accepted within the bound** (S5): a chunked body under 64 KiB is served; over it is 413 |
| A slow client sending head then body | One 10 s budget for both | Up to 10 s for the head plus 10 s for the body |
| `Date`/`Server` headers, header order | None / fixed | Whatever the kit emits |
| Malformed target (`//`, dot segments, bad `%`) | 404 | 400 `malformed_target`, OpenAI-shaped via the refusal hook |
| HTTP/1.0 | Answered `HTTP/1.1` (recorded: the old server never answered 1.0) | **Answered `HTTP/1.0`** (the kit mirrors the client's version; an earlier draft of this row and of section 3 said `HTTP/1.1`, which the implementation showed to be wrong) |
| `Cache-Control` on the stream | `no-cache` | `no-store` |
| `stream: false` | Ignored: always SSE | Unchanged (a separate feature, not part of this move) |
| Connection beyond the cap | `503` + `Retry-After: queueDeadline` | **Closed with no response at all: no 503, no Retry-After** (spike S5; a known regression, see the table below) |

**Found by the recording (2026-10-30), to be preserved unless the owner decides otherwise:**
- The old stream has no `Content-Length` and ends by closing the connection; the kit's event stream is chunked, so `Transfer-Encoding: chunked` appears. Allowed header change; the body bytes must match.
- `Authorization` accepts a lowercase scheme (`bearer <key>`) and a key with a trailing space (200). Keep: the auth middleware moves unchanged.
- `max_tokens: null` is treated as absent; `64.0` and `1e1` are accepted as whole numbers. Keep (the parse moves unchanged).
- The busy refusal carries `Retry-After` equal to the queue deadline and no `Date`. The preflight sends `Access-Control-Allow-Methods: POST` and `Access-Control-Allow-Headers: Authorization, Content-Type` (K2's defaults match).
- A query string on the route (`?stream=true`) is ignored and the request is served. Keep.

**Differences the replay found beyond the table above (accepted 2026-10-30).** The old server's recorded exchanges
were replayed against the new one (`WireCharacterizationTests`); these nine differ and were not predicted. Each is an
`intended` entry with its reason in that test, and the replay fails on any difference that is not in it.

| # | Exchange | Old server | New server | Cause |
|---|---|---|---|---|
| 1 | ~~A connection past `maxConnections`~~ **fixed in 0.2.1** | `503` + `Retry-After: 30` | ~~**Closed with no response**~~ `503` + `Retry-After` again (plaintext; over TLS closed before the handshake) | The kit's registry dropped the connection (0.5.0). **Was a known regression against the old server**, caused by the kit and not worked around in this package; clients see a reset and have no `Retry-After` to honour |
| 2 | No `Host` header | `421` | `400 Malformed HTTP request.` | The transport requires exactly one `Host` before any middleware runs |
| 3 | ~~`HTTP/2.0` on an HTTP/1 request line~~ **fixed in 0.2.1** | `505` | ~~`400`~~ `505` again | The transport's parser |
| 4 | A body shorter than its `Content-Length`; a client that sends nothing | `400`; `408` | Connection closed, no response | The transport; the head timer starts at the first byte, and a short body closes |
| 5 | Preflight from an allowed origin for a path that is not the route | `404` | `204` (any path) | The kit's CORS layer answers every preflight itself; it also omits `Content-Length: 0` |
| 6 | ~~Status line for a body over the bound~~ **fixed in 0.2.1** | `413 Content Too Large` | ~~`413 Payload Too Large`~~ `Content Too Large` again | The transport's reason phrase |
| 7 | A body declared over the bound, with a wrong or missing key | `401` | `413` | The route bound is checked before any middleware, so the body is refused unread (the property the old order protected) |
| 8 | An `HTTP/1.0` request | Answered `HTTP/1.1` | Answered `HTTP/1.0` | The kit mirrors the client's version |
| 9 | A header limit below 1 KiB | Accepted (the old test used 512) | `ServiceError.invalidConfiguration` | `ServiceConfiguration` bounds `maximumHeaderBytes` to 1 KiB or more |

## 8. Test strategy (TDD; every assertion able to fail for the reason its test is named for)

The established recipe from the three earlier migrations, adapted to a server that has a *demo mode with a
deterministic model* (zero weights, so the output is a fixed function of the prompt):

1. **Characterize first (red/green on the old code).** Record, from the *current* server, every route's status,
   headers (minus `Date`) and body for a fixed set of requests: each refusal in the `decide` ladder (401, 403, 404,
   405, 408, 411, 413, 421, 501, 505, 400 for each body fault), the sequence-budget boundary (exactly at, one over),
   `max_tokens` edge cases, a full SSE exchange with the demo model, a queue timeout, a client leaving while queued,
   and a client leaving mid-stream. Keep it as `Fixtures/exchanges.json`; expected values are the *old server's* bytes,
   never the new code's output.
2. **Port the 114 server-level tests** (`HTTPServer{Access,Limits,Queue,SequenceBudget,Bind,Plaintext}Tests`,
   `SSEWriterTests`, `InferenceQueueTests`, `APIKeyTests`) by changing only construction. A test that cannot be ported
   without weakening its assertion is reported, not edited.
3. **Intended differences** (section 7) are a table in code, each with a reason, and the replay fails on any
   difference not in it.
4. **Spikes first** (section 10) before any production code: S1 to S4.
5. **Hand mutation** of the new code: the gate's release, the cancellation flag, the deadline comparison, the budget
   boundary, the auth middleware order, the preflight-before-auth order. Survivors become tests or the code goes.
6. **Real socket and real TLS test** with SwiftTLSKit's test certificates: SSE over HTTP/1.1 and over HTTP/2, a client
   dropping mid-stream, graceful shutdown with a stream in flight.
7. Quality gate 0/0 with no suppression; DocC on every public symbol.

## 9. Adversarial review

- **"The kit's cancellation might not fire for an idle handler."** Then queue abandonment and prefill cancellation
  regress silently. Hence K4/S1 before anything else, and the gate's tests wait on events, not on the clock.
- **"Blocking a pool thread for 30 minutes will wedge the process."** Avoided by the dedicated thread (4.4); S3 holds
  eight such generations' worth of idle waiters and asserts other routes still answer.
- **"The slot can leak."** Three exits (stream ends, producer throws, response discarded) all go through
  `observing`; S4 kills the connection between handler return and first write and asserts the next request is
  admitted.
- **"A missing middleware order check lets an unauthenticated preflight run inference."** A preflight is answered
  without reaching the handler; a test sends `OPTIONS` with a hostile `Origin`, and `POST` with a valid `Origin` and no
  key, and asserts the second is 401.
- **"Telling a model server to speak HTTP/2 adds risk for no gain."** Default `protocols: .http1`, matching today;
  HTTP/2 is one flag and can be tried on the owner's schedule (O4).
- **"A larger dependency graph for a Metal engine."** True (section 6). Alternative C removes it at the cost of a second
  package.
- **"This is churn: nothing runs `--model` yet."** The server is demo-only today, so the migration has no production
  blast radius, which is why it is cheap to do *now*, before real weights make the server something to protect.

## 10. Spikes (answer these before writing the service)

| # | Question | Pass condition |
|---|---|---|
| S1 | Does the kit cancel an idle handler when the client closes? (K4) | A handler suspended in `Task.sleep` sees `Task.isCancelled` within 1 s of the client's `close()`, over HTTP/1.1 `closeAfterResponse` and over HTTP/2; and likewise a stream producer that is not writing |
| S2 | Does `eventStream` produce the exact bytes the old `SSEWriter` did, and does an OpenAI client library parse them? | Byte-compare a recorded exchange; parse with a stock client |
| S3 | Does a dedicated `Thread` bridged by `AsyncThrowingStream` keep the other routes responsive under a long generation, and honour backpressure? | `GET` of an unknown route answers in under 100 ms during a 60 s generation; a paused reader stops token production |
| S4 | Is the slot released if the connection dies between the handler returning and the stream starting? | The next request is admitted within 1 s |
| S5 | What does the kit do past the connection cap (`Retry-After`?) and with a chunked upload? | Read both off the wire; set them to match or document the difference |

## 11. Alternatives considered

- **A. A new library target in this package (recommended).** Testable, keeps `SwiftMoE` free of NIO.
- **B. Put it all in the executable.** Smallest change, but the server logic becomes untestable except through a
  process, which is a step backwards from today's 114 tests.
- **C. A separate package, `SwiftMoEServer`.** Cleanest dependency story (the engine package stays dependency-free
  to its consumers) and the right answer if the engine gains other consumers. More ceremony now. Revisit when a second
  consumer of `SwiftMoE` appears.
- **D. Keep the hand-rolled server and only add what is missing.** Rejected by decision D4, and it leaves three
  open master-plan items permanently open.
- **E. Make the kit's streaming take a synchronous producer.** Would hide the blocking-thread problem inside the kit,
  where it would need its own thread policy. Rejected: that policy belongs to the application that owns the GPU.

## 12. Build order

1. Kit 0.5.0 on its own branch in `SwiftHTTPService`: S1 first (a test, then a fix only if red), then K1, K2, and K3 if
   O2 chooses it. Release and tag; this package pins it.
2. Here, on a branch: spikes S2 to S5; characterization recording (section 8.1) committed on the *old* code.
3. `SwiftMoEHTTP`: move `APIKey` and `ChatRequest` (tests unchanged), then `InferenceGate`, then the generation
   bridge, then the router, middleware and entry point, each red then green.
4. Point the executable at it; run the ported tests and the replay; hand mutations.
5. Delete the old server files; update `README.md` (usage, the reverse-proxy recipe per O2), `CHANGELOG.md` (0.2.0,
   with the removed API listed), `project/master_plan.md` (strike the three items the kit closes, keeping their
   reasoning), `HANDOFF.md`, a session summary.
6. Tag `v0.2.0`.

## 13. Decisions (owner, 2026-10-30)

The owner agreed with the whole proposal, so each recommendation stands: **O1** remove the public `HTTPServer` outright; **O2** TLS terminated in the process, so no K3 and `--allow-plaintext` goes; **O3** keep the truncated stream; **O4** HTTP/2 off; **O5** `maximumStreamDuration` of 2 hours; **O6** 0.2.0. The kit therefore needs K1, K2 and the K4 spike. The questions as asked:

- **O1.** Remove the library's public `HTTPServer` outright (my recommendation: nothing depends on it), or leave a
  deprecated shim for one release?
- **O2.** Terminate TLS in the process with the kit and SwiftTLSKit (recommended; `--allow-plaintext` and the
  proxy recipe go), or keep the proxy recipe and add the kit acknowledgement (K3)?
- **O3.** A generation that fails mid-stream currently ends the stream with no error event, on purpose, and the master
  plan lists adding one as open. Keep the truncation (recommended: it is the only signal every client treats as a
  failure) or send an OpenAI-style error event first?
- **O4.** HTTP/2 for the SSE route: off (matches today, recommended) or on?
- **O5.** `maximumStreamDuration`. The kit's default is one hour; 8,192 tokens at 4.4 tok/s is 31 minutes, and a
  slower configuration will exceed an hour. I propose 2 hours as the bound that closes the open "response deadline"
  item, stated in the README as the longest completion the server will deliver.
- **O6.** Version: 0.2.0 (the API break), or hold at 0.1.x by keeping the old type names as aliases?

## 14. Future directions

- `stream: false` (a single JSON completion) and `messages` beyond the last one belong to the real `--model` work,
  not this move.
- A kit-level `ConcurrencyLimit` middleware (in-flight bound, queue with deadline, 503 and `Retry-After`) is what
  `InferenceGate` is. It stays here until a second service needs the same, then it is promoted.
- A per-peer connection cap, so unauthenticated clients cannot hold the 16 slots (the master plan's open item), is a
  kit feature, not this package's.
