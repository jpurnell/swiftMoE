# Changelog

All notable changes to SwiftMoE are documented in this file.

## [Unreleased]

## [0.2.1] - 2026-10-30

On SwiftHTTPService 0.6.0, which fixes the regression 0.2.0 recorded. No API change.

### Fixed
- ~~**Known regression (0.2.0): a connection past `maxConnections` is closed with no response**, with no
  `503` and no `Retry-After`.~~ Kit 0.6.0 answers a plaintext connection over the cap with `503` and
  `Retry-After`; `ChatService` gives it the old body (`The server is at its connection limit.`,
  `server_error`) through `TransportRefusal.busy`, and sets `Limits.busyRetryAfter` to the queue
  deadline in whole seconds, rounded up, so `Retry-After` is the old value (30 with the default
  limits). Over TLS the connection is still closed before the handshake with nothing written: nothing
  can be said to a client that has not completed it.
- An `HTTP/2.0` request line is `505`, with the old body (`Only HTTP/1.0 and HTTP/1.1 are supported.`),
  as it was before 0.2.0.
- A connection that sends nothing is answered `408` at the read deadline (`idleTimeout` is set to
  `readDeadline`), as before 0.2.0.
- The `413` status line reads `Content Too Large` again (kit 0.6.0).

### Changed
- `ServiceConfiguration(connections:)` is `connectionReuse:` (the old name is a deprecated shim for one
  kit release). The dependency is `.upToNextMinor(from: "0.6.0")`.
- `TransportRefusal` gained `.busy` and `.versionNotSupported`; `ChatService` maps both.

### Differences from 0.1.0 that remain (accepted, in `WireCharacterizationTests` as `intended`)
Six, down from nine, each rechecked against the old fixture:
- A request with no `Host` is `400`, not `421` (the transport refuses it before any middleware).
- A body that ends before its `Content-Length` closes the connection with no `400`.
- A preflight from an allowed origin is answered `204` for any path (was `404` for a path other than
  the route).
- The `204` preflight carries no `Content-Length: 0`.
- An HTTP/1.0 request is answered `HTTP/1.0` (the old server answered `HTTP/1.1`).
- A header limit below 1 KiB is refused at start-up (the old server's test used 512 bytes).

Plus the six intended changes of 0.2.0 (malformed targets 400, chunked accepted, `Cache-Control:
no-store`, chunked framing, `Date`, head and body each get the read deadline).

### Verification
255 tests in 40 suites. Hand mutation of the new pieces, all killed: X1 `busyRetryAfter` not wired
(kit default 1 s); X2 `busyRetryAfter` a constant; X3 busy mapped to the internal-error body; X4 505
mapped to the malformed body; X5 idle deadline left at the kit default; X6 busy message changed.

## [0.2.0] - 2026-10-30

The HTTP server moves onto [SwiftHTTPService](https://github.com/jpurnell/SwiftHTTPService) 0.5.0.
Nothing is left of the raw-socket listener: about 1,880 lines of it are deleted, and the part
of the service that is particular to a model server (the credential, the sequence budget, the
queue for the one model, the bridge to the generator thread) is rewritten around the kit's
`Router`, middleware and event streams. A minor version, because the public API of the library
changes (below).

### Removed (breaking)
Public API of the `SwiftMoE` library, all of it the old server's, which now lives in the new
`SwiftMoEHTTP` library (`import SwiftMoEHTTP`):

| Removed from `SwiftMoE` | Now |
|---|---|
| `HTTPServer` (+ `Request`, `Tokenizer`, `BoundAddress`, `openListener()`, `start()`, `stop()`) | `ChatService` (+ `ChatCompletionRequest`, `Tokenizer`, `Generate`, `start()`, `run()`, `shutdown()`) |
| `HTTPServer.Limits` | `InferenceLimits` |
| `HTTPServer.Authentication` | `Authentication` |
| `HTTPServerError` | `ServiceError` |
| `SSEWriter` and its handler contract `(Request, SSEWriter) -> Void` | `ChatService.Generate`: `(request, shouldContinue, emit) throws -> Void` |
| `APIKey`, `BearerCredential` | the same names, moved |
| `Limits.refusalDrainDeadline` | none: the transport owns connection teardown |
| `--allow-plaintext`, `ServiceError.plaintextNotAcknowledged`, `HTTPServer.requirePlaintextAcknowledged` | none: see TLS below |

`SwiftMoE` imports no networking code; `SwiftMoEHTTP` depends on `SwiftHTTPService` and
`SwiftHTTPServiceNIO`, which bring `swift-nio`, `swift-nio-http2`, `swift-http-types`,
`swift-nio-extras` and `SwiftTLSKit` into the package's resolution. The chat client now links
`SwiftMoEHTTP` for `APIKey`. `SwiftMoEHTTP` is a new library product.

### Added
- **TLS in the process.** `--tls-cert <chain.pem>` and `--tls-key <key.pem>`. Off loopback the
  server refuses to start without them; `SIGHUP` reloads the identity without a restart. The
  reverse-proxy recipe in the README is superseded and `--allow-plaintext` is gone (decision O2).
- **A bound on how long a response may take**: `InferenceLimits.maxStreamDuration`, two hours
  (decision O5). This closes the master plan's open "no bound on how long one response may take
  to deliver". A reader that stops reading ends its stream after `writeDeadline` (30 s).
- **Graceful shutdown**: SIGINT and SIGTERM stop accepting, let a stream in flight finish for
  `shutdownGrace` (10 s), then close.
- **`InferenceGate`**, an actor in place of `InferenceQueue`: first come first served, a
  deadline, a waiter whose client leaves (its task is cancelled) is dropped, and a slot handed to
  a waiter that was cancelled in the same instant is passed on. The slot is released once, through
  `ResponseStream.observing`, which also fires when the response is discarded before its stream
  starts.
- **`GenerationBridge`**: the synchronous `TokenGenerator.generate` runs on a thread of its own,
  tokens reach the stream through an `AsyncThrowingStream`, one token waits at most (a reader that
  stops reading stops the generation), and `shouldContinue` reads a cancellation flag before every
  token, prompt tokens included. The slot is not given up until that thread has left the generator.
- `ServiceError.invalidConfiguration`, `.exposedPlaintext`, `.tlsIdentity`, `.bindFailed`.

### Behaviour that changes
Recorded as the old server's bytes in `Tests/SwiftMoETests/Fixtures/exchanges.json` (85 exchanges,
from tag v0.1.0) and replayed against the new server by `WireCharacterizationTests`.

Intended (decided in the design proposal, section 7):
- A malformed target (`//v1/chat/completions`, dot segments, a bad `%` escape) is `400
  Malformed request target.`, not `404`.
- A chunked upload within the body bound is accepted; the old `501` was a limit of the
  hand-rolled reader.
- A stream carries `Cache-Control: no-store` (was `no-cache`), is chunk-encoded rather than
  ended by closing, and the responses carry `Date`.
- The head and the body each get the read deadline (10 s), not 10 s between them.
- HTTP/2 is off (decision O4); HTTP/1.1 only, one request per connection.
- A generation that fails mid-stream still ends the stream with no finish chunk and no `[DONE]`
  (decision O3).

### Changed
Found by replaying the old server's recorded exchanges and accepted as intended on 2026-10-30 (the
table is in `WireCharacterizationTests`, which fails on any difference not in it, and in section 7
of the proposal). All are the kit's behaviour at 0.5.0; none is worked around in this package.

- ~~**Known regression: a connection past `maxConnections` is closed with no response.**~~ *(fixed in 0.2.1 on kit 0.6.0)* There is no
  `503` and no `Retry-After` any more (spike S5). A client sees a reset and has nothing to honour.
- A request with no `Host` is `400`, not `421` (the transport refuses it before any middleware).
- ~~An `HTTP/2.0` request line is `400`, not `505`.~~ *(fixed in 0.2.1)*
- A body that ends before its `Content-Length` closes the connection with no `400`; a client that
  connects and sends nothing is closed at the idle deadline with no `408` *(the 408 is back in 0.2.1)*.
- A preflight from an allowed origin is answered `204` for any path (was `404` for a path other than
  the route), and carries no `Content-Length: 0`.
- ~~The `413` status line reads `Payload Too Large` (was `Content Too Large`).~~ *(fixed in 0.2.1)*
- A body declared over the bound is `413` before the credential is checked (was `401`); either way it
  is refused unread.
- An HTTP/1.0 request is answered `HTTP/1.0` (the old server answered `HTTP/1.1`).
- A header limit below 1 KiB is refused at start-up (`ServiceError.invalidConfiguration`).

The ported server tests keep their assertions except where one of the above forced a change; each
such test says `DIFFERENCE` in a comment above it. Tests that could not be ported: the
`HTTPRequestHead` parser tests (the parser is gone; the kit tests its own) and the
`SSEWriter` socket tests (replaced by `ChatChunkTests` and `GenerationBridgeTests`).

### Verification
- Spikes S2 to S5 of the proposal are tests (`SpikeTests`): the event-stream bytes equal the
  recorded old bytes; other routes answer within 100 ms during a generation with eight requests
  waiting; a reader that never reads stops the generation; a response discarded before its stream
  starts releases the slot, and clients that vanish right after sending leak none.
- Hand mutation of the new code (each mutant failed at least one test; the release in M1 and M9 is
  reported after the gate has taken the slot back, which the S4 test needs):
  - M1 gate release is a no-op: killed (hang, 7 tests failed)
  - M2 `shouldContinue` ignores the cancellation flag: killed (4)
  - M3 `emit` ignores cancellation: killed (1)
  - M4 queue deadline never expires: killed (the queue tests hang until the harness times out)
  - M5 queue deadline already expired: killed (5)
  - M6 budget boundary `<=` becomes `<`: killed (5)
  - M7 bearer layer ahead of the CORS layer (a preflight would need a key): killed (8)
  - M8 `Host` check after the CORS layer: killed (1)
  - M9 slot not released through `observing`: killed (hang, 4)
  - M10 producer returns before the generator thread has left: killed (2)
  - M11 finish chunk omitted: killed (32)
  - M12 gate serves the newest waiter first: killed (3)
  - M13 a waiter cancelled as it is handed the slot keeps it: killed (1)
  - M14 backpressure credit not returned after a write: killed (5)

  No mutant survived. The first run of M4 did not compile (an unused `self`); it was re-made and run.

## [0.1.0] - 2026-10-09

The first tagged version. A `0.x` minor, because the engine has not yet been validated against
real weights and its API is not settled.

### Security — breaking: bounded work, and no silent truncation
The server bounded what a client could send and how long it could take sending it. It did not
bound what the request then cost: a prompt could be eight times the context, a request could
wait for the model indefinitely, a generation ran on after its client had gone, and off
loopback the key crossed the network in clear text behind nothing but a log line.

- **Sequence budget.** The prompt is counted by the server's tokenizer, and prompt tokens plus
  requested completion tokens must fit in `HTTPServer.Limits.maxSequenceTokens` — 8192,
  `TokenGenerator.defaultMaxSequenceLength`; `swift-moe-server` sets it from its generator.
  Over it is `400 Bad Request`: `Prompt (N tokens) plus completion (M tokens) is T tokens; the
  limit for a sequence is L. Shorten the prompt or lower max_tokens.` The completion is not
  shortened to fit. Before this the only bound on a prompt was the 64 KiB body, which the
  placeholder tokenizer makes 65,536 tokens.
  - `HTTPServer.init` takes `tokenizer:` (`HTTPServer.Tokenizer`, `@Sendable (String) -> [Int]`),
    with no default: the budget is only as true as the count it is given.
  - The handler is `(HTTPServer.Request, SSEWriter) -> Void`. `Request` carries `prompt`,
    `promptTokens` — the tokens that were counted — and `maxTokens`.
  - `Limits.validate` refuses `maxCompletionTokens` above `maxSequenceTokens`, with
    `HTTPServerError.limitAboveLimit(name:ceiling:)`; a default token count above the ceiling
    reports the same way where it used to say "must be greater than zero".
- **The inference layer no longer drops context.** `KVCache.append` returned without a word
  once the cache was full, so every later token attended to a history that had stopped
  growing. Both overloads now throw
  `FlashMoEError.sequenceCapacityExceeded(capacity:required:)` and record nothing;
  `FullAttention.forward`, `LayerPipeline.forward` and `TokenGenerator.generate` are `throws`.
  `generate` checks `promptTokens.count + maxTokens` against `maxSequenceLength` before it
  processes a token.
- **One caller's prompt no longer reaches the next caller's answer.** `TokenGenerator.generate`
  restarted at position 0 on every call but kept the previous call's KV entries and
  linear-attention state, and the server never reset them: a second request attended to the
  first one's keys and values. `generate` now resets the generator first — each call is a new
  sequence.
- **Queue.** A validated request that finds the model busy waits its turn, first come first
  served, for at most `Limits.queueDeadline` (30 s). After that it is `503 Service Unavailable`
  with `Retry-After` — the deadline in whole seconds — and `The server is busy with another
  request and could not start this one in time. Try again later.` It used to wait on a lock,
  with no deadline, holding one of the 16 connections. The queue's depth is bounded by
  `maxConnections`: every waiter is a connection. The connection-limit `503` now carries
  `Retry-After` too.
- **A client that leaves stops costing.** A queued request whose client has hung up gives up
  its place and its connection within a tenth of a second and is never run. A running
  generation stops at the next token — prompt tokens included, which is new: prefill writes
  nothing, so it could not be interrupted. `SSEWriter.clientHasDisconnected` is the check;
  `TokenGenerator.generate(… shouldContinue: …)` is where it is asked.
- **Plain text has to be asked for.** The server does not speak TLS. `openListener()` now throws
  `HTTPServerError.plaintextNotAcknowledged(host:)` for any address outside `127.0.0.0/8`
  unless the server was created with `allowPlaintext: true` — `swift-moe-server
  --allow-plaintext`, which means "a TLS-terminating proxy on the same trust boundary fronts
  this port" and encrypts nothing. Nothing is bound, and the model is not loaded, when it is
  refused. The README has the proxy recipe.

### Fixed — server
- An SSE event carrying a control character was not JSON. `SSEWriter` escaped `"`, `\`, and
  three of the thirty-two characters below U+0020; the demo server, which maps token ids onto
  the first 128 code points, wrote the other twenty-nine raw. All are escaped now.
- A completed stream could be lost to a client that had sent anything after its request: the
  socket was closed over the unread bytes, which resets the connection. A stream now ends the
  way a refusal does — write side shut, the remainder discarded for at most
  `refusalDrainDeadline`.
- `TokenGenerator.generate` with a negative `maxTokens` trapped on `0..<maxTokens`; it generates
  nothing.

### Migration — bounded work
- **Starting the server for other machines** needs the flag as well as the key:
  `swift-moe-server --host 0.0.0.0 --api-key-file <path> --allow-plaintext`, and a
  TLS-terminating proxy in front of it. Better: leave it on `127.0.0.1` and run the proxy on
  the same machine.
- **Clients must keep their side of the connection open** until they have the answer. A client
  that shuts its sending side after the request — `nc -N`, `shutdown(SHUT_WR)` — is
  indistinguishable from one that hung up, and its generation is stopped. `curl`, URLSession
  and the OpenAI SDKs do not do this.
- **Requests that relied on leniency**: a prompt that does not fit beside its `max_tokens` is
  `400`. `max_tokens: 8192` now only fits beside an empty prompt.
- **Clients should honour `Retry-After`** on `503`.
- **Library callers**:
  ```swift
  let server = HTTPServer(
      port: 8080,
      authentication: .bearer(BearerCredential(key: key)),
      tokenizer: { prompt in tokenizer.encode(prompt) }
  ) { request, writer in
      writer.sendHeaders()
      do {
          try generator.generate(
              promptTokens: request.promptTokens, maxTokens: request.maxTokens,
              weightFile: weights, expertFDs: experts, layerWeights: layers,
              shouldContinue: { !writer.clientHasDisconnected },
              onToken: { writer.sendDelta(token: tokenizer.decode($0)) })
          writer.sendDone()
      } catch {
          // No [DONE]: a stream that just stops is how the client learns it is incomplete.
      }
  }
  ```
  `KVCache.append`, `FullAttention.forward`, `LayerPipeline.forward` and
  `TokenGenerator.generate` need `try`. A `switch` over `FlashMoEError` or `HTTPServerError`
  without a `default` needs the new cases. `FlashMoEError` is `Equatable`.

### Security — breaking: the server requires a credential
The HTTP server ran inference for anything that could connect, told every web page it could
read the answer, and let one client decide how much work it did and for how long.

- **Credential.** Every request needs `Authorization: Bearer <key>`. Anything else is
  `401 Unauthorized` with `WWW-Authenticate: Bearer`, decided before the body is read. The
  key is at least 32 printable characters, read from the `SWIFT_MOE_API_KEY` variable or from
  `--api-key-file <path>` (refused if group or other hold any permission on it), never from
  the command line and never logged. The server keeps the key's SHA-256 digest and compares
  digests without an early exit.
  - Without a key, `swift-moe-server` does not start. `--no-auth` starts it without one **on a
    loopback address only**; on any other address a key is required and `--no-auth` is refused.
  - `HTTPServer.init(host:port:authentication:allowedOrigins:limits:allowPlaintext:tokenizer:handler:)` —
    `authentication` has no default. `HTTPServer.Authentication` is `.bearer(BearerCredential)`
    or `.unauthenticatedLoopback`; `openListener()` throws
    `HTTPServerError.credentialRequired(host:)` for the latter off loopback and binds nothing.
  - New: `APIKey`, `BearerCredential`, `HTTPServer.Authentication.resolve(host:keyFile:environment:noAuth:)`,
    `HTTPServerError`.
- **CORS.** `Access-Control-Allow-Origin: *` is gone from preflight and from the stream. By
  default no CORS header is sent at all. `--allow-origin <origin>` (repeatable; `allowedOrigins:`)
  names origins that are echoed exactly, with `Vary: Origin`; preflight is answered only for
  those. A request carrying any other `Origin` is refused with `403` — without that, a page
  could still *start* inference on an unauthenticated loopback server with a form post.
- **Host.** On a loopback bind, `Host` must be the bound address or `localhost`, with the bound
  port; otherwise `421 Misdirected Request`. This is the DNS-rebinding defence.
- **Token ceiling.** `max_tokens` / `max_completion_tokens` must be a whole number from 1 to
  `HTTPServer.Limits.maxCompletionTokens` — 8192, `TokenGenerator.defaultMaxSequenceLength`,
  the positions the KV caches are allocated for. Zero, negative, fractional, non-numeric and
  over-limit values are `400` with `<field> must be a whole number from 1 to 8192.` Nothing is
  clamped: a caller who asked for more than it can have is told, not short-changed.
- **Size and time.** Request head ≤ 16 KiB (`431`); body ≤ 64 KiB (`413`), refused from
  `Content-Length` before any of it is read; the whole request must arrive within 10 s (`408`),
  enforced with `poll(2)` as a budget rather than a gap, so one byte every nine seconds does not
  qualify. Writes time out after 30 s. A POST without `Content-Length` is `411`; chunked is `501`.
  All configurable through `HTTPServer.Limits`.
- **A refusal is delivered.** After any refusal the server shuts its write side and discards the
  rest of the upload — unbuffered, for at most 2 s — before closing. Closing at once turns the
  client's remaining upload into a reset, and the client never sees the status.
- **Concurrency.** Each connection is read on its own thread (at most 16; the next is `503`), so
  a silent client no longer stops the server. The handler still runs one request at a time;
  see "Queue" above for what the others do meanwhile.
- **Routing.** The request line is parsed into method, target and version and compared exactly:
  `POST /v1/chat/completions` (query string allowed) is the route, another method is `405` with
  `Allow: POST`, another path is `404`. The old test — "starts with `POST` and contains the
  path anywhere in the headers" — routed `POST /other` with a matching `Referer`.
- **Client.** `swift-moe-chat` sends the key (`--api-key-file`, or `SWIFT_MOE_API_KEY`), prints a
  refusal instead of nothing, and logs the session id as `.private` rather than `.public`.

### Migration — callers of the server
- **Every client must send the key.** Generate one and give it to both sides:
  ```bash
  umask 077 && openssl rand -hex 32 > ~/.swift-moe-key
  swift run swift-moe-server --demo --api-key-file ~/.swift-moe-key
  curl -N http://127.0.0.1:8080/v1/chat/completions \
    -H "Authorization: Bearer $(cat ~/.swift-moe-key)" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'
  ```
  OpenAI SDKs send this header already: set their `api_key` to the key and `base_url` to
  `http://127.0.0.1:8080/v1`.
- **To keep the old behaviour on your own machine**, start with `--no-auth`. It is refused for
  any `--host` that is not loopback; there is no way to serve other machines without a key
  (and, since the entry above, without `--allow-plaintext`).
- **Browser front-ends** get no CORS headers until their origin is named:
  `--allow-origin http://localhost:3000`. The origin is compared byte for byte — scheme, host
  and port, no trailing slash.
- **Reverse proxies** in front of a loopback bind must send the upstream authority as `Host`
  (`127.0.0.1:8080`). nginx's `proxy_pass` does by default; Caddy needs
  `header_up Host {upstream_hostport}`.
- **Requests that relied on leniency** now fail loudly: `max_tokens` above 8192 or below 1
  (`400`), a body over 64 KiB (`413` — it was previously ignored and the request run with an
  empty prompt), a body that is not a JSON object (`400` — previously an empty prompt and 100
  tokens), a POST with no `Content-Length` (`411`).
- **Library callers**: `HTTPServer(port:handler:)` and `HTTPServer(host:port:handler:)` no
  longer compile. Add `authentication:`:
  ```swift
  let key = try APIKey(contentsOf: keyFileURL)
  let server = HTTPServer(port: 8080, authentication: .bearer(BearerCredential(key: key)),
                          tokenizer: tokenize) { request, writer in … }
  ```
  (`tokenizer:` and the two-argument handler are from "Migration — bounded work" above.)
  The handler is now called on a background thread, still one call at a time.
  `HTTPServer` is `Sendable`.

### Security — breaking (earlier in this cycle)
- **The HTTP server bound every interface and said it was on `localhost`.** `HTTPServer` bound
  `INADDR_ANY` and logged `http://localhost:<port>/…`. The server has no authentication, so
  any machine that could route to the port could run inference. It now binds `127.0.0.1`
  unless told otherwise, and the log line is the address read back from the socket with
  `getsockname`, with a warning when that address is not loopback.
  - `HTTPServer.init(host:port:handler:)` — new `host` parameter, default
    `HTTPServer.loopbackHost` (`"127.0.0.1"`). It takes an IPv4 literal; a host name is
    refused with `FlashMoEError.invalidBindAddress(host:)` rather than resolved.
  - `HTTPServer.openListener()` binds and returns an `HTTPServer.BoundAddress`;
    `HTTPServer.boundAddress` reads it back later. Port `0` reports the port the kernel chose.
  - `HTTPServer.ipv4Address(_:)`, `HTTPServer.isLoopback(_:)`.
  - `swift-moe-server --host <ipv4>`.
- **A new session's id was a UUID.** `SessionStore(sessionID: nil)` used `UUID().uuidString`.
  A UUID is built to be unique, not unguessable. The id is now 32 bytes from a
  `RandomNumberGenerator`, as 64 lowercase hex digits (`SessionStore.makeSessionID(using:)`),
  and the chat client passes the system generator. Session files already on disk keep their
  names and still resume by id.

### Migration
- **Anyone who reached the server from another machine**: it no longer answers there. Pass
  the address to publish on — `swift-moe-server --host 0.0.0.0` — together with a key and
  `--allow-plaintext` (see above; when this entry was written the server was unauthenticated
  and the flag did not exist).
- `SessionStore()` / `SessionStore(sessionID:)` no longer compile. Name the generator:
  ```swift
  var entropy = SystemRandomNumberGenerator()
  let store = SessionStore(sessionID: resumedID, using: &entropy)
  ```
- A `switch` over `FlashMoEError` without a `default` needs the new
  `.invalidBindAddress(host:)` case.

### Fixed
- Six containment checks — the weight manifest, the session store, and the chat and server
  entry points — used `path.hasPrefix(root.path)`. `…/sessions-other/x.jsonl` begins with
  `…/sessions`, so a session id of `../sessions-other/x` was written outside the sessions
  directory (the id comes from the chat CLI's own configuration). All six use
  `PathContainment.isContained(_:in:)`, which compares whole components after resolving `..`
  and symbolic links; tested for a sibling, a climb out, and a link out.

### Added
- `MTLCommandBuffer.waitUntilCompletedChecked(_:)` — traps on a failed dispatch instead
  of letting callers read a stale buffer as if it were a result
- `MTLComputeCommandEncoder.dispatchExactly(threadCount:threadsPerThreadgroup:device:)` —
  dispatches exactly the threads an operation needs, with a rounded fallback for devices
  without non-uniform threadgroup support
- `TestPaths` — resolves test paths against an explicit root and rejects any that escape it
- `DequantMatvecV3Tests` — numeric GPU-vs-CPU comparison for `dequant_matvec_4bit_v3`
  across row counts that do and do not fill the final threadgroup
- Swift package with complete inference engine for Mixture-of-Experts models
- Metal compute shaders for 4-bit/2-bit dequantized matvec, RMS norm, SwiGLU, attention
- OpenAI-compatible HTTP server with SSE streaming
- Interactive chat client with tool calling support
- BPE tokenizer (pure Swift, no Python dependency)
- 226 unit and integration tests with synthetic fixtures

### Changed
- Replaced deprecated CBLAS calls with vDSP equivalents
- FMA-optimized dequant kernel (+12% throughput)
- `dequant_matvec_4bit_v3` strides its cooperative input load by the actual threadgroup
  size rather than a hardcoded 256, so a partially populated final threadgroup still
  fills `x_shared` completely
- Elementwise GPU dispatches (SwiGLU, residual add, RMS norm apply, MoE combine,
  conv1d step) and the v3 matvec now dispatch exact thread counts instead of rounding
  the grid up to whole threadgroups
- `IOReadStats.throughputBytesPerSec` is now `Double?`: `nil` when `totalMs` is not a
  positive, finite duration. It used to answer `0`, which reads as a measured rate
- The `SwiftMoE` target no longer excludes `SwiftMoE.docc`, so DocC receives the
  catalogue and its articles and symbol links are actually checked
- Tests resolve repo paths from `#filePath` rather than the process working directory,
  and use URL-based `FileManager` APIs throughout

### Fixed
- Three `Int(_:)` conversions of floating-point values that nothing showed to be
  representable — `Int(.nan)` and `Int(.infinity)` trap. `ModelConfig.rotaryDim` now
  returns a documented 0 ("rotate nothing") when `partialRotary` is not a fraction in
  `0...1`; `LayerTiming.formatSummary()` pads on the rendered integer part, so a
  non-finite accumulator prints as `nan`/`inf`; the chat client's timeout log formats
  the deadline as a `Double` instead of converting it
- Eleven GPU tests opened with
  `guard let shaderURL = ShaderLibraryTests.shaderURL else { return }`, so a checkout
  without `metal_infer/shaders.metal` got eleven green tests that compiled no shader,
  built no `MetalContext`, and asserted nothing — across `ShaderLibrary`,
  `MetalContext`, `BatchMatvec`, `DequantMatvecV3`, `TokenGenerator` and `Integration`.
  The condition is now `ShaderLibraryTests.shadersAvailable`, consumed by an
  `.enabled(if:)` trait so the framework records a skip; inside the body the URL is
  unwrapped with `try #require`, because once the trait says the file is there a nil is
  a defect and not an absence. `MetalContextTests.setWeights` had the same shape after
  `posix_memalign` — `guard let aligned = ptr else { return }` skipped the `setWeights`
  call the test is named for — and now checks the errno and requires the pointer
- `compute_decay_beta` indexed six buffers by thread id with no bound and no element
  count to bound against; it now takes `num_v_heads` and guards on it
- Command buffers were read after `waitUntilCompleted()` with no check of `status` or
  `error`, so a failed dispatch was indistinguishable from a successful one that
  computed different numbers (`DeferredExpertState`, `LayerPipeline`, `CMD2Encoder`)
- The chat client waited on `DispatchSemaphore` with no deadline, so an unresponsive
  server blocked it forever; the wait is now bounded and cancels the request
- Conv1d state shift in linear attention

[Unreleased]: https://github.com/jpurnell/swiftMoE/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/jpurnell/swiftMoE/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/jpurnell/swiftMoE/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jpurnell/swiftMoE/releases/tag/v0.1.0
