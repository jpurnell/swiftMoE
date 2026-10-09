# ``SwiftMoE``

A Metal-backed inference engine for mixture-of-experts transformers, decomposed
from a single 1,500-line Objective-C function into documented pipeline phases.

## Overview

SwiftMoE runs MoE decode on Apple GPUs by overlapping three command buffers per
transformer layer. The expensive part of MoE inference is not arithmetic — it is
getting the selected experts off the SSD and onto the GPU before the GPU runs dry.
The pipeline is built around hiding that latency:

- **CMD1** — attention projections, submitted immediately when the previous
  layer's combine ran on the GPU.
- **CMD2** — fused output projection, residual, norm, routing and shared expert,
  in a single command buffer of 8–12 encoders.
- **CMD3** — expert forwards and combine, committed without waiting so the GPU
  continues while the CPU reads the next layer's experts.

The per-layer budget is 2.9 ms, which is why resource ownership is expressed with
`~Copyable` types rather than classes: there is no room for reference-counting
traffic in the hot path. That decision and its consequences are recorded in
`project/decisions/architecture_decisions.md`.

## Serving

``HTTPServer`` exposes `POST /v1/chat/completions`. Reaching its handler means running
inference, so that is what it guards:

- **Credential.** A server is created with an ``HTTPServer/Authentication`` and there is no
  default. ``HTTPServer/Authentication/bearer(_:)`` requires `Authorization: Bearer <key>` and
  answers `401` otherwise, before reading the body. An ``APIKey`` comes from the
  `SWIFT_MOE_API_KEY` variable or an owner-only file; the server keeps a ``BearerCredential`` —
  the key's SHA-256 digest — and compares digests in constant time.
  ``HTTPServer/Authentication/unauthenticatedLoopback`` checks nothing, and
  ``HTTPServer/openListener()`` refuses it for any address outside `127.0.0.0/8`.
- **Origin and Host.** No CORS header is sent unless the request's `Origin` is on the server's
  allowlist, in which case that origin is echoed with `Vary: Origin`; any other `Origin` is
  refused. On a loopback bind the `Host` header must name the bound address or `localhost`.
- **Limits.** ``HTTPServer/Limits`` caps tokens per request (8192,
  ``TokenGenerator/defaultMaxSequenceLength``), header and body size, and the time a client may
  take. A request over a limit is refused with a status and a sentence; nothing is clamped.
- **Sequence.** The server is given an ``HTTPServer/Tokenizer`` and counts the prompt with it.
  Prompt tokens plus requested completion tokens must fit in
  ``HTTPServer/Limits/maxSequenceTokens``, or the request is refused with a `400` that gives
  the numbers. The handler receives the counted tokens in an ``HTTPServer/Request``. Beneath
  that, ``KVCache`` throws ``FlashMoEError/sequenceCapacityExceeded(capacity:required:)``
  rather than record nothing once it is full, so a sequence can be refused but never quietly
  truncated.
- **Transport.** The server speaks plain HTTP. ``HTTPServer/openListener()`` refuses any
  address outside `127.0.0.0/8` unless the server was created with `allowPlaintext: true`,
  which states that a TLS-terminating proxy fronts the port.

The listener binds `127.0.0.1` unless a caller names another address, and
``HTTPServer/openListener()`` returns the address the socket actually holds. Connections are
read concurrently; the handler runs one request at a time. A request that arrives meanwhile
waits its turn for at most ``HTTPServer/Limits/queueDeadline`` and is then answered `503` with
`Retry-After`. A client that disconnects gives up its place, and
``SSEWriter/clientHasDisconnected`` lets a handler stop generating for a client that has gone.
``HTTPServerError`` describes every reason a server refuses to start.

``SessionStore`` names conversation files by an id drawn from a
`RandomNumberGenerator` — 32 bytes, hex — rather than a UUID.

## Topics

### Pipeline

- ``LayerPipeline``
- ``MetalContext``

### Configuration

- ``ModelConfig``

### Serving

- ``HTTPServer``
- ``APIKey``
- ``BearerCredential``
- ``HTTPServerError``
- ``SSEWriter``
- ``SessionStore``
