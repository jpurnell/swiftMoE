# ``SwiftMoEHTTP``

The OpenAI-compatible chat-completions service for the SwiftMoE engine, built on SwiftHTTPService.

## Overview

``ChatService`` exposes `POST /v1/chat/completions` and streams the completion as Server-Sent
Events. Reaching its generator means running inference, so that is what it guards:

- **Credential.** A service is created with an ``Authentication`` and there is no default.
  ``Authentication/bearer(_:)`` requires `Authorization: Bearer <key>` and answers `401` otherwise,
  before the request is looked at. An ``APIKey`` comes from the `SWIFT_MOE_API_KEY` variable or an
  owner-only file; the service keeps a ``BearerCredential`` (the key's SHA-256 digest) and compares
  digests in constant time. ``Authentication/unauthenticatedLoopback`` checks nothing, and
  ``ChatService/start()`` refuses it for any address outside `127.0.0.0/8` and `::1`.
- **Origin and Host.** No CORS header is sent unless the request's `Origin` is on the allow-list, in
  which case that origin is echoed with `Vary: Origin`; any other `Origin` is refused, and a
  preflight from a listed origin is answered before the credential is checked. On a loopback bind the
  `Host` header must name the bound address or `localhost`.
- **Limits.** ``InferenceLimits`` caps tokens per request (8192,
  `TokenGenerator.defaultMaxSequenceLength`), header and body size, and the time a client
  may take. A request over a limit is refused with a status and a sentence; nothing is clamped.
- **Sequence.** The service is given a ``ChatService/Tokenizer`` and counts the prompt with it.
  Prompt tokens plus requested completion tokens must fit in ``InferenceLimits/maxSequenceTokens``,
  or the request is refused with a `400` that gives the numbers.
- **Transport.** HTTP/1.1, one request per connection. Off loopback the service terminates TLS
  itself and refuses to start without a certificate and key. A stream ends after
  ``InferenceLimits/maxStreamDuration`` (two hours by default).

One model runs one request at a time. A request that arrives meanwhile waits its turn, in order, for
at most ``InferenceLimits/queueDeadline`` and is then answered `503` with `Retry-After`. A client
that disconnects gives up its place, and a generation whose client leaves is told to stop through
the `shouldContinue` closure its ``ChatService/Generate`` receives.

## Topics

### Serving

- ``ChatService``
- ``ChatCompletionRequest``
- ``InferenceLimits``

### Credentials

- ``Authentication``
- ``APIKey``
- ``BearerCredential``
- ``ServiceError``
