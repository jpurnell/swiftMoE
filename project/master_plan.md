# SwiftMoE Master Plan

**Purpose:** Source of truth for the SwiftMoE inference engine project.

---

## Project Overview

### Mission
A Swift inference engine for Mixture-of-Experts language models on Apple Silicon. Streams 200GB+ models from NVMe SSD through a custom Metal compute pipeline using ~6GB resident memory. Supports any MoE architecture via runtime-configurable `ModelConfig`.

### Origin
Based on the [Flash-MoE](https://github.com/danveloper/flash-moe) inference engine, which demonstrated running a 397B parameter model at 5.74 tok/s on consumer hardware. The original was written in Objective-C/C. SwiftMoE is a complete rewrite in Swift with runtime configurability and comprehensive test coverage.

### Key Differentiators
- **Model-agnostic**: Runtime `ModelConfig` with presets — no hardcoded architecture constants
- **Fully tested**: 255 tests with tiny synthetic model fixtures (no 210GB download needed)
- **All GPU paths**: 7 GPU optimization paths with automatic CPU fallback
- **Pure Swift**: No C dependencies in the main library, and no networking code in it either
- **OpenAI-compatible**: `/v1/chat/completions` HTTP service (`SwiftMoEHTTP`, on SwiftHTTPService) with SSE
  streaming, bearer-key authentication, TLS in the process, and a bound on what one request can cost
  (sequence budget, queue deadline, cancellation on disconnect, a two-hour bound on a response)

---

## Architecture

### Technology Stack
- **Language:** Swift 6 (`swift-tools-version: 6.2`)
- **Frameworks:** Metal, Foundation, Accelerate (BLAS)
- **Build System:** Swift Package Manager
- **Testing:** Swift Testing framework (`@Test`, `#expect`)
- **GPU:** Metal Shading Language (from original, unchanged)

### Module Structure

```
SwiftMoE/
├── Sources/SwiftMoE/
│   ├── Model/         ModelConfig, FlashMoEError
│   ├── IO/            ExpertFile, AlignedBuffer, IOPool, WeightFile, WeightManifest
│   ├── Metal/         MetalContext, ShaderLibrary, BatchMatvec, ExpertEncoder,
│   │                  CMD2Encoder, GPULinearAttention, GPUFullAttention,
│   │                  ProjectionBuffers, ExpertBuffers, AttentionBuffers,
│   │                  LinearAttentionBuffers, CombineBuffers
│   ├── Attention/     FullAttention, LinearAttention, RMSNorm, RoPE, Softmax, BFloat16
│   ├── Inference/     LayerPipeline, TokenGenerator, DeferredExpertState, KVCache,
│   │                  LinearAttentionState, TopK, Embedding, LayerWeightCache, BPETokenizer
│   └── Server/        SessionStore
├── Sources/SwiftMoEHTTP/     The OpenAI-compatible service on SwiftHTTPService:
│                             ChatService, InferenceGate, GenerationBridge, InferenceLimits,
│                             APIKey (+ BearerCredential, Authentication, ServiceError),
│                             ChatRequest, ChatChunk, OpenAIError
├── Sources/SwiftMoEServer/   HTTP server executable
├── Sources/SwiftMoEChat/     Interactive TUI executable
├── Tests/SwiftMoETests/      255 tests; Fixtures/exchanges.json is the old server's recorded wire
└── metal_infer/              Original Obj-C reference implementation
```

---

## Current Status

### Complete
- [x] I/O subsystem (ExpertFile, AlignedBuffer, IOPool, WeightFile)
- [x] Metal context with 5 buffer groups (quantization-aware sizing)
- [x] All 19 Metal shader pipeline states compiled
- [x] CPU attention: FullAttention (RoPE + GQA) + LinearAttention (GatedDeltaNet + BLAS)
- [x] GPU pipeline: CMD1 projections + GPU linear attention (5 encoders)
- [x] GPU pipeline: CMD2 fused post-attention (8 encoders)
- [x] GPU pipeline: CMD2 GPU full attention (4 encoders)
- [x] GPU pipeline: CMD3 expert compute + combine + deferred
- [x] GPU pipeline: CMD3 fast path (skip deferred wait)
- [x] Token generation loop (embed → 60 layers → norm → lm_head → argmax)
- [x] BPE tokenizer (pure Swift, binary format compatible)
- [x] HTTP server with SSE streaming (OpenAI-compatible) — binds loopback by default;
      a wider bind is an explicit `host`
- [x] HTTP server access control: bearer credential (required off loopback; `--no-auth` on
      loopback only), origin allowlist in place of `Access-Control-Allow-Origin: *`, `Host`
      check on loopback, a ceiling on `max_tokens`, header/body/time limits, exact routing,
      and a thread per connection so one silent client cannot stop the server
- [x] HTTP server bounded work (2026-10-09): prompt tokens + completion tokens held to the
      sequence the KV caches hold, counted by the server's own tokenizer (400, never shortened);
      a queue for the model with a deadline (503 + `Retry-After`) in place of a lock; a client
      that disconnects gives up its queue place or stops its generation at the next token; a
      non-loopback bind refused unless plain text is acknowledged (`--allow-plaintext`)
- [x] HTTP server on SwiftHTTPService (2026-10-30, v0.2.0): the raw-socket listener is deleted; the
      service lives in `SwiftMoEHTTP`, which is the only target that imports the kit, so `SwiftMoE`
      imports no networking. TLS is terminated in the process (`--tls-cert`, `--tls-key`, `SIGHUP`
      reload), a response is bounded at two hours, shutdown is graceful, and the inference slot is an
      actor (`InferenceGate`) released through the stream's `observing`. Design:
      `project/plans/proposals/DesignProposal_HTTPServerOnSwiftHTTPService.md`
- [x] Inference layer refuses to drop context: `KVCache.append` throws
      `FlashMoEError.sequenceCapacityExceeded` at capacity; `TokenGenerator.generate` checks the
      whole sequence up front, resets state per call, and can be cancelled during prefill
- [x] Chat TUI with session persistence
- [x] Runtime ModelConfig with presets (.qwen397B, .tiny)
- [x] Synthetic test fixtures (SyntheticFixtures + ModelConfig.tiny)
- [x] GPU dispatch safety: every kernel bounds its thread id, every blocking wait checks
      command-buffer status, elementwise dispatches are exact rather than rounded

### Remaining
- [ ] Download Qwen3.5-397B weights and validate numerical equivalence
- [ ] Performance benchmarking against original C engine
- [ ] DocC documentation generation
- [ ] Additional model presets (DeepSeek-V3, Mixtral)
- [x] ~~HTTP server: authentication (bearer token), a restricted CORS origin instead of `*`,
      a read deadline, and a cap on `max_tokens` — none exist; loopback is the only control~~
      Shipped 2026-10-05 on `fix/server-credential-and-cors`; see Current Status.
- [x] ~~HTTP server: TLS. The bearer key crosses the network in clear text, so a non-loopback
      bind needs a TLS-terminating proxy in front of it.~~
      *(2026-10-09 this was restated as a decision, not a gap: "the raw-socket server will not grow
      TLS of its own"; the exposure could not happen by accident, a non-loopback bind was refused
      without `--allow-plaintext`, and the README carried a reverse-proxy recipe. Left open, because
      nothing here encrypted anything.)* **Shipped 2026-10-30 in 0.2.0**, by moving to the kit
      rather than growing TLS in the raw-socket code: the process terminates TLS (decision O2),
      `--allow-plaintext` and the recipe are gone.
- [x] ~~HTTP server: a prompt-length budget. The body is capped at 64 KiB, but the placeholder
      tokenizer makes that up to 65,536 prompt tokens against KV caches of 8,192 positions;
      the cap belongs with the real tokenizer (`--model` mode), which does not exist yet~~
      Shipped 2026-10-09 on `fix/bounded-work`, **despite the bar recorded here** ("belongs with
      the real tokenizer"). The reasoning that changed: the budget does not need the real
      tokenizer, it needs *whichever tokenizer the server is using* — so `HTTPServer` now takes
      one and counts with it, and `--model` mode will pass the BPE tokenizer through the same
      parameter. Waiting for `--model` would have left the demo server truncating context
      silently in the meantime.
- [x] ~~HTTP server: a validated request queues behind the running inference with no deadline
      of its own, holding one of the 16 connection slots for as long as that takes~~
      Shipped 2026-10-09 on `fix/bounded-work`: `InferenceQueue`, `Limits.queueDeadline`.
- [x] ~~HTTP server: no bound on how long one response may take to *deliver*. The write timeout
      (30 s) is per `write(2)`, so a client that reads slowly but steadily holds the model for
      as long as it likes — hours, at the kernel's low-water mark. Needs a whole-stream budget
      (a `Limits.responseDeadline`), and a decision about what it does to a legitimately long
      generation: 8,192 tokens at 4.4 tok/s is already 31 minutes~~
      Shipped 2026-10-30 in 0.2.0 as `InferenceLimits.maxStreamDuration`, two hours (decision O5):
      the longest completion the server will deliver, which is about four times the 31 minutes.
      A reader that stops reading ends its stream after the 30 s write deadline.
- [x] ~~HTTP server, **known regression** (found 2026-10-30, accepted as intended, caused by the kit):
      a connection past `maxConnections` is closed with no response, so the old `503` with
      `Retry-After` is gone. Needs a kit change to restore~~ Fixed 2026-10-30 in 0.2.1 on kit 0.6.0
      (plaintext: `503` with `Retry-After` = the queue deadline; over TLS the connection is closed
      before the handshake, nothing can be written). The 505, the silent-client 408 and the 413
      phrase came back with it. Six accepted differences remain (CHANGELOG 0.2.1).
- [ ] HTTP server: the 16 connection slots can be held by clients that have not authenticated —
      each for at most the 10 s read deadline, but reconnecting costs nothing. There is no
      per-address limit; a fronting proxy is the place for one
- [ ] HTTP server: a generation that fails after the stream has started ends the stream with
      no error event — the client sees a stream without `[DONE]`. **Decided (O3, 2026-10-30): keep
      the truncation**, the only signal every client treats as a failure. Left open as a deliberate
      behaviour; revisit if a client needs the reason
- [ ] `swift-moe-server --model`: still unimplemented, so the real-weights path has never
      served a request and the server's tokenizer is the one-token-per-byte placeholder
- [ ] A `LICENSE` file. The README defers to the upstream Flash-MoE repository; the package
      itself carries no licence, which matters more once there is a tag to depend on

---

## Quality Standards

- Zero compiler warnings
- Quality gate clean at 0 errors / 0 warnings (`--check all`: 49 checkers, 4 of them not
  applicable to this package and skipped), no overrides
- 255 tests, all passing
- No force unwraps, force casts, or `try!`
- No hardcoded domain constants (ADR-005)
- Integration tests use `ModelConfig.tiny` (no model download required)

---

**Last Updated:** 2026-10-30 — 0.2.1: the kit moved to 0.6.0 and the connection-cap regression recorded for
0.2.0 is fixed (struck through above, reasoning kept). Reconciled: Current Status, Remaining (six accepted
differences remain, down from nine), tests 254 → 255.

Earlier same day: 0.2.0 — the HTTP server moved onto SwiftHTTPService. Struck through, with their
reasoning kept: the TLS item (shipped by moving to the kit, which the 2026-10-09 note had said the
raw-socket server would not do) and the whole-response-deadline item (a two-hour `maxStreamDuration`).
The mid-stream error event stays open as the decided behaviour (O3). The 2026-10-09 note that the
raw-socket server "will not grow TLS of its own" was true of that server and was overtaken by
replacing it.

Earlier: 2026-10-09 — tagged v0.1.0, the first tag. Reconciled for the first release after the
bounded-work fix: the prompt-length budget and the queued-request items shipped and are struck
through, the first with a note that it shipped despite the bar it was recorded with; the TLS item
was restated as a decision and left open; five things found while doing it were added (response
delivery time, unauthenticated connection slots, no mid-stream error event, `--model`, a licence
file).

Earlier: 2026-10-05 — reconciled after the server credential and CORS fix: the "HTTP server:
authentication…" item under Remaining shipped and is struck through there and recorded under
Current Status; three things it leaves open (TLS, a prompt-length budget, queued requests) were
added to Remaining.
