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
- **Fully tested**: 226 tests with tiny synthetic model fixtures (no 210GB download needed)
- **All GPU paths**: 7 GPU optimization paths with automatic CPU fallback
- **Pure Swift**: No C dependencies in the main library
- **OpenAI-compatible**: `/v1/chat/completions` HTTP server with SSE streaming, bearer-key authentication,
  and a bound on what one request can cost (sequence budget, queue deadline, cancellation on disconnect)

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
│   └── Server/        HTTPServer (+ Request, Tokenizer), HTTPServerLimits,
│                      APIKey (+ BearerCredential, HTTPServerError),
│                      HTTPConnection, HTTPRequestHead, HTTPRefusal, ClientConnection,
│                      InferenceQueue, SSEWriter, SessionStore
├── Sources/SwiftMoEServer/   HTTP server executable
├── Sources/SwiftMoEChat/     Interactive TUI executable
├── Tests/SwiftMoETests/      226 tests, 35 suites
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
- [ ] HTTP server: TLS. The bearer key crosses the network in clear text, so a non-loopback
      bind needs a TLS-terminating proxy in front of it.
      **Still not shipped, and now a decision rather than a gap** (2026-10-09): the raw-socket server will not grow TLS of
      its own. What shipped instead is that the exposure cannot happen by accident — a
      non-loopback bind is refused without `--allow-plaintext` — and a reverse-proxy recipe in
      the README. The key still crosses the proxy-to-server hop in clear text when that hop is
      not loopback. Left open, unticked, because nothing here encrypts anything.
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
- [ ] HTTP server: no bound on how long one response may take to *deliver*. The write timeout
      (30 s) is per `write(2)`, so a client that reads slowly but steadily holds the model for
      as long as it likes — hours, at the kernel's low-water mark. Needs a whole-stream budget
      (a `Limits.responseDeadline`), and a decision about what it does to a legitimately long
      generation: 8,192 tokens at 4.4 tok/s is already 31 minutes
- [ ] HTTP server: the 16 connection slots can be held by clients that have not authenticated —
      each for at most the 10 s read deadline, but reconnecting costs nothing. There is no
      per-address limit; a fronting proxy is the place for one
- [ ] HTTP server: a generation that fails after the stream has started ends the stream with
      no error event — the client sees a stream without `[DONE]`. `SSEWriter` has no
      `sendError`
- [ ] `swift-moe-server --model`: still unimplemented, so the real-weights path has never
      served a request and the server's tokenizer is the one-token-per-byte placeholder
- [ ] A `LICENSE` file. The README defers to the upstream Flash-MoE repository; the package
      itself carries no licence, which matters more once there is a tag to depend on

---

## Quality Standards

- Zero compiler warnings
- Quality gate clean at 0 errors / 0 warnings (`--check all`: 49 checkers, 4 of them not
  applicable to this package and skipped), no overrides
- 226 tests, all passing
- No force unwraps, force casts, or `try!`
- No hardcoded domain constants (ADR-005)
- Integration tests use `ModelConfig.tiny` (no model download required)

---

**Last Updated:** 2026-10-09 — reconciled for the first release (intended 0.1.0) after the
bounded-work fix. Under Remaining: the prompt-length budget and the queued-request items shipped
and are struck through, the first with a note that it shipped despite the bar it was recorded
with; the TLS item is restated as a decision (no TLS in the raw-socket server; plain text must be
acknowledged) and left open; five things found while doing it are added (response delivery
time, unauthenticated connection slots, no mid-stream error event, `--model`, a licence file).
Current Status gains the two shipped items. Test count 166 → 226 and suites 29 → 35 in the
Overview, Module Structure and Quality Standards; the Server row lists `InferenceQueue` and
`HTTPServerLimits`; the checker count is corrected from 45 to what `--check all` runs today;
Technology Stack said Swift 6.0 where the manifest says tools 6.2.

Earlier: 2026-10-05 — reconciled after the server credential and CORS fix: the "HTTP server:
authentication…" item under Remaining shipped and is struck through there and recorded under
Current Status; three things it leaves open (TLS, a prompt-length budget, queued requests) were
added to Remaining.
