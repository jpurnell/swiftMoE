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
- **Fully tested**: 166 tests with tiny synthetic model fixtures (no 210GB download needed)
- **All GPU paths**: 7 GPU optimization paths with automatic CPU fallback
- **Pure Swift**: No C dependencies in the main library
- **OpenAI-compatible**: `/v1/chat/completions` HTTP server with SSE streaming and bearer-key authentication

---

## Architecture

### Technology Stack
- **Language:** Swift 6.0
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
│   └── Server/        HTTPServer, APIKey (+ BearerCredential, HTTPServerError),
│                      HTTPConnection, HTTPRequestHead, HTTPRefusal, ClientConnection,
│                      SSEWriter, SessionStore
├── Sources/SwiftMoEServer/   HTTP server executable
├── Sources/SwiftMoEChat/     Interactive TUI executable
├── Tests/SwiftMoETests/      166 tests, 29 suites
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
      bind needs a TLS-terminating proxy in front of it
- [ ] HTTP server: a prompt-length budget. The body is capped at 64 KiB, but the placeholder
      tokenizer makes that up to 65,536 prompt tokens against KV caches of 8,192 positions;
      the cap belongs with the real tokenizer (`--model` mode), which does not exist yet
- [ ] HTTP server: a validated request queues behind the running inference with no deadline
      of its own, holding one of the 16 connection slots for as long as that takes

---

## Quality Standards

- Zero compiler warnings
- Quality gate clean at 0 errors / 0 warnings across all 45 checkers, no overrides
- 166 tests, all passing
- No force unwraps, force casts, or `try!`
- No hardcoded domain constants (ADR-005)
- Integration tests use `ModelConfig.tiny` (no model download required)

---

**Last Updated:** 2026-10-05 — reconciled after the server credential and CORS fix: the
"HTTP server: authentication…" item under Remaining shipped and is struck through there and
recorded under Current Status; three things it leaves open (TLS, a prompt-length budget, queued
requests) are added to Remaining. Test count 105 → 166 and suites 26 → 29 in the Overview,
Module Structure and Quality Standards; the Server row of Module Structure lists the new files.
