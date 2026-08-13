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
- **Fully tested**: 82 tests with tiny synthetic model fixtures (no 210GB download needed)
- **All GPU paths**: 7 GPU optimization paths with automatic CPU fallback
- **Pure Swift**: No C dependencies in the main library
- **OpenAI-compatible**: `/v1/chat/completions` HTTP server with SSE streaming

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
│   └── Server/        HTTPServer, SSEWriter, SessionStore
├── Sources/SwiftMoEServer/   HTTP server executable
├── Sources/SwiftMoEChat/     Interactive TUI executable
├── Tests/SwiftMoETests/      82 tests, 21 suites
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
- [x] HTTP server with SSE streaming (OpenAI-compatible)
- [x] Chat TUI with session persistence
- [x] Runtime ModelConfig with presets (.qwen397B, .tiny)
- [x] Synthetic test fixtures (SyntheticFixtures + ModelConfig.tiny)

### Remaining
- [ ] Download Qwen3.5-397B weights and validate numerical equivalence
- [ ] Performance benchmarking against original C engine
- [ ] DocC documentation generation
- [ ] Additional model presets (DeepSeek-V3, Mixtral)

---

## Quality Standards

- Zero compiler warnings
- 82 tests, all passing
- No force unwraps, force casts, or `try!`
- No hardcoded domain constants (ADR-005)
- Integration tests use `ModelConfig.tiny` (no model download required)

---

**Last Updated:** 2026-04-05
