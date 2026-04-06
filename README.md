# SwiftMoE

A Swift inference engine for Mixture-of-Experts language models on Apple Silicon, based on the [Flash-MoE paper](paper/flash_moe.pdf).

Streams 200GB+ MoE models from NVMe SSD through a custom Metal compute pipeline, using only ~6GB of resident memory. Supports any MoE architecture through runtime-configurable `ModelConfig`.

## Features

- **NVMe Expert Streaming** -- Expert weights loaded on-demand via parallel `pread()` from SSD
- **Metal GPU Pipeline** -- Fused 3-command-buffer pipeline (CMD1/CMD2/CMD3) with deferred expert compute
- **Full + Linear Attention** -- GQA with RoPE (full) and BLAS-accelerated GatedDeltaNet (linear)
- **2-bit/4-bit Quantization** -- Quantization-aware buffer sizing saves ~64MB in 2-bit mode
- **Runtime Configurable** -- `ModelConfig` presets for any MoE architecture (Qwen, DeepSeek, etc.)
- **OpenAI-Compatible Server** -- `/v1/chat/completions` with SSE streaming
- **Pure Swift** -- No Python, no ML frameworks, no C dependencies (except optional linenoise for TUI)
- **82 Tests** -- Full TDD coverage with tiny synthetic model fixtures

## Quick Start

```bash
# Build
swift build

# Run demo server (synthetic tiny model, no download needed)
swift run swift-moe-server --demo --port 8080

# In another terminal:
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'

# Run tests
swift test
```

## Architecture

```
SwiftMoE/
  Sources/
    SwiftMoE/
      Model/          ModelConfig (runtime presets), FlashMoEError
      IO/             ExpertFile, AlignedBuffer, IOPool, WeightFile, WeightManifest
      Metal/          MetalContext, ShaderLibrary, BatchMatvec, ExpertEncoder,
                      CMD2Encoder, GPULinearAttention, GPUFullAttention,
                      ProjectionBuffers, ExpertBuffers, AttentionBuffers,
                      LinearAttentionBuffers, CombineBuffers
      Attention/      FullAttention, LinearAttention (BLAS), RMSNorm, RoPE,
                      Softmax, BFloat16
      Inference/      LayerPipeline, TokenGenerator, DeferredExpertState,
                      KVCache, LinearAttentionState, TopK, Embedding,
                      LayerWeightCache, BPETokenizer
      Server/         HTTPServer, SSEWriter, SessionStore
    SwiftMoEServer/   Executable: OpenAI-compatible HTTP server
    SwiftMoEChat/     Executable: Interactive TUI chat client
  Tests/
    SwiftMoETests/    82 tests across 21 suites (0.5s)
  metal_infer/        Original Obj-C/Metal reference implementation
```

## GPU Pipeline

Each transformer layer executes a 3-command-buffer pipeline:

```
CMD1: Attention projections (Q/K/V or QKV/Z/Beta/Alpha)
      + GPU linear attention (conv1d, delta-net, gated norm) [45 layers]

CMD2: o_proj + residual_add + rms_norm + routing + shared expert
      + GPU full attention (scores, softmax, values, sigmoid gate) [15 layers]
      All fused into single command buffer (8-12 encoders, 1 commit)

CMD3: Expert forward passes (gate+up+SwiGLU+down) for K experts
      + shared expert SwiGLU + down_proj
      + GPU-side combine + residual + RMS norm for next layer
      DEFERRED: committed async, completed at start of next layer
```

## Model Support

Configure any MoE architecture via `ModelConfig`:

```swift
let config = ModelConfig(
    hiddenDim: 4096, numLayers: 60, numAttentionHeads: 32,
    numKVHeads: 2, headDim: 256, vocabSize: 248_320,
    numExperts: 512, moeIntermediate: 1024, ...
)

// Or use a preset:
let config = ModelConfig.qwen397B   // Qwen3.5-397B-A17B
let config = ModelConfig.tiny       // Testing (hidden=64, 2 layers)
```

## Origin

This is a Swift modernization of the [Flash-MoE](https://github.com/danveloper/flash-moe) inference engine, which demonstrated running a 397B parameter model at 5.74 tok/s on a MacBook Pro with 48GB RAM. The original was written in Objective-C/C with hand-tuned Metal shaders during a 24-hour human-AI collaboration.

The Swift version preserves all GPU optimization paths while adding type safety, runtime configurability, structured concurrency, and comprehensive test coverage.

## Papers

- [Flash-MoE: Streaming a 397B Parameter MoE from NVMe at 5.7 Tokens/Second](paper/flash_moe.pdf)
- [LLM in a Flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514)

## License

See the original [Flash-MoE repository](https://github.com/danveloper/flash-moe) for license terms.
