# SwiftMoE

A Swift inference engine for Mixture-of-Experts language models on Apple Silicon, based on the [Flash-MoE paper](paper/flash_moe.pdf).

Streams 200GB+ MoE models from NVMe SSD through a custom Metal compute pipeline, using only ~6GB of resident memory. Supports any MoE architecture through runtime-configurable `ModelConfig`.

## Features

- **NVMe Expert Streaming** -- Expert weights loaded on-demand via parallel `pread()` from SSD
- **Metal GPU Pipeline** -- Fused 3-command-buffer pipeline (CMD1/CMD2/CMD3) with deferred expert compute
- **Full + Linear Attention** -- GQA with RoPE (full) and BLAS-accelerated GatedDeltaNet (linear)
- **2-bit/4-bit Quantization** -- Quantization-aware buffer sizing saves ~64MB in 2-bit mode
- **Runtime Configurable** -- `ModelConfig` presets for any MoE architecture (Qwen, DeepSeek, etc.)
- **OpenAI-Compatible Server** -- `/v1/chat/completions` with SSE streaming, bearer-key authentication
- **Pure Swift** -- No Python, no ML frameworks, no C dependencies (except optional linenoise for TUI)
- **166 Tests** -- Full TDD coverage with tiny synthetic model fixtures

## Quick Start

```bash
# Build
swift build

# Run demo server (synthetic tiny model, no download needed)
umask 077 && openssl rand -hex 32 > ~/.swift-moe-key
swift run swift-moe-server --demo --port 8080 --api-key-file ~/.swift-moe-key

# In another terminal:
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/.swift-moe-key)" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'

# Run tests
swift test
```

### The server requires a key

Every request must carry `Authorization: Bearer <key>`; without it the answer is
`401 Unauthorized`. The server reads the key from a file or from the environment — never from
the command line, where `ps` would show it — and will not start without one:

```bash
umask 077 && openssl rand -hex 32 > ~/.swift-moe-key          # 64 characters; minimum is 32
swift run swift-moe-server --demo --api-key-file ~/.swift-moe-key
# or: SWIFT_MOE_API_KEY=$(cat ~/.swift-moe-key) swift run swift-moe-server --demo
swift run swift-moe-chat --api-key-file ~/.swift-moe-key
```

The key file must not be readable, writable or executable by group or other (`chmod 600`).

| Situation | What happens |
|---|---|
| Key configured | Bearer authentication, on any `--host` |
| No key, loopback, `--no-auth` | Starts unauthenticated, with a warning in the log |
| No key, loopback, no flag | Refuses to start |
| No key, any other `--host` | Refuses to start; `--no-auth` does not change this |
| Key and `--no-auth` | Refuses to start: a contradiction |

It listens on `127.0.0.1` unless `--host <ipv4>` says otherwise. The key crosses the network in
clear text, so off loopback put TLS in front of it.

**Browsers.** No CORS headers are sent, and a request carrying an `Origin` is refused (`403`),
unless the origin was allowed: `--allow-origin http://localhost:3000` (repeatable, compared
exactly). On loopback the `Host` header must be `127.0.0.1:<port>` or `localhost:<port>`
(`421` otherwise), which stops DNS rebinding.

**Limits.** A request outside these is refused with a status and a sentence, not trimmed:

| Limit | Default | Over it |
|---|---|---|
| `max_tokens` / `max_completion_tokens` | whole number, 1…8192 | `400` |
| Request body | 64 KiB | `413` |
| Request line and headers | 16 KiB | `431` |
| Time to send the whole request | 10 s | `408` |
| Connections being read at once | 16 | `503` |

8192 is the number of positions the KV caches are allocated for
(`TokenGenerator.defaultMaxSequenceLength`). Library users set these with `HTTPServer.Limits`
and create the server with `HTTPServer(port:authentication:allowedOrigins:limits:handler:)`.

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
      Server/         HTTPServer, APIKey, HTTPConnection, HTTPRequestHead,
                      HTTPRefusal, ClientConnection, SSEWriter, SessionStore
    SwiftMoEServer/   Executable: OpenAI-compatible HTTP server
    SwiftMoEChat/     Executable: Interactive TUI chat client
  Tests/
    SwiftMoETests/    166 tests across 29 suites (2s)
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
