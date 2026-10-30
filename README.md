# SwiftMoE

A Swift inference engine for Mixture-of-Experts language models on Apple Silicon, based on the [Flash-MoE paper](paper/flash_moe.pdf).

Streams 200GB+ MoE models from NVMe SSD through a custom Metal compute pipeline, using only ~6GB of resident memory. Supports any MoE architecture through runtime-configurable `ModelConfig`.

## Features

- **NVMe Expert Streaming** -- Expert weights loaded on-demand via parallel `pread()` from SSD
- **Metal GPU Pipeline** -- Fused 3-command-buffer pipeline (CMD1/CMD2/CMD3) with deferred expert compute
- **Full + Linear Attention** -- GQA with RoPE (full) and BLAS-accelerated GatedDeltaNet (linear)
- **2-bit/4-bit Quantization** -- Quantization-aware buffer sizing saves ~64MB in 2-bit mode
- **Runtime Configurable** -- `ModelConfig` presets for any MoE architecture (Qwen, DeepSeek, etc.)
- **OpenAI-Compatible Server** -- `/v1/chat/completions` with SSE streaming, bearer-key authentication, and a bound on everything a request can cost
- **Pure Swift** -- No Python, no ML frameworks, no C dependencies (except optional linenoise for TUI)
- **254 Tests** -- Tiny synthetic model fixtures; the server is tested over a real loopback socket

## Status

Pre-1.0: the current version is 0.2.0 (0.1.0 was the first tag). The engine is verified against synthetic fixtures
and a CPU reference; it has **not** yet been run against the real Qwen3.5-397B weights, and
`swift-moe-server` serves only the synthetic `--demo` model (`--model` is not implemented).

## Requirements

- macOS 14 or later, Apple Silicon, a Metal-capable GPU
- Swift 6.2 toolchain (`swift-tools-version: 6.2`)

## Installation

As a package dependency:

```swift
dependencies: [
    .package(url: "https://github.com/jpurnell/swiftMoE", from: "0.2.0"),
],
targets: [
    .target(name: "YourTarget", dependencies: [.product(name: "SwiftMoE", package: "swiftMoE")]),
]
```

The package also builds two executables, `swift-moe-server` and `swift-moe-chat`.

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
| Key configured, loopback | Bearer authentication |
| Key configured, any other `--host` | Refuses to start unless `--tls-cert` and `--tls-key` are given (below) |
| No key, loopback, `--no-auth` | Starts unauthenticated, with a warning in the log |
| No key, loopback, no flag | Refuses to start |
| No key, any other `--host` | Refuses to start; neither `--no-auth` nor TLS changes this |
| Key and `--no-auth` | Refuses to start: a contradiction |

It listens on `127.0.0.1` unless `--host <address>` says otherwise.

### TLS: the server terminates it

On loopback the server speaks plain HTTP and nothing crosses a network. Anywhere else the key, the
prompts and the completions would be readable to anything on the path, so the server refuses to
start unless it holds a certificate and key of its own:

```bash
swift run swift-moe-server --demo --host 0.0.0.0 --port 8443 \
    --api-key-file ~/.swift-moe-key \
    --tls-cert /etc/ssl/moe.example.com/fullchain.pem \
    --tls-key  /etc/ssl/moe.example.com/privkey.pem
```

`--tls-cert` is the PEM chain, leaf first; `--tls-key` is the PEM private key and must not be
readable by group or other (`chmod 600`). Without them, off loopback:

```
swift-moe-server: Refusing to listen on 0.0.0.0 in plain text: other machines can reach it, and
the API key and every prompt would cross the network unencrypted. Give the server a certificate
and key (--tls-cert and --tls-key), or bind loopback (--host 127.0.0.1).
```

`SIGHUP` reloads the certificate and key from the same files without a restart; new connections
get the new identity, and if the new files are unfit the old one keeps being served. The server
speaks HTTP/1.1 only, one request per connection.

**The reverse-proxy recipe is superseded.** Earlier versions served plain HTTP and told you to put
Caddy or nginx in front, with `--allow-plaintext` to say so. That flag is gone (0.2.0): the
transport has no way to be told that something else encrypts the hop. A proxy on the same machine
still works against a loopback bind and still has to send the upstream `Host`
(`127.0.0.1:<port>`) and not buffer the response; a proxy on another machine now needs the server
to terminate TLS as well.

**How long a response may take.** A stream ends after two hours, whatever it is doing
(`InferenceLimits.maxStreamDuration`). At the measured 4.4 tokens per second a full 8,192-token
completion takes about 31 minutes, so two hours is the longest completion the server will deliver.
A reader that stops reading ends its stream after 30 seconds (`writeDeadline`).

**Browsers.** No CORS headers are sent, and a request carrying an `Origin` is refused (`403`),
unless the origin was allowed: `--allow-origin http://localhost:3000` (repeatable, compared
exactly). On loopback the `Host` header must be `127.0.0.1:<port>` or `localhost:<port>`
(`421` otherwise), which stops DNS rebinding.

**Limits.** A request outside these is refused with a status and a sentence, not trimmed:

| Limit | Default | Over it |
|---|---|---|
| `max_tokens` / `max_completion_tokens` | whole number, 1…8192 | `400` |
| Prompt tokens + completion tokens | 8192 | `400`, giving all three numbers |
| Request body | 64 KiB | `413` |
| Request line and headers | 16 KiB | `431` |
| Time to send the request head, and then the body (each) | 10 s | `408` |
| Time waiting for the model behind another request | 30 s | `503` with `Retry-After` |
| Connections at once — sending, waiting, or being answered | 16 | the connection is closed with no response |
| Time one response may take to deliver | 2 h | the stream ends, truncated |

8192 is the number of positions the KV caches are allocated for
(`TokenGenerator.defaultMaxSequenceLength`). The prompt is counted with the tokenizer the server
was given, and a request that does not fit is refused, not shortened: the inference layer will
not drop context either — a sequence past the caches' capacity is a thrown
`FlashMoEError.sequenceCapacityExceeded`, never a quietly truncated history.

One request runs at a time; the others wait in arrival order. The number waiting is bounded by
the connection limit, since each is a connection. A client that disconnects gives up its place
in the queue, or, if its request is running, stops the generation at the next token. Keep your
side of the connection open until you have the answer: a half-closed connection looks the same
as a closed one.

Library users import `SwiftMoEHTTP`, set these with `InferenceLimits`, and create the service with
`ChatService(host:port:authentication:allowedOrigins:limits:tls:tokenizer:generate:)`. The
`SwiftMoE` library itself imports no networking code.

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
      Server/         SessionStore
    SwiftMoEHTTP/     Library: the OpenAI-compatible service on SwiftHTTPService --
                      ChatService, InferenceGate, GenerationBridge, InferenceLimits,
                      APIKey, ChatRequest, ChatChunk, OpenAIError
    SwiftMoEServer/   Executable: OpenAI-compatible HTTP server
    SwiftMoEChat/     Executable: Interactive TUI chat client
  Tests/
    SwiftMoETests/    254 tests across 40 suites
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
