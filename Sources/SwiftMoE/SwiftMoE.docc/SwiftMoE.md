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

``HTTPServer`` exposes `/v1/chat/completions` with no authentication, so it binds the
loopback interface unless a caller names another address. `HTTPServer(port:handler:)`
listens on `127.0.0.1`; `HTTPServer(host:port:handler:)` is how a wider bind is written
down. ``HTTPServer/openListener()`` returns the address the socket actually holds.

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
- ``SSEWriter``
- ``SessionStore``
