# Changelog

All notable changes to SwiftMoE are documented in this file.

## [Unreleased]

### Added
- `MTLCommandBuffer.waitUntilCompletedChecked(_:)` — traps on a failed dispatch instead
  of letting callers read a stale buffer as if it were a result
- `MTLComputeCommandEncoder.dispatchExactly(threadCount:threadsPerThreadgroup:device:)` —
  dispatches exactly the threads an operation needs, with a rounded fallback for devices
  without non-uniform threadgroup support
- `TestPaths` — resolves test paths against an explicit root and rejects any that escape it
- `DequantMatvecV3Tests` — numeric GPU-vs-CPU comparison for `dequant_matvec_4bit_v3`
  across row counts that do and do not fill the final threadgroup
- Swift package with complete inference engine for Mixture-of-Experts models
- Metal compute shaders for 4-bit/2-bit dequantized matvec, RMS norm, SwiGLU, attention
- OpenAI-compatible HTTP server with SSE streaming
- Interactive chat client with tool calling support
- BPE tokenizer (pure Swift, no Python dependency)
- 84 unit and integration tests with synthetic fixtures

### Changed
- Replaced deprecated CBLAS calls with vDSP equivalents
- FMA-optimized dequant kernel (+12% throughput)
- `dequant_matvec_4bit_v3` strides its cooperative input load by the actual threadgroup
  size rather than a hardcoded 256, so a partially populated final threadgroup still
  fills `x_shared` completely
- Elementwise GPU dispatches (SwiGLU, residual add, RMS norm apply, MoE combine,
  conv1d step) and the v3 matvec now dispatch exact thread counts instead of rounding
  the grid up to whole threadgroups
- Tests resolve repo paths from `#filePath` rather than the process working directory,
  and use URL-based `FileManager` APIs throughout

### Fixed
- `compute_decay_beta` indexed six buffers by thread id with no bound and no element
  count to bound against; it now takes `num_v_heads` and guards on it
- Command buffers were read after `waitUntilCompleted()` with no check of `status` or
  `error`, so a failed dispatch was indistinguishable from a successful one that
  computed different numbers (`DeferredExpertState`, `LayerPipeline`, `CMD2Encoder`)
- The chat client waited on `DispatchSemaphore` with no deadline, so an unresponsive
  server blocked it forever; the wait is now bounded and cancels the request
- Conv1d state shift in linear attention
