# Changelog

All notable changes to SwiftMoE are documented in this file.

## [Unreleased]

No version of SwiftMoE has been tagged, so there is no number to bump; the next tag should be
a minor (`0.x`) or major release, because the two changes below break source.

### Security — breaking
- **The HTTP server bound every interface and said it was on `localhost`.** `HTTPServer` bound
  `INADDR_ANY` and logged `http://localhost:<port>/…`. The server has no authentication, so
  any machine that could route to the port could run inference. It now binds `127.0.0.1`
  unless told otherwise, and the log line is the address read back from the socket with
  `getsockname`, with a warning when that address is not loopback.
  - `HTTPServer.init(host:port:handler:)` — new `host` parameter, default
    `HTTPServer.loopbackHost` (`"127.0.0.1"`). It takes an IPv4 literal; a host name is
    refused with `FlashMoEError.invalidBindAddress(host:)` rather than resolved.
  - `HTTPServer.openListener()` binds and returns an `HTTPServer.BoundAddress`;
    `HTTPServer.boundAddress` reads it back later. Port `0` reports the port the kernel chose.
  - `HTTPServer.ipv4Address(_:)`, `HTTPServer.isLoopback(_:)`.
  - `swift-moe-server --host <ipv4>`.
- **A new session's id was a UUID.** `SessionStore(sessionID: nil)` used `UUID().uuidString`.
  A UUID is built to be unique, not unguessable. The id is now 32 bytes from a
  `RandomNumberGenerator`, as 64 lowercase hex digits (`SessionStore.makeSessionID(using:)`),
  and the chat client passes the system generator. Session files already on disk keep their
  names and still resume by id.

### Migration
- **Anyone who reached the server from another machine**: it no longer answers there. Pass
  the address to publish on — `swift-moe-server --host 0.0.0.0`, or
  `HTTPServer(host: "0.0.0.0", port: 8080) { … }` — knowing that it is unauthenticated.
  Prefer leaving it on loopback behind a reverse proxy that authenticates.
- `HTTPServer(port:handler:)` compiles unchanged and now means loopback.
- `SessionStore()` / `SessionStore(sessionID:)` no longer compile. Name the generator:
  ```swift
  var entropy = SystemRandomNumberGenerator()
  let store = SessionStore(sessionID: resumedID, using: &entropy)
  ```
- A `switch` over `FlashMoEError` without a `default` needs the new
  `.invalidBindAddress(host:)` case.

### Fixed
- Six containment checks — the weight manifest, the session store, and the chat and server
  entry points — used `path.hasPrefix(root.path)`. `…/sessions-other/x.jsonl` begins with
  `…/sessions`, so a session id of `../sessions-other/x` was written outside the sessions
  directory (the id comes from the chat CLI's own configuration). All six use
  `PathContainment.isContained(_:in:)`, which compares whole components after resolving `..`
  and symbolic links; tested for a sibling, a climb out, and a link out.

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
- 91 unit and integration tests with synthetic fixtures

### Changed
- Replaced deprecated CBLAS calls with vDSP equivalents
- FMA-optimized dequant kernel (+12% throughput)
- `dequant_matvec_4bit_v3` strides its cooperative input load by the actual threadgroup
  size rather than a hardcoded 256, so a partially populated final threadgroup still
  fills `x_shared` completely
- Elementwise GPU dispatches (SwiGLU, residual add, RMS norm apply, MoE combine,
  conv1d step) and the v3 matvec now dispatch exact thread counts instead of rounding
  the grid up to whole threadgroups
- `IOReadStats.throughputBytesPerSec` is now `Double?`: `nil` when `totalMs` is not a
  positive, finite duration. It used to answer `0`, which reads as a measured rate
- The `SwiftMoE` target no longer excludes `SwiftMoE.docc`, so DocC receives the
  catalogue and its articles and symbol links are actually checked
- Tests resolve repo paths from `#filePath` rather than the process working directory,
  and use URL-based `FileManager` APIs throughout

### Fixed
- Three `Int(_:)` conversions of floating-point values that nothing showed to be
  representable — `Int(.nan)` and `Int(.infinity)` trap. `ModelConfig.rotaryDim` now
  returns a documented 0 ("rotate nothing") when `partialRotary` is not a fraction in
  `0...1`; `LayerTiming.formatSummary()` pads on the rendered integer part, so a
  non-finite accumulator prints as `nan`/`inf`; the chat client's timeout log formats
  the deadline as a `Double` instead of converting it
- Eleven GPU tests opened with
  `guard let shaderURL = ShaderLibraryTests.shaderURL else { return }`, so a checkout
  without `metal_infer/shaders.metal` got eleven green tests that compiled no shader,
  built no `MetalContext`, and asserted nothing — across `ShaderLibrary`,
  `MetalContext`, `BatchMatvec`, `DequantMatvecV3`, `TokenGenerator` and `Integration`.
  The condition is now `ShaderLibraryTests.shadersAvailable`, consumed by an
  `.enabled(if:)` trait so the framework records a skip; inside the body the URL is
  unwrapped with `try #require`, because once the trait says the file is there a nil is
  a defect and not an absence. `MetalContextTests.setWeights` had the same shape after
  `posix_memalign` — `guard let aligned = ptr else { return }` skipped the `setWeights`
  call the test is named for — and now checks the errno and requires the pointer
- `compute_decay_beta` indexed six buffers by thread id with no bound and no element
  count to bound against; it now takes `num_v_heads` and guards on it
- Command buffers were read after `waitUntilCompleted()` with no check of `status` or
  `error`, so a failed dispatch was indistinguishable from a successful one that
  computed different numbers (`DeferredExpertState`, `LayerPipeline`, `CMD2Encoder`)
- The chat client waited on `DispatchSemaphore` with no deadline, so an unresponsive
  server blocked it forever; the wait is now bounded and cancels the request
- Conv1d state shift in linear attention
