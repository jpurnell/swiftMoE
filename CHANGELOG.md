# Changelog

All notable changes to SwiftMoE are documented in this file.

## [Unreleased]

### Added
- Swift package with complete inference engine for Mixture-of-Experts models
- Metal compute shaders for 4-bit/2-bit dequantized matvec, RMS norm, SwiGLU, attention
- OpenAI-compatible HTTP server with SSE streaming
- Interactive chat client with tool calling support
- BPE tokenizer (pure Swift, no Python dependency)
- 82 unit and integration tests with synthetic fixtures

### Changed
- Replaced deprecated CBLAS calls with vDSP equivalents
- FMA-optimized dequant kernel (+12% throughput)

### Fixed
- Conv1d state shift in linear attention
