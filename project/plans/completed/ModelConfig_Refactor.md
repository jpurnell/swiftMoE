# Design Proposal: ModelConfig Refactor — Enum → Runtime Struct

**Status:** APPROVED

---

## 1. Objective

Replace the hardcoded `ModelConfig` enum (compile-time constants for Qwen3.5-397B only) with a runtime-configurable struct that can describe any MoE transformer. This enables testing with small synthetic models and supports future models (DeepSeek-V3, Mixtral, etc.).

---

## 2. What Changes

### Before (enum with static constants)
```swift
public enum ModelConfig {
    public static let hiddenDim = 4096
    public static let numLayers = 60
    // ... 40+ hardcoded constants
}
```

### After (initializable struct)
```swift
public struct ModelConfig: Sendable {
    public let hiddenDim: Int
    public let numLayers: Int
    public let numExperts: Int
    // ...

    /// Qwen3.5-397B-A17B preset.
    public static let qwen397B = ModelConfig(...)

    /// Tiny model for testing (hidden=64, 2 layers, 4 experts).
    public static let tiny = ModelConfig(...)

    /// Load from a JSON config file.
    public init(configPath: String) throws
}
```

### What stays compile-time
- `dmaAlignment` (hardware constant: 2MB for Apple Silicon DMA)
- Metal threadgroup sizes (shader constants, not model-dependent)
- Quantization bit packing mechanics (shader assumption)

### What becomes runtime
Everything model-specific: dimensions, layer counts, expert counts, attention config, token IDs, expert binary layout offsets.

---

## 3. Affected Types

Every type that currently references `ModelConfig.someConstant` needs to either:
- Accept a `ModelConfig` parameter, or
- Be initialized with the relevant dimensions

| Type | Change |
|------|--------|
| `ModelConfig` | enum → struct with presets |
| `ExpertBuffers` | Takes config for expert sizes |
| `ProjectionBuffers` | Takes config for dimensions |
| `AttentionBuffers` | Takes config for head counts, KV dim |
| `LinearAttentionBuffers` | Takes config for linear attn dimensions |
| `CombineBuffers` | Takes config for hidden dim |
| `MetalContext` | Takes config, passes to buffer groups |
| `RoPE` | Already parameterized (good) |
| `RMSNorm` | Already parameterized (good) |
| `Softmax` | No change needed |
| `BatchMatvec` | No change needed (dimensions in specs) |
| `ExpertEncoder` | Takes config for offsets |
| `LayerPipeline` | Takes config |
| `TokenGenerator` | Takes config |
| `KVCache` | Already parameterized (good) |
| `LinearAttentionState` | Already partially parameterized |
| `LayerWeightCache` | Takes config for layer iteration |
| `Embedding` | Takes config for vocab/hidden dims |
| `TopK` | No change needed |
| `WeightManifest` | No change needed |
| `WeightFile` | No change needed |
| `ExpertFile` | No change needed |
| `AlignedBuffer` | No change needed |
| `IOPool` | No change needed |

**Good news:** RoPE, RMSNorm, Softmax, BatchMatvec, KVCache, TopK, and all I/O types already take dimensions as parameters. The refactor primarily hits the Metal buffer groups and the inference orchestration.

---

## 4. Test Strategy

- **Tiny preset test**: Create `ModelConfig.tiny` (hidden=64, 2 layers, 4 experts, vocab=32) and run the full pipeline end-to-end
- **Qwen preset test**: Verify `ModelConfig.qwen397B` produces the same values as the old enum
- **All existing tests**: Must continue passing (some will need config parameter added)

---

## 5. ADR

```yaml
id: ADR-004
date: 2026-04-05
status: accepted
category: architecture
title: Runtime ModelConfig instead of compile-time constants
decision: ModelConfig is a runtime struct with model presets, not a compile-time enum
rationale:
  - "Enables testing with tiny synthetic models without 210GB download"
  - "Supports multiple MoE architectures (DeepSeek-V3, Mixtral, future models)"
  - "The paper explicitly says the approach generalizes to any MoE model"
alternatives_rejected:
  - "Keep hardcoded: Works for one model but untestable without 210GB weights"
  - "Generic over config type: Over-engineered, runtime struct is simpler"
```

---

**Created:** 2026-04-05
