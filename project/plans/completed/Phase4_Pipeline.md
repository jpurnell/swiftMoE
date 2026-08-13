# Design Proposal: Phase 4 — Pipeline Orchestration

**Status:** APPROVED (moved directly to UPCOMING)

---

## 1. Objective

Port the `fused_layer_forward` function (~1,500 lines, infer.m:3998-5462) — the heart of the inference engine — which orchestrates the CMD1→CMD2→CMD3 pipeline across all 60 transformer layers. Also port the deferred expert state machine and the token generation loop.

---

## 2. Decomposition Strategy

The original is a single function with deeply nested conditionals. The Swift version decomposes it into **pipeline phases** that map to the original's documented structure:

```
Sources/FlashMoE/Inference/
├── DeferredExpertState.swift   # Async GPU expert state machine
├── LayerWeightCache.swift      # Pre-computed weight pointers per layer
├── LayerPipeline.swift         # fused_layer_forward decomposed into phases
├── TokenGenerator.swift        # Top-level loop: embed → 60 layers → norm → sample
├── TopK.swift                  # CPU top-K routing + weight normalization
├── KVCache.swift               # Full attention KV cache
├── LinearAttentionState.swift  # GatedDeltaNet persistent state (CPU path)
```

The key insight: `fused_layer_forward` is really 4 sequential phases:
1. **Deferred completion + CMD1** (attention projections)
2. **CPU attention** (full or linear)
3. **CMD2** (o_proj + residual + norm + routing + shared expert)
4. **Expert I/O + CMD3** (pread + expert forward + combine, DEFERRED)

Each phase becomes a method on `LayerPipeline`.

---

## 3. Key Design Decisions

- **LayerPipeline is a class** (mutable state: deferred expert, scratch buffers)
- **DeferredExpertState** replaces the C global `g_deferred`
- **Scratch buffers** allocated once, reused across all 60 layers (matching original)
- **No speculative routing** — disabled in the original (`spec_routing_enabled = 0`)
- **No LZ4/cache paths** initially — "Trust the OS" is the production configuration
- **Timing instrumentation** preserved as optional

---

## 4. Test Strategy

Without the 210GB model, we test:
- `TopK`: Known scores → correct top-K indices + weights
- `DeferredExpertState`: State transitions (active/inactive, GPU combined flag)
- `KVCache`: Append + lookup at positions
- **Integration**: `LayerPipeline` + `TokenGenerator` tested end-to-end when model is available

---

**Created:** 2026-04-05
