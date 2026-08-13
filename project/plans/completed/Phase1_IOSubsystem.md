# Design Proposal: Phase 1 — I/O Subsystem + Project Structure

**Status:** PROPOSAL — awaiting approval

---

## 1. Objective

Establish the Swift Package Manager project structure and implement the I/O subsystem — the foundation that every subsequent phase depends on. The I/O subsystem is the single most performance-critical component (47% of per-layer time is expert I/O at 1.37ms/layer), making it the ideal first target: if Swift can match pread throughput, everything else follows.

**Master Plan Reference:** Phase 1 — Foundation

---

## 2. Proposed Architecture

**New Files:**

```
FlashMoE/
├── Package.swift
├── Sources/
│   ├── FlashMoE/
│   │   ├── Model/
│   │   │   └── ModelConfig.swift          # Model constants (HIDDEN_DIM, NUM_LAYERS, etc.)
│   │   ├── IO/
│   │   │   ├── ExpertFile.swift           # ~Copyable file descriptor wrapper
│   │   │   ├── AlignedBuffer.swift        # ~Copyable 2MB-aligned memory + MTLBuffer
│   │   │   ├── IOPool.swift               # Parallel pread via structured concurrency
│   │   │   └── WeightManifest.swift       # JSON tensor manifest loader
│   │   └── Interop/
│   │       └── BridgingHeader.swift       # C function imports for pread, posix_memalign
│   ├── CTokenizer/                        # C target wrapping tokenizer.h
│   │   ├── include/
│   │   │   └── tokenizer.h
│   │   └── tokenizer_impl.c              # #define TOKENIZER_IMPL + #include
│   └── CLineNoise/                        # C target wrapping linenoise
│       ├── include/
│       │   └── linenoise.h
│       └── linenoise.c
├── Tests/
│   └── FlashMoETests/
│       ├── ExpertFileTests.swift
│       ├── AlignedBufferTests.swift
│       ├── IOPoolTests.swift
│       └── WeightManifestTests.swift
└── metal_infer/
    ├── shaders.metal                      # Unchanged
    └── ...                                # Original Obj-C files (reference)
```

**Modified Files:** None (greenfield Swift alongside existing Obj-C)

**Module Placement:** `FlashMoE` library target + `CTokenizer` and `CLineNoise` C targets

---

## 3. API Surface

### ExpertFile (~Copyable)

```swift
/// Owns a file descriptor for a packed expert binary file.
/// Provides pread access with zero-copy into Metal-shared buffers.
public struct ExpertFile: ~Copyable {
    /// Opens a packed expert file (e.g., "packed_experts/layer_00.bin").
    /// - Throws: FlashMoEError.fileNotFound if path doesn't exist
    public init(path: String) throws

    /// The number of experts in this file (always 512 for Qwen3.5).
    public var expertCount: Int { get }

    /// Reads a single expert's weights into the destination buffer.
    /// Uses pread for position-independent, thread-safe reads.
    /// - Parameters:
    ///   - expertIndex: Expert index (0..<512)
    ///   - destination: Pointer to pre-allocated buffer (must be ≥ expertSize bytes)
    ///   - expertSize: Bytes per expert (EXPERT_SIZE or EXPERT_SIZE_2BIT)
    /// - Returns: Number of bytes actually read
    /// - Throws: FlashMoEError.readFailed on I/O error
    @discardableResult
    public borrowing func readExpert(
        index expertIndex: Int,
        into destination: UnsafeMutableRawPointer,
        expertSize: Int
    ) throws -> Int

    /// The underlying file descriptor (for use with dispatch/GCD if needed).
    public borrowing var fileDescriptor: Int32 { get }

    deinit  // closes fd
}
```

### AlignedBuffer (~Copyable)

```swift
/// Owns a 2MB-aligned memory region backed by a Metal shared buffer.
/// The alignment is critical for DMA efficiency (3.6x faster than 16KB).
public struct AlignedBuffer: ~Copyable {
    /// Allocates a 2MB-aligned buffer of the given size.
    /// Wraps it in a Metal shared buffer via newBufferWithBytesNoCopy.
    /// - Parameters:
    ///   - device: Metal device for buffer creation
    ///   - size: Buffer size in bytes
    ///   - alignment: Memory alignment (default: 2MB for DMA)
    /// - Throws: FlashMoEError.bufferAllocationFailed
    public init(device: MTLDevice, size: Int, alignment: Int = 2 * 1024 * 1024) throws

    /// Raw pointer to the aligned memory.
    public borrowing var pointer: UnsafeMutableRawPointer { get }

    /// The Metal buffer wrapping this memory (StorageModeShared).
    public borrowing var metalBuffer: MTLBuffer { get }

    /// Buffer size in bytes.
    public var size: Int { get }

    deinit  // frees aligned memory (Metal buffer deallocator handles this)
}
```

### IOPool

```swift
/// Manages parallel pread operations for expert weight loading.
/// Replaces the hand-rolled pthread pool from infer.m:2970-3120.
public struct IOPool {
    /// Number of concurrent I/O operations.
    public let concurrency: Int

    public init(concurrency: Int = 4)

    /// Reads K experts in parallel from the given file into destination buffers.
    /// - Parameters:
    ///   - file: The expert file to read from (borrowed, not consumed)
    ///   - expertIndices: Which experts to load (e.g., top-K routing results)
    ///   - destinations: Pre-allocated buffers, one per expert
    ///   - expertSize: Bytes per expert
    /// - Throws: FlashMoEError.readFailed if any read fails
    public func readExperts(
        from file: borrowing ExpertFile,
        indices expertIndices: [Int],
        into destinations: [UnsafeMutableRawPointer],
        expertSize: Int
    ) async throws
}
```

### ModelConfig

```swift
/// Compile-time constants for Qwen3.5-397B-A17B.
public enum ModelConfig {
    public static let hiddenDim = 4096
    public static let numLayers = 60
    public static let numAttentionHeads = 32
    public static let numKVHeads = 2
    public static let headDim = 256
    public static let vocabSize = 248_320
    public static let numExperts = 512
    public static let expertSize4Bit = 7_077_888
    public static let expertSize2Bit = 3_932_160
    // ... remaining constants
}
```

### WeightManifest

```swift
/// Parses model_weights.json to locate tensors in model_weights.bin.
public struct WeightManifest: Sendable {
    public struct TensorInfo: Sendable {
        public let name: String
        public let offset: Int
        public let size: Int
        public let shape: [Int]
        public let dtype: String  // "U32", "BF16", "F32"
    }

    /// Loads and parses the JSON manifest.
    public init(path: String) throws

    /// O(1) tensor lookup by name.
    public subscript(name: String) -> TensorInfo? { get }

    /// Total number of tensors.
    public var count: Int { get }
}
```

---

## 4. MCP Schema

Not applicable for this phase — I/O subsystem is internal infrastructure, not a user-facing API.

---

## 5. Constraints & Compliance

| Constraint | Compliance |
|------------|-----------|
| **Performance parity** | pread throughput must match Obj-C (measured via benchmark) |
| **~Copyable ownership** | ExpertFile and AlignedBuffer are non-copyable, enforce single ownership |
| **Sendable** | ModelConfig (enum), WeightManifest (immutable after init), TensorInfo (value type) |
| **No force unwraps** | All failable operations throw FlashMoEError |
| **No new dependencies** | Uses only Foundation, Metal, Darwin/POSIX |
| **Swift 6 concurrency** | IOPool uses async/await, no manual pthread |
| **DMA alignment** | AlignedBuffer preserves the 2MB alignment from original |

---

## 6. Backend Abstraction

Not applicable — this phase is pure I/O and memory management, no compute.

---

## 7. Dependencies

**Internal Dependencies:** None (this is the foundation)

**External Dependencies:** None

**System Dependencies:**
- `Darwin.POSIX` — pread, open, close, posix_memalign
- `Metal` — MTLDevice, MTLBuffer, MTLResourceStorageModeShared
- `Foundation` — JSONSerialization (for manifest parsing)

---

## 8. Test Strategy

**Test Categories:**

### ExpertFile Tests
- **Golden path:** Open a test file, pread known bytes at known offset, verify content
- **Edge cases:** Expert index 0, expert index 511 (boundaries)
- **Error handling:** File not found → throws `.fileNotFound`
- **Ownership:** Verify file descriptor is closed on deinit (check with fcntl)

### AlignedBuffer Tests
- **Golden path:** Allocate buffer, write data via pointer, read via Metal buffer contents
- **Alignment verification:** Check that pointer address is 2MB-aligned (`pointer % (2*1024*1024) == 0`)
- **Metal integration:** Buffer can be used as MTLBuffer argument to compute encoder
- **Error handling:** Zero-size allocation behavior

### IOPool Tests
- **Golden path:** Read 4 experts in parallel, verify all contain correct data
- **Concurrency:** Multiple concurrent readExperts calls don't interfere
- **Performance:** Parallel reads faster than sequential (logical assertion, not wall-clock)

### WeightManifest Tests
- **Golden path:** Parse sample JSON, look up known tensor by name
- **O(1) lookup:** Verify subscript returns correct offset/size
- **Missing tensor:** Returns nil, doesn't crash

**Reference Truth:** The original Obj-C code's behavior is the reference. We will create small test fixture files (not the full 209GB model) to validate I/O correctness.

**Validation Trace:**
- Create a 1MB test file with known byte pattern
- `ExpertFile.readExpert(index: 0, into: buffer, expertSize: 1024)` → first 1024 bytes match pattern
- `AlignedBuffer(device:size:).pointer` address `% 2_097_152 == 0`

---

## 9. Architecture Decision Review

**ADR Check:**
- [x] Reviewed `06_ARCHITECTURE_DECISIONS.md` — empty (first decisions for this project)
- [ ] New ADR required? Yes — two new entries:

**New ADR Drafts:**

```yaml
id: ADR-001
date: 2026-04-05
status: proposed
category: architecture
title: Use ~Copyable types for system resource ownership
decision: |
  ExpertFile (file descriptor) and AlignedBuffer (posix_memalign'd memory + MTLBuffer)
  use Swift's ~Copyable (move-only) types to enforce single ownership at compile time.
rationale:
  - Eliminates use-after-close and double-free bugs by construction
  - Zero runtime overhead compared to manual C resource management
  - Better than ARC-based class wrappers (no reference counting traffic in hot path)
alternatives_rejected:
  - "Class with deinit: Adds ARC overhead in tight loops (2.9ms per-layer budget)"
  - "Manual C-style management: The original approach; error-prone, not testable"
```

```yaml
id: ADR-002
date: 2026-04-05
status: proposed
category: concurrency
title: Use Swift structured concurrency for parallel I/O
decision: |
  Replace the hand-rolled pthread_mutex/pthread_cond thread pool with
  Swift async/await and TaskGroup for parallel expert pread operations.
rationale:
  - Compile-time Sendable checking prevents data races
  - Cancellation and priority propagation for free
  - The original had race conditions (42% experiment discard rate, many from races)
  - TaskGroup maps directly to the "read K experts in parallel" pattern
alternatives_rejected:
  - "Keep pthreads: Works but not testable, no compile-time safety"
  - "GCD dispatch_apply: Already tried in original; comparable performance"
  - "OperationQueue: Higher overhead than TaskGroup for this pattern"
```

---

## 10. Open Questions — RESOLVED

1. **Package structure for Metal shaders:** ~~Should shaders be bundled as a resource in the SPM package, or compiled at runtime from source?~~ **RESOLVED:** Runtime compilation from source, matching the original. Simpler for development, 180ms startup cost is acceptable.

2. **Test fixtures:** ~~Should we create small synthetic expert files for testing, or require the actual model weights?~~ **RESOLVED:** Create synthetic fixtures (e.g., 4KB "mini experts") with known byte patterns. Reproducibility is essential — tests must be fast, portable, and CI-friendly.

3. **async vs sync pread:** ~~Should we use `DispatchIO` or a custom executor?~~ **RESOLVED:** Start with `Task { pread(...) }` wrapping the blocking call. Add latency instrumentation to IOPool so we can track whether cooperative threading adds measurable overhead vs. the pthread baseline. If benchmarks show degradation, fall back to `DispatchQueue`-based approach.

---

## 11. Documentation Strategy

**Documentation Type:** API Docs Only (for Phase 1)

**Complexity Threshold Check:**
- Does it combine 3+ APIs? No — each type is independent
- Does explanation require 50+ lines? No
- Does it need theory/background context? No (systems programming, not domain theory)

No narrative article required. DocC comments on each type are sufficient.

---

**Created:** 2026-04-05
