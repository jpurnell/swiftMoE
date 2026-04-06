import Metal

/// Owns a 2MB-aligned memory region backed by a Metal shared buffer.
///
/// The 2MB alignment is critical for DMA efficiency when loading expert weights
/// from NVMe SSD. Apple Silicon's DMA controller transfers data in aligned chunks;
/// 2MB alignment yields 3.6x faster reads than the default 16KB page alignment
/// (see Flash-MoE paper Section 3.6).
///
/// ## Design Notes
/// - Reference type (class) because these buffers are stored in arrays (`ExpertBuffers`)
///   and `Array<T>` requires `T: Copyable`. ARC overhead is negligible — buffers are
///   allocated once at startup, never in the per-layer hot path.
/// - The Metal buffer is created with `newBufferWithBytesNoCopy`, meaning the GPU
///   and CPU share the same physical memory — no copies on Apple's unified architecture.
/// - Memory is freed in `deinit` via the raw pointer (not the Metal buffer's deallocator).
public final class AlignedBuffer {
    private let rawPointer: UnsafeMutableRawPointer
    private let _metalBuffer: MTLBuffer
    private let _size: Int

    /// Allocates a 2MB-aligned buffer and wraps it in a Metal shared buffer.
    ///
    /// - Parameters:
    ///   - device: The Metal device used to create the MTLBuffer.
    ///   - size: Buffer size in bytes.
    ///   - alignment: Memory alignment in bytes. Defaults to 2MB for DMA optimization.
    /// - Throws: ``FlashMoEError/bufferAllocationFailed(size:)`` if allocation fails.
    public init(device: MTLDevice, size: Int, alignment: Int = ModelConfig.dmaAlignment) throws {
        var ptr: UnsafeMutableRawPointer?
        let result = posix_memalign(&ptr, alignment, size)
        guard result == 0, let aligned = ptr else {
            throw FlashMoEError.bufferAllocationFailed(size: size)
        }

        // Zero-initialize to avoid undefined behavior on first read
        memset(aligned, 0, size)

        guard let metalBuf = device.makeBuffer(
            bytesNoCopy: aligned,
            length: size,
            options: .storageModeShared,
            deallocator: nil  // We manage deallocation in deinit
        ) else {
            free(aligned)
            throw FlashMoEError.bufferAllocationFailed(size: size)
        }

        self.rawPointer = aligned
        self._metalBuffer = metalBuf
        self._size = size
    }

    /// Raw pointer to the aligned memory.
    ///
    /// Use this for `pread` destination or direct CPU-side data access.
    public var pointer: UnsafeMutableRawPointer {
        rawPointer
    }

    /// The Metal buffer wrapping this memory (`StorageModeShared`).
    ///
    /// Pass this to Metal compute encoders as a buffer argument. The GPU
    /// accesses the same physical memory as the CPU pointer.
    public var metalBuffer: MTLBuffer {
        _metalBuffer
    }

    /// Buffer size in bytes.
    public var size: Int {
        _size
    }

    deinit {
        free(rawPointer)
    }
}
