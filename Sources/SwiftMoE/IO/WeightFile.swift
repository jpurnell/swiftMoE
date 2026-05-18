import Foundation

/// Memory-mapped access to the non-expert weight file (`model_weights.bin`).
///
/// The entire ~5.5GB file is `mmap`'d read-only at startup. Individual tensors
/// are accessed via byte offsets from the manifest. On Apple Silicon's unified
/// memory, the mmap'd region is directly addressable by both CPU and GPU.
///
/// ## Lifecycle
/// The file remains mapped for the process lifetime. The `deinit` unmaps it.
public final class WeightFile {

    /// Raw pointer to the mmap'd data.
    public let data: UnsafeMutableRawPointer

    /// Size of the mapped region in bytes.
    public let size: Int

    /// Tensor manifest for name → offset lookup.
    public let manifest: WeightManifest

    /// File descriptor (kept open for mmap lifetime).
    private let fd: Int32

    /// Opens and memory-maps the weight file.
    ///
    /// - Parameters:
    ///   - weightsPath: Path to `model_weights.bin`.
    ///   - manifestPath: Path to `model_weights.json`.
    /// - Throws: ``FlashMoEError/fileNotFound(path:)`` if either file is missing.
    public init(weightsPath: String, manifestPath: String) throws {
        self.manifest = try WeightManifest(path: manifestPath)

        let fd = open(weightsPath, O_RDONLY)
        guard fd >= 0 else {
            throw FlashMoEError.fileNotFound(path: weightsPath)
        }
        self.fd = fd

        // Get file size
        var stat = stat()
        fstat(fd, &stat)
        let fileSize = Int(stat.st_size)
        self.size = fileSize

        // mmap the entire file read-only
        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != MAP_FAILED else {
            close(fd)
            throw FlashMoEError.readFailed(errno: errno, context: "mmap \(weightsPath)")
        }
        self.data = mapped

        // Hint for sequential access during startup
        madvise(mapped, fileSize, MADV_SEQUENTIAL)
    }

    /// Returns a typed pointer to a tensor within the mmap'd data.
    ///
    /// - Parameters:
    ///   - name: Full tensor name (e.g., "model.layers.0.input_layernorm.weight").
    ///   - type: The element type to bind the pointer to (e.g., `Float.self`, `UInt32.self`).
    /// - Returns: Pointer to the tensor data, or `nil` if the tensor isn't in the manifest.
    public func tensorPointer<T>(name: String, as type: T.Type = T.self) -> UnsafePointer<T>? {
        guard let info = manifest[name] else { return nil }
        return UnsafePointer((data + info.offset).assumingMemoryBound(to: T.self))
    }

    /// Returns a mutable raw pointer at a tensor's offset (for GPU buffer wrapping).
    public func tensorRawPointer(name: String) -> UnsafeRawPointer? {
        guard let info = manifest[name] else { return nil }
        return UnsafeRawPointer(data + info.offset)
    }

    deinit {
        munmap(data, size)
        close(fd)
    }
}
