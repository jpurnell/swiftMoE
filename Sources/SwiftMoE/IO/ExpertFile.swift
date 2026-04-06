import Foundation

/// Owns a file descriptor for a packed expert binary file.
///
/// Uses `pread` for position-independent, thread-safe reads directly into
/// Metal-shared buffers. The file descriptor is closed when this value is consumed.
///
/// ## Design Notes
/// - `~Copyable`: enforces single ownership of the file descriptor at compile time.
/// - `borrowing` on read methods: the file is borrowed, not consumed, allowing
///   multiple reads from the same file across its lifetime.
/// - `pread` is used instead of `read` because it's stateless — multiple threads
///   can read from the same fd simultaneously without seeking.
public struct ExpertFile: ~Copyable {
    private let fd: Int32
    private let path: String

    /// Opens a packed expert file.
    ///
    /// - Parameter path: Path to the packed expert binary (e.g., "packed_experts/layer_00.bin").
    /// - Throws: ``FlashMoEError/fileNotFound(path:)`` if the file does not exist.
    public init(path: String) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw FlashMoEError.fileNotFound(path: path)
        }
        self.fd = fd
        self.path = path
    }

    /// The underlying POSIX file descriptor.
    ///
    /// Exposed for interop with GCD dispatch groups or other low-level I/O
    /// patterns that need the raw fd.
    public var fileDescriptor: Int32 {
        fd
    }

    /// Reads a single expert's weights into the destination buffer using `pread`.
    ///
    /// `pread` is position-independent and thread-safe: the file offset is computed
    /// from `expertIndex * expertSize`, and no shared file position is modified.
    ///
    /// - Parameters:
    ///   - expertIndex: Expert index within the file (0-based).
    ///   - destination: Pointer to pre-allocated memory (must be >= `expertSize` bytes).
    ///   - expertSize: Bytes per expert (use ``ModelConfig/expertSize4Bit`` or ``ModelConfig/expertSize2Bit``).
    /// - Returns: Number of bytes actually read.
    /// - Throws: ``FlashMoEError/readFailed(errno:context:)`` if the read fails.
    @discardableResult
    public func readExpert(
        index expertIndex: Int,
        into destination: UnsafeMutableRawPointer,
        expertSize: Int
    ) throws -> Int {
        let offset = off_t(expertIndex) * off_t(expertSize)
        let bytesRead = pread(fd, destination, expertSize, offset)
        guard bytesRead >= 0 else {
            throw FlashMoEError.readFailed(
                errno: errno,
                context: "pread expert \(expertIndex) from \(path)"
            )
        }
        return bytesRead
    }

    deinit {
        close(fd)
    }
}
