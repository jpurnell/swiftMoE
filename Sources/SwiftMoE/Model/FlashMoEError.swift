import Foundation

/// Errors thrown by the FlashMoE inference engine.
public enum FlashMoEError: Error, Sendable {
    /// No Metal-capable GPU device found on this system.
    case metalUnavailable

    /// A required file (weight file, expert file, manifest) was not found.
    case fileNotFound(path: String)

    /// A pread or file I/O operation failed.
    case readFailed(errno: Int32, context: String)

    /// The JSON weight manifest could not be parsed.
    case manifestParseFailed(reason: String)

    /// Memory allocation failed (posix_memalign or Metal buffer).
    case bufferAllocationFailed(size: Int)

    /// Metal shader source failed to compile.
    case shaderCompilationFailed(reason: String)
}
