import Foundation

/// Errors thrown by the FlashMoE inference engine.
public enum FlashMoEError: Error, Sendable, Equatable {
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

    /// A feature is not yet implemented.
    case notImplemented(feature: String)

    /// A resolved path escapes its allowed directory.
    case pathTraversal(path: String, allowedRoot: String)

    /// A listener was asked to bind something that is not an IPv4 literal.
    ///
    /// Host names are refused rather than resolved, so a name can never widen a listener past
    /// the address its caller wrote down.
    case invalidBindAddress(host: String)

    /// A sequence needs more positions than the KV caches were allocated for.
    ///
    /// The caches do not grow. Recording nothing past the end would leave every later token
    /// attending to a history with a hole in it, and nothing in the output would say so; this
    /// is thrown instead, before the position is computed.
    ///
    /// - Parameters:
    ///   - capacity: Positions the caches hold.
    ///   - required: Positions the sequence needs — for an append, the one being written.
    case sequenceCapacityExceeded(capacity: Int, required: Int)
}
