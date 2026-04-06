import Metal
import Foundation

/// Top-level Metal context owning all GPU resources for inference.
///
/// Created once at startup. Provides access to the device, command queue,
/// compiled shader pipelines, and all pre-allocated buffer groups.
///
/// ## Memory Budget
/// Total Metal scratch allocation:
/// - 4-bit mode: ~200MB (128MB expert data + ~72MB everything else)
/// - 2-bit mode: ~136MB (64MB expert data + ~72MB everything else)
///
/// The remaining ~41GB of the 48GB unified memory is available for the OS page cache,
/// which manages expert file caching.
public final class MetalContext {

    /// The model configuration driving buffer sizes and kernel parameters.
    public let config: ModelConfig

    /// The Metal device (GPU).
    public let device: MTLDevice

    /// Command queue for submitting work to the GPU.
    public let queue: MTLCommandQueue

    /// Compiled compute pipeline states.
    public let shaders: ShaderLibrary

    /// Projection buffers (input, output, batch slots).
    public let projections: ProjectionBuffers

    /// Expert weight buffers (double-buffered, 2MB-aligned).
    public let experts: ExpertBuffers

    /// Full attention buffers (KV caches, scores, scratch).
    public let attention: AttentionBuffers

    /// GatedDeltaNet linear attention state and scratch buffers.
    public let linearAttention: LinearAttentionBuffers

    /// CMD3 combine buffers (residual, hidden, params, norms).
    public let combine: CombineBuffers

    /// The mmap'd weight file wrapped as a Metal buffer (zero-copy on unified memory).
    /// Set via ``setWeights(_:size:)`` after loading the weight file.
    public private(set) var weightBuffer: MTLBuffer?

    /// Shared event for CPU-GPU synchronization in the async pipeline.
    public let pipelineEvent: MTLSharedEvent

    /// Monotonically increasing event counter for pipeline synchronization.
    public private(set) var eventValue: UInt64 = 0

    /// Creates the Metal context: compiles shaders, allocates all buffers.
    ///
    /// - Parameters:
    ///   - config: Model configuration describing architecture dimensions.
    ///   - shaderPath: Path to `shaders.metal` source file.
    ///   - use2Bit: If true, sizes expert data buffers for 2-bit quantization (saves ~64MB).
    /// - Throws: ``FlashMoEError`` if Metal device is unavailable or shader compilation fails.
    public init(config: ModelConfig, shaderPath: String, use2Bit: Bool = false) throws {
        self.config = config

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FlashMoEError.metalUnavailable
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw FlashMoEError.metalUnavailable
        }
        self.queue = queue

        self.shaders = try ShaderLibrary(device: device, shaderPath: shaderPath)
        self.projections = try ProjectionBuffers(device: device, config: config)
        self.experts = try ExpertBuffers(device: device, config: config, use2Bit: use2Bit)
        self.attention = try AttentionBuffers(device: device, config: config)
        self.linearAttention = try LinearAttentionBuffers(device: device, config: config)
        self.combine = try CombineBuffers(device: device, config: config)

        guard let event = device.makeSharedEvent() else {
            throw FlashMoEError.metalUnavailable
        }
        self.pipelineEvent = event
    }

    /// Wraps the mmap'd weight file as a Metal buffer (zero-copy on unified memory).
    ///
    /// The mmap'd address must be page-aligned (16KB on Apple Silicon).
    /// This is guaranteed by `mmap` but not by `malloc`.
    ///
    /// - Parameters:
    ///   - data: Pointer to the mmap'd weight file.
    ///   - size: Size of the weight file in bytes.
    public func setWeights(_ data: UnsafeMutableRawPointer, size: Int) {
        let pageSize = 16384
        let alignedSize = (size + pageSize - 1) & ~(pageSize - 1)

        weightBuffer = device.makeBuffer(
            bytesNoCopy: data,
            length: alignedSize,
            options: .storageModeShared,
            deallocator: nil  // mmap owns the memory
        )
    }

    /// Resets all linear attention state buffers (delta-net + conv1d).
    ///
    /// Call this at the start of each new token generation to clear
    /// the recurrence state from the previous generation.
    public func resetLinearAttentionState() {
        linearAttention.resetState()
    }

    /// Increments and returns the next pipeline event value.
    public func nextEventValue() -> UInt64 {
        eventValue += 1
        return eventValue
    }
}
