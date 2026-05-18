import Metal

/// Specification for a single GPU dequantized matrix-vector multiply.
///
/// Maps directly to the C struct `BatchMatvecSpec` in `infer.m:1350`.
/// Each spec points to weight/scale/bias data within the mmap'd weight file
/// and specifies which `batch_out[]` slot receives the GPU result.
public struct BatchMatvecSpec {
    /// Packed quantized weights (pointer into mmap'd weight file).
    public let weights: UnsafeRawPointer
    /// Quantization scales (BF16, pointer into mmap'd weight file).
    public let scales: UnsafeRawPointer
    /// Quantization biases (BF16, pointer into mmap'd weight file).
    public let biases: UnsafeRawPointer
    /// CPU destination for the result (copied from GPU after completion).
    public var outputCPU: UnsafeMutablePointer<Float>
    /// Output dimension (number of rows in the weight matrix).
    public let outDim: UInt32
    /// Input dimension (number of columns, before packing).
    public let inDim: UInt32
    /// Quantization group size (64 for Qwen3.5).
    public let groupSize: UInt32
    /// Index into `MetalContext.projections.batchSlots` for GPU output.
    public let batchSlot: Int

    /// Creates a new BatchMatvecSpec with the given weight pointers and dimensions.
    public init(
        weights: UnsafeRawPointer,
        scales: UnsafeRawPointer,
        biases: UnsafeRawPointer,
        outputCPU: UnsafeMutablePointer<Float>,
        outDim: UInt32,
        inDim: UInt32,
        groupSize: UInt32,
        batchSlot: Int
    ) {
        self.weights = weights
        self.scales = scales
        self.biases = biases
        self.outputCPU = outputCPU
        self.outDim = outDim
        self.inDim = inDim
        self.groupSize = groupSize
        self.batchSlot = batchSlot
    }
}

/// GPU batched matrix-vector multiplication helpers.
///
/// These functions encode dequantized matvec dispatches into Metal command buffers
/// using byte offsets into the mmap'd weight file (wrapped as a single Metal buffer).
///
/// ## Command Buffer Protocol
/// `encode` does NOT commit the command buffer — the caller batches multiple
/// operations and commits once, reducing CPU-GPU synchronization overhead.
///
/// Matches `gpu_encode_batch_matvec` and `gpu_flush_batch_results` in `infer.m:1415-1459`.
public enum BatchMatvec {

    /// Encodes multiple dequant matvec dispatches into an existing command buffer.
    ///
    /// Each spec becomes one compute dispatch using either `matvec_v3` (for in_dim ≤ 4096)
    /// or `matvec_fast` (for larger dimensions). The weight/scale/bias pointers must point
    /// into the mmap'd weight file that's been set via `MetalContext.setWeights`.
    ///
    /// - Parameters:
    ///   - context: Metal context with compiled shaders and allocated buffers.
    ///   - commandBuffer: Command buffer to encode into (not committed).
    ///   - specs: Array of matvec specifications to dispatch.
    public static func encode(
        context: MetalContext,
        commandBuffer: MTLCommandBuffer,
        specs: [BatchMatvecSpec]
    ) {
        guard let wfBuf = context.weightBuffer else { return }
        let wfBase = UInt(bitPattern: wfBuf.contents())

        for spec in specs {
            // Compute byte offsets from the weight file base
            let wOffset = UInt(bitPattern: spec.weights) - wfBase
            let sOffset = UInt(bitPattern: spec.scales) - wfBase
            let bOffset = UInt(bitPattern: spec.biases) - wfBase

            // Select kernel based on input dimension
            let pipeline: MTLComputePipelineState
            let threadsPerTG: Int
            let numTG: Int

            if spec.inDim <= 4096 {
                pipeline = context.shaders.matvecV3
                threadsPerTG = 256  // 8 SIMD groups of 32
                numTG = (Int(spec.outDim) + 7) / 8  // 8 rows per threadgroup
            } else {
                pipeline = context.shaders.matvecFast
                threadsPerTG = 64
                numTG = Int(spec.outDim)
            }

            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(wfBuf, offset: Int(wOffset), index: 0)  // weights
            encoder.setBuffer(wfBuf, offset: Int(sOffset), index: 1)  // scales
            encoder.setBuffer(wfBuf, offset: Int(bOffset), index: 2)  // biases
            encoder.setBuffer(context.projections.input, offset: 0, index: 3)  // input vector
            encoder.setBuffer(context.projections.batchSlots[spec.batchSlot], offset: 0, index: 4)  // output

            var outDim = spec.outDim
            var inDim = spec.inDim
            var groupSize = spec.groupSize
            encoder.setBytes(&outDim, length: 4, index: 5)
            encoder.setBytes(&inDim, length: 4, index: 6)
            encoder.setBytes(&groupSize, length: 4, index: 7)

            encoder.dispatchThreadgroups(
                MTLSize(width: numTG, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1)
            )
            encoder.endEncoding()
        }
    }

    /// Copies GPU results from batch output slots back to CPU arrays.
    ///
    /// Call this after the command buffer has completed (after `waitUntilCompleted`).
    ///
    /// - Parameters:
    ///   - context: Metal context with batch output buffers.
    ///   - specs: The same specs that were encoded, providing CPU destinations.
    public static func flushResults(
        context: MetalContext,
        specs: [BatchMatvecSpec]
    ) {
        for spec in specs {
            let gpuBuf = context.projections.batchSlots[spec.batchSlot]
            let byteCount = Int(spec.outDim) * MemoryLayout<Float>.size
            memcpy(spec.outputCPU, gpuBuf.contents(), byteCount)
        }
    }
}
