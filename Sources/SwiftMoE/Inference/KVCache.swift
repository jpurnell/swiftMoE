/// Key-value cache for full attention layers.
///
/// Only 15 of the 60 layers use full attention. Each maintains a KV cache
/// that grows with sequence length. The cache stores K and V vectors
/// contiguously for all positions processed so far.
public struct KVCache {
    private var kCache: [Float]
    private var vCache: [Float]
    private let kvDim: Int
    private let maxLength: Int
    private(set) var length: Int = 0

    /// Creates a KV cache for a single attention layer.
    ///
    /// - Parameters:
    ///   - kvDim: Key/value dimension (NUM_KV_HEADS * HEAD_DIM = 512).
    ///   - maxLength: Maximum sequence length to pre-allocate for.
    public init(kvDim: Int, maxLength: Int) {
        self.kvDim = kvDim
        self.maxLength = maxLength
        self.kCache = [Float](repeating: 0, count: maxLength * kvDim)
        self.vCache = [Float](repeating: 0, count: maxLength * kvDim)
    }

    /// Appends a new K/V pair at the current position.
    ///
    /// - Parameters:
    ///   - k: Key vector [kvDim].
    ///   - v: Value vector [kvDim].
    public mutating func append(k: [Float], v: [Float]) {
        guard length < maxLength else { return }
        let offset = length * kvDim
        kCache.replaceSubrange(offset..<offset + kvDim, with: k)
        vCache.replaceSubrange(offset..<offset + kvDim, with: v)
        length += 1
    }

    /// Appends K/V from raw pointers (zero-copy for GPU mirror updates).
    public mutating func append(kPtr: UnsafePointer<Float>, vPtr: UnsafePointer<Float>) {
        guard length < maxLength else { return }
        let offset = length * kvDim
        for i in 0..<kvDim {
            kCache[offset + i] = kPtr[i]
            vCache[offset + i] = vPtr[i]
        }
        length += 1
    }

    /// Provides read access to the K cache.
    public func withKCache<T>(_ body: (UnsafePointer<Float>) -> T) -> T {
        kCache.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else {
                preconditionFailure("KVCache K buffer is empty")
            }
            return body(base)
        }
    }

    /// Provides read access to the V cache.
    public func withVCache<T>(_ body: (UnsafePointer<Float>) -> T) -> T {
        vCache.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else {
                preconditionFailure("KVCache V buffer is empty")
            }
            return body(base)
        }
    }

    /// Resets the cache to empty (for new generation).
    public mutating func reset() {
        length = 0
    }
}
