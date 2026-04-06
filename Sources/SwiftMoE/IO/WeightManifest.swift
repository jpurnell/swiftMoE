import Foundation

/// Parses `model_weights.json` to locate tensors within `model_weights.bin`.
///
/// The manifest maps tensor names (e.g., "model.layers.0.self_attn.q_proj.weight")
/// to their byte offset, size, shape, and dtype within the binary file.
/// Lookup is O(1) via a hash table, matching the FNV-1a hash table in `infer.m:464-497`.
public struct WeightManifest: Sendable {

    /// Information about a single tensor in the weight file.
    public struct TensorInfo: Sendable {
        /// Tensor name (e.g., "model.layers.0.input_layernorm.weight").
        public let name: String

        /// Byte offset within model_weights.bin.
        public let offset: Int

        /// Size in bytes.
        public let size: Int

        /// Tensor shape (e.g., [4096] or [248320, 4096]).
        public let shape: [Int]

        /// Data type: "U32" (packed weights), "BF16" (scales/biases), or "F32" (norms).
        public let dtype: String
    }

    private let tensors: [String: TensorInfo]

    /// Loads and parses the JSON manifest.
    ///
    /// - Parameter path: Path to `model_weights.json`.
    /// - Throws: ``FlashMoEError/fileNotFound(path:)`` or ``FlashMoEError/manifestParseFailed(reason:)``.
    public init(path: String) throws {
        guard let data = FileManager.default.contents(atPath: path) else {
            throw FlashMoEError.fileNotFound(path: path)
        }

        let json: Any
        do {
            json = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw FlashMoEError.manifestParseFailed(reason: error.localizedDescription)
        }

        guard let root = json as? [String: Any],
              let tensorDict = root["tensors"] as? [String: [String: Any]] else {
            throw FlashMoEError.manifestParseFailed(reason: "Missing 'tensors' key or invalid structure")
        }

        var result: [String: TensorInfo] = [:]
        result.reserveCapacity(tensorDict.count)

        for (name, info) in tensorDict {
            guard let offset = (info["offset"] as? NSNumber)?.intValue,
                  let size = (info["size"] as? NSNumber)?.intValue,
                  let shapeArray = info["shape"] as? [NSNumber],
                  let dtype = info["dtype"] as? String else {
                continue
            }

            let shape = shapeArray.map { $0.intValue }
            result[name] = TensorInfo(
                name: name,
                offset: offset,
                size: size,
                shape: shape,
                dtype: dtype
            )
        }

        self.tensors = result
    }

    /// O(1) tensor lookup by name.
    ///
    /// - Parameter name: Full tensor name (e.g., "model.embed_tokens.weight").
    /// - Returns: Tensor info, or `nil` if the name is not found.
    public subscript(name: String) -> TensorInfo? {
        tensors[name]
    }

    /// Total number of tensors in the manifest.
    public var count: Int {
        tensors.count
    }
}
