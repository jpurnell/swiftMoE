import Foundation
import os

private let logger = Logger(subsystem: "com.swiftmoe", category: "manifest")

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
    /// - Parameters:
    ///   - path: Path to `model_weights.json`.
    ///   - allowedRoot: Root directory the resolved path must stay within. Defaults to the
    ///     parent directory of `path`.
    /// - Throws: ``FlashMoEError/fileNotFound(path:)``, ``FlashMoEError/pathTraversal(path:allowedRoot:)``,
    ///   or ``FlashMoEError/manifestParseFailed(reason:)``.
    public init(path: String, allowedRoot: String? = nil) throws {
        // Resolve and validate the path to prevent CWE-22 path traversal.
        let resolvedURL = URL(fileURLWithPath: path).standardized
        let rootURL: URL
        if let allowedRoot {
            rootURL = URL(fileURLWithPath: allowedRoot).standardized
        } else {
            rootURL = resolvedURL.deletingLastPathComponent()
        }
        guard resolvedURL.path.hasPrefix(rootURL.path) else {
            throw FlashMoEError.pathTraversal(path: resolvedURL.path, allowedRoot: rootURL.path)
        }

        let data: Data
        do {
            data = try Data(contentsOf: resolvedURL)
        } catch {
            throw FlashMoEError.fileNotFound(path: resolvedURL.path)
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
                // silent: malformed tensor entries are skipped; the caller validates required tensors via subscript
                logger.debug("Skipping malformed tensor entry: \(name, privacy: .public)")
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
        logger.debug("Loaded manifest with \(result.count, privacy: .public) tensors from \(resolvedURL.path, privacy: .public)")
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
