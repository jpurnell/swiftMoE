import Foundation

/// Resolves repo and scratch paths for tests without trusting the working directory.
///
/// Every lookup is standardized (so `..` and relative segments collapse) and then
/// checked for containment in an explicit allowed root. A path that resolves
/// outside its root is rejected rather than opened or deleted. All filesystem
/// access goes through URL-based APIs so the resolved, validated URL is what
/// reaches the filesystem — never a caller-shaped string.
enum TestPaths {

    /// Repo root, derived from this file's compile-time location rather than `cwd`.
    static let repoRoot: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // SwiftMoETests/
        .deletingLastPathComponent()  // Tests/
        .deletingLastPathComponent()  // repo root
        .standardizedFileURL

    /// The system temporary directory, standardized once for containment checks.
    static let scratchRoot: URL = FileManager.default.temporaryDirectory.standardizedFileURL

    /// True when `url` is `root` itself or lies beneath it.
    ///
    /// Both sides are standardized before comparison, so a candidate like
    /// `root/../etc/passwd` fails the check instead of resolving past the root.
    static func isContained(_ url: URL, in root: URL) -> Bool {
        let resolved = url.standardizedFileURL.path
        let base = root.standardizedFileURL.path
        return resolved == base || resolved.hasPrefix(base.hasSuffix("/") ? base : base + "/")
    }

    /// Resolves `relativePath` inside `root`, returning nil if it escapes the root.
    static func resolve(_ relativePath: String, within root: URL) -> URL? {
        let candidate = root.appendingPathComponent(relativePath).standardizedFileURL
        guard isContained(candidate, in: root) else { return nil }
        return candidate
    }

    /// True when `url` names an existing regular file (not a directory).
    static func isRegularFile(_ url: URL) -> Bool {
        let values = try? url.standardizedFileURL.resourceValues(forKeys: [.isRegularFileKey])
        return values?.isRegularFile ?? false
    }

    /// Resolves `relativePath` inside `root` and returns it only if it names an
    /// existing regular file. Nil means "escaped the root" or "not present" —
    /// callers treat both as "skip this test".
    static func existingFile(_ relativePath: String, within root: URL) -> URL? {
        guard let candidate = resolve(relativePath, within: root), isRegularFile(candidate) else {
            return nil
        }
        return candidate
    }

    /// Resolves an existing regular file inside the repo root.
    static func existingRepoFile(_ relativePath: String) -> URL? {
        existingFile(relativePath, within: repoRoot)
    }

    /// True when `path` names an existing regular file inside `root`.
    static func fileExists(_ path: String, within root: URL) -> Bool {
        let candidate = URL(fileURLWithPath: path).standardizedFileURL
        return isContained(candidate, in: root) && isRegularFile(candidate)
    }

    /// Reads a file that must live inside `root`. Returns nil if it escapes the
    /// root or cannot be read.
    static func contents(of path: String, within root: URL) -> Data? {
        let candidate = URL(fileURLWithPath: path).standardizedFileURL
        guard isContained(candidate, in: root) else { return nil }
        return try? Data(contentsOf: candidate)
    }

    /// Creates a uniquely named scratch directory beneath the system temp dir.
    static func makeScratchDirectory(prefix: String) throws -> URL {
        let directory = scratchRoot
            .appendingPathComponent("\(prefix)_\(UUID().uuidString)")
            .standardizedFileURL
        guard isContained(directory, in: scratchRoot), directory.path != scratchRoot.path else {
            throw TestPathError.escapesRoot(directory.path, root: scratchRoot.path)
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// Best-effort removal of a scratch directory, refusing anything that
    /// resolves outside the system temp dir or names the temp dir itself.
    static func removeScratchDirectory(_ path: String) {
        let directory = URL(fileURLWithPath: path).standardizedFileURL
        guard isContained(directory, in: scratchRoot), directory.path != scratchRoot.path else { return }
        try? FileManager.default.removeItem(at: directory)  // silent: best-effort test cleanup
    }
}

enum TestPathError: Error, CustomStringConvertible {
    case escapesRoot(String, root: String)

    var description: String {
        switch self {
        case .escapesRoot(let path, let root):
            return "Resolved path \(path) escapes its allowed root \(root)"
        }
    }
}
