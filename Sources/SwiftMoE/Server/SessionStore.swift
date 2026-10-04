import Foundation
#if canImport(os)
import os
#endif

private let logger = Logger(subsystem: "com.swiftmoe", category: "session")

/// Persists conversation history as JSONL files.
///
/// Each session is stored in `~/.flash-moe/sessions/<session_id>.jsonl`.
/// Each line is a JSON object with `role` and `content` fields.
///
/// Matches the session persistence in `chat.m:52-100`.
public final class SessionStore {

    /// Directory for session files.
    public let sessionsDir: String

    /// Allowed root directory for all file operations (CWE-22 prevention).
    private let allowedRoot: URL

    /// Current session ID.
    public private(set) var sessionID: String

    /// Number of random bytes in a generated session id.
    private static let sessionIDByteCount = 32

    /// Generates a session id: 32 bytes from `generator`, as 64 lowercase hex digits.
    ///
    /// The id names the file a conversation is stored in, so it is drawn to be unguessable
    /// rather than merely unique. A UUID is the wrong tool for that — RFC 4122 §6 says not to
    /// assume UUIDs are hard to guess.
    ///
    /// The generator is a parameter so that the program's entry point names it once, and so a
    /// test can state the bytes it expects. Anything that names a real session must pass
    /// `SystemRandomNumberGenerator`: an id is only as unpredictable as its generator.
    ///
    /// Four 64-bit words are drawn and each is rendered most-significant byte first, so the
    /// output is a function of the generator's words alone.
    ///
    /// - Parameter generator: Source of randomness.
    /// - Returns: A 64-character lowercase hexadecimal string.
    public static func makeSessionID(using generator: inout some RandomNumberGenerator) -> String {
        let wordCount = sessionIDByteCount / MemoryLayout<UInt64>.size
        let hexDigitsPerWord = MemoryLayout<UInt64>.size * 2
        return (0..<wordCount).map { _ in
            let hex = String(generator.next(), radix: 16)
            return String(repeating: "0", count: hexDigitsPerWord - hex.count) + hex
        }.joined()
    }

    /// Creates a new session store, optionally resuming an existing session.
    ///
    /// ```swift
    /// var entropy = SystemRandomNumberGenerator()
    /// let store = SessionStore(sessionID: nil, using: &entropy)
    /// ```
    ///
    /// - Parameters:
    ///   - sessionID: Id of a session to resume. When `nil`, a new id is drawn from
    ///     `generator` with ``makeSessionID(using:)``.
    ///   - generator: Source of randomness for a new id. Pass `SystemRandomNumberGenerator`.
    public convenience init(sessionID: String? = nil, using generator: inout some RandomNumberGenerator) {
        let home = ProcessInfo.processInfo.environment["HOME"] ?? "/tmp"
        self.init(sessionID: sessionID, sessionsDirectory: "\(home)/.flash-moe/sessions", using: &generator)
    }

    /// Creates a session store rooted at an explicit directory.
    ///
    /// - Parameters:
    ///   - sessionID: Id of a session to resume, or `nil` to generate one.
    ///   - sessionsDirectory: Directory that holds the session files.
    ///   - generator: Source of randomness for a new id.
    init(sessionID: String?, sessionsDirectory dir: String, using generator: inout some RandomNumberGenerator) {
        self.sessionsDir = dir
        self.allowedRoot = URL(fileURLWithPath: dir).standardized
        self.sessionID = sessionID ?? Self.makeSessionID(using: &generator)

        let dirURL = URL(fileURLWithPath: dir).standardized
        guard PathContainment.isContained(dirURL, in: allowedRoot) else { return }
        do {
            try FileManager.default.createDirectory(
                at: dirURL, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create sessions directory: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Path to the current session file, validated against the sessions directory.
    public var sessionPath: String {
        let url = URL(fileURLWithPath: sessionsDir)
            .appendingPathComponent(sessionID)
            .appendingPathExtension("jsonl")
            .standardized
        guard PathContainment.isContained(url, in: allowedRoot) else {
            return allowedRoot.appendingPathComponent("invalid.jsonl").path
        }
        return url.path
    }

    /// Validates a path stays within the allowed root directory.
    private func validated(_ path: String) -> URL? {
        let resolved = URL(fileURLWithPath: path).standardized
        guard PathContainment.isContained(resolved, in: allowedRoot) else {
            logger.error("Path traversal blocked: \(path, privacy: .private)")
            return nil
        }
        return resolved
    }

    /// Appends a message to the current session.
    ///
    /// - Parameters:
    ///   - role: Message role ("user" or "assistant").
    ///   - content: Message text.
    public func appendMessage(role: String, content: String) {
        guard let validatedURL = validated(sessionPath) else { return }
        let escaped = content
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        let line = "{\"role\":\"\(role)\",\"content\":\"\(escaped)\"}\n"

        guard let data = line.data(using: .utf8) else { return }
        if FileManager.default.isReadableFile(atPath: validatedURL.path) {
            do {
                let handle = try FileHandle(forWritingTo: validatedURL)
                handle.seekToEndOfFile()
                handle.write(data)
                handle.closeFile()
            } catch {
                logger.debug("Failed to open session file for append: \(error.localizedDescription, privacy: .public)")
            }
        } else {
            do {
                try data.write(to: validatedURL)
            } catch {
                logger.debug("Failed to create session file: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    /// Loads all messages from a session file.
    ///
    /// - Returns: Array of (role, content) tuples.
    public func loadMessages() -> [(role: String, content: String)] {
        guard let validatedURL = validated(sessionPath) else { return [] }
        let data: Data
        do {
            data = try Data(contentsOf: validatedURL)
        } catch {
            logger.debug("Session file not readable: \(error.localizedDescription, privacy: .public)")
            return []
        }
        guard let text = String(data: data, encoding: .utf8) else {
            return []
        }

        // Split on the Unicode newline property, not the "\n" literal. "\r\n" is a
        // single Character in Swift, so a literal "\n" never matches it and a
        // CRLF-written session file comes back as one element holding the whole document.
        return text.split(whereSeparator: \.isNewline).compactMap { line in
            guard let lineData = String(line).data(using: .utf8) else { return nil }
            do {
                guard let json = try JSONSerialization.jsonObject(with: lineData) as? [String: String],
                      let role = json["role"],
                      let content = json["content"] else {
                    return nil
                }
                return (role: role, content: content)
            } catch {
                logger.debug("Failed to parse session line: \(error.localizedDescription, privacy: .public)")
                return nil
            }
        }
    }

    /// Lists all available session IDs.
    public func listSessions() -> [String] {
        guard let validatedURL = validated(sessionsDir) else { return [] }
        let urls: [URL]
        do {
            urls = try FileManager.default.contentsOfDirectory(
                at: validatedURL, includingPropertiesForKeys: nil)
        } catch {
            logger.debug("No sessions directory: \(error.localizedDescription, privacy: .public)")
            return []
        }
        let files = urls.map { $0.lastPathComponent }
        return files
            .filter { $0.hasSuffix(".jsonl") }
            .map { String($0.dropLast(6)) }
            .sorted()
    }
}
