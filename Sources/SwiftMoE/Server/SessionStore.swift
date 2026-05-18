import Foundation
import os

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

    /// Creates a new session store, optionally resuming an existing session.
    public init(sessionID: String? = nil) {
        let home = ProcessInfo.processInfo.environment["HOME"] ?? "/tmp"
        let dir = "\(home)/.flash-moe/sessions"
        self.sessionsDir = dir
        self.allowedRoot = URL(fileURLWithPath: dir).standardized
        self.sessionID = sessionID ?? UUID().uuidString

        let dirURL = URL(fileURLWithPath: dir).standardized
        guard dirURL.path.hasPrefix(allowedRoot.path) else { return }
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
        guard url.path.hasPrefix(allowedRoot.path) else {
            return allowedRoot.appendingPathComponent("invalid.jsonl").path
        }
        return url.path
    }

    /// Validates a path stays within the allowed root directory.
    private func validated(_ path: String) -> URL? {
        let resolved = URL(fileURLWithPath: path).standardized
        guard resolved.path.hasPrefix(allowedRoot.path) else {
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

        return text.split(separator: "\n").compactMap { line in
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
