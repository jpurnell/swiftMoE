import Foundation

/// Persists conversation history as JSONL files.
///
/// Each session is stored in `~/.flash-moe/sessions/<session_id>.jsonl`.
/// Each line is a JSON object with `role` and `content` fields.
///
/// Matches the session persistence in `chat.m:52-100`.
public final class SessionStore {

    /// Directory for session files.
    public let sessionsDir: String

    /// Current session ID.
    public private(set) var sessionID: String

    public init(sessionID: String? = nil) {
        let home = ProcessInfo.processInfo.environment["HOME"] ?? "/tmp"
        self.sessionsDir = "\(home)/.flash-moe/sessions"
        self.sessionID = sessionID ?? UUID().uuidString

        // Create sessions directory
        try? FileManager.default.createDirectory(
            atPath: sessionsDir, withIntermediateDirectories: true)
    }

    /// Path to the current session file.
    public var sessionPath: String {
        "\(sessionsDir)/\(sessionID).jsonl"
    }

    /// Appends a message to the current session.
    ///
    /// - Parameters:
    ///   - role: Message role ("user" or "assistant").
    ///   - content: Message text.
    public func appendMessage(role: String, content: String) {
        let escaped = content
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        let line = "{\"role\":\"\(role)\",\"content\":\"\(escaped)\"}\n"

        if let data = line.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: sessionPath) {
                if let handle = FileHandle(forWritingAtPath: sessionPath) {
                    handle.seekToEndOfFile()
                    handle.write(data)
                    handle.closeFile()
                }
            } else {
                FileManager.default.createFile(atPath: sessionPath, contents: data)
            }
        }
    }

    /// Loads all messages from a session file.
    ///
    /// - Returns: Array of (role, content) tuples.
    public func loadMessages() -> [(role: String, content: String)] {
        guard let data = FileManager.default.contents(atPath: sessionPath),
              let text = String(data: data, encoding: .utf8) else {
            return []
        }

        return text.split(separator: "\n").compactMap { line in
            guard let lineData = String(line).data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: lineData) as? [String: String],
                  let role = json["role"],
                  let content = json["content"] else {
                return nil
            }
            return (role: role, content: content)
        }
    }

    /// Lists all available session IDs.
    public func listSessions() -> [String] {
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: sessionsDir) else {
            return []
        }
        return files
            .filter { $0.hasSuffix(".jsonl") }
            .map { String($0.dropLast(6)) }  // Remove .jsonl
            .sorted()
    }
}
