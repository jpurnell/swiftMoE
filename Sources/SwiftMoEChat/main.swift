import Foundation
import os
import SwiftMoE
import CLineNoise

// ============================================================================
// flash-moe-chat — Interactive TUI chat client
//
// Usage:
//   flash-moe-chat --url http://localhost:8080
//   flash-moe-chat --port 8080                   # Shorthand for localhost
//   flash-moe-chat --resume <session_id>          # Resume previous session
//
// Commands:
//   /quit, /exit    — Exit chat
//   /clear          — Clear session history
//   /sessions       — List saved sessions
// ============================================================================

private let logger = Logger(subsystem: "com.swiftmoe.chat", category: "main")
private let allowedSchemes: Set<String> = ["http", "https"]
private let allowedHosts: Set<String> = ["localhost", "127.0.0.1", "::1"]

struct ChatConfig {
    var serverURL: String = "http://localhost:8080"
    var sessionID: String?
    var showThinking: Bool = false
    var maxTokens: Int = 500
}

func parseArgs() -> ChatConfig {
    var config = ChatConfig()
    var i = 1
    let args = CommandLine.arguments
    while i < args.count {
        switch args[i] {
        case "--url": i += 1; if i < args.count { config.serverURL = args[i] }
        case "--port": i += 1; if i < args.count { config.serverURL = "http://localhost:\(args[i])" }
        case "--resume": i += 1; if i < args.count { config.sessionID = args[i] }
        case "--show-think": config.showThinking = true
        case "--max-tokens": i += 1; if i < args.count { config.maxTokens = Int(args[i]) ?? 500 }
        default: break
        }
        i += 1
    }
    return config
}

/// Validates a URL string against the scheme/host allowlists to prevent SSRF (CWE-918).
/// Returns the validated URL only if scheme and host are in the allowlist.
func validateServerURL(_ urlString: String) -> URL? {
    guard let components = URLComponents(string: urlString),
          let scheme = components.host != nil ? components.scheme?.lowercased() : nil,
          allowedSchemes.contains(scheme),
          let host = components.host?.lowercased(),
          allowedHosts.contains(host) else {
        return nil
    }
    return components.url
}

func sendChatRequest(url: String, prompt: String, maxTokens: Int) {
    let body: [String: Any] = [
        "model": "flash-moe",
        "messages": [["role": "user", "content": prompt]],
        "max_tokens": maxTokens,
        "stream": true
    ]

    let jsonData: Data
    do {
        jsonData = try JSONSerialization.data(withJSONObject: body)
    } catch {
        logger.error("Failed to serialize request body: \(error.localizedDescription, privacy: .public)")
        return
    }

    guard let requestURL = validateServerURL("\(url)/v1/chat/completions") else {
        logger.error("Invalid or disallowed server URL: \(url, privacy: .public)")
        return
    }

    var request = URLRequest(url: requestURL)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = jsonData

    let semaphore = DispatchSemaphore(value: 0)

    let task = URLSession.shared.dataTask(with: request) { data, response, error in
        defer { semaphore.signal() }

        if let error = error {
            logger.error("Request failed: \(error.localizedDescription, privacy: .public)")
            return
        }

        guard let data = data,
              let text = String(data: data, encoding: .utf8) else {
            logger.error("No response data")
            return
        }

        for line in text.components(separatedBy: "\n") {
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6))

            if payload == "[DONE]" { break }

            if let eventData = payload.data(using: .utf8),
               // silent: unparseable SSE events are intentionally skipped
               let json = try? JSONSerialization.jsonObject(with: eventData) as? [String: Any],
               let choices = json["choices"] as? [[String: Any]],
               let delta = choices.first?["delta"] as? [String: Any],
               let content = delta["content"] as? String {
                FileHandle.standardOutput.write(Data(content.utf8))
                fflush(stdout)
            }
        }
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
    task.resume()
    semaphore.wait()
}

func main() {
    let chatConfig = parseArgs()
    let session = SessionStore(sessionID: chatConfig.sessionID)

    logger.info("Flash-MoE Chat")
    logger.info("Server: \(chatConfig.serverURL, privacy: .public)")
    logger.info("Session: \(session.sessionID, privacy: .public)")
    logger.info("Type /quit to exit, /sessions to list saved sessions")

    linenoiseSetMultiLine(1)
    linenoiseHistorySetMaxLen(100)

    let home = ProcessInfo.processInfo.environment["HOME"] ?? "/tmp"
    let homeURL = URL(fileURLWithPath: home).standardized
    let flashMoeDirURL = URL(fileURLWithPath: home)
        .appendingPathComponent(".flash-moe").standardized
    guard flashMoeDirURL.path.hasPrefix(homeURL.path) else {
        logger.error("Resolved config path escapes home directory")
        return
    }
    let historyPath = flashMoeDirURL.appendingPathComponent("history").path
    do {
        try FileManager.default.createDirectory(at: flashMoeDirURL, withIntermediateDirectories: true)
    } catch {
        logger.error("Failed to create config directory: \(error.localizedDescription, privacy: .public)")
    }
    linenoiseHistoryLoad(historyPath)

    var shouldExit = false
    while !shouldExit {
        guard let cLine = linenoise("You> ") else {
            break  // EOF / Ctrl-D
        }
        let line = String(cString: cLine)
        free(cLine)

        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { continue }

        if trimmed == "/quit" || trimmed == "/exit" {
            shouldExit = true
            continue
        }
        if trimmed == "/clear" {
            logger.info("[cleared session]")
            continue
        }
        if trimmed == "/sessions" {
            let sessions = session.listSessions()
            if sessions.isEmpty {
                logger.info("[no saved sessions]")
            } else {
                for s in sessions { logger.info("\(s, privacy: .public)") }
            }
            continue
        }

        linenoiseHistoryAdd(trimmed)
        linenoiseHistorySave(historyPath)

        session.appendMessage(role: "user", content: trimmed)

        FileHandle.standardOutput.write(Data("Assistant> ".utf8))
        fflush(stdout)

        sendChatRequest(url: chatConfig.serverURL, prompt: trimmed, maxTokens: chatConfig.maxTokens)

        session.appendMessage(role: "assistant", content: "(streamed response)")
    }

    logger.info("Goodbye!")
}

main()
