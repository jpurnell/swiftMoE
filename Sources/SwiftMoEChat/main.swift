import Foundation
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

func sendChatRequest(url: String, prompt: String, maxTokens: Int) {
    // Build request body
    let body: [String: Any] = [
        "model": "flash-moe",
        "messages": [["role": "user", "content": prompt]],
        "max_tokens": maxTokens,
        "stream": true
    ]

    guard let jsonData = try? JSONSerialization.data(withJSONObject: body),
          let requestURL = URL(string: "\(url)/v1/chat/completions") else {
        print("[error] Failed to create request")
        return
    }

    var request = URLRequest(url: requestURL)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = jsonData

    // Synchronous request with streaming response
    let semaphore = DispatchSemaphore(value: 0)

    let task = URLSession.shared.dataTask(with: request) { data, response, error in
        defer { semaphore.signal() }

        if let error = error {
            print("\n[error] \(error.localizedDescription)")
            return
        }

        guard let data = data,
              let text = String(data: data, encoding: .utf8) else {
            print("\n[error] No response data")
            return
        }

        // Parse SSE events
        for line in text.components(separatedBy: "\n") {
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6))

            if payload == "[DONE]" { break }

            if let eventData = payload.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: eventData) as? [String: Any],
               let choices = json["choices"] as? [[String: Any]],
               let delta = choices.first?["delta"] as? [String: Any],
               let content = delta["content"] as? String {
                print(content, terminator: "")
                fflush(stdout)
            }
        }
        print()  // Newline after response
    }
    task.resume()
    semaphore.wait()
}

func main() {
    let chatConfig = parseArgs()
    let session = SessionStore(sessionID: chatConfig.sessionID)

    print("Flash-MoE Chat")
    print("Server: \(chatConfig.serverURL)")
    print("Session: \(session.sessionID)")
    print("Type /quit to exit, /sessions to list saved sessions\n")

    // Set up linenoise
    linenoiseSetMultiLine(1)
    linenoiseHistorySetMaxLen(100)

    // Load history
    let home = ProcessInfo.processInfo.environment["HOME"] ?? "/tmp"
    let historyPath = "\(home)/.flash-moe/history"
    try? FileManager.default.createDirectory(
        atPath: "\(home)/.flash-moe", withIntermediateDirectories: true)
    linenoiseHistoryLoad(historyPath)

    // Chat loop
    while true {
        guard let cLine = linenoise("You> ") else {
            break  // EOF / Ctrl-D
        }
        let line = String(cString: cLine)
        free(cLine)

        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { continue }

        // Handle commands
        if trimmed == "/quit" || trimmed == "/exit" {
            break
        }
        if trimmed == "/clear" {
            print("[cleared session]")
            continue
        }
        if trimmed == "/sessions" {
            let sessions = session.listSessions()
            if sessions.isEmpty {
                print("[no saved sessions]")
            } else {
                for s in sessions { print("  \(s)") }
            }
            continue
        }

        // Add to history
        linenoiseHistoryAdd(trimmed)
        linenoiseHistorySave(historyPath)

        // Save user message
        session.appendMessage(role: "user", content: trimmed)

        // Send to server and stream response
        print("Assistant> ", terminator: "")
        fflush(stdout)

        sendChatRequest(url: chatConfig.serverURL, prompt: trimmed, maxTokens: chatConfig.maxTokens)

        // Save assistant response (simplified — full impl would capture streamed tokens)
        session.appendMessage(role: "assistant", content: "(streamed response)")
    }

    print("\nGoodbye!")
}

main()
