import Foundation
import os
import Testing
@testable import SwiftMoE
@testable import SwiftMoEHTTP

// MARK: - The wire characterization

// This file records what the server puts on the wire for a fixed table of raw requests, and
// replays the table against whatever server `makeWireTarget` builds.
//
// The expected values in `Fixtures/exchanges.json` are the bytes the original, hand-rolled
// `HTTPServer` wrote. They were recorded once (`RECORD=1 swift test --filter WireCharacterization`)
// and are never regenerated to make a later server pass: a difference is either a regression, or
// it is listed on its exchange as an intended difference with the reason.
//
// Two places name the server under test and nothing else does: `makeWireTarget`, and the
// adapters in "Handlers" below, which turn a `WireHandler` into what the server calls.

// MARK: - The server under test (the one place that constructs it)

/// What a scenario needs from a running server, whichever implementation is behind it.
final class WireTarget: Sendable {
    private let running: RunningServer

    fileprivate init(running: RunningServer) {
        self.running = running
    }

    /// The port the server listens on, on loopback.
    var port: UInt16 { UInt16(running.bound.port) }
    /// What the handler was called with, in order.
    var handlerCalls: [HandlerCall] { running.handlerCalls }

    func request(
        _ method: String = "POST",
        _ target: String = "/v1/chat/completions",
        host: String? = nil,
        key: String? = RunningServer.key,
        origin: String? = nil,
        extraHeaders: [String] = [],
        body: String? = #"{"messages":[{"role":"user","content":"hi"}]}"#
    ) -> [UInt8] {
        running.request(method, target, host: host, key: key, origin: origin,
                        extraHeaders: extraHeaders, body: body)
    }

    /// Sends `bytes` on a fresh connection and returns everything the server wrote back.
    func exchange(_ bytes: [UInt8]) throws -> String { try running.exchange(bytes) }

    /// Opens a connection and sends `bytes`, leaving the response unread.
    func open(sending bytes: [UInt8]) throws -> ClientSocket {
        let client = try ClientSocket(port: port)
        client.send(bytes)
        return client
    }

    /// Waits for the server to report that a request began to wait for the model.
    func awaitQueued() -> Bool { running.events.awaitNext(.requestQueued) }
    /// Waits for the server to report that a waiting request's client left.
    func awaitAbandoned() -> Bool { running.events.awaitNext(.queuedRequestAbandoned) }

    func shutdown() { running.shutdown() }
}

/// How a handler behaves, described without naming the server's types.
enum WireHandler: Sendable {
    /// Stream headers, then `[DONE]`.
    case emptyStream
    /// Headers, one delta per string, a finish chunk, `[DONE]`: the shape of the real handler.
    case tokens([String])
    /// Headers and deltas, then the handler returns as if generation had failed.
    case truncated([String])
    /// Holds the model for a request whose prompt is `hold`.
    case hold(Hold)
    /// Streams until it sees that its client has gone.
    case streamUntilClientLeaves(LeaveProbe)
    /// The demo model, driven by the real generator.
    case demoModel(DemoModel)
}

/// A server's configuration, in terms the scenarios share.
struct WireProfile: Sendable {
    var authenticated = true
    var allowedOrigins: [String] = []
    var limits = InferenceLimits()
    var handler = WireHandler.emptyStream
    /// The demo tokenizer wraps ids into the vocabulary; the default is one token per byte.
    var wrapsTokensIntoVocabulary: Int?
}

/// Builds and starts a server on a loopback port for `profile`.
func makeWireTarget(_ profile: WireProfile) throws -> WireTarget {
    var tokenizer = RunningServer.byteTokenizer
    if let vocabulary = profile.wrapsTokensIntoVocabulary {
        tokenizer = { prompt in
            let tokens = prompt.utf8.map { Int($0) % vocabulary }
            return tokens.isEmpty ? [0] : tokens
        }
    }
    let running = try RunningServer(authenticated: profile.authenticated,
                                    allowedOrigins: profile.allowedOrigins,
                                    limits: profile.limits,
                                    tokenizer: tokenizer,
                                    respond: profile.handler.respond)
    return WireTarget(running: running)
}

// MARK: - Handlers (old-server adapters)

extension WireHandler {
    /// What the server calls for each request that passed every check.
    fileprivate var respond: StubResponse {
        switch self {
        case .emptyStream:
            return RunningServer.emptyStream
        case .tokens(let tokens):
            return { _, _, emit in
                for token in tokens where !emit(token) { return }
            }
        case .truncated(let tokens):
            return { _, _, emit in
                for token in tokens where !emit(token) { return }
                throw WireHandlerFailure.generationFailed
            }
        case .hold(let hold):
            return hold.respond
        case .streamUntilClientLeaves(let probe):
            return probe.respond
        case .demoModel(let model):
            return model.respond
        }
    }
}

/// What a handler throws to end its stream as a failed generation does.
enum WireHandlerFailure: Error {
    case generationFailed
}

/// A handler that streams until it sees its client leave, and says what it saw.
final class LeaveProbe: Sendable {
    /// Tokens the handler would send if its client never left.
    static let cap = 500
    private static let allowance: DispatchTimeInterval = .seconds(30)

    private let streaming = DispatchSemaphore(value: 0)
    private let finished = DispatchSemaphore(value: 0)
    private let state = OSAllocatedUnfairLock(initialState: (sent: 0, sawDisconnect: false))

    var respond: StubResponse {
        { [self] _, shouldContinue, emit in
            var sent = 0
            while sent < Self.cap {
                guard emit("x") else { break }
                sent += 1
                if sent == 1 { streaming.signal() }
                // Paces the loop; the client's departure is what ends it.
                Thread.sleep(forTimeInterval: 0.02)
                if !shouldContinue() { break }
            }
            let result = (sent: sent, sawDisconnect: !shouldContinue())
            state.withLock { $0 = result }
            finished.signal()
        }
    }

    /// Waits until the first token has been written.
    func awaitStreaming() -> Bool { streaming.wait(timeout: .now() + Self.allowance) == .success }

    /// Waits for the handler to return, then reports what it saw.
    func awaitFinished() -> (sent: Int, sawDisconnect: Bool)? {
        guard finished.wait(timeout: .now() + Self.allowance) == .success else { return nil }
        return state.withLock { $0 }
    }
}

/// The demo model of `swift-moe-server --demo`: `ModelConfig.tiny` with zero weights, so the
/// output is a fixed function of the prompt. Needs the Metal shaders.
final class DemoModel: Sendable {
    // Justification: the server runs the handler one request at a time, so the generator and its buffers are never used from two threads at once.
    private final class Rig: @unchecked Sendable {
        let config = ModelConfig.tiny
        let fixtures: SyntheticFixtures.FixturePaths
        let generator: TokenGenerator
        let weightFile: WeightFile
        let layerWeights: [LayerWeightPointers]
        var expertFDs: [Int32] = []

        init() throws {
            fixtures = try SyntheticFixtures.create(config: config, numLayers: config.numLayers,
                                                    numExperts: config.numExperts)
            let shaderURL = try #require(ShaderLibraryTests.shaderURL)
            let context = try MetalContext(config: config, shaderPath: shaderURL.path, use2Bit: false)
            weightFile = try WeightFile(weightsPath: fixtures.weightsPath, manifestPath: fixtures.manifestPath)
            layerWeights = LayerWeightCacheBuilder.build(from: weightFile, config: config)
            for path in fixtures.expertPaths {
                let descriptor = open(path, O_RDONLY)
                try #require(descriptor >= 0)
                expertFDs.append(descriptor)
            }
            context.setWeights(weightFile.data, size: weightFile.size)
            generator = TokenGenerator(context: context, config: config,
                                       activeExperts: config.numExpertsPerToken)
        }

        func close() {
            expertFDs.forEach { Darwin.close($0) }
            SyntheticFixtures.cleanup(fixtures)
        }
    }

    private let rig: Rig

    init() throws {
        rig = try Rig()
    }

    /// The tiny model's vocabulary, which the tokenizer wraps ids into.
    var vocabularySize: Int { rig.config.vocabSize }
    /// How many positions the generator holds.
    var sequenceLength: Int { rig.generator.maxSequenceLength }

    func close() { rig.close() }

    /// The generator of `Sources/SwiftMoEServer/main.swift`.
    var respond: StubResponse {
        { [rig] request, shouldContinue, emit in
            try rig.generator.generate(
                promptTokens: request.promptTokens,
                maxTokens: request.maxTokens,
                weightFile: rig.weightFile,
                expertFDs: rig.expertFDs,
                layerWeights: rig.layerWeights,
                use2Bit: false,
                shouldContinue: shouldContinue,
                onToken: { token in emit(String(UnicodeScalar(UInt8(token % 128)))) })
        }
    }
}

// MARK: - What is recorded

/// One response, reduced to what a client sees and normalised for what varies run to run.
struct Recorded: Codable, Equatable, Sendable {
    /// The status line, or `nil` when the server wrote nothing.
    var status: String?
    /// Header fields in the order written, `Date` removed.
    var headers: [[String]]
    /// The body, with the per-request id in stream chunks replaced by `req_<ID>`.
    var body: String
    /// What else the scenario established: what the handler was called with, whether it ran.
    var facts: [String]

    private static let idPattern = "req_[0-9A-Fa-f-]{36}"

    /// Parses everything a server wrote on one connection.
    static func parse(_ raw: String, facts: [String] = []) -> Recorded {
        guard !raw.isEmpty else { return Recorded(status: nil, headers: [], body: "", facts: facts) }
        let separator = "\r\n\r\n"
        var head = raw
        var body = ""
        if let range = raw.range(of: separator) {
            head = String(raw[..<range.lowerBound])
            body = String(raw[range.upperBound...])
        }
        let lines = head.split(separator: "\r\n" as Character, omittingEmptySubsequences: false).map(String.init)
        let fields: [[String]] = lines.dropFirst().compactMap { line in
            guard let colon = line.range(of: ": ") else { return [line, ""] }
            return [String(line[..<colon.lowerBound]), String(line[colon.upperBound...])]
        }.filter { $0[0].lowercased() != "date" }
        let chunked = fields.contains { $0[0].lowercased() == "transfer-encoding" && $0[1].lowercased() == "chunked" }
        if chunked { body = Canonical.dechunk(body) }
        let normalised = body.replacingOccurrences(of: idPattern, with: "req_<ID>", options: .regularExpression)
        return Recorded(status: lines.first, headers: fields, body: normalised, facts: facts)
    }

    /// A prompt as it reads in a fact: short ones whole, long ones as their length and ends.
    private static func abbreviated(_ prompt: String) -> String {
        let shown = 20
        guard prompt.count > 2 * shown else { return prompt }
        return "\(prompt.prefix(shown))...\(prompt.suffix(shown)) (\(prompt.count) characters)"
    }

    /// How the handler was called, as sentences that read the same across servers.
    static func facts(of calls: [HandlerCall]) -> [String] {
        calls.isEmpty
            ? ["handler not called"]
            : calls.map { "handler called: prompt=\(abbreviated($0.prompt)) maxTokens=\($0.maxTokens)" }
    }
}

/// One row of the table.
struct WireExchange: Sendable, CustomTestStringConvertible {
    let name: String
    /// Why the new server is expected to answer this differently, when it is.
    let expectedToChange: String?
    /// Header names (lowercased) the new server is expected to change, with the reason. The
    /// body, the status and every other header still have to match.
    let headerChanges: [String: String]
    /// Whether the scenario needs the Metal shaders.
    let needsMetal: Bool
    /// A difference found when the exchange was first replayed against the new server, accepted as
    /// intended (2026-10-30) with this reason. The replay lists it and does not fail on it; any
    /// difference on an exchange that has neither this nor `expectedToChange` fails.
    let intended: String?
    /// Whether the exchange's handler is the empty stub. The old stub wrote only `[DONE]`; the
    /// service finishes every completed stream with a `finish_reason` chunk first, so that chunk
    /// is taken out of the new body before it is compared. The real handler always wrote the chunk.
    let stubStream: Bool
    let run: @Sendable () throws -> Recorded

    var testDescription: String { name }
}

/// What is stored in `Fixtures/exchanges.json`.
struct RecordedExchange: Codable, Equatable {
    var name: String
    var expectedToChange: String?
    var recorded: Recorded
}

struct ExchangeFixture: Codable {
    var note: String
    var exchanges: [RecordedExchange]
}

// MARK: - Comparing

enum WireComparison {

    /// Headers a server may add or drop without it being a difference worth reporting.
    static let transportHeaders: Set<String> = ["date", "server"]

    /// Every way `actual` differs from `expected`, as sentences.
    ///
    /// Header order is not compared: it is the transport's to choose. Header names compare
    /// case-insensitively.
    static func differences(expected: Recorded, actual: Recorded) -> [String] {
        var found: [String] = []
        if expected.status != actual.status {
            found.append("status line: expected \(expected.status ?? "nothing"), got \(actual.status ?? "nothing")")
        }
        let expectedHeaders = normalised(expected.headers)
        let actualHeaders = normalised(actual.headers)
        for header in Set(expectedHeaders).subtracting(actualHeaders).sorted() {
            found.append("header missing: \(header)")
        }
        for header in Set(actualHeaders).subtracting(expectedHeaders).sorted() {
            found.append("header added: \(header)")
        }
        if expected.body != actual.body {
            found.append("body: expected \(expected.body.debugDescription), got \(actual.body.debugDescription)")
        }
        if expected.facts != actual.facts {
            found.append("facts: expected \(expected.facts), got \(actual.facts)")
        }
        return found
    }

    private static func normalised(_ headers: [[String]]) -> [String] {
        headers.compactMap { field in
            guard field.count == 2, !transportHeaders.contains(field[0].lowercased()) else { return nil }
            return "\(field[0].lowercased()): \(field[1])"
        }
    }

    /// The header name a difference line is about, for matching against `headerChanges`.
    static func headerName(of difference: String) -> String? {
        for prefix in ["header missing: ", "header added: "] where difference.hasPrefix(prefix) {
            return difference.dropFirst(prefix.count).split(separator: ":", maxSplits: 1).first.map(String.init)
        }
        return nil
    }
}

// MARK: - The table

enum WireExchanges {

    /// The headers expected to change on every stream. The second is not in the proposal's
    /// section 7: the old stream has no length and ends by closing the connection, and the
    /// kit's event stream is chunk-encoded.
    static let cacheControl = [
        "cache-control": "no-cache becomes no-store on the stream (proposal section 7)",
        "transfer-encoding": "the kit chunk-encodes the stream; the old one is close-delimited (not in proposal section 7)",
    ]
    private static let allowedOrigin = "https://localhost:3000"

    /// A body with a prompt of `hi` and the given extra members.
    private static func chat(_ members: String = "") -> String {
        let extra = members.isEmpty ? "" : "," + members
        return #"{"messages":[{"role":"user","content":"hi"}]\#(extra)}"#
    }

    private static func chat(prompt: String, members: String = "") -> String {
        let extra = members.isEmpty ? "" : "," + members
        return #"{"messages":[{"role":"user","content":"\#(prompt)"}]\#(extra)}"#
    }

    /// A row that sends one request to a freshly built server and records the answer.
    private static func plain(
        _ name: String,
        intended: String? = nil,
        profile: WireProfile = WireProfile(),
        expectedToChange: String? = nil,
        headerChanges: [String: String] = [:],
        request: @escaping @Sendable (WireTarget) -> [UInt8]
    ) -> WireExchange {
        let stub: Bool
        if case .emptyStream = profile.handler { stub = true } else { stub = false }
        return WireExchange(name: name, expectedToChange: expectedToChange, headerChanges: headerChanges,
                     needsMetal: false, intended: intended, stubStream: stub) {
            let target = try makeWireTarget(profile)
            defer { target.shutdown() }
            let raw = try target.exchange(request(target))
            return Recorded.parse(raw, facts: Recorded.facts(of: target.handlerCalls))
        }
    }

    /// A row with its own scenario.
    private static func scenario(
        _ name: String,
        needsMetal: Bool = false,
        expectedToChange: String? = nil,
        intended: String? = nil,
        stubStream: Bool = false,
        headerChanges: [String: String] = [:],
        run: @escaping @Sendable () throws -> Recorded
    ) -> WireExchange {
        WireExchange(name: name, expectedToChange: expectedToChange, headerChanges: headerChanges,
                     needsMetal: needsMetal, intended: intended, stubStream: stubStream, run: run)
    }

    private static func profile(
        origins: [String] = [],
        handler: WireHandler = .emptyStream,
        limits adjust: (inout InferenceLimits) -> Void = { _ in }
    ) -> WireProfile {
        var profile = WireProfile()
        profile.allowedOrigins = origins
        profile.handler = handler
        adjust(&profile.limits)
        return profile
    }

    private static func raw(_ text: String) -> [UInt8] { Array(text.utf8) }

    /// The `finish_reason` event the service writes before `[DONE]` on every completed stream.
    static let stubFinishEvent =
        #"data: {"id":"req_<ID>","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"# + "\n\n"

    /// Every exchange that needs no GPU.
    static var all: [WireExchange] {
        ladder + bodies + budget + streams + capacity + changes
    }

    // MARK: The decision ladder

    private static let ladder: [WireExchange] = [
        plain("version_505", intended: "The kit's HTTP parser answers 400 Malformed HTTP request for an HTTP/2.0 request line; the old server answered 505 (not in proposal sections 3 or 7)") { target in
            raw("POST /v1/chat/completions HTTP/2.0\r\nHost: 127.0.0.1:\(target.port)\r\n\r\n")
        },
        plain("host_421_wrong_host") { target in target.request(host: "evil.example") },
        plain("host_421_wrong_port") { target in target.request(host: "127.0.0.1:1") },
        plain("host_421_missing_host", intended: "The transport refuses a request with no Host header as 400 before any middleware runs; the old server answered 421 (not in the proposal)") { target in target.request(host: "") },
        plain("origin_403_no_allowlist") { target in target.request(origin: "https://evil.example") },
        plain("origin_403_not_on_allowlist", profile: profile(origins: [allowedOrigin])) { target in
            target.request(origin: "https://evil.example")
        },
        plain("preflight_204_allowed_origin", intended: "The kit's CORS layer sends no Content-Length: 0 on the 204 preflight (not in the proposal)", profile: profile(origins: [allowedOrigin])) { target in
            target.request("OPTIONS", key: nil, origin: allowedOrigin,
                           extraHeaders: ["Access-Control-Request-Method: POST"], body: nil)
        },
        plain("preflight_403_disallowed_origin", profile: profile(origins: [allowedOrigin])) { target in
            target.request("OPTIONS", key: nil, origin: "https://evil.example",
                           extraHeaders: ["Access-Control-Request-Method: POST"], body: nil)
        },
        plain("preflight_404_other_route", intended: "The kit answers a preflight from an allowed origin for any path (204); the old server answered 404 for a path that is not the route (not in the proposal)", profile: profile(origins: [allowedOrigin])) { target in
            target.request("OPTIONS", "/other", key: nil, origin: allowedOrigin, body: nil)
        },
        plain("auth_401_missing") { target in target.request(key: nil) },
        plain("auth_401_wrong_key") { target in target.request(key: RunningServer.wrongKey) },
        plain("auth_401_wrong_scheme") { target in
            target.request(key: nil, extraHeaders: ["Authorization: Basic \(RunningServer.key)"])
        },
        plain("auth_401_bearer_without_token") { target in
            target.request(key: nil, extraHeaders: ["Authorization: Bearer"])
        },
        plain("auth_200_lowercase_scheme_accepted") { target in
            target.request(key: nil, extraHeaders: ["Authorization: bearer \(RunningServer.key)"])
        },
        plain("auth_401_malformed_key_short") { target in target.request(key: "short") },
        plain("auth_200_key_with_trailing_space_trimmed") { target in target.request(key: RunningServer.key + " ") },
        plain("auth_401_beats_404", profile: WireProfile()) { target in target.request("POST", "/other", key: nil) },
        plain("route_404_other_path") { target in target.request("POST", "/other") },
        plain("route_404_trailing_slash") { target in target.request("POST", "/v1/chat/completions/") },
        plain("route_200_query_string_ignored") { target in
            target.request("POST", "/v1/chat/completions?stream=true")
        },
        plain("route_405_get") { target in target.request("GET", body: nil) },
        plain("route_405_put") { target in target.request("PUT") },
        plain("route_405_options_without_origin") { target in target.request("OPTIONS", body: nil) },
        plain("transfer_encoding_501_chunked",
              expectedToChange: "A chunked upload may be accepted or refused by the kit; the old 501 was a limit of the hand-rolled reader (proposal section 7, spike S5)") { target in
            target.request(extraHeaders: ["Transfer-Encoding: chunked"], body: nil)
        },
        plain("length_411_missing") { target in target.request(body: nil) },
        plain("length_400_not_a_number") { target in
            target.request(extraHeaders: ["Content-Length: ten"], body: nil)
        },
        plain("length_400_negative") { target in
            target.request(extraHeaders: ["Content-Length: -1"], body: nil)
        },
        plain("body_413_declared_over_limit", intended: "The 413 status line reads Payload Too Large (the transport's reason phrase); the old one read Content Too Large") { target in
            target.request(extraHeaders: ["Content-Length: 65537"], body: nil)
        },
        scenario("headers_431_too_large", intended: "The kit refuses a header limit below 1 KiB; the old server's test used 512 bytes, so this input cannot be replayed") {
            let target: WireTarget
            do {
                target = try makeWireTarget(profile { $0.maxHeaderBytes = 512 })
            } catch let error as ServiceError {
                return Recorded(status: nil, headers: [], body: "", facts: ["the server refused to start: \(error)"])
            }
            defer { target.shutdown() }
            let raw = try target.exchange(target.request(extraHeaders: ["X-Padding: \(String(repeating: "x", count: 600))"]))
            return Recorded.parse(raw, facts: Recorded.facts(of: target.handlerCalls))
        },
        plain("request_line_400_garbage") { _ in raw("GARBAGE\r\n\r\n") },
        scenario("timeout_408_head_never_finishes") {
            let target = try makeWireTarget(profile { $0.readDeadline = .milliseconds(300) })
            defer { target.shutdown() }
            let client = try target.open(sending: raw("POST /v1/chat/completions HTTP/1.1\r\nHost: "))
            defer { client.close() }
            return Recorded.parse(client.readToEnd(), facts: Recorded.facts(of: target.handlerCalls))
        },
        scenario("timeout_408_body_never_finishes") {
            let target = try makeWireTarget(profile { $0.readDeadline = .milliseconds(300) })
            defer { target.shutdown() }
            let client = try target.open(sending: target.request(extraHeaders: ["Content-Length: 500"], body: nil))
            defer { client.close() }
            return Recorded.parse(client.readToEnd(), facts: Recorded.facts(of: target.handlerCalls))
        },
        scenario("body_400_truncated_by_client_close", intended: "The transport closes the connection without a response when the body ends early; the old server answered 400 (not in the proposal)") {
            let target = try makeWireTarget(WireProfile())
            defer { target.shutdown() }
            let client = try target.open(sending: target.request(extraHeaders: ["Content-Length: 500"], body: nil)
                                            + raw(#"{"messages":"#))
            defer { client.close() }
            shutdown(client.descriptor, SHUT_WR)
            return Recorded.parse(client.readToEnd(), facts: Recorded.facts(of: target.handlerCalls))
        },
    ]

    // MARK: Body faults and token counts

    /// One request whose body carries `members` beside a prompt of `hi`.
    private static func tokens(_ name: String, _ members: String) -> WireExchange {
        plain(name) { target in target.request(body: chat(members)) }
    }

    private static let bodies: [WireExchange] = [
        plain("body_400_not_json") { target in target.request(body: "this is not json") },
        plain("body_400_json_array") { target in target.request(body: "[1,2,3]") },
        plain("body_400_empty") { target in target.request(body: "") },
        plain("body_400_json_string") { target in target.request(body: #""hi""#) },
        plain("body_200_no_messages") { target in target.request(body: "{}") },
        tokens("max_tokens_200_absent_uses_default", ""),
        tokens("max_tokens_200_whole", #""max_tokens":7"#),
        tokens("max_tokens_200_whole_written_as_float", #""max_tokens":64.0"#),
        tokens("max_tokens_200_whole_written_with_exponent", #""max_tokens":1e1"#),
        tokens("max_tokens_200_null_is_absent", #""max_tokens":null"#),
        tokens("max_tokens_400_string", #""max_tokens":"ten""#),
        tokens("max_tokens_400_boolean", #""max_tokens":true"#),
        tokens("max_tokens_400_fractional", #""max_tokens":1.5"#),
        tokens("max_tokens_400_negative", #""max_tokens":-1"#),
        tokens("max_tokens_400_zero", #""max_tokens":0"#),
        tokens("max_tokens_400_above_ceiling", #""max_tokens":8193"#),
        tokens("max_tokens_400_astronomical", #""max_tokens":1e30"#),
        tokens("max_tokens_400_array", #""max_tokens":[1]"#),
        tokens("max_completion_tokens_200_whole", #""max_completion_tokens":9"#),
        tokens("max_completion_tokens_400_fractional", #""max_completion_tokens":2.5"#),
        tokens("max_completion_tokens_400_boolean", #""max_completion_tokens":false"#),
        tokens("max_completion_tokens_400_zero", #""max_completion_tokens":0"#),
        tokens("max_completion_tokens_400_above_ceiling", #""max_completion_tokens":9000"#),
        tokens("precedence_max_completion_tokens_wins", #""max_tokens":10,"max_completion_tokens":20"#),
        tokens("precedence_400_bad_max_tokens_beside_good_completion", #""max_tokens":0,"max_completion_tokens":20"#),
        tokens("precedence_400_bad_completion_beside_good_max_tokens", #""max_tokens":10,"max_completion_tokens":0"#),
        plain("prompt_is_last_message") { target in
            target.request(body: #"{"messages":[{"role":"user","content":"first"},{"role":"user","content":"last"}]}"#)
        },
        plain("prompt_non_string_content_is_empty") { target in
            target.request(body: #"{"messages":[{"role":"user","content":42}]}"#)
        },
    ]

    // MARK: The sequence budget

    /// Limits small enough to cross the boundary with a short prompt: 64 positions in all.
    private static let smallSequence = profile { limits in
        limits.maxSequenceTokens = 64
        limits.maxCompletionTokens = 64
        limits.defaultCompletionTokens = 10
    }

    private static let budget: [WireExchange] = [
        plain("budget_200_exactly_at_limit", profile: smallSequence) { target in
            target.request(body: chat(prompt: String(repeating: "p", count: 54), members: #""max_tokens":10"#))
        },
        plain("budget_400_one_over_limit", profile: smallSequence) { target in
            target.request(body: chat(prompt: String(repeating: "p", count: 55), members: #""max_tokens":10"#))
        },
        plain("budget_200_default_limits_exactly_at_limit") { target in
            target.request(body: chat(prompt: String(repeating: "p", count: 8092)))
        },
        plain("budget_400_default_limits_one_over_limit") { target in
            target.request(body: chat(prompt: String(repeating: "p", count: 8093)))
        },
        plain("budget_400_ceiling_completion_leaves_no_room") { target in
            target.request(body: chat(#""max_tokens":8192"#))
        },
        plain("budget_400_default_completion_is_counted", profile: smallSequence) { target in
            target.request(body: chat(prompt: String(repeating: "p", count: 55)))
        },
    ]

    // MARK: Streams

    private static let streams: [WireExchange] = [
        plain("sse_tokens_complete_stream", profile: profile(handler: .tokens(["Hel", "lo ", "\"w\"\n\u{1}"])),
              headerChanges: cacheControl) { target in
            target.request(body: chat(prompt: "greet", members: #""max_tokens":3"#))
        },
        plain("sse_tokens_cors_allowed_origin",
              profile: profile(origins: [allowedOrigin], handler: .tokens(["ok"])),
              headerChanges: cacheControl) { target in
            target.request(origin: allowedOrigin)
        },
        plain("sse_vary_origin_without_origin_header",
              profile: profile(origins: [allowedOrigin], handler: .tokens(["ok"])),
              headerChanges: cacheControl) { target in
            target.request()
        },
        plain("sse_truncated_when_generation_fails", profile: profile(handler: .truncated(["par", "tial"])),
              headerChanges: cacheControl) { target in
            target.request()
        },
        plain("sse_stub_empty_stream", headerChanges: cacheControl) { target in target.request() },
        plain("sse_unauthenticated_loopback", profile: unauthenticated, headerChanges: cacheControl) { target in
            target.request(key: nil)
        },
        scenario("sse_demo_model_complete_stream", needsMetal: true, headerChanges: cacheControl) {
            let model = try DemoModel()
            defer { model.close() }
            var demo = WireProfile()
            demo.handler = .demoModel(model)
            demo.wrapsTokensIntoVocabulary = model.vocabularySize
            demo.limits.maxSequenceTokens = model.sequenceLength
            demo.limits.maxCompletionTokens = min(demo.limits.maxCompletionTokens, model.sequenceLength)
            demo.limits.defaultCompletionTokens = min(demo.limits.defaultCompletionTokens, demo.limits.maxCompletionTokens)
            let target = try makeWireTarget(demo)
            defer { target.shutdown() }
            let raw = try target.exchange(target.request(body: chat(#""max_tokens":3"#)))
            return Recorded.parse(raw, facts: Recorded.facts(of: target.handlerCalls))
        },
    ]

    private static var unauthenticated: WireProfile {
        var profile = WireProfile()
        profile.authenticated = false
        profile.handler = .tokens(["ok"])
        return profile
    }

    // MARK: Capacity, queueing, and clients that leave

    private static let capacity: [WireExchange] = [
        scenario("busy_503_past_max_connections", intended: "KNOWN REGRESSION, caused by SwiftHTTPService 0.5.0 and not worked around here: the transport closes a connection past the cap without any response, so there is no 503 and no Retry-After (spike S5; proposal section 3 said to verify)") {
            let target = try makeWireTarget(profile { $0.maxConnections = 1 })
            defer { target.shutdown() }
            let holder = try ClientSocket(port: target.port)
            defer { holder.close() }
            let raw = try target.exchange(target.request())
            return Recorded.parse(raw, facts: Recorded.facts(of: target.handlerCalls))
        },
        scenario("queue_timeout_503_with_retry_after", stubStream: true) {
            let hold = Hold()
            let target = try makeWireTarget(profile(handler: .hold(hold)) { $0.queueDeadline = .milliseconds(100) })
            defer { target.shutdown() }
            let first = try target.open(sending: target.request(body: chat(prompt: "hold")))
            defer { first.close() }
            guard hold.awaitStarted() else { throw WireScenarioError.handlerNeverStarted }
            let raw = try target.exchange(target.request(body: chat(prompt: "second")))
            hold.release()
            let firstStatus = Recorded.parse(first.readToEnd()).status ?? "nothing"
            return Recorded.parse(raw, facts: Recorded.facts(of: target.handlerCalls)
                + ["held request was answered: \(firstStatus)"])
        },
        scenario("queue_runs_waiter_after_release_in_order", stubStream: true) {
            let hold = Hold()
            let target = try makeWireTarget(profile(handler: .hold(hold)))
            defer { target.shutdown() }
            let first = try target.open(sending: target.request(body: chat(prompt: "hold")))
            defer { first.close() }
            guard hold.awaitStarted() else { throw WireScenarioError.handlerNeverStarted }
            let second = try target.open(sending: target.request(body: chat(prompt: "second")))
            defer { second.close() }
            guard target.awaitQueued() else { throw WireScenarioError.eventNeverCame }
            hold.release()
            _ = first.readToEnd()
            return Recorded.parse(second.readToEnd(), facts: Recorded.facts(of: target.handlerCalls))
        },
        scenario("client_leaves_while_queued", stubStream: true) {
            let hold = Hold()
            let target = try makeWireTarget(profile(handler: .hold(hold)) { $0.maxConnections = 2 })
            defer { target.shutdown() }
            let first = try target.open(sending: target.request(body: chat(prompt: "hold")))
            defer { first.close() }
            guard hold.awaitStarted() else { throw WireScenarioError.handlerNeverStarted }
            let second = try target.open(sending: target.request(body: chat(prompt: "second")))
            guard target.awaitQueued() else { throw WireScenarioError.eventNeverCame }
            second.close()
            guard target.awaitAbandoned() else {
                throw WireScenarioError.eventNeverCame
            }
            // Both connections were in use; the third is served only because the second's was freed.
            let third = try target.open(sending: target.request(body: chat(prompt: "third")))
            defer { third.close() }
            guard target.awaitQueued() else { throw WireScenarioError.eventNeverCame }
            hold.release()
            _ = first.readToEnd()
            let thirdAnswer = Recorded.parse(third.readToEnd())
            return Recorded(status: thirdAnswer.status, headers: thirdAnswer.headers, body: thirdAnswer.body,
                            facts: ["queued request abandoned: reported",
                                    "connection of the abandoned request: freed"]
                                + Recorded.facts(of: target.handlerCalls))
        },
        scenario("client_leaves_mid_stream") {
            let probe = LeaveProbe()
            let target = try makeWireTarget(profile(handler: .streamUntilClientLeaves(probe)))
            defer { target.shutdown() }
            let client = try target.open(sending: target.request())
            guard probe.awaitStreaming() else { throw WireScenarioError.handlerNeverStarted }
            client.close()
            guard let seen = probe.awaitFinished() else { throw WireScenarioError.handlerNeverStarted }
            return Recorded(status: nil, headers: [], body: "", facts: [
                "handler observed clientHasDisconnected: \(seen.sawDisconnect)",
                "generation stopped before its cap of \(LeaveProbe.cap) tokens: \(seen.sent < LeaveProbe.cap)",
            ])
        },
    ]

    // MARK: Behaviour that is expected to change

    private static let malformedTarget = "A malformed target becomes 400 malformed_target, OpenAI-shaped (proposal section 7)"

    private static let changes: [WireExchange] = [
        plain("target_404_double_slash", expectedToChange: malformedTarget) { target in
            target.request("POST", "//v1/chat/completions")
        },
        plain("target_404_dot_segments", expectedToChange: malformedTarget) { target in
            target.request("POST", "/v1/../v1/chat/completions")
        },
        plain("target_404_single_dot_segment", expectedToChange: malformedTarget) { target in
            target.request("POST", "/v1/./chat/completions")
        },
        plain("target_404_bad_percent_escape", expectedToChange: malformedTarget) { target in
            target.request("POST", "/v1/chat/%zz")
        },
        plain("target_404_truncated_percent_escape", expectedToChange: malformedTarget) { target in
            target.request("POST", "/v1/chat/completions%")
        },
        // The proposal's table says the old server answers HTTP/1.0 with `HTTP/1.0`. It does not:
        // every response it writes begins `HTTP/1.1`. So this exchange is not marked as changing.
        plain("version_http10_request", intended: "The kit answers an HTTP/1.0 request with an HTTP/1.0 status line; the old server answered HTTP/1.1 (the proposal says the kit answers 1.1)", headerChanges: cacheControl) { target in
            raw("POST /v1/chat/completions HTTP/1.0\r\nHost: 127.0.0.1:\(target.port)\r\n"
                + "Authorization: Bearer \(RunningServer.key)\r\nContent-Length: \(chat().utf8.count)\r\n\r\n"
                + chat())
        },
    ]
}

enum WireScenarioError: Error {
    case handlerNeverStarted
    case eventNeverCame
}

// MARK: - The fixture

enum WireFixture {

    static let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/exchanges.json")

    static let note = "Recorded from the original hand-rolled HTTPServer at tag v0.1.0. These are the old server's bytes: "
        + "never regenerate them to make a later server pass. Differences that are intended are listed on their "
        + "exchange (expectedToChange) or in the test's headerChanges, with the reason."

    static var isRecording: Bool { ProcessInfo.processInfo.environment["RECORD"] == "1" }

    static func load() throws -> [String: RecordedExchange] {
        let data = try Data(contentsOf: url)
        let fixture = try JSONDecoder().decode(ExchangeFixture.self, from: data)
        return Dictionary(uniqueKeysWithValues: fixture.exchanges.map { ($0.name, $0) })
    }

    static func write(_ exchanges: [RecordedExchange]) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(ExchangeFixture(note: note, exchanges: exchanges))
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }
}

// MARK: - The tests

extension HTTPServiceTests {
@Suite("Wire characterization", .serialized)
struct WireCharacterizationTests {

    private static let recording = WireFixture.isRecording

    /// Runs `exchange` and compares it with the fixture.
    ///
    /// - Returns: The differences that are not intended. Intended ones are printed.
    private func replay(_ exchange: WireExchange) throws -> [String] {
        let fixture = try WireFixture.load()
        let stored = try #require(fixture[exchange.name], "no recorded exchange named \(exchange.name)")
        #expect(stored.expectedToChange == exchange.expectedToChange,
                "the fixture and the table disagree about whether \(exchange.name) is expected to change")

        var actual = try exchange.run()
        if exchange.stubStream {
            actual.body = actual.body.replacingOccurrences(of: WireExchanges.stubFinishEvent, with: "")
        }
        var differences = WireComparison.differences(expected: stored.recorded, actual: actual)

        var intended: [String] = []
        if let reason = exchange.intended {
            intended = differences.map { "\($0) [INTENDED: \(reason)]" }
            differences = []
        } else if let reason = exchange.expectedToChange {
            intended = differences.map { "\($0) [expected to change: \(reason)]" }
            differences = []
        } else {
            // Every event stream carries the stream header changes, whichever exchange produced it.
            let isStream = actual.headers.contains { $0[0].lowercased() == "content-type" && $0[1] == "text/event-stream" }
            let allowed = exchange.headerChanges.merging(isStream ? WireExchanges.cacheControl : [:]) { first, _ in first }
            let split = differences.partitioned { difference in
                WireComparison.headerName(of: difference).map { allowed[$0] != nil } ?? false
            }
            differences = split.rest
            intended = split.matching.map { difference in
                let name = WireComparison.headerName(of: difference) ?? ""
                return "\(difference) [expected to change: \(allowed[name] ?? "")]"
            }
        }
        intended.forEach { print("INTENDED DIFFERENCE in \(exchange.name): \($0)") }
        return differences
    }

    @Test("Each recorded exchange replays", .enabled(if: !WireCharacterizationTests.recording),
          arguments: WireExchanges.all.filter { !$0.needsMetal })
    func replayExchange(exchange: WireExchange) throws {
        let differences = try replay(exchange)
        #expect(differences.isEmpty, "\(exchange.name) differs from the recorded exchange: \(differences)")
    }

    @Test("The demo model's stream replays", .enabled(if: !WireCharacterizationTests.recording
                                                       && ShaderLibraryTests.shadersAvailable,
                                                       "requires metal_infer/shaders.metal"),
          arguments: WireExchanges.all.filter(\.needsMetal))
    func replayDemoModel(exchange: WireExchange) throws {
        let differences = try replay(exchange)
        #expect(differences.isEmpty, "\(exchange.name) differs from the recorded exchange: \(differences)")
    }

    @Test("Every intended difference is written down with a reason, and none is also an expected change")
    func intendedDifferencesHaveReasons() {
        for exchange in WireExchanges.all {
            if let reason = exchange.intended {
                #expect(reason.count > 30, "\(exchange.name) needs a reason")
                #expect(exchange.expectedToChange == nil, "\(exchange.name) is listed twice")
            }
        }
        #expect(WireExchanges.all.filter { $0.intended != nil }.count == 9)
        #expect(WireExchanges.all.filter { $0.expectedToChange != nil }.count == 6)
    }

    @Test("The fixture holds exactly the table's exchanges", .enabled(if: !WireCharacterizationTests.recording))
    func fixtureMatchesTable() throws {
        let fixture = try WireFixture.load()
        let names = WireExchanges.all.map(\.name)
        #expect(Set(names).count == names.count, "exchange names must be unique")
        #expect(Set(fixture.keys) == Set(names))
    }

    @Test("Records the old server's exchanges into the fixture (RECORD=1 only)",
          .enabled(if: WireCharacterizationTests.recording))
    func record() throws {
        #expect(ShaderLibraryTests.shadersAvailable, "recording needs the Metal shaders for the demo-model stream")
        var recorded: [RecordedExchange] = []
        for exchange in WireExchanges.all {
            recorded.append(RecordedExchange(name: exchange.name, expectedToChange: exchange.expectedToChange,
                                             recorded: try exchange.run()))
        }
        try WireFixture.write(recorded)
        print("recorded \(recorded.count) exchanges to \(WireFixture.url.path)")
    }

    // MARK: The stream's bytes

    @Test("A completed stream is SSE data events of chat.completion.chunk JSON, a finish chunk, then [DONE]")
    func streamShape() throws {
        let target = try makeWireTarget({
            var profile = WireProfile()
            profile.handler = .tokens(["Hel", "lo"])
            return profile
        }())
        defer { target.shutdown() }
        let raw = try target.exchange(target.request())
        let parts = raw.components(separatedBy: "\r\n\r\n")
        let body = parts.dropFirst().joined(separator: "\r\n\r\n")

        #expect(body.hasSuffix("\n\n"))
        let events = body.components(separatedBy: "\n\n").dropLast()
        #expect(events.count == 4)
        #expect(events.allSatisfy { $0.hasPrefix("data: ") && !$0.dropFirst(6).contains("\n") })
        #expect(events.last == "data: [DONE]")

        var ids: Set<String> = []
        var contents: [String] = []
        var finishes: [Any] = []
        for event in events.dropLast() {
            let json = try #require(try JSONSerialization.jsonObject(with: Data(event.dropFirst(6).utf8)) as? [String: Any])
            #expect(json["object"] as? String == "chat.completion.chunk")
            ids.insert(try #require(json["id"] as? String))
            let choice = try #require((json["choices"] as? [[String: Any]])?.first)
            #expect(choice["index"] as? Int == 0)
            let delta = try #require(choice["delta"] as? [String: Any])
            if let content = delta["content"] as? String { contents.append(content) }
            finishes.append(choice["finish_reason"] ?? "absent")
        }
        #expect(ids.count == 1 && ids.first?.hasPrefix("req_") == true)
        #expect(contents == ["Hel", "lo"])
        // `finish_reason` is JSON null on every delta and "stop" on the last chunk, whose delta is empty.
        #expect(finishes.dropLast().allSatisfy { $0 is NSNull })
        #expect(finishes.last as? String == "stop")

        let exactly = "data: {\"id\":\"req_<ID>\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n"
        #expect(Recorded.parse(raw).body.hasPrefix(exactly))
    }

    @Test("A truncated stream ends without a finish chunk or [DONE]")
    func truncatedStreamHasNoTerminator() throws {
        var profile = WireProfile()
        profile.handler = .truncated(["par"])
        let target = try makeWireTarget(profile)
        defer { target.shutdown() }
        let body = Recorded.parse(try target.exchange(target.request())).body
        #expect(body.contains("\"content\":\"par\""))
        #expect(!body.contains("finish_reason\":\"stop\""))
        #expect(!body.contains("[DONE]"))
    }
}
}

private extension Array {
    /// Splits into the elements that satisfy `predicate` and those that do not, keeping order.
    func partitioned(by predicate: (Element) -> Bool) -> (matching: [Element], rest: [Element]) {
        var matching: [Element] = []
        var rest: [Element] = []
        for element in self {
            if predicate(element) { matching.append(element) } else { rest.append(element) }
        }
        return (matching, rest)
    }
}
