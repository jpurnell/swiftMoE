import Foundation
import Testing
@testable import SwiftMoE

/// How long a prompt may be: prompt tokens plus requested completion tokens against the
/// positions a sequence holds.
///
/// The body limit used to be the only bound on a prompt — 64 KiB, which the placeholder
/// tokenizer turns into 65,536 tokens for KV caches of 8,192 positions.
@Suite("HTTP server sequence budget")
struct HTTPServerSequenceBudgetTests {

    /// A sequence of 32 positions, at most 16 of them completion, 8 when the request does not say.
    private static var smallLimits: HTTPServer.Limits {
        var limits = HTTPServer.Limits()
        limits.maxSequenceTokens = 32
        limits.maxCompletionTokens = 16
        limits.defaultCompletionTokens = 8
        return limits
    }

    private static func body(prompt: String, tokens: Int? = nil) -> String {
        let count = tokens.map { #","max_tokens":\#($0)"# } ?? ""
        return #"{"messages":[{"role":"user","content":"\#(prompt)"}]\#(count)}"#
    }

    private static func tooLong(prompt: Int, completion: Int, limit: Int) -> String {
        Expected.refusal(
            "400 Bad Request",
            "Prompt (\(prompt) tokens) plus completion (\(completion) tokens) is \(prompt + completion) tokens; "
                + "the limit for a sequence is \(limit). Shorten the prompt or lower max_tokens.")
    }

    // MARK: - The limit itself

    @Test("The sequence limit defaults to the positions the KV caches are allocated for")
    func defaultLimit() {
        #expect(HTTPServer.Limits().maxSequenceTokens == 8192)
        #expect(HTTPServer.Limits().maxSequenceTokens == TokenGenerator.defaultMaxSequenceLength)
    }

    @Test("A sequence limit of zero stops the listener from opening")
    func zeroLimit() {
        var limits = HTTPServer.Limits()
        limits.maxSequenceTokens = 0
        let server = HTTPServer(port: 0, authentication: .unauthenticatedLoopback, limits: limits,
                                tokenizer: RunningServer.byteTokenizer) { _, _ in }
        #expect(throws: HTTPServerError.invalidLimit(name: "maxSequenceTokens")) {
            _ = try server.openListener()
        }
        #expect(server.boundAddress == nil)
    }

    @Test("A completion ceiling above the sequence limit stops the listener from opening")
    func completionCeilingAboveSequence() {
        var limits = HTTPServer.Limits()
        limits.maxSequenceTokens = 4096
        let server = HTTPServer(port: 0, authentication: .unauthenticatedLoopback, limits: limits,
                                tokenizer: RunningServer.byteTokenizer) { _, _ in }
        #expect(throws: HTTPServerError.limitAboveLimit(name: "maxCompletionTokens",
                                                        ceiling: "maxSequenceTokens")) {
            _ = try server.openListener()
        }
        #expect(server.boundAddress == nil)
    }

    @Test("The operator is told which limit is too large and what bounds it")
    func limitAboveLimitMessage() {
        let error = HTTPServerError.limitAboveLimit(name: "maxCompletionTokens", ceiling: "maxSequenceTokens")
        #expect(error.errorDescription
            == "The limit maxCompletionTokens must not be greater than maxSequenceTokens.")
    }

    // MARK: - Requests

    @Test("A prompt and completion that exactly fill the sequence reach the handler, tokens and all")
    func exactlyAtTheLimit() throws {
        let running = try RunningServer(limits: Self.smallLimits)
        defer { running.shutdown() }

        let prompt = String(repeating: "p", count: 24)
        #expect(try running.exchange(running.request(body: Self.body(prompt: prompt))) == Expected.stream())
        #expect(try running.exchange(running.request(body: Self.body(prompt: "abcdefghijklmnop", tokens: 16)))
            == Expected.stream())
        #expect(running.handledRequests == [
            HTTPServer.Request(prompt: prompt, promptTokens: [Int](repeating: 0x70, count: 24), maxTokens: 8),
            HTTPServer.Request(prompt: "abcdefghijklmnop", promptTokens: Array(0x61...0x70), maxTokens: 16),
        ])
    }

    @Test("One prompt token over is a 400 that gives the three numbers, and the handler is not called")
    func promptOneTokenOver() throws {
        let running = try RunningServer(limits: Self.smallLimits)
        defer { running.shutdown() }

        let response = try running.exchange(running.request(
            body: Self.body(prompt: String(repeating: "p", count: 25))))
        #expect(response == Self.tooLong(prompt: 25, completion: 8, limit: 32))
        #expect(running.handlerCalls == [])
    }

    @Test("One completion token over is refused the same way: the completion is not shortened to fit")
    func completionOneTokenOver() throws {
        let running = try RunningServer(limits: Self.smallLimits)
        defer { running.shutdown() }

        let response = try running.exchange(running.request(
            body: Self.body(prompt: String(repeating: "p", count: 24), tokens: 9)))
        #expect(response == Self.tooLong(prompt: 24, completion: 9, limit: 32))
        #expect(running.handlerCalls == [])
    }

    @Test("The count is the server's tokenizer's, not the prompt's size in bytes")
    func countIsTheTokenizers() throws {
        // One token per word: a 47-byte prompt of 24 words.
        let words: HTTPServer.Tokenizer = { prompt in
            prompt.split(separator: " ").map(\.count)
        }
        let running = try RunningServer(limits: Self.smallLimits, tokenizer: words)
        defer { running.shutdown() }

        let twentyFour = [String](repeating: "w", count: 24).joined(separator: " ")
        let twentyFive = [String](repeating: "w", count: 25).joined(separator: " ")
        #expect(try running.exchange(running.request(body: Self.body(prompt: twentyFour))) == Expected.stream())
        #expect(try running.exchange(running.request(body: Self.body(prompt: twentyFive)))
            == Self.tooLong(prompt: 25, completion: 8, limit: 32))
        #expect(running.handledRequests == [
            HTTPServer.Request(prompt: twentyFour, promptTokens: [Int](repeating: 1, count: 24), maxTokens: 8),
        ])
    }

    @Test("A tokenizer that adds tokens of its own is counted with them")
    func tokenizerOverhead() throws {
        // A beginning-of-sequence token ahead of one token per byte.
        let withMarker: HTTPServer.Tokenizer = { prompt in [1] + prompt.utf8.map { Int($0) } }
        let running = try RunningServer(limits: Self.smallLimits, tokenizer: withMarker)
        defer { running.shutdown() }

        let response = try running.exchange(running.request(
            body: Self.body(prompt: String(repeating: "p", count: 24))))
        #expect(response == Self.tooLong(prompt: 25, completion: 8, limit: 32))
        #expect(running.handlerCalls == [])
    }

    @Test("With the default limits a 9,000-byte prompt no longer reaches the handler")
    func defaultLimitsRefuseALongPrompt() throws {
        let running = try RunningServer()
        defer { running.shutdown() }

        let response = try running.exchange(running.request(
            body: Self.body(prompt: String(repeating: "p", count: 9000))))
        #expect(response == Self.tooLong(prompt: 9000, completion: 100, limit: 8192))
        #expect(running.handlerCalls == [])
    }

    @Test("The completion ceiling still applies on its own, and is reported as before")
    func completionCeilingStillApplies() throws {
        let running = try RunningServer(limits: Self.smallLimits)
        defer { running.shutdown() }

        let response = try running.exchange(running.request(body: Self.body(prompt: "hi", tokens: 17)))
        #expect(response == Expected.refusal("400 Bad Request", "max_tokens must be a whole number from 1 to 16."))
        #expect(running.handlerCalls == [])
    }

    @Test("A request with no messages has an empty prompt, counted as the tokenizer counts it")
    func emptyPrompt() throws {
        let running = try RunningServer(limits: Self.smallLimits)
        defer { running.shutdown() }

        #expect(try running.exchange(running.request(body: #"{"max_tokens":16}"#)) == Expected.stream())
        #expect(running.handledRequests == [HTTPServer.Request(prompt: "", promptTokens: [], maxTokens: 16)])
    }
}
