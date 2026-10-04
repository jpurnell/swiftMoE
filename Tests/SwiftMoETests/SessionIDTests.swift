import Foundation
import Testing
@testable import SwiftMoE

/// A generator that hands out a fixed sequence of words, so the encoding is checked exactly.
private struct FixedWords: RandomNumberGenerator {
    var words: [UInt64]
    var index = 0

    mutating func next() -> UInt64 {
        defer { index += 1 }
        return words[index % words.count]
    }
}

/// How a session gets its id when the caller does not supply one.
///
/// The default was `UUID().uuidString`. A version-4 UUID carries 122 random bits and RFC 4122
/// says outright that it is not to be used as a security capability: it is built to be unique,
/// not to be unguessable. The id names the file a conversation is stored in, so it is drawn from
/// a `RandomNumberGenerator` instead — 32 bytes, hex — and the chat client's entry point passes the
/// system generator.
@Suite("Session id generation")
struct SessionIDTests {

    private static let words: [UInt64] = [
        0x0001_0203_0405_0607, 0x0809_0A0B_0C0D_0E0F,
        0xF0F1_F2F3_F4F5_F6F7, 0xF8F9_FAFB_FCFD_FEFF,
        0x1111_1111_1111_1111, 0x2222_2222_2222_2222,
        0x3333_3333_3333_3333, 0x0000_0000_0000_0004,
    ]
    private static let firstID = "000102030405060708090a0b0c0d0e0ff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"
    private static let secondID = "1111111111111111222222222222222233333333333333330000000000000004"

    private static func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("moe-session-id-\(UUID().uuidString)")
    }

    @Test("An id is the generator's four words, most significant byte first")
    func encodingIsExact() {
        var generator = FixedWords(words: Self.words)
        let id = SessionStore.makeSessionID(using: &generator)
        #expect(id == Self.firstID)
        #expect(id.count == 64)
        #expect(generator.index == 4)
    }

    @Test("Each id consumes fresh words, and leading zeros are kept")
    func successiveIDsDiffer() {
        var generator = FixedWords(words: Self.words)
        let first = SessionStore.makeSessionID(using: &generator)
        let second = SessionStore.makeSessionID(using: &generator)
        #expect(first == Self.firstID)
        #expect(second == Self.secondID)
        #expect(generator.index == 8)
    }

    @Test("A store created without an id draws one from the generator, not a UUID")
    func storeDefault() {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) } // silent: test cleanup; a leftover temp directory changes no result

        var generator = FixedWords(words: Self.words)
        let store = SessionStore(sessionID: nil, sessionsDirectory: directory.path, using: &generator)
        #expect(store.sessionID == Self.firstID)
        #expect(UUID(uuidString: store.sessionID) == nil)
        #expect(store.sessionPath == directory.standardized
            .appendingPathComponent("\(Self.firstID).jsonl").path)
    }

    @Test("A store keeps the id it was given and draws nothing")
    func storeKeepsSuppliedID() {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) } // silent: test cleanup; a leftover temp directory changes no result

        var generator = FixedWords(words: Self.words)
        let store = SessionStore(sessionID: "resumed", sessionsDirectory: directory.path, using: &generator)
        #expect(store.sessionID == "resumed")
        #expect(generator.index == 0)
    }
}
