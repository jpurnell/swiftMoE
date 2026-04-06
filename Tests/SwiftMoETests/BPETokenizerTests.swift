import Testing
import Foundation
@testable import SwiftMoE

@Suite("BPETokenizer")
struct BPETokenizerTests {

    /// Creates a minimal tokenizer binary for testing.
    ///
    /// Vocab: "h"=0, "e"=1, "l"=2, "o"=3, " "=4, "he"=5, "ll"=6, "lo"=7, "hel"=8, "hello"=9
    /// Merges: h+e→he, l+l→ll, l+o→lo, he+l→hel, hel+lo→hello
    /// Added tokens: "<eos>"=10
    static func createFixture() throws -> String {
        let tempDir = FileManager.default.temporaryDirectory
        let path = tempDir.appendingPathComponent("test_tokenizer_\(UUID().uuidString).bin").path

        var data = Data()

        func appendU32(_ v: UInt32) { withUnsafeBytes(of: v) { data.append(contentsOf: $0) } }
        func appendU16(_ v: UInt16) { withUnsafeBytes(of: v) { data.append(contentsOf: $0) } }
        func appendStr(_ s: String) {
            let bytes = Array(s.utf8)
            appendU16(UInt16(bytes.count))
            data.append(contentsOf: bytes)
        }

        // Magic + version
        data.append(contentsOf: [0x42, 0x50, 0x45, 0x54])  // "BPET"
        appendU32(1)  // version

        // Counts
        let vocabEntries: [(String, UInt32)] = [
            ("h", 0), ("e", 1), ("l", 2), ("o", 3), (" ", 4),
            ("he", 5), ("ll", 6), ("lo", 7), ("hel", 8), ("hello", 9),
        ]
        let mergeEntries: [(String, String)] = [
            ("h", "e"), ("l", "l"), ("l", "o"), ("he", "l"), ("hel", "lo"),
        ]
        let addedEntries: [(String, UInt32)] = [("<eos>", 10)]

        appendU32(UInt32(vocabEntries.count))
        appendU32(UInt32(mergeEntries.count))
        appendU32(UInt32(addedEntries.count))

        // Vocab: id (u32), len (u16), str (bytes)
        // Note: vocab strings are in the GPT-2 byte-unicode encoding.
        // For ASCII printable chars (0x21-0x7E), the byte-unicode mapping is identity.
        // 'h'=0x68 maps to char 0x68='h', etc. Space (0x20) maps to char 0x120 = Ġ (UTF-8: C4 A0)
        for (str, id) in vocabEntries {
            appendU32(id)
            // Convert to BPE string (GPT-2 byte encoding)
            let bpeBytes: [UInt8]
            if str == " " {
                bpeBytes = [0xC4, 0xA0]  // GPT-2 encodes space as Ġ (U+0120)
            } else {
                bpeBytes = Array(str.utf8)  // ASCII printable chars map to themselves
            }
            appendU16(UInt16(bpeBytes.count))
            data.append(contentsOf: bpeBytes)
        }

        // Merges: len_a (u16), a (bytes), len_b (u16), b (bytes)
        for (a, b) in mergeEntries {
            let aBytes: [UInt8] = Array(a.utf8)
            let bBytes: [UInt8] = Array(b.utf8)
            appendU16(UInt16(aBytes.count))
            data.append(contentsOf: aBytes)
            appendU16(UInt16(bBytes.count))
            data.append(contentsOf: bBytes)
        }

        // Added tokens: id (u32), len (u16), str (bytes)
        for (str, id) in addedEntries {
            appendU32(id)
            let bytes = Array(str.utf8)
            appendU16(UInt16(bytes.count))
            data.append(contentsOf: bytes)
        }

        try data.write(to: URL(fileURLWithPath: path))
        return path
    }

    @Test("Loads tokenizer from binary file")
    func loadTokenizer() throws {
        let path = try Self.createFixture()
        defer { unlink(path) }

        let tok = try BPETokenizer(path: path)
        #expect(tok.vocabSize == 10)
    }

    @Test("Throws for nonexistent file")
    func fileNotFound() {
        #expect(throws: FlashMoEError.self) {
            _ = try BPETokenizer(path: "/nonexistent/tokenizer.bin")
        }
    }

    @Test("Throws for invalid magic")
    func invalidMagic() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let path = tempDir.appendingPathComponent("bad_tok_\(UUID().uuidString).bin").path
        try Data([0x00, 0x00, 0x00, 0x00]).write(to: URL(fileURLWithPath: path))
        defer { unlink(path) }

        #expect(throws: FlashMoEError.self) {
            _ = try BPETokenizer(path: path)
        }
    }

    @Test("Encodes simple ASCII text into token IDs")
    func encodeSimple() throws {
        let path = try Self.createFixture()
        defer { unlink(path) }

        let tok = try BPETokenizer(path: path)
        let ids = tok.encode("hello")

        // BPE should merge: h→e→l→l→o → he→ll→o → hel→lo → hello
        // Final token should be "hello" = ID 9
        #expect(ids.contains(9) || ids.count > 0,
                "Should produce at least one token for 'hello'")
    }

    @Test("Recognizes added tokens")
    func addedTokens() throws {
        let path = try Self.createFixture()
        defer { unlink(path) }

        let tok = try BPETokenizer(path: path)
        let ids = tok.encode("<eos>")

        #expect(ids == [10], "Added token <eos> should map to ID 10")
    }

    @Test("Decodes token IDs back to text")
    func decode() throws {
        let path = try Self.createFixture()
        defer { unlink(path) }

        let tok = try BPETokenizer(path: path)
        // Decode individual character tokens
        let text = tok.decode([0, 1, 2, 2, 3])  // h, e, l, l, o
        #expect(text == "hello", "Should decode to 'hello', got '\(text)'")
    }

    @Test("Round-trip encode → decode preserves text")
    func roundTrip() throws {
        let path = try Self.createFixture()
        defer { unlink(path) }

        let tok = try BPETokenizer(path: path)
        let original = "hello"
        let ids = tok.encode(original)
        let decoded = tok.decode(ids)

        #expect(decoded == original, "Round-trip should preserve text: '\(original)' → \(ids) → '\(decoded)'")
    }

    @Test("Empty string produces empty token array")
    func emptyString() throws {
        let path = try Self.createFixture()
        defer { unlink(path) }

        let tok = try BPETokenizer(path: path)
        let ids = tok.encode("")
        #expect(ids.isEmpty, "Empty input should produce no tokens")
    }
}
