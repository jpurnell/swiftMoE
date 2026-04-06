import Foundation

/// Byte-Pair Encoding tokenizer for Qwen3/GPT-style models.
///
/// Loads from a binary format (`tokenizer.bin`), performs pre-tokenization
/// (whitespace splitting, contraction handling, multi-byte UTF-8 awareness),
/// then iterative BPE merging to produce token IDs.
///
/// ## Binary Format
/// ```
/// "BPET" magic (4 bytes), version u32, vocab_size u32, num_merges u32, num_added u32
/// Vocab:  [id: u32, len: u16, str: [u8; len]] × vocab_size
/// Merges: [len_a: u16, a: [u8; len_a], len_b: u16, b: [u8; len_b]] × num_merges
/// Added:  [id: u32, len: u16, str: [u8; len]] × num_added
/// ```
public final class BPETokenizer {

    /// O(1) vocab lookup: BPE token bytes → token ID.
    private var vocabLookup: [[UInt8]: Int] = [:]

    /// O(1) merge lookup: merge key → priority (lower = higher priority).
    private var mergeLookup: [[UInt8]: Int] = [:]

    /// Added tokens (special sequences matched before BPE).
    private var addedTokens: [(bytes: [UInt8], id: Int)] = []

    /// Reverse lookup: token ID → BPE string bytes (for decoding).
    private var idToBytes: [Int: [UInt8]] = [:]

    /// GPT-2 byte-to-unicode table.
    private var byteToChar: [UInt32] = Array(repeating: 0, count: 256)
    /// Reverse: unicode codepoint → raw byte.
    private var charToByte: [UInt8] = Array(repeating: 0, count: 512)

    /// Number of tokens in the vocabulary.
    public private(set) var vocabSize: Int = 0

    // MARK: - Loading

    /// Loads a tokenizer from a binary file.
    ///
    /// - Parameter path: Path to `tokenizer.bin`.
    /// - Throws: ``FlashMoEError/fileNotFound(path:)`` or ``FlashMoEError/manifestParseFailed(reason:)``.
    public init(path: String) throws {
        buildByteUnicodeTable()

        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
            throw FlashMoEError.fileNotFound(path: path)
        }

        var offset = 0

        func readU32() throws -> UInt32 {
            guard offset + 4 <= data.count else {
                throw FlashMoEError.manifestParseFailed(reason: "Unexpected EOF")
            }
            let v = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
            offset += 4; return v
        }
        func readU16() throws -> UInt16 {
            guard offset + 2 <= data.count else {
                throw FlashMoEError.manifestParseFailed(reason: "Unexpected EOF")
            }
            let v = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self) }
            offset += 2; return v
        }
        func readBytes(_ n: Int) throws -> [UInt8] {
            guard offset + n <= data.count else {
                throw FlashMoEError.manifestParseFailed(reason: "Unexpected EOF")
            }
            let b = [UInt8](data[offset..<offset+n]); offset += n; return b
        }

        // Header
        let magic = try readBytes(4)
        guard magic == [0x42, 0x50, 0x45, 0x54] else {
            throw FlashMoEError.manifestParseFailed(reason: "Invalid magic")
        }
        guard try readU32() == 1 else {
            throw FlashMoEError.manifestParseFailed(reason: "Unsupported version")
        }

        let vocabCount = Int(try readU32())
        let mergeCount = Int(try readU32())
        let addedCount = Int(try readU32())
        self.vocabSize = vocabCount

        // Vocab
        for _ in 0..<vocabCount {
            let id = Int(try readU32())
            let len = Int(try readU16())
            let str = try readBytes(len)
            vocabLookup[str] = id
            idToBytes[id] = str
        }

        // Merges
        for i in 0..<mergeCount {
            let lenA = Int(try readU16())
            let a = try readBytes(lenA)
            let lenB = Int(try readU16())
            let b = try readBytes(lenB)
            var key = a; key.append(0xFF); key.append(contentsOf: b)
            mergeLookup[key] = i
        }

        // Added tokens
        for _ in 0..<addedCount {
            let id = Int(try readU32())
            let len = Int(try readU16())
            let str = try readBytes(len)
            addedTokens.append((bytes: str, id: id))
        }
    }

    // MARK: - Encoding

    /// Encodes text into token IDs.
    public func encode(_ text: String) -> [Int] {
        let bytes = Array(text.utf8)
        if bytes.isEmpty { return [] }
        var result: [Int] = []
        var pos = 0

        while pos < bytes.count {
            // Check added tokens (longest match first)
            var foundAdded = false
            var bestLen = 0, bestID = 0
            for added in addedTokens {
                let n = added.bytes.count
                if n > bestLen && pos + n <= bytes.count
                    && Array(bytes[pos..<pos+n]) == added.bytes {
                    bestLen = n; bestID = added.id; foundAdded = true
                }
            }
            if foundAdded {
                result.append(bestID); pos += bestLen; continue
            }

            // Find next added token boundary
            var chunkEnd = bytes.count
            for added in addedTokens {
                let n = added.bytes.count
                for j in (pos+1)..<(bytes.count - n + 1) {
                    if Array(bytes[j..<j+n]) == added.bytes {
                        chunkEnd = min(chunkEnd, j); break
                    }
                }
            }

            // Pre-tokenize chunk and BPE-encode each span
            let chunk = Array(bytes[pos..<chunkEnd])
            let spans = pretokenize(chunk)
            for span in spans {
                let piece = Array(chunk[span.0..<span.1])
                let bpeStr = bytesToBPE(piece)
                result.append(contentsOf: bpeMerge(bpeStr))
            }
            pos = chunkEnd
        }
        return result
    }

    // MARK: - Decoding

    /// Decodes token IDs back to a string.
    public func decode(_ ids: [Int]) -> String {
        var raw: [UInt8] = []
        for id in ids {
            if let bpeBytes = idToBytes[id] {
                raw.append(contentsOf: bpeToBytes(bpeBytes))
            }
        }
        return String(bytes: raw, encoding: .utf8) ?? String(bytes: raw, encoding: .ascii) ?? ""
    }

    // MARK: - Private: GPT-2 Byte-Unicode Table

    private func buildByteUnicodeTable() {
        var n: UInt32 = 0
        for b in 0..<256 {
            if (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
                byteToChar[b] = UInt32(b)
            } else {
                byteToChar[b] = 256 + n; n += 1
            }
        }
        for b in 0..<256 {
            let cp = Int(byteToChar[b])
            if cp < charToByte.count { charToByte[cp] = UInt8(b) }
        }
    }

    /// Converts raw bytes to BPE string (GPT-2 byte-unicode encoding).
    private func bytesToBPE(_ raw: [UInt8]) -> [UInt8] {
        var out: [UInt8] = []
        for byte in raw {
            let cp = byteToChar[Int(byte)]
            if cp < 0x80 { out.append(UInt8(cp)) }
            else if cp < 0x800 {
                out.append(UInt8(0xC0 | (cp >> 6)))
                out.append(UInt8(0x80 | (cp & 0x3F)))
            } else {
                out.append(UInt8(0xE0 | (cp >> 12)))
                out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
                out.append(UInt8(0x80 | (cp & 0x3F)))
            }
        }
        return out
    }

    /// Converts BPE string back to raw bytes.
    private func bpeToBytes(_ bpe: [UInt8]) -> [UInt8] {
        var out: [UInt8] = []
        var i = 0
        while i < bpe.count {
            let c = bpe[i]
            let cp: UInt32
            if c < 0x80 { cp = UInt32(c); i += 1 }
            else if c < 0xE0, i + 1 < bpe.count {
                cp = (UInt32(c & 0x1F) << 6) | UInt32(bpe[i+1] & 0x3F); i += 2
            } else if c < 0xF0, i + 2 < bpe.count {
                cp = (UInt32(c & 0x0F) << 12) | (UInt32(bpe[i+1] & 0x3F) << 6) | UInt32(bpe[i+2] & 0x3F); i += 3
            } else { i += 1; continue }
            if Int(cp) < charToByte.count { out.append(charToByte[Int(cp)]) }
        }
        return out
    }

    // MARK: - Private: Pre-tokenization

    /// Splits text into spans for BPE processing.
    /// Returns array of (start, end) byte ranges.
    private func pretokenize(_ text: [UInt8]) -> [(Int, Int)] {
        var spans: [(Int, Int)] = []
        var i = 0
        while i < text.count {
            let c = text[i]
            // Whitespace
            if c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D {
                var j = i
                while j < text.count && (text[j] == 0x20 || text[j] == 0x09 || text[j] == 0x0A || text[j] == 0x0D) { j += 1 }
                spans.append((i, j)); i = j; continue
            }
            // Letters (ASCII or multi-byte UTF-8)
            if c >= 0xC0 || (c >= 0x41 && c <= 0x5A) || (c >= 0x61 && c <= 0x7A) {
                var j = i
                while j < text.count {
                    let jc = text[j]
                    if jc >= 0xC0 { j += jc < 0xE0 ? 2 : jc < 0xF0 ? 3 : 4 }
                    else if (jc >= 0x41 && jc <= 0x5A) || (jc >= 0x61 && jc <= 0x7A) { j += 1 }
                    else { break }
                }
                spans.append((i, j)); i = j; continue
            }
            // Single character
            spans.append((i, i + 1)); i += 1
        }
        return spans
    }

    // MARK: - Private: BPE Merging

    /// Performs iterative BPE merging on a BPE-encoded string.
    private func bpeMerge(_ bpeStr: [UInt8]) -> [Int] {
        if bpeStr.isEmpty { return [] }

        // Split into UTF-8 characters as linked list
        var pieces: [(bytes: [UInt8], next: Int)] = []
        var i = 0
        while i < bpeStr.count {
            let c = bpeStr[i]
            let clen = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4
            let end = min(i + clen, bpeStr.count)
            pieces.append((bytes: Array(bpeStr[i..<end]), next: pieces.count + 1))
            i = end
        }
        if pieces.isEmpty { return [] }
        pieces[pieces.count - 1].next = -1

        var active = pieces.count

        while active > 1 {
            var bestPrio = Int.max, bestIdx = -1
            var ci = 0
            while ci != -1 {
                let ni = pieces[ci].next
                if ni == -1 { break }
                var key = pieces[ci].bytes; key.append(0xFF); key.append(contentsOf: pieces[ni].bytes)
                if let prio = mergeLookup[key], prio < bestPrio {
                    bestPrio = prio; bestIdx = ci
                }
                ci = ni
            }
            if bestIdx == -1 { break }

            let ni = pieces[bestIdx].next
            pieces[bestIdx].bytes.append(contentsOf: pieces[ni].bytes)
            pieces[bestIdx].next = pieces[ni].next
            active -= 1
        }

        // Look up token IDs
        var result: [Int] = []
        var ci = 0
        while ci != -1 {
            if let id = vocabLookup[pieces[ci].bytes] {
                result.append(id)
            } else {
                // Byte-level fallback
                for byte in pieces[ci].bytes {
                    let bpe = bytesToBPE([byte])
                    if let id = vocabLookup[bpe] { result.append(id) }
                }
            }
            ci = pieces[ci].next
        }
        return result
    }
}
