import Foundation

/// The request line and header fields of an HTTP/1.x request, parsed rather than searched.
///
/// The server used to decide what a request was by asking whether the whole header block
/// began with `POST` and contained `/v1/chat/completions` anywhere — so the route's name in a
/// `Referer` was as good as the route. This type splits the request line into its three parts
/// and the target into path and query, so routing compares a method and a path exactly.
struct HTTPRequestHead: Equatable, Sendable {

    /// Request method, exactly as sent: `POST`, `GET`, `OPTIONS`.
    let method: String
    /// Path of the request target, without the query string.
    let path: String
    /// Query string after the first `?`, or `nil` when there is none.
    let query: String?
    /// Protocol version, e.g. `HTTP/1.1`.
    let version: String
    /// Header fields, keyed by lowercased name. Repeated fields are joined with `", "`.
    let headers: [String: String]

    /// Fields a request may carry once. Each decides who the caller is or how much to read, so
    /// two differing copies would let a proxy and this server disagree about the request.
    private static let singletonFields: Set<String> = [
        "host", "authorization", "content-length", "origin", "transfer-encoding",
    ]

    private static let carriageReturn: UInt8 = 0x0D
    private static let lineFeed: UInt8 = 0x0A
    private static let space: UInt8 = 0x20
    private static let tab: UInt8 = 0x09
    private static let colon: UInt8 = 0x3A

    /// Parses a request head.
    ///
    /// - Parameter bytes: Everything up to, but not including, the blank line that ends the head.
    /// - Returns: The parsed head, or `nil` when the bytes are not a request line followed by
    ///   well-formed header fields: wrong part count, a method that is not upper-case letters,
    ///   a target that is not an absolute path, a version that does not begin `HTTP/`, a folded
    ///   or nameless field, a repeated singleton field, or text that is not UTF-8.
    static func parse(_ bytes: [UInt8]) -> HTTPRequestHead? {
        var lines = splitLines(bytes)
        guard !lines.isEmpty else { return nil }
        let requestLine = lines.removeFirst()

        let parts = requestLine.split(separator: space, omittingEmptySubsequences: false)
        guard parts.count == 3,
              let method = String(validatingUTF8Bytes: parts[0]),
              let target = String(validatingUTF8Bytes: parts[1]),
              let version = String(validatingUTF8Bytes: parts[2]),
              !method.isEmpty, method.utf8.allSatisfy({ $0 >= 0x41 && $0 <= 0x5A }),
              target.hasPrefix("/"),
              version.hasPrefix("HTTP/") else {
            return nil
        }

        var headers: [String: String] = [:]
        for line in lines {
            guard let field = parseField(line) else { return nil }
            if let existing = headers[field.name] {
                guard !singletonFields.contains(field.name) else { return nil }
                headers[field.name] = existing + ", " + field.value
            } else {
                headers[field.name] = field.value
            }
        }

        let pathAndQuery = target.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
        return HTTPRequestHead(
            method: method,
            path: String(pathAndQuery[0]),
            query: pathAndQuery.count == 2 ? String(pathAndQuery[1]) : nil,
            version: version,
            headers: headers
        )
    }

    /// Splits on CRLF. A bare CR or LF stays inside its line, where field parsing rejects it.
    private static func splitLines(_ bytes: [UInt8]) -> [ArraySlice<UInt8>] {
        var lines: [ArraySlice<UInt8>] = []
        var start = bytes.startIndex
        var index = bytes.startIndex
        while index < bytes.endIndex {
            let next = index + 1
            if bytes[index] == carriageReturn, next < bytes.endIndex, bytes[next] == lineFeed {
                lines.append(bytes[start..<index])
                start = next + 1
                index = next + 1
            } else {
                index = next
            }
        }
        if start < bytes.endIndex {
            lines.append(bytes[start..<bytes.endIndex])
        }
        return lines
    }

    /// Parses one `name: value` line.
    private static func parseField(_ line: ArraySlice<UInt8>) -> (name: String, value: String)? {
        guard let separator = line.firstIndex(of: colon), separator > line.startIndex else { return nil }
        let nameBytes = line[line.startIndex..<separator]
        // A token: visible ASCII, which excludes the leading space of a folded line.
        guard nameBytes.allSatisfy({ $0 > space && $0 < 0x7F }),
              let name = String(validatingUTF8Bytes: nameBytes) else {
            return nil
        }

        var valueBytes = line[(separator + 1)...]
        while let first = valueBytes.first, first == space || first == tab {
            valueBytes = valueBytes.dropFirst()
        }
        while let last = valueBytes.last, last == space || last == tab {
            valueBytes = valueBytes.dropLast()
        }
        guard !valueBytes.contains(carriageReturn), !valueBytes.contains(lineFeed),
              let value = String(validatingUTF8Bytes: valueBytes) else {
            return nil
        }
        return (name.lowercased(), value)
    }
}

extension String {
    /// Decodes UTF-8, returning `nil` for an invalid sequence instead of repairing it.
    fileprivate init?(validatingUTF8Bytes bytes: ArraySlice<UInt8>) {
        guard let decoded = String(bytes: bytes, encoding: .utf8) else { return nil }
        self = decoded
    }
}
