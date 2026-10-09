import CryptoKit
import Foundation
#if canImport(os)
import os
#endif

private let logger = Logger(subsystem: "com.swiftmoe", category: "server.credential")

/// Why an ``HTTPServer`` refused its configuration.
///
/// Each case is something an operator has to fix before the server will listen; none is
/// produced by a request.
public enum HTTPServerError: Error, Equatable, Sendable, LocalizedError {
    /// The key is shorter than ``APIKey/minimumLength`` characters.
    case keyTooShort(minimum: Int)
    /// The key contains a space, a control character, or anything outside printable ASCII.
    case keyNotPrintable
    /// The key file is missing, unreadable, or not a regular file.
    case keyFileUnreadable(path: String)
    /// The key file grants some permission to group or other.
    case keyFilePermissions(path: String, mode: Int)
    /// The key file is larger than ``APIKey/maximumFileBytes``.
    case keyFileTooLarge(path: String, limit: Int)
    /// No key was supplied, and nothing excused the listener on `host` from needing one.
    case credentialRequired(host: String)
    /// `--no-auth` was given together with a key.
    case conflictingAuthentication
    /// An allowed origin is not of the form `scheme://host[:port]`.
    case invalidOrigin(String)
    /// A limit in ``HTTPServer/Limits`` is zero or negative.
    case invalidLimit(name: String)
    /// One limit in ``HTTPServer/Limits`` is larger than another that bounds it.
    case limitAboveLimit(name: String, ceiling: String)
    /// A command-line option that takes a value was given without one.
    case missingValue(option: String)

    /// A sentence an operator can act on.
    public var errorDescription: String? {
        switch self {
        case .keyTooShort(let minimum):
            return "The API key must be at least \(minimum) characters. Generate one with `openssl rand -hex 32`."
        case .keyNotPrintable:
            return "The API key must be printable ASCII with no spaces."
        case .keyFileUnreadable(let path):
            return "The API key file \(path) could not be read as a regular file."
        case .keyFilePermissions(let path, let mode):
            return "The API key file \(path) has mode \(String(mode, radix: 8)); it must not be readable, writable or executable by group or other (chmod 600)."
        case .keyFileTooLarge(let path, let limit):
            return "The API key file \(path) is larger than \(limit) bytes."
        case .credentialRequired(let host):
            let remedy = "No API key is configured for the listener on \(host). Set \(APIKey.environmentVariable) or pass --api-key-file <path>."
            if HTTPServer.isLoopback(host) {
                return remedy + " On loopback, --no-auth starts it without one."
            }
            return remedy + " A listener that is not on loopback cannot run without one."
        case .conflictingAuthentication:
            return "--no-auth was given together with an API key. Remove one of them."
        case .invalidOrigin(let origin):
            return "\"\(origin)\" is not an origin. Use scheme://host[:port] with no path, for example http://localhost:3000."
        case .invalidLimit(let name):
            return "The limit \(name) must be greater than zero."
        case .limitAboveLimit(let name, let ceiling):
            return "The limit \(name) must not be greater than \(ceiling)."
        case .missingValue(let option):
            return "\(option) requires a value."
        }
    }
}

/// The shared key a caller presents as `Authorization: Bearer <key>`.
///
/// A key is only ever read from the environment or from a file — there is no literal default,
/// and its text is not reachable through `description`, so it cannot be interpolated into a log
/// line by accident. The server does not keep an `APIKey`: it keeps a ``BearerCredential``,
/// which holds the key's SHA-256 digest. The chat client keeps one, because it has to send it.
///
/// ```swift
/// let key = try APIKey.load(keyFile: nil, environment: ["SWIFT_MOE_API_KEY": String(repeating: "k", count: 32)])
/// print(key.map(String.init(describing:)) ?? "none")   // APIKey(<redacted>)
/// ```
public struct APIKey: Sendable, CustomStringConvertible, CustomDebugStringConvertible {

    /// Shortest key accepted, in characters.
    ///
    /// The key is the only thing between the network and the model, and it is guessed online,
    /// one request at a time. 32 hexadecimal characters carry 128 bits, which is the floor;
    /// `openssl rand -hex 32` produces 64.
    public static let minimumLength = 32

    /// Largest key file read, in bytes. A key is a line of text; anything larger is the wrong file.
    public static let maximumFileBytes = 4096

    /// Name of the environment variable a key is read from.
    public static let environmentVariable = "SWIFT_MOE_API_KEY"

    private let text: String

    /// Wraps key text, after trimming surrounding whitespace.
    ///
    /// - Parameter text: The key, as read from a variable or a file.
    /// - Throws: ``HTTPServerError/keyTooShort(minimum:)`` under ``minimumLength`` characters;
    ///   ``HTTPServerError/keyNotPrintable`` when the key holds anything but printable ASCII —
    ///   a space or a line break inside a key cannot be carried in a header.
    public init(_ text: String) throws {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.utf8.count >= Self.minimumLength else {
            throw HTTPServerError.keyTooShort(minimum: Self.minimumLength)
        }
        guard trimmed.utf8.allSatisfy({ $0 > 0x20 && $0 < 0x7F }) else {
            throw HTTPServerError.keyNotPrintable
        }
        self.text = trimmed
    }

    /// Reads a key from a file that only its owner can access.
    ///
    /// The permission check is made with `fstat(2)` on the descriptor the key is then read
    /// from, so the file checked and the file read are the same file.
    ///
    /// - Parameter url: Location of the key file.
    /// - Throws: ``HTTPServerError/keyFileUnreadable(path:)``,
    ///   ``HTTPServerError/keyFilePermissions(path:mode:)`` when group or other hold any
    ///   permission bit, ``HTTPServerError/keyFileTooLarge(path:limit:)``, or the errors of
    ///   ``init(_:)``.
    public init(contentsOf url: URL) throws {
        let path = url.path
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw HTTPServerError.keyFileUnreadable(path: path)
        }
        defer {
            do {
                try handle.close()
            } catch {
                // Nothing was written through this handle, so a failed close loses no data.
                logger.debug("Closing the API key file failed: \(error.localizedDescription, privacy: .public)")
            }
        }

        var status = stat()
        guard fstat(handle.fileDescriptor, &status) == 0,
              status.st_mode & S_IFMT == S_IFREG else {
            throw HTTPServerError.keyFileUnreadable(path: path)
        }
        let mode = Int(status.st_mode & 0o777)
        guard mode & 0o077 == 0 else {
            throw HTTPServerError.keyFilePermissions(path: path, mode: mode)
        }

        let contents: Data
        do {
            contents = try handle.read(upToCount: Self.maximumFileBytes + 1) ?? Data()
        } catch {
            throw HTTPServerError.keyFileUnreadable(path: path)
        }
        guard contents.count <= Self.maximumFileBytes else {
            throw HTTPServerError.keyFileTooLarge(path: path, limit: Self.maximumFileBytes)
        }
        try self.init(String(decoding: contents, as: UTF8.self))
    }

    /// Finds the configured key, if there is one.
    ///
    /// A key file wins over the environment: a path typed on the command line is the more
    /// deliberate of the two. A variable that is set but unusable is an error rather than
    /// "no key", so a typo cannot turn into an unauthenticated start.
    ///
    /// - Parameters:
    ///   - keyFile: Path given as `--api-key-file`, or `nil`.
    ///   - environment: The process environment.
    /// - Returns: The key, or `nil` when neither source is present.
    /// - Throws: The errors of ``init(contentsOf:)`` and ``init(_:)``.
    public static func load(keyFile: String?, environment: [String: String]) throws -> APIKey? {
        if let keyFile {
            return try APIKey(contentsOf: URL(fileURLWithPath: keyFile).standardizedFileURL)
        }
        if let value = environment[environmentVariable] {
            return try APIKey(value)
        }
        return nil
    }

    /// The `Authorization` header value a client sends: `Bearer <key>`.
    public var authorizationHeaderValue: String { "Bearer \(text)" }

    /// SHA-256 of the key's UTF-8 bytes.
    var digest: [UInt8] { Array(SHA256.hash(data: Data(text.utf8))) }

    /// Always `APIKey(<redacted>)`.
    public var description: String { "APIKey(<redacted>)" }

    /// Always `APIKey(<redacted>)`.
    public var debugDescription: String { description }
}

/// What the server keeps of its key: the SHA-256 digest, and a way to compare against it.
///
/// A presented key is hashed and the two digests are compared without an early exit, so the
/// time a refusal takes says nothing about how much of a guess was right. Hashing first also
/// makes both sides the same length whatever was presented.
public struct BearerCredential: Sendable, CustomStringConvertible {

    private let digest: [UInt8]

    /// Derives the credential from a key. The key itself is not retained.
    ///
    /// - Parameter key: The configured key.
    public init(key: APIKey) {
        self.digest = key.digest
    }

    /// Whether an `Authorization` header value carries the configured key.
    ///
    /// - Parameter authorization: The header's value, or `nil` when the header is absent.
    /// - Returns: `true` only for the scheme `Bearer` (any letter case) followed by the key.
    public func accepts(authorization: String?) -> Bool {
        guard let authorization else { return false }
        let parts = authorization.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
        guard parts.count == 2, parts[0].lowercased() == "bearer" else { return false }
        let presented = parts[1].trimmingCharacters(in: .whitespaces)
        guard !presented.isEmpty else { return false }
        let presentedDigest = Array(SHA256.hash(data: Data(presented.utf8)))
        return Self.constantTimeEqual(presentedDigest, digest)
    }

    /// Compares two byte strings, touching every byte of both whatever they hold.
    ///
    /// - Returns: `true` when the two are the same length and equal.
    static func constantTimeEqual(_ lhs: [UInt8], _ rhs: [UInt8]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        var difference: UInt8 = 0
        for index in lhs.indices {
            difference |= lhs[index] ^ rhs[index]
        }
        return difference == 0
    }

    /// Always `BearerCredential(<sha256 digest>)`; the digest is not printed either.
    public var description: String { "BearerCredential(<sha256 digest>)" }
}

extension HTTPServer {

    /// How a listener decides who may run inference.
    ///
    /// There is no default: every ``HTTPServer`` is created with one of these written down.
    public enum Authentication: Sendable {
        /// Every request must carry `Authorization: Bearer <key>`.
        case bearer(BearerCredential)

        /// No credential is checked. Accepted on a loopback bind only:
        /// ``HTTPServer/openListener()`` throws ``HTTPServerError/credentialRequired(host:)``
        /// for any other address.
        case unauthenticatedLoopback

        /// Turns what an operator supplied into a decision.
        ///
        /// | key | `noAuth` | host | result |
        /// |---|---|---|---|
        /// | present | no | any | ``bearer(_:)`` |
        /// | present | yes | any | ``HTTPServerError/conflictingAuthentication`` |
        /// | absent | yes | loopback | ``unauthenticatedLoopback`` |
        /// | absent | yes | other | ``HTTPServerError/credentialRequired(host:)`` |
        /// | absent | no | any | ``HTTPServerError/credentialRequired(host:)`` |
        ///
        /// - Parameters:
        ///   - host: The address the listener will bind.
        ///   - keyFile: Path given as `--api-key-file`, or `nil`.
        ///   - environment: The process environment.
        ///   - noAuth: Whether `--no-auth` was given.
        /// - Returns: The authentication the listener will enforce.
        /// - Throws: ``HTTPServerError`` as tabulated, and the key-loading errors of
        ///   ``APIKey/load(keyFile:environment:)``.
        public static func resolve(
            host: String,
            keyFile: String?,
            environment: [String: String],
            noAuth: Bool
        ) throws -> Authentication {
            let key = try APIKey.load(keyFile: keyFile, environment: environment)
            if let key {
                guard !noAuth else { throw HTTPServerError.conflictingAuthentication }
                return .bearer(BearerCredential(key: key))
            }
            guard noAuth, HTTPServer.isLoopback(host) else {
                throw HTTPServerError.credentialRequired(host: host)
            }
            // SECURITY: the operator passed --no-auth and the bind address is loopback, both checked above
            return .unauthenticatedLoopback
        }
    }
}
