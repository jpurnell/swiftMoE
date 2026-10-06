import Foundation
import Testing
@testable import SwiftMoE

/// Where the server's credential comes from, and how a presented one is compared.
///
/// The server ran inference for anything that could connect. A caller now presents
/// `Authorization: Bearer <key>`; the key is read from the environment or from a file nobody
/// else can read, and what the server keeps is its SHA-256 digest, not the key.
@Suite("API key and bearer credential")
struct APIKeyTests {

    /// 32 characters: exactly the minimum length.
    static let minimalKey = String(repeating: "k", count: APIKey.minimumLength)
    /// 40 characters, distinct from `minimalKey` in every position.
    static let otherKey = String(repeating: "z", count: 40)

    private static func scratchFile(mode: Int, contents: String) throws -> URL {
        let url = try #require(TestPaths.resolve("moe-api-key-\(UUID().uuidString)",
                                                 within: TestPaths.scratchRoot))
        try Data(contents.utf8).write(to: url)
        try FileManager.default.setAttributes([.posixPermissions: mode], ofItemAtPath: url.path)
        return url
    }

    private static func remove(_ url: URL) {
        try? FileManager.default.removeItem(at: url) // silent: test cleanup; a leftover temp file changes no result
    }

    // MARK: - Key text

    @Test("A key of the minimum length is accepted, and one character fewer is refused")
    func minimumLength() throws {
        #expect(APIKey.minimumLength == 32)
        let key = try APIKey(Self.minimalKey)
        #expect(key.authorizationHeaderValue == "Bearer \(Self.minimalKey)")

        #expect(throws: HTTPServerError.keyTooShort(minimum: 32)) {
            _ = try APIKey(String(repeating: "k", count: 31))
        }
        #expect(throws: HTTPServerError.keyTooShort(minimum: 32)) {
            _ = try APIKey("")
        }
    }

    @Test("Surrounding whitespace is trimmed; whitespace inside a key is refused")
    func whitespace() throws {
        let key = try APIKey("  \(Self.minimalKey)\n")
        #expect(key.authorizationHeaderValue == "Bearer \(Self.minimalKey)")

        #expect(throws: HTTPServerError.keyNotPrintable) {
            _ = try APIKey(String(repeating: "k", count: 20) + " " + String(repeating: "k", count: 20))
        }
        #expect(throws: HTTPServerError.keyNotPrintable) {
            _ = try APIKey(String(repeating: "k", count: 20) + "\r\n" + String(repeating: "k", count: 20))
        }
    }

    @Test("A key never appears in its own description")
    func descriptionIsRedacted() throws {
        let key = try APIKey(Self.minimalKey)
        #expect(String(describing: key) == "APIKey(<redacted>)")
        #expect(String(reflecting: key) == "APIKey(<redacted>)")
        #expect("\(BearerCredential(key: key))" == "BearerCredential(<sha256 digest>)")
    }

    // MARK: - Key file

    @Test("A key file readable only by its owner is read, trailing newline and all")
    func ownerOnlyFile() throws {
        let url = try Self.scratchFile(mode: 0o600, contents: Self.otherKey + "\n")
        defer { Self.remove(url) }
        let key = try APIKey(contentsOf: url)
        #expect(key.authorizationHeaderValue == "Bearer \(Self.otherKey)")
    }

    @Test("A key file that group or other can read is refused, with the mode reported",
          arguments: [0o640, 0o604, 0o644, 0o660, 0o666, 0o610, 0o601])
    func permissiveFile(mode: Int) throws {
        let url = try Self.scratchFile(mode: mode, contents: Self.otherKey)
        defer { Self.remove(url) }
        #expect(throws: HTTPServerError.keyFilePermissions(path: url.path, mode: mode)) {
            _ = try APIKey(contentsOf: url)
        }
    }

    @Test("A missing key file, a directory, and an oversized file are each refused")
    func unusableFiles() throws {
        let missing = try #require(TestPaths.resolve("moe-api-key-missing-\(UUID().uuidString)",
                                                     within: TestPaths.scratchRoot))
        #expect(throws: HTTPServerError.keyFileUnreadable(path: missing.path)) {
            _ = try APIKey(contentsOf: missing)
        }

        #expect(throws: HTTPServerError.keyFileUnreadable(path: TestPaths.scratchRoot.path)) {
            _ = try APIKey(contentsOf: TestPaths.scratchRoot)
        }

        let big = try Self.scratchFile(mode: 0o600,
                                       contents: String(repeating: "k", count: APIKey.maximumFileBytes + 1))
        defer { Self.remove(big) }
        #expect(throws: HTTPServerError.keyFileTooLarge(path: big.path, limit: 4096)) {
            _ = try APIKey(contentsOf: big)
        }
    }

    // MARK: - Sources

    @Test("With no file and no variable there is no key")
    func noSource() throws {
        let key = try APIKey.load(keyFile: nil, environment: ["PATH": "/usr/bin"])
        #expect(key == nil)
    }

    @Test("The variable SWIFT_MOE_API_KEY supplies the key; set but too short is an error, not absence")
    func environmentSource() throws {
        #expect(APIKey.environmentVariable == "SWIFT_MOE_API_KEY")
        let key = try APIKey.load(keyFile: nil, environment: ["SWIFT_MOE_API_KEY": Self.minimalKey])
        #expect(key?.authorizationHeaderValue == "Bearer \(Self.minimalKey)")

        #expect(throws: HTTPServerError.keyTooShort(minimum: 32)) {
            _ = try APIKey.load(keyFile: nil, environment: ["SWIFT_MOE_API_KEY": ""])
        }
    }

    @Test("A key file takes precedence over the variable")
    func fileBeatsEnvironment() throws {
        let url = try Self.scratchFile(mode: 0o600, contents: Self.otherKey)
        defer { Self.remove(url) }
        let key = try APIKey.load(keyFile: url.path, environment: ["SWIFT_MOE_API_KEY": Self.minimalKey])
        #expect(key?.authorizationHeaderValue == "Bearer \(Self.otherKey)")
    }

    // MARK: - Comparison

    @Test("Only `Bearer <the key>` is accepted")
    func acceptance() throws {
        let credential = BearerCredential(key: try APIKey(Self.minimalKey))
        #expect(credential.accepts(authorization: "Bearer \(Self.minimalKey)"))
        #expect(credential.accepts(authorization: "bearer \(Self.minimalKey)"))
        #expect(credential.accepts(authorization: "Bearer   \(Self.minimalKey)  "))

        #expect(!credential.accepts(authorization: nil))
        #expect(!credential.accepts(authorization: ""))
        #expect(!credential.accepts(authorization: "Bearer"))
        #expect(!credential.accepts(authorization: "Bearer "))
        #expect(!credential.accepts(authorization: Self.minimalKey))
        #expect(!credential.accepts(authorization: "Basic \(Self.minimalKey)"))
        #expect(!credential.accepts(authorization: "Bearer \(Self.otherKey)"))
        #expect(!credential.accepts(authorization: "Bearer \(Self.minimalKey)k"))
        #expect(!credential.accepts(authorization: "Bearer \(String(Self.minimalKey.dropLast()))"))
        #expect(!credential.accepts(authorization: "Bearer \(Self.minimalKey.uppercased())"))
    }

    @Test("Digest comparison: equal, differing in one byte, differing in length")
    func constantTimeEquality() {
        let digest: [UInt8] = Array(0..<32)
        var lastByteDiffers = digest
        lastByteDiffers[31] = 0xFF
        var firstByteDiffers = digest
        firstByteDiffers[0] = 0xFF

        #expect(BearerCredential.constantTimeEqual(digest, digest))
        #expect(!BearerCredential.constantTimeEqual(digest, lastByteDiffers))
        #expect(!BearerCredential.constantTimeEqual(digest, firstByteDiffers))
        #expect(!BearerCredential.constantTimeEqual(digest, Array(digest.dropLast())))
        #expect(!BearerCredential.constantTimeEqual(digest, []))
        #expect(BearerCredential.constantTimeEqual([], []))
    }

    // MARK: - Deciding how a listener authenticates

    @Test("A key means bearer authentication, on loopback and off it")
    func keyMeansBearer() throws {
        for host in ["127.0.0.1", "0.0.0.0", "10.0.1.114"] {
            let resolved = try HTTPServer.Authentication.resolve(
                host: host, keyFile: nil,
                environment: ["SWIFT_MOE_API_KEY": Self.minimalKey], noAuth: false)
            guard case .bearer(let credential) = resolved else {
                Issue.record("\(host) resolved to \(resolved) instead of bearer")
                continue
            }
            #expect(credential.accepts(authorization: "Bearer \(Self.minimalKey)"))
        }
    }

    @Test("No key and no flag refuses to start, on loopback and off it")
    func noKeyRefuses() {
        for host in ["127.0.0.1", "0.0.0.0", "10.0.1.114"] {
            #expect(throws: HTTPServerError.credentialRequired(host: host)) {
                _ = try HTTPServer.Authentication.resolve(host: host, keyFile: nil,
                                                          environment: [:], noAuth: false)
            }
        }
    }

    @Test("--no-auth is honoured on loopback only")
    func noAuthIsLoopbackOnly() throws {
        let resolved = try HTTPServer.Authentication.resolve(host: "127.0.0.1", keyFile: nil,
                                                             environment: [:], noAuth: true)
        guard case .unauthenticatedLoopback = resolved else {
            Issue.record("loopback --no-auth resolved to \(resolved)")
            return
        }

        for host in ["0.0.0.0", "10.0.1.114", "localhost"] {
            #expect(throws: HTTPServerError.credentialRequired(host: host)) {
                _ = try HTTPServer.Authentication.resolve(host: host, keyFile: nil,
                                                          environment: [:], noAuth: true)
            }
        }
    }

    @Test("--no-auth together with a key is a contradiction, not a silent choice")
    func noAuthWithKeyIsRefused() {
        #expect(throws: HTTPServerError.conflictingAuthentication) {
            _ = try HTTPServer.Authentication.resolve(
                host: "127.0.0.1", keyFile: nil,
                environment: ["SWIFT_MOE_API_KEY": Self.minimalKey], noAuth: true)
        }
    }

    @Test("Every refusal has a written explanation")
    func errorDescriptions() {
        #expect(HTTPServerError.keyTooShort(minimum: 32).errorDescription
            == "The API key must be at least 32 characters. Generate one with `openssl rand -hex 32`.")
        #expect(HTTPServerError.keyNotPrintable.errorDescription
            == "The API key must be printable ASCII with no spaces.")
        #expect(HTTPServerError.keyFileUnreadable(path: "/x/key").errorDescription
            == "The API key file /x/key could not be read as a regular file.")
        #expect(HTTPServerError.keyFilePermissions(path: "/x/key", mode: 0o644).errorDescription
            == "The API key file /x/key has mode 644; it must not be readable, writable or executable by group or other (chmod 600).")
        #expect(HTTPServerError.keyFileTooLarge(path: "/x/key", limit: 4096).errorDescription
            == "The API key file /x/key is larger than 4096 bytes.")
        #expect(HTTPServerError.credentialRequired(host: "0.0.0.0").errorDescription
            == "No API key is configured for the listener on 0.0.0.0. Set SWIFT_MOE_API_KEY or pass --api-key-file <path>. A listener that is not on loopback cannot run without one.")
        #expect(HTTPServerError.credentialRequired(host: "127.0.0.1").errorDescription
            == "No API key is configured for the listener on 127.0.0.1. Set SWIFT_MOE_API_KEY or pass --api-key-file <path>. On loopback, --no-auth starts it without one.")
        #expect(HTTPServerError.conflictingAuthentication.errorDescription
            == "--no-auth was given together with an API key. Remove one of them.")
        #expect(HTTPServerError.invalidOrigin("*").errorDescription
            == "\"*\" is not an origin. Use scheme://host[:port] with no path, for example http://localhost:3000.")
        #expect(HTTPServerError.invalidLimit(name: "maxBodyBytes").errorDescription
            == "The limit maxBodyBytes must be greater than zero.")
        #expect(HTTPServerError.missingValue(option: "--api-key-file").errorDescription
            == "--api-key-file requires a value.")
    }
}
