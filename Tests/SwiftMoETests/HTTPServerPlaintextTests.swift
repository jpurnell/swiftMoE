import Foundation
import SwiftHTTPService
import Testing
@testable import SwiftMoE
@testable import SwiftMoEHTTP

/// When the service will listen somewhere other machines can reach.
///
/// The key and every prompt cross the network, so off loopback the service terminates TLS itself
/// (`--tls-cert`, `--tls-key`) or does not start. There is no way to say "a proxy handles it":
/// that acknowledgement (`--allow-plaintext`) was removed with the move onto SwiftHTTPService.
extension HTTPServiceTests {
@Suite("HTTP server plain-text exposure")
struct HTTPServerPlaintextTests {

    /// An address from TEST-NET-1 (RFC 5737): not loopback, and never assigned to an interface,
    /// so a listener that gets as far as `bind(2)` fails there instead of opening a port.
    private static let unassignedHost = "192.0.2.1"

    private static func makeService(host: String, tls: TLSFiles? = nil, authenticated: Bool = true) throws -> ChatService {
        let authentication: Authentication = authenticated
            ? .bearer(BearerCredential(key: try APIKey(RunningServer.key)))
            : .unauthenticatedLoopback
        return ChatService(host: host, port: 0, authentication: authentication, tls: tls,
                           tokenizer: RunningServer.byteTokenizer) { _, _, _ in }
    }

    @Test("Off loopback, a key is not enough: without TLS nothing is bound",
          arguments: ["0.0.0.0", "192.0.2.1", "10.0.1.114"])
    func exposedPlaintextIsRefused(host: String) throws {
        let service = try Self.makeService(host: host)
        #expect(throws: ServiceError.exposedPlaintext(host: host)) {
            _ = try blocking { try await service.start() }
        }
    }

    @Test("With a TLS identity the listener goes on to bind")
    func tlsProceeds() throws {
        let tls = try TestTLS()
        defer { tls.cleanUp() }
        let service = try Self.makeService(host: Self.unassignedHost, tls: tls.files)
        // The address belongs to no interface, so reaching bind(2) is as far as this can go:
        // the failure is the socket's, which shows the exposure check let it through.
        #expect(throws: ServiceError.bindFailed(host: Self.unassignedHost)) {
            _ = try blocking { try await service.start() }
        }
    }

    @Test("Loopback needs no TLS")
    func loopbackIsNotExposed() throws {
        let service = try Self.makeService(host: "127.0.0.1")
        let bound = try blocking { try await service.start() }
        defer { blocking { await service.shutdown() } }
        #expect(bound.host == "127.0.0.1")
    }

    @Test("TLS does not excuse a missing key: that is still what is reported")
    func tlsIsNotACredential() throws {
        let tls = try TestTLS()
        defer { tls.cleanUp() }
        let service = try Self.makeService(host: "0.0.0.0", tls: tls.files, authenticated: false)
        #expect(throws: ServiceError.credentialRequired(host: "0.0.0.0")) {
            _ = try blocking { try await service.start() }
        }
    }

    @Test("A certificate and key that cannot be used stop the service from starting, naming the fault")
    func unfitIdentity() throws {
        let service = try Self.makeService(
            host: "127.0.0.1", tls: TLSFiles(certificateChainPath: "/nonexistent/chain.pem", privateKeyPath: "/nonexistent/key.pem"))
        do {
            _ = try blocking { try await service.start() }
            Issue.record("a missing certificate was accepted")
        } catch ServiceError.tlsIdentity(let detail) {
            #expect(detail.contains("/nonexistent/chain.pem"))
        } catch {
            Issue.record("failed with \(error) instead of tlsIdentity")
        }
    }

    @Test("The refusal names the flags and the loopback alternative")
    func refusalMessage() {
        let message = ServiceError.exposedPlaintext(host: "0.0.0.0").errorDescription
        #expect(message == "Refusing to listen on 0.0.0.0 in plain text: other machines can reach it, and the "
            + "API key and every prompt would cross the network unencrypted. Give the server a "
            + "certificate and key (--tls-cert and --tls-key), or bind loopback (--host 127.0.0.1).")
    }
}
}
