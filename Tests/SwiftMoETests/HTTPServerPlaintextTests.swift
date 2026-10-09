import Foundation
import Testing
@testable import SwiftMoE

/// When the server will listen in plain text somewhere other machines can reach.
///
/// The server does not speak TLS. Off loopback it requires a bearer key and then sends that
/// key, and every prompt, unencrypted — so a bind other machines can reach has to be asked
/// for by name, by an operator who has put something that does speak TLS in front of it.
@Suite("HTTP server plain-text exposure")
struct HTTPServerPlaintextTests {

    /// An address from TEST-NET-1 (RFC 5737): not loopback, and never assigned to an interface,
    /// so a listener that gets as far as `bind(2)` fails there instead of opening a port.
    private static let unassignedHost = "192.0.2.1"

    private static func makeServer(host: String, allowPlaintext: Bool, authenticated: Bool = true) throws -> HTTPServer {
        let authentication: HTTPServer.Authentication = authenticated
            ? .bearer(BearerCredential(key: try APIKey(RunningServer.key)))
            : .unauthenticatedLoopback
        return HTTPServer(host: host, port: 0, authentication: authentication, allowPlaintext: allowPlaintext,
                          tokenizer: RunningServer.byteTokenizer) { _, _ in }
    }

    @Test("Off loopback, a key is not enough: without the acknowledgement nothing is bound",
          arguments: ["0.0.0.0", "192.0.2.1", "10.0.1.114"])
    func exposedPlaintextIsRefused(host: String) throws {
        let server = try Self.makeServer(host: host, allowPlaintext: false)
        #expect(throws: HTTPServerError.plaintextNotAcknowledged(host: host)) {
            _ = try server.openListener()
        }
        #expect(server.boundAddress == nil)
        #expect(!server.allowsPlaintext)
    }

    @Test("With the acknowledgement the listener goes on to bind")
    func acknowledgedPlaintextProceeds() throws {
        let server = try Self.makeServer(host: Self.unassignedHost, allowPlaintext: true)
        #expect(server.allowsPlaintext)
        // The address belongs to no interface, so reaching bind(2) is as far as this can go:
        // the failure is the socket's, which shows the exposure check let it through.
        #expect(throws: FlashMoEError.readFailed(errno: EADDRNOTAVAIL, context: "bind(192.0.2.1:0)")) {
            _ = try server.openListener()
        }
        #expect(server.boundAddress == nil)
    }

    @Test("Loopback needs no acknowledgement, and giving one changes nothing",
          arguments: [false, true])
    func loopbackIsNotExposed(allowPlaintext: Bool) throws {
        let server = try Self.makeServer(host: "127.0.0.1", allowPlaintext: allowPlaintext)
        let bound = try server.openListener()
        defer { server.stop() }
        #expect(bound.host == "127.0.0.1")
    }

    @Test("The acknowledgement does not excuse a missing key: that is still what is reported")
    func acknowledgementIsNotACredential() throws {
        let server = try Self.makeServer(host: "0.0.0.0", allowPlaintext: true, authenticated: false)
        #expect(throws: HTTPServerError.credentialRequired(host: "0.0.0.0")) {
            _ = try server.openListener()
        }
        #expect(server.boundAddress == nil)
    }

    @Test("The check on its own: loopback passes, anything else needs the acknowledgement")
    func requirePlaintextAcknowledged() throws {
        try HTTPServer.requirePlaintextAcknowledged(host: "127.0.0.1", allowPlaintext: false)
        try HTTPServer.requirePlaintextAcknowledged(host: "127.8.9.10", allowPlaintext: false)
        try HTTPServer.requirePlaintextAcknowledged(host: "0.0.0.0", allowPlaintext: true)
        #expect(throws: HTTPServerError.plaintextNotAcknowledged(host: "0.0.0.0")) {
            try HTTPServer.requirePlaintextAcknowledged(host: "0.0.0.0", allowPlaintext: false)
        }
        // Not an address at all: not loopback, so not excused.
        #expect(throws: HTTPServerError.plaintextNotAcknowledged(host: "localhost")) {
            try HTTPServer.requirePlaintextAcknowledged(host: "localhost", allowPlaintext: false)
        }
    }

    @Test("The refusal names the flag, what it asserts, and the loopback alternative")
    func refusalMessage() {
        let message = HTTPServerError.plaintextNotAcknowledged(host: "0.0.0.0").errorDescription
        #expect(message == "Refusing to listen on 0.0.0.0 in plain text: other machines can reach it, and the "
            + "API key and every prompt would cross the network unencrypted. This server does not speak TLS. "
            + "Bind loopback (--host 127.0.0.1) and put a TLS-terminating proxy on this machine in front of it; "
            + "or, if a TLS-terminating proxy on the same trust boundary already fronts this port, "
            + "pass --allow-plaintext.")
    }
}
