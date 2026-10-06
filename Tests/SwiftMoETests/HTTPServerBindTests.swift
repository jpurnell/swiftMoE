import Foundation
import Testing
@testable import SwiftMoE

/// Which address the HTTP server's socket is bound to.
///
/// The server bound `INADDR_ANY` while its log line said `localhost`, so it was reachable from
/// every network the machine was on. These tests read the address back from the socket itself with `getsockname`,
/// because the address a caller asked for and the address a socket holds are different facts.
@Suite("HTTP server bind address")
struct HTTPServerBindTests {

    private static func makeServer(host: String? = nil) -> HTTPServer {
        guard let host else {
            return HTTPServer(port: 0, authentication: .unauthenticatedLoopback) { _, _, _ in }
        }
        return HTTPServer(host: host, port: 0, authentication: .unauthenticatedLoopback) { _, _, _ in }
    }

    /// Connects a client socket to `host:port`, returning `connect(2)`'s result.
    private static func connectResult(host: String, port: UInt16) -> Int32 {
        guard let address = HTTPServer.ipv4Address(host) else { return -1 }
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = address

        let size = MemoryLayout<sockaddr_in>.size
        let raw = UnsafeMutableRawPointer.allocate(byteCount: size,
                                                    alignment: MemoryLayout<sockaddr_in>.alignment)
        defer { raw.deallocate() }
        raw.storeBytes(of: addr, as: sockaddr_in.self)
        return connect(fd, raw.assumingMemoryBound(to: sockaddr.self), socklen_t(size))
    }

    @Test("A server created without a host is bound to loopback")
    func defaultBindsLoopback() throws {
        let server = Self.makeServer()
        let bound = try server.openListener()
        defer { server.stop() }

        #expect(server.host == "127.0.0.1")
        #expect(bound.host == "127.0.0.1")
        #expect(server.boundAddress == bound)
        #expect(Self.connectResult(host: "127.0.0.1", port: bound.port) == 0)
    }

    @Test("Port 0 reports the port the kernel assigned, not the one requested")
    func ephemeralPortIsReported() throws {
        let server = Self.makeServer()
        let bound = try server.openListener()
        defer { server.stop() }

        #expect(server.port == 0)
        #expect(Self.connectResult(host: bound.host, port: bound.port) == 0)
        #expect(bound.endpoint == "http://127.0.0.1:\(bound.port)/v1/chat/completions")
    }

    @Test("A stopped server has no bound address")
    func stoppedServerHasNoAddress() throws {
        let server = Self.makeServer()
        _ = try server.openListener()
        server.stop()
        #expect(server.boundAddress == nil)
    }

    @Test("A host that is not an IPv4 literal is refused, not resolved or widened")
    func invalidHostIsRefused() {
        for host in ["localhost", "", "256.0.0.1", "::1", "0"] {
            let server = Self.makeServer(host: host)
            do {
                _ = try server.openListener()
                server.stop()
                Issue.record("\(host) was accepted as a bind address")
            } catch FlashMoEError.invalidBindAddress(let refused) {
                #expect(refused == host)
            } catch {
                Issue.record("\(host) failed with \(error) instead of invalidBindAddress")
            }
        }
    }

    @Test("Address parsing: loopback, all interfaces, and a specific interface")
    func addressParsing() {
        #expect(HTTPServer.ipv4Address("127.0.0.1") == INADDR_LOOPBACK.bigEndian)
        #expect(HTTPServer.ipv4Address("0.0.0.0") == INADDR_ANY)
        #expect(HTTPServer.ipv4Address("10.0.1.114") == UInt32(0x0A00_0172).bigEndian)
        #expect(HTTPServer.ipv4Address("localhost") == nil)
    }

    @Test("Only 127.0.0.0/8 counts as loopback")
    func loopbackClassification() {
        #expect(HTTPServer.isLoopback("127.0.0.1"))
        #expect(HTTPServer.isLoopback("127.255.255.254"))
        #expect(!HTTPServer.isLoopback("0.0.0.0"))
        #expect(!HTTPServer.isLoopback("10.0.1.114"))
        #expect(!HTTPServer.isLoopback("localhost"))
    }
}
