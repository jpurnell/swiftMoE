import Foundation
import Testing
@testable import SwiftMoE
@testable import SwiftMoEHTTP

/// Which address the HTTP service's socket is bound to.
///
/// The old server bound `INADDR_ANY` while its log line said `localhost`, so it was reachable from
/// every network the machine was on. These tests connect to the address the socket reports,
/// because the address a caller asked for and the address a socket holds are different facts.
extension HTTPServiceTests {
@Suite("HTTP server bind address")
struct HTTPServerBindTests {

    private static func makeService(host: String? = nil) -> ChatService {
        let generate: ChatService.Generate = { _, _, _ in }
        guard let host else {
            return ChatService(port: 0, authentication: .unauthenticatedLoopback,
                               tokenizer: RunningServer.byteTokenizer, generate: generate)
        }
        return ChatService(host: host, port: 0, authentication: .unauthenticatedLoopback,
                           tokenizer: RunningServer.byteTokenizer, generate: generate)
    }

    /// Connects a client socket to `host:port`, returning `connect(2)`'s result.
    private static func connectResult(host: String, port: UInt16) -> Int32 {
        var address = in_addr()
        guard inet_pton(AF_INET, host, &address) == 1 else { return -1 }
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr = address

        let size = MemoryLayout<sockaddr_in>.size
        let raw = UnsafeMutableRawPointer.allocate(byteCount: size,
                                                    alignment: MemoryLayout<sockaddr_in>.alignment)
        defer { raw.deallocate() }
        raw.storeBytes(of: addr, as: sockaddr_in.self)
        return connect(fd, raw.assumingMemoryBound(to: sockaddr.self), socklen_t(size))
    }

    @Test("A service created without a host is bound to loopback")
    func defaultBindsLoopback() throws {
        let service = Self.makeService()
        let bound = try blocking { try await service.start() }
        defer { blocking { await service.shutdown() } }

        #expect(bound.host == "127.0.0.1")
        #expect(Self.connectResult(host: "127.0.0.1", port: UInt16(bound.port)) == 0)
    }

    @Test("Port 0 reports the port the kernel assigned, not the one requested")
    func ephemeralPortIsReported() throws {
        let service = Self.makeService()
        let bound = try blocking { try await service.start() }
        defer { blocking { await service.shutdown() } }

        #expect((1...65535).contains(bound.port))
        #expect(Self.connectResult(host: bound.host, port: UInt16(bound.port)) == 0)
    }

    @Test("A stopped service no longer accepts connections")
    func stoppedServiceRefusesConnections() throws {
        let service = Self.makeService()
        let bound = try blocking { try await service.start() }
        blocking { await service.shutdown() }
        #expect(Self.connectResult(host: bound.host, port: UInt16(bound.port)) == -1)
    }

    @Test("A host that is blank, or that is not an address, is refused and binds nothing",
          arguments: ["", "256.0.0.1", "0"])
    func invalidHostIsRefused(host: String) {
        let service = Self.makeService(host: host)
        #expect(throws: ServiceError.self) {
            _ = try blocking { try await service.start() }
        }
    }

    @Test("Loopback is 127.0.0.0/8 and ::1; a name never is, but localhost is a local bind")
    func loopbackClassification() {
        #expect(ChatService.isLoopback("127.0.0.1"))
        #expect(ChatService.isLoopback("127.255.255.254"))
        #expect(ChatService.isLoopback("::1"))
        #expect(!ChatService.isLoopback("localhost"))
        #expect(ChatService.isLocalBind("localhost"))
        #expect(!ChatService.isLocalBind("10.0.1.114"))
        #expect(!ChatService.isLoopback("0.0.0.0"))
        #expect(!ChatService.isLoopback("10.0.1.114"))
        #expect(!ChatService.isLoopback("128.0.0.1"))
        #expect(!ChatService.isLoopback(""))
    }
}
}
