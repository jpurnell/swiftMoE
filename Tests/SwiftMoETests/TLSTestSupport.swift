import Foundation
import NIOCore
import NIOPosix
import SwiftHTTPService
import SwiftTLSKit
import SwiftTLSKitTestSupport
import Testing

/// A generated authority and leaf, written to a temporary directory as the files a server reads.
///
/// SwiftTLSKitTestSupport makes the certificates when the test runs, so no key is checked in. The key
/// file is written with mode 0600, because SwiftTLSKit refuses a private key that others can read.
struct TestTLS: Sendable {
    let directory: URL
    let authority: TestAuthority
    let chainPath: String
    let keyPath: String

    var files: TLSFiles { TLSFiles(certificateChainPath: chainPath, privateKeyPath: keyPath) }

    init(seed: UInt64 = 1) throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("tls-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        chainPath = directory.appendingPathComponent("chain.pem").path
        keyPath = directory.appendingPathComponent("key.pem").path
        authority = try TestAuthority(commonName: "Test Authority", generator: SeededGenerator(seed: seed))
        let leaf = try authority.issueLeaf(dnsNames: ["localhost"], ipv4Addresses: [[127, 0, 0, 1]])
        try Data(leaf.chainPEM.utf8).write(to: URL(fileURLWithPath: chainPath))
        FileManager.default.createFile(
            atPath: keyPath, contents: Data(leaf.privateKeyPEM.utf8), attributes: [.posixPermissions: 0o600])
    }

    func clientContext() throws -> ClientTLSContext {
        try authority.clientContext(applicationProtocols: ["http/1.1"])
    }

    func cleanUp() {
        try? FileManager.default.removeItem(at: directory)
    }
}

/// Bytes a TLS client received, collected on the event loop.
// Justification: the handler's state is touched only by its channel's event loop and read after the channel has closed.
final class TLSCollector: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = ByteBuffer
    private var bytes = ByteBuffer()
    private let done: EventLoopPromise<String>

    init(done: EventLoopPromise<String>) { self.done = done }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        var buffer = unwrapInboundIn(data)
        bytes.writeBuffer(&buffer)
    }

    func channelInactive(context: ChannelHandlerContext) {
        done.succeed(String(decoding: bytes.readableBytesView, as: UTF8.self))
    }

    func errorCaught(context: ChannelHandlerContext, error: any Error) {
        context.close(promise: nil)
    }
}

/// Sends `request` over TLS to `127.0.0.1:port` and returns everything the server wrote before it closed.
func tlsExchange(port: Int, tls: TestTLS, request: String) async throws -> String {
    let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    let context = try tls.clientContext()
    let channel = try await ClientBootstrap(group: group)
        .channelInitializer { channel in
            channel.eventLoop.makeCompletedFuture {
                try context.addHandlers(to: channel.pipeline.syncOperations, serverHostname: "localhost")
            }
        }
        .connect(host: "127.0.0.1", port: port)
        .get()
    let done = channel.eventLoop.makePromise(of: String.self)
    try await channel.pipeline.addHandler(TLSCollector(done: done)).get()
    try await channel.writeAndFlush(ByteBuffer(string: request)).get()
    let received = try await done.futureResult.get()
    try await group.shutdownGracefully()
    return received
}
