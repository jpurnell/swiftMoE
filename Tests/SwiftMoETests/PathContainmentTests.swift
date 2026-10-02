import Foundation
import Testing
@testable import SwiftMoE

/// Whether a path lies inside a directory.
///
/// Six checks — the weight manifest, the session store, the chat and server entry points — used
/// `path.hasPrefix(root.path)`. `/home/u/.flash-moe/sessions-other/x.jsonl` begins with
/// `/home/u/.flash-moe/sessions`, so a session id of `../sessions-other/x` was written outside the
/// sessions directory.
@Suite("Path containment")
struct PathContainmentTests {

    private let root = URL(fileURLWithPath: "/home/u/.flash-moe/sessions")

    @Test("A file in the directory is inside it")
    func childIsContained() {
        #expect(PathContainment.isContained(root.appendingPathComponent("abc.jsonl"), in: root))
    }

    @Test("A sibling whose name extends the directory's is not inside it")
    func siblingIsNotContained() {
        let escaped = root.appendingPathComponent("../sessions-other/x.jsonl")
        #expect(!PathContainment.isContained(escaped, in: root))
    }

    @Test("A path that climbs out is not inside it")
    func dotDotIsNotContained() {
        #expect(!PathContainment.isContained(root.appendingPathComponent("../../.ssh/id"), in: root))
    }

    @Test("A link inside the directory that leads out is not inside it")
    func linkOutIsNotContained() throws {
        let fm = FileManager.default
        let parent = fm.temporaryDirectory.appendingPathComponent("moe-containment-\(UUID().uuidString)")
        let inside = parent.appendingPathComponent("sessions")
        let outside = parent.appendingPathComponent("elsewhere")
        try fm.createDirectory(at: inside, withIntermediateDirectories: true)
        try fm.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: parent) } // silent: test cleanup; a leftover temp directory changes no result
        try fm.createSymbolicLink(at: inside.appendingPathComponent("link"), withDestinationURL: outside)
        #expect(!PathContainment.isContained(inside.appendingPathComponent("link/x.jsonl"), in: inside))
    }
}
