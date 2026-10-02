import Foundation

/// Whether a path lies inside a directory.
///
/// One check, used wherever a path is about to be read or written under a directory this package
/// owns — sessions, weights, temp files. It replaces `path.hasPrefix(base.path)`, which was
/// wrong in two ways that a reader of the call site could not see:
///
/// - **No separator.** `…/sessions-other` begins with `…/sessions`. Comparing whole path
///   components does not have that problem.
/// - **Links.** `standardizedFileURL` removes `..` and does not follow symbolic links, so a
///   link inside the directory that leads out of it passed. Both sides are resolved first.
public enum PathContainment {

    /// Whether `url` is `base` or lies beneath it, after resolving `..` and symbolic links.
    ///
    /// - Parameters:
    ///   - url: The path about to be used.
    ///   - base: The directory it must stay inside.
    /// - Returns: `true` when every component of `base` is a leading component of `url`.
    public static func isContained(_ url: URL, in base: URL) -> Bool {
        resolved(url).starts(with: resolved(base))
    }

    /// The path's components with `..` removed and every symbolic link followed.
    ///
    /// `resolvingSymlinksInPath()` gives up on a path whose last component does not exist —
    /// the usual case for a file about to be written — and returns it unresolved, link and
    /// all. So the deepest ancestor that does exist is resolved, and the rest is appended.
    private static func resolved(_ url: URL) -> [String] {
        var existing = url.standardizedFileURL
        var pending: [String] = []
        // SECURITY: an existence probe on the path being judged; nothing is opened, read or written here
        while !FileManager.default.fileExists(atPath: existing.path), existing.pathComponents.count > 1 {
            pending.insert(existing.lastPathComponent, at: 0)
            existing = existing.deletingLastPathComponent()
        }
        return existing.resolvingSymlinksInPath().pathComponents + pending
    }
}
