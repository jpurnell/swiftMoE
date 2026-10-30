# Session Summary — Kit 0.6.0, v0.2.1

**Date:** 2026-10-30
**Branch:** `fix/kit-0.6.0` (from `main` at v0.2.0), merged by fast-forward, tagged `v0.2.1`
**Outcome:** SwiftHTTPService 0.6.0 fixes what 0.2.0 recorded as a known regression. Gate 0/0; 255 tests.

## What changed
- Dependency `.upToNextMinor(from: "0.6.0")`; `connectionReuse:` replaces `connections:`.
- `ChatService.refusal(for:)` maps `.busy` (old text, `server_error`) and `.versionNotSupported` (old 505 text).
- `Limits.busyRetryAfter` = the queue deadline in whole seconds, rounded up: `Retry-After: 30` by default.
- `idleTimeout` = `readDeadline`, so a silent connection is answered 408 at the same deadline as before.
- The replay table shrank from nine `intended` to six. The 503 past the cap, the 505 and the 413 phrase
  moved out and are now required to match the recording; the silent-client 408 is a ported Limits test.
  Each of the six that remain was rechecked against the old fixture by removing its marker and replaying.

## Mutations (all killed)
X1 `busyRetryAfter` not wired; X2 a constant; X3 busy mapped to the internal-error body; X4 505 mapped to
the malformed body; X5 idle deadline left at the kit default; X6 busy message changed.

## Not verified
Over TLS the over-cap connection is closed before the handshake by design (kit); not exercised here.
