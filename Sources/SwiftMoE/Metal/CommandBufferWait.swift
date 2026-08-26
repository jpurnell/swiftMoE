import Metal

extension MTLCommandBuffer {
    /// Waits for this command buffer and traps if the dispatch did not succeed.
    ///
    /// A failed command buffer leaves its output buffers holding whatever was in
    /// them before the dispatch. Callers that read results straight after the
    /// wait therefore cannot distinguish a GPU fault from a successful run that
    /// computed different numbers — the failure would surface much later as
    /// wrong tokens, with nothing pointing back at the dispatch that caused it.
    /// Checking `status` here turns that into an immediate, located failure.
    ///
    /// - Parameter label: Names the dispatch in the trap message.
    func waitUntilCompletedChecked(
        _ label: String,
        file: StaticString = #fileID,
        line: UInt = #line
    ) {
        waitUntilCompleted()
        guard self.status == .completed else {
            let reason = self.error.map { String(describing: $0) } ?? "status rawValue \(self.status.rawValue)"
            preconditionFailure("GPU dispatch '\(label)' did not complete: \(reason)", file: file, line: line)
        }
    }
}
