import Metal

extension MTLComputeCommandEncoder {
    /// Dispatches exactly `threadCount` threads instead of rounding the grid up
    /// to whole threadgroups.
    ///
    /// `dispatchThreadgroups` takes a count of threadgroups, so a `threadCount`
    /// that is not a multiple of the threadgroup width launches surplus threads
    /// that go on to index past the end of every buffer. `dispatchThreads` sizes
    /// the final threadgroup to fit, so those threads are never created.
    ///
    /// The kernel must tolerate a final threadgroup that is only partially
    /// populated. Two shapes qualify:
    ///
    /// - Threads are independent — no threadgroup barrier, no SIMD-group
    ///   reduction — so a short group is simply less work.
    /// - The kernel derives its own width from `[[threads_per_threadgroup]]`
    ///   rather than assuming one, and `threadCount` is a whole multiple of the
    ///   SIMD width so any reduction still sees full SIMD groups.
    ///
    /// A kernel that hardcodes its threadgroup width and then reduces across it
    /// does not qualify and must keep using `dispatchThreadgroups`.
    ///
    /// Devices without non-uniform threadgroup support fall back to the rounded
    /// dispatch, which is what the call sites did before; every kernel routed
    /// through here also bounds its own thread id, so the fallback stays safe.
    func dispatchExactly(
        threadCount: Int,
        threadsPerThreadgroup width: Int,
        device: MTLDevice
    ) {
        let threadgroup = MTLSize(width: width, height: 1, depth: 1)
        if device.supportsFamily(.apple4) {
            self.dispatchThreads(
                MTLSize(width: threadCount, height: 1, depth: 1),
                threadsPerThreadgroup: threadgroup)
        } else {
            let groups = (threadCount + width - 1) / width
            self.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: threadgroup)
        }
    }
}
