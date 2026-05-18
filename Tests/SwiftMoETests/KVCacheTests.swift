import Testing
@testable import SwiftMoE

@Suite("KVCache")
struct KVCacheTests {

    @Test("Stores and retrieves K/V at position 0")
    func appendAndRetrieve() {
        let kvDim = 4
        var cache = KVCache(kvDim: kvDim, maxLength: 16)

        let k: [Float] = [1.0, 2.0, 3.0, 4.0]
        let v: [Float] = [5.0, 6.0, 7.0, 8.0]
        cache.append(k: k, v: v)

        #expect(cache.length == 1)
        cache.withKCache { kPtr in
            #expect(abs(kPtr[0] - 1.0) < 1e-6)
            #expect(abs(kPtr[3] - 4.0) < 1e-6)
        }
        cache.withVCache { vPtr in
            #expect(abs(vPtr[0] - 5.0) < 1e-6)
        }
    }

    @Test("Multiple appends grow length")
    func multipleAppends() {
        let kvDim = 2
        var cache = KVCache(kvDim: kvDim, maxLength: 16)

        cache.append(k: [1.0, 2.0], v: [3.0, 4.0])
        cache.append(k: [5.0, 6.0], v: [7.0, 8.0])

        #expect(cache.length == 2)
        cache.withKCache { kPtr in
            // Position 1, element 0
            #expect(abs(kPtr[kvDim + 0] - 5.0) < 1e-6)
        }
    }

    @Test("Reset clears to zero length")
    func reset() {
        let kvDim = 2
        var cache = KVCache(kvDim: kvDim, maxLength: 8)

        cache.append(k: [1.0, 2.0], v: [3.0, 4.0])
        #expect(cache.length == 1)

        cache.reset()
        #expect(cache.length == 0)
    }
}
