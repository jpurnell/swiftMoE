import Testing
@testable import SwiftMoE

@Suite("KVCache")
struct KVCacheTests {

    @Test("Stores and retrieves K/V at position 0")
    func appendAndRetrieve() throws {
        let kvDim = 4
        var cache = KVCache(kvDim: kvDim, maxLength: 16)

        let k: [Float] = [1.0, 2.0, 3.0, 4.0]
        let v: [Float] = [5.0, 6.0, 7.0, 8.0]
        try cache.append(k: k, v: v)

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
    func multipleAppends() throws {
        let kvDim = 2
        var cache = KVCache(kvDim: kvDim, maxLength: 16)

        try cache.append(k: [1.0, 2.0], v: [3.0, 4.0])
        try cache.append(k: [5.0, 6.0], v: [7.0, 8.0])

        #expect(cache.length == 2)
        cache.withKCache { kPtr in
            // Position 1, element 0
            #expect(abs(kPtr[kvDim + 0] - 5.0) < 1e-6)
        }
    }

    @Test("Reset clears to zero length")
    func reset() throws {
        let kvDim = 2
        var cache = KVCache(kvDim: kvDim, maxLength: 8)

        try cache.append(k: [1.0, 2.0], v: [3.0, 4.0])
        #expect(cache.length == 1)

        cache.reset()
        #expect(cache.length == 0)
    }

    @Test("Capacity is the length the cache was created with")
    func capacity() {
        #expect(KVCache(kvDim: 2, maxLength: 8).capacity == 8)
    }

    @Test("An append past capacity throws and records nothing")
    func appendPastCapacity() throws {
        var cache = KVCache(kvDim: 2, maxLength: 2)
        try cache.append(k: [1.0, 2.0], v: [3.0, 4.0])
        try cache.append(k: [5.0, 6.0], v: [7.0, 8.0])

        #expect(throws: FlashMoEError.sequenceCapacityExceeded(capacity: 2, required: 3)) {
            try cache.append(k: [9.0, 9.0], v: [9.0, 9.0])
        }
        #expect(cache.length == 2)
        // The two recorded positions are still exactly what was stored: copies, not computations.
        let keys = cache.withKCache { Array(UnsafeBufferPointer(start: $0, count: 4)) }
        let stored: [Float] = [1.0, 2.0, 5.0, 6.0]
        #expect(keys.count == stored.count
            && zip(keys, stored).allSatisfy { $0.bitPattern == $1.bitPattern })
    }

    @Test("The pointer append past capacity throws too")
    func pointerAppendPastCapacity() throws {
        var cache = KVCache(kvDim: 2, maxLength: 1)
        let key: [Float] = [1.0, 2.0]
        let value: [Float] = [3.0, 4.0]
        try key.withUnsafeBufferPointer { keyBuffer in
            try value.withUnsafeBufferPointer { valueBuffer in
                let keyBase = try #require(keyBuffer.baseAddress)
                let valueBase = try #require(valueBuffer.baseAddress)
                try cache.append(kPtr: keyBase, vPtr: valueBase)
                #expect(throws: FlashMoEError.sequenceCapacityExceeded(capacity: 1, required: 2)) {
                    try cache.append(kPtr: keyBase, vPtr: valueBase)
                }
            }
        }
        #expect(cache.length == 1)
    }

    @Test("After a reset the whole capacity is available again")
    func resetRestoresCapacity() throws {
        var cache = KVCache(kvDim: 2, maxLength: 1)
        try cache.append(k: [1.0, 2.0], v: [3.0, 4.0])
        cache.reset()
        try cache.append(k: [5.0, 6.0], v: [7.0, 8.0])
        #expect(cache.length == 1)
    }
}
