import Testing
@testable import SwiftMoE

@Suite("BFloat16")
struct BFloat16Tests {

    @Test("Known BF16 values convert correctly to Float32")
    func knownValues() {
        // 1.0 in BF16 = 0x3F80
        #expect(bf16ToFloat(0x3F80) == 1.0)
        // -1.0 in BF16 = 0xBF80
        #expect(bf16ToFloat(0xBF80) == -1.0)
        // 0.0 in BF16 = 0x0000
        #expect(bf16ToFloat(0x0000) == 0.0)
        // 2.0 in BF16 = 0x4000
        #expect(bf16ToFloat(0x4000) == 2.0)
    }

    @Test("Float32 to BF16 truncation preserves value within precision")
    func roundTrip() {
        let values: [Float] = [1.0, -1.0, 0.5, 3.14159, 100.0, 0.001]
        for v in values {
            let bf16 = floatToBf16(v)
            let recovered = bf16ToFloat(bf16)
            // BF16 has 7 bits of mantissa → ~2 decimal digits of precision
            let relativeError = abs(recovered - v) / max(abs(v), 1e-10)
            #expect(relativeError < 0.01,
                    "Round-trip error too large for \(v): got \(recovered)")
        }
    }

    @Test("BF16 conversion matches the C implementation bit pattern")
    func bitPattern() {
        // The C code: bf16_to_f32 does (uint32_t)bf16 << 16, then memcpy to float
        // floatToBf16 does memcpy to uint32, then >> 16
        let val: Float = 42.0
        let bf16 = floatToBf16(val)
        let bits = val.bitPattern >> 16
        #expect(bf16 == UInt16(bits))
    }
}
