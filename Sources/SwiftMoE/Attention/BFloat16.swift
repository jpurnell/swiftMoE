/// Converts a BFloat16 value (stored as UInt16) to Float32.
///
/// BFloat16 is the upper 16 bits of an IEEE 754 float32. Conversion is a
/// simple left-shift into the high bits of a 32-bit word.
///
/// This matches the C implementation in `infer.m`:
/// ```c
/// static float bf16_to_f32(uint16_t bf16) {
///     uint32_t bits = (uint32_t)bf16 << 16;
///     float f;
///     memcpy(&f, &bits, 4);
///     return f;
/// }
/// ```
@inline(__always)
public func bf16ToFloat(_ bf16: UInt16) -> Float {
    Float(bitPattern: UInt32(bf16) << 16)
}

/// Converts a Float32 value to BFloat16 (truncation, no rounding).
///
/// Drops the lower 16 bits of mantissa. This matches the C implementation:
/// ```c
/// static uint16_t f32_to_bf16(float f) {
///     uint32_t bits;
///     memcpy(&bits, &f, 4);
///     return (uint16_t)(bits >> 16);
/// }
/// ```
@inline(__always)
public func floatToBf16(_ f: Float) -> UInt16 {
    UInt16(f.bitPattern >> 16)
}
