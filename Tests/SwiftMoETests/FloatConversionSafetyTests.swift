import Testing
import Foundation
@testable import SwiftMoE

/// Tests that floating-point values which cannot be represented as an `Int`
/// (NaN, infinity) are refused or formatted rather than trapping.
@Suite("Float conversion safety")
struct FloatConversionSafetyTests {

    @Test("Throughput is bytes over seconds for a positive duration")
    func throughputForPositiveDuration() {
        let stats = IOReadStats(totalMs: 500, readCount: 4, totalBytes: 1000)
        #expect(stats.throughputBytesPerSec == 2000)
    }

    @Test("Throughput is nil when the duration cannot divide",
          arguments: [0.0, -1.0, Double.nan, Double.infinity])
    func throughputUndefined(totalMs: Double) {
        let stats = IOReadStats(totalMs: totalMs, readCount: 4, totalBytes: 1000)
        #expect(stats.throughputBytesPerSec == nil)
    }

    @Test("Presets keep their rotary dimensions")
    func presetRotaryDims() {
        #expect(ModelConfig.qwen397B.rotaryDim == 64)
        #expect(ModelConfig.tiny.rotaryDim == 8)
    }

    @Test("Rotary dimension truncates toward zero")
    func rotaryDimTruncates() {
        #expect(ModelConfig.rotaryDim(headDim: 10, partialRotary: 0.25) == 2)
    }

    @Test("Rotary dimension is 0 when the fraction is not within 0...1",
          arguments: [Float.nan, Float.infinity, -Float.infinity, -0.25, 1.5])
    func rotaryDimRefusesBadFraction(partialRotary: Float) {
        #expect(ModelConfig.rotaryDim(headDim: 256, partialRotary: partialRotary) == 0)
    }

    @Test("Timing summary pads the integer part to six columns")
    func timingSummaryAlignment() {
        var timing = LayerTiming()
        timing.layerCount = 2
        timing.cmd1Wait = 2.5
        timing.expertIO = 246.9
        let summary = timing.formatSummary()
        #expect(summary.contains("cmd1_wait:           1.25\n"))
        #expect(summary.contains("expert_io:         123.45\n"))
    }

    @Test("Timing summary formats non-finite accumulators instead of trapping",
          arguments: [Double.nan, Double.infinity, 1e300])
    func timingSummaryNonFinite(value: Double) {
        var timing = LayerTiming()
        timing.layerCount = 1
        timing.totalLayer = value
        #expect(timing.formatSummary().contains("total_layer:"))
    }
}
