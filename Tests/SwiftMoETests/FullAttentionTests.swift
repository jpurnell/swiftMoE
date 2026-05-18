import Testing
import Foundation
@testable import SwiftMoE

@Suite("FullAttention")
struct FullAttentionTests {

    @Test("Produces output of correct dimension")
    func outputDimension() {
        let config = ModelConfig.tiny
        let qDim = config.numAttentionHeads * config.headDim
        let kvDim = config.numKVHeads * config.headDim

        // Q proj output is [numHeads * headDim * 2] (query + gate interleaved)
        var qProjOut = [Float](repeating: 0.1, count: qDim * 2)
        var kOut = [Float](repeating: 0.1, count: kvDim)
        var vOut = [Float](repeating: 0.1, count: kvDim)
        var output = [Float](repeating: -999, count: qDim)
        var kvCache = KVCache(kvDim: kvDim, maxLength: 16)

        output.withUnsafeMutableBufferPointer { outBuf in
            guard let outBase = outBuf.baseAddress else { return }
            FullAttention.forward(
                qProjOut: &qProjOut,
                kOut: &kOut,
                vOut: &vOut,
                kvCache: &kvCache,
                position: 0,
                config: config,
                qNormW: nil,
                kNormW: nil,
                output: outBase
            )
        }

        // Output should be written (finite attention result) and KV cache should have 1 entry
        #expect(kvCache.length == 1)
        #expect(output[0].isFinite, "Output should be a finite value after attention forward")
        #expect(output[0] >= -100 && output[0] <= 100,
                "Output should be in a reasonable range, got \(output[0])")
    }

    @Test("Multiple tokens grow KV cache")
    func kvCacheGrows() {
        let config = ModelConfig.tiny
        let qDim = config.numAttentionHeads * config.headDim
        let kvDim = config.numKVHeads * config.headDim

        var qProjOut = [Float](repeating: 0.1, count: qDim * 2)
        var kOut = [Float](repeating: 0.1, count: kvDim)
        var vOut = [Float](repeating: 0.1, count: kvDim)
        var output = [Float](repeating: 0, count: qDim)
        var kvCache = KVCache(kvDim: kvDim, maxLength: 16)

        // Process 3 tokens
        for pos in 0..<3 {
            output.withUnsafeMutableBufferPointer { outBuf in
                guard let outBase = outBuf.baseAddress else { return }
                FullAttention.forward(
                    qProjOut: &qProjOut, kOut: &kOut, vOut: &vOut,
                    kvCache: &kvCache, position: pos, config: config,
                    qNormW: nil, kNormW: nil, output: outBase
                )
            }
        }

        #expect(kvCache.length == 3)
    }
}
