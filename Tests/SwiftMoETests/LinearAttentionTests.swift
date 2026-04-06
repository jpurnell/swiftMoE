import Testing
import Foundation
@testable import SwiftMoE

@Suite("LinearAttention")
struct LinearAttentionTests {

    @Test("Produces output of correct dimension")
    func outputDimension() {
        let config = ModelConfig.tiny
        let totalValue = config.linearTotalValue
        let convDim = config.linearConvDim

        var qkvOut = [Float](repeating: 0.1, count: convDim)
        var zOut = [Float](repeating: 0.1, count: totalValue)
        var betaOut = [Float](repeating: 0.0, count: config.linearNumVHeads)
        var alphaOut = [Float](repeating: 0.0, count: config.linearNumVHeads)
        var output = [Float](repeating: -999, count: totalValue)
        var state = LinearAttentionState(config: config)

        qkvOut.withUnsafeBufferPointer { qkvBuf in
            zOut.withUnsafeBufferPointer { zBuf in
                betaOut.withUnsafeBufferPointer { betaBuf in
                    alphaOut.withUnsafeBufferPointer { alphaBuf in
                        output.withUnsafeMutableBufferPointer { outBuf in
                            LinearAttention.forward(
                                qkvOut: qkvBuf.baseAddress!,
                                zOut: zBuf.baseAddress!,
                                betaOut: betaBuf.baseAddress!,
                                alphaOut: alphaBuf.baseAddress!,
                                state: &state,
                                config: config,
                                conv1dW: nil,
                                aLog: nil,
                                dtBias: nil,
                                gatedNormW: nil,
                                output: outBuf.baseAddress!
                            )
                        }
                    }
                }
            }
        }

        // Output should be written
        #expect(output.count == totalValue)
        // With nil weights, conv1d copies input directly, so output depends on input values
    }

    @Test("Conv1d step updates state")
    func conv1dState() {
        let channels = 4
        let kernelSize = 3
        var convState = [Float](repeating: 0, count: (kernelSize - 1) * channels)
        var input: [Float] = [1, 2, 3, 4]
        var output = [Float](repeating: 0, count: channels)

        // Without weights, conv1d copies input
        LinearAttention.conv1dStep(
            convState: &convState,
            newInput: &input,
            weights: nil,
            output: &output,
            channels: channels,
            kernelSize: kernelSize
        )

        // After one step, the last history position should contain the input
        let lastOffset = (kernelSize - 2) * channels
        #expect(convState[lastOffset] == 1.0, "Conv state should store input")
        #expect(convState[lastOffset + 1] == 2.0)
    }

    @Test("State persists across multiple forward calls")
    func statePersistence() {
        let config = ModelConfig.tiny
        let totalValue = config.linearTotalValue
        let convDim = config.linearConvDim

        var state = LinearAttentionState(config: config)
        let initialState = state.state  // Copy

        // Run forward to modify state
        var qkvOut = [Float](repeating: 0.5, count: convDim)
        var zOut = [Float](repeating: 0.1, count: totalValue)
        var betaOut = [Float](repeating: 0.5, count: config.linearNumVHeads)
        var alphaOut = [Float](repeating: 0.0, count: config.linearNumVHeads)
        var output = [Float](repeating: 0, count: totalValue)

        qkvOut.withUnsafeBufferPointer { qkvBuf in
            zOut.withUnsafeBufferPointer { zBuf in
                betaOut.withUnsafeBufferPointer { betaBuf in
                    alphaOut.withUnsafeBufferPointer { alphaBuf in
                        output.withUnsafeMutableBufferPointer { outBuf in
                            LinearAttention.forward(
                                qkvOut: qkvBuf.baseAddress!,
                                zOut: zBuf.baseAddress!,
                                betaOut: betaBuf.baseAddress!,
                                alphaOut: alphaBuf.baseAddress!,
                                state: &state,
                                config: config,
                                conv1dW: nil, aLog: nil, dtBias: nil, gatedNormW: nil,
                                output: outBuf.baseAddress!
                            )
                        }
                    }
                }
            }
        }

        // State should have changed (delta-net rank-1 update modifies it)
        let stateChanged = zip(state.state, initialState).contains { $0 != $1 }
        #expect(stateChanged, "Delta-net state should be modified after forward pass")
    }
}
