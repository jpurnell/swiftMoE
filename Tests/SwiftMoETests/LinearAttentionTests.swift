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

        let qkvOut = [Float](repeating: 0.1, count: convDim)
        let zOut = [Float](repeating: 0.1, count: totalValue)
        let betaOut = [Float](repeating: 0.0, count: config.linearNumVHeads)
        let alphaOut = [Float](repeating: 0.0, count: config.linearNumVHeads)
        var output = [Float](repeating: -999, count: totalValue)
        var state = LinearAttentionState(config: config)

        qkvOut.withUnsafeBufferPointer { qkvBuf in
            guard let qkvBase = qkvBuf.baseAddress else { return }
            zOut.withUnsafeBufferPointer { zBuf in
                guard let zBase = zBuf.baseAddress else { return }
                betaOut.withUnsafeBufferPointer { betaBuf in
                    guard let betaBase = betaBuf.baseAddress else { return }
                    alphaOut.withUnsafeBufferPointer { alphaBuf in
                        guard let alphaBase = alphaBuf.baseAddress else { return }
                        output.withUnsafeMutableBufferPointer { outBuf in
                            guard let outBase = outBuf.baseAddress else { return }
                            LinearAttention.forward(
                                qkvOut: qkvBase,
                                zOut: zBase,
                                betaOut: betaBase,
                                alphaOut: alphaBase,
                                state: &state,
                                config: config,
                                conv1dW: nil,
                                aLog: nil,
                                dtBias: nil,
                                gatedNormW: nil,
                                output: outBase
                            )
                        }
                    }
                }
            }
        }

        #expect(output.count == totalValue)
    }

    @Test("Conv1d step updates state")
    func conv1dState() {
        let channels = 4
        let kernelSize = 3
        var convState = [Float](repeating: 0, count: (kernelSize - 1) * channels)
        var input: [Float] = [1, 2, 3, 4]
        var output = [Float](repeating: 0, count: channels)

        LinearAttention.conv1dStep(
            convState: &convState,
            newInput: &input,
            weights: nil,
            output: &output,
            channels: channels,
            kernelSize: kernelSize
        )

        let lastOffset = (kernelSize - 2) * channels
        #expect(abs(convState[lastOffset] - 1.0) < 1e-6, "Conv state should store input")
        #expect(abs(convState[lastOffset + 1] - 2.0) < 1e-6)
    }

    @Test("State persists across multiple forward calls")
    func statePersistence() {
        let config = ModelConfig.tiny
        let totalValue = config.linearTotalValue
        let convDim = config.linearConvDim

        var state = LinearAttentionState(config: config)
        let initialState = state.state

        let qkvOut = [Float](repeating: 0.5, count: convDim)
        let zOut = [Float](repeating: 0.1, count: totalValue)
        let betaOut = [Float](repeating: 0.5, count: config.linearNumVHeads)
        let alphaOut = [Float](repeating: 0.0, count: config.linearNumVHeads)
        var output = [Float](repeating: 0, count: totalValue)

        qkvOut.withUnsafeBufferPointer { qkvBuf in
            guard let qkvBase = qkvBuf.baseAddress else { return }
            zOut.withUnsafeBufferPointer { zBuf in
                guard let zBase = zBuf.baseAddress else { return }
                betaOut.withUnsafeBufferPointer { betaBuf in
                    guard let betaBase = betaBuf.baseAddress else { return }
                    alphaOut.withUnsafeBufferPointer { alphaBuf in
                        guard let alphaBase = alphaBuf.baseAddress else { return }
                        output.withUnsafeMutableBufferPointer { outBuf in
                            guard let outBase = outBuf.baseAddress else { return }
                            LinearAttention.forward(
                                qkvOut: qkvBase,
                                zOut: zBase,
                                betaOut: betaBase,
                                alphaOut: alphaBase,
                                state: &state,
                                config: config,
                                conv1dW: nil, aLog: nil, dtBias: nil, gatedNormW: nil,
                                output: outBase
                            )
                        }
                    }
                }
            }
        }

        // State should have changed (delta-net rank-1 update modifies it)
        let stateChanged = zip(state.state, initialState).contains { abs($0 - $1) > 1e-6 }
        #expect(stateChanged, "Delta-net state should be modified after forward pass")
    }
}
