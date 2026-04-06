import Testing
import Metal
@testable import SwiftMoE

@Suite("DeferredExpertState")
struct DeferredExpertStateTests {

    @Test("Starts inactive")
    func initialState() {
        let state = DeferredExpertState()
        #expect(state.isActive == false)
        #expect(state.isGPUCombined == false)
    }

    @Test("Can be activated and deactivated")
    func activation() {
        var state = DeferredExpertState()

        state.activate(
            expertWeights: [0.5, 0.3, 0.1, 0.1],
            valid: [true, true, true, false],
            sharedGateScore: 0.7,
            layerIndex: 5,
            gpuCombined: true
        )

        #expect(state.isActive == true)
        #expect(state.isGPUCombined == true)
        #expect(state.actualK == 4)
        #expect(state.layerIndex == 5)
        #expect(abs(state.sharedGateScore - 0.7) < 1e-6)

        state.deactivate()
        #expect(state.isActive == false)
        #expect(state.isGPUCombined == false)
    }
}
