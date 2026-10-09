import Testing
@testable import Agents

// Ported from tests/test_output_guardrail_cancellation.py.
@Suite struct OutputGuardrailCancellationTests {
    @Test func testRunOutputGuardrailsAwaitsCancelledTasks() async throws {
        let slowStarted = AsyncEvent()
        let cancelledObserved = AsyncEvent()

        let slowThenObserveCancel = OutputGuardrail { _, _, _ in
            slowStarted.set()
            do {
                try await waitForever()
            } catch is CancellationError {
                cancelledObserved.set()
                throw CancellationError()
            }
            return GuardrailFunctionOutput(outputInfo: nil, tripwireTriggered: false)
        }
        let fastTripwire = OutputGuardrail { _, _, _ in
            // Wait until the slow guardrail is parked so the cancellation reaches its handler.
            try await slowStarted.wait()
            return GuardrailFunctionOutput(outputInfo: nil, tripwireTriggered: true)
        }

        await expectErrorAsync(OutputGuardrailTripwireTriggered.self) {
            _ = try await runOutputGuardrails([slowThenObserveCancel, fastTripwire], Agent(name: "test"), "agent output", RunContextWrapper())
        }

        // The slow guardrail must have observed cancellation before `runOutputGuardrails` returned.
        #expect(cancelledObserved.isSet)
    }
}
