import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_call_model_input_filter_unit.py.
//
// Swift: `test_call_model_input_filter_invalid_return_type_raises_unit` is skipped because the
// filter's return type is checked by the compiler.
@Suite struct CallModelInputFilterUnitTests {
    private func okMessage() -> TResponseOutputItem {
        [
            "id": "1", "type": "message", "role": "assistant", "status": "completed",
            "content": [["text": "ok", "type": "output_text", "annotations": [], "logprobs": []]],
        ]
    }

    @Test func testCallModelInputFilterSyncNonStreamedUnit() async throws {
        let model = try ScriptedModel(steps: [.output([okMessage()])])
        let agent = Agent(name: "test", model: .instance(model))
        let filter: CallModelInputFilter = { data in
            ModelInputData(input: data.modelData.input + [["content": "added-sync", "role": "user"]], instructions: "filtered-sync")
        }

        _ = try await Runner.run(agent, input: "start", runConfig: RunConfig(callModelInputFilter: filter))

        let call = try #require(model.calls.last)
        #expect(call.systemInstructions == "filtered-sync")
        guard case .items(let input) = call.input else {
            Issue.record("Expected list input")
            return
        }
        #expect(input.count == 2)
        #expect(input.last?.string("content") == "added-sync")
    }

    @Test func testCallModelInputFilterAsyncStreamedUnit() async throws {
        let model = try ScriptedModel(steps: [.output([okMessage()])])
        let agent = Agent(name: "test", model: .instance(model))
        let filter: CallModelInputFilter = { data in
            await Task.yield()
            return ModelInputData(input: data.modelData.input + [["content": "added-async", "role": "user"]], instructions: "filtered-async")
        }

        let result = try Runner.runStreamed(agent, input: "start", runConfig: RunConfig(callModelInputFilter: filter))
        for try await _ in result.streamEvents() {}

        let call = try #require(model.calls.last)
        #expect(call.systemInstructions == "filtered-async")
        guard case .items(let input) = call.input else {
            Issue.record("Expected list input")
            return
        }
        #expect(input.count == 2)
        #expect(input.last?.string("content") == "added-async")
    }
}
