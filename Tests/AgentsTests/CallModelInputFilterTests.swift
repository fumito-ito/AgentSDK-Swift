import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_call_model_input_filter.py.
//
// Swift: `test_call_model_input_filter_invalid_return_type_raises` is skipped because the
// filter's return type is checked by the compiler. Streamed and non-streamed variants share one
// parameterized test.
@Suite struct CallModelInputFilterTests {
    private static let sensitiveErrorMessage = "sensitive-filter-error"

    private func run(_ agent: Agent, filter: @escaping CallModelInputFilter, streamed: Bool) async throws {
        let runConfig = RunConfig(callModelInputFilter: filter)
        if streamed {
            let result = try Runner.runStreamed(agent, input: "start", runConfig: runConfig)
            for try await _ in result.streamEvents() {}
        } else {
            _ = try await Runner.run(agent, input: "start", runConfig: runConfig)
        }
    }

    private func makeAgent() throws -> (Agent, ScriptedModel) {
        let model = try ScriptedModel(steps: [.output([getTextMessage("ok")])])
        return (Agent(name: "test", model: .instance(model)), model)
    }

    private func sentInput(_ model: ScriptedModel) -> [TResponseInputItem] {
        guard case .items(let items)? = model.calls.last?.input else { return [] }
        return items
    }

    private func sentItemTypes(_ items: [TResponseInputItem]) -> [String] {
        items.compactMap { $0.string("type") }
    }

    @Test(arguments: [false, true])
    func testCallModelInputFilterAddsInputAndInstructions(streamed: Bool) async throws {
        // Ports test_call_model_input_filter_sync_non_streamed and _async_streamed.
        let (agent, model) = try makeAgent()
        let label = streamed ? "async" : "sync"
        let filter: CallModelInputFilter = { data in
            if streamed { await Task.yield() }
            return ModelInputData(input: data.modelData.input + [getTextInputItem("added-\(label)")], instructions: "filtered-\(label)")
        }

        try await run(agent, filter: filter, streamed: streamed)

        #expect(model.calls.last?.systemInstructions == "filtered-\(label)")
        let input = sentInput(model)
        #expect(input.count == 2)
        #expect(input.last?.string("content") == "added-\(label)")
    }

    @Test(.tracing, arguments: [(false, "Error details are redacted."), (true, sensitiveErrorMessage)])
    func testCallModelInputFilterErrorRespectsSensitiveDataSetting(traceIncludeSensitiveData: Bool, expectedError: String) async throws {
        let filter: CallModelInputFilter = { _ in throw ValueError(Self.sensitiveErrorMessage) }

        await expectErrorAsync(ValueError.self, containing: Self.sensitiveErrorMessage) {
            _ = try await Runner.run(
                Agent(name: "test", model: .instance(try ScriptedModel(emitTraces: false))),
                input: "start",
                runConfig: RunConfig(traceIncludeSensitiveData: traceIncludeSensitiveData, callModelInputFilter: filter)
            )
        }

        #expect(fetchSpanErrors("turn") == [["message": "Error in call_model_input_filter", "data": ["error": .string(expectedError)]]])
    }

    @Test(arguments: [false, true])
    func testCallModelInputFilterPrefersLatestDuplicateOutputs(streamed: Bool) async throws {
        let (agent, model) = try makeAgent()
        let callId = streamed ? "dup-call-stream" : "dup-call"
        let duplicateOld: TResponseInputItem = ["type": "function_call_output", "call_id": .string(callId), "output": "old-value"]
        let duplicateNew: TResponseInputItem = ["type": "function_call_output", "call_id": .string(callId), "output": "new-value"]
        let filter: CallModelInputFilter = { data in
            ModelInputData(input: [duplicateOld, duplicateNew] + data.modelData.input, instructions: data.modelData.instructions)
        }

        try await run(agent, filter: filter, streamed: streamed)

        let outputs = sentInput(model).filter { $0.string("type") == "function_call_output" && $0.string("call_id") == callId }
        #expect(outputs.count == 1)
        #expect(outputs.first?.string("output") == "new-value")
    }

    private static let duplicateToolCallInput: [TResponseInputItem] = [
        ["type": "function_call", "call_id": "ordered-call", "name": "tool_ordered", "arguments": "{}"],
        ["type": "function_call_output", "call_id": "ordered-call", "output": "result"],
        ["type": "function_call", "call_id": "ordered-call", "name": "tool_ordered", "arguments": "{}"],
    ]

    private static let duplicateToolOutputInput: [TResponseInputItem] = [
        ["type": "function_call_output", "call_id": "ordered-output", "output": "old"],
        ["type": "function_call", "call_id": "ordered-output", "name": "tool_ordered", "arguments": "{}"],
        ["type": "function_call_output", "call_id": "ordered-output", "output": "new"],
    ]

    private static let duplicateReasoningInput: [TResponseInputItem] = [
        ["type": "reasoning", "id": "ordered-reasoning", "summary": [["type": "summary_text", "text": "old"]]],
        ["type": "function_call", "call_id": "reasoning-call", "name": "tool_ordered", "arguments": "{}"],
        ["type": "reasoning", "id": "ordered-reasoning", "summary": [["type": "summary_text", "text": "new"]]],
    ]

    @Test(arguments: [false, true])
    func testCallModelInputFilterKeepsDuplicateItemOrder(streamed: Bool) async throws {
        let (agent, model) = try makeAgent()
        let filter: CallModelInputFilter = { data in
            ModelInputData(input: data.modelData.input + Self.duplicateToolCallInput, instructions: data.modelData.instructions)
        }

        try await run(agent, filter: filter, streamed: streamed)

        // Collapsing the repeated call must not move it behind its output.
        #expect(sentItemTypes(sentInput(model)) == ["function_call", "function_call_output"])
    }

    @Test(arguments: [false, true])
    func testCallModelInputFilterKeepsDuplicateOutputOrder(streamed: Bool) async throws {
        let (agent, model) = try makeAgent()
        let filter: CallModelInputFilter = { data in
            ModelInputData(input: data.modelData.input + Self.duplicateToolOutputInput, instructions: data.modelData.instructions)
        }

        try await run(agent, filter: filter, streamed: streamed)

        let input = sentInput(model)
        #expect(sentItemTypes(input) == ["function_call", "function_call_output"])
        #expect(input.last?.string("output") == "new")
    }

    @Test func testCallModelInputFilterKeepsReasoningBeforeRequiredFollower() async throws {
        let (agent, model) = try makeAgent()
        let filter: CallModelInputFilter = { data in
            ModelInputData(input: data.modelData.input + Self.duplicateReasoningInput, instructions: data.modelData.instructions)
        }

        try await run(agent, filter: filter, streamed: false)

        let input = sentInput(model)
        #expect(sentItemTypes(input) == ["reasoning", "function_call"])
        let reasoningItems = input.filter { $0.string("type") == "reasoning" }
        #expect(reasoningItems.count == 1)
        #expect(reasoningItems.first?["summary"] == [["type": "summary_text", "text": "new"]])
    }
}
