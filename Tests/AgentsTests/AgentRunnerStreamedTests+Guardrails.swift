import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_runner_streamed.py (guardrails, events, and approval resumes).
//
// Swift: `test_streamed_finalizer_failure_follows_both_data_policies` monkeypatches a private
// finalizer and checks Python log-record internals, so it is skipped.
// `test_stream_input_persistence_saves_only_new_turn_input` wraps a private persistence function;
// its observable behavior is asserted through the session contents instead.
extension AgentRunnerStreamedTests {
    private func drainStream(_ result: RunResultStreaming) async throws {
        for try await _ in result.streamEvents() {}
    }

    private func tripwire() -> InputGuardrail {
        InputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: true) }
    }

    // MARK: - Input guardrails

    @Test func testInputGuardrailTripwireTriggeredCausesExceptionStreamed() async throws {
        let agent = Agent(name: "test", model: .instance(try ScriptedModel(steps: [.output([])])), inputGuardrails: [tripwire()])

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            try await drainStream(try Runner.runStreamed(agent, input: "user_message"))
        }
    }

    @Test(arguments: [true, false])
    func testInputGuardrailStreamedPersistsOnlyUserInputOnTrip(runInParallel: Bool) async throws {
        let session = SimpleListSession()
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("should_not_be_saved")]))
        let guardrail = InputGuardrail(runInParallel: runInParallel) { _, _, _ in
            if runInParallel { try await sleep(seconds: 0.01) }
            return GuardrailFunctionOutput(tripwireTriggered: true)
        }
        let agent = Agent(name: "test", model: .instance(model), inputGuardrails: [guardrail])

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            try await drainStream(try Runner.runStreamed(agent, input: "user_message", session: session))
        }

        let items = try await session.getItems(limit: nil)
        #expect(items.count == 1)
        #expect(items.first?.string("role") == "user")
    }

    @Test func testStreamInputPersistenceSavesOnlyNewTurnInput() async throws {
        let session = SimpleListSession()
        let model = try ScriptedModel()
        try model.extend([.output([getTextMessage("first")]), .output([getTextMessage("second")])])
        let agent = Agent(name: "test", model: .instance(model))
        let runConfig = RunConfig(sessionInputCallback: { existing, new in existing + new })

        try await drainStream(try Runner.runStreamed(agent, input: .items([getTextInputItem("hello")]), runConfig: runConfig, session: session))
        try await drainStream(try Runner.runStreamed(agent, input: .items([getTextInputItem("next")]), runConfig: runConfig, session: session))

        let userItems = try await session.getItems(limit: nil).filter { $0.string("role") == "user" }
        #expect(userItems.map { $0.string("content") } == ["hello", "next"], "each turn should persist only its own input once")
    }

    @Test func testSlowInputGuardrailStillRaisesExceptionStreamed() async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("ok")]))
        let guardrail = InputGuardrail { _, _, _ in
            try await sleep(seconds: 0.05)
            return GuardrailFunctionOutput(tripwireTriggered: true)
        }
        let agent = Agent(name: "test", model: .instance(model), inputGuardrails: [guardrail])

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            try await drainStream(try Runner.runStreamed(agent, input: "user_message"))
        }
    }

    @Test func testRunInputGuardrailTripwireTriggeredCausesExceptionStreamed() async throws {
        let agent = Agent(name: "test", model: .instance(try ScriptedModel(steps: [.output([])])))

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            try await drainStream(try Runner.runStreamed(agent, input: "user_message", runConfig: RunConfig(inputGuardrails: [tripwire()])))
        }
    }

    // MARK: - Output guardrails

    private func outputTripwire() -> OutputGuardrail {
        OutputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: true) }
    }

    @Test func testOutputGuardrailTripwireTriggeredCausesExceptionStreamed() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("first_test")])])
        let agent = Agent(name: "test", model: .instance(model), outputGuardrails: [outputTripwire()])

        await expectErrorAsync(OutputGuardrailTripwireTriggered.self) {
            try await drainStream(try Runner.runStreamed(agent, input: "user_message"))
        }
    }

    @Test func testOutputGuardrailTripwireRaisesFromRunLoopTaskBeforeStreamConsumption() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("first_test")])])
        let agent = Agent(name: "test", model: .instance(model), outputGuardrails: [outputTripwire()])

        let result = try Runner.runStreamed(agent, input: "user_message")
        let task = try #require(result.runLoopTask)

        await expectErrorAsync(OutputGuardrailTripwireTriggered.self) { try await task.value }
        #expect(result.finalOutput == nil)
        #expect(result.isComplete)
    }

    @Test func testOutputGuardrailExceptionRaisesFromRunLoopTaskBeforeStreamConsumption() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("first_test")])])
        let agent = Agent(name: "test", model: .instance(model), outputGuardrails: [OutputGuardrail { _, _, _ in throw TestError("guardrail failed") }])

        let result = try Runner.runStreamed(agent, input: "user_message")
        let task = try #require(result.runLoopTask)

        await expectErrorAsync(TestError.self, containing: "guardrail failed") { try await task.value }
        #expect(result.finalOutput == nil)
        #expect(result.isComplete)
    }

    @Test func testRunOutputGuardrailTripwireTriggeredCausesExceptionStreamed() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("first_test")])])
        let agent = Agent(name: "test", model: .instance(model))

        await expectErrorAsync(OutputGuardrailTripwireTriggered.self) {
            try await drainStream(try Runner.runStreamed(agent, input: "user_message", runConfig: RunConfig(outputGuardrails: [outputTripwire()])))
        }
    }

    // MARK: - Events

    private struct EventFoo: Codable, Sendable, Equatable {
        var bar: String
    }

    @Test func testStreamingEvents() async throws {
        let model = try ScriptedModel()
        let agent1 = Agent(
            name: "test",
            tools: [getFunctionTool(name: "bar", returnValue: "bar_result")],
            model: .instance(model),
            outputType: try AgentOutputSchema(EventFoo.self)
        )
        let agent2 = Agent(name: "test", tools: [getFunctionTool(name: "foo", returnValue: "foo_result")], handoffs: [.agent(agent1)], model: .instance(model))
        try model.extend([
            .output([getFunctionToolCall("foo", arguments: #"{"bar": "baz"}"#, callId: "call_foo")]),
            .output([getTextMessage("a_message"), getHandoffToolCall(agent1)]),
            .output([getFunctionToolCall("bar", arguments: #"{"bar": "baz"}"#, callId: "call_bar")]),
            .output([getFinalOutputMessage(#"{"bar": "baz"}"#)]),
        ])

        var eventCounts: [String: Int] = [:]
        var items: [RunItem] = []
        var agentUpdates: [AgentUpdatedStreamEvent] = []
        let result = try Runner.runStreamed(
            agent2,
            input: .items([getTextInputItem("user_message"), getTextInputItem("another_message")]),
            runConfig: RunConfig(nestHandoffHistory: true)
        )
        for try await event in result.streamEvents() {
            eventCounts[event.type, default: 0] += 1
            switch event {
            case .runItem(let itemEvent): items.append(itemEvent.item)
            case .agentUpdated(let update): agentUpdates.append(update)
            case .rawResponse: break
            }
        }

        #expect(result.finalOutput as? EventFoo == EventFoo(bar: "baz"))
        #expect(result.rawResponses.count == 4)
        #expect(try result.toInputList().count == 10)
        #expect(try result.toInputList(mode: .normalized).count == 6)
        #expect(result.lastAgent === agent1)
        // 2 tool calls, 2 tool outputs, 2 messages, 1 handoff call, and 1 handoff output.
        #expect(eventCounts["run_item_stream_event"] == 8)
        #expect(items.count == 8)
        #expect(agentUpdates.count == 2, "should have 2 agent updated events")
        #expect(agentUpdates.first?.newAgent === agent2, "should have started with agent_2")
        #expect(agentUpdates.last?.newAgent === agent1, "should have handed off to agent_1")
    }

    @Test func testDynamicToolAdditionRunStreamed() async throws {
        let model = try ScriptedModel()
        let executed = SharedBox(false)
        let agent = Agent(name: "test", model: .instance(model), toolUseBehavior: .runLLMAgain)
        let tool2 = try FunctionTool.create(name: "tool2") { _ in
            executed.mutate { $0 = true }
            return "result2"
        }
        let addTool = try FunctionTool.create(name: "add_tool") { [agent] _ in
            agent.tools.append(tool2)
            return "added"
        }
        agent.tools.append(addTool)
        try model.extend([
            .output([getFunctionToolCall("add_tool", arguments: "{}", callId: "call-add-tool")]),
            .output([getFunctionToolCall("tool2", arguments: "{}", callId: "call-tool-two")]),
            .output([getTextMessage("done")]),
        ])

        let result = try Runner.runStreamed(agent, input: "start")
        try await drainStream(result)

        #expect(executed.value)
        #expect(result.finalOutput as? String == "done")
    }

    @Test func testStreamStepItemsToQueueHandlesToolApprovalItem() async throws {
        let agent = Agent(name: "test", model: .instance(try ScriptedModel()))
        let approval = ToolApprovalItem(agent: agent, rawItem: getFunctionToolCall("test_tool", arguments: "{}"))
        let result = RunResultStreaming(input: "x", currentAgent: agent, contextWrapper: RunContextWrapper(), maxTurns: 1)

        streamStepItems([approval], to: result)
        result.finish()

        var count = 0
        for try await _ in result.streamEvents() { count += 1 }
        #expect(count == 0)
    }

    // MARK: - Approval resumes

    @Test func testStreamingHITLResumeWithApprovedTools() async throws {
        let toolCalled = SharedBox(false)
        let tool = try FunctionTool.create(name: "test_tool", needsApproval: .always) { _ in
            toolCalled.mutate { $0 = true }
            return "tool_result"
        }
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [tool], model: .instance(model))
        try model.extend([.output([getFunctionToolCall("test_tool", arguments: "{}")]), .output([getTextMessage("done")])])

        let first = try Runner.runStreamed(agent, input: "Use test_tool")
        try await drainStream(first)
        let state = first.toState()
        try state.approve(try #require(first.interruptions.first))
        let resumed = try Runner.runStreamed(agent, state: state)
        try await drainStream(resumed)

        #expect(toolCalled.value)
        #expect(resumed.finalOutput as? String == "done")
    }

    @Test func testStreamingResumeWithSessionDoesNotDuplicateItems() async throws {
        let tool = try FunctionTool.create(name: "test_tool", needsApproval: .always) { _ in "tool_result" }
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [tool], model: .instance(model))
        let session = SimpleListSession()
        try model.extend([.output([getFunctionToolCall("test_tool", arguments: "{}", callId: "call-resume")]), .output([getTextMessage("done")])])

        let first = try Runner.runStreamed(agent, input: "Use test_tool", session: session)
        try await drainStream(first)
        let state = first.toState()
        try state.approve(try #require(first.interruptions.first))
        let resumed = try Runner.runStreamed(agent, state: state, session: session)
        try await drainStream(resumed)
        #expect(resumed.finalOutput as? String == "done")

        let saved = try await session.getItems(limit: nil)
        #expect(saved.filter { $0.string("type") == "function_call" && $0.string("call_id") == "call-resume" }.count == 1)
        #expect(saved.filter { $0.string("type") == "function_call_output" && $0.string("call_id") == "call-resume" }.count == 1)
    }

    @Test(arguments: [false, true])
    func testRunLLMAgainApprovalPersistsCompletedSibling(streamed: Bool) async throws {
        let sideEffects = SharedBox<[String]>([])
        let approvalTool = try FunctionTool.create(name: "approval_tool", needsApproval: .always) { _ in
            sideEffects.mutate { $0.append("approved") }
            return "approved-output"
        }
        let siblingTool = try FunctionTool.create(name: "sibling_tool") { _ in
            sideEffects.mutate { $0.append("sibling") }
            return "sibling-output"
        }
        let model = try ScriptedModel(steps: [
            .output([
                getFunctionToolCall("approval_tool", arguments: "{}", callId: "call-approved"),
                getFunctionToolCall("sibling_tool", arguments: "{}", callId: "call-sibling"),
            ]),
            .output([getTextMessage("done")]),
        ])
        let agent = Agent(
            name: "test",
            tools: [approvalTool, siblingTool],
            model: .instance(model),
            outputGuardrails: [OutputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: false) }]
        )
        let session = SimpleListSession()

        let firstInterruptions: [ToolApprovalItem]
        let state: RunState
        if streamed {
            let first = try Runner.runStreamed(agent, input: "Use both tools", session: session)
            try await drainStream(first)
            firstInterruptions = first.interruptions
            state = first.toState()
        } else {
            let first = try await Runner.run(agent, input: "Use both tools", session: session)
            firstInterruptions = first.interruptions
            state = first.toState()
        }
        #expect(firstInterruptions.count == 1)
        #expect(sideEffects.value == ["sibling"])

        let siblingItems = try await session.getItems(limit: nil).filter { $0.string("call_id") == "call-sibling" }
        #expect(siblingItems.map { $0.string("type") } == ["function_call", "function_call_output"])
        #expect(siblingItems.last?.string("output") == "sibling-output")

        #expect(state.currentTurnPersistedItemCount > 0)
        try state.approve(try #require(firstInterruptions.first))
        let finalOutput: (any Sendable)?
        if streamed {
            let resumed = try Runner.runStreamed(agent, state: state, session: session)
            try await drainStream(resumed)
            finalOutput = resumed.finalOutput
        } else {
            finalOutput = try await Runner.run(agent, state: state, session: session).finalOutput
        }

        #expect(finalOutput as? String == "done")
        #expect(sideEffects.value == ["sibling", "approved"])
        let saved = try await session.getItems(limit: nil)
        for callId in ["call-sibling", "call-approved"] {
            #expect(saved.filter { $0.string("call_id") == callId }.map { $0.string("type") } == ["function_call", "function_call_output"])
        }
    }
}
