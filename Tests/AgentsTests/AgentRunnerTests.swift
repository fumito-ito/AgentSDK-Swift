import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_runner.py.
//
// Swift: blocked-output replacement, sandbox memory, OpenAI Conversations sessions, and
// `SQLiteSession` cases are pending their ports.
@Suite struct AgentRunnerTests {
    private func findReasoningInputItem(_ items: [TResponseInputItem]) -> TResponseInputItem? {
        items.first { $0.string("type") == "reasoning" }
    }

    private func findReasoningInputItem(_ input: RunInput?) -> TResponseInputItem? {
        guard case .items(let items)? = input else { return nil }
        return findReasoningInputItem(items)
    }

    private func reasoningItem(_ id: String, summary: String? = "Thinking...") -> TResponseOutputItem {
        let summaryParts: [JSONValue] = summary.map { [["text": .string($0), "type": "summary_text"]] } ?? []
        return ["id": .string(id), "type": "reasoning", "summary": .array(summaryParts)]
    }

    private func functionCallOutputs(_ input: RunInput?) -> [JSONObject] {
        guard case .items(let items)? = input else { return [] }
        return items.filter { $0.string("type") == "function_call_output" }
    }

    private func jsonArgs(_ object: JSONObject) -> String {
        JSONValue.object(object).pythonJSONString()
    }

    // MARK: - Basic runs

    @Test func testSimpleFirstRun() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", model: .instance(model))
        try model.enqueue(.output([getTextMessage("first")]))

        var result = try await Runner.run(agent, input: "test")
        #expect(result.input == "test")
        #expect(result.newItems.count == 1, "exactly one item should be generated")
        #expect(result.finalOutput as? String == "first")
        #expect(result.rawResponses.count == 1, "exactly one model response should be generated")
        #expect(result.rawResponses[0].output == [getTextMessage("first")])
        #expect(result.lastAgent === agent)
        #expect(try result.toInputList().count == 2, "should have original input and generated item")

        try model.enqueue(.output([getTextMessage("second")]))
        result = try await Runner.run(agent, input: .items([getTextInputItem("message"), getTextInputItem("another_message")]))
        #expect(result.newItems.count == 1, "exactly one item should be generated")
        #expect(result.finalOutput as? String == "second")
        #expect(result.rawResponses.count == 1, "exactly one model response should be generated")
        #expect(try result.toInputList().count == 3, "should have original input and generated item")
    }

    @Test func testSubsequentRuns() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", model: .instance(model))
        try model.enqueue(.output([getTextMessage("third")]))

        var result = try await Runner.run(agent, input: "test")
        #expect(result.input == "test")
        #expect(result.newItems.count == 1, "exactly one item should be generated")
        #expect(try result.toInputList().count == 2, "should have original input and generated item")

        try model.enqueue(.output([getTextMessage("fourth")]))
        result = try await Runner.run(agent, input: .items(try result.toInputList()))
        #expect(result.input.inputItems.count == 2, "should have previous input but got \(result.input)")
        #expect(result.newItems.count == 1, "exactly one item should be generated")
        #expect(result.finalOutput as? String == "fourth")
        #expect(result.rawResponses.count == 1, "exactly one model response should be generated")
        #expect(result.rawResponses[0].output == [getTextMessage("fourth")])
        #expect(result.lastAgent === agent)
        #expect(try result.toInputList().count == 3, "should have original input and generated items")
    }

    @Test func testToolCallRuns() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "foo", returnValue: "tool_result")], model: .instance(model))
        try model.extend([
            .output([getTextMessage("a_message"), getFunctionToolCall("foo", arguments: jsonArgs(["a": "b"]))]),
            .output([getTextMessage("done")]),
        ])

        let result = try await Runner.run(agent, input: "user_message")

        #expect(result.finalOutput as? String == "done")
        #expect(result.rawResponses.count == 2, "should have two responses")
        #expect(try result.toInputList().count == 5, "original input, message, tool call, tool result, and done message")
    }

    @Test func testParallelToolCallWithCancelledSiblingReachesFinalOutput() async throws {
        let model = try ScriptedModel()
        let agent = Agent(
            name: "test",
            tools: [
                try FunctionTool.create(name: "ok_tool") { _ in "ok" },
                try FunctionTool.create(name: "cancel_tool") { _ in throw CancellationError() },
            ],
            model: .instance(model)
        )
        try model.extend([
            .output([
                getFunctionToolCall("ok_tool", arguments: "{}", callId: "call_ok"),
                getFunctionToolCall("cancel_tool", arguments: "{}", callId: "call_cancel"),
            ]),
            .output([getTextMessage("final answer")]),
        ])

        let result = try await Runner.run(agent, input: "user_message")

        #expect(result.finalOutput as? String == "final answer")
        #expect(result.rawResponses.count == 2)
        #expect(functionCallOutputs(model.calls.last?.input) == [
            ["call_id": "call_ok", "output": "ok", "type": "function_call_output"],
            ["call_id": "call_cancel", "output": "An error occurred while running the tool. Please try again.", "type": "function_call_output"],
        ])
    }

    @Test func testSingleToolCallWithCancelledToolReachesFinalOutput() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [try FunctionTool.create(name: "cancel_tool") { _ in throw CancellationError() }], model: .instance(model))
        try model.extend([
            .output([getFunctionToolCall("cancel_tool", arguments: "{}", callId: "call_cancel")]),
            .output([getTextMessage("final answer")]),
        ])

        let result = try await Runner.run(agent, input: "user_message")

        #expect(result.finalOutput as? String == "final answer")
        #expect(result.rawResponses.count == 2)
        #expect(functionCallOutputs(model.calls.last?.input) == [
            ["call_id": "call_cancel", "output": "An error occurred while running the tool. Please try again.", "type": "function_call_output"],
        ])
    }

    // MARK: - Reasoning item IDs

    @Test func testReasoningItemIdPolicyOmitsFollowUpReasoningIds() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "foo", returnValue: "tool_result")], model: .instance(model))
        try model.extend([
            .output([reasoningItem("rs_first"), getFunctionToolCall("foo", arguments: jsonArgs(["a": "b"]), callId: "call_first")]),
            .output([getTextMessage("done")]),
        ])

        let result = try await Runner.run(agent, input: "hello", runConfig: RunConfig(reasoningItemIdPolicy: .omit))

        #expect(result.finalOutput as? String == "done")
        let secondRequestReasoning = try #require(findReasoningInputItem(model.calls.last?.input))
        #expect(secondRequestReasoning["id"] == nil)
        let historyReasoning = try #require(findReasoningInputItem(try result.toInputList()))
        #expect(historyReasoning["id"] == nil)
    }

    @Test func testCallModelInputFilterCanReintroduceReasoningIds() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "foo", returnValue: "tool_result")], model: .instance(model))
        try model.extend([
            .output([reasoningItem("rs_filter"), getFunctionToolCall("foo", arguments: jsonArgs(["a": "b"]), callId: "call_filter")]),
            .output([getTextMessage("done")]),
        ])
        let filter: CallModelInputFilter = { data in
            var modelData = data.modelData
            modelData.input = modelData.input.map { item in
                guard item.string("type") == "reasoning", item["id"] == nil else { return item }
                var updated = item
                updated["id"] = "rs_reintroduced"
                return updated
            }
            return modelData
        }

        let result = try await Runner.run(
            agent,
            input: "hello",
            runConfig: RunConfig(callModelInputFilter: filter, reasoningItemIdPolicy: .omit)
        )

        #expect(result.finalOutput as? String == "done")
        let secondRequestReasoning = try #require(findReasoningInputItem(model.calls.last?.input))
        #expect(secondRequestReasoning["id"] == "rs_reintroduced")
        let historyReasoning = try #require(findReasoningInputItem(try result.toInputList()))
        #expect(historyReasoning["id"] == nil)
    }

    @Test func testOmitPolicyStripsReasoningIdsAlreadyStoredInTheSession() async throws {
        let revoked = SharedBox<Set<String>>([])
        let base = try ScriptedModel()
        let model = HookedModel(base: base, getResponse: { request, base in
            for item in request.input.inputItems where item.string("type") == "reasoning" {
                if let id = item.string("id"), revoked.value.contains(id) {
                    throw OpenAIAPIError(statusCode: 404, message: "Item with id '\(id)' not found.")
                }
            }
            return try await base.getResponse(request)
        })
        let specialist = Agent(name: "specialist", model: .instance(model))
        let triage = Agent(name: "triage", handoffs: [.agent(specialist)], model: .instance(model))
        let session = SimpleListSession(sessionId: "issue-2020")

        try base.extend([
            .output([reasoningItem("rs_triage", summary: nil), getHandoffToolCall(specialist)]),
            .output([getTextMessage("handled")]),
        ])
        let first = try await Runner.run(triage, input: "hello", session: session)
        #expect(first.finalOutput as? String == "handled")
        let stored = try #require(findReasoningInputItem(try await session.getItems(limit: nil)))
        #expect(stored["id"] == "rs_triage")

        revoked.mutate { $0.insert("rs_triage") }
        try base.extend([.output([getTextMessage("done")])])
        let second = try await Runner.run(
            triage,
            input: "anything else?",
            runConfig: RunConfig(reasoningItemIdPolicy: .omit),
            session: session
        )

        #expect(second.finalOutput as? String == "done")
        let replayed = try #require(findReasoningInputItem(base.calls.last?.input))
        #expect(replayed["id"] == nil)
    }

    @Test func testResumedRunUsesSerializedReasoningItemIdPolicy() async throws {
        let model = try ScriptedModel()
        let approvalTool = try FunctionTool.create(name: "approval_tool", needsApproval: .always) { _ in "ok" }
        let agent = Agent(name: "test", tools: [approvalTool], model: .instance(model))
        try model.extend([
            .output([reasoningItem("rs_resume"), getFunctionToolCall("approval_tool", arguments: "{}", callId: "call_resume")]),
            .output([getTextMessage("done")]),
        ])

        let firstRun = try await Runner.run(agent, input: "hello", runConfig: RunConfig(reasoningItemIdPolicy: .omit))
        #expect(firstRun.interruptions.count == 1)

        let state = firstRun.toState()
        try state.approve(firstRun.interruptions[0])
        let restoredState = try await RunState.fromString(agent, try state.toString())

        let resumed = try await Runner.run(agent, state: restoredState)

        #expect(resumed.finalOutput as? String == "done")
        let secondRequestReasoning = try #require(findReasoningInputItem(model.calls.last?.input))
        #expect(secondRequestReasoning["id"] == nil)
    }

    // MARK: - Tool input guardrails and approvals

    @Test func testPendingApprovalSkipsToolInputGuardrailsByDefault() async throws {
        let model = try ScriptedModel()
        let runs = SharedBox(0)
        let guardrail = toolInputGuardrail { _ in
            runs.mutate { $0 += 1 }
            return .allow()
        }
        let approvalTool = try FunctionTool.create(name: "approval_tool", needsApproval: .always, toolInputGuardrails: [guardrail]) { _ in "ok" }
        let agent = Agent(name: "test", tools: [approvalTool], model: .instance(model))
        try model.enqueue(.output([getFunctionToolCall("approval_tool", arguments: "{}", callId: "call_default")]))

        let result = try await Runner.run(agent, input: "hello")

        #expect(result.interruptions.count == 1)
        #expect(runs.value == 0)
        #expect(result.toolInputGuardrailResults.isEmpty)
    }

    @Test func testPreApprovalToolInputGuardrailsCanRejectBeforePendingApproval() async throws {
        let model = try ScriptedModel()
        let executed = SharedBox(false)
        let guardrail = toolInputGuardrail { _ in .rejectContent("blocked before approval") }
        let approvalTool = try FunctionTool.create(name: "approval_tool", needsApproval: .always, toolInputGuardrails: [guardrail]) { _ in
            executed.mutate { $0 = true }
            return "ok"
        }
        let agent = Agent(name: "test", tools: [approvalTool], model: .instance(model))
        try model.extend([
            .output([getFunctionToolCall("approval_tool", arguments: "{}", callId: "call_reject")]),
            .output([getTextMessage("done")]),
        ])

        let result = try await Runner.run(
            agent,
            input: "hello",
            runConfig: RunConfig(toolExecution: try ToolExecutionConfig(preApprovalToolInputGuardrails: true))
        )

        #expect(result.finalOutput as? String == "done")
        #expect(result.interruptions.isEmpty)
        #expect(!executed.value)
        #expect(result.toolInputGuardrailResults.count == 1)
        #expect(result.newItems.contains { ($0 as? ToolCallOutputItem)?.output as? String == "blocked before approval" })
    }

    @Test func testPreApprovalToolInputGuardrailsRerunAfterResume() async throws {
        let model = try ScriptedModel()
        let guardrailRuns = SharedBox(0)
        let executed = SharedBox(0)
        let guardrail = toolInputGuardrail { _ in
            guardrailRuns.mutate { $0 += 1 }
            return .allow()
        }
        let approvalTool = try FunctionTool.create(name: "approval_tool", needsApproval: .always, toolInputGuardrails: [guardrail]) { _ in
            executed.mutate { $0 += 1 }
            return "ok"
        }
        let agent = Agent(name: "test", tools: [approvalTool], model: .instance(model))
        try model.extend([
            .output([getFunctionToolCall("approval_tool", arguments: "{}", callId: "call_resume")]),
            .output([getTextMessage("done")]),
        ])
        let runConfig = RunConfig(toolExecution: try ToolExecutionConfig(preApprovalToolInputGuardrails: true))

        let first = try await Runner.run(agent, input: "hello", runConfig: runConfig)
        #expect(first.interruptions.count == 1)
        #expect(guardrailRuns.value == 1)
        #expect(executed.value == 0)
        #expect(first.toolInputGuardrailResults.count == 1)

        let state = first.toState()
        try state.approve(first.interruptions[0])
        let restoredState = try await RunState.fromString(agent, try state.toString())

        let resumed = try await Runner.run(agent, state: restoredState, runConfig: runConfig)

        #expect(resumed.finalOutput as? String == "done")
        #expect(guardrailRuns.value == 2)
        #expect(executed.value == 1)
        #expect(resumed.toolInputGuardrailResults.count == 1)
    }

    @Test func testToolCallContextIncludesCurrentAgent() async throws {
        let model = try ScriptedModel()
        let captured = SharedBox<[ToolContext]>([])
        let foo = try FunctionTool.create(name: "foo") { context in
            captured.mutate { $0.append(context) }
            return "tool_result"
        }
        let agent = Agent(name: "test", tools: [foo], model: .instance(model))
        try model.extend([.output([getFunctionToolCall("foo", arguments: "{}")]), .output([getTextMessage("done")])])

        let result = try await Runner.run(agent, input: "user_message")

        #expect(result.finalOutput as? String == "done")
        #expect(captured.value.count == 1)
        #expect(captured.value.first?.agent === agent)
    }

    // MARK: - Handoffs

    @Test func testHandoffs() async throws {
        let model = try ScriptedModel()
        let agent1 = Agent(name: "test", model: .instance(model))
        let agent2 = Agent(name: "test", model: .instance(model))
        let agent3 = Agent(
            name: "test",
            tools: [getFunctionTool(name: "some_function", returnValue: "result")],
            handoffs: [.agent(agent1), .agent(agent2)],
            model: .instance(model)
        )
        try model.extend([
            .output([getFunctionToolCall("some_function", arguments: jsonArgs(["a": "b"]))]),
            .output([getTextMessage("a_message"), getHandoffToolCall(agent1)]),
            .output([getTextMessage("done")]),
        ])

        let result = try await Runner.run(agent3, input: "user_message")

        #expect(result.finalOutput as? String == "done")
        #expect(result.rawResponses.count == 3, "should have three model responses")
        #expect(try result.toInputList().count == 7, "summary message, tool call, tool result, message, handoff, handoff result, and done message")
        // Swift: upstream compares dataclasses by value, and agent_1 and agent_2 are field-identical.
        // Both handoffs derive `transfer_to_test`, so the collision policy routes to the last one.
        #expect(result.lastAgent === agent1 || result.lastAgent === agent2, "should have handed off to agent_1")
    }

    // MARK: - Nested handoff history

    private func inputTypes(_ input: [TResponseInputItem]) -> [String] {
        input.compactMap { $0.string("type") }
    }

    @Test func testNestedHandoffFiltersModelInputButPreservesSessionItems() async throws {
        let model = try ScriptedModel()
        let delegate = Agent(name: "delegate", model: .instance(model))
        let triage = Agent(
            name: "triage",
            tools: [getFunctionTool(name: "some_function", returnValue: "result")],
            handoffs: [.agent(delegate)],
            model: .instance(model)
        )
        try model.extend([
            .output([getFunctionToolCall("some_function", arguments: jsonArgs(["a": "b"]))]),
            .output([getTextMessage("a_message"), getHandoffToolCall(delegate)]),
            .output([getTextMessage("done")]),
        ])
        let captured = SharedBox<[[String]]>([])
        let capture: CallModelInputFilter = { data in
            captured.mutate { $0.append(self.inputTypes(data.modelData.input)) }
            return data.modelData
        }
        let session = SimpleListSession()

        let result = try await Runner.run(
            triage,
            input: "user_message",
            runConfig: RunConfig(nestHandoffHistory: true, callModelInputFilter: capture),
            session: session
        )

        #expect(result.finalOutput as? String == "done")
        #expect(captured.value.count >= 3)
        let handoffInputTypes = captured.value[2]
        #expect(!handoffInputTypes.contains("function_call"))
        #expect(!handoffInputTypes.contains("function_call_output"))
        #expect(result.newItems.contains { $0 is ToolCallOutputItem })
        #expect(result.newItems.contains { $0 is HandoffOutputItem })
        #expect(try await session.getItems(limit: nil).contains { $0.string("type") == "function_call_output" })
    }

    @Test func testNestedHandoffFiltersReasoningItemsFromModelInput() async throws {
        let model = try ScriptedModel()
        let delegate = Agent(name: "delegate", model: .instance(model))
        let triage = Agent(name: "triage", handoffs: [.agent(delegate)], model: .instance(model))
        try model.extend([
            .output([reasoningItem("reasoning_1", summary: "Thinking about a handoff."), getHandoffToolCall(delegate)]),
            .output([getTextMessage("done")]),
        ])
        let captured = SharedBox<[[TResponseInputItem]]>([])
        let capture: CallModelInputFilter = { data in
            captured.mutate { $0.append(data.modelData.input) }
            return data.modelData
        }

        let result = try await Runner.run(
            triage,
            input: "user_message",
            runConfig: RunConfig(nestHandoffHistory: true, callModelInputFilter: capture)
        )

        #expect(result.finalOutput as? String == "done")
        #expect(captured.value.count >= 2)
        #expect(!inputTypes(captured.value[1]).contains("reasoning"))
    }

    @Test func testResumePreservesFilteredModelInputAfterHandoff() async throws {
        let model = try ScriptedModel()
        let approvalTool = try FunctionTool.create(name: "approval_tool", needsApproval: .always) { _ in "ok" }
        let delegate = Agent(name: "delegate", tools: [approvalTool], model: .instance(model))
        let triage = Agent(
            name: "triage",
            tools: [getFunctionTool(name: "some_function", returnValue: "result")],
            handoffs: [.agent(delegate)],
            model: .instance(model)
        )
        try model.extend([
            .output([getFunctionToolCall("some_function", arguments: jsonArgs(["a": "b"]), callId: "triage-call")]),
            .output([getTextMessage("a_message"), getHandoffToolCall(delegate)]),
            .output([getFunctionToolCall("approval_tool", arguments: "{}", callId: "delegate-call")]),
            .output([getTextMessage("done")]),
        ])
        let callIds = SharedBox<[Set<String>]>([])
        let outputCallIds = SharedBox<[Set<String>]>([])
        let capture: CallModelInputFilter = { data in
            var calls: Set<String> = []
            var outputs: Set<String> = []
            for item in data.modelData.input {
                guard let callId = item.string("call_id") else { continue }
                if item.string("type") == "function_call" { calls.insert(callId) }
                if item.string("type") == "function_call_output" { outputs.insert(callId) }
            }
            callIds.mutate { $0.append(calls) }
            outputCallIds.mutate { $0.append(outputs) }
            return data.modelData
        }
        let runConfig = RunConfig(nestHandoffHistory: true, callModelInputFilter: capture)

        let first = try await Runner.run(triage, input: "user_message", runConfig: runConfig)
        #expect(!first.interruptions.isEmpty)
        let state = first.toState()
        try state.approve(first.interruptions[0])
        let resumed = try await Runner.run(triage, state: state, runConfig: runConfig)

        let lastCallIds = try #require(callIds.value.last)
        let lastOutputCallIds = try #require(outputCallIds.value.last)
        #expect(!lastCallIds.contains("triage-call"))
        #expect(!lastOutputCallIds.contains("triage-call"))
        #expect(lastCallIds.contains("delegate-call"))
        #expect(lastOutputCallIds.contains("delegate-call"))
        #expect(resumed.finalOutput as? String == "done")
    }

    @Test func testResumedStateUpdatesAgentAfterHandoff() async throws {
        let model = try ScriptedModel()
        let triageTool = try FunctionTool.create(name: "triage_tool", needsApproval: .always) { _ in "ok" }
        let delegateTool = try FunctionTool.create(name: "delegate_tool", needsApproval: .always) { _ in "ok" }
        let delegate = Agent(name: "delegate", tools: [delegateTool], model: .instance(model))
        let triage = Agent(name: "triage", tools: [triageTool], handoffs: [.agent(delegate)], model: .instance(model))
        try model.extend([
            .output([getFunctionToolCall("triage_tool", arguments: "{}", callId: "triage-1")]),
            .output([getTextMessage("handoff"), getHandoffToolCall(delegate)]),
            .output([getFunctionToolCall("delegate_tool", arguments: "{}", callId: "delegate-1")]),
        ])

        let first = try await Runner.run(triage, input: "user_message")
        #expect(!first.interruptions.isEmpty)
        let state = first.toState()
        try state.approve(first.interruptions[0])

        let second = try await Runner.run(triage, state: state)
        #expect(!second.interruptions.isEmpty)
        #expect(second.interruptions.contains { $0.toolName == delegateTool.name }, "handoff should switch approvals to the delegate agent")
        #expect(state.currentAgent === delegate)
    }

    // MARK: - Structured output and filters

    private struct Foo: Codable, Sendable, Equatable {
        var bar: String
    }

    @Test func testStructuredOutput() async throws {
        let model = try ScriptedModel()
        let agent1 = Agent(
            name: "test",
            tools: [getFunctionTool(name: "bar", returnValue: "bar_result")],
            model: .instance(model),
            outputType: try AgentOutputSchema(Foo.self)
        )
        let agent2 = Agent(
            name: "test",
            tools: [getFunctionTool(name: "foo", returnValue: "foo_result")],
            handoffs: [.agent(agent1)],
            model: .instance(model)
        )
        try model.extend([
            .output([getFunctionToolCall("foo", arguments: jsonArgs(["bar": "baz"]), callId: "call_foo")]),
            .output([getTextMessage("a_message"), getHandoffToolCall(agent1)]),
            .output([
                getTextMessage(jsonArgs(["bar": "preamble"])),
                getFunctionToolCall("bar", arguments: jsonArgs(["bar": "baz"]), callId: "call_bar"),
            ]),
            .output([getFinalOutputMessage(jsonArgs(["bar": "baz"]))]),
        ])

        let result = try await Runner.run(
            agent2,
            input: .items([getTextInputItem("user_message"), getTextInputItem("another_message")]),
            runConfig: RunConfig(nestHandoffHistory: true)
        )

        #expect(result.finalOutput as? Foo == Foo(bar: "baz"))
        #expect(result.rawResponses.count == 4, "should have four model responses")
        #expect(try result.toInputList().count == 11, "ordered history segments plus calls, messages, handoff items, and final output")
        #expect(try result.toInputList(mode: .normalized).count == 7, "normalized replay input")
        #expect(result.lastAgent === agent1, "should have handed off to agent_1")
    }

    @Test func testHandoffFilters() async throws {
        let model = try ScriptedModel()
        let agent1 = Agent(name: "test", model: .instance(model))
        let removeNewItems: HandoffInputFilter = { data in
            HandoffInputData(inputHistory: data.inputHistory, preHandoffItems: [], newItems: [], runContext: data.runContext)
        }
        let agent2 = Agent(name: "test", handoffs: [.handoff(handoff(agent1, inputFilter: removeNewItems))], model: .instance(model))
        try model.extend([
            .output([getTextMessage("1"), getTextMessage("2"), getHandoffToolCall(agent1)]),
            .output([getTextMessage("last")]),
        ])

        let result = try await Runner.run(agent2, input: "user_message")

        #expect(result.finalOutput as? String == "last")
        #expect(result.rawResponses.count == 2, "should have two model responses")
        #expect(try result.toInputList().count == 2, "should only have 2 inputs: orig input and last message")
    }

    // MARK: - Opt-in handoff history

    private func asMessage(_ item: TResponseInputItem, sourceLocation: SourceLocation = #_sourceLocation) -> TResponseInputItem {
        let role = item.string("role")
        #expect(["assistant", "user", "system", "developer"].contains(role ?? ""), sourceLocation: sourceLocation)
        return item
    }

    private func inputMessageText(_ item: TResponseInputItem) -> String {
        let message = asMessage(item)
        if let text = message.string("content") { return text }
        return (message["content"]?.arrayValue ?? []).compactMap { $0.objectValue?.string("text") }.joined()
    }

    /// Runs `agent` and drains the stream when `streamed` (upstream `_run_agent_with_optional_streaming`).
    private func runAgent(
        _ agent: Agent,
        input: RunInput,
        streamed: Bool,
        runConfig: RunConfig? = nil,
        autoPreviousResponseId: Bool = false
    ) async throws -> any RunResultBase {
        if streamed {
            let result = try Runner.runStreamed(agent, input: input, runConfig: runConfig, autoPreviousResponseId: autoPreviousResponseId)
            for try await _ in result.streamEvents() {}
            return result
        }
        return try await Runner.run(agent, input: input, runConfig: runConfig, autoPreviousResponseId: autoPreviousResponseId)
    }

    @Test func testOptInHandoffHistoryNestedAndFiltersRespected() async throws {
        let model = try ScriptedModel()
        let delegate = Agent(name: "delegate", model: .instance(model))
        let triage = Agent(name: "triage", handoffs: [.agent(delegate)], model: .instance(model))
        try model.extend([
            .output([getTextMessage("triage summary"), getHandoffToolCall(delegate)]),
            .output([getTextMessage("resolution")]),
        ])

        let result = try await Runner.run(triage, input: "user_message", runConfig: RunConfig(nestHandoffHistory: true))

        guard case .items(let input) = result.input else {
            Issue.record("Expected list input")
            return
        }
        #expect(input.count == 3)
        let summary = asMessage(input[0])
        #expect(summary.string("role") == "assistant")
        let summaryContent = try #require(summary.string("content"))
        #expect(summaryContent.contains("<CONVERSATION HISTORY>"))
        #expect(!summaryContent.contains("triage summary"))
        #expect(summaryContent.contains("user_message"))
        #expect(inputMessageText(input[1]) == "triage summary")
        #expect(inputMessageText(input[2]).contains("transfer_to_delegate"))
        let delegateInput = try #require(model.calls.last?.input.inputItems)
        #expect(delegateInput.count == 3)
        #expect(inputMessageText(delegateInput[1]) == "triage summary")

        let passthroughModel = try ScriptedModel()
        let passthroughDelegate = Agent(name: "delegate", model: .instance(passthroughModel))
        let triageWithFilter = Agent(
            name: "triage",
            handoffs: [.handoff(handoff(passthroughDelegate, inputFilter: { $0 }))],
            model: .instance(passthroughModel)
        )
        try passthroughModel.extend([
            .output([getTextMessage("triage summary"), getHandoffToolCall(passthroughDelegate)]),
            .output([getTextMessage("resolution")]),
        ])

        let filteredResult = try await Runner.run(triageWithFilter, input: "user_message", runConfig: RunConfig(nestHandoffHistory: true))

        #expect(filteredResult.input == "user_message")
    }

    @Test func testOptInHandoffHistoryAccumulatesAcrossMultipleHandoffs() async throws {
        let triageModel = try ScriptedModel()
        let delegateModel = try ScriptedModel()
        let closerModel = try ScriptedModel()
        let closer = Agent(name: "closer", model: .instance(closerModel))
        let delegate = Agent(name: "delegate", handoffs: [.agent(closer)], model: .instance(delegateModel))
        let triage = Agent(name: "triage", handoffs: [.agent(delegate)], model: .instance(triageModel))
        try triageModel.extend([.output([getTextMessage("triage summary"), getHandoffToolCall(delegate)])])
        try delegateModel.extend([.output([getTextMessage("delegate update"), getHandoffToolCall(closer)])])
        try closerModel.extend([.output([getTextMessage("resolution")])])

        let result = try await Runner.run(triage, input: "user_question", runConfig: RunConfig(nestHandoffHistory: true))

        #expect(result.finalOutput as? String == "resolution")
        let closerInput = try #require(closerModel.calls.first?.input.inputItems)
        let summary = asMessage(closerInput[0])
        #expect(summary.string("role") == "assistant")
        let summaryContent = try #require(summary.string("content"))
        #expect(summaryContent.components(separatedBy: "<CONVERSATION HISTORY>").count - 1 == 1)
        #expect(summaryContent.contains("triage summary"))
        #expect(!summaryContent.contains("delegate update"))
        #expect(summaryContent.contains("user_question"))
        #expect(closerInput.count == 3)
        #expect(inputMessageText(closerInput[1]) == "delegate update")
        #expect(inputMessageText(closerInput[2]).contains("transfer_to_closer"))
    }

    @Test(arguments: [false, true], ["run_config", "handoff"])
    func testServerManagedHandoffHistoryAutoDisablesWithWarning(streamed: Bool, nestSource: String) async throws {
        let triageModel = try ScriptedModel()
        let delegateModel = try ScriptedModel()
        let delegate = Agent(name: "delegate", model: .instance(delegateModel))
        let runConfig: RunConfig
        let triageHandoffs: [HandoffTarget]
        if nestSource == "handoff" {
            triageHandoffs = [.handoff(handoff(delegate, nestHandoffHistory: true))]
            runConfig = RunConfig()
        } else {
            triageHandoffs = [.agent(delegate)]
            runConfig = RunConfig(nestHandoffHistory: true)
        }
        let triage = Agent(name: "triage", handoffs: triageHandoffs, model: .instance(triageModel))
        try triageModel.extend([.output([getTextMessage("triage summary"), getHandoffToolCall(delegate)])])
        try delegateModel.extend([.output([getTextMessage("done")])])

        try await captureLogs(level: .warning) { capture in
            let result = try await runAgent(triage, input: "user_message", streamed: streamed, runConfig: runConfig, autoPreviousResponseId: true)
            #expect(result.finalOutput as? String == "done")
            #expect(capture.messages.joined(separator: "\n").contains("do not support nest_handoff_history"))
        }
        let delegateInput = try #require(delegateModel.calls.first?.input.inputItems)
        #expect(delegateInput.count == 1)
        #expect(delegateInput[0].string("type") == "function_call_output")
        #expect(delegateInput[0]["output"]?.stringValue?.contains("delegate") == true)
        #expect(!delegateInput.contains { $0.string("role") == "assistant" && ($0.string("content") ?? "").contains("<CONVERSATION HISTORY>") })
    }

    @Test(arguments: [false, true], ["run_config", "handoff"])
    func testServerManagedHandoffInputFiltersStillRaise(streamed: Bool, filterSource: String) async throws {
        let triageModel = try ScriptedModel()
        let delegateModel = try ScriptedModel()
        let delegate = Agent(name: "delegate", model: .instance(delegateModel))
        let passthrough: HandoffInputFilter = { $0 }
        let runConfig: RunConfig
        let triageHandoffs: [HandoffTarget]
        if filterSource == "handoff" {
            triageHandoffs = [.handoff(handoff(delegate, inputFilter: passthrough))]
            runConfig = RunConfig()
        } else {
            triageHandoffs = [.agent(delegate)]
            runConfig = RunConfig(handoffInputFilter: passthrough)
        }
        let triage = Agent(name: "triage", handoffs: triageHandoffs, model: .instance(triageModel))
        try triageModel.extend([.output([getTextMessage("triage summary"), getHandoffToolCall(delegate)])])
        try delegateModel.extend([.output([getTextMessage("done")])])

        await expectErrorAsync(UserError.self, containing: "Server-managed conversations do not support handoff input filters") {
            _ = try await runAgent(triage, input: "user_message", streamed: streamed, runConfig: runConfig, autoPreviousResponseId: true)
        }
        #expect(delegateModel.calls.isEmpty)
    }

    @Test func testAsyncInputFilterSupported() async throws {
        let model = try ScriptedModel()
        let agent1 = Agent(name: "test", model: .instance(model))
        let agent2 = Agent(name: "test", handoffs: [
            .handoff(Handoff(
                toolName: Handoff.defaultToolName(agent1),
                toolDescription: Handoff.defaultToolDescription(agent1),
                inputJSONSchema: [:],
                onInvokeHandoff: { _, _ in agent1 },
                agentName: agent1.name,
                inputFilter: { data in
                    await Task.yield()
                    return data
                }
            )),
        ], model: .instance(model))
        try model.extend([
            .output([getTextMessage("1"), getTextMessage("2"), getHandoffToolCall(agent1)]),
            .output([getTextMessage("last")]),
        ])

        let result = try await Runner.run(agent2, input: "user_message")
        #expect(result.finalOutput as? String == "last")
    }

    private struct FooInput: Codable, Sendable {
        var bar: String
    }

    @Test func testHandoffOnInput() async throws {
        let callOutput = SharedBox<String?>(nil)
        let model = try ScriptedModel()
        let agent1 = Agent(name: "test", model: .instance(model))
        let agent2 = Agent(name: "test", handoffs: [
            .handoff(try handoff(agent1, inputType: FooInput.self, onHandoff: { _, data in callOutput.mutate { $0 = data.bar } })),
        ], model: .instance(model))
        try model.extend([
            .output([getTextMessage("1"), getTextMessage("2"), getHandoffToolCall(agent1, args: jsonArgs(["bar": "test_input"]))]),
            .output([getTextMessage("last")]),
        ])

        let result = try await Runner.run(agent2, input: "user_message")

        #expect(result.finalOutput as? String == "last")
        #expect(callOutput.value == "test_input", "should have called the handoff with the correct input")
    }

    @Test func testInvalidHandoffInputJSONCausesError() async throws {
        let object = try handoff(Agent(name: "test"), inputType: FooInput.self, onHandoff: { _, _ in })
        await expectErrorAsync(ModelBehaviorError.self) {
            _ = try await object.onInvokeHandoff(RunContextWrapper(), "invalid")
        }
    }

    // MARK: - Input guardrails

    @Test func testInputGuardrailTripwireTriggeredCausesException() async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("user_message")]))
        let agent = Agent(
            name: "test",
            model: .instance(model),
            inputGuardrails: [InputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: true) }]
        )

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            _ = try await Runner.run(agent, input: "user_message")
        }
    }

    @Test func testInputGuardrailTripwireDoesNotSaveAssistantMessageToSession() async throws {
        let session = SimpleListSession()
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("should_not_be_saved")]))
        let guardrail = InputGuardrail { _, _, _ in
            try await sleep(seconds: 0.01)
            return GuardrailFunctionOutput(tripwireTriggered: true)
        }
        let agent = Agent(name: "test", model: .instance(model), inputGuardrails: [guardrail])

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            _ = try await Runner.run(agent, input: "user_message", session: session)
        }

        let items = try await session.getItems(limit: nil)
        #expect(items.count == 1)
        #expect(items.first?.string("role") == "user")
    }
}
