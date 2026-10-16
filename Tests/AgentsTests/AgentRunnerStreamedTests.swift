import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_runner_streamed.py.
//
// Swift: websocket-model, Chat Completions guardrail-policy, blocked-output replacement,
// serialized-RunState, and OpenAI Conversations session cases are pending their ports.
// `SQLiteSession` cases use `SimpleListSession`. Stream-closing-task identity checks are skipped
// (Swift streams are cancelled through `onTermination`).
@Suite struct AgentRunnerStreamedTests {
    private func consume(_ result: RunResultStreaming) async throws -> [StreamEvent] {
        var events: [StreamEvent] = []
        for try await event in result.streamEvents() { events.append(event) }
        return events
    }

    private func jsonArgs(_ object: JSONObject) -> String {
        JSONValue.object(object).pythonJSONString()
    }

    private func networkRetrySettings() -> ModelSettings {
        ModelSettings(retry: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError()))
    }

    // MARK: - Basics

    @Test func testSimpleFirstRun() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", model: .instance(model))
        try model.enqueue(.output([getTextMessage("first")]))

        var result = try Runner.runStreamed(agent, input: "test")
        _ = try await consume(result)
        #expect(result.input == "test")
        #expect(result.newItems.count == 1, "exactly one item should be generated")
        #expect(result.finalOutput as? String == "first")
        #expect(result.rawResponses.count == 1, "exactly one model response should be generated")
        #expect(result.rawResponses[0].output == [getTextMessage("first")])
        #expect(result.lastAgent === agent)
        #expect(try result.toInputList().count == 2, "should have original input and generated item")

        try model.enqueue(.output([getTextMessage("second")]))
        result = try Runner.runStreamed(agent, input: .items([getTextInputItem("message"), getTextInputItem("another_message")]))
        _ = try await consume(result)
        #expect(result.newItems.count == 1, "exactly one item should be generated")
        #expect(result.finalOutput as? String == "second")
        #expect(result.rawResponses.count == 1, "exactly one model response should be generated")
        #expect(try result.toInputList().count == 3, "should have original input and generated item")
    }

    @Test func testEmptyListInputReachesModel() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", model: .instance(model))
        try model.enqueue(.output([getTextMessage("first")]))

        let result = try Runner.runStreamed(agent, input: .items([]))
        _ = try await consume(result)

        #expect(result.finalOutput as? String == "first")
        #expect(model.calls.last?.input == .items([]))
    }

    @Test func testStreamedToolNotFoundBehaviorReturnsErrorToModel() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", model: .instance(model))
        try model.extend([
            .output([getFunctionToolCall("missing_tool", arguments: "{}", callId: "call_missing")]),
            .output([getTextMessage("recovered")]),
        ])

        let result = try Runner.runStreamed(agent, input: "start", runConfig: RunConfig(toolNotFoundBehavior: .returnErrorToModel))
        _ = try await consume(result)

        #expect(result.finalOutput as? String == "recovered")
        let outputs = (model.calls.last?.input.inputItems ?? []).filter { $0.string("type") == "function_call_output" }
        #expect(outputs.map { $0.string("call_id") } == ["call_missing"])
        #expect(outputs.map { $0.string("output") } == ["Tool 'missing_tool' not found."])
    }

    // MARK: - Terminal events

    @Test(arguments: ["response.incomplete", "response.failed"])
    func testStreamedRunRejectsFailedTerminalResponsePayloadEvents(terminalEventType: String) async throws {
        let model = HookedModel(base: try ScriptedModel(), stream: { _, _ in
            makeStream { continuation in
                let response = getResponseObj([getTextMessage("partial final")], responseId: "resp-partial")
                continuation.yield(["type": .string(terminalEventType), "response": .object(response.json), "sequence_number": 0])
            }
        })
        let agent = Agent(name: "test", model: .instance(model))

        let result = try Runner.runStreamed(agent, input: "test")
        var events: [StreamEvent] = []
        await expectErrorAsync(ModelBehaviorError.self, containing: terminalEventType) {
            for try await event in result.streamEvents() { events.append(event) }
        }

        #expect(events.count == 2)
        if case .agentUpdated = events.first {} else { Issue.record("Expected an agent updated event first") }
        if case .rawResponse(let raw) = events.last {
            #expect(raw.data.string("type") == terminalEventType)
        } else {
            Issue.record("Expected a raw response event")
        }
        #expect(result.finalOutput == nil)
        #expect(result.rawResponses.isEmpty)
    }

    @Test func testStreamedRunRejectsResponseErrorTerminalEvent() async throws {
        let model = HookedModel(base: try ScriptedModel(), stream: { _, _ in
            makeStream { continuation in
                continuation.yield(["type": "error", "code": "invalid_request_error", "message": "bad request", "param": nil, "sequence_number": 0])
            }
        })
        let agent = Agent(name: "test", model: .instance(model))

        let result = try Runner.runStreamed(agent, input: "test")
        var events: [StreamEvent] = []
        await expectErrorAsync(ModelBehaviorError.self, containing: "error") {
            for try await event in result.streamEvents() { events.append(event) }
        }

        #expect(events.count == 2)
        if case .rawResponse(let raw) = events.last {
            #expect(raw.data.string("type") == "error")
            #expect(raw.data.string("code") == "invalid_request_error")
            #expect(raw.data.string("message") == "bad request")
        } else {
            Issue.record("Expected a raw response event")
        }
        #expect(result.finalOutput == nil)
        #expect(result.rawResponses.isEmpty)
    }

    @Test(arguments: [nil, false, true] as [Bool?])
    func testStreamedRunExposesRequestIdOnRawResponses(preserveRawUsage: Bool?) async throws {
        let model = HookedModel(base: try ScriptedModel(), stream: { _, _ in
            makeStream { continuation in
                var response = getResponseObj([getTextMessage("partial final")], responseId: "resp-partial").json
                response[sseRequestIdKey] = "req_streamed_result_123"
                attachRawUsageSnapshot(&response, ["input_tokens": 3, "input_tokens_details": ["cached_tokens": 0]])
                continuation.yield(["type": "response.completed", "response": .object(response), "sequence_number": 0])
            }
        })
        let agent = Agent(name: "test", model: .instance(model), modelSettings: ModelSettings(preserveRawUsage: preserveRawUsage))

        let result = try Runner.runStreamed(agent, input: "test")
        _ = try await consume(result)

        #expect(result.rawResponses.count == 1)
        #expect(result.rawResponses.first?.requestId == "req_streamed_result_123")
        #expect(result.rawResponses.first?.rawUsage == (preserveRawUsage == true ? ["input_tokens": 3, "input_tokens_details": ["cached_tokens": 0]] : nil))
    }

    // MARK: - Usage and retries

    @Test func testStreamedRunPreservesRequestUsageEntriesAfterRetry() async throws {
        let model = try ScriptedModel()
        model.setDefaultUsage(Usage(requests: 1, inputTokens: 10, outputTokens: 5, totalTokens: 15))
        try model.extend([
            .raiseError(OpenAIAPIError(message: "connection error", isConnectionError: true)),
            .output([getTextMessage("done")]),
        ])
        let agent = Agent(name: "test", model: .instance(model), modelSettings: networkRetrySettings())

        let result = try Runner.runStreamed(agent, input: "test")
        _ = try await consume(result)

        let usage = result.contextWrapper.usage
        #expect(usage.requests == 2)
        #expect(usage.requestUsageEntries.count == 2)
        #expect(usage.requestUsageEntries.first?.totalTokens == 0)
        #expect(usage.requestUsageEntries.last?.inputTokens == 10)
        #expect(usage.requestUsageEntries.last?.outputTokens == 5)
        #expect(usage.requestUsageEntries.last?.totalTokens == 15)
    }

    @Test func testStreamedRunCountsRetryAttemptsWhenTerminalUsageMissing() async throws {
        let model = try ScriptedModel(steps: [
            .raiseError(OpenAIAPIError(message: "connection error", isConnectionError: true)),
            .stream { _ in
                makeStream { continuation in
                    var response = getResponseObj([getTextMessage("done")]).json
                    response["usage"] = .null
                    continuation.yield(["type": "response.completed", "response": .object(response), "sequence_number": 0])
                }
            },
        ])
        let agent = Agent(name: "test", model: .instance(model), modelSettings: networkRetrySettings())

        let result = try Runner.runStreamed(agent, input: "test")
        _ = try await consume(result)

        let usage = result.contextWrapper.usage
        #expect(model.calls.count == 2)
        #expect(usage.requests == 2)
        #expect(usage.requestUsageEntries.count == 2)
        #expect(usage.requestUsageEntries.allSatisfy { $0.totalTokens == 0 })
    }

    @Test func testStreamedModelRetryDoesNotRewindCommittedSessionInput() async throws {
        let model = try ScriptedModel()
        try model.extend([
            .raiseError(OpenAIAPIError(message: "connection error", isConnectionError: true)),
            .output([getTextMessage("done")]),
        ])
        let agent = Agent(name: "test", model: .instance(model), modelSettings: networkRetrySettings())
        let session = CountingSession(history: [getTextInputItem("previous")])

        _ = try await consume(try Runner.runStreamed(agent, input: "test", session: session))

        #expect(try await session.getItems(limit: nil).map { $0.string("role") } == ["user", "user", "assistant"])
        #expect(session.popCalls == 0)
    }

    @Test func testStreamedRunPreservesRequestUsageEntriesAfterConversationLockedRetry() async throws {
        let model = try ScriptedModel()
        model.setDefaultUsage(Usage(requests: 1, inputTokens: 10, outputTokens: 5, totalTokens: 15))
        try model.extend([
            .raiseError(OpenAIAPIError(statusCode: 400, errorCode: "conversation_locked", message: "Conversation locked")),
            .output([getTextMessage("done")]),
        ])
        let agent = Agent(name: "test", model: .instance(model), modelSettings: networkRetrySettings())

        let result = try Runner.runStreamed(agent, input: "test")
        _ = try await consume(result)

        let usage = result.contextWrapper.usage
        #expect(usage.requests == 2)
        #expect(usage.requestUsageEntries.count == 2)
        #expect(usage.requestUsageEntries.first?.totalTokens == 0)
        #expect(usage.requestUsageEntries.last?.totalTokens == 15)
    }

    // MARK: - Turns

    @Test func testSubsequentRuns() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", model: .instance(model))
        try model.enqueue(.output([getTextMessage("third")]))

        var result = try Runner.runStreamed(agent, input: "test")
        _ = try await consume(result)
        #expect(result.input == "test")
        #expect(result.newItems.count == 1)
        #expect(try result.toInputList().count == 2)

        try model.enqueue(.output([getTextMessage("fourth")]))
        result = try Runner.runStreamed(agent, input: .items(try result.toInputList()))
        _ = try await consume(result)
        #expect(result.input.inputItems.count == 2)
        #expect(result.newItems.count == 1)
        #expect(result.finalOutput as? String == "fourth")
        #expect(result.rawResponses.count == 1)
        #expect(result.rawResponses[0].output == [getTextMessage("fourth")])
        #expect(result.lastAgent === agent)
        #expect(try result.toInputList().count == 3)
    }

    @Test func testToolCallRuns() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "foo", returnValue: "tool_result")], model: .instance(model))
        try model.extend([
            .output([getTextMessage("a_message"), getFunctionToolCall("foo", arguments: jsonArgs(["a": "b"]))]),
            .output([getTextMessage("done")]),
        ])

        let result = try Runner.runStreamed(agent, input: "user_message")
        _ = try await consume(result)

        #expect(result.finalOutput as? String == "done")
        #expect(result.rawResponses.count == 2)
        #expect(try result.toInputList().count == 5)
    }

    @Test func testStreamedParallelToolCallWithCancelledSiblingReachesFinalOutput() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [
            try FunctionTool.create(name: "ok_tool") { _ in "ok" },
            try FunctionTool.create(name: "cancel_tool") { _ in throw CancellationError() },
        ], model: .instance(model))
        try model.extend([
            .output([
                getFunctionToolCall("ok_tool", arguments: "{}", callId: "call_ok"),
                getFunctionToolCall("cancel_tool", arguments: "{}", callId: "call_cancel"),
            ]),
            .output([getTextMessage("final answer")]),
        ])

        let result = try Runner.runStreamed(agent, input: "user_message")
        _ = try await consume(result)

        #expect(result.finalOutput as? String == "final answer")
        let outputs = (model.calls.last?.input.inputItems ?? []).filter { $0.string("type") == "function_call_output" }
        #expect(outputs == [
            ["call_id": "call_ok", "output": "ok", "type": "function_call_output"],
            ["call_id": "call_cancel", "output": "An error occurred while running the tool. Please try again.", "type": "function_call_output"],
        ])
    }

    @Test func testStreamedReasoningItemIdPolicyOmitsFollowUpReasoningIds() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "foo", returnValue: "tool_result")], model: .instance(model))
        try model.extend([
            .output([
                ["id": "rs_first", "type": "reasoning", "summary": [["text": "Thinking...", "type": "summary_text"]]],
                getFunctionToolCall("foo", arguments: jsonArgs(["a": "b"]), callId: "call_first"),
            ]),
            .output([getTextMessage("done")]),
        ])

        let result = try Runner.runStreamed(agent, input: "hello", runConfig: RunConfig(reasoningItemIdPolicy: .omit))
        _ = try await consume(result)

        #expect(result.finalOutput as? String == "done")
        let reasoning = try #require(model.calls.last?.input.inputItems.first { $0.string("type") == "reasoning" })
        #expect(reasoning["id"] == nil)
        let history = try #require(try result.toInputList().first { $0.string("type") == "reasoning" })
        #expect(history["id"] == nil)
    }

    @Test func testStreamedRunAgainPersistsToolItemsToSession() async throws {
        let model = try ScriptedModel()
        let callId = "call-session-run-again"
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "foo", returnValue: "tool_result")], model: .instance(model))
        let session = SimpleListSession()
        try model.extend([
            .output([getFunctionToolCall("foo", arguments: jsonArgs(["a": "b"]), callId: callId)]),
            .output([getTextMessage("done")]),
        ])

        _ = try await consume(try Runner.runStreamed(agent, input: "user_message", session: session))

        let saved = try await session.getItems(limit: nil)
        #expect(saved.contains { $0.string("type") == "function_call" && $0.string("call_id") == callId })
        #expect(saved.contains { $0.string("type") == "function_call_output" && $0.string("call_id") == callId })
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

        let result = try Runner.runStreamed(agent3, input: "user_message")
        _ = try await consume(result)

        #expect(result.finalOutput as? String == "done")
        #expect(result.rawResponses.count == 3)
        #expect(try result.toInputList().count == 7)
        // Swift: see AgentRunnerTests.testHandoffs for why either field-identical agent is accepted.
        #expect(result.lastAgent === agent1 || result.lastAgent === agent2)
    }

    private struct Foo: Codable, Sendable, Equatable {
        var bar: String
    }

    @Test func testStructuredOutput() async throws {
        let model = try ScriptedModel()
        let agent1 = Agent(name: "test", tools: [getFunctionTool(name: "bar", returnValue: "bar_result")], model: .instance(model), outputType: try AgentOutputSchema(Foo.self))
        let agent2 = Agent(name: "test", tools: [getFunctionTool(name: "foo", returnValue: "foo_result")], handoffs: [.agent(agent1)], model: .instance(model))
        try model.extend([
            .output([getFunctionToolCall("foo", arguments: jsonArgs(["bar": "baz"]), callId: "call_foo")]),
            .output([getTextMessage("a_message"), getHandoffToolCall(agent1)]),
            .output([getTextMessage(jsonArgs(["bar": "preamble"])), getFunctionToolCall("bar", arguments: jsonArgs(["bar": "baz"]), callId: "call_bar")]),
            .output([getFinalOutputMessage(jsonArgs(["bar": "baz"]))]),
        ])

        let result = try Runner.runStreamed(
            agent2,
            input: .items([getTextInputItem("user_message"), getTextInputItem("another_message")]),
            runConfig: RunConfig(nestHandoffHistory: true)
        )
        _ = try await consume(result)

        #expect(result.finalOutput as? Foo == Foo(bar: "baz"))
        #expect(result.rawResponses.count == 4)
        #expect(try result.toInputList().count == 11)
        #expect(try result.toInputList(mode: .normalized).count == 7)
        #expect(result.lastAgent === agent1)
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

        let result = try Runner.runStreamed(agent2, input: "user_message")
        _ = try await consume(result)

        #expect(result.finalOutput as? String == "last")
        #expect(result.rawResponses.count == 2)
        #expect(try result.toInputList().count == 2)
    }
}
