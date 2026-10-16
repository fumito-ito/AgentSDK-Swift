import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_runner.py (output guardrails with sessions, tool use behavior,
// model settings, and server-managed conversations).
//
// Swift: `test_session_persists_only_new_step_items` monkeypatches runner internals and is
// skipped. The tripwire variant of `test_resumed_final_tool_persists_call_and_output_after_output_guardrail`
// needs blocked-output replacement, which is not ported. `run_sync` variants run asynchronously.
extension AgentRunnerTests {
    private func types(_ items: [TResponseInputItem]) -> [String?] {
        items.map { $0.string("type") ?? $0.string("role") }
    }

    private func drainEvents(_ result: RunResultStreaming) async throws {
        for try await _ in result.streamEvents() {}
    }

    // MARK: - Output guardrails and sessions

    @Test func testOutputGuardrailTripwireTriggeredCausesException() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", model: .instance(model), outputGuardrails: [OutputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: true) }])
        try model.enqueue(.output([getTextMessage("user_message")]))

        await expectErrorAsync(OutputGuardrailTripwireTriggered.self) {
            _ = try await Runner.run(agent, input: "user_message")
        }
    }

    @Test func testOutputGuardrailTripwireDoesNotSaveAssistantMessageToSession() async throws {
        let session = SimpleListSession()
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("should_not_be_saved")]))
        let agent = Agent(name: "test", model: .instance(model), outputGuardrails: [OutputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: true) }])

        await expectErrorAsync(OutputGuardrailTripwireTriggered.self) {
            _ = try await Runner.run(agent, input: "user_message", session: session)
        }

        #expect(types(try await session.getItems(limit: nil)) == ["user"])
    }

    @Test(arguments: [false, true])
    func testOutputGuardrailErrorPreservesFinalOutputInSession(streamed: Bool) async throws {
        let session = SimpleListSession()
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("preserved_on_guardrail_error")]))
        let agent = Agent(name: "test", model: .instance(model), outputGuardrails: [OutputGuardrail { _, _, _ in throw TestError("guardrail failed") }])

        await expectErrorAsync(TestError.self, containing: "guardrail failed") {
            if streamed {
                try await drainEvents(try Runner.runStreamed(agent, input: "user_message", session: session))
            } else {
                _ = try await Runner.run(agent, input: "user_message", session: session)
            }
        }

        #expect(types(try await session.getItems(limit: nil)) == ["user", "message"])
    }

    @Test func testOutputGuardrailCancellationPreservesFinalOutputInSession() async throws {
        let guardrailStarted = AsyncEvent()
        let session = SimpleListSession()
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("preserved_on_guardrail_cancellation")]))
        let guardrail = OutputGuardrail { _, _, _ in
            guardrailStarted.set()
            try await waitForever()
            fatalError("unreachable")
        }
        let agent = Agent(name: "test", model: .instance(model), outputGuardrails: [guardrail])

        let runTask = Task { try await Runner.run(agent, input: "user_message", session: session) }
        try await guardrailStarted.wait(timeout: 5)
        runTask.cancel()

        await expectErrorAsync(CancellationError.self) { _ = try await runTask.value }
        #expect(types(try await session.getItems(limit: nil)) == ["user", "message"])
    }

    @Test func testResumedFinalOutputPersistsOnceAfterPassingOutputGuardrail() async throws {
        struct FooArgs: Codable, Sendable { var a: String }
        let foo = try FunctionTool.create(name: "foo", argumentsType: FooArgs.self) { _, args in "result:\(args.a)" }
        let session = SimpleListSession()
        let model = try ScriptedModel()
        try model.enqueue(.output([getFunctionToolCall("foo", arguments: #"{"a": "b"}"#)]))
        let agent = Agent(
            name: "test",
            tools: [foo],
            model: .instance(model),
            outputGuardrails: [OutputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: false) }]
        )

        let streamed = try Runner.runStreamed(agent, input: "user_message", session: session)
        for try await event in streamed.streamEvents() {
            if case .runItem(let itemEvent) = event, itemEvent.name == "tool_output" {
                streamed.cancel(mode: .afterTurn)
            }
        }
        let itemsBeforeResume = try await session.getItems(limit: nil)
        let state = streamed.toState()
        state.currentTurnPersistedItemCount = 2

        try model.enqueue(.output([getTextMessage("accepted_final")]))
        let resumed = try await Runner.run(agent, state: state, session: session)
        #expect(resumed.finalOutput as? String == "accepted_final")

        let itemsAfterResume = try await session.getItems(limit: nil)
        #expect(Array(itemsAfterResume.prefix(itemsBeforeResume.count)) == itemsBeforeResume)
        let assistantMessages = itemsAfterResume.filter { $0.string("role") == "assistant" && $0.string("type") == "message" }
        #expect(assistantMessages.count == 1)
        let content = assistantMessages.first?["content"]?.arrayValue ?? []
        #expect(content.contains { $0.objectValue?.string("text") == "accepted_final" })
    }

    @Test func testResumedFinalToolPersistsCallAndOutputAfterOutputGuardrail() async throws {
        let commitTool = try FunctionTool.create(name: "commit_tool") { _ in "committed-result" }
        let session = SimpleListSession()
        let model = try ScriptedModel()
        try model.enqueue(.output([getFunctionToolCall("commit_tool", arguments: "{}", callId: "call-first")]))
        let agent = Agent(
            name: "test",
            tools: [commitTool],
            model: .instance(model),
            outputGuardrails: [OutputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: false) }]
        )

        let streamed = try Runner.runStreamed(agent, input: "user_message", session: session)
        for try await event in streamed.streamEvents() {
            if case .runItem(let itemEvent) = event, itemEvent.name == "tool_output" {
                streamed.cancel(mode: .afterTurn)
            }
        }
        let state = streamed.toState()
        #expect(state.currentTurnPersistedItemCount == 2)

        agent.toolUseBehavior = .stopOnFirstTool
        try model.enqueue(.output([getFunctionToolCall("commit_tool", arguments: "{}", callId: "call-second")]))
        let result = try await Runner.run(agent, state: state, session: session)
        #expect(result.finalOutput as? String == "committed-result")

        #expect(state.currentTurnPersistedItemCount == 2)
        let items = try await session.getItems(limit: nil)
        #expect(items.map { "\($0.string("type") ?? $0.string("role") ?? "None")|\($0.string("call_id") ?? "None")" } == [
            "user|None",
            "function_call|call-first",
            "function_call_output|call-first",
            "function_call|call-second",
            "function_call_output|call-second",
        ])
        #expect(items.last?.string("output") == "committed-result")
    }

    @Test func testInputGuardrailNoTripwireContinuesExecution() async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("response")]))
        let agent = Agent(name: "test", model: .instance(model), inputGuardrails: [InputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: false) }])

        let result = try await Runner.run(agent, input: "user_message")
        #expect(result.finalOutput as? String == "response")
    }

    @Test func testOutputGuardrailNoTripwireContinuesExecution() async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("response")]))
        let agent = Agent(name: "test", model: .instance(model), outputGuardrails: [OutputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: false) }])

        let result = try await Runner.run(agent, input: "user_message")
        #expect(result.finalOutput as? String == "response")
    }

    // MARK: - Tool use behavior and settings

    private struct FooOutput: Codable, Sendable, Equatable {
        var bar: String
    }

    private func toolOne() throws -> FunctionTool {
        try FunctionTool.create(name: "test_tool_one") { _ in FooOutput(bar: "tool_one_result") }
    }

    private func toolTwo() throws -> FunctionTool {
        try FunctionTool.create(name: "test_tool_two") { _ in "tool_two_result" }
    }

    @Test func testToolUseBehaviorFirstOutput() async throws {
        let model = try ScriptedModel()
        let agent = Agent(
            name: "test",
            tools: [getFunctionTool(name: "foo", returnValue: "tool_result"), try toolOne(), try toolTwo()],
            model: .instance(model),
            outputType: try AgentOutputSchema(FooOutput.self),
            toolUseBehavior: .stopOnFirstTool
        )
        let firstTurn: [TResponseOutputItem] = [
            getTextMessage("a_message"),
            getFunctionToolCall("test_tool_one", callId: "tool-one"),
            getFunctionToolCall("test_tool_two", callId: "tool-two"),
        ]
        try model.extend([.output(firstTurn)])

        let result = try await Runner.run(agent, input: "user_message")

        #expect(result.finalOutput as? FooOutput == FooOutput(bar: "tool_one_result"), "should have used the first tool result")
    }

    @Test func testToolUseBehaviorCustomFunction() async throws {
        let model = try ScriptedModel()
        let agent = Agent(
            name: "test",
            tools: [getFunctionTool(name: "foo", returnValue: "tool_result"), try toolOne(), try toolTwo()],
            model: .instance(model),
            toolUseBehavior: .custom { _, results in
                results.map(\.tool.name).contains("test_tool_one")
                    ? ToolsToFinalOutputResult(isFinalOutput: true, finalOutput: "the_final_output")
                    : ToolsToFinalOutputResult(isFinalOutput: false)
            }
        )
        try model.extend([
            .output([getTextMessage("a_message"), getFunctionToolCall("test_tool_two", callId: "call-tool-two-first")]),
            .output([
                getTextMessage("a_message"),
                getFunctionToolCall("test_tool_one", callId: "call-tool-one"),
                getFunctionToolCall("test_tool_two", callId: "call-tool-two-second"),
            ]),
        ])

        let result = try await Runner.run(agent, input: "user_message")

        #expect(result.rawResponses.count == 2, "should have two model responses")
        #expect(result.finalOutput as? String == "the_final_output", "should have used the custom function")
    }

    @Test func testModelSettingsOverride() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", model: .instance(model), modelSettings: ModelSettings(temperature: 1.0, maxTokens: 1000))
        try model.extend([.output([getTextMessage("a_message")])])

        _ = try await Runner.run(agent, input: "user_message", runConfig: RunConfig(modelSettings: ModelSettings(temperature: 0.5)))

        #expect(model.calls.last?.modelSettings.temperature == 0.5)
        #expect(model.calls.last?.modelSettings.maxTokens == 1000)
    }

    // MARK: - Server-managed conversations

    @Test func testPreviousResponseIdPassedBetweenRuns() async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("done")]))
        let agent = Agent(name: "test", model: .instance(model))

        #expect(model.calls.isEmpty)
        _ = try await Runner.run(agent, input: "test", previousResponseId: "resp-non-streamed-test")
        #expect(model.calls.last?.previousResponseId == "resp-non-streamed-test")
    }

    enum ServerManaged: String, CaseIterable, Sendable {
        case conversation, previousResponse, auto
    }

    @Test(arguments: ServerManaged.allCases)
    func testRunRejectsSessionWithServerManagedConversation(mode: ServerManaged) async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("done")]))
        let agent = Agent(name: "test", model: .instance(model))

        await expectErrorAsync(UserError.self, containing: "Session persistence") {
            _ = try await Runner.run(
                agent,
                input: "test",
                previousResponseId: mode == .previousResponse ? "resp-test" : nil,
                autoPreviousResponseId: mode == .auto,
                conversationId: mode == .conversation ? "conv-test" : nil,
                session: SimpleListSession()
            )
        }
    }

    @Test(arguments: ServerManaged.allCases)
    func testRunStreamedRejectsSessionWithServerManagedConversation(mode: ServerManaged) throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("done")]))
        let agent = Agent(name: "test", model: .instance(model))

        expectError(UserError.self, containing: "Session persistence") {
            _ = try Runner.runStreamed(
                agent,
                input: "test",
                previousResponseId: mode == .previousResponse ? "resp-test" : nil,
                autoPreviousResponseId: mode == .auto,
                conversationId: mode == .conversation ? "conv-test" : nil,
                session: SimpleListSession()
            )
        }
    }

    @Test func testRunRejectsSessionWithResumedConversationState() async throws {
        let agent = Agent(name: "test", model: .instance(try ScriptedModel()))
        let state = RunState(context: RunContextWrapper(), originalInput: "hello", startingAgent: agent, conversationId: "conv-test")

        await expectErrorAsync(UserError.self, containing: "Session persistence") {
            _ = try await Runner.run(agent, state: state, session: SimpleListSession())
        }
        expectError(UserError.self, containing: "Session persistence") {
            _ = try Runner.runStreamed(agent, state: state, session: SimpleListSession())
        }
    }

    @Test(arguments: [false, true])
    func testMultiTurnPreviousResponseIdPassedBetweenRuns(streamed: Bool) async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "foo", returnValue: "tool_result")], model: .instance(model))
        try model.extend([
            .output([getTextMessage("a_message"), getFunctionToolCall("foo", arguments: #"{"a": "b"}"#)]),
            .output([getTextMessage("done")]),
        ])

        if streamed {
            try await drainEvents(try Runner.runStreamed(agent, input: "test", previousResponseId: "resp-stream-test"))
        } else {
            _ = try await Runner.run(agent, input: "test", previousResponseId: "resp-test-123")
        }

        #expect(model.calls.last?.previousResponseId == "resp-789")
    }

    @Test func testPreviousResponseIdPassedBetweenRunsStreamed() async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("done")]))
        let agent = Agent(name: "test", model: .instance(model))

        try await drainEvents(try Runner.runStreamed(agent, input: "test", previousResponseId: "resp-stream-test"))

        #expect(model.calls.last?.previousResponseId == "resp-stream-test")
    }

    @Test(arguments: [false, true])
    func testConversationIdOnlySendsNewItemsMultiTurn(streamed: Bool) async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "test_func", returnValue: "tool_result")], model: .instance(model))
        try model.extend([
            .output([getTextMessage("a_message"), getFunctionToolCall("test_func", arguments: #"{"arg": "foo"}"#, callId: "call-test-func-first")]),
            .output([getTextMessage("b_message"), getFunctionToolCall("test_func", arguments: #"{"arg": "bar"}"#, callId: "call-test-func-second")]),
            .output([getTextMessage("done")]),
        ])

        let finalOutput: (any Sendable)?
        if streamed {
            let result = try Runner.runStreamed(agent, input: "user_message", conversationId: "conv-test-123")
            try await drainEvents(result)
            finalOutput = result.finalOutput
        } else {
            finalOutput = try await Runner.run(agent, input: "user_message", conversationId: "conv-test-123").finalOutput
        }
        #expect(finalOutput as? String == "done")

        let firstInput = try #require(model.calls.first?.input.inputItems)
        #expect(firstInput.count == 1)
        #expect(firstInput.first?.string("role") == "user")
        #expect(firstInput.first?.string("content") == "user_message")
        let lastInput = try #require(model.calls.last?.input.inputItems)
        #expect(lastInput.count == 1)
        #expect(lastInput.first?.string("type") == "function_call_output")
        #expect(lastInput.first?.string("call_id") != nil)
    }

    @Test(arguments: [false, true])
    func testPreviousResponseIdOnlySendsNewItemsMultiTurn(streamed: Bool) async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "test_func", returnValue: "tool_result")], model: .instance(model))
        try model.extend([
            .output([getTextMessage("a_message"), getFunctionToolCall("test_func", arguments: #"{"arg": "foo"}"#)]),
            .output([getTextMessage("done")]),
        ])

        if streamed {
            try await drainEvents(try Runner.runStreamed(agent, input: "user_message", previousResponseId: "initial-response-123"))
        } else {
            let result = try await Runner.run(agent, input: "user_message", previousResponseId: "initial-response-123")
            #expect(result.finalOutput as? String == "done")
        }

        let firstInput = try #require(model.calls.first?.input.inputItems)
        #expect(firstInput.count == 1)
        #expect(firstInput.first?.string("content") == "user_message")
        let lastInput = try #require(model.calls.last?.input.inputItems)
        #expect(lastInput.count == 1)
        #expect(lastInput.first?.string("type") == "function_call_output")
        #expect(model.calls.last?.previousResponseId == "resp-789")
    }

    @Test(arguments: [false, true])
    func testPreviousResponseIdRetryDoesNotResendInitialInputMultiTurn(streamed: Bool) async throws {
        let model = try ScriptedModel()
        let agent = Agent(
            name: "test",
            tools: [getFunctionTool(name: "test_func", returnValue: "tool_result")],
            model: .instance(model),
            modelSettings: ModelSettings(retry: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError()))
        )
        try model.extend([
            .raiseError(
                OpenAIAPIError(message: "connection error", isConnectionError: true),
                retryAdvice: ModelRetryAdvice(suggested: true, replaySafety: "safe")
            ),
            .output([getTextMessage("a_message"), getFunctionToolCall("test_func", arguments: #"{"arg": "foo"}"#)]),
            .output([getTextMessage("done")]),
        ])

        if streamed {
            let result = try Runner.runStreamed(agent, input: "user_message", previousResponseId: "initial-response-123")
            try await drainEvents(result)
            #expect(result.finalOutput as? String == "done")
        } else {
            let result = try await Runner.run(agent, input: "user_message", previousResponseId: "initial-response-123")
            #expect(result.finalOutput as? String == "done")
        }

        let lastInput = try #require(model.calls.last?.input.inputItems)
        #expect(lastInput.count == 1)
        #expect(lastInput.first?.string("type") == "function_call_output")
    }
}
