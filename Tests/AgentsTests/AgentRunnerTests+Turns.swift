import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_runner.py (response chaining, input replay, dynamic tools, and
// missing-tool behavior).
extension AgentRunnerTests {
    private func runPossiblyStreamed(
        _ agent: Agent,
        input: RunInput,
        streamed: Bool,
        runConfig: RunConfig? = nil,
        previousResponseId: String? = nil,
        autoPreviousResponseId: Bool = false,
        session: (any Session)? = nil
    ) async throws -> (any Sendable)? {
        if streamed {
            let result = try Runner.runStreamed(
                agent, input: input, runConfig: runConfig, previousResponseId: previousResponseId,
                autoPreviousResponseId: autoPreviousResponseId, session: session
            )
            for try await _ in result.streamEvents() {}
            return result.finalOutput
        }
        return try await Runner.run(
            agent, input: input, runConfig: runConfig, previousResponseId: previousResponseId,
            autoPreviousResponseId: autoPreviousResponseId, session: session
        ).finalOutput
    }

    private func testFuncAgent(_ model: ScriptedModel) -> Agent {
        Agent(name: "test", tools: [getFunctionTool(name: "test_func", returnValue: "tool_result")], model: .instance(model))
    }

    // MARK: - Chaining

    @Test func testAutoPreviousResponseIdRetriesWhenPolicyApprovesUnsafeReplay() async throws {
        let seen = SharedBox<[RetryPolicyContext]>([])
        let policy = RetryPolicy { context in
            seen.mutate { $0.append(context) }
            return RetryDecision(retry: true, approveUnsafeReplay: true)
        }
        let model = try ScriptedModel()
        try model.extend([
            .output([getFunctionToolCall("test_func", arguments: #"{"arg": "foo"}"#)]),
            .raiseError(
                OpenAIAPIError(message: "connection closed after response processing started", isConnectionError: true),
                retryAdvice: ModelRetryAdvice(suggested: false, replaySafety: "unsafe", responseStarted: true)
            ),
            .output([getTextMessage("done")]),
        ])
        let agent = Agent(
            name: "test",
            tools: [getFunctionTool(name: "test_func", returnValue: "tool_result")],
            model: .instance(model),
            modelSettings: ModelSettings(retry: ModelRetrySettings(maxRetries: 1, policy: policy))
        )

        let result = try await Runner.run(agent, input: "user_message", autoPreviousResponseId: true)

        #expect(result.finalOutput as? String == "done")
        #expect(seen.value.count == 1)
        let context = try #require(seen.value.first)
        #expect(context.previousResponseId == "resp-789")
        #expect(context.conversationId == nil)
        #expect(context.statefulRequest)
        #expect(context.responseStarted)
        #expect(context.replaySafety == "unsafe")
    }

    @Test(arguments: [false, true])
    func testDefaultSendAllItems(streamed: Bool) async throws {
        let model = try ScriptedModel()
        try model.extend([
            .output([getTextMessage("a_message"), getFunctionToolCall("test_func", arguments: #"{"arg": "foo"}"#)]),
            .output([getTextMessage("done")]),
        ])

        let finalOutput = try await runPossiblyStreamed(testFuncAgent(model), input: "user_message", streamed: streamed)
        #expect(finalOutput as? String == "done")

        let lastInput = try #require(model.calls.last?.input.inputItems)
        #expect(lastInput.count == 4)
        #expect(lastInput[0].string("role") == "user")
        #expect(lastInput[0].string("content") == "user_message")
        #expect(lastInput[1].string("role") == "assistant")
        #expect(lastInput[2].string("name") == "test_func")
        #expect(lastInput[2].string("arguments") == #"{"arg": "foo"}"#)
        #expect(lastInput[3].string("type") == "function_call_output")
        #expect(lastInput[3].string("call_id") != nil)
    }

    @Test(arguments: [false, true])
    func testDefaultMultiTurnDropsOrphanHostedShellCalls(streamed: Bool) async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "hosted-shell", tools: [try ShellTool(environment: ["type": "container_auto"])], model: .instance(model))
        let shellTurn = [makeShellCall("call_shell_1", idValue: "shell_1", commands: ["echo hi"])]
        try model.extend([
            streamed ? getExactOutputStreamStep(shellTurn) : .output(shellTurn),
            .output([getTextMessage("done")]),
        ])

        let finalOutput = try await runPossiblyStreamed(agent, input: "user_message", streamed: streamed)
        #expect(finalOutput as? String == "done")

        let lastInput = try #require(model.calls.last?.input.inputItems)
        #expect(lastInput.count == 1)
        #expect(!lastInput.contains { $0.string("type") == "shell_call" })
        #expect(lastInput.first?.string("role") == "user")
        #expect(lastInput.first?.string("content") == "user_message")
    }

    @Test(arguments: [false, true], [false, true])
    func testManualPendingShellCallInputIsPreserved(streamed: Bool, withSession: Bool) async throws {
        let model = try ScriptedModel()
        let pendingShellCall = makeShellCall("manual_shell", idValue: "shell_1", commands: ["echo hi"])
        try model.extend([
            .output([getFunctionToolCall("test_func", arguments: #"{"arg": "foo"}"#)]),
            .output([getTextMessage("done")]),
        ])

        let finalOutput = try await runPossiblyStreamed(
            testFuncAgent(model), input: .items([pendingShellCall]), streamed: streamed,
            session: withSession ? SimpleListSession() : nil
        )
        #expect(finalOutput as? String == "done")

        let isManualShell: (TResponseInputItem) -> Bool = { $0.string("type") == "shell_call" && $0.string("call_id") == "manual_shell" }
        #expect(model.calls.first?.input.inputItems.contains(where: isManualShell) == true)
        #expect(model.calls.last?.input.inputItems.contains(where: isManualShell) == true)
    }

    @Test(arguments: [false, true])
    func testAutoPreviousResponseIdMultiTurn(streamed: Bool) async throws {
        let model = try ScriptedModel()
        try model.extend([
            .output([getTextMessage("a_message"), getFunctionToolCall("test_func", arguments: #"{"arg": "foo"}"#)]),
            .output([getTextMessage("done")]),
        ])

        let finalOutput = try await runPossiblyStreamed(testFuncAgent(model), input: "user_message", streamed: streamed, autoPreviousResponseId: true)
        #expect(finalOutput as? String == "done")

        let firstInput = try #require(model.calls.first?.input.inputItems)
        #expect(firstInput.count == 1)
        #expect(firstInput.first?.string("role") == "user")
        #expect(firstInput.first?.string("content") == "user_message")
        #expect(model.calls.first?.previousResponseId == nil)
        let lastInput = try #require(model.calls.last?.input.inputItems)
        #expect(lastInput.count == 1)
        #expect(lastInput.first?.string("type") == "function_call_output")
        #expect(model.calls.last?.previousResponseId == "resp-789")
    }

    @Test func testWithoutPreviousResponseIdAndAutoPreviousResponseIdNoChaining() async throws {
        let model = try ScriptedModel()
        try model.extend([
            .output([getTextMessage("a_message"), getFunctionToolCall("test_func", arguments: #"{"arg": "foo"}"#)]),
            .output([getTextMessage("done")]),
        ])

        let result = try await Runner.run(testFuncAgent(model), input: "user_message")
        #expect(result.finalOutput as? String == "done")

        #expect(model.calls.first?.input.inputItems.count == 1)
        #expect(model.calls.first?.previousResponseId == nil)
        #expect(model.calls.last?.input.inputItems.count == 4)
        #expect(model.calls.last?.previousResponseId == nil)
    }

    // MARK: - Tools

    @Test func testDynamicToolAdditionRun() async throws {
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

        let result = try await Runner.run(agent, input: "start")

        #expect(executed.value)
        #expect(result.finalOutput as? String == "done")
    }

    private func functionOutputs(_ model: ScriptedModel) -> [JSONObject] {
        (model.calls.last?.input.inputItems ?? []).filter { $0.string("type") == "function_call_output" }
    }

    @Test func testToolNotFoundBehaviorReturnsErrorToModel() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", model: .instance(model), toolUseBehavior: .runLLMAgain)
        try model.extend([
            .output([getFunctionToolCall("missing_tool", arguments: "{}", callId: "call_missing")]),
            .output([getTextMessage("recovered")]),
        ])

        let result = try await Runner.run(agent, input: "start", runConfig: RunConfig(toolNotFoundBehavior: .returnErrorToModel))

        #expect(result.finalOutput as? String == "recovered")
        #expect(functionOutputs(model) == [["call_id": "call_missing", "output": "Tool 'missing_tool' not found.", "type": "function_call_output"]])
    }

    @Test func testToolNotFoundBehaviorUsesToolErrorFormatter() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", model: .instance(model), toolUseBehavior: .runLLMAgain)
        try model.extend([
            .output([getFunctionToolCall("missing_tool", arguments: "{}", callId: "call_missing")]),
            .output([getTextMessage("recovered")]),
        ])
        let seenKinds = SharedBox<[String]>([])
        let formatter: ToolErrorFormatter = { args in
            seenKinds.mutate { $0.append(args.kind) }
            guard args.kind == "tool_not_found" else { return nil }
            return "\(args.toolName) unavailable for \(args.callId)"
        }

        let result = try await Runner.run(
            agent,
            input: "start",
            runConfig: RunConfig(toolErrorFormatter: formatter, toolNotFoundBehavior: .returnErrorToModel)
        )

        #expect(result.finalOutput as? String == "recovered")
        #expect(seenKinds.value == ["tool_not_found"])
        #expect(functionOutputs(model) == [["call_id": "call_missing", "output": "missing_tool unavailable for call_missing", "type": "function_call_output"]])
    }

    @Test func testToolNotFoundBehaviorHandlesMixedFunctionToolCalls() async throws {
        let model = try ScriptedModel()
        let calls = SharedBox<[String]>([])
        let knownTool = try FunctionTool.create(name: "known_tool") { _ in
            calls.mutate { $0.append("known_tool") }
            return "known result"
        }
        let agent = Agent(name: "test", tools: [knownTool], model: .instance(model), toolUseBehavior: .runLLMAgain)
        try model.extend([
            .output([
                getFunctionToolCall("missing_tool", arguments: "{}", callId: "call_missing"),
                getFunctionToolCall("known_tool", arguments: "{}", callId: "call_known"),
            ]),
            .output([getTextMessage("done")]),
        ])

        let result = try await Runner.run(agent, input: "start", runConfig: RunConfig(toolNotFoundBehavior: .returnErrorToModel))

        #expect(calls.value == ["known_tool"])
        #expect(result.finalOutput as? String == "done")
        var outputs: [String: String] = [:]
        for item in functionOutputs(model) {
            if let callId = item.string("call_id") { outputs[callId] = item.string("output") }
        }
        #expect(outputs == ["call_known": "known result", "call_missing": "Tool 'missing_tool' not found."])
    }
}
