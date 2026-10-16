import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_hooks.py.
//
// Swift: `test_sync_run_hook_with_llm` is skipped (no `Runner.run_sync`), and
// `test_runner_run_rejects_agent_hooks`, `test_runner_run_streamed_rejects_agent_hooks`, and
// `test_validate_run_hooks_rejects_non_hook_objects` are skipped because the `hooks` parameter is
// typed as `RunHooks`, so other objects cannot be passed.
@Suite struct RunHooksTests {
    private func makeAgent() throws -> (Agent, ScriptedModel) {
        let model = try ScriptedModel()
        let agent = Agent(name: "A", tools: [getFunctionTool(name: "f", returnValue: "res")], handoffs: [], model: .instance(model))
        return (agent, model)
    }

    // MARK: - LLM hooks

    @Test func testAsyncRunHooksWithLLM() async throws {
        let hooks = RunHooksForTests()
        let (agent, model) = try makeAgent()
        try model.enqueue(.output([getTextMessage("hello")]))

        _ = try await Runner.run(agent, input: "hello", hooks: hooks)

        #expect(hooks.events == ["on_agent_start": 1, "on_llm_start": 1, "on_llm_end": 1, "on_agent_end": 1])
    }

    @Test func testStreamedRunHooksWithLLM() async throws {
        let hooks = RunHooksForTests()
        let (agent, model) = try makeAgent()
        try model.enqueue(.output([getTextMessage("hello")]))

        let stream = try Runner.runStreamed(agent, input: "hello", hooks: hooks)
        for try await _ in stream.streamEvents() {}

        #expect(hooks.events == ["on_agent_start": 1, "on_llm_start": 1, "on_llm_end": 1, "on_agent_end": 1])
    }

    @Test func testAsyncRunHooksWithAgentHooksWithLLM() async throws {
        let hooks = RunHooksForTests()
        let agentHooks = AgentHooksForTests()
        let model = try ScriptedModel()
        let agent = Agent(name: "A", tools: [getFunctionTool(name: "f", returnValue: "res")], handoffs: [], model: .instance(model), hooks: agentHooks)
        try model.enqueue(.output([getTextMessage("hello")]))

        _ = try await Runner.run(agent, input: "hello", hooks: hooks)

        #expect(hooks.events == ["on_agent_start": 1, "on_llm_start": 1, "on_llm_end": 1, "on_agent_end": 1])
        #expect(agentHooks.events == ["on_start": 1, "on_llm_start": 1, "on_llm_end": 1, "on_end": 1])
    }

    @Test func testRunHooksLLMErrorNonStreaming() async throws {
        // Swift: the scripted model raises instead of monkeypatching `get_response`.
        let hooks = RunHooksForTests()
        let (agent, model) = try makeAgent()
        try model.enqueue(.raiseError(TestError("boom")))

        await expectErrorAsync(TestError.self, containing: "boom") {
            _ = try await Runner.run(agent, input: "hello", hooks: hooks)
        }

        // Current behavior is that hooks will not fire on LLM failure
        #expect(hooks.events["on_agent_start"] == 1)
        #expect(hooks.events["on_llm_start"] == 1)
        #expect(hooks.events["on_llm_end"] == nil)
        #expect(hooks.events["on_agent_end"] == nil)
    }

    @Test func testStreamedRunHooksLLMError() async throws {
        let hooks = RunHooksForTests()
        let failingStream: ModelStep = .stream { _ in
            makeStream { continuation in
                continuation.yield(["foo": "bar"])
                throw TestError("stream blew up")
            }
        }
        let model = try ScriptedModel(steps: [failingStream])
        let agent = Agent(name: "A", tools: [getFunctionTool(name: "f", returnValue: "res")], handoffs: [], model: .instance(model))

        let stream = try Runner.runStreamed(agent, input: "hello", hooks: hooks)

        // Consuming the stream should surface the exception
        await expectErrorAsync(TestError.self, containing: "stream blew up") {
            for try await _ in stream.streamEvents() {}
        }

        // Current behavior: success-only on_llm_end; ensure starts fired but ends did not.
        #expect(hooks.events["on_agent_start"] == 1)
        #expect(hooks.events["on_llm_start"] == 1)
        #expect(hooks.events["on_llm_end"] == nil)
        #expect(hooks.events["on_agent_end"] == nil)
    }

    // MARK: - Turn input

    private final class RunHooksWithTurnInput: RunHooks, @unchecked Sendable {
        let capture = TurnInputCapture()

        override func onAgentStart(context: AgentHookContext, agent: AgentBase) async throws {
            capture.append(context.turnInput)
        }
    }

    @Test func testRunHooksReceivesTurnInputString() async throws {
        let hooks = RunHooksWithTurnInput()
        let model = try ScriptedModel(steps: [.output([getTextMessage("response")])])
        let agent = Agent(name: "test", model: .instance(model))

        _ = try await Runner.run(agent, input: "hello world", hooks: hooks)

        let turnInputs = hooks.capture.capturedTurnInputs
        #expect(turnInputs.count == 1)
        let turnInput = turnInputs[0]
        #expect(turnInput.count == 1)
        #expect(turnInput[0].string("content") == "hello world")
        #expect(turnInput[0].string("role") == "user")
    }

    @Test func testRunHooksReceivesTurnInputList() async throws {
        let hooks = RunHooksWithTurnInput()
        let model = try ScriptedModel(steps: [.output([getTextMessage("response")])])
        let agent = Agent(name: "test", model: .instance(model))
        let inputItems: [TResponseInputItem] = [
            ["role": "user", "content": "first message"],
            ["role": "user", "content": "second message"],
        ]

        _ = try await Runner.run(agent, input: .items(inputItems), hooks: hooks)

        let turnInputs = hooks.capture.capturedTurnInputs
        #expect(turnInputs.count == 1)
        #expect(turnInputs[0].map { $0.string("content") } == ["first message", "second message"])
    }

    @Test func testRunHooksReceivesTurnInputStreamed() async throws {
        let hooks = RunHooksWithTurnInput()
        let model = try ScriptedModel(steps: [.output([getTextMessage("response")])])
        let agent = Agent(name: "test", model: .instance(model))

        let result = try Runner.runStreamed(agent, input: "streamed input", hooks: hooks)
        for try await _ in result.streamEvents() {}

        let turnInputs = hooks.capture.capturedTurnInputs
        #expect(turnInputs.count == 1)
        #expect(turnInputs[0].count == 1)
        #expect(turnInputs[0][0].string("content") == "streamed input")
    }

    // MARK: - Tools and handoffs

    @Test func testRunHooksCountToolAndHandoffInvocations() async throws {
        let hooks = RunHooksForTests()
        let model = try ScriptedModel()
        let agent1 = Agent(name: "test_1", model: .instance(model))
        let agent2 = Agent(name: "test_2", tools: [getFunctionTool(name: "some_function", returnValue: "result")], handoffs: [.agent(agent1)], model: .instance(model))
        try model.extend([
            .output([getFunctionToolCall("some_function", arguments: #"{"a": "b"}"#)]),
            .output([getTextMessage("a_message"), getHandoffToolCall(agent1)]),
            .output([getTextMessage("done")]),
        ])

        _ = try await Runner.run(agent2, input: "user_message", hooks: hooks)

        #expect(hooks.events["on_tool_start"] == 1)
        #expect(hooks.events["on_tool_end"] == 1)
        #expect(hooks.events["on_handoff"] == 1)
        #expect(hooks.events["on_agent_start"] == 2)
        #expect(hooks.events["on_agent_end"] == 1)
        #expect(hooks.toolContextIds.count == 1)
    }

    @Test func testStreamedRunHooksCountToolAndHandoffInvocations() async throws {
        let hooks = RunHooksForTests()
        let model = try ScriptedModel()
        let agent1 = Agent(name: "test_1", model: .instance(model))
        let agent2 = Agent(name: "test_2", tools: [getFunctionTool(name: "some_function", returnValue: "result")], handoffs: [.agent(agent1)], model: .instance(model))
        try model.extend([
            .output([
                getFunctionToolCall("some_function", arguments: #"{"a": "b"}"#, callId: "call_1"),
                getFunctionToolCall("some_function", arguments: #"{"a": "b"}"#, callId: "call_2"),
            ]),
            .output([getTextMessage("a_message"), getHandoffToolCall(agent1, callId: "handoff_1")]),
            .output([getTextMessage("done")]),
        ])

        let stream = try Runner.runStreamed(agent2, input: "user_message", hooks: hooks)
        for try await _ in stream.streamEvents() {}

        #expect(hooks.events["on_tool_start"] == 2)
        #expect(hooks.events["on_tool_end"] == 2)
        #expect(hooks.events["on_handoff"] == 1)
        #expect(hooks.events["on_agent_start"] == 2)
        #expect(hooks.events["on_agent_end"] == 1)
        #expect(hooks.toolContextIds.count == 2)
    }

    private final class Metadata: Sendable, CustomStringConvertible {
        var description: String { #"{"status": "ok", "count": 1}"# }
    }

    private final class ResultBox: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: (any Sendable)?
        var result: (any Sendable)? {
            get { lock.locked { stored } }
            set { lock.locked { stored = newValue } }
        }
    }

    private final class RecordingRunHooks: RunHooks, @unchecked Sendable {
        let box = ResultBox()
        override func onToolEnd(context: RunContextWrapper, agent: AgentBase, tool: any Tool, result: any Sendable) async throws {
            box.result = result
        }
    }

    private final class RecordingAgentHooks: AgentHooks, @unchecked Sendable {
        let box = ResultBox()
        override func onToolEnd(context: RunContextWrapper, agent: AgentBase, tool: any Tool, result: any Sendable) async throws {
            box.result = result
        }
    }

    @Test func testToolEndHooksReceiveRawFunctionToolResult() async throws {
        let metadataResult = Metadata()
        let getMetadata = try FunctionTool.create(name: "get_metadata") { _ in metadataResult }
        let runHooks = RecordingRunHooks()
        let agentHooks = RecordingAgentHooks()
        let model = try ScriptedModel(steps: [
            .output([getFunctionToolCall("get_metadata", arguments: "{}")]),
            .output([getTextMessage("done")]),
        ])
        let agent = Agent(name: "test", tools: [getMetadata], model: .instance(model), hooks: agentHooks)

        _ = try await Runner.run(agent, input: "user_message", hooks: runHooks)

        #expect(runHooks.box.result as? Metadata === metadataResult)
        #expect(agentHooks.box.result as? Metadata === metadataResult)
    }
}
