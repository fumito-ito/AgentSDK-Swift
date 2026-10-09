import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_hooks.py.
@Suite struct AgentHooksTests {
    private struct Foo: Codable, Sendable {
        var a: String
    }

    private struct Agents3 {
        let model: ScriptedModel
        let agent1: Agent
        let agent3: Agent
    }

    private func makeAgents(hooks: AgentHooks, structured: Bool = false) throws -> Agents3 {
        let model = try ScriptedModel()
        let agent1 = Agent(name: "test_1", model: .instance(model))
        let agent2 = Agent(name: "test_2", model: .instance(model))
        let agent3 = Agent(
            name: "test_3",
            tools: [getFunctionTool(name: "some_function", returnValue: "result")],
            handoffs: [.agent(agent1), .agent(agent2)],
            model: .instance(model),
            outputType: structured ? try AgentOutputSchema(Foo.self) : nil,
            hooks: hooks
        )
        agent1.handoffs.append(.agent(agent3))
        return Agents3(model: model, agent1: agent1, agent3: agent3)
    }

    private func run(_ agent: Agent, streamed: Bool) async throws {
        if streamed {
            let output = try Runner.runStreamed(agent, input: "user_message")
            for try await _ in output.streamEvents() {}
        } else {
            _ = try await Runner.run(agent, input: "user_message")
        }
    }

    private func newHooks() -> AgentHooksForTests {
        AgentHooksForTests(recordsLLMEvents: false, toolContextEvents: ["on_tool_start", "on_tool_end"])
    }

    private let toolCallArgs = #"{"a": "b"}"#

    @Test func testFalsyAgentHooksAreInvoked() async throws {
        // Swift: there is no truthiness, so this checks that hooks fire for a tool-calling run.
        let hooks = newHooks()
        let model = try ScriptedModel(steps: [
            .output([getFunctionToolCall("some_function", arguments: toolCallArgs)]),
            .output([getTextMessage("done")]),
        ])
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "some_function", returnValue: "result")], model: .instance(model), hooks: hooks)

        _ = try await Runner.run(agent, input: "user_message")

        #expect(hooks.events == ["on_start": 1, "on_tool_start": 1, "on_tool_end": 1, "on_end": 1])
    }

    @Test(arguments: [false, true])
    func testAgentHooks(streamed: Bool) async throws {
        // Ports test_non_streamed_agent_hooks and test_streamed_agent_hooks.
        let hooks = newHooks()
        let agents = try makeAgents(hooks: hooks)
        let model = agents.model

        try model.enqueue(.output([getTextMessage("user_message")]))
        try await run(agents.agent3, streamed: streamed)
        #expect(hooks.events == ["on_start": 1, "on_end": 1])
        hooks.reset()

        if !streamed {
            try model.extend([
                .output([getFunctionToolCall("some_function", arguments: toolCallArgs, callId: "tool_1")]),
                .output([getTextMessage("done")]),
            ])
            try await run(agents.agent3, streamed: false)
            #expect(hooks.toolContextIds.count == 2)
            #expect(Set(hooks.toolContextIds).count == 1)
            hooks.reset()
        }

        try model.extend([
            // First turn: a tool call
            .output([getFunctionToolCall("some_function", arguments: toolCallArgs)]),
            // Second turn: a message and a handoff
            .output([getTextMessage("a_message"), getHandoffToolCall(agents.agent1)]),
            // Third turn: text message
            .output([getTextMessage("done")]),
        ])
        try await run(agents.agent3, streamed: streamed)
        // Shouldn't have on_end because it's not the last agent
        #expect(hooks.events == ["on_start": 1, "on_tool_start": 1, "on_tool_end": 1, "on_handoff": 1])
        hooks.reset()

        try model.extend([
            // First turn: a tool call
            .output([getFunctionToolCall("some_function", arguments: toolCallArgs, callId: "tool_1")]),
            // Second turn: a message, another tool call, and a handoff
            .output([
                getTextMessage("a_message"),
                getFunctionToolCall("some_function", arguments: toolCallArgs, callId: "tool_2"),
                getHandoffToolCall(agents.agent1),
            ]),
            // Third turn: a message and a handoff back to the orig agent
            .output([getTextMessage("a_message"), getHandoffToolCall(agents.agent3)]),
            // Fourth turn: text message
            .output([getTextMessage("done")]),
        ])
        try await run(agents.agent3, streamed: streamed)
        #expect(hooks.events == ["on_start": 2, "on_tool_start": 2, "on_tool_end": 2, "on_handoff": 1, "on_end": 1])
    }

    @Test(arguments: [false, true])
    func testStructuredOutputAgentHooks(streamed: Bool) async throws {
        // Ports test_structured_output_non_streamed_agent_hooks and the streamed variant.
        let hooks = newHooks()
        let agents = try makeAgents(hooks: hooks, structured: true)
        let model = agents.model

        try model.enqueue(.output([getFinalOutputMessage(toolCallArgs)]))
        try await run(agents.agent3, streamed: streamed)
        #expect(hooks.events == ["on_start": 1, "on_end": 1])
        hooks.reset()

        try model.extend([
            .output([getFunctionToolCall("some_function", arguments: toolCallArgs)]),
            .output([getTextMessage("a_message"), getHandoffToolCall(agents.agent1)]),
            // Third turn: end message (for agent 1)
            .output([getTextMessage("done")]),
        ])
        // Upstream runs this step non-streamed in both variants.
        try await run(agents.agent3, streamed: false)
        #expect(hooks.events == ["on_start": 1, "on_tool_start": 1, "on_tool_end": 1, "on_handoff": 1])
        hooks.reset()

        try model.extend([
            .output([getFunctionToolCall("some_function", arguments: toolCallArgs, callId: "tool_1")]),
            .output([
                getTextMessage("a_message"),
                getFunctionToolCall("some_function", arguments: toolCallArgs, callId: "tool_2"),
                getHandoffToolCall(agents.agent1),
            ]),
            .output([getTextMessage("a_message"), getHandoffToolCall(agents.agent3)]),
            // Fourth turn: end message (for agent 3)
            .output([getFinalOutputMessage(toolCallArgs)]),
        ])
        try await run(agents.agent3, streamed: streamed)
        #expect(hooks.events == ["on_start": 2, "on_tool_start": 2, "on_tool_end": 2, "on_handoff": 1, "on_end": 1])
    }

    @Test func testBaseAgentHooksDontCrash() async throws {
        let agents = try makeAgents(hooks: AgentHooks(), structured: true)
        let model = agents.model

        try model.enqueue(.output([getFinalOutputMessage(toolCallArgs)]))
        try await run(agents.agent3, streamed: true)

        try model.extend([
            .output([getFunctionToolCall("some_function", arguments: toolCallArgs)]),
            .output([getTextMessage("a_message"), getHandoffToolCall(agents.agent1)]),
            .output([getTextMessage("done")]),
        ])
        try await run(agents.agent3, streamed: false)

        try model.extend([
            .output([getFunctionToolCall("some_function", arguments: toolCallArgs)]),
            .output([getTextMessage("a_message"), getFunctionToolCall("some_function", arguments: toolCallArgs), getHandoffToolCall(agents.agent1)]),
            .output([getTextMessage("a_message"), getHandoffToolCall(agents.agent3)]),
            .output([getFinalOutputMessage(toolCallArgs)]),
        ])
        try await run(agents.agent3, streamed: true)
    }

    // MARK: - Turn input

    private final class AgentHooksWithTurnInput: AgentHooks, @unchecked Sendable {
        let capture = TurnInputCapture()

        override func onStart(context: AgentHookContext, agent: AgentBase) async throws {
            capture.append(context.turnInput)
        }
    }

    @Test func testAgentHooksReceivesTurnInputString() async throws {
        let hooks = AgentHooksWithTurnInput()
        let model = try ScriptedModel(steps: [.output([getTextMessage("response")])])
        let agent = Agent(name: "test", model: .instance(model), hooks: hooks)

        _ = try await Runner.run(agent, input: "hello world")

        let turnInputs = hooks.capture.capturedTurnInputs
        #expect(turnInputs.count == 1)
        #expect(turnInputs[0].count == 1)
        #expect(turnInputs[0][0].string("content") == "hello world")
        #expect(turnInputs[0][0].string("role") == "user")
    }

    @Test func testAgentHooksReceivesTurnInputList() async throws {
        let hooks = AgentHooksWithTurnInput()
        let model = try ScriptedModel(steps: [.output([getTextMessage("response")])])
        let agent = Agent(name: "test", model: .instance(model), hooks: hooks)
        let inputItems: [TResponseInputItem] = [
            ["role": "user", "content": "first message"],
            ["role": "user", "content": "second message"],
        ]

        _ = try await Runner.run(agent, input: .items(inputItems))

        let turnInputs = hooks.capture.capturedTurnInputs
        #expect(turnInputs.count == 1)
        #expect(turnInputs[0].map { $0.string("content") } == ["first message", "second message"])
    }

    @Test func testAgentHooksReceivesTurnInputStreamed() async throws {
        let hooks = AgentHooksWithTurnInput()
        let model = try ScriptedModel(steps: [.output([getTextMessage("response")])])
        let agent = Agent(name: "test", model: .instance(model), hooks: hooks)

        let result = try Runner.runStreamed(agent, input: "streamed input")
        for try await _ in result.streamEvents() {}

        let turnInputs = hooks.capture.capturedTurnInputs
        #expect(turnInputs.count == 1)
        #expect(turnInputs[0].count == 1)
        #expect(turnInputs[0][0].string("content") == "streamed input")
    }
}
