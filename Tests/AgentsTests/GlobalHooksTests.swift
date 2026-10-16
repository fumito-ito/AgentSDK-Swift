import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_global_hooks.py.
@Suite struct GlobalHooksTests {
    private struct Foo: Codable, Sendable {
        var a: String
    }

    private func makeAgents(structured: Bool = false) throws -> (model: ScriptedModel, agent1: Agent, agent3: Agent) {
        let model = try ScriptedModel()
        let agent1 = Agent(name: "test_1", model: .instance(model))
        let agent2 = Agent(name: "test_2", model: .instance(model))
        let agent3 = Agent(
            name: "test_3",
            tools: [getFunctionTool(name: "some_function", returnValue: "result")],
            handoffs: [.agent(agent1), .agent(agent2)],
            model: .instance(model),
            outputType: structured ? try AgentOutputSchema(Foo.self) : nil
        )
        agent1.handoffs.append(.agent(agent3))
        return (model, agent1, agent3)
    }

    private func run(_ agent: Agent, hooks: RunHooks, streamed: Bool) async throws {
        if streamed {
            let output = try Runner.runStreamed(agent, input: "user_message", hooks: hooks)
            for try await _ in output.streamEvents() {}
        } else {
            _ = try await Runner.run(agent, input: "user_message", hooks: hooks)
        }
    }

    private let toolCallArgs = #"{"a": "b"}"#

    @Test(arguments: [false, true], [false, true])
    func testAgentHooks(streamed: Bool, structured: Bool) async throws {
        // Ports test_non_streamed_agent_hooks, test_streamed_agent_hooks, and the structured-output variants.
        let hooks = RunHooksForTests(recordsLLMEvents: false, toolContextEvents: ["on_tool_start", "on_tool_end"])
        let (model, agent1, agent3) = try makeAgents(structured: structured)
        let finalMessage = structured ? getFinalOutputMessage(toolCallArgs) : getTextMessage("user_message")
        let lastMessage = structured ? getFinalOutputMessage(toolCallArgs) : getTextMessage("done")

        try model.enqueue(.output([finalMessage]))
        try await run(agent3, hooks: hooks, streamed: streamed)
        #expect(hooks.events == ["on_agent_start": 1, "on_agent_end": 1])
        hooks.reset()

        if !streamed, !structured {
            try model.extend([
                .output([getFunctionToolCall("some_function", arguments: toolCallArgs)]),
                .output([getTextMessage("done")]),
            ])
            try await run(agent3, hooks: hooks, streamed: false)
            #expect(hooks.toolContextIds.count == 2)
            #expect(Set(hooks.toolContextIds).count == 1)
            hooks.reset()
        }

        try model.extend([
            // First turn: a tool call
            .output([getFunctionToolCall("some_function", arguments: toolCallArgs)]),
            // Second turn: a message and a handoff
            .output([getTextMessage("a_message"), getHandoffToolCall(agent1)]),
            // Third turn: text message
            .output([getTextMessage("done")]),
        ])
        try await run(agent3, hooks: hooks, streamed: streamed)
        // on_agent_start fires once for agent_3's segment and once for agent_1.
        #expect(hooks.events == ["on_agent_start": 2, "on_tool_start": 1, "on_tool_end": 1, "on_handoff": 1, "on_agent_end": 1])
        hooks.reset()

        try model.extend([
            // First turn: a tool call
            .output([getFunctionToolCall("some_function", arguments: toolCallArgs, callId: "tool_1")]),
            // Second turn: a message, another tool call, and a handoff
            .output([
                getTextMessage("a_message"),
                getFunctionToolCall("some_function", arguments: toolCallArgs, callId: "tool_2"),
                getHandoffToolCall(agent1),
            ]),
            // Third turn: a message and a handoff back to the orig agent
            .output([getTextMessage("a_message"), getHandoffToolCall(agent3)]),
            // Fourth turn: final message
            .output([lastMessage]),
        ])
        try await run(agent3, hooks: hooks, streamed: streamed)
        // agent_3 starts, hands off to agent_1, which hands back to agent_3.
        #expect(hooks.events == ["on_agent_start": 3, "on_tool_start": 2, "on_tool_end": 2, "on_handoff": 2, "on_agent_end": 1])
    }
}
