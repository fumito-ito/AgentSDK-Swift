import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_llm_hooks.py.
//
// Swift: `test_sync_agent_hook_with_llm` is skipped because Swift has no `Runner.run_sync`.
@Suite struct AgentLLMHooksTests {
    private func makeAgent(_ hooks: AgentHooks) throws -> (Agent, ScriptedModel) {
        let model = try ScriptedModel()
        let agent = Agent(name: "A", tools: [getFunctionTool(name: "f", returnValue: "res")], handoffs: [], model: .instance(model), hooks: hooks)
        return (agent, model)
    }

    @Test func testAsyncAgentHooksWithLLM() async throws {
        let hooks = AgentHooksForTests()
        let (agent, model) = try makeAgent(hooks)
        // Simulate a single LLM call producing an output:
        try model.enqueue(.output([getTextMessage("hello")]))

        _ = try await Runner.run(agent, input: "hello")

        // Expect one on_start, one on_llm_start, one on_llm_end, and one on_end
        #expect(hooks.events == ["on_start": 1, "on_llm_start": 1, "on_llm_end": 1, "on_end": 1])
    }

    @Test func testStreamedAgentHooksWithLLM() async throws {
        let hooks = AgentHooksForTests()
        let (agent, model) = try makeAgent(hooks)
        try model.enqueue(.output([getTextMessage("hello")]))

        let stream = try Runner.runStreamed(agent, input: "hello")
        for try await _ in stream.streamEvents() {}

        #expect(hooks.events == ["on_start": 1, "on_llm_start": 1, "on_llm_end": 1, "on_end": 1])
    }
}
