import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_memory_leak.py.
@Suite struct AgentMemoryLeakTests {
    @Test func testAgentIsReleasedAfterRun() async throws {
        let scriptedModel = try ScriptedModel(steps: [.output([getTextMessage("Paris")])])
        weak var agentRef: Agent?

        do {
            let agent = Agent(name: "leak-test-agent", instructions: "Answer questions.", model: .instance(scriptedModel))
            agentRef = agent
            // Running the agent should not leave behind strong references once the result goes out of scope.
            _ = try await Runner.run(agent, input: "What is the capital of France?")
        }

        #expect(agentRef == nil)
    }
}
