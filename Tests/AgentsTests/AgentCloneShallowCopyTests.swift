import Testing
@testable import Agents

// Ported from tests/test_agent_clone_shallow_copy.py.
//
// Swift: `tools` and `handoffs` are value-type arrays, so list identity cannot be shared between
// agents. These tests check that the entries are shared, and
// `test_agent_clone_shared_list_mutation_affects_both_agents` is skipped.
@Suite struct AgentCloneShallowCopyTests {
    private struct NameArgs: Codable, Sendable {
        var name: String
    }

    private let greet: FunctionTool = {
        // swiftlint:disable:next force_try
        try! FunctionTool.create(name: "greet", argumentsType: NameArgs.self) { _, args in "Hello, \(args.name)!" }
    }()

    @Test func testAgentCloneShallowCopy() throws {
        let targetAgent = Agent(name: "Target")
        let original = Agent(name: "Original", tools: [greet], instructions: "Testing clone shallow copy", handoffs: [.handoff(handoff(targetAgent))])

        let cloned = original.clone { $0.name = "Cloned" }

        #expect(cloned !== original)
        #expect(cloned.name == "Cloned")
        guard case .text(let clonedInstructions)? = cloned.instructions, case .text(let originalInstructions)? = original.instructions else {
            Issue.record("Expected text instructions")
            return
        }
        #expect(clonedInstructions == originalInstructions)
        #expect(cloned.tools[0] as? FunctionTool === original.tools[0] as? FunctionTool, "Tool objects should be same instance")
        guard case .handoff(let clonedHandoff) = cloned.handoffs[0], case .handoff(let originalHandoff) = original.handoffs[0] else {
            Issue.record("Expected handoff entries")
            return
        }
        #expect(clonedHandoff.toolName == originalHandoff.toolName)
        #expect(clonedHandoff.agentName == originalHandoff.agentName)
    }

    @Test func testAgentCloneKeepsListAttributesItIsNotGiven() {
        let original = Agent(name: "Original", tools: [greet], handoffs: [.agent(Agent(name: "Target"))])

        let cloned = original.clone { $0.name = "Cloned" }

        #expect(cloned.tools.count == original.tools.count)
        #expect(cloned.tools[0] as? FunctionTool === greet)
        #expect(cloned.handoffs.count == original.handoffs.count)
    }

    @Test func testAgentCloneUsesAGivenListAsIs() throws {
        let farewell = try FunctionTool.create(name: "farewell", argumentsType: NameArgs.self) { _, args in "Goodbye, \(args.name)!" }
        let original = Agent(name: "Original", tools: [greet])
        let supplied: [any Tool] = [farewell]

        let cloned = original.clone {
            $0.name = "Cloned"
            $0.tools = supplied
        }

        #expect(cloned.tools.map(\.name) == ["farewell"])
        #expect(original.tools.map(\.name) == ["greet"])
        #expect(cloned.tools.allSatisfy { $0 as? FunctionTool !== greet })
    }

    @Test func testAgentCloneStillSharesWhenGivenTheOriginalList() {
        let original = Agent(name: "Original", tools: [greet])

        let cloned = original.clone {
            $0.name = "Cloned"
            $0.tools = original.tools
        }

        #expect(cloned.tools[0] as? FunctionTool === original.tools[0] as? FunctionTool)
    }
}
