import Foundation
import Testing
@testable import Agents

// Ported from tests/test_agent_config.py.
//
// Swift: `TestAgentValidation` (runtime type checks of constructor arguments) and
// `test_agent_does_not_promote_model_settings_to_constructor` are skipped because the compiler
// enforces those types. Dictionary model settings are covered through `ModelSettings(json:)`.
@Suite struct AgentConfigTests {
    private struct Foo: Codable, Sendable {
        var bar: String
    }

    // MARK: - Instructions

    @Test func testSystemInstructions() async throws {
        let agent = Agent(name: "test", instructions: "abc123")
        let context = RunContextWrapper()

        #expect(try await agent.getSystemPrompt(context) == "abc123")

        let syncAgent = agent.clone { $0.instructions = .dynamic { _, _ in "sync_123" } }
        #expect(try await syncAgent.getSystemPrompt(context) == "sync_123")

        let asyncAgent = syncAgent.clone {
            $0.instructions = .dynamic { _, _ in
                await Task.yield()
                return "async_123"
            }
        }
        #expect(try await asyncAgent.getSystemPrompt(context) == "async_123")
    }

    // MARK: - Handoffs

    @Test func testHandoffWithAgents() async throws {
        let agent1 = Agent(name: "agent_1")
        let agent2 = Agent(name: "agent_2")
        let agent3 = Agent(name: "agent_3", handoffs: [.agent(agent1), .agent(agent2)])

        let handoffs = try await getHandoffs(agent3, RunContextWrapper())
        #expect(handoffs.count == 2)
        #expect(handoffs[0].agentName == "agent_1")
        #expect(handoffs[1].agentName == "agent_2")

        let firstReturn = try await handoffs[0].onInvokeHandoff(RunContextWrapper(), "")
        #expect(firstReturn === agent1)
        let secondReturn = try await handoffs[1].onInvokeHandoff(RunContextWrapper(), "")
        #expect(secondReturn === agent2)
    }

    @Test func testHandoffWithHandoffObj() async throws {
        let agent1 = Agent(name: "agent_1")
        let agent2 = Agent(name: "agent_2")
        let agent3 = Agent(name: "agent_3", handoffs: [
            .handoff(handoff(agent1)),
            .handoff(handoff(agent2, toolNameOverride: "transfer_to_2", toolDescriptionOverride: "description_2")),
        ])

        let handoffs = try await getHandoffs(agent3, RunContextWrapper())
        #expect(handoffs.count == 2)
        #expect(handoffs[0].agentName == "agent_1")
        #expect(handoffs[1].agentName == "agent_2")
        #expect(handoffs[0].toolName == Handoff.defaultToolName(agent1))
        #expect(handoffs[1].toolName == "transfer_to_2")
        #expect(handoffs[0].toolDescription == Handoff.defaultToolDescription(agent1))
        #expect(handoffs[1].toolDescription == "description_2")

        let firstReturn = try await handoffs[0].onInvokeHandoff(RunContextWrapper(), "")
        #expect(firstReturn === agent1)
        let secondReturn = try await handoffs[1].onInvokeHandoff(RunContextWrapper(), "")
        #expect(secondReturn === agent2)
    }

    @Test func testHandoffWithHandoffObjAndAgent() async throws {
        let agent1 = Agent(name: "agent_1")
        let agent2 = Agent(name: "agent_2")
        let agent3 = Agent(name: "agent_3", handoffs: [.handoff(handoff(agent1)), .agent(agent2)])

        let handoffs = try await getHandoffs(agent3, RunContextWrapper())
        #expect(handoffs.count == 2)
        #expect(handoffs[0].agentName == "agent_1")
        #expect(handoffs[1].agentName == "agent_2")
        #expect(handoffs[0].toolName == Handoff.defaultToolName(agent1))
        #expect(handoffs[1].toolName == Handoff.defaultToolName(agent2))
        #expect(handoffs[0].toolDescription == Handoff.defaultToolDescription(agent1))
        #expect(handoffs[1].toolDescription == Handoff.defaultToolDescription(agent2))

        let firstReturn = try await handoffs[0].onInvokeHandoff(RunContextWrapper(), "")
        #expect(firstReturn === agent1)
        let secondReturn = try await handoffs[1].onInvokeHandoff(RunContextWrapper(), "")
        #expect(secondReturn === agent2)
    }

    // MARK: - Cloning and output

    @Test func testAgentCloning() {
        let agent = Agent(name: "test", handoffDescription: "test_description", model: "o3-mini")

        let cloned = agent.clone {
            $0.handoffDescription = "new_description"
            $0.model = "o1"
        }

        #expect(cloned.name == "test")
        #expect(cloned.handoffDescription == "new_description")
        #expect(cloned.model?.name == "o1")
    }

    @Test func testAgentFinalOutput() throws {
        let agent = Agent(name: "test", outputType: try AgentOutputSchema(Foo.self))

        let schema = try #require(getOutputSchema(agent))
        #expect(schema is AgentOutputSchema<Foo>)
        #expect(schema.isStrictJSONSchema())
        #expect(try !schema.jsonSchema().isEmpty)
        #expect(!schema.isPlainText())
    }

    // MARK: - Dictionary model settings

    @Test func testAgentModelSettingsDictionaryPreservesOpenAIReasoningExtensions() throws {
        let settings = try ModelSettings(json: [
            "reasoning": ["context": "all_turns", "future_reasoning_option": "enabled"],
            "context_management": [["type": "compaction", "compact_threshold": 244_800]],
            "retry": ["max_retries": 0, "backoff": ["jitter": false]],
        ])
        let agent = Agent(name: "test", modelSettings: settings)

        let reasoning = try #require(agent.modelSettings.reasoning)
        #expect(reasoning.context == "all_turns")
        #expect(reasoning.json["future_reasoning_option"] == "enabled")
        #expect(agent.modelSettings.contextManagement == [["type": "compaction", "compact_threshold": 244_800]])
        let retry = try #require(agent.modelSettings.retry)
        #expect(retry.maxRetries == 0)
        #expect(retry.backoff?.jitter == false)
    }

    @Test(arguments: [
        (["temperatur": 0.2] as JSONObject, "Unknown model settings: temperatur"),
        (["retry": ["max_retry": 2]] as JSONObject, "Unknown model settings in retry: max_retry"),
        (["retry": ["backoff": ["initial_delai": 1]]] as JSONObject, "Unknown model settings in retry.backoff: initial_delai"),
        (
            ["context_management": [["type": "compaction", "compact_threshold_typo": 1]]] as JSONObject,
            "Unknown model settings in context_management[0]: compact_threshold_typo"
        ),
    ])
    func testAgentRejectsUnknownFirstPartyDictionaryModelSettings(settings: JSONObject, message: String) {
        expectError(TypeError.self, containing: message) {
            _ = try ModelSettings(json: settings)
        }
    }
}
