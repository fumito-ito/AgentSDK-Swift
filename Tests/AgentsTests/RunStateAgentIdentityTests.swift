import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_state_agent_identity.py.
//
// Swift: not ported:
// - `test_from_json_restores_falsy_current_agent_via_identity_map`: Swift has no truthiness, so
//   it is the same as `testFromJSONRestoresDuplicateNameCurrentAgentByIdentity`.
// - `test_capability_identity_uses_config_but_not_bound_session`: sandbox capabilities are out of
//   scope (porting/decisions.md D2).
// - `test_build_agent_map_supports_legacy_handoff_agent_attribute` and
//   `..._legacy_non_handoff_agent_wrapper`: `Agent.handoffs` is a typed `HandoffTarget` list, so
//   legacy duck-typed wrappers cannot be constructed.

private func identity(of target: Agent, in identityMap: [String: Agent]) -> String? {
    identityMap.first { $0.value === target }?.key
}

private func emptyContext() -> RunContextWrapper {
    RunContextWrapper(context: JSONObject())
}

@Suite struct RunStateAgentIdentityTests {
    // MARK: - Identity keys

    @Test func testFromJSONRestoresDuplicateNameCurrentAgentByIdentity() async throws {
        let second = Agent(name: "duplicate")
        let first = Agent(name: "duplicate", handoffs: [.agent(second)])
        second.handoffs = [.agent(first)]
        let state = makeRunState(first, context: emptyContext(), originalInput: "input1", maxTurns: 2)
        state.currentAgent = second

        let jsonData = try state.toJSON()
        #expect(jsonData["current_agent"] == ["name": "duplicate", "identity": "duplicate#2"])

        let restored = try await RunState.fromJSON(first, jsonData)
        #expect(restored.currentAgent === second)
    }

    @Test func testBuildAgentIdentityMapAvoidsLiteralSuffixCollisions() {
        let first = Agent(name: "sandbox")
        let literalSuffix = Agent(name: "sandbox#2")
        let second = Agent(name: "sandbox")
        first.handoffs = [.agent(literalSuffix), .agent(second)]
        literalSuffix.handoffs = [.agent(first), .agent(second)]
        second.handoffs = [.agent(first), .agent(literalSuffix)]

        let identityMap = buildAgentIdentityMap(first)

        #expect(identityMap.count == 3)
        #expect(identityMap["sandbox"] === first)
        #expect(identityMap["sandbox#2"] === literalSuffix)
        #expect(identityMap["sandbox#3"] === second)
    }

    private func reorderedGraphs() throws -> (first: (root: Agent, alpha: Agent, beta: Agent), second: (root: Agent, alpha: Agent, beta: Agent)) {
        let alphaTool = try FunctionTool.create(name: "alpha_tool") { _ in "alpha" }
        let betaTool = try FunctionTool.create(name: "beta_tool") { _ in "beta" }
        func graph(betaFirst: Bool) -> (root: Agent, alpha: Agent, beta: Agent) {
            let alpha = Agent(name: "sandbox", tools: [alphaTool], instructions: "Alpha")
            let beta = Agent(name: "sandbox", tools: [betaTool], instructions: "Beta")
            let root = Agent(name: "triage", handoffs: betaFirst ? [.agent(beta), .agent(alpha)] : [.agent(alpha), .agent(beta)])
            alpha.handoffs = [.agent(root)]
            beta.handoffs = [.agent(root)]
            return (root, alpha, beta)
        }
        return (graph(betaFirst: true), graph(betaFirst: false))
    }

    @Test func testBuildAgentIdentityMapIsStableAcrossReorderedDuplicateAgents() throws {
        let (first, second) = try reorderedGraphs()

        let firstMap = buildAgentIdentityMap(first.root)
        let secondMap = buildAgentIdentityMap(second.root)

        #expect(identity(of: first.alpha, in: firstMap) == identity(of: second.alpha, in: secondMap))
        #expect(identity(of: first.beta, in: firstMap) == identity(of: second.beta, in: secondMap))
    }

    @Test func testFromJSONRestoresDuplicateNameCurrentAgentWithReorderedGraph() async throws {
        let (first, restoredGraph) = try reorderedGraphs()
        let state = makeRunState(first.root, context: emptyContext(), originalInput: "input1", maxTurns: 2)
        state.currentAgent = first.beta
        let jsonData = try state.toJSON()

        let restored = try await RunState.fromJSON(restoredGraph.root, jsonData)
        #expect(restored.currentAgent === restoredGraph.beta)
    }

    @Test func testFromJSONRestoresBareDuplicateNameCurrentAgentViaIdentityMap() async throws {
        let first = Agent(name: "duplicate", instructions: "zeta")
        let second = Agent(name: "duplicate", instructions: "alpha")
        let root = Agent(name: "triage", handoffs: [.agent(first), .agent(second)])
        first.handoffs = [.agent(root)]
        second.handoffs = [.agent(root)]
        let state = makeRunState(root, context: emptyContext(), originalInput: "input1", maxTurns: 2)
        state.currentAgent = second

        let jsonData = try state.toJSON()
        #expect(jsonData["current_agent"] == ["name": "duplicate"])

        let restored = try await RunState.fromJSON(root, jsonData)
        #expect(restored.currentAgent === second)
    }

    private func toolUseBehaviorGraph(stopFirst: Bool) -> (root: Agent, runAgain: Agent, stop: Agent) {
        let runAgain = Agent(name: "sandbox", instructions: "Shared instructions.", toolUseBehavior: .runLLMAgain)
        let stop = Agent(name: "sandbox", instructions: "Shared instructions.", toolUseBehavior: .stopOnFirstTool)
        let root = Agent(name: "triage", handoffs: stopFirst ? [.agent(stop), .agent(runAgain)] : [.agent(runAgain), .agent(stop)])
        runAgain.handoffs = [.agent(root)]
        stop.handoffs = [.agent(root)]
        return (root, runAgain, stop)
    }

    @Test func testBuildAgentIdentityMapUsesToolUseBehaviorForDuplicateNames() {
        let first = toolUseBehaviorGraph(stopFirst: false)
        let second = toolUseBehaviorGraph(stopFirst: true)

        let firstMap = buildAgentIdentityMap(first.root)
        let secondMap = buildAgentIdentityMap(second.root)

        #expect(identity(of: first.runAgain, in: firstMap) == identity(of: second.runAgain, in: secondMap))
        #expect(identity(of: first.stop, in: firstMap) == identity(of: second.stop, in: secondMap))
    }

    @Test func testFromJSONRestoresDuplicateNameCurrentAgentWhenToolUseBehaviorDiffers() async throws {
        let first = toolUseBehaviorGraph(stopFirst: false)
        let state = makeRunState(first.root, context: emptyContext(), originalInput: "input1", maxTurns: 2)
        state.currentAgent = first.stop
        let jsonData = try state.toJSON()

        let restoredGraph = toolUseBehaviorGraph(stopFirst: true)
        let restored = try await RunState.fromJSON(restoredGraph.root, jsonData)
        #expect(restored.currentAgent === restoredGraph.stop)
    }

    // Swift: no upstream counterpart. Pins upstream's `agent_tool_target` entry in tool signatures,
    // which tells apart duplicates whose agent tools share a name but wrap different agents.
    private func agentToolGraph(swapped: Bool) throws -> (root: Agent, usesEnglish: Agent, usesFrench: Agent) {
        let english = Agent(name: "english", instructions: "Answer in English.")
        let french = Agent(name: "french", instructions: "Answer in French.")
        let usesEnglish = Agent(
            name: "duplicate",
            tools: [try english.asTool(toolName: "translate", toolDescription: "Translate the text.")]
        )
        let usesFrench = Agent(
            name: "duplicate",
            tools: [try french.asTool(toolName: "translate", toolDescription: "Translate the text.")]
        )
        let root = Agent(name: "triage", handoffs: swapped ? [.agent(usesFrench), .agent(usesEnglish)] : [.agent(usesEnglish), .agent(usesFrench)])
        return (root, usesEnglish, usesFrench)
    }

    @Test func testBuildAgentIdentityMapUsesAgentToolTargetForDuplicateNames() throws {
        let first = try agentToolGraph(swapped: false)
        let second = try agentToolGraph(swapped: true)

        let firstMap = buildAgentIdentityMap(first.root)
        let secondMap = buildAgentIdentityMap(second.root)

        #expect(identity(of: first.usesEnglish, in: firstMap) == identity(of: second.usesEnglish, in: secondMap))
        #expect(identity(of: first.usesFrench, in: firstMap) == identity(of: second.usesFrench, in: secondMap))
    }

    @Test func testFromJSONRejectsMissingSavedDuplicateIdentity() async throws {
        let second = Agent(name: "duplicate", instructions: "Second")
        let first = Agent(name: "duplicate", instructions: "First", handoffs: [.agent(second)])
        second.handoffs = [.agent(first)]
        let state = makeRunState(first, context: emptyContext(), originalInput: "input1", maxTurns: 2)
        state.currentAgent = second

        let jsonData = try state.toJSON()
        let restoredRoot = Agent(name: "duplicate", instructions: "First")

        await expectErrorAsync(UserError.self, containing: "agent identity") {
            _ = try await RunState.fromJSON(restoredRoot, jsonData)
        }
    }

    @Test func testResultToStatePreservesDuplicateNameRootAndOwnedState() async throws {
        let approvalTool = try FunctionTool.create(name: "approval_tool", needsApproval: true) { _ in "approved" }
        let firstModel = try ScriptedModel()
        let secondModel = try ScriptedModel()
        let first = Agent(name: "duplicate", model: .instance(firstModel))
        let second = Agent(
            name: "duplicate",
            tools: [approvalTool],
            model: .instance(secondModel),
            modelSettings: ModelSettings(toolChoice: .required)
        )
        first.handoffs = [.agent(second)]
        second.handoffs = [.agent(first)]
        try firstModel.enqueue(.output([getHandoffToolCall(second)]))
        try secondModel.enqueue(.output([getFunctionToolCall("approval_tool", arguments: "{}", callId: "call_approval")]))

        let result = try await Runner.run(first, input: "start")
        #expect(!result.interruptions.isEmpty)

        let state = result.toState()
        #expect(state.startingAgent === first)
        #expect(state.currentAgent === second)

        let jsonData = try state.toJSON()
        let secondReference: JSONValue = ["name": "duplicate", "identity": "duplicate#2"]
        #expect(jsonData["current_agent"] == secondReference)
        #expect(jsonData["tool_use_tracker"]?.objectValue?["duplicate#2"] == ["approval_tool"])
        let step = try #require(jsonData["current_step"]?.objectValue)
        #expect(step["data"]?.objectValue?["interruptions"]?.arrayValue?.first?.objectValue?["agent"] == secondReference)

        let approvalItems = (jsonData["generated_items"]?.arrayValue ?? []).compactMap(\.objectValue).filter {
            $0["type"] == "tool_call_item" && $0["raw_item"]?.objectValue?["call_id"] == "call_approval"
        }
        #expect(approvalItems.count == 1)
        #expect(approvalItems.first?["agent"] == secondReference)
        #expect(approvalItems.first?["raw_item"] == [
            "arguments": "{}", "call_id": "call_approval", "id": "1", "name": "approval_tool", "type": "function_call",
        ])

        let restored = try await RunState.fromJSON(first, jsonData)
        #expect(restored.startingAgent === first)
        #expect(restored.currentAgent === second)
        #expect(restored.getInterruptions().first?.agent === second)
        #expect(restored.generatedItems.contains { item in
            (item as? ToolCallItem).map { $0.agent === second && $0.rawItem["call_id"] == "call_approval" } ?? false
        })
    }

    // MARK: - Agent map

    @Test func testBuildAgentMapCollectsAgentsWithoutLooping() {
        let agentA = Agent(name: "AgentA")
        let agentB = Agent(name: "AgentB")
        agentA.handoffs = [.agent(agentB)]
        agentB.handoffs = [.agent(agentA)]

        let agentMap = buildAgentMap(agentA)

        #expect(agentMap["AgentA"]?.name == agentA.name)
        #expect(agentMap["AgentB"]?.name == agentB.name)
        #expect(agentMap.keys.sorted() == ["AgentA", "AgentB"])
    }

    @Test func testBuildAgentMapHandlesComplexHandoffGraphs() {
        let agentA = Agent(name: "A")
        let agentB = Agent(name: "B")
        let agentC = Agent(name: "C")
        let agentD = Agent(name: "D")
        agentA.handoffs = [.agent(agentB), .agent(agentC)]
        agentB.handoffs = [.agent(agentD)]
        agentC.handoffs = [.agent(agentD)]

        let agentMap = buildAgentMap(agentA)

        #expect(agentMap.count == 4)
        #expect(["A", "B", "C", "D"].allSatisfy { agentMap[$0] != nil })
    }

    @Test func testBuildAgentMapHandlesHandoffObjects() {
        let agentA = Agent(name: "AgentA")
        let agentB = Agent(name: "AgentB")
        agentA.handoffs = [.handoff(handoff(agentB))]

        #expect(buildAgentMap(agentA).keys.sorted() == ["AgentA", "AgentB"])
    }

    @Test func testBuildAgentMapSkipsUnresolvedHandoffObjects() {
        let agentA = Agent(name: "AgentA")
        let agentB = Agent(name: "AgentB")
        let detachedHandoff = Handoff(
            toolName: "transfer_to_agent_b",
            toolDescription: "Transfer to AgentB.",
            inputJSONSchema: [:],
            onInvokeHandoff: { _, _ in agentB },
            agentName: agentB.name
        )
        agentA.handoffs = [.handoff(detachedHandoff)]

        #expect(buildAgentMap(agentA).keys.sorted() == ["AgentA"])
    }

    // MARK: - Item ownership

    @Test func testSerializationUsesDuplicateIdentitiesForHandoffAndOutputGuardrails() async throws {
        let first = Agent(name: "duplicate")
        let second = Agent(name: "duplicate")
        let third = Agent(name: "duplicate")
        first.handoffs = [.agent(second), .agent(third)]
        second.handoffs = [.agent(third)]
        third.handoffs = [.agent(first)]
        let state = makeRunState(first, context: emptyContext(), originalInput: "test handoff", maxTurns: 2)
        state.currentAgent = second
        state.generatedItems = [
            HandoffOutputItem(agent: second, rawItem: ["type": "handoff_output", "status": "completed"], sourceAgent: second, targetAgent: third),
        ]
        let guardrailOutput = GuardrailFunctionOutput(outputInfo: ["guardrail": "ok"] as JSONObject, tripwireTriggered: false)
        state.outputGuardrailResults = [
            OutputGuardrailResult(
                guardrail: OutputGuardrail(name: "duplicate_output_guardrail") { _, _, _ in guardrailOutput },
                agentOutput: "done",
                agent: third,
                output: guardrailOutput
            ),
        ]

        let jsonData = try state.toJSON()
        let itemData = try #require(jsonData["generated_items"]?.arrayValue?.first?.objectValue)
        #expect(itemData["agent"] == ["name": "duplicate", "identity": "duplicate#2"])
        #expect(itemData["source_agent"] == ["name": "duplicate", "identity": "duplicate#2"])
        #expect(itemData["target_agent"] == ["name": "duplicate", "identity": "duplicate#3"])
        #expect(jsonData["output_guardrail_results"]?.arrayValue?.first?.objectValue?["agent"] == ["name": "duplicate", "identity": "duplicate#3"])

        let restored = try await RunState.fromJSON(first, jsonData)
        let restoredItem = try #require(restored.generatedItems.first as? HandoffOutputItem)
        #expect(restoredItem.agent === second)
        #expect(restoredItem.sourceAgent === second)
        #expect(restoredItem.targetAgent === third)
        #expect(restored.outputGuardrailResults.first?.agent === third)
    }
}
