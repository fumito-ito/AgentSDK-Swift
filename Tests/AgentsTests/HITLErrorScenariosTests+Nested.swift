import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_hitl_error_scenarios.py: the nested `Agent.asTool()` scenarios.

extension HITLErrorScenariosTests {
    // MARK: - Nested agent tools

    @Test func testNestedAgentToolResumesAfterRejection() async throws {
        let innerHITLTool = try FunctionTool.create(name: "inner_hitl_tool", needsApproval: .always) { _ in "ok" }
        let innerModel = try ScriptedModel()
        let innerAgent = makeAgent(model: innerModel, tools: [innerHITLTool], name: "Inner")
        try innerModel.extend([
            ModelStep(output: [makeFunctionToolCall(innerHITLTool.name, callId: "inner-1")]),
            ModelStep(output: [makeFunctionToolCall(innerHITLTool.name, callId: "inner-2")]),
            ModelStep(output: [getTextMessage("done")]),
        ])
        let agentTool = try innerAgent.asTool(
            toolName: "inner_agent_tool", toolDescription: "Inner agent tool with HITL", needsApproval: .always
        )
        let outerModel = try ScriptedModel()
        let outerAgent = makeAgent(model: outerModel, tools: [agentTool], name: "Outer")
        try outerModel.extend([ModelStep(output: [makeFunctionToolCall(agentTool.name, callId: "outer-1", arguments: #"{"input":"hi"}"#)])])

        let first = try await Runner.run(outerAgent, input: "start")
        #expect(!first.interruptions.isEmpty, "agent tool should request approval first")
        #expect(first.interruptions.first?.toolName == agentTool.name)

        let stateAfterOuterApproval = first.toState()
        try stateAfterOuterApproval.approve(try #require(first.interruptions.first), alwaysApprove: true)

        let second = try await Runner.run(outerAgent, state: stateAfterOuterApproval)
        #expect(!second.interruptions.isEmpty, "inner tool should request approval on first run")
        #expect(second.interruptions.first?.toolName == innerHITLTool.name)

        let stateAfterInnerReject = second.toState()
        try stateAfterInnerReject.reject(try #require(second.interruptions.first))

        let third = try await Runner.run(outerAgent, state: stateAfterInnerReject)
        #expect(!third.interruptions.isEmpty, "nested agent should resume and request new approval")
        #expect(third.interruptions.first?.toolName == innerHITLTool.name)
        #expect(third.interruptions.first.flatMap { getToolApprovalItemCallId($0) } == "inner-2")
        let rejectionOutputs = third.newItems.compactMap { $0 as? ToolCallOutputItem }.filter {
            ($0.output as? String) == hitlRejectionMessage && $0.callId == "outer-1"
        }
        #expect(rejectionOutputs.isEmpty, "Nested rejection should not short-circuit the agent tool")
    }

    @Test func testChangedNestedParentFailsBeforeToolInventoryCallbacks() async throws {
        let enabledCalls = SharedBox<[String]>([])
        let innerHITLTool = try FunctionTool.create(name: "inner_hitl_tool", needsApproval: .always) { _ in "ok" }
        let observer = try FunctionTool.create(name: "observer", isEnabled: .dynamic { _, agent in
            enabledCalls.mutate { $0.append(agent.name) }
            return true
        }) { _ in "unused" }
        let innerModel = try ScriptedModel(steps: [ModelStep(output: [makeFunctionToolCall(innerHITLTool.name, callId: "inner-1")])])
        let innerAgent = makeAgent(model: innerModel, tools: [innerHITLTool], name: "Inner")
        let agentTool = try innerAgent.asTool(
            toolName: "inner_agent_tool", toolDescription: "Inner agent tool with HITL", needsApproval: .always
        )
        let outerModel = try ScriptedModel(steps: [
            ModelStep(output: [makeFunctionToolCall(agentTool.name, callId: "outer-1", arguments: #"{"input":"safe"}"#)]),
        ])
        let outerAgent = makeAgent(model: outerModel, tools: [agentTool, observer], name: "Outer")

        let first = try await Runner.run(outerAgent, input: "start")
        let firstState = first.toState()
        try firstState.approve(try #require(first.interruptions.first))
        let second = try await Runner.run(outerAgent, state: firstState)
        #expect(second.interruptions.first?.toolName == innerHITLTool.name)

        let resumeState = second.toState()
        var processed = try #require(resumeState.lastProcessedResponse)
        enabledCalls.value = []
        var changedCall = processed.functions[0].toolCall.json
        changedCall["arguments"] = #"{"input":"evil"}"#
        processed.functions[0].toolCall = try #require(ResponseFunctionToolCall(json: changedCall))
        resumeState.lastProcessedResponse = processed
        try resumeState.approve(try #require(second.interruptions.first))

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Runner.run(outerAgent, state: resumeState)
        }
        #expect(enabledCalls.value.isEmpty)
    }

    @Test func testNestedAgentToolInterruptionsRemainDistinctAcrossOuterCalls() async throws {
        let innerHITLTool = try FunctionTool.create(name: "inner_hitl_tool", needsApproval: .always) { _ in "ok" }
        let innerModel = try ScriptedModel()
        let innerAgent = makeAgent(model: innerModel, tools: [innerHITLTool], name: "Inner")
        try innerModel.extend([
            ModelStep(output: [makeFunctionToolCall(innerHITLTool.name, callId: "inner-1")]),
            ModelStep(output: [makeFunctionToolCall(innerHITLTool.name, callId: "inner-2")]),
        ])
        let agentTool = try innerAgent.asTool(toolName: "inner_agent_tool", toolDescription: "Inner agent tool", needsApproval: .never)
        let outerModel = try ScriptedModel()
        let outerAgent = makeAgent(model: outerModel, tools: [agentTool], name: "Outer")
        try outerModel.extend([
            ModelStep(output: [
                makeFunctionToolCall(agentTool.name, callId: "outer-a", arguments: #"{"input":"a"}"#),
                makeFunctionToolCall(agentTool.name, callId: "outer-b", arguments: #"{"input":"b"}"#),
            ]),
        ])

        let result = try await Runner.run(outerAgent, input: "start")

        #expect(!result.interruptions.isEmpty, "nested agent tool should request approvals")
        let nestedInterruptions = result.interruptions.filter { $0.toolName == innerHITLTool.name }
        #expect(nestedInterruptions.count == 2)
    }

    @Test func testNestedAgentToolDoesNotInheritParentApprovals() async throws {
        let outerSharedTool = try FunctionTool.create(name: "shared_tool", needsApproval: .always) { _ in "outer" }
        let innerSharedTool = try FunctionTool.create(name: "shared_tool", needsApproval: .always) { _ in "inner" }
        let innerModel = try ScriptedModel()
        let innerAgent = makeAgent(model: innerModel, tools: [innerSharedTool], name: "Inner")
        try innerModel.extend([ModelStep(output: [makeFunctionToolCall(innerSharedTool.name, callId: "dup")])])
        let agentTool = try innerAgent.asTool(toolName: "inner_agent_tool", toolDescription: "Inner agent tool", needsApproval: .never)
        let outerModel = try ScriptedModel()
        let outerAgent = makeAgent(model: outerModel, tools: [outerSharedTool, agentTool], name: "Outer")
        try outerModel.extend([
            ModelStep(output: [makeFunctionToolCall(outerSharedTool.name, callId: "dup")]),
            ModelStep(output: [makeFunctionToolCall(agentTool.name, callId: "outer-agent", arguments: #"{"input":"hi"}"#)]),
        ])

        let first = try await Runner.run(outerAgent, input: "start")
        #expect(!first.interruptions.isEmpty, "parent tool should request approval first")

        let approvedState = first.toState()
        try approvedState.approve(try #require(first.interruptions.first))

        let second = try await Runner.run(outerAgent, state: approvedState)
        #expect(!second.interruptions.isEmpty, "nested tool should still require approval")
        #expect(
            second.interruptions.contains { $0.toolName == innerSharedTool.name },
            "inner tool approvals should not inherit parent approvals"
        )
    }

    @Test func testAgentAsToolWithNestedApprovalsPropagates() async throws {
        let (nestedModel, spanishAgent) = try makeModelAndAgent(name: "spanish_agent")
        let toolCalls = SharedBox<[String]>([])
        let getCurrentTimestamp = try FunctionTool.create(name: "get_current_timestamp", needsApproval: .always) { _ in
            toolCalls.mutate { $0.append("called") }
            return "timestamp"
        }
        spanishAgent.tools = [getCurrentTimestamp]
        try nestedModel.extend([
            ModelStep(output: [makeFunctionToolCall("get_current_timestamp")]),
            ModelStep(output: [getTextMessage("hola")]),
        ])
        let orchestratorModel = try ScriptedModel()
        let orchestrator = makeAgent(
            model: orchestratorModel,
            tools: [try spanishAgent.asTool(toolName: "respond_spanish", toolDescription: "Respond in Spanish", needsApproval: .always)],
            name: "orchestrator"
        )
        try orchestratorModel.extend([
            ModelStep(output: [makeFunctionToolCall("respond_spanish", callId: "spanish-call", arguments: #"{"input": "hola"}"#)]),
            ModelStep(output: [getTextMessage("done")]),
        ])

        let first = try await Runner.run(orchestrator, input: "hola")
        #expect(!first.interruptions.isEmpty, "Outer agent tool should require approval")

        let state = try approveFirstInterruption(first, alwaysApprove: true)
        let resumed = try await Runner.run(orchestrator, state: state)
        #expect(!resumed.interruptions.isEmpty, "Nested agent tool approval should bubble up")
        #expect(resumed.interruptions.first?.toolName == "get_current_timestamp")
        _ = try resumed.toInputList()
        #expect(toolCalls.value.isEmpty, "Nested tool should not execute before approval")

        let finalState = try approveFirstInterruption(resumed, alwaysApprove: true)
        let final = try await Runner.run(orchestrator, state: finalState)
        #expect(final.finalOutput as? String == "done")
        #expect(toolCalls.value == ["called"])
    }

    /// Counts the outer agent tool's callbacks.
    private final class CountingHooks: RunHooks, @unchecked Sendable {
        let counts: SharedBox<[String: Int]>

        init(counts: SharedBox<[String: Int]>) {
            self.counts = counts
        }

        override func onToolStart(context: RunContextWrapper, agent: AgentBase, tool: any Tool) async throws {
            if tool.name == "delegate" { counts.mutate { $0["start", default: 0] += 1 } }
        }

        override func onToolEnd(context: RunContextWrapper, agent: AgentBase, tool: any Tool, result: any Sendable) async throws {
            if tool.name == "delegate" { counts.mutate { $0["end", default: 0] += 1 } }
        }
    }

    @Test(arguments: [false, true])
    func testNestedAgentToolContinuationRunsOuterCallbacksOnce(streamed: Bool) async throws {
        let (nestedModel, nestedAgent) = try makeModelAndAgent(name: "nested_agent")
        let innerCalls = SharedBox<[String]>([])
        let counts = SharedBox<[String: Int]>([:])
        @Sendable func bump(_ key: String) { counts.mutate { $0[key, default: 0] += 1 } }

        let innerTool = try FunctionTool.create(name: "inner_tool", needsApproval: .always) { _ in
            innerCalls.mutate { $0.append("called") }
            return "inner output"
        }
        nestedAgent.tools = [innerTool]
        try nestedModel.extend([
            ModelStep(output: [makeFunctionToolCall("inner_tool", callId: "inner-call-\(streamed)")]),
            ModelStep(output: [getTextMessage("nested done")]),
        ])

        let outerTool = try nestedAgent.asTool(
            toolName: "delegate",
            toolDescription: "Delegate to the nested agent",
            customOutputExtractor: { result in
                bump("custom_output")
                return result.finalOutput as? String ?? ""
            }
        )
        let trackInput = ToolInputGuardrail(name: "track_input") { _ in
            bump("input_guardrail")
            return .allow()
        }
        let trackOutput = ToolOutputGuardrail(name: "track_output") { _ in
            bump("output_guardrail")
            return .allow()
        }
        outerTool.toolInputGuardrails = [trackInput]
        outerTool.toolOutputGuardrails = [trackOutput]
        outerTool.customDataExtractor = { _ in
            bump("extractor")
            return ["nested": true]
        }

        let outerModel = try ScriptedModel()
        try outerModel.extend([
            ModelStep(output: [makeFunctionToolCall("delegate", callId: "outer-call-\(streamed)", arguments: #"{"input":"hello"}"#)]),
            ModelStep(output: [getTextMessage("outer done")]),
        ])
        let outerAgent = makeAgent(model: outerModel, tools: [outerTool], name: "outer_agent")
        let hooks = CountingHooks(counts: counts)

        func run(_ input: RunInput) async throws -> any RunResultBase & RunStateConvertible {
            guard streamed else { return try await Runner.run(outerAgent, input: input, hooks: hooks) }
            let result = try Runner.runStreamed(outerAgent, input: input, hooks: hooks)
            try await consumeStream(result)
            return result
        }
        func resume(_ state: RunState) async throws -> any RunResultBase & RunStateConvertible {
            guard streamed else { return try await Runner.run(outerAgent, state: state, hooks: hooks) }
            let result = try Runner.runStreamed(outerAgent, state: state, hooks: hooks)
            try await consumeStream(result)
            return result
        }

        let interrupted = try await run("start")
        #expect(!interrupted.interruptions.isEmpty)
        #expect(counts.value["input_guardrail", default: 0] <= 1)
        #expect(counts.value["start", default: 0] <= 1)
        #expect(counts.value["output_guardrail", default: 0] == 0)
        #expect(counts.value["custom_output", default: 0] == 0)
        #expect(counts.value["extractor", default: 0] == 0)
        #expect(counts.value["end", default: 0] == 0)

        let state = interrupted.toState()
        try state.approve(try #require(state.getInterruptions().first))
        let restored = try await RunState.fromJSON(outerAgent, try state.toJSON())
        let final = try await resume(restored)

        #expect(final.finalOutput as? String == "outer done")
        #expect(innerCalls.value == ["called"])
        #expect(counts.value == [
            "input_guardrail": 1,
            "start": 1,
            "output_guardrail": 1,
            "custom_output": 1,
            "extractor": 1,
            "end": 1,
        ])
    }
}
