import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_state.py (classes TestDeserializeHelpers and TestRunStateResumption).
//
// Swift: nested agent-tool run state (`agent_tool_state.py`) and approval ambiguity checks across
// nested states are not ported yet. The tests that depend on them (`test_ambiguous_*`,
// `test_unsafe_*`, `test_uncopyable_*`, `test_safe_noncanonical_*`, `test_exact_uncopyable_*`,
// `test_completed_current_invocation_does_not_own_nested_approval`,
// `test_native_current_and_nested_approval_identity_fails_closed`, `test_*nested_agent_tool_*`
// except the plain round trip, and
// `test_nested_resume_checkpoint_to_state_keeps_its_owned_decision_ledger`) are pending in
// porting/manifest.json. `test_result_to_state_detaches_tool_decision_ledgers` is in
// RunStateTests+InvocationLedger.swift.

@Suite struct RunStateDeserializeHelpersTests {
    @Test func testSerializationIncludesHandoffFields() async throws {
        let agentA = Agent(name: "AgentA")
        let agentB = Agent(name: "AgentB")
        agentA.handoffs = [.agent(agentB)]
        let state = makeState(agentA, context: RunContextWrapper(context: JSONObject()), originalInput: "test handoff", maxTurns: 2)
        state.generatedItems.append(
            HandoffOutputItem(agent: agentB, rawItem: ["type": "handoff_output", "status": "completed"], sourceAgent: agentA, targetAgent: agentB)
        )

        let jsonData = try state.toJSON()
        let items = try #require(jsonData["generated_items"]?.arrayValue)
        #expect(items.count == 1)
        let itemData = try #require(items[0].objectValue)
        #expect(itemData["source_agent"]?.objectValue?["name"] == "AgentA")
        #expect(itemData["target_agent"]?.objectValue?["name"] == "AgentB")

        let restored = try await RunState.fromString(agentA, try state.toString())
        #expect(restored.generatedItems.count == 1)
        #expect(restored.generatedItems[0].type == "handoff_output_item")
    }

    @Test func testModelResponseSerializationRoundtrip() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test", maxTurns: 2)
        state.modelResponses.append(ModelResponse(
            output: [makeMessageOutput(messageId: "msg1", text: "Hello")],
            usage: Usage(requests: 1, inputTokens: 10, outputTokens: 20, totalTokens: 30),
            responseId: "resp123",
            requestId: "req123",
            rawUsage: ["input_tokens": 10, "provider_metric": 0]
        ))

        let serialized = try state.toJSON()
        #expect(serialized["model_responses"]?.arrayValue?[0].objectValue?["raw_usage"] == nil)
        let restored = try await RunState.fromString(agent, try state.toString())

        #expect(restored.modelResponses.count == 1)
        #expect(restored.modelResponses[0].responseId == "resp123")
        #expect(restored.modelResponses[0].requestId == "req123")
        #expect(restored.modelResponses[0].rawUsage == nil)
        #expect(restored.modelResponses[0].usage.requests == 1)
        #expect(restored.modelResponses[0].usage.inputTokens == 10)
    }

    @Test func testInterruptionsSerializationRoundtrip() async throws {
        let agent = Agent(name: "InterruptAgent")
        let rawItem = ResponseFunctionToolCall(id: "1", callId: "call789", name: "sensitive_tool", arguments: #"{"data": "value"}"#, status: "completed").json
        let state = makeStateWithInterruptions(agent, [ToolApprovalItem(agent: agent, rawItem: rawItem)], originalInput: "test", maxTurns: 2)

        let restored = try await RunState.fromString(agent, try state.toString())

        guard case .interruption(let interruptions, _)? = restored.currentStep else {
            Issue.record("expected an interruption step")
            return
        }
        #expect(interruptions.count == 1)
        #expect(interruptions[0].rawItem["name"] == "sensitive_tool")
    }

    @Test func testNestedAgentToolInterruptionsRoundtrip() async throws {
        let innerAgent = Agent(name: "InnerAgent")
        let outerAgent = Agent(name: "OuterAgent")
        outerAgent.tools = [try innerAgent.asTool(toolName: "inner_agent_tool", toolDescription: "Inner agent tool", needsApproval: .always)]
        let approvalItem = ToolApprovalItem(agent: innerAgent, rawItem: makeFunctionToolCall("sensitive_tool", callId: "inner-1"))
        let state = makeStateWithInterruptions(outerAgent, [approvalItem], originalInput: "test", maxTurns: 2)

        let restored = try await RunState.fromString(outerAgent, try state.toString())

        let interruptions = restored.getInterruptions()
        #expect(interruptions.count == 1)
        #expect(interruptions[0].agent.name == "InnerAgent")
        #expect(interruptions[0].rawItem["name"] == "sensitive_tool")
    }

    @Test func testJSONDecodeErrorHandling() async {
        let agent = Agent(name: "TestAgent")
        let sentinel = "malformed-json-secret"

        let error = await expectErrorAsync(UserError.self, containing: "Failed to parse run state JSON") {
            _ = try await RunState.fromString(agent, #"{ "sandbox": "\#(sentinel)" "#)
        }

        // Swift: errors carry no traceback, so only the message is checked for the payload.
        #expect(error.map { !describeError($0).contains(sentinel) } == true)
    }

    @Test func testMissingAgentInMapError() async throws {
        let agentA = Agent(name: "AgentA")
        let state = makeState(agentA, context: RunContextWrapper(context: JSONObject()), originalInput: "test", maxTurns: 2)
        let jsonString = try state.toString()

        let agentB = Agent(name: "AgentB")
        await expectErrorAsync(UserError.self, containing: "Run state agent not found in agent map") {
            _ = try await RunState.fromString(agentB, jsonString)
        }
    }
}

@Suite struct RunStateResumptionTests {
    private struct ApprovalArgs: Decodable, Sendable { var value: String }

    @Test(arguments: [false, true], [true, false])
    func testResumeExecutesOnlyExactOverrideResult(streamed: Bool, stickyApproved: Bool) async throws {
        let executions = SharedBox<[String]>([])
        let approvalTool = try FunctionTool.create(name: "approval_tool", argumentsType: ApprovalArgs.self, needsApproval: true) { _, args in
            executions.mutate { $0.append(args.value) }
            return "approved:\(args.value)"
        }
        let model = try ScriptedModel(steps: [
            .output([
                getFunctionToolCall("approval_tool", arguments: #"{"value": "sticky"}"#, callId: "sticky-call"),
                getFunctionToolCall("approval_tool", arguments: #"{"value": "exception"}"#, callId: "exception-call"),
            ]),
            .output([getFinalOutputMessage("done")]),
        ])
        let agent = Agent(name: "MixedApprovalAgent", tools: [approvalTool], model: .instance(model))

        let initial = try await Runner.run(agent, input: "start")
        let state = initial.toState()
        var interruptions: [String: ToolApprovalItem] = [:]
        for interruption in state.getInterruptions() {
            interruptions[interruption.callId ?? ""] = interruption
        }
        if stickyApproved {
            try state.approve(try #require(interruptions["sticky-call"]), alwaysApprove: true)
            try state.reject(try #require(interruptions["exception-call"]), rejectionMessage: "denied exactly")
        } else {
            try state.reject(try #require(interruptions["sticky-call"]), alwaysReject: true, rejectionMessage: "denied by default")
            try state.approve(try #require(interruptions["exception-call"]))
        }

        let restored = try await RunState.fromString(agent, try state.toString())
        let finalOutput: (any Sendable)?
        let resumedInterruptions: [ToolApprovalItem]
        if streamed {
            let resumed = try Runner.runStreamed(agent, state: restored)
            for try await _ in resumed.streamEvents() {}
            finalOutput = resumed.finalOutput
            resumedInterruptions = resumed.interruptions
        } else {
            let resumed = try await Runner.run(agent, state: restored)
            finalOutput = resumed.finalOutput
            resumedInterruptions = resumed.interruptions
        }

        #expect(finalOutput as? String == "done")
        #expect(resumedInterruptions.isEmpty)
        #expect(executions.value == (stickyApproved ? ["sticky"] : ["exception"]))
    }

    @Test func testResumeFromRunState() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result1 = try await Runner.run(agent, input: "First input")

        let state = result1.toState()
        try model.enqueue(.output([getTextMessage("Second response")]))
        let result2 = try await Runner.run(agent, state: state)

        #expect(result2.finalOutput as? String == "Second response")
    }

    @Test func testResumeFromRunStateDoesNotMutateSourceResult() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result1 = try await Runner.run(agent, input: "First input")
        #expect(result1.rawResponses.count == 1)

        let state = result1.toState()
        try model.enqueue(.output([getTextMessage("Second response")]))
        let result2 = try await Runner.run(agent, state: state)

        #expect(result2.rawResponses.count == 2)
        #expect(result1.rawResponses.count == 1)
    }

    @Test func testResumeFromRunStateDoesNotMutateSourceResultUsage() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result1 = try await Runner.run(agent, input: "First input")
        let requestsAfterFirstRun = result1.contextWrapper.usage.requests

        let state = result1.toState()
        try model.enqueue(.output([getTextMessage("Second response")]))
        let result2 = try await Runner.run(agent, state: state)

        #expect(result2.contextWrapper.usage.requests > requestsAfterFirstRun)
        #expect(result1.contextWrapper.usage.requests == requestsAfterFirstRun)
        #expect(result1.contextWrapper.usageBox !== result2.contextWrapper.usageBox)
    }

    @Test func testTwoCheckpointsFromOneResultDoNotShareUsage() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result = try await Runner.run(agent, input: "First input")

        let firstCheckpoint = result.toState()
        let secondCheckpoint = result.toState()

        try model.enqueue(.output([getTextMessage("Second response")]))
        let firstResume = try await Runner.run(agent, state: firstCheckpoint)
        try model.enqueue(.output([getTextMessage("Third response")]))
        let secondResume = try await Runner.run(agent, state: secondCheckpoint)

        #expect(firstResume.contextWrapper.usage.requests == 2)
        #expect(secondResume.contextWrapper.usage.requests == 2)
    }

    @Test func testResumeDoesNotAppendToTheStateItResumedFrom() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result1 = try await Runner.run(agent, input: "First input")
        let state = result1.toState()
        let serializedBefore = try state.toJSON()["model_responses"]

        try model.enqueue(.output([getTextMessage("Second response")]))
        let result2 = try await Runner.run(agent, state: state)
        #expect(result2.rawResponses.count == 2)

        // The state is a snapshot of the first turn, so the second run's response must not land
        // in it, neither in memory nor in the serialized snapshot.
        #expect(state.modelResponses.count == 1)
        #expect(try state.toJSON()["model_responses"] == serializedBefore)

        // Re-running the same checkpoint therefore replays only its own history.
        try model.enqueue(.output([getTextMessage("Third response")]))
        let result3 = try await Runner.run(agent, state: state)
        #expect(result3.rawResponses.count == 2)
    }

    @Test func testStreamedResumeDoesNotAppendToTheStateItResumedFrom() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result1 = try await Runner.run(agent, input: "First input")
        let state = result1.toState()
        let serializedBefore = try state.toJSON()["session_items"]

        try model.enqueue(.output([getTextMessage("Second response")]))
        let result2 = try Runner.runStreamed(agent, state: state)
        for try await _ in result2.streamEvents() {}
        #expect(result2.newItems.count == 2)

        #expect(state.sessionItems.count == 1)
        #expect(try state.toJSON()["session_items"] == serializedBefore)

        // Without this, the abandoned attempt's message leaks into the replayed history.
        try model.enqueue(.output([getTextMessage("Third response")]))
        let result3 = try Runner.runStreamed(agent, state: state)
        for try await _ in result3.streamEvents() {}
        #expect(result3.newItems.count == 2)
        #expect(try result3.toInputList().count == 3)
    }

    @Test func testResumedMaxTurnsHandlerDoesNotAppendToStateItems() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result1 = try await Runner.run(agent, input: "First input", maxTurns: 1)
        let state = result1.toState()
        let serializedBefore = try state.toJSON()["generated_items"]

        let handlers = RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: "fallback") })
        let result2 = try await Runner.run(agent, state: state, errorHandlers: handlers)
        #expect(result2.finalOutput as? String == "fallback")

        #expect(state.generatedItems.count == 1)
        #expect(try state.toJSON()["generated_items"] == serializedBefore)
    }

    @Test func testFreshRunsStillReportTheirOwnHistory() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result1 = try await Runner.run(agent, input: "First input")
        #expect(result1.rawResponses.count == 1)
        #expect(result1.newItems.count == 1)

        try model.enqueue(.output([getTextMessage("Streamed response")]))
        let result2 = try Runner.runStreamed(agent, input: "Second input")
        for try await _ in result2.streamEvents() {}
        #expect(result2.rawResponses.count == 1)
        #expect(result2.newItems.count == 1)
    }

    @Test func testResumeFromRunStateWithContext() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result1 = try await Runner.run(agent, input: "First input", context: ["key": "value1"] as JSONObject)

        let state = result1.toState()
        let context2: JSONObject = ["key": "value2"]
        try model.enqueue(.output([getTextMessage("Second response")]))
        let result2 = try await Runner.run(agent, state: state, context: context2)

        #expect(result2.finalOutput as? String == "Second response")
        #expect(result2.contextWrapper.context(as: JSONObject.self) == context2)
        #expect(state.context.context(as: JSONObject.self) == context2)
    }

    @Test func testResumeFromRunStateWithConversationId() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result1 = try await Runner.run(agent, input: "First input", conversationId: "conv123")

        let state = result1.toState()
        try model.enqueue(.output([getTextMessage("Second response")]))
        let result2 = try await Runner.run(agent, state: state, conversationId: "conv123")

        #expect(result2.finalOutput as? String == "Second response")
    }

    @Test func testResumeFromRunStateWithPreviousResponseId() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result1 = try await Runner.run(agent, input: "First input", previousResponseId: "resp123")

        let state = result1.toState()
        try model.enqueue(.output([getTextMessage("Second response")]))
        let result2 = try await Runner.run(agent, state: state, previousResponseId: "resp123")

        #expect(result2.finalOutput as? String == "Second response")
    }

    @Test func testResumeFromRunStateWithInterruption() async throws {
        let model = try ScriptedModel()
        let tool = try FunctionTool.create(name: "test_tool") { _ in "tool_result" }
        let agent = Agent(name: "TestAgent", tools: [tool], model: .instance(model))
        try model.enqueue(.output([getFunctionToolCall("test_tool", arguments: "{}")]))
        try model.enqueue(.output([]))
        let result1 = try await Runner.run(agent, input: "First input")

        let state = result1.toState()
        if let interruption = state.getInterruptions().first {
            try state.approve(interruption)
        }

        try model.enqueue(.output([getTextMessage("Second response")]))
        let result2 = try await Runner.run(agent, state: state)

        #expect(result2.finalOutput as? String == "Second response")
    }

    @Test func testResumeFromRunStateStreamed() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "TestAgent", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let result1 = try await Runner.run(agent, input: "First input")

        let state = result1.toState()
        try model.enqueue(.output([getTextMessage("Second response")]))
        let result2 = try Runner.runStreamed(agent, state: state)
        for try await _ in result2.streamEvents() {}

        #expect(result2.finalOutput as? String == "Second response")
    }

    @Test func testResumeFromRunStateStreamedUsesContextFromState() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("done")])])
        let agent = Agent(name: "TestAgent", model: .instance(model))
        let state = makeState(agent, context: RunContextWrapper(context: ["key": "value"] as JSONObject), originalInput: "test", maxTurns: 1)

        let result = try Runner.runStreamed(agent, state: state)
        for try await _ in result.streamEvents() {}

        #expect(result.finalOutput as? String == "done")
    }

    @Test func testResumeFromRunStateStreamedWithContextOverride() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("done")])])
        let agent = Agent(name: "TestAgent", model: .instance(model))
        let state = makeState(agent, context: RunContextWrapper(context: ["key": "value1"] as JSONObject), originalInput: "test", maxTurns: 1)

        let overrideContext: JSONObject = ["key": "value2"]
        let result = try Runner.runStreamed(agent, state: state, context: overrideContext)
        for try await _ in result.streamEvents() {}

        #expect(result.finalOutput as? String == "done")
        #expect(result.contextWrapper.context(as: JSONObject.self) == overrideContext)
    }

    @Test func testRunResultStreamingToStateWithInterruptions() async throws {
        let model = try ScriptedModel(steps: [
            .output([getFunctionToolCall("test_tool", arguments: "{}")]),
            .output([getTextMessage("done")]),
        ])
        let tool = try FunctionTool.create(name: "test_tool", needsApproval: true) { _ in "result" }
        let agent = Agent(name: "TestAgent", tools: [tool], model: .instance(model))

        let result = try Runner.runStreamed(agent, input: "test")
        for try await _ in result.streamEvents() {}
        #expect(!result.interruptions.isEmpty)

        let state = result.toState()
        guard case .interruption(let interruptions, _)? = state.currentStep else {
            Issue.record("expected an interruption step")
            return
        }
        #expect(interruptions.count == result.interruptions.count)
    }
}
