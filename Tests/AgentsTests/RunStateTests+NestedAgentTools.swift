import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_state.py: the nested agent-tool state cases of TestDeserializeHelpers
// and TestRunStateResumption, and the module-level nested `Agent.asTool()` resume tests.
//
// Swift: upstream records pending nested states without a scope. Swift Testing runs cases in
// parallel, so these tests give each state its own scope instead of sharing the unscoped bucket.
// Not ported: `test_unsafe_*`, `test_uncopyable_*`, `test_safe_noncanonical_*`, and
// `test_exact_uncopyable_*` build approvals whose raw items Python cannot copy (an `object()`
// field or a broken `__deepcopy__`). Swift raw items are JSON values and always copy.

/// Records a serializable nested interruption for an outer function call (upstream
/// `record_pending_nested_agent_tool_state`).
private func recordPendingNestedAgentToolState(
    _ agent: Agent,
    _ toolCall: ResponseFunctionToolCall,
    innerCallId: String,
    scope: AgentToolStateScope?
) {
    let nestedApproval = makeToolApprovalItem(agent, callId: innerCallId, name: "inner_sensitive_tool")
    let nestedState = makeStateWithInterruptions(agent, [nestedApproval], originalInput: "nested input for \(innerCallId)")
    recordAgentToolRunResult(toolCall.json, .serialized(nestedState, interruptions: nestedState.getInterruptions()), scope: scope)
}

/// Creates a context whose run state gets a private agent-tool scope.
private func scopedContext() -> RunContextWrapper {
    let context = RunContextWrapper(context: JSONObject())
    setAgentToolStateScope(context, AgentToolStateScope())
    return context
}

/// Returns whether a model call input already carries a tool output.
private func hasFunctionCallOutput(_ input: RunInput) -> Bool {
    input.inputItems.contains { $0.type == "function_call_output" }
}

/// Requests the given tool calls, then finishes once a tool output is present in the input
/// (upstream `make_tool_calls_then_final_model`).
private func makeToolCallsThenFinalModel(
    toolCalls: [(name: String, arguments: String)],
    finalText: String,
    callPrefix: String,
    expectedCalls: Int = 2
) throws -> ScriptedModel {
    let step = ModelStep.respond { call in
        if hasFunctionCallOutput(call.input) {
            return .step(.output([getTextMessage(finalText)], responseId: "\(callPrefix)-done"))
        }
        return .step(.output(
            toolCalls.map { makeFunctionToolCall($0.name, callId: "\(callPrefix)-\($0.name)", arguments: $0.arguments) },
            responseId: "\(callPrefix)-call"
        ))
    }
    return try ScriptedModel(steps: Array(repeating: step, count: expectedCalls))
}

private struct TextArgs: Codable, Sendable { var text: String }

/// Removes `call_id` from a serialized function action's tool call.
private func removeCallId(_ stateJSON: inout JSONObject, functionIndex: Int) throws {
    var processed = try #require(stateJSON["last_processed_response"]?.objectValue)
    var functions = try #require(processed["functions"]?.arrayValue)
    var entry = try #require(functions[functionIndex].objectValue)
    var toolCall = try #require(entry["tool_call"]?.objectValue)
    toolCall.removeValue(forKey: "call_id")
    entry["tool_call"] = .object(toolCall)
    functions[functionIndex] = .object(entry)
    processed["functions"] = .array(functions)
    stateJSON["last_processed_response"] = .object(processed)
}

// MARK: - Approval ownership

extension RunStateDeserializeHelpersTests {
    @Test(arguments: [false, true])
    func testAmbiguousCurrentAndNestedApprovalIdentityFailsClosed(roundTrip: Bool) async throws {
        let agent = Agent(name: "Agent")
        let sensitiveTool = try FunctionTool.create(name: "sensitive") { _ in "sensitive" }
        let nestedTool = try FunctionTool.create(name: "nested_agent_tool") { _ in "nested" }
        agent.tools = [sensitiveTool, nestedTool]

        let currentCall = makeToolCall("shared", name: "sensitive")
        let nestedOuterCall = makeToolCall("outer-nested", name: "nested_agent_tool")
        let currentApproval = ToolApprovalItem(agent: agent, rawItem: currentCall.json)
        let nestedApproval = ToolApprovalItem(agent: agent, rawItem: currentCall.json)
        let state = makeStateWithInterruptions(agent, [currentApproval, nestedApproval])
        setAgentToolStateScope(state.context, AgentToolStateScope())
        state.agentToolStateScope = getAgentToolStateScope(state.context)
        state.lastProcessedResponse = makeProcessedResponse(functions: [
            ToolRunFunction(toolCall: currentCall, functionTool: sensitiveTool),
            ToolRunFunction(toolCall: nestedOuterCall, functionTool: nestedTool),
        ])
        try state.context.toolInvocationStatus(currentCall.json)

        let nestedState = makeStateWithInterruptions(agent, [nestedApproval])
        recordAgentToolRunResult(nestedOuterCall.json, .serialized(nestedState, interruptions: [nestedApproval]), scope: state.agentToolStateScope)

        var targetState = state
        if roundTrip {
            targetState = try await RunState.fromJSON(agent, try state.toJSON())
        }

        expectError(UserError.self, containing: "current run and a nested agent-tool run") {
            try targetState.approve(targetState.getInterruptions()[0])
        }
    }

    @Test(arguments: ["snapshot", "exact"], [true, false])
    func testAmbiguousCurrentApprovalIdentityFailsClosed(approvalInput: String, approve: Bool) throws {
        let agent = Agent(name: "AmbiguousCurrentAgent")
        let rawItem = ResponseFunctionToolCall(callId: "shared-current", name: "toolA", arguments: "{}").json
        let first = ToolApprovalItem(
            agent: agent, rawItem: rawItem, toolName: "toolA", toolLookupKey: .deferredTopLevel("toolA"), allowBareNameAlias: true
        )
        let second = ToolApprovalItem(
            agent: agent, rawItem: rawItem, toolName: "toolA", toolLookupKey: .deferredTopLevel("toolA"), allowBareNameAlias: false
        )
        let state = makeStateWithInterruptions(agent, [first, second])

        let selected = approvalInput == "snapshot" ? state.getInterruptions()[1] : second
        expectError(UserError.self, containing: "multiple current pending approvals") {
            if approve {
                try state.approve(selected)
            } else {
                try state.reject(selected)
            }
        }

        #expect(state.context.isToolApproved(toolName: "toolA", callId: "shared-current") == nil)
    }

    // Swift: only the JSON-safe `unsafe_metadata=False` case; see the file header.
    @Test(arguments: [true, false])
    func testAmbiguousExactNestedApprovalIdentityFailsClosed(approve: Bool) throws {
        let agent = Agent(name: "AmbiguousNestedAgent")
        let rawItem = ResponseFunctionToolCall(callId: "shared-nested", name: "toolA", arguments: "{}").json
        let first = ToolApprovalItem(agent: agent, rawItem: rawItem, toolName: "toolA")
        let second = ToolApprovalItem(agent: agent, rawItem: rawItem, toolName: "toolA")
        let nestedState = makeStateWithInterruptions(agent, [first, second])
        let outerState = makeStateWithInterruptions(agent, [first])
        outerState.agentToolStateScope = AgentToolStateScope()
        let outerCall = makeToolCall("outer-ambiguous-nested", name: "nested_agent_tool")
        let nestedTool = try FunctionTool.create(name: "nested_agent_tool") { _ in "nested" }
        outerState.lastProcessedResponse = makeProcessedResponse(functions: [ToolRunFunction(toolCall: outerCall, functionTool: nestedTool)])
        recordAgentToolRunResult(outerCall.json, .serialized(nestedState, interruptions: [first]), scope: outerState.agentToolStateScope)

        expectError(UserError.self, containing: "multiple current pending approvals") {
            if approve {
                try outerState.approve(first)
            } else {
                try outerState.reject(first)
            }
        }
        #expect(nestedState.context.isToolApproved(toolName: "toolA", callId: "shared-nested") == nil)
    }

    @Test(arguments: [false, true], [true, false])
    func testCompletedCurrentInvocationDoesNotOwnNestedApproval(roundTrip: Bool, approve: Bool) async throws {
        let agent = Agent(name: "Agent")
        let sensitiveTool = try FunctionTool.create(name: "sensitive") { _ in "sensitive" }
        let nestedTool = try FunctionTool.create(name: "nested_agent_tool") { _ in "nested" }
        agent.tools = [sensitiveTool, nestedTool]

        let completedCall = makeToolCall("shared", name: "sensitive")
        let nestedOuterCall = makeToolCall("outer-nested", name: "nested_agent_tool")
        let nestedApproval = ToolApprovalItem(agent: agent, rawItem: completedCall.json)
        let state = makeStateWithInterruptions(agent, [nestedApproval])
        state.agentToolStateScope = AgentToolStateScope()
        state.lastProcessedResponse = makeProcessedResponse(functions: [
            ToolRunFunction(toolCall: completedCall, functionTool: sensitiveTool),
            ToolRunFunction(toolCall: nestedOuterCall, functionTool: nestedTool),
        ])
        try state.context.toolInvocationStatus(completedCall.json)
        let completedOutput: JSONObject = ["type": "function_call_output", "call_id": .string(completedCall.callId), "output": "done"]
        state.context.markToolCallCompleted(completedOutput)
        state.generatedItems = [
            ToolCallItem(agent: agent, rawItem: completedCall.json),
            ToolCallOutputItem(agent: agent, rawItem: completedOutput, output: "done"),
        ]

        let nestedState = makeStateWithInterruptions(agent, [nestedApproval])
        recordAgentToolRunResult(nestedOuterCall.json, .serialized(nestedState, interruptions: [nestedApproval]), scope: state.agentToolStateScope)

        var targetState = state
        var targetNestedCall = nestedOuterCall
        if roundTrip {
            targetState = try await RunState.fromJSON(agent, try state.toJSON())
            targetNestedCall = try #require(targetState.lastProcessedResponse?.functions[1].toolCall)
        }

        let targetApproval = targetState.getInterruptions()[0]
        if approve {
            try targetState.approve(targetApproval)
        } else {
            try targetState.reject(targetApproval)
        }

        let pendingResult = try #require(peekAgentToolRunResult(targetNestedCall.json, scope: targetState.agentToolStateScope))
        let targetNestedState = pendingResult.toState()
        #expect(try targetNestedState.context.getApprovalStatus(toolName: "sensitive", callId: "shared", existingPending: targetApproval) == approve)
    }

    @Test(arguments: [false, true], [true, false])
    func testNativeCurrentAndNestedApprovalIdentityFailsClosed(roundTrip: Bool, approve: Bool) async throws {
        let agent = Agent(name: "Agent")
        let shellTool = try ShellTool(executor: { _ in .text("done") }, needsApproval: .required(true))
        let nestedTool = try FunctionTool.create(name: "nested_agent_tool") { _ in "nested" }
        agent.tools = [shellTool, nestedTool]

        let currentCall = makeShellCall("shared")
        let nestedOuterCall = makeToolCall("outer-nested", name: "nested_agent_tool")
        let currentApproval = ToolApprovalItem(agent: agent, rawItem: currentCall, toolName: shellTool.name)
        let nestedApproval = ToolApprovalItem(agent: agent, rawItem: currentCall, toolName: shellTool.name)
        let state = makeStateWithInterruptions(agent, [currentApproval, nestedApproval])
        state.agentToolStateScope = AgentToolStateScope()
        state.lastProcessedResponse = makeProcessedResponse(
            functions: [ToolRunFunction(toolCall: nestedOuterCall, functionTool: nestedTool)],
            shellCalls: [ToolRunShellCall(toolCall: currentCall, shellTool: shellTool)]
        )
        try state.context.toolInvocationStatus(currentCall, toolName: shellTool.name)

        let nestedState = makeStateWithInterruptions(agent, [nestedApproval])
        recordAgentToolRunResult(nestedOuterCall.json, .serialized(nestedState, interruptions: [nestedApproval]), scope: state.agentToolStateScope)

        var targetState = state
        if roundTrip {
            targetState = try await RunState.fromJSON(agent, try state.toJSON())
        }

        expectError(UserError.self, containing: "current run and a nested agent-tool run") {
            if approve {
                try targetState.approve(targetState.getInterruptions()[0])
            } else {
                try targetState.reject(targetState.getInterruptions()[0])
            }
        }
    }
}

// MARK: - Serialized nested states

extension RunStateDeserializeHelpersTests {
    @Test(arguments: ["disabled", "removed", "malformed_call"])
    func testNestedAgentToolStateSurvivesWhenEarlierFunctionIsDropped(dropMode: String) async throws {
        let context = scopedContext()
        let agent = Agent(name: "OuterAgent")
        let earlierToolEnabled = SharedBox(true)
        let conditionalTool = try FunctionTool.create(name: "conditional_tool", isEnabled: .dynamic { _, _ in earlierToolEnabled.value }) { _ in
            "conditional"
        }
        let nestedTool = try FunctionTool.create(name: "nested_agent_tool") { _ in "nested" }
        agent.tools = [conditionalTool, nestedTool]

        let conditionalCall = makeToolCall("conditional-call", name: "conditional_tool")
        let nestedCall = makeToolCall("nested-call", name: "nested_agent_tool")
        let state = makeState(agent, context: context)
        state.lastProcessedResponse = makeProcessedResponse(functions: [
            ToolRunFunction(toolCall: conditionalCall, functionTool: conditionalTool),
            ToolRunFunction(toolCall: nestedCall, functionTool: nestedTool),
        ])
        recordPendingNestedAgentToolState(agent, nestedCall, innerCallId: "inner-call", scope: state.agentToolStateScope)

        var stateJSON = try state.toJSON()
        switch dropMode {
        case "disabled": earlierToolEnabled.value = false
        case "removed": agent.tools = [nestedTool]
        default: try removeCallId(&stateJSON, functionIndex: 0)
        }

        let restored = try await RunState.fromJSON(agent, stateJSON)

        let functions = try #require(restored.lastProcessedResponse?.functions)
        #expect(functions.count == 1)
        let restoredCall = try #require(functions.first?.toolCall)
        #expect(restoredCall.callId == "nested-call")
        let pendingResult = try #require(peekAgentToolRunResult(restoredCall.json, scope: restored.agentToolStateScope))
        #expect(pendingResult.interruptions.count == 1)
        #expect(pendingResult.interruptions.first?.rawItem.string("call_id") == "inner-call")
    }

    @Test func testDroppedNestedAgentToolStateIsNotMovedToLaterFunction() async throws {
        let context = scopedContext()
        let agent = Agent(name: "OuterAgent")
        let droppedTool = try FunctionTool.create(name: "dropped_agent_tool") { _ in "dropped" }
        let retainedTool = try FunctionTool.create(name: "retained_tool") { _ in "retained" }
        agent.tools = [droppedTool, retainedTool]

        let droppedCall = makeToolCall("dropped-call", name: "dropped_agent_tool")
        let retainedCall = makeToolCall("retained-call", name: "retained_tool")
        let state = makeState(agent, context: context)
        state.lastProcessedResponse = makeProcessedResponse(functions: [
            ToolRunFunction(toolCall: droppedCall, functionTool: droppedTool),
            ToolRunFunction(toolCall: retainedCall, functionTool: retainedTool),
        ])
        recordPendingNestedAgentToolState(agent, droppedCall, innerCallId: "dropped-inner-call", scope: state.agentToolStateScope)

        let stateJSON = try state.toJSON()
        agent.tools = [retainedTool]

        let restored = try await RunState.fromJSON(agent, stateJSON)

        let functions = try #require(restored.lastProcessedResponse?.functions)
        #expect(functions.count == 1)
        let restoredCall = try #require(functions.first?.toolCall)
        #expect(restoredCall.callId == "retained-call")
        #expect(peekAgentToolRunResult(restoredCall.json, scope: restored.agentToolStateScope) == nil)
    }

    @Test func testMultipleNestedAgentToolStatesSurviveMultipleDroppedFunctions() async throws {
        let context = scopedContext()
        let agent = Agent(name: "OuterAgent")
        let earlierToolEnabled = SharedBox(true)
        let disabledTool = try FunctionTool.create(name: "disabled_tool", isEnabled: .dynamic { _, _ in earlierToolEnabled.value }) { _ in
            "disabled"
        }
        let firstNestedTool = try FunctionTool.create(name: "first_agent_tool") { _ in "first" }
        let malformedTool = try FunctionTool.create(name: "malformed_tool") { _ in "malformed" }
        let secondNestedTool = try FunctionTool.create(name: "second_agent_tool") { _ in "second" }
        agent.tools = [disabledTool, firstNestedTool, malformedTool, secondNestedTool]

        let disabledCall = makeToolCall("disabled-call", name: "disabled_tool")
        let firstNestedCall = makeToolCall("first-call", name: "first_agent_tool")
        let malformedCall = makeToolCall("malformed-call", name: "malformed_tool")
        let secondNestedCall = makeToolCall("second-call", name: "second_agent_tool")
        let state = makeState(agent, context: context)
        state.lastProcessedResponse = makeProcessedResponse(functions: [
            ToolRunFunction(toolCall: disabledCall, functionTool: disabledTool),
            ToolRunFunction(toolCall: firstNestedCall, functionTool: firstNestedTool),
            ToolRunFunction(toolCall: malformedCall, functionTool: malformedTool),
            ToolRunFunction(toolCall: secondNestedCall, functionTool: secondNestedTool),
        ])
        let innerCallIds = ["first-inner-call", "second-inner-call"]
        for (nestedCall, innerCallId) in zip([firstNestedCall, secondNestedCall], innerCallIds) {
            recordPendingNestedAgentToolState(agent, nestedCall, innerCallId: innerCallId, scope: state.agentToolStateScope)
        }

        var stateJSON = try state.toJSON()
        earlierToolEnabled.value = false
        try removeCallId(&stateJSON, functionIndex: 2)

        let restored = try await RunState.fromJSON(agent, stateJSON)

        let restoredCalls = try #require(restored.lastProcessedResponse?.functions).map(\.toolCall)
        #expect(restoredCalls.map(\.callId) == ["first-call", "second-call"])
        for (restoredCall, expectedInnerCallId) in zip(restoredCalls, innerCallIds) {
            let pendingResult = try #require(peekAgentToolRunResult(restoredCall.json, scope: restored.agentToolStateScope))
            #expect(pendingResult.interruptions.count == 1)
            #expect(pendingResult.interruptions.first?.rawItem.string("call_id") == expectedInnerCallId)
        }
    }
}

// MARK: - Nested resumes

extension RunStateDeserializeHelpersTests {
    /// Requests `toolName` (after an optional preceding call) until a tool output arrives.
    private static func makeResumeAwareToolModel(
        toolName: String,
        toolArguments: String,
        finalText: String,
        callPrefix: String,
        precedingToolName: String? = nil
    ) throws -> ScriptedModel {
        let toolCallCount = SharedBox(0)
        let modelId = UUID().uuidString
        let step = ModelStep.respond { call in
            if hasFunctionCallOutput(call.input) {
                return .step(.output([getTextMessage(finalText)], responseId: "\(callPrefix)-done"))
            }
            let count = toolCallCount.mutate { value -> Int in
                value += 1
                return value
            }
            var output: [TResponseOutputItem] = []
            if let precedingToolName {
                output.append(makeFunctionToolCall(precedingToolName, callId: "\(callPrefix)-preceding-\(count)"))
            }
            output.append(makeFunctionToolCall(toolName, callId: "\(callPrefix)-\(modelId)-\(count)", arguments: toolArguments))
            return .step(.output(output, responseId: "\(callPrefix)-call-\(count)"))
        }
        return try ScriptedModel(steps: Array(repeating: step, count: 3))
    }

    // Swift: upstream deletes the first result and runs the garbage collector before restoring;
    // Swift has no collector to run, so the JSON round trip alone is exercised.
    @Test(arguments: [true, false])
    func testNestedAgentToolHitlResumeSurvivesJSONRoundTripAfterGc(approveNestedTool: Bool) async throws {
        let toolCalls = SharedBox<[String]>([])
        let innerSensitiveTool = try FunctionTool.create(name: "inner_sensitive_tool", argumentsType: TextArgs.self, needsApproval: .always) { _, args in
            toolCalls.mutate { $0.append(args.text) }
            return "approved:\(args.text)"
        }
        let innerModel = try Self.makeResumeAwareToolModel(
            toolName: "inner_sensitive_tool", toolArguments: #"{"text": "hello"}"#, finalText: "inner-complete", callPrefix: "inner"
        )
        let innerAgent = makeAgent(model: innerModel, tools: [innerSensitiveTool], name: "InnerAgent")
        let outerTool = try innerAgent.asTool(toolName: "inner_agent_tool", toolDescription: "Inner agent tool")
        let outerModel = try Self.makeResumeAwareToolModel(
            toolName: "inner_agent_tool", toolArguments: #"{"input": "hello"}"#, finalText: "outer-complete", callPrefix: "outer",
            precedingToolName: "conditional_outer_tool"
        )
        let outerToolEnabled = SharedBox(true)
        let conditionalOuterTool = try FunctionTool.create(name: "conditional_outer_tool", isEnabled: .dynamic { _, _ in outerToolEnabled.value }) { _ in
            "conditional-complete"
        }
        let outerAgent = makeAgent(model: outerModel, tools: [conditionalOuterTool, outerTool], name: "OuterAgent")

        let firstResult = try await Runner.run(outerAgent, input: "start")
        #expect(firstResult.finalOutput == nil)
        #expect(!firstResult.interruptions.isEmpty)

        let stateJSON = try firstResult.toState().toJSON()
        let serializedFunctions = try #require(stateJSON["last_processed_response"]?.objectValue?["functions"]?.arrayValue)
        #expect(serializedFunctions.compactMap { $0.objectValue?["tool_call"]?.objectValue?.string("name") } == [
            "conditional_outer_tool",
            "inner_agent_tool",
        ])
        outerToolEnabled.value = false

        let restoredStateOne = try await RunState.fromJSON(outerAgent, stateJSON)
        let restoredStateTwo = try await RunState.fromJSON(outerAgent, stateJSON)

        let restoredInterruptionsOne = restoredStateOne.getInterruptions()
        let restoredInterruptionsTwo = restoredStateTwo.getInterruptions()
        #expect(restoredInterruptionsOne.count == 1)
        #expect(restoredInterruptionsTwo.count == 1)
        if approveNestedTool {
            try restoredStateOne.approve(restoredInterruptionsOne[0])
            try restoredStateTwo.approve(restoredInterruptionsTwo[0])
        } else {
            try restoredStateOne.reject(restoredInterruptionsOne[0])
            try restoredStateTwo.reject(restoredInterruptionsTwo[0])
        }

        let resumedResultOne = try await Runner.run(outerAgent, state: restoredStateOne)
        let resumedResultTwo = try await Runner.run(outerAgent, state: restoredStateTwo)

        #expect(resumedResultOne.finalOutput as? String == "outer-complete")
        #expect(resumedResultOne.interruptions.isEmpty)
        #expect(resumedResultTwo.finalOutput as? String == "outer-complete")
        #expect(resumedResultTwo.interruptions.isEmpty)
        #expect(toolCalls.value == (approveNestedTool ? ["hello", "hello"] : []))
        try innerModel.assertComplete()
        try outerModel.assertComplete()
    }

    enum DuplicateNameGraph: String, CaseIterable, Sendable, CustomTestStringConvertible {
        case innerSharesOuterName = "inner_shares_outer_name"
        case siblingAgentsShareName = "sibling_agents_share_name"
        case clonedSiblingAgents = "cloned_sibling_agents"

        var testDescription: String { rawValue }
    }

    @Test(arguments: DuplicateNameGraph.allCases, [true, false])
    func testNestedAgentToolHitlResumeWithDuplicateAgentNames(agentGraph: DuplicateNameGraph, approveNestedTool: Bool) async throws {
        let toolCalls = SharedBox<[String]>([])
        func makeSensitiveTool(_ label: String) throws -> FunctionTool {
            try FunctionTool.create(name: "inner_sensitive_tool", argumentsType: TextArgs.self, needsApproval: .always) { _, args in
                toolCalls.mutate { $0.append("\(label):\(args.text)") }
                return "approved:\(args.text)"
            }
        }
        func makeInnerModel(_ label: String) throws -> ScriptedModel {
            try makeToolCallsThenFinalModel(
                toolCalls: [("inner_sensitive_tool", #"{"text": "hello"}"#)], finalText: "\(label)-complete", callPrefix: label
            )
        }

        let targetModel = try makeInnerModel("target")
        let targetTools: [any Tool] = [try makeSensitiveTool("target")]
        var outerName = "OuterAgent"
        var siblingAgent: Agent?
        let targetAgent: Agent
        switch agentGraph {
        case .innerSharesOuterName:
            outerName = "Assistant"
            targetAgent = makeAgent(model: targetModel, tools: targetTools, name: "Assistant")
        case .siblingAgentsShareName:
            siblingAgent = makeAgent(model: try makeInnerModel("sibling"), tools: [try makeSensitiveTool("sibling")], name: "Worker")
            targetAgent = makeAgent(model: targetModel, tools: targetTools, name: "Worker")
        case .clonedSiblingAgents:
            let baseAgent = Agent(name: "Worker")
            let siblingModel = try makeInnerModel("sibling")
            let siblingTools: [any Tool] = [try makeSensitiveTool("sibling")]
            siblingAgent = baseAgent.clone {
                $0.model = .instance(siblingModel)
                $0.tools = siblingTools
            }
            targetAgent = baseAgent.clone {
                $0.model = .instance(targetModel)
                $0.tools = targetTools
            }
        }

        // The sibling is registered first so that the target needs a suffixed identity key.
        var outerTools: [any Tool] = []
        if let siblingAgent {
            outerTools.append(try siblingAgent.asTool(toolName: "sibling_agent_tool", toolDescription: "Sibling agent tool"))
        }
        outerTools.append(try targetAgent.asTool(toolName: "target_agent_tool", toolDescription: "Target agent tool"))
        let outerModel = try makeToolCallsThenFinalModel(
            toolCalls: [("target_agent_tool", #"{"input": "hello"}"#)], finalText: "outer-complete", callPrefix: "outer"
        )
        let outerAgent = makeAgent(model: outerModel, tools: outerTools, name: outerName)

        let firstResult = try await Runner.run(outerAgent, input: "start")
        #expect(firstResult.finalOutput == nil)
        #expect(firstResult.interruptions.count == 1)

        let stateJSON = try JSONValue.parse(try firstResult.toState().toString()).objectValue ?? [:]

        let restoredState = try await RunState.fromJSON(outerAgent, stateJSON)
        let restoredInterruptions = restoredState.getInterruptions()
        #expect(restoredInterruptions.count == 1)
        #expect(restoredInterruptions.first?.agent === targetAgent)
        if approveNestedTool {
            try restoredState.approve(restoredInterruptions[0])
        } else {
            try restoredState.reject(restoredInterruptions[0])
        }

        let resumedResult = try await Runner.run(outerAgent, state: restoredState)

        #expect(resumedResult.finalOutput as? String == "outer-complete")
        #expect(resumedResult.interruptions.isEmpty)
        #expect(toolCalls.value == (approveNestedTool ? ["target:hello"] : []))
        try targetModel.assertComplete()
        try outerModel.assertComplete()
    }

    @Test func testMultipleNestedAgentToolHitlResumesWithDuplicateAgentNames() async throws {
        let toolCalls = SharedBox<[String]>([])
        func makeWorker(_ label: String) throws -> (Agent, ScriptedModel) {
            let tool = try FunctionTool.create(name: "inner_sensitive_tool", argumentsType: TextArgs.self, needsApproval: .always) { _, args in
                toolCalls.mutate { $0.append("\(label):\(args.text)") }
                return "approved:\(args.text)"
            }
            let model = try makeToolCallsThenFinalModel(
                toolCalls: [("inner_sensitive_tool", #"{"text": "hello"}"#)], finalText: "\(label)-complete", callPrefix: label
            )
            return (makeAgent(model: model, tools: [tool], name: "Worker"), model)
        }

        let (firstWorker, firstModel) = try makeWorker("first")
        let (secondWorker, secondModel) = try makeWorker("second")
        let outerModel = try makeToolCallsThenFinalModel(
            toolCalls: [("first_agent_tool", #"{"input": "hello"}"#), ("second_agent_tool", #"{"input": "hello"}"#)],
            finalText: "outer-complete",
            callPrefix: "outer"
        )
        let outerAgent = makeAgent(
            model: outerModel,
            tools: [
                try firstWorker.asTool(toolName: "first_agent_tool", toolDescription: "First agent tool"),
                try secondWorker.asTool(toolName: "second_agent_tool", toolDescription: "Second agent tool"),
            ],
            name: "OuterAgent"
        )

        let firstResult = try await Runner.run(outerAgent, input: "start")
        #expect(firstResult.finalOutput == nil)
        #expect(firstResult.interruptions.count == 2)

        let stateJSON = try JSONValue.parse(try firstResult.toState().toString()).objectValue ?? [:]

        let restoredState = try await RunState.fromJSON(outerAgent, stateJSON)
        let restoredInterruptions = restoredState.getInterruptions()
        #expect(Set(restoredInterruptions.map { ObjectIdentifier($0.agent) }) == [ObjectIdentifier(firstWorker), ObjectIdentifier(secondWorker)])
        for interruption in restoredInterruptions {
            if interruption.agent === secondWorker {
                try restoredState.approve(interruption)
            } else {
                try restoredState.reject(interruption)
            }
        }

        let resumedResult = try await Runner.run(outerAgent, state: restoredState)

        #expect(resumedResult.finalOutput as? String == "outer-complete")
        #expect(resumedResult.interruptions.isEmpty)
        #expect(toolCalls.value == ["second:hello"])
        try firstModel.assertComplete()
        try secondModel.assertComplete()
        try outerModel.assertComplete()
    }
}

// MARK: - Resume checkpoints

extension RunStateResumptionTests {
    @Test func testNestedResumeCheckpointToStateKeepsItsOwnedDecisionLedger() {
        let agent = Agent(name: "NestedAgent")
        let approvalItem = makeToolApprovalItem(agent, callId: "nested-call")
        let state = makeStateWithInterruptions(agent, [approvalItem])
        let checkpoint = AgentToolRunRecord.resumeCheckpoint(state, approvalIdentities: [])

        #expect(checkpoint.toState() === state)
    }
}

// MARK: - Module-level nested resumes

@Suite struct RunStateNestedAgentToolResumeTests {
    @Test func testResumeNestedAgentAsToolWithContextOverride() async throws {
        let seenContexts = SharedBox<[JSONObject]>([])
        let seenToolInputs = SharedBox<[JSONValue?]>([])
        let calls = SharedBox<[String]>([])
        let needsOk = try FunctionTool.create(name: "needs_ok", argumentsType: TextArgs.self, needsApproval: .always) { context, args in
            seenContexts.mutate { $0.append(context.context(as: JSONObject.self) ?? [:]) }
            seenToolInputs.mutate { $0.append(context.toolInput as? JSONValue) }
            calls.mutate { $0.append(args.text) }
            return args.text
        }
        let nestedTurnUsage = Usage(requests: 1, inputTokens: 17, outputTokens: 3, totalTokens: 20)
        let nestedModel = try ScriptedModel()
        nestedModel.setDefaultUsage(nestedTurnUsage)
        let nestedAgent = makeAgent(model: nestedModel, tools: [needsOk], name: "nested")
        try nestedModel.extend([
            .output([getFunctionToolCall("needs_ok", arguments: #"{"text": "one"}"#, callId: "inner-1")]),
            .output([getFinalOutputMessage("nested-done")]),
        ])
        // Swift: `parametersSchema` stands in for upstream's `parameters=NestedParams` dataclass.
        let outerModel = try ScriptedModel()
        let nestedParams: JSONObject = [
            "type": "object",
            "properties": ["input": ["type": "string"]],
            "required": ["input"],
            "additionalProperties": false,
        ]
        let nestedTool = try nestedAgent.asTool(toolName: "nested_agent", toolDescription: "Run nested agent", parametersSchema: nestedParams)
        let outer = makeAgent(model: outerModel, tools: [nestedTool], name: "outer")
        try outerModel.extend([
            .output([getFunctionToolCall("nested_agent", arguments: #"{"input": "hi"}"#, callId: "outer-1")]),
            .output([getFinalOutputMessage("done")]),
        ])

        let first = try await Runner.run(outer, input: "hi", context: ["user": "original"] as JSONObject)
        #expect(first.interruptions.first?.toolName == "needs_ok")

        let state = first.toState()
        state.context.toolInput = ["scoped": true] as JSONObject
        try state.approve(try #require(first.interruptions.first))
        let restored = try await RunState.fromJSON(outer, try state.toJSON())
        let restoredWrapper = restored.context
        #expect(restoredWrapper.toolInput as? JSONValue == ["scoped": true])
        #expect(restoredWrapper.approvals.isEmpty)
        let restoredCall = try #require(restored.lastProcessedResponse?.functions.first?.toolCall)
        let restoredNestedResult = try #require(peekAgentToolRunResult(restoredCall.json, scope: restored.agentToolStateScope))
        #expect(!restoredNestedResult.toState().context.approvals.isEmpty)
        let usageBeforeResume = restoredWrapper.usage.inputTokens
        let override: JSONObject = ["user": "reviewer"]

        let resumed = try await Runner.run(outer, state: restored, context: override)

        #expect(resumed.finalOutput as? String == "done")
        #expect(resumed.interruptions.isEmpty)
        #expect(calls.value == ["one"])
        #expect(seenContexts.value == [override])
        #expect(seenToolInputs.value == [["input": "hi"]])
        #expect(resumed.contextWrapper === restoredWrapper)
        #expect(resumed.contextWrapper.context(as: JSONObject.self) == override)
        #expect(resumed.contextWrapper.toolInput as? JSONValue == ["scoped": true])
        #expect(resumed.contextWrapper.toolStateBox === restoredWrapper.toolStateBox)
        // Nested post-resume model turns must keep accruing on the parent usage object.
        #expect(resumed.contextWrapper.usage.inputTokens == usageBeforeResume + nestedTurnUsage.inputTokens)
    }

    struct RecursiveCase: Sendable, CustomTestStringConvertible {
        var streamed: Bool
        var approve: Bool
        var approvalTiming: String
        var nestingEdges: Int

        var testDescription: String {
            "streamed=\(streamed) approve=\(approve) timing=\(approvalTiming) edges=\(nestingEdges)"
        }

        // Swift: Swift Testing accepts at most two argument collections, so the four upstream
        // parameters are combined into one list of cases.
        static let all: [RecursiveCase] = [false, true].flatMap { streamed in
            [true, false].flatMap { approve in
                ["live", "before_restore", "after_restore"].flatMap { timing in
                    [2, 3].map { RecursiveCase(streamed: streamed, approve: approve, approvalTiming: timing, nestingEdges: $0) }
                }
            }
        }
    }

    @Test(arguments: RecursiveCase.all)
    func testResumeRecursivelyNestedAgentAsToolDecision(testCase: RecursiveCase) async throws {
        let calls = SharedBox<[String]>([])
        let protected = try FunctionTool.create(name: "protected", argumentsType: TextArgs.self, needsApproval: .always) { _, args in
            calls.mutate { $0.append(args.text) }
            return "approved:\(args.text)"
        }
        let leafModel = try ScriptedModel()
        try leafModel.extend([
            .output([getFunctionToolCall("protected", arguments: #"{"text": "one"}"#, callId: "inner-1")]),
            .output([getFinalOutputMessage("inner-done")]),
        ])
        var outer = makeAgent(model: leafModel, tools: [protected], name: "inner")
        for edge in 0..<testCase.nestingEdges {
            let toolName = "agent_tool_\(edge)"
            let model = try ScriptedModel()
            try model.extend([
                .output([getFunctionToolCall(toolName, arguments: #"{"input": "go"}"#, callId: "agent-call-\(edge)")]),
                .output([getFinalOutputMessage("done-\(edge)")]),
            ])
            outer = makeAgent(model: model, tools: [try outer.asTool(toolName: toolName, toolDescription: "Run nested agent")], name: "agent-\(edge)")
        }
        let rootAgent = outer

        var state: RunState
        if testCase.streamed {
            let first = try Runner.runStreamed(rootAgent, input: "start")
            try await consumeStream(first)
            state = first.toState()
        } else {
            state = try await Runner.run(rootAgent, input: "start").toState()
        }
        #expect(state.getInterruptions().count == 1)

        func applyDecision(_ state: RunState) throws {
            let interruption = try #require(state.getInterruptions().first)
            if testCase.approve {
                try state.approve(interruption)
            } else {
                try state.reject(interruption)
            }
        }
        switch testCase.approvalTiming {
        case "before_restore":
            try applyDecision(state)
            state = try await RunState.fromJSON(rootAgent, try state.toJSON())
        case "after_restore":
            state = try await RunState.fromJSON(rootAgent, try state.toJSON())
            try applyDecision(state)
        default:
            try applyDecision(state)
        }

        let finalOutput: (any Sendable)?
        let interruptions: [ToolApprovalItem]
        if testCase.streamed {
            let resumed = try Runner.runStreamed(rootAgent, state: state)
            try await consumeStream(resumed)
            finalOutput = resumed.finalOutput
            interruptions = resumed.interruptions
        } else {
            let resumed = try await Runner.run(rootAgent, state: state)
            finalOutput = resumed.finalOutput
            interruptions = resumed.interruptions
        }

        #expect(finalOutput as? String == "done-\(testCase.nestingEdges - 1)")
        #expect(interruptions.isEmpty)
        #expect(calls.value == (testCase.approve ? ["one"] : []))
    }
}
