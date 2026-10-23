import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_tool_approval_call_id_reuse.py (approval-bound invocation cases).
extension ToolApprovalCallIdReuseTests {
    // MARK: - Approval-bound invocations

    @Test(arguments: RunMode.allCases)
    func testIdenticalApprovalBoundSiblingsExecuteOnce(mode: RunMode) async throws {
        let executed = SharedBox<[String]>([])
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
            executed.mutate { $0.append(args.value) }
            return args.value
        }
        let duplicatedCall = getFunctionToolCall("record_value", arguments: #"{"value": "safe"}"#, callId: "call-duplicate")
        let model = try ScriptedModel()
        try model.extend([.output([duplicatedCall, duplicatedCall]), .output([getTextMessage("done")])])
        let runConfig = RunConfig(modelProvider: ScriptedProvider(model))
        let agent = Agent(name: "custom-provider-agent", tools: [recordValue], model: .name("scripted-provider-model"))

        let first = try await run(agent, .text("record a value"), runConfig: runConfig, mode: mode)
        #expect(first.interruptions.count == 1)
        let state = first.state()
        try state.approve(first.interruptions[0])

        let resumed = try await run(agent, .state(state), runConfig: runConfig, mode: mode)

        #expect(resumed.finalOutput as? String == "done")
        #expect(executed.value == ["safe"])
    }

    @Test(arguments: RunMode.allCases)
    func testNewCallIdRequiresANewPerCallApproval(mode: RunMode) async throws {
        let (agent, runConfig, executed) = try buildScenario(secondCallId: "call_1", secondValue: "changed")

        let first = try await run(agent, .text("record a value"), runConfig: runConfig, mode: mode)
        let state = first.state()
        try state.approve(first.interruptions[0])

        let resumed = try await run(agent, .state(state), runConfig: runConfig, mode: mode)

        #expect(resumed.interruptions.count == 1)
        #expect(resumed.interruptions.first?.arguments == #"{"value": "changed"}"#)
        #expect(executed.value == ["safe"])
    }

    @Test(arguments: [true, false])
    func testSerializedStickyDecisionRejectsChangedReusedCallId(approve: Bool) async throws {
        let (agent, runConfig, executed) = try buildScenario(secondCallId: "call_0", secondValue: "changed")

        let first = try await Runner.run(agent, input: "record a value", runConfig: runConfig)
        let state = first.toState()
        if approve {
            try state.approve(first.interruptions[0], alwaysApprove: true)
        } else {
            try state.reject(first.interruptions[0], alwaysReject: true)
        }
        let restoredState = try await RunState.fromJSON(agent, try state.toJSON())

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Runner.run(agent, state: restoredState, runConfig: runConfig)
        }

        #expect(executed.value == (approve ? ["safe"] : []))
    }

    @Test func testStickyUpgradeRejectsChangedReuseOfPriorCallId() async throws {
        let (agent, runConfig, executed) = try buildScenario(secondCallId: "call_0", secondValue: "changed")

        let first = try await Runner.run(agent, input: "record a value", runConfig: runConfig)
        let state = first.toState()
        try state.approve(first.interruptions[0])
        try state.approve(first.interruptions[0], alwaysApprove: true)

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Runner.run(agent, state: state, runConfig: runConfig)
        }

        #expect(executed.value == ["safe"])
    }

    struct IntValueArgs: Decodable, Sendable { var value: Int }

    @Test func testSerializedStickyIdenticalSiblingsExecuteOnce() async throws {
        let executed = SharedBox<[String]>([])
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: IntValueArgs.self, needsApproval: true) { _, args in
            executed.mutate { $0.append(String(args.value)) }
            return String(args.value)
        }
        let duplicate = getFunctionToolCall("record_value", arguments: #"{"value":1}"#, callId: "call_0")
        let model = try ScriptedModel(steps: [.output([duplicate, duplicate])])
        try model.enqueue(.output([getTextMessage("done")]))
        let agent = Agent(name: "agent", tools: [recordValue], model: .instance(model))

        let first = try await Runner.run(agent, input: "record a value")
        let state = first.toState()
        try state.approve(first.interruptions[0], alwaysApprove: true)
        let restored = try await RunState.fromJSON(agent, try state.toJSON())

        let resumed = try await Runner.run(agent, state: restored)

        #expect(resumed.finalOutput as? String == "done")
        #expect(executed.value == ["1"])
    }

    func functionApproval(
        _ agent: Agent,
        _ name: String,
        _ arguments: String,
        callId: String,
        namespace: String? = nil,
        lookupKey: FunctionToolLookupKey? = nil,
        allowBareNameAlias: Bool = false
    ) -> ToolApprovalItem {
        ToolApprovalItem(
            agent: agent,
            rawItem: getFunctionToolCall(name, arguments: arguments, callId: callId),
            toolName: name,
            toolNamespace: namespace,
            toolLookupKey: lookupKey,
            allowBareNameAlias: allowBareNameAlias
        )
    }

    func decide(_ context: RunContextWrapper, _ item: ToolApprovalItem, approve: Bool) throws {
        if approve {
            try context.approveTool(item, alwaysApprove: true)
        } else {
            try context.rejectTool(item, alwaysReject: true)
        }
    }

    @Test(arguments: [true, false])
    func testStickyDecisionPreservesPriorPerCallBinding(approve: Bool) throws {
        let agent = Agent(name: "agent")
        let context = RunContextWrapper(context: ())
        let first = functionApproval(agent, "tool_a", #"{"value":1}"#, callId: "call_0")
        let sticky = functionApproval(agent, "tool_a", #"{"value":2}"#, callId: "call_1")
        let changed = functionApproval(agent, "tool_a", #"{"value":3}"#, callId: "call_0")
        try context.approveTool(first)
        try decide(context, sticky, approve: approve)

        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try context.getApprovalStatus(toolName: "tool_a", callId: "call_0", currentInvocation: changed)
        }
    }

    @Test(arguments: [true, false])
    func testStickyDecisionDoesNotDisableOtherToolBinding(approve: Bool) throws {
        let agent = Agent(name: "agent")
        let context = RunContextWrapper(context: ())
        let other = functionApproval(agent, "tool_b", #"{"value":1}"#, callId: "call_0")
        let sticky = functionApproval(agent, "tool_a", #"{"value":2}"#, callId: "call_1")
        let changedOther = functionApproval(agent, "tool_b", #"{"value":3}"#, callId: "call_0")
        try context.approveTool(other)
        try decide(context, sticky, approve: approve)

        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try context.getApprovalStatus(toolName: "tool_b", callId: "call_0", currentInvocation: changedOther)
        }
    }

    @Test(arguments: [true, false])
    func testMatchingStickyDecisionDoesNotMaskOtherToolBinding(approve: Bool) throws {
        let agent = Agent(name: "agent")
        let context = RunContextWrapper(context: ())
        let other = functionApproval(agent, "tool_b", #"{"value":1}"#, callId: "call_0")
        let sticky = functionApproval(agent, "tool_a", #"{"value":2}"#, callId: "call_1")
        let current = functionApproval(agent, "tool_a", #"{"value":3}"#, callId: "call_0")
        try context.approveTool(other)
        try decide(context, sticky, approve: approve)

        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try context.getApprovalStatus(toolName: "tool_a", callId: "call_0", currentInvocation: current)
        }
    }

    @Test(arguments: [true, false])
    func testDeferredStickyDecisionPreservesMirroredPerCallBinding(approve: Bool) throws {
        let agent = Agent(name: "agent")
        let context = RunContextWrapper(context: ())
        func approvalItem(_ callId: String, _ value: Int) -> ToolApprovalItem {
            functionApproval(
                agent, "lookup", #"{"value": \#(value)}"#, callId: callId,
                namespace: "lookup", lookupKey: .deferredTopLevel("lookup"), allowBareNameAlias: true
            )
        }
        try context.approveTool(approvalItem("call_0", 1))
        try decide(context, approvalItem("call_1", 2), approve: approve)

        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try context.getApprovalStatus(
                toolName: "lookup", callId: "call_0", toolNamespace: "lookup",
                toolLookupKey: .deferredTopLevel("lookup"), currentInvocation: approvalItem("call_0", 3)
            )
        }
    }

    @Test func testStickyFunctionApprovalRejectsSameIdShellCall() throws {
        let agent = Agent(name: "agent")
        let context = RunContextWrapper(context: ())
        try context.approveTool(functionApproval(agent, "shell", "{}", callId: "call_0"), alwaysApprove: true)
        let shellItem = ToolApprovalItem(
            agent: agent,
            rawItem: ["type": "shell_call", "call_id": "call_0", "action": ["commands": ["echo safe"]]],
            toolName: "shell"
        )

        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try context.getApprovalStatus(toolName: "shell", callId: "call_0", currentInvocation: shellItem)
        }
    }

    @Test func testChangedToolUnderApprovedCallIdFailsBeforeSecondToolStarts() async throws {
        let executed = SharedBox<[String]>([])
        let firstTool = try FunctionTool.create(name: "first_tool", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
            executed.mutate { $0.append("first:\(args.value)") }
            return args.value
        }
        let secondTool = try FunctionTool.create(name: "second_tool", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
            executed.mutate { $0.append("second:\(args.value)") }
            return args.value
        }
        let model = try ScriptedModel()
        try model.extend([
            .output([getFunctionToolCall("first_tool", arguments: #"{"value": "same"}"#, callId: "call_0")]),
            .output([getFunctionToolCall("second_tool", arguments: #"{"value": "same"}"#, callId: "call_0")]),
        ])
        let runConfig = RunConfig(modelProvider: ScriptedProvider(model))
        let agent = Agent(name: "custom-provider-agent", tools: [firstTool, secondTool], model: .name("scripted-provider-model"))

        let first = try await Runner.run(agent, input: "run tools", runConfig: runConfig)
        let state = first.toState()
        try state.approve(first.interruptions[0])

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Runner.run(agent, state: state, runConfig: runConfig)
        }

        #expect(executed.value == ["first:same"])
    }

    @Test func testApprovedCallIdReusedForAnotherInvocationTypeFailsBeforeExecution() async throws {
        let executed = SharedBox<[String]>([])
        let approvedTool = try FunctionTool.create(name: "approved_tool", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
            executed.mutate { $0.append("function:\(args.value)") }
            return args.value
        }
        let customTool = try CustomTool(
            name: "raw_editor",
            description: "Edit raw text.",
            onInvokeTool: { _, rawInput in
                executed.mutate { $0.append("custom:\(rawInput)") }
                return rawInput
            },
            format: ["type": "text"]
        )
        let model = try ScriptedModel()
        try model.extend([
            .output([getFunctionToolCall("approved_tool", arguments: #"{"value": "safe"}"#, callId: "call_0")]),
            .output([["type": "custom_tool_call", "name": "raw_editor", "call_id": "call_0", "input": "changed-kind"]]),
        ])
        let runConfig = RunConfig(modelProvider: ScriptedProvider(model))
        let agent = Agent(name: "custom-provider-agent", tools: [approvedTool, customTool], model: .name("scripted-provider-model"))

        let first = try await Runner.run(agent, input: "run tools", runConfig: runConfig)
        let state = first.toState()
        try state.approve(first.interruptions[0])

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Runner.run(agent, state: state, runConfig: runConfig)
        }

        #expect(executed.value == ["function:safe"])
    }

    @Test func testChangedMissingToolUnderApprovedCallIdFailsBeforeSiblingTool() async throws {
        let executed = SharedBox<[String]>([])
        let approvalChecks = SharedBox<[String]>([])
        let approvedTool = try FunctionTool.create(name: "approved_tool", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
            executed.mutate { $0.append("approved:\(args.value)") }
            return args.value
        }
        let siblingTool = try FunctionTool.create(name: "sibling_tool", needsApproval: .dynamic { _, _, callId in
            approvalChecks.mutate { $0.append(callId) }
            return false
        }) { _ -> String in
            executed.mutate { $0.append("sibling") }
            return "sibling"
        }
        let model = try ScriptedModel()
        try model.extend([
            .output([getFunctionToolCall("approved_tool", arguments: #"{"value": "safe"}"#, callId: "call_0")]),
            .output([
                getFunctionToolCall("sibling_tool", arguments: "{}", callId: "call_1"),
                getFunctionToolCall("missing_tool", arguments: "{}", callId: "call_0"),
            ]),
        ])
        let runConfig = RunConfig(modelProvider: ScriptedProvider(model), toolNotFoundBehavior: .returnErrorToModel)
        let agent = Agent(name: "custom-provider-agent", tools: [approvedTool, siblingTool], model: .name("scripted-provider-model"))

        let first = try await Runner.run(agent, input: "run tools", runConfig: runConfig)
        let state = first.toState()
        try state.approve(first.interruptions[0])

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Runner.run(agent, state: state, runConfig: runConfig)
        }

        #expect(executed.value == ["approved:safe"])
        #expect(approvalChecks.value.isEmpty)
    }
}
