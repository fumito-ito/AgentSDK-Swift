import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_hitl_error_scenarios.py: regression tests for HITL edge cases.
//
// Swift: the tests are split by topic across `HITLErrorScenariosTests+Resume.swift` (direct
// `resolveInterruptedTurn` calls), `HITLErrorScenariosTests+Approvals.swift` (resume selection,
// execution-time decisions, and MCP callbacks), and `HITLErrorScenariosTests+Nested.swift` (nested
// `Agent.asTool()` scenarios).
//
// Not ported here:
// - `test_function_needs_approval_invalid_type_raises` and
//   `test_resume_invalid_needs_approval_raises`: `ToolApprovalRequirement` is either a Bool or a
//   policy, so an invalid setting cannot be constructed.
// - `test_resume_checkpoints_computer_output_before_custom_data_failure`: `ComputerTool` has no
//   `custom_data_extractor` yet.
//
// Upstream passes an execution clone bound to its public agent (`set_public_agent`). Only sandbox
// agents create such clones (D2), so the Swift tests resume with the public agent itself.
@Suite struct HITLErrorScenariosTests {
    struct NoArgs: Codable, Sendable {}

    /// Minimal computer implementation that records method calls.
    final class TrackingComputer: Computer, @unchecked Sendable {
        private let lock = NSLock()
        private var recorded: [String] = []

        var calls: [String] { lock.locked { recorded } }
        var environment: Environment? { "mac" }
        var dimensions: (width: Int, height: Int)? { (1, 1) }

        private func record(_ call: String) { lock.locked { recorded.append(call) } }

        func screenshot() async throws -> String {
            record("screenshot")
            return "img"
        }

        func click(x: Int, y: Int, button: Button) async throws { record("click") }
        func doubleClick(x: Int, y: Int) async throws { record("double_click") }
        func scroll(x: Int, y: Int, scrollX: Int, scrollY: Int) async throws { record("scroll") }
        func type(text: String) async throws { record("type") }
        func wait() async throws { record("wait") }
        func move(x: Int, y: Int) async throws { record("move") }
        func keypress(keys: [String]) async throws { record("keypress") }
        func drag(path: [(x: Int, y: Int)]) async throws { record("drag") }
    }

    enum ToolKind: String, CaseIterable, Sendable, CustomTestStringConvertible {
        case shell
        case applyPatch = "apply_patch"

        var testDescription: String { rawValue }
    }

    private static func shellApprovalSetup() throws -> ApprovalScenario {
        let tool = try ShellTool(executor: { _ in .text("shell_output") }, needsApproval: .dynamic { _, _, _ in true })
        let shellCall = makeShellCall("call_shell_1", idValue: "shell_1", commands: ["echo test"])
        return ApprovalScenario(tool: tool, rawCall: shellCall, finalOutput: getTextMessage("done")) { result in
            let shellOutputs = collectToolOutputs(result.newItems, outputType: "shell_call_output")
            #expect(!shellOutputs.isEmpty, "Shell tool should have been executed after approval")
            #expect(shellOutputs.contains { String(describing: $0.output).contains("shell_output") })
        }
    }

    private static func applyPatchApprovalSetup() throws -> ApprovalScenario {
        let editor = RecordingEditor()
        let tool = try ApplyPatchTool(editor: editor, needsApproval: .dynamic { _, _, _ in true })
        let applyPatchCall = makeApplyPatchDict("call_apply_1")
        return ApprovalScenario(tool: tool, rawCall: applyPatchCall, finalOutput: getTextMessage("done")) { result in
            let outputs = collectToolOutputs(result.newItems, outputType: "apply_patch_call_output")
            #expect(!outputs.isEmpty, "ApplyPatch tool should have been executed after approval")
            #expect(!editor.operations.isEmpty, "Editor should have been called")
        }
    }

    // MARK: - Resumed approvals

    @Test(arguments: ToolKind.allCases)
    func testResumedHitlExecutesApprovedTools(toolKind: ToolKind) async throws {
        let scenario = try toolKind == .shell ? Self.shellApprovalSetup() : Self.applyPatchApprovalSetup()
        let userInput = toolKind == .shell ? "run shell command" : "update file"
        let (model, agent) = try makeModelAndAgent(tools: [scenario.tool])

        let result = try await runAndResumeAfterApproval(agent, model, scenario.rawCall, scenario.finalOutput, userInput: userInput)

        scenario.assertResult(result)
    }

    @Test(arguments: ToolKind.allCases)
    func testResumingSkipsApprovalsForNonHitlTools(toolKind: ToolKind) async throws {
        let shellRuns = SharedBox<[String]>([])
        let editor = RecordingEditor()
        let autoTool: any Tool
        let rawCall: TResponseOutputItem
        let outputType: String
        switch toolKind {
        case .shell:
            autoTool = try ShellTool(executor: { _ in
                shellRuns.mutate { $0.append("run") }
                return .text("shell_output")
            })
            rawCall = makeShellCall("call_shell_auto", idValue: "shell_auto", commands: ["echo auto"])
            outputType = "shell_call_output"
        case .applyPatch:
            autoTool = try ApplyPatchTool(editor: editor)
            rawCall = makeApplyPatchDict("call_apply_auto")
            outputType = "apply_patch_call_output"
        }
        let approvalTool = try FunctionTool.create(name: "needs_hitl", argumentsType: NoArgs.self, needsApproval: requireApproval) { _, _ in
            "approved"
        }
        let (model, agent) = try makeModelAndAgent(tools: [autoTool, approvalTool])
        let functionCall = makeFunctionToolCall(approvalTool.name, callId: "call-func-auto")
        try queueFunctionCallAndText(model, functionCall, firstTurnExtra: [rawCall], followup: [getTextMessage("done")])

        let first = try await Runner.run(agent, input: .text("resume approvals"))
        #expect(!first.interruptions.isEmpty, "function tool should require approval")

        let resumed = try await resumeAfterFirstApproval(agent, first, alwaysApprove: true)

        #expect(resumed.interruptions.isEmpty, "non-HITL tools should not request approval on resume")
        let outputs = collectToolOutputs(resumed.newItems, outputType: outputType)
        #expect(outputs.count == 1, "\(toolKind) should run exactly once without extra approvals")
        switch toolKind {
        case .shell: #expect(shellRuns.value.count == 1, "shell should execute automatically when resuming")
        case .applyPatch: #expect(editor.operations.count == 1, "apply_patch should execute once when resuming")
        }
    }

    // MARK: - Pending approvals

    @Test(arguments: ToolKind.allCases)
    func testPendingApprovalsStayPendingOnResume(toolKind: ToolKind) async throws {
        let editor = RecordingEditor()
        let scenario: PendingScenario
        let outputType: String
        switch toolKind {
        case .shell:
            let tool = try ShellTool(executor: { _ in .text("shell_output") }, needsApproval: .required(true))
            scenario = PendingScenario(tool: tool, rawCall: makeShellCall("call_shell_pending", idValue: "shell_pending", commands: ["echo pending"]))
            outputType = "shell_call_output"
        case .applyPatch:
            let tool = try ApplyPatchTool(editor: editor, needsApproval: .required(true))
            scenario = PendingScenario(tool: tool, rawCall: makeApplyPatchDict("call_apply_pending")) { _ in
                #expect(editor.operations.isEmpty, "editor should not run before approval")
            }
            outputType = "apply_patch_call_output"
        }
        let model = try ScriptedModel()

        let resumed = try await assertPendingResume(
            scenario.tool, model, scenario.rawCall, userInput: "resume pending approval", outputType: outputType
        )

        scenario.assertResult?(resumed)
    }

    @Test func testResumeDoesNotDuplicatePendingShellApprovals() async throws {
        let tool = try ShellTool(executor: { _ in .text("shell_output") }, needsApproval: .required(true))
        let (model, agent) = try makeModelAndAgent(tools: [tool])
        let rawCall = makeShellCall("call_shell_pending_dup", idValue: "shell_pending_dup", commands: ["echo pending"])
        let callId = try #require(rawCall.string("call_id"), "shell call must have a call_id")

        try model.enqueue(.output([rawCall]))
        let first = try await Runner.run(agent, input: .text("run shell"))
        #expect(!first.interruptions.isEmpty, "shell tool should require approval")

        let resumed = try await Runner.run(agent, state: first.toState())
        let pendingItems = resumed.newItems.compactMap { $0 as? ToolApprovalItem }.filter { $0.rawItem.callIdOrId == callId }
        #expect(pendingItems.count == 1)
    }

    @Test func testResumingPendingMcpApprovalsRaisesTypeerror() throws {
        let (_, agent) = try makeModelAndAgent()
        let mcpApprovalItem = makeMCPApprovalItem(agent, callId: "mcp-approval-1", includeProviderData: false)

        var pendingHostedMCPApprovals: Set<ToolApprovalItem> = []
        pendingHostedMCPApprovals.insert(mcpApprovalItem)

        #expect(pendingHostedMCPApprovals.contains(mcpApprovalItem))
    }

    // MARK: - Routing

    @Test func testRouteLocalShellCallsToRemoteShellTool() async throws {
        let remoteShellExecuted = SharedBox<Int>(0)
        let localShellExecuted = SharedBox<Int>(0)
        let shellTool = try ShellTool(executor: { _ in
            remoteShellExecuted.mutate { $0 += 1 }
            return .text("remote_output")
        })
        let localShellTool = LocalShellTool { _ in
            localShellExecuted.mutate { $0 += 1 }
            return "local_output"
        }
        let (model, agent) = try makeModelAndAgent(tools: [shellTool, localShellTool])
        let localShellCall: TResponseOutputItem = [
            "id": "local_1",
            "call_id": "call_local_1",
            "type": "local_shell_call",
            "action": ["type": "exec", "command": ["echo", "test"], "env": [:]],
            "status": "in_progress",
        ]
        try model.enqueue(.output([localShellCall]))
        try model.enqueue(.output([]))

        _ = try await Runner.run(agent, input: .text("run local shell"))

        #expect(localShellExecuted.value > 0, "LocalShellTool should have been executed")
        #expect(remoteShellExecuted.value == 0, "ShellTool should not have been executed for local shell call")
    }

    // MARK: - Turn counters

    @Test func testPreserveMaxTurnsWhenResumingFromRunresultState() async throws {
        let tool = try FunctionTool.create(name: "test_tool", argumentsType: NoArgs.self, needsApproval: requireApproval) { _, _ in
            "tool_result"
        }
        let (model, agent) = try makeModelAndAgent(tools: [tool])
        try model.extend([.output([makeFunctionToolCall("test_tool", callId: "call-1")])])

        let result1 = try await Runner.run(agent, input: .text("call test_tool"), maxTurns: 20)
        #expect(!result1.interruptions.isEmpty, "should have an interruption")

        let state = try approveFirstInterruption(result1, alwaysApprove: true)
        // Provide 10 more turns (turns 2-11) to ensure we exceed the default 10 but not 20.
        try model.extend((0..<10).map { index in
            .output([getTextMessage("turn \(index + 2)"), makeFunctionToolCall("test_tool", callId: "call-\(index + 2)")])
        })
        try model.enqueue(.output([]))

        _ = try await Runner.run(agent, state: state)
    }

    @Test func testCurrentTurnNotPreservedInToState() async throws {
        let tool = try FunctionTool.create(name: "test_tool", argumentsType: NoArgs.self, needsApproval: requireApproval) { _, _ in
            "tool_result"
        }
        let (model, agent) = try makeModelAndAgent(tools: [tool])
        try model.enqueue(.output([makeFunctionToolCall("test_tool", callId: "call-1")]))

        let result1 = try await Runner.run(agent, input: .text("call test_tool"))
        #expect(!result1.interruptions.isEmpty, "should have interruption on turn 1")

        let state1 = result1.toState()

        #expect(state1.currentTurn == 1, "to_state() should preserve the current turn counter.")
    }

    @Test func testPreservePersistedItemCounterWhenResumingStreamedRuns() async throws {
        let (model, agent) = try makeModelAndAgent()
        // Simulate a turn interrupted mid-persistence: 5 items generated, 3 actually saved.
        let state = RunState(context: makeContextWrapper(), originalInput: .text("test input"), startingAgent: agent, maxTurns: 10)
        for index in 0..<5 {
            state.generatedItems.append(MessageOutputItem(agent: agent, rawItem: [
                "id": .string("msg_\(index)"),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [["type": "output_text", "text": .string("Message \(index)"), "annotations": [], "logprobs": []]],
            ]))
        }
        state.currentTurnPersistedItemCount = 3
        state.modelResponses = [ModelResponse(output: [getTextMessage("test")], usage: Usage(), responseId: "resp_1")]
        try model.enqueue(.output([getTextMessage("done")]))

        let result = try Runner.runStreamed(agent, state: state)

        #expect(
            result.toState().currentTurnPersistedItemCount == 3,
            "The counter should reflect persisted items, not the generated item count."
        )
        try await consumeStream(result)
    }

    // MARK: - Serialization

    @Test(arguments: ToolKind.allCases)
    func testDeserializeInterruptionsPreserveToolCalls(toolKind: ToolKind) async throws {
        let tool: any Tool
        let rawCall: TResponseOutputItem
        let userInput: String
        switch toolKind {
        case .shell:
            tool = try ShellTool(executor: { _ in .text("output") }, needsApproval: .dynamic { _, _, _ in true })
            rawCall = makeShellCall("call_shell_1", idValue: "shell_1", commands: ["echo test"])
            userInput = "run shell"
        case .applyPatch:
            tool = try ApplyPatchTool(editor: RecordingEditor(), needsApproval: .dynamic { _, _, _ in true })
            rawCall = makeApplyPatchDict("call_apply_1")
            userInput = "update file"
        }
        let (model, agent) = try makeModelAndAgent(tools: [tool])

        try await assertRoundtripToolName(agent, model, rawCall, toolKind.rawValue, userInput: userInput)
    }

    @Test(arguments: [true, false])
    func testDeserializeInterruptionsPreserveMcpTools(includeProviderData: Bool) async throws {
        let (_, agent) = try makeModelAndAgent()
        let mcpApprovalItem = makeMCPApprovalItem(agent, callId: "mcp-approval-1", includeProviderData: includeProviderData)
        let state = makeStateWithInterruptions(agent, [mcpApprovalItem])

        let stateJSON = try state.toJSON()

        let deserialized = try await RunState.fromJSON(agent, stateJSON)
        let interruptions = deserialized.getInterruptions()
        #expect(!interruptions.isEmpty, "Interruptions should be preserved after deserialization")
        #expect(interruptions.first?.toolName == "test_mcp_tool", "MCP tool approval should be preserved, not converted to function")
    }

    @Test func testHostedMcpApprovalWithUnknownLegacyIdentityRequiresReapproval() throws {
        let agent = makeAgent()
        let contextWrapper = makeContextWrapper()
        let approvalItem = makeMCPApprovalItem(
            agent,
            callId: "mcp-123",
            toolName: nil,
            providerData: ["type": "mcp_approval_request"],
            includeName: false,
            useCallId: false
        )

        expectError(ModelBehaviorError.self, containing: "canonical invocation identity") { try contextWrapper.approveTool(approvalItem) }
    }

    @Test func testPreserveToolOutputTypesDuringSerialization() async throws {
        let (_, agent) = try makeModelAndAgent()
        let computerOutput: JSONObject = [
            "type": "computer_call_output",
            "call_id": "call_computer_1",
            "output": ["type": "computer_screenshot", "image_url": "base64_screenshot_data"],
        ]
        try await assertToolOutputRoundtrip(agent, computerOutput, "computer_call_output", output: "screenshot_data")

        let shellOutput: JSONObject = [
            "type": "local_shell_call_output",
            "id": "shell_1",
            "call_id": "call_shell_1",
            "output": "command output",
        ]
        try await assertToolOutputRoundtrip(agent, shellOutput, "local_shell_call_output")
    }

    // MARK: - Callable approval arguments

    @Test(arguments: ["", #"{"subject": "refund""#, "null", "[]", #"{"amount": NaN}"#, #"{"amount": Infinity}"#, #"{"amount": -Infinity}"#])
    func testCallableFunctionApprovalFailsClosedForInvalidArguments(arguments: String) async throws {
        let approvalInputs = SharedBox<[JSONObject]>([])
        let toolInputs = SharedBox<[String]>([])
        let tool = try FunctionTool(
            name: "send_email",
            description: "Send an email.",
            paramsJSONSchema: ["type": "object", "properties": [:]],
            onInvokeTool: { _, rawArguments in
                toolInputs.mutate { $0.append(rawArguments) }
                return "sent"
            },
            needsApproval: .dynamic { _, params, _ in
                approvalInputs.mutate { $0.append(params) }
                return false
            }
        )
        let (model, agent) = try makeModelAndAgent(tools: [tool])
        try model.enqueue(.output([makeFunctionToolCall(tool.name, callId: "call-invalid", arguments: arguments)]))

        let result = try await Runner.run(agent, input: .text("send an email"))

        #expect(result.interruptions.count == 1)
        #expect(result.interruptions.first?.toolName == tool.name)
        #expect(approvalInputs.value.isEmpty)
        #expect(toolInputs.value.isEmpty)
    }

    @Test func testCallableFunctionApprovalReceivesValidObjectArguments() async throws {
        let approvalInputs = SharedBox<[JSONObject]>([])
        let toolInputs = SharedBox<[String]>([])
        let tool = try FunctionTool(
            name: "send_email",
            description: "Send an email.",
            paramsJSONSchema: ["type": "object", "properties": ["subject": ["type": "string"]]],
            onInvokeTool: { _, rawArguments in
                toolInputs.mutate { $0.append(rawArguments) }
                return "sent"
            },
            needsApproval: .dynamic { _, params, _ in
                approvalInputs.mutate { $0.append(params) }
                return false
            }
        )
        let arguments = #"{"subject": "status update"}"#
        let (model, agent) = try makeModelAndAgent(tools: [tool])
        try model.extend([
            .output([makeFunctionToolCall(tool.name, callId: "call-valid", arguments: arguments)]),
            .output([getTextMessage("done")]),
        ])

        let result = try await Runner.run(agent, input: .text("send an email"))

        #expect(result.finalOutput as? String == "done")
        #expect(!approvalInputs.value.isEmpty)
        #expect(approvalInputs.value.allSatisfy { $0 == ["subject": "status update"] })
        #expect(toolInputs.value == [arguments])
    }

    // MARK: - Resolved decisions on the run path

    @Test func testExecutePathSkipsNeedsApprovalCheckerWhenStatusResolved() async throws {
        struct ValueArgs: Codable, Sendable { var value: String }
        let checkerCalls = SharedBox<[String]>([])
        let sensitive = try FunctionTool.create(
            name: "sensitive",
            argumentsType: ValueArgs.self,
            needsApproval: .dynamic { _, _, callId in
                let count = checkerCalls.mutate { calls -> Int in
                    calls.append(callId)
                    return calls.count
                }
                if count > 1 { throw TestError("resolved approval must bypass needs_approval") }
                return true
            }
        ) { _, args in "ran:\(args.value)" }
        let model = try ScriptedModel()
        let agent = Agent(name: "agent", tools: [sensitive], model: .instance(model))
        try model.extend([
            .output([makeFunctionToolCall(sensitive.name, callId: "call-1", arguments: #"{"value":"x"}"#)]),
            .output([getTextMessage("done")]),
        ])

        let first = try await Runner.run(agent, input: .text("hello"))
        #expect(first.interruptions.count == 1)
        #expect(checkerCalls.value == ["call-1"])

        let state = first.toState()
        try state.approve(try #require(first.interruptions.first))
        let resumed = try await Runner.run(agent, state: state)

        #expect(resumed.finalOutput as? String == "done")
        #expect(checkerCalls.value == ["call-1"])
        #expect(resumed.newItems.contains { ($0 as? ToolCallOutputItem)?.output as? String == "ran:x" })
    }

    @Test func testResumeCheckpointsToolOutputBeforeToolUseBehaviorFailure() async throws {
        struct ValueArgs: Codable, Sendable { var value: String }
        let executions = SharedBox<[String]>([])
        let sensitive = try FunctionTool.create(name: "sensitive", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
            executions.mutate { $0.append(args.value) }
            return "ran:\(args.value)"
        }
        let model = try ScriptedModel()
        let agent = Agent(
            name: "agent",
            tools: [sensitive],
            model: .instance(model),
            toolUseBehavior: .custom { _, _ in throw TestError("tool use behavior failed") }
        )
        try model.extend([
            .output([makeFunctionToolCall(sensitive.name, callId: "call-1", arguments: #"{"value":"x"}"#)]),
            .output([getTextMessage("done")]),
        ])

        let first = try await Runner.run(agent, input: .text("hello"))
        let state = first.toState()
        try state.approve(try #require(first.interruptions.first))

        let error = await expectErrorAsync(TestError.self) { _ = try await Runner.run(agent, state: state) }
        #expect(error?.message == "tool use behavior failed")

        #expect(executions.value == ["x"])
        #expect(state.generatedItems.contains { ($0 as? ToolCallOutputItem)?.output as? String == "ran:x" })

        agent.toolUseBehavior = .runLLMAgain
        let resumed = try await Runner.run(agent, state: state)

        #expect(resumed.finalOutput as? String == "done")
        #expect(executions.value == ["x"])
    }
}
