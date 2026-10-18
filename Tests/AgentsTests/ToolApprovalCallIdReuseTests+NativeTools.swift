import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_tool_approval_call_id_reuse.py (native tool cases).
extension ToolApprovalCallIdReuseTests {
    // MARK: - Native tools

    @Test(arguments: ["custom", "function"])
    func testCompletedApplyPatchFallbackRunStateRoundTrip(fallbackType: String) async throws {
        let editor = RecordingEditor()
        let tool = try ApplyPatchTool(editor: editor)
        let operation = #"{"type": "update_file", "path": "safe.txt", "diff": "-old\\n+new\\n"}"#
        let call: JSONObject = fallbackType == "custom"
            ? ["type": "custom_tool_call", "name": "apply_patch", "call_id": "patch_0", "input": .string(operation)]
            : ["type": "function_call", "name": "apply_patch", "call_id": "patch_0", "arguments": .string(operation)]
        let model = try ScriptedModel(steps: [.output([call])])
        try model.enqueue(.output([getTextMessage("done")]))
        let agent = Agent(name: "agent", tools: [tool], model: .instance(model))

        let result = try await Runner.run(agent, input: "update the file")
        let restored = try await RunState.fromJSON(agent, try result.toState().toJSON())

        #expect(result.finalOutput as? String == "done")
        #expect(editor.operations.map(\.path) == ["safe.txt"])
        #expect(restored.context.toolInvocations["patch_0"]?.completed == true)
    }

    @Test func testStreamedFunctionApplyPatchReplayEmitsOneToolCalledEvent() async throws {
        let editor = RecordingEditor()
        let operation = #"{"type": "update_file", "path": "safe.txt", "diff": "-old\\n+new\\n"}"#
        let call: JSONObject = ["type": "function_call", "name": "apply_patch", "call_id": "patch_0", "arguments": .string(operation)]
        let model = try ScriptedModel()
        try model.extend([.output([call]), .output([call]), .output([getTextMessage("done")])])
        let agent = Agent(name: "agent", tools: [try ApplyPatchTool(editor: editor)], model: .instance(model))
        let events = SharedBox<[StreamEvent]>([])

        let streamed = try await run(agent, .text("update the file"), mode: .streamed, events: events)

        let toolCalled = events.value.compactMap { event -> RunItemStreamEvent? in
            guard case .runItem(let item) = event, item.name == "tool_called" else { return nil }
            return item
        }
        #expect(streamed.finalOutput as? String == "done")
        #expect(editor.operations.map(\.path) == ["safe.txt"])
        #expect(toolCalled.count == 1)
        #expect(toolCalled.first.flatMap { ($0.item as? ToolCallItem)?.callId } == "patch_0")
    }

    @Test(arguments: [
        (makeShellCall("call_0", commands: ["echo safe"]), "safe_shell", "other_shell"),
        (makeApplyPatchDict("call_0"), "safe_patch", "other_patch"),
    ])
    func testNativeToolApprovalScopeRejectsResolvedToolReplacement(rawItem: JSONObject, firstToolName: String, replacementToolName: String) throws {
        let context = RunContextWrapper(context: ())
        let approvalItem = ToolApprovalItem(agent: Agent(name: "agent"), rawItem: rawItem, toolName: firstToolName)
        try context.approveTool(approvalItem)

        #expect(try context.approvedToolInvocationStatus(rawItem, toolName: firstToolName) != nil)
        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            try context.approvedToolInvocationStatus(rawItem, toolName: replacementToolName)
        }
    }

    func recordingShell(_ name: String, needsApproval: Bool, executions: SharedBox<[String]>) throws -> ShellTool {
        try ShellTool(
            executor: { request in
                executions.mutate { $0.append(contentsOf: request.data.action.commands) }
                return .text("ok")
            },
            name: name,
            needsApproval: .required(needsApproval)
        )
    }

    @Test func testCustomNamedShellStickyApprovalAppliesToFreshCallIds() async throws {
        let executed = SharedBox<[String]>([])
        let model = try ScriptedModel()
        try model.extend([
            .output([makeShellCall("call_0", commands: ["echo first"])]),
            .output([makeShellCall("call_1", commands: ["echo second"])]),
            .output([getTextMessage("done")]),
        ])
        let agent = Agent(name: "agent", tools: [try recordingShell("safe_shell", needsApproval: true, executions: executed)], model: .instance(model))

        let first = try await Runner.run(agent, input: "run commands")
        let state = first.toState()
        try state.approve(first.interruptions[0], alwaysApprove: true)
        let resumed = try await Runner.run(agent, state: state)

        #expect(resumed.finalOutput as? String == "done")
        #expect(executed.value == ["echo first", "echo second"])
    }

    @Test func testCustomNamedShellReplacementRejectsCompletedCallIdReplay() async throws {
        let firstExecutions = SharedBox<[String]>([])
        let replacementExecutions = SharedBox<[String]>([])
        let call = makeShellCall("call_0", commands: ["echo safe"])
        let model = try ScriptedModel(steps: [.output([call])])
        let agent = Agent(name: "agent", tools: [try recordingShell("safe_shell", needsApproval: true, executions: firstExecutions)], model: .instance(model))

        let first = try await Runner.run(agent, input: "run command")
        let state = first.toState()
        try state.approve(first.interruptions[0])
        try model.enqueue(.output([getTextMessage("done")]))
        let completed = try await Runner.run(agent, state: state)

        agent.tools = [try recordingShell("other_shell", needsApproval: false, executions: replacementExecutions)]
        try model.enqueue(.output([call]))

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Runner.run(agent, state: completed.toState())
        }

        #expect(firstExecutions.value == ["echo safe"])
        #expect(replacementExecutions.value.isEmpty)
    }

    @Test func testCancelledRejectionFormatterLeavesInvocationExecuted() async throws {
        let toolCall: JSONObject = ["type": "function_call", "name": "approval_tool", "call_id": "call_rejected", "arguments": "{}"]
        let context = RunContextWrapper(context: ())
        #expect(
            try context.toolInvocationStatus(toolCall)
                == ToolInvocationStatus(invocationType: "function_call", callId: "call_rejected", completed: false, executed: false)
        )
        let formatterEntered = AsyncEvent()
        let runConfig = RunConfig(toolErrorFormatter: { _ in
            formatterEntered.set()
            try await AsyncEvent().wait()
            return "rejected"
        })

        let task = Task {
            try await resolveApprovalRejectionMessage(
                contextWrapper: context, runConfig: runConfig, toolType: "function",
                toolName: "approval_tool", callId: "call_rejected", toolCall: toolCall
            )
        }
        try await formatterEntered.wait()
        task.cancel()
        _ = try? await task.value

        #expect(
            try context.toolInvocationStatus(toolCall)
                == ToolInvocationStatus(invocationType: "function_call", callId: "call_rejected", completed: false, executed: true)
        )
    }
}
