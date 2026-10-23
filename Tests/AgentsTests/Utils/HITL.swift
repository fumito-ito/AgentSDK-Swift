import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Port of tests/utils/hitl.py.

/// Builds a `shell_call` payload with optional overrides.
func makeShellCall(_ callId: String, idValue: String? = nil, commands: [String]? = nil, status: String = "in_progress") -> TResponseOutputItem {
    [
        "type": "shell_call",
        "id": .string(idValue ?? callId),
        "call_id": .string(callId),
        "status": .string(status),
        "action": ["type": "exec", "commands": .array((commands ?? ["echo test"]).map(JSONValue.string)), "timeout_ms": 1000],
    ]
}

/// Creates an apply_patch_call payload.
func makeApplyPatchDict(_ callId: String, diff: String = "-a\n+b\n") -> TResponseOutputItem {
    [
        "type": "apply_patch_call",
        "call_id": .string(callId),
        "operation": ["type": "update_file", "path": "test.md", "diff": .string(diff)],
    ]
}

/// Editor that records operations for testing.
final class RecordingEditor: ApplyPatchEditor, @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [ApplyPatchOperation] = []

    var operations: [ApplyPatchOperation] { lock.locked { recorded } }

    private func record(_ operation: ApplyPatchOperation) {
        lock.locked { recorded.append(operation) }
    }

    func createFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? {
        record(operation)
        return ApplyPatchResult(status: "completed", output: "Created \(operation.path)")
    }

    func updateFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? {
        record(operation)
        return ApplyPatchResult(status: "completed", output: "Updated \(operation.path)")
    }

    func deleteFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? {
        record(operation)
        return ApplyPatchResult(status: "completed", output: "Deleted \(operation.path)")
    }
}

/// The default message sent to the model when a tool call is rejected.
let hitlRejectionMessage = "Tool execution was not approved."

/// Creates an empty `RunContextWrapper` for HITL tests.
func makeContextWrapper() -> RunContextWrapper {
    RunContextWrapper(context: JSONObject())
}

/// Creates a `RunState` primed with interruptions.
func makeStateWithInterruptions(
    _ agent: Agent,
    _ interruptions: [ToolApprovalItem],
    originalInput: String = "test",
    maxTurns: Int = 10
) -> RunState {
    let state = RunState(context: makeContextWrapper(), originalInput: .text(originalInput), startingAgent: agent, maxTurns: maxTurns)
    state.currentStep = .interruption(interruptions: interruptions, responseAccepted: false)
    return state
}

/// Creates a `function_call` payload for HITL scenarios.
func makeFunctionToolCall(_ name: String, callId: String = "call-1", arguments: String = "{}", namespace: String? = nil) -> TResponseOutputItem {
    ResponseFunctionToolCall(callId: callId, name: name, arguments: arguments, namespace: namespace).json
}

/// Builds a test agent with an optional model and tools.
func makeAgent(model: (any Model)? = nil, tools: [any Tool] = [], name: String = "TestAgent") -> Agent {
    Agent(name: name, tools: tools, model: model.map(AgentModel.instance))
}

/// Builds a `ScriptedModel` with a paired agent for HITL tests.
func makeModelAndAgent(tools: [any Tool] = [], name: String = "TestAgent") throws -> (ScriptedModel, Agent) {
    let model = try ScriptedModel()
    return (model, makeAgent(model: model, tools: tools, name: name))
}

/// Runs until an interruption, optionally mutates the state, then resumes.
func runAndResumeWithMutation(
    _ agent: Agent,
    _ model: ScriptedModel,
    _ turnOutputs: [[TResponseOutputItem]],
    userInput: String,
    mutateState: ((RunState, ToolApprovalItem) throws -> Void)? = nil
) async throws -> (RunResult, RunResult) {
    for output in turnOutputs {
        try model.enqueue(.output(output))
    }
    let first = try await Runner.run(agent, input: .text(userInput))
    let interruption = try #require(first.interruptions.first, "expected an approval interruption")
    let state = first.toState()
    try mutateState?(state, interruption)
    let resumed = try await Runner.run(agent, state: state)
    return (first, resumed)
}
