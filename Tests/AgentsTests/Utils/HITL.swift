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

/// Container for approval-driven tool scenarios.
struct ApprovalScenario {
    var tool: any Tool
    var rawCall: TResponseOutputItem
    var finalOutput: TResponseOutputItem
    var assertResult: (RunResult) -> Void
}

/// Container for scenarios with pending approvals.
struct PendingScenario {
    var tool: any Tool
    var rawCall: TResponseOutputItem
    var assertResult: ((RunResult) -> Void)?
}

/// Runs once with a tool call, serializes the state, and deserializes it.
func roundtripInterruptionsViaRun(_ agent: Agent, _ model: ScriptedModel, _ rawCall: TResponseOutputItem, userInput: String = "test") async throws -> [ToolApprovalItem] {
    try model.enqueue(.output([rawCall]))
    let result = try await Runner.run(agent, input: .text(userInput))
    #expect(!result.interruptions.isEmpty, "expected an interruption")
    let state = result.toState()
    let deserialized = try await RunState.fromJSON(agent, try state.toJSON())
    return deserialized.getInterruptions()
}

/// Asserts that deserialized interruptions keep the tool name intact.
func assertRoundtripToolName(
    _ agent: Agent,
    _ model: ScriptedModel,
    _ rawCall: TResponseOutputItem,
    _ expectedToolName: String,
    userInput: String,
    sourceLocation: SourceLocation = #_sourceLocation
) async throws {
    let interruptions = try await roundtripInterruptionsViaRun(agent, model, rawCall, userInput: userInput)
    #expect(!interruptions.isEmpty, "Interruptions should be preserved after deserialization", sourceLocation: sourceLocation)
    #expect(
        interruptions.first?.toolName == expectedToolName,
        "\(expectedToolName) tool approval should be preserved, not converted to function",
        sourceLocation: sourceLocation
    )
}

/// Ensures tool outputs keep their type through serialization and deserialization.
func assertToolOutputRoundtrip(
    _ agent: Agent,
    _ rawOutput: JSONObject,
    _ expectedType: String,
    output: any Sendable = "command output",
    sourceLocation: SourceLocation = #_sourceLocation
) async throws {
    let state = RunState(context: makeContextWrapper(), originalInput: .text("test"), startingAgent: agent, maxTurns: 3)
    state.generatedItems = [ToolCallOutputItem(agent: agent, rawItem: rawOutput, output: output)]

    let json = try state.toJSON()

    let generatedItemsJSON = json["generated_items"]?.arrayValue ?? []
    #expect(generatedItemsJSON.count == 1, "\(expectedType) item should be serialized", sourceLocation: sourceLocation)
    let serializedType = generatedItemsJSON.first?.objectValue?["raw_item"]?.objectValue?.string("type")
    #expect(serializedType == expectedType, "Serialization should not coerce tool outputs.", sourceLocation: sourceLocation)

    let deserialized = try await RunState.fromJSON(agent, json)

    #expect(deserialized.generatedItems.count == 1, "\(expectedType) item should be deserialized.", sourceLocation: sourceLocation)
    let item = try #require(deserialized.generatedItems.first as? ToolCallOutputItem, sourceLocation: sourceLocation)
    #expect(item.rawItem.type == expectedType, "Serialization should preserve the tool output type.", sourceLocation: sourceLocation)
}

/// Runs once, then resumes from the produced state.
func runAndResume(_ agent: Agent, _ model: ScriptedModel, _ rawCall: TResponseOutputItem, userInput: String) async throws -> RunResult {
    try model.enqueue(.output([rawCall]))
    let first = try await Runner.run(agent, input: .text(userInput))
    return try await Runner.run(agent, state: first.toState())
}

/// Approves the first interruption on the result and returns the updated state.
func approveFirstInterruption(_ result: RunResult, alwaysApprove: Bool = false) throws -> RunState {
    let interruption = try #require(result.interruptions.first, "expected an approval interruption")
    let state = result.toState()
    try state.approve(interruption, alwaysApprove: alwaysApprove)
    return state
}

/// Approves the first interruption and resumes the run.
func resumeAfterFirstApproval(_ agent: Agent, _ result: RunResult, alwaysApprove: Bool = false) async throws -> RunResult {
    let state = try approveFirstInterruption(result, alwaysApprove: alwaysApprove)
    return try await Runner.run(agent, state: state)
}

/// Runs, approves the first interruption, and resumes.
func runAndResumeAfterApproval(
    _ agent: Agent,
    _ model: ScriptedModel,
    _ rawCall: TResponseOutputItem,
    _ finalOutput: TResponseOutputItem,
    userInput: String
) async throws -> RunResult {
    try model.enqueue(.output([rawCall]))
    let first = try await Runner.run(agent, input: .text(userInput))
    let state = try approveFirstInterruption(first, alwaysApprove: true)
    try model.enqueue(.output([finalOutput]))
    return try await Runner.run(agent, state: state)
}

/// Returns the `ToolCallOutputItem`s whose raw item has `outputType`.
func collectToolOutputs(_ items: [RunItem], outputType: String) -> [ToolCallOutputItem] {
    items.compactMap { $0 as? ToolCallOutputItem }.filter { $0.rawItem.type == outputType }
}

/// Drains all stream events to completion.
func consumeStream(_ result: RunResultStreaming) async throws {
    for try await _ in result.streamEvents() {}
}

/// Asserts the result contains exactly one approval interruption and returns it.
@discardableResult
func assertSingleApprovalInterruption(_ result: SingleStepResult, toolName: String? = nil) throws -> ToolApprovalItem {
    guard case .interruption(let interruptions) = result.nextStep else {
        Issue.record("expected an interruption")
        throw TestError("expected an interruption")
    }
    #expect(interruptions.count == 1)
    let interruption = try #require(interruptions.first)
    if let toolName { #expect(interruption.toolName == toolName) }
    return interruption
}

/// Approval policy that always requires a HITL decision.
let requireApproval: ToolApprovalRequirement = .dynamic { _, _, _ in true }

/// Queues a function call turn followed by a follow-up turn on the scripted model.
func queueFunctionCallAndText(
    _ model: ScriptedModel,
    _ functionCall: TResponseOutputItem,
    firstTurnExtra: [TResponseOutputItem] = [],
    followup: [TResponseOutputItem] = []
) throws {
    #expect(functionCall.type == "function_call", "queueFunctionCallAndText expects a function call item")
    try model.extend([ModelStep(output: [functionCall] + firstTurnExtra), ModelStep(output: followup)])
}

/// Runs, resumes, and asserts pending approvals stay pending.
@discardableResult
func assertPendingResume(
    _ tool: any Tool,
    _ model: ScriptedModel,
    _ rawCall: TResponseOutputItem,
    userInput: String,
    outputType: String,
    sourceLocation: SourceLocation = #_sourceLocation
) async throws -> RunResult {
    let agent = makeAgent(model: model, tools: [tool])

    let resumed = try await runAndResume(agent, model, rawCall, userInput: userInput)

    #expect(!resumed.interruptions.isEmpty, "pending approval should remain after resuming", sourceLocation: sourceLocation)
    #expect(resumed.interruptions.contains { $0.toolName == tool.name }, sourceLocation: sourceLocation)
    #expect(
        collectToolOutputs(resumed.newItems, outputType: outputType).isEmpty,
        "\(outputType) should not execute without approval",
        sourceLocation: sourceLocation
    )
    return resumed
}

/// Builds a hosted MCP tool call payload for approvals.
func makeMCPRawItem(
    callId: String = "call_mcp_1",
    includeProviderData: Bool = true,
    toolName: String = "test_mcp_tool",
    providerData: JSONObject? = nil,
    includeName: Bool = true,
    useCallId: Bool = true
) -> JSONObject {
    var rawItem: JSONObject = ["type": "hosted_tool_call"]
    if includeName { rawItem["name"] = .string(toolName) }
    if includeProviderData {
        rawItem[useCallId ? "call_id" : "id"] = .string(callId)
        rawItem["provider_data"] = .object(providerData ?? ["type": "mcp_approval_request", "id": "req-1", "server_label": "test_server"])
    } else {
        rawItem["id"] = .string(callId)
    }
    return rawItem
}

/// Creates a `ToolApprovalItem` for MCP or hosted tool calls.
func makeMCPApprovalItem(
    _ agent: Agent,
    callId: String = "call_mcp_1",
    includeProviderData: Bool = true,
    toolName: String? = "test_mcp_tool",
    providerData: JSONObject? = nil,
    includeName: Bool = true,
    useCallId: Bool = true
) -> ToolApprovalItem {
    let rawItem = makeMCPRawItem(
        callId: callId,
        includeProviderData: includeProviderData,
        toolName: toolName ?? "unknown_mcp_tool",
        providerData: providerData,
        includeName: includeName,
        useCallId: useCallId
    )
    return ToolApprovalItem(agent: agent, rawItem: rawItem, toolName: toolName)
}
