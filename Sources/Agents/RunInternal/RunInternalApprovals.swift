import Foundation

// Port of src/agents/run_internal/approvals.py: execution-time helpers that coordinate approval
// placeholders and normalization.

/// Emits a synthetic tool output so users see why an approval failed.
func appendApprovalErrorOutput(
    generatedItems: inout [RunItem],
    agent: Agent,
    toolCall: JSONObject?,
    toolName: String,
    callId: String?,
    message: String,
    toolOrigin: ToolOrigin? = nil
) throws {
    let errorToolCall = buildFunctionToolCallForApprovalError(toolCall, toolName: toolName, callId: callId)
    generatedItems.append(ToolCallOutputItem(
        agent: agent,
        rawItem: try ItemHelpers.toolCallOutputItem(errorToolCall, output: message),
        output: message,
        toolOrigin: toolOrigin
    ))
}

/// Keeps only approval items from a mixed interruption payload.
func filterToolApprovals(_ interruptions: [Any]) -> [ToolApprovalItem] {
    interruptions.compactMap { $0 as? ToolApprovalItem }
}

/// Returns approvals from a step that may or may not contain interruptions.
func approvalsFromStep(_ step: NextStep?) -> [ToolApprovalItem] {
    step?.interruptions ?? []
}

/// Appends tool outputs to model input while skipping approval placeholders.
func appendInputItemsExcludingApprovals(
    _ baseInput: inout [TResponseInputItem],
    _ items: [RunItem],
    reasoningItemIdPolicy: ReasoningItemIdPolicy? = nil
) throws {
    for item in items {
        if let converted = try runItemToInputItem(item, reasoningItemIdPolicy: reasoningItemIdPolicy) {
            baseInput.append(converted)
        }
    }
}

/// Coerces raw tool call payloads into a normalized `function_call` for approval errors.
func buildFunctionToolCallForApprovalError(_ toolCall: JSONObject?, toolName: String, callId: String?) -> ResponseFunctionToolCall {
    if let toolCall, let typed = ResponseFunctionToolCall(json: toolCall) {
        return typed
    }
    let namespace = toolCall?.string("namespace").flatMap { $0.isEmpty ? nil : $0 }
    return ResponseFunctionToolCall(
        callId: callId ?? "unknown",
        name: toolName,
        arguments: "{}",
        namespace: namespace,
        status: "completed"
    )
}
