import Foundation

// Port of src/agents/util/_approvals.py.
//
// Swift: `evaluate_needs_approval_setting` is not ported; `ToolApprovalRequirement` is already
// either a Bool or a policy, so the invalid-setting error cannot occur.

/// Returns parsed object arguments, or `nil` when an approval policy cannot inspect them.
///
/// Swift: Python rejects the `NaN` and `Infinity` constants while parsing. The Swift parser
/// accepts them, so any non-finite number is rejected after parsing.
func parseFunctionToolArguments(_ arguments: String?) -> JSONObject? {
    guard let arguments, !arguments.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
          case .object(let parsed)? = try? JSONValue.parse(arguments),
          isFiniteJSON(.object(parsed)) else {
        return nil
    }
    return parsed
}

private func isFiniteJSON(_ value: JSONValue) -> Bool {
    switch value {
    case .double(let double): return double.isFinite
    case .array(let items): return items.allSatisfy(isFiniteJSON)
    case .object(let object): return object.values.allSatisfy(isFiniteJSON)
    default: return true
    }
}

/// Evaluates a policy against unchanged input, retaining one prepared invocation.
func evaluateFunctionToolApproval(
    _ tool: FunctionTool,
    context: RunContextWrapper,
    arguments: String,
    callId: String
) async throws -> FunctionToolApproval {
    var result = FunctionToolApproval(.requireApproval, tool: tool, arguments: arguments)
    let needsApproval: Bool
    switch tool.needsApproval {
    case .required(let value):
        needsApproval = value
    case .dynamic(let policy):
        guard let params = parseFunctionToolArguments(arguments) else { return result }
        if let invoker = tool.failureHandlingInvoker, let prepared = invoker.prepareArguments(arguments, toolName: tool.name) {
            guard prepared.unchanged else { return result }
            result.prepared = prepared
        }
        needsApproval = try await policy(context, params, callId)
    }
    try result.checkInvocation(tool, arguments: arguments)
    result.outcome = needsApproval ? .requireApproval : .invoke
    if needsApproval { result.prepared = nil }
    return result
}
