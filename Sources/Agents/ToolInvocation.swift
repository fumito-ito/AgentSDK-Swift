import Foundation

// Port of src/agents/_tool_invocation.py: canonical identities for provider tool invocations.
//
// The approval scope and fingerprint digests are written to `RunState` snapshots, so they hash the
// same text as upstream: `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.

private let toolInvocationTypes: Set<String> = [
    "apply_patch_call",
    "computer_call",
    "custom_tool_call",
    "function_call",
    "local_shell_call",
    "mcp_approval_request",
    "shell_call",
]

private let toolOutputTypes: [String: String] = [
    "apply_patch_call_output": "apply_patch_call",
    "computer_call_output": "computer_call",
    "custom_tool_call_output": "custom_tool_call",
    "function_call_output": "function_call",
    "local_shell_call_output": "local_shell_call",
    "mcp_approval_response": "mcp_approval_request",
    "shell_call_output": "shell_call",
]

private let semanticFields = [
    "type",
    "name",
    "namespace",
    "server_label",
    "arguments",
    "input",
    "action",
    "actions",
    "pending_safety_checks",
    "operation",
    "operations",
    "environment",
    "caller",
]

/// The canonical identity of one tool invocation.
struct ToolInvocationIdentity: Hashable, Sendable {
    /// The invocation item type, for example `function_call`.
    var type: String
    /// The provider call ID (the request ID for hosted MCP approvals).
    var callId: String
    /// A digest of the tool the invocation is authorized against.
    var approvalScope: String
    /// A digest of the invocation's semantic payload, including its arguments.
    var fingerprint: String
}

/// Returns whether a value names a canonical tool invocation type.
func isToolInvocationType(_ value: JSONValue?) -> Bool {
    guard let type = value?.stringValue else { return false }
    return toolInvocationTypes.contains(type)
}

/// Returns whether a value is a canonical lowercase SHA-256 digest.
func isToolInvocationDigest(_ value: JSONValue?) -> Bool {
    guard let digest = value?.stringValue else { return false }
    return digest.count == 64 && digest.allSatisfy { "0123456789abcdef".contains($0) }
}

private func normalizeInvocationValue(_ value: JSONValue, excludeNull: Bool = false) -> JSONValue {
    switch value {
    case .object(let object):
        var normalized = JSONObject()
        for (key, item) in object where !(excludeNull && item.isNull) {
            normalized[key] = normalizeInvocationValue(item, excludeNull: excludeNull)
        }
        return .object(normalized)
    case .array(let items):
        return .array(items.map { normalizeInvocationValue($0, excludeNull: excludeNull) })
    default:
        return value
    }
}

/// Parses string arguments as JSON so formatting differences do not change the fingerprint.
private func normalizeInvocationArguments(_ value: JSONValue) -> JSONValue {
    guard case .string(let text) = value else { return normalizeInvocationValue(value) }
    // Swift: the parser already rejects the NaN/Infinity constants upstream refuses here.
    guard let parsed = try? JSONValue.parse(text) else { return value }
    return normalizeInvocationValue(parsed)
}

private func unwrapHostedMCPApproval(_ rawItem: JSONObject) -> JSONObject? {
    guard rawItem.type == "hosted_tool_call",
          case .object(let providerData)? = rawItem["provider_data"],
          providerData.type == "mcp_approval_request" else {
        return rawItem
    }
    guard let requestIdentity = getHostedMCPApprovalRequestIdentity(rawItem) else { return nil }
    var merged = rawItem
    for (key, value) in providerData {
        merged[key] = value
    }
    if let requestId = requestIdentity.requestId {
        merged["id"] = .string(requestId)
    } else {
        merged.removeValue(forKey: "id")
    }
    if let toolName = requestIdentity.toolName {
        merged["name"] = .string(toolName)
    }
    return merged
}

private func lookupKeyIdentityValue(_ key: FunctionToolLookupKey) -> JSONValue {
    switch key {
    case .bare(let name): return ["bare", .string(name)]
    case .namespaced(let namespace, let name): return ["namespaced", .string(namespace), .string(name)]
    case .deferredTopLevel(let name): return ["deferred_top_level", .string(name)]
    }
}

private func invocationFingerprint(_ payload: JSONObject) -> String {
    sha256Hex(JSONValue.object(payload).pythonJSONString(sortedKeys: true, compact: true, ensureASCII: false))
}

/// Returns the invocation type, provider call ID, approval scope, and semantic fingerprint.
func toolInvocationIdentity(
    _ rawItem: JSONObject,
    toolLookupKey: FunctionToolLookupKey? = nil,
    toolName: String? = nil,
    invocationRole: String? = nil
) -> ToolInvocationIdentity? {
    guard let (invocationType, callId) = toolInvocationCallId(rawItem), let callId,
          let (scopeType, approvalScope) = toolInvocationApprovalScope(
              rawItem,
              toolLookupKey: toolLookupKey,
              toolName: toolName,
              invocationRole: invocationRole
          ),
          scopeType == invocationType,
          let mapping = unwrapHostedMCPApproval(rawItem) else {
        return nil
    }

    switch invocationType {
    case "function_call", "mcp_approval_request":
        if mapping["arguments"] == nil { return nil }
    case "custom_tool_call":
        guard let name = mapping.string("name"), !name.isEmpty, mapping["input"] != nil else { return nil }
    case "computer_call", "local_shell_call", "shell_call":
        if mapping["action"] == nil { return nil }
    case "apply_patch_call":
        if mapping["operation"] == nil, mapping["operations"] == nil { return nil }
    default:
        break
    }

    var semanticPayload: JSONObject = ["approval_scope": .string(approvalScope)]
    for field in semanticFields {
        if invocationType == "function_call", field == "name" || field == "namespace" { continue }
        guard let value = mapping[field], !value.isNull else { continue }
        semanticPayload[field] = field == "arguments"
            ? normalizeInvocationArguments(value)
            : normalizeInvocationValue(value, excludeNull: true)
    }
    return ToolInvocationIdentity(
        type: invocationType,
        callId: callId,
        approvalScope: approvalScope,
        fingerprint: invocationFingerprint(semanticPayload)
    )
}

/// Returns a recognized invocation type and its valid non-empty call ID, if present.
func toolInvocationCallId(_ rawItem: JSONObject) -> (type: String, callId: String?)? {
    guard let mapping = unwrapHostedMCPApproval(rawItem), let invocationType = mapping.type,
          toolInvocationTypes.contains(invocationType) else {
        return nil
    }
    let candidate = invocationType == "mcp_approval_request" ? mapping.string("id") : mapping.string("call_id")
    return (invocationType, candidate?.isEmpty == false ? candidate : nil)
}

/// Returns the stable authorization scope for a recognized tool invocation.
func toolInvocationApprovalScope(
    _ rawItem: JSONObject,
    toolLookupKey: FunctionToolLookupKey? = nil,
    toolName: String? = nil,
    invocationRole: String? = nil
) -> (type: String, scope: String)? {
    guard let mapping = unwrapHostedMCPApproval(rawItem), let invocationType = mapping.type,
          toolInvocationTypes.contains(invocationType) else {
        return nil
    }
    var payload: JSONObject = ["type": .string(invocationType)]
    if let invocationRole {
        payload["invocation_role"] = .string(invocationRole)
    }
    switch invocationType {
    case "function_call":
        guard let lookupKey = toolLookupKey ?? getFunctionToolLookupKeyForCall(mapping) else { return nil }
        payload["tool_lookup_key"] = lookupKeyIdentityValue(lookupKey)
    case "mcp_approval_request":
        guard let name = mapping.string("name"), !name.isEmpty,
              let serverLabel = mapping.string("server_label"), !serverLabel.isEmpty else {
            return nil
        }
        payload["name"] = .string(name)
        payload["server_label"] = .string(serverLabel)
    default:
        if let resolved = (toolName?.isEmpty == false ? toolName : nil) ?? mapping.string("name"), !resolved.isEmpty {
            payload["name"] = .string(resolved)
        }
    }
    return (invocationType, invocationFingerprint(payload))
}

/// Returns whether an item represents a hosted MCP approval request.
func isMCPApprovalInvocation(_ rawItem: JSONObject) -> Bool {
    unwrapHostedMCPApproval(rawItem)?.type == "mcp_approval_request"
}

/// Returns the invocation type and call ID completed by a tool output item.
func toolOutputIdentity(_ rawItem: JSONObject) -> (type: String, callId: String)? {
    guard let outputType = rawItem.type, let invocationType = toolOutputTypes[outputType] else { return nil }
    let candidate = outputType == "mcp_approval_response" ? rawItem.string("approval_request_id") : rawItem.string("call_id")
    guard let callId = candidate, !callId.isEmpty else { return nil }
    return (invocationType, callId)
}
