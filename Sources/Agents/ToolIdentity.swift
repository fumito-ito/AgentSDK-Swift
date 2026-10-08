import Foundation

// Port of src/agents/_tool_identity.py: canonical helpers for function-tool naming, namespaces,
// lookup, and approval keys.

/// The collision-free lookup key for a function tool name/namespace pair.
public enum FunctionToolLookupKey: Sendable, Hashable {
    case bare(String)
    case namespaced(namespace: String, name: String)
    case deferredTopLevel(String)

    /// The tool name carried by the key (`lookup_key[-1]` upstream).
    public var name: String {
        switch self {
        case .bare(let name), .deferredTopLevel(let name): return name
        case .namespaced(_, let name): return name
        }
    }

    public var json: JSONObject {
        switch self {
        case .bare(let name): return ["kind": "bare", "name": .string(name)]
        case .namespaced(let namespace, let name):
            return ["kind": "namespaced", "namespace": .string(namespace), "name": .string(name)]
        case .deferredTopLevel(let name): return ["kind": "deferred_top_level", "name": .string(name)]
        }
    }

    public init?(json: JSONValue?) {
        guard case .object(let data)? = json, let kind = data.string("kind"),
              let name = data.string("name"), !name.isEmpty else { return nil }
        switch kind {
        case "bare": self = .bare(name)
        case "deferred_top_level": self = .deferredTopLevel(name)
        case "namespaced":
            guard let namespace = data.string("namespace"), !namespace.isEmpty else { return nil }
            self = .namespaced(namespace: namespace, name: name)
        default: return nil
        }
    }
}

/// Validated identity fields from a hosted MCP approval request.
public struct HostedMCPApprovalRequestIdentity: Sendable, Equatable {
    public var requestId: String?
    public var serverLabel: String?
    public var toolName: String?

    /// The persistent identity when all required fields are available.
    public var approvalIdentity: (serverLabel: String, toolName: String)? {
        guard let serverLabel, let toolName else { return nil }
        return (serverLabel, toolName)
    }
}

private func nonEmpty(_ value: String?) -> String? {
    guard let value, !value.isEmpty else { return nil }
    return value
}

/// Returns strictly validated identity fields for a hosted MCP approval request.
public func getHostedMCPApprovalRequestIdentity(_ rawItem: JSONObject) -> HostedMCPApprovalRequestIdentity? {
    let requestId: String?
    let toolName: String?
    let request: JSONObject
    switch rawItem.type {
    case "mcp_approval_request":
        request = rawItem
        requestId = nonEmpty(request.string("id"))
        toolName = nonEmpty(request.string("name"))
    case "hosted_tool_call":
        guard case .object(let providerData)? = rawItem["provider_data"], providerData.type == "mcp_approval_request" else {
            return nil
        }
        request = providerData
        if providerData["id"] == nil {
            requestId = nonEmpty(rawItem.string("call_id")) ?? nonEmpty(rawItem.string("id"))
        } else {
            requestId = nonEmpty(providerData.string("id"))
        }
        toolName = nonEmpty(providerData.string("name")) ?? nonEmpty(rawItem.string("name"))
    default:
        return nil
    }
    return HostedMCPApprovalRequestIdentity(
        requestId: requestId,
        serverLabel: nonEmpty(request.string("server_label")),
        toolName: toolName
    )
}

/// Returns the canonical call ID for a tool approval item.
public func getToolApprovalItemCallId(_ item: ToolApprovalItem) -> String? {
    if let hosted = getHostedMCPApprovalRequestIdentity(item.rawItem) {
        return hosted.requestId
    }
    return nonEmpty(item.rawItem.string("call_id")) ?? nonEmpty(item.rawItem.string("id"))
}

/// Returns `namespace.name` when a namespace exists, otherwise `name`.
public func toolQualifiedName(_ name: String?, namespace: String? = nil) -> String? {
    guard let name, !name.isEmpty else { return nil }
    if let namespace, !namespace.isEmpty {
        return "\(namespace).\(name)"
    }
    return name
}

/// Returns a display-friendly tool name, collapsing synthetic deferred namespaces.
public func toolTraceName(_ name: String?, namespace: String? = nil) -> String? {
    if isReservedSyntheticToolNamespace(name, namespace) {
        return name
    }
    return toolQualifiedName(name, namespace: namespace)
}

/// Returns `true` when a namespace matches the reserved deferred top-level wire shape.
public func isReservedSyntheticToolNamespace(_ name: String?, _ namespace: String?) -> Bool {
    guard let name, !name.isEmpty, let namespace, !namespace.isEmpty else { return false }
    return namespace == name
}

public func getToolCallNamespace(_ toolCall: JSONObject) -> String? {
    nonEmpty(toolCall.string("namespace"))
}

public func getToolCallName(_ toolCall: JSONObject) -> String? {
    nonEmpty(toolCall.string("name"))
}

public func getToolCallQualifiedName(_ toolCall: JSONObject) -> String? {
    toolQualifiedName(getToolCallName(toolCall), namespace: getToolCallNamespace(toolCall))
}

public func getToolCallTraceName(_ toolCall: JSONObject) -> String? {
    toolTraceName(getToolCallName(toolCall), namespace: getToolCallNamespace(toolCall))
}

/// Returns the collision-free lookup key for a function tool name/namespace pair.
public func getFunctionToolLookupKey(_ toolName: String?, _ toolNamespace: String? = nil) -> FunctionToolLookupKey? {
    guard let toolName, !toolName.isEmpty else { return nil }
    if isReservedSyntheticToolNamespace(toolName, toolNamespace) {
        return .deferredTopLevel(toolName)
    }
    if let toolNamespace, !toolNamespace.isEmpty {
        return .namespaced(namespace: toolNamespace, name: toolName)
    }
    return .bare(toolName)
}

public func getFunctionToolLookupKeyForCall(_ toolCall: JSONObject) -> FunctionToolLookupKey? {
    getFunctionToolLookupKey(getToolCallName(toolCall), getToolCallNamespace(toolCall))
}

public func getFunctionToolLookupKeyForTool(_ tool: FunctionTool) -> FunctionToolLookupKey? {
    guard !tool.name.isEmpty else { return nil }
    if isDeferredTopLevelFunctionTool(tool) {
        return .deferredTopLevel(tool.name)
    }
    return getFunctionToolLookupKey(tool.name, tool.toolNamespace)
}

/// Returns `true` when the tool is deferred-loading without an explicit namespace.
public func isDeferredTopLevelFunctionTool(_ tool: FunctionTool) -> Bool {
    tool.deferLoading && (tool.toolNamespace?.isEmpty ?? true) && !tool.name.isEmpty
}

/// Returns the canonical dispatch key for a function tool.
public func getFunctionToolDispatchName(_ tool: FunctionTool) -> String? {
    toolQualifiedName(tool.name, namespace: tool.toolNamespace)
}

public func getFunctionToolQualifiedName(_ tool: FunctionTool) -> String? {
    getFunctionToolDispatchName(tool)
}

public func getFunctionToolTraceName(_ tool: FunctionTool) -> String? {
    guard !tool.name.isEmpty else { return nil }
    return toolTraceName(tool.name, namespace: tool.toolNamespace)
}

/// Returns all lookup keys that should resolve this function tool.
public func getFunctionToolLookupKeys(_ tool: FunctionTool) -> [FunctionToolLookupKey] {
    guard !tool.name.isEmpty else { return [] }
    var keys: [FunctionToolLookupKey] = []
    if let dispatchKey = getFunctionToolLookupKey(tool.name, tool.toolNamespace), !isDeferredTopLevelFunctionTool(tool) {
        keys.append(dispatchKey)
    }
    if isDeferredTopLevelFunctionTool(tool), !keys.contains(.deferredTopLevel(tool.name)) {
        keys.append(.deferredTopLevel(tool.name))
    }
    return keys
}

/// Allows bare-name approval aliases only for deferred top-level tools without visible peers.
public func shouldAllowBareNameApprovalAlias(_ tool: FunctionTool, allTools: [any Tool]) -> Bool {
    guard !tool.name.isEmpty, isDeferredTopLevelFunctionTool(tool) else { return false }
    for case let candidate as FunctionTool in allTools where candidate !== tool && candidate.name == tool.name {
        if candidate.toolNamespace != nil { continue }
        if candidate.deferLoading { continue }
        return false
    }
    return true
}

/// Rejects reserved namespace shapes that collide with deferred top-level tool calls.
public func validateFunctionToolNamespaceShape(_ toolName: String?, _ toolNamespace: String?) throws {
    guard isReservedSyntheticToolNamespace(toolName, toolNamespace) else { return }
    let reserved = toolQualifiedName(toolName, namespace: toolNamespace) ?? toolName ?? "unknown_tool"
    throw UserError(
        "Responses tool-search reserves the synthetic namespace `\(reserved)` for deferred "
            + "top-level function tools. Rename the namespace or tool name to avoid ambiguous dispatch."
    )
}

private enum CollisionEntry {
    case tool(Int, FunctionTool)
    case handoff(Int, Handoff)

    var isTool: Bool {
        if case .tool = self { return true }
        return false
    }
}

private func formatToolNameCollisionMessage(_ name: String, _ entries: [CollisionEntry]) -> String {
    var derivedOwners: [(agent: String, parameter: String)] = []
    for entry in entries {
        switch entry {
        case .tool(_, let tool):
            if let identity = tool.agentToolDefaultIdentity, tool.name == identity.toolName,
               !derivedOwners.contains(where: { $0.agent == identity.agentName }) {
                derivedOwners.append((identity.agentName, "tool_name"))
            }
        case .handoff(_, let handoff):
            if let identity = handoff.defaultToolIdentity, handoff.toolName == identity.toolName,
               !derivedOwners.contains(where: { $0.agent == identity.agentName }) {
                derivedOwners.append((identity.agentName, "tool_name_override"))
            }
        }
    }
    if entries.count == 2, derivedOwners.count == 2 {
        let prior = derivedOwners[0]
        let current = derivedOwners[1]
        let configurationType: String
        let nameLabel: String
        let overrideInstruction: String
        if prior.parameter == "tool_name", current.parameter == "tool_name" {
            configurationType = "agent tool"
            nameLabel = "tool name"
            overrideInstruction = "`tool_name=`"
        } else if prior.parameter == "tool_name_override", current.parameter == "tool_name_override" {
            configurationType = "handoff"
            nameLabel = "handoff tool name"
            overrideInstruction = "`tool_name_override=`"
        } else {
            configurationType = "agent routing"
            nameLabel = "tool name"
            overrideInstruction = "`tool_name=` or `tool_name_override=`"
        }
        return "Ambiguous \(configurationType) configuration: agents '\(prior.agent)' and '\(current.agent)' "
            + "both derive the \(nameLabel) `\(name)`. Pass an explicit \(overrideInstruction) to one of them."
    }
    let allTools = entries.allSatisfy(\.isTool)
    let allHandoffs = entries.allSatisfy { !$0.isTool }
    if allTools {
        return "Ambiguous function tool configuration: the tool name `\(name)` is used by multiple tools. "
            + "Assign a unique routed name to every colliding function tool with `name_override=`, "
            + "`tool_name=`, or a namespace."
    }
    if allHandoffs {
        return "Ambiguous handoff configuration: the handoff tool name `\(name)` is used by multiple handoffs. "
            + "Pass a unique `tool_name_override=` to each handoff."
    }
    return "Ambiguous tool routing configuration: the tool name `\(name)` is used by both a function tool "
        + "and a handoff. Assign a unique routed name to every colliding function tool and handoff with "
        + "`name_override=`, `tool_name=`, `tool_name_override=`, or a namespace."
}

/// Resolves bare function-tool and handoff name collisions before model exposure.
public func resolveToolNameCollisions(
    _ tools: [any Tool],
    handoffs: [Handoff] = [],
    collisionPolicy: ToolNameCollisionPolicy
) throws -> (tools: [any Tool], handoffs: [Handoff]) {
    try validateFunctionToolLookupConfiguration(tools)

    var order: [String] = []
    var owners: [String: [CollisionEntry]] = [:]
    for (index, tool) in tools.enumerated() {
        guard let functionTool = tool as? FunctionTool,
              case .bare(let name)? = getFunctionToolLookupKeyForTool(functionTool) else { continue }
        if owners[name] == nil { order.append(name) }
        owners[name, default: []].append(.tool(index, functionTool))
    }
    for (index, handoff) in handoffs.enumerated() where !handoff.toolName.isEmpty {
        if owners[handoff.toolName] == nil { order.append(handoff.toolName) }
        owners[handoff.toolName, default: []].append(.handoff(index, handoff))
    }

    var retainedTools = Set(tools.indices)
    var retainedHandoffs = Set(handoffs.indices)
    for name in order {
        let entries = owners[name] ?? []
        guard entries.count >= 2 else { continue }
        if collisionPolicy == .error {
            throw UserError(formatToolNameCollisionMessage(name, entries))
        }
        if AgentsDebug.dontLogToolData {
            logger.warning("Tool name collision detected. Assign unique routed tool names or enable tool data logging for details.")
        } else {
            logger.warning(formatToolNameCollisionMessage(name, entries))
        }
        let winner = entries.last { !$0.isTool } ?? entries.last!
        for entry in entries {
            switch (entry, winner) {
            case let (.tool(index, _), .tool(winnerIndex, _)) where index == winnerIndex:
                continue
            case let (.handoff(index, _), .handoff(winnerIndex, _)) where index == winnerIndex:
                continue
            case let (.tool(index, _), _):
                retainedTools.remove(index)
            case let (.handoff(index, _), _):
                retainedHandoffs.remove(index)
            }
        }
    }
    return (
        tools.enumerated().filter { retainedTools.contains($0.offset) }.map(\.element),
        handoffs.enumerated().filter { retainedHandoffs.contains($0.offset) }.map(\.element)
    )
}

/// Rejects function-tool combinations that are ambiguous on the Responses wire.
public func validateFunctionToolLookupConfiguration(_ tools: [any Tool]) throws {
    var qualifiedOwners: [String: FunctionTool] = [:]
    var deferredOwners: Set<String> = []
    for case let tool as FunctionTool in tools {
        try validateFunctionToolNamespaceShape(tool.name, tool.toolNamespace)
        if isDeferredTopLevelFunctionTool(tool) {
            if deferredOwners.contains(tool.name) {
                throw UserError(
                    "Ambiguous function tool configuration: the deferred top-level tool name `\(tool.name)` "
                        + "is used by multiple tools. Rename one of the deferred-loading top-level function tools "
                        + "to avoid ambiguous dispatch."
                )
            }
            deferredOwners.insert(tool.name)
        }
        guard let qualified = getFunctionToolQualifiedName(tool) else { continue }
        guard let prior = qualifiedOwners[qualified] else {
            qualifiedOwners[qualified] = tool
            continue
        }
        if tool.toolNamespace == nil, prior.toolNamespace == nil { continue }
        throw UserError(
            "Ambiguous function tool configuration: the qualified name `\(qualified)` is used by multiple tools. "
                + "Rename the namespace-wrapped function or dotted top-level tool to avoid ambiguous dispatch."
        )
    }
}

/// Builds a function-tool lookup map using last-wins precedence.
public func buildFunctionToolLookupMap(_ tools: [any Tool]) throws -> [FunctionToolLookupKey: FunctionTool] {
    try validateFunctionToolLookupConfiguration(tools)
    var map: [FunctionToolLookupKey: FunctionTool] = [:]
    for case let tool as FunctionTool in tools {
        for key in getFunctionToolLookupKeys(tool) {
            map[key] = tool
        }
    }
    return map
}

/// Returns approval keys for a tool name/namespace pair.
public func getFunctionToolApprovalKeys(
    toolName: String?,
    toolNamespace: String? = nil,
    allowBareNameAlias: Bool = false,
    toolLookupKey: FunctionToolLookupKey? = nil,
    preferLegacySameNameNamespace: Bool = false,
    includeLegacyDeferredKey: Bool = false
) -> [String] {
    guard let toolName, !toolName.isEmpty else { return [] }
    var keys: [String] = []
    var lookupKey = toolLookupKey
    if lookupKey == nil, !(preferLegacySameNameNamespace && isReservedSyntheticToolNamespace(toolName, toolNamespace)) {
        lookupKey = getFunctionToolLookupKey(toolName, toolNamespace)
    }
    let qualifiedName = toolQualifiedName(toolName, namespace: toolNamespace)
    if allowBareNameAlias { keys.append(toolName) }
    if let lookupKey {
        let key: String?
        switch lookupKey {
        case .namespaced(let namespace, let name): key = toolQualifiedName(name, namespace: namespace)
        case .deferredTopLevel(let name): key = "deferred_top_level:\(name)"
        case .bare(let name): key = name
        }
        if let key, !keys.contains(key) { keys.append(key) }
        if includeLegacyDeferredKey, case .deferredTopLevel = lookupKey, let qualifiedName, !keys.contains(qualifiedName) {
            keys.append(qualifiedName)
        }
    } else if let qualifiedName, !keys.contains(qualifiedName) {
        keys.append(qualifiedName)
    }
    if keys.isEmpty { keys.append(toolName) }
    return keys
}

/// Strips synthetic namespaces from deferred top-level tool calls.
public func normalizeToolCallForFunctionTool(_ toolCall: JSONObject, tool: FunctionTool) -> JSONObject {
    guard isDeferredTopLevelFunctionTool(tool), getToolCallName(toolCall) == tool.name,
          getToolCallNamespace(toolCall) == tool.name else { return toolCall }
    return toolCall.removing("namespace")
}

/// Fills an absent call namespace from a persisted lookup key.
public func restoreToolCallRoutingIdentity(_ toolCall: JSONObject, lookupKey: FunctionToolLookupKey?) -> JSONObject {
    guard let lookupKey, getToolCallName(toolCall) == lookupKey.name, getToolCallNamespace(toolCall) == nil else {
        return toolCall
    }
    var restored = toolCall
    switch lookupKey {
    case .bare: return toolCall
    case .namespaced(let namespace, _): restored["namespace"] = .string(namespace)
    case .deferredTopLevel(let name): restored["namespace"] = .string(name)
    }
    return restored
}
