import Foundation

// Port of the invocation validation and de-duplication parts of
// src/agents/run_internal/tool_planning.py.
//
// Swift: upstream identifies the response item behind each tool run by sharing the raw item object
// (`id(raw_item)`). Swift raw items are values, so each tool run keeps a reference to its call item
// (`callItem`) and skipped invocations are tracked by that item's identity.

private let nativeToolNames = [
    "apply_patch_call": "apply_patch",
    "computer_call": "computer",
    "local_shell_call": "local_shell",
    "shell_call": "shell",
]

/// A call identity used when a tool call's ID may be missing (upstream `_tool_call_identity`).
private struct ToolCallIdentityKey: Hashable {
    var callId: String?
    var name: String?
    var arguments: String?
}

private func toolCallIdentityKey(_ raw: JSONObject) -> ToolCallIdentityKey {
    let callId = [raw.string("call_id"), raw.string("id")].compactMap { $0 }.first { !$0.isEmpty }
    let arguments = raw["arguments"].flatMap { $0.isNull ? nil : $0 } ?? raw["input"]
    let hashable: String?
    switch arguments {
    case nil, .null?: hashable = nil
    case .string(let text)?: hashable = text
    case let value?: hashable = value.jsonString(sortedKeys: true)
    }
    return ToolCallIdentityKey(callId: callId, name: raw.string("name"), arguments: hashable)
}

private func callItemId(_ item: RunItem?) -> ObjectIdentifier? {
    item.map { ObjectIdentifier($0) }
}

/// Rejects changed same-ID MCP siblings and coalesces exact duplicates.
func preflightMCPApprovalRequests(_ requests: [ToolRunMCPApprovalRequest]) throws -> (requests: [ToolRunMCPApprovalRequest], skipped: Set<ObjectIdentifier>) {
    var seenByCall: [String: ToolInvocationIdentity] = [:]
    var deduped: [ToolRunMCPApprovalRequest] = []
    var skipped: Set<ObjectIdentifier> = []
    for request in requests {
        guard let identity = toolInvocationIdentity(request.requestItem.json) else {
            deduped.append(request)
            continue
        }
        let callKey = "\(identity.type)\u{0}\(identity.callId)"
        guard let existing = seenByCall[callKey] else {
            seenByCall[callKey] = identity
            deduped.append(request)
            continue
        }
        if existing != identity {
            throw ModelBehaviorError(
                "Model reused an approval-gated tool call ID for a different invocation. "
                    + "Use a unique call ID for each approval-gated invocation."
            )
        }
        if let id = callItemId(request.callItem) { skipped.insert(id) }
    }
    return (deduped, skipped)
}

/// Returns new items while skipping tool call duplicates already seen by identity.
func dedupeToolCallItems(existingItems: [RunItem], newItems: [RunItem], skipped: Set<ObjectIdentifier>) -> [RunItem] {
    var existingCallKeys: Set<ToolCallIdentityKey> = []
    for case let item as ToolCallItem in existingItems {
        existingCallKeys.insert(toolCallIdentityKey(item.rawItem))
    }
    var deduped: [RunItem] = []
    for item in newItems {
        if item is ToolCallItem || item is HandoffCallItem || item is MCPApprovalRequestItem {
            if skipped.contains(ObjectIdentifier(item)) { continue }
            let key = toolCallIdentityKey(item.rawItem)
            if existingCallKeys.contains(key) { continue }
            existingCallKeys.insert(key)
        }
        deduped.append(item)
    }
    return deduped
}

/// Validates approval-bound calls and records their committed outputs.
func registerToolCallItems(_ contextWrapper: RunContextWrapper, _ items: [RunItem], validateInvocations: Bool = true) throws {
    for case let item as ToolApprovalItem in items {
        try contextWrapper.restorePendingApprovalBinding(item)
    }
    for item in items {
        let isCallItem = item is ToolCallItem || item is HandoffCallItem || item is MCPApprovalRequestItem
        guard isCallItem else { continue }
        if !validateInvocations, item is ToolCallItem || item is MCPApprovalRequestItem {
            let rawType = item.rawItem.type
            let toolName = item.rawItem.string("name") ?? rawType.flatMap { nativeToolNames[$0] }
            if let toolName {
                try contextWrapper.restorePendingApprovalBinding(ToolApprovalItem(
                    agent: item.agent,
                    rawItem: item.rawItem,
                    toolName: toolName,
                    toolNamespace: getToolCallNamespace(item.rawItem),
                    toolLookupKey: rawType == "function_call" ? getFunctionToolLookupKeyForCall(item.rawItem) : nil
                ))
            }
        }
        if !validateInvocations { continue }
        if item is ToolCallItem, item.rawItem.type == "function_call" {
            // Resolved function calls are validated from the plan, where canonical routing identity
            // is available. Raw calls can omit deferred-loading routing metadata.
            continue
        }
        try contextWrapper.toolInvocationStatus(
            item.rawItem,
            toolName: (item as? ToolCallItem)?.toolName,
            invocationRole: item is HandoffCallItem ? "handoff" : nil
        )
    }
    for item in items where item is ToolCallOutputItem || item is HandoffOutputItem || item is MCPApprovalResponseItem {
        contextWrapper.markToolCallCompleted(item.rawItem)
    }
}

/// Validates unresolved function calls before any sibling tool starts.
func validateUnresolvedFunctionCalls(_ contextWrapper: RunContextWrapper, _ runs: [ToolRunFunctionNotFound]) throws {
    for run in runs {
        try contextWrapper.toolInvocationStatus(run.toolCall.json)
    }
}

/// Validates and coalesces one response's tool invocations before user callbacks run.
@discardableResult
func dedupeProcessedResponseInvocations(
    _ processedResponse: inout ProcessedResponse,
    contextWrapper: RunContextWrapper,
    existingItems: [RunItem],
    deferredBindingValidation: Set<ObjectIdentifier> = [],
    filterCompleted: Bool = true
) throws -> Set<ObjectIdentifier> {
    let completedHistoricalInvocations = try completedHistoricalInvocations(existingItems, contextWrapper: contextWrapper)
    var deduplicator = ResponseInvocationDeduplicator(
        contextWrapper: contextWrapper,
        completedHistoricalInvocations: completedHistoricalInvocations,
        deferredBindingValidation: deferredBindingValidation,
        filterCompleted: filterCompleted
    )
    let preflight = try preflightMCPApprovalRequests(processedResponse.mcpApprovalRequests)
    processedResponse.mcpApprovalRequests = preflight.requests
    deduplicator.skipped = preflight.skipped

    processedResponse.functions = try processedResponse.functions.filter {
        try deduplicator.shouldKeep($0.toolCall.json, callItem: $0.callItem, toolLookupKey: getFunctionToolLookupKeyForTool($0.functionTool))
    }
    processedResponse.handoffs = try processedResponse.handoffs.filter {
        try deduplicator.shouldKeep($0.toolCall.json, callItem: $0.callItem, invocationRole: "handoff")
    }
    processedResponse.functionToolsNotFound = try processedResponse.functionToolsNotFound.filter {
        try deduplicator.shouldKeep($0.toolCall.json, callItem: $0.callItem)
    }
    processedResponse.computerActions = try processedResponse.computerActions.filter {
        try deduplicator.shouldKeep($0.toolCall.json, callItem: $0.callItem, toolName: $0.computerTool.name)
    }
    processedResponse.customToolCalls = try processedResponse.customToolCalls.filter {
        try deduplicator.shouldKeep($0.toolCall.json, callItem: $0.callItem, toolName: $0.customTool.name)
    }
    processedResponse.localShellCalls = try processedResponse.localShellCalls.filter {
        try deduplicator.shouldKeep($0.toolCall.json, callItem: $0.callItem, toolName: $0.localShellTool.name)
    }
    processedResponse.shellCalls = try processedResponse.shellCalls.filter {
        try deduplicator.shouldKeep($0.toolCall, callItem: $0.callItem, toolName: $0.shellTool.name)
    }
    processedResponse.applyPatchCalls = try processedResponse.applyPatchCalls.filter {
        try deduplicator.shouldKeep($0.toolCall, callItem: $0.callItem, toolName: $0.applyPatchTool.name)
    }
    processedResponse.mcpApprovalRequests = try processedResponse.mcpApprovalRequests.filter {
        try deduplicator.shouldKeep($0.requestItem.json, callItem: $0.callItem)
    }

    let skipped = deduplicator.skipped
    processedResponse.newItems = dropSkippedCallItems(processedResponse.newItems, skipped: skipped)
    return skipped
}

/// Removes skipped call items and the reasoning items that lead directly into them.
private func dropSkippedCallItems(_ items: [RunItem], skipped: Set<ObjectIdentifier>) -> [RunItem] {
    var dropped: Set<Int> = []
    for (index, item) in items.enumerated() where item is ToolCallItem || item is HandoffCallItem || item is MCPApprovalRequestItem {
        if skipped.contains(ObjectIdentifier(item)) { dropped.insert(index) }
    }
    var droppedReasoning: Set<Int> = []
    for index in items.indices.reversed() where items[index] is ReasoningItem {
        for next in (index + 1)..<items.count where !(items[next] is ReasoningItem) {
            if dropped.contains(next) { droppedReasoning.insert(index) }
            break
        }
    }
    let excluded = dropped.union(droppedReasoning)
    return items.enumerated().filter { !excluded.contains($0.offset) }.map(\.element)
}

/// Collects the identities of historical invocations whose outputs are already committed.
private func completedHistoricalInvocations(_ existingItems: [RunItem], contextWrapper: RunContextWrapper) throws -> [String: ToolInvocationIdentity] {
    var completedOutputKeys: Set<String> = []
    for item in existingItems {
        if let output = toolOutputIdentity(item.rawItem) {
            completedOutputKeys.insert("\(output.type)\u{0}\(output.callId)")
        }
    }
    let allowLegacy = contextWrapper.allowLegacyApprovalBindingReconstruction
    let invocations = contextWrapper.toolInvocations
    var completed: [String: ToolInvocationIdentity] = [:]
    for item in existingItems {
        let approval = item as? ToolApprovalItem
        let toolCallItem = item as? ToolCallItem
        var identity = toolInvocationIdentity(
            item.rawItem,
            toolLookupKey: approval?.toolLookupKey,
            toolName: toolCallItem?.toolName ?? approval?.toolName,
            invocationRole: item is HandoffCallItem ? "handoff" : nil
        )
        if allowLegacy, let toolCallItem, toolCallItem.toolName == nil,
           let callIdentity = toolInvocationCallId(item.rawItem), let callId = callIdentity.callId,
           let record = invocations[callId], record.completed, record.invocationType == callIdentity.type {
            identity = ToolInvocationIdentity(type: record.invocationType, callId: callId, approvalScope: record.approvalScope, fingerprint: record.fingerprint)
        }
        guard let identity, completedOutputKeys.contains("\(identity.type)\u{0}\(identity.callId)") else { continue }
        if let previous = completed[identity.callId], !previous.sameInvocation(as: identity) {
            throw ModelBehaviorError(
                "Run history reused a tool call ID for different completed invocations. Use a unique call ID for each tool invocation."
            )
        }
        completed[identity.callId] = identity
    }
    return completed
}

extension ToolInvocationIdentity {
    /// Compares the `(type, call_id, fingerprint)` triple upstream uses for invocation identity.
    func sameInvocation(as other: ToolInvocationIdentity) -> Bool {
        type == other.type && callId == other.callId && fingerprint == other.fingerprint
    }
}

/// The per-response state of `_dedupe_processed_response_invocations`'s `should_keep`.
private struct ResponseInvocationDeduplicator {
    let contextWrapper: RunContextWrapper
    let completedHistoricalInvocations: [String: ToolInvocationIdentity]
    let deferredBindingValidation: Set<ObjectIdentifier>
    let filterCompleted: Bool
    var skipped: Set<ObjectIdentifier> = []
    var currentResponseInvocations: [String: ToolInvocationIdentity] = [:]
    var uncanonicalResponseCallIds: Set<String> = []

    private static let reusedInResponse = "Model reused a tool call ID for a different invocation in one response. Use a unique call ID for each tool invocation."

    init(
        contextWrapper: RunContextWrapper,
        completedHistoricalInvocations: [String: ToolInvocationIdentity],
        deferredBindingValidation: Set<ObjectIdentifier>,
        filterCompleted: Bool
    ) {
        self.contextWrapper = contextWrapper
        self.completedHistoricalInvocations = completedHistoricalInvocations
        self.deferredBindingValidation = deferredBindingValidation
        self.filterCompleted = filterCompleted
    }

    private mutating func skip(_ callItem: RunItem?) {
        if let id = callItemId(callItem) { skipped.insert(id) }
    }

    mutating func shouldKeep(
        _ rawItem: JSONObject,
        callItem: RunItem?,
        toolLookupKey: FunctionToolLookupKey? = nil,
        toolName: String? = nil,
        invocationRole: String? = nil
    ) throws -> Bool {
        let callIdentity = toolInvocationCallId(rawItem)
        if let callIdentity, callIdentity.callId == nil {
            throw ModelBehaviorError("Tool invocations require a non-empty string call ID before execution.")
        }
        guard let identity = toolInvocationIdentity(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: invocationRole) else {
            try contextWrapper.toolInvocationStatus(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: invocationRole)
            if let callId = callIdentity?.callId {
                if currentResponseInvocations[callId] != nil || uncanonicalResponseCallIds.contains(callId) {
                    throw ModelBehaviorError(Self.reusedInResponse)
                }
                uncanonicalResponseCallIds.insert(callId)
            }
            return true
        }
        if uncanonicalResponseCallIds.contains(identity.callId) {
            throw ModelBehaviorError(Self.reusedInResponse)
        }
        let historical = completedHistoricalInvocations[identity.callId]
        if let historical {
            guard historical.sameInvocation(as: identity) else {
                throw ModelBehaviorError(
                    "Model reused a completed tool call ID for a different invocation. Use a unique call ID for each tool invocation."
                )
            }
            if filterCompleted {
                skip(callItem)
                return false
            }
        }
        if let previous = currentResponseInvocations[identity.callId] {
            guard previous.sameInvocation(as: identity) else { throw ModelBehaviorError(Self.reusedInResponse) }
            skip(callItem)
            return false
        }
        currentResponseInvocations[identity.callId] = identity

        if callItemId(callItem).map(deferredBindingValidation.contains) != true {
            let binding: ToolInvocationStatus?
            do {
                binding = try contextWrapper.toolInvocationStatus(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: invocationRole)
            } catch let error as ModelBehaviorError {
                // A completed exact sibling with the same provider ID can predate approval binding,
                // so preserve that released cross-kind resume behavior. Changed content has no exact
                // historical identity and still fails closed.
                if historical?.sameInvocation(as: identity) == true { return true }
                throw error
            }
            if filterCompleted, binding?.completed == true {
                skip(callItem)
                return false
            }
            if let binding, binding.executed, !binding.completed {
                throw ModelBehaviorError(
                    "A tool call already executed, but its output was not committed. Start a new run instead of retrying the invocation."
                )
            }
        }
        return true
    }
}
