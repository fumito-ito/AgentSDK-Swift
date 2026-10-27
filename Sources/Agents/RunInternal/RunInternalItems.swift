import Foundation

// Port of src/agents/run_internal/items.py: input normalization helpers and builders for
// synthetic run items used during tool execution.
//
// Swift: the nested-history provenance helpers (`NestedHistoryOwnedItem*`,
// `reconcile_nested_history_owned_input_after_rewrite`, ...) are not ported; see
// porting/manifest.json.

let rejectionMessage = defaultApprovalRejectionMessage
let toolCallSessionDescriptionKey = "_agents_tool_description"
let toolCallSessionTitleKey = "_agents_tool_title"

let toolCallToOutputType: [String: String] = [
    "program": "program_output",
    "function_call": "function_call_output",
    "custom_tool_call": "custom_tool_call_output",
    "shell_call": "shell_call_output",
    "apply_patch_call": "apply_patch_call_output",
    "computer_call": "computer_call_output",
    "local_shell_call": "local_shell_call_output",
    "tool_search_call": "tool_search_output",
]

/// Items that must retain their original position relative to required follower items.
private let dedupeEarliestAnchorItemTypes: Set<String> = Set(toolCallToOutputType.keys).union(["mcp_approval_request", "reasoning"])

private let programOwnedHostedItemTypes: Set<String> = [
    "hosted_tool_call", "file_search_call", "web_search_call", "code_interpreter_call",
    "image_generation_call", "mcp_list_tools", "mcp_call", "mcp_approval_request", "mcp_approval_response",
]

/// Converts a run item to model input, optionally stripping reasoning IDs.
func runItemToInputItem(_ runItem: RunItem, reasoningItemIdPolicy: ReasoningItemIdPolicy? = nil) throws -> TResponseInputItem? {
    if runItem.type == "tool_approval_item" { return nil }
    var item = try runItem.toInputItem()
    if item["status"] == .null || (item.contains("status") && item["status"]?.isNull == true) {
        item.removeValue(forKey: "status")
    }
    if reasoningItemIdPolicy == .omit, runItem.type == "reasoning_item" {
        return withoutReasoningItemId(item)
    }
    return item
}

/// Converts run items to model input items while skipping approvals.
func runItemsToInputItems(_ runItems: [RunItem], reasoningItemIdPolicy: ReasoningItemIdPolicy? = nil) throws -> [TResponseInputItem] {
    try runItems.compactMap { try runItemToInputItem($0, reasoningItemIdPolicy: reasoningItemIdPolicy) }
}

/// Removes tool and program call items without corresponding outputs so resumptions or retries
/// do not replay stale calls. Reasoning items that immediately precede a dropped call are also
/// removed, since the Responses API rejects reasoning items without their following item.
func dropOrphanFunctionCalls(
    _ items: [TResponseInputItem],
    pruningIndexes: Set<Int>? = nil,
    outputPruningIndexes: Set<Int>? = nil
) -> [TResponseInputItem] {
    let completed = completedCallIdsByType(items)
    let matchedAnonymousToolSearchCalls = matchedAnonymousToolSearchCallIndexes(items)
    var activeProgramCallIds: Set<String> = []
    var orphanProgramCallIds: Set<String> = []

    for (index, entry) in items.enumerated() {
        if let pruningIndexes, !pruningIndexes.contains(index) { continue }
        guard entry.type == "program", let callId = entry.string("call_id") else { continue }
        if completed["program_output"]?.contains(callId) == true { continue }
        let hasRetainedChild = items.enumerated().contains { candidateIndex, candidate in
            programCallerId(candidate) == callId && isRetainedProgramOwnedItem(candidate, candidateIndex, pruningIndexes)
        }
        if hasRetainedChild {
            activeProgramCallIds.insert(callId)
        } else {
            orphanProgramCallIds.insert(callId)
        }
    }

    var dropped: Set<Int> = []
    var reasoningTriggers: Set<Int> = []
    for (index, entry) in items.enumerated() {
        guard let entryType = entry.type else { continue }
        if let pruningIndexes, !pruningIndexes.contains(index) { continue }
        let callerId = programCallerId(entry)
        if let callerId, orphanProgramCallIds.contains(callerId) {
            dropped.insert(index)
            reasoningTriggers.insert(index)
            continue
        }
        guard let outputType = toolCallToOutputType[entryType] else { continue }
        if callerId != nil, isPendingHostedShellCall(entry) { continue }
        let callId = entry.string("call_id")
        if entryType == "program", let callId, activeProgramCallIds.contains(callId) { continue }
        if let callId, completed[outputType]?.contains(callId) == true { continue }
        if entryType == "tool_search_call", callId == nil, matchedAnonymousToolSearchCalls.contains(index) { continue }
        dropped.insert(index)
        reasoningTriggers.insert(index)
    }

    let available = availableCallIdsByOutputType(items, excludedIndexes: dropped)
    if let outputPruningIndexes {
        for index in outputPruningIndexes where !dropped.contains(index) && items.indices.contains(index) {
            let entry = items[index]
            guard let entryType = entry.type, let availableIds = available[entryType] else { continue }
            if let callId = entry.string("call_id"), !availableIds.contains(callId) {
                dropped.insert(index)
            }
        }
    }

    if dropped.isEmpty { return items }
    return dropReasoningItemsPrecedingDroppedCalls(items, dropped, reasoningTriggers)
}

private func dropReasoningItemsPrecedingDroppedCalls(
    _ items: [TResponseInputItem],
    _ dropped: Set<Int>,
    _ reasoningTriggers: Set<Int>
) -> [TResponseInputItem] {
    var dropReasoning: Set<Int> = []
    for index in stride(from: items.count - 1, through: 0, by: -1) {
        let entry = items[index]
        guard entry.type == "reasoning", !dropped.contains(index) else { continue }
        for nextIndex in (index + 1)..<max(index + 1, items.count) {
            if dropReasoning.contains(nextIndex) { continue }
            if items[nextIndex].type == "reasoning" { continue }
            if reasoningTriggers.contains(nextIndex) { dropReasoning.insert(index) }
            break
        }
    }
    let excluded = dropped.union(dropReasoning)
    return items.enumerated().filter { !excluded.contains($0.offset) }.map(\.element)
}

/// Normalizes input items for API submission.
func normalizeInputItemsForAPI(_ items: [TResponseInputItem]) -> [TResponseInputItem] {
    items.map(stripInternalInputItemMetadata)
}

/// Normalizes model input while pruning orphans only from runner-generated history.
func prepareModelInputItems(_ callerItems: [TResponseInputItem], _ generatedItems: [TResponseInputItem] = []) -> [TResponseInputItem] {
    let normalizedCaller = normalizeInputItemsForAPI(callerItems)
    if generatedItems.isEmpty { return normalizedCaller }
    return normalizedCaller + dropOrphanFunctionCalls(normalizeInputItemsForAPI(generatedItems))
}

/// Normalizes resumed list inputs and drops orphan tool calls.
func normalizeResumedInput(_ input: RunInput) -> RunInput {
    guard case .items(let items) = input else { return input }
    return .items(dropOrphanFunctionCalls(normalizeInputItemsForAPI(items)))
}

/// A hashable fingerprint used to dedupe or rewind input items across resumes.
///
/// The text matches upstream's `json.dumps(payload, sort_keys=True)` so that digests derived from
/// it (`digestInputItem`) agree with snapshots written by the Python SDK.
func fingerprintInputItem(_ item: TResponseInputItem, ignoreIdsForMatching: Bool = false) -> String {
    var payload = stripInternalInputItemMetadata(item)
    if ignoreIdsForMatching { payload.removeValue(forKey: "id") }
    return JSONValue.object(payload).pythonJSONString(sortedKeys: true)
}

/// A fixed-size digest of an input item for durable occurrence tracking.
func digestInputItem(_ item: TResponseInputItem, ignoreIdsForMatching: Bool = false) -> String {
    var coerced = stripInternalInputItemMetadata(item)
    if coerced.string("role") == "assistant" {
        if let content = coerced.string("content") {
            coerced["content"] = [["type": "output_text", "text": .string(content)]]
        }
        let status = coerced["status"]
        if status == nil || status == .null || status == "completed" {
            coerced.removeValue(forKey: "status")
        }
    }
    return sha256Hex(fingerprintInputItem(coerced, ignoreIdsForMatching: ignoreIdsForMatching))
}

/// Returns a stable identity key when items carry explicit identifiers.
private func dedupeKey(_ item: TResponseInputItem) -> String? {
    let role = item["role"]
    let itemType = item.string("type") ?? item.string("role")
    if (role != nil && role != .null) || itemType == "message" { return nil }
    let callId = item.string("call_id")
    let callTypes = Set(toolCallToOutputType.keys).union(toolCallToOutputType.values)
    if let callId, let itemType, callTypes.contains(itemType) {
        return "call_id:\(itemType):\(callId)"
    }
    var itemId = item.string("id")
    if itemId == fakeResponsesId { itemId = nil }
    if let itemId { return "id:\(itemType ?? "None"):\(itemId)" }
    if let callId { return "call_id:\(itemType ?? "None"):\(callId)" }
    if let approvalRequestId = item.string("approval_request_id") {
        return "approval_request_id:\(itemType ?? "None"):\(approvalRequestId)"
    }
    return nil
}

/// Removes session and output-only metadata before replaying items as input.
func stripInternalInputItemMetadata(_ item: TResponseInputItem) -> TResponseInputItem {
    item.removing(toolCallSessionDescriptionKey, toolCallSessionTitleKey, "created_by")
}

/// Applies the reasoning item ID policy to already-converted input items.
func applyReasoningItemIdPolicy(_ items: [TResponseInputItem], _ policy: ReasoningItemIdPolicy?) -> [TResponseInputItem] {
    guard policy == .omit else { return items }
    return items.map(withoutReasoningItemId)
}

private func withoutReasoningItemId(_ item: TResponseInputItem) -> TResponseInputItem {
    guard item.type == "reasoning", item.contains("id") else { return item }
    return item.removing("id")
}

/// Removes duplicate items that share stable identifiers to avoid re-sending tool outputs.
func deduplicateInputItems(_ items: [TResponseInputItem]) -> [TResponseInputItem] {
    var seen: Set<String> = []
    var result: [TResponseInputItem] = []
    for item in items {
        guard let key = dedupeKey(item) else {
            result.append(item)
            continue
        }
        if seen.insert(key).inserted {
            result.append(item)
        }
    }
    return result
}

/// Deduplicates by stable identifiers while keeping the latest value.
func deduplicateInputItemsPreferringLatest(_ items: [TResponseInputItem]) -> [TResponseInputItem] {
    var latestByKey: [String: TResponseInputItem] = [:]
    var anchorIndexByKey: [String: Int] = [:]
    for (index, item) in items.enumerated() {
        guard let key = dedupeKey(item) else { continue }
        latestByKey[key] = item
        let isAnchor = item.type.map(dedupeEarliestAnchorItemTypes.contains) ?? false
        if anchorIndexByKey[key] == nil || !isAnchor {
            anchorIndexByKey[key] = index
        }
    }
    var result: [TResponseInputItem] = []
    for (index, item) in items.enumerated() {
        guard let key = dedupeKey(item) else {
            result.append(item)
            continue
        }
        if anchorIndexByKey[key] == index, let latest = latestByKey[key] {
            result.append(latest)
        }
    }
    return result
}

/// Encodes SDK-generated programmatic tool errors as provider-compatible JSON objects.
func functionToolErrorOutput(toolCall: JSONObject, output: any Sendable, outputJSONSchema: JSONObject?) -> any Sendable {
    guard outputJSONSchema != nil, let string = output as? String, isProgrammaticToolCall(toolCall) else {
        return output
    }
    return JSONValue.object(["error": .string(string)]).jsonString()
}

/// Builds a `ToolCallOutputItem` representing a rejected function tool call.
func functionRejectionItem(
    agent: Agent,
    toolCall: ResponseFunctionToolCall,
    rejectionMessage message: String = rejectionMessage,
    outputJSONSchema: JSONObject? = nil,
    toolStateScope: AgentToolStateScope? = nil,
    toolOrigin: ToolOrigin? = nil
) throws -> ToolCallOutputItem {
    dropAgentToolRunResult(toolCall.json, scope: toolStateScope)
    let providerOutput = functionToolErrorOutput(toolCall: toolCall.json, output: message, outputJSONSchema: outputJSONSchema)
    return ToolCallOutputItem(
        agent: agent,
        rawItem: try ItemHelpers.toolCallOutputItem(toolCall, output: providerOutput),
        output: message,
        toolOrigin: toolOrigin
    )
}

/// Builds a `ToolCallOutputItem` representing a rejected shell call.
func shellRejectionItem(agent: Agent, callId: String, toolCall: JSONObject? = nil, rejectionMessage message: String = rejectionMessage) -> ToolCallOutputItem {
    var raw: JSONObject = [
        "type": "shell_call_output",
        "call_id": .string(callId),
        "output": [["stdout": "", "stderr": .string(message), "outcome": ["type": "exit", "exit_code": 1]]],
    ]
    if let toolCall { ItemHelpers.copyToolCallCaller(toolCall, into: &raw) }
    return ToolCallOutputItem(agent: agent, rawItem: raw, output: message)
}

/// Builds a `ToolCallOutputItem` representing a rejected apply_patch call.
func applyPatchRejectionItem(
    agent: Agent,
    callId: String,
    toolCall: JSONObject? = nil,
    outputType: String = "apply_patch_call_output",
    rejectionMessage message: String = rejectionMessage
) -> ToolCallOutputItem {
    var raw: JSONObject = ["type": .string(outputType), "call_id": .string(callId), "output": .string(message)]
    if outputType == "apply_patch_call_output" { raw["status"] = "failed" }
    if let toolCall { ItemHelpers.copyToolCallCaller(toolCall, into: &raw) }
    return ToolCallOutputItem(agent: agent, rawItem: raw, output: message)
}

/// Pulls the request id from hosted MCP approval payloads.
func extractMCPRequestId(_ rawItem: JSONObject) -> String? {
    if let hosted = getHostedMCPApprovalRequestIdentity(rawItem) { return hosted.requestId }
    if case .object(let providerData)? = rawItem["provider_data"], let id = providerData.string("id") { return id }
    return rawItem.string("id") ?? rawItem.string("call_id")
}

// MARK: - Private helpers

private func completedCallIdsByType(_ payload: [TResponseInputItem]) -> [String: Set<String>] {
    var completed: [String: Set<String>] = Dictionary(uniqueKeysWithValues: toolCallToOutputType.values.map { ($0, []) })
    for entry in payload {
        guard let type = entry.type, completed[type] != nil, let callId = entry.string("call_id") else { continue }
        completed[type]?.insert(callId)
    }
    return completed
}

private func availableCallIdsByOutputType(_ payload: [TResponseInputItem], excludedIndexes: Set<Int>) -> [String: Set<String>] {
    var available: [String: Set<String>] = Dictionary(uniqueKeysWithValues: toolCallToOutputType.values.map { ($0, []) })
    for (index, entry) in payload.enumerated() where !excludedIndexes.contains(index) {
        guard let type = entry.type, let outputType = toolCallToOutputType[type], let callId = entry.string("call_id") else { continue }
        available[outputType]?.insert(callId)
    }
    return available
}

private func programCallerId(_ entry: TResponseInputItem) -> String? {
    guard case .object(let caller)? = entry["caller"], caller.type == "program" else { return nil }
    return caller.string("caller_id")
}

private func isPendingHostedShellCall(_ entry: TResponseInputItem) -> Bool {
    guard entry.type == "shell_call" else { return false }
    let status = entry["status"]
    return status == nil || status == .null || status == "in_progress"
}

private func isRetainedProgramOwnedItem(_ entry: TResponseInputItem, _ index: Int, _ pruningIndexes: Set<Int>?) -> Bool {
    if let pruningIndexes { return !pruningIndexes.contains(index) }
    if isPendingHostedShellCall(entry) { return true }
    guard let type = entry.type else { return false }
    return programOwnedHostedItemTypes.contains(type) || toolCallToOutputType.values.contains(type)
}

private func matchedAnonymousToolSearchCallIndexes(_ payload: [TResponseInputItem]) -> Set<Int> {
    var matched: Set<Int> = []
    var pendingOutputs = 0
    for index in stride(from: payload.count - 1, through: 0, by: -1) {
        let entry = payload[index]
        if entry.type == "tool_search_output", entry.string("call_id") == nil {
            pendingOutputs += 1
            continue
        }
        if entry.type == "tool_search_call", entry.string("call_id") == nil, pendingOutputs > 0 {
            matched.insert(index)
            pendingOutputs -= 1
        }
    }
    return matched
}
