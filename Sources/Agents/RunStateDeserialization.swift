import Foundation

// Port of the restoration half of src/agents/run_state.py (`RunState.from_json`,
// `_build_run_state_from_json`, and the `_deserialize_*` helpers). See porting/decisions.md D13.

extension RunState {
    /// Deserializes a run state from a JSON string written by `toString()`.
    ///
    /// Only deserialize state from trusted storage or after the application verifies the integrity
    /// and ownership of the complete snapshot. Serialized state includes tool approvals and pending
    /// tool calls; this method does not authenticate that state. Do not pass client-supplied state
    /// directly to this method. Keep the snapshot on the server and apply authorized approval
    /// decisions to that server-owned state instead. Neither `contextOverride` nor `strictContext`
    /// verifies snapshot integrity.
    ///
    /// - Parameters:
    ///   - initialAgent: The initial agent, used to resolve agents referenced by the snapshot.
    ///   - stateString: The JSON string to deserialize.
    ///   - contextOverride: Optional context value or `RunContextWrapper` to use instead of the
    ///     serialized context.
    ///   - contextDeserializer: Optional function to rebuild non-mapping context values.
    ///   - strictContext: When `true`, require a deserializer or override for non-mapping contexts.
    /// - Throws: `UserError` if the string is invalid JSON or has an incompatible schema version.
    public static func fromString(
        _ initialAgent: Agent,
        _ stateString: String,
        contextOverride: (any Sendable)? = nil,
        contextDeserializer: ContextDeserializer? = nil,
        strictContext: Bool = false
    ) async throws -> RunState {
        let parsed: JSONValue
        do {
            parsed = try JSONValue.parse(stateString)
        } catch {
            // Swift: the parse error is dropped so snapshot contents never reach the message.
            throw UserError("Failed to parse run state JSON")
        }
        guard case .object(let stateJSON) = parsed else {
            throw UserError("Run state JSON must be an object")
        }
        return try await fromJSON(
            initialAgent,
            stateJSON,
            contextOverride: contextOverride,
            contextDeserializer: contextDeserializer,
            strictContext: strictContext
        )
    }

    /// Deserializes a run state from a JSON object written by `toJSON()`.
    ///
    /// The same trust requirements as `fromString` apply.
    ///
    /// - Throws: `UserError` if the object has an incompatible schema version or references
    ///   agents that are not reachable from `initialAgent`.
    public static func fromJSON(
        _ initialAgent: Agent,
        _ stateJSON: JSONObject,
        contextOverride: (any Sendable)? = nil,
        contextDeserializer: ContextDeserializer? = nil,
        strictContext: Bool = false
    ) async throws -> RunState {
        try validateRunStateSchemaVersion(stateJSON)
        if let sandbox = stateJSON["sandbox"], sandbox.objectValue == nil {
            throw ValueError("RunState sandbox resume state has an invalid envelope")
        }
        return try await buildRunStateFromJSON(
            initialAgent: initialAgent,
            stateJSON: stateJSON,
            contextOverride: contextOverride,
            contextDeserializer: contextDeserializer,
            strictContext: strictContext
        )
    }
}

@discardableResult
func validateRunStateSchemaVersion(_ stateJSON: JSONObject) throws -> String {
    guard let rawVersion = stateJSON["$schemaVersion"], rawVersion.isTruthy else {
        throw UserError("Run state is missing schema version")
    }
    guard case .string(let version) = rawVersion else {
        throw UserError("Run state schema version has an invalid type")
    }
    guard RunState.supportedSchemaVersions.contains(version) else {
        let supported = RunState.supportedSchemaVersions.sorted().joined(separator: ", ")
        throw UserError(
            "Run state schema version is not supported. Supported versions are: \(supported). "
                + "New snapshots are written as version \(RunState.currentSchemaVersion)."
        )
    }
    return version
}

/// Rebuilds a `RunState` from a JSON payload.
///
/// Context restoration follows this precedence order:
///
/// 1. `contextOverride` when supplied.
/// 2. `contextDeserializer` applied to serialized mapping data.
/// 3. Direct mapping restore for contexts that were serialized as plain mappings.
///
/// When the snapshot metadata indicates that the original context type could not round-trip
/// safely, this warns or throws (in `strictContext` mode) rather than silently claiming that the
/// rebuilt mapping is equivalent to the original object.
func buildRunStateFromJSON(
    initialAgent: Agent,
    stateJSON: JSONObject,
    contextOverride: (any Sendable)?,
    contextDeserializer: ContextDeserializer?,
    strictContext: Bool
) async throws -> RunState {
    let schemaVersion = try validateRunStateSchemaVersion(stateJSON)
    let schema = runStateSchemaVersionTuple(schemaVersion)
    if schema < runStateSchemaVersionTuple(RunState.programmaticToolCallingMinSchemaVersion),
       runStateUsesProgrammaticToolCalling(stateJSON) {
        throw UserError(
            "Run state contains Programmatic Tool Calling data but uses schema version \(schemaVersion). "
                + "Programmatic Tool Calling requires schema version \(RunState.programmaticToolCallingMinSchemaVersion) or later."
        )
    }

    let agents = RunStateAgentResolver(initialAgent: initialAgent)
    guard let currentAgent = try agents.resolve(stateJSON["current_agent"]) else {
        throw UserError("Run state agent not found in agent map")
    }

    let contextData = stateJSON["context"]?.objectValue ?? [:]
    let usage = deserializeUsage(contextData["usage"]?.objectValue ?? [:])
    let context = try restoreContext(
        contextData: contextData,
        contextOverride: contextOverride,
        contextDeserializer: contextDeserializer,
        strictContext: strictContext
    )
    context.usage = usage
    try context.toolStateBox.mutate { state in
        state.restoredUnboundApprovalCallIds = []
        state.allowLegacyApprovalBindingReconstruction = schema < (1, 15)
        state.rebuildApprovals(contextData["approvals"])
        if schema >= (1, 15) {
            try state.rebuildToolInvocations(contextData["tool_invocations"] ?? [:]) { UserError($0) }
        } else {
            state.toolInvocations = [:]
        }
        if schema >= runStateSchemaVersionTuple(RunState.hostedMCPApprovalsMinSchemaVersion) {
            state.rebuildHostedMCPApprovals(contextData["hosted_mcp_approvals"])
        }
        try restoreFunctionToolApprovals(contextData["function_tool_approvals"], schema: schema, state: &state, agents: agents)
        if schema >= (1, 15) {
            state.markRestoredUnboundApprovalCallIds()
        }
    }
    if contextOverride == nil, let toolInput = contextData["tool_input"], !toolInput.isNull, context.toolInput == nil {
        context.toolInput = toolInput
    }

    let originalInput: RunInput
    switch stateJSON["original_input"] {
    case .string(let text)?: originalInput = .text(text)
    case .array(let items)?: originalInput = .items(items.compactMap(\.objectValue))
    default: originalInput = .items([])
    }

    let maxTurns: Int?
    switch stateJSON["max_turns"] {
    case .null?, nil: maxTurns = nil
    case let value?: maxTurns = value.intValue
    }
    let state = RunState(
        context: context,
        originalInput: originalInput,
        startingAgent: initialAgent,
        maxTurns: maxTurns,
        conversationId: stateJSON.string("conversation_id"),
        previousResponseId: stateJSON.string("previous_response_id"),
        autoPreviousResponseId: stateJSON["auto_previous_response_id"]?.isTruthy ?? false
    )
    state.currentAgent = currentAgent
    state.schemaVersion = schemaVersion
    state.currentTurn = stateJSON["current_turn"]?.intValue ?? 0

    if let pendingInput = stateJSON["pending_input"] {
        guard let items = pendingInput.arrayValue else { throw UserError("Run state pending_input must be a list") }
        state.pendingInputItems = items.compactMap(\.objectValue)
    }
    state.modelResponses = deserializeModelResponses(stateJSON["model_responses"]?.arrayValue ?? [])

    let serializedGeneratedItems = stateJSON["generated_items"]?.arrayValue ?? []
    let (generatedItems, generatedSourceIndexes) = try deserializeItemsWithSourceIndexes(serializedGeneratedItems, agents: agents)
    state.generatedItems = generatedItems

    if let processedData = stateJSON["last_processed_response"]?.objectValue {
        state.lastProcessedResponse = try await deserializeProcessedResponse(
            processedData, currentAgent: currentAgent, context: context, agents: agents
        )
    }

    let session = try restoreSessionItems(
        state,
        stateJSON,
        agents: agents,
        serializedGeneratedItems: serializedGeneratedItems,
        generatedSourceIndexes: generatedSourceIndexes
    )
    try validateNestedHistoryOwnership(
        state,
        stateJSON["nested_history_owned_session_item_refs"],
        hasSessionItems: stateJSON.contains("session_items"),
        serializedSessionItemCount: session.serializedCount,
        sessionSourceIndexes: session.sourceIndexes
    )

    state.markGeneratedItemsMergedWithLastProcessed()

    try restoreGuardrailResults(state, stateJSON, agents: agents, currentAgent: currentAgent)
    try restoreCurrentStep(state, stateJSON["current_step"]?.objectValue, agents: agents)

    state.currentTurnPersistedItemCount = stateJSON["current_turn_persisted_item_count"]?.intValue ?? 0
    if let terminal = stateJSON["terminal_unrecoverable"], !terminal.isNull {
        // An older label never wrote this marker, so honoring one would let a snapshot claim a
        // resume boundary the schema it declares does not have.
        guard schema >= (1, 17), terminal == .bool(true) else {
            throw UserError("Run state terminal marker is invalid")
        }
    }
    state.reasoningItemIdPolicy = stateJSON.string("reasoning_item_id_policy").flatMap(ReasoningItemIdPolicy.init(rawValue:))
    state.generatedPromptCacheKey = stateJSON.string("generated_prompt_cache_key")
    state.setToolUseTrackerSnapshot(toolUseTrackerSnapshot(from: stateJSON["tool_use_tracker"]))
    state.traceState = TraceState.fromJSON(stateJSON["trace"]?.objectValue)
    try state.validateCompletedToolInvocations(reconstructLegacy: schema < (1, 15))
    return state
}

// MARK: - State sections

private func restoreSessionItems(
    _ state: RunState,
    _ stateJSON: JSONObject,
    agents: RunStateAgentResolver,
    serializedGeneratedItems: [JSONValue],
    generatedSourceIndexes: [Int]
) throws -> (serializedCount: Int, sourceIndexes: [Int]) {
    let serializedSessionItems: [JSONValue]
    let sessionSourceIndexes: [Int]
    if stateJSON.contains("session_items") {
        serializedSessionItems = stateJSON["session_items"]?.arrayValue ?? []
        let restored = try deserializeItemsWithSourceIndexes(serializedSessionItems, agents: agents)
        state.sessionItems = restored.items
        sessionSourceIndexes = restored.sourceIndexes
    } else {
        serializedSessionItems = []
        state.sessionItems = state.mergeGeneratedItemsWithProcessed()
        sessionSourceIndexes = Array(state.sessionItems.indices)
    }
    restoreGeneratedSessionOccurrences(
        state,
        indexes: stateJSON["generated_session_item_indexes"],
        serializedGeneratedItems: serializedGeneratedItems,
        generatedSourceIndexes: generatedSourceIndexes,
        serializedSessionItems: serializedSessionItems,
        sessionSourceIndexes: sessionSourceIndexes
    )
    return (serializedSessionItems.count, sessionSourceIndexes)
}

private func restoreGuardrailResults(_ state: RunState, _ stateJSON: JSONObject, agents: RunStateAgentResolver, currentAgent: Agent) throws {
    state.inputGuardrailResults = deserializeInputGuardrailResults(stateJSON["input_guardrail_results"]?.arrayValue ?? [])
    state.outputGuardrailResults = try deserializeOutputGuardrailResults(
        stateJSON["output_guardrail_results"]?.arrayValue ?? [], agents: agents, fallbackAgent: currentAgent
    )
    state.toolInputGuardrailResults = deserializeToolGuardrailEntries(
        stateJSON["tool_input_guardrail_results"]?.arrayValue ?? [], expectedType: "tool_input"
    ).map { name, output in
        ToolInputGuardrailResult(guardrail: ToolInputGuardrail(name: name) { _ in output }, output: output)
    }
    state.toolOutputGuardrailResults = deserializeToolGuardrailEntries(
        stateJSON["tool_output_guardrail_results"]?.arrayValue ?? [], expectedType: "tool_output"
    ).map { name, output in
        ToolOutputGuardrailResult(guardrail: ToolOutputGuardrail(name: name) { _ in output }, output: output)
    }
}

private func restoreCurrentStep(_ state: RunState, _ stepData: JSONObject?, agents: RunStateAgentResolver) throws {
    guard let stepData else { return }
    switch stepData.string("type") {
    case "next_step_run_again":
        state.currentStep = .runAgain
    case "next_step_interruption":
        let data = stepData["data"]?.objectValue ?? [:]
        let interruptionsData = data["interruptions"]?.arrayValue ?? stepData["interruptions"]?.arrayValue ?? []
        var interruptions: [ToolApprovalItem] = []
        for entry in interruptionsData {
            guard let entry = entry.objectValue else { continue }
            if let item = try deserializeToolApprovalItem(entry, agents: agents, fallbackAgent: nil) {
                interruptions.append(item)
            }
        }
        let responseAccepted = data["response_accepted"]?.isTruthy ?? false
        state.currentStep = .interruption(interruptions: interruptions, responseAccepted: responseAccepted)
        state.context.toolStateBox.mutate { context in
            for item in interruptions { context.markRestoredUnboundPendingApproval(item) }
        }
        state.llmEndHooksStarted = data["llm_end_hooks_started"]?.isTruthy ?? true
        if responseAccepted {
            state.clearGeneratedItemsLastProcessedMarker()
        }
    default:
        break
    }
}

// MARK: - Context

private func restoreContext(
    contextData: JSONObject,
    contextOverride: (any Sendable)?,
    contextDeserializer: ContextDeserializer?,
    strictContext: Bool
) throws -> RunContextWrapper {
    let serializedContext: JSONValue = contextData["context"] ?? [:]
    let contextMeta = contextData["context_meta"]?.objectValue

    // If the context was originally a custom type and no override or deserializer is supplied,
    // surface the risk of losing behavior or state during restore.
    if contextOverride == nil, contextDeserializer == nil, contextMetaRequiresDeserializer(contextMeta) {
        let message = contextMetaWarningMessage(contextMeta)
        if strictContext { throw UserError(message) }
        logger.warning(message)
    }

    if let wrapper = contextOverride as? RunContextWrapper {
        return try requirePlainWrapper(wrapper)
    }
    if let contextOverride {
        return RunContextWrapper(context: contextOverride)
    }
    if serializedContext.isNull {
        return RunContextWrapper()
    }
    if let contextDeserializer {
        guard let mapping = serializedContext.objectValue else {
            throw UserError("Serialized run state context must be a mapping to use context_deserializer.")
        }
        let rebuilt: any Sendable
        do {
            rebuilt = try contextDeserializer(mapping)
        } catch {
            throw UserError("Context deserializer failed while rebuilding RunState context.")
        }
        if let wrapper = rebuilt as? RunContextWrapper {
            return try requirePlainWrapper(wrapper)
        }
        return RunContextWrapper(context: rebuilt)
    }
    guard let mapping = serializedContext.objectValue else {
        throw UserError("Serialized run state context must be a mapping. Please provide one.")
    }
    return RunContextWrapper(context: mapping)
}

private func requirePlainWrapper(_ wrapper: RunContextWrapper) throws -> RunContextWrapper {
    guard type(of: wrapper) == RunContextWrapper.self else {
        throw UserError(
            "RunState restoration does not support RunContextWrapper subclasses; "
                + "provide the custom context value directly or wrap it in RunContextWrapper."
        )
    }
    return wrapper
}

/// Returns `true` when metadata indicates a non-mapping context needs help to restore.
private func contextMetaRequiresDeserializer(_ meta: JSONObject?) -> Bool {
    guard let meta else { return false }
    if meta["omitted"]?.isTruthy == true { return true }
    return meta["requires_deserializer"]?.isTruthy == true
}

/// Builds a warning message describing context deserialization requirements.
private func contextMetaWarningMessage(_ meta: JSONObject?) -> String {
    guard let meta else {
        return "RunState context was serialized from a custom type; provide context_deserializer "
            + "or context_override to restore it."
    }
    if meta["omitted"]?.isTruthy == true {
        return "RunState context was omitted during serialization; provide context_override to supply it."
    }
    return "RunState context requires explicit restoration; provide context_deserializer or context_override to restore it."
}

/// Validates and restores agent-owned function approvals.
private func restoreFunctionToolApprovals(_ value: JSONValue?, schema: (Int, Int), state: inout ToolApprovalState, agents: RunStateAgentResolver) throws {
    let entries: [JSONValue]
    switch value {
    case nil, .null?: entries = []
    case .array(let list)?: entries = list
    default: throw UserError("RunState function_tool_approvals must be a list.")
    }
    if !entries.isEmpty, schema < (1, 18) {
        throw UserError("Agent-owned function approvals require RunState schema 1.18.")
    }
    let ambiguous = entries.isEmpty ? [] : (agents.initialAgent.map(getAmbiguousAgentIds) ?? [])
    for entry in entries {
        guard let entry = entry.objectValue, let toolKey = entry.string("tool_key"), let decision = entry["decision"]?.objectValue else {
            throw UserError("Invalid agent-owned function approval record.")
        }
        guard let owner = try agents.resolve(entry["agent"]) else {
            throw UserError("Function approval owner is absent from the restored graph.")
        }
        // Traversal-derived suffixes cannot authorize indistinguishable owners.
        if ambiguous.contains(ObjectIdentifier(owner)) { continue }
        state.approvals[.function(FunctionToolApprovalKey(owner: owner, toolKey: toolKey))] = ApprovalRecord(json: decision)
    }
}

private func toolUseTrackerSnapshot(from value: JSONValue?) -> [String: [String]] {
    guard let object = value?.objectValue else { return [:] }
    var snapshot: [String: [String]] = [:]
    for (agentName, tools) in object {
        snapshot[agentName] = (tools.arrayValue ?? []).compactMap(\.stringValue)
    }
    return snapshot
}

// MARK: - Agents

/// Resolves serialized agent references against the restored agent graph.
struct RunStateAgentResolver {
    let initialAgent: Agent?
    let agentMap: [String: Agent]
    let identityMap: [String: Agent]?

    init(initialAgent: Agent) {
        self.initialAgent = initialAgent
        agentMap = buildAgentMap(initialAgent)
        identityMap = buildAgentIdentityMap(initialAgent)
    }

    /// Resolves names against an explicit map, without duplicate-name identities.
    init(agentMap: [String: Agent]) {
        initialAgent = nil
        self.agentMap = agentMap
        identityMap = nil
    }

    /// Resolves an agent from serialized data, falling back to `fallbackAgent`.
    func resolve(_ agentData: JSONValue?, fallbackAgent: Agent? = nil) throws -> Agent? {
        var agentName: String?
        var agentIdentity: String?
        switch agentData {
        case .object(let object)?:
            agentIdentity = object.string("identity")
            agentName = object.string("name")
        case .string(let name)?:
            agentName = name
        default:
            break
        }
        if let agentIdentity, let identityMap {
            if let resolved = identityMap[agentIdentity] { return resolved }
            throw UserError("Run state references an agent identity that is not present in the restored graph")
        }
        if let agentName, !agentName.isEmpty {
            return identityMap?[agentName] ?? agentMap[agentName] ?? fallbackAgent
        }
        return fallbackAgent
    }
}

// MARK: - Items

func deserializeModelResponses(_ responses: [JSONValue]) -> [ModelResponse] {
    responses.compactMap { value in
        guard let data = value.objectValue else { return nil }
        let output: [TResponseOutputItem] = (data["output"]?.arrayValue ?? []).compactMap { item in
            guard let object = item.objectValue else { return nil }
            return object.type == "message" ? normalizeMessageOutputItem(object) : object
        }
        return ModelResponse(
            output: output,
            usage: deserializeUsage(data["usage"]?.objectValue ?? [:]),
            responseId: data.string("response_id"),
            requestId: data.string("request_id")
        )
    }
}

/// Normalizes an assistant message whose content was stored as a plain string.
func normalizeMessageOutputItem(_ payload: JSONObject) -> JSONObject {
    guard payload.string("role") == "assistant", let content = payload.string("content") else { return payload }
    var normalized = payload
    normalized["content"] = [["type": "output_text", "text": .string(content)]]
    if normalized["status"] == nil { normalized["status"] = "completed" }
    return normalized
}

/// The reason a serialized item could not be restored.
private struct RunItemRestoreError: Error, CustomStringConvertible {
    var description: String
}

/// Deserializes a tool call output raw item, or returns `nil` when its shape is unknown.
func deserializeToolCallOutputRawItem(_ rawItem: JSONObject) throws -> JSONObject? {
    switch rawItem.type {
    case "function_call_output":
        guard rawItem.string("call_id") != nil, rawItem["output"] != nil else {
            throw RunItemRestoreError(description: "function_call_output requires call_id and output")
        }
        return rawItem
    case "computer_call_output":
        guard rawItem.string("call_id") != nil, rawItem["output"]?.objectValue != nil else {
            throw RunItemRestoreError(description: "computer_call_output requires call_id and output")
        }
        return rawItem
    case "local_shell_call_output":
        guard let callId = rawItem.string("call_id"), !callId.isEmpty, case .string? = rawItem["output"] else {
            throw RunItemRestoreError(description: "local_shell_call_output requires a call_id and string output")
        }
        return rawItem
    case "program_output", "shell_call_output", "apply_patch_call_output", "custom_tool_call_output":
        return rawItem
    default:
        return nil
    }
}

private func validateRequiredFields(_ rawItem: JSONObject, _ fields: [String], itemType: String) throws {
    for field in fields where rawItem[field] == nil {
        throw RunItemRestoreError(description: "\(itemType) is missing \(field)")
    }
}

func deserializeItemsWithSourceIndexes(_ itemsData: [JSONValue], agents: RunStateAgentResolver) throws -> (items: [RunItem], sourceIndexes: [Int]) {
    var items: [RunItem] = []
    var sourceIndexes: [Int] = []
    for (index, entry) in itemsData.enumerated() {
        if let item = try deserializeRunItem(entry, agents: agents) {
            items.append(item)
            sourceIndexes.append(index)
        }
    }
    return (items, sourceIndexes)
}

func deserializeItems(_ itemsData: [JSONValue], agents: RunStateAgentResolver) throws -> [RunItem] {
    try deserializeItemsWithSourceIndexes(itemsData, agents: agents).items
}

/// Resolves the agent that owns a serialized item, warning when it is missing.
private func resolveItemAgent(_ itemData: JSONObject, itemType: String, agents: RunStateAgentResolver) throws -> Agent? {
    var candidateName: String?
    let agentFields = itemType == "handoff_output_item" ? ["agent", "source_agent", "target_agent"] : ["agent"]
    for field in agentFields {
        let raw = itemData[field]
        if case .object(let object)? = raw { candidateName = object.string("name") ?? candidateName }
        if case .string(let name)? = raw { candidateName = name }
        if let agent = try agents.resolve(raw) { return agent }
    }
    if candidateName != nil {
        logger.warning("Agent not found, skipping item")
    } else {
        logger.warning("Item missing agent field, skipping")
    }
    return nil
}

/// Deserializes one run item, returning `nil` (with a warning) when it cannot be restored.
func deserializeRunItem(_ entry: JSONValue, agents: RunStateAgentResolver) throws -> RunItem? {
    guard let itemData = entry.objectValue, let itemType = itemData.string("type"), !itemType.isEmpty else {
        logger.warning("Item missing type field, skipping")
        return nil
    }

    guard let agent = try resolveItemAgent(itemData, itemType: itemType, agents: agents) else { return nil }

    let rawItem = itemData["raw_item"]?.objectValue ?? [:]
    do {
        switch itemType {
        case "input_item":
            if let inputId = itemData.string("input_id") {
                return InputItem(agent: agent, rawItem: rawItem, inputId: inputId)
            }
            return InputItem(agent: agent, rawItem: rawItem)
        case "message_output_item":
            let message = normalizeMessageOutputItem(rawItem)
            // Swift: mirrors pydantic validation of `ResponseOutputMessage`; only `status` may be missing.
            try validateRequiredFields(message, ["id", "content", "role"], itemType: itemType)
            return MessageOutputItem(agent: agent, rawItem: message)
        case "tool_search_call_item":
            return ToolSearchCallItem(agent: agent, rawItem: rawItem)
        case "tool_search_output_item":
            return ToolSearchOutputItem(agent: agent, rawItem: rawItem)
        case "tool_call_item":
            let item = ToolCallItem(
                agent: agent,
                rawItem: rawItem,
                description: itemData.string("description"),
                title: itemData.string("title"),
                toolOrigin: ToolOrigin.fromJSONDict(itemData["tool_origin"])
            )
            item.resolvedToolName = itemData.string("tool_name")
            return item
        case "tool_call_output_item":
            guard let outputRaw = try deserializeToolCallOutputRawItem(rawItem) else { return nil }
            let customData = itemData["custom_data"]?.objectValue
            return ToolCallOutputItem(
                agent: agent,
                rawItem: outputRaw,
                output: itemData["output"] ?? .string(""),
                toolOrigin: ToolOrigin.fromJSONDict(itemData["tool_origin"]),
                customData: (customData?.isEmpty ?? true) ? nil : customData
            )
        case "reasoning_item":
            try validateRequiredFields(rawItem, ["id", "summary"], itemType: itemType)
            return ReasoningItem(agent: agent, rawItem: rawItem)
        case "handoff_call_item":
            try validateRequiredFields(rawItem, ["call_id", "name", "arguments"], itemType: itemType)
            return HandoffCallItem(agent: agent, rawItem: rawItem)
        case "handoff_output_item":
            let source = try agents.resolve(itemData["source_agent"])
            let target = try agents.resolve(itemData["target_agent"])
            // If both agents cannot be resolved, skip this item gracefully.
            guard let source, let target else {
                logger.warning("Skipping handoff output item: could not resolve agents")
                return nil
            }
            return HandoffOutputItem(agent: agent, rawItem: rawItem, sourceAgent: source, targetAgent: target)
        case "compaction_item":
            return CompactionItem(agent: agent, rawItem: rawItem)
        case "mcp_list_tools_item":
            try validateRequiredFields(rawItem, ["id", "server_label", "tools"], itemType: itemType)
            return MCPListToolsItem(agent: agent, rawItem: rawItem)
        case "mcp_approval_request_item":
            try validateRequiredFields(rawItem, ["id", "arguments", "name", "server_label"], itemType: itemType)
            return MCPApprovalRequestItem(agent: agent, rawItem: rawItem)
        case "mcp_approval_response_item":
            try validateRequiredFields(rawItem, ["approval_request_id", "approve"], itemType: itemType)
            return MCPApprovalResponseItem(agent: agent, rawItem: rawItem)
        case "tool_approval_item":
            return try deserializeToolApprovalItem(itemData, agents: agents, fallbackAgent: agent)
        default:
            return nil
        }
    } catch let error as UserError {
        throw error
    } catch {
        logger.warning("Failed to deserialize item")
        return nil
    }
}

/// Deserializes a `ToolApprovalItem` from serialized data.
func deserializeToolApprovalItem(_ itemData: JSONObject, agents: RunStateAgentResolver, fallbackAgent: Agent?) throws -> ToolApprovalItem? {
    guard let agent = try agents.resolve(itemData["agent"], fallbackAgent: fallbackAgent) else { return nil }
    let rawItem = itemData["raw_item"]?.objectValue ?? itemData["rawItem"]?.objectValue ?? [:]
    return ToolApprovalItem(
        agent: agent,
        rawItem: rawItem,
        toolName: itemData.string("tool_name"),
        toolNamespace: itemData.string("tool_namespace"),
        toolOrigin: ToolOrigin.fromJSONDict(itemData["tool_origin"]),
        toolLookupKey: FunctionToolLookupKey(json: itemData["tool_lookup_key"]),
        allowBareNameAlias: itemData["allow_bare_name_alias"] == .bool(true)
    )
}

/// Restores generated items that alias session-history occurrences.
private func restoreGeneratedSessionOccurrences(
    _ state: RunState,
    indexes: JSONValue?,
    serializedGeneratedItems: [JSONValue],
    generatedSourceIndexes: [Int],
    serializedSessionItems: [JSONValue],
    sessionSourceIndexes: [Int]
) {
    guard let indexes, !indexes.isNull else { return }
    var restoredSessionIndexes: [Int: Int] = [:]
    for (restoredIndex, sourceIndex) in sessionSourceIndexes.enumerated() {
        restoredSessionIndexes[sourceIndex] = restoredIndex
    }
    guard let list = indexes.arrayValue, list.count == serializedGeneratedItems.count else {
        logger.warning("Ignoring invalid generated_session_item_indexes in serialized RunState")
        return
    }
    var used: Set<Int> = []
    var parsed: [Int?] = []
    for value in list {
        if value.isNull {
            parsed.append(nil)
            continue
        }
        guard case .int(let index) = value, index >= 0, index < serializedSessionItems.count, used.insert(index).inserted else {
            logger.warning("Ignoring invalid generated_session_item_indexes in serialized RunState")
            return
        }
        parsed.append(index)
    }
    for (restoredGeneratedIndex, generatedSourceIndex) in generatedSourceIndexes.enumerated() {
        guard let sessionSourceIndex = parsed[generatedSourceIndex],
              let restoredSessionIndex = restoredSessionIndexes[sessionSourceIndex] else { continue }
        guard serializedGeneratedItems[generatedSourceIndex] == serializedSessionItems[sessionSourceIndex] else {
            logger.warning("Ignoring mismatched generated/session occurrence in serialized RunState")
            continue
        }
        state.generatedItems[restoredGeneratedIndex] = state.sessionItems[restoredSessionIndex]
    }
}

/// Validates the nested history ownership sidecar.
///
/// Swift: nested handoff history ownership is not tracked on `RunState` yet, so valid entries are
/// checked exactly like upstream and then dropped.
private func validateNestedHistoryOwnership(
    _ state: RunState,
    _ value: JSONValue?,
    hasSessionItems: Bool,
    serializedSessionItemCount: Int,
    sessionSourceIndexes: [Int]
) throws {
    let refs: [JSONValue]
    switch value {
    case nil: refs = []
    case .array(let list)?: refs = list
    default: throw UserError("Run state nested_history_owned_session_item_refs must be a list of objects")
    }
    var restoredSessionIndexes: [Int: Int] = [:]
    for (restoredIndex, sourceIndex) in sessionSourceIndexes.enumerated() {
        restoredSessionIndexes[sourceIndex] = restoredIndex
    }
    for ref in refs {
        guard let ref = ref.objectValue,
              case .int(let sessionSourceIndex)? = ref["index"], sessionSourceIndex >= 0,
              let digest = ref.string("digest"), digest.count == 64,
              case .int(let inputIndex)? = ref["input_index"], inputIndex >= 0 else {
            throw UserError(
                "Run state nested_history_owned_session_item_refs entries must contain a "
                    + "non-negative integer index and input_index, and 64-character digest"
            )
        }
        if hasSessionItems, sessionSourceIndex >= serializedSessionItemCount {
            throw UserError("Run state nested history ownership references a missing session item")
        }
        guard let sessionIndex = restoredSessionIndexes[sessionSourceIndex] else {
            logger.warning("Ignoring nested history ownership for skipped session item at index \(sessionSourceIndex)")
            continue
        }
        guard case .items(let originalItems) = state.originalInput, inputIndex < originalItems.count else {
            throw UserError("Run state nested history ownership references a missing input item")
        }
        guard let runInputItem = try? runItemToInputItem(state.sessionItems[sessionIndex]),
              digestInputItem(runInputItem) == digest else {
            throw UserError("Run state nested history ownership session digest does not match")
        }
        guard digestInputItem(originalItems[inputIndex]) == digest else {
            throw UserError("Run state nested history ownership input digest does not match")
        }
    }
}

// MARK: - Processed response

/// Name-indexed tools of the current agent, used to rebind restored tool actions.
private struct RestoredToolMaps {
    var functions: [FunctionToolLookupKey: FunctionTool] = [:]
    var computers: [String: ComputerTool] = [:]
    var custom: [String: CustomTool] = [:]
    var localShell: [String: LocalShellTool] = [:]
    var shell: [String: ShellTool] = [:]
    var applyPatch: [String: ApplyPatchTool] = [:]
    var hostedMCP: [String: HostedMCPTool] = [:]
    var handoffs: [String: Handoff] = [:]

    init(allTools: [any Tool], currentAgent: Agent) throws {
        functions = try buildFunctionToolLookupMap(allTools)
        for tool in allTools {
            switch tool {
            case let computer as ComputerTool:
                computers[computer.name] = computer
            case let customTool as CustomTool:
                custom[customTool.name] = customTool
            case let local as LocalShellTool:
                localShell[local.name] = local
            case let shellTool as ShellTool:
                shell[shellTool.name] = shellTool
            case let patch as ApplyPatchTool:
                applyPatch[patch.name] = patch
            case let mcp as HostedMCPTool:
                if let label = mcp.toolConfig.string("server_label"), !label.isEmpty { hostedMCP[label] = mcp }
            default:
                break
            }
        }
        // Persisted runs may contain either the preview name or the GA alias; mirror both.
        if let computer = computers["computer_use_preview"], computers["computer"] == nil { computers["computer"] = computer }
        if let computer = computers["computer"], computers["computer_use_preview"] == nil { computers["computer_use_preview"] = computer }
        for target in currentAgent.handoffs {
            let handoff = target.resolved
            if !handoff.toolName.isEmpty { handoffs[handoff.toolName] = handoff }
        }
    }
}

/// Deserializes a `ProcessedResponse`, resolving tools against the current agent.
func deserializeProcessedResponse(
    _ data: JSONObject,
    currentAgent: Agent,
    context: RunContextWrapper,
    agents: RunStateAgentResolver
) async throws -> ProcessedResponse {
    let newItems = try deserializeItems(data["new_items"]?.arrayValue ?? [], agents: agents)
    let tools = try RestoredToolMaps(allTools: try await currentAgent.getAllTools(context), currentAgent: currentAgent)

    func entries(_ key: String) -> [JSONObject] {
        (data[key]?.arrayValue ?? []).compactMap(\.objectValue)
    }
    func toolName(_ entry: JSONObject, _ wrapperKey: String) -> String? {
        entry[wrapperKey]?.objectValue?.string("name")
    }
    func toolCall(_ entry: JSONObject) -> JSONObject {
        entry["tool_call"]?.objectValue ?? [:]
    }

    var functions: [ToolRunFunction] = []
    for entry in entries("functions") {
        guard let lookupKey = resolveFunctionLookupKey(entry), let tool = tools.functions[lookupKey],
              let call = validatedFunctionToolCall(toolCall(entry)) else { continue }
        functions.append(ToolRunFunction(toolCall: call, functionTool: tool))
    }

    var handoffs: [ToolRunHandoff] = []
    for entry in entries("handoffs") {
        guard let name = entry["handoff"]?.objectValue?.string("tool_name"), let handoff = tools.handoffs[name],
              let call = validatedFunctionToolCall(toolCall(entry)) else { continue }
        handoffs.append(ToolRunHandoff(handoff: handoff, toolCall: call))
    }

    var computerActions: [ToolRunComputerAction] = []
    for entry in entries("computer_actions") {
        guard let name = toolName(entry, "computer"), let tool = tools.computers[name],
              let call = ResponseComputerToolCall(json: toolCall(entry)), call.json.string("call_id") != nil else { continue }
        computerActions.append(ToolRunComputerAction(toolCall: call, computerTool: tool))
    }

    var customToolCalls: [ToolRunCustom] = []
    for entry in entries("custom_tool_actions") {
        guard let name = toolName(entry, "custom_tool"), let tool = tools.custom[name],
              let call = ResponseCustomToolCall(json: toolCall(entry)), call.json.string("call_id") != nil else { continue }
        customToolCalls.append(ToolRunCustom(toolCall: call, customTool: tool))
    }

    var localShellCalls: [ToolRunLocalShellCall] = []
    for entry in entries("local_shell_actions") {
        guard let name = toolName(entry, "local_shell"), let tool = tools.localShell[name],
              let call = LocalShellCall(json: toolCall(entry)) else { continue }
        localShellCalls.append(ToolRunLocalShellCall(toolCall: call, localShellTool: tool))
    }

    var shellCalls: [ToolRunShellCall] = []
    for entry in entries("shell_actions") {
        guard let name = toolName(entry, "shell"), let tool = tools.shell[name] else { continue }
        shellCalls.append(ToolRunShellCall(toolCall: toolCall(entry), shellTool: tool))
    }

    var applyPatchCalls: [ToolRunApplyPatchCall] = []
    for entry in entries("apply_patch_actions") {
        guard let name = toolName(entry, "apply_patch"), let tool = tools.applyPatch[name] else { continue }
        applyPatchCalls.append(ToolRunApplyPatchCall(toolCall: toolCall(entry), applyPatchTool: tool))
    }

    var mcpApprovalRequests: [ToolRunMCPApprovalRequest] = []
    for entry in entries("mcp_approval_requests") {
        let rawItem = entry["request_item"]?.objectValue?["raw_item"]?.objectValue ?? [:]
        guard let request = McpApprovalRequest(json: rawItem) else {
            throw ValueError("Invalid MCP approval request in serialized RunState")
        }
        guard let mcpToolData = entry["mcp_tool"]?.objectValue, !mcpToolData.isEmpty else { continue }
        if let tool = rawItem.string("server_label").flatMap({ tools.hostedMCP[$0] }) {
            mcpApprovalRequests.append(ToolRunMCPApprovalRequest(requestItem: request, mcpTool: tool))
        }
    }

    var interruptions: [ToolApprovalItem] = []
    for entry in entries("interruptions") {
        if let item = try deserializeToolApprovalItem(entry, agents: agents, fallbackAgent: currentAgent) {
            interruptions.append(item)
        }
    }

    return ProcessedResponse(
        newItems: newItems,
        handoffs: handoffs,
        functions: functions,
        computerActions: computerActions,
        localShellCalls: localShellCalls,
        shellCalls: shellCalls,
        applyPatchCalls: applyPatchCalls,
        toolsUsed: (data["tools_used"]?.arrayValue ?? []).compactMap(\.stringValue),
        mcpApprovalRequests: mcpApprovalRequests,
        interruptions: interruptions,
        customToolCalls: customToolCalls,
        mcpToolBindings: (data["mcp_tool_bindings"]?.objectValue ?? [:]).pairs.reduce(into: [:]) { bindings, pair in
            bindings[pair.key] = MCPToolBinding(json: pair.value)
        }
    )
}

/// Resolves the lookup key for a serialized function action.
private func resolveFunctionLookupKey(_ entry: JSONObject) -> FunctionToolLookupKey? {
    let toolData = entry["tool"]?.objectValue
    if let lookupKey = FunctionToolLookupKey(json: toolData?["lookupKey"]) {
        return lookupKey
    }
    if let call = entry["tool_call"]?.objectValue,
       let lookupKey = getFunctionToolLookupKey(call.string("name"), call.string("namespace")) {
        return lookupKey
    }
    guard let toolData else { return nil }
    return getFunctionToolLookupKey(toolData.string("name"), toolData.string("namespace"))
}

/// Returns the function call when it has every field upstream's `ResponseFunctionToolCall`
/// requires.
func validatedFunctionToolCall(_ json: JSONObject) -> ResponseFunctionToolCall? {
    guard json.type == "function_call", json.string("call_id") != nil, json.string("name") != nil,
          json.string("arguments") != nil else { return nil }
    return ResponseFunctionToolCall(json: json)
}

// MARK: - Guardrails

private func parseGuardrailEntry(_ entry: JSONValue, expectedType: String) -> (name: String, output: GuardrailFunctionOutput, entry: JSONObject)? {
    let entryDict = entry.objectValue ?? [:]
    let info = entryDict["guardrail"]?.objectValue ?? [:]
    if let type = info.string("type"), !type.isEmpty, type != expectedType { return nil }
    let name = info.string("name").flatMap { $0.isEmpty ? nil : $0 } ?? "deserialized_\(expectedType)_guardrail"
    let outputData = entryDict["output"]?.objectValue ?? [:]
    let outputInfo = outputData["outputInfo"].flatMap { $0.isNull ? nil : $0 }
    let output = GuardrailFunctionOutput(outputInfo: outputInfo, tripwireTriggered: outputData["tripwireTriggered"]?.isTruthy ?? false)
    return (name, output, entryDict)
}

/// Rehydrates input guardrail results from serialized data.
func deserializeInputGuardrailResults(_ results: [JSONValue]) -> [InputGuardrailResult] {
    results.compactMap { entry in
        guard let parsed = parseGuardrailEntry(entry, expectedType: "input") else { return nil }
        let output = parsed.output
        return InputGuardrailResult(guardrail: InputGuardrail(name: parsed.name) { _, _, _ in output }, output: output)
    }
}

/// Rehydrates output guardrail results from serialized data.
func deserializeOutputGuardrailResults(_ results: [JSONValue], agents: RunStateAgentResolver, fallbackAgent: Agent) throws -> [OutputGuardrailResult] {
    try results.compactMap { entry in
        guard let parsed = parseGuardrailEntry(entry, expectedType: "output") else { return nil }
        let output = parsed.output
        let agent = try agents.resolve(parsed.entry["agent"], fallbackAgent: fallbackAgent) ?? fallbackAgent
        return OutputGuardrailResult(
            guardrail: OutputGuardrail(name: parsed.name) { _, _, _ in output },
            agentOutput: parsed.entry["agentOutput"] ?? .null,
            agent: agent,
            output: output
        )
    }
}

/// Parses tool guardrail result entries.
func deserializeToolGuardrailEntries(_ results: [JSONValue], expectedType: String) -> [(String, ToolGuardrailFunctionOutput)] {
    results.compactMap { entry in
        let entryDict = entry.objectValue ?? [:]
        let info = entryDict["guardrail"]?.objectValue ?? [:]
        if let type = info.string("type"), !type.isEmpty, type != expectedType { return nil }
        let name = info.string("name").flatMap { $0.isEmpty ? nil : $0 } ?? "deserialized_\(expectedType)_guardrail"
        let outputData = entryDict["output"]?.objectValue ?? [:]
        let behavior = outputData["behavior"]?.objectValue.flatMap(ToolGuardrailBehavior.init(json:)) ?? .allow
        let outputInfo = outputData["outputInfo"].flatMap { $0.isNull ? nil : $0 }
        return (name, ToolGuardrailFunctionOutput(outputInfo: outputInfo, behavior: behavior))
    }
}

// MARK: - Programmatic tool calling

/// Returns whether a raw run item uses the programmatic calling protocol.
private func rawItemUsesProgrammaticToolCalling(_ value: JSONValue?) -> Bool {
    guard let object = value?.objectValue else { return false }
    if ["program", "program_output"].contains(object.type ?? "") { return true }
    guard let caller = object["caller"]?.objectValue else { return false }
    return caller.string("type") == "program" && caller.string("caller_id") != nil
}

/// Collects raw items from durable RunState locations.
private func runStateRawItems(_ stateJSON: JSONObject) -> [JSONValue] {
    var rawItems: [JSONValue] = []
    rawItems += stateJSON["original_input"]?.arrayValue ?? []
    rawItems += stateJSON["pending_input"]?.arrayValue ?? []
    for key in ["model_responses", "last_model_response"] {
        var responses = stateJSON[key]?.arrayValue ?? []
        if let single = stateJSON[key]?.objectValue { responses = [.object(single)] }
        for response in responses {
            rawItems += response.objectValue?["output"]?.arrayValue ?? []
        }
    }
    func rawItemsOf(_ items: [JSONValue]?) -> [JSONValue] {
        (items ?? []).compactMap { $0.objectValue?["raw_item"] }
    }
    for key in ["generated_items", "session_items"] {
        rawItems += rawItemsOf(stateJSON[key]?.arrayValue)
    }
    if let processed = stateJSON["last_processed_response"]?.objectValue {
        for key in ["new_items", "interruptions"] {
            rawItems += rawItemsOf(processed[key]?.arrayValue)
        }
        for key in ["functions", "computer_actions", "custom_tool_actions", "local_shell_actions", "shell_actions", "apply_patch_actions", "handoffs"] {
            rawItems += (processed[key]?.arrayValue ?? []).compactMap { $0.objectValue?["tool_call"] }
        }
    }
    if let stepData = stateJSON["current_step"]?.objectValue?["data"]?.objectValue {
        rawItems += rawItemsOf(stepData["interruptions"]?.arrayValue)
    }
    return rawItems
}

/// Inspects only durable run-item locations for programmatic calling data.
private func runStateUsesProgrammaticToolCalling(_ stateJSON: JSONObject) -> Bool {
    runStateRawItems(stateJSON).contains(where: rawItemUsesProgrammaticToolCalling)
}
