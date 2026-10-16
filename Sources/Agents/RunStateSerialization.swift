import Foundation

// Port of the serialization half of src/agents/run_state.py (`RunState.to_json` and the
// `_serialize_*` helpers). See porting/decisions.md D13.

/// Serializes a non-mapping run context into a JSON object (upstream `ContextSerializer`).
public typealias ContextSerializer = @Sendable (any Sendable) throws -> JSONObject

/// Rebuilds a run context from serialized data (upstream `ContextDeserializer`).
///
/// Return a `RunContextWrapper` to restore the wrapper itself, or any other value to use as the
/// wrapper's context.
public typealias ContextDeserializer = @Sendable (JSONObject) throws -> any Sendable

// MARK: - Schema policy

// RunState schema policy.
// 1. Keep schema versions shipped in releases readable.
// 2. Unreleased schema versions may be renumbered or squashed before release when their
//    intermediate snapshots are intentionally unsupported.
// 3. toJSON() always emits currentSchemaVersion.
// 4. Forward compatibility is intentionally fail-fast (older SDKs reject newer or unsupported
//    versions).
extension RunState {
    /// The schema version written by `toJSON()`.
    public static let currentSchemaVersion = "1.18"
    static let programmaticToolCallingMinSchemaVersion = "1.13"
    static let hostedMCPApprovalsMinSchemaVersion = "1.14"
    static let currentResponseOwnershipMinSchemaVersion = "1.17"

    /// One-line summaries of every readable schema version, in chronological order.
    ///
    /// Every schema bump must add a summary here.
    public static let schemaVersionSummaries: [(version: String, summary: String)] = [
        ("1.0", "Initial RunState snapshot format for HITL pause/resume flows."),
        ("1.1", "Same payload as 1.0, but introduces explicit backward-read support policy."),
        ("1.2", "Persists reasoning_item_id_policy for resumed and streamed follow-up turns."),
        ("1.3", "Updates resumed trace semantics to reattach traces without duplicate starts."),
        ("1.4", "Stores request_id alongside each serialized model response."),
        ("1.5", "Renumbered unreleased baseline for tool-search snapshots and richer tool metadata."),
        ("1.6", "Persists explicit approval rejection messages across resume flows."),
        ("1.7", "Persists duplicate-name agent identities across agent-owned state and sandbox resume state."),
        ("1.8", "Persists SDK-generated prompt cache keys across resume flows."),
        ("1.9", "Persists pending custom tool calls and tool origin metadata across resume flows."),
        ("1.10", "Allows serialized RunState snapshots to disable max_turns with null."),
        ("1.11", "Persists SDK-only custom data on tool output items across resume flows."),
        ("1.12", "Persists input cache-write token usage across resume flows."),
        ("1.13", "Persists programmatic tool calling and nested handoff history ownership across resume flows."),
        ("1.14", "Scopes hosted MCP approvals and restored requests by server label."),
        (
            "1.15",
            "Persists canonical tool invocation identity plus sanitized mount authority and trusted rebind metadata, "
                + "durable pending input, and resumable next-model-call state."
        ),
        (
            "1.16",
            "Persists Docker network-isolation state and lets an exact call approval decision override a sticky "
                + "decision for the same tool."
        ),
        (
            "1.17",
            "Persists Docker container labels and current-response generated-item ownership across resume flows, "
                + "including pending resumed Session writes and terminal-unrecoverable runs."
        ),
        (
            "1.18",
            "Binds restored local MCP calls to their configured server and original tool name, preserves independent "
                + "apply_patch approval scopes, and binds function-tool approval decisions to their owning agent, and "
                + "retains compaction metadata for pending Session writes, including acknowledgement and model-exchange "
                + "evidence for compaction retry, and records named Vercel sandboxes alongside exact execution IDs."
        ),
    ]

    /// The schema versions `fromJSON` accepts.
    public static let supportedSchemaVersions: Set<String> = Set(schemaVersionSummaries.map(\.version))
}

/// Parses a `major.minor` schema version for ordered comparisons.
func runStateSchemaVersionTuple(_ version: String) -> (Int, Int) {
    let parts = version.split(separator: ".", maxSplits: 1).map { Int($0) ?? 0 }
    return (parts.first ?? 0, parts.count > 1 ? parts[1] : 0)
}

// MARK: - Serialization

extension RunState {
    /// Serializes the run state to a JSON-compatible object that can be used to resume the run.
    ///
    /// - Parameters:
    ///   - contextSerializer: Optional function to serialize non-mapping context values.
    ///   - strictContext: When `true`, require mapping contexts or a `contextSerializer`.
    ///   - includeTracingAPIKey: When `true`, include the tracing API key in the trace payload.
    /// - Throws: `UserError` if the context cannot be serialized as requested.
    public func toJSON(
        contextSerializer: ContextSerializer? = nil,
        strictContext: Bool = false,
        includeTracingAPIKey: Bool = false
    ) throws -> JSONObject {
        guard let currentAgent else { throw UserError("Cannot serialize RunState: No current agent") }

        let identityKeys = buildAgentIdentityKeysById(startingAgent)
        let modelResponses = serializeModelResponses()
        let (contextPayload, contextMeta) = try serializeContextPayload(contextSerializer: contextSerializer, strictContext: strictContext)

        var contextEntry: JSONObject = [
            "usage": .object(serializeUsage(context.usage)),
            "approvals": .object(serializeApprovals()),
            // Swift: the canonical tool invocation ledger is not ported yet, so it is always empty.
            "tool_invocations": [:],
            "context": contextPayload,
            // Preserve metadata so deserialization can warn when context types were erased.
            "context_meta": .object(contextMeta),
        ]
        if let toolInput = context.toolInput, let serialized = serializeToolInput(toolInput) {
            contextEntry["tool_input"] = serialized
        }
        let hostedApprovals = serializeHostedMCPApprovals()
        if !hostedApprovals.isEmpty {
            contextEntry["hosted_mcp_approvals"] = .array(hostedApprovals)
        }

        let generatedItems = mergeGeneratedItemsWithProcessed()
        var result: JSONObject = [
            "$schemaVersion": .string(Self.currentSchemaVersion),
            "current_turn": .int(currentTurn),
            "current_agent": .object(serializeAgentReference(currentAgent, identityKeys)),
            "original_input": serializeOriginalInput(),
            "pending_input": serializeInput(.items(pendingInputItems)),
            "model_responses": .array(modelResponses.map(JSONValue.object)),
            "context": .object(contextEntry),
            "tool_use_tracker": .object(JSONObject(toolUseTrackerSnapshot.mapValues { .array($0.map(JSONValue.string)) })),
            "max_turns": maxTurns.map(JSONValue.int) ?? .null,
            "no_active_agent_run": true,
            "input_guardrail_results": .array(serializeInputGuardrailResults(inputGuardrailResults)),
            "output_guardrail_results": .array(serializeOutputGuardrailResults(outputGuardrailResults, identityKeys)),
            "tool_input_guardrail_results": .array(toolInputGuardrailResults.map {
                serializeToolGuardrailResult(name: $0.guardrail.getName(), output: $0.output, typeLabel: "tool_input")
            }),
            "tool_output_guardrail_results": .array(toolOutputGuardrailResults.map {
                serializeToolGuardrailResult(name: $0.guardrail.getName(), output: $0.output, typeLabel: "tool_output")
            }),
            "conversation_id": JSONValue(optional: conversationId),
            "previous_response_id": JSONValue(optional: previousResponseId),
            "auto_previous_response_id": .bool(autoPreviousResponseId),
            "generated_prompt_cache_key": JSONValue(optional: generatedPromptCacheKey),
            "reasoning_item_id_policy": JSONValue(optional: reasoningItemIdPolicy?.rawValue),
            // Swift: nested handoff history ownership is not tracked on RunState yet.
            "nested_history_owned_session_item_refs": [],
            "generated_session_item_indexes": .array(generatedSessionItemIndexes(generatedItems).map { $0.map(JSONValue.int) ?? .null }),
        ]
        if let ownership = currentResponseGeneratedItemOwnership(generatedItems) {
            result["current_response_generated_item_ownership"] = .object(ownership)
        }
        result["generated_items"] = .array(generatedItems.map { .object(serializeRunItem($0, identityKeys)) })
        result["session_items"] = .array(sessionItems.map { .object(serializeRunItem($0, identityKeys)) })
        result["current_step"] = serializeCurrentStep(identityKeys)
        result["last_model_response"] = modelResponses.last.map(JSONValue.object) ?? .null
        result["last_processed_response"] = lastProcessedResponse.map { .object(serializeProcessedResponse($0, identityKeys)) } ?? .null
        result["current_turn_persisted_item_count"] = .int(currentTurnPersistedItemCount)
        result["trace"] = traceState?.toJSON(includeTracingAPIKey: includeTracingAPIKey).map(JSONValue.object) ?? .null
        return result
    }

    /// Serializes the run state to a JSON string.
    ///
    /// - Parameter includeTracingAPIKey: When `true`, include the tracing API key in the trace
    ///   payload.
    public func toString(
        contextSerializer: ContextSerializer? = nil,
        strictContext: Bool = false,
        includeTracingAPIKey: Bool = false
    ) throws -> String {
        let json = try toJSON(contextSerializer: contextSerializer, strictContext: strictContext, includeTracingAPIKey: includeTracingAPIKey)
        return JSONValue.object(json).jsonString(prettyPrinted: true)
    }

    // MARK: Context

    /// Validates and serializes the stored run context.
    ///
    /// The returned metadata records how the context was serialized so restore-time code can
    /// decide whether a deserializer or override is required.
    ///
    /// Swift: upstream special-cases pydantic models (`model_dump`) and dataclasses (`asdict`).
    /// Their Swift counterpart is `Encodable`, recorded as `serialized_via: "encodable"`.
    func serializeContextPayload(contextSerializer: ContextSerializer?, strictContext: Bool) throws -> (JSONValue, JSONObject) {
        let raw = context.context
        if isNoneContext(raw) {
            return (.null, buildContextMeta(raw, serializedVia: "none", requiresDeserializer: false, omitted: false))
        }
        if let mapping = mappingContext(raw) {
            return (.object(mapping), buildContextMeta(raw, serializedVia: "mapping", requiresDeserializer: false, omitted: false))
        }
        if strictContext, contextSerializer == nil {
            // Avoid silently dropping non-mapping context data when strict mode is requested.
            throw UserError(
                "RunState serialization requires context to be a mapping when strict_context is True. "
                    + "Provide context_serializer to serialize custom contexts."
            )
        }
        if let contextSerializer {
            let serialized: JSONObject
            do {
                serialized = try contextSerializer(raw)
            } catch {
                throw UserError("Context serializer failed while serializing RunState context.")
            }
            return (.object(serialized), buildContextMeta(raw, serializedVia: "context_serializer", requiresDeserializer: true, omitted: false))
        }
        if let encodable = raw as? any Encodable, case .object(let serialized)? = try? JSONValue(encoding: encodable) {
            // The data persists, but the original type is lost unless the caller rebuilds it.
            logger.warning(
                "RunState context was serialized from an Encodable value. "
                    + "Provide context_deserializer or context_override to restore the original type."
            )
            return (.object(serialized), buildContextMeta(raw, serializedVia: "encodable", requiresDeserializer: true, omitted: false))
        }
        // Fall back to an empty object so the run state remains serializable, but explicitly warn
        // because the original context will be unavailable on restore.
        logger.warning(
            "RunState context of type \(type(of: raw)) is not serializable; storing empty context. "
                + "Provide context_serializer to preserve it."
        )
        return ([:], buildContextMeta(raw, serializedVia: "omitted", requiresDeserializer: true, omitted: true))
    }

    func serializeToolInput(_ toolInput: any Sendable) -> JSONValue? {
        if isNoneContext(toolInput) { return nil }
        return serializeOutputValue(toolInput)
    }

    // MARK: Approvals

    /// Serializes legacy and non-hosted approval records.
    func serializeApprovals() -> JSONObject {
        var serialized = JSONObject()
        for (key, record) in context.approvals.sorted(by: { $0.key < $1.key }) where hostedMCPApprovalIdentity(key) == nil {
            serialized[key] = .object(record.json)
        }
        return serialized
    }

    /// Serializes hosted MCP approvals with explicit typed identities.
    func serializeHostedMCPApprovals() -> [JSONValue] {
        context.approvals.sorted(by: { $0.key < $1.key }).compactMap { key, record in
            guard let identity = hostedMCPApprovalIdentity(key) else { return nil }
            return ["identity": .object(identity), "decision": .object(record.json)]
        }
    }

    // MARK: Inputs and responses

    func serializeModelResponses() -> [JSONObject] {
        modelResponses.map { response in
            [
                "usage": .object(serializeUsage(response.usage)),
                "output": .array(response.output.map(JSONValue.object)),
                "response_id": JSONValue(optional: response.responseId),
                "request_id": JSONValue(optional: response.requestId),
            ]
        }
    }

    /// Normalizes input into the shape expected by the Responses API.
    func serializeInput(_ input: RunInput) -> JSONValue {
        switch input {
        case .text(let text):
            return .string(text)
        case .items(let items):
            return .array(items.map { item in
                var normalized = item
                if normalized.string("role") == "assistant" {
                    if let content = normalized.string("content") {
                        normalized["content"] = [["type": "output_text", "text": .string(content)]]
                    }
                    if normalized["status"] == nil {
                        normalized["status"] = "completed"
                    }
                }
                return .object(normalized)
            })
        }
    }

    func serializeOriginalInput() -> JSONValue {
        serializeInput(originalInput)
    }

    // MARK: Items

    func serializeRunItem(_ item: RunItem, _ identityKeys: [ObjectIdentifier: String]?) -> JSONObject {
        var result: JSONObject = [
            "type": .string(item.type),
            "raw_item": .object(item.rawItem),
            "agent": .object(serializeAgentReference(item.agent, identityKeys)),
        ]
        if let input = item as? InputItem {
            result["input_id"] = .string(input.inputId)
        }
        if let output = item as? ToolCallOutputItem {
            result["output"] = serializeOutputValue(output.output)
        }
        if let handoff = item as? HandoffOutputItem {
            result["source_agent"] = .object(serializeAgentReference(handoff.sourceAgent, identityKeys))
            result["target_agent"] = .object(serializeAgentReference(handoff.targetAgent, identityKeys))
        }
        switch item {
        case let call as ToolCallItem:
            if let toolName = call.toolName { result["tool_name"] = .string(toolName) }
            if let description = call.description { result["description"] = .string(description) }
            if let title = call.title { result["title"] = .string(title) }
            if let origin = call.toolOrigin { result["tool_origin"] = .object(origin.toJSONDict()) }
        case let output as ToolCallOutputItem:
            if let origin = output.toolOrigin { result["tool_origin"] = .object(origin.toJSONDict()) }
            if let customData = output.customData, !customData.isEmpty { result["custom_data"] = .object(customData) }
        case let approval as ToolApprovalItem:
            if let toolName = approval.toolName { result["tool_name"] = .string(toolName) }
            if let namespace = approval.toolNamespace { result["tool_namespace"] = .string(namespace) }
            if let lookupKey = approval.toolLookupKey { result["tool_lookup_key"] = .object(lookupKey.json) }
            if approval.allowBareNameAlias { result["allow_bare_name_alias"] = true }
            if let origin = approval.toolOrigin { result["tool_origin"] = .object(origin.toJSONDict()) }
        default:
            break
        }
        return result
    }

    /// Returns a marker for the processed response already reflected in `generatedItems`.
    func currentGeneratedItemsMergeMarker() -> String? {
        guard let processed = lastProcessedResponse, !processed.newItems.isEmpty else { return nil }
        let identityKeys = buildAgentIdentityKeysById(startingAgent)
        let marker: JSONObject = [
            "current_turn": .int(currentTurn),
            "last_response_id": JSONValue(optional: modelResponses.last?.responseId),
            "new_items": .array(processed.newItems.map { .object(serializeRunItem($0, identityKeys)) }),
        ]
        return JSONValue.object(marker).jsonString(sortedKeys: true)
    }

    /// Remembers that `generatedItems` already include the current processed response.
    func markGeneratedItemsMergedWithLastProcessed() {
        generatedItemsLastProcessedMarker = currentGeneratedItemsMergeMarker()
    }

    /// Forgets any prior merge marker after `generatedItems` is replaced.
    func clearGeneratedItemsLastProcessedMarker() {
        generatedItemsLastProcessedMarker = nil
    }

    /// Merges persisted and newly processed items without duplication.
    func mergeGeneratedItemsWithProcessed() -> [RunItem] {
        var merged = generatedItems
        guard let processed = lastProcessedResponse, !processed.newItems.isEmpty else { return merged }
        let marker = currentGeneratedItemsMergeMarker()
        if let marker, generatedItemsLastProcessedMarker == marker {
            return merged
        }

        struct Pair: Hashable { var first: String; var second: String }
        var seenIdTypes: Set<Pair> = []
        var seenCallIds: Set<String> = []
        var seenCallIdTypes: Set<Pair> = []
        func identity(_ item: RunItem) -> (id: String?, type: String?, callId: String?) {
            let raw = item.rawItem
            return (raw.string("id"), raw.string("type") ?? item.type, raw.string("call_id"))
        }

        for existing in merged {
            let (itemId, itemType, callId) = identity(existing)
            if let itemId, !itemId.isEmpty, let itemType { seenIdTypes.insert(Pair(first: itemId, second: itemType)) }
            if let callId, !callId.isEmpty, let itemType {
                seenCallIdTypes.insert(Pair(first: callId, second: itemType))
            } else if let callId, !callId.isEmpty {
                seenCallIds.insert(callId)
            }
        }
        for newItem in processed.newItems {
            let (itemId, itemType, callId) = identity(newItem)
            if let callId, !callId.isEmpty, let itemType {
                if seenCallIdTypes.contains(Pair(first: callId, second: itemType)) { continue }
            } else if let callId, !callId.isEmpty, seenCallIds.contains(callId) {
                continue
            }
            if let itemId, !itemId.isEmpty, let itemType, seenIdTypes.contains(Pair(first: itemId, second: itemType)) { continue }
            if let itemId, !itemId.isEmpty, let itemType { seenIdTypes.insert(Pair(first: itemId, second: itemType)) }
            if let callId, !callId.isEmpty, let itemType {
                seenCallIdTypes.insert(Pair(first: callId, second: itemType))
            } else if let callId, !callId.isEmpty {
                seenCallIds.insert(callId)
            }
            merged.append(newItem)
        }
        if let marker { generatedItemsLastProcessedMarker = marker }
        return merged
    }

    /// Maps generated occurrences to the same live occurrences in session history.
    func generatedSessionItemIndexes(_ generated: [RunItem]) -> [Int?] {
        var indexesByIdentity: [ObjectIdentifier: [Int]] = [:]
        for (index, sessionItem) in sessionItems.enumerated() {
            indexesByIdentity[ObjectIdentifier(sessionItem), default: []].append(index)
        }
        var used: Set<Int> = []
        return generated.map { item in
            var candidates = indexesByIdentity[ObjectIdentifier(item)] ?? []
            while !candidates.isEmpty {
                let candidate = candidates.removeFirst()
                if used.insert(candidate).inserted {
                    indexesByIdentity[ObjectIdentifier(item)] = candidates
                    return candidate
                }
            }
            indexesByIdentity[ObjectIdentifier(item)] = candidates
            return nil
        }
    }

    /// Records the response range and approval occurrences from live item identities.
    func currentResponseGeneratedItemOwnership(_ generated: [RunItem]) -> JSONObject? {
        guard let processed = lastProcessedResponse,
              case .interruption(let interruptions, _)? = currentStep else { return nil }
        let processedItems = processed.newItems
        guard !processedItems.isEmpty, !interruptions.isEmpty, processedItems.count <= generated.count else { return nil }

        let candidateStarts = (0...(generated.count - processedItems.count)).filter { start in
            processedItems.enumerated().allSatisfy { offset, item in generated[start + offset] === item }
        }
        guard candidateStarts.count == 1, let start = candidateStarts.first else { return nil }
        var indexesByIdentity: [ObjectIdentifier: [Int]] = [:]
        for index in (start + processedItems.count)..<generated.count {
            indexesByIdentity[ObjectIdentifier(generated[index]), default: []].append(index)
        }
        var interruptionIndexes: [JSONValue] = []
        for item in interruptions {
            let indexes = indexesByIdentity.removeValue(forKey: ObjectIdentifier(item)) ?? []
            guard indexes.count == 1 else { return nil }
            interruptionIndexes.append(.int(indexes[0]))
        }
        return ["start": .int(start), "end": .int(generated.count), "interruptions": .array(interruptionIndexes)]
    }

    // MARK: Steps

    func serializeCurrentStep(_ identityKeys: [ObjectIdentifier: String]?) -> JSONValue {
        switch currentStep {
        case .runAgain?:
            return ["type": "next_step_run_again"]
        case .interruption(let interruptions, let responseAccepted)?:
            return [
                "type": "next_step_interruption",
                "data": [
                    "interruptions": .array(interruptions.map {
                        .object(serializeToolApprovalInterruption($0, includeToolName: $0.toolName != nil, identityKeys))
                    }),
                    "response_accepted": .bool(responseAccepted),
                    "llm_end_hooks_started": .bool(llmEndHooksStarted),
                ],
            ]
        case nil:
            return .null
        }
    }

    func serializeProcessedResponse(_ processed: ProcessedResponse, _ identityKeys: [ObjectIdentifier: String]?) -> JSONObject {
        var result: JSONObject = [
            "new_items": .array(processed.newItems.map { .object(serializeRunItem($0, identityKeys)) }),
            "tools_used": .array(processed.toolsUsed.map(JSONValue.string)),
        ]
        for (key, value) in serializeToolActionGroups(processed) {
            result[key] = value
        }
        result["interruptions"] = .array(processed.interruptions.map {
            .object(serializeToolApprovalInterruption($0, includeToolName: true, identityKeys))
        })
        // Swift: local MCP recipient bindings are not ported yet.
        result["mcp_tool_bindings"] = [:]
        return result
    }
}

// MARK: - Shared helpers

private func isNoneContext(_ value: any Sendable) -> Bool {
    if value is Void { return true }
    if let optional = value as? OptionalProtocol, optional.isNone { return true }
    if case .null? = value as? JSONValue { return true }
    return false
}

/// Returns the context as a JSON object when it is a mapping (upstream `isinstance(x, Mapping)`).
func mappingContext(_ value: any Sendable) -> JSONObject? {
    switch value {
    case let object as JSONObject: return object
    case let json as JSONValue: return json.objectValue
    case let dictionary as [String: JSONValue]: return JSONObject(dictionary)
    case let dictionary as [String: any Sendable]:
        guard case .object(let object)? = try? jsonDump(dictionary) else { return nil }
        return object
    default: return nil
    }
}

private func describeContextType(_ value: any Sendable) -> String {
    if isNoneContext(value) { return "none" }
    if mappingContext(value) != nil { return "mapping" }
    if value is any Encodable { return "encodable" }
    return "custom"
}

/// Returns `Module:TypeName` for debugging purposes.
private func contextClassPath(_ value: any Sendable) -> String? {
    let reflected = String(reflecting: type(of: value))
    guard let dot = reflected.firstIndex(of: ".") else { return nil }
    return "\(reflected[..<dot]):\(reflected[reflected.index(after: dot)...])"
}

/// Captures context serialization metadata for debugging and recovery hints.
private func buildContextMeta(_ original: any Sendable, serializedVia: String, requiresDeserializer: Bool, omitted: Bool) -> JSONObject {
    let originalType = describeContextType(original)
    var meta: JSONObject = [
        "original_type": .string(originalType),
        "serialized_via": .string(serializedVia),
        "requires_deserializer": .bool(requiresDeserializer),
        "omitted": .bool(omitted),
    ]
    if !["mapping", "none"].contains(originalType), let classPath = contextClassPath(original) {
        // Store the class path for reference only; it is never used to rebuild the type.
        meta["class_path"] = .string(classPath)
    }
    return meta
}

/// Returns the typed identity for a hosted MCP approval key, or `nil` for other keys.
///
/// Swift: approval records are keyed by strings, so hosted MCP identities are encoded as
/// `hosted_mcp:<server_label>:<tool_name>` and `hosted_mcp_call:<request_id>`.
func hostedMCPApprovalIdentity(_ key: String) -> JSONObject? {
    if key.hasPrefix("hosted_mcp_call:") {
        return ["type": "request", "request_id": .string(String(key.dropFirst("hosted_mcp_call:".count)))]
    }
    if key.hasPrefix("hosted_mcp:") {
        let parts = key.split(separator: ":", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count == 3 else { return nil }
        return ["type": "server_tool", "server_label": .string(String(parts[1])), "tool_name": .string(String(parts[2]))]
    }
    return nil
}

/// Serializes an agent reference with an optional duplicate-name identity key.
func serializeAgentReference(_ agent: Agent, _ identityKeys: [ObjectIdentifier: String]?) -> JSONObject {
    var entry: JSONObject = ["name": .string(agent.name)]
    if let identity = identityKeys?[ObjectIdentifier(agent)], identity != agent.name {
        entry["identity"] = .string(identity)
    }
    return entry
}

/// Converts a tool output value to plain JSON.
///
/// Containers are converted element by element, and values JSON cannot represent become their
/// string form, matching upstream's `json.dumps(value, default=str)`.
func serializeOutputValue(_ value: any Sendable) -> JSONValue {
    switch value {
    case let json as JSONValue: return json
    case let object as JSONObject: return .object(object)
    case let string as String: return .string(string)
    case let bool as Bool: return .bool(bool)
    case let int as Int: return .int(int)
    case let double as Double: return double.isFinite ? .double(double) : .string(stringify(value))
    case is Void: return .null
    case let optional as OptionalProtocol where optional.isNone: return .null
    case let list as [any Sendable]: return .array(list.map(serializeOutputValue))
    case let dictionary as [String: any Sendable]:
        return .object(JSONObject(dictionary.sorted { $0.key < $1.key }.map { ($0.key, serializeOutputValue($0.value)) }))
    case let encodable as any Encodable:
        return (try? JSONValue(encoding: encodable)) ?? .string(stringify(value))
    default:
        return .string(stringify(value))
    }
}

func serializeToolApprovalInterruption(
    _ interruption: ToolApprovalItem,
    includeToolName: Bool,
    _ identityKeys: [ObjectIdentifier: String]?
) -> JSONObject {
    var result: JSONObject = [
        "type": "tool_approval_item",
        "raw_item": .object(interruption.rawItem),
        "agent": .object(serializeAgentReference(interruption.agent, identityKeys)),
    ]
    if includeToolName, let toolName = interruption.toolName { result["tool_name"] = .string(toolName) }
    if let namespace = interruption.toolNamespace { result["tool_namespace"] = .string(namespace) }
    if let origin = interruption.toolOrigin { result["tool_origin"] = .object(origin.toJSONDict()) }
    if let lookupKey = interruption.toolLookupKey { result["tool_lookup_key"] = .object(lookupKey.json) }
    if interruption.allowBareNameAlias { result["allow_bare_name_alias"] = true }
    return result
}

/// Builds a dictionary of tool metadata for serialization.
func serializeToolMetadata(_ tool: any Tool, includeDescription: Bool = false, includeParamsSchema: Bool = false) -> JSONObject {
    var metadata: JSONObject = ["name": .string(tool.name)]
    if let function = tool as? FunctionTool {
        if let namespace = function.toolNamespace { metadata["namespace"] = .string(namespace) }
        if let qualified = getFunctionToolQualifiedName(function), qualified != function.name {
            metadata["qualifiedName"] = .string(qualified)
        }
        if let lookupKey = getFunctionToolLookupKeyForTool(function) { metadata["lookupKey"] = .object(lookupKey.json) }
        if includeDescription { metadata["description"] = .string(function.description) }
        if includeParamsSchema { metadata["paramsJsonSchema"] = .object(function.paramsJSONSchema) }
    } else if includeDescription, let custom = tool as? CustomTool {
        metadata["description"] = .string(custom.description)
    }
    return metadata
}

/// Serializes tool-related action groups using a shared spec.
func serializeToolActionGroups(_ processed: ProcessedResponse) -> [(String, JSONValue)] {
    func actions(_ entries: [(toolCall: JSONObject, tool: any Tool)], wrapperKey: String, includeParamsSchema: Bool = false) -> JSONValue {
        .array(entries.map { entry in
            [
                "tool_call": .object(entry.toolCall),
                wrapperKey: .object(serializeToolMetadata(entry.tool, includeDescription: true, includeParamsSchema: includeParamsSchema)),
            ]
        })
    }
    return [
        ("functions", actions(processed.functions.map { ($0.toolCall.json, $0.functionTool) }, wrapperKey: "tool", includeParamsSchema: true)),
        ("computer_actions", actions(processed.computerActions.map { ($0.toolCall.json, $0.computerTool) }, wrapperKey: "computer")),
        ("custom_tool_actions", actions(processed.customToolCalls.map { ($0.toolCall.json, $0.customTool) }, wrapperKey: "custom_tool")),
        ("local_shell_actions", actions(processed.localShellCalls.map { ($0.toolCall.json, $0.localShellTool) }, wrapperKey: "local_shell")),
        ("shell_actions", actions(processed.shellCalls.map { ($0.toolCall, $0.shellTool) }, wrapperKey: "shell")),
        ("apply_patch_actions", actions(processed.applyPatchCalls.map { ($0.toolCall, $0.applyPatchTool) }, wrapperKey: "apply_patch")),
        ("handoffs", .array(processed.handoffs.map { handoff in
            ["tool_call": .object(handoff.toolCall.json), "handoff": ["tool_name": .string(handoff.handoff.toolName)]]
        })),
        ("mcp_approval_requests", .array(processed.mcpApprovalRequests.map { request in
            ["request_item": ["raw_item": .object(request.requestItem.json)], "mcp_tool": .object(serializeMCPTool(request.mcpTool))]
        })),
    ]
}

/// Serializes a hosted MCP tool into a JSON-friendly mapping.
private func serializeMCPTool(_ tool: HostedMCPTool) -> JSONObject {
    ["name": .string(tool.name), "tool_config": .object(tool.toolConfig)]
}

/// Preserves structured guardrail payloads, falling back to their string form.
func serializeGuardrailPayload(_ value: (any Sendable)?) -> JSONValue {
    guard let value else { return .null }
    return serializeOutputValue(value)
}

/// Serializes input guardrail results for persistence.
func serializeInputGuardrailResults(_ results: [InputGuardrailResult]) -> [JSONValue] {
    results.map { result in
        [
            "guardrail": ["type": "input", "name": JSONValue(optional: result.guardrail.name)],
            "output": [
                "tripwireTriggered": .bool(result.output.tripwireTriggered),
                "outputInfo": serializeGuardrailPayload(result.output.outputInfo),
            ],
        ]
    }
}

/// Serializes output guardrail results for persistence.
func serializeOutputGuardrailResults(_ results: [OutputGuardrailResult], _ identityKeys: [ObjectIdentifier: String]?) -> [JSONValue] {
    results.map { result in
        [
            "guardrail": ["type": "output", "name": JSONValue(optional: result.guardrail.name)],
            "output": [
                "tripwireTriggered": .bool(result.output.tripwireTriggered),
                "outputInfo": serializeGuardrailPayload(result.output.outputInfo),
            ],
            "agentOutput": serializeGuardrailPayload(result.agentOutput),
            "agent": .object(serializeAgentReference(result.agent, identityKeys)),
        ]
    }
}

/// Serializes a tool guardrail result for persistence.
func serializeToolGuardrailResult(name: String, output: ToolGuardrailFunctionOutput, typeLabel: String) -> JSONValue {
    [
        "guardrail": ["type": .string(typeLabel), "name": .string(name)],
        "output": [
            "outputInfo": serializeGuardrailPayload(output.outputInfo),
            "behavior": .object(output.behavior.json),
        ],
    ]
}

extension ToolGuardrailBehavior {
    /// The behavior as its upstream `TypedDict` payload.
    var json: JSONObject {
        switch self {
        case .allow: return ["type": "allow"]
        case .rejectContent(let message): return ["type": "reject_content", "message": .string(message)]
        case .raiseException: return ["type": "raise_exception"]
        }
    }

    init?(json: JSONObject) {
        switch json.string("type") {
        case "allow": self = .allow
        case "reject_content": self = .rejectContent(message: json.string("message") ?? "")
        case "raise_exception": self = .raiseException
        default: return nil
        }
    }
}
