import Foundation

// Port of `_validate_completed_tool_invocations` in src/agents/run_state.py: reconciles the restored
// tool invocation ledger with the calls and outputs in a snapshot's history.
//
// Swift: split out of RunStateDeserialization.swift to stay within SwiftLint length limits
// (porting/decisions.md D13).

/// The `(type, call_id, fingerprint)` triple upstream uses to compare invocations.
private struct InvocationKey: Hashable {
    var type: String
    var callId: String
    var fingerprint: String

    init(_ identity: ToolInvocationIdentity) {
        type = identity.type
        callId = identity.callId
        fingerprint = identity.fingerprint
    }

    init(_ record: ToolInvocationRecord, callId: String) {
        type = record.invocationType
        self.callId = callId
        fingerprint = record.fingerprint
    }
}

/// How one restored item resolves to an invocation identity.
private struct InvocationDetails {
    var rawItem: JSONObject
    var toolLookupKey: FunctionToolLookupKey?
    var toolName: String?
    var invocationRole: String?
}

/// The identities one restored call item can have, in the order they were derived.
private struct InvocationOccurrence {
    var entries: [(key: InvocationKey, details: InvocationDetails)] = []

    var keys: [InvocationKey] { entries.map(\.key) }

    mutating func add(_ key: InvocationKey, _ details: InvocationDetails) {
        if !entries.contains(where: { $0.key == key }) { entries.append((key, details)) }
    }

    func details(for key: InvocationKey) -> InvocationDetails? {
        entries.first { $0.key == key }?.details
    }
}

/// The invocation type a native tool produces, for legacy snapshots without resolved tool names.
private func legacyNativeInvocationType(_ tool: any Tool) -> String? {
    switch tool {
    case is ComputerTool: return "computer_call"
    case is CustomTool: return "custom_tool_call"
    case is LocalShellTool: return "local_shell_call"
    case is ShellTool: return "shell_call"
    case is ApplyPatchTool: return "apply_patch_call"
    default: return nil
    }
}

/// Collects every restored call and output so the ledger can be checked against them.
private struct RestoredInvocationHistory {
    let applyPatchTools: [ApplyPatchTool]
    let legacyNativeToolNames: [String: Set<String>]
    var resolvedToolNames: [String: String] = [:]
    var occurrences: [InvocationOccurrence] = []
    var outputs: [String: [String: JSONObject]] = [:]
    var uncanonicalCallIds: Set<String> = []

    init(startingAgent: Agent, reconstructLegacy: Bool) {
        let agents = iterAgentGraph(startingAgent)
        applyPatchTools = agents.flatMap { $0.tools.compactMap { $0 as? ApplyPatchTool } }
        var names: [String: Set<String>] = [:]
        if reconstructLegacy {
            for agent in agents {
                for tool in agent.tools {
                    if let invocationType = legacyNativeInvocationType(tool) {
                        names[invocationType, default: []].insert(tool.name)
                    }
                }
            }
        }
        legacyNativeToolNames = names
    }

    mutating func collectResolvedToolName(_ rawItem: JSONObject, _ toolName: String?) {
        guard let toolName, !toolName.isEmpty, let callId = toolInvocationCallId(rawItem)?.callId else { return }
        if resolvedToolNames[callId] == nil { resolvedToolNames[callId] = toolName }
    }

    func output(type: String, callId: String) -> JSONObject? {
        outputs[type]?[callId]
    }

    func outputTypes(for callId: String) -> Set<String> {
        Set(outputs.compactMap { type, byCallId in byCallId[callId] == nil ? nil : type })
    }

    mutating func record(
        _ rawItem: JSONObject,
        toolLookupKey: FunctionToolLookupKey? = nil,
        toolName: String? = nil,
        invocationRole: String? = nil,
        allowHandoffAlternative: Bool = false
    ) {
        if let output = toolOutputIdentity(rawItem), outputs[output.type]?[output.callId] == nil {
            outputs[output.type, default: [:]][output.callId] = rawItem
        }
        let callIdentity = toolInvocationCallId(rawItem)
        var toolName = toolName
        if toolName == nil, let callIdentity, let callId = callIdentity.callId {
            toolName = resolvedToolNames[callId]
            if toolName == nil, let candidates = legacyNativeToolNames[callIdentity.type], candidates.count == 1 {
                toolName = candidates.first
            }
        }
        var occurrence = InvocationOccurrence()
        func addIdentity(_ role: String?) {
            if let identity = toolInvocationIdentity(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: role) {
                occurrence.add(InvocationKey(identity), InvocationDetails(rawItem: rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: role))
            }
        }
        addIdentity(invocationRole)
        if allowHandoffAlternative, invocationRole == nil { addIdentity("handoff") }
        let rawName = rawItem.string("name")
        if applyPatchTools.contains(where: { isApplyPatchName(rawName, $0) }),
           let fallback = (try? normalizeApplyPatchFallbackCall(rawItem)).flatMap({ $0 }),
           let identity = toolInvocationIdentity(fallback, toolName: toolName) {
            occurrence.add(InvocationKey(identity), InvocationDetails(rawItem: fallback, toolLookupKey: nil, toolName: toolName, invocationRole: nil))
        }
        if !occurrence.entries.isEmpty {
            occurrences.append(occurrence)
        } else if let callId = callIdentity?.callId {
            uncanonicalCallIds.insert(callId)
        }
    }

    mutating func record(_ item: RunItem) {
        let approval = item as? ToolApprovalItem
        record(
            item.rawItem,
            toolLookupKey: approval?.toolLookupKey,
            toolName: (item as? ToolCallItem)?.toolName ?? approval?.toolName,
            invocationRole: item is HandoffCallItem ? "handoff" : nil
        )
    }

    /// Groups occurrences whose identities all share one call ID.
    var occurrencesByCallId: [(callId: String, occurrences: [InvocationOccurrence])] {
        var grouped: [(callId: String, occurrences: [InvocationOccurrence])] = []
        for occurrence in occurrences {
            let callIds = Set(occurrence.keys.map(\.callId))
            guard callIds.count == 1, let callId = callIds.first else { continue }
            if let index = grouped.firstIndex(where: { $0.callId == callId }) {
                grouped[index].occurrences.append(occurrence)
            } else {
                grouped.append((callId, [occurrence]))
            }
        }
        return grouped
    }
}

extension RunState {
    /// Reconciles invocation bindings with restored calls and outputs.
    func validateCompletedToolInvocations(reconstructLegacy: Bool) throws {
        let completedRecords = context.toolInvocations.filter(\.value.completed)
        var history = RestoredInvocationHistory(startingAgent: startingAgent, reconstructLegacy: reconstructLegacy)
        collectResolvedToolNames(into: &history)

        for item in generatedItems { history.record(item) }
        for item in sessionItems { history.record(item) }
        for item in lastProcessedResponse?.newItems ?? [] { history.record(item) }
        var responses = modelResponses
        if lastProcessedResponse == nil, case .interruption(_, true)? = currentStep, !responses.isEmpty {
            // A server-accepted response is checkpointed before fallible local processing. Its raw
            // invocations remain durable for diagnostics, but they are not registered runtime work
            // unless response processing succeeds.
            responses.removeLast()
        }
        for response in responses {
            for rawItem in response.output { history.record(rawItem, allowHandoffAlternative: true) }
        }
        if case .items(let items) = originalInput {
            for rawItem in items { history.record(rawItem, allowHandoffAlternative: true) }
        }

        let grouped = history.occurrencesByCallId
        if reconstructLegacy {
            try reconstructLegacyInvocations(grouped, history: history)
        }
        let invocations = context.toolInvocations
        context.toolStateBox.mutate { state in
            for entry in grouped where invocations[entry.callId] == nil {
                state.restoredUnboundApprovalCallIds.insert(entry.callId)
            }
            state.restoredUnboundApprovalCallIds.formUnion(history.uncanonicalCallIds)
        }

        for (callId, record) in completedRecords {
            let expected = InvocationKey(record, callId: callId)
            let occurrences = grouped.first { $0.callId == callId }?.occurrences ?? []
            if occurrences.isEmpty || history.uncanonicalCallIds.contains(callId)
                || occurrences.contains(where: { !$0.keys.contains(expected) })
                || history.output(type: record.invocationType, callId: callId) == nil {
                throw UserError("RunState completed tool invocation does not match a restored tool call and output.")
            }
        }
    }

    private func collectResolvedToolNames(into history: inout RestoredInvocationHistory) {
        for item in generatedItems + sessionItems + (lastProcessedResponse?.newItems ?? []) {
            history.collectResolvedToolName(item.rawItem, (item as? ToolCallItem)?.toolName ?? (item as? ToolApprovalItem)?.toolName)
        }
        guard let processed = lastProcessedResponse else { return }
        for run in processed.computerActions { history.collectResolvedToolName(run.toolCall.json, run.computerTool.name) }
        for run in processed.customToolCalls { history.collectResolvedToolName(run.toolCall.json, run.customTool.name) }
        for run in processed.localShellCalls { history.collectResolvedToolName(run.toolCall.json, run.localShellTool.name) }
        for run in processed.shellCalls { history.collectResolvedToolName(run.toolCall, run.shellTool.name) }
        for run in processed.applyPatchCalls { history.collectResolvedToolName(run.toolCall, run.applyPatchTool.name) }
        for run in processed.functionToolsNotFound { history.collectResolvedToolName(run.toolCall.json, run.toolName) }
    }

    /// Rebuilds ledger records from history for snapshots written before the ledger existed.
    private func reconstructLegacyInvocations(
        _ grouped: [(callId: String, occurrences: [InvocationOccurrence])],
        history: RestoredInvocationHistory
    ) throws {
        for (callId, occurrences) in grouped {
            if context.toolInvocations[callId] != nil || history.uncanonicalCallIds.contains(callId) { continue }
            let outputTypes = history.outputTypes(for: callId)
            if outputTypes.isEmpty { continue }
            guard let first = occurrences.first else { continue }
            let common = first.keys.filter { key in occurrences.allSatisfy { $0.keys.contains(key) } }
            let completedKeys = common.filter { outputTypes.contains($0.type) }
            let key: InvocationKey
            let details: InvocationDetails
            if !completedKeys.isEmpty {
                key = completedKeys.first { first.details(for: $0)?.invocationRole == nil } ?? completedKeys[0]
                guard let found = occurrences.lazy.compactMap({ $0.details(for: key) }).first else { continue }
                details = found
            } else {
                guard let entry = first.entries.first else { continue }
                (key, details) = (entry.key, entry.details)
            }
            let status = try context.toolInvocationStatus(
                details.rawItem,
                toolLookupKey: details.toolLookupKey,
                toolName: details.toolName,
                invocationRole: details.invocationRole
            )
            guard status != nil else { continue }
            if !completedKeys.isEmpty, let output = history.output(type: key.type, callId: callId) {
                context.markToolCallCompleted(output)
            } else {
                try context.markToolInvocationExecuted(
                    details.rawItem,
                    toolLookupKey: details.toolLookupKey,
                    toolName: details.toolName,
                    invocationRole: details.invocationRole
                )
            }
        }

        for approval in currentStep?.interruptionItems ?? [] {
            do {
                try context.restorePendingApprovalBinding(approval)
            } catch is ModelBehaviorError {
                if let callId = RunContextWrapper.resolveCallId(approval) {
                    context.toolStateBox.mutate { _ = $0.restoredUnboundApprovalCallIds.insert(callId) }
                }
            }
        }
        context.toolStateBox.mutate { $0.markRestoredUnboundApprovalCallIds() }
    }
}
