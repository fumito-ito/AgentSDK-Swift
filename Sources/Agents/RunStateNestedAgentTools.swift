import Foundation

// Port of the nested agent-tool parts of src/agents/run_state.py: routing approvals to nested
// `Agent.asTool()` run states and copying a state for a result checkpoint (porting/decisions.md
// D20). Serialization of pending nested runs lives in RunStateSerialization.swift and
// RunStateDeserialization.swift.
//
// Swift: raw items are JSON values, so the copy failures upstream guards against when matching
// approvals (indeterminate matches, `Cannot safely copy pending tool approvals`) cannot happen.

extension RunState {
    /// Compares approval identity (upstream `_approval_items_match`).
    static func approvalItemsMatch(_ candidate: ToolApprovalItem, _ approvalItem: ToolApprovalItem) -> Bool {
        if candidate === approvalItem { return true }
        if candidate.agent !== approvalItem.agent { return false }
        guard let candidateIdentity = toolInvocationIdentity(
            candidate.rawItem, toolLookupKey: candidate.toolLookupKey, toolName: candidate.toolName
        ) else {
            return false
        }
        let approvalIdentity = toolInvocationIdentity(
            approvalItem.rawItem, toolLookupKey: approvalItem.toolLookupKey, toolName: approvalItem.toolName
        )
        return candidateIdentity == approvalIdentity
    }

    /// Resolves a detached approval snapshot to the current authoritative pending approval.
    func findCurrentApprovalItem(_ approvalItem: ToolApprovalItem) throws -> ToolApprovalItem? {
        guard case .interruption(let interruptions, _)? = currentStep else { return nil }
        let matches = interruptions.filter { Self.approvalItemsMatch($0, approvalItem) }
        if matches.count > 1 {
            throw UserError(
                "Cannot apply approval because multiple current pending approvals contain the "
                    + "same tool invocation identity, or because it belongs to both the current run "
                    + "and a nested agent-tool run. Use unique call IDs."
            )
        }
        return matches.first
    }

    /// The identities of every invocation queued in the last processed response.
    private func currentResponseIdentities(_ processed: ProcessedResponse) -> [ToolInvocationIdentity?] {
        processed.functions.map {
            toolInvocationIdentity($0.toolCall.json, toolLookupKey: getFunctionToolLookupKeyForTool($0.functionTool))
        }
            + processed.handoffs.map { toolInvocationIdentity($0.toolCall.json, invocationRole: "handoff") }
            + processed.computerActions.map { toolInvocationIdentity($0.toolCall.json, toolName: $0.computerTool.name) }
            + processed.customToolCalls.map { toolInvocationIdentity($0.toolCall.json, toolName: $0.customTool.name) }
            + processed.localShellCalls.map { toolInvocationIdentity($0.toolCall.json, toolName: $0.localShellTool.name) }
            + processed.shellCalls.map { toolInvocationIdentity($0.toolCall, toolName: $0.shellTool.name) }
            + processed.applyPatchCalls.map { toolInvocationIdentity($0.toolCall, toolName: $0.applyPatchTool.name) }
            + processed.functionToolsNotFound.map { toolInvocationIdentity($0.toolCall.json, toolName: $0.toolName) }
            + processed.mcpApprovalRequests.map { toolInvocationIdentity($0.requestItem.json) }
    }

    /// Finds the nested agent-tool state that owns an approval interruption.
    func findNestedApprovalState(_ approvalItem: ToolApprovalItem) throws -> (RunState, ToolApprovalItem)? {
        guard let processed = lastProcessedResponse else { return nil }

        var nestedCandidates: [(RunState, ToolApprovalItem)] = []
        for run in processed.functions {
            guard let pendingResult = peekAgentToolRunResult(run.toolCall.json, scope: agentToolStateScope) else { continue }
            let nestedState = pendingResult.toState()
            guard nestedState !== self else { continue }
            for candidate in pendingResult.interruptions {
                let recursiveOwner = try nestedState.findNestedApprovalState(candidate)
                nestedCandidates.append(recursiveOwner ?? (nestedState, candidate))
            }
        }

        let currentApprovalItem = try findCurrentApprovalItem(approvalItem)
        let canonicalMatches = nestedCandidates.filter { Self.approvalItemsMatch($0.1, approvalItem) }

        let identityItem = currentApprovalItem ?? approvalItem
        var currentStateOwnsApproval = false
        if let identity = toolInvocationIdentity(
            identityItem.rawItem, toolLookupKey: identityItem.toolLookupKey, toolName: identityItem.toolName
        ) {
            if let record = context.toolInvocations[identity.callId] {
                currentStateOwnsApproval = !record.completed && record.matches(identity)
            }
            currentStateOwnsApproval = currentStateOwnsApproval && currentResponseIdentities(processed).contains(identity)
        }

        if currentStateOwnsApproval, !canonicalMatches.isEmpty {
            throw UserError(
                "Cannot apply approval because the same tool invocation identity belongs to both "
                    + "the current run and a nested agent-tool run. Use distinct call IDs."
            )
        }
        if canonicalMatches.count == 1 { return canonicalMatches[0] }
        if canonicalMatches.count > 1 {
            throw UserError(
                "Cannot apply approval because multiple nested agent-tool runs contain the same "
                    + "tool invocation identity. Use unique call IDs within nested runs."
            )
        }
        return nil
    }

    /// Copies SDK-owned decision state when nesting this checkpoint in a result snapshot
    /// (upstream `_copy_for_result_checkpoint`).
    func copyForResultCheckpoint() -> RunState {
        let copied = RunState(
            context: context.copyForRunState(),
            originalInput: originalInput,
            startingAgent: startingAgent,
            maxTurns: maxTurns,
            conversationId: conversationId,
            previousResponseId: previousResponseId,
            autoPreviousResponseId: autoPreviousResponseId
        )
        copied.currentTurn = currentTurn
        copied.currentAgent = currentAgent
        copied.modelResponses = modelResponses
        copied.generatedItems = generatedItems
        copied.sessionItems = sessionItems
        copied.pendingInputItems = pendingInputItems
        copied.reasoningItemIdPolicy = reasoningItemIdPolicy
        copied.inputGuardrailResults = inputGuardrailResults
        copied.outputGuardrailResults = outputGuardrailResults
        copied.toolInputGuardrailResults = toolInputGuardrailResults
        copied.toolOutputGuardrailResults = toolOutputGuardrailResults
        copied.currentStep = currentStep
        copied.llmEndHooksStarted = llmEndHooksStarted
        copied.lastProcessedResponse = lastProcessedResponse
        copied.generatedItemsLastProcessedMarker = generatedItemsLastProcessedMarker
        copied.currentTurnPersistedItemCount = currentTurnPersistedItemCount
        copied.toolUseTrackerSnapshot = toolUseTrackerSnapshot
        copied.traceState = traceState
        copied.generatedPromptCacheKey = generatedPromptCacheKey
        copied.schemaVersion = schemaVersion

        for run in lastProcessedResponse?.functions ?? [] {
            guard let pendingResult = peekAgentToolRunResult(run.toolCall.json, scope: agentToolStateScope) else { continue }
            let interruptions = pendingResult.interruptions
            guard !interruptions.isEmpty else { continue }
            let nestedState = getAgentToolResumeState(pendingResult)?.copyForResultCheckpoint() ?? pendingResult.toState()
            guard nestedState !== self else { continue }
            recordAgentToolResumeState(
                run.toolCall.json, nestedState, scope: copied.agentToolStateScope, approvalItems: interruptions
            )
        }
        return copied
    }
}

/// Binds detached nested approval checkpoints to the scope of a new outer checkpoint (upstream
/// `_copy_pending_nested_agent_tool_states`).
///
/// The first conversion moves each pending nested run out of the result's scope into `templates`,
/// so every later conversion of the same result copies the same nested state.
func copyPendingNestedAgentToolStates(
    _ state: RunState,
    from result: any RunResultBase,
    templates: SharedBox<[AgentToolCallSignature: RunState]>
) {
    guard let processed = state.lastProcessedResponse else { return }
    let sourceScope = getAgentToolStateScope(result.contextWrapper)
    for run in processed.functions {
        let toolCall = run.toolCall.json
        let key = AgentToolCallSignature(toolCall)
        let nestedState: RunState
        let resolvedInterruptions: [ToolApprovalItem]
        if let template = templates.value[key] {
            nestedState = template.copyForResultCheckpoint()
            resolvedInterruptions = template.getInterruptions()
        } else {
            guard let pendingResult = peekAgentToolRunResult(toolCall, scope: sourceScope) else { continue }
            let pendingInterruptions = pendingResult.interruptions
            guard !pendingInterruptions.isEmpty else { continue }
            let template = getAgentToolResumeState(pendingResult)?.copyForResultCheckpoint() ?? pendingResult.toState()
            templates.mutate { $0[key] = template }
            dropAgentToolRunResult(toolCall, scope: sourceScope)
            nestedState = template.copyForResultCheckpoint()
            resolvedInterruptions = pendingInterruptions
        }
        guard nestedState !== state else { continue }
        recordAgentToolResumeState(toolCall, nestedState, scope: state.agentToolStateScope, approvalItems: resolvedInterruptions)
    }
}
