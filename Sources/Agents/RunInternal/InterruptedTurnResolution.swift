import Foundation

// Port of `resolve_interrupted_turn` from src/agents/run_internal/turn_resolution.py.
//
// Swift: the resume path lives in its own file so TurnResolution.swift stays within SwiftLint's
// length limits (porting/decisions.md D19). Upstream keeps its state in closures; Swift keeps it
// in `InterruptedTurnResolver`. Not ported yet: the `response_accepted` fast path, nested
// `Agent.asTool()` run state (`agent_tool_state.py`), and programmatic tool calling parents.
// Sandbox execution clones are out of scope (D2), so the public and execution agents are the same.

/// Resumes a turn that was interrupted for tool approvals.
func resolveInterruptedTurn(
    agent: Agent,
    allTools: [any Tool],
    originalInput: RunInput,
    originalPreStepItems: [RunItem],
    newResponse: ModelResponse,
    processedResponse: ProcessedResponse,
    hooks: RunHooks,
    contextWrapper: RunContextWrapper,
    runConfig: RunConfig,
    serverManagesConversation: Bool,
    errorHandlers: RunErrorHandlers?,
    runState: RunState? = nil
) async throws -> SingleStepResult {
    let resolver = InterruptedTurnResolver(
        agent: agent,
        allTools: allTools,
        originalInput: originalInput,
        originalPreStepItems: originalPreStepItems,
        newResponse: newResponse,
        processedResponse: processedResponse,
        hooks: hooks,
        contextWrapper: contextWrapper,
        runConfig: runConfig,
        serverManagesConversation: serverManagesConversation,
        runState: runState
    )
    return try await resolver.resolve()
}

/// A `(type, call_id)` pair indexing committed tool outputs (upstream `_build_tool_output_index`).
private struct ToolOutputKey: Hashable {
    var type: String
    var callId: String
}

private func buildToolOutputIndex(_ items: [RunItem]) -> Set<ToolOutputKey> {
    var index: Set<ToolOutputKey> = []
    for case let item as ToolCallOutputItem in items {
        if let type = item.rawItem.type, let callId = item.rawItem.callIdOrId {
            index.insert(ToolOutputKey(type: type, callId: callId))
        }
    }
    return index
}

private func extractShellCallId(_ toolCall: JSONObject) throws -> String {
    guard let value = toolCall.string("call_id"), !value.isEmpty else {
        throw ModelBehaviorError("Shell call is missing call_id.")
    }
    return value
}

private func extractApplyPatchCallId(_ toolCall: JSONObject) throws -> String {
    guard let value = toolCall.string("call_id"), !value.isEmpty else {
        throw ModelBehaviorError("Apply patch call is missing call_id.")
    }
    return value
}

private func customCallId(_ run: ToolRunCustom) throws -> String {
    guard let callId = run.toolCall.json.callIdOrId, !callId.isEmpty else {
        throw ModelBehaviorError("Custom tool call is missing call_id.")
    }
    return callId
}

private func computerCallId(_ run: ToolRunComputerAction) throws -> String {
    guard let callId = run.toolCall.json.callIdOrId, !callId.isEmpty else {
        throw ModelBehaviorError("Computer action is missing call_id.")
    }
    return callId
}

/// Keeps a resumed run's `RunState` current while tools finish, so a later failure (a hook, a
/// guardrail, cancellation) does not lose outputs whose side effects already happened (upstream
/// `_commit_tool_output` and `_checkpoint_new_items`).
private final class ResumeCheckpoint: @unchecked Sendable {
    private let runState: RunState?
    private let originalPreStepItems: [RunItem]
    private let contextWrapper: RunContextWrapper
    private let callPositions: [String: Int]
    private let committed = SharedBox<[RunItem]>([])

    init(runState: RunState?, originalPreStepItems: [RunItem], callPositions: [String: Int], contextWrapper: RunContextWrapper) {
        self.runState = runState
        self.originalPreStepItems = originalPreStepItems
        self.contextWrapper = contextWrapper
        self.callPositions = callPositions
    }

    private func position(_ item: RunItem) -> Int {
        item.rawItem.callIdOrId.flatMap { callPositions[$0] } ?? callPositions.count
    }

    /// Records one tool output in model order as soon as it exists.
    func commitToolOutput(_ item: RunItem) {
        committed.mutate { outputs in
            guard !outputs.contains(where: { $0 === item }) else { return }
            outputs.append(item)
            // A stable sort keeps completion order among outputs without a known call position.
            outputs = outputs.enumerated()
                .sorted { (position($0.element), $0.offset) < (position($1.element), $1.offset) }
                .map(\.element)
            runState?.generatedItems = originalPreStepItems + outputs
        }
        contextWrapper.markToolCallCompleted(item.rawItem)
    }

    /// Writes the resumed turn's items so far and registers their outputs.
    func checkpoint(_ newItems: [RunItem]) throws {
        committed.mutate { _ in runState?.generatedItems = originalPreStepItems + newItems }
        try registerToolCallItems(contextWrapper, newItems)
    }
}

/// The state of one resumed turn (upstream keeps it in the closures of `resolve_interrupted_turn`).
private final class InterruptedTurnResolver {
    let agent: Agent
    let allTools: [any Tool]
    let originalInput: RunInput
    let originalPreStepItems: [RunItem]
    let newResponse: ModelResponse
    var processed: ProcessedResponse
    let hooks: RunHooks
    let contextWrapper: RunContextWrapper
    let runConfig: RunConfig
    let serverManagesConversation: Bool
    let runState: RunState?

    let outputIndex: Set<ToolOutputKey>
    var pendingApprovalItems: [ToolApprovalItem] = []
    var allowLegacyNameAgentMatch = false
    var approvalItemsByCallId: [String: ToolApprovalItem] = [:]
    var functionApprovalItems: [ToolApprovalItem] = []
    var functionApprovalItemsByCallId: [String: ToolApprovalItem] = [:]
    /// Original approval → its validated stable copy.
    var validatedFunctionApprovals: [ObjectIdentifier: ToolApprovalItem] = [:]
    /// Stable copy → the original approval it was validated from.
    var stableFunctionApprovalSources: [ObjectIdentifier: ToolApprovalItem] = [:]

    var rejectedFunctionOutputs: [RunItem] = []
    var rejectedFunctionCallIds: Set<String> = []
    var pendingInterruptions: [ToolApprovalItem] = []
    var pendingInterruptionKeys: Set<String> = []

    var duplicateQueuedCallIds: Set<String> = []
    var nonFunctionOwnedCallIds: Set<String> = []
    var authoritativeResponseFunctionCalls: [ResponseFunctionToolCall] = []
    var responseCallPositions: [String: Int] = [:]
    var queuedFunctionCallIdsByLookupKey: [FunctionToolLookupKey: Set<String>] = [:]
    var queuedFunctionLookupKeysByCallId: [String: Set<FunctionToolLookupKey>] = [:]
    var canonicalQueuedFunctionCalls: [String: ResponseFunctionToolCall] = [:]

    var customCallsToReconcile: [ResponseCustomToolCall] = []
    var customCallIdentities: [String: ToolInvocationIdentity] = [:]
    var rejectedCustomApprovals: [(callId: String, approval: ToolApprovalItem, call: ResponseCustomToolCall)] = []

    var availableFunctionTools: [any Tool] = []
    var executedHandoffCallIds: Set<String> = []
    var callsToReconcile: [ResponseFunctionToolCall] = []
    var reconciledCallIds: Set<String> = []
    var missingFunctionTools: [ToolRunFunctionNotFound] = []
    var missingFunctionCallIds: Set<String> = []

    init(
        agent: Agent,
        allTools: [any Tool],
        originalInput: RunInput,
        originalPreStepItems: [RunItem],
        newResponse: ModelResponse,
        processedResponse: ProcessedResponse,
        hooks: RunHooks,
        contextWrapper: RunContextWrapper,
        runConfig: RunConfig,
        serverManagesConversation: Bool,
        runState: RunState?
    ) {
        self.agent = agent
        self.allTools = allTools
        self.originalInput = originalInput
        self.originalPreStepItems = originalPreStepItems
        self.newResponse = newResponse
        self.processed = processedResponse
        self.hooks = hooks
        self.contextWrapper = contextWrapper
        self.runConfig = runConfig
        self.serverManagesConversation = serverManagesConversation
        self.runState = runState
        self.outputIndex = buildToolOutputIndex(originalPreStepItems)
    }

    func hasOutputItem(_ callId: String, _ expectedType: String) -> Bool {
        outputIndex.contains(ToolOutputKey(type: expectedType, callId: callId))
    }

    // MARK: - Approval ownership

    /// Schema 1.6 and earlier only serialized approval owners by agent name. With duplicate-name
    /// agents, deserialization can legitimately resolve the approval to a sibling instance, so resume
    /// accepts a same-name match for those legacy snapshots. Schema 1.7+ persists duplicate-name
    /// identities, so newer snapshots keep requiring object identity.
    private static func allowsLegacyNameAgentMatch(_ runState: RunState?) -> Bool {
        guard let version = runState?.schemaVersion else { return false }
        let parts = version.split(separator: ".").map { Int($0) }
        guard !parts.contains(nil) else { return false }
        let numbers = parts.compactMap { $0 }
        return numbers.lexicographicallyPrecedes([1, 7])
    }

    func approvalMatchesAgent(_ approval: ToolApprovalItem) -> Bool {
        approval.agent === agent || (allowLegacyNameAgentMatch && approval.agent.name == agent.name)
    }

    private func pendingApprovalsFromState() -> [ToolApprovalItem] {
        if case .interruption(let interruptions, _)? = runState?.currentStep {
            return interruptions
        }
        return originalPreStepItems.compactMap { $0 as? ToolApprovalItem }
    }

    /// Adds a pending interruption once per call ID, writing a stable copy back to its source.
    ///
    /// Swift: upstream also moves the source to the current agent. `RunItem.agent` is immutable,
    /// so a legacy (schema < 1.7) approval restored to a same-name sibling keeps that owner.
    func addPendingInterruption(_ item: ToolApprovalItem) {
        var item = item
        if let source = stableFunctionApprovalSources[ObjectIdentifier(item)] {
            source.rawItem = item.rawItem
            source.toolName = item.toolName
            source.toolNamespace = item.toolNamespace
            source.toolOrigin = item.toolOrigin
            source.toolLookupKey = item.toolLookupKey
            source.allowBareNameAlias = item.allowBareNameAlias
            item = source
        }
        let key = getToolApprovalItemCallId(item) ?? "raw:\(ObjectIdentifier(item).hashValue)"
        guard pendingInterruptionKeys.insert(key).inserted else { return }
        pendingInterruptions.append(item)
    }

    func approvalPersistedLookupKey(_ approval: ToolApprovalItem) -> FunctionToolLookupKey? {
        if let persisted = approval.toolLookupKey { return persisted }
        guard let toolName = approval.toolName, !toolName.isEmpty else { return nil }
        return getFunctionToolLookupKey(toolName, approval.toolNamespace)
    }

    // MARK: - Resolution

    func resolve() async throws -> SingleStepResult {
        try registerToolCallItems(contextWrapper, originalPreStepItems, validateInvocations: false)
        contextWrapper.turnInput = ItemHelpers.inputToNewInputList(originalInput)

        pendingApprovalItems = pendingApprovalsFromState()
        allowLegacyNameAgentMatch = Self.allowsLegacyNameAgentMatch(runState)
        approvalItemsByCallId = indexApprovalItemsByCallId(pendingApprovalItems.filter(approvalMatchesAgent))

        try dedupeProcessedResponseInvocations(
            &processed, contextWrapper: contextWrapper, existingItems: originalPreStepItems, filterCompleted: false
        )
        indexQueuedCalls()
        if let malformed = try validateFunctionApprovals() {
            return malformed
        }
        try transferLegacyFunctionDecisions()
        try collectCustomCallsToReconcile()

        let classified = try await classifyReconciledCalls()
        let rejectedCustomApprovalOutputs = try await buildRejectedCustomApprovalOutputs()
        let (reconciledFunctions, reconciledHandoffs) = try await reconcileCalls(classified)
        try collectUnreconciledPendingApprovals()
        processed.handoffs = reconciledHandoffs

        let selectableFunctionRuns = reconciledFunctions.filter { !missingFunctionCallIds.contains($0.toolCall.callId) }
        let plan = try await buildResumePlan(selectableFunctionRuns)
        let missingOutputItems = try await buildToolNotFoundOutputItems(
            agent: agent, calls: missingFunctionTools, contextWrapper: contextWrapper, runConfig: runConfig
        )
        processed.functions = reconciledFunctions
        processed.functionToolsNotFound = missingFunctionTools

        return try await executeAndFinish(
            plan: plan,
            selectableFunctionRuns: selectableFunctionRuns,
            missingOutputItems: missingOutputItems,
            rejectedCustomApprovalOutputs: rejectedCustomApprovalOutputs
        )
    }
}

// MARK: - Queued calls and approvals

extension InterruptedTurnResolver {
    func indexQueuedCalls() {
        let queuedCallItems: [JSONObject] = processed.functions.map(\.toolCall.json)
            + processed.handoffs.map(\.toolCall.json)
            + processed.computerActions.map(\.toolCall.json)
            + processed.customToolCalls.map(\.toolCall.json)
            + processed.localShellCalls.map(\.toolCall.json)
            + processed.shellCalls.map(\.toolCall)
            + processed.applyPatchCalls.map(\.toolCall)
            + processed.mcpApprovalRequests.map(\.requestItem.json)
            + processed.functionToolsNotFound.map(\.toolCall.json)
        var counts: [String: Int] = [:]
        for item in queuedCallItems {
            if let callId = item.callIdOrId { counts[callId, default: 0] += 1 }
        }
        duplicateQueuedCallIds = Set(counts.filter { $0.value > 1 }.keys)
        let nonFunctionItems: [JSONObject] = processed.computerActions.map(\.toolCall.json)
            + processed.customToolCalls.map(\.toolCall.json)
            + processed.localShellCalls.map(\.toolCall.json)
            + processed.shellCalls.map(\.toolCall)
            + processed.applyPatchCalls.map(\.toolCall)
            + processed.mcpApprovalRequests.map(\.requestItem.json)
        nonFunctionOwnedCallIds = Set(nonFunctionItems.compactMap(\.callIdOrId))

        let responseFunctionCalls = newResponse.output.compactMap(ResponseFunctionToolCall.init(json:))
        authoritativeResponseFunctionCalls = responseFunctionCalls.filter { !nonFunctionOwnedCallIds.contains($0.callId) }
        for (index, output) in newResponse.output.enumerated() {
            if let callId = output.callIdOrId, responseCallPositions[callId] == nil {
                responseCallPositions[callId] = index
            }
        }

        let authoritativeFunctionCalls = processed.functions.map(\.toolCall) + authoritativeResponseFunctionCalls
        for call in authoritativeFunctionCalls {
            guard let lookupKey = getFunctionToolLookupKeyForCall(call.json) else { continue }
            queuedFunctionCallIdsByLookupKey[lookupKey, default: []].insert(call.callId)
            queuedFunctionLookupKeysByCallId[call.callId, default: []].insert(lookupKey)
        }
        for run in processed.functions where !duplicateQueuedCallIds.contains(run.toolCall.callId) {
            canonicalQueuedFunctionCalls[run.toolCall.callId] = run.toolCall
        }
        for call in authoritativeResponseFunctionCalls where !duplicateQueuedCallIds.contains(call.callId) {
            if canonicalQueuedFunctionCalls[call.callId] == nil { canonicalQueuedFunctionCalls[call.callId] = call }
        }
    }

    private var queuedFunctionCallIds: Set<String> {
        queuedFunctionCallIdsByLookupKey.values.reduce(into: Set<String>()) { $0.formUnion($1) }
    }

    private func isFunctionApproval(_ approval: ToolApprovalItem, queuedFunctionCallIds: Set<String>) -> Bool {
        let callId = getToolApprovalItemCallId(approval)
        if let callId, nonFunctionOwnedCallIds.contains(callId), !queuedFunctionCallIds.contains(callId) {
            return false
        }
        if approval.rawItem.type == "function_call" { return true }
        return approval.toolLookupKey != nil || callId.map(queuedFunctionCallIds.contains) == true
    }

    private func validateApprovalIdentity(_ approval: ToolApprovalItem) throws {
        guard let persistedKey = approvalPersistedLookupKey(approval),
              let rawName = approval.rawItem.string("name"), !rawName.isEmpty else { return }
        let rawNamespace = getToolCallNamespace(approval.rawItem)
        let rawKey = getFunctionToolLookupKey(rawName, rawNamespace)
        if persistedKey.name == rawName, rawNamespace == nil || persistedKey == rawKey { return }
        let rawIdentity = getToolCallQualifiedName(approval.rawItem) ?? rawName
        throw ModelBehaviorError(
            "Persisted tool identity \(persistedKey.pythonRepr) does not match raw tool call \(rawIdentity). "
                + "Restore a consistent RunState before resuming."
        )
    }

    /// Rebuilds a well-formed function call from an approval's raw item (upstream `_coerce_approval_call`).
    private func coerceApprovalCall(_ approval: ToolApprovalItem) -> ResponseFunctionToolCall? {
        let raw = approval.rawItem
        guard raw.type == "function_call", let name = raw.string("name"), let callId = raw.string("call_id"), !callId.isEmpty,
              let arguments = raw.string("arguments") else {
            return nil
        }
        var payload: JSONObject = ["type": "function_call", "name": .string(name), "call_id": .string(callId), "arguments": .string(arguments)]
        if let status = raw.string("status"), ["in_progress", "completed", "incomplete"].contains(status) {
            payload["status"] = .string(status)
        }
        if let namespace = getToolCallNamespace(raw) { payload["namespace"] = .string(namespace) }
        if let caller = raw["caller"], !caller.isNull { payload["caller"] = caller }
        if let itemId = raw.string("id") { payload["id"] = .string(itemId) }
        let persistedKey = approvalPersistedLookupKey(approval)
        if getToolCallNamespace(payload) == nil, let persistedKey {
            payload = restoreToolCallRoutingIdentity(payload, lookupKey: persistedKey)
        }
        return ResponseFunctionToolCall(json: payload)
    }

    /// Validates function approvals against the queued calls, returning an interruption that keeps
    /// every pending approval when any is malformed.
    func validateFunctionApprovals() throws -> SingleStepResult? {
        let queuedIds = queuedFunctionCallIds
        let functionApprovals = pendingApprovalItems.filter { approvalMatchesAgent($0) && isFunctionApproval($0, queuedFunctionCallIds: queuedIds) }
        var malformed = false
        var stableApprovals: [ToolApprovalItem] = []
        for approval in functionApprovals {
            try validateApprovalIdentity(approval)
            guard let call = coerceApprovalCall(approval) else {
                malformed = true
                continue
            }
            let persistedKey = approvalPersistedLookupKey(approval)
            let queuedLookupKeys = queuedFunctionLookupKeysByCallId[call.callId] ?? []
            if !queuedLookupKeys.isEmpty, persistedKey.map(queuedLookupKeys.contains) != true {
                malformed = true
                continue
            }
            let queuedCallIds = persistedKey.flatMap { queuedFunctionCallIdsByLookupKey[$0] } ?? []
            if !queuedCallIds.isEmpty, !queuedCallIds.contains(call.callId) {
                malformed = true
                continue
            }
            let stable = ToolApprovalItem(
                agent: agent,
                rawItem: call.json,
                toolName: approval.toolName,
                toolNamespace: approval.toolNamespace,
                toolOrigin: approval.toolOrigin,
                toolLookupKey: approval.toolLookupKey,
                allowBareNameAlias: approval.allowBareNameAlias
            )
            validatedFunctionApprovals[ObjectIdentifier(approval)] = stable
            stableFunctionApprovalSources[ObjectIdentifier(stable)] = approval
            stableApprovals.append(stable)
        }

        if malformed {
            processed.interruptions = pendingApprovalItems
            return SingleStepResult(
                originalInput: originalInput, modelResponse: newResponse, preStepItems: originalPreStepItems,
                newStepItems: [], nextStep: .interruption(pendingApprovalItems),
                toolInputGuardrailResults: [], toolOutputGuardrailResults: [], processedResponse: processed
            )
        }
        functionApprovalItems = stableApprovals
        for stable in stableApprovals {
            if let callId = stable.rawItem.string("call_id") { functionApprovalItemsByCallId[callId] = stable }
        }
        return nil
    }

    /// Before schema 1.7, duplicate-name owners could restore to a sibling. Honors the validated
    /// current decision, including a fresh sticky choice, without transferring its future scope to
    /// the reconciled owner.
    func transferLegacyFunctionDecisions() throws {
        guard allowLegacyNameAgentMatch else { return }
        for original in pendingApprovalItems {
            guard let stable = validatedFunctionApprovals[ObjectIdentifier(original)], original.agent !== stable.agent,
                  let callId = stable.rawItem.string("call_id") else { continue }
            let status = try contextWrapper.getApprovalStatus(toolName: original.toolName ?? "", callId: callId, existingPending: original)
            if status == true {
                try contextWrapper.approveTool(stable)
            } else if status == false {
                try contextWrapper.rejectTool(
                    stable,
                    rejectionMessage: contextWrapper.getRejectionMessage(toolName: original.toolName ?? "", callId: callId, existingPending: original)
                )
            }
        }
    }
}

// MARK: - Custom tool reconciliation

extension InterruptedTurnResolver {
    private func customReconciliationApproval(_ call: ResponseCustomToolCall, _ identity: ToolInvocationIdentity) throws
        -> (approval: ToolApprovalItem, status: Bool?)? {
        for approval in pendingApprovalItems where approvalMatchesAgent(approval) {
            guard approval.rawItem.type == "custom_tool_call", getToolApprovalItemCallId(approval) == call.callId,
                  let approvalName = approval.rawItem.string("name"),
                  toolInvocationIdentity(approval.rawItem, toolName: approvalName) == identity else { continue }
            let status = try contextWrapper.getApprovalStatus(
                toolName: approval.toolName ?? call.name, callId: call.callId,
                toolNamespace: approval.toolNamespace, existingPending: approval
            )
            return (approval, status)
        }
        return nil
    }

    private func appendCustomReconciliationCall(_ rawItem: JSONObject) throws {
        guard rawItem.type == "custom_tool_call" else { return }
        guard let call = ResponseCustomToolCall(json: rawItem), rawItem.string("name") != nil, rawItem.string("input") != nil else {
            throw ModelBehaviorError("Persisted custom tool call is invalid. Start a new run instead of resuming this RunState.")
        }
        guard let callId = call.json.callIdOrId, !callId.isEmpty else {
            throw ModelBehaviorError("Custom tool call is missing call_id.")
        }
        guard let identity = toolInvocationIdentity(call.json, toolName: call.name) else {
            throw ModelBehaviorError("Custom tool call has an invalid invocation identity.")
        }
        if let previous = customCallIdentities[callId] {
            guard previous == identity else {
                throw ModelBehaviorError(
                    "Run state reused a custom tool call ID for different invocations. Start a new run instead of resuming this RunState."
                )
            }
            return
        }
        customCallIdentities[callId] = identity
        if hasOutputItem(callId, "custom_tool_call_output") { return }
        if let decision = try customReconciliationApproval(call, identity) {
            switch decision.status {
            case false?:
                rejectedCustomApprovals.append((callId, decision.approval, call))
                return
            case nil:
                return
            case true?:
                break
            }
        }
        customCallsToReconcile.append(call)
    }

    func collectCustomCallsToReconcile() throws {
        for output in newResponse.output {
            try appendCustomReconciliationCall(output)
        }
        for run in processed.customToolCalls {
            try appendCustomReconciliationCall(run.toolCall.json)
        }
        for approval in pendingApprovalItems where approvalMatchesAgent(approval) {
            try appendCustomReconciliationCall(approval.rawItem)
        }
    }

    func buildRejectedCustomApprovalOutputs() async throws -> [RunItem] {
        var outputs: [RunItem] = []
        for (callId, approval, call) in rejectedCustomApprovals {
            let message = try await resolveApprovalRejectionMessage(
                contextWrapper: contextWrapper, runConfig: runConfig, toolType: "custom",
                toolName: approval.toolName ?? call.name, callId: callId, toolCall: call.json,
                toolNamespace: approval.toolNamespace, existingPending: approval
            )
            var raw: JSONObject = ["type": "custom_tool_call_output", "call_id": .string(callId), "output": .string(message)]
            ItemHelpers.copyToolCallCaller(call.json, into: &raw)
            outputs.append(ToolCallOutputItem(agent: agent, rawItem: raw, output: message))
        }
        return outputs
    }
}

// MARK: - Function and handoff reconciliation

extension InterruptedTurnResolver {
    private func appendReconciliationCall(_ call: ResponseFunctionToolCall) {
        let callId = call.callId
        guard !duplicateQueuedCallIds.contains(callId), !reconciledCallIds.contains(callId),
              !executedHandoffCallIds.contains(callId) else { return }
        // Swift: upstream keeps a call whose output exists while its nested `Agent.asTool()` run is
        // still interrupted; nested agent-tool state is not ported yet.
        if hasOutputItem(callId, "function_call_output") { return }
        var call = call
        if let queued = canonicalQueuedFunctionCalls[callId] {
            call = queued
        } else if let approval = functionApprovalItemsByCallId[callId], let approvalCall = ResponseFunctionToolCall(json: approval.rawItem) {
            call = approvalCall
        }
        reconciledCallIds.insert(callId)
        callsToReconcile.append(call)
    }

    /// Re-classifies every interrupted call against the current tool inventory.
    func classifyReconciledCalls() async throws -> ProcessedResponse {
        let availableHandoffs = try await getHandoffs(agent, contextWrapper)
        let currentToolInventory = try await agent.getAllTools(contextWrapper)
        let (resolvedTools, resolvedHandoffs) = try resolveToolNameCollisions(
            currentToolInventory, handoffs: availableHandoffs, collisionPolicy: runConfig.toolNameCollisionPolicy
        )
        let resolvedFunctionTools = resolvedTools.compactMap { $0 as? FunctionTool }
        let localFunctionToolIds = Set(agent.tools.compactMap { ($0 as? FunctionTool).map(ObjectIdentifier.init) })
        availableFunctionTools = resolvedFunctionTools.filter { localFunctionToolIds.contains(ObjectIdentifier($0)) }

        executedHandoffCallIds = Set(originalPreStepItems.compactMap { ($0 as? HandoffOutputItem)?.rawItem.callIdOrId })
        for call in authoritativeResponseFunctionCalls { appendReconciliationCall(call) }
        for run in processed.functions { appendReconciliationCall(run.toolCall) }
        for run in processed.handoffs { appendReconciliationCall(run.toolCall) }
        for approval in functionApprovalItems {
            if let call = ResponseFunctionToolCall(json: approval.rawItem) { appendReconciliationCall(call) }
        }

        let classifierTools: [any Tool] = resolvedFunctionTools
            + resolvedTools.filter { $0 is CustomTool || $0 is ApplyPatchTool }
        var classifierConfig = runConfig
        classifierConfig.toolNotFoundBehavior = .returnErrorToModel
        let classifierResponse = ModelResponse(
            output: callsToReconcile.map(\.json) + customCallsToReconcile.map(\.json),
            usage: newResponse.usage,
            responseId: newResponse.responseId,
            requestId: newResponse.requestId
        )
        return try processModelResponse(
            agent: agent,
            allTools: classifierTools,
            response: classifierResponse,
            outputSchema: getOutputSchema(agent),
            handoffs: resolvedHandoffs,
            runConfig: classifierConfig,
            allowApplyPatchFunctionFallback: false
        )
    }

    private func recordFunctionApprovalRejection(_ approvalRecord: ToolApprovalItem, callId: String) async throws {
        guard let rejectionCall = ResponseFunctionToolCall(json: approvalRecord.rawItem) else { return }
        let message = try await resolveApprovalRejectionMessage(
            contextWrapper: contextWrapper, runConfig: runConfig, toolType: "function",
            toolName: getToolCallTraceName(rejectionCall.json) ?? rejectionCall.name, callId: callId,
            toolCall: rejectionCall.json, toolNamespace: getToolCallNamespace(rejectionCall.json),
            toolLookupKey: approvalRecord.toolLookupKey, existingPending: approvalRecord
        )
        rejectedFunctionOutputs.append(try functionRejectionItem(
            agent: agent, toolCall: rejectionCall, rejectionMessage: message, toolOrigin: approvalRecord.toolOrigin
        ))
        rejectedFunctionCallIds.insert(callId)
    }

    /// Rebinds each interrupted call to the tool or handoff that currently owns it.
    func reconcileCalls(_ classified: ProcessedResponse) async throws -> ([ToolRunFunction], [ToolRunHandoff]) {
        let staleFunctions = Dictionary(
            processed.functions.filter { !duplicateQueuedCallIds.contains($0.toolCall.callId) }.map { ($0.toolCall.callId, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        let currentFunctions = Dictionary(classified.functions.map { ($0.toolCall.callId, $0) }, uniquingKeysWith: { first, _ in first })
        let currentHandoffs = Dictionary(classified.handoffs.map { ($0.toolCall.callId, $0) }, uniquingKeysWith: { first, _ in first })
        let currentMissing = Dictionary(classified.functionToolsNotFound.map { ($0.toolCall.callId, $0) }, uniquingKeysWith: { first, _ in first })
        processed.customToolCalls = try processed.customToolCalls.filter { hasOutputItem(try customCallId($0), "custom_tool_call_output") }
            + classified.customToolCalls

        var reconciledFunctions = processed.functions.filter { !reconciledCallIds.contains($0.toolCall.callId) }
        var reconciledHandoffs = processed.handoffs.filter { !reconciledCallIds.contains($0.toolCall.callId) }

        for call in callsToReconcile {
            let callId = call.callId
            let approvalRecord = functionApprovalItemsByCallId[callId]
            var approvalStatus: Bool? = true
            if let approvalRecord {
                approvalStatus = try contextWrapper.getApprovalStatus(
                    toolName: approvalRecord.toolName ?? call.name, callId: callId, toolNamespace: approvalRecord.toolNamespace,
                    existingPending: approvalRecord,
                    currentInvocation: ToolApprovalItem(agent: agent, rawItem: call.json, toolName: call.name, toolNamespace: getToolCallNamespace(call.json))
                )
            }
            let staleFunction = staleFunctions[callId]
            let originalBinding = processed.mcpToolBindings[callId]

            if let currentFunction = currentFunctions[callId] {
                let missingMCPBinding = originalBinding == nil && approvalRecord?.toolOrigin?.type == .mcp
                // Rejected calls cannot execute. MCP approvals must not authorize a different MCP
                // recipient or a local replacement's approval policy.
                if approvalStatus != false, missingMCPBinding || originalBinding != MCPToolBinding(currentFunction.functionTool) {
                    throw UserError(
                        "Cannot resume a local MCP tool call with a missing or different recipient binding. "
                            + "Restore the original MCP server configuration and tool listing, or start a new run."
                    )
                }
                reconciledFunctions.append(currentFunction)
                continue
            }

            let currentHandoff = currentHandoffs[callId]
            if let currentHandoff, let approvalRecord {
                approvalStatus = try contextWrapper.getApprovalStatus(
                    toolName: approvalRecord.toolName ?? call.name, callId: callId, toolNamespace: approvalRecord.toolNamespace,
                    existingPending: approvalRecord,
                    currentInvocation: ToolApprovalItem(
                        agent: agent, rawItem: currentHandoff.toolCall.json, toolName: currentHandoff.toolCall.name,
                        toolNamespace: getToolCallNamespace(currentHandoff.toolCall.json)
                    )
                )
            }
            if let currentHandoff, approvalStatus == true {
                if originalBinding != nil {
                    throw UserError("Cannot resume a local MCP tool call as a handoff. Restore the original tool configuration, or start a new run.")
                }
                reconciledHandoffs.append(currentHandoff)
                continue
            }

            if let missing = currentMissing[callId], approvalStatus == true {
                if runConfig.toolNotFoundBehavior != .returnErrorToModel {
                    let qualifiedName = getToolCallQualifiedName(missing.toolCall.json) ?? missing.toolName
                    throw ModelBehaviorError("Tool \(qualifiedName) not found in agent \(agent.name)")
                }
                missingFunctionCallIds.insert(callId)
                missingFunctionTools.append(missing)
                continue
            }

            if let staleFunction {
                reconciledFunctions.append(staleFunction)
            } else if let approvalRecord {
                if approvalStatus == nil {
                    addPendingInterruption(approvalRecord)
                } else if approvalStatus == false {
                    try await recordFunctionApprovalRejection(approvalRecord, callId: callId)
                }
            }
        }
        return (reconciledFunctions, reconciledHandoffs)
    }

    /// Keeps approvals the reconciliation did not resolve as pending interruptions.
    func collectUnreconciledPendingApprovals() throws {
        let collectorOwnedCallIds = Set(
            try processed.shellCalls.map { try extractShellCallId($0.toolCall) }
                + processed.applyPatchCalls.map { try extractApplyPatchCallId($0.toolCall) }
                + processed.customToolCalls.map { try customCallId($0) }
                + processed.mcpApprovalRequests.compactMap { extractMCPRequestId($0.requestItem.json) }
        )
        for original in pendingApprovalItems {
            guard approvalMatchesAgent(original) else {
                // Swift: upstream first routes the approval to a nested `Agent.asTool()` run state;
                // nested agent-tool state is not ported yet, so it stays pending.
                addPendingInterruption(original)
                continue
            }
            guard let snapshot = validatedFunctionApprovals[ObjectIdentifier(original)] else {
                let callId = getToolApprovalItemCallId(original)
                if let callId, collectorOwnedCallIds.contains(callId) { continue }
                let status = try callId.flatMap {
                    try contextWrapper.getApprovalStatus(
                        toolName: original.toolName ?? "", callId: $0, toolNamespace: original.toolNamespace, existingPending: original
                    )
                }
                if status == nil { addPendingInterruption(original) }
                continue
            }
            guard let callId = snapshot.rawItem.string("call_id"), !reconciledCallIds.contains(callId) else { continue }
            let status = try contextWrapper.getApprovalStatus(
                toolName: snapshot.toolName ?? "", callId: callId, toolNamespace: snapshot.toolNamespace, existingPending: snapshot
            )
            if status == nil { addPendingInterruption(snapshot) }
        }
    }
}

// MARK: - Resume plan

/// The approved work of a resumed turn and the rejections found while selecting it.
private struct ResumePlan {
    var plan: ToolExecutionPlan
    var rejectedShellResults: [RunItem]
    var rejectedApplyPatchResults: [RunItem]
    var rejectedCustomToolResults: [RunItem]
}

extension InterruptedTurnResolver {
    /// A resumed call that already has a pending approval, or whose policy fails, requires approval
    /// (upstream `_function_requires_approval`).
    private func functionRequiresApproval(_ run: ToolRunFunction) async throws -> FunctionToolApproval {
        let pending = FunctionToolApproval(.requireApproval, tool: run.functionTool, arguments: run.toolCall.arguments)
        if approvalItemsByCallId[run.toolCall.callId] != nil { return pending }
        do {
            return try await functionNeedsApproval(run.functionTool, contextWrapper, run.toolCall)
        } catch let error as UserError {
            throw error
        } catch {
            return pending
        }
    }

    private func functionOutputExists(_ run: ToolRunFunction) throws -> Bool {
        let callId = run.toolCall.callId
        guard !callId.isEmpty else { return false }
        if approvalItemsByCallId[callId] == nil, hasOutputItem(callId, "function_call_output") { return true }
        let binding = try contextWrapper.approvedToolInvocationStatus(
            run.toolCall.json, toolLookupKey: getFunctionToolLookupKeyForTool(run.functionTool)
        )
        return binding?.completed ?? false
    }

    private func recordFunctionRejection(_ callId: String?, _ toolCall: ResponseFunctionToolCall, _ functionTool: FunctionTool) async throws {
        if let callId, rejectedFunctionCallIds.contains(callId) { return }
        let approvalRecord = callId.flatMap { functionApprovalItemsByCallId[$0] ?? approvalItemsByCallId[$0] }
        var message = rejectionMessage
        if let callId {
            let toolNamespace = getToolCallNamespace(toolCall.json)
            let lookupKey = getFunctionToolLookupKeyForTool(functionTool)
            message = try await resolveApprovalRejectionMessage(
                contextWrapper: contextWrapper, runConfig: runConfig, toolType: "function",
                toolName: getToolCallTraceName(toolCall.json) ?? functionTool.name, callId: callId, toolCall: toolCall.json,
                toolNamespace: toolNamespace, toolLookupKey: lookupKey,
                existingPending: approvalRecord ?? ToolApprovalItem(
                    agent: agent, rawItem: toolCall.json, toolName: functionTool.name, toolNamespace: toolNamespace, toolLookupKey: lookupKey
                )
            )
        }
        rejectedFunctionOutputs.append(try functionRejectionItem(
            agent: agent, toolCall: toolCall, rejectionMessage: message, outputJSONSchema: functionTool.outputJSONSchema,
            toolOrigin: approvalRecord.map(\.toolOrigin) ?? getFunctionToolOrigin(functionTool)
        ))
        if let callId { rejectedFunctionCallIds.insert(callId) }
    }

    private func pendingFunctionItem(_ run: ToolRunFunction) -> ToolApprovalItem {
        ToolApprovalItem(
            agent: agent,
            rawItem: run.toolCall.json,
            toolName: run.functionTool.name,
            toolNamespace: getToolCallNamespace(run.toolCall.json),
            toolOrigin: getFunctionToolOrigin(run.functionTool),
            toolLookupKey: getFunctionToolLookupKeyForCall(run.toolCall.json),
            allowBareNameAlias: shouldAllowBareNameApprovalAlias(run.functionTool, allTools: availableFunctionTools)
        )
    }

    private func applyPatchNeedsApproval(_ run: ToolRunApplyPatchCall) async throws -> Bool {
        let operations = try coerceApplyPatchOperations(run.toolCall, contextWrapper: contextWrapper)
        let callId = try extractApplyPatchCallId(run.toolCall)
        for operation in operations {
            let needsApproval: Bool
            switch run.applyPatchTool.needsApproval {
            case .required(let value): needsApproval = value
            case .dynamic(let function): needsApproval = try await function(contextWrapper, operation, callId)
            }
            let status = try contextWrapper.getApprovalStatus(
                toolName: run.applyPatchTool.name, callId: callId, existingPending: approvalItemsByCallId[callId]
            )
            if status != nil || needsApproval { return needsApproval }
        }
        return false
    }

    private func nativeRejectionMessage(toolType: String, toolName: String, callId: String, toolCall: JSONObject) async throws -> String {
        try await resolveApprovalRejectionMessage(
            contextWrapper: contextWrapper, runConfig: runConfig, toolType: toolType, toolName: toolName, callId: callId,
            toolCall: toolCall, existingPending: ToolApprovalItem(agent: agent, rawItem: toolCall, toolName: toolName)
        )
    }

    private func collectNonFunctionRuns() async throws -> (
        shell: (approved: [ToolRunShellCall], rejections: [RunItem]),
        applyPatch: (approved: [ToolRunApplyPatchCall], rejections: [RunItem]),
        custom: (approved: [ToolRunCustom], rejections: [RunItem])
    ) {
        let shell = try await collectRunsByApproval(
            processed.shellCalls,
            callIdExtractor: { try extractShellCallId($0.toolCall) },
            toolNameResolver: { $0.shellTool.name },
            rejectionBuilder: { run, callId in
                let message = try await self.nativeRejectionMessage(toolType: "shell", toolName: run.shellTool.name, callId: callId, toolCall: run.toolCall)
                return shellRejectionItem(agent: self.agent, callId: callId, toolCall: run.toolCall, rejectionMessage: message)
            },
            contextWrapper: contextWrapper, approvalItemsByCallId: approvalItemsByCallId, agent: agent,
            pendingInterruptionAdder: addPendingInterruption,
            needsApprovalChecker: { run in
                let shellCall = try coerceShellCall(run.toolCall)
                switch run.shellTool.needsApproval {
                case .required(let value): return value
                case .dynamic(let function): return try await function(self.contextWrapper, shellCall.action, shellCall.callId)
                }
            },
            outputExistsChecker: { self.hasOutputItem($0, "shell_call_output") }
        )
        let applyPatch = try await collectRunsByApproval(
            processed.applyPatchCalls,
            callIdExtractor: { try extractApplyPatchCallId($0.toolCall) },
            toolNameResolver: { $0.applyPatchTool.name },
            rejectionBuilder: { run, callId in
                let message = try await self.nativeRejectionMessage(
                    toolType: "apply_patch", toolName: run.applyPatchTool.name, callId: callId, toolCall: run.toolCall
                )
                return applyPatchRejectionItem(agent: self.agent, callId: callId, toolCall: run.toolCall, rejectionMessage: message)
            },
            contextWrapper: contextWrapper, approvalItemsByCallId: approvalItemsByCallId, agent: agent,
            pendingInterruptionAdder: addPendingInterruption,
            needsApprovalChecker: applyPatchNeedsApproval,
            outputExistsChecker: { self.hasOutputItem($0, "apply_patch_call_output") }
        )
        let custom = try await collectRunsByApproval(
            processed.customToolCalls,
            callIdExtractor: customCallId,
            toolNameResolver: { $0.customTool.name },
            rejectionBuilder: { run, callId in
                let message = try await self.nativeRejectionMessage(toolType: "custom", toolName: run.customTool.name, callId: callId, toolCall: run.toolCall.json)
                var raw: JSONObject = ["type": "custom_tool_call_output", "call_id": .string(callId), "output": .string(message)]
                ItemHelpers.copyToolCallCaller(run.toolCall.json, into: &raw)
                return ToolCallOutputItem(agent: self.agent, rawItem: raw, output: message)
            },
            contextWrapper: contextWrapper, approvalItemsByCallId: approvalItemsByCallId, agent: agent,
            pendingInterruptionAdder: addPendingInterruption,
            needsApprovalChecker: { run in
                let callId = try customCallId(run)
                switch run.customTool.needsApproval {
                case .required(let value): return value
                case .dynamic(let function): return try await function(self.contextWrapper, run.toolCall.input, callId)
                }
            },
            outputExistsChecker: { self.hasOutputItem($0, "custom_tool_call_output") }
        )
        return (shell, applyPatch, custom)
    }

    fileprivate func buildResumePlan(_ selectableFunctionRuns: [ToolRunFunction]) async throws -> ResumePlan {
        // Validate every current execution candidate before any dynamic approval callback or
        // output-based replay suppression can run. This keeps a changed invocation under an approved
        // call ID from triggering sibling user code or hiding behind a previously committed output.
        for run in selectableFunctionRuns {
            try contextWrapper.approvedToolInvocationStatus(run.toolCall.json, toolLookupKey: getFunctionToolLookupKeyForTool(run.functionTool))
        }
        for run in processed.handoffs {
            try contextWrapper.approvedToolInvocationStatus(run.toolCall.json, invocationRole: "handoff")
        }
        try validateUnresolvedFunctionCalls(contextWrapper, missingFunctionTools)
        let functionRuns = try await selectFunctionToolRunsForResume(
            selectableFunctionRuns,
            approvalItemsByCallId: functionApprovalItemsByCallId,
            contextWrapper: contextWrapper,
            needsApprovalChecker: functionRequiresApproval,
            outputExistsChecker: functionOutputExists,
            recordRejection: recordFunctionRejection,
            pendingInterruptionAdder: addPendingInterruption,
            pendingItemBuilder: pendingFunctionItem
        )
        let computerActions = try processed.computerActions.filter { !hasOutputItem(try computerCallId($0), "computer_call_output") }
        let (shell, applyPatch, custom) = try await collectNonFunctionRuns()

        let (withCallback, manual) = partitionMCPApprovalRequests(processed.mcpApprovalRequests)
        let (approvedMCP, pendingMCP) = try collectManualMCPApprovals(
            agent: agent, requests: manual, contextWrapper: contextWrapper, existingPendingByCallId: approvalItemsByCallId
        )
        pendingMCP.forEach(addPendingInterruption)
        let plan = ToolExecutionPlan(
            functionRuns: functionRuns,
            computerActions: computerActions,
            customToolCalls: custom.approved,
            shellCalls: shell.approved,
            applyPatchCalls: applyPatch.approved,
            localShellCalls: [],
            approvedMCPResponses: approvedMCP,
            mcpRequestsWithCallback: withCallback
        )
        return ResumePlan(
            plan: plan,
            rejectedShellResults: shell.rejections,
            rejectedApplyPatchResults: applyPatch.rejections,
            rejectedCustomToolResults: custom.rejections
        )
    }
}

// MARK: - Execution

/// Collects new step items, skipping items already present by identity (upstream
/// `_make_unique_item_appender`).
private struct UniqueItemAppender {
    private var seen: Set<ObjectIdentifier>
    private(set) var items: [RunItem] = []

    init(existing: [RunItem]) {
        seen = Set(existing.map(ObjectIdentifier.init))
    }

    mutating func append(_ item: RunItem) {
        guard seen.insert(ObjectIdentifier(item)).inserted else { return }
        items.append(item)
    }
}

extension InterruptedTurnResolver {
    private func callPositions() -> [String: Int] {
        var positions = responseCallPositions
        var next = newResponse.output.count
        for call in callsToReconcile where positions[call.callId] == nil {
            positions[call.callId] = next
            next += 1
        }
        return positions
    }

    fileprivate func executeAndFinish(
        plan resumePlan: ResumePlan,
        selectableFunctionRuns: [ToolRunFunction],
        missingOutputItems: [RunItem],
        rejectedCustomApprovalOutputs: [RunItem]
    ) async throws -> SingleStepResult {
        let positions = callPositions()
        func position(_ item: RunItem) -> Int { item.rawItem.callIdOrId.flatMap { positions[$0] } ?? positions.count }
        let checkpoint = ResumeCheckpoint(
            runState: runState, originalPreStepItems: originalPreStepItems, callPositions: positions, contextWrapper: contextWrapper
        )
        let results = try await executeToolPlan(
            plan: resumePlan.plan, agent: agent, allTools: allTools, hooks: hooks, contextWrapper: contextWrapper,
            runConfig: runConfig, toolOutputCommitter: checkpoint.commitToolOutput
        )
        results.interruptions.forEach(addPendingInterruption)

        var appender = UniqueItemAppender(existing: originalPreStepItems)
        let functionOutcomes = (results.functionResults.compactMap(\.runItem) + missingOutputItems + rejectedFunctionOutputs)
            .enumerated()
            .sorted { (position($0.element), $0.offset) < (position($1.element), $1.offset) }
            .map(\.element)
        functionOutcomes.forEach { appender.append($0) }
        (results.computerResults + results.customToolResults + results.shellResults + results.applyPatchResults).forEach { appender.append($0) }
        pendingInterruptions.forEach { appender.append($0) }
        resumePlan.rejectedShellResults.forEach { appender.append($0) }
        resumePlan.rejectedCustomToolResults.forEach { appender.append($0) }
        rejectedCustomApprovalOutputs.forEach { appender.append($0) }
        resumePlan.rejectedApplyPatchResults.forEach { appender.append($0) }
        resumePlan.plan.approvedMCPResponses.forEach { appender.append($0) }
        try checkpoint.checkpoint(appender.items)

        func commitMissingState(_ result: SingleStepResult) throws -> SingleStepResult {
            try checkpoint.checkpoint(result.newStepItems)
            if !missingFunctionCallIds.isEmpty {
                processed.functions = processed.functions.filter { !missingFunctionCallIds.contains($0.toolCall.callId) }
            }
            return result
        }

        processed.interruptions = pendingInterruptions
        if !pendingInterruptions.isEmpty {
            return try commitMissingState(SingleStepResult(
                originalInput: originalInput, modelResponse: newResponse, preStepItems: originalPreStepItems,
                newStepItems: appender.items, nextStep: .interruption(pendingInterruptions),
                toolInputGuardrailResults: results.toolInputGuardrailResults,
                toolOutputGuardrailResults: results.toolOutputGuardrailResults,
                processedResponse: processed
            ))
        }

        try await executeMCPApprovalRequests(
            agent: agent, requests: resumePlan.plan.mcpRequestsWithCallback, contextWrapper: contextWrapper
        ).forEach { appender.append($0) }
        try checkpoint.checkpoint(appender.items)
        let (pendingHostedMCPApprovals, pendingHostedMCPApprovalIds) = try processHostedMCPApprovals(
            originalPreStepItems: originalPreStepItems, mcpApprovalRequests: processed.mcpApprovalRequests,
            contextWrapper: contextWrapper, agent: agent, appendItem: { appender.append($0) }
        )
        var preStepItems = originalPreStepItems.filter {
            shouldKeepHostedMCPItem(
                $0, pendingHostedMCPApprovals: pendingHostedMCPApprovals, pendingHostedMCPApprovalIds: pendingHostedMCPApprovalIds
            )
        }
        if !rejectedFunctionCallIds.isEmpty {
            preStepItems = preStepItems.filter {
                !($0 is ToolCallOutputItem && $0.rawItem.callIdOrId.map(rejectedFunctionCallIds.contains) == true)
            }
        }
        let newItems = appender.items

        let pendingHandoffs = processed.handoffs.filter {
            $0.toolCall.callId.isEmpty || !executedHandoffCallIds.contains($0.toolCall.callId)
        }
        if !pendingHandoffs.isEmpty {
            let runState = runState
            return try commitMissingState(try await executeHandoffs(
                agent: agent, originalInput: originalInput, preStepItems: preStepItems, newStepItems: newItems,
                newResponse: newResponse, runHandoffs: pendingHandoffs, hooks: hooks, contextWrapper: contextWrapper,
                runConfig: runConfig, serverManagesConversation: serverManagesConversation,
                toolInputGuardrailResults: results.toolInputGuardrailResults,
                toolOutputGuardrailResults: results.toolOutputGuardrailResults,
                handoffOutputCommitter: { _, newAgent, stepItems in
                    try checkpoint.checkpoint(stepItems)
                    runState?.currentAgent = newAgent
                }
            ))
        }

        // Completed siblings are deliberately skipped during resumed execution, but their results
        // still participate in the interrupted turn's final-output decision.
        var resultsByCallId: [String: FunctionToolResult] = [:]
        for result in results.functionResults {
            if let callId = result.runItem?.rawItem.callIdOrId { resultsByCallId[callId] = result }
        }
        var completedOutputs: [String: ToolCallOutputItem] = [:]
        for case let item as ToolCallOutputItem in originalPreStepItems + newItems where item.rawItem.type == "function_call_output" {
            if let callId = item.callId { completedOutputs[callId] = item }
        }
        let turnFunctionResults: [FunctionToolResult] = selectableFunctionRuns
            .enumerated()
            .sorted { (positions[$0.element.toolCall.callId] ?? positions.count, $0.offset) < (positions[$1.element.toolCall.callId] ?? positions.count, $1.offset) }
            .compactMap { _, run in
                if let result = resultsByCallId[run.toolCall.callId] { return result }
                guard let output = completedOutputs[run.toolCall.callId] else { return nil }
                return FunctionToolResult(tool: run.functionTool, output: output.output, runItem: output)
            }

        let toolUse = try await checkForFinalOutputFromTools(agent, turnFunctionResults, contextWrapper)
        if toolUse.isFinalOutput {
            var finalOutput = toolUse.finalOutput
            if getOutputSchema(agent) == nil { finalOutput = finalOutput.map { stringify($0) } }
            return try commitMissingState(try await executeFinalOutput(
                agent: agent, originalInput: originalInput, newResponse: newResponse, preStepItems: preStepItems,
                newStepItems: newItems, finalOutput: finalOutput ?? Optional<String>.none as any Sendable, hooks: hooks,
                contextWrapper: contextWrapper, toolInputGuardrailResults: results.toolInputGuardrailResults,
                toolOutputGuardrailResults: results.toolOutputGuardrailResults
            ))
        }

        return try commitMissingState(SingleStepResult(
            originalInput: originalInput, modelResponse: newResponse, preStepItems: preStepItems,
            newStepItems: newItems, nextStep: .runAgain,
            toolInputGuardrailResults: results.toolInputGuardrailResults,
            toolOutputGuardrailResults: results.toolOutputGuardrailResults
        ))
    }
}

/// Keeps only hosted MCP approvals that match pending requests from the provider.
func shouldKeepHostedMCPItem(
    _ item: RunItem,
    pendingHostedMCPApprovals: [ToolApprovalItem],
    pendingHostedMCPApprovalIds: Set<String>
) -> Bool {
    guard let approval = item as? ToolApprovalItem else { return true }
    guard getHostedMCPApprovalRequestIdentity(approval.rawItem) != nil else { return false }
    let requestId = extractMCPRequestId(approval.rawItem)
    return pendingHostedMCPApprovals.contains { $0 === approval } || requestId.map(pendingHostedMCPApprovalIds.contains) == true
}

extension FunctionToolLookupKey {
    /// The Python `repr` of upstream's tuple key, used in error messages.
    fileprivate var pythonRepr: String {
        switch self {
        case .bare(let name): return "('bare', '\(name)')"
        case .namespaced(let namespace, let name): return "('namespaced', '\(namespace)', '\(name)')"
        case .deferredTopLevel(let name): return "('deferred_top_level', '\(name)')"
        }
    }
}
