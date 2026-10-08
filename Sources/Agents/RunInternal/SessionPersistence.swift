import Foundation

// Port of src/agents/run_internal/session_persistence.py.
//
// Swift: pending-write checkpoints (`resume_pending_session_write`), compaction generations,
// OpenAI Conversations session sanitisation, and nested-history provenance are not ported; see
// porting/manifest.json. Python compares callback output by object identity first; Swift values
// have no identity, so new-turn items are recovered by content frequency only.

/// A stable representation of a session item for comparison.
private func sessionItemKey(_ item: TResponseInputItem) -> String {
    JSONValue.object(stripInternalInputItemMetadata(item)).jsonString(sortedKeys: true)
}

private func frequencyMap(_ items: [TResponseInputItem]) -> [String: Int] {
    var counts: [String: Int] = [:]
    for item in items { counts[sessionItemKey(item), default: 0] += 1 }
    return counts
}

/// Prepares model input from session history plus the new turn input.
///
/// Returns the prepared input for the model and the subset of items to append to the session.
func prepareInputWithSession(
    _ input: RunInput,
    session: (any Session)?,
    sessionInputCallback: SessionInputCallback?,
    sessionSettings: SessionSettings? = nil,
    includeHistoryInPreparedInput: Bool = true,
    preserveDroppedNewItems: Bool = false,
    reasoningItemIdPolicy: ReasoningItemIdPolicy? = nil
) async throws -> (RunInput, [TResponseInputItem]) {
    guard let session else { return (input, []) }
    let resolvedSettings = (session.sessionSettings ?? SessionSettings()).resolve(sessionSettings)
    let history = try await session.getItems(limit: resolvedSettings.limit)
    let convertedHistory = applyReasoningItemIdPolicy(history.map(stripInternalInputItemMetadata), reasoningItemIdPolicy)
    let newInputList = ItemHelpers.inputToNewInputList(input)

    var pruneHistoryIndexes: Set<Int> = []
    var outputPruningIndexes: Set<Int>?
    let preparedRaw: [TResponseInputItem]
    var appendedItems: [TResponseInputItem]

    // Without history in the prepared input, the callback is not consulted (as upstream).
    if let sessionInputCallback, includeHistoryInPreparedInput {
        let combined = try await sessionInputCallback(convertedHistory, newInputList)
        var historyCounts = frequencyMap(convertedHistory)
        var newCounts = frequencyMap(newInputList)
        appendedItems = []
        // Swift: upstream matches by object identity first; values have none, so this applies
        // upstream's content fallback (history before new input). See porting/decisions.md D11.
        for (index, item) in combined.enumerated() {
            let key = sessionItemKey(item)
            if let count = historyCounts[key], count > 0 {
                historyCounts[key] = count - 1
                pruneHistoryIndexes.insert(index)
            } else if let count = newCounts[key], count > 0 {
                newCounts[key] = count - 1
                appendedItems.append(item)
            } else {
                appendedItems.append(item)
            }
        }
        preparedRaw = combined
    } else {
        preparedRaw = includeHistoryInPreparedInput ? convertedHistory + newInputList : newInputList
        appendedItems = newInputList
        if includeHistoryInPreparedInput {
            pruneHistoryIndexes = Set(convertedHistory.indices)
            if resolvedSettings.limit != nil {
                outputPruningIndexes = pruneHistoryIndexes
            }
        }
    }

    let filtered = dropOrphanFunctionCalls(preparedRaw, pruningIndexes: pruneHistoryIndexes, outputPruningIndexes: outputPruningIndexes)
    let prepared = deduplicateInputItemsPreferringLatest(normalizeInputItemsForAPI(filtered))
    return (.items(prepared), normalizeInputItemsForAPI(appendedItems))
}

/// Persists input items when an input guardrail tripwire is triggered.
func persistSessionItemsForGuardrailTrip(
    session: (any Session)?,
    serverManagesConversation: Bool,
    sessionInputItemsForPersistence: [TResponseInputItem]?,
    originalUserInput: RunInput?,
    runState: RunState?,
    store: Bool? = nil
) async throws -> [TResponseInputItem]? {
    guard let session, !serverManagesConversation else { return sessionInputItemsForPersistence }
    var updated = sessionInputItemsForPersistence
    if updated == nil, let originalUserInput {
        updated = ItemHelpers.inputToNewInputList(originalUserInput)
    }
    try await saveResultToSession(session, originalInput: .items(updated ?? []), newItems: [], runState: runState, store: store)
    return updated
}

/// Returns the items to persist for a turn, preferring `sessionStepItems` when set.
func sessionItemsForTurn(_ turnResult: SingleStepResult) -> [RunItem] {
    turnResult.sessionStepItems ?? turnResult.newStepItems
}

/// Returns generated and session items for a resumed turn.
func resumedTurnItems(_ turnResult: SingleStepResult) -> ([RunItem], [RunItem]) {
    (turnResult.preStepItems + turnResult.newStepItems, sessionItemsForTurn(turnResult))
}

private func applyPostWriteCompaction(
    _ session: any Session,
    responseId: String?,
    store: Bool?,
    hasLocalToolOutputs: Bool
) async throws {
    guard let responseId, !responseId.isEmpty, let compactionSession = session as? any OpenAIResponsesCompactionAwareSession else {
        return
    }
    if hasLocalToolOutputs {
        logger.debug("skip: deferring compaction for response \(responseId) due to local tool outputs")
        return
    }
    try await compactionSession.runCompaction(OpenAIResponsesCompactionArgs(responseId: responseId, store: store, force: false))
}

/// Persists a turn to the session, tracking what was already saved so retries do not duplicate
/// inputs or tool outputs. Returns the number of new run items persisted.
@discardableResult
func saveResultToSession(
    _ session: (any Session)?,
    originalInput: RunInput,
    newItems: [RunItem],
    runState: RunState? = nil,
    responseId: String? = nil,
    reasoningItemIdPolicy: ReasoningItemIdPolicy? = nil,
    store: Bool? = nil
) async throws -> Int {
    let alreadyPersisted = runState?.currentTurnPersistedItemCount ?? 0
    guard let session else { return 0 }

    var newRunItems: [RunItem] = alreadyPersisted >= newItems.count ? [] : Array(newItems[alreadyPersisted...])
    if runState != nil, !newItems.isEmpty, !newRunItems.isEmpty {
        let pending = Set(newRunItems.map { ObjectIdentifier($0) })
        let missingOutputs = newItems.filter { $0 is ToolCallOutputItem && !pending.contains(ObjectIdentifier($0)) }
        if !missingOutputs.isEmpty {
            newRunItems = missingOutputs + newRunItems
        }
    }

    var inputList: [TResponseInputItem] = []
    if !originalInput.isEmpty {
        inputList = normalizeInputItemsForAPI(ItemHelpers.inputToNewInputList(originalInput))
    }
    let policy = reasoningItemIdPolicy ?? runState?.reasoningItemIdPolicy
    var newItemsAsInput: [TResponseInputItem] = []
    for item in newRunItems {
        if let converted = try runItemToInputItem(item, reasoningItemIdPolicy: policy) {
            newItemsAsInput.append(converted)
        }
    }
    let serializedNewItems = newItemsAsInput.map { fingerprintInputItem($0) }
    let itemsToSave = deduplicateInputItemsPreferringLatest(inputList + newItemsAsInput)
    var saveCounts: [String: Int] = [:]
    for item in itemsToSave { saveCounts[fingerprintInputItem(item), default: 0] += 1 }
    var savedRunItemsCount = 0
    for serialized in serializedNewItems {
        if let count = saveCounts[serialized], count > 0 {
            saveCounts[serialized] = count - 1
            savedRunItemsCount += 1
        }
    }

    guard !itemsToSave.isEmpty else {
        runState?.currentTurnPersistedItemCount = alreadyPersisted + savedRunItemsCount
        return savedRunItemsCount
    }
    let hasLocalToolOutputs = newItems.contains { $0 is ToolCallOutputItem || $0 is HandoffOutputItem }
    try await session.addItems(itemsToSave)
    runState?.currentTurnPersistedItemCount = alreadyPersisted + savedRunItemsCount
    try await applyPostWriteCompaction(session, responseId: responseId, store: store, hasLocalToolOutputs: hasLocalToolOutputs)
    return savedRunItemsCount
}

/// Saves resumed-turn items, skipping those already persisted. Returns the updated persisted count.
func saveResumedTurnItems(
    runState: RunState,
    session: (any Session)?,
    items: [RunItem],
    persistedCount: Int,
    responseId: String?,
    reasoningItemIdPolicy: ReasoningItemIdPolicy?,
    store: Bool?
) async throws -> Int {
    guard session != nil, !items.isEmpty else { return persistedCount }
    let saved = try await saveResultToSession(
        session, originalInput: .items([]), newItems: Array(items.dropFirst(min(persistedCount, items.count))),
        runState: nil, responseId: responseId, reasoningItemIdPolicy: reasoningItemIdPolicy, store: store
    )
    return persistedCount + saved
}

/// Best-effort rollback of items recently persisted to a session before a conversation retry, so
/// conversation-lock retries do not accumulate duplicate inputs.
func rewindSessionItems(_ session: (any Session)?, _ items: [TResponseInputItem], _ serverTracker: OpenAIServerConversationTracker? = nil) async {
    guard let session, !items.isEmpty else { return }
    let ignoreIds = session.ignoreIdsForMatching
    let targets = items.map { fingerprintInputItem($0, ignoreIdsForMatching: ignoreIds) }
    logger.debug("Rewinding session items due to conversation retry (targets=\(targets.count))")
    if !(AgentsDebug.dontLogModelData || AgentsDebug.dontLogToolData) {
        for (index, target) in targets.enumerated() {
            logger.debug("Rewind target \(index) (first 300 chars): \(target.prefix(300))")
        }
    }
    let rewound = await rewindSessionTailSuffix(
        session: session,
        expected: targets,
        ignoreIds: ignoreIds,
        mismatchWarning: "Skipping session rewind because the current tail does not match the retry-owned suffix",
        popFailureWarning: "Failed to rewind session item"
    )
    guard rewound else { return }
    await waitForSessionCleanup(session, targets, ignoreIds: ignoreIds)
    guard let serverTracker else { return }

    let latest: [TResponseInputItem]
    do {
        latest = try await session.getItems(limit: 1)
    } catch {
        logActionError("Failed to peek session items while rewinding", error, isToolAction: false)
        return
    }
    guard let latestItem = latest.first else { return }
    if let id = latestItem.string("id"), serverTracker.serverItemIds.contains(id) { return }
    let sessionItems: [TResponseInputItem]
    do {
        sessionItems = try await session.getItems(limit: nil)
    } catch {
        logActionError("Failed to inspect session tail while stripping stray items", error, isToolAction: false)
        return
    }
    let stray = collectRetryOwnedTailSerializations(sessionItems, serverTracker: serverTracker, ignoreIds: ignoreIds)
    guard !stray.isEmpty else { return }
    logger.debug("Stripping \(stray.count) retry-owned conversation items until the session tail reaches a known server item")
    _ = await rewindSessionTailSuffix(
        session: session,
        expected: stray,
        ignoreIds: ignoreIds,
        mismatchWarning: "Skipping stray session cleanup because the current tail no longer matches retry-owned conversation items",
        popFailureWarning: "Failed to strip stray session item"
    )
}

/// Confirms rewound items no longer appear in the session tail before the next retry.
func waitForSessionCleanup(_ session: (any Session)?, _ targets: [String], maxAttempts: Int = 5, ignoreIds: Bool = false) async {
    guard let session, !targets.isEmpty else { return }
    let window = targets.count + 2
    for attempt in 0..<maxAttempts {
        do {
            let tail = Set(try await session.getItems(limit: window).map { fingerprintInputItem($0, ignoreIdsForMatching: ignoreIds) })
            if !targets.contains(where: tail.contains) { return }
        } catch {
            logger.debug("Failed to verify session cleanup (attempt \(attempt + 1)): \(errorTypeName(error))")
        }
        try? await Task.sleep(nanoseconds: UInt64(0.1 * Double(attempt + 1) * 1_000_000_000))
    }
    logger.debug("Session cleanup verification exhausted attempts; targets may still linger temporarily")
}

/// Removes an exact serialized suffix from the session tail, aborting when the tail diverges.
private func rewindSessionTailSuffix(
    session: any Session,
    expected: [String],
    ignoreIds: Bool,
    mismatchWarning: String,
    popFailureWarning: String
) async -> Bool {
    guard !expected.isEmpty else { return true }
    let tail: [TResponseInputItem]
    do {
        tail = try await session.getItems(limit: expected.count)
    } catch {
        logActionWarning(popFailureWarning, error)
        return false
    }
    guard tail.count == expected.count, tail.map({ fingerprintInputItem($0, ignoreIdsForMatching: ignoreIds) }) == expected else {
        logger.warning(mismatchWarning)
        return false
    }
    var popped: [TResponseInputItem] = []
    for target in expected.reversed() {
        let item: TResponseInputItem?
        do {
            item = try await session.popItem()
        } catch {
            await restorePoppedSessionItems(session, popped)
            logActionWarning(popFailureWarning, error)
            return false
        }
        guard let item else {
            await restorePoppedSessionItems(session, popped)
            logger.warning(mismatchWarning)
            return false
        }
        popped.append(item)
        if fingerprintInputItem(item, ignoreIdsForMatching: ignoreIds) != target {
            await restorePoppedSessionItems(session, popped)
            logger.warning(mismatchWarning)
            return false
        }
    }
    return true
}

private func restorePoppedSessionItems(_ session: any Session, _ popped: [TResponseInputItem]) async {
    guard !popped.isEmpty else { return }
    do {
        try await session.addItems(popped.reversed())
    } catch {
        logActionWarning("Failed to restore session items after a rewind mismatch", error)
    }
}

/// Returns the contiguous retry-owned tail suffix that can be safely stripped.
func collectRetryOwnedTailSerializations(
    _ sessionItems: [TResponseInputItem],
    serverTracker: OpenAIServerConversationTracker,
    ignoreIds: Bool
) -> [String] {
    var strayTail: [String] = []
    for item in sessionItems.reversed() {
        if let id = item.string("id"), serverTracker.serverItemIds.contains(id) {
            return strayTail.reversed()
        }
        let serialized = fingerprintInputItem(item, ignoreIdsForMatching: ignoreIds)
        if serverTracker.sentItemFingerprints.contains(serialized) {
            strayTail.append(serialized)
            continue
        }
        logger.warning("Skipping stray session cleanup because the current tail contains items unrelated to this retry")
        return []
    }
    if !strayTail.isEmpty {
        logger.warning("Skipping stray session cleanup because no known server item was found before the session boundary")
    }
    return []
}

/// Raises when a session is combined with server-managed conversation settings.
func validateSessionConversationSettings(
    _ session: (any Session)?,
    conversationId: String?,
    previousResponseId: String?,
    autoPreviousResponseId: Bool
) throws {
    guard session != nil else { return }
    guard conversationId != nil || previousResponseId != nil || autoPreviousResponseId else { return }
    throw UserError("Session persistence cannot be combined with conversation_id, previous_response_id, or auto_previous_response_id.")
}

/// Returns whether any input guardrail in `results` tripped.
func inputGuardrailsTriggered(_ results: [InputGuardrailResult]) -> Bool {
    results.contains { $0.output.tripwireTriggered }
}
