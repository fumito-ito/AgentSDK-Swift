import Foundation

// Port of src/agents/run_internal/oai_conversation.py.
//
// Swift: Python tracks delivered inputs by object identity. Input items here are values, so the
// tracker records the identities of the `RunItem`s that produced each prepared input, falling
// back to content fingerprints. State hydration, pending-input admission lineage, and anonymous
// tool_search restoration are not ported; see porting/manifest.json.

private func normalizeServerItemId(_ value: JSONValue?) -> String? {
    guard let id = value?.stringValue, id != fakeResponsesId else { return nil }
    return id
}

private func trackerFingerprint(_ item: JSONObject) -> String {
    var payload = item
    if payload["status"]?.isNull == true { payload.removeValue(forKey: "status") }
    return fingerprintInputItem(payload)
}

/// Tracks server-managed conversation state (conversation IDs or response chaining) so each
/// turn only sends new deltas.
final class OpenAIServerConversationTracker: @unchecked Sendable {
    let conversationId: String?
    private(set) var previousResponseId: String?
    let autoPreviousResponseId: Bool
    let reasoningItemIdPolicy: ReasoningItemIdPolicy?

    private(set) var sentInitialInput = false
    private var remainingInitialInput: [TResponseInputItem]?
    var serverItemIds: Set<String> = []
    private var serverToolCallIds: Set<String> = []
    private var serverItemFingerprints: Set<String> = []
    var sentItemFingerprints: Set<String> = []
    private var sentRunItems: Set<ObjectIdentifier> = []
    /// Prepared input fingerprint → source run items, in preparation order.
    private var preparedSources: [String: [ObjectIdentifier]] = [:]

    init(
        conversationId: String? = nil,
        previousResponseId: String? = nil,
        autoPreviousResponseId: Bool = false,
        reasoningItemIdPolicy: ReasoningItemIdPolicy? = nil
    ) {
        self.conversationId = conversationId
        self.previousResponseId = previousResponseId
        self.autoPreviousResponseId = autoPreviousResponseId
        self.reasoningItemIdPolicy = reasoningItemIdPolicy
    }

    /// Primes the tracker from a resumed run so already-sent items are not resent.
    func hydrateFromState(originalInput: RunInput, generatedItems: [RunItem], modelResponses: [ModelResponse]) {
        sentInitialInput = true
        remainingInitialInput = nil
        for response in modelResponses {
            trackServerItems(response)
        }
        for item in generatedItems {
            sentRunItems.insert(ObjectIdentifier(item))
        }
    }

    /// Tracks server-acknowledged outputs to avoid re-sending them on retries.
    func trackServerItems(_ modelResponse: ModelResponse?) {
        guard let modelResponse else { return }
        var fingerprints: Set<String> = []
        for output in modelResponse.output {
            if let id = normalizeServerItemId(output["id"]) { serverItemIds.insert(id) }
            if let callId = output.string("call_id"), output.contains("output") {
                serverToolCallIds.insert(callId)
            }
            let fingerprint = trackerFingerprint(output)
            sentItemFingerprints.insert(fingerprint)
            serverItemFingerprints.insert(fingerprint)
            fingerprints.insert(fingerprint)
        }
        if let remaining = remainingInitialInput, !fingerprints.isEmpty {
            let filtered = remaining.filter { !fingerprints.contains(trackerFingerprint($0)) }
            remainingInitialInput = filtered.isEmpty ? nil : filtered
        }
        if conversationId == nil, previousResponseId != nil || autoPreviousResponseId, let responseId = modelResponse.responseId {
            previousResponseId = responseId
        }
    }

    /// Marks delivered inputs so they are not sent again after pauses or retries.
    func markInputAsSent(_ items: [TResponseInputItem]) {
        guard !items.isEmpty else { return }
        var delivered: Set<String> = []
        for item in items {
            let fingerprint = trackerFingerprint(item)
            delivered.insert(fingerprint)
            if var sources = preparedSources[fingerprint], !sources.isEmpty {
                sentRunItems.insert(sources.removeFirst())
                preparedSources[fingerprint] = sources
            } else {
                sentItemFingerprints.insert(fingerprint)
            }
        }
        guard let remaining = remainingInitialInput else { return }
        let filtered = remaining.filter { !delivered.contains(trackerFingerprint($0)) }
        remainingInitialInput = filtered.isEmpty ? nil : filtered
    }

    /// Marks inputs as accepted by the server after a successful request.
    func markInputAsAccepted(_ items: [TResponseInputItem]) {
        markInputAsSent(items)
    }

    /// Rewinds previously marked inputs so they can be resent.
    func rewindInput(_ items: [TResponseInputItem]) {
        guard !items.isEmpty else { return }
        for item in items {
            sentItemFingerprints.remove(trackerFingerprint(item))
        }
        sentRunItems.removeAll()
        logger.debug("Queued \(items.count) items to resend after conversation retry")
        remainingInitialInput = items + (remainingInitialInput ?? [])
    }

    /// Assembles the next model input while skipping duplicates and approvals.
    func prepareInput(_ originalInput: RunInput, _ generatedItems: [RunItem]) -> [TResponseInputItem] {
        preparedSources.removeAll()
        var preparedInitial: [TResponseInputItem] = []
        if !sentInitialInput {
            let initial = ItemHelpers.inputToNewInputList(originalInput)
            preparedInitial = normalizeInputItemsForAPI(initial)
            remainingInitialInput = initial.isEmpty ? nil : initial
            sentInitialInput = true
        } else if let remaining = remainingInitialInput, !remaining.isEmpty {
            preparedInitial = normalizeInputItemsForAPI(remaining)
        }

        var preparedGenerated: [TResponseInputItem] = []
        var sources: [ObjectIdentifier] = []
        for item in generatedItems {
            if item is ToolApprovalItem { continue }
            if sentRunItems.contains(ObjectIdentifier(item)) { continue }
            let raw = item.rawItem
            if let id = normalizeServerItemId(raw["id"]), serverItemIds.contains(id) { continue }
            if let callId = raw.string("call_id"), raw.contains("output"), serverToolCallIds.contains(callId) { continue }
            if serverItemFingerprints.contains(trackerFingerprint(raw)) { continue }
            guard let converted = try? runItemToInputItem(item, reasoningItemIdPolicy: reasoningItemIdPolicy) else { continue }
            if serverItemFingerprints.contains(trackerFingerprint(converted)) { continue }
            preparedGenerated.append(converted)
            sources.append(ObjectIdentifier(item))
        }
        let normalized = normalizeInputItemsForAPI(preparedGenerated)
        for (item, source) in zip(normalized, sources) {
            preparedSources[trackerFingerprint(item), default: []].append(source)
        }
        return preparedInitial + dropOrphanFunctionCalls(normalized)
    }
}
