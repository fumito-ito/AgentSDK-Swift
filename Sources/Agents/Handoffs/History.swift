import Foundation

// Port of src/agents/handoffs/history.py.
//
// Swift: the provenance tracking used for session de-duplication of nested items
// (`_nested_history_owned_items`) is not ported; see porting/manifest.json.

private let defaultConversationHistoryStart = "<CONVERSATION HISTORY>"
private let defaultConversationHistoryEnd = "</CONVERSATION HISTORY>"
private let conversationHistoryPreamble = "For context, here is the conversation so far between the user and the previous agent:"
private let legacyConversationHistoryPreamble = "For context, here is the conversation so far:"
private let supportedConversationHistoryPreambles: Set<String> = [conversationHistoryPreamble, legacyConversationHistoryPreamble]

/// Item types summarized in the conversation history and not forwarded verbatim.
private let summaryOnlyInputTypes: Set<String> = ["function_call", "function_call_output", "reasoning"]

private let conversationHistoryWrappers = SharedBox((start: defaultConversationHistoryStart, end: defaultConversationHistoryEnd))

/// Overrides the markers that wrap the generated conversation summary. Pass `nil` to leave
/// either side unchanged.
public func setConversationHistoryWrappers(start: String? = nil, end: String? = nil) {
    conversationHistoryWrappers.mutate { wrappers in
        if let start { wrappers.start = start }
        if let end { wrappers.end = end }
    }
}

/// Restores the default `<CONVERSATION HISTORY>` markers.
public func resetConversationHistoryWrappers() {
    conversationHistoryWrappers.value = (defaultConversationHistoryStart, defaultConversationHistoryEnd)
}

/// Returns the current start/end markers used for the nested conversation summary.
public func getConversationHistoryWrappers() -> (start: String, end: String) {
    conversationHistoryWrappers.value
}

/// Summarizes the previous transcript for the next agent without redacting sensitive data.
///
/// The transcript is built from `inputHistory`, `preHandoffItems`, and `newItems`. Any existing
/// `inputItems` override is ignored, so tool arguments and outputs from those fields can remain in
/// the generated summary text. Sanitize the source fields first if needed.
public func nestHandoffHistory(
    _ handoffInputData: HandoffInputData,
    historyMapper: HandoffHistoryMapper? = nil
) throws -> HandoffInputData {
    let normalizedHistory = handoffInputData.inputHistory.inputItems
    let flattenedHistory = flattenNestedHistoryMessages(normalizedHistory).map(stripInternalInputItemMetadata)

    var normalizedItems: [(item: TResponseInputItem, forward: Bool)] = []
    for runItem in handoffInputData.preHandoffItems where !(runItem is ToolApprovalItem) {
        let plain = try runItemToPlainInput(runItem)
        normalizedItems.append((plain, shouldForwardPreItem(plain)))
    }
    for runItem in handoffInputData.newItems where !(runItem is ToolApprovalItem) {
        let plain = try runItemToPlainInput(runItem)
        normalizedItems.append((plain, shouldForwardNewItem(plain)))
    }

    let historyItems: [TResponseInputItem]
    if let historyMapper {
        historyItems = historyMapper(flattenedHistory + normalizedItems.map(\.item))
    } else {
        historyItems = buildOrderedDefaultHistory(flattenedHistory, normalizedItems)
    }

    return handoffInputData.clone { data in
        data.inputHistory = .items(historyItems)
        data.preHandoffItems = []
        // The mapped history is the exact model input. New items stay unchanged for session
        // history.
        data.inputItems = []
    }
}

/// Returns a single assistant message summarizing the transcript.
public func defaultHandoffHistoryMapper(_ transcript: [TResponseInputItem]) -> [TResponseInputItem] {
    [buildSummaryMessage(transcript)]
}

private func runItemToPlainInput(_ runItem: RunItem) throws -> TResponseInputItem {
    guard let item = try runItemToInputItem(runItem) else {
        throw TypeError("Unsupported nested handoff run item: \(runItem.type)")
    }
    return item
}

private func buildOrderedDefaultHistory(
    _ flattenedHistory: [TResponseInputItem],
    _ normalizedItems: [(item: TResponseInputItem, forward: Bool)]
) -> [TResponseInputItem] {
    var historyItems: [TResponseInputItem] = []
    var pendingSummary = flattenedHistory
    for (plain, forward) in normalizedItems {
        if !forward {
            pendingSummary.append(plain)
            continue
        }
        if !pendingSummary.isEmpty || historyItems.isEmpty {
            historyItems.append(contentsOf: defaultHandoffHistoryMapper(pendingSummary))
            pendingSummary = []
        }
        historyItems.append(plain)
    }
    if !pendingSummary.isEmpty || historyItems.isEmpty {
        historyItems.append(contentsOf: defaultHandoffHistoryMapper(pendingSummary))
    }
    return historyItems
}

private func buildSummaryMessage(_ transcript: [TResponseInputItem]) -> TResponseInputItem {
    let summaryLines: [String] = transcript.isEmpty
        ? ["(no previous turns recorded)"]
        : transcript.enumerated().map { "\($0.offset + 1). \(formatTranscriptItem($0.element))" }
    let (start, end) = getConversationHistoryWrappers()
    let content = ([conversationHistoryPreamble, start] + summaryLines + [end]).joined(separator: "\n")
    return ["role": "assistant", "content": .string(content)]
}

private func formatTranscriptItem(_ item: TResponseInputItem) -> String {
    let item = stripInternalInputItemMetadata(item)
    if item.string("role") != nil {
        let content = item["content"]
        if content == nil || content == .null || (content?.stringValue.map { !$0.contains("\n") && !$0.contains("\r") } ?? false) {
            return formatTranscriptItemLegacy(item)
        }
    }
    return formatTranscriptItemJSON(item)
}

/// Serializes like `json.dumps(payload, ensure_ascii=False)`.
private func pythonDumps(_ value: JSONValue) -> String {
    switch value {
    case .array(let array):
        return "[" + array.map(pythonDumps).joined(separator: ", ") + "]"
    case .object(let object):
        return "{" + object.pairs.map { "\(JSONValue.escape($0.key)): \(pythonDumps($0.value))" }.joined(separator: ", ") + "}"
    case .string(let string):
        return JSONValue.escape(string)
    default:
        return value.jsonString()
    }
}

private func formatTranscriptItemJSON(_ item: TResponseInputItem) -> String {
    pythonDumps(.object(item.removing("provider_data")))
}

private func formatTranscriptItemLegacy(_ item: TResponseInputItem) -> String {
    if let role = item.string("role") {
        var prefix = role
        if let name = item.string("name"), !name.isEmpty {
            prefix = "\(prefix) (\(name))"
        }
        return "\(prefix): \(stringifyContent(item["content"]))"
    }
    let itemType = item.string("type") ?? "item"
    let rest = item.removing("type", "provider_data")
    let serialized = pythonDumps(.object(rest))
    return serialized.isEmpty ? itemType : "\(itemType): \(serialized)"
}

private func stringifyContent(_ content: JSONValue?) -> String {
    switch content {
    case nil, .null?: return ""
    case .string(let string)?: return string
    case let value?: return pythonDumps(value)
    }
}

private func flattenNestedHistoryMessages(_ items: [TResponseInputItem]) -> [TResponseInputItem] {
    var flattened: [TResponseInputItem] = []
    for item in items {
        if let nested = extractNestedHistoryTranscript(item) {
            flattened.append(contentsOf: nested)
        } else {
            flattened.append(item)
        }
    }
    return flattened
}

private func extractNestedHistoryTranscript(_ item: TResponseInputItem) -> [TResponseInputItem]? {
    guard item.string("role") == "assistant", let content = item.string("content") else { return nil }
    let (start, end) = getConversationHistoryWrappers()
    guard let newline = content.firstIndex(of: "\n") else { return nil }
    let preamble = String(content[..<newline])
    guard supportedConversationHistoryPreambles.contains(preamble) else { return nil }
    let wrapped = String(content[content.index(after: newline)...])
    let startWrapper = "\(start)\n"
    let endWrapper = "\n\(end)"
    guard wrapped.hasPrefix(startWrapper), wrapped.hasSuffix(endWrapper),
          wrapped.count >= startWrapper.count + endWrapper.count else { return nil }
    let body = String(wrapped.dropFirst(startWrapper.count).dropLast(endWrapper.count))
    return splitSummaryRecords(body).compactMap(parseSummaryLine)
}

private func splitSummaryRecords(_ body: String) -> [String] {
    var records: [String] = []
    var current: [String] = []
    var currentIsNumbered = false
    for rawLine in body.components(separatedBy: .newlines) {
        guard !rawLine.trimmingCharacters(in: .whitespaces).isEmpty else { continue }
        let startsNumbered = startsNumberedSummaryRecord(rawLine)
        if current.isEmpty {
            current = [rawLine.trimmingCharacters(in: .whitespaces)]
            currentIsNumbered = startsNumbered
            continue
        }
        if startsNumbered || !currentIsNumbered {
            records.append(current.joined(separator: "\n"))
            current = [rawLine.trimmingCharacters(in: .whitespaces)]
            currentIsNumbered = startsNumbered
            continue
        }
        current.append(String(rawLine.reversed().drop(while: { $0 == " " || $0 == "\t" }).reversed()))
    }
    if !current.isEmpty {
        records.append(current.joined(separator: "\n"))
    }
    return records
}

private func startsNumberedSummaryRecord(_ line: String) -> Bool {
    let stripped = line.drop(while: { $0 == " " || $0 == "\t" })
    guard let dot = stripped.firstIndex(of: ".") else { return false }
    let digits = stripped[..<dot]
    return !digits.isEmpty && digits.allSatisfy(\.isNumber)
}

private func stripSummaryLineNumber(_ stripped: String) -> String {
    guard let dot = stripped.firstIndex(of: ".") else { return stripped }
    let digits = stripped[..<dot]
    guard !digits.isEmpty, digits.allSatisfy(\.isNumber) else { return stripped }
    return String(stripped[stripped.index(after: dot)...].drop(while: { $0 == " " }))
}

private let knownTranscriptRoles: Set<String> = ["user", "assistant", "system", "developer"]

private func parseSummaryLine(_ line: String) -> TResponseInputItem? {
    var stripped = line.trimmingCharacters(in: .whitespaces)
    guard !stripped.isEmpty else { return nil }
    stripped = stripSummaryLineNumber(stripped)
    if case .object(let parsed)? = try? JSONValue.parse(stripped) {
        return stripInternalInputItemMetadata(parsed.removing("provider_data"))
    }
    guard let colon = stripped.firstIndex(of: ":") else {
        guard isBareRoleRecord(stripped) else { return nil }
        let (role, name) = splitRoleAndName(stripped)
        var recovered: JSONObject = ["role": .string(role)]
        if let name { recovered["name"] = .string(name) }
        recovered["content"] = ""
        return recovered
    }
    let roleText = stripped[..<colon].trimmingCharacters(in: .whitespaces)
    guard !roleText.isEmpty else { return nil }
    let (role, name) = splitRoleAndName(roleText)
    var reconstructed: JSONObject = ["role": .string(role)]
    if let name { reconstructed["name"] = .string(name) }
    let content = stripped[stripped.index(after: colon)...].trimmingCharacters(in: .whitespaces)
    if !content.isEmpty, let typed = parseLegacyTypedItem(role, content) {
        return typed
    }
    reconstructed["content"] = .string(content)
    return reconstructed
}

private func parseLegacyTypedItem(_ itemType: String, _ content: String) -> TResponseInputItem? {
    guard !knownTranscriptRoles.contains(itemType), case .object(var parsed)? = try? JSONValue.parse(content) else {
        return nil
    }
    parsed.removeValue(forKey: "provider_data")
    parsed["type"] = .string(itemType)
    return stripInternalInputItemMetadata(parsed)
}

private func isBareRoleRecord(_ stripped: String) -> Bool {
    var candidate = stripped
    if candidate.hasSuffix(")"), let open = candidate.lastIndex(of: "(") {
        candidate = candidate[..<open].trimmingCharacters(in: .whitespaces)
    }
    return knownTranscriptRoles.contains(candidate)
}

private func splitRoleAndName(_ roleText: String) -> (String, String?) {
    if roleText.hasSuffix(")"), let open = roleText.lastIndex(of: "(") {
        let possibleName = roleText[roleText.index(after: open)..<roleText.index(before: roleText.endIndex)]
            .trimmingCharacters(in: .whitespaces)
        let roleCandidate = roleText[..<open].trimmingCharacters(in: .whitespaces)
        if !possibleName.isEmpty {
            return (roleCandidate.isEmpty ? "developer" : roleCandidate, possibleName)
        }
    }
    return (roleText.isEmpty ? "developer" : roleText, nil)
}

private func isProgrammaticTranscriptItem(_ item: TResponseInputItem) -> Bool {
    if item.type == "program" || item.type == "program_output" { return true }
    if case .object(let caller)? = item["caller"] { return caller.type == "program" }
    return false
}

private func shouldForwardPreItem(_ item: TResponseInputItem) -> Bool {
    if isProgrammaticTranscriptItem(item) { return false }
    if item.string("role") == "assistant" { return false }
    if let type = item.type, summaryOnlyInputTypes.contains(type) { return false }
    return true
}

private func shouldForwardNewItem(_ item: TResponseInputItem) -> Bool {
    if isProgrammaticTranscriptItem(item) { return false }
    if let role = item.string("role"), !role.isEmpty { return true }
    if let type = item.type, summaryOnlyInputTypes.contains(type) { return false }
    return true
}
