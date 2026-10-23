import Foundation

/// A lock-protected box for state shared between a run context and its forks.
final class SharedBox<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Value

    init(_ value: Value) {
        storage = value
    }

    var value: Value {
        get {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            storage = newValue
        }
    }

    @discardableResult
    func mutate<Result>(_ body: (inout Value) throws -> Result) rethrows -> Result {
        lock.lock()
        defer { lock.unlock() }
        return try body(&storage)
    }
}

/// Approval or rejection state for one tool (upstream `_ApprovalRecord`).
///
/// `approved` and `rejected` are either permanent (`.all`) or scoped to specific call IDs.
public struct ApprovalRecord: Sendable, Equatable {
    public enum Decision: Sendable, Equatable {
        /// A permanent allow/deny for every call (`True`) or explicitly cleared (`False`).
        case all(Bool)
        /// Decisions scoped to these call IDs.
        case callIds([String])

        var callIds: [String]? {
            if case .callIds(let ids) = self { return ids }
            return nil
        }

        var json: JSONValue {
            switch self {
            case .all(let value): return .bool(value)
            case .callIds(let ids): return .array(ids.map(JSONValue.string))
            }
        }

        init(json: JSONValue?) {
            switch json {
            case .bool(let value)?: self = .all(value)
            case .array(let values)?: self = .callIds(values.compactMap(\.stringValue))
            default: self = .callIds([])
            }
        }
    }

    public var approved: Decision = .callIds([])
    public var rejected: Decision = .callIds([])
    public var rejectionMessages: [String: String] = [:]
    public var stickyRejectionMessage: String?
    /// The approval scope a sticky decision was made for, so it cannot authorize another tool.
    public var stickyScope: String?

    public init() {}

    /// Returns `true`/`false`/`nil` for a call ID, giving exact call decisions precedence.
    func status(for callId: String) -> Bool? {
        if approved.callIds?.contains(callId) == true { return true }
        if rejected.callIds?.contains(callId) == true { return false }
        // Approval takes precedence when sticky decisions conflict.
        if approved == .all(true) { return true }
        if rejected == .all(true) { return false }
        return nil
    }

    /// Returns only an exact-call decision.
    func perCallStatus(for callId: String) -> Bool? {
        if approved.callIds?.contains(callId) == true { return true }
        if rejected.callIds?.contains(callId) == true { return false }
        return nil
    }

    func rejectionMessage(for callId: String) -> String? {
        if approved.callIds?.contains(callId) == true { return nil }
        if rejected == .all(true) {
            return rejectionMessages[callId] ?? stickyRejectionMessage
        }
        if rejected.callIds?.contains(callId) == true {
            return rejectionMessages[callId]
        }
        return nil
    }

    var json: JSONObject {
        var json: JSONObject = [
            "approved": approved.json,
            "rejected": rejected.json,
        ]
        if !rejectionMessages.isEmpty {
            json["rejection_messages"] = .object(JSONObject(rejectionMessages.mapValues(JSONValue.string)))
        }
        if let stickyRejectionMessage {
            json["sticky_rejection_message"] = .string(stickyRejectionMessage)
        }
        if let stickyScope {
            json["sticky_scope"] = .string(stickyScope)
        }
        return json
    }

    init(json: JSONObject) {
        approved = Decision(json: json["approved"])
        rejected = Decision(json: json["rejected"])
        if case .object(let messages)? = json["rejection_messages"] {
            rejectionMessages = messages.dictionary.compactMapValues(\.stringValue)
        }
        stickyRejectionMessage = json.string("sticky_rejection_message")
        stickyScope = json.string("sticky_scope")
    }
}

/// Wraps the context object passed to `Runner.run()`, plus usage for the run so far.
///
/// Contexts are not passed to the LLM. They are a way to pass dependencies and data to code you
/// implement, like tool functions, callbacks, and hooks.
///
/// Swift: the context is type-erased (porting/decisions.md D4). Use `context(as:)` to read it with
/// its concrete type, and use a class for contexts that tools mutate.
open class RunContextWrapper: @unchecked Sendable {
    /// The context object passed by you to `Runner.run()`.
    public var context: any Sendable

    let usageBox: SharedBox<Usage>
    let toolStateBox: SharedBox<ToolApprovalState>

    /// The usage of the agent run so far. Forked contexts (for example `ToolContext`) share it.
    public var usage: Usage {
        get { usageBox.value }
        set { usageBox.value = newValue }
    }

    /// The input items for the current turn.
    public var turnInput: [TResponseInputItem]

    /// Structured input for the current agent tool run, when available.
    public var toolInput: (any Sendable)?

    public init(context: any Sendable = (), usage: Usage = Usage(), turnInput: [TResponseInputItem] = []) {
        self.context = context
        self.usageBox = SharedBox(usage)
        self.toolStateBox = SharedBox(ToolApprovalState())
        self.turnInput = turnInput
    }

    init(
        context: any Sendable,
        usageBox: SharedBox<Usage>,
        toolStateBox: SharedBox<ToolApprovalState>,
        turnInput: [TResponseInputItem],
        toolInput: (any Sendable)?
    ) {
        self.context = context
        self.usageBox = usageBox
        self.toolStateBox = toolStateBox
        self.turnInput = turnInput
        self.toolInput = toolInput
    }

    /// Returns the context cast to `type`, or `nil` when it has a different type.
    public func context<T>(as type: T.Type = T.self) -> T? {
        context as? T
    }

    /// Records usage from a model call. Safe to call concurrently.
    public func addUsage(_ other: Usage) {
        usageBox.mutate { $0.add(other) }
    }

    // MARK: - Approvals

    var approvals: [ApprovalKey: ApprovalRecord] {
        get { toolStateBox.value.approvals }
        set { toolStateBox.mutate { $0.approvals = newValue } }
    }

    var toolInvocations: [String: ToolInvocationRecord] {
        get { toolStateBox.value.toolInvocations }
        set { toolStateBox.mutate { $0.toolInvocations = newValue } }
    }

    static func resolveToolName(_ item: ToolApprovalItem) -> String {
        if let name = item.toolName, !name.isEmpty { return name }
        let raw = item.rawItem
        if let name = raw.string("name"), !name.isEmpty { return name }
        if let type = raw.string("type"), !type.isEmpty { return type }
        return "unknown_tool"
    }

    static func resolveToolNamespace(_ item: ToolApprovalItem) -> String? {
        if let namespace = item.toolNamespace, !namespace.isEmpty { return namespace }
        return item.rawItem.string("namespace")
    }

    static func resolveToolLookupKey(_ item: ToolApprovalItem) -> FunctionToolLookupKey? {
        if let key = item.toolLookupKey { return key }
        guard item.rawItem.type == "function_call" else { return nil }
        let name = resolveToolName(item)
        let namespace = resolveToolNamespace(item)
        if isReservedSyntheticToolNamespace(name, namespace) { return nil }
        return getFunctionToolLookupKey(name, namespace)
    }

    /// Returns all approval keys that should mirror this approval record.
    static func resolveApprovalKeys(_ item: ToolApprovalItem) -> [String] {
        let lookupKey = resolveToolLookupKey(item)
        return getFunctionToolApprovalKeys(
            toolName: resolveToolName(item),
            toolNamespace: resolveToolNamespace(item),
            allowBareNameAlias: item.allowBareNameAlias,
            toolLookupKey: lookupKey,
            preferLegacySameNameNamespace: lookupKey == nil
        )
    }

    static func resolveApprovalKey(_ item: ToolApprovalItem) -> String {
        let toolName = resolveToolName(item)
        let toolNamespace = resolveToolNamespace(item)
        let lookupKey = resolveToolLookupKey(item)
        let keys = getFunctionToolApprovalKeys(
            toolName: toolName,
            toolNamespace: toolNamespace,
            toolLookupKey: lookupKey,
            preferLegacySameNameNamespace: lookupKey == nil
        )
        return keys.last ?? toolQualifiedName(toolName, namespace: toolNamespace) ?? toolName
    }

    static func resolveCallId(_ item: ToolApprovalItem) -> String? {
        if let hosted = getHostedMCPApprovalRequestIdentity(item.rawItem) {
            return hosted.requestId
        }
        let raw = item.rawItem
        if case .object(let providerData)? = raw["provider_data"], providerData.type == "mcp_approval_request",
           let id = providerData.string("id") {
            return id
        }
        var candidate = raw.type == "mcp_approval_request" ? raw["id"] : raw["call_id"]
        if candidate == nil || candidate == .null, raw.type == nil {
            candidate = raw["id"]
        }
        switch candidate {
        case .string(let text)?: return text
        case nil, .null?: return nil
        case let value?: return value.jsonString()
        }
    }

    /// Approves a tool call, optionally for all future calls of the same tool.
    ///
    /// Permanent function-tool decisions apply only to the approval item's agent.
    public func approveTool(_ approvalItem: ToolApprovalItem, alwaysApprove: Bool = false) throws {
        try toolStateBox.mutate { try $0.applyApprovalDecision(approvalItem, always: alwaysApprove, approve: true) }
    }

    /// Rejects a tool call, optionally for all future calls of the same tool.
    ///
    /// Permanent function-tool decisions apply only to the approval item's agent.
    public func rejectTool(_ approvalItem: ToolApprovalItem, alwaysReject: Bool = false, rejectionMessage: String? = nil) throws {
        try toolStateBox.mutate {
            try $0.applyApprovalDecision(approvalItem, always: alwaysReject, approve: false, rejectionMessage: rejectionMessage)
        }
    }

    /// Resolves an execution decision using its current or pending invocation.
    ///
    /// Agent-owned function decisions require an invocation carrying the agent. Use
    /// `isToolApproved` for name-only inspection of an unambiguous function decision.
    public func getApprovalStatus(
        toolName: String,
        callId: String,
        toolNamespace: String? = nil,
        existingPending: ToolApprovalItem? = nil,
        toolLookupKey: FunctionToolLookupKey? = nil,
        currentInvocation: ToolApprovalItem? = nil
    ) throws -> Bool? {
        try toolStateBox.mutate {
            try $0.getApprovalStatus(
                toolName: toolName,
                callId: callId,
                toolNamespace: toolNamespace,
                existingPending: existingPending,
                toolLookupKey: toolLookupKey,
                currentInvocation: currentInvocation
            )
        }
    }

    /// Returns `true`/`false`/`nil` for the given tool call.
    public func isToolApproved(toolName: String, callId: String) -> Bool? {
        toolStateBox.value.isToolApproved(toolName: toolName, callId: callId)
    }

    /// Returns a stored rejection message for a tool call if one exists.
    public func getRejectionMessage(
        toolName: String,
        callId: String,
        toolNamespace: String? = nil,
        existingPending: ToolApprovalItem? = nil,
        toolLookupKey: FunctionToolLookupKey? = nil
    ) -> String? {
        toolStateBox.mutate {
            $0.getRejectionMessage(
                toolName: toolName,
                callId: callId,
                toolNamespace: toolNamespace,
                existingPending: existingPending,
                toolLookupKey: toolLookupKey
            )
        }
    }

    /// Restores approvals from serialized state.
    func rebuildApprovals(_ approvals: JSONValue?) {
        toolStateBox.mutate { $0.rebuildApprovals(approvals) }
    }

    // MARK: - Forks

    /// Creates a child context that shares approvals and usage, with tool input set.
    public func forkWithToolInput(_ toolInput: (any Sendable)?) -> RunContextWrapper {
        RunContextWrapper(context: context, usageBox: usageBox, toolStateBox: toolStateBox, turnInput: turnInput, toolInput: toolInput)
    }

    /// Creates a child context that shares approvals and usage without tool input.
    public func forkWithoutToolInput() -> RunContextWrapper {
        RunContextWrapper(context: context, usageBox: usageBox, toolStateBox: toolStateBox, turnInput: turnInput, toolInput: nil)
    }

    /// Copies SDK-owned state for an independently resumable checkpoint.
    func copyForRunState() -> RunContextWrapper {
        RunContextWrapper(
            context: context,
            usageBox: SharedBox(usage),
            toolStateBox: SharedBox(toolStateBox.value),
            turnInput: turnInput,
            toolInput: toolInput
        )
    }
}

/// Context passed to agent hooks (`on_start`, `on_end`).
public final class AgentHookContext: RunContextWrapper, @unchecked Sendable {
    /// Creates a hook context that shares usage and approvals with `wrapper`.
    public init(from wrapper: RunContextWrapper) {
        super.init(
            context: wrapper.context,
            usageBox: wrapper.usageBox,
            toolStateBox: wrapper.toolStateBox,
            turnInput: wrapper.turnInput,
            toolInput: wrapper.toolInput
        )
    }
}

// MARK: - Decisions

extension ToolApprovalState {
    /// Builds approval-key candidates in upstream order for status and rejection lookups.
    private static func candidateKeys(
        toolName: String,
        toolNamespace: String?,
        existingPending: ToolApprovalItem?,
        toolLookupKey: FunctionToolLookupKey?
    ) -> [String] {
        var candidates: [String] = []
        func add(_ key: String) {
            if !candidates.contains(key) { candidates.append(key) }
        }
        let explicitNamespace = (toolNamespace?.isEmpty ?? true) ? nil : toolNamespace
        let pendingNamespace = existingPending.flatMap(RunContextWrapper.resolveToolNamespace)
        let pendingKey = existingPending.map(RunContextWrapper.resolveApprovalKey)
        if let pendingKey { add(pendingKey) }
        var explicitKeys: [String] = []
        if explicitNamespace != nil || toolLookupKey != nil {
            explicitKeys = getFunctionToolApprovalKeys(
                toolName: toolName,
                toolNamespace: explicitNamespace,
                toolLookupKey: toolLookupKey,
                includeLegacyDeferredKey: true
            )
        }
        explicitKeys.forEach(add)
        if explicitKeys.isEmpty, pendingNamespace?.isEmpty == false, let pendingKey { add(pendingKey) }
        if explicitNamespace == nil, toolLookupKey == nil, existingPending == nil { add(toolName) }
        if let existingPending {
            RunContextWrapper.resolveApprovalKeys(existingPending).forEach(add)
            if pendingNamespace == nil { add(RunContextWrapper.resolveToolName(existingPending)) }
        }
        return candidates
    }

    func isToolApproved(toolName: String, callId: String) -> Bool? {
        if let status = approvals[.hostedMCPQuery(toolName: toolName, requestId: callId)]?.perCallStatus(for: callId) {
            return status
        }
        for keyed in approvalRecordsForKey(toolName, callId: callId) {
            if let status = keyed.record.status(for: callId) { return status }
        }
        return nil
    }

    mutating func getRejectionMessage(
        toolName: String,
        callId: String,
        toolNamespace: String?,
        existingPending: ToolApprovalItem?,
        toolLookupKey: FunctionToolLookupKey?
    ) -> String? {
        if let existingPending, getHostedMCPApprovalRequestIdentity(existingPending.rawItem) != nil {
            return resolveHostedMCPApprovalDecision(existingPending).rejectionMessage
        }
        if let record = approvals[.hostedMCPQuery(toolName: toolName, requestId: callId)], record.perCallStatus(for: callId) != nil {
            return record.rejectionMessage(for: callId)
        }
        let candidates = Self.candidateKeys(toolName: toolName, toolNamespace: toolNamespace, existingPending: existingPending, toolLookupKey: toolLookupKey)
        for candidate in candidates {
            for keyed in approvalRecordsForKey(candidate, existingPending, callId: callId) where keyed.record.status(for: callId) != nil {
                return keyed.record.rejectionMessage(for: callId)
            }
        }
        return nil
    }

    mutating func getApprovalStatus(
        toolName: String,
        callId: String,
        toolNamespace: String?,
        existingPending: ToolApprovalItem?,
        toolLookupKey: FunctionToolLookupKey?,
        currentInvocation: ToolApprovalItem?
    ) throws -> Bool? {
        guard !callId.isEmpty else {
            throw ModelBehaviorError("Approval-gated tool calls require a non-empty call ID.")
        }
        if let existingPending {
            try restorePendingApprovalBinding(existingPending)
            let pendingIdentity = toolInvocationIdentity(
                existingPending.rawItem,
                toolLookupKey: existingPending.toolLookupKey,
                toolName: existingPending.toolName
            )
            if pendingIdentity == nil {
                if let pendingCallId = RunContextWrapper.resolveCallId(existingPending),
                   currentInvocation == nil || toolInvocations[pendingCallId] == nil {
                    restoredUnboundApprovalCallIds.insert(pendingCallId)
                }
                if currentInvocation == nil { return nil }
            }
            if getHostedMCPApprovalRequestIdentity(existingPending.rawItem) != nil {
                guard let hostedStatus = resolveHostedMCPApprovalDecision(existingPending).status else { return nil }
                let effective = currentInvocation ?? existingPending
                let binding = try approvedToolInvocationStatus(
                    effective.rawItem,
                    toolLookupKey: effective.toolLookupKey,
                    toolName: effective.toolName,
                    approvalItem: effective
                )
                return binding != nil ? hostedStatus : nil
            }
        }

        let candidates = Self.candidateKeys(toolName: toolName, toolNamespace: toolNamespace, existingPending: existingPending, toolLookupKey: toolLookupKey)
        let selected = currentInvocation ?? existingPending
        if let selected, toolInvocations[callId] != nil {
            // Changing the approval owner must not bypass invocation reuse validation.
            _ = try toolInvocationStatus(selected.rawItem, toolLookupKey: selected.toolLookupKey, toolName: selected.toolName)
        }
        let scope = selected.flatMap {
            toolInvocationApprovalScope($0.rawItem, toolLookupKey: $0.toolLookupKey, toolName: $0.toolName)?.scope
        }
        var status: Bool?
        var matched: KeyedApprovalRecord?
        search: for candidate in candidates {
            // Runtime name-only queries (such as native operation callbacks) must not borrow an
            // agent-owned function decision from the inspection API.
            let records = selected != nil
                ? approvalRecordsForKey(candidate, selected, callId: callId, approvalScope: scope)
                : [approvalRecordForKey(candidate, callId)].compactMap { $0 }
            for keyed in records {
                if let recordStatus = keyed.record.status(for: callId) {
                    status = recordStatus
                    matched = keyed
                    break search
                }
            }
        }
        guard let status, let matched, let selected else { return status }

        if matched.record.isSticky {
            if matched.record.stickyScope == nil, allowLegacyApprovalBindingReconstruction, let key = matched.key,
               let scope = toolInvocationApprovalScope(selected.rawItem, toolLookupKey: selected.toolLookupKey, toolName: selected.toolName) {
                approvals[key]?.stickyScope = scope.scope
            }
            let binding = try approvedToolInvocationStatus(
                selected.rawItem,
                toolLookupKey: selected.toolLookupKey,
                toolName: selected.toolName,
                approvalItem: selected
            )
            return binding != nil ? status : nil
        }
        if let currentInvocation,
           toolInvocationIdentity(currentInvocation.rawItem, toolLookupKey: currentInvocation.toolLookupKey, toolName: currentInvocation.toolName) == nil {
            _ = try approvedToolInvocationStatus(
                currentInvocation.rawItem,
                toolLookupKey: currentInvocation.toolLookupKey,
                toolName: currentInvocation.toolName
            )
            return nil
        }
        let binding = try approvedToolInvocationStatus(
            selected.rawItem,
            toolLookupKey: selected.toolLookupKey,
            toolName: selected.toolName,
            approvalItem: selected
        )
        if binding == nil {
            if toolInvocationIdentity(selected.rawItem, toolLookupKey: selected.toolLookupKey, toolName: selected.toolName) != nil {
                return nil
            }
            if let existingPending,
               toolInvocationIdentity(existingPending.rawItem, toolLookupKey: existingPending.toolLookupKey, toolName: existingPending.toolName) == nil,
               isMCPApprovalInvocation(existingPending.rawItem) {
                return nil
            }
        }
        return status
    }

    /// Records an approval or rejection decision.
    mutating func applyApprovalDecision(
        _ approvalItem: ToolApprovalItem,
        always: Bool,
        approve: Bool,
        rejectionMessage: String? = nil
    ) throws {
        let hostedRequest = getHostedMCPApprovalRequestIdentity(approvalItem.rawItem)
        let callId: String?
        var hostedKey: ApprovalKey?
        if let hostedRequest {
            guard let requestId = hostedRequest.requestId else {
                throw UserError("Hosted MCP approval decisions require a non-empty request id.")
            }
            callId = requestId
            if let identity = hostedRequest.approvalIdentity {
                hostedKey = .hostedMCP(serverLabel: identity.serverLabel, toolName: identity.toolName)
            } else if always {
                throw UserError("Persistent hosted MCP approval decisions require a non-empty server_label and tool name.")
            } else {
                hostedKey = .hostedMCPCall(requestId: requestId)
            }
        } else {
            callId = RunContextWrapper.resolveCallId(approvalItem)
        }

        if let callIdentity = toolInvocationCallId(approvalItem.rawItem), callIdentity.callId == nil {
            throw ModelBehaviorError("Approval decisions require a non-empty call ID for recognized tool invocations.")
        }
        let raw = approvalItem.rawItem
        let rawCallId = raw["call_id"] != nil ? raw["call_id"] : raw["id"]
        if rawCallId == .string(""), !always {
            throw ModelBehaviorError("Per-call approval decisions require a non-empty call ID.")
        }
        let invocation = callId == nil
            ? nil
            : toolInvocationIdentity(raw, toolLookupKey: approvalItem.toolLookupKey, toolName: approvalItem.toolName)
        let hasRawCallId = rawCallId != nil && rawCallId != .null
        if (callId != nil && invocation == nil) || (callId == nil && hasRawCallId) {
            throw ModelBehaviorError("Approval decisions require a canonical invocation identity.")
        }
        let scope = toolInvocationApprovalScope(raw, toolLookupKey: approvalItem.toolLookupKey, toolName: approvalItem.toolName)?.scope
        if let invocation, let callId {
            guard invocation.callId == callId else {
                throw ModelBehaviorError("Approval decision call ID does not match its canonical invocation ID.")
            }
            let wasRestoredUnbound = restoredUnboundApprovalCallIds.remove(callId) != nil
            do {
                _ = try toolInvocationStatus(raw, toolLookupKey: approvalItem.toolLookupKey, toolName: approvalItem.toolName)
            } catch {
                if wasRestoredUnbound { restoredUnboundApprovalCallIds.insert(callId) }
                throw error
            }
        }

        var entries: [(key: ApprovalKey, sticky: Bool)] = []
        if let hostedRequest, let hostedKey, let callId {
            entries.append((getOrCreateApprovalEntry(hostedKey), always))
            if let hostedToolName = Self.resolveHostedMCPToolName(approvalItem, hostedRequest) {
                // Preserve exact name-based lookup without adding an authorization source.
                entries.append((getOrCreateApprovalEntry(.hostedMCPQuery(toolName: hostedToolName, requestId: callId)), false))
            }
        } else {
            let approvalKeys = RunContextWrapper.resolveApprovalKeys(approvalItem)
            let decisionKeys = always || callId == nil
                ? [RunContextWrapper.resolveApprovalKey(approvalItem)]
                : (approvalKeys.isEmpty ? ["unknown_tool"] : approvalKeys)
            for approvalKey in decisionKeys {
                let key = Self.functionApprovalKey(approvalKey, approvalItem) ?? .name(approvalKey)
                entries.append((getOrCreateApprovalEntry(key, approvalScope: scope), always))
            }
        }

        for entry in entries {
            approvals[entry.key] = Self.recordingDecision(
                in: approvals[entry.key] ?? ApprovalRecord(),
                sticky: entry.sticky || callId == nil,
                callId: callId,
                approve: approve,
                approvalScope: scope,
                rejectionMessage: rejectionMessage
            )
        }

        if invocation != nil, let callId {
            restoredUnboundApprovalCallIds.remove(callId)
        }
    }

    /// Returns `record` with one approval or rejection applied.
    private static func recordingDecision(
        in record: ApprovalRecord,
        sticky: Bool,
        callId: String?,
        approve: Bool,
        approvalScope: String?,
        rejectionMessage: String?
    ) -> ApprovalRecord {
        var record = record
        if sticky {
            record.stickyScope = approvalScope
            record.approved = .all(approve)
            record.rejected = approve ? .callIds([]) : .all(true)
            if approve {
                record.rejectionMessages.removeAll()
                record.stickyRejectionMessage = nil
            } else {
                if let callId {
                    record.rejectionMessages[callId] = rejectionMessage
                }
                record.stickyRejectionMessage = rejectionMessage
            }
            return record
        }
        guard let callId else { return record }
        if approve {
            if case .callIds(var ids) = record.rejected {
                ids.removeAll { $0 == callId }
                record.rejected = .callIds(ids)
            }
            if record.approved != .all(true) {
                var ids = record.approved.callIds ?? []
                if !ids.contains(callId) { ids.append(callId) }
                record.approved = .callIds(ids)
            }
            record.rejectionMessages.removeValue(forKey: callId)
        } else {
            if case .callIds(var ids) = record.approved {
                ids.removeAll { $0 == callId }
                record.approved = .callIds(ids)
            }
            if record.rejected != .all(true) {
                var ids = record.rejected.callIds ?? []
                if !ids.contains(callId) { ids.append(callId) }
                record.rejected = .callIds(ids)
            }
            record.rejectionMessages[callId] = rejectionMessage
        }
        return record
    }
}
