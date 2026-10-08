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
        return json
    }

    init(json: JSONObject) {
        approved = Decision(json: json["approved"])
        rejected = Decision(json: json["rejected"])
        if case .object(let messages)? = json["rejection_messages"] {
            rejectionMessages = messages.dictionary.compactMapValues(\.stringValue)
        }
        stickyRejectionMessage = json.string("sticky_rejection_message")
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
    let approvalsBox: SharedBox<[String: ApprovalRecord]>

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
        self.approvalsBox = SharedBox([:])
        self.turnInput = turnInput
    }

    init(
        context: any Sendable,
        usageBox: SharedBox<Usage>,
        approvalsBox: SharedBox<[String: ApprovalRecord]>,
        turnInput: [TResponseInputItem],
        toolInput: (any Sendable)?
    ) {
        self.context = context
        self.usageBox = usageBox
        self.approvalsBox = approvalsBox
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

    var approvals: [String: ApprovalRecord] {
        get { approvalsBox.value }
        set { approvalsBox.value = newValue }
    }

    static func resolveToolName(_ item: ToolApprovalItem) -> String {
        if let name = item.toolName, !name.isEmpty { return name }
        return item.rawItem.string("name") ?? item.rawItem.string("type") ?? "unknown_tool"
    }

    static func resolveToolNamespace(_ item: ToolApprovalItem) -> String? {
        if let namespace = item.toolNamespace, !namespace.isEmpty { return namespace }
        return item.rawItem.string("namespace")
    }

    static func resolveApprovalKeys(_ item: ToolApprovalItem) -> [String] {
        let keys = getFunctionToolApprovalKeys(
            toolName: resolveToolName(item),
            toolNamespace: resolveToolNamespace(item),
            allowBareNameAlias: item.allowBareNameAlias,
            toolLookupKey: item.toolLookupKey,
            preferLegacySameNameNamespace: item.toolLookupKey == nil
        )
        return keys.isEmpty ? [resolveToolName(item)] : keys
    }

    static func resolveApprovalKey(_ item: ToolApprovalItem) -> String {
        resolveApprovalKeys(item).last ?? resolveToolName(item)
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
        if raw.type == "mcp_approval_request" { return raw.string("id") }
        if let callId = raw.string("call_id") { return callId }
        if raw.type == nil { return raw.string("id") }
        return nil
    }

    private static func hostedApprovalKeys(_ item: ToolApprovalItem) -> (exact: String, sticky: String?)? {
        guard let hosted = getHostedMCPApprovalRequestIdentity(item.rawItem), let requestId = hosted.requestId else {
            return nil
        }
        let sticky = hosted.approvalIdentity.map { "hosted_mcp:\($0.serverLabel):\($0.toolName)" }
        return ("hosted_mcp_call:\(requestId)", sticky)
    }

    private func applyDecision(_ item: ToolApprovalItem, always: Bool, approve: Bool, rejectionMessage: String?) throws {
        let callId: String?
        let keys: [String]
        if let hosted = Self.hostedApprovalKeys(item) {
            callId = getHostedMCPApprovalRequestIdentity(item.rawItem)?.requestId
            if always, hosted.sticky == nil {
                throw UserError(
                    "Persistent hosted MCP approval decisions require a non-empty server_label and tool name."
                )
            }
            keys = [hosted.sticky ?? hosted.exact]
        } else {
            callId = Self.resolveCallId(item)
            let rawCallId = item.rawItem["call_id"]?.stringValue ?? item.rawItem.string("id")
            if rawCallId == "", !always {
                throw ModelBehaviorError("Per-call approval decisions require a non-empty call ID.")
            }
            let approvalKeys = Self.resolveApprovalKeys(item)
            keys = (always || callId == nil) ? [Self.resolveApprovalKey(item)] : approvalKeys
        }

        approvalsBox.mutate { approvals in
            for key in keys {
                var record = approvals[key] ?? ApprovalRecord()
                if always || callId == nil {
                    record.approved = .all(approve)
                    record.rejected = approve ? .callIds([]) : .all(true)
                    if !approve {
                        if let rejectionMessage, let callId {
                            record.rejectionMessages[callId] = rejectionMessage
                        } else if let callId {
                            record.rejectionMessages.removeValue(forKey: callId)
                        }
                        record.stickyRejectionMessage = rejectionMessage
                    } else {
                        record.rejectionMessages.removeAll()
                        record.stickyRejectionMessage = nil
                    }
                } else if let callId {
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
                        if let rejectionMessage {
                            record.rejectionMessages[callId] = rejectionMessage
                        } else {
                            record.rejectionMessages.removeValue(forKey: callId)
                        }
                    }
                }
                approvals[key] = record
            }
        }
    }

    /// Approves a tool call, optionally for all future calls of the same tool.
    public func approveTool(_ approvalItem: ToolApprovalItem, alwaysApprove: Bool = false) throws {
        try applyDecision(approvalItem, always: alwaysApprove, approve: true, rejectionMessage: nil)
    }

    /// Rejects a tool call, optionally for all future calls of the same tool.
    public func rejectTool(_ approvalItem: ToolApprovalItem, alwaysReject: Bool = false, rejectionMessage: String? = nil) throws {
        try applyDecision(approvalItem, always: alwaysReject, approve: false, rejectionMessage: rejectionMessage)
    }

    private func candidateKeys(toolName: String, toolNamespace: String?, existingPending: ToolApprovalItem?, toolLookupKey: FunctionToolLookupKey?) -> [String] {
        var candidates: [String] = []
        func add(_ key: String) {
            if !candidates.contains(key) { candidates.append(key) }
        }
        let explicitNamespace = (toolNamespace?.isEmpty ?? true) ? nil : toolNamespace
        let pendingNamespace = existingPending.flatMap(Self.resolveToolNamespace)
        let pendingKey = existingPending.map(Self.resolveApprovalKey)
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
        if explicitKeys.isEmpty, pendingNamespace != nil, let pendingKey { add(pendingKey) }
        if explicitNamespace == nil, toolLookupKey == nil, existingPending == nil { add(toolName) }
        if let existingPending {
            Self.resolveApprovalKeys(existingPending).forEach(add)
            if pendingNamespace == nil { add(Self.resolveToolName(existingPending)) }
        }
        return candidates
    }

    /// Resolves the approval decision for a tool call: `true`, `false`, or `nil` when undecided.
    public func getApprovalStatus(
        toolName: String,
        callId: String,
        toolNamespace: String? = nil,
        existingPending: ToolApprovalItem? = nil,
        toolLookupKey: FunctionToolLookupKey? = nil
    ) throws -> Bool? {
        guard !callId.isEmpty else {
            throw ModelBehaviorError("Approval-gated tool calls require a non-empty call ID.")
        }
        let approvals = self.approvals
        if let existingPending, let hosted = Self.hostedApprovalKeys(existingPending) {
            if let sticky = hosted.sticky, let status = approvals[sticky]?.status(for: callId) {
                return status
            }
            return approvals[hosted.exact]?.perCallStatus(for: callId)
        }
        for candidate in candidateKeys(toolName: toolName, toolNamespace: toolNamespace, existingPending: existingPending, toolLookupKey: toolLookupKey) {
            if let status = approvals[candidate]?.status(for: callId) {
                return status
            }
        }
        return nil
    }

    /// Returns `true`/`false`/`nil` for the given tool call by tool name.
    public func isToolApproved(toolName: String, callId: String) -> Bool? {
        approvals[toolName]?.status(for: callId)
    }

    /// Returns a stored rejection message for a tool call if one exists.
    public func getRejectionMessage(
        toolName: String,
        callId: String,
        toolNamespace: String? = nil,
        existingPending: ToolApprovalItem? = nil,
        toolLookupKey: FunctionToolLookupKey? = nil
    ) -> String? {
        let approvals = self.approvals
        if let existingPending, let hosted = Self.hostedApprovalKeys(existingPending) {
            if let sticky = hosted.sticky, let record = approvals[sticky], record.status(for: callId) != nil {
                return record.rejectionMessage(for: callId)
            }
            return approvals[hosted.exact]?.rejectionMessage(for: callId)
        }
        for candidate in candidateKeys(toolName: toolName, toolNamespace: toolNamespace, existingPending: existingPending, toolLookupKey: toolLookupKey) {
            if let record = approvals[candidate], record.status(for: callId) != nil {
                return record.rejectionMessage(for: callId)
            }
        }
        return nil
    }

    func rebuildApprovals(_ json: JSONObject) {
        var approvals: [String: ApprovalRecord] = [:]
        for (key, value) in json {
            if case .object(let record) = value {
                approvals[key] = ApprovalRecord(json: record)
            }
        }
        self.approvals = approvals
    }

    var serializedApprovals: JSONObject {
        JSONObject(approvals.mapValues { .object($0.json) })
    }

    // MARK: - Forks

    /// Creates a child context that shares approvals and usage, with tool input set.
    public func forkWithToolInput(_ toolInput: (any Sendable)?) -> RunContextWrapper {
        RunContextWrapper(context: context, usageBox: usageBox, approvalsBox: approvalsBox, turnInput: turnInput, toolInput: toolInput)
    }

    /// Creates a child context that shares approvals and usage without tool input.
    public func forkWithoutToolInput() -> RunContextWrapper {
        RunContextWrapper(context: context, usageBox: usageBox, approvalsBox: approvalsBox, turnInput: turnInput, toolInput: nil)
    }

    /// Copies SDK-owned state for an independently resumable checkpoint.
    func copyForRunState() -> RunContextWrapper {
        let copy = RunContextWrapper(
            context: context,
            usageBox: SharedBox(usage),
            approvalsBox: SharedBox(approvals),
            turnInput: turnInput,
            toolInput: toolInput
        )
        return copy
    }
}

/// Context passed to agent hooks (`on_start`, `on_end`).
public final class AgentHookContext: RunContextWrapper, @unchecked Sendable {
    /// Creates a hook context that shares usage and approvals with `wrapper`.
    public init(from wrapper: RunContextWrapper) {
        super.init(
            context: wrapper.context,
            usageBox: wrapper.usageBox,
            approvalsBox: wrapper.approvalsBox,
            turnInput: wrapper.turnInput,
            toolInput: wrapper.toolInput
        )
    }
}
