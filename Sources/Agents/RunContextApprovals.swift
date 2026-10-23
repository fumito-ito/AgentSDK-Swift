import Foundation

// Port of the approval and tool invocation bookkeeping in src/agents/run_context.py.
//
// Swift: upstream keeps this state in mutable dictionaries on `RunContextWrapper` and shares them
// between forks. Here it is a value type stored in one lock-protected box, so concurrent tool
// calls read and update it atomically (porting/decisions.md D16).

private let reusedCallIdMessage = "Model reused a tool call ID for a different invocation. Use a unique call ID for each tool invocation."

/// An owner-bound function-tool approval key (upstream `_FunctionToolApprovalKey`).
///
/// Function-tool decisions belong to the agent whose approval item they answered. Keys compare by
/// the owner's token, which is unique per agent instance.
struct FunctionToolApprovalKey: Hashable, @unchecked Sendable {
    let ownerToken: String
    let toolKey: String
    /// The owning agent, kept weakly so a checkpoint does not retain the application's agent graph.
    weak var owner: Agent?

    init(owner: Agent, toolKey: String) {
        self.ownerToken = owner.functionApprovalToken
        self.toolKey = toolKey
        self.owner = owner
    }

    static func == (lhs: FunctionToolApprovalKey, rhs: FunctionToolApprovalKey) -> Bool {
        lhs.ownerToken == rhs.ownerToken && lhs.toolKey == rhs.toolKey
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(ownerToken)
        hasher.combine(toolKey)
    }
}

/// The key of one approval record (upstream `_ApprovalKey`).
///
/// Swift: upstream mixes strings, owner-bound string subclasses, and hosted MCP tuples in one
/// dictionary. The cases keep those key spaces apart, so a tool name can never alias a hosted key.
enum ApprovalKey: Hashable, Sendable {
    /// A tool-name key, used for native tools and legacy function decisions.
    case name(String)
    /// An agent-owned function-tool key.
    case function(FunctionToolApprovalKey)
    /// `("hosted_mcp", server_label, tool_name)`: a persistent hosted MCP decision.
    case hostedMCP(serverLabel: String, toolName: String)
    /// `("hosted_mcp_call", request_id)`: an exact hosted MCP request without a full identity.
    case hostedMCPCall(requestId: String)
    /// `("hosted_mcp_query", tool_name, request_id)`: supports name-based inspection only.
    case hostedMCPQuery(toolName: String, requestId: String)

    var nameValue: String? {
        if case .name(let name) = self { return name }
        return nil
    }
}

extension ApprovalKey: ExpressibleByStringLiteral {
    /// A string literal is a tool-name key, matching upstream's plain string keys.
    init(stringLiteral value: String) {
        self = .name(value)
    }
}

/// Tracks the canonical identity and lifecycle of one provider tool call ID (upstream
/// `_ToolInvocationRecord`).
struct ToolInvocationRecord: Sendable, Equatable {
    var invocationType: String
    var approvalScope: String
    var fingerprint: String
    var executed = false
    var completed = false

    var json: JSONObject {
        [
            "type": .string(invocationType),
            "approval_scope": .string(approvalScope),
            "fingerprint": .string(fingerprint),
            "executed": .bool(executed),
            "completed": .bool(completed),
        ]
    }

    func matches(_ identity: ToolInvocationIdentity) -> Bool {
        invocationType == identity.type && approvalScope == identity.approvalScope && fingerprint == identity.fingerprint
    }
}

/// The registered state of an invocation: its identity plus whether it ran and finished.
struct ToolInvocationStatus: Sendable, Equatable {
    var invocationType: String
    var callId: String
    var completed: Bool
    var executed: Bool
}

/// An approval record together with the key it is stored under. The key is `nil` for a sanitized
/// copy that must not be written back.
struct KeyedApprovalRecord {
    var key: ApprovalKey?
    var record: ApprovalRecord
}

extension ApprovalRecord.Decision {
    var isSticky: Bool {
        if case .all = self { return true }
        return false
    }

    func contains(_ callId: String) -> Bool {
        callIds?.contains(callId) == true
    }
}

extension ApprovalRecord {
    var isSticky: Bool { approved.isSticky || rejected.isSticky }

    func hasPerCallDecision(_ callId: String) -> Bool {
        approved.contains(callId) || rejected.contains(callId)
    }
}

/// SDK-owned tool approval state shared by a run context and its forks.
struct ToolApprovalState: Sendable {
    var approvals: [ApprovalKey: ApprovalRecord] = [:]
    var toolInvocations: [String: ToolInvocationRecord] = [:]
    /// Whether restored decisions come from a schema older than the invocation ledger.
    var allowLegacyApprovalBindingReconstruction = false
    /// Restored per-call decisions that have no ledger binding and need a fresh decision.
    var restoredUnboundApprovalCallIds: Set<String> = []

    // MARK: - Keys

    static func approvalKeysForScope(_ key: ApprovalKey, _ approvalScope: String?) -> [ApprovalKey] {
        if key == .name("apply_patch"), let approvalScope {
            return [.name("apply_patch:\(approvalScope)"), key]
        }
        return [key]
    }

    static func functionApprovalKey(_ toolKey: String, _ approvalItem: ToolApprovalItem?) -> ApprovalKey? {
        guard let approvalItem, toolInvocationCallId(approvalItem.rawItem)?.type == "function_call" else { return nil }
        return .function(FunctionToolApprovalKey(owner: approvalItem.agent, toolKey: toolKey))
    }

    private func hasPerCallDecision(_ callId: String) -> Bool {
        approvals.values.contains { $0.hasPerCallDecision(callId) }
    }

    /// Resolves an owner-bound execution query or an unambiguous name-only query.
    func approvalRecordsForKey(
        _ toolKey: String,
        _ approvalItem: ToolApprovalItem? = nil,
        callId: String? = nil,
        approvalScope: String? = nil
    ) -> [KeyedApprovalRecord] {
        let scopedKey = Self.functionApprovalKey(toolKey, approvalItem)
        var records: [KeyedApprovalRecord] = []
        if let scopedKey {
            if let record = approvals[scopedKey] { records.append(KeyedApprovalRecord(key: scopedKey, record: record)) }
        } else if approvalItem == nil {
            var scopedRecords = approvals.compactMap { key, record -> KeyedApprovalRecord? in
                guard case .function(let functionKey) = key, functionKey.toolKey == toolKey else { return nil }
                return KeyedApprovalRecord(key: key, record: record)
            }
            // An exact call can identify its owner even when sticky defaults cannot.
            if scopedRecords.count > 1, let callId {
                scopedRecords = scopedRecords.filter { $0.record.perCallStatus(for: callId) != nil }
            }
            if scopedRecords.count == 1 { records.append(contentsOf: scopedRecords) }
        }
        var legacy: KeyedApprovalRecord?
        if scopedKey == nil, let callId {
            legacy = approvalRecordForKey(toolKey, callId, approvalScope: approvalScope)
        } else {
            legacy = approvals[.name(toolKey)].map { KeyedApprovalRecord(key: .name(toolKey), record: $0) }
        }
        if var legacy {
            if scopedKey != nil {
                // Old snapshots have no trustworthy owner for a sticky function grant. Retain only
                // actual per-call decisions; never infer an owner on first use.
                legacy.key = nil
                if legacy.record.approved.isSticky { legacy.record.approved = .callIds([]) }
                if legacy.record.rejected.isSticky { legacy.record.rejected = .callIds([]) }
                legacy.record.stickyScope = nil
                legacy.record.stickyRejectionMessage = nil
            }
            records.append(legacy)
        }
        return records
    }

    func approvalRecordForKey(_ approvalKey: String, _ callId: String, approvalScope: String? = nil) -> KeyedApprovalRecord? {
        let approvalScope = approvalScope ?? toolInvocations[callId]?.approvalScope
        for scopedKey in Self.approvalKeysForScope(.name(approvalKey), approvalScope) {
            guard let record = approvals[scopedKey], record.status(for: callId) != nil else { continue }
            if record.perCallStatus(for: callId) != nil {
                return KeyedApprovalRecord(key: scopedKey, record: record)
            }
            if approvalScope == nil || record.stickyScope == approvalScope
                || (record.stickyScope == nil && allowLegacyApprovalBindingReconstruction) {
                return KeyedApprovalRecord(key: scopedKey, record: record)
            }
        }
        return nil
    }

    mutating func getOrCreateApprovalEntry(_ approvalKey: ApprovalKey, approvalScope: String? = nil) -> ApprovalKey {
        var approvalKey = approvalKey
        var entry = approvals[approvalKey]
        let scopedKey = Self.approvalKeysForScope(approvalKey, approvalScope)[0]
        if scopedKey != approvalKey {
            var otherScope = false
            if let entry {
                let stickyMismatch = entry.stickyScope != nil && entry.stickyScope != approvalScope
                let callIds = (entry.approved.callIds ?? []) + (entry.rejected.callIds ?? [])
                let invocationMismatch = callIds.contains { toolInvocations[$0].map { $0.approvalScope != approvalScope } ?? false }
                otherScope = stickyMismatch || invocationMismatch
            }
            if approvals[scopedKey] != nil || otherScope {
                approvalKey = scopedKey
                entry = approvals[approvalKey]
            }
        }
        if entry == nil { approvals[approvalKey] = ApprovalRecord() }
        return approvalKey
    }

    // MARK: - Invocation ledger

    /// Validates an invocation and returns its status when an approval decision applies.
    mutating func approvedToolInvocationStatus(
        _ rawItem: JSONObject,
        toolLookupKey: FunctionToolLookupKey? = nil,
        toolName: String? = nil,
        invocationRole: String? = nil,
        approvalItem: ToolApprovalItem? = nil
    ) throws -> ToolInvocationStatus? {
        guard let status = try toolInvocationStatus(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: invocationRole),
              let identity = toolInvocationIdentity(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: invocationRole) else {
            return nil
        }
        let stickyKeys = matchingStickyApprovalKeys(
            rawItem,
            toolLookupKey: toolLookupKey,
            toolName: toolName,
            approvalScope: identity.approvalScope,
            approvalItem: approvalItem
        )
        if !hasPerCallDecision(identity.callId), stickyKeys.isEmpty { return nil }
        return status
    }

    /// Validates and registers one canonical invocation for a provider call ID.
    mutating func toolInvocationStatus(
        _ rawItem: JSONObject,
        toolLookupKey: FunctionToolLookupKey? = nil,
        toolName: String? = nil,
        invocationRole: String? = nil
    ) throws -> ToolInvocationStatus? {
        let rawCallId = toolInvocationCallId(rawItem)?.callId
        let isRestoredUnbound = rawCallId.map(restoredUnboundApprovalCallIds.contains) ?? false
        guard let identity = toolInvocationIdentity(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: invocationRole) else {
            if isMCPApprovalInvocation(rawItem) { return nil }
            if let rawCallId, toolInvocations[rawCallId] != nil {
                throw ModelBehaviorError(reusedCallIdMessage)
            }
            return nil
        }
        let record: ToolInvocationRecord
        if let existing = toolInvocations[identity.callId] {
            guard existing.matches(identity) else { throw ModelBehaviorError(reusedCallIdMessage) }
            record = existing
        } else {
            if isRestoredUnbound { return nil }
            record = ToolInvocationRecord(invocationType: identity.type, approvalScope: identity.approvalScope, fingerprint: identity.fingerprint)
            toolInvocations[identity.callId] = record
        }
        if isRestoredUnbound { return nil }
        return ToolInvocationStatus(invocationType: identity.type, callId: identity.callId, completed: record.completed, executed: record.executed)
    }

    /// Replaces an unresolved invocation identity before execution begins.
    mutating func rebindToolInvocation(
        _ rawItem: JSONObject,
        previousIdentity: ToolInvocationIdentity,
        toolLookupKey: FunctionToolLookupKey? = nil,
        toolName: String? = nil,
        invocationRole: String? = nil
    ) throws -> ToolInvocationStatus? {
        guard let identity = toolInvocationIdentity(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: invocationRole) else {
            return nil
        }
        guard var record = toolInvocations[identity.callId] else {
            return try toolInvocationStatus(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: invocationRole)
        }
        if record.matches(identity) {
            return ToolInvocationStatus(invocationType: identity.type, callId: identity.callId, completed: record.completed, executed: record.executed)
        }
        let matchesPrevious = record.invocationType == previousIdentity.type
            && identity.callId == previousIdentity.callId
            && record.fingerprint == previousIdentity.fingerprint
        if !matchesPrevious || record.executed || record.completed {
            throw ModelBehaviorError(reusedCallIdMessage)
        }
        record.invocationType = identity.type
        record.approvalScope = identity.approvalScope
        record.fingerprint = identity.fingerprint
        toolInvocations[identity.callId] = record
        return ToolInvocationStatus(invocationType: identity.type, callId: identity.callId, completed: false, executed: false)
    }

    /// Returns sticky approval keys that independently authorize this tool identity.
    func matchingStickyApprovalKeys(
        _ rawItem: JSONObject,
        toolLookupKey: FunctionToolLookupKey?,
        toolName: String? = nil,
        approvalScope: String,
        approvalItem: ToolApprovalItem? = nil
    ) -> Set<ApprovalKey> {
        var mapping = rawItem
        if rawItem.type == "hosted_tool_call", case .object(let providerData)? = rawItem["provider_data"],
           providerData.type == "mcp_approval_request" {
            for (key, value) in providerData { mapping[key] = value }
        }
        guard let invocationType = mapping.type else { return [] }
        var toolName = toolName?.isEmpty == false ? toolName : mapping.string("name")
        let toolNamespace = mapping.string("namespace")
        var approvalKeys: [ApprovalKey]
        switch invocationType {
        case "function_call":
            approvalKeys = getFunctionToolApprovalKeys(
                toolName: toolName,
                toolNamespace: toolNamespace,
                toolLookupKey: toolLookupKey,
                includeLegacyDeferredKey: true
            ).map(ApprovalKey.name)
        case "mcp_approval_request":
            if let serverLabel = mapping.string("server_label"), let toolName {
                approvalKeys = [.hostedMCP(serverLabel: serverLabel, toolName: toolName)]
            } else {
                approvalKeys = []
            }
        default:
            if toolName == nil {
                toolName = [
                    "apply_patch_call": "apply_patch",
                    "computer_call": "computer",
                    "local_shell_call": "local_shell",
                    "shell_call": "shell",
                ][invocationType]
            }
            approvalKeys = toolName.map { $0.isEmpty ? [] : [.name($0)] } ?? []
        }

        var matching: Set<ApprovalKey> = []
        for approvalKey in approvalKeys {
            var approvalKey = approvalKey
            if invocationType == "function_call" {
                guard let name = approvalKey.nameValue, let scopedKey = Self.functionApprovalKey(name, approvalItem) else { continue }
                approvalKey = scopedKey
            }
            for candidate in Self.approvalKeysForScope(approvalKey, approvalScope) {
                if let record = approvals[candidate], record.isSticky, record.stickyScope == approvalScope {
                    matching.insert(candidate)
                }
            }
        }
        return matching
    }

    /// Marks a canonical invocation completed when its output is committed.
    mutating func markToolCallCompleted(_ rawItem: JSONObject) {
        guard let identity = toolOutputIdentity(rawItem), var record = toolInvocations[identity.callId],
              record.invocationType == identity.type else {
            return
        }
        record.executed = true
        record.completed = true
        toolInvocations[identity.callId] = record
    }

    /// Marks an invocation executed before the first user-code side effect.
    mutating func markToolInvocationExecuted(
        _ rawItem: JSONObject,
        toolLookupKey: FunctionToolLookupKey? = nil,
        toolName: String? = nil,
        invocationRole: String? = nil
    ) throws {
        guard let status = try toolInvocationStatus(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: invocationRole) else {
            return
        }
        toolInvocations[status.callId]?.executed = true
    }

    /// Rebuilds a missing binding from a serialized pending approval item.
    mutating func restorePendingApprovalBinding(_ approvalItem: ToolApprovalItem) throws {
        guard allowLegacyApprovalBindingReconstruction else { return }
        var approvalKeys = RunContextWrapper.resolveApprovalKeys(approvalItem).map(ApprovalKey.name)
        if let hosted = getHostedMCPApprovalRequestIdentity(approvalItem.rawItem), let requestId = hosted.requestId {
            if let identity = hosted.approvalIdentity {
                approvalKeys.append(.hostedMCP(serverLabel: identity.serverLabel, toolName: identity.toolName))
            } else {
                approvalKeys.append(.hostedMCPCall(requestId: requestId))
            }
        }
        let scope = toolInvocationApprovalScope(approvalItem.rawItem, toolLookupKey: approvalItem.toolLookupKey, toolName: approvalItem.toolName)?.scope
        if let scope {
            for key in approvalKeys {
                if var record = approvals[key], record.isSticky {
                    record.stickyScope = record.stickyScope ?? scope
                    approvals[key] = record
                }
            }
        }
        guard let callId = RunContextWrapper.resolveCallId(approvalItem) else { return }
        guard let identity = toolInvocationIdentity(approvalItem.rawItem, toolLookupKey: approvalItem.toolLookupKey, toolName: approvalItem.toolName) else {
            restoredUnboundApprovalCallIds.insert(callId)
            return
        }
        let hasMatchingDecision = approvalKeys.contains { key in
            guard let record = approvals[key] else { return false }
            let hasStickyDecision = record.stickyScope == identity.approvalScope && record.status(for: callId) != nil
            return record.hasPerCallDecision(callId) || hasStickyDecision
        }
        if hasMatchingDecision {
            _ = try toolInvocationStatus(approvalItem.rawItem, toolLookupKey: approvalItem.toolLookupKey, toolName: approvalItem.toolName)
        }
    }

    /// Remembers a current-schema pending call whose sticky binding was not restored.
    mutating func markRestoredUnboundPendingApproval(_ approvalItem: ToolApprovalItem) {
        guard !allowLegacyApprovalBindingReconstruction, let callId = RunContextWrapper.resolveCallId(approvalItem) else { return }
        guard let identity = toolInvocationIdentity(approvalItem.rawItem, toolLookupKey: approvalItem.toolLookupKey, toolName: approvalItem.toolName),
              identity.callId == callId else {
            restoredUnboundApprovalCallIds.insert(callId)
            return
        }
        let stickyKeys = matchingStickyApprovalKeys(
            approvalItem.rawItem,
            toolLookupKey: approvalItem.toolLookupKey,
            toolName: approvalItem.toolName,
            approvalScope: identity.approvalScope,
            approvalItem: approvalItem
        )
        if !stickyKeys.isEmpty || hasPerCallDecision(callId) {
            if toolInvocations[callId].map({ !$0.matches(identity) }) ?? true {
                restoredUnboundApprovalCallIds.insert(callId)
            }
        }
    }

    // MARK: - Hosted MCP

    static func resolveHostedMCPToolName(_ approvalItem: ToolApprovalItem, _ hostedRequest: HostedMCPApprovalRequestIdentity) -> String? {
        if let toolName = hostedRequest.toolName { return toolName }
        if let toolName = approvalItem.toolName, !toolName.isEmpty { return toolName }
        return nil
    }

    /// Resolves the authoritative hosted MCP record and whether it is exact-call-only.
    func resolveHostedMCPApprovalRecord(
        _ approvalItem: ToolApprovalItem,
        allowLegacyExact: Bool
    ) -> (record: KeyedApprovalRecord?, requestId: String?, exactCallOnly: Bool) {
        guard let hostedRequest = getHostedMCPApprovalRequestIdentity(approvalItem.rawItem), let requestId = hostedRequest.requestId else {
            return (nil, nil, true)
        }
        if let identity = hostedRequest.approvalIdentity {
            let key = ApprovalKey.hostedMCP(serverLabel: identity.serverLabel, toolName: identity.toolName)
            if let record = approvals[key], record.status(for: requestId) != nil {
                return (KeyedApprovalRecord(key: key, record: record), requestId, false)
            }
        } else {
            let key = ApprovalKey.hostedMCPCall(requestId: requestId)
            if let record = approvals[key], record.perCallStatus(for: requestId) != nil {
                return (KeyedApprovalRecord(key: key, record: record), requestId, true)
            }
        }
        guard allowLegacyExact, let legacyKey = Self.resolveHostedMCPToolName(approvalItem, hostedRequest),
              let legacyRecord = approvals[.name(legacyKey)], legacyRecord.perCallStatus(for: requestId) != nil else {
            return (nil, requestId, true)
        }
        return (KeyedApprovalRecord(key: .name(legacyKey), record: legacyRecord), requestId, true)
    }

    /// Returns a hosted MCP decision and its rejection message from one record.
    mutating func resolveHostedMCPApprovalDecision(
        _ approvalItem: ToolApprovalItem,
        allowLegacyExact: Bool = true
    ) -> (status: Bool?, rejectionMessage: String?) {
        let resolved = resolveHostedMCPApprovalRecord(approvalItem, allowLegacyExact: allowLegacyExact)
        guard let keyed = resolved.record, let requestId = resolved.requestId else { return (nil, nil) }
        let status: Bool?
        if resolved.exactCallOnly {
            status = keyed.record.perCallStatus(for: requestId)
        } else {
            status = keyed.record.status(for: requestId)
            if status != nil, keyed.record.stickyScope == nil, allowLegacyApprovalBindingReconstruction, let key = keyed.key,
               let scope = toolInvocationApprovalScope(approvalItem.rawItem, toolLookupKey: approvalItem.toolLookupKey, toolName: approvalItem.toolName) {
                approvals[key]?.stickyScope = scope.scope
            }
        }
        return (status, keyed.record.rejectionMessage(for: requestId))
    }

    // MARK: - Restoration

    /// Restores approvals from serialized state.
    mutating func rebuildApprovals(_ value: JSONValue?) {
        approvals = [:]
        guard case .object(let object)? = value else { return }
        for (toolName, recordValue) in object {
            guard case .object(let record) = recordValue else { continue }
            approvals[.name(toolName)] = ApprovalRecord(json: record)
        }
    }

    /// Restores the current-schema canonical tool invocation ledger.
    mutating func rebuildToolInvocations(_ value: JSONValue?, validationError: (String) -> any Error) throws {
        toolInvocations = [:]
        guard case .object(let invocations)? = value else {
            throw validationError("RunState tool_invocations must be a mapping.")
        }
        for (callId, serialized) in invocations {
            guard !callId.isEmpty else {
                throw validationError("RunState tool_invocations contains an invalid call ID.")
            }
            guard case .object(let invocation) = serialized else {
                throw validationError("RunState tool invocation must be a mapping.")
            }
            guard isToolInvocationType(invocation["type"]),
                  isToolInvocationDigest(invocation["approval_scope"]),
                  isToolInvocationDigest(invocation["fingerprint"]),
                  case .bool(let executed)? = invocation["executed"],
                  case .bool(let completed)? = invocation["completed"],
                  !completed || executed,
                  let invocationType = invocation.string("type"),
                  let approvalScope = invocation.string("approval_scope"),
                  let fingerprint = invocation.string("fingerprint") else {
                throw validationError("RunState tool invocation contains invalid lifecycle data.")
            }
            toolInvocations[callId] = ToolInvocationRecord(
                invocationType: invocationType,
                approvalScope: approvalScope,
                fingerprint: fingerprint,
                executed: executed,
                completed: completed
            )
        }
    }

    /// Requires reapproval for restored per-call decisions without a ledger binding.
    mutating func markRestoredUnboundApprovalCallIds() {
        for record in approvals.values {
            for decision in [record.approved, record.rejected] {
                for callId in decision.callIds ?? [] where toolInvocations[callId] == nil {
                    restoredUnboundApprovalCallIds.insert(callId)
                }
            }
        }
    }

    /// Restores typed hosted MCP approval records from serialized state.
    mutating func rebuildHostedMCPApprovals(_ value: JSONValue?) {
        guard case .array(let entries)? = value else { return }
        for entry in entries {
            guard case .object(let entry) = entry, case .object(let identity)? = entry["identity"],
                  case .object(let decision)? = entry["decision"] else {
                continue
            }
            let key: ApprovalKey
            switch identity.string("type") {
            case "server_tool":
                guard let serverLabel = identity.string("server_label"), !serverLabel.isEmpty,
                      let toolName = identity.string("tool_name"), !toolName.isEmpty else { continue }
                key = .hostedMCP(serverLabel: serverLabel, toolName: toolName)
            case "request":
                guard let requestId = identity.string("request_id"), !requestId.isEmpty else { continue }
                key = .hostedMCPCall(requestId: requestId)
            case "query":
                guard let toolName = identity.string("tool_name"), !toolName.isEmpty,
                      let requestId = identity.string("request_id"), !requestId.isEmpty else { continue }
                key = .hostedMCPQuery(toolName: toolName, requestId: requestId)
            default:
                continue
            }
            approvals[key] = ApprovalRecord(json: decision)
        }
    }
}

// MARK: - Run context entry points

extension RunContextWrapper {
    /// Validates and registers one canonical invocation (upstream `_tool_invocation_status`).
    @discardableResult
    func toolInvocationStatus(
        _ rawItem: JSONObject,
        toolLookupKey: FunctionToolLookupKey? = nil,
        toolName: String? = nil,
        invocationRole: String? = nil
    ) throws -> ToolInvocationStatus? {
        try toolStateBox.mutate {
            try $0.toolInvocationStatus(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: invocationRole)
        }
    }

    /// Returns an invocation's status when an approval decision applies to it.
    @discardableResult
    func approvedToolInvocationStatus(
        _ rawItem: JSONObject,
        toolLookupKey: FunctionToolLookupKey? = nil,
        toolName: String? = nil,
        invocationRole: String? = nil,
        approvalItem: ToolApprovalItem? = nil
    ) throws -> ToolInvocationStatus? {
        try toolStateBox.mutate {
            try $0.approvedToolInvocationStatus(
                rawItem,
                toolLookupKey: toolLookupKey,
                toolName: toolName,
                invocationRole: invocationRole,
                approvalItem: approvalItem
            )
        }
    }

    /// Replaces an unresolved invocation identity before execution begins.
    @discardableResult
    func rebindToolInvocation(
        _ rawItem: JSONObject,
        previousIdentity: ToolInvocationIdentity,
        toolLookupKey: FunctionToolLookupKey? = nil,
        toolName: String? = nil,
        invocationRole: String? = nil
    ) throws -> ToolInvocationStatus? {
        try toolStateBox.mutate {
            try $0.rebindToolInvocation(
                rawItem,
                previousIdentity: previousIdentity,
                toolLookupKey: toolLookupKey,
                toolName: toolName,
                invocationRole: invocationRole
            )
        }
    }

    /// Marks an invocation executed before the first user-code side effect.
    func markToolInvocationExecuted(
        _ rawItem: JSONObject,
        toolLookupKey: FunctionToolLookupKey? = nil,
        toolName: String? = nil,
        invocationRole: String? = nil
    ) throws {
        try toolStateBox.mutate {
            try $0.markToolInvocationExecuted(rawItem, toolLookupKey: toolLookupKey, toolName: toolName, invocationRole: invocationRole)
        }
    }

    /// Marks a canonical invocation completed when its output is committed.
    func markToolCallCompleted(_ rawItem: JSONObject) {
        toolStateBox.mutate { $0.markToolCallCompleted(rawItem) }
    }

    /// Rebuilds a missing binding from a serialized pending approval item.
    func restorePendingApprovalBinding(_ approvalItem: ToolApprovalItem) throws {
        try toolStateBox.mutate { try $0.restorePendingApprovalBinding(approvalItem) }
    }

    /// Returns a hosted MCP decision and its rejection message from one record.
    func resolveHostedMCPApprovalDecision(_ approvalItem: ToolApprovalItem, allowLegacyExact: Bool = true) -> (status: Bool?, rejectionMessage: String?) {
        toolStateBox.mutate { $0.resolveHostedMCPApprovalDecision(approvalItem, allowLegacyExact: allowLegacyExact) }
    }

    var allowLegacyApprovalBindingReconstruction: Bool {
        toolStateBox.value.allowLegacyApprovalBindingReconstruction
    }
}
