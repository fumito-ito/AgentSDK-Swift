import Foundation

/// The context of a tool call.
public final class ToolContext: RunContextWrapper, @unchecked Sendable {
    /// The name of the tool being invoked.
    public var toolName: String
    /// The ID of the tool call.
    public var toolCallId: String
    /// The raw arguments string of the tool call.
    public var toolArguments: String
    /// The tool call object associated with this invocation.
    public var toolCall: TResponseOutputItem?
    /// The Responses API namespace for this tool call, when present.
    public var toolNamespace: String?
    /// The active agent for this tool call, when available.
    public var agent: AgentBase?
    /// The active run config for this tool call, when available.
    public var runConfig: RunConfig?

    /// SDK-only custom data attached to the emitted output item by tool adapters.
    var customData: JSONObject?
    /// Whether the default failure formatter produced the latest invocation result.
    var defaultFailureHandled = false
    /// Arguments prepared during approval evaluation, consumed by the next invocation.
    var functionToolArguments: PreparedFunctionArguments?

    public init(
        context: any Sendable,
        usage: Usage = Usage(),
        toolName: String,
        toolCallId: String,
        toolArguments: String,
        toolCall: TResponseOutputItem? = nil,
        toolNamespace: String? = nil,
        agent: AgentBase? = nil,
        runConfig: RunConfig? = nil,
        turnInput: [TResponseInputItem] = [],
        toolInput: (any Sendable)? = nil
    ) {
        self.toolName = toolName
        self.toolCallId = toolCallId
        self.toolArguments = toolArguments
        self.toolCall = toolCall
        self.toolNamespace = toolNamespace ?? toolCall?.string("namespace")
        self.agent = agent
        self.runConfig = runConfig
        super.init(
            context: context,
            usageBox: SharedBox(usage),
            toolStateBox: SharedBox(ToolApprovalState()),
            turnInput: turnInput,
            toolInput: toolInput
        )
    }

    private init(
        sharingStateWith wrapper: RunContextWrapper,
        toolName: String,
        toolCallId: String,
        toolArguments: String,
        toolCall: TResponseOutputItem?,
        toolNamespace: String?,
        agent: AgentBase?,
        runConfig: RunConfig?
    ) {
        self.toolName = toolName
        self.toolCallId = toolCallId
        self.toolArguments = toolArguments
        self.toolCall = toolCall
        self.toolNamespace = toolNamespace
        self.agent = agent
        self.runConfig = runConfig
        super.init(
            context: wrapper.context,
            usageBox: wrapper.usageBox,
            toolStateBox: wrapper.toolStateBox,
            turnInput: wrapper.turnInput,
            toolInput: wrapper.toolInput
        )
        agentToolStateScope = wrapper.agentToolStateScope
    }

    /// Creates the context of a nested `Agent.asTool()` run: it shares usage with `context` but
    /// keeps its own approvals, so a nested run does not inherit decisions from its parent.
    init(isolatingApprovalsFrom context: ToolContext, runConfig: RunConfig?) {
        self.toolName = context.toolName
        self.toolCallId = context.toolCallId
        self.toolArguments = context.toolArguments
        self.toolCall = context.toolCall
        self.toolNamespace = context.toolNamespace
        self.agent = context.agent
        self.runConfig = runConfig
        super.init(
            context: context.context,
            usageBox: context.usageBox,
            toolStateBox: SharedBox(ToolApprovalState()),
            turnInput: [],
            toolInput: nil
        )
    }

    /// The tool name qualified by namespace when available.
    public var qualifiedToolName: String {
        toolTraceName(toolName, namespace: toolNamespace) ?? toolName
    }

    // MARK: - Nested approvals

    /// Finds a pending nested agent-tool context that owns an approval item.
    func findNestedApprovalTarget(_ approvalItem: ToolApprovalItem) throws -> (RunContextWrapper, ToolApprovalItem)? {
        guard let toolCall, let pendingResult = peekAgentToolRunResult(toolCall, scope: agentToolStateScope) else { return nil }
        let nestedContext = pendingResult.toState().context
        guard nestedContext !== self else { return nil }

        let targetIdentity = toolInvocationIdentity(
            approvalItem.rawItem, toolLookupKey: approvalItem.toolLookupKey, toolName: approvalItem.toolName
        )
        var currentContextOwnsApproval = false
        if let targetIdentity, let record = toolInvocations[targetIdentity.callId] {
            currentContextOwnsApproval = !record.completed && record.matches(targetIdentity)
        }

        var exactMatch: ToolApprovalItem?
        var canonicalMatches: [ToolApprovalItem] = []
        for candidate in pendingResult.interruptions {
            if candidate === approvalItem {
                exactMatch = candidate
                continue
            }
            let candidateIdentity = toolInvocationIdentity(
                candidate.rawItem, toolLookupKey: candidate.toolLookupKey, toolName: candidate.toolName
            )
            if let targetIdentity, candidateIdentity == targetIdentity {
                canonicalMatches.append(candidate)
            }
        }
        if currentContextOwnsApproval, exactMatch != nil || !canonicalMatches.isEmpty {
            throw UserError(
                "Cannot apply approval because the same tool invocation identity belongs to both "
                    + "the current run and a nested agent-tool run."
            )
        }
        if let exactMatch { return (nestedContext, exactMatch) }
        if canonicalMatches.count == 1 { return (nestedContext, canonicalMatches[0]) }
        if canonicalMatches.count > 1 {
            throw UserError(
                "Cannot apply approval because multiple nested agent-tool calls contain the same "
                    + "tool invocation identity. Use distinct call IDs."
            )
        }
        return nil
    }

    /// Approves this context's call or routes a surfaced nested approval to its owner.
    override public func approveTool(_ approvalItem: ToolApprovalItem, alwaysApprove: Bool = false) throws {
        guard let (nestedContext, nestedItem) = try findNestedApprovalTarget(approvalItem) else {
            try applyApproval(approvalItem, alwaysApprove: alwaysApprove)
            return
        }
        try nestedContext.applyApproval(nestedItem, alwaysApprove: alwaysApprove)
    }

    /// Rejects this context's call or routes a surfaced nested rejection to its owner.
    override public func rejectTool(_ approvalItem: ToolApprovalItem, alwaysReject: Bool = false, rejectionMessage: String? = nil) throws {
        guard let (nestedContext, nestedItem) = try findNestedApprovalTarget(approvalItem) else {
            try applyRejection(approvalItem, alwaysReject: alwaysReject, rejectionMessage: rejectionMessage)
            return
        }
        try nestedContext.applyRejection(nestedItem, alwaysReject: alwaysReject, rejectionMessage: rejectionMessage)
    }

    /// Creates a `ToolContext` that shares usage and approvals with `context`.
    ///
    /// Throws `ValueError` when neither `toolName`/`toolArguments` nor `toolCall` supplies them.
    public static func fromAgentContext(
        _ context: RunContextWrapper,
        toolCallId: String,
        toolCall: TResponseOutputItem? = nil,
        agent: AgentBase? = nil,
        toolName: String? = nil,
        toolArguments: String? = nil,
        toolNamespace: String? = nil,
        runConfig: RunConfig? = nil
    ) throws -> ToolContext {
        guard let resolvedName = toolName ?? toolCall.map({ $0.string("name") ?? "" }) else {
            throw ValueError("tool_name must be passed to ToolContext")
        }
        guard let resolvedArguments = toolArguments ?? toolCall.map({ $0.string("arguments") ?? "" }) else {
            throw ValueError("tool_arguments must be passed to ToolContext")
        }
        let parentToolContext = context as? ToolContext
        return ToolContext(
            sharingStateWith: context,
            toolName: resolvedName,
            toolCallId: toolCallId,
            toolArguments: resolvedArguments,
            toolCall: toolCall,
            toolNamespace: toolNamespace ?? toolCall?.string("namespace"),
            agent: agent ?? parentToolContext?.agent,
            runConfig: runConfig ?? parentToolContext?.runConfig
        )
    }
}
