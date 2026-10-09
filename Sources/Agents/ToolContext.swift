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
            approvalsBox: SharedBox([:]),
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
            approvalsBox: wrapper.approvalsBox,
            turnInput: wrapper.turnInput,
            toolInput: wrapper.toolInput
        )
    }

    /// The tool name qualified by namespace when available.
    public var qualifiedToolName: String {
        toolTraceName(toolName, namespace: toolNamespace) ?? toolName
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
