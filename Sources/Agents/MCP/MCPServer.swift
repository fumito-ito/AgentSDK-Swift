import Foundation

// Core-facing MCP protocols (porting/decisions.md D1). Concrete servers (stdio, SSE, streamable
// HTTP) live in the `AgentsMCP` target so the MCP Swift SDK dependency is opt-in.

/// An MCP tool definition (`mcp.types.Tool`), JSON-backed.
public struct MCPTool: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.string("name") != nil else { return nil }
        self.json = json
    }

    public init(name: String, description: String? = nil, inputSchema: JSONObject = ["type": "object", "properties": [:]], title: String? = nil, meta: JSONObject? = nil) {
        var json: JSONObject = ["name": .string(name), "inputSchema": .object(inputSchema)]
        if let description { json["description"] = .string(description) }
        if let title { json["title"] = .string(title) }
        if let meta { json["_meta"] = .object(meta) }
        self.json = json
    }

    public var name: String { json.string("name") ?? "" }
    public var description: String? { json.string("description") }
    public var title: String? { json.string("title") ?? json["annotations"]?["title"]?.stringValue }
    public var inputSchema: JSONObject { json["inputSchema"]?.objectValue ?? json["input_schema"]?.objectValue ?? [:] }
    public var meta: JSONObject? { json["_meta"]?.objectValue ?? json["meta"]?.objectValue }
}

/// The result of calling an MCP tool (`mcp.types.CallToolResult`), JSON-backed.
public struct MCPCallToolResult: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        self.json = json
    }

    public init(content: [JSONObject], structuredContent: JSONObject? = nil, isError: Bool = false, meta: JSONObject? = nil) {
        var json: JSONObject = ["content": .array(content.map(JSONValue.object)), "isError": .bool(isError)]
        if let structuredContent { json["structuredContent"] = .object(structuredContent) }
        if let meta { json["_meta"] = .object(meta) }
        self.json = json
    }

    public var content: [JSONObject] { json["content"]?.arrayValue?.compactMap(\.objectValue) ?? [] }
    public var structuredContent: JSONObject? { json["structuredContent"]?.objectValue ?? json["structured_content"]?.objectValue }
    public var isError: Bool { json["isError"]?.boolValue ?? json["is_error"]?.boolValue ?? false }
    public var meta: JSONObject? { json["_meta"]?.objectValue ?? json["meta"]?.objectValue }
}

/// Context passed to dynamic MCP tool filters.
public struct ToolFilterContext: Sendable {
    /// The current run context.
    public var runContext: RunContextWrapper
    /// The agent requesting the tools.
    public var agent: AgentBase
    /// The name of the MCP server.
    public var serverName: String

    public init(runContext: RunContextWrapper, agent: AgentBase, serverName: String) {
        self.runContext = runContext
        self.agent = agent
        self.serverName = serverName
    }
}

/// Context passed to MCP tool metadata resolvers.
public struct MCPToolMetaContext: Sendable {
    public var runContext: RunContextWrapper
    public var serverName: String
    public var toolName: String
    public var arguments: JSONObject?
}

/// Context passed to MCP tool custom data extractors.
public struct MCPToolCustomDataContext: Sendable {
    public var runContext: RunContextWrapper
    public var serverName: String
    public var toolName: String
    public var toolDisplayName: String
    public var arguments: JSONObject
    public var resultMeta: JSONObject?
    public var structuredContent: JSONObject?
    public var isError: Bool
    public var toolOutput: JSONValue
}

/// The approval policy for tools on an MCP server.
public enum MCPRequireApproval: Sendable {
    case always
    case never
    /// Per-tool policy by tool name; unlisted tools do not require approval.
    case perTool([String: Bool])
    /// Decides per tool from `(run_context, agent, tool)`.
    case dynamic(@Sendable (RunContextWrapper, AgentBase, MCPTool) async throws -> Bool)

    /// Builds a per-tool policy from `always`/`never` tool name lists.
    public static func toolLists(always: [String] = [], never: [String] = []) throws -> MCPRequireApproval {
        let overlapping = Set(always).intersection(never).sorted()
        if !overlapping.isEmpty {
            throw UserError("Invalid require_approval tool list policy: tool names cannot appear in both always and never: \(overlapping).")
        }
        var mapping: [String: Bool] = [:]
        for name in always { mapping[name] = true }
        for name in never { mapping[name] = false }
        return .perTool(mapping)
    }
}

/// Base interface for Model Context Protocol servers.
public protocol MCPServer: AnyObject, Sendable {
    /// A readable name for the server.
    var name: String { get }
    /// Whether to use `structuredContent` from tool results when present.
    var useStructuredContent: Bool { get }
    /// The approval policy for this server's tools.
    var requireApproval: MCPRequireApproval { get }
    /// Optional failure formatter overriding the agent-level configuration. `nil` means unset.
    var failureErrorFunction: FailureErrorFunction? { get }
    /// Produces MCP request metadata (`_meta`) for tool calls.
    var toolMetaResolver: (@Sendable (MCPToolMetaContext) async throws -> JSONObject?)? { get }
    /// Produces SDK-only custom data for emitted MCP tool output items.
    var customDataExtractor: (@Sendable (MCPToolCustomDataContext) async throws -> JSONObject?)? { get }
    /// Guardrails applied to every tool on this server before invocation.
    var toolInputGuardrails: [ToolInputGuardrail]? { get }
    /// Guardrails applied to every tool on this server after invocation.
    var toolOutputGuardrails: [ToolOutputGuardrail]? { get }

    /// Connects to the server (for example spawning a subprocess or opening a connection).
    func connect() async throws
    /// Cleans up the server (for example closing a subprocess or connection).
    func cleanup() async throws
    /// Lists the tools available on the server.
    func listTools(runContext: RunContextWrapper?, agent: AgentBase?) async throws -> [MCPTool]
    /// Invokes a tool on the server.
    func callTool(_ toolName: String, arguments: JSONObject?, meta: JSONObject?) async throws -> MCPCallToolResult
    /// Lists the prompts available on the server.
    func listPrompts() async throws -> JSONObject
    /// Gets a specific prompt from the server.
    func getPrompt(_ name: String, arguments: JSONObject?) async throws -> JSONObject
    /// The most recently fetched tools list, if available.
    var cachedTools: [MCPTool]? { get }
}

extension MCPServer {
    public var useStructuredContent: Bool { false }
    public var requireApproval: MCPRequireApproval { .never }
    public var failureErrorFunction: FailureErrorFunction? { nil }
    public var toolMetaResolver: (@Sendable (MCPToolMetaContext) async throws -> JSONObject?)? { nil }
    public var customDataExtractor: (@Sendable (MCPToolCustomDataContext) async throws -> JSONObject?)? { nil }
    public var toolInputGuardrails: [ToolInputGuardrail]? { nil }
    public var toolOutputGuardrails: [ToolOutputGuardrail]? { nil }
    public var cachedTools: [MCPTool]? { nil }

    public func listPrompts() async throws -> JSONObject {
        throw UserError("MCP server '\(name)' does not support list_prompts.")
    }

    public func getPrompt(_ name: String, arguments: JSONObject?) async throws -> JSONObject {
        throw UserError("MCP server '\(self.name)' does not support get_prompt.")
    }

    /// Returns the `FunctionTool.needsApproval` value for a given MCP tool.
    func needsApproval(for tool: MCPTool, agent: AgentBase?) -> ToolApprovalRequirement {
        switch requireApproval {
        case .always: return .always
        case .never: return .never
        case .perTool(let mapping): return .required(mapping[tool.name] ?? false)
        case .dynamic(let policy):
            guard let agent else { return .always }
            return .dynamic { runContext, _, _ in try await policy(runContext, agent, tool) }
        }
    }
}

/// Error raised by MCP transports for protocol-level failures (for example a tool not found).
public struct MCPError: Error, CustomStringConvertible, Sendable {
    public var code: Int?
    public var message: String
    public init(code: Int? = nil, message: String) {
        self.code = code
        self.message = message
    }
    public var description: String { message }
}

/// Returns a server name safe for logs (credentials removed from URLs).
func getMCPServerLogName(_ name: String) -> String {
    guard var components = URLComponents(string: name), components.scheme != nil, components.host != nil else { return name }
    components.user = nil
    components.password = nil
    return components.string ?? name
}
