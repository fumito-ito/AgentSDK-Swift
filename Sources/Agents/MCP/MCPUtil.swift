import Foundation

private let mcpFunctionToolNameMaxLength = 64
private let mcpFunctionToolHashLength = 8

/// Utilities for interop between MCP and Agents SDK tools (src/agents/mcp/util.py).
public enum MCPUtil {
    /// Gets all function tools from a list of MCP servers.
    public static func getAllFunctionTools(
        servers: [any MCPServer],
        convertSchemasToStrict: Bool,
        runContext: RunContextWrapper,
        agent: AgentBase,
        failureErrorFunction: FailureErrorFunction = .default,
        includeServerInToolNames: Bool = false,
        reservedToolNames: Set<String>? = nil
    ) async throws -> [any Tool] {
        var tools: [any Tool] = []
        var toolNames: Set<String> = []

        if includeServerInToolNames {
            var batches: [(Int, any MCPServer, [MCPTool])] = []
            for (index, server) in servers.enumerated() {
                batches.append((index, server, try await listToolsWithSpan(server, runContext: runContext, agent: agent)))
            }
            let overrides = buildPrefixedToolNameOverrides(batches, reservedNames: reservedToolNames ?? [])
            for (serverIndex, server, mcpTools) in batches {
                let names = mcpTools.indices.map { overrides[[serverIndex, $0]]! }
                let functionTools = try convertMCPToolsToFunctionTools(
                    mcpTools, server: server, convertSchemasToStrict: convertSchemasToStrict, agent: agent,
                    failureErrorFunction: failureErrorFunction, toolNameOverrides: names, serverIndex: serverIndex
                )
                let serverNames = Set(functionTools.map(\.name))
                let duplicates = serverNames.intersection(toolNames).sorted()
                if !duplicates.isEmpty {
                    throw UserError("Duplicate tool names found across MCP servers: \(duplicates.map { "'\($0)'" }.joined(separator: ", "))")
                }
                toolNames.formUnion(serverNames)
                tools.append(contentsOf: functionTools)
            }
            return tools
        }

        for (serverIndex, server) in servers.enumerated() {
            let serverTools = try await getFunctionTools(
                server: server, convertSchemasToStrict: convertSchemasToStrict, runContext: runContext,
                agent: agent, failureErrorFunction: failureErrorFunction, serverIndex: serverIndex
            )
            let serverNames = Set(serverTools.map(\.name))
            let duplicates = serverNames.intersection(toolNames).sorted()
            if !duplicates.isEmpty {
                throw UserError(
                    "Duplicate tool names found across MCP servers: \(duplicates.map { "'\($0)'" }.joined(separator: ", ")). "
                        + "Pass `include_server_in_tool_names=True` to `MCPUtil.get_all_function_tools()` or set "
                        + "`mcp_config={'include_server_in_tool_names': True}` on the agent to prefix tool names "
                        + "with their server name and avoid collisions."
                )
            }
            toolNames.formUnion(serverNames)
            tools.append(contentsOf: serverTools)
        }
        return tools
    }

    /// Gets all function tools from a single MCP server.
    public static func getFunctionTools(
        server: any MCPServer,
        convertSchemasToStrict: Bool,
        runContext: RunContextWrapper,
        agent: AgentBase,
        failureErrorFunction: FailureErrorFunction = .default,
        serverIndex: Int = 0
    ) async throws -> [FunctionTool] {
        let mcpTools = try await listToolsWithSpan(server, runContext: runContext, agent: agent)
        return try convertMCPToolsToFunctionTools(
            mcpTools, server: server, convertSchemasToStrict: convertSchemasToStrict, agent: agent,
            failureErrorFunction: failureErrorFunction, toolNameOverrides: nil, serverIndex: serverIndex
        )
    }

    private static func listToolsWithSpan(_ server: any MCPServer, runContext: RunContextWrapper, agent: AgentBase) async throws -> [MCPTool] {
        let span = mcpToolsSpan(server: getMCPServerLogName(server.name))
        return try await span.run {
            let tools = try await server.listTools(runContext: runContext, agent: agent)
            span.data(as: MCPListToolsSpanData.self)?.result = tools.map(\.name)
            return tools
        }
    }

    private static func convertMCPToolsToFunctionTools(
        _ tools: [MCPTool],
        server: any MCPServer,
        convertSchemasToStrict: Bool,
        agent: AgentBase,
        failureErrorFunction: FailureErrorFunction,
        toolNameOverrides: [String]?,
        serverIndex: Int
    ) throws -> [FunctionTool] {
        try tools.enumerated().map { index, tool in
            let functionTool = try toFunctionTool(
                tool, server: server, convertSchemasToStrict: convertSchemasToStrict, agent: agent,
                failureErrorFunction: failureErrorFunction, toolNameOverride: toolNameOverrides?[index]
            )
            functionTool.mcpToolBinding = (getMCPServerLogName(server.name), tool.name, serverIndex)
            return functionTool
        }
    }

    static func safeToolNamePart(_ value: String, fallback: String) -> String {
        let safe = String(value.map { char -> Character in
            char.isASCII && (char.isLetter || char.isNumber || char == "_" || char == "-") ? char : "_"
        })
        let trimmed = safe.trimmingCharacters(in: CharacterSet(charactersIn: "_-"))
        return trimmed.isEmpty ? fallback : trimmed
    }

    static func shortenToolName(_ baseName: String, seed: String, forceHash: Bool = false) -> String {
        if !forceHash, baseName.count <= mcpFunctionToolNameMaxLength { return baseName }
        let hashSuffix = String(sha1Hex(seed).prefix(mcpFunctionToolHashLength))
        let suffix = "_\(hashSuffix)"
        let stemLength = mcpFunctionToolNameMaxLength - suffix.count
        var stem = String(baseName.prefix(stemLength))
        while stem.hasSuffix("_") || stem.hasSuffix("-") { stem.removeLast() }
        return (stem.isEmpty ? "mcp" : stem) + suffix
    }

    static func buildPrefixedToolBaseName(serverName: String, toolName: String) -> String {
        "mcp_\(safeToolNamePart(serverName, fallback: "server"))__\(safeToolNamePart(toolName, fallback: "tool"))"
    }

    static func buildPrefixedToolNameOverrides(
        _ batches: [(Int, any MCPServer, [MCPTool])],
        reservedNames: Set<String>
    ) -> [[Int]: String] {
        var baseCounts: [String: Int] = [:]
        for (_, server, tools) in batches {
            for tool in tools {
                baseCounts[buildPrefixedToolBaseName(serverName: getMCPServerLogName(server.name), toolName: tool.name), default: 0] += 1
            }
        }
        struct Candidate {
            var key: [Int]
            var baseName: String
            var seed: String
            var initialName: String
        }
        var candidates: [Candidate] = []
        for (serverIndex, server, tools) in batches {
            let serverName = getMCPServerLogName(server.name)
            for (toolIndex, tool) in tools.enumerated() {
                let base = buildPrefixedToolBaseName(serverName: serverName, toolName: tool.name)
                let seed = "\(serverName)\0\(tool.name)"
                let force = (baseCounts[base] ?? 0) > 1 || reservedNames.contains(base)
                candidates.append(Candidate(key: [serverIndex, toolIndex], baseName: base, seed: seed, initialName: shortenToolName(base, seed: seed, forceHash: force)))
            }
        }
        var used = reservedNames
        var overrides: [[Int]: String] = [:]
        for candidate in candidates.sorted(by: { ($0.initialName, $0.seed, $0.key[0], $0.key[1]) < ($1.initialName, $1.seed, $1.key[0], $1.key[1]) }) {
            var publicName = candidate.initialName
            var collision = 1
            while used.contains(publicName) {
                publicName = shortenToolName(candidate.baseName, seed: "\(candidate.seed)\0\(collision)", forceHash: true)
                collision += 1
            }
            used.insert(publicName)
            overrides[candidate.key] = publicName
        }
        return overrides
    }

    /// Converts an MCP tool to an Agents SDK function tool.
    public static func toFunctionTool(
        _ tool: MCPTool,
        server: any MCPServer,
        convertSchemasToStrict: Bool,
        agent: AgentBase? = nil,
        failureErrorFunction: FailureErrorFunction = .default,
        toolNameOverride: String? = nil
    ) throws -> FunctionTool {
        let publicName = toolNameOverride ?? tool.name
        let staticMeta = tool.meta
        let inputSchemaIsEmpty = tool.inputSchema.isEmpty
        var schema = tool.inputSchema
        // MCP doesn't require `properties`, but OpenAI does.
        if schema["properties"] == nil { schema["properties"] = .object([:]) }
        var isStrict = false
        if convertSchemasToStrict {
            do {
                schema = try ensureStrictJSONSchema(schema, rejectOpenObjects: !inputSchemaIsEmpty)
                isStrict = true
            } catch {
                if AgentsDebug.dontLogToolData {
                    logger.info("Error converting MCP schema to strict mode")
                } else {
                    logger.info("Error converting MCP schema to strict mode: \(error)")
                }
            }
        }
        let effectiveFailure = server.failureErrorFunction ?? failureErrorFunction
        let functionTool = try FunctionTool.makeFailureHandlingTool(
            name: publicName,
            description: tool.description ?? "",
            paramsJSONSchema: schema,
            invoke: { context, input in
                try await invokeMCPTool(server: server, tool: tool, context: context, inputJSON: input, meta: staticMeta, toolDisplayName: publicName)
            },
            failureErrorFunction: effectiveFailure,
            strictJSONSchema: isStrict,
            needsApproval: server.needsApproval(for: tool, agent: agent),
            toolInputGuardrails: server.toolInputGuardrails,
            toolOutputGuardrails: server.toolOutputGuardrails,
            toolOrigin: ToolOrigin(type: .mcp, mcpServerName: getMCPServerLogName(server.name)),
            mcpTitle: tool.title,
            onHandledError: { tool, error, input, context in
                reportHandledFunctionToolError(tool: tool, error: error, inputJSON: input, context: context, spanMessageForJSONDecodeError: nil, logLabel: "MCP tool")
            }
        )
        functionTool.mcpToolBinding = (getMCPServerLogName(server.name), tool.name, nil)
        return functionTool
    }

    /// Invokes an MCP tool and returns the result as a tool output.
    public static func invokeMCPTool(
        server: any MCPServer,
        tool: MCPTool,
        context: RunContextWrapper,
        inputJSON: String,
        meta: JSONObject? = nil,
        toolDisplayName: String? = nil
    ) async throws -> any Sendable {
        let displayName = toolDisplayName ?? tool.name
        let parsed: JSONValue
        do {
            parsed = inputJSON.isEmpty ? .object([:]) : try JSONValue.parse(inputJSON)
        } catch {
            let message = "Invalid JSON input for tool \(displayName)"
            if AgentsDebug.dontLogToolData { throw ModelBehaviorError(message) }
            throw ModelBehaviorError("\(message): \(inputJSON)")
        }
        guard case .object(let arguments) = parsed else {
            throw ModelBehaviorError("Invalid JSON input for tool \(displayName): expected a JSON object")
        }
        logger.debug(AgentsDebug.dontLogToolData ? "Invoking MCP tool" : "Invoking MCP tool \(displayName) with input \(inputJSON)")

        let result: MCPCallToolResult
        do {
            var resolvedMeta: JSONObject?
            if let resolver = server.toolMetaResolver {
                resolvedMeta = try await resolver(MCPToolMetaContext(runContext: context, serverName: server.name, toolName: tool.name, arguments: arguments))
            }
            var mergedMeta: JSONObject?
            if resolvedMeta != nil || meta != nil {
                mergedMeta = (resolvedMeta ?? [:]).merging(meta ?? [:])
            }
            result = try await server.callTool(tool.name, arguments: arguments, meta: mergedMeta)
        } catch let error as UserError {
            throw error
        } catch let error as MCPToolCancellationError {
            throw error
        } catch is CancellationError {
            throw MCPToolCancellationError(
                "Failed to call tool '\(tool.name)' on MCP server '\(getMCPServerLogName(server.name))': tool execution was cancelled."
            )
        } catch let error as MCPError {
            if AgentsDebug.dontLogToolData {
                logger.warning("MCP tool returned an error.")
            } else {
                logger.warning("MCP tool \(displayName) on server '\(getMCPServerLogName(server.name))' returned an error: \(error.message)")
            }
            throw error
        } catch {
            logActionError("Error invoking MCP tool", error, isToolAction: true)
            throw AgentsException("Error invoking MCP tool \(displayName) on server '\(getMCPServerLogName(server.name))': \(describeError(error))")
        }

        logger.debug(AgentsDebug.dontLogToolData ? "MCP tool completed." : "MCP tool \(displayName) returned \(result.json)")

        let toolOutput: JSONValue
        if server.useStructuredContent, let structured = result.structuredContent, !structured.isEmpty, !result.isError {
            toolOutput = .string(JSONValue.object(structured).pythonJSONString())
        } else {
            let items: [JSONValue] = result.content.map { item in
                switch item.type {
                case "text":
                    return ["type": "text", "text": .string(item.string("text") ?? "")]
                case "image":
                    let mime = item.string("mimeType") ?? item.string("mime_type") ?? "image/png"
                    return ["type": "image", "image_url": .string("data:\(mime);base64,\(item.string("data") ?? "")")]
                default:
                    return ["type": "text", "text": .string(JSONValue.object(item).jsonString())]
                }
            }
            toolOutput = items.count == 1 ? items[0] : .array(items)
        }

        if let extractor = server.customDataExtractor {
            let customData = try await extractor(MCPToolCustomDataContext(
                runContext: context, serverName: server.name, toolName: tool.name, toolDisplayName: displayName,
                arguments: arguments, resultMeta: result.meta, structuredContent: result.structuredContent,
                isError: result.isError, toolOutput: toolOutput
            ))
            if let customData, !customData.isEmpty, let toolContext = context as? ToolContext {
                toolContext.customData = customData
            }
        }

        if let span = getCurrentSpan() {
            if let data = span.data(as: FunctionSpanData.self) {
                let includeSensitive = (context as? ToolContext)?.runConfig?.traceIncludeSensitiveData ?? true
                if includeSensitive { data.output = toolOutput }
                data.mcpData = ["server": .string(getMCPServerLogName(server.name))]
            } else {
                logger.warning("Current span is not a FunctionSpanData; skipping tool output")
            }
        }
        return toolOutput
    }
}

/// Returns the lowercase hex SHA-1 digest of a string.
func sha1Hex(_ text: String) -> String {
    var h0: UInt32 = 0x67452301, h1: UInt32 = 0xEFCDAB89, h2: UInt32 = 0x98BADCFE, h3: UInt32 = 0x10325476, h4: UInt32 = 0xC3D2E1F0
    var message = Array(text.utf8)
    let bitLength = UInt64(message.count) * 8
    message.append(0x80)
    while message.count % 64 != 56 { message.append(0) }
    for shift in stride(from: 56, through: 0, by: -8) { message.append(UInt8((bitLength >> UInt64(shift)) & 0xff)) }
    func rotl(_ x: UInt32, _ n: UInt32) -> UInt32 { (x << n) | (x >> (32 - n)) }
    for chunk in stride(from: 0, to: message.count, by: 64) {
        var w = [UInt32](repeating: 0, count: 80)
        for i in 0..<16 {
            w[i] = UInt32(message[chunk + i * 4]) << 24 | UInt32(message[chunk + i * 4 + 1]) << 16
                | UInt32(message[chunk + i * 4 + 2]) << 8 | UInt32(message[chunk + i * 4 + 3])
        }
        for i in 16..<80 { w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1) }
        var (a, b, c, d, e) = (h0, h1, h2, h3, h4)
        for i in 0..<80 {
            let (f, k): (UInt32, UInt32)
            switch i {
            case 0..<20: (f, k) = ((b & c) | (~b & d), 0x5A827999)
            case 20..<40: (f, k) = (b ^ c ^ d, 0x6ED9EBA1)
            case 40..<60: (f, k) = ((b & c) | (b & d) | (c & d), 0x8F1BBCDC)
            default: (f, k) = (b ^ c ^ d, 0xCA62C1D6)
            }
            let temp = rotl(a, 5) &+ f &+ e &+ k &+ w[i]
            e = d; d = c; c = rotl(b, 30); b = a; a = temp
        }
        h0 = h0 &+ a; h1 = h1 &+ b; h2 = h2 &+ c; h3 = h3 &+ d; h4 = h4 &+ e
    }
    return [h0, h1, h2, h3, h4].map { String(format: "%08x", $0) }.joined()
}
