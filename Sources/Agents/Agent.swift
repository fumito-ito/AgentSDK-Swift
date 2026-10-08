import Foundation

/// The result of a custom tool-use behavior function.
public struct ToolsToFinalOutputResult: Sendable {
    /// Whether this is the final output. If `false`, the LLM runs again and receives the tool
    /// call output.
    public var isFinalOutput: Bool
    /// The final output. Must match the agent's output type when `isFinalOutput` is `true`.
    public var finalOutput: (any Sendable)?

    public init(isFinalOutput: Bool, finalOutput: (any Sendable)? = nil) {
        self.isFinalOutput = isFinalOutput
        self.finalOutput = finalOutput
    }
}

/// Decides whether tool results produce a final output.
public typealias ToolsToFinalOutputFunction = @Sendable (RunContextWrapper, [FunctionToolResult]) async throws -> ToolsToFinalOutputResult

/// Configures how tool use is handled.
public enum ToolUseBehavior: Sendable {
    /// Tools run, then the LLM receives the results and responds (the default).
    case runLLMAgain
    /// The output of the first tool call is used as the final output.
    case stopOnFirstTool
    /// The run stops if any of these tools is called; the first matching tool output is final.
    case stopAtTools([String])
    /// A custom function decides whether tool results are final.
    case custom(ToolsToFinalOutputFunction)
}

/// Configuration for MCP servers.
public struct MCPConfig: Sendable {
    /// Attempt to convert MCP schemas to strict-mode schemas (best effort). Defaults to `false`.
    public var convertSchemasToStrict: Bool
    /// Converts MCP tool failures into model-visible messages. `.none` raises errors instead.
    public var failureErrorFunction: FailureErrorFunction
    /// Expose local MCP tools with server-prefixed names to avoid collisions. Defaults to `false`.
    public var includeServerInToolNames: Bool

    public init(convertSchemasToStrict: Bool = false, failureErrorFunction: FailureErrorFunction = .default, includeServerInToolNames: Bool = false) {
        self.convertSchemasToStrict = convertSchemasToStrict
        self.failureErrorFunction = failureErrorFunction
        self.includeServerInToolNames = includeServerInToolNames
    }
}

/// A streaming event emitted when an agent is invoked as a tool.
public struct AgentToolStreamEvent: Sendable {
    /// The streaming event from the nested agent run.
    public var event: StreamEvent
    /// The nested agent emitting the event.
    public var agent: Agent
    /// The originating tool call, if available.
    public var toolCall: TResponseOutputItem?
}

/// The base class for `Agent` and `RealtimeAgent`.
///
/// Swift: `@unchecked Sendable` because upstream agents are mutable configuration objects.
/// Do not mutate an agent while a run that uses it is in progress.
open class AgentBase: @unchecked Sendable {
    /// The name of the agent.
    public var name: String
    /// A description of the agent, used when the agent is a handoff target.
    public var handoffDescription: String?
    /// The tools the agent can use.
    public var tools: [any Tool]
    /// MCP servers whose tools are included every time the agent runs. You must connect and
    /// clean up these servers yourself.
    public var mcpServers: [any MCPServer]
    /// Configuration for MCP servers.
    public var mcpConfig: MCPConfig

    public init(
        name: String,
        handoffDescription: String? = nil,
        tools: [any Tool] = [],
        mcpServers: [any MCPServer] = [],
        mcpConfig: MCPConfig = MCPConfig()
    ) {
        self.name = name
        self.handoffDescription = handoffDescription
        self.tools = tools
        self.mcpServers = mcpServers
        self.mcpConfig = mcpConfig
    }

    /// Names reserved by function tools and enabled handoffs, used to prefix MCP tool names.
    func mcpToolReservedNames(_ runContext: RunContextWrapper) async throws -> Set<String> {
        var reserved = Set(tools.compactMap { ($0 as? FunctionTool)?.name })
        if let agent = self as? Agent {
            for target in agent.handoffs {
                switch target {
                case .handoff(let handoff):
                    if try await handoff.isEnabled.resolve(runContext, self) {
                        reserved.insert(handoff.toolName)
                    }
                case .agent(let other):
                    reserved.insert(Handoff.defaultToolName(other))
                }
            }
        }
        return reserved
    }

    /// Fetches the available tools from the MCP servers.
    public func getMCPTools(_ runContext: RunContextWrapper) async throws -> [any Tool] {
        guard !mcpServers.isEmpty else { return [] }
        let reserved = mcpConfig.includeServerInToolNames ? try await mcpToolReservedNames(runContext) : nil
        return try await MCPUtil.getAllFunctionTools(
            servers: mcpServers,
            convertSchemasToStrict: mcpConfig.convertSchemasToStrict,
            runContext: runContext,
            agent: self,
            failureErrorFunction: mcpConfig.failureErrorFunction,
            includeServerInToolNames: mcpConfig.includeServerInToolNames,
            reservedToolNames: reserved
        )
    }

    /// All agent tools, including MCP tools and enabled function tools.
    public func getAllTools(_ runContext: RunContextWrapper) async throws -> [any Tool] {
        let snapshot = tools
        let mcpTools = try await getMCPTools(runContext)
        let enabled = try await withThrowingTaskGroup(of: (Int, Bool).self) { group in
            for (index, tool) in snapshot.enumerated() {
                group.addTask {
                    guard let functionTool = tool as? FunctionTool else { return (index, true) }
                    return (index, try await functionTool.isEnabled.resolve(runContext, self))
                }
            }
            var results = [Bool](repeating: false, count: snapshot.count)
            for try await (index, ok) in group {
                results[index] = ok
            }
            return results
        }
        let enabledTools = zip(snapshot, enabled).filter(\.1).map(\.0)
        return mcpTools + enabledTools
    }
}

/// An agent is an AI model configured with instructions, tools, guardrails, handoffs and more.
///
/// Passing `instructions`, the "system prompt" for the agent, is strongly recommended. You can
/// also pass `handoffDescription`, a human-readable description used when the agent is used
/// inside tools or handoffs.
public final class Agent: AgentBase, @unchecked Sendable {
    /// Instructions: a static string or a function of the context and agent.
    public enum Instructions: Sendable {
        case text(String)
        case dynamic(@Sendable (RunContextWrapper, Agent) async throws -> String)
    }

    /// The instructions for the agent, used as the system prompt.
    public var instructions: Instructions?
    /// A prompt object (or a function that returns one). Only usable with OpenAI Responses models.
    public var prompt: AgentPrompt?
    /// Sub-agents that the agent can delegate to.
    public var handoffs: [HandoffTarget]
    /// The model implementation to use. Defaults to `getDefaultModel()`.
    public var model: AgentModel?
    /// Model-specific tuning parameters.
    public var modelSettings: ModelSettings
    /// Checks that run in parallel to the agent's execution, before generating a response. Runs
    /// only if the agent is the first agent in the chain.
    public var inputGuardrails: [InputGuardrail]
    /// Checks that run on the final output of the agent.
    public var outputGuardrails: [OutputGuardrail]
    /// The output schema. `nil` means plain text output.
    public var outputType: (any AgentOutputSchemaBase)?
    /// Receives callbacks on lifecycle events for this agent.
    public var hooks: AgentHooks?
    /// How tool use is handled.
    public var toolUseBehavior: ToolUseBehavior
    /// Whether to reset the tool choice to the default after a tool has been called, to avoid
    /// infinite tool loops.
    public var resetToolChoice: Bool

    public init(
        name: String,
        handoffDescription: String? = nil,
        tools: [any Tool] = [],
        mcpServers: [any MCPServer] = [],
        mcpConfig: MCPConfig = MCPConfig(),
        instructions: String? = nil,
        prompt: AgentPrompt? = nil,
        handoffs: [HandoffTarget] = [],
        model: AgentModel? = nil,
        modelSettings: ModelSettings? = nil,
        inputGuardrails: [InputGuardrail] = [],
        outputGuardrails: [OutputGuardrail] = [],
        outputType: (any AgentOutputSchemaBase)? = nil,
        hooks: AgentHooks? = nil,
        toolUseBehavior: ToolUseBehavior = .runLLMAgain,
        resetToolChoice: Bool = true
    ) {
        self.instructions = instructions.map(Instructions.text)
        self.prompt = prompt
        self.handoffs = handoffs
        self.model = model
        self.modelSettings = modelSettings ?? Agent.initialModelSettings(for: model)
        self.inputGuardrails = inputGuardrails
        self.outputGuardrails = outputGuardrails
        self.outputType = outputType
        self.hooks = hooks
        self.toolUseBehavior = toolUseBehavior
        self.resetToolChoice = resetToolChoice
        super.init(name: name, handoffDescription: handoffDescription, tools: tools, mcpServers: mcpServers, mcpConfig: mcpConfig)
    }

    /// Creates an agent with dynamic instructions.
    public convenience init(
        name: String,
        handoffDescription: String? = nil,
        tools: [any Tool] = [],
        mcpServers: [any MCPServer] = [],
        mcpConfig: MCPConfig = MCPConfig(),
        instructions: @escaping @Sendable (RunContextWrapper, Agent) async throws -> String,
        prompt: AgentPrompt? = nil,
        handoffs: [HandoffTarget] = [],
        model: AgentModel? = nil,
        modelSettings: ModelSettings? = nil,
        inputGuardrails: [InputGuardrail] = [],
        outputGuardrails: [OutputGuardrail] = [],
        outputType: (any AgentOutputSchemaBase)? = nil,
        hooks: AgentHooks? = nil,
        toolUseBehavior: ToolUseBehavior = .runLLMAgain,
        resetToolChoice: Bool = true
    ) {
        self.init(
            name: name, handoffDescription: handoffDescription, tools: tools, mcpServers: mcpServers,
            mcpConfig: mcpConfig, instructions: nil, prompt: prompt, handoffs: handoffs, model: model,
            modelSettings: modelSettings, inputGuardrails: inputGuardrails, outputGuardrails: outputGuardrails,
            outputType: outputType, hooks: hooks, toolUseBehavior: toolUseBehavior, resetToolChoice: resetToolChoice
        )
        self.instructions = .dynamic(instructions)
    }

    static func initialModelSettings(for model: AgentModel?) -> ModelSettings {
        switch model {
        case nil: return getDefaultModelSettings()
        case .name(let name)?: return getDefaultModelSettings(name)
        case .instance?: return ModelSettings()
        }
    }

    /// Makes a copy of the agent with the given changes (`agent.clone(**kwargs)` upstream).
    ///
    /// The copy is shallow: lists such as `tools` and `handoffs` are copied as arrays, but the tool
    /// and handoff objects are shared. When `model` changes and the settings still match the
    /// implicit defaults of the previous model, the settings follow the new model's defaults.
    public func clone(_ modify: (Agent) -> Void = { _ in }) -> Agent {
        let copy = Agent(
            name: name,
            handoffDescription: handoffDescription,
            tools: tools,
            mcpServers: mcpServers,
            mcpConfig: mcpConfig,
            instructions: nil,
            prompt: prompt,
            handoffs: handoffs,
            model: model,
            modelSettings: modelSettings,
            inputGuardrails: inputGuardrails,
            outputGuardrails: outputGuardrails,
            outputType: outputType,
            hooks: hooks,
            toolUseBehavior: toolUseBehavior,
            resetToolChoice: resetToolChoice
        )
        copy.instructions = instructions
        let originalModelName = modelIdentity
        let settingsWereImplicit = modelSettings == Agent.initialModelSettings(for: model)
        modify(copy)
        if copy.modelIdentity != originalModelName, settingsWereImplicit, copy.modelSettings == modelSettings {
            copy.modelSettings = Agent.initialModelSettings(for: copy.model)
        }
        return copy
    }

    private var modelIdentity: String {
        switch model {
        case nil: return "<default>"
        case .name(let name)?: return "name:\(name)"
        case .instance(let instance)?: return "instance:\(ObjectIdentifier(instance).hashValue)"
        }
    }

    /// Returns the system prompt for the agent.
    public func getSystemPrompt(_ runContext: RunContextWrapper) async throws -> String? {
        switch instructions {
        case .text(let text)?: return text
        case .dynamic(let function)?: return try await function(runContext, self)
        case nil: return nil
        }
    }

    /// Returns the prompt for the agent.
    public func getPrompt(_ runContext: RunContextWrapper) async throws -> JSONObject? {
        try await PromptUtil.toModelInput(prompt, context: runContext, agent: self)
    }

    /// Transforms this agent into a tool, callable by other agents.
    ///
    /// This differs from handoffs in two ways: the new agent receives generated input rather than
    /// the conversation history, and the new agent is called as a tool while the original agent
    /// continues the conversation.
    ///
    /// - Parameters:
    ///   - toolName: The name of the tool. Defaults to the agent's name transformed into a valid
    ///     function name.
    ///   - toolDescription: What the tool does and when to use it.
    ///   - customOutputExtractor: Extracts the output from the nested run. When omitted, the final
    ///     output is used, falling back to the latest non-empty message or tool output.
    ///   - isEnabled: Whether the tool is enabled.
    ///   - onStream: Receives streaming events from the nested run; the nested agent then runs in
    ///     streaming mode.
    ///   - failureErrorFunction: Converts nested run failures into model-visible messages.
    ///   - needsApproval: Whether this agent tool pauses for approval.
    public func asTool(
        toolName: String? = nil,
        toolDescription: String? = nil,
        customOutputExtractor: (@Sendable (RunResultBase) async throws -> String)? = nil,
        isEnabled: ToolEnablement = .always,
        onStream: (@Sendable (AgentToolStreamEvent) async -> Void)? = nil,
        runConfig: RunConfig? = nil,
        maxTurns: Int? = nil,
        hooks: RunHooks? = nil,
        previousResponseId: String? = nil,
        conversationId: String? = nil,
        session: (any Session)? = nil,
        failureErrorFunction: FailureErrorFunction = .default,
        needsApproval: ToolApprovalRequirement = .never,
        parametersSchema: JSONObject? = nil,
        inputBuilder: StructuredToolInputBuilder? = nil,
        includeInputSchema: Bool = false
    ) throws -> FunctionTool {
        let resolvedName = toolName ?? transformStringFunctionStyle(name)
        let hasCustomParameters = parametersSchema != nil
        let includeSchema = includeInputSchema && hasCustomParameters
        let shouldCaptureToolInput = hasCustomParameters || includeSchema || inputBuilder != nil
        let paramsSchema = try ensureStrictJSONSchema(parametersSchema ?? JSONSchemaGenerator.schema(for: AgentAsToolInput.self))
        let schemaInfo = buildStructuredInputSchemaInfo(paramsSchema, includeJSONSchema: includeSchema)

        let invoke: @Sendable (ToolContext, String) async throws -> any Sendable = { context, inputJSON in
            let currentToolName = context.toolName.isEmpty ? resolvedName : context.toolName
            let json = try parseFunctionToolJSONInput(toolName: currentToolName, inputJSON: inputJSON)
            logFunctionToolInvocation(toolName: currentToolName, inputJSON: inputJSON)
            if !hasCustomParameters, json["input"]?.stringValue == nil {
                let base = "Invalid JSON input for tool \(currentToolName)"
                if AgentsDebug.dontLogToolData { throw ModelBehaviorError(base) }
                throw ModelBehaviorError("\(base): missing required string field 'input'")
            }
            let resolvedInput = try await resolveAgentToolInput(
                params: .object(json),
                schemaInfo: shouldCaptureToolInput ? schemaInfo : nil,
                inputBuilder: inputBuilder
            )
            let nestedContext = ToolContext.fromAgentContext(
                context,
                toolCallId: context.toolCallId,
                toolCall: context.toolCall,
                agent: context.agent,
                toolName: context.toolName,
                toolArguments: context.toolArguments,
                toolNamespace: context.toolNamespace,
                runConfig: runConfig ?? context.runConfig
            )
            if shouldCaptureToolInput {
                nestedContext.toolInput = JSONValue.object(json)
            }
            let resolvedMaxTurns = maxTurns ?? defaultMaxTurns
            let resolvedRunConfig = runConfig ?? context.runConfig ?? RunConfig()
            let result: RunResultBase
            if let onStream {
                let streaming = Runner.runStreamed(
                    self,
                    input: resolvedInput,
                    contextWrapper: nestedContext,
                    maxTurns: resolvedMaxTurns,
                    hooks: hooks,
                    runConfig: resolvedRunConfig,
                    previousResponseId: previousResponseId,
                    conversationId: conversationId,
                    session: session
                )
                var currentAgent = self
                for try await event in streaming.streamEvents() {
                    if case .agentUpdated(let updated) = event {
                        currentAgent = updated.newAgent
                    }
                    await onStream(AgentToolStreamEvent(event: event, agent: currentAgent, toolCall: context.toolCall))
                }
                result = streaming
            } else {
                result = try await Runner.run(
                    self,
                    input: resolvedInput,
                    contextWrapper: nestedContext,
                    maxTurns: resolvedMaxTurns,
                    hooks: hooks,
                    runConfig: resolvedRunConfig,
                    previousResponseId: previousResponseId,
                    conversationId: conversationId,
                    session: session
                )
            }
            if let customOutputExtractor {
                return try await customOutputExtractor(result)
            }
            if let output = result.finalOutput, !((output as? String)?.isEmpty ?? false) {
                return output
            }
            if !result.outputGuardrailResults.isEmpty {
                return result.finalOutput ?? ""
            }
            for item in result.newItems.reversed() {
                if let message = item as? MessageOutputItem {
                    let text = ItemHelpers.textMessageOutput(message)
                    if !text.isEmpty { return text }
                }
                if let output = item as? ToolCallOutputItem, let text = output.output as? String, !text.isEmpty {
                    return text
                }
            }
            return result.finalOutput ?? ""
        }

        let tool = try FunctionTool.makeFailureHandlingTool(
            name: resolvedName,
            description: toolDescription ?? "",
            paramsJSONSchema: paramsSchema,
            invoke: invoke,
            failureErrorFunction: failureErrorFunction,
            strictJSONSchema: true,
            isEnabled: isEnabled,
            needsApproval: needsApproval,
            toolOrigin: ToolOrigin(type: .agentAsTool, agentName: name, agentToolName: resolvedName)
        )
        tool.isAgentTool = true
        if toolName == nil {
            tool.agentToolDefaultIdentity = (name, resolvedName)
        }
        tool.agentInstance = self
        return tool
    }
}

extension Agent: CustomStringConvertible {
    public var description: String { "Agent(name=\"\(name)\", ...)" }
}
