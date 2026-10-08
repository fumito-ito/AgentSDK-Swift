import Foundation

// Port of src/agents/run_internal/run_loop.py (`run_single_turn`, `get_new_response`,
// `run_single_turn_streamed`) and run_internal/streaming.py.
//
// Swift: agent bindings (sandbox execution agents), prompt cache keys, and Responses compaction
// bookkeeping are not ported; see porting/manifest.json.

/// Emits run items as streaming events, skipping approval placeholders.
func streamStepItems(_ items: [RunItem], to result: RunResultStreaming) {
    for item in items {
        let name: String?
        switch item {
        case is MessageOutputItem: name = "message_output_created"
        case is HandoffCallItem: name = "handoff_requested"
        case is HandoffOutputItem: name = "handoff_occured"
        case is ToolCallItem: name = "tool_called"
        case is ToolSearchCallItem: name = "tool_search_called"
        case is ToolSearchOutputItem: name = "tool_search_output_created"
        case is ToolCallOutputItem: name = "tool_output"
        case is ReasoningItem: name = "reasoning_item_created"
        case is MCPApprovalRequestItem: name = "mcp_approval_requested"
        case is MCPApprovalResponseItem: name = "mcp_approval_response"
        case is MCPListToolsItem: name = "mcp_list_tools"
        case is ToolApprovalItem, is CompactionItem: name = nil
        default:
            logger.warning("Unexpected item type: \(type(of: item))")
            name = nil
        }
        if let name {
            result.emit(.runItem(RunItemStreamEvent(name: name, item: item)))
        }
    }
}

/// Notifies every model resolved during the run that its owning run has ended.
func cleanupModelsAfterRun(_ tracker: AgentToolUseTracker) async {
    for model in tracker.models {
        await model.cleanupOnRunEnd(owner: tracker)
    }
}

func prepareTurnInputItems(_ callerInput: RunInput, _ generatedItems: [RunItem], _ policy: ReasoningItemIdPolicy?) throws -> [TResponseInputItem] {
    let callerItems = ItemHelpers.inputToNewInputList(callerInput)
    let continuation = try runItemsToInputItems(generatedItems, reasoningItemIdPolicy: policy)
    return prepareModelInputItems(callerItems, continuation)
}

/// The trace name of a tool, or `nil` when it should not be listed.
func getToolTraceNameForTool(_ tool: any Tool) -> String? {
    if let functionTool = tool as? FunctionTool {
        return getFunctionToolTraceName(functionTool)
    }
    return tool.name
}

private func runAgentStartHooks(agent: Agent, hooks: RunHooks, contextWrapper: RunContextWrapper, turnInput: [TResponseInputItem]) async throws {
    let hookContext = AgentHookContext(from: contextWrapper)
    hookContext.turnInput = turnInput
    async let runHook: Void = hooks.onAgentStart(context: hookContext, agent: agent)
    async let agentHook: Void = agent.hooks?.onStart(context: hookContext, agent: agent) ?? ()
    _ = try await (runHook, agentHook)
}

private func runLLMStartHooks(agent: Agent, hooks: RunHooks, contextWrapper: RunContextWrapper, instructions: String?, input: [TResponseInputItem]) async throws {
    async let runHook: Void = hooks.onLLMStart(context: contextWrapper, agent: agent, systemPrompt: instructions, inputItems: input)
    async let agentHook: Void = agent.hooks?.onLLMStart(context: contextWrapper, agent: agent, systemPrompt: instructions, inputItems: input) ?? ()
    _ = try await (runHook, agentHook)
}

private func runLLMEndHooks(agent: Agent, hooks: RunHooks, contextWrapper: RunContextWrapper, response: ModelResponse) async throws {
    async let agentHook: Void = agent.hooks?.onLLMEnd(context: contextWrapper, agent: agent, response: response) ?? ()
    async let runHook: Void = hooks.onLLMEnd(context: contextWrapper, agent: agent, response: response)
    _ = try await (agentHook, runHook)
}

/// The tools, prompts, and handoffs resolved for one turn.
private struct TurnSetup {
    var allTools: [any Tool]
    var handoffs: [Handoff]
    var systemPrompt: String?
    var prompt: JSONObject?
    var outputSchema: (any AgentOutputSchemaBase)?
}

private func prepareTurn(
    agent: Agent,
    hooks: RunHooks,
    contextWrapper: RunContextWrapper,
    runConfig: RunConfig,
    turnInput: [TResponseInputItem],
    shouldRunAgentStartHooks: Bool,
    agentSpan: (any Span)?
) async throws -> TurnSetup {
    if shouldRunAgentStartHooks {
        try await runAgentStartHooks(agent: agent, hooks: hooks, contextWrapper: contextWrapper, turnInput: turnInput)
    }
    var allTools = try await agent.getAllTools(contextWrapper)
    allTools = try await initializeComputerTools(allTools, contextWrapper: contextWrapper)
    async let systemPrompt = agent.getSystemPrompt(contextWrapper)
    async let prompt = agent.getPrompt(contextWrapper)
    let resolvedSystemPrompt = try await systemPrompt
    let resolvedPrompt = try await prompt
    var handoffs = try await getHandoffs(agent, contextWrapper)
    (allTools, handoffs) = try resolveToolNameCollisions(allTools, handoffs: handoffs, collisionPolicy: runConfig.toolNameCollisionPolicy)
    if let data = agentSpan?.data(as: AgentSpanData.self) {
        data.handoffs = handoffs.map(\.agentName)
        data.tools = allTools.compactMap(getToolTraceNameForTool)
    }
    return TurnSetup(
        allTools: allTools,
        handoffs: handoffs,
        systemPrompt: resolvedSystemPrompt,
        prompt: resolvedPrompt,
        outputSchema: getOutputSchema(agent)
    )
}

/// Runs a single non-streaming turn of the agent loop.
func runSingleTurn(
    agent: Agent,
    originalInput: RunInput,
    generatedItems: [RunItem],
    hooks: RunHooks,
    contextWrapper: RunContextWrapper,
    runConfig: RunConfig,
    shouldRunAgentStartHooks: Bool,
    toolUseTracker: AgentToolUseTracker,
    serverConversationTracker: OpenAIServerConversationTracker? = nil,
    session: (any Session)? = nil,
    sessionItemsToRewind: [TResponseInputItem]? = nil,
    reasoningItemIdPolicy: ReasoningItemIdPolicy? = nil,
    errorHandlers: RunErrorHandlers? = nil,
    agentSpan: (any Span)? = nil
) async throws -> SingleStepResult {
    let turnInput = ItemHelpers.inputToNewInputList(originalInput)
    contextWrapper.turnInput = turnInput
    let setup = try await prepareTurn(
        agent: agent, hooks: hooks, contextWrapper: contextWrapper, runConfig: runConfig,
        turnInput: turnInput, shouldRunAgentStartHooks: shouldRunAgentStartHooks, agentSpan: agentSpan
    )
    let input: [TResponseInputItem]
    if let serverConversationTracker {
        input = serverConversationTracker.prepareInput(originalInput, generatedItems)
    } else {
        input = try prepareTurnInputItems(originalInput, generatedItems, reasoningItemIdPolicy)
    }
    let newResponse = try await getNewResponse(
        agent: agent, systemPrompt: setup.systemPrompt, input: input, outputSchema: setup.outputSchema,
        allTools: setup.allTools, handoffs: setup.handoffs, hooks: hooks, contextWrapper: contextWrapper,
        runConfig: runConfig, toolUseTracker: toolUseTracker, serverConversationTracker: serverConversationTracker,
        prompt: setup.prompt, session: session, sessionItemsToRewind: sessionItemsToRewind,
        deferLLMEndHooks: true
    )
    return try await getSingleStepResultFromResponse(
        agent: agent, allTools: setup.allTools, originalInput: originalInput, preStepItems: generatedItems,
        newResponse: newResponse, outputSchema: setup.outputSchema, handoffs: setup.handoffs, hooks: hooks,
        contextWrapper: contextWrapper, runConfig: runConfig, toolUseTracker: toolUseTracker,
        errorHandlers: errorHandlers, serverManagesConversation: serverConversationTracker != nil,
        afterInvocationValidation: { _ in
            try await runLLMEndHooks(agent: agent, hooks: hooks, contextWrapper: contextWrapper, response: newResponse)
        }
    )
}

/// Calls the model and returns the raw response, handling retries and hooks.
func getNewResponse(
    agent: Agent,
    systemPrompt: String?,
    input: [TResponseInputItem],
    outputSchema: (any AgentOutputSchemaBase)?,
    allTools: [any Tool],
    handoffs: [Handoff],
    hooks: RunHooks,
    contextWrapper: RunContextWrapper,
    runConfig: RunConfig,
    toolUseTracker: AgentToolUseTracker,
    serverConversationTracker: OpenAIServerConversationTracker?,
    prompt: JSONObject?,
    session: (any Session)? = nil,
    sessionItemsToRewind: [TResponseInputItem]? = nil,
    deferLLMEndHooks: Bool = false
) async throws -> ModelResponse {
    var filtered = try await maybeFilterModelInput(
        agent: agent, runConfig: runConfig, contextWrapper: contextWrapper,
        inputItems: input, systemInstructions: systemPrompt
    )
    filtered.input = deduplicateInputItemsPreferringLatest(filtered.input)

    let model = try getModel(agent, runConfig)
    toolUseTracker.recordModel(model)
    var modelSettings = getModelSettings(agent, runConfig)
    modelSettings = maybeResetToolChoice(agent, toolUseTracker, modelSettings)

    serverConversationTracker?.markInputAsSent(filtered.input)
    try await runLLMStartHooks(agent: agent, hooks: hooks, contextWrapper: contextWrapper, instructions: filtered.instructions, input: filtered.input)

    let previousResponseId = serverConversationTracker?.previousResponseId
    let conversationId = serverConversationTracker?.conversationId
    if let conversationId {
        logger.debug("Using conversation_id=\(conversationId)")
    } else {
        logger.debug("No conversation_id available for request")
    }

    let request = ModelRequest(
        systemInstructions: filtered.instructions,
        input: .items(filtered.input),
        modelSettings: modelSettings,
        tools: allTools,
        outputSchema: outputSchema,
        handoffs: handoffs,
        tracing: getModelTracingImpl(tracingDisabled: runConfig.tracingDisabled, traceIncludeSensitiveData: runConfig.traceIncludeSensitiveData),
        previousResponseId: previousResponseId,
        conversationId: conversationId,
        prompt: prompt
    )
    let filteredInput = filtered.input
    let newResponse = try await getResponseWithRetry(
        getResponse: { try await model.getResponse(request) },
        rewind: {
            guard let serverConversationTracker else { return }
            await rewindSessionItems(session, sessionItemsToRewind ?? [], serverConversationTracker)
            serverConversationTracker.rewindInput(filteredInput)
        },
        retrySettings: modelSettings.retry,
        getRetryAdvice: { model.getRetryAdvice($0) },
        previousResponseId: previousResponseId,
        conversationId: conversationId,
        timeout: modelSettings.timeout,
        replayUnsafeRequest: allTools.contains { $0 is ProgrammaticToolCallingTool }
    )
    if let serverConversationTracker {
        // Retry helpers rewind sent-input tracking before replaying; mark the input as delivered
        // again so subsequent turns only send new deltas.
        serverConversationTracker.markInputAsSent(filtered.input)
        serverConversationTracker.markInputAsAccepted(filtered.input)
        serverConversationTracker.trackServerItems(newResponse)
    }
    contextWrapper.addUsage(newResponse.usage)
    if !deferLLMEndHooks {
        try await runLLMEndHooks(agent: agent, hooks: hooks, contextWrapper: contextWrapper, response: newResponse)
    }
    return newResponse
}

/// Runs a single streamed turn and emits events as results arrive.
func runSingleTurnStreamed( // swiftlint:disable:this function_body_length
    streamedResult: RunResultStreaming,
    agent: Agent,
    hooks: RunHooks,
    contextWrapper: RunContextWrapper,
    runConfig: RunConfig,
    shouldRunAgentStartHooks: Bool,
    toolUseTracker: AgentToolUseTracker,
    serverConversationTracker: OpenAIServerConversationTracker? = nil,
    session: (any Session)? = nil,
    pendingServerItems: [RunItem]? = nil,
    reasoningItemIdPolicy: ReasoningItemIdPolicy? = nil,
    errorHandlers: RunErrorHandlers? = nil,
    agentSpan: (any Span)? = nil,
    beforeSideEffects: (() async throws -> Void)? = nil,
    persistInputIfNeeded: (() async throws -> Void)? = nil
) async throws -> SingleStepResult {
    let originalInput = streamedResult.input
    let turnInput = ItemHelpers.inputToNewInputList(originalInput)
    contextWrapper.turnInput = turnInput
    let setup = try await prepareTurn(
        agent: agent, hooks: hooks, contextWrapper: contextWrapper, runConfig: runConfig,
        turnInput: turnInput, shouldRunAgentStartHooks: shouldRunAgentStartHooks, agentSpan: agentSpan
    )
    streamedResult.update { $0.currentAgent = agent }

    let model = try getModel(agent, runConfig)
    toolUseTracker.recordModel(model)
    var modelSettings = getModelSettings(agent, runConfig)
    modelSettings = maybeResetToolChoice(agent, toolUseTracker, modelSettings)

    let modelInputItems = streamedResult.modelInputItems
    let input: [TResponseInputItem]
    if let serverConversationTracker {
        let items = (pendingServerItems?.isEmpty == false) ? pendingServerItems! : modelInputItems
        input = serverConversationTracker.prepareInput(originalInput, items)
    } else {
        input = try prepareTurnInputItems(originalInput, modelInputItems, reasoningItemIdPolicy)
    }
    var filtered = try await maybeFilterModelInput(
        agent: agent, runConfig: runConfig, contextWrapper: contextWrapper,
        inputItems: input, systemInstructions: setup.systemPrompt
    )
    filtered.input = deduplicateInputItemsPreferringLatest(filtered.input)
    serverConversationTracker?.markInputAsSent(filtered.input)
    try await runLLMStartHooks(agent: agent, hooks: hooks, contextWrapper: contextWrapper, instructions: filtered.instructions, input: filtered.input)
    try await persistInputIfNeeded?()

    let previousResponseId = serverConversationTracker?.previousResponseId
    let conversationId = serverConversationTracker?.conversationId
    let request = ModelRequest(
        systemInstructions: filtered.instructions,
        input: .items(filtered.input),
        modelSettings: modelSettings,
        tools: setup.allTools,
        outputSchema: setup.outputSchema,
        handoffs: setup.handoffs,
        tracing: getModelTracingImpl(tracingDisabled: runConfig.tracingDisabled, traceIncludeSensitiveData: runConfig.traceIncludeSensitiveData),
        previousResponseId: previousResponseId,
        conversationId: conversationId,
        prompt: setup.prompt
    )
    let filteredInput = filtered.input
    let failedAttempts = SharedBox(0)
    let stream = streamResponseWithRetry(
        getStream: { model.streamResponse(request) },
        rewind: { serverConversationTracker?.rewindInput(filteredInput) },
        retrySettings: modelSettings.retry,
        getRetryAdvice: { model.getRetryAdvice($0) },
        previousResponseId: previousResponseId,
        conversationId: conversationId,
        timeout: modelSettings.timeout,
        replayUnsafeRequest: setup.allTools.contains { $0 is ProgrammaticToolCallingTool },
        onFailedAttempts: { count in failedAttempts.mutate { $0 = count } }
    )

    var finalResponse: ModelResponse?
    var streamedOutput: [TResponseOutputItem] = []
    for try await event in stream {
        streamedResult.emit(.rawResponse(RawResponsesStreamEvent(data: event)))
        let eventType = event.type
        var terminal: ResponseObject?
        var isCompleted = false
        switch eventType {
        case "response.completed":
            isCompleted = true
            terminal = event["response"]?.objectValue.flatMap(ResponseObject.init(json:))
        case "response.incomplete", "response.failed":
            throw responseTerminalFailureError(eventType ?? "", event["response"]?.objectValue.flatMap(ResponseObject.init(json:)))
        case "error", "response.error":
            throw responseErrorEventFailureError(eventType ?? "", event)
        default:
            break
        }
        if let terminal {
            var output = terminal.output
            if isCompleted && output.isEmpty && !streamedOutput.isEmpty {
                // Some backends emit items only in item.done events and leave the terminal output empty.
                output = streamedOutput
            }
            // Always fold retry attempts into usage, even when the terminal response omits usage.
            let usage = applyRetryAttemptUsage(
                terminal.usage.map(responseUsageToUsageWithMetadata) ?? Usage(requests: requestsForResponseWithoutUsage(terminal.json)),
                failedAttempts: failedAttempts.value
            )
            finalResponse = ModelResponse(
                output: output,
                usage: usage,
                responseId: terminal.id,
                requestId: terminal.json.string(sseRequestIdKey),
                rawUsage: modelSettings.preserveRawUsage == true ? extractRawUsageSnapshot(terminal.json) : nil
            )
        }
        if eventType == "response.output_item.done", let item = event["item"]?.objectValue {
            streamedOutput.append(item)
        }
    }
    guard let finalResponse else {
        throw ModelBehaviorError("Model did not produce a final response!")
    }
    contextWrapper.addUsage(finalResponse.usage)
    if let serverConversationTracker {
        serverConversationTracker.markInputAsSent(filtered.input)
        serverConversationTracker.markInputAsAccepted(filtered.input)
        serverConversationTracker.trackServerItems(finalResponse)
    }

    let emitted = SharedBox<Set<ObjectIdentifier>>([])
    let result = try await getSingleStepResultFromResponse(
        agent: agent, allTools: setup.allTools, originalInput: originalInput, preStepItems: modelInputItems,
        newResponse: finalResponse, outputSchema: setup.outputSchema, handoffs: setup.handoffs, hooks: hooks,
        contextWrapper: contextWrapper, runConfig: runConfig, toolUseTracker: toolUseTracker,
        errorHandlers: errorHandlers, serverManagesConversation: serverConversationTracker != nil,
        afterInvocationValidation: { processed in
            if let processed {
                emitted.mutate { $0.formUnion(processed.newItems.map { ObjectIdentifier($0) }) }
                streamStepItems(processed.newItems, to: streamedResult)
            }
            try await runLLMEndHooks(agent: agent, hooks: hooks, contextWrapper: contextWrapper, response: finalResponse)
        },
        beforeSideEffects: beforeSideEffects
    )
    let alreadyEmitted = emitted.value
    let remaining = sessionItemsForTurn(result).filter { !alreadyEmitted.contains(ObjectIdentifier($0)) }
    streamStepItems(remaining, to: streamedResult)
    return result
}

// MARK: - Computer tools (tool_execution.py `initialize_computer_tools`, tool.py `dispose_resolved_computers`)

/// Provider-backed computer tools resolved per run context, for disposal at run end.
private let resolvedComputerTools = SharedBox<[ObjectIdentifier: [ComputerTool]]>([:])

/// Resolves computer tools and returns run-local copies for model invocation.
func initializeComputerTools(_ tools: [any Tool], contextWrapper: RunContextWrapper) async throws -> [any Tool] {
    var result: [any Tool] = []
    for tool in tools {
        guard let computerTool = tool as? ComputerTool, case .provider = computerTool.computer else {
            result.append(tool)
            continue
        }
        let resolved = try await computerTool.resolveComputer(runContext: contextWrapper)
        let key = ObjectIdentifier(contextWrapper.usageBox)
        resolvedComputerTools.mutate { registry in
            if !(registry[key] ?? []).contains(where: { $0 === computerTool }) {
                registry[key, default: []].append(computerTool)
            }
        }
        result.append(ComputerTool(computer: resolved, onSafetyCheck: computerTool.onSafetyCheck))
    }
    return result
}

/// Disposes any computer instances created for the provided run context.
func disposeResolvedComputers(runContext: RunContextWrapper) async {
    let key = ObjectIdentifier(runContext.usageBox)
    var tools: [ComputerTool] = []
    resolvedComputerTools.mutate { tools = $0.removeValue(forKey: key) ?? [] }
    for tool in tools {
        await tool.disposeResolvedComputers(runContext: runContext)
    }
}
