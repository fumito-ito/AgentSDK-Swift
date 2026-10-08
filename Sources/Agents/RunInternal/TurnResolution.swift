import Foundation

// Port of src/agents/run_internal/turn_resolution.py.
//
// Swift: programmatic tool calling parent validation, invocation de-duplication across resumes,
// blocked-output replacement, and nested-history provenance are not ported; see
// porting/manifest.json.

/// Returns whether a model output item routes to a handoff instead of a tool. Namespaced calls
/// never resolve to a handoff.
func isHandoffToolCall(_ output: TResponseOutputItem, _ handoffNames: Set<String>) -> Bool {
    guard let call = ResponseFunctionToolCall(json: output) else { return false }
    return getToolCallQualifiedName(call.json) == call.name && handoffNames.contains(call.name)
}

/// Maps a model response to run items and local tool work.
func processModelResponse( // swiftlint:disable:this cyclomatic_complexity function_body_length
    agent: Agent,
    allTools: [any Tool],
    response: ModelResponse,
    outputSchema: (any AgentOutputSchemaBase)?,
    handoffs: [Handoff],
    runConfig: RunConfig? = nil
) throws -> ProcessedResponse {
    var items: [RunItem] = []
    var runHandoffs: [ToolRunHandoff] = []
    var functions: [ToolRunFunction] = []
    var computerActions: [ToolRunComputerAction] = []
    var customToolCalls: [ToolRunCustom] = []
    var localShellCalls: [ToolRunLocalShellCall] = []
    var shellCalls: [ToolRunShellCall] = []
    var applyPatchCalls: [ToolRunApplyPatchCall] = []
    var mcpApprovalRequests: [ToolRunMCPApprovalRequest] = []
    var functionsNotFound: [ToolRunFunctionNotFound] = []
    var toolsUsed: [String] = []

    var handoffMap: [String: Handoff] = [:]
    for handoff in handoffs { handoffMap[handoff.toolName] = handoff }
    let functionMap = try buildFunctionToolLookupMap(allTools.filter { $0 is FunctionTool })
    var customToolMap: [String: CustomTool] = [:]
    for case let tool as CustomTool in allTools { customToolMap[tool.name] = tool }
    let computerTool = allTools.lazy.compactMap { $0 as? ComputerTool }.first
    let localShellTool = allTools.lazy.compactMap { $0 as? LocalShellTool }.first
    let shellTool = allTools.lazy.compactMap { $0 as? ShellTool }.first
    let applyPatchTool = allTools.lazy.compactMap { $0 as? ApplyPatchTool }.first
    var hostedMCPServers: [String: HostedMCPTool] = [:]
    for case let tool as HostedMCPTool in allTools {
        if let label = tool.toolConfig.string("server_label") { hostedMCPServers[label] = tool }
    }

    func fail(_ message: String, spanMessage: String, data: JSONObject = [:]) -> ModelBehaviorError {
        attachErrorToCurrentSpan(SpanError(message: spanMessage, data: data))
        return ModelBehaviorError(message)
    }

    for output in response.output {
        let outputType = output.type
        logger.debug("Processing output item type=\(outputType ?? "None")")
        switch outputType {
        case "shell_call":
            let raw = output.removing("created_by")
            guard let shellTool else {
                toolsUsed.append("shell")
                throw fail("Model produced shell call without a shell tool.", spanMessage: "Shell tool not found")
            }
            let item = ToolCallItem(agent: agent, rawItem: raw)
            item.resolvedToolName = shellTool.name
            items.append(item)
            toolsUsed.append(shellTool.name)
            guard shellTool.environment.string("type") == "local" else { continue }
            guard shellTool.executor != nil else {
                throw fail("Model produced local shell call without a local shell executor.", spanMessage: "Local shell executor not found")
            }
            shellCalls.append(ToolRunShellCall(toolCall: output, shellTool: shellTool))
        case "shell_call_output":
            toolsUsed.append(shellTool?.name ?? "shell")
            var raw = output.removing("created_by")
            if case .array(let chunks)? = raw["output"] {
                raw["output"] = .array(chunks.map { chunk in
                    guard case .object(let object) = chunk else { return chunk }
                    return .object(object.removing("created_by"))
                })
            }
            items.append(ToolCallOutputItem(agent: agent, rawItem: raw, output: raw["output"] ?? .null))
        case "apply_patch_call":
            let raw = output.removing("created_by")
            guard let applyPatchTool else {
                toolsUsed.append("apply_patch")
                throw fail("Model produced apply_patch call without an apply_patch tool.", spanMessage: "Apply patch tool not found")
            }
            let item = ToolCallItem(agent: agent, rawItem: raw)
            item.resolvedToolName = applyPatchTool.name
            items.append(item)
            toolsUsed.append(applyPatchTool.name)
            applyPatchCalls.append(ToolRunApplyPatchCall(toolCall: raw, applyPatchTool: applyPatchTool))
        case "compaction":
            items.append(CompactionItem(agent: agent, rawItem: output.removing("created_by")))
        case "tool_search_call":
            if output.string("execution") == "client" {
                throw ModelBehaviorError(
                    "Client-executed tool_search calls are not supported by the standard agent runner. Handle the "
                        + "tool_search_call yourself and return a matching tool_search_output item with the same call_id."
                )
            }
            items.append(ToolSearchCallItem(agent: agent, rawItem: output))
            toolsUsed.append("tool_search")
        case "tool_search_output":
            items.append(ToolSearchOutputItem(agent: agent, rawItem: output))
            toolsUsed.append("tool_search")
        case "message":
            items.append(MessageOutputItem(agent: agent, rawItem: output))
        case "file_search_call":
            items.append(ToolCallItem(agent: agent, rawItem: output))
            toolsUsed.append("file_search")
        case "web_search_call":
            items.append(ToolCallItem(agent: agent, rawItem: output))
            toolsUsed.append("web_search")
        case "reasoning":
            items.append(ReasoningItem(agent: agent, rawItem: output))
        case "computer_call":
            guard let computerTool, let call = ResponseComputerToolCall(json: output) else {
                toolsUsed.append("computer")
                throw fail("Model produced computer action without a computer tool.", spanMessage: "Computer tool not found")
            }
            let item = ToolCallItem(agent: agent, rawItem: output)
            item.resolvedToolName = computerTool.name
            items.append(item)
            toolsUsed.append(computerTool.name)
            computerActions.append(ToolRunComputerAction(toolCall: call, computerTool: computerTool))
        case "mcp_approval_request":
            guard let request = McpApprovalRequest(json: output) else { continue }
            guard let server = hostedMCPServers[request.serverLabel] else {
                throw fail("MCP server label \(request.serverLabel) not found", spanMessage: "MCP server label not found", data: ["server_label": .string(request.serverLabel)])
            }
            items.append(MCPApprovalRequestItem(agent: agent, rawItem: output))
            mcpApprovalRequests.append(ToolRunMCPApprovalRequest(requestItem: request, mcpTool: server))
            if server.onApprovalRequest == nil {
                logger.debug("Hosted MCP server \(request.serverLabel) has no on_approval_request hook; approvals will be surfaced as interruptions for the caller to handle.")
            }
        case "mcp_list_tools":
            items.append(MCPListToolsItem(agent: agent, rawItem: output))
        case "mcp_call":
            let item = ToolCallItem(
                agent: agent,
                rawItem: output,
                toolOrigin: ToolOrigin(type: .mcp, mcpServerName: output.string("server_label"))
            )
            items.append(item)
            toolsUsed.append("mcp")
        case "image_generation_call":
            items.append(ToolCallItem(agent: agent, rawItem: output))
            toolsUsed.append("image_generation")
        case "code_interpreter_call":
            items.append(ToolCallItem(agent: agent, rawItem: output))
            toolsUsed.append("code_interpreter")
        case "local_shell_call":
            guard let call = LocalShellCall(json: output) else { continue }
            if let localShellTool {
                let item = ToolCallItem(agent: agent, rawItem: output)
                item.resolvedToolName = localShellTool.name
                items.append(item)
                toolsUsed.append("local_shell")
                localShellCalls.append(ToolRunLocalShellCall(toolCall: call, localShellTool: localShellTool))
            } else if let shellTool {
                let item = ToolCallItem(agent: agent, rawItem: output)
                item.resolvedToolName = shellTool.name
                items.append(item)
                toolsUsed.append(shellTool.name)
                shellCalls.append(ToolRunShellCall(toolCall: output, shellTool: shellTool))
            } else {
                toolsUsed.append("local_shell")
                throw fail("Model produced local shell call without a local shell tool.", spanMessage: "Local shell tool not found")
            }
        case "custom_tool_call":
            guard let call = ResponseCustomToolCall(json: output) else { continue }
            if let customTool = customToolMap[call.name] {
                items.append(ToolCallItem(agent: agent, rawItem: output))
                toolsUsed.append(customTool.name)
                customToolCalls.append(ToolRunCustom(toolCall: call, customTool: customTool))
            } else if isApplyPatchName(call.name, applyPatchTool), let pseudo = try normalizeApplyPatchFallbackCall(output) {
                guard let applyPatchTool else {
                    toolsUsed.append("apply_patch")
                    throw fail("Model produced apply_patch call without an apply_patch tool.", spanMessage: "Apply patch tool not found")
                }
                let item = ToolCallItem(agent: agent, rawItem: pseudo)
                item.resolvedToolName = applyPatchTool.name
                items.append(item)
                toolsUsed.append(applyPatchTool.name)
                applyPatchCalls.append(ToolRunApplyPatchCall(toolCall: pseudo, applyPatchTool: applyPatchTool))
            } else {
                items.append(ToolCallItem(agent: agent, rawItem: output))
                throw fail("Tool \(call.name) not found in agent \(agent.name)", spanMessage: "Custom tool not found", data: ["tool_name": .string(call.name)])
            }
        case "function_call":
            guard let call = ResponseFunctionToolCall(json: output) else { continue }
            let lookupKey = getFunctionToolLookupKeyForCall(call.json)
            if isApplyPatchName(call.name, applyPatchTool), lookupKey.flatMap({ functionMap[$0] }) == nil,
               let pseudo = try normalizeApplyPatchFallbackCall(output) {
                guard let applyPatchTool else {
                    toolsUsed.append("apply_patch")
                    throw fail("Model produced apply_patch call without an apply_patch tool.", spanMessage: "Apply patch tool not found")
                }
                let item = ToolCallItem(agent: agent, rawItem: pseudo)
                item.resolvedToolName = applyPatchTool.name
                items.append(item)
                toolsUsed.append(applyPatchTool.name)
                applyPatchCalls.append(ToolRunApplyPatchCall(toolCall: pseudo, applyPatchTool: applyPatchTool))
                continue
            }
            toolsUsed.append(getToolCallTraceName(call.json) ?? call.name)
            let qualifiedName = getToolCallQualifiedName(call.json)
            if isHandoffToolCall(output, Set(handoffMap.keys)), let handoff = handoffMap[call.name] {
                items.append(HandoffCallItem(agent: agent, rawItem: output))
                runHandoffs.append(ToolRunHandoff(handoff: handoff, toolCall: call))
                continue
            }
            guard let functionTool = lookupKey.flatMap({ functionMap[$0] }) else {
                attachErrorToCurrentSpan(SpanError(message: "Tool not found", data: ["tool_name": .string(qualifiedName ?? call.name)]))
                if runConfig?.toolNotFoundBehavior == .returnErrorToModel {
                    items.append(ToolCallItem(agent: agent, rawItem: output))
                    functionsNotFound.append(ToolRunFunctionNotFound(toolCall: call, toolName: qualifiedName ?? call.name))
                    continue
                }
                throw ModelBehaviorError("Tool \(qualifiedName ?? call.name) not found in agent \(agent.name)")
            }
            items.append(ToolCallItem(
                agent: agent,
                rawItem: output,
                description: functionTool.description,
                title: functionTool.mcpTitle,
                toolOrigin: getFunctionToolOrigin(functionTool)
            ))
            functions.append(ToolRunFunction(toolCall: call, functionTool: functionTool))
        default:
            logger.warning("Unexpected output type, ignoring: \(outputType ?? "None")")
        }
    }

    return ProcessedResponse(
        newItems: items,
        handoffs: runHandoffs,
        functions: functions,
        computerActions: computerActions,
        localShellCalls: localShellCalls,
        shellCalls: shellCalls,
        applyPatchCalls: applyPatchCalls,
        toolsUsed: toolsUsed,
        mcpApprovalRequests: mcpApprovalRequests,
        interruptions: [],
        functionToolsNotFound: functionsNotFound,
        customToolCalls: customToolCalls
    )
}

/// Runs agent end hooks for a final output.
func runFinalOutputHooks(agent: Agent, hooks: RunHooks, contextWrapper: RunContextWrapper, finalOutput: any Sendable) async throws {
    let hookContext = AgentHookContext(from: contextWrapper)
    async let runHook: Void = hooks.onAgentEnd(context: hookContext, agent: agent, output: finalOutput)
    async let agentHook: Void = agent.hooks?.onEnd(context: hookContext, agent: agent, output: finalOutput) ?? ()
    _ = try await (runHook, agentHook)
}

/// Finalizes a turn once final output is known and runs end hooks.
func executeFinalOutput(
    agent: Agent,
    originalInput: RunInput,
    newResponse: ModelResponse,
    preStepItems: [RunItem],
    newStepItems: [RunItem],
    finalOutput: any Sendable,
    hooks: RunHooks,
    contextWrapper: RunContextWrapper,
    toolInputGuardrailResults: [ToolInputGuardrailResult],
    toolOutputGuardrailResults: [ToolOutputGuardrailResult]
) async throws -> SingleStepResult {
    try await runFinalOutputHooks(agent: agent, hooks: hooks, contextWrapper: contextWrapper, finalOutput: finalOutput)
    return SingleStepResult(
        originalInput: originalInput,
        modelResponse: newResponse,
        preStepItems: preStepItems,
        newStepItems: newStepItems,
        nextStep: .finalOutput(finalOutput),
        toolInputGuardrailResults: toolInputGuardrailResults,
        toolOutputGuardrailResults: toolOutputGuardrailResults
    )
}

/// Determines whether tool results produce a final output.
func checkForFinalOutputFromTools(_ agent: Agent, _ results: [FunctionToolResult], _ contextWrapper: RunContextWrapper) async throws -> ToolsToFinalOutputResult {
    guard !results.isEmpty else { return notFinalOutput }
    switch agent.toolUseBehavior {
    case .runLLMAgain:
        return notFinalOutput
    case .stopOnFirstTool:
        return ToolsToFinalOutputResult(isFinalOutput: true, finalOutput: results[0].output)
    case .stopAtTools(let names):
        for result in results where names.contains(result.tool.name) || names.contains(result.tool.qualifiedName) {
            return ToolsToFinalOutputResult(isFinalOutput: true, finalOutput: result.output)
        }
        return ToolsToFinalOutputResult(isFinalOutput: false)
    case .custom(let function):
        return try await function(contextWrapper, results)
    }
}

private func defaultToolNotFoundMessage(_ toolName: String) -> String {
    "Tool '\(toolName)' not found."
}

private func buildToolNotFoundOutputItems(agent: Agent, calls: [ToolRunFunctionNotFound], contextWrapper: RunContextWrapper, runConfig: RunConfig) async throws -> [RunItem] {
    var items: [RunItem] = []
    for call in calls {
        var message = defaultToolNotFoundMessage(call.toolName)
        if let formatter = runConfig.toolErrorFormatter {
            do {
                message = try await formatter(ToolErrorFormatterArgs(
                    kind: "tool_not_found", toolType: "function", toolName: call.toolName, callId: call.toolCall.callId,
                    defaultMessage: message, runContext: contextWrapper
                )) ?? message
            } catch {
                logActionError("Tool error formatter failed for missing tool", error, isToolAction: true)
            }
        }
        items.append(ToolCallOutputItem(agent: agent, rawItem: try ItemHelpers.toolCallOutputItem(call.toolCall, output: message), output: message))
    }
    return items
}

private func resolveServerManagedHandoffBehavior(
    fromAgent: Agent,
    toAgent: AgentBase,
    serverManagesConversation: Bool,
    inputFilter: HandoffInputFilter?,
    shouldNestHistory: Bool
) throws -> (HandoffInputFilter?, Bool) {
    guard serverManagesConversation else { return (inputFilter, shouldNestHistory) }
    if inputFilter != nil {
        throw UserError(
            "Server-managed conversations do not support handoff input filters. Remove Handoff.input_filter or "
                + "RunConfig.handoff_input_filter, or disable conversation_id, previous_response_id, and auto_previous_response_id."
        )
    }
    if shouldNestHistory {
        logger.warning(
            "Server-managed conversations do not support nest_handoff_history for handoff \(fromAgent.name) -> \(toAgent.name). "
                + "Disabling nested handoff history and continuing with delta-only input."
        )
    }
    return (inputFilter, false)
}

/// Executes a handoff and prepares the next turn for the new agent.
func executeHandoffs(
    agent: Agent,
    originalInput: RunInput,
    preStepItems: [RunItem],
    newStepItems: [RunItem],
    newResponse: ModelResponse,
    runHandoffs: [ToolRunHandoff],
    hooks: RunHooks,
    contextWrapper: RunContextWrapper,
    runConfig: RunConfig,
    serverManagesConversation: Bool,
    toolInputGuardrailResults: [ToolInputGuardrailResult],
    toolOutputGuardrailResults: [ToolOutputGuardrailResult]
) async throws -> SingleStepResult {
    var newStepItems = newStepItems
    var preStepItems = preStepItems
    var originalInput = originalInput
    let multipleHandoffs = runHandoffs.count > 1
    if multipleHandoffs {
        let message = "Multiple handoffs detected, ignoring this one."
        for extra in runHandoffs.dropFirst() {
            newStepItems.append(ToolCallOutputItem(agent: agent, rawItem: try ItemHelpers.toolCallOutputItem(extra.toolCall, output: message), output: message))
        }
    }
    let actual = runHandoffs[0]
    let span = handoffSpan(fromAgent: agent.name)
    var sessionStepItems: [RunItem]?
    var nestedHistoryOwnedItems: [RunItem]?
    let newAgent: Agent = try await span.run {
        let handoff = actual.handoff
        let target = try await handoff.onInvokeHandoff(contextWrapper, actual.toolCall.arguments)
        guard let newAgent = target as? Agent else {
            throw UserError("Handoff target is not an Agent")
        }
        span.data(as: HandoffSpanData.self)?.toAgent = newAgent.name
        if multipleHandoffs {
            span.setError(SpanError(message: "Multiple handoffs requested", data: [
                "requested_agents": .array(runHandoffs.map { .string($0.handoff.agentName) }),
            ]))
        }
        let handoffOutput = HandoffOutputItem(
            agent: agent,
            rawItem: try ItemHelpers.toolCallOutputItem(actual.toolCall, output: handoff.getTransferMessage(newAgent)),
            sourceAgent: agent,
            targetAgent: newAgent
        )
        newStepItems.append(handoffOutput)

        async let runHook: Void = hooks.onHandoff(context: contextWrapper, fromAgent: agent, toAgent: newAgent)
        async let agentHook: Void = agent.hooks?.onHandoff(context: contextWrapper, agent: newAgent, source: agent) ?? ()
        _ = try await (runHook, agentHook)

        var inputFilter = handoff.inputFilter ?? runConfig.handoffInputFilter
        var shouldNest = handoff.nestHandoffHistory ?? runConfig.nestHandoffHistory
        (inputFilter, shouldNest) = try resolveServerManagedHandoffBehavior(
            fromAgent: agent, toAgent: newAgent, serverManagesConversation: serverManagesConversation,
            inputFilter: inputFilter, shouldNestHistory: shouldNest
        )
        guard inputFilter != nil || shouldNest else { return newAgent }
        let data = HandoffInputData(
            inputHistory: originalInput,
            preHandoffItems: preStepItems,
            newItems: newStepItems,
            runContext: contextWrapper
        )
        if let inputFilter {
            logger.debug("Filtering handoff inputs for \(agent.name) -> \(newAgent.name)")
            let filtered = try await inputFilter(data)
            nestedHistoryOwnedItems = []
            originalInput = filtered.inputHistory
            preStepItems = filtered.preHandoffItems
            newStepItems = filtered.newItems
            if let inputItems = filtered.inputItems {
                sessionStepItems = filtered.newItems
                newStepItems = inputItems
            }
        } else if shouldNest {
            let (nested, owned) = try nestHandoffHistoryWithProvenance(data, historyMapper: runConfig.handoffHistoryMapper)
            nestedHistoryOwnedItems = owned.map(\.runItem)
            originalInput = nested.inputHistory
            preStepItems = nested.preHandoffItems
            sessionStepItems = nested.newItems
            newStepItems = nested.inputItems ?? nested.newItems
        }
        return newAgent
    }
    return SingleStepResult(
        originalInput: originalInput,
        modelResponse: newResponse,
        preStepItems: preStepItems,
        newStepItems: newStepItems,
        nextStep: .handoff(newAgent),
        toolInputGuardrailResults: toolInputGuardrailResults,
        toolOutputGuardrailResults: toolOutputGuardrailResults,
        sessionStepItems: sessionStepItems,
        nestedHistoryOwnedItems: nestedHistoryOwnedItems
    )
}

private func resolveInvalidFinalOutput(
    errorHandlers: RunErrorHandlers?,
    error: ModelBehaviorError,
    agent: Agent,
    originalInput: RunInput,
    newResponse: ModelResponse,
    newItems: [RunItem],
    contextWrapper: RunContextWrapper
) async throws -> (any Sendable, MessageOutputItem?)? {
    let data = buildRunErrorData(input: originalInput, newItems: newItems, rawResponses: [newResponse], lastAgent: agent)
    guard let result = try await resolveRunErrorHandlerResult(
        errorHandlers: errorHandlers, errorKind: "invalid_final_output", error: error, contextWrapper: contextWrapper, runData: data
    ) else { return nil }
    let finalOutput = try validateHandlerFinalOutput(agent, result.finalOutput)
    let message = result.includeInHistory ? createMessageOutputItem(agent, formatFinalOutputText(agent, finalOutput)) : nil
    return (finalOutput, message)
}

/// Runs one turn of the loop, coordinating tools, approvals, guardrails, and handoffs.
func executeToolsAndSideEffects( // swiftlint:disable:this function_body_length
    agent: Agent,
    allTools: [any Tool],
    originalInput: RunInput,
    preStepItems: [RunItem],
    newResponse: ModelResponse,
    processedResponse: ProcessedResponse,
    outputSchema: (any AgentOutputSchemaBase)?,
    hooks: RunHooks,
    contextWrapper: RunContextWrapper,
    runConfig: RunConfig,
    errorHandlers: RunErrorHandlers?,
    serverManagesConversation: Bool
) async throws -> SingleStepResult {
    var processedResponse = processedResponse
    let approvalItemsByCallId = indexApprovalItemsByCallId(preStepItems)
    let plan = try buildPlanForFreshTurn(processed: processedResponse, agent: agent, contextWrapper: contextWrapper, approvalItemsByCallId: approvalItemsByCallId)
    var newStepItems = processedResponse.newItems

    let results = try await executeToolPlan(plan: plan, agent: agent, allTools: allTools, hooks: hooks, contextWrapper: contextWrapper, runConfig: runConfig)
    newStepItems.append(contentsOf: results.resultItems)
    newStepItems.append(contentsOf: try await buildToolNotFoundOutputItems(
        agent: agent, calls: processedResponse.functionToolsNotFound, contextWrapper: contextWrapper, runConfig: runConfig
    ))

    var interruptions = results.interruptions
    newStepItems.append(contentsOf: plan.approvedMCPResponses)
    if !plan.pendingInterruptions.isEmpty {
        interruptions.append(contentsOf: plan.pendingInterruptions)
        newStepItems.append(contentsOf: plan.pendingInterruptions)
    }
    processedResponse.interruptions = interruptions

    if !interruptions.isEmpty {
        return SingleStepResult(
            originalInput: originalInput,
            modelResponse: newResponse,
            preStepItems: preStepItems,
            newStepItems: newStepItems,
            nextStep: .interruption(interruptions),
            toolInputGuardrailResults: results.toolInputGuardrailResults,
            toolOutputGuardrailResults: results.toolOutputGuardrailResults,
            processedResponse: processedResponse
        )
    }

    newStepItems.append(contentsOf: try await executeMCPApprovalRequests(agent: agent, requests: plan.mcpRequestsWithCallback, contextWrapper: contextWrapper))

    if !processedResponse.handoffs.isEmpty {
        return try await executeHandoffs(
            agent: agent, originalInput: originalInput, preStepItems: preStepItems, newStepItems: newStepItems,
            newResponse: newResponse, runHandoffs: processedResponse.handoffs, hooks: hooks, contextWrapper: contextWrapper,
            runConfig: runConfig, serverManagesConversation: serverManagesConversation,
            toolInputGuardrailResults: results.toolInputGuardrailResults, toolOutputGuardrailResults: results.toolOutputGuardrailResults
        )
    }

    let toolUse = try await checkForFinalOutputFromTools(agent, results.functionResults, contextWrapper)
    if toolUse.isFinalOutput {
        var finalOutput = toolUse.finalOutput
        if getOutputSchema(agent) == nil {
            finalOutput = finalOutput.map { stringify($0) }
        }
        if finalOutput == nil {
            logger.error("Model returned a final output of None. Not raising an error because we assume you know what you're doing.")
        }
        return try await executeFinalOutput(
            agent: agent, originalInput: originalInput, newResponse: newResponse, preStepItems: preStepItems,
            newStepItems: newStepItems, finalOutput: finalOutput ?? Optional<String>.none as any Sendable, hooks: hooks,
            contextWrapper: contextWrapper, toolInputGuardrailResults: results.toolInputGuardrailResults,
            toolOutputGuardrailResults: results.toolOutputGuardrailResults
        )
    }

    let messageItems = newStepItems.compactMap { $0 as? MessageOutputItem }
    let refusal = messageItems.last.flatMap { ItemHelpers.extractRefusal($0.rawItem) }
    let potentialFinalOutputText = messageItems.last.flatMap { ItemHelpers.extractText($0.rawItem) }

    func finalize(_ output: any Sendable) async throws -> SingleStepResult {
        try await executeFinalOutput(
            agent: agent, originalInput: originalInput, newResponse: newResponse, preStepItems: preStepItems,
            newStepItems: newStepItems, finalOutput: output, hooks: hooks, contextWrapper: contextWrapper,
            toolInputGuardrailResults: results.toolInputGuardrailResults, toolOutputGuardrailResults: results.toolOutputGuardrailResults
        )
    }

    if !processedResponse.hasToolsOrApprovalsToRun() {
        let hasToolActivityWithoutMessage = messageItems.isEmpty && !processedResponse.toolsUsed.isEmpty
        if !hasToolActivityWithoutMessage {
            if let refusal {
                let refusalError = ModelRefusalError(refusal: refusal)
                let data = buildRunErrorData(input: originalInput, newItems: preStepItems + newStepItems, rawResponses: [newResponse], lastAgent: agent)
                guard let handlerResult = try await resolveRunErrorHandlerResult(
                    errorHandlers: errorHandlers, errorKind: "model_refusal", error: refusalError, contextWrapper: contextWrapper, runData: data
                ) else {
                    throw refusalError
                }
                let finalOutput = try validateHandlerFinalOutput(agent, handlerResult.finalOutput)
                if handlerResult.includeInHistory {
                    newStepItems.append(createMessageOutputItem(agent, formatFinalOutputText(agent, finalOutput)))
                }
                return try await finalize(finalOutput)
            }
            if let outputSchema, !outputSchema.isPlainText() {
                if let text = potentialFinalOutputText {
                    do {
                        return try await finalize(try outputSchema.validateJSON(text))
                    } catch let error as ModelBehaviorError {
                        guard let resolved = try await resolveInvalidFinalOutput(
                            errorHandlers: errorHandlers, error: error, agent: agent, originalInput: originalInput,
                            newResponse: newResponse, newItems: preStepItems + newStepItems, contextWrapper: contextWrapper
                        ) else {
                            throw error
                        }
                        if let message = resolved.1 { newStepItems.append(message) }
                        return try await finalize(resolved.0)
                    }
                } else {
                    guard let resolved = try await resolveInvalidFinalOutput(
                        errorHandlers: errorHandlers,
                        error: ModelBehaviorError("Model returned no final output for the structured output type."),
                        agent: agent, originalInput: originalInput, newResponse: newResponse,
                        newItems: preStepItems + newStepItems, contextWrapper: contextWrapper
                    ) else {
                        return SingleStepResult(
                            originalInput: originalInput, modelResponse: newResponse, preStepItems: preStepItems,
                            newStepItems: newStepItems, nextStep: .runAgain,
                            toolInputGuardrailResults: results.toolInputGuardrailResults,
                            toolOutputGuardrailResults: results.toolOutputGuardrailResults
                        )
                    }
                    if let message = resolved.1 { newStepItems.append(message) }
                    return try await finalize(resolved.0)
                }
            }
            return try await finalize(potentialFinalOutputText ?? "")
        }
    }

    return SingleStepResult(
        originalInput: originalInput,
        modelResponse: newResponse,
        preStepItems: preStepItems,
        newStepItems: newStepItems,
        nextStep: .runAgain,
        toolInputGuardrailResults: results.toolInputGuardrailResults,
        toolOutputGuardrailResults: results.toolOutputGuardrailResults
    )
}

/// Processes a model response and executes its side effects.
func getSingleStepResultFromResponse(
    agent: Agent,
    allTools: [any Tool],
    originalInput: RunInput,
    preStepItems: [RunItem],
    newResponse: ModelResponse,
    outputSchema: (any AgentOutputSchemaBase)?,
    handoffs: [Handoff],
    hooks: RunHooks,
    contextWrapper: RunContextWrapper,
    runConfig: RunConfig,
    toolUseTracker: AgentToolUseTracker,
    errorHandlers: RunErrorHandlers?,
    serverManagesConversation: Bool,
    afterInvocationValidation: ((ProcessedResponse?) async throws -> Void)? = nil,
    beforeSideEffects: (() async throws -> Void)? = nil
) async throws -> SingleStepResult {
    let processed: ProcessedResponse
    do {
        processed = try processModelResponse(
            agent: agent, allTools: allTools, response: newResponse, outputSchema: outputSchema,
            handoffs: handoffs, runConfig: runConfig
        )
    } catch let error as ModelBehaviorError {
        try await afterInvocationValidation?(nil)
        throw error
    }
    try await afterInvocationValidation?(processed)
    try await beforeSideEffects?()
    toolUseTracker.recordProcessedResponse(agent, processed)
    return try await executeToolsAndSideEffects(
        agent: agent, allTools: allTools, originalInput: originalInput, preStepItems: preStepItems,
        newResponse: newResponse, processedResponse: processed, outputSchema: outputSchema, hooks: hooks,
        contextWrapper: contextWrapper, runConfig: runConfig, errorHandlers: errorHandlers,
        serverManagesConversation: serverManagesConversation
    )
}

/// Resumes a turn that was interrupted for tool approvals.
func resolveInterruptedTurn(
    agent: Agent,
    allTools: [any Tool],
    originalInput: RunInput,
    originalPreStepItems: [RunItem],
    newResponse: ModelResponse,
    processedResponse: ProcessedResponse,
    hooks: RunHooks,
    contextWrapper: RunContextWrapper,
    runConfig: RunConfig,
    serverManagesConversation: Bool,
    errorHandlers: RunErrorHandlers?
) async throws -> SingleStepResult {
    // Items from the interrupted turn that already have outputs are not re-executed.
    var outputCallIds: Set<String> = []
    for case let item as ToolCallOutputItem in originalPreStepItems {
        if let callId = item.callId { outputCallIds.insert(callId) }
    }
    let approvalItemsByCallId = indexApprovalItemsByCallId(originalPreStepItems)
    // Approval placeholders are replaced by tool outputs (or kept when still pending).
    let preStepItems = originalPreStepItems.filter { !($0 is ToolApprovalItem) }

    var pendingInterruptions: [ToolApprovalItem] = []
    var rejectionItems: [RunItem] = []
    var approvedFunctions: [ToolRunFunction] = []
    for run in processedResponse.functions where !outputCallIds.contains(run.toolCall.callId) {
        let existing = approvalItemsByCallId[run.toolCall.callId]
        let current = ToolApprovalItem(
            agent: agent, rawItem: run.toolCall.json, toolName: run.functionTool.name,
            toolNamespace: getToolCallNamespace(run.toolCall.json), toolOrigin: getFunctionToolOrigin(run.functionTool),
            toolLookupKey: getFunctionToolLookupKeyForTool(run.functionTool)
        )
        var status = try contextWrapper.getApprovalStatus(
            toolName: run.functionTool.name, callId: run.toolCall.callId, toolNamespace: getToolCallNamespace(run.toolCall.json),
            existingPending: existing ?? current, toolLookupKey: current.toolLookupKey
        )
        var requiresApproval = true
        if status == nil {
            requiresApproval = try await functionNeedsApproval(run.functionTool, contextWrapper, run.toolCall)
            status = try contextWrapper.getApprovalStatus(
                toolName: run.functionTool.name, callId: run.toolCall.callId, toolNamespace: getToolCallNamespace(run.toolCall.json),
                existingPending: existing ?? current, toolLookupKey: current.toolLookupKey
            )
        }
        switch status {
        case false?:
            let message = await resolveApprovalRejectionMessage(
                contextWrapper: contextWrapper, runConfig: runConfig, toolType: "function",
                toolName: run.functionTool.name, callId: run.toolCall.callId,
                toolNamespace: getToolCallNamespace(run.toolCall.json), toolLookupKey: current.toolLookupKey,
                existingPending: existing ?? current
            )
            rejectionItems.append(try functionRejectionItem(
                agent: agent, toolCall: run.toolCall, rejectionMessage: message,
                outputJSONSchema: run.functionTool.outputJSONSchema, toolOrigin: getFunctionToolOrigin(run.functionTool)
            ))
        case true?:
            approvedFunctions.append(run)
        case nil:
            if requiresApproval {
                pendingInterruptions.append(existing ?? current)
            } else {
                approvedFunctions.append(run)
            }
        }
    }

    var plan = ToolExecutionPlan(
        functionRuns: approvedFunctions,
        computerActions: processedResponse.computerActions.filter { !outputCallIds.contains($0.toolCall.callId) },
        customToolCalls: processedResponse.customToolCalls.filter { !outputCallIds.contains($0.toolCall.callId) },
        shellCalls: processedResponse.shellCalls.filter { !outputCallIds.contains($0.toolCall.string("call_id") ?? "") },
        applyPatchCalls: processedResponse.applyPatchCalls.filter { !outputCallIds.contains($0.toolCall.string("call_id") ?? "") },
        localShellCalls: [],
        pendingInterruptions: pendingInterruptions
    )
    let existingMCPResponses = Set(originalPreStepItems.compactMap { ($0 as? MCPApprovalResponseItem)?.rawItem.string("approval_request_id") })
    let mcpRequests = processedResponse.mcpApprovalRequests.filter { !existingMCPResponses.contains($0.requestItem.id) }
    let manual = mcpRequests.filter { $0.mcpTool.onApprovalRequest == nil }
    let (approvedMCP, pendingMCP) = try collectManualMCPApprovals(agent: agent, requests: manual, contextWrapper: contextWrapper, existingPendingByCallId: approvalItemsByCallId)
    plan.approvedMCPResponses = approvedMCP
    plan.pendingInterruptions.append(contentsOf: pendingMCP)
    plan.mcpRequestsWithCallback = mcpRequests.filter { $0.mcpTool.onApprovalRequest != nil }

    let results = try await executeToolPlan(plan: plan, agent: agent, allTools: allTools, hooks: hooks, contextWrapper: contextWrapper, runConfig: runConfig)
    var newStepItems: [RunItem] = rejectionItems + results.resultItems + plan.approvedMCPResponses
    var interruptions = results.interruptions + plan.pendingInterruptions
    newStepItems.append(contentsOf: plan.pendingInterruptions)
    newStepItems.append(contentsOf: try await executeMCPApprovalRequests(agent: agent, requests: plan.mcpRequestsWithCallback, contextWrapper: contextWrapper))

    var resumedProcessed = processedResponse
    resumedProcessed.interruptions = interruptions
    if !interruptions.isEmpty {
        interruptions = Array(interruptions)
        return SingleStepResult(
            originalInput: originalInput, modelResponse: newResponse, preStepItems: preStepItems,
            newStepItems: newStepItems, nextStep: .interruption(interruptions),
            toolInputGuardrailResults: results.toolInputGuardrailResults,
            toolOutputGuardrailResults: results.toolOutputGuardrailResults,
            processedResponse: resumedProcessed
        )
    }

    if !processedResponse.handoffs.isEmpty {
        return try await executeHandoffs(
            agent: agent, originalInput: originalInput, preStepItems: preStepItems, newStepItems: newStepItems,
            newResponse: newResponse, runHandoffs: processedResponse.handoffs, hooks: hooks, contextWrapper: contextWrapper,
            runConfig: runConfig, serverManagesConversation: serverManagesConversation,
            toolInputGuardrailResults: results.toolInputGuardrailResults, toolOutputGuardrailResults: results.toolOutputGuardrailResults
        )
    }

    let toolUse = try await checkForFinalOutputFromTools(agent, results.functionResults, contextWrapper)
    if toolUse.isFinalOutput {
        var finalOutput = toolUse.finalOutput
        if getOutputSchema(agent) == nil { finalOutput = finalOutput.map { stringify($0) } }
        return try await executeFinalOutput(
            agent: agent, originalInput: originalInput, newResponse: newResponse, preStepItems: preStepItems,
            newStepItems: newStepItems, finalOutput: finalOutput ?? Optional<String>.none as any Sendable, hooks: hooks,
            contextWrapper: contextWrapper, toolInputGuardrailResults: results.toolInputGuardrailResults,
            toolOutputGuardrailResults: results.toolOutputGuardrailResults
        )
    }

    return SingleStepResult(
        originalInput: originalInput, modelResponse: newResponse, preStepItems: preStepItems,
        newStepItems: newStepItems, nextStep: .runAgain,
        toolInputGuardrailResults: results.toolInputGuardrailResults,
        toolOutputGuardrailResults: results.toolOutputGuardrailResults
    )
}
