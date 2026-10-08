import Foundation

// Port of src/agents/run_internal/turn_preparation.py, guardrails.py, error_handlers.py, and
// the usage helpers in agent_runner_helpers.py.

/// Applies the optional `callModelInputFilter` to modify model input.
func maybeFilterModelInput(
    agent: Agent,
    runConfig: RunConfig,
    contextWrapper: RunContextWrapper,
    inputItems: [TResponseInputItem],
    systemInstructions: String?
) async throws -> ModelInputData {
    let base = ModelInputData(input: inputItems, instructions: systemInstructions)
    guard let filter = runConfig.callModelInputFilter else { return base }
    do {
        return try await filter(CallModelData(modelData: base, agent: agent, context: contextWrapper.context))
    } catch {
        attachErrorToCurrentSpan(SpanError(
            message: "Error in call_model_input_filter",
            data: ["error": .string(getTraceError(traceIncludeSensitiveData: runConfig.traceIncludeSensitiveData, errorMessage: describeError(error)))]
        ))
        throw error
    }
}

/// Returns the enabled handoffs for the agent.
func getHandoffs(_ agent: Agent, _ contextWrapper: RunContextWrapper) async throws -> [Handoff] {
    var enabled: [Handoff] = []
    for target in agent.handoffs {
        let resolved = target.resolved
        if try await resolved.isEnabled.resolve(contextWrapper, agent) {
            enabled.append(resolved)
        }
    }
    return enabled
}

/// Returns the resolved output schema for the agent, if any.
func getOutputSchema(_ agent: Agent) -> (any AgentOutputSchemaBase)? {
    guard let outputType = agent.outputType, !outputType.isPlainText() else { return nil }
    return outputType
}

/// Resolves the model instance for this run.
func getModel(_ agent: Agent, _ runConfig: RunConfig) throws -> any Model {
    switch runConfig.model {
    case .instance(let model)?: return model
    case .name(let name)?: return try runConfig.modelProvider.getModel(name)
    case nil: break
    }
    switch agent.model {
    case .instance(let model)?: return model
    case .name(let name)?: return try runConfig.modelProvider.getModel(name)
    case nil: return try runConfig.modelProvider.getModel(nil)
    }
}

private func modelSettingsForResolvedName(_ agent: Agent, _ runConfig: RunConfig) -> ModelSettings {
    switch runConfig.model {
    case .name(let name)?: return getDefaultModelSettings(name)
    case .instance?: return ModelSettings()
    case nil: return Agent.initialModelSettings(for: agent.model)
    }
}

/// Resolves model settings, keeping implicit defaults aligned with the resolved model name.
func getModelSettings(_ agent: Agent, _ runConfig: RunConfig) -> ModelSettings {
    var settings = agent.modelSettings
    if settings == Agent.initialModelSettings(for: agent.model) {
        settings = modelSettingsForResolvedName(agent, runConfig)
    }
    return settings.resolve(runConfig.modelSettings)
}

/// Resets `toolChoice` when the agent already used tools and should not be forced again.
func maybeResetToolChoice(_ agent: Agent, _ tracker: AgentToolUseTracker, _ settings: ModelSettings) -> ModelSettings {
    guard agent.resetToolChoice, tracker.hasUsedTools(agent) else { return settings }
    var reset = settings
    reset.toolChoice = nil
    return reset
}

// MARK: - Guardrails

func runSingleInputGuardrail(_ agent: Agent, _ guardrail: InputGuardrail, _ input: RunInput, _ context: RunContextWrapper) async throws -> InputGuardrailResult {
    let span = guardrailSpan(name: guardrail.getName())
    return try await span.run {
        let result = try await guardrail.run(agent: agent, input: input, context: context)
        span.data(as: GuardrailSpanData.self)?.triggered = result.output.tripwireTriggered
        return result
    }
}

func runSingleOutputGuardrail(_ guardrail: OutputGuardrail, _ agent: Agent, _ output: any Sendable, _ context: RunContextWrapper) async throws -> OutputGuardrailResult {
    let span = guardrailSpan(name: guardrail.getName())
    return try await span.run {
        let result = try await guardrail.run(context: context, agent: agent, agentOutput: output)
        span.data(as: GuardrailSpanData.self)?.triggered = result.output.tripwireTriggered
        return result
    }
}

/// Runs input guardrails concurrently and throws on tripwires. Results are recorded into `sink`
/// as each guardrail completes, including the tripping result.
@discardableResult
func runInputGuardrails(
    _ agent: Agent,
    _ guardrails: [InputGuardrail],
    _ input: RunInput,
    _ context: RunContextWrapper,
    sink: SharedBox<[InputGuardrailResult]>? = nil
) async throws -> [InputGuardrailResult] {
    guard !guardrails.isEmpty else { return [] }
    let currentSpan = Scope.currentSpan
    let currentTrace = Scope.currentTrace
    return try await withThrowingTaskGroup(of: InputGuardrailResult.self) { group in
        for guardrail in guardrails {
            group.addTask {
                try await Scope.$currentTrace.withValue(currentTrace) {
                    try await Scope.$currentSpan.withValue(currentSpan) {
                        try await runSingleInputGuardrail(agent, guardrail, input, context)
                    }
                }
            }
        }
        var results: [InputGuardrailResult] = []
        for try await result in group {
            results.append(result)
            sink?.mutate { $0.append(result) }
            if result.output.tripwireTriggered {
                group.cancelAll()
                attachErrorToCurrentSpan(SpanError(message: "Guardrail tripwire triggered", data: ["guardrail": .string(result.guardrail.getName())]))
                throw InputGuardrailTripwireTriggered(guardrailResult: result)
            }
        }
        return results
    }
}

/// Runs output guardrails in parallel and throws on tripwires.
@discardableResult
func runOutputGuardrails(
    _ guardrails: [OutputGuardrail],
    _ agent: Agent,
    _ output: any Sendable,
    _ context: RunContextWrapper,
    sink: SharedBox<[OutputGuardrailResult]>? = nil
) async throws -> [OutputGuardrailResult] {
    guard !guardrails.isEmpty else { return [] }
    let currentSpan = Scope.currentSpan
    let currentTrace = Scope.currentTrace
    return try await withThrowingTaskGroup(of: OutputGuardrailResult.self) { group in
        for guardrail in guardrails {
            group.addTask {
                try await Scope.$currentTrace.withValue(currentTrace) {
                    try await Scope.$currentSpan.withValue(currentSpan) {
                        try await runSingleOutputGuardrail(guardrail, agent, output, context)
                    }
                }
            }
        }
        var results: [OutputGuardrailResult] = []
        for try await result in group {
            results.append(result)
            sink?.mutate { $0.append(result) }
            if result.output.tripwireTriggered {
                group.cancelAll()
                attachErrorToCurrentSpan(SpanError(message: "Guardrail tripwire triggered", data: ["guardrail": .string(result.guardrail.getName())]))
                throw OutputGuardrailTripwireTriggered(guardrailResult: result)
            }
        }
        return results
    }
}

// MARK: - Usage helpers (agent_runner_helpers.py)

/// Creates a usage snapshot for computing invocation-local deltas.
func snapshotUsage(_ usage: Usage) -> Usage {
    Usage(
        requests: usage.requests,
        inputTokens: usage.inputTokens,
        inputTokensDetails: usage.inputTokensDetails,
        outputTokens: usage.outputTokens,
        outputTokensDetails: usage.outputTokensDetails,
        totalTokens: usage.totalTokens
    )
}

/// Returns the aggregate usage added between two snapshots.
func usageDelta(_ start: Usage, _ end: Usage) -> Usage {
    Usage(
        requests: end.requests - start.requests,
        inputTokens: end.inputTokens - start.inputTokens,
        inputTokensDetails: InputTokensDetails(
            cachedTokens: end.inputTokensDetails.cachedTokens - start.inputTokensDetails.cachedTokens,
            cacheWriteTokens: end.inputTokensDetails.cacheWriteTokens - start.inputTokensDetails.cacheWriteTokens
        ),
        outputTokens: end.outputTokens - start.outputTokens,
        outputTokensDetails: OutputTokensDetails(
            reasoningTokens: end.outputTokensDetails.reasoningTokens - start.outputTokensDetails.reasoningTokens
        ),
        totalTokens: end.totalTokens - start.totalTokens
    )
}

/// Attaches aggregate token usage to a task or turn span.
func attachUsageToSpan(_ span: (any Span)?, _ usage: Usage) {
    guard let span else { return }
    if let data = span.data(as: TaskSpanData.self) {
        data.usage = taskUsageToSpanData(usage)
    } else if let data = span.data(as: TurnSpanData.self) {
        data.usage = turnUsageToSpanData(usage)
    }
}

// MARK: - Error handlers (error_handlers.py)

let genericAgentErrorMessage = "Error in agent run"

/// Marks the agent span of a failed run with the generic error, unless a more specific error
/// already exists.
func attachGenericAgentError(_ span: (any Span)?, _ error: any Error, traceIncludeSensitiveData: Bool) {
    guard let span, !(error is CancellationError) else { return }
    if error is ModelBehaviorError || error is InputGuardrailTripwireTriggered || error is OutputGuardrailTripwireTriggered {
        return
    }
    if span.error != nil { return }
    let detail = traceIncludeSensitiveData ? describeError(error) : redactedTraceErrorMessage
    attachErrorToSpan(span, SpanError(message: genericAgentErrorMessage, data: ["error": .string(detail)]))
}

func buildRunErrorData(
    input: RunInput,
    newItems: [RunItem],
    rawResponses: [ModelResponse],
    lastAgent: Agent,
    reasoningItemIdPolicy: ReasoningItemIdPolicy? = nil
) -> RunErrorData {
    let output = (try? runItemsToInputItems(newItems, reasoningItemIdPolicy: reasoningItemIdPolicy)) ?? []
    return RunErrorData(
        input: input,
        newItems: newItems,
        history: input.inputItems + output,
        output: output,
        rawResponses: rawResponses,
        lastAgent: lastAgent
    )
}

/// Wraps a handler final output in `{"response": ...}` when the output schema is wrapped.
private func handlerPayloadValue(_ schema: any AgentOutputSchemaBase, _ finalOutput: any Sendable) throws -> JSONValue {
    let value = try jsonDump(finalOutput)
    guard (schema as? any WrappedOutputSchema)?.isWrapped == true else { return value }
    if case .object(let object) = value, object[wrapperDictKey] != nil { return value }
    return .object([wrapperDictKey: value])
}

/// Formats a final output as message text for history.
func formatFinalOutputText(_ agent: Agent, _ finalOutput: any Sendable) -> String {
    guard let schema = getOutputSchema(agent) else { return stringify(finalOutput) }
    guard let payload = try? handlerPayloadValue(schema, finalOutput) else { return stringify(finalOutput) }
    return payload.jsonString()
}

/// Validates an error handler's final output against the agent's output schema.
func validateHandlerFinalOutput(_ agent: Agent, _ finalOutput: any Sendable) throws -> any Sendable {
    guard let schema = getOutputSchema(agent) else { return finalOutput }
    let payload: String
    do {
        payload = try handlerPayloadValue(schema, finalOutput).jsonString()
    } catch {
        throw UserError("Invalid run error handler final_output for structured output.")
    }
    do {
        return try schema.validateJSON(payload)
    } catch is ModelBehaviorError {
        throw UserError("Invalid run error handler final_output for structured output.")
    }
}

func createMessageOutputItem(_ agent: Agent, _ outputText: String) -> MessageOutputItem {
    MessageOutputItem(
        agent: agent,
        rawItem: ResponseOutputMessage(id: fakeResponsesId, content: [ResponseOutputText(text: outputText).json]).json
    )
}

func resolveRunErrorHandlerResult(
    errorHandlers: RunErrorHandlers?,
    errorKind: String,
    error: AgentsException,
    contextWrapper: RunContextWrapper,
    runData: RunErrorData
) async throws -> RunErrorHandlerResult? {
    guard let handler = errorHandlers?.handler(for: errorKind) else { return nil }
    return try await handler(RunErrorHandlerInput(error: error, context: contextWrapper, runData: runData))
}
