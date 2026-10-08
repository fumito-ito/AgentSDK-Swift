import Foundation

// Port of src/agents/tracing/create.py and the public helpers in tracing/__init__.py.

/// Creates a new trace. The trace is not started; use `withTrace` or call `start()`/`finish()`.
public func trace(
    _ workflowName: String,
    traceId: String? = nil,
    groupId: String? = nil,
    metadata: JSONObject? = nil,
    tracing: TracingConfig? = nil,
    disabled: Bool = false
) -> any Trace {
    if getTraceProvider().getCurrentTrace() != nil {
        logger.warning("Trace already exists. Creating a new trace, but this is probably a mistake.")
    }
    return getTraceProvider().createTrace(
        name: workflowName, traceId: traceId, groupId: groupId, metadata: metadata, disabled: disabled, tracing: tracing
    )
}

/// Runs `body` inside a new trace (the `with trace(...):` equivalent).
public func withTrace<T>(
    _ workflowName: String,
    traceId: String? = nil,
    groupId: String? = nil,
    metadata: JSONObject? = nil,
    tracing: TracingConfig? = nil,
    disabled: Bool = false,
    _ body: () async throws -> T
) async rethrows -> T {
    let created = trace(workflowName, traceId: traceId, groupId: groupId, metadata: metadata, tracing: tracing, disabled: disabled)
    return try await created.run(body)
}

/// Returns the currently active trace, if present.
public func getCurrentTrace() -> (any Trace)? { getTraceProvider().getCurrentTrace() }

/// Returns the currently active span, if present.
public func getCurrentSpan() -> (any Span)? { getTraceProvider().getCurrentSpan() }

private func makeSpan(_ data: any SpanData, spanId: String?, parent: SpanParent?, disabled: Bool) -> any Span {
    getTraceProvider().createSpan(spanData: data, spanId: spanId, parent: parent, disabled: disabled)
}

/// Creates a new agent span. The span is not started automatically.
public func agentSpan(
    name: String,
    handoffs: [String]? = nil,
    tools: [String]? = nil,
    outputType: String? = nil,
    spanId: String? = nil,
    parent: SpanParent? = nil,
    disabled: Bool = false
) -> any Span {
    makeSpan(AgentSpanData(name: name, handoffs: handoffs, tools: tools, outputType: outputType), spanId: spanId, parent: parent, disabled: disabled)
}

/// Creates a new task span, representing one top-level Runner invocation.
public func taskSpan(name: String, spanId: String? = nil, parent: SpanParent? = nil, disabled: Bool = false) -> any Span {
    makeSpan(TaskSpanData(name: name), spanId: spanId, parent: parent, disabled: disabled)
}

/// Creates a new turn span.
public func turnSpan(turn: Int, agentName: String, spanId: String? = nil, parent: SpanParent? = nil, disabled: Bool = false) -> any Span {
    makeSpan(TurnSpanData(turn: turn, agentName: agentName), spanId: spanId, parent: parent, disabled: disabled)
}

/// Creates a new function span.
public func functionSpan(name: String, input: String? = nil, output: String? = nil, spanId: String? = nil, parent: SpanParent? = nil, disabled: Bool = false) -> any Span {
    makeSpan(FunctionSpanData(name: name, input: input, output: output), spanId: spanId, parent: parent, disabled: disabled)
}

/// Creates a new generation span, capturing one model generation.
public func generationSpan(
    input: [JSONObject]? = nil,
    output: [JSONObject]? = nil,
    model: String? = nil,
    modelConfig: JSONObject? = nil,
    usage: JSONObject? = nil,
    spanId: String? = nil,
    parent: SpanParent? = nil,
    disabled: Bool = false
) -> any Span {
    makeSpan(GenerationSpanData(input: input, output: output, model: model, modelConfig: modelConfig, usage: usage), spanId: spanId, parent: parent, disabled: disabled)
}

/// Creates a new response span.
public func responseSpan(response: ResponseObject? = nil, spanId: String? = nil, parent: SpanParent? = nil, disabled: Bool = false) -> any Span {
    makeSpan(ResponseSpanData(response: response), spanId: spanId, parent: parent, disabled: disabled)
}

/// Creates a new handoff span.
public func handoffSpan(fromAgent: String? = nil, toAgent: String? = nil, spanId: String? = nil, parent: SpanParent? = nil, disabled: Bool = false) -> any Span {
    makeSpan(HandoffSpanData(fromAgent: fromAgent, toAgent: toAgent), spanId: spanId, parent: parent, disabled: disabled)
}

/// Creates a new custom span.
public func customSpan(name: String, data: JSONObject? = nil, spanId: String? = nil, parent: SpanParent? = nil, disabled: Bool = false) -> any Span {
    makeSpan(CustomSpanData(name: name, data: data ?? [:]), spanId: spanId, parent: parent, disabled: disabled)
}

/// Creates a new guardrail span.
public func guardrailSpan(name: String, triggered: Bool = false, spanId: String? = nil, parent: SpanParent? = nil, disabled: Bool = false) -> any Span {
    makeSpan(GuardrailSpanData(name: name, triggered: triggered), spanId: spanId, parent: parent, disabled: disabled)
}

/// Creates a new transcription span.
public func transcriptionSpan(
    model: String? = nil,
    input: String? = nil,
    inputFormat: String? = "pcm",
    output: String? = nil,
    modelConfig: JSONObject? = nil,
    spanId: String? = nil,
    parent: SpanParent? = nil,
    disabled: Bool = false
) -> any Span {
    makeSpan(
        TranscriptionSpanData(input: input, inputFormat: inputFormat, output: output, model: model, modelConfig: modelConfig),
        spanId: spanId, parent: parent, disabled: disabled
    )
}

/// Creates a new speech span.
public func speechSpan(
    model: String? = nil,
    input: String? = nil,
    output: String? = nil,
    outputFormat: String? = "pcm",
    modelConfig: JSONObject? = nil,
    firstContentAt: String? = nil,
    spanId: String? = nil,
    parent: SpanParent? = nil,
    disabled: Bool = false
) -> any Span {
    makeSpan(
        SpeechSpanData(
            input: input, output: output, outputFormat: outputFormat, model: model,
            modelConfig: modelConfig, firstContentAt: firstContentAt
        ),
        spanId: spanId, parent: parent, disabled: disabled
    )
}

/// Creates a new speech group span.
public func speechGroupSpan(input: String? = nil, spanId: String? = nil, parent: SpanParent? = nil, disabled: Bool = false) -> any Span {
    makeSpan(SpeechGroupSpanData(input: input), spanId: spanId, parent: parent, disabled: disabled)
}

/// Creates a new MCP list-tools span.
public func mcpToolsSpan(server: String? = nil, result: [String]? = nil, spanId: String? = nil, parent: SpanParent? = nil, disabled: Bool = false) -> any Span {
    makeSpan(MCPListToolsSpanData(server: server, result: result), spanId: spanId, parent: parent, disabled: disabled)
}

/// Adds a new trace processor. This processor receives all traces and spans.
public func addTraceProcessor(_ processor: any TracingProcessor) {
    getTraceProvider().registerProcessor(processor)
}

/// Replaces the list of trace processors.
public func setTraceProcessors(_ processors: [any TracingProcessor]) {
    getTraceProvider().setProcessors(processors)
}

/// Sets whether tracing is globally disabled.
public func setTracingDisabled(_ disabled: Bool) {
    getTraceProvider().setDisabled(disabled)
}

/// Sets the OpenAI API key for the backend exporter.
public func setTracingExportAPIKey(_ apiKey: String) {
    defaultExporter().setAPIKey(apiKey)
}

/// Forces immediate export of buffered traces and spans.
public func flushTraces() {
    getTraceProvider().forceFlush()
}

// MARK: - Run traces (tracing/context.py)

func traceStateMatchesEffectiveSettings(
    _ state: TraceState,
    workflowName: String,
    traceId: String?,
    groupId: String?,
    metadata: JSONObject?,
    tracing: TracingConfig?
) -> Bool {
    guard let stateTraceId = state.traceId, stateTraceId == traceId else { return false }
    guard state.workflowName == workflowName, state.groupId == groupId, state.metadata == metadata else { return false }
    let key = tracing?.apiKey
    if let stateKey = state.tracingAPIKey { return stateKey == key }
    if let stateHash = state.tracingAPIKeyHash { return stateHash == hashTracingAPIKey(key) }
    return key == nil
}

/// Returns a trace for a run when one is not already active.
func createTraceForRun(
    workflowName: String,
    traceId: String?,
    groupId: String?,
    metadata: JSONObject?,
    tracing: TracingConfig?,
    disabled: Bool,
    traceState: TraceState? = nil,
    reattachResumedTrace: Bool = false
) -> (any Trace)? {
    if getCurrentTrace() != nil { return nil }
    if reattachResumedTrace, !disabled, let traceState, StartedTraceIds.wasStarted(traceState.traceId),
       traceStateMatchesEffectiveSettings(traceState, workflowName: workflowName, traceId: traceId, groupId: groupId, metadata: metadata, tracing: tracing) {
        return reattachTrace(traceState, tracingAPIKey: tracing?.apiKey)
    }
    return trace(workflowName, traceId: traceId, groupId: groupId, metadata: metadata, tracing: tracing, disabled: disabled)
}

/// Creates a trace when none exists and runs `body` within it (upstream `TraceCtxManager`).
func withTraceForRun<T>(
    workflowName: String,
    traceId: String?,
    groupId: String?,
    metadata: JSONObject?,
    tracing: TracingConfig?,
    disabled: Bool,
    traceState: TraceState? = nil,
    reattachResumedTrace: Bool = false,
    _ body: () async throws -> T
) async rethrows -> T {
    guard let created = createTraceForRun(
        workflowName: workflowName, traceId: traceId, groupId: groupId, metadata: metadata,
        tracing: tracing, disabled: disabled, traceState: traceState, reattachResumedTrace: reattachResumedTrace
    ) else {
        return try await body()
    }
    return try await created.run(body)
}

// MARK: - Error tracing (util/_error_tracing.py, util/_tool_errors.py)

public let redactedTraceErrorMessage = "Error details are redacted."
public let redactedToolErrorMessage = "Tool execution failed. Error details are redacted."

/// Returns a trace-safe error string based on the sensitive-data setting.
func getTraceError(traceIncludeSensitiveData: Bool, errorMessage: String, redactedMessage: String = redactedTraceErrorMessage) -> String {
    traceIncludeSensitiveData ? errorMessage : redactedMessage
}

/// Returns a trace-safe tool error string based on the sensitive-data setting.
func getTraceToolError(traceIncludeSensitiveData: Bool, errorMessage: String) -> String {
    getTraceError(traceIncludeSensitiveData: traceIncludeSensitiveData, errorMessage: errorMessage, redactedMessage: redactedToolErrorMessage)
}

public func attachErrorToSpan(_ span: any Span, _ error: SpanError) {
    span.setError(error)
}

public func attachErrorToCurrentSpan(_ error: SpanError) {
    if let span = getCurrentSpan() {
        attachErrorToSpan(span, error)
    } else if AgentsDebug.dontLogModelData || AgentsDebug.dontLogToolData {
        logger.warning("No active span; trace error was not attached")
    } else {
        logger.warning("No span to add error \(error.message) to")
    }
}

/// Records an already-known model failure on its span.
func recordModelErrorOnSpan(_ span: any Span, message: String, error: any Error, traceIncludeSensitiveData: Bool) {
    let text = traceIncludeSensitiveData ? describeError(error) : redactedTraceErrorMessage
    attachErrorToSpan(span, SpanError(message: message, data: ["error": .string(text)]))
}
