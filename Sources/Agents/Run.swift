import Foundation

// Port of src/agents/run.py (`Runner`, `AgentRunner._run_impl`, `AgentRunner.run_streamed`) and
// run_internal/run_loop.py (`start_streaming`).
//
// Swift: the streaming and non-streaming loops share one driver (`RunDriver`); the turn itself
// differs (`runSingleTurn` vs `runSingleTurnStreamed`). `run_sync` is not ported (Swift callers
// use `await`). Sandbox runtimes, blocked-output replacement, pending session-write
// checkpoints, and prompt cache keys are not ported; see porting/manifest.json.

/// Runs agents. `Runner` delegates to the default `AgentRunner`.
public enum Runner {
    /// Runs a workflow starting at the given agent.
    ///
    /// The agent runs in a loop until a final output is generated:
    /// 1. The agent is invoked with the given input.
    /// 2. If there is a final output (the agent produces something of type `agent.outputType`),
    ///    the loop terminates.
    /// 3. If there's a handoff, the loop runs again with the new agent.
    /// 4. Otherwise, tool calls (if any) run, and the loop runs again.
    ///
    /// Throws `MaxTurnsExceeded` if `maxTurns` is exceeded (unless handled), and a tripwire error
    /// such as `InputGuardrailTripwireTriggered` if a guardrail is tripped. Only the first agent's
    /// input guardrails run.
    ///
    /// - Parameters:
    ///   - startingAgent: The starting agent to run.
    ///   - input: The initial input: a user message string, or a list of input items.
    ///   - context: The context to run the agent with.
    ///   - maxTurns: The maximum number of turns (one AI invocation including its tool calls).
    ///     Pass `nil` to disable the limit.
    ///   - hooks: Receives callbacks on lifecycle events.
    ///   - runConfig: Global settings for the entire run.
    ///   - errorHandlers: Error handlers keyed by error kind.
    ///   - previousResponseId: The ID of the previous Responses API response.
    ///   - autoPreviousResponseId: Enables response chaining from the first turn.
    ///   - conversationId: A Responses API conversation ID used to read and write items.
    ///   - session: A session for automatic conversation history management.
    public static func run(
        _ startingAgent: Agent,
        input: RunInput,
        context: any Sendable = (),
        maxTurns: Int? = defaultMaxTurns,
        hooks: RunHooks? = nil,
        runConfig: RunConfig? = nil,
        errorHandlers: RunErrorHandlers? = nil,
        previousResponseId: String? = nil,
        autoPreviousResponseId: Bool = false,
        conversationId: String? = nil,
        session: (any Session)? = nil
    ) async throws -> RunResult {
        try await getDefaultAgentRunner().run(
            startingAgent, input: .input(input), contextWrapper: RunContextWrapper(context: context), maxTurns: maxTurns,
            hooks: hooks, runConfig: runConfig, errorHandlers: errorHandlers, previousResponseId: previousResponseId,
            autoPreviousResponseId: autoPreviousResponseId, conversationId: conversationId, session: session
        )
    }

    /// Resumes a run from a `RunState`, e.g. after approving or rejecting interruptions.
    public static func run(
        _ startingAgent: Agent,
        state: RunState,
        hooks: RunHooks? = nil,
        runConfig: RunConfig? = nil,
        errorHandlers: RunErrorHandlers? = nil,
        previousResponseId: String? = nil,
        autoPreviousResponseId: Bool = false,
        conversationId: String? = nil,
        session: (any Session)? = nil
    ) async throws -> RunResult {
        try await getDefaultAgentRunner().run(
            startingAgent, input: .state(state), contextWrapper: nil, maxTurns: state.maxTurns,
            hooks: hooks, runConfig: runConfig, errorHandlers: errorHandlers, previousResponseId: previousResponseId,
            autoPreviousResponseId: autoPreviousResponseId, conversationId: conversationId, session: session
        )
    }

    /// Runs with an existing context wrapper (used by `Agent.asTool`).
    static func run(
        _ startingAgent: Agent,
        input: RunInput,
        contextWrapper: RunContextWrapper,
        maxTurns: Int?,
        hooks: RunHooks?,
        runConfig: RunConfig,
        previousResponseId: String?,
        conversationId: String?,
        session: (any Session)?
    ) async throws -> RunResult {
        try await getDefaultAgentRunner().run(
            startingAgent, input: .input(input), contextWrapper: contextWrapper, maxTurns: maxTurns,
            hooks: hooks, runConfig: runConfig, errorHandlers: nil, previousResponseId: previousResponseId,
            autoPreviousResponseId: false, conversationId: conversationId, session: session
        )
    }

    /// Runs a workflow in streaming mode. Consume `streamEvents()` on the returned result to
    /// receive events as they are generated; the stream throws the same errors as `run`.
    ///
    /// Throws immediately for invalid configurations, such as a session combined with a
    /// server-managed conversation.
    public static func runStreamed(
        _ startingAgent: Agent,
        input: RunInput,
        context: any Sendable = (),
        maxTurns: Int? = defaultMaxTurns,
        hooks: RunHooks? = nil,
        runConfig: RunConfig? = nil,
        errorHandlers: RunErrorHandlers? = nil,
        previousResponseId: String? = nil,
        autoPreviousResponseId: Bool = false,
        conversationId: String? = nil,
        session: (any Session)? = nil
    ) throws -> RunResultStreaming {
        try getDefaultAgentRunner().runStreamed(
            startingAgent, input: .input(input), contextWrapper: RunContextWrapper(context: context), maxTurns: maxTurns,
            hooks: hooks, runConfig: runConfig, errorHandlers: errorHandlers, previousResponseId: previousResponseId,
            autoPreviousResponseId: autoPreviousResponseId, conversationId: conversationId, session: session
        )
    }

    /// Resumes a run from a `RunState` in streaming mode.
    public static func runStreamed(
        _ startingAgent: Agent,
        state: RunState,
        hooks: RunHooks? = nil,
        runConfig: RunConfig? = nil,
        errorHandlers: RunErrorHandlers? = nil,
        previousResponseId: String? = nil,
        autoPreviousResponseId: Bool = false,
        conversationId: String? = nil,
        session: (any Session)? = nil
    ) throws -> RunResultStreaming {
        try getDefaultAgentRunner().runStreamed(
            startingAgent, input: .state(state), contextWrapper: nil, maxTurns: state.maxTurns,
            hooks: hooks, runConfig: runConfig, errorHandlers: errorHandlers, previousResponseId: previousResponseId,
            autoPreviousResponseId: autoPreviousResponseId, conversationId: conversationId, session: session
        )
    }

    /// Runs with an existing context wrapper in streaming mode (used by `Agent.asTool`).
    static func runStreamed(
        _ startingAgent: Agent,
        input: RunInput,
        contextWrapper: RunContextWrapper,
        maxTurns: Int?,
        hooks: RunHooks?,
        runConfig: RunConfig,
        previousResponseId: String?,
        conversationId: String?,
        session: (any Session)?
    ) throws -> RunResultStreaming {
        try getDefaultAgentRunner().runStreamed(
            startingAgent, input: .input(input), contextWrapper: contextWrapper, maxTurns: maxTurns,
            hooks: hooks, runConfig: runConfig, errorHandlers: nil, previousResponseId: previousResponseId,
            autoPreviousResponseId: false, conversationId: conversationId, session: session
        )
    }
}

/// The input to a run: fresh input or a state to resume.
public enum RunnerInput: Sendable {
    case input(RunInput)
    case state(RunState)
}

/// The runner implementation used by `Runner`. Subclass to customize, then install it with
/// `setDefaultAgentRunner`.
open class AgentRunner: @unchecked Sendable {
    public init() {}

    open func run(
        _ startingAgent: Agent,
        input: RunnerInput,
        contextWrapper: RunContextWrapper?,
        maxTurns: Int?,
        hooks: RunHooks?,
        runConfig: RunConfig?,
        errorHandlers: RunErrorHandlers?,
        previousResponseId: String?,
        autoPreviousResponseId: Bool,
        conversationId: String?,
        session: (any Session)?
    ) async throws -> RunResult {
        let driver = try RunDriver(
            startingAgent: startingAgent, input: input, contextWrapper: contextWrapper, maxTurns: maxTurns,
            hooks: hooks ?? RunHooks(), runConfig: runConfig ?? RunConfig(), errorHandlers: errorHandlers,
            previousResponseId: previousResponseId, autoPreviousResponseId: autoPreviousResponseId,
            conversationId: conversationId, session: session, streamed: nil
        )
        return try await driver.execute()
    }

    open func runStreamed(
        _ startingAgent: Agent,
        input: RunnerInput,
        contextWrapper: RunContextWrapper?,
        maxTurns: Int?,
        hooks: RunHooks?,
        runConfig: RunConfig?,
        errorHandlers: RunErrorHandlers?,
        previousResponseId: String?,
        autoPreviousResponseId: Bool,
        conversationId: String?,
        session: (any Session)?
    ) throws -> RunResultStreaming {
        let wrapper: RunContextWrapper
        let resultInput: RunInput
        let resolvedMaxTurns: Int?
        switch input {
        case .input(let runInput):
            wrapper = contextWrapper ?? RunContextWrapper()
            resultInput = runInput
            resolvedMaxTurns = maxTurns
        case .state(let state):
            wrapper = state.context
            resultInput = state.originalInput
            resolvedMaxTurns = state.maxTurns
        }
        let resumed: RunState?
        if case .state(let state) = input { resumed = state } else { resumed = nil }
        try validateSessionConversationSettings(
            session,
            conversationId: conversationId ?? resumed?.conversationId,
            previousResponseId: previousResponseId ?? resumed?.previousResponseId,
            autoPreviousResponseId: autoPreviousResponseId || (resumed?.autoPreviousResponseId ?? false)
        )
        let currentAgent: Agent
        if case .state(let state) = input { currentAgent = state.currentAgent ?? startingAgent } else { currentAgent = startingAgent }
        let streamed = RunResultStreaming(input: resultInput, currentAgent: currentAgent, contextWrapper: wrapper, maxTurns: resolvedMaxTurns)
        let task = Task<Void, any Error> {
            do {
                let driver = try RunDriver(
                    startingAgent: startingAgent, input: input, contextWrapper: wrapper, maxTurns: resolvedMaxTurns,
                    hooks: hooks ?? RunHooks(), runConfig: runConfig ?? RunConfig(), errorHandlers: errorHandlers,
                    previousResponseId: previousResponseId, autoPreviousResponseId: autoPreviousResponseId,
                    conversationId: conversationId, session: session, streamed: streamed
                )
                _ = try await driver.execute()
                streamed.finish()
            } catch {
                streamed.finish(error: error)
                throw error
            }
        }
        streamed.attach(task: task)
        return streamed
    }
}

private let defaultRunnerBox = SharedBox<AgentRunner>(AgentRunner())

/// Sets the default runner used by `Runner`. Pass `nil` to restore the built-in runner.
public func setDefaultAgentRunner(_ runner: AgentRunner?) {
    defaultRunnerBox.mutate { $0 = runner ?? AgentRunner() }
}

/// Returns the default runner used by `Runner`.
public func getDefaultAgentRunner() -> AgentRunner {
    defaultRunnerBox.value
}

// MARK: - Driver

@discardableResult
private func withOptionalCurrentSpan<T>(_ span: (any Span)?, _ body: () async throws -> T) async rethrows -> T {
    guard let span else { return try await body() }
    return try await span.withCurrent(body)
}

private func hasOutputGuardrails(_ agent: Agent, _ runConfig: RunConfig) -> Bool {
    !agent.outputGuardrails.isEmpty || !(runConfig.outputGuardrails ?? []).isEmpty
}

private func shouldDeferInterruptedSessionItems(_ agent: Agent, _ runConfig: RunConfig) -> Bool {
    guard hasOutputGuardrails(agent, runConfig) else { return false }
    if case .runLLMAgain = agent.toolUseBehavior { return false }
    return true
}

/// The outcome of handling one turn result.
private enum TurnOutcome {
    case done(RunResult)
    case continueLoop
}

/// Drives one agent run (upstream `AgentRunner._run_impl` and `start_streaming`).
private final class RunDriver: @unchecked Sendable {
    let startingAgent: Agent
    let hooks: RunHooks
    let runConfig: RunConfig
    let errorHandlers: RunErrorHandlers?
    let session: (any Session)?
    let streamed: RunResultStreaming?
    let contextWrapper: RunContextWrapper
    let reasoningItemIdPolicy: ReasoningItemIdPolicy?
    let serverConversationTracker: OpenAIServerConversationTracker?
    let sessionPersistenceEnabled: Bool
    let toolUseTracker = AgentToolUseTracker()
    let runState: RunState
    let conversationId: String?
    let previousResponseId: String?
    let autoPreviousResponseId: Bool

    var isResumedState: Bool
    var maxTurns: Int?
    var rawInput: RunInput?
    var originalUserInput: RunInput?
    var originalInput: RunInput
    var startingInput: RunInput?
    var sessionInputItemsForPersistence: [TResponseInputItem]?
    var lastSavedInputSnapshotForRewind: [TResponseInputItem]?
    var currentTurn = 0
    var generatedItems: [RunItem] = []
    var sessionItems: [RunItem] = []
    var modelResponses: [ModelResponse] = []
    var pendingServerItems: [RunItem]?
    var inputGuardrailResults: [InputGuardrailResult] = []
    var inputGuardrailAttemptStart = 0
    var outputGuardrailResults: [OutputGuardrailResult] = []
    var toolInputGuardrailResults: [ToolInputGuardrailResult] = []
    var toolOutputGuardrailResults: [ToolOutputGuardrailResult] = []
    var currentAgent: Agent
    var nestedHistoryOwnedItems: [RunItem] = []
    var currentSpan: (any Span)?
    var shouldRunAgentStartHooks = true

    init(
        startingAgent: Agent,
        input: RunnerInput,
        contextWrapper: RunContextWrapper?,
        maxTurns: Int?,
        hooks: RunHooks,
        runConfig: RunConfig,
        errorHandlers: RunErrorHandlers?,
        previousResponseId: String?,
        autoPreviousResponseId: Bool,
        conversationId: String?,
        session: (any Session)?,
        streamed: RunResultStreaming?
    ) throws {
        self.startingAgent = startingAgent
        self.hooks = hooks
        self.runConfig = runConfig
        self.errorHandlers = errorHandlers
        self.session = session
        self.streamed = streamed
        self.maxTurns = maxTurns

        var conversationId = conversationId
        var previousResponseId = previousResponseId
        var autoPreviousResponseId = autoPreviousResponseId
        let state: RunState?
        if case .state(let resumed) = input { state = resumed } else { state = nil }
        self.reasoningItemIdPolicy = runConfig.reasoningItemIdPolicy ?? state?.reasoningItemIdPolicy
        if let state {
            state.reasoningItemIdPolicy = reasoningItemIdPolicy
            conversationId = conversationId ?? state.conversationId
            previousResponseId = previousResponseId ?? state.previousResponseId
            autoPreviousResponseId = autoPreviousResponseId || state.autoPreviousResponseId
        }
        try validateSessionConversationSettings(
            session, conversationId: conversationId, previousResponseId: previousResponseId,
            autoPreviousResponseId: autoPreviousResponseId
        )
        self.conversationId = conversationId
        self.previousResponseId = previousResponseId
        self.autoPreviousResponseId = autoPreviousResponseId

        let serverManages = conversationId != nil || previousResponseId != nil || autoPreviousResponseId
        serverConversationTracker = serverManages
            ? OpenAIServerConversationTracker(
                conversationId: conversationId, previousResponseId: previousResponseId,
                autoPreviousResponseId: autoPreviousResponseId, reasoningItemIdPolicy: reasoningItemIdPolicy
            )
            : nil
        sessionPersistenceEnabled = session != nil && !serverManages

        if let state {
            isResumedState = true
            self.contextWrapper = state.context
            self.runState = state
            self.maxTurns = state.maxTurns
            self.originalUserInput = state.originalInput
            self.originalInput = normalizeResumedInput(state.originalInput)
            self.startingInput = nil
            self.currentAgent = state.currentAgent ?? startingAgent
            sessionInputItemsForPersistence = session != nil ? [] : nil
            if let serverConversationTracker {
                serverConversationTracker.hydrateFromState(
                    originalInput: state.originalInput, generatedItems: state.generatedItems, modelResponses: state.modelResponses
                )
            }
            toolUseTracker.hydrate(from: state.toolUseTrackerSnapshot, startingAgent: state.startingAgent)
        } else {
            guard case .input(let raw) = input else { fatalError("unreachable") }
            isResumedState = false
            let wrapper = contextWrapper ?? RunContextWrapper()
            self.contextWrapper = wrapper
            self.rawInput = raw
            self.originalUserInput = raw
            self.originalInput = raw
            self.startingInput = raw
            self.currentAgent = startingAgent
            self.runState = RunState(
                context: wrapper, originalInput: raw, startingAgent: startingAgent, maxTurns: maxTurns,
                conversationId: conversationId, previousResponseId: previousResponseId,
                autoPreviousResponseId: autoPreviousResponseId
            )
            runState.reasoningItemIdPolicy = reasoningItemIdPolicy
        }
    }

    // MARK: Entry

    func execute() async throws -> RunResult {
        if !isResumedState, let rawInput {
            if serverConversationTracker != nil {
                let (prepared, _) = try await prepareInputWithSession(
                    rawInput, session: session, sessionInputCallback: runConfig.sessionInputCallback,
                    sessionSettings: runConfig.sessionSettings, includeHistoryInPreparedInput: false,
                    preserveDroppedNewItems: true, reasoningItemIdPolicy: reasoningItemIdPolicy
                )
                originalInput = prepared
                sessionInputItemsForPersistence = []
            } else {
                let (prepared, toPersist) = try await prepareInputWithSession(
                    rawInput, session: session, sessionInputCallback: runConfig.sessionInputCallback,
                    sessionSettings: runConfig.sessionSettings, reasoningItemIdPolicy: reasoningItemIdPolicy
                )
                originalInput = prepared
                sessionInputItemsForPersistence = session != nil ? toPersist : nil
            }
            startingInput = originalInput
            runState.originalInput = originalInput
        }

        let traceState = isResumedState ? runState.traceState : nil
        return try await withTraceForRun(
            workflowName: traceState?.workflowName ?? runConfig.workflowName,
            traceId: traceState?.traceId ?? runConfig.traceId,
            groupId: traceState?.groupId ?? runConfig.groupId,
            metadata: traceState?.metadata ?? runConfig.traceMetadata,
            tracing: runConfig.tracing,
            disabled: runConfig.tracingDisabled,
            traceState: traceState,
            reattachResumedTrace: isResumedState
        ) {
            runState.setTrace(getCurrentTrace())
            streamed?.update { $0.trace = getCurrentTrace() }
            return try await executeInTrace()
        }
    }

    private func executeInTrace() async throws -> RunResult {
        if isResumedState {
            currentTurn = runState.currentTurn
            generatedItems = runState.generatedItems
            sessionItems = runState.sessionItems
            modelResponses = runState.modelResponses
            inputGuardrailResults = runState.inputGuardrailResults
            outputGuardrailResults = runState.outputGuardrailResults
            toolInputGuardrailResults = runState.toolInputGuardrailResults
            toolOutputGuardrailResults = runState.toolOutputGuardrailResults
        }
        inputGuardrailAttemptStart = inputGuardrailResults.count
        syncStreamed()

        let currentTaskSpan = includeTaskAndTurnSpans(runConfig.tracing) ? taskSpan(name: runConfig.workflowName) : nil
        currentTaskSpan?.start()
        let taskUsageStart = snapshotUsage(contextWrapper.usage)
        defer {
            if let currentTaskSpan {
                attachUsageToSpan(currentTaskSpan, usageDelta(taskUsageStart, contextWrapper.usage))
                currentTaskSpan.finish()
            }
        }
        return try await withOptionalCurrentSpan(currentTaskSpan) {
            if !isResumedState, sessionPersistenceEnabled, sessionInputItemsForPersistence == nil, let originalUserInput {
                sessionInputItemsForPersistence = ItemHelpers.inputToNewInputList(originalUserInput)
            }
            if sessionPersistenceEnabled, let items = sessionInputItemsForPersistence, !items.isEmpty, streamed == nil {
                lastSavedInputSnapshotForRewind = items
                try await saveResultToSession(session, originalInput: .items(items), newItems: [], runState: runState, store: storeSetting)
                sessionInputItemsForPersistence = []
            }
            do {
                let result = try await runLoop()
                await cleanup()
                return result
            } catch {
                attachGenericAgentError(currentSpan, error, traceIncludeSensitiveData: runConfig.traceIncludeSensitiveData)
                if let exception = error as? AgentsException {
                    exception.runData = RunErrorDetails(
                        input: originalInput, newItems: sessionItems, rawResponses: modelResponses, lastAgent: currentAgent,
                        contextWrapper: contextWrapper, inputGuardrailResults: inputGuardrailResults,
                        outputGuardrailResults: outputGuardrailResults, toolInputGuardrailResults: toolInputGuardrailResults,
                        toolOutputGuardrailResults: toolOutputGuardrailResults
                    )
                }
                await cleanup()
                throw error
            }
        }
    }

    private func cleanup() async {
        await cleanupModelsAfterRun(toolUseTracker)
        await disposeResolvedComputers(runContext: contextWrapper)
        currentSpan?.finish()
        currentSpan = nil
    }

    private var storeSetting: Bool? {
        currentAgent.modelSettings.resolve(runConfig.modelSettings).store
    }

    private var attemptInputGuardrailResults: [InputGuardrailResult] {
        Array(inputGuardrailResults.dropFirst(inputGuardrailAttemptStart))
    }

    /// Mirrors loop state onto the streaming result.
    private func syncStreamed() {
        guard let streamed else { return }
        let input = originalInput
        let items = sessionItems
        let modelInput = generatedItems
        let responses = modelResponses
        let agent = currentAgent
        let turn = currentTurn
        let inputResults = inputGuardrailResults
        let outputResults = outputGuardrailResults
        let toolInputResults = toolInputGuardrailResults
        let toolOutputResults = toolOutputGuardrailResults
        let policy = reasoningItemIdPolicy
        let owned = nestedHistoryOwnedItems
        streamed.update {
            $0.nestedHistoryOwnedItems = owned
            $0.input = input
            $0.newItems = items
            $0.modelInputItems = modelInput
            $0.replayFromModelInputItems = !Self.sameItems(modelInput, items)
            $0.rawResponses = responses
            $0.currentAgent = agent
            $0.currentTurn = turn
            $0.inputGuardrailResults = inputResults
            $0.outputGuardrailResults = outputResults
            $0.toolInputGuardrailResults = toolInputResults
            $0.toolOutputGuardrailResults = toolOutputResults
            $0.reasoningItemIdPolicy = policy
        }
    }

    static func sameItems(_ lhs: [RunItem], _ rhs: [RunItem]) -> Bool {
        lhs.count == rhs.count && zip(lhs, rhs).allSatisfy { $0 === $1 }
    }

    // MARK: Loop

    private func runLoop() async throws -> RunResult {
        while true {
            if let streamed, currentTurn > 0 {
                // Let consumers process the completed turn before starting another one.
                await streamed.waitForTurnEventConsumption()
                if streamed.cancelMode == .afterTurn { return buildCancelledResult() }
            }
            try validateOutputGuardrailsWithServerManagedConversation()
            let resumingTurn = isResumedState
            let allInputGuardrails = currentTurn == 0 && !resumingTurn ? startingAgent.inputGuardrails + (runConfig.inputGuardrails ?? []) : []
            let sequentialGuardrails = allInputGuardrails.filter { !$0.runInParallel }
            let parallelGuardrails = allInputGuardrails.filter(\.runInParallel)

            if sessionPersistenceEnabled, let items = sessionInputItemsForPersistence, !items.isEmpty, streamed == nil {
                lastSavedInputSnapshotForRewind = items
                try await saveResultToSession(session, originalInput: .items(items), newItems: [], runState: runState, store: storeSetting)
                sessionInputItemsForPersistence = []
            }

            if isResumedState, case .interruption? = runState.currentStep {
                try await resumeInterruptedTurn()
                if let result = pendingResult {
                    pendingResult = nil
                    return result
                }
                continue
            }

            if runState.currentStep == nil { runState.currentStep = .runAgain }
            try await admitPendingInput()

            if currentSpan == nil {
                let outputTypeName = getOutputSchema(currentAgent)?.name() ?? "str"
                let span = agentSpan(name: currentAgent.name, handoffs: [], tools: [], outputType: outputTypeName)
                span.start()
                currentSpan = span
                streamed?.emit(.agentUpdated(AgentUpdatedStreamEvent(newAgent: currentAgent)))
            }

            currentTurn += 1
            syncStreamed()
            if let maxTurns, currentTurn > maxTurns {
                return try await handleMaxTurnsExceeded(maxTurns)
            }
            if !resumingTurn || { if case .runAgain? = runState.currentStep { return true } else { return false } }() {
                runState.currentTurnPersistedItemCount = 0
            }
            logger.debug("Running agent \(currentAgent.name) (turn \(currentTurn))")
            if sessionPersistenceEnabled {
                lastSavedInputSnapshotForRewind = ItemHelpers.inputToNewInputList(originalInput)
            }

            let itemsForModel = serverConversationTracker != nil && !(pendingServerItems ?? []).isEmpty ? pendingServerItems! : generatedItems
            let turnResult = try await withOptionalCurrentSpan(currentSpan) {
                try await runTurn(itemsForModel: itemsForModel, sequentialGuardrails: sequentialGuardrails, parallelGuardrails: parallelGuardrails)
            }
            shouldRunAgentStartHooks = false
            lastSavedInputSnapshotForRewind = nil

            if case .done(let result) = try await handleTurnResult(turnResult) {
                return result
            }
        }
    }

    private var pendingResult: RunResult?

    private func validateOutputGuardrailsWithServerManagedConversation() throws {
        guard conversationId != nil || previousResponseId != nil || autoPreviousResponseId else { return }
        guard hasOutputGuardrails(currentAgent, runConfig) else { return }
        if let model = try? getModel(currentAgent, runConfig), model is OpenAIChatCompletionsModel {
            throw UserError(
                "Output guardrails cannot be combined with server-managed conversations when using the Chat Completions model."
            )
        }
    }

    private func runTurn(
        itemsForModel: [RunItem],
        sequentialGuardrails: [InputGuardrail],
        parallelGuardrails: [InputGuardrail]
    ) async throws -> SingleStepResult {
        let turnUsageStart = snapshotUsage(contextWrapper.usage)
        let currentTurnSpan = includeTaskAndTurnSpans(runConfig.tracing) ? turnSpan(turn: currentTurn, agentName: currentAgent.name) : nil
        currentTurnSpan?.start()
        defer {
            if let currentTurnSpan {
                attachUsageToSpan(currentTurnSpan, usageDelta(turnUsageStart, contextWrapper.usage))
                currentTurnSpan.finish()
            }
        }
        return try await withOptionalCurrentSpan(currentTurnSpan) {
            let guardrailSink = SharedBox<[InputGuardrailResult]>([])
            defer {
                inputGuardrailResults.append(contentsOf: guardrailSink.value)
                runState.inputGuardrailResults = inputGuardrailResults
                syncStreamed()
            }
            let guardrailInput = originalInput
            if currentTurn <= 1, !sequentialGuardrails.isEmpty {
                do {
                    try await runInputGuardrails(startingAgent, sequentialGuardrails, guardrailInput, contextWrapper, sink: guardrailSink)
                } catch let error as InputGuardrailTripwireTriggered {
                    try await persistForGuardrailTrip()
                    throw error
                }
            }
            if currentTurn <= 1, !parallelGuardrails.isEmpty {
                do {
                    return try await withThrowingTaskGroup(of: SingleStepResult?.self) { group in
                        let agent = startingAgent
                        let context = contextWrapper
                        group.addTask {
                            try await runInputGuardrails(agent, parallelGuardrails, guardrailInput, context, sink: guardrailSink)
                            return nil
                        }
                        group.addTask {
                            try await self.invokeTurn(itemsForModel: itemsForModel, guardrailSink: guardrailSink)
                        }
                        var turnResult: SingleStepResult?
                        for try await value in group {
                            if let value { turnResult = value }
                        }
                        guard let turnResult else { throw AgentsException("Model turn did not complete") }
                        return turnResult
                    }
                } catch let error as InputGuardrailTripwireTriggered {
                    try await persistForGuardrailTrip()
                    throw error
                }
            }
            return try await invokeTurn(itemsForModel: itemsForModel, guardrailSink: guardrailSink)
        }
    }

    private func invokeTurn(itemsForModel: [RunItem], guardrailSink: SharedBox<[InputGuardrailResult]>) async throws -> SingleStepResult {
        if let streamed {
            streamed.update {
                $0.modelInputItems = itemsForModel
            }
            let persistInput: () async throws -> Void = { [self] in
                guard !inputGuardrailsTriggered(guardrailSink.value) else { return }
                try await persistStreamInputIfNeeded()
            }
            return try await runSingleTurnStreamed(
                streamedResult: streamed, agent: currentAgent, hooks: hooks, contextWrapper: contextWrapper,
                runConfig: runConfig, shouldRunAgentStartHooks: shouldRunAgentStartHooks, toolUseTracker: toolUseTracker,
                serverConversationTracker: serverConversationTracker, session: session, pendingServerItems: pendingServerItems,
                reasoningItemIdPolicy: reasoningItemIdPolicy, errorHandlers: errorHandlers, agentSpan: currentSpan,
                beforeSideEffects: {
                    if let tripped = guardrailSink.value.first(where: { $0.output.tripwireTriggered }) {
                        throw InputGuardrailTripwireTriggered(guardrailResult: tripped)
                    }
                },
                persistInputIfNeeded: persistInput
            )
        }
        return try await runSingleTurn(
            agent: currentAgent, originalInput: originalInput, generatedItems: itemsForModel, hooks: hooks,
            contextWrapper: contextWrapper, runConfig: runConfig, shouldRunAgentStartHooks: shouldRunAgentStartHooks,
            toolUseTracker: toolUseTracker, serverConversationTracker: serverConversationTracker, session: session,
            sessionItemsToRewind: !isResumedState && sessionPersistenceEnabled ? lastSavedInputSnapshotForRewind : nil,
            reasoningItemIdPolicy: reasoningItemIdPolicy, errorHandlers: errorHandlers, agentSpan: currentSpan
        )
    }

    /// Saves the run input to the session once, for streaming runs (upstream
    /// `_persist_stream_input_if_needed`).
    private func persistStreamInputIfNeeded() async throws {
        guard sessionPersistenceEnabled, let items = sessionInputItemsForPersistence, !items.isEmpty else { return }
        lastSavedInputSnapshotForRewind = items
        try await saveResultToSession(session, originalInput: .items(items), newItems: [], runState: runState, store: storeSetting)
        sessionInputItemsForPersistence = []
    }

    private func persistForGuardrailTrip() async throws {
        if streamed != nil, !sessionPersistenceEnabled { return }
        sessionInputItemsForPersistence = try await persistSessionItemsForGuardrailTrip(
            session: session, serverManagesConversation: serverConversationTracker != nil,
            sessionInputItemsForPersistence: sessionInputItemsForPersistence, originalUserInput: originalUserInput,
            runState: runState, store: storeSetting
        )
    }

    // MARK: Turn results

    private func handleTurnResult(_ turnResult: SingleStepResult) async throws -> TurnOutcome {
        modelResponses.append(turnResult.modelResponse)
        originalInput = turnResult.originalInput
        if let owned = turnResult.nestedHistoryOwnedItems { nestedHistoryOwnedItems = owned }
        generatedItems = turnResult.preStepItems + turnResult.newStepItems
        let turnSessionItems = sessionItemsForTurn(turnResult)
        sessionItems.append(contentsOf: turnSessionItems)
        if serverConversationTracker != nil {
            pendingServerItems = turnResult.newStepItems
            serverConversationTracker?.trackServerItems(turnResult.modelResponse)
        }
        toolInputGuardrailResults.append(contentsOf: turnResult.toolInputGuardrailResults)
        toolOutputGuardrailResults.append(contentsOf: turnResult.toolOutputGuardrailResults)
        syncStreamed()

        var itemsToSaveTurn = turnSessionItems
        if case .interruption = turnResult.nextStep {} else if sessionPersistenceEnabled {
            let outputCallIds = Set(turnResult.newStepItems.compactMap { item -> String? in
                guard item is ToolCallOutputItem, item.rawItem.type != "program_output" else { return nil }
                return item.rawItem.string("call_id")
            })
            for item in generatedItems where item is ToolCallItem {
                guard let callId = item.rawItem.string("call_id"), outputCallIds.contains(callId) else { continue }
                if !itemsToSaveTurn.contains(where: { $0 === item }) && runState.currentTurnPersistedItemCount == 0 {
                    itemsToSaveTurn.append(item)
                }
            }
            if case .finalOutput = turnResult.nextStep {} else if !itemsToSaveTurn.isEmpty {
                logger.debug("Persisting turn items (types=\(itemsToSaveTurn.map(\.type)))")
                if isResumedState {
                    let saved = try await saveResultToSession(
                        session, originalInput: .items([]), newItems: itemsToSaveTurn, runState: nil,
                        responseId: turnResult.modelResponse.responseId, reasoningItemIdPolicy: runState.reasoningItemIdPolicy,
                        store: storeSetting
                    )
                    runState.currentTurnPersistedItemCount += saved
                } else {
                    try await saveResultToSession(
                        session, originalInput: .items([]), newItems: itemsToSaveTurn, runState: runState,
                        responseId: turnResult.modelResponse.responseId, store: storeSetting
                    )
                }
            }
        }
        isResumedState = false

        switch turnResult.nextStep {
        case .finalOutput(let output):
            return .done(try await finishWithFinalOutput(output, turnResult: turnResult, turnSessionItems: turnSessionItems, resultInput: startingInput ?? .text("")))
        case .interruption(let interruptions):
            if sessionPersistenceEnabled, !shouldDeferInterruptedSessionItems(currentAgent, runConfig),
               !inputGuardrailsTriggered(attemptInputGuardrailResults) {
                try await saveResultToSession(
                    session, originalInput: .items(sessionInputItemsForPersistence ?? []), newItems: sessionItemsForTurn(turnResult),
                    runState: runState, responseId: turnResult.modelResponse.responseId, store: storeSetting
                )
                sessionInputItemsForPersistence = []
            }
            return .done(buildInterruptionResult(interruptions, turnResult: turnResult, resultInput: startingInput ?? .text("")))
        case .handoff(let newAgent):
            switchAgent(to: newAgent, turnResult: turnResult)
            return .continueLoop
        case .runAgain:
            if sessionPersistenceEnabled, !inputGuardrailsTriggered(inputGuardrailResults), runState.currentTurnPersistedItemCount == 0 {
                try await saveResultToSession(
                    session, originalInput: .items([]), newItems: sessionItemsForTurn(turnResult), runState: runState,
                    responseId: turnResult.modelResponse.responseId, store: storeSetting
                )
            }
            return .continueLoop
        }
    }

    private func switchAgent(to newAgent: Agent, turnResult: SingleStepResult) {
        currentAgent = newAgent
        runState.currentAgent = newAgent
        startingInput = turnResult.originalInput
        originalInput = turnResult.originalInput
        currentSpan?.finish()
        currentSpan = nil
        shouldRunAgentStartHooks = true
        syncStreamed()
    }

    private func finishWithFinalOutput(
        _ output: any Sendable,
        turnResult: SingleStepResult,
        turnSessionItems: [RunItem],
        resultInput: RunInput
    ) async throws -> RunResult {
        if hasOutputGuardrails(currentAgent, runConfig) {
            runState.toolOutputGuardrailResults = toolOutputGuardrailResults
        }
        let sink = SharedBox<[OutputGuardrailResult]>([])
        do {
            try await withOptionalCurrentSpan(currentSpan) {
                try await runOutputGuardrails(
                    currentAgent.outputGuardrails + (runConfig.outputGuardrails ?? []),
                    currentAgent, output, contextWrapper, sink: sink
                )
            }
            outputGuardrailResults.append(contentsOf: sink.value)
        } catch {
            outputGuardrailResults.append(contentsOf: sink.value)
            syncStreamed()
            if !(error is OutputGuardrailTripwireTriggered) {
                try await saveFinalTurnItemsAfterGuardrails(turnSessionItems, responseId: turnResult.modelResponse.responseId)
            }
            throw error
        }
        try await saveFinalTurnItemsAfterGuardrails(turnSessionItems, responseId: turnResult.modelResponse.responseId)
        streamed?.update { $0.finalOutput = output }
        let result = makeResult(input: resultInput, finalOutput: output, interruptions: [])
        runState.currentStep = nil
        return result
    }

    private func saveFinalTurnItemsAfterGuardrails(_ items: [RunItem], responseId: String?) async throws {
        guard sessionPersistenceEnabled, !items.isEmpty, !inputGuardrailsTriggered(attemptInputGuardrailResults) else { return }
        if runState.currentTurnPersistedItemCount > 0 {
            runState.currentTurnPersistedItemCount = try await saveResumedTurnItems(
                runState: runState, session: session, items: items, persistedCount: runState.currentTurnPersistedItemCount,
                responseId: responseId, reasoningItemIdPolicy: runState.reasoningItemIdPolicy, store: storeSetting
            )
            return
        }
        try await saveResultToSession(session, originalInput: .items([]), newItems: items, runState: runState, responseId: responseId, store: storeSetting)
    }

    private func makeResult(input: RunInput, finalOutput: (any Sendable)?, interruptions: [ToolApprovalItem], processedResponse: ProcessedResponse? = nil) -> RunResult {
        let result = RunResult(
            input: input,
            newItems: sessionItems,
            rawResponses: modelResponses,
            finalOutput: finalOutput,
            lastAgent: currentAgent,
            inputGuardrailResults: inputGuardrailResults,
            outputGuardrailResults: outputGuardrailResults,
            toolInputGuardrailResults: toolInputGuardrailResults,
            toolOutputGuardrailResults: toolOutputGuardrailResults,
            contextWrapper: contextWrapper,
            interruptions: interruptions,
            maxTurns: maxTurns
        )
        result.modelInputItems = generatedItems
        result.replayFromModelInputItems = !Self.sameItems(generatedItems, sessionItems)
        result.reasoningItemIdPolicy = reasoningItemIdPolicy
        result.nestedHistoryOwnedItems = nestedHistoryOwnedItems
        result.stateFields = ResultStateFields(
            currentTurn: currentTurn,
            lastProcessedResponse: processedResponse,
            currentTurnPersistedItemCount: runState.currentTurnPersistedItemCount,
            toolUseTrackerSnapshot: toolUseTracker.asSerializable(),
            originalInput: originalInput,
            conversationId: conversationId,
            previousResponseId: serverConversationTracker?.previousResponseId ?? previousResponseId,
            autoPreviousResponseId: autoPreviousResponseId,
            traceState: runState.traceState,
            startingAgent: runState.startingAgent,
            pendingInput: runState.pendingInputItems,
            currentStep: runState.currentStep
        )
        if let streamed {
            let fields = result.stateFields
            streamed.update {
                $0.interruptions = interruptions
                $0.stateFields = fields
            }
            syncStreamed()
        }
        return result
    }

    private func buildInterruptionResult(_ interruptions: [ToolApprovalItem], turnResult: SingleStepResult, resultInput: RunInput) -> RunResult {
        let processed = turnResult.processedResponse ?? runState.lastProcessedResponse
        runState.modelResponses = modelResponses
        runState.lastProcessedResponse = processed
        runState.generatedItems = generatedItems
        runState.sessionItems = sessionItems
        runState.currentStep = .interruption(interruptions: interruptions, responseAccepted: false)
        runState.currentTurn = currentTurn
        runState.toolInputGuardrailResults = toolInputGuardrailResults
        runState.toolOutputGuardrailResults = toolOutputGuardrailResults
        runState.inputGuardrailResults = inputGuardrailResults
        return makeResult(input: resultInput, finalOutput: nil, interruptions: interruptions, processedResponse: processed)
    }

    private func buildCancelledResult() -> RunResult {
        makeResult(input: startingInput ?? originalInput, finalOutput: nil, interruptions: [])
    }

    // MARK: Resume

    private func resumeInterruptedTurn() async throws {
        logger.debug("Continuing from interruption")
        guard let lastResponse = runState.modelResponses.last else {
            throw UserError("No model response found in previous state")
        }
        guard let processed = runState.lastProcessedResponse else {
            throw UserError("No processed response found in previous state")
        }
        if currentSpan == nil {
            let outputTypeName = getOutputSchema(currentAgent)?.name() ?? "str"
            let span = agentSpan(name: currentAgent.name, handoffs: [], tools: [], outputType: outputTypeName)
            span.start()
            currentSpan = span
        }
        let agent = currentAgent
        let turnResult = try await withOptionalCurrentSpan(currentSpan) {
            let allTools = try await agent.getAllTools(contextWrapper)
            return try await resolveInterruptedTurn(
                agent: agent, allTools: allTools, originalInput: originalInput, originalPreStepItems: generatedItems,
                newResponse: lastResponse, processedResponse: processed, hooks: hooks, contextWrapper: contextWrapper,
                runConfig: runConfig, serverManagesConversation: serverConversationTracker != nil, errorHandlers: errorHandlers
            )
        }
        toolUseTracker.recordProcessedResponse(currentAgent, processed)
        originalInput = turnResult.originalInput
        if let owned = turnResult.nestedHistoryOwnedItems { nestedHistoryOwnedItems = owned }
        let (resumedGenerated, turnSessionItems) = resumedTurnItems(turnResult)
        generatedItems = resumedGenerated
        sessionItems.append(contentsOf: turnSessionItems)
        runState.generatedItems = generatedItems
        runState.sessionItems = sessionItems
        runState.currentStep = .runAgain
        syncStreamed()
        streamStepItemsIfNeeded(turnSessionItems)

        let isFinal: Bool
        if case .finalOutput = turnResult.nextStep { isFinal = true } else { isFinal = false }
        let isInterruption: Bool
        if case .interruption = turnResult.nextStep { isInterruption = true } else { isInterruption = false }
        if sessionPersistenceEnabled, !turnSessionItems.isEmpty, !isFinal,
           !(isInterruption && shouldDeferInterruptedSessionItems(currentAgent, runConfig)) {
            runState.currentTurnPersistedItemCount = try await saveResumedTurnItems(
                runState: runState, session: session, items: turnSessionItems,
                persistedCount: runState.currentTurnPersistedItemCount, responseId: turnResult.modelResponse.responseId,
                reasoningItemIdPolicy: runState.reasoningItemIdPolicy, store: storeSetting
            )
        }
        isResumedState = false

        switch turnResult.nextStep {
        case .interruption(let interruptions):
            toolInputGuardrailResults.append(contentsOf: turnResult.toolInputGuardrailResults)
            toolOutputGuardrailResults.append(contentsOf: turnResult.toolOutputGuardrailResults)
            pendingResult = buildInterruptionResult(interruptions, turnResult: turnResult, resultInput: originalInput)
        case .runAgain:
            break
        case .finalOutput(let output):
            toolInputGuardrailResults.append(contentsOf: turnResult.toolInputGuardrailResults)
            toolOutputGuardrailResults.append(contentsOf: turnResult.toolOutputGuardrailResults)
            pendingResult = try await finishWithFinalOutput(output, turnResult: turnResult, turnSessionItems: turnSessionItems, resultInput: turnResult.originalInput)
        case .handoff(let newAgent):
            toolInputGuardrailResults.append(contentsOf: turnResult.toolInputGuardrailResults)
            toolOutputGuardrailResults.append(contentsOf: turnResult.toolOutputGuardrailResults)
            switchAgent(to: newAgent, turnResult: turnResult)
        }
    }

    private func streamStepItemsIfNeeded(_ items: [RunItem]) {
        guard let streamed else { return }
        streamStepItems(items, to: streamed)
    }

    /// Runs guardrails on input staged with `RunState.addInput` and admits it into history.
    private func admitPendingInput() async throws {
        let pending = runState.pendingInputItems
        guard !pending.isEmpty else { return }
        let guardrails = currentAgent.inputGuardrails + (runConfig.inputGuardrails ?? [])
        let sink = SharedBox<[InputGuardrailResult]>([])
        defer {
            inputGuardrailResults.append(contentsOf: sink.value)
            runState.inputGuardrailResults = inputGuardrailResults
        }
        try await runInputGuardrails(currentAgent, guardrails, .items(pending), contextWrapper, sink: sink)
        let admitted: [RunItem] = pending.map { InputItem(agent: currentAgent, rawItem: $0) }
        if sessionPersistenceEnabled {
            try await saveResultToSession(session, originalInput: .items(pending), newItems: [], runState: nil, store: storeSetting)
        }
        generatedItems.append(contentsOf: admitted)
        sessionItems.append(contentsOf: admitted)
        pendingServerItems?.append(contentsOf: admitted)
        runState.clearPendingInput()
        runState.generatedItems = generatedItems
        runState.sessionItems = sessionItems
    }

    // MARK: Max turns

    private func handleMaxTurnsExceeded(_ maxTurns: Int) async throws -> RunResult {
        if let currentSpan {
            attachErrorToSpan(currentSpan, SpanError(message: "Max turns exceeded", data: ["max_turns": .int(maxTurns)]))
        }
        let error = MaxTurnsExceeded("Max turns (\(maxTurns)) exceeded")
        let runData = buildRunErrorData(
            input: originalInput, newItems: sessionItems, rawResponses: modelResponses, lastAgent: currentAgent,
            reasoningItemIdPolicy: reasoningItemIdPolicy
        )
        guard let handlerResult = try await resolveRunErrorHandlerResult(
            errorHandlers: errorHandlers, errorKind: "max_turns", error: error, contextWrapper: contextWrapper, runData: runData
        ) else {
            throw error
        }
        if streamed != nil {
            try await persistStreamInputIfNeeded()
        }
        let validated = try validateHandlerFinalOutput(currentAgent, handlerResult.finalOutput)
        let synthesized = createMessageOutputItem(currentAgent, formatFinalOutputText(currentAgent, validated))
        try await runFinalOutputHooks(agent: currentAgent, hooks: hooks, contextWrapper: contextWrapper, finalOutput: validated)
        let sink = SharedBox<[OutputGuardrailResult]>([])
        do {
            try await runOutputGuardrails(currentAgent.outputGuardrails + (runConfig.outputGuardrails ?? []), currentAgent, validated, contextWrapper, sink: sink)
            outputGuardrailResults.append(contentsOf: sink.value)
        } catch {
            outputGuardrailResults.append(contentsOf: sink.value)
            if !(error is OutputGuardrailTripwireTriggered) {
                _ = try await saveMaxTurnsItems(handlerResult.includeInHistory ? [synthesized] : [])
            }
            syncStreamed()
            throw error
        }
        // The synthesized output is a new item, so the previous turn's persisted count does not apply.
        runState.currentTurnPersistedItemCount = try await saveMaxTurnsItems(handlerResult.includeInHistory ? [synthesized] : [])
        currentTurn = maxTurns
        streamed?.update { $0.finalOutput = validated }
        if let streamed, handlerResult.includeInHistory {
            streamStepItems([synthesized], to: streamed)
        }
        return makeResult(input: originalInput, finalOutput: validated, interruptions: runState.pendingInterruptions)
    }

    /// Persists and records max-turns handler items. Returns the number of items persisted.
    private func saveMaxTurnsItems(_ items: [RunItem]) async throws -> Int {
        var saved = 0
        if sessionPersistenceEnabled, !items.isEmpty, !inputGuardrailsTriggered(attemptInputGuardrailResults) {
            saved = try await saveResultToSession(
                session, originalInput: .items([]), newItems: items, runState: nil,
                reasoningItemIdPolicy: reasoningItemIdPolicy, store: storeSetting
            )
        }
        guard !items.isEmpty else { return saved }
        generatedItems.append(contentsOf: items)
        sessionItems.append(contentsOf: items)
        return saved
    }
}
