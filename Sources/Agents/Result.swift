import Foundation

// Port of src/agents/result.py.
//
// Swift: `release_agents` and weak last-agent references are not ported; Swift results own their
// agents for their lifetime. Nested-history ownership filtering in `to_input_list` is not ported.

/// How `toInputList` converts a run into input items.
public enum ToInputListMode: String, Sendable {
    /// Converts all `newItems` into a full plain-item history.
    case preserveAll = "preserve_all"
    /// Prefers the canonical continuation input when handoff filtering rewrote model history.
    case normalized
}

/// Metadata for results produced by `Agent.asTool()`.
public struct AgentToolInvocation: Sendable, Equatable {
    public let toolName: String
    public let toolCallId: String
    public let toolArguments: String
}

/// The data shared by `RunResult` and `RunResultStreaming`.
public protocol RunResultBase: AnyObject, Sendable {
    /// The original input items, i.e. the items before `run()` was called. May be mutated by
    /// handoff input filters.
    var input: RunInput { get }
    /// The new items generated during the run: messages, tool calls, their outputs, etc.
    var newItems: [RunItem] { get }
    /// The raw LLM responses generated during the run.
    var rawResponses: [ModelResponse] { get }
    /// The output of the last agent.
    var finalOutput: (any Sendable)? { get }
    /// Guardrail results for the input messages.
    var inputGuardrailResults: [InputGuardrailResult] { get }
    /// Guardrail results for the final output of the agent.
    var outputGuardrailResults: [OutputGuardrailResult] { get }
    /// Tool input guardrail results from all tools executed during the run.
    var toolInputGuardrailResults: [ToolInputGuardrailResult] { get }
    /// Tool output guardrail results from all tools executed during the run.
    var toolOutputGuardrailResults: [ToolOutputGuardrailResult] { get }
    /// The context wrapper for the run.
    var contextWrapper: RunContextWrapper { get }
    /// The last agent that was run.
    var lastAgent: Agent { get }
    /// Pending tool approval requests for this run.
    var interruptions: [ToolApprovalItem] { get }
}

/// Replay data the runner records on concrete results.
protocol RunResultReplayData {
    var modelInputItems: [RunItem] { get }
    var replayFromModelInputItems: Bool { get }
    var reasoningItemIdPolicy: ReasoningItemIdPolicy? { get }
    /// Session items also present verbatim in the nested input history.
    var nestedHistoryOwnedItems: [RunItem] { get }
}

/// Drops `newItems` occurrences that `input` already carries verbatim as nested history.
func excludingNestedHistoryOwnedItems(_ newItems: [RunItem], owned: [RunItem], input: RunInput, policy: ReasoningItemIdPolicy?) -> [RunItem] {
    guard !owned.isEmpty, case .items(let inputItems) = input else { return newItems }
    var available: [String: Int] = [:]
    for item in inputItems {
        available[fingerprintInputItem(item), default: 0] += 1
    }
    let ownedIds = Set(owned.map { ObjectIdentifier($0) })
    return newItems.filter { item in
        guard ownedIds.contains(ObjectIdentifier(item)),
              let converted = try? runItemToInputItem(item, reasoningItemIdPolicy: policy) else { return true }
        let key = fingerprintInputItem(converted)
        guard let count = available[key], count > 0 else { return true }
        available[key] = count - 1
        return false
    }
}

extension RunResultBase {
    /// Casts the final output to `type`.
    ///
    /// Returns `nil` when the output is not of the given type, or throws `TypeError` when
    /// `raiseIfIncorrectType` is `true`.
    public func finalOutputAs<T>(_ type: T.Type, raiseIfIncorrectType: Bool = false) throws -> T? {
        if let value = finalOutput as? T { return value }
        if raiseIfIncorrectType {
            throw TypeError("Final output is not of type \(T.self)")
        }
        return nil
    }

    /// Creates an input-item view of this run.
    public func toInputList(mode: ToInputListMode = .preserveAll) throws -> [TResponseInputItem] {
        let originalItems = ItemHelpers.inputToNewInputList(input)
        let replay = self as? any RunResultReplayData
        let policy = replay?.reasoningItemIdPolicy
        let replayItems: [TResponseInputItem]
        switch mode {
        case .preserveAll:
            let retained = excludingNestedHistoryOwnedItems(newItems, owned: replay?.nestedHistoryOwnedItems ?? [], input: input, policy: policy)
            replayItems = try runItemsToInputItems(retained, reasoningItemIdPolicy: policy)
        case .normalized:
            let source = (replay?.replayFromModelInputItems ?? false) ? (replay?.modelInputItems ?? newItems) : newItems
            replayItems = try runItemsToInputItems(source, reasoningItemIdPolicy: policy)
        }
        return originalItems + replayItems
    }

    /// Metadata for results produced by `Agent.asTool()`; `nil` for top-level runs.
    public var agentToolInvocation: AgentToolInvocation? {
        guard let toolContext = contextWrapper as? ToolContext else { return nil }
        return AgentToolInvocation(toolName: toolContext.toolName, toolCallId: toolContext.toolCallId, toolArguments: toolContext.toolArguments)
    }

    /// The response ID of the last model response.
    public var lastResponseId: String? {
        rawResponses.last?.responseId
    }
}

/// Fields shared by results when converting back to a `RunState`.
struct ResultStateFields {
    var currentTurn = 0
    var lastProcessedResponse: ProcessedResponse?
    var currentTurnPersistedItemCount = 0
    var toolUseTrackerSnapshot: [String: [String]] = [:]
    var originalInput: RunInput?
    var conversationId: String?
    var previousResponseId: String?
    var autoPreviousResponseId = false
    var traceState: TraceState?
    var startingAgent: Agent?
    var pendingInput: [TResponseInputItem] = []
    var currentStep: RunStateStep?
}

func populateState(_ state: RunState, from result: any RunResultBase & RunResultReplayData, fields: ResultStateFields) -> RunState {
    state.currentAgent = result.lastAgent
    state.generatedItems = result.modelInputItems.isEmpty && !result.replayFromModelInputItems ? result.newItems : result.modelInputItems
    state.sessionItems = result.newItems
    state.modelResponses = result.rawResponses
    state.inputGuardrailResults = result.inputGuardrailResults
    state.outputGuardrailResults = result.outputGuardrailResults
    state.toolInputGuardrailResults = result.toolInputGuardrailResults
    state.toolOutputGuardrailResults = result.toolOutputGuardrailResults
    state.lastProcessedResponse = fields.lastProcessedResponse
    state.currentTurn = fields.currentTurn
    state.currentTurnPersistedItemCount = fields.currentTurnPersistedItemCount
    state.setToolUseTrackerSnapshot(fields.toolUseTrackerSnapshot)
    state.conversationId = fields.conversationId
    state.previousResponseId = fields.previousResponseId
    state.autoPreviousResponseId = fields.autoPreviousResponseId
    state.pendingInputItems = fields.pendingInput
    state.currentStep = fields.currentStep
    state.reasoningItemIdPolicy = result.reasoningItemIdPolicy
    if !result.interruptions.isEmpty {
        state.currentStep = .interruption(interruptions: result.interruptions, responseAccepted: false)
    }
    state.traceState = fields.traceState
    return state
}

/// The result of a non-streaming agent run.
public final class RunResult: RunResultBase, RunResultReplayData, @unchecked Sendable, CustomStringConvertible {
    public let input: RunInput
    public let newItems: [RunItem]
    public let rawResponses: [ModelResponse]
    public let finalOutput: (any Sendable)?
    public let lastAgent: Agent
    public let inputGuardrailResults: [InputGuardrailResult]
    public let outputGuardrailResults: [OutputGuardrailResult]
    public let toolInputGuardrailResults: [ToolInputGuardrailResult]
    public let toolOutputGuardrailResults: [ToolOutputGuardrailResult]
    public let contextWrapper: RunContextWrapper
    public let interruptions: [ToolApprovalItem]
    /// The maximum number of turns allowed for this run, or `nil` for no limit.
    public let maxTurns: Int?

    var modelInputItems: [RunItem] = []
    var replayFromModelInputItems = false
    var reasoningItemIdPolicy: ReasoningItemIdPolicy?
    var nestedHistoryOwnedItems: [RunItem] = []
    var stateFields = ResultStateFields()

    init(
        input: RunInput,
        newItems: [RunItem],
        rawResponses: [ModelResponse],
        finalOutput: (any Sendable)?,
        lastAgent: Agent,
        inputGuardrailResults: [InputGuardrailResult],
        outputGuardrailResults: [OutputGuardrailResult],
        toolInputGuardrailResults: [ToolInputGuardrailResult],
        toolOutputGuardrailResults: [ToolOutputGuardrailResult],
        contextWrapper: RunContextWrapper,
        interruptions: [ToolApprovalItem],
        maxTurns: Int?
    ) {
        self.input = input
        self.newItems = newItems
        self.rawResponses = rawResponses
        self.finalOutput = finalOutput
        self.lastAgent = lastAgent
        self.inputGuardrailResults = inputGuardrailResults
        self.outputGuardrailResults = outputGuardrailResults
        self.toolInputGuardrailResults = toolInputGuardrailResults
        self.toolOutputGuardrailResults = toolOutputGuardrailResults
        self.contextWrapper = contextWrapper
        self.interruptions = interruptions
        self.maxTurns = maxTurns
    }

    /// Creates a `RunState` from this result to resume execution, e.g. after approving tool calls.
    ///
    /// ```swift
    /// let result = try await Runner.run(agent, input: "Use the delete_file tool")
    /// if let interruption = result.interruptions.first {
    ///     let state = result.toState()
    ///     try state.approve(interruption)
    ///     let resumed = try await Runner.run(agent, state: state)
    /// }
    /// ```
    public func toState() -> RunState {
        let state = RunState(
            context: contextWrapper.copyForRunState(),
            originalInput: stateFields.originalInput ?? input,
            startingAgent: stateFields.startingAgent ?? lastAgent,
            maxTurns: maxTurns
        )
        return populateState(state, from: self, fields: stateFields)
    }

    public var description: String {
        prettyPrintResult(self)
    }
}

/// How a streaming run is cancelled.
public enum StreamCancelMode: String, Sendable {
    /// Stop immediately, cancelling all tasks.
    case immediate
    /// Complete the current turn gracefully before stopping.
    case afterTurn = "after_turn"
}

/// The result of a streaming agent run. Consume `streamEvents()` to receive semantic events as
/// they are generated.
///
/// The event stream throws `MaxTurnsExceeded` when the agent exceeds `maxTurns`, and a tripwire
/// error when a guardrail is tripped.
public final class RunResultStreaming: RunResultBase, RunResultReplayData, @unchecked Sendable, CustomStringConvertible {
    private let lock = NSLock()

    private var storage: Storage
    struct Storage {
        var input: RunInput
        var newItems: [RunItem] = []
        var rawResponses: [ModelResponse] = []
        var finalOutput: (any Sendable)?
        var currentAgent: Agent
        var currentTurn = 0
        var inputGuardrailResults: [InputGuardrailResult] = []
        var outputGuardrailResults: [OutputGuardrailResult] = []
        var toolInputGuardrailResults: [ToolInputGuardrailResult] = []
        var toolOutputGuardrailResults: [ToolOutputGuardrailResult] = []
        var interruptions: [ToolApprovalItem] = []
        var isComplete = false
        var cancelMode: StreamCancelMode?
        var storedError: (any Error)?
        var modelInputItems: [RunItem] = []
        var replayFromModelInputItems = false
        var reasoningItemIdPolicy: ReasoningItemIdPolicy?
        var nestedHistoryOwnedItems: [RunItem] = []
        var stateFields = ResultStateFields()
        var trace: (any Trace)?
    }

    public let contextWrapper: RunContextWrapper
    /// The maximum number of turns the agent can run for, or `nil` for no limit.
    public let maxTurns: Int?

    private let events: AsyncThrowingStream<StreamEvent, any Error>
    private let continuation: AsyncThrowingStream<StreamEvent, any Error>.Continuation
    private var runTask: Task<Void, Never>?
    private var eventsConsumed = false

    init(input: RunInput, currentAgent: Agent, contextWrapper: RunContextWrapper, maxTurns: Int?) {
        self.storage = Storage(input: input, currentAgent: currentAgent)
        self.contextWrapper = contextWrapper
        self.maxTurns = maxTurns
        var continuation: AsyncThrowingStream<StreamEvent, any Error>.Continuation!
        self.events = AsyncThrowingStream(bufferingPolicy: .unbounded) { continuation = $0 }
        self.continuation = continuation
    }

    private func read<T>(_ body: (Storage) -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body(storage)
    }

    func update(_ body: (inout Storage) -> Void) {
        lock.lock()
        defer { lock.unlock() }
        body(&storage)
    }

    public var input: RunInput { read(\.input) }
    public var newItems: [RunItem] { read(\.newItems) }
    public var rawResponses: [ModelResponse] { read(\.rawResponses) }
    /// The final output of the agent; `nil` until the agent has finished running.
    public var finalOutput: (any Sendable)? { read(\.finalOutput) }
    /// The agent that is currently running.
    public var currentAgent: Agent { read(\.currentAgent) }
    public var lastAgent: Agent { currentAgent }
    /// The current turn number.
    public var currentTurn: Int { read(\.currentTurn) }
    public var inputGuardrailResults: [InputGuardrailResult] { read(\.inputGuardrailResults) }
    public var outputGuardrailResults: [OutputGuardrailResult] { read(\.outputGuardrailResults) }
    public var toolInputGuardrailResults: [ToolInputGuardrailResult] { read(\.toolInputGuardrailResults) }
    public var toolOutputGuardrailResults: [ToolOutputGuardrailResult] { read(\.toolOutputGuardrailResults) }
    public var interruptions: [ToolApprovalItem] { read(\.interruptions) }
    /// Whether the agent has finished running.
    public var isComplete: Bool { read(\.isComplete) }
    /// The trace for this run, if tracing is enabled.
    public var trace: (any Trace)? { read(\.trace) }
    var modelInputItems: [RunItem] { read(\.modelInputItems) }
    var replayFromModelInputItems: Bool { read(\.replayFromModelInputItems) }
    var reasoningItemIdPolicy: ReasoningItemIdPolicy? { read(\.reasoningItemIdPolicy) }
    var nestedHistoryOwnedItems: [RunItem] { read(\.nestedHistoryOwnedItems) }
    var cancelMode: StreamCancelMode? { read(\.cancelMode) }

    /// The error raised by the run loop, if any.
    public var runLoopException: (any Error)? { read(\.storedError) }

    func attach(task: Task<Void, Never>) {
        lock.lock()
        runTask = task
        lock.unlock()
    }

    func emit(_ event: StreamEvent) {
        guard !isComplete || cancelMode == nil else { return }
        continuation.yield(event)
    }

    /// Marks the run complete and closes the event stream, rethrowing `error` to consumers.
    func finish(error: (any Error)? = nil) {
        update {
            $0.isComplete = true
            if let error, $0.storedError == nil { $0.storedError = error }
        }
        if let error, !(error is CancellationError && cancelMode == .immediate) {
            continuation.finish(throwing: error)
        } else {
            continuation.finish()
        }
    }

    /// Cancels the streaming run.
    ///
    /// - `.immediate` stops immediately and cancels all tasks (default).
    /// - `.afterTurn` lets the current turn finish (model response, pending tools, session
    ///   persistence) and stops before the next turn begins.
    ///
    /// Continue consuming `streamEvents()` after calling `cancel` so cancellation completes.
    public func cancel(mode: StreamCancelMode = .immediate) {
        update { $0.cancelMode = mode }
        guard mode == .immediate else { return }
        lock.lock()
        let task = runTask
        lock.unlock()
        task?.cancel()
        update { $0.isComplete = true }
        continuation.finish()
    }

    /// Streams semantic events as they are generated. The sequence can be consumed once.
    ///
    /// Throws `MaxTurnsExceeded` when the agent exceeds the turn limit and a tripwire error when
    /// a guardrail is tripped.
    public func streamEvents() -> AsyncThrowingStream<StreamEvent, any Error> {
        events
    }

    /// Creates a `RunState` from this result to resume execution.
    public func toState() -> RunState {
        let fields = read(\.stateFields)
        let state = RunState(
            context: contextWrapper.copyForRunState(),
            originalInput: fields.originalInput ?? input,
            startingAgent: fields.startingAgent ?? currentAgent,
            maxTurns: maxTurns
        )
        return populateState(state, from: self, fields: fields)
    }

    public var description: String {
        prettyPrintRunResultStreaming(self)
    }
}

// MARK: - Pretty printing (util/_pretty_print.py)

private func indent(_ text: String, _ level: Int) -> String {
    let prefix = String(repeating: "  ", count: level)
    return text.split(separator: "\n", omittingEmptySubsequences: false).map { prefix + $0 }.joined(separator: "\n")
}

private func finalOutputString(_ result: any RunResultBase) -> String {
    guard let output = result.finalOutput else { return "None" }
    if let string = output as? String { return string }
    if let json = try? jsonDump(output) { return json.jsonString(prettyPrinted: true) }
    return stringify(output)
}

func prettyPrintResult(_ result: RunResult) -> String {
    var output = "RunResult:"
    output += "\n- Last agent: Agent(name=\"\(result.lastAgent.name)\", ...)"
    output += "\n- Final output (\(result.finalOutput.map { String(describing: type(of: $0)) } ?? "NoneType")):\n\(indent(finalOutputString(result), 2))"
    output += "\n- \(result.newItems.count) new item(s)"
    output += "\n- \(result.rawResponses.count) raw response(s)"
    output += "\n- \(result.inputGuardrailResults.count) input guardrail result(s)"
    output += "\n- \(result.outputGuardrailResults.count) output guardrail result(s)"
    output += "\n(See `RunResult` for more details)"
    return output
}

func prettyPrintRunErrorDetails(_ details: RunErrorDetails) -> String {
    var output = "RunErrorDetails:"
    output += "\n- Last agent: Agent(name=\"\(details.lastAgent.name)\", ...)"
    output += "\n- \(details.newItems.count) new item(s)"
    output += "\n- \(details.rawResponses.count) raw response(s)"
    output += "\n- \(details.inputGuardrailResults.count) input guardrail result(s)"
    output += "\n(See `RunErrorDetails` for more details)"
    return output
}

func prettyPrintRunResultStreaming(_ result: RunResultStreaming) -> String {
    var output = "RunResultStreaming:"
    output += "\n- Current agent: Agent(name=\"\(result.currentAgent.name)\", ...)"
    output += "\n- Current turn: \(result.currentTurn)"
    output += "\n- Max turns: \(result.maxTurns.map(String.init) ?? "None")"
    output += "\n- Is complete: \(result.isComplete ? "True" : "False")"
    output += "\n- Final output (\(result.finalOutput.map { String(describing: type(of: $0)) } ?? "NoneType")):\n\(indent(finalOutputString(result), 2))"
    output += "\n- \(result.newItems.count) new item(s)"
    output += "\n- \(result.rawResponses.count) raw response(s)"
    output += "\n- \(result.inputGuardrailResults.count) input guardrail result(s)"
    output += "\n- \(result.outputGuardrailResults.count) output guardrail result(s)"
    output += "\n(See `RunResultStreaming` for more details)"
    return output
}
