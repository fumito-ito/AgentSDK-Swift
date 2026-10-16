import Foundation

// Port of src/agents/run_state.py.
//
// Swift: this first port covers in-process pause/resume (approvals, pending input, and the
// interrupted step). JSON serialization (`to_json`/`from_json`), nested agent-tool states,
// pending session-write checkpoints, and sandbox payloads are not ported yet; see
// porting/manifest.json.

/// The resumable step stored in a `RunState`.
enum RunStateStep: Sendable {
    case interruption(interruptions: [ToolApprovalItem], responseAccepted: Bool)
    case runAgain
}

/// A snapshot of an agent run, including context, usage, and interruptions.
///
/// `RunState` is the pause/resume boundary for human-in-the-loop flows: approve or reject the
/// pending interruptions, then pass the state back to `Runner.run` to continue.
public final class RunState: @unchecked Sendable {
    var currentTurn = 0
    var currentAgent: Agent?
    let startingAgent: Agent
    var originalInput: RunInput
    var modelResponses: [ModelResponse] = []
    let context: RunContextWrapper
    /// Items used to build model input when resuming; may be filtered by handoffs.
    var generatedItems: [RunItem] = []
    /// Full, unfiltered run items for session history.
    var sessionItems: [RunItem] = []
    var pendingInputItems: [TResponseInputItem] = []
    var maxTurns: Int?
    var conversationId: String?
    var previousResponseId: String?
    var autoPreviousResponseId = false
    var reasoningItemIdPolicy: ReasoningItemIdPolicy?
    var inputGuardrailResults: [InputGuardrailResult] = []
    var outputGuardrailResults: [OutputGuardrailResult] = []
    var toolInputGuardrailResults: [ToolInputGuardrailResult] = []
    var toolOutputGuardrailResults: [ToolOutputGuardrailResult] = []
    var currentStep: RunStateStep?
    var lastProcessedResponse: ProcessedResponse?
    var currentTurnPersistedItemCount = 0
    var toolUseTrackerSnapshot: [String: [String]] = [:]
    var traceState: TraceState?

    public init(
        context: RunContextWrapper,
        originalInput: RunInput,
        startingAgent: Agent,
        maxTurns: Int? = defaultMaxTurns,
        conversationId: String? = nil,
        previousResponseId: String? = nil,
        autoPreviousResponseId: Bool = false
    ) {
        self.context = context
        self.originalInput = originalInput
        self.startingAgent = startingAgent
        self.currentAgent = startingAgent
        self.maxTurns = maxTurns
        self.conversationId = conversationId
        self.previousResponseId = previousResponseId
        self.autoPreviousResponseId = autoPreviousResponseId
    }

    /// The usage accumulated so far.
    public var usage: Usage { context.usage }

    /// The agent that will handle the next turn.
    public var lastAgent: Agent { currentAgent ?? startingAgent }

    /// Input currently staged for the next resumed model call.
    public var pendingInput: [TResponseInputItem] { pendingInputItems }

    /// Stages input for admission immediately before the next resumed model call.
    public func addInput(_ input: RunInput) throws {
        guard let currentStep else {
            throw UserError("Cannot add input to a terminal RunState")
        }
        if let maxTurns, currentTurn >= maxTurns {
            throw UserError("Cannot add input to a RunState with no remaining model turns")
        }
        if case .interruption(let interruptions, let responseAccepted) = currentStep {
            if responseAccepted {
                throw UserError("Cannot add input while an accepted model response is awaiting local processing")
            }
            guard let currentAgent else {
                throw UserError("Cannot add input to a RunState without a current agent")
            }
            let interruptedNames = Set(interruptions.compactMap(\.toolName))
            let stopsBeforeNextModel: Bool
            switch currentAgent.toolUseBehavior {
            case .stopOnFirstTool, .custom: stopsBeforeNextModel = true
            case .stopAtTools(let names): stopsBeforeNextModel = !interruptedNames.isDisjoint(with: names)
            case .runLLMAgain: stopsBeforeNextModel = false
            }
            if stopsBeforeNextModel {
                throw UserError("Cannot add input to an interrupted RunState whose tool result may end the run")
            }
        }
        pendingInputItems.append(contentsOf: ItemHelpers.inputToNewInputList(input))
    }

    /// Removes all input staged for the next resumed model call.
    public func clearPendingInput() {
        pendingInputItems = []
    }

    /// Returns the pending interruptions for the current step.
    public func getInterruptions() -> [ToolApprovalItem] {
        guard case .interruption(let interruptions, _)? = currentStep else { return [] }
        return interruptions
    }

    /// Approves a tool call; rerun with this state to continue.
    ///
    /// For function tools, `alwaysApprove` applies only to the approval item's agent.
    public func approve(_ approvalItem: ToolApprovalItem, alwaysApprove: Bool = false) throws {
        try context.approveTool(approvalItem, alwaysApprove: alwaysApprove)
    }

    /// Rejects a tool call; rerun with this state to continue.
    ///
    /// When `rejectionMessage` is provided, that exact text is sent back to the model when the run
    /// resumes. Otherwise the run-level tool error formatter or the SDK default message is used.
    public func reject(_ approvalItem: ToolApprovalItem, alwaysReject: Bool = false, rejectionMessage: String? = nil) throws {
        try context.rejectTool(approvalItem, alwaysReject: alwaysReject, rejectionMessage: rejectionMessage)
    }

    public func setTrace(_ trace: (any Trace)?) {
        traceState = TraceState.fromTrace(trace)
    }

    public func setToolUseTrackerSnapshot(_ snapshot: [String: [String]]?) {
        toolUseTrackerSnapshot = snapshot ?? [:]
    }

    public func getToolUseTrackerSnapshot() -> [String: [String]] {
        toolUseTrackerSnapshot
    }
}
