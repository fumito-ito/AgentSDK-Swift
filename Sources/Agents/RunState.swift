import Foundation

// Port of src/agents/run_state.py. Serialization lives in RunStateSerialization.swift and
// restoration in RunStateDeserialization.swift (porting/decisions.md D13).
//
// Swift: nested agent-tool approval routing lives in RunStateNestedAgentTools.swift (D20).
// Pending session-write checkpoints, programmatic tool calling resume checks, and sandbox
// payloads are not ported yet; see porting/manifest.json.

/// The resumable step stored in a `RunState`.
enum RunStateStep: Sendable {
    case interruption(interruptions: [ToolApprovalItem], responseAccepted: Bool)
    case runAgain

    /// The pending approvals of an interruption step.
    var interruptionItems: [ToolApprovalItem] {
        if case .interruption(let items, _) = self { return items }
        return []
    }
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
    /// Whether `on_llm_end` hooks already started for the interrupted response.
    var llmEndHooksStarted = true
    var lastProcessedResponse: ProcessedResponse?
    /// Tracks whether `generatedItems` already include the current `lastProcessedResponse`.
    var generatedItemsLastProcessedMarker: String?
    var currentTurnPersistedItemCount = 0
    var toolUseTrackerSnapshot: [String: [String]] = [:]
    var traceState: TraceState?
    /// SDK-generated prompt cache key to preserve across resume flows.
    var generatedPromptCacheKey: String?
    /// Schema version the snapshot was loaded from.
    var schemaVersion = RunState.currentSchemaVersion
    /// The private cache scope for nested `Agent.asTool()` results of this run.
    var agentToolStateScope: AgentToolStateScope?

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
        self.agentToolStateScope = getAgentToolStateScope(context)
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

    /// Returns detached copies of the pending interruptions for the current step.
    ///
    /// Changing a returned item does not change the pending approval; pass it to `approve` or
    /// `reject` to record a decision.
    public func getInterruptions() -> [ToolApprovalItem] {
        pendingInterruptions.map { item in
            let copy = ToolApprovalItem(
                agent: item.agent,
                rawItem: item.rawItem,
                toolName: item.toolName,
                toolNamespace: item.toolNamespace,
                toolOrigin: item.toolOrigin,
                toolLookupKey: item.toolLookupKey,
                allowBareNameAlias: item.allowBareNameAlias
            )
            copy.toolLookupKey = item.toolLookupKey
            return copy
        }
    }

    /// The pending interruptions themselves, for SDK code that needs their identity.
    var pendingInterruptions: [ToolApprovalItem] {
        guard case .interruption(let interruptions, _)? = currentStep else { return [] }
        return interruptions
    }

    /// Approves a tool call; rerun with this state to continue.
    ///
    /// For function tools, `alwaysApprove` applies only to the approval item's agent.
    public func approve(_ approvalItem: ToolApprovalItem, alwaysApprove: Bool = false) throws {
        if let (nestedState, nestedItem) = try findNestedApprovalState(approvalItem) {
            try nestedState.approve(nestedItem, alwaysApprove: alwaysApprove)
            return
        }
        try context.approveTool(try findCurrentApprovalItem(approvalItem) ?? approvalItem, alwaysApprove: alwaysApprove)
    }

    /// Rejects a tool call; rerun with this state to continue.
    ///
    /// When `rejectionMessage` is provided, that exact text is sent back to the model when the run
    /// resumes. Otherwise the run-level tool error formatter or the SDK default message is used.
    public func reject(_ approvalItem: ToolApprovalItem, alwaysReject: Bool = false, rejectionMessage: String? = nil) throws {
        if let (nestedState, nestedItem) = try findNestedApprovalState(approvalItem) {
            try nestedState.reject(nestedItem, alwaysReject: alwaysReject, rejectionMessage: rejectionMessage)
            return
        }
        try context.rejectTool(
            try findCurrentApprovalItem(approvalItem) ?? approvalItem, alwaysReject: alwaysReject, rejectionMessage: rejectionMessage
        )
    }

    /// Captures trace metadata for serialization and resumption.
    public func setTrace(_ trace: (any Trace)?) {
        traceState = TraceState.fromTrace(trace)
    }

    /// Stores a copy of the serialized tool-use tracker data.
    public func setToolUseTrackerSnapshot(_ snapshot: [String: [String]]?) {
        toolUseTrackerSnapshot = snapshot ?? [:]
    }

    /// Stores how reasoning item IDs should appear in next-turn model input.
    public func setReasoningItemIdPolicy(_ policy: ReasoningItemIdPolicy?) {
        reasoningItemIdPolicy = policy
    }

    /// Returns a copy of the tool-use tracker snapshot.
    public func getToolUseTrackerSnapshot() -> [String: [String]] {
        toolUseTrackerSnapshot
    }
}
