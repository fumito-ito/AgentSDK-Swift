import Foundation

// Port of src/agents/run_internal/run_steps.py: internal step/result data structures.

let notFinalOutput = ToolsToFinalOutputResult(isFinalOutput: false, finalOutput: nil)

struct ToolRunHandoff: Sendable {
    var handoff: Handoff
    var toolCall: ResponseFunctionToolCall
    /// The response item this run came from (upstream shares the raw item object), so
    /// invocation de-duplication can drop exactly that item.
    var callItem: RunItem?
}

struct ToolRunFunction: Sendable {
    var toolCall: ResponseFunctionToolCall
    var functionTool: FunctionTool
    /// The response item this run came from (upstream shares the raw item object), so
    /// invocation de-duplication can drop exactly that item.
    var callItem: RunItem?
}

struct ToolRunFunctionNotFound: Sendable {
    var toolCall: ResponseFunctionToolCall
    var toolName: String
    /// The response item this run came from (upstream shares the raw item object), so
    /// invocation de-duplication can drop exactly that item.
    var callItem: RunItem?
}

struct ToolRunComputerAction: Sendable {
    var toolCall: ResponseComputerToolCall
    var computerTool: ComputerTool
    /// The response item this run came from (upstream shares the raw item object), so
    /// invocation de-duplication can drop exactly that item.
    var callItem: RunItem?
}

struct ToolRunCustom: Sendable {
    var toolCall: ResponseCustomToolCall
    var customTool: CustomTool
    /// The response item this run came from (upstream shares the raw item object), so
    /// invocation de-duplication can drop exactly that item.
    var callItem: RunItem?
}

struct ToolRunMCPApprovalRequest: Sendable {
    var requestItem: McpApprovalRequest
    var mcpTool: HostedMCPTool
    /// The response item this run came from (upstream shares the raw item object), so
    /// invocation de-duplication can drop exactly that item.
    var callItem: RunItem?
}

struct ToolRunLocalShellCall: Sendable {
    var toolCall: LocalShellCall
    var localShellTool: LocalShellTool
    /// The response item this run came from (upstream shares the raw item object), so
    /// invocation de-duplication can drop exactly that item.
    var callItem: RunItem?
}

struct ToolRunShellCall: Sendable {
    var toolCall: JSONObject
    var shellTool: ShellTool
    /// The response item this run came from (upstream shares the raw item object), so
    /// invocation de-duplication can drop exactly that item.
    var callItem: RunItem?
}

struct ToolRunApplyPatchCall: Sendable {
    var toolCall: JSONObject
    var applyPatchTool: ApplyPatchTool
    /// The response item this run came from (upstream shares the raw item object), so
    /// invocation de-duplication can drop exactly that item.
    var callItem: RunItem?
}

/// The model response mapped to run items and local tool work.
struct ProcessedResponse: Sendable {
    var newItems: [RunItem]
    var handoffs: [ToolRunHandoff]
    var functions: [ToolRunFunction]
    var computerActions: [ToolRunComputerAction]
    var localShellCalls: [ToolRunLocalShellCall]
    var shellCalls: [ToolRunShellCall]
    var applyPatchCalls: [ToolRunApplyPatchCall]
    /// Names of all tools used, including hosted tools.
    var toolsUsed: [String]
    /// MCP approval requests (with or without callbacks).
    var mcpApprovalRequests: [ToolRunMCPApprovalRequest]
    /// Tool approval items awaiting a user decision.
    var interruptions: [ToolApprovalItem]
    var functionToolsNotFound: [ToolRunFunctionNotFound] = []
    var customToolCalls: [ToolRunCustom] = []

    /// Handoffs, functions, and local actions need local processing; hosted tools already ran.
    func hasToolsOrApprovalsToRun() -> Bool {
        !handoffs.isEmpty || !functions.isEmpty || !computerActions.isEmpty || !customToolCalls.isEmpty
            || !localShellCalls.isEmpty || !shellCalls.isEmpty || !applyPatchCalls.isEmpty
            || !mcpApprovalRequests.isEmpty || !functionToolsNotFound.isEmpty
    }

    func hasInterruptions() -> Bool { !interruptions.isEmpty }
}

/// The next step after a turn.
enum NextStep: Sendable {
    case handoff(Agent)
    case finalOutput(any Sendable)
    case runAgain
    case interruption([ToolApprovalItem])

    var interruptions: [ToolApprovalItem] {
        if case .interruption(let items) = self { return items }
        return []
    }
}

/// The result of one turn.
struct SingleStepResult: Sendable {
    /// The input items before `run()` was called. May be mutated by handoff input filters.
    var originalInput: RunInput
    /// The model response for the current step.
    var modelResponse: ModelResponse
    /// Items generated before the current step.
    var preStepItems: [RunItem]
    /// Items generated during the current step.
    var newStepItems: [RunItem]
    /// The next step to take.
    var nextStep: NextStep
    var toolInputGuardrailResults: [ToolInputGuardrailResult]
    var toolOutputGuardrailResults: [ToolOutputGuardrailResult]
    /// Full unfiltered items for session history, used instead of `newStepItems` when set.
    var sessionStepItems: [RunItem]?
    var outputGuardrailResults: [OutputGuardrailResult] = []
    /// Run items copied verbatim into nested input history by this step, or `nil` when the step
    /// did not rewrite the input history.
    var nestedHistoryOwnedItems: [RunItem]?
    /// The processed model response, needed for resuming from interruptions.
    var processedResponse: ProcessedResponse?

    /// Items generated during the agent run (everything after `originalInput`).
    var generatedItems: [RunItem] {
        preStepItems + (sessionStepItems ?? newStepItems)
    }
}

/// Tracks which tools an agent has used to support `resetToolChoice`.
final class AgentToolUseTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var agentToTools: [(agent: Agent, tools: [String])] = []
    private var agentMap: [String: Set<String>] = [:]
    private(set) var models: [any Model] = []

    func recordModel(_ model: any Model) {
        lock.lock()
        defer { lock.unlock() }
        if !models.contains(where: { $0 === model }) {
            models.append(model)
        }
    }

    func recordProcessedResponse(_ agent: Agent, _ processed: ProcessedResponse) {
        var iterator = processed.toolsUsed.makeIterator()
        var names: [String] = []
        for item in processed.newItems {
            let isToolItem = item is HandoffCallItem || item is ToolCallItem || item is ToolCallOutputItem
                || item is ToolSearchCallItem || item is ToolSearchOutputItem
            guard isToolItem else { continue }
            guard let name = iterator.next() else { break }
            if item is HandoffCallItem || item is ToolCallItem || item is ToolCallOutputItem {
                names.append(name)
            }
        }
        addToolUse(agent, names)
    }

    func addToolUse(_ agent: Agent, _ toolNames: [String]) {
        guard !toolNames.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        agentMap[agent.name, default: []].formUnion(toolNames)
        if let index = agentToTools.firstIndex(where: { $0.agent === agent }) {
            agentToTools[index].tools.append(contentsOf: toolNames)
        } else {
            agentToTools.append((agent, toolNames))
        }
    }

    func hasUsedTools(_ agent: Agent) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return agentToTools.first(where: { $0.agent === agent }).map { !$0.tools.isEmpty } ?? false
    }

    func asSerializable() -> [String: [String]] {
        lock.lock()
        defer { lock.unlock() }
        if !agentMap.isEmpty {
            return agentMap.mapValues { $0.sorted() }
        }
        var snapshot: [String: Set<String>] = [:]
        for (agent, names) in agentToTools {
            snapshot[agent.name, default: []].formUnion(names)
        }
        return snapshot.mapValues { $0.sorted() }
    }

    /// Converts the tracker into a snapshot keyed by stable agent identity (upstream
    /// `serialize_tool_use_tracker`), so duplicate-name agents keep separate entries.
    func serialize(startingAgent: Agent?) -> [String: [String]] {
        let identityKeys = startingAgent.map(buildAgentIdentityKeysById)
        lock.lock()
        defer { lock.unlock() }
        var snapshot: [String: [String]] = [:]
        for (agent, tools) in agentToTools {
            let key = identityKeys?[ObjectIdentifier(agent)] ?? agent.name
            snapshot[key, default: []].append(contentsOf: tools)
        }
        return snapshot
    }

    /// Seeds the tracker from a serialized snapshot (upstream `hydrate_tool_use_tracker`).
    ///
    /// Keys are resolved as agent identities first, then as agent names; unknown keys are skipped.
    func hydrate(from snapshot: [String: [String]], startingAgent: Agent) {
        guard !snapshot.isEmpty else { return }
        let agentsByName = buildAgentMap(startingAgent)
        let agentsByIdentity = buildAgentIdentityMap(startingAgent)
        for (key, tools) in snapshot {
            guard let agent = agentsByIdentity[key] ?? agentsByName[key] else { continue }
            addToolUse(agent, tools)
        }
    }
}
