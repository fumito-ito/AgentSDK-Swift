import Foundation

// Port of src/agents/agent_tool_state.py.
//
// Swift: upstream indexes nested agent-tool results by the identity of the tool call object, falls
// back to a signature for restored instances, and drops an entry when its tool call is garbage
// collected. Swift tool calls are JSON values without identity, so entries are keyed by the
// signature alone, and recording a second result for the same signature replaces the first.
// Entries live in an `AgentToolStateScope` that contexts and run states share by reference, so an
// entry is released with the last context or state that holds its scope (porting/decisions.md D20).

/// Converts a nested run result into a resumable `RunState`.
protocol RunStateConvertible: AnyObject {
    func toState() -> RunState
}

extension RunResult: RunStateConvertible {}
extension RunResultStreaming: RunStateConvertible {}

/// A nested `Agent.asTool()` run that a pending tool call can continue.
///
/// Upstream stores a `RunResult`, `RunResultStreaming`, `_AgentToolResumeCheckpoint`, or
/// `_SerializedAgentToolRunResult` and reads them by duck typing.
enum AgentToolRunRecord: @unchecked Sendable {
    /// A live nested run result.
    case result(any RunResultBase & RunStateConvertible)
    /// A nested `RunState` kept while an approved resume is in flight (upstream
    /// `_AgentToolResumeCheckpoint`).
    case resumeCheckpoint(RunState, approvalIdentities: Set<ToolInvocationIdentity>)
    /// A nested `RunState` restored from JSON (upstream `_SerializedAgentToolRunResult`).
    case serialized(RunState, interruptions: [ToolApprovalItem])

    /// The nested run's pending approvals.
    var interruptions: [ToolApprovalItem] {
        switch self {
        case .result(let result): return result.interruptions
        case .resumeCheckpoint(let state, _): return state.getInterruptions()
        case .serialized(_, let interruptions): return interruptions
        }
    }

    /// The nested run's final output; `nil` for stored states.
    var finalOutput: (any Sendable)? {
        if case .result(let result) = self { return result.finalOutput }
        return nil
    }

    /// The value exposed as `FunctionToolResult.agentRunResult`: the nested result itself when it
    /// is live, otherwise the stored nested state.
    var agentRunResultValue: any Sendable {
        switch self {
        case .result(let result): return result
        case .resumeCheckpoint(let state, _), .serialized(let state, _): return state
        }
    }

    /// Returns the nested run state. A live result creates a new state on each call.
    func toState() -> RunState {
        switch self {
        case .result(let result): return result.toState()
        case .resumeCheckpoint(let state, _), .serialized(let state, _): return state
        }
    }
}

/// The private cache scope for nested agent-tool results (upstream `_agent_tool_state_scope_id`).
final class AgentToolStateScope: @unchecked Sendable {
    /// The scope used when a context carries none (upstream `scope_id=None`).
    static let unscoped = AgentToolStateScope(id: "")

    let id: String
    private let lock = NSLock()
    private var records: [AgentToolCallSignature: AgentToolRunRecord] = [:]

    init(id: String = UUID().uuidString.lowercased()) {
        self.id = id
    }

    fileprivate func record(_ record: AgentToolRunRecord, for signature: AgentToolCallSignature) {
        lock.locked { records[signature] = record }
    }

    fileprivate func peek(_ signature: AgentToolCallSignature) -> AgentToolRunRecord? {
        lock.locked { records[signature] }
    }

    @discardableResult
    fileprivate func remove(_ signature: AgentToolCallSignature) -> AgentToolRunRecord? {
        lock.locked { records.removeValue(forKey: signature) }
    }

    /// Whether the scope stores no nested results.
    var isEmpty: Bool {
        lock.locked { records.isEmpty }
    }
}

/// A stable tool call signature for lookups across tool call instances.
struct AgentToolCallSignature: Hashable, Sendable {
    var callId: String?
    var name: String?
    var arguments: String?
    var type: String?
    var id: String?
    var status: String?

    init(_ toolCall: JSONObject) {
        callId = toolCall.string("call_id")
        name = toolCall.string("name")
        arguments = toolCall.string("arguments")
        type = toolCall.type
        id = toolCall.string("id")
        status = toolCall.string("status")
    }
}

/// Reads the private agent-tool cache scope from a context wrapper.
func getAgentToolStateScope(_ context: RunContextWrapper?) -> AgentToolStateScope? {
    context?.agentToolStateScope
}

/// Attaches or clears the private agent-tool cache scope on a context wrapper.
func setAgentToolStateScope(_ context: RunContextWrapper?, _ scope: AgentToolStateScope?) {
    context?.agentToolStateScope = scope
}

/// Stores the nested agent run result for a tool call.
func recordAgentToolRunResult(_ toolCall: JSONObject, _ record: AgentToolRunRecord, scope: AgentToolStateScope?) {
    (scope ?? .unscoped).record(record, for: AgentToolCallSignature(toolCall))
}

/// Keeps a live nested `RunState` checkpoint while an approved resume is in flight.
func recordAgentToolResumeState(
    _ toolCall: JSONObject,
    _ state: RunState,
    scope: AgentToolStateScope?,
    approvalItems: [ToolApprovalItem]? = nil
) {
    let items = approvalItems ?? state.getInterruptions()
    let identities = Set(items.compactMap { item in
        toolInvocationIdentity(item.rawItem, toolLookupKey: item.toolLookupKey, toolName: item.toolName)
    })
    recordAgentToolRunResult(toolCall, .resumeCheckpoint(state, approvalIdentities: identities), scope: scope)
}

/// Returns the live nested `RunState` stored in an in-flight resume checkpoint.
func getAgentToolResumeState(_ record: AgentToolRunRecord?) -> RunState? {
    if case .resumeCheckpoint(let state, _)? = record { return state }
    return nil
}

/// Returns whether an in-flight nested resume accepted the approval item.
func agentToolResumeCheckpointOwnsApproval(_ record: AgentToolRunRecord?, _ approvalItem: ToolApprovalItem) -> Bool {
    guard case .resumeCheckpoint(_, let identities)? = record,
          let identity = toolInvocationIdentity(
              approvalItem.rawItem, toolLookupKey: approvalItem.toolLookupKey, toolName: approvalItem.toolName
          ) else {
        return false
    }
    return identities.contains(identity)
}

/// Returns and drops the stored nested agent run result for a tool call.
func consumeAgentToolRunResult(_ toolCall: JSONObject, scope: AgentToolStateScope?) -> AgentToolRunRecord? {
    (scope ?? .unscoped).remove(AgentToolCallSignature(toolCall))
}

/// Returns the stored nested agent run result without removing it.
func peekAgentToolRunResult(_ toolCall: JSONObject, scope: AgentToolStateScope?) -> AgentToolRunRecord? {
    (scope ?? .unscoped).peek(AgentToolCallSignature(toolCall))
}

/// Drops the stored nested agent run result, if present.
func dropAgentToolRunResult(_ toolCall: JSONObject, scope: AgentToolStateScope?) {
    (scope ?? .unscoped).remove(AgentToolCallSignature(toolCall))
}
