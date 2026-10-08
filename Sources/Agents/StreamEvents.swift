import Foundation

/// A streaming event from the LLM, passed through directly from the model.
public struct RawResponsesStreamEvent: Sendable {
    /// The raw Responses API streaming event.
    public var data: TResponseStreamEvent
    public var type: String { "raw_response_event" }
}

/// A streaming event that wraps a `RunItem` as the agent processes the LLM response.
public struct RunItemStreamEvent: Sendable {
    /// The event name: `message_output_created`, `handoff_requested`, `handoff_occured` (sic),
    /// `tool_called`, `tool_search_called`, `tool_search_output_created`, `tool_output`,
    /// `reasoning_item_created`, `mcp_approval_requested`, `mcp_approval_response`, or
    /// `mcp_list_tools`.
    public var name: String
    /// The item that was created.
    public var item: RunItem
    public var type: String { "run_item_stream_event" }
}

/// Notifies that a new agent is running.
public struct AgentUpdatedStreamEvent: Sendable {
    public var newAgent: Agent
    public var type: String { "agent_updated_stream_event" }
}

/// A streaming event from an agent run.
public enum StreamEvent: Sendable {
    case rawResponse(RawResponsesStreamEvent)
    case runItem(RunItemStreamEvent)
    case agentUpdated(AgentUpdatedStreamEvent)

    /// The upstream `type` discriminator.
    public var type: String {
        switch self {
        case .rawResponse(let event): return event.type
        case .runItem(let event): return event.type
        case .agentUpdated(let event): return event.type
        }
    }
}
