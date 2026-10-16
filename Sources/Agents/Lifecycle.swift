import Foundation

/// Receives callbacks on lifecycle events in an agent run. Override the methods you need.
///
/// Swift: upstream `RunHooksBase` is a class with no-op methods; here it is a class with
/// overridable no-op methods so that subclasses can keep state.
open class RunHooks: @unchecked Sendable {
    public init() {}

    /// Called just before invoking the LLM for this agent.
    open func onLLMStart(context: RunContextWrapper, agent: Agent, systemPrompt: String?, inputItems: [TResponseInputItem]) async throws {}

    /// Called immediately after the LLM call returns for this agent.
    open func onLLMEnd(context: RunContextWrapper, agent: Agent, response: ModelResponse) async throws {}

    /// Called before the agent is invoked. Called each time the current agent changes.
    open func onAgentStart(context: AgentHookContext, agent: AgentBase) async throws {}

    /// Called when the agent produces a final output.
    open func onAgentEnd(context: AgentHookContext, agent: AgentBase, output: any Sendable) async throws {}

    /// Called when a handoff occurs.
    open func onHandoff(context: RunContextWrapper, fromAgent: AgentBase, toAgent: AgentBase) async throws {}

    /// Called immediately before a local tool is invoked.
    ///
    /// For function-tool invocations, `context` is typically a `ToolContext`.
    open func onToolStart(context: RunContextWrapper, agent: AgentBase, tool: any Tool) async throws {}

    /// Called immediately after a local tool is invoked.
    open func onToolEnd(context: RunContextWrapper, agent: AgentBase, tool: any Tool, result: any Sendable) async throws {}
}

/// Receives callbacks on lifecycle events for a specific agent. Set this on `agent.hooks`.
open class AgentHooks: @unchecked Sendable {
    public init() {}

    /// Called before the agent is invoked. Called each time the running agent is changed to this
    /// agent.
    open func onStart(context: AgentHookContext, agent: AgentBase) async throws {}

    /// Called when the agent produces a final output.
    open func onEnd(context: AgentHookContext, agent: AgentBase, output: any Sendable) async throws {}

    /// Called when the agent is being handed off to. `source` is the agent handing off.
    open func onHandoff(context: RunContextWrapper, agent: AgentBase, source: AgentBase) async throws {}

    /// Called immediately before a local tool is invoked.
    open func onToolStart(context: RunContextWrapper, agent: AgentBase, tool: any Tool) async throws {}

    /// Called immediately after a local tool is invoked.
    open func onToolEnd(context: RunContextWrapper, agent: AgentBase, tool: any Tool, result: any Sendable) async throws {}

    /// Called immediately before the agent issues an LLM call.
    open func onLLMStart(context: RunContextWrapper, agent: Agent, systemPrompt: String?, inputItems: [TResponseInputItem]) async throws {}

    /// Called immediately after the agent receives the LLM response.
    open func onLLMEnd(context: RunContextWrapper, agent: Agent, response: ModelResponse) async throws {}
}
