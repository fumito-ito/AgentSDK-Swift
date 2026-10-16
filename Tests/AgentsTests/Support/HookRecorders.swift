import Foundation
@testable import Agents

// Equivalents of the `RunHooksForTests` / `AgentHooksForTests` classes shared by
// tests/test_run_hooks.py, tests/test_agent_hooks.py, tests/test_global_hooks.py, and
// tests/test_agent_llm_hooks.py. The upstream variants differ only in which events they count and
// when they record tool call IDs, so the differences are initializer options here.

/// Thread-safe hook event counts.
final class HookEventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var counts: [String: Int] = [:]
    private var ids: [String] = []

    var events: [String: Int] { lock.locked { counts } }
    var toolContextIds: [String] { lock.locked { ids } }

    func record(_ event: String, toolContext context: RunContextWrapper? = nil) {
        lock.locked {
            counts[event, default: 0] += 1
            if let toolContext = context as? ToolContext {
                ids.append(toolContext.toolCallId)
            }
        }
    }

    func reset() {
        lock.locked {
            counts.removeAll()
            ids.removeAll()
        }
    }
}

/// Counts run-level hook invocations.
final class RunHooksForTests: RunHooks, @unchecked Sendable {
    let log = HookEventLog()
    private let recordsLLMEvents: Bool
    private let toolContextEvents: Set<String>

    var events: [String: Int] { log.events }
    var toolContextIds: [String] { log.toolContextIds }

    init(recordsLLMEvents: Bool = true, toolContextEvents: Set<String> = ["on_tool_end"]) {
        self.recordsLLMEvents = recordsLLMEvents
        self.toolContextEvents = toolContextEvents
    }

    func reset() { log.reset() }

    private func record(_ event: String, _ context: RunContextWrapper? = nil) {
        log.record(event, toolContext: toolContextEvents.contains(event) ? context : nil)
    }

    override func onAgentStart(context: AgentHookContext, agent: AgentBase) async throws { record("on_agent_start") }
    override func onAgentEnd(context: AgentHookContext, agent: AgentBase, output: any Sendable) async throws { record("on_agent_end") }
    override func onHandoff(context: RunContextWrapper, fromAgent: AgentBase, toAgent: AgentBase) async throws { record("on_handoff") }
    override func onToolStart(context: RunContextWrapper, agent: AgentBase, tool: any Tool) async throws { record("on_tool_start", context) }

    override func onToolEnd(context: RunContextWrapper, agent: AgentBase, tool: any Tool, result: any Sendable) async throws {
        record("on_tool_end", context)
    }

    override func onLLMStart(context: RunContextWrapper, agent: Agent, systemPrompt: String?, inputItems: [TResponseInputItem]) async throws {
        if recordsLLMEvents { record("on_llm_start") }
    }

    override func onLLMEnd(context: RunContextWrapper, agent: Agent, response: ModelResponse) async throws {
        if recordsLLMEvents { record("on_llm_end") }
    }
}

/// Counts agent-level hook invocations.
final class AgentHooksForTests: AgentHooks, @unchecked Sendable {
    let log = HookEventLog()
    private let recordsLLMEvents: Bool
    private let toolContextEvents: Set<String>

    var events: [String: Int] { log.events }
    var toolContextIds: [String] { log.toolContextIds }

    init(recordsLLMEvents: Bool = true, toolContextEvents: Set<String> = []) {
        self.recordsLLMEvents = recordsLLMEvents
        self.toolContextEvents = toolContextEvents
    }

    func reset() { log.reset() }

    private func record(_ event: String, _ context: RunContextWrapper? = nil) {
        log.record(event, toolContext: toolContextEvents.contains(event) ? context : nil)
    }

    override func onStart(context: AgentHookContext, agent: AgentBase) async throws { record("on_start") }
    override func onEnd(context: AgentHookContext, agent: AgentBase, output: any Sendable) async throws { record("on_end") }
    override func onHandoff(context: RunContextWrapper, agent: AgentBase, source: AgentBase) async throws { record("on_handoff") }
    override func onToolStart(context: RunContextWrapper, agent: AgentBase, tool: any Tool) async throws { record("on_tool_start", context) }

    override func onToolEnd(context: RunContextWrapper, agent: AgentBase, tool: any Tool, result: any Sendable) async throws {
        record("on_tool_end", context)
    }

    override func onLLMStart(context: RunContextWrapper, agent: Agent, systemPrompt: String?, inputItems: [TResponseInputItem]) async throws {
        if recordsLLMEvents { record("on_llm_start") }
    }

    override func onLLMEnd(context: RunContextWrapper, agent: Agent, response: ModelResponse) async throws {
        if recordsLLMEvents { record("on_llm_end") }
    }
}

/// Captures `turnInput` from agent start hooks.
final class TurnInputCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var inputs: [[TResponseInputItem]] = []

    var capturedTurnInputs: [[TResponseInputItem]] { lock.locked { inputs } }

    func append(_ input: [TResponseInputItem]) {
        lock.locked { inputs.append(input) }
    }
}
