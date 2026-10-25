import Foundation
import Testing
@testable import Agents

// Ported from tests/test_agent_tool_state.py.
//
// Swift: `test_drop_agent_tool_run_result_handles_cleared_globals` covers module globals cleared
// during interpreter shutdown; Swift entries live in a scope object, so it is not ported. Entries
// are keyed by tool call signature instead of object identity (porting/decisions.md D20), which
// changes the ambiguous-signature and garbage-collection cases; see the comments at those tests.
@Suite struct AgentToolStateTests {
    private func functionToolCall(_ name: String, _ arguments: String, callId: String) -> JSONObject {
        getFunctionToolCall(name, arguments: arguments, callId: callId)
    }

    private func makeRecord() -> AgentToolRunRecord {
        .serialized(RunState(context: RunContextWrapper(), originalInput: "", startingAgent: Agent(name: "nested")), interruptions: [])
    }

    private func state(_ record: AgentToolRunRecord?) -> RunState? {
        guard case .serialized(let state, _)? = record else { return nil }
        return state
    }

    @Test func testAgentToolStateScopeHelpersTolerateMissingOrReadonlyContexts() {
        let context = RunContextWrapper()
        let scope = AgentToolStateScope(id: "scope-1")

        setAgentToolStateScope(nil, AgentToolStateScope(id: "ignored"))
        setAgentToolStateScope(context, scope)
        #expect(getAgentToolStateScope(context) === scope)

        setAgentToolStateScope(context, nil)
        #expect(getAgentToolStateScope(context) == nil)
        // Swift: every context wrapper accepts a scope, so the read-only object case does not apply.
    }

    @Test func testAgentToolRunResultSupportsSignatureFallbackAcrossInstances() {
        let scope = AgentToolStateScope(id: "scope-1")
        let originalCall = functionToolCall("lookup_account", "{}", callId: "call-1")
        let restoredCall = functionToolCall("lookup_account", "{}", callId: "call-1")
        let runResult = makeRecord()

        recordAgentToolRunResult(originalCall, runResult, scope: scope)

        #expect(state(peekAgentToolRunResult(restoredCall, scope: scope)) === state(runResult))
        #expect(state(consumeAgentToolRunResult(restoredCall, scope: scope)) === state(runResult))
        #expect(peekAgentToolRunResult(originalCall, scope: scope) == nil)
        #expect(scope.isEmpty)
    }

    // Swift: upstream keeps two results recorded for distinct tool call objects with the same
    // signature and refuses an ambiguous signature lookup. Swift tool calls have no identity, so
    // the later record replaces the earlier one and a signature lookup finds it.
    @Test func testAgentToolRunResultReturnsNoneForAmbiguousSignatureMatches() {
        let scope = AgentToolStateScope(id: "scope-1")
        let firstCall = functionToolCall("lookup_account", "{}", callId: "call-1")
        let secondCall = functionToolCall("lookup_account", "{}", callId: "call-1")
        let restoredCall = functionToolCall("lookup_account", "{}", callId: "call-1")
        let firstResult = makeRecord()
        let secondResult = makeRecord()

        recordAgentToolRunResult(firstCall, firstResult, scope: scope)
        recordAgentToolRunResult(secondCall, secondResult, scope: scope)

        #expect(state(peekAgentToolRunResult(restoredCall, scope: scope)) === state(secondResult))
        #expect(peekAgentToolRunResult(restoredCall, scope: AgentToolStateScope(id: "other-scope")) == nil)

        dropAgentToolRunResult(restoredCall, scope: scope)

        #expect(peekAgentToolRunResult(firstCall, scope: scope) == nil)
        #expect(scope.isEmpty)
    }

    @Test func testAgentToolRunResultKeepsSameCallIsolatedByScope() {
        let firstScope = AgentToolStateScope(id: "scope-1")
        let secondScope = AgentToolStateScope(id: "scope-2")
        let toolCall = functionToolCall("lookup_account", "{}", callId: "call-1")
        let firstResult = makeRecord()
        let secondResult = makeRecord()

        recordAgentToolRunResult(toolCall, firstResult, scope: firstScope)
        recordAgentToolRunResult(toolCall, secondResult, scope: secondScope)

        #expect(state(peekAgentToolRunResult(toolCall, scope: firstScope)) === state(firstResult))
        #expect(state(peekAgentToolRunResult(toolCall, scope: secondScope)) === state(secondResult))
        #expect(state(consumeAgentToolRunResult(toolCall, scope: firstScope)) === state(firstResult))
        #expect(state(peekAgentToolRunResult(toolCall, scope: secondScope)) === state(secondResult))
    }

    // Swift: upstream drops an entry when its tool call object is garbage collected. Swift ties
    // entries to the scope instead, so the nested state is released with its scope.
    @Test func testAgentToolRunResultIsDroppedWhenToolCallIsCollected() {
        weak var nestedState: RunState?
        do {
            let scope = AgentToolStateScope(id: "scope-1")
            let record = makeRecord()
            nestedState = state(record)
            recordAgentToolRunResult(functionToolCall("lookup_account", "{}", callId: "call-1"), record, scope: scope)
            #expect(nestedState != nil)
        }

        #expect(nestedState == nil)
    }
}
