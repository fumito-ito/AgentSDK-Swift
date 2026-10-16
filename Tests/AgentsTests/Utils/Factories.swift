import Foundation
@testable import Agents

// Port of tests/utils/factories.py.
//
// Swift: `TestSessionState` builds sandbox state, which is out of scope (porting/decisions.md D2).

/// Builds a `function_call` payload with common defaults.
func makeToolCall(
    _ callId: String = "call_1",
    name: String = "test_tool",
    namespace: String? = nil,
    status: String? = "completed",
    arguments: String = "{}"
) -> ResponseFunctionToolCall {
    ResponseFunctionToolCall(callId: callId, name: name, arguments: arguments, namespace: namespace, status: status)
}

/// Creates a `ToolApprovalItem` backed by a function call.
///
/// Swift: upstream distinguishes an omitted `tool_lookup_key` (derive it) from an explicit `None`;
/// pass `deriveLookupKey: false` for the explicit `None` case.
func makeToolApprovalItem(
    _ agent: Agent,
    callId: String = "call_1",
    name: String = "test_tool",
    namespace: String? = nil,
    allowBareNameAlias: Bool = false,
    status: String? = "completed",
    arguments: String = "{}",
    toolLookupKey: FunctionToolLookupKey? = nil,
    deriveLookupKey: Bool = true
) -> ToolApprovalItem {
    let item = ToolApprovalItem(
        agent: agent,
        rawItem: makeToolCall(callId, name: name, namespace: namespace, status: status, arguments: arguments).json,
        toolNamespace: namespace,
        toolLookupKey: toolLookupKey ?? (deriveLookupKey ? getFunctionToolLookupKey(name, namespace) : nil),
        allowBareNameAlias: allowBareNameAlias
    )
    if !deriveLookupKey, toolLookupKey == nil { item.toolLookupKey = nil }
    return item
}

/// Creates a minimal assistant output message.
func makeMessageOutput(messageId: String = "msg_1", text: String = "Hello", status: String = "completed") -> TResponseOutputItem {
    ResponseOutputMessage(id: messageId, content: [ResponseOutputText(text: text).json], status: status).json
}

/// Creates a `RunState` with sensible defaults for tests.
func makeRunState(
    _ agent: Agent,
    context: RunContextWrapper? = nil,
    originalInput: RunInput = "input",
    maxTurns: Int? = 3
) -> RunState {
    RunState(
        context: context ?? RunContextWrapper(context: JSONObject()),
        originalInput: originalInput,
        startingAgent: agent,
        maxTurns: maxTurns
    )
}

/// Serializes and restores a `RunState`, optionally mutating the JSON in between.
func roundtripState(_ agent: Agent, _ state: RunState, mutateJSON: ((JSONObject) -> JSONObject)? = nil) async throws -> RunState {
    var json = try state.toJSON()
    if let mutateJSON { json = mutateJSON(json) }
    return try await RunState.fromJSON(agent, json)
}
