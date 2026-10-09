import Foundation
import Testing
@testable import Agents

// Ported from tests/test_items_helpers.py.
//
// Swift: agent release/weak-reference tests, pydantic validation (`ValidationError`,
// `model_construct`, `TypeAdapter` round trips), and deep-copy/mutation checks are skipped:
// items hold their agents strongly and JSON values are copied on write.
@Suite struct ItemsHelpersTests {
    private func makeMessage(_ content: [JSONValue]) -> TResponseOutputItem {
        ["id": "msg123", "content": .array(content), "role": "assistant", "status": "completed", "type": "message"]
    }

    private func text(_ value: String) -> JSONValue {
        ["annotations": [], "text": .string(value), "type": "output_text", "logprobs": []]
    }

    private func refusal(_ value: String) -> JSONValue {
        ["refusal": .string(value), "type": "refusal"]
    }

    private let functionCall: TResponseOutputItem = ["id": "tool123", "arguments": "{}", "call_id": "call123", "name": "func", "type": "function_call"]

    // MARK: - Extraction

    @Test func testExtractLastContentOfTextMessage() throws {
        #expect(try ItemHelpers.extractLastContent(makeMessage([text("Hello "), text("world!")])) == "world!")
    }

    @Test func testExtractLastContentOfRefusalMessage() throws {
        #expect(try ItemHelpers.extractLastContent(makeMessage([text("Before refusal"), refusal("I cannot do that")])) == "I cannot do that")
    }

    @Test func testExtractLastContentNonMessageReturnsEmpty() throws {
        #expect(try ItemHelpers.extractLastContent(functionCall) == "")
    }

    @Test func testExtractLastTextReturnsTextOnly() {
        #expect(ItemHelpers.extractLastText(makeMessage([text("part1"), text("part2")])) == "part2")
        #expect(ItemHelpers.extractLastText(makeMessage([text("part1"), refusal("no")])) == nil)
    }

    @Test func testExtractTextConcatenatesAllTextSegments() {
        #expect(ItemHelpers.extractText(makeMessage([text("part1"), refusal("no"), text("part2")])) == "part1part2")
        #expect(ItemHelpers.extractText(functionCall) == nil)
    }

    @Test func testExtractTextToleratesNoneTextContent() {
        let noneText: JSONValue = ["annotations": [], "text": nil, "type": "output_text", "logprobs": []]
        #expect(ItemHelpers.extractText(makeMessage([noneText])) == nil)
        #expect(ItemHelpers.extractText(makeMessage([text("hello"), noneText])) == "hello")
        #expect(ItemHelpers.extractText(makeMessage([noneText, text("hello")])) == "hello")
    }

    @Test func testInputToNewInputListFromString() {
        #expect(ItemHelpers.inputToNewInputList("hi") == [["content": "hi", "role": "user"]])
    }

    @Test func testTextMessageOutputConcatenatesTextSegments() {
        let item = MessageOutputItem(agent: Agent(name: "test"), rawItem: makeMessage([text("a"), refusal("denied"), text("b")]))
        #expect(ItemHelpers.textMessageOutput(item) == "ab")
    }

    @Test func testTextMessageOutputsAcrossListOfRunItems() {
        let item1 = MessageOutputItem(agent: Agent(name: "test"), rawItem: makeMessage([text("foo")]))
        let item2 = MessageOutputItem(agent: Agent(name: "test"), rawItem: makeMessage([text("bar")]))
        let reasoning = ReasoningItem(agent: Agent(name: "test"), rawItem: ["id": "rid", "summary": [], "type": "reasoning"])
        #expect(ItemHelpers.textMessageOutputs([item1, reasoning, item2]) == "foobar")
    }

    // MARK: - Run item conversion

    @Test func testHandoffOutputItemConvertsProtocolPayload() throws {
        let item = HandoffOutputItem(
            agent: Agent(name: "owner"),
            rawItem: ["type": "function_call_output", "call_id": "call-123", "output": "ok"],
            sourceAgent: Agent(name: "source"),
            targetAgent: Agent(name: "target")
        )
        let converted = try item.toInputItem()
        #expect(converted.string("type") == "function_call_output")
        #expect(converted.string("call_id") == "call-123")
        #expect(converted.string("output") == "ok")
    }

    @Test func testHandoffOutputItemStringifiesObjectOutput() throws {
        let item = HandoffOutputItem(
            agent: Agent(name: "owner"),
            rawItem: ["type": "function_call_output", "call_id": "call-obj", "output": ["assistant": "Weather Assistant"]],
            sourceAgent: Agent(name: "source"),
            targetAgent: Agent(name: "target")
        )
        let converted = try item.toInputItem()
        #expect(converted.string("call_id") == "call-obj")
        #expect(converted["output"] == ["assistant": "Weather Assistant"])
    }

    @Test func testToolCallOutputItemPreservesFunctionOutputStructure() throws {
        let rawItem: JSONObject = ["type": "function_call_output", "call_id": "call-keep", "output": [["type": "output_text", "text": "value"]]]
        let payload = try ToolCallOutputItem(agent: Agent(name: "tester"), rawItem: rawItem, output: "value").toInputItem()
        #expect(payload.string("type") == "function_call_output")
        #expect(payload["output"] == rawItem["output"])
    }

    @Test func testToolCallOutputItemConstructsFunctionCallOutputDict() throws {
        let raw: JSONObject = ["id": "call-abc", "arguments": #"{"x": 1}"#, "call_id": "call-abc", "name": "do_something", "type": "function_call"]
        let call = try #require(ResponseFunctionToolCall(json: raw))
        let payload = try ItemHelpers.toolCallOutputItem(call, output: "result-string")
        #expect(payload.string("type") == "function_call_output")
        #expect(payload.string("call_id") == "call-abc")
        #expect(payload.string("output") == "result-string")
    }

    // MARK: - ModelResponse.toInputItems

    private func roundTrip(_ output: TResponseOutputItem) -> [TResponseInputItem] {
        ModelResponse(output: [output], usage: Usage(), responseId: nil).toInputItems()
    }

    @Test func testToInputItemsForMessage() {
        let message = makeMessage([text("hello")])
        #expect(roundTrip(message) == [message])
    }

    @Test func testToInputItemsForFunctionCall() {
        #expect(roundTrip(functionCall) == [functionCall])
    }

    @Test func testToInputItemsForFileSearchCall() {
        let call: TResponseOutputItem = ["id": "fs1", "queries": ["query"], "status": "completed", "type": "file_search_call"]
        #expect(roundTrip(call) == [call])
    }

    @Test func testToInputItemsForWebSearchCall() {
        let call: TResponseOutputItem = ["id": "w1", "action": ["type": "search", "query": "query"], "status": "completed", "type": "web_search_call"]
        #expect(roundTrip(call) == [call])
    }

    @Test func testToInputItemsForComputerCallClick() {
        let call: TResponseOutputItem = [
            "id": "comp1", "action": ["type": "click", "x": 1, "y": 2, "button": "left"], "call_id": "comp1",
            "pending_safety_checks": [], "status": "completed", "type": "computer_call",
        ]
        #expect(roundTrip(call) == [call])
    }

    @Test func testToInputItemsForComputerCallBatchedActions() {
        let call: TResponseOutputItem = [
            "id": "comp2",
            "actions": [["type": "click", "x": 3, "y": 4, "button": "left"], ["type": "type", "text": "hello"]],
            "call_id": "comp2", "pending_safety_checks": [], "status": "completed", "type": "computer_call",
        ]
        #expect(roundTrip(call) == [call])
    }

    @Test func testToInputItemsForReasoning() {
        let reasoning: TResponseOutputItem = ["id": "rid1", "summary": [["text": "why", "type": "summary_text"]], "type": "reasoning"]
        #expect(roundTrip(reasoning) == [reasoning])
    }

    @Test func testToInputItemsForToolSearchStripsCreatedBy() {
        let call: TResponseOutputItem = [
            "id": "tsc_123", "call_id": "call_tsc_123", "arguments": ["query": "profile"], "execution": "server",
            "status": "completed", "type": "tool_search_call", "created_by": "server",
        ]
        let output: TResponseOutputItem = [
            "id": "tso_123", "call_id": "call_tsc_123", "execution": "server", "status": "completed", "tools": [],
            "type": "tool_search_output", "created_by": "server",
        ]
        let items = ModelResponse(output: [call, output], usage: Usage(), responseId: nil).toInputItems()
        #expect(items == [call.removing("created_by"), output.removing("created_by")])
    }

    @Test func testToInputItemsStripsCreatedByForNonToolSearchItems() {
        let applyPatch: TResponseOutputItem = [
            "id": "apc_1", "call_id": "call_1", "type": "apply_patch_call", "status": "completed",
            "operation": ["type": "delete_file", "path": "foo.py"], "created_by": "program_1",
        ]
        let functionOutput: TResponseOutputItem = [
            "id": "fco_1", "call_id": "call_1", "type": "function_call_output", "output": "done", "status": "completed", "created_by": "program_1",
        ]
        let items = ModelResponse(output: [applyPatch, functionOutput], usage: Usage(), responseId: nil).toInputItems()
        #expect(items == [applyPatch.removing("created_by"), functionOutput.removing("created_by")])
        #expect(items.allSatisfy { $0["created_by"] == nil })
    }

    @Test func testToolCallItemToInputItemStripsCreatedBy() throws {
        let raw: JSONObject = ["id": "fc_1", "arguments": "{}", "call_id": "call_1", "name": "lookup", "type": "function_call", "created_by": "server"]
        let item = ToolCallItem(agent: Agent(name: "A"), rawItem: raw)
        let replayed = try item.toInputItem()
        #expect(replayed.string("type") == "function_call")
        #expect(replayed["created_by"] == nil)
        #expect(item.rawItem.string("created_by") == "server")
    }

    @Test func testToolCallOutputItemToInputItemStripsCreatedBy() throws {
        let item = ToolCallOutputItem(
            agent: Agent(name: "A"),
            rawItem: ["type": "function_call_output", "call_id": "call_1", "output": "ok", "created_by": "server"],
            output: "ok"
        )
        let replayed = try item.toInputItem()
        #expect(replayed.string("type") == "function_call_output")
        #expect(replayed["created_by"] == nil)
        #expect(item.rawItem.string("created_by") == "server")
    }

    @Test func testDictShellCallOutputItemToInputItemSanitizesWithoutMutation() throws {
        let raw: JSONObject = [
            "type": "shell_call_output", "call_id": "call_1", "status": "completed", "shell_output": "legacy",
            "provider_data": ["provider": "value"], "created_by": "server",
            "output": [["stdout": "ok", "stderr": "", "outcome": ["type": "exit", "exit_code": 0], "created_by": "server"]],
        ]
        let item = ToolCallOutputItem(agent: Agent(name: "A"), rawItem: raw, output: "ok")

        let replayed = try item.toInputItem()

        #expect(replayed == [
            "type": "shell_call_output", "call_id": "call_1",
            "output": [["stdout": "ok", "stderr": "", "outcome": ["type": "exit", "exit_code": 0]]],
        ])
        #expect(item.rawItem == raw)
    }

    @Test func testToInputItemsStripsNestedCreatedByFromShellCallOutput() {
        let raw: TResponseOutputItem = [
            "id": "sco_1", "call_id": "call_1", "type": "shell_call_output", "status": "completed", "created_by": "program_top",
            "output": [["outcome": ["type": "exit", "exit_code": 0], "stdout": "hi", "stderr": "", "created_by": "program_chunk"]],
        ]
        let expected: TResponseInputItem = [
            "id": "sco_1", "call_id": "call_1", "type": "shell_call_output", "status": "completed",
            "output": [["outcome": ["type": "exit", "exit_code": 0], "stdout": "hi", "stderr": ""]],
        ]
        #expect(roundTrip(raw) == [expected])
    }

    @Test func testToolCallItemToInputItemKeepsPayloadAPISafe() throws {
        let item = ToolCallItem(
            agent: Agent(name: "test", instructions: "test"),
            rawItem: ["id": "fc_1", "call_id": "call_1", "name": "my_tool", "arguments": "{}", "type": "function_call", "status": "completed"],
            description: "A helpful tool",
            title: "My Tool"
        )
        let result = try item.toInputItem()
        #expect(result.string("type") == "function_call")
        #expect(result["title"] == nil)
        #expect(result["description"] == nil)
    }

    // MARK: - Names and call IDs

    @Test func testToolCallItemToolName() {
        let agent = Agent(name: "test")
        #expect(ToolCallItem(agent: agent, rawItem: ["id": "fc1", "call_id": "call_1", "name": "my_tool", "arguments": "{}", "type": "function_call"]).toolName == "my_tool")
        #expect(ToolCallItem(agent: agent, rawItem: ["type": "function_call", "name": "dict_tool", "call_id": "call_1", "arguments": "{}"]).toolName == "dict_tool")
        #expect(ToolCallItem(agent: agent, rawItem: ["id": "fs1", "queries": ["q"], "status": "completed", "type": "file_search_call"]).toolName == nil)
    }

    @Test func testToolCallItemCallId() {
        let agent = Agent(name: "test")
        #expect(ToolCallItem(agent: agent, rawItem: ["id": "fc1", "call_id": "call_abc", "name": "t", "arguments": "{}", "type": "function_call"]).callId == "call_abc")
        #expect(ToolCallItem(agent: agent, rawItem: ["id": "fs_xyz", "queries": ["q"], "status": "completed", "type": "file_search_call"]).callId == "fs_xyz")
    }

    @Test func testToolCallOutputItemCallId() {
        let agent = Agent(name: "test")
        #expect(ToolCallOutputItem(agent: agent, rawItem: ["type": "function_call_output", "call_id": "call_out_1", "output": "ok"], output: "ok").callId == "call_out_1")
        #expect(ToolCallOutputItem(agent: agent, rawItem: ["type": "function_call_output", "output": "ok"], output: "ok").callId == nil)
    }
}
