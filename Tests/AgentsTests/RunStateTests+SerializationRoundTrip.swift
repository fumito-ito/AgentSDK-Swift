import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_state.py (approval tests at the end of class TestRunState, and the
// classes TestSerializationRoundTrip, TestRunContextApprovals, and TestRunStateEdgeCases).
//
// Swift: the canonical tool invocation ledger (`_tool_invocation.py`) is not ported yet, so the
// approval tests that assert ledger binding (`test_schema_1_13_restores_pending_approval_binding_
// from_interruption` through `test_serialized_apply_patch_approval_binds_plural_operations`, and
// `test_context_override_discards_unbound_ids_from_previous_restore`) are pending in
// porting/manifest.json.

private func functionApproval(_ agent: Agent, name: String, callId: String, arguments: String = "") -> ToolApprovalItem {
    ToolApprovalItem(agent: agent, rawItem: ResponseFunctionToolCall(callId: callId, name: name, arguments: arguments, status: "completed").json)
}

/// Moves agent-owned function approvals into the legacy approvals map, as upstream tests do when
/// they downgrade a snapshot to an older schema.
private func moveFunctionToolApprovalsToLegacy(_ json: inout JSONObject) {
    guard var context = json["context"]?.objectValue else { return }
    var approvals = context["approvals"]?.objectValue ?? [:]
    for entry in context.removeValue(forKey: "function_tool_approvals")?.arrayValue ?? [] {
        if let key = entry.objectValue?.string("tool_key"), let decision = entry.objectValue?["decision"] {
            approvals[key] = decision
        }
    }
    context["approvals"] = .object(approvals)
    json["context"] = .object(context)
}

@Suite struct RunStateApprovalSerializationTests {
    @Test func testSerializesAndRestoresApprovals() async throws {
        let agent = Agent(name: "ApprovalAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        try state.approve(functionApproval(agent, name: "tool1", callId: "cid1"), alwaysApprove: true)
        try state.reject(functionApproval(agent, name: "tool2", callId: "cid2"), alwaysReject: true)

        let newState = try await RunState.fromString(agent, try state.toString())

        #expect(newState.context.isToolApproved(toolName: "tool1", callId: "cid1") == true)
        #expect(newState.context.isToolApproved(toolName: "tool2", callId: "cid2") == false)
        #expect(newState.context.getRejectionMessage(toolName: "tool2", callId: "cid2") == nil)
    }

    @Test(arguments: [true, false])
    func testExactCallOverrideRoundTripsWithStickyDefault(stickyApproved: Bool) async throws {
        let agent = Agent(name: "MixedApprovalAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        func approval(_ callId: String) -> ToolApprovalItem { functionApproval(agent, name: "tool1", callId: callId, arguments: "{}") }

        if stickyApproved {
            try state.approve(approval("sticky"), alwaysApprove: true)
            try state.reject(approval("exception"), rejectionMessage: "denied exactly")
        } else {
            try state.reject(approval("sticky"), alwaysReject: true, rejectionMessage: "denied by default")
            try state.approve(approval("exception"))
        }

        let serialized = try state.toJSON()
        #expect(serialized["$schemaVersion"] == .string(RunState.currentSchemaVersion))

        let restored = try await RunState.fromJSON(agent, serialized)
        #expect(restored.context.isToolApproved(toolName: "tool1", callId: "exception") == !stickyApproved)
        #expect(restored.context.isToolApproved(toolName: "tool1", callId: "other") == stickyApproved)
        #expect(restored.context.getRejectionMessage(toolName: "tool1", callId: "exception") == (stickyApproved ? "denied exactly" : nil))
    }

    @Test(arguments: [true, false])
    func testSchema115MixedApprovalRecordKeepsExactDecision(stickyApproved: Bool) async throws {
        let agent = Agent(name: "LegacyMixedApprovalAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        func approval(_ callId: String) -> ToolApprovalItem { functionApproval(agent, name: "tool1", callId: callId, arguments: "{}") }

        if stickyApproved {
            try state.approve(approval("sticky"), alwaysApprove: true)
            try state.reject(approval("exception"), rejectionMessage: "denied exactly")
        } else {
            try state.reject(approval("sticky"), alwaysReject: true)
            try state.approve(approval("exception"))
        }

        var serialized = try state.toJSON()
        moveFunctionToolApprovalsToLegacy(&serialized)
        serialized["$schemaVersion"] = "1.15"

        let restored = try await RunState.fromJSON(agent, serialized)
        #expect(restored.context.isToolApproved(toolName: "tool1", callId: "exception") == !stickyApproved)
        #expect(restored.context.isToolApproved(toolName: "tool1", callId: "other") == stickyApproved)
    }

    @Test func testSerializesAndRestoresRejectionMessages() async throws {
        let agent = Agent(name: "ApprovalMessageAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        try state.reject(functionApproval(agent, name: "tool2", callId: "cid2"), alwaysReject: true, rejectionMessage: "Denied by reviewer")

        let newState = try await RunState.fromString(agent, try state.toString())

        #expect(newState.context.getRejectionMessage(toolName: "tool2", callId: "cid2") == "Denied by reviewer")
        #expect(newState.context.getRejectionMessage(toolName: "tool2", callId: "cid3") == "Denied by reviewer")
    }

    @Test func testFromJSONAcceptsPreviousSchemaVersionWithoutRejectionMessages() async throws {
        let agent = Agent(name: "ApprovalLegacyAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        try state.reject(functionApproval(agent, name: "tool2", callId: "cid2"), rejectionMessage: "Denied by reviewer")

        var jsonData = try state.toJSON()
        moveFunctionToolApprovalsToLegacy(&jsonData)
        jsonData["$schemaVersion"] = "1.5"
        var context = try #require(jsonData["context"]?.objectValue)
        var approvals = try #require(context["approvals"]?.objectValue)
        var record = try #require(approvals["tool2"]?.objectValue)
        record.removeValue(forKey: "rejection_messages")
        approvals["tool2"] = .object(record)
        context["approvals"] = .object(approvals)
        jsonData["context"] = .object(context)

        let restored = try await RunState.fromJSON(agent, jsonData)

        #expect(restored.context.isToolApproved(toolName: "tool2", callId: "cid2") == false)
        #expect(restored.context.getRejectionMessage(toolName: "tool2", callId: "cid2") == nil)
    }

    @Test func testFromJSONWithContextOverrideUsesSerializedRejectionMessages() async throws {
        let agent = Agent(name: "ApprovalOverrideAgent")
        let state = makeState(agent, context: RunContextWrapper(context: ["source": "saved"] as JSONObject), originalInput: "test")
        let approvalItem = functionApproval(agent, name: "tool2", callId: "cid2")
        try state.reject(approvalItem, alwaysReject: true, rejectionMessage: "Denied by reviewer")

        let overrideContext = RunContextWrapper(context: ["source": "override"] as JSONObject)
        try overrideContext.rejectTool(approvalItem, alwaysReject: true, rejectionMessage: "override denial")

        let restored = try await RunState.fromJSON(agent, try state.toJSON(), contextOverride: overrideContext)

        #expect(restored.context === overrideContext)
        #expect(restored.context.context(as: JSONObject.self) == ["source": "override"])
        #expect(restored.context.getRejectionMessage(toolName: "tool2", callId: "cid2") == "Denied by reviewer")
        #expect(restored.context.getRejectionMessage(toolName: "tool2", callId: "cid3") == "Denied by reviewer")
    }
}

@Suite struct RunStateSerializationRoundTripTests {
    private func emptyState(_ agent: Agent, originalInput: RunInput = "test", maxTurns: Int? = 5) -> RunState {
        makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: originalInput, maxTurns: maxTurns)
    }

    @Test func testPreservesUsageData() async throws {
        let context = RunContextWrapper(context: JSONObject())
        context.usage.requests = 5
        context.usage.inputTokens = 100
        context.usage.outputTokens = 50
        context.usage.totalTokens = 150
        context.usage.inputTokensDetails = InputTokensDetails(cachedTokens: 3, cacheWriteTokens: 7)
        let agent = Agent(name: "UsageAgent")
        let state = makeState(agent, context: context, originalInput: "test", maxTurns: 10)

        let stringData = try state.toString()
        let serialized = try JSONObject.parse(stringData)
        let newState = try await RunState.fromString(agent, stringData)

        #expect(serialized["$schemaVersion"] == .string(RunState.currentSchemaVersion))
        #expect(serialized["context"]?.objectValue?["usage"]?.objectValue?["input_tokens_details"] == [["cached_tokens": 3, "cache_write_tokens": 7]])
        let usage = newState.context.usage
        #expect(usage.requests == 5)
        #expect(usage.inputTokens == 100)
        #expect(usage.outputTokens == 50)
        #expect(usage.totalTokens == 150)
        #expect(usage.inputTokensDetails.cachedTokens == 3)
        #expect(usage.inputTokensDetails.cacheWriteTokens == 7)
    }

    @Test func testRestoresSchema111UsageWithoutCacheWriteTokens() async throws {
        let agent = Agent(name: "UsageAgent")
        var stateJSON = try emptyState(agent, maxTurns: 10).toJSON()
        stateJSON["$schemaVersion"] = "1.11"
        var context = try #require(stateJSON["context"]?.objectValue)
        var usage = try #require(context["usage"]?.objectValue)
        usage["input_tokens_details"] = [["cached_tokens": 3]]
        context["usage"] = .object(usage)
        stateJSON["context"] = .object(context)

        let restored = try await RunState.fromJSON(agent, stateJSON)

        #expect(restored.context.usage.inputTokensDetails.cachedTokens == 3)
        #expect(restored.context.usage.inputTokensDetails.cacheWriteTokens == 0)
    }

    @Test func testSerializesGeneratedItems() throws {
        let agent = Agent(name: "ItemAgent")
        let state = emptyState(agent)
        state.generatedItems.append(MessageOutputItem(agent: agent, rawItem: makeMessageOutput(text: "Hello!")))

        let jsonData = try state.toJSON()
        #expect(jsonData["generated_items"]?.arrayValue?.count == 1)
        #expect(jsonData["generated_items"]?.arrayValue?[0].objectValue?["type"] == "message_output_item")
    }

    @Test func testSerializesCurrentStepInterruption() async throws {
        let agent = Agent(name: "InterruptAgent")
        let approvalItem = functionApproval(agent, name: "myTool", callId: "cid_int", arguments: #"{"arg": "value"}"#)
        let state = makeStateWithInterruptions(agent, [approvalItem], originalInput: "test")

        let jsonData = try state.toJSON()
        let currentStep = try #require(jsonData["current_step"]?.objectValue)
        #expect(currentStep["type"] == "next_step_interruption")
        #expect(currentStep["data"]?.objectValue?["interruptions"]?.arrayValue?.count == 1)

        let newState = try await RunState.fromJSON(agent, jsonData)
        guard case .interruption(let interruptions, _)? = newState.currentStep else {
            Issue.record("expected an interruption step")
            return
        }
        #expect(interruptions.count == 1)
        #expect(interruptions[0].name == "myTool")
    }

    @Test func testDeserializesVariousItemTypes() async throws {
        let agent = Agent(name: "ItemAgent")
        let state = emptyState(agent)
        state.generatedItems.append(MessageOutputItem(agent: agent, rawItem: makeMessageOutput(text: "Hello")))
        let toolCall = ResponseFunctionToolCall(callId: "call_1", name: "my_tool", arguments: #"{"arg": "val"}"#, status: "completed").json
        state.generatedItems.append(ToolCallItem(agent: agent, rawItem: toolCall, description: "My tool description", title: "My tool title"))
        let toolCallNoDesc = ResponseFunctionToolCall(callId: "call_2", name: "other_tool", arguments: "{}", status: "completed").json
        state.generatedItems.append(ToolCallItem(agent: agent, rawItem: toolCallNoDesc))
        state.generatedItems.append(
            ToolCallOutputItem(agent: agent, rawItem: ["type": "function_call_output", "call_id": "call_1", "output": "result"], output: "result")
        )

        let newState = try await RunState.fromJSON(agent, try state.toJSON())

        #expect(newState.generatedItems.count == 4)
        #expect(newState.generatedItems[0] is MessageOutputItem)
        let described = try #require(newState.generatedItems[1] as? ToolCallItem)
        let undescribed = try #require(newState.generatedItems[2] as? ToolCallItem)
        #expect(newState.generatedItems[3] is ToolCallOutputItem)
        #expect(described.description == "My tool description")
        #expect(described.title == "My tool title")
        #expect(undescribed.description == nil)
        #expect(undescribed.title == nil)
    }

    @Test func testDeserializesCustomToolCallOutputItems() async throws {
        let agent = Agent(name: "ItemAgent")
        let state = emptyState(agent)
        let customToolOutput: JSONObject = ["type": "custom_tool_call_output", "call_id": "call_custom_1", "output": "custom result"]
        state.generatedItems.append(ToolCallOutputItem(agent: agent, rawItem: customToolOutput, output: "custom result"))

        let newState = try await RunState.fromJSON(agent, try state.toJSON())

        #expect(newState.generatedItems.count == 1)
        let restored = try #require(newState.generatedItems[0] as? ToolCallOutputItem)
        #expect(restored.rawItem == customToolOutput)
        #expect(restored.output as? JSONValue == "custom result")
    }

    @Test func testDeserializesComputerCallOutputAcknowledgedSafetyChecks() async throws {
        let agent = Agent(name: "ItemAgent")
        let state = emptyState(agent)
        let expectedChecks: JSONValue = [["id": "sc_1", "code": "malicious_instructions", "message": "confirm"]]
        let computerToolOutput: JSONObject = [
            "type": "computer_call_output",
            "call_id": "call_computer_1",
            "output": ["type": "computer_screenshot", "image_url": "img"],
            "acknowledged_safety_checks": expectedChecks,
        ]
        state.generatedItems.append(ToolCallOutputItem(agent: agent, rawItem: computerToolOutput, output: "done"))

        let newState = try await RunState.fromJSON(agent, try state.toJSON())

        let restored = try #require(newState.generatedItems.first as? ToolCallOutputItem)
        #expect(restored.rawItem["acknowledged_safety_checks"] == expectedChecks)
        #expect(restored.rawItem["acknowledged_safety_checks"] == expectedChecks)

        // A restored state must serialize again for repeated pause/resume cycles.
        let roundTripped = try await RunState.fromString(agent, try newState.toString())
        #expect(roundTripped.generatedItems[0].rawItem["acknowledged_safety_checks"] == expectedChecks)
        _ = try roundTripped.toJSON()
    }

    @Test func testSerializesOutputContainersOfModels() async throws {
        // Swift: upstream's pydantic models and dataclasses map to `Encodable` values.
        struct Weather: Encodable, Sendable { var city: String; var temperature: Int }
        struct Reading: Encodable, Sendable { var value: Int; var label: String }
        let agent = Agent(name: "ItemAgent")
        let cases: [(any Sendable, JSONValue)] = [
            ([Weather(city: "sf", temperature: 18)] as [any Sendable], [["city": "sf", "temperature": 18]]),
            (["today": Weather(city: "sf", temperature: 18)] as [String: any Sendable], ["today": ["city": "sf", "temperature": 18]]),
            ([Reading(value: 1, label: "ok")] as [any Sendable], [["value": 1, "label": "ok"]]),
        ]
        for (output, expected) in cases {
            let state = emptyState(agent)
            state.generatedItems.append(
                ToolCallOutputItem(agent: agent, rawItem: ["type": "function_call_output", "call_id": "c1", "output": "x"], output: output)
            )

            let jsonData = try state.toJSON()
            #expect(jsonData["generated_items"]?.arrayValue?[0].objectValue?["output"] == expected)

            let newState = try await RunState.fromJSON(agent, jsonData)
            let restored = try #require(newState.generatedItems[0] as? ToolCallOutputItem)
            #expect(restored.output as? JSONValue == expected)
        }
    }

    @Test func testDeserializesToolCallOutputCustomData() async throws {
        let agent = Agent(name: "ItemAgent")
        let state = emptyState(agent)
        let customData: JSONObject = ["ui": ["kind": "chart"], "ids": ["a", "b"]]
        state.generatedItems.append(
            ToolCallOutputItem(
                agent: agent,
                rawItem: ["type": "function_call_output", "call_id": "call_custom_data", "output": "result"],
                output: "result",
                customData: customData
            )
        )

        let jsonData = try state.toJSON()
        let serializedItem = try #require(jsonData["generated_items"]?.arrayValue?[0].objectValue)
        #expect(serializedItem["custom_data"] == .object(customData))
        #expect(serializedItem["raw_item"]?.objectValue?["custom_data"] == nil)

        let newState = try await RunState.fromJSON(agent, jsonData)
        let restored = try #require(newState.generatedItems[0] as? ToolCallOutputItem)
        #expect(restored.customData == customData)
    }

    @Test func testPydanticToolOutputPreservesDefaultFields() async throws {
        // Swift: an `Encodable` struct stands in for the pydantic model. Synthesized `Encodable`
        // omits `nil` optionals, so `humidity` is absent instead of `null`.
        struct WeatherReport: Encodable, Sendable {
            var temperature: Int
            var unit = "celsius"
            var humidity: Int?
        }
        let agent = Agent(name: "ItemAgent")
        let state = emptyState(agent)
        state.generatedItems.append(
            ToolCallOutputItem(
                agent: agent,
                rawItem: ["type": "function_call_output", "call_id": "call_weather", "output": #"{"temperature":20,"unit":"celsius","humidity":null}"#],
                output: WeatherReport(temperature: 20)
            )
        )

        let jsonData = try state.toJSON()
        let expected: JSONValue = ["temperature": 20, "unit": "celsius"]
        #expect(jsonData["generated_items"]?.arrayValue?[0].objectValue?["output"] == expected)

        let newState = try await RunState.fromJSON(agent, jsonData)
        let restored = try #require(newState.generatedItems[0] as? ToolCallOutputItem)
        #expect(restored.output as? JSONValue == expected)
    }

    // Swift: test_non_utf8_bytes_tool_output_keeps_dict_shape is not ported; it depends on
    // Python `bytes` reprs and pydantic `mode="json"` failures.

    @Test func testSerializesOriginalInputWithFunctionCallOutput() throws {
        let agent = Agent(name: "TestAgent")
        let originalInput: RunInput = .items([
            ["type": "function_call", "call_id": "call_123", "name": "test_tool", "arguments": #"{"arg": "value"}"#],
            ["type": "function_call_output", "call_id": "call_123", "output": "result"],
        ])
        let state = emptyState(agent, originalInput: originalInput)

        let serialized = try #require(try state.toJSON()["original_input"]?.arrayValue)
        #expect(serialized.count == 2)
        let first = try #require(serialized[0].objectValue)
        #expect(first["type"] == "function_call")
        #expect(first["call_id"] == "call_123")
        #expect(first["name"] == "test_tool")
        let second = try #require(serialized[1].objectValue)
        #expect(second["type"] == "function_call_output")
        #expect(second["call_id"] == "call_123")
        #expect(second["name"] == nil)
        #expect(second["status"] == nil)
        #expect(second["output"] == "result")
    }

    @Test(arguments: [
        (["role": "assistant", "content": "This is a summary message"] as JSONObject, "completed", "This is a summary message"),
        (["role": "assistant", "status": "in_progress", "content": "In progress message"] as JSONObject, "in_progress", "In progress message"),
        (
            ["role": "assistant", "status": "completed", "content": [["type": "output_text", "text": "Already array format"]]] as JSONObject,
            "completed",
            "Already array format"
        ),
    ])
    func testSerializesAssistantMessages(message: JSONObject, expectedStatus: String, expectedText: String) throws {
        let agent = Agent(name: "TestAgent")
        let state = emptyState(agent, originalInput: .items([message]))

        let serialized = try #require(try state.toJSON()["original_input"]?.arrayValue)
        #expect(serialized.count == 1)
        let assistantMessage = try #require(serialized[0].objectValue)
        #expect(assistantMessage["role"] == "assistant")
        #expect(assistantMessage["status"] == .string(expectedStatus))
        let content = try #require(assistantMessage["content"]?.arrayValue?.first?.objectValue)
        #expect(content["type"] == "output_text")
        #expect(content["text"] == .string(expectedText))
    }

    @Test func testFromStringNormalizesOriginalInputDictItems() async throws {
        // Swift: run input items are `JSONObject` values, so the upstream non-mapping
        // `"simple_string"` entry cannot be represented and is dropped on restore.
        let agent = Agent(name: "TestAgent")
        let stateJSON: JSONObject = [
            "$schemaVersion": .string(RunState.currentSchemaVersion),
            "current_turn": 0,
            "current_agent": ["name": "TestAgent"],
            "original_input": [
                ["type": "function_call_output", "call_id": "call123", "name": "test_tool", "status": "completed", "output": "result"],
                "simple_string",
            ],
            "model_responses": [],
            "context": [
                "usage": [
                    "requests": 0, "input_tokens": 0, "input_tokens_details": [], "output_tokens": 0,
                    "output_tokens_details": [], "total_tokens": 0, "request_usage_entries": [],
                ],
                "approvals": [:],
                "context": [:],
            ],
            "tool_use_tracker": [:],
            "max_turns": 10,
            "noActiveAgentRun": true,
            "input_guardrail_results": [],
            "output_guardrail_results": [],
            "generated_items": [],
            "current_step": nil,
            "last_model_response": nil,
            "last_processed_response": nil,
            "current_turn_persisted_item_count": 0,
            "trace": nil,
        ]

        let state = try await RunState.fromJSON(agent, stateJSON)

        let items = state.originalInput.inputItems
        #expect(items.count == 1)
        #expect(items[0]["type"] == "function_call_output")
        #expect(items[0]["name"] == "test_tool")
        #expect(items[0]["status"] == "completed")
        #expect(items[0]["call_id"] == "call123")
    }

    // Swift: test_serializes_original_input_with_non_dict_items is not ported; run input items are
    // `JSONObject` values, so a bare string item cannot be constructed.

    @Test func testFromJSONPreservesFunctionOutputOriginalInput() async throws {
        let agent = Agent(name: "TestAgent")
        var stateJSON = try emptyState(agent, originalInput: "placeholder").toJSON()
        stateJSON["original_input"] = [
            ["type": "function_call", "call_id": "call_abc", "name": "demo_tool", "arguments": #"{"x":1}"#],
            ["type": "function_call_output", "call_id": "call_abc", "name": "demo_tool", "status": "completed", "output": "demo-output"],
        ]

        let restored = try await RunState.fromJSON(agent, stateJSON)

        let items = restored.originalInput.inputItems
        #expect(items.count == 2)
        #expect(items[0]["type"] == "function_call")
        #expect(items[1]["type"] == "function_call_output")
        #expect(items[1]["call_id"] == "call_abc")
        #expect(items[1]["output"] == "demo-output")
        #expect(items[1]["name"] == "demo_tool")
        #expect(items[1]["status"] == "completed")
    }

    @Test func testSerializeToolCallOutputLooksUpName() {
        let agent = Agent(name: "TestAgent")
        let state = emptyState(agent, originalInput: .items([]))
        let toolCall = ResponseFunctionToolCall(id: "fc_lookup", callId: "call_lookup", name: "lookup_tool", arguments: "{}", status: "completed")
        state.generatedItems.append(ToolCallItem(agent: agent, rawItem: toolCall.json))
        let outputItem = ToolCallOutputItem(agent: agent, rawItem: ["type": "function_call_output", "call_id": "call_lookup", "output": "ok"], output: "ok")

        let rawItem = state.serializeRunItem(outputItem, nil)["raw_item"]?.objectValue
        #expect(rawItem?["type"] == "function_call_output")
        #expect(rawItem?["call_id"] == "call_lookup")
        #expect(rawItem?["name"] == nil)
        #expect(rawItem?["status"] == nil)
    }

    enum LookupSource: String, CaseIterable, Sendable {
        case originalInput = "original_input"
        case generatedObject = "generated_object"
        case camelCaseCallId = "camel_case_call_id"
        case wrongTypeItems = "wrong_type_items"
        case emptyName = "empty_name"
        case generatedDict = "generated_dict"
        case lastProcessedResponse = "last_processed_response"
    }

    // Swift: the upstream `non_dict_items` case is not ported; bare string input items cannot be
    // constructed.
    @Test(arguments: LookupSource.allCases)
    func testLookupFunctionNameSources(source: LookupSource) {
        let agent = Agent(name: "TestAgent")
        let state = emptyState(agent, originalInput: .items([]))
        func appendInput(_ items: [JSONObject]) { state.originalInput = .items(state.originalInput.inputItems + items) }
        let callId: String
        let expectedName: String
        switch source {
        case .originalInput:
            appendInput([["type": "function_call", "call_id": "call_from_input", "name": "input_tool", "arguments": "{}"]])
            (callId, expectedName) = ("call_from_input", "input_tool")
        case .generatedObject:
            state.generatedItems.append(ToolCallItem(agent: agent, rawItem: makeToolCall("call_obj", name: "obj_tool").json))
            (callId, expectedName) = ("call_obj", "obj_tool")
        case .camelCaseCallId:
            appendInput([["type": "function_call", "call_id": "call_camel", "name": "camel_tool", "arguments": "{}"]])
            (callId, expectedName) = ("call_camel", "camel_tool")
        case .wrongTypeItems:
            appendInput([
                ["type": "message", "role": "user", "content": "Hello"],
                ["type": "function_call", "call_id": "call_valid", "name": "valid_tool", "arguments": "{}"],
            ])
            (callId, expectedName) = ("call_valid", "valid_tool")
        case .emptyName:
            appendInput([["type": "function_call", "call_id": "call_empty", "name": "", "arguments": "{}"]])
            (callId, expectedName) = ("call_empty", "")
        case .generatedDict:
            state.generatedItems.append(ToolCallItem(
                agent: agent,
                rawItem: ["type": "function_call", "call_id": "call_dict", "name": "dict_tool", "arguments": "{}", "status": "completed"]
            ))
            (callId, expectedName) = ("call_dict", "dict_tool")
        case .lastProcessedResponse:
            setLastProcessedResponse(state, agent, [ToolCallItem(agent: agent, rawItem: makeToolCall("call_last", name: "last_tool").json)])
            (callId, expectedName) = ("call_last", "last_tool")
        }

        #expect(state.lookupFunctionName(callId) == expectedName)
    }

    @Test func testDeserializationHandlesUnknownAgentGracefully() async throws {
        let agent = Agent(name: "KnownAgent")
        let state = emptyState(agent)
        state.generatedItems.append(MessageOutputItem(agent: agent, rawItem: makeMessageOutput(text: "Test")))

        var jsonData = try state.toJSON()
        var items = try #require(jsonData["generated_items"]?.arrayValue)
        var first = try #require(items[0].objectValue)
        first["agent"] = ["name": "UnknownAgent"]
        items[0] = .object(first)
        jsonData["generated_items"] = .array(items)

        let newState = try await RunState.fromJSON(agent, jsonData)
        #expect(newState.generatedItems.isEmpty)
    }

    @Test func testDeserializationHandlesMalformedItemsGracefully() async throws {
        let agent = Agent(name: "TestAgent")
        var jsonData = try emptyState(agent).toJSON()
        jsonData["generated_items"] = [
            ["type": "message_output_item", "agent": ["name": "TestAgent"], "raw_item": ["type": "message"]],
        ]

        let newState = try await RunState.fromJSON(agent, jsonData)
        #expect(newState.generatedItems.isEmpty)
    }
}

@Suite struct RunStateRunContextApprovalsTests {
    @Test func testApprovalTakesPrecedenceOverRejectionWhenBothTrue() {
        let context = RunContextWrapper(context: JSONObject())
        var record = ApprovalRecord()
        record.approved = .all(true)
        record.rejected = .all(true)
        context.approvals["test_tool"] = record

        #expect(context.isToolApproved(toolName: "test_tool", callId: "call_id") == true)
    }

    @Test func testIndividualApprovalTakesPrecedenceOverIndividualRejection() {
        let context = RunContextWrapper(context: JSONObject())
        var record = ApprovalRecord()
        record.approved = .callIds(["call_123"])
        record.rejected = .callIds(["call_123"])
        context.approvals["test_tool"] = record

        #expect(context.isToolApproved(toolName: "test_tool", callId: "call_123") == true)
    }

    @Test func testReturnsNoneWhenNoApprovalOrRejection() {
        let context = RunContextWrapper(context: JSONObject())
        context.approvals["test_tool"] = ApprovalRecord()

        #expect(context.isToolApproved(toolName: "test_tool", callId: "call_456") == nil)
    }
}

@Suite struct RunStateEdgeCasesTests {
    @Test func testToJSONRaisesWhenNoCurrentAgent() {
        let state = makeState(Agent(name: "TestAgent"), context: RunContextWrapper(context: JSONObject()), originalInput: "test", maxTurns: 5)
        state.currentAgent = nil

        expectError(UserError.self, containing: "Cannot serialize RunState: No current agent") {
            _ = try state.toJSON()
        }
    }

    // Swift: test_to_json_raises_when_no_context is not ported; `RunState.context` is non-optional.
}
