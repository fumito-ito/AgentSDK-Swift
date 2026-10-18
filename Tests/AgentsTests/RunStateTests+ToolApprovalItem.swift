import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_state.py (classes TestToolApprovalItem and
// TestDeserializeItemsEdgeCases, and the module-level resume, hosted MCP, and guardrail tests).
//
// Swift: not ported, and recorded in porting/manifest.json:
// - Tool invocation ledger and hosted MCP query records (`_tool_invocation.py`):
//   `test_approve_tool_rejects_uncanonical_hosted_call_dict`,
//   `test_hosted_mcp_approval_round_trip_uses_typed_identity_records`,
//   `test_incomplete_hosted_mcp_query_cannot_create_approval_authority`,
//   `test_hosted_mcp_rejection_query_round_trip_does_not_cross_servers`, and
//   `test_schema_1_13_hosted_mcp_orphaned_call_decisions_require_reapproval`.
// - Nested agent-tool run state (`agent_tool_state.py`):
//   `test_resume_nested_agent_as_tool_with_context_override` and
//   `test_resume_recursively_nested_agent_as_tool_decision`.
// - Bare string input items, which `RunInput.items` cannot hold:
//   `test_deserialize_items_handles_non_dict_items_in_original_input` and
//   `test_from_string_handles_non_dict_items_in_original_input`.

private func functionCall(_ name: String, callId: String, arguments: String = "{}") -> JSONObject {
    ResponseFunctionToolCall(callId: callId, name: name, arguments: arguments, status: "completed").json
}

private func deserialize(_ items: [JSONObject], _ agentMap: [String: Agent]) throws -> [RunItem] {
    try deserializeItems(items.map(JSONValue.object), agents: RunStateAgentResolver(agentMap: agentMap))
}

private let helloMessage: JSONValue = [
    "type": "message", "id": "msg1", "role": "assistant",
    "content": [["type": "output_text", "text": "Hello", "annotations": []]], "status": "completed",
]

@Suite struct RunStateToolApprovalItemTests {
    @Test func testToolApprovalItemWithExplicitToolName() {
        let item = ToolApprovalItem(agent: Agent(name: "TestAgent"), rawItem: functionCall("raw_tool_name", callId: "call123"), toolName: "explicit_tool_name")
        #expect(item.toolName == "explicit_tool_name")
        #expect(item.name == "explicit_tool_name")
    }

    @Test func testToolApprovalItemFallsBackToRawItemName() {
        let item = ToolApprovalItem(agent: Agent(name: "TestAgent"), rawItem: functionCall("raw_tool_name", callId: "call123"))
        #expect(item.toolName == "raw_tool_name")
        #expect(item.name == "raw_tool_name")
    }

    @Test func testToolApprovalItemWithDictRawItem() {
        let rawItem: JSONObject = ["type": "function_call", "name": "dict_tool_name", "call_id": "call456", "status": "completed", "arguments": "{}"]
        let item = ToolApprovalItem(agent: Agent(name: "TestAgent"), rawItem: rawItem, toolName: "explicit_name")
        #expect(item.toolName == "explicit_name")
        #expect(item.name == "explicit_name")
    }

    @Test func testApproveToolWithExplicitToolName() throws {
        let context = RunContextWrapper(context: JSONObject())
        let item = ToolApprovalItem(agent: Agent(name: "TestAgent"), rawItem: functionCall("raw_name", callId: "call123"), toolName: "explicit_name")
        try context.approveTool(item)
        #expect(context.isToolApproved(toolName: "explicit_name", callId: "call123") == true)
    }

    @Test func testRejectToolWithExplicitToolName() throws {
        let context = RunContextWrapper(context: JSONObject())
        let item = ToolApprovalItem(agent: Agent(name: "TestAgent"), rawItem: functionCall("raw_name", callId: "call789"), toolName: "explicit_name")
        try context.rejectTool(item)
        #expect(context.isToolApproved(toolName: "explicit_name", callId: "call789") == false)
    }

    @Test func testSerializeToolApprovalItemWithToolName() throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        state.generatedItems.append(ToolApprovalItem(agent: agent, rawItem: functionCall("raw_name", callId: "call123"), toolName: "explicit_name"))

        let generatedItems = try #require(try state.toJSON()["generated_items"]?.arrayValue)
        #expect(generatedItems.count == 1)
        #expect(generatedItems[0].objectValue?["type"] == "tool_approval_item")
        #expect(generatedItems[0].objectValue?["tool_name"] == "explicit_name")
    }

    @Test func testDeserializeToolApprovalItemWithToolName() throws {
        let result = try deserialize([
            [
                "type": "tool_approval_item", "agent": ["name": "TestAgent"], "tool_name": "explicit_tool_name",
                "raw_item": .object(functionCall("raw_tool_name", callId: "call123")),
            ],
        ], ["TestAgent": Agent(name: "TestAgent")])

        #expect(result.count == 1)
        let item = try #require(result.first as? ToolApprovalItem)
        #expect(item.toolName == "explicit_tool_name")
        #expect(item.name == "explicit_tool_name")
    }

    @Test func testRoundTripSerializationWithToolName() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        state.generatedItems.append(ToolApprovalItem(agent: agent, rawItem: functionCall("raw_name", callId: "call123"), toolName: "explicit_name"))

        let newState = try await RunState.fromJSON(agent, try state.toJSON())

        #expect(newState.generatedItems.count == 1)
        let restored = try #require(newState.generatedItems.first as? ToolApprovalItem)
        #expect(restored.toolName == "explicit_name")
        #expect(restored.name == "explicit_name")
    }

    @Test func testRoundTripSerializationPreservesAllowBareNameAlias() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        let rawItem: JSONObject = [
            "type": "function_call", "name": "get_weather", "call_id": "call123", "status": "completed", "arguments": "{}", "namespace": "get_weather",
        ]
        state.generatedItems.append(ToolApprovalItem(agent: agent, rawItem: rawItem, toolName: "get_weather", toolNamespace: "get_weather", allowBareNameAlias: true))

        let jsonData = try state.toJSON()
        #expect(jsonData["generated_items"]?.arrayValue?[0].objectValue?["allow_bare_name_alias"] == true)

        let newState = try await RunState.fromJSON(agent, jsonData)
        let restored = try #require(newState.generatedItems.first as? ToolApprovalItem)
        #expect(restored.allowBareNameAlias)
    }

    @Test func testToolApprovalItemArgumentsProperty() {
        let agent = Agent(name: "TestAgent")
        #expect(ToolApprovalItem(agent: agent, rawItem: functionCall("tool1", callId: "call1", arguments: #"{"city": "Oakland"}"#)).arguments == #"{"city": "Oakland"}"#)
        let withArguments: JSONObject = ["type": "function_call", "name": "tool2", "call_id": "call2", "status": "completed", "arguments": #"{"key": "value"}"#]
        #expect(ToolApprovalItem(agent: agent, rawItem: withArguments).arguments == #"{"key": "value"}"#)
        let withoutArguments: JSONObject = ["type": "function_call", "name": "tool3", "call_id": "call3", "status": "completed"]
        #expect(ToolApprovalItem(agent: agent, rawItem: withoutArguments).arguments == nil)
        #expect(ToolApprovalItem(agent: agent, rawItem: ["type": "unknown", "name": "tool4"]).arguments == nil)
    }

    @Test func testToolApprovalItemTracksNamespace() {
        let item = ToolApprovalItem(agent: Agent(name: "TestAgent"), rawItem: makeToolCall("call-ns-1", name: "lookup_account", namespace: "crm").json)
        #expect(item.toolName == "lookup_account")
        #expect(item.toolNamespace == "crm")
        #expect(item.qualifiedName == "crm.lookup_account")
    }

    @Test func testToolApprovalItemCollapsesSyntheticDeferredNamespaceInQualifiedName() {
        let item = ToolApprovalItem(agent: Agent(name: "TestAgent"), rawItem: makeToolCall("call-weather-1", name: "get_weather", namespace: "get_weather").json)
        #expect(item.toolName == "get_weather")
        #expect(item.toolNamespace == "get_weather")
        #expect(item.qualifiedName == "get_weather")
    }

    @Test func testRoundTripSerializationWithToolNamespace() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        state.generatedItems.append(ToolApprovalItem(agent: agent, rawItem: makeToolCall("call123", name: "lookup_account", namespace: "billing").json))

        let newState = try await RunState.fromJSON(agent, try state.toJSON())

        let restored = try #require(newState.generatedItems.first as? ToolApprovalItem)
        #expect(restored.toolName == "lookup_account")
        #expect(restored.toolNamespace == "billing")
        #expect(restored.qualifiedName == "billing.lookup_account")
    }

    @Test func testRoundTripSerializationPreservesToolLookupKey() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        state.generatedItems.append(ToolApprovalItem(
            agent: agent,
            rawItem: makeToolCall("call-weather", name: "get_weather", namespace: "get_weather").json,
            toolLookupKey: .deferredTopLevel("get_weather")
        ))

        let newState = try await RunState.fromJSON(agent, try state.toJSON())

        let restored = try #require(newState.generatedItems.first as? ToolApprovalItem)
        #expect(restored.toolLookupKey == .deferredTopLevel("get_weather"))
    }

    @Test func testRoundTripDeserializesStatuslessMessageOutputItems() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        let message: JSONObject = [
            "id": "msg_constructed", "type": "message", "role": "assistant",
            "content": [["type": "output_text", "text": "hello", "annotations": []]],
        ]
        state.generatedItems.append(MessageOutputItem(agent: agent, rawItem: message))

        let restored = try await RunState.fromJSON(agent, try state.toJSON())

        let restoredMessage = try #require(restored.generatedItems.first as? MessageOutputItem).rawItem
        #expect(restoredMessage == message)
        #expect(restoredMessage["status"] == nil)
    }

    @Test func testRoundTripDeserializesStatuslessModelResponseMessages() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: "test")
        let message: JSONObject = [
            "id": "msg_response", "type": "message", "role": "assistant",
            "content": [["type": "output_text", "text": "world", "annotations": []]],
        ]
        state.modelResponses.append(ModelResponse(output: [message], usage: Usage(), responseId: nil))

        let restored = try await RunState.fromJSON(agent, try state.toJSON())

        #expect(restored.modelResponses[0].output[0] == message)
    }

    @Test func testDeserializeItemsRestoresToolSearchItems() throws {
        let items = try deserialize([
            [
                "type": "tool_search_call_item", "agent": ["name": "TestAgent"],
                "raw_item": ["id": "tsc_state", "type": "tool_search_call", "arguments": ["paths": ["crm"], "query": "profile"], "execution": "server", "status": "completed"],
            ],
            [
                "type": "tool_search_output_item", "agent": ["name": "TestAgent"],
                "raw_item": [
                    "id": "tso_state", "type": "tool_search_output", "execution": "server", "status": "completed",
                    "tools": [
                        [
                            "type": "function", "name": "get_customer_profile", "description": "Fetch a CRM customer profile.",
                            "parameters": ["type": "object", "properties": ["customer_id": ["type": "string"]], "required": ["customer_id"]],
                            "defer_loading": true,
                        ],
                    ],
                ],
            ],
        ], ["TestAgent": Agent(name: "TestAgent")])

        #expect(items[0] is ToolSearchCallItem)
        #expect(items[1] is ToolSearchOutputItem)
        #expect(items[0].rawItem.type == "tool_search_call")
        #expect(items[1].rawItem.type == "tool_search_output")
    }

    @Test func testDeserializeItemsHandlesMissingAgentName() throws {
        let result = try deserialize([["type": "message_output_item", "raw_item": helloMessage]], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.isEmpty)
    }

    @Test func testDeserializeItemsHandlesStringAgentName() throws {
        let result = try deserialize([["type": "message_output_item", "agent": "TestAgent", "raw_item": helloMessage]], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.count == 1)
        #expect(result.first?.type == "message_output_item")
    }

    @Test func testDeserializeItemsHandlesAgentField() throws {
        let result = try deserialize([["type": "message_output_item", "agent": ["name": "TestAgent"], "raw_item": helloMessage]], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.count == 1)
        #expect(result.first?.type == "message_output_item")
    }

    @Test func testDeserializeItemsHandlesHandoffOutputSourceAgentString() throws {
        let map = ["Agent1": Agent(name: "Agent1"), "Agent2": Agent(name: "Agent2")]
        let result = try deserialize([
            [
                "type": "handoff_output_item", "source_agent": "Agent1", "target_agent": ["name": "Agent2"],
                "raw_item": ["role": "assistant", "content": "Handoff message"],
            ],
        ], map)
        #expect(result.count <= 1)
    }

    @Test func testDeserializeItemsHandlesHandoffOutputTargetAgentString() throws {
        let map = ["Agent1": Agent(name: "Agent1"), "Agent2": Agent(name: "Agent2")]
        let result = try deserialize([
            [
                "type": "handoff_output_item", "source_agent": ["name": "Agent1"], "target_agent": "Agent2",
                "raw_item": ["role": "assistant", "content": "Handoff message"],
            ],
        ], map)
        #expect(result.count <= 1)
    }

    @Test func testDeserializeItemsHandlesToolApprovalItemException() throws {
        let result = try deserialize([["type": "tool_approval_item", "agent": ["name": "TestAgent"], "raw_item": ["type": "invalid"]]], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.count == 1)
        #expect(result.first?.type == "tool_approval_item")
    }
}

@Suite struct RunStateDeserializeItemsEdgeCasesTests {
    private let handoffMessage: JSONValue = ["role": "assistant", "content": "Handoff message"]
    private let agentMap = ["Agent1": Agent(name: "Agent1"), "Agent2": Agent(name: "Agent2")]

    @Test func testDeserializeItemsHandlesHandoffOutputWithStringSourceAgent() throws {
        let result = try deserialize([["type": "handoff_output_item", "source_agent": "Agent1", "target_agent": ["name": "Agent2"], "raw_item": handoffMessage]], agentMap)
        #expect(result.count <= 1)
    }

    @Test func testDeserializeItemsHandlesHandoffOutputWithStringTargetAgent() throws {
        let result = try deserialize([["type": "handoff_output_item", "source_agent": ["name": "Agent1"], "target_agent": "Agent2", "raw_item": handoffMessage]], agentMap)
        #expect(result.count <= 1)
    }

    @Test func testDeserializeItemsHandlesHandoffOutputNoSourceNoTarget() throws {
        let result = try deserialize([["type": "handoff_output_item", "raw_item": handoffMessage]], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.isEmpty)
    }

    @Test func testFromJSONHandlesStringOriginalInput() async throws {
        let stateJSON: JSONObject = [
            "$schemaVersion": .string(RunState.currentSchemaVersion),
            "current_turn": 0,
            "current_agent": ["name": "TestAgent"],
            "original_input": "string_input",
            "max_turns": 5,
            "context": ["usage": ["requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0], "approvals": [:], "context": [:]],
            "generated_items": [],
            "model_responses": [],
        ]

        let state = try await RunState.fromJSON(Agent(name: "TestAgent"), stateJSON)
        #expect(state.originalInput == "string_input")
    }

    @Test func testLookupFunctionNameSearchesLastProcessedResponseNewItems() {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: .items([]), maxTurns: 5)
        let call1 = ResponseFunctionToolCall(id: "fc1", callId: "call1", name: "tool1", arguments: "{}", status: "completed").json
        let call2 = ResponseFunctionToolCall(id: "fc2", callId: "call2", name: "tool2", arguments: "{}", status: "completed").json
        state.lastProcessedResponse = makeProcessedResponse(newItems: [
            MessageOutputItem(agent: agent, rawItem: makeMessageOutput(messageId: "msg1", text: "Hello")),
            ToolCallItem(agent: agent, rawItem: call1),
            ToolCallItem(agent: agent, rawItem: call2),
        ])

        #expect(state.lookupFunctionName("call1") == "tool1")
        #expect(state.lookupFunctionName("call2") == "tool2")
        #expect(state.lookupFunctionName("missing") == "")
    }

    @Test func testFromJSONPreservesFunctionCallOutputItems() async throws {
        let stateJSON: JSONObject = [
            "$schemaVersion": .string(RunState.currentSchemaVersion),
            "current_turn": 0,
            "current_agent": ["name": "TestAgent"],
            "original_input": [["type": "function_call_output", "call_id": "call123", "name": "test_tool", "status": "completed", "output": "result"]],
            "max_turns": 5,
            "context": ["usage": ["requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0], "approvals": [:], "context": [:]],
            "generated_items": [],
            "model_responses": [],
        ]

        let state = try await RunState.fromJSON(Agent(name: "TestAgent"), stateJSON)

        let items = state.originalInput.inputItems
        #expect(items.count == 1)
        #expect(items[0]["type"] == "function_call_output")
        #expect(items[0]["name"] == "test_tool")
        #expect(items[0]["status"] == "completed")
    }

    @Test func testDeserializeItemsHandlesMissingTypeField() throws {
        let result = try deserialize([["agent": ["name": "TestAgent"], "raw_item": helloMessage]], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.isEmpty)
    }

    @Test func testDeserializeItemsHandlesDictTargetAgent() throws {
        let item: JSONObject = ["type": "handoff_output_item", "source_agent": ["name": "Agent1"], "target_agent": ["name": "Agent2"], "raw_item": handoffMessage]
        let result = try deserialize([item], agentMap)
        #expect(result.count == 1)
        #expect(result.first?.type == "handoff_output_item")
    }

    @Test func testDeserializeItemsHandlesHandoffOutputDictTargetAgent() throws {
        let result = try deserialize([["type": "handoff_output_item", "target_agent": ["name": "Agent2"], "raw_item": handoffMessage]], agentMap)
        #expect(result.count <= 1)
    }

    @Test func testDeserializeItemsHandlesHandoffOutputStringTargetAgentFallback() throws {
        let result = try deserialize([["type": "handoff_output_item", "target_agent": "Agent2", "raw_item": handoffMessage]], agentMap)
        #expect(result.count <= 1)
    }
}

@Suite struct RunStateResumeApprovalTests {
    private struct TextArgs: Decodable, Sendable { var text: String }

    private func needsOk(calls: SharedBox<[String]>, seenContexts: SharedBox<[JSONObject]>? = nil, seenToolInputs: SharedBox<[JSONValue?]>? = nil) throws -> FunctionTool {
        try FunctionTool.create(name: "needs_ok", argumentsType: TextArgs.self, needsApproval: true) { context, args in
            seenContexts?.mutate { $0.append(context.context(as: JSONObject.self) ?? [:]) }
            seenToolInputs?.mutate { $0.append(context.toolInput as? JSONValue) }
            calls.mutate { $0.append(args.text) }
            return args.text
        }
    }

    @Test func testResumePendingFunctionApprovalReinterrupts() async throws {
        let calls = SharedBox<[String]>([])
        let (model, agent) = try makeModelAndAgent(tools: [try needsOk(calls: calls)], name: "agent")

        let (first, resumed) = try await runAndResumeWithMutation(agent, model, [
            [getFunctionToolCall("needs_ok", arguments: #"{"text": "one"}"#, callId: "1")],
            [getTextMessage("done")],
        ], userInput: "hi")

        #expect(first.finalOutput == nil)
        #expect(resumed.finalOutput == nil)
        #expect(!resumed.interruptions.isEmpty)
        #expect(calls.value.isEmpty)
    }

    @Test func testResumeRejectedFunctionApprovalEmitsOutput() async throws {
        let calls = SharedBox<[String]>([])
        let (model, agent) = try makeModelAndAgent(tools: [try needsOk(calls: calls)], name: "agent")

        let (first, resumed) = try await runAndResumeWithMutation(agent, model, [
            [getFunctionToolCall("needs_ok", arguments: #"{"text": "one"}"#, callId: "1")],
            [getFinalOutputMessage("done")],
        ], userInput: "hi") { state, approval in
            try state.reject(approval)
        }

        #expect(first.finalOutput == nil)
        #expect(resumed.finalOutput as? String == "done")
        #expect(resumed.newItems.contains { ($0 as? ToolCallOutputItem)?.output as? String == hitlRejectionMessage })
        #expect(calls.value.isEmpty)
    }

    @Test func testResolveResumedContextKeepsRestoredWrapperAndReplacesAppContext() throws {
        let agent = Agent(name: "unit-agent")
        let restoredWrapper = RunContextWrapper(context: ["user": "original"] as JSONObject)
        restoredWrapper.toolInput = ["scoped": true] as JSONObject
        restoredWrapper.turnInput = [["role": "user", "content": "hi"]]
        var record = ApprovalRecord()
        record.approved = .callIds(["1"])
        restoredWrapper.approvals["needs_ok"] = record
        let state = makeState(agent, context: restoredWrapper, originalInput: "hi")
        let override: JSONObject = ["user": "reviewer"]

        let resolved = resolveResumedContext(runState: state, context: override)

        #expect(resolved === restoredWrapper)
        #expect(resolved === state.context)
        #expect(resolved.context(as: JSONObject.self) == override)
        #expect(resolved.usageBox === restoredWrapper.usageBox)
        #expect(resolved.toolStateBox === restoredWrapper.toolStateBox)
        #expect(resolved.approvals["needs_ok"]?.approved == .callIds(["1"]))
        #expect(resolved.turnInput == [["role": "user", "content": "hi"]])
        #expect(resolved.toolInput as? JSONObject == ["scoped": true])

        // Passing a wrapper only donates its application value; run-owned state stays.
        let donor = RunContextWrapper(context: ["user": "from-wrapper"] as JSONObject)
        donor.toolInput = ["should": "not-win"] as JSONObject
        let resolvedAgain = resolveResumedContext(runState: state, context: donor)
        #expect(resolvedAgain === restoredWrapper)
        #expect(resolvedAgain.context(as: JSONObject.self) == ["user": "from-wrapper"])
        #expect(resolvedAgain.toolInput as? JSONObject == ["scoped": true])
    }

    private func interruptedApprovalStateWithToolInput(
        calls: SharedBox<[String]>,
        seenContexts: SharedBox<[JSONObject]>,
        seenToolInputs: SharedBox<[JSONValue?]>
    ) async throws -> (Agent, RunState) {
        let (model, agent) = try makeModelAndAgent(tools: [try needsOk(calls: calls, seenContexts: seenContexts, seenToolInputs: seenToolInputs)], name: "agent")
        try model.enqueue(.output([getFunctionToolCall("needs_ok", arguments: #"{"text": "one"}"#, callId: "1")]))
        try model.enqueue(.output([getFinalOutputMessage("done")]))

        let first = try await Runner.run(agent, input: "hi", context: ["user": "original"] as JSONObject)
        let interruption = try #require(first.interruptions.first)
        let state = first.toState()
        state.context.toolInput = ["scoped": true] as JSONObject
        try state.approve(interruption)
        let restored = try await RunState.fromJSON(agent, try state.toJSON())
        #expect(restored.context.toolInput as? JSONValue == ["scoped": true])
        #expect(!restored.context.approvals.isEmpty)
        return (agent, restored)
    }

    @Test(arguments: [false, true])
    func testResumeApprovedFunctionApprovalWithContextOverride(streamed: Bool) async throws {
        // Ports test_resume_approved_function_approval_via_json_with_context_override and its
        // streamed variant.
        let calls = SharedBox<[String]>([])
        let seenContexts = SharedBox<[JSONObject]>([])
        let seenToolInputs = SharedBox<[JSONValue?]>([])
        let (agent, restored) = try await interruptedApprovalStateWithToolInput(calls: calls, seenContexts: seenContexts, seenToolInputs: seenToolInputs)
        let restoredWrapper = restored.context
        let override: JSONObject = ["user": "reviewer"]

        let result: any RunResultBase
        if streamed {
            let resumed = try Runner.runStreamed(agent, state: restored, context: override)
            for try await _ in resumed.streamEvents() {}
            result = resumed
        } else {
            result = try await Runner.run(agent, state: restored, context: override)
        }

        #expect(result.finalOutput as? String == "done")
        #expect(result.interruptions.isEmpty)
        #expect(calls.value == ["one"])
        #expect(seenContexts.value == [override])
        #expect(seenToolInputs.value == [["scoped": true]])
        #expect(result.contextWrapper === restoredWrapper)
        #expect(result.contextWrapper.context(as: JSONObject.self) == override)
        #expect(result.contextWrapper.toolInput as? JSONValue == ["scoped": true])
        #expect(result.contextWrapper.toolStateBox === restoredWrapper.toolStateBox)
    }
}

@Suite struct RunStateHostedMCPAndGuardrailTests {
    private func hostedApproval(_ agent: Agent, _ requestId: String, serverLabel: String = "server-a") -> ToolApprovalItem {
        ToolApprovalItem(agent: agent, rawItem: McpApprovalRequest(id: requestId, name: "lookup_account", arguments: "{}", serverLabel: serverLabel).json)
    }

    @Test func testHostedMCPApprovalRequestRestoresMatchingServerTool() async throws {
        let serverA = HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "server-a", "server_url": "https://server-a.example/mcp"])
        let serverB = HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "server-b", "server_url": "https://server-b.example/mcp"])
        let agent = Agent(name: "test", tools: [serverA, serverB])
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()))
        state.lastProcessedResponse = makeProcessedResponse(mcpApprovalRequests: [
            ToolRunMCPApprovalRequest(requestItem: McpApprovalRequest(id: "request-a", name: "lookup_account", arguments: "{}", serverLabel: "server-a"), mcpTool: serverA),
        ])

        let restored = try await RunState.fromJSON(agent, try state.toJSON())

        let requests = try #require(restored.lastProcessedResponse?.mcpApprovalRequests)
        #expect(requests.count == 1)
        #expect(requests.first?.mcpTool.toolConfig == serverA.toolConfig)
    }

    @Test func testHostedMCPExactRejectionOverridesStickyApprovalAfterRoundTrip() async throws {
        let agent = Agent(name: "test")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()))
        try state.approve(hostedApproval(agent, "sticky-request"), alwaysApprove: true)
        try state.reject(hostedApproval(agent, "exception-request"), rejectionMessage: "denied exactly")

        let restored = try await RunState.fromJSON(agent, try state.toJSON())

        let context = restored.context
        #expect(try context.getApprovalStatus(toolName: "lookup_account", callId: "exception-request", existingPending: hostedApproval(agent, "exception-request")) == false)
        let exception = hostedApproval(agent, "exception-request")
        #expect(context.getRejectionMessage(toolName: "lookup_account", callId: "exception-request", existingPending: exception) == "denied exactly")
        #expect(try context.getApprovalStatus(toolName: "lookup_account", callId: "other-request", existingPending: hostedApproval(agent, "other-request")) == true)
    }

    @Test func testSchema113IgnoresTypedHostedMCPApprovalRecords() async throws {
        let agent = Agent(name: "test")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()))
        try state.approve(hostedApproval(agent, "request-a"), alwaysApprove: true)
        var serialized = try state.toJSON()
        serialized["$schemaVersion"] = "1.13"

        let restored = try await RunState.fromJSON(agent, serialized)

        #expect(try restored.context.getApprovalStatus(toolName: "lookup_account", callId: "request-next", existingPending: hostedApproval(agent, "request-next")) == nil)
    }

    private struct Verdict: Codable, Sendable { var allowed: Bool; var reason: String }
    private struct Answer: Codable, Sendable { var text: String }

    @Test func testRunnerGuardrailModelsSurviveStateSerialization() async throws {
        // Swift: `Encodable` values stand in for upstream's pydantic models.
        let check = GuardrailFunctionOutput(outputInfo: Verdict(allowed: true, reason: "approved"), tripwireTriggered: false)
        let agent = Agent(
            name: "Audit",
            model: .instance(try ScriptedModel(steps: [.output([assistantMessage(#"{"text":"hello"}"#)])])),
            inputGuardrails: [InputGuardrail { _, _, _ in check }],
            outputGuardrails: [OutputGuardrail { _, _, _ in check }],
            outputType: try AgentOutputSchema(Answer.self)
        )
        let result = try await Runner.run(agent, input: "hello", runConfig: RunConfig(tracingDisabled: true))

        let restored = try await RunState.fromJSON(agent, try JSONObject.parse(try result.toState().toString()))

        let expected: JSONValue = ["allowed": true, "reason": "approved"]
        #expect(restored.inputGuardrailResults.first?.output.outputInfo as? JSONValue == expected)
        #expect(restored.outputGuardrailResults.first?.output.outputInfo as? JSONValue == expected)
        #expect(restored.outputGuardrailResults.first?.agentOutput as? JSONValue == ["text": "hello"])
    }

    @Test func testToolGuardrailDataclassesSurviveStateSerialization() async throws {
        let output = ToolGuardrailFunctionOutput(outputInfo: ["checks": [Verdict(allowed: true, reason: "approved")] as [any Sendable]] as [String: any Sendable], behavior: .allow)
        let agent = Agent(name: "Audit")
        let state = makeState(agent, context: RunContextWrapper())
        state.toolInputGuardrailResults = [ToolInputGuardrailResult(guardrail: ToolInputGuardrail(name: "input") { _ in output }, output: output)]
        state.toolOutputGuardrailResults = [ToolOutputGuardrailResult(guardrail: ToolOutputGuardrail(name: "output") { _ in output }, output: output)]

        let restored = try await RunState.fromJSON(agent, try JSONObject.parse(try state.toString()))

        let expected: JSONValue = ["checks": [["allowed": true, "reason": "approved"]]]
        #expect(restored.toolInputGuardrailResults.first?.output.outputInfo as? JSONValue == expected)
        #expect(restored.toolOutputGuardrailResults.first?.output.outputInfo as? JSONValue == expected)
    }

    private struct Diagnostic: Encodable, Sendable {
        var reason: String
        func encode(to encoder: any Encoder) throws {
            throw ValueError("Serializer unavailable.")
        }
    }

    @Test func testGuardrailStateKeepsFallbackWhenModelSerializerRaises() async throws {
        // Swift: the fallback is `String(describing:)`, the counterpart of upstream's `str()`.
        let diagnostic = Diagnostic(reason: "approved")
        let expected = JSONValue.string(String(describing: diagnostic))
        let output = GuardrailFunctionOutput(outputInfo: diagnostic, tripwireTriggered: false)
        let toolOutput = ToolGuardrailFunctionOutput(outputInfo: diagnostic, behavior: .allow)
        let agent = Agent(name: "Audit")
        let state = makeState(agent, context: RunContextWrapper())
        state.inputGuardrailResults = [InputGuardrailResult(guardrail: InputGuardrail { _, _, _ in output }, output: output)]
        state.outputGuardrailResults = [OutputGuardrailResult(guardrail: OutputGuardrail { _, _, _ in output }, agentOutput: diagnostic, agent: agent, output: output)]
        state.toolInputGuardrailResults = [ToolInputGuardrailResult(guardrail: ToolInputGuardrail { _ in toolOutput }, output: toolOutput)]
        state.toolOutputGuardrailResults = [ToolOutputGuardrailResult(guardrail: ToolOutputGuardrail { _ in toolOutput }, output: toolOutput)]

        let restored = try await RunState.fromJSON(agent, try JSONObject.parse(try state.toString()))

        #expect(restored.inputGuardrailResults.first?.output.outputInfo as? JSONValue == expected)
        #expect(restored.outputGuardrailResults.first?.output.outputInfo as? JSONValue == expected)
        #expect(restored.outputGuardrailResults.first?.agentOutput as? JSONValue == expected)
        #expect(restored.toolInputGuardrailResults.first?.output.outputInfo as? JSONValue == expected)
        #expect(restored.toolOutputGuardrailResults.first?.output.outputInfo as? JSONValue == expected)
    }
}
