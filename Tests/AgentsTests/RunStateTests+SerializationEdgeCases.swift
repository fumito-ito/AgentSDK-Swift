import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_state.py (class TestRunStateSerializationEdgeCases, item and
// processed-response cases). Schema, programmatic tool calling, and occurrence cases are in
// RunStateTests+SchemaEdgeCases.swift.
//
// Swift: `test_serialize_computer_action_with_description`, `..._shell_action_...`, and
// `..._apply_patch_action_...` assign an ad-hoc `description` attribute to tools that do not
// declare one; Swift tools cannot gain attributes, so they are not ported.
// `test_serialize_item_with_non_dict_non_model_raw_item` is not ported because raw items are
// always `JSONObject` values. `test_deserialize_processed_response_agent_without_get_all_tools`
// is covered by `testDeserializeProcessedResponseWithoutGetAllTools`, since every Swift agent
// has `getAllTools`.

private final class DummyComputer: Computer {
    func screenshot() async throws -> String { "screenshot" }
    func click(x: Int, y: Int, button: Button) async throws {}
    func doubleClick(x: Int, y: Int) async throws {}
    func scroll(x: Int, y: Int, scrollX: Int, scrollY: Int) async throws {}
    func type(text: String) async throws {}
    func wait() async throws {}
    func move(x: Int, y: Int) async throws {}
    func keypress(keys: [String]) async throws {}
    func drag(path: [(x: Int, y: Int)]) async throws {}
}

private struct DummyEditor: ApplyPatchEditor {
    func createFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? { nil }
    func updateFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? { nil }
    func deleteFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? { nil }
}

/// A value JSON cannot represent (upstream uses `datetime`).
private struct Timestamp: Sendable, CustomStringConvertible {
    var description: String { "2024-01-01 12:00:00" }
}

private func emptyContext() -> RunContextWrapper {
    RunContextWrapper(context: JSONObject())
}

private func processedResponseData(_ overrides: JSONObject) -> JSONObject {
    var data: JSONObject = [
        "new_items": [], "handoffs": [], "functions": [], "computer_actions": [], "local_shell_actions": [],
        "mcp_approval_requests": [], "tools_used": [], "interruptions": [],
    ]
    for (key, value) in overrides { data[key] = value }
    return data
}

private func processedResponseEntry(_ key: String, _ entry: JSONObject) -> JSONObject {
    processedResponseData([key: [.object(entry)]])
}

private func functionCallData(_ name: String, namespace: String? = nil, callId: String = "call123") -> JSONValue {
    var call: JSONObject = ["type": "function_call", "name": .string(name), "call_id": .string(callId), "status": "completed", "arguments": "{}"]
    if let namespace { call["namespace"] = .string(namespace) }
    return .object(call)
}

private func deserialize(_ items: [JSONObject], _ agentMap: [String: Agent]) throws -> [RunItem] {
    try deserializeItems(items.map(JSONValue.object), agents: RunStateAgentResolver(agentMap: agentMap))
}

private func deserializeOne(_ item: JSONObject, _ agentMap: [String: Agent]) throws -> [RunItem] {
    try deserialize([item], agentMap)
}

private func deserializeProcessed(_ data: JSONObject, _ agent: Agent, _ agentMap: [String: Agent] = [:]) async throws -> ProcessedResponse {
    try await deserializeProcessedResponse(data, currentAgent: agent, context: emptyContext(), agents: RunStateAgentResolver(agentMap: agentMap))
}

@Suite struct RunStateSerializationEdgeCasesTests {
    // MARK: - Serialization

    @Test func testToJSONIncludesToolCallItemsFromLastProcessedResponse() throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext())
        let toolCall = ResponseFunctionToolCall(callId: "call123", name: "test_tool", arguments: "{}", status: "completed").json
        state.lastProcessedResponse = makeProcessedResponse(newItems: [ToolCallItem(agent: agent, rawItem: toolCall)])

        let generatedItems = try #require(try state.toJSON()["generated_items"]?.arrayValue)
        #expect(generatedItems.count == 1)
        #expect(generatedItems[0].objectValue?["type"] == "tool_call_item")
        #expect(generatedItems[0].objectValue?["raw_item"]?.objectValue?["name"] == "test_tool")
    }

    @Test func testToJSONCamelizesNestedDictsAndLists() throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext())
        state.generatedItems.append(MessageOutputItem(agent: agent, rawItem: makeMessageOutput(messageId: "msg1", text: "Hello")))

        let generatedItems = try #require(try state.toJSON()["generated_items"]?.arrayValue)
        #expect(generatedItems.count == 1)
        let rawItem = try #require(generatedItems[0].objectValue?["raw_item"]?.objectValue)
        #expect(rawItem["response_id"] != nil || rawItem["id"] != nil)
    }

    @Test func testToStringSerializesNonJSONOutputs() throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext())
        state.generatedItems.append(ToolCallOutputItem(
            agent: agent,
            rawItem: ["type": "function_call_output", "call_id": "call123", "output": "ok"],
            output: ["timestamp": Timestamp()] as [String: any Sendable]
        ))

        let jsonData = try JSONObject.parse(try state.toString())

        let generatedItems = try #require(jsonData["generated_items"]?.arrayValue)
        #expect(generatedItems.count == 1)
        let outputPayload = try #require(generatedItems[0].objectValue?["output"]?.objectValue)
        #expect(outputPayload["timestamp"]?.stringValue != nil)
    }

    @Test func testFromJSONWithLastProcessedResponse() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext())
        let toolCall = ResponseFunctionToolCall(callId: "call123", name: "test_tool", arguments: "{}", status: "completed").json
        state.lastProcessedResponse = makeProcessedResponse(newItems: [ToolCallItem(agent: agent, rawItem: toolCall)])

        let newState = try await RunState.fromJSON(agent, try state.toJSON())

        let processed = try #require(newState.lastProcessedResponse)
        #expect(processed.newItems.count == 1)
        #expect(processed.newItems[0].type == "tool_call_item")
    }

    @Test func testLastProcessedResponseSerializesLocalShellActions() async throws {
        let localShellTool = LocalShellTool { _ in "ok" }
        let agent = Agent(name: "TestAgent", tools: [localShellTool])
        let state = makeState(agent, context: emptyContext())
        let localShellCall = try #require(LocalShellCall(json: [
            "type": "local_shell_call", "id": "ls1", "call_id": "call_local", "status": "completed",
            "action": ["commands": ["echo hi"], "timeout_ms": 1000],
        ]))
        state.lastProcessedResponse = makeProcessedResponse(localShellCalls: [ToolRunLocalShellCall(toolCall: localShellCall, localShellTool: localShellTool)])

        let jsonData = try state.toJSON()
        let lastProcessed = try #require(jsonData["last_processed_response"]?.objectValue)
        #expect(lastProcessed["local_shell_actions"]?.arrayValue?[0].objectValue?["local_shell"]?.objectValue?["name"] == "local_shell")

        let newState = try await RunState.fromJSON(agent, jsonData, contextOverride: JSONObject())
        let restored = try #require(newState.lastProcessedResponse?.localShellCalls.first)
        #expect(restored.localShellTool.name == "local_shell")
        #expect(restored.toolCall.callId == "call_local")
    }

    @Test func testSerializeToolActionGroups() throws {
        // Swift: upstream serializes duck-typed mock actions; Swift actions are typed, so real
        // tools with the same names are used.
        let functionTool = try FunctionTool.create(name: "func_tool") { _ in "ok" }
        let shellTool = try ShellTool(executor: { _ in .text("ok") }, name: "shell_tool")
        let applyPatchTool = try ApplyPatchTool(editor: DummyEditor(), name: "apply_patch_tool")
        let handoff = Agents.handoff(Agent(name: "Target"), toolNameOverride: "handoff_tool")
        let mcpTool = HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "server"])
        let computerCall = try #require(ResponseComputerToolCall(json: ["type": "computer_call", "call_id": "comp-call"]))
        let localShellCall = try #require(LocalShellCall(json: ["type": "local_shell_call", "call_id": "local-call"]))
        let processed = makeProcessedResponse(
            handoffs: [ToolRunHandoff(handoff: handoff, toolCall: makeToolCall("handoff-call", name: "handoff_tool"))],
            functions: [ToolRunFunction(toolCall: makeToolCall("func-call", name: "func_tool"), functionTool: functionTool)],
            computerActions: [ToolRunComputerAction(toolCall: computerCall, computerTool: ComputerTool(computer: DummyComputer()))],
            localShellCalls: [ToolRunLocalShellCall(toolCall: localShellCall, localShellTool: LocalShellTool { _ in "ok" })],
            shellCalls: [ToolRunShellCall(toolCall: ["type": "function_call", "call_id": "shell-call"], shellTool: shellTool)],
            applyPatchCalls: [ToolRunApplyPatchCall(toolCall: ["type": "function_call", "call_id": "patch-call"], applyPatchTool: applyPatchTool)],
            mcpApprovalRequests: [ToolRunMCPApprovalRequest(requestItem: McpApprovalRequest(id: "req", name: "mcp_tool", arguments: "{}", serverLabel: "server"), mcpTool: mcpTool)]
        )

        let serialized = JSONObject(serializeToolActionGroups(processed))
        #expect(Set(serialized.keys) == [
            "functions", "computer_actions", "custom_tool_actions", "local_shell_actions", "shell_actions",
            "apply_patch_actions", "handoffs", "mcp_approval_requests",
        ])
        let function = try #require(serialized["functions"]?.arrayValue?.first?.objectValue)
        #expect(function["tool"]?.objectValue?["name"] == "func_tool")
        #expect(function["tool_call"]?.objectValue?["call_id"] == "func-call")
        #expect(serialized["handoffs"]?.arrayValue?.first?.objectValue?["handoff"]?.objectValue?["tool_name"] == "handoff_tool")
        #expect(serialized["mcp_approval_requests"]?.arrayValue?.first?.objectValue?["mcp_tool"]?.objectValue?["name"] == "hosted_mcp")
    }

    @Test func testSerializeToolActionGroupsPreservesSyntheticNamespaceForDeferredTools() throws {
        let deferredTool = try FunctionTool.create(name: "get_weather", argumentsType: NoArguments.self, deferLoading: true) { _, _ in "Tokyo" }
        let toolCall = try #require(ResponseFunctionToolCall(json: getFunctionToolCall(
            "get_weather", arguments: #"{"city": "Tokyo"}"#, callId: "weather-call", namespace: "get_weather"
        )))
        let processed = makeProcessedResponse(functions: [ToolRunFunction(toolCall: toolCall, functionTool: deferredTool)])

        let function = try #require(JSONObject(serializeToolActionGroups(processed))["functions"]?.arrayValue?.first?.objectValue)
        let tool = try #require(function["tool"]?.objectValue)
        #expect(tool["name"] == "get_weather")
        #expect(tool["namespace"] == nil)
        #expect(tool["qualifiedName"] == nil)
        #expect(tool["lookupKey"] == ["kind": "deferred_top_level", "name": "get_weather"])
        #expect(function["tool_call"]?.objectValue?["namespace"] == "get_weather")
    }

    @Test func testSerializeGuardrailResults() {
        // Swift: input and output results are separate types, so each has its own serializer.
        let guardrailOutput = GuardrailFunctionOutput(outputInfo: ["info": "details"] as JSONObject, tripwireTriggered: false)
        let agent = Agent(name: "AgentA")
        let outputResult = OutputGuardrailResult(
            guardrail: OutputGuardrail(name: "output") { _, _, _ in guardrailOutput },
            agentOutput: "some_output",
            agent: agent,
            output: guardrailOutput
        )
        let inputResult = InputGuardrailResult(guardrail: InputGuardrail(name: "input") { _, _, _ in guardrailOutput }, output: guardrailOutput)

        let serialized = serializeInputGuardrailResults([inputResult]) + serializeOutputGuardrailResults([outputResult], nil)
        #expect(Set(serialized.compactMap { $0.objectValue?["guardrail"]?.objectValue?.string("type") }) == ["input", "output"])
        let outputEntry = serialized.compactMap(\.objectValue).first { $0["guardrail"]?.objectValue?["type"] == "output" }
        #expect(outputEntry?["agentOutput"] == "some_output")
        #expect(outputEntry?["agent"]?.objectValue?["name"] == "AgentA")
    }

    @Test func testSerializeHandoffWithNameFallback() throws {
        // Swift: every `Handoff` has a `toolName`, so the fallback is the handoff's own name.
        let agentA = Agent(name: "AgentA")
        let handoffRun = ToolRunHandoff(
            handoff: Agents.handoff(Agent(name: "Target"), toolNameOverride: "handoff_tool"),
            toolCall: ResponseFunctionToolCall(callId: "call123", name: "handoff_tool", arguments: "{}", status: "completed")
        )
        let state = makeState(agentA, context: emptyContext())
        state.lastProcessedResponse = makeProcessedResponse(handoffs: [handoffRun])

        let handoffs = try #require(try state.toJSON()["last_processed_response"]?.objectValue?["handoffs"]?.arrayValue)
        #expect(handoffs.count == 1)
        #expect(handoffs[0].objectValue?["handoff"]?.objectValue?["tool_name"] == "handoff_tool")
    }

    @Test func testSerializeFunctionWithDescriptionAndSchema() throws {
        let agent = Agent(name: "TestAgent")
        let tool = try FunctionTool(
            name: "test_tool",
            description: "Test tool description",
            paramsJSONSchema: ["type": "object", "properties": [:]],
            onInvokeTool: { _, _ in "result" }
        )
        let functionRun = ToolRunFunction(toolCall: makeToolCall("call123", name: "test_tool"), functionTool: tool)
        let state = makeState(agent, context: emptyContext())
        state.lastProcessedResponse = makeProcessedResponse(functions: [functionRun])

        let functions = try #require(try state.toJSON()["last_processed_response"]?.objectValue?["functions"]?.arrayValue)
        #expect(functions.count == 1)
        let toolData = try #require(functions[0].objectValue?["tool"]?.objectValue)
        #expect(toolData["description"] == "Test tool description")
        #expect(toolData["paramsJsonSchema"] != nil)
    }

    @Test func testSerializeMCPApprovalRequest() throws {
        let agent = Agent(name: "TestAgent")
        let requestRun = ToolRunMCPApprovalRequest(
            requestItem: McpApprovalRequest(id: "req123", name: "mcp_tool", arguments: "{}", serverLabel: "test_server"),
            mcpTool: HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "test_server"])
        )
        let state = makeState(agent, context: emptyContext())
        state.lastProcessedResponse = makeProcessedResponse(mcpApprovalRequests: [requestRun])

        let requests = try #require(try state.toJSON()["last_processed_response"]?.objectValue?["mcp_approval_requests"]?.arrayValue)
        #expect(requests.count == 1)
        #expect(requests[0].objectValue?["request_item"] != nil)
        // Swift: hosted MCP tools report the upstream `HostedMCPTool.name` ("hosted_mcp").
        #expect(requests[0].objectValue?["mcp_tool"]?.objectValue?["name"] == "hosted_mcp")
        _ = try state.toString()
    }

    @Test func testSerializeItemWithNonDictRawItem() throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext())
        state.generatedItems.append(MessageOutputItem(agent: agent, rawItem: makeMessageOutput(messageId: "msg1", text: "Hello")))

        let generatedItems = try #require(try state.toJSON()["generated_items"]?.arrayValue)
        #expect(generatedItems.count == 1)
        #expect(generatedItems[0].objectValue?["type"] == "message_output_item")
    }

    // MARK: - Item deserialization

    @Test func testDeserializeToolCallOutputItemDifferentTypes() throws {
        let agent = Agent(name: "TestAgent")
        let map = ["TestAgent": agent]

        let function = try deserializeOne([
            "type": "tool_call_output_item", "agent": ["name": "TestAgent"],
            "raw_item": ["type": "function_call_output", "call_id": "call123", "output": "result"],
        ], map)
        #expect(function.count == 1)
        #expect(function.first?.type == "tool_call_output_item")

        let computer = try deserializeOne([
            "type": "tool_call_output_item", "agent": ["name": "TestAgent"],
            "raw_item": ["type": "computer_call_output", "call_id": "call123", "output": ["type": "computer_screenshot", "screenshot": "screenshot"]],
        ], map)
        #expect(computer.count == 1)

        let shellRaw: JSONObject = ["type": "local_shell_call_output", "id": "shell123", "call_id": "call123", "output": "result"]
        let shell = try deserializeOne(["type": "tool_call_output_item", "agent": ["name": "TestAgent"], "raw_item": .object(shellRaw)], map)
        #expect(shell.count == 1)
        #expect(shell.first?.rawItem == shellRaw)
    }

    // Swift: the upstream `bytes-call-id` and `bytes-output` cases are not ported; JSON has no
    // bytes type.
    @Test(arguments: [
        ["type": "local_shell_call_output", "call_id": "call123"] as JSONObject,
        ["type": "local_shell_call_output", "id": "shell123", "output": "result"] as JSONObject,
        ["type": "local_shell_call_output", "call_id": 123, "output": "result"] as JSONObject,
        ["type": "local_shell_call_output", "call_id": "", "output": "result"] as JSONObject,
        ["type": "local_shell_call_output", "call_id": "call123", "output": 123] as JSONObject,
    ])
    func testDeserializeRejectsInvalidLocalShellCallOutput(rawItem: JSONObject) {
        #expect(throws: (any Error).self) {
            _ = try deserializeToolCallOutputRawItem(rawItem)
        }
    }

    @Test func testDeserializeReasoningItem() throws {
        let result = try deserializeOne([
            "type": "reasoning_item", "agent": ["name": "TestAgent"],
            "raw_item": ["type": "reasoning", "id": "reasoning123", "summary": [], "content": []],
        ], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.count == 1)
        #expect(result.first?.type == "reasoning_item")
    }

    @Test func testDeserializeCompactionItem() throws {
        let result = try deserializeOne([
            "type": "compaction_item", "agent": ["name": "TestAgent"], "raw_item": ["type": "compaction", "summary": "..."],
        ], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.count == 1)
        #expect(result.first?.type == "compaction_item")
        #expect(result.first?.rawItem.type == "compaction")
    }

    @Test func testDeserializeHandoffCallItem() throws {
        let result = try deserializeOne([
            "type": "handoff_call_item", "agent": ["name": "TestAgent"], "raw_item": functionCallData("handoff_tool"),
        ], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.count == 1)
        #expect(result.first?.type == "handoff_call_item")
    }

    @Test func testDeserializeHandoffOutputItemWithoutAgent() throws {
        let sourceAgent = Agent(name: "SourceAgent")
        let targetAgent = Agent(name: "TargetAgent")

        let result = try deserializeOne([
            "type": "handoff_output_item",
            "source_agent": ["name": "SourceAgent"],
            "target_agent": ["name": "TargetAgent"],
            "raw_item": ["type": "function_call_output", "call_id": "call123", "name": "transfer_to_weather", "status": "completed", "output": "payload"],
        ], ["SourceAgent": sourceAgent, "TargetAgent": targetAgent])

        #expect(result.count == 1)
        #expect(result.first?.type == "handoff_output_item")
        #expect(result.first?.agent === sourceAgent)
    }

    @Test func testDeserializeMCPItems() throws {
        let map = ["TestAgent": Agent(name: "TestAgent")]

        let list = try deserializeOne([
            "type": "mcp_list_tools_item", "agent": ["name": "TestAgent"],
            "raw_item": ["type": "mcp_list_tools", "id": "list123", "server_label": "test_server", "tools": []],
        ], map)
        #expect(list.count == 1)
        #expect(list.first?.type == "mcp_list_tools_item")

        let request = try deserializeOne([
            "type": "mcp_approval_request_item", "agent": ["name": "TestAgent"],
            "raw_item": ["type": "mcp_approval_request", "id": "req123", "name": "mcp_tool", "server_label": "test_server", "arguments": "{}"],
        ], map)
        #expect(request.count == 1)
        #expect(request.first?.type == "mcp_approval_request_item")

        let response = try deserializeOne([
            "type": "mcp_approval_response_item", "agent": ["name": "TestAgent"],
            "raw_item": [
                "type": "mcp_approval_response", "approval_request_id": "req123", "approve": true,
                "caller": ["type": "program", "caller_id": "program123"],
            ],
        ], map)
        #expect(response.count == 1)
        let responseItem = try #require(response.first as? MCPApprovalResponseItem)
        #expect(responseItem.rawItem["caller"] == ["type": "program", "caller_id": "program123"])
    }

    @Test func testDeserializeToolApprovalItem() throws {
        let result = try deserializeOne([
            "type": "tool_approval_item", "agent": ["name": "TestAgent"], "raw_item": functionCallData("test_tool"),
        ], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.count == 1)
        #expect(result.first?.type == "tool_approval_item")
    }

    @Test func testDeserializeItemsFallbackUnionType() throws {
        let result = try deserializeOne([
            "type": "tool_call_output_item", "agent": ["name": "TestAgent"],
            "raw_item": ["type": "function_call_output", "call_id": "call123", "output": "result"],
        ], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.count == 1)
        #expect(result.first?.type == "tool_call_output_item")
    }

    @Test func testDeserializeItemsUnionAdapterFallback() throws {
        let result = try deserializeOne([
            "type": "tool_call_output_item", "agent": ["name": "TestAgent"],
            "raw_item": ["call_id": "call123", "output": "result"], "output": "result",
        ], ["TestAgent": Agent(name: "TestAgent")])
        #expect(result.isEmpty)
    }

    // MARK: - Processed response deserialization

    @Test func testDeserializeProcessedResponseWithoutGetAllTools() async throws {
        let result = try await deserializeProcessed(processedResponseData([:]), Agent(name: "TestAgent"))
        #expect(result.functions.isEmpty)
    }

    @Test func testDeserializeProcessedResponseHandoffWithToolName() async throws {
        let agentA = Agent(name: "AgentA")
        let agentB = Agent(name: "AgentB")
        agentA.handoffs = [.handoff(Agents.handoff(agentB, toolNameOverride: "handoff_tool"))]

        let result = try await deserializeProcessed(
            processedResponseEntry("handoffs", ["tool_call": functionCallData("handoff_tool"), "handoff": ["tool_name": "handoff_tool"]]),
            agentA, ["AgentA": agentA, "AgentB": agentB]
        )
        #expect(result.handoffs.count == 1)
    }

    @Test func testDeserializeProcessedResponseHandoffFromDirectAgent() async throws {
        let agentB = Agent(name: "AgentB")
        let agentA = Agent(name: "AgentA", handoffs: [.agent(agentB)])
        let handoffName = Handoff.defaultToolName(agentB)

        let result = try await deserializeProcessed(
            processedResponseEntry("handoffs", ["tool_call": functionCallData(handoffName), "handoff": ["tool_name": .string(handoffName)]]),
            agentA, ["AgentA": agentA, "AgentB": agentB]
        )
        #expect(result.handoffs.count == 1)
        #expect(result.handoffs.first?.handoff.agentName == "AgentB")
    }

    @Test func testDeserializeProcessedResponseFunctionInToolsMap() async throws {
        let agent = Agent(name: "TestAgent")
        agent.tools = [try FunctionTool(name: "test_tool", description: "Test tool", paramsJSONSchema: ["type": "object", "properties": [:]], onInvokeTool: { _, _ in "result" })]

        let result = try await deserializeProcessed(
            processedResponseEntry("functions", ["tool_call": functionCallData("test_tool"), "tool": ["name": "test_tool"]]), agent
        )
        #expect(result.functions.count == 1)
    }

    @Test func testDeserializeProcessedResponseFunctionUsesNamespace() async throws {
        let agent = Agent(name: "TestAgent")
        let crmNamespace = try toolNamespace(name: "crm", description: "CRM tools", tools: [try FunctionTool.create(name: "lookup_account") { _ in "crm" }])
        let billingNamespace = try toolNamespace(name: "billing", description: "Billing tools", tools: [try FunctionTool.create(name: "lookup_account") { _ in "billing" }])
        agent.tools = crmNamespace + billingNamespace

        let result = try await deserializeProcessed(processedResponseEntry("functions", [
            "tool_call": functionCallData("lookup_account", namespace: "billing"),
            "tool": ["name": "lookup_account", "namespace": "billing"],
        ]), agent)

        #expect(result.functions.count == 1)
        #expect(result.functions.first?.functionTool === billingNamespace[0])
    }

    @Test func testDeserializeProcessedResponseRejectsQualifiedNameCollision() async throws {
        let agent = Agent(name: "TestAgent")
        let dottedTopLevel = try FunctionTool.create(name: "crm.lookup_account") { _ in "top" }
        let namespaced = try toolNamespace(name: "crm", description: "CRM tools", tools: [try FunctionTool.create(name: "lookup_account") { _ in "crm" }])[0]
        agent.tools = [dottedTopLevel, namespaced]

        await expectErrorAsync(UserError.self, containing: "qualified name `crm.lookup_account`") {
            _ = try await deserializeProcessed(processedResponseEntry("functions", [
                "tool_call": functionCallData("lookup_account", namespace: "crm"),
                "tool": ["name": "lookup_account", "namespace": "crm"],
            ]), agent)
        }
    }

    @Test func testDeserializeProcessedResponseUsesLastDuplicateTopLevelFunction() async throws {
        let agent = Agent(name: "TestAgent")
        let firstTool = try FunctionTool.create(name: "lookup") { _ in "first" }
        let secondTool = try FunctionTool.create(name: "lookup") { _ in "second" }
        agent.tools = [firstTool, secondTool]

        let result = try await deserializeProcessed(
            processedResponseEntry("functions", ["tool_call": functionCallData("lookup"), "tool": ["name": "lookup"]]), agent
        )
        #expect(result.functions.count == 1)
        #expect(result.functions.first?.functionTool === secondTool)
    }

    @Test func testDeserializeProcessedResponseUsesToolCallNamespaceForDeferredTopLevel() async throws {
        let agent = Agent(name: "TestAgent")
        let visibleTool = try FunctionTool.create(name: "lookup_account") { _ in "visible" }
        let deferredTool = try FunctionTool.create(name: "lookup_account", argumentsType: NoArguments.self, deferLoading: true) { _, _ in "deferred" }
        agent.tools = [visibleTool, deferredTool]

        let result = try await deserializeProcessed(processedResponseEntry("functions", [
            "tool_call": functionCallData("lookup_account", namespace: "lookup_account"), "tool": ["name": "lookup_account"],
        ]), agent)

        #expect(result.functions.count == 1)
        #expect(result.functions.first?.functionTool === deferredTool)
    }

    @Test func testDeserializeProcessedResponseUsesSerializedLookupKeyForDeferredTopLevel() async throws {
        let agent = Agent(name: "TestAgent")
        let visibleTool = try FunctionTool.create(name: "lookup_account") { _ in "visible" }
        let deferredTool = try FunctionTool.create(name: "lookup_account", argumentsType: NoArguments.self, deferLoading: true) { _, _ in "deferred" }
        agent.tools = [visibleTool, deferredTool]

        let result = try await deserializeProcessed(processedResponseEntry("functions", [
            "tool_call": functionCallData("lookup_account"),
            "tool": ["name": "lookup_account", "lookupKey": ["kind": "deferred_top_level", "name": "lookup_account"]],
        ]), agent)

        #expect(result.functions.count == 1)
        #expect(result.functions.first?.functionTool === deferredTool)
    }

    @Test func testDeserializeProcessedResponseComputerActionInMap() async throws {
        let agent = Agent(name: "TestAgent", tools: [ComputerTool(computer: DummyComputer())])

        let result = try await deserializeProcessed(processedResponseEntry("computer_actions", [
            "tool_call": [
                "type": "computer_call", "id": "1", "call_id": "call123", "status": "completed",
                "action": ["type": "screenshot"], "pendingSafetyChecks": [], "pending_safety_checks": [],
            ],
            "computer": ["name": "computer"],
        ]), agent)

        #expect(result.computerActions.count == 1)
    }

    @Test func testDeserializeProcessedResponseComputerActionAcceptsPreviewName() async throws {
        let agent = Agent(name: "TestAgent", tools: [ComputerTool(computer: DummyComputer())])

        let result = try await deserializeProcessed(processedResponseEntry("computer_actions", [
            "tool_call": [
                "type": "computer_call", "id": "1", "call_id": "call123", "status": "completed",
                "action": ["type": "screenshot"], "pending_safety_checks": [],
            ],
            "computer": ["name": "computer_use_preview"],
        ]), agent)

        #expect(result.computerActions.count == 1)
        #expect(result.computerActions.first?.computerTool.name == "computer_use_preview")
    }

    @Test func testDeserializeProcessedResponseShellActionWithValidationError() async throws {
        // Swift: shell calls are stored as JSON, so an invalid payload is kept as-is like upstream's
        // fallback.
        let shellTool = try ShellTool(executor: { _ in .text("test output") })
        let agent = Agent(name: "TestAgent", tools: [shellTool])

        let result = try await deserializeProcessed(processedResponseData([
            "shell_actions": [["tool_call": ["invalid_field": "invalid_value"], "shell": ["name": "shell"]]],
            "apply_patch_actions": [],
        ]), agent)

        #expect(result.shellCalls.count == 1)
        #expect(result.shellCalls.first?.toolCall == ["invalid_field": "invalid_value"])
        #expect(result.shellCalls.first?.shellTool.name == shellTool.name)
    }

    @Test func testDeserializeProcessedResponseApplyPatchActionWithException() async throws {
        let agent = Agent(name: "TestAgent", tools: [try ApplyPatchTool(editor: DummyEditor())])

        let result = try await deserializeProcessed(processedResponseData([
            "shell_actions": [],
            "apply_patch_actions": [["tool_call": ["type": "function_call", "invalid_field": "invalid_value"], "apply_patch": ["name": "apply_patch"]]],
        ]), agent)

        #expect(result.applyPatchCalls.count == 1)
        #expect(result.applyPatchCalls.first?.toolCall == ["type": "function_call", "invalid_field": "invalid_value"])
    }

    @Test func testDeserializeProcessedResponseLocalShellActionRoundTrip() async throws {
        let localShellTool = LocalShellTool { _ in "ok" }
        let agent = Agent(name: "TestAgent", tools: [localShellTool])

        let result = try await deserializeProcessed(processedResponseEntry("local_shell_actions", [
            "tool_call": [
                "type": "local_shell_call", "id": "ls1", "call_id": "call_local", "status": "completed",
                "action": ["commands": ["echo hi"], "timeout_ms": 1000],
            ],
            "local_shell": ["name": .string(localShellTool.name)],
        ]), agent)
        #expect(result.localShellCalls.count == 1)
        #expect(result.localShellCalls.first?.localShellTool.name == localShellTool.name)
        #expect(result.localShellCalls.first?.toolCall.callId == "call_local")
    }

    @Test func testDeserializeProcessedResponseMCPApprovalRequestFound() async throws {
        // Swift: upstream registers a non-hosted mock tool; only `HostedMCPTool` values are mapped,
        // so the request is skipped without failing, as upstream's comment allows.
        let agent = Agent(name: "TestAgent")
        let rawItem: JSONObject = ["type": "mcp_approval_request", "id": "req123", "name": "mcp_tool", "server_label": "test_server", "arguments": "{}"]
        let result = try await deserializeProcessed(processedResponseEntry("mcp_approval_requests", [
            "request_item": ["raw_item": .object(rawItem)],
            "mcp_tool": ["name": "mcp_tool"],
        ]), agent)
        #expect(result.mcpApprovalRequests.isEmpty)
    }

    @Test func testDeserializeProcessedResponseWithLastProcessedResponse() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext())
        state.lastProcessedResponse = makeProcessedResponse(newItems: [ToolCallItem(agent: agent, rawItem: makeToolCall("call123", name: "test_tool").json)])

        let newState = try await RunState.fromJSON(agent, try state.toJSON())
        #expect(newState.lastProcessedResponse?.newItems.count == 1)
    }

    @Test func testFromStringWithLastProcessedResponse() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext())
        state.lastProcessedResponse = makeProcessedResponse(newItems: [ToolCallItem(agent: agent, rawItem: makeToolCall("call123", name: "test_tool").json)])

        let newState = try await RunState.fromString(agent, try state.toString())
        #expect(newState.lastProcessedResponse?.newItems.count == 1)
    }

    @Test func testRunStateMergeKeepsToolOutputWithSameCallId() throws {
        let agent = Agent(name: "TestAgent")
        let toolCall = ResponseFunctionToolCall(callId: "call-merge-1", name: "test_tool", arguments: "{}", status: "completed")
        let toolOutput = ToolCallOutputItem(agent: agent, rawItem: try ItemHelpers.toolCallOutputItem(toolCall, output: "ok"), output: "ok")
        let state = makeState(agent, context: emptyContext())
        state.generatedItems = [ToolCallItem(agent: agent, rawItem: toolCall.json)]
        state.lastProcessedResponse = makeProcessedResponse(newItems: [toolOutput])

        let generatedTypes = try #require(try state.toJSON()["generated_items"]?.arrayValue).compactMap { $0.objectValue?.string("type") }
        #expect(generatedTypes.contains("tool_call_item"))
        #expect(generatedTypes.contains("tool_call_output_item"))
    }

    @Test func testDeserializeProcessedResponseHandoffWithNameFallback() async throws {
        // Swift: every `Handoff` has a `toolName`; this checks a handoff round trip by that name.
        let agentA = Agent(name: "AgentA")
        let handoff = Agents.handoff(Agent(name: "Target"), toolNameOverride: "handoff_tool")
        agentA.handoffs = [.handoff(handoff)]
        let state = makeState(agentA, context: emptyContext())
        state.lastProcessedResponse = makeProcessedResponse(handoffs: [
            ToolRunHandoff(handoff: handoff, toolCall: ResponseFunctionToolCall(callId: "call123", name: "handoff_tool", arguments: "{}", status: "completed")),
        ])

        let newState = try await RunState.fromJSON(agentA, try state.toJSON())
        #expect(newState.lastProcessedResponse?.handoffs.count == 1)
    }

    @Test func testDeserializeProcessedResponseMCPToolFound() async throws {
        let agent = Agent(name: "TestAgent")
        let mcpTool = HostedMCPTool(toolConfig: ["server_url": "http://test", "server_label": "test_server", "type": "mcp"])
        agent.tools = [mcpTool]
        let state = makeState(agent, context: emptyContext())
        state.lastProcessedResponse = makeProcessedResponse(mcpApprovalRequests: [
            ToolRunMCPApprovalRequest(
                requestItem: McpApprovalRequest(id: "req123", name: "mcp_tool", arguments: "{}", serverLabel: "test_server"),
                mcpTool: mcpTool
            ),
        ])

        let newState = try await RunState.fromJSON(agent, try state.toJSON())
        #expect(newState.lastProcessedResponse?.mcpApprovalRequests.count == 1)
    }

    @Test func testDeserializeProcessedResponseEmptyMCPToolData() async throws {
        let rawItem: JSONObject = ["type": "mcp_approval_request", "id": "req1", "server_label": "test_server", "name": "test_tool", "arguments": "{}"]
        let request: JSONObject = ["request_item": ["raw_item": .object(rawItem)], "mcp_tool": [:]]
        let result = try await deserializeProcessed([
            "new_items": [], "handoffs": [], "functions": [], "computer_actions": [], "tools_used": [],
            "mcp_approval_requests": [.object(request)],
        ], Agent(name: "TestAgent"))
        #expect(result.mcpApprovalRequests.isEmpty)
    }
}
