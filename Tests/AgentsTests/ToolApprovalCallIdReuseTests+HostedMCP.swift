import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_tool_approval_call_id_reuse.py (hosted MCP cases).
extension ToolApprovalCallIdReuseTests {
    // MARK: - Hosted MCP

    func mcpTool(onApprovalRequest: MCPToolApprovalFunction? = nil) -> HostedMCPTool {
        HostedMCPTool(
            toolConfig: ["type": "mcp", "server_label": "test_server", "server_url": "https://example.com", "require_approval": "always"],
            onApprovalRequest: onApprovalRequest
        )
    }

    func buildMCPApprovalRequest(
        _ agent: Agent,
        name: String = "lookup",
        serverLabel: String = "test_server",
        callId: String = "mcp_call_0",
        arguments: String? = nil
    ) -> (ToolRunMCPApprovalRequest, ToolApprovalItem) {
        let requestItem = McpApprovalRequest(id: callId, name: name, arguments: arguments ?? #"{"query": "safe"}"#, serverLabel: serverLabel)
        let request = ToolRunMCPApprovalRequest(requestItem: requestItem, mcpTool: mcpTool())
        return (request, ToolApprovalItem(agent: agent, rawItem: requestItem.json, toolName: name))
    }

    func unbindableCurrentMCPRequest() -> ToolRunMCPApprovalRequest {
        let requestItem = McpApprovalRequest(json: ["id": "mcp_call_0", "type": "mcp_approval_request", "name": "lookup", "server_label": "test_server"])
        // swiftlint:disable:next force_unwrapping
        return ToolRunMCPApprovalRequest(requestItem: requestItem!, mcpTool: mcpTool())
    }

    @Test func testWrappedMCPApprovalUsesTheSameCanonicalRequestId() throws {
        let agent = Agent(name: "mcp-approval-agent")
        let approvalItem = ToolApprovalItem(
            agent: agent,
            rawItem: [
                "type": "hosted_tool_call",
                "id": "outer-item-id",
                "call_id": "shared-request",
                "name": "lookup",
                "provider_data": [
                    "type": "mcp_approval_request",
                    "name": "lookup",
                    "server_label": "test_server",
                    "arguments": #"{"query":"safe"}"#,
                ],
            ],
            toolName: "lookup"
        )
        let (changedRequest, _) = buildMCPApprovalRequest(agent, callId: "shared-request", arguments: #"{"query":"changed"}"#)
        let context = RunContextWrapper(context: JSONObject())
        try context.approveTool(approvalItem)

        #expect(Set(context.toolInvocations.keys) == ["shared-request"])
        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try collectManualMCPApprovals(
                agent: agent, requests: [changedRequest], contextWrapper: context, existingPendingByCallId: ["shared-request": approvalItem]
            )
        }
    }

    @Test func testStickyMCPApprovalRejectsSameIdOnAnotherServer() throws {
        let agent = Agent(name: "mcp-approval-agent")
        let (_, approvalItem) = buildMCPApprovalRequest(agent, serverLabel: "server_a")
        let (changedRequest, _) = buildMCPApprovalRequest(agent, serverLabel: "server_b")
        let context = RunContextWrapper(context: JSONObject())
        try context.approveTool(approvalItem, alwaysApprove: true)

        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try collectManualMCPApprovals(
                agent: agent, requests: [changedRequest], contextWrapper: context, existingPendingByCallId: ["mcp_call_0": approvalItem]
            )
        }
    }

    @Test func testStickyMCPApprovalRepromptsNewIdOnAnotherServer() throws {
        let agent = Agent(name: "mcp-approval-agent")
        let (_, approvalItem) = buildMCPApprovalRequest(agent, serverLabel: "server_a")
        let (changedRequest, _) = buildMCPApprovalRequest(agent, serverLabel: "server_b", callId: "mcp_call_1")
        let context = RunContextWrapper(context: JSONObject())
        try context.approveTool(approvalItem, alwaysApprove: true)

        let (responses, pending) = try collectManualMCPApprovals(agent: agent, requests: [changedRequest], contextWrapper: context)

        #expect(responses.isEmpty)
        #expect(pending.count == 1)
        #expect(pending.first?.rawItem == changedRequest.requestItem.json)
    }

    @Test func testSerializedStickyMCPScopeRepromptsAnotherServer() async throws {
        let agent = Agent(name: "mcp-approval-agent")
        let (_, approvalItem) = buildMCPApprovalRequest(agent, serverLabel: "server_a")
        let state = RunState(context: RunContextWrapper(context: JSONObject()), originalInput: "", startingAgent: agent)
        try state.approve(approvalItem, alwaysApprove: true)
        let restored = try await RunState.fromJSON(agent, try state.toJSON())
        let (changedRequest, _) = buildMCPApprovalRequest(agent, serverLabel: "server_b", callId: "mcp_call_1")

        let (responses, pending) = try collectManualMCPApprovals(agent: agent, requests: [changedRequest], contextWrapper: restored.context)

        #expect(responses.isEmpty)
        #expect(pending.count == 1)
    }

    @Test(arguments: ["1.13", "1.14"], ["arguments", "server_label"])
    func testUnbindableLegacyMCPApprovalRequiresCurrentReapproval(schemaVersion: String, missingField: String) async throws {
        let agent = Agent(name: "mcp-approval-agent")
        let (currentRequest, _) = buildMCPApprovalRequest(agent)
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: currentRequest.requestItem.json, toolName: "lookup")
        let state = makeStateWithInterruptions(agent, [approvalItem])
        state.context.rebuildApprovals(["lookup": ["approved": ["mcp_call_0"], "rejected": []]])
        var serialized = try state.toJSON()
        serialized["$schemaVersion"] = .string(schemaVersion)
        var context = serialized["context"]?.objectValue ?? [:]
        context.removeValue(forKey: "tool_invocations")
        serialized["context"] = .object(context)
        var step = serialized["current_step"]?.objectValue ?? [:]
        var data = step["data"]?.objectValue ?? [:]
        var interruptions = data["interruptions"]?.arrayValue ?? []
        var interruption = interruptions.first?.objectValue ?? [:]
        interruption["raw_item"] = .object((interruption["raw_item"]?.objectValue ?? [:]).removing(missingField))
        interruptions[0] = .object(interruption)
        data["interruptions"] = .array(interruptions)
        step["data"] = .object(data)
        serialized["current_step"] = .object(step)

        let restored = try await RunState.fromJSON(agent, serialized)
        let restoredItem = try #require(restored.getInterruptions().first)

        let (responses, pending) = try collectManualMCPApprovals(
            agent: agent, requests: [currentRequest], contextWrapper: restored.context, existingPendingByCallId: ["mcp_call_0": restoredItem]
        )

        #expect(responses.isEmpty)
        #expect(pending.count == 1)
        #expect(pending.first?.rawItem == currentRequest.requestItem.json)
    }

    @Test func testUnbindableMCPCallbackRequestRequiresManualReapproval() async throws {
        let callbackCalls = SharedBox(0)
        let tool = mcpTool(onApprovalRequest: { _ in
            callbackCalls.mutate { $0 += 1 }
            return MCPToolApprovalFunctionResult(approve: true)
        })
        let request: JSONObject = ["id": "mcp_call_0", "type": "mcp_approval_request", "name": "lookup", "server_label": "test_server"]
        let agent = Agent(name: "mcp-approval-agent", tools: [tool], model: .instance(try ScriptedModel(steps: [.output([request])])))

        let result = try await Runner.run(agent, input: "lookup")

        #expect(callbackCalls.value == 0)
        #expect(result.interruptions.count == 1)
        #expect(result.interruptions.first?.rawItem == request)
    }

    @Test(arguments: [false, true])
    func testUnbindableCurrentManualMCPRequestRequiresReapproval(alwaysApprove: Bool) throws {
        let agent = Agent(name: "mcp-approval-agent")
        let (_, approvalItem) = buildMCPApprovalRequest(agent)
        let currentRequest = unbindableCurrentMCPRequest()
        let context = RunContextWrapper(context: JSONObject())
        try context.approveTool(approvalItem, alwaysApprove: alwaysApprove)

        let (responses, pending) = try collectManualMCPApprovals(
            agent: agent, requests: [currentRequest], contextWrapper: context, existingPendingByCallId: ["mcp_call_0": approvalItem]
        )

        #expect(responses.isEmpty)
        #expect(pending.count == 1)
        #expect(pending.first?.rawItem == currentRequest.requestItem.json)
    }

    @Test func testUnbindableCurrentHostedMCPRequestRequiresReapproval() throws {
        let agent = Agent(name: "mcp-approval-agent")
        let (_, approvalItem) = buildMCPApprovalRequest(agent)
        let currentRequest = unbindableCurrentMCPRequest()
        let context = RunContextWrapper(context: JSONObject())
        try context.approveTool(approvalItem)
        var appended: [RunItem] = []

        let (pending, pendingIds) = try processHostedMCPApprovals(
            originalPreStepItems: [approvalItem], mcpApprovalRequests: [currentRequest], contextWrapper: context, agent: agent
        ) { appended.append($0) }

        #expect(pending.count == 1)
        #expect(pending.first?.rawItem == currentRequest.requestItem.json)
        #expect(pendingIds == ["mcp_call_0"])
        #expect(appended.map(ObjectIdentifier.init) == pending.map(ObjectIdentifier.init))
    }

    @Test(arguments: [false, true])
    func testRunnerOmitsCompletedMCPApprovalRequestReplay(withCallback: Bool) async throws {
        let callbackCalls = SharedBox(0)
        let callback: MCPToolApprovalFunction = { _ in
            callbackCalls.mutate { $0 += 1 }
            return MCPToolApprovalFunctionResult(approve: true)
        }
        let tool = mcpTool(onApprovalRequest: withCallback ? callback : nil)
        let firstRequest = McpApprovalRequest(id: "mcp_call_0", name: "lookup", arguments: #"{"query": "safe", "limit": 1}"#, serverLabel: "test_server")
        let replayedRequest = McpApprovalRequest(id: "mcp_call_0", name: "lookup", arguments: #"{"limit":1,"query":"safe"}"#, serverLabel: "test_server")
        let model = try ScriptedModel()
        try model.extend([.output([firstRequest.json]), .output([replayedRequest.json]), .output([getTextMessage("done")])])
        let agent = Agent(name: "mcp-approval-agent", tools: [tool], model: .instance(model))

        let first = try await Runner.run(agent, input: "lookup")
        let result: RunResult
        if withCallback {
            result = first
        } else {
            #expect(first.interruptions.count == 1)
            let state = first.toState()
            try state.approve(first.interruptions[0])
            result = try await Runner.run(agent, state: state)
        }

        #expect(result.finalOutput as? String == "done")
        #expect(callbackCalls.value == (withCallback ? 1 : 0))
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.filter { $0.string("type") == "mcp_approval_request" && $0.string("id") == "mcp_call_0" }.count == 1)
        #expect(modelInput.filter { $0.string("type") == "mcp_approval_response" && $0.string("approval_request_id") == "mcp_call_0" }.count == 1)
    }

    @Test func testSerializedCompletedManualMCPApprovalSkipsExactReplay() async throws {
        let agent = Agent(name: "mcp-approval-agent")
        let context = RunContextWrapper(context: JSONObject())
        let (request, approvalItem) = buildMCPApprovalRequest(agent)
        let state = RunState(context: context, originalInput: "", startingAgent: agent)
        try state.approve(approvalItem)

        let (firstResponses, firstPending) = try collectManualMCPApprovals(
            agent: agent, requests: [request], contextWrapper: context, existingPendingByCallId: ["mcp_call_0": approvalItem]
        )

        #expect(firstResponses.count == 1)
        #expect(firstPending.isEmpty)
        let response = try #require(firstResponses.first)
        context.markToolCallCompleted(response.rawItem)
        state.generatedItems = [approvalItem, response]

        let restored = try await RunState.fromString(agent, try state.toString())
        let (replayedResponses, replayedPending) = try collectManualMCPApprovals(
            agent: agent, requests: [request], contextWrapper: restored.context, existingPendingByCallId: ["mcp_call_0": approvalItem]
        )

        #expect(replayedResponses.isEmpty)
        #expect(replayedPending.isEmpty)
    }

    @Test func testCompletedHostedMCPApprovalReconciliationSkipsExactReplay() throws {
        let agent = Agent(name: "mcp-approval-agent")
        let context = RunContextWrapper(context: JSONObject())
        let (request, approvalItem) = buildMCPApprovalRequest(agent)
        try context.approveTool(approvalItem)
        var appended: [RunItem] = []

        _ = try processHostedMCPApprovals(
            originalPreStepItems: [approvalItem], mcpApprovalRequests: [request], contextWrapper: context, agent: agent
        ) { appended.append($0) }

        #expect(appended.count == 1)
        context.markToolCallCompleted(appended[0].rawItem)
        appended.removeAll()

        let (pending, pendingIds) = try processHostedMCPApprovals(
            originalPreStepItems: [approvalItem], mcpApprovalRequests: [request], contextWrapper: context, agent: agent
        ) { appended.append($0) }

        #expect(appended.isEmpty)
        #expect(pending.isEmpty)
        #expect(pendingIds.isEmpty)
    }

    @Test func testStickyManualMCPApprovalRejectsSameIdForChangedToolName() throws {
        let agent = Agent(name: "mcp-approval-agent")
        let context = RunContextWrapper(context: JSONObject())
        let (_, approvalItem) = buildMCPApprovalRequest(agent)
        try context.approveTool(approvalItem, alwaysApprove: true)

        let (sameTool, _) = buildMCPApprovalRequest(agent, arguments: #"{"query": "changed"}"#)
        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try collectManualMCPApprovals(agent: agent, requests: [sameTool], contextWrapper: context, existingPendingByCallId: ["mcp_call_0": approvalItem])
        }
        let changedContext = RunContextWrapper(context: JSONObject())
        try changedContext.approveTool(approvalItem, alwaysApprove: true)
        let (changedTool, _) = buildMCPApprovalRequest(agent, name: "delete_all", arguments: #"{"confirm": true}"#)
        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try collectManualMCPApprovals(
                agent: agent, requests: [changedTool], contextWrapper: changedContext, existingPendingByCallId: ["mcp_call_0": approvalItem]
            )
        }
    }

    @Test func testStickyHostedMCPApprovalRejectsSameIdForChangedToolName() throws {
        let agent = Agent(name: "mcp-approval-agent")
        let context = RunContextWrapper(context: JSONObject())
        let (_, approvalItem) = buildMCPApprovalRequest(agent)
        try context.approveTool(approvalItem, alwaysApprove: true)

        let (sameTool, _) = buildMCPApprovalRequest(agent, arguments: #"{"query": "changed"}"#)
        var sameAppended: [RunItem] = []
        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try processHostedMCPApprovals(
                originalPreStepItems: [approvalItem], mcpApprovalRequests: [sameTool], contextWrapper: context, agent: agent
            ) { sameAppended.append($0) }
        }
        let changedContext = RunContextWrapper(context: JSONObject())
        try changedContext.approveTool(approvalItem, alwaysApprove: true)
        let (changedTool, _) = buildMCPApprovalRequest(agent, name: "delete_all", arguments: #"{"confirm": true}"#)
        var changedAppended: [RunItem] = []
        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try processHostedMCPApprovals(
                originalPreStepItems: [approvalItem], mcpApprovalRequests: [changedTool], contextWrapper: changedContext, agent: agent
            ) { changedAppended.append($0) }
        }

        #expect(sameAppended.isEmpty)
        #expect(changedAppended.isEmpty)
    }
}
