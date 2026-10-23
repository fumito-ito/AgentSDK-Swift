import Foundation
import Testing
@testable import Agents

// Ported from tests/test_run_internal_approvals.py.
//
// Swift: `_NamespacedToolCall` reads `namespace` from an arbitrary object. Swift raw tool calls
// are JSON, so only the dictionary and typed sources are tested.
@Suite struct RunInternalApprovalsTests {
    @Test func testFilterToolApprovalsKeepsOnlyApprovalItems() {
        let agent = Agent(name: "test")
        let approval = makeToolApprovalItem(agent)

        let filtered = filterToolApprovals(["text", approval, NSObject()])

        #expect(filtered.count == 1 && filtered.first === approval)
    }

    @Test func testApprovalsFromStepHandlesMissingAndMixedInterruptions() {
        let agent = Agent(name: "test")
        let approval = makeToolApprovalItem(agent)

        #expect(approvalsFromStep(nil).isEmpty)
        #expect(approvalsFromStep(.runAgain).isEmpty)
        let fromStep = approvalsFromStep(.interruption([approval]))
        #expect(fromStep.count == 1 && fromStep.first === approval)
    }

    @Test func testAppendInputItemsExcludingApprovalsSkipsApprovalPlaceholders() throws {
        let agent = Agent(name: "test")
        var baseInput: [TResponseInputItem] = []
        let message = MessageOutputItem(agent: agent, rawItem: makeMessageOutput(text: "done"))
        let approval = makeToolApprovalItem(agent, callId: "call_approval")

        try appendInputItemsExcludingApprovals(&baseInput, [message, approval])

        #expect(baseInput.count == 1)
        #expect(baseInput.first?.string("type") == "message")
    }

    @Test func testAppendApprovalErrorOutputEmitsFunctionToolOutput() throws {
        let agent = Agent(name: "test")
        var generatedItems: [RunItem] = []

        try appendApprovalErrorOutput(
            generatedItems: &generatedItems,
            agent: agent,
            toolCall: ["namespace": "dict_namespace"],
            toolName: "needs_approval",
            callId: nil,
            message: "approval denied"
        )

        #expect(generatedItems.count == 1)
        let outputItem = try #require(generatedItems.first as? ToolCallOutputItem)
        #expect(outputItem.agent === agent)
        #expect(outputItem.output as? String == "approval denied")
        #expect(outputItem.rawItem == [
            "type": "function_call_output",
            "call_id": "unknown",
            "output": "approval denied",
        ])
    }

    @Test func testBuildFunctionToolCallForApprovalErrorReusesTypedCalls() {
        let toolCall = makeToolCall("call_1", name: "typed_tool")

        #expect(buildFunctionToolCallForApprovalError(toolCall.json, toolName: "ignored", callId: "ignored").json == toolCall.json)
    }

    @Test func testBuildFunctionToolCallForApprovalErrorPreservesNamespaceSources() {
        let fromDict = buildFunctionToolCallForApprovalError(["namespace": "dict_namespace"], toolName: "dict_tool", callId: "call_dict")

        #expect(fromDict.namespace == "dict_namespace")
        #expect(fromDict.callId == "call_dict")
    }

    @Test func testBuildFunctionToolCallForApprovalErrorIgnoresEmptyNamespaces() {
        let toolCall = buildFunctionToolCallForApprovalError(["namespace": ""], toolName: "tool", callId: "call_1")

        #expect(toolCall.namespace == nil)
        #expect(toolCall.name == "tool")
        #expect(toolCall.arguments == "{}")
        #expect(toolCall.json.string("status") == "completed")
    }
}
