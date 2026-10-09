import Foundation
import Testing
@testable import Agents

// Ported from tests/test_run_context_wrapper.py.
//
// Swift: `test_run_context_to_str_or_none_handles_errors` is skipped because Swift string
// conversion cannot raise. The falsy-item test keeps its approval scenario; Swift has no truthiness.
@Suite struct RunContextWrapperTests {
    private func makeAgent() -> Agent {
        Agent(name: "test")
    }

    private func approvalItem(_ callId: String = "call-1", agent: Agent? = nil) -> ToolApprovalItem {
        ToolApprovalItem(
            agent: agent ?? makeAgent(),
            rawItem: ["type": "function_call", "name": "tool_call", "call_id": .string(callId), "arguments": "{}"]
        )
    }

    @Test func testRunContextResolveToolNameAndCallIdFallbacks() {
        let item = ToolApprovalItem(agent: makeAgent(), rawItem: ["name": "raw_tool", "id": "raw-id"], toolName: nil)

        #expect(RunContextWrapper.resolveToolName(item) == "raw_tool")
        #expect(RunContextWrapper.resolveCallId(item) == "raw-id")
    }

    @Test func testRunContextScopesApprovalsToCallIds() throws {
        let wrapper = RunContextWrapper(context: [String: String]())

        try wrapper.approveTool(approvalItem())

        #expect(wrapper.isToolApproved(toolName: "tool_call", callId: "call-1") == true)
        // A different call ID should require a fresh approval.
        #expect(wrapper.isToolApproved(toolName: "tool_call", callId: "call-2") == nil)
    }

    @Test func testRunContextScopesRejectionsToCallIds() throws {
        let wrapper = RunContextWrapper(context: [String: String]())

        try wrapper.rejectTool(approvalItem())

        #expect(wrapper.isToolApproved(toolName: "tool_call", callId: "call-1") == false)
        #expect(wrapper.isToolApproved(toolName: "tool_call", callId: "call-2") == nil)
    }

    @Test func testRunContextHonorsGlobalApprovalAndRejection() throws {
        let wrapper = RunContextWrapper(context: [String: String]())
        let approval = approvalItem()

        try wrapper.approveTool(approval, alwaysApprove: true)
        #expect(wrapper.isToolApproved(toolName: "tool_call", callId: "call-2") == true)

        try wrapper.rejectTool(approval, alwaysReject: true)
        #expect(wrapper.isToolApproved(toolName: "tool_call", callId: "call-3") == false)
    }

    @Test func testRunContextUsesFalsyPendingItemForStickyDecisions() throws {
        let approval = approvalItem()

        let approved = RunContextWrapper()
        try approved.approveTool(approval, alwaysApprove: true)
        #expect(try approved.getApprovalStatus(toolName: "tool_call", callId: "call-2", existingPending: approval) == true)

        let rejected = RunContextWrapper()
        try rejected.rejectTool(approval, alwaysReject: true, rejectionMessage: "Denied")
        #expect(rejected.getRejectionMessage(toolName: "tool_call", callId: "call-2", existingPending: approval) == "Denied")
    }

    @Test func testRunContextStoresPerCallRejectionMessages() throws {
        let wrapper = RunContextWrapper(context: [String: String]())

        try wrapper.rejectTool(approvalItem(), rejectionMessage: "Denied by policy")

        #expect(wrapper.getRejectionMessage(toolName: "tool_call", callId: "call-1") == "Denied by policy")
        #expect(wrapper.getRejectionMessage(toolName: "tool_call", callId: "call-2") == nil)
    }

    @Test func testRunContextStoresStickyRejectionMessagesForAlwaysReject() throws {
        let wrapper = RunContextWrapper(context: [String: String]())

        try wrapper.rejectTool(approvalItem(), alwaysReject: true, rejectionMessage: "")

        #expect(wrapper.getRejectionMessage(toolName: "tool_call", callId: "call-1") == "")
        #expect(wrapper.getRejectionMessage(toolName: "tool_call", callId: "call-2") == "")
    }

    @Test func testRunContextClearsRejectionMessageAfterApproval() throws {
        let wrapper = RunContextWrapper(context: [String: String]())
        let approval = approvalItem()

        try wrapper.rejectTool(approval, rejectionMessage: "Denied by policy")
        try wrapper.approveTool(approval)

        #expect(wrapper.getRejectionMessage(toolName: "tool_call", callId: "call-1") == nil)
    }

    @Test func testRunContextUnknownToolNameFallback() {
        let approval = ToolApprovalItem(agent: makeAgent(), rawItem: [:], toolName: nil)
        #expect(RunContextWrapper.resolveToolName(approval) == "unknown_tool")
    }

    @Test func testToolApprovalItemPreservesPositionalTypeArgument() {
        let raw: JSONObject = ["type": "function_call", "name": "lookup_account", "call_id": "call-1", "namespace": "billing"]

        let approval = ToolApprovalItem(agent: makeAgent(), rawItem: raw, toolName: "lookup_account")

        #expect(approval.type == "tool_approval_item")
        #expect(approval.toolName == "lookup_account")
        #expect(approval.toolNamespace == "billing")
    }

    // MARK: - Exact and sticky decisions

    @Test func testExactCallDecisionsOverrideStickyDefaults() throws {
        let agent = makeAgent()

        let approved = RunContextWrapper(context: [String: String]())
        try approved.approveTool(approvalItem("approve-sticky", agent: agent), alwaysApprove: true)
        try approved.rejectTool(approvalItem("approve-exception", agent: agent), rejectionMessage: "denied by user")

        #expect(approved.isToolApproved(toolName: "tool_call", callId: "approve-exception") == false)
        #expect(approved.getRejectionMessage(toolName: "tool_call", callId: "approve-exception") == "denied by user")
        #expect(approved.isToolApproved(toolName: "tool_call", callId: "approve-other") == true)

        let rejected = RunContextWrapper(context: [String: String]())
        try rejected.rejectTool(approvalItem("reject-sticky", agent: agent), alwaysReject: true, rejectionMessage: "denied by default")
        try rejected.approveTool(approvalItem("reject-exception", agent: agent))

        #expect(rejected.isToolApproved(toolName: "tool_call", callId: "reject-exception") == true)
        #expect(rejected.getRejectionMessage(toolName: "tool_call", callId: "reject-exception") == nil)
        #expect(rejected.isToolApproved(toolName: "tool_call", callId: "reject-other") == false)
        #expect(rejected.getRejectionMessage(toolName: "tool_call", callId: "reject-other") == "denied by default")
    }

    @Test func testMatchingExactCallDecisionsPreserveStickyDefaults() throws {
        let agent = makeAgent()

        let approved = RunContextWrapper(context: [String: String]())
        try approved.approveTool(approvalItem("approve-sticky", agent: agent), alwaysApprove: true)
        try approved.approveTool(approvalItem("approve-match", agent: agent))
        #expect(approved.isToolApproved(toolName: "tool_call", callId: "approve-other") == true)

        let rejected = RunContextWrapper(context: [String: String]())
        try rejected.rejectTool(approvalItem("reject-sticky", agent: agent), alwaysReject: true)
        try rejected.rejectTool(approvalItem("reject-match", agent: agent))
        #expect(rejected.isToolApproved(toolName: "tool_call", callId: "reject-other") == false)
    }

    @Test func testExactCallReversalsKeepOtherCallsOnStickyDefault() throws {
        let agent = makeAgent()

        let approved = RunContextWrapper(context: [String: String]())
        try approved.approveTool(approvalItem("approve-sticky", agent: agent), alwaysApprove: true)
        try approved.rejectTool(approvalItem("approve-exception", agent: agent), rejectionMessage: "denied")
        try approved.approveTool(approvalItem("approve-exception", agent: agent))

        #expect(approved.isToolApproved(toolName: "tool_call", callId: "approve-exception") == true)
        #expect(approved.getRejectionMessage(toolName: "tool_call", callId: "approve-exception") == nil)
        #expect(approved.isToolApproved(toolName: "tool_call", callId: "approve-other") == true)

        let rejected = RunContextWrapper(context: [String: String]())
        try rejected.rejectTool(approvalItem("reject-sticky", agent: agent), alwaysReject: true, rejectionMessage: "denied by default")
        try rejected.approveTool(approvalItem("reject-exception", agent: agent))
        try rejected.rejectTool(approvalItem("reject-exception", agent: agent), rejectionMessage: "denied exactly")

        #expect(rejected.isToolApproved(toolName: "tool_call", callId: "reject-exception") == false)
        #expect(rejected.getRejectionMessage(toolName: "tool_call", callId: "reject-exception") == "denied exactly")
        #expect(rejected.isToolApproved(toolName: "tool_call", callId: "reject-other") == false)
        #expect(rejected.getRejectionMessage(toolName: "tool_call", callId: "reject-other") == "denied by default")
    }
}
