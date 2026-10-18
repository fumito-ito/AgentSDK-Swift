import Foundation
import Testing
@testable import Agents

// Ported from tests/test_run_context_approvals.py.

private func makeHostedMCPApprovalItem(
    _ agent: Agent,
    requestId: String,
    serverLabel: String,
    toolName: String = "lookup_account"
) -> ToolApprovalItem {
    ToolApprovalItem(
        agent: agent,
        rawItem: [
            "id": .string(requestId),
            "type": "mcp_approval_request",
            "arguments": "{}",
            "name": .string(toolName),
            "server_label": .string(serverLabel),
        ]
    )
}

private func allowLegacyBindingReconstruction(_ context: RunContextWrapper) {
    context.toolStateBox.mutate { $0.allowLegacyApprovalBindingReconstruction = true }
}

@Suite struct RunContextApprovalsTests {
    // MARK: - Hosted MCP

    @Test func testHostedMCPPermanentApprovalIsScopedByServerLabel() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let serverA = makeHostedMCPApprovalItem(agent, requestId: "request-a-1", serverLabel: "server-a")
        let serverANext = makeHostedMCPApprovalItem(agent, requestId: "request-a-2", serverLabel: "server-a")
        let serverB = makeHostedMCPApprovalItem(agent, requestId: "request-b-1", serverLabel: "server-b")

        try contextWrapper.approveTool(serverA, alwaysApprove: true)

        #expect(try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "request-a-2", existingPending: serverANext) == true)
        #expect(try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "request-b-1", existingPending: serverB) == nil)
        #expect(contextWrapper.isToolApproved(toolName: "lookup_account", callId: "request-a-1") == true)
        #expect(contextWrapper.isToolApproved(toolName: "lookup_account", callId: "request-a-2") == nil)
        #expect(contextWrapper.approvals["lookup_account"] == nil)
    }

    @Test func testHostedMCPPermanentRejectionMessageIsScopedByServerLabel() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let serverA = makeHostedMCPApprovalItem(agent, requestId: "request-a-1", serverLabel: "server-a")
        let serverANext = makeHostedMCPApprovalItem(agent, requestId: "request-a-2", serverLabel: "server-a")
        let serverB = makeHostedMCPApprovalItem(agent, requestId: "request-b-1", serverLabel: "server-b")

        try contextWrapper.rejectTool(serverA, alwaysReject: true, rejectionMessage: "server-a denied")

        #expect(contextWrapper.getRejectionMessage(toolName: "lookup_account", callId: "request-a-2", existingPending: serverANext) == "server-a denied")
        #expect(try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "request-b-1", existingPending: serverB) == nil)
        #expect(contextWrapper.getRejectionMessage(toolName: "lookup_account", callId: "request-b-1", existingPending: serverB) == nil)
    }

    @Test(arguments: [(true, false), (true, true), (false, false), (false, true)])
    func testHostedMCPNameBasedQueryPreservesExactCallDecision(approved: Bool, always: Bool) throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let approvalItem = makeHostedMCPApprovalItem(agent, requestId: "request-a-1", serverLabel: "server-a")

        if approved {
            try contextWrapper.approveTool(approvalItem, alwaysApprove: always)
        } else {
            try contextWrapper.rejectTool(approvalItem, alwaysReject: always, rejectionMessage: "server-a denied")
        }

        #expect(contextWrapper.isToolApproved(toolName: "lookup_account", callId: "request-a-1") == approved)
        #expect(contextWrapper.isToolApproved(toolName: "lookup_account", callId: "request-a-2") == nil)
        if !approved {
            #expect(contextWrapper.getRejectionMessage(toolName: "lookup_account", callId: "request-a-1") == "server-a denied")
        }
    }

    @Test func testHostedMCPExactQueryPrecedesCollidingFunctionStickyDecision() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let functionItem = makeToolApprovalItem(agent, callId: "function-call", name: "lookup_account")
        let hostedItem = makeHostedMCPApprovalItem(agent, requestId: "hosted-call", serverLabel: "server-a")

        try contextWrapper.approveTool(functionItem, alwaysApprove: true)
        try contextWrapper.rejectTool(hostedItem, rejectionMessage: "hosted denial")

        #expect(contextWrapper.isToolApproved(toolName: "lookup_account", callId: "hosted-call") == false)
        #expect(contextWrapper.isToolApproved(toolName: "lookup_account", callId: "function-next") == true)
        #expect(contextWrapper.getRejectionMessage(toolName: "lookup_account", callId: "hosted-call") == "hosted denial")
    }

    @Test(arguments: [true, false])
    func testHostedMCPExactQueryDoesNotInheritFunctionRejectionReason(hostedApproved: Bool) throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let functionItem = makeToolApprovalItem(agent, callId: "function-call", name: "lookup_account")
        let hostedItem = makeHostedMCPApprovalItem(agent, requestId: "shared-call", serverLabel: "server-a")

        try contextWrapper.rejectTool(functionItem, rejectionMessage: "function denial")
        if hostedApproved {
            try contextWrapper.approveTool(hostedItem)
        } else {
            try contextWrapper.rejectTool(hostedItem)
        }

        #expect(contextWrapper.isToolApproved(toolName: "lookup_account", callId: "shared-call") == hostedApproved)
        #expect(contextWrapper.getRejectionMessage(toolName: "lookup_account", callId: "shared-call") == nil)
    }

    @Test(arguments: [true, false])
    func testHostedMCPExactQueryDoesNotAuthorizeOtherServer(approved: Bool) throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let serverA = makeHostedMCPApprovalItem(agent, requestId: "shared-request", serverLabel: "server-a")
        let serverB = makeHostedMCPApprovalItem(agent, requestId: "shared-request", serverLabel: "server-b")

        if approved {
            try contextWrapper.approveTool(serverA)
        } else {
            try contextWrapper.rejectTool(serverA, rejectionMessage: "server-a denied")
        }

        #expect(contextWrapper.isToolApproved(toolName: "lookup_account", callId: "shared-request") == approved)
        #expect(try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "shared-request", existingPending: serverB) == nil)
        #expect(contextWrapper.getRejectionMessage(toolName: "lookup_account", callId: "shared-request", existingPending: serverB) == nil)
    }

    @Test func testHostedMCPLegacyBareNameApprovalDoesNotGrantAccess() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let pending = makeHostedMCPApprovalItem(agent, requestId: "request-a-1", serverLabel: "server-a")
        contextWrapper.rebuildApprovals(["lookup_account": ["approved": true, "rejected": []]])

        #expect(try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "request-a-1", existingPending: pending) == nil)
    }

    @Test func testHostedMCPScopedIdentityCannotAliasLegacyToolName() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let pending = makeHostedMCPApprovalItem(agent, requestId: "request-a-1", serverLabel: "server-a")
        let collidingLegacyName = #"["hosted_mcp","server-a","lookup_account"]"#
        contextWrapper.rebuildApprovals(.object([collidingLegacyName: ["approved": true, "rejected": []]]))

        #expect(contextWrapper.isToolApproved(toolName: collidingLegacyName, callId: "legacy-call") == true)
        #expect(try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "request-a-1", existingPending: pending) == nil)
    }

    @Test func testHostedMCPLegacyExactCallDecisionsRemainUsable() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let approved = makeHostedMCPApprovalItem(agent, requestId: "request-approved", serverLabel: "server-a")
        let rejected = makeHostedMCPApprovalItem(agent, requestId: "request-rejected", serverLabel: "server-a")
        let rejectedWithoutRawName = ToolApprovalItem(
            agent: agent,
            rawItem: [
                "type": "hosted_tool_call",
                "provider_data": ["type": "mcp_approval_request", "id": "request-rejected"],
            ],
            toolName: "lookup_account"
        )
        contextWrapper.rebuildApprovals([
            "lookup_account": [
                "approved": ["request-approved"],
                "rejected": ["request-rejected"],
                "rejection_messages": ["request-rejected": "legacy exact denial"],
            ],
        ])
        allowLegacyBindingReconstruction(contextWrapper)

        #expect(try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "request-approved", existingPending: approved) == true)
        #expect(try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "request-rejected", existingPending: rejected) == false)
        #expect(
            contextWrapper.getRejectionMessage(toolName: "lookup_account", callId: "request-rejected", existingPending: rejectedWithoutRawName)
                == "legacy exact denial"
        )
        #expect(try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "request-rejected", existingPending: rejectedWithoutRawName) == nil)
    }

    @Test func testHostedMCPPersistentDecisionRequiresCompleteIdentity() {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let malformed = ToolApprovalItem(
            agent: agent,
            rawItem: [
                "type": "hosted_tool_call",
                "name": "lookup_account",
                "provider_data": ["type": "mcp_approval_request", "id": "request-a-1"],
            ]
        )

        expectError(UserError.self, containing: "non-empty server_label and tool name") {
            try contextWrapper.approveTool(malformed, alwaysApprove: true)
        }
    }

    @Test func testIncompleteHostedMCPUsesOnlyExactCallDecisions() throws {
        let agent = Agent(name: "test-agent")
        let malformed = ToolApprovalItem(
            agent: agent,
            rawItem: [
                "type": "hosted_tool_call",
                "name": "lookup_account",
                "id": "request-a-1",
                "provider_data": ["type": "mcp_approval_request", "id": "request-a-1"],
            ]
        )
        let contextWrapper = RunContextWrapper(context: ())
        contextWrapper.rebuildApprovals([
            "lookup_account": ["approved": true, "rejected": true, "sticky_rejection_message": "legacy denial"],
        ])

        #expect(try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "request-a-1", existingPending: malformed) == nil)
        #expect(contextWrapper.getRejectionMessage(toolName: "lookup_account", callId: "request-a-1", existingPending: malformed) == nil)

        expectError(ModelBehaviorError.self, containing: "canonical invocation identity") {
            try contextWrapper.approveTool(malformed)
        }
        expectError(ModelBehaviorError.self, containing: "canonical invocation identity") {
            try contextWrapper.rejectTool(malformed, rejectionMessage: "exact denial")
        }
    }

    @Test func testHostedMCPDecisionRequiresRequestId() {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let malformed = ToolApprovalItem(
            agent: agent,
            rawItem: [
                "type": "hosted_tool_call",
                "name": "lookup_account",
                "provider_data": ["type": "mcp_approval_request", "server_label": "server-a"],
            ]
        )

        expectError(UserError.self, containing: "non-empty request id") {
            try contextWrapper.approveTool(malformed)
        }

        #expect(contextWrapper.approvals.isEmpty)
    }

    @Test(arguments: ["", 123] as [JSONValue])
    func testHostedMCPInvalidRequestIdDoesNotMutateApprovals(requestId: JSONValue) {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let malformed = ToolApprovalItem(
            agent: agent,
            rawItem: [
                "type": "mcp_approval_request",
                "id": requestId,
                "arguments": "{}",
                "name": "lookup_account",
                "server_label": "server-a",
            ]
        )

        expectError(UserError.self, containing: "non-empty request id") {
            try contextWrapper.rejectTool(malformed, alwaysReject: true, rejectionMessage: "must not persist")
        }

        #expect(contextWrapper.approvals.isEmpty)
    }

    @Test func testHostedMCPProviderInvalidRequestIdDoesNotFallBackToOuterId() {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let malformed = ToolApprovalItem(
            agent: agent,
            rawItem: [
                "type": "hosted_tool_call",
                "call_id": "outer-id",
                "name": "lookup_account",
                "provider_data": [
                    "type": "mcp_approval_request",
                    "id": 123,
                    "server_label": "server-a",
                    "name": "lookup_account",
                ],
            ]
        )

        expectError(UserError.self, containing: "non-empty request id") {
            try contextWrapper.approveTool(malformed)
        }

        #expect(contextWrapper.approvals.isEmpty)
    }

    @Test func testHostedMCPRequestTypeIsNotUsedAsMissingToolName() {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let malformed = ToolApprovalItem(
            agent: agent,
            rawItem: [
                "type": "mcp_approval_request",
                "id": "request-a-1",
                "arguments": "{}",
                "server_label": "server-a",
            ]
        )

        expectError(UserError.self, containing: "non-empty server_label and tool name") {
            try contextWrapper.approveTool(malformed, alwaysApprove: true)
        }

        #expect(contextWrapper.approvals.isEmpty)
    }

    // MARK: - Function tools

    @Test func testLatestApprovalDecisionWinsForCallId() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let approvalItem = makeToolApprovalItem(agent, callId: "call-1", name: "test_tool")

        try contextWrapper.approveTool(approvalItem)
        #expect(contextWrapper.isToolApproved(toolName: "test_tool", callId: "call-1") == true)

        try contextWrapper.rejectTool(approvalItem)
        #expect(contextWrapper.isToolApproved(toolName: "test_tool", callId: "call-1") == false)

        try contextWrapper.approveTool(approvalItem)
        #expect(contextWrapper.isToolApproved(toolName: "test_tool", callId: "call-1") == true)
    }

    @Test func testNamespacedApprovalStatusDoesNotFallBackToBareToolDecisions() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let bareItem = makeToolApprovalItem(agent, callId: "call-bare", name: "lookup_account")
        let billingItem = makeToolApprovalItem(agent, callId: "call-billing", name: "lookup_account", namespace: "billing")

        try contextWrapper.approveTool(bareItem, alwaysApprove: true)

        #expect(
            try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "call-billing-2", toolNamespace: "billing", existingPending: billingItem)
                == nil
        )
        #expect(try contextWrapper.getApprovalStatus(toolName: "lookup_account", callId: "call-billing-2", existingPending: billingItem) == nil)
    }

    @Test func testNamespacedRejectionMessageDoesNotFallBackToBareToolDecisions() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let bareItem = makeToolApprovalItem(agent, callId: "call-bare", name: "lookup_account")
        let billingItem = makeToolApprovalItem(agent, callId: "call-billing", name: "lookup_account", namespace: "billing")

        try contextWrapper.rejectTool(bareItem, alwaysReject: true, rejectionMessage: "bare denial")

        #expect(
            contextWrapper.getRejectionMessage(toolName: "lookup_account", callId: "call-billing-2", toolNamespace: "billing", existingPending: billingItem)
                == nil
        )
        #expect(contextWrapper.getRejectionMessage(toolName: "lookup_account", callId: "call-bare-2") == "bare denial")
    }

    @Test func testDeferredTopLevelPerCallApprovalKeepsBareNameLookup() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let deferredItem = makeToolApprovalItem(agent, callId: "call-weather", name: "get_weather", namespace: "get_weather", allowBareNameAlias: true)

        try contextWrapper.approveTool(deferredItem)

        #expect(contextWrapper.isToolApproved(toolName: "get_weather", callId: "call-weather") == true)
    }

    @Test func testDeferredTopLevelRejectionMessageKeepsBareNameLookup() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let deferredItem = makeToolApprovalItem(agent, callId: "call-weather", name: "get_weather", namespace: "get_weather", allowBareNameAlias: true)

        try contextWrapper.rejectTool(deferredItem, rejectionMessage: "weather denied")

        #expect(contextWrapper.getRejectionMessage(toolName: "get_weather", callId: "call-weather") == "weather denied")
    }

    @Test func testDeferredTopLevelPermanentApprovalDoesNotAliasToBareName() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let deferredItem = makeToolApprovalItem(agent, callId: "call-weather", name: "get_weather", namespace: "get_weather", allowBareNameAlias: true)

        try contextWrapper.approveTool(deferredItem, alwaysApprove: true)

        #expect(contextWrapper.isToolApproved(toolName: "get_weather", callId: "call-weather-2") == nil)
        #expect(contextWrapper.isToolApproved(toolName: "deferred_top_level:get_weather", callId: "future") == true)
        #expect(
            try contextWrapper.getApprovalStatus(toolName: "get_weather", callId: "call-weather-2", toolNamespace: "get_weather", existingPending: deferredItem)
                == true
        )
    }

    @Test func testDeferredTopLevelLegacyPermanentApprovalRequiresReapproval() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let deferredItem = makeToolApprovalItem(agent, callId: "call-weather", name: "get_weather", namespace: "get_weather", allowBareNameAlias: true)

        contextWrapper.rebuildApprovals(["get_weather.get_weather": ["approved": true, "rejected": []]])
        allowLegacyBindingReconstruction(contextWrapper)

        #expect(
            try contextWrapper.getApprovalStatus(toolName: "get_weather", callId: "call-weather-2", toolNamespace: "get_weather", existingPending: deferredItem)
                == nil
        )
    }

    @Test func testRebuildApprovalsIgnoresMalformedApprovalValues() {
        let contextWrapper = RunContextWrapper(context: ())

        contextWrapper.rebuildApprovals(["not", "a", "mapping"])
        #expect(contextWrapper.approvals.isEmpty)

        // Swift: upstream also passes a non-string `123` key, which JSON objects cannot hold.
        contextWrapper.rebuildApprovals([
            "get_weather": [
                "approved": ["not": "valid"],
                "rejected": ["call-denied", 123],
                "rejection_messages": ["call-denied": "no"],
            ],
        ])

        #expect(contextWrapper.isToolApproved(toolName: "get_weather", callId: "any-call") == nil)
        #expect(contextWrapper.isToolApproved(toolName: "get_weather", callId: "call-denied") == false)
        #expect(contextWrapper.getRejectionMessage(toolName: "get_weather", callId: "call-denied") == "no")
        #expect(contextWrapper.isToolApproved(toolName: "123", callId: "any-call") == nil)
    }

    @Test func testDeferredTopLevelApprovalDoesNotAliasToVisibleBareSibling() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let deferredItem = makeToolApprovalItem(agent, callId: "call-lookup", name: "lookup_account", namespace: "lookup_account", allowBareNameAlias: false)

        try contextWrapper.approveTool(deferredItem, alwaysApprove: true)

        #expect(contextWrapper.isToolApproved(toolName: "lookup_account", callId: "call-visible-2") == nil)
        #expect(
            try contextWrapper.getApprovalStatus(
                toolName: "lookup_account",
                callId: "call-deferred-2",
                toolNamespace: "lookup_account",
                existingPending: deferredItem
            ) == true
        )
    }

    @Test func testExplicitSameNameNamespaceDoesNotAliasToBareTool() throws {
        let agent = Agent(name: "test-agent")
        let contextWrapper = RunContextWrapper(context: ())
        let explicitNamespacedItem = makeToolApprovalItem(agent, callId: "call-namespaced", name: "lookup_account", namespace: "lookup_account")

        try contextWrapper.approveTool(explicitNamespacedItem, alwaysApprove: true)

        #expect(contextWrapper.isToolApproved(toolName: "lookup_account", callId: "call-bare-2") == nil)
        #expect(
            try contextWrapper.getApprovalStatus(
                toolName: "lookup_account",
                callId: "call-namespaced-2",
                toolNamespace: "lookup_account",
                existingPending: explicitNamespacedItem
            ) == true
        )
    }
}
