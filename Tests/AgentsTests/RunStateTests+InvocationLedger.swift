import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_state.py (the approval binding and invocation ledger tests at the end
// of class TestRunState, from `test_schema_1_13_restores_pending_approval_binding_from_interruption`
// through `test_serialized_apply_patch_approval_binds_plural_operations`, and
// `test_context_override_discards_unbound_ids_from_previous_restore`).

/// Rewrites a snapshot the way a pre-ledger writer stored approvals.
private func downgrade(_ json: JSONObject, to schemaVersion: String?, keepLedger: Bool = false) -> JSONObject {
    var json = json
    var context = json["context"]?.objectValue ?? [:]
    var approvals = context["approvals"]?.objectValue ?? [:]
    for entry in context.removeValue(forKey: "function_tool_approvals")?.arrayValue ?? [] {
        if let toolKey = entry["tool_key"]?.stringValue, let decision = entry["decision"] {
            approvals[toolKey] = decision
        }
    }
    context["approvals"] = .object(approvals)
    if !keepLedger { context.removeValue(forKey: "tool_invocations") }
    json["context"] = .object(context)
    if let schemaVersion { json["$schemaVersion"] = .string(schemaVersion) }
    return json
}

private func removingLedger(_ json: JSONObject) -> JSONObject {
    var json = json
    var context = json["context"]?.objectValue ?? [:]
    context.removeValue(forKey: "tool_invocations")
    json["context"] = .object(context)
    return json
}

private func settingLedger(_ json: JSONObject, _ ledger: JSONValue) -> JSONObject {
    var json = json
    var context = json["context"]?.objectValue ?? [:]
    context["tool_invocations"] = ledger
    json["context"] = .object(context)
    return json
}

private func updatingInvocation(_ json: JSONObject, _ callId: String, _ update: (inout JSONObject) -> Void) -> JSONObject {
    var json = json
    var context = json["context"]?.objectValue ?? [:]
    var ledger = context["tool_invocations"]?.objectValue ?? [:]
    var invocation = ledger[callId]?.objectValue ?? [:]
    update(&invocation)
    ledger[callId] = .object(invocation)
    context["tool_invocations"] = .object(ledger)
    json["context"] = .object(context)
    return json
}

/// Removes a field from the first serialized interruption's raw item.
private func removingInterruptionField(_ json: JSONObject, _ field: String) -> JSONObject {
    var json = json
    var step = json["current_step"]?.objectValue ?? [:]
    var data = step["data"]?.objectValue ?? [:]
    var interruptions = data["interruptions"]?.arrayValue ?? []
    var interruption = interruptions.first?.objectValue ?? [:]
    interruption["raw_item"] = .object((interruption["raw_item"]?.objectValue ?? [:]).removing(field))
    interruptions[0] = .object(interruption)
    data["interruptions"] = .array(interruptions)
    step["data"] = .object(data)
    json["current_step"] = .object(step)
    return json
}

private func call(_ callId: String, _ value: String) -> JSONObject {
    makeFunctionToolCall("tool1", callId: callId, arguments: #"{"value":"\#(value)"}"#)
}

private func mcpRequest(_ arguments: String) -> JSONObject {
    McpApprovalRequest(id: "request-a", name: "lookup_account", arguments: arguments, serverLabel: "server-a").json
}

@Suite struct RunStateInvocationLedgerTests {
    // MARK: - Legacy reconstruction

    @Test func testSchema113RestoresPendingApprovalBindingFromInterruption() async throws {
        let agent = Agent(name: "ApprovalLegacyAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "safe"))
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem)

        let restored = try await RunState.fromJSON(agent, downgrade(try state.toJSON(), to: "1.13"))

        let restoredItem = try #require(restored.getInterruptions().first)
        #expect(try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", existingPending: restoredItem) == true)
        let changedItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "changed"))
        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", existingPending: restoredItem, currentInvocation: changedItem)
        }
    }

    @Test(arguments: ["1.13", "1.14"])
    func testLegacySchemaStickyApprovalBindsPendingFunctionInvocation(schemaVersion: String) async throws {
        let agent = Agent(name: "ApprovalLegacyAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "safe"))
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem, alwaysApprove: true)

        let restored = try await RunState.fromJSON(agent, downgrade(try state.toJSON(), to: schemaVersion))

        let restoredItem = try #require(restored.getInterruptions().first)
        let changedItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "changed"))
        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", existingPending: restoredItem, currentInvocation: changedItem)
        }
    }

    @Test func testSchema114StickyApprovalBindsPendingHostedMCPInvocation() async throws {
        let agent = Agent(name: "ApprovalLegacyAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: mcpRequest(#"{"value":"safe"}"#))
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem, alwaysApprove: true)
        var json = removingLedger(try state.toJSON())
        json["$schemaVersion"] = "1.14"

        let restored = try await RunState.fromJSON(agent, json)

        let restoredItem = try #require(restored.getInterruptions().first)
        let changedItem = ToolApprovalItem(agent: agent, rawItem: mcpRequest(#"{"value":"changed"}"#))
        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try restored.context.getApprovalStatus(
                toolName: "lookup_account", callId: "request-a", existingPending: restoredItem, currentInvocation: changedItem
            )
        }
    }

    // MARK: - Current schema bindings

    @Test func testCurrentSchemaDoesNotReconstructMissingApprovalBinding() async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "safe"))
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem)

        let restored = try await RunState.fromJSON(agent, removingLedger(try state.toJSON()))

        let restoredItem = try #require(restored.getInterruptions().first)
        #expect(try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", existingPending: restoredItem) == nil)
    }

    private final class ApprovalProbeHooks: RunHooks, @unchecked Sendable {
        let restoredItem: ToolApprovalItem
        let statuses = SharedBox<[Bool?]>([])

        init(restoredItem: ToolApprovalItem) {
            self.restoredItem = restoredItem
        }

        override func onAgentStart(context: AgentHookContext, agent: AgentBase) async throws {
            let status = try context.getApprovalStatus(toolName: "tool1", callId: "cid1", existingPending: restoredItem)
            statuses.mutate { $0.append(status) }
        }
    }

    @Test func testCurrentSchemaStickyApprovalRequiresRestoredPendingBinding() async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvedCall = call("cid1", "safe")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: approvedCall)
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem, alwaysApprove: true)

        let restored = try await RunState.fromJSON(agent, removingLedger(try state.toJSON()))

        let restoredItem = try #require(restored.getInterruptions().first)
        #expect(try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", existingPending: restoredItem) == nil)
        let changedItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "changed"))
        #expect(
            try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", existingPending: restoredItem, currentInvocation: changedItem)
                == nil
        )
        let freshItem = ToolApprovalItem(agent: agent, rawItem: call("cid-fresh", "fresh"))
        #expect(try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid-fresh", currentInvocation: freshItem) == true)

        let toolContext = try ToolContext.fromAgentContext(restored.context, toolCallId: "cid1", toolCall: approvedCall)
        #expect(try toolContext.getApprovalStatus(toolName: "tool1", callId: "cid1", existingPending: restoredItem) == nil)

        let hooks = ApprovalProbeHooks(restoredItem: restoredItem)
        let probeAgent = Agent(name: "ApprovalProbeAgent", model: .instance(try ScriptedModel(steps: [.output([getTextMessage("done")])])))
        _ = try await Runner.run(probeAgent, input: "probe approval state", context: restored.context, hooks: hooks)

        #expect(hooks.statuses.value == [nil])
        #expect(restored.context.toolInvocations["cid1"] == nil)
    }

    @Test(arguments: [false, true])
    func testCurrentSchemaMismatchedPendingLedgerBindingRequiresReapproval(sticky: Bool) async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "safe"))
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem, alwaysApprove: sticky)
        let changedCall = call("cid1", "changed")
        let changedIdentity = try #require(toolInvocationIdentity(changedCall))
        let json = updatingInvocation(try state.toJSON(), "cid1") { invocation in
            invocation["type"] = .string(changedIdentity.type)
            invocation["approval_scope"] = .string(changedIdentity.approvalScope)
            invocation["fingerprint"] = .string(changedIdentity.fingerprint)
        }

        let restored = try await RunState.fromJSON(agent, json)

        let restoredItem = try #require(restored.getInterruptions().first)
        let changedItem = ToolApprovalItem(agent: agent, rawItem: changedCall)
        #expect(
            try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", existingPending: restoredItem, currentInvocation: changedItem)
                == nil
        )
        if sticky {
            let freshItem = ToolApprovalItem(agent: agent, rawItem: call("cid-fresh", "fresh"))
            #expect(try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid-fresh", currentInvocation: freshItem) == true)
        }
    }

    // MARK: - Ledger validation

    @Test(arguments: [
        ("type", "unknown_tool_call"),
        ("approval_scope", "not-a-digest"),
        ("fingerprint", 123),
        ("fingerprint", .string(String(repeating: "A", count: 64))),
    ] as [(String, JSONValue)])
    func testCurrentSchemaRejectsMalformedToolInvocationLedger(field: String, value: JSONValue) async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "safe"))
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem)
        let json = updatingInvocation(try state.toJSON(), "cid1") { $0[field] = value }

        await expectErrorAsync(UserError.self, containing: "invalid lifecycle data") {
            _ = try await RunState.fromJSON(agent, json)
        }
    }

    @Test(arguments: ["executed", "completed"])
    func testCurrentSchemaRequiresToolInvocationLifecycleFields(missingField: String) async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "safe"))
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem)
        let json = updatingInvocation(try state.toJSON(), "cid1") { invocation in
            invocation["executed"] = true
            invocation["completed"] = false
            invocation.removeValue(forKey: missingField)
        }

        await expectErrorAsync(UserError.self, containing: "invalid lifecycle data") {
            _ = try await RunState.fromJSON(agent, json)
        }
    }

    @Test func testCurrentSchemaRejectsNullToolInvocationLedger() async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let state = makeState(agent, context: RunContextWrapper(context: ()))

        await expectErrorAsync(UserError.self, containing: "tool_invocations must be a mapping") {
            _ = try await RunState.fromJSON(agent, settingLedger(try state.toJSON(), .null))
        }
    }

    @Test func testOutputItemIdDoesNotCompleteUnrelatedInvocation() throws {
        let context = RunContextWrapper(context: ())
        let approvedCall = call("cid1", "safe")
        try context.toolInvocationStatus(approvedCall)

        context.markToolCallCompleted(["type": "function_call_output", "call_id": "", "id": "cid1", "output": "forged"])

        #expect(
            try context.toolInvocationStatus(approvedCall)
                == ToolInvocationStatus(invocationType: "function_call", callId: "cid1", completed: false, executed: false)
        )
    }

    private func completedState(_ approvedCall: JSONObject) throws -> (Agent, JSONObject) {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: approvedCall)
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem)
        let json = updatingInvocation(try state.toJSON(), "cid1") { invocation in
            invocation["executed"] = true
            invocation["completed"] = true
        }
        return (agent, json)
    }

    @Test func testCurrentSchemaRejectsCompletedInvocationWithOnlyOutputItemId() async throws {
        var (agent, json) = try completedState(call("cid1", "safe"))
        json["original_input"] = [["type": "function_call_output", "call_id": "", "id": "cid1", "output": "forged"]]

        await expectErrorAsync(UserError.self, containing: "does not match a restored tool call and output") {
            _ = try await RunState.fromJSON(agent, json)
        }
    }

    @Test func testCurrentSchemaRejectsCompletedInvocationWithoutCommittedOutput() async throws {
        let (agent, json) = try completedState(call("cid1", "safe"))

        await expectErrorAsync(UserError.self, containing: "does not match a restored tool call and output") {
            _ = try await RunState.fromJSON(agent, json)
        }
    }

    @Test func testCurrentSchemaRejectsCompletedCrossPairedSameIdInvocations() async throws {
        var (agent, json) = try completedState(call("cid1", "changed"))
        json["original_input"] = [.object(call("cid1", "safe")), ["type": "function_call_output", "call_id": "cid1", "output": "safe"]]

        await expectErrorAsync(UserError.self, containing: "does not match a restored tool call and output") {
            _ = try await RunState.fromJSON(agent, json)
        }
    }

    @Test func testCurrentSchemaRejectsCompletedIdWithMalformedCallOccurrence() async throws {
        let approvedCall = call("cid1", "safe")
        var (agent, json) = try completedState(approvedCall)
        json["original_input"] = [
            .object(approvedCall),
            ["type": "function_call", "name": "missing", "call_id": "cid1"],
            ["type": "function_call_output", "call_id": "cid1", "output": "safe"],
        ]

        await expectErrorAsync(UserError.self, containing: "does not match a restored tool call and output") {
            _ = try await RunState.fromJSON(agent, json)
        }
    }

    // MARK: - Unbindable approvals

    @Test func testCurrentSchemaMissingCallIdCannotCreateStickyApproval() async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: ["type": "function_call", "name": "tool1", "arguments": #"{"value":"safe"}"#])
        let state = makeStateWithInterruptions(agent, [approvalItem])
        let restored = try await RunState.fromJSON(agent, try state.toJSON())

        expectError(ModelBehaviorError.self, containing: "non-empty call ID") {
            try restored.approve(restored.getInterruptions()[0])
        }

        #expect(restored.context.approvals.isEmpty)
        let freshItem = ToolApprovalItem(agent: agent, rawItem: call("cid-fresh", "safe"))
        #expect(try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid-fresh", currentInvocation: freshItem) == nil)
    }

    @Test(arguments: [
        ["type": "function_call", "name": "tool1", "call_id": "cid1"],
        ["type": "mcp_approval_request", "name": "lookup_account", "server_label": "server-a", "id": "request-a"],
        ["type": "unknown_tool_call", "name": "tool1", "call_id": "cid1"],
        ["type": "unknown_tool_call", "name": "tool1", "id": "provider-id"],
        ["type": "mcp_approval_request", "name": "", "server_label": "server-a", "arguments": "{}", "id": "request-empty-name"],
        ["type": "mcp_approval_request", "name": "lookup_account", "server_label": nil, "arguments": "{}", "id": "request-null-server"],
        [
            "type": "hosted_tool_call",
            "call_id": "request-wrapped-empty-name",
            "provider_data": ["type": "mcp_approval_request", "name": "", "server_label": "server-a", "arguments": "{}"],
        ],
    ] as [JSONObject])
    func testApprovalDecisionRequiresCanonicalInvocation(rawItem: JSONObject) throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: rawItem)
        let state = makeStateWithInterruptions(agent, [approvalItem])

        expectError(ModelBehaviorError.self, containing: "canonical invocation identity") {
            try state.approve(approvalItem)
        }

        #expect(state.context.approvals.isEmpty)
    }

    @Test func testCurrentSchemaOrphanedPerCallApprovalRequiresReapproval() async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()))
        try state.approve(ToolApprovalItem(agent: agent, rawItem: call("cid1", "safe")))

        let restored = try await RunState.fromJSON(agent, settingLedger(try state.toJSON(), [:]))

        let changedItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "changed"))
        #expect(try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", currentInvocation: changedItem) == nil)
        #expect(restored.context.toolInvocations["cid1"] == nil)
    }

    @Test(arguments: ["1.13", "1.14"], ["safe", "changed"])
    func testLegacySchemaOrphanedPerCallApprovalRequiresReapproval(schemaVersion: String, value: String) async throws {
        let agent = Agent(name: "ApprovalLegacyAgent")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()))
        try state.approve(ToolApprovalItem(agent: agent, rawItem: call("cid1", "safe")))

        let restored = try await RunState.fromJSON(agent, downgrade(try state.toJSON(), to: schemaVersion))

        let currentItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", value))
        #expect(try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", currentInvocation: currentItem) == nil)
        #expect(restored.context.toolInvocations["cid1"] == nil)

        try restored.approve(currentItem)

        #expect(try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", currentInvocation: currentItem) == true)
    }

    @Test func testCurrentSchemaMissingLedgerMarksHistoricalStickyCallUnbound() async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvedCall = call("cid1", "safe")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()), originalInput: .items([approvedCall]))
        try state.approve(ToolApprovalItem(agent: agent, rawItem: approvedCall), alwaysApprove: true)

        let restored = try await RunState.fromJSON(agent, removingLedger(try state.toJSON()))

        let changedItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "changed"))
        #expect(try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", currentInvocation: changedItem) == nil)
        let freshItem = ToolApprovalItem(agent: agent, rawItem: call("cid-fresh", "fresh"))
        #expect(try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid-fresh", currentInvocation: freshItem) == true)
    }

    @Test(arguments: ["arguments", "server_label"])
    func testCurrentSchemaUnbindablePendingApprovalCannotBindReplacement(missingField: String) async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: mcpRequest(#"{"value":"safe"}"#))
        let state = makeStateWithInterruptions(agent, [approvalItem])
        state.context.rebuildApprovals(["lookup_account": ["approved": ["request-a"], "rejected": []]])
        let json = removingInterruptionField(removingLedger(try state.toJSON()), missingField)

        let restored = try await RunState.fromJSON(agent, json)

        let restoredItem = try #require(restored.getInterruptions().first)
        let currentItem = ToolApprovalItem(agent: agent, rawItem: mcpRequest(#"{"value":"changed"}"#))
        #expect(
            try restored.context.getApprovalStatus(
                toolName: "lookup_account", callId: "request-a", existingPending: restoredItem, currentInvocation: currentItem
            ) == nil
        )
    }

    @Test func testCurrentSchemaUnbindablePendingWithLedgerRequiresReapproval() async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvedItem = ToolApprovalItem(agent: agent, rawItem: mcpRequest(#"{"value":"safe"}"#))
        let state = makeStateWithInterruptions(agent, [approvedItem])
        try state.approve(approvedItem)
        let json = removingInterruptionField(try state.toJSON(), "arguments")

        let restored = try await RunState.fromJSON(agent, json)

        let restoredPending = try #require(restored.getInterruptions().first)
        let safeItem = ToolApprovalItem(agent: agent, rawItem: mcpRequest(#"{"value":"safe"}"#))
        func status() throws -> Bool? {
            try restored.context.getApprovalStatus(
                toolName: "lookup_account", callId: "request-a", existingPending: restoredPending, currentInvocation: safeItem
            )
        }
        #expect(try status() == nil)

        let changedItem = ToolApprovalItem(agent: agent, rawItem: mcpRequest(#"{"value":"changed"}"#))
        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            try restored.context.approveTool(changedItem)
        }
        #expect(try status() == nil)

        try restored.context.approveTool(safeItem)

        #expect(try status() == true)
    }

    @Test func testCurrentSchemaMissingLedgerRejectsMalformedCurrentAuthority() async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: call("cid1", "safe"))
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem)

        let restored = try await RunState.fromJSON(agent, settingLedger(try state.toJSON(), [:]))

        let restoredPending = try #require(restored.getInterruptions().first)
        let malformedCurrent = ToolApprovalItem(agent: agent, rawItem: ["type": "function_call", "name": "tool1", "call_id": "cid1"])
        #expect(
            try restored.context.getApprovalStatus(toolName: "tool1", callId: "cid1", existingPending: restoredPending, currentInvocation: malformedCurrent)
                == nil
        )
        #expect(restored.context.toolInvocations.isEmpty)
    }

    @Test(arguments: [false, true])
    func testSerializedApplyPatchApprovalBindsPluralOperations(alwaysApprove: Bool) async throws {
        let agent = Agent(name: "ApprovalCurrentAgent")
        func patchItem(_ path: String) -> ToolApprovalItem {
            ToolApprovalItem(
                agent: agent,
                rawItem: [
                    "type": "apply_patch_call",
                    "name": "apply_patch",
                    "call_id": "patch-call",
                    "operations": [["type": "delete_file", "path": .string(path)]],
                ],
                toolName: "apply_patch"
            )
        }
        let approvalItem = patchItem("safe.txt")
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem, alwaysApprove: alwaysApprove)

        let restored = try await RunState.fromJSON(agent, try state.toJSON())

        let restoredItem = try #require(restored.getInterruptions().first)
        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try restored.context.getApprovalStatus(
                toolName: "apply_patch", callId: "patch-call", existingPending: restoredItem, currentInvocation: patchItem("important.txt")
            )
        }
    }

    @Test func testContextOverrideDiscardsUnboundIdsFromPreviousRestore() async throws {
        let agent = Agent(name: "ApprovalOverrideAgent")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: makeFunctionToolCall("tool1", callId: "shared", arguments: #"{"value":"safe"}"#))
        let state = makeStateWithInterruptions(agent, [approvalItem])
        try state.approve(approvalItem)
        let malformed = settingLedger(try state.toJSON(), [:])
        let valid = try state.toJSON()
        let overrideContext = RunContextWrapper(context: JSONObject())

        _ = try await RunState.fromJSON(agent, malformed, contextOverride: overrideContext)
        #expect(overrideContext.toolStateBox.value.restoredUnboundApprovalCallIds == ["shared"])

        let restored = try await RunState.fromJSON(agent, valid, contextOverride: overrideContext)

        #expect(restored.context === overrideContext)
        #expect(overrideContext.toolStateBox.value.restoredUnboundApprovalCallIds.isEmpty)
        #expect(try overrideContext.getApprovalStatus(toolName: "tool1", callId: "shared", currentInvocation: approvalItem) == true)
    }

    // MARK: - Hosted MCP records

    // Ported from the module-level hosted MCP tests and
    // TestToolApprovalItem.test_approve_tool_rejects_uncanonical_hosted_call_dict.

    private func hostedApproval(_ agent: Agent, _ requestId: String, serverLabel: String = "server-a") -> ToolApprovalItem {
        ToolApprovalItem(
            agent: agent,
            rawItem: McpApprovalRequest(id: requestId, name: "lookup_account", arguments: "{}", serverLabel: serverLabel).json
        )
    }

    @Test func testApproveToolRejectsUncanonicalHostedCallDict() {
        let context = RunContextWrapper(context: JSONObject())
        let approvalItem = ToolApprovalItem(
            agent: Agent(name: "TestAgent"),
            rawItem: ["type": "hosted_tool_call", "name": "hosted_tool", "id": "hosted_call_123"]
        )

        expectError(ModelBehaviorError.self, containing: "canonical invocation identity") {
            try context.approveTool(approvalItem)
        }

        #expect(context.isToolApproved(toolName: "hosted_tool", callId: "hosted_call_123") == nil)
    }

    @Test func testHostedMCPApprovalRoundTripUsesTypedIdentityRecords() async throws {
        let agent = Agent(name: "test")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()))
        try state.approve(hostedApproval(agent, "request-a"), alwaysApprove: true)

        let serialized = try state.toJSON()

        let context = try #require(serialized["context"]?.objectValue)
        #expect(context["approvals"] == [:])
        let hostedApprovals = try #require(context["hosted_mcp_approvals"]?.arrayValue)
        #expect(hostedApprovals.map { $0["identity"] } == [
            ["type": "server_tool", "server_label": "server-a", "tool_name": "lookup_account"],
            ["type": "query", "tool_name": "lookup_account", "request_id": "request-a"],
        ])
        let serverDecision = try #require(hostedApprovals.first?["decision"]?.objectValue)
        #expect(serverDecision["approved"] == true)
        #expect(serverDecision["rejected"] == [])
        let stickyScope = try #require(serverDecision.string("sticky_scope"))
        let serverBinding = try #require(context["tool_invocations"]?["request-a"]?.objectValue)
        #expect(serverBinding.string("type") == "mcp_approval_request")
        #expect(serverBinding.string("approval_scope") == stickyScope)
        #expect(serverBinding.string("fingerprint") != nil)
        #expect(serverBinding["executed"] == false)
        #expect(serverBinding["completed"] == false)
        let queryDecision = try #require(hostedApprovals.last?["decision"]?.objectValue)
        #expect(queryDecision["approved"] == ["request-a"])
        #expect(queryDecision["rejected"] == [])
        #expect(queryDecision["invocations"] == nil)

        let restored = try await RunState.fromJSON(agent, serialized)

        #expect(restored.context.isToolApproved(toolName: "lookup_account", callId: "request-a") == true)
        #expect(restored.context.isToolApproved(toolName: "lookup_account", callId: "request-next") == nil)
        #expect(
            try restored.context.getApprovalStatus(
                toolName: "lookup_account", callId: "request-next", existingPending: hostedApproval(agent, "request-next")
            ) == true
        )
    }

    @Test func testIncompleteHostedMCPQueryCannotCreateApprovalAuthority() {
        let agent = Agent(name: "test")
        let context = RunContextWrapper(context: JSONObject())
        let state = makeState(agent, context: context)
        let approval = ToolApprovalItem(
            agent: agent,
            rawItem: ["type": "hosted_tool_call", "provider_data": ["type": "mcp_approval_request", "id": "request-a"]],
            toolName: "lookup_account"
        )

        expectError(ModelBehaviorError.self, containing: "canonical invocation identity") {
            try state.reject(approval, rejectionMessage: "exact denial")
        }

        #expect(context.approvals.isEmpty)
        #expect(state.serializeHostedMCPApprovals().isEmpty)
    }

    @Test func testHostedMCPRejectionQueryRoundTripDoesNotCrossServers() async throws {
        let agent = Agent(name: "test")
        let state = makeState(agent, context: RunContextWrapper(context: JSONObject()))
        let serverB = hostedApproval(agent, "shared-request", serverLabel: "server-b")
        try state.reject(hostedApproval(agent, "shared-request"), rejectionMessage: "server-a denied")

        let restored = try await RunState.fromJSON(agent, try state.toJSON())

        #expect(restored.context.isToolApproved(toolName: "lookup_account", callId: "shared-request") == false)
        #expect(restored.context.getRejectionMessage(toolName: "lookup_account", callId: "shared-request") == "server-a denied")
        #expect(try restored.context.getApprovalStatus(toolName: "lookup_account", callId: "shared-request", existingPending: serverB) == nil)
        #expect(restored.context.getRejectionMessage(toolName: "lookup_account", callId: "shared-request", existingPending: serverB) == nil)
    }

    @Test func testSchema113HostedMCPOrphanedCallDecisionsRequireReapproval() async throws {
        let agent = Agent(name: "test")
        let context = RunContextWrapper(context: JSONObject())
        context.rebuildApprovals([
            "lookup_account": [
                "approved": ["request-approved"],
                "rejected": ["request-rejected"],
                "rejection_messages": ["request-rejected": "legacy exact denial"],
            ],
        ])
        let state = makeState(agent, context: context)
        var serialized = try state.toJSON()
        serialized["$schemaVersion"] = "1.13"

        let restored = try await RunState.fromJSON(agent, serialized)

        let approved = hostedApproval(agent, "request-approved")
        let rejected = ToolApprovalItem(
            agent: agent,
            rawItem: ["type": "hosted_tool_call", "provider_data": ["type": "mcp_approval_request", "id": "request-rejected"]],
            toolName: "lookup_account"
        )
        #expect(try restored.context.getApprovalStatus(toolName: "lookup_account", callId: "request-approved", existingPending: approved) == nil)
        try restored.context.approveTool(approved)
        #expect(try restored.context.getApprovalStatus(toolName: "lookup_account", callId: "request-approved", existingPending: approved) == true)
        #expect(try restored.context.getApprovalStatus(toolName: "lookup_account", callId: "request-rejected", existingPending: rejected) == nil)
        #expect(restored.context.getRejectionMessage(toolName: "lookup_account", callId: "request-rejected", existingPending: rejected) == "legacy exact denial")
    }

    // MARK: - Checkpoint isolation

    // Ported from TestRunStateResumption.test_result_to_state_detaches_tool_decision_ledgers.
    @Test(arguments: [false, true])
    func testResultToStateDetachesToolDecisionLedgers(streamed: Bool) async throws {
        let model = try ScriptedModel()
        let executions = SharedBox<[String]>([])
        let approvalTool = try FunctionTool.create(name: "approval_tool", needsApproval: true) { _ in
            executions.mutate { $0.append("executed") }
            return "approved"
        }
        let agent = Agent(name: "TestAgent", tools: [approvalTool], model: .instance(model))
        try model.enqueue(.output([getFunctionToolCall("approval_tool", arguments: "{}")]))

        let decided: RunState
        let untouched: RunState
        if streamed {
            let result = try Runner.runStreamed(agent, input: "First input")
            for try await _ in result.streamEvents() {}
            (decided, untouched) = (result.toState(), result.toState())
        } else {
            let result = try await Runner.run(agent, input: "First input")
            (decided, untouched) = (result.toState(), result.toState())
        }
        let untouchedApprovalsBefore = try untouched.toJSON()["context"]?["function_tool_approvals"]

        try decided.approve(decided.getInterruptions()[0])

        #expect(decided.context !== untouched.context)
        #expect(decided.context.toolStateBox !== untouched.context.toolStateBox)
        #expect(try untouched.toJSON()["context"]?["function_tool_approvals"] == untouchedApprovalsBefore)

        let interruptions: [ToolApprovalItem]
        if streamed {
            let result = try Runner.runStreamed(agent, state: untouched)
            for try await _ in result.streamEvents() {}
            interruptions = result.interruptions
        } else {
            interruptions = try await Runner.run(agent, state: untouched).interruptions
        }

        #expect(!interruptions.isEmpty)
        #expect(executions.value.isEmpty)
    }
}
