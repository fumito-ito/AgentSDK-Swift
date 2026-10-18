import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_tool_approval_call_id_reuse.py.
//
// Swift: `test_web_search_source_schema_drift_does_not_warn_during_invocation_lookup`,
// `test_invocation_lookup_preserves_legacy_model_dump_signature`, and
// `test_invocation_lookup_propagates_internal_model_dump_type_error` exercise pydantic
// `model_dump` handling; Swift raw items are already JSON, so they are not ported.
@Suite struct ToolApprovalCallIdReuseTests {
    // MARK: - Canonical identity

    @Test func testCanonicalShellIdentityIgnoresStrippedProviderMetadata() {
        let providerCall: JSONObject = [
            "type": "shell_call",
            "call_id": "shell_0",
            "action": ["commands": ["echo safe"]],
            "created_by": "server",
        ]
        var persistedCall = providerCall
        persistedCall.removeValue(forKey: "created_by")

        #expect(toolInvocationIdentity(providerCall) == toolInvocationIdentity(persistedCall))
    }

    @Test func testCanonicalShellIdentityTreatsOptionalNullsAsOmitted() {
        let providerCall: JSONObject = [
            "type": "shell_call",
            "call_id": "shell_0",
            "action": [
                "commands": ["echo safe"],
                "max_output_length": nil,
                "timeout_ms": nil,
            ],
            "environment": nil,
        ]
        var normalizedCall = providerCall
        normalizedCall["action"] = ["commands": ["echo safe"]]
        normalizedCall.removeValue(forKey: "environment")

        #expect(toolInvocationIdentity(providerCall) == toolInvocationIdentity(normalizedCall))
    }

    @Test func testCanonicalFunctionIdentityPreservesNullArgumentValues() {
        let callWithNull: JSONObject = [
            "type": "function_call",
            "call_id": "call_0",
            "name": "lookup",
            "arguments": #"{"value": null}"#,
        ]
        let callWithoutValue: JSONObject = [
            "type": "function_call",
            "call_id": "call_0",
            "name": "lookup",
            "arguments": "{}",
        ]

        #expect(toolInvocationIdentity(callWithNull) != toolInvocationIdentity(callWithoutValue))
    }

    @Test(arguments: [
        ["type": "custom_tool_call", "name": "raw_editor", "call_id": "call_0"],
        ["type": "computer_call", "call_id": "call_0"],
        ["type": "local_shell_call", "call_id": "call_0"],
        ["type": "shell_call", "call_id": "call_0"],
        ["type": "apply_patch_call", "call_id": "call_0"],
    ] as [JSONObject])
    func testCanonicalIdentityRequiresEachToolPayload(rawItem: JSONObject) {
        #expect(toolInvocationIdentity(rawItem) == nil)
    }

    // Swift: no upstream counterpart. The digests below come from the upstream writer
    // (tests/fixtures/run_state/features/v1_15_canonical_invocation_identity.json), so they pin the
    // byte-for-byte hashing that lets Swift validate Python-written invocation ledgers.
    @Test func testCanonicalIdentityDigestsMatchUpstreamWriter() throws {
        let call: JSONObject = [
            "type": "function_call",
            "name": "lookup_account",
            "call_id": "function-request-1",
            "status": "completed",
            "arguments": #"{"account_id":"account-1"}"#,
        ]

        let identity = try #require(toolInvocationIdentity(call))

        #expect(identity.type == "function_call")
        #expect(identity.callId == "function-request-1")
        #expect(identity.approvalScope == "73dc418185c1335b2f8ae49c817524e389e0f47c666a4578a8fb9d921756bbd4")
        #expect(identity.fingerprint == "1f6f2dba51c70f18ab8f13f23bc34068e31ce849b6d7323575a0a99b7303a356")
    }
}
