import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_state.py (class TestRunStateSerializationEdgeCases, schema,
// programmatic tool calling, nested history, and generated/session occurrence cases).
//
// Swift: not ported, and recorded in porting/manifest.json:
// - `test_programmatic_tool_calling_rechecks_allowed_callers_on_resume` and the other resume
//   checks for programmatic callers: `run_internal/tool_caller.py` is not ported yet.
// - Nested history ownership round trips (`test_nested_history_ownership_round_trips_*`,
//   `..._normalizes_raw_assistant_input_digest`, `..._remaps_after_skipped_session_item`):
//   `RunState` validates the sidecar but does not retain ownership yet.
// - `test_copied_generated_item_round_trips_to_its_session_occurrence`: occurrence keys that
//   survive `deepcopy` have no Swift equivalent; run items are matched by identity.
// - Sandbox snapshot cases: `agents.sandbox` is out of scope (porting/decisions.md D2).

private func emptyContext() -> RunContextWrapper {
    RunContextWrapper(context: JSONObject())
}

private let minimalStateJSON: JSONObject = [
    "original_input": "test",
    "current_agent": ["name": "TestAgent"],
    "context": [
        "context": [:],
        "usage": ["requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0],
        "approvals": [:],
    ],
    "max_turns": 3,
    "current_turn": 0,
    "model_responses": [],
    "generated_items": [],
]

private func messageItem(_ agent: Agent, _ text: String) -> MessageOutputItem {
    MessageOutputItem(agent: agent, rawItem: makeMessageOutput(text: text))
}

private func updating(_ json: JSONObject, _ path: [String], _ transform: (inout JSONValue) -> Void) -> JSONObject {
    var root = JSONValue.object(json)
    func walk(_ value: inout JSONValue, _ remaining: ArraySlice<String>) {
        guard let key = remaining.first else {
            transform(&value)
            return
        }
        if case .object(var object) = value {
            var child = object[key] ?? .null
            walk(&child, remaining.dropFirst())
            object[key] = child
            value = .object(object)
        } else if case .array(var array) = value, let index = Int(key), array.indices.contains(index) {
            walk(&array[index], remaining.dropFirst())
            value = .array(array)
        }
    }
    walk(&root, path[...])
    return root.objectValue ?? json
}

@Suite struct RunStateSchemaEdgeCasesTests {
    // MARK: - Schema versions

    @Test func testFromJSONMissingSchemaVersion() async {
        await expectErrorAsync(UserError.self, containing: "Run state is missing schema version") {
            _ = try await RunState.fromJSON(Agent(name: "TestAgent"), minimalStateJSON)
        }
    }

    @Test(arguments: ["1.\(Int(RunState.currentSchemaVersion.split(separator: ".")[1])! + 1)", "2.0", "9.9"])
    func testFromJSONUnsupportedSchemaVersion(schemaVersion: String) async {
        var stateJSON = minimalStateJSON
        stateJSON["$schemaVersion"] = .string(schemaVersion)
        await expectErrorAsync(UserError.self, containing: "Run state schema version is not supported") {
            _ = try await RunState.fromJSON(Agent(name: "TestAgent"), stateJSON)
        }
    }

    @Test func testFromJSONChecksSchemaBeforeSandboxEnvelope() async {
        let stateJSON: JSONObject = ["$schemaVersion": "9.9", "sandbox": ["future-sandbox-value"]]
        await expectErrorAsync(UserError.self, containing: "Run state schema version is not supported") {
            _ = try await RunState.fromJSON(Agent(name: "TestAgent"), stateJSON)
        }
    }

    // Swift: `fromJSON` takes a `JSONObject`, so the non-object payload is only checked through
    // `fromString`.
    @Test(arguments: [
        ("[{\"secret_access_key\": \"malformed-schema-secret\"}]", "must be an object", false),
        ("{\"$schemaVersion\": {\"value\": \"malformed-schema-secret\"}}", "schema version has an invalid type", true),
        ("{\"$schemaVersion\": \"malformed-schema-secret\"}", "schema version is not supported", true),
    ])
    func testMalformedSchemaShapeRedactsPublicErrors(payload: String, message: String, isObject: Bool) async throws {
        let agent = Agent(name: "TestAgent")
        let sentinel = "malformed-schema-secret"

        let stringError = await expectErrorAsync(UserError.self, containing: message) {
            _ = try await RunState.fromString(agent, payload)
        }
        #expect(stringError.map { !describeError($0).contains(sentinel) } == true)

        if isObject {
            let object = try JSONObject.parse(payload)
            let jsonError = await expectErrorAsync(UserError.self, containing: message) {
                _ = try await RunState.fromJSON(agent, object)
            }
            #expect(jsonError.map { !describeError($0).contains(sentinel) } == true)
        }
    }

    @Test func testFromJSONAcceptsPreviousSchemaVersion() async throws {
        var stateJSON = minimalStateJSON
        stateJSON["$schemaVersion"] = "1.0"
        stateJSON = updating(stateJSON, ["context", "context"]) { $0 = ["foo": "bar"] }

        let restored = try await RunState.fromJSON(Agent(name: "TestAgent"), stateJSON)

        #expect(restored.currentAgent?.name == "TestAgent")
        #expect(restored.context.context(as: JSONObject.self) == ["foo": "bar"])
    }

    @Test func testSupportedSchemaVersionsMatchReleasedBoundary() {
        #expect(RunState.supportedSchemaVersions == [
            "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "1.12",
            "1.13", "1.14", "1.15", "1.16", "1.17", RunState.currentSchemaVersion,
        ])
    }

    @Test func testSupportedSchemaVersionsHaveNonEmptySummaries() {
        #expect(Set(RunState.schemaVersionSummaries.map(\.version)) == RunState.supportedSchemaVersions)
        #expect(RunState.schemaVersionSummaries.contains { $0.version == RunState.currentSchemaVersion })
        #expect(RunState.schemaVersionSummaries.allSatisfy { !$0.summary.trimmingCharacters(in: .whitespaces).isEmpty })
    }

    @Test func testFromJSONAcceptsSchemaVersion15WithoutSandboxPayload() async throws {
        var stateJSON = minimalStateJSON
        stateJSON["$schemaVersion"] = "1.5"
        stateJSON = updating(stateJSON, ["context", "context"]) { $0 = ["foo": "bar"] }

        let restored = try await RunState.fromJSON(Agent(name: "TestAgent"), stateJSON)

        #expect(restored.currentAgent?.name == "TestAgent")
        #expect(restored.context.context(as: JSONObject.self) == ["foo": "bar"])
    }

    @Test func testFromJSONAgentNotFound() async {
        var stateJSON = minimalStateJSON
        stateJSON["$schemaVersion"] = "1.0"
        stateJSON["current_agent"] = ["name": "NonExistentAgent"]

        await expectErrorAsync(UserError.self, containing: "Run state agent not found in agent map") {
            _ = try await RunState.fromJSON(Agent(name: "TestAgent"), stateJSON)
        }
    }

    // MARK: - Programmatic tool calling

    private let program: JSONObject = ["id": "program_item", "call_id": "call_program", "code": "lookup()", "fingerprint": "fingerprint", "type": "program"]

    @Test func testProgrammaticToolCallingRoundTripUsesCurrentSchema() async throws {
        // Swift: raw items are JSON, so the round trip preserves them by value instead of
        // rebuilding `Program`/`ProgramOutput` model types.
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext(), originalInput: "test")
        let functionCall: JSONObject = [
            "id": "function_item", "call_id": "call_function", "name": "lookup", "arguments": "{}",
            "caller": ["type": "program", "caller_id": "call_program"], "type": "function_call",
        ]
        let programOutput: JSONObject = ["id": "program_output_item", "call_id": "call_program", "result": "done", "status": "completed", "type": "program_output"]
        state.modelResponses = [ModelResponse(output: [program, functionCall, programOutput], usage: Usage(), responseId: "response_1")]
        state.generatedItems = [
            ToolCallItem(agent: agent, rawItem: program),
            ToolCallItem(agent: agent, rawItem: functionCall),
            ToolCallOutputItem(agent: agent, rawItem: programOutput, output: "done"),
        ]

        let jsonData = try state.toJSON()
        #expect(jsonData["$schemaVersion"] == .string(RunState.currentSchemaVersion))

        let restored = try await RunState.fromJSON(agent, jsonData)
        #expect(restored.modelResponses[0].output == [program, functionCall, programOutput])
        #expect(restored.modelResponses[0].output[1]["caller"] != nil)
        #expect(restored.generatedItems[0].rawItem.type == "program")
        #expect(restored.generatedItems[2].rawItem.type == "program_output")
    }

    @Test func testProgrammaticToolCallingRoundTripPreservesMappingItems() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext(), originalInput: "test")
        let mappingProgram: JSONObject = ["id": "program_item", "call_id": "call_program", "code": "lookup()", "type": "program"]
        let programOutput: JSONObject = ["id": "program_output_item", "call_id": "call_program", "result": "done", "type": "program_output"]
        state.modelResponses = [ModelResponse(output: [mappingProgram, programOutput], usage: Usage(), responseId: "response_1")]
        state.generatedItems = [
            ToolCallItem(agent: agent, rawItem: mappingProgram),
            ToolCallOutputItem(agent: agent, rawItem: programOutput, output: "done"),
        ]

        let restored = try await RunState.fromJSON(agent, try state.toJSON())

        #expect(restored.modelResponses[0].output == [mappingProgram, programOutput])
        #expect(restored.generatedItems[0].rawItem == mappingProgram)
        #expect(restored.generatedItems[1].rawItem == programOutput)
    }

    @Test func testPreviousSchemaRejectsProgrammaticToolCallingItems() async throws {
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext(), originalInput: "test")
        state.modelResponses = [ModelResponse(output: [program], usage: Usage(), responseId: "response_1")]
        var jsonData = try state.toJSON()
        jsonData["$schemaVersion"] = "1.12"

        await expectErrorAsync(UserError.self, containing: "Programmatic Tool Calling requires schema version") {
            _ = try await RunState.fromJSON(agent, jsonData)
        }
    }

    @Test func testSchema113AcceptsProgrammaticToolCallingItems() async throws {
        let agent = Agent(name: "TestAgent", tools: [ProgrammaticToolCallingTool()])
        let state = makeState(agent, context: emptyContext(), originalInput: "test")
        state.modelResponses = [ModelResponse(output: [program], usage: Usage(), responseId: "response_1")]
        var jsonData = try state.toJSON()
        jsonData["$schemaVersion"] = "1.13"

        let restored = try await RunState.fromJSON(agent, jsonData)

        #expect(restored.schemaVersion == "1.13")
        #expect(restored.modelResponses[0].output[0].type == "program")
    }

    @Test func testPreviousSchemaIgnoresProgramLikeArbitraryContext() async throws {
        let agent = Agent(name: "TestAgent")
        let context: JSONObject = ["payload": ["type": "program", "call_id": "not-a-run-item"]]
        let state = makeState(agent, context: RunContextWrapper(context: context), originalInput: "test")
        var jsonData = try state.toJSON()
        jsonData["$schemaVersion"] = "1.12"

        let restored = try await RunState.fromJSON(agent, jsonData)
        #expect(restored.context.context(as: JSONObject.self) == context)
    }

    // MARK: - Nested history ownership

    private func ownedState(_ agent: Agent, item: RunItem) throws -> (RunState, TResponseInputItem) {
        let inputItem = try #require(try runItemToInputItem(item))
        let state = makeState(agent, context: emptyContext(), originalInput: .items([inputItem]))
        state.generatedItems = [item]
        state.sessionItems = [item]
        return (state, inputItem)
    }

    @Test(arguments: [
        [:] as JSONValue,
        ["not-an-object"],
        [["index": -1, "digest": .string(String(repeating: "a", count: 64)), "input_index": 0]],
        [["index": 0, "digest": "short", "input_index": 0]],
        [["index": 0, "digest": .string(String(repeating: "a", count: 64)), "input_index": -1]],
        [["index": 9, "digest": .string(String(repeating: "a", count: 64)), "input_index": 0]],
        [["index": 0, "digest": .string(String(repeating: "a", count: 64)), "input_index": 9]],
        [["index": 0, "digest": .string(String(repeating: "a", count: 64)), "input_index": 0]],
    ] as [JSONValue])
    func testInvalidNestedHistoryOwnershipSidecarsAreRejected(invalidSidecar: JSONValue) async throws {
        let agent = Agent(name: "TestAgent")
        let (state, _) = try ownedState(agent, item: messageItem(agent, "owned"))
        var serialized = try state.toJSON()
        serialized["nested_history_owned_session_item_refs"] = invalidSidecar

        await expectErrorAsync(UserError.self) {
            _ = try await RunState.fromJSON(agent, serialized)
        }
    }

    @Test func testNestedHistoryOwnershipWithChangedInputDigestIsRejected() async throws {
        // Swift: `RunState` does not record ownership yet, so the sidecar is written directly.
        let agent = Agent(name: "TestAgent")
        let (state, inputItem) = try ownedState(agent, item: messageItem(agent, "owned"))
        var serialized = try state.toJSON()
        serialized["nested_history_owned_session_item_refs"] = [["index": 0, "digest": .string(digestInputItem(inputItem)), "input_index": 0]]
        serialized = updating(serialized, ["original_input", "0", "content", "0", "text"]) { $0 = "changed" }

        await expectErrorAsync(UserError.self, containing: "input digest does not match") {
            _ = try await RunState.fromJSON(agent, serialized)
        }
    }

    @Test func testNestedHistoryOwnershipForSkippedSessionItemIsIgnored() async throws {
        let agent = Agent(name: "TestAgent")
        let skipped = messageItem(agent, "skipped")
        let kept = messageItem(agent, "kept")
        let skippedInput = try #require(try runItemToInputItem(skipped))
        let state = makeState(agent, context: emptyContext(), originalInput: .items([skippedInput]))
        state.generatedItems = [skipped]
        state.sessionItems = [skipped, kept]
        var serialized = try state.toJSON()
        serialized["nested_history_owned_session_item_refs"] = [["index": 0, "digest": .string(digestInputItem(skippedInput)), "input_index": 0]]
        serialized = updating(serialized, ["session_items", "0", "agent", "name"]) { $0 = "UnknownAgent" }

        let (restored, messages) = try await captureLogs { capture in
            (try await RunState.fromJSON(agent, serialized), capture.messages)
        }

        #expect(restored.sessionItems.count == 1)
        #expect(restored.sessionItems[0].rawItem == kept.rawItem)
        #expect(restored.generatedItems[0].rawItem == skipped.rawItem)
        #expect(restored.generatedItems[0] !== restored.sessionItems[0])
        #expect(messages.contains { $0.contains("Ignoring nested history ownership for skipped session item") })
    }

    // MARK: - Generated and session occurrences

    @Test func testRepeatedGeneratedItemIdentityMapsToDistinctSessionOccurrences() async throws {
        let agent = Agent(name: "TestAgent")
        let repeated = messageItem(agent, "same")
        let state = makeState(agent, context: emptyContext())
        state.generatedItems = [repeated, repeated]
        state.sessionItems = [repeated, repeated]

        let serialized = try state.toJSON()
        let restored = try await RunState.fromJSON(agent, serialized)

        #expect(serialized["generated_session_item_indexes"] == [0, 1])
        #expect(restored.generatedItems[0] === restored.sessionItems[0])
        #expect(restored.generatedItems[1] === restored.sessionItems[1])
    }

    @Test(arguments: ["not-a-list", [0], [-1, nil], [2, nil], [true, nil], [0, 0]] as [JSONValue])
    func testInvalidGeneratedSessionItemIndexesAreIgnored(invalidMapping: JSONValue) async throws {
        let agent = Agent(name: "TestAgent")
        let first = messageItem(agent, "first")
        let second = messageItem(agent, "second")
        let state = makeState(agent, context: emptyContext())
        state.generatedItems = [first, second]
        state.sessionItems = [first, second]
        var serialized = try state.toJSON()
        serialized["generated_session_item_indexes"] = invalidMapping

        let (restored, messages) = try await captureLogs { capture in
            (try await RunState.fromJSON(agent, serialized), capture.messages)
        }

        #expect(zip(restored.generatedItems, restored.sessionItems).allSatisfy { $0 !== $1 })
        #expect(messages.contains { $0.contains("Ignoring invalid generated_session_item_indexes") })
    }

    @Test func testMismatchedGeneratedSessionOccurrenceIsNotAliased() async throws {
        let agent = Agent(name: "TestAgent")
        let item = messageItem(agent, "same")
        let state = makeState(agent, context: emptyContext())
        state.generatedItems = [item]
        state.sessionItems = [item]
        var serialized = try state.toJSON()
        serialized = updating(serialized, ["session_items", "0", "raw_item", "content", "0", "text"]) { $0 = "changed" }

        let (restored, messages) = try await captureLogs { capture in
            (try await RunState.fromJSON(agent, serialized), capture.messages)
        }

        #expect(restored.generatedItems[0] !== restored.sessionItems[0])
        #expect(messages.contains { $0.contains("Ignoring mismatched generated/session occurrence") })
    }

    @Test func testEqualGeneratedReplacementDoesNotClaimSessionOccurrence() async throws {
        let agent = Agent(name: "TestAgent")
        let sessionItem = messageItem(agent, "same")
        let replacement = MessageOutputItem(agent: agent, rawItem: sessionItem.rawItem)
        let state = makeState(agent, context: emptyContext())
        state.generatedItems = [replacement]
        state.sessionItems = [sessionItem]

        var serialized = try state.toJSON()
        let restored = try await RunState.fromJSON(agent, serialized)

        #expect(serialized["generated_session_item_indexes"] == [nil])
        #expect(restored.generatedItems[0] !== restored.sessionItems[0])

        serialized["$schemaVersion"] = "1.12"
        serialized.removeValue(forKey: "nested_history_owned_session_item_refs")
        serialized.removeValue(forKey: "generated_session_item_indexes")
        let restored112 = try await RunState.fromJSON(agent, serialized)
        #expect(restored112.generatedItems[0] !== restored112.sessionItems[0])
    }

    @Test func testAmbiguousCopiedGeneratedItemDoesNotClaimEqualSessionOccurrence() async throws {
        let agent = Agent(name: "TestAgent")
        let first = messageItem(agent, "same")
        let second = messageItem(agent, "same")
        let state = makeState(agent, context: emptyContext())
        state.generatedItems = [MessageOutputItem(agent: agent, rawItem: second.rawItem)]
        state.sessionItems = [first, second]

        let serialized = try state.toJSON()
        let restored = try await RunState.fromJSON(agent, serialized)

        #expect(serialized["generated_session_item_indexes"] == [nil])
        #expect(restored.sessionItems.allSatisfy { restored.generatedItems[0] !== $0 })
    }
}
