import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_state.py (class TestRunState).
//
// Swift: the upstream module is split across RunStateTests*.swift files, one per upstream test
// class, to stay within SwiftLint's type and file length limits.

/// Builds a `ProcessedResponse` with empty collections by default.
func makeProcessedResponse(
    newItems: [RunItem] = [],
    handoffs: [ToolRunHandoff] = [],
    functions: [ToolRunFunction] = [],
    computerActions: [ToolRunComputerAction] = [],
    localShellCalls: [ToolRunLocalShellCall] = [],
    shellCalls: [ToolRunShellCall] = [],
    applyPatchCalls: [ToolRunApplyPatchCall] = [],
    toolsUsed: [String] = [],
    mcpApprovalRequests: [ToolRunMCPApprovalRequest] = [],
    interruptions: [ToolApprovalItem] = []
) -> ProcessedResponse {
    ProcessedResponse(
        newItems: newItems,
        handoffs: handoffs,
        functions: functions,
        computerActions: computerActions,
        localShellCalls: localShellCalls,
        shellCalls: shellCalls,
        applyPatchCalls: applyPatchCalls,
        toolsUsed: toolsUsed,
        mcpApprovalRequests: mcpApprovalRequests,
        interruptions: interruptions
    )
}

/// Creates a `RunState` with the defaults used across the run state tests.
func makeState(_ agent: Agent, context: RunContextWrapper, originalInput: RunInput = "input", maxTurns: Int? = 3) -> RunState {
    makeRunState(agent, context: context, originalInput: originalInput, maxTurns: maxTurns)
}

/// Attaches a `lastProcessedResponse` to the state.
func setLastProcessedResponse(_ state: RunState, _ agent: Agent, _ newItems: [RunItem]) {
    state.lastProcessedResponse = makeProcessedResponse(newItems: newItems)
}

private func emptyContext() -> RunContextWrapper {
    RunContextWrapper(context: JSONObject())
}

/// A context type with no JSON representation.
private final class NonMappingContext: Sendable {}

/// Swift: stands in for upstream's `@dataclass SampleContext`.
private struct SampleContext: Codable, Sendable, Equatable {
    var value: String
}

/// A value JSON cannot represent (upstream uses `datetime`).
private struct ObservedAt: Sendable, CustomStringConvertible {
    var description: String { "2026-05-08 12:00:00" }
}

@Suite struct RunStateTests {
    // MARK: - Initialization and serialization

    @Test func testResultsToStatePreserveFalsyTraceState() async throws {
        // Swift: upstream subclasses TraceState to override `__bool__`; Swift has no truthiness, so
        // this checks that the result's trace state carries over to the state unchanged.
        let traceState = TraceState(traceId: "trace_falsy")

        let model = try ScriptedModel(steps: [.output([getFinalOutputMessage("done")])])
        let result = try await Runner.run(Agent(name: "test", model: .instance(model)), input: "input")
        result.stateFields.traceState = traceState
        #expect(result.toState().traceState == traceState)

        let streamingModel = try ScriptedModel(steps: [.output([getFinalOutputMessage("done")])])
        let streamingResult = try Runner.runStreamed(Agent(name: "streaming-test", model: .instance(streamingModel)), input: "input")
        for try await _ in streamingResult.streamEvents() {}
        streamingResult.update { $0.stateFields.traceState = traceState }
        #expect(streamingResult.toState().traceState == traceState)
    }

    @Test func testInitializesWithDefaultValues() {
        let context = RunContextWrapper(context: ["foo": "bar"] as JSONObject)
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: context)

        #expect(state.currentTurn == 0)
        #expect(state.currentAgent === agent)
        #expect(state.originalInput == "input")
        #expect(state.maxTurns == 3)
        #expect(state.modelResponses.isEmpty)
        #expect(state.generatedItems.isEmpty)
        #expect(state.currentStep == nil)
        #expect(state.context.context(as: JSONObject.self) == ["foo": "bar"])
    }

    @Test func testToJSONPreservesFalsyProcessedResponse() throws {
        // Swift: upstream subclasses ProcessedResponse to override `__bool__`; an empty processed
        // response is the closest equivalent.
        let state = makeState(Agent(name: "test"), context: emptyContext())
        state.lastProcessedResponse = makeProcessedResponse()

        #expect(try state.toJSON()["last_processed_response"]?.isNull == false)
    }

    @Test func testSetToolUseTrackerSnapshotFiltersNonStrings() async throws {
        // Swift: `setToolUseTrackerSnapshot` takes `[String: [String]]`, so non-string agent names
        // cannot be passed. Non-string tool names can only arrive through JSON, which is filtered
        // on restore.
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext())

        var json = try state.toJSON()
        json["tool_use_tracker"] = ["agent1": ["tool1", "tool2"], "agent2": ["tool4", 456, "tool5"]]
        let restored = try await RunState.fromJSON(agent, json)

        let result = restored.getToolUseTrackerSnapshot()
        #expect(result["agent1"] == ["tool1", "tool2"])
        #expect(result["agent2"] == ["tool4", "tool5"])
        #expect(result.count == 2)
    }

    @Test func testToJSONAndToStringProduceValidJSON() throws {
        let agent = Agent(name: "Agent1")
        let state = makeState(agent, context: emptyContext(), originalInput: "input1", maxTurns: 2)

        let jsonData = try state.toJSON()
        #expect(jsonData["$schemaVersion"] == .string(RunState.currentSchemaVersion))
        #expect(jsonData["current_turn"] == 0)
        #expect(jsonData["current_agent"] == ["name": "Agent1"])
        #expect(jsonData["original_input"] == "input1")
        #expect(jsonData["max_turns"] == 2)
        #expect(jsonData["generated_items"] == [])
        #expect(jsonData["model_responses"] == [])

        let stringData = try state.toString()
        #expect(try JSONValue.parse(stringData) == .object(jsonData))
    }

    @Test func testMaxTurnsNoneRoundTrips() async throws {
        let agent = Agent(name: "Agent1")
        let state = makeState(agent, context: emptyContext(), originalInput: "input1", maxTurns: nil)

        let jsonData = try state.toJSON()
        #expect(jsonData["max_turns"] == .null)

        let restored = try await RunState.fromJSON(agent, jsonData)
        #expect(restored.maxTurns == nil)
    }

    @Test func testReasoningItemIdPolicySurvivesSerialization() async throws {
        let agent = Agent(name: "AgentReasoningPolicy")
        let state = makeState(agent, context: emptyContext(), originalInput: "input1", maxTurns: 2)
        state.setReasoningItemIdPolicy(.omit)
        state.generatedItems = [ReasoningItem(agent: agent, rawItem: ResponseReasoningItem(id: "rs_state").json)]

        let jsonData = try state.toJSON()
        #expect(jsonData["reasoning_item_id_policy"] == "omit")

        let restored = try await RunState.fromString(agent, try state.toString())
        #expect(restored.reasoningItemIdPolicy == .omit)

        let restoredHistory = try runItemsToInputItems(restored.generatedItems, reasoningItemIdPolicy: restored.reasoningItemIdPolicy)
        #expect(restoredHistory.count == 1)
        #expect(restoredHistory[0].type == "reasoning")
        #expect(restoredHistory[0]["id"] == nil)
    }

    @Test func testToolInputSurvivesSerializationRoundTrip() async throws {
        let context = RunContextWrapper(context: ["foo": "bar"] as JSONObject)
        context.toolInput = ["text": "hola", "target": "en"] as JSONObject
        let agent = Agent(name: "ToolInputAgent")
        let state = makeState(agent, context: context, originalInput: "input1", maxTurns: 2)

        let restored = try await RunState.fromString(agent, try state.toString())
        #expect(restored.context.toolInput as? JSONValue == ["text": "hola", "target": "en"])
    }

    @Test(.tracing) func testTraceAPIKeySerializationIsOptIn() async throws {
        let agent = Agent(name: "Agent1")
        let state = makeState(agent, context: emptyContext(), originalInput: "input1", maxTurns: 2)

        await withTrace("test", tracing: TracingConfig(apiKey: "trace-key")) {
            state.setTrace(getCurrentTrace())
        }

        let defaultJSON = try state.toJSON()
        let defaultTrace = try #require(defaultJSON["trace"]?.objectValue)
        #expect(defaultTrace["tracing_api_key"] == nil)
        let keyHash = try #require(defaultTrace.string("tracing_api_key_hash"))
        #expect(!keyHash.isEmpty)
        #expect(keyHash != "trace-key")

        let optInTrace = try #require(try state.toJSON(includeTracingAPIKey: true)["trace"]?.objectValue)
        #expect(optInTrace["tracing_api_key"] == "trace-key")
        #expect(optInTrace.string("tracing_api_key_hash") == keyHash)

        let restoredWithKey = try await RunState.fromString(agent, try state.toString(includeTracingAPIKey: true))
        #expect(restoredWithKey.traceState?.tracingAPIKey == "trace-key")
        #expect(restoredWithKey.traceState?.tracingAPIKeyHash == keyHash)

        let restoredWithoutKey = try await RunState.fromString(agent, try state.toString())
        #expect(restoredWithoutKey.traceState?.tracingAPIKey == nil)
        #expect(restoredWithoutKey.traceState?.tracingAPIKeyHash == keyHash)
        // Swift: the `resolve_trace_settings` assertions are not ported; the runner resolves trace
        // settings inline and never forwards a restored API key as the run's tracing config.
    }

    @Test func testThrowsErrorIfSchemaVersionIsMissingOrInvalid() async throws {
        let agent = Agent(name: "Agent1")
        let state = makeState(agent, context: emptyContext(), originalInput: "input1", maxTurns: 2)

        var jsonData = try state.toJSON()
        jsonData.removeValue(forKey: "$schemaVersion")
        let stringData = JSONValue.object(jsonData).jsonString()
        await expectErrorAsync(UserError.self, containing: "Run state is missing schema version") {
            _ = try await RunState.fromString(agent, stringData)
        }

        jsonData["$schemaVersion"] = "0.1"
        let supported = RunState.supportedSchemaVersions.sorted().joined(separator: ", ")
        let invalidData = JSONValue.object(jsonData).jsonString()
        await expectErrorAsync(
            UserError.self,
            containing: "Run state schema version is not supported. Supported versions are: \(supported). "
                + "New snapshots are written as version \(RunState.currentSchemaVersion)."
        ) {
            _ = try await RunState.fromString(agent, invalidData)
        }
    }

    // MARK: - Approvals

    @Test func testApproveUpdatesContextApprovalsCorrectly() throws {
        let agent = Agent(name: "Agent2")
        let state = makeState(agent, context: emptyContext(), originalInput: "", maxTurns: 1)
        let approvalItem = makeToolApprovalItem(agent, callId: "cid123", name: "toolX", arguments: "arguments")

        try state.approve(approvalItem)

        #expect(state.context.isToolApproved(toolName: "toolX", callId: "cid123") == true)
    }

    @Test func testReturnsUndefinedWhenApprovalStatusIsUnknown() {
        #expect(emptyContext().isToolApproved(toolName: "unknownTool", callId: "cid999") == nil)
    }

    @Test func testRejectUpdatesContextApprovalsCorrectly() throws {
        let agent = Agent(name: "Agent3")
        let state = makeState(agent, context: emptyContext(), originalInput: "", maxTurns: 1)
        let approvalItem = makeToolApprovalItem(agent, callId: "cid456", name: "toolY", arguments: "arguments")

        try state.reject(approvalItem)

        #expect(state.context.isToolApproved(toolName: "toolY", callId: "cid456") == false)
    }

    @Test func testRejectStoresRejectionMessage() throws {
        let agent = Agent(name: "AgentRejectMessage")
        let state = makeState(agent, context: emptyContext(), originalInput: "", maxTurns: 1)
        let approvalItem = makeToolApprovalItem(agent, callId: "cid456", name: "toolY", arguments: "arguments")

        try state.reject(approvalItem, rejectionMessage: "Denied by reviewer")

        #expect(state.context.getRejectionMessage(toolName: "toolY", callId: "cid456") == "Denied by reviewer")
    }

    // MARK: - Context serialization

    @Test func testToJSONNonMappingContextWarnsAndOmits() async throws {
        let agent = Agent(name: "AgentMapping")
        let state = makeState(agent, context: RunContextWrapper(context: NonMappingContext()), originalInput: "input", maxTurns: 1)

        let (jsonData, messages) = try await captureLogs { capture in
            (try state.toJSON(), capture.messages)
        }

        let contextEntry = try #require(jsonData["context"]?.objectValue)
        #expect(contextEntry["context"] == [:])
        let contextMeta = try #require(contextEntry["context_meta"]?.objectValue)
        #expect(contextMeta["omitted"] == true)
        #expect(contextMeta["serialized_via"] == "omitted")
        #expect(messages.contains { $0.contains("not serializable") })
    }

    @Test func testToJSONStrictContextRequiresSerializer() {
        let agent = Agent(name: "AgentMapping")
        let state = makeState(agent, context: RunContextWrapper(context: NonMappingContext()), originalInput: "input", maxTurns: 1)

        expectError(UserError.self, containing: "context_serializer") {
            _ = try state.toJSON(strictContext: true)
        }
    }

    @Test func testFromJSONWithContextDeserializer() async throws {
        let agent = Agent(name: "AgentMapping")
        let state = makeState(agent, context: RunContextWrapper(context: SampleContext(value: "hello")), originalInput: "input", maxTurns: 1)
        let jsonData = try state.toJSON()

        let newState = try await RunState.fromJSON(agent, jsonData) { payload in
            try JSONValue.object(payload).decode(as: SampleContext.self)
        }

        #expect(newState.context.context(as: SampleContext.self) == SampleContext(value: "hello"))
    }

    @Test func testToJSONWithContextSerializerRecordsMetadata() throws {
        final class CustomContext: Sendable {
            let value: String
            init(value: String) { self.value = value }
        }
        let agent = Agent(name: "AgentMapping")
        let state = makeState(agent, context: RunContextWrapper(context: CustomContext(value: "ok")), originalInput: "input", maxTurns: 1)

        let jsonData = try state.toJSON { value in
            ["value": .string((value as? CustomContext)?.value ?? "")]
        }

        let contextEntry = try #require(jsonData["context"]?.objectValue)
        #expect(contextEntry["context"] == ["value": "ok"])
        let contextMeta = try #require(contextEntry["context_meta"]?.objectValue)
        #expect(contextMeta["serialized_via"] == "context_serializer")
        #expect(contextMeta["requires_deserializer"] == true)
        #expect(contextMeta["omitted"] == false)
    }

    @Test func testFromJSONWarnsWithoutDeserializer() async throws {
        let agent = Agent(name: "AgentMapping")
        let state = makeState(agent, context: RunContextWrapper(context: SampleContext(value: "hello")), originalInput: "input", maxTurns: 1)
        let jsonData = try state.toJSON()

        let messages = try await captureLogs { capture in
            _ = try await RunState.fromJSON(agent, jsonData)
            return capture.messages
        }

        #expect(messages.contains { $0.contains("context_deserializer") })
    }

    @Test func testFromJSONStrictContextRequiresDeserializer() async throws {
        let agent = Agent(name: "AgentMapping")
        let state = makeState(agent, context: RunContextWrapper(context: SampleContext(value: "hello")), originalInput: "input", maxTurns: 1)
        let jsonData = try state.toJSON()

        await expectErrorAsync(UserError.self, containing: "context_deserializer") {
            _ = try await RunState.fromJSON(agent, jsonData, strictContext: true)
        }
    }

    @Test func testFromJSONContextDeserializerCanReturnWrapper() async throws {
        let agent = Agent(name: "AgentMapping")
        let state = makeState(agent, context: RunContextWrapper(context: SampleContext(value: "hello")), originalInput: "input", maxTurns: 1)
        let jsonData = try state.toJSON()

        let newState = try await RunState.fromJSON(agent, jsonData) { payload in
            RunContextWrapper(context: try JSONValue.object(payload).decode(as: SampleContext.self))
        }

        #expect(newState.context.context(as: SampleContext.self) == SampleContext(value: "hello"))
    }

    @Test func testToJSONPydanticContextRecordsMetadata() async throws {
        // Swift: upstream's pydantic `model_dump` branch maps to `Encodable` contexts.
        let agent = Agent(name: "AgentMapping")
        let state = makeState(agent, context: RunContextWrapper(context: SampleContext(value: "hello")), originalInput: "input", maxTurns: 1)

        let (jsonData, messages) = try await captureLogs { capture in
            (try state.toJSON(), capture.messages)
        }

        let contextMeta = try #require(jsonData["context"]?.objectValue?["context_meta"]?.objectValue)
        #expect(contextMeta["original_type"] == "encodable")
        #expect(contextMeta["serialized_via"] == "encodable")
        #expect(contextMeta["requires_deserializer"] == true)
        #expect(contextMeta["omitted"] == false)
        #expect(messages.contains { $0.contains("Encodable value") })
    }

    // MARK: - Guardrail results

    @Test func testGuardrailResultsRoundTrip() async throws {
        let agent = Agent(name: "GuardrailAgent")
        let state = makeState(agent, context: emptyContext(), originalInput: "input", maxTurns: 1)
        let inputGuardrail = InputGuardrail(name: "input_guardrail") { _, _, _ in
            GuardrailFunctionOutput(outputInfo: ["input": "info"] as JSONObject, tripwireTriggered: false)
        }
        let outputGuardrail = OutputGuardrail(name: "output_guardrail") { _, _, _ in
            GuardrailFunctionOutput(outputInfo: ["output": "info"] as JSONObject, tripwireTriggered: true)
        }
        state.inputGuardrailResults = [
            InputGuardrailResult(
                guardrail: inputGuardrail,
                output: GuardrailFunctionOutput(outputInfo: ["input": "info"] as JSONObject, tripwireTriggered: false)
            ),
        ]
        state.outputGuardrailResults = [
            OutputGuardrailResult(
                guardrail: outputGuardrail,
                agentOutput: "final",
                agent: agent,
                output: GuardrailFunctionOutput(outputInfo: ["output": "info"] as JSONObject, tripwireTriggered: true)
            ),
        ]

        let restored = try await roundtripState(agent, state)

        #expect(restored.inputGuardrailResults.count == 1)
        let restoredInput = restored.inputGuardrailResults[0]
        #expect(restoredInput.guardrail.getName() == "input_guardrail")
        #expect(restoredInput.output.tripwireTriggered == false)
        #expect(restoredInput.output.outputInfo as? JSONValue == ["input": "info"])

        #expect(restored.outputGuardrailResults.count == 1)
        let restoredOutput = restored.outputGuardrailResults[0]
        #expect(restoredOutput.guardrail.getName() == "output_guardrail")
        #expect(restoredOutput.output.tripwireTriggered == true)
        #expect(restoredOutput.output.outputInfo as? JSONValue == ["output": "info"])
        #expect(restoredOutput.agentOutput as? JSONValue == "final")
        #expect(restoredOutput.agent.name == agent.name)
    }

    @Test func testGuardrailResultsToStringNormalizesNonJSONPayloads() throws {
        let agent = Agent(name: "GuardrailPayloadAgent")
        let state = makeState(agent, context: emptyContext(), originalInput: "input", maxTurns: 1)
        let observedAt = ObservedAt()
        let payload: [String: any Sendable] = ["observed_at": observedAt]
        state.inputGuardrailResults = [
            InputGuardrailResult(
                guardrail: InputGuardrail(name: "input_guardrail") { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: false) },
                output: GuardrailFunctionOutput(outputInfo: payload, tripwireTriggered: false)
            ),
        ]
        state.outputGuardrailResults = [
            OutputGuardrailResult(
                guardrail: OutputGuardrail(name: "output_guardrail") { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: false) },
                agentOutput: payload,
                agent: agent,
                output: GuardrailFunctionOutput(outputInfo: payload, tripwireTriggered: false)
            ),
        ]

        let serialized = try JSONObject.parse(try state.toString())

        let expected: JSONValue = ["observed_at": .string(observedAt.description)]
        #expect(serialized["input_guardrail_results"]?.arrayValue?[0].objectValue?["output"]?.objectValue?["outputInfo"] == expected)
        let outputResult = try #require(serialized["output_guardrail_results"]?.arrayValue?[0].objectValue)
        #expect(outputResult["output"]?.objectValue?["outputInfo"] == expected)
        #expect(outputResult["agentOutput"] == expected)
    }

    @Test func testToolGuardrailResultsRoundTrip() async throws {
        let agent = Agent(name: "ToolGuardrailAgent")
        let state = makeState(agent, context: emptyContext(), originalInput: "input", maxTurns: 1)
        state.toolInputGuardrailResults = [
            ToolInputGuardrailResult(
                guardrail: ToolInputGuardrail(name: "tool_input_guardrail") { _ in ToolGuardrailFunctionOutput(outputInfo: ["input": "info"] as JSONObject) },
                output: ToolGuardrailFunctionOutput(outputInfo: ["input": "info"] as JSONObject, behavior: .allow)
            ),
        ]
        state.toolOutputGuardrailResults = [
            ToolOutputGuardrailResult(
                guardrail: ToolOutputGuardrail(name: "tool_output_guardrail") { _ in ToolGuardrailFunctionOutput(outputInfo: ["output": "info"] as JSONObject) },
                output: ToolGuardrailFunctionOutput(outputInfo: ["output": "info"] as JSONObject, behavior: .allow)
            ),
        ]

        let restored = try await roundtripState(agent, state)

        #expect(restored.toolInputGuardrailResults.count == 1)
        let restoredToolInput = restored.toolInputGuardrailResults[0]
        #expect(restoredToolInput.guardrail.getName() == "tool_input_guardrail")
        #expect(restoredToolInput.output.behavior.type == "allow")
        #expect(restoredToolInput.output.outputInfo as? JSONValue == ["input": "info"])

        #expect(restored.toolOutputGuardrailResults.count == 1)
        let restoredToolOutput = restored.toolOutputGuardrailResults[0]
        #expect(restoredToolOutput.guardrail.getName() == "tool_output_guardrail")
        #expect(restoredToolOutput.output.behavior.type == "allow")
        #expect(restoredToolOutput.output.outputInfo as? JSONValue == ["output": "info"])
    }

    @Test func testToolGuardrailResultsToStringNormalizesNonJSONOutputInfo() throws {
        let agent = Agent(name: "ToolGuardrailPayloadAgent")
        let state = makeState(agent, context: emptyContext(), originalInput: "input", maxTurns: 1)
        let observedAt = ObservedAt()
        let payload: [String: any Sendable] = ["observed_at": observedAt]
        state.toolInputGuardrailResults = [
            ToolInputGuardrailResult(
                guardrail: ToolInputGuardrail(name: "tool_input_guardrail") { _ in ToolGuardrailFunctionOutput() },
                output: ToolGuardrailFunctionOutput(outputInfo: payload, behavior: .allow)
            ),
        ]
        state.toolOutputGuardrailResults = [
            ToolOutputGuardrailResult(
                guardrail: ToolOutputGuardrail(name: "tool_output_guardrail") { _ in ToolGuardrailFunctionOutput() },
                output: ToolGuardrailFunctionOutput(outputInfo: payload, behavior: .allow)
            ),
        ]

        let serialized = try JSONObject.parse(try state.toString())

        let expected: JSONValue = ["observed_at": .string(observedAt.description)]
        #expect(serialized["tool_input_guardrail_results"]?.arrayValue?[0].objectValue?["output"]?.objectValue?["outputInfo"] == expected)
        #expect(serialized["tool_output_guardrail_results"]?.arrayValue?[0].objectValue?["output"]?.objectValue?["outputInfo"] == expected)
    }

    // MARK: - Rejections

    @Test func testRejectPermanentlyWhenAlwaysRejectOptionIsPassed() throws {
        let agent = Agent(name: "Agent4")
        let state = makeState(agent, context: emptyContext(), originalInput: "", maxTurns: 1)
        let approvalItem = makeToolApprovalItem(agent, callId: "cid789", name: "toolZ", arguments: "arguments")

        try state.reject(approvalItem, alwaysReject: true)

        #expect(state.context.isToolApproved(toolName: "toolZ", callId: "cid789") == false)
        #expect(state.context.isToolApproved(toolName: "toolZ", callId: "future-call") == false)
    }

    @Test func testRejectionIsScopedToCallIds() throws {
        let agent = Agent(name: "AgentRejectReuse")
        let state = makeState(agent, context: emptyContext(), originalInput: "", maxTurns: 1)
        let approvalItem = makeToolApprovalItem(agent, callId: "cid789", name: "toolZ", arguments: "arguments")

        try state.reject(approvalItem)

        #expect(state.context.isToolApproved(toolName: "toolZ", callId: "cid789") == false)
        #expect(state.context.isToolApproved(toolName: "toolZ", callId: "cid999") == nil)
        #expect(state.context.getRejectionMessage(toolName: "toolZ", callId: "cid999") == nil)
    }

    @Test func testAlwaysRejectReusesRejectionMessageForFutureCalls() throws {
        let agent = Agent(name: "AgentStickyReject")
        let state = makeState(agent, context: emptyContext(), originalInput: "", maxTurns: 1)
        let approvalItem = makeToolApprovalItem(agent, callId: "cid789", name: "toolZ", arguments: "arguments")

        try state.reject(approvalItem, alwaysReject: true, rejectionMessage: "")

        #expect(state.context.getRejectionMessage(toolName: "toolZ", callId: "cid789") == "")
        #expect(state.context.getRejectionMessage(toolName: "toolZ", callId: "cid999") == "")
    }

    // Swift: test_approve_raises_when_context_is_none and test_reject_raises_when_context_is_none
    // are not ported; `RunState.context` is non-optional, so a state without a context cannot exist.

    // MARK: - Generated item merging

    @Test func testGeneratedItemsNotDuplicatedByLastProcessedResponse() async throws {
        let agent = Agent(name: "AgentDedup")
        let state = makeState(agent, context: emptyContext(), originalInput: "input", maxTurns: 2)
        let toolCallItem = ToolCallItem(agent: agent, rawItem: getFunctionToolCall("get_weather", callId: "call_1"))

        // Simulate a turn that produced a tool call and also stored it in lastProcessedResponse.
        state.generatedItems = [toolCallItem]
        state.lastProcessedResponse = makeProcessedResponse(newItems: [toolCallItem])

        let jsonData = try state.toJSON()
        let generatedItemsJSON = try #require(jsonData["generated_items"]?.arrayValue)
        #expect(generatedItemsJSON.count == 1)
        #expect(generatedItemsJSON[0].objectValue?["raw_item"]?.objectValue?["call_id"] == "call_1")

        let restored = try await RunState.fromJSON(agent, jsonData)
        #expect(restored.generatedItems.count == 1)
        #expect(restored.generatedItems[0].rawItem["call_id"] == "call_1")
    }

    private func toolSearchItems(_ agent: Agent) -> (call: JSONObject, output: JSONObject) {
        (
            ["type": "tool_search_call", "arguments": ["query": "account balance"], "execution": "server", "status": "completed"],
            ["type": "tool_search_output", "execution": "server", "status": "completed", "tools": []]
        )
    }

    @Test func testAnonymousToolSearchItemsKeepLaterSameContentSnapshot() throws {
        let agent = Agent(name: "AgentToolSearchMerge")
        let state = makeState(agent, context: emptyContext(), originalInput: "input", maxTurns: 2)
        let raw = toolSearchItems(agent)
        state.generatedItems = [ToolSearchCallItem(agent: agent, rawItem: raw.call), ToolSearchOutputItem(agent: agent, rawItem: raw.output)]
        state.lastProcessedResponse = makeProcessedResponse(newItems: [
            ToolSearchCallItem(agent: agent, rawItem: raw.call),
            ToolSearchOutputItem(agent: agent, rawItem: raw.output),
        ])

        let jsonData = try state.toJSON()
        #expect(jsonData["generated_items"]?.arrayValue?.compactMap { $0.objectValue?.string("type") } == [
            "tool_search_call_item", "tool_search_output_item", "tool_search_call_item", "tool_search_output_item",
        ])
    }

    @Test func testAnonymousToolSearchItemsNotDuplicatedAcrossRoundTrip() async throws {
        let agent = Agent(name: "AgentToolSearchDedup")
        let state = makeState(agent, context: emptyContext(), originalInput: "input", maxTurns: 2)
        let raw = toolSearchItems(agent)
        state.generatedItems = [
            ToolSearchCallItem(agent: agent, rawItem: raw.call),
            ToolSearchOutputItem(agent: agent, rawItem: raw.output),
            ToolSearchCallItem(agent: agent, rawItem: raw.call),
            ToolSearchOutputItem(agent: agent, rawItem: raw.output),
        ]
        state.lastProcessedResponse = makeProcessedResponse(newItems: [
            ToolSearchCallItem(agent: agent, rawItem: raw.call),
            ToolSearchOutputItem(agent: agent, rawItem: raw.output),
        ])
        state.markGeneratedItemsMergedWithLastProcessed()

        let expectedTypes = ["tool_search_call_item", "tool_search_output_item", "tool_search_call_item", "tool_search_output_item"]
        let jsonData = try state.toJSON()
        #expect(jsonData["generated_items"]?.arrayValue?.compactMap { $0.objectValue?.string("type") } == expectedTypes)

        let restored = try await RunState.fromJSON(agent, jsonData)
        let restoredJSON = try restored.toJSON()
        #expect(restoredJSON["generated_items"]?.arrayValue?.compactMap { $0.objectValue?.string("type") } == expectedTypes)
    }

    @Test func testToJSONDeduplicatesItemsWithDirectIdTypeAttributes() throws {
        // Swift: run items expose their type but no separate `id` attribute, so the ids live in the
        // raw items while the type comes from the item itself.
        final class MockItem: RunItemImpl, @unchecked Sendable {
            let itemType: String
            init(agent: Agent, id: String, type: String) {
                itemType = type
                super.init(agent: agent, rawItem: ["id": .string(id), "content": "test"])
            }
            override var type: String { itemType }
        }
        let agent = Agent(name: "TestAgent")
        let state = makeState(agent, context: emptyContext(), originalInput: "input", maxTurns: 2)
        state.generatedItems = [MockItem(agent: agent, id: "item_123", type: "message_output_item")]
        state.lastProcessedResponse = makeProcessedResponse(newItems: [
            MockItem(agent: agent, id: "item_123", type: "message_output_item"),
            MockItem(agent: agent, id: "item_456", type: "tool_call_item"),
        ])

        let jsonData = try state.toJSON()
        #expect(jsonData["generated_items"]?.arrayValue?.count == 2)
    }

    // MARK: - Reconstruction

    @Test func testFromStringReconstructsStateForSimpleAgent() async throws {
        let agent = Agent(name: "Solo")
        let state = makeState(agent, context: RunContextWrapper(context: ["a": 1] as JSONObject), originalInput: "orig", maxTurns: 7)
        state.currentTurn = 5

        let newState = try await RunState.fromString(agent, try state.toString())

        #expect(newState.maxTurns == 7)
        #expect(newState.currentTurn == 5)
        #expect(newState.currentAgent === agent)
        #expect(newState.context.context(as: JSONObject.self) == ["a": 1])
        #expect(newState.generatedItems.isEmpty)
        #expect(newState.modelResponses.isEmpty)
    }

    @Test func testFromJSONReconstructsState() async throws {
        let agent = Agent(name: "JsonAgent")
        let state = makeState(agent, context: RunContextWrapper(context: ["test": "data"] as JSONObject), originalInput: "test input", maxTurns: 5)
        state.currentTurn = 2

        let newState = try await RunState.fromJSON(agent, try state.toJSON())

        #expect(newState.maxTurns == 5)
        #expect(newState.currentTurn == 2)
        #expect(newState.currentAgent === agent)
        #expect(newState.context.context(as: JSONObject.self) == ["test": "data"])
    }

    // MARK: - Interruptions

    @Test func testGetInterruptionsReturnsEmptyWhenNoInterruptions() {
        let state = makeState(Agent(name: "Agent5"), context: emptyContext(), originalInput: "", maxTurns: 1)
        #expect(state.getInterruptions().isEmpty)
    }

    @Test func testGetInterruptionsReturnsInterruptionsWhenPresent() {
        let agent = Agent(name: "Agent6")
        let rawItem = ResponseFunctionToolCall(callId: "cid111", name: "toolA", arguments: "args", status: "completed").json
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: rawItem)
        let state = makeStateWithInterruptions(agent, [approvalItem], originalInput: "", maxTurns: 1)

        let interruptions = state.getInterruptions()
        #expect(interruptions.count == 1)
        #expect(interruptions[0] !== approvalItem)
        #expect(interruptions[0].agent === agent)
        #expect(interruptions[0].toolName == approvalItem.toolName)
        #expect(interruptions[0].rawItem == approvalItem.rawItem)
    }

    @Test(arguments: ["pydantic", "mapping"])
    func testGetInterruptionsReturnsDetachedItemSnapshots(rawItemKind: String) {
        // Swift: raw items are `JSONObject` values for both kinds, so copies are always detached.
        let agent = Agent(name: "SnapshotAgent")
        var rawItem: JSONObject = [
            "type": "function_call", "name": "toolA", "call_id": "cid-snapshot", "status": "completed",
            "arguments": #"{"value": "original"}"#,
        ]
        if rawItemKind == "mapping" { rawItem["metadata"] = ["tags": ["original"]] }
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: rawItem)
        let state = makeStateWithInterruptions(agent, [approvalItem])

        let interruption = state.getInterruptions()[0]
        interruption.toolName = "changed"
        interruption.rawItem["arguments"] = #"{"value": "changed"}"#
        if rawItemKind == "mapping" { interruption.rawItem["metadata"] = ["tags": ["original", "changed"]] }

        let pending = state.getInterruptions()[0]
        #expect(pending !== interruption)
        #expect(pending.agent === agent)
        #expect(pending.toolName == "toolA")
        #expect(pending.rawItem["arguments"] == #"{"value": "original"}"#)
        if rawItemKind == "mapping" { #expect(pending.rawItem["metadata"] == ["tags": ["original"]]) }
    }

    @Test(arguments: ["pydantic", "mapping"], [true, false])
    func testGetInterruptionsSnapshotsCanApplyApprovalDecisions(rawItemKind: String, approve: Bool) throws {
        let agent = Agent(name: "DecisionAgent")
        let rawItem: JSONObject = ["type": "function_call", "name": "toolA", "call_id": "cid-decision", "status": "completed", "arguments": "{}"]
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: rawItem)
        let state = makeStateWithInterruptions(agent, [approvalItem])

        let interruption = state.getInterruptions()[0]
        #expect(interruption !== approvalItem)
        if approve {
            try state.approve(interruption)
        } else {
            try state.reject(interruption)
        }

        #expect(state.context.isToolApproved(toolName: "toolA", callId: "cid-decision") == approve)
    }

    // Swift: the remaining test_get_interruptions_* cases exercise copying arbitrary Python objects
    // and pydantic models (deepcopy hooks, typed extras, descriptors, cyclic mappings). Swift raw
    // items are `JSONObject` values, which copy by value and cannot hold such objects.
}
