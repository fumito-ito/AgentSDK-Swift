import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_state_pending_input.py.
//
// Swift: the server-acceptance cases (`test_server_response_acceptance_commits_before_hook_failure`,
// `test_server_acceptance_commits_before_invocation_validation_failure`,
// `test_server_accepted_computer_start_hook_failure_is_not_replayed`, and
// `test_server_accepted_tool_side_effect_failure_is_safe`) need the accepted-response resume state
// and the tool invocation ledger, which are not ported yet (porting/manifest.json).

private func itemType(_ item: TResponseInputItem) -> String? {
    item.string("type") ?? item.string("role")
}

private func messageText(_ item: TResponseInputItem) -> String? {
    guard item.string("role") == "user" else { return nil }
    switch item["content"] {
    case .string(let text)?:
        return text
    case .array(let parts)?:
        return parts.compactMap(\.objectValue)
            .filter { ["input_text", "output_text"].contains($0.string("type") ?? "") }
            .map { $0.string("text") ?? "" }
            .joined()
    default:
        return nil
    }
}

private struct DestinationArgs: Decodable, Sendable { var destination: String }
private struct ValueArgs: Decodable, Sendable { var value: String }

/// Runs one streamed turn that calls a tool, cancels after the turn, and returns the resumable state.
private func makeAfterTurnState(
    session: SimpleListSession? = nil,
    autoPreviousResponseId: Bool = false
) async throws -> (ScriptedModel, Agent, RunState, SharedBox<[String]>) {
    let calls = SharedBox<[String]>([])
    let recordDestination = try FunctionTool.create(name: "record_destination", argumentsType: DestinationArgs.self) { _, args in
        calls.mutate { $0.append(args.destination) }
        return "recorded:\(args.destination)"
    }
    let model = try ScriptedModel()
    try model.enqueue(.output([getFunctionToolCall("record_destination", arguments: #"{"destination": "Paris"}"#, callId: "call-destination")]))
    let agent = Agent(name: "assistant", tools: [recordDestination], model: .instance(model))
    let streamed = try Runner.runStreamed(agent, input: "Initial request", autoPreviousResponseId: autoPreviousResponseId, session: session)
    for try await event in streamed.streamEvents() {
        if case .runItem(let itemEvent) = event, itemEvent.name == "tool_output" {
            streamed.cancel(mode: .afterTurn)
        }
    }
    let state = streamed.toState()
    guard case .runAgain? = state.currentStep else {
        Issue.record("expected a run-again step")
        return (model, agent, state, calls)
    }
    #expect(calls.value == ["Paris"])
    return (model, agent, state, calls)
}

private func protectedTool(calls: SharedBox<[String]>) throws -> FunctionTool {
    try FunctionTool.create(name: "protected_tool", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
        calls.mutate { $0.append(args.value) }
        return "approved:\(args.value)"
    }
}

@Suite struct RunStatePendingInputTests {
    @Test func testPendingInputPreservesOrderAndSerializationRoundTrips() async throws {
        let agent = Agent(name: "assistant")
        let state = RunState(context: RunContextWrapper(context: JSONObject()), originalInput: "Initial request", startingAgent: agent)
        state.currentStep = .runAgain
        let startingTurn = state.currentTurn

        try state.addInput("First late message")
        try state.addInput(.items([["role": "user", "content": "Second late message"]]))
        #expect(state.currentTurn == startingTurn)
        #expect(state.pendingInput.map(messageText) == ["First late message", "Second late message"])

        // Swift: `pendingInput` returns values, so a caller's copy is always detached.
        var detachedView = state.pendingInput
        detachedView[0]["content"] = "mutated"
        #expect(state.pendingInput.first.flatMap(messageText) == "First late message")

        let serialized = try state.toJSON()
        #expect(serialized["$schemaVersion"] == .string(RunState.currentSchemaVersion))
        let restored = try await RunState.fromJSON(agent, serialized)
        let restoredFromString = try await RunState.fromString(agent, try state.toString())
        for candidate in [restored, restoredFromString] {
            guard case .runAgain? = candidate.currentStep else {
                Issue.record("expected a run-again step")
                continue
            }
            #expect(candidate.pendingInput.map(messageText) == ["First late message", "Second late message"])
        }

        var legacy = try state.toJSON()
        legacy["$schemaVersion"] = "1.14"
        legacy.removeValue(forKey: "pending_input")
        legacy["current_step"] = .null
        let restoredLegacy = try await RunState.fromJSON(agent, legacy)
        #expect(restoredLegacy.pendingInput.isEmpty)
    }

    @Test func testAfterTurnResumeAdmitsInputAfterToolOutputExactlyOnce() async throws {
        let session = SimpleListSession()
        let (model, agent, state, calls) = try await makeAfterTurnState(session: session)
        try state.addInput("Change the destination to Tokyo")
        try model.enqueue(.output([getTextMessage("Updated")]))

        let result = try await Runner.run(agent, state: state, session: session)

        #expect(result.finalOutput as? String == "Updated")
        #expect(calls.value == ["Paris"])
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.map(itemType) == ["user", "function_call", "function_call_output", "user"])
        #expect(modelInput.filter { messageText($0) == "Change the destination to Tokyo" }.count == 1)
        #expect(state.pendingInput.isEmpty)

        let sessionItems = try await session.getItems(limit: nil)
        #expect(sessionItems.filter { messageText($0) == "Change the destination to Tokyo" }.count == 1)
        let replayItems = try result.toInputList()
        #expect(replayItems.filter { messageText($0) == "Change the destination to Tokyo" }.count == 1)
        for terminalState in [state, result.toState()] {
            expectError(UserError.self, containing: "terminal RunState") {
                try terminalState.addInput("Too late")
            }
        }
    }

    @Test func testStreamedResumeMatchesPendingInputOrdering() async throws {
        let (model, agent, state, calls) = try await makeAfterTurnState()
        try state.addInput("Change the destination to Tokyo")
        try model.enqueue(.output([getTextMessage("Updated")]))

        let result = try Runner.runStreamed(agent, state: state)
        for try await _ in result.streamEvents() {}

        #expect(result.finalOutput as? String == "Updated")
        #expect(calls.value == ["Paris"])
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.map(itemType) == ["user", "function_call", "function_call_output", "user"])
        #expect(modelInput.filter { messageText($0) == "Change the destination to Tokyo" }.count == 1)
        #expect(state.pendingInput.isEmpty)
        for terminalState in [state, result.toState()] {
            expectError(UserError.self, containing: "terminal RunState") {
                try terminalState.addInput("Too late")
            }
        }
    }

    @Test func testServerManagedResumeSendsPendingInputAsUnsentDeltaOnce() async throws {
        let (model, agent, state, calls) = try await makeAfterTurnState(autoPreviousResponseId: true)
        try state.addInput("Change the destination to Tokyo")
        try model.enqueue(.output([getTextMessage("Updated")]))

        let result = try await Runner.run(agent, state: state)

        #expect(result.finalOutput as? String == "Updated")
        #expect(calls.value == ["Paris"])
        #expect(model.calls.last?.previousResponseId == "resp-789")
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.map(itemType) == ["function_call_output", "user"])
        #expect(modelInput.filter { messageText($0) == "Change the destination to Tokyo" }.count == 1)
        #expect(state.pendingInput.isEmpty)
    }

    @Test func testServerTrackerDistinguishesIdenticalInputOccurrencesAfterRestore() {
        let agent = Agent(name: "assistant")
        let admittedFirst = InputItem(agent: agent, rawItem: ["role": "user", "content": "Repeat"])
        let admittedSecond = InputItem(agent: agent, rawItem: ["role": "user", "content": "Repeat"])
        let tracker = OpenAIServerConversationTracker(previousResponseId: "resp-latest")
        tracker.hydrateFromState(
            originalInput: "Initial request",
            generatedItems: [admittedFirst],
            modelResponses: [ModelResponse(output: [], usage: Usage(), responseId: "resp-latest")]
        )

        #expect(tracker.prepareInput("Initial request", [admittedFirst, admittedSecond]) == [admittedSecond.rawItem])
    }

    @Test(arguments: [false, true])
    func testServerManagedResumeSendsIdenticalLateInputInLaterOccurrence(streamedSecondResume: Bool) async throws {
        let (model, agent, initialState, calls) = try await makeAfterTurnState(autoPreviousResponseId: true)
        try initialState.addInput("Repeat")
        try model.enqueue(.output([getFunctionToolCall("record_destination", arguments: #"{"destination": "Rome"}"#, callId: "call-second-destination")]))

        let firstResume = try Runner.runStreamed(agent, state: initialState)
        for try await event in firstResume.streamEvents() {
            if case .runItem(let itemEvent) = event, itemEvent.name == "tool_output" {
                firstResume.cancel(mode: .afterTurn)
            }
        }

        let state = try await RunState.fromJSON(agent, try firstResume.toState().toJSON())
        let admittedBefore = try #require(state.generatedItems.compactMap { $0 as? InputItem }.first)
        try state.addInput("Repeat")
        try model.enqueue(.output([getTextMessage("Done")]))

        let finalOutput: (any Sendable)?
        if streamedSecondResume {
            let streamedResult = try Runner.runStreamed(agent, state: state)
            for try await _ in streamedResult.streamEvents() {}
            finalOutput = streamedResult.finalOutput
        } else {
            finalOutput = try await Runner.run(agent, state: state).finalOutput
        }

        #expect(finalOutput as? String == "Done")
        #expect(calls.value == ["Paris", "Rome"])
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.filter { messageText($0) == "Repeat" }.count == 1)
        let admittedAfter = state.generatedItems.compactMap { $0 as? InputItem }
        #expect(admittedAfter.filter { $0.inputId == admittedBefore.inputId }.count == 1)
        #expect(Set(admittedAfter.map(\.inputId)).count == 2)
    }

    @Test func testUnresolvedApprovalKeepsPendingInputUntilToolFinishes() async throws {
        let calls = SharedBox<[String]>([])
        let model = try ScriptedModel()
        try model.enqueue(.output([getFunctionToolCall("protected_tool", arguments: #"{"value":"one"}"#, callId: "call-protected")]))
        let agent = Agent(name: "assistant", tools: [try protectedTool(calls: calls)], model: .instance(model))
        let interrupted = try await Runner.run(agent, input: "Initial request")
        let state = interrupted.toState()
        try state.addInput("Late input")

        let stillInterrupted = try await Runner.run(agent, state: state)
        #expect(!stillInterrupted.interruptions.isEmpty)
        #expect(calls.value.isEmpty)
        #expect(state.pendingInput.first.flatMap(messageText) == "Late input")

        try state.approve(try #require(state.getInterruptions().first))
        try model.enqueue(.output([getTextMessage("Done")]))
        let resumed = try await Runner.run(agent, state: state)

        #expect(resumed.finalOutput as? String == "Done")
        #expect(calls.value == ["one"])
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.suffix(2).map(itemType) == ["function_call_output", "user"])
        #expect(modelInput.last.flatMap(messageText) == "Late input")
    }

    @Test func testStreamedAfterTurnCancelKeepsPendingInputForNextResume() async throws {
        let calls = SharedBox<[String]>([])
        let model = try ScriptedModel()
        try model.enqueue(.output([getFunctionToolCall("protected_tool", arguments: #"{"value":"one"}"#, callId: "call-protected")]))
        let agent = Agent(name: "assistant", tools: [try protectedTool(calls: calls)], model: .instance(model))
        let interrupted = try await Runner.run(agent, input: "Initial request")
        let state = interrupted.toState()
        try state.addInput("Late input")
        try state.approve(try #require(state.getInterruptions().first))

        let resumed = try Runner.runStreamed(agent, state: state)
        for try await event in resumed.streamEvents() {
            if case .runItem(let itemEvent) = event, itemEvent.name == "tool_output" {
                resumed.cancel(mode: .afterTurn)
            }
        }

        #expect(calls.value == ["one"])
        #expect(state.pendingInput.first.flatMap(messageText) == "Late input")

        try model.enqueue(.output([getTextMessage("Done")]))
        let result = try await Runner.run(agent, state: state)
        #expect(result.finalOutput as? String == "Done")
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.filter { messageText($0) == "Late input" }.count == 1)
    }

    enum TerminalBehavior: String, CaseIterable, Sendable {
        case stopOnFirstTool = "stop_on_first_tool"
        case stopAtTools = "stop_at_tool_names"
        case custom
    }

    @Test(arguments: [false, true], TerminalBehavior.allCases)
    func testInterruptionWithoutGuaranteedNextModelRejectsInput(streamed: Bool, behavior: TerminalBehavior) async throws {
        let tool = try FunctionTool.create(name: "protected_tool", argumentsType: ValueArgs.self, needsApproval: true) { _, args in args.value }
        let model = try ScriptedModel(steps: [.output([getFunctionToolCall("protected_tool", arguments: #"{"value":"one"}"#, callId: "call-protected-terminal")])])
        let toolUseBehavior: ToolUseBehavior
        switch behavior {
        case .stopOnFirstTool: toolUseBehavior = .stopOnFirstTool
        case .stopAtTools: toolUseBehavior = .stopAtTools(["protected_tool"])
        case .custom: toolUseBehavior = .custom { _, _ in ToolsToFinalOutputResult(isFinalOutput: false) }
        }
        let agent = Agent(name: "assistant", tools: [tool], model: .instance(model), toolUseBehavior: toolUseBehavior)
        let state: RunState
        if streamed {
            let interrupted = try Runner.runStreamed(agent, input: "Initial request")
            for try await _ in interrupted.streamEvents() {}
            state = interrupted.toState()
        } else {
            state = try await Runner.run(agent, input: "Initial request").toState()
        }

        let before = try state.toJSON()
        expectError(UserError.self, containing: "tool result may end the run") {
            try state.addInput("Late input")
        }
        #expect(try state.toJSON() == before)
    }

    // MARK: - Guardrails

    @Test func testPendingInputGuardrailTripKeepsInputRecoverable() async throws {
        let (model, agent, state, _) = try await makeAfterTurnState()
        let guardedInputs = SharedBox<[[TResponseInputItem]]>([])
        let tripPendingInput = InputGuardrail { _, _, input in
            guardedInputs.mutate { $0.append(input.inputItems) }
            return GuardrailFunctionOutput(tripwireTriggered: true)
        }
        agent.inputGuardrails = [tripPendingInput]
        try state.addInput("Unsafe late input")
        try model.enqueue(.output([getTextMessage("Must not run")]))
        let queuedOutputs = model.remainingSteps

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            _ = try await Runner.run(agent, state: state)
        }

        #expect(model.remainingSteps == queuedOutputs)
        #expect(guardedInputs.value.map { $0.map(messageText) } == [["Unsafe late input"]])
        #expect(state.pendingInput.first.flatMap(messageText) == "Unsafe late input")
        state.clearPendingInput()
        #expect(state.pendingInput.isEmpty)
    }

    @Test func testPendingInputRunsAgentAndRunConfigGuardrailsOnOnlyPending() async throws {
        let (model, agent, state, _) = try await makeAfterTurnState()
        let guardedInputs = SharedBox<[(String, [TResponseInputItem])]>([])
        let inspectAgentInput = InputGuardrail { _, _, input in
            guardedInputs.mutate { $0.append(("agent", input.inputItems)) }
            return GuardrailFunctionOutput(tripwireTriggered: false)
        }
        let inspectConfigInput = InputGuardrail { _, _, input in
            guardedInputs.mutate { $0.append(("config", input.inputItems)) }
            return GuardrailFunctionOutput(tripwireTriggered: false)
        }
        agent.inputGuardrails = [inspectAgentInput]
        let runConfig = RunConfig(inputGuardrails: [inspectConfigInput])
        try state.addInput("Guard only this")
        try model.enqueue(.output([getTextMessage("Done")]))

        let result = try await Runner.run(agent, state: state, runConfig: runConfig)

        #expect(result.finalOutput as? String == "Done")
        #expect(Set(guardedInputs.value.map(\.0)) == ["agent", "config"])
        #expect(guardedInputs.value.map { $0.1.map(messageText) } == [["Guard only this"], ["Guard only this"]])
    }

    @Test(arguments: [false, true])
    func testGuardrailRetryPersistsSuccessfulTurnWithSession(streamedRetry: Bool) async throws {
        let session = SimpleListSession()
        let (model, agent, state, _) = try await makeAfterTurnState(session: session)
        let shouldTrip = SharedBox(true)
        agent.inputGuardrails = [InputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: shouldTrip.value) }]
        try state.addInput("Late input")

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            if streamedRetry {
                let tripped = try Runner.runStreamed(agent, state: state, session: session)
                for try await _ in tripped.streamEvents() {}
            } else {
                _ = try await Runner.run(agent, state: state, session: session)
            }
        }

        shouldTrip.value = false
        try model.enqueue(.output([getTextMessage("Recovered")]))
        let finalOutput: (any Sendable)?
        if streamedRetry {
            let streamedResult = try Runner.runStreamed(agent, state: state, session: session)
            for try await _ in streamedResult.streamEvents() {}
            finalOutput = streamedResult.finalOutput
        } else {
            finalOutput = try await Runner.run(agent, state: state, session: session).finalOutput
        }

        #expect(finalOutput as? String == "Recovered")
        let sessionItems = try await session.getItems(limit: nil)
        #expect(sessionItems.filter { messageText($0) == "Late input" }.count == 1)
        #expect(sessionItems.last.flatMap(itemType) == "message")
        #expect(sessionItems.last?.string("role") == "assistant")
        #expect(state.inputGuardrailResults.map(\.output.tripwireTriggered) == [true, false])
    }

    // MARK: - Failed model requests

    @Test func testFailedModelRequestDoesNotDuplicateAdmittedInputOnResume() async throws {
        let (model, agent, initialState, _) = try await makeAfterTurnState()
        try initialState.addInput("Late input")
        try model.enqueue(.raiseError(TestError("model failed")))

        await expectErrorAsync(TestError.self, containing: "model failed") {
            _ = try await Runner.run(agent, state: initialState)
        }

        #expect(initialState.pendingInput.isEmpty)
        let admittedItems = initialState.generatedItems.compactMap { $0 as? InputItem }
        #expect(admittedItems.map { messageText($0.rawItem) } == ["Late input"])
        let admittedInputId = try #require(admittedItems.first?.inputId)

        let state = try await RunState.fromJSON(agent, try initialState.toJSON())
        #expect(state.generatedItems.compactMap { $0 as? InputItem }.first?.inputId == admittedInputId)
        try model.enqueue(.output([getTextMessage("Recovered")]))
        let result = try await Runner.run(agent, state: state)
        #expect(result.finalOutput as? String == "Recovered")
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.filter { messageText($0) == "Late input" }.count == 1)
    }

    @Test func testFailedModelRequestWithSessionPersistsAdmittedInputOnce() async throws {
        let session = SimpleListSession()
        let (model, agent, initialState, _) = try await makeAfterTurnState(session: session)
        try initialState.addInput("Late input")
        try model.enqueue(.raiseError(TestError("model failed")))

        await expectErrorAsync(TestError.self, containing: "model failed") {
            _ = try await Runner.run(agent, state: initialState, session: session)
        }

        #expect(initialState.pendingInput.isEmpty)
        #expect(try await session.getItems(limit: nil).filter { messageText($0) == "Late input" }.count == 1)

        let state = try await RunState.fromJSON(agent, try initialState.toJSON())
        try model.enqueue(.output([getTextMessage("Recovered")]))
        let result = try await Runner.run(agent, state: state, session: session)
        #expect(result.finalOutput as? String == "Recovered")
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.filter { messageText($0) == "Late input" }.count == 1)
        #expect(try await session.getItems(limit: nil).filter { messageText($0) == "Late input" }.count == 1)
    }

    @Test func testFailedServerManagedRequestKeepsPendingInputForRetry() async throws {
        let (model, agent, initialState, _) = try await makeAfterTurnState(autoPreviousResponseId: true)
        try initialState.addInput("Late input")
        try model.enqueue(.raiseError(TestError("model failed")))

        await expectErrorAsync(TestError.self, containing: "model failed") {
            _ = try await Runner.run(agent, state: initialState)
        }

        #expect(initialState.pendingInput.first.flatMap(messageText) == "Late input")
        let state = try await RunState.fromJSON(agent, try initialState.toJSON())
        try model.enqueue(.output([getTextMessage("Recovered")]))
        let result = try await Runner.run(agent, state: state)
        #expect(result.finalOutput as? String == "Recovered")
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.filter { messageText($0) == "Late input" }.count == 1)
        #expect(state.pendingInput.isEmpty)
    }

    // MARK: - Model input filters

    @Test func testServerFilterOmissionRemainsPendingForLaterNonstreamTurn() async throws {
        let (model, agent, state, calls) = try await makeAfterTurnState(autoPreviousResponseId: true)
        try state.addInput("Late input")
        try model.extend([
            .output([getFunctionToolCall("record_destination", arguments: #"{"destination": "Rome"}"#, callId: "call-filtered-destination")]),
            .output([getTextMessage("Done")]),
        ])
        let filterCalls = SharedBox(0)
        let omitFirstRequest: CallModelInputFilter = { data in
            let call = filterCalls.mutate { count -> Int in
                count += 1
                return count
            }
            return ModelInputData(input: call == 1 ? [] : data.modelData.input, instructions: data.modelData.instructions)
        }

        let result = try await Runner.run(agent, state: state, runConfig: RunConfig(callModelInputFilter: omitFirstRequest))

        #expect(result.finalOutput as? String == "Done")
        #expect(calls.value == ["Paris", "Rome"])
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.filter { messageText($0) == "Late input" }.count == 1)
        #expect(state.pendingInput.isEmpty)
    }

    @Test func testServerFilterOmissionSurvivesStreamedStateRoundTrip() async throws {
        let (model, agent, initialState, calls) = try await makeAfterTurnState(autoPreviousResponseId: true)
        try initialState.addInput("Late input")
        try model.enqueue(.output([getFunctionToolCall("record_destination", arguments: #"{"destination": "Rome"}"#, callId: "call-filtered-destination")]))
        let omitPending: CallModelInputFilter = { data in ModelInputData(input: [], instructions: data.modelData.instructions) }

        let filtered = try Runner.runStreamed(agent, state: initialState, runConfig: RunConfig(callModelInputFilter: omitPending))
        for try await event in filtered.streamEvents() {
            if case .runItem(let itemEvent) = event, itemEvent.name == "tool_output" {
                filtered.cancel(mode: .afterTurn)
            }
        }

        let state = try await RunState.fromJSON(agent, try filtered.toState().toJSON())
        #expect(state.pendingInput.map(messageText) == ["Late input"])
        #expect(!state.generatedItems.contains { $0 is InputItem })

        try model.enqueue(.output([getTextMessage("Done")]))
        let result = try await Runner.run(agent, state: state)
        #expect(result.finalOutput as? String == "Done")
        #expect(calls.value == ["Paris", "Rome"])
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.filter { messageText($0) == "Late input" }.count == 1)
    }

    @Test(arguments: [false, true])
    func testServerFilterReconstructedPendingRewriteIsRejected(streamed: Bool) async throws {
        let (model, agent, state, _) = try await makeAfterTurnState(autoPreviousResponseId: true)
        try state.addInput("Late input")
        try model.enqueue(.output([getTextMessage("Done")]))
        let reconstructPending: CallModelInputFilter = { data in
            let rewritten = data.modelData.input.map { item -> TResponseInputItem in
                messageText(item) == "Late input" ? ["role": "user", "content": "Filtered late input"] : item
            }
            return ModelInputData(input: rewritten, instructions: data.modelData.instructions)
        }
        let queuedOutputs = model.remainingSteps
        let runConfig = RunConfig(callModelInputFilter: reconstructPending)

        await expectErrorAsync(UserError.self, containing: "cannot safely associate") {
            if streamed {
                let failed = try Runner.runStreamed(agent, state: state, runConfig: runConfig)
                for try await _ in failed.streamEvents() {}
            } else {
                _ = try await Runner.run(agent, state: state, runConfig: runConfig)
            }
        }

        #expect(model.remainingSteps == queuedOutputs)
        #expect(state.pendingInput.map(messageText) == ["Late input"])
    }

    // Swift: test_server_filter_in_place_pending_rewrite_preserves_occurrence is not ported. Input
    // items are values, so an in-place rewrite is indistinguishable from a reconstructed item and
    // is rejected like one (porting/decisions.md D11).

    @Test func testTerminalStateRejectsPendingInputWithoutMutation() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("Done")])])
        let agent = Agent(name: "assistant", model: .instance(model))
        let state = try await Runner.run(agent, input: "Initial request").toState()
        let before = try state.toJSON()

        expectError(UserError.self, containing: "terminal RunState") {
            try state.addInput("Too late")
        }
        #expect(try state.toJSON() == before)
    }
}
