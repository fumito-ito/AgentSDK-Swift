import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_max_turns.py.
//
// Swift: `SQLiteSession` cases use `SimpleListSession` until the SQLite session is ported. Tests
// that inspect `RunState.to_json()` or resume serialized states are pending the RunState
// serialization port.
@Suite struct MaxTurnsTests {
    private struct Foo: Codable, Sendable {
        var a: String
    }

    private struct FooModel: Codable, Sendable, Equatable {
        var summary: String
    }

    private func someFunctionTurns(_ count: Int, final: String? = nil) -> [ModelStep] {
        let funcOutput = JSONValue.object(["a": "b"]).pythonJSONString()
        var steps = (1...count).map { index in
            ModelStep.output([getTextMessage("\(index)"), getFunctionToolCall("some_function", arguments: funcOutput, callId: "\(index)")])
        }
        if let final { steps.append(.output([getTextMessage(final)])) }
        return steps
    }

    private func drain(_ result: RunResultStreaming) async throws -> [StreamEvent] {
        var events: [StreamEvent] = []
        for try await event in result.streamEvents() { events.append(event) }
        return events
    }

    private func runItemEvents(_ events: [StreamEvent]) -> [RunItemStreamEvent] {
        events.compactMap { if case .runItem(let event) = $0 { return event } else { return nil } }
    }

    private func savedTypes(_ session: SimpleListSession) async throws -> [String] {
        try await session.getItems(limit: nil).map { $0.string("type") ?? $0.string("role") ?? "None" }
    }

    // MARK: - Limits

    @Test func testNonStreamedMaxTurns() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test_1", tools: [getFunctionTool(name: "some_function", returnValue: "result")], model: .instance(model))
        try model.extend(someFunctionTurns(5))

        await expectErrorAsync(MaxTurnsExceeded.self) {
            _ = try await Runner.run(agent, input: "user_message", maxTurns: 3)
        }
    }

    @Test func testNonStreamedMaxTurnsNoneDisablesLimit() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test_1", tools: [getFunctionTool(name: "some_function", returnValue: "result")], model: .instance(model))
        try model.extend(someFunctionTurns(4, final: "done"))

        let result = try await Runner.run(agent, input: "user_message", maxTurns: nil)

        #expect(result.finalOutput as? String == "done")
        #expect(result.maxTurns == nil)
    }

    @Test func testStreamedMaxTurns() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test_1", tools: [getFunctionTool(name: "some_function", returnValue: "result")], model: .instance(model))
        try model.extend(someFunctionTurns(5))

        await expectErrorAsync(MaxTurnsExceeded.self) {
            let output = try Runner.runStreamed(agent, input: "user_message", maxTurns: 3)
            _ = try await drain(output)
        }
    }

    @Test func testStreamedMaxTurnsNoneDisablesLimit() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test_1", tools: [getFunctionTool(name: "some_function", returnValue: "result")], model: .instance(model))
        try model.extend(someFunctionTurns(4, final: "done"))

        let result = try Runner.runStreamed(agent, input: "user_message", maxTurns: nil)
        _ = try await drain(result)

        #expect(result.finalOutput as? String == "done")
        #expect(result.maxTurns == nil)
    }

    // MARK: - Refusals

    @Test func testNonStreamedStructuredOutputRefusalRaisesWithoutRetry() async throws {
        let model = try ScriptedModel(steps: [.output([getRefusalMessage("I cannot help with that request.")])])
        let agent = Agent(name: "test_1", model: .instance(model), outputType: try AgentOutputSchema(FooModel.self))

        let error = await expectErrorAsync(ModelRefusalError.self) {
            _ = try await Runner.run(agent, input: "user_message", maxTurns: 3)
        }
        #expect(error?.refusal == "I cannot help with that request.")
        #expect(model.remainingSteps == 0)
    }

    @Test func testNonStreamedRefusalHandlerReturnsStructuredOutput() async throws {
        let model = try ScriptedModel(steps: [.output([getRefusalMessage("I cannot help with that request.")])])
        let agent = Agent(name: "test_1", model: .instance(model), outputType: try AgentOutputSchema(FooModel.self))

        let result = try await Runner.run(
            agent,
            input: "user_message",
            maxTurns: 3,
            errorHandlers: RunErrorHandlers(modelRefusal: { data in
                #expect((data.error as? ModelRefusalError)?.refusal == "I cannot help with that request.")
                #expect(!data.runData.rawResponses.isEmpty)
                return RunErrorHandlerResult(finalOutput: FooModel(summary: "safe fallback"))
            })
        )

        #expect(result.finalOutput as? FooModel == FooModel(summary: "safe fallback"))
        #expect(ItemHelpers.textMessageOutputs(result.newItems).hasSuffix(#"{"summary":"safe fallback"}"#))
    }

    @Test func testNonStreamedRefusalHandlerCanSkipHistory() async throws {
        let model = try ScriptedModel(steps: [.output([getRefusalMessage("I cannot help with that request.")])])
        let agent = Agent(name: "test_1", model: .instance(model))

        let result = try await Runner.run(
            agent,
            input: "user_message",
            errorHandlers: RunErrorHandlers(modelRefusal: { _ in
                RunErrorHandlerResult(finalOutput: "safe fallback", includeInHistory: false)
            })
        )

        #expect(result.finalOutput as? String == "safe fallback")
        #expect(ItemHelpers.textMessageOutputs(result.newItems) == "")
    }

    @Test func testStreamedRefusalHandlerReturnsOutput() async throws {
        let model = try ScriptedModel(steps: [.output([getRefusalMessage("I cannot help with that request.")])])
        let agent = Agent(name: "test_1", model: .instance(model))

        let result = try Runner.runStreamed(
            agent,
            input: "user_message",
            errorHandlers: RunErrorHandlers(modelRefusal: { _ in RunErrorHandlerResult(finalOutput: "safe fallback") })
        )
        let events = try await drain(result)

        #expect(result.finalOutput as? String == "safe fallback")
        #expect(runItemEvents(events).contains { event in
            guard event.name == "message_output_created", let item = event.item as? MessageOutputItem else { return false }
            return ItemHelpers.textMessageOutput(item) == "safe fallback"
        })
    }

    // MARK: - Structured output

    @Test func testStructuredOutputNonStreamedMaxTurns() async throws {
        let model = try ScriptedModel()
        let agent = Agent(
            name: "test_1",
            tools: [getFunctionTool(name: "tool_1", returnValue: "result")],
            model: .instance(model),
            outputType: try AgentOutputSchema(Foo.self)
        )
        try model.extend(Array(repeating: .output([getFunctionToolCall("tool_1")]), count: 5))

        await expectErrorAsync(MaxTurnsExceeded.self) {
            _ = try await Runner.run(agent, input: "user_message", maxTurns: 3)
        }
    }

    @Test func testStructuredOutputStreamedMaxTurns() async throws {
        let model = try ScriptedModel()
        let agent = Agent(
            name: "test_1",
            tools: [getFunctionTool(name: "tool_1", returnValue: "result")],
            model: .instance(model),
            outputType: try AgentOutputSchema(Foo.self)
        )
        try model.extend(Array(repeating: .output([getFunctionToolCall("tool_1")]), count: 5))

        await expectErrorAsync(MaxTurnsExceeded.self) {
            _ = try await drain(try Runner.runStreamed(agent, input: "user_message", maxTurns: 3))
        }
    }

    @Test func testStructuredOutputMaxTurnsHandlerInvalidOutput() async throws {
        let agent = Agent(name: "test_1", model: .instance(try ScriptedModel()), outputType: try AgentOutputSchema(Foo.self))

        await expectErrorAsync(UserError.self) {
            _ = try await Runner.run(
                agent,
                input: "user_message",
                maxTurns: 0,
                errorHandlers: RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: ["summary": "nope"] as JSONObject) })
            )
        }
    }

    @Test func testStructuredOutputMaxTurnsHandlerPydanticOutput() async throws {
        let agent = Agent(name: "test_1", model: .instance(try ScriptedModel()), outputType: try AgentOutputSchema(FooModel.self))

        let result = try await Runner.run(
            agent,
            input: "user_message",
            maxTurns: 0,
            errorHandlers: RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: FooModel(summary: "ok")) })
        )

        #expect(result.finalOutput as? FooModel == FooModel(summary: "ok"))
        #expect(ItemHelpers.textMessageOutputs(result.newItems) == #"{"summary":"ok"}"#)
    }

    @Test func testStructuredOutputMaxTurnsHandlerListOutput() async throws {
        let agent = Agent(name: "test_1", model: .instance(try ScriptedModel()), outputType: try AgentOutputSchema([String].self))

        let result = try await Runner.run(
            agent,
            input: "user_message",
            maxTurns: 0,
            errorHandlers: RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: ["a", "b"]) })
        )

        #expect(result.finalOutput as? [String] == ["a", "b"])
        #expect(ItemHelpers.textMessageOutputs(result.newItems) == #"{"response":["a","b"]}"#)
    }

    // MARK: - Max turns handlers

    @Test func testNonStreamedMaxTurnsHandlerReturnsOutput() async throws {
        let agent = Agent(name: "test_1", model: .instance(try ScriptedModel()))

        let result = try await Runner.run(
            agent,
            input: "user_message",
            maxTurns: 0,
            errorHandlers: RunErrorHandlers(maxTurns: { data in
                RunErrorHandlerResult(finalOutput: "summary:\(data.runData.history.count)")
            })
        )

        #expect(result.finalOutput as? String == "summary:1")
        #expect(ItemHelpers.textMessageOutputs(result.newItems) == "summary:1")
    }

    @Test func testNonStreamedMaxTurnsHandlerSkipHistory() async throws {
        let agent = Agent(name: "test_1", model: .instance(try ScriptedModel()))

        let result = try await Runner.run(
            agent,
            input: "user_message",
            maxTurns: 0,
            errorHandlers: RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: "summary", includeInHistory: false) })
        )

        #expect(result.finalOutput as? String == "summary")
        #expect(result.newItems.isEmpty)
    }

    @Test func testNonStreamedMaxTurnsHandlerRawDictOutput() async throws {
        let agent = Agent(name: "test_1", model: .instance(try ScriptedModel()))

        let result = try await Runner.run(
            agent,
            input: "user_message",
            maxTurns: 0,
            errorHandlers: RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: ["summary": "ok"] as JSONObject) })
        )

        #expect(result.finalOutput as? JSONObject == ["summary": "ok"])
    }

    @Test func testStreamedMaxTurnsHandlerReturnsOutput() async throws {
        let agent = Agent(name: "test_1", model: .instance(try ScriptedModel()))

        let result = try Runner.runStreamed(
            agent,
            input: "user_message",
            maxTurns: 0,
            errorHandlers: RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: "summary") })
        )
        let events = runItemEvents(try await drain(result))

        #expect(result.finalOutput as? String == "summary")
        #expect(events.count == 1)
        #expect(events.first?.name == "message_output_created")
        let item = try #require(events.first?.item as? MessageOutputItem)
        #expect(ItemHelpers.textMessageOutput(item) == "summary")
    }

    @Test func testStreamedMaxTurnsHandlerPydanticOutput() async throws {
        let agent = Agent(name: "test_1", model: .instance(try ScriptedModel()), outputType: try AgentOutputSchema(FooModel.self))

        let result = try Runner.runStreamed(
            agent,
            input: "user_message",
            maxTurns: 0,
            errorHandlers: RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: FooModel(summary: "ok")) })
        )
        let events = runItemEvents(try await drain(result))

        #expect(result.finalOutput as? FooModel == FooModel(summary: "ok"))
        #expect(events.count == 1)
        let item = try #require(events.first?.item as? MessageOutputItem)
        #expect(ItemHelpers.textMessageOutput(item) == #"{"summary":"ok"}"#)
    }

    @Test func testStreamedMaxTurnsHandlerListOutput() async throws {
        let agent = Agent(name: "test_1", model: .instance(try ScriptedModel()), outputType: try AgentOutputSchema([String].self))

        let result = try Runner.runStreamed(
            agent,
            input: "user_message",
            maxTurns: 0,
            errorHandlers: RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: ["a", "b"]) })
        )
        let events = runItemEvents(try await drain(result))

        #expect(result.finalOutput as? [String] == ["a", "b"])
        #expect(events.count == 1)
        let item = try #require(events.first?.item as? MessageOutputItem)
        #expect(ItemHelpers.textMessageOutput(item) == #"{"response":["a","b"]}"#)
    }

    // MARK: - Sessions

    private func runMaxTurnsHandlerWithSession(streamed: Bool) async throws -> [String] {
        let model = try ScriptedModel(steps: [
            .output([getFunctionToolCall("some_function", arguments: JSONValue.object(["a": "b"]).pythonJSONString())]),
        ])
        let agent = Agent(name: "test_1", tools: [getFunctionTool(name: "some_function", returnValue: "result")], model: .instance(model))
        let session = SimpleListSession(sessionId: "max-turns-handler")
        let handlers = RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: "fallback answer") })
        if streamed {
            let result = try Runner.runStreamed(agent, input: "user_message", maxTurns: 1, errorHandlers: handlers, session: session)
            _ = try await drain(result)
            #expect(result.finalOutput as? String == "fallback answer")
        } else {
            let result = try await Runner.run(agent, input: "user_message", maxTurns: 1, errorHandlers: handlers, session: session)
            #expect(result.finalOutput as? String == "fallback answer")
        }
        return try await savedTypes(session)
    }

    @Test func testNonStreamedMaxTurnsHandlerPersistsOutputToSession() async throws {
        let types = try await runMaxTurnsHandlerWithSession(streamed: false)
        #expect(types == ["user", "function_call", "function_call_output", "message"])
    }

    @Test func testStreamedMaxTurnsHandlerPersistsOutputToSession() async throws {
        let types = try await runMaxTurnsHandlerWithSession(streamed: true)
        #expect(types == ["user", "function_call", "function_call_output", "message"])
    }

    @Test(arguments: [false, true])
    func testMaxTurnsHandlerPersistedCountMatchesAfterToolTurn(includeInHistory: Bool) async throws {
        func runOnce(streamed: Bool) async throws -> (Int, [String]) {
            let model = try ScriptedModel(steps: [
                .output([getFunctionToolCall("some_function", arguments: JSONValue.object(["a": "b"]).pythonJSONString())]),
            ])
            let agent = Agent(name: "test", tools: [getFunctionTool(name: "some_function", returnValue: "result")], model: .instance(model))
            let session = SimpleListSession()
            let handlers = RunErrorHandlers(maxTurns: { _ in
                RunErrorHandlerResult(finalOutput: "fallback answer", includeInHistory: includeInHistory)
            })
            let state: RunState
            if streamed {
                let result = try Runner.runStreamed(agent, input: "user_message", maxTurns: 1, errorHandlers: handlers, session: session)
                _ = try await drain(result)
                state = result.toState()
            } else {
                state = try await Runner.run(agent, input: "user_message", maxTurns: 1, errorHandlers: handlers, session: session).toState()
            }
            return (state.currentTurnPersistedItemCount, try await savedTypes(session))
        }

        let nonStreamed = try await runOnce(streamed: false)
        let streamed = try await runOnce(streamed: true)

        var expectedTypes = ["user", "function_call", "function_call_output"]
        var expectedCount = 0
        if includeInHistory {
            expectedTypes.append("message")
            expectedCount = 1
        }
        #expect(nonStreamed.0 == expectedCount)
        #expect(nonStreamed.1 == expectedTypes)
        #expect(streamed.0 == expectedCount)
        #expect(streamed.1 == expectedTypes)
    }

    private struct GuardrailFailure: Error, CustomStringConvertible {
        var description: String { "guardrail failed" }
    }

    @Test(arguments: [false, true], ["pass", "error", "tripwire"])
    func testMaxTurnsHandlerOutputGuardrailSessionSemantics(streamed: Bool, outcome: String) async throws {
        let guardrail = OutputGuardrail { _, _, _ in
            if outcome == "error" { throw GuardrailFailure() }
            return GuardrailFunctionOutput(outputInfo: outcome, tripwireTriggered: outcome == "tripwire")
        }
        let agent = Agent(name: "test", model: .instance(try ScriptedModel()), outputGuardrails: [guardrail])
        let session = SimpleListSession()
        let handlers = RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: "fallback answer") })
        var streamedEvents: [StreamEvent] = []
        var streamedResult: RunResultStreaming?

        func runOnce() async throws -> any RunResultBase {
            if !streamed {
                return try await Runner.run(agent, input: "user_message", maxTurns: 0, errorHandlers: handlers, session: session)
            }
            let result = try Runner.runStreamed(agent, input: "user_message", maxTurns: 0, errorHandlers: handlers, session: session)
            streamedResult = result
            do {
                for try await event in result.streamEvents() { streamedEvents.append(event) }
            } catch {
                throw error
            }
            return result
        }

        switch outcome {
        case "error":
            await expectErrorAsync(GuardrailFailure.self) { _ = try await runOnce() }
        case "tripwire":
            let error = await expectErrorAsync(OutputGuardrailTripwireTriggered.self) { _ = try await runOnce() }
            #expect(error?.guardrailResult.agentOutput as? String == "fallback answer")
            #expect(error?.guardrailResult.output.outputInfo as? String == "tripwire")
        default:
            let result = try await runOnce()
            #expect(result.finalOutput as? String == "fallback answer")
            #expect(result.outputGuardrailResults.count == 1)
            let state = (result as? RunResult)?.toState() ?? (result as? RunResultStreaming)?.toState()
            #expect(state?.currentTurnPersistedItemCount == 1)
        }

        let types = try await savedTypes(session)
        #expect(types == (outcome == "tripwire" ? ["user"] : ["user", "message"]))

        let fallbackEvents = runItemEvents(streamedEvents).filter { event in
            guard let item = event.item as? MessageOutputItem else { return false }
            return ItemHelpers.textMessageOutput(item) == "fallback answer"
        }
        #expect(fallbackEvents.count == (streamed && outcome == "pass" ? 1 : 0))

        if streamed {
            let result = try #require(streamedResult)
            let expectedHistoryCount = outcome == "tripwire" ? 0 : 1
            #expect(result.newItems.filter { $0 is MessageOutputItem }.count == expectedHistoryCount)
            #expect(result.toState().sessionItems.filter { $0 is MessageOutputItem }.count == expectedHistoryCount)
        }
    }

    @Test func testStreamedMaxTurnsHandlerValidationFailurePersistsInput() async throws {
        let agent = Agent(name: "test", model: .instance(try ScriptedModel()), outputType: try AgentOutputSchema(Foo.self))
        let session = SimpleListSession()
        let result = try Runner.runStreamed(
            agent,
            input: "user_message",
            maxTurns: 0,
            errorHandlers: RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: ["summary": "invalid"] as JSONObject) }),
            session: session
        )

        await expectErrorAsync(UserError.self) { _ = try await drain(result) }

        #expect(try await savedTypes(session) == ["user"])
    }

    @Test(arguments: [false, true])
    func testMaxTurnsHandlerRecordsEqualMessageOccurrences(streamed: Bool) async throws {
        let model = try ScriptedModel(steps: [
            .output([
                getTextMessage("same answer"),
                getFunctionToolCall("some_function", arguments: JSONValue.object(["a": "b"]).pythonJSONString()),
            ]),
        ])
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "some_function", returnValue: "result")], model: .instance(model))
        let session = SimpleListSession()
        let handlers = RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: "same answer") })

        if streamed {
            _ = try await drain(try Runner.runStreamed(agent, input: "user_message", maxTurns: 1, errorHandlers: handlers, session: session))
        } else {
            _ = try await Runner.run(agent, input: "user_message", maxTurns: 1, errorHandlers: handlers, session: session)
        }

        let messages = try await session.getItems(limit: nil).filter { $0.string("type") == "message" }
        #expect(messages.count == 2)
    }
}
