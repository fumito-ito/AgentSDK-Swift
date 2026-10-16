import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_max_turns.py (cases that inspect serialized `RunState` snapshots).
extension MaxTurnsTests {
    private struct GuardrailFailed: Error, CustomStringConvertible {
        var description: String { "guardrail failed" }
    }

    private func callIds(_ items: [RunItem]) -> [String] {
        items.compactMap { $0.rawItem.string("call_id") }
    }

    private func jsonString(_ state: RunState) throws -> String {
        JSONValue.object(try state.toJSON()).jsonString()
    }

    @Test func testStreamedMaxTurnsTripPreservesCompletedToolPrefix() async throws {
        let model = try ScriptedModel(steps: [.output([getFunctionToolCall("some_function", arguments: "{}", callId: "accepted-call")])])
        let agent = Agent(
            name: "test",
            tools: [getFunctionTool(name: "some_function", returnValue: "accepted-output")],
            model: .instance(model),
            outputGuardrails: [OutputGuardrail { _, _, _ in GuardrailFunctionOutput(tripwireTriggered: true) }]
        )
        let session = SimpleListSession()
        let result = try Runner.runStreamed(
            agent,
            input: "run the tool",
            maxTurns: 1,
            errorHandlers: RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: "rejected fallback") }),
            session: session
        )

        await expectErrorAsync(OutputGuardrailTripwireTriggered.self) {
            for try await _ in result.streamEvents() {}
        }

        #expect(callIds(result.newItems) == ["accepted-call", "accepted-call"])
        #expect(callIds(result.modelInputItems) == ["accepted-call", "accepted-call"])
        let state = result.toState()
        #expect(callIds(state.generatedItems) == ["accepted-call", "accepted-call"])
        #expect(callIds(state.sessionItems) == ["accepted-call", "accepted-call"])
        let serializedState = try jsonString(state)
        #expect(serializedState.contains("accepted-output"))
        #expect(serializedState.contains("rejected fallback"))

        let saved = try await session.getItems(limit: nil).map { $0.string("type") ?? $0.string("role") ?? "None" }
        #expect(saved == ["user", "function_call", "function_call_output"])
    }

    enum ResumedOutcome: String, CaseIterable, Sendable {
        case success
        case guardrailError = "guardrail_error"
        case sessionFailure = "session_failure"
    }

    private final class FailingSession: SimpleListSession, @unchecked Sendable {
        let failMessageSave = SharedBox(false)

        override func addItems(_ newItems: [TResponseInputItem]) async throws {
            if failMessageSave.value, newItems.contains(where: { $0.string("type") == "message" }) {
                throw UserError("session save failed")
            }
            try await super.addItems(newItems)
        }
    }

    private static let resumedOutcomes: [(ResumedOutcome, Bool)] = [
        (.success, true), (.success, false), (.guardrailError, true), (.sessionFailure, true),
    ]

    @Test(arguments: [false, true], resumedOutcomes)
    func testResumedMaxTurnsHandlerDoesNotAppendToCallerStateItems(streamed: Bool, testCase: (ResumedOutcome, Bool)) async throws {
        let (outcome, includeInHistory) = testCase
        let model = try ScriptedModel(steps: [.output([getTextMessage("first response")])])
        let agent = Agent(name: "test", model: .instance(model))
        let session = FailingSession()
        let first = try await Runner.run(agent, input: "first input", maxTurns: 1, session: session)
        let state = first.toState()
        let generatedItemsBefore = try state.toJSON()["generated_items"]
        let sessionItemsBefore = try state.toJSON()["session_items"]

        if outcome == .guardrailError {
            agent.outputGuardrails = [OutputGuardrail { _, _, _ in throw GuardrailFailed() }]
        }
        session.failMessageSave.value = outcome == .sessionFailure
        let handlers = RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: "fallback answer", includeInHistory: includeInHistory) })

        func resume() async throws -> (any Sendable)? {
            if streamed {
                let result = try Runner.runStreamed(agent, state: state, errorHandlers: handlers, session: session)
                for try await _ in result.streamEvents() {}
                return result.finalOutput
            }
            return try await Runner.run(agent, state: state, errorHandlers: handlers, session: session).finalOutput
        }

        switch outcome {
        case .guardrailError:
            await expectErrorAsync(GuardrailFailed.self) { _ = try await resume() }
        case .sessionFailure:
            await expectErrorAsync(UserError.self, containing: "session save failed") { _ = try await resume() }
        case .success:
            #expect(try await resume() as? String == "fallback answer")
        }

        #expect(try state.toJSON()["generated_items"] == generatedItemsBefore)
        #expect(try state.toJSON()["session_items"] == sessionItemsBefore)
    }

    @Test(arguments: [false, true])
    func testResumedMaxTurnsHandlerPreservesCheckpointAfterContinuation(streamed: Bool) async throws {
        let agent = Agent(name: "test", model: .instance(try ScriptedModel(steps: [.output([getTextMessage("first response")])])))
        let first = try await Runner.run(agent, input: "first input", maxTurns: 2)
        let state = first.toState()
        let generatedBefore = try state.toJSON()["generated_items"]
        let sessionBefore = try state.toJSON()["session_items"]
        agent.model = .instance(try ScriptedModel(steps: [.output([getFunctionToolCall("some_function", arguments: #"{"a": "b"}"#)])]))
        agent.tools = [getFunctionTool(name: "some_function", returnValue: "result")]
        agent.outputGuardrails = [OutputGuardrail { _, _, _ in throw GuardrailFailed() }]
        let handlers = RunErrorHandlers(maxTurns: { _ in RunErrorHandlerResult(finalOutput: "fallback answer") })

        await expectErrorAsync(GuardrailFailed.self) {
            if streamed {
                let result = try Runner.runStreamed(agent, state: state, errorHandlers: handlers)
                for try await _ in result.streamEvents() {}
            } else {
                _ = try await Runner.run(agent, state: state, errorHandlers: handlers)
            }
        }

        #expect(try state.toJSON()["generated_items"] == generatedBefore)
        #expect(try state.toJSON()["session_items"] == sessionBefore)
    }
}
