import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run.py.
//
// Swift: there is no `Runner.run_sync`. Instead of a mock, the default runner is a subclass that
// delegates to the built-in runner and counts calls for its own agent, so suites running in
// parallel are unaffected.
@Suite(.serialized) struct RunTests {
    private final class CountingAgentRunner: AgentRunner, @unchecked Sendable {
        let trackedAgent: Agent
        let runCalls = SharedBox(0)
        let runStreamedCalls = SharedBox(0)

        init(trackedAgent: Agent) {
            self.trackedAgent = trackedAgent
        }

        override func run(
            _ startingAgent: Agent, input: RunnerInput, contextWrapper: RunContextWrapper?, maxTurns: Int?, hooks: RunHooks?,
            runConfig: RunConfig?, errorHandlers: RunErrorHandlers?, previousResponseId: String?, autoPreviousResponseId: Bool,
            conversationId: String?, session: (any Session)?
        ) async throws -> RunResult {
            if startingAgent === trackedAgent { runCalls.mutate { $0 += 1 } }
            return try await super.run(
                startingAgent, input: input, contextWrapper: contextWrapper, maxTurns: maxTurns, hooks: hooks, runConfig: runConfig,
                errorHandlers: errorHandlers, previousResponseId: previousResponseId, autoPreviousResponseId: autoPreviousResponseId,
                conversationId: conversationId, session: session
            )
        }

        override func runStreamed(
            _ startingAgent: Agent, input: RunnerInput, contextWrapper: RunContextWrapper?, maxTurns: Int?, hooks: RunHooks?,
            runConfig: RunConfig?, errorHandlers: RunErrorHandlers?, previousResponseId: String?, autoPreviousResponseId: Bool,
            conversationId: String?, session: (any Session)?
        ) throws -> RunResultStreaming {
            if startingAgent === trackedAgent { runStreamedCalls.mutate { $0 += 1 } }
            return try super.runStreamed(
                startingAgent, input: input, contextWrapper: contextWrapper, maxTurns: maxTurns, hooks: hooks, runConfig: runConfig,
                errorHandlers: errorHandlers, previousResponseId: previousResponseId, autoPreviousResponseId: autoPreviousResponseId,
                conversationId: conversationId, session: session
            )
        }
    }

    @Test func testStaticRunMethodsCallIntoDefaultRunner() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("done")]), .output([getTextMessage("done")])])
        let agent = Agent(name: "test", model: .instance(model))
        let runner = CountingAgentRunner(trackedAgent: agent)
        let original = getDefaultAgentRunner()
        setDefaultAgentRunner(runner)
        defer { setDefaultAgentRunner(original) }

        _ = try await Runner.run(agent, input: "test")
        #expect(runner.runCalls.value == 1)

        let streamed = try Runner.runStreamed(agent, input: "test")
        for try await _ in streamed.streamEvents() {}
        #expect(runner.runStreamedCalls.value == 1)
    }

    @Test func testRunPreservesDuplicateUserMessages() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("done")])])
        let agent = Agent(name: "test", model: .instance(model))
        let inputItems = [getTextInputItem("repeat"), getTextInputItem("repeat")]

        _ = try await Runner.run(agent, input: .items(inputItems))

        guard case .items(let sentInput)? = model.calls.last?.input else {
            Issue.record("Expected list input")
            return
        }
        #expect(sentInput.count == 2)
        #expect(sentInput[0].string("content") == "repeat")
        #expect(sentInput[1].string("content") == "repeat")
    }
}
