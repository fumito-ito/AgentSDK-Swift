import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_resume_tool_use_behavior.py.
@Suite struct ResumeToolUseBehaviorTests {
    enum Behavior: String, CaseIterable, Sendable {
        case stopAt = "stop_at"
        case first
        case custom
    }

    private struct Case: Sendable, CustomTestStringConvertible {
        var gatedFirst: Bool
        var streamed: Bool
        var serialized: Bool
        var reject: Bool
        var behavior: Behavior

        var testDescription: String {
            "gated_first=\(gatedFirst) streamed=\(streamed) serialized=\(serialized) reject=\(reject) behavior=\(behavior.rawValue)"
        }
    }

    private static let cases: [Case] = [false, true].flatMap { gatedFirst in
        [false, true].flatMap { streamed in
            [false, true].flatMap { serialized in
                [false, true].flatMap { reject in
                    Behavior.allCases.map {
                        Case(gatedFirst: gatedFirst, streamed: streamed, serialized: serialized, reject: reject, behavior: $0)
                    }
                }
            }
        }
    }

    @Test(arguments: cases)
    private func testResumePreservesCompletedToolResults(_ testCase: Case) async throws {
        let effects = SharedBox<[String]>([])
        let seen = SharedBox<[String]>([])
        let submitOrder = try FunctionTool.create(name: "submit_order") { _ in
            effects.mutate { $0.append("submit") }
            return "order submitted"
        }
        let notifyManager = try FunctionTool.create(name: "notify_manager", needsApproval: true) { _ in
            effects.mutate { $0.append("notify") }
            return "manager notified"
        }
        var calls = [
            getFunctionToolCall("submit_order", arguments: "{}", callId: "submit"),
            getFunctionToolCall("notify_manager", arguments: "{}", callId: "notify"),
        ]
        if testCase.gatedFirst {
            calls.reverse()
        }
        let model = try ScriptedModel(steps: [.output(calls)])
        let behavior: ToolUseBehavior
        switch testCase.behavior {
        case .stopAt: behavior = .stopAtTools(["submit_order"])
        case .first: behavior = .stopOnFirstTool
        case .custom:
            behavior = .custom { _, results in
                seen.mutate { $0.append(contentsOf: results.map(\.tool.name)) }
                return ToolsToFinalOutputResult(isFinalOutput: true, finalOutput: results[0].output)
            }
        }
        let agent = Agent(name: "orders", tools: [submitOrder, notifyManager], model: .instance(model), toolUseBehavior: behavior)

        var state: RunState
        if testCase.streamed {
            let streamed = try Runner.runStreamed(agent, input: "submit an order")
            for try await _ in streamed.streamEvents() {}
            state = streamed.toState()
        } else {
            state = try await Runner.run(agent, input: "submit an order").toState()
        }
        #expect(effects.value == ["submit"])
        if testCase.serialized {
            state = try await RunState.fromString(agent, try state.toString())
        }
        for interruption in state.getInterruptions() {
            if testCase.reject {
                try state.reject(interruption, rejectionMessage: "notification rejected")
            } else {
                try state.approve(interruption)
            }
        }

        let finalOutput: (any Sendable)?
        if testCase.streamed {
            let streamed = try Runner.runStreamed(agent, state: state)
            for try await _ in streamed.streamEvents() {}
            finalOutput = streamed.finalOutput
        } else {
            finalOutput = try await Runner.run(agent, state: state).finalOutput
        }

        var expected = "order submitted"
        if testCase.gatedFirst, testCase.behavior != .stopAt {
            expected = testCase.reject ? "notification rejected" : "manager notified"
        }
        #expect(finalOutput as? String == expected)
        #expect(model.calls.count == 1)
        #expect(effects.value == (testCase.reject ? ["submit"] : ["submit", "notify"]))
        if testCase.behavior == .custom {
            #expect(seen.value == (testCase.gatedFirst ? ["notify_manager", "submit_order"] : ["submit_order", "notify_manager"]))
        }
    }
}
