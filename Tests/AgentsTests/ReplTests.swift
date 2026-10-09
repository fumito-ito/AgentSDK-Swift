import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_repl.py.
//
// Swift: scripted input and captured output are injected instead of monkeypatching `input()`
// and reading `capsys`.
@Suite struct ReplTests {
    private final class Console: @unchecked Sendable {
        private var inputs: [String?]
        private(set) var output = ""

        init(_ inputs: [String?]) {
            self.inputs = inputs
        }

        func read(_ prompt: String) -> String? {
            inputs.isEmpty ? nil : inputs.removeFirst()
        }

        func write(_ text: String) {
            output += text
        }
    }

    @Test func testRunDemoLoopConversation() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("hello")]), .output([getTextMessage("good")])])
        let agent = Agent(name: "test", model: .instance(model))
        let console = Console(["Hi", "How are you?", "quit"])

        try await runDemoLoop(agent, stream: false, readInput: console.read, write: console.write)

        #expect(console.output.contains("hello"))
        #expect(console.output.contains("good"))
        #expect(model.calls.last?.input == .items([getTextInputItem("Hi"), getTextMessage("hello"), getTextInputItem("How are you?")]))
    }

    @Test func testRunDemoLoopStreaming() async throws {
        let model = try ScriptedModel()
        let targetAgent = Agent(name: "target", model: .instance(model))
        let agent = Agent(name: "test", tools: [getFunctionTool(name: "foo", returnValue: "tool_result")], handoffs: [.agent(targetAgent)], model: .instance(model))
        // A single user turn that exercises every streamed event branch.
        try model.extend([
            .output([getFunctionToolCall("foo", arguments: "{}")]),
            .output([getHandoffToolCall(targetAgent)]),
            .output([getTextMessage("all done")]),
        ])
        let console = Console(["Hello", "exit"])

        try await runDemoLoop(agent, stream: true, readInput: console.read, write: console.write)

        #expect(console.output.contains("all done"))
        #expect(console.output.contains("[tool called]"))
        #expect(console.output.contains("[tool output: tool_result]"))
        #expect(console.output.contains("[Agent updated: target]"))
    }

    @Test func testRunDemoLoopExitsOnEOF() async throws {
        let model = try ScriptedModel()
        let console = Console([])

        try await runDemoLoop(Agent(name: "test", model: .instance(model)), stream: false, readInput: console.read, write: console.write)

        // The loop should terminate cleanly without ever invoking the model.
        #expect(model.calls.isEmpty)
    }

    @Test func testRunDemoLoopSkipsEmptyInput() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("hello")])])
        let console = Console(["", "Hi", "quit"])

        try await runDemoLoop(Agent(name: "test", model: .instance(model)), stream: false, readInput: console.read, write: console.write)

        #expect(console.output.contains("hello"))
        #expect(model.calls.last?.input == .items([getTextInputItem("Hi")]))
    }
}
