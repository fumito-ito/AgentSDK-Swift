import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_tool_choice_reset.py.
@Suite struct ToolChoiceResetTests {
    @Test func testShouldResetToolChoiceDirect() {
        let agent = Agent(name: "test_agent")

        // Case 1: Empty tool use tracker should not change the nil tool choice.
        var settings = ModelSettings(toolChoice: nil)
        #expect(maybeResetToolChoice(agent, AgentToolUseTracker(), settings).toolChoice == settings.toolChoice)

        // Case 2: Empty tool use tracker should not change the "auto" tool choice.
        settings = ModelSettings(toolChoice: .auto)
        #expect(maybeResetToolChoice(agent, AgentToolUseTracker(), settings).toolChoice == settings.toolChoice)

        // Case 3: Empty tool use tracker should not change the "required" tool choice.
        settings = ModelSettings(toolChoice: .required)
        #expect(maybeResetToolChoice(agent, AgentToolUseTracker(), settings).toolChoice == settings.toolChoice)

        // Case 4: "required" with one tool should reset.
        var tracker = AgentToolUseTracker()
        tracker.addToolUse(agent, ["tool1"])
        #expect(maybeResetToolChoice(agent, tracker, ModelSettings(toolChoice: .required)).toolChoice == nil)

        // Case 5: "required" with multiple tools should reset.
        tracker = AgentToolUseTracker()
        tracker.addToolUse(agent, ["tool1", "tool2"])
        #expect(maybeResetToolChoice(agent, tracker, ModelSettings(toolChoice: .required)).toolChoice == nil)

        // Case 5b: a literal tool named "tool_search" counts like any other tool.
        tracker = AgentToolUseTracker()
        tracker.addToolUse(agent, ["tool_search"])
        #expect(maybeResetToolChoice(agent, tracker, ModelSettings(toolChoice: .required)).toolChoice == nil)

        // Case 6: Tool usage on a different agent should not affect the tool choice.
        settings = ModelSettings(toolChoice: "foo_bar")
        tracker = AgentToolUseTracker()
        tracker.addToolUse(Agent(name: "other_agent"), ["foo_bar", "baz"])
        #expect(maybeResetToolChoice(agent, tracker, settings).toolChoice == settings.toolChoice)

        // Case 7: "foo_bar" with multiple tools should reset.
        tracker = AgentToolUseTracker()
        tracker.addToolUse(agent, ["foo_bar", "baz"])
        #expect(maybeResetToolChoice(agent, tracker, settings).toolChoice == nil)
    }

    @Test func testRequiredToolChoiceWithMultipleRuns() async throws {
        let model = try ScriptedModel()
        try model.extend([.output([getTextMessage("First run response")]), .output([getTextMessage("Second run response")])])
        let agent = Agent(
            name: "test_agent",
            tools: [getFunctionTool(name: "custom_tool")],
            model: .instance(model),
            modelSettings: ModelSettings(toolChoice: .required)
        )

        let result1 = try await Runner.run(agent, input: "first run")
        #expect(result1.finalOutput as? String == "First run response")
        #expect(model.calls.last?.modelSettings.toolChoice == .required, "tool_choice should stay required")

        let result2 = try await Runner.run(agent, input: "second run")
        #expect(result2.finalOutput as? String == "Second run response")
        #expect(model.calls.last?.modelSettings.toolChoice == .required, "tool_choice should stay required")
    }

    @Test func testRequiredWithStopAtToolName() async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getFunctionToolCall("second_tool", arguments: "{}")]))
        let agent = Agent(
            name: "test_agent",
            tools: [
                getFunctionTool(name: "first_tool", returnValue: "first tool result"),
                getFunctionTool(name: "second_tool", returnValue: "second tool result"),
            ],
            model: .instance(model),
            modelSettings: ModelSettings(toolChoice: .required),
            toolUseBehavior: .stopAtTools(["second_tool"])
        )

        let result = try await Runner.run(agent, input: "run test")
        #expect(result.finalOutput as? String == "second tool result")
    }

    @Test func testSpecificToolChoice() async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("Test message")]))
        let agent = Agent(
            name: "test_agent",
            tools: [getFunctionTool(name: "tool1"), getFunctionTool(name: "tool2"), getFunctionTool(name: "tool3")],
            model: .instance(model),
            modelSettings: ModelSettings(toolChoice: "tool1")
        )

        let result = try await Runner.run(agent, input: "first run")
        #expect(result.finalOutput as? String == "Test message")
    }

    @Test func testRequiredWithSingleTool() async throws {
        let model = try ScriptedModel()
        try model.extend([.output([getFunctionToolCall("custom_tool", arguments: "{}")]), .output([getTextMessage("Final response")])])
        let agent = Agent(
            name: "test_agent",
            tools: [getFunctionTool(name: "custom_tool", returnValue: "tool result")],
            model: .instance(model),
            modelSettings: ModelSettings(toolChoice: .required)
        )

        let result = try await Runner.run(agent, input: "first run")
        #expect(result.finalOutput as? String == "Final response")
    }

    @Test func testDontResetToolChoiceIfNotRequired() async throws {
        let model = try ScriptedModel()
        try model.extend([.output([getFunctionToolCall("custom_tool", arguments: "{}")]), .output([getTextMessage("Final response")])])
        let agent = Agent(
            name: "test_agent",
            tools: [getFunctionTool(name: "custom_tool", returnValue: "tool result")],
            model: .instance(model),
            modelSettings: ModelSettings(toolChoice: .required),
            resetToolChoice: false
        )

        _ = try await Runner.run(agent, input: "test")

        #expect(model.calls.last?.modelSettings.toolChoice == .required)
    }
}
