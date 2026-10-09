import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_runner_streamed.py (streamed resumes, max turns, and tool
// guardrail result reporting).
//
// Swift: `test_streaming_resume_carries_persisted_count` monkeypatches a private save helper,
// `test_streaming_non_max_turns_exception_does_not_emit_queued_events` writes to the private
// event queue, and `test_streamed_trip_replaces_current_tool_output_guardrail_results` needs
// blocked-output replacement; they are skipped.
extension AgentRunnerStreamedTests {
    private func drainAll(_ result: RunResultStreaming) async throws {
        for try await _ in result.streamEvents() {}
    }

    /// Approves the first interruption and resumes in streaming mode (upstream
    /// `resume_streamed_after_first_approval`).
    private func resumeAfterFirstApproval(_ agent: Agent, _ first: RunResultStreaming, session: (any Session)? = nil) async throws -> RunResultStreaming {
        let state = first.toState()
        try state.approve(try #require(first.interruptions.first))
        let resumed = try Runner.runStreamed(agent, state: state, session: session)
        try await drainAll(resumed)
        return resumed
    }

    private func approvalTool(_ name: String = "test_tool", output: String = "tool_result") throws -> FunctionTool {
        try FunctionTool.create(name: name, needsApproval: .always) { _ in output }
    }

    // MARK: - Resumes

    @Test func testStreamingResumePreservesFilteredModelInputAfterHandoff() async throws {
        let model = try ScriptedModel()
        let delegate = Agent(name: "delegate", tools: [try approvalTool("approval_tool", output: "ok")], model: .instance(model))
        let triage = Agent(
            name: "triage",
            tools: [getFunctionTool(name: "some_function", returnValue: "result")],
            handoffs: [.agent(delegate)],
            model: .instance(model)
        )
        try model.extend([
            .output([getFunctionToolCall("some_function", arguments: #"{"a": "b"}"#, callId: "triage-call")]),
            .output([getTextMessage("a_message"), getHandoffToolCall(delegate)]),
            .output([getFunctionToolCall("approval_tool", arguments: "{}", callId: "delegate-call")]),
            .output([getTextMessage("done")]),
        ])
        let captured = SharedBox<[(Set<String>, Set<String>)]>([])
        let capture: CallModelInputFilter = { data in
            var calls: Set<String> = []
            var outputs: Set<String> = []
            for item in data.modelData.input {
                guard let callId = item.string("call_id") else { continue }
                if item.string("type") == "function_call" { calls.insert(callId) }
                if item.string("type") == "function_call_output" { outputs.insert(callId) }
            }
            captured.mutate { $0.append((calls, outputs)) }
            return data.modelData
        }
        let runConfig = RunConfig(nestHandoffHistory: true, callModelInputFilter: capture)

        let first = try Runner.runStreamed(triage, input: "user_message", runConfig: runConfig)
        try await drainAll(first)
        #expect(!first.interruptions.isEmpty)
        let state = first.toState()
        try state.approve(first.interruptions[0])
        let resumed = try Runner.runStreamed(triage, state: state, runConfig: runConfig)
        try await drainAll(resumed)

        let last = try #require(captured.value.last)
        #expect(!last.0.contains("triage-call"))
        #expect(!last.1.contains("triage-call"))
        #expect(last.0.contains("delegate-call"))
        #expect(last.1.contains("delegate-call"))
        #expect(resumed.finalOutput as? String == "done")
    }

    @Test func testStreamingResumePersistsToolOutputsOnRunAgain() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [try approvalTool()], model: .instance(model))
        let session = SimpleListSession()
        try model.extend([.output([getFunctionToolCall("test_tool", arguments: "{}", callId: "call-resume")]), .output([getTextMessage("done")])])

        let first = try Runner.runStreamed(agent, input: "Use test_tool", session: session)
        try await drainAll(first)
        _ = try await resumeAfterFirstApproval(agent, first, session: session)

        let saved = try await session.getItems(limit: nil)
        #expect(saved.contains { $0.string("type") == "function_call_output" && $0.string("call_id") == "call-resume" }, "approved tool outputs should be persisted on resume")
    }

    @Test func testStreamingHITLResumeEnforcesMaxTurns() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [try approvalTool()], model: .instance(model))
        try model.extend([.output([getFunctionToolCall("test_tool", arguments: "{}")]), .output([getTextMessage("done")])])

        let first = try Runner.runStreamed(agent, input: "Use test_tool", maxTurns: 1)
        try await drainAll(first)
        #expect(!first.interruptions.isEmpty)
        let state = first.toState()
        try state.approve(first.interruptions[0])

        await expectErrorAsync(MaxTurnsExceeded.self) {
            try await drainAll(try Runner.runStreamed(agent, state: state))
        }
    }

    @Test func testStreamingMaxTurnsEmitsPendingToolOutputEvents() async throws {
        let model = try ScriptedModel()
        let tool = try FunctionTool.create(name: "test_tool") { _ in "tool_result" }
        let agent = Agent(name: "test", tools: [tool], model: .instance(model))
        try model.extend([.output([getFunctionToolCall("test_tool", arguments: "{}")]), .output([getTextMessage("done")])])

        let result = try Runner.runStreamed(agent, input: "Use test_tool", maxTurns: 1)
        var itemTypes: [String] = []
        await expectErrorAsync(MaxTurnsExceeded.self) {
            for try await event in result.streamEvents() {
                if case .runItem(let itemEvent) = event { itemTypes.append(itemEvent.item.type) }
            }
        }

        #expect(itemTypes.contains("tool_call_item"))
        #expect(itemTypes.contains("tool_call_output_item"))
    }

    @Test func testStreamingHITLServerConversationTrackerPriming() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", model: .instance(model))
        try model.enqueue(.output([getTextMessage("First response")]))
        let first = try Runner.runStreamed(agent, input: "test", previousResponseId: "resp123", conversationId: "conv123")
        try await drainAll(first)

        try model.enqueue(.output([getTextMessage("Second response")]))
        let second = try Runner.runStreamed(agent, state: first.toState(), previousResponseId: "resp123", conversationId: "conv123")
        try await drainAll(second)

        #expect(second.finalOutput as? String == "Second response")
        #expect(!second.newItems.isEmpty)
    }

    // MARK: - Tool guardrail results

    private func guardedTool(name: String = "guarded_tool", needsApproval: ToolApprovalRequirement = .never) throws -> FunctionTool {
        try FunctionTool.create(
            name: name,
            needsApproval: needsApproval,
            toolInputGuardrails: [toolInputGuardrail { _ in .allow(outputInfo: "input-checked") }],
            toolOutputGuardrails: [toolOutputGuardrail { _ in .allow(outputInfo: "output-checked") }]
        ) { _ in needsApproval.isStaticallyFalse ? "tool-result" : "approved-result" }
    }

    @Test func testStreamedRunReportsToolGuardrailResults() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [try guardedTool()], model: .instance(model))
        try model.extend([.output([getFunctionToolCall("guarded_tool", arguments: "{}", callId: "call_1")]), .output([getTextMessage("done")])])

        let result = try Runner.runStreamed(agent, input: "hello")
        try await drainAll(result)

        #expect(result.finalOutput as? String == "done")
        #expect(result.toolInputGuardrailResults.map { $0.output.outputInfo as? String } == ["input-checked"])
        #expect(result.toolOutputGuardrailResults.map { $0.output.outputInfo as? String } == ["output-checked"])
    }

    @Test func testStreamedToolGuardrailResultsMatchNonStreamed() async throws {
        func build() throws -> Agent {
            let model = try ScriptedModel()
            try model.extend([
                .output([getFunctionToolCall("guarded_tool", arguments: "{}", callId: "call_1")]),
                .output([getFunctionToolCall("guarded_tool", arguments: "{}", callId: "call_2")]),
                .output([getTextMessage("done")]),
            ])
            return Agent(name: "test", tools: [try guardedTool()], model: .instance(model))
        }

        let nonStreamed = try await Runner.run(try build(), input: "hello")
        let streamed = try Runner.runStreamed(try build(), input: "hello")
        try await drainAll(streamed)

        #expect(nonStreamed.toolInputGuardrailResults.count == 2)
        #expect(nonStreamed.toolOutputGuardrailResults.count == 2)
        #expect(streamed.toolInputGuardrailResults.count == nonStreamed.toolInputGuardrailResults.count)
        #expect(streamed.toolOutputGuardrailResults.count == nonStreamed.toolOutputGuardrailResults.count)
    }

    @Test func testStreamedToolGuardrailResultsSurviveHandoff() async throws {
        let model = try ScriptedModel()
        let target = Agent(name: "target", model: .instance(model))
        let agent = Agent(name: "source", tools: [try guardedTool()], handoffs: [.agent(target)], model: .instance(model))
        try model.extend([
            .output([getFunctionToolCall("guarded_tool", arguments: "{}", callId: "call_1"), getHandoffToolCall(target)]),
            .output([getTextMessage("done")]),
        ])

        let result = try Runner.runStreamed(agent, input: "hello")
        try await drainAll(result)

        #expect(result.finalOutput as? String == "done")
        #expect(result.toolInputGuardrailResults.count == 1)
        #expect(result.toolOutputGuardrailResults.count == 1)
    }

    @Test func testStreamedInterruptionReportsToolGuardrailResults() async throws {
        let plainTool = try FunctionTool.create(
            name: "plain_tool",
            toolInputGuardrails: [toolInputGuardrail { _ in .allow(outputInfo: "input-checked") }]
        ) { _ in "plain-result" }
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [plainTool, try approvalTool("approval_tool", output: "approved-result")], model: .instance(model))
        try model.enqueue(.output([
            getFunctionToolCall("plain_tool", arguments: "{}", callId: "call_plain"),
            getFunctionToolCall("approval_tool", arguments: "{}", callId: "call_approval"),
        ]))

        let result = try Runner.runStreamed(agent, input: "hello")
        try await drainAll(result)

        #expect(result.interruptions.count == 1)
        #expect(result.toolInputGuardrailResults.map { $0.output.outputInfo as? String } == ["input-checked"])
    }

    @Test func testStreamedToolGuardrailResultsPersistIntoRunState() async throws {
        let model = try ScriptedModel()
        let agent = Agent(name: "test", tools: [try guardedTool()], model: .instance(model))
        try model.extend([.output([getFunctionToolCall("guarded_tool", arguments: "{}", callId: "call_1")]), .output([getTextMessage("done")])])

        let result = try Runner.runStreamed(agent, input: "hello")
        try await drainAll(result)

        let state = result.toState()
        #expect(state.toolInputGuardrailResults.count == 1)
        #expect(state.toolOutputGuardrailResults.count == 1)
    }

    enum ResumeShape: String, CaseIterable, Sendable {
        case runAgain, terminal, handoff
    }

    @Test(arguments: ResumeShape.allCases)
    func testStreamedResumeToolGuardrailResultsMatchNonStreamed(shape: ResumeShape) async throws {
        func build() throws -> Agent {
            let model = try ScriptedModel()
            let tool = try guardedTool(name: "approval_tool", needsApproval: .always)
            switch shape {
            case .runAgain:
                try model.extend([.output([getFunctionToolCall("approval_tool", arguments: "{}", callId: "call_approval")]), .output([getTextMessage("done")])])
                return Agent(name: "TestAgent", tools: [tool], model: .instance(model))
            case .terminal:
                try model.enqueue(.output([getFunctionToolCall("approval_tool", arguments: "{}", callId: "call_approval")]))
                return Agent(name: "TestAgent", tools: [tool], model: .instance(model), toolUseBehavior: .stopOnFirstTool)
            case .handoff:
                let target = Agent(name: "target", model: .instance(model))
                try model.extend([
                    .output([getFunctionToolCall("approval_tool", arguments: "{}", callId: "call_approval"), getHandoffToolCall(target)]),
                    .output([getTextMessage("done")]),
                ])
                return Agent(name: "TestAgent", tools: [tool], handoffs: [.agent(target)], model: .instance(model))
            }
        }

        let streamedAgent = try build()
        let streamedFirst = try Runner.runStreamed(streamedAgent, input: "hello")
        try await drainAll(streamedFirst)
        #expect(streamedFirst.interruptions.count == 1)
        let streamed = try await resumeAfterFirstApproval(streamedAgent, streamedFirst)

        let nonStreamedAgent = try build()
        let nonStreamedFirst = try await Runner.run(nonStreamedAgent, input: "hello")
        #expect(nonStreamedFirst.interruptions.count == 1)
        let state = nonStreamedFirst.toState()
        try state.approve(nonStreamedFirst.interruptions[0])
        let nonStreamed = try await Runner.run(nonStreamedAgent, state: state)

        let expectedOutput = shape == .terminal ? "approved-result" : "done"
        #expect(streamed.finalOutput as? String == expectedOutput)
        #expect(nonStreamed.finalOutput as? String == expectedOutput)
        if shape != .runAgain {
            #expect(streamed.toolInputGuardrailResults.map { $0.output.outputInfo as? String } == ["input-checked"])
            #expect(streamed.toolOutputGuardrailResults.map { $0.output.outputInfo as? String } == ["output-checked"])
        }
        #expect(streamed.toolInputGuardrailResults.count == nonStreamed.toolInputGuardrailResults.count)
        #expect(streamed.toolOutputGuardrailResults.count == nonStreamed.toolOutputGuardrailResults.count)
    }
}
