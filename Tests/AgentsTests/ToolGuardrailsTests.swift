import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_tool_guardrails.py.
//
// Swift: `test_invalid_tool_input_guardrail_raises_user_error` and the output variant are skipped
// because the guardrail function is typed. Sync and async guardrail variants collapse into one
// test each because Swift guardrail functions are always `async`. The decorator tests check
// explicit names only: Swift closures have no function name to default to.
@Suite struct ToolGuardrailsTests {
    private func mockToolContext(_ toolArguments: String = #"{"param": "value"}"#) -> ToolContext {
        ToolContext(context: (), toolName: "test_tool", toolCallId: "call_123", toolArguments: toolArguments)
    }

    private func inputGuardrail(triggers: Bool, outputInfo: (any Sendable)? = nil) -> ToolInputGuardrail {
        ToolInputGuardrail { _ in
            triggers ? .raiseException(outputInfo: outputInfo) : .allow(outputInfo: outputInfo)
        }
    }

    private func outputGuardrail(triggers: Bool, outputInfo: (any Sendable)? = nil) -> ToolOutputGuardrail {
        ToolOutputGuardrail { _ in
            triggers ? .raiseException(outputInfo: outputInfo) : .allow(outputInfo: outputInfo)
        }
    }

    private func inputData(_ toolArguments: String = #"{"param": "value"}"#) -> ToolInputGuardrailData {
        ToolInputGuardrailData(context: mockToolContext(toolArguments), agent: Agent(name: "test"))
    }

    private func outputData(_ output: String = "test output") -> ToolOutputGuardrailData {
        ToolOutputGuardrailData(context: mockToolContext(), agent: Agent(name: "test"), output: output)
    }

    // MARK: - Guardrail execution

    @Test func testToolInputGuardrail() async throws {
        // Ports test_sync_tool_input_guardrail and test_async_tool_input_guardrail.
        let data = inputData()

        var result = try await inputGuardrail(triggers: false).run(data)
        #expect(result.behavior.type == "allow")
        #expect(result.outputInfo == nil)

        result = try await inputGuardrail(triggers: true).run(data)
        #expect(result.behavior.type == "raise_exception")
        #expect(result.outputInfo == nil)

        result = try await inputGuardrail(triggers: true, outputInfo: "test_info").run(data)
        #expect(result.behavior.type == "raise_exception")
        #expect(result.outputInfo as? String == "test_info")
    }

    @Test func testToolOutputGuardrail() async throws {
        // Ports test_sync_tool_output_guardrail and test_async_tool_output_guardrail.
        let data = outputData()

        var result = try await outputGuardrail(triggers: false).run(data)
        #expect(result.behavior.type == "allow")
        #expect(result.outputInfo == nil)

        result = try await outputGuardrail(triggers: true).run(data)
        #expect(result.behavior.type == "raise_exception")
        #expect(result.outputInfo == nil)

        result = try await outputGuardrail(triggers: true, outputInfo: "test_info").run(data)
        #expect(result.behavior.type == "raise_exception")
        #expect(result.outputInfo as? String == "test_info")
    }

    @Test func testToolInputGuardrailDecorators() async throws {
        let guardrail = ToolInputGuardrail(name: "Custom input name") { _ in .allow(outputInfo: "test_2") }

        let result = try await guardrail.run(inputData())

        #expect(result.behavior.type == "allow")
        #expect(result.outputInfo as? String == "test_2")
        #expect(guardrail.getName() == "Custom input name")
    }

    @Test func testToolOutputGuardrailDecorators() async throws {
        let guardrail = ToolOutputGuardrail(name: "Custom output name") { _ in .allow(outputInfo: "test_4") }

        let result = try await guardrail.run(outputData())

        #expect(result.behavior.type == "allow")
        #expect(result.outputInfo as? String == "test_4")
        #expect(guardrail.getName() == "Custom output name")
    }

    // MARK: - Practical examples

    @Test func testPasswordBlockingInputGuardrail() async throws {
        let checkForPassword = ToolInputGuardrail(name: "check_for_password") { data in
            if data.context.toolArguments.lowercased().contains("password") {
                return .rejectContent("Tool call blocked: contains password", outputInfo: ["blocked_word": "password"])
            }
            return ToolGuardrailFunctionOutput(outputInfo: "safe_input")
        }

        var result = try await checkForPassword.run(inputData(#"{"message": "Hello password world"}"#))
        #expect(result.behavior == .rejectContent(message: "Tool call blocked: contains password"))
        #expect((result.outputInfo as? [String: String])?["blocked_word"] == "password")

        result = try await checkForPassword.run(inputData(#"{"message": "Hello safe world"}"#))
        #expect(result.behavior.type == "allow")
        #expect(result.outputInfo as? String == "safe_input")
    }

    @Test func testSSNBlockingOutputGuardrail() async throws {
        let checkForSSN = ToolOutputGuardrail(name: "check_for_ssn") { data in
            let output = String(describing: data.output).lowercased()
            if output.contains("ssn") || output.contains("123-45-6789") {
                return .raiseException(outputInfo: ["blocked_pattern": "SSN"])
            }
            return ToolGuardrailFunctionOutput(outputInfo: "safe_output")
        }

        var result = try await checkForSSN.run(outputData("User SSN is 123-45-6789"))
        #expect(result.behavior.type == "raise_exception")
        #expect((result.outputInfo as? [String: String])?["blocked_pattern"] == "SSN")

        result = try await checkForSSN.run(outputData("User name is John Doe"))
        #expect(result.behavior.type == "allow")
        #expect(result.outputInfo as? String == "safe_output")
    }

    // MARK: - Exceptions

    @Test func testToolInputGuardrailException() {
        let guardrail = ToolInputGuardrail(name: "test_guardrail") { _ in .raiseException(outputInfo: "test") }
        let output = ToolGuardrailFunctionOutput.raiseException(outputInfo: "test")

        let exception = ToolInputGuardrailTripwireTriggered(guardrail: guardrail, output: output)

        #expect(exception.guardrail.getName() == "test_guardrail")
        #expect(exception.output.behavior == output.behavior)
        #expect(exception.output.outputInfo as? String == "test")
        #expect(describeError(exception).contains("ToolInputGuardrail"))
    }

    @Test func testToolOutputGuardrailException() {
        let guardrail = ToolOutputGuardrail(name: "test_guardrail") { _ in .raiseException(outputInfo: "test") }
        let output = ToolGuardrailFunctionOutput.raiseException(outputInfo: "test")

        let exception = ToolOutputGuardrailTripwireTriggered(guardrail: guardrail, output: output)

        #expect(exception.guardrail.getName() == "test_guardrail")
        #expect(exception.output.behavior == output.behavior)
        #expect(exception.output.outputInfo as? String == "test")
        #expect(describeError(exception).contains("ToolOutputGuardrail"))
    }

    // MARK: - Behaviors

    @Test func testAllowBehavior() async throws {
        let guardrail = ToolInputGuardrail(name: "allow_guardrail") { _ in .allow(outputInfo: "allowed") }
        let result = try await guardrail.run(inputData())
        #expect(result.behavior.type == "allow")
        #expect(result.outputInfo as? String == "allowed")
    }

    @Test func testRejectContentBehavior() async throws {
        let guardrail = ToolInputGuardrail(name: "reject_content_guardrail") { _ in
            .rejectContent("Tool blocked by guardrail", outputInfo: "rejected")
        }
        let result = try await guardrail.run(inputData())
        #expect(result.behavior.type == "reject_content")
        #expect(result.behavior == .rejectContent(message: "Tool blocked by guardrail"))
        #expect(result.outputInfo as? String == "rejected")
    }

    @Test func testRaiseExceptionBehavior() async throws {
        let guardrail = ToolInputGuardrail(name: "raise_exception_guardrail") { _ in .raiseException(outputInfo: "exception") }
        let result = try await guardrail.run(inputData())
        #expect(result.behavior.type == "raise_exception")
        #expect(result.outputInfo as? String == "exception")
    }

    @Test func testMixedBehaviorOutputGuardrail() async throws {
        let mixedGuardrail = ToolOutputGuardrail(name: "mixed_guardrail") { data in
            let output = String(describing: data.output).lowercased()
            if output.contains("dangerous") {
                return .raiseException(outputInfo: ["reason": "dangerous_content"])
            } else if output.contains("sensitive") {
                return .rejectContent("Content was filtered", outputInfo: ["reason": "sensitive_content"])
            }
            return ToolGuardrailFunctionOutput(outputInfo: ["status": "clean"])
        }

        var result = try await mixedGuardrail.run(outputData("This is dangerous content"))
        #expect(result.behavior.type == "raise_exception")
        #expect((result.outputInfo as? [String: String])?["reason"] == "dangerous_content")

        result = try await mixedGuardrail.run(outputData("This is sensitive data"))
        #expect(result.behavior == .rejectContent(message: "Content was filtered"))
        #expect((result.outputInfo as? [String: String])?["reason"] == "sensitive_content")

        result = try await mixedGuardrail.run(outputData("This is clean content"))
        #expect(result.behavior.type == "allow")
        #expect((result.outputInfo as? [String: String])?["status"] == "clean")
    }

    // MARK: - Run data

    private struct QueryArgs: Codable, Sendable {
        var query: String
    }

    private func agentWithRepeatedGuardedToolCalls(
        inputGuardrails: [ToolInputGuardrail]? = nil,
        outputGuardrails: [ToolOutputGuardrail]? = nil
    ) throws -> Agent {
        let guarded = try FunctionTool.create(name: "guarded", argumentsType: QueryArgs.self) { _, _ in "tool output" }
        guarded.toolInputGuardrails = inputGuardrails ?? []
        guarded.toolOutputGuardrails = outputGuardrails ?? []
        let model = try ScriptedModel(steps: [
            .output([getFunctionToolCall("guarded", arguments: #"{"query": "secret"}"#, callId: "guarded_1")]),
            .output([getFunctionToolCall("guarded", arguments: #"{"query": "secret"}"#, callId: "guarded_2")]),
        ])
        return Agent(name: "guarded_tool_agent", tools: [guarded], model: .instance(model))
    }

    private func runAgent(_ agent: Agent, streaming: Bool, maxTurns: Int? = defaultMaxTurns) async throws {
        if streaming {
            let result = try Runner.runStreamed(agent, input: "go", maxTurns: maxTurns)
            for try await _ in result.streamEvents() {}
        } else {
            _ = try await Runner.run(agent, input: "go", maxTurns: maxTurns)
        }
    }

    @Test(arguments: [false, true])
    func testToolInputGuardrailResultsReportedOnMaxTurns(streaming: Bool) async throws {
        let guardrail = ToolInputGuardrail(name: "input_rejects") { _ in
            .rejectContent("blocked by policy", outputInfo: "input_rejected")
        }
        let agent = try agentWithRepeatedGuardedToolCalls(inputGuardrails: [guardrail])

        let error = await expectErrorAsync(MaxTurnsExceeded.self) {
            try await runAgent(agent, streaming: streaming, maxTurns: 2)
        }

        let runData = try #require(error?.runData)
        #expect(runData.toolInputGuardrailResults.map { $0.guardrail.getName() } == ["input_rejects", "input_rejects"])
        #expect(runData.toolOutputGuardrailResults.isEmpty)
    }

    @Test(arguments: [false, true])
    func testToolOutputGuardrailResultsReportedOnMaxTurns(streaming: Bool) async throws {
        let guardrail = ToolOutputGuardrail(name: "output_rejects") { _ in
            .rejectContent("blocked by policy", outputInfo: "output_rejected")
        }
        let agent = try agentWithRepeatedGuardedToolCalls(outputGuardrails: [guardrail])

        let error = await expectErrorAsync(MaxTurnsExceeded.self) {
            try await runAgent(agent, streaming: streaming, maxTurns: 2)
        }

        let runData = try #require(error?.runData)
        #expect(runData.toolOutputGuardrailResults.map { $0.guardrail.getName() } == ["output_rejects", "output_rejects"])
        #expect(runData.toolInputGuardrailResults.isEmpty)
    }

    @Test(arguments: [false, true])
    func testToolTripwirePreservesCompletedTurnResults(streaming: Bool) async throws {
        let guardrailRuns = SharedBox(0)
        let guardrail = ToolInputGuardrail(name: "allow_then_raise") { _ in
            let runs = guardrailRuns.mutate { value -> Int in
                value += 1
                return value
            }
            if runs == 2 {
                return .raiseException(outputInfo: "second_turn")
            }
            return .allow(outputInfo: "first_turn")
        }
        let agent = try agentWithRepeatedGuardedToolCalls(inputGuardrails: [guardrail])

        let error = await expectErrorAsync(ToolInputGuardrailTripwireTriggered.self) {
            try await runAgent(agent, streaming: streaming)
        }

        let exception = try #require(error)
        #expect(exception.guardrail.getName() == "allow_then_raise")
        #expect(exception.output.outputInfo as? String == "second_turn")
        let runData = try #require(exception.runData)
        #expect(runData.toolInputGuardrailResults.map { $0.output.outputInfo as? String } == ["first_turn"])
        #expect(runData.toolOutputGuardrailResults.isEmpty)
    }
}
