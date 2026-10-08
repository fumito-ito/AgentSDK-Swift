import Foundation
import Testing
@testable import Agents

// Ported from tests/test_result_cast.py.
//
// Swift: `release_agents`, weak agent references, `dataclasses.asdict`, frozen-dataclass checks,
// and pydantic rebuilds have no Swift counterpart (results own their agents and
// `AgentToolInvocation` is an immutable struct), so those tests are skipped.
@Suite struct ResultCastTests {
    private struct Foo: Equatable, Sendable {
        var bar: Int
    }

    private func createRunResult(_ finalOutput: (any Sendable)?, contextWrapper: RunContextWrapper = RunContextWrapper()) -> RunResult {
        RunResult(
            input: "test",
            newItems: [],
            rawResponses: [],
            finalOutput: finalOutput,
            lastAgent: Agent(name: "test"),
            inputGuardrailResults: [],
            outputGuardrailResults: [],
            toolInputGuardrailResults: [],
            toolOutputGuardrailResults: [],
            contextWrapper: contextWrapper,
            interruptions: [],
            maxTurns: defaultMaxTurns
        )
    }

    // MARK: - Casting

    @Test func testResultCastTypechecks() throws {
        #expect(try createRunResult(1).finalOutputAs(Int.self) == 1)
        #expect(try createRunResult("test").finalOutputAs(String.self) == "test")
        #expect(try createRunResult(Foo(bar: 1)).finalOutputAs(Foo.self) == Foo(bar: 1))
    }

    @Test func testBadCastDoesntRaise() throws {
        // Swift: an incorrect cast returns nil instead of the uncast value.
        #expect(try createRunResult(1).finalOutputAs(String.self) == nil)
        #expect(try createRunResult("test").finalOutputAs(Foo.self) == nil)
    }

    @Test func testBadCastWithParamRaises() {
        expectError(TypeError.self) { _ = try createRunResult(1).finalOutputAs(String.self, raiseIfIncorrectType: true) }
        expectError(TypeError.self) { _ = try createRunResult("test").finalOutputAs(Foo.self, raiseIfIncorrectType: true) }
        expectError(TypeError.self) { _ = try createRunResult(Foo(bar: 1)).finalOutputAs(Int.self, raiseIfIncorrectType: true) }
    }

    // MARK: - Agent tool invocation

    @Test func testRunResultAgentToolInvocationReturnsNoneForPlainContext() {
        #expect(createRunResult("ok").agentToolInvocation == nil)
    }

    @Test func testRunResultAgentToolInvocationReturnsImmutableMetadata() {
        let toolContext = ToolContext(context: (), toolName: "my_tool", toolCallId: "call_xyz", toolArguments: "{}")
        let result = createRunResult("ok", contextWrapper: toolContext)

        #expect(result.agentToolInvocation == AgentToolInvocation(toolName: "my_tool", toolCallId: "call_xyz", toolArguments: "{}"))
    }

    @Test func testRunResultStreamingAgentToolInvocationReturnsMetadata() {
        let toolContext = ToolContext(context: (), toolName: "stream_tool", toolCallId: "call_stream", toolArguments: #"{"input":"stream"}"#)
        let result = RunResultStreaming(input: "stream", currentAgent: Agent(name: "streaming-tool-agent"), contextWrapper: toolContext, maxTurns: 1)

        #expect(result.agentToolInvocation == AgentToolInvocation(
            toolName: "stream_tool",
            toolCallId: "call_stream",
            toolArguments: #"{"input":"stream"}"#
        ))
    }
}
