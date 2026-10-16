import Foundation
import Testing
@testable import Agents

// Ported from tests/test_tool_context.py.
//
// Swift: these upstream tests are skipped:
// - `test_tool_context_is_hashable_like_run_context_wrapper`: Python dataclass hashing.
// - `test_tool_context_missing_defaults_raise`: the initializer requires every field.
// - dictionary `run_config` normalization and unknown-field rejection: `runConfig` is typed.
// - `invoke_function_tool` tests that choose a plain `RunContextWrapper` from the handler's type
//   annotation: `onInvokeTool` always receives the `ToolContext`.
// `RunConfig` is a value type, so identity checks compare the configured model.
@Suite struct ToolContextTests {
    private func functionCall(_ name: String, callId: String, arguments: String = "{}", namespace: String? = nil) -> TResponseOutputItem {
        var call: TResponseOutputItem = ["type": "function_call", "name": .string(name), "call_id": .string(callId), "arguments": .string(arguments)]
        if let namespace {
            call["namespace"] = .string(namespace)
        }
        return call
    }

    @Test func testToolContextRequiresFields() {
        let ctx = RunContextWrapper(context: [String: String]())
        expectError(ValueError.self) {
            _ = try ToolContext.fromAgentContext(ctx, toolCallId: "call-1")
        }
    }

    @Test func testToolContextFromAgentContextPopulatesFields() throws {
        let toolCall = functionCall("test_tool", callId: "call-123", arguments: #"{"a": 1}"#)
        let agent = Agent(name: "agent")

        let toolContext = try ToolContext.fromAgentContext(RunContextWrapper(), toolCallId: "call-123", toolCall: toolCall, agent: agent)

        #expect(toolContext.toolName == "test_tool")
        #expect(toolContext.toolCallId == "call-123")
        #expect(toolContext.toolArguments == #"{"a": 1}"#)
        #expect(toolContext.agent === agent)
    }

    @Test func testToolContextAgentNoneByDefault() throws {
        let toolContext = try ToolContext.fromAgentContext(RunContextWrapper(), toolCallId: "call-1", toolCall: functionCall("test_tool", callId: "call-1"))
        #expect(toolContext.agent == nil)
    }

    @Test func testToolContextConstructorAcceptsAgentKeyword() {
        let agent = Agent(name: "direct-agent")
        let toolContext = ToolContext(context: [String: String](), toolName: "my_tool", toolCallId: "call-2", toolArguments: "{}", agent: agent)
        #expect(toolContext.agent === agent)
    }

    @Test func testToolContextConstructorInfersNamespaceFromToolCall() {
        let toolCall = functionCall("lookup_account", callId: "call-2", namespace: "billing")

        let toolContext = ToolContext(context: [String: String](), toolName: "lookup_account", toolCallId: "call-2", toolArguments: "{}", toolCall: toolCall)

        #expect(toolContext.toolNamespace == "billing")
        #expect(toolContext.qualifiedToolName == "billing.lookup_account")
    }

    @Test func testToolContextQualifiedToolNameCollapsesSyntheticNamespace() {
        let toolCall = functionCall("get_weather", callId: "call-weather", namespace: "get_weather")

        let toolContext = ToolContext(context: [String: String](), toolName: "get_weather", toolCallId: "call-weather", toolArguments: "{}", toolCall: toolCall)

        #expect(toolContext.toolNamespace == "get_weather")
        #expect(toolContext.qualifiedToolName == "get_weather")
    }

    @Test func testToolContextFromToolContextInheritsAgent() throws {
        let agent = Agent(name: "origin-agent")
        let parentContext = ToolContext(
            context: [String: String](), toolName: "test_tool", toolCallId: "call-3", toolArguments: "{}",
            toolCall: functionCall("test_tool", callId: "call-3"), agent: agent
        )

        let derivedContext = try ToolContext.fromAgentContext(parentContext, toolCallId: "call-4", toolCall: functionCall("test_tool", callId: "call-4"))

        #expect(derivedContext.agent === agent)
    }

    @Test func testToolContextFromToolContextInheritsRunConfig() throws {
        let parentContext = ToolContext(
            context: [String: String](), toolName: "test_tool", toolCallId: "call-3", toolArguments: "{}",
            toolCall: functionCall("test_tool", callId: "call-3"), runConfig: RunConfig(model: "gpt-4.1-mini")
        )

        let derivedContext = try ToolContext.fromAgentContext(parentContext, toolCallId: "call-4", toolCall: functionCall("test_tool", callId: "call-4"))

        #expect(derivedContext.runConfig?.model?.name == "gpt-4.1-mini")
    }

    @Test func testToolContextFromAgentContextPrefersExplicitRunConfig() throws {
        let toolContext = try ToolContext.fromAgentContext(
            RunContextWrapper(), toolCallId: "call-1", toolCall: functionCall("test_tool", callId: "call-1"), runConfig: RunConfig(model: "gpt-4.1")
        )
        #expect(toolContext.runConfig?.model?.name == "gpt-4.1")
    }

    @Test func testInvokeFunctionToolPreservesToolContextWhenRequested() async throws {
        let captured = SharedBox<ToolContext?>(nil)
        let functionTool = try FunctionTool(
            name: "tool_context_tool",
            description: "test",
            paramsJSONSchema: ["type": "object", "properties": [:]],
            onInvokeTool: { ctx, _ in
                captured.mutate { $0 = ctx }
                return ctx.toolName
            }
        )
        let toolContext = ToolContext(
            context: "Stormy", toolName: "tool_context_tool", toolCallId: "call-2", toolArguments: "{}",
            agent: Agent(name: "agent"), runConfig: RunConfig(model: "gpt-4.1-mini")
        )

        let result = try await invokeFunctionTool(tool: functionTool, context: toolContext, arguments: "{}")

        #expect(result as? String == "tool_context_tool")
        #expect(captured.value === toolContext)
    }
}
