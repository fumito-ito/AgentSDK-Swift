import Testing
@testable import Agents

// Ported from tests/test_custom_tool.py.
@Suite struct CustomToolTests {
    private func customToolCall() throws -> ResponseCustomToolCall {
        try #require(ResponseCustomToolCall(json: ["type": "custom_tool_call", "name": "raw_editor", "call_id": "call_custom", "input": "hello"]))
    }

    @Test func testCustomToolActionReturnsCustomToolCallOutput() async throws {
        let tool = try CustomTool(
            name: "raw_editor",
            description: "Edit raw text.",
            onInvokeTool: { ctx, rawInput in
                #expect(ctx.toolName == "raw_editor")
                #expect(ctx.toolArguments == "hello")
                return rawInput.uppercased()
            },
            format: ["type": "text"]
        )
        let agent = Agent(name: "custom-agent", tools: [tool])

        let result = try await CustomToolAction.execute(
            agent: agent, call: ToolRunCustom(toolCall: customToolCall(), customTool: tool), hooks: RunHooks(),
            contextWrapper: RunContextWrapper(), config: RunConfig()
        )

        let item = try #require(result as? ToolCallOutputItem)
        #expect(item.rawItem == ["type": "custom_tool_call_output", "call_id": "call_custom", "output": "HELLO"])
    }

    @Test func testCustomToolOnApprovalCallbackAutoRejectsWithReason() async throws {
        let tool = try CustomTool(
            name: "raw_editor",
            description: "Edit raw text.",
            onInvokeTool: { _, _ in
                Issue.record("rejected custom tool should not execute")
                return ""
            },
            format: ["type": "text"],
            needsApproval: .required(true),
            onApproval: { _, _ in ToolOnApprovalResult(approve: false, reason: "Not allowed") }
        )
        let agent = Agent(name: "custom-agent", tools: [tool])

        let result = try await CustomToolAction.execute(
            agent: agent, call: ToolRunCustom(toolCall: customToolCall(), customTool: tool), hooks: RunHooks(),
            contextWrapper: RunContextWrapper(), config: RunConfig()
        )

        let item = try #require(result as? ToolCallOutputItem)
        #expect(item.output as? String == "Not allowed")
        #expect(item.rawItem == ["type": "custom_tool_call_output", "call_id": "call_custom", "output": "Not allowed"])
    }
}
