import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_tool_use_behavior.py.
//
// Swift: `test_invalid_tool_use_behavior_raises` and the invalid-return cases of
// `test_custom_tool_use_behavior_return_contract` are skipped: `ToolUseBehavior` is an enum and
// custom behaviors must return `ToolsToFinalOutputResult`, so neither state can be expressed.
@Suite struct ToolUseBehaviorTests {
    private func makeFunctionToolResult(_ agent: Agent, _ output: String, toolName: String? = nil, tool: FunctionTool? = nil) -> FunctionToolResult {
        let tool = tool ?? getFunctionTool(name: toolName ?? "dummy", returnValue: output)
        let rawItem: JSONObject = ["call_id": "1", "output": .string(output), "type": "function_call_output"]
        let runItem = ToolCallOutputItem(agent: agent, rawItem: rawItem, output: output)
        return FunctionToolResult(tool: tool, output: output, runItem: runItem)
    }

    // MARK: - Built-in behaviors

    @Test func testNoToolResultsReturnsNotFinalOutput() async throws {
        let agent = Agent(name: "test")
        let result = try await checkForFinalOutputFromTools(agent, [], RunContextWrapper())
        #expect(!result.isFinalOutput)
        #expect(result.finalOutput == nil)
    }

    @Test func testRunLLMAgainBehavior() async throws {
        let agent = Agent(name: "test", toolUseBehavior: .runLLMAgain)
        let results = [makeFunctionToolResult(agent, "ignored")]
        let result = try await checkForFinalOutputFromTools(agent, results, RunContextWrapper())
        #expect(!result.isFinalOutput)
        #expect(result.finalOutput == nil)
    }

    @Test func testStopOnFirstToolBehavior() async throws {
        let agent = Agent(name: "test", toolUseBehavior: .stopOnFirstTool)
        let results = [makeFunctionToolResult(agent, "first_tool_output"), makeFunctionToolResult(agent, "ignored")]
        let result = try await checkForFinalOutputFromTools(agent, results, RunContextWrapper())
        #expect(result.isFinalOutput)
        #expect(result.finalOutput as? String == "first_tool_output")
    }

    // MARK: - Custom behaviors

    @Test func testCustomToolUseBehaviorSync() async throws {
        let agent = Agent(name: "test", toolUseBehavior: .custom { _, results in
            #expect(results.count == 3)
            return ToolsToFinalOutputResult(isFinalOutput: true, finalOutput: "custom")
        })
        let results = ["ignored1", "ignored2", "ignored3"].map { makeFunctionToolResult(agent, $0) }
        let result = try await checkForFinalOutputFromTools(agent, results, RunContextWrapper())
        #expect(result.isFinalOutput)
        #expect(result.finalOutput as? String == "custom")
    }

    @Test func testCustomToolUseBehaviorAsync() async throws {
        let agent = Agent(name: "test", toolUseBehavior: .custom { _, results in
            await Task.yield()
            #expect(results.count == 3)
            return ToolsToFinalOutputResult(isFinalOutput: true, finalOutput: "async_custom")
        })
        let results = ["ignored1", "ignored2", "ignored3"].map { makeFunctionToolResult(agent, $0) }
        let result = try await checkForFinalOutputFromTools(agent, results, RunContextWrapper())
        #expect(result.isFinalOutput)
        #expect(result.finalOutput as? String == "async_custom")
    }

    @Test func testCustomToolUseBehaviorAsyncCallableObject() async throws {
        let calls = SharedBox(0)
        let agent = Agent(name: "test", toolUseBehavior: .custom { _, results in
            calls.mutate { $0 += 1 }
            #expect(results.count == 2)
            return ToolsToFinalOutputResult(isFinalOutput: true, finalOutput: "async_callable")
        })
        let results = ["ignored1", "ignored2"].map { makeFunctionToolResult(agent, $0) }
        let result = try await checkForFinalOutputFromTools(agent, results, RunContextWrapper())
        #expect(result.isFinalOutput)
        #expect(result.finalOutput as? String == "async_callable")
        #expect(calls.value == 1)
    }

    @Test(arguments: [false, true])
    func testCustomToolUseBehaviorReturnContract(streamed: Bool) async throws {
        let calls = SharedBox(0)
        let model = try ScriptedModel(steps: [.output([functionCall("ping", arguments: [:], callId: "call_1")])])
        let agent = Agent(
            name: "test",
            tools: [getFunctionTool(name: "ping", returnValue: "pong")],
            model: .instance(model),
            toolUseBehavior: .custom { _, results in
                calls.mutate { $0 += 1 }
                #expect(results.count == 1)
                #expect(results[0].output as? String == "pong")
                return ToolsToFinalOutputResult(isFinalOutput: true, finalOutput: "done")
            }
        )

        let finalOutput: (any Sendable)?
        if streamed {
            let result = Runner.runStreamed(agent, input: "test")
            for try await _ in result.streamEvents() {}
            finalOutput = result.finalOutput
        } else {
            finalOutput = try await Runner.run(agent, input: "test").finalOutput
        }

        #expect(finalOutput as? String == "done")
        #expect(calls.value == 1)
        #expect(model.calls.count == 1)
    }

    // MARK: - Stop at tool names

    @Test func testToolNamesToStopAtBehavior() async throws {
        let agent = Agent(
            name: "test",
            tools: [
                getFunctionTool(name: "tool1", returnValue: "tool1_output"),
                getFunctionTool(name: "tool2", returnValue: "tool2_output"),
                getFunctionTool(name: "tool3", returnValue: "tool3_output"),
            ],
            toolUseBehavior: .stopAtTools(["tool1"])
        )

        var results = [makeFunctionToolResult(agent, "ignored1", toolName: "tool2"), makeFunctionToolResult(agent, "ignored3", toolName: "tool3")]
        var result = try await checkForFinalOutputFromTools(agent, results, RunContextWrapper())
        #expect(!result.isFinalOutput, "We should not have stopped at tool1")

        results = [
            makeFunctionToolResult(agent, "output1", toolName: "tool1"),
            makeFunctionToolResult(agent, "ignored2", toolName: "tool2"),
            makeFunctionToolResult(agent, "ignored3", toolName: "tool3"),
        ]
        result = try await checkForFinalOutputFromTools(agent, results, RunContextWrapper())
        #expect(result.isFinalOutput, "We should have stopped at tool1")
        #expect(result.finalOutput as? String == "output1")
    }

    private struct AccountArgs: Codable, Sendable {
        var accountId: String

        enum CodingKeys: String, CodingKey {
            case accountId = "account_id"
        }
    }

    @Test func testStopAtToolNamesSupportsPublicAndQualifiedNamesForNamespacedTools() async throws {
        let lookup = try FunctionTool.create(name: "lookup_account", argumentsType: AccountArgs.self) { _, args in args.accountId }
        let namespacedTool = try toolNamespace(name: "billing", description: "Billing tools", tools: [lookup])[0]
        let agent = Agent(name: "test", tools: [namespacedTool], toolUseBehavior: .stopAtTools(["lookup_account"]))

        let results = [makeFunctionToolResult(agent, "billing-output", tool: namespacedTool)]
        var result = try await checkForFinalOutputFromTools(agent, results, RunContextWrapper())
        #expect(result.isFinalOutput)
        #expect(result.finalOutput as? String == "billing-output")

        agent.toolUseBehavior = .stopAtTools(["billing.lookup_account"])
        result = try await checkForFinalOutputFromTools(agent, results, RunContextWrapper())
        #expect(result.isFinalOutput)
    }
}
