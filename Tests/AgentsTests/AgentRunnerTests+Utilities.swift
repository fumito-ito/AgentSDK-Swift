import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_runner.py (input normalization, conversation tracking, and
// function-tool queueing).
//
// Swift: `test_fingerprint_input_item_returns_none_when_model_dump_fails` and the
// `execute_approved_tools` cases are skipped: input items are always JSON values, and
// `execute_approved_tools` is an upstream helper the runner itself no longer calls.
extension AgentRunnerTests {
    @Test func testDropOrphanFunctionCallsRemovesOrphans() {
        let items: [TResponseInputItem] = [
            ["type": "function_call", "call_id": "call_orphan", "name": "tool_one", "arguments": "{}"],
            ["type": "message", "role": "user", "content": "hello"],
            ["type": "function_call", "call_id": "call_keep", "name": "tool_keep", "arguments": "{}"],
            ["type": "function_call_output", "call_id": "call_keep", "output": "done"],
            ["type": "shell_call", "call_id": "shell_orphan"],
            ["type": "shell_call", "call_id": "shell_keep"],
            ["type": "shell_call_output", "call_id": "shell_keep", "output": []],
            ["type": "apply_patch_call", "call_id": "patch_orphan"],
            ["type": "apply_patch_call", "call_id": "patch_keep"],
            ["type": "apply_patch_call_output", "call_id": "patch_keep", "output": "done"],
            ["type": "computer_call", "call_id": "computer_orphan"],
            ["type": "computer_call", "call_id": "computer_keep"],
            ["type": "computer_call_output", "call_id": "computer_keep", "output": [:]],
            ["type": "local_shell_call", "call_id": "local_shell_orphan"],
            ["type": "local_shell_call", "call_id": "local_shell_keep"],
            ["type": "local_shell_call_output", "call_id": "local_shell_keep", "output": ["stdout": "", "stderr": "", "outcome": [:]]],
        ]

        let filtered = dropOrphanFunctionCalls(items)

        let orphanIds: Set<String> = ["call_orphan", "shell_orphan", "patch_orphan", "computer_orphan", "local_shell_orphan"]
        #expect(!filtered.contains { orphanIds.contains($0.string("call_id") ?? "") })
        func hasCall(_ type: String, _ callId: String) -> Bool {
            filtered.contains { $0.string("type") == type && $0.string("call_id") == callId }
        }
        #expect(hasCall("function_call", "call_keep"))
        #expect(hasCall("shell_call", "shell_keep"))
        #expect(hasCall("apply_patch_call", "patch_keep"))
        #expect(hasCall("computer_call", "computer_keep"))
        #expect(hasCall("local_shell_call", "local_shell_keep"))
    }

    @Test func testNormalizeResumedInputDropsOrphanFunctionCalls() {
        let raw: [TResponseInputItem] = [
            ["type": "function_call", "call_id": "orphan_call", "name": "tool_orphan", "arguments": "{}"],
            ["type": "function_call", "call_id": "paired_call", "name": "tool_paired", "arguments": "{}"],
            ["type": "function_call_output", "call_id": "paired_call", "output": "ok"],
        ]

        let callIds = normalizeResumedInput(.items(raw)).inputItems.filter { $0.string("type") == "function_call" }.map { $0.string("call_id") }

        #expect(!callIds.contains("orphan_call"))
        #expect(callIds.contains("paired_call"))
    }

    @Test func testNormalizeResumedInputDropsOrphanToolSearchCalls() {
        let raw: [TResponseInputItem] = [
            ["type": "tool_search_call", "call_id": "orphan_search", "arguments": ["query": "orphan"], "execution": "server", "status": "completed"],
            ["type": "tool_search_call", "call_id": "paired_search", "arguments": ["query": "paired"], "execution": "server", "status": "completed"],
            ["type": "tool_search_output", "call_id": "paired_search", "execution": "server", "status": "completed", "tools": []],
        ]

        let callIds = normalizeResumedInput(.items(raw)).inputItems.filter { $0.string("type") == "tool_search_call" }.map { $0.string("call_id") }

        #expect(!callIds.contains("orphan_search"))
        #expect(callIds.contains("paired_search"))
    }

    @Test func testNormalizeResumedInputPreservesHostedToolSearchPairWithoutCallIds() {
        let raw: [TResponseInputItem] = [
            ["type": "tool_search_call", "call_id": nil, "arguments": ["query": "paired"], "execution": "server", "status": "completed"],
            ["type": "tool_search_output", "call_id": nil, "execution": "server", "status": "completed", "tools": []],
        ]

        #expect(normalizeResumedInput(.items(raw)).inputItems.map { $0.string("type") } == ["tool_search_call", "tool_search_output"])
    }

    @Test func testNormalizeResumedInputMatchesLatestAnonymousToolSearchCall() {
        let raw: [TResponseInputItem] = [
            ["type": "tool_search_call", "call_id": nil, "arguments": ["query": "orphan"], "execution": "server", "status": "completed"],
            ["type": "tool_search_call", "call_id": nil, "arguments": ["query": "paired"], "execution": "server", "status": "completed"],
            ["type": "tool_search_output", "call_id": nil, "execution": "server", "status": "completed", "tools": []],
        ]

        let normalized = normalizeResumedInput(.items(raw)).inputItems

        #expect(normalized.map { $0.string("type") } == ["tool_search_call", "tool_search_output"])
        #expect(normalized.first?["arguments"] == ["query": "paired"])
    }

    @Test func testNormalizeInputItemsForAPIPreservesProviderData() {
        let items: [TResponseInputItem] = [
            ["type": "function_call_output", "call_id": "call_norm", "status": "completed", "output": "out", "provider_data": ["trace": "keep"]],
            ["type": "message", "role": "user", "content": "hi", "provider_data": ["trace": "remove"]],
        ]

        let normalized = normalizeInputItemsForAPI(items)

        #expect(normalized[0].string("type") == "function_call_output")
        #expect(normalized[0].string("call_id") == "call_norm")
        #expect(normalized[0]["provider_data"] == ["trace": "keep"])
        #expect(normalized[1].string("role") == "user")
        #expect(normalized[1]["provider_data"] == ["trace": "remove"])
    }

    @Test func testServerConversationTrackerTracksPreviousResponseId() {
        let tracker = OpenAIServerConversationTracker(previousResponseId: "resp_a")
        tracker.trackServerItems(ModelResponse(output: [getTextMessage("hello")], usage: Usage(), responseId: "resp_b"))

        #expect(tracker.previousResponseId == "resp_b")
        #expect(tracker.serverItemIds.count == 1)
    }

    // MARK: - Function tool queue

    @Test func testFunctionToolQueuePreservesFIFOAcrossMultipleRefills() async throws {
        struct ValueArgs: Codable, Sendable { var value: Int }
        let started = SharedBox<[Int]>([])
        let completed = SharedBox<[Int]>([])
        let startedEvents = (0..<7).map { _ in AsyncEvent() }
        let releaseEvents = (0..<7).map { _ in AsyncEvent() }
        let trackedTool = try FunctionTool.create(name: "tracked_tool", argumentsType: ValueArgs.self) { _, args in
            started.mutate { $0.append(args.value) }
            startedEvents[args.value].set()
            try await releaseEvents[args.value].wait()
            completed.mutate { $0.append(args.value) }
            return "ok-\(args.value)"
        }
        let model = try ScriptedModel()
        try model.enqueue(.output((0..<7).map { getFunctionToolCall("tracked_tool", arguments: #"{"value": \#($0)}"#, callId: String($0)) }))
        try model.enqueue(.output([getTextMessage("done")]))
        let agent = Agent(name: "test", tools: [trackedTool], model: .instance(model))
        let runConfig = RunConfig(toolExecution: try ToolExecutionConfig(maxFunctionToolConcurrency: 2))

        let runTask = Task { try await Runner.run(agent, input: "run tools", runConfig: runConfig) }
        defer { runTask.cancel() }

        try await startedEvents[1].wait(timeout: 5)
        #expect(started.value == [0, 1])
        for value in 1...5 {
            releaseEvents[value].set()
            try await startedEvents[value + 1].wait(timeout: 5)
            #expect(started.value == Array(0...(value + 1)))
            #expect(completed.value == Array(1...value))
        }
        releaseEvents[6].set()
        releaseEvents[0].set()
        let result = try await withTestTimeout(5) { try await runTask.value }

        #expect(result.finalOutput as? String == "done")
        #expect(result.newItems.compactMap { ($0 as? ToolCallOutputItem)?.output as? String } == (0..<7).map { "ok-\($0)" })
    }
}
