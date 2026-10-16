import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_runner.py (session input preparation and persistence).
//
// Swift: identity-dependent callback cases are skipped (porting/decisions.md D11):
// `repeated_history_keeps_equal_new_item`, `async_callback_moves_repeated_history_item`,
// `history_moved_to_new_input_stays_history`, `callback_replaces_history_item`. Non-callable
// callbacks and non-list callback results cannot be expressed. OpenAI Conversations session
// cases are pending that session's port.
extension AgentRunnerTests {
    private func user(_ content: String) -> TResponseInputItem {
        ["role": "user", "content": .string(content)]
    }

    private func contents(_ items: [TResponseInputItem]) -> [String?] {
        items.map { $0.string("content") }
    }

    private func items(_ input: RunInput) -> [TResponseInputItem] {
        input.inputItems
    }

    // MARK: - Orphan pruning

    @Test func testPrepareInputWithSessionKeepsOrphanOutputWithoutLimit() async throws {
        let historyItem: TResponseInputItem = ["type": "function_call_output", "call_id": "call_prepare", "output": "ok"]
        let session = SimpleListSession(history: [historyItem])

        let (prepared, sessionItems) = try await prepareInputWithSession("hello", session: session, sessionInputCallback: nil)

        #expect(items(prepared) == [historyItem, user("hello")])
        #expect(sessionItems == [user("hello")])
    }

    @Test func testPrepareInputWithSessionDropsLimitedOrphanHistoryFunctionCallOutputs() async throws {
        let historyItem: TResponseInputItem = ["type": "function_call_output", "call_id": "call_prepare", "output": "ok"]
        let session = SimpleListSession(history: [historyItem])

        let (prepared, sessionItems) = try await prepareInputWithSession(
            "hello", session: session, sessionInputCallback: nil, sessionSettings: SessionSettings(limit: 1)
        )

        #expect(items(prepared) == [user("hello")])
        #expect(sessionItems == [user("hello")])
    }

    @Test func testPrepareInputWithSessionPreservesNewFunctionCallOutputs() async throws {
        let newOutput: TResponseInputItem = ["type": "function_call_output", "call_id": "call_prepare", "output": "ok"]

        let (prepared, sessionItems) = try await prepareInputWithSession(
            .items([newOutput]), session: SimpleListSession(), sessionInputCallback: nil, sessionSettings: SessionSettings(limit: 1)
        )

        #expect(items(prepared) == [newOutput])
        #expect(sessionItems == [newOutput])
    }

    @Test func testPrepareInputWithSessionLeavesCustomCallbackOutputUnchanged() async throws {
        let historyOutput: TResponseInputItem = ["type": "function_call_output", "call_id": "call_callback", "output": "ok"]
        let session = SimpleListSession(history: [historyOutput])

        let (prepared, sessionItems) = try await prepareInputWithSession(
            "hello", session: session, sessionInputCallback: { history, newInput in history + newInput },
            sessionSettings: SessionSettings(limit: 1)
        )

        #expect(items(prepared) == [historyOutput, user("hello")])
        #expect(sessionItems == [user("hello")])
    }

    @Test func testPrepareInputWithSessionDropsOutputForProgramOwnedCallPrunedWithParent() async throws {
        let program: TResponseInputItem = [
            "type": "program", "call_id": "program_orphan", "code": "return await tools.lookup({});", "fingerprint": "fingerprint:orphan",
        ]
        let functionCall: TResponseInputItem = [
            "type": "function_call", "call_id": "call_orphan", "name": "lookup", "arguments": "{}",
            "caller": ["type": "program", "caller_id": "program_orphan"],
        ]
        let functionOutput: TResponseInputItem = ["type": "function_call_output", "call_id": "call_orphan", "output": "ok"]
        let session = SimpleListSession(history: [program, functionCall, functionOutput])

        let (prepared, sessionItems) = try await prepareInputWithSession(
            "hello", session: session, sessionInputCallback: nil, sessionSettings: SessionSettings(limit: 3)
        )

        #expect(items(prepared) == [user("hello")])
        #expect(sessionItems == [user("hello")])
    }

    @Test func testPrepareInputWithSessionKeepsPairedHistoryFunctionCallOutputs() async throws {
        let functionCall: TResponseInputItem = ["type": "function_call", "call_id": "call_prepare", "name": "lookup", "arguments": "{}"]
        let functionCallOutput: TResponseInputItem = ["type": "function_call_output", "call_id": "call_prepare", "output": "ok"]
        let session = SimpleListSession(history: [functionCall, functionCallOutput])

        let (prepared, sessionItems) = try await prepareInputWithSession(
            "hello", session: session, sessionInputCallback: nil, sessionSettings: SessionSettings(limit: 2)
        )

        #expect(items(prepared) == [functionCall, functionCallOutput, user("hello")])
        #expect(sessionItems == [user("hello")])
    }

    @Test func testPrepareInputWithSessionPrefersLatestFunctionCallOutput() async throws {
        let historyOutput: TResponseInputItem = ["type": "function_call_output", "call_id": "call_latest", "output": "history-output"]
        let latestOutput: TResponseInputItem = ["type": "function_call_output", "call_id": "call_latest", "output": "new-output"]
        let session = SimpleListSession(history: [historyOutput])

        let (prepared, sessionItems) = try await prepareInputWithSession(.items([latestOutput]), session: session, sessionInputCallback: nil)

        let outputs = items(prepared).filter { $0.string("type") == "function_call_output" && $0.string("call_id") == "call_latest" }
        #expect(outputs.count == 1)
        #expect(outputs.first?.string("output") == "new-output")
        #expect(sessionItems.count == 1)
        #expect(sessionItems.first?.string("output") == "new-output")
    }

    @Test func testPrepareInputWithSessionDropsOrphanFunctionCalls() async throws {
        let orphanCall: TResponseInputItem = ["type": "function_call", "call_id": "orphan_call", "name": "tool_orphan", "arguments": "{}"]
        let session = SimpleListSession(history: [orphanCall])

        let (prepared, sessionItems) = try await prepareInputWithSession("hello", session: session, sessionInputCallback: nil)

        #expect(sessionItems.count == 1)
        #expect(!items(prepared).contains { $0.string("type") == "function_call" && $0.string("call_id") == "orphan_call" })
        #expect(items(prepared).contains { $0.string("role") == "user" && $0.string("content") == "hello" })
    }

    @Test func testPrepareInputWithSessionPreservesPendingNewShellCalls() async throws {
        let orphanCall: TResponseInputItem = ["type": "function_call", "call_id": "orphan_call", "name": "tool_orphan", "arguments": "{}"]
        let pendingShellCall: TResponseInputItem = [
            "type": "shell_call", "id": "shell_1", "call_id": "manual_shell", "status": "completed",
            "action": ["commands": ["echo hi"], "timeout_ms": nil, "max_output_length": nil],
        ]
        let session = SimpleListSession(history: [orphanCall])

        let (prepared, sessionItems) = try await prepareInputWithSession(.items([pendingShellCall]), session: session, sessionInputCallback: nil)

        #expect(sessionItems == [pendingShellCall])
        #expect(!items(prepared).contains { $0.string("type") == "function_call" && $0.string("call_id") == "orphan_call" })
        #expect(items(prepared).contains { $0.string("type") == "shell_call" && $0.string("call_id") == "manual_shell" })
    }

    // MARK: - Callbacks

    @Test func testPrepareInputWithSessionUsesSyncCallback() async throws {
        let session = SimpleListSession(history: [user("hi")])

        let (prepared, sessionItems) = try await prepareInputWithSession("second", session: session, sessionInputCallback: { history, newInput in
            #expect(history.first?.string("role") == "user")
            return history + newInput
        })

        #expect(items(prepared).count == 2)
        #expect(items(prepared).last?.string("role") == "user")
        #expect(items(prepared).last?.string("content") == "second")
        #expect(sessionItems.count == 1)
        #expect(sessionItems.first?.string("role") == "user")
    }

    @Test func testPrepareInputWithSessionAwaitsAsyncCallback() async throws {
        let session = SimpleListSession(history: [user("initial")])

        let (prepared, sessionItems) = try await prepareInputWithSession("later", session: session, sessionInputCallback: { history, newInput in
            await Task.yield()
            return history + newInput
        })

        #expect(items(prepared).count == 2)
        #expect(items(prepared).first?.string("content") == "initial")
        #expect(sessionItems.count == 1)
    }

    @Test func testPrepareInputWithSessionCallbackDropsNewItems() async throws {
        let session = SimpleListSession(history: [user("history")])

        let (prepared, sessionItems) = try await prepareInputWithSession("new", session: session, sessionInputCallback: { history, _ in history })

        #expect(items(prepared) == [user("history")])
        #expect(sessionItems.isEmpty)
    }

    @Test func testPrepareInputWithSessionCallbackReordersNewItems() async throws {
        let session = SimpleListSession(history: [user("history")])
        let newInput: RunInput = .items([getTextInputItem("first"), getTextInputItem("second")])

        let (prepared, sessionItems) = try await prepareInputWithSession(newInput, session: session, sessionInputCallback: { history, newInput in
            [newInput[1], history[0], newInput[0]]
        })

        #expect(contents(items(prepared)) == ["second", "history", "first"])
        #expect(contents(sessionItems) == ["second", "first"])
    }

    @Test func testPrepareInputWithSessionCallbackAcceptsExtraItems() async throws {
        let session = SimpleListSession(history: [user("history")])
        let extraItem: TResponseInputItem = ["role": "assistant", "content": "extra"]

        let (prepared, sessionItems) = try await prepareInputWithSession("new", session: session, sessionInputCallback: { history, newInput in
            [extraItem, history[0], newInput[0]]
        })

        #expect(contents(items(prepared)) == ["extra", "history", "new"])
        #expect(contents(sessionItems) == ["extra", "new"])
    }

    @Test func testPrepareInputWithSessionIgnoresCallbackWithoutHistory() async throws {
        let session = SimpleListSession(history: [user("history")])

        let (prepared, sessionItems) = try await prepareInputWithSession(
            "new", session: session, sessionInputCallback: { _, _ in [] },
            includeHistoryInPreparedInput: false, preserveDroppedNewItems: true
        )

        #expect(contents(items(prepared)) == ["new"])
        #expect(contents(sessionItems) == ["new"])
    }

    @Test func testPrepareInputWithSessionMatchesCopiedItemsByContent() async throws {
        let session = SimpleListSession(history: [user("history")])

        let (prepared, sessionItems) = try await prepareInputWithSession("new", session: session, sessionInputCallback: { history, newInput in
            [history[0], newInput[0]]
        })

        #expect(contents(items(prepared)) == ["history", "new"])
        #expect(contents(sessionItems) == ["new"])
    }

    @Test func testPrepareInputWithSessionExtraReconstructedHistoryItemStaysNew() async throws {
        let session = SimpleListSession(history: [user("history")])

        let (_, sessionItems) = try await prepareInputWithSession("new", session: session, sessionInputCallback: { history, newInput in
            [history[0], history[0], newInput[0]]
        })

        #expect(contents(sessionItems) == ["history", "new"])
    }
}

// MARK: - Retries, persistence, and rewinds

/// A run item with a fixed input payload (upstream `_DummyRunItem`).
final class DummyRunItem: RunItemImpl, @unchecked Sendable {
    private let itemType: String

    init(_ payload: JSONObject, itemType: String = "tool_call_output_item") {
        self.itemType = itemType
        super.init(agent: Agent(name: "dummy"), rawItem: payload)
    }

    override var type: String { itemType }
    override func toInputItem() throws -> TResponseInputItem { rawItem }
}

extension AgentRunnerTests {
    private func networkRetryAgent(_ model: ScriptedModel) -> Agent {
        Agent(
            name: "test",
            model: .instance(model),
            modelSettings: ModelSettings(retry: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError()))
        )
    }

    @Test func testNonStreamedModelRetryDoesNotRewindCommittedSessionInput() async throws {
        let model = try ScriptedModel()
        try model.extend([
            .raiseError(OpenAIAPIError(message: "connection error", isConnectionError: true)),
            .output([getTextMessage("done")]),
        ])
        let session = CountingSession(history: [getTextInputItem("previous")])

        let result = try await Runner.run(networkRetryAgent(model), input: "test", session: session)
        let saved = try await session.getItems(limit: nil)

        #expect(result.finalOutput as? String == "done")
        #expect(saved.map { $0.string("role") } == ["user", "user", "assistant"])
        #expect(saved.prefix(2).map { $0.string("content") } == ["previous", "test"])
        #expect(session.popCalls == 0)
    }

    @Test func testGetNewResponseUsesAgentRetrySettings() async throws {
        let model = try ScriptedModel()
        model.setDefaultUsage(Usage(requests: 1))
        try model.extend([
            .raiseError(OpenAIAPIError(message: "connection error", isConnectionError: true)),
            .output([getTextMessage("ok")]),
        ])

        let result = try await getNewResponse(
            agent: networkRetryAgent(model),
            systemPrompt: nil,
            input: [getTextInputItem("hello")],
            outputSchema: nil,
            allTools: [],
            handoffs: [],
            hooks: RunHooks(),
            contextWrapper: RunContextWrapper(),
            runConfig: RunConfig(),
            toolUseTracker: AgentToolUseTracker(),
            serverConversationTracker: nil,
            prompt: nil,
            session: nil,
            sessionItemsToRewind: []
        )

        #expect(result.usage.requests == 2)
    }

    @Test func testSaveResultToSessionPreservesFunctionOutputs() async throws {
        let session = SimpleListSession()
        let original: TResponseInputItem = ["type": "function_call_output", "call_id": "call_original", "output": "1"]
        let runItem = DummyRunItem(["type": "function_call_output", "call_id": "call_result", "output": "2"])

        try await saveResultToSession(session, originalInput: .items([original]), newItems: [runItem])

        #expect(session.savedItems.count == 2)
        for saved in session.savedItems {
            #expect(saved.string("type") == "function_call_output")
            #expect(saved["output"] != nil)
        }
    }

    @Test func testSaveResultToSessionPrefersLatestDuplicateFunctionOutputs() async throws {
        let session = SimpleListSession()
        let original: TResponseInputItem = ["type": "function_call_output", "call_id": "call_duplicate", "output": "old-output"]
        let newItem = DummyRunItem(["type": "function_call_output", "call_id": "call_duplicate", "output": "new-output"])

        try await saveResultToSession(session, originalInput: .items([original]), newItems: [newItem])

        let duplicates = session.savedItems.filter { $0.string("type") == "function_call_output" && $0.string("call_id") == "call_duplicate" }
        #expect(duplicates.count == 1)
        #expect(duplicates.first?.string("output") == "new-output")
    }

    @Test func testSaveResultToSessionKeepsToolCallBeforeItsOutput() async throws {
        let session = SimpleListSession()
        let call: TResponseInputItem = ["type": "function_call", "call_id": "call_ordered", "name": "tool_ordered", "arguments": "{}"]
        let output: TResponseInputItem = ["type": "function_call_output", "call_id": "call_ordered", "output": "result"]
        let repeatedCall = DummyRunItem(call, itemType: "tool_call_item")

        try await saveResultToSession(session, originalInput: .items([call, output]), newItems: [repeatedCall])

        #expect(session.savedItems.map { $0.string("type") } == ["function_call", "function_call_output"])
    }

    @Test func testSaveResultToSessionKeepsLatestOutputAfterItsCall() async throws {
        let session = SimpleListSession()
        let oldOutput: TResponseInputItem = ["type": "function_call_output", "call_id": "call_ordered", "output": "old"]
        let call: TResponseInputItem = ["type": "function_call", "call_id": "call_ordered", "name": "tool_ordered", "arguments": "{}"]
        let newOutput = DummyRunItem(["type": "function_call_output", "call_id": "call_ordered", "output": "new"])

        try await saveResultToSession(session, originalInput: .items([oldOutput, call]), newItems: [newOutput])

        #expect(session.savedItems.map { $0.string("type") } == ["function_call", "function_call_output"])
        #expect(session.savedItems.last?.string("output") == "new")
    }

    @Test func testRewindHandlesIdStrippedSessions() async throws {
        let session = IdStrippingSession()
        let item: TResponseInputItem = ["id": "message-1", "type": "message", "content": "hello"]
        try await session.addItems([item])

        await rewindSessionItems(session, [item])

        #expect(session.popCalls == 1)
        #expect(session.savedItems.isEmpty)
    }

    @Test(arguments: [true, false])
    func testRewindDebugLoggingRespectsModelAndToolPolicies(redacted: Bool) async throws {
        let secret = "SECRET_REWIND_SESSION_CONTENT"
        let session = IdStrippingSession()
        let item: TResponseInputItem = ["id": "message-1", "type": "message", "role": "user", "content": .string(secret)]
        try await session.addItems([item])

        await AgentsDebug.withOverrides(dontLogModelData: redacted, dontLogToolData: redacted) {
            await captureLogs(level: .debug) { capture in
                await rewindSessionItems(session, [item])
                let logged = capture.records.filter { $0.level == .debug }.map(\.message).joined(separator: "\n")
                #expect(!logged.contains(secret) == redacted)
            }
        }
    }

    @Test func testRewindFailureUsesPlaceholderFreeSharedLoggerMessage() async throws {
        final class FailingTailSession: SimpleListSession, @unchecked Sendable {
            override func getItems(limit: Int?) async throws -> [TResponseInputItem] {
                throw TestError("tail failure")
            }
        }
        let item: TResponseInputItem = ["type": "message", "role": "user", "content": "hi"]
        let session = FailingTailSession(history: [item])

        await captureLogs(level: .warning) { capture in
            await rewindSessionItems(session, [item])
            #expect(capture.messages.contains { $0.hasPrefix("Failed to rewind session item") })
        }
    }

    @Test func testRewindSkipsMismatchedTailSuffix() async throws {
        let target: TResponseInputItem = ["type": "message", "role": "user", "content": "target"]
        let unrelated: TResponseInputItem = ["type": "message", "role": "user", "content": "unrelated tail item"]
        let session = CountingSession(history: [target, unrelated])

        await rewindSessionItems(session, [target])

        #expect(session.popCalls == 0)
        #expect(session.savedItems == [target, unrelated])
    }

    @Test func testRewindPreservesUnrelatedTailItemsWhenServerTrackerCleanupRuns() async throws {
        let knownServerItem: TResponseInputItem = ["id": "msg_server_1", "type": "message", "role": "assistant", "content": "server item"]
        let unrelated: TResponseInputItem = ["type": "message", "role": "user", "content": "unrelated tail item"]
        let target: TResponseInputItem = ["type": "message", "role": "user", "content": "target"]
        let session = CountingSession(history: [knownServerItem, unrelated, target])
        let tracker = OpenAIServerConversationTracker()
        tracker.serverItemIds.insert("msg_server_1")

        await rewindSessionItems(session, [target], tracker)

        #expect(session.popCalls == 1)
        #expect(session.savedItems == [knownServerItem, unrelated])
    }

    @Test func testRewindStripsOnlyRetryOwnedTailItemsBeforeKnownServerItem() async throws {
        let knownServerItem: TResponseInputItem = ["id": "msg_server_1", "type": "message", "role": "assistant", "content": "server item"]
        let retryOwnedTail: TResponseInputItem = ["type": "message", "role": "user", "content": "retry-owned local item"]
        let target: TResponseInputItem = ["type": "message", "role": "user", "content": "target"]
        let session = CountingSession(history: [knownServerItem, retryOwnedTail, target])
        let tracker = OpenAIServerConversationTracker()
        tracker.serverItemIds.insert("msg_server_1")
        tracker.sentItemFingerprints.insert(fingerprintInputItem(retryOwnedTail))

        await rewindSessionItems(session, [target], tracker)

        #expect(session.popCalls == 2)
        #expect(session.savedItems == [knownServerItem])
    }

    @Test func testCollectRetryOwnedTailSerializationsReturnsEmptyForEmptySession() {
        #expect(collectRetryOwnedTailSerializations([], serverTracker: OpenAIServerConversationTracker(), ignoreIds: false).isEmpty)
    }

    @Test func testSaveResultToSessionDoesNotIncrementCounterWhenNothingSaved() async throws {
        let session = SimpleListSession()
        let agent = Agent(name: "agent", model: .instance(try ScriptedModel()))
        let approval = ToolApprovalItem(agent: agent, rawItem: ["type": "function_call", "call_id": "call-1", "name": "tool"])
        let runState = RunState(context: RunContextWrapper(), originalInput: "input", startingAgent: agent, maxTurns: 1)

        try await saveResultToSession(session, originalInput: .items([]), newItems: [approval], runState: runState)

        #expect(runState.currentTurnPersistedItemCount == 0)
        #expect(session.savedItems.isEmpty)
    }

    @Test func testSaveResultToSessionReturnsCountAndUpdatesState() async throws {
        let session = SimpleListSession()
        let agent = Agent(name: "agent", model: .instance(try ScriptedModel()))
        let runState = RunState(context: RunContextWrapper(), originalInput: "input", startingAgent: agent, maxTurns: 1)
        let approval = ToolApprovalItem(agent: agent, rawItem: ["type": "function_call", "call_id": "call-2", "name": "tool"])
        let output = DummyRunItem(["type": "message", "role": "assistant", "content": "ok"], itemType: "message_output_item")

        let savedCount = try await saveResultToSession(session, originalInput: .items([]), newItems: [output, approval], runState: runState)

        #expect(savedCount == 1)
        #expect(runState.currentTurnPersistedItemCount == 1)
        #expect(session.savedItems.count == 1)
        #expect(session.savedItems.first?.string("content") == "ok")
    }

    @Test func testSaveResultToSessionOmitsReasoningIdsWhenPolicyIsOmit() async throws {
        let session = SimpleListSession()
        let agent = Agent(name: "agent", model: .instance(try ScriptedModel()))
        let runState = RunState(context: RunContextWrapper(), originalInput: "input", startingAgent: agent, maxTurns: 1)
        runState.reasoningItemIdPolicy = .omit
        let reasoning = ReasoningItem(agent: agent, rawItem: ["type": "reasoning", "id": "rs_stream", "summary": []])

        let savedCount = try await saveResultToSession(session, originalInput: .items([]), newItems: [reasoning], runState: runState)

        #expect(savedCount == 1)
        #expect(session.savedItems.count == 1)
        #expect(session.savedItems.first?.string("type") == "reasoning")
        #expect(session.savedItems.first?["id"] == nil)
    }
}
