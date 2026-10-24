import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_tool_approval_call_id_reuse.py.
//
// Swift: `test_web_search_source_schema_drift_does_not_warn_during_invocation_lookup`,
// `test_invocation_lookup_preserves_legacy_model_dump_signature`, and
// `test_invocation_lookup_propagates_internal_model_dump_type_error` exercise pydantic
// `model_dump` handling; Swift raw items are already JSON, so they are not ported.
@Suite struct ToolApprovalCallIdReuseTests {
    // MARK: - Canonical identity

    @Test func testCanonicalShellIdentityIgnoresStrippedProviderMetadata() {
        let providerCall: JSONObject = [
            "type": "shell_call",
            "call_id": "shell_0",
            "action": ["commands": ["echo safe"]],
            "created_by": "server",
        ]
        var persistedCall = providerCall
        persistedCall.removeValue(forKey: "created_by")

        #expect(toolInvocationIdentity(providerCall) == toolInvocationIdentity(persistedCall))
    }

    @Test func testCanonicalShellIdentityTreatsOptionalNullsAsOmitted() {
        let providerCall: JSONObject = [
            "type": "shell_call",
            "call_id": "shell_0",
            "action": [
                "commands": ["echo safe"],
                "max_output_length": nil,
                "timeout_ms": nil,
            ],
            "environment": nil,
        ]
        var normalizedCall = providerCall
        normalizedCall["action"] = ["commands": ["echo safe"]]
        normalizedCall.removeValue(forKey: "environment")

        #expect(toolInvocationIdentity(providerCall) == toolInvocationIdentity(normalizedCall))
    }

    @Test func testCanonicalFunctionIdentityPreservesNullArgumentValues() {
        let callWithNull: JSONObject = [
            "type": "function_call",
            "call_id": "call_0",
            "name": "lookup",
            "arguments": #"{"value": null}"#,
        ]
        let callWithoutValue: JSONObject = [
            "type": "function_call",
            "call_id": "call_0",
            "name": "lookup",
            "arguments": "{}",
        ]

        #expect(toolInvocationIdentity(callWithNull) != toolInvocationIdentity(callWithoutValue))
    }

    @Test(arguments: [
        ["type": "custom_tool_call", "name": "raw_editor", "call_id": "call_0"],
        ["type": "computer_call", "call_id": "call_0"],
        ["type": "local_shell_call", "call_id": "call_0"],
        ["type": "shell_call", "call_id": "call_0"],
        ["type": "apply_patch_call", "call_id": "call_0"],
    ] as [JSONObject])
    func testCanonicalIdentityRequiresEachToolPayload(rawItem: JSONObject) {
        #expect(toolInvocationIdentity(rawItem) == nil)
    }

    // Swift: no upstream counterpart. The digests below come from the upstream writer
    // (tests/fixtures/run_state/features/v1_15_canonical_invocation_identity.json), so they pin the
    // byte-for-byte hashing that lets Swift validate Python-written invocation ledgers.
    @Test func testCanonicalIdentityDigestsMatchUpstreamWriter() throws {
        let call: JSONObject = [
            "type": "function_call",
            "name": "lookup_account",
            "call_id": "function-request-1",
            "status": "completed",
            "arguments": #"{"account_id":"account-1"}"#,
        ]

        let identity = try #require(toolInvocationIdentity(call))

        #expect(identity.type == "function_call")
        #expect(identity.callId == "function-request-1")
        #expect(identity.approvalScope == "73dc418185c1335b2f8ae49c817524e389e0f47c666a4578a8fb9d921756bbd4")
        #expect(identity.fingerprint == "1f6f2dba51c70f18ab8f13f23bc34068e31ce849b6d7323575a0a99b7303a356")
    }

    // MARK: - Runner

    enum RunMode: String, CaseIterable, Sendable {
        case nonStreamed = "non_streamed"
        case streamed
    }

    enum RunInputValue {
        case text(String)
        case state(RunState)
    }

    final class ScriptedProvider: ModelProvider, @unchecked Sendable {
        let model: ScriptedModel
        init(_ model: ScriptedModel) { self.model = model }
        func getModel(_ modelName: String?) throws -> any Model {
            #expect(modelName == "scripted-provider-model")
            return model
        }
        func aclose() async throws {}
    }

    /// The run result fields these tests inspect, from either a streamed or a non-streamed run.
    struct RunOutcome {
        var finalOutput: (any Sendable)?
        var interruptions: [ToolApprovalItem]
        var newItems: [RunItem]
        var state: () -> RunState
    }

    func run(
        _ agent: Agent,
        _ input: RunInputValue,
        runConfig: RunConfig = RunConfig(),
        mode: RunMode,
        hooks: RunHooks? = nil,
        context: RunContextWrapper? = nil,
        events: SharedBox<[StreamEvent]>? = nil
    ) async throws -> RunOutcome {
        if mode == .nonStreamed {
            let result: RunResult
            switch input {
            case .text(let text): result = try await Runner.run(agent, input: .text(text), context: context, hooks: hooks, runConfig: runConfig)
            case .state(let state): result = try await Runner.run(agent, state: state, hooks: hooks, runConfig: runConfig)
            }
            return RunOutcome(finalOutput: result.finalOutput, interruptions: result.interruptions, newItems: result.newItems, state: { result.toState() })
        }
        let result: RunResultStreaming
        switch input {
        case .text(let text): result = try Runner.runStreamed(agent, input: .text(text), context: context, hooks: hooks, runConfig: runConfig)
        case .state(let state): result = try Runner.runStreamed(agent, state: state, hooks: hooks, runConfig: runConfig)
        }
        for try await event in result.streamEvents() {
            events?.mutate { $0.append(event) }
        }
        return RunOutcome(finalOutput: result.finalOutput, interruptions: result.interruptions, newItems: result.newItems, state: { result.toState() })
    }

    func toolCalledEventNames(_ events: SharedBox<[StreamEvent]>) -> [String] {
        events.value.compactMap { event in
            guard case .runItem(let item) = event, item.name == "tool_called" else { return nil }
            return item.name
        }
    }

    struct ValueArgs: Codable, Sendable { var value: String }

    func buildScenario(secondCallId: String, secondValue: String) throws -> (Agent, RunConfig, SharedBox<[String]>) {
        let executed = SharedBox<[String]>([])
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
            executed.mutate { $0.append(args.value) }
            return args.value
        }
        let model = try ScriptedModel()
        try model.extend([
            .output([getFunctionToolCall("record_value", arguments: #"{"value": "safe"}"#, callId: "call_0")]),
            .output([getFunctionToolCall("record_value", arguments: #"{"value": "\#(secondValue)"}"#, callId: secondCallId)]),
            .output([getTextMessage("done")]),
        ])
        let agent = Agent(name: "custom-provider-agent", tools: [recordValue], model: .name("scripted-provider-model"))
        return (agent, RunConfig(modelProvider: ScriptedProvider(model)), executed)
    }

    @Test func testEmptyCustomToolCallIdFailsBeforeApprovalOrExecution() async throws {
        let callbacks = SharedBox<[String]>([])
        let executed = SharedBox<[String]>([])
        let tool = try CustomTool(
            name: "raw_editor",
            description: "Edit raw text.",
            onInvokeTool: { _, rawInput in
                executed.mutate { $0.append(rawInput) }
                return rawInput
            },
            format: ["type": "text"],
            needsApproval: .required(true),
            onApproval: { _, _ in
                callbacks.mutate { $0.append("approval") }
                return ToolOnApprovalResult(approve: true)
            }
        )
        let call: JSONObject = ["type": "custom_tool_call", "name": .string(tool.name), "call_id": "", "input": "changed"]
        let model = try ScriptedModel(steps: [.output([call])])
        let agent = Agent(name: "agent", tools: [tool], model: .instance(model))

        await expectErrorAsync(ModelBehaviorError.self, containing: "non-empty string call ID") {
            _ = try await Runner.run(agent, input: "run it")
        }

        #expect(callbacks.value.isEmpty)
        #expect(executed.value.isEmpty)
    }

    @Test func testEmptyUnresolvedFunctionCallIdFailsBeforeErrorFormatter() async throws {
        let formatterCalls = SharedBox<[String]>([])
        let call: JSONObject = ["id": "item_0", "type": "function_call", "name": "missing", "arguments": "{}", "call_id": ""]
        let model = try ScriptedModel(steps: [.output([call])])
        let agent = Agent(name: "agent", model: .instance(model))
        let runConfig = RunConfig(
            toolErrorFormatter: { args in
                formatterCalls.mutate { $0.append(args.toolName) }
                return "error"
            },
            toolNotFoundBehavior: .returnErrorToModel
        )

        await expectErrorAsync(ModelBehaviorError.self, containing: "non-empty string call ID") {
            _ = try await Runner.run(agent, input: "run it", runConfig: runConfig)
        }

        #expect(formatterCalls.value.isEmpty)
    }

    @Test func testBoundCallIdWithMissingArgumentsFailsBeforeErrorFormatter() async throws {
        let executed = SharedBox<[String]>([])
        let formatterCalls = SharedBox<[String]>([])
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self) { context, args in
            executed.mutate { $0.append(args.value) }
            context.toolStateBox.mutate { _ = $0.restoredUnboundApprovalCallIds.insert("shared") }
            return args.value
        }
        let model = try ScriptedModel()
        try model.extend([
            .output([getFunctionToolCall("record_value", arguments: #"{"value": "safe"}"#, callId: "shared")]),
            .output([["type": "function_call", "name": "missing", "call_id": "shared"]]),
        ])
        let agent = Agent(name: "agent", tools: [recordValue], model: .instance(model))
        let runConfig = RunConfig(
            toolErrorFormatter: { args in
                formatterCalls.mutate { $0.append(args.toolName) }
                return "error"
            },
            toolNotFoundBehavior: .returnErrorToModel
        )

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Runner.run(agent, input: "run it", runConfig: runConfig)
        }

        #expect(executed.value == ["safe"])
        #expect(formatterCalls.value.isEmpty)
    }

    @Test func testEmptyHandoffCallIdFailsBeforeHandoffCallback() async throws {
        let handoffCalls = SharedBox<[String]>([])
        let target = Agent(name: "target")
        let route = handoff(target, toolNameOverride: "route", onHandoff: { _ in handoffCalls.mutate { $0.append("route") } })
        let call: JSONObject = ["id": "item_0", "type": "function_call", "name": "route", "arguments": "{}", "call_id": ""]
        let model = try ScriptedModel(steps: [.output([call])])
        let agent = Agent(name: "agent", handoffs: [.handoff(route)], model: .instance(model))

        await expectErrorAsync(ModelBehaviorError.self, containing: "non-empty string call ID") {
            _ = try await Runner.run(agent, input: "route it")
        }

        #expect(handoffCalls.value.isEmpty)
    }

    final class FailingHandoffHooks: RunHooks, @unchecked Sendable {
        let hookCalls = SharedBox<[String]>([])

        override func onHandoff(context: RunContextWrapper, fromAgent: AgentBase, toAgent: AgentBase) async throws {
            hookCalls.mutate { $0.append(toAgent.name) }
            throw TestError("handoff hook failed")
        }
    }

    @Test func testFailedHandoffHookDoesNotCommitOutputOrRepeatCallback() async throws {
        let target = Agent(name: "target")
        let call = getHandoffToolCall(target, callId: "handoff_0")
        let model = try ScriptedModel(steps: [.output([call])])
        try model.enqueue(.output([call]))
        let agent = Agent(name: "source", handoffs: [.agent(target)], model: .instance(model))
        let context = RunContextWrapper(context: ())
        let hooks = FailingHandoffHooks()

        await expectErrorAsync(TestError.self, containing: "handoff hook failed") {
            _ = try await Runner.run(agent, input: "handoff", context: context, hooks: hooks)
        }

        #expect(
            try context.toolInvocationStatus(call, invocationRole: "handoff")
                == ToolInvocationStatus(invocationType: "function_call", callId: "handoff_0", completed: false, executed: true)
        )
        await expectErrorAsync(ModelBehaviorError.self, containing: "already executed") {
            _ = try await Runner.run(agent, input: "handoff", context: context, hooks: hooks)
        }

        #expect(hooks.hookCalls.value == ["target"])
    }

    @Test(arguments: RunMode.allCases)
    func testChangedArgumentsUnderReusedCallIdFailBeforeSecondSideEffect(mode: RunMode) async throws {
        let (agent, runConfig, executed) = try buildScenario(secondCallId: "call_0", secondValue: "changed")

        let first = try await run(agent, .text("record a value"), runConfig: runConfig, mode: mode)
        #expect(first.interruptions.count == 1)
        let state = first.state()
        try state.approve(first.interruptions[0])

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await run(agent, .state(state), runConfig: runConfig, mode: mode)
        }

        #expect(executed.value == ["safe"])
    }

    @Test(arguments: RunMode.allCases)
    func testExactReplayReusesCommittedOutputWithoutExecutingAgain(mode: RunMode) async throws {
        let (agent, runConfig, executed) = try buildScenario(secondCallId: "call_0", secondValue: "safe")

        let first = try await run(agent, .text("record a value"), runConfig: runConfig, mode: mode)
        let state = first.state()
        try state.approve(first.interruptions[0])

        let resumed = try await run(agent, .state(state), runConfig: runConfig, mode: mode)

        #expect(resumed.finalOutput as? String == "done")
        #expect(resumed.interruptions.isEmpty)
        #expect(executed.value == ["safe"])
    }

    @Test(arguments: RunMode.allCases)
    func testNonApprovalIdenticalSiblingsExecuteOnce(mode: RunMode) async throws {
        let executed = SharedBox<[String]>([])
        let events = SharedBox<[StreamEvent]>([])
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self) { _, args in
            executed.mutate { $0.append(args.value) }
            return args.value
        }
        let duplicate = getFunctionToolCall("record_value", arguments: #"{"value":"safe"}"#, callId: "call_0")
        let model = try ScriptedModel(steps: [.output([duplicate, duplicate])])
        try model.enqueue(.output([getTextMessage("done")]))
        let agent = Agent(name: "agent", tools: [recordValue], model: .instance(model))

        let result = try await run(agent, .text("record a value"), mode: mode, events: events)

        #expect(result.finalOutput as? String == "done")
        #expect(executed.value == ["safe"])
        if mode == .streamed {
            #expect(toolCalledEventNames(events) == ["tool_called"])
        }
    }

    @Test(arguments: RunMode.allCases)
    func testNonApprovalCompletedReplayDoesNotExecuteAgain(mode: RunMode) async throws {
        let executed = SharedBox<[String]>([])
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self) { _, args in
            executed.mutate { $0.append(args.value) }
            return args.value
        }
        let model = try ScriptedModel()
        try model.extend([
            .output([getFunctionToolCall("record_value", arguments: #"{"value":"safe"}"#, callId: "call_0")]),
            .output([getFunctionToolCall("record_value", arguments: #"{ "value" : "safe" }"#, callId: "call_0")]),
            .output([getTextMessage("done")]),
        ])
        let agent = Agent(name: "agent", tools: [recordValue], model: .instance(model))

        let result = try await run(agent, .text("record a value"), mode: mode)

        #expect(result.finalOutput as? String == "done")
        #expect(executed.value == ["safe"])
    }

    final class LLMEndRecordingHooks: RunHooks, @unchecked Sendable {
        let llmEndCalls = SharedBox(0)

        override func onLLMEnd(context: RunContextWrapper, agent: Agent, response: ModelResponse) async throws {
            llmEndCalls.mutate { $0 += 1 }
        }
    }

    @Test(arguments: RunMode.allCases)
    func testNonApprovalChangedCompletedCallIdFailsBeforeExecution(mode: RunMode) async throws {
        let executed = SharedBox<[String]>([])
        let events = SharedBox<[StreamEvent]>([])
        let hooks = LLMEndRecordingHooks()
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self) { _, args in
            executed.mutate { $0.append(args.value) }
            return args.value
        }
        let model = try ScriptedModel()
        try model.extend([
            .output([getFunctionToolCall("record_value", arguments: #"{"value":"safe"}"#, callId: "call_0")]),
            .output([getFunctionToolCall("record_value", arguments: #"{"value":"changed"}"#, callId: "call_0")]),
        ])
        let agent = Agent(name: "agent", tools: [recordValue], model: .instance(model))

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await run(agent, .text("record a value"), mode: mode, hooks: hooks, events: events)
        }

        #expect(executed.value == ["safe"])
        #expect(hooks.llmEndCalls.value == 1)
        if mode == .streamed {
            #expect(toolCalledEventNames(events) == ["tool_called"])
        }
    }
}
