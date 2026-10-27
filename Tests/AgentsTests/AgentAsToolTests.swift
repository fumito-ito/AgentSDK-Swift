import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_as_tool.py: the nested approval and resume cases.
//
// Swift: upstream drives most of these cases by monkeypatching `Runner.run` and storing duck-typed
// pending results. Swift cannot patch the runner per test, so these tests store a real
// interrupted nested `RunState` and let the nested agent resume with a scripted model. Where the
// stored state cannot be resumed (a bare approval item without a model response), the test
// checks the decision that reaches the nested context, which is what the patched runner asserted.
//
// Pending in porting/manifest.json: the rest of the module (derived names and collisions,
// enablement, output extraction, run config inheritance, structured input, streaming, failure
// handling, and the `on_stream_max_pending_events` backlog).
//
// Not ported:
// - `test_agent_as_tool_preserves_scope_for_nested_tool_context` and
//   `test_agent_as_tool_preserves_scope_for_nested_run_context_wrapper` read the scope inside the
//   patched `Runner.run`; a real nested run replaces it with a fresh scope, as upstream's does.
//   `Agent.asTool()` only receives a `ToolContext` in Swift, so the wrapper case does not apply.
@Suite struct AgentAsToolTests {
    private static let outerArguments = #"{"input": "hello"}"#

    /// A `ToolContext` for an outer `outer_tool` call with its own agent-tool scope.
    private func makeOuterToolContext(_ toolCall: TResponseOutputItem) -> ToolContext {
        let context = ToolContext(
            context: (),
            toolName: toolCall.string("name") ?? "",
            toolCallId: toolCall.string("call_id") ?? "",
            toolArguments: toolCall.string("arguments") ?? "",
            toolCall: toolCall
        )
        setAgentToolStateScope(context, AgentToolStateScope())
        return context
    }

    /// Runs `agent` until its `inner_tool` call is interrupted for approval.
    private func interruptedNestedState(_ agent: Agent, model: ScriptedModel, finalText: String) async throws -> (RunState, ToolApprovalItem) {
        try model.extend([
            .output([makeFunctionToolCall("inner_tool", callId: "inner-1")], usage: Usage(requests: 1)),
            .output([getTextMessage(finalText)], usage: Usage(requests: 1)),
        ])
        let interrupted = try await Runner.run(agent, input: "hello")
        let approval = try #require(interrupted.interruptions.first)
        return (interrupted.toState(), approval)
    }

    // MARK: - Nested decisions

    @Test func testAgentAsToolRejectedNestedApprovalResumesRun() async throws {
        let innerTool = try FunctionTool.create(name: "inner_tool", needsApproval: .always) { _ in "inner" }
        let model = try ScriptedModel()
        let agent = makeAgent(model: model, tools: [innerTool], name: "outer")
        let (resumeState, approvalItem) = try await interruptedNestedState(agent, model: model, finalText: "rejected")
        let toolCall = makeFunctionToolCall("outer_tool", callId: "outer-1", arguments: Self.outerArguments)
        let toolContext = makeOuterToolContext(toolCall)
        recordAgentToolRunResult(toolCall, .serialized(resumeState, interruptions: [approvalItem]), scope: toolContext.agentToolStateScope)

        try toolContext.rejectTool(approvalItem)
        #expect(resumeState.context.isToolApproved(toolName: "inner_tool", callId: "inner-1") == false)

        let resumedContexts = SharedBox<[ObjectIdentifier]>([])
        let tool = try agent.asTool(
            toolName: "outer_tool",
            toolDescription: "Outer agent tool",
            customOutputExtractor: { result in
                resumedContexts.mutate { $0.append(ObjectIdentifier(result.contextWrapper)) }
                return "from_resume"
            }
        )

        let output = try await tool.onInvokeTool(toolContext, Self.outerArguments)

        #expect(output as? String == "from_resume")
        #expect(resumedContexts.value == [ObjectIdentifier(resumeState.context)])
    }

    @Test func testAgentAsToolCachedResumeRebindsUsageToOuterContext() async throws {
        let innerTool = try FunctionTool.create(name: "inner_tool", needsApproval: .always) { _ in "inner" }
        let model = try ScriptedModel()
        let agent = makeAgent(model: model, tools: [innerTool], name: "outer")
        let (resumeState, approvalItem) = try await interruptedNestedState(agent, model: model, finalText: "done")
        try resumeState.approve(approvalItem)
        let toolCall = makeFunctionToolCall("outer_tool", callId: "outer-1", arguments: Self.outerArguments)
        let toolContext = makeOuterToolContext(toolCall)
        toolContext.usage.requests = 3
        #expect(resumeState.context.usageBox !== toolContext.usageBox)

        // Store it as an in-flight resume checkpoint so the cached resume branch runs.
        recordAgentToolResumeState(toolCall, resumeState, scope: toolContext.agentToolStateScope)

        let tool = try agent.asTool(
            toolName: "outer_tool", toolDescription: "Outer agent tool", customOutputExtractor: { _ in "from_resume" }
        )

        let output = try await tool.onInvokeTool(toolContext, Self.outerArguments)

        #expect(output as? String == "from_resume")
        #expect(resumeState.context.usageBox === toolContext.usageBox)
        // The nested post-resume model turn accrues on the outer usage.
        #expect(toolContext.usage.requests == 4)
    }

    @Test func testAgentAsToolWrappedHostedMCPExactDecisionResumesRun() throws {
        let agent = Agent(name: "outer")
        let toolCall = makeFunctionToolCall("outer_tool", callId: "outer-1", arguments: Self.outerArguments)
        let toolContext = makeOuterToolContext(toolCall)
        let approvalItem = ToolApprovalItem(
            agent: agent,
            rawItem: [
                "type": "hosted_tool_call",
                "provider_data": [
                    "type": "mcp_approval_request",
                    "id": "inner-1",
                    "name": "lookup_account",
                    "server_label": "accounts",
                    "arguments": "{}",
                ],
            ],
            toolName: "lookup_account"
        )
        let resumeState = makeStateWithInterruptions(agent, [approvalItem])
        recordAgentToolRunResult(toolCall, .serialized(resumeState, interruptions: [approvalItem]), scope: toolContext.agentToolStateScope)

        try toolContext.rejectTool(approvalItem, rejectionMessage: "exact denial")

        #expect(resumeState.context.isToolApproved(toolName: "lookup_account", callId: "inner-1") == false)
        #expect(resumeState.context.getRejectionMessage(toolName: "lookup_account", callId: "inner-1") == "exact denial")
        #expect(try agentToolNestedApprovalsStatus(.serialized(resumeState, interruptions: [approvalItem]), outerContext: toolContext) == .rejected)
    }

    @Test func testAgentAsToolNamespacedNestedAlwaysApproveStaysPermanent() throws {
        let agent = Agent(name: "outer")
        let toolCall = makeFunctionToolCall("outer_tool", callId: "outer-1", arguments: Self.outerArguments)
        let toolContext = makeOuterToolContext(toolCall)
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: makeFunctionToolCall("lookup_account", callId: "inner-1", namespace: "billing"))
        let resumeState = makeStateWithInterruptions(agent, [approvalItem])
        recordAgentToolRunResult(toolCall, .serialized(resumeState, interruptions: [approvalItem]), scope: toolContext.agentToolStateScope)

        try toolContext.approveTool(approvalItem, alwaysApprove: true)

        #expect(resumeState.context.isToolApproved(toolName: "billing.lookup_account", callId: "inner-1") == true)
        #expect(resumeState.context.isToolApproved(toolName: "billing.lookup_account", callId: "inner-2") == true)
        #expect(try agentToolNestedApprovalsStatus(.serialized(resumeState, interruptions: [approvalItem]), outerContext: toolContext) == .approved)
    }

    @Test(arguments: [true, false], [false, true])
    func testAgentAsToolToolContextAmbiguousApprovalIdentityFailsClosed(approve: Bool, cloneApprovalItem: Bool) throws {
        let agent = Agent(name: "Agent")
        let outerCall = makeFunctionToolCall("nested_agent_tool", callId: "outer-nested")
        let currentCall = makeFunctionToolCall("sensitive", callId: "shared")
        let currentApproval = ToolApprovalItem(agent: agent, rawItem: currentCall)
        let nestedApproval = cloneApprovalItem ? ToolApprovalItem(agent: agent, rawItem: currentCall) : currentApproval
        let toolContext = makeOuterToolContext(outerCall)
        try toolContext.toolInvocationStatus(currentCall)
        let resumeState = makeStateWithInterruptions(agent, [nestedApproval])
        recordAgentToolRunResult(outerCall, .serialized(resumeState, interruptions: [nestedApproval]), scope: getAgentToolStateScope(toolContext))

        expectError(UserError.self, containing: "current run and a nested agent-tool run") {
            if approve {
                try toolContext.approveTool(currentApproval)
            } else {
                try toolContext.rejectTool(currentApproval)
            }
        }

        #expect(try toolContext.getApprovalStatus(toolName: "sensitive", callId: "shared", existingPending: currentApproval) == nil)
        #expect(try resumeState.context.getApprovalStatus(toolName: "sensitive", callId: "shared", existingPending: nestedApproval) == nil)
    }

    @Test(arguments: [false, true])
    func testAgentAsToolResumeSurvivesCancellationAfterNestedOutputCommit(streamed: Bool) async throws {
        let toolAttempts = SharedBox<[String]>([])
        let nestedModelWaiting = AsyncEvent()
        let responseCalls = SharedBox(0)
        let sensitive = try FunctionTool.create(name: "sensitive", failureErrorFunction: .none, needsApproval: .always) { _ in
            toolAttempts.mutate { $0.append("ran") }
            return "inner value"
        }
        let innerBase = try ScriptedModel(steps: [.output([getFunctionToolCall("sensitive", arguments: "{}", callId: "inner_call")])])
        try innerBase.enqueue(.output([getTextMessage("inner done")]))
        /// Blocks the second model call until the task is cancelled.
        @Sendable func countCall() async throws {
            let count = responseCalls.mutate { value -> Int in
                value += 1
                return value
            }
            if count == 2 {
                nestedModelWaiting.set()
                try await waitForever()
            }
        }
        let innerModel = HookedModel(
            base: innerBase,
            getResponse: { request, base in
                try await countCall()
                return try await base.getResponse(request)
            },
            stream: { request, base in
                makeStream { continuation in
                    try await countCall()
                    for try await event in base.streamResponse(request) {
                        continuation.yield(event)
                    }
                }
            }
        )
        let innerAgent = makeAgent(model: innerModel, tools: [sensitive], name: "inner")
        let nestedTool = try innerAgent.asTool(toolName: "delegate", toolDescription: "Delegate")
        let outerModel = try ScriptedModel(steps: [.output([getFunctionToolCall("delegate", arguments: #"{"input":"hi"}"#, callId: "outer_call")])])
        try outerModel.enqueue(.output([getTextMessage("outer done")]))
        let outerAgent = makeAgent(model: outerModel, tools: [nestedTool], name: "outer")

        @Sendable func runOuter(_ input: RunnerInput) async throws -> (any Sendable)? {
            switch (input, streamed) {
            case (.input(let value), false):
                return try await Runner.run(outerAgent, input: value).finalOutput
            case (.state(let state), false):
                return try await Runner.run(outerAgent, state: state).finalOutput
            case (.input(let value), true):
                let result = try Runner.runStreamed(outerAgent, input: value)
                try await consumeStream(result)
                return result.finalOutput
            case (.state(let state), true):
                let result = try Runner.runStreamed(outerAgent, state: state)
                try await consumeStream(result)
                return result.finalOutput
            }
        }

        let interrupted = try await Runner.run(outerAgent, input: "go")
        let state = interrupted.toState()
        try state.approve(interrupted.interruptions[0])

        let resumeTask = Task { try await runOuter(.state(state)) }
        defer { resumeTask.cancel() }
        try await nestedModelWaiting.wait(timeout: 10)
        #expect(toolAttempts.value == ["ran"])
        #expect(responseCalls.value == 2)

        resumeTask.cancel()
        await expectErrorAsync(CancellationError.self) {
            _ = try await resumeTask.value
        }

        let finalOutput = try await runOuter(.state(state))

        #expect(finalOutput as? String == "outer done")
        #expect(toolAttempts.value == ["ran"])
        #expect(responseCalls.value == 3)
    }

    struct StickyCase: Sendable, CustomTestStringConvertible {
        var approve: Bool
        var sticky: Bool
        var legacySticky: Bool
        var expectedFollowup: Bool?

        var testDescription: String { "approve=\(approve) sticky=\(sticky) legacy=\(legacySticky)" }

        static let all = [
            StickyCase(approve: true, sticky: true, legacySticky: false, expectedFollowup: true),
            StickyCase(approve: false, sticky: true, legacySticky: false, expectedFollowup: false),
            StickyCase(approve: true, sticky: false, legacySticky: true, expectedFollowup: nil),
            StickyCase(approve: false, sticky: false, legacySticky: true, expectedFollowup: nil),
        ]
    }

    @Test(arguments: StickyCase.all)
    func testAgentAsToolHostedMCPNestedStickyDecisionStaysScoped(testCase: StickyCase) throws {
        let agent = Agent(name: "outer")
        let toolCall = makeFunctionToolCall("outer_tool", callId: "outer-1", arguments: Self.outerArguments)
        let toolContext = makeOuterToolContext(toolCall)
        func approvalRequest(_ id: String) -> ToolApprovalItem {
            ToolApprovalItem(agent: agent, rawItem: [
                "id": .string(id), "type": "mcp_approval_request", "server_label": "server-a", "arguments": "{}", "name": "lookup_account",
            ])
        }
        let approvalItem = approvalRequest("inner-1")
        let resumeState = makeStateWithInterruptions(agent, [approvalItem])
        let pendingResult = AgentToolRunRecord.serialized(resumeState, interruptions: [approvalItem])
        recordAgentToolRunResult(toolCall, pendingResult, scope: toolContext.agentToolStateScope)
        if testCase.legacySticky {
            toolContext.rebuildApprovals([
                "lookup_account": [
                    "approved": .bool(testCase.approve),
                    "rejected": testCase.approve ? [] : true,
                    "sticky_rejection_message": testCase.approve ? .null : "legacy denial",
                ],
            ])
            toolContext.toolStateBox.mutate { $0.allowLegacyApprovalBindingReconstruction = true }
        }
        if testCase.approve {
            try toolContext.approveTool(approvalItem, alwaysApprove: testCase.sticky)
        } else {
            try toolContext.rejectTool(approvalItem, alwaysReject: testCase.sticky, rejectionMessage: "server-a denied")
        }

        #expect(try agentToolNestedApprovalsStatus(pendingResult, outerContext: toolContext) == (testCase.approve ? .approved : .rejected))
        let nestedContext = resumeState.context
        #expect(try nestedContext.getApprovalStatus(toolName: "lookup_account", callId: "inner-1", existingPending: approvalItem) == testCase.approve)
        #expect(
            nestedContext.getRejectionMessage(toolName: "lookup_account", callId: "inner-1", existingPending: approvalItem)
                == (testCase.approve ? nil : "server-a denied")
        )
        let followup = approvalRequest("inner-2")
        #expect(try nestedContext.getApprovalStatus(toolName: "lookup_account", callId: "inner-2", existingPending: followup) == testCase.expectedFollowup)
        #expect(
            nestedContext.getRejectionMessage(toolName: "lookup_account", callId: "inner-2", existingPending: followup)
                == (testCase.expectedFollowup == false ? "server-a denied" : nil)
        )
    }

    @Test func testAgentAsToolDeferredLegacyNestedApprovalRequiresFreshDecision() async throws {
        let agent = Agent(name: "outer")
        let toolCall = makeFunctionToolCall("outer_tool", callId: "outer-1", arguments: Self.outerArguments)
        let toolContext = makeOuterToolContext(toolCall)
        func weatherCall(_ callId: String) -> ToolApprovalItem {
            ToolApprovalItem(
                agent: agent,
                rawItem: makeFunctionToolCall("get_weather", callId: callId, namespace: "get_weather"),
                toolLookupKey: .deferredTopLevel("get_weather")
            )
        }
        let approvalItem = weatherCall("inner-1")
        var legacyRecord = ApprovalRecord()
        legacyRecord.approved = .all(true)
        toolContext.approvals["get_weather.get_weather"] = legacyRecord
        toolContext.toolStateBox.mutate { $0.allowLegacyApprovalBindingReconstruction = true }
        let resumeState = makeStateWithInterruptions(agent, [approvalItem])
        let pendingResult = AgentToolRunRecord.serialized(resumeState, interruptions: [approvalItem])
        recordAgentToolRunResult(toolCall, pendingResult, scope: toolContext.agentToolStateScope)
        let tool = try agent.asTool(toolName: "outer_tool", toolDescription: "Outer agent tool")

        // The legacy deferred grant does not resume the nested run without a fresh decision.
        let output = try await tool.onInvokeTool(toolContext, Self.outerArguments)
        #expect((output as? String?) == .some(nil))

        try toolContext.approveTool(approvalItem, alwaysApprove: true)

        #expect(try agentToolNestedApprovalsStatus(pendingResult, outerContext: toolContext) == .approved)
        let nestedContext = resumeState.context
        #expect(try nestedContext.getApprovalStatus(toolName: "get_weather", callId: "inner-1", toolNamespace: "get_weather", existingPending: approvalItem) == true)
        #expect(
            try nestedContext.getApprovalStatus(toolName: "get_weather", callId: "inner-2", toolNamespace: "get_weather", existingPending: weatherCall("inner-2"))
                == true
        )
    }

    @Test func testAgentAsToolPreservesNamespaceForNestedToolContext() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("ok")])])
        let agent = makeAgent(model: model, name: "namespace-agent")
        let namespaces = SharedBox<[String?]>([])
        let qualifiedNames = SharedBox<[String]>([])
        let agentTool = try agent.asTool(toolName: "lookup_account", toolDescription: "Lookup account", customOutputExtractor: { result in
            // Swift: the nested run's wrapper is the nested ToolContext the patched runner saw.
            let nestedContext = result.contextWrapper as? ToolContext
            namespaces.mutate { $0.append(nestedContext?.toolNamespace) }
            qualifiedNames.mutate { $0.append(nestedContext?.qualifiedToolName ?? "") }
            return result.finalOutput as? String ?? ""
        })
        let tool = try toolNamespace(name: "billing", description: "Billing tools", tools: [agentTool])[0]
        let toolCall = makeFunctionToolCall("lookup_account", callId: "lookup-call", arguments: #"{"input":"hello"}"#, namespace: "billing")
        let toolContext = ToolContext(
            context: (),
            toolName: "lookup_account",
            toolCallId: "lookup-call",
            toolArguments: #"{"input":"hello"}"#,
            toolCall: toolCall,
            toolNamespace: "billing"
        )

        let output = try await tool.onInvokeTool(toolContext, #"{"input":"hello"}"#)

        #expect(output as? String == "ok")
        #expect(namespaces.value == ["billing"])
        #expect(qualifiedNames.value == ["billing.lookup_account"])
    }

    // MARK: - Fresh runs

    @Test(arguments: [false, true])
    func testFreshRunsIsolateNestedAgentToolApprovals(streamed: Bool) async throws {
        struct ValueArgs: Codable, Sendable { var value: String }
        let executed = SharedBox<[(context: JSONObject, value: String, usage: ObjectIdentifier)]>([])
        let confirm = try FunctionTool.create(name: "confirm", argumentsType: ValueArgs.self, failureErrorFunction: .none, needsApproval: .always) { context, args in
            executed.mutate { $0.append((context.context(as: JSONObject.self) ?? [:], args.value, ObjectIdentifier(context.usageBox))) }
            return args.value
        }
        let innerModel = try ScriptedModel(steps: [
            .output([getFunctionToolCall("confirm", arguments: #"{"value":"first"}"#, callId: "inner")]),
            .output([getFunctionToolCall("confirm", arguments: #"{"value":"second"}"#, callId: "inner")]),
            .output([getTextMessage("second confirmed")]),
            .output([getTextMessage("first confirmed")]),
        ])
        let inner = makeAgent(model: innerModel, tools: [confirm], name: "inner")
        let onStream: (@Sendable (AgentToolStreamEvent) async -> Void)? = streamed ? { @Sendable _ in } : nil
        let delegate = try inner.asTool(toolName: "delegate", toolDescription: "Delegate", onStream: onStream)
        let outerModel = try ScriptedModel(steps: [
            .output([getFunctionToolCall("delegate", arguments: #"{"input":"go"}"#, callId: "outer")]),
            .output([getFunctionToolCall("delegate", arguments: #"{"input":"go"}"#, callId: "outer")]),
            .output([getTextMessage("second done")]),
            .output([getTextMessage("first done")]),
        ])
        let outer = makeAgent(model: outerModel, tools: [delegate], name: "outer")

        func run(_ input: RunnerInput, _ context: JSONObject) async throws -> any RunResultBase & RunStateConvertible {
            switch (input, streamed) {
            case (.input(let value), false):
                return try await Runner.run(outer, input: value, context: context)
            case (.state(let state), false):
                return try await Runner.run(outer, state: state, context: context)
            case (.input(let value), true):
                let result = try Runner.runStreamed(outer, input: value, context: context)
                try await consumeStream(result)
                return result
            case (.state(let state), true):
                let result = try Runner.runStreamed(outer, state: state, context: context)
                try await consumeStream(result)
                return result
            }
        }

        let firstContext: JSONObject = ["user": "first"]
        let secondContext: JSONObject = ["user": "second"]
        let first = try await run(.input("go"), firstContext)
        // Keep the first result alive and paused while a second fresh run uses the same call fields.
        let second = try await run(.input("go"), secondContext)
        #expect(first.interruptions.count == 1)
        #expect(second.interruptions.count == 1)
        #expect(try JSONValue.parse(first.interruptions[0].rawItem.string("arguments") ?? "") == ["value": "first"])
        #expect(try JSONValue.parse(second.interruptions[0].rawItem.string("arguments") ?? "") == ["value": "second"])
        #expect(executed.value.isEmpty)

        let secondState = second.toState()
        try secondState.approve(second.interruptions[0])
        let secondDone = try await run(.state(secondState), secondContext)
        #expect(secondDone.finalOutput as? String == "second done")
        #expect(secondDone.interruptions.isEmpty)
        #expect(executed.value.map(\.context) == [secondContext])
        #expect(executed.value.map(\.value) == ["second"])
        #expect(executed.value.first?.usage == ObjectIdentifier(secondDone.contextWrapper.usageBox))

        // Completing the newer run must not consume the older run's pending approval or arguments.
        let stillPending = try await run(.state(first.toState()), firstContext)
        #expect(stillPending.interruptions.count == 1)
        #expect(executed.value.count == 1)
        let firstState = stillPending.toState()
        try firstState.approve(stillPending.interruptions[0])
        let firstDone = try await run(.state(firstState), firstContext)
        #expect(firstDone.finalOutput as? String == "first done")
        #expect(firstDone.interruptions.isEmpty)
        #expect(executed.value.map(\.context) == [secondContext, firstContext])
        #expect(executed.value.map(\.value) == ["second", "first"])
        #expect(executed.value.last?.usage == ObjectIdentifier(firstDone.contextWrapper.usageBox))
        try innerModel.assertComplete()
        try outerModel.assertComplete()
    }
}
