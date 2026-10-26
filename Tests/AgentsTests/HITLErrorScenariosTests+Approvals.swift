import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_hitl_error_scenarios.py: resume selection helpers, execution-time
// approval decisions, and hosted MCP approval callbacks.
extension HITLErrorScenariosTests {
    struct ValueArgs: Codable, Sendable { var value: String }

    static func requireApprovalEvaluation(_ run: ToolRunFunction) -> FunctionToolApproval {
        FunctionToolApproval(.requireApproval, tool: run.functionTool, arguments: run.toolCall.arguments)
    }

    // MARK: - Function selection on resume

    @Test func testResumeHonorsPermanentNamespacedFunctionApprovalWithNewCallId() async throws {
        let lookupAccount = try FunctionTool.create(name: "lookup_account", argumentsType: CustomerArgs.self, needsApproval: true) { _, args in
            args.customerId
        }
        let namespacedTool = try #require(try toolNamespace(name: "billing", description: "Billing tools", tools: [lookupAccount]).first)
        let contextWrapper = makeContextWrapper()
        let agent = Agent(name: "billing-agent")
        let approvedItem = ToolApprovalItem(
            agent: agent,
            rawItem: makeFunctionToolCall("lookup_account", callId: "approved-call", arguments: #"{"customer_id":"customer_1"}"#, namespace: "billing")
        )
        try contextWrapper.approveTool(approvedItem, alwaysApprove: true)
        let resumedRun = ToolRunFunction(
            toolCall: try Self.functionCall("lookup_account", callId: "resumed-call", arguments: #"{"customer_id":"customer_2"}"#, namespace: "billing"),
            functionTool: namespacedTool
        )
        var pending: [ToolApprovalItem] = []
        var rejections: [String?] = []

        let selected = try await selectFunctionToolRunsForResume(
            [resumedRun],
            approvalItemsByCallId: [:],
            contextWrapper: contextWrapper,
            needsApprovalChecker: { Self.requireApprovalEvaluation($0) },
            outputExistsChecker: { _ in false },
            recordRejection: { callId, _, _ in rejections.append(callId) },
            pendingInterruptionAdder: { pending.append($0) },
            pendingItemBuilder: { run in
                ToolApprovalItem(agent: agent, rawItem: run.toolCall.json, toolName: run.functionTool.name, toolNamespace: "billing")
            }
        )

        #expect(selected.map(\.toolCall.callId) == ["resumed-call"])
        #expect(pending.isEmpty)
        #expect(rejections.isEmpty)
    }

    @Test func testResumeSkipsNeedsApprovalCheckerWhenStatusResolved() async throws {
        let approveMe = try FunctionTool.create(name: "approve_me", argumentsType: ValueArgs.self, needsApproval: true) { _, args in args.value }
        let approvedCall = try Self.functionCall(approveMe.name, callId: "approved-call", arguments: #"{"value":"a"}"#)
        let rejectedCall = try Self.functionCall(approveMe.name, callId: "rejected-call", arguments: #"{"value":"b"}"#)
        let agent = Agent(name: "agent")
        let contextWrapper = makeContextWrapper()
        try contextWrapper.approveTool(ToolApprovalItem(agent: agent, rawItem: approvedCall.json))
        try contextWrapper.rejectTool(ToolApprovalItem(agent: agent, rawItem: rejectedCall.json))
        let runs = [
            ToolRunFunction(toolCall: approvedCall, functionTool: approveMe),
            ToolRunFunction(toolCall: rejectedCall, functionTool: approveMe),
        ]
        var checkerCalls: [String] = []
        var rejections: [String?] = []

        let selected = try await selectFunctionToolRunsForResume(
            runs,
            approvalItemsByCallId: [:],
            contextWrapper: contextWrapper,
            needsApprovalChecker: { run in
                checkerCalls.append(run.toolCall.callId)
                throw TestError("checker must not run for resolved approvals")
            },
            outputExistsChecker: { _ in false },
            recordRejection: { callId, _, _ in rejections.append(callId) },
            pendingInterruptionAdder: { _ in },
            pendingItemBuilder: { run in ToolApprovalItem(agent: agent, rawItem: run.toolCall.json) }
        )

        #expect(checkerCalls.isEmpty)
        #expect(selected.map(\.toolCall.callId) == ["approved-call"])
        #expect(rejections == ["rejected-call"])
    }

    // Swift: upstream makes the pending item falsy to check that it is reused by identity rather
    // than truthiness. Swift optionals have no truthiness, so identity is checked directly.
    @Test func testFunctionResumeReusesFalsyPendingItem() async throws {
        let approveMe = try FunctionTool.create(name: "approve_me", argumentsType: ValueArgs.self, needsApproval: true) { _, args in args.value }
        let toolCall = try Self.functionCall(approveMe.name, callId: "pending-function", arguments: #"{"value":"a"}"#)
        let agent = Agent(name: "agent", tools: [approveMe])
        let existingPending = ToolApprovalItem(agent: agent, rawItem: toolCall.json)
        let run = ToolRunFunction(toolCall: toolCall, functionTool: approveMe)
        var pending: [ToolApprovalItem] = []

        let selected = try await selectFunctionToolRunsForResume(
            [run],
            approvalItemsByCallId: [toolCall.callId: existingPending],
            contextWrapper: makeContextWrapper(),
            needsApprovalChecker: { Self.requireApprovalEvaluation($0) },
            outputExistsChecker: { _ in false },
            recordRejection: { _, _, _ in throw TestError("unresolved approval must not be recorded as a rejection") },
            pendingInterruptionAdder: { pending.append($0) },
            pendingItemBuilder: { _ in ToolApprovalItem(agent: agent, rawItem: toolCall.json) }
        )

        #expect(selected.isEmpty)
        #expect(pending.count == 1)
        #expect(pending.first === existingPending)
    }

    @Test func testResumeRechecksRejectionAfterFunctionApprovalChecker() async throws {
        let sensitive = try FunctionTool.create(name: "sensitive", argumentsType: NoArgs.self, needsApproval: true) { _, _ in "should-not-run" }
        let toolCall = try Self.functionCall(sensitive.name, callId: "call-concurrent-function")
        let run = ToolRunFunction(toolCall: toolCall, functionTool: sensitive)
        let agent = Agent(name: "agent", tools: [sensitive])
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: toolCall.json)
        let contextWrapper = makeContextWrapper()
        let checkerStarted = AsyncEvent()
        let releaseChecker = AsyncEvent()
        let pending = SharedBox<[ToolApprovalItem]>([])
        let rejections = SharedBox<[String?]>([])

        let selection = Task {
            try await selectFunctionToolRunsForResume(
                [run],
                approvalItemsByCallId: [toolCall.callId: approvalItem],
                contextWrapper: contextWrapper,
                needsApprovalChecker: { run in
                    checkerStarted.set()
                    try await releaseChecker.wait()
                    return Self.requireApprovalEvaluation(run)
                },
                outputExistsChecker: { _ in false },
                recordRejection: { callId, _, _ in rejections.mutate { $0.append(callId) } },
                pendingInterruptionAdder: { item in pending.mutate { $0.append(item) } },
                pendingItemBuilder: { _ in approvalItem }
            )
        }
        defer {
            releaseChecker.set()
            selection.cancel()
        }

        try await checkerStarted.wait(timeout: 5)
        try contextWrapper.rejectTool(approvalItem)
        releaseChecker.set()
        let selected = try await selection.value

        #expect(selected.isEmpty)
        #expect(pending.value.isEmpty)
        #expect(rejections.value == [toolCall.callId])
    }

    // MARK: - Non-function selection on resume

    static func shellRejection(agent: Agent) -> (ToolRunShellCall, String) async throws -> RunItem {
        { _, callId in
            ToolCallOutputItem(agent: agent, rawItem: ["type": "function_call_output", "call_id": .string(callId), "output": "rejected"], output: "rejected")
        }
    }

    @Test func testCollectRunsByApprovalSkipsCheckerWhenStatusResolved() async throws {
        let shellTool = try ShellTool(executor: { _ in .text("ok") }, needsApproval: .required(true))
        let approvedCall = makeShellCall("approved-shell")
        let rejectedCall = makeShellCall("rejected-shell")
        let agent = Agent(name: "agent")
        let contextWrapper = makeContextWrapper()
        try contextWrapper.approveTool(ToolApprovalItem(agent: agent, rawItem: approvedCall, toolName: shellTool.name))
        try contextWrapper.rejectTool(ToolApprovalItem(agent: agent, rawItem: rejectedCall, toolName: shellTool.name))
        let runs = [ToolRunShellCall(toolCall: approvedCall, shellTool: shellTool), ToolRunShellCall(toolCall: rejectedCall, shellTool: shellTool)]
        var checkerCalls: [String] = []

        let (approved, rejections) = try await collectRunsByApproval(
            runs,
            callIdExtractor: { $0.toolCall.string("call_id") ?? "" },
            toolNameResolver: { $0.shellTool.name },
            rejectionBuilder: Self.shellRejection(agent: agent),
            contextWrapper: contextWrapper,
            approvalItemsByCallId: [:],
            agent: agent,
            pendingInterruptionAdder: { _ in },
            needsApprovalChecker: { run in
                checkerCalls.append(run.toolCall.string("call_id") ?? "")
                throw TestError("checker must not run for resolved approvals")
            },
            outputExistsChecker: { _ in false }
        )

        #expect(checkerCalls.isEmpty)
        #expect(approved.map { $0.toolCall.string("call_id") } == ["approved-shell"])
        #expect(rejections.count == 1)
    }

    // Swift: see `testFunctionResumeReusesFalsyPendingItem`.
    @Test func testCollectRunsByApprovalReusesFalsyPendingItem() async throws {
        let shellTool = try ShellTool(executor: { _ in .text("ok") }, needsApproval: .required(true))
        let shellCall = makeShellCall("pending-shell")
        let agent = Agent(name: "agent")
        let existingPending = ToolApprovalItem(agent: agent, rawItem: shellCall, toolName: shellTool.name)
        let run = ToolRunShellCall(toolCall: shellCall, shellTool: shellTool)
        var pending: [ToolApprovalItem] = []

        let (approved, rejections) = try await collectRunsByApproval(
            [run],
            callIdExtractor: { $0.toolCall.string("call_id") ?? "" },
            toolNameResolver: { $0.shellTool.name },
            rejectionBuilder: Self.shellRejection(agent: agent),
            contextWrapper: makeContextWrapper(),
            approvalItemsByCallId: ["pending-shell": existingPending],
            agent: agent,
            pendingInterruptionAdder: { pending.append($0) },
            needsApprovalChecker: { _ in true }
        )

        #expect(approved.isEmpty)
        #expect(rejections.isEmpty)
        #expect(pending.count == 1)
        #expect(pending.first === existingPending)
    }

    // MARK: - Execution-time decisions

    @Test(arguments: [true, false])
    func testExecutePathPrefersDecisionResolvedDuringRejectingGuardrail(approved: Bool) async throws {
        let guardrailStarted = AsyncEvent()
        let releaseGuardrail = AsyncEvent()
        let guardrailCalls = SharedBox<Int>(0)
        let executed = SharedBox<[String]>([])
        let rejectingGuardrail = toolInputGuardrail { _ in
            guardrailCalls.mutate { $0 += 1 }
            guardrailStarted.set()
            try await releaseGuardrail.wait()
            return .rejectContent("guardrail rejection")
        }
        let sensitive = try FunctionTool.create(
            name: "sensitive", argumentsType: NoArgs.self, needsApproval: true, toolInputGuardrails: [rejectingGuardrail]
        ) { _, _ in
            executed.mutate { $0.append("ran") }
            return "tool output"
        }
        let toolCall = try Self.functionCall(sensitive.name, callId: "call-pending-guardrail")
        let toolRun = ToolRunFunction(toolCall: toolCall, functionTool: sensitive)
        let agent = Agent(name: "agent", tools: [sensitive])
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: toolCall.json)
        let contextWrapper = makeContextWrapper()
        let config = RunConfig(toolExecution: try ToolExecutionConfig(preApprovalToolInputGuardrails: true))
        let execution = Task {
            try await FunctionToolBatchExecutor(
                agent: agent, toolRuns: [toolRun], hooks: RunHooks(), contextWrapper: contextWrapper,
                config: config, availableFunctionTools: [sensitive]
            ).execute().0
        }
        defer {
            releaseGuardrail.set()
            execution.cancel()
        }

        try await guardrailStarted.wait(timeout: 5)
        if approved {
            try contextWrapper.approveTool(approvalItem)
        } else {
            try contextWrapper.rejectTool(approvalItem, rejectionMessage: "stored rejection")
        }
        releaseGuardrail.set()
        let results = try await execution.value

        #expect(results.count == 1)
        if approved {
            #expect(results.first?.output as? String == "guardrail rejection")
            #expect(guardrailCalls.value == 2)
        } else {
            #expect(results.first?.output as? String == "stored rejection")
            #expect(guardrailCalls.value == 1)
        }
        #expect(executed.value.isEmpty)
    }

    enum ExecutionToolKind: String, CaseIterable, Sendable {
        case function, shell, custom, applyPatch = "apply_patch"
    }

    @Test(arguments: ExecutionToolKind.allCases)
    func testExecutePathHonorsStickyRejectionBeforeChecker(toolKind: ExecutionToolKind) async throws {
        let executed = SharedBox<[String]>([])
        let contextWrapper = makeContextWrapper()
        let unexpected = "sticky rejection must bypass needs_approval"
        switch toolKind {
        case .function:
            let sensitive = try FunctionTool.create(
                name: "sensitive", argumentsType: NoArgs.self, needsApproval: .dynamic { _, _, _ in throw TestError(unexpected) }
            ) { _, _ in
                executed.mutate { $0.append("function") }
                return "should-not-run"
            }
            let agent = Agent(name: "agent", tools: [sensitive])
            try contextWrapper.rejectTool(
                ToolApprovalItem(agent: agent, rawItem: makeFunctionToolCall(sensitive.name, callId: "call-prior")), alwaysReject: true
            )
            let (functionResults, _, _) = try await FunctionToolBatchExecutor(
                agent: agent,
                toolRuns: [ToolRunFunction(toolCall: try Self.functionCall(sensitive.name, callId: "call-next"), functionTool: sensitive)],
                hooks: RunHooks(), contextWrapper: contextWrapper, config: RunConfig(), availableFunctionTools: [sensitive]
            ).execute()
            #expect(functionResults.map { $0.output as? String } == [hitlRejectionMessage])
        case .shell:
            let shellTool = try ShellTool(
                executor: { _ in
                    executed.mutate { $0.append("shell") }
                    return .text("should-not-run")
                },
                needsApproval: .dynamic { _, _, _ in throw TestError(unexpected) }
            )
            let agent = Agent(name: "agent", tools: [shellTool])
            try contextWrapper.rejectTool(ToolApprovalItem(agent: agent, rawItem: makeShellCall("call-prior"), toolName: shellTool.name), alwaysReject: true)
            let result = try await ShellAction.execute(
                agent: agent, call: ToolRunShellCall(toolCall: makeShellCall("call-next"), shellTool: shellTool),
                hooks: RunHooks(), contextWrapper: contextWrapper, config: RunConfig()
            )
            let output = try #require(result as? ToolCallOutputItem)
            #expect(String(describing: output.output).contains(hitlRejectionMessage))
        case .custom:
            let customTool = try CustomTool(
                name: "raw_editor",
                description: "Edit raw text.",
                onInvokeTool: { _, _ in
                    executed.mutate { $0.append("custom") }
                    return "should-not-run"
                },
                format: ["type": "text"],
                needsApproval: .dynamic { _, _, _ in throw TestError(unexpected) }
            )
            let agent = Agent(name: "agent", tools: [customTool])
            try contextWrapper.rejectTool(
                ToolApprovalItem(agent: agent, rawItem: Self.customToolCall(customTool.name, callId: "call-prior", input: "prior").json, toolName: customTool.name),
                alwaysReject: true
            )
            let result = try await CustomToolAction.execute(
                agent: agent, call: ToolRunCustom(toolCall: Self.customToolCall(customTool.name, callId: "call-next", input: "next"), customTool: customTool),
                hooks: RunHooks(), contextWrapper: contextWrapper, config: RunConfig()
            )
            let output = try #require(result as? ToolCallOutputItem)
            #expect(output.output as? String == hitlRejectionMessage)
        case .applyPatch:
            let editor = RecordingEditor()
            let applyPatchTool = try ApplyPatchTool(editor: editor, needsApproval: .dynamic { _, _, _ in throw TestError(unexpected) })
            let agent = Agent(name: "agent", tools: [applyPatchTool])
            try contextWrapper.rejectTool(
                ToolApprovalItem(agent: agent, rawItem: makeApplyPatchDict("call-prior"), toolName: applyPatchTool.name), alwaysReject: true
            )
            let result = try await ApplyPatchAction.execute(
                agent: agent, call: ToolRunApplyPatchCall(toolCall: makeApplyPatchDict("call-next"), applyPatchTool: applyPatchTool),
                hooks: RunHooks(), contextWrapper: contextWrapper, config: RunConfig()
            )
            let output = try #require(result as? ToolCallOutputItem)
            #expect(String(describing: output.output).contains(hitlRejectionMessage))
            #expect(editor.operations.isEmpty)
        }

        #expect(executed.value.isEmpty)
    }

    static func customToolCall(_ name: String, callId: String, input: String) -> ResponseCustomToolCall {
        ResponseCustomToolCall(json: ["type": "custom_tool_call", "name": .string(name), "call_id": .string(callId), "input": .string(input)])!
    }

    enum NonFunctionToolKind: String, CaseIterable, Sendable {
        case shell, custom, applyPatch = "apply_patch"
    }

    @Test(arguments: NonFunctionToolKind.allCases)
    func testToolExecutionRejectsChangedApprovalRecordedWhilePolicyWaits(toolKind: NonFunctionToolKind) async throws {
        let checkerStarted = AsyncEvent()
        let releaseChecker = AsyncEvent()
        let executed = SharedBox<[String]>([])
        let contextWrapper = makeContextWrapper()
        @Sendable func waitForRelease() async throws -> Bool {
            checkerStarted.set()
            try await releaseChecker.wait()
            return true
        }
        let editor = RecordingEditor()
        let agent: Agent
        let toolName: String
        let changedRaw: JSONObject
        let execution: Task<RunItem, any Error>
        switch toolKind {
        case .shell:
            let tool = try ShellTool(
                executor: { _ in
                    executed.mutate { $0.append("shell") }
                    return .text("should-not-run")
                },
                needsApproval: .dynamic { _, _, _ in try await waitForRelease() }
            )
            agent = Agent(name: "agent", tools: [tool])
            toolName = tool.name
            let currentRaw = makeShellCall("call-shared", commands: ["safe"])
            changedRaw = makeShellCall("call-shared", commands: ["changed"])
            execution = Task { [agent] in
                try await ShellAction.execute(
                    agent: agent, call: ToolRunShellCall(toolCall: currentRaw, shellTool: tool),
                    hooks: RunHooks(), contextWrapper: contextWrapper, config: RunConfig()
                )
            }
        case .custom:
            let tool = try CustomTool(
                name: "raw_editor",
                description: "Edit raw text.",
                onInvokeTool: { _, _ in
                    executed.mutate { $0.append("custom") }
                    return "should-not-run"
                },
                format: ["type": "text"],
                needsApproval: .dynamic { _, _, _ in try await waitForRelease() }
            )
            agent = Agent(name: "agent", tools: [tool])
            toolName = tool.name
            let currentRaw = Self.customToolCall(tool.name, callId: "call-shared", input: "safe")
            changedRaw = Self.customToolCall(tool.name, callId: "call-shared", input: "changed").json
            execution = Task { [agent] in
                try await CustomToolAction.execute(
                    agent: agent, call: ToolRunCustom(toolCall: currentRaw, customTool: tool),
                    hooks: RunHooks(), contextWrapper: contextWrapper, config: RunConfig()
                )
            }
        case .applyPatch:
            let tool = try ApplyPatchTool(editor: editor, needsApproval: .dynamic { _, _, _ in try await waitForRelease() })
            agent = Agent(name: "agent", tools: [tool])
            toolName = tool.name
            let currentRaw: JSONObject = ["type": "apply_patch_call", "call_id": "call-shared", "operation": ["type": "delete_file", "path": "safe.txt"]]
            changedRaw = ["type": "apply_patch_call", "call_id": "call-shared", "operation": ["type": "delete_file", "path": "changed.txt"]]
            execution = Task { [agent] in
                try await ApplyPatchAction.execute(
                    agent: agent, call: ToolRunApplyPatchCall(toolCall: currentRaw, applyPatchTool: tool),
                    hooks: RunHooks(), contextWrapper: contextWrapper, config: RunConfig()
                )
            }
        }
        defer {
            releaseChecker.set()
            execution.cancel()
        }

        try await checkerStarted.wait(timeout: 5)
        try contextWrapper.approveTool(ToolApprovalItem(agent: agent, rawItem: changedRaw, toolName: toolName))
        releaseChecker.set()
        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") { _ = try await execution.value }

        #expect(executed.value.isEmpty)
        #expect(editor.operations.isEmpty)
    }

    // MARK: - Hosted MCP callbacks

    @Test func testMcpCallbackExactRetryReusesStoredDecision() async throws {
        let callbackCalls = SharedBox<[String]>([])
        let agent = makeAgent()
        let contextWrapper = makeContextWrapper()
        let mcpTool = HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "server"]) { request in
            callbackCalls.mutate { $0.append(request.data.id) }
            return MCPToolApprovalFunctionResult(approve: true, reason: "ok")
        }
        let approvalRequest = ToolRunMCPApprovalRequest(requestItem: try Self.mcpApprovalRequest("mcp-callback-retry"), mcpTool: mcpTool)

        let first = try await executeMCPApprovalRequests(agent: agent, requests: [approvalRequest], contextWrapper: contextWrapper)
        let second = try await executeMCPApprovalRequests(agent: agent, requests: [approvalRequest], contextWrapper: contextWrapper)

        #expect(callbackCalls.value == ["mcp-callback-retry"])
        let responses = (first + second).compactMap { $0 as? MCPApprovalResponseItem }
        #expect(responses.map { $0.rawItem["approve"] } == [true, true])
    }

    @Test func testMcpCallbackFailureIsNotRetriedForSameRequest() async throws {
        let callbackCalls = SharedBox<[String]>([])
        let agent = makeAgent()
        let contextWrapper = makeContextWrapper()
        let mcpTool = HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "server"]) { request in
            callbackCalls.mutate { $0.append(request.data.id) }
            throw TestError("callback failed")
        }
        let approvalRequest = ToolRunMCPApprovalRequest(requestItem: try Self.mcpApprovalRequest("mcp-callback-failure"), mcpTool: mcpTool)

        let error = await expectErrorAsync(TestError.self) {
            _ = try await executeMCPApprovalRequests(agent: agent, requests: [approvalRequest], contextWrapper: contextWrapper)
        }
        #expect(error?.message == "callback failed")
        await expectErrorAsync(ModelBehaviorError.self, containing: "already ran") {
            _ = try await executeMCPApprovalRequests(agent: agent, requests: [approvalRequest], contextWrapper: contextWrapper)
        }

        #expect(callbackCalls.value == ["mcp-callback-failure"])
    }

    @Test func testMcpCallbackExactSiblingsInvokeCallbackOnce() async throws {
        let callbackCalls = SharedBox<[String]>([])
        let agent = makeAgent()
        let contextWrapper = makeContextWrapper()
        let mcpTool = HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "server"]) { request in
            callbackCalls.mutate { $0.append(request.data.id) }
            await Task.yield()
            return MCPToolApprovalFunctionResult(approve: true, reason: "ok")
        }
        let approvalRequests = try (0..<2).map { _ in
            ToolRunMCPApprovalRequest(requestItem: try Self.mcpApprovalRequest("mcp-callback-sibling"), mcpTool: mcpTool)
        }

        let responses = try await executeMCPApprovalRequests(agent: agent, requests: approvalRequests, contextWrapper: contextWrapper)

        #expect(callbackCalls.value == ["mcp-callback-sibling"])
        #expect(responses.count == 1)
    }

    @Test func testMcpCallbackChangedSameIdSiblingsFailBeforeCallbacks() async throws {
        let callbackCalls = SharedBox<[String]>([])
        let agent = makeAgent()
        let contextWrapper = makeContextWrapper()
        let mcpTool = HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "server"]) { request in
            callbackCalls.mutate { $0.append(request.data.arguments) }
            await Task.yield()
            return MCPToolApprovalFunctionResult(approve: true, reason: "ok")
        }
        let approvalRequests = try [#"{"q":1}"#, #"{"q":2}"#].map { arguments in
            ToolRunMCPApprovalRequest(requestItem: try Self.mcpApprovalRequest("mcp-callback-sibling", arguments: arguments), mcpTool: mcpTool)
        }

        await expectErrorAsync(ModelBehaviorError.self, containing: "reused an approval-gated tool call ID") {
            _ = try await executeMCPApprovalRequests(agent: agent, requests: approvalRequests, contextWrapper: contextWrapper)
        }

        #expect(callbackCalls.value.isEmpty)
    }
}
