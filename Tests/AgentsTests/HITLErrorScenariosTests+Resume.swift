import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_hitl_error_scenarios.py: tests that resume an interrupted turn directly.
extension HITLErrorScenariosTests {
    struct ReasonArgs: Codable, Sendable { var reason: String? }
    struct TextArgs: Codable, Sendable { var text: String? }
    struct CustomerArgs: Codable, Sendable { var customerId: String; enum CodingKeys: String, CodingKey { case customerId = "customer_id" } }

    static func emptyProcessedResponse() -> ProcessedResponse {
        ProcessedResponse(
            newItems: [], handoffs: [], functions: [], computerActions: [], localShellCalls: [], shellCalls: [],
            applyPatchCalls: [], toolsUsed: [], mcpApprovalRequests: [], interruptions: []
        )
    }

    static func emptyResponse() -> ModelResponse {
        ModelResponse(output: [], usage: Usage(), responseId: "resp")
    }

    static func functionCall(_ name: String, callId: String = "call-1", arguments: String = "{}", namespace: String? = nil) throws -> ResponseFunctionToolCall {
        try #require(ResponseFunctionToolCall(json: makeFunctionToolCall(name, callId: callId, arguments: arguments, namespace: namespace)))
    }

    /// Upstream `run_loop.resolve_interrupted_turn` with the agent's current tool inventory.
    static func resolveTurn(
        agent: Agent,
        originalInput: String,
        originalPreStepItems: [RunItem] = [],
        newResponse: ModelResponse = emptyResponse(),
        processedResponse: ProcessedResponse = emptyProcessedResponse(),
        contextWrapper: RunContextWrapper = makeContextWrapper(),
        runConfig: RunConfig = RunConfig(),
        runState: RunState? = nil
    ) async throws -> SingleStepResult {
        try await resolveInterruptedTurn(
            agent: agent,
            allTools: try await agent.getAllTools(contextWrapper),
            originalInput: .text(originalInput),
            originalPreStepItems: originalPreStepItems,
            newResponse: newResponse,
            processedResponse: processedResponse,
            hooks: RunHooks(),
            contextWrapper: contextWrapper,
            runConfig: runConfig,
            serverManagesConversation: false,
            errorHandlers: nil,
            runState: runState
        )
    }

    static func toolOutputs(_ items: [RunItem]) -> [ToolCallOutputItem] {
        items.compactMap { $0 as? ToolCallOutputItem }
    }

    static func isInterruption(_ step: NextStep) -> Bool {
        if case .interruption = step { return true }
        return false
    }

    static func isRunAgain(_ step: NextStep) -> Bool {
        if case .runAgain = step { return true }
        return false
    }

    // MARK: - Validation

    @Test func testShellCallWithoutCallIdRaises() async throws {
        let agent = makeAgent()
        let shellTool = try ShellTool(executor: { _ in .text("") })
        let shellCall: JSONObject = ["type": "shell_call", "action": ["commands": ["echo", "hi"]]]
        var processed = Self.emptyProcessedResponse()
        processed.shellCalls = [ToolRunShellCall(toolCall: shellCall, shellTool: shellTool)]

        await expectErrorAsync(ModelBehaviorError.self) {
            _ = try await Self.resolveTurn(agent: agent, originalInput: "test", processedResponse: processed)
        }
    }

    // MARK: - Rebuilding function runs from approvals

    @Test func testResumeRebuildsFunctionRunsFromPendingApprovals() async throws {
        let approveMe = try FunctionTool.create(name: "approve_me", argumentsType: ReasonArgs.self, needsApproval: true) { _, args in
            args.reason.map { "approved:\($0)" } ?? "approved"
        }
        let (_, agent) = try makeModelAndAgent(tools: [approveMe])
        let approvalRaw: JSONObject = [
            "type": "function_call",
            "name": .string(approveMe.name),
            "call_id": "call-rebuild-1",
            "arguments": #"{"reason": "ok"}"#,
            "status": "completed",
        ]
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: approvalRaw)
        let contextWrapper = makeContextWrapper()
        try contextWrapper.approveTool(approvalItem)
        let runState = makeStateWithInterruptions(agent, [approvalItem])

        let result = try await Self.resolveTurn(agent: agent, originalInput: "resume approvals", contextWrapper: contextWrapper, runState: runState)

        #expect(!Self.isInterruption(result.nextStep), "Approved function should run instead of requesting approval again")
        let executedCallIds = Set(Self.toolOutputs(result.newStepItems).compactMap(\.callId))
        #expect(executedCallIds.contains("call-rebuild-1"), "Function should be rebuilt and executed")
    }

    @Test func testResumeRebuildsDeferredFunctionRunsFromLookupKeyWithoutRawNamespace() async throws {
        let visible = try FunctionTool.create(name: "lookup_account", argumentsType: CustomerArgs.self, needsApproval: true) { _, args in
            "visible:\(args.customerId)"
        }
        let deferred = try FunctionTool.create(
            name: "lookup_account", argumentsType: CustomerArgs.self, needsApproval: true, deferLoading: true
        ) { _, args in "deferred:\(args.customerId)" }
        let (_, agent) = try makeModelAndAgent(tools: [visible, deferred])
        let approvalItem = ToolApprovalItem(
            agent: agent,
            rawItem: [
                "type": "function_call",
                "name": "lookup_account",
                "call_id": "call-deferred-rebuild",
                "arguments": #"{"customer_id":"customer_1"}"#,
                "status": "completed",
            ],
            toolName: "lookup_account",
            toolNamespace: "lookup_account",
            toolLookupKey: .deferredTopLevel("lookup_account")
        )
        let contextWrapper = makeContextWrapper()
        try contextWrapper.approveTool(approvalItem)
        let runState = makeStateWithInterruptions(agent, [approvalItem])

        let result = try await Self.resolveTurn(agent: agent, originalInput: "resume approvals", contextWrapper: contextWrapper, runState: runState)

        #expect(!Self.isInterruption(result.nextStep))
        let deferredOutputs = Self.toolOutputs(result.newStepItems).compactMap { $0.output as? String }.filter { $0 == "deferred:customer_1" }
        #expect(deferredOutputs == ["deferred:customer_1"])
    }

    @Test func testResumeDoesNotRebuildApprovedCallsForSameNamedSiblingAgent() async throws {
        let firstCalls = SharedBox<[String]>([])
        let secondCalls = SharedBox<[String]>([])
        let firstTool = try FunctionTool.create(name: "approval_tool", argumentsType: NoArgs.self, needsApproval: true) { _, _ in
            firstCalls.mutate { $0.append("first") }
            return "first"
        }
        let secondTool = try FunctionTool.create(name: "approval_tool", argumentsType: NoArgs.self, needsApproval: true) { _, _ in
            secondCalls.mutate { $0.append("second") }
            return "second"
        }
        let first = Agent(name: "sandbox", tools: [firstTool])
        let second = Agent(name: "sandbox", tools: [secondTool])
        first.handoffs = [.agent(second)]
        second.handoffs = [.agent(first)]
        let approvalItem = ToolApprovalItem(
            agent: second,
            rawItem: makeFunctionToolCall("approval_tool", callId: "call-sibling-approval", arguments: "{}"),
            toolName: "approval_tool"
        )
        let contextWrapper = makeContextWrapper()
        try contextWrapper.approveTool(approvalItem)
        let runState = makeStateWithInterruptions(first, [approvalItem])

        let result = try await Self.resolveTurn(agent: first, originalInput: "resume approvals", contextWrapper: contextWrapper, runState: runState)

        #expect(firstCalls.value.isEmpty)
        #expect(secondCalls.value.isEmpty)
        #expect(Self.toolOutputs(result.newStepItems).isEmpty)
    }

    @Test func testResumeRebuildsFunctionRunsFromObjectApprovals() async throws {
        let approveMe = try FunctionTool.create(name: "approve_me", argumentsType: ReasonArgs.self, needsApproval: true) { _, args in
            args.reason.map { "approved:\($0)" } ?? "approved"
        }
        let (_, agent) = try makeModelAndAgent(tools: [approveMe])
        let toolCall = makeFunctionToolCall(approveMe.name, callId: "call-rebuild-obj", arguments: #"{"reason": "ok"}"#)
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: toolCall)
        let contextWrapper = makeContextWrapper()
        try contextWrapper.approveTool(approvalItem)
        let runState = makeStateWithInterruptions(agent, [approvalItem])

        let result = try await Self.resolveTurn(agent: agent, originalInput: "resume approvals", contextWrapper: contextWrapper, runState: runState)

        #expect(!Self.isInterruption(result.nextStep))
        let executedCallIds = Set(Self.toolOutputs(result.newStepItems).compactMap(\.callId))
        #expect(executedCallIds.contains("call-rebuild-obj"), "Function should be rebuilt from ResponseFunctionToolCall approval")
    }

    /// Upstream `tests/mcp/helpers.py` `FakeMCPServer`, reduced to the tool listing and call log
    /// this scenario needs.
    final class FakeMCPServer: MCPServer, @unchecked Sendable {
        let name = "fake_mcp_server"
        let requireApproval: MCPRequireApproval
        private let lock = NSLock()
        private var tools: [MCPTool] = []
        private var calls: [String] = []

        init(requireApproval: MCPRequireApproval) {
            self.requireApproval = requireApproval
        }

        var toolCalls: [String] { lock.locked { calls } }

        func addTool(_ name: String, _ inputSchema: JSONObject) {
            lock.locked { tools.append(MCPTool(name: name, inputSchema: inputSchema)) }
        }

        func connect() async throws {}
        func cleanup() async throws {}

        func listTools(runContext: RunContextWrapper?, agent: AgentBase?) async throws -> [MCPTool] {
            lock.locked { tools }
        }

        func callTool(_ toolName: String, arguments: JSONObject?, meta: JSONObject?) async throws -> MCPCallToolResult {
            lock.locked { calls.append(toolName) }
            return MCPCallToolResult(content: [["type": "text", "text": .string("result_\(toolName)")]])
        }
    }

    @Test func testResumeRejectsLocalMcpApprovalWithoutRecipientBinding() async throws {
        let server = FakeMCPServer(requireApproval: .always)
        server.addTool("add", ["type": "object", "properties": [:]])
        let agent = Agent(name: "TestAgent", mcpServers: [server])
        let toolCall = makeFunctionToolCall("add", callId: "call-mcp-rebuild", arguments: #"{"value": 1}"#)
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: toolCall, toolName: "add")
        let contextWrapper = makeContextWrapper()
        try contextWrapper.approveTool(approvalItem)
        let runState = makeStateWithInterruptions(agent, [approvalItem])

        await expectErrorAsync(UserError.self, containing: "missing or different recipient binding") {
            _ = try await Self.resolveTurn(agent: agent, originalInput: "resume approvals", contextWrapper: contextWrapper, runState: runState)
        }

        #expect(server.toolCalls.isEmpty)
    }

    @Test func testResumeRebuildRejectionsUseDeferredToolDisplayName() async throws {
        let getWeather = try FunctionTool.create(name: "get_weather", argumentsType: NoArgs.self, deferLoading: true) { _, _ in "sunny" }
        let (_, agent) = try makeModelAndAgent(tools: [getWeather])
        let contextWrapper = makeContextWrapper()
        let rejectedCall = makeFunctionToolCall("get_weather", callId: "call-deferred-reject", namespace: "get_weather")
        let rejectedItem = ToolApprovalItem(agent: agent, rawItem: rejectedCall, toolName: "get_weather", toolNamespace: "get_weather")
        try contextWrapper.rejectTool(rejectedItem)
        let runState = makeStateWithInterruptions(agent, [rejectedItem])

        let result = try await Self.resolveTurn(
            agent: agent,
            originalInput: "resume approvals",
            contextWrapper: contextWrapper,
            runConfig: RunConfig(toolErrorFormatter: { args in "resume-level \(args.toolName) denied (\(args.callId))" }),
            runState: runState
        )

        let rejectionOutputs = Self.toolOutputs(result.newStepItems).compactMap { $0.output as? String }
        #expect(rejectionOutputs == ["resume-level get_weather denied (call-deferred-reject)"])
    }

    @Test func testRebuildFunctionRunsHandlesObjectPendingAndRejections() async throws {
        let rejectMe = try FunctionTool.create(name: "reject_me", argumentsType: TextArgs.self, needsApproval: true) { _, args in args.text ?? "nope" }
        let pendingMe = try FunctionTool.create(name: "pending_me", argumentsType: TextArgs.self, needsApproval: true) { _, args in args.text ?? "wait" }
        let (_, agent) = try makeModelAndAgent(tools: [rejectMe, pendingMe])
        let contextWrapper = makeContextWrapper()
        let rejectedItem = ToolApprovalItem(agent: agent, rawItem: makeFunctionToolCall(rejectMe.name, callId: "obj-reject"))
        let pendingItem = ToolApprovalItem(agent: agent, rawItem: makeFunctionToolCall(pendingMe.name, callId: "obj-pending"))
        try contextWrapper.rejectTool(rejectedItem)
        let runState = makeStateWithInterruptions(agent, [rejectedItem, pendingItem])

        let result = try await Self.resolveTurn(agent: agent, originalInput: "resume approvals", contextWrapper: contextWrapper, runState: runState)

        #expect(Self.isInterruption(result.nextStep))
        #expect(result.nextStep.interruptions.contains(pendingItem))
        let rejectionOutputs = Self.toolOutputs(result.newStepItems).filter { $0.output as? String == hitlRejectionMessage }
        #expect(!rejectionOutputs.isEmpty, "Rejected function call should emit rejection output")
    }

    @Test func testRebuildFunctionRunsHandlesPendingAndRejections() async throws {
        let rejectMe = try FunctionTool.create(name: "reject_me", argumentsType: TextArgs.self, needsApproval: true) { _, args in args.text ?? "nope" }
        let pendingMe = try FunctionTool.create(name: "pending_me", argumentsType: TextArgs.self, needsApproval: true) { _, args in args.text ?? "wait" }
        let (_, agent) = try makeModelAndAgent(tools: [rejectMe, pendingMe])
        let contextWrapper = makeContextWrapper()
        let rejectedItem = ToolApprovalItem(agent: agent, rawItem: [
            "type": "function_call", "name": .string(rejectMe.name), "call_id": "call-reject", "arguments": "{}",
        ])
        // Swift: upstream makes this item falsy to check identity rather than truthiness; Swift
        // optionals have no truthiness, so identity is checked directly.
        let pendingItem = ToolApprovalItem(agent: agent, rawItem: [
            "type": "function_call", "name": .string(pendingMe.name), "call_id": "call-pending", "arguments": "{}",
        ])
        try contextWrapper.rejectTool(rejectedItem)
        let runState = makeStateWithInterruptions(agent, [rejectedItem, pendingItem])

        let result = try await Self.resolveTurn(agent: agent, originalInput: "resume approvals", contextWrapper: contextWrapper, runState: runState)

        #expect(Self.isInterruption(result.nextStep))
        #expect(result.nextStep.interruptions.contains { $0 === pendingItem })
        #expect(result.newStepItems.contains { $0 === pendingItem })
        let rejectionOutputs = Self.toolOutputs(result.newStepItems).filter { $0.output as? String == hitlRejectionMessage }
        #expect(!rejectionOutputs.isEmpty, "Rejected function call should emit rejection output")
    }

    @Test(arguments: [
        (makeShellCall("call_shell_pending_rebuild", idValue: "shell_pending_rebuild", commands: ["echo pending"]), "shell"),
        (makeApplyPatchDict("call_apply_pending_rebuild"), "apply_patch"),
        (["type": "function_call", "name": "missing_tool", "call_id": "call_missing_tool", "arguments": "{}"], "missing_tool"),
    ] as [(JSONObject, String)])
    func testRebuildPreservesUnmatchedPendingApprovals(rawItem: JSONObject, toolName: String) async throws {
        let (_, agent) = try makeModelAndAgent()
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: rawItem, toolName: toolName)
        let runState = makeStateWithInterruptions(agent, [approvalItem])

        let result = try await Self.resolveTurn(agent: agent, originalInput: "resume approvals", runState: runState)

        #expect(Self.isInterruption(result.nextStep))
        #expect(result.nextStep.interruptions.contains(approvalItem))
    }

    @Test func testResumeFunctionRejectionOutputsUsePublicAgent() async throws {
        let rejectMe = try FunctionTool.create(name: "reject_me", argumentsType: TextArgs.self, needsApproval: true) { _, args in args.text ?? "nope" }
        let (_, publicAgent) = try makeModelAndAgent(tools: [rejectMe])
        let contextWrapper = makeContextWrapper()
        let rejectedItem = ToolApprovalItem(agent: publicAgent, rawItem: makeFunctionToolCall(rejectMe.name, callId: "obj-reject-public"))
        try contextWrapper.rejectTool(rejectedItem)
        let runState = makeStateWithInterruptions(publicAgent, [rejectedItem])

        let result = try await Self.resolveTurn(agent: publicAgent, originalInput: "resume approvals", contextWrapper: contextWrapper, runState: runState)

        let rejectionOutputs = Self.toolOutputs(result.newStepItems).filter { $0.output as? String == hitlRejectionMessage }
        #expect(!rejectionOutputs.isEmpty)
        #expect(rejectionOutputs.allSatisfy { $0.agent === publicAgent })
    }

    @Test(arguments: ToolKind.allCases)
    func testResumeNonFunctionRejectionOutputsUsePublicAgent(toolKind: ToolKind) async throws {
        let contextWrapper = makeContextWrapper()
        var processed = Self.emptyProcessedResponse()
        let publicAgent: Agent
        let rawItem: JSONObject
        let toolName: String
        switch toolKind {
        case .shell:
            let shellTool = try ShellTool(executor: { _ in .text("should_not_run") }, needsApproval: .required(true))
            (_, publicAgent) = try makeModelAndAgent(tools: [shellTool])
            rawItem = makeShellCall("call_reject_shell_public", idValue: "shell_reject_public", commands: ["echo test"], status: "in_progress")
            processed.shellCalls = [ToolRunShellCall(toolCall: rawItem, shellTool: shellTool)]
            toolName = shellTool.name
        case .applyPatch:
            let applyPatchTool = try ApplyPatchTool(editor: RecordingEditor(), needsApproval: .required(true))
            (_, publicAgent) = try makeModelAndAgent(tools: [applyPatchTool])
            rawItem = makeApplyPatchDict("call_apply_reject_public")
            processed.applyPatchCalls = [ToolRunApplyPatchCall(toolCall: rawItem, applyPatchTool: applyPatchTool)]
            toolName = applyPatchTool.name
        }
        let approvalItem = ToolApprovalItem(agent: publicAgent, rawItem: rawItem, toolName: toolName)
        try contextWrapper.rejectTool(approvalItem)

        let result = try await Self.resolveTurn(
            agent: publicAgent,
            originalInput: "resume rejection",
            processedResponse: processed,
            contextWrapper: contextWrapper,
            runState: makeStateWithInterruptions(publicAgent, [approvalItem])
        )

        let rejectionOutputs = Self.toolOutputs(result.newStepItems).filter { String(describing: $0.output).contains(hitlRejectionMessage) }
        #expect(!rejectionOutputs.isEmpty)
        #expect(rejectionOutputs.allSatisfy { $0.agent === publicAgent })
    }

    @Test func testResumeKeepsUnmatchedPendingApprovalsWithFunctionRuns() async throws {
        let outerTool = try FunctionTool.create(name: "outer_tool", argumentsType: NoArgs.self) { _, _ in "outer" }
        let innerTool = try FunctionTool.create(name: "inner_tool", argumentsType: NoArgs.self, needsApproval: true) { _, _ in "inner" }
        let (_, agent) = try makeModelAndAgent(tools: [outerTool, innerTool])
        let pendingItem = ToolApprovalItem(agent: agent, rawItem: makeFunctionToolCall(innerTool.name, callId: "call-inner"))
        let runState = makeStateWithInterruptions(agent, [pendingItem])
        var processed = Self.emptyProcessedResponse()
        processed.functions = [ToolRunFunction(toolCall: try Self.functionCall(outerTool.name, callId: "call-outer"), functionTool: outerTool)]

        let result = try await Self.resolveTurn(agent: agent, originalInput: "resume approvals", processedResponse: processed, runState: runState)

        #expect(Self.isInterruption(result.nextStep))
        #expect(result.nextStep.interruptions.contains(pendingItem))
    }

    // MARK: - Calls without approvals

    @Test func testResumeExecutesNonHitlFunctionCallsWithoutOutput() async throws {
        let alreadyRan = try FunctionTool.create(name: "already_ran", argumentsType: NoArgs.self) { _, _ in "done" }
        let (_, agent) = try makeModelAndAgent(tools: [alreadyRan])
        var processed = Self.emptyProcessedResponse()
        processed.functions = [ToolRunFunction(toolCall: try Self.functionCall(alreadyRan.name, callId: "call-skip"), functionTool: alreadyRan)]

        let result = try await Self.resolveTurn(agent: agent, originalInput: "resume run", processedResponse: processed)

        #expect(Self.isRunAgain(result.nextStep))
        #expect(
            Self.toolOutputs(result.newStepItems).contains { $0.output as? String == "done" },
            "Non-HITL tools should run on resume when output is missing"
        )
    }

    @Test func testResumeSkipsNonHitlFunctionCallsWithExistingOutput() async throws {
        let alreadyRan = try FunctionTool.create(name: "already_ran", argumentsType: NoArgs.self) { _, _ in "done" }
        let (_, agent) = try makeModelAndAgent(tools: [alreadyRan])
        let functionCall = try Self.functionCall(alreadyRan.name, callId: "call-skip")
        var processed = Self.emptyProcessedResponse()
        processed.functions = [ToolRunFunction(toolCall: functionCall, functionTool: alreadyRan)]
        let contextWrapper = makeContextWrapper()
        try contextWrapper.approveTool(ToolApprovalItem(agent: agent, rawItem: functionCall.json, toolName: alreadyRan.name), alwaysApprove: true)
        let originalPreStepItems: [RunItem] = [
            ToolCallOutputItem(agent: agent, rawItem: ["type": "function_call_output", "call_id": "call-skip", "output": "prior run"], output: "prior run"),
        ]

        let result = try await Self.resolveTurn(
            agent: agent, originalInput: "resume run", originalPreStepItems: originalPreStepItems,
            processedResponse: processed, contextWrapper: contextWrapper
        )

        #expect(Self.isRunAgain(result.nextStep))
        #expect(result.newStepItems.isEmpty, "Existing outputs should prevent re-execution on resume")
    }

    @Test func testResumeSkipsShellCallsWithExistingOutput() async throws {
        let shellTool = try ShellTool(executor: { _ in .text("should_not_run") }, needsApproval: .required(true))
        let (_, agent) = try makeModelAndAgent(tools: [shellTool])
        let shellCall = makeShellCall("call_shell_resume", idValue: "shell_resume", commands: ["echo done"], status: "completed")
        var processed = Self.emptyProcessedResponse()
        processed.shellCalls = [ToolRunShellCall(toolCall: shellCall, shellTool: shellTool)]
        let originalPreStepItems: [RunItem] = [
            ToolCallOutputItem(
                agent: agent,
                rawItem: ["type": "shell_call_output", "call_id": "call_shell_resume", "status": "completed", "output": "prior run"],
                output: "prior run"
            ),
        ]

        let result = try await Self.resolveTurn(
            agent: agent, originalInput: "resume shell", originalPreStepItems: originalPreStepItems, processedResponse: processed
        )

        #expect(Self.isRunAgain(result.nextStep))
        #expect(result.newStepItems.isEmpty, "Shell call should not run when output already exists")
    }

    @Test func testResumeValidatesChangedShellBeforeSiblingApprovalCallback() async throws {
        let checkerCalls = SharedBox<[String]>([])
        let siblingTool = try FunctionTool.create(
            name: "sibling_tool",
            argumentsType: NoArgs.self,
            needsApproval: .dynamic { _, _, callId in
                checkerCalls.mutate { $0.append(callId) }
                return false
            }
        ) { _, _ in "should-not-run" }
        let shellTool = try ShellTool(executor: { _ in .text("should-not-run") }, needsApproval: .required(true))
        let enabledCalls = SharedBox<[String]>([])
        let target = Agent(name: "target")
        let targetHandoff = handoff(target, isEnabled: .dynamic { _, _ in
            enabledCalls.mutate { $0.append("handoff") }
            return true
        })
        let agent = Agent(name: "agent", tools: [siblingTool, shellTool], handoffs: [.handoff(targetHandoff)])
        let contextWrapper = makeContextWrapper()
        let approvedShellCall = makeShellCall("call-reused", commands: ["echo safe"], status: "completed")
        let changedShellCall = makeShellCall("call-reused", commands: ["echo changed"], status: "completed")
        try contextWrapper.approveTool(ToolApprovalItem(agent: agent, rawItem: approvedShellCall, toolName: shellTool.name))
        var processed = Self.emptyProcessedResponse()
        processed.functions = [ToolRunFunction(toolCall: try Self.functionCall(siblingTool.name, callId: "call-sibling"), functionTool: siblingTool)]
        processed.shellCalls = [ToolRunShellCall(toolCall: changedShellCall, shellTool: shellTool)]
        let originalPreStepItems: [RunItem] = [
            ToolCallOutputItem(
                agent: agent,
                rawItem: ["type": "shell_call_output", "call_id": "call-reused", "status": "completed", "output": "prior run"],
                output: "prior run"
            ),
        ]

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Self.resolveTurn(
                agent: agent, originalInput: "resume run", originalPreStepItems: originalPreStepItems,
                processedResponse: processed, contextWrapper: contextWrapper
            )
        }

        #expect(checkerCalls.value.isEmpty)
        #expect(enabledCalls.value.isEmpty)
    }

    @Test func testResumeKeepsApprovedShellOutputsWithPendingInterruptions() async throws {
        let pendingTool = try FunctionTool.create(name: "pending_tool", argumentsType: NoArgs.self, needsApproval: true) { _, _ in "ok" }
        let shellTool = try ShellTool(executor: { _ in .text("shell-ok") }, needsApproval: .required(true))
        let (_, agent) = try makeModelAndAgent(tools: [pendingTool, shellTool])
        let contextWrapper = makeContextWrapper()
        let functionCall = try Self.functionCall(pendingTool.name, callId: "call-pending")
        let shellCall = makeShellCall("call_shell_ok", idValue: "shell_ok", commands: ["echo ok"], status: "completed")
        try contextWrapper.approveTool(ToolApprovalItem(agent: agent, rawItem: shellCall, toolName: shellTool.name))
        let pendingApproval = ToolApprovalItem(agent: agent, rawItem: functionCall.json, toolName: pendingTool.name)
        let runState = makeStateWithInterruptions(agent, [pendingApproval])
        var processed = Self.emptyProcessedResponse()
        processed.functions = [ToolRunFunction(toolCall: functionCall, functionTool: pendingTool)]
        processed.shellCalls = [ToolRunShellCall(toolCall: shellCall, shellTool: shellTool)]

        let result = try await Self.resolveTurn(
            agent: agent, originalInput: "resume shell with pending approval", processedResponse: processed,
            contextWrapper: contextWrapper, runState: runState
        )

        #expect(Self.isInterruption(result.nextStep))
        let shellOutputs = Self.toolOutputs(result.newStepItems).filter {
            $0.rawItem.type == "shell_call_output" && $0.rawItem.string("call_id") == "call_shell_ok"
        }
        #expect(!shellOutputs.isEmpty, "Approved shell output should be included with pending interruptions")
    }

    // MARK: - Computer actions

    static func screenshotCall(_ callId: String, status: String) throws -> ResponseComputerToolCall {
        try #require(ResponseComputerToolCall(json: [
            "type": "computer_call",
            "id": .string(callId),
            "call_id": .string(callId),
            "status": .string(status),
            "action": ["type": "screenshot"],
            "pending_safety_checks": [],
        ]))
    }

    @Test func testResumeExecutesPendingComputerActions() async throws {
        let computer = TrackingComputer()
        let computerTool = ComputerTool(computer: computer)
        let (_, agent) = try makeModelAndAgent(tools: [computerTool])
        var processed = Self.emptyProcessedResponse()
        processed.computerActions = [ToolRunComputerAction(toolCall: try Self.screenshotCall("comp_pending", status: "in_progress"), computerTool: computerTool)]
        processed.toolsUsed = [computerTool.name]

        let result = try await Self.resolveTurn(agent: agent, originalInput: "resume computer", processedResponse: processed)

        let outputs = Self.toolOutputs(result.newStepItems).filter { $0.rawItem.type == "computer_call_output" }
        #expect(!outputs.isEmpty, "Computer action should run when resuming without prior output")
        #expect(!computer.calls.isEmpty, "Computer should have been invoked")
        #expect(Self.isRunAgain(result.nextStep))
    }

    @Test func testResumeSkipsComputerActionsWithExistingOutput() async throws {
        let computer = TrackingComputer()
        let computerTool = ComputerTool(computer: computer)
        let (_, agent) = try makeModelAndAgent(tools: [computerTool])
        var processed = Self.emptyProcessedResponse()
        processed.computerActions = [ToolRunComputerAction(toolCall: try Self.screenshotCall("comp_skip", status: "completed"), computerTool: computerTool)]
        processed.toolsUsed = [computerTool.name]
        let originalPreStepItems: [RunItem] = [
            ToolCallOutputItem(
                agent: agent,
                rawItem: [
                    "type": "computer_call_output",
                    "call_id": "comp_skip",
                    "output": ["type": "computer_screenshot", "image_url": "data:image/png;base64,ok"],
                ],
                output: "image_url"
            ),
        ]

        let result = try await Self.resolveTurn(
            agent: agent, originalInput: "resume computer existing", originalPreStepItems: originalPreStepItems, processedResponse: processed
        )

        #expect(computer.calls.isEmpty, "Computer action should not run when output already exists")
        #expect(result.newStepItems.isEmpty, "No new items should be emitted when output exists")
        #expect(Self.isRunAgain(result.nextStep))
    }

    // MARK: - Shell rejections

    @Test func testRejectedShellCallsEmitRejectionOutput() async throws {
        let shellTool = try ShellTool(executor: { _ in .text("should_not_run") }, needsApproval: .required(true))
        let (_, agent) = try makeModelAndAgent(tools: [shellTool])
        let contextWrapper = makeContextWrapper()
        let shellCall = makeShellCall("call_reject_shell", idValue: "shell_reject", commands: ["echo test"], status: "in_progress")
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: shellCall, toolName: shellTool.name)
        try contextWrapper.rejectTool(approvalItem)
        var processed = Self.emptyProcessedResponse()
        processed.shellCalls = [ToolRunShellCall(toolCall: shellCall, shellTool: shellTool)]

        let result = try await Self.resolveTurn(
            agent: agent, originalInput: "resume shell rejection", processedResponse: processed,
            contextWrapper: contextWrapper, runState: makeStateWithInterruptions(agent, [approvalItem])
        )

        let rejectionOutputs = Self.toolOutputs(result.newStepItems).filter { item in
            guard item.rawItem.type == "shell_call_output", let first = item.rawItem["output"]?.arrayValue?.first?.objectValue else { return false }
            return first.string("stderr") == hitlRejectionMessage
        }
        #expect(!rejectionOutputs.isEmpty, "Rejected shell call should yield rejection output")
        #expect(Self.isRunAgain(result.nextStep))
    }

    @Test func testRejectedShellCallsWithExistingOutputAreNotDuplicated() async throws {
        let shellTool = try ShellTool(executor: { _ in .text("should_not_run") }, needsApproval: .required(true))
        let (_, agent) = try makeModelAndAgent(tools: [shellTool])
        let contextWrapper = makeContextWrapper()
        let shellCall = makeShellCall("call_reject_shell_dup", idValue: "shell_reject_dup", commands: ["echo test"], status: "in_progress")
        try contextWrapper.rejectTool(ToolApprovalItem(agent: agent, rawItem: shellCall, toolName: shellTool.name))
        var processed = Self.emptyProcessedResponse()
        processed.shellCalls = [ToolRunShellCall(toolCall: shellCall, shellTool: shellTool)]
        let originalPreStepItems: [RunItem] = [
            ToolCallOutputItem(
                agent: agent,
                rawItem: [
                    "type": "shell_call_output",
                    "call_id": "call_reject_shell_dup",
                    "output": [["stdout": "", "stderr": .string(hitlRejectionMessage), "outcome": ["type": "exit", "exit_code": 1]]],
                ],
                output: hitlRejectionMessage
            ),
        ]

        let result = try await Self.resolveTurn(
            agent: agent, originalInput: "resume shell rejection existing", originalPreStepItems: originalPreStepItems,
            processedResponse: processed, contextWrapper: contextWrapper
        )

        let duplicateRejections = Self.toolOutputs(result.newStepItems).filter {
            $0.rawItem.type == "shell_call_output" && String(describing: $0.output).contains(hitlRejectionMessage)
        }
        #expect(duplicateRejections.isEmpty, "No duplicate rejection outputs should be emitted")
        #expect(Self.isRunAgain(result.nextStep))
    }

    // MARK: - Concurrent decisions

    @Test(arguments: [true, false])
    func testResumeApplyPatchUsesConcurrentDecisionWithoutReinterrupting(approved: Bool) async throws {
        let checkerStarted = AsyncEvent()
        let releaseChecker = AsyncEvent()
        let checkedPaths = SharedBox<[String]>([])
        let editor = RecordingEditor()
        let applyPatchTool = try ApplyPatchTool(editor: editor, needsApproval: .dynamic { _, operation, _ in
            let count = checkedPaths.mutate { paths -> Int in
                paths.append(operation.path)
                return paths.count
            }
            if count > 1 { throw TestError("resolved rejection must stop later approval callbacks") }
            checkerStarted.set()
            try await releaseChecker.wait()
            return false
        })
        let (_, publicAgent) = try makeModelAndAgent(tools: [applyPatchTool])
        let rawItem: JSONObject = [
            "type": "apply_patch_call",
            "call_id": "call-concurrent-apply-patch",
            "operations": [
                ["type": "update_file", "path": "first.txt", "diff": "-old\n+new\n"],
                ["type": "delete_file", "path": "second.txt"],
            ],
        ]
        let approvalItem = ToolApprovalItem(agent: publicAgent, rawItem: rawItem, toolName: applyPatchTool.name)
        let contextWrapper = makeContextWrapper()
        var processed = Self.emptyProcessedResponse()
        processed.applyPatchCalls = [ToolRunApplyPatchCall(toolCall: rawItem, applyPatchTool: applyPatchTool)]
        let runState = makeStateWithInterruptions(publicAgent, [approvalItem])
        let resolution = Task {
            try await Self.resolveTurn(
                agent: publicAgent, originalInput: "resume apply patch", processedResponse: processed,
                contextWrapper: contextWrapper, runState: runState
            )
        }
        defer {
            releaseChecker.set()
            resolution.cancel()
        }

        try await checkerStarted.wait(timeout: 5)
        if approved {
            try contextWrapper.approveTool(approvalItem)
        } else {
            try contextWrapper.rejectTool(approvalItem)
        }
        releaseChecker.set()
        let result = try await resolution.value

        #expect(checkedPaths.value == ["first.txt"])
        #expect(!Self.isInterruption(result.nextStep))
        let rejectionOutputs = Self.toolOutputs(result.newStepItems).filter { $0.output as? String == hitlRejectionMessage }
        if approved {
            #expect(rejectionOutputs.isEmpty)
            #expect(editor.operations.count == 2)
        } else {
            #expect(rejectionOutputs.count == 1)
            #expect(editor.operations.isEmpty)
        }
    }

    @Test(arguments: ToolKind.allCases)
    func testResumePreservesApprovalCreatedDuringToolExecution(toolKind: ToolKind) async throws {
        let checkerCalls = SharedBox<Int>(0)
        let executed = SharedBox<[String]>([])
        let contextWrapper = makeContextWrapper()
        var processed = Self.emptyProcessedResponse()
        @Sendable func needsApproval() -> Bool {
            checkerCalls.mutate { calls -> Bool in
                calls += 1
                return calls == 2
            }
        }
        let editor = RecordingEditor()
        let tool: any Tool
        let rawItem: JSONObject
        switch toolKind {
        case .shell:
            let shellTool = try ShellTool(
                executor: { _ in
                    executed.mutate { $0.append("shell") }
                    return .text("should-not-run")
                },
                needsApproval: .dynamic { _, _, _ in needsApproval() }
            )
            rawItem = makeShellCall("call-execution-approval")
            processed.shellCalls = [ToolRunShellCall(toolCall: rawItem, shellTool: shellTool)]
            tool = shellTool
        case .applyPatch:
            let applyPatchTool = try ApplyPatchTool(editor: editor, needsApproval: .dynamic { _, _, _ in needsApproval() })
            rawItem = makeApplyPatchDict("call-execution-approval")
            processed.applyPatchCalls = [ToolRunApplyPatchCall(toolCall: rawItem, applyPatchTool: applyPatchTool)]
            tool = applyPatchTool
        }
        let (_, publicAgent) = try makeModelAndAgent(tools: [tool])
        let originalApproval = ToolApprovalItem(agent: publicAgent, rawItem: rawItem, toolName: tool.name)

        let result = try await Self.resolveTurn(
            agent: publicAgent, originalInput: "resume approval", processedResponse: processed,
            contextWrapper: contextWrapper, runState: makeStateWithInterruptions(publicAgent, [originalApproval])
        )

        #expect(checkerCalls.value == 2)
        #expect(Self.isInterruption(result.nextStep))
        #expect(result.nextStep.interruptions.map { $0.rawItem.string("call_id") } == ["call-execution-approval"])
        #expect(executed.value.isEmpty)
        #expect(editor.operations.isEmpty)
    }

    // MARK: - Handoffs

    @Test func testResumeRejectsChangedHandoffUnderApprovedFunctionCallId() async throws {
        let target = Agent(name: "target")
        let routeHandoff = handoff(target, toolNameOverride: "route")
        let agent = Agent(name: "agent", handoffs: [.handoff(routeHandoff)])
        let approvedCall = makeFunctionToolCall("route", callId: "call-shared", arguments: #"{"destination":"safe"}"#)
        let changedCall = try Self.functionCall("route", callId: "call-shared", arguments: #"{"destination":"safe"}"#)
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: approvedCall, toolName: "route")
        let contextWrapper = makeContextWrapper()
        try contextWrapper.approveTool(approvalItem, alwaysApprove: true)
        var processed = Self.emptyProcessedResponse()
        processed.handoffs = [ToolRunHandoff(handoff: routeHandoff, toolCall: changedCall)]

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Self.resolveTurn(
                agent: agent,
                originalInput: "resume handoff",
                originalPreStepItems: [approvalItem],
                newResponse: ModelResponse(output: [changedCall.json], usage: Usage(), responseId: "resp"),
                processedResponse: processed,
                contextWrapper: contextWrapper,
                runState: makeStateWithInterruptions(agent, [approvalItem])
            )
        }
    }

    // MARK: - Hosted MCP callbacks

    @Test func testMcpCallbackApprovalsAreProcessed() async throws {
        let agent = makeAgent()
        let mcpTool = HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "server"]) { _ in
            MCPToolApprovalFunctionResult(approve: true, reason: "ok")
        }
        var processed = Self.emptyProcessedResponse()
        processed.mcpApprovalRequests = [ToolRunMCPApprovalRequest(requestItem: try Self.mcpApprovalRequest("mcp-callback-1"), mcpTool: mcpTool)]

        let result = try await Self.resolveTurn(agent: agent, originalInput: "handle mcp", processedResponse: processed)

        #expect(
            result.newStepItems.contains { ($0 as? MCPApprovalResponseItem)?.rawItem["approve"] == true },
            "MCP callback approvals should emit approval responses"
        )
        #expect(Self.isRunAgain(result.nextStep))
    }

    static func mcpApprovalRequest(_ id: String, arguments: String = "{}") throws -> McpApprovalRequest {
        try #require(McpApprovalRequest(json: [
            "id": .string(id),
            "type": "mcp_approval_request",
            "server_label": "server",
            "arguments": .string(arguments),
            "name": "hosted_mcp",
        ]))
    }
}
