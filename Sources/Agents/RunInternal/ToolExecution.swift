import Foundation

// Port of the function tool, approval, and hosted MCP parts of
// src/agents/run_internal/tool_execution.py and tool_planning.py.
//
// Swift: the invocation-identity ledger that rejects call-ID reuse across resumes, nested
// agent-tool interruption routing, and the cooperative cancellation bookkeeping for sibling tool
// tasks are not ported; see porting/manifest.json.

/// Decides whether a function tool call requires approval, parsing the arguments for callable
/// policies. Invalid arguments require manual approval.
func functionNeedsApproval(_ tool: FunctionTool, _ contextWrapper: RunContextWrapper, _ toolCall: ResponseFunctionToolCall) async throws -> Bool {
    switch tool.needsApproval {
    case .required(let value):
        return value
    case .dynamic(let policy):
        guard case .object(let arguments)? = try? JSONValue.parse(toolCall.arguments.isEmpty ? "{}" : toolCall.arguments) else {
            return true
        }
        do {
            return try await policy(contextWrapper, arguments, toolCall.callId)
        } catch let error as UserError {
            throw error
        } catch {
            return true
        }
    }
}

/// Runs a function tool's input guardrails and returns a rejection message if any.
func executeToolInputGuardrails(
    tool: FunctionTool,
    toolContext: ToolContext,
    agent: Agent,
    results: SharedBox<[ToolInputGuardrailResult]>
) async throws -> String? {
    guard let guardrails = tool.toolInputGuardrails, !guardrails.isEmpty else { return nil }
    for guardrail in guardrails {
        let output = try await guardrail.run(ToolInputGuardrailData(context: toolContext, agent: agent))
        results.mutate { $0.append(ToolInputGuardrailResult(guardrail: guardrail, output: output)) }
        switch output.behavior {
        case .raiseException: throw ToolInputGuardrailTripwireTriggered(guardrail: guardrail, output: output)
        case .rejectContent(let message): return message
        case .allow: continue
        }
    }
    return nil
}

/// Runs a function tool's output guardrails and returns the final result.
func executeToolOutputGuardrails(
    tool: FunctionTool,
    toolContext: ToolContext,
    agent: Agent,
    realResult: any Sendable,
    results: SharedBox<[ToolOutputGuardrailResult]>
) async throws -> (output: any Sendable, isRejection: Bool) {
    guard let guardrails = tool.toolOutputGuardrails, !guardrails.isEmpty else { return (realResult, false) }
    for guardrail in guardrails {
        let output = try await guardrail.run(ToolOutputGuardrailData(context: toolContext, agent: agent, output: realResult))
        results.mutate { $0.append(ToolOutputGuardrailResult(guardrail: guardrail, output: output)) }
        switch output.behavior {
        case .raiseException: throw ToolOutputGuardrailTripwireTriggered(guardrail: guardrail, output: output)
        case .rejectContent(let message): return (message, true)
        case .allow: continue
        }
    }
    return (realResult, false)
}

/// Executes function tool calls, honoring approvals, guardrails, hooks, and concurrency limits.
/// Signals that a dispatched function tool reached its invocation point (or finished early).
///
/// Swift: asyncio starts tasks in creation order; Swift task-group children start in any order.
/// The batch executor waits on this signal before dispatching the next queued tool, so tools
/// start in FIFO order as upstream guarantees.
final class ToolDispatchSignal: @unchecked Sendable {
    @TaskLocal static var current: ToolDispatchSignal?

    private let lock = NSLock()
    private var signaled = false
    private var waiter: CheckedContinuation<Void, Never>?

    func signal() {
        let continuation: CheckedContinuation<Void, Never>? = lock.locked {
            guard !signaled else { return nil }
            signaled = true
            defer { waiter = nil }
            return waiter
        }
        continuation?.resume()
    }

    func wait() async {
        await withCheckedContinuation { continuation in
            let resumeNow: Bool = lock.locked {
                if signaled { return true }
                waiter = continuation
                return false
            }
            if resumeNow { continuation.resume() }
        }
    }
}

struct FunctionToolBatchExecutor {
    let agent: Agent
    let toolRuns: [ToolRunFunction]
    let hooks: RunHooks
    let contextWrapper: RunContextWrapper
    let config: RunConfig
    let availableFunctionTools: [any Tool]
    let inputGuardrailResults = SharedBox<[ToolInputGuardrailResult]>([])
    let outputGuardrailResults = SharedBox<[ToolOutputGuardrailResult]>([])

    func execute() async throws -> ([FunctionToolResult], [ToolInputGuardrailResult], [ToolOutputGuardrailResult]) {
        guard !toolRuns.isEmpty else { return ([], [], []) }
        let limit = config.toolExecution?.maxFunctionToolConcurrency ?? toolRuns.count
        var results = [FunctionToolResult?](repeating: nil, count: toolRuns.count)
        try await withThrowingTaskGroup(of: (Int, FunctionToolResult).self) { group in
            var nextIndex = 0
            func startNext() async {
                guard nextIndex < toolRuns.count else { return }
                let index = nextIndex
                let run = toolRuns[index]
                nextIndex += 1
                let dispatched = ToolDispatchSignal()
                group.addTask {
                    defer { dispatched.signal() }
                    return try await ToolDispatchSignal.$current.withValue(dispatched) {
                        (index, try await runSingleTool(run))
                    }
                }
                await dispatched.wait()
            }
            for _ in 0..<min(limit, toolRuns.count) { await startNext() }
            while let (index, result) = try await group.next() {
                results[index] = result
                await startNext()
            }
        }
        return (results.compactMap { $0 }, inputGuardrailResults.value, outputGuardrailResults.value)
    }

    private func runSingleTool(_ run: ToolRunFunction) async throws -> FunctionToolResult {
        let tool = run.functionTool
        let rawToolCall = run.toolCall
        let toolCall = ResponseFunctionToolCall(json: normalizeToolCallForFunctionTool(rawToolCall.json, tool: tool)) ?? rawToolCall
        let traceName = getToolCallTraceName(toolCall.json) ?? getFunctionToolTraceName(tool) ?? tool.name
        let span = functionSpan(name: traceName)
        return try await span.run {
            let toolContext = ToolContext.fromAgentContext(
                contextWrapper,
                toolCallId: toolCall.callId,
                toolCall: rawToolCall.json,
                agent: agent,
                toolNamespace: getToolCallNamespace(rawToolCall.json) ?? getToolCallNamespace(toolCall.json),
                runConfig: config
            )
            if config.traceIncludeSensitiveData {
                span.data(as: FunctionSpanData.self)?.input = toolCall.arguments
            }
            let result: FunctionToolResult
            do {
                if let approvalResult = try await maybeExecuteToolApproval(tool: tool, toolCall: toolCall, rawToolCall: rawToolCall, span: span, toolContext: toolContext) {
                    result = approvalResult
                } else {
                    result = try await executeSingleToolBody(tool: tool, toolCall: toolCall, toolContext: toolContext)
                }
            } catch {
                attachErrorToCurrentSpan(SpanError(message: "Error running tool", data: [
                    "tool_name": .string(tool.name),
                    "error": .string(getTraceToolError(traceIncludeSensitiveData: config.traceIncludeSensitiveData, errorMessage: describeError(error))),
                ]))
                if error is AgentsException || error is CancellationError { throw error }
                throw UserError("Error running tool \(tool.name): \(describeError(error))")
            }
            if config.traceIncludeSensitiveData {
                span.data(as: FunctionSpanData.self)?.output = result.output
            }
            return result
        }
    }

    private func maybeExecuteToolApproval(
        tool: FunctionTool,
        toolCall: ResponseFunctionToolCall,
        rawToolCall: ResponseFunctionToolCall,
        span: any Span,
        toolContext: ToolContext
    ) async throws -> FunctionToolResult? {
        var namespace = getToolCallNamespace(rawToolCall.json)
        if namespace == nil, isDeferredTopLevelFunctionTool(tool) { namespace = tool.name }
        var lookupKey = getFunctionToolLookupKeyForCall(rawToolCall.json)
        if isDeferredTopLevelFunctionTool(tool) { lookupKey = .deferredTopLevel(tool.name) }
        let approvalItem = ToolApprovalItem(
            agent: agent,
            rawItem: rawToolCall.json,
            toolName: tool.name,
            toolNamespace: namespace,
            toolOrigin: getFunctionToolOrigin(tool),
            toolLookupKey: lookupKey,
            allowBareNameAlias: shouldAllowBareNameApprovalAlias(tool, allTools: availableFunctionTools)
        )
        var status = try contextWrapper.getApprovalStatus(
            toolName: tool.name, callId: toolCall.callId, toolNamespace: namespace,
            existingPending: approvalItem, toolLookupKey: lookupKey
        )
        if status == nil {
            let needsApproval = try await functionNeedsApproval(tool, contextWrapper, toolCall)
            status = try contextWrapper.getApprovalStatus(
                toolName: tool.name, callId: toolCall.callId, toolNamespace: namespace,
                existingPending: approvalItem, toolLookupKey: lookupKey
            )
            if status == nil, !needsApproval { return nil }
        }
        if status == nil {
            if config.toolExecution?.preApprovalToolInputGuardrails == true {
                let rejected = try await executeToolInputGuardrails(tool: tool, toolContext: toolContext, agent: agent, results: inputGuardrailResults)
                status = try contextWrapper.getApprovalStatus(
                    toolName: tool.name, callId: toolCall.callId, toolNamespace: namespace,
                    existingPending: approvalItem, toolLookupKey: lookupKey
                )
                if status == nil, let rejected {
                    return FunctionToolResult(
                        tool: tool,
                        output: rejected,
                        runItem: try functionRejectionItem(
                            agent: agent, toolCall: toolCall, rejectionMessage: rejected,
                            outputJSONSchema: tool.outputJSONSchema, toolOrigin: getFunctionToolOrigin(tool)
                        )
                    )
                }
            }
        }
        if status == nil {
            return FunctionToolResult(tool: tool, output: Optional<String>.none as any Sendable, runItem: approvalItem)
        }
        if status == true { return nil }
        let message = await resolveApprovalRejectionMessage(
            contextWrapper: contextWrapper, runConfig: config, toolType: "function",
            toolName: toolTraceName(tool.name, namespace: namespace) ?? tool.name, callId: toolCall.callId,
            toolNamespace: namespace, toolLookupKey: lookupKey, existingPending: approvalItem
        )
        span.setError(SpanError(
            message: getTraceError(traceIncludeSensitiveData: config.traceIncludeSensitiveData, errorMessage: message, redactedMessage: "Tool execution rejected"),
            data: [
                "tool_name": .string(tool.name),
                "error": .string("Tool execution for \(toolCall.callId) was manually rejected by user."),
            ]
        ))
        return FunctionToolResult(
            tool: tool,
            output: message,
            runItem: try functionRejectionItem(
                agent: agent, toolCall: toolCall, rejectionMessage: message,
                outputJSONSchema: tool.outputJSONSchema, toolOrigin: getFunctionToolOrigin(tool)
            )
        )
    }

    private func executeSingleToolBody(tool: FunctionTool, toolCall: ResponseFunctionToolCall, toolContext: ToolContext) async throws -> FunctionToolResult {
        if let rejected = try await executeToolInputGuardrails(tool: tool, toolContext: toolContext, agent: agent, results: inputGuardrailResults) {
            let provider = functionToolErrorOutput(toolCall: toolCall.json, output: rejected, outputJSONSchema: tool.outputJSONSchema)
            return FunctionToolResult(
                tool: tool,
                output: rejected,
                runItem: ToolCallOutputItem(
                    agent: agent,
                    rawItem: try ItemHelpers.toolCallOutputItem(toolCall, output: provider),
                    output: rejected,
                    toolOrigin: getFunctionToolOrigin(tool)
                )
            )
        }

        async let startHook: Void = hooks.onToolStart(context: toolContext, agent: agent, tool: tool)
        async let agentStartHook: Void = agent.hooks?.onToolStart(context: toolContext, agent: agent, tool: tool) ?? ()
        _ = try await (startHook, agentStartHook)

        let invocation = try await invokeFunctionToolWithMetadata(tool: tool, context: toolContext, arguments: toolCall.arguments)
        let guarded = try await executeToolOutputGuardrails(
            tool: tool, toolContext: toolContext, agent: agent, realResult: invocation.output, results: outputGuardrailResults
        )
        let bypassOutputSchema = invocation.isSDKGeneratedError || guarded.isRejection
        let providerResult = bypassOutputSchema
            ? functionToolErrorOutput(toolCall: toolCall.json, output: guarded.output, outputJSONSchema: tool.outputJSONSchema)
            : guarded.output
        let rawOutput = try ItemHelpers.toolCallOutputItem(
            toolCall, output: providerResult, outputJSONSchema: bypassOutputSchema ? nil : tool.outputJSONSchema
        )
        let outputItem = ToolCallOutputItem(agent: agent, rawItem: rawOutput, output: guarded.output, toolOrigin: getFunctionToolOrigin(tool))
        var customData = toolContext.customData
        if let extractor = tool.customDataExtractor {
            let extracted = try await extractor(FunctionToolCustomDataContext(toolContext: toolContext, tool: tool, output: guarded.output, rawItem: rawOutput))
            if let extracted {
                customData = (customData ?? [:]).merging(extracted)
            }
        }
        if let customData, !customData.isEmpty {
            outputItem.customData = customData
        }

        async let endHook: Void = hooks.onToolEnd(context: toolContext, agent: agent, tool: tool, result: guarded.output)
        async let agentEndHook: Void = agent.hooks?.onToolEnd(context: toolContext, agent: agent, tool: tool, result: guarded.output) ?? ()
        _ = try await (endHook, agentEndHook)

        return FunctionToolResult(tool: tool, output: guarded.output, runItem: outputItem)
    }
}

// MARK: - Hosted MCP approvals

/// Runs hosted MCP approval callbacks and returns approval response items.
func executeMCPApprovalRequests(agent: Agent, requests: [ToolRunMCPApprovalRequest], contextWrapper: RunContextWrapper) async throws -> [RunItem] {
    var items: [RunItem] = []
    for request in requests {
        let requestItem = request.requestItem
        let approvalItem = ToolApprovalItem(agent: agent, rawItem: requestItem.json, toolName: requestItem.name)
        var status = try contextWrapper.getApprovalStatus(toolName: requestItem.name, callId: requestItem.id, existingPending: approvalItem)
        var reason = contextWrapper.getRejectionMessage(toolName: requestItem.name, callId: requestItem.id, existingPending: approvalItem)
        if status == nil, let callback = request.mcpTool.onApprovalRequest {
            let result = try await callback(MCPToolApprovalRequest(ctxWrapper: contextWrapper, data: requestItem))
            status = result.approve
            reason = result.reason
            if result.approve {
                try contextWrapper.approveTool(approvalItem)
            } else {
                try contextWrapper.rejectTool(approvalItem, rejectionMessage: result.reason)
            }
        }
        var raw: JSONObject = [
            "approval_request_id": .string(requestItem.id),
            "approve": .bool(status ?? false),
            "type": "mcp_approval_response",
        ]
        if status != true, let reason, !reason.isEmpty { raw["reason"] = .string(reason) }
        ItemHelpers.copyToolCallCaller(requestItem.json, into: &raw)
        items.append(MCPApprovalResponseItem(agent: agent, rawItem: raw))
    }
    return items
}

/// Bridges hosted MCP approval requests with manual approvals.
func collectManualMCPApprovals(
    agent: Agent,
    requests: [ToolRunMCPApprovalRequest],
    contextWrapper: RunContextWrapper,
    existingPendingByCallId: [String: ToolApprovalItem] = [:]
) throws -> (approved: [MCPApprovalResponseItem], pending: [ToolApprovalItem]) {
    var approved: [MCPApprovalResponseItem] = []
    var pending: [ToolApprovalItem] = []
    var seen: Set<String> = []
    for request in requests {
        let requestItem = request.requestItem
        let requestId = requestItem.id
        if !requestId.isEmpty {
            guard seen.insert(requestId).inserted else { continue }
        }
        let toolName = requestItem.name.isEmpty ? request.mcpTool.name : requestItem.name
        let approvalItem = existingPendingByCallId[requestId] ?? ToolApprovalItem(agent: agent, rawItem: requestItem.json, toolName: toolName)
        let status = try contextWrapper.getApprovalStatus(toolName: toolName, callId: requestId, existingPending: approvalItem)
        if let status, !requestId.isEmpty {
            var raw: JSONObject = ["type": "mcp_approval_response", "approval_request_id": .string(requestId), "approve": .bool(status)]
            if !status, let message = contextWrapper.getRejectionMessage(toolName: toolName, callId: requestId, existingPending: approvalItem) {
                raw["reason"] = .string(message)
            }
            ItemHelpers.copyToolCallCaller(requestItem.json, into: &raw)
            approved.append(MCPApprovalResponseItem(agent: agent, rawItem: raw))
            continue
        }
        if status != nil { continue }
        pending.append(approvalItem)
    }
    return (approved, pending)
}

/// Builds a mapping of tool call IDs to pending approval items.
func indexApprovalItemsByCallId(_ items: [RunItem]) -> [String: ToolApprovalItem] {
    var approvals: [String: ToolApprovalItem] = [:]
    for case let item as ToolApprovalItem in items {
        if let callId = getToolApprovalItemCallId(item) {
            approvals[callId] = item
        }
    }
    return approvals
}

// MARK: - Tool plan

/// Tool execution work to perform in a single turn.
struct ToolExecutionPlan: Sendable {
    var functionRuns: [ToolRunFunction] = []
    var computerActions: [ToolRunComputerAction] = []
    var customToolCalls: [ToolRunCustom] = []
    var shellCalls: [ToolRunShellCall] = []
    var applyPatchCalls: [ToolRunApplyPatchCall] = []
    var localShellCalls: [ToolRunLocalShellCall] = []
    var pendingInterruptions: [ToolApprovalItem] = []
    var approvedMCPResponses: [RunItem] = []
    var mcpRequestsWithCallback: [ToolRunMCPApprovalRequest] = []
}

/// The results of executing a tool plan.
struct ToolPlanResults: Sendable {
    var functionResults: [FunctionToolResult] = []
    var toolInputGuardrailResults: [ToolInputGuardrailResult] = []
    var toolOutputGuardrailResults: [ToolOutputGuardrailResult] = []
    var computerResults: [RunItem] = []
    var customToolResults: [RunItem] = []
    var shellResults: [RunItem] = []
    var applyPatchResults: [RunItem] = []
    var localShellResults: [RunItem] = []

    /// Ordered tool result items for inclusion in new step items.
    var resultItems: [RunItem] {
        functionResults.compactMap(\.runItem) + computerResults + customToolResults + shellResults + applyPatchResults + localShellResults
    }

    /// Tool approval interruptions from tool results.
    var interruptions: [ToolApprovalItem] {
        var items: [ToolApprovalItem] = []
        for result in functionResults {
            if let approval = result.runItem as? ToolApprovalItem { items.append(approval) }
            items.append(contentsOf: result.interruptions)
        }
        for item in customToolResults + shellResults + applyPatchResults {
            if let approval = item as? ToolApprovalItem { items.append(approval) }
        }
        return items
    }
}

/// Builds a plan for a fresh turn, splitting hosted MCP approvals into callback and manual buckets.
func buildPlanForFreshTurn(
    processed: ProcessedResponse,
    agent: Agent,
    contextWrapper: RunContextWrapper,
    approvalItemsByCallId: [String: ToolApprovalItem]
) throws -> ToolExecutionPlan {
    let withCallback = processed.mcpApprovalRequests.filter { $0.mcpTool.onApprovalRequest != nil }
    let manual = processed.mcpApprovalRequests.filter { $0.mcpTool.onApprovalRequest == nil }
    let (approved, pending) = try collectManualMCPApprovals(
        agent: agent, requests: manual, contextWrapper: contextWrapper, existingPendingByCallId: approvalItemsByCallId
    )
    return ToolExecutionPlan(
        functionRuns: processed.functions,
        computerActions: processed.computerActions,
        customToolCalls: processed.customToolCalls,
        shellCalls: processed.shellCalls,
        applyPatchCalls: processed.applyPatchCalls,
        localShellCalls: processed.localShellCalls,
        pendingInterruptions: pending,
        approvedMCPResponses: approved,
        mcpRequestsWithCallback: withCallback
    )
}

/// Executes tool runs captured in a plan, running each tool family concurrently.
func executeToolPlan(
    plan: ToolExecutionPlan,
    agent: Agent,
    allTools: [any Tool],
    hooks: RunHooks,
    contextWrapper: RunContextWrapper,
    runConfig: RunConfig
) async throws -> ToolPlanResults {
    async let functions = FunctionToolBatchExecutor(
        agent: agent, toolRuns: plan.functionRuns, hooks: hooks, contextWrapper: contextWrapper,
        config: runConfig, availableFunctionTools: allTools
    ).execute()
    async let computers: [RunItem] = {
        var items: [RunItem] = []
        for action in plan.computerActions {
            var acknowledged: [JSONObject]?
            if let onSafetyCheck = action.computerTool.onSafetyCheck {
                var checks: [JSONObject] = []
                for check in action.toolCall.pendingSafetyChecks {
                    if try await onSafetyCheck(ComputerToolSafetyCheckData(ctxWrapper: contextWrapper, agent: agent, toolCall: action.toolCall, safetyCheck: check)) {
                        var acknowledgedCheck: JSONObject = ["id": check["id"] ?? .null]
                        if let code = check["code"] { acknowledgedCheck["code"] = code }
                        if let message = check["message"] { acknowledgedCheck["message"] = message }
                        checks.append(acknowledgedCheck)
                    } else {
                        throw UserError("Computer tool safety check was not acknowledged")
                    }
                }
                acknowledged = checks
            }
            items.append(try await ComputerAction.execute(
                agent: agent, action: action, hooks: hooks, contextWrapper: contextWrapper,
                config: runConfig, acknowledgedSafetyChecks: acknowledged
            ))
        }
        return items
    }()
    async let customs: [RunItem] = {
        var items: [RunItem] = []
        for call in plan.customToolCalls {
            items.append(try await CustomToolAction.execute(agent: agent, call: call, hooks: hooks, contextWrapper: contextWrapper, config: runConfig))
        }
        return items
    }()
    async let shells: [RunItem] = {
        var items: [RunItem] = []
        for call in plan.shellCalls {
            items.append(try await ShellAction.execute(agent: agent, call: call, hooks: hooks, contextWrapper: contextWrapper, config: runConfig))
        }
        return items
    }()
    async let patches: [RunItem] = {
        var items: [RunItem] = []
        for call in plan.applyPatchCalls {
            items.append(try await ApplyPatchAction.execute(agent: agent, call: call, hooks: hooks, contextWrapper: contextWrapper, config: runConfig))
        }
        return items
    }()
    async let localShells: [RunItem] = {
        var items: [RunItem] = []
        for call in plan.localShellCalls {
            items.append(try await LocalShellAction.execute(agent: agent, call: call, hooks: hooks, contextWrapper: contextWrapper, config: runConfig))
        }
        return items
    }()
    let (functionResults, inputResults, outputResults) = try await functions
    return ToolPlanResults(
        functionResults: functionResults,
        toolInputGuardrailResults: inputResults,
        toolOutputGuardrailResults: outputResults,
        computerResults: try await computers,
        customToolResults: try await customs,
        shellResults: try await shells,
        applyPatchResults: try await patches,
        localShellResults: try await localShells
    )
}
