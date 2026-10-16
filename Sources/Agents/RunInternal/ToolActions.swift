import Foundation

// Port of src/agents/run_internal/tool_actions.py and the shell/apply_patch helpers in
// tool_execution.py.

/// Runs `body` in a function span when tracing is active.
func withToolFunctionSpan<T: Sendable>(
    config: RunConfig,
    toolName: String,
    run: (_ span: (any Span)?) async throws -> T
) async throws -> T {
    if config.tracingDisabled || getCurrentTrace() == nil {
        return try await run(nil)
    }
    let span = functionSpan(name: toolName)
    return try await span.run { try await run(span) }
}

func formatShellError(_ error: any Error) -> String {
    let message = describeError(error)
    return message.isEmpty ? errorTypeName(error) : message
}

private func callHooks(
    _ hooks: RunHooks,
    _ agent: Agent,
    context: RunContextWrapper,
    tool: any Tool,
    start: Bool,
    result: any Sendable = ""
) async throws {
    if start {
        async let runHook: Void = hooks.onToolStart(context: context, agent: agent, tool: tool)
        async let agentHook: Void = agent.hooks?.onToolStart(context: context, agent: agent, tool: tool) ?? ()
        _ = try await (runHook, agentHook)
    } else {
        async let runHook: Void = hooks.onToolEnd(context: context, agent: agent, tool: tool, result: result)
        async let agentHook: Void = agent.hooks?.onToolEnd(context: context, agent: agent, tool: tool, result: result) ?? ()
        _ = try await (runHook, agentHook)
    }
}

/// Builds an approval item, runs the `onApproval` hook if needed, and returns the latest status.
func resolveApprovalStatus(
    toolName: String,
    callId: String,
    rawItem: JSONObject,
    agent: Agent,
    contextWrapper: RunContextWrapper,
    onApproval: (@Sendable (RunContextWrapper, ToolApprovalItem) async throws -> ToolOnApprovalResult)?
) async throws -> (Bool?, ToolApprovalItem) {
    let item = ToolApprovalItem(agent: agent, rawItem: rawItem, toolName: toolName)
    var status = try contextWrapper.getApprovalStatus(toolName: toolName, callId: callId, existingPending: item)
    if status == nil, let onApproval {
        let decision = try await onApproval(contextWrapper, item)
        if decision.approve {
            try contextWrapper.approveTool(item)
        } else {
            let reason = (decision.reason?.isEmpty ?? true) ? nil : decision.reason
            try contextWrapper.rejectTool(item, rejectionMessage: reason)
        }
        status = try contextWrapper.getApprovalStatus(toolName: toolName, callId: callId, existingPending: item)
    }
    return (status, item)
}

/// Resolves the model-visible output text for approval rejections.
func resolveApprovalRejectionMessage(
    contextWrapper: RunContextWrapper,
    runConfig: RunConfig,
    toolType: String,
    toolName: String,
    callId: String,
    toolNamespace: String? = nil,
    toolLookupKey: FunctionToolLookupKey? = nil,
    existingPending: ToolApprovalItem? = nil
) async -> String {
    if let explicit = contextWrapper.getRejectionMessage(
        toolName: toolName, callId: callId, toolNamespace: toolNamespace,
        existingPending: existingPending, toolLookupKey: toolLookupKey
    ) {
        return explicit
    }
    guard let formatter = runConfig.toolErrorFormatter else { return rejectionMessage }
    do {
        let message = try await formatter(ToolErrorFormatterArgs(
            kind: "approval_rejected", toolType: toolType, toolName: toolName, callId: callId,
            defaultMessage: rejectionMessage, runContext: contextWrapper
        ))
        return message ?? rejectionMessage
    } catch {
        logActionError("Tool error formatter failed", error, isToolAction: true)
        return rejectionMessage
    }
}

// MARK: - Computer

enum ComputerAction {
    static let traceToolName = "computer"

    static func execute(
        agent: Agent,
        action: ToolRunComputerAction,
        hooks: RunHooks,
        contextWrapper: RunContextWrapper,
        config: RunConfig,
        acknowledgedSafetyChecks: [JSONObject]? = nil
    ) async throws -> RunItem {
        try await withToolFunctionSpan(config: config, toolName: action.computerTool.traceName) { span in
            if let span, config.traceIncludeSensitiveData {
                span.data(as: FunctionSpanData.self)?.input = traceInputPayload(action.toolCall)
            }
            let computer = try await action.computerTool.resolveComputer(runContext: contextWrapper)
            try await callHooks(hooks, agent, context: contextWrapper, tool: action.computerTool, start: true)
            let output: String
            do {
                output = try await executeActionAndCapture(computer, action.toolCall)
            } catch {
                span?.setError(SpanError(message: "Error running tool", data: [
                    "tool_name": .string(traceToolName),
                    "error": .string(getTraceToolError(traceIncludeSensitiveData: config.traceIncludeSensitiveData, errorMessage: formatShellError(error))),
                ]))
                logActionError("Failed to execute computer action", error, isToolAction: true)
                throw error
            }
            let imageURL = output.isEmpty ? "" : "data:image/png;base64,\(output)"
            var raw: JSONObject = [
                "call_id": .string(action.toolCall.callId),
                "output": ["type": "computer_screenshot", "image_url": .string(imageURL)],
                "type": "computer_call_output",
            ]
            if let acknowledgedSafetyChecks {
                raw["acknowledged_safety_checks"] = .array(acknowledgedSafetyChecks.map(JSONValue.object))
            }
            let item = ToolCallOutputItem(agent: agent, rawItem: raw, output: imageURL)
            try await callHooks(hooks, agent, context: contextWrapper, tool: action.computerTool, start: false, result: output)
            if let span, config.traceIncludeSensitiveData {
                span.data(as: FunctionSpanData.self)?.output = imageURL
            }
            return item
        }
    }

    private static func actions(_ toolCall: ResponseComputerToolCall) -> [JSONObject] {
        if let actions = toolCall.actions, !actions.isEmpty { return actions }
        if let action = toolCall.action { return [action] }
        return []
    }

    private static func traceInputPayload(_ toolCall: ResponseComputerToolCall) -> String? {
        let list = actions(toolCall)
        if let batched = toolCall.actions, !batched.isEmpty {
            return JSONValue.array(list.map(JSONValue.object)).pythonJSONString()
        }
        return list.first.map { JSONValue.object($0).pythonJSONString() }
    }

    static func executeActionAndCapture(_ computer: any Computer, _ toolCall: ResponseComputerToolCall) async throws -> String {
        var lastWasScreenshot = false
        var lastScreenshot = ""
        for action in actions(toolCall) {
            lastWasScreenshot = false
            let x = action["x"]?.intValue ?? 0
            let y = action["y"]?.intValue ?? 0
            switch action.string("type") {
            case "click":
                try await computer.click(x: x, y: y, button: action.string("button") ?? "left")
            case "double_click":
                try await computer.doubleClick(x: x, y: y)
            case "drag":
                let path = (action["path"]?.arrayValue ?? []).map { (x: $0["x"]?.intValue ?? 0, y: $0["y"]?.intValue ?? 0) }
                try await computer.drag(path: path)
            case "keypress":
                try await computer.keypress(keys: action["keys"]?.arrayValue?.compactMap(\.stringValue) ?? [])
            case "move":
                try await computer.move(x: x, y: y)
            case "screenshot":
                lastScreenshot = try await computer.screenshot()
                lastWasScreenshot = true
            case "scroll":
                try await computer.scroll(x: x, y: y, scrollX: action["scroll_x"]?.intValue ?? 0, scrollY: action["scroll_y"]?.intValue ?? 0)
            case "type":
                try await computer.type(text: action.string("text") ?? "")
            case "wait":
                try await computer.wait()
            case let other:
                throw ModelBehaviorError("Computer tool returned unknown action type '\(other ?? "None")'")
            }
        }
        if lastWasScreenshot { return lastScreenshot }
        return try await computer.screenshot()
    }
}

// MARK: - Local shell

enum LocalShellAction {
    static func execute(agent: Agent, call: ToolRunLocalShellCall, hooks: RunHooks, contextWrapper: RunContextWrapper, config: RunConfig) async throws -> RunItem {
        try await callHooks(hooks, agent, context: contextWrapper, tool: call.localShellTool, start: true)
        let result = try await call.localShellTool.executor(LocalShellCommandRequest(ctxWrapper: contextWrapper, data: call.toolCall))
        let raw: JSONObject = ["type": "local_shell_call_output", "call_id": .string(call.toolCall.callId), "output": .string(result)]
        let item = ToolCallOutputItem(agent: agent, rawItem: raw, output: result)
        try await callHooks(hooks, agent, context: contextWrapper, tool: call.localShellTool, start: false, result: result)
        return item
    }
}

// MARK: - Shell

func coerceShellCall(_ toolCall: JSONObject) throws -> ShellCallData {
    guard let callId = toolCall.string("call_id"), !callId.isEmpty else {
        throw ModelBehaviorError("Shell call is missing call_id.")
    }
    guard case .object(let action)? = toolCall["action"] else {
        throw ModelBehaviorError("Shell call is missing an action payload.")
    }
    guard case .array(let commandValues)? = action["commands"] else {
        throw ModelBehaviorError("Shell call action commands must be a sequence of command strings.")
    }
    let commands = commandValues.compactMap { value -> String? in
        if value.isNull { return nil }
        return value.stringValue ?? value.jsonString()
    }
    if commands.isEmpty {
        throw ModelBehaviorError("Shell call action must include at least one command.")
    }
    var timeoutMs: Int?
    for key in ["timeout_ms", "timeoutMs", "timeout"] {
        guard let candidate = action[key], !candidate.isNull, candidate != .int(0) else { continue }
        guard let int = candidate.intValue, candidate.boolValue == nil, int >= 0 else {
            throw ModelBehaviorError("Shell call action timeout must be a positive integer in milliseconds, zero, or None.")
        }
        timeoutMs = int
        break
    }
    let maxLength = (action["max_output_length"] ?? action["maxOutputLength"])?.doubleValue.map { Int($0) }
    let status = toolCall.string("status")?.lowercased()
    return ShellCallData(
        callId: callId,
        action: ShellActionRequest(commands: commands, timeoutMs: timeoutMs, maxOutputLength: maxLength),
        status: (status == "in_progress" || status == "completed") ? status : nil,
        raw: toolCall
    )
}

func normalizeShellOutput(_ entry: ShellCommandOutput) -> ShellCommandOutput { entry }

func serializeShellOutput(_ output: ShellCommandOutput) -> JSONObject {
    var outcome: JSONObject = ["type": .string(output.outcome.type)]
    var payload: JSONObject = ["stdout": .string(output.stdout), "stderr": .string(output.stderr), "status": .string(output.status)]
    if output.outcome.type == "exit" {
        outcome["exit_code"] = output.outcome.exitCode.map(JSONValue.int) ?? .null
    }
    payload["outcome"] = .object(outcome)
    if output.outcome.type == "exit", let exitCode = output.outcome.exitCode { payload["exit_code"] = .int(exitCode) }
    if let command = output.command { payload["command"] = .string(command) }
    if let providerData = output.providerData, !providerData.isEmpty { payload["provider_data"] = .object(providerData) }
    return payload
}

func resolveExitCode(_ raw: JSONValue?, _ outcomeStatus: String?) -> Int {
    if let code = raw?.intValue { return code }
    switch outcomeStatus?.lowercased() {
    case "failure": return 1
    default: return 0
    }
}

/// Renders shell outputs into human-readable text for tool responses.
func renderShellOutputs(_ outputs: [ShellCommandOutput]) -> String {
    if outputs.isEmpty { return "(no output)" }
    return outputs.map { result -> String in
        var lines: [String] = []
        if let command = result.command { lines.append("$ \(command)") }
        let stdout = result.stdout.trimmingTrailingNewlines
        let stderr = result.stderr.trimmingTrailingNewlines
        if !stdout.isEmpty { lines.append(stdout) }
        if !stderr.isEmpty {
            if !stdout.isEmpty { lines.append("") }
            lines.append("stderr:")
            lines.append(stderr)
        }
        if let code = result.exitCode, code != 0 { lines.append("exit code: \(code)") }
        if result.status == "timeout" { lines.append("status: timeout") }
        let chunk = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        return chunk.isEmpty ? "(no output)" : chunk
    }.joined(separator: "\n\n")
}

/// Truncates shell output streams to a maximum combined length.
func truncateShellOutputs(_ outputs: [ShellCommandOutput], maxLength: Int) -> [ShellCommandOutput] {
    var remaining = max(0, maxLength)
    return outputs.map { output in
        var stdout = ""
        var stderr = ""
        if remaining > 0, !output.stdout.isEmpty {
            stdout = String(output.stdout.prefix(remaining))
            remaining -= stdout.count
        }
        if remaining > 0, !output.stderr.isEmpty {
            stderr = String(output.stderr.prefix(remaining))
            remaining -= stderr.count
        }
        return ShellCommandOutput(stdout: stdout, stderr: stderr, outcome: output.outcome, command: output.command, providerData: output.providerData)
    }
}

/// Normalizes raw shell output entries into the model-facing payload.
func normalizeShellOutputEntries(_ entries: [JSONObject]) -> [JSONObject] {
    entries.map { entry in
        var sanitized = entry
        let status = sanitized.removeValue(forKey: "status")?.stringValue
        sanitized.removeValue(forKey: "provider_data")
        let rawExitCode = sanitized.removeValue(forKey: "exit_code")
        sanitized.removeValue(forKey: "command")
        switch sanitized["outcome"] {
        case .string(let outcomeValue)?:
            let resolvedType = status == "timeout" ? "timeout" : "exit"
            var outcome: JSONObject = ["type": .string(resolvedType)]
            if resolvedType == "exit" { outcome["exit_code"] = .int(resolveExitCode(rawExitCode, outcomeValue)) }
            sanitized["outcome"] = .object(outcome)
        case .object(var outcome)?:
            let outcomeStatus = outcome.removeValue(forKey: "status")?.stringValue
            if outcome.string("type") != "timeout", outcome["exit_code"] == nil || outcome["exit_code"] == .null {
                outcome["exit_code"] = .int(resolveExitCode(rawExitCode, outcomeStatus))
            }
            sanitized["outcome"] = .object(outcome)
        default:
            break
        }
        return sanitized
    }
}

extension String {
    fileprivate var trimmingTrailingNewlines: String {
        var copy = self
        while copy.hasSuffix("\n") { copy.removeLast() }
        return copy
    }
}

enum ShellAction {
    static func execute(agent: Agent, call: ToolRunShellCall, hooks: RunHooks, contextWrapper: RunContextWrapper, config: RunConfig) async throws -> RunItem {
        let shellCall = try coerceShellCall(call.toolCall)
        let shellTool = call.shellTool
        let currentItem = ToolApprovalItem(agent: agent, rawItem: call.toolCall, toolName: shellTool.name)
        return try await withToolFunctionSpan(config: config, toolName: shellTool.name) { span in
            if let span, config.traceIncludeSensitiveData {
                span.data(as: FunctionSpanData.self)?.input = JSONValue.object([
                    "commands": .array(shellCall.action.commands.map(JSONValue.string)),
                    "timeout_ms": shellCall.action.timeoutMs.map(JSONValue.int) ?? .null,
                    "max_output_length": shellCall.action.maxOutputLength.map(JSONValue.int) ?? .null,
                ]).pythonJSONString()
            }
            var status = try contextWrapper.getApprovalStatus(toolName: shellTool.name, callId: shellCall.callId, existingPending: currentItem)
            var needsApproval = false
            if status == nil {
                switch shellTool.needsApproval {
                case .required(let value): needsApproval = value
                case .dynamic(let function): needsApproval = try await function(contextWrapper, shellCall.action, shellCall.callId)
                }
                status = try contextWrapper.getApprovalStatus(toolName: shellTool.name, callId: shellCall.callId, existingPending: currentItem)
            }
            if status == nil, needsApproval {
                let (resolved, approvalItem) = try await resolveApprovalStatus(
                    toolName: shellTool.name, callId: shellCall.callId, rawItem: call.toolCall,
                    agent: agent, contextWrapper: contextWrapper, onApproval: shellTool.onApproval
                )
                status = resolved
                if status == nil { return approvalItem }
            }
            if status == false {
                let message = await resolveApprovalRejectionMessage(
                    contextWrapper: contextWrapper, runConfig: config, toolType: "shell",
                    toolName: shellTool.name, callId: shellCall.callId, existingPending: currentItem
                )
                return shellRejectionItem(agent: agent, callId: shellCall.callId, toolCall: call.toolCall, rejectionMessage: message)
            }
            try await callHooks(hooks, agent, context: contextWrapper, tool: shellTool, start: true)
            var outputStatus = "completed"
            var outputText = ""
            var shellOutputPayload: [JSONObject]?
            var providerMeta: JSONObject?
            var maxOutputLength: Int?
            let requestedMax = shellCall.action.maxOutputLength.map { max(0, $0) }
            do {
                guard let executor = shellTool.executor else {
                    throw ModelBehaviorError("Shell tool has no local executor configured.")
                }
                switch try await executor(ShellCommandRequest(ctxWrapper: contextWrapper, data: shellCall)) {
                case .result(let result):
                    var normalized = result.output
                    let resultMax = result.maxOutputLength.map { max(0, $0) }
                    switch (resultMax, requestedMax) {
                    case (nil, let requested): maxOutputLength = requested
                    case (let resultValue?, nil): maxOutputLength = resultValue
                    case (let resultValue?, let requested?): maxOutputLength = min(resultValue, requested)
                    }
                    if let maxOutputLength { normalized = truncateShellOutputs(normalized, maxLength: maxOutputLength) }
                    outputText = renderShellOutputs(normalized)
                    if let maxOutputLength { outputText = String(outputText.prefix(maxOutputLength)) }
                    shellOutputPayload = normalized.map(serializeShellOutput)
                    providerMeta = result.providerData
                case .text(let text):
                    outputText = text
                    if let requestedMax {
                        maxOutputLength = requestedMax
                        outputText = String(outputText.prefix(requestedMax))
                    }
                }
            } catch {
                outputStatus = "failed"
                outputText = formatShellError(error)
                span?.setError(SpanError(message: "Error running tool", data: [
                    "tool_name": .string(shellTool.name),
                    "error": .string(getTraceToolError(traceIncludeSensitiveData: config.traceIncludeSensitiveData, errorMessage: outputText)),
                ]))
                if let requestedMax {
                    maxOutputLength = requestedMax
                    outputText = String(outputText.prefix(requestedMax))
                }
                logActionError("Shell executor failed", error, isToolAction: true)
            }
            var rawEntries: [JSONObject]?
            if let shellOutputPayload, !shellOutputPayload.isEmpty {
                rawEntries = shellOutputPayload
            } else if !outputText.isEmpty {
                rawEntries = [
                    [
                        "stdout": .string(outputText), "stderr": "", "status": .string(outputStatus),
                        "outcome": .string(outputStatus == "completed" ? "success" : "failure"),
                    ],
                ]
            }
            var raw: JSONObject = [
                "type": "shell_call_output",
                "call_id": .string(shellCall.callId),
                "output": .array((rawEntries.map(normalizeShellOutputEntries) ?? []).map(JSONValue.object)),
                "status": .string(outputStatus),
            ]
            ItemHelpers.copyToolCallCaller(call.toolCall, into: &raw)
            if let maxOutputLength { raw["max_output_length"] = .int(maxOutputLength) }
            if let rawEntries { raw["shell_output"] = .array(rawEntries.map(JSONValue.object)) }
            if let providerMeta, !providerMeta.isEmpty { raw["provider_data"] = .object(providerMeta) }
            let item = ToolCallOutputItem(agent: agent, rawItem: raw, output: outputText)
            try await callHooks(hooks, agent, context: contextWrapper, tool: shellTool, start: false, result: outputText)
            if let span, config.traceIncludeSensitiveData {
                span.data(as: FunctionSpanData.self)?.output = outputText
            }
            return item
        }
    }
}

// MARK: - Apply patch

func isApplyPatchName(_ name: String?, _ tool: ApplyPatchTool?) -> Bool {
    guard let name, !name.isEmpty else { return false }
    let candidate = name.trimmingCharacters(in: .whitespaces).lowercased()
    if candidate.hasPrefix("apply_patch") { return true }
    if let tool, candidate == tool.name.trimmingCharacters(in: .whitespaces).lowercased() { return true }
    return false
}

private func parseApplyPatchJSON(_ payload: String, label: String) throws -> JSONObject {
    let parsed: JSONValue
    do {
        parsed = try JSONValue.parse(payload.isEmpty ? "{}" : payload)
    } catch {
        throw ModelBehaviorError("Invalid apply_patch \(label) JSON: \(error)")
    }
    guard case .object(let object) = parsed else {
        throw ModelBehaviorError("Apply patch \(label) must be a JSON object.")
    }
    return object
}

/// Normalizes supported custom/function apply_patch fallbacks into one pseudo-call.
func normalizeApplyPatchFallbackCall(_ toolCall: JSONObject) throws -> JSONObject? {
    var pseudo: JSONObject
    switch toolCall.type {
    case "custom_tool_call":
        var parsed = try parseApplyPatchJSON(toolCall.string("input") ?? "", label: "input")
        if !parsed.contains("operation"), !parsed.contains("operations") {
            parsed = ["operation": .object(parsed)]
        }
        pseudo = ["type": "apply_patch_call", "call_id": toolCall["call_id"] ?? .null]
        pseudo.merge(parsed)
    case "function_call":
        let parsed = try parseApplyPatchJSON(toolCall.string("arguments") ?? "", label: "arguments")
        pseudo = ["type": "apply_patch_call", "call_id": toolCall["call_id"] ?? .null, "operation": .object(parsed)]
    default:
        return nil
    }
    ItemHelpers.copyToolCallCaller(toolCall, into: &pseudo)
    return pseudo
}

func coerceApplyPatchOperations(_ toolCall: JSONObject, contextWrapper: RunContextWrapper) throws -> [ApplyPatchOperation] {
    if case .array(let rawOperations)? = toolCall["operations"] {
        let operations = try rawOperations.map { try coerceApplyPatchOperationPayload($0, contextWrapper: contextWrapper) }
        if operations.isEmpty { throw ModelBehaviorError("Apply patch call includes no operations.") }
        return operations
    }
    if let raw = toolCall["operation"], !raw.isNull {
        return [try coerceApplyPatchOperationPayload(raw, contextWrapper: contextWrapper)]
    }
    throw ModelBehaviorError("Apply patch call is missing an operation payload.")
}

private func coerceApplyPatchOperationPayload(_ raw: JSONValue, contextWrapper: RunContextWrapper) throws -> ApplyPatchOperation {
    guard case .object(let operation) = raw else {
        throw ModelBehaviorError("Apply patch call is missing an operation payload.")
    }
    let type = operation.string("type") ?? stringify(operation["type"] ?? .null)
    guard ["create_file", "update_file", "delete_file"].contains(type) else {
        throw ModelBehaviorError("Unknown apply_patch operation: \(type)")
    }
    guard let path = operation.string("path"), !path.isEmpty else {
        throw ModelBehaviorError("Apply patch operation is missing a valid path.")
    }
    var diff: String?
    if type == "create_file" || type == "update_file" {
        guard let value = operation.string("diff"), !value.isEmpty else {
            throw ModelBehaviorError("Apply patch operation \(type) is missing the required diff payload.")
        }
        diff = value
    }
    var moveTo: String?
    if let value = operation["move_to"], !value.isNull {
        guard let string = value.stringValue, !string.isEmpty else {
            throw ModelBehaviorError("Apply patch operation move_to must be a non-empty path.")
        }
        moveTo = string
    }
    return ApplyPatchOperation(type: type, path: path, diff: diff, ctxWrapper: contextWrapper, moveTo: moveTo)
}

enum ApplyPatchAction {
    static func execute(agent: Agent, call: ToolRunApplyPatchCall, hooks: RunHooks, contextWrapper: RunContextWrapper, config: RunConfig) async throws -> RunItem {
        let tool = call.applyPatchTool
        let operations = try coerceApplyPatchOperations(call.toolCall, contextWrapper: contextWrapper)
        guard let callId = call.toolCall.string("call_id"), !callId.isEmpty else {
            throw ModelBehaviorError("Apply patch call is missing call_id.")
        }
        let currentItem = ToolApprovalItem(agent: agent, rawItem: call.toolCall, toolName: tool.name)
        return try await withToolFunctionSpan(config: config, toolName: tool.name) { span in
            if let span, config.traceIncludeSensitiveData {
                span.data(as: FunctionSpanData.self)?.input = JSONValue.array(operations.map {
                    ["type": .string($0.type), "path": .string($0.path), "diff": $0.diff.map(JSONValue.string) ?? .null]
                }).pythonJSONString()
            }
            var status = try contextWrapper.getApprovalStatus(toolName: tool.name, callId: callId, existingPending: currentItem)
            var needsApproval = false
            if status == nil {
                for operation in operations {
                    switch tool.needsApproval {
                    case .required(let value): needsApproval = value
                    case .dynamic(let function): needsApproval = try await function(contextWrapper, operation, callId)
                    }
                    status = try contextWrapper.getApprovalStatus(toolName: tool.name, callId: callId, existingPending: currentItem)
                    if status != nil || needsApproval { break }
                }
            }
            if status == nil, needsApproval {
                let (resolved, approvalItem) = try await resolveApprovalStatus(
                    toolName: tool.name, callId: callId, rawItem: call.toolCall, agent: agent,
                    contextWrapper: contextWrapper, onApproval: tool.onApproval
                )
                status = resolved
                if status == nil { return approvalItem }
            }
            if status == false {
                let message = await resolveApprovalRejectionMessage(
                    contextWrapper: contextWrapper, runConfig: config, toolType: "apply_patch",
                    toolName: tool.name, callId: callId, existingPending: currentItem
                )
                return applyPatchRejectionItem(agent: agent, callId: callId, toolCall: call.toolCall, rejectionMessage: message)
            }
            try await callHooks(hooks, agent, context: contextWrapper, tool: tool, start: true)
            var outputStatus = "completed"
            var outputText = ""
            do {
                var outputs: [String] = []
                for operation in operations {
                    let result: ApplyPatchResult?
                    switch operation.type {
                    case "create_file": result = try await tool.editor.createFile(operation)
                    case "update_file": result = try await tool.editor.updateFile(operation)
                    default: result = try await tool.editor.deleteFile(operation)
                    }
                    if let result {
                        if result.status == "failed" { outputStatus = "failed" }
                        if let output = result.output, !output.isEmpty { outputs.append(output) }
                    }
                }
                outputText = outputs.joined(separator: "\n")
            } catch {
                outputStatus = "failed"
                outputText = formatShellError(error)
                span?.setError(SpanError(message: "Error running tool", data: [
                    "tool_name": .string(tool.name),
                    "error": .string(getTraceToolError(traceIncludeSensitiveData: config.traceIncludeSensitiveData, errorMessage: outputText)),
                ]))
                logActionError("Apply patch editor failed", error, isToolAction: true)
            }
            var raw: JSONObject = [
                "type": "apply_patch_call_output",
                "call_id": .string(callId),
                "status": .string(outputStatus),
            ]
            if !outputText.isEmpty { raw["output"] = .string(outputText) }
            ItemHelpers.copyToolCallCaller(call.toolCall, into: &raw)
            let item = ToolCallOutputItem(agent: agent, rawItem: raw, output: outputText)
            try await callHooks(hooks, agent, context: contextWrapper, tool: tool, start: false, result: outputText)
            if let span, config.traceIncludeSensitiveData {
                span.data(as: FunctionSpanData.self)?.output = outputText
            }
            return item
        }
    }
}

// MARK: - Custom tools

enum CustomToolAction {
    static func execute(agent: Agent, call: ToolRunCustom, hooks: RunHooks, contextWrapper: RunContextWrapper, config: RunConfig) async throws -> RunItem {
        let tool = call.customTool
        guard let callId = call.toolCall.json.string("call_id"), !callId.isEmpty else {
            throw ModelBehaviorError("Custom tool call is missing a non-empty call_id.")
        }
        guard let input = call.toolCall.json.string("input") else {
            throw ModelBehaviorError("Custom tool call is missing input.")
        }
        let toolContext = try ToolContext.fromAgentContext(
            contextWrapper, toolCallId: callId, agent: agent, toolName: tool.name, toolArguments: input, runConfig: config
        )
        let currentItem = ToolApprovalItem(agent: agent, rawItem: call.toolCall.json, toolName: tool.name)
        func outputItem(_ text: String) -> ToolCallOutputItem {
            var raw: JSONObject = ["type": "custom_tool_call_output", "call_id": .string(callId), "output": .string(text)]
            ItemHelpers.copyToolCallCaller(call.toolCall.json, into: &raw)
            return ToolCallOutputItem(agent: agent, rawItem: raw, output: text)
        }
        return try await withToolFunctionSpan(config: config, toolName: tool.name) { span in
            if let span, config.traceIncludeSensitiveData {
                span.data(as: FunctionSpanData.self)?.input = input
            }
            var status = try contextWrapper.getApprovalStatus(toolName: tool.name, callId: callId, existingPending: currentItem)
            var needsApproval = false
            if status == nil {
                switch tool.needsApproval {
                case .required(let value): needsApproval = value
                case .dynamic(let function): needsApproval = try await function(contextWrapper, input, callId)
                }
                status = try contextWrapper.getApprovalStatus(toolName: tool.name, callId: callId, existingPending: currentItem)
            }
            if status == nil, needsApproval {
                let (resolved, approvalItem) = try await resolveApprovalStatus(
                    toolName: tool.name, callId: callId, rawItem: call.toolCall.json, agent: agent,
                    contextWrapper: contextWrapper, onApproval: tool.onApproval
                )
                status = resolved
                if status == nil { return approvalItem }
            }
            if status == false {
                let message = await resolveApprovalRejectionMessage(
                    contextWrapper: contextWrapper, runConfig: config, toolType: "custom",
                    toolName: tool.name, callId: callId, existingPending: currentItem
                )
                return outputItem(message)
            }
            try await callHooks(hooks, agent, context: toolContext, tool: tool, start: true)
            var outputText: String
            do {
                outputText = stringify(try await tool.onInvokeTool(toolContext, input))
            } catch {
                outputText = formatShellError(error)
                span?.setError(SpanError(message: "Error running tool", data: [
                    "tool_name": .string(tool.name),
                    "error": .string(getTraceToolError(traceIncludeSensitiveData: config.traceIncludeSensitiveData, errorMessage: outputText)),
                ]))
                logActionError("Custom tool failed", error, isToolAction: true)
            }
            let item = outputItem(outputText)
            try await callHooks(hooks, agent, context: toolContext, tool: tool, start: false, result: outputText)
            if let span, config.traceIncludeSensitiveData {
                span.data(as: FunctionSpanData.self)?.output = outputText
            }
            return item
        }
    }
}
