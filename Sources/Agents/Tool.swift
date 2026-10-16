import Foundation

/// A tool that can be used in an agent.
///
/// Swift: upstream `Tool` is a union of concrete classes checked with `isinstance`. Here it is a
/// protocol; check concrete types with `tool as? FunctionTool`.
public protocol Tool: Sendable {
    /// The tool name used for hooks, tracing, and persisted state.
    var name: String { get }
}

/// Who may call a tool (`"direct"` or `"programmatic"`).
public enum ToolCaller: String, Sendable, Codable, CaseIterable {
    case direct
    case programmatic
}

/// Message returned to the model when a tool call is rejected without a custom message.
public let defaultApprovalRejectionMessage = "Tool execution was not approved."

/// How to handle tool timeouts.
public enum ToolTimeoutBehavior: String, Sendable {
    /// Return a model-visible timeout error string.
    case errorAsResult = "error_as_result"
    /// Raise a `ToolTimeoutError` and fail the run.
    case raiseException = "raise_exception"
}

/// Formats an error raised by a tool into a model-visible message.
public typealias ToolErrorFunction = @Sendable (RunContextWrapper, any Error) async throws -> String

/// Configures how function tool failures are reported.
public enum FailureErrorFunction: Sendable {
    /// Use `defaultToolErrorFunction`.
    case `default`
    /// Do not handle failures; errors propagate and fail the run.
    case none
    /// Use a custom formatter.
    case custom(ToolErrorFunction)
}

/// Whether a tool or handoff is enabled. Either static or computed per run.
public enum ToolEnablement: Sendable {
    case enabled(Bool)
    case dynamic(@Sendable (RunContextWrapper, AgentBase) async throws -> Bool)

    public static let always = ToolEnablement.enabled(true)

    func resolve(_ context: RunContextWrapper, _ agent: AgentBase) async throws -> Bool {
        switch self {
        case .enabled(let value): return value
        case .dynamic(let function): return try await function(context, agent)
        }
    }
}

extension ToolEnablement: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) { self = .enabled(value) }
}

/// Whether a function tool call needs approval before execution.
public enum ToolApprovalRequirement: Sendable {
    case required(Bool)
    /// Decides per call from `(run_context, tool_parameters, call_id)`.
    case dynamic(@Sendable (RunContextWrapper, JSONObject, String) async throws -> Bool)

    public static let never = ToolApprovalRequirement.required(false)
    public static let always = ToolApprovalRequirement.required(true)

    var isStaticallyFalse: Bool {
        if case .required(false) = self { return true }
        return false
    }
}

extension ToolApprovalRequirement: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) { self = .required(value) }
}

// MARK: - Structured tool outputs

/// A tool output that should be sent to the model as text.
public struct ToolOutputText: Sendable, Equatable {
    public var text: String
    public init(text: String) { self.text = text }
}

/// A tool output that should be sent to the model as an image.
public struct ToolOutputImage: Sendable, Equatable {
    public var imageURL: String?
    public var fileId: String?
    /// `"low"`, `"high"`, or `"auto"`.
    public var detail: String?

    public init(imageURL: String? = nil, fileId: String? = nil, detail: String? = nil) throws {
        guard imageURL != nil || fileId != nil else {
            throw ValueError("At least one of image_url or file_id must be provided")
        }
        self.imageURL = imageURL
        self.fileId = fileId
        self.detail = detail
    }
}

/// A tool output that should be sent to the model as a file.
public struct ToolOutputFileContent: Sendable, Equatable {
    public var fileData: String?
    public var fileURL: String?
    public var fileId: String?
    public var filename: String?

    public init(fileData: String? = nil, fileURL: String? = nil, fileId: String? = nil, filename: String? = nil) throws {
        guard fileData != nil || fileURL != nil || fileId != nil else {
            throw ValueError("At least one of file_data, file_url, or file_id must be provided")
        }
        self.fileData = fileData
        self.fileURL = fileURL
        self.fileId = fileId
        self.filename = filename
    }
}

/// The runtime source of a function-tool-backed run item.
public enum ToolOriginType: String, Sendable, Codable {
    case function
    case mcp
    case agentAsTool = "agent_as_tool"
}

/// Serializable metadata describing where a function-tool-backed item came from.
public struct ToolOrigin: Sendable, Equatable, Hashable {
    public var type: ToolOriginType
    public var mcpServerName: String?
    public var agentName: String?
    public var agentToolName: String?

    public init(type: ToolOriginType, mcpServerName: String? = nil, agentName: String? = nil, agentToolName: String? = nil) {
        self.type = type
        self.mcpServerName = mcpServerName
        self.agentName = agentName
        self.agentToolName = agentToolName
    }

    public func toJSONDict() -> JSONObject {
        var result: JSONObject = ["type": .string(type.rawValue)]
        if let mcpServerName { result["mcp_server_name"] = .string(mcpServerName) }
        if let agentName { result["agent_name"] = .string(agentName) }
        if let agentToolName { result["agent_tool_name"] = .string(agentToolName) }
        return result
    }

    public static func fromJSONDict(_ data: JSONValue?) -> ToolOrigin? {
        guard case .object(let object)? = data, let rawType = object.string("type"),
              let type = ToolOriginType(rawValue: rawType) else { return nil }
        return ToolOrigin(
            type: type,
            mcpServerName: object.string("mcp_server_name"),
            agentName: object.string("agent_name"),
            agentToolName: object.string("agent_tool_name")
        )
    }
}

/// The result of running a function tool.
public struct FunctionToolResult: Sendable {
    /// The tool that was run.
    public var tool: FunctionTool
    /// The output of the tool.
    public var output: any Sendable
    /// The run item produced as a result of the tool call, or `nil` when interrupted.
    public var runItem: RunItem?
    /// Interruptions from nested agent runs (for agent-as-tool).
    public var interruptions: [ToolApprovalItem]
    /// Nested agent run result (for agent-as-tool).
    public var agentRunResult: (any Sendable)?

    public init(tool: FunctionTool, output: any Sendable, runItem: RunItem?, interruptions: [ToolApprovalItem] = [], agentRunResult: (any Sendable)? = nil) {
        self.tool = tool
        self.output = output
        self.runItem = runItem
        self.interruptions = interruptions
        self.agentRunResult = agentRunResult
    }
}

/// Context passed to function-tool custom data extractors.
public struct FunctionToolCustomDataContext: Sendable {
    public var toolContext: ToolContext
    public var tool: FunctionTool
    public var output: any Sendable
    public var rawItem: JSONObject
}

/// Returns SDK-only custom data to attach to a tool output item.
public typealias FunctionToolCustomDataExtractor = @Sendable (FunctionToolCustomDataContext) async throws -> JSONObject?

// MARK: - FunctionTool

/// A tool that wraps a function. In most cases, use `FunctionTool.create` (the `function_tool`
/// equivalent) to build one from a `Decodable` argument type.
public final class FunctionTool: Tool, @unchecked Sendable {
    /// The name of the tool, as shown to the LLM.
    public var name: String
    /// A description of the tool, as shown to the LLM.
    public var description: String
    /// The JSON schema for the tool's parameters.
    public var paramsJSONSchema: JSONObject
    /// Invokes the tool with the tool context and the arguments from the LLM as a JSON string.
    ///
    /// Return a `String`, a structured tool output (`ToolOutputText`, `ToolOutputImage`,
    /// `ToolOutputFileContent`), an array of them, or any value with a string description.
    public var onInvokeTool: @Sendable (ToolContext, String) async throws -> any Sendable {
        didSet { failureHandlingInvoker = nil }
    }
    /// Whether the JSON schema is in strict mode.
    public var strictJSONSchema: Bool
    /// Whether the tool is enabled.
    public var isEnabled: ToolEnablement
    /// Optional input guardrails to run before invoking this tool.
    public var toolInputGuardrails: [ToolInputGuardrail]?
    /// Optional output guardrails to run after invoking this tool.
    public var toolOutputGuardrails: [ToolOutputGuardrail]?
    /// Whether the tool needs approval before execution.
    public var needsApproval: ToolApprovalRequirement
    /// Optional timeout (seconds) for each tool invocation.
    public var timeoutSeconds: Double?
    /// How to handle timeout events.
    public var timeoutBehavior: ToolTimeoutBehavior
    /// Optional formatter for timeout errors when `timeoutBehavior` is `.errorAsResult`.
    public var timeoutErrorFunction: ToolErrorFunction?
    /// Whether the Responses API should hide this tool definition until tool search loads it.
    public var deferLoading: Bool
    /// Optional callback that attaches SDK-only custom data to the tool output item.
    public var customDataExtractor: FunctionToolCustomDataExtractor?
    /// Callers that may invoke this tool on OpenAI Responses models.
    public var allowedCallers: [ToolCaller]?
    /// Optional JSON Schema describing this tool's output for programmatic callers.
    public var outputJSONSchema: JSONObject?

    var failureErrorFunction: ToolErrorFunction?
    var useDefaultFailureErrorFunction = true
    var isAgentTool = false
    var agentToolDefaultIdentity: (agentName: String, toolName: String)?
    weak var agentInstance: Agent?
    var toolNamespace: String?
    var toolNamespaceDescription: String?
    /// Whether the invoker signals FIFO dispatch itself, right before the user body runs.
    var signalsDispatchBeforeBody = false
    var mcpTitle: String?
    var toolOrigin: ToolOrigin?
    var emitToolOrigin = true
    var mcpToolBinding: (serverName: String, toolName: String, serverIndex: Int?)?
    /// The SDK failure-handling invoker behind `onInvokeTool`, rebound to each copy so copies
    /// resolve failure policy against themselves. Cleared when `onInvokeTool` is replaced.
    var failureHandlingInvoker: FailureHandlingInvoker?

    public init(
        name: String,
        description: String,
        paramsJSONSchema: JSONObject,
        onInvokeTool: @escaping @Sendable (ToolContext, String) async throws -> any Sendable,
        strictJSONSchema: Bool = true,
        isEnabled: ToolEnablement = .always,
        toolInputGuardrails: [ToolInputGuardrail]? = nil,
        toolOutputGuardrails: [ToolOutputGuardrail]? = nil,
        needsApproval: ToolApprovalRequirement = .never,
        timeoutSeconds: Double? = nil,
        timeoutBehavior: ToolTimeoutBehavior = .errorAsResult,
        timeoutErrorFunction: ToolErrorFunction? = nil,
        deferLoading: Bool = false,
        customDataExtractor: FunctionToolCustomDataExtractor? = nil,
        allowedCallers: [ToolCaller]? = nil,
        outputJSONSchema: JSONObject? = nil
    ) throws {
        self.name = name
        self.description = description
        self.paramsJSONSchema = strictJSONSchema ? try ensureStrictJSONSchema(paramsJSONSchema) : paramsJSONSchema
        self.onInvokeTool = onInvokeTool
        self.strictJSONSchema = strictJSONSchema
        self.isEnabled = isEnabled
        self.toolInputGuardrails = toolInputGuardrails
        self.toolOutputGuardrails = toolOutputGuardrails
        self.needsApproval = needsApproval
        self.timeoutSeconds = timeoutSeconds
        self.timeoutBehavior = timeoutBehavior
        self.timeoutErrorFunction = timeoutErrorFunction
        self.deferLoading = deferLoading
        self.customDataExtractor = customDataExtractor
        self.allowedCallers = try normalizeToolAllowedCallers(allowedCallers, toolName: name)
        if let outputJSONSchema {
            self.outputJSONSchema = try ensureStrictJSONSchema(outputJSONSchema)
        }
        try validateTimeoutConfig()
    }

    /// The public qualified name used to identify this function tool.
    public var qualifiedName: String {
        toolQualifiedName(name, namespace: toolNamespace) ?? name
    }

    /// Returns a shallow copy of the tool, mirroring `copy.copy(tool)`.
    public func copy() -> FunctionTool {
        // swiftlint:disable:next force_try
        let copied = try! FunctionTool(
            name: name,
            description: description,
            paramsJSONSchema: paramsJSONSchema,
            onInvokeTool: onInvokeTool,
            strictJSONSchema: false,
            isEnabled: isEnabled,
            toolInputGuardrails: toolInputGuardrails,
            toolOutputGuardrails: toolOutputGuardrails,
            needsApproval: needsApproval,
            timeoutSeconds: timeoutSeconds,
            timeoutBehavior: timeoutBehavior,
            timeoutErrorFunction: timeoutErrorFunction,
            deferLoading: deferLoading,
            customDataExtractor: customDataExtractor,
            allowedCallers: nil
        )
        copied.strictJSONSchema = strictJSONSchema
        copied.allowedCallers = allowedCallers
        copied.outputJSONSchema = outputJSONSchema
        copied.failureErrorFunction = failureErrorFunction
        copied.useDefaultFailureErrorFunction = useDefaultFailureErrorFunction
        copied.isAgentTool = isAgentTool
        copied.agentToolDefaultIdentity = agentToolDefaultIdentity
        copied.agentInstance = agentInstance
        copied.toolNamespace = toolNamespace
        copied.toolNamespaceDescription = toolNamespaceDescription
        copied.signalsDispatchBeforeBody = signalsDispatchBeforeBody
        copied.mcpTitle = mcpTitle
        copied.toolOrigin = toolOrigin
        copied.emitToolOrigin = emitToolOrigin
        copied.mcpToolBinding = mcpToolBinding
        if let failureHandlingInvoker {
            copied.onInvokeTool = failureHandlingInvoker.bound(to: copied)
            copied.failureHandlingInvoker = failureHandlingInvoker
        }
        return copied
    }

    private func validateTimeoutConfig() throws {
        if let timeoutSeconds {
            guard timeoutSeconds.isFinite else {
                throw ValueError("FunctionTool timeout_seconds must be a finite number.")
            }
            guard timeoutSeconds > 0 else {
                throw ValueError("FunctionTool timeout_seconds must be greater than 0.")
            }
        }
    }

    /// Sets the failure formatter used for runtime-generated error handling.
    func setFailureErrorFunction(_ option: FailureErrorFunction) {
        switch option {
        case .default:
            useDefaultFailureErrorFunction = true
            failureErrorFunction = nil
        case .none:
            useDefaultFailureErrorFunction = false
            failureErrorFunction = nil
        case .custom(let function):
            useDefaultFailureErrorFunction = false
            failureErrorFunction = function
        }
    }
}

/// Returns scalar origin metadata for a function tool.
public func getFunctionToolOrigin(_ tool: FunctionTool) -> ToolOrigin? {
    guard tool.emitToolOrigin else { return nil }
    return tool.toolOrigin ?? ToolOrigin(type: .function)
}

func normalizeToolAllowedCallers(_ callers: [ToolCaller]?, toolName: String) throws -> [ToolCaller]? {
    guard let callers else { return nil }
    if callers.isEmpty {
        throw UserError("Tool `\(toolName)` allowed_callers must include at least one caller.")
    }
    var normalized: [ToolCaller] = []
    for caller in callers where !normalized.contains(caller) {
        normalized.append(caller)
    }
    return normalized
}

// MARK: - Failure handling

/// Returns a fixed error response without exposing the exception to the model.
///
/// Provide a custom `failureErrorFunction` to return application-approved error details.
public let defaultToolErrorFunction: ToolErrorFunction = { _, _ in
    "An error occurred while running the tool. Please try again."
}

/// Builds the default message returned to the model when a tool times out.
public func defaultToolTimeoutErrorMessage(toolName: String, timeoutSeconds: Double) -> String {
    "Tool '\(toolName)' timed out after \(formatSeconds(timeoutSeconds)) seconds."
}

func isProgrammaticToolCall(_ toolCall: JSONObject?) -> Bool {
    guard case .object(let caller)? = toolCall?["caller"] else { return false }
    return caller.type == "program"
}

/// Returns the configured tool failure formatter for runtime-generated error handling.
func resolveFunctionToolFailureErrorFunction(_ tool: FunctionTool, context: RunContextWrapper?) -> ToolErrorFunction? {
    if tool.useDefaultFailureErrorFunction {
        if tool.outputJSONSchema != nil, let toolContext = context as? ToolContext, isProgrammaticToolCall(toolContext.toolCall) {
            return nil
        }
        return defaultToolErrorFunction
    }
    return tool.failureErrorFunction
}

/// Invokes the configured failure formatter, if one exists.
func maybeInvokeFunctionToolFailureErrorFunction(
    tool: FunctionTool,
    context: RunContextWrapper,
    error: any Error
) async throws -> String? {
    guard let function = resolveFunctionToolFailureErrorFunction(tool, context: context) else { return nil }
    let result = try await function(context, coerceToolErrorForFailureErrorFunction(error))
    if tool.useDefaultFailureErrorFunction, let toolContext = context as? ToolContext {
        toolContext.defaultFailureHandled = true
    }
    return result
}

/// Adapts a cancellation into a described error for tool failure formatters.
struct FunctionToolCancelledError: Error, CustomStringConvertible {
    let cancelledError: CancellationError
    var description: String { "Tool execution cancelled." }
}

/// Converts runtime failures into the error contract expected by tool failure formatters.
func coerceToolErrorForFailureErrorFunction(_ error: any Error) -> any Error {
    if let cancelled = error as? CancellationError {
        return FunctionToolCancelledError(cancelledError: cancelled)
    }
    return error
}

/// A function tool result with metadata for SDK-generated error outputs.
struct FunctionToolInvocationResult: Sendable {
    var output: any Sendable
    var isSDKGeneratedError = false
}

/// Invokes a function tool, enforcing timeout configuration when provided.
public func invokeFunctionTool(tool: FunctionTool, context: ToolContext, arguments: String) async throws -> any Sendable {
    try await invokeFunctionToolWithMetadata(tool: tool, context: context, arguments: arguments).output
}

func invokeFunctionToolWithMetadata(tool: FunctionTool, context: ToolContext, arguments: String) async throws -> FunctionToolInvocationResult {
    if !tool.signalsDispatchBeforeBody {
        ToolDispatchSignal.current?.signal()
    }
    context.defaultFailureHandled = false
    func consume() -> Bool {
        defer { context.defaultFailureHandled = false }
        return context.defaultFailureHandled
    }
    guard let timeoutSeconds = tool.timeoutSeconds else {
        let output = try await tool.onInvokeTool(context, arguments)
        return FunctionToolInvocationResult(output: output, isSDKGeneratedError: consume())
    }

    let outcome = try await withTimeout(seconds: timeoutSeconds) {
        try await tool.onInvokeTool(context, arguments)
    }
    switch outcome {
    case .completed(let output):
        return FunctionToolInvocationResult(output: output, isSDKGeneratedError: consume())
    case .timedOut:
        let timeoutError = ToolTimeoutError(toolName: tool.name, timeoutSeconds: timeoutSeconds)
        if tool.timeoutBehavior == .raiseException {
            throw timeoutError
        }
        guard let timeoutErrorFunction = tool.timeoutErrorFunction else {
            return FunctionToolInvocationResult(
                output: defaultToolTimeoutErrorMessage(toolName: tool.name, timeoutSeconds: timeoutSeconds),
                isSDKGeneratedError: true
            )
        }
        return FunctionToolInvocationResult(output: try await timeoutErrorFunction(context, timeoutError))
    }
}

/// The outcome of an operation raced against a timeout.
enum TimeoutOutcome<Value: Sendable>: Sendable {
    case completed(Value)
    case timedOut
}

/// Runs `operation`, cancelling it when it does not finish within `seconds`.
func withTimeout<Value: Sendable>(
    seconds: Double,
    _ operation: @escaping @Sendable () async throws -> Value
) async throws -> TimeoutOutcome<Value> {
    try await withThrowingTaskGroup(of: TimeoutOutcome<Value>.self) { group in
        group.addTask { .completed(try await operation()) }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(max(seconds, 0) * 1_000_000_000))
            return .timedOut
        }
        defer { group.cancelAll() }
        guard let first = try await group.next() else { return .timedOut }
        return first
    }
}

/// Decodes raw tool arguments with consistent diagnostics.
func parseFunctionToolJSONInput(toolName: String, inputJSON: String) throws -> JSONObject {
    let parsed: JSONValue
    do {
        parsed = inputJSON.isEmpty ? .object([:]) : try JSONValue.parse(inputJSON)
    } catch {
        let baseMessage = "Invalid JSON input for tool \(toolName)"
        if AgentsDebug.dontLogToolData {
            logger.debug(baseMessage)
            throw ModelBehaviorError(baseMessage)
        }
        let detailed = "\(baseMessage): \(inputJSON)"
        logger.debug(detailed)
        throw ModelBehaviorError(detailed)
    }
    guard case .object(let object) = parsed else {
        throw ModelBehaviorError("Invalid JSON input for tool \(toolName): expected a JSON object")
    }
    return object
}

func logFunctionToolInvocation(toolName: String, inputJSON: String) {
    if AgentsDebug.dontLogToolData {
        logger.debug("Invoking tool \(toolName)")
    } else {
        logger.debug("Invoking tool \(toolName) with input \(inputJSON)")
    }
}

/// Reports a handled function-tool failure to tracing and logs.
func reportHandledFunctionToolError(
    tool: FunctionTool,
    error: any Error,
    inputJSON: String,
    context: ToolContext,
    spanMessage: String = "Error running tool (non-fatal)",
    spanMessageForJSONDecodeError: String? = "Error running tool",
    logLabel: String = "Tool"
) {
    let includeSensitive = (context.runConfig?.traceIncludeSensitiveData ?? true)
        && !tool.useDefaultFailureErrorFunction
    let message = describeError(error)
    let isJSONError = (error as? ModelBehaviorError)?.message.hasPrefix("Invalid JSON input for tool") == true
    let resolvedSpanMessage = (includeSensitive && isJSONError) ? (spanMessageForJSONDecodeError ?? spanMessage) : spanMessage
    attachErrorToCurrentSpan(SpanError(
        message: resolvedSpanMessage,
        data: [
            "tool_name": .string(tool.name),
            "error": .string(getTraceToolError(traceIncludeSensitiveData: includeSensitive, errorMessage: includeSensitive ? message : "")),
        ]
    ))
    if AgentsDebug.dontLogToolData {
        logger.debug("\(logLabel) \(tool.name) failed")
    } else {
        logger.error("\(logLabel) \(tool.name) failed: \(inputJSON) \(message)")
    }
}

// MARK: - Function tool factories

extension FunctionTool {
    /// Creates a function tool whose arguments decode into `Args` (the `function_tool` equivalent).
    ///
    /// Swift: upstream derives the schema from the Python signature and docstring. Here the schema
    /// comes from `Args`'s `Decodable` conformance, or from `paramsJSONSchema` when given
    /// (porting/decisions.md D6).
    public static func create<Args: Decodable & Sendable>(
        name: String,
        description: String = "",
        argumentsType: Args.Type = Args.self,
        paramsJSONSchema: JSONObject? = nil,
        strictJSONSchema: Bool = true,
        failureErrorFunction: FailureErrorFunction = .default,
        isEnabled: ToolEnablement = .always,
        needsApproval: ToolApprovalRequirement = .never,
        toolInputGuardrails: [ToolInputGuardrail]? = nil,
        toolOutputGuardrails: [ToolOutputGuardrail]? = nil,
        timeoutSeconds: Double? = nil,
        timeoutBehavior: ToolTimeoutBehavior = .errorAsResult,
        timeoutErrorFunction: ToolErrorFunction? = nil,
        deferLoading: Bool = false,
        customDataExtractor: FunctionToolCustomDataExtractor? = nil,
        allowedCallers: [ToolCaller]? = nil,
        outputJSONSchema: JSONObject? = nil,
        _ body: @escaping @Sendable (ToolContext, Args) async throws -> any Sendable
    ) throws -> FunctionTool {
        let schema = try paramsJSONSchema ?? JSONSchemaGenerator.schema(for: Args.self)
        let invoke: @Sendable (ToolContext, String) async throws -> any Sendable = { context, input in
            let toolName = context.toolName.isEmpty ? name : context.toolName
            let json = try parseFunctionToolJSONInput(toolName: toolName, inputJSON: input)
            logFunctionToolInvocation(toolName: toolName, inputJSON: input)
            let arguments: Args
            do {
                arguments = try JSONValue.object(json).decode(as: Args.self)
            } catch {
                let baseMessage = "Invalid JSON input for tool \(toolName)"
                if AgentsDebug.dontLogToolData {
                    throw ModelBehaviorError(baseMessage)
                }
                throw ModelBehaviorError("\(baseMessage): \(error)")
            }
            // Release the next queued tool only once this tool's body is about to run.
            ToolDispatchSignal.current?.signal()
            let output = try await body(context, arguments)
            if AgentsDebug.dontLogToolData {
                logger.debug("Tool \(toolName) completed.")
            } else {
                logger.debug("Tool \(toolName) returned \(output)")
            }
            return output
        }
        let tool = try makeFailureHandlingTool(
            name: name,
            description: description,
            paramsJSONSchema: schema,
            invoke: invoke,
            failureErrorFunction: failureErrorFunction,
            strictJSONSchema: strictJSONSchema,
            isEnabled: isEnabled,
            needsApproval: needsApproval,
            toolInputGuardrails: toolInputGuardrails,
            toolOutputGuardrails: toolOutputGuardrails,
            timeoutSeconds: timeoutSeconds,
            timeoutBehavior: timeoutBehavior,
            timeoutErrorFunction: timeoutErrorFunction,
            deferLoading: deferLoading,
            customDataExtractor: customDataExtractor,
            allowedCallers: allowedCallers,
            outputJSONSchema: outputJSONSchema
        )
        tool.signalsDispatchBeforeBody = true
        return tool
    }

    /// Creates a function tool that takes no arguments.
    public static func create(
        name: String,
        description: String = "",
        failureErrorFunction: FailureErrorFunction = .default,
        isEnabled: ToolEnablement = .always,
        needsApproval: ToolApprovalRequirement = .never,
        toolInputGuardrails: [ToolInputGuardrail]? = nil,
        toolOutputGuardrails: [ToolOutputGuardrail]? = nil,
        timeoutSeconds: Double? = nil,
        timeoutBehavior: ToolTimeoutBehavior = .errorAsResult,
        timeoutErrorFunction: ToolErrorFunction? = nil,
        _ body: @escaping @Sendable (ToolContext) async throws -> any Sendable
    ) throws -> FunctionTool {
        try create(
            name: name,
            description: description,
            argumentsType: NoArguments.self,
            failureErrorFunction: failureErrorFunction,
            isEnabled: isEnabled,
            needsApproval: needsApproval,
            toolInputGuardrails: toolInputGuardrails,
            toolOutputGuardrails: toolOutputGuardrails,
            timeoutSeconds: timeoutSeconds,
            timeoutBehavior: timeoutBehavior,
            timeoutErrorFunction: timeoutErrorFunction
        ) { context, _ in try await body(context) }
    }

    /// Creates a `FunctionTool` whose invoker applies the failure formatter, like upstream
    /// `_build_wrapped_function_tool`.
    static func makeFailureHandlingTool(
        name: String,
        description: String,
        paramsJSONSchema: JSONObject,
        invoke: @escaping @Sendable (ToolContext, String) async throws -> any Sendable,
        failureErrorFunction: FailureErrorFunction = .default,
        strictJSONSchema: Bool = true,
        isEnabled: ToolEnablement = .always,
        needsApproval: ToolApprovalRequirement = .never,
        toolInputGuardrails: [ToolInputGuardrail]? = nil,
        toolOutputGuardrails: [ToolOutputGuardrail]? = nil,
        timeoutSeconds: Double? = nil,
        timeoutBehavior: ToolTimeoutBehavior = .errorAsResult,
        timeoutErrorFunction: ToolErrorFunction? = nil,
        deferLoading: Bool = false,
        customDataExtractor: FunctionToolCustomDataExtractor? = nil,
        allowedCallers: [ToolCaller]? = nil,
        outputJSONSchema: JSONObject? = nil,
        toolOrigin: ToolOrigin? = nil,
        mcpTitle: String? = nil,
        onHandledError: (@Sendable (FunctionTool, any Error, String, ToolContext) -> Void)? = nil
    ) throws -> FunctionTool {
        let invoker = FailureHandlingInvoker(invoke: invoke, onHandledError: onHandledError)
        let tool = try FunctionTool(
            name: name,
            description: description,
            paramsJSONSchema: paramsJSONSchema,
            onInvokeTool: invoke,
            strictJSONSchema: strictJSONSchema,
            isEnabled: isEnabled,
            toolInputGuardrails: toolInputGuardrails,
            toolOutputGuardrails: toolOutputGuardrails,
            needsApproval: needsApproval,
            timeoutSeconds: timeoutSeconds,
            timeoutBehavior: timeoutBehavior,
            timeoutErrorFunction: timeoutErrorFunction,
            deferLoading: deferLoading,
            customDataExtractor: customDataExtractor,
            allowedCallers: allowedCallers,
            outputJSONSchema: outputJSONSchema
        )
        tool.onInvokeTool = invoker.bound(to: tool)
        tool.failureHandlingInvoker = invoker
        tool.toolOrigin = toolOrigin
        tool.mcpTitle = mcpTitle
        tool.setFailureErrorFunction(failureErrorFunction)
        return tool
    }
}

/// Wraps a tool invoker with failure handling, like upstream `_FailureHandlingFunctionToolInvoker`.
struct FailureHandlingInvoker: Sendable {
    let invoke: @Sendable (ToolContext, String) async throws -> any Sendable
    let onHandledError: (@Sendable (FunctionTool, any Error, String, ToolContext) -> Void)?

    /// Returns an invoker that resolves failure policy against `tool`.
    func bound(to tool: FunctionTool) -> @Sendable (ToolContext, String) async throws -> any Sendable {
        let box = WeakToolBox(tool)
        return { [invoke, onHandledError] context, input in
            do {
                return try await invoke(context, input)
            } catch {
                guard let tool = box.tool else { throw error }
                if error is CancellationError, Task.isCancelled { throw error }
                guard let result = try await maybeInvokeFunctionToolFailureErrorFunction(
                    tool: tool, context: context, error: error
                ) else {
                    throw error
                }
                if let onHandledError {
                    onHandledError(tool, error, input, context)
                } else {
                    reportHandledFunctionToolError(tool: tool, error: error, inputJSON: input, context: context)
                }
                return result
            }
        }
    }
}

/// Holds the owning tool weakly so its invoker does not retain it.
private final class WeakToolBox: @unchecked Sendable {
    weak var tool: FunctionTool?

    init(_ tool: FunctionTool) {
        self.tool = tool
    }
}

/// Arguments type for tools that take no parameters.
public struct NoArguments: Codable, Sendable, Equatable {
    public init() {}
}

/// Attaches namespace metadata to function tools for OpenAI Responses tool search.
public func toolNamespace(name: String, description: String?, tools: [FunctionTool]) throws -> [FunctionTool] {
    let namespaceName = name.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !namespaceName.isEmpty else {
        throw UserError("tool_namespace() requires a non-empty namespace name.")
    }
    guard let description, !description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw UserError("tool_namespace() requires a non-empty description.")
    }
    return try tools.map { tool in
        try validateFunctionToolNamespaceShape(tool.name, namespaceName)
        let copied = tool.copy()
        copied.toolNamespace = namespaceName
        copied.toolNamespaceDescription = description.trimmingCharacters(in: .whitespacesAndNewlines)
        return copied
    }
}

/// Returns Responses-only features used by a function tool.
func getFunctionToolResponsesOnlyFeatures(_ tool: FunctionTool) -> [String] {
    var features: [String] = []
    if tool.toolNamespace != nil { features.append("tool_namespace()") }
    if tool.deferLoading { features.append("defer_loading=True") }
    if tool.allowedCallers != nil { features.append("allowed_callers") }
    if tool.outputJSONSchema != nil { features.append("output_json_schema") }
    return features
}

/// Rejects Responses-only function-tool features on unsupported backends.
func ensureFunctionToolSupportsResponsesOnlyFeatures(_ tool: FunctionTool, backendName: String) throws {
    let unsupported = getFunctionToolResponsesOnlyFeatures(tool)
    guard !unsupported.isEmpty else { return }
    throw UserError(
        "The following function-tool features are only supported with OpenAI Responses models: "
            + "\(unsupported.joined(separator: ", ")). Tool `\(tool.qualifiedName)` cannot be used with \(backendName)."
    )
}

// MARK: - Hosted tools

/// A hosted tool that lets the LLM search through a vector store.
public struct FileSearchTool: Tool {
    public var vectorStoreIds: [String]
    public var maxNumResults: Int?
    public var includeSearchResults: Bool
    public var rankingOptions: JSONObject?
    public var filters: JSONObject?

    public init(vectorStoreIds: [String], maxNumResults: Int? = nil, includeSearchResults: Bool = false, rankingOptions: JSONObject? = nil, filters: JSONObject? = nil) {
        self.vectorStoreIds = vectorStoreIds
        self.maxNumResults = maxNumResults
        self.includeSearchResults = includeSearchResults
        self.rankingOptions = rankingOptions
        self.filters = filters
    }

    public var name: String { "file_search" }
}

/// A hosted tool that lets the LLM search the web.
public struct WebSearchTool: Tool {
    public var userLocation: JSONObject?
    public var filters: JSONObject?
    /// `"low"`, `"medium"`, or `"high"`.
    public var searchContextSize: String
    public var externalWebAccess: Bool?
    public var searchContentTypes: [String]?
    public var imageSettings: JSONObject?

    public init(
        userLocation: JSONObject? = nil,
        filters: JSONObject? = nil,
        searchContextSize: String = "medium",
        externalWebAccess: Bool? = nil,
        searchContentTypes: [String]? = nil,
        imageSettings: JSONObject? = nil
    ) {
        self.userLocation = userLocation
        self.filters = filters
        self.searchContextSize = searchContextSize
        self.externalWebAccess = externalWebAccess
        self.searchContentTypes = searchContentTypes
        self.imageSettings = imageSettings
    }

    public var name: String { "web_search" }
}

/// Information about a computer tool safety check.
public struct ComputerToolSafetyCheckData: Sendable {
    public var ctxWrapper: RunContextWrapper
    public var agent: Agent
    public var toolCall: ResponseComputerToolCall
    public var safetyCheck: JSONObject
}

/// Creates or disposes per-run computers.
public struct ComputerProvider: Sendable {
    public var create: @Sendable (RunContextWrapper) async throws -> any Computer
    public var dispose: (@Sendable (RunContextWrapper, any Computer) async throws -> Void)?

    public init(
        create: @escaping @Sendable (RunContextWrapper) async throws -> any Computer,
        dispose: (@Sendable (RunContextWrapper, any Computer) async throws -> Void)? = nil
    ) {
        self.create = create
        self.dispose = dispose
    }
}

/// A local computer harness exposed through the Responses API computer tool.
public final class ComputerTool: Tool, @unchecked Sendable {
    public enum Configuration: Sendable {
        case computer(any Computer)
        case provider(ComputerProvider)
    }

    public var computer: Configuration
    public var onSafetyCheck: (@Sendable (ComputerToolSafetyCheckData) async throws -> Bool)?
    private let resolvedLock = NSLock()
    private var resolved: [ObjectIdentifier: any Computer] = [:]

    public init(computer: any Computer, onSafetyCheck: (@Sendable (ComputerToolSafetyCheckData) async throws -> Bool)? = nil) {
        self.computer = .computer(computer)
        self.onSafetyCheck = onSafetyCheck
    }

    public init(provider: ComputerProvider, onSafetyCheck: (@Sendable (ComputerToolSafetyCheckData) async throws -> Bool)? = nil) {
        self.computer = .provider(provider)
        self.onSafetyCheck = onSafetyCheck
    }

    /// Keeps the released preview-era runtime name for hooks and persisted state compatibility.
    public var name: String { "computer_use_preview" }
    /// The GA tool alias used in tracing.
    public var traceName: String { "computer" }

    /// Resolves a computer for a run context, initializing it if needed.
    public func resolveComputer(runContext: RunContextWrapper) async throws -> any Computer {
        switch computer {
        case .computer(let computer):
            return computer
        case .provider(let provider):
            let key = ObjectIdentifier(runContext.usageBox)
            if let cached = resolvedLock.locked({ resolved[key] }) { return cached }
            let created = try await provider.create(runContext)
            resolvedLock.locked { resolved[key] = created }
            return created
        }
    }

    /// Disposes any computer instances created for the provided run context.
    public func disposeResolvedComputers(runContext: RunContextWrapper) async {
        guard case .provider(let provider) = computer else { return }
        let key = ObjectIdentifier(runContext.usageBox)
        let created = resolvedLock.locked { resolved.removeValue(forKey: key) }
        guard let created, let dispose = provider.dispose else { return }
        do {
            try await dispose(runContext, created)
        } catch {
            logger.warning("Failed to dispose computer for run context: \(error)")
        }
    }
}

/// A request to approve a hosted MCP tool call.
public struct MCPToolApprovalRequest: Sendable {
    public var ctxWrapper: RunContextWrapper
    public var data: McpApprovalRequest
}

/// The result of an MCP tool approval function.
public struct MCPToolApprovalFunctionResult: Sendable, Equatable {
    public var approve: Bool
    public var reason: String?

    public init(approve: Bool, reason: String? = nil) {
        self.approve = approve
        self.reason = reason
    }
}

/// A function that approves or rejects a hosted MCP tool call.
public typealias MCPToolApprovalFunction = @Sendable (MCPToolApprovalRequest) async throws -> MCPToolApprovalFunctionResult

/// A tool that lets the LLM use a remote MCP server.
public struct HostedMCPTool: Tool {
    /// The MCP tool config (`{"type": "mcp", "server_label": ..., "server_url": ...}`).
    public var toolConfig: JSONObject
    /// Called when approval is requested for an MCP tool.
    public var onApprovalRequest: MCPToolApprovalFunction?

    public init(toolConfig: JSONObject, onApprovalRequest: MCPToolApprovalFunction? = nil) {
        self.toolConfig = toolConfig
        self.onApprovalRequest = onApprovalRequest
    }

    public var name: String { "hosted_mcp" }
}

/// A tool that lets the LLM execute code in a sandboxed environment.
public struct CodeInterpreterTool: Tool {
    public var toolConfig: JSONObject
    public init(toolConfig: JSONObject) { self.toolConfig = toolConfig }
    public var name: String { "code_interpreter" }
}

/// A tool that lets the LLM generate images.
public struct ImageGenerationTool: Tool {
    public var toolConfig: JSONObject
    public init(toolConfig: JSONObject) { self.toolConfig = toolConfig }
    public var name: String { "image_generation" }
}

/// A request to execute a command on a shell.
public struct LocalShellCommandRequest: Sendable {
    public var ctxWrapper: RunContextWrapper
    public var data: LocalShellCall
}

/// A tool that lets the LLM execute commands on a shell.
public struct LocalShellTool: Tool {
    public var executor: @Sendable (LocalShellCommandRequest) async throws -> String
    public init(executor: @escaping @Sendable (LocalShellCommandRequest) async throws -> String) {
        self.executor = executor
    }
    public var name: String { "local_shell" }
}

/// Describes the terminal condition of a shell command.
public struct ShellCallOutcome: Sendable, Equatable {
    /// `"exit"` or `"timeout"`.
    public var type: String
    public var exitCode: Int?

    public init(type: String, exitCode: Int? = nil) {
        self.type = type
        self.exitCode = exitCode
    }
}

/// Structured output for a single shell command execution.
public struct ShellCommandOutput: Sendable, Equatable {
    public var stdout: String
    public var stderr: String
    public var outcome: ShellCallOutcome
    public var command: String?
    public var providerData: JSONObject?

    public init(stdout: String = "", stderr: String = "", outcome: ShellCallOutcome = ShellCallOutcome(type: "exit"), command: String? = nil, providerData: JSONObject? = nil) {
        self.stdout = stdout
        self.stderr = stderr
        self.outcome = outcome
        self.command = command
        self.providerData = providerData
    }

    public var exitCode: Int? { outcome.exitCode }
    public var status: String { outcome.type == "timeout" ? "timeout" : "completed" }
}

/// Result returned by a shell executor.
public struct ShellResult: Sendable, Equatable {
    public var output: [ShellCommandOutput]
    public var maxOutputLength: Int?
    public var providerData: JSONObject?

    public init(output: [ShellCommandOutput], maxOutputLength: Int? = nil, providerData: JSONObject? = nil) {
        self.output = output
        self.maxOutputLength = maxOutputLength
        self.providerData = providerData
    }
}

/// Action payload for a next-generation shell call.
public struct ShellActionRequest: Sendable, Equatable {
    public var commands: [String]
    public var timeoutMs: Int?
    public var maxOutputLength: Int?

    public init(commands: [String], timeoutMs: Int? = nil, maxOutputLength: Int? = nil) {
        self.commands = commands
        self.timeoutMs = timeoutMs
        self.maxOutputLength = maxOutputLength
    }
}

/// Normalized shell call data provided to shell executors.
public struct ShellCallData: Sendable {
    public var callId: String
    public var action: ShellActionRequest
    public var status: String?
    public var raw: JSONObject?
}

/// A request to execute a modern shell call.
public struct ShellCommandRequest: Sendable {
    public var ctxWrapper: RunContextWrapper
    public var data: ShellCallData
}

/// The output of a shell executor: text or structured output.
public enum ShellExecutorOutput: Sendable {
    case text(String)
    case result(ShellResult)
}

/// The result of an `on_approval` callback for shell, apply_patch, and custom tools.
public struct ToolOnApprovalResult: Sendable, Equatable {
    public var approve: Bool
    public var reason: String?

    public init(approve: Bool, reason: String? = nil) {
        self.approve = approve
        self.reason = reason
    }
}

/// Next-generation shell tool.
public struct ShellTool: Tool {
    public var executor: (@Sendable (ShellCommandRequest) async throws -> ShellExecutorOutput)?
    public var name: String
    public var needsApproval: ShellApprovalRequirement
    public var onApproval: (@Sendable (RunContextWrapper, ToolApprovalItem) async throws -> ToolOnApprovalResult)?
    /// The execution environment (`{"type": "local"}` by default).
    public var environment: JSONObject
    public var allowedCallers: [ToolCaller]?

    public enum ShellApprovalRequirement: Sendable {
        case required(Bool)
        case dynamic(@Sendable (RunContextWrapper, ShellActionRequest, String) async throws -> Bool)
    }

    public init(
        executor: (@Sendable (ShellCommandRequest) async throws -> ShellExecutorOutput)? = nil,
        name: String = "shell",
        needsApproval: ShellApprovalRequirement = .required(false),
        onApproval: (@Sendable (RunContextWrapper, ToolApprovalItem) async throws -> ToolOnApprovalResult)? = nil,
        environment: JSONObject? = nil,
        allowedCallers: [ToolCaller]? = nil
    ) throws {
        self.executor = executor
        self.name = name
        self.needsApproval = needsApproval
        self.onApproval = onApproval
        var normalized = environment ?? ["type": "local"]
        if normalized["type"] == nil { normalized["type"] = "local" }
        self.environment = normalized
        self.allowedCallers = try normalizeToolAllowedCallers(allowedCallers, toolName: name)
        if normalized.string("type") == "local" {
            if executor == nil {
                throw UserError("ShellTool with local environment requires an executor.")
            }
            return
        }
        if executor != nil {
            throw UserError("ShellTool with hosted environment does not accept an executor.")
        }
        if case .required(false) = needsApproval, onApproval == nil {} else {
            throw UserError("ShellTool with hosted environment does not support needs_approval or on_approval.")
        }
    }

    public var type: String { "shell" }
}

/// Hosted apply_patch tool. Lets the model request file mutations via unified diffs.
public struct ApplyPatchTool: Tool {
    public var editor: any ApplyPatchEditor
    public var name: String
    public var needsApproval: ApplyPatchApprovalRequirement
    public var onApproval: (@Sendable (RunContextWrapper, ToolApprovalItem) async throws -> ToolOnApprovalResult)?
    public var allowedCallers: [ToolCaller]?

    public enum ApplyPatchApprovalRequirement: Sendable {
        case required(Bool)
        case dynamic(@Sendable (RunContextWrapper, ApplyPatchOperation, String) async throws -> Bool)
    }

    public init(
        editor: any ApplyPatchEditor,
        name: String = "apply_patch",
        needsApproval: ApplyPatchApprovalRequirement = .required(false),
        onApproval: (@Sendable (RunContextWrapper, ToolApprovalItem) async throws -> ToolOnApprovalResult)? = nil,
        allowedCallers: [ToolCaller]? = nil
    ) throws {
        self.editor = editor
        self.name = name
        self.needsApproval = needsApproval
        self.onApproval = onApproval
        self.allowedCallers = try normalizeToolAllowedCallers(allowedCallers, toolName: name)
    }

    public var type: String { "apply_patch" }
}

/// A Responses custom tool that uses one raw string input instead of JSON arguments.
public struct CustomTool: Tool {
    public var name: String
    public var description: String
    public var onInvokeTool: @Sendable (ToolContext, String) async throws -> any Sendable
    public var format: JSONValue?
    public var needsApproval: CustomApprovalRequirement
    public var onApproval: (@Sendable (RunContextWrapper, ToolApprovalItem) async throws -> ToolOnApprovalResult)?
    public var deferLoading: Bool
    public var allowedCallers: [ToolCaller]?

    public enum CustomApprovalRequirement: Sendable {
        case required(Bool)
        case dynamic(@Sendable (RunContextWrapper, String, String) async throws -> Bool)
    }

    public init(
        name: String,
        description: String,
        onInvokeTool: @escaping @Sendable (ToolContext, String) async throws -> any Sendable,
        format: JSONValue? = nil,
        needsApproval: CustomApprovalRequirement = .required(false),
        onApproval: (@Sendable (RunContextWrapper, ToolApprovalItem) async throws -> ToolOnApprovalResult)? = nil,
        deferLoading: Bool = false,
        allowedCallers: [ToolCaller]? = nil
    ) throws {
        self.name = name
        self.description = description
        self.onInvokeTool = onInvokeTool
        self.format = format
        self.needsApproval = needsApproval
        self.onApproval = onApproval
        self.deferLoading = deferLoading
        self.allowedCallers = try normalizeToolAllowedCallers(allowedCallers, toolName: name)
    }

    /// The Responses tool definition.
    public var toolConfig: JSONObject {
        var config: JSONObject = ["type": "custom", "name": .string(name), "description": .string(description)]
        if let format { config["format"] = format }
        if deferLoading { config["defer_loading"] = true }
        if let allowedCallers { config["allowed_callers"] = .array(allowedCallers.map { .string($0.rawValue) }) }
        return config
    }

    public var type: String { "custom" }
}

/// A hosted Responses API tool that lets the model search deferred tools by namespace.
public struct ToolSearchTool: Tool {
    public var description: String?
    /// `"server"` or `"client"`.
    public var execution: String?
    public var parameters: JSONValue?

    public init(description: String? = nil, execution: String? = nil, parameters: JSONValue? = nil) {
        self.description = description
        self.execution = execution
        self.parameters = parameters
    }

    public var name: String { "tool_search" }
}

/// A hosted Responses tool that lets generated JavaScript orchestrate other tools.
public struct ProgrammaticToolCallingTool: Tool {
    public init() {}
    public var name: String { "programmatic_tool_calling" }
}

// MARK: - Tool search and programmatic validation

func isResponsesToolSearchSurface(_ tool: any Tool) -> Bool {
    if let tool = tool as? FunctionTool { return tool.deferLoading || tool.toolNamespace != nil }
    if let tool = tool as? HostedMCPTool { return tool.toolConfig["defer_loading"]?.isTruthy ?? false }
    return false
}

func isRequiredToolSearchSurface(_ tool: any Tool) -> Bool {
    if let tool = tool as? FunctionTool { return tool.deferLoading }
    if let tool = tool as? HostedMCPTool { return tool.toolConfig["defer_loading"]?.isTruthy ?? false }
    return false
}

/// Validates the Responses-only tool_search and defer-loading contract.
func validateResponsesToolSearchConfiguration(_ tools: [any Tool], allowOpaqueSearchSurface: Bool = false) throws {
    let toolSearchCount = tools.filter { $0 is ToolSearchTool }.count
    let hasToolSearch = toolSearchCount > 0
    if toolSearchCount > 1 {
        throw UserError("Only one ToolSearchTool() is allowed when using OpenAI Responses models.")
    }
    try validateFunctionToolLookupConfiguration(tools)
    if tools.contains(where: isRequiredToolSearchSurface), !hasToolSearch {
        throw UserError("Deferred-loading Responses tools require ToolSearchTool() when using OpenAI Responses models.")
    }
    if hasToolSearch, !tools.contains(where: isResponsesToolSearchSurface), !allowOpaqueSearchSurface {
        throw UserError(
            "ToolSearchTool() requires at least one searchable Responses surface: a tool_namespace(...) "
                + "function tool, a deferred-loading function tool (`function_tool(..., defer_loading=True)`), "
                + "or a deferred-loading hosted MCP server (`HostedMCPTool(tool_config={..., 'defer_loading': True})`)."
        )
    }
}

private func toolAllowedCallers(_ tool: any Tool) -> [ToolCaller]? {
    switch tool {
    case let tool as FunctionTool: return tool.allowedCallers
    case let tool as CustomTool: return tool.allowedCallers
    case let tool as ShellTool: return tool.allowedCallers
    case let tool as ApplyPatchTool: return tool.allowedCallers
    case let tool as HostedMCPTool:
        return tool.toolConfig["allowed_callers"]?.arrayValue?.compactMap { $0.stringValue.flatMap(ToolCaller.init(rawValue:)) }
    case let tool as CodeInterpreterTool:
        return tool.toolConfig["allowed_callers"]?.arrayValue?.compactMap { $0.stringValue.flatMap(ToolCaller.init(rawValue:)) }
    default: return nil
    }
}

private func toolDisplayName(_ tool: any Tool) -> String {
    if let tool = tool as? FunctionTool { return tool.qualifiedName }
    if let tool = tool as? HostedMCPTool { return "hosted MCP server `\(tool.toolConfig.string("server_label") ?? "unknown")`" }
    return tool.name
}

/// Validates the complete Responses Programmatic Tool Calling configuration.
func validateResponsesProgrammaticToolCallingConfiguration(
    _ tools: [any Tool],
    toolChoice: ToolChoice? = nil,
    allowOpaqueToolSearchSurface: Bool = false
) throws {
    let programmaticCount = tools.filter { $0 is ProgrammaticToolCallingTool }.count
    if programmaticCount > 1 {
        throw UserError("Only one ProgrammaticToolCallingTool() is allowed when using OpenAI Responses models.")
    }
    let hasProgrammatic = programmaticCount > 0
    if toolChoice == .named("programmatic_tool_calling"), !hasProgrammatic {
        throw UserError(
            "tool_choice='programmatic_tool_calling' requires ProgrammaticToolCallingTool() when using OpenAI Responses models."
        )
    }
    var eligible = 0
    for tool in tools {
        guard let callers = toolAllowedCallers(tool), callers.contains(.programmatic) else { continue }
        eligible += 1
        if !callers.contains(.direct), !hasProgrammatic {
            throw UserError(
                "Tool `\(toolDisplayName(tool))` only allows programmatic callers and requires "
                    + "ProgrammaticToolCallingTool() when using OpenAI Responses models."
            )
        }
    }
    let hasToolSearch = tools.contains { $0 is ToolSearchTool }
    if hasProgrammatic, eligible == 0, !hasToolSearch, !allowOpaqueToolSearchSurface {
        throw UserError(
            "ProgrammaticToolCallingTool() requires at least one tool whose allowed_callers includes "
                + "'programmatic', a ToolSearchTool(), or an opaque prompt-managed tool surface."
        )
    }
}

/// Backend-specific converters validate reserved tool choices.
func ensureToolChoiceSupportsBackend(_ toolChoice: ToolChoice?, backendName: String) throws {
    if toolChoice == .named("programmatic_tool_calling") {
        throw UserError(
            "tool_choice='programmatic_tool_calling' is only supported with OpenAI Responses models "
                + "and cannot be used with \(backendName)."
        )
    }
}
