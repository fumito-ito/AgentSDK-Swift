import Foundation

/// The default maximum number of turns in a run.
public let defaultMaxTurns = 10

/// Returns the default for `traceIncludeSensitiveData` based on the environment.
func defaultTraceIncludeSensitiveData() -> Bool {
    let value = ProcessInfo.processInfo.environment["OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA"] ?? "true"
    return ["1", "true", "yes", "on"].contains(value.trimmingCharacters(in: .whitespaces).lowercased())
}

/// Container for the data that will be sent to the model.
public struct ModelInputData: Sendable {
    public var input: [TResponseInputItem]
    public var instructions: String?

    public init(input: [TResponseInputItem], instructions: String?) {
        self.input = input
        self.instructions = instructions
    }
}

/// Data passed to `RunConfig.callModelInputFilter` before each model call.
public struct CallModelData: Sendable {
    public var modelData: ModelInputData
    public var agent: Agent
    public var context: any Sendable
}

/// Edits the model input before each call.
public typealias CallModelInputFilter = @Sendable (CallModelData) async throws -> ModelInputData

/// Controls how reasoning items are converted to next-turn model input.
public enum ReasoningItemIdPolicy: String, Sendable, Codable {
    /// Keeps reasoning item IDs as-is.
    case preserve
    /// Strips reasoning item IDs from model input built by the runner.
    case omit
}

/// Controls unresolved function tool calls emitted by the model.
public enum ToolNotFoundBehavior: String, Sendable {
    /// Raises `ModelBehaviorError` (default).
    case raiseError = "raise_error"
    /// Returns a model-visible error output and continues the run.
    case returnErrorToModel = "return_error_to_model"
}

/// Controls collisions between function tool and handoff names.
public enum ToolNameCollisionPolicy: String, Sendable {
    /// Logs a warning and exposes only the current dispatch winner.
    case warn
    /// Raises `UserError` before the model is called.
    case error
}

/// Data passed to `RunConfig.toolErrorFormatter`.
public struct ToolErrorFormatterArgs: Sendable {
    /// `"approval_rejected"` or `"tool_not_found"`.
    public var kind: String
    /// `"function"`, `"computer"`, `"shell"`, `"apply_patch"`, or `"custom"`.
    public var toolType: String
    public var toolName: String
    public var callId: String
    /// The SDK default message for this error kind.
    public var defaultMessage: String
    public var runContext: RunContextWrapper
}

/// Formats tool error messages returned to the model. Returning `nil` uses the default message.
public typealias ToolErrorFormatter = @Sendable (ToolErrorFormatterArgs) async throws -> String?

/// Data passed to output-guardrail blocked-message formatters.
public struct OutputGuardrailBlockedMessageArgs: Sendable {
    public var defaultMessage: String
    public var guardrailName: String
    public var agent: Agent
    public var runContext: RunContextWrapper
}

/// Grouped SDK-side execution settings for local tool calls.
public struct ToolExecutionConfig: Sendable, Equatable {
    /// Maximum number of local function tool calls to execute concurrently. `nil` starts all.
    public var maxFunctionToolConcurrency: Int?
    /// Run function tool input guardrails before emitting a pending approval interruption.
    public var preApprovalToolInputGuardrails: Bool

    public init(maxFunctionToolConcurrency: Int? = nil, preApprovalToolInputGuardrails: Bool = false) throws {
        if let maxFunctionToolConcurrency, maxFunctionToolConcurrency < 1 {
            throw ValueError("tool_execution.max_function_tool_concurrency must be at least 1")
        }
        self.maxFunctionToolConcurrency = maxFunctionToolConcurrency
        self.preApprovalToolInputGuardrails = preApprovalToolInputGuardrails
    }
}

/// Configures settings for the entire agent run.
public struct RunConfig: Sendable {
    /// The model to use for the entire run, overriding every agent's model.
    public var model: AgentModel?
    /// The provider used to look up string model names. Defaults to `MultiProvider`.
    public var modelProvider: any ModelProvider
    /// Global model settings. Non-`nil` values override agent-specific settings.
    public var modelSettings: ModelSettings?
    /// A global input filter applied to all handoffs without their own filter.
    public var handoffInputFilter: HandoffInputFilter?
    /// Opt-in beta: compact prior run history into ordered assistant summary segments.
    public var nestHandoffHistory: Bool
    /// Maps the normalized transcript to the next agent's input history when nesting.
    public var handoffHistoryMapper: HandoffHistoryMapper?
    /// Input guardrails to run on the initial run input.
    public var inputGuardrails: [InputGuardrail]?
    /// Output guardrails to run on the final output of the run.
    public var outputGuardrails: [OutputGuardrail]?
    /// Whether tracing is disabled for the run.
    public var tracingDisabled: Bool
    /// Tracing configuration for this run.
    public var tracing: TracingConfig?
    /// Whether to include potentially sensitive data (tool inputs/outputs, generations) in traces.
    public var traceIncludeSensitiveData: Bool
    /// The name of the run, used for tracing.
    public var workflowName: String
    /// A custom trace ID.
    public var traceId: String?
    /// A grouping identifier linking multiple traces, e.g. a chat thread ID.
    public var groupId: String?
    /// Additional metadata to include with the trace.
    public var traceMetadata: JSONObject?
    /// Defines how session history combines with new input.
    public var sessionInputCallback: SessionInputCallback?
    /// Invoked immediately before calling the model to edit the model input.
    public var callModelInputFilter: CallModelInputFilter?
    /// Formats tool error messages returned to the model.
    public var toolErrorFormatter: ToolErrorFormatter?
    /// Session settings overriding the session's defaults.
    public var sessionSettings: SessionSettings?
    /// How reasoning items are converted to next-turn model input.
    public var reasoningItemIdPolicy: ReasoningItemIdPolicy?
    /// SDK-side execution settings for local tool calls.
    public var toolExecution: ToolExecutionConfig?
    /// How unresolved function tool calls are handled.
    public var toolNotFoundBehavior: ToolNotFoundBehavior
    /// How function tool and handoff name collisions are handled.
    public var toolNameCollisionPolicy: ToolNameCollisionPolicy
    /// The placeholder retained for terminal tool output rejected by an output guardrail.
    public var outputGuardrailBlockedMessage: String?

    public init(
        model: AgentModel? = nil,
        modelProvider: (any ModelProvider)? = nil,
        modelSettings: ModelSettings? = nil,
        handoffInputFilter: HandoffInputFilter? = nil,
        nestHandoffHistory: Bool = false,
        handoffHistoryMapper: HandoffHistoryMapper? = nil,
        inputGuardrails: [InputGuardrail]? = nil,
        outputGuardrails: [OutputGuardrail]? = nil,
        tracingDisabled: Bool = false,
        tracing: TracingConfig? = nil,
        traceIncludeSensitiveData: Bool? = nil,
        workflowName: String = "Agent workflow",
        traceId: String? = nil,
        groupId: String? = nil,
        traceMetadata: JSONObject? = nil,
        sessionInputCallback: SessionInputCallback? = nil,
        callModelInputFilter: CallModelInputFilter? = nil,
        toolErrorFormatter: ToolErrorFormatter? = nil,
        sessionSettings: SessionSettings? = nil,
        reasoningItemIdPolicy: ReasoningItemIdPolicy? = nil,
        toolExecution: ToolExecutionConfig? = nil,
        toolNotFoundBehavior: ToolNotFoundBehavior = .raiseError,
        toolNameCollisionPolicy: ToolNameCollisionPolicy = .warn,
        outputGuardrailBlockedMessage: String? = nil
    ) {
        self.model = model
        self.modelProvider = modelProvider ?? MultiProvider()
        self.modelSettings = modelSettings
        self.handoffInputFilter = handoffInputFilter
        self.nestHandoffHistory = nestHandoffHistory
        self.handoffHistoryMapper = handoffHistoryMapper
        self.inputGuardrails = inputGuardrails
        self.outputGuardrails = outputGuardrails
        self.tracingDisabled = tracingDisabled
        self.tracing = tracing
        self.traceIncludeSensitiveData = traceIncludeSensitiveData ?? defaultTraceIncludeSensitiveData()
        self.workflowName = workflowName
        self.traceId = traceId
        self.groupId = groupId
        self.traceMetadata = traceMetadata
        self.sessionInputCallback = sessionInputCallback
        self.callModelInputFilter = callModelInputFilter
        self.toolErrorFormatter = toolErrorFormatter
        self.sessionSettings = sessionSettings
        self.reasoningItemIdPolicy = reasoningItemIdPolicy
        self.toolExecution = toolExecution
        self.toolNotFoundBehavior = toolNotFoundBehavior
        self.toolNameCollisionPolicy = toolNameCollisionPolicy
        self.outputGuardrailBlockedMessage = outputGuardrailBlockedMessage
    }
}
