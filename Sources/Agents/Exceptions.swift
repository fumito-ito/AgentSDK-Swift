import Foundation

/// Data collected from an agent run when an exception occurs.
public struct RunErrorDetails: Sendable, CustomStringConvertible {
    public var input: RunInput
    public var newItems: [RunItem]
    public var rawResponses: [ModelResponse]
    public var lastAgent: Agent
    public var contextWrapper: RunContextWrapper
    public var inputGuardrailResults: [InputGuardrailResult]
    public var outputGuardrailResults: [OutputGuardrailResult]
    /// Tool input guardrail results accumulated from completed turns before the run failed.
    public var toolInputGuardrailResults: [ToolInputGuardrailResult]
    /// Tool output guardrail results accumulated from completed turns before the run failed.
    public var toolOutputGuardrailResults: [ToolOutputGuardrailResult]

    public init(
        input: RunInput,
        newItems: [RunItem],
        rawResponses: [ModelResponse],
        lastAgent: Agent,
        contextWrapper: RunContextWrapper,
        inputGuardrailResults: [InputGuardrailResult],
        outputGuardrailResults: [OutputGuardrailResult],
        toolInputGuardrailResults: [ToolInputGuardrailResult] = [],
        toolOutputGuardrailResults: [ToolOutputGuardrailResult] = []
    ) {
        self.input = input
        self.newItems = newItems
        self.rawResponses = rawResponses
        self.lastAgent = lastAgent
        self.contextWrapper = contextWrapper
        self.inputGuardrailResults = inputGuardrailResults
        self.outputGuardrailResults = outputGuardrailResults
        self.toolInputGuardrailResults = toolInputGuardrailResults
        self.toolOutputGuardrailResults = toolOutputGuardrailResults
    }

    public var description: String { prettyPrintRunErrorDetails(self) }
}

/// Base class for all exceptions in the Agents SDK.
///
/// Swift: upstream exceptions are a class hierarchy; this mirrors it so callers can catch the base
/// class (see porting/decisions.md D7). `runData` is attached by the runner after construction.
open class AgentsException: Error, CustomStringConvertible, @unchecked Sendable {
    /// The human-readable message, equivalent to `str(exception)`.
    public let message: String
    /// Run data attached by the runner when the exception escapes `Runner.run`.
    public var runData: RunErrorDetails?

    public init(_ message: String) {
        self.message = message
        self.runData = nil
    }

    open var description: String { message }

    /// The Python class name, used in tracing and error payloads.
    open var typeName: String { String(describing: type(of: self)) }
}

/// Exception raised when the maximum number of turns is exceeded.
public final class MaxTurnsExceeded: AgentsException, @unchecked Sendable {}

/// Exception raised when the model does something unexpected, e.g. calling a tool that doesn't
/// exist, or providing malformed JSON.
public final class ModelBehaviorError: AgentsException, @unchecked Sendable {}

/// Exception raised when the model refuses to produce the requested output.
public final class ModelRefusalError: AgentsException, @unchecked Sendable {
    /// The refusal text returned by the model.
    public let refusal: String

    public init(refusal: String) {
        self.refusal = refusal
        super.init("Model refused to produce output: \(refusal)")
    }
}

/// Exception raised when a model-call attempt exceeds its configured timeout.
public final class ModelTimeoutError: AgentsException, @unchecked Sendable {
    public let timeoutSeconds: Double

    public init(timeoutSeconds: Double) {
        self.timeoutSeconds = timeoutSeconds
        super.init("Model call timed out after \(formatSeconds(timeoutSeconds)) seconds.")
    }
}

/// Exception raised when the user makes an error using the SDK.
public final class UserError: AgentsException, @unchecked Sendable {}

/// Exception raised when an MCP tool call is internally cancelled.
public final class MCPToolCancellationError: AgentsException, @unchecked Sendable {}

/// Exception raised when a function tool invocation exceeds its timeout.
public final class ToolTimeoutError: AgentsException, @unchecked Sendable {
    public let toolName: String
    public let timeoutSeconds: Double

    public init(toolName: String, timeoutSeconds: Double) {
        self.toolName = toolName
        self.timeoutSeconds = timeoutSeconds
        super.init("Tool '\(toolName)' timed out after \(formatSeconds(timeoutSeconds)) seconds.")
    }
}

/// Exception raised when an input guardrail tripwire is triggered.
public final class InputGuardrailTripwireTriggered: AgentsException, @unchecked Sendable {
    /// The result data of the guardrail that was triggered.
    public let guardrailResult: InputGuardrailResult

    public init(guardrailResult: InputGuardrailResult) {
        self.guardrailResult = guardrailResult
        super.init("Guardrail InputGuardrail triggered tripwire")
    }
}

/// Exception raised when an output guardrail tripwire is triggered.
public final class OutputGuardrailTripwireTriggered: AgentsException, @unchecked Sendable {
    /// The result data of the guardrail that was triggered.
    public let guardrailResult: OutputGuardrailResult

    public init(guardrailResult: OutputGuardrailResult) {
        self.guardrailResult = guardrailResult
        super.init("Guardrail OutputGuardrail triggered tripwire")
    }
}

/// Exception raised when a tool input guardrail tripwire is triggered.
public final class ToolInputGuardrailTripwireTriggered: AgentsException, @unchecked Sendable {
    /// The guardrail that was triggered.
    public let guardrail: ToolInputGuardrail
    /// The output from the guardrail function.
    public let output: ToolGuardrailFunctionOutput

    public init(guardrail: ToolInputGuardrail, output: ToolGuardrailFunctionOutput) {
        self.guardrail = guardrail
        self.output = output
        super.init("Tool input guardrail ToolInputGuardrail triggered tripwire")
    }
}

/// Exception raised when a tool output guardrail tripwire is triggered.
public final class ToolOutputGuardrailTripwireTriggered: AgentsException, @unchecked Sendable {
    /// The guardrail that was triggered.
    public let guardrail: ToolOutputGuardrail
    /// The output from the guardrail function.
    public let output: ToolGuardrailFunctionOutput

    public init(guardrail: ToolOutputGuardrail, output: ToolGuardrailFunctionOutput) {
        self.guardrail = guardrail
        self.output = output
        super.init("Tool output guardrail ToolOutputGuardrail triggered tripwire")
    }
}

/// Formats seconds like Python's `f"{value:g}"`.
func formatSeconds(_ value: Double) -> String {
    if value.rounded() == value, abs(value) < 1e15 {
        return String(Int(value))
    }
    var text = String(format: "%.6g", value)
    if text.contains("e") { return text }
    while text.hasSuffix("0") { text.removeLast() }
    if text.hasSuffix(".") { text.removeLast() }
    return text
}

/// Returns a model-visible description of an arbitrary Swift error, equivalent to `str(error)`.
public func describeError(_ error: any Error) -> String {
    if let error = error as? AgentsException { return error.message }
    if let error = error as? LocalizedError, let description = error.errorDescription { return description }
    // `String(describing:)` uses the dynamic type's `CustomStringConvertible` conformance.
    return String(describing: error)
}

/// Returns the type name of an arbitrary Swift error, equivalent to `type(error).__name__`.
public func errorTypeName(_ error: any Error) -> String {
    if let error = error as? AgentsException { return error.typeName }
    return String(describing: type(of: error))
}
