import Foundation

/// How the runner responds to a tool guardrail result.
public enum ToolGuardrailBehavior: Sendable, Equatable {
    /// Allow normal tool execution to continue (default).
    case allow
    /// Reject the tool call/output but continue, sending `message` to the model instead.
    case rejectContent(message: String)
    /// Halt execution by raising a tool guardrail tripwire exception.
    case raiseException

    public var type: String {
        switch self {
        case .allow: return "allow"
        case .rejectContent: return "reject_content"
        case .raiseException: return "raise_exception"
        }
    }
}

/// The output of a tool guardrail function.
public struct ToolGuardrailFunctionOutput: Sendable {
    /// Optional data about checks performed.
    public var outputInfo: (any Sendable)?
    /// Defines how the system should respond when this guardrail result is processed.
    public var behavior: ToolGuardrailBehavior

    public init(outputInfo: (any Sendable)? = nil, behavior: ToolGuardrailBehavior = .allow) {
        self.outputInfo = outputInfo
        self.behavior = behavior
    }

    /// Allows normal execution.
    public static func allow(outputInfo: (any Sendable)? = nil) -> ToolGuardrailFunctionOutput {
        ToolGuardrailFunctionOutput(outputInfo: outputInfo, behavior: .allow)
    }

    /// Rejects the content and sends `message` to the model instead of the tool result.
    public static func rejectContent(_ message: String, outputInfo: (any Sendable)? = nil) -> ToolGuardrailFunctionOutput {
        ToolGuardrailFunctionOutput(outputInfo: outputInfo, behavior: .rejectContent(message: message))
    }

    /// Halts execution by raising a tripwire exception.
    public static func raiseException(outputInfo: (any Sendable)? = nil) -> ToolGuardrailFunctionOutput {
        ToolGuardrailFunctionOutput(outputInfo: outputInfo, behavior: .raiseException)
    }
}

/// Data passed to a tool input guardrail.
public struct ToolInputGuardrailData: Sendable {
    /// The tool context containing information about the current tool execution.
    public var context: ToolContext
    /// The agent that is executing the tool.
    public var agent: Agent

    public init(context: ToolContext, agent: Agent) {
        self.context = context
        self.agent = agent
    }
}

/// Data passed to a tool output guardrail.
public struct ToolOutputGuardrailData: Sendable {
    public var context: ToolContext
    public var agent: Agent
    /// The output produced by the tool function.
    public var output: any Sendable

    public init(context: ToolContext, agent: Agent, output: any Sendable) {
        self.context = context
        self.agent = agent
        self.output = output
    }
}

/// The result of a tool input guardrail run.
public struct ToolInputGuardrailResult: Sendable {
    public var guardrail: ToolInputGuardrail
    public var output: ToolGuardrailFunctionOutput

    public init(guardrail: ToolInputGuardrail, output: ToolGuardrailFunctionOutput) {
        self.guardrail = guardrail
        self.output = output
    }
}

/// The result of a tool output guardrail run.
public struct ToolOutputGuardrailResult: Sendable {
    public var guardrail: ToolOutputGuardrail
    public var output: ToolGuardrailFunctionOutput

    public init(guardrail: ToolOutputGuardrail, output: ToolGuardrailFunctionOutput) {
        self.guardrail = guardrail
        self.output = output
    }
}

/// A guardrail that runs before a function tool is invoked.
public struct ToolInputGuardrail: Sendable {
    public typealias Function = @Sendable (ToolInputGuardrailData) async throws -> ToolGuardrailFunctionOutput

    public var guardrailFunction: Function
    public var name: String?

    public init(name: String? = nil, guardrailFunction: @escaping Function) {
        self.guardrailFunction = guardrailFunction
        self.name = name
    }

    public func getName() -> String { name ?? "tool_input_guardrail" }

    public func run(_ data: ToolInputGuardrailData) async throws -> ToolGuardrailFunctionOutput {
        try await guardrailFunction(data)
    }
}

/// A guardrail that runs after a function tool returns.
public struct ToolOutputGuardrail: Sendable {
    public typealias Function = @Sendable (ToolOutputGuardrailData) async throws -> ToolGuardrailFunctionOutput

    public var guardrailFunction: Function
    public var name: String?

    public init(name: String? = nil, guardrailFunction: @escaping Function) {
        self.guardrailFunction = guardrailFunction
        self.name = name
    }

    public func getName() -> String { name ?? "tool_output_guardrail" }

    public func run(_ data: ToolOutputGuardrailData) async throws -> ToolGuardrailFunctionOutput {
        try await guardrailFunction(data)
    }
}

/// Creates a `ToolInputGuardrail` (the `@tool_input_guardrail` decorator equivalent).
public func toolInputGuardrail(name: String? = nil, _ function: @escaping ToolInputGuardrail.Function) -> ToolInputGuardrail {
    ToolInputGuardrail(name: name, guardrailFunction: function)
}

/// Creates a `ToolOutputGuardrail` (the `@tool_output_guardrail` decorator equivalent).
public func toolOutputGuardrail(name: String? = nil, _ function: @escaping ToolOutputGuardrail.Function) -> ToolOutputGuardrail {
    ToolOutputGuardrail(name: name, guardrailFunction: function)
}
