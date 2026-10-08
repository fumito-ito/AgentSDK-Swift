import Foundation

/// The output of a guardrail function.
public struct GuardrailFunctionOutput: Sendable {
    /// Optional information about the guardrail's output, such as the checks it performed.
    public var outputInfo: (any Sendable)?
    /// Whether the tripwire was triggered. If triggered, the agent's execution will be halted.
    public var tripwireTriggered: Bool

    public init(outputInfo: (any Sendable)? = nil, tripwireTriggered: Bool) {
        self.outputInfo = outputInfo
        self.tripwireTriggered = tripwireTriggered
    }
}

/// The result of an input guardrail run.
public struct InputGuardrailResult: Sendable {
    /// The guardrail that was run.
    public var guardrail: InputGuardrail
    /// The output of the guardrail function.
    public var output: GuardrailFunctionOutput

    public init(guardrail: InputGuardrail, output: GuardrailFunctionOutput) {
        self.guardrail = guardrail
        self.output = output
    }
}

/// The result of an output guardrail run.
public struct OutputGuardrailResult: Sendable {
    /// The guardrail that was run.
    public var guardrail: OutputGuardrail
    /// The output of the agent that was checked by the guardrail.
    public var agentOutput: any Sendable
    /// The agent that was checked by the guardrail.
    public var agent: Agent
    /// The output of the guardrail function.
    public var output: GuardrailFunctionOutput

    public init(guardrail: OutputGuardrail, agentOutput: any Sendable, agent: Agent, output: GuardrailFunctionOutput) {
        self.guardrail = guardrail
        self.agentOutput = agentOutput
        self.agent = agent
        self.output = output
    }
}

/// Input guardrails are checks that run either in parallel with the agent or before it starts.
///
/// Guardrail functions return a `GuardrailFunctionOutput`. If its `tripwireTriggered` is `true`,
/// the agent's execution stops immediately and `InputGuardrailTripwireTriggered` is thrown.
public struct InputGuardrail: Sendable {
    public typealias Function = @Sendable (RunContextWrapper, Agent, RunInput) async throws -> GuardrailFunctionOutput

    /// Receives the agent input and the context, and returns a `GuardrailFunctionOutput`.
    public var guardrailFunction: Function
    /// The name of the guardrail, used for tracing.
    public var name: String?
    /// Whether the guardrail runs concurrently with the agent (`true`, default) or before the
    /// agent starts (`false`).
    public var runInParallel: Bool

    public init(name: String? = nil, runInParallel: Bool = true, guardrailFunction: @escaping Function) {
        self.guardrailFunction = guardrailFunction
        self.name = name
        self.runInParallel = runInParallel
    }

    /// Swift: Python falls back to the function's `__name__`; Swift closures have no name, so
    /// unnamed guardrails use `"input_guardrail"`.
    public func getName() -> String { name ?? "input_guardrail" }

    public func run(agent: Agent, input: RunInput, context: RunContextWrapper) async throws -> InputGuardrailResult {
        InputGuardrailResult(guardrail: self, output: try await guardrailFunction(context, agent, input))
    }
}

/// Output guardrails are checks that run on the final output of an agent.
///
/// If the output's `tripwireTriggered` is `true`, `OutputGuardrailTripwireTriggered` is thrown.
public struct OutputGuardrail: Sendable {
    public typealias Function = @Sendable (RunContextWrapper, Agent, any Sendable) async throws -> GuardrailFunctionOutput

    /// Receives the final agent, its output, and the context.
    public var guardrailFunction: Function
    /// The name of the guardrail, used for tracing.
    public var name: String?

    public init(name: String? = nil, guardrailFunction: @escaping Function) {
        self.guardrailFunction = guardrailFunction
        self.name = name
    }

    /// Swift: unnamed guardrails use `"output_guardrail"` (see `InputGuardrail.getName()`).
    public func getName() -> String { name ?? "output_guardrail" }

    public func run(context: RunContextWrapper, agent: Agent, agentOutput: any Sendable) async throws -> OutputGuardrailResult {
        OutputGuardrailResult(
            guardrail: self,
            agentOutput: agentOutput,
            agent: agent,
            output: try await guardrailFunction(context, agent, agentOutput)
        )
    }
}

/// Creates an `InputGuardrail` (the `@input_guardrail` decorator equivalent).
public func inputGuardrail(
    name: String? = nil,
    runInParallel: Bool = true,
    _ function: @escaping InputGuardrail.Function
) -> InputGuardrail {
    InputGuardrail(name: name, runInParallel: runInParallel, guardrailFunction: function)
}

/// Creates an `OutputGuardrail` (the `@output_guardrail` decorator equivalent).
public func outputGuardrail(name: String? = nil, _ function: @escaping OutputGuardrail.Function) -> OutputGuardrail {
    OutputGuardrail(name: name, guardrailFunction: function)
}
