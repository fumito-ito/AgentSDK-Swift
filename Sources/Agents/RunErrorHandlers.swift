import Foundation

/// Data available to run error handlers.
public struct RunErrorData: Sendable {
    public var input: RunInput
    public var newItems: [RunItem]
    public var history: [TResponseInputItem]
    public var output: [TResponseInputItem]
    public var rawResponses: [ModelResponse]
    public var lastAgent: Agent
}

/// The input passed to a run error handler.
public struct RunErrorHandlerInput: Sendable {
    /// `MaxTurnsExceeded`, `ModelRefusalError`, or `ModelBehaviorError`.
    public var error: AgentsException
    public var context: RunContextWrapper
    public var runData: RunErrorData
}

/// The result returned by a run error handler.
public struct RunErrorHandlerResult: Sendable {
    public var finalOutput: any Sendable
    public var includeInHistory: Bool

    public init(finalOutput: any Sendable, includeInHistory: Bool = true) {
        self.finalOutput = finalOutput
        self.includeInHistory = includeInHistory
    }
}

/// Handles a run error. Returning `nil` lets the error propagate.
public typealias RunErrorHandler = @Sendable (RunErrorHandlerInput) async throws -> RunErrorHandlerResult?

/// Error handlers keyed by error kind.
public struct RunErrorHandlers: Sendable {
    public var maxTurns: RunErrorHandler?
    public var modelRefusal: RunErrorHandler?
    public var invalidFinalOutput: RunErrorHandler?

    public init(maxTurns: RunErrorHandler? = nil, modelRefusal: RunErrorHandler? = nil, invalidFinalOutput: RunErrorHandler? = nil) {
        self.maxTurns = maxTurns
        self.modelRefusal = modelRefusal
        self.invalidFinalOutput = invalidFinalOutput
    }

    func handler(for kind: String) -> RunErrorHandler? {
        switch kind {
        case "max_turns": return maxTurns
        case "model_refusal": return modelRefusal
        case "invalid_final_output": return invalidFinalOutput
        default: return nil
        }
    }
}
