import Foundation

/// Tracing configuration passed to models.
public enum ModelTracing: Int, Sendable {
    /// Tracing is disabled entirely.
    case disabled = 0
    /// Tracing is enabled, and all data is included.
    case enabled = 1
    /// Tracing is enabled, but inputs/outputs are not included.
    case enabledWithoutData = 2

    public func isDisabled() -> Bool { self == .disabled }
    public func includeData() -> Bool { self == .enabled }
}

/// Returns the `ModelTracing` setting based on run-level tracing configuration.
public func getModelTracingImpl(tracingDisabled: Bool, traceIncludeSensitiveData: Bool) -> ModelTracing {
    if tracingDisabled { return .disabled }
    return traceIncludeSensitiveData ? .enabled : .enabledWithoutData
}

/// The arguments of one model call, shared by `getResponse` and `streamResponse`.
public struct ModelRequest: Sendable {
    public var systemInstructions: String?
    public var input: RunInput
    public var modelSettings: ModelSettings
    public var tools: [any Tool]
    public var outputSchema: (any AgentOutputSchemaBase)?
    public var handoffs: [Handoff]
    public var tracing: ModelTracing
    public var previousResponseId: String?
    public var conversationId: String?
    public var prompt: JSONObject?

    public init(
        systemInstructions: String?,
        input: RunInput,
        modelSettings: ModelSettings,
        tools: [any Tool],
        outputSchema: (any AgentOutputSchemaBase)?,
        handoffs: [Handoff],
        tracing: ModelTracing,
        previousResponseId: String? = nil,
        conversationId: String? = nil,
        prompt: JSONObject? = nil
    ) {
        self.systemInstructions = systemInstructions
        self.input = input
        self.modelSettings = modelSettings
        self.tools = tools
        self.outputSchema = outputSchema
        self.handoffs = handoffs
        self.tracing = tracing
        self.previousResponseId = previousResponseId
        self.conversationId = conversationId
        self.prompt = prompt
    }
}

/// The base interface for calling an LLM.
///
/// Model implementations must assign a non-empty call ID to each tool invocation. A call ID must
/// identify one canonical invocation for the lifetime of the run and must not be reused for a
/// changed tool identity or payload. Tool outputs must retain the call ID for correlation.
///
/// Swift: upstream passes the call arguments individually; here they are grouped in
/// `ModelRequest` so streaming and non-streaming calls share one signature.
public protocol Model: AnyObject, Sendable {
    /// Gets a full response from the model.
    func getResponse(_ request: ModelRequest) async throws -> ModelResponse
    /// Streams a response from the model as Responses API stream events.
    func streamResponse(_ request: ModelRequest) -> AsyncThrowingStream<TResponseStreamEvent, any Error>
    /// Returns provider-specific retry guidance for a failed model request.
    func getRetryAdvice(_ request: ModelRetryAdviceRequest) -> ModelRetryAdvice?
    /// Releases any resources held by the model.
    func close() async
    /// Releases run-scoped resources after the runner finishes using this model.
    func cleanupOnRunEnd(owner: AnyObject) async
}

extension Model {
    public func getRetryAdvice(_ request: ModelRetryAdviceRequest) -> ModelRetryAdvice? { nil }
    public func close() async {}
    public func cleanupOnRunEnd(owner: AnyObject) async {}
}

/// Looks up models by name.
public protocol ModelProvider: AnyObject, Sendable {
    /// Gets a model by name.
    func getModel(_ modelName: String?) throws -> any Model
    /// Releases any resources held by the provider.
    func aclose() async
}

extension ModelProvider {
    public func aclose() async {}
}

/// The model used by an agent or run: a name resolved by the provider, or a model instance.
public enum AgentModel: Sendable {
    case name(String)
    case instance(any Model)

    public var name: String? {
        if case .name(let name) = self { return name }
        return nil
    }
}

extension AgentModel: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self = .name(value) }
}
