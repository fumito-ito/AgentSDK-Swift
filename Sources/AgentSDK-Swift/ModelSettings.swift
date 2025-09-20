import Foundation

/// Settings for configuring model behavior
public struct ModelSettings: Sendable {
    public enum ToolChoice: Sendable, Equatable {
        case auto
        case required
        case none
        case named(String)
    }

    public enum TruncationStrategy: String, Sendable, Equatable {
        case auto
        case disabled
    }

    public struct Reasoning: Sendable, Equatable {
        public enum Effort: String, Sendable {
            case minimal
            case low
            case medium
            case high
        }

        public var effort: Effort?

        public init(effort: Effort? = nil) {
            self.effort = effort
        }
    }

    public enum Verbosity: String, Sendable, Equatable {
        case low
        case medium
        case high
    }

    /// The name of the model to use
    public var modelName: String
    /// Temperature controls randomness (0.0 to 1.0)
    public var temperature: Double?
    /// Top-p controls diversity of output (0.0 to 1.0)
    public var topP: Double?
    /// Penalizes repeated tokens
    public var frequencyPenalty: Double?
    /// Encourages exploring new topics
    public var presencePenalty: Double?
    /// Configures how the model chooses tools
    public var toolChoice: ToolChoice?
    /// Enables multiple tool calls in a single turn when supported
    public var parallelToolCalls: Bool?
    /// Controls truncation behavior for long conversations
    public var truncation: TruncationStrategy?
    /// Maximum number of tokens to generate
    public var maxTokens: Int?
    /// Reasoning configuration for reasoning-capable models
    public var reasoning: Reasoning?
    /// Verbosity configuration for the response
    public var verbosity: Verbosity?
    /// Metadata forwarded to the model provider
    public var metadata: [String: String]?
    /// Whether to store the generated response server-side
    public var store: Bool?
    /// Whether to include usage statistics in responses
    public var includeUsage: Bool?
    /// Extra fields to include in the model response payload
    public var responseInclude: [String]?
    /// Number of top tokens to return log probabilities for
    public var topLogprobs: Int?
    /// Additional query parameters for the provider
    public var extraQuery: [String: String]?
    /// Additional body parameters for the provider
    public var extraBody: [String: String]?
    /// Additional headers for the provider
    public var extraHeaders: [String: String]?
    /// Response formats to use (e.g., JSON)
    public var responseFormat: ResponseFormat?
    /// Seeds for deterministic generation
    public var seed: Int?
    /// Additional model-specific parameters
    public var additionalParameters: [String: String]

    public init(
        modelName: String = "gpt-4.1",
        temperature: Double? = nil,
        topP: Double? = nil,
        frequencyPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        toolChoice: ToolChoice? = nil,
        parallelToolCalls: Bool? = nil,
        truncation: TruncationStrategy? = nil,
        maxTokens: Int? = nil,
        reasoning: Reasoning? = nil,
        verbosity: Verbosity? = nil,
        metadata: [String: String]? = nil,
        store: Bool? = nil,
        includeUsage: Bool? = nil,
        responseInclude: [String]? = nil,
        topLogprobs: Int? = nil,
        extraQuery: [String: String]? = nil,
        extraBody: [String: String]? = nil,
        extraHeaders: [String: String]? = nil,
        responseFormat: ResponseFormat? = nil,
        seed: Int? = nil,
        additionalParameters: [String: String] = [:]
    ) {
        self.modelName = modelName
        self.temperature = temperature
        self.topP = topP
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.toolChoice = toolChoice
        self.parallelToolCalls = parallelToolCalls
        self.truncation = truncation
        self.maxTokens = maxTokens
        self.reasoning = reasoning
        self.verbosity = verbosity
        self.metadata = metadata
        self.store = store
        self.includeUsage = includeUsage
        self.responseInclude = responseInclude
        self.topLogprobs = topLogprobs
        self.extraQuery = extraQuery
        self.extraBody = extraBody
        self.extraHeaders = extraHeaders
        self.responseFormat = responseFormat
        self.seed = seed
        self.additionalParameters = additionalParameters
    }

    /// Creates a copy of these settings with optional overrides
    public func with(
        modelName: String? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        maxTokens: Int? = nil,
        responseFormat: ResponseFormat? = nil,
        seed: Int? = nil,
        additionalParameters: [String: String]? = nil,
        frequencyPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        toolChoice: ToolChoice? = nil,
        parallelToolCalls: Bool? = nil,
        truncation: TruncationStrategy? = nil,
        reasoning: Reasoning? = nil,
        verbosity: Verbosity? = nil,
        metadata: [String: String]? = nil,
        store: Bool? = nil,
        includeUsage: Bool? = nil,
        responseInclude: [String]? = nil,
        topLogprobs: Int? = nil,
        extraQuery: [String: String]? = nil,
        extraBody: [String: String]? = nil,
        extraHeaders: [String: String]? = nil
    ) -> ModelSettings {
        var settings = self
        if let modelName { settings.modelName = modelName }
        if let temperature { settings.temperature = temperature }
        if let topP { settings.topP = topP }
        if let maxTokens { settings.maxTokens = maxTokens }
        if let responseFormat { settings.responseFormat = responseFormat }
        if let seed { settings.seed = seed }
        if let additionalParameters { settings.additionalParameters = additionalParameters }
        if let frequencyPenalty { settings.frequencyPenalty = frequencyPenalty }
        if let presencePenalty { settings.presencePenalty = presencePenalty }
        if let toolChoice { settings.toolChoice = toolChoice }
        if let parallelToolCalls { settings.parallelToolCalls = parallelToolCalls }
        if let truncation { settings.truncation = truncation }
        if let reasoning { settings.reasoning = reasoning }
        if let verbosity { settings.verbosity = verbosity }
        if let metadata { settings.metadata = metadata }
        if let store { settings.store = store }
        if let includeUsage { settings.includeUsage = includeUsage }
        if let responseInclude { settings.responseInclude = responseInclude }
        if let topLogprobs { settings.topLogprobs = topLogprobs }
        if let extraQuery { settings.extraQuery = extraQuery }
        if let extraBody { settings.extraBody = extraBody }
        if let extraHeaders { settings.extraHeaders = extraHeaders }
        return settings
    }

    /// Produces a new settings object by overlaying non-nil values from another instance.
    public func merged(with override: ModelSettings?) -> ModelSettings {
        guard let override else { return self }
        var merged = self
        if override.modelName != self.modelName { merged.modelName = override.modelName }
        if let value = override.temperature { merged.temperature = value }
        if let value = override.topP { merged.topP = value }
        if let value = override.maxTokens { merged.maxTokens = value }
        if let value = override.responseFormat { merged.responseFormat = value }
        if let value = override.seed { merged.seed = value }
        if let value = override.frequencyPenalty { merged.frequencyPenalty = value }
        if let value = override.presencePenalty { merged.presencePenalty = value }
        if let value = override.toolChoice { merged.toolChoice = value }
        if let value = override.parallelToolCalls { merged.parallelToolCalls = value }
        if let value = override.truncation { merged.truncation = value }
        if let value = override.reasoning { merged.reasoning = value }
        if let value = override.verbosity { merged.verbosity = value }
        if let value = override.metadata { merged.metadata = value }
        if let value = override.store { merged.store = value }
        if let value = override.includeUsage { merged.includeUsage = value }
        if let value = override.responseInclude { merged.responseInclude = value }
        if let value = override.topLogprobs { merged.topLogprobs = value }
        if let value = override.extraQuery {
            merged.extraQuery = merged.extraQuery?.merging(value, uniquingKeysWith: { _, new in new }) ?? value
        }
        if let value = override.extraBody {
            merged.extraBody = merged.extraBody?.merging(value, uniquingKeysWith: { _, new in new }) ?? value
        }
        if let value = override.extraHeaders {
            merged.extraHeaders = merged.extraHeaders?.merging(value, uniquingKeysWith: { _, new in new }) ?? value
        }
        if !override.additionalParameters.isEmpty {
            merged.additionalParameters = merged.additionalParameters.merging(override.additionalParameters) { _, new in new }
        }
        return merged
    }

    /// Generates a serialisable representation compatible with provider SDKs.
    public func toDictionaryRepresentation() -> [String: Any] {
        var dict: [String: Any] = [:]
        if let temperature { dict["temperature"] = temperature }
        if let topP { dict["top_p"] = topP }
        if let frequencyPenalty { dict["frequency_penalty"] = frequencyPenalty }
        if let presencePenalty { dict["presence_penalty"] = presencePenalty }
        if let toolChoice {
            switch toolChoice {
            case .auto: dict["tool_choice"] = "auto"
            case .required: dict["tool_choice"] = "required"
            case .none: dict["tool_choice"] = "none"
            case .named(let name): dict["tool_choice"] = name
            }
        }
        if let parallelToolCalls { dict["parallel_tool_calls"] = parallelToolCalls }
        if let truncation { dict["truncation"] = truncation.rawValue }
        if let maxTokens { dict["max_tokens"] = maxTokens }
        if let reasoning {
            var reasoningDict: [String: Any] = [:]
            if let effort = reasoning.effort { reasoningDict["effort"] = effort.rawValue }
            dict["reasoning"] = reasoningDict
        }
        if let verbosity { dict["verbosity"] = verbosity.rawValue }
        if let metadata { dict["metadata"] = metadata }
        if let store { dict["store"] = store }
        if let includeUsage { dict["include_usage"] = includeUsage }
        if let responseInclude { dict["response_include"] = responseInclude }
        if let topLogprobs { dict["top_logprobs"] = topLogprobs }
        if let extraQuery { dict["extra_query"] = extraQuery }
        if let extraBody { dict["extra_body"] = extraBody }
        if let extraHeaders { dict["extra_headers"] = extraHeaders }
        if let responseFormat { dict["response_format"] = responseFormat.jsonValue }
        if let seed { dict["seed"] = seed }
        if !additionalParameters.isEmpty {
            for (key, value) in additionalParameters {
                dict[key] = value
            }
        }
        return dict
    }

    public enum ResponseFormat: Sendable {
        case json
        case text

        public var jsonValue: String {
            switch self {
            case .json: return "json_object"
            case .text: return "text"
            }
        }
    }
}
