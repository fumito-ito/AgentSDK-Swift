import Foundation

/// Forces the model to call a specific tool on a hosted MCP server.
public struct MCPToolChoice: Sendable, Equatable, Hashable {
    public var serverLabel: String
    public var name: String

    public init(serverLabel: String, name: String) {
        self.serverLabel = serverLabel
        self.name = name
    }
}

/// The tool choice to use when calling the model.
public enum ToolChoice: Sendable, Equatable, Hashable {
    case auto
    case required
    case none
    /// Forces a specific tool by name.
    case named(String)
    /// Forces a specific hosted MCP tool.
    case mcp(MCPToolChoice)

    /// Creates a tool choice from its string form (`"auto"`, `"required"`, `"none"`, or a tool name).
    public init(_ value: String) {
        switch value {
        case "auto": self = .auto
        case "required": self = .required
        case "none": self = .none
        default: self = .named(value)
        }
    }

    /// The string form for string-valued choices.
    public var stringValue: String? {
        switch self {
        case .auto: return "auto"
        case .required: return "required"
        case .none: return "none"
        case .named(let name): return name
        case .mcp: return nil
        }
    }

    public var json: JSONValue {
        switch self {
        case .mcp(let choice):
            return ["server_label": .string(choice.serverLabel), "name": .string(choice.name)]
        default:
            return .string(stringValue!)
        }
    }

    init?(json: JSONValue) {
        switch json {
        case .string(let value):
            self.init(value)
        case .object(let object):
            guard let label = object.string("server_label"), let name = object.string("name") else { return nil }
            self = .mcp(MCPToolChoice(serverLabel: label, name: name))
        default:
            return nil
        }
    }
}

extension ToolChoice: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self.init(value) }
}

/// Reasoning configuration (`openai.types.shared.Reasoning`).
///
/// Swift: JSON-backed so that new server fields (for example `context` or `mode`) round-trip
/// unchanged, as with pydantic `model_extra`.
public struct Reasoning: Sendable, Equatable, Hashable {
    public var json: JSONObject

    public init(json: JSONObject = [:]) {
        self.json = json
    }

    public init(effort: String? = nil, summary: String? = nil, mode: String? = nil, context: String? = nil, generateSummary: String? = nil) {
        var json = JSONObject()
        if let effort { json["effort"] = .string(effort) }
        if let summary { json["summary"] = .string(summary) }
        if let mode { json["mode"] = .string(mode) }
        if let context { json["context"] = .string(context) }
        if let generateSummary { json["generate_summary"] = .string(generateSummary) }
        self.json = json
    }

    public var effort: String? {
        get { json.string("effort") }
        set { json["effort"] = newValue.map(JSONValue.string) }
    }

    public var summary: String? {
        get { json.string("summary") }
        set { json["summary"] = newValue.map(JSONValue.string) }
    }

    public var mode: String? { json.string("mode") }
    public var context: String? { json.string("context") }

    /// Serialized form, with the known fields always present like pydantic dumps.
    var dumped: JSONObject {
        var result: JSONObject = ["effort": json["effort"] ?? .null, "generate_summary": json["generate_summary"] ?? .null, "summary": json["summary"] ?? .null]
        result.merge(json)
        return result
    }
}

/// Settings to use when calling an LLM.
///
/// Holds optional model configuration parameters (e.g. temperature, top_p, penalties,
/// truncation). Not all models/providers support all of these parameters.
public struct ModelSettings: Sendable, Equatable {
    /// The temperature to use when calling the model.
    public var temperature: Double?
    /// The top_p to use when calling the model.
    public var topP: Double?
    /// The frequency penalty to use when calling the model.
    public var frequencyPenalty: Double?
    /// The presence penalty to use when calling the model.
    public var presencePenalty: Double?
    /// The tool choice to use when calling the model.
    public var toolChoice: ToolChoice?
    /// Controls whether the model can make multiple parallel tool calls in a single turn.
    public var parallelToolCalls: Bool?
    /// The truncation strategy (`"auto"` or `"disabled"`).
    public var truncation: String?
    /// The maximum number of output tokens to generate.
    public var maxTokens: Int?
    /// Configuration options for reasoning models.
    public var reasoning: Reasoning?
    /// Constrains the verbosity of the model's response (`"low"`, `"medium"`, `"high"`).
    public var verbosity: String?
    /// Metadata to include with the model response call.
    public var metadata: [String: String]?
    /// Whether to store the generated model response for later retrieval.
    public var store: Bool?
    /// The retention policy for the prompt cache (`"in_memory"` or `"24h"`).
    public var promptCacheRetention: String?
    /// Whether to include usage chunk. Only available for Chat Completions API.
    public var includeUsage: Bool?
    /// Additional output data to include in the model response.
    public var responseInclude: [String]?
    /// Number of top tokens to return logprobs for.
    public var topLogprobs: Int?
    /// Additional query fields to provide with the request.
    public var extraQuery: JSONObject?
    /// Additional body fields to provide with the request.
    public var extraBody: JSONObject?
    /// Additional headers to provide with the request. A `nil` value omits the header.
    public var extraHeaders: [String: String?]?
    /// Arbitrary keyword arguments to pass to the model API call.
    public var extraArgs: JSONObject?
    /// Opt-in runner-managed retry settings for model calls.
    public var retry: ModelRetrySettings?
    /// Context management entries for OpenAI Responses API requests.
    public var contextManagement: [JSONObject]?
    /// Prompt-cache configuration for OpenAI API requests.
    public var promptCacheOptions: JSONObject?
    /// Whether to preserve the provider usage payload on completed model responses.
    public var preserveRawUsage: Bool?
    /// Maximum duration in seconds for each model-call attempt.
    public var timeout: Double?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        frequencyPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        toolChoice: ToolChoice? = nil,
        parallelToolCalls: Bool? = nil,
        truncation: String? = nil,
        maxTokens: Int? = nil,
        reasoning: Reasoning? = nil,
        verbosity: String? = nil,
        metadata: [String: String]? = nil,
        store: Bool? = nil,
        promptCacheRetention: String? = nil,
        includeUsage: Bool? = nil,
        responseInclude: [String]? = nil,
        topLogprobs: Int? = nil,
        extraQuery: JSONObject? = nil,
        extraBody: JSONObject? = nil,
        extraHeaders: [String: String?]? = nil,
        extraArgs: JSONObject? = nil,
        retry: ModelRetrySettings? = nil,
        contextManagement: [JSONObject]? = nil,
        promptCacheOptions: JSONObject? = nil,
        preserveRawUsage: Bool? = nil,
        timeout: Double? = nil
    ) {
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
        self.promptCacheRetention = promptCacheRetention
        self.includeUsage = includeUsage
        self.responseInclude = responseInclude
        self.topLogprobs = topLogprobs
        self.extraQuery = extraQuery
        self.extraBody = extraBody
        self.extraHeaders = extraHeaders
        self.extraArgs = extraArgs
        self.retry = retry
        self.contextManagement = contextManagement
        self.promptCacheOptions = promptCacheOptions
        self.preserveRawUsage = preserveRawUsage
        self.timeout = timeout
    }

    /// Field names in declaration order, matching upstream `dataclasses.fields(ModelSettings)`.
    public static let fieldNames: [String] = [
        "temperature", "top_p", "frequency_penalty", "presence_penalty", "tool_choice",
        "parallel_tool_calls", "truncation", "max_tokens", "reasoning", "verbosity", "metadata",
        "store", "prompt_cache_retention", "include_usage", "response_include", "top_logprobs",
        "extra_query", "extra_body", "extra_headers", "extra_args", "retry",
        "context_management", "prompt_cache_options", "preserve_raw_usage", "timeout",
    ]

    static let traceableFieldNames: [String] = [
        "temperature", "top_p", "frequency_penalty", "presence_penalty", "tool_choice",
        "parallel_tool_calls", "truncation", "max_tokens", "reasoning", "verbosity", "metadata",
        "store", "prompt_cache_retention", "include_usage", "response_include", "top_logprobs",
        "retry", "context_management", "prompt_cache_options", "timeout",
    ]

    /// Produces new settings by overlaying any non-`nil` values from `override` on top of this
    /// instance. `extraArgs` dictionaries are merged and retry settings are deep-merged.
    public func resolve(_ override: ModelSettings?) -> ModelSettings {
        guard let override else { return self }
        var resolved = self
        resolved.temperature = override.temperature ?? temperature
        resolved.topP = override.topP ?? topP
        resolved.frequencyPenalty = override.frequencyPenalty ?? frequencyPenalty
        resolved.presencePenalty = override.presencePenalty ?? presencePenalty
        resolved.toolChoice = override.toolChoice ?? toolChoice
        resolved.parallelToolCalls = override.parallelToolCalls ?? parallelToolCalls
        resolved.truncation = override.truncation ?? truncation
        resolved.maxTokens = override.maxTokens ?? maxTokens
        resolved.reasoning = override.reasoning ?? reasoning
        resolved.verbosity = override.verbosity ?? verbosity
        resolved.metadata = override.metadata ?? metadata
        resolved.store = override.store ?? store
        resolved.promptCacheRetention = override.promptCacheRetention ?? promptCacheRetention
        resolved.includeUsage = override.includeUsage ?? includeUsage
        resolved.responseInclude = override.responseInclude ?? responseInclude
        resolved.topLogprobs = override.topLogprobs ?? topLogprobs
        resolved.extraQuery = override.extraQuery ?? extraQuery
        resolved.extraBody = override.extraBody ?? extraBody
        resolved.extraHeaders = override.extraHeaders ?? extraHeaders
        resolved.contextManagement = override.contextManagement ?? contextManagement
        resolved.promptCacheOptions = override.promptCacheOptions ?? promptCacheOptions
        resolved.preserveRawUsage = override.preserveRawUsage ?? preserveRawUsage
        resolved.timeout = override.timeout ?? timeout

        if extraArgs != nil || override.extraArgs != nil {
            var merged = JSONObject()
            if let extraArgs { merged.merge(extraArgs) }
            if let overrideArgs = override.extraArgs { merged.merge(overrideArgs) }
            resolved.extraArgs = merged.isEmpty ? nil : merged
        }
        if retry != nil || override.retry != nil {
            resolved.retry = ModelRetrySettings.merge(retry, override.retry)
        }
        return resolved
    }

    /// Overlays a dictionary of upstream field names, rejecting unknown fields.
    ///
    /// Swift: mirrors `ModelSettings.resolve(dict)`; `null` values do not override.
    public func resolve(_ override: JSONObject) throws -> ModelSettings {
        resolve(try ModelSettings(json: override))
    }

    /// Serializes the settings like `TypeAdapter(ModelSettings).dump_python(mode="json")`.
    public func toJSONDict() -> JSONObject {
        var result = JSONObject()
        for name in Self.fieldNames {
            result[name] = jsonValue(for: name)
        }
        return result
    }

    /// Serializes settings for tracing without provider-specific request extras.
    public func toTraceableDict() -> JSONObject {
        var result = JSONObject()
        for name in Self.traceableFieldNames {
            result[name] = jsonValue(for: name)
        }
        return result
    }

    // swiftlint:disable:next cyclomatic_complexity
    private func jsonValue(for name: String) -> JSONValue {
        switch name {
        case "temperature": return temperature.map(JSONValue.double) ?? .null
        case "top_p": return topP.map(JSONValue.double) ?? .null
        case "frequency_penalty": return frequencyPenalty.map(JSONValue.double) ?? .null
        case "presence_penalty": return presencePenalty.map(JSONValue.double) ?? .null
        case "tool_choice": return toolChoice?.json ?? .null
        case "parallel_tool_calls": return parallelToolCalls.map(JSONValue.bool) ?? .null
        case "truncation": return truncation.map(JSONValue.string) ?? .null
        case "max_tokens": return maxTokens.map(JSONValue.int) ?? .null
        case "reasoning": return reasoning.map { .object($0.dumped) } ?? .null
        case "verbosity": return verbosity.map(JSONValue.string) ?? .null
        case "metadata": return metadata.map { .object(JSONObject($0.mapValues(JSONValue.string))) } ?? .null
        case "store": return store.map(JSONValue.bool) ?? .null
        case "prompt_cache_retention": return promptCacheRetention.map(JSONValue.string) ?? .null
        case "include_usage": return includeUsage.map(JSONValue.bool) ?? .null
        case "response_include": return responseInclude.map { .array($0.map(JSONValue.string)) } ?? .null
        case "top_logprobs": return topLogprobs.map(JSONValue.int) ?? .null
        case "extra_query": return extraQuery.map(JSONValue.object) ?? .null
        case "extra_body": return extraBody.map(JSONValue.object) ?? .null
        case "extra_headers":
            return extraHeaders.map { headers in
                .object(JSONObject(headers.mapValues { $0.map(JSONValue.string) ?? .null }))
            } ?? .null
        case "extra_args": return extraArgs.map(JSONValue.object) ?? .null
        case "retry": return retry.map { .object($0.toJSONDict()) } ?? .null
        case "context_management": return contextManagement.map { .array($0.map(JSONValue.object)) } ?? .null
        case "prompt_cache_options": return promptCacheOptions.map(JSONValue.object) ?? .null
        case "preserve_raw_usage": return preserveRawUsage.map(JSONValue.bool) ?? .null
        case "timeout": return timeout.map(JSONValue.double) ?? .null
        default: return .null
        }
    }

    /// Fields of the OpenAI `ContextManagement` and `PromptCacheOptions` request types.
    static let contextManagementFields: Set<String> = ["type", "compact_threshold"]
    static let promptCacheOptionFields: Set<String> = ["mode", "ttl"]

    private static func validateFields(_ value: JSONObject, allowed: Set<String>, label: String) throws {
        let unknown = value.keys.filter { !allowed.contains($0) }.sorted()
        if !unknown.isEmpty {
            throw TypeError("Unknown model settings in \(label): \(unknown.joined(separator: ", "))")
        }
    }

    /// Deserializes settings from their JSON dictionary form. Unknown fields raise `TypeError`.
    public init(json: JSONObject) throws {
        let unknown = json.keys.filter { !Self.fieldNames.contains($0) }.sorted()
        if !unknown.isEmpty {
            throw TypeError("Unknown model settings: \(unknown.joined(separator: ", "))")
        }
        self.init()
        temperature = json["temperature"]?.doubleValue
        topP = json["top_p"]?.doubleValue
        frequencyPenalty = json["frequency_penalty"]?.doubleValue
        presencePenalty = json["presence_penalty"]?.doubleValue
        if let choice = json["tool_choice"], !choice.isNull {
            if case .object(let object) = choice {
                let unknownChoice = object.keys.filter { !["server_label", "name"].contains($0) }.sorted()
                if !unknownChoice.isEmpty {
                    throw TypeError("Unknown model settings in tool_choice: \(unknownChoice.joined(separator: ", "))")
                }
            }
            toolChoice = ToolChoice(json: choice)
        }
        parallelToolCalls = json["parallel_tool_calls"]?.boolValue
        truncation = json["truncation"]?.stringValue
        maxTokens = json["max_tokens"]?.intValue
        if case .object(let object)? = json["reasoning"] {
            reasoning = Reasoning(json: object.filter { _, value in !value.isNull })
        }
        verbosity = json["verbosity"]?.stringValue
        if case .object(let object)? = json["metadata"] {
            metadata = object.dictionary.compactMapValues(\.stringValue)
        }
        store = json["store"]?.boolValue
        promptCacheRetention = json["prompt_cache_retention"]?.stringValue
        includeUsage = json["include_usage"]?.boolValue
        responseInclude = json["response_include"]?.arrayValue?.compactMap(\.stringValue)
        topLogprobs = json["top_logprobs"]?.intValue
        extraQuery = json["extra_query"]?.objectValue
        extraBody = json["extra_body"]?.objectValue
        if case .object(let object)? = json["extra_headers"] {
            extraHeaders = object.dictionary.mapValues(\.stringValue)
        }
        extraArgs = json["extra_args"]?.objectValue
        if case .object(let retryObject)? = json["retry"] {
            let unknownRetry = retryObject.keys.filter { !["max_retries", "backoff", "policy"].contains($0) }.sorted()
            if !unknownRetry.isEmpty {
                throw TypeError("Unknown model settings in retry: \(unknownRetry.joined(separator: ", "))")
            }
            var backoff: ModelRetryBackoffSettings?
            if case .object(let backoffObject)? = retryObject["backoff"] {
                let allowed = ["initial_delay", "max_delay", "multiplier", "jitter"]
                let unknownBackoff = backoffObject.keys.filter { !allowed.contains($0) }.sorted()
                if !unknownBackoff.isEmpty {
                    throw TypeError("Unknown model settings in retry.backoff: \(unknownBackoff.joined(separator: ", "))")
                }
                backoff = ModelRetryBackoffSettings(
                    initialDelay: backoffObject["initial_delay"]?.doubleValue,
                    maxDelay: backoffObject["max_delay"]?.doubleValue,
                    multiplier: backoffObject["multiplier"]?.doubleValue,
                    jitter: backoffObject["jitter"]?.boolValue
                )
            }
            retry = ModelRetrySettings(maxRetries: retryObject["max_retries"]?.intValue, backoff: backoff)
        }
        contextManagement = json["context_management"]?.arrayValue?.compactMap(\.objectValue)
        for (index, entry) in (contextManagement ?? []).enumerated() {
            try Self.validateFields(entry, allowed: Self.contextManagementFields, label: "context_management[\(index)]")
        }
        promptCacheOptions = json["prompt_cache_options"]?.objectValue
        if let promptCacheOptions {
            try Self.validateFields(promptCacheOptions, allowed: Self.promptCacheOptionFields, label: "prompt_cache_options")
        }
        preserveRawUsage = json["preserve_raw_usage"]?.boolValue
        timeout = json["timeout"]?.doubleValue
        if let timeout, !(timeout > 0 && timeout.isFinite) {
            throw ValueError("timeout must be a finite number greater than 0")
        }
    }
}
