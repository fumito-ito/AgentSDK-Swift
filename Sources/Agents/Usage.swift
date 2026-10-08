import Foundation

/// Details about input tokens, matching the Responses API `input_tokens_details` shape.
public struct InputTokensDetails: Sendable, Equatable, Hashable {
    /// Tokens served from the prompt cache.
    public var cachedTokens: Int
    /// Tokens written to the prompt cache.
    public var cacheWriteTokens: Int

    public init(cachedTokens: Int = 0, cacheWriteTokens: Int = 0) {
        self.cachedTokens = cachedTokens
        self.cacheWriteTokens = cacheWriteTokens
    }

    /// Normalizes Responses or Chat Completions token details (`prompt_tokens_details`) into the
    /// Responses shape. Missing or `null` values become zero.
    public init(chatCompletionsPromptDetails details: JSONObject?) {
        self.init(json: details)
    }

    /// Normalizes a JSON payload (`{"cached_tokens": .., "cache_write_tokens": ..}`). Missing or
    /// `null` values become zero.
    public init(json: JSONObject?) {
        self.cachedTokens = json?["cached_tokens"]?.intValue ?? 0
        self.cacheWriteTokens = json?["cache_write_tokens"]?.intValue ?? 0
    }

    public var json: JSONObject {
        ["cached_tokens": .int(cachedTokens), "cache_write_tokens": .int(cacheWriteTokens)]
    }
}

/// Details about output tokens, matching the Responses API `output_tokens_details` shape.
public struct OutputTokensDetails: Sendable, Equatable, Hashable {
    /// Tokens used for reasoning.
    public var reasoningTokens: Int

    public init(reasoningTokens: Int = 0) {
        self.reasoningTokens = reasoningTokens
    }

    /// Normalizes Chat Completions `completion_tokens_details` into the Responses shape.
    public init(chatCompletionsCompletionDetails details: JSONObject?) {
        self.init(json: details)
    }

    /// Normalizes a JSON payload. Missing or `null` values become zero.
    public init(json: JSONObject?) {
        self.reasoningTokens = json?["reasoning_tokens"]?.intValue ?? 0
    }

    public var json: JSONObject {
        ["reasoning_tokens": .int(reasoningTokens)]
    }
}

/// Usage details for a single API request.
public struct RequestUsage: Sendable, Equatable, Hashable {
    /// Input tokens for this individual request.
    public var inputTokens: Int
    /// Output tokens for this individual request.
    public var outputTokens: Int
    /// Total tokens (input + output) for this individual request.
    public var totalTokens: Int
    /// Details about the input tokens for this individual request.
    public var inputTokensDetails: InputTokensDetails
    /// Details about the output tokens for this individual request.
    public var outputTokensDetails: OutputTokensDetails

    public init(
        inputTokens: Int,
        outputTokens: Int,
        totalTokens: Int,
        inputTokensDetails: InputTokensDetails = InputTokensDetails(),
        outputTokensDetails: OutputTokensDetails = OutputTokensDetails()
    ) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.totalTokens = totalTokens
        self.inputTokensDetails = inputTokensDetails
        self.outputTokensDetails = outputTokensDetails
    }
}

/// Aggregated token usage across one or more model requests.
public struct Usage: Sendable, Equatable, Hashable {
    /// Total requests made to the LLM API.
    public var requests: Int
    /// Total input tokens sent, across all requests.
    public var inputTokens: Int
    /// Details about the input tokens, matching responses API usage details.
    public var inputTokensDetails: InputTokensDetails
    /// Total output tokens received, across all requests.
    public var outputTokens: Int
    /// Details about the output tokens, matching responses API usage details.
    public var outputTokensDetails: OutputTokensDetails
    /// Total tokens sent and received, across all requests.
    public var totalTokens: Int
    /// Per-request breakdown, for accurate per-request cost calculation.
    ///
    /// Each call to `add(_:)` automatically creates an entry when the added usage represents a
    /// single request with non-zero tokens.
    public var requestUsageEntries: [RequestUsage]

    public init(
        requests: Int = 0,
        inputTokens: Int = 0,
        inputTokensDetails: InputTokensDetails = InputTokensDetails(),
        outputTokens: Int = 0,
        outputTokensDetails: OutputTokensDetails = OutputTokensDetails(),
        totalTokens: Int = 0,
        requestUsageEntries: [RequestUsage] = []
    ) {
        self.requests = requests
        self.inputTokens = inputTokens
        self.inputTokensDetails = inputTokensDetails
        self.outputTokens = outputTokens
        self.outputTokensDetails = outputTokensDetails
        self.totalTokens = totalTokens
        self.requestUsageEntries = requestUsageEntries
    }

    /// Adds another usage to this one, aggregating all fields and preserving per-request entries.
    public mutating func add(_ other: Usage) {
        requests += other.requests
        inputTokens += other.inputTokens
        outputTokens += other.outputTokens
        totalTokens += other.totalTokens
        inputTokensDetails = InputTokensDetails(
            cachedTokens: inputTokensDetails.cachedTokens + other.inputTokensDetails.cachedTokens,
            cacheWriteTokens: inputTokensDetails.cacheWriteTokens + other.inputTokensDetails.cacheWriteTokens
        )
        outputTokensDetails = OutputTokensDetails(
            reasoningTokens: outputTokensDetails.reasoningTokens + other.outputTokensDetails.reasoningTokens
        )

        // Existing per-request breakdowns carry authoritative nested details, so they win over an
        // entry synthesized from the top-level fields.
        if !other.requestUsageEntries.isEmpty {
            requestUsageEntries.append(contentsOf: other.requestUsageEntries)
        } else if other.requests == 1,
                  other.inputTokens > 0 || other.outputTokens > 0 || other.totalTokens > 0 {
            requestUsageEntries.append(
                RequestUsage(
                    inputTokens: other.inputTokens,
                    outputTokens: other.outputTokens,
                    totalTokens: other.totalTokens,
                    inputTokensDetails: other.inputTokensDetails,
                    outputTokensDetails: other.outputTokensDetails
                )
            )
        }
    }
}

// MARK: - Raw usage snapshots

/// Returns a detached JSON-compatible usage object without adding omitted fields, or `nil` when the
/// payload is not a JSON object.
public func rawUsageSnapshot(_ rawUsage: JSONValue?) -> JSONObject? {
    guard case .object(let object)? = rawUsage else { return nil }
    return object
}

// MARK: - Serialization

private func coerceInputTokenDetails(_ raw: JSONValue?) -> InputTokensDetails {
    var candidate = raw
    if case .array(let array)? = candidate, let first = array.first {
        candidate = first
    }
    return InputTokensDetails(json: candidate?.objectValue)
}

private func coerceOutputTokenDetails(_ raw: JSONValue?) -> OutputTokensDetails {
    var candidate = raw
    if case .array(let array)? = candidate, let first = array.first {
        candidate = first
    }
    return OutputTokensDetails(json: candidate?.objectValue)
}

/// Rebuilds a `Usage` from serialized JSON data.
public func deserializeUsage(_ usageData: JSONObject) -> Usage {
    let entries: [RequestUsage] = (usageData["request_usage_entries"]?.arrayValue ?? []).compactMap { raw in
        guard let entry = raw.objectValue else { return nil }
        return RequestUsage(
            inputTokens: entry["input_tokens"]?.intValue ?? 0,
            outputTokens: entry["output_tokens"]?.intValue ?? 0,
            totalTokens: entry["total_tokens"]?.intValue ?? 0,
            inputTokensDetails: coerceInputTokenDetails(entry["input_tokens_details"]),
            outputTokensDetails: coerceOutputTokenDetails(entry["output_tokens_details"])
        )
    }
    return Usage(
        requests: usageData["requests"]?.intValue ?? 0,
        inputTokens: usageData["input_tokens"]?.intValue ?? 0,
        inputTokensDetails: coerceInputTokenDetails(usageData["input_tokens_details"]),
        outputTokens: usageData["output_tokens"]?.intValue ?? 0,
        outputTokensDetails: coerceOutputTokenDetails(usageData["output_tokens_details"]),
        totalTokens: usageData["total_tokens"]?.intValue ?? 0,
        requestUsageEntries: entries
    )
}

/// Serializes a `Usage` into a JSON-friendly dictionary.
public func serializeUsage(_ usage: Usage) -> JSONObject {
    let entries: [JSONValue] = usage.requestUsageEntries.map { entry in
        .object([
            "input_tokens": .int(entry.inputTokens),
            "output_tokens": .int(entry.outputTokens),
            "total_tokens": .int(entry.totalTokens),
            "input_tokens_details": .object(entry.inputTokensDetails.json),
            "output_tokens_details": .object(entry.outputTokensDetails.json),
        ])
    }
    return [
        "requests": .int(usage.requests),
        "input_tokens": .int(usage.inputTokens),
        "input_tokens_details": [.object(usage.inputTokensDetails.json)],
        "output_tokens": .int(usage.outputTokens),
        "output_tokens_details": [.object(usage.outputTokensDetails.json)],
        "total_tokens": .int(usage.totalTokens),
        "request_usage_entries": .array(entries),
    ]
}

/// Serializes full per-model-call usage for tracing span data.
public func modelUsageToSpanUsage(_ usage: Usage) -> JSONObject {
    [
        "requests": .int(usage.requests),
        "input_tokens": .int(usage.inputTokens),
        "output_tokens": .int(usage.outputTokens),
        "total_tokens": .int(usage.totalTokens),
        "input_tokens_details": .object(usage.inputTokensDetails.json),
        "output_tokens_details": .object(usage.outputTokensDetails.json),
    ]
}

/// Serializes aggregate task/run usage for tracing span metadata.
public func totalUsageToSpanMetadata(_ usage: Usage) -> JSONObject {
    [
        "requests": .int(usage.requests),
        "input_tokens": .int(usage.inputTokens),
        "output_tokens": .int(usage.outputTokens),
        "total_tokens": .int(usage.totalTokens),
        "cached_input_tokens": .int(usage.inputTokensDetails.cachedTokens),
        "cache_write_input_tokens": .int(usage.inputTokensDetails.cacheWriteTokens),
    ]
}

/// Serializes aggregate per-turn usage for custom turn span data.
public func turnUsageToSpanData(_ usage: Usage) -> JSONObject {
    [
        "input_tokens": .int(usage.inputTokens),
        "output_tokens": .int(usage.outputTokens),
        "cached_input_tokens": .int(usage.inputTokensDetails.cachedTokens),
        "cache_write_input_tokens": .int(usage.inputTokensDetails.cacheWriteTokens),
    ]
}

/// Serializes aggregate per-task usage for custom task span data.
public func taskUsageToSpanData(_ usage: Usage) -> JSONObject {
    var data = turnUsageToSpanData(usage)
    data["requests"] = .int(usage.requests)
    data["total_tokens"] = .int(usage.totalTokens)
    return data
}

/// Converts a Responses API `usage` payload into a `Usage` value for one request.
public func responseUsageToUsage(_ responseUsage: JSONObject?, requests: Int = 1) -> Usage {
    guard let responseUsage else { return Usage(requests: requests) }
    return Usage(
        requests: requests,
        inputTokens: responseUsage["input_tokens"]?.intValue ?? 0,
        inputTokensDetails: InputTokensDetails(json: responseUsage["input_tokens_details"]?.objectValue),
        outputTokens: responseUsage["output_tokens"]?.intValue ?? 0,
        outputTokensDetails: OutputTokensDetails(json: responseUsage["output_tokens_details"]?.objectValue),
        totalTokens: responseUsage["total_tokens"]?.intValue ?? 0
    )
}

/// Converts a `Usage` value into a Responses API `usage` payload.
public func usageToResponseUsage(_ usage: Usage) -> JSONObject {
    [
        "input_tokens": .int(usage.inputTokens),
        "output_tokens": .int(usage.outputTokens),
        "total_tokens": .int(usage.totalTokens),
        "input_tokens_details": .object(usage.inputTokensDetails.json),
        "output_tokens_details": .object(usage.outputTokensDetails.json),
    ]
}

/// The key under which adapters attach a lossless normalized `Usage` to a Responses `usage`
/// payload (upstream stores it as a private attribute on the pydantic object).
let normalizedUsageKey = "_agents_sdk_normalized_usage"
let requestUsagesKey = "_agents_sdk_request_usages"
let requestCountKey = "_agents_sdk_request_count"
let requestWithoutUsageKey = "_agents_sdk_request_without_usage"

/// Attaches a normalized usage snapshot for lossless internal conversion.
func attachNormalizedUsage(_ target: inout JSONObject, _ usage: Usage) {
    target[normalizedUsageKey] = .object(serializeUsage(usage))
}

/// Converts Responses API usage, including adapter-supplied per-request details.
func responseUsageToUsageWithMetadata(_ responseUsage: JSONObject) -> Usage {
    if let normalized = responseUsage[normalizedUsageKey]?.objectValue {
        return deserializeUsage(normalized)
    }
    let requestCount = responseUsage[requestCountKey]?.intValue ?? 1
    if let requestUsages = responseUsage[requestUsagesKey]?.arrayValue {
        var usage = Usage()
        for case .object(let entry) in requestUsages {
            usage.add(responseUsageToUsage(entry, requests: 1))
        }
        usage.requests = max(usage.requests, requestCount)
        return usage
    }
    return responseUsageToUsage(responseUsage, requests: requestCount)
}

/// How many requests a usage-less response represents. Defaults to zero so adapters must opt in.
func requestsForResponseWithoutUsage(_ response: JSONObject) -> Int {
    switch response[requestWithoutUsageKey] {
    case .bool(true)?: return 1
    case .int(let count)? where count > 0: return count
    default: return 0
    }
}

/// Strips SDK-internal usage metadata keys before exposing a usage payload.
func strippingInternalUsageMetadata(_ usage: JSONObject) -> JSONObject {
    usage.removing(normalizedUsageKey, requestUsagesKey, requestCountKey)
}
