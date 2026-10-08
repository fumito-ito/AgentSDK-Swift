import Foundation

// Port of src/agents/models/_openai_shared.py, _openai_retry.py, and _response_terminal.py.

/// The transport used for the OpenAI Responses API.
public enum OpenAIResponsesTransport: String, Sendable {
    case http
    case websocket
}

struct OpenAIDefaults {
    var openAIKey: String?
    var openAIClient: OpenAIClient?
    var useResponsesByDefault = true
    var responsesTransport: OpenAIResponsesTransport = .http
}

let openAIDefaults = SharedBox(OpenAIDefaults())

/// Sets the default OpenAI API key used for requests and tracing export.
public func setDefaultOpenAIKey(_ key: String, useForTracing: Bool = true) {
    openAIDefaults.mutate { $0.openAIKey = key }
    if useForTracing {
        setTracingExportAPIKey(key)
    }
}

func getDefaultOpenAIKey() -> String? { openAIDefaults.value.openAIKey }

/// Sets the default OpenAI client used for requests (and tracing export when requested).
public func setDefaultOpenAIClient(_ client: OpenAIClient, useForTracing: Bool = true) {
    openAIDefaults.mutate { $0.openAIClient = client }
    if useForTracing, let key = client.apiKey {
        setTracingExportAPIKey(key)
    }
}

func getDefaultOpenAIClient() -> OpenAIClient? { openAIDefaults.value.openAIClient }

/// Sets whether the OpenAI provider uses the Responses API (`true`) or Chat Completions by default.
public func setDefaultOpenAIAPI(_ api: String) {
    openAIDefaults.mutate { $0.useResponsesByDefault = api != "chat_completions" }
}

func setUseResponsesByDefault(_ useResponses: Bool) {
    openAIDefaults.mutate { $0.useResponsesByDefault = useResponses }
}

func getUseResponsesByDefault() -> Bool { openAIDefaults.value.useResponsesByDefault }

/// Sets the default transport for the OpenAI Responses API.
public func setDefaultOpenAIResponsesTransport(_ transport: OpenAIResponsesTransport) {
    openAIDefaults.mutate { $0.responsesTransport = transport }
}

func getDefaultOpenAIResponsesTransport() -> OpenAIResponsesTransport { openAIDefaults.value.responsesTransport }

// MARK: - Retry advice (_openai_retry.py)

private func errorDescription(_ error: any Error) -> String {
    describeError(error)
}

/// OpenAI-specific retry guidance for a failed request.
func getOpenAIRetryAdvice(_ request: ModelRetryAdviceRequest) -> ModelRetryAdvice? {
    let error = request.error
    let message = errorDescription(error)
    if message.lowercased().contains("the request may have been accepted, so the sdk will not automatically retry this websocket request.") {
        return ModelRetryAdvice(suggested: false, replaySafety: "unsafe", reason: message)
    }
    let apiError = error as? OpenAIAPIError
    let retryAfter = apiError?.retryAfter
    let normalized = ModelRetryNormalizedError(
        statusCode: apiError?.statusCode,
        errorCode: apiError?.errorCode,
        message: message,
        requestId: apiError?.requestId,
        retryAfter: retryAfter,
        isAbort: false,
        isNetworkError: apiError?.isConnectionError ?? false,
        isTimeout: apiError?.isTimeout ?? false
    )
    let statefulRequest = !(request.previousResponseId ?? "").isEmpty || !(request.conversationId ?? "").isEmpty
    if let header = apiError?.shouldRetryHeader?.lowercased().trimmingCharacters(in: .whitespaces) {
        if header == "true" {
            return ModelRetryAdvice(suggested: true, retryAfter: retryAfter, replaySafety: "safe", reason: message, normalized: normalized)
        }
        if header == "false" {
            return ModelRetryAdvice(suggested: false, retryAfter: retryAfter, reason: message, normalized: normalized)
        }
    }
    if normalized.isNetworkError || normalized.isTimeout {
        return ModelRetryAdvice(suggested: true, retryAfter: retryAfter, reason: message, normalized: normalized)
    }
    if let status = normalized.statusCode, [408, 409, 429].contains(status) || status >= 500 {
        var advice = ModelRetryAdvice(suggested: true, retryAfter: retryAfter, reason: message, normalized: normalized)
        if statefulRequest { advice.replaySafety = "safe" }
        return advice
    }
    if let retryAfter {
        return ModelRetryAdvice(retryAfter: retryAfter, reason: message, normalized: normalized)
    }
    return nil
}

// MARK: - Terminal failures (_response_terminal.py)

private func reprJSON(_ value: JSONValue) -> String {
    value.stringValue ?? value.jsonString()
}

func formatResponseTerminalFailure(_ eventType: String, _ response: ResponseObject?) -> String {
    var message = "Responses stream ended with terminal event `\(eventType)`."
    guard let response else { return message }
    var details: [String] = []
    if let status = response.status, !status.isEmpty { details.append("status=\(status)") }
    if let error = response.json["error"], !error.isNull { details.append("error=\(reprJSON(error))") }
    if let incomplete = response.json["incomplete_details"], !incomplete.isNull { details.append("incomplete_details=\(reprJSON(incomplete))") }
    if !details.isEmpty { message += " \(details.joined(separator: "; "))." }
    return message
}

func formatResponseErrorEvent(_ eventType: String, _ event: JSONObject) -> String {
    var message = "Responses stream ended with terminal event `\(eventType)`."
    var details: [String] = []
    if let code = event["code"], !code.isNull, code.isTruthy { details.append("code=\(reprJSON(code))") }
    if let error = event.string("message"), !error.isEmpty { details.append("message=\(error)") }
    if let param = event["param"], !param.isNull, param.isTruthy { details.append("param=\(reprJSON(param))") }
    if !details.isEmpty { message += " \(details.joined(separator: "; "))." }
    return message
}

/// The error raised for a terminal `response.incomplete` / `response.failed` event.
func responseTerminalFailureError(_ eventType: String, _ response: ResponseObject?) -> ModelBehaviorError {
    ModelBehaviorError(formatResponseTerminalFailure(eventType, response))
}

/// The error raised for an `error` / `response.error` stream event.
func responseErrorEventFailureError(_ eventType: String, _ event: TResponseStreamEvent) -> ModelBehaviorError {
    ModelBehaviorError(formatResponseErrorEvent(eventType, event))
}
