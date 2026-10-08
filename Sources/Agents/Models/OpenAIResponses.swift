import Foundation

// Port of src/agents/models/openai_responses.py.
//
// Swift: the websocket transport (`OpenAIResponsesWSModel`) is not ported; see
// porting/manifest.json.

let agentsUserAgent = "Agents/Swift \(agentsSDKVersion)"
private let defaultHeaders: [String: String?] = ["User-Agent": agentsUserAgent]

/// Per-task header overrides applied to OpenAI requests (upstream `_HEADERS_OVERRIDE`).
public enum OpenAIHeadersOverride {
    @TaskLocal public static var headers: [String: String?]?
}

/// A `Model` backed by the OpenAI Responses API.
public final class OpenAIResponsesModel: Model, @unchecked Sendable {
    public let model: String
    let modelIsExplicit: Bool
    private let client: OpenAIClient

    public init(model: String, openAIClient: OpenAIClient, modelIsExplicit: Bool = true) {
        self.model = model
        self.client = openAIClient
        self.modelIsExplicit = modelIsExplicit
    }

    public func getRetryAdvice(_ request: ModelRetryAdviceRequest) -> ModelRetryAdvice? {
        getOpenAIRetryAdvice(request)
    }

    public func getResponse(_ request: ModelRequest) async throws -> ModelResponse {
        let span = responseSpan(disabled: request.tracing.isDisabled())
        return try await span.run {
            do {
                let body = try buildResponseCreateBody(request, stream: false)
                let (json, requestId) = try await client.post(
                    path: "responses", body: body.body, extraHeaders: body.headers, extraQuery: request.modelSettings.extraQuery
                )
                guard let response = ResponseObject(json: json) else {
                    throw ModelBehaviorError("Invalid Responses API payload")
                }
                if let status = response.status, status == "failed" || status == "incomplete" {
                    throw responseTerminalFailureError("response.\(status)", response)
                }
                if AgentsDebug.dontLogModelData {
                    logger.debug("LLM responded")
                } else {
                    logger.debug("LLM resp:\n\(JSONValue.array(response.output.map(JSONValue.object)).jsonString(prettyPrinted: true))\n")
                }
                // A completed response without usage still represents one adapter-owned request.
                let usage = response.usage.map(responseUsageToUsageWithMetadata) ?? Usage(requests: 1)
                if let data = span.data(as: ResponseSpanData.self) {
                    if response.usage != nil || usage.requests > 0 {
                        data.usage = modelUsageToSpanUsage(usage)
                    }
                    if request.tracing.includeData() {
                        data.response = response
                        data.input = request.input
                    } else if !request.tracing.isDisabled(), client.isOfficialOpenAIEndpoint {
                        data.responseIdOverride = response.id
                    }
                }
                return ModelResponse(
                    output: response.output,
                    usage: usage,
                    responseId: response.id,
                    requestId: requestId,
                    rawUsage: request.modelSettings.preserveRawUsage == true ? response.usage : nil
                )
            } catch {
                span.setError(SpanError(
                    message: "Error getting response",
                    data: ["error": .string(request.tracing.includeData() ? describeError(error) : redactedTraceErrorMessage)]
                ))
                var message = "Error getting response"
                if !AgentsDebug.dontLogModelData {
                    message += " (request_id: \((error as? OpenAIAPIError)?.requestId ?? "None"))"
                }
                logActionError(message, error, isToolAction: false)
                throw error
            }
        }
    }

    public func streamResponse(_ request: ModelRequest) -> AsyncThrowingStream<TResponseStreamEvent, any Error> {
        let span = responseSpan(disabled: request.tracing.isDisabled())
        let client = self.client
        let bodyResult = Result { try buildResponseCreateBody(request, stream: true) }
        return AsyncThrowingStream { continuation in
            let task = Task {
                span.start()
                defer { span.finish() }
                do {
                    let body = try bodyResult.get()
                    var terminalFailure: ModelBehaviorError?
                    let events = client.stream(path: "responses", body: body.body, extraHeaders: body.headers, extraQuery: request.modelSettings.extraQuery)
                    for try await chunk in events {
                        var event = chunk
                        let type = event.string("type")
                        switch type {
                        case "response.completed":
                            if var response = event["response"]?.objectValue {
                                let responseObject = ResponseObject(json: response)
                                let usage = responseObject?.usage.map(responseUsageToUsageWithMetadata) ?? Usage()
                                if let data = span.data(as: ResponseSpanData.self) {
                                    if responseObject?.usage != nil || usage.requests > 0 {
                                        data.usage = modelUsageToSpanUsage(usage)
                                    }
                                    if request.tracing.includeData() {
                                        data.response = responseObject
                                        data.input = request.input
                                    } else if !request.tracing.isDisabled(), client.isOfficialOpenAIEndpoint {
                                        data.responseIdOverride = responseObject?.id
                                    }
                                }
                                if responseObject?.usage == nil {
                                    response[requestWithoutUsageKey] = true
                                }
                                if request.modelSettings.preserveRawUsage == true {
                                    attachRawUsageSnapshot(&response, responseObject?.usage)
                                }
                                event["response"] = .object(response)
                            }
                        case "response.failed", "response.incomplete":
                            terminalFailure = responseTerminalFailureError(type ?? "", event["response"]?.objectValue.flatMap(ResponseObject.init(json:)))
                        case "error", "response.error":
                            terminalFailure = responseErrorEventFailureError(type ?? "", event)
                        default:
                            break
                        }
                        if let terminalFailure {
                            recordModelErrorOnSpan(span, message: "Error streaming response", error: terminalFailure, traceIncludeSensitiveData: request.tracing.includeData())
                        }
                        continuation.yield(event)
                        if let terminalFailure {
                            throw terminalFailure
                        }
                    }
                    continuation.finish()
                } catch {
                    if !(error is CancellationError) {
                        span.setError(SpanError(
                            message: "Error streaming response",
                            data: ["error": .string(request.tracing.includeData() ? describeError(error) : redactedTraceErrorMessage)]
                        ))
                        logActionError("Error streaming response", error, isToolAction: false)
                    }
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: Request building

    struct RequestBody {
        var body: JSONObject
        var headers: [String: String?]
    }

    func buildResponseCreateBody(_ request: ModelRequest, stream: Bool) throws -> RequestBody {
        let settings = request.modelSettings
        var listInput = ItemHelpers.inputToNewInputList(request.input)
        listInput = removeOpenAIResponsesAPIIncompatibleFields(listInput)

        let shouldOmitModel = request.prompt != nil && !modelIsExplicit
        let effectiveModel: String? = shouldOmitModel ? nil : model
        let computerToolModel = ResponsesConverter.resolveComputerToolModel(requestModel: effectiveModel, tools: request.tools)
        let toolChoice = try ResponsesConverter.convertToolChoice(settings.toolChoice, tools: request.tools, handoffs: request.handoffs, model: computerToolModel)
        let converted = try ResponsesConverter.convertTools(
            request.tools, request.handoffs,
            allowOpaqueToolSearchSurface: request.prompt != nil,
            model: computerToolModel,
            toolChoice: settings.toolChoice
        )
        let shouldOmitTools = request.prompt != nil && converted.tools.isEmpty
        var responseFormat = try ResponsesConverter.getResponseFormat(request.outputSchema)

        var include = converted.includes
        for value in settings.responseInclude ?? [] where !include.contains(value) { include.append(value) }
        if settings.topLogprobs != nil, !include.contains("message.output_text.logprobs") {
            include.append("message.output_text.logprobs")
        }

        if AgentsDebug.dontLogModelData {
            logger.debug("Calling LLM")
        } else {
            logger.debug(
                "Calling LLM \(model) with input:\n\(JSONValue.array(listInput.map(JSONValue.object)).jsonString(prettyPrinted: true))\n"
                    + "Tools:\n\(JSONValue.array(converted.tools.map(JSONValue.object)).jsonString(prettyPrinted: true))\n"
                    + "Stream: \(stream)\nTool choice: \(toolChoice.map { $0.jsonString() } ?? "NOT_GIVEN")\n"
                    + "Response format: \(responseFormat.map { JSONValue.object($0).jsonString() } ?? "NOT_GIVEN")\n"
                    + "Previous response id: \(request.previousResponseId ?? "None")\nConversation id: \(request.conversationId ?? "None")\n"
            )
        }

        var extraArgs = settings.extraArgs ?? [:]
        if let topLogprobs = settings.topLogprobs { extraArgs["top_logprobs"] = .int(topLogprobs) }
        if let verbosity = settings.verbosity {
            if responseFormat != nil {
                responseFormat?["verbosity"] = .string(verbosity)
            } else {
                responseFormat = ["verbosity": .string(verbosity)]
            }
        }

        var body: JSONObject = [:]
        func set(_ key: String, _ value: JSONValue?) {
            if let value { body[key] = value }
        }
        set("previous_response_id", request.previousResponseId.map(JSONValue.string))
        set("conversation", request.conversationId.map(JSONValue.string))
        set("instructions", request.systemInstructions.map(JSONValue.string))
        set("model", effectiveModel.map(JSONValue.string))
        body["input"] = .array(listInput.map(JSONValue.object))
        body["include"] = .array(include.map(JSONValue.string))
        if !shouldOmitTools { body["tools"] = .array(converted.tools.map(JSONValue.object)) }
        set("prompt", request.prompt.map(JSONValue.object))
        set("temperature", settings.temperature.map(JSONValue.double))
        set("top_p", settings.topP.map(JSONValue.double))
        set("truncation", settings.truncation.map(JSONValue.string))
        set("max_output_tokens", settings.maxTokens.map(JSONValue.int))
        if let toolChoice, !(shouldOmitTools && toolChoice.objectValue != nil) {
            body["tool_choice"] = toolChoice
        }
        if request.prompt != nil || !converted.tools.isEmpty {
            set("parallel_tool_calls", settings.parallelToolCalls.map(JSONValue.bool))
        }
        if stream { body["stream"] = true }
        set("text", responseFormat.map(JSONValue.object))
        set("store", settings.store.map(JSONValue.bool))
        set("prompt_cache_retention", settings.promptCacheRetention.map(JSONValue.string))
        set("prompt_cache_options", settings.promptCacheOptions.map(JSONValue.object))
        set("reasoning", settings.reasoning.map { .object($0.json) })
        set("metadata", settings.metadata.map { metadata in
            .object(JSONObject(metadata.sorted { $0.key < $1.key }.map { ($0.key, JSONValue.string($0.value)) }))
        })
        set("context_management", settings.contextManagement.map { .array($0.map(JSONValue.object)) })

        let duplicates = extraArgs.keys.filter { body.contains($0) }.sorted()
        if duplicates.count == 1 {
            throw TypeError("responses.create() got multiple values for keyword argument '\(duplicates[0])'")
        } else if duplicates.count > 1 {
            throw TypeError("responses.create() got multiple values for keyword arguments \(duplicates.map { "'\($0)'" }.joined(separator: ", "))")
        }
        body.merge(extraArgs)
        if let extraBody = settings.extraBody { body.merge(extraBody) }
        return RequestBody(body: body, headers: mergeHeaders(settings))
    }

    private func mergeHeaders(_ settings: ModelSettings) -> [String: String?] {
        var headers = defaultHeaders
        for (key, value) in settings.extraHeaders ?? [:] { headers[key] = value }
        for (key, value) in OpenAIHeadersOverride.headers ?? [:] { headers[key] = value }
        return headers
    }

    /// Removes input fields the Responses API rejects: provider data, fake IDs, and provider
    /// reasoning items.
    func removeOpenAIResponsesAPIIncompatibleFields(_ listInput: [TResponseInputItem]) -> [TResponseInputItem] {
        let needsCleaning = listInput.contains { $0["provider_data"]?.isTruthy == true || $0.string("id") == fakeResponsesId }
        guard needsCleaning else { return listInput }
        return listInput.compactMap { item in
            if item.string("type") == "reasoning", item["provider_data"]?.isTruthy == true { return nil }
            var cleaned = item
            if cleaned.string("id") == fakeResponsesId { cleaned.removeValue(forKey: "id") }
            cleaned.removeValue(forKey: "provider_data")
            return cleaned
        }
    }
}

/// Converted tools plus the `include` values they require.
struct ConvertedTools {
    var tools: [JSONObject]
    var includes: [String]
}

/// Converts SDK tools and settings into Responses API payloads.
enum ResponsesConverter {
    static func convertShellEnvironment(_ environment: JSONObject?) -> JSONObject {
        guard var payload = environment else { return ["type": "local"] }
        if payload["type"] == nil { payload["type"] = "local" }
        return payload
    }

    static func convertToolChoice(_ toolChoice: ToolChoice?, tools: [any Tool]?, handoffs: [Handoff]?, model: String?) throws -> JSONValue? {
        guard let toolChoice else { return nil }
        switch toolChoice {
        case .mcp(let choice):
            return ["server_label": .string(choice.serverLabel), "type": "mcp", "name": .string(choice.name)]
        case .required:
            try validateRequiredToolChoice(tools: tools)
            return "required"
        case .auto:
            return "auto"
        case .none:
            return "none"
        case .named(let name):
            switch name {
            case "required":
                try validateRequiredToolChoice(tools: tools)
                return "required"
            case "auto": return "auto"
            case "none": return "none"
            case "programmatic_tool_calling": return ["type": "programmatic_tool_calling"]
            case "file_search": return ["type": "file_search"]
            case "web_search": return ["type": "web_search"]
            case "web_search_preview": return ["type": "web_search_preview"]
            case "computer", "computer_use", "computer_use_preview":
                if hasTool(ComputerTool.self, tools) {
                    return convertBuiltinComputerToolChoice(name, model: model)
                }
                if name == "computer_use_preview" { return ["type": "computer_use_preview"] }
                try validateNamedFunctionToolChoice(name, tools: tools, handoffs: handoffs)
                return ["type": "function", "name": .string(name)]
            case "image_generation": return ["type": "image_generation"]
            case "code_interpreter": return ["type": "code_interpreter"]
            case "shell" where hasTool(ShellTool.self, tools): return ["type": "shell"]
            case "apply_patch" where hasTool(ApplyPatchTool.self, tools): return ["type": "apply_patch"]
            case "mcp": return ["type": "mcp"]
            default:
                try validateNamedFunctionToolChoice(name, tools: tools, handoffs: handoffs)
                return ["type": "function", "name": .string(name)]
            }
        }
    }

    private static func hasTool<T>(_ type: T.Type, _ tools: [any Tool]?) -> Bool {
        (tools ?? []).contains { $0 is T }
    }

    static func validateRequiredToolChoice(tools: [any Tool]?) throws {
        guard let tools, !tools.isEmpty else { return }
        if tools.contains(where: { $0 is ToolSearchTool }) { return }
        if tools.contains(where: isRequiredToolSearchSurface) {
            throw UserError(
                "tool_choice='required' is not currently supported when deferred-loading Responses tools are configured "
                    + "without ToolSearchTool() on the OpenAI Responses API. Add ToolSearchTool() or use `auto`."
            )
        }
    }

    static func validateNamedFunctionToolChoice(_ toolChoice: String, tools: [any Tool]?, handoffs: [Handoff]?) throws {
        if (tools ?? []).isEmpty && (handoffs ?? []).isEmpty { return }
        var topLevel: Set<String> = []
        var allLocal: Set<String> = []
        var deferredOnly: Set<String> = []
        var namespaced: Set<String> = []
        var namespaceNames: Set<String> = []
        let hasHostedToolSearch = (tools ?? []).contains { $0 is ToolSearchTool }
        for handoff in handoffs ?? [] {
            topLevel.insert(handoff.toolName)
            allLocal.insert(handoff.toolName)
        }
        for case let tool as FunctionTool in tools ?? [] {
            allLocal.insert(tool.name)
            guard let namespace = tool.toolNamespace else {
                if tool.deferLoading { deferredOnly.insert(tool.name) } else { topLevel.insert(tool.name) }
                continue
            }
            namespaced.insert(tool.name)
            namespaceNames.insert(namespace)
        }
        if toolChoice == "tool_search", hasHostedToolSearch, !allLocal.contains(toolChoice) {
            throw UserError(
                "tool_choice='tool_search' is not supported for ToolSearchTool() on the OpenAI Responses API. Use `auto` or "
                    + "`required`, or target a real top-level function tool named `tool_search`."
            )
        }
        if toolChoice == "tool_search", !hasHostedToolSearch, !allLocal.contains(toolChoice) {
            throw UserError(
                "tool_choice='tool_search' requires ToolSearchTool() or a real top-level function tool named `tool_search` on the OpenAI Responses API."
            )
        }
        if (namespaced.contains(toolChoice) || namespaceNames.contains(toolChoice)) && !topLevel.contains(toolChoice) {
            throw UserError(
                "Named tool_choice must target a callable tool, not a namespace wrapper or bare inner name from tool_namespace(), "
                    + "on the OpenAI Responses API. Use `auto`, `required`, `none`, or target a top-level or qualified namespaced function tool."
            )
        }
        if deferredOnly.contains(toolChoice), !topLevel.contains(toolChoice) {
            throw UserError(
                "Named tool_choice is not currently supported for deferred-loading function tools on the OpenAI Responses API. "
                    + "Use `auto`, `required`, `none`, or load the tool via ToolSearchTool() first."
            )
        }
    }

    static func isPreviewComputerModel(_ model: String?) -> Bool {
        model?.hasPrefix("computer-use-preview") ?? false
    }

    static func isGAComputerModel(_ model: String?) -> Bool {
        guard let model else { return false }
        return model.hasPrefix("gpt-5.4") || model.hasPrefix("gpt-5.5")
    }

    static func resolveComputerToolModel(requestModel: String?, tools: [any Tool]?) -> String? {
        hasTool(ComputerTool.self, tools) ? requestModel : nil
    }

    static func shouldUsePreviewComputerTool(model: String?, toolChoice: ToolChoice?) -> Bool {
        if isPreviewComputerModel(model) { return true }
        if model != nil { return false }
        if case .named(let name)? = toolChoice, name == "computer" || name == "computer_use" { return false }
        return true
    }

    static func convertBuiltinComputerToolChoice(_ toolChoice: String, model: String?) -> JSONValue {
        if isPreviewComputerModel(model) { return ["type": "computer_use_preview"] }
        if shouldUsePreviewComputerTool(model: model, toolChoice: .named(toolChoice)) { return ["type": "computer_use_preview"] }
        return ["type": "computer"]
    }

    static func getResponseFormat(_ outputSchema: (any AgentOutputSchemaBase)?) throws -> JSONObject? {
        guard let outputSchema, !outputSchema.isPlainText() else { return nil }
        return [
            "format": [
                "type": "json_schema",
                "name": "final_output",
                "schema": .object(try outputSchema.jsonSchema()),
                "strict": .bool(outputSchema.isStrictJSONSchema()),
            ],
        ]
    }

    static func convertTools(
        _ tools: [any Tool],
        _ handoffs: [Handoff],
        allowOpaqueToolSearchSurface: Bool = false,
        model: String? = nil,
        toolChoice: ToolChoice? = nil
    ) throws -> ConvertedTools {
        var converted: [JSONObject?] = []
        var includes: [String] = []
        var namespaceIndex: [String: Int] = [:]
        var namespaceOrder: [String] = []
        var namespaceTools: [String: [JSONObject]] = [:]
        var namespaceDescriptions: [String: String] = [:]
        let usePreviewComputerTool = shouldUsePreviewComputerTool(model: model, toolChoice: toolChoice)
        try validateResponsesToolSearchConfiguration(tools, allowOpaqueSearchSurface: allowOpaqueToolSearchSurface)
        try validateResponsesProgrammaticToolCallingConfiguration(tools, toolChoice: toolChoice, allowOpaqueToolSearchSurface: allowOpaqueToolSearchSurface)
        let computerTools = tools.filter { $0 is ComputerTool }
        if computerTools.count > 1 {
            throw UserError("You can only provide one computer tool. Got \(computerTools.count)")
        }
        for tool in tools {
            if let functionTool = tool as? FunctionTool, let namespace = functionTool.toolNamespace, !namespace.isEmpty {
                let description = functionTool.toolNamespaceDescription ?? ""
                if namespaceIndex[namespace] == nil {
                    namespaceIndex[namespace] = converted.count
                    namespaceOrder.append(namespace)
                    converted.append(nil)
                    namespaceTools[namespace] = []
                    namespaceDescriptions[namespace] = description
                } else if namespaceDescriptions[namespace] != description {
                    throw UserError("All tools in namespace '\(namespace)' must share the same description.")
                }
                namespaceTools[namespace, default: []].append(convertFunctionTool(functionTool, includeDeferLoading: true))
                continue
            }
            let (payload, include) = try convertTool(tool, usePreviewComputerTool: usePreviewComputerTool)
            converted.append(payload)
            if let include { includes.append(include) }
        }
        for namespace in namespaceOrder {
            guard let index = namespaceIndex[namespace] else { continue }
            converted[index] = [
                "type": "namespace",
                "name": .string(namespace),
                "description": .string(namespaceDescriptions[namespace] ?? ""),
                "tools": .array((namespaceTools[namespace] ?? []).map(JSONValue.object)),
            ]
        }
        for handoff in handoffs {
            converted.append(convertHandoffTool(handoff))
        }
        return ConvertedTools(tools: converted.compactMap { $0 }, includes: includes)
    }

    static func convertFunctionTool(_ tool: FunctionTool, includeDeferLoading: Bool = true) -> JSONObject {
        var payload: JSONObject = [
            "name": .string(tool.name),
            "parameters": .object(tool.paramsJSONSchema),
            "strict": .bool(tool.strictJSONSchema),
            "type": "function",
            "description": .string(tool.description),
        ]
        if includeDeferLoading, tool.deferLoading { payload["defer_loading"] = true }
        if let callers = tool.allowedCallers { payload["allowed_callers"] = .array(callers.map { .string($0.rawValue) }) }
        if let schema = tool.outputJSONSchema { payload["output_schema"] = .object(schema) }
        return payload
    }

    static func convertPreviewComputerTool(_ tool: ComputerTool) throws -> JSONObject {
        guard case .computer(let computer) = tool.computer else {
            throw UserError(
                "Computer tool is not initialized for serialization. Call resolve_computer({ tool, run_context }) "
                    + "with a run context first when building payloads manually."
            )
        }
        guard let environment = computer.environment, let dimensions = computer.dimensions else {
            throw UserError(
                "Preview computer tool payloads require `environment` and `dimensions` on the Computer/AsyncComputer implementation."
            )
        }
        return [
            "type": "computer_use_preview",
            "environment": .string(environment),
            "display_width": .int(dimensions.width),
            "display_height": .int(dimensions.height),
        ]
    }

    // swiftlint:disable:next cyclomatic_complexity
    static func convertTool(_ tool: any Tool, usePreviewComputerTool: Bool = false) throws -> (JSONObject, String?) {
        switch tool {
        case let tool as FunctionTool:
            return (convertFunctionTool(tool), nil)
        case let tool as WebSearchTool:
            var payload: JSONObject = [
                "type": "web_search",
                "filters": tool.filters.map(JSONValue.object) ?? .null,
                "user_location": tool.userLocation.map(JSONValue.object) ?? .null,
                "search_context_size": .string(tool.searchContextSize),
            ]
            if let access = tool.externalWebAccess { payload["external_web_access"] = .bool(access) }
            if let types = tool.searchContentTypes { payload["search_content_types"] = .array(types.map(JSONValue.string)) }
            if let settings = tool.imageSettings { payload["image_settings"] = .object(settings) }
            let include = (tool.searchContentTypes?.contains("image") ?? false) ? "web_search_call.results" : nil
            return (payload, include)
        case let tool as FileSearchTool:
            var payload: JSONObject = ["type": "file_search", "vector_store_ids": .array(tool.vectorStoreIds.map(JSONValue.string))]
            if let maxResults = tool.maxNumResults {
                guard (0...50).contains(maxResults) else {
                    throw UserError("FileSearchTool max_num_results must be zero, an integer between 1 and 50, or None.")
                }
                if maxResults > 0 { payload["max_num_results"] = .int(maxResults) }
            }
            if let ranking = tool.rankingOptions, !ranking.isEmpty { payload["ranking_options"] = .object(ranking) }
            if let filters = tool.filters, !filters.isEmpty { payload["filters"] = .object(filters) }
            return (payload, tool.includeSearchResults ? "file_search_call.results" : nil)
        case let tool as ComputerTool:
            return (usePreviewComputerTool ? try convertPreviewComputerTool(tool) : ["type": "computer"], nil)
        case let tool as CustomTool:
            return (tool.toolConfig, nil)
        case let tool as HostedMCPTool:
            return (tool.toolConfig, nil)
        case let tool as ApplyPatchTool:
            var payload: JSONObject = ["type": "apply_patch"]
            if let callers = tool.allowedCallers { payload["allowed_callers"] = .array(callers.map { .string($0.rawValue) }) }
            return (payload, nil)
        case let tool as ShellTool:
            var payload: JSONObject = ["type": "shell", "environment": .object(convertShellEnvironment(tool.environment))]
            if let callers = tool.allowedCallers { payload["allowed_callers"] = .array(callers.map { .string($0.rawValue) }) }
            return (payload, nil)
        case let tool as ImageGenerationTool:
            return (tool.toolConfig, nil)
        case let tool as CodeInterpreterTool:
            return (tool.toolConfig, nil)
        case is LocalShellTool:
            return (["type": "local_shell"], nil)
        case let tool as ToolSearchTool:
            var payload: JSONObject = ["type": "tool_search"]
            if let description = tool.description { payload["description"] = .string(description) }
            if let execution = tool.execution { payload["execution"] = .string(execution) }
            if let parameters = tool.parameters { payload["parameters"] = parameters }
            return (payload, nil)
        case is ProgrammaticToolCallingTool:
            return (["type": "programmatic_tool_calling"], nil)
        default:
            throw UserError("Unknown tool type: \(type(of: tool)), tool")
        }
    }

    static func convertHandoffTool(_ handoff: Handoff) -> JSONObject {
        [
            "name": .string(handoff.toolName),
            "parameters": .object(handoff.inputJSONSchema),
            "strict": .bool(handoff.strictJSONSchema),
            "type": "function",
            "description": .string(handoff.toolDescription),
        ]
    }
}
