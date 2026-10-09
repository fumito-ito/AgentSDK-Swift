import Foundation
import Testing
@testable import Agents

// Ported from tests/models/test_openai_responses.py.
//
// Swift: requests go through the real `OpenAIClient` and an `HTTPStub` instead of fake
// `AsyncOpenAI` objects or `httpx` mock transports. `_build_response_create_kwargs` maps to
// `buildResponseCreateBody`, whose body already merges `extra_body`; `extra_query` and headers
// travel separately. Skipped upstream groups:
// - every `OpenAIResponsesWSModel` / websocket test (the websocket transport is not ported);
// - openai-python streaming context-manager mechanics (`_fetch_response` parse/close/`__aexit__`
//   bookkeeping, inner HTTP stream `aclose`, close-failure handling), which have no `URLSession`
//   equivalent;
// - `test_image_generation_config_remains_mutable` (dataclass `fields`/`asdict`/`replace`);
// - dictionary `ModelSettings` normalization (settings are typed).
@Suite struct OpenAIResponsesTests {
    func makeRequest(
        input: RunInput = "hi",
        modelSettings: ModelSettings = ModelSettings(),
        tools: [any Tool] = [],
        handoffs: [Handoff] = [],
        tracing: ModelTracing = .disabled,
        previousResponseId: String? = nil,
        prompt: JSONObject? = nil
    ) -> ModelRequest {
        ModelRequest(
            systemInstructions: nil, input: input, modelSettings: modelSettings, tools: tools, outputSchema: nil,
            handoffs: handoffs, tracing: tracing, previousResponseId: previousResponseId, prompt: prompt
        )
    }

    func completedEvent(_ response: JSONObject, sequenceNumber: Int = 0) -> JSONObject {
        ["type": "response.completed", "response": .object(response), "sequence_number": .int(sequenceNumber)]
    }

    /// A stub that answers JSON or SSE with an empty completed response.
    func emptyResponseStub(stream: Bool, headers: [String: String] = [:]) -> HTTPStub {
        HTTPStub { _ in
            let response = getResponseObj([], responseId: "resp-id").json
            return stream ? .sse([completedEvent(response)], headers: headers) : .json(response, headers: headers)
        }
    }

    func send(_ model: OpenAIResponsesModel, _ request: ModelRequest, stream: Bool) async throws {
        if stream {
            for try await _ in model.streamResponse(request) {}
        } else {
            _ = try await model.getResponse(request)
        }
    }

    // MARK: - Usage normalization

    @Test(arguments: [false, true], [
        ([nil, 5, 12], [0, 5, 12]),
        ([7, nil, 12], [7, 0, 12]),
        ([7, 5, nil], [7, 5, 0]),
        ([nil, nil, nil], [0, 0, 0]),
        ([0, 0, 0], [0, 0, 0]),
        ([7, 5, 12], [7, 5, 12]),
    ] as [([Int?], [Int])])
    func testRunnerNormalizesNullableResponsesUsage(stream: Bool, values: ([Int?], [Int])) async throws {
        let (counts, expected) = values
        let usage: JSONObject = [
            "input_tokens": counts[0].map(JSONValue.int) ?? .null,
            "output_tokens": counts[1].map(JSONValue.int) ?? .null,
            "total_tokens": counts[2].map(JSONValue.int) ?? .null,
            "input_tokens_details": ["cached_tokens": 3],
            "output_tokens_details": ["reasoning_tokens": 2],
        ]
        // Exercise provider parsing and usage conversion instead of a normalized scripted model.
        let stub = HTTPStub { _ in
            var payload = getResponseObj([getTextMessage("Hello")]).json
            payload["usage"] = .object(usage)
            return stream ? .sse([self.completedEvent(payload)]) : .json(payload)
        }
        let agent = Agent(
            name: "test",
            model: .instance(OpenAIResponsesModel(model: "fake", openAIClient: stub.makeClient())),
            modelSettings: ModelSettings(preserveRawUsage: true)
        )

        let finalOutput: (any Sendable)?
        let normalized: Usage
        let rawUsage: JSONObject?
        if stream {
            let result = try Runner.runStreamed(agent, input: "hi")
            for try await _ in result.streamEvents() {}
            finalOutput = result.finalOutput
            normalized = result.contextWrapper.usage
            rawUsage = result.rawResponses.first?.rawUsage
        } else {
            let result = try await Runner.run(agent, input: "hi")
            finalOutput = result.finalOutput
            normalized = result.contextWrapper.usage
            rawUsage = result.rawResponses.first?.rawUsage
        }

        #expect(finalOutput as? String == "Hello")
        #expect(normalized.requests == 1)
        #expect([normalized.inputTokens, normalized.outputTokens, normalized.totalTokens] == expected)
        #expect(normalized.inputTokensDetails.cachedTokens == 3)
        #expect(normalized.outputTokensDetails.reasoningTokens == 2)
        #expect(rawUsage == usage)
        if expected == [0, 0, 0] {
            #expect(normalized.requestUsageEntries.isEmpty)
        } else {
            #expect(normalized.requestUsageEntries.count == 1)
            let entry = try #require(normalized.requestUsageEntries.first)
            #expect([entry.inputTokens, entry.outputTokens, entry.totalTokens] == expected)
            #expect(entry.inputTokensDetails.cachedTokens == 3)
            #expect(entry.outputTokensDetails.reasoningTokens == 2)
        }
    }

    // MARK: - Request payloads

    @Test(arguments: [false, true], [false, true])
    func testUnpairedFunctionOutputPreservedInResponsesRequest(stream: Bool, explicitNullCallId: Bool) async throws {
        // Responses keeps external-context output and its source without inventing a call ID.
        let stub = emptyResponseStub(stream: stream)
        var expectedInput: JSONObject = [
            "type": "function_call_output",
            "name": "notifications",
            "namespace": "slack",
            "output": [
                ["type": "input_text", "text": "Alice mentioned you in #deployments."],
                ["type": "input_image", "image_url": "https://example.com/image.png"],
            ],
        ]
        if explicitNullCallId {
            expectedInput["call_id"] = .null
        }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())

        try await send(model, makeRequest(input: .items([expectedInput])), stream: stream)

        #expect(stub.requests.map { $0.body?["input"] } == [.array([.object(expectedInput)])])
    }

    @Test(arguments: [false, true], [
        ([:], []),
        (["search_content_types": ["text"]], []),
        (["search_content_types": ["image", "text"], "image_settings": ["max_results": 3, "caption": false]], ["web_search_call.results"]),
    ] as [(JSONObject, [String])])
    func testWebSearchImageOptionsReachResponsesRequest(stream: Bool, options: (JSONObject, [String])) async throws {
        let (toolOptions, expectedInclude) = options
        let stub = emptyResponseStub(stream: stream)
        let model = OpenAIResponsesModel(model: "gpt-5.6", openAIClient: stub.makeClient())
        let tool = WebSearchTool(
            searchContentTypes: toolOptions["search_content_types"]?.arrayValue?.compactMap(\.stringValue),
            imageSettings: toolOptions["image_settings"]?.objectValue
        )

        try await send(model, makeRequest(input: "Find images of the Golden Gate Bridge.", tools: [tool]), stream: stream)

        #expect(stub.requests.count == 1)
        let body = try #require(stub.requests.first?.body)
        var expectedTool: JSONObject = ["type": "web_search", "filters": nil, "user_location": nil, "search_context_size": "medium"]
        expectedTool.merge(toolOptions)
        #expect(body["tools"] == [.object(expectedTool)])
        #expect(body["include"] ?? [] == .array(expectedInclude.map(JSONValue.string)))
    }

    @Test(arguments: [false, true], [
        ["type": "image_generation"],
        [
            "type": "image_generation", "model": "gpt-image-1", "quality": "high", "action": "edit",
            "input_fidelity": "high", "input_image_mask": ["file_id": "file-mask"],
        ],
        [
            "type": "image_generation", "model": "gpt-image-2.5-sunburst", "quality": "max", "action": "generate",
            "size": "1536x864", "background": "transparent", "output_format": "webp", "output_compression": 0,
            "partial_images": 0, "moderation": "auto",
        ],
        [
            "type": "image_generation", "model": "gpt-image-2.5-flare", "quality": "xhigh", "action": "auto",
            "size": "auto", "output_format": "png", "partial_images": 2,
        ],
    ] as [JSONObject])
    func testImageGenerationOptionsReachResponsesRequest(stream: Bool, toolConfig: JSONObject) async throws {
        let stub = emptyResponseStub(stream: stream)
        let tool = ImageGenerationTool(toolConfig: toolConfig)
        let agent = Agent(
            name: "Image generator",
            tools: [tool],
            model: .instance(OpenAIResponsesModel(model: "gpt-5.5", openAIClient: stub.makeClient()))
        )
        let message: TResponseInputItem = [
            "role": "user",
            "content": [
                ["type": "input_text", "text": "Generate an image, or edit this image."],
                ["type": "input_image", "file_id": "file-image", "detail": "auto"],
            ],
        ]
        let prompt = [message]

        if stream {
            let result = try Runner.runStreamed(agent, input: .items(prompt))
            for try await _ in result.streamEvents() {}
        } else {
            _ = try await Runner.run(agent, input: .items(prompt))
        }

        #expect(stub.requests.count == 1)
        #expect(stub.requests.first?.body?["tools"] == [.object(toolConfig)])
        #expect(tool.toolConfig == toolConfig)
        #expect(stub.requests.first?.body?.string("model") == "gpt-5.5")
    }

    @Test(arguments: [true, false, nil] as [Bool?], ["none", "function", "handoff"])
    func testParallelToolCallsFollowConvertedResponsesTools(parallelToolCalls: Bool?, toolSource: String) throws {
        let tools: [any Tool] = toolSource == "function" ? [try FunctionTool.create(name: "test_tool") { _ in "ok" }] : []
        let handoffs = toolSource == "handoff" ? [handoff(Agent(name: "handoff"))] : []
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: OpenAIClient(apiKey: "test_key"))

        let body = try model.buildResponseCreateBody(
            makeRequest(modelSettings: ModelSettings(parallelToolCalls: parallelToolCalls), tools: tools, handoffs: handoffs), stream: false
        ).body

        let expected = toolSource != "none" ? parallelToolCalls.map(JSONValue.bool) : nil
        #expect(body["parallel_tool_calls"] == expected)
        #expect((body["tools"]?.arrayValue?.isEmpty == false) == (toolSource != "none"))
    }

    @Test(arguments: [true, false, nil] as [Bool?])
    func testParallelToolCallsPreserveStoredPromptOverrides(parallelToolCalls: Bool?) throws {
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: OpenAIClient(apiKey: "test_key"))

        let body = try model.buildResponseCreateBody(
            makeRequest(modelSettings: ModelSettings(parallelToolCalls: parallelToolCalls), prompt: ["id": "pmpt_123"]), stream: false
        ).body

        #expect(body["parallel_tool_calls"] == parallelToolCalls.map(JSONValue.bool))
        #expect(body["tools"] == nil)
    }

    @Test(arguments: [nil, "test_user_agent"] as [String?])
    func testUserAgentHeaderResponses(overrideUserAgent: String?) async throws {
        let stub = emptyResponseStub(stream: true)
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())
        let expectedUserAgent = overrideUserAgent ?? "Agents/Swift \(agentsSDKVersion)"

        if let overrideUserAgent {
            try await OpenAIHeadersOverride.$headers.withValue(["User-Agent": overrideUserAgent]) {
                try await send(model, makeRequest(), stream: true)
            }
        } else {
            try await send(model, makeRequest(), stream: true)
        }

        #expect(stub.requests.first?.header("User-Agent") == expectedUserAgent)
    }

    @Test func testGetResponseExposesRequestId() async throws {
        let stub = HTTPStub { _ in
            .json(getResponseObj([], responseId: "resp-request-id").json, headers: ["x-request-id": "req_nonstream_123"])
        }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())

        let response = try await model.getResponse(makeRequest(modelSettings: ModelSettings(preserveRawUsage: true)))

        #expect(response.responseId == "resp-request-id")
        #expect(response.requestId == "req_nonstream_123")
        #expect(response.rawUsage?["input_tokens_details"]?.objectValue?["cached_tokens"] == 0)
    }

    @Test(.tracing) func testGetResponseSpanExportsUsage() async throws {
        let stub = HTTPStub { _ in
            .json(getResponseObj(
                [], responseId: "resp-usage", usage: Usage(requests: 1, inputTokens: 10, outputTokens: 4, totalTokens: 14)
            ).json)
        }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())

        try await trace("test").run {
            _ = try await model.getResponse(makeRequest(tracing: .enabled))
        }

        let responseSpans = fetchOrderedSpans().filter { $0.spanData.type == "response" }.compactMap { $0.export() }
        #expect(responseSpans.count == 1)
        let expected: JSONObject = [
            "type": "response",
            "response_id": "resp-usage",
            "usage": [
                "requests": 1, "input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
                "input_tokens_details": ["cached_tokens": 0, "cache_write_tokens": 0],
                "output_tokens_details": ["reasoning_tokens": 0],
            ],
        ]
        #expect(responseSpans.first?["span_data"]?.objectValue == expected)
    }

    @Test func testFetchResponseStreamAttachesRequestIdToTerminalResponse() async throws {
        let stub = HTTPStub { _ in
            .sse([self.completedEvent(getResponseObj([], responseId: "resp-stream-request-id").json)], headers: ["x-request-id": "req_stream_123"])
        }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())

        var events: [TResponseStreamEvent] = []
        for try await event in model.streamResponse(makeRequest()) {
            events.append(event)
        }

        #expect(events.count == 1)
        #expect(events.first?["response"]?.objectValue?.string(sseRequestIdKey) == "req_stream_123")
    }

    @Test func testFetchResponseStreamWithoutRequestIdStillReturnsEvents() async throws {
        let stub = HTTPStub { _ in
            .sse([self.completedEvent(getResponseObj([], responseId: "resp-stream-request-id").json)])
        }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())

        var events: [TResponseStreamEvent] = []
        for try await event in model.streamResponse(makeRequest()) {
            events.append(event)
        }

        #expect(events.count == 1)
        #expect(events.first?["response"]?.objectValue?[sseRequestIdKey] == nil)
    }

    @Test(.tracing) func testStreamSpanRecordsIOWhenConsumerStopsAtCompleted() async throws {
        // A consumer that stops at `response.completed` must still see the response and input on
        // the span, so they are attached before the terminal event is yielded.
        let response = getResponseObj(
            [], responseId: "resp-stream-early-close", usage: Usage(requests: 1, inputTokens: 10, outputTokens: 4, totalTokens: 14)
        ).json
        let stub = HTTPStub { _ in .sse([self.completedEvent(response)]) }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())

        try await trace("test").run {
            for try await event in model.streamResponse(makeRequest(input: "the user prompt", tracing: .enabled))
                where event.string("type") == "response.completed" {
                break
            }
        }
        // Swift: stopping early cancels the producer task, which finishes the span asynchronously
        // (upstream awaits `aclose()`), so wait for the finished span.
        try await withTestTimeout(5) {
            while !fetchOrderedSpans().contains(where: { $0.spanData.type == "response" }) {
                await Task.yield()
            }
        }

        let spanData = fetchOrderedSpans().compactMap { $0.spanData as? ResponseSpanData }
        #expect(spanData.count == 1)
        #expect(spanData.first?.response != nil)
        #expect(spanData.first?.input != nil)
        #expect(spanData.first?.usage != nil)
    }
}

// MARK: - Request body construction

extension OpenAIResponsesTests {
    private func buildBody(_ settings: ModelSettings, model name: String = "gpt-4", stream: Bool = false, previousResponseId: String? = nil) throws -> JSONObject {
        let model = OpenAIResponsesModel(model: name, openAIClient: OpenAIClient(apiKey: "test_key"))
        return try model.buildResponseCreateBody(makeRequest(modelSettings: settings, previousResponseId: previousResponseId), stream: stream).body
    }

    @Test func testBuildResponseCreateKwargsRejectsDuplicateExtraArgsKeys() {
        expectError(TypeError.self, containing: "multiple values for keyword argument 'stream'") {
            _ = try buildBody(ModelSettings(extraArgs: ["stream": false]), stream: true)
        }
    }

    @Test func testBuildResponseCreateKwargsIncludesExtraArgsPromptCacheKey() throws {
        #expect(try buildBody(ModelSettings(extraArgs: ["prompt_cache_key": "cache-key"]))["prompt_cache_key"] == "cache-key")
    }

    @Test func testBuildResponseCreateKwargsIncludesContextManagement() throws {
        let contextManagement: [JSONObject] = [["type": "compaction", "compact_threshold": 200_000]]
        #expect(try buildBody(ModelSettings(contextManagement: contextManagement))["context_management"] == .array(contextManagement.map(JSONValue.object)))
    }

    @Test func testBuildResponseCreateKwargsIncludesGpt56RequestControls() throws {
        let reasoning = Reasoning(effort: "max", mode: "pro", context: "all_turns")
        let promptCacheOptions: JSONObject = ["mode": "explicit", "ttl": "30m"]

        let body = try buildBody(
            ModelSettings(reasoning: reasoning, promptCacheRetention: "24h", promptCacheOptions: promptCacheOptions),
            model: "gpt-5.6-sol", previousResponseId: "resp-previous"
        )

        #expect(body["reasoning"] == .object(reasoning.json))
        #expect(body["prompt_cache_retention"] == "24h")
        #expect(body["prompt_cache_options"] == .object(promptCacheOptions))
        #expect(body["previous_response_id"] == "resp-previous")
    }

    @Test func testResponsesRequestsNormalizeAgentSettings() async throws {
        // Swift: settings are typed, so only the ModelSettings variant applies.
        let stub = emptyResponseStub(stream: false)
        let settings = ModelSettings(
            temperature: 0.0, topP: 1.0, frequencyPenalty: 0.0, presencePenalty: 0.0, parallelToolCalls: false, maxTokens: 64,
            reasoning: Reasoning(effort: "low", context: "all_turns"), metadata: ["request": "example"], store: false,
            promptCacheRetention: "24h", extraQuery: ["model_settings_parity": "verified"],
            extraBody: ["prompt_cache_key": "extra-body-cache-key"], extraHeaders: ["x-model-settings-parity": "preserved"],
            retry: ModelRetrySettings(maxRetries: 0, backoff: ModelRetryBackoffSettings(initialDelay: 0.0, jitter: false)),
            contextManagement: [["type": "compaction", "compact_threshold": 200_000]],
            promptCacheOptions: ["mode": "explicit", "ttl": "30m"]
        )
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())
        let agent = Agent(name: "test", model: .instance(model), modelSettings: settings)

        _ = try await Runner.run(agent, input: "hi")

        let request = try #require(stub.requests.first)
        let body = try #require(request.body)
        #expect(body["reasoning"] == ["effort": "low", "context": "all_turns"])
        #expect(body["context_management"] == [["type": "compaction", "compact_threshold": 200_000]])
        #expect(body["prompt_cache_options"] == ["mode": "explicit", "ttl": "30m"])
        #expect(body["prompt_cache_retention"] == "24h")
        #expect(body["store"] == false)
        #expect(body["metadata"] == ["request": "example"])
        #expect(body["temperature"] == 0.0)
        #expect(body["top_p"] == 1.0)
        #expect(body["max_output_tokens"] == 64)
        #expect(body["max_tokens"] == nil)
        #expect(body["parallel_tool_calls"] == nil)
        #expect(request.header("x-model-settings-parity") == "preserved")
        #expect(request.url.query == "model_settings_parity=verified")
        #expect(body["prompt_cache_key"] == "extra-body-cache-key")
        #expect(body["retry"] == nil)
        #expect(body["frequency_penalty"] == nil)
        #expect(body["presence_penalty"] == nil)
    }

    @Test func testBuildResponseCreateKwargsRejectsDuplicatePromptCacheOptionsExtraArgs() {
        expectError(TypeError.self, containing: "prompt_cache_options") {
            _ = try buildBody(
                ModelSettings(extraArgs: ["prompt_cache_options": ["mode": "implicit"]], promptCacheOptions: ["mode": "explicit", "ttl": "30m"]),
                model: "gpt-5.6-sol"
            )
        }
    }

    @Test func testBuildResponseCreateKwargsAllowsPromptCacheOptionsExtraArgsWhenDirectOmitted() throws {
        let options: JSONObject = ["mode": "explicit", "ttl": "30m"]
        #expect(try buildBody(ModelSettings(extraArgs: ["prompt_cache_options": .object(options)]), model: "gpt-5.6-sol")["prompt_cache_options"] == .object(options))
    }

    @Test func testBuildResponseCreateKwargsAllowsExtraArgWhenExplicitArgIsOmitted() throws {
        let contextManagement: JSONValue = [["type": "compaction", "compact_threshold": 200_000]]
        #expect(try buildBody(ModelSettings(extraArgs: ["context_management": contextManagement]))["context_management"] == contextManagement)
    }

    @Test func testBuildResponseCreateKwargsRejectsDuplicateContextManagementExtraArgs() {
        expectError(TypeError.self, containing: "context_management") {
            _ = try buildBody(ModelSettings(
                extraArgs: ["context_management": [["type": "compaction"]]],
                contextManagement: [["type": "compaction", "compact_threshold": 200_000]]
            ))
        }
    }

    @Test(arguments: [false, true])
    func testGetResponseWithOfficialClientTransportExtraKwargs(applied: Bool) async throws {
        // Ports _accepts_unset_transport_extra_kwargs and _applies_transport_extra_kwargs (and
        // test_build_response_create_kwargs_keeps_unset_transport_extra_kwargs_as_none).
        let stub = emptyResponseStub(stream: false)
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())
        let settings = applied
            ? ModelSettings(extraQuery: ["api-version": "2026-01-01-preview"], extraBody: ["extra_transport_field": "enabled"])
            : ModelSettings()

        _ = try await model.getResponse(makeRequest(modelSettings: settings))

        #expect(stub.requests.count == 1)
        let request = try #require(stub.requests.first)
        let suffix = applied ? "?api-version=2026-01-01-preview" : ""
        #expect(request.url.absoluteString == "https://\(stub.host)/v1/responses\(suffix)")
        #expect(request.body?["extra_transport_field"] == (applied ? "enabled" : nil))
    }

    @Test func testCustomBaseURLPromptCacheKeyUsesModelSettingsOnly() throws {
        #expect(try buildBody(ModelSettings())["prompt_cache_key"] == nil)
        #expect(try buildBody(ModelSettings(extraArgs: ["prompt_cache_key": "cache-key"]))["prompt_cache_key"] == "cache-key")
    }

    @Test func testBuildResponseCreateKwargsPreservesUnknownResponseIncludeValues() throws {
        #expect(try buildBody(ModelSettings(responseInclude: ["response.future_flag"]))["include"] == ["response.future_flag"])
    }
}
