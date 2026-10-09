import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing
@testable import Agents

// Continuation of tests/models/test_openai_responses.py: prompt-managed requests, computer tool
// selection, terminal failures, retry advice, and responses without usage.
//
// Swift: `_get_client().with_options(max_retries=0)` maps to `OpenAIClient` reading
// `ModelRetryScope.providerManagedRetriesDisabled`, covered by the retry tests; the cancellation
// test for an in-flight `aclose` and the `_ResponseStreamWithRequestId` close-ordering tests have no
// `URLSession` equivalent.
extension OpenAIResponsesTests {
    private final class MetadataComputer: Computer {
        let hasMetadata: Bool

        init(hasMetadata: Bool = true) {
            self.hasMetadata = hasMetadata
        }

        var environment: Environment? { hasMetadata ? "mac" : nil }
        var dimensions: (width: Int, height: Int)? { hasMetadata ? (800, 600) : nil }
        func screenshot() async throws -> String { "screenshot" }
        func click(x: Int, y: Int, button: Button) async throws {}
        func doubleClick(x: Int, y: Int) async throws {}
        func scroll(x: Int, y: Int, scrollX: Int, scrollY: Int) async throws {}
        func type(text: String) async throws {}
        func wait() async throws {}
        func move(x: Int, y: Int) async throws {}
        func keypress(keys: [String]) async throws {}
        func drag(path: [(x: Int, y: Int)]) async throws {}
    }

    private struct DummyEditor: ApplyPatchEditor {
        func createFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? { nil }
        func updateFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? { nil }
        func deleteFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? { nil }
    }

    private var previewComputerPayload: JSONValue {
        ["type": "computer_use_preview", "environment": "mac", "display_width": 800, "display_height": 600]
    }

    /// Sends one non-streamed request and returns the captured body.
    private func capturedBody(model name: String = "gpt-4", modelIsExplicit: Bool = true, _ request: ModelRequest) async throws -> JSONObject? {
        let stub = emptyResponseStub(stream: false)
        let model = OpenAIResponsesModel(model: name, openAIClient: stub.makeClient(), modelIsExplicit: modelIsExplicit)
        _ = try await model.getResponse(request)
        return stub.requests.first?.body
    }

    // MARK: - Prompt-managed requests

    @Test func testPromptIdOmitsModelParameter() async throws {
        let body = try await capturedBody(modelIsExplicit: false, makeRequest(prompt: ["id": "pmpt_123"]))
        #expect(body?["prompt"] == ["id": "pmpt_123"])
        #expect(body?["model"] == nil)
    }

    @Test func testPromptIdOmitsToolsParameterWhenNoToolsConfigured() async throws {
        let body = try await capturedBody(modelIsExplicit: false, makeRequest(prompt: ["id": "pmpt_123"]))
        #expect(body?["tools"] == nil)
    }

    @Test func testPromptIdOmitsToolChoiceWhenNoToolsConfigured() async throws {
        let request = makeRequest(modelSettings: ModelSettings(toolChoice: .named("web_search_preview")), prompt: ["id": "pmpt_123"])
        let body = try await capturedBody(modelIsExplicit: false, request)
        #expect(body?["tools"] == nil)
        #expect(body?["tool_choice"] == nil)
    }

    @Test(arguments: ["none", "required"])
    func testPromptIdKeepsLiteralToolChoiceWithoutLocalTools(toolChoice: String) async throws {
        let request = makeRequest(modelSettings: ModelSettings(toolChoice: ToolChoice(toolChoice)), prompt: ["id": "pmpt_123"])
        let body = try await capturedBody(modelIsExplicit: false, request)
        #expect(body?["tools"] == nil)
        #expect(body?["tool_choice"] == .string(toolChoice))
    }

    @Test func testPromptIdKeepsExplicitToolSearchWithoutLocalSurface() async throws {
        let body = try await capturedBody(modelIsExplicit: false, makeRequest(tools: [ToolSearchTool()], prompt: ["id": "pmpt_123"]))
        #expect(body?["prompt"] == ["id": "pmpt_123"])
        #expect(body?["tools"] == [["type": "tool_search"]])
    }

    // MARK: - Computer tool selection

    @Test func testGAComputerToolDoesNotRequirePreviewMetadata() async throws {
        let request = makeRequest(tools: [ComputerTool(computer: MetadataComputer(hasMetadata: false))])
        let body = try await capturedBody(model: "gpt-5.4", request)
        #expect(body?["tools"] == [["type": "computer"]])
    }

    @Test func testPromptIdUsesPreviewComputerPayloadWhenPromptOwnsModel() async throws {
        let request = makeRequest(tools: [ComputerTool(computer: MetadataComputer())], prompt: ["id": "pmpt_123"])
        let body = try await capturedBody(model: "gpt-5.4", modelIsExplicit: false, request)
        #expect(body?["model"] == nil)
        #expect(body?["tool_choice"] == nil)
        #expect(body?["tools"] == [previewComputerPayload])
    }

    @Test func testPromptIdComputerWithoutPreviewMetadataRaisesClearError() async throws {
        let stub = emptyResponseStub(stream: false)
        let model = OpenAIResponsesModel(model: "gpt-5.4", openAIClient: stub.makeClient(), modelIsExplicit: false)

        await expectErrorAsync(UserError.self, containing: "Preview computer tool payloads require `environment` and `dimensions`") {
            _ = try await model.getResponse(makeRequest(tools: [ComputerTool(computer: MetadataComputer(hasMetadata: false))], prompt: ["id": "pmpt_123"]))
        }
        #expect(stub.requests.isEmpty)
    }

    @Test func testPromptIdUnresolvedComputerUsesPreviewPayloadShape() async throws {
        let stub = emptyResponseStub(stream: false)
        let model = OpenAIResponsesModel(model: "gpt-5.4", openAIClient: stub.makeClient(), modelIsExplicit: false)
        let tool = ComputerTool(provider: ComputerProvider { _ in MetadataComputer() })

        await expectErrorAsync(UserError.self, containing: "Computer tool is not initialized for serialization") {
            _ = try await model.getResponse(makeRequest(tools: [tool], prompt: ["id": "pmpt_123"]))
        }
        #expect(stub.requests.isEmpty)
    }

    @Test(arguments: ["computer", "computer_use"])
    func testPromptIdExplicitGAComputerToolChoiceUsesGASelectorAndTool(toolChoice: String) async throws {
        let request = makeRequest(
            modelSettings: ModelSettings(toolChoice: .named(toolChoice)), tools: [ComputerTool(computer: MetadataComputer())], prompt: ["id": "pmpt_123"]
        )
        let body = try await capturedBody(model: "gpt-5.4", modelIsExplicit: false, request)
        #expect(body?["model"] == nil)
        #expect(body?["tool_choice"] == ["type": "computer"])
        #expect(body?["tools"] == [["type": "computer"]])
    }

    @Test(arguments: ["computer", "computer_use"])
    func testPreviewModelForcedComputerToolChoiceUsesPreviewSelector(toolChoice: String) async throws {
        let request = makeRequest(modelSettings: ModelSettings(toolChoice: .named(toolChoice)), tools: [ComputerTool(computer: MetadataComputer())])
        let body = try await capturedBody(model: "computer-use-preview", request)
        #expect(body?["model"] == "computer-use-preview")
        #expect(body?["tool_choice"] == ["type": "computer_use_preview"])
        #expect(body?["tools"] == [previewComputerPayload])
    }

    @Test(arguments: ["shell", "apply_patch"])
    func testBuiltinToolChoiceSelectsBuiltinAmongMultipleTools(toolChoice: String) async throws {
        let builtinTool: any Tool = toolChoice == "shell" ? try ShellTool(executor: { _ in .text("ok") }) : try ApplyPatchTool(editor: DummyEditor())
        let tools: [any Tool] = [
            try FunctionTool.create(name: "lookup_account") { _ in "ok" },
            builtinTool,
            try FunctionTool.create(name: "summarize") { _ in "ok" },
        ]
        let request = makeRequest(modelSettings: ModelSettings(toolChoice: .named(toolChoice)), tools: tools, handoffs: [handoff(Agent(name: "Target"))])

        let body = try await capturedBody(model: "gpt-5.4", request)

        #expect(body?["tool_choice"] == ["type": .string(toolChoice)])
        #expect(body?["tools"]?.arrayValue?.contains { $0.objectValue?.string("type") == toolChoice } == true)
    }

    // MARK: - Terminal failures

    private func responseWithTerminalStatus(_ status: String) -> JSONObject {
        [
            "id": "resp-terminal", "created_at": 0, "model": "fake", "object": "response", "output": [], "tool_choice": "none",
            "tools": [], "top_p": nil, "parallel_tool_calls": false, "usage": nil, "status": .string(status),
            "incomplete_details": status == "incomplete" ? ["reason": "max_output_tokens"] : nil,
        ]
    }

    @Test(arguments: ["incomplete", "failed"])
    func testGetResponseRejectsFailedTerminalResponseStatus(status: String) async throws {
        let response = responseWithTerminalStatus(status)
        let stub = HTTPStub { _ in .json(response) }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())

        await expectErrorAsync(ModelBehaviorError.self, containing: "response.\(status)") {
            _ = try await model.getResponse(makeRequest())
        }
    }

    @Test func testStreamResponseRejectsResponseErrorTerminalEvent() async throws {
        let errorEvent: JSONObject = ["type": "error", "code": "invalid_request_error", "message": "bad request", "param": nil, "sequence_number": 0]
        let stub = HTTPStub { _ in .sse([errorEvent]) }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())
        let events = SharedBox<[TResponseStreamEvent]>([])

        await expectErrorAsync(ModelBehaviorError.self, containing: "invalid_request_error") {
            for try await event in model.streamResponse(makeRequest()) {
                events.mutate { $0.append(event) }
            }
        }

        #expect(events.value.count == 1)
        #expect(events.value.first?.string("type") == "error")
        #expect(events.value.first?.string("code") == "invalid_request_error")
        #expect(events.value.first?.string("message") == "bad request")
    }

    // MARK: - Retry advice

    private func httpError(status: Int, headers: [String: String] = [:], body: JSONObject) -> OpenAIAPIError {
        let url = URL(string: "https://api.openai.com/v1/responses")!
        let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)
        return OpenAIClient.apiError(status: status, data: Data(body.jsonString().utf8), response: response)
    }

    private var connectionError: any Error { OpenAIClient.mapTransportError(URLError(.networkConnectionLost)) }

    @Test func testGetRetryAdviceUsesOpenAIHeaders() throws {
        let error = httpError(
            status: 429, headers: ["x-should-retry": "true", "retry-after-ms": "250", "x-request-id": "req_456"],
            body: ["error": ["code": "rate_limit"]]
        )
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: OpenAIClient(apiKey: "test_key"))

        let advice = try #require(model.getRetryAdvice(ModelRetryAdviceRequest(error: error, attempt: 1, stream: false)))

        #expect(advice.suggested == true)
        #expect(advice.retryAfter == 0.25)
        #expect(advice.replaySafety == "safe")
        #expect(advice.normalized?.errorCode == "rate_limit")
        #expect(advice.normalized?.statusCode == 429)
        #expect(advice.normalized?.requestId == "req_456")
    }

    @Test(arguments: [nil, "resp_prev"] as [String?])
    func testGetRetryAdviceKeepsTransportFailuresRetryable(previousResponseId: String?) throws {
        // Ports _keeps_stateful_transport_failures_ambiguous and _keeps_stateless_transport_failures_retryable.
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: OpenAIClient(apiKey: "test_key"))

        let advice = try #require(model.getRetryAdvice(ModelRetryAdviceRequest(
            error: connectionError, attempt: 1, stream: false, previousResponseId: previousResponseId
        )))

        #expect(advice.suggested == true)
        #expect(advice.replaySafety == nil)
        #expect(advice.normalized?.isNetworkError == true)
    }

    @Test func testGetRetryAdviceMarksStatefulHTTPFailuresReplaySafe() throws {
        let error = httpError(status: 429, body: ["error": ["code": "rate_limit"]])
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: OpenAIClient(apiKey: "test_key"))

        let advice = try #require(model.getRetryAdvice(ModelRetryAdviceRequest(error: error, attempt: 1, stream: false, previousResponseId: "resp_prev")))

        #expect(advice.suggested == true)
        #expect(advice.replaySafety == "safe")
        #expect(advice.normalized?.statusCode == 429)
    }

    @Test func testGetClientDisablesProviderManagedRetriesWhenRequested() async throws {
        // Swift: also ports test_get_client_disables_provider_managed_retries_on_runner_retry by
        // checking that the client makes a single attempt inside the disabled scope.
        let stub = HTTPStub { _ in .json(["error": ["message": "overloaded"]], status: 503) }
        let client = stub.makeClient(maxRetries: 1)
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: client)

        await expectErrorAsync(OpenAIAPIError.self) {
            try await ModelRetryScope.$providerManagedRetriesDisabled.withValue(true) {
                _ = try await model.getResponse(makeRequest())
            }
        }

        #expect(stub.requests.count == 1)
    }

    // MARK: - Responses without usage

    private var responseWithoutUsage: JSONObject {
        [
            "id": "resp-no-usage", "created_at": 0, "model": "fake", "object": "response", "output": [], "tool_choice": "none",
            "tools": [], "top_p": nil, "parallel_tool_calls": false, "usage": nil,
        ]
    }

    @Test func testRequestIsCountedWhenResponsesProviderOmitsUsage() async throws {
        let response = responseWithoutUsage
        let stub = HTTPStub { _ in .json(response) }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())

        let result = try await model.getResponse(makeRequest(modelSettings: ModelSettings(preserveRawUsage: true)))

        #expect(result.usage.requests == 1)
        #expect(result.usage.totalTokens == 0)
        #expect(result.rawUsage == nil)
    }

    @Test func testStreamedRequestIsCountedWhenResponsesProviderOmitsUsage() async throws {
        // Also covers test_fallback_response_stream_counts_request_without_usage.
        let response = responseWithoutUsage
        let stub = HTTPStub { _ in .sse([self.completedEvent(response)]) }
        let agent = Agent(name: "test", model: .instance(OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())))

        let result = try Runner.runStreamed(agent, input: "hi")
        for try await _ in result.streamEvents() {}

        #expect(result.contextWrapper.usage.requests == 1)
        #expect(result.contextWrapper.usage.totalTokens == 0)
    }

    @Test(.tracing) func testResponseSpanCountsRequestWithoutUsage() async throws {
        let response = responseWithoutUsage
        let stub = HTTPStub { _ in .json(response) }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())

        try await trace("test").run {
            _ = try await model.getResponse(makeRequest(tracing: .enabled))
        }

        let spans = fetchOrderedSpans().filter { $0.spanData.type == "response" }.compactMap { $0.export() }
        #expect(spans.count == 1)
        let usage = spans.first?["span_data"]?.objectValue?["usage"]?.objectValue
        #expect(usage?["requests"] == 1)
        #expect(usage?["total_tokens"] == 0)
    }

    @Test(.tracing) func testStreamedSpanCountsRequestBeforeTerminalEventClose() async throws {
        let response = responseWithoutUsage
        let stub = HTTPStub { _ in .sse([self.completedEvent(response)]) }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())

        try await trace("test").run {
            for try await event in model.streamResponse(makeRequest(tracing: .enabled)) where event.string("type") == "response.completed" {
                break
            }
        }
        try await withTestTimeout(5) {
            while !fetchOrderedSpans().contains(where: { $0.spanData.type == "response" }) {
                await Task.yield()
            }
        }

        let spans = fetchOrderedSpans().filter { $0.spanData.type == "response" }.compactMap { $0.export() }
        #expect(spans.count == 1)
        let usage = spans.first?["span_data"]?.objectValue?["usage"]?.objectValue
        #expect(usage?["requests"] == 1)
        #expect(usage?["total_tokens"] == 0)
    }
}
