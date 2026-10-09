import Testing
@testable import Agents

// Ported from tests/models/test_remove_openai_responses_api_incompatible_fields.py.
//
// Swift: `result is list_input` identity checks become equality, since arrays are values.
@Suite struct RemoveOpenAIResponsesAPIIncompatibleFieldsTests {
    private let model = OpenAIResponsesModel(model: "gpt-5", openAIClient: OpenAIClient(apiKey: "test_key"))

    private func clean(_ input: [TResponseInputItem]) -> [TResponseInputItem] {
        model.removeOpenAIResponsesAPIIncompatibleFields(input)
    }

    @Test func testReturnsUnchangedWhenNoProviderData() {
        let input: [TResponseInputItem] = [["type": "message", "content": "hello"], ["type": "function_call", "call_id": "call_123", "name": "test"]]
        #expect(clean(input) == input)
    }

    @Test func testRemovesReasoningItemsWithProviderData() {
        let result = clean([
            ["type": "message", "content": "hello"],
            ["type": "reasoning", "provider_data": ["model": "gemini/gemini-3"]],
            ["type": "function_call", "call_id": "call_123"],
        ])
        #expect(result == [["type": "message", "content": "hello"], ["type": "function_call", "call_id": "call_123"]])
    }

    @Test func testKeepsReasoningItemsWithoutProviderData() {
        let result = clean([["type": "reasoning", "summary": []], ["type": "message", "content": "hello", "provider_data": ["foo": "bar"]]])
        #expect(result == [["type": "reasoning", "summary": []], ["type": "message", "content": "hello"]])
    }

    @Test func testRemovesProviderDataFromAllItems() {
        let result = clean([
            ["type": "message", "content": "hello", "provider_data": ["model": "gemini/gemini-3"]],
            ["type": "function_call", "call_id": "call_123", "provider_data": ["model": "gemini/gemini-3"]],
        ])
        #expect(result.count == 2)
        #expect(result.allSatisfy { $0["provider_data"] == nil })
    }

    @Test func testRemovesFakeResponsesId() {
        let result = clean([["type": "message", "id": .string(fakeResponsesId), "content": "hello", "provider_data": ["model": "gemini/gemini-3"]]])
        #expect(result.count == 1)
        #expect(result[0]["id"] == nil)
        #expect(result[0].string("content") == "hello")
    }

    @Test func testRemovesFakeResponsesIdWithoutProviderData() {
        // Several SDK paths build output items with the placeholder ID and no provider_data.
        let result = clean([["role": "user", "content": "hi"], ["type": "message", "id": .string(fakeResponsesId), "content": "hello"]])
        #expect(result.count == 2)
        #expect(result[1]["id"] == nil)
        #expect(result[1].string("content") == "hello")
    }

    @Test func testPreservesRealIds() {
        let result = clean([["type": "message", "id": "msg_real123", "content": "hello", "provider_data": [:]]])
        #expect(result[0].string("id") == "msg_real123")
    }

    @Test func testHandlesEmptyList() {
        #expect(clean([]).isEmpty)
    }

    @Test func testCombinedScenario() {
        let result = clean([
            ["type": "message", "content": "user input"],
            ["type": "reasoning", "summary": [], "provider_data": ["model": "gemini/gemini-3"]],
            ["type": "function_call", "call_id": "call_abc_123", "name": "get_weather", "provider_data": ["model": "gemini/gemini-3"]],
            ["type": "function_call_output", "call_id": "call_abc_123", "output": #"{"temp": 72}"#],
            ["type": "message", "id": .string(fakeResponsesId), "content": "The weather is 72F", "provider_data": ["model": "gemini/gemini-3"]],
        ])

        // Should have 4 items (reasoning with provider_data removed).
        #expect(result.count == 4)
        #expect(result[0] == ["type": "message", "content": "user input"])
        #expect(result[1].string("type") == "function_call")
        #expect(result[1].string("call_id") == "call_abc_123")
        #expect(result[1]["provider_data"] == nil)
        #expect(result[2].string("type") == "function_call_output")
        #expect(result[2].string("call_id") == "call_abc_123")
        #expect(result[3].string("type") == "message")
        #expect(result[3].string("content") == "The weather is 72F")
        #expect(result[3]["id"] == nil)
        #expect(result[3]["provider_data"] == nil)
    }

    @Test func testRequestPayloadDropsSDKGeneratedPlaceholderIds() throws {
        // A replayed SDK-built assistant message must not send its placeholder ID upstream.
        let message = createMessageOutputItem(Agent(name: "test"), "the run error handler final output").rawItem
        let request = ModelRequest(
            systemInstructions: nil, input: .items([["role": "user", "content": "hi"], message]), modelSettings: ModelSettings(),
            tools: [], outputSchema: nil, handoffs: [], tracing: .disabled
        )

        let body = try model.buildResponseCreateBody(request, stream: false).body

        #expect(message.string("id") == fakeResponsesId)
        #expect(body["input"]?.arrayValue?[1].objectValue?["id"] == nil)
    }
}
