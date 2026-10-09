import Foundation
import Testing
@testable import Agents

// Ported from tests/test_extra_headers.py.
//
// Swift: requests go through an `HTTPStub` instead of a dummy client object.
// `test_extra_headers_passed_to_openai_client` covers Chat Completions, which is not ported yet.
@Suite struct ExtraHeadersTests {
    @Test func testExtraHeadersPassedToOpenAIResponsesModel() async throws {
        let stub = HTTPStub { _ in
            .json([
                "id": "dummy", "object": "response", "output": [],
                "usage": [
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "input_tokens_details": ["cached_tokens": 0], "output_tokens_details": ["reasoning_tokens": 0],
                ],
            ])
        }
        let model = OpenAIResponsesModel(model: "gpt-4", openAIClient: stub.makeClient())

        _ = try await model.getResponse(ModelRequest(
            systemInstructions: nil, input: "hi", modelSettings: ModelSettings(extraHeaders: ["X-Test-Header": "test-value"]),
            tools: [], outputSchema: nil, handoffs: [], tracing: .disabled, previousResponseId: nil
        ))

        #expect(stub.requests.count == 1)
        #expect(stub.requests.first?.header("X-Test-Header") == "test-value")
    }
}
