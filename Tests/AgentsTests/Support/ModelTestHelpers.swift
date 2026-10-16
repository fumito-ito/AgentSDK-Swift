import Foundation
@testable import Agents
import AgentsTesting

// Port of tests/model_test_helpers.py.

/// Builds an OpenAI response object for adapter-level tests.
func getResponseObj(_ output: [TResponseOutputItem], responseId: String? = nil, usage: Usage? = nil) -> ResponseObject {
    var response = ResponseObject(
        id: responseId ?? "resp-789",
        output: output,
        usage: [
            "input_tokens": .int(usage?.inputTokens ?? 0),
            "output_tokens": .int(usage?.outputTokens ?? 0),
            "total_tokens": .int(usage?.totalTokens ?? 0),
            "input_tokens_details": [
                "cache_write_tokens": .int(usage?.inputTokensDetails.cacheWriteTokens ?? 0),
                "cached_tokens": .int(usage?.inputTokensDetails.cachedTokens ?? 0),
            ],
            "output_tokens_details": ["reasoning_tokens": .int(usage?.outputTokensDetails.reasoningTokens ?? 0)],
        ],
        model: "test_model"
    )
    response.json["created_at"] = 123
    return response
}

/// Builds an exact normalized stream for tests whose subject is downstream processing.
func getExactOutputStreamStep(_ output: [TResponseOutputItem]) -> ModelStep {
    ModelStep.stream { _ in
        AsyncThrowingStream { continuation in
            for (index, item) in output.enumerated() {
                continuation.yield([
                    "type": "response.output_item.done",
                    "item": .object(item),
                    "output_index": .int(index),
                    "sequence_number": .int(index),
                ])
            }
            continuation.yield([
                "type": "response.completed",
                "response": .object(getResponseObj(output).json),
                "sequence_number": .int(output.count),
            ])
            continuation.finish()
        }
    }
}
