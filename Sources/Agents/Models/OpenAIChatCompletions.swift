import Foundation

// Port of src/agents/models/openai_chatcompletions.py.

/// A `Model` backed by the OpenAI Chat Completions API.
public final class OpenAIChatCompletionsModel: Model, @unchecked Sendable {
    public let model: String
    let client: OpenAIClient
    let strictFeatureValidation: Bool
    let bufferStreamedToolCalls: Bool

    public init(model: String, openAIClient: OpenAIClient, strictFeatureValidation: Bool = false, bufferStreamedToolCalls: Bool = false) {
        self.model = model
        self.client = openAIClient
        self.strictFeatureValidation = strictFeatureValidation
        self.bufferStreamedToolCalls = bufferStreamedToolCalls
    }

    public func getRetryAdvice(_ request: ModelRetryAdviceRequest) -> ModelRetryAdvice? {
        getOpenAIRetryAdvice(request)
    }

    public func getResponse(_ request: ModelRequest) async throws -> ModelResponse {
        throw UserError("OpenAIChatCompletionsModel is not ported yet")
    }

    public func streamResponse(_ request: ModelRequest) -> AsyncThrowingStream<TResponseStreamEvent, any Error> {
        AsyncThrowingStream { $0.finish(throwing: UserError("OpenAIChatCompletionsModel is not ported yet")) }
    }
}
