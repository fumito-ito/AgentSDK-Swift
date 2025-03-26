import Foundation

/// Default voice for TTS
public let DEFAULT_VOICE = "ash"

/// OpenAI text-to-speech model
public class OpenAITTSModel: TTSModel {
    /// The model name
    public let modelName: String
    
    /// The OpenAI API client
    private let apiClient: OpenAIAPIClient
    
    /// Create a new OpenAI TTS model
    /// - Parameters:
    ///   - model: The name of the model to use
    ///   - apiClient: The OpenAI API client
    public init(model: String, apiClient: OpenAIAPIClient) {
        self.modelName = model
        self.apiClient = apiClient
    }
    
    /// Run the text-to-speech model
    /// - Parameters:
    ///   - text: The text to convert to audio
    ///   - settings: The settings to use
    /// - Returns: An async stream of audio data
    public func run(text: String, settings: TTSModelSettings) -> AsyncThrowingStream<Data, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let request = SpeechRequest(
                        model: modelName,
                        input: text,
                        voice: settings.voice ?? DEFAULT_VOICE,
                        responseFormat: "pcm",
                        speed: settings.speed,
                        instructions: settings.instructions
                    )
                    
                    let stream = try await apiClient.createSpeechWithStreaming(request: request)
                    
                    for try await chunk in stream {
                        continuation.yield(chunk)
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

/// Request for the OpenAI speech endpoint
public struct SpeechRequest: Encodable {
    /// The model to use
    let model: String
    
    /// The text to generate audio for
    let input: String
    
    /// The voice to use
    let voice: String
    
    /// The format of the response
    let responseFormat: String
    
    /// The speed of the generated audio
    let speed: Float?
    
    /// Instructions for the model
    let instructions: String
    
    enum CodingKeys: String, CodingKey {
        case model, input, voice
        case responseFormat = "response_format"
        case speed
        case instructions = "extra_body"
    }
}