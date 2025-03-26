import Foundation

/// Default STT model name
public let DEFAULT_STT_MODEL = "gpt-4o-transcribe"

/// Default TTS model name
public let DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"

/// OpenAI voice model provider
public class OpenAIVoiceModelProvider: VoiceModelProvider {
    /// The OpenAI API client
    private let apiClient: OpenAIAPIClient
    
    /// Create a new OpenAI voice model provider
    /// - Parameter apiClient: The OpenAI API client
    public init(apiClient: OpenAIAPIClient) {
        self.apiClient = apiClient
    }
    
    /// Get a speech-to-text model by name
    /// - Parameter modelName: The name of the model
    /// - Returns: The speech-to-text model
    public func getSTTModel(modelName: String?) -> STTModel {
        return OpenAISTTModel(model: modelName ?? DEFAULT_STT_MODEL, apiClient: apiClient)
    }
    
    /// Get a text-to-speech model by name
    /// - Parameter modelName: The name of the model
    /// - Returns: The text-to-speech model
    public func getTTSModel(modelName: String?) -> TTSModel {
        return OpenAITTSModel(model: modelName ?? DEFAULT_TTS_MODEL, apiClient: apiClient)
    }
}