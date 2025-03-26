import Foundation

/// OpenAI API client interface for voice models
public protocol OpenAIAPIClient {
    /// Create a transcription from audio
    /// - Parameter request: The transcription request
    /// - Returns: The transcription response
    func createTranscription(request: TranscriptionRequest) async throws -> TranscriptionResponse
    
    /// Create speech from text with streaming
    /// - Parameter request: The speech request
    /// - Returns: A stream of audio data
    func createSpeechWithStreaming(request: SpeechRequest) async throws -> AsyncThrowingStream<Data, Error>
    
    /// The API key
    var apiKey: String { get }
}