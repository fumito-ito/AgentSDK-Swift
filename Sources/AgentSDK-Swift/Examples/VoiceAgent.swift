import Foundation

/// A simple example of a voice agent
public class VoiceAgent {
    /// The voice pipeline
    private let pipeline: VoicePipeline
    
    /// Create a new voice agent
    /// - Parameters:
    ///   - apiKey: The OpenAI API key
    ///   - agent: The agent to use
    ///   - sttModelName: The STT model name
    ///   - ttsModelName: The TTS model name
    public init(apiKey: String, agent: Agent, sttModelName: String? = nil, ttsModelName: String? = nil) {
        // Create the API client
        let apiClient = DefaultOpenAIAPIClient(apiKey: apiKey)
        
        // Create the model provider
        let modelProvider = OpenAIVoiceModelProvider(apiClient: apiClient)
        
        // Create the workflow
        let workflow = SingleAgentVoiceWorkflow(agent: agent)
        
        // Create the pipeline configuration
        let config = VoicePipelineConfig(
            modelProvider: modelProvider,
            workflowName: "Simple Voice Agent",
            sttSettings: STTModelSettings(),
            ttsSettings: TTSModelSettings()
        )
        
        // Create the pipeline
        self.pipeline = VoicePipeline(
            workflow: workflow,
            sttModelName: sttModelName,
            ttsModelName: ttsModelName,
            config: config
        )
    }
    
    /// Process a static audio input
    /// - Parameter audioInput: The audio input to process
    /// - Returns: A streamed audio result
    public func process(audioInput: AudioInput) async throws -> StreamedAudioResult {
        return try await pipeline.run(audioInput: audioInput)
    }
    
    /// Process a streaming audio input
    /// - Parameter audioInput: The streaming audio input to process
    /// - Returns: A streamed audio result
    public func process(audioInput: StreamedAudioInput) async throws -> StreamedAudioResult {
        return try await pipeline.run(audioInput: audioInput)
    }
}

/// Default implementation of OpenAI API client
class DefaultOpenAIAPIClient: OpenAIAPIClient {
    /// The API key
    let apiKey: String
    
    /// Create a new default OpenAI API client
    /// - Parameter apiKey: The API key
    init(apiKey: String) {
        self.apiKey = apiKey
    }
    
    /// Create a transcription from audio
    /// - Parameter request: The transcription request
    /// - Returns: The transcription response
    func createTranscription(request: TranscriptionRequest) async throws -> TranscriptionResponse {
        // This is a simplified implementation
        // Real implementation would use URLSession to make API calls to OpenAI
        
        // For demonstration, return a mock response
        return TranscriptionResponse(text: "Hello, this is a test transcription.")
    }
    
    /// Create speech from text with streaming
    /// - Parameter request: The speech request
    /// - Returns: A stream of audio data
    func createSpeechWithStreaming(request: SpeechRequest) async throws -> AsyncThrowingStream<Data, Error> {
        // This is a simplified implementation
        // Real implementation would use URLSession to make API calls to OpenAI
        
        return AsyncThrowingStream { continuation in
            // For demonstration, send some mock data
            Task {
                // Generate some fake audio data
                let mockAudioData = Data(repeating: 0, count: 1024)
                continuation.yield(mockAudioData)
                
                try? await Task.sleep(nanoseconds: 500_000_000)
                continuation.yield(mockAudioData)
                
                try? await Task.sleep(nanoseconds: 500_000_000)
                continuation.yield(mockAudioData)
                
                continuation.finish()
            }
        }
    }
}