import Foundation
#if canImport(AVFoundation)
import AVFoundation
#endif

/// Settings for a text-to-speech model
public struct TTSModelSettings {
    /// The voice to use for the TTS model
    public var voice: String?
    
    /// Buffer size for audio streaming
    public var bufferSize: Int
    
    /// Instructions for the TTS model
    public var instructions: String
    
    /// Speed of voice (between 0.25 and 4.0)
    public var speed: Float?
    
    public init(
        voice: String? = nil,
        bufferSize: Int = 120,
        instructions: String = "You will receive partial sentences. Do not complete the sentence, just read out the text.",
        speed: Float? = nil
    ) {
        self.voice = voice
        self.bufferSize = bufferSize
        self.instructions = instructions
        self.speed = speed
    }
}

/// Settings for a speech-to-text model
public struct STTModelSettings {
    /// Instructions for the model to follow
    public var prompt: String?
    
    /// The language of the audio input
    public var language: String?
    
    /// Temperature for the model
    public var temperature: Float?
    
    /// Turn detection settings
    public var turnDetection: [String: Any]?
    
    public init(
        prompt: String? = nil,
        language: String? = nil,
        temperature: Float? = nil,
        turnDetection: [String: Any]? = ["type": "semantic_vad"]
    ) {
        self.prompt = prompt
        self.language = language
        self.temperature = temperature
        self.turnDetection = turnDetection
    }
}

/// Protocol for text-to-speech models
public protocol TTSModel {
    /// The name of the TTS model
    var modelName: String { get }
    
    /// Convert text to streaming audio
    /// - Parameters:
    ///   - text: The text to convert to audio
    ///   - settings: Settings for the TTS model
    /// - Returns: An async sequence of audio data
    func run(text: String, settings: TTSModelSettings) async throws -> AsyncThrowingStream<Data, Error>
}

/// Protocol for speech-to-text models
public protocol STTModel {
    /// The name of the STT model
    var modelName: String { get }
    
    /// Transcribe audio to text
    /// - Parameters:
    ///   - input: The audio input to transcribe
    ///   - settings: Settings for the STT model
    /// - Returns: The transcribed text
    func transcribe(input: AudioInput, settings: STTModelSettings) async throws -> String
    
    /// Create a streamed transcription session
    /// - Parameters:
    ///   - input: The streamed audio input
    ///   - settings: Settings for the STT model
    /// - Returns: A streamed transcription session
    func createSession(input: StreamedAudioInput, settings: STTModelSettings) async throws -> StreamedTranscriptionSession
}

/// Protocol for a streamed transcription session
public protocol StreamedTranscriptionSession {
    /// Stream of transcribed text turns
    /// - Returns: An async sequence of transcribed text turns
    func transcribeTurns() -> AsyncThrowingStream<String, Error>
    
    /// Close the session
    func close() async throws
}

/// Protocol for a voice model provider
public protocol VoiceModelProvider {
    /// Get a speech-to-text model by name
    /// - Parameter modelName: The name of the model
    /// - Returns: The speech-to-text model
    func getSTTModel(modelName: String?) -> STTModel
    
    /// Get a text-to-speech model by name
    /// - Parameter modelName: The name of the model
    /// - Returns: The text-to-speech model
    func getTTSModel(modelName: String?) -> TTSModel
}