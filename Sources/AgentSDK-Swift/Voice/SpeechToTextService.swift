import Foundation
import Alamofire

/// Protocol for services that convert speech audio to text
public protocol SpeechToTextService {
    /// Transcribes audio data to text
    /// - Parameter audioData: The audio data to transcribe
    /// - Returns: The transcribed text
    func transcribe(audioData: Data) async throws -> String
}

/// Implementation of SpeechToTextService using OpenAI's Whisper model
public class OpenAIWhisperService: SpeechToTextService {
    /// The base URL for the OpenAI API
    private let baseURL = "https://api.openai.com/v1/audio/transcriptions"
    
    /// The API key for the OpenAI API
    private let apiKey: String
    
    /// The model to use for transcription (default is "whisper-1")
    private let model: String
    
    /// Creates a new OpenAI Whisper service
    /// - Parameters:
    ///   - apiKey: The API key for the OpenAI API
    ///   - model: The model to use for transcription (default is "whisper-1")
    public init(apiKey: String, model: String = "whisper-1") {
        self.apiKey = apiKey
        self.model = model
    }
    
    /// Transcribes audio data to text using OpenAI's Whisper model
    /// - Parameter audioData: The audio data to transcribe
    /// - Returns: The transcribed text
    public func transcribe(audioData: Data) async throws -> String {
        // Create a multipart form with the audio data and model
        let headers: HTTPHeaders = [
            "Authorization": "Bearer \(apiKey)",
            "Content-Type": "multipart/form-data"
        ]
        
        // Create a temporary file for the audio data
        let audioURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("audio.wav")
        try audioData.write(to: audioURL)
        
        // Create the multipart form data
        return try await withCheckedThrowingContinuation { continuation in
            AF.upload(
                multipartFormData: { multipartFormData in
                    multipartFormData.append(audioURL, withName: "file")
                    multipartFormData.append(Data(self.model.utf8), withName: "model")
                },
                to: baseURL,
                headers: headers
            )
            .validate()
            .responseDecodable(of: WhisperResponse.self) { response in
                switch response.result {
                case .success(let whisperResponse):
                    continuation.resume(returning: whisperResponse.text)
                case .failure(let error):
                    continuation.resume(throwing: SpeechToTextError.apiError(error))
                }
            }
        }
    }
    
    /// Response object for Whisper API
    private struct WhisperResponse: Decodable {
        let text: String
    }
}

/// Implementation of SpeechToTextService using Apple's Speech framework
public class AppleSpeechService: SpeechToTextService {
    /// Creates a new Apple Speech service
    public init() {}
    
    /// Transcribes audio data to text using Apple's Speech framework
    /// - Parameter audioData: The audio data to transcribe
    /// - Returns: The transcribed text
    public func transcribe(audioData: Data) async throws -> String {
        // Note: This is a placeholder. In a real implementation, you would use
        // Apple's Speech framework (Speech.framework) which requires more complex
        // setup and usage than can be shown here. The actual implementation would
        // involve SFSpeechRecognizer, SFSpeechRecognitionRequest, etc.
        
        // For now, throw an error indicating this is not implemented
        throw SpeechToTextError.notImplemented("Apple Speech service is not yet implemented")
    }
}

/// Errors that can occur during speech-to-text processing
public enum SpeechToTextError: Error {
    case apiError(Error)
    case invalidAudioFormat
    case transcriptionFailed
    case notImplemented(String)
} 