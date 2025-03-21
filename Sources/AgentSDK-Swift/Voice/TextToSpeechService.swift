import Foundation
import Alamofire
import AVFoundation

/// Protocol for services that convert text to speech audio
public protocol TextToSpeechService {
    /// Synthesizes text into audio data
    /// - Parameter text: The text to synthesize
    /// - Returns: The synthesized audio data
    func synthesize(text: String) async throws -> Data
}

/// Implementation of TextToSpeechService using OpenAI's TTS models
public class OpenAITTSService: TextToSpeechService {
    /// The base URL for the OpenAI API
    private let baseURL = "https://api.openai.com/v1/audio/speech"
    
    /// The API key for the OpenAI API
    private let apiKey: String
    
    /// The model to use for speech synthesis (default is "tts-1")
    private let model: String
    
    /// The voice to use for speech synthesis (default is "alloy")
    private let voice: String
    
    /// Creates a new OpenAI TTS service
    /// - Parameters:
    ///   - apiKey: The API key for the OpenAI API
    ///   - model: The model to use for speech synthesis (default is "tts-1")
    ///   - voice: The voice to use for speech synthesis (default is "alloy")
    public init(apiKey: String, model: String = "tts-1", voice: String = "alloy") {
        self.apiKey = apiKey
        self.model = model
        self.voice = voice
    }
    
    /// Synthesizes text into audio data using OpenAI's TTS model
    /// - Parameter text: The text to synthesize
    /// - Returns: The synthesized audio data
    public func synthesize(text: String) async throws -> Data {
        let headers: HTTPHeaders = [
            "Authorization": "Bearer \(apiKey)",
            "Content-Type": "application/json"
        ]
        
        let parameters: [String: Any] = [
            "model": model,
            "input": text,
            "voice": voice
        ]
        
        return try await withCheckedThrowingContinuation { continuation in
            AF.request(
                baseURL,
                method: .post,
                parameters: parameters,
                encoding: JSONEncoding.default,
                headers: headers
            )
            .validate()
            .responseData { response in
                switch response.result {
                case .success(let data):
                    continuation.resume(returning: data)
                case .failure(let error):
                    continuation.resume(throwing: TextToSpeechError.apiError(error))
                }
            }
        }
    }
}

/// Implementation of TextToSpeechService using Apple's AVSpeechSynthesizer
public class AppleTTSService: TextToSpeechService {
    /// The speech synthesizer
    private let synthesizer = AVSpeechSynthesizer()
    
    /// The voice to use (default is the system voice)
    private let voice: AVSpeechSynthesisVoice?
    
    /// Creates a new Apple TTS service
    /// - Parameter voice: The voice to use (default is the system voice)
    public init(voice: AVSpeechSynthesisVoice? = nil) {
        self.voice = voice
    }
    
    /// Synthesizes text into audio data using Apple's AVSpeechSynthesizer
    /// - Parameter text: The text to synthesize
    /// - Returns: The synthesized audio data
    public func synthesize(text: String) async throws -> Data {
        // Note: This implementation is simplified. In a real implementation,
        // you would need to handle the audio session and capture the audio output.
        
        // For now, use a simple approach to capture audio to a file
        let outputURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("speech.wav")
        
        // Create utterance
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice ?? AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.volume = 1.0
        
        // Synchronize synthesis using continuation
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            // Set up audio session for playback
            do {
                let audioSession = AVAudioSession.sharedInstance()
                try audioSession.setCategory(.playback, mode: .default)
                try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            } catch {
                continuation.resume(throwing: TextToSpeechError.audioSessionError(error))
                return
            }
            
            // Delegate to handle completion
            let delegate = SpeechDelegate {
                continuation.resume()
            }
            synthesizer.delegate = delegate
            
            // Start speaking
            synthesizer.speak(utterance)
        }
        
        // In a real implementation, we would capture the audio output
        // For now, return empty data as a placeholder
        // A real implementation would require additional work to capture the audio output
        return Data()
    }
    
    /// Private delegate to handle speech synthesizer events
    private class SpeechDelegate: NSObject, AVSpeechSynthesizerDelegate {
        private let completionHandler: () -> Void
        
        init(completionHandler: @escaping () -> Void) {
            self.completionHandler = completionHandler
            super.init()
        }
        
        func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
            completionHandler()
        }
    }
}

/// Errors that can occur during text-to-speech processing
public enum TextToSpeechError: Error {
    case apiError(Error)
    case synthesisError
    case audioSessionError(Error)
} 