import Foundation
import AVFoundation

/// Represents a voice-enabled agent that can handle speech input and output
public class VoiceAgent<Context> {
    /// The underlying text agent used for processing
    private let agent: Agent<Context>
    
    /// The speech-to-text service for transcribing user audio
    private let speechToTextService: SpeechToTextService
    
    /// The text-to-speech service for generating audio responses
    private let textToSpeechService: TextToSpeechService
    
    /// Audio session for managing audio input/output
    private let audioSession = AVAudioSession.sharedInstance()
    
    /// Create a new voice agent with the specified configuration
    /// - Parameters:
    ///   - agent: The underlying text agent
    ///   - speechToTextService: The service for transcribing speech to text
    ///   - textToSpeechService: The service for converting text to speech
    public init(
        agent: Agent<Context>,
        speechToTextService: SpeechToTextService,
        textToSpeechService: TextToSpeechService
    ) {
        self.agent = agent
        self.speechToTextService = speechToTextService
        self.textToSpeechService = textToSpeechService
    }
    
    /// Process voice input and return voice output
    /// - Parameters:
    ///   - audioData: The audio data containing the user's speech
    ///   - context: The context for the agent
    ///   - progressHandler: Optional handler for receiving progress updates
    /// - Returns: Audio data containing the agent's spoken response
    public func process(
        audioData: Data,
        context: Context,
        progressHandler: VoiceProgressHandler? = nil
    ) async throws -> Data {
        // Report progress: Starting speech recognition
        progressHandler?(.recognizingSpeech)
        
        // 1. Convert speech to text
        let transcription = try await speechToTextService.transcribe(audioData: audioData)
        
        // Report progress: Processing with agent
        progressHandler?(.processingWithAgent(transcription))
        
        // 2. Process the transcription with the agent
        let result = try await AgentRunner.run(
            agent: agent,
            input: transcription,
            context: context
        )
        
        // Report progress: Generating speech
        progressHandler?(.generatingSpeech(result.finalOutput))
        
        // 3. Convert the result to speech
        return try await textToSpeechService.synthesize(text: result.finalOutput)
    }
    
    /// Process voice input and stream the voice output
    /// - Parameters:
    ///   - audioData: The audio data containing the user's speech
    ///   - context: The context for the agent
    ///   - progressHandler: Optional handler for receiving progress updates
    ///   - audioChunkHandler: Handler for receiving audio chunks as they're generated
    public func processStreamed(
        audioData: Data,
        context: Context,
        progressHandler: VoiceProgressHandler? = nil,
        audioChunkHandler: @escaping (Data) async -> Void
    ) async throws {
        // Report progress: Starting speech recognition
        progressHandler?(.recognizingSpeech)
        
        // 1. Convert speech to text
        let transcription = try await speechToTextService.transcribe(audioData: audioData)
        
        // Report progress: Processing with agent
        progressHandler?(.processingWithAgent(transcription))
        
        // 2. Process the transcription with the agent and stream the result
        var textBuffer = ""
        
        try await AgentRunner.runStreamed(
            agent: agent,
            input: transcription,
            context: context
        ) { chunk in
            textBuffer += chunk
            
            // Check if we have enough text to generate speech
            if textBuffer.contains(where: { $0 == "." || $0 == "?" || $0 == "!" || $0 == "\n" }) {
                // Extract complete sentences
                let components = textBuffer.components(separatedBy: CharacterSet(charactersIn: ".!?\n"))
                
                if components.count > 1 {
                    let completeText = components[0..<components.count-1].joined(separator: ".")
                    if !completeText.isEmpty {
                        // Report progress: Generating speech
                        progressHandler?(.generatingSpeech(completeText))
                        
                        // 3. Convert the text chunk to speech
                        if let audioChunk = try? await textToSpeechService.synthesize(text: completeText + ".") {
                            // Send the audio chunk to the handler
                            await audioChunkHandler(audioChunk)
                        }
                        
                        // Update the buffer with the remaining text
                        textBuffer = components.last ?? ""
                    }
                }
            }
        }
        
        // Process any remaining text
        if !textBuffer.isEmpty {
            // Report progress: Generating speech for remaining text
            progressHandler?(.generatingSpeech(textBuffer))
            
            // Convert the remaining text to speech
            if let audioChunk = try? await textToSpeechService.synthesize(text: textBuffer) {
                // Send the final audio chunk to the handler
                await audioChunkHandler(audioChunk)
            }
        }
    }
    
    /// Start recording audio from the microphone
    /// - Returns: A tuple containing the recording task and a future that will be fulfilled with the recorded audio data
    public func startRecording() async throws -> (task: Task<Void, Error>, audioFuture: Task<Data, Error>) {
        // Configure audio session for recording
        try audioSession.setCategory(.playAndRecord, mode: .default)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        // Create a recorder
        let recorder = try AVAudioRecorder(
            url: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("recording.wav"),
            settings: [
                AVFormatIDKey: Int(kAudioFormatLinearPCM),
                AVSampleRateKey: 44100.0,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ]
        )
        
        // Start recording
        recorder.record()
        
        // Create a future for the recorded audio data
        let audioFuture = Task<Data, Error> {
            // This will be resolved when recording stops
            try await withCheckedThrowingContinuation { continuation in
                recorder.delegate = AVAudioRecorderDelegate(
                    finishedRecording: { success in
                        if success, let data = try? Data(contentsOf: recorder.url) {
                            continuation.resume(returning: data)
                        } else {
                            continuation.resume(throwing: VoiceAgentError.recordingFailed)
                        }
                    }
                )
            }
        }
        
        // Create a task for the recording process
        let recordingTask = Task {
            // Wait for some condition to stop recording
            // For example, silence detection, maximum duration, etc.
            try await Task.sleep(nanoseconds: 10_000_000_000) // 10 seconds timeout
            recorder.stop()
        }
        
        return (task: recordingTask, audioFuture: audioFuture)
    }
    
    /// Errors that can occur during voice agent operation
    public enum VoiceAgentError: Error {
        case recordingFailed
        case playbackFailed
        case audioSessionError(Error)
    }
}

/// Progress updates during voice agent processing
public enum VoiceProgress {
    case recognizingSpeech
    case processingWithAgent(String)
    case generatingSpeech(String)
}

/// Type alias for voice progress handler
public typealias VoiceProgressHandler = (VoiceProgress) -> Void

/// Simple AVAudioRecorderDelegate implementation
private class AVAudioRecorderDelegate: NSObject, AVFoundation.AVAudioRecorderDelegate {
    private let finishedRecording: (Bool) -> Void
    
    init(finishedRecording: @escaping (Bool) -> Void) {
        self.finishedRecording = finishedRecording
        super.init()
    }
    
    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        finishedRecording(flag)
    }
} 