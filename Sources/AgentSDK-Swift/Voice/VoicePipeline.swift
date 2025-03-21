import Foundation
import AVFoundation

/// A pipeline for processing voice input through an agent and generating voice output
public class VoicePipeline<Context> {
    /// The speech-to-text service for transcribing user audio
    private let speechToTextService: SpeechToTextService
    
    /// The agent for processing the transcribed text
    private let agent: Agent<Context>
    
    /// The text-to-speech service for generating audio responses
    private let textToSpeechService: TextToSpeechService
    
    /// Audio session for managing audio input/output
    private let audioSession = AVAudioSession.sharedInstance()
    
    /// Creates a new voice pipeline with the specified components
    /// - Parameters:
    ///   - speechToTextService: The service for transcribing speech to text
    ///   - agent: The agent for processing the transcribed text
    ///   - textToSpeechService: The service for converting text to speech
    public init(
        speechToTextService: SpeechToTextService,
        agent: Agent<Context>,
        textToSpeechService: TextToSpeechService
    ) {
        self.speechToTextService = speechToTextService
        self.agent = agent
        self.textToSpeechService = textToSpeechService
    }
    
    /// Convenience initializer that creates a pipeline with OpenAI's latest voice models
    /// - Parameters:
    ///   - openAIApiKey: The OpenAI API key
    ///   - agent: The agent for processing the transcribed text
    public convenience init(
        openAIApiKey: String,
        agent: Agent<Context>
    ) {
        // Configure agent to use gpt-4o if not already set
        if agent.modelSettings.modelName != "gpt-4o" {
            agent.modelSettings = ModelSettings(modelName: "gpt-4o")
        }
        
        let sttService = OpenAIWhisperService(apiKey: openAIApiKey, model: "gpt-4o-transcribe")
        let ttsService = OpenAITTSService(apiKey: openAIApiKey, model: "gpt-4o-mini-tts")
        
        self.init(
            speechToTextService: sttService,
            agent: agent,
            textToSpeechService: ttsService
        )
    }
    
    /// Process audio data through the pipeline (STT -> Agent -> TTS)
    /// - Parameters:
    ///   - audioData: The audio data to process
    ///   - context: The context for the agent
    ///   - progressHandler: Optional handler for receiving progress updates
    /// - Returns: Audio data containing the agent's spoken response
    public func process(
        audioData: Data,
        context: Context,
        progressHandler: VoiceProgressHandler? = nil
    ) async throws -> VoicePipelineResult {
        let startTime = Date()
        
        // Step 1: Speech-to-Text
        progressHandler?(.recognizingSpeech)
        let transcriptionStartTime = Date()
        let transcription = try await speechToTextService.transcribe(audioData: audioData)
        let transcriptionEndTime = Date()
        
        // Step 2: Agent Processing
        progressHandler?(.processingWithAgent(transcription))
        let agentStartTime = Date()
        let result = try await AgentRunner.run(
            agent: agent,
            input: transcription,
            context: context
        )
        let agentEndTime = Date()
        
        // Step 3: Text-to-Speech
        progressHandler?(.generatingSpeech(result.finalOutput))
        let ttsStartTime = Date()
        let audioResponse = try await textToSpeechService.synthesize(text: result.finalOutput)
        let ttsEndTime = Date()
        
        // Create metrics
        let metrics = VoicePipelineMetrics(
            totalDuration: ttsEndTime.timeIntervalSince(startTime),
            transcriptionDuration: transcriptionEndTime.timeIntervalSince(transcriptionStartTime),
            agentDuration: agentEndTime.timeIntervalSince(agentStartTime),
            ttsDuration: ttsEndTime.timeIntervalSince(ttsStartTime)
        )
        
        return VoicePipelineResult(
            inputTranscription: transcription,
            agentResponse: result.finalOutput,
            audioResponse: audioResponse,
            messages: result.messages,
            metrics: metrics
        )
    }
    
    /// Process audio input in streaming mode
    /// - Parameters:
    ///   - audioData: The audio data to process
    ///   - context: The context for the agent
    ///   - progressHandler: Optional handler for receiving progress updates
    ///   - audioChunkHandler: Handler for receiving audio chunks as they're generated
    public func processStreamed(
        audioData: Data,
        context: Context,
        progressHandler: VoiceProgressHandler? = nil,
        audioChunkHandler: @escaping (Data) async -> Void
    ) async throws -> VoicePipelineResult {
        let startTime = Date()
        
        // Step 1: Speech-to-Text
        progressHandler?(.recognizingSpeech)
        let transcriptionStartTime = Date()
        let transcription = try await speechToTextService.transcribe(audioData: audioData)
        let transcriptionEndTime = Date()
        
        // Step 2: Agent Processing with Streaming
        progressHandler?(.processingWithAgent(transcription))
        let agentStartTime = Date()
        
        // Buffer for text chunks and collecting messages
        var textBuffer = ""
        var finalOutput = ""
        var messages: [Message] = []
        
        // Use the streaming version of agent runner
        try await AgentRunner.runStreamed(
            agent: agent,
            input: transcription,
            context: context
        ) { chunk in
            textBuffer += chunk
            
            // Check if we have a complete sentence or paragraph
            if textBuffer.contains(where: { $0 == "." || $0 == "?" || $0 == "!" || $0 == "\n" }) {
                // Extract complete sentences
                let components = textBuffer.components(separatedBy: CharacterSet(charactersIn: ".!?\n"))
                
                if components.count > 1 {
                    let completeText = components[0..<components.count-1].joined(separator: ".")
                    
                    if !completeText.isEmpty {
                        // Step 3: Text-to-Speech for this chunk
                        progressHandler?(.generatingSpeech(completeText))
                        
                        do {
                            let audioChunk = try await textToSpeechService.synthesize(text: completeText + ".")
                            await audioChunkHandler(audioChunk)
                        } catch {
                            // Continue even if TTS fails for a chunk
                            print("TTS error for chunk: \(error)")
                        }
                        
                        // Update the buffer with remaining text
                        textBuffer = components.last ?? ""
                        
                        // Add to final output
                        finalOutput += completeText + "."
                    }
                }
            }
        }
        let agentEndTime = Date()
        
        // Process any remaining text
        if !textBuffer.isEmpty {
            // Synthesize remaining text
            progressHandler?(.generatingSpeech(textBuffer))
            
            do {
                let finalAudioChunk = try await textToSpeechService.synthesize(text: textBuffer)
                await audioChunkHandler(finalAudioChunk)
                
                // Add to final output
                finalOutput += textBuffer
            } catch {
                // Continue even if TTS fails for the final chunk
                print("TTS error for final chunk: \(error)")
            }
        }
        
        // Create metrics
        let ttsEndTime = Date()
        let metrics = VoicePipelineMetrics(
            totalDuration: ttsEndTime.timeIntervalSince(startTime),
            transcriptionDuration: transcriptionEndTime.timeIntervalSince(transcriptionStartTime),
            agentDuration: agentEndTime.timeIntervalSince(agentStartTime),
            ttsDuration: ttsEndTime.timeIntervalSince(agentEndTime)
        )
        
        // Create complete audio response (this would ideally be cached during streaming)
        let completeAudioResponse = try await textToSpeechService.synthesize(text: finalOutput)
        
        return VoicePipelineResult(
            inputTranscription: transcription,
            agentResponse: finalOutput,
            audioResponse: completeAudioResponse,
            messages: messages,
            metrics: metrics
        )
    }
    
    /// Start listening for voice input and process it through the pipeline
    /// - Parameters:
    ///   - context: The context for the agent
    ///   - progressHandler: Optional handler for receiving progress updates
    ///   - audioCompletionHandler: Handler for receiving the final audio response
    /// - Returns: A task that can be cancelled to stop the listening process
    public func listen(
        context: Context,
        progressHandler: VoiceProgressHandler? = nil,
        audioCompletionHandler: @escaping (Result<VoicePipelineResult, Error>) -> Void
    ) -> Task<Void, Error> {
        return Task {
            do {
                // Configure audio session
                try audioSession.setCategory(.playAndRecord, mode: .default)
                try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
                
                // Create audio recorder
                let recordingURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("recording.wav")
                let recorder = try AVAudioRecorder(
                    url: recordingURL,
                    settings: [
                        AVFormatIDKey: Int(kAudioFormatLinearPCM),
                        AVSampleRateKey: 44100.0,
                        AVNumberOfChannelsKey: 1,
                        AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
                    ]
                )
                
                // Start recording
                recorder.record()
                
                // Wait for some condition to stop recording
                // For example, silence detection, maximum duration, etc.
                try await Task.sleep(nanoseconds: 10_000_000_000) // 10 seconds timeout
                recorder.stop()
                
                // Get the recorded audio data
                guard let audioData = try? Data(contentsOf: recordingURL) else {
                    throw VoicePipelineError.recordingFailed
                }
                
                // Process the audio through the pipeline
                let result = try await process(
                    audioData: audioData,
                    context: context,
                    progressHandler: progressHandler
                )
                
                // Report completion
                audioCompletionHandler(.success(result))
            } catch {
                audioCompletionHandler(.failure(error))
            }
        }
    }
}

/// Represents the result of processing through a voice pipeline
public struct VoicePipelineResult {
    /// The transcription of the input audio
    public let inputTranscription: String
    
    /// The text response from the agent
    public let agentResponse: String
    
    /// The audio data containing the spoken response
    public let audioResponse: Data
    
    /// The messages exchanged during the agent run
    public let messages: [Message]
    
    /// Metrics about the pipeline execution
    public let metrics: VoicePipelineMetrics
}

/// Metrics about the execution of a voice pipeline
public struct VoicePipelineMetrics {
    /// The total duration of the pipeline execution
    public let totalDuration: TimeInterval
    
    /// The duration of the speech-to-text conversion
    public let transcriptionDuration: TimeInterval
    
    /// The duration of the agent processing
    public let agentDuration: TimeInterval
    
    /// The duration of the text-to-speech conversion
    public let ttsDuration: TimeInterval
}

/// Errors that can occur during voice pipeline execution
public enum VoicePipelineError: Error {
    case recordingFailed
    case sttError(Error)
    case agentError(Error)
    case ttsError(Error)
} 