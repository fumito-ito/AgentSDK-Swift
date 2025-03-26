import Foundation

/// An opinionated voice agent pipeline
public class VoicePipeline {
    /// The voice workflow to run
    private let workflow: VoiceWorkflow
    
    /// The STT model to use
    private var sttModel: STTModel?
    
    /// The TTS model to use
    private var ttsModel: TTSModel?
    
    /// The name of the STT model
    private let sttModelName: String?
    
    /// The name of the TTS model
    private let ttsModelName: String?
    
    /// The pipeline configuration
    private let config: VoicePipelineConfig
    
    /// Create a new voice pipeline
    /// - Parameters:
    ///   - workflow: The workflow to run
    ///   - sttModel: The speech-to-text model to use
    ///   - ttsModel: The text-to-speech model to use
    ///   - config: The pipeline configuration
    public init(
        workflow: VoiceWorkflow,
        sttModel: (any STTModel)? = nil,
        ttsModel: (any TTSModel)? = nil,
        sttModelName: String? = nil,
        ttsModelName: String? = nil,
        config: VoicePipelineConfig
    ) {
        self.workflow = workflow
        self.sttModel = sttModel
        self.ttsModel = ttsModel
        self.sttModelName = sttModelName
        self.ttsModelName = ttsModelName
        self.config = config
    }
    
    /// Run the voice pipeline
    /// - Parameter audioInput: The audio input to process
    /// - Returns: A streamed audio result
    public func run(audioInput: AudioInput) async throws -> StreamedAudioResult {
        return try await runSingleTurn(audioInput: audioInput)
    }
    
    /// Run the voice pipeline with streamed audio
    /// - Parameter audioInput: The streamed audio input to process
    /// - Returns: A streamed audio result
    public func run(audioInput: StreamedAudioInput) async throws -> StreamedAudioResult {
        return try await runMultiTurn(audioInput: audioInput)
    }
    
    /// Get the TTS model, initializing it if needed
    /// - Returns: The TTS model
    private func getTTSModel() -> TTSModel {
        if ttsModel == nil {
            ttsModel = config.modelProvider.getTTSModel(modelName: ttsModelName)
        }
        return ttsModel!
    }
    
    /// Get the STT model, initializing it if needed
    /// - Returns: The STT model
    private func getSTTModel() -> STTModel {
        if sttModel == nil {
            sttModel = config.modelProvider.getSTTModel(modelName: sttModelName)
        }
        return sttModel!
    }
    
    /// Process a single audio input
    /// - Parameter audioInput: The audio input to process
    /// - Returns: The transcribed text
    private func processAudioInput(_ audioInput: AudioInput) async throws -> String {
        let model = getSTTModel()
        return try await model.transcribe(input: audioInput, settings: config.sttSettings)
    }
    
    /// Run the voice pipeline for a single turn
    /// - Parameter audioInput: The audio input to process
    /// - Returns: A streamed audio result
    private func runSingleTurn(audioInput: AudioInput) async throws -> StreamedAudioResult {
        let inputText = try await processAudioInput(audioInput)
        let output = StreamedAudioResult(ttsModel: getTTSModel(), ttsSettings: config.ttsSettings)
        
        Task {
            do {
                let textStream = workflow.run(transcription: inputText)
                
                for try await textEvent in textStream {
                    try await output.addText(textEvent)
                }
                
                try await output.turnDone()
                try await output.done()
            } catch {
                print("Error processing single turn: \(error)")
            }
        }
        
        return output
    }
    
    /// Run the voice pipeline for multiple turns
    /// - Parameter audioInput: The streamed audio input to process
    /// - Returns: A streamed audio result
    private func runMultiTurn(audioInput: StreamedAudioInput) async throws -> StreamedAudioResult {
        let output = StreamedAudioResult(ttsModel: getTTSModel(), ttsSettings: config.ttsSettings)
        
        let transcriptionSession = try await getSTTModel().createSession(
            input: audioInput,
            settings: config.sttSettings
        )
        
        Task {
            do {
                let turns = transcriptionSession.transcribeTurns()
                
                for try await inputText in turns {
                    let result = workflow.run(transcription: inputText)
                    
                    for try await textEvent in result {
                        try await output.addText(textEvent)
                    }
                    
                    try await output.turnDone()
                }
            } catch {
                print("Error processing turns: \(error)")
            } finally {
                try? await transcriptionSession.close()
                try? await output.done()
            }
        }
        
        return output
    }
}