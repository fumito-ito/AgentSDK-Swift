import Foundation

/// Configuration for a voice pipeline
public struct VoicePipelineConfig {
    /// The voice model provider to use
    public var modelProvider: VoiceModelProvider
    
    /// The name of the workflow
    public var workflowName: String
    
    /// A unique group ID for the session
    public var groupId: String
    
    /// STT model settings
    public var sttSettings: STTModelSettings
    
    /// TTS model settings
    public var ttsSettings: TTSModelSettings
    
    /// Create a new voice pipeline configuration
    /// - Parameters:
    ///   - modelProvider: The voice model provider to use
    ///   - workflowName: The name of the workflow
    ///   - groupId: A unique group ID for the session
    ///   - sttSettings: STT model settings
    ///   - ttsSettings: TTS model settings
    public init(
        modelProvider: VoiceModelProvider,
        workflowName: String = "Voice Agent",
        groupId: String = UUID().uuidString,
        sttSettings: STTModelSettings = STTModelSettings(),
        ttsSettings: TTSModelSettings = TTSModelSettings()
    ) {
        self.modelProvider = modelProvider
        self.workflowName = workflowName
        self.groupId = groupId
        self.sttSettings = sttSettings
        self.ttsSettings = ttsSettings
    }
}