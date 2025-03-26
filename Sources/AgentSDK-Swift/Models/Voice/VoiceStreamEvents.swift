import Foundation

/// An event from the voice pipeline
public enum VoiceStreamEvent {
    /// Audio data event
    case audio(data: Data)
    
    /// Lifecycle event
    case lifecycle(event: VoiceLifecycleEvent)
    
    /// Error event
    case error(error: Error)
}

/// Lifecycle events for voice processing
public enum VoiceLifecycleEvent: String {
    /// A turn of conversation has started
    case turnStarted = "turn_started"
    
    /// A turn of conversation has ended
    case turnEnded = "turn_ended"
    
    /// The entire session has ended
    case sessionEnded = "session_ended"
}