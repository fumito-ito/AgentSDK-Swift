import Foundation

/// The result of a voice pipeline, streaming events and audio data as they're generated
public class StreamedAudioResult {
    /// The TTS model used to generate audio
    private let ttsModel: TTSModel
    
    /// Settings for the TTS model
    private let ttsSettings: TTSModelSettings
    
    /// The total text output so far
    public private(set) var totalOutputText: String = ""
    
    /// Buffer for partial text
    private var textBuffer: String = ""
    
    /// Buffer for turn text
    private var turnTextBuffer: String = ""
    
    /// Flag indicating if turn processing has started
    private var startedProcessingTurn: Bool = false
    
    /// Flag indicating if processing is done
    private var doneProcessing: Bool = false
    
    /// Flag indicating if session is completed
    private var completedSession: Bool = false
    
    /// Task continuation for the event stream
    private var continuation: AsyncThrowingStream<VoiceStreamEvent, Error>.Continuation?
    
    /// Create a new streamed audio result
    /// - Parameters:
    ///   - ttsModel: The TTS model to use
    ///   - ttsSettings: The TTS settings to use
    public init(ttsModel: TTSModel, ttsSettings: TTSModelSettings) {
        self.ttsModel = ttsModel
        self.ttsSettings = ttsSettings
    }
    
    /// Stream the events and audio data as they're generated
    /// - Returns: An async stream of voice stream events
    public func stream() -> AsyncThrowingStream<VoiceStreamEvent, Error> {
        return AsyncThrowingStream { continuation in
            self.continuation = continuation
        }
    }
    
    /// Add text to the result
    /// - Parameter text: The text to add
    /// - Returns: Async task result
    public func addText(_ text: String) async throws {
        if !startedProcessingTurn {
            startedProcessingTurn = true
            continuation?.yield(.lifecycle(event: .turnStarted))
        }
        
        textBuffer += text
        totalOutputText += text
        turnTextBuffer += text
        
        // A simple text splitter that processes when we have enough characters
        // This is a simplified version of the Python implementation's sentence splitter
        if textBuffer.count >= 20 {
            let textToProcess = textBuffer
            textBuffer = ""
            
            // Process the text chunk with TTS
            try await processText(textToProcess, finishTurn: false)
        }
    }
    
    /// Signal that a turn is done
    /// - Returns: Async task result
    public func turnDone() async throws {
        if !textBuffer.isEmpty {
            try await processText(textBuffer, finishTurn: true)
            textBuffer = ""
        }
        
        doneProcessing = true
        turnTextBuffer = ""
        startedProcessingTurn = false
    }
    
    /// Signal that the session is done
    /// - Returns: Async task result
    public func done() async throws {
        completedSession = true
        continuation?.yield(.lifecycle(event: .sessionEnded))
        continuation?.finish()
    }
    
    /// Process text through the TTS model
    /// - Parameters:
    ///   - text: Text to process
    ///   - finishTurn: Whether this is the end of a turn
    private func processText(_ text: String, finishTurn: Bool) async throws {
        do {
            let audioStream = try await ttsModel.run(text: text, settings: ttsSettings)
            
            var buffer = Data()
            
            for try await chunk in audioStream {
                buffer.append(chunk)
                
                // When buffer reaches sufficient size, send it
                if buffer.count >= ttsSettings.bufferSize {
                    continuation?.yield(.audio(data: buffer))
                    buffer = Data()
                }
            }
            
            // Send any remaining buffer
            if !buffer.isEmpty {
                continuation?.yield(.audio(data: buffer))
            }
            
            if finishTurn {
                continuation?.yield(.lifecycle(event: .turnEnded))
            }
        } catch {
            continuation?.yield(.error(error: error))
            throw error
        }
    }
}