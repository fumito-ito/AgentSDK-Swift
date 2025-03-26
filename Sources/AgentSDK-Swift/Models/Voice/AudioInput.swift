import Foundation

/// Default sample rate for audio processing
public let DEFAULT_SAMPLE_RATE = 24000

/// Represents static audio data to be used as input
public class AudioInput {
    /// The raw audio data
    public let buffer: Data
    
    /// The sample rate of the audio data
    public let frameRate: Int
    
    /// The sample width of the audio data
    public let sampleWidth: Int
    
    /// The number of channels in the audio data
    public let channels: Int
    
    /// Create a new audio input
    /// - Parameters:
    ///   - buffer: The raw audio data
    ///   - frameRate: The sample rate of the audio data (default: 24000)
    ///   - sampleWidth: The sample width of the audio data (default: 2)
    ///   - channels: The number of channels in the audio data (default: 1)
    public init(
        buffer: Data,
        frameRate: Int = DEFAULT_SAMPLE_RATE,
        sampleWidth: Int = 2,
        channels: Int = 1
    ) {
        self.buffer = buffer
        self.frameRate = frameRate
        self.sampleWidth = sampleWidth
        self.channels = channels
    }
    
    /// Convert the audio data to a base64 encoded string
    /// - Returns: A base64 encoded string of the audio data
    public func toBase64() -> String {
        return buffer.base64EncodedString()
    }
    
    /// Create a WAV format audio file from the buffer
    /// - Returns: WAV format audio data
    public func toAudioFile() -> Data {
        // This is a simplified implementation
        // Real implementation would need to properly format WAV headers
        // For now, just returning the raw buffer
        return buffer
    }
}

/// Represents a stream of audio data that can be added to over time
public class StreamedAudioInput {
    /// Continuation for providing audio data
    private var continuation: AsyncStream<Data>.Continuation?
    
    /// The stream of audio data
    public let stream: AsyncStream<Data>
    
    /// Create a new streamed audio input
    public init() {
        var continuation: AsyncStream<Data>.Continuation!
        self.stream = AsyncStream<Data> { cont in
            continuation = cont
        }
        self.continuation = continuation
    }
    
    /// Add audio data to the stream
    /// - Parameter audio: The audio data to add
    public func addAudio(_ audio: Data) {
        continuation?.yield(audio)
    }
    
    /// Complete the stream
    public func finish() {
        continuation?.finish()
    }
}