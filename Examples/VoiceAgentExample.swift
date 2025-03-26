import Foundation
import AgentSDK_Swift

/// Demo application showing how to use the voice agent features
@main
struct VoiceAgentExample {
    /// Main entry point
    static func main() async throws {
        guard let apiKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"] else {
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Please set the OPENAI_API_KEY environment variable to your OpenAI API key")
            exit(1)
        }
        
        print("🔊 AgentSDK-Swift Voice Agent Demo")
        print("==================================")
        
        try await runVoiceAgent(apiKey: apiKey)
    }
    
    /// Runs a voice agent example
    /// - Parameter apiKey: OpenAI API key
    static func runVoiceAgent(apiKey: String) async throws {
        // Create an agent with a simple conversation tool
        let agent = createConversationalAgent()
        
        // Create a voice agent with the conversational agent
        let voiceAgent = VoiceAgent(apiKey: apiKey, agent: agent)
        
        // Simulate audio input (in a real app, this would come from the microphone)
        let audioData = simulateAudioInput()
        
        // Process the audio input
        print("\nProcessing audio input...")
        let audioInput = AudioInput(buffer: audioData)
        
        // Process the audio and get the result
        let result = try await voiceAgent.process(audioInput: audioInput)
        
        // Display the result
        print("\nAudio processed. Response:")
        for try await audioChunk in result.audioStream {
            // In a real app, we would play this audio through speakers
            // Here we'll just print the size of each chunk
            print("Audio chunk received: \(audioChunk.count) bytes")
        }
        
        // Demo streaming audio input
        print("\nSimulating streaming audio input...")
        let streamingInput = StreamedAudioInput()
        
        // Start processing the streaming audio
        let streamingResult = try await voiceAgent.process(audioInput: streamingInput)
        
        // Display streaming results
        Task {
            for try await audioChunk in streamingResult.audioStream {
                print("Streaming audio chunk received: \(audioChunk.count) bytes")
            }
        }
        
        // Send audio chunks
        for _ in 1...3 {
            try await Task.sleep(nanoseconds: 1_000_000_000)
            streamingInput.addAudio(simulateAudioInput())
        }
        
        // Complete the stream
        try await Task.sleep(nanoseconds: 1_000_000_000)
        streamingInput.finish()
        
        // Wait for processing to complete
        try await Task.sleep(nanoseconds: 2_000_000_000)
        
        print("\nDemo complete! 👋\n")
    }
    
    /// Creates a conversational agent
    /// - Returns: A configured Agent instance
    static func createConversationalAgent() -> Agent<Void> {
        return Agent<Void>(
            name: "VoiceAssistant",
            instructions: """
            You are a helpful voice assistant.
            Keep your responses concise and conversational.
            Speak naturally as you would in a voice conversation.
            """
        )
    }
    
    /// Simulates audio input for demonstration purposes
    /// - Returns: Simulated audio data
    static func simulateAudioInput() -> Data {
        // In a real application, this would be actual audio data from a microphone
        // For this example, we'll just create some random data
        return Data(repeating: 0, count: 4096)
    }
}