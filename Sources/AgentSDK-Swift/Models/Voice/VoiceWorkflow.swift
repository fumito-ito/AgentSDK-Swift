import Foundation

/// Protocol for a voice workflow
public protocol VoiceWorkflow {
    /// Run the voice workflow with the given transcription
    /// - Parameter transcription: The transcribed text input
    /// - Returns: An async stream of text output
    func run(transcription: String) -> AsyncThrowingStream<String, Error>
}

/// A simple voice workflow that runs a single agent
public class SingleAgentVoiceWorkflow: VoiceWorkflow {
    /// The agent to run
    private let agent: Agent
    
    /// Input history of the conversation
    private var inputHistory: [ResponseInputItem] = []
    
    /// Create a new single agent voice workflow
    /// - Parameter agent: The agent to run
    public init(agent: Agent) {
        self.agent = agent
    }
    
    /// Run the voice workflow with the given transcription
    /// - Parameter transcription: The transcribed text input
    /// - Returns: An async stream of text output
    public func run(transcription: String) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    // Add the transcription to the input history
                    inputHistory.append(ResponseInputItem(role: .user, content: transcription))
                    
                    // Run the agent
                    let result = try await AgentRunner.runWithStreaming(agent: agent, input: inputHistory)
                    
                    // Stream the text from the result
                    for try await event in result.stream() {
                        // Only yield text output events
                        if case .textDelta(let delta) = event {
                            continuation.yield(delta)
                        }
                    }
                    
                    // Update the input history
                    if let lastOutput = result.output {
                        inputHistory.append(ResponseInputItem(role: .assistant, content: lastOutput.content))
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}