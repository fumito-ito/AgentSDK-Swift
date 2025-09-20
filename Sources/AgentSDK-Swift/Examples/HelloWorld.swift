import Foundation

/// Example showing basic agent usage with a tool
public struct HelloWorldExample {
    /// Runs the hello world example
    /// - Parameter apiKey: The OpenAI API key
    public static func run(apiKey: String) async throws {
        // Register OpenAI models
        await ModelProvider.shared.registerOpenAIModels(apiKey: apiKey)

        // Create a tool that returns the current time
        let currentTimeTool = Tool<Void>(
            name: "getCurrentTime",
            description: "Get the current time",
            parameters: [],
            availability: .always,
            execute: { _, _ in
                let formatter = DateFormatter()
                formatter.timeStyle = .medium
                formatter.dateStyle = .medium
                return formatter.string(from: Date())
            }
        )

        // Create an agent with the current time tool
        let agent = Agent<Void>(
            name: "TimeAssistant",
            instructions: """
            You are a helpful assistant that can tell users the current time.
            When asked about the time, use the getCurrentTime tool to provide an accurate response.
            """
        ).addTool(currentTimeTool)

        // Run the agent
        print("Running agent...")

        let result = try await AgentRunner.run(
            agent: agent,
            input: "What time is it right now?",
            context: ()
        )

        // Print the result
        print("Agent response:")
        print(result.finalOutput)
    }

    /// Runs the hello world example with streaming
    /// - Parameter apiKey: The OpenAI API key
    public static func runStreamed(apiKey: String) async throws {
        // Register OpenAI models
        await ModelProvider.shared.registerOpenAIModels(apiKey: apiKey)

        // Create a tool that returns the current time
        let currentTimeTool = Tool<Void>(
            name: "getCurrentTime",
            description: "Get the current time",
            parameters: [],
            execute: { _, _ in
                let formatter = DateFormatter()
                formatter.timeStyle = .medium
                formatter.dateStyle = .medium
                return formatter.string(from: Date())
            }
        )

        // Create an agent with the current time tool
        let agent = Agent<Void>(
            name: "TimeAssistant",
            instructions: """
            You are a helpful assistant that can tell users the current time.
            When asked about the time, use the getCurrentTime tool to provide an accurate response.
            """
        ).addTool(currentTimeTool)

        // Run the agent with streaming
        print("Running agent with streaming...")

        _ = try await AgentRunner.runStreamed(
            agent: agent,
            input: "What time is it right now?",
            context: ()
        ) { content in
            // Print each content chunk as it arrives
            print(content, terminator: "")
        }

        // Print completion message
        print("\nAgent response complete.")
    }
}
