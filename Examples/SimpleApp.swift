import Foundation
import AgentSDK_Swift

/// Simple demo app showing basic usage of AgentSDK-Swift
@main
struct SimpleApp {
    /// Main entry point
    static func main() async throws {
        guard let apiKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"] else {
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Please set the OPENAI_API_KEY environment variable to your OpenAI API key")
            exit(1)
        }

        print("🤖 AgentSDK-Swift Simple Demo")
        print("==============================")

        try await runSimpleAgent(apiKey: apiKey)
    }

    /// Runs a simple agent example
    /// - Parameter apiKey: OpenAI API key
    static func runSimpleAgent(apiKey: String) async throws {
        // Register models
        await ModelProvider.shared.registerOpenAIModels(apiKey: apiKey)

        // Create a tool that calculates a sum
        let calculateTool = Tool<Void>(
            name: "calculateSum",
            description: "Calculate the sum of two numbers",
            parameters: [
                Tool.Parameter(
                    name: "a",
                    description: "First number",
                    type: .number
                ),
                Tool.Parameter(
                    name: "b",
                    description: "Second number",
                    type: .number
                )
            ],
            execute: { parameters, _ in
                guard let a = parameters["a"] as? Double,
                      let b = parameters["b"] as? Double else {
                    return "Invalid numbers provided"
                }

                let sum = a + b
                return "The sum of \(a) and \(b) is \(sum)"
            }
        )

        // Create agent with the calculation tool
        let agent = Agent<Void>(
            name: "CalculatorAssistant",
            instructions: """
            You are a helpful assistant that can perform math calculations.
            When asked about calculations, use the calculateSum tool to add numbers together.
            """
        ).addTool(calculateTool)

        // Input with streaming
        print("\nSending query: What is 42 + 17?")

        _ = try await AgentRunner.runStreamed(
            agent: agent,
            input: "What is 42 + 17?",
            context: ()
        ) { content in
            print(content, terminator: "")
        }

        print("\n\nDemo complete! 👋\n")
    }
}
