# AgentSDK-Swift

A Swift implementation of the OpenAI Agents SDK, allowing you to build AI agent applications with tools, guardrails, and multi-agent workflows.

## Status

🚧 **Early Development** - This project is a Swift port of the [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-py) and is currently in early development.

## Features

- 🤖 Create AI agents with custom instructions and tools
- 🛠️ Define and use tools as Swift functions
- 🔄 Hand off between multiple agents for complex workflows
- 🛡️ Apply guardrails to ensure safe and high-quality outputs
- 📊 Stream responses in real-time
- 📝 Support for OpenAI's latest models 
- 🔊 Voice agent support with speech-to-text and text-to-speech

## Requirements

- Swift 6.0+
- macOS 13.0+ / iOS 16.0+
- OpenAI API Key

## Installation

### Swift Package Manager

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/fumito-ito/AgentSDK-Swift.git", from: "0.1.0")
]
```

## Quick Start

Here's a simple example to get you started:

```swift
import AgentSDK_Swift

// Register OpenAI models
ModelProvider.shared.registerOpenAIModels(apiKey: "your-api-key-here")

// Create a tool
let weatherTool = Tool<Void>(
    name: "getWeather",
    description: "Get the current weather for a location",
    parameters: [
        Tool.Parameter(
            name: "location",
            description: "The location to get weather for",
            type: .string
        )
    ],
    execute: { params, _ in
        let location = params["location"] as? String ?? "Unknown"
        return "It's sunny and 72°F in \(location)"
    }
)

// Create an agent
let agent = Agent<Void>(
    name: "WeatherAssistant",
    instructions: "You are a helpful weather assistant."
).addTool(weatherTool)

// Run the agent
Task {
    do {
        let result = try await AgentRunner.run(
            agent: agent,
            input: "What's the weather like in San Francisco?",
            context: ()
        )
        
        print(result.finalOutput)
    } catch {
        print("Error: \(error)")
    }
}
```

## Advanced Usage

### Voice Agents

```swift
import AgentSDK_Swift

// Create a simple agent
let agent = Agent(
    modelSettings: ModelSettings(
        provider: .openAI,
        model: "gpt-4o",
        apiKey: "your-api-key-here"
    ),
    systemPrompt: "You are a helpful voice assistant."
)

// Create a voice agent
let voiceAgent = VoiceAgent(
    apiKey: "your-api-key-here",
    agent: agent,
    sttModelName: "gpt-4o-transcribe",
    ttsModelName: "gpt-4o-mini-tts"
)

// Process audio input
let audioInput = AudioInput(buffer: audioData)
let result = try await voiceAgent.process(audioInput: audioInput)

// Stream the results
for try await event in result.stream() {
    switch event {
    case .audio(let data):
        // Play the audio data
        playAudio(data)
    case .lifecycle(let event):
        print("Lifecycle event: \(event.rawValue)")
    case .error(let error):
        print("Error: \(error)")
    }
}
```

### Using Guardrails

```swift
// Create a length guardrail
let lengthGuardrail = InputLengthGuardrail(maxLength: 500)

// Create an agent with the guardrail
let agent = Agent<Void>(
    name: "AssistantWithGuardrails",
    instructions: "You are a helpful assistant.",
    guardrails: [lengthGuardrail]
)
```

### Multi-Agent Handoffs

```swift
// Create specialized agents
let weatherAgent = Agent<Void>(name: "WeatherAgent", instructions: "...")
    .addTool(weatherTool)

let travelAgent = Agent<Void>(name: "TravelAgent", instructions: "...")
    .addTool(travelTool)

// Create main agent with handoff to weather agent
let mainAgent = Agent<Void>(
    name: "MainAgent",
    instructions: "You are a helpful assistant.",
    handoffs: [
        Handoff.withKeywords(
            agent: weatherAgent,
            keywords: ["weather", "temperature", "forecast"]
        ),
        Handoff.withKeywords(
            agent: travelAgent,
            keywords: ["travel", "flight", "hotel", "booking"]
        )
    ]
)
```

### Streaming Responses

```swift
// Run with streaming
let result = try await AgentRunner.runStreamed(
    agent: agent,
    input: "Tell me about the weather in London",
    context: ()
) { content in
    // Process each chunk as it arrives
    print(content, terminator: "")
}
```

## Running the Examples

The project includes example applications that demonstrate how to use the SDK:

### Simple Text-based Agent

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Run the example
swift run SimpleApp
```

### Voice Agent Example

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Run the voice agent example
swift run VoiceAgentExample
```

## Documentation

Documentation is currently in development. For now, please refer to the source code and examples for usage guidance.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
