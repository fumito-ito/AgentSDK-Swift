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

## Requirements

- Swift 6.0+
- macOS 13.0+ / iOS 16.0+
- OpenAI API Key

## Installation

### Swift Package Manager

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/AgentSDK-Swift.git", from: "0.1.0")
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

## Running the Example

The project includes a simple example application that demonstrates how to use the SDK:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Run the example
swift run SimpleApp
```

## Documentation

Documentation is currently in development. For now, please refer to the source code and examples for usage guidance.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Voice Agent Functionality

AgentSDK-Swift now includes voice agent capabilities, allowing you to create agents that can process spoken input and generate spoken responses. This is similar to the OpenAI Agents Python SDK voice capabilities.

### Voice Pipeline

The voice pipeline processes audio through three main steps:

1. **Speech-to-Text**: Converts user's speech to text using services like OpenAI Whisper
2. **Agent Processing**: Processes the text with an agent to generate a response
3. **Text-to-Speech**: Converts the agent's response back to speech

### Available Components

- **VoicePipeline**: Orchestrates the entire voice processing workflow
- **SpeechToTextService**: Interface for speech recognition with OpenAI Whisper implementation
- **TextToSpeechService**: Interface for speech synthesis with OpenAI TTS implementation
- **VoiceAgent**: Combines the agent with voice processing capabilities

### Example Usage

```swift
// Create a regular agent with instructions
let agent = Agent<Void>(
    name: "Voice Assistant",
    instructions: "You are a helpful voice assistant. Keep responses concise and conversational."
)

// Add tools to the agent if needed
agent.addTool(/* your tool definition */)

// Create a voice pipeline using OpenAI services
let pipeline = VoicePipeline(
    openAIApiKey: "your-api-key",
    agent: agent
)

// Process audio input
let audioData: Data = /* your audio data */
let result = try await pipeline.process(
    audioData: audioData,
    context: yourContext
)

// Access the results
print("Transcription: \(result.inputTranscription)")
print("Agent response: \(result.agentResponse)")

// Play the audio response
playAudio(result.audioResponse)

// Or use streaming mode for more responsive interactions
try await pipeline.processStreamed(
    audioData: audioData,
    context: yourContext
) { audioChunk in
    // Play each chunk of audio as it's generated
    playAudio(audioChunk)
}
```

### Requirements

- iOS 16+ or macOS 13+
- OpenAI API key with access to Whisper and TTS models
- AVFoundation for audio recording and playback

See the `Examples/VoiceAgentExample.swift` file for a complete example of how to use the voice agent functionality.