import Testing
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import AgentSDK_Swift

// MARK: - OpenAI Model Tests

@Test func testOpenAIModelCreation() async throws {
    // Create an OpenAI model
    let apiKey = "test-api-key"
    _ = OpenAIModel(apiKey: apiKey)

    // Just check instantiation works - no assertions needed
    #expect(Bool(true))
}

@Test func testCustomBaseURL() async throws {
    // Create an OpenAI model with custom base URL
    let customBaseURL = URL(string: "https://custom-openai-api.example.com/v1")!
    _ = OpenAIModel(apiKey: "test-key", apiBaseURL: customBaseURL)

    // Just check instantiation works
    #expect(Bool(true))
}

@Test func testModelSettings() async throws {
    // This is mostly a compilation validation test
    _ = [
        Message(role: .user, content: .text("Hello, how are you?"))
    ]

    let settings = ModelSettings(
        modelName: "test-model",
        temperature: 0.7,
        topP: 0.9,
        maxTokens: 1000
    )

    // Check settings values
    #expect(settings.modelName == "test-model")
    #expect(settings.temperature == 0.7)
    #expect(settings.topP == 0.9)
    #expect(settings.maxTokens == 1000)
}

@Test func testMessageContent() async throws {
    // Create text message
    let textMessage = Message(
        role: .user,
        content: .text("Hello")
    )

    // Check message content type
    if case .text(let content) = textMessage.content {
        #expect(content == "Hello")
    } else {
        #expect(Bool(false), "Should be text content")
    }

    // Check role
    #expect(textMessage.role == .user)
}
