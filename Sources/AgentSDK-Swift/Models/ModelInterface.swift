import Foundation

/// Protocol defining the interface for language models
public protocol ModelInterface: Sendable {
    /// Gets a response from the model
    /// - Parameters:
    ///   - messages: The messages to send to the model
    ///   - settings: The settings to use for the model call
    /// - Returns: The model response
    func getResponse(messages: [Message], settings: ModelSettings) async throws -> ModelResponse

    /// Gets a streamed response from the model
    /// - Parameters:
    ///   - messages: The messages to send to the model
    ///   - settings: The settings to use for the model call
    ///   - callback: The callback to call for each streamed chunk
    func getStreamedResponse(
        messages: [Message],
        settings: ModelSettings,
        callback: @escaping (ModelStreamEvent) async -> Void
    ) async throws -> ModelResponse
}

/// Represents a message for a model
public struct Message {
    /// The role of the message sender
    public let role: Role

    /// The content of the message
    public let content: MessageContent

    /// Creates a new message
    /// - Parameters:
    ///   - role: The role of the message sender
    ///   - content: The content of the message
    public init(role: Role, content: MessageContent) {
        self.role = role
        self.content = content
    }

    /// Creates a new user message with text content
    /// - Parameter text: The text content
    /// - Returns: A new user message
    public static func user(_ text: String) -> Message {
        Message(role: .user, content: .text(text))
    }

    /// Creates a new assistant message with text content
    /// - Parameter text: The text content
    /// - Returns: A new assistant message
    public static func assistant(_ text: String) -> Message {
        Message(role: .assistant, content: .text(text))
    }

    /// Creates a new system message with text content
    /// - Parameter text: The text content
    /// - Returns: A new system message
    public static func system(_ text: String) -> Message {
        Message(role: .system, content: .text(text))
    }

    /// Represents the role of a message sender
    public enum Role: String {
        case system
        case user
        case assistant
        case tool
    }
}

/// Represents the content of a message
public enum MessageContent {
    case text(String)
    case toolResults(ToolResult)

    /// Represents the result of a tool call
    public struct ToolResult {
        /// The ID of the tool call
        public let toolCallId: String

        /// The result of the tool call
        public let result: String

        /// Creates a new tool result
        /// - Parameters:
        ///   - toolCallId: The ID of the tool call
        ///   - result: The result of the tool call
        public init(toolCallId: String, result: String) {
            self.toolCallId = toolCallId
            self.result = result
        }
    }
}

/// Represents a response from a model
public struct ModelResponse {
    /// The generated text content
    public let content: String

    /// The tool calls made by the model
    public let toolCalls: [ToolCall]

    /// Whether the response was flagged for moderation
    public let flagged: Bool

    /// The reason the response was flagged, if applicable
    public let flaggedReason: String?

    /// Usage statistics for the model call
    public let usage: Usage?

    /// Creates a new model response
    /// - Parameters:
    ///   - content: The generated text content
    ///   - toolCalls: The tool calls made by the model
    ///   - flagged: Whether the response was flagged for moderation
    ///   - flaggedReason: The reason the response was flagged, if applicable
    ///   - usage: Usage statistics for the model call
    public init(
        content: String,
        toolCalls: [ToolCall] = [],
        flagged: Bool = false,
        flaggedReason: String? = nil,
        usage: Usage? = nil
    ) {
        self.content = content
        self.toolCalls = toolCalls
        self.flagged = flagged
        self.flaggedReason = flaggedReason
        self.usage = usage
    }

    /// Represents a tool call made by the model
    public struct ToolCall {
        /// The ID of the tool call
        public let id: String

        /// The name of the tool being called
        public let name: String

        /// The parameters for the tool call
        public let parameters: [String: Any]

        /// Creates a new tool call
        /// - Parameters:
        ///   - id: The ID of the tool call
        ///   - name: The name of the tool being called
        ///   - parameters: The parameters for the tool call
        public init(id: String, name: String, parameters: [String: Any]) {
            self.id = id
            self.name = name
            self.parameters = parameters
        }
    }

    /// Represents usage statistics for a model call
    public struct Usage {
        /// The number of prompt tokens used
        public let promptTokens: Int

        /// The number of completion tokens used
        public let completionTokens: Int

        /// The total number of tokens used
        public let totalTokens: Int

        /// Creates a new usage statistics object
        /// - Parameters:
        ///   - promptTokens: The number of prompt tokens used
        ///   - completionTokens: The number of completion tokens used
        ///   - totalTokens: The total number of tokens used
        public init(promptTokens: Int, completionTokens: Int, totalTokens: Int) {
            self.promptTokens = promptTokens
            self.completionTokens = completionTokens
            self.totalTokens = totalTokens
        }
    }
}

/// Represents an event from a streamed model response
public enum ModelStreamEvent {
    /// A content chunk was received
    case content(String)

    /// A tool call was received
    case toolCall(ModelResponse.ToolCall)

    /// The stream has ended
    case end
}

/// Factory for creating model instances by name
public actor ModelProvider {
    /// The shared instance of the model provider
    public static let shared = ModelProvider()

    /// Dictionary mapping model names to factory functions
    private var modelFactories: [String: () -> ModelInterface] = [:]

    private init() {}

    /// Registers a model factory with the provider
    /// - Parameters:
    ///   - modelName: The name of the model
    ///   - factory: The factory function for creating the model
    public func register(modelName: String, factory: @escaping () -> ModelInterface) {
        modelFactories[modelName] = factory
    }

    /// Gets a model by name
    /// - Parameter modelName: The name of the model
    /// - Returns: The model instance
    /// - Throws: An error if the model is not registered
    public func getModel(modelName: String) throws -> ModelInterface {
        guard let factory = modelFactories[modelName] else {
            throw ModelProviderError.modelNotFound(modelName: modelName)
        }

        return factory()
    }

    /// Errors that can occur when using the model provider
    public enum ModelProviderError: Error {
        /// The requested model was not found
        case modelNotFound(modelName: String)
    }
}
