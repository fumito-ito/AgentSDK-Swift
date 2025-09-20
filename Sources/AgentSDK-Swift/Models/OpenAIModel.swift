import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import OpenAPIRuntime

/// Implementation of ModelInterface for OpenAI models
public final class OpenAIModel: ModelInterface {
    /// The API key for OpenAI
    private let apiKey: String

    /// The API base URL
    private let apiBaseURL: URL

    /// The URL session used for network requests
    private let urlSession: URLSession

    /// Creates a new OpenAI model
    /// - Parameters:
    ///   - apiKey: The API key for OpenAI
    ///   - apiBaseURL: The API base URL (defaults to OpenAI's API)
    ///   - urlSession: Optional custom URL session
    public init(
        apiKey: String,
        apiBaseURL: URL = URL(string: "https://api.openai.com/v1")!,
        urlSession: URLSession? = nil
    ) {
        self.apiKey = apiKey
        self.apiBaseURL = apiBaseURL
        self.urlSession = urlSession ?? URLSession.shared
    }

    /// Gets a response from the model
    /// - Parameters:
    ///   - messages: The messages to send to the model
    ///   - settings: The settings to use for the model call
    /// - Returns: The model response
    public func getResponse(messages: [Message], settings: ModelSettings) async throws -> ModelResponse {
        let requestBody = try createRequestBody(messages: messages, settings: settings)

        let endpoint = "\(apiBaseURL)/chat/completions"

        // Create request
        var request = createURLRequest(url: endpoint)

        // Add request body
        let bodyData = try JSONEncoder().encode(requestBody)
        request.httpBody = bodyData

        // Send request
        let (data, response) = try await urlSession.data(for: request, delegate: nil)

        // Check response status
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            let errorString = String(data: data, encoding: .utf8) ?? "Unknown error"
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw OpenAIModelError.requestFailed(statusCode: statusCode, message: errorString)
        }

        // Parse response
        let openAIResponse = try JSONDecoder().decode(ChatCompletionResponse.self, from: data)

        return try convertResponse(openAIResponse)
    }

    /// Gets a streamed response from the model
    /// - Parameters:
    ///   - messages: The messages to send to the model
    ///   - settings: The settings to use for the model call
    ///   - callback: The callback to call for each streamed chunk
    public func getStreamedResponse(
        messages: [Message],
        settings: ModelSettings,
        callback: @escaping (ModelStreamEvent) async -> Void
    ) async throws -> ModelResponse {
        let streamSettings = settings

        // Create request body with stream enabled
        var requestBody = try createRequestBody(messages: messages, settings: streamSettings)
        requestBody.stream = true

        let endpoint = "\(apiBaseURL)/chat/completions"

        // Create request
        var request = createURLRequest(url: endpoint)

        // Add request body
        let bodyData = try JSONEncoder().encode(requestBody)
        request.httpBody = bodyData

        // Create URLSession task
        let (data, response) = try await urlSession.data(for: request)

        // Check response status
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            let errorString = String(data: data, encoding: .utf8) ?? "Unknown error"
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw OpenAIModelError.requestFailed(statusCode: statusCode, message: errorString)
        }

        // Process streamed response
        var contentBuffer = ""
        var toolCalls: [ModelResponse.ToolCall] = []

        // Convert data to string and process line by line
        if let responseStr = String(data: data, encoding: .utf8) {
            let lines = responseStr.split(separator: "\n")

            for line in lines {
                if line.hasPrefix("data: ") {
                    let dataContent = line.dropFirst(6)

                    // Check for the "[DONE]" message
                    if dataContent == "[DONE]" {
                        await callback(.end)
                        continue
                    }

                    // Parse the JSON chunk
                    do {
                        let chunkData = Data(dataContent.utf8)
                        let chunkResponse = try JSONDecoder().decode(ChatCompletionChunk.self, from: chunkData)

                        if let choice = chunkResponse.choices.first {
                            if let content = choice.delta.content, !content.isEmpty {
                                contentBuffer += content
                                await callback(.content(content))
                            }

                            if let toolCall = choice.delta.toolCalls?.first {
                                // Handle tool call delta
                                if let existingToolCall = toolCalls.first(where: { $0.id == toolCall.id }) {
                                    // Update existing tool call
                                    if let index = toolCalls.firstIndex(where: { $0.id == toolCall.id }) {
                                        var params = existingToolCall.parameters

                                        if let function = toolCall.function {
                                            if let name = function.name {
                                                toolCalls[index] = ModelResponse.ToolCall(
                                                    id: existingToolCall.id,
                                                    name: name,
                                                    parameters: params
                                                )
                                            }

                                            if let arguments = function.arguments {
                                                do {
                                                    if let jsonData = arguments.data(using: .utf8),
                                                       let jsonParams = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
                                                        // Merge with existing parameters
                                                        for (key, value) in jsonParams {
                                                            params[key] = value
                                                        }

                                                        toolCalls[index] = ModelResponse.ToolCall(
                                                            id: existingToolCall.id,
                                                            name: existingToolCall.name,
                                                            parameters: params
                                                        )
                                                    }
                                                } catch {
                                                    // Ignore parsing errors for partial JSON
                                                }
                                            }
                                        }
                                    }
                                } else if let id = toolCall.id, let function = toolCall.function, let name = function.name {
                                    // Create new tool call
                                    var params: [String: Any] = [:]

                                    if let arguments = function.arguments {
                                        do {
                                            if let jsonData = arguments.data(using: .utf8),
                                               let jsonParams = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
                                                params = jsonParams
                                            }
                                        } catch {
                                            // Ignore parsing errors for partial JSON
                                        }
                                    }

                                    let newToolCall = ModelResponse.ToolCall(id: id, name: name, parameters: params)
                                    toolCalls.append(newToolCall)
                                    await callback(.toolCall(newToolCall))
                                }
                            }
                        }
                    } catch {
                        // Ignore partial JSON errors
                    }
                }
            }
        }

        // Create final response
        return ModelResponse(
            content: contentBuffer,
            toolCalls: toolCalls
        )
    }

    /// Creates a URLRequest configured with the appropriate headers
    /// - Parameter url: The URL string for the request
    /// - Returns: A configured URLRequest
    private func createURLRequest(url: String) -> URLRequest {
        var request = URLRequest(url: URL(string: url)!)
        request.httpMethod = "POST"
        request.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 600 // 10 minute timeout
        return request
    }

    /// Creates a request body for the OpenAI API
    /// - Parameters:
    ///   - messages: The messages to send to the model
    ///   - settings: The settings to use for the model call
    /// - Returns: The request body
    private func createRequestBody(messages: [Message], settings: ModelSettings) throws -> ChatCompletionRequest {
        // Convert messages to OpenAI format
        let openAIMessages = messages.map { message -> ChatMessage in
            let role = message.role.rawValue

            switch message.content {
            case .text(let text):
                return ChatMessage(role: role, content: text)

            case .toolResults(let toolResult):
                return ChatMessage(
                    role: role,
                    toolCallId: toolResult.toolCallId,
                    content: toolResult.result
                )
            }
        }

        // Convert tools to OpenAI format
        let tools: [OpenAITool]? = nil // Implement tool conversion if needed

        // Create request body
        var request = ChatCompletionRequest(
            model: settings.modelName,
            messages: openAIMessages,
            tools: tools
        )

        // Add optional parameters from settings
        if let temperature = settings.temperature {
            request.temperature = temperature
        }

        if let topP = settings.topP {
            request.topP = topP
        }

        if let maxTokens = settings.maxTokens {
            request.maxTokens = maxTokens
        }

        if let responseFormat = settings.responseFormat {
            request.responseFormat = ["type": responseFormat.jsonValue]
        }

        if let seed = settings.seed {
            request.seed = seed
        }

        // Add any additional parameters
        for (_, _) in settings.additionalParameters {
            // This is a simplification - in a real implementation, we would need to properly handle adding these params
        }

        return request
    }

    /// Converts an OpenAI response to a ModelResponse
    /// - Parameter response: The OpenAI response
    /// - Returns: The converted ModelResponse
    private func convertResponse(_ response: ChatCompletionResponse) throws -> ModelResponse {
        guard let choice = response.choices.first else {
            throw OpenAIModelError.emptyResponse
        }

        // Get content
        let content = choice.message.content ?? ""

        // Get tool calls if any
        var toolCalls: [ModelResponse.ToolCall] = []

        if let openAIToolCalls = choice.message.toolCalls {
            for toolCall in openAIToolCalls {
                do {
                    let arguments = toolCall.function.arguments
                    let argsData = arguments.data(using: .utf8) ?? Data()
                    let params = try JSONSerialization.jsonObject(with: argsData) as? [String: Any] ?? [:]

                    toolCalls.append(ModelResponse.ToolCall(
                        id: toolCall.id,
                        name: toolCall.function.name,
                        parameters: params
                    ))
                } catch {
                    throw OpenAIModelError.invalidToolCallArguments(error)
                }
            }
        }

        // Get usage statistics
        var usage: ModelResponse.Usage?
        if let responseUsage = response.usage {
            usage = ModelResponse.Usage(
                promptTokens: responseUsage.promptTokens,
                completionTokens: responseUsage.completionTokens,
                totalTokens: responseUsage.totalTokens
            )
        }

        return ModelResponse(
            content: content,
            toolCalls: toolCalls,
            usage: usage
        )
    }

    /// Errors that can occur when using the OpenAI model
    public enum OpenAIModelError: Error {
        case requestFailed(statusCode: Int, message: String)
        case emptyResponse
        case invalidToolCallArguments(Error)
    }

    // OpenAI API Types

    /// Request for the OpenAI chat completions API
    private struct ChatCompletionRequest: Encodable {
        let model: String
        let messages: [ChatMessage]
        let tools: [OpenAITool]?
        var temperature: Double?
        var topP: Double?
        var maxTokens: Int?
        var responseFormat: [String: String]?
        var seed: Int?
        var stream: Bool = false
    }

    /// Tool for the OpenAI chat completions API
    private struct OpenAITool: Encodable {
        let type: String
        let function: FunctionDefinition
    }

    /// Message for the OpenAI chat completions API
    private struct ChatMessage: Encodable {
        let role: String
        var content: String?
        var toolCallId: String?

        init(role: String, content: String) {
            self.role = role
            self.content = content
        }

        init(role: String, toolCallId: String, content: String) {
            self.role = role
            self.toolCallId = toolCallId
            self.content = content
        }
    }

    /// Function definition for the OpenAI chat completions API
    private struct FunctionDefinition: Encodable {
        let name: String
        let description: String
        let parameters: [String: Any]

        enum CodingKeys: String, CodingKey {
            case name, description, parameters
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(name, forKey: .name)
            try container.encode(description, forKey: .description)

            // Encode parameters dictionary as a raw JSON string
            let parametersData = try JSONSerialization.data(withJSONObject: parameters)
            let parametersString = String(data: parametersData, encoding: .utf8) ?? "{}"
            try container.encode(parametersString, forKey: .parameters)
        }
    }

    /// Response from the OpenAI chat completions API
    private struct ChatCompletionResponse: Decodable {
        let id: String
        let object: String
        let created: Int
        let model: String
        let choices: [Choice]
        let usage: Usage?

        struct Choice: Decodable {
            let index: Int
            let message: Message
            let finishReason: String
        }

        struct Message: Decodable {
            let role: String
            let content: String?
            let toolCalls: [ToolCall]?
        }

        struct ToolCall: Decodable {
            let id: String
            let type: String
            let function: Function
        }

        struct Function: Decodable {
            let name: String
            let arguments: String
        }

        struct Usage: Decodable {
            let promptTokens: Int
            let completionTokens: Int
            let totalTokens: Int
        }
    }

    /// Chunk response from the OpenAI chat completions API when streaming
    private struct ChatCompletionChunk: Decodable {
        let id: String
        let object: String
        let created: Int
        let model: String
        let choices: [Choice]

        struct Choice: Decodable {
            let index: Int
            let delta: Delta
            let finishReason: String?
        }

        struct Delta: Decodable {
            let role: String?
            let content: String?
            let toolCalls: [ToolCall]?

            enum CodingKeys: String, CodingKey {
                case role, content
                case toolCalls = "tool_calls"
            }
        }

        struct ToolCall: Decodable {
            let id: String?
            let type: String?
            let function: Function?

            enum CodingKeys: String, CodingKey {
                case id, type, function
            }
        }

        struct Function: Decodable {
            let name: String?
            let arguments: String?

            enum CodingKeys: String, CodingKey {
                case name, arguments
            }
        }
    }
}

/// Extension to register OpenAI models with the model provider
public extension ModelProvider {
    /// Registers OpenAI models with the model provider
    /// - Parameter apiKey: The API key for OpenAI
    func registerOpenAIModels(apiKey: String) {
        // Register default OpenAI models
        register(modelName: "gpt-4-turbo") {
            OpenAIModel(apiKey: apiKey)
        }

        register(modelName: "gpt-4") {
            OpenAIModel(apiKey: apiKey)
        }

        register(modelName: "gpt-3.5-turbo") {
            OpenAIModel(apiKey: apiKey)
        }
    }
}
