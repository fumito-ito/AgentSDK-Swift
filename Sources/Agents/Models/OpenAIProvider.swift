import Foundation

// Port of src/agents/models/openai_provider.py.
//
// Swift: the websocket Responses transport and agent registration are not ported; see
// porting/manifest.json.

/// The default model name kept for backward compatibility; prefer `getDefaultModel()`.
public let defaultModel = "gpt-4o"

/// Looks up OpenAI models (Responses or Chat Completions) by name.
public final class OpenAIProvider: ModelProvider, @unchecked Sendable {
    private let lock = NSLock()
    private var client: OpenAIClient?
    private let storedAPIKey: String?
    private let storedBaseURL: URL?
    private let storedOrganization: String?
    private let storedProject: String?
    private let useResponses: Bool
    private let strictFeatureValidation: Bool
    private let bufferStreamedToolCalls: Bool

    /// - Parameters:
    ///   - apiKey: The API key for the OpenAI client. Defaults to the configured default key or
    ///     `OPENAI_API_KEY`.
    ///   - baseURL: The base URL for the OpenAI client. Defaults to `OPENAI_BASE_URL` or the
    ///     official endpoint.
    ///   - openAIClient: An existing client. Don't combine with the other client options.
    ///   - organization: The organization for the OpenAI client.
    ///   - project: The project for the OpenAI client.
    ///   - useResponses: Whether to use the Responses API (defaults to `setDefaultOpenAIAPI`).
    ///   - strictFeatureValidation: Whether Chat Completions models throw `UserError` for
    ///     Responses-only features such as `previousResponseId` or `prompt`.
    ///   - bufferStreamedToolCalls: Whether Chat Completions models buffer streamed tool-call
    ///     deltas until the provider stream finishes.
    public init(
        apiKey: String? = nil,
        baseURL: URL? = nil,
        openAIClient: OpenAIClient? = nil,
        organization: String? = nil,
        project: String? = nil,
        useResponses: Bool? = nil,
        strictFeatureValidation: Bool = false,
        bufferStreamedToolCalls: Bool = false
    ) throws {
        if openAIClient != nil, apiKey != nil || baseURL != nil || organization != nil || project != nil {
            throw UserError("Don't provide api_key, base_url, websocket_base_url, organization, or project if you provide openai_client")
        }
        self.client = openAIClient
        self.storedAPIKey = apiKey
        self.storedBaseURL = baseURL
        self.storedOrganization = organization
        self.storedProject = project
        self.useResponses = useResponses ?? getUseResponsesByDefault()
        self.strictFeatureValidation = strictFeatureValidation
        self.bufferStreamedToolCalls = bufferStreamedToolCalls
    }

    /// Creates a provider without client options; this initializer cannot fail.
    public convenience init() {
        // swiftlint:disable:next force_try
        try! self.init(apiKey: nil)
    }

    // The client is created lazily so constructing a provider never requires an API key.
    func getClient() -> OpenAIClient {
        lock.lock()
        defer { lock.unlock() }
        if let client { return client }
        let hasExplicitOptions = storedAPIKey != nil || storedBaseURL != nil || storedOrganization != nil || storedProject != nil
        let created = (hasExplicitOptions ? nil : getDefaultOpenAIClient())
            ?? OpenAIClient(
                apiKey: storedAPIKey ?? getDefaultOpenAIKey(),
                baseURL: storedBaseURL,
                organization: storedOrganization,
                project: storedProject
            )
        client = created
        return created
    }

    public func getModel(_ modelName: String?) throws -> any Model {
        let isExplicit = modelName != nil
        let resolved = modelName ?? getDefaultModel()
        let client = getClient()
        if useResponses {
            return OpenAIResponsesModel(model: resolved, openAIClient: client, modelIsExplicit: isExplicit)
        }
        return OpenAIChatCompletionsModel(
            model: resolved,
            openAIClient: client,
            strictFeatureValidation: strictFeatureValidation,
            bufferStreamedToolCalls: bufferStreamedToolCalls
        )
    }
}
