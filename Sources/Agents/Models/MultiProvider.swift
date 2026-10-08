import Foundation

// Port of src/agents/models/multi_provider.py.
//
// Swift: the `litellm/` and `any-llm/` fallback providers are third-party integrations and are
// out of scope; those prefixes throw `UserError` unless mapped explicitly.

/// A map of model name prefixes to model providers.
public final class MultiProviderMap: @unchecked Sendable {
    private let lock = NSLock()
    private var mapping: [String: any ModelProvider] = [:]

    public init() {}

    /// Returns whether the given prefix is in the mapping.
    public func hasPrefix(_ prefix: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return mapping[prefix] != nil
    }

    /// Returns a copy of the current prefix → provider mapping.
    public func getMapping() -> [String: any ModelProvider] {
        lock.lock()
        defer { lock.unlock() }
        return mapping
    }

    /// Overwrites the current mapping.
    public func setMapping(_ newMapping: [String: any ModelProvider]) {
        lock.lock()
        mapping = newMapping
        lock.unlock()
    }

    /// Returns the provider for `prefix`, e.g. `"openai"` or `"my_prefix"`.
    public func getProvider(_ prefix: String) -> (any ModelProvider)? {
        lock.lock()
        defer { lock.unlock() }
        return mapping[prefix]
    }

    /// Adds a prefix → provider mapping.
    public func addProvider(_ prefix: String, _ provider: any ModelProvider) {
        lock.lock()
        mapping[prefix] = provider
        lock.unlock()
    }

    /// Removes the mapping for `prefix`.
    public func removeProvider(_ prefix: String) {
        lock.lock()
        mapping.removeValue(forKey: prefix)
        lock.unlock()
    }
}

/// How `openai/...` model names are interpreted.
public enum MultiProviderOpenAIPrefixMode: String, Sendable {
    /// Strip the `openai/` prefix before calling the OpenAI provider.
    case alias
    /// Keep the full string, for OpenAI-compatible endpoints with namespaced model IDs.
    case modelId = "model_id"
}

/// How prefixes outside the provider map are handled.
public enum MultiProviderUnknownPrefixMode: String, Sendable {
    /// Throw `UserError`.
    case error
    /// Pass the full string through to the OpenAI provider.
    case modelId = "model_id"
}

/// Maps model names to providers by prefix: `openai/` or no prefix → `OpenAIProvider`.
public final class MultiProvider: ModelProvider, @unchecked Sendable {
    public let providerMap: MultiProviderMap?
    public let openAIProvider: OpenAIProvider
    private let openAIPrefixMode: MultiProviderOpenAIPrefixMode
    private let unknownPrefixMode: MultiProviderUnknownPrefixMode

    public init(
        providerMap: MultiProviderMap? = nil,
        openAIAPIKey: String? = nil,
        openAIBaseURL: URL? = nil,
        openAIClient: OpenAIClient? = nil,
        openAIOrganization: String? = nil,
        openAIProject: String? = nil,
        openAIUseResponses: Bool? = nil,
        openAIStrictFeatureValidation: Bool = false,
        openAIPrefixMode: MultiProviderOpenAIPrefixMode = .alias,
        unknownPrefixMode: MultiProviderUnknownPrefixMode = .error,
        openAIBufferStreamedToolCalls: Bool = false
    ) throws {
        self.providerMap = providerMap
        self.openAIProvider = try OpenAIProvider(
            apiKey: openAIAPIKey,
            baseURL: openAIBaseURL,
            openAIClient: openAIClient,
            organization: openAIOrganization,
            project: openAIProject,
            useResponses: openAIUseResponses,
            strictFeatureValidation: openAIStrictFeatureValidation,
            bufferStreamedToolCalls: openAIBufferStreamedToolCalls
        )
        self.openAIPrefixMode = openAIPrefixMode
        self.unknownPrefixMode = unknownPrefixMode
    }

    /// Creates a provider with the default mapping; this initializer cannot fail.
    public convenience init() {
        // swiftlint:disable:next force_try
        try! self.init(providerMap: nil)
    }

    private func prefixAndModelName(_ modelName: String) -> (String?, String) {
        guard let slash = modelName.firstIndex(of: "/") else { return (nil, modelName) }
        return (String(modelName[..<slash]), String(modelName[modelName.index(after: slash)...]))
    }

    private func resolvePrefixedModel(original: String, prefix: String, stripped: String) throws -> (any ModelProvider, String) {
        // Explicit provider_map entries always win over the built-in routing.
        if let provider = providerMap?.getProvider(prefix) {
            return (provider, stripped)
        }
        if prefix == "litellm" || prefix == "any-llm" {
            throw UserError("Unknown prefix: \(prefix)")
        }
        if prefix == "openai" {
            return (openAIProvider, openAIPrefixMode == .alias ? stripped : original)
        }
        if unknownPrefixMode == .modelId {
            return (openAIProvider, original)
        }
        throw UserError("Unknown prefix: \(prefix)")
    }

    /// Returns a model for `modelName`. A prefix ending in `/` selects the provider; names
    /// without a prefix use the OpenAI provider.
    public func getModel(_ modelName: String?) throws -> any Model {
        guard let modelName else { return try openAIProvider.getModel(nil) }
        let (prefix, stripped) = prefixAndModelName(modelName)
        guard let prefix else { return try openAIProvider.getModel(stripped) }
        let (provider, resolved) = try resolvePrefixedModel(original: modelName, prefix: prefix, stripped: stripped)
        return try provider.getModel(resolved)
    }

    /// Closes cached resources held by child providers.
    public func aclose() async {
        var providers: [any ModelProvider] = [openAIProvider]
        providers.append(contentsOf: providerMap?.getMapping().values.map { $0 } ?? [])
        var seen: Set<ObjectIdentifier> = []
        for provider in providers where provider !== self {
            guard seen.insert(ObjectIdentifier(provider)).inserted else { continue }
            await provider.aclose()
        }
    }
}
