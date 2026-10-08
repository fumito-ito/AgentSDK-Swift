import Foundation

/// Prompt configuration to use for interacting with an OpenAI model.
public struct Prompt: Sendable, Equatable {
    /// The unique ID of the prompt.
    public var id: String
    /// Optional version of the prompt.
    public var version: String?
    /// Optional variables to substitute into the prompt.
    public var variables: JSONObject?

    public init(id: String, version: String? = nil, variables: JSONObject? = nil) {
        self.id = id
        self.version = version
        self.variables = variables
    }
}

/// Inputs to a function that dynamically generates a prompt.
public struct GenerateDynamicPromptData: Sendable {
    public var context: RunContextWrapper
    public var agent: Agent
}

/// A function that dynamically generates a prompt.
public typealias DynamicPromptFunction = @Sendable (GenerateDynamicPromptData) async throws -> Prompt

/// A static prompt or a function that produces one.
public enum AgentPrompt: Sendable {
    case prompt(Prompt)
    case dynamic(DynamicPromptFunction)
}

enum PromptUtil {
    static func toModelInput(_ prompt: AgentPrompt?, context: RunContextWrapper, agent: Agent) async throws -> JSONObject? {
        guard let prompt else { return nil }
        let resolved: Prompt
        switch prompt {
        case .prompt(let value): resolved = value
        case .dynamic(let function): resolved = try await function(GenerateDynamicPromptData(context: context, agent: agent))
        }
        return [
            "id": .string(resolved.id),
            "version": resolved.version.map(JSONValue.string) ?? .null,
            "variables": resolved.variables.map(JSONValue.object) ?? .null,
        ]
    }
}
