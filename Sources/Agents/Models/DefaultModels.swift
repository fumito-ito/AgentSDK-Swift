import Foundation

public let openAIDefaultModelEnvVariableName = "OPENAI_DEFAULT_MODEL"

private let gpt5ChatModelPatterns = [
    "^gpt-5-chat-latest$", "^gpt-5\\.1-chat-latest$", "^gpt-5\\.2-chat-latest$", "^gpt-5\\.3-chat-latest$",
]

private let gpt5DefaultReasoningEffortPatterns: [(String, String)] = [
    ("^gpt-5(?:-\\d{4}-\\d{2}-\\d{2})?$", "low"),
    ("^gpt-5\\.1(?:-\\d{4}-\\d{2}-\\d{2})?$", "none"),
    ("^gpt-5\\.2(?:-\\d{4}-\\d{2}-\\d{2})?$", "none"),
    ("^gpt-5\\.2-pro(?:-\\d{4}-\\d{2}-\\d{2})?$", "medium"),
    ("^gpt-5\\.2-codex$", "low"),
    ("^gpt-5\\.3-codex$", "none"),
    ("^gpt-5\\.4(?:-\\d{4}-\\d{2}-\\d{2})?$", "none"),
    ("^gpt-5\\.4-pro(?:-\\d{4}-\\d{2}-\\d{2})?$", "medium"),
    ("^gpt-5\\.4-mini(?:-\\d{4}-\\d{2}-\\d{2})?$", "none"),
    ("^gpt-5\\.4-nano(?:-\\d{4}-\\d{2}-\\d{2})?$", "none"),
    ("^gpt-5\\.5(?:-\\d{4}-\\d{2}-\\d{2})?$", "none"),
    ("^gpt-5\\.6(?:-\\d{4}-\\d{2}-\\d{2})?$", "none"),
    ("^gpt-5\\.6-sol(?:-\\d{4}-\\d{2}-\\d{2})?$", "none"),
    ("^gpt-5\\.6-terra(?:-\\d{4}-\\d{2}-\\d{2})?$", "none"),
    ("^gpt-5\\.6-luna(?:-\\d{4}-\\d{2}-\\d{2})?$", "none"),
]

private func fullMatch(_ pattern: String, _ value: String) -> Bool {
    value.range(of: pattern, options: .regularExpression) != nil
}

/// Returns `true` if the model name is a GPT-5 model and reasoning settings are required.
public func gpt5ReasoningSettingsRequired(_ modelName: String) -> Bool {
    if gpt5ChatModelPatterns.contains(where: { fullMatch($0, modelName) }) {
        // Chat-latest aliases do not accept reasoning.effort.
        return false
    }
    return modelName.hasPrefix("gpt-5")
}

/// Returns `true` if the default model is a GPT-5 model.
public func isGPT5Default() -> Bool {
    gpt5ReasoningSettingsRequired(getDefaultModel())
}

/// Returns the default model name (`OPENAI_DEFAULT_MODEL`, or `gpt-5.6-luna`).
public func getDefaultModel() -> String {
    (ProcessInfo.processInfo.environment[openAIDefaultModelEnvVariableName] ?? "gpt-5.6-luna").lowercased()
}

/// Returns the default model settings for `model` (or the default model).
public func getDefaultModelSettings(_ model: String? = nil) -> ModelSettings {
    let resolved = model ?? getDefaultModel()
    guard gpt5ReasoningSettingsRequired(resolved) else { return ModelSettings() }
    if let effort = gpt5DefaultReasoningEffortPatterns.first(where: { fullMatch($0.0, resolved) })?.1 {
        return ModelSettings(reasoning: Reasoning(effort: effort), verbosity: "low")
    }
    // Keep the GPT-5 verbosity default, but omit reasoning.effort for variants whose supported
    // values are not confirmed yet.
    return ModelSettings(verbosity: "low")
}

/// A placeholder ID used to fill in the `id` field in Responses API related objects created from
/// non-Responses APIs.
public let fakeResponsesId = "__fake_id__"
