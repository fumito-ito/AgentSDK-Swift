import Foundation
import Testing
@testable import Agents

// Ported from tests/models/test_default_models.py.
//
// Swift: environment-dependent tests set `OPENAI_DEFAULT_MODEL` with `withEnvironment` in this
// serialized suite.
@Suite(.serialized) struct DefaultModelsTests {
    private func gpt5DefaultSettings(_ reasoningEffort: String?) -> ModelSettings {
        guard let reasoningEffort else { return ModelSettings(verbosity: "low") }
        return ModelSettings(reasoning: Reasoning(effort: reasoningEffort), verbosity: "low")
    }

    @Test func testDefaultModelIsGpt56Luna() async {
        await withEnvironment([openAIDefaultModelEnvVariableName: nil]) {
            #expect(getDefaultModel() == "gpt-5.6-luna")
            #expect(isGPT5Default())
            #expect(gpt5ReasoningSettingsRequired(getDefaultModel()))
            #expect(getDefaultModelSettings() == gpt5DefaultSettings("none"))
        }
    }

    @Test func testIsGpt5DefaultWithRealModelName() async {
        await withEnvironment([openAIDefaultModelEnvVariableName: "gpt-5.4"]) {
            #expect(getDefaultModel() == "gpt-5.4")
            #expect(isGPT5Default())
        }
    }

    @Test func testIsGpt5DefaultReturnsFalseForNonGpt5DefaultModel() async {
        await withEnvironment([openAIDefaultModelEnvVariableName: "gpt-4.1"]) {
            #expect(getDefaultModel() == "gpt-4.1")
            #expect(!isGPT5Default())
        }
    }

    @Test func testGpt5ReasoningSettingsRequiredDetectsGpt5ModelsWhileIgnoringChatLatest() {
        for model in ["gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.2-codex", "gpt-5.2-pro", "gpt-5.4-pro", "gpt-5.5", "gpt-5-mini", "gpt-5-nano"] {
            #expect(gpt5ReasoningSettingsRequired(model), "\(model)")
        }
        for model in ["gpt-5-chat-latest", "gpt-5.1-chat-latest", "gpt-5.2-chat-latest", "gpt-5.3-chat-latest"] {
            #expect(!gpt5ReasoningSettingsRequired(model), "\(model)")
        }
    }

    @Test func testGpt5ReasoningSettingsRequiredReturnsFalseForNonGpt5Models() {
        #expect(!gpt5ReasoningSettingsRequired("gpt-4.1"))
    }

    @Test(arguments: [
        "gpt-5.1", "gpt-5.1-2025-11-13", "gpt-5.2", "gpt-5.2-2025-12-11", "gpt-5.3-codex", "gpt-5.4",
        "gpt-5.4-2026-03-05", "gpt-5.4-mini-2026-03-17", "gpt-5.4-nano-2026-03-17", "gpt-5.4-mini", "gpt-5.4-nano",
        "gpt-5.5", "gpt-5.5-2026-04-23", "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    ])
    func testGetDefaultModelSettingsReturnsNoneReasoningDefaults(model: String) {
        // Ports the gpt-5.1 through gpt-5.6 "none reasoning" default tests.
        #expect(getDefaultModelSettings(model) == gpt5DefaultSettings("none"))
    }

    @Test(arguments: ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
    func testAgentUsesGpt56ModelSettingsFromDefaultModelEnv(model: String) async {
        await withEnvironment([openAIDefaultModelEnvVariableName: model.uppercased()]) {
            let agent = Agent(name: "test")

            #expect(getDefaultModel() == model)
            #expect(agent.model == nil)
            #expect(agent.modelSettings == gpt5DefaultSettings("none"))
        }
    }

    @Test func testGetDefaultModelSettingsReturnsLowReasoningDefaults() {
        // Ports the base gpt-5 and gpt-5.2-codex tests.
        #expect(getDefaultModelSettings("gpt-5") == gpt5DefaultSettings("low"))
        #expect(getDefaultModelSettings("gpt-5-2025-08-07") == gpt5DefaultSettings("low"))
        #expect(getDefaultModelSettings("gpt-5.2-codex") == gpt5DefaultSettings("low"))
    }

    @Test func testGetDefaultModelSettingsReturnsMediumReasoningDefaultsForGpt5ProModels() {
        for model in ["gpt-5.2-pro", "gpt-5.2-pro-2025-12-11", "gpt-5.4-pro", "gpt-5.4-pro-2026-03-05"] {
            #expect(getDefaultModelSettings(model) == gpt5DefaultSettings("medium"), "\(model)")
        }
    }

    @Test func testGetDefaultModelSettingsOmitsReasoningForUnconfirmedGpt5Variants() {
        for model in ["gpt-5-mini", "gpt-5-mini-2025-08-07", "gpt-5-nano", "gpt-5-nano-2025-08-07", "gpt-5.1-codex"] {
            #expect(getDefaultModelSettings(model) == gpt5DefaultSettings(nil), "\(model)")
        }
    }

    @Test func testGetDefaultModelSettingsReturnsEmptySettingsForGpt5ChatLatestAliases() {
        for model in ["gpt-5-chat-latest", "gpt-5.1-chat-latest", "gpt-5.2-chat-latest", "gpt-5.3-chat-latest"] {
            #expect(getDefaultModelSettings(model) == ModelSettings(), "\(model)")
        }
    }

    @Test func testGetDefaultModelSettingsReturnsEmptySettingsForNonGpt5Models() {
        #expect(getDefaultModelSettings("gpt-4.1") == ModelSettings())
    }

    @Test func testAgentUsesGpt5DefaultModelSettings() async {
        await withEnvironment([openAIDefaultModelEnvVariableName: "gpt-5"]) {
            let agent = Agent(name: "test")
            #expect(agent.model == nil)
            #expect(agent.modelSettings.reasoning?.json["effort"] == "low")
            #expect(agent.modelSettings.verbosity == "low")
        }
    }

    @Test func testAgentUsesModelSpecificSettingsForExplicitGpt5Models() {
        let agent = Agent(name: "test", model: "gpt-5")
        #expect(agent.model?.name == "gpt-5")
        #expect(agent.modelSettings == getDefaultModelSettings("gpt-5"))
        #expect(agent.modelSettings.reasoning?.json["effort"] == "low")
    }

    @Test func testAgentUsesEmptySettingsForExplicitNonGpt5Models() {
        let agent = Agent(name: "test", model: "gpt-4.1")
        #expect(agent.model?.name == "gpt-4.1")
        #expect(agent.modelSettings == ModelSettings())
    }

    @Test func testAgentCloneRecomputesImplicitSettingsWhenModelChanges() {
        let agent = Agent(name: "test", model: "gpt-5")
        let cloned = agent.clone { $0.model = "gpt-5.4-mini" }
        #expect(cloned.model?.name == "gpt-5.4-mini")
        #expect(cloned.modelSettings == getDefaultModelSettings("gpt-5.4-mini"))
        #expect(cloned.modelSettings.reasoning?.json["effort"] == "none")
    }

    @Test func testAgentClonePreservesExplicitSettingsWhenModelChanges() {
        let modelSettings = ModelSettings(temperature: 0.3)
        let agent = Agent(name: "test", model: "gpt-5", modelSettings: modelSettings)
        let cloned = agent.clone { $0.model = "gpt-5.4-mini" }
        #expect(cloned.model?.name == "gpt-5.4-mini")
        #expect(cloned.modelSettings == modelSettings)
    }

    @Test func testAgentResetsModelSettingsForNonGpt5Models() async {
        await withEnvironment([openAIDefaultModelEnvVariableName: "gpt-5"]) {
            let agent = Agent(name: "test", model: "gpt-4.1")
            #expect(agent.model?.name == "gpt-4.1")
            #expect(agent.modelSettings == ModelSettings())
        }
    }
}
