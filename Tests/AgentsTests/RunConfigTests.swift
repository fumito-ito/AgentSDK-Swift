import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_config.py.
//
// Swift: sandbox configuration is out of scope (D2). Dictionary-form `RunConfig` arguments,
// async/invalid `output_guardrail_blocked_message` values, and invalid enum strings cannot be
// expressed in Swift, so those cases are skipped. Dictionary settings are covered through the
// `init(json:)` initializers of the nested settings types.
@Suite(.serialized) struct RunConfigTests {
    /// A model provider that always returns the same model and records the requested name.
    private final class DummyProvider: ModelProvider, @unchecked Sendable {
        private let lock = NSLock()
        private var requested: String?
        let modelToReturn: any Model

        init(modelToReturn: (any Model)? = nil) throws {
            self.modelToReturn = try modelToReturn ?? ScriptedModel()
        }

        var lastRequested: String? { lock.locked { requested } }

        func getModel(_ modelName: String?) throws -> any Model {
            lock.locked { requested = modelName }
            return modelToReturn
        }
    }

    // MARK: - Configuration values

    @Test func testRunConfigNormalizesFirstPartyDictionarySettings() throws {
        let config = RunConfig(
            modelSettings: try ModelSettings(json: ["reasoning": ["context": "all_turns"], "temperature": 0.0]),
            sessionSettings: SessionSettings(limit: 5),
            toolExecution: try ToolExecutionConfig(maxFunctionToolConcurrency: 2)
        )

        #expect(config.modelSettings?.reasoning?.context == "all_turns")
        #expect(config.modelSettings?.temperature == 0.0)
        #expect(config.sessionSettings?.limit == 5)
        #expect(config.toolExecution?.maxFunctionToolConcurrency == 2)
    }

    @Test func testRunConfigPreservesTypedConfigurationInstances() {
        let settings = ModelSettings(temperature: 0.2)
        let sessionSettings = SessionSettings(limit: 3)
        let config = RunConfig(modelSettings: settings, sessionSettings: sessionSettings)

        #expect(config.modelSettings == settings)
        #expect(config.sessionSettings == sessionSettings)
    }

    @Test func testRunConfigAcceptsOutputGuardrailBlockedMessageCustomizers() {
        #expect(RunConfig(outputGuardrailBlockedMessage: "custom").outputGuardrailBlockedMessage == "custom")
    }

    @Test func testRunConfigRejectsUnknownFirstPartyDictionaryFields() {
        expectError(TypeError.self, containing: "Unknown model settings: temperatur") {
            _ = try ModelSettings(json: ["temperatur": 0.2])
        }
    }

    @Test func testRunnerAcceptsDictionaryRunConfiguration() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("done")])])
        let agent = Agent(name: "test", model: .instance(model))

        let result = try await Runner.run(agent, input: "hello", runConfig: RunConfig(modelSettings: try ModelSettings(json: ["temperature": 0.0])))

        #expect(result.finalOutput as? String == "done")
    }

    // MARK: - Model resolution

    @Test func testModelProviderOnRunConfigIsUsedForAgentModelName() async throws {
        let scriptedModel = try ScriptedModel(steps: [.output([getTextMessage("from-provider")])])
        let provider = try DummyProvider(modelToReturn: scriptedModel)
        let agent = Agent(name: "test", model: "test-model")

        let result = try await Runner.run(agent, input: "any", runConfig: RunConfig(modelProvider: provider))

        #expect(provider.lastRequested == "test-model")
        #expect(result.finalOutput as? String == "from-provider")
    }

    @Test func testRunConfigModelNameOverrideTakesPrecedence() async throws {
        let scriptedModel = try ScriptedModel(steps: [.output([getTextMessage("override-name")])])
        let provider = try DummyProvider(modelToReturn: scriptedModel)
        let agent = Agent(name: "test", model: "agent-model")

        let result = try await Runner.run(agent, input: "any", runConfig: RunConfig(model: "override-name", modelProvider: provider))

        #expect(provider.lastRequested == "override-name")
        #expect(result.finalOutput as? String == "override-name")
    }

    @Test(arguments: [("gpt-5", "low"), ("gpt-5.6", "none")])
    func testRunConfigModelNameOverrideUsesModelSpecificDefaultSettings(modelName: String, reasoningEffort: String) async throws {
        try await withEnvironment(["OPENAI_DEFAULT_MODEL": "gpt-5.4-mini"]) {
            let scriptedModel = try ScriptedModel(steps: [.output([getTextMessage("override-name")])])
            let provider = try DummyProvider(modelToReturn: scriptedModel)
            let agent = Agent(name: "test")

            let result = try await Runner.run(agent, input: "any", runConfig: RunConfig(model: .name(modelName), modelProvider: provider))

            #expect(result.finalOutput as? String == "override-name")
            let settings = try #require(scriptedModel.calls.first?.modelSettings)
            #expect(settings.reasoning?.effort == reasoningEffort)
            #expect(settings.verbosity == "low")
        }
    }

    @Test func testRunConfigModelSettingsOverrideImplicitModelSpecificDefaults() async throws {
        try await withEnvironment(["OPENAI_DEFAULT_MODEL": "gpt-5.4-mini"]) {
            let scriptedModel = try ScriptedModel(steps: [.output([getTextMessage("override-name")])])
            let provider = try DummyProvider(modelToReturn: scriptedModel)
            let agent = Agent(name: "test")
            let runConfig = RunConfig(model: "gpt-5", modelProvider: provider, modelSettings: ModelSettings(temperature: 0.3))

            let result = try await Runner.run(agent, input: "any", runConfig: runConfig)

            #expect(result.finalOutput as? String == "override-name")
            let settings = try #require(scriptedModel.calls.first?.modelSettings)
            #expect(settings.reasoning?.effort == "low")
            #expect(settings.verbosity == "low")
            #expect(settings.temperature == 0.3)
        }
    }

    @Test func testRunConfigModelOverrideObjectTakesPrecedence() async throws {
        let scriptedModel = try ScriptedModel(steps: [.output([getTextMessage("override-object")])])
        let agent = Agent(name: "test", model: "agent-model")

        let result = try await Runner.run(agent, input: "any", runConfig: RunConfig(model: .instance(scriptedModel)))

        #expect(result.finalOutput as? String == "override-object")
    }

    @Test func testAgentModelObjectIsUsedWhenPresent() async throws {
        let scriptedModel = try ScriptedModel(steps: [.output([getTextMessage("from-agent-object")])])
        let provider = try DummyProvider()
        let agent = Agent(name: "test", model: .instance(scriptedModel))

        let result = try await Runner.run(agent, input: "any", runConfig: RunConfig(modelProvider: provider))

        #expect(provider.lastRequested == nil)
        #expect(result.finalOutput as? String == "from-agent-object")
    }

    // MARK: - Sensitive trace data

    @Test func testTraceIncludeSensitiveDataDefaultsToTrueWhenEnvNotSet() async {
        await withEnvironment(["OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA": nil]) {
            #expect(RunConfig().traceIncludeSensitiveData)
        }
    }

    @Test(arguments: [
        ("true", true), ("True", true), ("1", true), ("yes", true), ("on", true),
        ("false", false), ("False", false), ("0", false), ("no", false), ("off", false),
    ])
    func testTraceIncludeSensitiveDataFollowsEnvValue(envValue: String, expected: Bool) async {
        await withEnvironment(["OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA": envValue]) {
            #expect(RunConfig().traceIncludeSensitiveData == expected)
        }
    }

    @Test func testTraceIncludeSensitiveDataExplicitOverrideTakesPrecedence() async {
        await withEnvironment(["OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA": "false"]) {
            #expect(RunConfig(traceIncludeSensitiveData: true).traceIncludeSensitiveData)
        }
        await withEnvironment(["OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA": "true"]) {
            #expect(!RunConfig(traceIncludeSensitiveData: false).traceIncludeSensitiveData)
        }
    }

    // MARK: - Tool execution and policies

    @Test func testToolExecutionConfigRejectsInvalidFunctionToolConcurrency() {
        expectError(ValueError.self, containing: "tool_execution.max_function_tool_concurrency must be at least 1") {
            _ = try ToolExecutionConfig(maxFunctionToolConcurrency: 0)
        }
    }

    @Test func testToolExecutionConfigIsPublicFromAgentsPackage() throws {
        let config = RunConfig(toolExecution: try ToolExecutionConfig(maxFunctionToolConcurrency: 2))
        #expect(config.toolExecution?.maxFunctionToolConcurrency == 2)
    }

    @Test func testToolNotFoundBehaviorDefaultsToRaiseError() {
        #expect(RunConfig().toolNotFoundBehavior == .raiseError)
    }

    @Test func testToolNotFoundBehaviorIsPublicFromAgentsPackage() {
        #expect(RunConfig(toolNotFoundBehavior: .returnErrorToModel).toolNotFoundBehavior.rawValue == "return_error_to_model")
    }

    @Test func testToolNameCollisionPolicyDefaultsToWarn() {
        #expect(RunConfig().toolNameCollisionPolicy == .warn)
    }

    @Test func testToolNameCollisionPolicyIsPublicFromAgentsPackage() {
        #expect(RunConfig(toolNameCollisionPolicy: .error).toolNameCollisionPolicy.rawValue == "error")
    }
}
