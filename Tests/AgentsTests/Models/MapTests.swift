import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/models/test_map.py.
//
// Swift: the `litellm/` and `any-llm/` providers are out of scope (D2), the websocket transport is
// not ported, and prefix modes are enums, so those tests are skipped. Instead of replacing
// `OpenAIProvider` with a fake, the tests inspect the model name the real provider builds.
@Suite struct MapTests {
    private final class RecordingProvider: ModelProvider, @unchecked Sendable {
        let requested = SharedBox<[String?]>([])
        let model: any Model

        init(model: any Model) {
            self.model = model
        }

        func getModel(_ modelName: String?) throws -> any Model {
            requested.mutate { $0.append(modelName) }
            return model
        }
    }

    private final class CloseTrackingProvider: ModelProvider, @unchecked Sendable {
        let name: String
        let error: (any Error)?
        let log: SharedBox<[String]>

        init(_ name: String, error: (any Error)? = nil, log: SharedBox<[String]>) {
            self.name = name
            self.error = error
            self.log = log
        }

        func getModel(_ modelName: String?) throws -> any Model {
            try ScriptedModel()
        }

        func aclose() async throws {
            log.mutate { $0.append(name) }
            if let error { throw error }
        }
    }

    private func responsesModelName(_ model: any Model) -> String? {
        (model as? OpenAIResponsesModel)?.model
    }

    @Test func testNoPrefixIsOpenAI() throws {
        let agent = Agent(name: "test", instructions: "", model: "gpt-4o")
        #expect(try getModel(agent, RunConfig()) is OpenAIResponsesModel)
    }

    @Test func testOpenAIPrefixIsOpenAI() throws {
        let agent = Agent(name: "test", instructions: "", model: "openai/gpt-4o")
        #expect(try getModel(agent, RunConfig()) is OpenAIResponsesModel)
    }

    @Test func testMultiProviderForwardsOpenAIBufferStreamedToolCallsToChatModel() throws {
        let provider = try MultiProvider(openAIClient: OpenAIClient(apiKey: "test_key"), openAIUseResponses: false, openAIBufferStreamedToolCalls: true)

        let model = try provider.getModel("gpt-4o")

        let chatModel = try #require(model as? OpenAIChatCompletionsModel)
        #expect(chatModel.bufferStreamedToolCalls)
    }

    @Test func testOpenAIPrefixDefaultsToAliasMode() throws {
        let provider = try MultiProvider(openAIAPIKey: "test_key")
        #expect(responsesModelName(try provider.getModel("openai/gpt-4o")) == "gpt-4o")
    }

    @Test func testOpenAIPrefixCanBePreservedAsLiteralModelId() throws {
        let provider = try MultiProvider(openAIAPIKey: "test_key", openAIPrefixMode: .modelId)
        #expect(responsesModelName(try provider.getModel("openai/gpt-4o")) == "openai/gpt-4o")
    }

    @Test func testUnknownPrefixDefaultsToError() {
        expectError(UserError.self, containing: "Unknown prefix: openrouter") {
            _ = try MultiProvider().getModel("openrouter/openai/gpt-4o")
        }
    }

    @Test func testUnknownPrefixCanBePreservedForOpenAICompatibleModelIds() throws {
        let provider = try MultiProvider(openAIAPIKey: "test_key", unknownPrefixMode: .modelId)
        #expect(responsesModelName(try provider.getModel("openrouter/openai/gpt-4o")) == "openrouter/openai/gpt-4o")
    }

    @Test func testProviderMapEntriesOverrideOpenAIPrefixMode() throws {
        let custom = RecordingProvider(model: try ScriptedModel())
        let providerMap = MultiProviderMap()
        providerMap.addProvider("openai", custom)

        _ = try MultiProvider(providerMap: providerMap, openAIPrefixMode: .modelId).getModel("openai/gpt-4o")

        #expect(custom.requested.value == ["gpt-4o"])
    }

    @Test func testProviderMapRoutesToCustomProvider() throws {
        // Ports test_provider_map_routes_to_falsey_provider; Swift has no truthiness.
        let expectedModel = try ScriptedModel()
        let custom = RecordingProvider(model: expectedModel)
        let providerMap = MultiProviderMap()
        providerMap.addProvider("custom", custom)

        let result = try MultiProvider(providerMap: providerMap).getModel("custom/test-model")

        #expect(result as? ScriptedModel === expectedModel)
        #expect(custom.requested.value == ["test-model"])
    }

    @Test func testMultiProviderAcloseContinuesAndPreservesFirstFailure() async throws {
        let log = SharedBox<[String]>([])
        let providerMap = MultiProviderMap()
        providerMap.addProvider("failing", CloseTrackingProvider("failing", error: TestError("close failed"), log: log))
        providerMap.addProvider("later", CloseTrackingProvider("later", error: ValueError("later close failed"), log: log))
        let provider = try MultiProvider(providerMap: providerMap, openAIAPIKey: "test")

        let error = await expectErrorAsync(TestError.self) {
            try await provider.aclose()
        }

        #expect(error?.description == "close failed")
        #expect(log.value == ["failing", "later"])
    }

    @Test func testMultiProviderAclosePropagatesCancellation() async throws {
        // Ports test_multi_provider_aclose_propagates_hybrid_cancellation.
        let log = SharedBox<[String]>([])
        let providerMap = MultiProviderMap()
        providerMap.addProvider("failing", CloseTrackingProvider("failing", error: TestError("close failed"), log: log))
        providerMap.addProvider("cancelling", CloseTrackingProvider("cancelling", error: CancellationError(), log: log))
        providerMap.addProvider("later", CloseTrackingProvider("later", log: log))
        let provider = try MultiProvider(providerMap: providerMap, openAIAPIKey: "test")

        await expectErrorAsync(CancellationError.self) {
            try await provider.aclose()
        }

        #expect(log.value == ["failing", "cancelling"])
    }
}
