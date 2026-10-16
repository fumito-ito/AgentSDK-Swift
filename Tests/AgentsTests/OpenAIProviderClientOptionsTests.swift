import Foundation
import Testing
@testable import Agents

// Ported from tests/test_openai_provider_client_options.py.
//
// Swift: there is no websocket transport, and `baseURL` is a `URL`, so the websocket and
// empty-base-URL cases are skipped. The default client is installed briefly instead of patching
// `get_default_openai_client`; no other suite reads it.
@Suite(.serialized) struct OpenAIProviderClientOptionsTests {
    @Test(arguments: [("organization", "org-test"), ("project", "proj-test")])
    func testOpenAIProviderRejectsIgnoredOptionsWithExplicitClient(option: String, value: String) {
        let client = OpenAIClient(apiKey: "test_key")
        expectError(UserError.self, containing: "organization, or project") {
            if option == "organization" {
                _ = try OpenAIProvider(openAIClient: client, organization: value)
            } else {
                _ = try OpenAIProvider(openAIClient: client, project: value)
            }
        }
    }

    private func withDefaultClient<T>(_ body: (OpenAIClient) throws -> T) rethrows -> T {
        let previous = openAIDefaults.value.openAIClient
        let defaultClient = OpenAIClient(apiKey: "sk-default")
        openAIDefaults.mutate { $0.openAIClient = defaultClient }
        defer { openAIDefaults.mutate { $0.openAIClient = previous } }
        return try body(defaultClient)
    }

    @Test(arguments: ["api_key", "base_url", "organization", "project"])
    func testOpenAIProviderExplicitOptionsOverrideDefaultClient(option: String) throws {
        try withDefaultClient { defaultClient in
            let provider: OpenAIProvider
            switch option {
            case "api_key": provider = try OpenAIProvider(apiKey: "sk-provider")
            case "base_url": provider = try OpenAIProvider(baseURL: URL(string: "https://provider.example.test/v1"))
            case "organization": provider = try OpenAIProvider(organization: "org-provider")
            default: provider = try OpenAIProvider(project: "proj-provider")
            }

            let client = provider.getClient()

            #expect(client !== defaultClient)
            switch option {
            case "api_key": #expect(client.apiKey == "sk-provider")
            case "base_url": #expect(client.baseURL.absoluteString == "https://provider.example.test/v1")
            case "organization": #expect(client.organization == "org-provider")
            default: #expect(client.project == "proj-provider")
            }
        }
    }

    @Test func testOpenAIProviderPreservesExplicitEmptyOptions() throws {
        try withDefaultClient { defaultClient in
            let provider = try OpenAIProvider(apiKey: "")
            let client = provider.getClient()
            #expect(client !== defaultClient)
            #expect(client.apiKey == "")
        }
    }
}
