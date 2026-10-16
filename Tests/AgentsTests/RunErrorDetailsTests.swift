import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_error_details.py.
@Suite struct RunErrorDetailsTests {
    private func makeAgent() throws -> (Agent, ScriptedModel) {
        let model = try ScriptedModel(steps: [
            .output([getTextMessage("1"), getFunctionToolCall("foo", arguments: #"{"a": "b"}"#)]),
            .output([getTextMessage("done")]),
        ])
        return (Agent(name: "test", tools: [getFunctionTool(name: "foo", returnValue: "res")], model: .instance(model)), model)
    }

    @Test func testRunErrorIncludesData() async throws {
        let (agent, _) = try makeAgent()

        let error = await expectErrorAsync(MaxTurnsExceeded.self) {
            _ = try await Runner.run(agent, input: "hello", maxTurns: 1)
        }

        let data = try #require(error?.runData)
        #expect(data.lastAgent === agent)
        #expect(data.rawResponses.count == 1)
        #expect(!data.newItems.isEmpty)
    }

    @Test func testStreamedRunErrorIncludesData() async throws {
        let (agent, _) = try makeAgent()
        let result = try Runner.runStreamed(agent, input: "hello", maxTurns: 1)

        let error = await expectErrorAsync(MaxTurnsExceeded.self) {
            for try await _ in result.streamEvents() {}
        }

        let data = try #require(error?.runData)
        #expect(data.lastAgent === agent)
        #expect(data.rawResponses.count == 1)
        #expect(!data.newItems.isEmpty)
    }
}
