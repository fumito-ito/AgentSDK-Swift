import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_usage.py.
@Suite struct UsageTests {
    // MARK: - Defaults

    @Test func testUsageDefaultsCacheWriteTokensToZero() {
        let usage = Usage()

        #expect(usage.inputTokensDetails.cachedTokens == 0)
        #expect(usage.inputTokensDetails.cacheWriteTokens == 0)
    }

    @Test func testUsageRequestUsageEntriesDefaultEmpty() {
        let usage = Usage()
        #expect(usage.requestUsageEntries.isEmpty)
    }

    // MARK: - Raw usage snapshots

    @Test func testRawUsageSnapshotPreservesPresenceAndIsDetached() {
        var rawUsage: JSONObject = [
            "input_tokens": 3,
            "input_tokens_details": ["cached_tokens": 0],
            "provider_metric": nil,
        ]

        let snapshot = rawUsageSnapshot(.object(rawUsage))
        rawUsage["input_tokens_details"] = ["cached_tokens": 9]

        #expect(snapshot == [
            "input_tokens": 3,
            "input_tokens_details": ["cached_tokens": 0],
            "provider_metric": nil,
        ])
    }

    @Test func testRawUsageSnapshotRejectsNonObjectValues() {
        // Swift: JSONValue cannot hold arbitrary objects, so the equivalent invalid input is a
        // non-object payload.
        #expect(rawUsageSnapshot(.string("not usage")) == nil)
        #expect(rawUsageSnapshot(nil) == nil)
    }

    @Test func testRunnerRunCarriesRequestUsageEntries() async throws {
        let usage = Usage(
            requests: 1,
            inputTokens: 10,
            outputTokens: 5,
            totalTokens: 15,
            requestUsageEntries: [
                RequestUsage(
                    inputTokens: 10,
                    outputTokens: 5,
                    totalTokens: 15,
                    inputTokensDetails: InputTokensDetails(cachedTokens: 0, cacheWriteTokens: 0),
                    outputTokensDetails: OutputTokensDetails(reasoningTokens: 0)
                ),
            ]
        )
        let model = try ScriptedModel(steps: [.output([getTextMessage("done")])])
        model.setDefaultUsage(usage)
        let agent = Agent(name: "usage-agent", model: .instance(model))

        let result = try await Runner.run(agent, input: "hi")

        let propagated = result.contextWrapper.usage
        #expect(propagated.requests == 1)
        #expect(propagated.totalTokens == 15)
        #expect(propagated.requestUsageEntries.count == 1)
        let entry = try #require(propagated.requestUsageEntries.first)
        #expect(entry.inputTokens == 10)
        #expect(entry.outputTokens == 5)
        #expect(entry.totalTokens == 15)
    }

    // MARK: - Aggregation

    @Test func testUsageAddAggregatesAllFields() {
        var u1 = Usage(
            requests: 1,
            inputTokens: 10,
            inputTokensDetails: InputTokensDetails(cachedTokens: 3, cacheWriteTokens: 5),
            outputTokens: 20,
            outputTokensDetails: OutputTokensDetails(reasoningTokens: 5),
            totalTokens: 30
        )
        let u2 = Usage(
            requests: 2,
            inputTokens: 7,
            inputTokensDetails: InputTokensDetails(cachedTokens: 4, cacheWriteTokens: 6),
            outputTokens: 8,
            outputTokensDetails: OutputTokensDetails(reasoningTokens: 6),
            totalTokens: 15
        )

        u1.add(u2)

        #expect(u1.requests == 3)
        #expect(u1.inputTokens == 17)
        #expect(u1.outputTokens == 28)
        #expect(u1.totalTokens == 45)
        #expect(u1.inputTokensDetails.cachedTokens == 7)
        #expect(u1.inputTokensDetails.cacheWriteTokens == 11)
        #expect(u1.outputTokensDetails.reasoningTokens == 11)
    }

    @Test func testUsageAddAggregatesWithNoneValues() {
        var u1 = Usage()
        let u2 = Usage(
            requests: 2,
            inputTokens: 7,
            inputTokensDetails: InputTokensDetails(cachedTokens: 4, cacheWriteTokens: 0),
            outputTokens: 8,
            outputTokensDetails: OutputTokensDetails(reasoningTokens: 6),
            totalTokens: 15
        )

        u1.add(u2)

        #expect(u1.requests == 2)
        #expect(u1.inputTokens == 7)
        #expect(u1.outputTokens == 8)
        #expect(u1.totalTokens == 15)
        #expect(u1.inputTokensDetails.cachedTokens == 4)
        #expect(u1.outputTokensDetails.reasoningTokens == 6)
    }

    @Test func testRequestUsageCreation() {
        let requestUsage = RequestUsage(
            inputTokens: 100,
            outputTokens: 200,
            totalTokens: 300,
            inputTokensDetails: InputTokensDetails(cachedTokens: 10),
            outputTokensDetails: OutputTokensDetails(reasoningTokens: 20)
        )

        #expect(requestUsage.inputTokens == 100)
        #expect(requestUsage.outputTokens == 200)
        #expect(requestUsage.totalTokens == 300)
        #expect(requestUsage.inputTokensDetails.cachedTokens == 10)
        #expect(requestUsage.outputTokensDetails.reasoningTokens == 20)
    }

    @Test func testUsageAddPreservesSingleRequest() throws {
        var u1 = Usage()
        let u2 = Usage(
            requests: 1,
            inputTokens: 100,
            inputTokensDetails: InputTokensDetails(cachedTokens: 10),
            outputTokens: 200,
            outputTokensDetails: OutputTokensDetails(reasoningTokens: 20),
            totalTokens: 300
        )

        u1.add(u2)

        #expect(u1.requestUsageEntries.count == 1)
        let requestUsage = try #require(u1.requestUsageEntries.first)
        #expect(requestUsage.inputTokens == 100)
        #expect(requestUsage.outputTokens == 200)
        #expect(requestUsage.totalTokens == 300)
        #expect(requestUsage.inputTokensDetails.cachedTokens == 10)
        #expect(requestUsage.outputTokensDetails.reasoningTokens == 20)
    }

    @Test func testUsageAddIgnoresZeroTokenRequests() {
        var u1 = Usage()
        let u2 = Usage(requests: 1, inputTokens: 0, outputTokens: 0, totalTokens: 0)

        u1.add(u2)

        #expect(u1.requestUsageEntries.isEmpty)
    }

    @Test func testUsageAddIgnoresMultiRequestUsage() {
        var u1 = Usage()
        let u2 = Usage(
            requests: 3,
            inputTokens: 100,
            inputTokensDetails: InputTokensDetails(cachedTokens: 10),
            outputTokens: 200,
            outputTokensDetails: OutputTokensDetails(reasoningTokens: 20),
            totalTokens: 300
        )

        u1.add(u2)

        #expect(u1.requestUsageEntries.isEmpty)
    }

    @Test func testUsageAddMergesExistingRequestUsageEntries() {
        var u1 = Usage()
        u1.add(Usage(
            requests: 1,
            inputTokens: 100,
            inputTokensDetails: InputTokensDetails(cachedTokens: 10),
            outputTokens: 200,
            outputTokensDetails: OutputTokensDetails(reasoningTokens: 20),
            totalTokens: 300
        ))
        u1.add(Usage(
            requests: 1,
            inputTokens: 50,
            inputTokensDetails: InputTokensDetails(cachedTokens: 5),
            outputTokens: 75,
            outputTokensDetails: OutputTokensDetails(reasoningTokens: 10),
            totalTokens: 125
        ))

        #expect(u1.requestUsageEntries.count == 2)
        #expect(u1.requestUsageEntries[0].inputTokens == 100)
        #expect(u1.requestUsageEntries[0].outputTokens == 200)
        #expect(u1.requestUsageEntries[0].totalTokens == 300)
        #expect(u1.requestUsageEntries[1].inputTokens == 50)
        #expect(u1.requestUsageEntries[1].outputTokens == 75)
        #expect(u1.requestUsageEntries[1].totalTokens == 125)
    }

    @Test func testUsageAddPreservesExistingEntriesWhenTopLevelAlsoSet() throws {
        var u1 = Usage()
        let u2 = Usage(
            requests: 1,
            inputTokens: 100,
            outputTokens: 50,
            totalTokens: 150,
            requestUsageEntries: [
                RequestUsage(
                    inputTokens: 100,
                    outputTokens: 50,
                    totalTokens: 150,
                    inputTokensDetails: InputTokensDetails(cachedTokens: 10),
                    outputTokensDetails: OutputTokensDetails(reasoningTokens: 5)
                ),
            ]
        )

        u1.add(u2)

        #expect(u1.requestUsageEntries.count == 1)
        let entry = try #require(u1.requestUsageEntries.first)
        #expect(entry.inputTokensDetails.cachedTokens == 10)
        #expect(entry.outputTokensDetails.reasoningTokens == 5)
    }

    @Test func testUsageAddDetachesPreExistingRequestUsageEntries() {
        var sourceEntry = RequestUsage(
            inputTokens: 100,
            outputTokens: 50,
            totalTokens: 150,
            inputTokensDetails: InputTokensDetails(cachedTokens: 10),
            outputTokensDetails: OutputTokensDetails(reasoningTokens: 5)
        )
        let source = Usage(
            requests: 1, inputTokens: 100, outputTokens: 50, totalTokens: 150,
            requestUsageEntries: [sourceEntry]
        )
        var aggregate = Usage()

        aggregate.add(source)
        sourceEntry.inputTokens = 999
        sourceEntry.inputTokensDetails.cachedTokens = 99

        #expect(aggregate.requestUsageEntries[0].inputTokens == 100)
        #expect(aggregate.requestUsageEntries[0].inputTokensDetails.cachedTokens == 10)
        #expect(aggregate.requestUsageEntries[0].outputTokensDetails.reasoningTokens == 5)
    }

    @Test func testAnthropicCostCalculationScenario() {
        var usage = Usage()
        usage.add(Usage(requests: 1, inputTokens: 100_000, outputTokens: 50_000, totalTokens: 150_000))
        usage.add(Usage(requests: 1, inputTokens: 150_000, outputTokens: 75_000, totalTokens: 225_000))
        usage.add(Usage(requests: 1, inputTokens: 80_000, outputTokens: 40_000, totalTokens: 120_000))

        #expect(usage.requests == 3)
        #expect(usage.inputTokens == 330_000)
        #expect(usage.outputTokens == 165_000)
        #expect(usage.totalTokens == 495_000)
        #expect(usage.requestUsageEntries.map(\.inputTokens) == [100_000, 150_000, 80_000])
        for request in usage.requestUsageEntries {
            #expect(request.inputTokens < 200_000)
            #expect(request.outputTokens < 200_000)
        }
    }

    // MARK: - Normalization

    @Test func testUsageNormalizesChatCompletionsTypes() {
        // Chat Completions uses prompt/completion token details; the converter maps them onto the
        // Responses API detail types.
        let usage = Usage(
            requests: 1,
            inputTokens: 200,
            inputTokensDetails: InputTokensDetails(chatCompletionsPromptDetails: [
                "audio_tokens": 10, "cached_tokens": 50, "cache_write_tokens": 7,
            ]),
            outputTokens: 150,
            outputTokensDetails: OutputTokensDetails(chatCompletionsCompletionDetails: [
                "accepted_prediction_tokens": 5, "audio_tokens": 10,
                "reasoning_tokens": 100, "rejected_prediction_tokens": 2,
            ]),
            totalTokens: 350
        )

        #expect(usage.inputTokensDetails.cachedTokens == 50)
        #expect(usage.inputTokensDetails.cacheWriteTokens == 7)
        #expect(usage.outputTokensDetails.reasoningTokens == 100)
    }

    @Test func testUsageNormalizesNoneTokenDetails() {
        let usage = Usage(
            requests: 1,
            inputTokens: 100,
            inputTokensDetails: InputTokensDetails(chatCompletionsPromptDetails: nil),
            outputTokens: 50,
            outputTokensDetails: OutputTokensDetails(chatCompletionsCompletionDetails: ["reasoning_tokens": nil]),
            totalTokens: 150
        )
        #expect(usage.inputTokensDetails.cachedTokens == 0)
        #expect(usage.inputTokensDetails.cacheWriteTokens == 0)
        #expect(usage.outputTokensDetails.reasoningTokens == 0)
    }

    // MARK: - Serialization

    @Test func testUsageSerializationPreservesCacheWriteTokens() {
        let usage = Usage(
            requests: 1,
            inputTokens: 20,
            inputTokensDetails: InputTokensDetails(cachedTokens: 3, cacheWriteTokens: 7),
            outputTokens: 5,
            totalTokens: 25,
            requestUsageEntries: [
                RequestUsage(
                    inputTokens: 20,
                    outputTokens: 5,
                    totalTokens: 25,
                    inputTokensDetails: InputTokensDetails(cachedTokens: 3, cacheWriteTokens: 7),
                    outputTokensDetails: OutputTokensDetails(reasoningTokens: 0)
                ),
            ]
        )

        let serialized = serializeUsage(usage)
        let restored = deserializeUsage(serialized)

        #expect(serialized["input_tokens_details"] == [["cached_tokens": 3, "cache_write_tokens": 7]])
        #expect(restored.inputTokensDetails.cacheWriteTokens == 7)
        #expect(restored.requestUsageEntries[0].inputTokensDetails.cacheWriteTokens == 7)
        #expect(restored == usage)
    }

    @Test func testUsageDeserializationDefaultsLegacyCacheWriteTokens() {
        let restored = deserializeUsage([
            "requests": 1,
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
            "input_tokens_details": [["cached_tokens": 3]],
            "request_usage_entries": [
                [
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "total_tokens": 25,
                    "input_tokens_details": ["cached_tokens": 3],
                ],
            ],
        ])

        #expect(restored.inputTokensDetails.cachedTokens == 3)
        #expect(restored.inputTokensDetails.cacheWriteTokens == 0)
        #expect(restored.requestUsageEntries[0].inputTokensDetails.cachedTokens == 3)
        #expect(restored.requestUsageEntries[0].inputTokensDetails.cacheWriteTokens == 0)
    }

    @Test func testUsageSnapshotDeltaAndSpanPreserveCacheWriteTokens() {
        let start = Usage(
            requests: 1,
            inputTokens: 10,
            inputTokensDetails: InputTokensDetails(cachedTokens: 3, cacheWriteTokens: 2),
            outputTokens: 4,
            totalTokens: 14
        )
        let end = Usage(
            requests: 2,
            inputTokens: 30,
            inputTokensDetails: InputTokensDetails(cachedTokens: 8, cacheWriteTokens: 9),
            outputTokens: 10,
            totalTokens: 40
        )

        let snapshot = snapshotUsage(start)
        let delta = usageDelta(snapshot, end)
        let spanUsage = modelUsageToSpanUsage(delta)

        #expect(snapshot.inputTokensDetails.cacheWriteTokens == 2)
        #expect(delta.inputTokensDetails.cachedTokens == 5)
        #expect(delta.inputTokensDetails.cacheWriteTokens == 7)
        #expect(spanUsage["input_tokens_details"] == ["cached_tokens": 5, "cache_write_tokens": 7])
    }
}
