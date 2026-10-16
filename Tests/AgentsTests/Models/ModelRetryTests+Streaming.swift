import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing
@testable import Agents

// Continuation of tests/models/test_model_retry.py: streamed retries and websocket retry flags.
//
// Swift: `contextvars` scoping tests, provider-iterator `aclose`/`close` bookkeeping, and blocked
// stream cleanup tests are skipped; `AsyncThrowingStream` termination replaces explicit closing.
extension ModelRetryTests {
    private static let created: TResponseStreamEvent = ["type": "response.created"]
    private static let inProgress: TResponseStreamEvent = ["type": "response.in_progress"]

    private func streamRetry(
        _ recorder: Recorder,
        settings: ModelRetrySettings?,
        advice: @escaping @Sendable (ModelRetryAdviceRequest) -> ModelRetryAdvice? = { _ in nil },
        previousResponseId: String? = nil,
        timeout: Double? = nil,
        rewindAllowed: Bool = true,
        failedAttempts: SharedBox<Int>? = nil,
        _ makeEvents: @escaping @Sendable (Int) -> AsyncThrowingStream<TResponseStreamEvent, any Error>
    ) -> AsyncThrowingStream<TResponseStreamEvent, any Error> {
        var onFailedAttempts: (@Sendable (Int) -> Void)?
        if let failedAttempts {
            onFailedAttempts = { count in failedAttempts.mutate { $0 = count } }
        }
        let getStream: @Sendable () -> AsyncThrowingStream<TResponseStreamEvent, any Error> = { makeEvents(recorder.call()) }
        let rewind: @Sendable () async throws -> Void = {
            if !rewindAllowed { Issue.record("Streaming retry should not rewind state") }
            recorder.rewind()
        }
        return streamResponseWithRetry(
            getStream: getStream,
            rewind: rewind,
            retrySettings: settings,
            getRetryAdvice: advice,
            previousResponseId: previousResponseId,
            conversationId: nil,
            timeout: timeout,
            onFailedAttempts: onFailedAttempts
        )
    }

    /// A stream that fails on the first attempt and yields `events` afterwards.
    private typealias StreamFactory = @Sendable (Int) -> AsyncThrowingStream<TResponseStreamEvent, any Error>

    private func failFirst(_ error: @escaping @Sendable () -> any Error, events: [TResponseStreamEvent] = [created]) -> StreamFactory {
        { attempt in
            makeStream { continuation in
                if attempt == 1 { throw error() }
                for event in events { continuation.yield(event) }
            }
        }
    }

    private func collect(_ recorder: Recorder, _ stream: () -> AsyncThrowingStream<TResponseStreamEvent, any Error>) async throws -> [TResponseStreamEvent] {
        try await recorder.withRecordedSleeps {
            var events: [TResponseStreamEvent] = []
            for try await event in stream() {
                events.append(event)
            }
            return events
        }
    }

    // MARK: - Streamed retries

    @Test func testStreamResponseWithRetryRetriesBeforeFirstEvent() async throws {
        let recorder = Recorder()
        let failedAttempts = SharedBox(0)
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0.25, jitter: false), policy: retryPolicies.networkError())

        let events = try await collect(recorder) {
            streamRetry(recorder, settings: settings, failedAttempts: failedAttempts, failFirst { self.connectionError() })
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.sleeps.value == [0.25])
        #expect(failedAttempts.value == 1)
        #expect(events == [Self.created])
    }

    @Test func testStreamResponseWithRetryRetriesTimeoutBeforeOutput() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0), policy: retryPolicies.networkError())

        let events = try await collect(recorder) {
            streamRetry(recorder, settings: settings, timeout: 0.01) { attempt in
                makeStream { continuation in
                    if attempt == 1 { try await waitForever() }
                    continuation.yield(Self.created)
                }
            }
        }

        #expect(recorder.calls.value == 2)
        #expect(events == [Self.created])
    }

    @Test func testTimedStreamEarlyCloseRunsProducerCleanup() async throws {
        // Ports test_timed_stream_early_close_allows_cooperative_cleanup.
        let cleaned = AsyncEvent()
        do {
            let stream = streamRetry(Recorder(), settings: nil, timeout: 30) { _ in
                makeStream { continuation in
                    defer { cleaned.set() }
                    continuation.yield(Self.created)
                    try await waitForever()
                }
            }
            for try await event in stream {
                #expect(event == Self.created)
                break
            }
        }

        try await cleaned.wait(timeout: 5)
    }

    @Test func testTimedStreamParentCancellationAllowsCooperativeCleanup() async throws {
        let readStarted = AsyncEvent()
        let cleaned = AsyncEvent()
        let task = Task {
            let stream = streamRetry(Recorder(), settings: nil, timeout: 1) { _ in
                makeStream { _ in
                    defer { cleaned.set() }
                    readStarted.set()
                    try await waitForever()
                }
            }
            for try await _ in stream {}
        }
        try await readStarted.wait()

        task.cancel()

        // Swift: a cancelled consumer sees the end of an `AsyncThrowingStream` rather than a thrown
        // `CancellationError`; what matters is that no other error replaces the cancellation.
        try await expectOnlyCancellation(task)
        try await cleaned.wait(timeout: 5)
    }

    @Test func testStreamResponseWithRetryDoesNotRetryTimeoutAfterOutput() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0), policy: retryPolicies.networkError())

        await expectErrorAsync(ModelTimeoutError.self) {
            _ = try await collect(recorder) {
                streamRetry(recorder, settings: settings, timeout: 0.01) { _ in
                    makeStream { continuation in
                        continuation.yield(["type": "response.output_item.added"])
                        try await waitForever()
                    }
                }
            }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testStreamTimeoutReportsConfiguredAttemptTimeout() async throws {
        let stream = streamRetry(Recorder(), settings: nil, timeout: 0.05) { _ in
            makeStream { continuation in
                continuation.yield(Self.created)
                try await waitForever()
            }
        }
        var iterator = stream.makeAsyncIterator()

        #expect(try await iterator.next() == Self.created)
        try await sleep(seconds: 0.03)
        let error = await expectErrorAsync(ModelTimeoutError.self) {
            _ = try await iterator.next()
        }

        #expect(error?.timeoutSeconds == 0.05)
        #expect(error?.message == "Model call timed out after 0.05 seconds.")
    }

    @Test func testStreamResponseWithRetryPreservesParentCancellationOverCleanupError() async throws {
        let started = AsyncEvent()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0), policy: retryPolicies.networkError())
        let task = Task {
            let stream = streamRetry(Recorder(), settings: settings, timeout: 10) { _ in
                makeStream { _ in
                    started.set()
                    do {
                        try await waitForever()
                    } catch {
                        throw TestError("stream cleanup failed")
                    }
                }
            }
            for try await _ in stream {}
        }
        try await started.wait()

        task.cancel()

        try await expectOnlyCancellation(task)
    }

    /// Awaits `task`, failing if it throws anything other than `CancellationError`.
    private func expectOnlyCancellation(_ task: Task<Void, any Error>, sourceLocation: SourceLocation = #_sourceLocation) async throws {
        do {
            try await task.value
        } catch is CancellationError {
            return
        } catch {
            Issue.record("Expected cancellation, got \(error)", sourceLocation: sourceLocation)
        }
        #expect(task.isCancelled, sourceLocation: sourceLocation)
    }
}

// MARK: - Streamed provider retry flags

extension ModelRetryTests {
    @Test func testStreamResponseWithRetryKeepsProviderRetriesOnFirstAttempt() async throws {
        // Also covers test_stream_response_with_retry_does_not_leak_provider_retry_disable_to_consumer.
        let recorder = Recorder()
        let consumerFlags = SharedBox<[Bool]>([])

        let events = try await recorder.withRecordedSleeps {
            var events: [TResponseStreamEvent] = []
            let stream = streamRetry(recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError()), failFirst { self.connectionError() })
            for try await event in stream {
                consumerFlags.mutate { $0.append(ModelRetryScope.providerManagedRetriesDisabled) }
                events.append(event)
            }
            return events
        }

        #expect(recorder.providerRetryFlags.value == [false, true])
        #expect(consumerFlags.value == [false])
        #expect(events == [Self.created])
    }

    @Test func testStreamResponseWithRetryDisablesProviderRetriesOnFirstStatefulProviderHint() async throws {
        let recorder = Recorder()

        let events = try await collect(recorder) {
            streamRetry(
                recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.providerSuggested()),
                advice: fixedAdvice(ModelRetryAdvice(suggested: true, replaySafety: "safe")), previousResponseId: "resp_prev",
                failFirst { self.connectionError() }
            )
        }

        #expect(recorder.providerRetryFlags.value == [true, true])
        #expect(events == [Self.created])
    }

    @Test func testStreamResponseWithRetryDisablesStatefulProviderRetriesWithNarrowPolicy() async throws {
        let recorder = Recorder()

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await collect(recorder) {
                streamRetry(
                    recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.httpStatus([429])),
                    previousResponseId: "resp_prev", rewindAllowed: false
                ) { _ in makeStream { _ in throw self.connectionError() } }
            }
        }

        #expect(recorder.calls.value == 1)
        #expect(recorder.providerRetryFlags.value == [true])
    }

    @Test func testStreamResponseWithRetryKeepsProviderRetriesWithoutRunnerPolicy() async throws {
        let recorder = Recorder()

        let events = try await collect(recorder) {
            streamRetry(recorder, settings: ModelRetrySettings(maxRetries: 2)) { _ in
                makeStream { continuation in continuation.yield(Self.created) }
            }
        }

        #expect(recorder.calls.value == 1)
        #expect(recorder.providerRetryFlags.value == [false])
        #expect(events == [Self.created])
    }

    // MARK: - Websocket pre-event retry flags

    @Test(arguments: [
        (ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError()), true),
        (ModelRetrySettings(maxRetries: 0, policy: retryPolicies.never()), true),
    ] as [(ModelRetrySettings, Bool)])
    func testGetResponseWithRetryDisablesWebsocketPreEventRetries(settings: ModelRetrySettings, expected: Bool) async throws {
        // Ports _disables_websocket_pre_event_retries_when_runner_managed and _when_disabled.
        let recorder = Recorder()
        let websocketFlags = SharedBox<[Bool]>([])

        _ = try await retry(recorder, settings: settings) { call in
            websocketFlags.mutate { $0.append(ModelRetryScope.websocketPreEventRetriesDisabled) }
            if call == 1, settings.maxRetries == 1 { throw self.connectionError() }
            return self.okResponse("resp_disable_ws_hidden_retry")
        }

        #expect(websocketFlags.value.allSatisfy { $0 == expected })
        #expect(!websocketFlags.value.isEmpty)
    }

    @Test(arguments: [
        ModelRetrySettings(maxRetries: 1, policy: retryPolicies.httpStatus([429])),
        ModelRetrySettings(maxRetries: 1, policy: retryPolicies.all(retryPolicies.networkError(), retryPolicies.httpStatus([500]))),
    ])
    func testStreamResponseWithRetryKeepsWebsocketPreEventRetries(settings: ModelRetrySettings) async throws {
        // Ports _with_unrelated_policy and _for_partial_all_policy.
        let recorder = Recorder()
        let websocketFlags = SharedBox<[Bool]>([])

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await collect(recorder) {
                streamRetry(recorder, settings: settings, previousResponseId: "resp_prev", rewindAllowed: false) { _ in
                    websocketFlags.mutate { $0.append(ModelRetryScope.websocketPreEventRetriesDisabled) }
                    return makeStream { _ in throw self.connectionError() }
                }
            }
        }

        #expect(recorder.calls.value == 1)
        // Swift: the flag is set for the whole attempt; the stream factory runs inside it.
        #expect(websocketFlags.value == [true])
    }
}

// MARK: - Streamed replay safety

extension ModelRetryTests {
    @Test func testStreamResponseWithRetryTreatsTimeoutErrorAsRetryable() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0.25, jitter: false), policy: retryPolicies.networkError())

        let events = try await collect(recorder) {
            streamRetry(recorder, settings: settings, failFirst { URLError(.timedOut) })
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.sleeps.value == [0.25])
        #expect(events == [Self.created])
    }

    @Test(arguments: ["connection", "http_500"])
    func testStreamResponseWithRetryAllowsStatefulRetryWhenProviderMarksSafe(failure: String) async throws {
        // Ports _allows_stateful_retry_when_provider_marks_safe and _for_http_failure_advice.
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(jitter: false), policy: retryPolicies.providerSuggested())
        let advice: @Sendable (ModelRetryAdviceRequest) -> ModelRetryAdvice?
        let error: @Sendable () -> any Error
        if failure == "connection" {
            advice = { _ in ModelRetryAdvice(suggested: true, replaySafety: "safe") }
            error = { OpenAIAPIError(message: "connection error", isConnectionError: true) }
        } else {
            advice = { getOpenAIRetryAdvice($0) }
            let httpError = statusErrorWithoutCode(500, "server_error")
            error = { httpError }
        }

        let events = try await collect(recorder) {
            streamRetry(recorder, settings: settings, advice: advice, previousResponseId: "resp_prev", failFirst(error))
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.sleeps.value == [0.25])
        #expect(events == [Self.created])
    }

    @Test func testStreamResponseWithRetryAllowsCustomPolicyToOverrideProviderVeto() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(jitter: false), policy: retryPolicies.httpStatus([429]))
        let advice = ModelRetryAdvice(suggested: false, reason: "server veto", normalized: ModelRetryNormalizedError(statusCode: 429))

        let events = try await collect(recorder) {
            streamRetry(recorder, settings: settings, advice: fixedAdvice(advice), failFirst { self.statusErrorWithoutCode(429, "rate_limit") })
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.sleeps.value == [0.25])
        #expect(events == [Self.created])
    }

    @Test func testStreamResponseWithRetryRejectsStatefulRetryWithoutReplaySafety() async throws {
        let recorder = Recorder()

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await collect(recorder) {
                streamRetry(
                    recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.providerSuggested()),
                    advice: fixedAdvice(ModelRetryAdvice(suggested: true)), previousResponseId: "resp_prev", rewindAllowed: false
                ) { _ in makeStream { _ in throw self.connectionError() } }
            }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testStreamResponseWithRetryStopsAfterRetryBudgetExhausted() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0.25, jitter: false), policy: retryPolicies.networkError())

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await collect(recorder) {
                streamRetry(recorder, settings: settings) { _ in makeStream { _ in throw self.connectionError() } }
            }
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.sleeps.value == [0.25])
    }

    @Test func testStreamResponseWithRetryRetriesAfterPreOutputEvent() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0.25, jitter: false), policy: retryPolicies.networkError())

        let events = try await collect(recorder) {
            streamRetry(recorder, settings: settings) { attempt in
                makeStream { continuation in
                    continuation.yield(Self.created)
                    if attempt == 1 { throw self.connectionError() }
                    continuation.yield(Self.inProgress)
                }
            }
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.sleeps.value == [0.25])
        #expect(events == [Self.created, Self.created, Self.inProgress])
    }

    @Test func testStreamResponseWithRetryDoesNotRetryAfterOutputEvent() async throws {
        let recorder = Recorder()

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await collect(recorder) {
                streamRetry(recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError()), rewindAllowed: false) { _ in
                    makeStream { continuation in
                        continuation.yield(["type": "response.output_item.added"])
                        throw self.connectionError()
                    }
                }
            }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testStreamResponseWithRetryPreservesConversationLockedCompatibility() async throws {
        let recorder = Recorder()
        let failedAttempts = SharedBox(0)

        let events = try await collect(recorder) {
            streamRetry(
                recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError()), previousResponseId: "resp_prev",
                failedAttempts: failedAttempts, failFirst { self.conversationLockedError() }
            )
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(failedAttempts.value == 1)
        #expect(recorder.sleeps.value == [1.0])
        #expect(events == [Self.created])
    }

    @Test func testStreamResponseWithRetryDisablesProviderRetriesOnStatefulCompatReplay() async throws {
        let recorder = Recorder()

        let events = try await collect(recorder) {
            streamRetry(recorder, settings: ModelRetrySettings(maxRetries: 1), previousResponseId: "resp_prev", failFirst { self.conversationLockedError() })
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.providerRetryFlags.value == [false, true])
        #expect(recorder.sleeps.value == [1.0])
        #expect(events == [Self.created])
    }

    @Test func testStreamResponseWithRetryClosesCurrentStreamWhenConsumerStopsEarly() async throws {
        let closed = AsyncEvent()
        // Swift: releasing the outer stream after an early exit is the equivalent of `aclose()`.
        do {
            let stream = streamRetry(Recorder(), settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError()), rewindAllowed: false) { _ in
                AsyncThrowingStream { continuation in
                    continuation.yield(Self.created)
                    continuation.yield(Self.inProgress)
                    continuation.onTermination = { _ in closed.set() }
                }
            }
            for try await event in stream {
                #expect(event == Self.created)
                break
            }
        }

        try await closed.wait(timeout: 5)
    }
}
