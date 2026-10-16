import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/models/test_model_retry.py.
//
// Swift: provider errors are `OpenAIAPIError` values, `asyncio.sleep` is replaced through the
// `RetrySleep.override` task-local, and settings validation runs on use (D12). Skipped upstream
// tests:
// - traceback/frame-local and `__cause__`/`__context__` hygiene checks (Python exception internals);
// - second-cancellation and blocked-cleanup timing tests (structured concurrency always awaits
//   child cleanup, so there is no abandoned cleanup to cancel again);
// - `test_retry_capabilities_preserve_falsey_policy` (Python truthiness and private attributes);
// - websocket transport tests (not ported).
@Suite struct ModelRetryTests {
    /// Records calls, rewinds, sleeps, and provider retry flags across concurrent closures.
    final class Recorder: @unchecked Sendable {
        let calls = SharedBox(0)
        let rewinds = SharedBox(0)
        let sleeps = SharedBox<[Double]>([])
        let providerRetryFlags = SharedBox<[Bool]>([])

        /// Increments the call count and returns the new value, recording the provider retry flag.
        func call() -> Int {
            providerRetryFlags.mutate { $0.append(ModelRetryScope.providerManagedRetriesDisabled) }
            return calls.mutate { value -> Int in
                value += 1
                return value
            }
        }

        func rewind() {
            rewinds.mutate { $0 += 1 }
        }

        func withRecordedSleeps<T>(_ body: () async throws -> T) async rethrows -> T {
            try await RetrySleep.$override.withValue({ [sleeps] delay in sleeps.mutate { $0.append(delay) } }, operation: body)
        }
    }

    func connectionError(_ message: String = "connection error") -> OpenAIAPIError {
        OpenAIAPIError(message: message, isConnectionError: true)
    }

    func conversationLockedError() -> OpenAIAPIError {
        OpenAIAPIError(statusCode: 400, errorCode: "conversation_locked", message: "locked")
    }

    func statusError(_ status: Int, code: String = "server_error") -> OpenAIAPIError {
        OpenAIAPIError(statusCode: status, errorCode: code, message: code)
    }

    /// An HTTP error whose code comes only from the response body.
    func statusErrorWithoutCode(_ status: Int, _ bodyCode: String = "server_error", headers: [String: String] = [:]) -> OpenAIAPIError {
        let url = URL(string: "https://example.com")!
        let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)
        let body: JSONObject = ["error": ["code": .string(bodyCode), "message": .string(bodyCode)]]
        return OpenAIClient.apiError(status: status, data: Data(body.jsonString().utf8), response: response)
    }

    /// Returns a retry-advice provider that always answers `advice`.
    func fixedAdvice(_ advice: ModelRetryAdvice?) -> @Sendable (ModelRetryAdviceRequest) -> ModelRetryAdvice? {
        { _ in advice }
    }

    func okResponse(_ responseId: String, usage: Usage = Usage(requests: 1)) -> ModelResponse {
        ModelResponse(output: [getTextMessage("ok")], usage: usage, responseId: responseId)
    }

    func retry(
        _ recorder: Recorder,
        settings: ModelRetrySettings?,
        advice: @escaping (ModelRetryAdviceRequest) -> ModelRetryAdvice? = { _ in nil },
        previousResponseId: String? = nil,
        conversationId: String? = nil,
        timeout: Double? = nil,
        replayUnsafeRequest: Bool = false,
        rewindAllowed: Bool = true,
        _ getResponse: @escaping @Sendable (Int) async throws -> ModelResponse
    ) async throws -> ModelResponse {
        try await recorder.withRecordedSleeps {
            try await getResponseWithRetry(
                getResponse: { try await getResponse(recorder.call()) },
                rewind: {
                    if !rewindAllowed { Issue.record("Retry should not rewind state") }
                    recorder.rewind()
                },
                retrySettings: settings,
                getRetryAdvice: advice,
                previousResponseId: previousResponseId,
                conversationId: conversationId,
                timeout: timeout,
                replayUnsafeRequest: replayUnsafeRequest
            )
        }
    }

    // MARK: - Settings validation

    @Test(arguments: [
        ModelRetryBackoffSettings(initialDelay: -0.1), ModelRetryBackoffSettings(maxDelay: -0.1), ModelRetryBackoffSettings(multiplier: -0.1),
    ])
    func testModelRetryBackoffSettingsRejectNegativeValues(backoff: ModelRetryBackoffSettings) {
        expectError(ValueError.self, containing: "greater than or equal to 0") { try backoff.validate() }
    }

    @Test func testModelRetrySettingsRejectsNegativeBackoffDict() {
        let settings = ModelSettings(retry: ModelRetrySettings(backoff: ModelRetryBackoffSettings(initialDelay: -0.1)))
        expectError(ValueError.self, containing: "greater than or equal to 0") { try settings.validate() }
    }

    @Test func testModelRetryBackoffSettingsAllowZeroValues() throws {
        let backoff = ModelRetryBackoffSettings(initialDelay: 0, maxDelay: 0, multiplier: 0)
        try backoff.validate()
        #expect(backoff.initialDelay == 0)
        #expect(backoff.maxDelay == 0)
        #expect(backoff.multiplier == 0)
    }

    @Test(arguments: [0, -0.1, .infinity, .nan] as [Double])
    func testModelSettingsRejectsInvalidModelCallTimeout(timeout: Double) {
        expectError(ValueError.self) { try ModelSettings(timeout: timeout).validate() }
    }

    @Test func testModelSettingsAcceptsFiniteModelCallTimeout() throws {
        let settings = ModelSettings(timeout: 1.25)
        try settings.validate()
        #expect(settings.timeout == 1.25)
        #expect(settings.toJSONDict()["timeout"] == 1.25)
    }

    // MARK: - Timeouts

    @Test func testGetResponseWithRetryTimesOutAndRetriesStatelessAttempt() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0), policy: retryPolicies.networkError())

        let result = try await retry(recorder, settings: settings, timeout: 0.01) { call in
            if call == 1 { try await waitForever() }
            return self.okResponse("resp")
        }

        #expect(recorder.calls.value == 2)
        #expect(result.responseId == "resp")
    }

    @Test func testGetResponseWithRetryBlocksUnsafeStatefulTimeoutReplay() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0), policy: retryPolicies.networkError())

        await expectErrorAsync(ModelTimeoutError.self, containing: "timed out after 0.01 seconds") {
            _ = try await retry(recorder, settings: settings, conversationId: "conv_123", timeout: 0.01) { _ in
                try await waitForever()
                return self.okResponse("unreachable")
            }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testModelTimeoutDiscardsCleanupErrorDetails() async throws {
        // Swift: ports the user-visible part of test_model_timeout_discards_cleanup_exception_graph.
        let sensitivePayload = "sensitive provider payload"

        let error = await expectErrorAsync(ModelTimeoutError.self) {
            _ = try await retry(Recorder(), settings: nil, timeout: 0.01) { _ in
                do {
                    try await waitForever()
                } catch {
                    throw TestError(sensitivePayload)
                }
                return self.okResponse("unreachable")
            }
        }

        #expect(error.map { !describeError($0).contains(sensitivePayload) } == true)
    }

    @Test(arguments: ["text", "tool_arguments"])
    func testStreamTimeoutDeliversPreviousEventAndClosesStream(payloadKind: String) async throws {
        // Ports the observable behavior of test_stream_timeout_discards_previous_event_from_retry_traceback.
        let sensitivePayload = "synthetic-sensitive-stream-payload"
        let event: TResponseStreamEvent = payloadKind == "text"
            ? [
                "type": "response.output_text.delta", "delta": .string(sensitivePayload), "content_index": 0, "output_index": 0,
                "item_id": "synthetic-item", "sequence_number": 0, "logprobs": [],
            ]
            : [
                "type": "response.function_call_arguments.delta", "delta": .string(sensitivePayload), "output_index": 0,
                "item_id": "synthetic-item", "sequence_number": 0,
            ]
        let closed = AsyncEvent()
        let step: ModelStep = .stream { _ in
            makeStream { continuation in
                defer { closed.set() }
                continuation.yield(event)
                try await waitForever()
            }
        }
        let model = try ScriptedModel(steps: [step])
        let agent = Agent(name: "test", model: .instance(model), modelSettings: ModelSettings(timeout: 0.05))
        let result = try Runner.runStreamed(agent, input: "synthetic-input", runConfig: RunConfig(traceIncludeSensitiveData: false))
        let delivered = SharedBox<[TResponseStreamEvent]>([])

        let error = await expectErrorAsync(ModelTimeoutError.self) {
            for try await streamEvent in result.streamEvents() {
                if case .rawResponse(let raw) = streamEvent {
                    delivered.mutate { $0.append(raw.data) }
                }
            }
        }

        #expect(delivered.value == [event])
        try await closed.wait(timeout: 5)
        #expect(error?.timeoutSeconds == 0.05)
    }

    // MARK: - Cancellation

    @Test func testGetResponseWithRetryPreservesParentCancellation() async throws {
        let started = AsyncEvent()
        let task = Task {
            try await retry(Recorder(), settings: nil, timeout: 10) { _ in
                started.set()
                try await waitForever()
                return self.okResponse("unreachable")
            }
        }
        try await started.wait()

        task.cancel()

        await expectErrorAsync(CancellationError.self) { _ = try await task.value }
    }

    @Test func testGetResponseWithRetryPreservesParentCancellationOverCleanupError() async throws {
        let started = AsyncEvent()
        let task = Task {
            try await retry(Recorder(), settings: nil, timeout: 10) { _ in
                started.set()
                do {
                    try await waitForever()
                } catch {
                    throw TestError("cleanup failed")
                }
                return self.okResponse("unreachable")
            }
        }
        try await started.wait()

        task.cancel()

        await expectErrorAsync(CancellationError.self) { _ = try await task.value }
    }

    // MARK: - Programmatic requests

    @Test func testProgrammaticRequestDisablesHiddenProviderRetries() async throws {
        let recorder = Recorder()

        let result = try await retry(recorder, settings: nil, replayUnsafeRequest: true) { _ in self.okResponse("resp") }

        #expect(result.responseId == "resp")
        #expect(recorder.providerRetryFlags.value == [true])
    }

    @Test func testProgrammaticRequestRequiresReplayApprovalForRunnerRetry() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0), policy: retryPolicies.networkError())

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(recorder, settings: settings, replayUnsafeRequest: true) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testGetOpenAIRetryAdviceReturnsNoneForNonRetryableStatus() {
        #expect(getOpenAIRetryAdvice(ModelRetryAdviceRequest(error: statusError(400, code: "invalid_request_error"), attempt: 1, stream: false)) == nil)
    }
}

// MARK: - Non-streamed retries

extension ModelRetryTests {
    @Test func testGetResponseWithRetryRetriesAndAugmentsUsage() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0.5, jitter: false), policy: retryPolicies.networkError())

        let result = try await retry(recorder, settings: settings) { call in
            if call == 1 { throw self.connectionError() }
            return self.okResponse("resp_123")
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.sleeps.value == [0.5])
        #expect(result.usage.requests == 2)
    }

    @Test func testGetResponseWithRetryKeepsProviderRetriesOnFirstAttempt() async throws {
        let recorder = Recorder()

        _ = try await retry(recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError())) { call in
            if call == 1 { throw self.connectionError() }
            return self.okResponse("resp_provider_retry_flag")
        }

        #expect(recorder.providerRetryFlags.value == [false, true])
    }

    @Test func testGetResponseWithRetryDisablesProviderRetriesOnFirstStatefulProviderHint() async throws {
        let recorder = Recorder()

        _ = try await retry(
            recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.providerSuggested()),
            advice: fixedAdvice(ModelRetryAdvice(suggested: true, replaySafety: "safe")), previousResponseId: "resp_prev"
        ) { call in
            if call == 1 { throw self.connectionError() }
            return self.okResponse("resp_stateful_provider_retry_flag")
        }

        #expect(recorder.providerRetryFlags.value == [true, true])
    }

    @Test(arguments: [
        ("http_status", ModelRetrySettings(maxRetries: 1, policy: retryPolicies.httpStatus([429])), true),
        ("omitted_budget", ModelRetrySettings(policy: retryPolicies.networkError()), false),
    ] as [(String, ModelRetrySettings, Bool)])
    func testGetResponseWithRetryStatefulConnectionFailureWithoutRetry(label: String, settings: ModelRetrySettings, providerRetriesDisabled: Bool) async throws {
        // Ports _disables_stateful_provider_retries_with_narrow_policy and
        // _keeps_stateful_provider_retries_when_budget_omitted.
        let recorder = Recorder()

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(recorder, settings: settings, previousResponseId: "resp_prev", rewindAllowed: false) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 1)
        #expect(recorder.providerRetryFlags.value == [providerRetriesDisabled])
    }

    @Test(arguments: [
        (ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError()), 500, "server_error"),
        (ModelRetrySettings(maxRetries: 1, policy: retryPolicies.any(retryPolicies.networkError(), retryPolicies.httpStatus([500]))), 429, "rate_limit_exceeded"),
    ] as [(ModelRetrySettings, Int, String)])
    func testGetResponseWithRetryDisablesStatefulProviderRetriesForPartialPolicies(settings: ModelRetrySettings, status: Int, code: String) async throws {
        // Ports _for_network_only_policy and _for_partial_policy.
        let recorder = Recorder()

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(recorder, settings: settings, previousResponseId: "resp_prev", rewindAllowed: false) { _ in
                throw self.statusError(status, code: code)
            }
        }

        #expect(recorder.calls.value == 1)
        #expect(recorder.providerRetryFlags.value == [true])
    }

    @Test func testGetResponseWithRetryDisablesProviderRetriesWhenExplicitlyDisabled() async throws {
        let recorder = Recorder()

        _ = try await retry(recorder, settings: ModelRetrySettings(maxRetries: 0, policy: retryPolicies.never())) { _ in
            self.okResponse("resp_provider_retry_preserved")
        }

        #expect(recorder.calls.value == 1)
        #expect(recorder.providerRetryFlags.value == [true])
    }

    @Test func testGetResponseWithRetryKeepsProviderRetriesWithoutRunnerPolicy() async throws {
        let recorder = Recorder()

        _ = try await retry(recorder, settings: ModelRetrySettings(maxRetries: 2)) { _ in self.okResponse("resp_provider_retry_without_policy") }

        #expect(recorder.calls.value == 1)
        #expect(recorder.providerRetryFlags.value == [false])
    }

    @Test(arguments: [Usage(requests: 1, inputTokens: 11, outputTokens: 7, totalTokens: 18), Usage(requests: 1)])
    func testGetResponseWithRetryPreservesSuccessfulRequestUsageEntry(usage: Usage) async throws {
        // Ports _preserves_successful_request_usage_entry and the zero-token variant.
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(jitter: false), policy: retryPolicies.networkError())

        let result = try await retry(recorder, settings: settings) { call in
            if call == 1 { throw self.connectionError() }
            return self.okResponse("resp_usage_entries", usage: usage)
        }

        #expect(result.usage.requests == 2)
        #expect(result.usage.requestUsageEntries.count == 2)
        #expect(result.usage.requestUsageEntries.first?.totalTokens == 0)
        #expect(result.usage.requestUsageEntries.last?.inputTokens == usage.inputTokens)
        #expect(result.usage.requestUsageEntries.last?.outputTokens == usage.outputTokens)
        #expect(result.usage.requestUsageEntries.last?.totalTokens == usage.totalTokens)
    }

    @Test func testGetResponseWithRetryPreservesInferredNormalizedErrorFlags() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(jitter: false), policy: retryPolicies.networkError())

        let result = try await retry(recorder, settings: settings, advice: fixedAdvice(ModelRetryAdvice(normalized: ModelRetryNormalizedError(statusCode: 429)))) { call in
            if call == 1 { throw self.connectionError() }
            return self.okResponse("resp_partial_normalized")
        }

        #expect(recorder.calls.value == 2)
        #expect(result.responseId == "resp_partial_normalized")
    }

    @Test func testGetResponseWithRetryHonorsExplicitFalseProviderNormalizedOverride() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(jitter: false), policy: retryPolicies.networkError())

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(
                recorder, settings: settings,
                advice: fixedAdvice(ModelRetryAdvice(normalized: ModelRetryNormalizedError(isNetworkError: false, isTimeout: false))),
                rewindAllowed: false
            ) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testGetResponseWithRetryHonorsExplicitNoneRetryAfterOverride() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(jitter: false), policy: retryPolicies.retryAfter())

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(
                recorder, settings: settings,
                advice: fixedAdvice(ModelRetryAdvice(normalized: ModelRetryNormalizedError(retryAfter: .some(nil)))),
                rewindAllowed: false
            ) { _ in throw self.statusErrorWithoutCode(429, "rate_limit", headers: ["retry-after-ms": "1250"]) }
        }

        #expect(recorder.calls.value == 1)
    }
}

// MARK: - Conversation-locked compatibility and replay safety

extension ModelRetryTests {
    @Test func testGetResponseWithRetryPreservesConversationLockedCompatibility() async throws {
        let recorder = Recorder()

        let result = try await retry(recorder, settings: nil) { call in
            if call == 1 { throw self.conversationLockedError() }
            return self.okResponse("resp_compat", usage: Usage(requests: 1, inputTokens: 3, outputTokens: 2, totalTokens: 5))
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.sleeps.value == [1.0])
        #expect(result.usage.requests == 2)
        #expect(result.usage.requestUsageEntries.map(\.totalTokens) == [0, 5])
    }

    @Test func testGetResponseWithRetryDisablesProviderRetriesOnStatefulCompatReplay() async throws {
        let recorder = Recorder()

        let result = try await retry(recorder, settings: nil, previousResponseId: "resp_prev") { call in
            if call == 1 { throw self.conversationLockedError() }
            return self.okResponse("resp_stateful_compat_disable_none")
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.providerRetryFlags.value == [false, true])
        #expect(recorder.sleeps.value == [1.0])
        #expect(result.responseId == "resp_stateful_compat_disable_none")
    }

    @Test func testGetResponseWithRetryRespectsExplicitDisableForConversationLocked() async throws {
        let recorder = Recorder()

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(recorder, settings: ModelRetrySettings(maxRetries: 0, policy: retryPolicies.never())) { _ in throw self.conversationLockedError() }
        }

        #expect(recorder.calls.value == 1)
        #expect(recorder.rewinds.value == 0)
        #expect(recorder.sleeps.value.isEmpty)
    }

    @Test func testGetResponseWithRetryKeepsConversationLockedCompatibilityWithRetry() async throws {
        let recorder = Recorder()

        let result = try await retry(recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError())) { call in
            if call == 1 { throw self.conversationLockedError() }
            return self.okResponse("resp_locked_retry_enabled")
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.sleeps.value == [1.0])
        #expect(result.responseId == "resp_locked_retry_enabled")
    }

    @Test(arguments: ["provider_suggested", "network_error"])
    func testGetResponseWithRetryAllowsStatefulRetryWhenProviderMarksSafe(policy: String) async throws {
        // Ports _allows_stateful_retry_when_provider_marks_safe and
        // _allows_provider_safe_stateful_retry_for_generic_policy.
        let recorder = Recorder()
        let retryPolicy = policy == "provider_suggested" ? retryPolicies.providerSuggested() : retryPolicies.networkError()

        let result = try await retry(
            recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicy),
            advice: fixedAdvice(ModelRetryAdvice(suggested: true, replaySafety: "safe")), previousResponseId: "resp_prev"
        ) { call in
            if call == 1 { throw self.connectionError() }
            return self.okResponse("resp_stateful")
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(result.usage.requests == 2)
    }

    @Test func testGetResponseWithRetryAllowsStatefulRetryForHTTPFailureAdvice() async throws {
        let recorder = Recorder()

        let result = try await retry(
            recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.providerSuggested()),
            advice: getOpenAIRetryAdvice, previousResponseId: "resp_prev"
        ) { call in
            if call == 1 { throw self.statusErrorWithoutCode(429, "rate_limit") }
            return self.okResponse("resp_stateful_http_failure")
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(result.responseId == "resp_stateful_http_failure")
        #expect(result.usage.requests == 2)
    }

    @Test func testGetResponseWithRetryRejectsStatefulRetryWithoutReplaySafety() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(jitter: false), policy: retryPolicies.providerSuggested())

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(
                recorder, settings: settings, advice: fixedAdvice(ModelRetryAdvice(suggested: true)), previousResponseId: "resp_prev", rewindAllowed: false
            ) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testGetResponseWithRetryExposesProviderErrorCodeToRetryPolicies() async throws {
        let recorder = Recorder()
        let policy = RetryPolicy(predicate: { $0.normalized.errorCode == "rate_limit" })
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(jitter: false), policy: policy)

        let result = try await retry(recorder, settings: settings, advice: getOpenAIRetryAdvice) { call in
            if call == 1 { throw self.statusErrorWithoutCode(429, "rate_limit") }
            return self.okResponse("resp_rate_limit_retry")
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(result.responseId == "resp_rate_limit_retry")
        #expect(result.usage.requests == 2)
    }

    @Test func testGetResponseWithRetryStopsAfterRetryBudgetExhausted() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 0.5, jitter: false), policy: retryPolicies.networkError())

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(recorder, settings: settings) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.sleeps.value == [0.5])
    }

    @Test func testGetResponseWithRetryCapsConversationLockedCompatibilityRetries() async throws {
        let recorder = Recorder()

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(recorder, settings: nil) { _ in throw self.conversationLockedError() }
        }

        #expect(recorder.calls.value == 4)
        #expect(recorder.rewinds.value == 3)
        #expect(recorder.sleeps.value == [1.0, 2.0, 4.0])
    }

    @Test func testGetResponseWithRetryPrefersRetryAfterOverBackoff() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, backoff: ModelRetryBackoffSettings(initialDelay: 5.0, jitter: false), policy: retryPolicies.networkError())

        let result = try await retry(recorder, settings: settings, advice: fixedAdvice(ModelRetryAdvice(suggested: true, retryAfter: 1.75))) { call in
            if call == 1 { throw self.connectionError() }
            return self.okResponse("resp_retry_after", usage: Usage(requests: 0))
        }

        #expect(recorder.rewinds.value == 1)
        #expect(recorder.sleeps.value == [1.75])
        #expect(result.usage.requests == 2)
    }

    @Test func testGetResponseWithRetryHonorsProviderHardVeto() async throws {
        let recorder = Recorder()
        let settings = ModelRetrySettings(maxRetries: 1, policy: retryPolicies.any(retryPolicies.providerSuggested(), retryPolicies.networkError()))

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(
                recorder, settings: settings, advice: fixedAdvice(ModelRetryAdvice(suggested: false, reason: "server veto")), rewindAllowed: false
            ) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testGetResponseWithRetryAllowsCustomPolicyToOverrideProviderVeto() async throws {
        let recorder = Recorder()
        let advice = ModelRetryAdvice(
            suggested: false, retryAfter: 1.75, reason: "server veto", normalized: ModelRetryNormalizedError(statusCode: 429, retryAfter: 1.75)
        )

        let result = try await retry(recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.retryAfter()), advice: fixedAdvice(advice)) { call in
            if call == 1 { throw self.statusErrorWithoutCode(429, "rate_limit") }
            return self.okResponse("resp_custom_policy_override")
        }

        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
        #expect(recorder.sleeps.value == [1.75])
        #expect(result.usage.requests == 2)
    }

    @Test func testRetryPoliciesAnyMergesLaterPositiveMetadata() async {
        let policy = retryPolicies.any(retryPolicies.networkError(), retryPolicies.retryAfter())
        let context = RetryPolicyContext(
            error: connectionError(), attempt: 1, maxRetries: 2, stream: false,
            normalized: ModelRetryNormalizedError(retryAfter: 1.75, isNetworkError: true), providerAdvice: ModelRetryAdvice(retryAfter: 1.75)
        )

        let decision = await policy(context)

        #expect(decision.retry)
        #expect(decision.delay == 1.75)
    }

    @Test func testGetResponseWithRetryHonorsUnsafeReplayVeto() async throws {
        let recorder = Recorder()

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(
                recorder, settings: ModelRetrySettings(maxRetries: 1, policy: retryPolicies.networkError()),
                advice: fixedAdvice(ModelRetryAdvice(suggested: true, replaySafety: "unsafe")), rewindAllowed: false
            ) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 1)
    }
}
