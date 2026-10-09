import Foundation
import Testing
@testable import Agents

// Continuation of tests/models/test_model_retry.py: replay-safety authority and unsafe-replay
// approvals.
//
// Swift: `RetryPolicyContext` and `ModelRetryAdvice` are value types, so a policy cannot mutate
// the provider's advice; the "cannot mutate provider authority" tests check that a policy's own
// decision cannot override the provider evidence.
extension ModelRetryTests {
    /// Advice matching the Responses websocket adapter after a response-started disconnect.
    private static func wsResponseStartedAdvice(_ request: ModelRetryAdviceRequest) -> ModelRetryAdvice? {
        ModelRetryAdvice(suggested: false, replaySafety: "unsafe", reason: describeError(request.error), responseStarted: true)
    }

    private var instantRetry: ModelRetryBackoffSettings { ModelRetryBackoffSettings(initialDelay: 0) }

    private func approvingPolicy(seen: SharedBox<[RetryPolicyContext]>? = nil, approveUnsafeReplay: Bool = true) -> RetryPolicy {
        RetryPolicy { context in
            seen?.mutate { $0.append(context) }
            return RetryDecision(retry: true, reason: "Approved one replay of a read-only model turn", approveUnsafeReplay: approveUnsafeReplay)
        }
    }

    @Test func testRetryPolicyContextNormalizesUnknownProviderReplaySafety() {
        let context = RetryPolicyContext(
            error: connectionError(), attempt: 1, maxRetries: 1, stream: false, normalized: ModelRetryNormalizedError(),
            providerAdvice: ModelRetryAdvice(replaySafety: "conditional")
        )
        #expect(context.replaySafety == "unknown")
    }

    @Test func testProviderSuggestedPreservesPublicVetoForUnsafeAdvice() async {
        let context = RetryPolicyContext(
            error: connectionError(), attempt: 1, maxRetries: 1, stream: false, normalized: ModelRetryNormalizedError(isNetworkError: true),
            providerAdvice: ModelRetryAdvice(suggested: false, replaySafety: "unsafe", reason: "provider veto")
        )

        let decision = await retryPolicies.providerSuggested()(context)

        #expect(!decision.retry)
        #expect(decision.reason == "provider veto")
    }

    @Test func testProviderUnsafeReplayRetriesWhenPolicyApprovesExplicitly() async throws {
        let recorder = Recorder()
        let seen = SharedBox<[RetryPolicyContext]>([])

        let result = try await retry(
            recorder, settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: approvingPolicy(seen: seen)),
            advice: Self.wsResponseStartedAdvice
        ) { call in
            if call == 1 { throw self.connectionError("no close frame received or sent") }
            return ModelResponse(output: [getTextMessage("ok")], usage: Usage(), responseId: "resp")
        }

        #expect(result.responseId == "resp")
        #expect(recorder.calls.value == 2)
        // The policy must be able to scope its approval, so it needs the structured context.
        #expect(seen.value.count == 1)
        #expect(seen.value.first?.stream == false)
        #expect(seen.value.first?.responseStarted == true)
        #expect(seen.value.first?.replaySafety == "unsafe")
        #expect(seen.value.first?.statefulRequest == false)
    }

    @Test func testProviderUnsafeReplayStillBlocksAnOrdinaryRetryDecision() async throws {
        let recorder = Recorder()
        // No `approveUnsafeReplay`, so this must not bypass replay protection.
        let policy = approvingPolicy(approveUnsafeReplay: false)

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(recorder, settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: policy), advice: Self.wsResponseStartedAdvice) { _ in
                throw self.connectionError("no close frame received or sent")
            }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testProviderUnsafeReplayApprovalIsIgnoredForStreamedRequests() async throws {
        let calls = SharedBox(0)
        let stream = streamResponseWithRetry(
            getStream: {
                calls.mutate { $0 += 1 }
                return makeStream { _ in throw OpenAIAPIError(message: "no close frame received or sent", isConnectionError: true) }
            },
            rewind: {},
            retrySettings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: approvingPolicy()),
            getRetryAdvice: { Self.wsResponseStartedAdvice($0) },
            previousResponseId: nil,
            conversationId: nil
        )

        await expectErrorAsync(OpenAIAPIError.self) {
            for try await _ in stream {}
        }

        #expect(calls.value == 1)
    }

    @Test func testUnsafeReplayApprovalDoesNotReplayProgrammaticToolCalling() async throws {
        let recorder = Recorder()
        let policyCalls = SharedBox(0)
        let policy = RetryPolicy { _ in
            policyCalls.mutate { $0 += 1 }
            return RetryDecision(retry: true, approveUnsafeReplay: true)
        }

        // `replayUnsafeRequest` is the request-level veto the runner sets for Programmatic Tool
        // Calling. Application approval must not lift it.
        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(
                recorder, settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: policy),
                advice: Self.wsResponseStartedAdvice, replayUnsafeRequest: true
            ) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 1)
        #expect(policyCalls.value == 0)
    }

    @Test(arguments: ["all", "any"], [false, true])
    func testUnsafeReplayApprovalResolvesProviderVetoInPolicyCombinators(combinator: String, providerFirst: Bool) async throws {
        let recorder = Recorder()
        let approving = RetryPolicy { _ in RetryDecision(retry: true, approveUnsafeReplay: true) }
        let providerPolicy = retryPolicies.providerSuggested()
        let policies = providerFirst ? [providerPolicy, approving] : [approving, providerPolicy]
        let combined = combinator == "all" ? retryPolicies.all(policies) : retryPolicies.any(policies)

        let result = try await retry(recorder, settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: combined), advice: Self.wsResponseStartedAdvice) { call in
            if call == 1 { throw self.connectionError("no close frame received or sent") }
            return ModelResponse(output: [getTextMessage("ok")], usage: Usage(), responseId: "resp")
        }

        #expect(result.responseId == "resp")
        #expect(recorder.calls.value == 2)
    }

    @Test func testProviderNormalizationCannotClearRawAbortVeto() async throws {
        let recorder = Recorder()
        let policyCalls = SharedBox(0)
        let policy = RetryPolicy { _ in
            policyCalls.mutate { $0 += 1 }
            return RetryDecision(retry: true, approveUnsafeReplay: true)
        }
        let advice = ModelRetryAdvice(suggested: false, replaySafety: "unsafe", normalized: ModelRetryNormalizedError(isAbort: false))

        // Swift: a cancelled URL request is the abort-like error that `URLSession` raises.
        await expectErrorAsync(URLError.self) {
            _ = try await retry(
                recorder, settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: policy), advice: fixedAdvice(advice), rewindAllowed: false
            ) { _ in throw URLError(.cancelled) }
        }

        #expect(recorder.calls.value == 1)
        #expect(policyCalls.value == 0)
    }

    @Test func testPolicyCannotPromoteUnknownProviderReplaySafety() async throws {
        let recorder = Recorder()
        let policyCalls = SharedBox(0)
        let policy = RetryPolicy { context in
            policyCalls.mutate { $0 += 1 }
            #expect(context.providerAdvice != nil)
            #expect(context.replaySafety == "unknown")
            return RetryDecision(retry: true)
        }

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(
                recorder, settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: policy),
                advice: fixedAdvice(ModelRetryAdvice(suggested: true, replaySafety: "conditional")), previousResponseId: "resp_1", rewindAllowed: false
            ) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 1)
        #expect(policyCalls.value == 1)
    }

    @Test func testPolicyCannotRevokeProviderSafeReplayEvidence() async throws {
        let recorder = Recorder()
        let policy = RetryPolicy { context in
            #expect(context.replaySafety == "safe")
            return RetryDecision(retry: true)
        }

        let result = try await retry(
            recorder, settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: policy),
            advice: fixedAdvice(ModelRetryAdvice(suggested: true, replaySafety: "safe")), previousResponseId: "resp_1"
        ) { call in
            if call == 1 { throw self.connectionError() }
            return ModelResponse(output: [getTextMessage("ok")], usage: Usage(), responseId: "resp")
        }

        #expect(result.responseId == "resp")
        #expect(recorder.calls.value == 2)
        #expect(recorder.rewinds.value == 1)
    }

    @Test(arguments: ["all", "any"], [(false, "unsafe", "unsafe"), (nil, "conditional", "unknown")] as [(Bool?, String, String)])
    func testComposedPolicyCannotOverrideProviderAuthority(combinator: String, advice: (Bool?, String, String)) async throws {
        let (suggested, initialSafety, expectedContextSafety) = advice
        let recorder = Recorder()
        let permissive = RetryPolicy { context in
            #expect(context.replaySafety == expectedContextSafety)
            #expect(!context.responseStarted)
            return RetryDecision(retry: true)
        }
        let policies = [permissive, retryPolicies.providerSuggested()]
        let policy = combinator == "all" ? retryPolicies.all(policies) : retryPolicies.any(policies)

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(
                recorder, settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: policy),
                advice: fixedAdvice(ModelRetryAdvice(suggested: suggested, replaySafety: initialSafety, responseStarted: false)),
                previousResponseId: "resp_1", rewindAllowed: false
            ) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testComposedPolicyCannotOverrideProviderAuthorityForPTC() async throws {
        let recorder = Recorder()
        let permissive = RetryPolicy { _ in RetryDecision(retry: true) }

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(
                recorder,
                settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: retryPolicies.all(permissive, retryPolicies.providerSuggested())),
                advice: fixedAdvice(ModelRetryAdvice(suggested: nil, replaySafety: "conditional")), replayUnsafeRequest: true, rewindAllowed: false
            ) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testUnsafeReplayApprovalAppliesToAStatefulRequest() async throws {
        // `autoPreviousResponseId` produces exactly this shape: a stateful request that fails closed
        // by default but carries no application-local side effects.
        let recorder = Recorder()

        let result = try await retry(
            recorder, settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: approvingPolicy()),
            advice: Self.wsResponseStartedAdvice, previousResponseId: "resp_1"
        ) { call in
            if call == 1 { throw self.connectionError("no close frame received or sent") }
            return ModelResponse(output: [getTextMessage("ok")], usage: Usage(), responseId: "resp")
        }

        #expect(result.responseId == "resp")
        #expect(recorder.calls.value == 2)
    }

    @Test func testStatefulRequestStillFailsClosedWithoutExplicitApproval() async throws {
        let recorder = Recorder()

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(
                recorder, settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: approvingPolicy(approveUnsafeReplay: false)),
                advice: Self.wsResponseStartedAdvice, conversationId: "conv_1"
            ) { _ in throw self.connectionError("no close frame received or sent") }
        }

        #expect(recorder.calls.value == 1)
    }

    @Test func testUnsafeReplayApprovalDoesNotLiftAStatefulRequestWithUnknownSafety() async throws {
        // No provider advice, so replay safety is unknown rather than unsafe. The approval is
        // scoped to provider-marked unsafe failures, so the stateful gate must stay closed.
        let recorder = Recorder()

        await expectErrorAsync(OpenAIAPIError.self) {
            _ = try await retry(
                recorder, settings: ModelRetrySettings(maxRetries: 1, backoff: instantRetry, policy: approvingPolicy()), previousResponseId: "resp_1"
            ) { _ in throw self.connectionError() }
        }

        #expect(recorder.calls.value == 1)
    }
}
