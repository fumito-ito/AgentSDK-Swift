import Foundation

/// Backoff configuration for runner-managed model retries.
public struct ModelRetryBackoffSettings: Sendable, Equatable, Hashable {
    /// Delay in seconds before the first retry attempt.
    public var initialDelay: Double?
    /// Maximum delay in seconds between retry attempts.
    public var maxDelay: Double?
    /// Multiplier applied after each retry attempt.
    public var multiplier: Double?
    /// Whether to apply random jitter to the computed delay.
    public var jitter: Bool?

    public init(initialDelay: Double? = nil, maxDelay: Double? = nil, multiplier: Double? = nil, jitter: Bool? = nil) {
        self.initialDelay = initialDelay
        self.maxDelay = maxDelay
        self.multiplier = multiplier
        self.jitter = jitter
    }

    public func toJSONDict() -> JSONObject {
        [
            "initial_delay": initialDelay.map(JSONValue.double) ?? .null,
            "max_delay": maxDelay.map(JSONValue.double) ?? .null,
            "multiplier": multiplier.map(JSONValue.double) ?? .null,
            "jitter": jitter.map(JSONValue.bool) ?? .null,
        ]
    }

    /// Overlays the non-`nil` values of `override`.
    func merged(with override: ModelRetryBackoffSettings?) -> ModelRetryBackoffSettings {
        guard let override else { return self }
        return ModelRetryBackoffSettings(
            initialDelay: override.initialDelay ?? initialDelay,
            maxDelay: override.maxDelay ?? maxDelay,
            multiplier: override.multiplier ?? multiplier,
            jitter: override.jitter ?? jitter
        )
    }
}

/// Normalized error facts exposed to retry policies.
public struct ModelRetryNormalizedError: Sendable, Equatable {
    public var statusCode: Int?
    public var errorCode: String?
    public var message: String?
    public var requestId: String?
    public var retryAfter: Double?
    public var isAbort: Bool
    public var isNetworkError: Bool
    public var isTimeout: Bool

    public init(
        statusCode: Int? = nil,
        errorCode: String? = nil,
        message: String? = nil,
        requestId: String? = nil,
        retryAfter: Double? = nil,
        isAbort: Bool = false,
        isNetworkError: Bool = false,
        isTimeout: Bool = false
    ) {
        self.statusCode = statusCode
        self.errorCode = errorCode
        self.message = message
        self.requestId = requestId
        self.retryAfter = retryAfter
        self.isAbort = isAbort
        self.isNetworkError = isNetworkError
        self.isTimeout = isTimeout
    }
}

/// Provider-specific retry guidance returned by model adapters.
public struct ModelRetryAdvice: Sendable, Equatable {
    public var suggested: Bool?
    public var retryAfter: Double?
    /// `"safe"`, `"unsafe"`, or `nil` when unknown.
    public var replaySafety: String?
    public var reason: String?
    public var normalized: ModelRetryNormalizedError?
    /// Whether the provider had begun emitting the response when the failure occurred.
    public var responseStarted: Bool

    public init(
        suggested: Bool? = nil,
        retryAfter: Double? = nil,
        replaySafety: String? = nil,
        reason: String? = nil,
        normalized: ModelRetryNormalizedError? = nil,
        responseStarted: Bool = false
    ) {
        self.suggested = suggested
        self.retryAfter = retryAfter
        self.replaySafety = replaySafety
        self.reason = reason
        self.normalized = normalized
        self.responseStarted = responseStarted
    }
}

/// Context passed to a model adapter when deriving retry advice.
public struct ModelRetryAdviceRequest: Sendable {
    public var error: any Error
    public var attempt: Int
    public var stream: Bool
    public var previousResponseId: String?
    public var conversationId: String?

    public init(error: any Error, attempt: Int, stream: Bool, previousResponseId: String? = nil, conversationId: String? = nil) {
        self.error = error
        self.attempt = attempt
        self.stream = stream
        self.previousResponseId = previousResponseId
        self.conversationId = conversationId
    }
}

/// Explicit retry decision returned by retry policies.
public struct RetryDecision: Sendable, Equatable {
    public var retry: Bool
    public var delay: Double?
    public var reason: String?
    /// Explicit application approval to replay a request the provider marked replay-unsafe.
    ///
    /// An ordinary `RetryDecision(retry: true)` never bypasses replay protection. Set this only for
    /// workloads where repeating provider-side work that may already have happened is acceptable.
    public var approveUnsafeReplay: Bool

    var hardVeto = false
    var delegableReplayVeto = false
    var approvesReplay = false

    public init(retry: Bool, delay: Double? = nil, reason: String? = nil, approveUnsafeReplay: Bool = false) {
        self.retry = retry
        self.delay = delay
        self.reason = reason
        self.approveUnsafeReplay = approveUnsafeReplay
    }

    public static func == (lhs: RetryDecision, rhs: RetryDecision) -> Bool {
        lhs.retry == rhs.retry && lhs.delay == rhs.delay && lhs.reason == rhs.reason
            && lhs.approveUnsafeReplay == rhs.approveUnsafeReplay
    }
}

/// Context passed to runtime retry policy callbacks.
public struct RetryPolicyContext: Sendable {
    public var error: any Error
    public var attempt: Int
    public var maxRetries: Int
    public var stream: Bool
    public var normalized: ModelRetryNormalizedError
    public var providerAdvice: ModelRetryAdvice?
    public var previousResponseId: String?
    public var conversationId: String?

    private let providerSuggested: Bool?
    private let providerReplaySafety: String
    private let providerResponseStarted: Bool

    public init(
        error: any Error,
        attempt: Int,
        maxRetries: Int,
        stream: Bool,
        normalized: ModelRetryNormalizedError,
        providerAdvice: ModelRetryAdvice? = nil,
        previousResponseId: String? = nil,
        conversationId: String? = nil
    ) {
        self.error = error
        self.attempt = attempt
        self.maxRetries = maxRetries
        self.stream = stream
        self.normalized = normalized
        self.providerAdvice = providerAdvice
        self.previousResponseId = previousResponseId
        self.conversationId = conversationId
        let replaySafety = providerAdvice?.replaySafety
        self.providerSuggested = providerAdvice?.suggested
        self.providerReplaySafety = (replaySafety == "safe" || replaySafety == "unsafe") ? replaySafety! : "unknown"
        self.providerResponseStarted = providerAdvice?.responseStarted ?? false
    }

    /// Whether the provider had begun emitting the response when the failure occurred.
    public var responseStarted: Bool { providerResponseStarted }

    /// Provider replay classification: `"safe"`, `"unsafe"` or `"unknown"`.
    public var replaySafety: String { providerReplaySafety }

    /// Whether the request carried `previous_response_id` or `conversation_id`.
    public var statefulRequest: Bool {
        !(previousResponseId ?? "").isEmpty || !(conversationId ?? "").isEmpty
    }

    var authoritySuggested: Bool? { providerSuggested }
}

/// A retry policy callback. Policies may return `Bool` or `RetryDecision`.
public struct RetryPolicy: Sendable {
    let evaluate: @Sendable (RetryPolicyContext) async -> RetryDecision
    let retriesSafeTransportErrors: Bool
    let retriesAllTransientErrors: Bool

    /// Creates a policy that returns an explicit decision.
    public init(_ evaluate: @escaping @Sendable (RetryPolicyContext) async -> RetryDecision) {
        self.init(evaluate: evaluate, retriesSafeTransportErrors: false, retriesAllTransientErrors: false)
    }

    /// Creates a policy that returns whether to retry.
    public init(predicate: @escaping @Sendable (RetryPolicyContext) async -> Bool) {
        self.init(
            evaluate: { RetryDecision(retry: await predicate($0)) },
            retriesSafeTransportErrors: false,
            retriesAllTransientErrors: false
        )
    }

    init(
        evaluate: @escaping @Sendable (RetryPolicyContext) async -> RetryDecision,
        retriesSafeTransportErrors: Bool,
        retriesAllTransientErrors: Bool
    ) {
        self.evaluate = evaluate
        self.retriesSafeTransportErrors = retriesSafeTransportErrors
        self.retriesAllTransientErrors = retriesAllTransientErrors
    }

    public func callAsFunction(_ context: RetryPolicyContext) async -> RetryDecision {
        await evaluate(context)
    }
}

func retryPolicyRetriesSafeTransportErrors(_ policy: RetryPolicy?) -> Bool {
    policy?.retriesSafeTransportErrors ?? false
}

func retryPolicyRetriesAllTransientErrors(_ policy: RetryPolicy?) -> Bool {
    policy?.retriesAllTransientErrors ?? false
}

/// Opt-in runner-managed retry settings for model calls.
public struct ModelRetrySettings: Sendable {
    /// Retries allowed after the initial model request.
    public var maxRetries: Int?
    /// Backoff settings applied when the policy retries without an explicit delay.
    public var backoff: ModelRetryBackoffSettings?
    /// Runtime-only retry policy callback. This field is not serialized.
    public var policy: RetryPolicy?

    public init(maxRetries: Int? = nil, backoff: ModelRetryBackoffSettings? = nil, policy: RetryPolicy? = nil) {
        self.maxRetries = maxRetries
        self.backoff = backoff
        self.policy = policy
    }

    public func toJSONDict() -> JSONObject {
        [
            "max_retries": maxRetries.map(JSONValue.int) ?? .null,
            "backoff": backoff.map { .object($0.toJSONDict()) } ?? .null,
        ]
    }

    /// Merges retry settings, keeping inherited values where the override is `nil`.
    static func merge(_ inherited: ModelRetrySettings?, _ override: ModelRetrySettings?) -> ModelRetrySettings? {
        guard let inherited else { return override }
        guard let override else { return inherited }
        let backoff: ModelRetryBackoffSettings?
        if let inheritedBackoff = inherited.backoff {
            backoff = inheritedBackoff.merged(with: override.backoff)
        } else {
            backoff = override.backoff
        }
        return ModelRetrySettings(
            maxRetries: override.maxRetries ?? inherited.maxRetries,
            backoff: backoff,
            policy: override.policy ?? inherited.policy
        )
    }
}

extension ModelRetrySettings: Equatable {
    public static func == (lhs: ModelRetrySettings, rhs: ModelRetrySettings) -> Bool {
        lhs.maxRetries == rhs.maxRetries && lhs.backoff == rhs.backoff && (lhs.policy == nil) == (rhs.policy == nil)
    }
}

private func withHardVeto(_ decision: RetryDecision) -> RetryDecision {
    var decision = decision
    decision.hardVeto = true
    return decision
}

private func withDelegableReplayVeto(_ decision: RetryDecision) -> RetryDecision {
    var decision = decision
    decision.hardVeto = true
    decision.delegableReplayVeto = true
    return decision
}

private func withReplaySafeApproval(_ decision: RetryDecision) -> RetryDecision {
    var decision = decision
    decision.approvesReplay = true
    return decision
}

private func mergePositiveRetryDecisions(_ existing: RetryDecision, _ incoming: RetryDecision) -> RetryDecision {
    var merged = RetryDecision(
        retry: true,
        delay: existing.delay,
        reason: existing.reason,
        approveUnsafeReplay: existing.approveUnsafeReplay || incoming.approveUnsafeReplay
    )
    if existing.approvesReplay { merged = withReplaySafeApproval(merged) }
    if let delay = incoming.delay { merged.delay = delay }
    if let reason = incoming.reason { merged.reason = reason }
    if incoming.approvesReplay { merged = withReplaySafeApproval(merged) }
    return merged
}

private func resolveDelegableReplayVeto(_ veto: RetryDecision, _ approving: RetryDecision) -> RetryDecision {
    guard approving.retry, approving.approveUnsafeReplay else { return veto }
    var resolved = RetryDecision(
        retry: true,
        delay: approving.delay,
        reason: approving.reason ?? veto.reason,
        approveUnsafeReplay: true
    )
    if approving.approvesReplay { resolved = withReplaySafeApproval(resolved) }
    return resolved
}

/// Built-in retry policies, mirroring upstream `retry_policies`.
public struct RetryPolicies: Sendable {
    public func never() -> RetryPolicy {
        RetryPolicy(evaluate: { _ in RetryDecision(retry: false) }, retriesSafeTransportErrors: false, retriesAllTransientErrors: false)
    }

    public func providerSuggested() -> RetryPolicy {
        RetryPolicy(
            evaluate: { context in
                let advice = context.providerAdvice
                let reason = advice?.reason
                guard let suggested = context.authoritySuggested else {
                    return RetryDecision(retry: false)
                }
                if !suggested {
                    if context.replaySafety == "unsafe" {
                        return withDelegableReplayVeto(RetryDecision(retry: false, reason: reason))
                    }
                    return withHardVeto(RetryDecision(retry: false, reason: reason))
                }
                let decision = RetryDecision(retry: true, delay: advice?.retryAfter, reason: reason)
                if context.replaySafety == "safe" {
                    return withReplaySafeApproval(decision)
                }
                return decision
            },
            retriesSafeTransportErrors: true,
            retriesAllTransientErrors: false
        )
    }

    public func networkError() -> RetryPolicy {
        RetryPolicy(
            evaluate: { RetryDecision(retry: $0.normalized.isNetworkError || $0.normalized.isTimeout) },
            retriesSafeTransportErrors: true,
            retriesAllTransientErrors: false
        )
    }

    public func retryAfter() -> RetryPolicy {
        RetryPolicy(
            evaluate: { context in
                guard let delay = context.normalized.retryAfter ?? context.providerAdvice?.retryAfter else {
                    return RetryDecision(retry: false)
                }
                return RetryDecision(retry: true, delay: delay)
            },
            retriesSafeTransportErrors: false,
            retriesAllTransientErrors: false
        )
    }

    public func httpStatus(_ statuses: [Int]) -> RetryPolicy {
        let allowed = Set(statuses)
        return RetryPolicy(
            evaluate: { context in
                guard let status = context.normalized.statusCode else { return RetryDecision(retry: false) }
                return RetryDecision(retry: allowed.contains(status))
            },
            retriesSafeTransportErrors: false,
            retriesAllTransientErrors: false
        )
    }

    public func all(_ policies: RetryPolicy...) -> RetryPolicy {
        all(policies)
    }

    public func all(_ policies: [RetryPolicy]) -> RetryPolicy {
        if policies.isEmpty { return never() }
        return RetryPolicy(
            evaluate: { context in
                var merged = RetryDecision(retry: true)
                var delegableVeto: RetryDecision?
                for predicate in policies {
                    let decision = await predicate(context)
                    if decision.hardVeto {
                        if decision.delegableReplayVeto {
                            if delegableVeto == nil { delegableVeto = decision }
                            continue
                        }
                        return decision
                    }
                    if !decision.retry { return decision }
                    if let delay = decision.delay { merged.delay = delay }
                    if let reason = decision.reason { merged.reason = reason }
                    if decision.approveUnsafeReplay { merged.approveUnsafeReplay = true }
                    if decision.approvesReplay { merged = withReplaySafeApproval(merged) }
                }
                if let delegableVeto {
                    return resolveDelegableReplayVeto(delegableVeto, merged)
                }
                return merged
            },
            retriesSafeTransportErrors: policies.allSatisfy(\.retriesSafeTransportErrors),
            retriesAllTransientErrors: policies.allSatisfy(\.retriesAllTransientErrors)
        )
    }

    public func any(_ policies: RetryPolicy...) -> RetryPolicy {
        any(policies)
    }

    public func any(_ policies: [RetryPolicy]) -> RetryPolicy {
        if policies.isEmpty { return never() }
        return RetryPolicy(
            evaluate: { context in
                var firstPositive: RetryDecision?
                var lastNegative: RetryDecision?
                var delegableVeto: RetryDecision?
                for predicate in policies {
                    let decision = await predicate(context)
                    if decision.hardVeto {
                        if decision.delegableReplayVeto {
                            if delegableVeto == nil { delegableVeto = decision }
                            continue
                        }
                        return decision
                    }
                    if decision.retry {
                        firstPositive = firstPositive.map { mergePositiveRetryDecisions($0, decision) } ?? decision
                        continue
                    }
                    lastNegative = decision
                }
                if let delegableVeto {
                    guard let firstPositive else { return delegableVeto }
                    return resolveDelegableReplayVeto(delegableVeto, firstPositive)
                }
                return firstPositive ?? lastNegative ?? RetryDecision(retry: false)
            },
            retriesSafeTransportErrors: policies.contains(where: \.retriesSafeTransportErrors),
            retriesAllTransientErrors: policies.contains(where: \.retriesAllTransientErrors)
        )
    }
}

/// Built-in retry policies (`retry_policies` upstream).
public let retryPolicies = RetryPolicies()
