import Foundation

// Port of src/agents/run_internal/model_retry.py.
//
// Swift: upstream inspects httpx/openai exception classes. Here provider errors expose their facts
// through `ModelAPIError`, and transport failures are recognised as `URLError`.

let defaultInitialDelaySeconds = 0.25
let defaultMaxDelaySeconds = 2.0
let defaultBackoffMultiplier = 2.0
let defaultBackoffJitter = true
let compatibilityConversationLockedRetries = 3
private let retrySafeStreamEventTypes: Set<String> = ["response.created", "response.in_progress"]

/// Facts a provider error exposes to the retry machinery.
public protocol ModelAPIError: Error {
    var statusCode: Int? { get }
    var errorCode: String? { get }
    var requestId: String? { get }
    /// The server-suggested retry delay in seconds, if any.
    var retryAfter: Double? { get }
    /// Whether the client failed before receiving a response (connection or timeout failure).
    var isConnectionError: Bool { get }
    var isTimeout: Bool { get }
}

extension ModelAPIError {
    public var isConnectionError: Bool { false }
    public var isTimeout: Bool { false }
}

/// Task-scoped switches that tell providers whether they may retry on their own.
public enum ModelRetryScope {
    /// When `true`, providers must not run hidden SDK-level retries for the current request.
    @TaskLocal public static var providerManagedRetriesDisabled = false
    /// When `true`, websocket transports must not retry before the first event.
    @TaskLocal public static var websocketPreEventRetriesDisabled = false
}

func isConversationLockedError(_ error: any Error) -> Bool {
    guard let apiError = error as? any ModelAPIError else { return false }
    return apiError.statusCode == 400 && apiError.errorCode == "conversation_locked"
}

func isAbortLikeError(_ error: any Error) -> Bool {
    if error is ModelTimeoutError { return false }
    if error is CancellationError { return true }
    if let urlError = error as? URLError, urlError.code == .cancelled { return true }
    return false
}

func isNetworkLikeError(_ error: any Error) -> Bool {
    if let apiError = error as? any ModelAPIError, apiError.isConnectionError || apiError.isTimeout {
        return true
    }
    if let urlError = error as? URLError {
        switch urlError.code {
        case .timedOut, .cannotConnectToHost, .networkConnectionLost, .notConnectedToInternet,
             .cannotFindHost, .dnsLookupFailed, .secureConnectionFailed:
            return true
        default:
            break
        }
    }
    let message = describeError(error).lowercased()
    return message.contains("connection error") || message.contains("network error")
        || message.contains("socket hang up") || message.contains("connection closed")
}

private func isTimeoutLikeError(_ error: any Error) -> Bool {
    if error is ModelTimeoutError { return true }
    if let apiError = error as? any ModelAPIError, apiError.isTimeout { return true }
    if let urlError = error as? URLError, urlError.code == .timedOut { return true }
    return false
}

func normalizeRetryError(_ error: any Error, _ providerAdvice: ModelRetryAdvice?) -> ModelRetryNormalizedError {
    let apiError = error as? any ModelAPIError
    var normalized = ModelRetryNormalizedError(
        statusCode: apiError?.statusCode,
        errorCode: apiError?.errorCode,
        message: describeError(error),
        requestId: apiError?.requestId,
        retryAfter: apiError?.retryAfter,
        isAbort: isAbortLikeError(error),
        isNetworkError: isNetworkLikeError(error),
        isTimeout: isTimeoutLikeError(error)
    )
    guard let providerAdvice else { return normalized }
    if let retryAfter = providerAdvice.retryAfter {
        normalized.retryAfter = retryAfter
    }
    if let override = providerAdvice.normalized {
        let explicit = override.explicitFields
        if explicit.contains("status_code") { normalized.statusCode = override.statusCode }
        if explicit.contains("error_code") { normalized.errorCode = override.errorCode }
        if explicit.contains("message") { normalized.message = override.message }
        if explicit.contains("request_id") { normalized.requestId = override.requestId }
        if explicit.contains("retry_after") { normalized.retryAfter = override.retryAfter }
        if explicit.contains("is_network_error") { normalized.isNetworkError = override.isNetworkError }
        if explicit.contains("is_timeout") { normalized.isTimeout = override.isTimeout }
        if explicit.contains("is_abort") {
            // Provider normalization may add abort evidence but cannot clear an inferred abort.
            normalized.isAbort = normalized.isAbort || override.isAbort
        }
    }
    return normalized
}

func defaultRetryDelay(_ attempt: Int, _ backoff: ModelRetryBackoffSettings?) -> Double {
    let initialDelay = backoff?.initialDelay ?? defaultInitialDelaySeconds
    let maxDelay = backoff?.maxDelay ?? defaultMaxDelaySeconds
    let multiplier = backoff?.multiplier ?? defaultBackoffMultiplier
    let useJitter = backoff?.jitter ?? defaultBackoffJitter
    let base = min(initialDelay * pow(multiplier, Double(max(attempt - 1, 0))), maxDelay)
    guard useJitter else { return base }
    return min(max(base * (0.875 + Double.random(in: 0..<1) * 0.25), 0), maxDelay)
}

/// Task-scoped replacement for the retry backoff sleep, used by tests in place of patching
/// `asyncio.sleep`.
enum RetrySleep {
    @TaskLocal static var override: (@Sendable (Double) async throws -> Void)?
}

private func sleepForRetry(_ delay: Double) async throws {
    if let override = RetrySleep.override {
        try await override(delay)
        return
    }
    guard delay > 0 else { return }
    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
}

/// Awaits one model operation, turning only this deadline into a `ModelTimeoutError`.
func awaitModelAttempt<Value: Sendable>(
    timeout: Double?,
    timeoutErrorSeconds: Double? = nil,
    _ operation: @escaping @Sendable () async throws -> Value
) async throws -> Value {
    guard let timeout else { return try await operation() }
    switch try await withTimeout(seconds: timeout, operation) {
    case .completed(let value):
        return value
    case .timedOut:
        throw ModelTimeoutError(timeoutSeconds: timeoutErrorSeconds ?? timeout)
    }
}

private func zeroRequestUsageEntry() -> RequestUsage {
    RequestUsage(inputTokens: 0, outputTokens: 0, totalTokens: 0, inputTokensDetails: InputTokensDetails(), outputTokensDetails: OutputTokensDetails())
}

private func requestUsageEntry(from usage: Usage) -> RequestUsage {
    RequestUsage(
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        totalTokens: usage.totalTokens,
        inputTokensDetails: usage.inputTokensDetails,
        outputTokensDetails: usage.outputTokensDetails
    )
}

/// Counts failed retry attempts as zero-token requests on the successful response usage.
func applyRetryAttemptUsage(_ usage: Usage, failedAttempts: Int) -> Usage {
    guard failedAttempts > 0 else { return usage }
    var usage = usage
    var successful = usage.requestUsageEntries
    if successful.isEmpty {
        successful.append(requestUsageEntry(from: usage))
    }
    usage.requests = max(usage.requests, 1) + failedAttempts
    usage.requestUsageEntries = Array(repeating: zeroRequestUsageEntry(), count: failedAttempts) + successful
    return usage
}

private func streamEventBlocksRetry(_ event: TResponseStreamEvent) -> Bool {
    !retrySafeStreamEventTypes.contains(event.type ?? "")
}

func evaluateRetry(
    error: any Error,
    attempt: Int,
    maxRetries: Int,
    retryPolicy: RetryPolicy?,
    retryBackoff: ModelRetryBackoffSettings?,
    stream: Bool,
    replayUnsafeRequest: Bool,
    emittedRetryUnsafeEvent: Bool,
    providerAdvice: ModelRetryAdvice?,
    previousResponseId: String? = nil,
    conversationId: String? = nil
) async -> RetryDecision {
    if attempt > maxRetries { return RetryDecision(retry: false) }
    let normalized = normalizeRetryError(error, providerAdvice)
    let context = RetryPolicyContext(
        error: error, attempt: attempt, maxRetries: maxRetries, stream: stream, normalized: normalized,
        providerAdvice: providerAdvice, previousResponseId: previousResponseId, conversationId: conversationId
    )
    let providerMarksUnsafe = context.replaySafety == "unsafe"
    let providerMarksSafe = context.replaySafety == "safe"
    // Aborts, and failures that already emitted user-visible streamed output, are absolute vetoes.
    if normalized.isAbort || emittedRetryUnsafeEvent {
        return RetryDecision(retry: false, reason: providerAdvice?.reason)
    }
    if providerMarksUnsafe && (stream || replayUnsafeRequest) {
        return RetryDecision(retry: false, reason: providerAdvice?.reason)
    }
    guard let retryPolicy else { return RetryDecision(retry: false) }
    let decision = await retryPolicy(context)
    guard decision.retry else { return decision }

    let statefulRequest = !(previousResponseId ?? "").isEmpty || !(conversationId ?? "").isEmpty
    let fallbackReason = decision.reason ?? providerAdvice?.reason
    // 1. A request-level replay veto covers application-local side effects.
    if replayUnsafeRequest && !decision.approvesReplay && !providerMarksSafe {
        return RetryDecision(retry: false, reason: fallbackReason)
    }
    // 2. Stateful requests fail closed unless replay is explicitly approved.
    if statefulRequest && !(decision.approvesReplay || providerMarksSafe || (decision.approveUnsafeReplay && providerMarksUnsafe)) {
        return RetryDecision(retry: false, reason: fallbackReason)
    }
    // 3. Provider-marked replay unsafety requires explicit approval.
    if providerMarksUnsafe && !(decision.approvesReplay || decision.approveUnsafeReplay) {
        return RetryDecision(retry: false, reason: fallbackReason)
    }
    return RetryDecision(
        retry: true,
        delay: decision.delay ?? normalized.retryAfter ?? defaultRetryDelay(attempt, retryBackoff),
        reason: fallbackReason
    )
}

private func shouldPreserveConversationLockedCompatibility(_ settings: ModelRetrySettings?) -> Bool {
    guard let settings else { return true }
    guard let maxRetries = settings.maxRetries else { return true }
    return maxRetries > 0
}

func shouldDisableProviderManagedRetries(
    _ settings: ModelRetrySettings?,
    attempt: Int,
    statefulRequest: Bool,
    replayUnsafeRequest: Bool
) -> Bool {
    if replayUnsafeRequest { return true }
    if let maxRetries = settings?.maxRetries, maxRetries <= 0 { return true }
    if attempt > 1 {
        if statefulRequest { return true }
        guard let settings, settings.policy != nil else { return false }
        return max(settings.maxRetries ?? 0, 0) > 0
    }
    guard let settings, statefulRequest else { return false }
    guard let maxRetries = settings.maxRetries else { return false }
    return maxRetries > 0 && settings.policy != nil
}

func shouldDisableWebsocketPreEventRetry(_ settings: ModelRetrySettings?) -> Bool {
    guard let settings else { return false }
    if let maxRetries = settings.maxRetries, maxRetries <= 0 { return true }
    return settings.policy != nil && max(settings.maxRetries ?? 0, 0) > 0
}

private struct RetryLoopState {
    var requestAttempt = 1
    var policyAttempt = 1
    var failedPolicyAttempts = 0
    var compatibilityRetriesTaken = 0

    var failedAttempts: Int { failedPolicyAttempts + compatibilityRetriesTaken }
}

/// Decides whether to retry after a failure; returns the delay to wait, or `nil` to rethrow.
private func nextRetryDelay(
    error: any Error,
    state: inout RetryLoopState,
    settings: ModelRetrySettings?,
    getRetryAdvice: (ModelRetryAdviceRequest) -> ModelRetryAdvice?,
    stream: Bool,
    replayUnsafeRequest: Bool,
    emittedRetryUnsafeEvent: Bool,
    previousResponseId: String?,
    conversationId: String?
) async -> Double? {
    if !replayUnsafeRequest, isConversationLockedError(error), shouldPreserveConversationLockedCompatibility(settings),
       state.compatibilityRetriesTaken < compatibilityConversationLockedRetries {
        state.compatibilityRetriesTaken += 1
        let delay = pow(2.0, Double(state.compatibilityRetriesTaken - 1))
        logger.debug("Conversation locked, retrying in \(delay)s (attempt \(state.compatibilityRetriesTaken)/\(compatibilityConversationLockedRetries)).")
        state.requestAttempt += 1
        return delay
    }
    let advice = getRetryAdvice(ModelRetryAdviceRequest(
        error: error, attempt: state.policyAttempt, stream: stream,
        previousResponseId: previousResponseId, conversationId: conversationId
    ))
    let decision = await evaluateRetry(
        error: error,
        attempt: state.policyAttempt,
        maxRetries: max(settings?.maxRetries ?? 0, 0),
        retryPolicy: settings?.policy,
        retryBackoff: settings?.backoff,
        stream: stream,
        replayUnsafeRequest: replayUnsafeRequest,
        emittedRetryUnsafeEvent: emittedRetryUnsafeEvent,
        providerAdvice: advice,
        previousResponseId: previousResponseId,
        conversationId: conversationId
    )
    guard decision.retry else { return nil }
    logger.debug("Retrying failed model request in \(decision.delay ?? 0)s (attempt \(state.policyAttempt)/\(settings?.maxRetries ?? 0)).")
    state.requestAttempt += 1
    state.policyAttempt += 1
    state.failedPolicyAttempts += 1
    return decision.delay ?? 0
}

/// Calls the model, retrying according to `retrySettings` and provider advice.
func getResponseWithRetry(
    getResponse: @escaping @Sendable () async throws -> ModelResponse,
    rewind: () async throws -> Void,
    retrySettings: ModelRetrySettings?,
    getRetryAdvice: (ModelRetryAdviceRequest) -> ModelRetryAdvice?,
    previousResponseId: String?,
    conversationId: String?,
    timeout: Double? = nil,
    replayUnsafeRequest: Bool = false
) async throws -> ModelResponse {
    var state = RetryLoopState()
    let disableWebsocketPreEventRetry = replayUnsafeRequest || shouldDisableWebsocketPreEventRetry(retrySettings)
    let statefulRequest = !(previousResponseId ?? "").isEmpty || !(conversationId ?? "").isEmpty
    while true {
        do {
            let disableProviderRetries = shouldDisableProviderManagedRetries(
                retrySettings, attempt: state.requestAttempt, statefulRequest: statefulRequest, replayUnsafeRequest: replayUnsafeRequest
            )
            var response = try await ModelRetryScope.$providerManagedRetriesDisabled.withValue(disableProviderRetries) {
                try await ModelRetryScope.$websocketPreEventRetriesDisabled.withValue(disableWebsocketPreEventRetry) {
                    try await awaitModelAttempt(timeout: timeout, getResponse)
                }
            }
            response.usage = applyRetryAttemptUsage(response.usage, failedAttempts: state.failedAttempts)
            return response
        } catch {
            if error is CancellationError { throw error }
            guard let delay = await nextRetryDelay(
                error: error, state: &state, settings: retrySettings, getRetryAdvice: getRetryAdvice, stream: false,
                replayUnsafeRequest: replayUnsafeRequest, emittedRetryUnsafeEvent: false,
                previousResponseId: previousResponseId, conversationId: conversationId
            ) else {
                throw error
            }
            try await rewind()
            try await sleepForRetry(delay)
        }
    }
}

/// Streams the model response, retrying failures that happen before retry-unsafe events.
///
/// `onFailedAttempts` receives the number of failed attempts before each yielded event so callers
/// can account for retry usage.
func streamResponseWithRetry(
    getStream: @escaping @Sendable () -> AsyncThrowingStream<TResponseStreamEvent, any Error>,
    rewind: @escaping @Sendable () async throws -> Void,
    retrySettings: ModelRetrySettings?,
    getRetryAdvice: @escaping @Sendable (ModelRetryAdviceRequest) -> ModelRetryAdvice?,
    previousResponseId: String?,
    conversationId: String?,
    timeout: Double? = nil,
    replayUnsafeRequest: Bool = false,
    onFailedAttempts: (@Sendable (Int) -> Void)? = nil
) -> AsyncThrowingStream<TResponseStreamEvent, any Error> {
    AsyncThrowingStream { continuation in
        let task = Task {
            var state = RetryLoopState()
            let disableWebsocketPreEventRetry = replayUnsafeRequest || shouldDisableWebsocketPreEventRetry(retrySettings)
            let statefulRequest = !(previousResponseId ?? "").isEmpty || !(conversationId ?? "").isEmpty
            while true {
                var emittedRetryUnsafeEvent = false
                let deadline = timeout.map { Date().addingTimeInterval($0) }
                do {
                    let disableProviderRetries = shouldDisableProviderManagedRetries(
                        retrySettings, attempt: state.requestAttempt, statefulRequest: statefulRequest, replayUnsafeRequest: replayUnsafeRequest
                    )
                    let stream = ModelRetryScope.$providerManagedRetriesDisabled.withValue(disableProviderRetries) {
                        ModelRetryScope.$websocketPreEventRetriesDisabled.withValue(disableWebsocketPreEventRetry) {
                            getStream()
                        }
                    }
                    let iterator = SendableIterator(stream.makeAsyncIterator())
                    while true {
                        let next: TResponseStreamEvent?
                        if let deadline, let timeout {
                            let remaining = max(deadline.timeIntervalSinceNow, 0)
                            next = try await awaitModelAttempt(timeout: remaining, timeoutErrorSeconds: timeout) {
                                try await iterator.next()
                            }
                        } else {
                            next = try await iterator.next()
                        }
                        guard let event = next else {
                            continuation.finish()
                            return
                        }
                        if streamEventBlocksRetry(event) { emittedRetryUnsafeEvent = true }
                        onFailedAttempts?(state.failedAttempts)
                        continuation.yield(event)
                    }
                } catch {
                    if error is CancellationError || Task.isCancelled {
                        continuation.finish(throwing: error)
                        return
                    }
                    guard let delay = await nextRetryDelay(
                        error: error, state: &state, settings: retrySettings, getRetryAdvice: getRetryAdvice, stream: true,
                        replayUnsafeRequest: replayUnsafeRequest, emittedRetryUnsafeEvent: emittedRetryUnsafeEvent,
                        previousResponseId: previousResponseId, conversationId: conversationId
                    ) else {
                        continuation.finish(throwing: error)
                        return
                    }
                    do {
                        try await rewind()
                        try await sleepForRetry(delay)
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
            }
        }
        continuation.onTermination = { _ in task.cancel() }
    }
}

/// Wraps a stream iterator so it can be advanced from a child task.
final class SendableIterator<Element: Sendable>: @unchecked Sendable {
    private var iterator: AsyncThrowingStream<Element, any Error>.AsyncIterator

    init(_ iterator: AsyncThrowingStream<Element, any Error>.AsyncIterator) {
        self.iterator = iterator
    }

    func next() async throws -> Element? {
        try await iterator.next()
    }
}
