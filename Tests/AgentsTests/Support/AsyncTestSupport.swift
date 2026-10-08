import Foundation
@testable import Agents
import AgentsTesting

/// Equivalent of `asyncio.Event` for tests. Waiting is cancellable, as in asyncio.
final class AsyncEvent: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false

    var isSet: Bool { lock.locked { flag } }

    func set() {
        lock.locked { flag = true }
    }

    /// Waits until the event is set, throwing `CancellationError` when the task is cancelled.
    func wait() async throws {
        while !isSet {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 1_000_000)
        }
    }

    /// Waits until the event is set or `timeout` elapses (upstream `asyncio.wait_for`).
    func wait(timeout: Double) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !isSet {
            try Task.checkCancellation()
            if Date() >= deadline { throw TestTimeoutError() }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
    }
}

struct TestTimeoutError: Error, CustomStringConvertible {
    var description: String { "Timed out" }
}

/// Sleeps for `seconds` (upstream `asyncio.sleep`).
func sleep(seconds: Double) async throws {
    try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
}

/// Suspends until the task is cancelled (upstream `await asyncio.Event().wait()`).
func waitForever() async throws {
    while true {
        try await Task.sleep(nanoseconds: 10_000_000)
    }
}

/// Runs `operation`, failing with `TestTimeoutError` after `seconds`.
func withTestTimeout<T: Sendable>(_ seconds: Double, _ operation: @escaping @Sendable () async throws -> T) async throws -> T {
    switch try await withTimeout(seconds: seconds, operation) {
    case .completed(let value): return value
    case .timedOut: throw TestTimeoutError()
    }
}

/// A simple error with a message, standing in for Python's `RuntimeError` / `ValueError`.
struct TestError: Error, CustomStringConvertible, Equatable {
    let message: String
    init(_ message: String) { self.message = message }
    var description: String { message }
}

/// Wraps a `ScriptedModel` so tests can intercept calls (upstream `patch.object(model, ...)`).
final class HookedModel: Model, @unchecked Sendable {
    typealias GetResponseHook = @Sendable (ModelRequest, ScriptedModel) async throws -> ModelResponse
    typealias StreamHook = @Sendable (ModelRequest, ScriptedModel) -> AsyncThrowingStream<TResponseStreamEvent, any Error>

    let base: ScriptedModel
    var getResponseHook: GetResponseHook?
    var streamHook: StreamHook?
    var cleanupHook: (@Sendable () async -> Void)?

    init(base: ScriptedModel, getResponse: GetResponseHook? = nil, stream: StreamHook? = nil) {
        self.base = base
        self.getResponseHook = getResponse
        self.streamHook = stream
    }

    var calls: [ModelCall] { base.calls }

    func getResponse(_ request: ModelRequest) async throws -> ModelResponse {
        if let getResponseHook { return try await getResponseHook(request, base) }
        return try await base.getResponse(request)
    }

    func streamResponse(_ request: ModelRequest) -> AsyncThrowingStream<TResponseStreamEvent, any Error> {
        if let streamHook { return streamHook(request, base) }
        return base.streamResponse(request)
    }

    func getRetryAdvice(_ request: ModelRetryAdviceRequest) -> ModelRetryAdvice? {
        base.getRetryAdvice(request)
    }

    func cleanupOnRunEnd(owner: AnyObject) async {
        await cleanupHook?()
    }
}

/// Builds a stream that runs `body` with a continuation in a cancellable task.
func makeStream(
    _ body: @escaping @Sendable (AsyncThrowingStream<TResponseStreamEvent, any Error>.Continuation) async throws -> Void
) -> AsyncThrowingStream<TResponseStreamEvent, any Error> {
    AsyncThrowingStream { continuation in
        let task = Task {
            do {
                try await body(continuation)
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
    }
}
