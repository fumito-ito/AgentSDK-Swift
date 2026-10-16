import Foundation

/// An error that occurred during span execution.
public struct SpanError: Sendable, Equatable {
    /// A human-readable error description.
    public var message: String
    /// Optional additional error context.
    public var data: JSONObject?

    public init(message: String, data: JSONObject? = nil) {
        self.message = message
        self.data = data
    }

    public var json: JSONObject {
        ["message": .string(message), "data": data.map(JSONValue.object) ?? .null]
    }
}

/// Span-specific data.
public protocol SpanData: AnyObject, Sendable {
    /// The span type, e.g. `"agent"` or `"generation"`.
    var type: String { get }
    /// Exports the data for tracing backends.
    func export() -> JSONObject
    /// Span metadata merged into the exported span payload.
    var metadata: JSONObject? { get }
}

extension SpanData {
    public var metadata: JSONObject? { nil }
}

/// A traceable operation with timing and context.
///
/// Swift: Python marks spans as current with `contextvars`. Swift task-locals can only be bound
/// for a scope, so use `withCurrent` (or the `with...Span` helpers) to make a span current for a
/// block of code. See porting/decisions.md D8.
public protocol Span: AnyObject, Sendable {
    var traceId: String { get }
    var spanId: String { get }
    var spanData: any SpanData { get }
    var parentId: String? { get }
    var error: SpanError? { get }
    var startedAt: String? { get }
    var endedAt: String? { get }
    var tracingAPIKey: String? { get }
    var traceMetadata: JSONObject? { get }

    /// Starts the span.
    func start()
    /// Finishes the span.
    func finish()
    /// Records an error on the span.
    func setError(_ error: SpanError)
    /// Exports the span for tracing backends, or `nil` for no-op spans.
    func export() -> JSONObject?
}

extension Span {
    /// Runs `body` with this span as the current span.
    public func withCurrent<T>(_ body: () async throws -> T) async rethrows -> T {
        try await Scope.$currentSpan.withValue(self, operation: body)
    }

    /// Runs `body` with this span as the current span.
    public func withCurrent<T>(_ body: () throws -> T) rethrows -> T {
        try Scope.$currentSpan.withValue(self, operation: body)
    }

    /// Starts the span, runs `body` with it as current, and finishes it afterwards.
    public func run<T>(_ body: () async throws -> T) async rethrows -> T {
        start()
        defer { finish() }
        return try await withCurrent(body)
    }
}

/// Typed access to span data.
extension Span {
    public func data<T: SpanData>(as type: T.Type = T.self) -> T? {
        spanData as? T
    }
}

/// A no-op span that records nothing. Used when tracing is disabled.
public final class NoOpSpan: Span, @unchecked Sendable {
    public let spanData: any SpanData

    public init(spanData: any SpanData) {
        self.spanData = spanData
    }

    public var traceId: String { "no-op" }
    public var spanId: String { "no-op" }
    public var parentId: String? { nil }
    public var error: SpanError? { nil }
    public var startedAt: String? { nil }
    public var endedAt: String? { nil }
    public var tracingAPIKey: String? { nil }
    public var traceMetadata: JSONObject? { nil }

    public func start() {}
    public func finish() {}
    public func setError(_ error: SpanError) {}
    public func export() -> JSONObject? { nil }
}

private let spanMetadataRoutingKeys = ["agent_harness_id"]

/// A span that is recorded by the tracing library.
public final class SpanImpl: Span, @unchecked Sendable {
    public let traceId: String
    public let spanId: String
    public let parentId: String?
    public let spanData: any SpanData
    public let tracingAPIKey: String?
    public let traceMetadata: JSONObject?
    private let processor: any TracingProcessor
    private let lock = NSLock()
    private var startedAtValue: String?
    private var endedAtValue: String?
    private var errorValue: SpanError?

    public init(
        traceId: String,
        spanId: String?,
        parentId: String?,
        processor: any TracingProcessor,
        spanData: any SpanData,
        tracingAPIKey: String?,
        traceMetadata: JSONObject? = nil
    ) {
        self.traceId = traceId
        self.spanId = spanId ?? genSpanId()
        self.parentId = parentId
        self.processor = processor
        self.spanData = spanData
        self.tracingAPIKey = tracingAPIKey
        self.traceMetadata = traceMetadata
    }

    public var startedAt: String? { lock.lock(); defer { lock.unlock() }; return startedAtValue }
    public var endedAt: String? { lock.lock(); defer { lock.unlock() }; return endedAtValue }
    public var error: SpanError? { lock.lock(); defer { lock.unlock() }; return errorValue }

    public func start() {
        lock.lock()
        if startedAtValue != nil {
            lock.unlock()
            logger.warning("Span already started")
            return
        }
        startedAtValue = timeISO()
        lock.unlock()
        processor.onSpanStart(self)
    }

    public func finish() {
        lock.lock()
        if endedAtValue != nil {
            lock.unlock()
            logger.warning("Span already finished")
            return
        }
        endedAtValue = timeISO()
        lock.unlock()
        processor.onSpanEnd(self)
    }

    public func setError(_ error: SpanError) {
        lock.lock()
        errorValue = error
        lock.unlock()
    }

    public func export() -> JSONObject? {
        var payload: JSONObject = [
            "object": "trace.span",
            "id": .string(spanId),
            "trace_id": .string(traceId),
            "parent_id": parentId.map(JSONValue.string) ?? .null,
            "started_at": startedAt.map(JSONValue.string) ?? .null,
            "ended_at": endedAt.map(JSONValue.string) ?? .null,
            "span_data": .object(spanData.export()),
            "error": error.map { .object($0.json) } ?? .null,
        ]
        var metadata = JSONObject()
        if let traceMetadata {
            for key in spanMetadataRoutingKeys {
                if let value = traceMetadata[key] { metadata[key] = value }
            }
        }
        if let spanMetadata = spanData.metadata {
            for (key, value) in spanMetadata where metadata[key] == nil {
                metadata[key] = value
            }
        }
        if !metadata.isEmpty {
            payload["metadata"] = .object(metadata)
        }
        return payload
    }
}
