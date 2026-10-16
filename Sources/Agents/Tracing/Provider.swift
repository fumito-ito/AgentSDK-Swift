import Foundation

/// Holds the current trace and span for the running task (upstream `Scope` with `contextvars`).
public enum Scope {
    @TaskLocal public static var currentSpan: (any Span)?
    @TaskLocal public static var currentTrace: (any Trace)?

    public static func getCurrentSpan() -> (any Span)? { currentSpan }
    public static func getCurrentTrace() -> (any Trace)? { currentTrace }
}

/// Processes and monitors traces and spans.
///
/// Processors receive notifications when traces and spans start and end. All methods should be
/// thread-safe, return quickly, and handle their own errors.
public protocol TracingProcessor: AnyObject, Sendable {
    func onTraceStart(_ trace: any Trace)
    func onTraceEnd(_ trace: any Trace)
    func onSpanStart(_ span: any Span)
    func onSpanEnd(_ span: any Span)
    /// Called when the application stops to clean up resources.
    func shutdown(timeout: Double?)
    /// Forces immediate processing of any queued traces/spans.
    func forceFlush()
}

extension TracingProcessor {
    public func shutdown() { shutdown(timeout: nil) }
}

/// Exports traces and spans, for example to a log or a backend.
public protocol TracingExporter: AnyObject, Sendable {
    func export(_ items: [TraceOrSpan])
}

/// An exported trace or span.
public enum TraceOrSpan: Sendable {
    case trace(any Trace)
    case span(any Span)

    public var tracingAPIKey: String? {
        switch self {
        case .trace(let trace): return trace.tracingAPIKey
        case .span(let span): return span.tracingAPIKey
        }
    }

    public func export() -> JSONObject? {
        switch self {
        case .trace(let trace): return trace.export()
        case .span(let span): return span.export()
        }
    }
}

/// Forwards all calls to a list of processors, in order of registration.
public final class SynchronousMultiTracingProcessor: TracingProcessor, @unchecked Sendable {
    private let lock = NSLock()
    private var processors: [any TracingProcessor] = []

    public init() {}

    var currentProcessors: [any TracingProcessor] {
        lock.lock()
        defer { lock.unlock() }
        return processors
    }

    /// Adds a processor. Each processor receives all traces and spans.
    public func addTracingProcessor(_ processor: any TracingProcessor) {
        lock.lock()
        processors.append(processor)
        lock.unlock()
    }

    /// Replaces the list of processors.
    public func setProcessors(_ processors: [any TracingProcessor]) {
        lock.lock()
        self.processors = processors
        lock.unlock()
    }

    public func onTraceStart(_ trace: any Trace) { currentProcessors.forEach { $0.onTraceStart(trace) } }
    public func onTraceEnd(_ trace: any Trace) { currentProcessors.forEach { $0.onTraceEnd(trace) } }
    public func onSpanStart(_ span: any Span) { currentProcessors.forEach { $0.onSpanStart(span) } }
    public func onSpanEnd(_ span: any Span) { currentProcessors.forEach { $0.onSpanEnd(span) } }

    public func shutdown(timeout: Double?) {
        let deadline = timeout.map { Date().addingTimeInterval($0) }
        for processor in currentProcessors {
            logger.debug("Shutting down trace processor")
            let remaining = deadline.map { max(0, $0.timeIntervalSinceNow) }
            if let remaining, remaining <= 0 {
                logger.warning("[non-fatal] Tracing: shutdown timeout reached before processor cleanup.")
                return
            }
            processor.shutdown(timeout: remaining)
        }
    }

    public func forceFlush() { currentProcessors.forEach { $0.forceFlush() } }
}

/// Configuration for tracing behavior and export.
public struct TracingConfig: Sendable, Equatable {
    /// Optional API key used to export traces.
    public var apiKey: String?
    /// Whether the runner creates task and turn spans. Defaults to `true` when omitted.
    public var includeTaskAndTurnSpans: Bool?

    public init(apiKey: String? = nil, includeTaskAndTurnSpans: Bool? = nil) {
        self.apiKey = apiKey
        self.includeTaskAndTurnSpans = includeTaskAndTurnSpans
    }
}

/// Resolves whether automatic runner task and turn spans are enabled.
func includeTaskAndTurnSpans(_ config: TracingConfig?) -> Bool {
    config?.includeTaskAndTurnSpans ?? true
}

/// The parent of a new span.
public enum SpanParent: Sendable {
    case trace(any Trace)
    case span(any Span)
}

/// Creates traces and spans.
public protocol TraceProvider: AnyObject, Sendable {
    func registerProcessor(_ processor: any TracingProcessor)
    func setProcessors(_ processors: [any TracingProcessor])
    func getCurrentTrace() -> (any Trace)?
    func getCurrentSpan() -> (any Span)?
    func setDisabled(_ disabled: Bool)
    func timeISO() -> String
    func genTraceId() -> String
    func genSpanId() -> String
    func genGroupId() -> String
    func createTrace(name: String, traceId: String?, groupId: String?, metadata: JSONObject?, disabled: Bool, tracing: TracingConfig?) -> any Trace
    func createSpan(spanData: any SpanData, spanId: String?, parent: SpanParent?, disabled: Bool) -> any Span
    func forceFlush()
    func shutdown(timeout: Double?)
}

/// The default trace provider.
public final class DefaultTraceProvider: TraceProvider, @unchecked Sendable {
    let multiProcessor = SynchronousMultiTracingProcessor()
    private let lock = NSLock()
    private var envDisabled: Bool?
    private var manualDisabled: Bool?

    public init() {}

    public func registerProcessor(_ processor: any TracingProcessor) {
        multiProcessor.addTracingProcessor(processor)
    }

    public func setProcessors(_ processors: [any TracingProcessor]) {
        multiProcessor.setProcessors(processors)
    }

    public func getCurrentTrace() -> (any Trace)? { Scope.currentTrace }
    public func getCurrentSpan() -> (any Span)? { Scope.currentSpan }

    public func setDisabled(_ disabled: Bool) {
        lock.lock()
        manualDisabled = disabled
        lock.unlock()
    }

    var isDisabled: Bool {
        lock.lock()
        defer { lock.unlock() }
        if envDisabled == nil {
            let value = ProcessInfo.processInfo.environment["OPENAI_AGENTS_DISABLE_TRACING"]?.lowercased() ?? "false"
            envDisabled = value == "true" || value == "1"
        }
        return manualDisabled ?? envDisabled ?? false
    }

    public func timeISO() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: Date())
    }

    private func hex(_ count: Int) -> String {
        String(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(count))
    }

    public func genTraceId() -> String { "trace_\(hex(32))" }
    public func genSpanId() -> String { "span_\(hex(24))" }
    public func genGroupId() -> String { "group_\(hex(24))" }

    public func createTrace(name: String, traceId: String?, groupId: String?, metadata: JSONObject?, disabled: Bool, tracing: TracingConfig?) -> any Trace {
        if isDisabled || disabled {
            logger.debug("Tracing is disabled. Not creating trace")
            return NoOpTrace()
        }
        let resolvedId = traceId ?? genTraceId()
        logger.debug("Creating trace with id \(resolvedId)")
        return TraceImpl(
            name: name,
            traceId: resolvedId,
            groupId: groupId,
            metadata: metadata,
            processor: multiProcessor,
            tracingAPIKey: tracing?.apiKey
        )
    }

    public func createSpan(spanData: any SpanData, spanId: String?, parent: SpanParent?, disabled: Bool) -> any Span {
        if isDisabled || disabled {
            logger.debug("Tracing is disabled. Not creating span")
            return NoOpSpan(spanData: spanData)
        }
        if spanId == "no-op" {
            return NoOpSpan(spanData: spanData)
        }
        let traceId: String
        let parentId: String?
        let tracingAPIKey: String?
        let traceMetadata: JSONObject?
        switch parent {
        case nil:
            let currentSpan = Scope.currentSpan
            guard let currentTrace = Scope.currentTrace else {
                logger.debug("No active trace. Make sure to start a trace with `trace()` first Returning NoOpSpan.")
                return NoOpSpan(spanData: spanData)
            }
            if isNoOp(currentTrace) || currentSpan.map(isNoOp) == true {
                return NoOpSpan(spanData: spanData)
            }
            parentId = currentSpan?.spanId
            traceId = currentTrace.traceId
            tracingAPIKey = currentTrace.tracingAPIKey
            traceMetadata = currentTrace.metadata
        case .trace(let trace)?:
            if isNoOp(trace) { return NoOpSpan(spanData: spanData) }
            traceId = trace.traceId
            parentId = nil
            tracingAPIKey = trace.tracingAPIKey
            traceMetadata = trace.metadata
        case .span(let span)?:
            if isNoOp(span) { return NoOpSpan(spanData: spanData) }
            parentId = span.spanId
            traceId = span.traceId
            tracingAPIKey = span.tracingAPIKey
            traceMetadata = span.traceMetadata
        }
        return SpanImpl(
            traceId: traceId,
            spanId: spanId ?? genSpanId(),
            parentId: parentId,
            processor: multiProcessor,
            spanData: spanData,
            tracingAPIKey: tracingAPIKey,
            traceMetadata: traceMetadata
        )
    }

    public func forceFlush() {
        multiProcessor.forceFlush()
    }

    public func shutdown(timeout: Double?) {
        logger.debug("Shutting down trace provider")
        multiProcessor.shutdown(timeout: timeout)
    }
}

private func isNoOp(_ span: any Span) -> Bool {
    span is NoOpSpan || span.spanId == "no-op" || span.traceId == "no-op"
}

private func isNoOp(_ trace: any Trace) -> Bool {
    trace is NoOpTrace || trace.traceId == "no-op"
}

private let globalTraceProvider = SharedBox<(any TraceProvider)?>(nil)

/// Sets the global trace provider used by tracing utilities.
public func setTraceProvider(_ provider: any TraceProvider) {
    globalTraceProvider.value = provider
}

/// Returns the global trace provider, creating the default provider and processor lazily.
public func getTraceProvider() -> any TraceProvider {
    globalTraceProvider.mutate { provider in
        if let provider { return provider }
        let created = DefaultTraceProvider()
        created.registerProcessor(defaultProcessor())
        provider = created
        return created
    }
}

func timeISO() -> String { getTraceProvider().timeISO() }

/// Generates a new trace ID.
public func genTraceId() -> String { getTraceProvider().genTraceId() }

/// Generates a new span ID.
public func genSpanId() -> String { getTraceProvider().genSpanId() }

/// Generates a new group ID.
public func genGroupId() -> String { getTraceProvider().genGroupId() }
