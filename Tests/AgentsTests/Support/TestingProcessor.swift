import Foundation
import Testing
@testable import Agents

// Port of tests/testing_processor.py and the tracing fixtures in tests/conftest.py.
//
// Swift: upstream installs one global processor and clears it before each (serial) test. Swift
// Testing runs tests in parallel, so a routing processor forwards events to the
// `SpanProcessorForTests` bound to the current task with the `.tracing` trait.

/// A processor that stores traces and finished spans in memory.
final class SpanProcessorForTests: TracingProcessor, @unchecked Sendable {
    enum Event: String, Sendable {
        case traceStart = "trace_start"
        case traceEnd = "trace_end"
        case spanStart = "span_start"
        case spanEnd = "span_end"
    }

    private let lock = NSLock()
    private var spans: [any Span] = []
    private var traces: [any Trace] = []
    private var recordedEvents: [Event] = []
    private var startOrder: [String: Int] = [:]

    func onTraceStart(_ trace: any Trace) {
        lock.locked {
            traces.append(trace)
            recordedEvents.append(.traceStart)
        }
    }

    func onTraceEnd(_ trace: any Trace) {
        lock.locked { recordedEvents.append(.traceEnd) }
    }

    func onSpanStart(_ span: any Span) {
        lock.locked {
            recordedEvents.append(.spanStart)
            let key = "\(span.traceId)/\(span.spanId)"
            if startOrder[key] == nil { startOrder[key] = startOrder.count }
        }
    }

    func onSpanEnd(_ span: any Span) {
        lock.locked {
            recordedEvents.append(.spanEnd)
            spans.append(span)
        }
    }

    func shutdown(timeout: Double?) {}
    func forceFlush() {}

    /// Finished spans in start order.
    func getOrderedSpans(includingEmpty: Bool = false) -> [any Span] {
        lock.locked {
            spans
                .filter { includingEmpty || $0.export() != nil }
                .sorted {
                    let lhs = ($0.startedAt ?? "", startOrder["\($0.traceId)/\($0.spanId)"] ?? 0)
                    let rhs = ($1.startedAt ?? "", startOrder["\($1.traceId)/\($1.spanId)"] ?? 0)
                    return lhs < rhs
                }
        }
    }

    func getTraces(includingEmpty: Bool = false) -> [any Trace] {
        lock.locked { traces.filter { includingEmpty || $0.export() != nil } }
    }

    var events: [Event] { lock.locked { recordedEvents } }

    func clear() {
        lock.locked {
            spans.removeAll()
            traces.removeAll()
            recordedEvents.removeAll()
            startOrder.removeAll()
        }
    }
}

/// Forwards tracing events to the collector bound to the current task.
private final class RoutingTestProcessor: TracingProcessor, @unchecked Sendable {
    func onTraceStart(_ trace: any Trace) { TestTracing.current?.onTraceStart(trace) }
    func onTraceEnd(_ trace: any Trace) { TestTracing.current?.onTraceEnd(trace) }
    func onSpanStart(_ span: any Span) { TestTracing.current?.onSpanStart(span) }
    func onSpanEnd(_ span: any Span) { TestTracing.current?.onSpanEnd(span) }
    func shutdown(timeout: Double?) {}
    func forceFlush() {}
}

enum TestTracing {
    @TaskLocal static var current: SpanProcessorForTests?

    /// Installs a trace provider that never exports and routes events to `current`.
    static let installed: Bool = {
        let provider = DefaultTraceProvider()
        provider.setProcessors([RoutingTestProcessor()])
        provider.setDisabled(false)
        setTraceProvider(provider)
        return true
    }()
}

/// Binds a fresh `SpanProcessorForTests` to each test (upstream `clear_span_processor`).
struct TracingTrait: TestTrait, SuiteTrait, TestScoping {
    var isRecursive: Bool { true }

    func provideScope(for test: Test, testCase: Test.Case?, performing function: @Sendable () async throws -> Void) async throws {
        _ = TestTracing.installed
        guard testCase != nil || !test.isSuite else {
            try await function()
            return
        }
        try await TestTracing.$current.withValue(SpanProcessorForTests()) {
            try await function()
        }
    }
}

extension Trait where Self == TracingTrait {
    /// Records traces and spans for the test in an isolated `SpanProcessorForTests`.
    static var tracing: TracingTrait { TracingTrait() }
}

private var processor: SpanProcessorForTests {
    guard let current = TestTracing.current else {
        fatalError("Tracing helpers require the .tracing trait")
    }
    return current
}

func fetchOrderedSpans() -> [any Span] { processor.getOrderedSpans() }

func fetchTraces() -> [any Trace] { processor.getTraces() }

func fetchEvents() -> [SpanProcessorForTests.Event] { processor.events }

/// Errors recorded on exported spans of the given type.
func fetchSpanErrors(_ spanType: String) -> [JSONObject] {
    fetchOrderedSpans().compactMap { span -> JSONObject? in
        guard let exported = span.export(), let spanData = exported["span_data"]?.objectValue else { return nil }
        var exportedType = spanData.string("type")
        if exportedType == "custom", let data = spanData["data"]?.objectValue {
            exportedType = data.string("sdk_span_type") ?? exportedType
        }
        guard exportedType == spanType else { return nil }
        return exported["error"]?.objectValue
    }
}

func assertNoSpans(sourceLocation: SourceLocation = #_sourceLocation) {
    let spans = fetchOrderedSpans()
    #expect(spans.isEmpty, "Expected 0 spans, got \(spans.count)", sourceLocation: sourceLocation)
}

func assertNoTraces(sourceLocation: SourceLocation = #_sourceLocation) {
    let traces = fetchTraces()
    #expect(traces.isEmpty, "Expected 0 traces, got \(traces.count)", sourceLocation: sourceLocation)
    assertNoSpans(sourceLocation: sourceLocation)
}

/// Traces and spans as nested dictionaries, with IDs and timestamps removed.
func fetchNormalizedSpans(keepSpanId: Bool = false, keepTraceId: Bool = false) -> [JSONValue] {
    final class Node {
        var object: JSONObject
        var children: [Node] = []
        init(_ object: JSONObject) { self.object = object }

        func resolved() -> JSONValue {
            var object = self.object
            if !children.isEmpty {
                object["children"] = .array(children.map { $0.resolved() })
            }
            return .object(object)
        }
    }

    var nodes: [String: Node] = [:]
    var roots: [Node] = []
    for traceObject in fetchTraces() {
        guard var trace = traceObject.export() else { continue }
        #expect(trace.removeValue(forKey: "object") == "trace")
        #expect(trace.string("id")?.hasPrefix("trace_") == true)
        if !keepTraceId { trace.removeValue(forKey: "id") }
        let node = Node(trace.filter { _, value in !value.isNull })
        nodes["\(traceObject.traceId)/"] = node
        roots.append(node)
    }
    #expect(!roots.isEmpty, "Use assertNoTraces() to check for empty traces")

    for spanObject in fetchOrderedSpans() {
        guard var span = spanObject.export() else { continue }
        #expect(span.removeValue(forKey: "object") == "trace.span")
        #expect(span.string("id")?.hasPrefix("span_") == true)
        if !keepSpanId { span.removeValue(forKey: "id") }
        span.removeValue(forKey: "started_at")
        span.removeValue(forKey: "ended_at")
        let parentId = span.removeValue(forKey: "parent_id")?.stringValue ?? ""
        var spanData = span.removeValue(forKey: "span_data")?.objectValue ?? [:]
        let type = spanData.removeValue(forKey: "type") ?? .null
        let traceId = span.removeValue(forKey: "trace_id")?.stringValue ?? ""
        var normalized: JSONObject = ["type": type]
        normalized.merge(span.filter { _, value in !value.isNull })
        spanData = spanData.filter { _, value in !value.isNull }
        if !spanData.isEmpty { normalized["data"] = .object(spanData) }

        var sdkSpanType: String?
        if type == "custom" { sdkSpanType = spanData["data"]?.objectValue?.string("sdk_span_type") }
        let parentKey = "\(traceId)/\(parentId)"
        if type == "task" || type == "turn" || sdkSpanType == "task" || sdkSpanType == "turn" {
            guard let parent = nodes[parentKey] else { continue }
            if let error = normalized["error"], parent.object["error"] == nil {
                parent.object["error"] = error
            }
            nodes["\(spanObject.traceId)/\(spanObject.spanId)"] = parent
            continue
        }
        let node = Node(normalized)
        nodes["\(spanObject.traceId)/\(spanObject.spanId)"] = node
        nodes[parentKey]?.children.append(node)
    }
    return roots.map { $0.resolved() }
}
