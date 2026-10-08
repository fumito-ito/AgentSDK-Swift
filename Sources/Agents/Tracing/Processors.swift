import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Prints traces and spans to the console.
public final class ConsoleSpanExporter: TracingExporter, @unchecked Sendable {
    public init() {}

    public func export(_ items: [TraceOrSpan]) {
        for item in items {
            switch item {
            case .trace(let trace):
                print("[Exporter] Export trace_id=\(trace.traceId), name=\(trace.name)")
            case .span(let span):
                print("[Exporter] Export span: \(span.export() ?? [:])")
            }
        }
    }
}

/// A synchronous HTTP transport used by the trace exporter's background worker.
public protocol TracingHTTPTransport: Sendable {
    /// Sends a request and returns the status code, headers, and body.
    func send(_ request: URLRequest) throws -> (status: Int, headers: [String: String], body: Data)
}

/// The default transport, backed by `URLSession`.
public struct URLSessionTracingHTTPTransport: TracingHTTPTransport {
    private let session: URLSession

    public init(session: URLSession = .shared) {
        self.session = session
    }

    public func send(_ request: URLRequest) throws -> (status: Int, headers: [String: String], body: Data) {
        let semaphore = DispatchSemaphore(value: 0)
        let box = SharedBox<Result<(Int, [String: String], Data), any Error>?>(nil)
        let task = session.dataTask(with: request) { data, response, error in
            if let error {
                box.value = .failure(error)
            } else {
                let http = response as? HTTPURLResponse
                var headers: [String: String] = [:]
                for (key, value) in http?.allHeaderFields ?? [:] {
                    headers[String(describing: key).lowercased()] = String(describing: value)
                }
                box.value = .success((http?.statusCode ?? 0, headers, data ?? Data()))
            }
            semaphore.signal()
        }
        task.resume()
        semaphore.wait()
        switch box.value {
        case .success(let value)?: return value
        case .failure(let error)?: throw error
        case nil: throw URLError(.unknown)
        }
    }
}

/// Exports traces and spans to the OpenAI traces ingest endpoint.
public final class BackendSpanExporter: TracingExporter, @unchecked Sendable {
    public static let openAITracingIngestEndpoint = "https://api.openai.com/v1/traces/ingest"
    static let maxFieldBytes = 100_000
    static let truncationSuffix = "... [truncated]"
    static let allowedUsageKeys: Set<String> = ["input_tokens", "output_tokens"]
    static let usageSpanTypes: Set<String> = ["generation"]
    static let retryableClientStatusCodes: Set<Int> = [408, 409, 429]

    private let lock = NSLock()
    private var apiKeyValue: String?
    private let organizationValue: String?
    private let projectValue: String?
    public let endpoint: String
    public let maxRetries: Int
    public let baseDelay: Double
    public let maxDelay: Double
    private let transport: any TracingHTTPTransport
    private let shutdownRequested = SharedBox(false)

    public init(
        apiKey: String? = nil,
        organization: String? = nil,
        project: String? = nil,
        endpoint: String = BackendSpanExporter.openAITracingIngestEndpoint,
        maxRetries: Int = 3,
        baseDelay: Double = 1.0,
        maxDelay: Double = 30.0,
        transport: any TracingHTTPTransport = URLSessionTracingHTTPTransport()
    ) {
        self.apiKeyValue = apiKey
        self.organizationValue = organization
        self.projectValue = project
        self.endpoint = endpoint
        self.maxRetries = maxRetries
        self.baseDelay = baseDelay
        self.maxDelay = maxDelay
        self.transport = transport
    }

    /// Sets the OpenAI API key for the exporter.
    public func setAPIKey(_ apiKey: String) {
        lock.lock()
        apiKeyValue = apiKey
        lock.unlock()
    }

    /// The API key, falling back to `OPENAI_API_KEY`.
    public var apiKey: String? {
        lock.lock()
        defer { lock.unlock() }
        if let apiKeyValue, !apiKeyValue.isEmpty { return apiKeyValue }
        if let env = ProcessInfo.processInfo.environment["OPENAI_API_KEY"], !env.isEmpty {
            apiKeyValue = env
            return env
        }
        return nil
    }

    public var organization: String? { organizationValue ?? ProcessInfo.processInfo.environment["OPENAI_ORG_ID"] }
    public var project: String? { projectValue ?? ProcessInfo.processInfo.environment["OPENAI_PROJECT_ID"] }

    public func export(_ items: [TraceOrSpan]) {
        export(items, deadline: nil)
    }

    func requestShutdown() {
        shutdownRequested.value = true
    }

    func export(_ items: [TraceOrSpan], deadline: Date?) {
        guard !items.isEmpty else { return }
        var groupOrder: [String?] = []
        var groups: [String?: [TraceOrSpan]] = [:]
        for item in items {
            let key = item.tracingAPIKey
            if groups[key] == nil { groupOrder.append(key) }
            groups[key, default: []].append(item)
        }
        for key in groupOrder {
            guard let grouped = groups[key] else { continue }
            guard let apiKey = key ?? self.apiKey else {
                logger.warning("OPENAI_API_KEY is not set, skipping trace export")
                continue
            }
            let sanitize = endpoint.trimmingTrailingSlash == Self.openAITracingIngestEndpoint.trimmingTrailingSlash
            let data: [JSONValue] = grouped.compactMap { item in
                guard let exported = item.export() else { return nil }
                return .object(sanitize ? sanitizeForOpenAITracingAPI(exported) : exported)
            }
            let payload: JSONValue = ["data": .array(data)]
            guard let url = URL(string: endpoint) else { return }
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.httpBody = Data(payload.jsonString().utf8)
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("traces=v1", forHTTPHeaderField: "OpenAI-Beta")
            if let organization { request.setValue(organization, forHTTPHeaderField: "OpenAI-Organization") }
            if let project { request.setValue(project, forHTTPHeaderField: "OpenAI-Project") }

            var attempt = 0
            var delay = baseDelay
            while true {
                var retryAfter: Double?
                if let deadline {
                    let remaining = deadline.timeIntervalSinceNow
                    if remaining <= 0 {
                        logger.warning("[non-fatal] Tracing: export deadline reached, giving up on this batch.")
                        break
                    }
                    request.timeoutInterval = remaining
                } else {
                    request.timeoutInterval = 60
                }
                attempt += 1
                do {
                    let response = try transport.send(request)
                    if response.status < 300 {
                        logger.debug("Exported \(grouped.count) items")
                        break
                    }
                    let allowsRetry = response.headers["x-should-retry"]?.trimmingCharacters(in: .whitespaces).lowercased() != "false"
                    if Self.retryableClientStatusCodes.contains(response.status), allowsRetry {
                        retryAfter = parseRetryAfter(response.headers)
                        logger.warning("[non-fatal] Tracing: client error \(response.status), retrying.")
                    } else if (400..<500).contains(response.status) {
                        if AgentsDebug.dontLogModelData || AgentsDebug.dontLogToolData {
                            logger.error("[non-fatal] Tracing client error \(response.status). Response data is redacted.")
                        } else {
                            logger.error("[non-fatal] Tracing client error \(response.status): \(String(decoding: response.body, as: UTF8.self))")
                        }
                        break
                    } else if !allowsRetry {
                        logger.error("[non-fatal] Tracing: server forbade retry for \(response.status).")
                        break
                    } else {
                        logger.warning("[non-fatal] Tracing: server error \(response.status), retrying.")
                    }
                } catch {
                    logger.warning("[non-fatal] Tracing request failed: \(errorTypeName(error))")
                }
                if attempt >= maxRetries {
                    logger.error("[non-fatal] Tracing: max retries reached, giving up on this batch.")
                    break
                }
                var sleepTime = delay + Double.random(in: 0...(0.1 * delay))
                if let retryAfter {
                    sleepTime = min(max(sleepTime, retryAfter), maxDelay)
                }
                if let deadline {
                    let remaining = deadline.timeIntervalSinceNow
                    if remaining <= 0 { break }
                    Thread.sleep(forTimeInterval: min(sleepTime, remaining))
                    if sleepTime >= remaining { break }
                } else {
                    if shutdownRequested.value {
                        logger.warning("[non-fatal] Tracing: shutdown requested during retry backoff, giving up.")
                        break
                    }
                    Thread.sleep(forTimeInterval: sleepTime)
                }
                delay = min(delay * 2, maxDelay)
            }
        }
    }

    private func parseRetryAfter(_ headers: [String: String]) -> Double? {
        if let ms = headers["retry-after-ms"].flatMap(Double.init), ms.isFinite {
            return ms / 1000
        }
        if let seconds = headers["retry-after"].flatMap(Double.init), seconds.isFinite {
            return seconds
        }
        return nil
    }

    /// Omits reasoning from generation data and enforces traces ingest field limits.
    func sanitizeForOpenAITracingAPI(_ payload: JSONObject) -> JSONObject {
        guard case .object(var spanData)? = payload["span_data"] else { return payload }
        let spanType = spanData.string("type")
        for field in ["input", "output"] {
            guard var value = spanData[field] else { continue }
            if spanType == "generation" {
                value = omitGenerationReasoning(value)
            }
            spanData[field] = truncateForLimit(value, maxBytes: Self.maxFieldBytes)
        }
        if spanType.map(Self.usageSpanTypes.contains) != true {
            spanData.removeValue(forKey: "usage")
        } else if case .object(let usage)? = spanData["usage"] {
            if let sanitized = sanitizeGenerationUsage(usage) {
                spanData["usage"] = .object(sanitized)
            } else {
                spanData.removeValue(forKey: "usage")
            }
        }
        var result = payload
        result["span_data"] = .object(spanData)
        return result
    }

    private func omitGenerationReasoning(_ value: JSONValue) -> JSONValue {
        guard case .array(let items) = value else { return value }
        func omitMessageReasoning(_ item: JSONValue) -> JSONValue {
            guard case .object(var object) = item else { return item }
            if object.string("role") == "assistant" {
                for key in ["reasoning", "reasoning_content", "thinking_blocks", "thinking"] {
                    object.removeValue(forKey: key)
                }
                if case .array(let content)? = object["content"] {
                    object["content"] = .array(content.filter { part in
                        guard case .object(let partObject) = part else { return true }
                        return partObject.string("type") != "thinking" && partObject.string("type") != "redacted_thinking"
                    })
                }
            }
            return .object(object)
        }
        func omitItems(_ items: [JSONValue]) -> [JSONValue] {
            items.filter { $0["type"]?.stringValue != "reasoning" }.map(omitMessageReasoning)
        }
        return .array(omitItems(items).map { item in
            guard case .object(var object) = item, object.string("object") == "response",
                  case .array(let output)? = object["output"] else { return item }
            object["output"] = .array(omitItems(output))
            return .object(object)
        })
    }

    private func sanitizeGenerationUsage(_ usage: JSONObject) -> JSONObject? {
        guard let input = usage["input_tokens"], input.doubleValue != nil, input.boolValue == nil,
              let output = usage["output_tokens"], output.doubleValue != nil, output.boolValue == nil else {
            return nil
        }
        var details = usage["details"]?.objectValue ?? JSONObject()
        for (key, value) in usage where !Self.allowedUsageKeys.contains(key) && key != "details" && !value.isNull {
            details[key] = value
        }
        var sanitized: JSONObject = ["input_tokens": input, "output_tokens": output]
        if !details.isEmpty { sanitized["details"] = .object(details) }
        return sanitized
    }

    private func jsonSize(_ value: JSONValue) -> Int {
        value.jsonString().utf8.count
    }

    private func truncateForLimit(_ value: JSONValue, maxBytes: Int) -> JSONValue {
        if jsonSize(value) <= maxBytes { return value }
        switch value {
        case .string(let string):
            let suffix = Self.truncationSuffix
            let budget = maxBytes - jsonSize(.string(suffix))
            guard budget > 0 else { return .string("") }
            var characters = Int(Double(string.count) * Double(budget) / Double(max(jsonSize(value), 1)))
            var candidate = String(string.prefix(max(0, characters))) + suffix
            while jsonSize(.string(candidate)) > maxBytes, characters > 0 {
                characters = max(0, characters - max(1, characters / 10))
                candidate = String(string.prefix(characters)) + suffix
            }
            return .string(candidate)
        case .array(var array):
            while !array.isEmpty, jsonSize(.array(array)) > maxBytes {
                if let largest = array.indices.max(by: { jsonSize(array[$0]) < jsonSize(array[$1]) }) {
                    let others = jsonSize(.array(array)) - jsonSize(array[largest])
                    let truncated = truncateForLimit(array[largest], maxBytes: max(0, maxBytes - others))
                    if truncated == array[largest] {
                        array.remove(at: largest)
                    } else {
                        array[largest] = truncated
                    }
                }
            }
            return .array(array)
        case .object(var object):
            while !object.isEmpty, jsonSize(.object(object)) > maxBytes {
                if let largest = object.keys.max(by: { jsonSize(object[$0]!) < jsonSize(object[$1]!) }) {
                    let child = object[largest]!
                    let others = jsonSize(.object(object)) - jsonSize(child)
                    let truncated = truncateForLimit(child, maxBytes: max(0, maxBytes - others))
                    if truncated == child {
                        object.removeValue(forKey: largest)
                    } else {
                        object[largest] = truncated
                    }
                }
            }
            return .object(object)
        default:
            return value
        }
    }
}

extension String {
    fileprivate var trimmingTrailingSlash: String {
        var copy = self
        while copy.hasSuffix("/") { copy.removeLast() }
        return copy
    }
}

/// Exports traces and spans in batches from a background thread.
///
/// Spans are stored in memory until they are exported.
public final class BatchTraceProcessor: TracingProcessor, @unchecked Sendable {
    private let exporter: any TracingExporter
    private let maxQueueSize: Int
    private let maxBatchSize: Int
    private let scheduleDelay: Double
    private let exportTriggerSize: Int
    private let lock = NSLock()
    private let exportLock = NSLock()
    private var queue: [TraceOrSpan] = []
    private var workerStarted = false
    private var shutdownFlag = false
    private var shutdownDeadline: Date?
    private let workerDone = DispatchSemaphore(value: 0)

    public init(
        exporter: any TracingExporter,
        maxQueueSize: Int = 8192,
        maxBatchSize: Int = 128,
        scheduleDelay: Double = 5.0,
        exportTriggerRatio: Double = 0.7
    ) {
        self.exporter = exporter
        self.maxQueueSize = maxQueueSize
        self.maxBatchSize = maxBatchSize
        self.scheduleDelay = scheduleDelay
        self.exportTriggerSize = max(1, Int(Double(maxQueueSize) * exportTriggerRatio))
    }

    private func ensureWorkerStarted() {
        lock.lock()
        guard !workerStarted, !shutdownFlag else {
            lock.unlock()
            return
        }
        workerStarted = true
        lock.unlock()
        let thread = Thread { [weak self] in self?.run() }
        thread.name = "openai-agents-trace-export"
        thread.start()
    }

    private func enqueue(_ item: TraceOrSpan, label: String) {
        ensureWorkerStarted()
        lock.lock()
        if queue.count >= maxQueueSize {
            lock.unlock()
            logger.warning("Queue is full, dropping \(label).")
            return
        }
        queue.append(item)
        lock.unlock()
    }

    public func onTraceStart(_ trace: any Trace) { enqueue(.trace(trace), label: "trace") }
    public func onTraceEnd(_ trace: any Trace) {}
    public func onSpanStart(_ span: any Span) {}
    public func onSpanEnd(_ span: any Span) { enqueue(.span(span), label: "span") }

    public func shutdown(timeout: Double?) {
        lock.lock()
        shutdownFlag = true
        let started = workerStarted
        shutdownDeadline = timeout.map { Date().addingTimeInterval($0) }
        lock.unlock()
        if timeout != nil, let backend = exporter as? BackendSpanExporter {
            backend.requestShutdown()
        }
        if started {
            let result: DispatchTimeoutResult
            if let timeout {
                result = workerDone.wait(timeout: .now() + timeout)
            } else {
                workerDone.wait()
                result = .success
            }
            if result == .timedOut {
                logger.warning("[non-fatal] Tracing: shutdown timeout reached; dropping queued traces.")
            }
        } else {
            exportBatches(deadline: shutdownDeadline)
        }
    }

    public func forceFlush() {
        exportBatches(deadline: nil)
    }

    private func run() {
        var nextExport = Date().addingTimeInterval(scheduleDelay)
        while true {
            lock.lock()
            let stop = shutdownFlag
            let size = queue.count
            lock.unlock()
            if stop { break }
            if Date() >= nextExport || size >= exportTriggerSize {
                exportBatches(deadline: nil)
                nextExport = Date().addingTimeInterval(scheduleDelay)
            } else {
                Thread.sleep(forTimeInterval: 0.2)
            }
        }
        lock.lock()
        let deadline = shutdownDeadline
        lock.unlock()
        exportBatches(deadline: deadline)
        workerDone.signal()
    }

    private func exportBatches(deadline: Date?) {
        exportLock.lock()
        defer { exportLock.unlock() }
        while true {
            if let deadline, Date() >= deadline {
                logger.warning("[non-fatal] Tracing: export deadline reached; dropping queued traces.")
                break
            }
            lock.lock()
            let batch = Array(queue.prefix(maxBatchSize))
            queue.removeFirst(batch.count)
            lock.unlock()
            if batch.isEmpty { break }
            if let deadline, let backend = exporter as? BackendSpanExporter {
                backend.export(batch, deadline: deadline)
            } else {
                exporter.export(batch)
            }
        }
    }
}

private let defaultExporterBox = SharedBox<BackendSpanExporter?>(nil)
private let defaultProcessorBox = SharedBox<BatchTraceProcessor?>(nil)

/// The default exporter, which exports traces and spans to the backend in batches.
public func defaultExporter() -> BackendSpanExporter {
    defaultExporterBox.mutate { exporter in
        if let exporter { return exporter }
        let created = BackendSpanExporter()
        exporter = created
        return created
    }
}

/// The default processor, which exports traces and spans to the backend in batches.
public func defaultProcessor() -> BatchTraceProcessor {
    defaultProcessorBox.mutate { processor in
        if let processor { return processor }
        let created = BatchTraceProcessor(exporter: defaultExporter())
        processor = created
        return created
    }
}
