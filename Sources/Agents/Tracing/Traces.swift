import Foundation

#if canImport(CryptoKit)
import CryptoKit
#endif

/// A trace represents a logical workflow (e.g. "Customer Service Query") and contains all the
/// spans that occur during that workflow.
public protocol Trace: AnyObject, Sendable {
    var traceId: String { get }
    var name: String { get }
    var tracingAPIKey: String? { get }
    /// Starts the trace.
    func start()
    /// Finishes the trace.
    func finish()
    /// Exports the trace for tracing backends, or `nil` for no-op traces.
    func export() -> JSONObject?
}

extension Trace {
    /// Serializes the trace, optionally including the tracing API key.
    public func toJSON(includeTracingAPIKey: Bool = false) -> JSONObject? {
        guard var payload = export() else { return nil }
        if includeTracingAPIKey, let key = tracingAPIKey, !key.isEmpty {
            payload["tracing_api_key"] = .string(key)
        }
        return payload
    }

    /// Runs `body` with this trace as the current trace.
    public func withCurrent<T>(_ body: () async throws -> T) async rethrows -> T {
        try await Scope.$currentTrace.withValue(self) {
            try await Scope.$currentSpan.withValue(nil, operation: body)
        }
    }

    /// Starts the trace, runs `body` with it as current, and finishes it afterwards.
    public func run<T>(_ body: () async throws -> T) async rethrows -> T {
        start()
        defer { finish() }
        return try await withCurrent(body)
    }

    /// Trace metadata, when the trace carries it.
    public var metadata: JSONObject? {
        export()?["metadata"]?.objectValue
    }
}

func hashTracingAPIKey(_ key: String?) -> String? {
    guard let key else { return nil }
    return sha256Hex(key)
}

/// Returns the lowercase hex SHA-256 digest of a string.
func sha256Hex(_ text: String) -> String {
    #if canImport(CryptoKit)
    return SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    #else
    return SHA256Fallback.hexDigest(Array(text.utf8))
    #endif
}

/// Serializable trace metadata used to resume traces across runs.
public struct TraceState: Sendable, Equatable {
    public var traceId: String?
    public var workflowName: String?
    public var groupId: String?
    public var metadata: JSONObject?
    public var tracingAPIKey: String?
    public var tracingAPIKeyHash: String?
    public var objectType: String?
    public var extra: JSONObject

    public init(
        traceId: String? = nil,
        workflowName: String? = nil,
        groupId: String? = nil,
        metadata: JSONObject? = nil,
        tracingAPIKey: String? = nil,
        tracingAPIKeyHash: String? = nil,
        objectType: String? = nil,
        extra: JSONObject = [:]
    ) {
        self.traceId = traceId
        self.workflowName = workflowName
        self.groupId = groupId
        self.metadata = metadata
        self.tracingAPIKey = tracingAPIKey
        self.tracingAPIKeyHash = tracingAPIKeyHash
        self.objectType = objectType
        self.extra = extra
    }

    public static func fromTrace(_ trace: (any Trace)?) -> TraceState? {
        guard let trace else { return nil }
        return fromJSON(trace.toJSON(includeTracingAPIKey: true))
    }

    public static func fromJSON(_ payload: JSONObject?) -> TraceState? {
        guard var data = payload, !data.isEmpty else { return nil }
        let objectType = data.removeValue(forKey: "object")?.stringValue
        let traceId = data.removeValue(forKey: "id")?.stringValue ?? data.removeValue(forKey: "trace_id")?.stringValue
        let workflowName = data.removeValue(forKey: "workflow_name")?.stringValue
        let groupId = data.removeValue(forKey: "group_id")?.stringValue
        let metadata = data.removeValue(forKey: "metadata")?.objectValue
        let key = data.removeValue(forKey: "tracing_api_key")?.stringValue
        let keyHash = data.removeValue(forKey: "tracing_api_key_hash")?.stringValue
        return TraceState(
            traceId: traceId,
            workflowName: workflowName,
            groupId: groupId,
            metadata: metadata,
            tracingAPIKey: key,
            tracingAPIKeyHash: hashTracingAPIKey(key) ?? keyHash,
            objectType: objectType,
            extra: data
        )
    }

    public func toJSON(includeTracingAPIKey: Bool = false) -> JSONObject? {
        if traceId == nil, workflowName == nil, groupId == nil, metadata == nil, tracingAPIKey == nil,
           tracingAPIKeyHash == nil, objectType == nil, extra.isEmpty {
            return nil
        }
        var payload = JSONObject()
        if let objectType, !objectType.isEmpty { payload["object"] = .string(objectType) }
        if let traceId, !traceId.isEmpty { payload["id"] = .string(traceId) }
        if let workflowName { payload["workflow_name"] = .string(workflowName) }
        if let groupId { payload["group_id"] = .string(groupId) }
        if let metadata { payload["metadata"] = .object(metadata) }
        if includeTracingAPIKey, let tracingAPIKey, !tracingAPIKey.isEmpty {
            payload["tracing_api_key"] = .string(tracingAPIKey)
        }
        if let tracingAPIKeyHash, !tracingAPIKeyHash.isEmpty {
            payload["tracing_api_key_hash"] = .string(tracingAPIKeyHash)
        }
        for (key, value) in extra where payload[key] == nil {
            payload[key] = value
        }
        return payload
    }
}

/// Remembers recently started trace IDs so resumed runs can reattach to live traces.
enum StartedTraceIds {
    private static let limit = 4096
    private static let box = SharedBox<[String]>([])

    static func markStarted(_ traceId: String?) {
        guard let traceId, !traceId.isEmpty, traceId != "no-op" else { return }
        box.mutate { ids in
            ids.removeAll { $0 == traceId }
            ids.append(traceId)
            if ids.count > limit { ids.removeFirst(ids.count - limit) }
        }
    }

    static func wasStarted(_ traceId: String?) -> Bool {
        guard let traceId, !traceId.isEmpty, traceId != "no-op" else { return false }
        return box.value.contains(traceId)
    }
}

/// A no-op trace used when tracing is disabled.
public final class NoOpTrace: Trace, @unchecked Sendable {
    public init() {}
    public var traceId: String { "no-op" }
    public var name: String { "no-op" }
    public var tracingAPIKey: String? { nil }
    public func start() {}
    public func finish() {}
    public func export() -> JSONObject? { nil }
}

/// A trace that will be recorded by the tracing library.
public final class TraceImpl: Trace, @unchecked Sendable {
    public let name: String
    public let traceId: String
    public let tracingAPIKey: String?
    public var groupId: String?
    public var traceMetadataValue: JSONObject?
    private let processor: any TracingProcessor
    private let lock = NSLock()
    private var started = false

    public init(
        name: String,
        traceId: String?,
        groupId: String?,
        metadata: JSONObject?,
        processor: any TracingProcessor,
        tracingAPIKey: String? = nil
    ) {
        self.name = name
        self.traceId = traceId ?? genTraceId()
        self.groupId = groupId
        self.traceMetadataValue = metadata
        self.processor = processor
        self.tracingAPIKey = tracingAPIKey
    }

    public func start() {
        lock.lock()
        if started {
            lock.unlock()
            return
        }
        started = true
        lock.unlock()
        processor.onTraceStart(self)
        StartedTraceIds.markStarted(traceId)
    }

    public func finish() {
        lock.lock()
        let wasStarted = started
        lock.unlock()
        guard wasStarted else { return }
        processor.onTraceEnd(self)
    }

    public func export() -> JSONObject? {
        [
            "object": "trace",
            "id": .string(traceId),
            "workflow_name": .string(name),
            "group_id": groupId.map(JSONValue.string) ?? .null,
            "metadata": traceMetadataValue.map(JSONValue.object) ?? .null,
        ]
    }
}

/// A trace reattached from serialized state; it does not re-emit trace start/end events.
public final class ReattachedTrace: Trace, @unchecked Sendable {
    public let name: String
    public let traceId: String
    public let tracingAPIKey: String?
    public var groupId: String?
    public var traceMetadataValue: JSONObject?

    public init(name: String, traceId: String, groupId: String?, metadata: JSONObject?, tracingAPIKey: String?) {
        self.name = name
        self.traceId = traceId
        self.groupId = groupId
        self.traceMetadataValue = metadata
        self.tracingAPIKey = tracingAPIKey
    }

    public func start() { StartedTraceIds.markStarted(traceId) }
    public func finish() {}

    public func export() -> JSONObject? {
        [
            "object": "trace",
            "id": .string(traceId),
            "workflow_name": .string(name),
            "group_id": groupId.map(JSONValue.string) ?? .null,
            "metadata": traceMetadataValue.map(JSONValue.object) ?? .null,
        ]
    }
}

/// Reattaches a trace from serialized state.
public func reattachTrace(_ state: TraceState, tracingAPIKey: String? = nil) -> (any Trace)? {
    guard let traceId = state.traceId else { return nil }
    return ReattachedTrace(
        name: state.workflowName ?? "Agent workflow",
        traceId: traceId,
        groupId: state.groupId,
        metadata: state.metadata,
        tracingAPIKey: state.tracingAPIKey ?? tracingAPIKey
    )
}

#if !canImport(CryptoKit)
/// Minimal SHA-256 for platforms without CryptoKit (Linux).
enum SHA256Fallback {
    static func hexDigest(_ message: [UInt8]) -> String {
        var hash: [UInt32] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ]
        let k: [UInt32] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ]
        var data = message
        let bitLength = UInt64(message.count) * 8
        data.append(0x80)
        while data.count % 64 != 56 { data.append(0) }
        for shift in stride(from: 56, through: 0, by: -8) { data.append(UInt8((bitLength >> UInt64(shift)) & 0xff)) }
        func rotr(_ x: UInt32, _ n: UInt32) -> UInt32 { (x >> n) | (x << (32 - n)) }
        for chunk in stride(from: 0, to: data.count, by: 64) {
            var w = [UInt32](repeating: 0, count: 64)
            for i in 0..<16 {
                w[i] = UInt32(data[chunk + i * 4]) << 24 | UInt32(data[chunk + i * 4 + 1]) << 16
                    | UInt32(data[chunk + i * 4 + 2]) << 8 | UInt32(data[chunk + i * 4 + 3])
            }
            for i in 16..<64 {
                let s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
                let s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
                w[i] = w[i - 16] &+ s0 &+ w[i - 7] &+ s1
            }
            var (a, b, c, d, e, f, g, h) = (hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7])
            for i in 0..<64 {
                let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
                let ch = (e & f) ^ (~e & g)
                let temp1 = h &+ s1 &+ ch &+ k[i] &+ w[i]
                let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
                let maj = (a & b) ^ (a & c) ^ (b & c)
                let temp2 = s0 &+ maj
                h = g; g = f; f = e; e = d &+ temp1; d = c; c = b; b = a; a = temp1 &+ temp2
            }
            hash[0] = hash[0] &+ a; hash[1] = hash[1] &+ b; hash[2] = hash[2] &+ c; hash[3] = hash[3] &+ d
            hash[4] = hash[4] &+ e; hash[5] = hash[5] &+ f; hash[6] = hash[6] &+ g; hash[7] = hash[7] &+ h
        }
        return hash.map { String(format: "%08x", $0) }.joined()
    }
}
#endif
