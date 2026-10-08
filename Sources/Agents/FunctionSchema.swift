import Foundation

// Swift replacement for src/agents/function_schema.py (porting/decisions.md D6).
//
// Upstream inspects Python signatures, type hints, and docstrings. Swift has no runtime signature
// reflection, so schemas are derived from a `Decodable` type by decoding it with a probing decoder
// that records the keys and value types it is asked for. Types can customize their schema by
// conforming to `JSONSchemaRepresentable` or describe fields with `JSONSchemaDescribable`.

/// A type that supplies its own JSON schema.
public protocol JSONSchemaRepresentable {
    /// The JSON schema describing values of this type.
    static var jsonSchema: JSONObject { get }
}

/// A type that adds descriptions to its generated JSON schema.
public protocol JSONSchemaDescribable {
    /// A description of the type itself.
    static var schemaDescription: String? { get }
    /// Descriptions keyed by coding key (the JSON property name).
    static var schemaPropertyDescriptions: [String: String] { get }
}

extension JSONSchemaDescribable {
    public static var schemaDescription: String? { nil }
    public static var schemaPropertyDescriptions: [String: String] { [:] }
}

/// Errors raised while generating a schema from a `Decodable` type.
public struct JSONSchemaGenerationError: Error, CustomStringConvertible, Sendable {
    public let description: String
    init(_ description: String) { self.description = description }
}

/// Generates JSON schemas from `Decodable` types.
public enum JSONSchemaGenerator {
    /// Returns the JSON schema for `type`.
    public static func schema<T: Decodable>(for type: T.Type) throws -> JSONObject {
        let probe = SchemaProbe()
        return try probe.schema(for: type)
    }
}

/// Builds schemas by decoding with dummy values.
private final class SchemaProbe {
    private var depth = 0

    func schema<T: Decodable>(for type: T.Type) throws -> JSONObject {
        depth += 1
        defer { depth -= 1 }
        if depth > 32 {
            throw JSONSchemaGenerationError("Recursive types are not supported for schema generation: \(T.self)")
        }
        if let representable = type as? JSONSchemaRepresentable.Type {
            return representable.jsonSchema
        }
        if let primitive = primitiveSchema(for: type) {
            return primitive
        }
        if let enumSchema = enumerationSchema(for: type) {
            return enumSchema
        }
        let recorder = SchemaRecorder(probe: self)
        let decoder = ProbeDecoder(recorder: recorder)
        _ = try T(from: decoder)
        var schema = recorder.result
        if let describable = type as? JSONSchemaDescribable.Type {
            if let description = describable.schemaDescription {
                schema["description"] = .string(description)
            }
            if case .object(var properties)? = schema["properties"] {
                for (key, description) in describable.schemaPropertyDescriptions {
                    if case .object(var property)? = properties[key] {
                        property["description"] = .string(description)
                        properties[key] = .object(property)
                    }
                }
                schema["properties"] = .object(properties)
            }
        }
        return schema
    }

    /// Returns a placeholder value that `T(from:)` accepts while probing.
    func placeholder<T: Decodable>(for type: T.Type) throws -> T {
        if let value = primitivePlaceholder(for: type) as? T {
            return value
        }
        if let first = firstCase(of: type) {
            return first
        }
        let recorder = SchemaRecorder(probe: self, recording: false)
        return try T(from: ProbeDecoder(recorder: recorder))
    }

    private func primitiveSchema(for type: Any.Type) -> JSONObject? {
        switch type {
        case is String.Type, is Substring.Type, is Character.Type: return ["type": "string"]
        case is UUID.Type: return ["type": "string", "format": "uuid"]
        case is URL.Type: return ["type": "string", "format": "uri"]
        case is Bool.Type: return ["type": "boolean"]
        case is Int.Type, is Int8.Type, is Int16.Type, is Int32.Type, is Int64.Type,
             is UInt.Type, is UInt8.Type, is UInt16.Type, is UInt32.Type, is UInt64.Type:
            return ["type": "integer"]
        case is Double.Type, is Float.Type, is Decimal.Type: return ["type": "number"]
        case is JSONValue.Type: return [:]
        case is JSONObject.Type: return ["type": "object"]
        default: return nil
        }
    }

    private func primitivePlaceholder(for type: Any.Type) -> Any? {
        switch type {
        case is String.Type: return ""
        case is Substring.Type: return Substring("")
        case is Character.Type: return Character(" ")
        case is UUID.Type: return UUID()
        case is URL.Type: return URL(string: "https://example.com")!
        case is Bool.Type: return false
        case is Int.Type: return 0
        case is Int8.Type: return Int8(0)
        case is Int16.Type: return Int16(0)
        case is Int32.Type: return Int32(0)
        case is Int64.Type: return Int64(0)
        case is UInt.Type: return UInt(0)
        case is UInt8.Type: return UInt8(0)
        case is UInt16.Type: return UInt16(0)
        case is UInt32.Type: return UInt32(0)
        case is UInt64.Type: return UInt64(0)
        case is Double.Type: return 0.0
        case is Float.Type: return Float(0)
        case is Decimal.Type: return Decimal(0)
        case is JSONValue.Type: return JSONValue.null
        case is JSONObject.Type: return JSONObject()
        default: return nil
        }
    }

    private func allCases(of type: Any.Type) -> [Any]? {
        guard let caseIterable = type as? any CaseIterable.Type else { return nil }
        func open<C: CaseIterable>(_ type: C.Type) -> [Any] { Array(type.allCases) }
        return open(caseIterable)
    }

    private func firstCase<T>(of type: T.Type) -> T? {
        allCases(of: type)?.first as? T
    }

    private func enumerationSchema(for type: Any.Type) -> JSONObject? {
        guard let cases = allCases(of: type), !cases.isEmpty else { return nil }
        var values: [JSONValue] = []
        for value in cases {
            guard let raw = (value as? any RawRepresentable)?.rawValue else { return nil }
            switch raw {
            case let string as String: values.append(.string(string))
            case let int as Int: values.append(.int(int))
            default: return nil
            }
        }
        let isString = values.allSatisfy { $0.stringValue != nil }
        return ["type": isString ? "string" : "integer", "enum": .array(values)]
    }
}

/// Records the properties requested while decoding one object.
private final class SchemaRecorder {
    let probe: SchemaProbe
    let recording: Bool
    var properties = JSONObject()
    var required: [String] = []
    var nestedRecorders: [(key: String, recorder: SchemaRecorder)] = []
    var singleValueSchema: JSONObject?
    var arrayItemSchema: JSONObject?
    var kind: Kind = .unknown

    enum Kind { case unknown, keyed, unkeyed, single }

    init(probe: SchemaProbe, recording: Bool = true) {
        self.probe = probe
        self.recording = recording
    }

    var result: JSONObject {
        switch kind {
        case .keyed, .unknown:
            var properties = self.properties
            for (key, recorder) in nestedRecorders where properties[key] == nil {
                properties[key] = .object(recorder.result)
            }
            let required = self.required + nestedRecorders.map(\.key).filter { !self.required.contains($0) }
            return [
                "type": "object",
                "properties": .object(properties),
                "required": .array(required.map(JSONValue.string)),
                "additionalProperties": false,
            ]
        case .unkeyed:
            return ["type": "array", "items": .object(arrayItemSchema ?? [:])]
        case .single:
            return singleValueSchema ?? [:]
        }
    }

    func record(key: String, schema: JSONObject, optional: Bool) {
        guard recording, properties[key] == nil else { return }
        if optional {
            properties[key] = .object(nullable(schema))
        } else {
            properties[key] = .object(schema)
            required.append(key)
        }
    }

    func nullable(_ schema: JSONObject) -> JSONObject {
        if case .string(let type)? = schema["type"], schema.count == 1 {
            return ["type": [.string(type), "null"]]
        }
        return ["anyOf": [.object(schema), ["type": "null"]]]
    }
}

private struct ProbeDecoder: Decoder {
    let recorder: SchemaRecorder
    var codingPath: [CodingKey] = []
    var userInfo: [CodingUserInfoKey: Any] = [:]

    func container<Key: CodingKey>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> {
        recorder.kind = .keyed
        return KeyedDecodingContainer(ProbeKeyedContainer<Key>(recorder: recorder))
    }

    func unkeyedContainer() throws -> UnkeyedDecodingContainer {
        recorder.kind = .unkeyed
        return ProbeUnkeyedContainer(recorder: recorder)
    }

    func singleValueContainer() throws -> SingleValueDecodingContainer {
        recorder.kind = .single
        return ProbeSingleValueContainer(recorder: recorder)
    }
}

private struct ProbeKeyedContainer<Key: CodingKey>: KeyedDecodingContainerProtocol {
    let recorder: SchemaRecorder
    var codingPath: [CodingKey] = []
    var allKeys: [Key] { [] }

    func contains(_ key: Key) -> Bool { true }

    func decodeNil(forKey key: Key) throws -> Bool { false }

    func decode<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> T {
        recorder.record(key: key.stringValue, schema: try recorder.probe.schema(for: type), optional: false)
        return try recorder.probe.placeholder(for: type)
    }

    func decodeIfPresent<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> T? {
        recorder.record(key: key.stringValue, schema: try recorder.probe.schema(for: type), optional: true)
        return nil
    }

    func nestedContainer<NestedKey: CodingKey>(keyedBy type: NestedKey.Type, forKey key: Key) throws -> KeyedDecodingContainer<NestedKey> {
        let nested = SchemaRecorder(probe: recorder.probe, recording: recorder.recording)
        nested.kind = .keyed
        if recorder.recording { recorder.nestedRecorders.append((key.stringValue, nested)) }
        return KeyedDecodingContainer(ProbeKeyedContainer<NestedKey>(recorder: nested))
    }

    func nestedUnkeyedContainer(forKey key: Key) throws -> UnkeyedDecodingContainer {
        let nested = SchemaRecorder(probe: recorder.probe, recording: recorder.recording)
        nested.kind = .unkeyed
        if recorder.recording { recorder.nestedRecorders.append((key.stringValue, nested)) }
        return ProbeUnkeyedContainer(recorder: nested)
    }

    func superDecoder() throws -> Decoder { ProbeDecoder(recorder: recorder) }
    func superDecoder(forKey key: Key) throws -> Decoder { ProbeDecoder(recorder: recorder) }
}

private struct ProbeUnkeyedContainer: UnkeyedDecodingContainer {
    let recorder: SchemaRecorder
    var codingPath: [CodingKey] = []
    var count: Int? { 1 }
    var currentIndex = 0
    var isAtEnd: Bool { currentIndex >= 1 }

    mutating func decodeNil() throws -> Bool { false }

    mutating func decode<T: Decodable>(_ type: T.Type) throws -> T {
        recorder.arrayItemSchema = try recorder.probe.schema(for: type)
        currentIndex += 1
        return try recorder.probe.placeholder(for: type)
    }

    mutating func nestedContainer<NestedKey: CodingKey>(keyedBy type: NestedKey.Type) throws -> KeyedDecodingContainer<NestedKey> {
        currentIndex += 1
        let nested = SchemaRecorder(probe: recorder.probe)
        nested.kind = .keyed
        recorder.arrayItemSchema = nested.result
        return KeyedDecodingContainer(ProbeKeyedContainer<NestedKey>(recorder: nested))
    }

    mutating func nestedUnkeyedContainer() throws -> UnkeyedDecodingContainer {
        currentIndex += 1
        return ProbeUnkeyedContainer(recorder: SchemaRecorder(probe: recorder.probe))
    }

    mutating func superDecoder() throws -> Decoder { ProbeDecoder(recorder: recorder) }
}

private struct ProbeSingleValueContainer: SingleValueDecodingContainer {
    let recorder: SchemaRecorder
    var codingPath: [CodingKey] = []

    func decodeNil() -> Bool { false }

    func decode<T: Decodable>(_ type: T.Type) throws -> T {
        recorder.singleValueSchema = try recorder.probe.schema(for: type)
        return try recorder.probe.placeholder(for: type)
    }
}
