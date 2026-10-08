import Foundation

/// Encodes `Encodable` values into `JSONValue`, preserving the order in which keys are encoded.
///
/// Swift: upstream serializes pydantic models in field order. `JSONEncoder` does not guarantee key
/// order, so this encoder records keys as they are encoded (declaration order for synthesized
/// `Codable` conformances).
final class JSONValueEncoder {
    func encode<T: Encodable>(_ value: T) throws -> JSONValue {
        let storage = EncodingStorage()
        let encoder = JSONValueEncoderImpl(storage: storage, codingPath: [])
        try encoder.encodeValue(value)
        return storage.value.resolved()
    }
}

/// A mutable JSON tree used while encoding.
private final class EncodingNode {
    enum Kind {
        case value(JSONValue)
        case object([(String, EncodingNode)])
        case array([EncodingNode])
        case unset
    }

    var kind: Kind = .unset

    func resolved() -> JSONValue {
        switch kind {
        case .value(let value): return value
        case .unset: return .object([:])
        case .array(let nodes): return .array(nodes.map { $0.resolved() })
        case .object(let pairs):
            var object = JSONObject()
            for (key, node) in pairs { object[key] = node.resolved() }
            return .object(object)
        }
    }

    func child(forKey key: String) -> EncodingNode {
        if case .object(var pairs) = kind {
            if let existing = pairs.first(where: { $0.0 == key }) { return existing.1 }
            let node = EncodingNode()
            pairs.append((key, node))
            kind = .object(pairs)
            return node
        }
        let node = EncodingNode()
        kind = .object([(key, node)])
        return node
    }

    func appendChild() -> EncodingNode {
        let node = EncodingNode()
        if case .array(var nodes) = kind {
            nodes.append(node)
            kind = .array(nodes)
        } else {
            kind = .array([node])
        }
        return node
    }
}

private final class EncodingStorage {
    let value = EncodingNode()
}

private struct JSONValueEncoderImpl: Encoder {
    let node: EncodingNode
    var codingPath: [any CodingKey]
    var userInfo: [CodingUserInfoKey: Any] { [:] }

    init(storage: EncodingStorage, codingPath: [any CodingKey]) {
        self.node = storage.value
        self.codingPath = codingPath
    }

    init(node: EncodingNode, codingPath: [any CodingKey]) {
        self.node = node
        self.codingPath = codingPath
    }

    func container<Key: CodingKey>(keyedBy type: Key.Type) -> KeyedEncodingContainer<Key> {
        if case .object = node.kind {} else { node.kind = .object([]) }
        return KeyedEncodingContainer(KeyedContainer<Key>(node: node, codingPath: codingPath))
    }

    func unkeyedContainer() -> any UnkeyedEncodingContainer {
        if case .array = node.kind {} else { node.kind = .array([]) }
        return UnkeyedContainer(node: node, codingPath: codingPath)
    }

    func singleValueContainer() -> any SingleValueEncodingContainer {
        SingleValueContainer(node: node, codingPath: codingPath)
    }

    func encodeValue<T: Encodable>(_ value: T) throws {
        try encode(value, into: node, codingPath: codingPath)
    }
}

/// Encodes values with a direct JSON representation without going through `Encodable`.
private func primitiveJSON(_ value: any Encodable) -> JSONValue? {
    switch value {
    case let json as JSONValue: return json
    case let object as JSONObject: return .object(object)
    case let string as String: return .string(string)
    case let bool as Bool: return .bool(bool)
    case let int as Int: return .int(int)
    case let double as Double: return .double(double)
    case let float as Float: return .double(Double(float))
    case let date as Date: return .double(date.timeIntervalSinceReferenceDate)
    case let url as URL: return .string(url.absoluteString)
    case let data as Data: return .string(data.base64EncodedString())
    default: return nil
    }
}

private func encode<T: Encodable>(_ value: T, into node: EncodingNode, codingPath: [any CodingKey]) throws {
    if let primitive = primitiveJSON(value) {
        node.kind = .value(primitive)
        return
    }
    try value.encode(to: JSONValueEncoderImpl(node: node, codingPath: codingPath))
}

private struct KeyedContainer<Key: CodingKey>: KeyedEncodingContainerProtocol {
    let node: EncodingNode
    var codingPath: [any CodingKey]

    mutating func encodeNil(forKey key: Key) throws {
        node.child(forKey: key.stringValue).kind = .value(.null)
    }

    mutating func encode<T: Encodable>(_ value: T, forKey key: Key) throws {
        try Agents.encode(value, into: node.child(forKey: key.stringValue), codingPath: codingPath + [key])
    }

    mutating func nestedContainer<NestedKey: CodingKey>(keyedBy keyType: NestedKey.Type, forKey key: Key) -> KeyedEncodingContainer<NestedKey> {
        let child = node.child(forKey: key.stringValue)
        child.kind = .object([])
        return KeyedEncodingContainer(KeyedContainer<NestedKey>(node: child, codingPath: codingPath + [key]))
    }

    mutating func nestedUnkeyedContainer(forKey key: Key) -> any UnkeyedEncodingContainer {
        let child = node.child(forKey: key.stringValue)
        child.kind = .array([])
        return UnkeyedContainer(node: child, codingPath: codingPath + [key])
    }

    mutating func superEncoder() -> any Encoder {
        JSONValueEncoderImpl(node: node.child(forKey: "super"), codingPath: codingPath)
    }

    mutating func superEncoder(forKey key: Key) -> any Encoder {
        JSONValueEncoderImpl(node: node.child(forKey: key.stringValue), codingPath: codingPath + [key])
    }

    // Optionals encoded with `encodeIfPresent` are omitted when nil, as with `JSONEncoder`.
}

private struct IndexKey: CodingKey {
    var stringValue: String
    var intValue: Int?

    init(_ index: Int) {
        stringValue = "Index \(index)"
        intValue = index
    }

    init?(stringValue: String) {
        self.stringValue = stringValue
    }

    init?(intValue: Int) {
        self.init(intValue)
    }
}

private struct UnkeyedContainer: UnkeyedEncodingContainer {
    let node: EncodingNode
    var codingPath: [any CodingKey]
    var count: Int {
        if case .array(let nodes) = node.kind { return nodes.count }
        return 0
    }

    mutating func encodeNil() throws {
        node.appendChild().kind = .value(.null)
    }

    mutating func encode<T: Encodable>(_ value: T) throws {
        let key = IndexKey(count)
        try Agents.encode(value, into: node.appendChild(), codingPath: codingPath + [key])
    }

    mutating func nestedContainer<NestedKey: CodingKey>(keyedBy keyType: NestedKey.Type) -> KeyedEncodingContainer<NestedKey> {
        let key = IndexKey(count)
        let child = node.appendChild()
        child.kind = .object([])
        return KeyedEncodingContainer(KeyedContainer<NestedKey>(node: child, codingPath: codingPath + [key]))
    }

    mutating func nestedUnkeyedContainer() -> any UnkeyedEncodingContainer {
        let key = IndexKey(count)
        let child = node.appendChild()
        child.kind = .array([])
        return UnkeyedContainer(node: child, codingPath: codingPath + [key])
    }

    mutating func superEncoder() -> any Encoder {
        JSONValueEncoderImpl(node: node.appendChild(), codingPath: codingPath)
    }
}

private struct SingleValueContainer: SingleValueEncodingContainer {
    let node: EncodingNode
    var codingPath: [any CodingKey]

    mutating func encodeNil() throws { node.kind = .value(.null) }
    mutating func encode(_ value: Bool) throws { node.kind = .value(.bool(value)) }
    mutating func encode(_ value: String) throws { node.kind = .value(.string(value)) }
    mutating func encode(_ value: Double) throws { node.kind = .value(.double(value)) }
    mutating func encode(_ value: Float) throws { node.kind = .value(.double(Double(value))) }
    mutating func encode(_ value: Int) throws { node.kind = .value(.int(value)) }
    mutating func encode(_ value: Int8) throws { node.kind = .value(.int(Int(value))) }
    mutating func encode(_ value: Int16) throws { node.kind = .value(.int(Int(value))) }
    mutating func encode(_ value: Int32) throws { node.kind = .value(.int(Int(value))) }
    mutating func encode(_ value: Int64) throws { node.kind = .value(.int(Int(value))) }
    mutating func encode(_ value: UInt) throws { node.kind = .value(.int(Int(value))) }
    mutating func encode(_ value: UInt8) throws { node.kind = .value(.int(Int(value))) }
    mutating func encode(_ value: UInt16) throws { node.kind = .value(.int(Int(value))) }
    mutating func encode(_ value: UInt32) throws { node.kind = .value(.int(Int(value))) }
    mutating func encode(_ value: UInt64) throws { node.kind = .value(.int(Int(value))) }

    mutating func encode<T: Encodable>(_ value: T) throws {
        try Agents.encode(value, into: node, codingPath: codingPath)
    }
}
