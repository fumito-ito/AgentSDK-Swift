import Foundation

/// An insertion-ordered JSON object, the Swift equivalent of a Python `dict` holding JSON data.
///
/// Swift: Python dictionaries preserve insertion order, and upstream relies on that order (for
/// example the `required` list of strict JSON schemas and serialized tool arguments). Equality
/// ignores order, as in Python. See porting/decisions.md D3.
public struct JSONObject: Sendable {
    public private(set) var keys: [String]
    private var storage: [String: JSONValue]

    public init() {
        keys = []
        storage = [:]
    }

    public init(minimumCapacity: Int) {
        keys = []
        keys.reserveCapacity(minimumCapacity)
        storage = Dictionary(minimumCapacity: minimumCapacity)
    }

    /// Creates an object from ordered key-value pairs. Later duplicates overwrite earlier values
    /// but keep the first position, like a Python dict literal.
    public init<S: Sequence>(_ pairs: S) where S.Element == (String, JSONValue) {
        self.init()
        for (key, value) in pairs {
            self[key] = value
        }
    }

    /// Creates an object from a Swift dictionary. Keys are sorted to give a deterministic order.
    public init(_ dictionary: [String: JSONValue]) {
        self.init(dictionary.sorted { $0.key < $1.key }.map { ($0.key, $0.value) })
    }

    public var count: Int { keys.count }
    public var isEmpty: Bool { keys.isEmpty }

    /// The values in key order.
    public var values: [JSONValue] { keys.map { storage[$0]! } }

    /// Ordered key-value pairs.
    public var pairs: [(key: String, value: JSONValue)] { keys.map { ($0, storage[$0]!) } }

    public subscript(key: String) -> JSONValue? {
        get { storage[key] }
        set {
            if let newValue {
                if storage.updateValue(newValue, forKey: key) == nil {
                    keys.append(key)
                }
            } else {
                removeValue(forKey: key)
            }
        }
    }

    /// Returns the value for `key`, or `defaultValue` when the key is missing. Mirrors `dict.get`.
    public subscript(key: String, default defaultValue: @autoclosure () -> JSONValue) -> JSONValue {
        storage[key] ?? defaultValue()
    }

    public func contains(_ key: String) -> Bool { storage[key] != nil }

    @discardableResult
    public mutating func removeValue(forKey key: String) -> JSONValue? {
        guard let removed = storage.removeValue(forKey: key) else { return nil }
        keys.removeAll { $0 == key }
        return removed
    }

    /// Merges `other` into this object; existing keys keep their position (like `dict.update`).
    public mutating func merge(_ other: JSONObject) {
        for (key, value) in other.pairs {
            self[key] = value
        }
    }

    /// Returns a copy merged with `other`, mirroring `{**self, **other}`.
    public func merging(_ other: JSONObject) -> JSONObject {
        var copy = self
        copy.merge(other)
        return copy
    }

    public func mapValues(_ transform: (JSONValue) throws -> JSONValue) rethrows -> JSONObject {
        var result = JSONObject(minimumCapacity: count)
        for key in keys {
            result[key] = try transform(storage[key]!)
        }
        return result
    }

    public func filter(_ isIncluded: (String, JSONValue) throws -> Bool) rethrows -> JSONObject {
        var result = JSONObject()
        for key in keys where try isIncluded(key, storage[key]!) {
            result[key] = storage[key]!
        }
        return result
    }

    /// Returns a copy without the given keys.
    public func removing(_ keysToRemove: String...) -> JSONObject {
        let removed = Set(keysToRemove)
        return filter { key, _ in !removed.contains(key) }
    }

    /// The object as an unordered Swift dictionary.
    public var dictionary: [String: JSONValue] { storage }
}

extension JSONObject: Sequence {
    public func makeIterator() -> AnyIterator<(key: String, value: JSONValue)> {
        var index = 0
        let keys = self.keys
        let storage = self.storage
        return AnyIterator {
            guard index < keys.count else { return nil }
            defer { index += 1 }
            return (keys[index], storage[keys[index]]!)
        }
    }
}

extension JSONObject: Equatable {
    public static func == (lhs: JSONObject, rhs: JSONObject) -> Bool {
        lhs.storage == rhs.storage
    }
}

extension JSONObject: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(storage)
    }
}

extension JSONObject: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self.init(elements)
    }
}

extension JSONObject: Codable {
    private struct AnyKey: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    public init(from decoder: Decoder) throws {
        // Swift: Foundation's keyed decoding does not expose source order, so keys decoded through
        // `Decoder` are sorted. Use `JSONValue.parse` to preserve order.
        let container = try decoder.container(keyedBy: AnyKey.self)
        self.init()
        for key in container.allKeys.sorted(by: { $0.stringValue < $1.stringValue }) {
            self[key.stringValue] = try container.decode(JSONValue.self, forKey: key)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: AnyKey.self)
        for (key, value) in pairs {
            try container.encode(value, forKey: AnyKey(stringValue: key))
        }
    }
}

extension JSONObject: CustomStringConvertible {
    public var description: String { JSONValue.object(self).jsonString(sortedKeys: false) }
}

extension JSONObject {
    /// Serializes the object to compact JSON text, preserving key order.
    public func jsonString() -> String { JSONValue.object(self).jsonString(sortedKeys: false) }

    /// Returns the string member for `key`, if present and a string.
    public func string(_ key: String) -> String? { self[key]?.stringValue }

    /// Parses JSON text that must contain an object.
    public static func parse(_ text: String) throws -> JSONObject {
        guard case .object(let object) = try JSONValue.parse(text) else {
            throw JSONSerializationError("Expected a JSON object")
        }
        return object
    }
}
