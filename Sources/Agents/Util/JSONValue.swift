import Foundation

/// A dynamically typed JSON value.
///
/// Integers and floating point numbers are kept distinct so that payloads round-trip without
/// turning `1` into `1.0`, but they compare equal when numerically equal (matching Python, where
/// `1 == 1.0`).
public enum JSONValue: Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object(JSONObject)
}

// MARK: - Accessors

extension JSONValue {
    public var isNull: Bool {
        if case .null = self { return true }
        return false
    }

    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    public var boolValue: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }

    /// The value as an integer. Doubles with an exact integer value are accepted.
    public var intValue: Int? {
        switch self {
        case .int(let value):
            return value
        case .double(let value):
            guard value.rounded() == value, let int = Int(exactly: value) else { return nil }
            return int
        default:
            return nil
        }
    }

    /// The value as a double. Integers are converted.
    public var doubleValue: Double? {
        switch self {
        case .int(let value): return Double(value)
        case .double(let value): return value
        default: return nil
        }
    }

    public var arrayValue: [JSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }

    public var objectValue: JSONObject? {
        if case .object(let value) = self { return value }
        return nil
    }

    /// Returns the member for `key` when this value is an object, otherwise `nil`.
    public subscript(key: String) -> JSONValue? {
        get { objectValue?[key] }
        set {
            guard case .object(var object) = self else { return }
            object[key] = newValue
            self = .object(object)
        }
    }

    /// Returns the element at `index` when this value is an array and the index is in range.
    public subscript(index: Int) -> JSONValue? {
        guard case .array(let array) = self, array.indices.contains(index) else { return nil }
        return array[index]
    }

    /// The Python `type(value).__name__` equivalent, used in error messages.
    public var typeName: String {
        switch self {
        case .null: return "NoneType"
        case .bool: return "bool"
        case .int: return "int"
        case .double: return "float"
        case .string: return "str"
        case .array: return "list"
        case .object: return "dict"
        }
    }

    /// Python truthiness: `None`, `False`, `0`, empty strings, arrays, and objects are falsy.
    public var isTruthy: Bool {
        switch self {
        case .null: return false
        case .bool(let value): return value
        case .int(let value): return value != 0
        case .double(let value): return value != 0
        case .string(let value): return !value.isEmpty
        case .array(let value): return !value.isEmpty
        case .object(let value): return !value.isEmpty
        }
    }
}

// MARK: - Equatable / Hashable

extension JSONValue: Equatable {
    public static func == (lhs: JSONValue, rhs: JSONValue) -> Bool {
        switch (lhs, rhs) {
        case (.null, .null):
            return true
        case let (.bool(left), .bool(right)):
            return left == right
        case let (.string(left), .string(right)):
            return left == right
        case let (.array(left), .array(right)):
            return left == right
        case let (.object(left), .object(right)):
            return left == right
        case let (.int(left), .int(right)):
            return left == right
        case (.int, .double), (.double, .int), (.double, .double):
            return lhs.doubleValue == rhs.doubleValue
        default:
            return false
        }
    }
}

extension JSONValue: Hashable {
    public func hash(into hasher: inout Hasher) {
        switch self {
        case .null:
            hasher.combine(0)
        case .bool(let value):
            hasher.combine(1)
            hasher.combine(value)
        case .int, .double:
            hasher.combine(2)
            hasher.combine(doubleValue)
        case .string(let value):
            hasher.combine(3)
            hasher.combine(value)
        case .array(let value):
            hasher.combine(4)
            hasher.combine(value)
        case .object(let value):
            hasher.combine(5)
            hasher.combine(value)
        }
    }
}

// MARK: - Literals

extension JSONValue: ExpressibleByNilLiteral {
    public init(nilLiteral: ()) { self = .null }
}

extension JSONValue: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) { self = .bool(value) }
}

extension JSONValue: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) { self = .int(value) }
}

extension JSONValue: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) { self = .double(value) }
}

extension JSONValue: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
}

extension JSONValue: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
}

extension JSONValue: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self = .object(JSONObject(elements))
    }
}

// MARK: - Convenience initializers

extension JSONValue {
    public init(_ value: String) { self = .string(value) }
    public init(_ value: Bool) { self = .bool(value) }
    public init(_ value: Int) { self = .int(value) }
    public init(_ value: Double) { self = .double(value) }
    public init(_ value: [JSONValue]) { self = .array(value) }
    public init(_ value: JSONObject) { self = .object(value) }

    /// Wraps an optional string, mapping `nil` to `.null`.
    public init(optional value: String?) {
        self = value.map(JSONValue.string) ?? .null
    }
}

// MARK: - Codable

extension JSONValue: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode(JSONObject.self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}

// MARK: - Serialization

/// Errors raised when converting between JSON text and `JSONValue`.
public struct JSONSerializationError: Error, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

extension JSONValue {
    /// Parses JSON text. Mirrors `json.loads`.
    public static func parse(_ text: String) throws -> JSONValue {
        try parse(Data(text.utf8))
    }

    /// Parses JSON data. Mirrors `json.loads`.
    public static func parse(_ data: Data) throws -> JSONValue {
        var parser = JSONTextParser(bytes: Array(data))
        return try parser.parseDocument()
    }

    /// Serializes to compact JSON text (`separators=(",", ":")`), preserving key order by default.
    public func jsonString(sortedKeys: Bool = false, prettyPrinted: Bool = false) -> String {
        var output = ""
        JSONValue.write(self, into: &output, sortedKeys: sortedKeys, pretty: prettyPrinted, indent: 0)
        return output
    }

    /// Serializes like Python's `json.dumps(value)` with default separators (`", "` and `": "`)
    /// and `ensure_ascii=True`.
    public func pythonJSONString() -> String {
        var output = ""
        JSONValue.writePython(self, into: &output)
        return output
    }

    private static func write(_ value: JSONValue, into output: inout String, sortedKeys: Bool, pretty: Bool, indent: Int) {
        switch value {
        case .null:
            output += "null"
        case .bool(let bool):
            output += bool ? "true" : "false"
        case .int(let int):
            output += String(int)
        case .double(let double):
            output += formatDouble(double)
        case .string(let string):
            output += escape(string)
        case .array(let array):
            if array.isEmpty {
                output += "[]"
                return
            }
            output += "["
            for (index, element) in array.enumerated() {
                if index > 0 { output += "," }
                if pretty { output += "\n" + String(repeating: "  ", count: indent + 1) }
                write(element, into: &output, sortedKeys: sortedKeys, pretty: pretty, indent: indent + 1)
            }
            if pretty { output += "\n" + String(repeating: "  ", count: indent) }
            output += "]"
        case .object(let object):
            if object.isEmpty {
                output += "{}"
                return
            }
            output += "{"
            let keys = sortedKeys ? object.keys.sorted() : object.keys
            for (index, key) in keys.enumerated() {
                if index > 0 { output += "," }
                if pretty { output += "\n" + String(repeating: "  ", count: indent + 1) }
                output += escape(key)
                output += pretty ? ": " : ":"
                write(object[key] ?? .null, into: &output, sortedKeys: sortedKeys, pretty: pretty, indent: indent + 1)
            }
            if pretty { output += "\n" + String(repeating: "  ", count: indent) }
            output += "}"
        }
    }

    private static func writePython(_ value: JSONValue, into output: inout String) {
        switch value {
        case .array(let array):
            output += "["
            for (index, element) in array.enumerated() {
                if index > 0 { output += ", " }
                writePython(element, into: &output)
            }
            output += "]"
        case .object(let object):
            output += "{"
            for (index, key) in object.keys.enumerated() {
                if index > 0 { output += ", " }
                output += escape(key, asciiOnly: true)
                output += ": "
                writePython(object[key] ?? .null, into: &output)
            }
            output += "}"
        case .string(let string):
            output += escape(string, asciiOnly: true)
        default:
            write(value, into: &output, sortedKeys: false, pretty: false, indent: 0)
        }
    }

    static func formatDouble(_ double: Double) -> String {
        if double.isNaN { return "NaN" }
        if double.isInfinite { return double > 0 ? "Infinity" : "-Infinity" }
        if double.rounded() == double, abs(double) < 1e16 {
            return String(format: "%.1f", double)
        }
        return "\(double)"
    }

    static func escape(_ string: String, asciiOnly: Bool = false) -> String {
        var result = "\""
        result.reserveCapacity(string.count + 2)
        for scalar in string.unicodeScalars {
            switch scalar {
            case "\"": result += "\\\""
            case "\\": result += "\\\\"
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            case "\u{08}": result += "\\b"
            case "\u{0C}": result += "\\f"
            default:
                if scalar.value < 0x20 || (asciiOnly && scalar.value > 0x7E) {
                    if scalar.value > 0xFFFF {
                        let value = scalar.value - 0x10000
                        let high = 0xD800 + (value >> 10)
                        let low = 0xDC00 + (value & 0x3FF)
                        result += String(format: "\\u%04x\\u%04x", high, low)
                    } else {
                        result += String(format: "\\u%04x", scalar.value)
                    }
                } else {
                    result.unicodeScalars.append(scalar)
                }
            }
        }
        result += "\""
        return result
    }
}

extension JSONValue: CustomStringConvertible {
    public var description: String { jsonString() }
}

// MARK: - Bridging Codable values

extension JSONValue {
    /// Encodes any `Encodable` value into a `JSONValue`.
    public init<T: Encodable>(encoding value: T, encoder: JSONEncoder = JSONEncoder()) throws {
        let data = try encoder.encode(value)
        self = try JSONValue.parse(data)
    }

    /// Decodes this value into a `Decodable` type.
    public func decode<T: Decodable>(as type: T.Type = T.self, decoder: JSONDecoder = JSONDecoder()) throws -> T {
        let data = Data(jsonString().utf8)
        return try decoder.decode(type, from: data)
    }
}

// MARK: - Parser

/// A small JSON parser that preserves object key order (Foundation's decoders do not).
struct JSONTextParser {
    let bytes: [UInt8]
    var index = 0

    init(bytes: [UInt8]) {
        self.bytes = bytes
    }

    mutating func parseDocument() throws -> JSONValue {
        skipWhitespace()
        let value = try parseValue(depth: 0)
        skipWhitespace()
        guard index == bytes.count else { throw error("Extra data") }
        return value
    }

    private func error(_ message: String) -> JSONSerializationError {
        JSONSerializationError("\(message): line 1 column \(index + 1) (char \(index))")
    }

    private mutating func skipWhitespace() {
        while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) {
            index += 1
        }
    }

    private mutating func parseValue(depth: Int) throws -> JSONValue {
        guard depth < 512 else { throw error("Maximum nesting depth exceeded") }
        guard index < bytes.count else { throw error("Expecting value") }
        switch bytes[index] {
        case UInt8(ascii: "{"): return try parseObject(depth: depth)
        case UInt8(ascii: "["): return try parseArray(depth: depth)
        case UInt8(ascii: "\""): return .string(try parseString())
        case UInt8(ascii: "t"): try expectLiteral("true"); return .bool(true)
        case UInt8(ascii: "f"): try expectLiteral("false"); return .bool(false)
        case UInt8(ascii: "n"): try expectLiteral("null"); return .null
        case UInt8(ascii: "N"): try expectLiteral("NaN"); return .double(.nan)
        case UInt8(ascii: "I"): try expectLiteral("Infinity"); return .double(.infinity)
        default: return try parseNumber()
        }
    }

    private mutating func expectLiteral(_ literal: String) throws {
        let literalBytes = Array(literal.utf8)
        guard index + literalBytes.count <= bytes.count,
              Array(bytes[index..<index + literalBytes.count]) == literalBytes else {
            throw error("Expecting value")
        }
        index += literalBytes.count
    }

    private mutating func parseObject(depth: Int) throws -> JSONValue {
        index += 1
        var object = JSONObject()
        skipWhitespace()
        if index < bytes.count, bytes[index] == UInt8(ascii: "}") {
            index += 1
            return .object(object)
        }
        while true {
            skipWhitespace()
            guard index < bytes.count, bytes[index] == UInt8(ascii: "\"") else {
                throw error("Expecting property name enclosed in double quotes")
            }
            let key = try parseString()
            skipWhitespace()
            guard index < bytes.count, bytes[index] == UInt8(ascii: ":") else {
                throw error("Expecting ':' delimiter")
            }
            index += 1
            skipWhitespace()
            object[key] = try parseValue(depth: depth + 1)
            skipWhitespace()
            guard index < bytes.count else { throw error("Expecting ',' delimiter") }
            if bytes[index] == UInt8(ascii: ",") {
                index += 1
                continue
            }
            if bytes[index] == UInt8(ascii: "}") {
                index += 1
                return .object(object)
            }
            throw error("Expecting ',' delimiter")
        }
    }

    private mutating func parseArray(depth: Int) throws -> JSONValue {
        index += 1
        var array: [JSONValue] = []
        skipWhitespace()
        if index < bytes.count, bytes[index] == UInt8(ascii: "]") {
            index += 1
            return .array(array)
        }
        while true {
            skipWhitespace()
            array.append(try parseValue(depth: depth + 1))
            skipWhitespace()
            guard index < bytes.count else { throw error("Expecting ',' delimiter") }
            if bytes[index] == UInt8(ascii: ",") {
                index += 1
                continue
            }
            if bytes[index] == UInt8(ascii: "]") {
                index += 1
                return .array(array)
            }
            throw error("Expecting ',' delimiter")
        }
    }

    private mutating func parseHex4() throws -> UInt32 {
        guard index + 4 <= bytes.count,
              let value = UInt32(String(decoding: bytes[index..<index + 4], as: UTF8.self), radix: 16) else {
            throw error("Invalid \\uXXXX escape")
        }
        index += 4
        return value
    }

    private mutating func parseString() throws -> String {
        index += 1
        var runStart = index
        var result = ""
        func flushRun(_ end: Int) {
            if end > runStart {
                result += String(decoding: bytes[runStart..<end], as: UTF8.self)
            }
        }
        while index < bytes.count {
            let byte = bytes[index]
            if byte == UInt8(ascii: "\"") {
                flushRun(index)
                index += 1
                return result
            }
            if byte == UInt8(ascii: "\\") {
                flushRun(index)
                index += 1
                guard index < bytes.count else { throw error("Unterminated string") }
                let escape = bytes[index]
                index += 1
                switch escape {
                case UInt8(ascii: "\""): result += "\""
                case UInt8(ascii: "\\"): result += "\\"
                case UInt8(ascii: "/"): result += "/"
                case UInt8(ascii: "b"): result += "\u{08}"
                case UInt8(ascii: "f"): result += "\u{0C}"
                case UInt8(ascii: "n"): result += "\n"
                case UInt8(ascii: "r"): result += "\r"
                case UInt8(ascii: "t"): result += "\t"
                case UInt8(ascii: "u"):
                    var code = try parseHex4()
                    if (0xD800...0xDBFF).contains(code), index + 6 <= bytes.count,
                       bytes[index] == UInt8(ascii: "\\"), bytes[index + 1] == UInt8(ascii: "u") {
                        let save = index
                        index += 2
                        let low = try parseHex4()
                        if (0xDC00...0xDFFF).contains(low) {
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        } else {
                            index = save
                        }
                    }
                    result.unicodeScalars.append(Unicode.Scalar(code) ?? "\u{FFFD}")
                default:
                    throw error("Invalid \\escape")
                }
                runStart = index
                continue
            }
            if byte < 0x20 {
                throw error("Invalid control character")
            }
            index += 1
        }
        throw error("Unterminated string")
    }

    private mutating func parseNumber() throws -> JSONValue {
        let start = index
        var isDouble = false
        if index < bytes.count, bytes[index] == UInt8(ascii: "-") {
            index += 1
            if index < bytes.count, bytes[index] == UInt8(ascii: "I") {
                try expectLiteral("Infinity")
                return .double(-.infinity)
            }
        }
        while index < bytes.count {
            let byte = bytes[index]
            if (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte) {
                index += 1
            } else if byte == UInt8(ascii: ".") || byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E")
                        || byte == UInt8(ascii: "+") || (byte == UInt8(ascii: "-") && index > start) {
                isDouble = true
                index += 1
            } else {
                break
            }
        }
        let text = String(decoding: bytes[start..<index], as: UTF8.self)
        guard !text.isEmpty, text != "-" else { throw error("Expecting value") }
        if !isDouble, let int = Int(text) {
            return .int(int)
        }
        guard let double = Double(text) else { throw error("Expecting value") }
        return .double(double)
    }
}
