import Foundation

/// Swift equivalent of Python's built-in `TypeError` where upstream raises it.
public struct TypeError: Error, CustomStringConvertible, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}

/// Swift equivalent of Python's built-in `ValueError` where upstream raises it.
public struct ValueError: Error, CustomStringConvertible, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}

private let emptyStrictSchema: JSONObject = [
    "additionalProperties": false,
    "type": "object",
    "properties": [:],
    "required": [],
]

/// Upper bound on how many schema nodes strict conversion will expand. The limit only trips on
/// pathological input such as a `$ref` fan-out that would otherwise expand exponentially.
let maxStrictSchemaNodes = 100_000

/// Keeps recursive copying and normalization well below the recursion limit.
let maxStrictSchemaDepth = 100

private let additionalPropertiesError =
    "additionalProperties should not be set for object types. This could be because "
    + "you're using an older version of Pydantic, or because you configured additional "
    + "properties to be allowed. If you really need this, update the function or output tool "
    + "to not use a strict schema."

private let openObjectError =
    "JSON schema contains an object that permits undeclared properties and cannot be converted "
    + "to a strict schema without changing its accepted values."

private let unvalidatedRefError =
    "JSON schema contains a reference whose target was not validated for strict mode."

private let nestedResourceRefError =
    "JSON schema contains a reference owned by or crossing a nested `$id` resource that cannot "
    + "be resolved against the document root."

private let schemaDepthError =
    "JSON schema is too deeply nested to process safely. Simplify or flatten the schema."

/// Keywords that may legally sit alongside a `$ref` without adding validation constraints.
private let refNonConstrainingSiblings: Set<String> = [
    "$anchor", "$comment", "$defs", "$schema", "contentEncoding", "contentMediaType",
    "contentSchema", "default", "definitions", "deprecated", "description", "examples",
    "readOnly", "title", "writeOnly",
]
private let rootRefNonConstrainingSiblings = refNonConstrainingSiblings.union(["$id"])

/// Rejects container nesting that is unsafe for recursive schema processing.
/// Returns the maximum container depth of `schema`.
@discardableResult
func validateJSONSchemaDepth(_ schema: JSONValue) throws -> Int {
    var maxDepth = 0
    var stack: [(JSONValue, Int)] = [(schema, 1)]
    while let (value, depth) = stack.popLast() {
        if depth > maxStrictSchemaDepth {
            throw UserError(schemaDepthError)
        }
        maxDepth = max(maxDepth, depth)
        switch value {
        case .object(let object):
            for child in object.values {
                if case .object = child { stack.append((child, depth + 1)) }
                if case .array = child { stack.append((child, depth + 1)) }
            }
        case .array(let array):
            for child in array {
                if case .object = child { stack.append((child, depth + 1)) }
                if case .array = child { stack.append((child, depth + 1)) }
            }
        default:
            break
        }
    }
    return maxDepth
}

/// Schemas nested deeper than this are converted on a large-stack thread (see `runOnLargeStack`).
private let inlineConversionDepth = 12

/// Converts the given JSON schema so it conforms to the `strict` standard that the OpenAI API
/// expects.
///
/// Swift: upstream mutates the schema in place and returns it. Swift values are copied, so the
/// converted schema is returned and in-place effects on the document root are replayed through a
/// mutable root (see `StrictSchemaConverter`).
public func ensureStrictJSONSchema(_ schema: JSONObject) throws -> JSONObject {
    try ensureStrictJSONSchema(schema, rejectOpenObjects: false, maxSchemaNodes: maxStrictSchemaNodes)
}

func ensureStrictJSONSchema(
    _ schema: JSONObject,
    rejectOpenObjects: Bool,
    maxSchemaNodes: Int = maxStrictSchemaNodes
) throws -> JSONObject {
    let depth = try validateJSONSchemaDepth(.object(schema))
    if schema.isEmpty {
        return emptyStrictSchema
    }
    // `$ref` chains also recurse, so any schema with definitions takes the large-stack path.
    let hasDefinitions = schema["$defs"] != nil || schema["definitions"] != nil
    func convert() throws -> JSONObject {
        let converter = StrictSchemaConverter(root: schema, limit: maxSchemaNodes, rejectOpenObjects: rejectOpenObjects)
        let result = try converter.convert(.object(schema), path: [], depth: 1, insideNestedResource: false)
        return try ensureStrictRoot(result)
    }
    if depth <= inlineConversionDepth && !hasDefinitions {
        return try convert()
    }
    return try runOnLargeStack(convert)
}

private func ensureStrictRoot(_ schema: JSONObject) throws -> JSONObject {
    var schema = schema
    if case .array? = schema["anyOf"] {
        throw UserError("The root of a strict JSON schema must not use `anyOf`.")
    }
    if case .array(let types)? = schema["type"], types.contains("object") {
        if types == ["object"] {
            schema["type"] = "object"
        } else {
            throw UserError(
                "The root of a strict JSON schema must be a non-nullable object, but its type is "
                    + "\(pythonRepr(.array(types))). Make the root a plain object, or update the function or output tool to "
                    + "not use a strict schema."
            )
        }
    }
    return schema
}

private enum SchemaPathComponent {
    case key(String)
    case index(Int)
}

/// Replays upstream's in-place conversion over Swift values. Every converted node is written back
/// into `root` at its path, so later `$ref` resolution observes already-converted definitions
/// exactly as upstream does when it mutates the shared document.
private final class StrictSchemaConverter {
    var root: JSONValue
    var remaining: Int
    let rejectOpenObjects: Bool

    init(root: JSONObject, limit: Int, rejectOpenObjects: Bool) {
        self.root = .object(root)
        self.remaining = limit
        self.rejectOpenObjects = rejectOpenObjects
    }

    func spend() throws {
        remaining -= 1
        if remaining < 0 {
            throw UserError(
                "JSON schema is too large to convert to a strict schema. This can happen when a "
                    + "schema expands `$ref`s exponentially, which may indicate a malformed or malicious "
                    + "schema."
            )
        }
    }

    // Mirrors upstream `_ensure_strict_json_schema` branch for branch.
    // swiftlint:disable:next cyclomatic_complexity function_body_length
    func convert(
        _ value: JSONValue,
        path: [SchemaPathComponent],
        depth: Int,
        insideNestedResource: Bool
    ) throws -> JSONObject {
        if depth > maxStrictSchemaDepth {
            throw UserError(schemaDepthError)
        }
        guard case .object(var schema) = value else {
            throw TypeError("Expected \(pythonRepr(value)) to be a dictionary; path=\(pathDescription(path))")
        }
        let isRoot = path.isEmpty
        let explicitlyClosed = schema["additionalProperties"] == .bool(false)

        try spend()
        let nextDepth = depth + 1
        let insideNestedResource = insideNestedResource || (!isRoot && declaresSchemaResource(schema))

        if schema.contains("$ref") {
            if insideNestedResource {
                throw UserError(nestedResourceRefError)
            }
            let allowed = isRoot ? rootRefNonConstrainingSiblings : refNonConstrainingSiblings
            let incompatible = Set(schema.keys).subtracting(["$ref"]).subtracting(allowed)
            if !incompatible.isEmpty {
                throw UserError(
                    "JSON schema contains a `$ref` with incompatible sibling keyword(s) "
                        + "(\(incompatible.sorted().joined(separator: ", "))) that cannot be merged "
                        + "without changing its accepted values."
                )
            }
        }

        for definitionsKey in ["$defs", "definitions"] {
            guard case .object(let definitions)? = schema[definitionsKey] else { continue }
            var converted = definitions
            for (name, definition) in definitions.pairs {
                let childPath = path + [.key(definitionsKey), .key(name)]
                let result = try convert(
                    definition, path: childPath, depth: nextDepth, insideNestedResource: insideNestedResource
                )
                converted[name] = .object(result)
            }
            schema[definitionsKey] = .object(converted)
            writeBack(schema, at: path)
        }

        var type = schema["type"]
        let properties = schema["properties"]
        if type == nil, case .object? = properties {
            type = "object"
            schema["type"] = "object"
        } else if type == nil, let additional = schema["additionalProperties"], additional != .bool(false) {
            throw UserError(additionalPropertiesError)
        }
        let isObject: Bool
        switch type {
        case .string("object")?: isObject = true
        case .array(let types)?: isObject = types.contains("object")
        default: isObject = false
        }
        let hasNoDeclaredProperties = !schema.contains("properties") || properties == .object([:])
        if rejectOpenObjects, isObject, hasNoDeclaredProperties, schema["additionalProperties"] != .bool(false) {
            throw UserError(openObjectError)
        }
        if isObject, !schema.contains("additionalProperties") {
            schema["additionalProperties"] = false
        } else if isObject, let additional = schema["additionalProperties"], additional != .bool(false) {
            throw UserError(additionalPropertiesError)
        }

        // Object types.
        if case .object(let propertyMap)? = properties {
            schema["required"] = .array(propertyMap.keys.map(JSONValue.string))
            var converted = JSONObject()
            for (key, propertySchema) in propertyMap.pairs {
                converted[key] = .object(try convert(
                    propertySchema,
                    path: path + [.key("properties"), .key(key)],
                    depth: nextDepth,
                    insideNestedResource: insideNestedResource
                ))
            }
            schema["properties"] = .object(converted)
            writeBack(schema, at: path)
        }

        // Arrays.
        if case .object? = schema["items"], let items = schema["items"] {
            schema["items"] = .object(try convert(
                items, path: path + [.key("items")], depth: nextDepth, insideNestedResource: insideNestedResource
            ))
            writeBack(schema, at: path)
        }

        // Unions.
        if case .array(let anyOf)? = schema["anyOf"] {
            var converted: [JSONValue] = []
            for (index, variant) in anyOf.enumerated() {
                converted.append(.object(try convert(
                    variant,
                    path: path + [.key("anyOf"), .index(index)],
                    depth: nextDepth,
                    insideNestedResource: insideNestedResource
                )))
            }
            schema["anyOf"] = .array(converted)
            writeBack(schema, at: path)
        }

        // oneOf is not supported by structured outputs in nested contexts, so it becomes anyOf.
        if case .array(let oneOf)? = schema["oneOf"] {
            var existingAnyOf: [JSONValue] = []
            if case .array(let existing)? = schema["anyOf"] {
                existingAnyOf = existing
            }
            var converted: [JSONValue] = []
            for (index, variant) in oneOf.enumerated() {
                converted.append(.object(try convert(
                    variant,
                    path: path + [.key("oneOf"), .index(index)],
                    depth: nextDepth,
                    insideNestedResource: insideNestedResource
                )))
            }
            schema["anyOf"] = .array(existingAnyOf + converted)
            schema.removeValue(forKey: "oneOf")
            writeBack(schema, at: path)
        }

        // Intersections.
        if case .array(let allOf)? = schema["allOf"] {
            if allOf.count == 1 {
                let entry = allOf[0]
                let strictEntry: JSONObject
                if case .object(let entryObject) = entry, entryObject.contains("$ref"),
                   Set(entryObject.keys).subtracting(["$ref"]).subtracting(refNonConstrainingSiblings).isEmpty {
                    if insideNestedResource {
                        throw UserError(nestedResourceRefError)
                    }
                    try spend()
                    strictEntry = try resolveNonConstrainingRefChain(entryObject)
                } else {
                    strictEntry = try convert(
                        entry,
                        path: path + [.key("allOf"), .index(0)],
                        depth: nextDepth,
                        insideNestedResource: insideNestedResource
                    )
                }
                schema.removeValue(forKey: "allOf")
                if !explicitlyClosed {
                    // Close the final merged object below, after any remaining allOf is expanded.
                    schema.removeValue(forKey: "additionalProperties")
                }
                let merged = try mergeSingleAllOf(entry: strictEntry, parent: schema)
                writeBack(merged, at: path)
                let result = try convert(
                    .object(merged), path: path, depth: nextDepth, insideNestedResource: insideNestedResource
                )
                writeBack(result, at: path)
                return result
            } else {
                var converted: [JSONValue] = []
                for (index, entry) in allOf.enumerated() {
                    converted.append(.object(try convert(
                        entry,
                        path: path + [.key("allOf"), .index(index)],
                        depth: nextDepth,
                        insideNestedResource: insideNestedResource
                    )))
                }
                schema["allOf"] = .array(converted)
                writeBack(schema, at: path)
            }
        }

        // Strip `None` defaults as there's no meaningful distinction here.
        if schema["default"] == .null {
            schema.removeValue(forKey: "default")
        }

        // `$ref`s cannot carry other properties, e.g. `{"$ref": "...", "description": "..."}`, so
        // the reference is unraveled into `{"type": "string", "description": "..."}`.
        if let ref = schema["$ref"], ref.isTruthy, schema.count > 1 {
            guard case .string(let refString) = ref else {
                throw ValueError("Received non-string $ref - \(ref)")
            }
            let resolved = try resolveRef(root: root, ref: refString)
            guard case .object(let resolvedObject) = resolved else {
                throw ValueError("Expected `$ref: \(refString)` to resolve to a dictionary but got \(resolved)")
            }
            // Pop the current `$ref` first so chained refs survive for the expansion below.
            schema.removeValue(forKey: "$ref")
            // Properties from the schema take priority over the ones on the `$ref`.
            var expanded = schema
            for (key, value) in resolvedObject.merging(schema).pairs {
                expanded[key] = value
            }
            writeBack(expanded, at: path)
            let result = try convert(
                .object(expanded), path: path, depth: nextDepth, insideNestedResource: insideNestedResource
            )
            writeBack(result, at: path)
            return result
        }

        if rejectOpenObjects, schema.contains("$ref") {
            throw UserError(unvalidatedRefError)
        }

        writeBack(schema, at: path)
        return schema
    }

    func resolveNonConstrainingRefChain(_ schema: JSONObject) throws -> JSONObject {
        var resolved = schema
        var seenRefs: Set<String> = []
        var carriedSiblings = JSONObject()
        while true {
            guard resolved.contains("$ref") else { break }
            if !Set(resolved.keys).subtracting(["$ref"]).subtracting(refNonConstrainingSiblings).isEmpty {
                break
            }
            guard case .string(let ref)? = resolved["$ref"] else {
                throw ValueError("Received non-string $ref - \(resolved["$ref"] ?? .null)")
            }
            if seenRefs.contains(ref) {
                throw UserError("JSON schema contains a circular `$ref` chain.")
            }
            seenRefs.insert(ref)
            carriedSiblings = resolved.filter { key, _ in key != "$ref" }.merging(carriedSiblings)
            try spend()
            let target = try resolveRef(root: root, ref: ref)
            guard case .object(let targetObject) = target else {
                throw ValueError("Expected `$ref: \(ref)` to resolve to a dictionary but got \(target)")
            }
            resolved = targetObject
        }
        return resolved.merging(carriedSiblings)
    }

    func writeBack(_ schema: JSONObject, at path: [SchemaPathComponent]) {
        root = Self.setting(.object(schema), in: root, at: path[...]) ?? root
    }

    private static func setting(_ value: JSONValue, in container: JSONValue, at path: ArraySlice<SchemaPathComponent>) -> JSONValue? {
        guard let first = path.first else { return value }
        let rest = path.dropFirst()
        switch (first, container) {
        case (.key(let key), .object(var object)):
            guard let child = object[key], let updated = setting(value, in: child, at: rest) else { return nil }
            object[key] = updated
            return .object(object)
        case (.index(let index), .array(var array)):
            guard array.indices.contains(index), let updated = setting(value, in: array[index], at: rest) else { return nil }
            array[index] = updated
            return .array(array)
        default:
            return nil
        }
    }
}

/// Resolves a local JSON pointer `$ref` against the document root.
func resolveRef(root: JSONValue, ref: String) throws -> JSONValue {
    guard ref.hasPrefix("#/") else {
        throw ValueError("Unexpected $ref format '\(ref)'; Does not start with #/")
    }
    guard let decoded = String(ref.dropFirst(2)).removingPercentEncoding else {
        throw ValueError("Invalid percent-encoding in $ref '\(ref)'")
    }
    var resolved = root
    for rawKey in decoded.split(separator: "/", omittingEmptySubsequences: false) {
        let key = rawKey.replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")
        guard case .object(let object) = resolved, let value = object[key] else {
            throw ValueError("Unresolvable $ref \(ref)")
        }
        guard case .object(let valueObject) = value else {
            throw ValueError("encountered non-dictionary entry while resolving \(ref) - \(resolved)")
        }
        resolved = value
        if declaresSchemaResource(valueObject) {
            throw UserError(nestedResourceRefError)
        }
    }
    return resolved
}

private func declaresSchemaResource(_ schema: JSONObject) -> Bool {
    if case .string? = schema["$id"] { return true }
    return false
}

private func mergeSingleAllOf(entry: JSONObject, parent: JSONObject) throws -> JSONObject {
    var merged = entry
    var incompatible: [String] = []
    if parent["additionalProperties"] == .bool(false),
       !parent.contains("properties") || parent["properties"] == .object([:]),
       case .object(let entryProperties)? = entry["properties"], !entryProperties.isEmpty {
        // additionalProperties only sees properties declared in its own schema, not allOf.
        incompatible.append("properties")
    }
    for (key, parentValue) in parent.pairs {
        if !merged.contains(key) {
            merged[key] = parentValue
        } else if refNonConstrainingSiblings.contains(key) {
            merged[key] = parentValue
        } else if jsonValuesEqual(parentValue, merged[key]!) {
            continue
        } else if (key == "properties" && parentValue == .object([:])) || (key == "required" && parentValue == .array([])) {
            continue
        } else {
            incompatible.append(key)
        }
    }
    if !incompatible.isEmpty {
        throw UserError(
            "JSON schema contains a singleton `allOf` entry with incompatible parent "
                + "keyword(s) (\(incompatible.sorted().joined(separator: ", "))) that cannot be merged "
                + "without changing its accepted values."
        )
    }
    return merged
}

/// JSON equality that keeps booleans distinct from numbers, as upstream does.
private func jsonValuesEqual(_ left: JSONValue, _ right: JSONValue) -> Bool {
    switch (left, right) {
    case let (.array(lhs), .array(rhs)):
        return lhs.count == rhs.count && zip(lhs, rhs).allSatisfy(jsonValuesEqual)
    case let (.object(lhs), .object(rhs)):
        return Set(lhs.keys) == Set(rhs.keys) && lhs.keys.allSatisfy { jsonValuesEqual(lhs[$0]!, rhs[$0]!) }
    default:
        return left == right
    }
}

private func pathDescription(_ path: [SchemaPathComponent]) -> String {
    let parts = path.map { component -> String in
        switch component {
        case .key(let key): return "'\(key)'"
        case .index(let index): return "'\(index)'"
        }
    }
    return "(" + parts.joined(separator: ", ") + (parts.count == 1 ? ",)" : ")")
}

/// Renders a JSON value like Python's `repr` for error messages.
func pythonRepr(_ value: JSONValue) -> String {
    switch value {
    case .null: return "None"
    case .bool(let bool): return bool ? "True" : "False"
    case .int(let int): return String(int)
    case .double(let double): return JSONValue.formatDouble(double)
    case .string(let string): return "'\(string)'"
    case .array(let array): return "[" + array.map(pythonRepr).joined(separator: ", ") + "]"
    case .object(let object):
        return "{" + object.pairs.map { "'\($0.key)': \(pythonRepr($0.value))" }.joined(separator: ", ") + "}"
    }
}
