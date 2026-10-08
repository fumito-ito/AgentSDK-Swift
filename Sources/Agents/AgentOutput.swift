import Foundation

private let wrapperDictKey = "response"

/// Captures the JSON schema of the output, and validates/parses JSON produced by the LLM into the
/// output type.
public protocol AgentOutputSchemaBase: Sendable {
    /// Whether the output type is plain text (versus a JSON object).
    func isPlainText() -> Bool
    /// The name of the output type.
    func name() -> String
    /// The JSON schema of the output. Only called when the output type is not plain text.
    func jsonSchema() throws -> JSONObject
    /// Whether the JSON schema is in strict mode.
    func isStrictJSONSchema() -> Bool
    /// Validates a JSON string against the output type and returns the validated object, or throws
    /// `ModelBehaviorError` when the JSON is invalid.
    func validateJSON(_ jsonString: String) throws -> any Sendable
}

/// The default `AgentOutputSchemaBase` for a `Decodable` output type.
///
/// Swift: upstream wraps types that are not JSON objects (for example `list[str]`) in
/// `{"response": ...}`. Here, types whose generated schema is not an object are wrapped the same
/// way.
public struct AgentOutputSchema<Output: Decodable & Sendable>: AgentOutputSchemaBase {
    /// Whether the output type is wrapped in `{"response": ...}`.
    public let isWrapped: Bool
    private let outputSchema: JSONObject
    private let strict: Bool

    public init(_ outputType: Output.Type = Output.self, strictJSONSchema: Bool = true) throws {
        self.strict = strictJSONSchema
        var schema = try JSONSchemaGenerator.schema(for: outputType)
        let isObject = schema["type"] == "object"
        isWrapped = !isObject
        if isWrapped {
            schema = [
                "type": "object",
                "properties": [wrapperDictKey: .object(schema)],
                "required": [.string(wrapperDictKey)],
                "additionalProperties": false,
            ]
        }
        if strictJSONSchema {
            do {
                schema = try ensureStrictJSONSchema(schema)
            } catch is UserError {
                throw UserError(
                    "Strict JSON schema is enabled, but the output type is not valid. Either make the output type strict, "
                        + "or wrap your type with AgentOutputSchema(YourType, strict_json_schema=False)"
                )
            }
        }
        outputSchema = schema
    }

    public func isPlainText() -> Bool { Output.self == String.self }

    public func isStrictJSONSchema() -> Bool { strict }

    public func jsonSchema() throws -> JSONObject {
        if isPlainText() {
            throw UserError("Output type is plain text, so no JSON schema is available")
        }
        return outputSchema
    }

    public func validateJSON(_ jsonString: String) throws -> any Sendable {
        let parsed: JSONValue
        do {
            parsed = try JSONValue.parse(jsonString)
        } catch {
            throw invalidJSON(jsonString, error)
        }
        var payload = parsed
        if isWrapped {
            guard case .object(let object) = parsed else {
                attachErrorToCurrentSpan(SpanError(message: "Invalid JSON", data: ["details": .string("Expected a dict, got \(parsed.typeName)")]))
                throw ModelBehaviorError("Expected a dict, got \(parsed.typeName) for JSON: \(jsonString)")
            }
            guard let wrapped = object[wrapperDictKey] else {
                attachErrorToCurrentSpan(SpanError(message: "Invalid JSON", data: ["details": .string("Could not find key \(wrapperDictKey) in JSON")]))
                throw ModelBehaviorError("Could not find key \(wrapperDictKey) in JSON: \(jsonString)")
            }
            payload = wrapped
        }
        do {
            return try payload.decode(as: Output.self)
        } catch {
            throw invalidJSON(jsonString, error)
        }
    }

    private func invalidJSON(_ jsonString: String, _ error: any Error) -> ModelBehaviorError {
        attachErrorToCurrentSpan(SpanError(message: "Invalid JSON provided", data: [:]))
        if AgentsDebug.dontLogModelData {
            return ModelBehaviorError("Invalid JSON when parsing model output")
        }
        return ModelBehaviorError("Invalid JSON when parsing \(jsonString) for \(name()); \(error)")
    }

    public func name() -> String { String(describing: Output.self) }
}

/// An output schema backed by a raw JSON schema, returning the parsed `JSONValue`.
///
/// Use this when the output shape is only known as JSON schema (for example from configuration).
public struct JSONSchemaOutputSchema: AgentOutputSchemaBase {
    private let schema: JSONObject
    private let outputName: String
    private let strict: Bool

    public init(name: String, jsonSchema: JSONObject, strictJSONSchema: Bool = true) throws {
        self.outputName = name
        self.strict = strictJSONSchema
        self.schema = strictJSONSchema ? try ensureStrictJSONSchema(jsonSchema) : jsonSchema
    }

    public func isPlainText() -> Bool { false }
    public func name() -> String { outputName }
    public func jsonSchema() throws -> JSONObject { schema }
    public func isStrictJSONSchema() -> Bool { strict }

    public func validateJSON(_ jsonString: String) throws -> any Sendable {
        do {
            return try JSONValue.parse(jsonString)
        } catch {
            attachErrorToCurrentSpan(SpanError(message: "Invalid JSON provided", data: [:]))
            if AgentsDebug.dontLogModelData {
                throw ModelBehaviorError("Invalid JSON when parsing model output")
            }
            throw ModelBehaviorError("Invalid JSON when parsing \(jsonString) for \(outputName); \(error)")
        }
    }
}

/// The plain-text output schema, used when an agent has no output type.
public struct PlainTextOutputSchema: AgentOutputSchemaBase {
    public init() {}
    public func isPlainText() -> Bool { true }
    public func name() -> String { "str" }
    public func jsonSchema() throws -> JSONObject {
        throw UserError("Output type is plain text, so no JSON schema is available")
    }
    public func isStrictJSONSchema() -> Bool { true }
    public func validateJSON(_ jsonString: String) throws -> any Sendable { jsonString }
}
