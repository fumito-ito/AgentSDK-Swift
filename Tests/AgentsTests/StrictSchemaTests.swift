import Foundation
import Testing
@testable import Agents

// Ported from tests/test_strict_schema.py and tests/test_strict_schema_oneof.py.
@Suite struct StrictSchemaTests {
    private func nestedObjectSchema(depth: Int) -> JSONObject {
        var current: JSONObject = ["type": "object", "properties": [:]]
        for _ in 0..<depth {
            current = ["type": "object", "properties": ["child": .object(current)]]
        }
        return current
    }

    private func chainedRefSchema(depth: Int) -> JSONObject {
        var definitions = JSONObject()
        for index in 0..<depth {
            definitions["L\(index)"] = ["$ref": .string("#/$defs/L\(index + 1)")]
        }
        definitions["L\(depth)"] = ["type": "string"]
        return [
            "$defs": .object(definitions),
            "type": "object",
            "properties": ["value": ["$ref": "#/$defs/L0", "description": "value"]],
        ]
    }

    private func node(_ value: JSONObject, _ path: String...) -> JSONValue? {
        var current: JSONValue? = .object(value)
        for key in path {
            if let index = Int(key) {
                current = current?[index]
            } else {
                current = current?[key]
            }
        }
        return current
    }

    // MARK: - Empty and invalid schemas

    @Test func testEmptySchemaHasAdditionalPropertiesFalse() throws {
        let strictSchema = try ensureStrictJSONSchema([:])
        #expect(strictSchema["additionalProperties"] == false)
    }

    @Test func testEmptySchemaReturnsFreshCopy() throws {
        var first = try ensureStrictJSONSchema([:])
        first["additionalProperties"] = true
        let second = try ensureStrictJSONSchema([:])
        #expect(second == [
            "additionalProperties": false,
            "type": "object",
            "properties": [:],
            "required": [],
        ])
    }

    @Test func testNonDictSchemaErrors() {
        // Swift: the root parameter is typed as `JSONObject`; nested non-dictionary nodes still
        // raise `TypeError`.
        expectError(TypeError.self) {
            _ = try ensureStrictJSONSchema(["type": "object", "properties": ["a": 1]])
        }
    }

    @Test func testDeeplyNestedSchemaIsRejectedBeforeRecursiveConversion() {
        // Swift: building and releasing a 1,000-level JSON value recurses, so the fixture lives
        // on a large stack (porting/decisions.md D9).
        runOnLargeStack {
            expectError(UserError.self, containing: "too deeply nested") {
                _ = try ensureStrictJSONSchema(nestedObjectSchema(depth: 1_000))
            }
        }
    }

    @Test func testReasonablyNestedSchemaRemainsSupported() throws {
        let result = try ensureStrictJSONSchema(nestedObjectSchema(depth: 10))
        #expect(result["additionalProperties"] == false)
    }

    @Test func testDeeplyChainedRefsAreRejectedBeforeRecursiveConversion() {
        expectError(UserError.self, containing: "too deeply nested") {
            _ = try ensureStrictJSONSchema(chainedRefSchema(depth: 1_000))
        }
    }

    // MARK: - Objects

    @Test func testObjectWithoutAdditionalProperties() throws {
        let result = try ensureStrictJSONSchema(["type": "object", "properties": ["a": ["type": "string"]]])
        #expect(result["type"] == "object")
        #expect(result["additionalProperties"] == false)
        #expect(result["required"] == ["a"])
        #expect(node(result, "properties", "a") == ["type": "string"])
    }

    @Test func testOpenObjectRejectionIsOptIn() throws {
        let schema: JSONObject = ["type": "object", "properties": [:]]
        let result = try ensureStrictJSONSchema(schema)
        #expect(result["additionalProperties"] == false)
        expectError(UserError.self, containing: "permits undeclared properties") {
            _ = try ensureStrictJSONSchema(schema, rejectOpenObjects: true)
        }
    }

    @Test(arguments: [
        ("#/$defs/value", ["value": ["type": "string"]] as JSONObject),
        ("#/$defs/a%20b", ["a b": ["type": "string"]] as JSONObject),
    ])
    func testOpenObjectRejectionRejectsPreservedPureRefs(ref: String, definitions: JSONObject) throws {
        let schema: JSONObject = [
            "$defs": .object(definitions),
            "type": "object",
            "properties": ["value": ["$ref": .string(ref)]],
        ]
        let defaultResult = try ensureStrictJSONSchema(schema)
        #expect(node(defaultResult, "properties", "value") == ["$ref": .string(ref)])
        expectError(UserError.self, containing: "reference whose target was not validated") {
            _ = try ensureStrictJSONSchema(schema, rejectOpenObjects: true)
        }
    }

    @Test func testTypelessRootIsNormalizedToObject() throws {
        let result = try ensureStrictJSONSchema(["properties": ["a": ["type": "string"]]])
        #expect(result == [
            "type": "object",
            "properties": ["a": ["type": "string"]],
            "additionalProperties": false,
            "required": ["a"],
        ])
    }

    @Test func testNullableObjectRootErrors() {
        expectError(UserError.self, containing: "root of a strict JSON schema") {
            _ = try ensureStrictJSONSchema(["type": ["object", "null"], "properties": ["a": ["type": "string"]]])
        }
    }

    @Test func testOpenMapRootErrors() {
        expectError(UserError.self) {
            _ = try ensureStrictJSONSchema(["additionalProperties": ["type": "string"]])
        }
    }

    @Test func testNestedTypelessOpenMapErrors() {
        expectError(UserError.self) {
            _ = try ensureStrictJSONSchema([
                "type": "object",
                "properties": ["metadata": ["additionalProperties": ["type": "string"]]],
            ])
        }
    }

    @Test(arguments: ["anyOf", "oneOf"])
    func testUnionRootErrors(unionKeyword: String) {
        expectError(UserError.self, containing: "root of a strict JSON schema") {
            _ = try ensureStrictJSONSchema([
                unionKeyword: [["properties": ["a": ["type": "string"]]], ["type": "null"]],
            ])
        }
    }

    @Test func testNestedTypelessObjectsGetAdditionalProperties() throws {
        let cases: [(JSONObject, [String])] = [
            (["type": "object", "properties": ["config": ["properties": [:]]]], ["properties", "config"]),
            (["type": "array", "items": ["properties": [:]]], ["items"]),
            (
                ["type": "object", "properties": ["value": ["anyOf": [["properties": [:]], ["type": "null"]]]]],
                ["properties", "value", "anyOf", "0"]
            ),
        ]
        for (schema, path) in cases {
            let result = try ensureStrictJSONSchema(schema)
            var current: JSONValue? = .object(result)
            for key in path {
                current = Int(key).map { current?[$0] } ?? current?[key]
            }
            #expect(current?["type"] == "object")
            #expect(current?["additionalProperties"] == false)
        }
    }

    @Test func testNestedNullableObjectPreservesTypeUnion() throws {
        let result = try ensureStrictJSONSchema([
            "type": "object",
            "properties": [
                "config": ["type": ["object", "null"], "properties": ["key": ["type": "string"]]],
            ],
        ])
        #expect(node(result, "properties", "config") == [
            "type": ["object", "null"],
            "properties": ["key": ["type": "string"]],
            "additionalProperties": false,
            "required": ["key"],
        ])
    }

    @Test func testObjectWithTrueAdditionalProperties() {
        expectError(UserError.self) {
            _ = try ensureStrictJSONSchema([
                "type": "object", "properties": ["a": ["type": "number"]], "additionalProperties": true,
            ])
        }
    }

    @Test func testTypelessObjectWithAdditionalPropertiesErrors() {
        expectError(UserError.self) {
            _ = try ensureStrictJSONSchema(["properties": ["a": ["type": "number"]], "additionalProperties": true])
        }
    }

    @Test func testExplicitNonObjectWithPropertiesIsNotClosed() throws {
        let schema: JSONObject = ["type": "string", "properties": [:], "required": [], "additionalProperties": true]
        #expect(try ensureStrictJSONSchema(schema) == schema)
    }

    @Test func testObjectWithEmptyDictAdditionalProperties() {
        expectError(UserError.self) {
            _ = try ensureStrictJSONSchema([
                "type": "object", "properties": ["a": ["type": "string"]], "additionalProperties": [:],
            ])
        }
    }

    @Test func testObjectWithSchemaAdditionalProperties() {
        expectError(UserError.self) {
            _ = try ensureStrictJSONSchema([
                "type": "object", "properties": ["a": ["type": "string"]],
                "additionalProperties": ["type": "string"],
            ])
        }
    }

    @Test func testObjectWithFalseAdditionalPropertiesIsAllowed() throws {
        let result = try ensureStrictJSONSchema([
            "type": "object", "properties": ["a": ["type": "string"]], "additionalProperties": false,
        ])
        #expect(result["additionalProperties"] == false)
        #expect(result["required"] == ["a"])
    }

    @Test func testRequiredFollowsPropertyOrder() throws {
        // Swift: guards the ordered JSONObject that upstream gets from Python dicts.
        let result = try ensureStrictJSONSchema([
            "type": "object",
            "properties": ["zeta": ["type": "string"], "alpha": ["type": "string"], "mid": ["type": "string"]],
        ])
        #expect(result["required"] == ["zeta", "alpha", "mid"])
    }

    // MARK: - Arrays, unions, and defaults

    @Test func testArrayItemsProcessingAndDefaultRemoval() throws {
        let result = try ensureStrictJSONSchema(["type": "array", "items": ["type": "number", "default": nil]])
        #expect(node(result, "items", "default") == nil)
        #expect(node(result, "items", "type") == "number")
    }

    @Test func testAnyOfProcessing() throws {
        let result = try ensureStrictJSONSchema([
            "type": "object",
            "properties": [
                "value": [
                    "anyOf": [
                        ["type": "object", "properties": ["a": ["type": "string"]]],
                        ["type": "number", "default": nil],
                    ],
                ],
            ],
        ])
        let variant0 = node(result, "properties", "value", "anyOf", "0")
        #expect(variant0?["type"] == "object")
        #expect(variant0?["additionalProperties"] == false)
        #expect(variant0?["required"] == ["a"])
        let variant1 = node(result, "properties", "value", "anyOf", "1")
        #expect(variant1?["type"] == "number")
        #expect(variant1?["default"] == nil)
    }

    @Test func testDefaultRemovalOnNonObject() throws {
        let result = try ensureStrictJSONSchema(["type": "string", "default": nil])
        #expect(result["type"] == "string")
        #expect(result["default"] == nil)
    }

    // MARK: - allOf

    @Test func testAllOfSingleEntryMerging() throws {
        let result = try ensureStrictJSONSchema([
            "type": "object",
            "allOf": [["properties": ["a": ["type": "boolean"]]]],
        ])
        #expect(result["allOf"] == nil)
        #expect(result["additionalProperties"] == false)
        #expect(result["required"] == ["a"])
        #expect(node(result, "properties", "a", "type") == "boolean")
    }

    @Test func testAllOfSingleRefEntryMerging() throws {
        let result = try ensureStrictJSONSchema([
            "$defs": [
                "Inner": ["type": "object", "properties": ["b": ["type": "string"]], "required": ["b"]],
                "Outer": ["$ref": "#/$defs/Inner"],
            ],
            "type": "object",
            "allOf": [["$ref": "#/$defs/Outer"]],
        ])
        #expect(result["allOf"] == nil)
        #expect(result["$ref"] == nil)
        #expect(result["type"] == "object")
        #expect(result["properties"] == ["b": ["type": "string"]])
        #expect(result["required"] == ["b"])
        #expect(result["additionalProperties"] == false)
    }

    @Test func testAllOfSingleRefEntryPreservesAnnotatedAliases() throws {
        let result = try ensureStrictJSONSchema([
            "components": [
                "schemas": [
                    "Inner": ["type": "object", "description": "inner", "properties": ["value": ["type": "string"]]],
                    "Outer": ["$ref": "#/components/schemas/Inner", "description": "outer"],
                ],
            ],
            "type": "object",
            "allOf": [["$ref": "#/components/schemas/Outer", "title": "entry"]],
        ])
        #expect(result["$ref"] == nil)
        #expect(result["description"] == "outer")
        #expect(result["title"] == "entry")
        #expect(result["properties"] == ["value": ["type": "string"]])
        #expect(result["required"] == ["value"])
        #expect(result["additionalProperties"] == false)
    }

    @Test func testAllOfSingleRefEntryRejectsOverlappingParentConstraints() {
        expectError(UserError.self, containing: "singleton `allOf`") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["type": "object", "properties": ["inner": ["type": "string"]]]],
                "type": "object",
                "properties": ["outer": ["type": "string"]],
                "allOf": [["$ref": "#/$defs/T", "description": "alias"]],
            ])
        }
    }

    @Test func testNestedSingleAllOfRejectsOverlappingParentConstraints() {
        expectError(UserError.self, containing: "singleton `allOf`") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["type": "object", "properties": ["inner": ["type": "string"]]]],
                "type": "object",
                "properties": ["outer": ["type": "string"]],
                "allOf": [["allOf": [["$ref": "#/$defs/T"]]]],
            ])
        }
    }

    @Test(arguments: [
        (JSONValue.int(1), JSONValue.bool(true)),
        (["nested": [1]] as JSONValue, ["nested": [true]] as JSONValue),
    ])
    func testAllOfSingleRefEntryRejectsJSONDistinctEqualPythonValues(referencedValue: JSONValue, parentValue: JSONValue) {
        expectError(UserError.self, containing: "singleton `allOf`") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["const": referencedValue]],
                "const": parentValue,
                "allOf": [["$ref": "#/$defs/T"]],
            ])
        }
    }

    @Test func testAllOfSingleRefEntryAcceptsEqualJSONNumbers() throws {
        let result = try ensureStrictJSONSchema([
            "$defs": ["T": ["const": 1]],
            "const": 1.0,
            "allOf": [["$ref": "#/$defs/T"]],
        ])
        #expect(result["const"] == .int(1))
        if case .int? = result["const"] {} else { Issue.record("const should stay an integer") }
    }

    @Test func testAllOfSingleCircularRefIsRejected() {
        expectError(UserError.self, containing: "circular") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["A": ["$ref": "#/$defs/B"], "B": ["$ref": "#/$defs/A"]],
                "type": "object",
                "allOf": [["$ref": "#/$defs/A"]],
            ])
        }
    }

    @Test func testAllOfSingleAnnotatedCircularRefIsRejected() {
        expectError(UserError.self, containing: "circular") {
            _ = try ensureStrictJSONSchema([
                "components": [
                    "schemas": [
                        "A": ["$ref": "#/components/schemas/B", "description": "a"],
                        "B": ["$ref": "#/components/schemas/A", "description": "b"],
                    ],
                ],
                "type": "object",
                "allOf": [["$ref": "#/components/schemas/A"]],
            ])
        }
    }

    @Test func testAllOfSingleRefChainSpendsNodeBudget() {
        expectError(UserError.self, containing: "too large") {
            _ = try ensureStrictJSONSchema(
                [
                    "components": [
                        "schemas": [
                            "A": ["$ref": "#/components/schemas/B"],
                            "B": ["$ref": "#/components/schemas/C"],
                            "C": ["type": "object", "properties": [:]],
                        ],
                    ],
                    "type": "object",
                    "allOf": [["$ref": "#/components/schemas/A"]],
                ],
                rejectOpenObjects: false,
                maxSchemaNodes: 4
            )
        }
    }

    @Test func testAllOfSingleRefRejectsNestedIdBeforePromotingTarget() {
        expectError(UserError.self, containing: "nested `$id`") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["type": "string"]],
                "contentSchema": ["$id": "https://example.test/nested", "$ref": "#/$defs/T"],
                "type": "object",
                "allOf": [["$ref": "#/contentSchema"]],
            ])
        }
    }

    @Test func testAllOfSingleRefRejectsNestedIdOwnerBeforeResolution() {
        expectError(UserError.self, containing: "nested `$id` resource") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["type": "string"]],
                "type": "object",
                "properties": [
                    "node": [
                        "$id": "https://example.test/nested",
                        "$defs": ["T": ["type": "integer"]],
                        "allOf": [["$ref": "#/$defs/T"]],
                    ],
                ],
            ])
        }
    }

    @Test func testAllOfSingleRefRejectsDescendantNestedIdOwnerBeforeResolution() {
        expectError(UserError.self, containing: "nested `$id` resource") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["type": "string"]],
                "type": "object",
                "properties": [
                    "node": [
                        "$id": "https://example.test/nested",
                        "$defs": ["T": ["type": "integer"]],
                        "type": "object",
                        "properties": ["child": ["allOf": [["$ref": "#/$defs/T"]]]],
                    ],
                ],
            ])
        }
    }

    @Test func testAllOfSingleRefRejectsPromotedNestedIdWithDescendantRef() {
        expectError(UserError.self, containing: "nested `$id` resource") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["type": "string"]],
                "contentSchema": [
                    "$id": "https://example.test/nested",
                    "$defs": ["T": ["type": "integer"]],
                    "type": "object",
                    "properties": ["value": ["$ref": "#/$defs/T"]],
                ],
                "type": "object",
                "allOf": [["$ref": "#/contentSchema"]],
            ])
        }
    }

    @Test func testAllOfSingleRefRejectsTargetBelowNestedIdResource() {
        expectError(UserError.self, containing: "nested `$id` resource") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["type": "string"]],
                "contentSchema": [
                    "$id": "https://example.test/nested",
                    "$defs": ["T": ["type": "integer"]],
                    "target": ["$ref": "#/$defs/T"],
                ],
                "type": "object",
                "allOf": [["$ref": "#/contentSchema/target"]],
            ])
        }
    }

    @Test(arguments: [
        (["$defs": ["$id": ["type": "object", "properties": [:]]]] as JSONObject, "#/$defs/$id"),
        (
            ["components": ["schemas": ["$id": ["type": "object", "properties": [:]]]]] as JSONObject,
            "#/components/schemas/$id"
        ),
    ])
    func testAllOfSingleRefAllowsIdAsSchemaMapMemberName(container: JSONObject, ref: String) throws {
        var schema = container
        schema["type"] = "object"
        schema["allOf"] = [["$ref": .string(ref)]]
        let result = try ensureStrictJSONSchema(schema)
        #expect(result["type"] == "object")
        #expect(result["properties"] == .object([:]))
        #expect(result["additionalProperties"] == false)
    }

    @Test func testRefAllowsUnrelatedIdDefinitionName() throws {
        let result = try ensureStrictJSONSchema([
            "$defs": ["$id": ["type": "integer"], "T": ["type": "string"]],
            "type": "object",
            "properties": ["value": ["$ref": "#/$defs/T", "description": "value"]],
        ])
        #expect(node(result, "properties", "value") == ["type": "string", "description": "value"])
    }

    @Test(arguments: [("a/b", "a~1b"), ("a~b", "a~0b"), ("a~1b", "a~01b")])
    func testAllOfSingleRefEntryDecodesJSONPointerTokens(definitionName: String, refToken: String) throws {
        let result = try ensureStrictJSONSchema([
            "$defs": [definitionName: ["type": "object", "properties": ["value": ["type": "string"]]]],
            "type": "object",
            "allOf": [["$ref": .string("#/$defs/\(refToken)")]],
        ])
        #expect(result["properties"] == ["value": ["type": "string"]])
        #expect(result["required"] == ["value"])
        #expect(result["additionalProperties"] == false)
    }

    @Test(arguments: [JSONValue.bool(true), JSONValue.object([:])])
    func testAllOfSingleEntryCannotOverwriteStrictObject(additionalProperties: JSONValue) {
        expectError(UserError.self) {
            _ = try ensureStrictJSONSchema([
                "properties": ["a": ["type": "string"]],
                "allOf": [["additionalProperties": additionalProperties]],
            ])
        }
    }

    // MARK: - $ref expansion

    @Test func testRefExpansion() throws {
        let result = try ensureStrictJSONSchema([
            "definitions": ["refObj": ["type": "string", "default": nil]],
            "type": "object",
            "properties": ["a": ["$ref": "#/definitions/refObj", "description": "desc"]],
        ])
        let aSchema = node(result, "properties", "a")
        #expect(aSchema?["type"] == "string")
        #expect(aSchema?["description"] == "desc")
        #expect(aSchema?["default"] == nil)
    }

    @Test func testRefNoExpansionWhenAlone() throws {
        let result = try ensureStrictJSONSchema(["$ref": "#/definitions/refObj"])
        #expect(result == ["$ref": "#/definitions/refObj"])
    }

    @Test func testInvalidRefFormat() {
        expectError(ValueError.self) {
            _ = try ensureStrictJSONSchema([
                "type": "object", "properties": ["a": ["$ref": "invalid", "description": "desc"]],
            ])
        }
    }

    @Test func testChainedRefWithSiblingKeysIsResolved() throws {
        let result = try ensureStrictJSONSchema([
            "$defs": ["Inner": ["type": "string"], "Outer": ["$ref": "#/$defs/Inner"]],
            "type": "object",
            "properties": ["a": ["$ref": "#/$defs/Outer", "description": "desc"]],
        ])
        let aSchema = node(result, "properties", "a")
        #expect(aSchema?["type"] == "string")
        #expect(aSchema?["description"] == "desc")
        #expect(aSchema?["$ref"] == nil)
    }

    @Test func testRefExpansionBombIsRejected() {
        let depth = 30
        var defs = JSONObject()
        for index in 0..<depth {
            defs["L\(index)"] = [
                "type": "object",
                "properties": [
                    "a": ["$ref": .string("#/$defs/L\(index + 1)"), "title": "t"],
                    "b": ["$ref": .string("#/$defs/L\(index + 1)"), "title": "t"],
                ],
            ]
        }
        defs["L\(depth)"] = ["type": "string"]
        expectError(UserError.self) {
            _ = try ensureStrictJSONSchema([
                "$defs": .object(defs),
                "type": "object",
                "properties": ["root": ["$ref": "#/$defs/L0", "title": "t"]],
            ])
        }
    }

    @Test func testRefWithIncompatibleSiblingIsRejected() {
        expectError(UserError.self, containing: "incompatible sibling") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["type": "object", "properties": ["b": ["type": "string"]], "required": ["b"]]],
                "type": "object",
                "properties": [
                    "node": ["properties": ["a": ["type": "string"]], "required": ["a"], "$ref": "#/$defs/T"],
                ],
            ])
        }
    }

    @Test func testRefWithIncompatibleTypeSiblingIsRejected() {
        expectError(UserError.self, containing: "incompatible sibling") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["type": "object", "properties": ["b": ["type": "string"]], "required": ["b"]]],
                "type": "object",
                "properties": ["node": ["type": "string", "$ref": "#/$defs/T"]],
            ])
        }
    }

    @Test func testRefWithSingleAllOfSiblingIsRejected() {
        expectError(UserError.self, containing: "incompatible sibling") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["A": ["type": "string"], "B": ["type": "integer"]],
                "type": "object",
                "properties": ["node": ["$ref": "#/$defs/A", "allOf": [["$ref": "#/$defs/B"]]]],
            ])
        }
    }

    @Test func testRefWithValidationSiblingIsRejected() {
        expectError(UserError.self, containing: "incompatible sibling") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["type": "string"]],
                "type": "object",
                "properties": ["node": ["$ref": "#/$defs/T", "minLength": 1]],
            ])
        }
    }

    @Test func testRefWithInteractingObjectSiblingIsRejected() {
        expectError(UserError.self, containing: "incompatible sibling") {
            _ = try ensureStrictJSONSchema([
                "$defs": [
                    "T": [
                        "type": "object", "properties": ["b": ["type": "string"]], "required": ["b"],
                        "additionalProperties": false,
                    ],
                ],
                "type": "object",
                "properties": ["node": ["$ref": "#/$defs/T", "additionalProperties": false]],
            ])
        }
    }

    @Test func testRefWithAnnotationSiblingIsStillExpanded() throws {
        let result = try ensureStrictJSONSchema([
            "$defs": ["T": ["type": "object", "properties": ["b": ["type": "string"]], "required": ["b"]]],
            "type": "object",
            "properties": [
                "node": [
                    "contentMediaType": "application/json",
                    "description": "a node",
                    "title": "Node",
                    "$ref": "#/$defs/T",
                ],
            ],
        ])
        let nodeSchema = node(result, "properties", "node")
        #expect(nodeSchema?["type"] == "object")
        #expect(nodeSchema?["properties"] == ["b": ["type": "string"]])
        #expect(nodeSchema?["required"] == ["b"])
        #expect(nodeSchema?["description"] == "a node")
        #expect(nodeSchema?["title"] == "Node")
        #expect(nodeSchema?["contentMediaType"] == "application/json")
        #expect(nodeSchema?["$ref"] == nil)
    }

    @Test func testRefWithSchemaMetadataIsStillExpanded() throws {
        let result = try ensureStrictJSONSchema([
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": ["T": ["type": "object", "properties": ["value": ["type": "string"]]]],
            "$ref": "#/$defs/T",
        ])
        #expect(result["$schema"] == "https://json-schema.org/draft/2020-12/schema")
        #expect(result["type"] == "object")
        #expect(result["properties"] == ["value": ["type": "string"]])
        #expect(result["required"] == ["value"])
        #expect(result["additionalProperties"] == false)
        #expect(result["$ref"] == nil)
    }

    @Test func testRootRefWithIdIsStillExpanded() throws {
        let result = try ensureStrictJSONSchema([
            "$id": "https://example.test/root",
            "$defs": ["T": ["type": "object", "properties": ["value": ["type": "string"]]]],
            "$ref": "#/$defs/T",
        ])
        #expect(result["$id"] == "https://example.test/root")
        #expect(result["type"] == "object")
        #expect(result["properties"] == ["value": ["type": "string"]])
        #expect(result["required"] == ["value"])
        #expect(result["additionalProperties"] == false)
        #expect(result["$ref"] == nil)
    }

    @Test func testNestedRefWithIdIsRejectedBeforeResolution() {
        expectError(UserError.self, containing: "nested `$id` resource") {
            _ = try ensureStrictJSONSchema([
                "$defs": ["T": ["type": "string"]],
                "type": "object",
                "properties": [
                    "value": [
                        "$id": "https://example.test/nested",
                        "$defs": ["T": ["type": "integer"]],
                        "$ref": "#/$defs/T",
                    ],
                ],
            ])
        }
    }

    @Test func testRefWithAnchorIsStillExpanded() throws {
        let result = try ensureStrictJSONSchema([
            "$defs": ["T": ["type": "string"]],
            "type": "object",
            "properties": ["value": ["$anchor": "value", "$ref": "#/$defs/T"]],
        ])
        #expect(node(result, "properties", "value") == ["$anchor": "value", "type": "string"])
    }

    // MARK: - oneOf (tests/test_strict_schema_oneof.py)

    @Test func testOneOfConvertedToAnyOf() throws {
        let result = try ensureStrictJSONSchema([
            "type": "object",
            "properties": ["value": ["oneOf": [["type": "string"], ["type": "integer"]]]],
        ])
        #expect(result == [
            "type": "object",
            "properties": ["value": ["anyOf": [["type": "string"], ["type": "integer"]]]],
            "additionalProperties": false,
            "required": ["value"],
        ])
    }

    @Test func testNestedOneOfInArrayItems() throws {
        let discriminator: JSONValue = [
            "propertyName": "action",
            "mapping": [
                "buy_fruit": "#/components/schemas/BuyFruitStep",
                "buy_food": "#/components/schemas/BuyFoodStep",
            ],
        ]
        let result = try ensureStrictJSONSchema([
            "type": "object",
            "properties": [
                "steps": [
                    "type": "array",
                    "items": [
                        "oneOf": [
                            [
                                "type": "object",
                                "properties": ["action": ["type": "string", "const": "buy_fruit"], "color": ["type": "string"]],
                                "required": ["action", "color"],
                            ],
                            [
                                "type": "object",
                                "properties": ["action": ["type": "string", "const": "buy_food"], "price": ["type": "integer"]],
                                "required": ["action", "price"],
                            ],
                        ],
                        "discriminator": discriminator,
                    ],
                ],
            ],
        ])
        #expect(result == [
            "type": "object",
            "properties": [
                "steps": [
                    "type": "array",
                    "items": [
                        "anyOf": [
                            [
                                "type": "object",
                                "properties": ["action": ["type": "string", "const": "buy_fruit"], "color": ["type": "string"]],
                                "required": ["action", "color"],
                                "additionalProperties": false,
                            ],
                            [
                                "type": "object",
                                "properties": ["action": ["type": "string", "const": "buy_food"], "price": ["type": "integer"]],
                                "required": ["action", "price"],
                                "additionalProperties": false,
                            ],
                        ],
                        "discriminator": discriminator,
                    ],
                ],
            ],
            "additionalProperties": false,
            "required": ["steps"],
        ])
    }

    @Test func testOneOfMergedWithExistingAnyOf() throws {
        let result = try ensureStrictJSONSchema([
            "type": "object",
            "properties": [
                "value": [
                    "anyOf": [["type": "string"]],
                    "oneOf": [["type": "integer"], ["type": "boolean"]],
                ],
            ],
        ])
        #expect(node(result, "properties", "value") == [
            "anyOf": [["type": "string"], ["type": "integer"], ["type": "boolean"]],
        ])
    }
}
