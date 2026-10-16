import Foundation
import Testing
@testable import Agents

// Swift-only tests for the JSON foundation (no upstream counterpart; see porting/decisions.md D3).
@Suite struct JSONValueTests {
    @Test func testParsePreservesKeyOrder() throws {
        let value = try JSONValue.parse(#"{"b": 1, "a": {"z": true, "y": null}, "c": [1, 2.5, "x"]}"#)
        let object = try #require(value.objectValue)
        #expect(object.keys == ["b", "a", "c"])
        #expect(object["a"]?.objectValue?.keys == ["z", "y"])
        #expect(value.jsonString() == #"{"b":1,"a":{"z":true,"y":null},"c":[1,2.5,"x"]}"#)
    }

    @Test func testEqualityIgnoresKeyOrderAndNumericKind() {
        let left: JSONValue = ["a": 1, "b": 2]
        let right: JSONValue = ["b": 2.0, "a": 1]
        #expect(left == right)
    }

    @Test func testPythonStyleSerialization() {
        let value: JSONValue = ["assistant": "Agent é", "n": 1.0]
        #expect(value.pythonJSONString() == #"{"assistant": "Agent \u00e9", "n": 1.0}"#)
    }

    @Test func testParseEscapesAndUnicode() throws {
        let value = try JSONValue.parse(#""line\nbreak é 😀 \"q\"""#)
        #expect(value.stringValue == "line\nbreak é 😀 \"q\"")
        #expect(value.jsonString() == #""line\nbreak é 😀 \"q\"""#)
    }

    @Test func testParseRejectsInvalidJSON() {
        #expect(throws: JSONSerializationError.self) { try JSONValue.parse("{") }
        #expect(throws: JSONSerializationError.self) { try JSONValue.parse("[1,]x") }
        #expect(throws: JSONSerializationError.self) { try JSONValue.parse("") }
    }

    @Test func testSubscriptAssignmentKeepsPosition() {
        var object: JSONObject = ["a": 1, "b": 2]
        object["a"] = 3
        object["c"] = 4
        object["b"] = nil
        #expect(object.keys == ["a", "c"])
        #expect(object["a"] == 3)
    }

    @Test func testCodableRoundTrip() throws {
        struct Payload: Codable, Equatable { let name: String; let count: Int }
        let json = try JSONValue(encoding: Payload(name: "x", count: 2))
        #expect(json == ["name": "x", "count": 2])
        #expect(try json.decode(as: Payload.self) == Payload(name: "x", count: 2))
    }
}
