import Testing
@testable import Agents

// Ported from tests/test_transforms.py.
@Suite struct TransformsTests {
    @Test(arguments: zip(["My Tool", "My-Tool"], ["my_tool", "my_tool"]))
    func testTransformStringFunctionStyleWarnsForReplacedCharacters(name: String, transformed: String) async {
        await captureLogs(level: .warning) { logs in
            #expect(transformStringFunctionStyle(name) == transformed)

            let text = logs.messages.joined(separator: "\n")
            #expect(text.contains("Tool name '\(name)' contains invalid characters"))
            #expect(text.contains("transformed to '\(transformed)'"))
        }
    }

    @Test(arguments: zip(["MyTool", "transfer_to_Agent", "snake_case"], ["mytool", "transfer_to_agent", "snake_case"]))
    func testTransformStringFunctionStyleDoesNotWarnForCaseOnlyChanges(name: String, transformed: String) async {
        await captureLogs(level: .warning) { logs in
            #expect(transformStringFunctionStyle(name) == transformed)
            #expect(logs.records.isEmpty)
        }
    }
}
