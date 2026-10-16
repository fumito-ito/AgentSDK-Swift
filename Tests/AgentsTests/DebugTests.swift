import Testing
@testable import Agents

// Ported from tests/test_debug.py.
//
// Swift: the environment is passed to the loaders instead of patching the process environment,
// so the tests stay safe under parallel execution.
@Suite struct DebugTests {
    @Test(arguments: [
        (nil, true), ("0", false), ("1", true), ("true", true), ("false", false),
    ] as [(String?, Bool)])
    func testDontLogModelData(value: String?, expected: Bool) {
        let environment = value.map { ["OPENAI_AGENTS_DONT_LOG_MODEL_DATA": $0] } ?? [:]
        #expect(AgentsDebug.loadDontLogModelData(environment: environment) == expected)
    }

    @Test(arguments: [
        (nil, true), ("0", false), ("1", true), ("true", true), ("false", false),
    ] as [(String?, Bool)])
    func testDontLogToolData(value: String?, expected: Bool) {
        let environment = value.map { ["OPENAI_AGENTS_DONT_LOG_TOOL_DATA": $0] } ?? [:]
        #expect(AgentsDebug.loadDontLogToolData(environment: environment) == expected)
    }
}
