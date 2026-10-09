import Testing
import Agents

// Ported from tests/test_handoff_prompt.py.
@Suite struct HandoffPromptTests {
    @Test func testPromptWithHandoffInstructionsIncludesPrefix() {
        let prompt = "Handle the transfer smoothly."
        let result = promptWithHandoffInstructions(prompt)

        #expect(result.hasPrefix(recommendedPromptPrefix))
        #expect(result.hasSuffix(prompt))
    }
}
