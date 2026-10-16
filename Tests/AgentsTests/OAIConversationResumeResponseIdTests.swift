import Testing
@testable import Agents

// Ported from tests/test_oaiconv_resume_response_id.py.
@Suite struct OAIConversationResumeResponseIdTests {
    @Test func testHydrateFromStateUsesLatestNonNoneResponseId() {
        let tracker = OpenAIServerConversationTracker(conversationId: nil, previousResponseId: nil, autoPreviousResponseId: true)
        let responses = [
            ModelResponse(output: [], usage: Usage(), responseId: "resp_first"),
            ModelResponse(output: [], usage: Usage(), responseId: "resp_second"),
            ModelResponse(output: [], usage: Usage(), responseId: nil),
        ]

        tracker.hydrateFromState(originalInput: .items([]), generatedItems: [], modelResponses: responses)

        #expect(tracker.previousResponseId == "resp_second")
    }
}
