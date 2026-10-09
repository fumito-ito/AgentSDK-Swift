import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing
@testable import Agents

// Ported from tests/models/test_openai_retry_helpers.py.
//
// Swift: retry metadata lives on typed `OpenAIAPIError` values instead of arbitrary exception
// attributes and `__cause__` chains, so the attribute-introspection tests
// (`get_status_code`, `get_error_code`, `response_headers`, outer-exception precedence, httpx
// header classes, and transport-family normalization) are skipped. `unsafe_to_replay` is only set
// by the websocket transport, which is not ported.
@Suite struct OpenAIRetryHelpersTests {
    private func makeRequest(_ error: any Error) -> ModelRetryAdviceRequest {
        ModelRetryAdviceRequest(error: error, attempt: 1, stream: false)
    }

    private func httpError(status: Int, headers: [String: String], body: JSONObject = [:]) -> OpenAIAPIError {
        let url = URL(string: "https://api.openai.com/v1/responses")!
        let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)
        return OpenAIClient.apiError(status: status, data: Data(body.jsonString().utf8), response: response)
    }

    @Test func testHeaderLookupPlainMappingMatchesCaseInsensitively() {
        let headers = ["Retry-After": "5", "X-Other": "ignored"]
        #expect(headerLookup(headers, "retry-after") == "5")
        #expect(headerLookup(headers, "missing") == nil)
        #expect(headerLookup(nil, "retry-after") == nil)
    }

    @Test func testParseRetryAfterMsInvalidReturnsNone() {
        #expect(parseRetryAfterMs(nil) == nil)
        #expect(parseRetryAfterMs("not-a-number") == nil)
        #expect(parseRetryAfterMs("-100") == nil)
        #expect(parseRetryAfterMs("1500") == 1.5)
    }

    @Test func testParseRetryAfterNumericAndHTTPDate() throws {
        #expect(parseRetryAfterValue(nil) == nil)
        #expect(parseRetryAfterValue("2") == 2.0)
        #expect(parseRetryAfterValue("-1") == nil)

        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "GMT")
        formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
        let future = formatter.string(from: Date().addingTimeInterval(120))
        let parsed = try #require(parseRetryAfterValue(future))
        #expect(parsed > 0)

        #expect(parseRetryAfterValue("definitely not a date") == nil)
    }

    @Test func testProviderAndRunnerRetryNormalizationShareMetadata() throws {
        let error = httpError(status: 429, headers: ["x-request-id": "req_test", "retry-after-ms": "1500"], body: ["error": ["code": "rate_limit_exceeded"]])

        let advice = try #require(getOpenAIRetryAdvice(makeRequest(error)))
        let runnerNormalized = normalizeRetryError(error, nil)

        let normalized = try #require(advice.normalized)
        #expect(normalized.statusCode == runnerNormalized.statusCode)
        #expect(normalized.errorCode == runnerNormalized.errorCode)
        #expect(normalized.requestId == runnerNormalized.requestId)
        #expect(normalized.retryAfter == runnerNormalized.retryAfter)
        #expect(runnerNormalized.retryAfter == 1.5)
    }

    @Test func testAdviceWebsocketRequestIsUnsafe() throws {
        let message = "The request may have been accepted, so the SDK will not automatically retry this websocket request."
        let advice = try #require(getOpenAIRetryAdvice(makeRequest(TestError(message))))
        #expect(advice.suggested == false)
        #expect(advice.replaySafety == "unsafe")
    }

    @Test func testAdviceRespectsXShouldRetryFalse() throws {
        let advice = try #require(getOpenAIRetryAdvice(makeRequest(httpError(status: 500, headers: ["x-should-retry": "false"]))))
        #expect(advice.suggested == false)
    }

    @Test func testAdviceReturnsRetryAfterOnlyWhenNoOtherSignal() throws {
        // A 400 with no x-should-retry header and no network/timeout signal would not normally
        // retry, but a retry-after header still yields advice carrying the delay.
        let advice = try #require(getOpenAIRetryAdvice(makeRequest(httpError(status: 400, headers: ["retry-after": "2"]))))
        #expect(advice.retryAfter == 2.0)
        // This branch only conveys the server-provided delay; `suggested` keeps its unset default.
        #expect(advice.suggested == nil)
    }
}
