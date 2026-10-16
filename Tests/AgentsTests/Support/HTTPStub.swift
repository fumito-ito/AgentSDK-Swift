import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import Agents

// Replaces upstream's fake `AsyncOpenAI` clients. Each `HTTPStub` owns a unique host, and
// `StubURLProtocol` answers requests for that host from the stub's handler, so model-level tests
// exercise the real `OpenAIClient` without network access and stay isolated under parallel runs.

/// A canned HTTP response.
struct StubResponse: Sendable {
    var status: Int = 200
    var headers: [String: String] = ["Content-Type": "application/json"]
    /// Body chunks delivered in order (one chunk for JSON, several for SSE).
    var chunks: [Data]

    static func json(_ object: JSONObject, status: Int = 200, headers: [String: String] = [:]) -> StubResponse {
        var allHeaders = ["Content-Type": "application/json"]
        allHeaders.merge(headers) { _, new in new }
        return StubResponse(status: status, headers: allHeaders, chunks: [Data(object.jsonString().utf8)])
    }

    static func sse(_ events: [JSONObject], headers: [String: String] = [:]) -> StubResponse {
        var allHeaders = ["Content-Type": "text/event-stream"]
        allHeaders.merge(headers) { _, new in new }
        let chunks = events.map { Data("data: \($0.jsonString())\n\n".utf8) } + [Data("data: [DONE]\n\n".utf8)]
        return StubResponse(status: 200, headers: allHeaders, chunks: chunks)
    }
}

/// A request captured by an `HTTPStub`.
struct CapturedRequest: Sendable {
    var url: URL
    var headers: [String: String]
    var body: JSONObject?

    func header(_ name: String) -> String? {
        headers.first { $0.key.caseInsensitiveCompare(name) == .orderedSame }?.value
    }
}

final class HTTPStub: @unchecked Sendable {
    typealias Handler = @Sendable (CapturedRequest) throws -> StubResponse

    let host = "stub-\(UUID().uuidString.lowercased()).test"
    let handler: Handler
    private let captured = SharedBox<[CapturedRequest]>([])

    var requests: [CapturedRequest] { captured.value }
    var baseURL: URL { URL(string: "https://\(host)/v1")! }

    init(_ handler: @escaping Handler) {
        self.handler = handler
        StubURLProtocol.register(self)
    }

    deinit {
        StubURLProtocol.unregister(host)
    }

    func record(_ request: CapturedRequest) {
        captured.mutate { $0.append(request) }
    }

    /// A session whose requests are answered by this stub.
    func makeSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    /// An `OpenAIClient` that talks to this stub with retries disabled.
    func makeClient(apiKey: String = "test_key", maxRetries: Int = 0) -> OpenAIClient {
        OpenAIClient(apiKey: apiKey, baseURL: baseURL, maxRetries: maxRetries, session: makeSession())
    }
}

final class StubURLProtocol: URLProtocol, @unchecked Sendable {
    private static let stubs = SharedBox<[String: HTTPStub]>([:])

    static func register(_ stub: HTTPStub) {
        stubs.mutate { $0[stub.host] = stub }
    }

    static func unregister(_ host: String) {
        stubs.mutate { _ = $0.removeValue(forKey: host) }
    }

    private static func stub(for request: URLRequest) -> HTTPStub? {
        guard let host = request.url?.host else { return nil }
        return stubs.value[host]
    }

    override static func canInit(with request: URLRequest) -> Bool {
        stub(for: request) != nil
    }

    override static func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let stub = Self.stub(for: request), let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL))
            return
        }
        let captured = CapturedRequest(url: url, headers: request.allHTTPHeaderFields ?? [:], body: Self.readBody(request))
        stub.record(captured)
        do {
            let response = try stub.handler(captured)
            let httpResponse = HTTPURLResponse(url: url, statusCode: response.status, httpVersion: "HTTP/1.1", headerFields: response.headers)!
            client?.urlProtocol(self, didReceive: httpResponse, cacheStoragePolicy: .notAllowed)
            for chunk in response.chunks {
                client?.urlProtocol(self, didLoad: chunk)
            }
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}

    private static func readBody(_ request: URLRequest) -> JSONObject? {
        var data = request.httpBody
        if data == nil, let stream = request.httpBodyStream {
            var buffer = Data()
            stream.open()
            defer { stream.close() }
            let size = 4_096
            var chunk = [UInt8](repeating: 0, count: size)
            while stream.hasBytesAvailable {
                let read = stream.read(&chunk, maxLength: size)
                if read <= 0 { break }
                buffer.append(chunk, count: read)
            }
            data = buffer
        }
        guard let data, case .object(let object)? = try? JSONValue.parse(data) else { return nil }
        return object
    }
}
