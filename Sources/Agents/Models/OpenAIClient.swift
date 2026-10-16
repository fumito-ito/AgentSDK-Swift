import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

// Swift: replaces the `openai` Python package (`AsyncOpenAI`). Only the endpoints the SDK uses
// are implemented: Responses (create, streamed create) and Chat Completions (create, streamed
// create). Requests are sent with URLSession; streaming uses a data delegate so it also works with
// FoundationNetworking on Linux.

/// An error returned by the OpenAI API (upstream `openai.APIStatusError` / `APIConnectionError`).
public struct OpenAIAPIError: ModelAPIError, CustomStringConvertible, Sendable {
    public var statusCode: Int?
    public var errorCode: String?
    public var errorType: String?
    public var message: String
    public var requestId: String?
    public var retryAfter: Double?
    public var body: JSONValue?
    public var isConnectionError: Bool
    public var isTimeout: Bool
    /// The `x-should-retry` response header, if present.
    public var shouldRetryHeader: String?

    public init(
        statusCode: Int? = nil,
        errorCode: String? = nil,
        errorType: String? = nil,
        message: String,
        requestId: String? = nil,
        retryAfter: Double? = nil,
        body: JSONValue? = nil,
        isConnectionError: Bool = false,
        isTimeout: Bool = false,
        shouldRetryHeader: String? = nil
    ) {
        self.statusCode = statusCode
        self.errorCode = errorCode
        self.errorType = errorType
        self.message = message
        self.requestId = requestId
        self.retryAfter = retryAfter
        self.body = body
        self.isConnectionError = isConnectionError
        self.isTimeout = isTimeout
        self.shouldRetryHeader = shouldRetryHeader
    }

    public var description: String {
        if let statusCode { return "Error code: \(statusCode) - \(message)" }
        return message
    }
}

/// A minimal OpenAI API client.
public final class OpenAIClient: @unchecked Sendable {
    public static let defaultBaseURL = URL(string: "https://api.openai.com/v1")!

    public let apiKey: String?
    public let baseURL: URL
    public let organization: String?
    public let project: String?
    /// Hidden SDK retries for retryable failures (disabled per-request by the runner's retry policy).
    public let maxRetries: Int
    public let timeout: TimeInterval
    private let session: URLSession

    public init(
        apiKey: String? = nil,
        baseURL: URL? = nil,
        organization: String? = nil,
        project: String? = nil,
        maxRetries: Int = 2,
        timeout: TimeInterval = 600,
        session: URLSession? = nil
    ) {
        let environment = ProcessInfo.processInfo.environment
        self.apiKey = apiKey ?? environment["OPENAI_API_KEY"]
        self.baseURL = baseURL ?? environment["OPENAI_BASE_URL"].flatMap(URL.init(string:)) ?? Self.defaultBaseURL
        self.organization = organization ?? environment["OPENAI_ORG_ID"]
        self.project = project ?? environment["OPENAI_PROJECT_ID"]
        self.maxRetries = maxRetries
        self.timeout = timeout
        if let session {
            self.session = session
        } else {
            let configuration = URLSessionConfiguration.default
            configuration.timeoutIntervalForRequest = timeout
            configuration.timeoutIntervalForResource = timeout
            self.session = URLSession(configuration: configuration)
        }
    }

    /// Whether this client targets the official OpenAI endpoint.
    public var isOfficialOpenAIEndpoint: Bool {
        baseURL.host == "api.openai.com"
    }

    // MARK: Requests

    private func makeRequest(path: String, body: JSONObject, extraHeaders: [String: String?], extraQuery: JSONObject?, stream: Bool) throws -> URLRequest {
        guard let apiKey, !apiKey.isEmpty else {
            throw UserError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )
        }
        var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        if let extraQuery, !extraQuery.isEmpty {
            components.queryItems = extraQuery.pairs.map { key, value in
                URLQueryItem(name: key, value: value.stringValue ?? value.jsonString())
            }
        }
        var request = URLRequest(url: components.url!)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(stream ? "text/event-stream" : "application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        if let organization { request.setValue(organization, forHTTPHeaderField: "OpenAI-Organization") }
        if let project { request.setValue(project, forHTTPHeaderField: "OpenAI-Project") }
        for (name, value) in extraHeaders {
            request.setValue(value, forHTTPHeaderField: name)
        }
        request.httpBody = Data(JSONValue.object(body).jsonString().utf8)
        return request
    }

    private var effectiveMaxRetries: Int {
        ModelRetryScope.providerManagedRetriesDisabled ? 0 : maxRetries
    }

    /// POSTs a JSON body and returns the decoded JSON object and request ID.
    public func post(path: String, body: JSONObject, extraHeaders: [String: String?] = [:], extraQuery: JSONObject? = nil) async throws -> (JSONObject, String?) {
        let request = try makeRequest(path: path, body: body, extraHeaders: extraHeaders, extraQuery: extraQuery, stream: false)
        var attempt = 0
        while true {
            do {
                let (data, response) = try await performDataTask(request)
                let http = response as? HTTPURLResponse
                let requestId = http?.value(forHTTPHeaderField: "x-request-id")
                let status = http?.statusCode ?? 0
                guard (200..<300).contains(status) else {
                    throw Self.apiError(status: status, data: data, response: http)
                }
                guard case .object(let object) = try JSONValue.parse(data) else {
                    throw OpenAIAPIError(statusCode: status, message: "Expected a JSON object response", requestId: requestId)
                }
                return (object, requestId)
            } catch {
                let mapped = Self.mapTransportError(error)
                guard attempt < effectiveMaxRetries, Self.shouldRetry(mapped) else { throw mapped }
                attempt += 1
                try await Task.sleep(nanoseconds: UInt64(Self.retryDelay(attempt: attempt, error: mapped) * 1_000_000_000))
            }
        }
    }

    /// POSTs a JSON body and streams server-sent events as JSON objects.
    public func stream(path: String, body: JSONObject, extraHeaders: [String: String?] = [:], extraQuery: JSONObject? = nil) -> AsyncThrowingStream<JSONObject, any Error> {
        var body = body
        body["stream"] = true
        let requestResult = Result { try makeRequest(path: path, body: body, extraHeaders: extraHeaders, extraQuery: extraQuery, stream: true) }
        let session = self.session
        return AsyncThrowingStream { continuation in
            let request: URLRequest
            switch requestResult {
            case .success(let value): request = value
            case .failure(let error):
                continuation.finish(throwing: error)
                return
            }
            let delegate = SSEStreamDelegate(continuation: continuation)
            let streamSession = URLSession(configuration: session.configuration, delegate: delegate, delegateQueue: nil)
            let task = streamSession.dataTask(with: request)
            continuation.onTermination = { _ in
                task.cancel()
                streamSession.invalidateAndCancel()
            }
            task.resume()
        }
    }

    private func performDataTask(_ request: URLRequest) async throws -> (Data, URLResponse) {
        try await withCheckedThrowingContinuation { continuation in
            let task = session.dataTask(with: request) { data, response, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if let response {
                    continuation.resume(returning: (data ?? Data(), response))
                } else {
                    // swift-corelibs-foundation has no `URLResponse()`, so a missing response is an error.
                    continuation.resume(throwing: URLError(.badServerResponse))
                }
            }
            task.resume()
        }
    }

    // MARK: Errors

    static func apiError(status: Int, data: Data, response: HTTPURLResponse?) -> OpenAIAPIError {
        let body = try? JSONValue.parse(data)
        let errorObject = body?.objectValue?["error"]?.objectValue
        let message = errorObject?.string("message") ?? String(data: data, encoding: .utf8) ?? "HTTP \(status)"
        let headers = response?.allHeaderFields.reduce(into: [String: String]()) { result, pair in
            if let key = pair.key as? String, let value = pair.value as? String { result[key] = value }
        }
        let retryAfter = getRetryAfter(headers)
        return OpenAIAPIError(
            statusCode: status,
            errorCode: errorObject?.string("code"),
            errorType: errorObject?.string("type"),
            message: message,
            requestId: response?.value(forHTTPHeaderField: "x-request-id"),
            retryAfter: retryAfter,
            body: body,
            shouldRetryHeader: response?.value(forHTTPHeaderField: "x-should-retry")
        )
    }

    static func mapTransportError(_ error: any Error) -> any Error {
        guard let urlError = error as? URLError else { return error }
        if urlError.code == .cancelled { return CancellationError() }
        return OpenAIAPIError(
            message: urlError.code == .timedOut ? "Request timed out." : "Connection error.",
            isConnectionError: true,
            isTimeout: urlError.code == .timedOut
        )
    }

    static func shouldRetry(_ error: any Error) -> Bool {
        guard let apiError = error as? OpenAIAPIError else { return false }
        if let header = apiError.shouldRetryHeader?.lowercased() {
            if header == "true" { return true }
            if header == "false" { return false }
        }
        if apiError.isConnectionError { return true }
        guard let status = apiError.statusCode else { return false }
        return status == 408 || status == 409 || status == 429 || status >= 500
    }

    static func retryDelay(attempt: Int, error: any Error) -> Double {
        if let retryAfter = (error as? OpenAIAPIError)?.retryAfter, retryAfter > 0, retryAfter <= 60 {
            return retryAfter
        }
        let base = min(0.5 * pow(2.0, Double(attempt - 1)), 8.0)
        return base * (1 - 0.25 * Double.random(in: 0..<1))
    }
}

/// Parses a server-sent event stream into JSON objects.
private final class SSEStreamDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let continuation: AsyncThrowingStream<JSONObject, any Error>.Continuation
    private var buffer = Data()
    private var dataLines: [String] = []
    private var eventName: String?
    private var statusCode = 200
    private var errorBody = Data()
    private var response: HTTPURLResponse?
    private var finished = false

    init(continuation: AsyncThrowingStream<JSONObject, any Error>.Continuation) {
        self.continuation = continuation
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        self.response = response as? HTTPURLResponse
        statusCode = self.response?.statusCode ?? 200
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard (200..<300).contains(statusCode) else {
            errorBody.append(data)
            return
        }
        buffer.append(data)
        while let newline = buffer.firstIndex(of: UInt8(ascii: "\n")) {
            var lineData = buffer[buffer.startIndex..<newline]
            buffer.removeSubrange(buffer.startIndex...newline)
            if lineData.last == UInt8(ascii: "\r") { lineData = lineData.dropLast() }
            handleLine(String(decoding: lineData, as: UTF8.self))
        }
    }

    private func handleLine(_ line: String) {
        if line.isEmpty {
            dispatchEvent()
            return
        }
        if line.hasPrefix(":") { return }
        let field: Substring
        var value: Substring
        if let colon = line.firstIndex(of: ":") {
            field = line[line.startIndex..<colon]
            value = line[line.index(after: colon)...]
            if value.hasPrefix(" ") { value = value.dropFirst() }
        } else {
            field = Substring(line)
            value = ""
        }
        switch field {
        case "data": dataLines.append(String(value))
        case "event": eventName = String(value)
        default: break
        }
    }

    private func dispatchEvent() {
        defer {
            dataLines = []
            eventName = nil
        }
        guard !dataLines.isEmpty else { return }
        let payload = dataLines.joined(separator: "\n")
        if payload == "[DONE]" { return }
        do {
            guard case .object(var object) = try JSONValue.parse(payload) else { return }
            if eventName == "error" || (object["error"] != nil && object["type"] == nil) {
                let errorObject = object["error"]?.objectValue ?? object
                finish(throwing: OpenAIAPIError(
                    errorCode: errorObject.string("code"),
                    errorType: errorObject.string("type"),
                    message: errorObject.string("message") ?? payload,
                    requestId: response?.value(forHTTPHeaderField: "x-request-id"),
                    body: .object(object)
                ))
                return
            }
            // Expose the transport request ID on terminal response payloads (upstream `_request_id`).
            if let requestId = response?.value(forHTTPHeaderField: "x-request-id"),
               object.string("type") == "response.completed", case .object(var terminal)? = object["response"] {
                terminal[sseRequestIdKey] = .string(requestId)
                object["response"] = .object(terminal)
            }
            continuation.yield(object)
        } catch {
            finish(throwing: OpenAIAPIError(message: "Could not parse stream event: \(payload)"))
        }
    }

    private func finish(throwing error: (any Error)?) {
        guard !finished else { return }
        finished = true
        if let error {
            continuation.finish(throwing: error)
        } else {
            continuation.finish()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        defer { session.finishTasksAndInvalidate() }
        if let error {
            finish(throwing: OpenAIClient.mapTransportError(error))
            return
        }
        guard (200..<300).contains(statusCode) else {
            finish(throwing: OpenAIClient.apiError(status: statusCode, data: errorBody, response: response))
            return
        }
        if !buffer.isEmpty {
            handleLine(String(decoding: buffer, as: UTF8.self))
            buffer.removeAll()
        }
        dispatchEvent()
        finish(throwing: nil)
    }
}

/// The key under which the stream attaches the transport request ID to terminal responses.
let sseRequestIdKey = "_agents_sdk_request_id"
