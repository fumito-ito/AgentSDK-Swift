import Foundation
import Agents

// Port of src/agents/testing/model.py.
//
// Swift: step validation reasons are reported through `InvalidModelStep.reason`, but the
// dictionary step form (`ModelStepSpec`) is not ported because Swift steps are typed.

/// Base error for an invalid or incompletely consumed model script.
public class ModelScriptError: Error, CustomStringConvertible, @unchecked Sendable {
    public let message: String

    public init(_ message: String) {
        self.message = message
    }

    public var description: String { message }
}

/// Raised when a model step is invalid before it enters the script queue.
public final class InvalidModelStep: ModelScriptError, @unchecked Sendable {
    public let reason: String
    public let inputIndex: Int

    public init(_ message: String, reason: String, inputIndex: Int) {
        self.reason = reason
        self.inputIndex = inputIndex
        super.init(message)
    }
}

/// Raised when the model is called after all configured steps were consumed.
public final class UnexpectedModelCall: ModelScriptError, @unchecked Sendable {
    public let call: ModelCall
    public let callIndex: Int

    public init(_ message: String, call: ModelCall, callIndex: Int) {
        self.call = call
        self.callIndex = callIndex
        super.init(message)
    }
}

/// Raised when a test finishes before consuming every configured step.
public final class UnconsumedModelSteps: ModelScriptError, @unchecked Sendable {
    public let remainingSteps: Int

    public init(_ message: String, remainingSteps: Int) {
        self.remainingSteps = remainingSteps
        super.init(message)
    }
}

/// A recorded call at the provider-neutral `Model` boundary.
public struct ModelCall: Sendable {
    public let systemInstructions: String?
    public let input: RunInput
    public let modelSettings: ModelSettings
    public let tools: [any Tool]
    public let outputSchema: (any AgentOutputSchemaBase)?
    public let handoffs: [Handoff]
    public let tracing: ModelTracing
    public let previousResponseId: String?
    public let conversationId: String?
    public let prompt: JSONObject?
    public let streamed: Bool
}

/// The result a responder may return for a call.
public enum ModelStepResult: Sendable {
    case step(ModelStep)
    case response(ModelResponse)
    case output([TResponseOutputItem])
    case error(any Error)
}

/// One deterministic model call result.
///
/// `output` uses the normalized SDK output-item boundary. Set `error` to throw from the model
/// call, `responder` to derive the result from the recorded call, or `streamEvents` to supply an
/// exact normalized event stream for advanced streaming tests.
public struct ModelStep: Sendable {
    public typealias Responder = @Sendable (ModelCall) async throws -> ModelStepResult
    public typealias StreamFactory = @Sendable (ModelCall) -> AsyncThrowingStream<TResponseStreamEvent, any Error>

    public enum StreamEvents: Sendable {
        case events([TResponseStreamEvent])
        case factory(StreamFactory)
    }

    public var output: [TResponseOutputItem]
    public var usage: Usage
    public var responseId: String?
    public var requestId: String?
    public var rawUsage: JSONObject?
    public var error: (any Error)?
    public var responder: Responder?
    public var streamEvents: StreamEvents?
    public var retryAdvice: ModelRetryAdvice?

    public init(
        output: [TResponseOutputItem] = [],
        usage: Usage = Usage(),
        responseId: String? = "resp-789",
        requestId: String? = nil,
        rawUsage: JSONObject? = nil,
        error: (any Error)? = nil,
        responder: Responder? = nil,
        streamEvents: StreamEvents? = nil,
        retryAdvice: ModelRetryAdvice? = nil
    ) {
        self.output = output
        self.usage = usage
        self.responseId = responseId
        self.requestId = requestId
        self.rawUsage = rawUsage
        self.error = error
        self.responder = responder
        self.streamEvents = streamEvents
        self.retryAdvice = retryAdvice
    }

    /// A step that returns `output`.
    public static func output(_ output: [TResponseOutputItem], usage: Usage = Usage(), responseId: String? = "resp-789") -> ModelStep {
        ModelStep(output: output, usage: usage, responseId: responseId)
    }

    /// A step that throws `error` with optional provider retry guidance.
    public static func raiseError(_ error: any Error, retryAdvice: ModelRetryAdvice? = nil) -> ModelStep {
        ModelStep(error: error, retryAdvice: retryAdvice)
    }

    /// A step whose result is derived from the recorded call.
    public static func respond(_ responder: @escaping Responder) -> ModelStep {
        ModelStep(responder: responder)
    }

    /// A step with an exact normalized stream-event sequence.
    public static func stream(_ events: [TResponseStreamEvent], output: [TResponseOutputItem] = [], usage: Usage? = nil, responseId: String? = "resp-789") -> ModelStep {
        ModelStep(output: output, usage: usage ?? Usage(), responseId: responseId, streamEvents: .events(events))
    }

    /// A step with a stream-event factory.
    public static func stream(_ factory: @escaping StreamFactory, output: [TResponseOutputItem] = [], usage: Usage? = nil, responseId: String? = "resp-789") -> ModelStep {
        ModelStep(output: output, usage: usage ?? Usage(), responseId: responseId, streamEvents: .factory(factory))
    }
}

/// A deterministic provider-neutral model for testing agent workflows.
public final class ScriptedModel: Model, @unchecked Sendable {
    private let lock = NSLock()
    private var steps: [ModelStep]
    private let emitTraces: Bool
    private var defaultUsage: Usage?
    private var recordedCalls: [ModelCall] = []
    private var retryAdviceByError: [(error: any Error, advice: ModelRetryAdvice)] = []

    public init(steps: [ModelStep] = [], emitTraces: Bool = false, defaultUsage: Usage? = nil) throws {
        for (index, step) in steps.enumerated() {
            try Self.validate(step, inputIndex: index)
        }
        self.steps = steps
        self.emitTraces = emitTraces
        self.defaultUsage = defaultUsage
    }

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    /// The recorded model calls.
    public var calls: [ModelCall] { locked { recordedCalls } }
    /// The number of configured model calls that have not run yet.
    public var remainingSteps: Int { locked { steps.count } }
    /// The first recorded call, if any.
    public var firstCall: ModelCall? { locked { recordedCalls.first } }
    /// The most recent recorded call, if any.
    public var lastCall: ModelCall? { locked { recordedCalls.last } }

    /// Appends one model step.
    public func enqueue(_ step: ModelStep) throws {
        try Self.validate(step, inputIndex: 0)
        locked { steps.append(step) }
    }

    /// Appends multiple model steps.
    public func extend(_ newSteps: [ModelStep]) throws {
        for (index, step) in newSteps.enumerated() {
            try Self.validate(step, inputIndex: index)
        }
        locked { steps.append(contentsOf: newSteps) }
    }

    /// Sets usage for scripted steps that do not provide their own usage.
    public func setDefaultUsage(_ usage: Usage?) {
        locked { defaultUsage = usage }
    }

    /// Returns retry advice attached to the exact scripted error that was raised.
    ///
    /// Swift: class errors are matched by identity, as upstream; value-type errors have no
    /// identity and are matched by their reflected value.
    public func getRetryAdvice(_ request: ModelRetryAdviceRequest) -> ModelRetryAdvice? {
        locked {
            retryAdviceByError.first { Self.isSameError($0.error, request.error) }?.advice
        }
    }

    private static func isSameError(_ lhs: any Error, _ rhs: any Error) -> Bool {
        if type(of: lhs) is AnyClass, type(of: rhs) is AnyClass {
            return (lhs as AnyObject) === (rhs as AnyObject)
        }
        return type(of: lhs) == type(of: rhs) && String(reflecting: lhs) == String(reflecting: rhs)
    }

    /// Throws when configured steps remain unconsumed.
    public func assertComplete() throws {
        let remaining = remainingSteps
        if remaining > 0 {
            throw UnconsumedModelSteps("\(remaining) scripted model step(s) were not consumed.", remainingSteps: remaining)
        }
    }

    public func getResponse(_ request: ModelRequest) async throws -> ModelResponse {
        let call = record(request, streamed: false)
        let span = generationSpan(disabled: !emitTraces)
        return try await span.run {
            do {
                let step = try await nextResolvedStep(call)
                if let error = step.error {
                    rememberRetryAdvice(step, error)
                    throw error
                }
                return try Self.modelResponse(step, call.modelSettings)
            } catch {
                Self.setSpanError(span, error, call.tracing)
                throw error
            }
        }
    }

    public func streamResponse(_ request: ModelRequest) -> AsyncThrowingStream<TResponseStreamEvent, any Error> {
        let call = record(request, streamed: true)
        let span = generationSpan(disabled: !emitTraces)
        return AsyncThrowingStream { continuation in
            let task = Task {
                span.start()
                defer { span.finish() }
                do {
                    let step = try await span.withCurrent { try await self.nextResolvedStep(call) }
                    if let error = step.error {
                        self.rememberRetryAdvice(step, error)
                        throw error
                    }
                    switch step.streamEvents {
                    case .factory(let factory)?:
                        for try await event in factory(call) {
                            continuation.yield(event)
                        }
                    case .events(let events)?:
                        for event in events { continuation.yield(event) }
                    case nil:
                        let events = try streamEventsForStep(step, preserveRawUsage: call.modelSettings.preserveRawUsage == true)
                        for event in events { continuation.yield(event) }
                    }
                    continuation.finish()
                } catch {
                    Self.setSpanError(span, error, call.tracing)
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func record(_ request: ModelRequest, streamed: Bool) -> ModelCall {
        let call = ModelCall(
            systemInstructions: request.systemInstructions,
            input: request.input,
            modelSettings: request.modelSettings,
            tools: request.tools,
            outputSchema: request.outputSchema,
            handoffs: request.handoffs,
            tracing: request.tracing,
            previousResponseId: request.previousResponseId,
            conversationId: request.conversationId,
            prompt: request.prompt,
            streamed: streamed
        )
        locked { recordedCalls.append(call) }
        return call
    }

    private func nextResolvedStep(_ call: ModelCall) async throws -> ModelStep {
        let popped: ModelStep? = locked {
            steps.isEmpty ? nil : steps.removeFirst()
        }
        guard var step = popped else {
            let mode = call.streamed ? "streaming" : "non-streaming"
            let callIndex = calls.count - 1
            throw UnexpectedModelCall(
                "Unexpected \(mode) model call #\(callIndex + 1): no scripted steps remain.",
                call: call,
                callIndex: callIndex
            )
        }
        while let responder = step.responder {
            step = try Self.coerce(try await responder(call))
        }
        if step.usage == Usage() {
            step.usage = locked { defaultUsage } ?? Usage(requests: 1)
        }
        return step
    }

    private static func coerce(_ result: ModelStepResult) throws -> ModelStep {
        let step: ModelStep
        switch result {
        case .step(let value): step = value
        case .response(let response):
            step = ModelStep(output: response.output, usage: response.usage, responseId: response.responseId, requestId: response.requestId, rawUsage: response.rawUsage)
        case .output(let output): step = ModelStep(output: output)
        case .error(let error): step = .raiseError(error)
        }
        try validate(step, inputIndex: 0)
        return step
    }

    private static func validate(_ step: ModelStep, inputIndex: Int) throws {
        let outcomes = [step.error != nil, step.responder != nil, step.streamEvents != nil].filter { $0 }.count
        if outcomes > 1 || (step.error != nil && !step.output.isEmpty) || (step.responder != nil && !step.output.isEmpty) {
            throw InvalidModelStep(
                "Model step at index \(inputIndex) is invalid: sets conflicting outcomes",
                reason: "conflicting_outcomes",
                inputIndex: inputIndex
            )
        }
        if step.retryAdvice != nil && step.error == nil {
            throw InvalidModelStep(
                "Model step at index \(inputIndex) is invalid: retry_advice requires error",
                reason: "invalid_retry_advice",
                inputIndex: inputIndex
            )
        }
    }

    private func rememberRetryAdvice(_ step: ModelStep, _ error: any Error) {
        locked {
            retryAdviceByError.removeAll { Self.isSameError($0.error, error) }
            if let advice = step.retryAdvice {
                retryAdviceByError.append((error, advice))
            }
        }
    }

    private static func modelResponse(_ step: ModelStep, _ settings: ModelSettings) throws -> ModelResponse {
        ModelResponse(
            output: try convertOutputItems(step.output),
            usage: step.usage,
            responseId: step.responseId,
            requestId: step.requestId,
            rawUsage: settings.preserveRawUsage == true ? step.rawUsage : nil
        )
    }

    private static func setSpanError(_ span: any Span, _ error: any Error, _ tracing: ModelTracing) {
        let message = tracing.includeData() ? String(describing: error) : redactedTraceErrorMessage
        span.setError(SpanError(message: "Error", data: ["name": .string(String(describing: type(of: error))), "message": .string(message)]))
    }
}

/// Builds one normalized assistant text output item.
public func assistantMessage(_ text: String, itemId: String = "scripted-message") -> TResponseOutputItem {
    [
        "id": .string(itemId),
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            ["text": .string(text), "type": "output_text", "annotations": [], "logprobs": []],
        ],
    ]
}

/// Builds one normalized function-tool call output item.
public func functionCall(
    _ name: String,
    arguments: String,
    callId: String,
    itemId: String? = nil,
    namespace: String? = nil
) -> TResponseOutputItem {
    var item: JSONObject = [
        "id": .string(itemId ?? callId),
        "call_id": .string(callId),
        "type": "function_call",
        "name": .string(name),
        "arguments": .string(arguments),
    ]
    if let namespace { item["namespace"] = .string(namespace) }
    return item
}

/// Builds one normalized function-tool call output item from JSON arguments.
public func functionCall(
    _ name: String,
    arguments: JSONObject,
    callId: String,
    itemId: String? = nil,
    namespace: String? = nil
) -> TResponseOutputItem {
    functionCall(name, arguments: JSONValue.object(arguments).jsonString(), callId: callId, itemId: itemId, namespace: namespace)
}

private func convertOutputItems(_ output: [TResponseOutputItem]) throws -> [TResponseOutputItem] {
    try output.map { item in
        guard item.string("type") == "apply_patch_call" else { return item }
        guard let callId = item.string("call_id"), !callId.isEmpty else {
            throw ModelBehaviorError("Tool invocations require a non-empty string call ID before execution.")
        }
        var converted: JSONObject = ["type": "apply_patch_call"]
        if item.contains("id") {
            guard let itemId = item.string("id"), !itemId.isEmpty else {
                throw ModelBehaviorError("Apply-patch tool calls require a non-empty string item ID when provided.")
            }
            converted["id"] = .string(itemId)
        } else {
            converted["id"] = .string(callId)
        }
        converted["call_id"] = .string(callId)
        converted["status"] = item["status"] ?? "completed"
        converted["operation"] = item["operation"] ?? .null
        converted["caller"] = item["caller"] ?? .null
        return converted
    }
}

private func inProgressOutputItem(_ item: TResponseOutputItem) -> TResponseOutputItem {
    var copy = item
    switch item.string("type") {
    case "message":
        copy["content"] = []
    case "function_call":
        copy["arguments"] = ""
    case "reasoning":
        copy["summary"] = []
        if copy.contains("content") { copy["content"] = [] }
    default:
        break
    }
    copy["status"] = "in_progress"
    return copy
}

private func responseUsageForUsage(_ usage: Usage) -> JSONObject {
    [
        "input_tokens": .int(usage.inputTokens),
        "output_tokens": .int(usage.outputTokens),
        "total_tokens": .int(usage.totalTokens),
        "input_tokens_details": [
            "cache_write_tokens": .int(usage.inputTokensDetails.cacheWriteTokens),
            "cached_tokens": .int(usage.inputTokensDetails.cachedTokens),
        ],
        "output_tokens_details": ["reasoning_tokens": .int(usage.outputTokensDetails.reasoningTokens)],
    ]
}

private func responseForStep(_ step: ModelStep, _ output: [TResponseOutputItem], preserveRawUsage: Bool) -> JSONObject {
    var usage = responseUsageForUsage(step.usage)
    usage["_agents_sdk_normalized_usage"] = .object(serializeUsage(step.usage))
    usage["_agents_sdk_request_count"] = .int(step.usage.requests)
    if preserveRawUsage, let rawUsage = step.rawUsage {
        usage = rawUsage.merging(usage.filter { key, _ in key.hasPrefix("_agents_sdk") })
    }
    var response = ResponseObject(id: step.responseId, output: output, usage: usage).json
    response["status"] = "completed"
    return response
}

private func streamEventsForStep(_ step: ModelStep, preserveRawUsage: Bool) throws -> [TResponseStreamEvent] {
    let output = try convertOutputItems(step.output)
    let supported: Set<String> = ["apply_patch_call", "function_call", "message", "reasoning"]
    if let unsupported = output.first(where: { !supported.contains($0.string("type") ?? "") }) {
        throw ModelBehaviorError(
            "Automatic streaming does not support \(unsupported.string("type") ?? "unknown"). "
                + "Use ModelStep.stream(...) to provide exact normalized stream events."
        )
    }
    let response = responseForStep(step, output, preserveRawUsage: preserveRawUsage)
    var inProgress = response
    inProgress["output"] = []
    inProgress["status"] = "in_progress"
    inProgress["usage"] = .null

    var events: [TResponseStreamEvent] = []
    var sequence = 0
    func append(_ event: JSONObject) {
        var event = event
        event["sequence_number"] = .int(sequence)
        sequence += 1
        events.append(event)
    }
    append(["type": "response.created", "response": .object(inProgress)])
    append(["type": "response.in_progress", "response": .object(inProgress)])

    for (outputIndex, item) in output.enumerated() {
        append(["type": "response.output_item.added", "item": .object(inProgressOutputItem(item)), "output_index": .int(outputIndex)])
        let itemId = item["id"] ?? .null
        switch item.string("type") {
        case "reasoning":
            for (summaryIndex, summary) in (item["summary"]?.arrayValue ?? []).enumerated() {
                let text = summary.objectValue?["text"] ?? ""
                let partType = summary.objectValue?["type"] ?? "summary_text"
                let base: JSONObject = ["item_id": itemId, "output_index": .int(outputIndex), "summary_index": .int(summaryIndex)]
                append(base.merging(["type": "response.reasoning_summary_part.added", "part": ["text": "", "type": partType]]))
                append(base.merging(["type": "response.reasoning_summary_text.delta", "delta": text]))
                append(base.merging(["type": "response.reasoning_summary_text.done", "text": text]))
                append(base.merging(["type": "response.reasoning_summary_part.done", "part": ["text": text, "type": partType]]))
            }
            for (contentIndex, content) in (item["content"]?.arrayValue ?? []).enumerated() {
                let text = content.objectValue?["text"] ?? ""
                let base: JSONObject = ["item_id": itemId, "output_index": .int(outputIndex), "content_index": .int(contentIndex)]
                append(base.merging(["type": "response.reasoning_text.delta", "delta": text]))
                append(base.merging(["type": "response.reasoning_text.done", "text": text]))
            }
        case "function_call":
            let arguments = item["arguments"] ?? ""
            let base: JSONObject = ["item_id": itemId, "output_index": .int(outputIndex)]
            append(base.merging(["type": "response.function_call_arguments.delta", "delta": arguments]))
            append(base.merging(["type": "response.function_call_arguments.done", "arguments": arguments]))
        case "message":
            for (contentIndex, part) in (item["content"]?.arrayValue ?? []).enumerated() {
                guard let part = part.objectValue else { continue }
                let base: JSONObject = ["item_id": itemId, "output_index": .int(outputIndex), "content_index": .int(contentIndex)]
                if part.string("type") == "refusal" {
                    let refusal = part["refusal"] ?? ""
                    append(base.merging(["type": "response.content_part.added", "part": ["type": "refusal", "refusal": ""]]))
                    append(base.merging(["type": "response.refusal.delta", "delta": refusal]))
                    append(base.merging(["type": "response.refusal.done", "refusal": refusal]))
                    append(base.merging(["type": "response.content_part.done", "part": .object(part)]))
                } else {
                    let text = part["text"] ?? ""
                    append(base.merging([
                        "type": "response.content_part.added",
                        "part": ["type": "output_text", "text": "", "annotations": [], "logprobs": []],
                    ]))
                    append(base.merging(["type": "response.output_text.delta", "delta": text, "logprobs": []]))
                    append(base.merging(["type": "response.output_text.done", "text": text, "logprobs": []]))
                    append(base.merging(["type": "response.content_part.done", "part": .object(part)]))
                }
            }
        default:
            break
        }
        append(["type": "response.output_item.done", "item": .object(item), "output_index": .int(outputIndex)])
    }
    append(["type": "response.completed", "response": .object(response)])
    return events
}
