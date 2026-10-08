import Foundation

// Port of src/agents/tracing/span_data.py. Span data are classes because upstream mutates span
// data after creation (for example `span.span_data.output = ...`).

/// An agent span.
public final class AgentSpanData: SpanData, @unchecked Sendable {
    public var name: String
    public var handoffs: [String]?
    public var tools: [String]?
    public var outputType: String?
    public var metadata: JSONObject?

    public init(name: String, handoffs: [String]? = nil, tools: [String]? = nil, outputType: String? = nil, metadata: JSONObject? = nil) {
        self.name = name
        self.handoffs = handoffs
        self.tools = tools
        self.outputType = outputType
        self.metadata = metadata
    }

    public var type: String { "agent" }

    public func export() -> JSONObject {
        [
            "type": .string(type),
            "name": .string(name),
            "handoffs": handoffs.map { .array($0.map(JSONValue.string)) } ?? .null,
            "tools": tools.map { .array($0.map(JSONValue.string)) } ?? .null,
            "output_type": outputType.map(JSONValue.string) ?? .null,
        ]
    }
}

/// A task span: one top-level Runner invocation.
public final class TaskSpanData: SpanData, @unchecked Sendable {
    public var name: String
    public var usage: JSONObject?
    public var metadata: JSONObject?

    public init(name: String, usage: JSONObject? = nil, metadata: JSONObject? = nil) {
        self.name = name
        self.usage = usage
        self.metadata = metadata
    }

    public var type: String { "task" }

    public func export() -> JSONObject {
        var data: JSONObject = ["sdk_span_type": .string(type), "name": .string(name)]
        if let usage { data["usage"] = .object(usage) }
        return ["type": "custom", "name": .string(type), "data": .object(data)]
    }
}

/// A turn span: one model turn inside a task.
public final class TurnSpanData: SpanData, @unchecked Sendable {
    public var turn: Int
    public var agentName: String
    public var usage: JSONObject?
    public var metadata: JSONObject?

    public init(turn: Int, agentName: String, usage: JSONObject? = nil, metadata: JSONObject? = nil) {
        self.turn = turn
        self.agentName = agentName
        self.usage = usage
        self.metadata = metadata
    }

    public var type: String { "turn" }

    public func export() -> JSONObject {
        var data: JSONObject = ["sdk_span_type": .string(type), "turn": .int(turn), "agent_name": .string(agentName)]
        if let usage { data["usage"] = .object(usage) }
        return ["type": "custom", "name": .string(type), "data": .object(data)]
    }
}

/// A function (tool) span.
public final class FunctionSpanData: SpanData, @unchecked Sendable {
    public var name: String
    public var input: String?
    public var output: (any Sendable)?
    public var mcpData: JSONObject?

    public init(name: String, input: String?, output: (any Sendable)?, mcpData: JSONObject? = nil) {
        self.name = name
        self.input = input
        self.output = output
        self.mcpData = mcpData
    }

    public var type: String { "function" }

    public func export() -> JSONObject {
        [
            "type": .string(type),
            "name": .string(name),
            "input": input.map(JSONValue.string) ?? .null,
            "output": output.map { .string(stringify($0)) } ?? .null,
            "mcp_data": mcpData.map(JSONValue.object) ?? .null,
        ]
    }
}

/// A generation span (one LLM call).
public final class GenerationSpanData: SpanData, @unchecked Sendable {
    public var input: [JSONObject]?
    public var output: [JSONObject]?
    public var model: String?
    public var modelConfig: JSONObject?
    public var usage: JSONObject?

    public init(input: [JSONObject]? = nil, output: [JSONObject]? = nil, model: String? = nil, modelConfig: JSONObject? = nil, usage: JSONObject? = nil) {
        self.input = input
        self.output = output
        self.model = model
        self.modelConfig = modelConfig
        self.usage = usage
    }

    public var type: String { "generation" }

    public func export() -> JSONObject {
        [
            "type": .string(type),
            "input": input.map { .array($0.map(JSONValue.object)) } ?? .null,
            "output": output.map { .array($0.map(JSONValue.object)) } ?? .null,
            "model": model.map(JSONValue.string) ?? .null,
            "model_config": modelConfig.map(JSONValue.object) ?? .null,
            "usage": usage.map(JSONValue.object) ?? .null,
        ]
    }
}

/// A response span (one Responses API call).
public final class ResponseSpanData: SpanData, @unchecked Sendable {
    public var response: ResponseObject?
    /// Not used by the OpenAI trace processors, but useful for other processors.
    public var input: RunInput?
    public var usage: JSONObject?
    var responseIdOverride: String?

    public init(response: ResponseObject? = nil, input: RunInput? = nil, usage: JSONObject? = nil) {
        self.response = response
        self.input = input
        self.usage = usage
    }

    public var type: String { "response" }

    public func export() -> JSONObject {
        [
            "type": .string(type),
            "response_id": (response?.id ?? responseIdOverride).map(JSONValue.string) ?? .null,
            "usage": usage.map(JSONValue.object) ?? .null,
        ]
    }
}

/// A handoff span.
public final class HandoffSpanData: SpanData, @unchecked Sendable {
    public var fromAgent: String?
    public var toAgent: String?

    public init(fromAgent: String?, toAgent: String?) {
        self.fromAgent = fromAgent
        self.toAgent = toAgent
    }

    public var type: String { "handoff" }

    public func export() -> JSONObject {
        [
            "type": .string(type),
            "from_agent": fromAgent.map(JSONValue.string) ?? .null,
            "to_agent": toAgent.map(JSONValue.string) ?? .null,
        ]
    }
}

/// A custom span with a name and a data property bag.
public final class CustomSpanData: SpanData, @unchecked Sendable {
    public var name: String
    public var data: JSONObject

    public init(name: String, data: JSONObject = [:]) {
        self.name = name
        self.data = data
    }

    public var type: String { "custom" }

    public func export() -> JSONObject {
        ["type": .string(type), "name": .string(name), "data": .object(data)]
    }
}

/// A guardrail span.
public final class GuardrailSpanData: SpanData, @unchecked Sendable {
    public var name: String
    public var triggered: Bool

    public init(name: String, triggered: Bool = false) {
        self.name = name
        self.triggered = triggered
    }

    public var type: String { "guardrail" }

    public func export() -> JSONObject {
        ["type": .string(type), "name": .string(name), "triggered": .bool(triggered)]
    }
}

/// A speech-to-text span.
public final class TranscriptionSpanData: SpanData, @unchecked Sendable {
    public var input: String?
    public var inputFormat: String?
    public var output: String?
    public var model: String?
    public var modelConfig: JSONObject?

    public init(input: String? = nil, inputFormat: String? = "pcm", output: String? = nil, model: String? = nil, modelConfig: JSONObject? = nil) {
        self.input = input
        self.inputFormat = inputFormat
        self.output = output
        self.model = model
        self.modelConfig = modelConfig
    }

    public var type: String { "transcription" }

    public func export() -> JSONObject {
        [
            "type": .string(type),
            "input": ["data": .string(input ?? ""), "format": inputFormat.map(JSONValue.string) ?? .null],
            "output": output.map(JSONValue.string) ?? .null,
            "model": model.map(JSONValue.string) ?? .null,
            "model_config": modelConfig.map(JSONValue.object) ?? .null,
        ]
    }
}

/// A text-to-speech span.
public final class SpeechSpanData: SpanData, @unchecked Sendable {
    public var input: String?
    public var output: String?
    public var outputFormat: String?
    public var model: String?
    public var modelConfig: JSONObject?
    public var firstContentAt: String?

    public init(input: String? = nil, output: String? = nil, outputFormat: String? = "pcm", model: String? = nil, modelConfig: JSONObject? = nil, firstContentAt: String? = nil) {
        self.input = input
        self.output = output
        self.outputFormat = outputFormat
        self.model = model
        self.modelConfig = modelConfig
        self.firstContentAt = firstContentAt
    }

    public var type: String { "speech" }

    public func export() -> JSONObject {
        [
            "type": .string(type),
            "input": input.map(JSONValue.string) ?? .null,
            "output": ["data": .string(output ?? ""), "format": outputFormat.map(JSONValue.string) ?? .null],
            "model": model.map(JSONValue.string) ?? .null,
            "model_config": modelConfig.map(JSONValue.object) ?? .null,
            "first_content_at": firstContentAt.map(JSONValue.string) ?? .null,
        ]
    }
}

/// A span grouping speech spans.
public final class SpeechGroupSpanData: SpanData, @unchecked Sendable {
    public var input: String?

    public init(input: String? = nil) {
        self.input = input
    }

    public var type: String { "speech_group" }

    public func export() -> JSONObject {
        ["type": .string(type), "input": input.map(JSONValue.string) ?? .null]
    }
}

/// An MCP list-tools span.
public final class MCPListToolsSpanData: SpanData, @unchecked Sendable {
    public var server: String?
    public var result: [String]?

    public init(server: String? = nil, result: [String]? = nil) {
        self.server = server
        self.result = result
    }

    public var type: String { "mcp_tools" }

    public func export() -> JSONObject {
        [
            "type": .string(type),
            "server": server.map(JSONValue.string) ?? .null,
            "result": result.map { .array($0.map(JSONValue.string)) } ?? .null,
        ]
    }
}
