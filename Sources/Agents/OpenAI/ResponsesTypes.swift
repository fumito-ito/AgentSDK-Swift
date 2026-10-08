import Foundation

// Swift replacements for the `openai.types.responses` models that upstream imports. See
// porting/decisions.md D3: every item is a `JSONObject`, and the types below are typed views that
// read and write that object. Unknown fields are preserved.

/// A Responses API input item (`ResponseInputItemParam`), which upstream models as a `TypedDict`.
public typealias TResponseInputItem = JSONObject

/// A Responses API output item (`ResponseOutputItem`).
public typealias TResponseOutputItem = JSONObject

/// A Responses API streaming event (`ResponseStreamEvent`).
public typealias TResponseStreamEvent = JSONObject

/// A typed view over a JSON object.
public protocol JSONBacked: Sendable, Equatable {
    /// The underlying JSON payload.
    var json: JSONObject { get set }
    /// Wraps `json` when it has the expected shape, mirroring `isinstance` checks upstream.
    init?(json: JSONObject)
}

extension JSONBacked {
    public static func == (lhs: Self, rhs: Self) -> Bool { lhs.json == rhs.json }
}

extension JSONObject {
    /// The `type` discriminator of a wire item, if present.
    public var type: String? { string("type") }

    /// `call_id`, falling back to `id`, as upstream does for tool items.
    public var callIdOrId: String? {
        if let callId = string("call_id"), !callId.isEmpty { return callId }
        return string("id")
    }
}

// MARK: - Output message

/// `ResponseOutputText` content part.
public struct ResponseOutputText: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.type == "output_text" else { return nil }
        self.json = json
    }

    public init(text: String, annotations: [JSONValue] = [], logprobs: [JSONValue]? = []) {
        var json: JSONObject = ["text": .string(text), "type": "output_text", "annotations": .array(annotations)]
        if let logprobs { json["logprobs"] = .array(logprobs) }
        self.json = json
    }

    public var text: String { json.string("text") ?? "" }
    public var annotations: [JSONValue] { json["annotations"]?.arrayValue ?? [] }
    public var logprobs: [JSONValue] { json["logprobs"]?.arrayValue ?? [] }
}

/// `ResponseOutputRefusal` content part.
public struct ResponseOutputRefusal: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.type == "refusal" else { return nil }
        self.json = json
    }

    public init(refusal: String) {
        json = ["refusal": .string(refusal), "type": "refusal"]
    }

    public var refusal: String { json.string("refusal") ?? "" }
}

/// `ResponseOutputMessage`.
public struct ResponseOutputMessage: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.type == "message", json.string("role") == "assistant" || json["role"] == nil else { return nil }
        self.json = json
    }

    public init(id: String, content: [JSONObject], status: String = "completed", role: String = "assistant") {
        json = [
            "id": .string(id),
            "content": .array(content.map(JSONValue.object)),
            "role": .string(role),
            "status": .string(status),
            "type": "message",
        ]
    }

    public var id: String { json.string("id") ?? "" }
    public var status: String? { json.string("status") }
    public var content: [JSONObject] { json["content"]?.arrayValue?.compactMap(\.objectValue) ?? [] }
}

// MARK: - Function calls

/// `ResponseFunctionToolCall`.
public struct ResponseFunctionToolCall: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.type == "function_call" else { return nil }
        self.json = json
    }

    public init(
        id: String? = nil,
        callId: String,
        name: String,
        arguments: String,
        namespace: String? = nil,
        status: String? = nil
    ) {
        var json = JSONObject()
        if let id { json["id"] = .string(id) }
        json["call_id"] = .string(callId)
        json["type"] = "function_call"
        json["name"] = .string(name)
        json["arguments"] = .string(arguments)
        if let namespace { json["namespace"] = .string(namespace) }
        if let status { json["status"] = .string(status) }
        self.json = json
    }

    public var id: String? { json.string("id") }
    public var callId: String { json.string("call_id") ?? "" }
    public var name: String { json.string("name") ?? "" }
    public var arguments: String { json.string("arguments") ?? "" }
    public var namespace: String? { json.string("namespace") }
    public var status: String? { json.string("status") }
    /// The program caller relationship (programmatic tool calling), when present.
    public var caller: JSONValue? { json["caller"] }
}

/// `ResponseCustomToolCall`.
public struct ResponseCustomToolCall: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.type == "custom_tool_call" else { return nil }
        self.json = json
    }

    public var callId: String { json.string("call_id") ?? "" }
    public var name: String { json.string("name") ?? "" }
    public var input: String { json.string("input") ?? "" }
}

// MARK: - Reasoning

/// `ResponseReasoningItem`.
public struct ResponseReasoningItem: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.type == "reasoning" else { return nil }
        self.json = json
    }

    public init(id: String, summary: [String] = [], content: [String]? = nil, encryptedContent: String? = nil) {
        var json: JSONObject = [
            "id": .string(id),
            "summary": .array(summary.map { ["text": .string($0), "type": "summary_text"] }),
            "type": "reasoning",
        ]
        if let content {
            json["content"] = .array(content.map { ["text": .string($0), "type": "reasoning_text"] })
        }
        if let encryptedContent { json["encrypted_content"] = .string(encryptedContent) }
        self.json = json
    }

    public var id: String { json.string("id") ?? "" }
    public var summary: [JSONObject] { json["summary"]?.arrayValue?.compactMap(\.objectValue) ?? [] }
    public var content: [JSONObject]? { json["content"]?.arrayValue?.compactMap(\.objectValue) }
}

// MARK: - Hosted tool calls

/// `ResponseComputerToolCall`.
public struct ResponseComputerToolCall: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.type == "computer_call" else { return nil }
        self.json = json
    }

    public var id: String? { json.string("id") }
    public var callId: String { json.string("call_id") ?? "" }
    public var action: JSONObject? { json["action"]?.objectValue }
    public var actions: [JSONObject]? { json["actions"]?.arrayValue?.compactMap(\.objectValue) }
    public var pendingSafetyChecks: [JSONObject] {
        json["pending_safety_checks"]?.arrayValue?.compactMap(\.objectValue) ?? []
    }
}

/// `LocalShellCall`.
public struct LocalShellCall: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.type == "local_shell_call" else { return nil }
        self.json = json
    }

    public var id: String? { json.string("id") }
    public var callId: String { json.string("call_id") ?? "" }
    public var action: JSONObject { json["action"]?.objectValue ?? [:] }
}

/// `ResponseFunctionShellToolCall` (`shell_call`).
public struct ResponseFunctionShellToolCall: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.type == "shell_call" else { return nil }
        self.json = json
    }

    public var callId: String { json.string("call_id") ?? json.string("id") ?? "" }
    public var action: JSONObject { json["action"]?.objectValue ?? [:] }
}

/// `ResponseApplyPatchToolCall` (`apply_patch_call`).
public struct ResponseApplyPatchToolCall: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.type == "apply_patch_call" else { return nil }
        self.json = json
    }

    public var callId: String { json.string("call_id") ?? "" }
    public var operation: JSONObject? { json["operation"]?.objectValue }
}

/// `McpApprovalRequest`.
public struct McpApprovalRequest: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        guard json.type == "mcp_approval_request" else { return nil }
        self.json = json
    }

    public init(id: String, name: String, arguments: String, serverLabel: String) {
        json = [
            "id": .string(id),
            "arguments": .string(arguments),
            "name": .string(name),
            "server_label": .string(serverLabel),
            "type": "mcp_approval_request",
        ]
    }

    public var id: String { json.string("id") ?? "" }
    public var name: String { json.string("name") ?? "" }
    public var arguments: String { json.string("arguments") ?? "" }
    public var serverLabel: String { json.string("server_label") ?? "" }
}

/// Item types that represent hosted or local tool calls (upstream `ToolCallItemTypes`).
public let toolCallItemTypes: Set<String> = [
    "function_call", "computer_call", "file_search_call", "web_search_call", "mcp_call",
    "code_interpreter_call", "image_generation_call", "local_shell_call", "program",
    "custom_tool_call", "shell_call", "apply_patch_call",
]

// MARK: - Response object

/// `openai.types.responses.Response`.
public struct ResponseObject: JSONBacked {
    public var json: JSONObject

    public init?(json: JSONObject) {
        self.json = json
    }

    public init(
        id: String?,
        output: [TResponseOutputItem],
        usage: JSONObject?,
        status: String = "completed",
        model: String = "scripted-model"
    ) {
        json = [
            "id": id.map(JSONValue.string) ?? .null,
            "created_at": 0,
            "model": .string(model),
            "object": "response",
            "output": .array(output.map(JSONValue.object)),
            "tool_choice": "none",
            "tools": [],
            "top_p": nil,
            "parallel_tool_calls": false,
            "status": .string(status),
            "usage": usage.map(JSONValue.object) ?? .null,
        ]
    }

    public var id: String? { json.string("id") }
    public var output: [TResponseOutputItem] { json["output"]?.arrayValue?.compactMap(\.objectValue) ?? [] }
    public var usage: JSONObject? { json["usage"]?.objectValue }
    public var status: String? { json.string("status") }
}
