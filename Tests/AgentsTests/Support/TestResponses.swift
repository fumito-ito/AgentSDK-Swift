import Foundation
@testable import Agents

// Port of tests/test_responses.py (shared response builders for tests).

func getTextInputItem(_ content: String) -> TResponseInputItem {
    ["content": .string(content), "role": "user"]
}

func getTextMessage(_ content: String) -> TResponseOutputItem {
    [
        "id": "1",
        "type": "message",
        "role": "assistant",
        "content": [["text": .string(content), "type": "output_text", "annotations": [], "logprobs": []]],
        "status": "completed",
    ]
}

func getRefusalMessage(_ refusal: String) -> TResponseOutputItem {
    [
        "id": "1",
        "type": "message",
        "role": "assistant",
        "content": [["refusal": .string(refusal), "type": "refusal"]],
        "status": "completed",
    ]
}

func getFunctionTool(name: String? = nil, returnValue: String? = nil, hideErrors: Bool = false) -> FunctionTool {
    // swiftlint:disable:next force_try
    try! FunctionTool.create(name: name ?? "foo", failureErrorFunction: hideErrors ? .none : .default) { _ in
        returnValue ?? "result_ok"
    }
}

func getFunctionToolCall(_ name: String, arguments: String? = nil, callId: String? = nil, namespace: String? = nil) -> TResponseOutputItem {
    var item: JSONObject = [
        "id": "1",
        "call_id": .string(callId ?? "2"),
        "type": "function_call",
        "name": .string(name),
        "arguments": .string(arguments ?? ""),
    ]
    if let namespace { item["namespace"] = .string(namespace) }
    return item
}

func getHandoffToolCall(_ toAgent: Agent, overrideName: String? = nil, args: String? = nil, callId: String? = nil) -> TResponseOutputItem {
    let name = overrideName ?? Handoff.defaultToolName(toAgent)
    return getFunctionToolCall(name, arguments: args, callId: callId ?? "handoff_\(toAgent.name)")
}

func getFinalOutputMessage(_ args: String) -> TResponseOutputItem {
    getTextMessage(args)
}
