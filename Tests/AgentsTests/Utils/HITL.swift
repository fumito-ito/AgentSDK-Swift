import Foundation
@testable import Agents

// Port of the payload builders in tests/utils/hitl.py.

/// Builds a `shell_call` payload with optional overrides.
func makeShellCall(_ callId: String, idValue: String? = nil, commands: [String]? = nil, status: String = "in_progress") -> TResponseOutputItem {
    [
        "type": "shell_call",
        "id": .string(idValue ?? callId),
        "call_id": .string(callId),
        "status": .string(status),
        "action": ["type": "exec", "commands": .array((commands ?? ["echo test"]).map(JSONValue.string)), "timeout_ms": 1000],
    ]
}
