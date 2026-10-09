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

/// Editor that records operations for testing.
final class RecordingEditor: ApplyPatchEditor, @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [ApplyPatchOperation] = []

    var operations: [ApplyPatchOperation] { lock.locked { recorded } }

    private func record(_ operation: ApplyPatchOperation) {
        lock.locked { recorded.append(operation) }
    }

    func createFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? {
        record(operation)
        return ApplyPatchResult(status: "completed", output: "Created \(operation.path)")
    }

    func updateFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? {
        record(operation)
        return ApplyPatchResult(status: "completed", output: "Updated \(operation.path)")
    }

    func deleteFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? {
        record(operation)
        return ApplyPatchResult(status: "completed", output: "Deleted \(operation.path)")
    }
}
