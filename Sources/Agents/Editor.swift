import Foundation

/// The type of an apply_patch operation (`"create_file"`, `"update_file"`, `"delete_file"`).
public typealias ApplyPatchOperationType = String

/// Represents a single apply_patch editor operation requested by the model.
public struct ApplyPatchOperation: Sendable {
    public var type: ApplyPatchOperationType
    public var path: String
    public var diff: String?
    public var ctxWrapper: RunContextWrapper?
    public var moveTo: String?

    public init(type: ApplyPatchOperationType, path: String, diff: String? = nil, ctxWrapper: RunContextWrapper? = nil, moveTo: String? = nil) {
        self.type = type
        self.path = path
        self.diff = diff
        self.ctxWrapper = ctxWrapper
        self.moveTo = moveTo
    }
}

/// Optional metadata returned by editor operations.
public struct ApplyPatchResult: Sendable, Equatable {
    /// `"completed"` or `"failed"`.
    public var status: String?
    public var output: String?

    public init(status: String? = nil, output: String? = nil) {
        self.status = status
        self.output = output
    }
}

/// Host-defined editor that applies diffs on disk.
public protocol ApplyPatchEditor: Sendable {
    func createFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult?
    func updateFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult?
    func deleteFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult?
}
