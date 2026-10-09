import Testing
@testable import Agents

// Ported from tests/test_hitl_utils.py.
@Suite struct HITLUtilsTests {
    @Test func testRecordingEditorRecordsOperations() async throws {
        let editor = RecordingEditor()
        let operation = ApplyPatchOperation(type: "create_file", path: "file.txt")

        _ = try await editor.createFile(operation)
        _ = try await editor.updateFile(operation)
        _ = try await editor.deleteFile(operation)

        #expect(editor.operations.map(\.path) == ["file.txt", "file.txt", "file.txt"])
    }
}
