import Testing
@testable import Agents

// Ported from tests/test_tool_metadata.py.
@Suite struct ToolMetadataTests {
    private final class DummyComputer: Computer {
        func screenshot() async throws -> String { "" }
        func click(x: Int, y: Int, button: Button) async throws {}
        func doubleClick(x: Int, y: Int) async throws {}
        func scroll(x: Int, y: Int, scrollX: Int, scrollY: Int) async throws {}
        func type(text: String) async throws {}
        func wait() async throws {}
        func move(x: Int, y: Int) async throws {}
        func keypress(keys: [String]) async throws {}
        func drag(path: [(x: Int, y: Int)]) async throws {}
    }

    private struct DummyEditor: ApplyPatchEditor {
        func createFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? { nil }
        func updateFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? { nil }
        func deleteFile(_ operation: ApplyPatchOperation) async throws -> ApplyPatchResult? { nil }
    }

    @Test func testToolNameProperties() throws {
        let dummyComputer = DummyComputer()

        #expect(FileSearchTool(vectorStoreIds: []).name == "file_search")
        #expect(WebSearchTool().name == "web_search")
        #expect(ComputerTool(computer: dummyComputer).name == "computer_use_preview")
        #expect(ComputerTool(computer: dummyComputer).traceName == "computer")
        #expect(HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "demo"]).name == "hosted_mcp")
        #expect(CodeInterpreterTool(toolConfig: ["type": "code_interpreter", "container": "python"]).name == "code_interpreter")
        #expect(ImageGenerationTool(toolConfig: ["type": "image_generation", "model": "gpt-image-1"]).name == "image_generation")
        #expect(LocalShellTool { _ in "ok" }.name == "local_shell")
        let shellTool = try ShellTool(executor: { _ in .text("ok") })
        #expect(shellTool.type == "shell")
        #expect(shellTool.environment == ["type": "local"])
        #expect(try ApplyPatchTool(editor: DummyEditor()).type == "apply_patch")
    }

    @Test func testShellCommandOutputStatusProperty() {
        let output = ShellCommandOutput(outcome: ShellCallOutcome(type: "timeout"))
        #expect(output.status == "timeout")
    }

    @Test func testToolContextFromAgentContext() throws {
        let ctx = RunContextWrapper(context: ["foo": "bar"])
        let toolCall = try ToolContext.fromAgentContext(ctx, toolCallId: "123", toolCall: ["name": "demo", "arguments": "{}"])
        #expect(toolCall.toolName == "demo")
    }
}
