import Foundation
import Testing
@testable import Agents

// Ported from tests/models/test_openai_responses_converter.py.
//
// Swift: `openai.omit` maps to `nil`. These upstream cases are skipped because the Swift types
// rule them out at compile time: non-integer `max_num_results` values, pydantic web-search filter
// models (`model_extra`), positional `WebSearchTool` arguments, and the `agents.__all__` export
// check.
@Suite struct OpenAIResponsesConverterTests {
    private final class DummyComputer: Computer {
        var environment: Environment? { "mac" }
        var dimensions: (width: Int, height: Int)? { (800, 600) }
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

    private struct ValueArgs: Codable, Sendable {
        var value: String
    }

    private func functionTool(_ name: String, deferLoading: Bool = false) throws -> FunctionTool {
        try FunctionTool.create(name: name, argumentsType: ValueArgs.self, deferLoading: deferLoading) { _, args in args.value }
    }

    private func crmTools(_ tools: [FunctionTool], description: String = "CRM tools") throws -> [FunctionTool] {
        try toolNamespace(name: "crm", description: description, tools: tools)
    }

    private func deferredHostedMCP() -> HostedMCPTool {
        HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "crm_server", "server_url": "https://example.com/mcp", "defer_loading": true])
    }

    private func toolChoice(_ value: String, tools: [any Tool]? = nil, handoffs: [Handoff]? = nil, model: String? = nil) throws -> JSONValue? {
        try ResponsesConverter.convertToolChoice(ToolChoice(value), tools: tools, handoffs: handoffs, model: model)
    }

    private func convert(
        _ tools: [any Tool],
        handoffs: [Handoff] = [],
        allowOpaqueToolSearchSurface: Bool = false,
        model: String? = nil,
        toolChoice: ToolChoice? = nil
    ) throws -> ConvertedTools {
        try ResponsesConverter.convertTools(
            tools, handoffs, allowOpaqueToolSearchSurface: allowOpaqueToolSearchSurface, model: model, toolChoice: toolChoice
        )
    }

    private func functionPayload(_ tool: FunctionTool, deferLoading: Bool = false) -> JSONObject {
        var payload: JSONObject = [
            "type": "function", "name": .string(tool.name), "description": .string(tool.description),
            "parameters": .object(tool.paramsJSONSchema), "strict": true,
        ]
        if deferLoading { payload["defer_loading"] = true }
        return payload
    }

    private let previewComputerPayload: JSONObject = [
        "type": "computer_use_preview", "environment": "mac", "display_width": 800, "display_height": 600,
    ]

    // MARK: - Tool choice

    @Test func testConvertToolChoiceStandardValues() throws {
        #expect(try ResponsesConverter.convertToolChoice(nil, tools: nil, handoffs: nil, model: nil) == nil)
        #expect(try toolChoice("auto") == "auto")
        #expect(try toolChoice("required") == "required")
        #expect(try toolChoice("none") == "none")
        // Special tool types are represented as dicts of type only.
        #expect(try toolChoice("file_search") == ["type": "file_search"])
        #expect(try toolChoice("web_search_preview") == ["type": "web_search_preview"])
        // Arbitrary string should be interpreted as a function name.
        #expect(try toolChoice("my_function") == ["type": "function", "name": "my_function"])
    }

    @Test func testConvertToolChoiceComputerVariantsFollowEffectiveModel() throws {
        let tools: [any Tool] = [ComputerTool(computer: DummyComputer())]

        #expect(try toolChoice("computer", tools: tools, model: "gpt-5.4") == ["type": "computer"])
        #expect(try toolChoice("computer_use", tools: tools, model: "gpt-5.4") == ["type": "computer"])
        #expect(try toolChoice("computer_use_preview", tools: tools, model: "gpt-5.4") == ["type": "computer"])
        #expect(try toolChoice("computer_use_preview", tools: tools, model: "computer-use-preview") == ["type": "computer_use_preview"])
        #expect(try toolChoice("computer", tools: tools, model: "computer-use-preview") == ["type": "computer_use_preview"])
        #expect(try toolChoice("computer_use", tools: tools, model: "computer-use-preview") == ["type": "computer_use_preview"])
        #expect(try toolChoice("computer_use", tools: tools, model: nil) == ["type": "computer"])
        #expect(try toolChoice("computer", tools: tools, model: nil) == ["type": "computer"])
    }

    @Test func testConvertToolChoiceAllowsFunctionNamedComputerWithoutComputerTool() throws {
        #expect(try toolChoice("computer", tools: [functionTool("computer")]) == ["type": "function", "name": "computer"])
        #expect(try toolChoice("computer_use", tools: [functionTool("computer_use")]) == ["type": "function", "name": "computer_use"])
    }

    @Test func testConvertToolChoiceBuiltinShellAndApplyPatch() throws {
        let shellTool = try ShellTool(executor: { _ in .text("ok") })
        let applyPatchTool = try ApplyPatchTool(editor: DummyEditor())

        #expect(try toolChoice("shell", tools: [shellTool]) == ["type": "shell"])
        #expect(try toolChoice("apply_patch", tools: [applyPatchTool]) == ["type": "apply_patch"])
    }

    @Test func testConvertToolChoiceAllowsFunctionNamedToolSearch() throws {
        #expect(try toolChoice("tool_search", tools: [functionTool("tool_search")]) == ["type": "function", "name": "tool_search"])
    }

    @Test func testConvertToolChoiceRejectsHostedToolSearchChoice() throws {
        let deferredTool = try functionTool("lookup_weather", deferLoading: true)
        expectError(UserError.self, containing: "ToolSearchTool()") {
            _ = try toolChoice("tool_search", tools: [deferredTool, ToolSearchTool()])
        }
    }

    @Test func testConvertToolChoiceRejectsToolSearchWithoutMatchingDefinition() throws {
        let namespacedTool = try crmTools([functionTool("lookup_weather")])[0]
        expectError(UserError.self, containing: "requires ToolSearchTool() or a real top-level function tool named `tool_search`") {
            _ = try toolChoice("tool_search", tools: [namespacedTool])
        }
    }

    @Test func testConvertToolChoiceAllowsFunctionNamedToolSearchWithHostedToolSearch() throws {
        let tools: [any Tool] = [try functionTool("tool_search"), try functionTool("lookup_weather", deferLoading: true), ToolSearchTool()]
        #expect(try toolChoice("tool_search", tools: tools) == ["type": "function", "name": "tool_search"])
    }

    @Test func testConvertToolChoiceRequiredAllowsEagerNamespaceTools() throws {
        // Ports the _without_tool_search and _with_tool_search variants.
        let tools = try crmTools([functionTool("lookup_account")])
        #expect(try toolChoice("required", tools: tools) == "required")
        #expect(try toolChoice("required", tools: tools + [ToolSearchTool()]) == "required")
    }

    @Test func testConvertToolChoiceRequiredRejectsDeferredFunctionTools() throws {
        let tools: [any Tool] = [try functionTool("lookup_account", deferLoading: true)]
        expectError(UserError.self, containing: "ToolSearchTool()") { _ = try toolChoice("required", tools: tools) }
    }

    @Test func testConvertToolChoiceRequiredAllowsDeferredFunctionToolsWithToolSearch() throws {
        let tools: [any Tool] = [try functionTool("lookup_account", deferLoading: true), ToolSearchTool()]
        #expect(try toolChoice("required", tools: tools) == "required")
    }

    @Test func testConvertToolChoiceRequiredAllowsDeferredHostedMCPToolsWithToolSearch() throws {
        #expect(try toolChoice("required", tools: [deferredHostedMCP(), ToolSearchTool()]) == "required")
    }

    @Test func testConvertToolChoiceAllowsQualifiedNamespacedFunctionTools() throws {
        let namespacedTool = try crmTools([functionTool("lookup_account")])[0]
        #expect(try toolChoice("crm.lookup_account", tools: [namespacedTool]) == ["type": "function", "name": "crm.lookup_account"])
    }

    @Test func testConvertToolChoiceRejectsNamespaceWrapperAndBareInnerName() throws {
        let namespacedTool = try crmTools([functionTool("lookup_account")])[0]
        expectError(UserError.self, containing: "tool_namespace()") { _ = try toolChoice("lookup_account", tools: [namespacedTool]) }
        expectError(UserError.self, containing: "tool_namespace()") { _ = try toolChoice("crm", tools: [namespacedTool]) }
    }

    @Test func testConvertToolChoiceAllowsTopLevelFunctionWithNamespacedToolsPresent() throws {
        let tools: [any Tool] = [try functionTool("lookup_account"), try crmTools([functionTool("lookup_account")])[0]]
        #expect(try toolChoice("lookup_account", tools: tools) == ["type": "function", "name": "lookup_account"])
    }

    @Test func testConvertToolChoiceAllowsHandoffWithNamespacedFunctionNameClash() throws {
        let namespacedTool = try crmTools([functionTool("lookup_account")])[0]
        let transferHandoff = handoff(Agent(name: "specialist"), toolNameOverride: "lookup_account")
        #expect(try toolChoice("lookup_account", tools: [namespacedTool], handoffs: [transferHandoff]) == ["type": "function", "name": "lookup_account"])
    }

    @Test func testConvertToolChoiceRejectsDeferredOnlyFunctionTools() throws {
        let deferredTool = try functionTool("lookup_account", deferLoading: true)
        expectError(UserError.self, containing: "deferred-loading function tools") { _ = try toolChoice("lookup_account", tools: [deferredTool]) }
    }

    @Test func testConvertToolChoiceAllowsVisibleTopLevelFunctionWithDeferredPeer() throws {
        let tools: [any Tool] = [try functionTool("lookup_account"), try functionTool("lookup_account", deferLoading: true)]
        #expect(try toolChoice("lookup_account", tools: tools) == ["type": "function", "name": "lookup_account"])
    }

    // MARK: - Response format

    private struct OutModel: Codable, Sendable {
        var foo: Int
        var bar: String
    }

    @Test func testGetResponseFormatPlainTextAndJSONSchema() throws {
        // Default output (nil) and an explicit plain-text schema are considered plain text.
        #expect(try ResponsesConverter.getResponseFormat(nil) == nil)
        #expect(try ResponsesConverter.getResponseFormat(AgentOutputSchema(String.self)) == nil)

        let outSchema = try AgentOutputSchema(OutModel.self)
        let format = try #require(try ResponsesConverter.getResponseFormat(outSchema))
        let inner = try #require(format["format"]?.objectValue)
        #expect(inner.string("type") == "json_schema")
        #expect(inner.string("name") == "final_output")
        #expect(inner["schema"]?.objectValue != nil)
        // Should include a strict flag matching the schema's strictness setting.
        #expect(inner["strict"] == .bool(outSchema.isStrictJSONSchema()))
    }
}

// MARK: - Tool conversion

extension OpenAIResponsesConverterTests {
    @Test func testConvertToolsBasicTypesAndIncludes() throws {
        let toolFn = try functionTool("fn")
        let fileTool = FileSearchTool(vectorStoreIds: ["vs1"], maxNumResults: 3, includeSearchResults: true)
        let webTool = WebSearchTool(userLocation: nil, searchContextSize: "high")
        let compTool = ComputerTool(computer: DummyComputer())
        let tools: [any Tool] = [toolFn, fileTool, webTool, compTool]

        let converted = try convert(tools, model: "gpt-5.4")

        // The includes list should have exactly the include for file search.
        #expect(converted.includes == ["file_search_call.results"])
        #expect(converted.tools.count == 4)
        let types = converted.tools.compactMap { $0.string("type") }
        #expect(Set(types) == ["function", "file_search", "web_search", "computer"])
        let fileParams = try #require(converted.tools.first { $0.string("type") == "file_search" })
        #expect(fileParams["max_num_results"] == 3)
        #expect(fileParams["vector_store_ids"] == ["vs1"])
        let webParams = try #require(converted.tools.first { $0.string("type") == "web_search" })
        #expect(webParams["user_location"] == .null)
        #expect(webParams.string("search_context_size") == "high")
        #expect(webParams["external_web_access"] == nil)
        #expect(webParams["search_content_types"] == nil)
        #expect(webParams["image_settings"] == nil)
        // Verify computer tool uses the GA built-in tool payload.
        #expect(converted.tools.first { $0.string("type") == "computer" } == ["type": "computer"])
        let fnParams = try #require(converted.tools.first { $0.string("type") == "function" })
        #expect(fnParams.string("name") == toolFn.name)
        #expect(fnParams.string("description") == toolFn.description)

        // Only one computer tool should be allowed.
        expectError(UserError.self) { _ = try convert([compTool, compTool]) }
    }

    @Test(arguments: [1, 50])
    func testConvertFileSearchToolPreservesSupportedResultLimits(maxNumResults: Int) throws {
        let converted = try convert([FileSearchTool(vectorStoreIds: ["vs1"], maxNumResults: maxNumResults)])
        #expect(converted.tools.first { $0.string("type") == "file_search" }?["max_num_results"] == .int(maxNumResults))
    }

    @Test(arguments: [nil, 0] as [Int?])
    func testConvertFileSearchToolOmitsProviderDefaultResultLimit(maxNumResults: Int?) throws {
        let converted = try convert([FileSearchTool(vectorStoreIds: ["vs1"], maxNumResults: maxNumResults)])
        #expect(converted.tools.first { $0.string("type") == "file_search" }?["max_num_results"] == nil)
    }

    @Test(arguments: [-1, 51])
    func testConvertFileSearchToolRejectsUnsupportedResultLimits(maxNumResults: Int) {
        expectError(UserError.self, containing: "max_num_results must be zero, an integer between 1 and 50") {
            _ = try convert([FileSearchTool(vectorStoreIds: ["vs1"], maxNumResults: maxNumResults)])
        }
    }

    @Test func testConvertToolsIncludesWebSearchContentTypesAndImageSettings() throws {
        let webTool = WebSearchTool(searchContentTypes: ["text", "image"], imageSettings: ["max_results": 3, "caption": true])

        let converted = try convert([webTool], model: "gpt-5.6")

        // Image results arrive through the web_search_call.results include.
        #expect(converted.includes == ["web_search_call.results"])
        let expected: JSONObject = [
            "type": "web_search", "filters": nil, "user_location": nil, "search_context_size": "medium",
            "search_content_types": ["text", "image"], "image_settings": ["max_results": 3, "caption": true],
        ]
        #expect(converted.tools == [expected])
    }

    @Test func testConvertToolsImageOnlyUsesDefaultImageSettings() throws {
        let converted = try convert([WebSearchTool(searchContentTypes: ["image"])])

        #expect(converted.includes == ["web_search_call.results"])
        #expect(converted.tools[0]["search_content_types"] == ["image"])
        #expect(converted.tools[0]["image_settings"] == nil)
    }

    @Test func testConvertToolsTextOnlyContentTypesAddsNoInclude() throws {
        let converted = try convert([WebSearchTool(searchContentTypes: ["text"])], model: "gpt-5.6")

        #expect(converted.includes.isEmpty)
        #expect(converted.tools[0]["search_content_types"] == ["text"])
    }

    @Test func testConvertToolsIncludesExplicitFalseExternalWebAccess() throws {
        // Also covers test_web_search_tool_preserves_existing_positional_parameters.
        let converted = try convert([WebSearchTool(externalWebAccess: false)], model: "gpt-5.4")

        #expect(converted.includes.isEmpty)
        let expected: JSONObject = ["type": "web_search", "filters": nil, "user_location": nil, "search_context_size": "medium", "external_web_access": false]
        #expect(converted.tools == [expected])
    }

    @Test func testWebSearchFiltersPreserveExistingProviderPayload() throws {
        let filters: JSONObject = ["allowed_domains": ["example.com"]]

        let converted = try convert([WebSearchTool(filters: filters)], model: "gpt-5.4")

        let expected: JSONObject = ["type": "web_search", "filters": .object(filters), "user_location": nil, "search_context_size": "medium"]
        #expect(converted.tools == [expected])
    }

    @Test func testWebSearchFiltersPreserveOpenAIForwardCompatibleFields() throws {
        let converted = try convert([WebSearchTool(filters: ["future_filter": ["example.com"]])])
        #expect(converted.tools[0]["filters"] == ["future_filter": ["example.com"]])
    }

    @Test func testConvertToolsUsesPreviewComputerPayloadForPreviewModel() throws {
        let converted = try convert([ComputerTool(computer: DummyComputer())], model: "computer-use-preview")
        #expect(converted.tools == [previewComputerPayload])
    }

    @Test func testConvertToolsPromptManagedComputerDefaultsToPreviewPayload() throws {
        let converted = try convert([ComputerTool(computer: DummyComputer())], model: nil)
        #expect(converted.tools == [previewComputerPayload])
    }

    @Test func testConvertToolsShellLocalEnvironment() throws {
        let converted = try convert([ShellTool(executor: { _ in .text("ok") })])

        #expect(converted.tools == [["type": "shell", "environment": ["type": "local"]]])
        #expect(converted.includes.isEmpty)
    }

    @Test func testConvertToolsShellContainerReferenceEnvironment() throws {
        let environment: JSONObject = ["type": "container_reference", "container_id": "cntr_123"]
        let converted = try convert([ShellTool(environment: environment)])
        #expect(converted.tools == [["type": "shell", "environment": .object(environment)]])
    }

    @Test func testConvertToolsShellContainerAutoEnvironment() throws {
        let environment: JSONObject = [
            "type": "container_auto",
            "file_ids": ["file-123"],
            "memory_limit": "1g",
            "network_policy": [
                "type": "allowlist",
                "allowed_domains": ["example.com"],
                "domain_secrets": [["domain": "example.com", "name": "TOKEN", "value": "secret"]],
            ],
            "skills": [
                ["type": "skill_reference", "skill_id": "skill_123", "version": "latest"],
                [
                    "type": "inline", "name": "csv-workbench", "description": "Analyze CSV files.",
                    "source": ["type": "base64", "media_type": "application/zip", "data": "ZmFrZS16aXA="],
                ],
            ],
        ]

        let converted = try convert([ShellTool(environment: environment)])

        #expect(converted.tools == [["type": "shell", "environment": .object(environment)]])
    }

    @Test func testConvertToolsShellEnvironmentPassesThroughUnknownFields() throws {
        let environment: JSONObject = [
            "type": "container_auto",
            "network_policy": ["type": "future_mode", "allowed_domains": ["example.com"], "some_new_field": "keep-me"],
        ]
        let converted = try convert([ShellTool(environment: environment)])
        #expect(converted.tools == [["type": "shell", "environment": .object(environment)]])
    }
}

// MARK: - Tool search and namespaces

extension OpenAIResponsesConverterTests {
    @Test func testConvertToolsToolSearchAndNamespaces() throws {
        let eagerTool = try functionTool("get_customer_profile")
        let deferredTool = try functionTool("list_open_orders", deferLoading: true)

        let converted = try convert(try crmTools([eagerTool, deferredTool], description: "CRM tools for customer lookups.") + [ToolSearchTool()])

        #expect(converted.includes.isEmpty)
        let namespace: JSONObject = [
            "type": "namespace", "name": "crm", "description": "CRM tools for customer lookups.",
            "tools": [.object(functionPayload(eagerTool)), .object(functionPayload(deferredTool, deferLoading: true))],
        ]
        #expect(converted.tools == [namespace, ["type": "tool_search"]])
    }

    @Test func testConvertToolsTopLevelDeferredFunctionRequiresToolSearch() throws {
        let deferredTool = try functionTool("get_weather", deferLoading: true)
        expectError(UserError.self, containing: "ToolSearchTool()") { _ = try convert([deferredTool]) }
    }

    @Test func testConvertToolsRejectsToolSearchWithoutDeferredFunction() throws {
        let eagerTool = try functionTool("get_weather")
        expectError(UserError.self, containing: "ToolSearchTool() requires at least one searchable Responses surface") {
            _ = try convert([eagerTool, ToolSearchTool()])
        }
    }

    @Test func testConvertToolsAllowsPromptManagedToolSearchWithoutLocalSurface() throws {
        let converted = try convert([ToolSearchTool()], allowOpaqueToolSearchSurface: true)
        #expect(converted.tools == [["type": "tool_search"]])
    }

    @Test func testConvertToolsRejectsDuplicateToolSearchTools() throws {
        let deferredTool = try functionTool("get_weather", deferLoading: true)
        expectError(UserError.self, containing: "Only one ToolSearchTool() is allowed") {
            _ = try convert([deferredTool, ToolSearchTool(), ToolSearchTool()])
        }
    }

    @Test func testConvertToolsTopLevelDeferredFunctionWithToolSearch() throws {
        let deferredTool = try functionTool("get_weather", deferLoading: true)

        let converted = try convert([deferredTool, ToolSearchTool()])

        #expect(converted.tools == [functionPayload(deferredTool, deferLoading: true), ["type": "tool_search"]])
    }

    private var queryParameters: JSONValue { ["type": "object", "properties": ["query": ["type": "string"]], "required": ["query"]] }

    @Test func testConvertToolsPreservesToolSearchConfigFields() throws {
        let deferredTool = try functionTool("get_weather", deferLoading: true)
        let toolSearch = ToolSearchTool(description: "Search deferred tools on the server.", execution: "server", parameters: queryParameters)

        let converted = try convert([deferredTool, toolSearch])

        let expected: JSONObject = [
            "type": "tool_search", "description": "Search deferred tools on the server.", "execution": "server", "parameters": queryParameters,
        ]
        #expect(converted.tools.last == expected)
    }

    @Test func testConvertToolsAllowsClientExecutedToolSearchForManualFlows() throws {
        let deferredTool = try functionTool("get_weather", deferLoading: true)

        let converted = try convert([deferredTool, ToolSearchTool(execution: "client", parameters: queryParameters)])

        #expect(converted.tools.last == ["type": "tool_search", "execution": "client", "parameters": queryParameters])
    }

    @Test func testConvertToolsNamespaceOnlyTools() throws {
        // Ports _namespace_only_allows_eager_namespaces_without_tool_search and
        // _allows_tool_search_with_namespace_only_tools.
        let crmTool = try functionTool("lookup_account")
        let namespace: JSONObject = [
            "type": "namespace", "name": "crm", "description": "CRM tools", "tools": [.object(functionPayload(crmTool))],
        ]

        #expect(try convert(try crmTools([crmTool])).tools == [namespace])
        #expect(try convert(try crmTools([crmTool]) + [ToolSearchTool()]).tools == [namespace, ["type": "tool_search"]])
    }

    @Test func testConvertToolsDeferredHostedMCPRequiresToolSearch() {
        expectError(UserError.self, containing: "ToolSearchTool()") { _ = try convert([deferredHostedMCP()]) }
    }

    @Test func testConvertToolsDeferredHostedMCPWithToolSearch() throws {
        let converted = try convert([deferredHostedMCP(), ToolSearchTool()])
        #expect(converted.tools == [deferredHostedMCP().toolConfig, ["type": "tool_search"]])
    }

    @Test func testConvertToolsRejectsReservedSameNameNamespaceShape() throws {
        let invalidTool = try functionTool("lookup_account")
        invalidTool.toolNamespace = "lookup_account"
        invalidTool.toolNamespaceDescription = "Same-name namespace"

        expectError(UserError.self, containing: "synthetic namespace `lookup_account.lookup_account`") {
            _ = try convert([invalidTool, ToolSearchTool()])
        }
    }

    @Test func testConvertToolsRejectsQualifiedNameCollisionWithDottedTopLevelTool() throws {
        let dottedTopLevelTool = try functionTool("crm.lookup_account")
        let namespacedTool = try crmTools([functionTool("lookup_account")])[0]

        expectError(UserError.self, containing: "qualified name `crm.lookup_account`") {
            _ = try convert([dottedTopLevelTool, namespacedTool, ToolSearchTool()])
        }
    }

    @Test func testConvertToolsRejectsDuplicateDeferredTopLevelNames() throws {
        let first = try functionTool("lookup_account", deferLoading: true)
        let second = try functionTool("lookup_account", deferLoading: true)

        expectError(UserError.self, containing: "deferred top-level tool name `lookup_account`") {
            _ = try convert([first, second, ToolSearchTool()])
        }
    }

    @Test func testConvertToolsAllowsDottedNonFunctionToolNameWithNamespacedFunction() throws {
        let shellTool = try ShellTool(executor: { _ in .text("ok") }, name: "crm.lookup_account")
        let namespacedTool = try crmTools([functionTool("lookup_account")])[0]

        let converted = try convert([shellTool, namespacedTool])

        #expect(converted.tools.count == 2)
        let namespace = try #require(converted.tools.first { $0.string("type") == "namespace" })
        let shell = try #require(converted.tools.first { $0.string("type") == "shell" })
        #expect(shell["environment"] == ["type": "local"])
        #expect(namespace.string("name") == "crm")
        #expect(namespace["tools"]?.arrayValue?.first?.objectValue?.string("name") == "lookup_account")
    }

    @Test func testConvertToolsIncludesHandoffs() throws {
        let agent = Agent(name: "support", handoffDescription: "Handles support")

        let converted = try convert([], handoffs: [handoff(agent)])

        #expect(converted.tools.count == 1)
        let handoffTool = converted.tools[0]
        #expect(handoffTool.string("type") == "function")
        #expect(handoffTool.string("name") == Handoff.defaultToolName(agent))
        #expect(handoffTool.string("description") == Handoff.defaultToolDescription(agent))
        // No includes for handoffs by default.
        #expect(converted.includes.isEmpty)
    }
}

// MARK: - Computer tool models

extension OpenAIResponsesConverterTests {
    private func unresolvedComputerTool() -> ComputerTool {
        ComputerTool(provider: ComputerProvider { _ in DummyComputer() })
    }

    @Test(arguments: ["gpt-5.4", "gpt-5.5"])
    func testConvertToolsAcceptsUnresolvedComputerInitializer(model: String) throws {
        #expect(try convert([unresolvedComputerTool()], model: model).tools == [["type": "computer"]])
    }

    @Test func testResolveComputerToolModelReturnsNoneWhenRequestModelIsOmitted() {
        #expect(ResponsesConverter.resolveComputerToolModel(requestModel: nil, tools: [unresolvedComputerTool()]) == nil)
    }

    @Test(arguments: ["gpt-5.4", "gpt-5.5"])
    func testConvertToolsPreviewToolChoiceUsesGAPayloadForGAModel(model: String) throws {
        let converted = try convert([unresolvedComputerTool()], model: model, toolChoice: .named("computer_use_preview"))
        #expect(converted.tools == [["type": "computer"]])
    }

    @Test func testConvertToolsPromptManagedComputerRespectsExplicitGAToolChoice() throws {
        let converted = try convert([unresolvedComputerTool()], model: nil, toolChoice: .named("computer_use"))
        #expect(converted.tools == [["type": "computer"]])
    }

    @Test func testConvertToolsPromptManagedComputerAcceptsMCPToolChoice() throws {
        let converted = try convert(
            [ComputerTool(computer: DummyComputer())], model: nil, toolChoice: .mcp(MCPToolChoice(serverLabel: "remote", name: "lookup_account"))
        )
        #expect(converted.tools == [previewComputerPayload])
    }
}
