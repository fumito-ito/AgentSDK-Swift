import Foundation
import Testing
@testable import Agents

// Ported from tests/test_function_tool.py.
//
// Swift: these upstream tests are skipped because the machinery they exercise does not exist in
// Swift (porting/decisions.md D6):
// - keyword-only `tool_namespace` arguments (Swift argument labels are always required);
// - `asyncio.to_thread` dispatch and event-loop blocking of sync handlers, and the
//   sync-handler timeout restriction (every Swift tool body is `async`);
// - pydantic `Field` constraints on variadic parameters and every `**kwargs` test (Swift tools
//   decode a single `Decodable` argument type);
// - monkeypatching `default_tool_error_function`, `copy.deepcopy`, and `dataclasses.replace`
//   with private fields (Swift tools are copied with `copy()`);
// - `FunctionTool` subclass constructors, `__post_init__`, and bound-method invoker proxies
//   (`FunctionTool` is a `final class`);
// - runtime type checks for `timeout_seconds`, `timeout_behavior`, and `timeout_error_function`
//   (enforced by the Swift type system).
// JSON Schema validator assertions (`Draft202012Validator`) are replaced with assertions on the
// converted schema.
@Suite struct FunctionToolTests {
    private func makeContext(_ tool: FunctionTool, _ arguments: String, context: any Sendable = ()) -> ToolContext {
        ToolContext(context: context, toolName: tool.name, toolCallId: "1", toolArguments: arguments)
    }

    private func noopTool(
        name: String = "test",
        schema: JSONObject,
        strict: Bool = true,
        outputJSONSchema: JSONObject? = nil
    ) throws -> FunctionTool {
        try FunctionTool(
            name: name,
            description: "test",
            paramsJSONSchema: schema,
            onInvokeTool: { _, input in input },
            strictJSONSchema: strict,
            outputJSONSchema: outputJSONSchema
        )
    }

    private func nestedObjectSchema(depth: Int) -> JSONObject {
        var current: JSONObject = ["type": "object", "properties": [:]]
        for _ in 0..<depth {
            current = ["type": "object", "properties": ["child": .object(current)]]
        }
        return current
    }

    private func chainedRefSchema(depth: Int) -> JSONObject {
        var definitions = JSONObject()
        for index in 0..<depth {
            definitions["L\(index)"] = ["$ref": .string("#/$defs/L\(index + 1)")]
        }
        definitions["L\(depth)"] = ["type": "string"]
        return ["$defs": .object(definitions), "type": "object", "properties": ["value": ["$ref": "#/$defs/L0", "description": "value"]]]
    }

    private func arglessTool(name: String = "argless_function") throws -> FunctionTool {
        try FunctionTool.create(name: name) { _ in "ok" }
    }

    // MARK: - Tool namespaces

    @Test func testToolNamespaceCopiesToolsWithMetadata() throws {
        let tool = try arglessTool()

        let namespacedTools = try toolNamespace(name: "crm", description: "CRM tools", tools: [tool])

        #expect(namespacedTools.count == 1)
        #expect(namespacedTools[0] !== tool)
        #expect(namespacedTools[0].toolNamespace == "crm")
        #expect(namespacedTools[0].toolNamespaceDescription == "CRM tools")
        #expect(namespacedTools[0].qualifiedName == "crm.argless_function")
        #expect(tool.toolNamespace == nil)
        #expect(tool.qualifiedName == "argless_function")
    }

    @Test func testToolNamespaceRequiresNonEmptyDescription() throws {
        let tool = try arglessTool()
        expectError(UserError.self, containing: "non-empty description") {
            _ = try toolNamespace(name: "crm", description: nil, tools: [tool])
        }
        expectError(UserError.self, containing: "non-empty description") {
            _ = try toolNamespace(name: "crm", description: "   ", tools: [tool])
        }
    }

    @Test func testToolNamespaceRejectsReservedSameNameShape() throws {
        let tool = try arglessTool(name: "lookup_account")
        expectError(UserError.self, containing: "synthetic namespace `lookup_account.lookup_account`") {
            _ = try toolNamespace(name: "lookup_account", description: "Same-name namespace", tools: [tool])
        }
    }

    // MARK: - Function tools

    @Test func testArglessFunction() async throws {
        let tool = try arglessTool()
        #expect(tool.name == "argless_function")

        let result = try await tool.onInvokeTool(makeContext(tool, ""), "")
        #expect(result as? String == "ok")
    }

    @Test func testArglessWithContext() async throws {
        let tool = try FunctionTool.create(name: "argless_with_context") { (_: ToolContext) in "ok" }
        #expect(tool.name == "argless_with_context")

        var result = try await tool.onInvokeTool(makeContext(tool, ""), "")
        #expect(result as? String == "ok")

        // Extra JSON should not raise an error
        result = try await tool.onInvokeTool(makeContext(tool, #"{"a": 1}"#), #"{"a": 1}"#)
        #expect(result as? String == "ok")
    }

    private struct SimpleArgs: Codable, Sendable {
        var a: Int
        var b: Int

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            a = try container.decode(Int.self, forKey: .a)
            b = try container.decodeIfPresent(Int.self, forKey: .b) ?? 5
        }
    }

    private func simpleFunctionTool(
        name: String = "simple_function",
        description: String = "",
        failureErrorFunction: FailureErrorFunction = .default
    ) throws -> FunctionTool {
        try FunctionTool.create(name: name, description: description, argumentsType: SimpleArgs.self, failureErrorFunction: failureErrorFunction) { _, args in
            args.a + args.b
        }
    }

    @Test func testSimpleFunction() async throws {
        let tool = try simpleFunctionTool(failureErrorFunction: .none)
        #expect(tool.name == "simple_function")

        var result = try await tool.onInvokeTool(makeContext(tool, #"{"a": 1}"#), #"{"a": 1}"#)
        #expect(result as? Int == 6)

        result = try await tool.onInvokeTool(makeContext(tool, #"{"a": 1, "b": 2}"#), #"{"a": 1, "b": 2}"#)
        #expect(result as? Int == 3)

        // Missing required argument should raise an error
        await expectErrorAsync(ModelBehaviorError.self) {
            _ = try await tool.onInvokeTool(makeContext(tool, ""), "")
        }
    }

    private struct Foo: Codable, Sendable {
        var a: Int
        var b: Int

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            a = try container.decode(Int.self, forKey: .a)
            b = try container.decodeIfPresent(Int.self, forKey: .b) ?? 5
        }
    }

    private struct Bar: Codable, Sendable {
        var x: String
        var y: Int
    }

    private struct ComplexArgs: Codable, Sendable {
        var foo: Foo
        var bar: Bar
        var baz: String

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            foo = try container.decode(Foo.self, forKey: .foo)
            bar = try container.decode(Bar.self, forKey: .bar)
            baz = try container.decodeIfPresent(String.self, forKey: .baz) ?? "hello"
        }
    }

    private func complexArgsTool(failureErrorFunction: FailureErrorFunction = .default) throws -> FunctionTool {
        try FunctionTool.create(name: "complex_args_function", argumentsType: ComplexArgs.self, failureErrorFunction: failureErrorFunction) { _, args in
            "\(args.foo.a + args.foo.b) \(args.bar.x)\(args.bar.y) \(args.baz)"
        }
    }

    @Test func testComplexArgsFunction() async throws {
        let tool = try complexArgsTool(failureErrorFunction: .none)
        #expect(tool.name == "complex_args_function")

        var json = #"{"foo": {"a": 1, "b": 5}, "bar": {"x": "hello", "y": 10}}"#
        var result = try await tool.onInvokeTool(makeContext(tool, json), json)
        #expect(result as? String == "6 hello10 hello")

        json = #"{"foo": {"a": 1, "b": 2}, "bar": {"x": "hello", "y": 10}}"#
        result = try await tool.onInvokeTool(makeContext(tool, json), json)
        #expect(result as? String == "3 hello10 hello")

        json = #"{"foo": {"a": 1, "b": 2}, "bar": {"x": "hello", "y": 10}, "baz": "world"}"#
        result = try await tool.onInvokeTool(makeContext(tool, json), json)
        #expect(result as? String == "3 hello10 world")

        // Missing required argument should raise an error
        await expectErrorAsync(ModelBehaviorError.self) {
            _ = try await tool.onInvokeTool(makeContext(tool, #"{"foo": {"a": 1}}"#), #"{"foo": {"a": 1}}"#)
        }
    }

    @Test func testFunctionConfigOverrides() throws {
        var tool = try simpleFunctionTool(name: "custom_name")
        #expect(tool.name == "custom_name")

        tool = try simpleFunctionTool(description: "custom description")
        #expect(tool.description == "custom description")

        tool = try simpleFunctionTool(name: "custom_name", description: "custom description")
        #expect(tool.name == "custom_name")
        #expect(tool.description == "custom description")
    }

    @Test func testFuncSchemaIsStrict() throws {
        var tool = try simpleFunctionTool()
        #expect(tool.strictJSONSchema, "Should be strict by default")
        #expect(tool.paramsJSONSchema["additionalProperties"] == false)

        tool = try complexArgsTool()
        #expect(tool.strictJSONSchema, "Should be strict by default")
        #expect(tool.paramsJSONSchema["additionalProperties"] == false)
    }

    // MARK: - Manual schemas

    @Test func testManualFunctionToolNormalizesTypelessObjectSchemas() throws {
        let tool = try noopTool(schema: [
            "properties": [
                "config": ["properties": ["key": ["type": "string"]]],
                "optional": ["type": ["object", "null"], "properties": ["value": ["type": "integer"]]],
            ],
        ])

        #expect(tool.strictJSONSchema)
        let expected: JSONObject = [
            "type": "object",
            "properties": [
                "config": ["type": "object", "properties": ["key": ["type": "string"]], "additionalProperties": false, "required": ["key"]],
                "optional": [
                    "type": ["object", "null"], "properties": ["value": ["type": "integer"]], "additionalProperties": false, "required": ["value"],
                ],
            ],
            "additionalProperties": false,
            "required": ["config", "optional"],
        ]
        #expect(tool.paramsJSONSchema == expected)
    }

    @Test(arguments: zip(
        ["Value", "A B", "café", "A%20B", "a/b", "a~1b", "A+B", "A+B", "Value"],
        ["#/$defs/Value", "#/$defs/A%20B", "#/$defs/caf%C3%A9", "#/$defs/A%2520B", "#/$defs/a%7E1b", "#/$defs/a%7E01b", "#/$defs/A+B", "#/$defs/A%2BB", "#/%24defs%2FValue"]
    ))
    func testFunctionToolURIFragmentPreservesSchemaMeaning(definitionName: String, ref: String) throws {
        let schema: JSONObject = [
            "type": "object",
            "properties": ["value": ["$ref": .string(ref), "description": "value"]],
            "$defs": [definitionName: ["type": "string"]],
        ]

        let tool = try noopTool(schema: schema)

        #expect(tool.paramsJSONSchema["properties"]?.objectValue?["value"] == ["type": "string", "description": "value"])
    }

    @Test func testFunctionToolURIFragmentDoesNotSelectLiteralPercentName() throws {
        let tool = try noopTool(schema: [
            "type": "object",
            "properties": ["value": ["$ref": "#/$defs/A%20B", "description": "value"]],
            "$defs": ["A B": ["type": "string"], "A%20B": ["type": "integer"]],
        ])

        #expect(tool.paramsJSONSchema["properties"]?.objectValue?["value"]?.objectValue?.string("type") == "string")
    }

    @Test func testFunctionToolURIFragmentRejectsInvalidUTF8() {
        // Swift: upstream raises `UnicodeDecodeError`; Swift reports the invalid percent-encoding as `ValueError`.
        expectError(ValueError.self) {
            _ = try noopTool(schema: [
                "type": "object",
                "properties": ["value": ["$ref": "#/$defs/%FF", "description": "value"]],
                "$defs": ["\u{FFFD}": ["type": "string"]],
            ])
        }
    }

    @Test func testFunctionToolURIFragmentRejectsNestedResource() {
        expectError(UserError.self, containing: "nested `$id` resource") {
            _ = try noopTool(schema: [
                "type": "object",
                "properties": ["value": ["$ref": "#/$defs/A%20B", "description": "value"]],
                "$defs": ["A B": ["$id": "https://example.test/nested", "type": "string"]],
            ])
        }
    }

    @Test func testManualFunctionToolRejectsRootUnion() {
        expectError(UserError.self, containing: "root of a strict JSON schema") {
            _ = try noopTool(schema: ["anyOf": [["properties": ["value": ["type": "string"]]], ["type": "null"]]])
        }
    }

    @Test func testManualFunctionToolRejectsNestedTypelessOpenMap() {
        expectError(UserError.self, containing: "additionalProperties") {
            _ = try noopTool(schema: ["type": "object", "properties": ["metadata": ["additionalProperties": ["type": "string"]]]])
        }
    }

    private struct FunctionArgs: Codable, Sendable {
        var data: String
    }

    @Test func testManualFunctionToolCreationWorks() async throws {
        let runFunction: @Sendable (ToolContext, String) async throws -> any Sendable = { _, args in
            let parsed = try JSONValue.parse(args).decode(as: FunctionArgs.self)
            return "\(parsed.data)_done"
        }
        // Swift: the literal mirrors pydantic's `FunctionArgs.model_json_schema()`.
        let schema: JSONObject = [
            "properties": ["data": ["title": "Data", "type": "string"]], "required": ["data"], "title": "FunctionArgs", "type": "object",
        ]

        let tool = try FunctionTool(name: "test", description: "Processes extracted user data", paramsJSONSchema: schema, onInvokeTool: runFunction)

        #expect(tool.name == "test")
        #expect(tool.description == "Processes extracted user data")
        for (key, value) in schema {
            #expect(tool.paramsJSONSchema[key] == value)
        }
        #expect(tool.strictJSONSchema)
        var result = try await tool.onInvokeTool(makeContext(tool, #"{"data": "hello"}"#), #"{"data": "hello"}"#)
        #expect(result as? String == "hello_done")

        let toolNotStrict = try FunctionTool(
            name: "test", description: "Processes extracted user data", paramsJSONSchema: schema, onInvokeTool: runFunction, strictJSONSchema: false
        )

        #expect(!toolNotStrict.strictJSONSchema)
        #expect(toolNotStrict.paramsJSONSchema["additionalProperties"] == nil)
        let arguments = #"{"data": "hello", "bar": "baz"}"#
        result = try await toolNotStrict.onInvokeTool(makeContext(toolNotStrict, arguments), arguments)
        #expect(result as? String == "hello_done")
    }

    // MARK: - Failure error functions

    private func failingTool(failureErrorFunction: FailureErrorFunction = .default) throws -> FunctionTool {
        try FunctionTool.create(name: "my_func", argumentsType: SimpleArgs.self, failureErrorFunction: failureErrorFunction) { _, _ -> Int in
            throw ValueError("test")
        }
    }

    @Test func testFunctionToolDefaultErrorWorks() async throws {
        let tool = try failingTool()
        let ctx = makeContext(tool, "")
        let defaultMessage = "An error occurred while running the tool. Please try again."

        var result = try await tool.onInvokeTool(ctx, "")
        #expect(result as? String == defaultMessage)

        result = try await tool.onInvokeTool(ctx, "{}")
        #expect(result as? String == defaultMessage)

        let expected = try await defaultToolErrorFunction(ctx, ValueError("test"))
        result = try await tool.onInvokeTool(ctx, #"{"a": 1}"#)
        #expect(result as? String == expected)

        result = try await tool.onInvokeTool(ctx, #"{"a": 1, "b": 2}"#)
        #expect(result as? String == expected)
    }

    @Test(arguments: [false, true])
    func testCustomErrorFunctionWorks(async: Bool) async throws {
        // Ports both test_sync_custom_error_function_works and test_async_custom_error_function_works.
        let tool = try failingTool(failureErrorFunction: .custom { _, error in
            if async { await Task.yield() }
            return "error_\(errorTypeName(error))"
        })
        let ctx = makeContext(tool, "")

        var result = try await tool.onInvokeTool(ctx, "")
        #expect(result as? String == "error_ModelBehaviorError")

        result = try await tool.onInvokeTool(ctx, "{}")
        #expect(result as? String == "error_ModelBehaviorError")

        result = try await tool.onInvokeTool(ctx, #"{"a": 1}"#)
        #expect(result as? String == "error_ValueError")

        result = try await tool.onInvokeTool(ctx, #"{"a": 1, "b": 2}"#)
        #expect(result as? String == "error_ValueError")
    }

    @Test func testAsyncFailureErrorFunctionIsAwaited() async throws {
        let boom = try FunctionTool.create(name: "boom", failureErrorFunction: .custom { _, error in
            await Task.yield()
            return "handled:\(describeError(error))"
        }) { _ -> String in
            throw TestError("kapow")
        }

        let result = try await boom.onInvokeTool(ToolContext(context: (), toolName: boom.name, toolCallId: "boom", toolArguments: "{}"), "{}")
        #expect((result as? String)?.hasPrefix("handled:") == true)
    }

    @Test func testFailureErrorFunctionNormalizesCancelledErrorToException() async throws {
        let seenError = SharedBox<String?>(nil)
        let tool = try FunctionTool.create(name: "tool", failureErrorFunction: .custom { _, error in
            #expect(!(error is CancellationError))
            seenError.mutate { $0 = describeError(error) }
            return "handled:\(describeError(error))"
        }) { _ in "ok" }

        let result = try await maybeInvokeFunctionToolFailureErrorFunction(tool: tool, context: RunContextWrapper(), error: CancellationError())

        #expect(result == "handled:Tool execution cancelled.")
        #expect(seenError.value == "Tool execution cancelled.")
    }

    @Test func testManualFunctionToolUsesDefaultFailureErrorFunction() async throws {
        let manualTool = try FunctionTool(
            name: "manual_cancel_tool",
            description: "manual cancel",
            paramsJSONSchema: [:],
            onInvokeTool: { _, _ in throw CancellationError() }
        )

        let result = try await maybeInvokeFunctionToolFailureErrorFunction(tool: manualTool, context: RunContextWrapper(), error: CancellationError())

        #expect(result == "An error occurred while running the tool. Please try again.")
        #expect(manualTool.useDefaultFailureErrorFunction)
        #expect(resolveFunctionToolFailureErrorFunction(manualTool, context: nil) != nil)
    }

    @Test func testFailureErrorFunctionSurvivesCopy() async throws {
        // Ports test_failure_error_function_survives_dataclasses_replace and
        // test_default_failure_error_function_survives_deepcopy with `copy()`.
        let tool = try FunctionTool.create(name: "tool", failureErrorFunction: .custom { _, error in "handled:\(describeError(error))" }) { _ in "ok" }
        let copiedTool = tool.copy()
        copiedTool.name = "copied_tool"

        let result = try await maybeInvokeFunctionToolFailureErrorFunction(tool: copiedTool, context: RunContextWrapper(), error: CancellationError())
        #expect(result == "handled:Tool execution cancelled.")

        let defaultTool = try FunctionTool.create(name: "boom") { _ -> String in throw TestError("kapow") }.copy()
        let defaultResult = try await maybeInvokeFunctionToolFailureErrorFunction(tool: defaultTool, context: RunContextWrapper(), error: CancellationError())
        #expect(defaultResult == "An error occurred while running the tool. Please try again.")
        #expect(defaultTool.useDefaultFailureErrorFunction)
    }

    @Test func testCopiedFunctionToolNormalFailureUsesCopiedPolicy() async throws {
        // Ports test_replaced_function_tool_normal_failure_uses_replaced_policy and
        // test_shallow_copied_function_tool_normal_failure_uses_copied_policy.
        let originalTool = try FunctionTool.create(name: "boom") { _ -> String in throw TestError("kapow") }

        let copiedTool = originalTool.copy()
        copiedTool.name = "copied_tool"
        copiedTool.setFailureErrorFunction(.none)

        await expectErrorAsync(TestError.self, containing: "kapow") {
            _ = try await copiedTool.onInvokeTool(makeContext(copiedTool, ""), "")
        }
        let originalResult = try await originalTool.onInvokeTool(makeContext(originalTool, ""), "")
        #expect(originalResult as? String == "An error occurred while running the tool. Please try again.")
    }

    private struct EchoArgs: Codable, Sendable {
        var value: String
    }

    private func echoTool(name: String) throws -> FunctionTool {
        try FunctionTool.create(name: name, argumentsType: EchoArgs.self, failureErrorFunction: .none) { _, args in args.value }
    }

    @Test func testCopiedFunctionToolInvalidInputUsesCurrentName() async throws {
        let copiedTool = try echoTool(name: "original_tool").copy()
        copiedTool.name = "copied_tool"

        await expectErrorAsync(ModelBehaviorError.self, containing: "Invalid JSON input for tool copied_tool") {
            _ = try await copiedTool.onInvokeTool(makeContext(copiedTool, "{}"), "{}")
        }
    }

    // MARK: - Enablement

    private struct BoolCtx: Sendable {
        var enableTools: Bool
    }

    @Test func testIsEnabledBoolAndCallable() async throws {
        let disabledTool = try FunctionTool.create(name: "disabled_tool", isEnabled: false) { _ in "nope" }
        let anotherTool = try FunctionTool.create(name: "another_tool", isEnabled: .dynamic { ctx, _ in
            ctx.context(as: BoolCtx.self)?.enableTools ?? false
        }) { _ in "hi" }
        let thirdTool = try FunctionTool(
            name: "third_tool",
            description: "third tool",
            paramsJSONSchema: [:],
            onInvokeTool: { _, _ in "third" },
            isEnabled: .dynamic { ctx, _ in ctx.context(as: BoolCtx.self)?.enableTools ?? false }
        )

        let agent = Agent(name: "t", tools: [disabledTool, anotherTool, thirdTool])
        let context1 = RunContextWrapper(context: BoolCtx(enableTools: false))
        let context2 = RunContextWrapper(context: BoolCtx(enableTools: true))

        var toolsWithContext = try await agent.getAllTools(context1)
        #expect(toolsWithContext.isEmpty)

        toolsWithContext = try await agent.getAllTools(context2)
        #expect(toolsWithContext.map(\.name) == ["another_tool", "third_tool"])
    }

    @Test func testGetAllToolsCancelsSiblingEnablementChecksOnError() async throws {
        let slowStarted = AsyncEvent()
        let slowCancelled = AsyncEvent()
        let slowFinished = AsyncEvent()

        let slowTool = try FunctionTool.create(name: "slow_tool", isEnabled: .dynamic { _, _ in
            slowStarted.set()
            defer { slowFinished.set() }
            do {
                try await waitForever()
            } catch is CancellationError {
                slowCancelled.set()
                throw CancellationError()
            }
            return true
        }) { _ in "slow" }
        let failingTool = try FunctionTool.create(name: "failing_tool", isEnabled: .dynamic { _, _ in
            try await slowStarted.wait()
            throw TestError("enablement failed")
        }) { _ in "failing" }

        let agent = Agent(name: "t", tools: [slowTool, failingTool])

        await expectErrorAsync(TestError.self, containing: "enablement failed") {
            _ = try await agent.getAllTools(RunContextWrapper())
        }

        #expect(slowCancelled.isSet)
        #expect(slowFinished.isSet)
    }

    @Test func testGetAllToolsPreservesExplicitToolSearchWhenDeferredToolsAreDisabled() async throws {
        let deferredLookup = try FunctionTool.create(name: "deferred_lookup", argumentsType: NoArguments.self, isEnabled: .dynamic { ctx, _ in
            ctx.context(as: BoolCtx.self)?.enableTools ?? false
        }, deferLoading: true) { _, _ in "loaded" }
        let agent = Agent(name: "t", tools: [deferredLookup, ToolSearchTool()])

        let toolsWithDisabledContext = try await agent.getAllTools(RunContextWrapper(context: BoolCtx(enableTools: false)))
        #expect(toolsWithDisabledContext.count == 1)
        #expect(toolsWithDisabledContext[0] is ToolSearchTool)

        let toolsWithEnabledContext = try await agent.getAllTools(RunContextWrapper(context: BoolCtx(enableTools: true)))
        #expect(toolsWithEnabledContext[0] as? FunctionTool === deferredLookup)
        #expect(toolsWithEnabledContext[1] is ToolSearchTool)
    }

    private struct AccountArgs: Codable, Sendable {
        var accountId: String

        enum CodingKeys: String, CodingKey {
            case accountId = "account_id"
        }
    }

    @Test func testGetAllToolsKeepsToolSearchForNamespaceOnlyTools() async throws {
        let lookup = try FunctionTool.create(name: "lookup_account", argumentsType: AccountArgs.self) { _, args in args.accountId }
        let namespacedLookup = try toolNamespace(name: "crm", description: "CRM tools", tools: [lookup])[0]
        let agent = Agent(name: "t", tools: [namespacedLookup, ToolSearchTool()])

        let tools = try await agent.getAllTools(RunContextWrapper(context: BoolCtx(enableTools: false)))

        #expect(tools[0] as? FunctionTool === namespacedLookup)
        #expect(tools[1] is ToolSearchTool)
    }

    @Test func testGetAllToolsKeepsToolSearchForDeferredHostedMCP() async throws {
        let hostedMCP = HostedMCPTool(toolConfig: [
            "type": "mcp", "server_label": "crm_server", "server_url": "https://example.com/mcp", "defer_loading": true,
        ])
        let agent = Agent(name: "t", tools: [hostedMCP, ToolSearchTool()])

        let tools = try await agent.getAllTools(RunContextWrapper(context: BoolCtx(enableTools: false)))

        #expect((tools[0] as? HostedMCPTool)?.toolConfig == hostedMCP.toolConfig)
        #expect(tools[1] is ToolSearchTool)
    }
}

// MARK: - Schema conversion

extension FunctionToolTests {
    @Test func testFunctionToolDoesNotMutateParamsJSONSchema() throws {
        let schema: JSONObject = ["type": "object", "properties": ["x": ["type": "string"]]]
        let snapshot = schema

        let tool = try noopTool(name: "t", schema: schema)

        #expect(schema == snapshot)
        #expect(tool.paramsJSONSchema["additionalProperties"] == false)
        #expect(tool.paramsJSONSchema["required"] == ["x"])
    }

    @Test(arguments: zip(
        [["type": "object", "properties": [:]], ["type": "object"], [:], ["type": "object", "properties": [:]]] as [JSONObject],
        [true, true, true, false]
    ))
    func testFunctionToolRejectsSingleAllOfThatBroadensClosedParent(parent: JSONObject, useRef: Bool) throws {
        let entry: JSONObject = ["type": "object", "properties": ["value": ["type": "string"]]]
        var schema = parent
        schema["additionalProperties"] = false
        schema["$defs"] = ["Entry": .object(entry)]
        schema["allOf"] = [useRef ? ["$ref": "#/$defs/Entry"] : .object(entry)]
        let original = schema

        expectError(UserError.self, containing: "singleton `allOf`") {
            _ = try noopTool(name: "t", schema: schema)
        }

        #expect(schema == original)
        let nonStrictTool = try noopTool(name: "t", schema: schema, strict: false)
        #expect(nonStrictTool.paramsJSONSchema == original)
    }

    @Test(arguments: [["type": "object"], ["type": "object", "properties": [:]]] as [JSONObject])
    func testFunctionToolSingleAllOfWrapperRemainsStrict(parent: JSONObject) throws {
        var schema = parent
        schema["description"] = "wrapper"
        schema["allOf"] = [["type": "object", "properties": ["value": ["type": "string"]]]]
        let original = schema

        let tool = try noopTool(name: "t", schema: schema)

        #expect(schema == original)
        #expect(tool.paramsJSONSchema.string("description") == "wrapper")
        #expect(tool.paramsJSONSchema["required"] == ["value"])
        #expect(tool.paramsJSONSchema["additionalProperties"] == false)
        #expect(tool.paramsJSONSchema["properties"]?.objectValue?["value"] == ["type": "string"])
    }

    @Test func testFunctionToolNestedSingleAllOfRefWrapperRemainsStrict() throws {
        let schema: JSONObject = [
            "type": "object",
            "components": [
                "schemas": [
                    "Inner": ["type": "object", "properties": ["value": ["type": "string"]]],
                    "Outer": ["type": "object", "allOf": [["$ref": "#/components/schemas/Inner"]]],
                ],
            ],
            "allOf": [["$ref": "#/components/schemas/Outer"]],
        ]
        let original = schema

        let tool = try noopTool(name: "t", schema: schema)

        #expect(schema == original)
        #expect(tool.paramsJSONSchema["required"] == ["value"])
        #expect(tool.paramsJSONSchema["allOf"] == nil)
        #expect(tool.paramsJSONSchema["additionalProperties"] == false)
    }

    @Test func testFunctionToolSingleAllOfClosedEmptyObjectRemainsStrict() throws {
        let tool = try noopTool(name: "t", schema: [
            "type": "object", "properties": [:], "additionalProperties": false, "allOf": [["type": "object", "properties": [:]]],
        ])

        #expect(tool.paramsJSONSchema["additionalProperties"] == false)
        let declaresProperties = tool.paramsJSONSchema["properties"]?.objectValue.map { !$0.isEmpty } ?? false
        #expect(!declaresProperties)
        if case .array(let allOf)? = tool.paramsJSONSchema["allOf"] {
            for case .object(let branch) in allOf {
                #expect(branch["additionalProperties"] == false)
            }
        }
    }

    @Test func testFunctionToolRejectsDeepSchemaBeforeCopying() {
        // Swift: building, comparing, and releasing a 1,000-level JSON value recurses, so the
        // fixture lives on a large stack (porting/decisions.md D9).
        runOnLargeStack {
            let schema = nestedObjectSchema(depth: 1_000)

            expectError(UserError.self, containing: "too deeply nested") {
                _ = try noopTool(name: "strict_tool", schema: schema)
            }

            let nonStrictTool = try? noopTool(name: "non_strict_tool", schema: schema, strict: false)
            #expect(nonStrictTool?.paramsJSONSchema == schema)
        }
    }

    @Test func testFunctionToolRejectsDeeplyChainedRefsBeforeConversion() {
        expectError(UserError.self, containing: "too deeply nested") {
            _ = try noopTool(name: "strict_tool", schema: chainedRefSchema(depth: 1_000))
        }
    }

    @Test func testFunctionToolRejectsDeepOutputSchemaBeforeCopying() {
        runOnLargeStack {
            expectError(UserError.self, containing: "too deeply nested") {
                _ = try noopTool(name: "output_tool", schema: ["type": "object", "properties": [:]], outputJSONSchema: nestedObjectSchema(depth: 1_000))
            }
        }
    }
}

// MARK: - Invalid input and logging

extension FunctionToolTests {
    @Test(arguments: ["[]", #""value""#, "123", "null", "true"])
    func testFunctionToolRejectsNonObjectJSONInput(inputJSON: String) async throws {
        let tool = try echoTool(name: "echo_tool")

        await expectErrorAsync(ModelBehaviorError.self, containing: "Invalid JSON input for tool echo_tool: expected a JSON object") {
            _ = try await tool.onInvokeTool(makeContext(tool, inputJSON), inputJSON)
        }
    }

    @Test func testFunctionToolBadJSONRedactsPayloadWhenDontLogToolData() async throws {
        let tool = try echoTool(name: "echo_tool")
        let badJSON = #"{"secret":"SECRET_TOKEN_123""#

        try await AgentsDebug.withOverrides(dontLogToolData: true) {
            try await captureLogs(level: .debug) { logs in
                let error = await expectErrorAsync(ModelBehaviorError.self) {
                    _ = try await tool.onInvokeTool(makeContext(tool, badJSON), badJSON)
                }
                #expect(error?.message == "Invalid JSON input for tool echo_tool")
                #expect(!logs.messages.joined(separator: "\n").contains("SECRET_TOKEN_123"))
            }
        }
    }

    @Test func testFunctionToolBadJSONIncludesPayloadWhenToolLoggingEnabled() async throws {
        let tool = try echoTool(name: "echo_tool")
        let badJSON = #"{"secret":"SECRET_TOKEN_123""#

        try await AgentsDebug.withOverrides(dontLogToolData: false) {
            try await captureLogs(level: .debug) { logs in
                let error = await expectErrorAsync(ModelBehaviorError.self) {
                    _ = try await tool.onInvokeTool(makeContext(tool, badJSON), badJSON)
                }
                #expect(error?.message == "Invalid JSON input for tool echo_tool: \(badJSON)")
                #expect(logs.messages.joined(separator: "\n").contains("SECRET_TOKEN_123"))
            }
        }
    }

    private struct SensitiveContext: Sendable, CustomStringConvertible {
        var description: String { "CONTEXT_SECRET_SENTINEL" }
    }

    @Test func testFunctionToolArgumentLoggingExcludesLiveContext() async throws {
        let modelArgument = "MODEL_ARGUMENT_SENTINEL"
        let tool = try FunctionTool.create(name: "echo", argumentsType: EchoArgs.self) { ctx, args in
            #expect(ctx.context(as: SensitiveContext.self) != nil)
            return args.value
        }
        let arguments = #"{"value": "\#(modelArgument)"}"#
        let liveContext = ToolContext(context: SensitiveContext(), toolName: tool.name, toolCallId: "sensitive-context", toolArguments: arguments)

        try await AgentsDebug.withOverrides(dontLogToolData: false) {
            try await captureLogs(level: .debug) { logs in
                let result = try await tool.onInvokeTool(liveContext, arguments)
                #expect(result as? String == modelArgument)
                let text = logs.messages.joined(separator: "\n")
                #expect(text.contains(modelArgument))
                #expect(!text.contains("CONTEXT_SECRET_SENTINEL"))
            }
        }
    }
}

// MARK: - Guardrails and timeouts

extension FunctionToolTests {
    @Test func testFunctionToolAcceptsGuardrailArguments() throws {
        // Ports test_function_tool_accepts_guardrail_arguments and the decorator variant.
        let rejectArgsGuardrail = ToolInputGuardrail(name: "reject_args_guardrail") { data in
            .rejectContent("blocked", outputInfo: ["tool": data.context.toolName] as [String: String])
        }
        let allowOutputGuardrail = ToolOutputGuardrail(name: "allow_output_guardrail") { data in
            .allow(outputInfo: ["echo": data.output])
        }

        let tool = try FunctionTool.create(
            name: "simple_function",
            argumentsType: SimpleArgs.self,
            toolInputGuardrails: [rejectArgsGuardrail],
            toolOutputGuardrails: [allowOutputGuardrail]
        ) { _, args in args.a + args.b }

        #expect(tool.toolInputGuardrails?.map { $0.getName() } == ["reject_args_guardrail"])
        #expect(tool.toolOutputGuardrails?.map { $0.getName() } == ["allow_output_guardrail"])
    }

    private func slowTool(
        timeoutSeconds: Double = 0.01,
        timeoutBehavior: ToolTimeoutBehavior = .errorAsResult,
        timeoutErrorFunction: ToolErrorFunction? = nil
    ) throws -> FunctionTool {
        try FunctionTool.create(
            name: "slow_tool",
            timeoutSeconds: timeoutSeconds,
            timeoutBehavior: timeoutBehavior,
            timeoutErrorFunction: timeoutErrorFunction
        ) { _ in
            try await waitForever()
            return "slow"
        }
    }

    @Test func testInvokeFunctionToolTimeoutReturnsDefaultMessage() async throws {
        let tool = try slowTool()
        let ctx = ToolContext(context: (), toolName: tool.name, toolCallId: "slow", toolArguments: "{}")

        let result = try await invokeFunctionTool(tool: tool, context: ctx, arguments: "{}")

        let message = try #require(result as? String)
        #expect(message.lowercased().contains("timed out"))
        #expect(message.contains("0.01"))
    }

    @Test func testInvokeFunctionToolTimeoutUsesCustomErrorFunction() async throws {
        let tool = try slowTool(timeoutErrorFunction: { _, error in
            let timeoutError = try #require(error as? ToolTimeoutError)
            return "custom_timeout:\(timeoutError.toolName):\(formatSeconds(timeoutError.timeoutSeconds))"
        })
        let ctx = ToolContext(context: (), toolName: tool.name, toolCallId: "slow", toolArguments: "{}")

        let result = try await invokeFunctionTool(tool: tool, context: ctx, arguments: "{}")

        #expect(result as? String == "custom_timeout:slow_tool:0.01")
    }

    @Test func testInvokeFunctionToolTimeoutCanRaiseException() async throws {
        let tool = try slowTool(timeoutBehavior: .raiseException)
        let ctx = ToolContext(context: (), toolName: tool.name, toolCallId: "slow", toolArguments: "{}")

        await expectErrorAsync(ToolTimeoutError.self, containing: "timed out") {
            _ = try await invokeFunctionTool(tool: tool, context: ctx, arguments: "{}")
        }
    }

    private struct InternalTimeoutError: Error, CustomStringConvertible {
        var description: String
    }

    @Test func testInvokeFunctionToolDoesNotRewriteToolRaisedTimeoutError() async throws {
        let tool = try FunctionTool.create(name: "timeout_tool", failureErrorFunction: .none, timeoutSeconds: 1.0) { _ -> String in
            throw InternalTimeoutError(description: "tool_internal_timeout")
        }
        let ctx = ToolContext(context: (), toolName: tool.name, toolCallId: "timeout", toolArguments: "{}")

        await expectErrorAsync(InternalTimeoutError.self, containing: "tool_internal_timeout") {
            _ = try await invokeFunctionTool(tool: tool, context: ctx, arguments: "{}")
        }
    }

    @Test func testInvokeFunctionToolDoesNotRewriteManualToolRaisedTimeoutError() async throws {
        let manualTool = try FunctionTool(
            name: "manual_timeout_tool",
            description: "manual timeout",
            paramsJSONSchema: [:],
            onInvokeTool: { _, _ in throw InternalTimeoutError(description: "manual_tool_internal_timeout") },
            timeoutSeconds: 1.0
        )
        let ctx = ToolContext(context: (), toolName: manualTool.name, toolCallId: "timeout", toolArguments: "{}")

        await expectErrorAsync(InternalTimeoutError.self, containing: "manual_tool_internal_timeout") {
            _ = try await invokeFunctionTool(tool: manualTool, context: ctx, arguments: "{}")
        }
    }

    @Test func testFunctionToolTimeoutSecondsMustBePositiveNumber() {
        func makeTool(_ timeout: Double) throws {
            _ = try FunctionTool(name: "bad_timeout", description: "bad", paramsJSONSchema: [:], onInvokeTool: { _, _ in "ok" }, timeoutSeconds: timeout)
        }
        expectError(ValueError.self, containing: "greater than 0") { try makeTool(0.0) }
        expectError(ValueError.self, containing: "finite number") { try makeTool(.infinity) }
        expectError(ValueError.self, containing: "finite number") { try makeTool(.nan) }
    }
}
