import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_function_tool_approval_arguments.py: conditional approval observes the
// same invocation that can execute.
//
// Swift: upstream prepares arguments with the generated pydantic model and treats application
// validation as user code that must wait for approval. Swift tools decode a `Codable` arguments
// type, and Swift cannot tell a synthesized conformance from a hand-written one, so decoding is
// treated as data-only: the policy runs when the decoded arguments re-encode to the raw arguments
// (porting/decisions.md D18). The tests that exercise pydantic machinery are not ported:
// `test_application_validation_waits_for_explicit_approval`, `test_plugin_validation_*`,
// `test_compiled_pattern_requires_explicit_approval`, `test_pattern_validation_runs_after_approval`,
// `test_validator_exception_preserves_failure_policy`, `test_input_guardrail_blocks_application_validation`,
// and `test_invalid_keyword_reconstruction_uses_failure_policy` (`**kwargs`). The realtime modes
// and `test_realtime_*` wait for the AgentsRealtime port.
@Suite struct FunctionToolApprovalArgumentsTests {
    typealias RunMode = ToolApprovalCallIdReuseTests.RunMode
    typealias RunInputValue = ToolApprovalCallIdReuseTests.RunInputValue

    private func runAgent(_ agent: Agent, _ input: RunInputValue, mode: RunMode = .nonStreamed, hooks: RunHooks? = nil) async throws
        -> ToolApprovalCallIdReuseTests.RunOutcome {
        try await ToolApprovalCallIdReuseTests().run(agent, input, runConfig: RunConfig(tracingDisabled: true), mode: mode, hooks: hooks)
    }

    private func scriptedAgent(_ tool: FunctionTool, _ arguments: String) throws -> Agent {
        let model = try ScriptedModel(steps: [
            .output([getFunctionToolCall(tool.name, arguments: arguments, callId: "call_test")]),
            .output([getTextMessage("done")]),
        ])
        return Agent(name: "test", tools: [tool], model: .instance(model))
    }

    private func toolOutputs(_ items: [RunItem]) -> [String] {
        items.compactMap { ($0 as? ToolCallOutputItem)?.output as? String }
    }

    // MARK: - Argument types

    /// Upstream `target: Annotated[str, Field(alias="endpoint")] = "protected"`.
    private struct EndpointArgs: Codable, Sendable {
        var target: String
        enum CodingKeys: String, CodingKey { case target = "endpoint" }
        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            target = try container.decodeIfPresent(String.self, forKey: .target) ?? "protected"
        }
    }

    /// Upstream `environment: str = "safe"`.
    private struct EnvironmentArgs: Codable, Sendable {
        var environment: String
        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            environment = try container.decodeIfPresent(String.self, forKey: .environment) ?? "safe"
        }
    }

    // MARK: - Policy inputs

    @Test(arguments: RunMode.allCases, [
        ("{}", 0, true),
        (#"{"endpoint":"protected"}"#, 1, true),
        (#"{"endpoint":"safe"}"#, 1, false),
    ])
    func testDefaultsAndAliasKeepExistingPolicy(mode: RunMode, testCase: (arguments: String, expectedCalls: Int, interrupted: Bool)) async throws {
        let seen = SharedBox<[JSONObject]>([])
        let executed = SharedBox<[String]>([])
        let operation = try FunctionTool.create(
            name: "operation",
            argumentsType: EndpointArgs.self,
            strictJSONSchema: false,
            needsApproval: .dynamic { _, params, _ in
                seen.mutate { $0.append(params) }
                return params["endpoint"] == "protected"
            }
        ) { _, args in
            executed.mutate { $0.append(args.target) }
            return args.target
        }

        let result = try await runAgent(try scriptedAgent(operation, testCase.arguments), .text("go"), mode: mode)

        #expect(!result.interruptions.isEmpty == testCase.interrupted)
        #expect(seen.value.count == testCase.expectedCalls)
        #expect(executed.value == (testCase.interrupted ? [] : ["safe"]))
    }

    @Test(arguments: RunMode.allCases)
    func testBenignOmissionRequiresApprovalAndExplicitInputRuns(mode: RunMode) async throws {
        let seen = SharedBox<[JSONObject]>([])
        let executed = SharedBox<[String]>([])
        let operation = try FunctionTool.create(
            name: "operation",
            argumentsType: EnvironmentArgs.self,
            strictJSONSchema: false,
            needsApproval: .dynamic { _, params, _ in
                seen.mutate { $0.append(params) }
                return params["environment"] == nil
            }
        ) { _, args in
            executed.mutate { $0.append(args.environment) }
            return args.environment
        }

        let omitted = try await runAgent(try scriptedAgent(operation, "{}"), .text("go"), mode: mode)
        #expect(!omitted.interruptions.isEmpty && seen.value.isEmpty && executed.value.isEmpty)
        let explicit = try await runAgent(try scriptedAgent(operation, #"{"environment":"safe"}"#), .text("go"), mode: mode)
        #expect(explicit.finalOutput as? String == "done" && explicit.interruptions.isEmpty)
        #expect(seen.value == [["environment": "safe"]] && executed.value == ["safe"])
    }

    /// Upstream `Request` whose `targets` validator lowercases each value.
    private struct LowercasedTargets: Codable, Sendable {
        var targets: [String: String]
        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            targets = try container.decode([String: String].self, forKey: .targets).mapValues { $0.lowercased() }
        }
    }

    private struct LowercasedRequestArgs: Codable, Sendable { var request: LowercasedTargets }

    // Swift: only the `case` transformation is ported. Swift dictionaries have no order to change,
    // in-place mutation of the raw input is not possible, and an unchanged Codable request is
    // inspectable (D18), so the `order`, `in_place`, and `unchanged` cases do not apply. Decoding
    // runs while the call is inspected, so `validations` is not asserted.
    @Test func testNestedTransformationsRequireApproval() async throws {
        let seen = SharedBox<[JSONObject]>([])
        let executed = SharedBox<[String]>([])
        let operation = try FunctionTool.create(
            name: "operation",
            argumentsType: LowercasedRequestArgs.self,
            strictJSONSchema: false,
            needsApproval: .dynamic { _, params, _ in
                seen.mutate { $0.append(params) }
                return true
            }
        ) { _, args in
            let target = args.request.targets["z"] ?? ""
            executed.mutate { $0.append(target) }
            return target
        }
        let agent = try scriptedAgent(operation, #"{"request":{"targets":{"z":"PROD","a":"prod"}}}"#)

        let result = try await runAgent(agent, .text("go"))
        #expect(!result.interruptions.isEmpty && executed.value.isEmpty && seen.value.isEmpty)
        let state = result.state()
        try state.reject(try #require(state.getInterruptions().first))
        _ = try await runAgent(agent, .state(state))

        #expect(executed.value.isEmpty && seen.value.isEmpty)
    }

    // Swift: Swift Testing takes at most two argument collections, so the `serialize` and
    // `approve` parameters are combined.
    @Test(arguments: RunMode.allCases, [(false, false), (false, true), (true, false), (true, true)])
    func testManualDecisionRoundtripDoesNotRerunPredicate(mode: RunMode, options: (serialize: Bool, approve: Bool)) async throws {
        let (serialize, approve) = options
        let calls = SharedBox<[JSONObject]>([])
        let effects = SharedBox<[String]>([])
        let operation = try FunctionTool.create(
            name: "operation",
            argumentsType: EnvironmentArgs.self,
            strictJSONSchema: false,
            needsApproval: .dynamic { _, params, _ in
                calls.mutate { $0.append(params) }
                return false
            }
        ) { _, args in
            effects.mutate { $0.append(args.environment) }
            return args.environment
        }
        let agent = try scriptedAgent(operation, "{}")

        let paused = try await runAgent(agent, .text("go"), mode: mode)
        var state = paused.state()
        if serialize {
            state = try await RunState.fromString(agent, try state.toString())
        }
        let pending = try #require(state.getInterruptions().first)
        if approve {
            try state.approve(pending)
        } else {
            try state.reject(pending)
        }
        let result = try await runAgent(agent, .state(state), mode: mode)

        #expect(result.interruptions.isEmpty && calls.value.isEmpty)
        #expect(effects.value == (approve ? ["safe"] : []))
    }

    // MARK: - Failure policy

    enum FailurePolicy: String, CaseIterable, Sendable {
        case `default`, custom, raise
    }

    private final class ToolStartHooks: RunHooks, @unchecked Sendable {
        let callbacks: SharedBox<[String]>
        init(_ callbacks: SharedBox<[String]>) { self.callbacks = callbacks }
        override func onToolStart(context: RunContextWrapper, agent: AgentBase, tool: any Tool) async throws {
            callbacks.mutate { $0.append("start") }
        }
    }

    private struct NumberArgs: Codable, Sendable { var number: Int }

    // Swift: Swift Testing takes at most two argument collections, so `failure` and `approve` are
    // combined.
    @Test(arguments: FailurePolicy.allCases.flatMap { failure in [(failure, false), (failure, true)] }, RunMode.allCases)
    func testInvalidTypedInputRequiresApprovalBeforeFailurePolicy(options: (failure: FailurePolicy, approve: Bool), mode: RunMode) async throws {
        let (failure, approve) = options
        let seen = SharedBox<[JSONObject]>([])
        let effects = SharedBox<[Int]>([])
        let formatted = SharedBox<[String]>([])
        let callbacks = SharedBox<[String]>([])
        let guardrail = ToolInputGuardrail(name: "guardrail") { _ in
            callbacks.mutate { $0.append("guardrail") }
            return .allow()
        }
        let failureFunction: FailureErrorFunction
        switch failure {
        case .default: failureFunction = .default
        case .raise: failureFunction = .none
        case .custom:
            failureFunction = .custom { _, error in
                formatted.mutate { $0.append(String(describing: type(of: error))) }
                return "correct the number"
            }
        }
        let operation = try FunctionTool.create(
            name: "operation",
            argumentsType: NumberArgs.self,
            failureErrorFunction: failureFunction,
            needsApproval: .dynamic { _, params, _ in
                seen.mutate { $0.append(params) }
                return true
            },
            toolInputGuardrails: [guardrail]
        ) { _, args in
            effects.mutate { $0.append(args.number) }
            return String(args.number)
        }
        let agent = try scriptedAgent(operation, #"{"number":"invalid"}"#)
        let hooks = ToolStartHooks(callbacks)

        let paused = try await runAgent(agent, .text("go"), mode: mode, hooks: hooks)
        #expect(!paused.interruptions.isEmpty)
        #expect(seen.value.isEmpty && formatted.value.isEmpty && callbacks.value.isEmpty && effects.value.isEmpty)
        let state = try await RunState.fromString(agent, try paused.state().toString())
        let pending = try #require(state.getInterruptions().first)
        if !approve {
            try state.reject(pending)
            let result = try await runAgent(agent, .state(state), mode: mode, hooks: hooks)
            #expect(result.interruptions.isEmpty && formatted.value.isEmpty && callbacks.value.isEmpty && effects.value.isEmpty)
        } else {
            try state.approve(pending)
            if failure == .raise {
                await #expect(throws: ModelBehaviorError.self) {
                    _ = try await runAgent(agent, .state(state), mode: mode, hooks: hooks)
                }
            } else {
                let result = try await runAgent(agent, .state(state), mode: mode, hooks: hooks)
                #expect(result.interruptions.isEmpty)
                let outputs = toolOutputs(result.newItems)
                #expect(outputs.count == 1)
                if failure == .custom {
                    #expect(outputs == ["correct the number"])
                }
            }
            #expect(callbacks.value.contains("start"))
        }
        #expect(seen.value.isEmpty && effects.value.isEmpty)
        #expect(formatted.value == (approve && failure == .custom ? ["ModelBehaviorError"] : []))
        #expect(callbacks.value.filter { $0 == "guardrail" }.count == (approve ? 1 : 0))
    }

    // MARK: - Prepared arguments

    private struct EnvironmentRequestArgs: Codable, Sendable { var request: [String: String] }

    // Swift: `copy.copy` and `dataclasses.replace` both map to `FunctionTool.copy()`. Policy
    // parameters are a value type, so a policy cannot mutate the arguments it inspected.
    @Test func testPreparationIsPreservedOnCopiedTools() async throws {
        let seen = SharedBox<[JSONObject]>([])
        let effects = SharedBox<[String]>([])
        let operation = try FunctionTool.create(
            name: "operation",
            argumentsType: EnvironmentRequestArgs.self,
            strictJSONSchema: false,
            needsApproval: .dynamic { _, params, _ in
                var params = params
                seen.mutate { $0.append(params) }
                params["request"] = ["environment": "changed"]
                return false
            }
        ) { _, args in
            let environment = args.request["environment"] ?? ""
            effects.mutate { $0.append(environment) }
            return environment
        }

        let copied = operation.copy()
        let result = try await runAgent(try scriptedAgent(copied, #"{"request":{"environment":"safe"}}"#), .text("go"))

        #expect(result.interruptions.isEmpty)
        #expect(effects.value == ["safe"] && seen.value == [["request": ["environment": "safe"]]])
    }

    private struct ValueRequestArgs: Codable, Sendable { var request: [String: Int] }

    // Swift: upstream's two policies wait on each other while both calls are being approved. The
    // Swift batch executor starts the next queued tool only once the previous one reaches its body
    // (FIFO dispatch), so the policies here do not block; the test still checks that each call runs
    // with its own prepared arguments.
    @Test func testParallelCallsKeepDistinctPreparedValues() async throws {
        let executed = SharedBox<[Int]>([])
        let operation = try FunctionTool.create(
            name: "operation",
            argumentsType: ValueRequestArgs.self,
            strictJSONSchema: false,
            needsApproval: .dynamic { _, _, _ in false }
        ) { _, args in
            let value = args.request["value"] ?? 0
            executed.mutate { $0.append(value) }
            return String(value)
        }
        let model = try ScriptedModel(steps: [
            .output([
                getFunctionToolCall(operation.name, arguments: #"{"request":{"value":1}}"#, callId: "one"),
                getFunctionToolCall(operation.name, arguments: #"{"request":{"value":2}}"#, callId: "two"),
            ]),
            .output([getTextMessage("done")]),
        ])

        let result = try await runAgent(Agent(name: "test", tools: [operation], model: .instance(model)), .text("go"))

        #expect(result.finalOutput as? String == "done" && executed.value.sorted() == [1, 2])
        #expect(Set(toolOutputs(result.newItems)) == ["1", "2"])
    }

    private struct IntValueArgs: Codable, Sendable { var value: Int }
    private struct DoubleValueArgs: Codable, Sendable { var value: Double }

    enum ArgumentKind: String, Sendable { case int, double }

    // Swift: the `tuple[int, ...]` case has no Codable counterpart.
    @Test(arguments: [
        (ArgumentKind.int, #"{"value":"1"}"#, true),
        (ArgumentKind.int, #"{"value":true}"#, true),
        (ArgumentKind.double, #"{"value":1}"#, true),
        (ArgumentKind.double, #"{"value":1.0}"#, false),
        (ArgumentKind.int, #"{"value":1,"extra":2}"#, true),
    ])
    func testChangedOrUninspectableArgumentRequiresManualApproval(kind: ArgumentKind, arguments: String, interrupted: Bool) async throws {
        let seen = SharedBox<[JSONObject]>([])
        let effects = SharedBox<[String]>([])
        let policy = ToolApprovalRequirement.dynamic { _, params, _ in
            seen.mutate { $0.append(params) }
            return false
        }
        let operation: FunctionTool
        switch kind {
        case .int:
            operation = try FunctionTool.create(name: "operation", argumentsType: IntValueArgs.self, strictJSONSchema: false, needsApproval: policy) { _, args in
                effects.mutate { $0.append(String(args.value)) }
                return "ok"
            }
        case .double:
            operation = try FunctionTool.create(name: "operation", argumentsType: DoubleValueArgs.self, strictJSONSchema: false, needsApproval: policy) { _, args in
                effects.mutate { $0.append(String(args.value)) }
                return "ok"
            }
        }

        let result = try await runAgent(try scriptedAgent(operation, arguments), .text("go"))

        #expect(!result.interruptions.isEmpty == interrupted)
        #expect(seen.value.count == (interrupted ? 0 : 1))
        #expect(effects.value.count == (interrupted ? 0 : 1))
    }

    private struct OrderedArgs: Codable, Sendable {
        var first: String
        var second: String
    }

    @Test func testRootParameterOrderDoesNotChangeApproval() async throws {
        let seen = SharedBox<[[String]]>([])
        let operation = try FunctionTool.create(
            name: "operation",
            argumentsType: OrderedArgs.self,
            needsApproval: .dynamic { _, params, _ in
                seen.mutate { $0.append(params.keys) }
                return false
            }
        ) { _, args in args.first + args.second }

        let result = try await runAgent(try scriptedAgent(operation, #"{"second":"b","first":"a"}"#), .text("go"))

        #expect(result.interruptions.isEmpty && seen.value == [["second", "first"]])
    }

    private struct ResumedValueArgs: Codable, Sendable { var value: String }

    @Test(arguments: [false, true])
    func testResumeSelectionReusesPreparationOrRequiresApproval(invalid: Bool) async throws {
        // The resolver controls a queued call without a recorded approval. This isolates the
        // supported in-memory selection/execution handoff before any body is run.
        let policies = SharedBox<[JSONObject]>([])
        let effects = SharedBox<[String]>([])
        let errors = SharedBox<[String]>([])
        let operation = try FunctionTool.create(
            name: "operation",
            argumentsType: ResumedValueArgs.self,
            failureErrorFunction: .custom { _, error in
                errors.mutate { $0.append(String(describing: type(of: error))) }
                return "correct input"
            },
            needsApproval: .dynamic { _, params, _ in
                policies.mutate { $0.append(params) }
                return false
            }
        ) { _, args in
            effects.mutate { $0.append(args.value) }
            return args.value
        }
        let arguments = invalid ? #"{"value":3}"# : #"{"value":"safe"}"#
        let callItem = getFunctionToolCall(operation.name, arguments: arguments, callId: "resumed")
        let call = try #require(ResponseFunctionToolCall(json: callItem))
        let agent = try scriptedAgent(operation, arguments)
        let processed = ProcessedResponse(
            newItems: [],
            handoffs: [],
            functions: [ToolRunFunction(toolCall: call, functionTool: operation)],
            computerActions: [],
            localShellCalls: [],
            shellCalls: [],
            applyPatchCalls: [],
            toolsUsed: [],
            mcpApprovalRequests: [],
            interruptions: []
        )

        let result = try await resolveInterruptedTurn(
            agent: agent,
            allTools: [operation],
            originalInput: .text("go"),
            originalPreStepItems: [],
            newResponse: ModelResponse(output: [callItem], usage: Usage(), responseId: "response"),
            processedResponse: processed,
            hooks: RunHooks(),
            contextWrapper: RunContextWrapper(),
            runConfig: RunConfig(tracingDisabled: true),
            serverManagesConversation: false,
            errorHandlers: nil
        )

        #expect(effects.value == (invalid ? [] : ["safe"]))
        #expect(policies.value.count == (invalid ? 0 : 1))
        #expect(errors.value.isEmpty)
        #expect(!result.nextStep.interruptions.isEmpty == invalid)
        #expect(toolOutputs(result.newStepItems) == (invalid ? [] : ["safe"]))
    }

    private struct EndpointTarget: Codable, Sendable {
        var target: String
        enum CodingKeys: String, CodingKey { case target = "endpoint" }
    }

    private struct EndpointRequestArgs: Codable, Sendable { var request: EndpointTarget }

    // Swift: upstream's nested pydantic model is application code, so the policy never runs. A
    // nested Codable type is inspectable (D18): the policy sees the original `endpoint` key and
    // requires approval. `AliasChoices` has no Codable counterpart.
    @Test func testNestedInputAliasesKeepOriginalPolicyKeys() async throws {
        let seen = SharedBox<[JSONObject]>([])
        let effects = SharedBox<[String]>([])
        let operation = try FunctionTool.create(
            name: "operation",
            argumentsType: EndpointRequestArgs.self,
            strictJSONSchema: false,
            needsApproval: .dynamic { _, params, _ in
                seen.mutate { $0.append(params) }
                guard case .object(let request)? = params["request"] else { return true }
                return request["endpoint"] == "protected"
            }
        ) { _, args in
            effects.mutate { $0.append(args.request.target) }
            return args.request.target
        }

        let result = try await runAgent(try scriptedAgent(operation, #"{"request":{"endpoint":"protected"}}"#), .text("go"))

        #expect(!result.interruptions.isEmpty && effects.value.isEmpty)
        #expect(seen.value == [["request": ["endpoint": "protected"]]])
    }

    private struct NamedRequestArgs: Codable, Sendable { var request: [String: String] }

    @Test func testCancelledPolicyDoesNotReusePreparationForANewRun() async throws {
        let started = AsyncEvent()
        let policies = SharedBox<[JSONObject]>([])
        let effects = SharedBox<[String]>([])
        let operation = try FunctionTool.create(
            name: "operation",
            argumentsType: NamedRequestArgs.self,
            strictJSONSchema: false,
            needsApproval: .dynamic { _, params, _ in
                let count = policies.mutate { policies -> Int in
                    policies.append(params)
                    return policies.count
                }
                if count == 1 {
                    started.set()
                    try await waitForever()
                }
                return false
            }
        ) { _, args in
            let value = args.request["value"] ?? ""
            effects.mutate { $0.append(value) }
            return value
        }
        let firstAgent = try scriptedAgent(operation, #"{"request":{"value":"first"}}"#)

        let task = Task { _ = try await runAgent(firstAgent, .text("go")) }
        try await started.wait(timeout: 1)
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        let result = try await runAgent(try scriptedAgent(operation, #"{"request":{"value":"second"}}"#), .text("go"))

        #expect(result.interruptions.isEmpty)
        #expect(policies.value == [["request": ["value": "first"]], ["request": ["value": "second"]]])
        #expect(effects.value == ["second"])
    }
}
