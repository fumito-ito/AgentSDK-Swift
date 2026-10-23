import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_tool_approval_call_id_reuse.py (execution lifecycle cases).
extension ToolApprovalCallIdReuseTests {
    // MARK: - Execution lifecycle

    final class FailingLLMEndHooks: RunHooks, @unchecked Sendable {
        override func onLLMEnd(context: RunContextWrapper, agent: Agent, response: ModelResponse) async throws {
            throw TestError("llm end failed")
        }
    }

    @Test func testStreamedValidatedItemsPrecedeLLMEndHookFailure() async throws {
        let executed = SharedBox<[String]>([])
        let events = SharedBox<[StreamEvent]>([])
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self) { _, args in
            executed.mutate { $0.append(args.value) }
            return args.value
        }
        let model = try ScriptedModel(steps: [.output([getFunctionToolCall("record_value", arguments: #"{"value":"safe"}"#, callId: "call_0")])])
        let agent = Agent(name: "agent", tools: [recordValue], model: .instance(model))

        await expectErrorAsync(TestError.self, containing: "llm end failed") {
            _ = try await run(agent, .text("record a value"), mode: .streamed, hooks: FailingLLMEndHooks(), events: events)
        }

        let names = events.value.compactMap { event -> String? in
            if case .runItem(let item) = event { return item.name }
            return nil
        }
        #expect(names == ["tool_called"])
        #expect(executed.value.isEmpty)
    }

    @Test func testNonApprovalFailedToolBodyDoesNotReexecute() async throws {
        let attempts = SharedBox<[String]>([])
        let performSideEffect = try FunctionTool.create(name: "perform_side_effect", failureErrorFunction: .none) { _ -> String in
            attempts.mutate { $0.append("ran") }
            throw TestError("failed after side effect")
        }
        let call = getFunctionToolCall("perform_side_effect", arguments: "{}", callId: "call_0")
        let model = try ScriptedModel()
        try model.extend([.output([call]), .output([call])])
        let agent = Agent(name: "agent", tools: [performSideEffect], model: .instance(model))
        let context = RunContextWrapper(context: ())

        await expectErrorAsync(UserError.self, containing: "failed after side effect") {
            _ = try await Runner.run(agent, input: "run it", context: context)
        }
        await expectErrorAsync(ModelBehaviorError.self, containing: "already executed") {
            _ = try await Runner.run(agent, input: "run it", context: context)
        }

        #expect(attempts.value == ["ran"])
    }

    @Test func testNonApprovalCustomToolIdenticalSiblingsExecuteOnce() async throws {
        let executed = SharedBox<[String]>([])
        let tool = try CustomTool(
            name: "raw_editor",
            description: "Edit raw text.",
            onInvokeTool: { _, value in
                executed.mutate { $0.append(value) }
                return value
            },
            format: ["type": "text"]
        )
        let duplicate: JSONObject = ["type": "custom_tool_call", "name": .string(tool.name), "call_id": "call_0", "input": "safe"]
        let model = try ScriptedModel(steps: [.output([duplicate, duplicate])])
        try model.enqueue(.output([getTextMessage("done")]))
        let agent = Agent(name: "agent", tools: [tool], model: .instance(model))

        let result = try await Runner.run(agent, input: "edit text")

        #expect(result.finalOutput as? String == "done")
        #expect(executed.value == ["safe"])
    }

    // Swift: test_approved_custom_tool_resume_rebinds_to_current_tool and
    // test_approved_custom_tool_resume_fails_when_current_tool_is_missing need resume-time
    // reconciliation against the current tool inventory, which is not ported yet
    // (porting/manifest.json).

    @Test func testApprovedApplyPatchCustomResumeUsesPublicTool() async throws {
        let editor = RecordingEditor()
        let tool = try ApplyPatchTool(editor: editor, needsApproval: .required(true))
        let call: JSONObject = [
            "type": "custom_tool_call",
            "name": "apply_patch",
            "call_id": "patch_0",
            "input": #"{"type": "update_file", "path": "safe.txt", "diff": "-old\\n+new\\n"}"#,
        ]
        let model = try ScriptedModel(steps: [.output([call]), .output([getTextMessage("done")])])
        let agent = Agent(name: "agent", tools: [tool], model: .instance(model))

        let first = try await Runner.run(agent, input: "update the file")
        let state = first.toState()
        try state.approve(state.getInterruptions()[0])

        let resumed = try await Runner.run(agent, state: state)

        #expect(resumed.finalOutput as? String == "done")
        #expect(editor.operations.map(\.path) == ["safe.txt"])
    }

    func rawEditor(needsApproval: Bool, executed: SharedBox<[String]>) throws -> CustomTool {
        try CustomTool(
            name: "raw_editor",
            description: "Edit raw text.",
            onInvokeTool: { _, value in
                executed.mutate { $0.append(value) }
                return value
            },
            format: ["type": "text"],
            needsApproval: .required(needsApproval)
        )
    }

    @Test func testPendingCustomToolResumeSurvivesMissingCurrentTool() async throws {
        let executed = SharedBox<[String]>([])
        let call: JSONObject = ["type": "custom_tool_call", "name": "raw_editor", "call_id": "call_0", "input": "safe"]
        let agent = Agent(name: "agent", tools: [try rawEditor(needsApproval: true, executed: executed)], model: .instance(try ScriptedModel(steps: [.output([call])])))

        let first = try await Runner.run(agent, input: "edit")
        let state = first.toState()
        agent.tools = []

        let resumed = try await Runner.run(agent, state: state)

        #expect(resumed.interruptions.count == 1)
        #expect(executed.value.isEmpty)
    }

    @Test(arguments: [false, true])
    func testRejectedCustomToolResumeEmitsOutputWhenCurrentToolIsMissing(serializeState: Bool) async throws {
        let executed = SharedBox<[String]>([])
        let call: JSONObject = ["type": "custom_tool_call", "name": "raw_editor", "call_id": "call_0", "input": "safe"]
        let model = try ScriptedModel(steps: [.output([call]), .output([getTextMessage("done")])])
        let agent = Agent(name: "agent", tools: [try rawEditor(needsApproval: true, executed: executed)], model: .instance(model))

        let first = try await Runner.run(agent, input: "edit text")
        var state = first.toState()
        try state.reject(state.getInterruptions()[0])
        agent.tools = []
        if serializeState {
            state = try await RunState.fromJSON(agent, try state.toJSON())
        }

        let resumed = try await Runner.run(agent, state: state)

        #expect(resumed.finalOutput as? String == "done")
        #expect(executed.value.isEmpty)
        let modelInput = try #require(model.calls.last?.input.inputItems)
        let rejectionOutputs = modelInput.filter { $0.string("type") == "custom_tool_call_output" && $0.string("call_id") == "call_0" }
        #expect(rejectionOutputs.count == 1)
    }

    @Test(arguments: RunMode.allCases)
    func testChangedSameIdSiblingsFailBeforeApprovalCallback(mode: RunMode) async throws {
        let approvalCalls = SharedBox<[String]>([])
        let recordValue = try FunctionTool.create(
            name: "record_value",
            argumentsType: ValueArgs.self,
            needsApproval: .dynamic { _, arguments, _ in
                approvalCalls.mutate { $0.append(arguments.string("value") ?? "") }
                return true
            }
        ) { _, args in args.value }
        let siblings = [
            getFunctionToolCall("record_value", arguments: #"{"value":"one"}"#, callId: "call_0"),
            getFunctionToolCall("record_value", arguments: #"{"value":"two"}"#, callId: "call_0"),
        ]
        let model = try ScriptedModel(steps: [.output(siblings)])
        let agent = Agent(name: "agent", tools: [recordValue], model: .instance(model))

        await expectErrorAsync(ModelBehaviorError.self, containing: "one response") {
            _ = try await run(agent, .text("record values"), mode: mode)
        }

        #expect(approvalCalls.value.isEmpty)
    }

    // MARK: - Response processing errors

    func step(_ output: [TResponseOutputItem], _ mode: RunMode) -> ModelStep {
        mode == .streamed ? getExactOutputStreamStep(output) : .output(output)
    }

    @Test(arguments: RunMode.allCases)
    func testResponseProcessingErrorStillInvokesLLMEnd(mode: RunMode) async throws {
        let hooks = LLMEndRecordingHooks()
        let model = try ScriptedModel(steps: [step([makeShellCall("call_0", commands: ["echo safe"])], mode)])
        let agent = Agent(name: "agent", model: .instance(model))

        await expectErrorAsync(ModelBehaviorError.self, containing: "without a shell tool") {
            _ = try await run(agent, .text("run command"), mode: mode, hooks: hooks)
        }

        #expect(hooks.llmEndCalls.value == 1)
    }

    @Test(arguments: RunMode.allCases)
    func testProcessingErrorWithChangedCallIdStillSuppressesLLMEnd(mode: RunMode) async throws {
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self) { _, args in args.value }
        let outputs: [[TResponseOutputItem]] = [
            [getFunctionToolCall("record_value", arguments: #"{"value":"safe"}"#, callId: "shared")],
            [
                getFunctionToolCall("record_value", arguments: #"{"value":"changed"}"#, callId: "shared"),
                makeShellCall("shell_0", commands: ["echo safe"]),
            ],
        ]
        let model = try ScriptedModel(steps: outputs.map { step($0, mode) })
        let hooks = LLMEndRecordingHooks()
        let agent = Agent(name: "agent", tools: [recordValue], model: .instance(model))

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await run(agent, .text("record value"), mode: mode, hooks: hooks)
        }

        #expect(hooks.llmEndCalls.value == 1)
    }

    @Test(arguments: RunMode.allCases)
    func testProcessingErrorWithRepeatedUncanonicalIdSuppressesLLMEnd(mode: RunMode) async throws {
        let firstCall: JSONObject = ["type": "custom_tool_call", "name": "first_missing_tool", "call_id": "shared"]
        let secondCall: JSONObject = ["type": "custom_tool_call", "name": "second_missing_tool", "call_id": "shared"]
        let hooks = LLMEndRecordingHooks()
        let model = try ScriptedModel(steps: [step([firstCall, secondCall], mode)])
        let agent = Agent(name: "agent", model: .instance(model))

        await expectErrorAsync(ModelBehaviorError.self, containing: "one response") {
            _ = try await run(agent, .text("run tool"), mode: mode, hooks: hooks)
        }

        #expect(hooks.llmEndCalls.value == 0)
    }

    @Test(arguments: RunMode.allCases)
    func testChangedSameIdSiblingsFailBeforeNonApprovalExecution(mode: RunMode) async throws {
        let executed = SharedBox<[String]>([])
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self) { _, args in
            executed.mutate { $0.append(args.value) }
            return args.value
        }
        let siblings = [
            getFunctionToolCall("record_value", arguments: #"{"value":"one"}"#, callId: "call_0"),
            getFunctionToolCall("record_value", arguments: #"{"value":"two"}"#, callId: "call_0"),
        ]
        let agent = Agent(name: "agent", tools: [recordValue], model: .instance(try ScriptedModel(steps: [.output(siblings)])))

        await expectErrorAsync(ModelBehaviorError.self, containing: "one response") {
            _ = try await run(agent, .text("record values"), mode: mode)
        }

        #expect(executed.value.isEmpty)
    }

    // MARK: - Computer

    final class RecordingComputer: Computer, @unchecked Sendable {
        let screenshots = SharedBox<[String]>([])
        func screenshot() async throws -> String {
            screenshots.mutate { $0.append("screenshot") }
            return "img"
        }
        func click(x: Int, y: Int, button: Button) async throws {}
        func doubleClick(x: Int, y: Int) async throws {}
        func scroll(x: Int, y: Int, scrollX: Int, scrollY: Int) async throws {}
        func type(text: String) async throws {}
        func wait() async throws {}
        func move(x: Int, y: Int) async throws {}
        func keypress(keys: [String]) async throws {}
        func drag(path: [(x: Int, y: Int)]) async throws {}
    }

    func computerCall(itemId: String, safetyCheck: (id: String, code: String, message: String)?) -> JSONObject {
        let checks: [JSONValue] = safetyCheck.map { [["id": .string($0.id), "code": .string($0.code), "message": .string($0.message)]] } ?? []
        return [
            "id": .string(itemId),
            "type": "computer_call",
            "action": ["type": "screenshot"],
            "call_id": "computer-call",
            "pending_safety_checks": .array(checks),
            "status": "completed",
        ]
    }

    func safetyCheckedComputerTool(_ computer: RecordingComputer, safetyChecks: SharedBox<[String]>) -> ComputerTool {
        ComputerTool(computer: computer, onSafetyCheck: { data in
            safetyChecks.mutate { $0.append(data.safetyCheck.string("id") ?? "") }
            return true
        })
    }

    @Test(arguments: RunMode.allCases)
    func testChangedComputerSafetyChecksFailBeforeSameResponseEffects(mode: RunMode) async throws {
        let safetyChecks = SharedBox<[String]>([])
        let computer = RecordingComputer()
        let firstCall = computerCall(itemId: "computer-item-1", safetyCheck: ("safety-1", "code-1", "first"))
        let changedCall = computerCall(itemId: "computer-item-2", safetyCheck: ("safety-2", "code-2", "changed"))
        let agent = Agent(
            name: "computer-agent",
            tools: [safetyCheckedComputerTool(computer, safetyChecks: safetyChecks)],
            model: .instance(try ScriptedModel(steps: [step([firstCall, changedCall], mode)]))
        )

        await expectErrorAsync(ModelBehaviorError.self, containing: "one response") {
            _ = try await run(agent, .text("use computer"), mode: mode)
        }

        #expect(safetyChecks.value.isEmpty)
        #expect(computer.screenshots.value.isEmpty)
    }

    @Test(arguments: RunMode.allCases)
    func testChangedComputerSafetyChecksFailBeforeCompletedReplayEffects(mode: RunMode) async throws {
        let safetyChecks = SharedBox<[String]>([])
        let computer = RecordingComputer()
        let firstCall = computerCall(itemId: "computer-item-1", safetyCheck: ("safety-1", "code-1", "first"))
        let changedCall = computerCall(itemId: "computer-item-2", safetyCheck: ("safety-2", "code-2", "changed"))
        let model = try ScriptedModel(steps: [step([firstCall], mode), step([changedCall], mode)])
        let agent = Agent(name: "computer-agent", tools: [safetyCheckedComputerTool(computer, safetyChecks: safetyChecks)], model: .instance(model))

        await expectErrorAsync(ModelBehaviorError.self, containing: "completed tool call ID") {
            _ = try await run(agent, .text("use computer"), mode: mode)
        }

        #expect(safetyChecks.value == ["safety-1"])
        #expect(computer.screenshots.value == ["screenshot"])
    }

    final class FailOnceToolEndHooks: RunHooks, @unchecked Sendable {
        let failed = SharedBox(false)
        let toolName: String?
        let message: String
        let waitBeforeFailing: AsyncEvent?

        init(toolName: String? = nil, message: String = "end hook failed", waitBeforeFailing: AsyncEvent? = nil) {
            self.toolName = toolName
            self.message = message
            self.waitBeforeFailing = waitBeforeFailing
        }

        override func onToolEnd(context: RunContextWrapper, agent: AgentBase, tool: any Tool, result: any Sendable) async throws {
            if let toolName, tool.name != toolName { return }
            guard !failed.value else { return }
            try await waitBeforeFailing?.wait()
            failed.value = true
            throw TestError(message)
        }
    }

    @Test func testComputerHookFailureDoesNotRepeatSideEffect() async throws {
        let computer = RecordingComputer()
        let call = computerCall(itemId: "computer-item", safetyCheck: nil)
        let model = try ScriptedModel()
        try model.extend([.output([call]), .output([call])])
        let agent = Agent(name: "computer-agent", tools: [ComputerTool(computer: computer)], model: .instance(model))
        let context = RunContextWrapper(context: ())
        let hooks = FailOnceToolEndHooks()

        await expectErrorAsync(TestError.self, containing: "end hook failed") {
            _ = try await Runner.run(agent, input: "use computer", context: context, hooks: hooks)
        }
        await expectErrorAsync(ModelBehaviorError.self, containing: "already executed") {
            _ = try await Runner.run(agent, input: "use computer", context: context, hooks: hooks)
        }

        #expect(computer.screenshots.value == ["screenshot"])
    }

    // MARK: - Exact replay

    func tiedReasoningScenario() throws -> (Agent, ScriptedModel, SharedBox<[String]>) {
        let executed = SharedBox<[String]>([])
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
            executed.mutate { $0.append(args.value) }
            return args.value
        }
        let model = try ScriptedModel()
        try model.extend([
            .output([getFunctionToolCall("record_value", arguments: #"{"value":"safe"}"#, callId: "call_0")]),
            .output([
                ["id": "rs_replay", "summary": [], "type": "reasoning"],
                getFunctionToolCall("record_value", arguments: #"{ "value" : "safe" }"#, callId: "call_0"),
            ]),
            .output([getTextMessage("done")]),
        ])
        return (Agent(name: "agent", tools: [recordValue], model: .instance(model)), model, executed)
    }

    @Test func testExactReplayDropsTiedReasoningItem() async throws {
        let (agent, model, executed) = try tiedReasoningScenario()
        let first = try await Runner.run(agent, input: "record a value")
        let state = first.toState()
        try state.approve(first.interruptions[0])

        let resumed = try await Runner.run(agent, state: state)

        #expect(resumed.finalOutput as? String == "done")
        #expect(executed.value == ["safe"])
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(!modelInput.contains { $0.string("id") == "rs_replay" })
        #expect(modelInput.filter { $0.string("type") == "function_call" && $0.string("call_id") == "call_0" }.count == 1)
    }

    @Test func testStreamedExactReplayDoesNotEmitTiedReasoningItem() async throws {
        let (agent, _, executed) = try tiedReasoningScenario()
        let first = try await Runner.run(agent, input: "record a value")
        let state = first.toState()
        try state.approve(first.interruptions[0])
        let events = SharedBox<[StreamEvent]>([])

        let resumed = try await run(agent, .state(state), mode: .streamed, events: events)

        let itemEvents = events.value.compactMap { event -> RunItemStreamEvent? in
            if case .runItem(let item) = event { return item }
            return nil
        }
        #expect(resumed.finalOutput as? String == "done")
        #expect(executed.value == ["safe"])
        #expect(!itemEvents.contains { $0.name == "reasoning_item_created" && $0.item.rawItem.string("id") == "rs_replay" })
        #expect(!itemEvents.contains { $0.name == "tool_called" })
    }

    // MARK: - Failures after side effects

    @Test func testFailedToolEndHookDoesNotReexecuteApprovedCall() async throws {
        let (agent, runConfig, executed) = try buildScenario(secondCallId: "call_0", secondValue: "safe")
        let hooks = FailOnceToolEndHooks()
        let first = try await Runner.run(agent, input: "record a value", runConfig: runConfig)
        let state = first.toState()
        try state.approve(first.interruptions[0])

        await expectErrorAsync(UserError.self, containing: "end hook failed") {
            _ = try await Runner.run(agent, state: state, hooks: hooks, runConfig: runConfig)
        }
        let resumed = try await Runner.run(agent, state: state, hooks: hooks, runConfig: runConfig)

        #expect(resumed.finalOutput as? String == "done")
        #expect(executed.value == ["safe"])
    }

    @Test func testFailedOutputGuardrailDoesNotReexecuteApprovedCall() async throws {
        let executed = SharedBox<[String]>([])
        let rejectOutput = ToolOutputGuardrail { _ in .raiseException(outputInfo: "blocked") }
        let recordValue = try FunctionTool.create(name: "record_value", needsApproval: true, toolOutputGuardrails: [rejectOutput]) { _ in
            executed.mutate { $0.append("ran") }
            return "sensitive"
        }
        let model = try ScriptedModel(steps: [.output([getFunctionToolCall("record_value", arguments: "{}", callId: "call_0")])])
        let agent = Agent(name: "agent", tools: [recordValue], model: .instance(model))
        let first = try await Runner.run(agent, input: "record a value")
        let state = first.toState()
        try state.approve(first.interruptions[0])

        await expectErrorAsync(ToolOutputGuardrailTripwireTriggered.self) {
            _ = try await Runner.run(agent, state: state)
        }

        let restored = try await RunState.fromJSON(agent, try state.toJSON())
        await expectErrorAsync(ModelBehaviorError.self, containing: "already executed") {
            _ = try await Runner.run(agent, state: restored)
        }

        #expect(executed.value == ["ran"])
    }

    @Test func testFailedApprovedToolBodyDoesNotReexecute() async throws {
        let attempts = SharedBox<[String]>([])
        let performSideEffect = try FunctionTool.create(name: "perform_side_effect", failureErrorFunction: .none, needsApproval: true) { _ -> String in
            attempts.mutate { $0.append("ran") }
            throw TestError("failed after side effect")
        }
        let model = try ScriptedModel(steps: [.output([getFunctionToolCall("perform_side_effect", arguments: "{}", callId: "call_0")])])
        let agent = Agent(name: "agent", tools: [performSideEffect], model: .instance(model))
        let first = try await Runner.run(agent, input: "run it")
        let state = first.toState()
        try state.approve(first.interruptions[0])

        await expectErrorAsync(UserError.self, containing: "failed after side effect") {
            _ = try await Runner.run(agent, state: state)
        }
        await expectErrorAsync(ModelBehaviorError.self, containing: "already executed") {
            _ = try await Runner.run(agent, state: state)
        }

        #expect(attempts.value == ["ran"])
    }

    @Test func testCancelledApprovedToolBodyDoesNotReexecute() async throws {
        let attempts = SharedBox<[String]>([])
        let started = AsyncEvent()
        let keepRunning = AsyncEvent()
        let performSideEffect = try FunctionTool.create(name: "perform_side_effect", failureErrorFunction: .none, needsApproval: true) { _ -> String in
            attempts.mutate { $0.append("ran") }
            started.set()
            try await keepRunning.wait()
            return "done"
        }
        let model = try ScriptedModel(steps: [.output([getFunctionToolCall("perform_side_effect", arguments: "{}", callId: "call_0")])])
        let agent = Agent(name: "agent", tools: [performSideEffect], model: .instance(model))
        let first = try await Runner.run(agent, input: "run it")
        let state = first.toState()
        try state.approve(first.interruptions[0])

        let resumeTask = Task { try await Runner.run(agent, state: state) }
        try await started.wait()
        resumeTask.cancel()
        await expectErrorAsync(CancellationError.self) { _ = try await resumeTask.value }

        await expectErrorAsync(ModelBehaviorError.self, containing: "already executed") {
            _ = try await Runner.run(agent, state: state)
        }

        #expect(attempts.value == ["ran"])
    }

    final class FailingAgentToolStartHooks: RunHooks, @unchecked Sendable {
        let toolName: String
        let hookCalls: SharedBox<[String]>

        init(toolName: String, hookCalls: SharedBox<[String]>) {
            self.toolName = toolName
            self.hookCalls = hookCalls
        }

        override func onToolStart(context: RunContextWrapper, agent: AgentBase, tool: any Tool) async throws {
            guard tool.name == toolName else { return }
            hookCalls.mutate { $0.append("ran") }
            throw TestError("failed after side effect")
        }
    }

    @Test func testFailedApprovedAgentToolStartDoesNotReexecute() async throws {
        let hookCalls = SharedBox<[String]>([])
        let innerAgent = Agent(name: "inner", model: .instance(try ScriptedModel(steps: [.output([getTextMessage("inner done")])])))
        let agentTool = try innerAgent.asTool(toolName: "delegate", toolDescription: "Delegate work.", needsApproval: true)
        let outerModel = try ScriptedModel(steps: [.output([getFunctionToolCall("delegate", arguments: #"{"input":"hi"}"#, callId: "call_0")])])
        let outerAgent = Agent(name: "outer", tools: [agentTool], model: .instance(outerModel))
        let first = try await Runner.run(outerAgent, input: "delegate")
        let state = first.toState()
        try state.approve(first.interruptions[0])

        await expectErrorAsync(UserError.self, containing: "failed after side effect") {
            _ = try await Runner.run(outerAgent, state: state, hooks: FailingAgentToolStartHooks(toolName: "delegate", hookCalls: hookCalls))
        }
        await expectErrorAsync(ModelBehaviorError.self, containing: "already executed") {
            _ = try await Runner.run(outerAgent, state: state, hooks: FailingAgentToolStartHooks(toolName: "delegate", hookCalls: hookCalls))
        }

        #expect(hookCalls.value == ["ran"])
    }

    @Test func testFailedParallelToolEndHookCheckpointsOutputsInModelOrder() async throws {
        let executed = SharedBox<[String]>([])
        let firstFinished = AsyncEvent()
        let firstTool = try FunctionTool.create(name: "first_tool", needsApproval: true) { _ -> String in
            try await Task.sleep(nanoseconds: 10_000_000)
            executed.mutate { $0.append("first") }
            firstFinished.set()
            return "first"
        }
        let secondTool = try FunctionTool.create(name: "second_tool", needsApproval: true) { _ -> String in
            executed.mutate { $0.append("second") }
            return "second"
        }
        let calls = [
            getFunctionToolCall("first_tool", arguments: "{}", callId: "call_first"),
            getFunctionToolCall("second_tool", arguments: "{}", callId: "call_second"),
        ]
        let model = try ScriptedModel(steps: [.output(calls)])
        try model.enqueue(.output([getTextMessage("done")]))
        let agent = Agent(name: "agent", tools: [firstTool, secondTool], model: .instance(model))
        let hooks = FailOnceToolEndHooks(toolName: "second_tool", message: "second end hook failed", waitBeforeFailing: firstFinished)

        let first = try await Runner.run(agent, input: "run tools")
        let state = first.toState()
        for interruption in first.interruptions {
            try state.approve(interruption)
        }

        await expectErrorAsync(UserError.self, containing: "second end hook failed") {
            _ = try await Runner.run(agent, state: state, hooks: hooks)
        }
        let resumed = try await Runner.run(agent, state: state, hooks: hooks)

        #expect(resumed.finalOutput as? String == "done")
        #expect(executed.value.sorted() == ["first", "second"])
        let modelInput = try #require(model.calls.last?.input.inputItems)
        #expect(modelInput.filter { $0.string("type") == "function_call_output" }.compactMap { $0.string("call_id") } == ["call_first", "call_second"])
    }
}
