import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_tool_approval_call_id_reuse.py (serialized and legacy snapshot cases).
extension ToolApprovalCallIdReuseTests {
    // MARK: - Serialized and legacy snapshots

    func buildSerializedReplayScenario(replayValue: String) throws -> (Agent, RunConfig, ScriptedModel, SharedBox<[String]>) {
        let executed = SharedBox<[String]>([])
        let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
            executed.mutate { $0.append("record:\(args.value)") }
            return args.value
        }
        let approvalGate = try FunctionTool.create(name: "approval_gate", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
            executed.mutate { $0.append("gate:\(args.value)") }
            return args.value
        }
        let model = try ScriptedModel()
        try model.extend([
            .output([getFunctionToolCall("record_value", arguments: #"{"value": "safe"}"#, callId: "call_0")]),
            .output([getFunctionToolCall("approval_gate", arguments: #"{"value": "pause"}"#, callId: "call_1")]),
            .output([getFunctionToolCall("record_value", arguments: #"{"value":"\#(replayValue)"}"#, callId: "call_0")]),
            .output([getTextMessage("done")]),
        ])
        let runConfig = RunConfig(modelProvider: ScriptedProvider(model))
        let agent = Agent(name: "custom-provider-agent", tools: [recordValue, approvalGate], model: .name("scripted-provider-model"))
        return (agent, runConfig, model, executed)
    }

    @Test func testSerializedPendingApprovalKeepsItsInvocationBinding() async throws {
        let (agent, runConfig, executed) = try buildScenario(secondCallId: "call_0", secondValue: "changed")

        let first = try await Runner.run(agent, input: "record a value", runConfig: runConfig)
        let state = first.toState()
        try state.approve(first.interruptions[0])
        let restored = try await RunState.fromString(agent, try state.toString())

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Runner.run(agent, state: restored, runConfig: runConfig)
        }

        #expect(executed.value == ["safe"])
    }

    @Test(arguments: RunMode.allCases)
    func testSerializedCompletedApprovalRejectsChangedReplay(mode: RunMode) async throws {
        let (agent, runConfig, _, executed) = try buildSerializedReplayScenario(replayValue: "changed")

        let first = try await run(agent, .text("record a value"), runConfig: runConfig, mode: mode)
        let state = first.state()
        try state.approve(first.interruptions[0])
        let second = try await run(agent, .state(state), runConfig: runConfig, mode: mode)
        let restored = try await RunState.fromString(agent, try second.state().toString())
        try restored.approve(restored.getInterruptions()[0])

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await run(agent, .state(restored), runConfig: runConfig, mode: mode)
        }

        #expect(executed.value == ["record:safe", "gate:pause"])
    }

    @Test(arguments: RunMode.allCases)
    func testSerializedCompletedApprovalSkipsExactReplay(mode: RunMode) async throws {
        let (agent, runConfig, model, executed) = try buildSerializedReplayScenario(replayValue: "safe")

        let first = try await run(agent, .text("record a value"), runConfig: runConfig, mode: mode)
        let state = first.state()
        try state.approve(first.interruptions[0])
        let second = try await run(agent, .state(state), runConfig: runConfig, mode: mode)
        let restored = try await RunState.fromString(agent, try second.state().toString())
        try restored.approve(restored.getInterruptions()[0])

        let result = try await run(agent, .state(restored), runConfig: runConfig, mode: mode)

        #expect(result.finalOutput as? String == "done")
        #expect(executed.value == ["record:safe", "gate:pause"])
        let modelInput = try #require(model.calls.last?.input.inputItems)
        for callId in ["call_0", "call_1"] {
            #expect(modelInput.filter { $0.string("type") == "function_call" && $0.string("call_id") == callId }.count == 1)
            #expect(modelInput.filter { $0.string("type") == "function_call_output" && $0.string("call_id") == callId }.count == 1)
        }
    }

    /// Rewrites a snapshot the way a schema 1.13 writer stored approvals, without the ledger.
    func legacySnapshot(_ json: JSONObject, schemaVersion: String = "1.13") -> JSONObject {
        var json = json
        var context = json["context"]?.objectValue ?? [:]
        var approvals = context["approvals"]?.objectValue ?? [:]
        for entry in context["function_tool_approvals"]?.arrayValue ?? [] {
            if let toolKey = entry["tool_key"]?.stringValue, let decision = entry["decision"] {
                approvals[toolKey] = decision
            }
        }
        context.removeValue(forKey: "function_tool_approvals")
        context.removeValue(forKey: "tool_invocations")
        context["approvals"] = .object(approvals)
        json["context"] = .object(context)
        json["$schemaVersion"] = .string(schemaVersion)
        return json
    }

    @Test func testSchema113CustomNamedShellSkipsExactCompletedReplay() async throws {
        let executions = SharedBox<[String]>([])
        let call = makeShellCall("legacy-shell", commands: ["echo safe"])
        let model = try ScriptedModel(steps: [.output([call])])
        try model.enqueue(.output([getTextMessage("done")]))
        let shell = try ShellTool(
            executor: { request in
                executions.mutate { $0.append(request.data.callId) }
                return .text("ok")
            },
            name: "safe_shell"
        )
        let agent = Agent(name: "agent", tools: [shell], model: .instance(model))

        let completed = try await Runner.run(agent, input: "run")
        var serialized = legacySnapshot(try completed.toState().toJSON())
        for key in ["generated_items", "session_items"] {
            serialized[key] = .array((serialized[key]?.arrayValue ?? []).map { item in
                guard case .object(let object) = item else { return item }
                return .object(object.removing("tool_name"))
            })
        }
        if case .object(var processed)? = serialized["last_processed_response"] {
            processed["new_items"] = .array((processed["new_items"]?.arrayValue ?? []).map { item in
                guard case .object(let object) = item else { return item }
                return .object(object.removing("tool_name"))
            })
            serialized["last_processed_response"] = .object(processed)
        }

        let restored = try await RunState.fromJSON(agent, serialized)
        let expected = try #require(toolInvocationIdentity(call, toolName: "safe_shell"))
        let record = try #require(restored.context.toolInvocations["legacy-shell"])
        #expect(record.approvalScope == expected.approvalScope)
        #expect(record.fingerprint == expected.fingerprint)
        try model.extend([.output([call]), .output([getTextMessage("done again")])])
        let resumed = try await Runner.run(agent, state: restored)

        #expect(resumed.finalOutput as? String == "done again")
        #expect(executions.value == ["legacy-shell"])
    }

    @Test(arguments: ["1.13", "1.14"], RunMode.allCases)
    func testLegacySchemaHistoricalStickyCallIdIsNotReexecuted(schemaVersion: String, mode: RunMode) async throws {
        for replayValue in ["safe", "changed"] {
            let executed = SharedBox<[String]>([])
            let recordValue = try FunctionTool.create(name: "record_value", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
                executed.mutate { $0.append(args.value) }
                return args.value
            }
            let historicalCall = getFunctionToolCall("record_value", arguments: #"{"value":"safe"}"#, callId: "call_0")
            let model = try ScriptedModel(steps: [.output([getFunctionToolCall("record_value", arguments: #"{"value": "\#(replayValue)"}"#, callId: "call_0")])])
            try model.enqueue(.output([getTextMessage("done")]))
            let agent = Agent(name: "agent", tools: [recordValue], model: .instance(model))
            let context = RunContextWrapper(context: ())
            try context.approveTool(ToolApprovalItem(agent: agent, rawItem: historicalCall), alwaysApprove: true)
            let state = RunState(
                context: context,
                originalInput: .items([historicalCall, ["type": "function_call_output", "call_id": "call_0", "output": "safe"]]),
                startingAgent: agent
            )
            let restored = try await RunState.fromJSON(agent, legacySnapshot(try state.toJSON(), schemaVersion: schemaVersion))

            if replayValue == "changed" {
                await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
                    _ = try await run(agent, .state(restored), mode: mode)
                }
            } else {
                let result = try await run(agent, .state(restored), mode: mode)
                #expect(result.finalOutput as? String == "done")
            }

            #expect(executed.value.isEmpty, "replay \(replayValue)")
        }
    }

    // Swift: upstream drives `_collect_runs_by_approval` directly. Swift evaluates native tool
    // approvals inside each action, so this checks the lookup that helper performs first: the
    // changed pending call fails against the restored binding before any approval policy runs.
    @Test(arguments: ["1.13", "1.14"])
    func testLegacyChangedPendingCallFailsBeforeApprovalCallback(schemaVersion: String) async throws {
        let approvalCalls = SharedBox<[String]>([])
        let recordValue = try FunctionTool.create(
            name: "record_value",
            argumentsType: ValueArgs.self,
            needsApproval: .dynamic { _, arguments, _ in
                approvalCalls.mutate { $0.append(arguments.string("value") ?? "") }
                return true
            }
        ) { _, args in args.value }
        let historicalCall = getFunctionToolCall("record_value", arguments: #"{"value":"safe"}"#, callId: "call_0")
        let changedCall = getFunctionToolCall("record_value", arguments: #"{"value":"changed"}"#, callId: "call_0")
        let agent = Agent(name: "agent", tools: [recordValue])
        let context = RunContextWrapper(context: ())
        try context.approveTool(ToolApprovalItem(agent: agent, rawItem: historicalCall))
        let state = RunState(
            context: context,
            originalInput: .items([historicalCall, ["type": "function_call_output", "call_id": "call_0", "output": "safe"]]),
            startingAgent: agent,
            maxTurns: 10
        )
        state.currentStep = .interruption(interruptions: [ToolApprovalItem(agent: agent, rawItem: changedCall)], responseAccepted: false)
        let restored = try await RunState.fromJSON(agent, legacySnapshot(try state.toJSON(), schemaVersion: schemaVersion))
        let restoredPending = try #require(restored.getInterruptions().first)
        let current = ToolApprovalItem(agent: agent, rawItem: changedCall, toolName: "record_value")

        expectError(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try restored.context.getApprovalStatus(
                toolName: "record_value", callId: "call_0", existingPending: restoredPending, currentInvocation: current
            )
        }

        #expect(approvalCalls.value.isEmpty)
    }

    @Test func testSchema113CompletedApprovalRejectsChangedReplay() async throws {
        let (agent, runConfig, _, executed) = try buildSerializedReplayScenario(replayValue: "changed")

        let first = try await Runner.run(agent, input: "record a value", runConfig: runConfig)
        let state = first.toState()
        try state.approve(first.interruptions[0])
        let second = try await Runner.run(agent, state: state, runConfig: runConfig)
        let restored = try await RunState.fromJSON(agent, legacySnapshot(try second.toState().toJSON()))
        try restored.approve(restored.getInterruptions()[0])

        await expectErrorAsync(ModelBehaviorError.self, containing: "unique call ID") {
            _ = try await Runner.run(agent, state: restored, runConfig: runConfig)
        }

        #expect(executed.value == ["record:safe", "gate:pause"])
    }

    @Test func testSchema113CompletedApprovalSkipsExactReplay() async throws {
        let (agent, runConfig, _, executed) = try buildSerializedReplayScenario(replayValue: "safe")

        let first = try await Runner.run(agent, input: "record a value", runConfig: runConfig)
        let state = first.toState()
        try state.approve(first.interruptions[0])
        let second = try await Runner.run(agent, state: state, runConfig: runConfig)
        let restored = try await RunState.fromJSON(agent, legacySnapshot(try second.toState().toJSON()))
        try restored.approve(restored.getInterruptions()[0])

        let result = try await Runner.run(agent, state: restored, runConfig: runConfig)

        #expect(result.finalOutput as? String == "done")
        #expect(executed.value == ["record:safe", "gate:pause"])
    }
}
