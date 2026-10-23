import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_scoped_approvals.py.
//
// Swift: the context-converter cases (`test_context_converters_preserve_function_decisions`,
// `test_context_converters_preserve_distinct_same_named_owners`,
// `test_python_context_conversion_preserves_mixed_hosted_decisions`) round-trip a
// RunContextWrapper through pydantic and Temporal, which have no Swift equivalent. The runner cases
// that use a converted context are ported with the wrapper reused directly, which keeps their
// owner-binding checks. `test_function_decision_cannot_skip_sandbox_operation_approval` is out of
// scope (sandbox, porting/decisions.md D2).
@Suite struct AgentScopedApprovalsTests {
    private func toolOutputs(_ items: [RunItem]) -> [String] {
        items.compactMap { ($0 as? ToolCallOutputItem).map { stringify($0.output) } }
    }

    private func step(_ output: [TResponseOutputItem], streamed: Bool) -> ModelStep {
        streamed ? getExactOutputStreamStep(output) : .output(output)
    }

    /// The result fields these tests inspect, from either a streamed or a non-streamed run.
    private struct Outcome {
        var finalOutput: (any Sendable)?
        var interruptions: [ToolApprovalItem]
        var newItems: [RunItem]
        var makeState: () -> RunState

        func toState() -> RunState { makeState() }
    }

    private func outcome(_ result: RunResult) -> Outcome {
        Outcome(finalOutput: result.finalOutput, interruptions: result.interruptions, newItems: result.newItems) { result.toState() }
    }

    private func outcome(_ result: RunResultStreaming) async throws -> Outcome {
        for try await _ in result.streamEvents() {}
        return Outcome(finalOutput: result.finalOutput, interruptions: result.interruptions, newItems: result.newItems) { result.toState() }
    }

    private func run(_ agent: Agent, _ input: RunInput, streamed: Bool, context: RunContextWrapper? = nil) async throws -> Outcome {
        let context = context ?? RunContextWrapper(context: ())
        if streamed {
            return try await outcome(try Runner.runStreamed(agent, input: input, context: context))
        }
        return outcome(try await Runner.run(agent, input: input, context: context))
    }

    private func run(_ agent: Agent, _ state: RunState, streamed: Bool) async throws -> Outcome {
        if streamed {
            return try await outcome(try Runner.runStreamed(agent, state: state))
        }
        return outcome(try await Runner.run(agent, state: state))
    }

    // MARK: - Handoffs

    @Test(arguments: [false, true], [false, true])
    func testStickyApprovalStaysWithAgentAcrossHandoffs(streamed: Bool, serialized: Bool) async throws {
        for approve in [false, true] {
            let executed = SharedBox<[String]>([])
            let policies = SharedBox<[String]>([])
            let policy = ToolApprovalRequirement.dynamic { _, _, callId in
                policies.mutate { $0.append(callId) }
                return true
            }
            let firstOperation = try FunctionTool.create(name: "operate", needsApproval: policy) { _ in
                executed.mutate { $0.append("first") }
                return "first"
            }
            let secondOperation = try FunctionTool.create(name: "operate", needsApproval: policy) { _ in
                executed.mutate { $0.append("second") }
                return "second"
            }
            var first = Agent(name: "same-name", tools: [firstOperation], model: .instance(try ScriptedModel(steps: [
                step([getFunctionToolCall("operate", arguments: "{}", callId: "first-1")], streamed: streamed),
                step([getFunctionToolCall("to_second", arguments: "{}", callId: "handoff-1")], streamed: streamed),
                step([getFunctionToolCall("operate", arguments: "{}", callId: "first-2")], streamed: streamed),
                step([getTextMessage("done")], streamed: streamed),
            ])))
            var second = Agent(name: "same-name", tools: [secondOperation], model: .instance(try ScriptedModel(steps: [
                step([getFunctionToolCall("operate", arguments: "{}", callId: "second-1")], streamed: streamed),
                step([getFunctionToolCall("operate", arguments: "{}", callId: "second-2")], streamed: streamed),
                step([getFunctionToolCall("to_first", arguments: "{}", callId: "handoff-2")], streamed: streamed),
            ])))
            first.handoffs = [.handoff(handoff(second, toolNameOverride: "to_second"))]
            second.handoffs = [.handoff(handoff(first, toolNameOverride: "to_first"))]

            func roundtrip(_ state: RunState) async throws -> RunState {
                let serializedState = try state.toString()
                // Recreate the configured graph, retaining duplicate names and graph positions.
                first = first.clone()
                second = second.clone()
                first.handoffs = [.handoff(handoff(second, toolNameOverride: "to_second"))]
                second.handoffs = [.handoff(handoff(first, toolNameOverride: "to_first"))]
                return try await RunState.fromString(first, serializedState)
            }

            var result = try await run(first, "start", streamed: streamed)
            var state = result.toState()
            if approve {
                try state.approve(result.interruptions[0], alwaysApprove: true)
            } else {
                try state.reject(result.interruptions[0], alwaysReject: true, rejectionMessage: "first denied")
            }
            if serialized { state = try await roundtrip(state) }
            result = try await run(first, state, streamed: streamed)
            #expect(result.interruptions.count == 1)
            #expect(result.interruptions.first?.agent === second)
            #expect(policies.value == ["first-1", "second-1"])
            #expect(executed.value == (approve ? ["first"] : []))

            state = result.toState()
            if approve {
                try state.reject(result.interruptions[0], alwaysReject: true, rejectionMessage: "second denied")
            } else {
                try state.approve(result.interruptions[0], alwaysApprove: true)
            }
            if serialized { state = try await roundtrip(state) }
            result = try await run(first, state, streamed: streamed)
            #expect(result.finalOutput as? String == "done")
            #expect(result.interruptions.isEmpty)
            #expect(policies.value == ["first-1", "second-1"])
            #expect(executed.value == (approve ? ["first", "first"] : ["second", "second"]))
            let rejectedAgent = approve ? second : first
            let reason = approve ? "second denied" : "first denied"
            let rejections = result.newItems.compactMap { item -> String? in
                guard let output = item as? ToolCallOutputItem, output.agent === rejectedAgent,
                      ["first-1", "first-2", "second-1", "second-2"].contains(output.rawItem.string("call_id") ?? "") else { return nil }
                return stringify(output.output)
            }
            #expect(rejections == [reason, reason], "approve \(approve)")
        }
    }

    @Test(arguments: [false, true])
    func testLegacyStickyDecisionsRequireApprovalAfterRestore(approve: Bool) async throws {
        let executed = SharedBox<[String]>([])
        let policies = SharedBox<[String]>([])
        let operate = try FunctionTool.create(name: "operate", needsApproval: .dynamic { _, _, callId in
            policies.mutate { $0.append(callId) }
            return true
        }) { _ in
            executed.mutate { $0.append("called") }
            return "result"
        }
        let model = try ScriptedModel(steps: [
            .output([getFunctionToolCall("operate", arguments: "{}", callId: "pending")]),
            .output([getTextMessage("done")]),
        ])
        let source = Agent(
            name: "source",
            tools: [operate],
            model: .instance(try ScriptedModel(steps: [.output([getFunctionToolCall("transfer_to_destination", arguments: "{}", callId: "transfer")])]))
        )
        let destination = Agent(name: "destination", tools: [operate], model: .instance(model))
        source.handoffs = [.agent(destination)]
        let initial = try await Runner.run(source, input: "start")
        var serialized = try initial.toState().toJSON()
        serialized["$schemaVersion"] = "1.17"
        // The released format cannot tell which agent granted this permanent decision.
        var context = serialized["context"]?.objectValue ?? [:]
        context["approvals"] = ["operate": ["approved": .bool(approve), "rejected": .bool(!approve)]]
        serialized["context"] = .object(context)
        let restored = try await RunState.fromJSON(source, serialized)

        var result = try await Runner.run(source, state: restored)
        #expect(result.interruptions.count == 1)
        #expect(result.interruptions.first?.agent === destination)
        #expect(executed.value.isEmpty)
        #expect(policies.value == ["pending"])
        let resumed = result.toState()
        try resumed.approve(result.interruptions[0])
        result = try await Runner.run(source, state: resumed)

        #expect(result.finalOutput as? String == "done")
        #expect(executed.value == ["called"])
    }

    // MARK: - Name-only inspection

    @Test func testNameOnlyQueryDoesNotChooseBetweenAgentOwners() throws {
        let context = RunContextWrapper(context: ())
        let first = Agent(name: "same")
        let second = Agent(name: "same")
        try context.approveTool(ToolApprovalItem(agent: first, rawItem: getFunctionToolCall("operate", arguments: "{}", callId: "first")), alwaysApprove: true)

        #expect(context.isToolApproved(toolName: "operate", callId: "future") == true)
        #expect(try context.getApprovalStatus(toolName: "operate", callId: "future") == nil)
        try context.rejectTool(
            ToolApprovalItem(agent: second, rawItem: getFunctionToolCall("operate", arguments: "{}", callId: "second")),
            alwaysReject: true,
            rejectionMessage: "second denied"
        )

        #expect(context.isToolApproved(toolName: "operate", callId: "future") == nil)
        #expect(context.getRejectionMessage(toolName: "operate", callId: "future") == nil)
    }

    // Swift: only the `converter="none"` case; the converters are pydantic and Temporal.
    @Test func testNameOnlyInspectionSelectsAUniqueExactCall() throws {
        let first = Agent(name: "same")
        let second = Agent(name: "same")
        let context = RunContextWrapper(context: ["tenant": "synthetic"] as JSONObject)
        let approved = ToolApprovalItem(agent: first, rawItem: getFunctionToolCall("operate", arguments: "{}", callId: "call-a"))
        let rejected = ToolApprovalItem(agent: second, rawItem: getFunctionToolCall("operate", arguments: "{}", callId: "call-b"))
        try context.approveTool(approved)
        try context.rejectTool(rejected, rejectionMessage: "Second call denied")

        #expect(context.isToolApproved(toolName: "operate", callId: "call-a") == true)
        #expect(context.isToolApproved(toolName: "operate", callId: "call-b") == false)
        #expect(context.getRejectionMessage(toolName: "operate", callId: "call-a") == nil)
        #expect(context.getRejectionMessage(toolName: "operate", callId: "call-b") == "Second call denied")
        #expect(context.isToolApproved(toolName: "operate", callId: "future") == nil)
        // Inspection must not become an execution grant for the other owner.
        let otherOwner = ToolApprovalItem(agent: second, rawItem: approved.rawItem)
        #expect(try context.getApprovalStatus(toolName: "operate", callId: "call-a", currentInvocation: otherOwner) == nil)

        try context.approveTool(rejected, alwaysApprove: true)
        try context.rejectTool(approved, rejectionMessage: "First call denied")
        #expect(context.isToolApproved(toolName: "operate", callId: "call-a") == false)
        #expect(context.getRejectionMessage(toolName: "operate", callId: "call-a") == "First call denied")
        // A unique exact decision wins inspection over another owner's sticky default.
        #expect(context.isToolApproved(toolName: "operate", callId: "future") == nil)
        #expect(context.getRejectionMessage(toolName: "operate", callId: "future") == nil)
    }

    // MARK: - Native tools

    @Test(arguments: [false, true])
    func testFunctionStickyRejectionDoesNotReplaceNativeRejectionMessage(nativeSticky: Bool) async throws {
        let functionShell = try FunctionTool.create(name: "shell", needsApproval: true) { _ -> String in
            Issue.record("Rejected function must not execute")
            return ""
        }
        let shellTool = try ShellTool(executor: { _ in
            Issue.record("Rejected shell must not execute")
            return .text("")
        }, needsApproval: .required(true))
        var destinationSteps: [ModelStep] = [.output([makeShellCall("native")])]
        if nativeSticky { destinationSteps.append(.output([makeShellCall("native-again")])) }
        destinationSteps.append(.output([getTextMessage("done")]))
        let destination = Agent(name: "destination", tools: [shellTool], model: .instance(try ScriptedModel(steps: destinationSteps)))
        let source = Agent(
            name: "source",
            tools: [functionShell],
            handoffs: [.agent(destination)],
            model: .instance(try ScriptedModel(steps: [
                .output([getFunctionToolCall("shell", arguments: "{}", callId: "function")]),
                .output([getFunctionToolCall("transfer_to_destination", arguments: "{}", callId: "transfer")]),
            ]))
        )

        var result = try await Runner.run(source, input: "start")
        var state = result.toState()
        try state.reject(result.interruptions[0], alwaysReject: true, rejectionMessage: "function denied")
        result = try await Runner.run(source, state: state)
        #expect(result.interruptions.count == 1)
        #expect(result.interruptions.first?.agent === destination)
        state = result.toState()
        try state.reject(result.interruptions[0], alwaysReject: nativeSticky, rejectionMessage: "native denied")
        result = try await Runner.run(source, state: state)

        #expect(result.finalOutput as? String == "done")
        #expect(toolOutputs(result.newItems) == ["function denied"] + Array(repeating: "native denied", count: nativeSticky ? 2 : 1))
    }

    @Test func testFunctionExactApprovalPreservesNativeStickyRejection() async throws {
        let functionShell = try FunctionTool.create(name: "shell", needsApproval: true) { _ in "function result" }
        let shellTool = try ShellTool(executor: { _ in
            Issue.record("Native sticky rejection must remain in force")
            return .text("")
        }, needsApproval: .required(true))
        let first = Agent(name: "native", tools: [shellTool], model: .instance(try ScriptedModel(steps: [
            .output([makeShellCall("native-1")]),
            .output([getFunctionToolCall("to_function", arguments: "{}", callId: "handoff-1")]),
            .output([makeShellCall("native-2")]),
            .output([getTextMessage("done")]),
        ])))
        let second = Agent(name: "function", tools: [functionShell], model: .instance(try ScriptedModel(steps: [
            .output([getFunctionToolCall("shell", arguments: "{}", callId: "function-1")]),
            .output([getFunctionToolCall("to_native", arguments: "{}", callId: "handoff-2")]),
        ])))
        first.handoffs = [.handoff(handoff(second, toolNameOverride: "to_function"))]
        second.handoffs = [.handoff(handoff(first, toolNameOverride: "to_native"))]

        var result = try await Runner.run(first, input: "start")
        var state = result.toState()
        try state.reject(result.interruptions[0], alwaysReject: true, rejectionMessage: "native denied")
        result = try await Runner.run(first, state: state)
        #expect(result.interruptions.count == 1)
        #expect(result.interruptions.first?.agent === second)
        state = result.toState()
        try state.approve(result.interruptions[0])
        result = try await Runner.run(first, state: state)

        #expect(result.finalOutput as? String == "done")
        #expect(toolOutputs(result.newItems) == ["native denied", "function result", "native denied"])
    }

    // MARK: - Reused contexts

    private enum Decision: CaseIterable {
        case approve, reject, alwaysApprove, alwaysReject

        var approved: Bool { self == .approve || self == .alwaysApprove }
        var always: Bool { self == .alwaysApprove || self == .alwaysReject }
    }

    private func decide(_ context: RunContextWrapper, _ item: ToolApprovalItem, _ decision: Decision, rejectionMessage: String) throws {
        if decision.approved {
            try context.approveTool(item, alwaysApprove: decision.always)
        } else {
            try context.rejectTool(item, alwaysReject: decision.always, rejectionMessage: rejectionMessage)
        }
    }

    // Swift: upstream converts the context before the run; here the wrapper is reused directly.
    @Test(arguments: [false, true])
    func testRunnerExecutesConvertedContextDecisions(streamed: Bool) async throws {
        for decision in Decision.allCases {
            let executed = SharedBox<[String]>([])
            let operate = try FunctionTool.create(name: "operate", needsApproval: true) { _ in
                executed.mutate { $0.append("executed") }
                return "result"
            }
            let source = Agent(name: "owner", tools: [operate])
            let call = getFunctionToolCall("operate", arguments: "{}", callId: "first")
            let context = RunContextWrapper(context: ["tenant": "synthetic"] as JSONObject)
            try decide(context, ToolApprovalItem(agent: source, rawItem: call), decision, rejectionMessage: "Owner rejected")
            var steps: [[TResponseOutputItem]] = [[call]]
            if decision.always { steps.append([getFunctionToolCall("operate", arguments: "{}", callId: "next")]) }
            steps.append([getTextMessage("done")])
            source.model = .instance(try ScriptedModel(steps: steps.map { step($0, streamed: streamed) }))

            let result = try await run(source, "start", streamed: streamed, context: context)

            #expect(result.interruptions.isEmpty)
            #expect(result.finalOutput as? String == "done")
            let callCount = decision.always ? 2 : 1
            #expect(executed.value == (decision.approved ? Array(repeating: "executed", count: callCount) : []))
            #expect(toolOutputs(result.newItems) == Array(repeating: decision.approved ? "result" : "Owner rejected", count: callCount))
            // Executable rebinding must also leave a graph-serializable checkpoint.
            _ = try await RunState.fromString(source, try result.toState().toString())
        }
    }

    // Swift: upstream converts the context before the run; here the wrapper is reused directly.
    @Test(arguments: [false, true], [false, true])
    func testConvertedContextDoesNotAuthorizeSameNamedDistinctCapability(always: Bool, streamed: Bool) async throws {
        let executed = SharedBox<[String]>([])
        let lowPrivilege = try FunctionTool.create(name: "operate", needsApproval: true) { _ in "public data" }
        let privileged = try FunctionTool.create(name: "operate", needsApproval: true) { _ in
            executed.mutate { $0.append("privileged") }
            return "private data"
        }
        let source = Agent(name: "owner", tools: [lowPrivilege])
        let call = getFunctionToolCall("operate", arguments: "{}", callId: "approved")
        let context = RunContextWrapper(context: ["tenant": "synthetic"] as JSONObject)
        try context.approveTool(ToolApprovalItem(agent: source, rawItem: call), alwaysApprove: always)
        let requested = always ? getFunctionToolCall("operate", arguments: "{}", callId: "next") : call
        let configured = Agent(
            name: "owner",
            tools: [privileged],
            model: .instance(try ScriptedModel(steps: [step([requested], streamed: streamed), step([getTextMessage("done")], streamed: streamed)]))
        )

        let result = try await run(configured, "start", streamed: streamed, context: context)

        #expect(executed.value.isEmpty)
        #expect(result.interruptions.count == 1)
        #expect(result.interruptions.first?.agent === configured)
        #expect(!result.newItems.contains { $0 is ToolCallOutputItem })
        let state = result.toState()
        try state.approve(result.interruptions[0])
        let resumed = try await run(configured, state, streamed: streamed)

        #expect(resumed.finalOutput as? String == "done")
        #expect(executed.value == ["privileged"])
    }

    // Swift: upstream converts the context through JSON; here the wrapper is reused directly. The
    // `copy` case (`copy.copy` of an agent) has no Swift equivalent; `clone` covers a recreated
    // agent.
    @Test(arguments: ["target", "source", "missing", "clone"])
    func testRunnerRequiresReapprovalForUnresolvedContextOwners(ambiguity: String) async throws {
        let operate = try FunctionTool.create(name: "operate", needsApproval: true) { _ -> String in
            Issue.record("Unresolved decisions must not execute")
            return ""
        }
        let source = Agent(name: "owner", tools: [operate])
        let call = getFunctionToolCall("operate", arguments: "{}", callId: "first")
        let context = RunContextWrapper(context: ["tenant": "synthetic"] as JSONObject)
        try context.approveTool(ToolApprovalItem(agent: source, rawItem: call), alwaysApprove: true)
        if ambiguity == "source" {
            let sibling = source.clone()
            try context.approveTool(ToolApprovalItem(agent: sibling, rawItem: getFunctionToolCall("operate", arguments: "{}", callId: "other")), alwaysApprove: true)
        }
        let model = try ScriptedModel(steps: [.output([call])])
        let configured = source.clone {
            $0.name = ambiguity == "missing" ? "different" : "owner"
            $0.model = .instance(model)
        }
        if ambiguity == "target" { configured.handoffs = [.agent(source)] }

        let result = try await Runner.run(configured, input: "start", context: context)

        #expect(result.interruptions.count == 1)
        #expect(result.interruptions.first?.agent === configured)
        let state = result.toState()
        try state.reject(result.interruptions[0], rejectionMessage: "Fresh decision")
        _ = try await RunState.fromString(configured, try state.toString())
    }

    private struct ValueArgs: Decodable, Sendable { var value: String }

    // Swift: upstream converts the context before the run; here the wrapper is reused directly.
    @Test func testConvertedContextPreservesApprovedInvocation() async throws {
        let executed = SharedBox<[String]>([])
        let operate = try FunctionTool.create(name: "operate", argumentsType: ValueArgs.self, needsApproval: true) { _, args in
            executed.mutate { $0.append(args.value) }
            return args.value
        }
        let source = Agent(name: "owner", tools: [operate])
        let context = RunContextWrapper(context: ["tenant": "synthetic"] as JSONObject)
        try context.approveTool(ToolApprovalItem(agent: source, rawItem: getFunctionToolCall("operate", arguments: #"{"value":"approved"}"#, callId: "same")))
        source.model = .instance(try ScriptedModel(steps: [
            .output([getFunctionToolCall("operate", arguments: #"{"value":"changed"}"#, callId: "same")]),
            .output([getTextMessage("done")]),
        ]))

        await expectErrorAsync(ModelBehaviorError.self, containing: "reused a tool call ID") {
            _ = try await Runner.run(source, input: "start", context: context)
        }

        #expect(executed.value.isEmpty)
    }

    @Test func testReusedContextSnapshotsOnlyCurrentGraphApprovals() async throws {
        let operate = try FunctionTool.create(name: "operate", needsApproval: true) { _ in "done" }
        let first = Agent(name: "first", tools: [operate], model: .instance(try ScriptedModel(steps: [
            .output([getFunctionToolCall("operate", arguments: "{}", callId: "first")]),
            .output([getTextMessage("done")]),
        ])))
        let context = RunContextWrapper(context: ["tenant": "synthetic"] as JSONObject)
        let pending = try await Runner.run(first, input: "start", context: context)
        try context.approveTool(pending.interruptions[0], alwaysApprove: true)
        let second = Agent(name: "second", tools: [operate], model: .instance(try ScriptedModel(steps: [
            .output([getFunctionToolCall("operate", arguments: "{}", callId: "second")]),
        ])))

        let result = try await Runner.run(second, input: "start", context: context)
        #expect(result.interruptions.count == 1)
        let state = result.toState()
        try state.reject(result.interruptions[0], rejectionMessage: "Second denied")
        let restored = try await RunState.fromString(second, try state.toString())

        #expect(restored.context.getRejectionMessage(toolName: "operate", callId: "second") == "Second denied")
        // Snapshot projection does not revoke the reusable wrapper's other owner.
        let future = ToolApprovalItem(agent: first, rawItem: getFunctionToolCall("operate", arguments: "{}", callId: "future"))
        #expect(try context.getApprovalStatus(toolName: "operate", callId: "future", currentInvocation: future) == true)
    }

    // Swift: upstream converts the context before the second run; here the wrapper is reused.
    @Test func testConvertedContextDoesNotRepeatCompletedCall() async throws {
        let executed = SharedBox<[String]>([])
        let operate = try FunctionTool.create(name: "operate", needsApproval: true) { _ in
            executed.mutate { $0.append("executed") }
            return "result"
        }
        let call = getFunctionToolCall("operate", arguments: "{}", callId: "same")
        let agent = Agent(name: "owner", tools: [operate], model: .instance(try ScriptedModel(steps: [.output([call]), .output([getTextMessage("done")])])))
        let context = RunContextWrapper(context: ["tenant": "synthetic"] as JSONObject)
        try context.approveTool(ToolApprovalItem(agent: agent, rawItem: call), alwaysApprove: true)
        _ = try await Runner.run(agent, input: "start", context: context)
        agent.model = .instance(try ScriptedModel(steps: [.output([call]), .output([getTextMessage("done")])]))

        let result = try await Runner.run(agent, input: "start", context: context)

        #expect(result.finalOutput as? String == "done")
        #expect(result.interruptions.isEmpty)
        #expect(executed.value == ["executed"])
    }

    // Swift: upstream converts the context before the runs; here the wrapper is reused directly.
    @Test func testConvertedContextRetainsOtherGraphDecisions() async throws {
        for decision in Decision.allCases {
            let executed = SharedBox<[String]>([])
            let operate = try FunctionTool.create(name: "operate", needsApproval: true) { _ in
                executed.mutate { $0.append("executed") }
                return "result"
            }
            let context = RunContextWrapper(context: ["tenant": "synthetic"] as JSONObject)
            let agents = [Agent(name: "owner", tools: [operate]), Agent(name: "owner", tools: [operate])]
            let calls = ["first", "second"].map { getFunctionToolCall("operate", arguments: "{}", callId: $0) }
            for (agent, call) in zip(agents, calls) {
                try decide(context, ToolApprovalItem(agent: agent, rawItem: call), decision, rejectionMessage: "Denied")
            }
            for (agent, call) in zip(agents, calls) {
                agent.model = .instance(try ScriptedModel(steps: [.output([call]), .output([getTextMessage("done")])]))
                let result = try await Runner.run(agent, input: "start", context: context)
                #expect(result.interruptions.isEmpty)
                #expect(result.finalOutput as? String == "done")
                #expect(toolOutputs(result.newItems) == [decision.approved ? "result" : "Denied"])
            }
            #expect(executed.value == (decision.approved ? ["executed", "executed"] : []))
        }
    }

    @Test(arguments: [false, true])
    func testApprovalContextDoesNotRetainDiscardedAgentGraphs(approve: Bool) throws {
        let context = RunContextWrapper(context: ["tenant": "synthetic"] as JSONObject)
        var references: [() -> AnyObject?] = []
        for index in 0..<10 {
            // Swift: a `do` scope releases the agents at its end; `autoreleasepool` is Darwin-only.
            do {
                let child = Agent(name: "child")
                let agent = Agent(name: "owner", handoffs: [.agent(child)])
                references.append { [weak agent] in agent }
                references.append { [weak child] in child }
                let item = ToolApprovalItem(agent: agent, rawItem: getFunctionToolCall("operate", arguments: "{}", callId: String(index)))
                if approve {
                    try context.approveTool(item)
                } else {
                    try context.rejectTool(item, rejectionMessage: "Denied")
                }
            }
        }

        #expect(references.allSatisfy { $0() == nil })
        // Compact decisions remain inspectable without retaining application graphs.
        #expect(context.approvals.count == 10)
    }

    @Test(arguments: [false, true], [false, true])
    func testTiedDuplicateOwnerRestoreRequiresReapproval(tiedAtWrite: Bool, always: Bool) async throws {
        let executed = SharedBox<[String]>([])
        let call = getFunctionToolCall("operate", arguments: "{}", callId: "pending")
        func makeAgent(_ label: String, _ instructions: String) throws -> Agent {
            let operation = try FunctionTool.create(name: "operate", needsApproval: true) { _ in
                executed.mutate { $0.append(label) }
                return label
            }
            return Agent(
                name: "same",
                tools: [operation],
                instructions: instructions,
                model: .instance(try ScriptedModel(steps: [.output([call]), .output([getTextMessage("done")])]))
            )
        }
        let first = try makeAgent("first", tiedAtWrite ? "same" : "first")
        let second = try makeAgent("second", tiedAtWrite ? "same" : "second")
        let root = Agent(
            name: "root",
            handoffs: [.handoff(handoff(first, toolNameOverride: "to_first")), .handoff(handoff(second, toolNameOverride: "to_second"))],
            model: .instance(try ScriptedModel(steps: [.output([getFunctionToolCall("to_first", arguments: "{}", callId: "handoff")])]))
        )
        let result = try await Runner.run(root, input: "start")
        #expect(result.interruptions.first?.agent === first)
        let state = result.toState()
        try state.approve(result.interruptions[0], alwaysApprove: always)
        let payload = try state.toString()
        // Identical visible signatures cannot distinguish these different closures.
        let restoredFirst = try makeAgent("restored-first", "same")
        let restoredSecond = try makeAgent("restored-second", "same")
        let configured = root.clone {
            $0.handoffs = [
                .handoff(handoff(restoredSecond, toolNameOverride: "to_second")),
                .handoff(handoff(restoredFirst, toolNameOverride: "to_first")),
            ]
        }

        let restored = try await RunState.fromString(configured, payload)
        let resumed = try await Runner.run(configured, state: restored)

        #expect(resumed.interruptions.count == 1)
        #expect(executed.value.isEmpty)
    }
}
