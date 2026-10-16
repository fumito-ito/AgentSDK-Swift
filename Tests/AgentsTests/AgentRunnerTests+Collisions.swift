import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_agent_runner.py (hosted MCP approvals, tool name collisions, handoff
// enablement context, and the default runner).
//
// Swift: falsy runner/agent cases, legacy positional arguments, and log-record formatting
// internals (`record.args`) have no Swift equivalent and are skipped.
extension AgentRunnerTests {
    private func runMaybeStreamed(_ agent: Agent, input: RunInput, streamed: Bool, runConfig: RunConfig? = nil) async throws -> any RunResultBase {
        if streamed {
            let result = try Runner.runStreamed(agent, input: input, runConfig: runConfig)
            for try await _ in result.streamEvents() {}
            return result
        }
        return try await Runner.run(agent, input: input, runConfig: runConfig)
    }

    private func resumeMaybeStreamed(_ agent: Agent, state: RunState, streamed: Bool) async throws -> any RunResultBase {
        if streamed {
            let result = try Runner.runStreamed(agent, state: state)
            for try await _ in result.streamEvents() {}
            return result
        }
        return try await Runner.run(agent, state: state)
    }

    private func toState(_ result: any RunResultBase) throws -> RunState {
        if let result = result as? RunResult { return result.toState() }
        return try #require(result as? RunResultStreaming).toState()
    }

    // MARK: - Hosted MCP approvals

    @Test(arguments: [false, true])
    func testPersistentHostedMCPApprovalDoesNotCrossServers(streamed: Bool) async throws {
        let model = try ScriptedModel()
        let serverA = HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "server-a", "server_url": "https://server-a.example/mcp"])
        let serverB = HostedMCPTool(toolConfig: ["type": "mcp", "server_label": "server-b", "server_url": "https://server-b.example/mcp"])
        let outputs: [[TResponseOutputItem]] = ["a", "b"].map { label in
            [["id": .string("request-\(label)"), "type": "mcp_approval_request", "arguments": "{}", "name": "lookup_account", "server_label": .string("server-\(label)")]]
        }
        try model.extend(outputs.map { streamed ? getExactOutputStreamStep($0) : .output($0) })
        let agent = Agent(name: "test", tools: [serverA, serverB], model: .instance(model))

        let first = try await runMaybeStreamed(agent, input: "hello", streamed: streamed)
        #expect(first.interruptions.count == 1)
        #expect(first.interruptions.first?.rawItem.string("server_label") == "server-a")

        let state = try toState(first)
        try state.approve(first.interruptions[0], alwaysApprove: true)
        let resumed = try await resumeMaybeStreamed(agent, state: state, streamed: streamed)

        #expect(resumed.interruptions.count == 1)
        #expect(resumed.interruptions.first?.rawItem.string("server_label") == "server-b")
    }

    // MARK: - Tool name collisions

    enum CollisionSurface: String, CaseIterable, Sendable {
        case agentTool = "agent_tool", handoff, mixed
    }

    @Test(arguments: CollisionSurface.allCases, [false, true])
    func testRunReportsDerivedAgentNameCollisionsBeforeModelCall(surface: CollisionSurface, streamed: Bool) async throws {
        for policy in [ToolNameCollisionPolicy.warn, .error] {
            let model = try ScriptedModel(steps: [.output([getTextMessage("done")])])
            let billing = Agent(name: "Billing Agent")
            let normalizedBilling = Agent(name: "billing agent")
            let agent: Agent
            switch surface {
            case .agentTool:
                agent = Agent(name: "triage", tools: [
                    try billing.asTool(toolDescription: "First billing agent"),
                    try normalizedBilling.asTool(toolDescription: "Second billing agent"),
                ], model: .instance(model))
            case .handoff:
                agent = Agent(name: "triage", handoffs: [.agent(billing), .agent(normalizedBilling)], model: .instance(model))
            case .mixed:
                agent = Agent(
                    name: "triage",
                    tools: [try Agent(name: "transfer to Billing Agent").asTool(toolDescription: "Billing tool")],
                    handoffs: [.agent(billing)],
                    model: .instance(model)
                )
            }
            let runConfig = RunConfig(toolNameCollisionPolicy: policy)

            if policy == .error {
                let error = await expectErrorAsync(UserError.self) {
                    _ = try await runMaybeStreamed(agent, input: "Route this request", streamed: streamed, runConfig: runConfig)
                }
                let message = error?.message ?? ""
                #expect(["Ambiguous agent tool configuration", "Ambiguous handoff configuration", "Ambiguous agent routing configuration"].contains { message.contains($0) })
                #expect(model.calls.isEmpty)
            } else {
                try await AgentsDebug.withOverrides(dontLogToolData: false) {
                    try await captureLogs(level: .warning) { capture in
                        _ = try await runMaybeStreamed(agent, input: "Route this request", streamed: streamed, runConfig: runConfig)
                        #expect(!model.calls.isEmpty)
                        let collisionMessages = capture.messages.filter { $0.hasPrefix("Ambiguous ") }
                        #expect(collisionMessages.count == 1)
                        #expect(collisionMessages.first?.contains("Pass an explicit") == true)
                    }
                }
            }
        }
    }

    @Test(arguments: [false, true])
    func testRunWarnsAndKeepsLastDuplicateFunctionTool(streamed: Bool) async throws {
        let calls = SharedBox<[String]>([])
        let firstLookup = try FunctionTool.create(name: "lookup") { _ in
            calls.mutate { $0.append("first") }
            return "first"
        }
        let secondLookup = try FunctionTool.create(name: "lookup") { _ in
            calls.mutate { $0.append("second") }
            return "second"
        }
        let model = try ScriptedModel(steps: [.output([getFunctionToolCall("lookup", arguments: "{}")])])
        try model.enqueue(.output([getTextMessage("done")]))
        let agent = Agent(name: "agent", tools: [firstLookup, secondLookup], model: .instance(model))

        try await AgentsDebug.withOverrides(dontLogToolData: false) {
            try await captureLogs(level: .warning) { capture in
                _ = try await runMaybeStreamed(agent, input: "Look this up", streamed: streamed)
                #expect(calls.value == ["second"])
                #expect(model.calls.first?.tools.count == 1)
                #expect(model.calls.first?.tools.first as? FunctionTool === secondLookup)
                let collisionMessages = capture.messages.filter { $0.hasPrefix("Ambiguous ") }
                #expect(collisionMessages.count == 2)
                #expect(collisionMessages.allSatisfy {
                    $0 == "Ambiguous function tool configuration: the tool name `lookup` is used by multiple tools. "
                        + "Assign a unique routed name to every colliding function tool with `name_override=`, `tool_name=`, or a namespace."
                })
            }
        }
    }

    @Test func testCollisionWarningRedactsToolData() async throws {
        let secretToolName = "tenant_secret_tool_token"
        let first = try FunctionTool.create(name: secretToolName) { _ in "first" }
        let second = try FunctionTool.create(name: secretToolName) { _ in "second" }

        try await AgentsDebug.withOverrides(dontLogToolData: true) {
            try await captureLogs(level: .warning) { capture in
                let resolved = try resolveToolNameCollisions([first, second], collisionPolicy: .warn)
                #expect(resolved.tools.count == 1)
                #expect(resolved.tools.first as? FunctionTool === second)
                #expect(resolved.handoffs.isEmpty)
                #expect(capture.messages == ["Tool name collision detected. Assign unique routed tool names or enable tool data logging for details."])
                #expect(!capture.messages.joined().contains(secretToolName))
            }
        }
    }

    @Test func testCollisionWarningPreservesToolDiagnosticsWhenEnabled() async throws {
        let toolName = "diagnostic_tool_name"
        let first = try FunctionTool.create(name: toolName) { _ in "first" }
        let second = try FunctionTool.create(name: toolName) { _ in "second" }

        try await AgentsDebug.withOverrides(dontLogToolData: false) {
            try await captureLogs(level: .warning) { capture in
                _ = try resolveToolNameCollisions([first, second], collisionPolicy: .warn)
                #expect(capture.records.count == 1)
                #expect(capture.messages.first?.contains(toolName) == true)
            }
        }
    }

    @Test(arguments: [false, true])
    func testRunRejectsDuplicateFunctionToolsInErrorMode(streamed: Bool) async throws {
        let first = try FunctionTool.create(name: "lookup") { _ in "first" }
        let second = try FunctionTool.create(name: "lookup") { _ in "second" }
        let model = try ScriptedModel(steps: [.output([getTextMessage("done")])])
        let agent = Agent(name: "agent", tools: [first, second], model: .instance(model))

        await expectErrorAsync(UserError.self, containing: "the tool name `lookup` is used by multiple tools") {
            _ = try await runMaybeStreamed(agent, input: "Look this up", streamed: streamed, runConfig: RunConfig(toolNameCollisionPolicy: .error))
        }
        #expect(model.calls.isEmpty)
    }

    @Test func testRunWarnsOnceForRepeatedSourceAgentName() async throws {
        let model = try ScriptedModel(steps: [.output([getTextMessage("done")])])
        let agent = Agent(name: "orchestrator", tools: [
            try Agent(name: "Refund").asTool(toolDescription: "First refund agent"),
            try Agent(name: "refund").asTool(toolDescription: "Second refund agent"),
            try Agent(name: "refund").asTool(toolDescription: "Third refund agent"),
        ], model: .instance(model))

        try await AgentsDebug.withOverrides(dontLogToolData: false) {
            try await captureLogs(level: .warning) { capture in
                _ = try await Runner.run(agent, input: "Route this request")
                #expect(capture.messages.filter { $0.hasPrefix("Ambiguous ") } == [
                    "Ambiguous function tool configuration: the tool name `refund` is used by multiple tools. "
                        + "Assign a unique routed name to every colliding function tool with `name_override=`, `tool_name=`, or a namespace.",
                ])
                #expect(model.calls.first?.tools.count == 1)
                #expect(model.calls.first?.tools.first as? FunctionTool === agent.tools.last as? FunctionTool)
            }
        }
    }

    @Test func testMultiwayMixedCollisionReportsEveryOwnerMustBeUnique() async throws {
        let first = try FunctionTool.create(name: "route") { _ in "first" }
        let second = try FunctionTool.create(name: "route") { _ in "second" }
        let routeHandoff = handoff(Agent(name: "Billing"), toolNameOverride: "route")

        try await AgentsDebug.withOverrides(dontLogToolData: false) {
            try await captureLogs(level: .warning) { capture in
                let resolved = try resolveToolNameCollisions([first, second], handoffs: [routeHandoff], collisionPolicy: .warn)
                #expect(resolved.tools.isEmpty)
                #expect(resolved.handoffs.count == 1)
                #expect(resolved.handoffs.first === routeHandoff)
                #expect(capture.messages == [
                    "Ambiguous tool routing configuration: the tool name `route` is used by both a function tool and a handoff. "
                        + "Assign a unique routed name to every colliding function tool and handoff with `name_override=`, "
                        + "`tool_name=`, `tool_name_override=`, or a namespace.",
                ])
            }
        }
    }

    // MARK: - Turn context

    private final class FlagContext: @unchecked Sendable {
        private let lock = NSLock()
        private var flags: [String: Bool] = [:]

        subscript(key: String) -> Bool? {
            get { lock.locked { flags[key] } }
            set { lock.locked { flags[key] = newValue } }
        }

        var snapshot: [String: Bool] { lock.locked { flags } }
    }

    private final class InitializeContextHooks: RunHooks, @unchecked Sendable {
        override func onAgentStart(context: AgentHookContext, agent: AgentBase) async throws {
            if agent.name == "source" {
                context.context(as: FlagContext.self)?["hook_initialized"] = true
            }
        }
    }

    @Test(arguments: [false, true])
    func testHandoffEnablementUsesInitializedTurnContext(streamed: Bool) async throws {
        let model = try ScriptedModel()
        let target = Agent(name: "target", model: .instance(model))
        try model.extend([.output([getHandoffToolCall(target)]), .output([getTextMessage("done")])])
        let observed = SharedBox<[([TResponseInputItem], [String: Bool])]>([])
        let isEnabled = ToolEnablement.dynamic { context, agent in
            let flags = context.context(as: FlagContext.self)
            observed.mutate { $0.append((context.turnInput, flags?.snapshot ?? [:])) }
            return agent.name == "source"
                && context.turnInput == [["content": "current turn", "role": "user"]]
                && flags?["hook_initialized"] == true
                && flags?["prompt_initialized"] == true
        }
        let source = Agent(
            name: "source",
            prompt: .dynamic { data in
                data.context.context(as: FlagContext.self)?["prompt_initialized"] = true
                return Prompt(id: "prompt-id")
            },
            handoffs: [.handoff(handoff(target, isEnabled: isEnabled))],
            model: .instance(model)
        )
        let context = FlagContext()

        if streamed {
            let result = try Runner.runStreamed(source, input: "current turn", context: context, hooks: InitializeContextHooks())
            for try await _ in result.streamEvents() {}
        } else {
            _ = try await Runner.run(source, input: "current turn", context: context, hooks: InitializeContextHooks())
        }

        #expect(observed.value.count == 1)
        #expect(observed.value.first?.0 == [["content": "current turn", "role": "user"]])
        #expect(observed.value.first?.1 == ["hook_initialized": true, "prompt_initialized": true])
    }

    // MARK: - Default runner and config defaults

    @Test func testSetDefaultAgentRunnerRoundtrip() {
        let original = getDefaultAgentRunner()
        defer { setDefaultAgentRunner(original) }
        let runner = AgentRunner()
        setDefaultAgentRunner(runner)
        #expect(getDefaultAgentRunner() === runner)

        setDefaultAgentRunner(nil)
        #expect(getDefaultAgentRunner() !== runner)
    }

    @Test func testRunConfigDefaultsNestedHandoffHistoryOptIn() {
        #expect(!RunConfig().nestHandoffHistory)
    }
}
