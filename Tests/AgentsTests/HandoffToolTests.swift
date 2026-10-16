import Foundation
import Testing
@testable import Agents

// Ported from tests/test_handoff_tool.py.
//
// Swift: `on_handoff` signature validation, `input_type` without `on_handoff`, non-callable
// callbacks, and callable-object awaiting are enforced by the type system, so those tests are
// skipped. `test_handoff_lenient_json_allows_type_coercion` covers pydantic's lenient mode and is
// skipped (Swift decoding is always strict).
@Suite struct HandoffToolTests {
    private struct Foo: Codable, Sendable, Equatable {
        var bar: String
    }

    private struct StrictInput: Codable, Sendable {
        var name: String
        var age: Int
    }

    private func messageItem(_ content: String, _ agent: Agent) -> MessageOutputItem {
        MessageOutputItem(agent: agent, rawItem: [
            "id": "123",
            "status": "completed",
            "role": "assistant",
            "type": "message",
            "content": [["text": .string(content), "type": "output_text", "annotations": [], "logprobs": []]],
        ])
    }

    private func getLen(_ data: HandoffInputData) -> Int {
        let inputLength: Int
        if case .items(let items) = data.inputHistory { inputLength = items.count } else { inputLength = 1 }
        return inputLength + data.preHandoffItems.count + data.newItems.count
    }

    // MARK: - Setup

    @Test func testSingleHandoffSetup() async throws {
        let agent1 = Agent(name: "test_1")
        let agent2 = Agent(name: "test_2", handoffs: [.agent(agent1)])

        #expect(agent1.handoffs.isEmpty)
        #expect(agent2.handoffs.count == 1)
        #expect(try await getHandoffs(agent1, RunContextWrapper(context: agent1)).isEmpty)

        let handoffs = try await getHandoffs(agent2, RunContextWrapper(context: agent2))
        #expect(handoffs.count == 1)
        #expect(handoffs[0].toolName == Handoff.defaultToolName(agent1))
        #expect(handoffs[0].toolDescription == Handoff.defaultToolDescription(agent1))
        #expect(handoffs[0].agentName == agent1.name)
    }

    @Test func testMultipleHandoffsSetup() async throws {
        let agent1 = Agent(name: "test_1")
        let agent2 = Agent(name: "test_2")
        let agent3 = Agent(name: "test_3", handoffs: [.agent(agent1), .agent(agent2)])

        let handoffs = try await getHandoffs(agent3, RunContextWrapper(context: agent3))
        #expect(handoffs.count == 2)
        #expect(handoffs.map(\.toolName) == [Handoff.defaultToolName(agent1), Handoff.defaultToolName(agent2)])
        #expect(handoffs.map(\.toolDescription) == [Handoff.defaultToolDescription(agent1), Handoff.defaultToolDescription(agent2)])
        #expect(handoffs.map(\.agentName) == [agent1.name, agent2.name])
    }

    // MARK: - Name collisions

    @Test func testHandoffsRejectCollidingDerivedNames() async throws {
        let triage = Agent(name: "triage", handoffs: [.agent(Agent(name: "Billing Agent")), .agent(Agent(name: "billing agent"))])
        let handoffs = try await getHandoffs(triage, RunContextWrapper())

        let error = await expectErrorAsync(UserError.self) {
            _ = try resolveToolNameCollisions([], handoffs: handoffs, collisionPolicy: .error)
        }
        #expect(error?.message == "Ambiguous handoff configuration: agents 'Billing Agent' and 'billing agent' both derive "
            + "the handoff tool name `transfer_to_billing_agent`. Pass an explicit `tool_name_override=` to one of them.")
    }

    @Test func testHandoffDerivedNameCollisionAllowsExplicitOverride() async throws {
        let triage = Agent(name: "triage", handoffs: [
            .agent(Agent(name: "Billing Agent")),
            .handoff(handoff(Agent(name: "billing agent"), toolNameOverride: "transfer_to_normalized_billing")),
        ])

        let handoffs = try await getHandoffs(triage, RunContextWrapper())

        #expect(handoffs.map(\.toolName) == ["transfer_to_billing_agent", "transfer_to_normalized_billing"])
    }

    @Test func testHandoffRejectsDistinctAgentsWithTheSameName() async throws {
        let triage = Agent(name: "triage", handoffs: [.agent(Agent(name: "Billing")), .agent(Agent(name: "Billing"))])
        let handoffs = try await getHandoffs(triage, RunContextWrapper())

        expectError(UserError.self, containing: "handoff tool name `transfer_to_billing` is used by multiple handoffs") {
            _ = try resolveToolNameCollisions([], handoffs: handoffs, collisionPolicy: .error)
        }
    }

    @Test func testHandoffWarnsAndKeepsLastDistinctAgentWithTheSameName() async throws {
        let triage = Agent(name: "triage", handoffs: [.agent(Agent(name: "Billing")), .agent(Agent(name: "Billing"))])
        let handoffs = try await getHandoffs(triage, RunContextWrapper())

        try await AgentsDebug.withOverrides(dontLogToolData: false) {
            try await captureLogs(level: .warning) { capture in
                let resolved = try resolveToolNameCollisions([], handoffs: handoffs, collisionPolicy: .warn)
                #expect(resolved.handoffs.count == 1)
                #expect(resolved.handoffs.first === handoffs[1])
                #expect(capture.messages.contains { $0.contains("handoff tool name `transfer_to_billing` is used by multiple handoffs") })
            }
        }
    }

    @Test func testHandoffsIgnoreDisabledDerivedNameCollision() async throws {
        let triage = Agent(name: "triage", handoffs: [
            .agent(Agent(name: "Billing Agent")),
            .handoff(handoff(Agent(name: "billing agent"), isEnabled: false)),
        ])

        let handoffs = try await getHandoffs(triage, RunContextWrapper())

        #expect(handoffs.map(\.agentName) == ["Billing Agent"])
    }

    @Test func testHandoffDefaultIdentityTracksTheCurrentToolName() async throws {
        let derived = handoff(Agent(name: "Billing Agent"))
        let copied = derived.copy()
        let renamed = derived.copy()
        renamed.toolName = "transfer_to_primary_billing"
        let normalizedBilling = Agent(name: "billing agent")

        let copiedTriage = Agent(name: "copied", handoffs: [.handoff(copied), .agent(normalizedBilling)])
        let copiedHandoffs = try await getHandoffs(copiedTriage, RunContextWrapper())
        expectError(UserError.self, containing: "Ambiguous handoff configuration") {
            _ = try resolveToolNameCollisions([], handoffs: copiedHandoffs, collisionPolicy: .error)
        }

        let renamedTriage = Agent(name: "renamed", handoffs: [.handoff(renamed), .agent(normalizedBilling)])
        let handoffs = try await getHandoffs(renamedTriage, RunContextWrapper())
        _ = try resolveToolNameCollisions([], handoffs: handoffs, collisionPolicy: .error)
        #expect(handoffs.map(\.toolName) == ["transfer_to_primary_billing", "transfer_to_billing_agent"])
    }

    @Test func testDefaultHandoffToolNameAllowsWhitespaceWithoutWarning() async {
        await captureLogs(level: .warning) { capture in
            #expect(Handoff.defaultToolName(Agent(name: "Refund agent")) == "transfer_to_refund_agent")
            #expect(capture.records.isEmpty)
        }
    }

    @Test func testDefaultHandoffToolNameWarnsForNonWhitespaceInvalidCharacters() async {
        await captureLogs(level: .warning) { capture in
            #expect(Handoff.defaultToolName(Agent(name: "Refund/agent")) == "transfer_to_refund_agent")
            #expect(capture.records.count == 1)
            #expect(capture.records.first?.message.contains("contains invalid characters for function calling") == true)
        }
    }

    @Test func testCustomHandoffSetup() async throws {
        let agent1 = Agent(name: "test_1")
        let agent2 = Agent(name: "test_2")
        let agent3 = Agent(name: "test_3", handoffs: [
            .agent(agent1),
            .handoff(handoff(agent2, toolNameOverride: "custom_tool_name", toolDescriptionOverride: "custom tool description")),
        ])

        let handoffs = try await getHandoffs(agent3, RunContextWrapper(context: agent3))
        #expect(handoffs.count == 2)
        #expect(handoffs[0].toolName == Handoff.defaultToolName(agent1))
        #expect(handoffs[0].toolDescription == Handoff.defaultToolDescription(agent1))
        #expect(handoffs[0].agentName == agent1.name)
        #expect(handoffs[1].toolName == "custom_tool_name")
        #expect(handoffs[1].toolDescription == "custom tool description")
        #expect(handoffs[1].agentName == agent2.name)
    }

    // MARK: - Typed input

    @Test func testHandoffInputType() async throws {
        let agent = Agent(name: "test")
        let object = try handoff(agent, inputType: Foo.self, onHandoff: { _, _ in })
        let expected = try ensureStrictJSONSchema(JSONSchemaGenerator.schema(for: Foo.self))
        for (key, value) in expected {
            #expect(object.inputJSONSchema[key] == value)
        }

        await expectErrorAsync(ModelBehaviorError.self) {
            _ = try await object.onInvokeHandoff(RunContextWrapper(context: agent), "not json")
        }
        await expectErrorAsync(ModelBehaviorError.self) {
            _ = try await object.onInvokeHandoff(RunContextWrapper(context: agent), "")
        }

        let invoked = try await object.onInvokeHandoff(RunContextWrapper(context: agent), #"{"bar":"baz"}"#)
        #expect(invoked === agent)
    }

    @Test func testOnHandoffCalled() async throws {
        let received = SharedBox<[Foo]>([])
        let agent = Agent(name: "test")
        let object = try handoff(agent, inputType: Foo.self, onHandoff: { _, input in received.mutate { $0.append(input) } })

        let invoked = try await object.onInvokeHandoff(RunContextWrapper(context: agent), #"{"bar":"baz"}"#)

        #expect(invoked === agent)
        #expect(received.value == [Foo(bar: "baz")], "on_handoff should have been called")
    }

    @Test func testOnHandoffWithoutInputCalled() async throws {
        let wasCalled = SharedBox(false)
        let agent = Agent(name: "test")
        let object = handoff(agent, onHandoff: { _ in wasCalled.mutate { $0 = true } })

        let invoked = try await object.onInvokeHandoff(RunContextWrapper(context: agent), "")

        #expect(invoked === agent)
        #expect(wasCalled.value, "on_handoff should have been called")
    }

    @Test func testAsyncOnHandoffWithoutInputCalled() async throws {
        let wasCalled = SharedBox(false)
        let agent = Agent(name: "test")
        let object = handoff(agent, onHandoff: { _ in
            await Task.yield()
            wasCalled.mutate { $0 = true }
        })

        let invoked = try await object.onInvokeHandoff(RunContextWrapper(context: agent), "")

        #expect(invoked === agent)
        #expect(wasCalled.value, "on_handoff should have been called")
    }

    // MARK: - Input data

    @Test func testHandoffInputData() {
        let agent = Agent(name: "test")

        var data = HandoffInputData(inputHistory: "", preHandoffItems: [], newItems: [], runContext: RunContextWrapper())
        #expect(getLen(data) == 1)

        data = HandoffInputData(inputHistory: .items([["role": "user", "content": "foo"]]), preHandoffItems: [], newItems: [], runContext: RunContextWrapper())
        #expect(getLen(data) == 1)

        data = HandoffInputData(
            inputHistory: .items([["role": "user", "content": "foo"], ["role": "assistant", "content": "bar"]]),
            preHandoffItems: [],
            newItems: [],
            runContext: RunContextWrapper()
        )
        #expect(getLen(data) == 2)

        data = HandoffInputData(
            inputHistory: .items([["role": "user", "content": "foo"]]),
            preHandoffItems: [messageItem("foo", agent), messageItem("foo2", agent)],
            newItems: [messageItem("bar", agent), messageItem("baz", agent)],
            runContext: RunContextWrapper()
        )
        #expect(getLen(data) == 5)

        data = HandoffInputData(
            inputHistory: .items([["role": "user", "content": "foo"], ["role": "assistant", "content": "bar"]]),
            preHandoffItems: [messageItem("baz", agent)],
            newItems: [messageItem("baz", agent), messageItem("qux", agent)],
            runContext: RunContextWrapper()
        )
        #expect(getLen(data) == 5)
    }

    @Test func testHandoffInputSchemaIsStrict() throws {
        let object = try handoff(Agent(name: "test"), inputType: Foo.self, onHandoff: { _, _ in })

        #expect(object.strictJSONSchema, "Input schema should be strict")
        #expect(object.inputJSONSchema["additionalProperties"] == false, "Input schema should be strict and have additionalProperties=False")
    }

    @Test func testHandoffRejectsNullableInputRoot() {
        expectError(UserError.self, containing: "root of a strict JSON schema") {
            _ = try handoff(Agent(name: "test"), inputType: Foo?.self, onHandoff: { _, _ in })
        }
    }

    @Test func testGetTransferMessageIsValidJSON() throws {
        let agent = Agent(name: "foo")
        let transfer = handoff(agent).getTransferMessage(agent)
        #expect(try JSONValue.parse(transfer) == ["assistant": .string(agent.name)])
    }

    // MARK: - Enablement

    @Test func testHandoffIsEnabledBool() async throws {
        let agent = Agent(name: "test")
        #expect(try await handoff(agent).isEnabled.resolve(RunContextWrapper(), agent))
        #expect(try await handoff(agent, isEnabled: true).isEnabled.resolve(RunContextWrapper(), agent))
        #expect(try await !handoff(agent, isEnabled: false).isEnabled.resolve(RunContextWrapper(), agent))
    }

    @Test func testHandoffIsEnabledCallable() async throws {
        let agent = Agent(name: "test")
        let enabled = handoff(agent, isEnabled: .dynamic { _, _ in true })
        #expect(try await enabled.isEnabled.resolve(RunContextWrapper(context: agent), agent))
        let disabled = handoff(agent, isEnabled: .dynamic { _, _ in false })
        #expect(try await !disabled.isEnabled.resolve(RunContextWrapper(context: agent), agent))
        let asyncEnabled = handoff(agent, isEnabled: .dynamic { _, _ in
            await Task.yield()
            return true
        })
        #expect(try await asyncEnabled.isEnabled.resolve(RunContextWrapper(context: agent), agent))
    }

    @Test func testHandoffIsEnabledFilteringIntegration() async throws {
        let mainAgent = Agent(name: "main_agent", handoffs: [
            .handoff(handoff(Agent(name: "agent_1"), isEnabled: true)),
            .handoff(handoff(Agent(name: "agent_2"), isEnabled: false)),
            .handoff(handoff(Agent(name: "agent_3"), isEnabled: .dynamic { _, _ in true })),
        ])

        let filtered = try await getHandoffs(mainAgent, RunContextWrapper(context: mainAgent))

        #expect(filtered.count == 2)
        #expect(Set(filtered.map(\.agentName)) == ["agent_1", "agent_3"])
    }

    @Test func testGetHandoffsCancelsSiblingEnablementChecksOnError() async throws {
        let slowStarted = AsyncEvent()
        let slowCancelled = AsyncEvent()
        let slowFinished = AsyncEvent()
        let parent = Agent(name: "parent", handoffs: [
            .handoff(handoff(Agent(name: "slow"), isEnabled: .dynamic { _, _ in
                slowStarted.set()
                defer { slowFinished.set() }
                do {
                    try await waitForever()
                } catch is CancellationError {
                    slowCancelled.set()
                    throw CancellationError()
                }
                return true
            })),
            .handoff(handoff(Agent(name: "failing"), isEnabled: .dynamic { _, _ in
                try await slowStarted.wait()
                throw TestError("enablement failed")
            })),
        ])

        await expectErrorAsync(TestError.self, containing: "enablement failed") {
            _ = try await getHandoffs(parent, RunContextWrapper())
        }
        #expect(slowCancelled.isSet)
        #expect(slowFinished.isSet)
    }

    @Test func testHandoffIsEnabledSyncCallableFalseFiltersHandoff() async throws {
        let mainAgent = Agent(name: "main", handoffs: [.handoff(handoff(Agent(name: "target"), isEnabled: .dynamic { _, _ in false }))])
        #expect(try await getHandoffs(mainAgent, RunContextWrapper(context: mainAgent)).isEmpty)
    }

    @Test func testHandoffDirectSyncIsEnabledCallableFiltersHandoff() async throws {
        let object = Handoff(
            toolName: "transfer_to_target",
            toolDescription: "Transfer to target.",
            inputJSONSchema: [:],
            onInvokeHandoff: { _, _ in Agent(name: "target") },
            agentName: "target",
            isEnabled: .dynamic { _, _ in false }
        )
        let mainAgent = Agent(name: "main", handoffs: [.handoff(object)])

        #expect(try await getHandoffs(mainAgent, RunContextWrapper(context: mainAgent)).isEmpty)
    }

    // MARK: - Strict JSON

    @Test func testHandoffStrictJSONRejectsTypeCoercion() async throws {
        let agent = Agent(name: "test")
        let object = try handoff(agent, inputType: StrictInput.self, onHandoff: { _, _ in })
        #expect(object.strictJSONSchema)

        await expectErrorAsync(ModelBehaviorError.self, containing: "Invalid JSON") {
            _ = try await object.onInvokeHandoff(RunContextWrapper(context: agent), #"{"name": "Alice", "age": "25"}"#)
        }

        let result = try await object.onInvokeHandoff(RunContextWrapper(context: agent), #"{"name": "Alice", "age": 25}"#)
        #expect(result === agent)
    }
}
