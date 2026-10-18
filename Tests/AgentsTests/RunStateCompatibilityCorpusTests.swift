import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_run_state_compatibility_corpus.py and the fixture helpers in
// integration_tests/_contract_state.py.
//
// The fixtures under Fixtures/run_state are copied unchanged from upstream
// tests/fixtures/run_state. The security/ fixture and its credential tests cover sandbox state,
// which is out of scope (porting/decisions.md D2).
//
// Swift: not ported:
// - The `*_uses_immutable_expected_payload` tests monkeypatch `RunState.from_json`/`to_json`.
// - The redaction, exception-graph, traceback, and parser-state tests exercise Python exception
//   internals; Swift errors carry no tracebacks or chained causes.

private let fixtureRoot: URL = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!.appendingPathComponent("run_state")

private func loadFixture(_ relativePath: String) throws -> JSONObject {
    try JSONObject.parse(String(contentsOf: fixtureRoot.appendingPathComponent(relativePath), encoding: .utf8))
}

private let sources: JSONObject = {
    // swiftlint:disable:next force_try
    try! loadFixture("sources.json")
}()

private struct CorpusEntry: Sendable, CustomTestStringConvertible {
    var feature: String
    var version: String
    var fixture: String
    var testDescription: String { feature }
}

private let versionEntries: [CorpusEntry] = (sources["versions"]?.objectValue ?? [:])
    .sorted { $0.key < $1.key }
    .map { CorpusEntry(feature: "minimal-\($0.key)", version: $0.key, fixture: $0.value.objectValue?.string("fixture") ?? "") }

private let featureEntries: [CorpusEntry] = (sources["features"]?.arrayValue ?? []).compactMap { value in
    guard let entry = value.objectValue else { return nil }
    return CorpusEntry(feature: entry.string("feature") ?? "", version: entry.string("version") ?? "", fixture: entry.string("fixture") ?? "")
}

/// Features whose durable fields depend on modules that are not ported yet, with the reason.
private let pendingFeatures: [String: String] = [
    "duplicate_agent_identity_and_sandbox": "sandbox resume state is out of scope (D2)",
    "docker_labels": "sandbox resume state is out of scope (D2)",
    "nested_history_ownership": "RunState does not retain nested history ownership yet",
    "hosted_mcp_approval_scope": "hosted MCP query approval records are not ported yet",
    "canonical_invocation_identity": "the tool invocation ledger is not ported yet",
    "per_call_approval_override": "the tool invocation ledger and sticky approval scopes are not ported yet",
]

/// Removes the schema version, the only field a rewrite may change.
private func normalizedDurableState(_ payload: JSONObject) -> JSONObject {
    payload.removing("$schemaVersion")
}

private func pythonTypeName(_ value: JSONValue) -> String {
    switch value {
    case .object: return "dict"
    case .array: return "list"
    case .string: return "str"
    case .bool: return "bool"
    case .int: return "int"
    case .double: return "float"
    case .null: return "NoneType"
    }
}

/// Returns every place where `actual` drops or changes a value present in `expected`.
private func findSubsetErrors(_ expected: JSONValue, _ actual: JSONValue, path: String = "state") -> [String] {
    switch expected {
    case .object(let expectedObject):
        guard case .object(let actualObject) = actual else {
            return ["\(path) changed type from mapping to \(pythonTypeName(actual))"]
        }
        return expectedObject.flatMap { key, value -> [String] in
            guard let actualValue = actualObject[key] else { return ["\(path).\(key) was dropped"] }
            return findSubsetErrors(value, actualValue, path: "\(path).\(key)")
        }
    case .array(let expectedArray):
        guard case .array(let actualArray) = actual else {
            return ["\(path) changed type from list to \(pythonTypeName(actual))"]
        }
        guard expectedArray.count == actualArray.count else {
            return ["\(path) changed length from \(expectedArray.count) to \(actualArray.count)"]
        }
        return zip(expectedArray, actualArray).enumerated().flatMap { index, pair in
            findSubsetErrors(pair.0, pair.1, path: "\(path)[\(index)]")
        }
    default:
        guard pythonTypeName(expected) == pythonTypeName(actual) else {
            return ["\(path) changed type from \(pythonTypeName(expected)) to \(pythonTypeName(actual))"]
        }
        return expected == actual ? [] : ["\(path) changed from \(expected) to \(actual)"]
    }
}

private func restoreAgent(_ payload: JSONObject) -> Agent {
    let currentAgent = payload["current_agent"]?.objectValue
    let name = currentAgent?.string("name") ?? "compat-agent"
    if currentAgent?.string("identity") == "\(name)#2" {
        return Agent(name: name, handoffs: [.handoff(handoff(Agent(name: name)))])
    }
    return Agent(name: name)
}

/// Restores a historical fixture, rewrites it at the current schema, and checks that no durable
/// field is lost and that the rewrite is idempotent.
private func validateHistoricalRunStateFixture(_ relativePath: String) async throws -> [String] {
    let payload = try loadFixture(relativePath)
    let name = (relativePath as NSString).lastPathComponent
    let originalVersion = payload.string("$schemaVersion") ?? ""
    let agent = restoreAgent(payload)
    let canonical = try await RunState.fromJSON(agent, payload).toJSON()

    var errors: [String] = []
    if canonical.string("$schemaVersion") != RunState.currentSchemaVersion {
        errors.append("\(name) rewrote as \(canonical["$schemaVersion"] ?? .null), expected \(RunState.currentSchemaVersion)")
    }
    errors += findSubsetErrors(.object(normalizedDurableState(payload)), .object(normalizedDurableState(canonical))).map { "\(name): \($0)" }

    let recanonical = try await RunState.fromJSON(agent, canonical).toJSON()
    if recanonical != canonical {
        errors.append("\(name) was not idempotent after rewriting schema \(originalVersion) to \(RunState.currentSchemaVersion)")
    }
    return errors
}

private struct AccountArgs: Decodable, Sendable {
    var accountId: String
    enum CodingKeys: String, CodingKey { case accountId = "account_id" }
}

/// Resumes a historical fixture and checks that its approval decision controls the tool.
private func validateHistoricalResumeBehavior(_ relativePath: String, feature: String, decision: String?) async throws -> [String] {
    let invocations = SharedBox(0)
    let toolName = feature == "canonical_invocation_identity" ? "lookup_account" : "historical_approval"
    let tool = try FunctionTool.create(name: toolName, argumentsType: AccountArgs.self, needsApproval: true) { _, args in
        invocations.mutate { $0 += 1 }
        return "approved:\(args.accountId)"
    }
    var turns: [[TResponseOutputItem]] = []
    let expectedInvocations: Int
    let expectedToolOutput: String
    switch (feature, decision) {
    case ("canonical_invocation_identity", _):
        turns.append([ResponseFunctionToolCall(callId: "function-request-1", name: "lookup_account", arguments: #"{"account_id":"account-1"}"#, status: "completed").json])
        (expectedInvocations, expectedToolOutput) = (1, "approved:account-1")
    case ("pending_tool_approval", "approve"):
        (expectedInvocations, expectedToolOutput) = (1, "approved:account-1")
    case ("pending_tool_approval", "reject"):
        (expectedInvocations, expectedToolOutput) = (0, "Candidate rejected historical approval")
    default:
        throw ValueError("Unsupported historical resume feature: \(feature)")
    }
    turns.append([ResponseOutputMessage(id: "historical-resume-final", content: [ResponseOutputText(text: "resume complete").json]).json])
    let model = try ScriptedModel(steps: turns.map { ModelStep.output($0, responseId: "queued-fake-response") })
    let agent = Agent(name: "compat-agent", tools: [tool], model: .instance(model))
    let restored = try await RunState.fromJSON(agent, try loadFixture(relativePath))
    let name = (relativePath as NSString).lastPathComponent
    if feature == "pending_tool_approval" {
        let interruptions = restored.getInterruptions()
        guard interruptions.count == 1 else { return ["\(name) did not restore its historical pending approval"] }
        if decision == "approve" {
            try restored.approve(interruptions[0])
        } else {
            try restored.reject(interruptions[0], rejectionMessage: "Candidate rejected historical approval")
        }
    }
    let result = try await Runner.run(agent, state: restored)

    var errors: [String] = []
    if !result.interruptions.isEmpty {
        errors.append("\(name) interrupted instead of applying its historical decision")
    }
    if invocations.value != expectedInvocations {
        errors.append("\(name) invoked its approval-controlled tool \(invocations.value) times, expected \(expectedInvocations)")
    }
    let toolOutputs = result.newItems.compactMap { ($0 as? ToolCallOutputItem)?.output as? String }
    if !toolOutputs.contains(expectedToolOutput) {
        errors.append("\(name) did not preserve the historical tool decision output \(expectedToolOutput)")
    }
    if result.finalOutput as? String != "resume complete" {
        errors.append("\(name) did not complete its resumed run")
    }
    return errors
}

@Suite struct RunStateCompatibilityCorpusTests {
    // MARK: - Comparison helper

    @Test(arguments: ["no_active_agent_run", "last_model_response", "generated_session_item_indexes", "conversation_id", "input_guardrail_results", "tool_use_tracker"])
    func testHistoricalStateComparisonCoversEveryDurableField(fieldName: String) {
        let historical: JSONObject = ["$schemaVersion": "1.0", fieldName: ["value": "preserve-me"]]
        let canonical: JSONObject = ["$schemaVersion": "1.15"]

        let errors = findSubsetErrors(.object(normalizedDurableState(historical)), .object(normalizedDurableState(canonical)))

        #expect(errors == ["state.\(fieldName) was dropped"])
    }

    @Test func testHistoricalStateComparisonPreservesJSONScalarTypes() {
        let errors = findSubsetErrors(["no_active_agent_run": true, "current_turn": 1], ["no_active_agent_run": 1, "current_turn": 1.0])

        #expect(errors == [
            "state.no_active_agent_run changed type from bool to int",
            "state.current_turn changed type from int to float",
        ])
    }

    // MARK: - Historical fixtures

    @Test func testHistoricalFixtureCorpusMatchesSupportedSchemaVersions() {
        #expect(sources["baseline"] == "v0.19.4")
        let versions = Set((sources["versions"]?.objectValue ?? [:]).keys)
        // The current writer can introduce a version before a historical writer exists.
        #expect(versions.union([RunState.currentSchemaVersion]) == RunState.supportedSchemaVersions)
        #expect((sources["versions"]?.objectValue ?? [:]).values.allSatisfy { !($0.objectValue?.string("commit") ?? "").isEmpty })
        #expect(Set(featureEntries.map(\.version)) == versions.subtracting(["1.0", "1.1"]))
        let provenances = Set((sources["features"]?.arrayValue ?? []).compactMap { $0.objectValue?.string("provenance") })
        #expect(provenances == ["historical_writer", "canonical_compatibility"])
    }

    @Test(arguments: versionEntries)
    private func testHistoricalMinimalRunStateRewritesIdempotently(entry: CorpusEntry) async throws {
        let payload = try loadFixture(entry.fixture)
        #expect(payload.string("$schemaVersion") == entry.version)
        #expect(try await validateHistoricalRunStateFixture(entry.fixture) == [])
    }

    @Test(arguments: featureEntries)
    private func testHistoricalFeatureRunStateRewritesSemantically(entry: CorpusEntry) async throws {
        let payload = try loadFixture(entry.fixture)
        #expect(payload.string("$schemaVersion") == entry.version)
        let errors = try await validateHistoricalRunStateFixture(entry.fixture)
        if let reason = pendingFeatures[entry.feature] {
            withKnownIssue(Comment(rawValue: reason)) {
                #expect(errors == [])
            }
        } else {
            #expect(errors == [])
        }
    }

    @Test(arguments: [("pending_tool_approval", "approve"), ("pending_tool_approval", "reject"), ("canonical_invocation_identity", nil)] as [(String, String?)])
    func testHistoricalApprovalDecisionsControlResumedRuns(feature: String, decision: String?) async throws {
        let fixture: String
        if feature == "pending_tool_approval" {
            fixture = sources["resume"]?.objectValue?.string("fixture") ?? ""
        } else {
            fixture = featureEntries.first { $0.feature == feature }?.fixture ?? ""
        }
        #expect(try await validateHistoricalResumeBehavior(fixture, feature: feature, decision: decision) == [])
    }

    @Test(arguments: [
        ("missing_version.json", "missing schema version"),
        ("future_version.json", "schema version is not supported"),
        ("malformed_current_agent.json", "Run state agent not found in agent map"),
    ])
    func testInvalidRunStateFixturesFailWithoutDisclosingValues(fixtureName: String, message: String) async throws {
        let payload = try loadFixture("negative/\(fixtureName)")
        let sentinel = "RUNSTATE_SECRET_SENTINEL_42"
        #expect(JSONValue.object(payload).jsonString().contains(sentinel))

        let (error, messages) = await captureLogs(level: .debug) { capture in
            (await expectErrorAsync(UserError.self, containing: message) {
                _ = try await RunState.fromJSON(Agent(name: "compat-agent"), payload)
            }, capture.messages)
        }

        #expect(error.map { !describeError($0).contains(sentinel) } == true)
        #expect(!messages.contains { $0.contains(sentinel) })
    }

    // MARK: - Context wrappers

    private final class ForeignContextWrapper: RunContextWrapper, @unchecked Sendable {}

    @Test(arguments: ["context_override", "context_deserializer"])
    func testRunStateRejectsContextWrapperSubclassesBeforeRestoration(entryPoint: String) async throws {
        // Swift: the subclass cannot intercept restoration, so this checks the rejection and
        // that the foreign wrapper is left untouched.
        let context = ForeignContextWrapper(context: ["custom": true] as JSONObject)
        var payload = try loadFixture("features/v1_15_canonical_invocation_identity.json")
        var contextEntry = payload["context"]?.objectValue ?? [:]
        contextEntry["tool_input"] = ["durable": "expected"]
        payload["context"] = .object(contextEntry)

        let error = await expectErrorAsync(UserError.self, containing: "RunState restoration does not support RunContextWrapper subclasses") {
            if entryPoint == "context_override" {
                _ = try await RunState.fromJSON(Agent(name: "compat-agent"), payload, contextOverride: context)
            } else {
                _ = try await RunState.fromJSON(Agent(name: "compat-agent"), payload) { _ in context }
            }
        }

        #expect(error.map { describeError($0).hasSuffix("provide the custom context value directly or wrap it in RunContextWrapper.") } == true)
        #expect(context.usage.requests == 0)
        #expect(context.approvals.isEmpty)
        #expect(context.toolInput == nil)
    }

    @Test func testRunStateRestoresAnExactContextWrapperWithoutReplacingIt() async throws {
        let context = RunContextWrapper(context: ["custom": true] as JSONObject)
        var payload = try loadFixture("features/v1_15_canonical_invocation_identity.json")
        var contextEntry = payload["context"]?.objectValue ?? [:]
        contextEntry["tool_input"] = ["durable": "expected"]
        payload["context"] = .object(contextEntry)

        let restored = try await RunState.fromJSON(Agent(name: "compat-agent"), payload) { _ in context }

        #expect(restored.context === context)
        #expect(context.context(as: JSONObject.self) == ["custom": true])
        #expect(context.toolInput as? JSONValue == ["durable": "expected"])
        // Swift: upstream also asserts the restored `_tool_invocations` ledger, which is not
        // ported yet.
    }
}
