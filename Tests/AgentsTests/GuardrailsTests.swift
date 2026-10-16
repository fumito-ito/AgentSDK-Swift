import Foundation
import Testing
@testable import Agents
import AgentsTesting

// Ported from tests/test_guardrails.py.
//
// Swift: invalid guardrail-function values and default names derived from Python function names
// cannot be expressed, so those assertions are skipped. `run_sync` variants are skipped (no
// `run_sync` in Swift). Tests that reach into private streaming tasks
// (`_input_guardrails_task`, `run_input_guardrails_with_queue`,
// `should_cancel_parallel_model_task_on_input_guardrail_trip`) are skipped; their observable
// behavior is covered by the public-API tests below.
@Suite struct GuardrailsTests {
    private let shortDelay = 0.01
    private let mediumDelay = 0.03

    private func syncGuardrail(triggers: Bool, outputInfo: String? = nil) -> InputGuardrail.Function {
        { _, _, _ in GuardrailFunctionOutput(outputInfo: outputInfo, tripwireTriggered: triggers) }
    }

    private func outputGuardrailFunction(triggers: Bool, outputInfo: String? = nil) -> OutputGuardrail.Function {
        { _, _, _ in GuardrailFunctionOutput(outputInfo: outputInfo, tripwireTriggered: triggers) }
    }

    private func drain(_ result: RunResultStreaming) async throws {
        for try await _ in result.streamEvents() {}
    }

    private func names(_ results: [InputGuardrailResult]) -> [String] { results.map { $0.guardrail.getName() } }
    private func names(_ results: [OutputGuardrailResult]) -> [String] { results.map { $0.guardrail.getName() } }

    // MARK: - Guardrail objects

    @Test func testRunInputGuardrailsReturnsEmptyForNoGuardrails() async throws {
        let result = try await runInputGuardrails(Agent(name: "test"), [], "test", RunContextWrapper())
        #expect(result.isEmpty)
    }

    @Test func testSyncInputGuardrail() async throws {
        var result = try await InputGuardrail(guardrailFunction: syncGuardrail(triggers: false))
            .run(agent: Agent(name: "test"), input: "test", context: RunContextWrapper())
        #expect(!result.output.tripwireTriggered)
        #expect(result.output.outputInfo == nil)

        result = try await InputGuardrail(guardrailFunction: syncGuardrail(triggers: true))
            .run(agent: Agent(name: "test"), input: "test", context: RunContextWrapper())
        #expect(result.output.tripwireTriggered)
        #expect(result.output.outputInfo == nil)

        result = try await InputGuardrail(guardrailFunction: syncGuardrail(triggers: true, outputInfo: "test"))
            .run(agent: Agent(name: "test"), input: "test", context: RunContextWrapper())
        #expect(result.output.tripwireTriggered)
        #expect(result.output.outputInfo as? String == "test")
    }

    @Test func testAsyncInputGuardrail() async throws {
        let guardrail = InputGuardrail { _, _, _ in
            await Task.yield()
            return GuardrailFunctionOutput(outputInfo: "test", tripwireTriggered: true)
        }
        let result = try await guardrail.run(agent: Agent(name: "test"), input: "test", context: RunContextWrapper())
        #expect(result.output.tripwireTriggered)
        #expect(result.output.outputInfo as? String == "test")
    }

    @Test func testSyncOutputGuardrail() async throws {
        var result = try await OutputGuardrail(guardrailFunction: outputGuardrailFunction(triggers: false))
            .run(context: RunContextWrapper(), agent: Agent(name: "test"), agentOutput: "test")
        #expect(!result.output.tripwireTriggered)
        #expect(result.output.outputInfo == nil)

        result = try await OutputGuardrail(guardrailFunction: outputGuardrailFunction(triggers: true))
            .run(context: RunContextWrapper(), agent: Agent(name: "test"), agentOutput: "test")
        #expect(result.output.tripwireTriggered)

        result = try await OutputGuardrail(guardrailFunction: outputGuardrailFunction(triggers: true, outputInfo: "test"))
            .run(context: RunContextWrapper(), agent: Agent(name: "test"), agentOutput: "test")
        #expect(result.output.tripwireTriggered)
        #expect(result.output.outputInfo as? String == "test")
    }

    @Test func testAsyncOutputGuardrail() async throws {
        let guardrail = OutputGuardrail { _, _, _ in
            await Task.yield()
            return GuardrailFunctionOutput(outputInfo: "test", tripwireTriggered: true)
        }
        let result = try await guardrail.run(context: RunContextWrapper(), agent: Agent(name: "test"), agentOutput: "test")
        #expect(result.output.tripwireTriggered)
        #expect(result.output.outputInfo as? String == "test")
    }

    @Test func testInputGuardrailDecorators() async throws {
        let named = inputGuardrail(name: "Custom name") { _, _, _ in
            GuardrailFunctionOutput(outputInfo: "test_2", tripwireTriggered: false)
        }
        let result = try await named.run(agent: Agent(name: "test"), input: "test", context: RunContextWrapper())
        #expect(!result.output.tripwireTriggered)
        #expect(result.output.outputInfo as? String == "test_2")
        #expect(named.getName() == "Custom name")
    }

    @Test func testOutputGuardrailDecorators() async throws {
        let named = outputGuardrail(name: "Custom name") { _, _, _ in
            GuardrailFunctionOutput(outputInfo: "test_4", tripwireTriggered: false)
        }
        let result = try await named.run(context: RunContextWrapper(), agent: Agent(name: "test"), agentOutput: "test")
        #expect(!result.output.tripwireTriggered)
        #expect(result.output.outputInfo as? String == "test_4")
        #expect(named.getName() == "Custom name")
    }

    @Test func testInputGuardrailRunInParallelDefault() {
        #expect(InputGuardrail(guardrailFunction: syncGuardrail(triggers: false)).runInParallel)
    }

    @Test func testInputGuardrailRunInParallelFalse() {
        #expect(!InputGuardrail(runInParallel: false, guardrailFunction: syncGuardrail(triggers: false)).runInParallel)
    }

    @Test func testInputGuardrailDecoratorWithRunInParallel() async throws {
        let blocking = inputGuardrail(runInParallel: false) { _, _, _ in
            GuardrailFunctionOutput(outputInfo: "blocking", tripwireTriggered: false)
        }
        #expect(!blocking.runInParallel)
        let result = try await blocking.run(agent: Agent(name: "test"), input: "test", context: RunContextWrapper())
        #expect(!result.output.tripwireTriggered)
        #expect(result.output.outputInfo as? String == "blocking")
    }

    @Test func testInputGuardrailDecoratorWithNameAndRunInParallel() {
        let guardrail = inputGuardrail(name: "custom_name", runInParallel: false) { _, _, _ in
            GuardrailFunctionOutput(outputInfo: "named_blocking", tripwireTriggered: false)
        }
        #expect(guardrail.getName() == "custom_name")
        #expect(!guardrail.runInParallel)
    }

    // MARK: - Parallel and blocking execution

    @Test(arguments: [false, true])
    func testParallelGuardrailRunsConcurrentlyWithAgent(streamed: Bool) async throws {
        let executed = AsyncEvent()
        let delay = mediumDelay
        let guardrail = inputGuardrail(runInParallel: true) { _, _, _ in
            try await sleep(seconds: delay)
            executed.set()
            return GuardrailFunctionOutput(outputInfo: "parallel_ok", tripwireTriggered: false)
        }
        let model = try ScriptedModel()
        let agent = Agent(name: "test_agent", instructions: "Reply with 'hello'", model: .instance(model), inputGuardrails: [guardrail])
        try model.enqueue(.output([getTextMessage("hello")]))

        if streamed {
            let result = try Runner.runStreamed(agent, input: "test input")
            var receivedEvents = false
            for try await _ in result.streamEvents() { receivedEvents = true }
            #expect(receivedEvents)
        } else {
            let result = try await Runner.run(agent, input: "test input")
            #expect(result.finalOutput != nil)
            #expect(result.inputGuardrailResults.count == 1)
            #expect(result.inputGuardrailResults.first?.output.outputInfo as? String == "parallel_ok")
        }
        #expect(executed.isSet)
        #expect(!model.calls.isEmpty, "Model should have been called in parallel mode")
    }

    @Test(arguments: [false, true])
    func testBlockingGuardrailPreventsAgentExecution(streamed: Bool) async throws {
        let executed = AsyncEvent()
        let delay = mediumDelay
        let guardrail = inputGuardrail(runInParallel: false) { _, _, _ in
            executed.set()
            try await sleep(seconds: delay)
            return GuardrailFunctionOutput(outputInfo: "security_violation", tripwireTriggered: true)
        }
        let model = try ScriptedModel()
        let agent = Agent(name: "test_agent", instructions: "Reply with 'hello'", model: .instance(model), inputGuardrails: [guardrail])
        try model.enqueue(.output([getTextMessage("hello")]))

        let error = await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            if streamed {
                try await drain(try Runner.runStreamed(agent, input: "test input"))
            } else {
                _ = try await Runner.run(agent, input: "test input")
            }
        }
        #expect(executed.isSet)
        #expect(error?.guardrailResult.output.outputInfo as? String == "security_violation")
        #expect(model.calls.isEmpty, "Model should not have been called")
    }

    @Test(arguments: [false, true])
    func testParallelGuardrailMayNotPreventToolExecution(streamed: Bool) async throws {
        let toolExecuted = AsyncEvent()
        let guardrailExecuted = AsyncEvent()
        let fastTool = try FunctionTool.create(name: "fast_tool") { _ in
            toolExecuted.set()
            return "tool_executed"
        }
        let guardrail = inputGuardrail(runInParallel: true) { _, _, _ in
            // Trip only after the tool ran; if parallel guardrails gated the turn this would time out.
            try await toolExecuted.wait(timeout: 5)
            guardrailExecuted.set()
            return GuardrailFunctionOutput(outputInfo: "slow_parallel_triggered", tripwireTriggered: true)
        }
        let model = try ScriptedModel()
        let agent = Agent(
            name: "agent_with_tools",
            tools: [fastTool],
            instructions: "Call the fast_tool immediately",
            model: .instance(model),
            inputGuardrails: [guardrail]
        )
        try model.enqueue(.output([getFunctionToolCall("fast_tool", arguments: "{}")]))
        try model.enqueue(.output([getTextMessage("done")]))

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            if streamed {
                try await drain(try Runner.runStreamed(agent, input: "trigger guardrail"))
            } else {
                _ = try await Runner.run(agent, input: "trigger guardrail")
            }
        }
        #expect(guardrailExecuted.isSet)
        #expect(toolExecuted.isSet, "Expected tool to execute before slow parallel guardrail triggered")
        #expect(!model.calls.isEmpty, "Model should have been called in parallel mode")
    }

    @Test func testParallelGuardrailTripCancelsModelTask() async throws {
        let modelStarted = AsyncEvent()
        let modelCancelled = AsyncEvent()
        let modelFinished = AsyncEvent()
        let guardrail = inputGuardrail(runInParallel: true) { _, _, _ in
            try await modelStarted.wait(timeout: 1)
            return GuardrailFunctionOutput(outputInfo: "parallel_tripwire", tripwireTriggered: true)
        }
        let base = try ScriptedModel()
        try base.enqueue(.output([getTextMessage("should_not_finish")]))
        let model = HookedModel(base: base, getResponse: { request, base in
            modelStarted.set()
            defer { modelFinished.set() }
            do {
                // Swift: upstream sleeps 0.02s. Under a loaded parallel test run the tripwire can
                // take longer than that to cancel the model, so the model would finish first.
                try await sleep(seconds: 5)
                return try await base.getResponse(request)
            } catch is CancellationError {
                modelCancelled.set()
                throw CancellationError()
            }
        })
        let agent = Agent(name: "parallel_tripwire_agent", instructions: "Reply with 'hello'", model: .instance(model), inputGuardrails: [guardrail])

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            _ = try await Runner.run(agent, input: "trigger guardrail")
        }

        try await modelFinished.wait(timeout: 1)
        #expect(modelStarted.isSet)
        #expect(modelCancelled.isSet)
    }

    @Test func testModelErrorCancelsParallelInputGuardrailTask() async throws {
        let guardrailStarted = AsyncEvent()
        let guardrailCancelled = AsyncEvent()
        let guardrailFinished = AsyncEvent()
        let guardrail = inputGuardrail(runInParallel: true) { _, _, _ in
            guardrailStarted.set()
            do {
                try await waitForever()
                guardrailFinished.set()
                return GuardrailFunctionOutput(outputInfo: "parallel_ok", tripwireTriggered: false)
            } catch is CancellationError {
                guardrailCancelled.set()
                throw CancellationError()
            }
        }
        let model = HookedModel(base: try ScriptedModel(), getResponse: { _, _ in
            try await guardrailStarted.wait(timeout: 1)
            throw TestError("model boom")
        })
        let agent = Agent(name: "model_error_agent", model: .instance(model), inputGuardrails: [guardrail])

        await expectErrorAsync(TestError.self, containing: "model boom") {
            _ = try await withTestTimeout(5) { try await Runner.run(agent, input: "trigger guardrail") }
        }

        #expect(guardrailStarted.isSet)
        #expect(guardrailCancelled.isSet)
        #expect(!guardrailFinished.isSet)
    }

    @Test func testParallelGuardrailNonTripwireErrorNotSwallowed() async throws {
        let modelStarted = AsyncEvent()
        let modelCancelled = AsyncEvent()
        let modelFinished = AsyncEvent()
        let guardrail = inputGuardrail(runInParallel: true) { _, _, _ in
            try await modelStarted.wait(timeout: 1)
            throw TestError("guardrail boom")
        }
        let base = try ScriptedModel()
        try base.enqueue(.output([getTextMessage("should_not_finish")]))
        let model = HookedModel(base: base, getResponse: { request, base in
            modelStarted.set()
            defer { modelFinished.set() }
            do {
                try await waitForever()
                return try await base.getResponse(request)
            } catch is CancellationError {
                modelCancelled.set()
                throw CancellationError()
            }
        })
        let agent = Agent(name: "guardrail_error_agent", model: .instance(model), inputGuardrails: [guardrail])

        await expectErrorAsync(TestError.self, containing: "guardrail boom") {
            _ = try await withTestTimeout(5) { try await Runner.run(agent, input: "trigger guardrail") }
        }

        try await modelFinished.wait(timeout: 1)
        #expect(modelStarted.isSet)
        #expect(modelCancelled.isSet)
    }

    @Test func testParallelGuardrailErrorCancelsStreamingModel() async throws {
        let modelStarted = AsyncEvent()
        let modelCancelled = AsyncEvent()
        let modelFinished = AsyncEvent()
        let guardrail = inputGuardrail(runInParallel: true) { _, _, _ in
            try await modelStarted.wait(timeout: 1)
            throw TestError("guardrail boom")
        }
        let model = HookedModel(base: try ScriptedModel(), stream: { _, _ in
            makeStream { _ in
                modelStarted.set()
                defer { modelFinished.set() }
                do {
                    try await waitForever()
                } catch is CancellationError {
                    modelCancelled.set()
                    throw CancellationError()
                }
            }
        })
        let agent = Agent(name: "streaming_guardrail_error_agent", model: .instance(model), inputGuardrails: [guardrail])

        let result = try Runner.runStreamed(agent, input: "trigger guardrail")
        await expectErrorAsync(TestError.self, containing: "guardrail boom") {
            try await withTestTimeout(1) { try await drain(result) }
        }

        try await modelFinished.wait(timeout: 1)
        #expect(modelStarted.isSet)
        #expect(modelCancelled.isSet)
    }

    @Test func testParallelGuardrailTripBeforeToolExecutionStopsStreamingTurn() async throws {
        let toolExecuted = AsyncEvent()
        let modelStarted = AsyncEvent()
        let guardrailTripped = AsyncEvent()
        let dangerousTool = try FunctionTool.create(name: "dangerous_tool") { _ in
            toolExecuted.set()
            return "tool_executed"
        }
        let guardrail = inputGuardrail(runInParallel: true) { _, _, _ in
            try await modelStarted.wait(timeout: 1)
            guardrailTripped.set()
            return GuardrailFunctionOutput(outputInfo: "parallel_trip_before_tool_execution", tripwireTriggered: true)
        }
        let base = try ScriptedModel()
        try base.enqueue(.output([getFunctionToolCall("dangerous_tool", arguments: "{}")]))
        try base.enqueue(.output([getTextMessage("done")]))
        let delay = shortDelay
        let model = HookedModel(base: base, stream: { request, base in
            makeStream { continuation in
                modelStarted.set()
                try await guardrailTripped.wait(timeout: 1)
                try await sleep(seconds: delay)
                for try await event in base.streamResponse(request) { continuation.yield(event) }
            }
        })
        let agent = Agent(
            name: "streaming_guardrail_hardening_agent",
            tools: [dangerousTool],
            instructions: "Call the dangerous_tool immediately",
            model: .instance(model),
            inputGuardrails: [guardrail]
        )

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            try await drain(try Runner.runStreamed(agent, input: "trigger guardrail"))
        }

        #expect(modelStarted.isSet)
        #expect(guardrailTripped.isSet)
        #expect(!toolExecuted.isSet)
    }

    @Test func testParallelGuardrailTripWithSlowCancelSiblingStopsStreamingTurn() async throws {
        let toolExecuted = AsyncEvent()
        let modelStarted = AsyncEvent()
        let guardrailTripped = AsyncEvent()
        let slowCancelStarted = AsyncEvent()
        let slowCancelFinished = AsyncEvent()
        let dangerousTool = try FunctionTool.create(name: "dangerous_tool") { _ in
            toolExecuted.set()
            return "tool_executed"
        }
        let tripwire = inputGuardrail(runInParallel: true) { _, _, _ in
            try await modelStarted.wait(timeout: 1)
            guardrailTripped.set()
            return GuardrailFunctionOutput(outputInfo: "parallel_trip_before_tool_execution_with_slow_cancel", tripwireTriggered: true)
        }
        let delay = shortDelay
        let slowToCancel = inputGuardrail(runInParallel: true) { _, _, _ in
            do {
                try await waitForever()
                return GuardrailFunctionOutput(outputInfo: "slow_to_cancel_guardrail_completed", tripwireTriggered: false)
            } catch is CancellationError {
                slowCancelStarted.set()
                // Swift: sleeping in a cancelled task returns immediately, so spin briefly instead.
                let deadline = Date().addingTimeInterval(delay)
                while Date() < deadline { await Task.yield() }
                slowCancelFinished.set()
                throw CancellationError()
            }
        }
        let base = try ScriptedModel()
        try base.enqueue(.output([getFunctionToolCall("dangerous_tool", arguments: "{}")]))
        try base.enqueue(.output([getTextMessage("done")]))
        let model = HookedModel(base: base, stream: { request, base in
            makeStream { continuation in
                modelStarted.set()
                try await guardrailTripped.wait(timeout: 1)
                try await slowCancelStarted.wait(timeout: 1)
                for try await event in base.streamResponse(request) { continuation.yield(event) }
            }
        })
        let agent = Agent(
            name: "streaming_guardrail_slow_cancel_agent",
            tools: [dangerousTool],
            instructions: "Call the dangerous_tool immediately",
            model: .instance(model),
            inputGuardrails: [tripwire, slowToCancel]
        )

        let error = await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            try await drain(try Runner.runStreamed(agent, input: "trigger guardrail"))
        }

        let runData = try #require(error?.runData)
        #expect(runData.inputGuardrailResults.map { $0.output.outputInfo as? String } == ["parallel_trip_before_tool_execution_with_slow_cancel"])
        #expect(modelStarted.isSet)
        #expect(guardrailTripped.isSet)
        #expect(slowCancelStarted.isSet)
        #expect(slowCancelFinished.isSet)
        #expect(!toolExecuted.isSet)
    }

    @Test(arguments: [false, true])
    func testBlockingGuardrailPreventsToolExecution(streamed: Bool) async throws {
        let toolExecuted = AsyncEvent()
        let guardrailExecuted = AsyncEvent()
        let dangerousTool = try FunctionTool.create(name: "dangerous_tool") { _ in
            toolExecuted.set()
            return "tool_executed"
        }
        let delay = mediumDelay
        let guardrail = inputGuardrail(runInParallel: false) { _, _, _ in
            try await sleep(seconds: delay)
            guardrailExecuted.set()
            return GuardrailFunctionOutput(outputInfo: "blocked_dangerous_input", tripwireTriggered: true)
        }
        let model = try ScriptedModel()
        let agent = Agent(
            name: "agent_with_tools",
            tools: [dangerousTool],
            instructions: "Call the dangerous_tool immediately",
            model: .instance(model),
            inputGuardrails: [guardrail]
        )
        try model.enqueue(.output([getFunctionToolCall("dangerous_tool", arguments: "{}")]))

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            if streamed {
                try await drain(try Runner.runStreamed(agent, input: "trigger guardrail"))
            } else {
                _ = try await Runner.run(agent, input: "trigger guardrail")
            }
        }
        #expect(guardrailExecuted.isSet)
        #expect(!toolExecuted.isSet)
        #expect(model.calls.isEmpty, "Model should not have been called")
    }

    @Test(arguments: [false, true], [false, true])
    func testGuardrailPassesAgentContinues(streamed: Bool, runInParallel: Bool) async throws {
        let executed = AsyncEvent()
        let delay = shortDelay
        let guardrail = inputGuardrail(runInParallel: runInParallel) { _, _, _ in
            try await sleep(seconds: delay)
            executed.set()
            return GuardrailFunctionOutput(outputInfo: "passed", tripwireTriggered: false)
        }
        let model = try ScriptedModel()
        let agent = Agent(name: "test_agent", instructions: "Reply with 'hello'", model: .instance(model), inputGuardrails: [guardrail])
        try model.enqueue(.output([getTextMessage("hello")]))

        if streamed {
            let result = try Runner.runStreamed(agent, input: "test input")
            try await drain(result)
            #expect(result.finalOutput != nil)
        } else {
            let result = try await Runner.run(agent, input: "test input")
            #expect(result.finalOutput != nil)
            #expect(result.inputGuardrailResults.count == 1)
            #expect(!result.inputGuardrailResults[0].output.tripwireTriggered)
        }
        #expect(executed.isSet)
        #expect(!model.calls.isEmpty)
    }

    @Test(arguments: [false, true])
    func testMixedBlockingAndParallelGuardrails(streamed: Bool) async throws {
        let blockingFinished = AsyncEvent()
        let parallelStarted = AsyncEvent()
        let modelCalled = AsyncEvent()
        let parallelFinished = AsyncEvent()
        let observed = SharedBox<[String: Bool]>([:])
        let delay = mediumDelay
        let blocking = inputGuardrail(runInParallel: false) { _, _, _ in
            try await sleep(seconds: delay)
            blockingFinished.set()
            return GuardrailFunctionOutput(outputInfo: "blocking_passed", tripwireTriggered: false)
        }
        let parallel = inputGuardrail(runInParallel: true) { _, _, _ in
            observed.mutate { $0["blocking_finished_at_parallel_start"] = blockingFinished.isSet }
            parallelStarted.set()
            try await modelCalled.wait(timeout: 5)
            parallelFinished.set()
            return GuardrailFunctionOutput(outputInfo: "parallel_passed", tripwireTriggered: false)
        }
        let base = try ScriptedModel()
        try base.enqueue(.output([getTextMessage("hello")]))
        let track: @Sendable () async throws -> Void = {
            observed.mutate { $0["blocking_finished_at_model_call"] = blockingFinished.isSet }
            try await parallelStarted.wait(timeout: 5)
            observed.mutate { $0["parallel_finished_at_model_call"] = parallelFinished.isSet }
            modelCalled.set()
        }
        let model = HookedModel(
            base: base,
            getResponse: { request, base in
                try await track()
                return try await base.getResponse(request)
            },
            stream: { request, base in
                makeStream { continuation in
                    try await track()
                    for try await event in base.streamResponse(request) { continuation.yield(event) }
                }
            }
        )
        let agent = Agent(name: "mixed_agent", instructions: "Reply with 'hello'", model: .instance(model), inputGuardrails: [blocking, parallel])

        if streamed {
            let result = try Runner.runStreamed(agent, input: "test input")
            try await drain(result)
            #expect(result.finalOutput != nil)
            #expect(result.inputGuardrailResults.count == 2)
        } else {
            let result = try await Runner.run(agent, input: "test input")
            #expect(result.finalOutput != nil)
            #expect(result.inputGuardrailResults.count == 2)
        }
        #expect(observed.value["blocking_finished_at_parallel_start"] == true, "Blocking must complete before parallel starts")
        #expect(observed.value["blocking_finished_at_model_call"] == true, "Blocking must complete before model is called")
        #expect(observed.value["parallel_finished_at_model_call"] == false, "Model called while parallel guardrail still running")
        #expect(parallelFinished.isSet, "Parallel guardrail should have completed")
    }

    @Test(arguments: [false, true])
    func testMultipleBlockingGuardrailsOneTriggers(streamed: Bool) async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("hello")]))
        let agent = Agent(
            name: "test_agent",
            model: .instance(model),
            inputGuardrails: [
                inputGuardrail(name: "passes", runInParallel: false) { _, _, _ in
                    GuardrailFunctionOutput(outputInfo: "passed", tripwireTriggered: false)
                },
                inputGuardrail(name: "trips", runInParallel: false) { _, _, _ in
                    GuardrailFunctionOutput(outputInfo: "triggered", tripwireTriggered: true)
                },
            ]
        )

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            if streamed {
                try await drain(try Runner.runStreamed(agent, input: "test input"))
            } else {
                _ = try await Runner.run(agent, input: "test input")
            }
        }
        #expect(model.calls.isEmpty)
    }

    @Test func testGuardrailViaAgentAndRunConfigEquivalent() async throws {
        let agentLevel = SharedBox(false)
        let configLevel = SharedBox(false)
        let agentCheck = inputGuardrail(runInParallel: false) { _, _, _ in
            agentLevel.mutate { $0 = true }
            return GuardrailFunctionOutput(outputInfo: "agent_level_passed", tripwireTriggered: false)
        }
        let configCheck = inputGuardrail(runInParallel: false) { _, _, _ in
            configLevel.mutate { $0 = true }
            return GuardrailFunctionOutput(outputInfo: "config_level_passed", tripwireTriggered: false)
        }
        let model1 = try ScriptedModel()
        try model1.enqueue(.output([getTextMessage("hello")]))
        let agentWithGuardrail = Agent(name: "test_agent", instructions: "Reply with 'hello'", model: .instance(model1), inputGuardrails: [agentCheck])
        let model2 = try ScriptedModel()
        try model2.enqueue(.output([getTextMessage("hello")]))
        let agentWithoutGuardrail = Agent(name: "test_agent", instructions: "Reply with 'hello'", model: .instance(model2))

        let result1 = try await Runner.run(agentWithGuardrail, input: "test input")
        let result2 = try await Runner.run(agentWithoutGuardrail, input: "test input", runConfig: RunConfig(inputGuardrails: [configCheck]))

        #expect(agentLevel.value)
        #expect(configLevel.value)
        #expect(result1.inputGuardrailResults.first?.output.outputInfo as? String == "agent_level_passed")
        #expect(result2.inputGuardrailResults.first?.output.outputInfo as? String == "config_level_passed")
        #expect(result1.inputGuardrailResults.count == 1)
        #expect(result2.inputGuardrailResults.count == 1)
        #expect(!model1.calls.isEmpty)
        #expect(!model2.calls.isEmpty)
    }

    @Test(arguments: [false, true])
    func testBlockingGuardrailCancelsRemainingOnTrigger(streamed: Bool) async throws {
        let fastExecuted = AsyncEvent()
        let slowExecuted = AsyncEvent()
        let slowCancelled = AsyncEvent()
        let slowStarted = AsyncEvent()
        let fast = inputGuardrail(runInParallel: false) { _, _, _ in
            try await slowStarted.wait(timeout: 5)
            fastExecuted.set()
            return GuardrailFunctionOutput(outputInfo: "fast_triggered", tripwireTriggered: true)
        }
        let slow = inputGuardrail(runInParallel: false) { _, _, _ in
            slowStarted.set()
            do {
                try await waitForever()
                slowExecuted.set()
                return GuardrailFunctionOutput(outputInfo: "slow_completed", tripwireTriggered: false)
            } catch is CancellationError {
                slowCancelled.set()
                throw CancellationError()
            }
        }
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("hello")]))
        let agent = Agent(name: "test_agent", instructions: "Reply with 'hello'", model: .instance(model), inputGuardrails: [fast, slow])

        await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            if streamed {
                try await withTestTimeout(5) { try await drain(try Runner.runStreamed(agent, input: "test input")) }
            } else {
                _ = try await withTestTimeout(5) { try await Runner.run(agent, input: "test input") }
            }
        }
        #expect(fastExecuted.isSet, "Fast guardrail should have executed")
        #expect(slowCancelled.isSet, "Slow guardrail should have been cancelled")
        #expect(!slowExecuted.isSet, "Slow guardrail should NOT have completed execution")
        #expect(model.calls.isEmpty, "Model should not have been called when guardrail triggered")
    }

    // MARK: - Sibling cancellation

    @Test func testInputGuardrailRaiseCancelsSiblings() async throws {
        let siblingStarted = AsyncEvent()
        let siblingCompleted = AsyncEvent()
        let siblingCancelled = AsyncEvent()
        let slow = InputGuardrail { _, _, _ in
            siblingStarted.set()
            do {
                try await sleep(seconds: 0.5)
            } catch is CancellationError {
                siblingCancelled.set()
                throw CancellationError()
            }
            siblingCompleted.set()
            return GuardrailFunctionOutput(tripwireTriggered: false)
        }
        let raising = InputGuardrail { _, _, _ in
            try await siblingStarted.wait()
            throw TestError("boom")
        }

        await expectErrorAsync(TestError.self, containing: "boom") {
            _ = try await runInputGuardrails(Agent(name: "t"), [slow, raising], "x", RunContextWrapper())
        }
        #expect(siblingCancelled.isSet, "Sibling task should have been cancelled")
        #expect(!siblingCompleted.isSet, "Sibling task should not have completed")
    }

    @Test func testOutputGuardrailRaiseCancelsSiblings() async throws {
        let siblingStarted = AsyncEvent()
        let siblingCompleted = AsyncEvent()
        let siblingCancelled = AsyncEvent()
        let slow = OutputGuardrail { _, _, _ in
            siblingStarted.set()
            do {
                try await sleep(seconds: 0.5)
            } catch is CancellationError {
                siblingCancelled.set()
                throw CancellationError()
            }
            siblingCompleted.set()
            return GuardrailFunctionOutput(tripwireTriggered: false)
        }
        let raising = OutputGuardrail { _, _, _ in
            try await siblingStarted.wait()
            throw TestError("boom")
        }

        await expectErrorAsync(TestError.self, containing: "boom") {
            _ = try await runOutputGuardrails([slow, raising], Agent(name: "t"), "out", RunContextWrapper())
        }
        #expect(siblingCancelled.isSet, "Sibling task should have been cancelled")
        #expect(!siblingCompleted.isSet, "Sibling task should not have completed")
    }

    // MARK: - Result reporting

    private func orderedInputGuardrails(secondTriggers: Bool, secondRaises: Bool = false, runInParallel: Bool = false) -> [InputGuardrail] {
        let firstDone = AsyncEvent()
        return [
            InputGuardrail(name: "passes", runInParallel: runInParallel) { _, _, _ in
                firstDone.set()
                return GuardrailFunctionOutput(outputInfo: "passes", tripwireTriggered: false)
            },
            InputGuardrail(name: secondRaises ? "raises" : "trips", runInParallel: runInParallel) { _, _, _ in
                try await firstDone.wait()
                // Swift: asyncio resumes this waiter only after the first guardrail finished;
                // threads can finish both at once, so give the first result time to land.
                try await sleep(seconds: 0.01)
                if secondRaises { throw TestError("guardrail exploded") }
                return GuardrailFunctionOutput(outputInfo: "second", tripwireTriggered: secondTriggers)
            },
        ]
    }

    private func orderedOutputGuardrails(secondTriggers: Bool, secondRaises: Bool = false) -> [OutputGuardrail] {
        let firstDone = AsyncEvent()
        return [
            OutputGuardrail(name: "passes") { _, _, _ in
                firstDone.set()
                return GuardrailFunctionOutput(outputInfo: "passes", tripwireTriggered: false)
            },
            OutputGuardrail(name: secondRaises ? "raises" : "trips") { _, _, _ in
                try await firstDone.wait()
                // Swift: asyncio resumes this waiter only after the first guardrail finished;
                // threads can finish both at once, so give the first result time to land.
                try await sleep(seconds: 0.01)
                if secondRaises { throw TestError("guardrail exploded") }
                return GuardrailFunctionOutput(outputInfo: "second", tripwireTriggered: secondTriggers)
            },
        ]
    }

    @Test(arguments: [false, true])
    func testInputGuardrailTripwireReportsResults(runInParallel: Bool) async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("hello")]))
        let agent = Agent(name: "guardrail_results_agent", model: .instance(model), inputGuardrails: orderedInputGuardrails(secondTriggers: true, runInParallel: runInParallel))

        let error = await expectErrorAsync(InputGuardrailTripwireTriggered.self) {
            _ = try await Runner.run(agent, input: "test input")
        }

        let runData = try #require(error?.runData)
        #expect(names(runData.inputGuardrailResults) == ["passes", "trips"])
        #expect(error?.guardrailResult.guardrail.getName() == "trips")
    }

    @Test(arguments: [false, true])
    func testInputGuardrailTripwireReportsResultsStreamed(runInParallel: Bool) async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("hello")]))
        let agent = Agent(name: "guardrail_results_agent", model: .instance(model), inputGuardrails: orderedInputGuardrails(secondTriggers: true, runInParallel: runInParallel))

        let result = try Runner.runStreamed(agent, input: "test input")
        let error = await expectErrorAsync(InputGuardrailTripwireTriggered.self) { try await drain(result) }

        let runData = try #require(error?.runData)
        #expect(names(runData.inputGuardrailResults) == ["passes", "trips"])
        #expect(names(result.inputGuardrailResults) == ["passes", "trips"])
    }

    @Test func testInputGuardrailResultsReportedOnSuccess() async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("hello")]))
        let agent = Agent(
            name: "guardrail_results_agent",
            model: .instance(model),
            inputGuardrails: [
                InputGuardrail(name: "blocking", runInParallel: false, guardrailFunction: syncGuardrail(triggers: false)),
                InputGuardrail(name: "parallel", runInParallel: true, guardrailFunction: syncGuardrail(triggers: false)),
            ]
        )

        let result = try await Runner.run(agent, input: "test input")

        #expect(names(result.inputGuardrailResults) == ["blocking", "parallel"])
    }

    @Test func testInputGuardrailExceptionReportsCompletedResults() async throws {
        let collected = SharedBox<[InputGuardrailResult]>([])
        await expectErrorAsync(TestError.self, containing: "guardrail exploded") {
            _ = try await runInputGuardrails(
                Agent(name: "t"),
                orderedInputGuardrails(secondTriggers: false, secondRaises: true),
                "test input",
                RunContextWrapper(),
                sink: collected
            )
        }
        #expect(names(collected.value) == ["passes"])
    }

    @Test(arguments: [false, true])
    func testInputGuardrailExceptionReportsCompletedResultsStreamed(runInParallel: Bool) async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("hello")]))
        let agent = Agent(
            name: "guardrail_results_agent",
            model: .instance(model),
            inputGuardrails: orderedInputGuardrails(secondTriggers: false, secondRaises: true, runInParallel: runInParallel)
        )

        let result = try Runner.runStreamed(agent, input: "test input")
        await expectErrorAsync(TestError.self, containing: "guardrail exploded") { try await drain(result) }

        #expect(names(result.inputGuardrailResults) == ["passes"])
    }

    @Test(arguments: [false, true])
    func testOutputGuardrailTripwireReportsResults(streamed: Bool) async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("hello")]))
        let agent = Agent(name: "output_guardrail_results_agent", model: .instance(model), outputGuardrails: orderedOutputGuardrails(secondTriggers: true))

        var streamedResult: RunResultStreaming?
        let error = await expectErrorAsync(OutputGuardrailTripwireTriggered.self) {
            if streamed {
                let result = try Runner.runStreamed(agent, input: "test input")
                streamedResult = result
                try await drain(result)
            } else {
                _ = try await Runner.run(agent, input: "test input")
            }
        }

        let runData = try #require(error?.runData)
        #expect(names(runData.outputGuardrailResults) == ["passes", "trips"])
        #expect(error?.guardrailResult.guardrail.getName() == "trips")
        if let streamedResult {
            #expect(names(streamedResult.outputGuardrailResults) == ["passes", "trips"])
        }
    }

    @Test func testOutputGuardrailResultsReportedOnSuccess() async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("hello")]))
        let agent = Agent(name: "output_guardrail_results_agent", model: .instance(model), outputGuardrails: orderedOutputGuardrails(secondTriggers: false))

        let result = try await Runner.run(agent, input: "test input")

        #expect(names(result.outputGuardrailResults) == ["passes", "trips"])
    }

    @Test func testOutputGuardrailExceptionReportsCompletedResults() async throws {
        let collected = SharedBox<[OutputGuardrailResult]>([])
        await expectErrorAsync(TestError.self, containing: "guardrail exploded") {
            _ = try await runOutputGuardrails(
                orderedOutputGuardrails(secondTriggers: false, secondRaises: true),
                Agent(name: "t"),
                "out",
                RunContextWrapper(),
                sink: collected
            )
        }
        #expect(names(collected.value) == ["passes"])
    }

    @Test func testOutputGuardrailExceptionReportsCompletedResultsStreamed() async throws {
        let model = try ScriptedModel()
        try model.enqueue(.output([getTextMessage("hello")]))
        let agent = Agent(
            name: "output_guardrail_results_agent",
            model: .instance(model),
            outputGuardrails: orderedOutputGuardrails(secondTriggers: false, secondRaises: true)
        )

        let result = try Runner.runStreamed(agent, input: "test input")
        await expectErrorAsync(TestError.self, containing: "guardrail exploded") { try await drain(result) }

        #expect(names(result.outputGuardrailResults) == ["passes"])
    }
}
