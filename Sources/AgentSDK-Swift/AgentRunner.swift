import Foundation

/// Static class for running agents
public struct AgentRunner {
    /// Executes an agent using the configured model provider.
    /// - Parameters:
    ///   - agent: The agent to run.
    ///   - input: The user input that starts the conversation.
    ///   - context: Arbitrary state passed through the run.
    /// - Returns: The completed run result containing the final output, messages, and usage.
    /// - Throws: `RunnerError` when model lookup, execution, or guardrail evaluation fails.
    public static func run<Context>(
        agent: Agent<Context>,
        input: String,
        context: Context
    ) async throws -> Run<Context>.Result {
        do {
            let model = try await ModelProvider.shared.getModel(modelName: agent.modelSettings.modelName)
            let run = Run(agent: agent, input: input, context: context, model: model)
            return try await run.execute()
        } catch let error as ModelProvider.ModelProviderError {
            throw RunnerError.modelError(error)
        } catch let error as Run<Context>.RunError {
            throw RunnerError.runError(error)
        } catch {
            throw RunnerError.unknownError(error)
        }
    }

    /// Executes an agent while streaming intermediate output chunks to the supplied handler.
    /// - Parameters:
    ///   - agent: The agent to run.
    ///   - input: The user input that starts the conversation.
    ///   - context: Arbitrary state passed through the run.
    ///   - streamHandler: Callback that receives streamed content chunks.
    /// - Returns: The completed run result containing the final output, messages, and usage.
    /// - Throws: `RunnerError` when model lookup, execution, or guardrail evaluation fails.
    public static func runStreamed<Context>(
        agent: Agent<Context>,
        input: String,
        context: Context,
        streamHandler: @escaping (String) async -> Void
    ) async throws -> Run<Context>.Result {
        do {
            let model = try await ModelProvider.shared.getModel(modelName: agent.modelSettings.modelName)
            let runContext = RunContext(value: context)
            return try await runStreamedInternal(
                agent: agent,
                input: input,
                runContext: runContext,
                model: model,
                settings: agent.modelSettings,
                streamHandler: streamHandler
            )
        } catch let error as ModelProvider.ModelProviderError {
            throw RunnerError.modelError(error)
        } catch let error as Run<Context>.RunError {
            throw RunnerError.runError(error)
        } catch {
            throw RunnerError.unknownError(error)
        }
    }

    private static func runStreamedInternal<Context>(
        agent: Agent<Context>,
        input: String,
        runContext: RunContext<Context>,
        model: ModelInterface,
        settings: ModelSettings,
        streamHandler: @escaping (String) async -> Void,
        turn: Int = 0,
        maxTurns: Int = 10
    ) async throws -> Run<Context>.Result {
        guard turn < maxTurns else {
            throw Run<Context>.RunError.maxTurnsExceeded(maxTurns)
        }

        let systemInstructions = try await agent.resolveInstructions(runContext: runContext)

        var validatedInput = input
        for guardrail in agent.inputGuardrails {
            do {
                validatedInput = try guardrail.validate(validatedInput, context: runContext.value)
            } catch let error as GuardrailError {
                throw Run<Context>.RunError.guardrailError(error)
            }
        }

        for handoff in agent.handoffs {
            if handoff.filter.shouldHandoff(input: validatedInput, context: runContext.value) {
                let result = try await runStreamedInternal(
                    agent: handoff.agent,
                    input: validatedInput,
                    runContext: runContext,
                    model: model,
                    settings: handoff.agent.modelSettings,
                    streamHandler: streamHandler,
                    turn: turn,
                    maxTurns: maxTurns
                )
                return result
            }
        }

        var messages: [Message] = []
        if let systemInstructions {
            messages.append(.system(systemInstructions))
        }
        messages.append(.user(validatedInput))

        var currentTurn = turn
        while currentTurn < maxTurns {
            currentTurn += 1
            let enabledTools = await agent.enabledTools(for: runContext)
            var toolCalls: [ModelResponse.ToolCall] = []
            let response = try await model.getStreamedResponse(
                messages: messages,
                settings: settings
            ) { event in
                switch event {
                case .content(let content):
                    await streamHandler(content)
                case .toolCall(let toolCall):
                    toolCalls.append(toolCall)
                case .end:
                    break
                }
            }
            runContext.recordUsage(response.usage)
            messages.append(.assistant(response.content))

            if response.toolCalls.isEmpty {
                var finalOutput = response.content
                for guardrail in agent.outputGuardrails {
                    do {
                        finalOutput = try guardrail.validate(finalOutput, context: runContext.value)
                    } catch let error as GuardrailError {
                        throw Run<Context>.RunError.guardrailError(error)
                    }
                }
                return Run.Result(
                    finalOutput: finalOutput,
                    messages: messages,
                    usage: runContext.usage
                )
            }

            let processing = try await processToolCalls(
                toolCalls,
                enabledTools: enabledTools,
                runContext: runContext,
                streamHandler: streamHandler
            )
            for toolMessage in processing.messageResults {
                messages.append(Message(role: .tool, content: .toolResults(toolMessage)))
            }

            if let finalFromTools = try await resolveToolBehavior(
                processing.callResults,
                behavior: agent.toolUseBehavior,
                runContext: runContext
            ) {
                var finalOutput = finalFromTools
                for guardrail in agent.outputGuardrails {
                    do {
                        finalOutput = try guardrail.validate(finalOutput, context: runContext.value)
                    } catch let error as GuardrailError {
                        throw Run<Context>.RunError.guardrailError(error)
                    }
                }
                messages.append(.assistant(finalOutput))
                await streamHandler(finalOutput)
                return Run.Result(
                    finalOutput: finalOutput,
                    messages: messages,
                    usage: runContext.usage
                )
            }
        }

        throw Run<Context>.RunError.maxTurnsExceeded(maxTurns)
    }

    private static func processToolCalls<Context>(
        _ toolCalls: [ModelResponse.ToolCall],
        enabledTools: [Tool<Context>],
        runContext: RunContext<Context>,
        streamHandler: @escaping (String) async -> Void
    ) async throws -> ToolProcessingOutcome<Context> {
        let toolMap = Dictionary(uniqueKeysWithValues: enabledTools.map { ($0.name, $0) })
        var messageResults: [MessageContent.ToolResult] = []
        var callResults: [Agent<Context>.ToolCallResult] = []

        for toolCall in toolCalls {
            guard let tool = toolMap[toolCall.name] else {
                throw Run<Context>.RunError.toolNotFound("Tool \(toolCall.name) not found")
            }

            await streamHandler("\nExecuting tool: \(toolCall.name)...\n")
            do {
                let rawResult = try await tool.invoke(
                    parameters: toolCall.parameters,
                    runContext: runContext
                )
                let resultString = stringifyToolResult(rawResult)
                await streamHandler("\nTool result: \(resultString)\n")
                let toolResult = MessageContent.ToolResult(
                    toolCallId: toolCall.id,
                    result: resultString
                )
                messageResults.append(toolResult)
                callResults.append(Agent.ToolCallResult(
                    id: toolCall.id,
                    name: toolCall.name,
                    output: resultString
                ))
            } catch {
                throw Run<Context>.RunError.toolExecutionError(toolName: toolCall.name, error: error)
            }
        }

        return ToolProcessingOutcome(
            messageResults: messageResults,
            callResults: callResults
        )
    }

    private static func stringifyToolResult(_ result: Any) -> String {
        if let stringResult = result as? String {
            return stringResult
        }
        if let data = try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted]),
           let jsonString = String(data: data, encoding: .utf8) {
            return jsonString
        }
        return String(describing: result)
    }

    private static func resolveToolBehavior<Context>(
        _ toolResults: [Agent<Context>.ToolCallResult],
        behavior: Agent<Context>.ToolUseBehavior,
        runContext: RunContext<Context>
    ) async throws -> String? {
        guard !toolResults.isEmpty else { return nil }
        switch behavior {
        case .runLLMAgain:
            return nil
        case .stopOnFirstTool:
            return toolResults.first?.output
        case .stopAtTools(let names):
            if let match = toolResults.first(where: { names.contains($0.name) }) {
                return match.output
            }
            return nil
        case .custom(let handler):
            let decision = try await handler(runContext, toolResults)
            return decision.isFinalOutput ? decision.finalOutput ?? toolResults.last?.output : nil
        }
    }

    public enum RunnerError: Error {
        case modelError(ModelProvider.ModelProviderError)
        case runError(any Error)
        case guardrailError(GuardrailError)
        case toolNotFound(String)
        case toolExecutionError(toolName: String, error: Error)
        case unknownError(Error)
    }

    private struct ToolProcessingOutcome<Context> {
        let messageResults: [MessageContent.ToolResult]
        let callResults: [Agent<Context>.ToolCallResult]
    }
}
