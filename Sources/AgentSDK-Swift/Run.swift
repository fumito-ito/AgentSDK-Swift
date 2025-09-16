import Foundation

/// Represents a single run of an agent
public final class Run<Context> {
    /// The agent being run
    public let agent: Agent<Context>
    
    /// The input for the run
    public let input: String
    
    /// The run context wrapper containing the caller context and usage information
    public let runContext: RunContext<Context>
    
    /// The history of messages for the run
    public private(set) var messages: [Message] = []
    
    /// The current state of the run
    public private(set) var state: State = .notStarted
    
    /// The model used for the run
    private let model: ModelInterface
    
    /// Maximum number of turns before giving up. Mirrors Python default.
    private let maxTurns: Int
    
    /// Creates a new run
    /// - Parameters:
    ///   - agent: The agent to run
    ///   - input: The input for the run
    ///   - context: The context for the run
    ///   - model: The model to use for the run
    ///   - maxTurns: Maximum number of iterations allowed in the run loop
    public init(
        agent: Agent<Context>,
        input: String,
        context: Context,
        model: ModelInterface,
        maxTurns: Int = 10
    ) {
        self.agent = agent
        self.input = input
        self.model = model
        self.maxTurns = maxTurns
        self.runContext = RunContext(value: context)
    }
    
    /// Executes the run
    /// - Returns: The result of the run
    /// - Throws: RunError if there is a problem during execution
    public func execute() async throws -> Result {
        guard state == .notStarted else {
            throw RunError.invalidState("Run has already been started")
        }
        
        state = .running
        
        do {
            if let systemInstructions = try await agent.resolveInstructions(runContext: runContext) {
                messages.append(.system(systemInstructions))
            }
            
            var validatedInput = input
            for guardrail in agent.inputGuardrails {
                do {
                    validatedInput = try guardrail.validate(validatedInput, context: runContext.value)
                } catch let error as GuardrailError {
                    state = .failed
                    throw RunError.guardrailError(error)
                }
            }
            
            // Check for handoffs before running the agent
            for handoff in agent.handoffs {
                if handoff.filter.shouldHandoff(input: validatedInput, context: runContext.value) {
                    let handoffRun = Run(
                        agent: handoff.agent,
                        input: validatedInput,
                        context: runContext.value,
                        model: model,
                        maxTurns: maxTurns
                    )
                    let result = try await handoffRun.execute()
                    runContext.mergeUsage(from: handoffRun.runContext)
                    messages = handoffRun.messages
                    state = .completed
                    return result
                }
            }
            
            messages.append(.user(validatedInput))
            
            var currentTurn = 0
            while currentTurn < maxTurns {
                currentTurn += 1
                let enabledTools = await agent.enabledTools(for: runContext)
                let response = try await model.getResponse(
                    messages: messages,
                    settings: agent.modelSettings
                )
                runContext.recordUsage(response.usage)
                messages.append(.assistant(response.content))
                
                if response.toolCalls.isEmpty {
                    var finalOutput = response.content
                    for guardrail in agent.outputGuardrails {
                        do {
                            finalOutput = try guardrail.validate(finalOutput, context: runContext.value)
                        } catch let error as GuardrailError {
                            state = .failed
                            throw RunError.guardrailError(error)
                        }
                    }
                    state = .completed
                    return Result(finalOutput: finalOutput, messages: messages, usage: runContext.usage)
                }
                
                let toolProcessing = try await processToolCalls(
                    response.toolCalls,
                    availableTools: enabledTools
                )
                for message in toolProcessing.messageResults {
                    messages.append(Message(role: .tool, content: .toolResults(message)))
                }
                
                if let finalFromTools = try await resolveToolBehavior(
                    toolProcessing.callResults,
                    behavior: agent.toolUseBehavior
                ) {
                    var finalOutput = finalFromTools
                    for guardrail in agent.outputGuardrails {
                        do {
                            finalOutput = try guardrail.validate(finalOutput, context: runContext.value)
                        } catch let error as GuardrailError {
                            state = .failed
                            throw RunError.guardrailError(error)
                        }
                    }
                    messages.append(.assistant(finalOutput))
                    state = .completed
                    return Result(finalOutput: finalOutput, messages: messages, usage: runContext.usage)
                }
                
                // Continue loop with tool results appended to message history.
            }
            
            state = .failed
            throw RunError.maxTurnsExceeded(maxTurns)
        } catch let error as RunError {
            state = .failed
            throw error
        } catch {
            state = .failed
            throw RunError.executionError(error)
        }
    }
    
    private func processToolCalls(
        _ toolCalls: [ModelResponse.ToolCall],
        availableTools: [Tool<Context>]
    ) async throws -> ToolProcessingOutcome {
        let toolMap = Dictionary(uniqueKeysWithValues: availableTools.map { ($0.name, $0) })
        var messageResults: [MessageContent.ToolResult] = []
        var callResults: [Agent<Context>.ToolCallResult] = []
        
        for toolCall in toolCalls {
            guard let tool = toolMap[toolCall.name] else {
                throw RunError.toolNotFound("Tool \(toolCall.name) not found")
            }
            
            do {
                let result = try await tool.invoke(
                    parameters: toolCall.parameters,
                    runContext: runContext
                )
                let resultString = stringifyToolResult(result)
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
                throw RunError.toolExecutionError(toolName: toolCall.name, error: error)
            }
        }
        
        return ToolProcessingOutcome(
            messageResults: messageResults,
            callResults: callResults
        )
    }
    
    private func stringifyToolResult(_ result: Any) -> String {
        if let stringResult = result as? String {
            return stringResult
        }
        if let data = try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted]),
           let jsonString = String(data: data, encoding: .utf8) {
            return jsonString
        }
        return String(describing: result)
    }
    
    private func resolveToolBehavior(
        _ toolResults: [Agent<Context>.ToolCallResult],
        behavior: Agent<Context>.ToolUseBehavior
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
    
    /// Represents the result of a run
    public struct Result {
        /// The final output from the agent
        public let finalOutput: String
        
        /// The complete message history for the run
        public let messages: [Message]
        
        /// Aggregated usage information for the run
        public let usage: Usage
    }
    
    /// Represents the state of a run
    public enum State {
        case notStarted
        case running
        case completed
        case failed
    }
    
    /// Errors that can occur during a run
    public enum RunError: Error {
        case invalidState(String)
        case maxTurnsExceeded(Int)
        case guardrailError(GuardrailError)
        case toolNotFound(String)
        case toolExecutionError(toolName: String, error: Error)
        case executionError(Error)
    }
    
    private struct ToolProcessingOutcome {
        let messageResults: [MessageContent.ToolResult]
        let callResults: [Agent<Context>.ToolCallResult]
    }
}
