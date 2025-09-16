import Foundation

/// Represents the instructions backing an agent.
public enum AgentInstructions<Context>: Sendable {
    case literal(String)
    case dynamic(@Sendable (_ context: RunContext<Context>, _ agent: Agent<Context>) async throws -> String)
}

/// Represents an AI agent capable of interacting with tools, handling conversations,
/// and producing outputs based on instructions.
public final class Agent<Context> {
    /// Describes how tool usage should influence the run loop.
    public enum ToolUseBehavior: Sendable {
        case runLLMAgain
        case stopOnFirstTool
        case stopAtTools(Set<String>)
        case custom(@Sendable (_ context: RunContext<Context>, _ toolResults: [ToolCallResult]) async throws -> ToolsToFinalOutputResult)
    }
    
    /// Wraps the outcome of processing tool calls.
    public struct ToolsToFinalOutputResult: Sendable {
        public let isFinalOutput: Bool
        public let finalOutput: String?
        
        public init(isFinalOutput: Bool, finalOutput: String?) {
            self.isFinalOutput = isFinalOutput
            self.finalOutput = finalOutput
        }
    }
    
    /// Captures the result of a tool invocation for decision making.
    public struct ToolCallResult: Sendable {
        public let id: String
        public let name: String
        public let output: String
        
        public init(id: String, name: String, output: String) {
            self.id = id
            self.name = name
            self.output = output
        }
    }
    
    /// The name of the agent, used for identification
    public let name: AgentName
    
    /// Optional description used when referenced from handoffs or other agents.
    public var handoffDescription: String?
    
    /// Instructions that guide the agent's behavior.
    public var instructions: AgentInstructions<Context>?
    
    /// Tools available to the agent
    public private(set) var tools: [Tool<Context>]
    
    /// Guardrails that enforce constraints on agent input
    public private(set) var inputGuardrails: [AnyInputGuardrail<Context>]
    
    /// Guardrails that enforce constraints on agent output
    public private(set) var outputGuardrails: [AnyOutputGuardrail<Context>]
    
    /// Handoffs for delegating work to other agents
    public private(set) var handoffs: [Handoff<Context>]
    
    /// Settings for the model used by this agent
    public var modelSettings: ModelSettings
    
    /// Determines how tool use is handled for this agent.
    public var toolUseBehavior: ToolUseBehavior
    
    /// Whether to reset the tool choice back to default after a call.
    public var resetToolChoice: Bool
    
    /// Creates a new agent with the specified configuration
    /// - Parameters:
    ///   - name: The name of the agent
    ///   - instructions: Instructions for guiding agent behavior
    ///   - handoffDescription: Optional description used for handoffs
    ///   - tools: Optional array of tools available to the agent
    ///   - inputGuardrails: Optional array of input guardrails for the agent
    ///   - outputGuardrails: Optional array of output guardrails for the agent
    ///   - handoffs: Optional array of handoffs for the agent
    ///   - modelSettings: Optional model settings for the agent
    ///   - toolUseBehavior: Strategy for handling tool calls
    ///   - resetToolChoice: Whether tool choice should reset after invocation
    public init(
        name: AgentName,
        instructions: String,
        handoffDescription: String? = nil,
        tools: [Tool<Context>] = [],
        inputGuardrails: [AnyInputGuardrail<Context>] = [],
        outputGuardrails: [AnyOutputGuardrail<Context>] = [],
        handoffs: [Handoff<Context>] = [],
        modelSettings: ModelSettings = ModelSettings(),
        toolUseBehavior: ToolUseBehavior = .runLLMAgain,
        resetToolChoice: Bool = true
    ) {
        self.name = name
        self.instructions = .literal(instructions)
        self.handoffDescription = handoffDescription
        self.tools = tools
        self.inputGuardrails = inputGuardrails
        self.outputGuardrails = outputGuardrails
        self.handoffs = handoffs
        self.modelSettings = modelSettings
        self.toolUseBehavior = toolUseBehavior
        self.resetToolChoice = resetToolChoice
    }
    
    /// Alternate initializer for dynamic instructions while keeping other defaults.
    public init(
        name: AgentName,
        instructions: AgentInstructions<Context>?,
        handoffDescription: String? = nil,
        tools: [Tool<Context>] = [],
        inputGuardrails: [AnyInputGuardrail<Context>] = [],
        outputGuardrails: [AnyOutputGuardrail<Context>] = [],
        handoffs: [Handoff<Context>] = [],
        modelSettings: ModelSettings = ModelSettings(),
        toolUseBehavior: ToolUseBehavior = .runLLMAgain,
        resetToolChoice: Bool = true
    ) {
        self.name = name
        self.instructions = instructions
        self.handoffDescription = handoffDescription
        self.tools = tools
        self.inputGuardrails = inputGuardrails
        self.outputGuardrails = outputGuardrails
        self.handoffs = handoffs
        self.modelSettings = modelSettings
        self.toolUseBehavior = toolUseBehavior
        self.resetToolChoice = resetToolChoice
    }
    
    /// Adds a tool to the agent
    /// - Parameter tool: The tool to add
    /// - Returns: Self for method chaining
    @discardableResult
    public func addTool(_ tool: Tool<Context>) -> Self {
        tools.append(tool)
        return self
    }
    
    /// Adds multiple tools to the agent
    /// - Parameter tools: The tools to add
    /// - Returns: Self for method chaining
    @discardableResult
    public func addTools(_ tools: [Tool<Context>]) -> Self {
        self.tools.append(contentsOf: tools)
        return self
    }
    
    /// Adds an input guardrail to the agent.
    @discardableResult
    public func addInputGuardrail<G: InputGuardrail>(_ guardrail: G) -> Self where G.Context == Context {
        inputGuardrails.append(AnyInputGuardrail(guardrail))
        return self
    }
    
    /// Adds an output guardrail to the agent.
    @discardableResult
    public func addOutputGuardrail<G: OutputGuardrail>(_ guardrail: G) -> Self where G.Context == Context {
        outputGuardrails.append(AnyOutputGuardrail(guardrail))
        return self
    }
    
    /// Adds a handoff to the agent
    /// - Parameter handoff: The handoff to add
    /// - Returns: Self for method chaining
    @discardableResult
    public func addHandoff(_ handoff: Handoff<Context>) -> Self {
        handoffs.append(handoff)
        return self
    }
    
    /// Creates a copy of this agent
    /// - Returns: A new agent with the same configuration
    public func clone(
        name: AgentName? = nil,
        instructions: AgentInstructions<Context>? = nil,
        handoffDescription: String? = nil,
        tools: [Tool<Context>]? = nil,
        inputGuardrails: [AnyInputGuardrail<Context>]? = nil,
        outputGuardrails: [AnyOutputGuardrail<Context>]? = nil,
        handoffs: [Handoff<Context>]? = nil,
        modelSettings: ModelSettings? = nil,
        toolUseBehavior: ToolUseBehavior? = nil,
        resetToolChoice: Bool? = nil
    ) -> Agent<Context> {
        Agent(
            name: name ?? self.name,
            instructions: instructions ?? self.instructions,
            handoffDescription: handoffDescription ?? self.handoffDescription,
            tools: tools ?? self.tools,
            inputGuardrails: inputGuardrails ?? self.inputGuardrails,
            outputGuardrails: outputGuardrails ?? self.outputGuardrails,
            handoffs: handoffs ?? self.handoffs,
            modelSettings: modelSettings ?? self.modelSettings,
            toolUseBehavior: toolUseBehavior ?? self.toolUseBehavior,
            resetToolChoice: resetToolChoice ?? self.resetToolChoice
        )
    }
    
    /// Resolves the active instructions based on the run context.
    /// - Parameter runContext: The context of the current run.
    /// - Returns: The resolved instructions, if any.
    public func resolveInstructions(runContext: RunContext<Context>) async throws -> String? {
        switch instructions {
        case .literal(let value):
            return value
        case .dynamic(let closure):
            return try await closure(runContext, self)
        case .none:
            return nil
        }
    }
    
    /// Returns the tools enabled for the provided run context.
    /// - Parameter runContext: The current run context.
    /// - Returns: Enabled tools ready for invocation.
    public func enabledTools(for runContext: RunContext<Context>) async -> [Tool<Context>] {
        await withTaskGroup(of: (Int, Tool<Context>)?.self) { group in
            for (index, tool) in tools.enumerated() {
                group.addTask {
                    if await tool.isEnabled(for: runContext) {
                        return (index, tool)
                    }
                    return nil
                }
            }
            var enabled: [(Int, Tool<Context>)] = []
            for await result in group {
                if let value = result {
                    enabled.append(value)
                }
            }
            return enabled.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
    }
}
