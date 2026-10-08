import Foundation

/// The data passed to a handoff input filter.
public struct HandoffInputData: Sendable {
    /// The input history before `Runner.run()` was called.
    public var inputHistory: RunInput
    /// The items generated before the agent turn where the handoff was invoked.
    public var preHandoffItems: [RunItem]
    /// The new items generated during the current agent turn, including the item that triggered
    /// the handoff and the tool output message representing the response from the handoff output.
    public var newItems: [RunItem]
    /// The run context at the time the handoff was invoked.
    public var runContext: RunContextWrapper?
    /// Items to include in the next agent's input instead of `newItems`, when set. This filters
    /// model input while preserving `newItems` for session history.
    public var inputItems: [RunItem]?

    public init(
        inputHistory: RunInput,
        preHandoffItems: [RunItem],
        newItems: [RunItem],
        runContext: RunContextWrapper? = nil,
        inputItems: [RunItem]? = nil
    ) {
        self.inputHistory = inputHistory
        self.preHandoffItems = preHandoffItems
        self.newItems = newItems
        self.runContext = runContext
        self.inputItems = inputItems
    }

    /// Makes a copy with the given changes, mirroring `HandoffInputData.clone(**kwargs)`.
    public func clone(_ modify: (inout HandoffInputData) -> Void) -> HandoffInputData {
        var copy = self
        modify(&copy)
        return copy
    }
}

/// Filters the input data passed to the next agent.
public typealias HandoffInputFilter = @Sendable (HandoffInputData) async throws -> HandoffInputData

/// Maps the previous transcript to the nested summary payload.
public typealias HandoffHistoryMapper = @Sendable ([TResponseInputItem]) -> [TResponseInputItem]

/// A handoff is when an agent delegates a task to another agent.
///
/// For example, in a customer support scenario a "triage agent" determines which agent should
/// handle the request, and sub-agents specialize in areas like billing or account management.
public final class Handoff: @unchecked Sendable {
    /// The name of the tool that represents the handoff.
    public var toolName: String
    /// The description of the tool that represents the handoff.
    public var toolDescription: String
    /// The JSON schema for the handoff tool-call arguments.
    public var inputJSONSchema: JSONObject
    /// Invokes the handoff with the run context and the arguments from the LLM as a JSON string
    /// (or an empty string when the schema is empty). Must return an agent.
    public var onInvokeHandoff: @Sendable (RunContextWrapper, String) async throws -> AgentBase
    /// The name of the agent that is being handed off to.
    public var agentName: String
    /// Filters the inputs that are passed to the next agent.
    public var inputFilter: HandoffInputFilter?
    /// Overrides the run-level `nestHandoffHistory` behavior for this handoff only.
    public var nestHandoffHistory: Bool?
    /// Whether the input JSON schema is in strict mode.
    public var strictJSONSchema: Bool
    /// Whether the handoff is enabled.
    public var isEnabled: ToolEnablement

    /// The target agent when constructed via `handoff()`.
    weak var agentRef: AgentBase?
    /// The target agent name and derived tool name when the default name was used.
    var defaultToolIdentity: (agentName: String, toolName: String)?

    public init(
        toolName: String,
        toolDescription: String,
        inputJSONSchema: JSONObject,
        onInvokeHandoff: @escaping @Sendable (RunContextWrapper, String) async throws -> AgentBase,
        agentName: String,
        inputFilter: HandoffInputFilter? = nil,
        nestHandoffHistory: Bool? = nil,
        strictJSONSchema: Bool = true,
        isEnabled: ToolEnablement = .always
    ) {
        self.toolName = toolName
        self.toolDescription = toolDescription
        self.inputJSONSchema = inputJSONSchema
        self.onInvokeHandoff = onInvokeHandoff
        self.agentName = agentName
        self.inputFilter = inputFilter
        self.nestHandoffHistory = nestHandoffHistory
        self.strictJSONSchema = strictJSONSchema
        self.isEnabled = isEnabled
    }

    /// The tool output message sent after a handoff.
    public func getTransferMessage(_ agent: AgentBase) -> String {
        JSONValue.object(["assistant": .string(agent.name)]).pythonJSONString()
    }

    public static func defaultToolName(_ agent: AgentBase) -> String {
        transformStringFunctionStyle("transfer_to_\(agent.name)", warnOnWhitespace: false)
    }

    public static func defaultToolDescription(_ agent: AgentBase) -> String {
        "Handoff to the \(agent.name) agent to handle the request. \(agent.handoffDescription ?? "")"
    }
}

/// Creates a handoff to `agent`.
///
/// - Parameters:
///   - toolNameOverride: Optional override for the name of the tool that represents the handoff.
///   - toolDescriptionOverride: Optional override for the tool description.
///   - onHandoff: Runs when the handoff is invoked, for side effects or bookkeeping.
///   - inputFilter: Filters the inputs that are passed to the next agent.
///   - nestHandoffHistory: Optional override for the run-level `nestHandoffHistory` flag.
///   - isEnabled: Whether the handoff is enabled. Disabled handoffs are hidden from the LLM.
public func handoff(
    _ agent: Agent,
    toolNameOverride: String? = nil,
    toolDescriptionOverride: String? = nil,
    onHandoff: (@Sendable (RunContextWrapper) async throws -> Void)? = nil,
    inputFilter: HandoffInputFilter? = nil,
    nestHandoffHistory: Bool? = nil,
    isEnabled: ToolEnablement = .always
) -> Handoff {
    makeHandoff(
        agent,
        toolNameOverride: toolNameOverride,
        toolDescriptionOverride: toolDescriptionOverride,
        inputJSONSchema: [:],
        invoke: { context, _ in try await onHandoff?(context) },
        inputFilter: inputFilter,
        nestHandoffHistory: nestHandoffHistory,
        isEnabled: isEnabled
    )
}

/// Creates a handoff whose tool-call arguments are validated against `inputType` and passed to
/// `onHandoff`. This only affects the handoff tool payload, not the next agent's main input.
public func handoff<Input: Decodable & Sendable>(
    _ agent: Agent,
    inputType: Input.Type,
    toolNameOverride: String? = nil,
    toolDescriptionOverride: String? = nil,
    onHandoff: @escaping @Sendable (RunContextWrapper, Input) async throws -> Void,
    inputFilter: HandoffInputFilter? = nil,
    nestHandoffHistory: Bool? = nil,
    isEnabled: ToolEnablement = .always
) throws -> Handoff {
    let schema = try JSONSchemaGenerator.schema(for: inputType)
    return makeHandoff(
        agent,
        toolNameOverride: toolNameOverride,
        toolDescriptionOverride: toolDescriptionOverride,
        inputJSONSchema: schema,
        invoke: { context, inputJSON in
            guard let inputJSON else {
                attachErrorToCurrentSpan(SpanError(
                    message: "Handoff function expected non-null input, but got None",
                    data: ["details": "input_json is None"]
                ))
                throw ModelBehaviorError("Handoff function expected non-null input, but got None")
            }
            let validated: Input
            do {
                validated = try JSONValue.parse(inputJSON).decode(as: Input.self)
            } catch {
                attachErrorToCurrentSpan(SpanError(message: "Invalid JSON provided", data: [:]))
                if AgentsDebug.dontLogModelData || AgentsDebug.dontLogToolData {
                    throw ModelBehaviorError("Invalid JSON when parsing model output")
                }
                throw ModelBehaviorError("Invalid JSON when parsing \(inputJSON) for \(Input.self); \(error)")
            }
            try await onHandoff(context, validated)
        },
        inputFilter: inputFilter,
        nestHandoffHistory: nestHandoffHistory,
        isEnabled: isEnabled
    )
}

private func makeHandoff(
    _ agent: Agent,
    toolNameOverride: String?,
    toolDescriptionOverride: String?,
    inputJSONSchema: JSONObject,
    invoke: @escaping @Sendable (RunContextWrapper, String?) async throws -> Void,
    inputFilter: HandoffInputFilter?,
    nestHandoffHistory: Bool?,
    isEnabled: ToolEnablement
) -> Handoff {
    let toolName = toolNameOverride ?? Handoff.defaultToolName(agent)
    let toolDescription = toolDescriptionOverride ?? Handoff.defaultToolDescription(agent)
    // Always ensure the input JSON schema is in strict mode.
    let strictSchema = (try? ensureStrictJSONSchema(inputJSONSchema)) ?? inputJSONSchema
    let handoffObject = Handoff(
        toolName: toolName,
        toolDescription: toolDescription,
        inputJSONSchema: strictSchema,
        onInvokeHandoff: { context, inputJSON in
            try await invoke(context, inputJSON.isEmpty && inputJSONSchema.isEmpty ? nil : inputJSON)
            return agent
        },
        agentName: agent.name,
        inputFilter: inputFilter,
        nestHandoffHistory: nestHandoffHistory,
        isEnabled: isEnabled
    )
    handoffObject.agentRef = agent
    if toolNameOverride == nil {
        handoffObject.defaultToolIdentity = (agent.name, toolName)
    }
    return handoffObject
}

/// An entry in `Agent.handoffs`: an agent (converted with `handoff()` at run time) or a
/// configured `Handoff`.
///
/// Swift: upstream accepts `list[Agent | Handoff]`; Swift needs an explicit sum type.
public enum HandoffTarget: Sendable {
    case agent(Agent)
    case handoff(Handoff)

    /// Resolves the entry to a `Handoff`.
    public var resolved: Handoff {
        switch self {
        case .agent(let agent): return Agents.handoff(agent)
        case .handoff(let handoff): return handoff
        }
    }
}
