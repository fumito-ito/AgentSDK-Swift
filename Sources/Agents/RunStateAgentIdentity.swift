import Foundation

// Port of src/agents/_run_state_agent_identity.py: stable agent graph identities shared by
// RunState and tool tracking.
//
// Swift: the sandbox capability signatures are out of scope (porting/decisions.md D2), and agent
// signatures are built from the agent configuration Swift can describe (porting/decisions.md D14).

/// Returns the agents reachable from `initialAgent` in breadth-first order.
///
/// Handoff targets and agent-as-tool instances are both followed so that nested approvals can be
/// restored.
func iterAgentGraph(_ initialAgent: Agent) -> [Agent] {
    var ordered: [Agent] = []
    var queue: [Agent] = [initialAgent]
    var seen: Set<ObjectIdentifier> = []
    while !queue.isEmpty {
        let current = queue.removeFirst()
        guard seen.insert(ObjectIdentifier(current)).inserted else { continue }
        ordered.append(current)
        for target in current.handoffs {
            switch target {
            case .agent(let other):
                queue.append(other)
            case .handoff(let handoff):
                if let other = handoff.agentRef as? Agent {
                    queue.append(other)
                } else {
                    logger.debug("Skipping unresolved handoff target while building agent map: \(handoff.agentName)")
                }
            }
        }
        for case let tool as FunctionTool in current.tools {
            if let nested = tool.agentInstance { queue.append(nested) }
        }
    }
    return ordered
}

/// Builds a map of agent names to agents by traversing handoffs and agent tools.
///
/// The first agent reached for each name wins.
func buildAgentMap(_ initialAgent: Agent) -> [String: Agent] {
    var map: [String: Agent] = [:]
    for agent in iterAgentGraph(initialAgent) where map[agent.name] == nil {
        map[agent.name] = agent
    }
    return map
}

private func stableIdentityText(_ value: JSONValue) -> String {
    value.jsonString(sortedKeys: true)
}

private func toolIdentityText(_ tool: any Tool) -> String {
    var signature: JSONObject = [
        "type": .string(String(describing: type(of: tool))),
        "name": .string(tool.name),
    ]
    if let function = tool as? FunctionTool {
        signature["description"] = .string(function.description)
        signature["params_json_schema"] = .object(function.paramsJSONSchema)
        if let namespace = function.toolNamespace { signature["namespace"] = .string(namespace) }
        if let nested = function.agentInstance { signature["agent_tool_target"] = .string(nested.name) }
    }
    return stableIdentityText(.object(signature))
}

private func handoffIdentityText(_ target: HandoffTarget) -> String {
    switch target {
    case .agent(let agent):
        return stableIdentityText(["type": "agent", "agent_name": .string(agent.name)])
    case .handoff(let handoff):
        return stableIdentityText([
            "type": "handoff",
            "tool_name": .string(handoff.toolName),
            "tool_description": .string(handoff.toolDescription),
            "agent_name": .string(handoff.agentName),
            "strict_json_schema": .bool(handoff.strictJSONSchema),
        ])
    }
}

private func toolUseBehaviorIdentity(_ behavior: ToolUseBehavior) -> JSONValue {
    switch behavior {
    case .runLLMAgain: return "run_llm_again"
    case .stopOnFirstTool: return "stop_on_first_tool"
    case .stopAtTools(let names): return ["stop_at_tool_names": .array(names.map(JSONValue.string))]
    case .custom: return "custom"
    }
}

/// Describes an agent's configuration so duplicate-name agents sort independently of traversal.
///
/// Swift: upstream normalizes every agent attribute, including callables by qualified name. Swift
/// closures have no stable name, so dynamic instructions, custom tool-use behaviors, and hooks
/// contribute only their kind.
private func agentIdentitySignature(_ agent: Agent) -> String {
    var signature: JSONObject = [
        "agent_type": .string(String(describing: type(of: agent))),
        "handoff_description": JSONValue(optional: agent.handoffDescription),
        "model_settings": .object(agent.modelSettings.toJSONDict()),
        "tool_use_behavior": toolUseBehaviorIdentity(agent.toolUseBehavior),
        "reset_tool_choice": .bool(agent.resetToolChoice),
        "tools": .array(agent.tools.map(toolIdentityText).sorted().map(JSONValue.string)),
        "handoffs": .array(agent.handoffs.map(handoffIdentityText).sorted().map(JSONValue.string)),
        "input_guardrails": .array(agent.inputGuardrails.map { JSONValue(optional: $0.name) }),
        "output_guardrails": .array(agent.outputGuardrails.map { JSONValue(optional: $0.name) }),
        "mcp_servers": .array(agent.mcpServers.map { .string($0.name) }.sorted { $0.stringValue! < $1.stringValue! }),
    ]
    switch agent.instructions {
    case .text(let text)?: signature["instructions"] = .string(text)
    case .dynamic?: signature["instructions"] = ["callable": "dynamic"]
    case nil: signature["instructions"] = .null
    }
    switch agent.model {
    case .name(let name)?: signature["model"] = .string(name)
    case .instance(let model)?: signature["model"] = ["type": .string(String(describing: type(of: model)))]
    case nil: signature["model"] = .null
    }
    signature["output_type"] = agent.outputType.map { .string($0.name()) } ?? .null
    signature["hooks"] = agent.hooks.map { ["type": .string(String(describing: type(of: $0)))] } ?? .null
    return stableIdentityText(.object(signature))
}

private struct AgentIdentitySortKey: Comparable {
    var priority: Int
    var signature: String
    var originalIndex: Int

    static func < (lhs: AgentIdentitySortKey, rhs: AgentIdentitySortKey) -> Bool {
        (lhs.priority, lhs.signature, lhs.originalIndex) < (rhs.priority, rhs.signature, rhs.originalIndex)
    }
}

private func agentIdentitySortKey(_ agent: Agent, rootAgent: Agent, originalIndex: Int) -> AgentIdentitySortKey {
    AgentIdentitySortKey(priority: agent === rootAgent ? 0 : 1, signature: agentIdentitySignature(agent), originalIndex: originalIndex)
}

/// Finds owners that are distinguishable only by graph traversal order.
func getAmbiguousAgentIds(_ initialAgent: Agent) -> Set<ObjectIdentifier> {
    var groups: [String: [ObjectIdentifier]] = [:]
    for agent in iterAgentGraph(initialAgent) {
        let key = agentIdentitySortKey(agent, rootAgent: initialAgent, originalIndex: 0)
        groups["\(agent.name)\u{0}\(key.priority)\u{0}\(key.signature)", default: []].append(ObjectIdentifier(agent))
    }
    return Set(groups.values.filter { $0.count > 1 }.flatMap { $0 })
}

/// Builds a stable identity map that preserves duplicate agent names.
///
/// The first agent for each name keeps the bare name; duplicates get `name#2`, `name#3`, and so
/// on, skipping suffixes that collide with literal agent names.
func buildAgentIdentityMap(_ initialAgent: Agent) -> [String: Agent] {
    let orderedAgents = iterAgentGraph(initialAgent)
    var originalIndices: [ObjectIdentifier: Int] = [:]
    for (index, agent) in orderedAgents.enumerated() { originalIndices[ObjectIdentifier(agent)] = index }
    let literalNames = Set(orderedAgents.map(\.name))
    var agentsByName: [String: [Agent]] = [:]
    for agent in orderedAgents { agentsByName[agent.name, default: []].append(agent) }

    var identityMap: [String: Agent] = [:]
    var usedIdentities: Set<String> = []
    var processedNames: Set<String> = []
    for agent in orderedAgents {
        let agentName = agent.name
        guard processedNames.insert(agentName).inserted else { continue }
        let sortedGroup = (agentsByName[agentName] ?? []).sorted {
            agentIdentitySortKey($0, rootAgent: initialAgent, originalIndex: originalIndices[ObjectIdentifier($0)] ?? 0)
                < agentIdentitySortKey($1, rootAgent: initialAgent, originalIndex: originalIndices[ObjectIdentifier($1)] ?? 0)
        }
        guard let baseAgent = sortedGroup.first else { continue }
        usedIdentities.insert(agentName)
        identityMap[agentName] = baseAgent
        var nextIndex = 2
        for duplicate in sortedGroup.dropFirst() {
            var candidate = "\(agentName)#\(nextIndex)"
            while usedIdentities.contains(candidate) || literalNames.contains(candidate) {
                nextIndex += 1
                candidate = "\(agentName)#\(nextIndex)"
            }
            usedIdentities.insert(candidate)
            identityMap[candidate] = duplicate
            nextIndex += 1
        }
    }
    return identityMap
}

/// Builds stable identity keys for the reachable agent graph.
func buildAgentIdentityKeysById(_ initialAgent: Agent) -> [ObjectIdentifier: String] {
    var keys: [ObjectIdentifier: String] = [:]
    for (identity, agent) in buildAgentIdentityMap(initialAgent) {
        keys[ObjectIdentifier(agent)] = identity
    }
    return keys
}
