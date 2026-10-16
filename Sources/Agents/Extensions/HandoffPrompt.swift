import Foundation

/// A recommended prompt prefix for agents that use handoffs. We recommend including this or
/// similar instructions in any agents that use handoffs.
public let recommendedPromptPrefix = "# System context\n"
    + "You are part of a multi-agent system called the Agents SDK, designed to make agent "
    + "coordination and execution easy. Agents uses two primary abstraction: **Agents** and "
    + "**Handoffs**. An agent encompasses instructions and tools and can hand off a "
    + "conversation to another agent when appropriate. "
    + "Handoffs are achieved by calling a handoff function, generally named "
    + "`transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background;"
    + " do not mention or draw attention to these transfers in your conversation with the user.\n"

/// Adds recommended instructions to the prompt for agents that use handoffs.
public func promptWithHandoffInstructions(_ prompt: String) -> String {
    "\(recommendedPromptPrefix)\n\n\(prompt)"
}
