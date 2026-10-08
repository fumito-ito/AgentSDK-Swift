import Foundation
import Agents

// Port of examples/basic/hello_world.py. Requires OPENAI_API_KEY.

let agent = Agent(
    name: "Assistant",
    instructions: "You only respond in haikus."
)

let result = try await Runner.run(agent, input: "Tell me about recursion in programming.")
print(result.finalOutput ?? "")
// Function calls itself,
// Looping in smaller pieces,
// Endless by design.
