import Foundation

/// Runs a simple REPL loop with the given agent.
///
/// This utility allows quick manual testing and debugging of an agent from the command line.
/// Conversation state is preserved across turns. Enter `exit` or `quit` to stop the loop.
///
/// Swift: `readInput` and `write` default to standard input and output; tests inject them in
/// place of upstream's monkeypatched `input()` and captured stdout.
///
/// - Parameters:
///   - agent: The starting agent to run.
///   - stream: Whether to stream the agent output.
///   - context: Additional context information to pass to the runner.
///   - maxTurns: Maximum number of turns for the runner to iterate. Pass `nil` to disable the
///     turn limit.
public func runDemoLoop(
    _ agent: Agent,
    stream: Bool = true,
    context: any Sendable = (),
    maxTurns: Int? = defaultMaxTurns,
    readInput: (String) -> String? = { prompt in
        print(prompt, terminator: "")
        return readLine()
    },
    write: (String) -> Void = { text in
        print(text, terminator: "")
        // Flush every stream: the global `stdout` is not concurrency-safe under Swift 6 on Linux.
        fflush(nil)
    }
) async throws {
    var currentAgent = agent
    var inputItems: [TResponseInputItem] = []
    while true {
        guard let userInput = readInput(" > ") else {
            write("\n")
            break
        }
        let trimmed = userInput.trimmingCharacters(in: .whitespacesAndNewlines)
        if ["exit", "quit"].contains(trimmed.lowercased()) {
            break
        }
        if trimmed.isEmpty {
            continue
        }

        inputItems.append(["role": "user", "content": .string(userInput)])

        let result: any RunResultBase
        if stream {
            let streamed = try Runner.runStreamed(currentAgent, input: .items(inputItems), context: context, maxTurns: maxTurns)
            for try await event in streamed.streamEvents() {
                switch event {
                case .rawResponse(let raw):
                    if raw.data.type == "response.output_text.delta", let delta = raw.data.string("delta") {
                        write(delta)
                    }
                case .runItem(let runItem):
                    if runItem.item.type == "tool_call_item" {
                        write("\n[tool called]\n")
                    } else if let output = runItem.item as? ToolCallOutputItem {
                        write("\n[tool output: \(output.output)]\n")
                    }
                case .agentUpdated(let updated):
                    write("\n[Agent updated: \(updated.newAgent.name)]\n")
                }
            }
            write("\n")
            result = streamed
        } else {
            let completed = try await Runner.run(currentAgent, input: .items(inputItems), context: context, maxTurns: maxTurns)
            if let finalOutput = completed.finalOutput {
                write("\(finalOutput)\n")
            }
            result = completed
        }

        currentAgent = result.lastAgent
        inputItems = try result.toInputList()
    }
}
