import Foundation

/// Tracks token and request usage for a run.
public struct Usage: Sendable {
    /// Total requests made to the model provider.
    public private(set) var requests: Int
    /// Total input tokens sent across all requests.
    public private(set) var inputTokens: Int
    /// Total output tokens received across all requests.
    public private(set) var outputTokens: Int
    /// Combined input and output tokens across all requests.
    public private(set) var totalTokens: Int

    /// Creates a new usage tracker.
    /// - Parameters:
    ///   - requests: Initial number of requests.
    ///   - inputTokens: Initial number of input tokens.
    ///   - outputTokens: Initial number of output tokens.
    ///   - totalTokens: Initial total tokens.
    public init(
        requests: Int = 0,
        inputTokens: Int = 0,
        outputTokens: Int = 0,
        totalTokens: Int = 0
    ) {
        self.requests = requests
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.totalTokens = totalTokens
    }

    /// Adds usage details from a model response.
    /// - Parameter usage: Usage information returned by the model response.
    public mutating func add(_ usage: ModelResponse.Usage?) {
        guard let usage = usage else { return }
        requests += 1
        inputTokens += usage.promptTokens
        outputTokens += usage.completionTokens
        totalTokens += usage.totalTokens
    }

    /// Merges another usage tracker into this one.
    /// - Parameter other: The usage to merge.
    public mutating func merge(_ other: Usage) {
        requests += other.requests
        inputTokens += other.inputTokens
        outputTokens += other.outputTokens
        totalTokens += other.totalTokens
    }
}
