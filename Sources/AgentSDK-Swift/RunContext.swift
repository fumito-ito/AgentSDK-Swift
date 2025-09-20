import Foundation

/// Wraps the caller-provided context and aggregates usage for the duration of a run.
public final class RunContext<Context>: @unchecked Sendable {
    /// The user provided context object.
    public let value: Context
    /// Accumulated model usage.
    public private(set) var usage: Usage

    /// Creates a new run context wrapper.
    /// - Parameters:
    ///   - value: The underlying context value.
    ///   - usage: Optional initial usage information.
    public init(value: Context, usage: Usage = Usage()) {
        self.value = value
        self.usage = usage
    }

    /// Updates the usage with information from a model response.
    /// - Parameter usage: The usage information returned by the model.
    public func recordUsage(_ usage: ModelResponse.Usage?) {
        var aggregated = self.usage
        aggregated.add(usage)
        self.usage = aggregated
    }

    /// Merges usage from another run context.
    /// - Parameter other: The run context to merge.
    public func mergeUsage(from other: RunContext<Context>) {
        var aggregated = usage
        aggregated.merge(other.usage)
        usage = aggregated
    }
}
