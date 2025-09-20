import Foundation

/// Represents a handoff from one agent to another
public struct Handoff<Context> {
    /// The agent to hand off to
    public let agent: Agent<Context>

    /// The filter to determine whether to hand off
    public let filter: any HandoffFilter<Context>

    /// Creates a new handoff
    /// - Parameters:
    ///   - agent: The agent to hand off to
    ///   - filter: The filter to determine whether to hand off
    public init(agent: Agent<Context>, filter: any HandoffFilter<Context>) {
        self.agent = agent
        self.filter = filter
    }

    /// Creates a new handoff with a keyword filter
    /// - Parameters:
    ///   - agent: The agent to hand off to
    ///   - keywords: The keywords to trigger the handoff
    ///   - caseSensitive: Whether the keyword matching is case sensitive
    /// - Returns: A new handoff
    public static func withKeywords(
        agent: Agent<Context>,
        keywords: [String],
        caseSensitive: Bool = false
    ) -> Handoff<Context> {
        let filter = KeywordHandoffFilter<Context>(
            keywords: keywords,
            caseSensitive: caseSensitive
        )

        return Handoff(agent: agent, filter: filter)
    }

    /// Creates a new handoff with a custom filter function
    /// - Parameters:
    ///   - agent: The agent to hand off to
    ///   - filterFunction: The function to determine whether to hand off
    /// - Returns: A new handoff
    public static func withCustomFilter(
        agent: Agent<Context>,
        filterFunction: @escaping (String, Context) -> Bool
    ) -> Handoff<Context> {
        let filter = CustomHandoffFilter<Context>(filterFunction: filterFunction)
        return Handoff(agent: agent, filter: filter)
    }
}

/// Protocol for determining whether to hand off to another agent
public protocol HandoffFilter<Context> {
    /// The context type for the filter
    associatedtype Context

    /// Determines whether to hand off to another agent
    /// - Parameters:
    ///   - input: The input to check
    ///   - context: The context for the check
    /// - Returns: True if the input should trigger a handoff, false otherwise
    func shouldHandoff(input: String, context: Context) -> Bool
}

/// A handoff filter that triggers on keywords
public struct KeywordHandoffFilter<Context>: HandoffFilter {
    /// The keywords to trigger the handoff
    private let keywords: [String]

    /// Whether the keyword matching is case sensitive
    private let caseSensitive: Bool

    /// Creates a new keyword handoff filter
    /// - Parameters:
    ///   - keywords: The keywords to trigger the handoff
    ///   - caseSensitive: Whether the keyword matching is case sensitive
    public init(keywords: [String], caseSensitive: Bool = false) {
        self.keywords = keywords
        self.caseSensitive = caseSensitive
    }

    /// Determines whether to hand off to another agent based on keywords
    /// - Parameters:
    ///   - input: The input to check
    ///   - context: The context for the check
    /// - Returns: True if the input contains any of the keywords, false otherwise
    public func shouldHandoff(input: String, context: Context) -> Bool {
        let searchInput = caseSensitive ? input : input.lowercased()

        for keyword in keywords {
            let searchKeyword = caseSensitive ? keyword : keyword.lowercased()
            if searchInput.contains(searchKeyword) {
                return true
            }
        }

        return false
    }
}

/// A handoff filter that uses a custom function
public struct CustomHandoffFilter<Context>: HandoffFilter {
    /// The function to determine whether to hand off
    private let filterFunction: (String, Context) -> Bool

    /// Creates a new custom handoff filter
    /// - Parameter filterFunction: The function to determine whether to hand off
    public init(filterFunction: @escaping (String, Context) -> Bool) {
        self.filterFunction = filterFunction
    }

    /// Determines whether to hand off to another agent using the custom function
    /// - Parameters:
    ///   - input: The input to check
    ///   - context: The context for the check
    /// - Returns: The result of the filter function
    public func shouldHandoff(input: String, context: Context) -> Bool {
        filterFunction(input, context)
    }
}
