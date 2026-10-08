import Foundation

/// Settings for session operations.
public struct SessionSettings: Sendable, Equatable {
    /// Maximum number of items to retrieve. `nil` retrieves all items.
    public var limit: Int?

    public init(limit: Int? = nil) {
        self.limit = limit
    }

    /// Overlays any non-`nil` values from `override`.
    public func resolve(_ override: SessionSettings?) -> SessionSettings {
        guard let override else { return self }
        return SessionSettings(limit: override.limit ?? limit)
    }

    public func toDict() -> JSONObject {
        ["limit": limit.map(JSONValue.int) ?? .null]
    }
}

/// Resolves the effective limit for session operations.
public func resolveSessionLimit(_ explicitLimit: Int?, _ settings: SessionSettings?) -> Int? {
    explicitLimit ?? settings?.limit
}

/// Stores conversation history for a specific session, so agents maintain context without
/// explicit manual memory management.
public protocol Session: AnyObject, Sendable {
    /// The session identifier.
    var sessionId: String { get }
    /// Default settings for this session.
    var sessionSettings: SessionSettings? { get }
    /// Retrieves the conversation history. With `limit`, returns the latest N items in
    /// chronological order.
    func getItems(limit: Int?) async throws -> [TResponseInputItem]
    /// Adds new items to the conversation history.
    func addItems(_ items: [TResponseInputItem]) async throws
    /// Removes and returns the most recent item, if any.
    func popItem() async throws -> TResponseInputItem?
    /// Clears all items for this session.
    func clearSession() async throws
}

extension Session {
    public var sessionSettings: SessionSettings? { nil }

    public func getItems() async throws -> [TResponseInputItem] {
        try await getItems(limit: nil)
    }
}

/// Arguments for `OpenAIResponsesCompactionAwareSession.runCompaction`.
public struct OpenAIResponsesCompactionArgs: Sendable, Equatable {
    public var responseId: String?
    /// `"previous_response_id"`, `"input"`, or `"auto"`.
    public var compactionMode: String?
    public var store: Bool?
    public var force: Bool?

    public init(responseId: String? = nil, compactionMode: String? = nil, store: Bool? = nil, force: Bool? = nil) {
        self.responseId = responseId
        self.compactionMode = compactionMode
        self.store = store
        self.force = force
    }
}

/// A session that can compact its history through the Responses API.
public protocol OpenAIResponsesCompactionAwareSession: Session {
    func runCompaction(_ args: OpenAIResponsesCompactionArgs?) async throws
}

/// Returns whether `session` supports Responses API compaction.
public func isOpenAIResponsesCompactionAwareSession(_ session: (any Session)?) -> Bool {
    session is any OpenAIResponsesCompactionAwareSession
}

/// Combines session history with new input before a run.
public typealias SessionInputCallback = @Sendable ([TResponseInputItem], [TResponseInputItem]) async throws -> [TResponseInputItem]
