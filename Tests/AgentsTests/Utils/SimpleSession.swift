import Foundation
@testable import Agents

// Port of tests/utils/simple_session.py.

/// A minimal in-memory session implementation for tests.
class SimpleListSession: Session, @unchecked Sendable {
    let sessionId: String
    var sessionSettings: SessionSettings?
    private let lock = NSLock()
    private var items: [TResponseInputItem]

    init(sessionId: String = "test", history: [TResponseInputItem] = []) {
        self.sessionId = sessionId
        self.items = history
    }

    /// The items saved so far, for inspection.
    var savedItems: [TResponseInputItem] { lock.locked { items } }

    func getItems(limit: Int?) async throws -> [TResponseInputItem] {
        lock.locked {
            guard let limit else { return items }
            if limit <= 0 { return [] }
            return Array(items.suffix(limit))
        }
    }

    func addItems(_ newItems: [TResponseInputItem]) async throws {
        lock.locked { items.append(contentsOf: newItems) }
    }

    func popItem() async throws -> TResponseInputItem? {
        lock.locked { items.popLast() }
    }

    func clearSession() async throws {
        lock.locked { items.removeAll() }
    }
}

/// Session that tracks how many times `popItem` is invoked (for rewind tests).
class CountingSession: SimpleListSession, @unchecked Sendable {
    private let counter = SharedBox(0)
    var popCalls: Int { counter.value }

    override func popItem() async throws -> TResponseInputItem? {
        counter.mutate { $0 += 1 }
        return try await super.popItem()
    }
}

/// Session that strips IDs on add to mimic hosted stores that reassign IDs.
final class IdStrippingSession: CountingSession, @unchecked Sendable {
    override func addItems(_ newItems: [TResponseInputItem]) async throws {
        try await super.addItems(newItems.map { $0.removing("id") })
    }
}
