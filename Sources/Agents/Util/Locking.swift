import Foundation

extension NSLock {
    /// Runs `body` while holding the lock. Safe to call from asynchronous contexts.
    func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
