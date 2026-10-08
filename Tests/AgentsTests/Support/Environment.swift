import Foundation
@testable import Agents

// Equivalent of pytest's `monkeypatch.setenv` / `monkeypatch.delenv`.

private let environmentLock = NSLock()

/// Runs `body` with environment variables temporarily set (`nil` unsets), restoring them after.
///
/// Suites that call this should be `.serialized`; the process environment is global.
func withEnvironment<T>(_ values: [String: String?], _ body: () async throws -> T) async rethrows -> T {
    let previous: [String: String?] = environmentLock.locked {
        var previous: [String: String?] = [:]
        for (key, value) in values {
            previous[key] = ProcessInfo.processInfo.environment[key]
            if let value { setenv(key, value, 1) } else { unsetenv(key) }
        }
        return previous
    }
    defer {
        environmentLock.locked {
            for (key, value) in previous {
                if let value { setenv(key, value, 1) } else { unsetenv(key) }
            }
        }
    }
    return try await body()
}
