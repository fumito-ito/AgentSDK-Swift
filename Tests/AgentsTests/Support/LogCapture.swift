import Foundation
@testable import Agents

// Equivalent of pytest's `caplog` fixture.

/// Collects SDK log records emitted by the current task (and its child tasks).
final class LogCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var collected: [LogRecord] = []
    let level: LogLevel

    init(level: LogLevel) {
        self.level = level
    }

    var records: [LogRecord] { lock.locked { collected } }
    var messages: [String] { records.map(\.message) }

    func append(_ record: LogRecord) {
        guard record.level >= level else { return }
        lock.locked { collected.append(record) }
    }

    @TaskLocal static var current: LogCapture?

    static let installed: Bool = {
        logger.addHandler { record in LogCapture.current?.append(record) }
        if logger.level > .debug { logger.level = .debug }
        return true
    }()
}

/// Runs `body` while capturing SDK log records at or above `level`.
func captureLogs<T>(level: LogLevel = .warning, _ body: (LogCapture) async throws -> T) async rethrows -> T {
    _ = LogCapture.installed
    let capture = LogCapture(level: level)
    return try await LogCapture.$current.withValue(capture) {
        try await body(capture)
    }
}
