import Foundation

/// Log levels matching Python's `logging` levels.
public enum LogLevel: Int, Sendable, Comparable, CustomStringConvertible {
    case debug = 10
    case info = 20
    case warning = 30
    case error = 40

    public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool { lhs.rawValue < rhs.rawValue }

    public var description: String {
        switch self {
        case .debug: return "DEBUG"
        case .info: return "INFO"
        case .warning: return "WARNING"
        case .error: return "ERROR"
        }
    }
}

/// A log record delivered to a log handler.
public struct LogRecord: Sendable {
    public var level: LogLevel
    public var message: String
    public var loggerName: String
}

/// The SDK logger (`logging.getLogger("openai.agents")` upstream).
///
/// Swift: Foundation has no portable logging framework (`os.Logger` is Apple-only), so the SDK
/// logs through handlers that applications can replace. Messages are evaluated lazily.
public final class AgentsLogger: @unchecked Sendable {
    public typealias Handler = @Sendable (LogRecord) -> Void

    public let name: String
    private let lock = NSLock()
    private var handlers: [Handler] = []
    private var minimumLevel: LogLevel = .warning

    init(name: String) {
        self.name = name
    }

    /// The minimum level that is delivered to handlers.
    public var level: LogLevel {
        get { lock.lock(); defer { lock.unlock() }; return minimumLevel }
        set { lock.lock(); minimumLevel = newValue; lock.unlock() }
    }

    /// Adds a handler that receives every record at or above `level`.
    public func addHandler(_ handler: @escaping Handler) {
        lock.lock()
        handlers.append(handler)
        lock.unlock()
    }

    /// Removes all handlers.
    public func removeAllHandlers() {
        lock.lock()
        handlers.removeAll()
        lock.unlock()
    }

    public func isEnabled(for level: LogLevel) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return level >= minimumLevel && !handlers.isEmpty
    }

    public func log(_ level: LogLevel, _ message: @autoclosure () -> String) {
        lock.lock()
        let enabled = level >= minimumLevel
        let currentHandlers = handlers
        lock.unlock()
        guard enabled, !currentHandlers.isEmpty else { return }
        let record = LogRecord(level: level, message: message(), loggerName: name)
        for handler in currentHandlers {
            handler(record)
        }
    }

    public func debug(_ message: @autoclosure () -> String) { log(.debug, message()) }
    public func info(_ message: @autoclosure () -> String) { log(.info, message()) }
    public func warning(_ message: @autoclosure () -> String) { log(.warning, message()) }
    public func error(_ message: @autoclosure () -> String) { log(.error, message()) }
}

/// The SDK logger.
public let logger = AgentsLogger(name: "openai.agents")

/// Enables verbose logging to stdout. This is useful for debugging.
public func enableVerboseStdoutLogging() {
    logger.level = .debug
    logger.removeAllHandlers()
    logger.addHandler { record in
        print("\(record.level) [\(record.loggerName)] \(record.message)")
    }
}

/// Logs a model or tool action error with redaction applied when data logging is disabled.
func logActionError(_ message: String, _ error: any Error, isToolAction: Bool) {
    let redact = isToolAction ? AgentsDebug.dontLogToolData : AgentsDebug.dontLogModelData
    if redact {
        logger.error("\(message): \(errorTypeName(error))")
    } else {
        logger.error("\(message): \(describeError(error))")
    }
}

/// Logs a model-and-tool action warning with redaction applied when data logging is disabled
/// (upstream `log_model_and_tool_action_warning`).
func logActionWarning(_ message: String, _ error: any Error) {
    if AgentsDebug.dontLogModelData || AgentsDebug.dontLogToolData {
        logger.warning("\(message): \(errorTypeName(error))")
    } else {
        logger.warning("\(message): \(describeError(error))")
    }
}

/// Debug flags read from the environment (src/agents/_debug.py).
public enum AgentsDebug {
    private static func flag(_ name: String, default defaultValue: Bool) -> Bool {
        guard let value = ProcessInfo.processInfo.environment[name] else { return defaultValue }
        return value == "1" || value.lowercased() == "true"
    }

    private static let state = SharedBox((
        dontLogModelData: flag("OPENAI_AGENTS_DONT_LOG_MODEL_DATA", default: true),
        dontLogToolData: flag("OPENAI_AGENTS_DONT_LOG_TOOL_DATA", default: true)
    ))

    /// Task-scoped overrides for the flags, used by `withOverrides`.
    struct Overrides: Sendable {
        var dontLogModelData: Bool?
        var dontLogToolData: Bool?
    }

    @TaskLocal static var overrides = Overrides()

    /// By default LLM inputs/outputs are not logged. Set `OPENAI_AGENTS_DONT_LOG_MODEL_DATA=0` to
    /// log them.
    public static var dontLogModelData: Bool {
        get { overrides.dontLogModelData ?? state.value.dontLogModelData }
        set { state.mutate { $0.dontLogModelData = newValue } }
    }

    /// By default tool inputs/outputs are not logged. Set `OPENAI_AGENTS_DONT_LOG_TOOL_DATA=0` to
    /// log them.
    public static var dontLogToolData: Bool {
        get { overrides.dontLogToolData ?? state.value.dontLogToolData }
        set { state.mutate { $0.dontLogToolData = newValue } }
    }

    /// Runs `operation` with the flags overridden for the current task and its children.
    public static func withOverrides<T>(
        dontLogModelData: Bool? = nil,
        dontLogToolData: Bool? = nil,
        _ operation: () async throws -> T
    ) async rethrows -> T {
        var next = overrides
        if let dontLogModelData { next.dontLogModelData = dontLogModelData }
        if let dontLogToolData { next.dontLogToolData = dontLogToolData }
        return try await $overrides.withValue(next, operation: operation)
    }
}
