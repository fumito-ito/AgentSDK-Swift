import Foundation

// Port of the header helpers in src/agents/models/_retry_runtime.py.
//
// Swift: upstream inspects arbitrary exception attributes and `__cause__` chains. Swift errors
// carry retry metadata as typed `ModelAPIError` properties, so only the header parsing is ported.

/// Returns the header value for `name`, matching case-insensitively.
func headerLookup(_ headers: [String: String]?, _ name: String) -> String? {
    guard let headers else { return nil }
    if let exact = headers[name] { return exact }
    return headers.first { $0.key.caseInsensitiveCompare(name) == .orderedSame }?.value
}

/// Parses a `retry-after-ms` header into seconds, rejecting invalid or negative values.
func parseRetryAfterMs(_ value: String?) -> Double? {
    guard let value, let parsed = Double(value.trimmingCharacters(in: .whitespaces)) else { return nil }
    let seconds = parsed / 1000.0
    return seconds >= 0 ? seconds : nil
}

private let httpDateFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = TimeZone(identifier: "GMT")
    formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
    return formatter
}()

private let httpDateLock = NSLock()

/// Parses a `retry-after` header given in seconds or as an HTTP date.
func parseRetryAfterValue(_ value: String?, now: Date = Date()) -> Double? {
    guard let value else { return nil }
    let trimmed = value.trimmingCharacters(in: .whitespaces)
    if let parsed = Double(trimmed) {
        return parsed >= 0 ? parsed : nil
    }
    guard let date = httpDateLock.locked({ httpDateFormatter.date(from: trimmed) }) else { return nil }
    return max(date.timeIntervalSince(now), 0)
}

/// Returns the server-suggested retry delay from response headers.
func getRetryAfter(_ headers: [String: String]?) -> Double? {
    parseRetryAfterMs(headerLookup(headers, "retry-after-ms")) ?? parseRetryAfterValue(headerLookup(headers, "retry-after"))
}
