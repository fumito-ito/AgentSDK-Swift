import Foundation
import Testing
@testable import Agents

/// Equivalent of `pytest.raises(ErrorType, match=substring)`.
func expectError<E: Error>(
    _ type: E.Type,
    containing substring: String? = nil,
    sourceLocation: SourceLocation = #_sourceLocation,
    _ body: () throws -> Void
) {
    do {
        try body()
        Issue.record("Expected \(String(describing: E.self)) to be thrown", sourceLocation: sourceLocation)
    } catch let error as E {
        if let substring {
            let message = describeError(error)
            #expect(message.contains(substring), "\(message) does not contain \(substring)", sourceLocation: sourceLocation)
        }
    } catch {
        Issue.record("Expected \(String(describing: E.self)) but got \(Swift.type(of: error)): \(error)", sourceLocation: sourceLocation)
    }
}

/// Async equivalent of `pytest.raises(ErrorType, match=substring)`. Returns the thrown error.
@discardableResult
func expectErrorAsync<E: Error>(
    _ type: E.Type,
    containing substring: String? = nil,
    sourceLocation: SourceLocation = #_sourceLocation,
    _ body: () async throws -> Void
) async -> E? {
    do {
        try await body()
        Issue.record("Expected \(String(describing: E.self)) to be thrown", sourceLocation: sourceLocation)
        return nil
    } catch let error as E {
        if let substring {
            let message = describeError(error)
            #expect(message.contains(substring), "\(message) does not contain \(substring)", sourceLocation: sourceLocation)
        }
        return error
    } catch {
        Issue.record("Expected \(String(describing: E.self)) but got \(Swift.type(of: error)): \(error)", sourceLocation: sourceLocation)
        return nil
    }
}
