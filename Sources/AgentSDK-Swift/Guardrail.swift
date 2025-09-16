import Foundation

/// Marker protocol for guardrail errors.
public enum GuardrailError: Error, Sendable {
    case invalidInput(reason: String)
    case invalidOutput(reason: String)
}

/// Protocol for enforcing constraints on agent input.
public protocol InputGuardrail<Context>: Sendable {
    associatedtype Context
    func validate(_ input: String, context: Context) throws -> String
}

/// Protocol for enforcing constraints on agent output.
public protocol OutputGuardrail<Context>: Sendable {
    associatedtype Context
    func validate(_ output: String, context: Context) throws -> String
}

/// Type-erased wrapper for input guardrails.
public struct AnyInputGuardrail<Context>: Sendable {
    private let validator: @Sendable (String, Context) throws -> String
    
    /// Creates a type-erased wrapper around a strongly typed guardrail.
    /// - Parameter guardrail: The concrete guardrail to wrap.
    public init<G: InputGuardrail>(_ guardrail: G) where G.Context == Context {
        validator = guardrail.validate
    }
    
    /// Creates a type-erased wrapper from a validation closure.
    /// - Parameter validator: Closure that performs validation for the provided context.
    public init(_ validator: @Sendable @escaping (String, Context) throws -> String) {
        self.validator = validator
    }
    
    /// Validates input text using the wrapped guardrail logic.
    /// - Parameters:
    ///   - input: The input text to validate.
    ///   - context: The associated context forwarded to the guardrail.
    /// - Returns: The validated (and potentially transformed) input string.
    /// - Throws: `GuardrailError` if the guardrail fails validation.
    public func validate(_ input: String, context: Context) throws -> String {
        try validator(input, context)
    }
}

/// Type-erased wrapper for output guardrails.
public struct AnyOutputGuardrail<Context>: Sendable {
    private let validator: @Sendable (String, Context) throws -> String
    
    /// Creates a type-erased wrapper around a strongly typed guardrail.
    /// - Parameter guardrail: The concrete guardrail to wrap.
    public init<G: OutputGuardrail>(_ guardrail: G) where G.Context == Context {
        validator = guardrail.validate
    }
    
    /// Creates a type-erased wrapper from a validation closure.
    /// - Parameter validator: Closure that performs validation for the provided context.
    public init(_ validator: @Sendable @escaping (String, Context) throws -> String) {
        self.validator = validator
    }
    
    /// Validates output text using the wrapped guardrail logic.
    /// - Parameters:
    ///   - output: The output text to validate.
    ///   - context: The associated context forwarded to the guardrail.
    /// - Returns: The validated (and potentially transformed) output string.
    /// - Throws: `GuardrailError` if the guardrail fails validation.
    public func validate(_ output: String, context: Context) throws -> String {
        try validator(output, context)
    }
}

/// A guardrail that enforces constraints on input length.
public struct InputLengthGuardrail: InputGuardrail {
    public typealias Context = Void

    private let maxLength: Int
    
    /// Creates a guardrail that enforces a maximum input length.
    /// - Parameter maxLength: The maximum number of characters permitted.
    public init(maxLength: Int) {
        self.maxLength = maxLength
    }
    
    /// Validates that the provided input does not exceed the configured maximum length.
    /// - Parameters:
    ///   - input: The input text to validate.
    ///   - context: The context supplied during validation (unused by default).
    /// - Returns: The original input if validation succeeds.
    /// - Throws: `GuardrailError.invalidInput` when the input is too long.
    public func validate(_ input: String, context: Context) throws -> String {
        if input.count > maxLength {
            throw GuardrailError.invalidInput(
                reason: "Input is too long. Maximum length is \(maxLength) characters."
            )
        }
        return input
    }
}

/// A guardrail that enforces constraints on output content using a regular expression.
public struct RegexContentGuardrail: OutputGuardrail {
    public typealias Context = Void

    private let regex: NSRegularExpression
    private let blockMatches: Bool
    
    /// Creates a guardrail that evaluates outputs against a regular expression.
    /// - Parameters:
    ///   - pattern: The regex pattern used for validation.
    ///   - blockMatches: When `true`, outputs matching the pattern are blocked; when `false`,
    ///     outputs must contain a match.
    public init(pattern: String, blockMatches: Bool = true) throws {
        self.regex = try NSRegularExpression(pattern: pattern, options: [])
        self.blockMatches = blockMatches
    }
    
    /// Validates that the output satisfies the regex constraint.
    /// - Parameters:
    ///   - output: The output text to validate.
    ///   - context: Ignored placeholder context for protocol conformance.
    /// - Returns: The original output if validation succeeds.
    /// - Throws: `GuardrailError.invalidOutput` when the regex constraint is violated.
    public func validate(_ output: String, context: Context) throws -> String {
        let range = NSRange(location: 0, length: output.utf16.count)
        let matches = regex.matches(in: output, options: [], range: range)
        
        if blockMatches && !matches.isEmpty {
            throw GuardrailError.invalidOutput(reason: "Output contains blocked content.")
        } else if !blockMatches && matches.isEmpty {
            throw GuardrailError.invalidOutput(reason: "Output does not contain required content.")
        }
        
        return output
    }
}
