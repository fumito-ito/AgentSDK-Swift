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
    
    public init<G: InputGuardrail>(_ guardrail: G) where G.Context == Context {
        validator = guardrail.validate
    }
    
    public init(_ validator: @Sendable @escaping (String, Context) throws -> String) {
        self.validator = validator
    }
    
    public func validate(_ input: String, context: Context) throws -> String {
        try validator(input, context)
    }
}

/// Type-erased wrapper for output guardrails.
public struct AnyOutputGuardrail<Context>: Sendable {
    private let validator: @Sendable (String, Context) throws -> String
    
    public init<G: OutputGuardrail>(_ guardrail: G) where G.Context == Context {
        validator = guardrail.validate
    }
    
    public init(_ validator: @Sendable @escaping (String, Context) throws -> String) {
        self.validator = validator
    }
    
    public func validate(_ output: String, context: Context) throws -> String {
        try validator(output, context)
    }
}

/// A guardrail that enforces constraints on input length.
public struct InputLengthGuardrail: InputGuardrail {
    public typealias Context = Void

    private let maxLength: Int
    
    public init(maxLength: Int) {
        self.maxLength = maxLength
    }
    
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
    
    public init(pattern: String, blockMatches: Bool = true) throws {
        self.regex = try NSRegularExpression(pattern: pattern, options: [])
        self.blockMatches = blockMatches
    }
    
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
