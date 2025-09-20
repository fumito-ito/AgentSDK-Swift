import Foundation

/// Represents a tool that can be used by an agent to perform actions
public struct Tool<Context>: Sendable {
    /// Controls the availability of the tool for a given run context.
    public enum Availability: Sendable {
        case always
        case disabled
        case whenEnabled(@Sendable (_ context: RunContext<Context>) async -> Bool)

        func resolve(for context: RunContext<Context>) async -> Bool {
            switch self {
            case .always:
                return true
            case .disabled:
                return false
            case .whenEnabled(let closure):
                return await closure(context)
            }
        }
    }

    /// The name of the tool
    public let name: String

    /// A description of what the tool does
    public let description: String

    /// The parameters required by the tool
    public let parameters: [Parameter]

    /// Availability strategy for the tool
    public let availability: Availability

    /// The function to execute when the tool is called
    private let executeClosure: @Sendable (ToolParameters, RunContext<Context>) async throws -> Any

    /// Creates a new tool
    /// - Parameters:
    ///   - name: The name of the tool
    ///   - description: A description of what the tool does
    ///   - parameters: The parameters required by the tool
    ///   - availability: Availability strategy for the tool
    ///   - execute: The function to execute when the tool is called that receives the run context
    public init(
        name: String,
        description: String,
        parameters: [Parameter] = [],
        availability: Availability = .always,
        executeClosure: @Sendable @escaping (ToolParameters, RunContext<Context>) async throws -> Any
    ) {
        self.name = name
        self.description = description
        self.parameters = parameters
        self.availability = availability
        self.executeClosure = executeClosure
    }

    /// Convenience initializer mirroring the previous signature that only exposed context value.
    /// - Parameters:
    ///   - name: The name of the tool
    ///   - description: A description of what the tool does
    ///   - parameters: The parameters required by the tool
    ///   - availability: Availability strategy for the tool
    ///   - execute: The function to execute when the tool is called
    public init(
        name: String,
        description: String,
        parameters: [Parameter] = [],
        availability: Availability = .always,
        execute: @Sendable @escaping (ToolParameters, Context) async throws -> Any
    ) {
        self.init(
            name: name,
            description: description,
            parameters: parameters,
            availability: availability
        ) { params, runContext in
            try await execute(params, runContext.value)
        }
    }

    /// Executes the tool with the provided parameters and context
    /// - Parameters:
    ///   - parameters: The parameters for the tool execution
    ///   - runContext: The run context for the tool execution
    /// - Returns: The result of the tool execution
    public func invoke(
        parameters: ToolParameters,
        runContext: RunContext<Context>
    ) async throws -> Any {
        try await executeClosure(parameters, runContext)
    }

    /// Executes the tool with the provided parameters and context (backwards compatibility helper)
    /// - Parameters:
    ///   - parameters: The parameters for the tool execution
    ///   - context: The context for the tool execution
    /// - Returns: The result of the tool execution
    @available(*, deprecated, message: "Use invoke(parameters:runContext:) instead to access usage.")
    public func callAsFunction(_ parameters: ToolParameters, context: Context) async throws -> Any {
        let runContext = RunContext(value: context)
        return try await invoke(parameters: parameters, runContext: runContext)
    }

    /// Determines whether the tool is enabled for the provided run context.
    /// - Parameter runContext: The run context to check.
    /// - Returns: True if the tool may be invoked.
    public func isEnabled(for runContext: RunContext<Context>) async -> Bool {
        await availability.resolve(for: runContext)
    }

    /// Represents a parameter for a tool
    public struct Parameter: Sendable {
        /// The name of the parameter
        public let name: String

        /// A description of the parameter
        public let description: String

        /// The type of the parameter
        public let type: ParameterType

        /// Whether the parameter is required
        public let required: Bool

        /// Creates a new parameter
        /// - Parameters:
        ///   - name: The name of the parameter
        ///   - description: A description of the parameter
        ///   - type: The type of the parameter
        ///   - required: Whether the parameter is required
        public init(name: String, description: String, type: ParameterType, required: Bool = true) {
            self.name = name
            self.description = description
            self.type = type
            self.required = required
        }
    }

    /// Represents the type of a parameter
    public enum ParameterType: Sendable {
        case string
        case number
        case boolean
        case array
        case object

        /// Returns the string representation of the type for OpenAI
        public var jsonType: String {
            switch self {
            case .string: return "string"
            case .number: return "number"
            case .boolean: return "boolean"
            case .array: return "array"
            case .object: return "object"
            }
        }
    }
}

/// Represents the parameters passed to a tool
public typealias ToolParameters = [String: Any]

/// Creates a function tool from a function
/// - Parameters:
///   - name: The name of the tool
///   - description: A description of what the tool does
///   - function: The function to execute when the tool is called
/// - Returns: A new function tool
public func functionTool<Context, Input: Decodable, Output>(
    name: String,
    description: String,
    availability: Tool<Context>.Availability = .always,
    function: @Sendable @escaping (Input, RunContext<Context>) async throws -> Output
) -> Tool<Context> {
    Tool(name: name, description: description, availability: availability) { parameters, runContext in
        // Convert parameters dictionary to Input type
        let data = try JSONSerialization.data(withJSONObject: parameters)
        let input = try JSONDecoder().decode(Input.self, from: data)

        // Call the function with the decoded input and run context
        return try await function(input, runContext)
    }
}

/// Convenience overload preserving the previous signature using only the context value.
public func functionTool<Context, Input: Decodable, Output>(
    name: String,
    description: String,
    availability: Tool<Context>.Availability = .always,
    function: @Sendable @escaping (Input, Context) async throws -> Output
) -> Tool<Context> {
    Tool(name: name, description: description, availability: availability) { parameters, runContext in
        // Convert parameters dictionary to Input type
        let data = try JSONSerialization.data(withJSONObject: parameters)
        let input = try JSONDecoder().decode(Input.self, from: data)

        // Call the function with the decoded input and context value
        return try await function(input, runContext.value)
    }
}
