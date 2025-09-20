import Testing
@testable import AgentSDK_Swift

// MARK: - Agent Tests

@Test func testAgentCreation() async throws {
    // Create a simple agent
    let agent = Agent<Void>(
        name: "TestAgent",
        instructions: "You are a helpful assistant."
    )

    #expect(agent.name == "TestAgent")
    if case .literal(let instructions)? = agent.instructions {
        #expect(instructions == "You are a helpful assistant.")
    } else {
        #expect(Bool(false), "Instructions should be literal")
    }
    #expect(agent.tools.isEmpty)
    #expect(agent.inputGuardrails.isEmpty)
    #expect(agent.outputGuardrails.isEmpty)
    #expect(agent.handoffs.isEmpty)
}

@Test func testAgentCreationWithFullConfig() async throws {
    // Create tools
    let tool1 = Tool<Void>(
        name: "echo",
        description: "Echoes the input",
        execute: { params, _ in
            return params["text"] as? String ?? "No text provided"
        }
    )

    let tool2 = Tool<Void>(
        name: "reverse",
        description: "Reverses the input",
        execute: { params, _ in
            let text = params["text"] as? String ?? ""
            return String(text.reversed())
        }
    )

    // Create guardrails
    let inputGuardrail = AnyInputGuardrail(InputLengthGuardrail(maxLength: 100))

    // Create model settings
    let modelSettings = ModelSettings(
        modelName: "test-model",
        temperature: 0.7,
        topP: 0.9,
        maxTokens: 1000
    )

    // Create agent with all components
    let agent = Agent<Void>(
        name: "FullConfigAgent",
        instructions: "You are a comprehensive test agent.",
        tools: [tool1, tool2],
        inputGuardrails: [inputGuardrail],
        modelSettings: modelSettings
    )

    #expect(agent.name == "FullConfigAgent")
    if case .literal(let instructions)? = agent.instructions {
        #expect(instructions == "You are a comprehensive test agent.")
    } else {
        #expect(Bool(false), "Instructions should be literal")
    }
    #expect(agent.tools.count == 2)
    #expect(agent.tools[0].name == "echo")
    #expect(agent.tools[1].name == "reverse")
    #expect(agent.inputGuardrails.count == 1)
    #expect(agent.modelSettings.modelName == "test-model")
    #expect(agent.modelSettings.temperature == 0.7)
    #expect(agent.modelSettings.topP == 0.9)
    #expect(agent.modelSettings.maxTokens == 1000)
}

@Test func testAgentMethodChaining() async throws {
    // Create tools
    let tool1 = Tool<Void>(
        name: "echo",
        description: "Echoes the input",
        execute: { params, _ in
            return params["text"] as? String ?? "No text provided"
        }
    )

    let tool2 = Tool<Void>(
        name: "reverse",
        description: "Reverses the input",
        execute: { params, _ in
            let text = params["text"] as? String ?? ""
            return String(text.reversed())
        }
    )

    // Create guardrails
    let inputGuardrail = InputLengthGuardrail(maxLength: 100)

    // Create agent with method chaining
    let agent = Agent<Void>(name: "ChainedAgent", instructions: "You are a method-chained agent.")
        .addTool(tool1)
        .addTool(tool2)
        .addInputGuardrail(inputGuardrail)

    #expect(agent.name == "ChainedAgent")
    #expect(agent.tools.count == 2)
    #expect(agent.inputGuardrails.count == 1)
}

@Test func testAgentClone() async throws {
    // Create initial agent
    let originalAgent = Agent<Void>(
        name: "OriginalAgent",
        instructions: "You are the original agent."
    ).addTool(Tool<Void>(
        name: "echo",
        description: "Echoes the input",
        execute: { params, _ in
            return params["text"] as? String ?? ""
        }
    ))

    // Clone the agent
    let clonedAgent = originalAgent.clone()

    // Verify the clone has the same properties
    #expect(clonedAgent.name == originalAgent.name)
    switch (clonedAgent.instructions, originalAgent.instructions) {
    case (.literal(let lhs)?, .literal(let rhs)?):
        #expect(lhs == rhs)
    case (.none, .none):
        break
    default:
        #expect(Bool(false), "Instructions mismatch")
    }
    #expect(clonedAgent.tools.count == originalAgent.tools.count)
    #expect(clonedAgent.tools[0].name == originalAgent.tools[0].name)

    // Verify that modifying the clone doesn't affect the original
    clonedAgent.addTool(Tool<Void>(
        name: "newTool",
        description: "A new tool",
        execute: { _, _ in return "result" }
    ))

    #expect(clonedAgent.tools.count == 2)
    #expect(originalAgent.tools.count == 1)
}

// MARK: - Tool Tests

@Test func testToolCreation() async throws {
    // Create a simple tool
    let tool = Tool<Void>(
        name: "echo",
        description: "Echoes the input",
        parameters: [
            Tool.Parameter(
                name: "text",
                description: "The text to echo",
                type: .string
            )
        ],
        execute: { params, _ in
            return params["text"] as? String ?? "No text provided"
        }
    )

    #expect(tool.name == "echo")
    #expect(tool.description == "Echoes the input")
    #expect(tool.parameters.count == 1)
    #expect(tool.parameters[0].name == "text")
    #expect(tool.parameters[0].description == "The text to echo")
    #expect(tool.parameters[0].type == .string)
    #expect(tool.parameters[0].required == true)
}

@Test func testToolParameterTypes() async throws {
    // Create a tool with different parameter types
    let tool = Tool<Void>(
        name: "multiTypeTest",
        description: "Tests different parameter types",
        parameters: [
            Tool.Parameter(name: "stringParam", description: "A string", type: .string),
            Tool.Parameter(name: "numberParam", description: "A number", type: .number),
            Tool.Parameter(name: "boolParam", description: "A boolean", type: .boolean),
            Tool.Parameter(name: "arrayParam", description: "An array", type: .array),
            Tool.Parameter(name: "objectParam", description: "An object", type: .object),
            Tool.Parameter(name: "optionalParam", description: "Optional", type: .string, required: false)
        ],
        execute: { _, _ in return "result" }
    )

    #expect(tool.parameters.count == 6)
    #expect(tool.parameters[0].type.jsonType == "string")
    #expect(tool.parameters[1].type.jsonType == "number")
    #expect(tool.parameters[2].type.jsonType == "boolean")
    #expect(tool.parameters[3].type.jsonType == "array")
    #expect(tool.parameters[4].type.jsonType == "object")
    #expect(tool.parameters[5].required == false)
}

@Test func testToolExecution() async throws {
    // Create a tool that performs an operation
    let calculator = Tool<Void>(
        name: "add",
        description: "Adds two numbers",
        parameters: [
            Tool.Parameter(name: "paramA", description: "First number", type: .number),
            Tool.Parameter(name: "paramB", description: "Second number", type: .number)
        ],
        execute: { params, _ in
            // Integer numbers might be parsed as different numeric types
            // We convert everything to Int for consistency
            if let paramA = params["paramA"] as? Int, let paramB = params["paramB"] as? Int {
                return paramA + paramB
            } else if let paramA = params["paramA"] as? Double, let paramB = params["paramB"] as? Double {
                return Int(paramA + paramB)
            } else {
                return 0
            }
        }
    )

    // Execute the tool
    let runContext = RunContext(value: ())
    let result = try await calculator.invoke(parameters: ["paramA": 5, "paramB": 3], runContext: runContext)

    #expect(result as? Int == 8)
}

@Test func testAddingToolToAgent() async throws {
    // Create a simple agent
    let agent = Agent<Void>(
        name: "TestAgent",
        instructions: "You are a helpful assistant."
    )

    // Create a simple tool
    let tool = Tool<Void>(
        name: "echo",
        description: "Echoes the input",
        execute: { params, _ in
            return params["text"] as? String ?? "No text provided"
        }
    )

    // Add tool to agent
    let updatedAgent = agent.addTool(tool)

    #expect(updatedAgent.tools.count == 1)
    #expect(updatedAgent.tools[0].name == "echo")
}

@Test func testAddingMultipleToolsToAgent() async throws {
    // Create a simple agent
    let agent = Agent<Void>(
        name: "TestAgent",
        instructions: "You are a helpful assistant."
    )

    // Create tools
    let tool1 = Tool<Void>(name: "tool1", description: "First tool", execute: { _, _ in return "1" })
    let tool2 = Tool<Void>(name: "tool2", description: "Second tool", execute: { _, _ in return "2" })
    let tool3 = Tool<Void>(name: "tool3", description: "Third tool", execute: { _, _ in return "3" })

    // Add multiple tools at once
    let updatedAgent = agent.addTools([tool1, tool2, tool3])

    #expect(updatedAgent.tools.count == 3)
    #expect(updatedAgent.tools[0].name == "tool1")
    #expect(updatedAgent.tools[1].name == "tool2")
    #expect(updatedAgent.tools[2].name == "tool3")
}

@Test func testTypedTool() async throws {
    // Define input and output using a simple struct
    struct AddInput: Codable {
        let inputA: Int
        let inputB: Int
    }

    // Create a tool with manual parameter handling
    let addTool = Tool<Void>(
        name: "add",
        description: "Adds two numbers",
        parameters: [
            Tool.Parameter(name: "inputA", description: "First number", type: .number),
            Tool.Parameter(name: "inputB", description: "Second number", type: .number)
        ],
        execute: { params, _ in
            // Parse the parameters manually
            guard let inputA = params["inputA"] as? Int,
                  let inputB = params["inputB"] as? Int else {
                return 0
            }
            return inputA + inputB
        }
    )

    // Execute the tool
    let result = try await addTool.invoke(parameters: ["inputA": 10, "inputB": 20], runContext: RunContext(value: ()))

    #expect(result as? Int == 30)
}

// MARK: - Guardrail Tests

@Test func testGuardrailValidation() async throws {
    // Create a simple input length guardrail
    let guardrail = InputLengthGuardrail(maxLength: 10)

    // Test valid input
    let validInput = "Hello"
    let processedInput = try guardrail.validate(validInput, context: ())
    #expect(processedInput == validInput)

    // Test invalid input
    let invalidInput = "This is a very long input that exceeds the maximum length"
    do {
        _ = try guardrail.validate(invalidInput, context: ())
        #expect(Bool(false), "Should have thrown an error")
    } catch let error as GuardrailError {
        switch error {
        case .invalidInput(let reason):
            #expect(reason.contains("Maximum length is 10"))
        default:
            #expect(Bool(false), "Wrong error type")
        }
    }
}

@Test func testRegexContentGuardrail() async throws {
    // Create a regex guardrail to block content containing "forbidden"
    let blockingGuardrail = try RegexContentGuardrail(pattern: "forbidden", blockMatches: true)

    // Test valid output (doesn't contain the blocked word)
    let validOutput = "This is an allowed message"
    let processedOutput = try blockingGuardrail.validate(validOutput, context: ())
    #expect(processedOutput == validOutput)

    // Test invalid output (contains the blocked word)
    let invalidOutput = "This message contains forbidden content"
    do {
        _ = try blockingGuardrail.validate(invalidOutput, context: ())
        #expect(Bool(false), "Should have thrown an error")
    } catch let error as GuardrailError {
        switch error {
        case .invalidOutput(let reason):
            #expect(reason.contains("blocked content"))
        default:
            #expect(Bool(false), "Wrong error type")
        }
    }

    // Create a regex guardrail to require content matching "required"
    let requiringGuardrail = try RegexContentGuardrail(pattern: "required", blockMatches: false)

    // Test valid output (contains the required word)
    let validRequiredOutput = "This message contains required content"
    let processedRequiredOutput = try requiringGuardrail.validate(validRequiredOutput, context: ())
    #expect(processedRequiredOutput == validRequiredOutput)

    // Test invalid output (doesn't contain the required word)
    let invalidRequiredOutput = "This message doesn't have the necessary text"
    do {
        _ = try requiringGuardrail.validate(invalidRequiredOutput, context: ())
        #expect(Bool(false), "Should have thrown an error")
    } catch let error as GuardrailError {
        switch error {
        case .invalidOutput(let reason):
            #expect(reason.contains("required content"))
        default:
            #expect(Bool(false), "Wrong error type")
        }
    }
}

// MARK: - Model Settings Tests

@Test func testModelSettingsCreation() async throws {
    // Create model settings with all parameters
    let settings = ModelSettings(
        modelName: "test-model",
        temperature: 0.8,
        topP: 0.95,
        maxTokens: 2000,
        responseFormat: .json,
        seed: 12345,
        additionalParameters: ["custom": "value"]
    )

    #expect(settings.modelName == "test-model")
    #expect(settings.temperature == 0.8)
    #expect(settings.topP == 0.95)
    #expect(settings.maxTokens == 2000)
    #expect(settings.responseFormat == .json)
    #expect(settings.seed == 12345)
    #expect(settings.additionalParameters["custom"] == "value")
}

@Test func testDefaultModelSettings() async throws {
    // Create model settings with defaults
    let settings = ModelSettings()

    #expect(settings.modelName == "gpt-4.1")
    #expect(settings.temperature == nil)
    #expect(settings.topP == nil)
    #expect(settings.maxTokens == nil)
    #expect(settings.responseFormat == nil)
    #expect(settings.seed == nil)
    #expect(settings.additionalParameters.isEmpty)
}

@Test func testResponseFormatJsonValue() async throws {
    // Test JSON value for text response format
    let textFormat = ModelSettings.ResponseFormat.text
    #expect(textFormat.jsonValue == "text")

    // Test JSON value for JSON response format
    let jsonFormat = ModelSettings.ResponseFormat.json
    #expect(jsonFormat.jsonValue == "json_object")
}

@Test func testUpdateModelSettings() async throws {
    // Create initial settings
    var settings = ModelSettings(modelName: "initial-model", temperature: 0.7)

    // Update settings
    settings.modelName = "updated-model"
    settings.temperature = 0.9
    settings.maxTokens = 500

    #expect(settings.modelName == "updated-model")
    #expect(settings.temperature == 0.9)
    #expect(settings.maxTokens == 500)
}
