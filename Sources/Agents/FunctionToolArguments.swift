import Foundation

// Port of src/agents/_function_tool_arguments.py: per-invocation preparation and conservative
// approval inspection.
//
// Swift: upstream inspects the generated pydantic model and refuses to validate before approval
// when validation could run application code. Swift tools decode a `Codable` arguments type, and
// a synthesized conformance cannot be told apart from a hand-written one, so decoding is treated
// as data-only. Arguments are unchanged when the decoded value re-encodes to the raw arguments;
// a type that is only `Decodable` cannot be compared and always requires approval
// (porting/decisions.md D18).

/// Identifies the preparation function that produced prepared arguments (upstream compares the
/// `_prepare_arguments` closure by identity).
final class PreparedArgumentsOwner: Sendable {}

/// Identifies the invoker a function tool had when an approval was evaluated (upstream compares
/// `on_invoke_tool` by identity). A new identity is assigned whenever `onInvokeTool` changes.
final class FunctionToolInvokerIdentity: Sendable {}

/// Arguments decoded once for a single invocation, reused by the invoker after approval.
struct PreparedFunctionArguments: Sendable {
    var owner: PreparedArgumentsOwner
    var arguments: String
    /// The decoded arguments, or `nil` when the arguments could not be inspected before approval.
    var value: (any Sendable)?
    var unchanged = false
}

/// The approval outcome for one function invocation, with the preparation it may reuse.
struct FunctionToolApproval: Sendable {
    enum Outcome: Sendable {
        case invoke
        case requireApproval
    }

    var outcome: Outcome
    var tool: FunctionTool
    var invoker: FunctionToolInvokerIdentity
    var arguments: String
    var prepared: PreparedFunctionArguments?

    init(_ outcome: Outcome, tool: FunctionTool, arguments: String, prepared: PreparedFunctionArguments? = nil) {
        self.outcome = outcome
        self.tool = tool
        invoker = tool.invokerIdentity
        self.arguments = arguments
        self.prepared = prepared
    }

    func checkInvocation(_ tool: FunctionTool, arguments: String) throws {
        if self.tool !== tool || invoker !== tool.invokerIdentity || self.arguments != arguments {
            throw UserError("Function tool invocation changed during approval evaluation.")
        }
    }
}

/// Returns whether `value` encodes to exactly the raw arguments, without comparing object key
/// order (upstream `arguments_unchanged`).
///
/// Swift: upstream ignores key order only for the root parameter object. Swift encoders do not
/// preserve dictionary order, so nested key order is not compared either.
func argumentsUnchanged(_ value: any Sendable, raw: JSONObject) -> Bool {
    guard let encodable = value as? any Encodable, let encoded = try? JSONValue(encoding: encodable) else {
        return false
    }
    return sameJSON(encoded, .object(raw))
}

/// Compares JSON values by type and value: `1` and `1.0` differ, as in upstream `_same_json`.
private func sameJSON(_ left: JSONValue, _ right: JSONValue) -> Bool {
    switch (left, right) {
    case (.null, .null):
        return true
    case let (.bool(left), .bool(right)):
        return left == right
    case let (.int(left), .int(right)):
        return left == right
    case let (.double(left), .double(right)):
        return left.isFinite && left.bitPattern == right.bitPattern
    case let (.string(left), .string(right)):
        return left == right
    case let (.array(left), .array(right)):
        return left.count == right.count && zip(left, right).allSatisfy { sameJSON($0, $1) }
    case let (.object(left), .object(right)):
        return Set(left.keys) == Set(right.keys) && left.keys.allSatisfy { key in
            guard let leftValue = left[key], let rightValue = right[key] else { return false }
            return sameJSON(leftValue, rightValue)
        }
    default:
        return false
    }
}
