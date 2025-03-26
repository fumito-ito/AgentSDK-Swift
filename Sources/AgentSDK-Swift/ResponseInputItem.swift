import Foundation

/// Role of a response input item
public enum ResponseRole: String, Codable {
    /// User role
    case user
    
    /// Assistant role
    case assistant
    
    /// System role
    case system
}

/// Represents an item in the conversation history
public struct ResponseInputItem: Codable {
    /// The role of the sender
    public let role: ResponseRole
    
    /// The content of the message
    public let content: String
    
    /// Create a new response input item
    /// - Parameters:
    ///   - role: The role of the sender
    ///   - content: The content of the message
    public init(role: ResponseRole, content: String) {
        self.role = role
        self.content = content
    }
}