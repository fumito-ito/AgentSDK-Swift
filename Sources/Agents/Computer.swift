import Foundation

/// The computer environment (`"mac"`, `"windows"`, `"ubuntu"`, `"browser"`).
public typealias Environment = String

/// A mouse button (`"left"`, `"right"`, `"wheel"`, `"back"`, `"forward"`).
public typealias Button = String

/// The local runtime behind `ComputerTool`.
///
/// Swift: upstream has separate `Computer` and `AsyncComputer` base classes. Swift async protocol
/// requirements can be satisfied by synchronous implementations, so one protocol covers both.
public protocol Computer: AnyObject, Sendable {
    var environment: Environment? { get }
    var dimensions: (width: Int, height: Int)? { get }
    func screenshot() async throws -> String
    func click(x: Int, y: Int, button: Button) async throws
    func doubleClick(x: Int, y: Int) async throws
    func scroll(x: Int, y: Int, scrollX: Int, scrollY: Int) async throws
    func type(text: String) async throws
    func wait() async throws
    func move(x: Int, y: Int) async throws
    func keypress(keys: [String]) async throws
    func drag(path: [(x: Int, y: Int)]) async throws
}

extension Computer {
    public var environment: Environment? { nil }
    public var dimensions: (width: Int, height: Int)? { nil }
}

/// Alias kept for parity with upstream `AsyncComputer`.
public typealias AsyncComputer = Computer
