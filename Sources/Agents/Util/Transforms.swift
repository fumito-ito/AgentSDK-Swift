import Foundation

/// Converts a name into a valid function-calling identifier (`[a-z0-9_]`), warning when the name
/// had to change.
public func transformStringFunctionStyle(_ name: String, warnOnWhitespace: Bool = true) -> String {
    let whitespaceNormalized = String(name.unicodeScalars.map { CharacterSet.whitespacesAndNewlines.contains($0) ? "_" : Character($0) })
    let transformed = String(whitespaceNormalized.unicodeScalars.map { scalar -> Character in
        let isASCIIAlnum = scalar.isASCII && (CharacterSet.alphanumerics.contains(scalar))
        return isASCIIAlnum || scalar == "_" ? Character(scalar) : "_"
    })
    let finalName = transformed.lowercased()
    if transformed != name, warnOnWhitespace || transformed != whitespaceNormalized {
        logger.warning(
            "Tool name '\(name)' contains invalid characters for function calling and has been transformed to "
                + "'\(finalName)'. Please use only letters, digits, and underscores to avoid potential naming conflicts."
        )
    }
    return finalName
}
