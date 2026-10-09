import Foundation

// Port of src/agents/apply_diff.py: a utility for applying V4A diffs against text inputs.
//
// Swift: `"\r\n"` is a single `Character`, so line splitting and newline detection work on
// Unicode scalars to match Python's code-point string semantics.

/// How `applyDiff` interprets the diff.
public enum ApplyDiffMode: String, Sendable {
    /// An update diff with context hunks.
    case `default`
    /// A create-file diff containing only `+` lines.
    case create
}

/// Applies a V4A diff to the provided text.
///
/// This parser understands both the create-file syntax (only `+` prefixed lines) and the default
/// update syntax that includes context hunks.
public func applyDiff(_ input: String, _ diff: String, mode: ApplyDiffMode = .default) throws -> String {
    let newline = ApplyDiff.detectNewline(input, diff, mode: mode)
    let diffLines = ApplyDiff.normalizeDiffLines(diff)
    if mode == .create {
        return try ApplyDiff.parseCreateDiff(diffLines, newline: newline)
    }
    let normalizedInput = ApplyDiff.normalizeTextNewlines(input)
    let parsed = try ApplyDiff.parseUpdateDiff(diffLines, normalizedInput)
    return try ApplyDiff.applyChunks(normalizedInput, parsed.chunks, newline: newline)
}

/// Internal helpers for `applyDiff`, kept addressable for direct tests.
enum ApplyDiff {
    struct Chunk: Sendable, Equatable {
        var origIndex: Int
        var delLines: [String]
        var insLines: [String]
    }

    final class ParserState {
        var lines: [String]
        var index: Int
        var fuzz: Int

        init(lines: [String], index: Int = 0, fuzz: Int = 0) {
            self.lines = lines
            self.index = index
            self.fuzz = fuzz
        }
    }

    struct ParsedUpdateDiff {
        var chunks: [Chunk]
        var fuzz: Int
    }

    struct ReadSectionResult {
        var nextContext: [String]
        var sectionChunks: [Chunk]
        var endIndex: Int
        var eof: Bool
    }

    struct ContextMatch: Equatable {
        var newIndex: Int
        var fuzz: Int
    }

    static let endPatch = "*** End Patch"
    static let endFile = "*** End of File"
    static let sectionTerminators = [endPatch, "*** Update File:", "*** Delete File:", "*** Add File:"]
    static let endSectionMarkers = sectionTerminators + [endFile]

    // MARK: - Newlines

    /// Splits on `"\n"` code points, keeping empty pieces like `str.split("\n")`.
    static func splitLines(_ text: String) -> [String] {
        text.unicodeScalars.split(separator: "\n", omittingEmptySubsequences: false).map { String(String.UnicodeScalarView($0)) }
    }

    static func normalizeDiffLines(_ diff: String) -> [String] {
        var lines = splitLines(diff).map { line -> String in
            var scalars = line.unicodeScalars
            while scalars.last == "\r" { scalars.removeLast() }
            return String(scalars)
        }
        if lines.last == "" {
            lines.removeLast()
        }
        return lines
    }

    static func containsCRLF(_ text: String) -> Bool {
        var previous: Unicode.Scalar?
        for scalar in text.unicodeScalars {
            if previous == "\r", scalar == "\n" { return true }
            previous = scalar
        }
        return false
    }

    static func detectNewlineFromText(_ text: String) -> String {
        containsCRLF(text) ? "\r\n" : "\n"
    }

    static func detectNewline(_ input: String, _ diff: String, mode: ApplyDiffMode) -> String {
        // Create-file diffs don't have an input to infer newline style from. Use the diff's
        // newline style if present, otherwise default to LF.
        if mode != .create, input.unicodeScalars.contains("\n") {
            return detectNewlineFromText(input)
        }
        return detectNewlineFromText(diff)
    }

    static func normalizeTextNewlines(_ text: String) -> String {
        // Normalize CRLF to LF for parsing/matching. Newline style is restored when emitting.
        var scalars = String.UnicodeScalarView()
        var pendingCR = false
        for scalar in text.unicodeScalars {
            if pendingCR {
                if scalar != "\n" { scalars.append("\r") }
                pendingCR = false
            }
            if scalar == "\r" {
                pendingCR = true
            } else {
                scalars.append(scalar)
            }
        }
        if pendingCR { scalars.append("\r") }
        return String(scalars)
    }

    // MARK: - Parsing

    static func isDone(_ state: ParserState, _ prefixes: [String]) -> Bool {
        if state.index >= state.lines.count { return true }
        return prefixes.contains { state.lines[state.index].hasPrefix($0) }
    }

    static func readStr(_ state: ParserState, _ prefix: String) -> String {
        guard state.index < state.lines.count else { return "" }
        let current = state.lines[state.index]
        if current.hasPrefix(prefix) {
            state.index += 1
            return String(current.dropFirst(prefix.count))
        }
        return ""
    }

    static func parseCreateDiff(_ lines: [String], newline: String) throws -> String {
        let parser = ParserState(lines: lines + [endPatch])
        var output: [String] = []
        while !isDone(parser, sectionTerminators) {
            if parser.index >= parser.lines.count { break }
            let line = parser.lines[parser.index]
            parser.index += 1
            guard line.hasPrefix("+") else {
                throw ValueError("Invalid Add File Line: \(line)")
            }
            output.append(String(line.dropFirst()))
        }
        return output.joined(separator: newline)
    }

    static func parseUpdateDiff(_ lines: [String], _ input: String) throws -> ParsedUpdateDiff {
        let parser = ParserState(lines: lines + [endPatch])
        let inputLines = splitLines(input)
        var chunks: [Chunk] = []
        var cursor = 0

        while !isDone(parser, endSectionMarkers) {
            let (anchors, anchorCount) = readAnchors(parser)

            if !(anchorCount > 0 || cursor == 0) {
                let currentLine = parser.index < parser.lines.count ? parser.lines[parser.index] : ""
                throw ValueError("Invalid Line:\n\(currentLine)")
            }

            let requireAnchorMatch = anchorCount > 1
            for (index, anchor) in anchors.enumerated() {
                cursor = try advanceCursorToAnchor(
                    anchor, inputLines, cursor, parser, requireMatch: requireAnchorMatch, forceForwardSearch: index > 0
                )
            }

            let section = try readSection(parser.lines, parser.index)
            let findResult = findContext(inputLines, section.nextContext, start: cursor, eof: section.eof)
            if findResult.newIndex == -1 {
                let contextText = section.nextContext.joined(separator: "\n")
                if section.eof {
                    throw ValueError("Invalid EOF Context \(cursor):\n\(contextText)")
                }
                throw ValueError("Invalid Context \(cursor):\n\(contextText)")
            }

            cursor = findResult.newIndex + section.nextContext.count
            parser.fuzz += findResult.fuzz
            parser.index = section.endIndex

            for chunk in section.sectionChunks {
                chunks.append(Chunk(origIndex: chunk.origIndex + findResult.newIndex, delLines: chunk.delLines, insLines: chunk.insLines))
            }
        }

        return ParsedUpdateDiff(chunks: chunks, fuzz: parser.fuzz)
    }

    /// Consumes the `@@` header lines that introduce one hunk.
    ///
    /// The patch format lets a hunk carry several stacked headers, so nested code can be located
    /// when a single header plus context is still ambiguous. Returns the non-empty headers in the
    /// order they should narrow the search, plus the total number of consumed headers, including
    /// bare `@@` markers.
    static func readAnchors(_ parser: ParserState) -> (anchors: [String], count: Int) {
        var anchors: [String] = []
        var anchorCount = 0
        while true {
            let startIndex = parser.index
            let anchor = readStr(parser, "@@ ")
            var consumed = parser.index != startIndex
            if !consumed, parser.index < parser.lines.count, parser.lines[parser.index] == "@@" {
                parser.index += 1
                consumed = true
            }
            if !consumed { break }
            anchorCount += 1
            if !strip(anchor).isEmpty {
                anchors.append(anchor)
            }
        }
        return (anchors, anchorCount)
    }

    static func advanceCursorToAnchor(
        _ anchor: String,
        _ inputLines: [String],
        _ cursor: Int,
        _ parser: ParserState,
        requireMatch: Bool = false,
        forceForwardSearch: Bool = false
    ) throws -> Int {
        var cursor = cursor
        var found = false

        let hasExactMatchBeforeCursor = !forceForwardSearch && inputLines.prefix(cursor).contains(anchor)
        if hasExactMatchBeforeCursor {
            found = true
        } else if let index = inputLines.indices.dropFirst(cursor).first(where: { inputLines[$0] == anchor }) {
            cursor = index + 1
            found = true
        }

        if !found {
            let trimmedAnchor = strip(anchor)
            let hasTrimmedMatchBeforeCursor = !forceForwardSearch && inputLines.prefix(cursor).contains { strip($0) == trimmedAnchor }
            if hasTrimmedMatchBeforeCursor {
                found = true
            } else if let index = inputLines.indices.dropFirst(cursor).first(where: { strip(inputLines[$0]) == trimmedAnchor }) {
                cursor = index + 1
                parser.fuzz += 1
                found = true
            }
        }

        if requireMatch, !found {
            throw ValueError("Invalid Anchor \(cursor):\n\(anchor)")
        }
        return cursor
    }

    private enum SectionMode {
        case keep, add, delete
    }

    static func readSection(_ lines: [String], _ startIndex: Int) throws -> ReadSectionResult {
        var context: [String] = []
        var delLines: [String] = []
        var insLines: [String] = []
        var sectionChunks: [Chunk] = []
        var mode = SectionMode.keep
        var index = startIndex
        let origIndex = index

        while index < lines.count {
            let raw = lines[index]
            if raw.hasPrefix("@@") || raw.hasPrefix(endPatch) || raw.hasPrefix("*** Update File:")
                || raw.hasPrefix("*** Delete File:") || raw.hasPrefix("*** Add File:") || raw.hasPrefix(endFile) {
                break
            }
            if raw == "***" { break }
            if raw.hasPrefix("***") {
                throw ValueError("Invalid Line: \(raw)")
            }

            index += 1
            let lastMode = mode
            let line = raw.isEmpty ? " " : raw
            switch line.unicodeScalars.first {
            case "+": mode = .add
            case "-": mode = .delete
            case " ": mode = .keep
            default: throw ValueError("Invalid Line: \(line)")
            }

            let lineContent = String(String.UnicodeScalarView(line.unicodeScalars.dropFirst()))
            let switchingToContext = mode == .keep && lastMode != mode
            if switchingToContext, !delLines.isEmpty || !insLines.isEmpty {
                sectionChunks.append(Chunk(origIndex: context.count - delLines.count, delLines: delLines, insLines: insLines))
                delLines = []
                insLines = []
            }

            switch mode {
            case .delete:
                delLines.append(lineContent)
                context.append(lineContent)
            case .add:
                insLines.append(lineContent)
            case .keep:
                context.append(lineContent)
            }
        }

        if !delLines.isEmpty || !insLines.isEmpty {
            sectionChunks.append(Chunk(origIndex: context.count - delLines.count, delLines: delLines, insLines: insLines))
        }

        if index < lines.count, lines[index] == endFile {
            return ReadSectionResult(nextContext: context, sectionChunks: sectionChunks, endIndex: index + 1, eof: true)
        }

        if index == origIndex {
            let nextLine = index < lines.count ? lines[index] : ""
            throw ValueError("Nothing in this section - index=\(index) \(nextLine)")
        }

        return ReadSectionResult(nextContext: context, sectionChunks: sectionChunks, endIndex: index, eof: false)
    }

    // MARK: - Matching

    static func findContext(_ lines: [String], _ context: [String], start: Int, eof: Bool) -> ContextMatch {
        guard eof else { return findContextCore(lines, context, start: start) }
        // Splitting text that ends in a newline keeps a trailing empty element. Treat that as the
        // file terminator, not as a line, so EOF hunks append after the last real line instead of
        // after a phantom blank.
        let searchLines = eofSearchLines(lines)
        let endStart = max(0, searchLines.count - context.count)
        let endMatch = findContextCore(searchLines, context, start: endStart)
        if endMatch.newIndex != -1 {
            return endMatch
        }
        let fallback = findContextCore(searchLines, context, start: min(start, searchLines.count))
        return ContextMatch(newIndex: fallback.newIndex, fuzz: fallback.fuzz + 10_000)
    }

    static func eofSearchLines(_ lines: [String]) -> [String] {
        lines.last == "" ? Array(lines.dropLast()) : lines
    }

    static func findContextCore(_ lines: [String], _ context: [String], start: Int) -> ContextMatch {
        if context.isEmpty {
            return ContextMatch(newIndex: start, fuzz: 0)
        }
        let candidates = start < lines.count ? Array(start..<lines.count) : []
        if let index = candidates.first(where: { equalsSlice(lines, context, $0) { $0 } }) {
            return ContextMatch(newIndex: index, fuzz: 0)
        }
        if let index = candidates.first(where: { equalsSlice(lines, context, $0, rstrip) }) {
            return ContextMatch(newIndex: index, fuzz: 1)
        }
        if let index = candidates.first(where: { equalsSlice(lines, context, $0, strip) }) {
            return ContextMatch(newIndex: index, fuzz: 100)
        }
        return ContextMatch(newIndex: -1, fuzz: 0)
    }

    static func equalsSlice(_ source: [String], _ target: [String], _ start: Int, _ mapFn: (String) -> String) -> Bool {
        if start + target.count > source.count { return false }
        for (offset, targetValue) in target.enumerated() where mapFn(source[start + offset]) != mapFn(targetValue) {
            return false
        }
        return true
    }

    static func applyChunks(_ input: String, _ chunks: [Chunk], newline: String) throws -> String {
        let origLines = splitLines(input)
        var destLines: [String] = []
        var cursor = 0

        for chunk in chunks {
            if chunk.origIndex > origLines.count {
                throw ValueError("applyDiff: chunk.origIndex \(chunk.origIndex) > input length \(origLines.count)")
            }
            if cursor > chunk.origIndex {
                throw ValueError("applyDiff: overlapping chunk at \(chunk.origIndex) (cursor \(cursor))")
            }
            destLines.append(contentsOf: origLines[cursor..<chunk.origIndex])
            cursor = chunk.origIndex
            destLines.append(contentsOf: chunk.insLines)
            cursor += chunk.delLines.count
        }

        if cursor < origLines.count {
            destLines.append(contentsOf: origLines[cursor...])
        }
        return destLines.joined(separator: newline)
    }

    // MARK: - Whitespace

    static func strip(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func rstrip(_ value: String) -> String {
        var scalars = value.unicodeScalars
        while let last = scalars.last, CharacterSet.whitespacesAndNewlines.contains(last) {
            scalars.removeLast()
        }
        return String(scalars)
    }
}
