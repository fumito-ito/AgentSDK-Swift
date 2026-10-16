import Testing
import Agents

// Ported from tests/test_apply_diff.py.
@Suite struct ApplyDiffTests {
    private func lines(_ values: [String], separator: String = "\n") -> String {
        values.joined(separator: separator)
    }

    @Test func testApplyDiffWithFloatingHunkAddsLines() throws {
        let diff = lines(["@@", "+hello", "+world"])  // no trailing newline
        #expect(try applyDiff("", diff) == "hello\nworld\n")
    }

    @Test func testApplyDiffWithEmptyInputAndCRLFDiffPreservesCRLF() throws {
        let diff = lines(["@@", "+hello", "+world"], separator: "\r\n")
        #expect(try applyDiff("", diff) == "hello\r\nworld\r\n")
    }

    @Test func testApplyDiffCreateModeRequiresPlusPrefix() {
        expectError(ValueError.self) { _ = try applyDiff("", "plain line", mode: .create) }
    }

    @Test func testApplyDiffCreateModePreservesTrailingNewline() throws {
        #expect(try applyDiff("", lines(["+hello", "+world", "+"]), mode: .create) == "hello\nworld\n")
    }

    @Test func testApplyDiffAppliesContextualReplacement() throws {
        let diff = lines(["@@ line1", "-line2", "+updated", " line3"])
        #expect(try applyDiff("line1\nline2\nline3\n", diff) == "line1\nupdated\nline3\n")
    }

    @Test func testApplyDiffAppliesStackedAnchorsFromTheToolDescription() throws {
        let input = lines(["class BaseClass", "    def search():", "        pass", "", "class Subclass", "    def search():", "        pass"]) + "\n"
        let diff = lines([
            "@@ class BaseClass", "@@     def search():", "-        pass", "+        raise NotImplementedError()", "",
            "@@ class Subclass", "@@     def search():", "-        pass", "+        raise NotImplementedError()",
        ])

        let expected = lines([
            "class BaseClass", "    def search():", "        raise NotImplementedError()", "",
            "class Subclass", "    def search():", "        raise NotImplementedError()",
        ]) + "\n"
        #expect(try applyDiff(input, diff) == expected)
    }

    @Test func testApplyDiffReusesAPriorParentAnchorAcrossStackedHunks() throws {
        let input = lines(["class Target", "    def first():", "        pass", "", "    def second():", "        pass"]) + "\n"
        let diff = lines([
            "@@ class Target", "@@     def first():", "-        pass", "+        return 1",
            "@@ class Target", "@@     def second():", "-        pass", "+        return 2",
        ])

        let expected = lines(["class Target", "    def first():", "        return 1", "", "    def second():", "        return 2"]) + "\n"
        #expect(try applyDiff(input, diff) == expected)
    }

    @Test func testApplyDiffStackedAnchorsNarrowToTheNamedBlock() throws {
        let input = lines([
            "class First", "    def target():", "        return 0", "",
            "class Second", "    def helper():", "        pass", "", "    def target():", "        pass",
        ]) + "\n"
        let diff = lines(["@@ class Second", "@@     def target():", "-        pass", "+        return 1"])

        let expected = lines([
            "class First", "    def target():", "        return 0", "",
            "class Second", "    def helper():", "        pass", "", "    def target():", "        return 1",
        ]) + "\n"
        #expect(try applyDiff(input, diff) == expected)
    }

    @Test func testApplyDiffSingleAnchorStaysAdvisoryWhenUnmatched() throws {
        #expect(try applyDiff("a\nb\n", lines(["@@ nope", "-b", "+B"])) == "a\nB\n")
    }

    @Test func testApplyDiffRejectsPartiallyMatchedStackedAnchors() {
        let input = lines(["class Target", "    def helper():", "        pass", "", "    def desired():", "        return 1"]) + "\n"
        let diff = lines(["@@ class Target", "@@     def missing():", "-        pass", "+        return 99"])
        expectError(ValueError.self, containing: "Invalid Anchor") { _ = try applyDiff(input, diff) }
    }

    @Test func testApplyDiffRejectsStackedAnchorsWhenTheFirstIsMissing() {
        let diff = lines(["@@ class Target", "@@     def desired():", "-        pass", "+        return 99"])
        expectError(ValueError.self, containing: "Invalid Anchor") {
            _ = try applyDiff("class Wrong\n    def desired():\n        pass\n", diff)
        }
    }

    @Test func testApplyDiffRejectsAMissingAnchorFollowedByABareMarker() {
        expectError(ValueError.self, containing: "Invalid Anchor") {
            _ = try applyDiff("a\nb\n", lines(["@@ missing", "@@", "-b", "+B"]))
        }
    }

    @Test func testApplyDiffStackedAnchorsAcceptATrailingBareAnchor() throws {
        let diff = lines(["@@ class Only", "@@", "-        pass", "+        return 1"])
        #expect(try applyDiff("class Only\n    def run():\n        pass\n", diff) == "class Only\n    def run():\n        return 1\n")
    }

    @Test func testApplyDiffRaisesOnContextMismatch() {
        expectError(ValueError.self) { _ = try applyDiff("one\ntwo\n", lines(["@@ -1,2 +1,2 @@", " x", "-two", "+2"])) }
    }

    @Test func testApplyDiffWithCRLFInputAndLFDiffPreservesCRLF() throws {
        let diff = lines(["@@ line1", "-line2", "+updated", " line3"])
        #expect(try applyDiff("line1\r\nline2\r\nline3\r\n", diff) == "line1\r\nupdated\r\nline3\r\n")
    }

    @Test func testApplyDiffWithLFInputAndCRLFDiffPreservesLF() throws {
        let diff = lines(["@@ line1", "-line2", "+updated", " line3"], separator: "\r\n")
        #expect(try applyDiff("line1\nline2\nline3\n", diff) == "line1\nupdated\nline3\n")
    }

    @Test func testApplyDiffWithCRLFInputAndCRLFDiffPreservesCRLF() throws {
        let diff = lines(["@@ line1", "-line2", "+updated", " line3"], separator: "\r\n")
        #expect(try applyDiff("line1\r\nline2\r\nline3\r\n", diff) == "line1\r\nupdated\r\nline3\r\n")
    }

    @Test func testApplyDiffCreateModePreservesCRLFNewlines() throws {
        #expect(try applyDiff("", lines(["+hello", "+world", "+"], separator: "\r\n"), mode: .create) == "hello\r\nworld\r\n")
    }

    @Test func testApplyDiffEOFHunkAppendsWithoutABlankLine() throws {
        // Files that end in a newline must not grow a phantom blank line at EOF.
        #expect(try applyDiff("a\nb\n", "@@\n+c\n*** End of File") == "a\nb\nc\n")
        #expect(try applyDiff("a\nb", "@@\n+c\n*** End of File") == "a\nb\nc")
    }

    @Test func testApplyDiffEOFHunkMatchesTheLastContextOccurrence() throws {
        #expect(try applyDiff("x\nfoo\ny\nfoo\n", " foo\n+added\n*** End of File") == "x\nfoo\ny\nfoo\nadded\n")
    }

    @Test func testApplyDiffEOFHunkPreservesCRLF() throws {
        #expect(try applyDiff("a\r\nb\r\n", "@@\n+c\n*** End of File") == "a\r\nb\r\nc\r\n")
    }
}
