import Testing
@testable import Agents

// Ported from tests/test_apply_diff_helpers.py.
@Suite struct ApplyDiffHelpersTests {
    @Test func testNormalizeDiffLinesDropsTrailingBlank() {
        #expect(ApplyDiff.normalizeDiffLines("a\nb\n") == ["a", "b"])
    }

    @Test func testIsDoneTrueWhenIndexOutOfRange() {
        let state = ApplyDiff.ParserState(lines: ["line"], index: 1)
        #expect(ApplyDiff.isDone(state, []))
    }

    @Test func testReadStrReturnsEmptyWhenMissingPrefix() {
        let state = ApplyDiff.ParserState(lines: ["value"], index: 0)
        #expect(ApplyDiff.readStr(state, "nomatch") == "")
        #expect(state.index == 0)
    }

    @Test func testReadSectionReturnsEOFFlag() throws {
        #expect(try ApplyDiff.readSection(["*** End of File"], 0).eof)
    }

    @Test func testReadSectionRaisesOnInvalidMarker() {
        expectError(ValueError.self) { _ = try ApplyDiff.readSection(["*** Bad Marker"], 0) }
    }

    @Test func testReadSectionRaisesWhenEmptySegment() {
        expectError(ValueError.self) { _ = try ApplyDiff.readSection([], 0) }
    }

    @Test func testFindContextEOFFallbacks() {
        let match = ApplyDiff.findContext(["one"], ["missing"], start: 0, eof: true)
        #expect(match.newIndex == -1)
        #expect(match.fuzz >= 10_000)
    }

    @Test func testFindContextEOFIgnoresTrailingEmptySplitElement() {
        let match = ApplyDiff.findContext(["a", "b", ""], [], start: 0, eof: true)
        #expect(match.newIndex == 2)
        #expect(match.fuzz == 0)

        let lastFoo = ApplyDiff.findContext(["x", "foo", "y", "foo", ""], ["foo"], start: 0, eof: true)
        #expect(lastFoo.newIndex == 3)
        #expect(lastFoo.fuzz == 0)
    }

    @Test func testFindContextCoreStrippedMatches() {
        let match = ApplyDiff.findContextCore([" line "], ["line"], start: 0)
        #expect(match.newIndex == 0)
        #expect(match.fuzz == 100)
    }

    @Test func testApplyChunksRejectsBadChunks() {
        expectError(ValueError.self) {
            _ = try ApplyDiff.applyChunks("abc", [ApplyDiff.Chunk(origIndex: 10, delLines: [], insLines: [])], newline: "\n")
        }
        expectError(ValueError.self) {
            _ = try ApplyDiff.applyChunks(
                "abc",
                [ApplyDiff.Chunk(origIndex: 0, delLines: ["a"], insLines: []), ApplyDiff.Chunk(origIndex: 0, delLines: ["b"], insLines: [])],
                newline: "\n"
            )
        }
    }
}
