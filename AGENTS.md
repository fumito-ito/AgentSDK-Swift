# Contributor Guide

This file is the single source of instructions for humans and coding agents working in this repository. `CLAUDE.md` is a symlink to this file.

## Project Overview

AgentSDK-Swift is a Swift port of [openai-agents-python](https://github.com/openai/openai-agents-python) ("upstream"). The goal is behavioral parity with upstream, expressed with idiomatic Swift. `UPSTREAM_VERSION` records the upstream release this port tracks, and the Swift package is released with the same version number.

## Porting Rules

These rules apply to the initial port and to every upstream catch-up.

### Tests are the specification

1. Port the upstream tests for the affected module first. The ported tests are the specification of record.
2. Then port or update the implementation until the ported tests pass.
3. Change a ported test only when the Python behavior cannot be expressed in Swift. Record the reason in a comment at the test (`// Swift: ...`) and, for anything structural, in `porting/decisions.md`.
4. Python tests that exercise purely Python-specific machinery (pytest fixtures, import paths, `asyncio` internals, pydantic, docstring parsing, packaging, release automation) are not ported. Record them as `skipped` with a reason in `porting/manifest.json`.

### Module layout mirrors upstream

- Mirror the upstream package layout by default: one upstream module maps to one Swift file with the PascalCase name, in a directory that mirrors the upstream subpackage. For example, `src/agents/run_internal/tool_execution.py` maps to `Sources/Agents/RunInternal/ToolExecution.swift`, and `tests/test_agent_runner.py` maps to `Tests/AgentsTests/AgentRunnerTests.swift`.
- Deviate only when Swift has a clearly better structure (for example module boundaries for optional dependencies, access control, or a type that must be split for the compiler). Every deviation must be recorded in `porting/decisions.md`.
- Keep public type and member names recognizable from upstream: `snake_case` becomes `camelCase`, Python classes keep their names (`RunConfig`, `RunResultStreaming`, `InputGuardrailTripwireTriggered`).

### Swift targets

| Swift target | Upstream package | Notes |
|---|---|---|
| `Agents` | `agents` (core), `agents.models`, `agents.tracing`, `agents.memory`, `agents.handoffs`, `agents.run_internal`, `agents.util` | Subpackages are directories inside the target, because they share internal state with the runner. |
| `AgentsMCP` | `agents.mcp` | Separate target so the MCP Swift SDK dependency is opt-in. Core-facing MCP protocols live in `Agents`. |
| `AgentsRealtime` | `agents.realtime` | |
| `AgentsVoice` | `agents.voice` | |
| `AgentsTesting` | `agents.testing` | Scripted models used by the test suites and by SDK users. |

Upstream `agents.sandbox`, `agents.extensions`, and other provider integrations are out of scope; `porting/manifest.json` records them as `out_of_scope`. Use the `package` access level for symbols that sibling targets need but SDK users must not.

### Swift-specific foundations

- The upstream code depends on the `openai` Python package. The Swift port has no third-party OpenAI SDK; the OpenAI wire types and HTTP/WebSocket client live in `Sources/Agents/OpenAI/`. Wire types are JSON-backed views (`JSONValue` storage with typed accessors) so that unknown fields round-trip unchanged.
- Python `dict` input items map to `JSONObject` (`[String: JSONValue]`), which supports dictionary literals in tests.
- Python `TContext` is type-erased: the run context is stored as `any Sendable` and read with `RunContextWrapper.context(as:)`. Ported types are not generic over the context (see `porting/decisions.md` D4).
- Python `contextvars` map to `@TaskLocal`; `asyncio` tasks map to structured concurrency (`async let`, `TaskGroup`); async iterators map to `AsyncThrowingStream`.
- Python exceptions map to Swift error classes with the same names under `AgentsException`. Catching code uses `catch let error as MaxTurnsExceeded`.
- Networking uses `URLSession` only. Handle Linux with `#if canImport(FoundationNetworking)`; do not use APIs that are unavailable in swift-corelibs-foundation without a fallback.

## Upstream Sync Workflow

A catch-up moves the port from the version in `UPSTREAM_VERSION` to a newer upstream release.

1. `make upstream VERSION=<new>` checks out the new release under `.vendor/openai-agents-python` (git-ignored, read-only reference).
2. Compare the tracked and new releases, and map each changed upstream file to Swift files through `porting/manifest.json`.
3. For each affected module, update the ported tests first, then the implementation.
4. Update `porting/manifest.json` for every touched file, append Swift-specific choices to `porting/decisions.md`, and set `UPSTREAM_VERSION` to the new release.
5. Run `make check`. Open a pull request titled `Sync with openai-agents-python v<new>` that summarizes ported changes, skipped changes with reasons, and known gaps.

Release numbering follows upstream: after a sync PR for `v<new>` is merged, the release is tagged `<new>`.

## Commands

```bash
make bootstrap                 # install pinned tools (mise) and resolve packages
make build                     # swift build
make test                      # swift test (works with Command Line Tools only)
make test FILTER=AgentTests    # run a subset
make lint                      # SwiftLint, strict
make check                     # lint + build + test; run before every PR
make upstream [VERSION=x.y.z]  # check out an upstream release into .vendor/
```

Use `make test` rather than `swift test` directly: without Xcode, plain `swift test` cannot find the `Testing` module. Live API examples need `OPENAI_API_KEY`; never commit keys.

## Code Style

- 4-space indentation. PascalCase for types, camelCase for functions, variables, and parameters.
- Explicit access control on every declaration (`public`, `package`, `internal`, `private`).
- `///` documentation comments on public API. Port upstream docstrings when they describe behavior.
- Protocol-oriented design, generics for type safety, dependency injection through initializers.
- `async`/`await` for asynchronous work. Types crossing concurrency domains must be `Sendable`. Prefer value types and immutable data; use `final class` where upstream relies on identity (for example `Agent`), and document any `@unchecked Sendable`.
- Foundation first, minimal imports. Do not add package dependencies without recording the reason in `porting/decisions.md`.
- Errors are structured types with descriptive messages; propagate with `throws` and keep `do`/`catch` blocks narrow.
- Do not hard-wrap prose in Markdown files.

## Test Guidelines

- Use Swift Testing: `@Test`, `#expect`, `#require`, and `@Suite`. Do not use XCTest.
- One Swift test file per upstream test module, with the same base name (`test_tool_use_behavior.py` → `ToolUseBehaviorTests.swift`). Keep the upstream test function name in camelCase so tests can be traced back (`test_max_turns_exceeded` → `testMaxTurnsExceeded`).
- Follow Arrange, Act, Assert. Group with `// MARK: - Category`.
- Prefer the scripted models in `AgentsTesting` over ad-hoc fakes, mirroring upstream guidance. Keep a specialized fake only when the test needs wire-level conversion, malformed streams, or controlled concurrency, and say why in the test.
- Tests must be deterministic: wait on observable state, not wall-clock sleeps. No network access in unit tests.
- Use `@testable import` only when the upstream test also reaches into internals.

## Pull Requests and Commits

- Commit messages are concise and imperative. Keep commits focused.
- Every PR states which upstream version and modules it covers, how it was verified (`make check` output), and what was intentionally skipped.
- Never commit secrets, real customer data, or recorded API payloads with real content.
