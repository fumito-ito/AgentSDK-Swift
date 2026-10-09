# Porting Decisions

Swift-specific deviations from openai-agents-python. Each entry records what differs from upstream, why, and how to apply it during future syncs. Append new entries; do not rewrite history except to mark an entry as superseded.

## D1. Swift targets follow upstream subpackages, with core subpackages kept in one target

- **Decision:** `Agents` contains the upstream core package plus `models`, `tracing`, `memory`, `handoffs`, `run_internal`, and `util` as directories. `mcp`, `realtime`, `voice`, and `testing` become the separate targets `AgentsMCP`, `AgentsRealtime`, `AgentsVoice`, and `AgentsTesting`.
- **Why:** The core subpackages share runner internals that Swift `internal` access cannot cross module boundaries without widening visibility. The separated subpackages are optional features; separate targets keep their dependencies (for example the MCP Swift SDK) opt-in.
- **Apply:** Sibling targets reach core internals through `package` access, never `public`.

## D2. Sandbox, extensions, and third-party provider integrations are out of scope

- **Decision:** `agents.sandbox`, `agents.extensions`, and modules that only adapt third-party Python libraries are not ported.
- **Why:** They wrap Python-only ecosystems (Docker/E2B/Modal sandboxes, LiteLLM, SQLAlchemy, Redis clients) with no Swift counterpart worth maintaining.
- **Apply:** Record these files as `out_of_scope` in `porting/manifest.json`. Core code paths that reference them (for example sandbox agent preparation) are omitted, and the omission is noted at the call site.
- **Exception:** `agents.extensions.handoff_prompt` and `agents.extensions.handoff_filters` are pure SDK code with no third-party dependency, so they are ported into `Sources/Agents/Extensions/`.

## D3. No third-party OpenAI SDK; JSON-backed wire types

- **Decision:** OpenAI request/response types are implemented in `Sources/Agents/OpenAI/` as JSON-backed views: each type stores a `JSONObject` and exposes typed accessors. Input items are plain `JSONObject` values, matching upstream's `TypedDict` usage.
- **Why:** Upstream relies on the `openai` Python package. Swift OpenAI SDKs lag the API and do not round-trip unknown fields. JSON-backed views keep new server fields intact (upstream's `model_dump(exclude_unset=True)` semantics) and make test fixtures dictionary literals.

## D4. `TContext` is type-erased to `any Sendable`

- **Decision:** `Agent`, `RunContextWrapper`, `FunctionTool`, `Handoff`, guardrails, hooks, run items, results, and errors are not generic. The user context is stored as `any Sendable` and read with `RunContextWrapper.context(as:)`. Use a class for mutable contexts so that tool mutations are visible to the caller, as in Python.
- **Why:** Upstream annotates `TContext` but stores `Agent[Any]` in run items, errors, results, guardrail results, and run state. A Swift generic would have to be threaded through every one of those types and through MCP and Realtime, and erased again wherever upstream mixes agents. Type erasure mirrors upstream's runtime model and keeps future syncs mechanical.
- **Apply:** Do not add generic `Context` parameters to ported types. Offer typed convenience APIs (for example `context(as:)`) instead.
- **Supersedes:** The initial version of this entry, which made these types generic.

## D7. Exceptions are classes in a hierarchy rooted at `AgentsException`

- **Decision:** Upstream exceptions are Swift classes: `open class AgentsException: Error` with subclasses of the same names (`MaxTurnsExceeded`, `ModelBehaviorError`, `UserError`, ...).
- **Why:** Upstream catches by base class and attaches `run_data` after construction. A class hierarchy supports `catch let error as AgentsException` and mutable `runData` without rebuilding error values.

## D5. Concurrency and context propagation

- **Decision:** `contextvars` → `@TaskLocal`; `asyncio.create_task`/`gather` → structured concurrency; async generators → `AsyncThrowingStream`; cancellation → Swift task cancellation.
- **Why:** These are the native Swift equivalents and preserve upstream ordering guarantees when used with `TaskGroup` result ordering made explicit.

## D6. Function tool schemas come from `Decodable` types instead of signature introspection

- **Decision:** Upstream builds JSON schemas from Python function signatures and docstrings. Swift has no runtime signature reflection, so function tools take a `Decodable` argument type whose schema is derived from its `Decodable` conformance (or supplied explicitly). `strict_schema.py` is ported as-is because it operates on JSON.
- **Why:** This is the closest Swift equivalent that keeps schema and decoding in one source of truth.
- **Apply:** Upstream tests for docstring parsing and `inspect.signature` handling are `skipped`.

## D8. Current spans and traces are scoped, not set and reset

- **Decision:** `Scope.currentSpan` / `Scope.currentTrace` are `@TaskLocal` values. Spans and traces become current through scoped APIs (`span.withCurrent { }`, `span.run { }`, `trace.run { }`) instead of upstream's `start(mark_as_current=True)` / `finish(reset_current=True)` pairs. Long-lived spans (the runner's task and agent spans) are started and finished explicitly, and each unit of work runs inside `withCurrent`.
- **Why:** Swift task-locals can only be bound for the duration of a closure. Scoped binding also makes it impossible to leak a current span past its owner, which upstream guards against with tokens.
- **Apply:** When porting code that marks a span current, wrap the code that should observe it in `withCurrent`. Child tasks inherit the binding automatically; code that crosses into unstructured tasks must capture and rebind it.

## D9. Deeply recursive JSON processing runs on a large stack

- **Decision:** `ensureStrictJSONSchema` converts schemas that are nested more than a few levels, or that use `$defs`/`definitions`, on a dedicated thread with a 64 MiB stack (`runOnLargeStack`). Tests that build 1,000-level fixtures also run on a large stack.
- **Why:** Upstream relies on Python's recursion limit. Swift concurrency threads have 512 KiB stacks, and nested `JSONValue` trees are released recursively, so schemas inside upstream's limits (depth 100) overflowed the stack.
- **Apply:** Use `runOnLargeStack` for other recursive JSON walkers that accept untrusted depth.

## D10. Swift file names are unique per target

- **Decision:** When two upstream modules share a basename within one Swift target, the nested one gets a prefixed name: `run_internal/items.py` → `RunInternal/RunInternalItems.swift`.
- **Why:** Swift Package Manager rejects two sources with the same basename in one target.

## D11. Session input callbacks are matched by content

- **Decision:** After a `sessionInputCallback` combines history and new input, the runner decides which returned items are new (and must be persisted) by content: each returned item first consumes a matching history occurrence, then a matching new-input occurrence; anything else is new.
- **Why:** Upstream matches by object identity first, including items the callback mutated in place inside the history list it received. Swift session items are values, so identity and in-place mutation are not observable.
- **Apply:** Upstream tests that depend on identity (repeated equal history objects, history items moved into the new-input list, in-place history replacement) are skipped.
