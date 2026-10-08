#!/usr/bin/env python3
"""Maintain porting/manifest.json, the map from upstream files to Swift files.

The manifest records, for every Python file under ``src/agents`` and ``tests`` in the
tracked upstream release, which Swift files implement it and its porting status.

Statuses:
    ported        The Swift files implement the upstream file at the tracked version.
    partial       Some behavior is ported; ``notes`` lists what is missing.
    pending       In scope but not yet ported.
    skipped       In scope, intentionally not ported (Python-specific); ``notes`` says why.
    out_of_scope  The upstream area is excluded from the port (see porting/decisions.md D2).

Commands:
    sync      Add new upstream files, drop removed ones, classify new files by rule.
    report    Print status counts, optionally filtered by path prefix.
    affected  Print manifest entries for upstream files changed between two releases.
    set       Update the status/swift/notes of one or more entries.
    suggest   Print the conventional Swift path for an upstream file.

Only the Python standard library is used so the script runs anywhere.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "porting" / "manifest.json"
DEFAULT_UPSTREAM = ROOT / ".vendor" / "openai-agents-python"
STATUSES = ("ported", "partial", "pending", "skipped", "out_of_scope")

# Rules applied in order to newly discovered upstream files. The first match wins.
# Each rule is (glob, status, note).
CLASSIFICATION_RULES: list[tuple[str, str, str]] = [
    ("src/agents/sandbox/*", "out_of_scope", "Sandbox runtime is out of scope (D2)."),
    ("src/agents/testing/sandbox.py", "out_of_scope", "Sandbox runtime is out of scope (D2)."),
    ("src/agents/extensions/handoff_filters.py", "pending", ""),
    ("src/agents/extensions/handoff_prompt.py", "pending", ""),
    ("src/agents/extensions/__init__.py", "skipped", "Package marker only."),
    ("src/agents/extensions/*", "out_of_scope", "Extensions wrap Python-only libraries (D2)."),
    ("src/agents/voice/imports.py", "skipped", "Python optional-dependency import guard."),
    ("src/agents/_httpx_compat.py", "skipped", "httpx compatibility shim."),
    ("src/agents/_debug.py", "skipped", "Python environment debug flags; see Logger."),
    ("src/agents/version.py", "pending", ""),
    ("src/agents/*/__init__.py", "pending", "Re-exports; map to the module's public surface."),
    ("tests/sandbox/*", "out_of_scope", "Sandbox runtime is out of scope (D2)."),
    ("tests/extensions/*", "out_of_scope", "Extensions are out of scope (D2)."),
    ("tests/fastapi/*", "out_of_scope", "FastAPI integration example."),
    ("tests/docs/*", "skipped", "Python docs tooling."),
    ("tests/fixtures/*", "pending", "Fixtures consumed by ported tests."),
    ("tests/*/__init__.py", "skipped", "Package marker only."),
    ("tests/__init__.py", "skipped", "Package marker only."),
    ("tests/conftest.py", "skipped", "pytest fixtures; equivalent setup lives in Swift test helpers."),
    ("tests/test_release_*.py", "skipped", "Python release automation."),
    ("tests/test_released_api_contract.py", "skipped", "Python released-API manifest."),
    ("tests/test_update_rclone_pin.py", "out_of_scope", "Sandbox tooling."),
    ("tests/test_scripted_sandbox.py", "out_of_scope", "Sandbox runtime is out of scope (D2)."),
    ("tests/test_*_example.py", "skipped", "Exercises Python example applications."),
    ("tests/test_example_workflows.py", "skipped", "Exercises Python example applications."),
    ("tests/test_live_example.py", "skipped", "Exercises Python example applications."),
    ("tests/test_run_examples_script.py", "skipped", "Python examples runner script."),
    ("tests/test_shell_examples.py", "skipped", "Exercises Python example applications."),
    ("tests/test_usaspending_setup_db.py", "skipped", "Exercises Python example applications."),
    ("tests/test_financial_research_budget.py", "skipped", "Exercises Python example applications."),
    ("tests/test_repository_workflow_interfaces.py", "skipped", "Python repository CI workflows."),
    ("tests/test_code_change_verification_*.py", "skipped", "Python repository agent skills."),
    ("tests/test_run_serial_tests.py", "skipped", "pytest serial runner."),
    ("tests/test_asyncio_*.py", "skipped", "Python asyncio helper internals."),
    ("tests/test_test_environment.py", "skipped", "pytest environment checks."),
    ("tests/test_visualization.py", "out_of_scope", "Graphviz visualization extension."),
    ("tests/realtime/test_twilio_*.py", "skipped", "Exercises Python example applications."),
    ("tests/realtime/test_app_server_*.py", "skipped", "Exercises Python example applications."),
    ("tests/models/test_litellm_*.py", "out_of_scope", "LiteLLM extension."),
    ("tests/models/test_any_llm_model.py", "out_of_scope", "any-llm extension."),
    ("tests/memory/test_sqlalchemy_*.py", "out_of_scope", "SQLAlchemy extension."),
    ("*", "pending", ""),
]

TOP_LEVEL_TARGETS = {
    "mcp": ("Sources/AgentsMCP", "Tests/AgentsMCPTests"),
    "realtime": ("Sources/AgentsRealtime", "Tests/AgentsRealtimeTests"),
    "voice": ("Sources/AgentsVoice", "Tests/AgentsVoiceTests"),
    "testing": ("Sources/AgentsTesting", "Tests/AgentsTestingTests"),
}


def pascal(name: str) -> str:
    """Convert ``run_internal`` or ``_tool_identity`` to ``RunInternal`` / ``ToolIdentity``."""
    parts = [p for p in re.split(r"[_\-]", name) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


def suggest_swift_path(upstream_path: str) -> str:
    """Return the conventional Swift path for an upstream file (see AGENTS.md)."""
    path = Path(upstream_path)
    if path.parts[:2] == ("src", "agents"):
        rel = list(path.parts[2:])
        base_src, _ = "Sources/Agents", None
        if rel and rel[0] in TOP_LEVEL_TARGETS and len(rel) > 1:
            base_src = TOP_LEVEL_TARGETS[rel[0]][0]
            rel = rel[1:]
        dirs = [pascal(d) for d in rel[:-1]]
        stem = path.stem
        if stem == "__init__":
            stem = dirs[-1] if dirs else "Agents"
        return "/".join([base_src, *dirs, pascal(stem) + ".swift"])
    if path.parts[0] == "tests":
        rel = list(path.parts[1:])
        base = "Tests/AgentsTests"
        if rel and rel[0] in TOP_LEVEL_TARGETS and len(rel) > 1:
            base = TOP_LEVEL_TARGETS[rel[0]][1]
            rel = rel[1:]
        dirs = [pascal(d) for d in rel[:-1]]
        stem = path.stem
        if path.suffix != ".py":
            return "/".join([base, *dirs, path.name])
        if stem.startswith("test_"):
            stem = stem[len("test_") :]
            return "/".join([base, *dirs, pascal(stem) + "Tests.swift"])
        return "/".join([base, *dirs, pascal(stem) + ".swift"])
    return upstream_path


def classify(upstream_path: str) -> tuple[str, str]:
    for pattern, status, note in CLASSIFICATION_RULES:
        if fnmatch.fnmatch(upstream_path, pattern):
            return status, note
    return "pending", ""


def list_upstream_files(upstream: Path) -> list[str]:
    files: list[str] = []
    for top in ("src/agents", "tests"):
        for p in sorted((upstream / top).rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            rel = p.relative_to(upstream).as_posix()
            if p.suffix == ".py" or rel.startswith("tests/fixtures/"):
                files.append(rel)
    return files


def load() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"upstream": {"repo": "openai/openai-agents-python", "version": None}, "files": {}}


def save(manifest: dict) -> None:
    manifest["files"] = dict(sorted(manifest["files"].items()))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def upstream_version(upstream: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(upstream), "describe", "--tags", "--exact-match"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().removeprefix("v")


def cmd_sync(args: argparse.Namespace) -> int:
    upstream = Path(args.upstream)
    manifest = load()
    files = manifest["files"]
    current = set(list_upstream_files(upstream))
    added = sorted(current - files.keys())
    removed = sorted(files.keys() - current)
    for path in added:
        status, note = classify(path)
        files[path] = {"status": status, "swift": [], "notes": note}
    for path in removed:
        del files[path]
    manifest["upstream"]["version"] = upstream_version(upstream)
    save(manifest)
    print(f"added {len(added)} entries, removed {len(removed)} entries")
    for path in added:
        print(f"  + {path} [{files[path]['status']}]")
    for path in removed:
        print(f"  - {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    manifest = load()
    counts: dict[str, int] = dict.fromkeys(STATUSES, 0)
    rows = []
    for path, entry in manifest["files"].items():
        if args.prefix and not path.startswith(args.prefix):
            continue
        counts[entry["status"]] += 1
        if args.status and entry["status"] in args.status:
            rows.append((path, entry))
    print(f"upstream version: {manifest['upstream']['version']}")
    for status in STATUSES:
        print(f"  {status:<13} {counts[status]}")
    for path, entry in rows:
        swift = ", ".join(entry["swift"]) or "-"
        note = f"  # {entry['notes']}" if entry["notes"] else ""
        print(f"{entry['status']:<13} {path} -> {swift}{note}")
    return 0


def cmd_affected(args: argparse.Namespace) -> int:
    upstream = Path(args.upstream)
    from_tag, to_tag = f"v{args.from_version.removeprefix('v')}", f"v{args.to_version.removeprefix('v')}"
    for tag in (from_tag, to_tag):
        subprocess.run(
            ["git", "-C", str(upstream), "fetch", "--quiet", "--depth", "1", "origin", f"refs/tags/{tag}:refs/tags/{tag}"],
            check=True,
        )
    diff = subprocess.run(
        ["git", "-C", str(upstream), "diff", "--name-status", "--no-renames", from_tag, to_tag, "--", "src/agents", "tests"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    manifest = load()["files"]
    result = []
    for line in diff.splitlines():
        change, path = line.split("\t", 1)
        entry = manifest.get(path)
        status, note = (entry["status"], entry["notes"]) if entry else classify(path)
        swift = entry["swift"] if entry and entry["swift"] else [suggest_swift_path(path)]
        result.append({"change": change, "path": path, "status": status, "swift": swift, "notes": note})
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for row in result:
            if row["status"] == "out_of_scope" and not args.all:
                continue
            print(f"{row['change']} {row['path']} [{row['status']}] -> {', '.join(row['swift'])}")
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    manifest = load()
    for path in args.paths:
        if path not in manifest["files"]:
            print(f"unknown upstream path: {path}", file=sys.stderr)
            return 1
        entry = manifest["files"][path]
        if args.status:
            entry["status"] = args.status
        if args.swift is not None:
            entry["swift"] = args.swift
        if args.notes is not None:
            entry["notes"] = args.notes
    save(manifest)
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    for path in args.paths:
        print(suggest_swift_path(path))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM), help="path to the upstream checkout")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sync").set_defaults(func=cmd_sync)

    report = sub.add_parser("report")
    report.add_argument("--prefix", default="")
    report.add_argument("--status", nargs="*", choices=STATUSES)
    report.set_defaults(func=cmd_report)

    affected = sub.add_parser("affected")
    affected.add_argument("from_version")
    affected.add_argument("to_version")
    affected.add_argument("--json", action="store_true")
    affected.add_argument("--all", action="store_true", help="include out_of_scope files")
    affected.set_defaults(func=cmd_affected)

    set_cmd = sub.add_parser("set")
    set_cmd.add_argument("paths", nargs="+")
    set_cmd.add_argument("--status", choices=STATUSES)
    set_cmd.add_argument("--swift", nargs="*")
    set_cmd.add_argument("--notes")
    set_cmd.set_defaults(func=cmd_set)

    suggest = sub.add_parser("suggest")
    suggest.add_argument("paths", nargs="+")
    suggest.set_defaults(func=cmd_suggest)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
