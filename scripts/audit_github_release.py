#!/usr/bin/env python3
"""Fail closed when the staged GitHub Pages release crosses its declared boundary."""

from __future__ import annotations

import subprocess
import sys


MAX_GIT_BLOB = 100 * 1024 * 1024
MAX_PAGES_TREE = 1024 * 1024 * 1024
ALLOWED_FILES = {
    ".gitattributes",
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    "DECISIONS.md",
    "HANDOFF.md",
    "HISTORY.md",
    "README.md",
    "TASKS.md",
    "pyproject.toml",
    "uv.lock",
}
ALLOWED_PREFIXES = (
    ".claude/agent-memory/",
    ".codex/knowledge/",
    "data/",
    "docs/",
    "notes/",
    "scripts/",
    "src/",
    "tests/",
)
FORBIDDEN_FILES = {".claude/stats.json"}
FORBIDDEN_PREFIXES = (".codex/operators/", "data/annotation_ledger/")


def git(*arguments: str, text: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=text
    )
    return completed.stdout


def nul_paths(payload: bytes) -> list[str]:
    return [item.decode("utf-8") for item in payload.split(b"\0") if item]


def index_size(path: str) -> int | None:
    try:
        return int(git("cat-file", "-s", f":{path}", text=True).strip())
    except subprocess.CalledProcessError:
        return None


def index_mode(path: str) -> str | None:
    rows = git("ls-files", "--stage", "--", path, text=True).splitlines()
    return rows[0].split(maxsplit=1)[0] if rows else None


def allowed(path: str) -> bool:
    return path in ALLOWED_FILES or path.startswith(ALLOWED_PREFIXES)


def forbidden(path: str) -> bool:
    return path in FORBIDDEN_FILES or path.startswith(FORBIDDEN_PREFIXES)


def release_path_error(path: str, *, present_in_index: bool) -> str | None:
    """Return a boundary error, allowing a staged deletion of forbidden state."""
    if forbidden(path):
        if present_in_index:
            return f"forbidden release path: {path}"
        return None
    if not allowed(path):
        return f"path is outside the release allowlist: {path}"
    return None


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/master"
    changed = nul_paths(
        git(
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            "-z",
            base,
            "--",
        )
    )
    errors: list[str] = []
    if not changed:
        errors.append(f"no staged release changes relative to {base}")

    for path in changed:
        size = index_size(path)
        path_error = release_path_error(path, present_in_index=size is not None)
        if path_error:
            errors.append(path_error)
        if size is not None and size > MAX_GIT_BLOB:
            errors.append(f"Git blob exceeds 100 MiB: {path} ({size} bytes)")
        if path.startswith("docs/") and index_mode(path) == "120000":
            errors.append(f"symbolic link is not allowed under docs/: {path}")

    indexed_docs = nul_paths(git("ls-files", "-z", "--", "docs"))
    docs_size = sum(index_size(path) or 0 for path in indexed_docs)
    if docs_size > MAX_PAGES_TREE:
        errors.append(f"docs/ exceeds 1 GiB: {docs_size} bytes")
    for required in ("docs/index.html", "docs/.nojekyll"):
        if index_size(required) is None:
            errors.append(f"required Pages file is not staged in the index: {required}")

    if errors:
        print("GitHub release audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"GitHub release audit passed: {len(changed)} changed paths; "
        f"docs/ contains {len(indexed_docs)} files and {docs_size} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
