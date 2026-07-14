---
name: verification-workflow
description: How to run tests and build the presidential_profiles static site during VERIFY-APP stage
metadata:
  type: reference
---

Verification recipe for the `presidential_profiles` static-site generator.

**Interpreter:** the repo venv is x86_64 under Rosetta, living at the MAIN repo path (worktrees have no `.venv` of their own):
`arch -x86_64 /Users/jacobpress/Desktop/Projects/presidential_profiles/.venv/bin/python`

**Test suite** (run from worktree root): `... -m pytest -v`. pyproject sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, so no PYTHONPATH needed for pytest.

**Full site build:** `PYTHONPATH=src arch -x86_64 <venv-python> -m presidential_profiles.site` (console script `pp-site` → `presidential_profiles.site:main`). Expected output: `docs/index.html`, 45 profile pages + index under `docs/presidents/`, `docs/explorer.html`, 16 issue pages + index under `docs/issues/`, `docs/compare.html`.

**Build is deterministic:** rebuilding produces byte-identical `docs/` — `git status --porcelain docs/` stays empty if the committed docs already reflect the current code. So a *dirty* `docs/` after a rebuild is a real signal that generated content changed (not noise). Useful as a regression check.

**Content sanity-check (the real regression test for text↔label pairing bugs):** issue pages under `docs/issues/*.html` render three quotes each (Early / At its peak / Most recent) with a `President, Title, Year` citation; profile pages under `docs/presidents/*.html` render per-issue cards each with a verbatim quote + citation. Extract these and confirm each quote is on-topic for its labeled issue and the citation is internally consistent.

**Gitignore note:** `.pytest_cache/` self-ignores (pytest writes its own `.gitignore` inside), so it won't show as untracked. `.claude/agent-memory/` is agent infra, not a project artifact.
