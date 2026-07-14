---
name: run-commands
description: How to run tests and the site build in the presidential_profiles repo (x86_64 Rosetta venv)
metadata:
  type: project
---

The repo venv is x86_64 under Rosetta and lives at the MAIN repo path, not inside worktrees. Worktrees have no `.venv` of their own.

- Run tests: `arch -x86_64 /Users/jacobpress/Desktop/Projects/presidential_profiles/.venv/bin/python -m pytest -q`
- Rebuild the static site: `PYTHONPATH=src arch -x86_64 /Users/jacobpress/Desktop/Projects/presidential_profiles/.venv/bin/python -m presidential_profiles.site` (writes to `docs/`).

**Why:** Invoking the venv python directly (or an arm64 interpreter) fails on native deps; the `arch -x86_64` prefix is required. Not obvious without being told, and costly to rediscover.

**How to apply:** Use these exact invocations to verify behavior preservation after any refactor. The site build's issue pages come from `issues_site.render_issue`; profile/issue/compare pages regenerate deterministically, so a byte-for-byte diff of `docs/` against a pre-change snapshot is a reliable behavior check.
