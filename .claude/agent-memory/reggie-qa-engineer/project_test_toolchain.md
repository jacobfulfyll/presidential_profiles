---
name: test-toolchain
description: How to install and run pytest in the presidential-profiles repo (uv-managed x86_64 venv, no pip)
metadata:
  type: reference
---

Running/adding Python test tooling in `presidential-profiles`.

- The shared venv (`/Users/jacobpress/Desktop/Projects/presidential_profiles/.venv`) is **uv-managed and has no `pip`** — `python -m pip install ...` fails with "No module named pip". Install dev deps with:
  `VIRTUAL_ENV=/Users/jacobpress/Desktop/Projects/presidential_profiles/.venv arch -x86_64 uv pip install "pytest>=8.0"`
- The venv is **x86_64 under Rosetta**; always prefix the interpreter with `arch -x86_64`. Run tests with:
  `arch -x86_64 /Users/jacobpress/Desktop/Projects/presidential_profiles/.venv/bin/python -m pytest`
- pytest config lives in `pyproject.toml` under `[tool.pytest.ini_options]` with `pythonpath = ["src"]` (package is `src/presidential_profiles`) and `testpaths = ["tests"]`. `pytest>=8.0` is declared under `[project.optional-dependencies].dev`.

**Why:** first test infra was introduced July 2026; these were non-obvious discoveries (no pip in the venv, Rosetta arch prefix required).
**How to apply:** reuse these exact invocations for any future test/lint work here instead of rediscovering them.

**Worktree path gotcha:** module data paths (`PARAGRAPHS_PATH`, `issues.PARA_LABELS_PATH`, `corpus.DATA_DIR`) are `REPO_ROOT / "data"` where `REPO_ROOT` resolves relative to the imported package's `__file__`. The shared venv lives at the MAIN repo, so a bare `python -` script run from a `.worktree/` checkout imports the MAIN-repo package and reads MAIN-repo data (stale/un-migrated). pytest is fine because `pythonpath=["src"]` prepends the worktree src. To validate a worktree's committed data artifacts, read the parquet by **absolute worktree path**, don't rely on the module constant.
