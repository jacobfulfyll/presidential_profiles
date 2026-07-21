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

**More gotchas (found writing the annotate/llm_annotations suite, July 2026):**
- pytest (9.1.1) is already installed in the shared venv even when a given worktree's `pyproject.toml` does NOT declare it. `uv pip install "pytest>=8.0"` is then a no-op ("Checked 1 package"). A fresh worktree may still need `[tool.pytest.ini_options]` (pythonpath/testpaths) + the `dev` optional-dep added to its own pyproject. Do NOT run `uv sync` here — it targets `.venv` in the worktree (absent) and prunes; use `uv pip install` per above.
- `tests/` has no `__init__.py`; under pytest's default prepend import mode each test module is imported by basename, so `tests` is NOT an importable package — `from tests.conftest import X` / `from tests.test_other import Y` fails with ModuleNotFoundError. Put shared helpers in `conftest.py` fixtures or define them locally in each test.
- Faking the Anthropic **paid path** offline: the CLI functions do `import anthropic; anthropic.Anthropic()` inside the function body, while `annotate.py` imports `anthropic.types.*` at module load. So `monkeypatch.setattr(anthropic, "Anthropic", fake)` in-place is the clean seam — it swaps the client class but leaves the types intact for import. Pair with an autouse fixture that `delenv`s `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` so any unpatched real construction fails loudly.
- **Coverage:** `pytest-cov` is NOT in the shared venv by default, though it IS declared (`pyproject.toml` `[project.optional-dependencies].dev`, `pytest-cov>=5.0`) and resolved in `uv.lock` at 7.1.0. To install: `VIRTUAL_ENV=<mainrepo>/.venv arch -x86_64 uv pip install "pytest-cov>=5.0"` (leaves uv.lock untouched — do NOT `uv sync`). Run: `... -m pytest tests/ --cov=presidential_profiles.<mod> --cov-report=term-missing`. `.coverage` is written to CWD — `rm -f .coverage` after.
  **But weigh the cost first when working in a worktree:** that install mutates the SHARED `.venv` at the main repo that every worktree imports through, so it is an environment change the task does not own. If the task forbids touching the shared venv, say so honestly — do NOT claim "pytest-cov would require editing pyproject.toml", which is false and was flagged as a wrong rationale in the `issue-attention-over-time` QUALITY-CHECK. Mutation probing is the stronger substitute anyway: coverage says a line RAN, a surviving mutant says no assertion DEPENDED on it.
