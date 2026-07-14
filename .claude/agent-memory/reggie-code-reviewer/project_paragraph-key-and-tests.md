---
name: paragraph-key-and-tests
description: Canonical paragraph join key, positional-alignment review heuristic, and how to run the test suite in presidential_profiles
metadata:
  type: project
---

Paragraph-level parquet tables in this repo are keyed on `(doc_name, para_idx)`.
`paragraphs.parquet` and `paragraph_issues.parquet` (CorEx issue labels) must be
joined with `.merge(on=["doc_name","para_idx"], how="inner", validate="one_to_one")`.

**Why:** Historically these two tables were aligned purely by row order, guarded
only by a length check that could not detect reordering — a silent
mislabel/corruption path (fixed in task fix-paragraph-issues-key, 2026-07).

**How to apply (review heuristic):** Treat any positional / row-order alignment of
two paragraph-level tables as a corruption-class finding. The canonical key is
above. Note that `how="inner"` silently drops rows if the two files' key sets
diverge — the old length-mismatch guard is NOT preserved by `validate="one_to_one"`
(which only checks key uniqueness, not set equality). A post-merge
`len(merged) == len(both sources)` assertion is the belt-and-suspenders fix.

`site.py:fig_issues_decade` and `explorer.py` load `paragraph_issues.parquet` but
only `groupby(<named column>)` on it — they never join it to paragraph text, so
they correctly do NOT need the keyed merge. Verify this by reading actual usage
before flagging them.

**Tests:** pytest was introduced in this task (`tests/`, first test framework in
the repo). Run with:
`arch -x86_64 /Users/jacobpress/Desktop/Projects/presidential_profiles/.venv/bin/python -m pytest tests/ -q`
(repo venv is x86_64 under Rosetta at the MAIN repo path; worktrees have no own
`.venv`). `pyproject.toml` sets `pythonpath=["src"]` so imports resolve without an
editable install. See [[convergence-investigation-direction]] for why this key fix
was a prerequisite.
