---
name: mirror-tests-need-real-fn-anchor
description: In presidential_profiles, "mirror" tests of inline profiles.py logic catch constant-drift but not formula-shape drift; anchor with a real-fn call via the parquet-monkeypatch harness
metadata:
  type: feedback
---

When QA'ing tests for `profiles.issue_cards` logic (eligibility / `strengths()` /
`topic_of_day` / `low_confidence`), remember these predicates are **inline
closures** inside `issue_cards`, so the test suite mirrors them by re-expressing
the formula against the *imported* thresholds (`REL_DISTINCT_PP`,
`RAW_ELEVATED_MULT`, `MIN_ISSUE_PARAS`, `SPARSE_MIN_SPEECHES`).

**Why:** a mirror that imports the constants FAILS if a constant drifts (good),
but PASSES even if the real source's formula *shape* changes (e.g. `abs(rel) <`
becomes `rel <`) because it tests its own copy. The test file's own DRIFT NOTE
recommends hoisting the closures to module scope — but that is a source change,
out of scope for a QUALITY-CHECK pass.

**How to apply:** you do NOT need the source hoisted to close the gap. `issue_cards`
can be driven hermetically over synthetic parquets by monkeypatching
`profiles.PARAGRAPHS_PATH` and `issues.PARA_LABELS_PATH` at tmp_path parquets
(pattern already in `tests/test_keyed_merge.py::_install_parquets`). `issue_df` /
`df` (titles) are passed directly and are fully hand-computable — no 36k corpus,
no CorEx. Add ONE real-fn anchor asserting the highest-value new behavior (the
raw-axis-only eligibility bug fix + `topic_of_day=True`); watch the FP boundary
(`raw_strength` at exactly 1.0 is fragile — pick shares that land ~1.04, not 1.0).
The plotly/HTML layer (`profiles_site.py`) is intentionally not unit-tested — its
0% coverage is an accepted convention, not a gap. See [[test-toolchain]].
