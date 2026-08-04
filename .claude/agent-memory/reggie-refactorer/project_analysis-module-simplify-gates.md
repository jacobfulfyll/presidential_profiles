---
name: analysis-module-simplify-gates
description: The two behaviour gates that make a SIMPLIFY pass on this repo's derived-artifact + chart modules safe (parquet digest + docs/ rebuild), and what such a pass actually yields
metadata:
  type: project
---

Every derived-artifact module here (`combat.py`, `eras.py`, `bands.py`, `triangulate.py`)
ships the same two-gate proof that a refactor changed nothing. Use BOTH; they cover
disjoint code.

1. **Artifact byte gate.** Regenerate straight into `data/` (`python -m
   presidential_profiles.bands`, `.combat`, `.eras`) and check `git status --short data/`
   is empty. These modules carry no wall-clock stamp on purpose, so a dirty status means
   numbers moved. Covers the compute path only.
2. **`docs/` rebuild gate.** `PYTHONPATH=src ... -m presidential_profiles.site` regenerates
   every page deterministically; `git status --short docs/` empty is a byte-for-byte proof
   for the CHART code, which the artifact gate cannot reach at all. Delete
   `docs/explorer/meta.json` first (it is expected to regenerate). Takes ~2 min.

**Yield is small and that is correct.** `eras.py` was -3 lines; `bands.py`/`site.py`/
`issues_site.py` (topic-chart-upgrades, 2026-07-21) netted +positive because the wins were
docstrings recording conventions rather than deletions. In this repo the highest-value
SIMPLIFY output is usually **prose that names the real mechanism**, not fewer lines — the
codebase is already dense and deliberately over-commented, and CLAUDE.md's standing lesson
is that wrong stated mechanisms outlive wrong numbers.

**Reliable removals found in this family:** dead function params still passed at the call
site (`fig_issues_decade`'s `issue_df`/`scores`; `_write_portraits`'s `per_era`), unused
imports left behind by a rewrite, `merge(left_on=X, right_on=X)` → `on=X`, and repeated
`DISPLAY_LABELS.get(name, name)` lookups inside one function.

**Do not touch:** bootstraps, composition rules, three-way status strings, merge guards,
run-splitting index arithmetic. These are individually test-pinned and load-bearing, and
every pipeline before you already verified them.

**Third gate, added by `suppress-degenerate-band-intervals`: a stale `docs/` now FAILS the
suite** (an end-to-end test compares committed HTML against the parquet). Rebuild the site
before the final run, not after.

**Interpolating typed prose in a meta JSON is usually byte-neutral.** Replacing a typed
`n**-n` ladder + conclusion with values derived from `CI_LOW` reproduced `bands_meta.json`
byte-for-byte, because the block was correct at today's constants — the fix only changes
what happens after a retune. Expect no diff and verify rather than assuming one; write to a
`tempfile.mkdtemp()` via `write_bands(table, tmp/...)` to compare digests without touching
committed artifacts.

**Distinguish typed-vs-derived from self-defeating-rationale.** Both look like "prose that
should be computed", but the first is correct as published and only breaks after a retune
(keep it correct), while the second was false the day it shipped (correct it). Say which in
the test docstring — it tells the next reader whether a failure means regression or
discovery.
