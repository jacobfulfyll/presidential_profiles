---
name: analysis-module-simplify-shape
description: What a SIMPLIFY pass on this repo's big analysis modules (attention.py, taxonomy.py, annotate.py) actually yields — expect a POSITIVE line delta and dedup-by-naming, not deletion
metadata:
  type: project
---

This repo's large analysis modules (`attention.py` 1,444 lines, plus the `taxonomy.py` /
`annotate.py` family) arrive at SIMPLIFY already tight — three review stages at 9.0–9.5 run
first, so obvious duplication is gone. Expect the net line delta to go **positive**.

**Why:** the judge-sanctioned work-order items are inherently additive (documentation fixes,
merge guards, wiring an unwired report into `main()`), and a large fraction of these files is
deliberate scientific provenance — `attention.py` carries a ~180-line pre-registration docstring
recording thresholds fixed *before* results were seen. Compressing it is a regression, not a win.
Measured composition after the pass: 771 code / 556 docstring / 90 comment / 232 blank.

**How to apply:**
- Report the delta split by category (code vs docs vs blank) or the number looks like failure.
  On `issue-attention-over-time`: +120 total = ~+87 docs, ~+14 blank, ~+19 code — and ~15 of
  those 19 code lines were the two *commissioned behaviour additions*, so pure restructuring was
  roughly line-neutral while removing 5 duplicated expressions.
- The real wins here are **dedup by naming**, not deletion: `level_assignments` (the level-1
  rollup dedup rule, duplicated across 3 call sites), `level1_parents`, `headline_rows` (the
  `(level2, raw)` filter repeated 3x), `lifecycle_class` (the born/died/revived rule, previously
  an inline if/elif that no doc stated), `_attach_era_shares` (a nested comprehension-in-loop).
- Two functions can share *edges* without sharing an implementation: `era_name` (scalar loop) and
  `era_series` (`pd.cut`) had to keep BOTH public names (20+ parametrized tests call both). The
  fix was shared `_ERA_LABELS`/`_ERA_EDGES` constants plus making the scalar loop use `pd.cut`'s
  half-open `(prev_hi, hi]` convention, so their equivalence is by construction rather than
  coincidence. Do not collapse one into the other.
- Do NOT convert `era_series` to `.map(era_name)`: `pd.cut` returns a Categorical, and
  `groupby(observed=False)` downstream depends on the full category set.

See [[byte-oracle-beats-work-order]] and [[annotation-money-path-guards]].
