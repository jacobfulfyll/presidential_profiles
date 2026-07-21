---
name: verification-register-note
description: How to verify register.py + notes/register-findings-v1.md (note-vs-parquet cross-check recipe, degenerate-CI gotcha, double-rounding artifact)
metadata:
  type: reference
---

Verifying `presidential_profiles.register` and its published research note. Complements
[[verification-workflow]] (Rosetta venv + worktree PYTHONPATH discipline).

**Rebuild is fast and byte-deterministic:** `PYTHONPATH=src <rosetta-py> -m presidential_profiles.register`
runs in ~5.3s, writes 15,804 rows / 396,141 bytes. Deleting and regenerating twice reproduced
sha256 `680d51ca3ee…` exactly both times — so a dirty `data/register/trends.parquet` after a
rebuild is a real signal, like `docs/`. No test reads the parquet, so deleting it is safe.

**Cross-checking a research note against its parquet — the recipe that worked.** Build a tiny
query helper (`era(measure, taxonomy, treatment)` / `st(measure, tax, treat, statistic, field)`
over the long-format frame), then transcribe each note table as a Python list of tuples and
assert with a `round(actual, dp) == claimed` checker. 378 tabulated claims took one script and
caught 7 last-digit misses that eyeballing would have missed. Do the table-by-table exploration
FIRST (to learn the schema), then the assertion sweep as evidence.

**Not every note claim lives in the parquet.** The era-counts table, rarefaction table, reference
genre mix, label-resolution counts, and the `neither`-stance series are computed by
`era_counts` / `rarefied_effective_topics` / `reference_genre_mix` / `label_resolution_report` or
straight off the `SpeechPanel` — none are written to `trends.parquet`. Verify those by importing
the module and calling the function, not by querying the file. In particular there is **no
`neither_share` measure** and `pv_mixed` is not published, so the note's 48.0% → 8.9% / 20.5%
figures are only reproducible from the panel.

**DEGENERATE-CI GOTCHA (the finding worth remembering): 49.4% of `trends.parquet` rows have
`ci_low == ci_high`.** All 6,624 `unit=year, genre_treatment=sotu_only` rows have
`n_speeches == 1` — a speech-clustered bootstrap over ONE cluster resamples the same speech every
time, so the "CI" is just the point estimate repeated. Plus 972 single-speech raw-year rows.
Nothing in the schema flags this; `n_speeches` is the only tell. Era/trend rows are unaffected
(their 12 zero-width CIs are genuine structural zeros: `opponents`=0 in 1770, `zero_topic_share`=0).
Always check `ci_low == ci_high` counts against `n_speeches == 1` on any bootstrap artifact here.

**Bootstrap point estimate can fall OUTSIDE its own percentile CI** for entropy measures: 3 rows
(year 1984, llm_level2 `effective_topics`/`_plugin`/`normalized_entropy`) have `value > ci_high`.
Expected, not a bug — a resample covers only ~63% of distinct speeches, so replicate topic
richness is biased low relative to the full-sample plug-in. Worth reporting, not failing.

**Double-rounding is the realistic prose defect, not wrong data.** The 7 near-misses were all the
author rounding twice (8.334966 → 8.335 → "8.34"). Errors ≤0.006, never flipping a sign,
significance verdict or ordering. Report as one systematic cosmetic finding, not 7 defects.

**Independent quantities to sanity-check while you are in there:** era counts must sum to the real
corpus (1,057 speeches / 36,229 paragraphs / 4,179,266 paragraph-words); `sotu_only` era
`n_speeches` must equal `era_counts.n_sotu_speeches`; the within-speech ICC the note cites (0.119)
is NOT computed anywhere in the code — a one-way ANOVA ICC gives 0.116 for `n_legacy` but 0.135
for `n_llm` and 0.295 for `llm_zero`, so "runs to 0.119" understates it.
