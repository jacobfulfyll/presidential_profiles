---
name: attention-test-seams
description: Hermetic-test seams and threshold-pinning recipes for presidential_profiles attention.py (topic lifecycles, genre strata, speech-clustered bootstrap)
metadata:
  type: reference
---

Authoring offline tests for `attention.py` (topic attention over time). Builds on
[[test-toolchain]] for the worktree pytest invocation.

**Seams (all read at call time, so monkeypatchable):**
- `A.PARA_LABELS_PATH` and `A.CROSSWALK_PATH` are looked up in module globals at
  call time — patch freely. `A.GROUND_TRUTH` too (`run_ground_truth` reads it).
- `load_taxonomy(path=TAXONOMY_PATH)` and `exemplar_quotes(paragraphs_path=
  PARAGRAPHS_PATH)` bind their defaults at DEF time — patching the constant does
  NOT affect them. Pass the path explicitly, or (for `main()`) monkeypatch
  `A.load_taxonomy` itself.
- `load_inputs` becomes fully hermetic by writing synthetic
  `paragraph_annotations.parquet` / `speech_annotations.parquet` into conftest's
  autouse `redirect_annotation_dirs` tmp root + patching `A.PARA_LABELS_PATH`.
  The annotation frames need a `run_id` column (`_require_key` demands it).

**Pinning recipes that worked (all verified by mutation, 33/33 caught):**
- *Three-leg thresholds*: build ONE curve per leg where that leg is the sole
  binding constraint, with the other two legs comfortably satisfied by
  realistic derived numbers (share = n_topic_window / n_total_window). For a
  `max(absolute, relative)` floor you need BOTH a curve where the absolute half
  binds and one where the relative half binds — either alone survives a
  `max`→`min` mutation.
- *Post-stratification*: to prove renormalized weights sum to 1 without being
  able to see the weights, give every stratum the SAME within-stratum share —
  the combined value can only equal it if Σw = 1.
- *`_decline_ratio`-style windowed means*: never make the window flat, or
  "mean over the window" and "value at the centre year" are indistinguishable
  and the half-width mutation survives. Spike one year inside the window.
- *Cluster bootstrap*: two independent pins are needed. (a) CI width vs the
  closed form `2*1.96*sqrt(p(1-p)/n)` with rel=0.20 catches a wrong draw COUNT;
  (b) a maximally-clustered fixture (homogeneous speeches) plus a from-scratch
  numpy paragraph-level bootstrap computed in the test catches resampling the
  wrong UNIT — assert clustered width > 2x naive, not a magic number.
- Seed-sensitivity tests need a corpus wide enough that the 2.5/97.5 percentiles
  are not saturated; a 6-speech fixture gives identical lo/hi for every seed and
  the test passes vacuously.

**pandas 3.0 gotchas here:** `DataFrame.where(series_cond, other_df)` broadcasts
the Series along ROWS (index-aligned), which is what `_combine_strata`'s fallback
relies on. Parquet round-trips object columns to `ArrowStringArray` and `None` to
NaN, so `assert_frame_equal` on a written lifecycle table needs
`.fillna("").astype(str)` on the object columns rather than `check_dtype=False`.
Booleans come back as `np.True_`, so use `bool(row[col]) is True`.

**Mutation harness note:** `attention.py` is UNTRACKED while the task is in
flight — see [[mutation-harness-untracked-files]]. 33/33 mutations caught after
two rounds; see [[pin-dont-fix-known-defects]] for the defects deliberately left
unfixed.

**QUALITY-CHECK findings (2026-07-21), all mutation-verified:**
- *`_bootstrap_corpus` is unusable for genre standardization.* Both its eras hold
  3 SOTU + 3 OTHER paragraphs and its corpus genre weights are 6/6, so
  post-stratification collapses onto pooling and
  `_era_shares(..., "genre_standardized")` is **elementwise identical to
  `"raw"`**. Three mutants survived on it: unify eligibility to
  `den >= MIN_GENRE_STRATUM`, collapse the whole branch to pooled raw, and swap
  corpus weights for equal weights. `_mix_shift_corpus` has the same defect at
  the year level (exact 50/50 corpus ⇒ equal weights == corpus weights).
- *The fix that works:* a `_stratum_corpus(n)` builder — founding SOTU stratum 4
  paragraphs / 1 topic hit, founding OTHER stratum `n` paragraphs all hits,
  modern filler pinning corpus weights at OTHER 4 / SOTU 12. Four rules then give
  four different numbers (0.4375 actual / (1+n)/(4+n) pooled / 0.625 equal-weight
  / 0.25 at `den >= 3`). `n=3` pins post-stratification, `n=2` pins the
  eligibility divergence. Assert `_combine_strata` on the SAME 2-vs-4 config in
  the same test (it returns 0.25) — that makes defect #9 numeric, not rhetorical.
- *The CLI `--quotes` test was non-hermetic in a load-bearing way.* `main()` calls
  `exemplar_quotes` with no `paragraphs_path`, and the default binds at DEF time,
  so it read the real 36k parquet; the synthetic keys matched nothing and the
  unguarded inner join dropped all 24 pooled rows, so the test passed over a DEAD
  quotes path. Seam: `monkeypatch.setattr(A, "exemplar_quotes",
  functools.partial(A.exemplar_quotes, paragraphs_path=tmp_parquet))`. This also
  removed a spurious coupling: adding `_require_full_merge` at line 1204 used to
  fail the CLI test, contradicting the blast-radius table's "nothing fails".
- *All 12 `All-NaN slice` RuntimeWarnings* come from `np.nanpercentile` inside
  `bootstrap_era_shares`, in exactly the 10 `TestBootstrapEraShares` tests whose
  fixtures do not populate all 9 eras (`-W error::RuntimeWarning` proves it).
  Nothing else in the suite emits one. LEAVE is correct.
