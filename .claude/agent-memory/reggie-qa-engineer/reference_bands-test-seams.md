---
name: bands-test-seams
description: Hermetic test seams for bands.py / issues_site.band_traces / site.llm_divergence_sentence + build_html, and the fixture-symmetry traps that hid four mutants
metadata:
  type: reference
---

Seams for the `topic-chart-upgrades` band layer (`tests/test_bands.py`,
`tests/test_band_charts.py`).

**Undo conftest's redirect to read the frozen annotator parquets.**
`redirect_annotation_dirs` is autouse and repoints `llm_annotations.ANNOTATIONS_DIR`
at tmp_path, so `load_paragraph_annotations` cannot see the real tables. Point it
back read-only with `monkeypatch.setattr(ann, "ANNOTATIONS_DIR",
attention.TAXONOMY_PATH.parent)` — `attention.TAXONOMY_PATH` is bound at import and
is never redirected, so it is a reliable handle on the real dir.

**`llm_bands` runs hermetically on injected components**: monkeypatch
`attention.load_inputs` (returns `(paragraphs, assignments)`; only `paragraphs`
needs `doc_name` + `era`), `attention.bootstrap_era_shares`, and
`bands.disagreement_half_widths`. Everything the function itself owns — merge
guard, three-way status, widen-never-narrow, clip — then runs for real on ~2 rows.

**`paired_annotations` runs hermetically too**: monkeypatch
`bands.load_paragraph_annotations` (imported INTO the bands namespace, so patch it
there, not on `llm_annotations`) plus `issues.PARA_LABELS_PATH` at a tmp parquet of
`(doc_name, para_idx, year)`.

**The whole builder is cheap**: `corex_bands()` 0.5 s, `llm_bands()` 0.6 s, and
`build_bands()` + `write_bands()` reproduces the committed `data/bands.parquet` and
`bands_meta.json` **byte-for-byte**. A build-twice-and-sha256-against-the-shipped-
artifact test is ~2 s and is the strongest determinism anchor available here.

**Fixture-symmetry traps that hid mutants in the first batch** (see
[[mutation-check-recurring-holes]], [[same-frame-invariance-tests]]):
- a thin-vs-dense bootstrap fixture where every speech has the SAME share makes
  every resample identical and both widths 0 — the "thin band is wider" assertion
  passes vacuously. Speeches must disagree, and the dense arm needs a
  `width > 0` guard. All-or-nothing dense speeches over-widen; use 4/5/6-of-10.
- a polygon fixture with flat `lo`/`hi` cannot see `y=hi + lo[::-1]` losing its
  reversal. Vary both bounds per period and zip x against y.
- `have_interval = lo.notna() & hi.notna()` survives `&`→`|` unless one test has
  exactly ONE bound missing.
- `_ci_status(n_speeches, n_paragraphs)` survives an argument swap unless a
  fixture is asymmetric across the two floors (40 speeches / 10 paragraphs).
- `rows_with_component_applied` survives `applied`→`n_llm` against the real
  artifact, where all 450 rows apply. Needs a synthetic mixed table.

**Equivalent mutant, do not chase**: `band_traces`'s `if not have.any(): return []`
— the groupby below yields no groups on an all-False mask anyway.

**`bands.main` is testable without touching source, but only via `write_bands`.**
`write_bands` binds `path: Path = BANDS_PATH` at DEF time, so monkeypatching
`B.BANDS_PATH` does NOT redirect it and a naive `main()` call overwrites the
committed artifact. Monkeypatch `B.write_bands` (and `B.build_bands` → the shipped
frame, for speed) instead, and digest `data/bands.parquet` before/after inside the
fixture so a refactor that reintroduces the real write fails loudly. Assert the
redirected file EXISTS afterwards — asserting only on stdout leaves a `main()` that
builds and reports but never persists fully green.

**`print_checks` mutates global pandas state** (`pd.set_option("display.width", 200)`)
with no restore, so calling it in a test leaks into every later test's repr width.
Harmless today; worth `monkeypatch.context` if it ever matters.

**`fig_issues_decade(band_table=None)` is cheap to test, contrary to first
impression**: `issue_df`/`scores` are now dead parameters, and
`topic_quality.surfaced_discovered()` returns a single extra column, so a synthetic
frame with three boolean columns drives BOTH branches through the real
`display_issues` / `DISCOVERED_LABELS` plumbing. Testing the two branches against
each other on one fixture (fallback drops 1785, banded draws it) is what makes the
"the `>= 40` mask became a rendering distinction" claim assertable at figure level.
