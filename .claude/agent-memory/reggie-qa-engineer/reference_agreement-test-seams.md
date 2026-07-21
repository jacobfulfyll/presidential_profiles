---
name: agreement-test-seams
description: Offline test seams for presidential_profiles agreement.py (inter-model draw + metrics + build/report) and the annotate.py --model/--sample MOD
metadata:
  type: reference
---

Testing `agreement.py` (inter-model Sonnet-vs-Opus agreement) + the annotate.py
`--model`/`--sample` MOD. Builds on [[annotate-cli-test-seams]] / [[test-toolchain]].

- **Pure metrics are the easy win**: `compute_kappa` (identical→1.0, inverted→-1.0,
  degenerate single-class→`(nan, n)`, empty→`(nan,0)`), `jaccard`/`compute_jaccard`
  (both-empty→1.0, disjoint→0, mean over sequence), `compute_exact_match`,
  `compute_entity_metrics` (keyed `(doc, para, normalized_name)`; name-match vs
  stance-agreement vs per-side unmatched-rate all separate; both-empty→name 1.0 +
  stance NaN), and `build_metrics_table` are all pure over tiny frames — no I/O.
  `build_metrics_table` output cols are exactly
  `[field, era_bin, era_label, metric, value, n]`; the OVERALL row is `era_bin=-1`
  (`G.OVERALL_BIN`), `era_label="overall"`; factual fields only appear when a
  non-None `merged_speech` is passed.
- **M3 documented gap**: metric helpers assume list-like topics. `jaccard(None, x)`
  raises `TypeError` (null topic NOT coerced to empty set). Pinned as current
  behavior — flip the test if agreement.py later coerces null→[].
- **draw_sample is fully injectable**: pass `speeches=<synthetic frame with
  doc_name+year>`, `path=tmp_path/...`, `draw_date=` (deterministic), `force=True`.
  Bin sizing = `min(n, max(min_per_bin, int(0.25*n + 0.5)))` — round-HALF-UP then
  min-3 clamp. Use n=4 (→clamp to 3), n=14 (→4, proves round-half-up≠floor), n=40
  (→10) to exercise all three on distinct 30-yr bins. Overwrite without
  `force=True` → `FileExistsError`. Assert `n_sampled < n_corpus` so determinism
  isn't vacuous; a different seed must move membership (proves seed is threaded).
- **build_agreement / agreement_report seam**: both take `speeches=` (synthetic),
  `out_path`/`agreement_path`/`report_path`/`sample_path` — all injectable, so
  NEVER touch the module-level `SAMPLE_PATH`/`AGREEMENT_PATH` (computed at import
  from the REAL `ann.ANNOTATIONS_DIR`). Runtime table reads go through
  `ann.annotation_path` (conftest's `redirect_annotation_dirs` already points it at
  tmp_path) BUT the coverage arithmetic reads `ann.PARAGRAPHS_PATH` (NOT
  redirected) — monkeypatch it to a tiny synthetic corpus parquet (needs a `text`
  column for the report's top-disagreement quotes). Write the primary table as
  `paragraph_annotations.parquet` and the Opus one as
  `A._table_name("paragraph_annotations","claude-opus-4-8")` =
  `paragraph_annotations__opus4-8`.
- **one_to_one is belt-and-suspenders**: the merge `validate="one_to_one"` can't be
  independently tripped — `ann.load_paragraph_annotations` (`_require_key`) raises
  `ValueError("...duplicate key...")` FIRST. So the observable dup-key guard is the
  loader; test a duplicated `(doc_name, para_idx)` in the primary parquet →
  `build_agreement` raises "duplicate key".
- **M2 coverage is advisory**: partial Opus coverage (Opus table missing some
  sampled paras) → build prints `WARNING ... < 100%` AND embeds it in the report
  markdown, but STILL writes the table over the overlap (not a hard exit). Assert
  via `capsys` and by reading the `.md`.
- **annotate MOD helpers**: `_resolve_model(None)`→default, unknown→`SystemExit`;
  `_rates("claude-opus-4-8")`==(5,25) date-independent, `cost_usd(..., model=opus)`
  applies the 50% batch discount → 2.50/12.50; `_intro_applies(opus)` always False.
  `_validate_request` thinking-disabled guard is keyed `params["model"] in MODELS`
  so it fires on Opus too — build a minimal req dict
  (`MessageCreateParamsNonStreaming(model=, max_tokens=, messages=, thinking=)`)
  rather than the full corpus path. `_load_sample_docs` → `SystemExit` on missing
  file / empty list / unknown-doc (stale-sample) . Byte-identical regression:
  `build_requests()` == `build_requests(model=MODEL)`; Opus request differs ONLY
  in the `model` field (param keys are `{model,max_tokens,thinking,output_config,
  system,messages}`). `cmd_dry_run(args(model="claude-opus-4-8", sample=path))`
  works offline (guard with `anthropic_construction_bomb`); `--pilot`+`--sample`
  together → SystemExit.

**Verified 2026-07-21 against IMPLEMENT 93a3429: 78 new tests, all green, 407 total.
No src bugs found; judge minors M1-M3 confirmed as documented behavior, not fixed
(WRITE-TESTS is tests-only).**
