---
name: era-atlas-simplify-shape
description: How to run the byte gate on eras.py (era-atlas) safely without touching the paid portrait path, and what a SIMPLIFY pass there yields
metadata:
  type: project
---

`src/presidential_profiles/eras.py` (~1250 lines, the `era-atlas` task) joins the byte-gate
analysis-module family ([[analysis-module-refactor-gates]]), with one module-specific split that
makes the gate safe to run:

- **Deterministic byte gate:** `eras.run()` (call the function, not the CLI) writes EXACTLY the 7
  parquets + `eras_meta.json` into `ERAS_DIR` (defaults to `data/eras/`, resolves to the worktree
  under `PYTHONPATH=<worktree>/src`). Regenerate straight into `data/eras/` and
  `git status --short data/eras/` → empty means byte-identical. Confirmed reproducible on
  2026-07-21. No wall-clock stamp (combat precedent), so a dirty status means numbers moved.
- **`run()` NEVER touches the paid path.** Portraits live behind a separate entry
  (`generate_portraits` → `_write_portraits`) that `run()` does not call, so the byte gate leaves
  `era_portraits.parquet`, `manifests/`, and `portraits_estimate.json` untouched — no money risk
  in regenerating. Only `generate_portraits`'s per-era `client.messages.create` loop spends; the
  WRITE helper `_write_portraits` is fully OFFLINE and directly test-covered (4 tests), so genuine
  dead code THERE is removable and verified by the suite, not the byte gate (refines
  [[annotation-money-path-guards]]: the write-side of a paid module can still be a clean SIMPLIFY
  target).

**SIMPLIFY pass (2026-07-21, IMPLEMENT 9.2):** small negative delta (eras.py -3 lines). All wins
were dead-code removal: unused local (`speech_unit`), unused params (`_boundaries_at_k`'s
`center_years`, `_write_portraits`'s `per_era` + its feeder dict, `_match_canonical`'s dead
`tolerance` default), a dead `bins` local in `periodization`, and one truthful-docstring +
`tuple[int | None, int]`→`tuple[int, int]` fix on `_match_canonical` (the tolerance is
single-sourced on `BIN_WIDTH`: one decision site `abs(gap) <= BIN_WIDTH` in `periodization`, plus
meta reporting `match_tolerance_years: BIN_WIDTH` — both reference the constant, not copies).
Removing the two private-helper params required mechanical edits to `tests/test_eras.py`
(positional call-site churn only, no assertion changes). Suite 1689 passed before and after.
