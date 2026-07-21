---
name: behavior-preservation-checks
description: Cheap byte-level and number-level ways to prove a refactor changed nothing in presidential_profiles
metadata:
  type: project
---

Two checks in this repo prove behavior preservation far more strongly than the test suite
alone, and both are cheap:

1. **Byte-identical artifact rebuild.** `topic_quality.build_names(path=<tmp>)` regenerates
   `data/topic_display_names.json` in ~40s from the real corpus with no API call. Diff the
   bytes against the committed file. It reproduces exactly *when the provenance date still
   matches* (`date.today()` is stamped into the payload), so this check has a shelf life —
   after the date rolls over, compare the JSON with `provenance.date` removed.
2. **Recompute the published statistics.** Assemble the pipeline by hand
   (`load_arms` → `load_crosswalk` → `project_to_legacy` → `agreement_table` ×2 →
   `agreement_drivers`) instead of calling `triangulate.run()`, which persists two parquets
   into `data/`. Every figure in `notes/topic-comparison-report-v1.md` §0/§6/§7 falls out of
   that one call.

**Why:** the numbers in the reports are quoted downstream and pinned by tests, and the
`data/` artifacts are frozen outputs of paid runs. A green suite does not prove a
published figure held.

**How to apply:** run check 2 before and after any edit inside `triangulate.py`; run check 1
before and after any edit touching the names-file write path. Both are read-only if you keep
outputs in the scratchpad — confirm with `git status --short data/` afterwards.

Related: [[run-commands]], [[verify-before-rewriting-justifications]]
