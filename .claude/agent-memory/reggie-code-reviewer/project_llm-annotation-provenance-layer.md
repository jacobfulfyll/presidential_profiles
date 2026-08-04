---
name: llm-annotation-provenance-layer
description: The LLM-annotation derived-data layer (llm_annotations.py + annotate.py / pp-annotate) and its review-calibration values
metadata:
  type: project
---

`src/presidential_profiles/llm_annotations.py` (storage/manifest/migration) and
`src/presidential_profiles/annotate.py` (`pp-annotate` Batches CLI) form the
LLM-derived-data layer, added in task `build-annotation-provenance-layer`
(2026-07). Annotations are keyed by `doc_name` / `(doc_name, para_idx)` (never
row order), every row carries a `run_id` -> JSON manifest, and `corpus_fingerprint`
hashes the sorted doc_name list + row counts.

`invocation_tone.json` was MIGRATED to `invocation_tone.parquet` + a manifest and
the source JSON was **deliberately deleted** (git history only). Consequence:
`migrate_invocation_tone(force=True)` cannot re-derive from the working tree —
the tests exercise it via `tests/fixtures/invocation_tone.json` with a
monkeypatched `INVOCATION_TONE_JSON` path instead.

**Why (review calibration):** In this repo, provenance / reproducibility is a
*hard value* (same falsification ethic as [[convergence-investigation-direction]]).
Weight reproducibility and "can a future maintainer audit how this byte was
produced" gaps higher than you would on a typical CRUD codebase.

**How to apply:** When reviewing this layer, the highest-value bug is a
silently-wrong parquet or an unstable fingerprint. Money-path guards
(cost gates, atomic results cache, stale-batch refetch, thinking-disabled) are
extensively pre-vetted — if a task brief says they're vetted, don't re-litigate;
spend effort on loaders, migration, manifest round-trip, and the FieldSpec seam.
