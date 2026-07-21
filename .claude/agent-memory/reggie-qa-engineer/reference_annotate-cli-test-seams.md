---
name: annotate-cli-test-seams
description: Hermetic-test seams for presidential_profiles annotate.py (pilot, QA gating, speech-unit e2e, cross-batch accumulation, sealing) — the non-obvious hooks
metadata:
  type: reference
---

Authoring offline tests for `annotate.py` cmd_* (the paid annotation pass). Builds on
[[test-toolchain]]'s forging idiom (conftest `forge_run`/`patch_anthropic`/`args`).

- **Hermetic corpus for submit/QA**: cmd_submit and `_compute_qa` read the REAL 36k
  corpus via `A.load` (imported into annotate as a module name) and
  `ann.PARAGRAPHS_PATH`. Monkeypatch BOTH to tiny synthetic frames for full control:
  `monkeypatch.setattr(A, "load", lambda: df)` and
  `monkeypatch.setattr(ann, "PARAGRAPHS_PATH", tmp_parquet)`. The autouse
  `redirect_annotation_dirs` only redirects the write-side dirs, NOT these read paths.
- **cmd_qa writes to REAL repo by default**: `QA_REPORT_PATH = REPO_ROOT/notes/
  annotation-qa-v1.md`. Every cmd_qa test MUST pass `out=str(tmp_path/...)` or it
  dirties the worktree. Args fixture now carries `pilot`/`resubmit_sealed`/`out`.
- **Speech-unit forged result** (unit="speech", e.g. `speech_annotations`): the text
  block is the payload object DIRECTLY (`{"speech_type":...}`), NOT wrapped in an
  `annotations` array — that wrapper is paragraph-unit only. `forge_run.succeeded_line`
  is paragraph-shaped; write speech lines by hand. Ingest keys the row by doc_name.
- **Cross-batch accumulation**: manifest spend accumulates via `state.batch_usage`
  (keyed by batch_id). To simulate `submit --force` between two ingests, call
  `A._write_state(run_id, batch_id="B", results_batch_id="B", submitted_custom_ids=[...])`
  (a merge — preserves batch_usage), then overwrite `results.jsonl`. Re-ingesting the
  same batch replaces its entry (idempotent), so cost never double-counts.
- **Sealing**: `invalid_request` failures -> `state.sealed_permanent`; other error types
  (overloaded/expired) are NOT sealed. submit's resume drops sealed doc_names unless
  `resubmit_sealed=True`. Assert via `A._custom_id("<spec>", doc)` membership in the
  post-submit `state.submitted_custom_ids`.
- **run_id sanitization** lives in `main()` (rejects `/ \ ..` via `parser.error`, exit
  code 2, stderr "must not contain"). Test through `main()` with
  `monkeypatch.setattr(sys, "argv", [...])`; assert `not ann.RUNS_DIR.exists()` to prove
  it fired before any dir was touched.
- **Pilot snapshot**: `load()` is only 1057 rows — cheap enough to pin the exact 20
  `_pilot_speeches` doc_names as a real-corpus snapshot (Washington 1796 -> Biden 2021;
  note the last doc_name lacks the `/the-presidency/...` prefix). Add a synthetic-frame
  test for stratification + order-independence; assert the shuffle actually reordered
  before asserting order-invariance (else the reorder claim is vacuous).
- **Few-shot enum guard**: parse `av1._JUDGMENT_FEWSHOTS` with
  `re.findall(r"^ANNOTATION: (\{.*\})$", ..., re.M)`; assert each topic ∈ the frozen 50
  read straight from `taxonomy_v1.json` (not via `av1.TOPIC_NAMES`, so a hardcoded
  divergent list is caught). The schema topic enum == that exact 50-name list.

**Verified 2026-07-20 against IMPLEMENT checkpoint 706359c: all 9 targets, no impl bugs.**
