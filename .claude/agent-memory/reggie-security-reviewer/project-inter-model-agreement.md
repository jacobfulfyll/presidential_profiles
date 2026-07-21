---
name: project-inter-model-agreement
description: Security posture of the inter-model-agreement-check task — agreement.py is no-network/no-paid; annotate.py --model/--sample additions preserve gate ordering + add two resume-identity guards
metadata:
  type: project
---

`inter-model-agreement-check` (worktree branch, SECURITY PASS 2026-07-21) adds a second-opinion Opus 4.8 annotation pass and a per-field/per-era agreement report. Reviewed clean.

**`agreement.py` (NEW) is a NON-paid, NO-network analysis module** — imports numpy/pandas/`sklearn.metrics.cohen_kappa_score`, `json.loads`, `pd.read_parquet` only. No anthropic import, no `.create()`, no eval/exec/pickle/subprocess/yaml. All write paths are module constants (`SAMPLE_PATH`, `AGREEMENT_PATH`, `REPORT_PATH`) or operator CLI args (`--out`, `--sample`) — same accepted operator-supplied posture as `run_id` / taxonomy `--out`, no model-controlled string reaches a path.

**Money path stays entirely in `annotate.py`; the `--model`/`--sample` additions preserve the existing gate discipline.** `_resolve_model` validates `--model` against the `MODELS` allowlist (SystemExit otherwise); `_table_name(spec_name, model)` builds output names from a fixed spec name + a hardcoded `MODEL_TABLE_SUFFIX` dict, so a non-default model writes SEPARATE per-model-suffixed parquets and can never clobber the primary tables (whose loaders raise on duplicate keys). In `cmd_submit` the order is: allowlist-resolve → recorded-model guard → recorded-sample-sha guard → build → resume/sealed filters → `_validate_request` (thinking-disabled guard now keyed `params['model'] in MODELS`, covers Opus too) → max-cost brake → `--yes` brake → THEN `anthropic.Anthropic()` construction → `.create()`. `import anthropic` sits at the top of `cmd_submit` but importing ≠ constructing (no creds/network); construction stays behind all brakes, consistent with conftest's `anthropic_construction_bomb`.

**Two new resume-identity guards (M1 fix, commit 0340966) that `--force` does NOT bypass:** a resubmission that forgets `--model` (would silently fall back to sonnet) or drops/swaps `--sample` (would silently widen to the full corpus) raises SystemExit via `state['model']` / `state['sample_docs_sha256']` mismatch. `--force` only bypasses the "already has batch_id" re-submit check, not model/sample/cost/yes.

**New state fields `sample_path` + `sample_docs_sha256`** live in `state.json` under `RUNS_DIR = data/llm_annotations/runs/` (gitignored, .gitignore line 17) — a repo-relative path + a hex digest, no secret material; they do NOT propagate into the committed manifest. Committed `agreement_sample_v1.json` = version/seed/fraction/bins/doc_names (public URL slugs) + corpus fingerprint sha256 — 0 secret hits.

**One LOW (future-proofing, not blocking):** `_render_top_disagreements`/`_metric_block` interpolate model-supplied strings (topics list, `proposal_values`, `entity (stance)`) RAW into `notes/agreement-report-v1.md`. Fine today — it's a human-read `notes/` markdown file, NOT part of the `docs/` GitHub-Pages output, and the downstream charts task consumes the parquet TABLE not the markdown. Per the repo's trusted-provenance posture ([[project-attack-surface]]) model/taxonomy strings are trusted. IF this report is ever rendered to HTML on the public site, these model strings need `html.escape()` like the corpus quotes already get in profiles.py. Corpus paragraph text quoted in the report is public-domain speech (trusted); backtick-wrapped `doc_name` is a public slug.
