---
name: verification-agreement-layer
description: How to verify the inter-model (Opus vs Sonnet) agreement layer end-to-end, offline, against real data
metadata:
  type: reference
---

Verifying `presidential_profiles.agreement` (inter-model-agreement-check task). All checks are
OFFLINE/$0 — never run `pp-annotate submit/status/ingest` (paid). Import must resolve to the
worktree: `PYTHONPATH=src arch -x86_64 <mainrepo>/.venv/bin/python -c "import presidential_profiles as m; print(m.__file__)"`.

**Sample determinism (criterion: refuse + reproduce):** `agreement.draw_sample(force=False)` raises
`FileExistsError` before any write (guard is first statement) — safe to call against the committed
path. To prove seed reproduces membership, redraw to a scratch path with `force=True` and compare
`set(res["doc_names"])` to the committed JSON — should be identical (266 speeches, seed 20260721).
The committed FILE, not the seed, is the source of truth. `agreement.sample_doc_names()` returns the
266; corpus fingerprint sha256 `23eff144...` matches current 1057-speech corpus.

**Second-opinion routing:** non-default `--model` writes to per-model-suffixed parquets
(`paragraph_annotations__opus4-8.parquet` etc.), NEVER the primary tables. Verify primary untouched:
row counts 36,229 / 1,057 / 27,214 and NO `opus-*` run_ids in them. Opus paragraph key
`(doc_name,para_idx)` is unique; entities key is NOT unique in EITHER table (multi-entity per para) —
that's expected, not a dup bug.

**Coverage math:** sample = 9,048 paragraphs across 266 speeches. Opus covered 8,570 (94.72%). The
inner join equals all 8,570 opus rows (keys identical, no orphans). Composition: 238 fully covered,
24 PARTIALLY covered (370 paras dropped by array-collapse), 4 fully missing (108 paras). The report
WARNING says "4 speeches missing" — accurate for fully-missing, but the 24 partial speeches are the
larger slice of the gap; the 94.72% paragraph figure is the authoritative one.

**kappa-0 is the near-empty-class phenomenon, not a data error:** party_attack pre-1848 kappa=0.000
because Opus flagged ZERO positives (constant rater → sklearn returns 0). Base rates: 1789-1818
primary 1/242, opus 0/242; 1819-1848 primary 15/978, opus 0/978. General pattern: Opus is
SYSTEMATICALLY more conservative than Sonnet on all combativeness flags across all eras (flags ~⅓ to
⅕ as often in low cells) — genuine inter-model threshold divergence, values all schema-valid, NOT a
prompt/schema misread. Top-10 disagreements read as genuine ambiguity + Opus occasionally assigning
era-mismatched topic labels (e.g. "Early Naval Wars" on a 1976 speech) — model-quality curiosity,
not a bug.

**GOTCHA — qa --converged mutates a committed file:** running `pp-annotate qa --converged` rewrites
`notes/annotation-qa-v1.md`. The GATE lines are deterministic (byte-identical), but Amendment #2's
chunk-escalated-straggler list is derived from the GITIGNORED ephemeral `data/llm_annotations/runs/`
state — which now holds only opus runs (primary chunk-run state was cleaned up), so it regenerates to
"none" and produces a spurious diff. `git checkout -- notes/annotation-qa-v1.md` after. The primary
gates themselves (coverage 100/100, party 6.56% / enemy 21.07% / zero_sum 7.68% non-degenerate,
entity-in-para 85.81%) are unaffected — that's what criterion 8 actually cares about.

**Spend:** three opus manifests sum to $30.5211 ($1.6142 + $21.5415 + $7.3654). `--max-cost-usd` is
an ESTIMATE-time gate, not a hard billing cap — judgment round billed $21.54 vs its $21 cap. Watch
for ceiling breaches vs the recorded authorization.
