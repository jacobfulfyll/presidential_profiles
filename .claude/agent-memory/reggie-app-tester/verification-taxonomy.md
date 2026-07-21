---
name: verification-taxonomy
description: How to verify the corpus-taxonomy module (presidential_profiles.taxonomy) and its committed artifacts without spending money
metadata:
  type: reference
---

Verifying `presidential_profiles.taxonomy` (era-stratified sampling -> per-era LLM proposal -> merge -> crosswalk -> held-out coverage). The REAL paid run already happened; its artifacts are committed. Complements [[verification-workflow]] (site build) and [[verification-embed-clusters]] (clusters). Same money-guard discipline as the annotate layer in [[verification-workflow]].

**MONEY-PATH GUARD (verified):** `python -m presidential_profiles.taxonomy` (no flags) makes PAID Opus+Sonnet calls — never run it. The `--dry-run` flag is provably $0: in `run()`, the `dry_run` branch `return`s BEFORE `client = _client()`. `_client()` is the ONLY anthropic construction and the ONLY `import anthropic` in the module (inside the function body), so the dry-run path never imports or constructs a client. Run it with the key absent: `env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN`. Args: `--dry-run --run-id --per-era --seed --cost-ceiling` (default ceiling $12). Note the paid Opus proposal/merge/crosswalk/anachronism calls use `thinking={"type":"adaptive"}` on purpose; only the bulk Sonnet coverage labeler sets `thinking={"type":"disabled"}` (the silent-cost trap from [[verification-workflow]]).

**Dry-run smoke:** prints `[sample] 720 training paragraphs across 9 eras; 200 held-out (100 unlabeled)` then a summary JSON with `eras` 1770..2010 (nine 30-yr bands, 80 each). Writes NOTHING — `git status` stays clean afterward (dry_run returns before any `_atomic_write_json`).

**Committed artifacts (in `data/llm_annotations/`):**
- `taxonomy_v1.json`: 17 level-1 domains (12 policy + 5 non-policy), 50 level-2 topics. Each level-2 topic carries BOTH `exemplars` (short sample-ids like p0084) AND resolved `exemplar_paragraphs` ([{doc_name, para_idx}]) — consumers use the latter. 149 exemplar pairs total; all 50 topics have >=1. Resolving pairs against `data/paragraphs.parquet` (key `(doc_name, para_idx)`) yields on-topic real paragraphs.
- `crosswalk_v1.json`: all 16 `CROSSWALK_ISSUES` (15 legacy + "Security & peace") map to >=1 level-2 topic; no dangling topic refs.
- `manifests/taxonomy-v1-20260719.json`: round-trips via `llm_annotations.read_manifest("taxonomy-v1-20260719")`. cost_usd 2.4404, model `claude-opus-4-8 (+ claude-sonnet-5 for coverage)`, coverage 100% (no merge revision).

**Fingerprint provenance check (the key end-to-end proof):** `llm_annotations.corpus_fingerprint()` computed fresh over the current corpus == the manifest's `corpus_fingerprint` exactly: `{n_speeches: 1057, n_paragraphs: 36229, doc_name_sha256: 23eff144...9bcc}`. Match proves the taxonomy was derived from THIS corpus.

**Full suite:** 236 passed in ~4s (62 taxonomy tests within it). Import-sweep `import presidential_profiles.{profiles,issues,llm_annotations,embed_topics,taxonomy}` all OK.
