---
name: verification-era-atlas
description: How to verify eras.py (era-atlas — fingerprints/similarity/periodization/portraits) end-to-end and re-derive notes/era-atlas-v1.md; the residual-drift traps that survive the pipeline's own number sweeps
metadata:
  type: reference
---

Verifying `presidential_profiles.eras` + `notes/era-atlas-v1.md`. Deterministic build ≈ 3s over
36,229 paragraphs; portraits are a FROZEN paid artifact ($0.0595) — only `portraits --dry-run` is
runnable.

**Invocation** (see [[verification-workflow]] for the Rosetta interpreter, [[verification-combat]]
for the sibling module). CLI: `python -m presidential_profiles.eras` (default = deterministic
build, prints headline nearest-neighbor summary) and `... eras portraits --dry-run` (estimate only,
$0, prints `ceiling $5.00`). `run()` regenerates all 7 parquets + `eras_meta.json` byte-identically;
`git status --short data/eras/` stays clean on rerun. The dry-run also calls `run()` and rewrites the
deterministic `portraits_estimate.json` (also byte-identical) but NEVER touches `era_portraits.parquet`
or `manifests/` — confirm by sha+mtime, not just git.

**$0 proof for the dry-run**: `generate_portraits(dry_run=True)` returns before the `import anthropic`
line; assert `'anthropic' not in sys.modules` after a dry-run, and that portrait/manifest mtimes are
unchanged. Real run refuses without `--yes` and above the `PORTRAIT_COST_CEILING_USD=5.00` ceiling.

**Staleness-guard live-fire** (the `derived-tables-go-stale` guard): `check_staleness(fingerprint=None)`
reads `eras.COMBAT_META_PATH`. Live-fire WITHOUT touching the real file — `shutil.copy` combat_meta.json
to scratch, corrupt `corpus_fingerprint`, monkeypatch `E.COMBAT_META_PATH = tmp`, call `check_staleness()`:
returns `match=False` and logs `WARNING "STALE INPUT: ... does not match the live corpus"`. Restore the
module constant and re-sha the real file (per [[feedback_perturb-restore-discipline]]).

**Seam state**: every row of `era_similarity` / `president_similarity` / `nearest_neighbors` carries
`ci_components="sampling_only"` + `agreement_source="absent:agreement_v1.parquet"`;
`data/llm_annotations/agreement_v1.parquet` must stay absent.

**Note re-derivation — two methodology traps:**
1. **Exclude the self-similarity diagonal.** `era_similarity.parquet` includes `unit_i==unit_j`
   (cosine 1.0). Forgetting to drop it makes every "nearest neighbor / rank / #1→#2 gap" wrong (self
   ranks #1). Filter `unit_j != unit_i` before ranking anything.
2. **After the numeric cells all check out, the residual drift is in ORDINAL descriptors and
   CHARACTERIZATION glosses** — the deepest sub-class of CLAUDE.md's "sentences with a bare number".
   On this note every cosine/count/boundary/Spearman/periodization cell re-derived EXACTLY (0.315,
   0.264 gap, 10.8×, Spearman 0.70, 6-of-9 / 0-of-9 adjacency, 6-of-8 bin / 4-of-8 president distinct
   canonicals, LOO-identical at every k both grains, $0.0595). The only defects were: an off-by-one
   ordinal ("third-from-last" for a neighbor that is 4th-from-last — the *cosine* beside it was
   right), a gloss that covered 2 of 3 exceptions ("the 3 exceptions are same-century ties" — the
   third crosses 20th→21st c by center-year), and a provenance-location claim (manifest "contains
   exemplar IDs" — they live in the portraits parquet, which the manifest's own `notes` field states).
   These survive the pipeline's own number-drift sweeps because they are not bare digits. Grep the
   note for every ordinal word (`third-from-last`, `second`, `last`) and every "N of M ... are X"
   generalization and re-derive the *ordinal/predicate*, not just the value next to it.

**Portrait traceability** reproduces cleanly: each portrait's claims (incl. NEGATIVE-z axes read as
absences — "hope-language nearly disappears" ← marker_nrc_hope −2.0) trace to that era's top |z| axes
in `era_fingerprints`; all 9×4 exemplar `(doc_name, para_idx)` triples exist in `data/paragraphs.parquet`.
