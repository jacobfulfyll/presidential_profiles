# era-atlas / eras.py — build + gotchas (learned 2026-07-21)

## Environment (cost me real time)
- **The venv lives in the MAIN repo, not the worktree.** `.worktree/<slug>/.venv/bin/python`
  does NOT exist — invoke `/Users/.../presidential_profiles/.venv/bin/python` (absolute)
  with `export PYTHONPATH=$PWD/src` from the worktree. Confirm `import presidential_profiles`
  resolves inside `.worktree/...`.
- **Paid runs need the key sourced:** `set -a; source .env.local; set +a` (the file holds
  `export ANTHROPIC_API_KEY=...`). Documented in taxonomy.py's module header; the shell env
  does NOT carry it by default.

## File-boundary trap
- **`llm_annotations.write_manifest` HARDCODES `data/llm_annotations/manifests/`** (via that
  module's own `MANIFESTS_DIR`). Reusing it from a new module silently writes the manifest into
  the frozen paid-annotation area — a file-boundary violation. For a new module, build the
  `Manifest` dataclass and `write_text(json.dumps(asdict(m), indent=2, sort_keys=True))` to your
  OWN dir. (Caught only by `git status data/llm_annotations/`.)

## Input-schema facts (measure before aggregating)
- `speech_markers.parquet` marker columns are raw **COUNTS**, not rates — convert to per-1k-word
  rates via `sum(count)/sum(n_words)*1000`. Same for `speech_stats` `i_count`/`we_count`/`modal_*`
  (counts → per-1k-token).
- `paragraph_annotations` carries per-paragraph `party_attack`/`enemy_naming`/`zero_sum` flags AND
  `proposal_values` (proposal/values/neither/mixed) — register is computable per unit directly.
- **combat.parquet `raw` treatment is entirely `ci_status=ok`;** suppression (`suppressed_n_floor`,
  9 cells) only hits `genre_standardized`/`sotu_only`. Per-unit raw flag means == combat `raw`
  rate EXACTLY (max abs diff 0.0) — validate against it and you satisfy the ci_status gate cleanly
  without touching a suppressed cell.
- **`trends.parquet` `unit=='era'` rows use 30-year `taxonomy.ERA_SPAN` bands** (period
  1770,1800,…2010), NOT `trends.ERAS`. Do not reuse them for a trends.ERAS analysis — recompute
  per-era measures yourself. (Same warning as CLAUDE.md's method_agreement note, different table.)

## Reuse confirmed
- Topic normalization: `attention.canonical_label_map` + `attention.level1_parents` +
  `attention.normalize_topics` (raises on unmapped, de-dups case variants). `attention.era_series`
  for year→trends.ERAS. Do NOT roll a third normalizer.
- Detrend: `similarity.build_adjusted`'s local-mean-within-ERA_WINDOW(=24) pattern ports directly
  to fingerprints. At a coarse grain (9 eras) a fixed 24y window captures only self — adapt to
  adjacent-unit subtraction and say so.

## Determinism
- `sklearn.AgglomerativeClustering(linkage='ward', connectivity=chain_graph)` is deterministic and
  gives temporally-CONTIGUOUS clusters (a path-graph connectivity forces it) → cluster boundaries
  ARE period boundaries. Avoid MDS (random, needs seed). No wall-clock in the meta json (combat
  precedent) → `git status data/eras/` dirty means numbers moved.

## Findings-note arithmetic
- Re-verify every DERIVED number, not just base values pulled from parquet: a "roughly five times"
  multiplier was actually 10.8×, and a stated range endpoint (+0.34) was really the #1 cosine
  (+0.315). Recompute ratios/spreads/min-max, don't eyeball them.
