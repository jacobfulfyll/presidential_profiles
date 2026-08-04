---
name: paid-derived-module-recipe
description: Audit shape for this repo's PAID variant of the derived-analysis module (era-atlas) — the extra checks beyond the $0 recipe when an Anthropic path is present
metadata:
  type: project
---

`era-atlas` (`eras.py`, 2026-07-21) is the first derived-analysis module with a PAID Anthropic
path (LLM era portraits) rather than pure $0 compute. Extends [[derived-analysis-module-recipe]];
the $0 checks still apply to its deterministic `run()`. What the paid variant adds:

**Spend-control chain to verify (all present + test-enforced in era-atlas — the reference shape):**
- CLI: `--dry-run`/`--run` in a `mutually_exclusive_group`, `--yes` required to spend. `dry_run =
  not args.run` (so bare `--dry-run` or neither → safe default). Guard order in the paid function:
  dry-run early-return ($0) → `est_cost > CEILING` raise → `not yes` raise → ONLY THEN
  `import anthropic; anthropic.Anthropic()`. The zero-arg constructor is the ONLY key read (SDK
  env mechanism); there is no dotenv loader in the repo, so `conftest`'s key-deletion is airtight.
- The `anthropic_construction_bomb` fixture (conftest) makes `Anthropic()` raise if built; the
  dry-run / no-yes / over-ceiling tests all take it, proving no client is constructed before a
  gate fires. If a paid module lacks this fixture on its guard tests, that's a gap.
- Post-hoc actual-cost tripwire after the run, and a partial manifest written on mid-loop failure
  recording ACCUMULATED ACTUAL cost (invariant: a manifest can never claim cheaper-than-actual).

**Two findings the shape produced (both reported, neither a vuln / neither FAILs):**
1. **No-client placeholder clobber (MEDIUM, integrity).** On `--run --yes` with the key NOT
   sourced, the graceful seam writes `era_portraits.parquet` with `status=not_generated` —
   OVERWRITING the committed real paid portraits — and writes NO manifest, leaving the existing
   generated-run manifest desynced from a not_generated parquet. Same frozen-paid-artifact clobber
   class as [[cli-path-arg-containment]] (git-recoverable, operator already has FS write) → MEDIUM,
   not a security bug. Fix direction: refuse to overwrite an existing `generated` artifact with
   placeholders (or gate behind an explicit flag).
2. **Usage accounted AFTER the response-parse line (LOW, provenance).** Loop order is
   `create()` → `texts[era] = "".join(b.text ...)` → `usage += msg.usage.*`. A billed 200 whose
   content fails to parse raises before the usage `+=`, so the partial manifest undercounts that
   billed request. Trigger prob is tiny (SDK 200s are well-formed) but it's the exact
   "usage read before/after the response that raised?" question — fix is to move the two usage
   `+=` lines immediately after `create()`, before any parsing.

**Confirmed-clean, do not re-flag:** `run()` (deterministic) never writes the portraits parquet or
manifests, so the free rebuild can't clobber the paid artifact; `_write_manifest` targets
`data/eras/manifests/` NOT `llm_annotations.write_manifest` (that misroute was caught pre-commit);
run_id is date-derived only (no era-NAME-derived filenames); CLI has NO `--out` path arg (the clean
`register.py` shape, not the path-containment pattern).
