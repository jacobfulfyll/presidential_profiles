---
name: verification-workflow
description: How to run tests and build the presidential_profiles static site during VERIFY-APP stage
metadata:
  type: reference
---

Verification recipe for the `presidential_profiles` static-site generator.

**Interpreter:** the repo venv is x86_64 under Rosetta, living at the MAIN repo path (worktrees have no `.venv` of their own):
`arch -x86_64 /Users/jacobpress/Desktop/Projects/presidential_profiles/.venv/bin/python`

**Test suite** (run from worktree root): `... -m pytest -v`. pyproject sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, so no PYTHONPATH needed for pytest.

**Full site build:** `PYTHONPATH=src arch -x86_64 <venv-python> -m presidential_profiles.site` (console script `pp-site` → `presidential_profiles.site:main`). Expected output: `docs/index.html`, 45 profile pages + index under `docs/presidents/`, `docs/explorer.html`, 16 issue pages + index under `docs/issues/`, `docs/compare.html`.

**Build is deterministic — re-confirmed 2026-07-21 at 220 docs + 10 outputs files.** A full rebuild produced a **zero-line diff**; `git status --porcelain docs/ outputs/` stayed empty. So a *dirty* `docs/` after a rebuild is a real signal that generated content changed, never noise. This is the single most useful regression check in the repo.

> Task briefings sometimes assert "figures regenerate with byte-level differences every run, so expect diff noise — discard it with `git checkout -- docs/`". That was **false** when tested. Always rebuild and *look* before accepting a licence to discard churn: if the build really is deterministic, blanket-discarding would throw away a genuine content regression. Test the claim, don't inherit it.

**The generator never prunes.** It writes pages but does not delete stale ones. Rename/unsurface a topic and the old `docs/issues/<old-slug>.html` stays on disk, merely unlinked from `index.html`. So "the page still exists" is NOT evidence a topic is still surfaced — check `docs/issues/index.html`'s `href=` list and the topic count in `docs/explorer/meta.json` instead.

**`docs/explorer/meta.json` is mtime-guarded** against `word_families.FAMILIES_PATH` only, so issue/topic-label changes never trigger an explorer rebuild. `rm -f docs/explorer/meta.json` before building whenever verifying anything label-related, or you will inspect a stale payload. Its `topics` dict is keyed by **display** name, which makes it the cleanest single assertion for "what does the site actually surface".

**Content sanity-check (the real regression test for text↔label pairing bugs):** issue pages under `docs/issues/*.html` render three quotes each (Early / At its peak / Most recent) with a `President, Title, Year` citation; profile pages under `docs/presidents/*.html` render per-issue cards each with a verbatim quote + citation. Extract these and confirm each quote is on-topic for its labeled issue and the citation is internally consistent.

**Gitignore note:** `.pytest_cache/` self-ignores (pytest writes its own `.gitignore` inside), so it won't show as untracked. `.claude/agent-memory/` is agent infra, not a project artifact.

**LLM annotation layer (`pp-annotate`, module `presidential_profiles.annotate`):** verify $0 by prefixing every invocation with `env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN`. `dry-run` is fully offline — it never imports anthropic or constructs a client (that only happens on `--count-tokens`/`submit`/`status`/`ingest`), so it runs clean with no creds. Full-corpus dry-run (no `--limit`) is the headline: 1,057 requests (1/speech), 36,229 paragraphs, ~26 MB JSONL (cap 256 MB), ~$8.53 (batch −50%, intro pricing through 2026-08-31). It writes `data/llm_annotations/runs/<run_id>/{requests.jsonl,requests_index.json,estimate.json}` — that whole `runs/` dir is gitignored (line 17); only the parquet + `manifests/*.json` are tracked. `submit` has two independent money-gates that both fire *before* the client is constructed (annotate.py:580): refuses without `--yes`, and refuses if estimate > `--max-cost-usd` (default $25). `status`/`submit`/`ingest` require `--run-id`; `status` on a dry-run dir exits 1 cleanly ("no batch_id") since dry-run writes no state.json. Real requests must set `thinking={"type":"disabled"}` (Sonnet-5 defaults to adaptive thinking = silent cost) and must NOT carry temperature/top_p/top_k/budget_tokens/fallbacks. Clean up probe runs with `rm -rf data/llm_annotations/runs/<id>`; never delete the committed `invocation_tone.parquet`/manifest. `pp-analyze`/`pp-site` do not import the annotate or llm_annotations modules — the paid layer is fully decoupled from the free rebuild path.
