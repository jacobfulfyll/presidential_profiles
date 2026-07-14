# Tasks

Direction: `notes/convergence-investigation-direction.md`
Adversarial review of the convergence design (13 confirmed / 9 partial findings) reshaped this
backlog on 2026-07-13 — see `.pipeline/convergence-analysis/task.md` for what changed and why.

## Active Tasks

### build-annotation-provenance-layer
**Task**: LLM derived-data layer, manifests, pp-annotate CLI, invocation_tone migration
**Pipeline**: code-workflow
**Branch**: task/build-annotation-provenance-layer
**Worktree**: .worktree/build-annotation-provenance-layer
**Base**: master
**Started**: 2026-07-13
**Files**:
- NEW: src/presidential_profiles/llm_annotations.py
- NEW: src/presidential_profiles/annotate.py
- MOD: pyproject.toml
- NEW: data/llm_annotations/

---

## Backlog

### Data Foundation
- [ ] discover-corpus-taxonomy: Derive a two-level corpus-native taxonomy + crosswalk to the legacy 15 [P1] [complex] [tier: opus:high] [code] [planned] [depends: build-annotation-provenance-layer, fix-paragraph-issues-key]
  files: src/presidential_profiles/embed_topics.py (NEW), src/presidential_profiles/taxonomy.py (NEW), data/llm_annotations/taxonomy_v1.json (NEW), data/llm_annotations/crosswalk_v1.json (NEW)
- [ ] run-llm-annotation-pass: Annotate 36k paragraphs + 1057 speeches via Sonnet 5 batch (<=$50) [P1] [complex] [tier: opus:high] [code] [planned] [depends: discover-corpus-taxonomy]
  files: src/presidential_profiles/prompts/annotation_v1.py (NEW), src/presidential_profiles/annotate.py (MOD), data/llm_annotations/paragraph_annotations.parquet (NEW), data/llm_annotations/speech_annotations.parquet (NEW), data/llm_annotations/paragraph_entities.parquet (NEW)
- [ ] inter-model-agreement-check: Opus 4.8 second pass on a persisted 25% sample; publish disagreement [P2] [moderate] [tier: opus:medium] [code] [planned] [depends: run-llm-annotation-pass]
  files: src/presidential_profiles/agreement.py (NEW), data/llm_annotations/agreement_sample_v1.json (NEW), data/llm_annotations/agreement_v1.parquet (NEW), notes/agreement-report-v1.md (NEW)
- [ ] build-word-families: Group word forms (immigrant/immigrants/immigration) into one node in the word graph [P2] [complex] [tier: opus:high] [code] [planned] [conflicts: fix-paragraph-issues-key, topic-chart-upgrades]
  files: src/presidential_profiles/word_families.py (NEW), src/presidential_profiles/explorer.py (MOD), src/presidential_profiles/trends.py (MOD), src/presidential_profiles/site.py (MOD), pyproject.toml (MOD), data/word_families.json (NEW), notes/word-families-review.md (NEW)

### Analysis & Findings
- [ ] topic-method-comparison: LLM vs CorEx label agreement, coherence scoring, cluster naming [P2] [moderate] [tier: opus:medium] [code] [planned] [depends: run-llm-annotation-pass] [conflicts: fix-paragraph-issues-key]
  files: src/presidential_profiles/triangulate.py (NEW), src/presidential_profiles/topic_quality.py (NEW), src/presidential_profiles/issues.py (MOD)
- [ ] breadth-depth-register: Did the formal record get broader, shallower, more values-driven? [P2] [complex] [tier: opus:high] [code] [planned] [depends: run-llm-annotation-pass]
  files: src/presidential_profiles/register.py (NEW), data/register/trends.parquet (NEW), notes/register-findings-v1.md (NEW)
- [ ] issue-attention-over-time: Topic birth, death, and revival across 240 years [P2] [moderate] [tier: opus:medium] [code] [planned] [depends: run-llm-annotation-pass, discover-corpus-taxonomy]
  files: src/presidential_profiles/attention.py (NEW), data/attention/topic_lifecycles.parquet (NEW), notes/attention-findings-v1.md (NEW)
- [ ] combativeness-over-time: Where does today land against the 1860s and 1930s? [P2] [moderate] [tier: opus:medium] [code] [planned] [depends: run-llm-annotation-pass]
  files: src/presidential_profiles/combat.py (NEW), data/combat/combativeness.parquet (NEW), notes/combativeness-findings-v1.md (NEW)
- [ ] era-atlas: Era fingerprints, similarity matrix, data-driven periodization, LLM portraits [P2] [complex] [tier: opus:high] [code] [planned] [depends: breadth-depth-register, issue-attention-over-time, combativeness-over-time]
  files: src/presidential_profiles/eras.py (NEW), data/eras/era_fingerprints.parquet (NEW), data/eras/era_similarity.parquet (NEW), notes/era-atlas-v1.md (NEW)
- [ ] convergence-analysis: Pre-registered test of agenda convergence (rebuilt after adversarial review) [P3] [complex] [tier: opus:high] [code] [planned] [depends: era-atlas, topic-method-comparison]
  files: src/presidential_profiles/convergence.py (NEW), notes/convergence-prereg-v1.md (NEW), notes/convergence-findings-v1.md (NEW), data/convergence/ (NEW)

### Site Presentation
- [ ] topic-chart-upgrades: Confidence bands (sampling + annotator disagreement) on issue trend charts [P3] [moderate] [tier: opus:medium] [design] [planned] [depends: inter-model-agreement-check] [conflicts: profile-issue-views, build-word-families]
  files: src/presidential_profiles/bands.py (NEW), src/presidential_profiles/site.py (MOD), src/presidential_profiles/issues_site.py (MOD)
- [ ] profile-issue-views: Raw attention + era-relative + "topic of the day" flag on profiles [P3] [moderate] [tier: opus:medium] [design] [planned] [conflicts: topic-chart-upgrades]
  files: src/presidential_profiles/profiles.py (MOD), src/presidential_profiles/profiles_site.py (MOD)

### Ungroomed
Discovered during build-annotation-provenance-layer (2026-07-14). Not fixed — logged only.
- [ ] migrate-invocation-tone-not-rerunnable: `migrate_invocation_tone(force=True)` reads `invocation_tone.json`, which the migration deletes — so `force=True` on a clean checkout raises FileNotFoundError. Recovering the source needs `git show`. The sharp edge of "delete the original, git history preserves it."
- [ ] submit-double-loads-corpus: `annotate.py cmd_submit` re-derives the corpus fingerprint with a second `load()` + `read_parquet` after `_prepare()` already loaded both — a redundant full corpus load on the paid path.
- [ ] invocation-windows-bleed-across-speeches: `profiles.invocations()` computes tone windows on the *concatenated* per-president transcript, so a window near a speech boundary can pull text from an adjacent speech. Affects window text only, not match counts.
- [ ] nondeterministic-plot-output: `outputs/figures/*.png` and `outputs/interactive/*.html` are tracked but regenerate with byte-level differences on every `pp-analyze` run, so any task that runs the pipeline shows unrelated diff noise.
- [ ] nrc-lexicon-refetch-in-worktree: `pp-site` re-downloads the NRC Emotion Lexicon in a fresh worktree because `data/raw/` is gitignored — a network fetch in an otherwise offline pipeline.
- [ ] resume-manifest-accumulation: On `submit --force` under an EXISTING run_id, `cmd_ingest` overwrites `manifests/<run_id>.json` with only the last batch's `cost_usd`/`batch_id`/`n_requests` — understating cumulative spend and mis-attributing earlier rows (violates the layer's "a manifest can never claim a run was cheaper than it was" invariant). Bounded: the new-run_id-per-pass idiom is already correct. Fix options: refuse `--force` when the manifest exists, scope the manifest by batch_id, or accumulate across ingests. Address in `run-llm-annotation-pass` — the first task to exercise a real cross-batch resume. (reggie-judge, IMPLEMENT re-review, 9.22 PASS)
