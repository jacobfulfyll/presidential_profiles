# Tasks

Direction: `notes/convergence-investigation-direction.md`
Adversarial review of the convergence design (13 confirmed / 9 partial findings) reshaped this
backlog on 2026-07-13 — see `.pipeline/convergence-analysis/task.md` for what changed and why.

## Active Tasks

### topic-method-comparison
**Task**: LLM vs CorEx label agreement, coherence scoring, cluster naming
**Pipeline**: code-workflow
**Branch**: task/topic-method-comparison
**Worktree**: .worktree/topic-method-comparison
**Base**: master
**Started**: 2026-07-21
**Files**:
- NEW: src/presidential_profiles/triangulate.py
- NEW: src/presidential_profiles/topic_quality.py
- MOD: src/presidential_profiles/issues.py

---

## Backlog

### Data Foundation
- [ ] inter-model-agreement-check: Opus 4.8 second pass on a persisted 25% sample; publish disagreement [P2] [moderate] [tier: opus:medium] [code] [planned] [depends: run-llm-annotation-pass]
  files: src/presidential_profiles/agreement.py (NEW), data/llm_annotations/agreement_sample_v1.json (NEW), data/llm_annotations/agreement_v1.parquet (NEW), notes/agreement-report-v1.md (NEW)

### Analysis & Findings
- [ ] breadth-depth-register: Did the formal record get broader, shallower, more values-driven? [P2] [complex] [tier: opus:high] [code] [planned] [depends: run-llm-annotation-pass]
  files: src/presidential_profiles/register.py (NEW), data/register/trends.parquet (NEW), notes/register-findings-v1.md (NEW)
- [ ] issue-attention-over-time: Topic birth, death, and revival across 240 years [P2] [moderate] [tier: opus:medium] [code] [planned] [depends: run-llm-annotation-pass, derive-corpus-taxonomy]
  files: src/presidential_profiles/attention.py (NEW), data/attention/topic_lifecycles.parquet (NEW), notes/attention-findings-v1.md (NEW)
- [ ] combativeness-over-time: Where does today land against the 1860s and 1930s? [P2] [moderate] [tier: opus:medium] [code] [planned] [depends: run-llm-annotation-pass]
  files: src/presidential_profiles/combat.py (NEW), data/combat/combativeness.parquet (NEW), notes/combativeness-findings-v1.md (NEW)
- [ ] era-atlas: Era fingerprints, similarity matrix, data-driven periodization, LLM portraits [P2] [complex] [tier: opus:high] [code] [planned] [depends: breadth-depth-register, issue-attention-over-time, combativeness-over-time]
  files: src/presidential_profiles/eras.py (NEW), data/eras/era_fingerprints.parquet (NEW), data/eras/era_similarity.parquet (NEW), notes/era-atlas-v1.md (NEW)
- [ ] convergence-analysis: Pre-registered test of agenda convergence (rebuilt after adversarial review) [P3] [complex] [tier: opus:high] [code] [planned] [depends: era-atlas, topic-method-comparison]
  files: src/presidential_profiles/convergence.py (NEW), notes/convergence-prereg-v1.md (NEW), notes/convergence-findings-v1.md (NEW), data/convergence/ (NEW)

### Site Presentation
- [ ] topic-chart-upgrades: Confidence bands (sampling + annotator disagreement) on issue trend charts [P3] [moderate] [tier: opus:medium] [design] [planned] [depends: inter-model-agreement-check] [conflicts: profile-issue-views]
  files: src/presidential_profiles/bands.py (NEW), src/presidential_profiles/site.py (MOD), src/presidential_profiles/issues_site.py (MOD)
- [ ] centralize-issue-display-names: `issues + ["Discovered 5"]` is hardcoded in five modules (`profiles.py:281`, `profiles_site.py:334`, `explorer.py:118`, `issues_site.py:247`, `site.py:361` + `site.py:1121`). Deferred from `profile-issue-views` (its acceptance criterion 4) at PICKUP on 2026-07-16: the "centralized names file" that criterion assumed does not exist and is blocked behind `topic-method-comparison`. Note task.md called this "the two hardcoded sites" — it is five. [P3] [moderate] [code] [depends: topic-method-comparison]

### Ungroomed
Discovered during build-annotation-provenance-layer (2026-07-14). Not fixed — logged only.
- [ ] migrate-invocation-tone-not-rerunnable: `migrate_invocation_tone(force=True)` reads `invocation_tone.json`, which the migration deletes — so `force=True` on a clean checkout raises FileNotFoundError. Recovering the source needs `git show`. The sharp edge of "delete the original, git history preserves it."
- [ ] submit-double-loads-corpus: `annotate.py cmd_submit` re-derives the corpus fingerprint with a second `load()` + `read_parquet` after `_prepare()` already loaded both — a redundant full corpus load on the paid path.
- [ ] invocation-windows-bleed-across-speeches: `profiles.invocations()` computes tone windows on the *concatenated* per-president transcript, so a window near a speech boundary can pull text from an adjacent speech. Affects window text only, not match counts.
- [ ] nondeterministic-plot-output: `outputs/figures/*.png` and `outputs/interactive/*.html` are tracked but regenerate with byte-level differences on every `pp-analyze` run, so any task that runs the pipeline shows unrelated diff noise.
- [ ] nrc-lexicon-refetch-in-worktree: `pp-site` re-downloads the NRC Emotion Lexicon in a fresh worktree because `data/raw/` is gitignored — a network fetch in an otherwise offline pipeline.
- [ ] resume-manifest-accumulation: On `submit --force` under an EXISTING run_id, `cmd_ingest` overwrites `manifests/<run_id>.json` with only the last batch's `cost_usd`/`batch_id`/`n_requests` — understating cumulative spend and mis-attributing earlier rows (violates the layer's "a manifest can never claim a run was cheaper than it was" invariant). Bounded: the new-run_id-per-pass idiom is already correct. Fix options: refuse `--force` when the manifest exists, scope the manifest by batch_id, or accumulate across ingests. Address in `run-llm-annotation-pass` — the first task to exercise a real cross-batch resume. (reggie-judge, IMPLEMENT re-review, 9.22 PASS)
- [ ] permanent-failures-not-terminally-sealed: A permanent `invalid_request` batch failure writes no rows, so `_already_ingested` treats it like a retryable one — the next `submit` re-requests and **re-pays** for a request that will deterministically fail again. The permanent/retryable classification has machine effect nowhere; it lives only in manifest `notes` + stdout. May be intended (human fixes the malformed spec before resubmit), but it's a real re-bill edge on the paid path. Decide in `run-llm-annotation-pass`. (QUALITY-CHECK, 2026-07-14)
- [ ] speech-unit-ingest-untested-e2e: No test drives a `unit="speech"` spec through `build_requests`/`cmd_ingest` (the branch keying speech-level rows by `doc_name` alone and writing their manifest). Currently unreachable via the CLI — the shipped registry has only the paragraph `placeholder` spec — so not urgent, but an end-to-end ingest test should land when `run-llm-annotation-pass` adds a real speech-unit spec. (QUALITY-CHECK, 2026-07-14)
- [ ] sanitize-run-id-cli-arg: `run_id` flows unsanitized into `_run_dir(run_id)` and `manifest_path` (`RUNS_DIR / run_id`, `MANIFESTS_DIR / f"{run_id}.json"`); a `run_id` of `../../x` or an absolute path escapes `data/llm_annotations/`. Operator-supplied, not a trust boundary (they could write the file directly), so a fat-finger/UX guard, not a vulnerability — reject `run_id` containing `/`, `\`, or `..` early with a clear message. Low priority; `run-llm-annotation-pass` touches `annotate.py`. (SECURITY-REVIEW LOW, 2026-07-14)
- [ ] topic-of-day-param-units-hazard: `is_topic_of_day(raw_strength, rel)` (`profiles.py:312-316`) mixes a normalized raw strength (1.0 = threshold) with an unnormalized rel in percentage points — the two float params look interchangeable but aren't; a caller passing the normalized rel from `issue_strengths` would silently shrink the flag band. Correct at the single call site and covered by tests; a param rename (`rel_pp`) would remove the hazard. (reggie-judge, VERIFY-APP profile-issue-views, 2026-07-19)
- [ ] presidents-index-era-relative-only-framing: After profile-issue-views, profile cards rank by strongest-on-either-axis (raw or era-relative), but the presidents INDEX still (a) ranks its per-president badge by `-rel` alone (`profiles_site.py` render_index) and (b) carries subtitle copy "the issues they pressed harder than their contemporaries … Card badge: the issue they emphasized most vs their era" (`profiles_site.py:362-364`) — so a president whose index badge reads "↑ Trade" can open to a profile led by a raw-axis issue like War & military. Pre-existing behavior, deliberately out of scope for profile-issue-views (index code behavior-unchanged by that diff). Fix together: either label the badge "most distinctive vs era" and refresh the subtitle, or rank the badge by the same max-axis strength profiles use. (REVIEW MINOR + SYNC-DOCS judge, profile-issue-views, 2026-07-19)
- [ ] education-page-reconstructs-diffusely: The corpus-derived taxonomy_v1 has no standalone Education level-2 topic — the crosswalk maps the Education issue page onto Welfare/Poverty & Economic Opportunity, Prosperity/Jobs & the Middle Class, and Civil Rights/Voting Rights & Discrimination. A genuine corpus finding (schooling was framed inside opportunity/welfare), but the Education page will reconstruct more diffusely than the other 15 — needs a reviewer's eye before `run-llm-annotation-pass` freezes downstream expectations. (IMPLEMENT, derive-corpus-taxonomy, 2026-07-19)
- [ ] annotation-pass-label-validation: In the taxonomy-v1 held-out coverage run, 2 of 292 label assignments (~0.7%) came back as topic strings not exactly matching a level-2 name. `run-llm-annotation-pass` must validate model-returned labels against the frozen level-2 name set (exact-match + reject/repair loop) rather than trusting free-text. (IMPLEMENT, derive-corpus-taxonomy, 2026-07-19)
- [ ] discovered-topics-1-7-unnamed: `data/paragraph_issues.parquet` carries `Discovered 1–7` columns, but only `Discovered 5` has a surfaced identity ("Security & peace"); 1–4, 6, 7 remain unnamed free topics. Not a bug — an observation for whoever curates the discovered family next. (IMPLEMENT, derive-corpus-taxonomy, 2026-07-19)
- [ ] manifest-no-structured-failure-counts: `llm_annotations.Manifest` has no structured failure-count fields — failure counts live only in the free-text `notes` string, so QA must re-read the cached `results.jsonl` to get schema-failure rates. Fine while the raw cache exists; a Manifest field would make failure rates durable. (IMPLEMENT, run-llm-annotation-pass, 2026-07-20)
- [ ] taxonomy-rerun-provenance-hardening: Three low-severity items for the NEXT taxonomy regeneration (frozen v1 unaffected — `merge_revised: false`, 50 unique names, 0 anachronisms): (a) `REVISE_SYSTEM` is a money-path prompt outside `_all_prompts()`/`_prompt_hash` — record it in the provenance `prompts` dict when `merge_revised` is True, or fold into the hash and re-freeze; (b) `duplicate_level2_names` is report-only — consider promoting to a `bail_reasons` condition (same structural-integrity class as `orphan_parents`) since a colliding pair would hand `run-llm-annotation-pass` an ambiguous label key while passing the gate; (c) `anachronisms_flagged` remains surfaced-not-enforced per the plan's manual-revision design. (REVIEW round 2, derive-corpus-taxonomy, 2026-07-19)
