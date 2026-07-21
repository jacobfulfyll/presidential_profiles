# Tasks

Direction: `notes/convergence-investigation-direction.md`
Adversarial review of the convergence design (13 confirmed / 9 partial findings) reshaped this
backlog on 2026-07-13 — see `.pipeline/convergence-analysis/task.md` for what changed and why.

## Active Tasks

### inter-model-agreement-check
**Task**: Opus 4.8 second pass on a persisted 25% sample; publish disagreement
**Pipeline**: code-workflow
**Branch**: task/inter-model-agreement-check
**Worktree**: .worktree/inter-model-agreement-check
**Base**: master
**Started**: 2026-07-21
**Files**:
- NEW: src/presidential_profiles/agreement.py
- MOD: src/presidential_profiles/annotate.py
- NEW: data/llm_annotations/agreement_sample_v1.json
- NEW: data/llm_annotations/agreement_v1.parquet
- NEW: notes/agreement-report-v1.md

---

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

### issue-attention-over-time
**Task**: Topic birth, death, and revival across 240 years
**Pipeline**: code-workflow
**Branch**: task/issue-attention-over-time
**Worktree**: .worktree/issue-attention-over-time
**Base**: master
**Started**: 2026-07-21
**Files**:
- NEW: src/presidential_profiles/attention.py
- NEW: data/attention/topic_lifecycles.parquet
- NEW: notes/attention-findings-v1.md

### breadth-depth-register
**Task**: Did the formal record get broader, shallower, more values-driven?
**Pipeline**: code-workflow
**Branch**: task/breadth-depth-register
**Worktree**: .worktree/breadth-depth-register
**Base**: master
**Started**: 2026-07-21
**Files**:
- NEW: src/presidential_profiles/register.py
- NEW: data/register/trends.parquet
- NEW: notes/register-findings-v1.md

### combativeness-over-time
**Task**: Where does today land against the 1860s and 1930s?
**Pipeline**: code-workflow
**Branch**: task/combativeness-over-time
**Worktree**: .worktree/combativeness-over-time
**Base**: master
**Started**: 2026-07-21
**Files**:
- NEW: src/presidential_profiles/combat.py
- NEW: data/combat/combativeness.parquet
- NEW: notes/combativeness-findings-v1.md

---

## Backlog

### Analysis & Findings
- [ ] era-atlas: Era fingerprints, similarity matrix, data-driven periodization, LLM portraits [P2] [complex] [tier: opus:high] [code] [planned] [depends: breadth-depth-register, issue-attention-over-time, combativeness-over-time]
  files: src/presidential_profiles/eras.py (NEW), data/eras/era_fingerprints.parquet (NEW), data/eras/era_similarity.parquet (NEW), notes/era-atlas-v1.md (NEW)
- [ ] convergence-analysis: Pre-registered test of agenda convergence (rebuilt after adversarial review) [P3] [complex] [tier: opus:high] [code] [planned] [depends: era-atlas, topic-method-comparison]
  files: src/presidential_profiles/convergence.py (NEW), notes/convergence-prereg-v1.md (NEW), notes/convergence-findings-v1.md (NEW), data/convergence/ (NEW)

### Site Presentation
- [ ] topic-chart-upgrades: Confidence bands (sampling + annotator disagreement) on issue trend charts [P3] [moderate] [tier: opus:medium] [design] [planned] [depends: inter-model-agreement-check] [conflicts: profile-issue-views]
  files: src/presidential_profiles/bands.py (NEW), src/presidential_profiles/site.py (MOD), src/presidential_profiles/issues_site.py (MOD)
- [ ] centralize-issue-display-names: `issues + ["Discovered 5"]` is hardcoded in five modules (`profiles.py:281`, `profiles_site.py:334`, `explorer.py:118`, `issues_site.py:247`, `site.py:361` + `site.py:1121`). Deferred from `profile-issue-views` (its acceptance criterion 4) at PICKUP on 2026-07-16: the "centralized names file" that criterion assumed does not exist and is blocked behind `topic-method-comparison`. Note task.md called this "the two hardcoded sites" — it is five. [P3] [moderate] [code] [depends: topic-method-comparison] **LARGELY SUPERSEDED 2026-07-21**: `topic-method-comparison` centralized all six append sites onto `data/topic_display_names.json` behavior-preservingly, and `taxonomy.CROSSWALK_ISSUES` (added 2026-07-20) already provides the centralized legacy-issue list this entry says "does not exist". Residual scope only: the 5 *other* `"Discovered 5"` literals that were deliberately left alone — `profiles_site.py` `DISCOVERED_LABELS`, `explorer.py` inline label ternary, `profiles.py` `_EXTRA_ANCHORS` + `_WAR_SPEC`. Re-groom before picking up; the line numbers in this entry are stale (`profiles.py` is 340 not 281, `profiles_site.py` is 379 not 334).

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
- [ ] education-page-reconstructs-diffusely: The corpus-derived taxonomy_v1 has no standalone Education level-2 topic — the crosswalk maps the Education issue page onto Welfare/Poverty & Economic Opportunity, Prosperity/Jobs & the Middle Class, and Civil Rights/Voting Rights & Discrimination. A genuine corpus finding (schooling was framed inside opportunity/welfare), but the Education page will reconstruct more diffusely than the other 15 — needs a reviewer's eye before `run-llm-annotation-pass` freezes downstream expectations. (IMPLEMENT, derive-corpus-taxonomy, 2026-07-19) **CONFIRMED WITH NUMBERS 2026-07-21** (topic-method-comparison): Education has the **lowest LLM↔CorEx agreement of all 16 issues** (Jaccard 0.123, κ 0.170) despite having the *second-highest* anchor coherence (NPMI 0.372) — i.e. its poor agreement is a taxonomy gap, not a measurement artifact. `taxonomy_v1.json` is frozen, so this is a known, quantified limitation of every Education-page finding until a taxonomy v2.
- [ ] annotation-pass-label-validation: In the taxonomy-v1 held-out coverage run, 2 of 292 label assignments (~0.7%) came back as topic strings not exactly matching a level-2 name. `run-llm-annotation-pass` must validate model-returned labels against the frozen level-2 name set (exact-match + reject/repair loop) rather than trusting free-text. (IMPLEMENT, derive-corpus-taxonomy, 2026-07-19)
- [ ] discovered-topics-1-7-unnamed: `data/paragraph_issues.parquet` carries `Discovered 1–7` columns, but only `Discovered 5` has a surfaced identity ("Security & peace"); 1–4, 6, 7 remain unnamed free topics. Not a bug — an observation for whoever curates the discovered family next. (IMPLEMENT, derive-corpus-taxonomy, 2026-07-19) **Update (IMPLEMENT, breadth-depth-register, 2026-07-21):** the `Discovered 5` → "Security & peace" mapping lives ONLY in `taxonomy.py`'s `SECURITY_PEACE` constant, and `docs/issues/security-and-peace.html` is published off it — so a consumer reading the parquet alone cannot tell what any `Discovered N` column means. Column-level provenance belongs with the data, not in one module's constant.
- [ ] manifest-no-structured-failure-counts: `llm_annotations.Manifest` has no structured failure-count fields — failure counts live only in the free-text `notes` string, so QA must re-read the cached `results.jsonl` to get schema-failure rates. Fine while the raw cache exists; a Manifest field would make failure rates durable. (IMPLEMENT, run-llm-annotation-pass, 2026-07-20)
- [ ] taxonomy-rerun-provenance-hardening: Three low-severity items for the NEXT taxonomy regeneration (frozen v1 unaffected — `merge_revised: false`, 50 unique names, 0 anachronisms): (a) `REVISE_SYSTEM` is a money-path prompt outside `_all_prompts()`/`_prompt_hash` — record it in the provenance `prompts` dict when `merge_revised` is True, or fold into the hash and re-freeze; (b) `duplicate_level2_names` is report-only — consider promoting to a `bail_reasons` condition (same structural-integrity class as `orphan_parents`) since a colliding pair would hand `run-llm-annotation-pass` an ambiguous label key while passing the gate; (c) `anachronisms_flagged` remains surfaced-not-enforced per the plan's manual-revision design. (REVIEW round 2, derive-corpus-taxonomy, 2026-07-19)
- [ ] entity-type-nation-includes-us-states: `paragraph_entities.parquet` types seceding US states (South Carolina, Virginia, Texas, Georgia, Arkansas, North Carolina, Tennessee) and Wisconsin as `nation` — 17 of 179 Civil War-era adversarial `nation` entities. Defensible in context (they claimed sovereignty), but it silently biases any foreign/domestic split built on `type`. Affects a FROZEN paid artifact, so it needs documenting (or a derived-column correction), never an edit in place. `combativeness-over-time` disclosed it and showed the robustness check cuts in its own finding's favor. (IMPLEMENT, combativeness-over-time, 2026-07-21)
- [ ] eras-bands-mask-regime-changes: `trends.ERAS` bands average over within-era regime changes — "War & New Deal" (1933-1945) fuses a domestic-attack decade (1930s: party_attack 0.181, foreign enemy-naming 0.010) with a foreign-enemy decade (1940s: 0.056 / 0.217), two nearly inverse profiles collapsed into one point. Same issue dilutes the 1860s with Reconstruction. Directly relevant to `era-atlas`, whose job is data-driven periodization — this is evidence the fixed bands are the wrong axis for some questions. (IMPLEMENT, combativeness-over-time, 2026-07-21)
- [ ] bca-bootstrap-for-thin-strata: `combat.py` uses a percentile bootstrap, which is uncentred where a genre stratum is thin — the Civil War genre-standardized `enemy_naming` point (0.215) sits near the upper edge of its own interval [0.129, 0.261] because a 3-speech stratum carries 10.4% of the reference weight. BCa (bias-corrected accelerated) would correct it. Disclosed in the report rather than fixed; worth revisiting if genre-standardized intervals get load-bearing downstream. (IMPLEMENT, combativeness-over-time, 2026-07-21)
- [ ] display-issues-dead-constant: `profiles_site.py:17` declares `DISPLAY_ISSUES = None  # filled from meta at build time` — it is assigned and never read anywhere. Dead since before topic-method-comparison; left in place because that task's boundary allowed display-name wiring only. Delete it. (IMPLEMENT, topic-method-comparison, 2026-07-21)
- [ ] union-projection-inflates-wide-crosswalk-issues: `triangulate.py` projects LLM level-2 topics onto the legacy 15 by **union** (a paragraph counts for issue L iff ANY of its topics is in L's crosswalk row). For issues with a wide crosswalk fan-out this inflates apparent LLM breadth, and fan-out correlates negatively with agreement (rho = -0.47 vs kappa, p = 0.074 — suggestive, not significant at n=15). Concrete consequence: part of the "rename" signal for Civil rights & race in the 1770 era is fan-out via `indian affairs`/`elections` topics, not vocabulary euphemism. An argmax or weighted projection would give different numbers; the union choice is documented in the report but never sensitivity-tested. Worth a second projection as a robustness check before `convergence-analysis` leans on these numbers. (IMPLEMENT, topic-method-comparison, 2026-07-21)
- [ ] marker-regexes-era-biased: `indices.py:MARKERS` regexes are era-biased and undocumented as such. `mechanism` (a depth proxy) leans on `appropriat*`, `statute`, `clause`, so its ~5x decline may be lexical drift rather than rhetorical change; `opponents` includes `fake news` / `radical left`, which cannot fire before ~1990 and therefore *guarantee* a monotone rise. Every marker-based trend in `notes/register-findings-v1.md` inherits this. Fix direction: date-stamp each regex's earliest plausible firing and either split pre/post-coinage variants or publish the vocabulary-availability window next to the trend. (IMPLEMENT, breadth-depth-register, 2026-07-21)
- [ ] register-imports-drag-in-corex: `register.py` transitively imports `corextopic` and `model2vec` (~1.7s) purely to read the constants `ERA_SPAN` / `LEGACY_ISSUES`, via `taxonomy.py -> issues.py`. `taxonomy.py` imports `issues.py` only for `PARA_LABELS_PATH`. Moving that one path constant to `corpus.py` breaks the chain for every downstream module. (IMPLEMENT, breadth-depth-register, 2026-07-21)
- [ ] speech-type-other-bucket-silently-dropped: `speech_annotations.speech_type` has an `other` bucket of 8 speeches that no genre treatment can place — they are counted in the `raw` arm but silently excluded from both `sotu_only` and `genre_standardized`. Small, but it means the three arms are not computed on identical denominators and nothing says so. Either surface the excluded count per arm or assign them explicitly. (IMPLEMENT, breadth-depth-register, 2026-07-21)
- [ ] zero-topic-paragraphs-cluster-in-modern-speeches: `zero_topic_share` reaches 0.041 in the 2010 era (raw, CI [0.003, 0.097]); the 404 zero-topic paragraphs concentrate in a few modern non-SOTU speeches rather than spreading evenly. That pattern looks more like per-document annotation failure than genuinely topic-free text. Cross-check the affected doc_names against `notes/annotation-qa-v1.md` before any downstream task treats zero-topic as a substantive signal. (IMPLEMENT, breadth-depth-register, 2026-07-21)
- [ ] taxonomy-v1-missing-prohibition-and-terrorism: `taxonomy_v1` has no Prohibition topic and no standalone Terrorism topic, so two of five pre-registered ground-truth checks in `issue-attention-over-time` were unevaluable — though both concerns are demonstrably in the corpus (35 Prohibition paragraphs in 1918-33, 26 absorbed by `Crime, Insurrection & Federal Law Enforcement`, which runs 4.3-5.2% in 1917-31 vs ~1.0-1.7% before; the 9/11 discontinuity is located correctly but inside a topic bundling four decades of intervention). Candidate v2 splits. (IMPLEMENT, issue-attention-over-time, 2026-07-21)
- [ ] crosswalk-v1-one-directional: 7 of 50 level-2 topics have no legacy parent in `crosswalk_v1.json` (`Civil Service Reform & the Merit System`, `Constitutional Union & Federalism`, `Executive Departments...`, `Partisan Combat...`, `Personal Narrative...`, `Presidential Humility...`, `Territorial Organization...`). `check_crosswalk` only asserts every legacy *issue* maps to >=1 topic, never the reverse, so any LLM<->CorEx comparison silently loses those 7. (IMPLEMENT, issue-attention-over-time, 2026-07-21)
- [ ] duplicate-paragraph-topic-pairs: ~722 duplicate (paragraph, topic) pairs survive in `paragraph_annotations` after case normalization (52,855 raw assignments -> 52,133 distinct). Related to but distinct from `annotation-pass-label-validation`: that item is about labels off the taxonomy, this is about the same label repeated on one paragraph. `attention.py` de-duplicates locally; the storage layer does not. (IMPLEMENT, issue-attention-over-time, 2026-07-21)
- [ ] anachronistic-labels-mechanically-detectable: Stray impossible (topic, year) pairs in `paragraph_annotations` — an 1861 Lincoln paragraph labeled `World War II Military Operations`, an 1838 Van Buren paragraph labeled `Great Depression Recovery`, `Cold War` in 1847, `WWI` in 1864. Seven topics have a historically impossible raw first appearance. Each level-2 taxonomy entry already carries `era_of_birth`, so these are mechanically detectable in a cheap sweep. `issue-attention-over-time`'s substantive-year threshold is what stops them shipping as findings — nothing upstream catches them. (IMPLEMENT, issue-attention-over-time, 2026-07-21)
- [ ] require-full-merge-is-private-and-expensive: `taxonomy._require_full_merge` is the repo's canonical keyed-merge guard per CLAUDE.md, but it is private-by-name and lives in a module that transitively imports `corextopic` + sklearn — so every consumer pays a heavy import to follow the documented convention. Candidate: move to a shared public utility module. Overlaps `register-imports-drag-in-corex`; fix together. (IMPLEMENT, issue-attention-over-time, 2026-07-21)
