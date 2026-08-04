# Presidential Profiles

## Environment
- The `.venv` is x86_64 under Rosetta. Always invoke Python as `arch -x86_64 .venv/bin/python`
  (or `arch -x86_64 uv run ...`) — a plain `uv run`/`python` invocation on Apple Silicon can
  resolve to the wrong architecture's interpreter.
- The venv is uv-managed and has no `pip`; install packages with `uv pip install ...` (with
  `VIRTUAL_ENV` pointed at `.venv`) rather than `python -m pip`.
- Running from a **git worktree** (e.g. `.worktree/<slug>/`): the editable install's `.pth`
  hardcodes the **main repo's** `src`, so a bare `python`/`pytest` imports main-repo code even
  when your cwd is the worktree — you'd silently test the wrong source. Set
  `PYTHONPATH=<worktree>/src` (pytest already prepends the relative `src`), and confirm with
  `python -c "import presidential_profiles as m; print(m.__file__)"` before trusting results.

## Data conventions
- Paragraph-level tables (`data/paragraphs.parquet`, `data/paragraph_issues.parquet`) share a
  real `(doc_name, para_idx)` key. Always join on it (`validate="one_to_one"`) — never assume
  row order lines the two tables up, even though it happens to hold today. This replaced a
  positional-alignment bug that could silently mislabel every paragraph on reorder.
- `validate="one_to_one"` catches duplicate keys but NOT diverging key *sets* — an inner join
  silently drops non-matching rows. When co-derived tables must stay row-complete, also assert
  merged length == each input's length after the join (see `taxonomy.py::_require_full_merge`).
- Artifacts under `data/llm_annotations/` (e.g. `taxonomy_v1.json`, `crosswalk_v1.json`, and the
  full-corpus annotation parquets `paragraph_annotations` / `speech_annotations` /
  `paragraph_entities`) are frozen, provenance-stamped outputs of paid LLM runs — never
  regenerate or hand-edit them casually; each has a manifest in `data/llm_annotations/manifests/`
  recording models, prompts, sample IDs, and actual cost. The annotation QA report + its
  pre-registration amendments live at `notes/annotation-qa-v1.md`.
- **`paragraph_annotations.topics` needs two normalizations before any counting**, and skipping
  either silently fragments real topics. (a) **Case variants**: 58 distinct label strings appear
  against `taxonomy_v1`'s 50 level-2 names — the 8 extras are pure case drift ("… & The
  Environment", "War Of 1812"). `{name.casefold(): name}` over the 50 names is 1:1 and maps all of
  them with zero residue; **raise on any unmapped label** rather than warn, since one means the
  taxonomy and the annotations have diverged. (b) **Duplicate `(paragraph, topic)` pairs**: 722 of
  them survive normalization, so 52,855 raw assignments are 52,133 distinct (mean 1.439 topics per
  paragraph, not 1.46). Quote the post-dedup figure. See `attention.py::normalize_topics`.
- `data/attention/` and `data/combat/` are the opposite case: **derived, deterministic, and safe to
  regenerate** with
  `python -m presidential_profiles.combat` (pure local compute over the frozen parquets, $0, zero
  API calls). All 10 outputs are byte-identical on rerun **on any date** — a wall-clock stamp was
  deliberately removed from `combat_meta.json` so that a dirty `git status data/combat/` means the
  numbers actually moved, not that the clock did. Keep it that way; provenance identity is the
  `corpus_fingerprint`, and *when* it ran is git's job.
- `ci_status` (in `data/combat/{combativeness,peak_decades,ratios}.parquet`) is the
  machine-readable **trust gate** for downstream consumers — `era-atlas` depends on this table.
  Never read `rate`/`ratio` without it: `suppressed_n_floor` = too few speech clusters for an
  interval to mean anything, `low_cluster_caution` = an interval exists but is thin. A consumer
  that plots `rate` blind will publish the n=3 War & New Deal SOTU cell as a peak.
- Era-grain analyses reuse `trends.ERAS` (9 named eras) rather than inventing a periodization —
  data-driven periodization belongs to `era-atlas`, and two competing era axes would make every
  cross-task number unjoinable.
- When a dependency's artifact does not exist yet (parallel worktrees make this normal), build the
  **seam, not a stub**: an optional loader that returns `None` when the file is absent, plus
  columns on every output row recording that the component is missing (see
  `combat.load_agreement_bands` / `ci_components="sampling_only"`). Never fabricate, stub, or
  hardcode a magnitude, and never create the other task's file.
- `data/bands.parquet` + `data/bands_meta.json` (the confidence bands under the issue/topic trend
  charts) are the same derived, deterministic, **$0** class as `data/attention/`, `data/combat/`
  and `data/eras/`: regenerate with `python -m presidential_profiles.bands` — pure local compute
  over the frozen parquets, **zero API calls**, byte-identical on rerun with **no wall-clock
  stamp**, so a dirty `git status data/bands.parquet` means the numbers moved, not that the clock
  did. Two surfaces, two uncertainty budgets: `corex_issues` rows are permanently
  `ci_components="sampling_only"` because those labels come from a seeded CorEx topic model with
  **no LLM annotator anywhere in the pipeline** — annotator disagreement is *categorically
  inapplicable* there, not merely absent, which is a different thing from the missing-dependency
  seam above. `llm_topics` rows carry `sampling+annotator_disagreement`.
- **`bands.parquet` has TWO trust gates, at two different grains.** `ci_status` is per-PERIOD by
  construction (`_ci_status` takes only `n_speeches`/`n_paragraphs`, so all 22 series in a period
  share it) and its docstring forbids overloading it — two definitions of one column would make it
  unreadable to the consumer that keys on it, which is why this became a new column rather than a
  new `ci_status` value. `interval_unresolvable` is per-CELL: the bootstrap ran, returned a
  **zero-width** interval, and the period had fewer than `MIN_CLUSTERS_FOR_RESOLVABLE_CI` = 4 speech
  clusters. Those 12 cells (all at 1785) publish `lo`/`hi`/`lo_sampling`/`hi_sampling` as **null**
  and keep `point` — the share is real, only the interval was unknowable — and the charts draw a
  **hollow ring** so a non-hovering reader sees the difference. **A null bound is not a zero
  bound**: a consumer that fills or coerces null `lo`/`hi` to `0.0` re-creates exactly the
  fake-precise cell this column exists to withdraw. **The gate is a conjunction and must stay
  one**: zero-width alone would suppress the 7 legitimate confident zeros at 1790/1795/1800/1810
  (13-14 speeches that genuinely never touched the issue — a finding); too-few-clusters alone would
  suppress 1785's Religion & values (0.0-66.7%), whose two speeches disagree and whose enormous
  band is the honest signal. The floor of 4 is **derived, not chosen** — `n**-n`, the probability of
  the most concentrated resample, first drops below `CI_LOW/100` at n=4, so that is the first n at
  which a 2.5th percentile can exclude anything — and it is *computed* by
  `bands._min_clusters_for_resolvable_ci`, so retuning `CI_LOW` moves the floor with it rather than
  silently invalidating a hardcoded 4. `MIN_CLUSTERS_FOR_CI` stays at 2, because raising it deletes
  1785 outright.
- Two counting traps in that block, both of the bare-count drift class this repo keeps hitting.
  (a) **12 flagged cells render 11 rings** — `Discovered 4` has `surface: false` in
  `topic_display_names.json`, so it has no page and no panel. 12 ≠ 11 is correct, not a shortfall.
  (b) **19 is a pre-gate census, not a property of the shipped table**: before the gate there were
  19 zero-width `corex_issues` cells (7 kept + 12 withdrawn); the artifact on disk holds **7**.
  Quoting 19 as what the table contains is a tense error, and it has already been made twice.
- **Suppressing a zero-width band is a no-op for the reader** — it already occupied zero pixels.
  Any future "hide the untrustworthy cell" change must add a *positive* marker, or it changes the
  data and nothing on the page.
- **`agreement_v1.parquet` is deliberately NOT read by `bands.py`** — a future consumer tempted to
  "wire up the agreement table" needs to know this was considered and rejected on units, not
  overlooked. The fastest check: the table has **no `disagreement_half_width` column at all** — its
  schema is `field / era_bin / era_label / metric / value / n` (`agreement.py:370`), and
  `combat.AGREEMENT_REQUIRED_COLUMNS` names `{era, flag, disagreement_half_width}`, a contract the
  real file has never met (so `combat.load_agreement_bands` now *raises* on it rather than
  returning `None`). Beyond the missing column: its topic metric is a **jaccard**, with no
  conversion into percentage points of
  paragraph share (any factor would be a fabricated magnitude), and it is keyed on
  `taxonomy.ERA_SPAN`, not `trends.ERAS`. Disagreement is instead measured directly as
  `|share_primary − share_secondary| / 2` over the 8,570 paragraphs both annotators labeled, on the
  `trends.ERAS` reporting axis — same quantity, same units, same axis, no cross-axis mapping. That
  paired set is a **document-level** sample (262 of the 266 sampled speeches; 8,570 of the corpus's
  36,229 paragraphs, 23.7%) and its disagreement is *assumed to transfer* to full-corpus eras;
  within-speech clustering is modelled in the sampling bootstrap but **not** in the half-width.
- `ci_status` on `bands.parquet` is the trust gate under the same doctrine as `data/combat/` —
  never read `lo`/`hi` without it, and (since `interval_unresolvable` landed) never assume they are
  present at all. But its constants **shadow `combat.py`'s names with different
  values** (`MIN_CLUSTERS_FOR_CI` 2 vs 5, `LOW_CLUSTER_CAUTION` 8 vs 20) because the grain differs:
  these guard a 5-year period, combat's guard an era-grain genre stratum. Same column, same
  direction, different thresholds — do not read one artifact's gate across the other.
- **A refusal guard must run BEFORE the write it refuses.** `bands.py`'s divergence check sat after
  `to_parquet`, so a "refused" build had already replaced the parquet and left it sitting beside a
  meta sidecar describing the previous numbers — a state strictly worse than either clean outcome,
  in a repo whose provenance rule is that a dirty `git status data/` means the numbers moved.
  Validate first, then write; and when a module emits a table plus a sidecar, they land together or
  not at all.
- **The coherence-threshold asymmetry is deliberate, not a bug.** `topic_quality.py` computes
  NPMI for all 22 CorEx topics but the noise gate (`classify_discovered`) applies to the 7
  *discovered* topics only. Four *anchored* issues — Immigration (-0.028), Foreign policy
  (0.079), Infrastructure (0.089), Civil rights & race (0.091) — actually score below every
  other coherent topic, right next to `Discovered 3` (-0.047), the corpus's one real noise
  topic. They are exempt by design: their anchor sets deliberately span vocabulary that never
  co-occurs in one paragraph (railroad and broadband are one issue, "Infrastructure", on
  purpose). Documented in `embed_topics.py`, `topic_quality.py`, and
  `issues.py::attach_coherence`, and test-enforced (`TestCoherenceAsymmetry` and siblings in
  `tests/test_topic_quality.py`) — do not "fix" it into a single global floor.
- **`method_agreement.parquet`'s `era` column is `trends.ERAS` (9 named, reporting axis), not
  `taxonomy.ERA_SPAN`** (30-year bands, a *sampling* stratification — the anachronism guard for
  taxonomy discovery, deliberately atheoretical). Any new per-era reporting table should reuse
  `trends.ERAS`, not reach for `ERA_SPAN`. On this table `era == NaN` marks the whole-corpus row;
  `triangulate.assign_eras()` raises rather than letting an out-of-band year fall through to NaN
  and be silently read as a corpus-wide statistic.
- `data/topic_display_names.json` is meant to be hand-edited, and degrades two different ways: a
  topic missing its `display` value falls back to the raw column name (`Discovered 6`) —
  lossless; a topic missing from the file, or the file missing entirely, drops out of
  `surfaced_discovered()` — safe (no crash) but lossy, since it silently removes a live site
  page. `display` strings are not currently HTML-escaped on the way into rendered `docs/` pages
  (backlogged as `escape-display-names-in-rendered-html`).
- `data/method_compositions.parquet` and `data/method_agreement.parquet` are safe to delete and
  regenerate: `triangulate.run()` is pure local computation over already-frozen parquets ($0, no
  API calls), and reproduces byte-identical output on rerun.
- `data/convergence/` is the same derived/deterministic/**$0** class (`python -m
  presidential_profiles.convergence`, ~19 min at R=2000, byte-identical on any date, no wall-clock
  stamp), with two guards the other layers lack. (a) **The three-leg `_selftest` is a bail
  condition, not a diagnostic**: it runs before `out_dir` is even resolved, and a failing leg
  aborts before a single real number exists. Skipping it needs a verbatim token AND forbids writing
  the published layer. (b) **A run below the pre-registered R=2000 refuses to overwrite
  `data/convergence/`** — pass `--out-dir`. Both exist because an under-powered or ungated number
  that reaches the published layer is indistinguishable from a real one after the fact.
- **`convergence.py`'s permutation null does NOT absorb the estimator's residual bias**, and the
  obvious claim that it does is false — it survived several review rounds inside this project
  before being caught by re-derivation. The observed curve samples each president's **in-window**
  pool (mean size 17 → 36 across the corpus, Spearman vs window centre **+0.85**); a permuted slot
  samples the donor's **global career** pool (**+0.06**). Permuting donors decorrelates pool size
  from time, destroying the supply drift that *causes* the bias. The artifact proves it in one
  column: `selftest.a_cluster_null_rho` = −0.089 vs the null's `null_mean` = +0.014 — if it were
  absorbed those would coincide. Net effect: the test is mildly **anti-conservative toward
  convergence** (~6.3% size at nominal 5%), which leaves a *no-convergence* finding conservative
  and would matter enormously to a future run that found a decline.
- Reading `data/convergence/`: `excess_ratio` is the rival-hypothesis statistic (observed ÷
  entropy-matched) and is **not** immune to a rising no-topic share, though `floor_corrected` is.
  Never compare dispersion **levels** across arms — `llm_all` has 51 bins vs the CorEx arms' 16.
  `n_windows` is deliberately absent from `permutation_null.parquet`; join it from
  `dispersion_curves.parquet`. And at D=20 the seed-to-seed sd of the primary rho is **0.032**,
  larger than the point estimate itself — so **no jackknife delta is a president effect**.

## Batches-API annotation lessons (paid, learned 2026-07-20/21)
- **Structured-output arrays "collapse"**: Sonnet 5 sometimes emits ONE complete array item and
  stops (`end_turn`, not truncation) — observed at array sizes 64, 15, 5, even 2; ~12-16% of
  requests. Prompt contracts and higher effort barely move it. Only a 1-item array is
  structurally immune. `pp-annotate submit --chunk-size N` is the convergence lever; the run
  needed the full ladder 25→10→5→2→1 to reach 100% coverage.
- **`invalid_request_error` is two different failures**: billing blocks ("credit balance" in the
  message → retryable, never seal) and genuinely malformed requests (→ sealed permanent). See
  `_classify_failure` in annotate.py; classification is by message because the API type alone
  cannot distinguish them.
- **Batch prompt caching hit ~50% at 1,000-request scale** (22-53% variance on small batches),
  so conservative no-cache estimates run ~20-30% high — the safe direction.
- **Batch `request_counts` can be stale for HOURS** (observed: 265/266 requests complete while
  the counter showed 0 succeeded for 12h). The billing dashboard is the true progress signal;
  `cancel` on a stuck-"in_progress" batch releases already-completed results and bills only those
  — both Opus agreement rounds were recovered this way losing 1 and 3 requests respectively.
- **Cost extrapolations must use MEASURED output rates, not planning guesses**: the agreement
  task's plan assumed 0.3M output tokens for a 25% sample; the production-measured rate
  (~115 tok/paragraph, baked into `JUDGMENT_EST_OUTPUT_TOKENS_PER_PARA`) put it at ~1.04M —
  a 3.5× miss that tripped the cost bail. Always dry-run against current spec constants.

## Second-opinion passes (inter-model agreement, added 2026-07-22)
- A non-default `--model` on `pp-annotate` writes to per-model-suffixed parquets
  (`paragraph_annotations__opus4-8.parquet` …) routed by `state.json`'s recorded model —
  NEVER into the primary tables (their loaders raise on duplicate keys). Keys stay identical.
- Resume identity is guarded: `state.json` records `model` + `sample_docs_sha256`; a
  resubmission round that switches model or drops/swaps `--sample` aborts before any
  corpus load or client construction. Always pass BOTH flags on ladder rounds anyway.
- Null semantics in `agreement.py` metrics (nulls are absent in practice; guards are
  crash-proofing): kappa excludes null pairs pairwise (n = compared pairs — sklearn raises
  on mixed None/bool), null topics coerce to the empty set, report renders null flags as
  `null`. The persisted sample FILE (`agreement_sample_v1.json`), not its seed, is the
  source of truth for membership.

## Paragraph-rate estimator lessons (learned 2026-07-21)
- **Cluster the bootstrap on speeches, not paragraphs.** Within-speech ICC on paragraph-level
  annotation flags runs as high as 0.17, so a paragraph-resampled interval is far too narrow.
- **Two floors, two different n's.** When an estimator is a weighted average over strata (e.g.
  `combat.py`'s genre-standardized rate), a minimum-n floor on the cell's *total* cluster count
  does **not** bind on its *effective* one: a 3-speech stratum carried ~49% of the renormalized
  weight while the cell reported `n_speeches=52` and published `ci_status="ok"`. Publish the
  effective quantity (`effective_min_cluster`, `weight_from_thin_strata`) and downgrade status on
  thin-stratum weight, not on the total. Nothing about this bug is visible in any single number —
  it survived unit tests, mutation testing, and end-to-end verification.
- **Raising the stratum floor is not the fix.** Going 3 → 5 collapsed genre coverage (War & New
  Deal `ref_weight_covered` 0.86 → 0.22), trading an interval problem for a worse
  representativeness one. Discipline the *interval*; the point estimate was never the defect.

## Era atlas layer (data/eras/, learned 2026-07-21)
- `data/eras/` follows the combat/attention pattern: derived, deterministic, $0 to regenerate
  (`python -m presidential_profiles.eras`), byte-identical on rerun. The exceptions are
  `era_portraits.parquet` + `data/eras/manifests/` — frozen PAID artifacts ($0.0595, 9 portraits,
  claude-sonnet-5). `eras.py` refuses to overwrite a `generated` portraits parquet with placeholders
  (a no-key `--run` must never silently downgrade a paid artifact).
- **Similarity consumers**: read `ci_components` before trusting any interval (`sampling_only` until
  `agreement_v1.parquet` is propagated — the file now exists but era-atlas predates it; regenerating
  with bands is a groomed follow-up). The RAW matrix tracks time by construction (Spearman 0.70 with
  temporal proximity); only the DETRENDED matrix answers rhyme questions. Present era's detrended
  nearest neighbor: Civil War & Reconstruction (0.315, mutual).
- **Count periodization recovery in DISTINCT canonical boundaries, not matching discovered rows.**
  Two discovered boundaries can hit one canonical (1881 + 1882 both match 1878 across Garfield's
  single-speech presidency) — the note shipped "5 of 8" for what is 4 of 8 distinct and was caught
  only by recomputing from `periodization.parquet`. Same bare-count drift class as ever.
- **Paid-module recipe additions** (extend the annotate.py conventions): accumulate `usage` from a
  response IMMEDIATELY after `create()` returns, before any content parsing — a billed-but-unparseable
  200 must still be counted, or the partial manifest under-reports spend. On mid-loop failure,
  persist a partial manifest (accumulated actual cost, n_succeeded, PARTIAL note) BEFORE re-raising.
  `llm_annotations.write_manifest` hardcodes its own manifests dir — a new layer writing manifests
  elsewhere must write them directly, not reuse it.

## Published research notes are a deliverable, not documentation (learned 2026-07-21)
`notes/*-findings-*.md` get read as findings. `issue-attention-over-time` shipped a note whose
tables were flawless and whose **prose contained 16 defects** — and the lesson is what they had in
common, not their number:
- **Every one was invisible to a reader of the note alone.** They surfaced only by recomputing from
  the parquet. Reviewing a findings note by reading it proves nothing; the only verification that
  works is re-deriving each number from the artifact. Budget for that, and have the note **state
  which cells were verified and which were not** — a note that declares its own boundary is worth
  more than one that implies completeness.
- **The drift concentrates in sentences containing a bare number**, never in the generated tables.
  Counts are the weakest class ("seven other topics" → 9; "eight topics" → 10; "four absolute
  births" → 2; "eighty years later" → 92). Sweep every count claim; spot-checking misses them.
- **Prose can carry the wrong *mechanism* while every number is right.** One claim attributed a
  1790 peak to annual-message enumeration when 14 of its 17 paragraphs came from a single
  non-annual-message speech — a different artifact class that genre standardization does not absorb
  either. The finding survived (a counterfactual left it at 13.7% vs an 8.06 ceiling), but the
  stated reason was wrong. Check explanations, not just values.
- **Double-rounding** (2 dp then 1 dp) has now produced one-increment-high cells in two independent
  notes. Round from the fraction in one step, and state the fraction-vs-percentage-point convention.
- A "largest N" table must actually be sorted by the quantity it ranks — one shipped omitting its
  2nd and 4th largest rows while including the 9th and 10th.
- **A provenance artifact's explanatory strings are prose, and drift the same way.** Four
  wrong-number-in-prose defects surfaced on `topic-chart-upgrades`; the one that actually shipped
  was in `data/bands_meta.json`, the **only** new prose block without a re-derivation test — the
  two blocks that had one stayed correct. Pin prose numbers wherever they live (meta JSON, module
  docstrings), not only in `notes/`.
- **Fixing a prose defect in one place is not fixing it — grep the claim everywhere first.** The
  same false sentence lived in two docstrings; the follow-on task corrected the census in one, and
  in the same paragraph one line above it shipped a *second* wrong count ("sixty speeches" for
  cells that sit at 13-14), which then contradicted the CLAUDE.md block the same commit added. A
  number that justifies a design decision gets restated in every place that decision is explained —
  module docstring, meta sidecar, CLAUDE.md, README — so the unit of repair is the claim across the
  tree, never the line you were looking at.
- **A rationale can be self-defeating, not merely wrong.** That same meta justified diverging from
  `combat.py`'s cluster floors with "the corpus median is ~30 speeches per period". The median is
  20.0 — and at 30 the argument would have pointed the *other* way, i.e. the stated premise would
  have refuted the choice it was offered to support. Correcting the number is half the fix; also
  assert that the premise **entails** the conclusion (`combat.LOW_CLUSTER_CAUTION >= median`), so a
  later retune of either side fails a test instead of quietly inverting the argument.

## Mechanize note completeness — reading for it does not converge (learned 2026-07-21)
`breadth-depth-register` hit one defect **nine times**: a statistic the analysis computed, bearing
on a claim the note made, printed nowhere. Careful adversarial reading found them one at a time
across three review rounds and never converged. Enumerating the artifact found six more in a single
pass. The generalizable shape:
- **Enumerate, don't read.** For every `(measure, taxonomy, treatment, statistic)` cell in the
  output, assert it is either printed in the prose or inside an **explicitly declared scope rule**,
  and run that assertion in CI (`test_every_significant_unprinted_cell_falls_inside_a_declared_scope_rule`).
  A note's "reports every arm it computed" is a testable claim — make it one, or don't make it.
- **A value-matching sweep needs a significant-digit floor.** Rendering a value at 0 decimals makes
  `0.7833` match a bare `1` present in almost any paragraph. That one artifact silently marked 77
  significant cells as "printed". Require ≥2 significant digits and pin it
  (`test_renderings_never_matches_on_fewer_than_two_significant_digits`).
- **The matcher must fail SAFE.** A missing/wrong section alias should produce a *false leak*, never
  a silent miss. Verify by deleting aliases one at a time and confirming the count only rises.
- **Completeness cuts both ways.** The rule that forces printing a damaging arm equally forces
  printing a helpful one. Two of the nine instances *understated* findings (a contrast at the
  bootstrap floor on all three arms filed as "marginal"; a 33-of-33-at-floor block unprinted).
  Over-correction is a real late-round failure mode — fixing bias in one direction while creating it
  in the other.
- **A guard can be blind to the defect shape it was written for.** The scanner added to catch a false
  "pre-declared" claim *skipped* segments containing a negation near "declared" — exactly where the
  original defect lived, so the bug re-introduced verbatim left it green. **Always mutate the
  original defect back in and confirm the new guard fails.**
- **Never backdate a pre-registration.** When prose claimed a marker was "pre-declared" and the
  declaration table did not contain it, the fix is to strike the claim — not to add the row.

## Testing
- `tests/` (pytest) covers the paragraph/issue-label keyed-merge logic. Run with
  `uv sync --extra dev && uv run pytest`, or directly:
  `arch -x86_64 .venv/bin/python -m pytest`.
- Favor small synthetic DataFrames over the real 36k-row corpus or the real CorEx topic model
  (`issues.build_issues()` fits a full model — too slow/heavy for unit tests). Mirror the
  specific logic under test on a tiny frame instead of invoking the real pipeline function.
- Money-path guard: `python -m presidential_profiles.taxonomy` (like `pp-annotate submit`) makes
  PAID Anthropic API calls; `--dry-run` is $0. Tests must never construct an anthropic client —
  `tests/conftest.py`'s autouse `_no_anthropic_creds` deletes the API key so any stray client
  construction fails loudly. Fake API responses with `types.SimpleNamespace`, keep `import
  anthropic` lazy inside the one client-constructing function.
- **An always-true assertion is invisible to coverage AND to mutation testing.**
  `assert np.isnan(x) is not available` passes for every combination of both operands, because
  `np.isnan` returns `np.bool_` and never the Python `True`/`False` singletons. Coverage sees the
  line execute; a source mutant that genuinely breaks the behaviour still leaves the test green. No
  tool in the suite reports this shape — grep for identity comparisons (`is True` / `is False` /
  `is not`) against numpy scalars and rewrite them as `==` / `bool(...)`.
- **Assert the value, not merely the conclusion it supports.** A docstring claiming `n**-n`
  underflows "near n=178" (it is 149) survived because the only assertion was
  `underflow_n < _MAX_CLUSTER_SEARCH` — true of anything under 1000, so the guard could not see a
  number 29 off. Same family: an equality assertion cannot distinguish a field that was **read**
  from its source from one that merely happens to **match** it, so proving a constant is genuinely
  derived (`MIN_CLUSTERS_FOR_RESOLVABLE_CI` from `CI_LOW`) takes a monkeypatch of the source plus a
  rebuild, never `assert floor == 4`.
- **A visual check cannot see a point that is outside the axis.** A hardcoded
  `X_RANGE = [1786, 2029]` cropped the 1785 bucket on *both* chart surfaces — the one period the
  whole change existed to stop hiding — and every screenshot review passed, because band, marker
  and hover were off-axis rather than visibly broken. Derive plot bounds from the plotted data in
  one place (`issues_site.x_range_covering`), and verify a recovered point by querying the figure's
  traces, not by looking at the picture.

## Mutation testing beats coverage, and a passing test is not a working test (learned 2026-07-22)
`convergence-analysis` shipped a 149-test suite at **99% line coverage** of the module. Mutation
testing then killed 61 of 62 mutants — but **14 of the first 27 SURVIVED**, including the ones
guarding the permutation null (the entire inference), the jackknife, and `significant_decline`.
Coverage measures *execution*; it cannot see *discrimination*. The recurring shapes:
- **Fixture-dependent vacuity was the single largest source (6 of 30 survivors).** A test whose
  fixture lacks the property under test passes for the wrong reason and is invisible to coverage,
  to a re-read, and to mutation of unrelated code. Fix: **every test asserts its fixture has the
  property first** (`assert len(pool) >= 5` before asserting an ordering over the pool), or pins a
  measured magnitude that fails loudly on re-seed. A re-seed must break the test, not silence it.
- **A test can pass against the exact defect it is named for.** `test_cluster_rarefaction_...`
  passed with the sampler swapped for the superseded paragraph-rarefied one, because its assertions
  (`shape ==`, `sum == 1.0`) are invariant to S and B. Likewise a "floor is bounded" test was
  guaranteed by an `np.clip` upstream. **The question to ask of any assertion: what single-line
  source change would make this fail?** If you cannot name one, it is decoration.
- **Identity is not evidence.** `per_slot == [value, value]` at two slots is an algebraic identity
  (symmetric JSD, zero diagonal) that holds for *any* implementation. Prefer a fixture where the
  quantities genuinely differ.
- **A "tautology" may be load-bearing.** Replacing that identity with an `argmax` discriminator
  silently regressed the sole killer of a divisor mutant, because `argmax` is scale-invariant.
  Before deleting an assertion as vacuous, check what it currently kills — the fix was an exact
  identity pin (`mean(per_slot) == the pairwise mean`, true only for the correct divisor).
- **Write the guard broader than the known instance.** A loop-variable-leak check written as a
  sweep over *every* function (not just the reported one) immediately found a second live instance.

## Prose tests must assert the RENDERED SENTENCE, not the source value (learned 2026-07-22)
`notes/convergence-findings-v1.md` shipped with `tests/test_convergence_note_claims.py`. The first
draft — 65 tests, all green — asserted values pulled from `data/convergence/*.parquet`. Mutating
the **note** rather than the code exposed the hole: changing `rank 6 of 38` → `rank 5 of 38` passed
every test, because not one of them read the prose. **Artifact-pinning is necessary and not
sufficient**; a note-claims suite that only checks artifacts is structurally blind to prose drift,
which is the entire defect class it exists to catch.
- Build the expected string **from the artifact** and assert it appears: `f"rank {rank} of {n}" in
  note`. Hardcoding the string re-creates the drift in the test.
- **Prove it by mutating the PROSE, not the code.** Four prose-only defects (`rank 5 of 38`,
  `lowest of all 45`, `7 of 105 windows`, `218 of 218`) each now fail their own guard.
- Comparisons need whitespace normalization and `^> ` stripping — a hard-wrapped blockquote is
  formatting; rewording is not. Normalize, don't loosen.
- **Double-rounding struck a third time**, in this note's own draft: "4.6×" from the rounded
  0.406/0.089, where the fraction gives 4.549 → **4.5×**. Knowing the rule did not prevent it.
  State the convention in the note *and* pin the figure with a test that recomputes it.

## A refusal guard must be cheap as well as early (learned 2026-07-22)
The existing rule is that a refusal runs BEFORE the write it refuses. `convergence.py` added the
cost corollary: its frozen-artifacts guard was first placed after the three-leg gate, so refusing an
illegal `--out-dir` would have cost ~4 minutes of Monte Carlo first. A refusal that expensive is one
people route around. Check it at function entry, and pin the ordering with a test that rigs the
expensive step to raise if reached. (The mutation proof is unusual and worth recognizing: removing
that guard makes the refusal tests **hang** rather than fail, because execution falls through into
the real gate — the hang *is* the evidence.)

## Read-only review systematically mis-adjudicates whether a test can fail (learned 2026-07-22)
Across this task, five quality gates ran without a Bash tool and reasoned statically. **Three of
their load-bearing claims were wrong**, each caught by executing it: a mutant reported as
"survives all 199 tests" actually died (`1 failed, 198 passed`); a `curve.used` conjunct reported as
differing between two call sites was present in both; a guard reported as fine was blind. They were
also right about things a reader would miss — one recovered a prior stage's mutation results from
the session scratchpad and used them to catch a genuine regression. The lesson is not "reviews are
unreliable" but **weight executed evidence over traced evidence, and re-run any claim that decides
a gate.** A reviewer without execution should say so explicitly and label each verdict.
