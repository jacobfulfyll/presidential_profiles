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
