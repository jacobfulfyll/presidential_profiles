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
- `data/combat/` is the opposite case: **derived, deterministic, and safe to regenerate** with
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
