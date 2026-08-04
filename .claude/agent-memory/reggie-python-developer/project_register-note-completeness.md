# register.py / register-findings-v1.md — completeness discipline

## The recurring defect class (9 instances across 4 review rounds)
"A `(measure, taxonomy, genre_treatment, statistic)` cell that `build_trends`
computed, that bears on a claim, and that appears nowhere in the prose." Rounds
1–2 found the note *withholding* evidence that weakened its claims. Round 3 found
the inverse: over-correction — hedges asserted past what the artifact supports
(`nrc_fear` called "the least stable measure"; sub-claim 3 called the
"weakest-evidenced" when only one of its two block contrasts is marginal).
**Accuracy runs in both directions; a hedge can be as wrong as a boast.**

## Governing rule the note itself states
"Marginal on one contrast is not marginal on the measure." Apply it symmetrically.
The measures with two block contrasts (`delta_modern_minus_early` and the post hoc
`delta_modern_minus_postbellum`) routinely disagree; scope every verdict sentence
to the contrast it is about.

## The value-matching sweep (how omissions are actually found)
`tests/test_register_note_claims.py::
test_every_significant_unprinted_cell_falls_inside_a_declared_scope_rule` sweeps
all 444 trend cells against the prose. Three matcher rules, each closing a
false-positive channel:
- **never render at 0 dp** — `0.7833` → `"1"` matches almost any prose. This one
  artifact hid 77 of the 89 significant unprinted cells from the earlier sweep.
- **whole-token match** (`(?<![0-9.])…(?![0-9])`) so `0.78` cannot match inside
  `10.783`.
- **measure-scoped**: the match must land in a `#`/`##` section naming the measure
  (alias map for `effective_topics` / `non_policy_share` / …), so `nrc_fear`'s
  +0.783 is not satisfied by `effective_topics`' +0.783 four sections away.
Residual collisions make the matcher over-permissive, which weakens the test but
cannot make it fail spuriously — the safe direction.

## Pre-registration is a *table*, not prose
The note's Pre-declaration table is the single source of truth for what was
declared. `nrc_fear` / `nrc_hope` were called "pre-declared" in four places and are
in neither the table nor `.pipeline/breadth-depth-register/{task,CONTEXT}.md`
(CONTEXT.md line 109 lists them as *available columns*, never a predicted
direction). Never backdate a prediction into the table to make prose true — strike
the prose. `test_the_only_measures_the_note_calls_declared_are_the_declared_ones`
pins this by segmenting the note on sentence/bullet boundaries (a ±N-character
window leaks across bullets) and asserting claimed ⊆ table.

## Gotchas confirmed
- `effective_topics_plugin` is **not** a monotone transform of `effective_topics`
  (Miller-Madow is data-dependent). It happens to induce identical era ranks and
  identical Spearman point values, but its p-values differ (0.028 vs 0.037, 0.005
  vs 0.008, 0.0005 vs 0.0010 on level-1). `normalized_entropy` *is* monotone —
  p-values match too.
- `effective_topics` llm_level1 is the only published measure whose significant
  contrasts point in opposite directions (endpoint +2.11/+1.97 sig positive,
  post-1860 −0.95/−0.90 sig negative, same arms). The twins inherit it.
- Note prints bounds ("no more than 4.02"), not values — assert `<= bound` plus a
  tightness check, not `_assert_printed` equality.
- `data/register/trends.parquet` must stay byte-identical:
  sha256 `73ea145e…`, 403,615 bytes, 16,243 rows. Regenerate with
  `PYTHONPATH=$PWD/src arch -x86_64 <mainrepo>/.venv/bin/python -m
  presidential_profiles.register` and re-hash after any `register.py` edit.
- Worktree: always `export PYTHONPATH=$PWD/src` and verify
  `python -c "import presidential_profiles as m; print(m.__file__)"` resolves
  inside the worktree — the editable-install `.pth` points at the main repo.
