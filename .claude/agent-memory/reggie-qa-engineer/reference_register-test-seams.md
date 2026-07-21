---
name: register-test-seams
description: Hermetic test seams + fixture design for register.py (breadth/depth/register trends) — synthetic-panel builders, the spanning-genre constraint, and the fixtures that make bootstrap/standardization mutations detectable
metadata:
  type: reference
---

`register.py` is pure-except-`load_inputs`, so every function is drivable on
synthetic DataFrames. Shared builders live in `tests/conftest.py`
(`register_taxonomy` / `register_corpus` / `register_panel`) because `tests/`
is not a package — see [[test-toolchain]].

**Seams worth knowing:**
- `load_inputs` reads 7 path constants + `TAXONOMY_PATH`, all module-level and
  monkeypatchable at tmp_path parquets. `main()` is best driven by patching
  `R.load_inputs` — but note `main` evaluates `inputs.paragraphs` *before*
  calling the patched `build_paragraph_frame`, so a `lambda: "inputs"` stub
  fails; return a `SimpleNamespace` with all 8 dataclass fields.
- `reference_genre_mix` RAISES unless the set of genres present in every era is
  exactly `REFERENCE_GENRES`. So ANY fixture that reaches `build_trends` must
  give all four reference genres to every era, and must not give a fifth genre
  to all of them. This bites when adding a "thin year" row: put it inside an
  existing era, not a new one.
- `SpeechPanel.scalars` needs all 30 columns `measures_from_sums` reads; build
  it with a defaults dict rather than through `build_speech_panel` when you want
  hand-computable sums.

**Fixture properties that decide whether a mutation is detectable (learned the
hard way, 2026-07-21):**
- A panel whose speeches are IDENTICAL within an era has zero bootstrap
  variance, so `weights[1:] = alpha` (resample-nothing) survives and
  "different seed changes the CI" fails. Give speeches within-era spread
  (e.g. words/paragraph target-50 / target / target+50) that still averages to
  the era target.
- Direct standardization preserves total PARAGRAPH mass exactly, so raw vs
  reweighted `n_paragraphs` is an EQUIVALENT mutant — no test can separate them.
  To pin `_estimate_group`'s "counts are unweighted" contract you need genres
  differing in BOTH paragraphs-per-speech and words-per-paragraph; then
  `n_speeches` and `n_words` diverge and both mutations are caught.
- `trend_statistics`' `if late and base` vs `if late or base` is also
  equivalent: an empty index list makes `values[:, []].mean(axis=1)` nan, so
  both branches emit nan. Report, don't chase.

**The systematically thin area is `build_speech_panel`'s COPY block, not the
estimators** (QUALITY-CHECK, 2026-07-21). The measure tests hand
`measures_from_sums` a sums frame directly, so the ~17 columns the panel copies
straight off `speech_stats` / `speech_markers` (`n_tokens`, `n_sents`,
`i_count`, `we_count`, `n_words`, 13 style markers) were never asserted at all:
zeroing every marker, feeding `n_words` from `n_tokens`, and swapping the i/we
pair all SURVIVED. Fix with one test that sets pairwise-distinct values and reads
them back off `panel.scalars`, with marker names written as LITERALS (both the
conftest builder and every expectation otherwise iterate `R.STYLE_MARKERS`, so
deleting a marker removes input and assertion in one step).

**Fixtures that are too symmetric to separate `==` from `!=` / `sum` from `any`:**
- No fixture gave a paragraph 2+ legacy issues, so `n_legacy = sum(axis=1)` ->
  `max(axis=1)` survived — on the measure the "shallower" headline rests on.
- The 2-paragraph fixture is 1 proposal + 1 values, so `pv_{level} = (pv ==
  level)` -> `!=` gives the identical count for every level. Need an asymmetric
  spread (1/2/1/3 over the four levels) plus a "the four flags partition the
  speech" assertion.
- Case variants are only tested through `build_paragraph_frame`, so the panel
  reading `work["topics"]` instead of `topics_resolved` survived — that is
  exactly the silent 111-assignment drop the module's docstring exists to
  prevent. Put a variant in a fixture that reaches `build_speech_panel`.
- Genres laid out contiguously + identical topic rows hide position/weight
  misalignment: `_cell_plan` emits cells in REFERENCE_GENRES order, so
  `positions` is interleaved only when genres alternate in the panel. Alternate
  them AND give the genres different breadth.
- Panels always fed era-ascending, so `sorted(eras)` -> `list(eras)` survived.
  Feed one panel reversed and assert `eras != sorted(eras)` as the non-vacuity
  guard.

**Proving the bootstrap is speech-clustered (the plan's forbidden error):** the
sharpest discriminator is a grid check, not an interval-width check. Four
speeches of 50 homogeneous paragraphs each (2 all-unlabelled, 2 all-labelled)
force every replicate share onto {0, .25, .5, .75, 1}; a paragraph-level
bootstrap lands off that grid on nearly every draw. Pair it with
`weights[1:] @ paragraph_counts` being constant (no replicate ever splits a
speech). Both catch `multinomial(n*50, ...)/50`, the faithful paragraph-bootstrap
mutation.
