---
name: chart-and-band-review-heuristics
description: Five recurring defect shapes when reviewing chart/uncertainty-band tasks in presidential_profiles (found on topic-chart-upgrades, 2026-07-21)
metadata:
  type: project
---

Checks that paid off on `topic-chart-upgrades` (bands.parquet + banded trend charts).
All five are cheap and none were caught by the 115-test suite the task shipped.

**1. Artifact rows != rendered panels.** `bands.parquet` holds 22 CorEx series; only 16 are
ever drawn (`topic_quality.display_issues` filters the 7 `Discovered N` topics down to 1).
A docstring claiming "on the 22 CorEx issue panels exactly two overflow" was wrong on both
numbers (16 panels, 1 overflows). **How to apply:** whenever a docstring or note quantifies a
rendering effect, recompute it over the set that is actually *rendered*, not over the parquet.

**2. A newly-introduced axis `range=` can silently undo the task's own headline fix.** The task
existed to recover the 1785 five-year bucket from a `>=40` hard mask; the new banded
small-multiples path passed `x_range=X_RANGE = [1786, 2029]`, cropping the 1785 vertex off the
left edge of every dashboard panel. The unbanded fallback and the issue pages autorange, so the
two surfaces disagree. **How to apply:** diff the x/y ranges of the new render path against the
old one, and check the boundary period the task set out to recover is inside them.

**3. A loader seam that handles WHOLE-FILE absence usually does not handle PER-KEY absence.**
`bands.load_bands()` returns None gracefully and `fig_issue_timeline` falls back; but
`bands.series_band()` also returns None per series, and `site._banded_small_multiples` feeds
that straight into `line_traces` -> `TypeError: 'NoneType' object is not subscriptable`. Same
class as CLAUDE.md's "total vs partial filter failure". **How to apply:** for every optional
loader, enumerate its *other* None-returning sibling and check each consumer.

**4. Hover labels are published prose.** When dots changed meaning (per-president-at-term-midpoint
-> per-president-per-period), the issue-page hover was updated ("in this period") and the
dashboard hover was not — so the dashboard label now describes a different quantity than it
plots. **How to apply:** when a data function's semantics change, grep every `hovertemplate` /
`customdata` that consumes it, not just the one the diff touched.

**5. "Every paragraph" claims about the second annotator.** The inter-model agreement layer covers
8,570 of 36,229 paragraphs (262 of 1,057 speeches, ~24%) — a DOCUMENT-level subsample. Prose that
says the corpus was "read again by a second model" without the fraction overstates the annotator
component's basis. **How to apply:** any rendered sentence about Sonnet-vs-Opus must carry the
subsample size or an explicit "a sample of".

**6. An ILLUSTRATIVE magnitude survives the census correction placed next to it.**
`suppress-degenerate-band-intervals` rewrote the exact docstring line that read "the many
legitimate `lo=hi=0` cells where **sixty speeches** genuinely never touched an issue" in order to
append a correct 7/12/19 census — and left "sixty" standing, though those 7 cells sit at 13-14
speeches. A sibling new docstring said "dozens of speeches" for the same 7. Both read as rhetoric,
not as claims, which is why three prior reviewers passed them. **How to apply:** when a diff adds
an enumeration to a paragraph, re-read the *unchanged* clauses of that same paragraph against the
new numbers — the vague magnitude is where the defect hides once the precise one is fixed.

**7. Side-claims of pure arithmetic in new comments are never checked by anyone.**
The same diff justified a loop cap with "`n**-n` underflows to exactly 0.0 near n=178" (it is
n=149), repeated verbatim in a test docstring. The conclusion was unaffected, so no test could
catch it. **How to apply:** any float/combinatorial assertion in a new comment is a 5-second
Python one-liner — run it, especially where it is the sole justification for a constant.

**8. A "refuse to write" guard added late in a writer leaves a partial artifact.**
`bands.write_bands` calls `table.to_parquet(path)` on line 853 and the new self-consistency guard
raises on line 902 — so the refusal leaves an orphaned parquet beside a stale sidecar, the one
state worse than either extreme in a repo whose doctrine is "a dirty `git status` means the
numbers moved". The test asserted only "no partial *meta*". **How to apply:** for any guard added
to a multi-file writer, check its line number against every `write`/`to_parquet` call in the same
function, and read the test's assertion against the *set* of outputs, not the one it names.

**9. Centralising a column read into a helper can widen an AST/source guard's blind spot.**
Moving `band["interval_unresolvable"]` into `unresolved_mask` added `if "interval_unresolvable"
not in band:` to that helper — so the guard asserting the helper "still reads the column" is now
satisfied by the membership string alone, and passes on a body mutated to `return
pd.Series(False, ...)`. Nine behavioural tests still catch it, so it is INFO, not a finding.
**How to apply:** when a refactor adds a string literal of the column name to a function a
source-inspection test guards, re-run the original mutation — the guard may have gone quietly soft.

Related: [[project_research-report-review-heuristics]], [[project_threshold-anchoring-value]].
