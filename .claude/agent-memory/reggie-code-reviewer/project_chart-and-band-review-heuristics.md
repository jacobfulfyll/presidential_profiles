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

Related: [[project_research-report-review-heuristics]], [[project_threshold-anchoring-value]].
