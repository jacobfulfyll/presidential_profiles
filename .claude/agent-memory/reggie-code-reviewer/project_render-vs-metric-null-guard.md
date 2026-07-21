---
name: render-vs-metric-null-guard
description: When a null-coercion fix lands on a compute/metric path, the sibling render/display path that touches the same field raw usually still crashes
metadata:
  type: project
---

Recurring gap class in this repo's annotation/agreement analysis code. Found on
`inter-model-agreement-check` (2026-07-21, REVIEW).

The QA pass hardened the METRIC path against null topics cells on ingested Opus
rows (M3 fix: `jaccard`/`_as_label_set` in agreement.py coerce None/NaN → empty
set). But the QUALITATIVE RENDER path (`_render_top_disagreements`) reads the
SAME field raw: `sorted(set(r.topics_primary))` — which raises TypeError on a
null cell (`set(None)` / `set(nan)`), crashing the report deliverable.

Worse, it's SELECTION-BIASED toward triggering: a null on one side vs a real
label on the other yields jaccard 0 → max topic-disagreement → lands in the
top-k the render path exists to show. So the exact rows most likely to hit the
raw `set()` are the ones the qualitative section renders.

**How to apply (review heuristic):** whenever you see a defensive null/empty
coercion added to a compute/metric helper, GREP every other call site that
touches the same field — especially f-string / display / `nlargest`-quote paths —
and confirm they route through the SAME guard (`_as_label_set`, `_fmt`, etc.),
not a raw `set()`/`sorted()`/iteration. The metric-path test (here:
`test_jaccard_null_topics_coerce_to_empty_set_m3_fix`) does NOT cover the render
path; a green suite hides the gap. Primary corpus had 0/36,229 null topics, so
the crash rides entirely on the incoming paid Opus table — invisible until real
data lands. Ties to [[pipeline-review-heuristics]] (green suite misses cross-
state paths) and [[llm-annotation-provenance-layer]] (deliverable integrity is a
hard value here).

**Second, thinner note (per-model table routing):** the Opus second-opinion
ingest routes to per-model-suffixed parquets (`paragraph_annotations__opus4-8`)
via `annotate._table_name(spec, model)`, keyed off `state["model"]` in
`cmd_ingest` — never a CLI flag — so it can't clobber the primary parquets. This
routing is only UNIT-tested (`_table_name`) + proven by the single-round factual
ingest; the multi-round append on the suffixed table (array-collapse resume
rounds) shares the default path's merge code (transitively covered by
`test_annotate_accumulation`) but has NO direct real-`cmd_ingest` integration
test. Per [[pipeline-review-heuristics]] #3, drive real cmd_submit+cmd_ingest
with a written per-model parquet to close cross-round state gaps.
