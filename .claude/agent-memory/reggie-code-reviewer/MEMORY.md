# Memory Index

- [Paragraph key & tests](project_paragraph-key-and-tests.md) — canonical (doc_name, para_idx) join key, positional-alignment review heuristic, how to run pytest
- [LLM annotation provenance layer](project_llm-annotation-provenance-layer.md) — llm_annotations.py + pp-annotate; provenance/reproducibility is a hard review value here
- [Threshold anchoring value](project_threshold-anchoring-value.md) — analytic thresholds must be pre-hoc + anchored to existing constants, not outcome-tuned; verify reuse claims
- [Pipeline review heuristics](project_pipeline-review-heuristics.md) — TASKS.md two-dot-diff FALSE POSITIVE; run() bail-gate pattern; total-vs-partial filter fallback bug class
- [Render vs metric null guard](project_render-vs-metric-null-guard.md) — null-coercion fixes on compute paths miss the sibling render path; per-model ingest routing only unit-tested
- [Research-report review heuristics](project_research-report-review-heuristics.md) — 13 checks; #5 selective-statistic reuse is the dominant class (9 instances in one task); #5b/#5c: a mechanical sweep does NOT exhaust it — audit the declared scope rules
- [Research-report review heuristics](project_research-report-review-heuristics.md) — 13 checks; #5 selective-statistic reuse is the dominant class (6 instances in one task) — build the full parquet cell grid and grep the note for every measure
- [Chart & band review heuristics](project_chart-and-band-review-heuristics.md) — 5 checks for chart/uncertainty-band tasks: artifact rows != rendered panels, new axis ranges, per-key seam absence, hover labels as prose
- [Analysis-report audit heuristics](project_analysis-report-audit-heuristics.md) — the recurring "true conclusion, invalid justification" defect and the 4 recomputations that catch it
