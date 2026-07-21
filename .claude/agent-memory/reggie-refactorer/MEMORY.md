# Memory Index

- [Run commands](project_run-commands.md) — tests/site build use the x86_64 Rosetta venv at the main repo path (`arch -x86_64 .../.venv/bin/python`)
- [Annotation money-path guards](feedback_annotation-money-path-guards.md) — don't "simplify" money-path guards in annotate.py/llm_annotations.py/taxonomy.py; untestable run()s → keep mechanical, simplify in pure helpers
- [Byte oracle beats the work order](feedback_byte-oracle-beats-work-order.md) — "judge measured it output-neutral" and "# pragma: no cover" are claims; verify per-edit against the artifact sha
- [Analysis-module SIMPLIFY shape](project_analysis-module-simplify-shape.md) — attention.py/taxonomy.py passes end POSITIVE on lines; wins are dedup-by-naming, and the pre-registration docstring is provenance
- [Analysis-module refactor gates](feedback_analysis-module-refactor-gates.md) — gate register/combat/indices/issues/trends refactors on regenerated-artifact bytes (`git status data/<mod>/`), not pytest; byte-safe move list; `path=CONSTANT` defaults aren't a test seam
- [Behavior-preservation checks](project_behavior-preservation-checks.md) — byte-identical names-file rebuild + recomputing the report's published stats without touching `data/`
- [Verify before rewriting justifications](feedback_verify-before-rewriting-justifications.md) — recompute the numbers before replacing prose reasoning in `notes/`; the obvious replacement was also wrong
- [Anti-vacuity assertions](feedback_anti-vacuity-assertions.md) — `all()`/subset/filter assertions here pass on empty frames; anchor non-emptiness, plant the case explicitly
- [Load-bearing comments in this repo](project_load-bearing-intent-comments.md) — some "inconsistencies" are documented-on-purpose; check for a WHY block before simplifying
