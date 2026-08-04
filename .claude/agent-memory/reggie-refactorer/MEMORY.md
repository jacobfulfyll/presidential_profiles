# Memory Index

- [Run commands](project_run-commands.md) — tests/site build use the x86_64 Rosetta venv at the main repo path (`arch -x86_64 .../.venv/bin/python`)
- [Analysis-module SIMPLIFY gates](project_analysis-module-simplify-gates.md) — the parquet-digest + `docs/`-rebuild pair that proves zero behaviour change, and what a pass actually yields
- [Era-atlas simplify shape](project_era-atlas-simplify-shape.md) — how to run the byte gate on `eras.py` without touching the paid portrait path
- [Convergence simplify shape](project_convergence-simplify-shape.md) — the 12-min byte gate on `convergence.py`, and why SIMPLIFY there is net-positive lines by design
- [Fix fixture generators by documenting](feedback_fix-fixture-generators-by-documenting.md) — record the convention in the module docstring; never rewrite helpers to vary defaults per row
- [Replace tautological assertions with discriminators](feedback_replace-tautological-assertions-with-fixture-discriminators.md) — build the case where the property is false; mutate every guard you add or touch
- [Replace tautological assertions with discriminators](feedback_replace-tautological-assertions-with-fixture-discriminators.md) — build a fixture where the identity breaks; never just rename the test
