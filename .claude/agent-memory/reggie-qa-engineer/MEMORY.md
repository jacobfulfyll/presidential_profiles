# Memory Index

- [Test toolchain](project_test_toolchain.md) — uv-managed x86_64 venv has no pip; use `uv pip install`, `arch -x86_64` prefix, pytest config in pyproject
- [Mirror tests need a real-fn anchor](feedback_mirror-tests-need-real-fn-anchor.md) — issue_cards inline-logic mirrors miss formula-shape drift; anchor via parquet-monkeypatch harness
- [Pipeline dead target constants](feedback_pipeline-dead-target-constants.md) — validators ship named targets they don't assert; make live with minimal additive validate_* change + committed-artifact anchor
- [Mutation-check recurring holes](feedback_mutation-check-recurring-holes.md) — survivor classes: symmetric fixtures, constant-parametrized tests, belt-and-suspenders merges
- [annotate.py CLI test seams](reference_annotate-cli-test-seams.md) — hermetic hooks for pilot/QA-gating/speech-e2e/accumulation/sealing/run-id; monkeypatch A.load+PARAGRAPHS_PATH; cmd_qa needs out=
- [agreement.py test seams](reference_agreement-test-seams.md) — inter-model draw/metrics/build/report + annotate --model/--sample MOD; pure metrics, injectable speeches=, monkeypatch PARAGRAPHS_PATH; one_to_one is belt-and-suspenders
- [register.py test seams](reference_register-test-seams.md) — conftest panel/corpus builders, spanning-genre constraint, fixture spread needed to make bootstrap + standardization mutations detectable
- [Untracked files break git-checkout mutation harness](feedback_mutation-harness-untracked-files.md) — snapshot bytes instead; new modules from IMPLEMENT are `??` in git
- [attention.py test seams](reference_attention-test-seams.md) — def-time-bound defaults block path patching; fixture-symmetry traps; pinning recipes; pandas 3 parquet traps
- [Pin, don't fix, judge-carried defects](feedback_pin-dont-fix-known-defects.md) — assert observed behaviour with a LOUDLY named test + Discovered Issues entry, never the docstring's claim
- [One mutation batch is not a plateau](feedback_one-mutation-batch-is-not-a-plateau.md) — keep running fresh batches until one is all-caught; aim at wiring (copied columns, self-referential constants, call-site args), not algebra
- [combat.py test seams](reference_combat-test-seams.md) — injectable load_frame inputs, import-time-bound path defaults, real-corpus rate + ratio anchors, the 3 equivalent mutants
- [Same-frame invariance tests are vacuous](feedback_same-frame-invariance-tests.md) — shuffle tests must assert on CROSS-frame columns; `.sample(frac=1)` can no-op; untied fixtures never exercise a tie-break
- [A deferred seam's active branch is the deliverable](feedback_deferred-seam-is-the-deliverable.md) — optional-loader criteria ship 100% untested "if present" paths; grep every call site for the non-default arg
