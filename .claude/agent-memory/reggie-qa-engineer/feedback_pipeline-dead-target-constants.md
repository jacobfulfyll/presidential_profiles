---
name: pipeline-dead-target-constants
description: presidential_profiles LLM-pipeline modules ship named target constants their validators don't assert; WRITE-TESTS is expected to make them live with a minimal additive validate_* change + anchor on committed JSON artifacts
metadata:
  type: feedback
---

In this repo's provenance/validation modules (`taxonomy.py`, and see
[[mirror-tests-need-real-fn-anchor]] for `issue_cards`), the pure validation
functions ship with **named target constants that the validator never asserts** —
e.g. `taxonomy.REQUIRED_NON_POLICY` (the 4 required non-policy buckets) existed
but `validate_structure` never checked it; `compute_coverage` counted any
non-empty label list without validating names against the frozen level-2 set.
These are the exact "dead constant" gaps a quality gate flags.

**Why:** the modules are authored top-down from a plan's acceptance criteria, so
the *targets* get named as constants before the *checks* enforcing them are
written. The IMPLEMENT stage can pass its own gate while leaving the constant
inert.

**How to apply:** WRITE-TESTS here is explicitly allowed a minimal, **additive**
source change to make the target testable — grow the validate_/compute_ function
with new report fields (`missing_non_policy`/`non_policy_complete`,
`compute_coverage(valid_names=...)`, an `unknown_labels()` reporter) rather than
changing existing return keys, then wire it into `run()` so it isn't dead again.
Match the merge's OWN wording semantically (distinctive-keyword substring match,
not exact string equality — the LLM names "Faith & National Values" for the
"values appeal" bucket). Anchor the new check on the **committed artifacts**
(`data/llm_annotations/taxonomy_v1.json`, `crosswalk_v1.json`) as fast read-only
tests — read them by absolute worktree path (`Path(__file__).resolve().parents[1]
/ "data" / ...`), NOT the module path constant, because conftest's autouse
`redirect_annotation_dirs` repoints `ANNOTATIONS_DIR` at tmp_path and under a
worktree the constant can resolve to the main repo. Never touch the money path
(`_client()`, `run(dry_run=False)`); test paid-call helpers like `_extract_json`
with faked response objects (`types.SimpleNamespace(stop_reason=..., content=[...])`).
See [[test-toolchain]] for the exact worktree pytest invocation.
