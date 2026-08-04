---
name: mutation-check-recurring-holes
description: Recurring surviving-mutant classes in presidential_profiles suites — symmetric fixtures, constant-parametrized tests, belt-and-suspenders keyed merges, invalid syntax-error probes
metadata:
  type: feedback
---

QUALITY-CHECK mutation spot-checks on `taxonomy.py` (and the same shape recurs in
other validation modules here) surface two recurring survivor classes worth
probing for FIRST — the money-path functions are correctly uncovered by design, so
the real signal is in the pure logic:

1. **Tests parametrized over the very constant they should pin.** The non-policy
   suite did `@pytest.mark.parametrize(list(T.REQUIRED_NON_POLICY))` and asserted
   `missing_non_policy == list(T.REQUIRED_NON_POLICY)`. Dropping a bucket from the
   constant SURVIVES every such test (they just check fewer buckets). Kill it with a
   **decoupled literal spec-anchor**: `set(T.REQUIRED_NON_POLICY) == {the 4 AC names}`
   plus `set(SYNONYMS) == set(REQUIRED_NON_POLICY)`. This is the test-side twin of
   [[pipeline-dead-target-constants]] — there WRITE-TESTS makes the constant live in
   the source; here QUALITY-CHECK anchors the requirement as a literal in the test.

2. **Keyed merges guarded transitively (belt-and-suspenders), so a single-line
   downgrade survives.** `attach_metadata` does paras×issues then ×clusters, BOTH
   `validate="one_to_one"`. `test_duplicate_key_raises` duplicates the key in all
   three frames, so downgrading ONLY the issues merge survives (the clusters merge
   backstops the fan-out with its own raise). Downgrading BOTH is caught. So the
   CLAUDE.md-critical issues-merge invariant is enforced but NOT attributable to its
   own line. Can't isolate without a source change (the downstream one_to_one always
   re-catches), so REPORT as low-severity attribution gap, don't implement.

**Non-empty backfill branches often go unexercised.** `draw_era_samples`'s backfill
(`fill = _sample_ids(rest, ...)`) only draws real rows when unlabeled > per_era//2
AND labeled < per_era - half — the small-early-era case. Every existing test hit the
branch with `rest` empty (fill=[] no-op), so `fill = []` mutation SURVIVED. Force it:
60 unlabeled + 10 labeled, per_era=80 → 40+10+20-backfill = 70 rows.

**Threshold-anchoring survivors in gate tests (annotate.py cmd_qa, 2026-07-20).**
The pilot-QA gate tests (`test_annotate_qa.py`) drive each gate with a clearly-HEALTHY
fixture (100% coverage, 30% flag) and a clearly-BAD one (80% coverage, 100%/0% flag).
That proves the gate *fires* pass/fail correctly, but leaves the exact threshold
CONSTANTS unpinned: `QA_MIN_COVERAGE` survived 0.99→0.85 AND 0.99→0.999, and
`QA_FLAG_DEGENERATE_HIGH` survived 0.60→0.95 — every fixture stays on the same side of
the moved line. Same survivor class as the taxonomy constant-parametrized tests above.
Kill with a decoupled literal spec-anchor pinning each gate constant to task.md's
Verification Strategy values (coverage 0.99 / schema 0.01 / degen-high 0.60 / degen-low
0.005 / entity 0.80) — I added `test_qa_gate_thresholds_match_the_verification_strategy_spec`.
NOTE this is a durability/regression gap, NOT a current-correctness hole: the gate's
behavioral contract (auto-proceed healthy, stop bad) is fully tested, so it still
safely guards a paid run at the committed constants. All 8 required money-path
mutations (thinking-disable, max_tokens 2000+80n, cross-batch accumulate, entity
zero-clear, pilot random_state/strata, title-leak, schema-enum drop, seal-retryable)
were CAUGHT with zero survivors — this suite is strong; the QA thresholds were the
only gap found.

**SYMMETRIC FIXTURES neutralize weighting and eligibility rules (found 2026-07-21,
`attention.py` QUALITY-CHECK).** The single richest survivor class here. Whenever a
function combines strata/groups with weights or an eligibility threshold, a fixture
whose groups are the SAME SIZE — or whose corpus weights come out equal — makes the
weighted answer numerically identical to the pooled answer, and every assertion on it
is vacuous. Three independent mutants survived on one such fixture (unify the
eligibility rule, delete the whole standardization branch, swap corpus weights for
equal weights). A fixture value sitting EXACTLY on the threshold (3 paragraphs when
`MIN_GENRE_STRATUM == 3`) is the same bug in its purest form. **Rule: for any
weighted/post-stratified combiner, build the fixture so that pooled, equal-weight,
actual-weight, and threshold-excluded all give DIFFERENT numbers, then assert the
actual one and at least one of the alternatives.** Cheap detector: compute the
function under two treatments on the fixture and assert they differ — if
`genre_standardized == raw` elementwise, the fixture proves nothing.

**A "caught" mutant that only produced a SyntaxError proves nothing.** One of the 33
WRITE-TESTS probes deleted a line out of an `if` body, leaving `if …:` with no block;
pytest reported `1 error in 0.20s` (collection failure) and the harness scored it
CAUGHT. Real score was 32/33 valid. **Make the harness treat a run with 0 collected
tests, or a collection error, as INVALID rather than CAUGHT** — check for `error in`
as well as exit code 5. Repaired probes (`x = x` instead of deleting the line) were
genuinely caught, so the tests were fine; the evidence was not.

**Mutation-harness gotchas:** revert with `git checkout -- <file>` (clean base
guarantees byte-identical); verify `git diff --stat` empty after EACH mutation.
An exact-string-replace helper that ASSERTS `s.count(old)==1` (aborts otherwise)
makes a stale mutation impossible to apply silently; full annotate suite is ~4s so
running it whole per probe (richest failing-test data) beats per-file selection.
pytest exit code 5 ("no tests ran") is non-zero and masquerades as CAUGHT — pass
each `path::Class` node as a SEPARATE argv element, never one space-joined string.
Coverage of `taxonomy.py` from its own tests is ~60% and that is HEALTHY: the ~40%
miss is the isolated `_client()`/`run()`/API-call money path (correctly untested);
the pure functions are high. See [[test-toolchain]] for the worktree pytest invocation.
