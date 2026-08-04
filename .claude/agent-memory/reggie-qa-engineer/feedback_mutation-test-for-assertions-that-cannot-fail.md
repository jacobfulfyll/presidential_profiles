---
name: mutation-test-for-assertions-that-cannot-fail
description: High coverage hides assertions that cannot fail; mutation-test every load-bearing behavior, and watch for the six recurring vacuity shapes this repo keeps producing
metadata:
  type: feedback
---

Coverage percentage says nothing about whether a suite can FAIL. On
`convergence-analysis` (2026-07-21) a 149-test suite at **99% coverage** let
**30 of 62 semantic mutants through**, including swapping the whole estimator
for the broken one the task exists to avoid.

**Why:** the user found by execution that
`test_cluster_rarefaction_always_uses_exactly_m_paragraphs` PASSED against
`draw_compositions_paragraph` — its assertions were `c.shape == (...)` and
`c.sum(-1) == 1`, and a mean of rows that each sum to 1 sums to 1 at ANY S and
B. A test whose name claims a property it cannot detect is worse than no test:
it manufactures confidence. They asked me to assume that was a class, not an
instance. It was.

**How to apply — the six vacuity shapes that recurred here:**
1. **Shape/normalization assertions on a sampler.** They hold for every
   sampler. Discriminate with a fixture whose STRUCTURE the estimator must
   respect: internally-homogeneous speeches + `S=1` means a cluster draw is a
   *pure corner* and a paragraph draw is a mixture (100% vs 0.5% pure).
2. **Bounds the code already guarantees.** `0 <= v <= 1` after an `np.clip` is a
   tautology. Assert against a *contrast* value instead (block re-deal 0.580 vs
   paragraph re-deal 0.042), or an exactness property (a mean of {0,1} values
   over N draws must be an exact multiple of 1/N).
3. **Fixture-dependent vacuity — the biggest single source.** A `MIN_PRESIDENTS`
   floor test on a corpus whose eligible count never hits the floor; a
   "never-eligible presidents are left alone" loop on a design where everyone is
   eligible; a two-sided-p test whose null has mean == median; a dedup test
   where #paragraphs == #assignments. **Assert the fixture has the property
   first** (`assert outsiders == [...]`, `assert -1 < rho < 0`), so a re-seed
   fails loudly instead of going silent.
4. **`@property` statistics go untested.** `floor_corrected` and `excess_ratio`
   were published columns and decision-table cells with zero direct tests.
   Hand-build the container (a `Curve`) and pin them on hand-derivable values.
5. **Conjunctions tested on one side only.** Every `and` in the decision table
   was exercised with both operands equal, so `and -> or` survived on the
   `excess_ratio` rule. **Parametrize over the arms.**
6. **A second spelling of the same rule.** `ends_2014` lives in both
   `curve_statistics` and `_curve_rows`; killing the mutant in one leaves the
   other unguarded. Grep for the rule, guard each site.

**Method that worked:** JSON list of `{id, old, new}` string patches, a runner
that applies -> `pytest -q --tb=no -rf` -> `git checkout --` in a `finally`.
Run all; re-run only the survivors after writing kills; then re-run the FULL
set at the end to prove no regression. ~7s per mutant made 62 mutants cheap.
Related: [[structural-guards-scan-the-whole-module]], [[convergence-test-seams]].
