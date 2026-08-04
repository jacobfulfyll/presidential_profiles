---
name: kappa-metric-test-blind-spots
description: Two blind spots when testing Cohen's-kappa / chance-corrected metrics in presidential_profiles agreement.py — degenerate-class guard and chance-correction
metadata:
  type: feedback
---

When QA'ing the inter-model agreement metric functions (`agreement.compute_kappa`
and friends), two mutation-analysis holes recur that a first-pass test suite
misses. Both were found closing gaps on `inter-model-agreement-check`.

**1. A `math.isnan` assertion does NOT pin the degenerate-single-class guard.**
`sklearn.metrics.cohen_kappa_score` on an all-one-class input returns `nan`
*already* AND emits `UndefinedMetricWarning` + `UserWarning`. So `compute_kappa`'s
early-return guard changes nothing about the value — its real job is suppressing
the warning. Deleting the guard passes any `isnan(value)` test silently.
**How to apply:** pin the guard with `warnings.catch_warnings()` +
`simplefilter("error")` around the call and assert it still returns nan — that
makes guard-deletion actually fail.

**2. kappa tests covering only {1.0 identical, -1.0 inverted, NaN degenerate} do
NOT prove chance-correction.** A fake metric returning raw agreement, or the
linear `2*agreement-1`, gives 1.0 on identical and -1.0 on inverted too, so every
one of those tests still passes — yet the whole falsification premise (per the
CONTEXT gloss: "both models saying no-flag 95% of the time is not impressive") is
that kappa corrects for chance.
**How to apply:** add a class-imbalanced witness where raw agreement is HIGH but
kappa is NEGATIVE — e.g. `a=[0]*9+[1]`, `b=[0]*8+[1,0]`: raw agreement 0.8, Cohen
kappa = (0.8-0.82)/(1-0.82) = -1/9 ≈ -0.111. Assert `value < 0` and
`approx(-1/9)`. That single test kills both the raw-agreement and `2*agreement-1`
mutants.

Related metric-test guidance: [[mirror-tests-need-real-fn-anchor]]. Toolchain:
[[test-toolchain]].
