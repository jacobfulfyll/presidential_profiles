---
name: pin-the-conclusion-not-the-shape
description: For a report claim of the form "the effect does not survive control X", assert the attenuation itself plus a paired contrast fixture — key-existence tests are not coverage
metadata:
  type: feedback
---

When a `notes/*.md` report states a **statistical conclusion**, the test must
assert the conclusion, not the presence of the statistic. In this repo the
worked example is report §0's *"the association does not survive the control"*:
the suite originally asserted only that both `divergence_vs_jaccard` and
`divergence_vs_fill_CONTROLLED` were keys in the returned dict, plus that every
rho was in [-1, 1]. An `agreement_drivers` that aliased the controlled statistic
straight back to the uncontrolled one passed 152 of 153 tests.

**Why:** falsification is a hard user value here (see the guiding value in
`notes/convergence-investigation-direction.md`). A partly-tautological finding
that replaces a falsified prediction is the exact failure mode the value exists
to prevent — so the disclosure that neutralises it has to be executable, not
prose. IMPLEMENT already failed one gate on precisely this.

**How to apply — the two-fixture pattern.** One fixture alone is not enough:
"controlled rho is small" also passes against an implementation that computes
the control wrongly and always returns noise.

1. *Effect-is-pure-artifact fixture* — construct inputs where the confound is
   the ONLY link (for Jaccard/ceiling: boolean sets nested so overlap sits near
   `min/max`, with the realised fill jittered in an order deliberately
   uncorrelated with the predictor). Assert uncontrolled |rho| high, controlled
   |rho| low, `p_controlled > p_uncontrolled`, and both compared at the same n.
2. *Real-effect contrast fixture* — same base rates and ceilings, fill made to
   track the predictor. Assert the controlled statistic STAYS large. This is
   what proves the control is a measurement.
3. Assert the fixture's own premise first (`fill.min() > 0.8`, predictor range
   > 2.0, `len(t) == 15`) or a weak controlled rho proves nothing.

Same discipline for anti-vacuity bars: a spread test (`max - min > 0.05`) does
not imply a binding ceiling — pair it with a level test (`max < 0.9`), or fills
clustered at 0.95-0.99 pass while the control is effectively dead.

See [[report-numbers-need-artifact-anchors]] and
[[triangulate-monkeypatch-harness]].
