---
name: report-numbers-need-artifact-anchors
description: Numbers quoted in notes/*.md must be regression-checked against the committed data artifact, and a test docstring that mis-describes its body is a suspected coverage hole
metadata:
  type: feedback
---

Two calibration lessons from QUALITY-CHECK on `topic-method-comparison`.

**1. A published figure needs an artifact anchor, not a hand-written fixture.**
Report §3's anchored-coherence caveat table (Immigration −0.028, Foreign policy
0.079, Infrastructure 0.089, Civil rights & race 0.091) lived in the suite only
as `_rec(-0.028)`-style inputs to gate tests. Those pin the GATE's behaviour at
those values and say nothing about whether `data/issues_meta.json` still holds
them — so the artifact could be regenerated on a drifted vocabulary and every
test stayed green while the report's central claim became false.
*How to apply:* read the real artifact through the module constant
(`issues.ISSUES_META_PATH`, `topic_quality.load_names()`), assert published
3-decimal figures with `abs=5e-4`, assert the ORDERING the report argues from
(the caveat four really are the bottom four), and cross-check co-derived
artifacts against each other with `==` — `issues_meta.json` and
`topic_display_names.json` are written by one run and quoted by different
report sections, so a partial regeneration is silent otherwise.
Pair with an autouse conftest guard that fails any test which WRITES a committed
artifact, or the live reads become order-dependent.

**2. A docstring that does not match its body is a coverage hole, not a typo.**
`test_scoring_is_pure_with_respect_to_the_corpus` claimed "two disjoint corpora
give the same topic different scores" but actually varied the TOPIC while
passing a reversed (identical-multiset) corpus. Verified by mutation: a
`compute_coherence` that cached the first corpus it ever saw and ignored `texts`
thereafter PASSED the old test and FAILED the restructured one.
*How to apply:* when a judge or reviewer flags a mismatched docstring, restructure
the test so the docstring becomes true rather than editing the prose — the
docstring is usually describing the test someone MEANT to write, and that test is
the one with teeth. Split off the incidental property (row-order invariance) into
its own named test instead of deleting it.

See [[pin-the-conclusion-not-the-shape]] and
[[synthetic-corpus-derivable-npmi]].
