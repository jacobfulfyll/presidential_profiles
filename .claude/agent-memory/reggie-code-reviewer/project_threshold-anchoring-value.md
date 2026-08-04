---
name: threshold-anchoring-value
description: In this repo, analytic thresholds must be pre-hoc and anchored to pre-existing codebase constants, not tuned to outcomes — a hard review value
metadata:
  type: project
---

When reviewing analysis/scoring code in presidential_profiles, treat threshold
*provenance* as a first-class review concern, not a nit. A magic number that
selects/flags/ranks records (an eligibility cutoff, a "topic of the day" flag,
a sparse-record gate) should be:
  - documented pre-hoc (stated from first principles, with the reasoning inline),
  - anchored to a constant the codebase already uses elsewhere (so the same
    question gets the same answer in two places), and
  - explicitly NOT justified by "which presidents it happens to flag."

Example seen in task `profile-issue-views` (profiles.py:20-44): REL_DISTINCT_PP
reuses the old 0.75pp era-distinctive line; RAW_ELEVATED_MULT=1.5 reuses the
1.5x term-concentration ratio in issue_cards(); SPARSE_MIN_SPEECHES=5 mirrors
site.py's `n_speeches < 5` sparse cutoff. Verify these reuse claims by grepping
the cited source — they were all accurate there, but the *claim* is the thing
to check.

**Why:** Same falsification ethic as [[convergence-investigation-direction]] and
the provenance value in [[llm-annotation-provenance-layer]]. In a research
codebase, an outcome-tuned threshold is a silent p-hack; the author flags it as
pre-hoc precisely because it's load-bearing for the finding's credibility.

**How to apply:** If a new threshold has no anchor or reads like it was picked
to produce a nice-looking result, flag it (MAJOR — undermines the analysis),
even though the code "works." If the author anchors it to an existing constant
and documents the reasoning, that satisfies the value — don't nitpick the
specific value.
