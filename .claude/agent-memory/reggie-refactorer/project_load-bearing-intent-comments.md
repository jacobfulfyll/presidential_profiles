---
name: load-bearing-intent-comments
description: presidential_profiles encodes deliberate asymmetries as ALL-CAPS "do not fix this" comment blocks — treat them as specifications, not noise
metadata:
  type: project
---

Several things in `src/presidential_profiles/` look like bugs and are documented, tested
decisions. They carry ALL-CAPS header blocks (`COHERENCE CAVEAT — READ BEFORE REUSING npmi`,
`THE COHERENCE ASYMMETRY — DO NOT "FIX" THIS`, `THE CEILING CONTROL — why fill exists`,
`PROJECTION DIRECTION`, `REUSE, NOT REIMPLEMENTATION`). Known live ones:

- NPMI is computed for all 22 CorEx topics but thresholded only on the 7 discovered ones —
  anchored issues deliberately span vocabulary that never co-occurs.
- `status` (pre-registered quantitative gate) and `surface` (editorial) are separate so
  editorial judgment cannot be laundered through a number.
- `topic_quality.py` imports `_fit_vectorizer` / `_npmi` from `embed_topics.py` rather than
  cloning them, so CorEx and cluster coherence stay on one yardstick.
- Two local imports exist for real reasons: `issues.attach_coherence` and
  `topic_quality.validate_surfaced` both break import cycles and say so. A third one
  (`scipy.stats` inside `agreement_drivers`) had no such note and was incidental — moved to
  module scope.

**Why:** each block exists because the thing beneath it already invited a "fix" once. Removing
one deletes the record of a decision, and at least one is covered by tests that fail loudly.

**How to apply:** before consolidating two near-identical code paths or removing an apparent
inconsistency, grep upward for a capitalised header in the enclosing docstring. If the WHY is
documented, leave the code and say so in the report. Trim only comments that restate what the
code plainly does.

Related: [[behavior-preservation-checks]]
