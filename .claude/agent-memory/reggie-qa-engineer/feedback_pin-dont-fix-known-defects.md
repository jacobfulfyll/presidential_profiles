---
name: pin-dont-fix-known-defects
description: In presidential_profiles WRITE-TESTS, pin current behaviour for judge-carried-forward defects and label the test loudly, rather than asserting the docstring's claim
metadata:
  type: feedback
---

When the IMPLEMENT judge hands WRITE-TESTS a *carry-forward defect list*, write
the test against **observed current behaviour**, name the test so the deviation
is visible, and cross-reference the defect number in the docstring — do NOT
assert what the source's own docstring claims.

**Why:** a test asserting the intended behaviour fails immediately and looks like
a broken test suite, which pressures the next stage into an out-of-scope source
change; a test asserting current behaviour *silently* makes the defect look
intentional and it never gets fixed. The loudly-named pin gets both: the suite is
green for REVIEW, and SIMPLIFY/QUALITY-CHECK cannot fix the defect without
deliberately editing a test that says why it exists.

**How to apply:** name the test for the deviation
(`test_filter_4_is_SKIPPED_when_the_dying_topic_has_no_crosswalk_parent`), open
the docstring with "PINS CURRENT BEHAVIOUR, WHICH CONTRADICTS THE DOCSTRING",
quote the source line number, and repeat it under `## Discovered Issues` in the
handoff. Used for `attention.py:916` (filter 4 skipped when `legacy` is empty)
and `attention.py:1085` (`_era_shares` uses `den > 0` while `_combine_strata`
uses `den >= MIN_GENRE_STRATUM` — the CI grid and the point curves genuinely
disagree). See [[attention-test-seams]].

Corollary for guards that fire on nothing in production (`UNDER_POWERED_N`, an
anachronism check): a synthetic test is the ONLY thing keeping them from rotting
into dead code — assert them at BOTH the unit level and once end-to-end.
