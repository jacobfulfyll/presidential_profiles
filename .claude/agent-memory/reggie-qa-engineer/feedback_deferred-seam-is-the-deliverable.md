---
name: deferred-seam-is-the-deliverable
description: When an acceptance criterion is deferred behind an optional-input seam, the seam's ACTIVE branch is the deliverable and is usually 100% untested because the real input file is absent
metadata:
  type: feedback
---

When a task defers an acceptance criterion by shipping an **optional loader**
("if the file exists, use it; else return None"), the *active* branch is the
deliverable — and it is the branch nothing exercises, because the file is absent
by construction.

**Why:** `combat.py` QUALITY-CHECK (2026-07-21). The annotator-disagreement seam
was tested thoroughly for the MARGINAL table (`rate_table(bands=...)`) but
`ratio_table` was never once called with `bands=` non-None. `_widen_ratio` +
the populated branch of `_band_lookup` were the module's only uncovered lines
(17 statements, 96% -> the missing 4%), and **every** mutation to them survived:
deleting the widening entirely, inverting its direction, emptying the lookup,
marking every row banded regardless of match. `build_combativeness` could also
drop `bands` on the way into either table undetected, since `bands is None`
today makes both branches identical. Meanwhile the findings note promised the
quoted headline "4.8x [2.3, 14.3]" would widen along with the marginals.

**How to apply:**
1. Grep for the seam's parameter at every call site (`grep -n "ratio_table"`)
   and check whether ANY test passes the non-default value. A coverage run with
   `--cov-report=term-missing` finds it in one command.
2. Test the pure widening/merging helper directly with hand-computable
   arithmetic (identity at zero half-width, direction, composition, floor at 0,
   unbounded-above when the band can zero a denominator, NaN passthrough so a
   suppressed cell never acquires an interval).
3. Drive it end to end by monkeypatching the LOADER (`C.load_agreement_bands`),
   never by creating the real artifact path — that file belongs to another task.
4. Assert the per-cell flag (`disagreement_band_applied`) is per-(era, flag),
   not "a band table existed".

Related: [[mutation-check-recurring-holes]] (constant self-reference),
[[combat-test-seams]].
