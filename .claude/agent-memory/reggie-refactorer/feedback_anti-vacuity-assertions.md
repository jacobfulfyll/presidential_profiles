---
name: anti-vacuity-assertions
description: Filter/quantifier assertions in this repo's tests must be preceded by a non-emptiness anchor — several were passing on empty frames
metadata:
  type: feedback
---

Before trusting an assertion of the form `all(...)`, `set(x) <= y`, `(frame[col] > k).all()`
or `len(out) < len(input)`, assert that the collection is non-empty *and* the right size.
Several tests in `tests/test_triangulate.py` were fully vacuous: a `rename_candidates`
result was empty on the synthetic fixture, so the threshold checks, the merge-back and the
`len(out) < len(drift)` "the filter did something" comment were all trivially true.

**Why:** this codebase's fixtures draw labels randomly (p=0.15) across eras that are
statistically indistinguishable, so filters that model a real-world signal legitimately
return nothing. The test then passes against an implementation that returns nothing at all.
Reviewers repeatedly flag this shape here — it is the house failure mode, not a one-off.

**How to apply:** when a fixture genuinely cannot produce the case, *plant* it rather than
weakening the assertion — add an explicit keyword to the fixture builder (e.g.
`_write_arms(..., collapse_corex_issue="Education")`) so the planting is visible and named,
and assert the planted item appears in the output. Prove the strengthened assertion bites by
running it against the pre-fix code (`git stash push <file>`, run, `git stash pop`).

Related: [[behavior-preservation-checks]]
