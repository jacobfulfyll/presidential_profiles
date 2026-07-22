---
name: numpy-bool-identity-assertions
description: An `is`/`is not` comparison against a numpy scalar can never fail — sweep for it explicitly, because coverage reports and mutation batches both miss it
metadata:
  type: feedback
---

Never write `assert np.isnan(x) is not expected` (or `is True` / `is False`) against a
**numpy** scalar. `np.isnan` returns `np.bool_`, which is never the Python `True`/`False`
singleton, so the identity comparison is true for **every** combination of both operands.
It is a green test asserting nothing. Use `assert bool(np.isnan(x)) is expected`, or
better, branch and assert on the VALUE (`== pytest.approx(...)` on one arm,
`bool(np.isnan(...))` on the other).

**Why:** shipped in `tests/test_bands.py` and caught by the reggie-judge at QUALITY-CHECK,
not by me. The line sat one row below a correct `assert bool(row["available"]) is available`
— the two look identical and only one works. It was the ONLY assertion on the
"below `MIN_PAIRED_PARAGRAPHS` ⇒ half-width is NaN" branch, so a whole guard was
effectively untested behind a passing test.

**How to apply:**
- Neither coverage nor a source-mutation batch finds this class. The line *executes*, and a
  source mutant that breaks the behaviour still leaves the always-true assertion green — the
  mutant is caught only if some OTHER test covers it. So it needs its own grep:
  `grep -n " is not \| is True\| is False"` across new test files, and check each hit's left
  operand is a genuine Python bool (plotly/pandas attribute) rather than a numpy scalar.
- Generalize the sweep to "assertions whose truth does not depend on the value under test":
  bare `assert markers` where an empty overlay still produces a trace; `for x in xs: assert ...`
  with no length guard on `xs`; `sorted(a) == [...]` where the pairing between two columns is
  the thing that matters (`dict(zip(names, values))` instead).
- After fixing one, mutate the source so the branch misbehaves and confirm the FIXED assertion
  fails. A fix that is still green under that mutation is not a fix.

See [[bands-test-seams]], [[mutation-check-recurring-holes]].
