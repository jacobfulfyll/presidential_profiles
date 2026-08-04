---
name: fix-fixture-generators-by-documenting
description: When a test-fixture helper systematically generates weak fixtures, record the convention rather than rewriting the helper
metadata:
  type: feedback
---

When a repeated test defect traces back to a fixture HELPER (e.g. every helper built on
`r.get(key, ONE_SCALAR)`, so any field a test does not vary is identical on every row —
which makes symmetric fixtures that pass while the code is wrong), the fix is to **record
the convention where a future test author will read it**, not to make the helper vary its
defaults.

**Why:** the oracles in these suites are hand-computed against the flat defaults; randomising
or per-row-varying them trades a known, documented sharp edge for an unreadable suite and
destroys the property that makes the tests trustworthy. Reviewers on this repo have
explicitly advised against the rewrite.

**How to apply:** put the rule in the test MODULE docstring (the thing a new test author
reads first) plus a one-line pointer in each helper's own docstring — not in an inline
comment nobody opens. State it as an imperative: *no helper default may be shared by two
rows a test distinguishes.* Add "do NOT fix this by varying the defaults" so the next agent
does not undo it. Cite the worked example already in the file.

Two adjacent rules from the same review, worth repeating:
- **A fixture whose defaults contradict the invariant the real function guarantees is a
  trap even when inert.** `_half_widths` defaulted `share_primary == share_secondary == 0.2`
  next to `hw = 0.02` when the real relation is `hw == |sp - ss| / 2`. Derive the dependent
  defaults from each other so every row is coherent.
- **A test-side leak of a global (e.g. `pd.set_option` in a CLI reporter with no restore) is
  fixable with `pd.option_context` in an autouse class fixture, without touching source.**
  Prove it matters by deleting the guard and watching a probe test fail — CLAUDE.md's
  "mutate the original defect back in" rule applies to containment guards too.
