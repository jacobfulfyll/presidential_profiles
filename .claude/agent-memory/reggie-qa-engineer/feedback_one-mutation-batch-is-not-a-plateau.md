---
name: one-mutation-batch-is-not-a-plateau
description: A single clean mutation batch does not mean the suite is done — keep running fresh batches until one comes back all-caught, and target the wiring rather than the algorithms
metadata:
  type: feedback
---

When QUALITY-CHECKing a suite, do NOT stop after one mutation batch comes back
mostly-caught. Run fresh batches until an entire batch is caught with no fixes
applied.

**Why:** on `register.py` (2026-07-21) the WRITE-TESTS stage had already run 72
mutations with 69 caught and scored 9.20/10. Four further batches of *novel*
mutations found **21 more survivors** — and the survivor count only reached zero
in the fourth batch (10/15, then 7/12, then 4/10, then 8/10 caught on first
pass). Each round of fixes made the next round's survivors harder to find but
did not exhaust them. A single batch measures the batch, not the suite.

**How to apply:**
- Read the previous stage's harness files first (they are usually left in the
  scratchpad) so every probe you write is genuinely novel. Diff your candidate
  list against theirs before running.
- Aim at **wiring, not algorithms**. The prior stage's 72 probes were nearly all
  estimator algebra and guard clauses, and those were well covered. Every one of
  my 21 survivors was plumbing: a column copied once and never read back, a
  constant the fixture derives from itself, a call-site argument, a row-order
  assumption, a slice offset. Concretely productive probe families:
  - `frame[col] = source[col]` -> `np.zeros(...)` for every copied column
  - `sum(axis=1)` -> `max(axis=1)`, `== x` -> `!= x` (survives whenever the
    fixture is symmetric in that dimension)
  - a call site's argument swapped for a plausible sibling
    (`n_paragraphs` -> `para_words`)
  - `sorted(xs)` -> `list(xs)`, `[1:]` -> `[:-1]`, `<` -> `<=`
  - `constant.shape[-1]` -> a value that coincides on the current fixture
- Self-referential constants are the single richest vein: any test doing
  `for x in MODULE.CONSTANT` cannot detect a deletion FROM that constant. Anchor
  the constant against independent literals, and key the fixture off the literals
  too. Same family as [[mutation-check-recurring-holes]]'s
  constant-parametrized tests.
- Judge-supplied gap lists are a starting point, not the scope. The three gaps
  handed to me were all real, but they were 3 of 24 — the other 21 came from
  probing on my own.
