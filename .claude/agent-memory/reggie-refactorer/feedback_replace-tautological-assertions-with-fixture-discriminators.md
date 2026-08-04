---
name: replace-tautological-assertions-with-fixture-discriminators
description: When SIMPLIFY is handed a "this assertion is an algebraic identity" finding, build a fixture where the identity breaks — do not just rename the test
metadata:
  type: feedback
---

When a work order says an assertion is tautological (guaranteed by `np.clip`, by a symmetry,
by an identity at n=2), replace it with a fixture large or uneven enough that the identity
stops holding, rather than renaming the test and striking the docstring claim.

**Why:** on `convergence-analysis` three separate assertions were found to be algebraically
true — `0 <= floor <= 1` (guaranteed by `np.clip`), `sum(axis=-1) == 1` (true of any mean of
rows summing to 1 at ANY S and B, so it passed against the *broken* estimator the whole task
exists to avoid), and `per_slot == [value, value]` at two slots (symmetric JSD + zero diagonal
⇒ both entries equal the single pairwise distance for ANY re-deal). Each "passed" for the
wrong reason and manufactured confidence. Renaming would have preserved the hole.

**How to apply:**
- Find the parameter that makes the identity collapse — usually a count (2 slots → 3),
  an unevenness (equal pools → `[2, 10, 10]`), or a degenerate S/B (S=1 makes "a draw is a
  whole speech" *observable* rather than statistical).
- Assert an ORDERING or an ARGMAX, not a magnitude, when the magnitude is still identity-bound:
  at 3 slots `mean(per_slot) == value` is *still* an identity, but *which* slot is largest is
  a fact about the pools the call was handed. Permute the inputs and assert the answer follows.
- Always assert the fixture HAS the property first ("the two samplers genuinely differ here"),
  so a re-seed fails loudly instead of going vacuous.
- Verify the discriminator across ~4 seeds before committing to the tolerance. Mine had
  ~0.06-0.08 of separation against an asserted floor of 0.02.

**The same rule covers every guard you ADD or TOUCH, not just tautologies you are handed.**
Construct the input where the property is false and confirm the check fails. Three shapes
found this way on the `bands.py` line of work, none by rereading:
- **A y-value assertion over a symmetric fixture compares a number to itself.** Vary the
  fixture on every axis the assertion distinguishes (period AND series), then mutate the
  source to mispair them. See [[fix-fixture-generators-by-documenting]].
- **An AST/source guard matching a substring will match the DOCSTRING.** Extracting a
  duplicated predicate into a named helper broke a guard asserting the gate expression names
  a column; teaching the guard to follow the helper then made it pass on a helper whose
  *body* no longer read the column, because its docstring said the name. Strip docstrings
  (`ast.Expr` wrapping a `str` `ast.Constant`) from the body before matching, and re-run
  the mutation.
- **Writing a re-derivation assertion often falsifies the prose it pins.** A docstring naming
  "the thinnest era at 55 speeches / 1,193 paragraphs" conflated two axes — one era is
  thinnest by speeches, a different one by paragraphs. Invisible to rereading; immediate once
  the assertion had to call `idxmin()` per axis. Prose pins belong in the existing
  re-derivation class, not a new one ([[analysis-module-simplify-gates]]).
