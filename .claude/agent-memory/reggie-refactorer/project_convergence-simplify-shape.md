---
name: convergence-simplify-shape
description: How to run the byte gate on convergence.py (12-min regeneration), and why SIMPLIFY there is net-POSITIVE lines by design
metadata:
  type: project
---

`src/presidential_profiles/convergence.py` (~1900 lines, the `convergence-analysis` task)
joins the byte-gate family ([[analysis-module-simplify-gates]]) with the most expensive gate
in the repo.

- **Byte gate:** `python -m presidential_profiles.convergence` writes 4 parquets +
  `convergence_meta.json` into `data/convergence/`; `git status --short data/convergence/`
  empty means byte-identical. Confirmed reproducible 2026-07-21 across a 5-change set.
  **It costs ~12 minutes at R=2000** (~4 of them the three-leg `_selftest`), so BATCH every
  `convergence.py` edit into one change set and gate once. Redirect through `tee`, not
  `| tail` — a pipe to `tail` swallows all progress output until the process exits.
- **A comment-only follow-up edit does not need a re-gate.** Prove it instead: reconstruct the
  gated source in memory and compare `ast.dump(ast.parse(...))` both ways. That took seconds
  and is a real proof, not a shrug.

**This module's complexity is load-bearing and the SIMPLIFY yield is +lines, correctly.**
An adversarial review established that the obvious form of the analysis manufactures its own
finding (rho = −0.605 from pure noise), so `build_convergence`'s explicitness, the selftest
legs and the arm/treatment separation are all deliberate. The only real deletion available was
one genuinely unreachable branch (`merge(how="left", validate="many_to_one")` cannot change the
row count — `MergeError` fires first). Everything else of value was **naming and disclosure**:
an inline `np.where` trust gate lifted into a named `paradox_status` beside the other three
(that inline-ness is exactly why it drifted out of the shared vocabulary test), a duplicated
`ends_2014` rule collapsed into `ends_2014_mask` for its two consumers, and a docstring
disclosing that two thresholds are post-hoc. Consistent with the repo pattern: here the
highest-value SIMPLIFY output is prose that names the real mechanism.

**Refactor hazard, still live:** `tests/test_convergence.py`'s `0.580 ± 0.005` and `0.274 ± 0.02`
are RNG-call-order pins on `speech_block_floor`. Any change to the number or order of `rng`
calls inside that function breaks them even when behaviour is correct. Intended — do not
"fix" the tests.
