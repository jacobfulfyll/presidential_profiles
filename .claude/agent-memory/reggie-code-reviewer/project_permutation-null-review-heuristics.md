---
name: permutation-null-review-heuristics
description: How to review a permutation/bootstrap null in this repo — the "the null absorbs it by construction" claim is testable in ~2 minutes and was FALSE on convergence-analysis
metadata:
  type: project
---

Found on `convergence-analysis` (2026-07-22, REVIEW) after five prior gates missed it.
The module's load-bearing justification was **"the permutation null absorbs the
estimator's residual bias by construction, so the inference is unaffected."** It is false,
and three cheap executed checks prove it. Run all three on any resampling null here.

**Why:** this repo's dominant defect class is a *true conclusion supported by an argument
that does not establish it* (see [[analysis-report-audit-heuristics]]). A null's
"absorbs by construction" clause is exactly that shape — it reads as a proof and is
actually an empirical claim nobody measured.

**How to apply — three checks, all executable in minutes:**

1. **Compare the null's own centre to the estimator's centre on a true-null corpus.**
   The module ships its own generator (`_synthetic_composition`) and its own null
   (`_donor_map` + `dispersion_curve(donor=...)`), so ~30 corpora x 150 permutations is
   ~3 minutes. Measured: estimator centre **-0.047 (SE 0.026, n=32)**, permutation-null
   centre **-0.002**. Absorption would require these to coincide; they do not.
   The published artifact says the same thing in one line — `null_mean` on
   `permutation_null.parquet` was **+0.014** while the estimator's own bias was -0.089.
   *Read `null_mean` first; if it is ~0 and the estimator is not, absorption is false.*
2. **Ask what the null holds fixed vs what it re-randomizes, then measure the drift it
   claims to absorb.** Here the observed curve samples each president's **in-window**
   speech pool and the permuted curve samples the donor's **global career** pool
   (`convergence.py:914-917`). That structurally decorrelates pool size from time:
   on the real design, Spearman(window centre, mean pool size) is **+0.846 observed
   -> +0.099 permuted**. A nuisance the null erases is a nuisance it cannot absorb.
3. **Check that the size/calibration leg runs the CONFIGURATION the real run uses.**
   `_selftest` leg (c) shortened its corpus to 24 presidents / 46 windows for runtime,
   with a comment arguing precision depends on replicate count, not corpus length. True
   about precision, false about the estimand: the estimator's bias is **-0.016 at 46
   windows** and **~-0.06 at 106 windows** (the real run's 105). The gate ran where the
   defect it gates does not exist. Re-run the size leg at the real window count — here
   it still passed (2/32 = 6.25% at nominal 5%), so WARNING not blocker, but only
   because I measured it.

**Fourth check, different class — Monte-Carlo noise of the point estimate.** Re-run the
published curve at ~20 fresh seeds on the same data/design. Measured replication sd of
the primary rho: **0.030** (`corex_all`), **0.039** (`corex_sotu`) at D=20 draws. The
published primary rho was **-0.0047** — i.e. an order of magnitude smaller than its own
rarefaction noise, and every leave-one-out jackknife delta (leave-Trump-out +0.0225,
delta +0.027) is inside one MC sd. Nothing in the artifact publishes this, so any prose
reading a jackknife delta as a president effect is unsupported. **Always measure the
seed-to-seed sd of a Monte-Carlo point estimate before letting a note interpret deltas.**

**Calibration:** the *conclusion* survived all four checks (no-convergence, and the
residual points toward convergence, which is what makes it conservative). Only the
*justification* was wrong. Block on the justification anyway — it was mandated into the
findings note by a carry-forward, and a false "by construction" in a deliverable is the
failure the falsification value exists to prevent.

**Two fail-open guards worth grepping for in any module like this:**
- A string-selector between a correct and a superseded algorithm written as
  `good if s == "x" else bad` (`convergence.py:900`): `None`, `""`, `"CLUSTER"` and every
  typo silently select the broken branch. Sibling `build_design` validates its two enums
  with raises — the inconsistency inside one file is the tell.
- An artifact loader that guards the missing FILE but not the missing COLUMNS
  (`_stylistic_anchor`): renaming the vector columns yields **0.0 for every president**
  plus a provenance string still naming the file. Fabricated magnitude, forbidden here.
