---
name: verification-convergence
description: How to verify the convergence layer (data/convergence/ + convergence.py) end-to-end — 19-min regeneration, prereg-vs-artifact scoring recipe, and the transitive-anthropic $0-guard hole
metadata:
  type: reference
---

Verifying `presidential_profiles.convergence` — a pre-registered inferential layer whose
deliverable is `notes/convergence-prereg-v1.md` **plus** the git ordering of two commits.

**Cost/time.** $0, zero network. Full regeneration `python -m presidential_profiles.convergence`
took **1,138 s / 19 min** (selftest ~2.5 min + 3 arms × 2,000 permutations). CONTEXT advertises
"~12 min" — budget 20 with any other Python running. Byte-identical on rerun AND after
`rm -rf data/convergence/` (`out_dir.mkdir(parents=True, exist_ok=True)`), so a delete+regenerate
is a free second determinism replicate. Meta has no wall-clock stamp; `MASTER_SEED = 20260721`
is a seed that *looks* like a date — a naive date-regex sweep will false-positive on it.

**The $0 proof needs three legs, and the obvious one is wrong.** `anthropic` IS in `sys.modules`
during a run: `convergence.py:72 from .eras import check_staleness` → `eras.py:59 from .annotate
import _rates` → `annotate.py:39 from anthropic.types...`. Pre-existing via era-atlas (`combat.py`
is clean). The module's own guard (`tests/test_convergence.py:1633`) only ASTs `convergence.py`'s
own source, so it cannot see this. Prove spend instead by:
1. a `sitecustomize.py` first on `PYTHONPATH` that patches `socket.socket.connect` /
   `connect_ex` / `create_connection` / `getaddrinfo` to raise + record, and dumps a JSON report
   at `atexit` — zero cost in numeric loops (only import + socket machinery is hooked);
2. a wrapper that imports the module, sets `anthropic.Anthropic.__init__` (and Async/Bedrock/
   Vertex) to raise, then calls `C.main([])` — the `combat.py` booby-trap pattern
   (`tests/test_combat_contracts.py:1044`), which `test_convergence.py` lacks;
3. `meta.api_calls == 0`.
Both full runs: zero network attempts, trap never fired.

**Scoring a pre-registration against its artifact.** Do not read the prereg for agreement —
recompute. What actually worked, all from the parquets alone:
- Re-derive `permutation_null.rho` with `scipy.spearmanr` over `dispersion_curves` rows where
  `used_in_trend` (and `& in_ends_2014` for that treatment). All 27 cells matched to 1e-9.
- Re-derive the **design** from `paragraph_issues` + `speech_annotations` without the module:
  qualifying = speech with ≥B paragraphs of the arm's genre; eligible = ≥S qualifying in-window
  speeches. This reproduced the prereg's pre-declared coverage exactly (105/105 min 3 median 5
  max 7; SOTU 101/105 with the four uncovered starts 1931/33/35/37; Trump in 5 windows).
- Permutation p-values must satisfy `p*(R+1) ∈ ℤ` — a cheap, sharp integrality check.
- Extract the prereg's decision-table headline sentences by splitting the markdown row on `|`
  and taking the last cell, then compare to `convergence.HEADLINES` **verbatim**. A regex over
  `"…"` spans grabs table plumbing and gives false negatives.
- `101 vs 92` for the SOTU arm is not a defect: 101 = windows where dispersion is *defined*
  (E≥2), 92 = windows entering a trend (E≥3, since E=2 is `suppressed_n_floor`).

**Live-fire the STOP gate cheaply.** Monkeypatch `C.SELFTEST_CORPORA=1`, `SELFTEST_REPLICATES=2`,
`SELFTEST_PERMUTATIONS=5` and `SELFTEST_CLUSTER_FLAT_MAX_ABS_RHO=0.0`, then
`build_convergence(out_dir=scratch, n_permutations=5)` → `AssertionError`, scratch dir never
created, committed shas unchanged. ~30 s instead of 19 min.

**Coherence read that catches a fabricated story.** Spearman every *component* against
`window_center`, not just the headline: `mean_within_president_entropy` ↑, `entropy_matched` ↓,
`floor` ↑. Broader agendas mechanically predict smaller pairwise JSD, so `entropy_matched` MUST
fall if entropy rises — if it does not, the rival-null arm is broken. See
[[verification-combat]] and [[verification-era-atlas]] for the sibling derived layers.
