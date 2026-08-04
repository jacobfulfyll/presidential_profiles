---
name: convergence-test-seams
description: Hermetic-test seams for presidential_profiles convergence.py (cluster-rarefied dispersion, permutation null, entropy-matched rival null) — hand-built Composition, tiny arms, the gated selftest bypass, and the rival-null corpus pair
metadata:
  type: reference
---

Authoring offline tests for `convergence.py` (the pre-registered agenda-convergence
test). Builds on [[test-toolchain]] and [[eras-test-seams]].

**Everything runs off a hand-built `Composition`.** `_synthetic_composition` is the
*selftest's* generator (it exists to carry the measured confound: drifting speech
supply, within-speech ICC) and its parameters are signature-bound, so it is a poor
fixture for behavioural tests. Build `C.Composition(...)` directly from an explicit
`[(president, [year per speech], bin-weight vector)]` spec, then drive the real
`build_design` / `dispersion_curve`. Rows MUST be grouped by `doc_name` or
`build_design` refuses.

**Use a tiny arm for design-level tests.** `ArmSpec("tiny","corex",None,2,2,"test")`
(S=2 x B=2) makes eligibility hand-reasonable. The corpus span must be >= 30 years
or `rolling_windows` returns an EMPTY grid (WINDOW_LEN=30). A "coextensive" corpus —
N presidents each speaking every other year 1800-1860 — puts exactly N eligible
presidents in all 16 windows, the cleanest handle on the `MIN_PRESIDENTS=3` floor.

**`DRAWS` / `FLOOR_DRAWS` are SIGNATURE-bound defaults** (`n_draws: int = DRAWS`), so
monkeypatching `C.DRAWS` is inert. `SELFTEST_CORPORA` / `SELFTEST_REPLICATES` /
`SELFTEST_PERMUTATIONS` / `ALPHA` / every `SELFTEST_*` threshold ARE read in the body
and monkeypatch fine — that plus a wrapper around `_synthetic_composition`
(n_presidents=10, speeches 8-10, 8 paras) runs the REAL `_selftest` in <0.3s. Use
`SELFTEST_CORPORA=2`, not 1: the legs report a `ddof=1` sd.

**`build_convergence` runs on a 6-president corpus in ~0.3s** with `_selftest`,
`build_compositions` and `ARMS` monkeypatched (arms must still be named
`corex_all` + `corex_sotu` or `score_decision_table` raises). Give presidents
DISTINCT agendas: `paradox_table`'s leave-one-out z divides by the sd of the OTHER
presidents' distances, which is 0 on identical presidents (and n<3 ranked
presidents gives a NaN / divide-by-zero RuntimeWarning).

**Rival-null (ALIKE vs BROADER) corpus pair — the construction that works.**
Verified stable across 4 corpus seeds x 2 draw seeds:
- BROADER (40 bins, 20 presidents, 6-yr terms): president p owns bin p, and puts
  fraction `b_p` (0 -> 0.95 linearly) on a flat background over all bins. Result:
  `mean_entropy` rho ~ +0.998 (0.5 -> 3.9), `dispersion` rho ~ -0.995 (0.98 ->
  0.58), and `excess_ratio` pinned in **[0.96, 1.08]** at every window.
- ALIKE (12 bins, uniform over exactly 3 of them, breadth fixed at log2 3):
  the share drawn from a common core {0,1,2} rises 0 -> 3. `dispersion` rho ~ -0.95,
  `excess_ratio` rho ~ -0.94, final `excess_ratio` ~ 0.18.
- **Assert on the excess_ratio LEVEL band, not its rho, for BROADER.** The rho of a
  flat-but-noisy series is unstable (-0.02 to -0.80 across seeds); the level band is
  rock solid and is the substantive claim. Same trap on ALIKE's `mean_entropy`.
- Random-support broadening does NOT work (chance alignment drifts; excess_ratio rho
  came out +0.48). The spike-plus-uniform-background construction does.

**Discriminating fixtures added at QUALITY-CHECK (mutation testing).** The
generic `_hand_corpus` draws a president's paragraphs i.i.d. from one weight
vector, which cannot express speech-level structure — so `_per_speech_corpus`
(a `bin_of(p, s, i)` callable, `None` = unlabelled, `president_step` staggers
careers) carries the tests that separate cluster from paragraph behaviour:
- **internally-pure speeches + `S=1`** -> every cluster draw is a PURE CORNER
  (max mass 1.0); the paragraph sampler comes out pure 0.5% of the time.
- **paragraph `i` in bin `i`** -> `c * S*B` integral AND some entry exactly
  `1/(S*B)` pins m from both sides.
- **mutually disjoint pure speeches** -> observed dispersion is exactly 1 bit,
  so the block floor's 0.580 (a mean of {0,1}, hence an exact multiple of
  1/n_draws) separates from a paragraph re-deal's 0.042 and a no-re-deal 1.0.
  Slot pools `[2, 10, 10]` at `S=8` separate preserved (0.260-0.274 over 8
  seeds) from equalized blocks (0.193-0.220).
- **rising no-topic share, disjoint substantive bins** -> `dispersion` rho
  -0.998 and `excess_ratio` rho -0.997 with ZERO real convergence: the boundary
  on prereg section 3's density-immunity claim.
- `_coextensive_corpus(2)` = every window on `suppressed_n_floor`;
  `_coextensive_corpus(3)` = every window in the trend and every leave-one-out
  empty. A/B/C over 1800-1900 plus a D over 1800-1830 gives jackknife retention
  15/36 — the only fixture that catches swapped `jackknife_status` arguments.

**Real-corpus anchors (2026-07-21, after WRITE-TESTS regeneration):** decision cell
`no_convergence`; selftest legs -0.089 / -0.406 / -0.970 / size 0.050; jackknife 111
rows (38 corex_all / 35 corex_sotu / 38 llm_all), ALL `ci_status="ok"`, worst window
retention 0.913 (Nixon & LBJ on corex_sotu; James Madison 96/105 on corex_all);
permutation_null 30 rows all `ok`; `n_windows_full_design` = 105 corex_all/llm_all,
**92 corex_sotu**. All five artifacts byte-identical across consecutive full runs
(~12 min each at R=2000).
