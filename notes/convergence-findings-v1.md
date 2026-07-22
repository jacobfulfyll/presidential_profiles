# Have presidential agendas converged? — findings v1

**Answer: no.** Across 240 years there is no significant decline in between-president agenda
dispersion on either primary arm. The pre-committed headline for the cell that fired:

> Across 240 years we find no significant decline in between-president agenda dispersion; at 68%
> power this is weak evidence against convergence, not a refutation of it.

Companion: [`convergence-prereg-v1.md`](convergence-prereg-v1.md), committed **before** any result
existed (`41d3b22`, one file, nothing under `src/` or `data/`; an ancestor of every commit that
touches an artifact). That ordering is the whole point — the decision rules below could not have
been chosen after seeing the curves. **The pre-registration has never been amended.** Where this
note disagrees with it, the disagreement is disclosed here rather than edited into it (§7).

Every number below was re-derived from `data/convergence/*.parquet`, not copied from prose. §8
states exactly which cells were verified and which were not.

---

## 1. The result

Primary statistics, rolling 30-year windows stepped 2 years, all windows, R = 2,000 president
permutations:

| arm | role | rho | one-sided p | perm. 5% critical value | significant decline |
|---|---|---|---|---|---|
| `corex_all` | **primary** | **−0.0047** | 0.4808 | −0.5246 | No |
| `corex_sotu` | **co-primary** | **+0.4254** | 0.8891 | −0.5514 | No |
| `llm_all` | sensitivity | +0.1895 | 0.7456 | −0.5283 | No |

Nothing is close. The primary arm's rho sits **0.15 seed-replication standard deviations from
zero** (§6.3) — flat at the estimator's own resolution — and the co-primary arm points the *other
way*. The decision table's rule 1 (arms disagree in direction *and* significance) does not fire
because neither arm is significant; rule 2 fires, giving the headline above.

Note how far the honest critical value sits from the naive one. The adversarial review that
rebuilt this design measured the overlapping-window Spearman p-value at **58% type-I error**; its
honest one-sided 5% cutoff was rho ≤ −0.41 rather than the −0.16 a formula would have accepted.
The permutation null puts it at **−0.52 to −0.55** depending on arm. A design using
`scipy.spearmanr`'s p-value would have had roughly a coin-flip's chance of announcing convergence
from noise.

### All statistics, both grids

| arm | statistic | rho | one-sided p | two-sided p |
|---|---|---|---|---|
| `corex_all` | dispersion | −0.0047 | 0.4808 | 0.9580 |
| `corex_all` | floor_corrected | −0.2175 | 0.2764 | 0.5657 |
| `corex_all` | excess_ratio | +0.2300 | 0.7351 | 0.5192 |
| `corex_all` | pair_regression | −0.1584 | 0.1284 | 0.2719 |
| `corex_sotu` | dispersion | +0.4254 | 0.8891 | 0.2034 |
| `corex_sotu` | floor_corrected | +0.2915 | 0.7621 | 0.4923 |
| `corex_sotu` | excess_ratio | +0.7833 | 0.9965 | **0.0095** |
| `corex_sotu` | pair_regression | +0.1530 | 0.8546 | 0.2799 |
| `llm_all` | dispersion | +0.1895 | 0.7456 | 0.5257 |
| `llm_all` | floor_corrected | −0.0665 | 0.5882 | 0.8461 |
| `llm_all` | excess_ratio | +0.2950 | 0.8411 | 0.2879 |
| `llm_all` | pair_regression | −0.0702 | 0.2769 | 0.5507 |

Sensitivity grids agree. Dating subset (windows ending ≤ 2014): +0.0970 / +0.3995 / +0.2767 — no
sign flip, so the result is not an artifact of the Trump years. Non-overlapping windows (n = 8, low
power stated in advance): −0.2143 / +0.6429 / +0.2619.

**The one bolded cell is not a finding.** `corex_sotu`'s `excess_ratio` is nominally significant
two-sided (p = 0.0095) — in the **anti-convergence** direction. The design is one-sided, so it
scores nothing and `significant_decline` is correctly `False`. It is reported here as an
**unregistered two-sided observation**, never as a result.

---

## 2. Why this is not the obvious analysis

The obvious version of this study **manufactures the finding**. Two independent mechanisms each
produce a convincing convergence decline out of nothing:

1. **Rarefying paragraphs does not rarefy independent observations.** Issue labels cluster within
   speeches, and the number of distinct speeches behind a president's in-window paragraphs climbs
   steeply with time. On a clustered null with zero real convergence, the paragraph-rarefied
   estimator returns a strong spurious decline.
2. **Overlapping-window p-values are invalid by orders of magnitude** (58% type-I error at nominal
   5%).

The rebuilt design rarefies **speech clusters** (S = 8 speeches × B = 6 paragraphs; SOTU arm
S = 4 × B = 12, both m = 48) and calibrates inference by **full-pipeline president permutation**.
`scipy.spearmanr`'s p-value is banned everywhere in this analysis, including the jackknife and every
control curve; the one call site returns a bare statistic and every published row records
`scipy_pvalue_used = False`.

### The gate that had to pass first

`_selftest` runs before any real number is computed, and a failing leg aborts before `out_dir` is
even created. Measured on 12 synthetic corpora (legs a/b) and 40 replicate corpora × 150
permutations (leg c):

| leg | what it proves | measured | threshold |
|---|---|---|---|
| (a) clustered null, **cluster**-rarefied | flat where there is nothing | **−0.089** (sd 0.136) | \|rho\| ≤ 0.25 |
| (a) same corpora, **paragraph**-rarefied | the old design still breaks | **−0.406** (sd 0.087) | ≤ −0.30 |
| (b) injected convergence | it can still detect a real one | **−0.970** (sd 0.014) | ≤ −0.50 |
| (c) permutation-null size | the inference is calibrated | **0.050** | ≤ 0.152 |

Leg (a) is a *pair*, and that is what makes it meaningful: on the **same corpora**, the superseded
estimator reproduces a strong spurious decline while the pre-registered one does not. "Flat"
therefore means "the fix works", not "there was no signal to find".

> **Do not quote −0.605 as this module's output.** That is the adversarial review's figure on real
> data. This module's synthetic-corpus contrast is **−0.406**; it claims only the same sign and
> shape.

---

## 3. The more interesting result: breadth, not convergence

The headline is a null. The component trends are not, and they cohere into a mechanism.

Spearman vs window centre, rolling grid, trend-eligible windows:

| arm | dispersion | floor | floor_corrected | entropy_matched | excess_ratio | mean entropy |
|---|---|---|---|---|---|---|
| `corex_all` | −0.005 | +0.171 | −0.218 | −0.439 | +0.230 | +0.326 |
| `corex_sotu` | +0.425 | +0.308 | +0.292 | −0.763 | +0.783 | +0.710 |
| `llm_all` | +0.189 | +0.216 | −0.066 | −0.868 | +0.295 | +0.871 |

Read left to right:

1. **Agendas genuinely got broader.** Mean within-president entropy rises in all three arms —
   `corex_all` 2.751 → 2.954 bits, `corex_sotu` 2.900 → 3.233, `llm_all` 3.257 → 3.730.
2. **Broader agendas mechanically look more alike.** Two presidents who each touch everything
   overlap more by construction. The entropy-matched null — which asks "how much dispersion would
   we see from breadth *alone*?" — duly collapses (−0.44 to −0.87).
3. **But observed dispersion does not fall.** So `excess_ratio` (observed ÷ entropy-matched)
   **rises** in every arm.

**Presidents became less alike than their own broadening predicts.** The mundane rival explanation
for an apparent convergence — everyone now talks about everything — is exactly what the
entropy-matched null was pre-registered to separate out, and the data separates *away* from
convergence rather than toward it.

This is the result most likely to be misread, so state it precisely: it is **not** evidence of
divergence. No `excess_ratio` trend is significant against its permutation null (one-sided p 0.735
/ 0.997 / 0.841). The claim is narrower and sturdier — the intuition that modern presidents sound
interchangeable is picking up **breadth**, and breadth does not survive being controlled for.

---

## 4. Coverage

| arm | rolling windows | used in trend | eligible presidents (min/median/max) | ever eligible | bins | qualifying speeches |
|---|---|---|---|---|---|---|
| `corex_all` | 105 | **105** | 3 / 5 / 7 | 38 | 16 | 923 |
| `corex_sotu` | 105 | **92** | 1 / 5 / 6 | 35 | 16 | 216 |
| `llm_all` | 105 | **105** | 3 / 5 / 7 | 38 | 51 | 923 |

**On the SOTU arm's 92, which is not the number the pre-registration advertises.** Prereg §4.4
pre-declared coverage as **101 of 105** — windows where dispersion is *defined* (≥ 2 eligible
presidents) — and correctly named the four uncovered windows (starting 1931, 1933, 1935, 1937).
But §4.4's own trust gate suppresses windows with exactly 2 eligible presidents, so **92** is the
number that enters a trend. Both are true of different quantities; 92 is the decision-relevant one
and leads here. The two bases coincide on the primary arm (105), which is why the gap is invisible
there. The prereg is not amended.

**216 of 218, not 218.** Prereg §2.2 describes the SOTU arm as drawing on "the 218" typed annual
messages; §4.1's B = 12 floor leaves **216** qualifying. The two excluded are Washington's first
(8 paragraphs) and second (10 paragraphs) annual messages.

The `other` speech-type bucket (8 speeches) enters the CorEx and LLM arms and is excluded from the
SOTU arm — stated rather than silently dropped.

---

## 5. The secondary hypotheses

**H2a — "Trump is the most rhetorically distinct president" — is an ANCHOR and scores zero.** It
was already published in this project's README before the pre-registration was written; scoring it
would be confirmation theater. Read-only, it reproduces: Trump's mean stylistic similarity to the
other 37 presidents is **0.7447**, the **lowest of all 38**. `anchor_is_scored = False` on every
row, and it is not an input to the decision table.

**H2b — "Trump is nonetheless agenda-typical" — is the only scored half, and it is not
significant.** Bias-corrected agenda distance 0.0122 against the other presidents' mean of 0.0285;
**rank 6 of 38**; percentile 0.158; z = −0.93. The pre-registered support rule (percentile ≤ 0.50
**and** below the others' mean) is met — but §7.2 also defines the one-sided p *as* that
percentile, and **0.158 > α = 0.05**. The row carries `ci_status = low_cluster_caution`: Trump is
eligible in **5 of 105 windows**, and they heavily overlap.

The honest phrasing is *"supported by the pre-registered criterion, at p = 0.158, on 5 overlapping
windows"* — never "significant", and never "Trump is agenda-typical" without the caveat. H1's Trump
clause was demoted to a stated boundary condition for the same reason: 95% of the curve contains no
Trump paragraphs at all.

### Jackknife

No single president's deletion moves any arm across its critical value.

| arm | rho range across 38/35/38 deletions | any significant | leave-Trump-out |
|---|---|---|---|
| `corex_all` | −0.2323 … +0.0636 | No | **+0.0225** (p 0.5167) |
| `corex_sotu` | +0.2241 … +0.6118 | No | +0.4101 (p 0.8796) |
| `llm_all` | −0.0577 … +0.2296 | No | +0.1567 (p 0.7101) |

The most influential deletion on the primary arm is leave-Monroe-out (−0.2323, p 0.2349) — still
far from −0.5246. **But see §6.3: none of these deltas is interpretable as a president effect.**

---

## 6. What is wrong with this analysis

### 6.1 The estimator carries a residual bias, and the permutation null does *not* absorb it

This corrects a claim that survived several review rounds inside this project before being caught
by re-deriving it from the artifact.

The cluster-rarefied estimator's centre on **true-null** synthetic corpora is **−0.089**
(sd 0.136, n = 12 corpora) — not zero. A **4.5×** improvement on the paragraph-rarefied −0.406, but
not elimination. Mechanism: sampling speeches with replacement removes the finite-population correction
from the *sample*, but not from the precision of a president's realized in-window pool mean, which
scales as 1/n while the speech supply drifts upward across the corpus.

It is tempting — and wrong — to say the permutation null absorbs this because it re-runs the same
estimator with the design held fixed. **It does not.** The observed curve samples each president's
**in-window** pool, whose mean size climbs 17 → 36 speeches across the corpus (Spearman vs window
centre **+0.85**). A permuted slot samples the donor's **global career** pool, and a random
bijection over donors decorrelates pool size from window position (the same Spearman falls to
**+0.06**). The permutation destroys the very supply drift that creates the bias, so the null
cannot contain it.

The artifact says so in one column: `selftest.a_cluster_null_rho` is **−0.0893** while the null's
`null_mean` for `corex_all`/dispersion is **+0.0142**. If the bias were absorbed those would
coincide; they differ by essentially the whole bias.

**Consequence, stated plainly because it is load-bearing.** The test is **mildly anti-conservative
toward convergence** — a residual pointing at a decline makes a decline *easier* to detect.
Measured size at the run's own window count is ≈ 6.3% at nominal 5%. This does **not** invalidate a
no-convergence finding: a thumb on the scale in favour of the alternative, on a run that found none,
leaves the conclusion **conservative**. It would matter a great deal to a future run that *did*
find a decline, which would need this subtracted before the decline could be believed.

Prereg §5 contains the same false "absorbs … the drifting speech supply" wording. It is committed
pre-registration and is **not** edited; this section is the disclosure.

### 6.2 Three residuals, all pointing the same way

| residual | direction | absorbed? |
|---|---|---|
| estimator null-centre bias (−0.089) | toward convergence | **no** (§6.1) |
| rising no-topic share | toward convergence | partly — `floor_corrected` yes, `excess_ratio` **no** |
| observed in-window vs null global-career pools | toward convergence | n/a — it *is* the §6.1 mechanism |

On the second: the composition equation gives every paragraph mass 1 split across its labels, so
label-density drift cannot move the measure *by itself*. That does **not** make it immune to the
**no-topic bin's share** rising, which pulls every composition toward one shared corner and
mechanically compresses pairwise distance. `floor_corrected` largely absorbs this (the floor
inherits the window's no-topic share); **`excess_ratio` does not** and will read a rising no-topic
share as "more alike". Prereg §3's immunity claim is narrower than it reads.

All three push **toward** convergence while the answer is **no convergence**. That is what makes
the finding conservative rather than fragile.

### 6.3 Monte-Carlo noise exceeds the point estimate, and no jackknife delta is interpretable

At D = 20 draws per window, re-running the primary arm's curve on 10 fresh seeds (production
configuration, same data, same design) gives sd = **0.032**, mean −0.008, range −0.065 … +0.047.

- The published −0.0047 is **0.15 sd** from zero. Flat, and not meaningfully distinguishable from
  any other near-zero value.
- **Leave-Trump-out's delta (+0.027) is 0.86 sd.** So are the rest. **No jackknife delta on this
  arm may be read as a president effect** — the jackknife shows robustness (nothing crosses the
  critical value), not influence.

Nothing in `jackknife.parquet` reveals this: its `ci_status` prices *window retention*, not
sampling noise. This paragraph is the only place it is recorded.

### 6.4 Trust gates, and one threshold chosen after the fact

`ci_status` is on every plottable row of all four tables. `JACKKNIFE_SUPPRESS_RETENTION = 0.75` and
`JACKKNIFE_CAUTION_RETENTION = 0.90` are **post-hoc** — chosen while writing the tests, when the
retention distribution was already computable. Acceptable because the gate can only *downgrade* and
no pre-registered inference reads it, but it is disclosed rather than backdated. Worst observed
retention is **0.913** (Nixon and Lyndon Johnson on `corex_sotu`, 84/92; Madison 96/105 on the two
105-window arms) — every row lands `ok`, but only 1.3 points above the caution line. Prereg §7.1
declares "deleting one president barely perturbs the design" as prose with no operational
threshold; 0.913 is its empirical support, now checkable.

The per-window speech-block floor is published per window and is emphatically **not**
time-constant: it drifts 59.0% / 48.6% / 65.1% of its own minimum across the three arms. Publishing
it as a scalar, as the original design would have, would have been wrong.

### 6.5 Frozen-taxonomy and artifact limits

- `taxonomy_v1` is frozen and has known gaps: no standalone Education topic, no modern monetary
  policy, no Prohibition, no standalone Terrorism, and 7 of 50 level-2 topics have no legacy parent.
  The LLM arm inherits all of them.
- The LLM arm uses **native** level-2 labels, deliberately avoiding `triangulate.py`'s union
  projection. Its dispersion **level** is still not comparable to a CorEx arm's — different bin
  count (51 vs 16) and different budget. **Only trends within an arm are compared, never levels
  across arms.**
- No annotator-disagreement component is propagated into these curves
  (`ci_components = "permutation_trend_only"`).
- `indices.py`'s `MARKERS` regexes are era-biased and are not used anywhere here.

---

## 7. Where this note and the pre-registration disagree

The pre-registration is never edited. Three divergences, all disclosed above and repeated here so a
reader scoring the study can find them in one place:

1. **§5's "the drifting speech supply"** is listed among what the permutation null absorbs. It does
   not (§6.1).
2. **§4.4's 101-of-105 SOTU coverage** is a different base from the 92 that enters a trend (§4).
   §2.2's "the 218" SOTU speeches is 216 after the B = 12 floor.
3. **§10's "`convergence.py` never imports `anthropic`"** is true of the *file* and false of the
   *process*: `convergence` → `eras.check_staleness` → `annotate` → `anthropic.types`, inherited
   from the era-atlas layer. **The $0 guarantee holds regardless** — no client is ever constructed
   and no socket connection is attempted, both verified at runtime, and `api_calls: 0`.

None of the three changes a published number or the decision cell.

---

## 8. Verification boundary

**Conventions.** Every figure is rounded **from the full-precision value in one step** — never
rounded twice. (Drafting this note produced one such defect: "4.6×" computed from the already-
rounded 0.406/0.089, where the fraction gives 4.55 → **4.5×**. Double-rounding has now produced
wrong cells in three separate notes in this project.) Correlations are Spearman rho on window-level
series; `rho` and `p` are reported at 4 decimal places as stored, other quantities at the precision
shown. Dispersion is Jensen-Shannon divergence in **bits**; entropy likewise. Shares are fractions,
not percentage points, except where a figure carries an explicit `%`.

**Every number in this note is pinned by a re-derivation test**
(`tests/test_convergence_note_claims.py`), so a future change to the artifacts fails CI rather than
silently falsifying the prose.

**Verified by re-derivation from the artifacts** (not read off prose):

- All 27 published rho values, recomputed independently from `dispersion_curves.parquet` with
  `scipy.spearmanr` — **27 of 27 match at 1e-9**.
- Coverage counts, eligible-president distributions, and the four uncovered SOTU windows,
  recomputed from the raw corpus without module code.
- Every arithmetic identity: `floor_corrected = dispersion − floor`,
  `excess_ratio = dispersion ÷ entropy_matched`, `corrected_distance`, ranks, percentiles, z.
- Determinism: two full R = 2,000 regenerations, one after deleting `data/convergence/` outright,
  both byte-identical to the committed artifacts.
- The $0 guard, at runtime: a build with every Anthropic client constructor booby-trapped
  completes; a socket-level block records zero connection attempts.
- The pre-registration ordering, from git.
- The component trends, entropy levels, jackknife ranges and floor drift quoted in §3–§6.
- The seed-replication sd in §6.3 (10 seeds, production configuration).

**Not verified / known open:**

- The `pair_regression` statistic is the one published number a reader **cannot** re-derive from
  the artifacts — its pair accumulators are never written out. Reproducible only by re-running.
- `n_windows` is not carried on `permutation_null.parquet`; join it from
  `dispersion_curves.parquet`. Prereg §7.4's "(99 windows)" for the dating subset holds for
  `corex_all` and `llm_all`; the SOTU arm's is 86.
- Non-overlapping windows publish `window_center` for the truncated final window (1999–2026) as
  2013.5, where the true midpoint is 2012.5. Harmless to every rho (Spearman is rank-based and the
  centres remain strictly increasing), but it is a 1-year mislabel on 3 rows.
- The 68% power figure is carried from the pre-registration's design analysis and was not
  re-derived here.

---

## 9. Reproducing this

```
PYTHONPATH=src arch -x86_64 .venv/bin/python -m presidential_profiles.convergence
```

Pure local compute over frozen parquets: **$0, zero API calls**, ~19 minutes at R = 2,000.
Byte-identical on rerun on any date — there is no wall-clock stamp anywhere in the meta, so a dirty
`git status data/convergence/` means the numbers moved, not that the clock did. Provenance identity
is the `corpus_fingerprint`; *when* it ran is git's job.

A run below the pre-registered R = 2,000 refuses to write the published layer; pass `--out-dir` to
send it elsewhere. The three-leg gate cannot be skipped without a verbatim token, and a bypassed
run may not write the published layer at all.

Corpus: 1,057 speeches, 36,229 paragraphs, 45 presidents, 1789–2026.
