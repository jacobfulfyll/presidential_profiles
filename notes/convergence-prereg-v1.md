# Pre-registration v1 — Have presidential agendas converged?

**Status: PRE-REGISTRATION. Written and committed BEFORE any result of this analysis existed.**
The git history is the proof: this file lands in a commit that touches nothing under `src/` or
`data/`, and the commit that produces `data/convergence/` comes after it. Nothing in this document
may be amended after results are seen. If the findings note later wants to report something this
document does not contain, that thing is **exploratory** and must be labelled exploratory — the fix
for a missing row here is to strike the claim there, never to backdate a row into this file.

---

## 0. Why this design is strange

The obvious version of this analysis **manufactures the finding**. An adversarial review on
2026-07-13 (4 attack lenses, 41 raw attacks, 22 distinct, 13 confirmed / 9 partial, **0 rejected**,
each verified against this repo's real data) established two independent mechanisms, either of which
alone produces a convincing decline out of nothing:

1. **Rarefying paragraphs does not rarefy independent observations.** Issue labels cluster within
   speeches (within-speech ICC 0.05–0.17; mean 0.059, War & military 0.119, Money & banking 0.082).
   The number of distinct speeches behind a president's 50 in-window paragraphs rises from ~10
   (1800–1850) to ~38 (2000+) — Spearman(window centre, n_speeches) = **+0.450, p = 5.8e-35**. Early
   windows therefore carry systematically *less* information at identical paragraph count, holding
   the Jensen–Shannon noise floor higher early and letting it decay thereafter. On a clustered null
   with **zero real convergence, the paragraph-rarefied estimator returns rho = −0.605**.
2. **The p-values were invalid by one to three orders of magnitude.** 30-year windows stepped 2 years
   share 93% of their content; there are ~8 independent blocks in 240 years. Measured type-I error at
   nominal 5%: **58.5%** (at nominal 1%: 47.0%). The honest one-sided 5% critical value for that
   design is rho ≤ −0.41, not −0.16.

Everything below is the rebuilt design. It is deliberately more expensive than the question seems to
warrant. It is not to be simplified back toward the original.

---

## 1. Hypotheses

**H1 (primary).** Between-president dispersion of *agenda composition* has declined over
1789–2026: presidents talk about a more similar mix of issues now than they used to.
Directional (decline), one-sided.

**H1 boundary condition (stated, not a hypothesis).** Donald Trump has paragraphs in only **5 of the
105 rolling windows** — 95% of the curve contains no Trump paragraphs at all. H1 is therefore a claim
about the 240-year trend, **not** a claim about Trump. Any statement about *when* a decline began, or
about whether it survives the present era, uses the separate dating procedure in §7.4 (trend
re-estimated on windows ending in or before 2014), which cannot be driven by Trump-era windows.

**H2a — ANCHOR, SCORES ZERO.** "Trump is the most rhetorically distinct president." This is
**already published in this repo's README** (mean cosine similarity to the other 44 presidents,
`data/president_embeddings.parquet`). Pre-registering an already-known result would be confirmation
theater. It is carried in `paradox.parquet` as a labelled anchor column so the paradox can be read
in one place, and it contributes **zero** evidential weight to any conclusion in the findings note.

**H2b (test).** Conditional on H2a, Trump is nevertheless *agenda-typical*: his issue composition
sits closer to his contemporaries than the median president's does to theirs, once the estimator's
bias floor is subtracted. Directional (Trump more typical than median), one-sided.

**Rival hypothesis (mandatory, §6).** A decline in between-president dispersion can arise with no
increase in agreement at all, simply because each president's own agenda got **broader** (flatter
composition ⇒ mechanically smaller Jensen–Shannon divergence between any two of them). "More alike"
and "each broader" must be separated, or the headline is unearned.

---

## 2. Unit of analysis, window grid, and arms

### 2.1 Windows
Rolling windows of **30 calendar years, inclusive `[start, start+29]`, stepped 2 years**, starting at
the corpus's first year:

    start ∈ {1789, 1791, …, 1997}   →   105 windows, spanning 1789–2026

Window centre is `start + 14.5`. Every corpus year 1789–2026 falls inside at least one window.

*Declared deviation.* The 2026-07-13 review memo used a grid of **104** windows anchored one year
later. Anchoring at 1789 covers the corpus's first year and its last (2025–2026 hold 562 paragraphs
across 11 speeches, which the memo's grid drops). The consequential facts are unchanged: Trump is
present in **5** windows on both grids. This file's numbers are stated against the 105-window grid;
where the findings note quotes a review-memo figure it must say which grid it is on.

A second, **non-overlapping** window set is declared here as a low-power sensitivity (§7.5):
`start ∈ {1789, 1819, …, 1999}`, 8 windows, the last truncated at the corpus end (1999–2026).

### 2.2 Arms
| Arm | Label set | Speeches | Role |
|---|---|---|---|
| `corex_all` | 15 anchored CorEx issues + no-topic bin (**16 bins**) | all 1,057 | **PRIMARY** |
| `corex_sotu` | same 16 bins | the 218 `state_of_the_union_or_annual_message` speeches | **CO-PRIMARY** |
| `llm_all` | `taxonomy_v1`'s 50 level-2 topics + no-topic bin (**51 bins**) | all 1,057 | sensitivity |

**No vote rule.** A ">= 3 of 4 methods decline" confirmation rule fires **12–46% under a true null**;
the arms are not independent evidence. The decision table (§8) reads the primary and the co-primary,
in a fixed order, and nothing else.

**Never compare dispersion LEVELS across arms.** The arms have different bin counts and different
sampling budgets, so their Jensen–Shannon levels are not commensurable. Only the *trend within an
arm* is interpreted. This rule is doubly binding on `llm_all`: it also inherits the known
`union-projection-inflates-wide-crosswalk-issues` limitation of the LLM layer.

*Why `llm_all` uses the native 50 level-2 topics rather than the legacy 15.* Projecting LLM topics
onto the legacy 15 goes through `triangulate.py`'s **union** crosswalk, which inflates apparent
breadth for wide-fan-out issues and is never sensitivity-tested; the crosswalk is also
one-directional (7 of 50 level-2 topics have no legacy parent). Using the native level-2 label set
avoids that projection entirely. It is the reason `llm_all` has 51 bins and therefore the reason its
levels can never be compared to the CorEx arms'.

**Genre bucket disposition (explicit, not silent).** `speech_annotations.speech_type` has an `other`
bucket of **8 speeches**. Those 8 land in `corex_all` and `llm_all` (which take every genre) and are
**excluded from `corex_sotu`**, which is defined by a single positive speech type. No speech is
dropped from any arm for lack of a type: all 1,057 speeches are typed.

---

## 3. Composition — an equation, with an explicit no-topic bin

Let a paragraph *i* carry label set `L_i` (|L_i| = k_i) drawn from the arm's `K−1` substantive bins.
The arm's `K`-th bin is **"no topic"**. For a sampled multiset of paragraphs *M* (|M| = m):

    c[j]  =  (1/m) · Σ_{i∈M} [  1{k_i ≥ 1} · 1{j ∈ L_i} / k_i   +   1{k_i = 0} · 1{j = no-topic}  ]

Every paragraph contributes **total mass exactly 1**, split evenly across its labels; a paragraph
with no label contributes its whole mass to the no-topic bin. Therefore `Σ_j c[j] = 1` identically,
and **label-density drift cannot move the measure by itself** — which matters, because the corpus's
labels-per-paragraph drifts from 1.49 (1920s) to 0.83 (2020s), 28.7% of paragraphs carry zero
anchored issues and 33.4% carry two or more (max 8).

For `llm_all`, `topics` is normalized **before** any counting, in this exact order, with
`attention.canonical_label_map` / `attention.normalize_topics` (one normalizer, used throughout;
`taxonomy._norm_name` is not used here):
1. **case-fold to canonical** — 58 distinct label strings appear against `taxonomy_v1`'s 50 level-2
   names; the 8 extras are pure case drift. **Any unmapped label raises**, it is never counted.
2. **de-duplicate `(paragraph, topic)`** — 722 duplicate pairs survive normalization; 52,855 raw
   assignments are **52,133 distinct** (mean **1.439** topics per paragraph). Only post-dedup
   figures are ever quoted.

All paragraph-level joins use the real `(doc_name, para_idx)` key with `validate="one_to_one"` **and**
an assertion that the merged length equals each input's length (an inner join silently drops
non-matching rows; `taxonomy.py::_require_full_merge`).

---

## 4. Estimator — cluster rarefaction

### 4.1 Eligibility (design, fixed before any draw)
A **qualifying speech** for an arm is a speech that (a) belongs to the arm's genre set and (b) has at
least `B` paragraphs. A president is **eligible** in window *w* if they have at least `S` qualifying
speeches whose year lies in *w*.

| Arm | S (speeches) | B (paragraphs/speech) | m = S·B |
|---|---|---|---|
| `corex_all`, `llm_all` | **8** | **6** | 48 |
| `corex_sotu` | **4** | **12** | 48 |

The SOTU arm trades speeches for depth because the corpus holds only 218 annual messages; m is held
at 48 in both so that the *paragraph* budget is identical and only the *cluster* budget differs.

### 4.2 The draw
For each window *w*, each eligible president *p*, and each of **D = 20** independent draws:
1. draw `S` qualifying in-window speeches of *p* **with replacement**;
2. from each drawn speech, draw `B` of its paragraphs **with replacement**;
3. form `c_{p,w,d}` by the equation in §3 over those `m = 48` paragraphs.

**Why with replacement at both stages, and why this is the whole point.** Without-replacement
sampling carries a finite-population correction `(1 − S/n)` whose size depends on the pool `n` — and
`n` is exactly the quantity that drifts from ~10 to ~38 across the corpus. That would install a
*second*, opposite-signed, era-varying noise floor in place of the one being removed. With
replacement, the sampling variance of `c_{p,w,d}` is `σ²_between-speech / S` regardless of pool size,
so the estimator's noise is design-invariant by construction. Rarefying **speeches**, not
paragraphs, is the fix that the review validated (clustered-null rho **−0.042** vs **−0.605**).

### 4.3 Dispersion
For window *w*, draw *d*, with `E_w ≥ 2` eligible presidents, dispersion is the mean over all
unordered pairs of the **Jensen–Shannon divergence in bits**:

    JSD(P,Q) = H((P+Q)/2) − ½H(P) − ½H(Q),   H = Shannon entropy, base 2,   JSD ∈ [0,1]

    disp_w = (1/D) Σ_d  mean_{p<q}  JSD(c_{p,w,d}, c_{q,w,d})

### 4.4 Trust gate (`ci_status`, the `combat.py` pattern)
Published on **every window row**, so no consumer can plot an estimate blind:

| `E_w` | `ci_status` | effect |
|---|---|---|
| 0–1 | `no_data` | dispersion is NaN |
| 2 | `suppressed_n_floor` | a single pair is not a dispersion; excluded from every trend |
| 3–4 | `low_cluster_caution` | published and used, flagged |
| ≥ 5 | `ok` | published and used |

Windows with `no_data` or `suppressed_n_floor` are **excluded from every trend statistic**. The
exclusion set is a property of the design, not of the data values, so it is **identical inside every
permutation** — the null and the observed statistic are computed on exactly the same windows.

*Pre-declared coverage (a design fact, measured before any dispersion was computed):* `corex_all` is
eligible in **105/105** windows with **min 3, median 5, max 7** eligible presidents; `corex_sotu` in
**101/105** (the four uncovered windows start 1931, 1933, 1935, 1937), min 1, median 5, max 6.

### 4.5 Trend statistic
Spearman rho between window centre year and `disp_w` over the retained windows.

> **`scipy.stats.spearmanr`'s p-value is BANNED from this analysis** — from H1, H2, the jackknife,
> the rival test, every control curve and every sensitivity. Measured type-I error on these
> overlapping windows is **58%**. scipy is used only to obtain the rho *statistic*; the p-value is
> discarded at the call site, in code, by a wrapper that returns a float and never the tuple. Every
> p-value reported anywhere in this task comes from §5.

---

## 5. Inference — full-pipeline president permutation

**R = 2,000** permutations. Let `Pi` be the set of presidents who are eligible in at least one window
of the arm (donors are drawn from `Pi` so that every donor has at least `S` qualifying speeches).
Draw a uniform random bijection `π: Pi → Pi`. Then re-run the **entire pipeline**: the design —
which presidents are eligible in which window, and the `ci_status` exclusions — is held exactly
fixed, and president *p*'s slot is filled with content drawn from donor **π(p)**'s qualifying speech
pool by the identical §4.2 draw. Recompute every window's dispersion, the derived series of §6, and
every statistic; store the resulting rho.

This absorbs the nuisances a formula ignores: window overlap (93% shared content, ~8 independent
blocks in 240 years), the drifting number of eligible presidents, the drifting speech supply, and the
Monte-Carlo noise of the rarefaction itself. It is a bijection rather than an i.i.d. relabelling so
that a president moves as a unit and the *persistence of the same presidents across adjacent windows*
— the dominant source of autocorrelation in the curve — is preserved.

**p-values.** One-sided, in the hypothesized direction:
`p = (1 + #{r : rho_r ≤ rho_obs}) / (R + 1)`. Two-sided `p` is also reported. **α = 0.05, one-sided.**
The permutation-calibrated critical value (the 5th percentile of the null rho distribution) is
published for every statistic, so a reader can score the table without re-running anything.

**Power (repeated verbatim from the plan, and it is not optional).**
*At the permutation-calibrated critical value, this design has **68% power** against the effect size
it is hunting — not the ~96% the invalid p-values implied. A null result is therefore **weakly
informative, not a refutation**.*

---

## 6. The two things that must be subtracted before "convergence" may be said

### 6.1 Per-window speech-block permutation floor (its own published panel)
The floor answers: *how much dispersion would this window show if its presidents had no distinct
agendas at all?* Within window *w*, pool the qualifying speeches of all `E_w` eligible presidents and
re-deal them to the `E_w` slots **as whole speech blocks**, preserving each slot's speech count; then
run the identical §4.2 draw and §4.3 dispersion. **F = 50** block re-deals per window.

The floor is published **per window**, as a column beside the curve — never a scalar, and never
assumed time-constant, because on real multi-label data it **drifts 29–53%**. The derived series

    floor_corrected_w = disp_w − floor_w

carries its own rho and its own permutation p (the floor is recomputed inside every permutation, with
the same F, so null and observed are computed identically).

### 6.2 Entropy-matched rival test — "more alike" vs "each broader"
For each window and draw, take the same `E_w` rarefied compositions and independently permute the
**bin order** of each one. Bin-shuffling preserves each president's entropy *exactly* (it is a
permutation of the same probability values) while destroying any agreement between presidents. Its
mean pairwise JSD is therefore the dispersion those breadths alone predict, absent a shared agenda:

    excess_ratio_w = disp_w / entropy_matched_w        (∈ (0, 1] in practice)

`disp_w` can fall purely because presidents got broader. `excess_ratio_w` can only fall if presidents
became **more alike than their own breadth accounts for**. Mean within-president composition entropy
is published per window alongside both series so the mechanism is visible, not asserted.

`excess_ratio` carries its own rho and its own permutation p, recomputed inside every permutation.

---

## 7. Secondary and sensitivity analyses (all declared here, all scored by §5)

**7.1 Jackknife by president.** For every president in `Pi`, delete them from the design entirely and
recompute the `rolling` dispersion rho for all three arms — **including leave-Trump-out**. Deleting a
president can push a window below the §4.4 floor; those windows drop and the count is published per
row. Each jackknife rho is scored against the **full-design** permutation null for its arm (declared
approximation: 45 × 2,000 re-permutations is not affordable, and the null's spread is a property of
the design, which the jackknife barely perturbs). Pre-declared read: H1 is **fragile** if deleting any
single president moves the primary arm's rho across the critical value.

**7.2 H2b — agenda typicality (the only scored half of H2).** For each president *p* in `Pi`:

    d_p = mean over p's eligible windows, over draws, of  mean_q JSD(c_p, c_q)   [q ≠ p, eligible in w]
    d̂_p = d_p − mean over the same windows of floor_w      ← the bias correction

The bias correction is what makes `d̂_p` comparable across presidents who sit in windows of different
size and different underlying topic entropy; an uncorrected `d_p` would rank presidents partly by
which era they happened to occupy. **Score:** Trump's percentile of `d̂` among the ranked presidents
(1 = most agenda-typical). Under the §5 permutation null a president's percentile is uniform, so the
one-sided p for H2b **is** that percentile. **H2b is supported iff Trump's percentile ≤ 0.50 and
`d̂_Trump` is below the mean of the others**; the effect size reported is
`z = (d̂_Trump − mean d̂_others) / sd(d̂_others)`. Power here is poor by construction — Trump is
eligible in 5 heavily-overlapping windows — and the findings note must say so wherever it says
anything else about H2b.

**7.3 The paradox table.** `paradox.parquet` carries `d̂_p` (the §7.2 test) beside the **H2a anchor**
column (mean cosine similarity to the other 44 presidents, read from `data/president_embeddings.parquet`,
not recomputed and not scored). The anchor column is labelled as an anchor in the table itself.

**7.4 Dating procedure (H1's Trump clause).** The primary rho is additionally computed on the subset
of rolling windows whose **end year ≤ 2014** (99 windows), scored by the same permutation null on the
same subset. Any claim about the trend's existence *independent of the Trump era* quotes this number
and only this number.

**7.5 Non-overlapping windows.** rho over the 8 non-overlapping 30-year windows of §2.1.
**Power is low and is stated in advance**; this is reported, never decisive.

**7.6 President-pair regression.** For each unordered pair of presidents ever co-eligible, the mean
rarefied JSD over their shared windows and draws, against the mean centre year of those shared
windows; Spearman across pairs, scored by the same permutation null. **Power ≈ 53%, stated in
advance**; reported, never decisive. Computed on the `rolling` window set only.

**7.7 Declared scope rules (what is deliberately NOT computed).** `ends_2014`, `nonoverlapping` and
the pair regression are computed for **all three arms**. The jackknife is computed for all three arms.
`paradox.parquet` is computed for the **primary arm only** — `d̂` is not commensurable across arms
with different bin counts (§2.2), so a three-arm paradox table would invite exactly the cross-arm
level comparison this document forbids. No other cell is omitted.

---

## 8. Decision table — evaluated in this order, each cell with its literal headline

Write `sig_neg(X)` for "rho_X < 0 **and** its one-sided permutation p ≤ 0.05". A = primary
(`corex_all`), C = co-primary (`corex_sotu`), both on the `rolling`/all-windows treatment. The rules
are evaluated **in order** and the first one that fires decides:

| # | Rule | Cell | Pre-committed headline sentence (verbatim) |
|---|---|---|---|
| 1 | `sig_neg(A.dispersion)` XOR `sig_neg(C.dispersion)` | **ARTIFACT** | "The two pre-registered arms disagree: one shows a significant decline in between-president agenda dispersion and the other does not, so this corpus cannot tell us whether presidential agendas converged." |
| 2 | neither `sig_neg(A.dispersion)` nor `sig_neg(C.dispersion)` | **NO CONVERGENCE** | "Across 240 years we find no significant decline in between-president agenda dispersion; at 68% power this is weak evidence against convergence, not a refutation of it." |
| 3 | both dispersions sig_neg, but NOT (`sig_neg(A.floor_corrected)` and `sig_neg(C.floor_corrected)`) | **ARTIFACT** | "Between-president agenda dispersion does decline, but the decline does not survive subtraction of the per-window noise floor, so it is a property of the measurement rather than of the presidencies." |
| 4 | both dispersions and both floor-corrected sig_neg, and both `sig_neg(*.excess_ratio)` | **CONVERGENCE** | "Presidential agendas have converged: between-president dispersion of issue composition declines over 1789–2026 in both pre-registered arms, survives the per-window noise floor, and survives an entropy-matched null — presidents are more alike than their own broadening agendas can explain." |
| 5 | otherwise (both dispersions and both floor-corrected sig_neg, excess_ratio not) | **BROADENING** | "Between-president agenda dispersion declines, but an entropy-matched null accounts for it: presidents did not converge on a shared agenda so much as each of them started talking about everything." |

Nothing else may be reported as the headline. Sub-results (the LLM arm, the dating subset, the
non-overlapping windows, the pair regression, H2b, the jackknife) qualify the headline in the body
and never replace it.

---

## 9. `_selftest` — the gate that must pass before the real data is touched

`convergence._selftest()` runs on **synthetic** data only and must demonstrate all three legs. If any
leg fails, the run **STOPS** and nothing is written. Running an estimator that cannot pass its own
null is exactly how the original design got here.

| Leg | Construction | Pass criterion |
|---|---|---|
| **(a) flat on the CLUSTERED null** | Synthetic corpus with the real design's *shape*: a time-invariant common topic distribution, speech-level Dirichlet tilts (within-speech clustering), and a per-president in-window speech supply that grows over time — i.e. **zero real convergence**. | Cluster-rarefied \|rho\| ≤ **0.25** (validated design: −0.042). **And the contrast must hold**: the paragraph-only rarefied estimator on the *same* data must return rho ≤ **−0.30** (validated: −0.605). Passing leg (a) without the contrast would not prove the fix, only the absence of a signal. |
| **(b) detects injected convergence** | Same generator, plus a president-specific tilt whose magnitude shrinks linearly with time. | Cluster-rarefied rho ≤ **−0.50** (validated: −0.854). |
| **(c) permutation-null size ≈ 5%** | **40** independent clustered-null replicate corpora; for each, the §5 permutation p at **150** permutations. | Empirical size at nominal α = 0.05 ≤ **0.152** (the 3σ upper bound at 40 replicates under a true 5%; validated: 4.8%). The measured value is reported whether it passes or not. |

---

## 10. Artifact contract

`data/convergence/` follows the `combat.py` / `eras.py` derived-layer contract: **derived,
deterministic, $0, zero API calls**, regenerable with

    PYTHONPATH=src arch -x86_64 .venv/bin/python -m presidential_profiles.convergence

**Byte-identical on rerun on any date.** No wall-clock stamp anywhere: a dirty
`git status data/convergence/` must mean the numbers moved, not that the clock did. Provenance
identity is `corpus_fingerprint`; *when* it ran is git's job. Every RNG is explicitly seeded and every
seed is recorded in `convergence_meta.json`. `convergence.py` never imports `anthropic` and never
constructs a client.

Outputs: `dispersion_curves.parquet` (one row per arm × window set × window, carrying `dispersion`,
`entropy_matched`, `excess_ratio`, `floor`, `floor_corrected`, `mean_entropy`, `n_eligible`,
`ci_status`), `permutation_null.parquet` (one row per arm × treatment × statistic, carrying observed
rho, one- and two-sided p, null mean/sd and the 5th-percentile critical value),
`jackknife.parquet`, `paradox.parquet`, `convergence_meta.json`.

## 11. Pre-declared constants

| Constant | Value |
|---|---|
| Window length / step | 30 years / 2 years |
| Rolling windows | 105 (1789–2026) |
| Non-overlapping windows | 8 |
| S × B (`corex_all`, `llm_all`) | 8 × 6 = 48 |
| S × B (`corex_sotu`) | 4 × 12 = 48 |
| Rarefaction draws per window, D | 20 |
| Speech-block floor re-deals per window, F | 50 |
| Permutations, R | 2,000 |
| α (one-sided) | 0.05 |
| Minimum eligible presidents for a usable window | 3 |
| Selftest replicates / permutations | 40 / 150 |
| Master seed | 20260721 |

## 12. Known limitations, cited rather than re-litigated

- `taxonomy_v1` is frozen and has documented gaps: no standalone Education topic, no modern monetary
  policy, no Prohibition, no standalone Terrorism; 7 of 50 level-2 topics have no legacy parent
  (`crosswalk-v1-one-directional`). These bound what `llm_all` can say.
- The LLM layer's union projection (`union-projection-inflates-wide-crosswalk-issues`) is avoided
  here by using native level-2 labels, but the underlying annotations are unchanged.
- `indices.py:MARKERS` is not used anywhere in this analysis: those regexes are era-biased
  (`opponents` is structurally zero before ~1990).
- Intervals in this analysis are permutation-calibrated *trend* tests; they contain no
  annotator-disagreement component. `agreement_v1.parquet` bands are not propagated into the
  dispersion curves, and the outputs say so.
- 68% power. Again: a null result here is weakly informative, not a refutation.
