# Breadth, depth and register of the formal presidential record — v1

**Status: EXPLORATORY. Nothing in this note is a confirmatory test.**

Every quantity here had already been observed before it was measured properly. The
trends were first computed as a side effect of the adversarial review of the
convergence design (2026-07-13), which is exactly why they cannot now be
pre-registered as hypotheses — writing down a prediction for a number you have
already seen is theater, not evidence. What this pass can honestly do, and does,
is re-measure everything with corrected estimators on a second independent
labeler, and test each trend against the confounds that could fake it. A
confirmatory design is proposed at the end; it has not been run.

Produced by `src/presidential_profiles/register.py`. Full series with confidence
intervals: `data/register/trends.parquet` (16,243 rows). Bootstrap seed 20260721,
2,000 replicates, clustered on speeches. No LLM calls were made; every input was
already on disk.

---

## The claim under test

> Presidents now touch more subjects, say less about each, and spend a growing
> share of their most formal speech not on policy at all.

Three sub-claims. **One survives with a large caveat, two die.** The corpus does
contain a very strong trend — it is simply not this one.

---

## Summary of verdicts

| # | Sub-claim | Verdict |
|---|---|---|
| 1 | Presidents touch **more subjects** | **Survives** every taxonomy and genre control — but it is a **pre-1860** change. Since the Civil War era ‖, breadth is flat (legacy 15 on all three arms; the 50 LLM topics on two of three — the exception is +1.20, p = 0.042 inside annual messages) or significantly **falling on two of three arms** (17 LLM domains; the raw arm is −0.62, p = 0.061). The word "now" is wrong. **But the 2010 era is nonetheless the highest or second-highest of the nine on 8 of 9 arms** — a block contrast and an endpoint rank answer different questions, and both are given in sub-claim 1. |
| 2 | Presidents **say less about each** | **Dead.** The per-paragraph decline is mostly paragraph-length shrinkage. Per 1,000 words, issue density **rises** significantly under all three genre treatments and both taxonomies. Sign reverses. Post-1860 ‖ the rise holds on **five of six** arms § ; the raw legacy-15 arm is flat (−0.24, p = 0.39). Per *paragraph*, measured from 1860 ‖, the two labelers split rather than agree — legacy 15 falls on raw (−0.260) and level-2 rises under both controls (+0.141, +0.120); see sub-claim 2. |
| 3 | **More speech off-policy** | **Split.** Dies on the legacy labeler under genre control on the corpus-long contrast (sign flips negative) — **but measured from 1860 ‖ that labeler agrees**, positive on all three arms and at the bootstrap floor on raw (+0.105, p = 0.0005). Survives on the LLM labeler under all three treatments §, but the effect is **+3.1 points (SOTU-only) / +3.0 (genre-standardized)** — +5.0 raw — and the modern level is *below* the founding era's. |
| — | (bonus) More **values-driven** | **Dead as stated** across the corpus; **true** if measured from 1860 ‖. And it is not values displacing policy — both rise as neutral description collapses (measured from the 19th-century plateau, not from the founding era; see the bonus section). |
| ★ | **The register changed** | **Survives everything.** The strongest, most robust finding in the set, and it was not the headline. |

‖ Every contrast against the 1860/1890/1920 block (`delta_modern_minus_postbellum`,
printed here as "modern − postbellum", "post-1860", "measured from 1860",
"measured from the 1860/1890/1920 block", or as a block mean the postbellum eras
appear in) is the **post hoc** statistic declared below. It is marked ‖ at every
one of its appearances in this note outside the declaration itself —
`grep -n 'postbellum\|post-1860\|measured from 1860\|1860/1890/1920'` and check
that every hit either carries the marker or is part of one of the two declaration
passages that define the statistic (this footnote and the Pre-declaration
section).

§ Arm counts are not counts of independent controls: `sotu_only` and
`genre_standardized` share **74% of paragraph mass by construction** (93% in the
1770 and 2010 eras). See "Sample sizes" below.

---

## Pre-declaration

The table below was written down **before** `build_trends` was run for the first
time. All arms declared in it are reported in this note, including the ones that
came out against the headline. No arm was added, dropped or re-binned to protect a
result. (This is a weak form of pre-registration — the analyst declared it to
himself, unwitnessed, in a session where the legacy-15 figures were already known
from the 2026-07-13 review. It constrains the LLM-taxonomy and genre arms, which
were genuinely unmeasured; it constrains the legacy-15 arm not at all.)

| Measure | Headline predicts | Rival explanation predicts |
|---|---|---|
| `effective_topics`, legacy 15 | rise | (this *is* the original observation) |
| `effective_topics`, LLM level-2 / level-1 | rise | taxonomy-fit artifact → flat or fall |
| `labels_per_paragraph` | fall | cross-labeler disagreement → flat/rise |
| `labels_per_1k_words` | fall | paragraph-length shrinkage → flat |
| `non_policy_share` | rise | cross-labeler disagreement → flat/fall |
| `proposal_vs_values` | fall | flat/rise kills it |
| `mechanism`, `fk_grade` | fall | — |
| `religiosity`, `hype`, `us_them` | rise | — |

One statistic, `delta_modern_minus_postbellum`, was added **post hoc** after the
era series were computed. It carries the marker **‖** at every appearance in this
note outside this section and the footnote above — the grep recipe in that
footnote is the check, and it covers the summary table, every contrast table, the
multiplicity and genre-disagreement sections and the confirmatory design — and it
is labelled post hoc in the module
source, on the `delta_modern_minus_postbellum` key of `register.py`'s
`BLOCK_CONTRASTS` (the other key in that dict, `delta_modern_minus_early`, is
pre-declared, so the comment names which one it means).

It was added because it *weakens* the headline, not because it rescues it — see
sub-claim 1. **It does not always weaken it, and every place it does not is now
reported:** on the legacy labeler in sub-claim 3 it is the arm that runs the
headline's way (+0.105 raw, at the bootstrap floor); on `labels_per_paragraph` it
gives the largest decline in that measure anywhere in the note (−0.260, against
the −0.145 an earlier draft quoted); and on the whole register block all 33 cells
are at the floor, which makes the star finding *stronger* rather than weaker. A
statistic introduced to be adversarial has to be reported when it stops being
adversarial, or it is just a differently-shaped selection rule.

An earlier draft of this paragraph asserted that flagging, and the flagging did
not exist: two labelled mentions against six unlabelled ones, the summary verdict
table among them. That is recorded rather than quietly fixed, because a false
claim about one's own labelling discipline is worse than no claim.

One measure, `neither_share`, was added **after the write-up**, for a different
reason: the "neutral description collapsed" figure in the bonus sub-claim was the
only number in this note that a reader could not check against the parquet the
note advertises. It is the arithmetic residual of the two components of the
pre-declared `proposal_vs_values` ratio — no new arm, no new contrast, no change
to any other value in the table.

---

## Sample sizes, published next to everything

The thin tail is real and no CI should be read as if it were not. The 1770 era is
28 speeches, 355 paragraphs, 11 annual messages — **and two presidents.**

| era | speeches | **presidents** | paragraphs | words | SOTU speeches | SOTU paragraphs | largest president's share of era paragraphs |
|---|---|---|---|---|---|---|---|
| 1770 | 28 | **2** | 355 | 44,493 | 11 | 185 | **71.3%** (Washington) |
| 1800 | 68 | 6 | 1,381 | 175,727 | 30 | 1,018 | 29.5% (Monroe) |
| 1830 | 113 | 10 | 4,492 | 610,490 | 30 | 2,600 | 23.4% (Jackson) |
| 1860 | 128 | 9 | 3,992 | 497,246 | 30 | 2,581 | 20.4% (Grant) |
| 1890 | 112 | 7 | 5,064 | 635,951 | 30 | 3,707 | 30.2% (T. Roosevelt) |
| 1920 | 122 | 6 | 2,828 | 351,606 | 16 | 804 | 38.3% (F. D. Roosevelt) |
| 1950 | 191 | 8 | 6,635 | 678,053 | 26 | 1,560 | 40.9% (L. B. Johnson) |
| 1980 | 174 | 6 | 5,870 | 637,581 | 28 | 1,441 | 33.6% (Reagan) |
| 2010 | 121 | **3** | 5,612 | 548,119 | 17 | 1,295 | **50.8%** (Trump) |

**The president counts are the number that should worry a reader most, and they
were missing from the first draft of this note.** Every claim here is phrased as
a claim about *presidents* ("presidents touch more subjects"), but the estimator's
resampling unit is the speech and the reporting unit is the era. At the two
endpoints those come close to being claims about individuals: the 1770 era is
Washington and Adams, and 71.3% of its paragraphs are Washington's; the 2010 era
is Obama, Trump and Biden, and 50.8% of its paragraphs are Trump's. A
speech-clustered interval treats 28 Washington-and-Adams speeches as 28 draws;
as evidence about *presidents* they are closer to two. Nothing in this note
prices that in, and no interval here should be read as if it did. The endpoint
quantities are the most exposed — `delta_last_minus_first`, and the 1770 and
2010 columns of every level table. The block contrasts are much less exposed:
`delta_modern_minus_early` pools 14 presidents against 16 with **no overlap**,
and `delta_modern_minus_postbellum` ‖ pools 14 against 18 sharing only Truman.
This is one reason the block contrasts, and not the endpoints, carry the verdicts
in this note.

**The three genre treatments are not three independent controls.** The reference
mix is 74.4% annual messages by paragraph share, so `genre_standardized` is
three-quarters the same paragraphs as `sotu_only` in every era — and in the 1770
and 2010 eras, where thin cells force the mix down to {SOTU, inaugural},
**93.1%** the same. Wherever this note counts arms ("five of six", "all three
treatments", "two of three"), read it as roughly **two** independent genre
controls, not three: `raw`, and a genre-held-fixed reading that `sotu_only` and
`genre_standardized` mostly share. The cases that genuinely carry three
independent votes are the ones where `raw` disagrees with both controls, and
those are listed in "Where the three genre treatments disagree".

---

## Method notes that change the numbers

**Bootstrap unit is the speech.** Paragraphs inside one speech are not independent
draws, so a paragraph-level bootstrap would claim 36,229 independent observations
where there are 1,057.

The figure previously quoted here — a within-speech ICC of 0.119 — was **inherited
from the 2026-07-13 adversarial panel**, which measured it on CorEx labels, and no
code in this repo computed it. It is now computed by shipped code
(`register.within_speech_icc`, a one-way random-effects ICC(1) with speeches as
groups and the unequal-group-size correction `k0`, run on the same paragraph frame
the trends are built from), and the inherited number turns out to be an
*understatement*:

| paragraph quantity | ICC |
|---|---|
| `n_legacy` — CorEx labels per paragraph | 0.116 |
| `n_llm` — `taxonomy_v1` level-2 topics per paragraph | 0.135 |
| `legacy_zero` — paragraph carries no anchored issue | 0.103 |
| `llm_all_non_policy` — all topics non-policy-parented | 0.155 |
| `llm_zero` — paragraph got no topic at all | 0.294 |

Run over the 15 legacy issues one at a time — the panel's own unit of analysis —
the same function gives a mean of 0.110 and a maximum of 0.171 (`Money &
banking`), against the panel's "mean 0.059, up to 0.119". The disagreement is not
adjudicated here (different estimator, possibly a different label frame), but it
runs in the conservative direction: the recomputed per-issue mean is nearly double
the panel's, the recomputed per-issue maximum is above the panel's, and every
LLM-label quantity — the arm on which this note's LLM findings rest — exceeds
0.119. The one value that sits below it, `legacy_zero` at 0.103, is still an order
of magnitude away from the zero that a paragraph-level bootstrap assumes. Nothing
downstream depends on the number: the bootstrap unit was fixed by design before
any of these were computed, and no published figure changes if the ICC is 0.06 or
0.30.

**Intervals and p-values.** All intervals are speech-clustered percentile
intervals; p-values are two-sided bootstrap-percentile p-values
(`2 · min(P(θ*≤0), P(θ*≥0))`, floored at 1/B = 0.0005), which is an interval
inversion, **not** an exact test. A reported `p = 0.0005` means "no replicate
crossed zero", not "p = 5 × 10⁻⁴".

**Multiplicity: 444 tests, uncorrected, and what that costs.** `trends.parquet`
contains **444 hypothesis tests** — 37 measure × taxonomy columns × 3 genre
treatments × 4 trend statistics — and not one p-value in this note is corrected
for that. 344 of the 444 are below 0.05. If every test were null and independent,
about 22 would be. So the exposure is real but it is not the whole table: the
distribution is strongly bimodal, with **277 tests at the resolution floor**
(p ≤ 1/B, "no replicate crossed zero"), **50 in the [0.004, 0.05) band** where a
correction bites, and the remaining **17 in (0.0005, 0.004)** — which a
family-wise threshold of 0.05/444 = 1.13 × 10⁻⁴ excludes as well.

This note is exploratory and reports every arm it computed, including the arms
that kill its claims, so the response is disclosure rather than a family-wise
correction: with a *predicted* direction stated on fewer than a dozen of the 444,
a Holm correction over the whole table would mostly penalize having published the
nulls. The disclosure has to be specific, so here it is.

*A hard limit first.* A bootstrap of B replicates cannot report a p below 1/B.
At B = 2,000 the floor is 5 × 10⁻⁴, while the Holm/Bonferroni first step over
444 tests is 0.05/444 = **1.13 × 10⁻⁴**. **No test in this table can be shown to
clear a family-wise threshold at all**, however large its effect — the estimator
does not have the resolution. Settling it would need B ≳ 9,000, which is a
cheap rerun and was not done. Every statement below about what "survives" is
therefore a statement about which tests are *not excluded*, not which are proven.

**Load-bearing claims that a correction would not exclude** (p at the floor, so
the only thing standing between them and a corrected threshold is B; the effect
sizes are also large, and the direction is confirmed on the second labeler or the
second genre control):

| claim | arm | p |
|---|---|---|
| the register block — `mechanism`, `hedges`, `fk_grade`, `words_per_sentence`, `opponents`, `boosters`, `hype`, `doom`, `future`, `nostalgia`, `superlatives` | all 11, on **all three** treatments, on **both** block contrasts (modern − early *and* post-1860 ‖) | 0.0005 |
| `labels_per_1k_words` rises (sub-claim 2's reversal), modern − early | 5 of 6 arms | 0.0005 (raw legacy 15 is +1.03, p = 0.006 — marginal) |
| `mean_paragraph_words` falls (the mechanism), modern − early *and* post-1860 ‖ | all three, both contrasts | 0.0005 |
| `neither_share` falls (the bonus finding), modern − early | all three | 0.0005 |
| `effective_topics` rises across the corpus (Spearman) | legacy 15 and level-2, raw + SOTU + std | 0.0005 |
| `values_share` rises post-1860 ‖ (the bonus rebuttal's stronger half) | all three | 0.0005 |
| `non_policy_share` (**legacy 15**) rises post-1860 ‖ — the largest of sub-claim 3's pro-headline effects, but on one arm of three | raw, +0.105 | 0.0005 |
| `non_policy_share` (**LLM level-2**) rises post-1860 ‖ — sub-claim 3's block contrast, and the most broadly based pro-headline result here | all three: raw +0.074, sotu_only +0.053, genre-std +0.063 | 0.0005 |
| `labels_per_paragraph` **falls** post-1860 ‖ on legacy 15 (against sub-claim 2's reversal) | raw, −0.260 | 0.0005 |
| `labels_per_paragraph` **rises** post-1860 ‖ on LLM level-2 (for it) | sotu_only +0.141 / genre-std +0.120 | 0.0005 |

The last four rows are listed together on purpose. Two of them argue against this
note's verdicts and two argue for the headline it spends most of its length
dismantling; all four are at the floor, and none is discounted here for that.

**Load-bearing claims that become marginal under any correction** — every one of
these sits in the [0.001, 0.042] band and would not clear even a modest
family-wise threshold:

| claim | arm | p |
|---|---|---|
| breadth **falls** post-1860 ‖ on LLM level-1 | sotu_only −0.95 / genre-std −0.90 | **0.004 / 0.009** |
| sub-claim 3's **`modern − early`** contrast: `non_policy_share` (LLM) rises (its post-1860 ‖ contrast is at the floor on all three arms — table above) | raw / sotu / std | **0.009 / 0.014 / 0.009** |
| `non_policy_share` (LLM) Spearman rises | sotu_only | **0.040** |
| the one significant post-1860 ‖ breadth rise | level-2, sotu_only, +1.20 | **0.042** |
| `religiosity` and `nrc_fear` rise *only* under genre control | sotu_only | **0.002 / 0.009** |
| `nrc_hope` falls under genre control | genre-std | **0.021** |
| `effective_topics` level-1 Spearman | sotu_only / genre-std | **0.008 / 0.037** |
| `us_them` rises raw on **modern − early** (already reported as a genre artifact) | raw | **0.003** |
| `labels_per_paragraph` falls raw on **modern − early** (sub-claim 2's original evidence) | raw legacy 15 | **0.001** |
| `nrc_fear` **falls** raw post-1860 ‖ (the opposite sign to its own genre-controlled `modern − early` contrast) | raw | **0.001** |

The `labels_per_paragraph` and `non_policy_share` (LLM) rows are the
modern − early contrast **only**. An earlier draft of this section filed those
measures here and stopped, which understated both: the *same* measure, taxonomy
and arm has a post-1860 ‖ contrast of −0.260 at the resolution floor for
`labels_per_paragraph`, and `non_policy_share` (LLM) is at the floor on **all
three** arms of that contrast. Both belong in the table above and are now in it.
Marginal on one contrast is not marginal on the measure.

Read together: **the register finding and the density reversal are the two a
correction would not exclude; the post-1860 ‖ level-1 decline is not; and
sub-claim 3 falls on both sides — its `modern − early` contrast would be
excluded, its post-1860 ‖ contrast would not.** That cuts both ways, and
deliberately so — the marginal list contains the note's own strongest
self-criticism (the level-1 decline) as well as one of the two contrasts carrying
the last surviving piece of the original headline. Neither is dropped, and
neither is promoted.

Finally, the *null* verdicts in this note — sub-claim 2's per-paragraph decline
dying under genre control, breadth being flat post-1860 ‖ on the legacy 15,
`proposal_vs_values` being flat, `us_them` dying under control — are unaffected
by multiplicity in the direction that matters. Correcting can only make a
failure-to-reject more likely. Every conclusion this note reaches *against* the
headline is therefore at least as safe after correction as before it.

**Half the parquet's intervals are degenerate, and the schema does not flag it.**
8,013 of the 16,243 rows (49.3%) have `ci_low == ci_high`. Almost all of them are
year-level rows resting on a single speech: **every one of the 6,808
`unit=year, genre_treatment=sotu_only` rows has `n_speeches == 1`** (184 published
years × 37 measure-taxonomy columns), and a further 999 `unit=year, raw` rows do
too (27 years). A bootstrap clustered on speeches has exactly one cluster to draw
from there, so
every replicate is the same speech and the "interval" is the point estimate
printed three times. It is not a narrow interval — it is the *absence* of an
interval. Anyone plotting the year series straight out of the advertised parquet
would see apparently-zero uncertainty at every SOTU year, which is the opposite of
the truth. **Read `n_speeches` before reading any interval**; the era rows, which
is where every claim in this note is made, do not have this problem.

The remaining **206** rows break down as follows, and an earlier draft of this
paragraph waved all of them through as "markers at zero in every early-era draw",
which is wrong on both halves of the phrase:

- **146 are `zero_topic_share`** (137 year rows, 7 era rows, 2 trend rows), every
  one of them exactly 0.0 — genuinely constant across replicates, and correctly
  degenerate. But they are not an early-era phenomenon: the year rows occur in all
  nine era bands, and **37 of the 144 that carry a period sit in the 1920 and 2010
  bands** (the LLM assigns a topic to essentially every paragraph in the thick
  modern SOTU cells, so the share pins at zero there too).
- **58 more are marker rates at exactly 0.0** — `opponents` in the 1770 era and
  through most of the 19th century, plus scattered `doom`, `hype`, `boosters`,
  `future`, `nostalgia`, `superlatives`, `religiosity`, `neither_share` and
  `non_policy_share` year cells. These *are* the constant-at-zero case the earlier
  wording described, and they are benign.
- **2 are not benign, and were previously mislabelled as such.** They are the only
  two of the 206 with a **non-zero** value, and both rest on **two speeches**:
  year 1891, `effective_topics_plugin`, legacy 15, raw (10.421, `n_speeches` = 2,
  1,527 valid replicates) and year 1904, `proposal_vs_values`, raw (0.546,
  `n_speeches` = 2, 1,514 valid replicates). With two clusters a speech-clustered
  resample draws from three distinct multisets, so the percentile interval can
  collapse onto a single value that is *not* a constant of the data — the same
  single-cluster hazard, one step less obvious. **Read `n_speeches`**: 1 is not
  the only value of it that should stop a reader.

**Three rows have `value` outside their interval.** Year 1984, `llm_level2`,
`raw`: `effective_topics` (21.066 against a `ci_high` of 21.007) and its
`effective_topics_plugin` and `normalized_entropy` twins. This is expected rather
than a bug: a bootstrap resample of that cell's 9 speeches covers only ~65% of
them (`1 − (1 − 1/n)^n`, tending to 63%), so replicate topic richness sits
systematically *below* the full-sample plug-in and the percentile interval can sit
entirely under the point estimate.
It is the same small-N entropy bias the Miller-Madow term addresses, showing up in
the resampling distribution instead of the estimate. No other row in the table has
`value` outside its interval on either side.

**Label case-variants were repaired, not dropped.** 111 of 52,855 label
assignments (0.21%) are case-variants of real `taxonomy_v1` names. All resolve
case-insensitively; `register.py` raises if any does not. 84 of the 111 land on
one modern topic, so an exact-match join would have biased modern policy attention
*downward* — flattering sub-claim 3.

**Non-policy share is two different measures and is never merged into one series.**
The labelers disagree about what an unlabeled paragraph is: CorEx leaves 28.7% of
paragraphs with zero anchored issues, the LLM leaves 1.1% with zero topics. So on
the legacy taxonomy the measure is the zero-anchored-issue rate; on `taxonomy_v1`
it is the share of paragraphs whose topics are *all* non-policy-parented, with the
1.1% zero-label residue published separately as `zero_topic_share`.

**Genre standardization.** Reference mix = corpus-pooled **paragraph** share of
the four genres present in every era: annual message/SOTU 0.744, special message
to Congress 0.149, inaugural 0.055, veto or signing statement 0.052. Standardizing
to all ten genres is impossible without inventing counterfactual cells (press
conferences do not exist before the 20th century); a uniform mix would score a
signing statement as heavily as an annual message. Cells under 3 speeches are
dropped and weights renormalized — **which matters at both endpoints**: the 1770
and 2010 eras each lose special messages and veto statements, so their
standardized estimates rest on a renormalized {SOTU, inaugural} mix that is ~93%
SOTU. In those two eras `genre_standardized` is close to `sotu_only` by
construction, and its agreement with `sotu_only` there is not independent
evidence.

**Entropy bias.** Plug-in Shannon entropy is biased downward at small N, and the
early eras are the small ones — the bias points in the direction the headline
needs. `effective_topics` carries the Miller-Madow correction; the uncorrected
figure ships as `effective_topics_plugin`. Miller-Madow is only first-order, so a
rarefaction check is also reported below.

**Which of the 444 tests this note tabulates, and what happens to the rest.** The
note claims to report every arm it computed. That claim is only meaningful if it
is stated as a rule a reader can check, so here is the rule. For every measure the
note reaches a verdict on, **all three genre treatments** of the statistic
carrying that verdict are printed, none omitted. The statistics *not* printed
arm-by-arm are the ones no verdict rests on, and they are these four groups:

1. **`delta_last_minus_first`, wherever the verdict rests on a block contrast.**
   This note says at "Sample sizes" that endpoint contrasts are the most exposed
   statistic in the table and that the block contrasts carry the verdicts, so the
   endpoint columns are summarised rather than tabulated per arm — and summarised
   in the direction that does not flatter the note. For the eleven register
   markers all 33 endpoint cells are significant and 32 are at the floor (the
   exception is `mechanism` under `sotu_only`, p = 0.002); for the four
   non-surviving markers `nrc_hope` is at the floor on all three and `us_them` is
   significant on all three, both of which are stated where those markers are
   discussed, while `nrc_fear` and `religiosity` are non-significant on all three
   (p = 0.26 to 0.95) — so nothing significant is hiding in this group. They are
   omitted because they are exposed, not because they disagree. Where an endpoint
   contrast *does* disagree with a block contrast — `neither_share`,
   `non_policy_share`, `labels_per_paragraph`, and most sharply
   `effective_topics` — it is tabulated or stated. `effective_topics` on LLM
   level-1 is the **only** published measure in the 444 carrying two *significant*
   contrasts of opposite sign (its `effective_topics_plugin` and
   `normalized_entropy` twins inherit it, being the same series up to an
   order-preserving transform — see scope rule 3, where the two differ): on both genre-controlled arms the endpoint contrast is
   significantly **positive** (+2.11 sotu_only, p = 0.026; +1.97 genre-std, p =
   0.024) while the post-1860 ‖ contrast is significantly **negative** (−0.95, p =
   0.004; −0.90, p = 0.009) — and those are exactly the arms carrying sub-claim
   1's "significantly falling" verdict. That is the 1950/1980 trough seen from the
   other end, and it is a reason to trust the reconciliation in sub-claim 1 rather
   than a contradiction of it: 2010 is above 1770 and below the postbellum ‖ block
   at the same time. `labels_per_1k_words` is the other omission worth naming: 5
   of its 6 endpoint cells are significant (+2.54 to +4.12, p = 0.0005–0.008; the
   exception is raw legacy 15, +1.41, p = 0.189), all in the same direction as the
   block contrasts sub-claim 2 uses, so nothing there changes a verdict either.
2. **The register block's `modern − early` on `raw` and `genre_standardized`.**
   The table prints the `sotu_only` column and then states the property that
   matters, which is checked rather than asserted: the contrast is at the
   bootstrap floor on **all three arms for all eleven markers**, same sign
   throughout. The two unprinted columns are **not** interchangeable with the
   printed one, and an earlier draft said they were ("the same size to within the
   width of the printed intervals", which is false): **8 of the 22 sit outside the
   printed `sotu_only` interval**, and on `fk_grade` the raw arm (−11.46) misses
   the printed −8.99 [−9.51, −8.49] by 2.47 — 2.4× that interval's width, and the
   only cell where the gap exceeds the width at all. The checkable bound is that
   **no unprinted cell differs from its printed twin by more than 4.02** in the
   marker's own units (`hedges`, raw). The divergence runs both ways rather than
   flattering the printed column: the raw arm is *larger* than the printed figure
   on 4 of the 11 markers and smaller on 7. Nothing in this note's conclusions
   turns on the exact values, and every one of them is in the parquet.
3. **`effective_topics_plugin` and `normalized_entropy`, except where compared.**
   `normalized_entropy` is a monotone transform of `effective_topics`, so its era
   ranks are *identical* and its Spearman values **and p-values** equal the
   published ones on all nine arms, digit for digit. The plugin is not a monotone
   transform — the Miller-Madow term is data-dependent — but it happens to induce
   the same era ordering, so its nine Spearman point values match too; only the
   p-values differ, and slightly (0.028 vs 0.037, 0.005 vs 0.008, 0.0005 vs
   0.0010, all on level-1). Printing either column would be printing the same nine
   numbers a second and third time. Their block and endpoint contrasts differ
   (log scale for `normalized_entropy`, no bias correction for the plugin), and
   the plugin's are compared against the published series below because the
   entropy correction is a live objection. `normalized_entropy`'s are not, because
   nothing in this note is derived from that column and it is not a rescaling
   anyone should read a magnitude off — see its own paragraph above.
4. **The four-way stance split's `delta_last_minus_first`.** Reported for
   `neither_share`, where the note leans on it as a self-criticism; not tabulated
   for `proposal_share` / `values_share` / `proposal_vs_values`, where it is
   non-significant on all nine arms (p = 0.32 to 0.88) and would only pad the
   section.

Everything else that is significant and would otherwise go undiscussed is
reported, here or in the section it bears on — checked by
`test_every_significant_unprinted_cell_falls_inside_a_declared_scope_rule`. Two
measures have no section of their own at all:

- **`self_reference`** (`I / (I + we)`) **falls** across the corpus on all three
  arms: Spearman −0.300 raw / −0.317 SOTU / −0.350 genre-std, `modern − early`
  −0.066 / −0.073 / −0.120, all p ≤ 0.002. Presidents got *less* first-person
  singular relative to first-person plural, not more. It sits beside the register
  block and points the same way — away from the individual voice, toward the
  collective one — but it was not declared and it is not part of the star finding.
  Its **post-1860 ‖ block contrast**, which an earlier draft of this bullet left
  out while the sentence above it promised completeness, is **−0.023 raw
  (p = 0.153), −0.074 SOTU and −0.103 genre-std (both p = 0.0005)** and its
  endpoint contrast is −0.202 / −0.136 / −0.197 (p = 0.0005 / 0.005 / 0.0005). So
  the fall is post-1860 ‖ as well as corpus-long, but *only* once genre is held
  fixed: on the raw arm the post-1860 ‖ contrast is the one statistic of this
  measure's twelve that does not reach significance.
- **`effective_topics_plugin`**, the uncorrected twin of `effective_topics`,
  changes nothing material. Its post-1860 ‖ block contrasts are +0.22 / +0.12 /
  +0.20 (legacy 15), +0.49 / +1.19 / +0.08 (level-2) and −0.61 / −0.95 / −0.91
  (level-1), i.e. within 0.04 of the Miller-Madow figures on every one of the nine
  arms (largest gap 0.033, on level-2 raw), with the same signs and the same
  significance verdicts. On `modern − early` the two twins agree in sign on all
  nine arms with a largest gap of 0.098 (level-2 raw: +8.43 plugin against +8.33
  corrected) and on `delta_last_minus_first` a largest gap of 0.235 (level-2
  genre-std) — larger than the block gap, and in the direction that *flatters* the
  headline, which is the reason the corrected series is the published one. The
  entropy correction is not doing the work in any conclusion here.

**`normalized_entropy` is not bounded at 1.0, despite the name.** It is
`log(effective_topics) / log(K)`: the numerator carries the Miller-Madow
correction and the denominator does not, so the ratio can exceed 1 by roughly
`(K_observed − 1) / (2N log K)` — only where the sample is small enough for the
correction to be a material share of the entropy. In the shipped table it does so
on **exactly one row of 16,243** (year 1965, legacy 15, `sotu_only`: 1.0137, a
single 34-paragraph speech). It is left as it is on purpose: switching the
numerator to plug-in entropy to buy a clean [0, 1] range would make this column
disagree with `effective_topics`, which is the published series, and would perturb
a provenance-stamped artifact for a one-row cosmetic gain. Read the column as
"0–1 up to the bias-correction term". Nothing in this note is derived from it.

---

## Sub-claim 1: "presidents touch more subjects"

### The taxonomy-fit control — the bail condition did NOT fire

The plan's bail condition was: *if the entropy rise vanishes on the corpus-native
taxonomy, stop and report that.* It does not vanish. The rise is present and
significant on all three taxonomies under all three genre treatments.

Effective topics (Miller-Madow), raw treatment, first era → last era:

| taxonomy | 1770 | 1860 | 2010 | Spearman vs era [95% CI] | p |
|---|---|---|---|---|---|
| legacy 15 | 9.50 | 11.25 | 11.68 | +0.817 [0.683, 0.967] | 0.0005 |
| LLM level-2 (50) | 14.42 | 24.92 | 26.98 | +0.883 [0.733, 0.983] | 0.0005 |
| LLM level-1 (17) | 9.47 | 10.42 | 11.89 | +0.567 [0.200, 0.750] | 0.0010 |

The legacy-15 series reproduces the original observation closely (panel: 9.44 →
11.23; here 9.50 → 11.68). Under `sotu_only` and `genre_standardized` the Spearman
stays positive and significant on every taxonomy (legacy 15: +0.733 and +0.883;
level-2: +0.833 and +0.783; level-1: +0.417 and +0.483, all p ≤ 0.037). **This
sub-claim is not a taxonomy artifact and not a genre artifact.**

### But the rise is pre-1860, and the headline says "now"

The era series does not rise steadily. It climbs hard to the 1860 era and then sits
flat for 150 years. Contrast the two block statistics for `effective_topics`:

| taxonomy | treatment | modern − early | p | modern − postbellum ‖ | p |
|---|---|---|---|---|---|
| legacy 15 | raw | +1.97 | 0.0005 | **+0.22** | **0.211** |
| legacy 15 | sotu_only | +2.44 | 0.0005 | **+0.13** | **0.506** |
| legacy 15 | genre-std | +2.35 | 0.0005 | **+0.21** | **0.300** |
| LLM level-2 | raw | +8.33 | 0.0005 | **+0.46** | **0.309** |
| LLM level-2 | sotu_only | +8.62 | 0.0005 | +1.20 | 0.042 |
| LLM level-2 | genre-std | +7.75 | 0.0005 | **+0.11** | **0.711** |
| LLM level-1 | raw | +0.82 | 0.0100 | **−0.62** | **0.061** |
| LLM level-1 | sotu_only | +1.33 | 0.0005 | **−0.95** | **0.004** |
| LLM level-1 | genre-std | +0.77 | 0.0330 | **−0.90** | **0.009** |

Since the Civil War era ‖, breadth on the legacy 15 has not moved (p = 0.21–0.51),
breadth over 50 LLM topics has not moved except marginally inside annual messages
(p = 0.04, and 0.31/0.71 elsewhere), and breadth over the 17 **domains** has
significantly *declined* on two of three treatments — bearing in mind that
`sotu_only` and `genre-std` are ~74% the same paragraphs §, so "two of three
treatments" is closer to "one of two independent readings, and it is the
genre-controlled one".

### The other half of that verdict: where the 2010 era actually ranks

The block contrast above is not the only thing a reader needs, and a note that
prints era ranks for the sub-claim it is *killing* (sub-claim 2, below, where the
2010 era is **not** the maximum on the raw arms) owes the same disclosure to the
sub-claim it is *advancing*. So: **the 2010 era is the corpus maximum of
`effective_topics` on 4 of the 9 taxonomy × treatment arms, and top-two on 8 of
9.**

Rank of each era, 1 = highest, `effective_topics`:

| era | legacy raw | legacy SOTU | legacy std | L2 raw | L2 SOTU | L2 std | L1 raw | L1 SOTU | L1 std |
|---|---|---|---|---|---|---|---|---|---|
| 1770 | 8 | 8 | 8 | 9 | 9 | 9 | 8 | 8 | 8 |
| 1800 | 9 | 9 | 9 | 8 | 8 | 8 | 9 | 9 | 9 |
| 1830 | 7 | 7 | 7 | 7 | 7 | 7 | 4 | 6 | 5 |
| 1860 | 3 | 4 | 6 | 4 | 4 | 4 | 5 | **2** | 4 |
| 1890 | 4 | 5 | 4 | 6 | 6 | 5 | 3 | 3 | **1** |
| 1920 | 5 | **1** | 3 | 2 | 3 | 3 | 2 | **1** | 3 |
| 1950 | 6 | 3 | 2 | 5 | **1** | **1** | 7 | 5 | 6 |
| 1980 | 2 | 6 | 5 | 3 | 5 | 6 | 6 | 7 | 7 |
| **2010** | **1** | 2 | **1** | **1** | 2 | 2 | **1** | 4 | 2 |

The sharpest case is the arm this note calls "significantly falling". **On LLM
level-1 under `raw`, the 2010 era is the series maximum: 11.89, against 10.42 in
1860.** Both numbers are printed in the taxonomy-fit table three paragraphs
above, and an earlier draft of this note put them there and then said "breadth
has significantly declined" without reconciling the two.

**How they reconcile.** They are different questions and both answers are true.

- The block contrast averages three eras against three eras. On LLM level-1 the
  post-1860 ‖ decline is produced **on two of three arms entirely, and on
  `sotu_only` almost entirely, by the 1950 and 1980 eras**. Against the
  postbellum ‖ block mean, the three modern eras come in at −1.35 / −1.21 /
  **+0.71** (raw), −1.09 / −1.47 / −0.30 (SOTU) and −1.17 / −1.66 / **+0.12**
  (genre-std). On raw and genre-std the 2010 era is *above* the postbellum ‖ mean
  and is dragging the contrast back toward zero, so those two declines are
  produced entirely by 1950 and 1980. On `sotu_only` the 2010 era is **−0.30
  below** the block mean and so contributes **10.5%** of that arm's decline — and
  `sotu_only` is one of the two arms on which the decline is significant, so
  "entirely" would have been overstated exactly where it matters most. The
  1950/1980 trough still supplies the other ~90%.
- The rank is a statement about one era, resting on 3 presidents and one
  bootstrap cell, with **no test behind the ordering**. On the legacy 15 raw
  arm the rank-1 and rank-2 eras are 2010 at 11.679 [11.270, 11.953] and 1980 at
  11.666 [11.294, 11.938] — a gap of 0.013 between two indistinguishable cells.
  "Rank 1" is not "significantly the highest", and the table above should be read
  as a description of the series shape, not as nine ordered findings.

So the honest reading is **not** "modern breadth is low" — it plainly is not; on
most arms the 2010 era is the highest of the nine. It is that **the mid-20th
century (1950/1980) is a trough**, and the post-1860 ‖ block contrast is measuring
that trough as much as it is measuring anything about the present. A reader given
only the block contrast would infer a modern narrowing that the era series does
not contain — which is the same error, in the opposite direction, that the block
contrast was added to prevent.

What does *not* change: the large move is still pre-1860. On LLM level-2 raw the
early→postbellum ‖ step is +7.88 of the +8.33 total (**94.5%**); on the legacy 15
raw it is +1.75 of +1.97 (89.0%). And no arm shows a *significant* post-1860 ‖
rise except level-2 `sotu_only` (+1.20, p = 0.042 — marginal under multiplicity).
"Presidents now touch more subjects than in 1860" is not supported; "presidents
now touch fewer" is also not supported except on LLM level-1 under genre control,
where the decline is real, is marginal under multiplicity, and is a 1950s–1980s
phenomenon rather than a present-day one.

### Rarefaction: is the pre-1860 rise just sample size?

Partly. Subsampling whole speeches until each era reaches the smallest era's
paragraph budget (355 paragraphs, 400 draws, seed 20260721).

**The contrast reported here is now the same contrast the conclusions use.** An
earlier draft tabulated only *endpoint* differences (1770→2010, 1860→2010) while
the conclusions they were cited for are *block* contrasts. Endpoint and block are
different quantities — that is the whole point of the section above — so the
endpoint columns were checking something the note does not claim. Both are given
below; the block columns are the ones that bear on the verdicts.

| taxonomy | full 1770→2010 | rarefied | full modern − early | rarefied | full modern − postbellum ‖ | rarefied |
|---|---|---|---|---|---|---|
| legacy 15 | +2.18 | +1.66 | +1.97 | +1.72 | +0.22 | **+0.07** |
| LLM level-2 | +12.56 | +8.25 | +8.33 | +6.19 | +0.46 | **+0.13** |
| LLM level-1 | +2.42 | +1.40 | +0.82 | +0.59 | −0.62 | **−0.76** |

On the endpoint contrast, between 23% (legacy 15) and 42% (LLM level-1) of the
apparent long-run rise is sample-size bias — 34% on LLM level-2. On the
**`modern − early`** block contrast the bias is smaller, because the blocks
average out the thinnest single cells: 13% (legacy 15), 26% (level-2), 29%
(level-1). Those three shares are `modern − early` figures and only those; the
same arithmetic on the **post-1860 ‖** block contrast gives a completely different
picture, and the table above is what a reader should work from rather than these
percentages: there the bias share is **68%** (legacy 15, +0.22 → +0.07), **71%**
(level-2, +0.46 → +0.13) and **−22%** (level-1, −0.62 → −0.76, where
rarefaction *deepens* the decline instead of shrinking it). The post-1860 ‖
contrasts are small differences of large numbers, so a small absolute correction
is a large proportional one — which is a reason to read those two flatness
verdicts off the sign and the interval, not off the percentage. The remainder is
real, but the level-1 endpoint share is large enough that "roughly a quarter to a
third", the summary this note first reached for, understates it on the taxonomy
where breadth is *falling* post-1860 ‖.

**The post-1860 ‖ conclusions survive rarefaction, and two of them get stronger.**
The legacy-15 and level-2 contrasts shrink toward zero, so the flatness verdict is
if anything understated in the published table; the level-1 decline gets *larger*.
Sample-size bias was working in favour of the modern era on every arm — the
direction that would have manufactured a modern rise.

**Residual mismatch, disclosed.** `rarefied_effective_topics` runs on the full
(raw) sample only — it subsamples whole speeches of every genre. The verdicts in
this section prefer the genre-controlled arms, so this check speaks directly to
the `raw` column and only by analogy to the other two. Building a genre-held-fixed
rarefaction would mean re-deriving the paragraph budget inside annual messages
(185 paragraphs in the 1770 era rather than 355) and is not done here. What can
be said is the direction of the residual risk, and it does not favour the note:
on the two genre-controlled arms the level-1 post-1860 ‖ decline is *larger* than
on raw (−0.95 and −0.90 against −0.62), so a genre-controlled rarefaction would
be expected to deepen that decline, not remove it; and the legacy-15 post-1860 ‖
contrast is at its *largest* on the raw arm (+0.22, against +0.13 SOTU and +0.21
genre-std), so rarefying raw tests the flatness verdict where a rise had the most
room to appear.

**Honest statement of sub-claim 1:** presidential agendas broadened substantially
between Washington and Lincoln, on both taxonomies, inside annual messages alone,
and after correcting for sample size. They have not broadened significantly since
— but neither have they narrowed in the present: the 2010 era is the highest of
the nine on 4 of 9 arms and top-two on 8 of 9, and the one significant post-1860 ‖
decline (LLM level-1, under genre control) is driven by the 1950 and 1980 eras,
not by the modern end of the series.

---

## Sub-claim 2: "presidents say less about each" — dead

### Per paragraph, the decline is weak and does not survive genre control

`labels_per_paragraph`:

| taxonomy | treatment | Spearman | p | modern − early [95% CI] | p |
|---|---|---|---|---|---|
| legacy 15 | raw | −0.283 | 0.109 | −0.145 [−0.246, −0.049] | 0.0010 |
| legacy 15 | sotu_only | **+0.250** | 0.367 | **+0.063** [−0.068, +0.194] | 0.345 |
| legacy 15 | genre-std | **+0.483** | 0.405 | **+0.041** [−0.077, +0.158] | 0.501 |
| LLM level-2 | raw | +0.133 | 0.589 | +0.026 [−0.038, +0.087] | 0.413 |
| LLM level-2 | sotu_only | +0.150 | 0.181 | +0.050 [−0.019, +0.121] | 0.168 |
| LLM level-2 | genre-std | +0.250 | 0.165 | +0.048 [−0.016, +0.110] | 0.143 |

**On `modern − early` the sign flips** the moment genre is controlled, and on that
contrast nothing is significant except the raw legacy figure — which is −0.145
labels, or −11.8% of the early block's 1.233, not the −44% the headline rests on.
The original 1.49 → 0.83 figure is reproducible here as a **1920-era-vs-2010-era
raw contrast** (1.455 → 0.985), but 1920 is the series *maximum* and the series is
not monotone. It is an endpoint pair, not a trend.

That is a statement about `modern − early`, and it is **not** true of the other
block contrast, which this note computed and an earlier draft printed nowhere. All
six arms of `delta_modern_minus_postbellum` ‖, none omitted:

| taxonomy | treatment | modern − postbellum ‖ [95% CI] | p |
|---|---|---|---|
| legacy 15 | raw | **−0.260 [−0.319, −0.199]** | **0.0005** |
| legacy 15 | sotu_only | −0.061 [−0.146, +0.030] | 0.190 |
| legacy 15 | genre-std | −0.060 [−0.137, +0.020] | 0.145 |
| LLM level-2 | raw | +0.051 [−0.004, +0.102] | 0.065 |
| LLM level-2 | sotu_only | **+0.141 [+0.076, +0.204]** | **0.0005** |
| LLM level-2 | genre-std | **+0.120 [+0.062, +0.173]** | **0.0005** |

**What this contrast actually shows, stated against the note's own interest.**
Three of the six are significant and all three are at the resolution floor. The
legacy-15 sign does **not** flip under genre control here — it is negative on all
three arms (−0.260, −0.061, −0.060); the controls shrink it by about three
quarters and take it out of significance, but they do not reverse it. On LLM
level-2 the sign is positive on all three and significant under both controls. So
post-1860 ‖ the two labelers **disagree in sign**, which is a weaker and more honest
position than either "the sign flips under control" (false here) or "nothing is
significant" (false three times over).

It also means this note has been quoting the *smaller* of the two available
contrasts to size the decline it dismisses. Measured from 1860 ‖ instead of from
the founding block, the raw legacy-15 fall is **−0.260, or −19.3% of the
postbellum ‖ block's 1.347** — larger than the −0.145 / −11.8% quoted above, and at
the bootstrap floor rather than at p = 0.001. Both are real; the note reports the
larger one now that it reports both. (`delta_last_minus_first` is the third
contrast and is non-significant on all six arms, p = 0.24 to 0.85, so it
adjudicates nothing either way.)

**The verdict does not move, and is not being defended by omission.** Sub-claim 2
dies on `labels_per_1k_words`, not on `labels_per_paragraph`: the length-invariant
measure rises post-1860 ‖ on five of six arms (below), and `mean_paragraph_words`
falls −22.0 raw / −23.0 SOTU / −24.2 genre-std post-1860 ‖, all three at the
floor. The mechanism — shorter paragraphs, not shallower treatment — is intact
whichever baseline is used. What the block contrast adds is that on the legacy 15,
measured from 1860 ‖ and without genre control, labels *per paragraph* genuinely did
fall, and by more than the figure this note had been quoting.

### Per 1,000 words, the sign reverses and the effect is strong

`labels_per_1k_words` — the length-invariant twin:

| taxonomy | treatment | 1770 | 2010 | Spearman | p | modern − early | p |
|---|---|---|---|---|---|---|---|
| legacy 15 | sotu_only | 10.72 | 14.33 | **+0.917** | 0.0005 | +3.54 [2.53, 4.54] | 0.0005 |
| legacy 15 | genre-std | 10.24 | 14.33 | **+0.900** | 0.0005 | +3.27 [2.36, 4.14] | 0.0005 |
| legacy 15 | raw | 8.68 | 10.08 | +0.500 | 0.0200 | +1.03 [0.23, 1.81] | 0.0060 |
| LLM level-2 | sotu_only | 13.15 | 16.50 | **+0.617** | 0.0005 | +3.72 [2.86, 4.54] | 0.0005 |
| LLM level-2 | raw | 12.14 | 14.68 | **+0.667** | 0.0005 | +3.18 [2.52, 3.84] | 0.0005 |
| LLM level-2 | genre-std | 12.64 | 16.75 | **+0.617** | 0.0005 | +3.73 [2.96, 4.45] | 0.0005 |

Unlike breadth, this one mostly *does* survive post-1860 ‖. All six arms of
`delta_modern_minus_postbellum` ‖, none omitted:

| taxonomy | treatment | modern − postbellum ‖ [95% CI] | p |
|---|---|---|---|
| legacy 15 | raw | **−0.24 [−0.76, +0.32]** | **0.387** |
| legacy 15 | sotu_only | +2.20 [1.50, 2.91] | 0.0005 |
| legacy 15 | genre-std | +2.19 [1.54, 2.84] | 0.0005 |
| LLM level-2 | raw | +2.98 [2.45, 3.50] | 0.0005 |
| LLM level-2 | sotu_only | +4.14 [3.49, 4.85] | 0.0005 |
| LLM level-2 | genre-std | +4.00 [3.36, 4.60] | 0.0005 |

**Five of the six survive; one does not.** The legacy-15 *raw* arm is −0.24 with an
interval straddling zero (p = 0.39) — the only arm of this measure anywhere in the
note that fails, and it fails against the direction being argued. It is the same
arm that is weakest across the corpus (+1.03 modern − early, against +3.27 and
+3.54 under genre control); the likely reason, though this note does not test it,
is that the raw arm pools in the short modern non-SOTU genres in which CorEx's
19th-century-fitted anchors fire least. It does not reproduce on the second
labeler, where all three treatments survive.

So the honest form of the claim is "density per unit of speech is higher than the
postbellum era ‖ on five of six arms, and flat on the sixth" — not "on all of
them". And "five of six" is generous even so §: the two surviving legacy-15 arms
(+2.20 SOTU, +2.19 genre-std) are three-quarters the same paragraphs, so on the
legacy 15 this is really **one independent genre-controlled reading that
survives against one raw reading that does not**, and the four-of-four on the
LLM labeler is likewise two independent readings, not four. The finding stands on
the cross-*labeler* replication, which is genuinely independent, more than on the
arm count.

### The mechanism: paragraphs got shorter

`mean_paragraph_words` falls from 135.9 (1830 era) to 97.7 (2010), and inside
annual messages alone from 134.0 to 92.9 — Spearman −0.833 raw / −0.633 SOTU /
−0.633 genre-std, modern − early = −26.7 raw [−30.7, −23.1], −27.1 SOTU
[−32.1, −21.8] and −27.9 genre-std, all p = 0.0005.

This is the one measure in sub-claim 2 whose verdict does not depend on the
baseline, and all twelve of its trend cells are given rather than the three the
argument needs. Post-1860 ‖ it falls **−22.0 raw [−24.7, −19.4] / −23.0 SOTU
[−27.6, −18.4] / −24.2 genre-std [−28.1, −19.9]**, and on the endpoint contrast
−27.7 / −27.1 / −31.2 — twelve of twelve at the bootstrap floor, on every
treatment and every one of the four statistics. Paragraph shrinkage is not an
artifact of which block you measure from.

**Honest statement of sub-claim 2:** presidents did not start saying less about
each subject. They started writing shorter paragraphs. Per unit of speech the
modern era touches *more* issue-labels than the founding era, not fewer, on every
arm. Whether it touches more than *any* earlier era depends on the treatment: the
2010 era is the maximum of the nine on all four genre-controlled arms, but on the
raw arms it is not — the raw legacy series peaks in the 1920 era (11.70 against
10.08 in 2010, ranking the modern era 6th of nine) and the raw level-2 series in
the 1950 era (15.15 against 14.68, 2nd of nine). The "shallower" finding was a
typographic artifact of using the paragraph as the denominator; the "highest ever"
reading is only safe once genre is held fixed.

---

## Sub-claim 3: "more speech off-policy" — split by labeler

### Legacy labeler (zero-anchored-issue rate): dies under genre control

| treatment | 1770 | 1920 | 2010 | Spearman | p | modern − early | p |
|---|---|---|---|---|---|---|---|
| raw | 0.366 | 0.195 | 0.382 | +0.183 | 0.232 | +0.045 [0.006, 0.084] | 0.022 |
| sotu_only | 0.286 | 0.134 | 0.242 | **−0.233** | 0.154 | **−0.031** [−0.073, +0.009] | 0.143 |
| genre-std | 0.300 | 0.154 | 0.251 | **−0.383** | 0.095 | **−0.030** [−0.067, +0.007] | 0.123 |

The famous 19.5% → 38.2% is reproduced exactly as a **1920-vs-2010 raw** contrast.
Both endpoints are extremes of a non-monotone series whose 1770 value (36.6%) is
already near its 2010 value. Under either genre control the sign flips negative.

**The post-1860 ‖ contrast on this labeler, which an earlier draft omitted, runs
the other way — in the headline's favour.** The table above reports Spearman and
`modern − early` only. The block contrast the note added specifically because it
*weakens* claims is printed for the LLM labeler in the very next table and was
not printed here, and on this labeler it does not weaken the claim:

| treatment | modern − postbellum ‖ | p |
|---|---|---|
| raw | **+0.105** | **0.0005** |
| sotu_only | +0.026 | 0.063 |
| genre-std | +0.023 | 0.087 |

So "dies on the legacy labeler under genre control" is true of the corpus-long
contrast and **not** true of the post-1860 ‖ one, which is positive on all three
arms and significant on raw. Measured from the 1860/1890/1920 block ‖, the legacy
labeler agrees with the LLM labeler that off-policy speech rose; it is only
measured from the founding block that the two disagree. That is a point *for* the
original headline, and omitting it was the same failure of symmetry the note
corrects elsewhere — this time in the direction that flattered the note's own
skepticism. The verdict is unchanged (the two genre-controlled arms are still
n.s. at 0.063 and 0.087, and both would be excluded by any multiplicity
correction), but the reader now has the arm that argues against it.

### LLM labeler (all topics non-policy-parented): survives, but small

| treatment | 1770 | 1830 | 2010 | Spearman | p | modern − early | p | modern − postbellum ‖ | p |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.192 | 0.053 | 0.174 | +0.283 | 0.0005 | +0.050 | 0.009 | +0.074 | 0.0005 |
| sotu_only | 0.141 | 0.061 | 0.117 | +0.100 | 0.040 | +0.031 | 0.014 | +0.053 | 0.0005 |
| genre-std | 0.167 | 0.057 | 0.136 | +0.183 | 0.0005 | +0.030 | 0.009 | +0.063 | 0.0005 |

This is the one part of the headline that survives every control *including* the
post-1860 ‖ contrast. But note the shape: the series is U-shaped, and the modern
level (17.4%) is **below** the founding era's (19.2%).
`delta_last_minus_first` is negative and non-significant on all three treatments.
Note also that the evidence is **not** the same strength on the two block
contrasts. The three `modern − early` p-values (0.009 / 0.014 / 0.009) sit in the
band where a multiplicity correction bites, which on *that* contrast makes this
the weakest-evidenced of the surviving claims; the post-1860 ‖ contrast is at the
bootstrap floor on all three arms and sits in the "would not be excluded" table.
Both are printed above, and this is the last surviving piece of the original
headline either way.

Where the U bottoms is not the same under every treatment. Under `raw` and
`genre_standardized` it is the 1830 era (5.3% and 5.7%); under `sotu_only` the
minimum is the **1920** era (5.60%), fractionally below 1830 (6.08%) — close
enough that the two are not really distinguishable, but "bottoming in the 1830s"
is only exactly true on two of the three treatments.

Two different baselines, kept apart because they give different numbers. The
tabulated `modern − early` contrast above measures the modern block against the
**1770/1800/1830 block**, whose average is pulled up by the 19.2% founding era:
that is the +3.0 to +5.0 points in the table (raw +5.0, SOTU +3.1, genre-std
+3.0). Measured instead from each treatment's actual **trough** to the 2010 era,
the rise is larger: +12.1 (raw, from 1830), +6.1 (SOTU, from 1920), +7.9
(genre-std, from 1830). Neither baseline gives a doubling of the founding-era
level, which is what the original 19.5% → 38.2% framing implies.

`zero_topic_share` (the 1.1% residue) is 0.041 in the 2010 era under `raw` and
exactly 0.000 under both genre controls. An earlier draft called it "a modern
non-SOTU genre artifact, not a trend", and the second half of that is wrong
against this note's own artifact: on the **raw** arm all four of its trend
statistics are significant at the bootstrap floor (Spearman +0.733;
`modern − early` +0.019; post-1860 ‖ +0.019; endpoint +0.041, all p = 0.0005). It
is a perfectly good trend — inside the raw arm. What is true is that it is
*entirely* a genre effect: under `genre_standardized` nothing is significant
(+0.002, −0.001, and an endpoint of exactly 0.000), and under `sotu_only` only the
post-1860 ‖ contrast clears 0.05, at +0.0026 (p = 0.016) — a quarter of a
percentage point. The correct statement is "a real trend in the modern non-SOTU
genres and nowhere else", which supports the same conclusion by a route that does
not require denying three floor-level cells.

**Honest statement of sub-claim 3:** relative to the founding block (1770–1859), a
modestly larger share of presidential paragraphs is purely ceremonial, personal,
procedural or values-based — **about three points** once genre is held fixed (+3.1
SOTU-only, +3.0 genre-standardized), and five points on the uncontrolled raw arm.
The three-point figure is the one this note stands behind, because §"Where the
three genre treatments disagree" commits to the genre-controlled reading
everywhere else, and quoting the larger raw number in the summary while preferring
the controls in the method would be having it both ways. The effect is visible
only on the LLM labeler *when measured from the founding block*; relative to the
founding era alone there is no increase at all; and the legacy labeler shows the
opposite sign once genre is held fixed — though, measured from 1860 ‖ instead, the
legacy labeler agrees (+0.105 raw, p = 0.0005; +0.026 and +0.023 under the
controls, n.s.).

---

## Bonus: "more values-driven" — dead as stated

`proposal_vs_values` (proposal / (proposal + values), unambiguous paragraphs only)
is **flat with an unstable sign** across the full corpus:

| treatment | Spearman | p | modern − early | p |
|---|---|---|---|---|
| raw | −0.250 | 0.391 | −0.029 | 0.334 |
| sotu_only | +0.117 | 0.943 | +0.018 | 0.515 |
| genre-std | −0.067 | 0.972 | +0.011 | 0.642 |

Measured from 1860 ‖ it *does* fall significantly (raw −0.131 p = 0.0005; SOTU
−0.092 p = 0.002; genre-std −0.079 p = 0.0005) — but the founding era was itself
values-heavy, so the full-corpus trend is flat.

More importantly the ratio hides what actually moved. **Both** stance shares rise:
`values_share` +0.153 and `proposal_share` +0.180 (SOTU, modern − early, both
p = 0.0005; the same rise appears on the other two arms — raw +0.098 / +0.064,
genre-std +0.161 / +0.162, all p ≤ 0.006).

**That rebuttal must be checked on the baseline it is rebutting, and on one arm it
fails.** The finding being explained away is the post-1860 ‖ fall in the ratio, but
the evidence just quoted is `modern − early`. Switching baselines mid-argument is
the error this note flags elsewhere, so here are the same two shares on the
contrast that is actually at issue, all six arms:

| share | raw | sotu_only | genre-std |
|---|---|---|---|
| `values_share`, modern − postbellum ‖ | **+0.122**, p = 0.0005 | **+0.179**, p = 0.0005 | **+0.175**, p = 0.0005 |
| `proposal_share`, modern − postbellum ‖ | **−0.011, p = 0.524** | **+0.083**, p = 0.0005 | **+0.085**, p = 0.0005 |

So "both stance shares rise" holds **on five of six arms** §, not six. On the
uncontrolled arm, at the very baseline this section's verdict names, `proposal_share`
is **flat** (−0.011, p = 0.52) while `values_share` rises at the floor. There, and
only there, the ratio's post-1860 ‖ fall is values rising against a *static*
proposal share — which is closer to the "more values-driven" reading this
paragraph is denying than the paragraph admitted. The rebuttal survives on the two
genre-controlled arms and on both raw and controlled arms of `modern − early`, and
`values_share` rises at the floor on every arm of every block contrast; but a
reader entitled to the strongest version of the opposing case gets it from the raw
column above. (`proposal_share`'s raw Spearman is likewise the only non-significant
one of its three, +0.333, p = 0.151, against +0.733, p = 0.0005 under both
controls; `values_share`'s three are +0.333 raw, +0.583 SOTU and +0.667 genre-std,
all at the floor.)

What collapses is `neither_share` —
descriptive, non-stance prose — which inside annual messages falls from 48.0%
(1800 era) to 8.9% (1980) and 20.5% (2010): Spearman −0.717, modern − early
−0.262 [−0.309, −0.216], p = 0.0005. It falls significantly under all three genre
treatments (raw −0.093, genre-std −0.254, both p = 0.0005), so this is not the
genre mix either. The Spearman is negative and significant on all three as well
(raw −0.483, p = 0.006; SOTU −0.717 and genre-std −0.783, both p = 0.0005) — given
per arm because this section says below that "the block contrast and the Spearman
are what carry this finding", which is not checkable from one arm of one of them.
On the post-1860 ‖ contrast it also falls on all three —
**−0.060 raw (p = 0.005), −0.176 SOTU and −0.192 genre-std (both p = 0.0005)** —
so unlike the two stance shares, this half of the bonus finding does not depend on
which block it is measured from.

**The collapse is measured from the 19th-century plateau, not from the founding
era — and the endpoint contrast does not support it.** The same disclosure
sub-claim 3 makes about its own baseline applies here and was missing from an
earlier draft. The 1770 era's `neither_share` is *already* low (31.9% SOTU,
24.8% raw), so the fall is from the 1800–1890 plateau rather than from the corpus
start. That plateau has to be quoted **per arm**, because an earlier draft mixed
its top from one arm with its floor from another: on `sotu_only` it runs 48.0%
(1800) down to 37.3% (1860); on `raw`, 43.2% (1800) down to 37.1% (1890); on
`genre-std`, 45.4% (1800) down to 37.6% (1890). `delta_last_minus_first` — the 1770→2010
endpoint — is **−0.114, p = 0.062 (SOTU); −0.114, p = 0.047 (genre-std); and
+0.092, p = 0.077 (raw)**: on the uncontrolled arm the 2010 era's neutral-prose
share sits *above* the founding era's, and on the two controlled arms the decline
against 1770 is at or beyond the edge of significance. The block contrast and the
Spearman are what carry this finding; the endpoint contrast does not, and saying
so is the price of having said the same thing about sub-claim 3.

`neither_share` ships in `trends.parquet` beside the other two stance measures so
that this claim can be looked up rather than taken on trust. It was added at
write-up time, after the paragraph above was drafted, for exactly that reason: it
was the one figure in this note that the advertised parquet did not contain. It is
the arithmetic residual of `proposal_share` and `values_share`, not a new arm —
and because each of those counts `mixed` on its own side, adding it makes the full
four-way split recoverable:
`mixed_share = proposal_share + values_share + neither_share − 1`.

**Honest statement:** presidents did not trade policy for values. They traded
*neutral description* for both — measured against the 19th-century annual
message, which was a departmental report, rather than against the founding era,
which already argued. The modern annual message argues too.

---

## ★ What actually survives everything: the register changed

Not the headline, and much stronger than it. Every marker below moves the same
direction under **all three** genre treatments — and, unlike everywhere else in
this note, that is not just a sign agreement: the `modern − early` contrast is
significant at the bootstrap floor (p = 0.0005) on **all three arms for all
eleven markers**, so this is the one block where the arm-dependence caveat § does
not soften anything. Contrast tabulated is modern − early, `sotu_only`.

Monotonicity, stated precisely rather than as "near-perfect": eight of the eleven
have |Spearman| ≥ 0.90 under `sotu_only` (the weakest two of those eight are
`mechanism` and `hype` at 0.917); the other three are `superlatives` (+0.77),
`future` (+0.67) and `nostalgia` (+0.47), and those three are visible in the
Spearman column below rather than described.

**Units are not the same down the column**, so each row carries its own. Nine of
the eleven are lexicon counts per 10,000 words. The two marked ‡ are not:
`fk_grade` is a Flesch-Kincaid **grade level** and `words_per_sentence` is **words
per sentence**. Read as per-10k rates they would say that 19th-century annual
messages contained 19 grade levels and 37 sentence-lengths per 10,000 words, which
is not a quantity.

| marker | unit | 1770 SOTU | 2010 SOTU | Spearman (raw / SOTU / std) | Δ SOTU [95% CI] | p |
|---|---|---|---|---|---|---|
| `mechanism` (policy machinery) | per 10k words | 47.7 | 17.7 | −0.72 / **−0.92** / −0.80 | −27.0 [−32.9, −21.1] | 0.0005 |
| `hedges` | per 10k words | 46.8 | 7.5 | **−1.00** / −0.95 / −1.00 | −35.7 [−39.1, −32.2] | 0.0005 |
| ‡ `fk_grade` | grade level | 19.0 | 8.6 | **−1.00** / −0.95 / −0.98 | −9.0 [−9.5, −8.5] | 0.0005 |
| ‡ `words_per_sentence` | words/sentence | 37.2 | 15.5 | −0.98 / −0.95 / −0.98 | −19.5 [−20.6, −18.5] | 0.0005 |
| `opponents` | per 10k words | 0.0 | 12.5 | **+1.00** / **+1.00** / +0.98 | +6.5 [5.1, 7.9] | 0.0005 |
| `boosters` | per 10k words | 5.4 | 26.5 | +0.87 / +0.93 / +0.97 | +10.4 [7.2, 13.9] | 0.0005 |
| `hype` | per 10k words | 1.8 | 17.3 | +0.85 / +0.92 / +0.93 | +8.9 [6.9, 11.0] | 0.0005 |
| `doom` | per 10k words | 1.4 | 10.1 | +0.73 / +0.98 / +0.98 | +6.0 [4.8, 7.2] | 0.0005 |
| `future` | per 10k words | 5.9 | 16.4 | +0.70 / +0.67 / +0.67 | +13.0 [10.6, 15.4] | 0.0005 |
| `nostalgia` | per 10k words | 7.7 | 15.6 | +0.83 / +0.47 / +0.67 | +6.5 [4.7, 8.3] | 0.0005 |
| `superlatives` | per 10k words | 5.9 | 15.3 | +0.73 / +0.77 / +0.80 | +6.5 [4.6, 8.5] | 0.0005 |

The formal presidential record became **shorter-sentenced, plainer, far less
hedged, far less procedural, and far more evaluative** — more boosting, more
opponents, more hype and doom, more appeals forward and backward in time. This
holds inside the annual message alone, so it is about presidents, not about the
corpus admitting new genres.

### And it survives the post hoc contrast too, which was not previously printed

Everywhere else the post-1860 ‖ block contrast *weakens* a claim, which is why
this note added it. On the register block it does not, and leaving it out was the
one omission here that made a finding look **weaker** than it is — completeness
has to cut both ways:

| marker | raw | sotu_only | genre-std |
|---|---|---|---|
| `mechanism` | −26.4 | −21.2 | −24.7 |
| `hedges` | −17.2 | −22.9 | −22.6 |
| ‡ `fk_grade` | −6.4 | −5.7 | −5.9 |
| ‡ `words_per_sentence` | −11.8 | −11.1 | −11.2 |
| `opponents` | +6.0 | +5.0 | +4.9 |
| `boosters` | +9.4 | +8.3 | +7.5 |
| `hype` | +7.3 | +7.9 | +7.6 |
| `doom` | +2.4 | +4.1 | +4.0 |
| `future` | +9.0 | +12.8 | +12.3 |
| `nostalgia` | +3.9 | +6.5 | +6.3 |
| `superlatives` | +5.1 | +6.6 | +5.8 |

**All 33 cells are significant at the bootstrap floor (p = 0.0005), with the same
sign as the `modern − early` contrast in every one.** The endpoint contrast agrees
too: 33 of 33 significant, 32 of them at the floor (`mechanism` under `sotu_only`
is p = 0.002). Counting the whole block — eleven markers × three genre treatments
× four trend statistics — **132 of 132 tests are significant and 131 of the 132
are at the resolution floor.** That is what "survives everything" is standing on,
and it is now checkable rather than asserted. Nothing else in this note has that
property, and the post hoc contrast that killed or qualified every other claim is
the one that confirms this one.

**Markers that do NOT survive.** `us_them` and `religiosity` are in the
Pre-declaration table; `nrc_fear` and `nrc_hope` were **not declared** there, and
are reported for the reason `self_reference` is — they were computed. Contrast is
`modern − early`:

| marker | raw | sotu_only | genre-std | reading (on this contrast) |
|---|---|---|---|---|
| `us_them` | +12.7, p = 0.003 | **+0.13, p = 0.984** | **−0.53, p = 0.871** | genre artifact; dies under control |
| `nrc_hope` | −74.9, p = 0.0005 | **−14.3, p = 0.196** | −24.1, p = 0.021 | mostly genre |
| `nrc_fear` | **+1.5, p = 0.838** | +30.7, p = 0.009 | +27.9, p = 0.009 | *only* visible under genre control |
| `religiosity` | **+2.1, p = 0.186** | +3.6, p = 0.002 | +5.3, p = 0.0005 | *only* visible under genre control |

`us_them` was pre-declared as rising and does not. `religiosity` was pre-declared
as rising and does so only once genre is held fixed — the raw series is flat
because modern non-SOTU genres dilute it — and undeclared `nrc_fear` behaves the
same way. Both readings are published; neither is dropped. Note that these two are
also the register-block
results that a multiplicity correction would demote (p = 0.009 and 0.002 under
`sotu_only`), and that they rest on the *dependent* pair of arms § — the two
genre-controlled readings that share 74% of their paragraphs. They are the
weakest members of the star finding, not representative of it.

**Every reading in that table is a reading of `modern − early`, and three of the
four reverse on the post hoc contrast.** These four markers are the only place in
the note where a verdict was published off one block contrast without the other
being checked, and the other one disagrees:

| marker | raw | sotu_only | genre-std | reading (post-1860 ‖) |
|---|---|---|---|---|
| `us_them` | **+26.4**, p = 0.0005 | **+18.5**, p = 0.0005 | **+15.0**, p = 0.0005 | rises at the floor on **all three** — not a genre artifact here |
| `nrc_hope` | **−36.3**, p = 0.0005 | −3.2, p = 0.822 | −1.3, p = 0.937 | raw only; the controls are flat, not merely weaker |
| `nrc_fear` | **−20.5**, p = 0.001 | +10.1, p = 0.238 | +4.9, p = 0.528 | **falls** raw; nothing under control — the opposite pattern |
| `religiosity` | **+6.5**, p = 0.0005 | **+7.6**, p = 0.0005 | **+8.5**, p = 0.0005 | rises at the floor on **all three**, raw included |

So, precisely:

- **`us_them` does not "die under control" — it dies under control *on the
  corpus-long contrast only*.** Post-1860 it rises at the bootstrap floor on all
  three arms, and its endpoint contrast is significant on all three too (+51.4
  raw, +23.4 SOTU, +22.6 genre-std; p = 0.0005 / 0.036 / 0.022). Its four
  statistics split three ways, not two: `modern − early` says "genre artifact",
  the Spearman says **no trend anywhere** (raw +0.150, p = 0.147; SOTU +0.083 and
  genre-std +0.033, both n.s. — so there was nothing on raw for the control to
  kill), and the post-1860 ‖ and endpoint contrasts say "real on every arm". The
  declared prediction was that `us_them` rises; measured from 1860 ‖, it does.
- **`religiosity` is not "*only* visible under genre control".** That is true of
  `modern − early` and of the Spearman (raw +0.367, p = 0.105, against +0.267
  SOTU, p = 0.030 and +0.400 genre-std, p = 0.001), and false of the
  post-1860 ‖ contrast, where the raw arm is +6.5 at the floor — three times the
  +2.1 the table above reports and significant rather than not. The direction of
  the correction here runs *for* the pre-declared prediction, which is why it is
  spelled out rather than left in the parquet.
- **`nrc_fear` reverses sign between arms, not between statistics.** It was never
  declared. It rises under genre control on `modern − early` (+30.7 / +27.9) and
  **falls significantly on raw post-1860 ‖** (−20.5, p = 0.001), with both
  controls null on that contrast. Its Spearmans — left out of an earlier draft
  while the completeness rule above promised them — are **+0.100 raw
  (p = 0.857), +0.733 SOTU (p = 0.029) and +0.783 genre-std (p = 0.028)**, and its
  endpoint contrast is n.s. on all three (p = 0.32 to 0.52). So under genre
  control two of its four statistics are significantly positive and none is
  significantly negative: it is the **raw arm**, not the measure, that is
  unstable, and calling it "the least stable measure in the note" (an earlier
  draft did) overstated it in the direction of the note's own skepticism. No
  single reading of it is defended.
- **`nrc_hope` is the one whose earlier summary was simply wrong**, and it was
  wrong against a number the note already printed: it is called a pure genre
  artifact below, but its `modern − early` fall under `genre_standardized` is
  −24.1 at p = 0.021 — significant, and listed as such in the multiplicity
  section — and its Spearman is significantly negative on **all three** arms
  (−0.667 raw and −0.267 genre-std, both p = 0.0005; −0.233 SOTU, p = 0.016), as
  is its endpoint contrast (p = 0.0005). Only the post-1860 ‖ contrast supports
  "nothing under control". "Mostly genre" is right; "nothing under control" was
  not.

---

## Where the three genre treatments disagree, and which reading is honest

1. **`labels_per_paragraph`** — on `modern − early`, raw says "falling" and both
   controls say "flat or rising". **Honest reading: the controls.** The raw
   decline is the changing mix of genres (short modern remarks) plus shorter
   paragraphs, not presidents compressing their treatment of issues. **This is a
   statement about `modern − early` and does not carry to the post hoc contrast:**
   post-1860 ‖ the legacy-15 arms are negative on *all three* treatments (−0.260
   raw at the floor, −0.061 and −0.060 n.s.), so there the controls do not say
   "flat or rising" — they say "the same direction, three-quarters smaller, and no
   longer significant". The cross-*labeler* split does the adjudicating on that
   contrast instead: level-2 is positive on all three arms and significant under
   both controls. See sub-claim 2.
2. **`non_policy_share` (legacy 15)** — on `modern − early`, raw says "rising" and
   both controls say "falling, n.s.". **Honest reading: the controls.** This one
   is the *form* of presidential speech changing, not the presidents. **On the
   post-1860 ‖ contrast there is no disagreement to adjudicate:** all three arms
   are positive (+0.105 raw at the bootstrap floor, +0.026 and +0.023 at p = 0.063
   and 0.087), so the labeler that kills sub-claim 3 on the corpus-long contrast
   agrees with it on the post hoc one. The verdict below rests on the two n.s.
   controlled arms and is unchanged, but "both controls say falling" is true of
   exactly one of the two block contrasts.
3. **`us_them`, `nrc_hope`** — significant raw on `modern − early`, and the
   controls disagree, but not in the same way and the item has to be scoped per
   marker. `us_them`: nothing under either control on `modern − early`
   (p = 0.98 / 0.87). `nrc_hope`: **`sotu_only` is null (−14.3, p = 0.196) while
   `genre_standardized` is significant (−24.1, p = 0.021)** on that same contrast,
   so "nothing under control" was never true of it. **Honest reading: the
   controls**, and on `modern − early` `us_them` is a genre artifact. Neither is a
   clean genre artifact across the table: `us_them` is at the floor on all three
   arms post-1860 ‖ and significant on all three at the endpoints, and `nrc_hope`
   is significantly negative on all three arms of the Spearman and on all three at
   the endpoints. See the register section, where all four non-surviving markers
   are given on both block contrasts.
4. **`nrc_fear`, `religiosity`** — nothing raw, significant under control. **Honest
   reading: the controls again**, and this is the direction that costs the analyst
   something: holding genre fixed *creates* two findings as well as destroying
   four. Applying the control only when it helps would be the error.
5. **Everything in the register block, `mean_paragraph_words`, and the
   corpus-long form (Spearman, `modern − early`) of `labels_per_1k_words` and
   `effective_topics`** — all three treatments agree. No adjudication needed.
6. **The post-1860 ‖ contrasts on those last two are the exception, and are
   listed here rather than buried.** `labels_per_1k_words` on the legacy 15
   survives post-1860 ‖ under both controls (+2.20, +2.19) and **fails raw**
   (−0.24, p = 0.39); **honest reading: the controls**, so the finding stands,
   but it stands on five arms of six § and the note says so where it reports it.
   `effective_topics` on LLM level-2 is significant post-1860 ‖ only under
   `sotu_only` (+1.20, p = 0.042) and null under raw (p = 0.31) *and*
   genre-standardized (p = 0.71) — here the two controls disagree with **each
   other**, so the "prefer the controls" rule adjudicates nothing. That result is
   therefore reported as one significant arm out of three and is not treated as a
   post-1860 ‖ rise.
7. **The post-1860 ‖ block contrast and the 2010 era's rank disagree on
   `effective_topics`, and neither is wrong.** The block contrast is flat or
   negative while the 2010 era is the highest of the nine on 4 of 9 arms. The
   reconciliation — that the 1950/1980 trough drives the block contrast — is in
   sub-claim 1 and is not adjudicated away here, because the two statistics
   answer different questions and a reader needs both.

**Arm-dependence caveat §.** `sotu_only` and `genre_standardized` share **74.4%**
of paragraph mass corpus-wide and **93.1%** in the 1770 and 2010 eras (see "Sample
sizes"), so read the three arms as two — `raw`, and a genre-held-fixed reading —
and discount every "two of three" / "five of six" / "all three treatments" count
accordingly. The one place this does not bite is the register block, where all
three arms are at the bootstrap floor and the raw arm agrees.

---

## The honest headline

> **Presidential agendas broadened between Washington and Lincoln and have not
> broadened *significantly* since. What changed in the modern era is not the
> range of subjects or the depth of treatment — per 1,000 words presidents cover
> more issues than in any earlier era, once genre is held fixed — but the
> register: the formal record shed its procedural, hedged, departmental prose for
> shorter, plainer, far more evaluative speech.**

The original headline's three parts fare as follows: *"touch more subjects"* is
true but 19th-century; *"say less about each"* is false and reverses sign when
measured per unit of speech — though measured from 1860 ‖ rather than from the
founding block, labels *per paragraph* do fall on the legacy 15 under `raw`
(−0.260, p = 0.0005), so the per-paragraph claim is baseline-dependent rather than
simply dead; *"more speech off-policy"* is a real but modest effect — **+3.1 /
+3.0 points under the two genre controls**, +5.0 raw — visible on one labeler of
two **when measured from the founding block, and on both when measured from
1860 ‖** (the legacy labeler gives +0.105 raw at the bootstrap floor there, its
strongest pro-headline result anywhere in this note), with its modern level below
the founding era's. On the founding-block contrast it is the note's most
multiplicity-exposed surviving claim (p = 0.009–0.014); on the post-1860 ‖ one it
is at the bootstrap floor on all three arms.

Two things this headline is careful **not** to say. It does not say breadth
*narrowed*: the 2010 era is the highest of the nine on 4 of 9 taxonomy × arm
combinations and top-two on 8 of 9, and the one significant post-1860 ‖ decline
(LLM level-1) is a 1950s–1980s trough rather than a modern one. And "once genre
is held fixed" is one reading, not two: the two genre-controlled arms share
three-quarters of their paragraphs §.

---

## What would falsify this

The claims that survive are stated so they can be killed. Each of these outcomes
would overturn a conclusion above.

**Against "breadth plateaued after 1860":**
- A topic taxonomy with finer modern resolution (the corpus-native taxonomy has
  50 level-2 topics for 240 years; modern policy space is plausibly more
  differentiated than any single taxonomy can express) showing a significant
  post-1860 ‖ rise. The level-1 result already runs the other way, so this is a live
  possibility, and it is the single most likely way this note is wrong.
- Rarefaction at a much larger common budget (the 355-paragraph budget is set by
  the thinnest era and is coarse) reversing the post-1860 ‖ flatness.
- Any measure of breadth not based on label counts — e.g. embedding dispersion of
  paragraphs within a president — showing a modern rise.

**Against "depth did not fall":**
- A depth measure that is not a label count. Labels-per-1,000-words rises partly
  because the LLM assigns 1.46 topics per paragraph almost regardless of content.
  A measure of *sustained* treatment — consecutive-paragraph runs on one topic,
  or mean topic-run length within a speech — could fall even as label density
  rises. **This is the strongest untested rival to sub-claim 2's reversal**, and
  it is not measured here.
- Evidence that modern paragraph segmentation in the corpus is an artifact of
  transcription (speech-to-text line breaks) rather than authorship. That would
  make *both* the per-paragraph and per-1,000-word series suspect.

**Against "the register changed":**
- Marker rates are regex counts and are era-biased by vocabulary drift. If
  `mechanism`'s decline is driven by specific archaic words (`appropriat*`,
  `statute`) rather than by the concept, the finding is lexical, not rhetorical.
  A word-family-level replication (`word_families.py`) would settle this.
- If `fk_grade` and `words_per_sentence` decline is driven by punctuation
  conventions in 19th-century transcripts, the readability half collapses.
- If the effect disappears within-president (i.e. it is entirely composition
  across presidents rather than a trend any president participates in).

**Against everything:**
- A different era binning (20-year or 40-year bands) reversing a sign. Only 30-year
  bands were computed; the binning was inherited from `taxonomy.py::ERA_SPAN`, not
  chosen here, but it was also not varied.
- The LLM annotation pass being unreliable on 19th-century text specifically. The
  QA report (`notes/annotation-qa-v1.md`) should be read alongside this note; any
  era-varying label quality would masquerade as an era trend.

---

## Proposed confirmatory design (not run)

Everything above is exploratory. A genuine confirmatory test needs data that has
not been looked at. Proposed:

1. **Held-out slice.** The corpus is fully annotated, so a held-out *speech* slice
   must be carved and sealed before any further measurement. Proposal: withhold a
   random 20% of speeches, stratified by (era × speech_type), sealed by hash and
   recorded in a manifest, and re-derive nothing on it until the pre-registration
   is filed. Alternatively — and better — extend the corpus forward or sideways
   (e.g. Miller Center speeches not currently in `speeches.parquet`) so the
   held-out set is genuinely new text.

2. **Pre-register exactly three directional hypotheses**, each with its estimator,
   treatment arm and decision rule fixed in advance:
   - H1: `effective_topics` on `taxonomy_v1` level-2, `sotu_only`,
     `delta_modern_minus_postbellum` ‖ > 0. *Predicted outcome: FAIL* (this note
     finds +1.20, p = 0.042 on that arm alone and null on the other two).
   - H2: `labels_per_1k_words` on `taxonomy_v1` level-2, `sotu_only`,
     Spearman vs era > 0. *Predicted outcome: PASS.*
   - H3: `mechanism` per 10k, `sotu_only`, Spearman vs era < 0. *Predicted
     outcome: PASS.*

3. **Fixed analysis parameters**, declared before unsealing: 30-year eras,
   speech-clustered bootstrap, seed 20260721, Miller-Madow entropy, α = 0.05,
   Holm-Bonferroni over the three hypotheses (0.05/3 at the first step) —
   decided in advance, not after. **At least 20,000 bootstrap replicates**, not
   the 2,000 used here: a percentile p-value cannot go below 1/B, and 2,000 is
   too coarse to resolve a Holm-corrected threshold. That limit is not
   hypothetical — it is why the 444 tests in *this* note cannot be corrected at
   all (see "Multiplicity" above).

   Applying Holm to three future tests while running 444 uncorrected ones is an
   inconsistency a reader is entitled to point at. The defence is that the two
   are different objects: this note reports a whole exploratory table, arms that
   fail included, and correcting it would penalize the completeness. A
   confirmatory pass reports exactly three pre-committed decisions, which is
   precisely the setting a family-wise correction is for. The defence would stop
   working the moment this note started quoting individual p-values as evidence
   of a discovery, which is why the "Multiplicity" section names which claims
   would and would not survive.

4. **Pre-committed publication.** All three outcomes ship regardless of sign. The
   value of H1 is precisely that this note predicts it fails.

A confirmatory pass is only worth running if step 1 can produce genuinely unseen
text. Re-splitting the already-annotated corpus and calling the held-out part
"fresh" would be theater of the same kind this note is trying to avoid.

---

## Reproducing

From the **main repo root** (the `.venv` lives there; a git worktree has none, and
its editable-install `.pth` points back at the main repo's `src` — hence the
explicit `PYTHONPATH`):

```bash
PYTHONPATH=src arch -x86_64 .venv/bin/python -m presidential_profiles.register
```

Deterministic: same seed → byte-identical `data/register/trends.parquet`. Runs in
~6 seconds, makes no network calls, and writes nothing under
`data/llm_annotations/`.

Public API: `load_inputs`, `build_taxonomy_index`, `build_paragraph_frame`,
`build_speech_panel`, `measures_from_sums`, `effective_topics`,
`reference_genre_mix`, `genre_cells`, `trend_statistics`, `spearman_vs_position`,
`bootstrap_p_value`, `build_trends`, `rarefied_effective_topics`, `era_counts`,
`within_speech_icc`, `label_resolution_report`, `write_trends`.

Diagnostics that deliberately do **not** reach the parquet, because their spread
is not sampling uncertainty and would be misread beside a bootstrap CI:
`rarefied_effective_topics` (subsampling variation) and `within_speech_icc`.

`trends.parquet` columns: `unit` ∈ {year, era, trend}, `period`, `measure`,
`taxonomy` ∈ {legacy15, llm_level2, llm_level1, none}, `genre_treatment`,
`statistic` ∈ {level, spearman_vs_era, delta_modern_minus_early,
delta_modern_minus_postbellum ‖, delta_last_minus_first}, `value`, `ci_low`,
`ci_high`, `p_value`, `n_speeches`, `n_paragraphs`, `n_words`, `n_bootstrap`,
`n_bootstrap_valid`, `bootstrap_seed`.
