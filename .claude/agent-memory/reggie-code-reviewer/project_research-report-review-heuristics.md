---
name: research-report-review-heuristics
description: High-yield checks for REVIEW on presidential_profiles analysis tasks that ship a notes/*.md research claim plus data/*.parquet artifacts
metadata:
  type: project
---

Found on `combativeness-over-time` (2026-07-21, combat.py + notes/combativeness-findings-v1.md).
All three survived IMPLEMENT 9.05, 96% mutation, 99% coverage, and a VERIFY-APP that checked
60+ report figures against the parquets — because none of them is a wrong number.

**1. An n-floor on a WEIGHTED estimator must bind on the thinnest weight-bearing stratum,
not the cell total.** `combat.py` suppresses a CI below 5 speech clusters — but for the
`genre_standardized` treatment it counts `n_speeches` as the SUM over genre strata, while
`MIN_CELL_SPEECHES=3` admits 3-speech strata carrying up to 49% of the renormalized weight.
Result: rows marked `ci_status="ok"` / `n_speeches=52` whose dominant contributor is the same
3 speeches the `sotu_only` row suppresses and the report bolds "do not read as estimates."
- **How to apply:** for any direct-standardization / poststratification / weighted-average
  estimator, compute `weight x cluster_count` per stratum and compare against the module's own
  floor. Also check that raising the stratum floor to match the cluster floor doesn't reverse a
  stated conclusion — here it flipped §5.4's "Civil War edges ahead of the founding."
- Related: two floors in one file with different values and no shared anchor is the
  [[threshold-anchoring-value]] smell even when both are individually documented.

**2. Cluster unit vs. the unit the CLAIM is about.** The bootstrap clustered on speeches
(justified by within-speech ICC 0.17) and never examined within-*president* correlation. The
headline cell was 10 speeches / **2 presidents**, and the claim was about "the presidency."
- **How to apply:** for any clustered/hierarchical interval, ask what the sentence generalizes
  over. If the claim's unit is coarser than the resampling unit, the interval understates and
  the report must say so. Count the coarser unit — it is often startlingly small.

**3. Numbers in the report prose that no shipped code produces.** Whole load-bearing
subsections (entity type-mix shares, per-president splits, per-genre decompositions,
robustness recounts) were hand-computed in-session and appear nowhere in `combat.py` or the
eight parquets. Every one I recomputed was CORRECT — that is exactly why it slips every prior
stage. It still breaks the repo's reproducibility value ([[llm-annotation-provenance-layer]]).
- **How to apply:** grep the report for every numeral, then ask "which function emits this?"
  If the answer is "none," it is a provenance finding regardless of correctness.

**4. Sweep the Limitations section against the headline sections.** §8 limitations that
contradict §1-§5 conclusions are the highest-value report findings and prior stages never look
for them (they verify numbers, not entailment). Real hits here: a coverage threshold in §8 that
silently indicts the headline era; "residual era-bias is not eliminated" in §8 vs "era anchoring
is sound" in the verdict line; "a third administration could move it" in §8 vs "regime shift,
not a personality" in §4.

**5. Selective-statistic reuse — THE dominant defect class in this repo's notes.** Six
instances found across `breadth-depth-register` rounds 1-2 (three by review, two by the
corrective pass unaided, one by review in round 2 *after* the pass had corrected four).
When a note volunteers a statistic as a self-critical caveat against a claim it *rejects*,
grep for that same statistic on every claim it *advances* — and vice versa.
- **The mechanical version that finds it every time:** build the full
  `(measure x taxonomy x treatment x statistic)` grid from the parquet, then grep the note for
  each measure name. Any cell that is computed, significant, and never printed is a finding.
  `df[df.unit=='trend'].groupby(['measure','taxonomy','statistic','genre_treatment'])` — 5 min.
- Round-2 hit: sub-claim 2 killed `labels_per_paragraph` using Spearman + `modern − early`
  only; the omitted post hoc block contrast is −0.260 at the bootstrap FLOOR on the arm the
  note calls weak (it prints the smaller −0.145, p=0.001, and files it under "marginal"),
  and the controls do not flip sign there, falsifying "the sign flips the moment genre is
  controlled." **A corrective pass fixes the instances it looks for; it does not fix the class.**
- **Sibling check — period ranks.** Any "flat / declining since <era>" verdict built on a
  BLOCK contrast: print the full series and rank the last period. Here the "significantly falling"
  taxonomy had its most recent era as the corpus MAXIMUM, the decline being a mid-century dip. The
  note applied exactly this rank disclosure to the sub-claim it was killing and not to the one it
  was advancing. **Asymmetry of applied rigor is the finding, not a wrong number.**
- **Sibling check — baseline switching mid-argument.** A rebuttal computed off a *different*
  baseline than the result it rebuts. Bonus section: `proposal_vs_values` falls post-1860, and
  the rebuttal ("both stance shares rise") is `modern − early`; post-1860 `proposal_share` on
  raw is −0.011, p=0.52. Check the rebuttal uses the contrast being rebutted.

- **Round-3 refinement — the mechanical sweep does NOT exhaust the class.** A pass that
  enumerates all 444 cells and regex-matches each value against the note's prose still leaves
  three residues, and all three are found only by reading:
  (i) **a matcher artifact** — rendering a value at 0 decimals makes `0.7833` match a bare
  `1` in the region, marking a genuinely unprinted cell "printed". Drop the 0-dp rendering
  and re-run; that alone surfaced `nrc_fear`'s Spearman (+0.78/+0.73, p≈0.028, significant on
  both controls, discussed nowhere).
  (ii) **a printed number carrying an overstated interpretation** — the sweep can only see
  whether the digits appear, never what the sentence does with them. Here sub-claim 3's
  post-1860 contrast is printed at the floor on all three arms *and* the note calls the same
  claim "the weakest-evidenced of the surviving claims" / "most multiplicity-exposed",
  and files it in the marginal multiplicity table only. **Exact mirror of the round-2
  blocker, in the opposite direction** — the pass had just written the governing rule
  ("marginal on one contrast is not marginal on the measure") and applied it only to the
  claim it was killing.
  (iii) **a prose claim with no artifact behind it at all** — `nrc_fear` "declared as
  rising" / "the opposite sign to its declared direction" asserted 3×, while the note's own
  Pre-declaration table lists neither `nrc_fear` nor `nrc_hope`. Diff the declaration table
  against every "declared"/"pre-declared" grep hit; the sweep is blind to this by design.

**5c. RE-REVIEW: when a pass answers an omission finding with DECLARED SCOPE RULES, audit
the rules, not the count.** Each rule is a falsifiable claim about the artifact. On
`breadth-depth-register` round 3, four rules justified 49 unprinted cells and three were
partly false: "the two unprinted columns are the same size to within the width of the printed
intervals" (false for `fk_grade`: 2.47 gap vs a 1.02-wide printed CI); "where an endpoint
contrast disagrees with a block contrast — <3 measures named> — it is tabulated in full"
(enumeration incomplete: `effective_topics` is the only measure in the table with two
*significant* contrasts of opposite sign, and `labels_per_1k_words` has 5/6 endpoint cells
significant and unprinted); and a blanket "everything else significant is reported".
The one rule that held was the strongest-sounding one (monotone-transform Spearmans identical
digit-for-digit — verified 0.0 max abs diff on all 9 arms; note the *p-values* still differ
slightly for the plug-in twin, which proves it is not literally a monotone transform).
- **How to apply:** for each rule, write the one-line pandas check that falsifies it. Also
  check `set(rule-covered cells) ⊇ set(significant unprinted cells)` — the gap is the finding.

**5b. RE-REVIEW: a new disclosure leaves orphans on the summary surfaces.** When a pass adds a
counter-datum deep in a section, check it propagated to (a) the summary verdict table, (b) the
adjudication/"which reading is honest" section, (c) the headline paragraph, (d) the multiplicity
/ robustness tables. On round 2 the legacy-labeler post-1860 agreement (+0.105, p at floor) was
added to sub-claim 3 and reached none of the four. Same shape as the stale-number sweep in #12.

**6. Verify the note's claims ABOUT ITSELF, with grep.** "It is flagged as post hoc everywhere it
appears" — `grep -in "post hoc"` returned 2 lines; the statistic appeared unlabelled in 6 more,
including the table producing the headline verdict. Same for "all arms reported", "every input
checked". These are checkable in one command and prior stages never run it.
- **Round-2 refinement:** when the fix ships a *grep recipe* as proof, run the recipe AND hunt
  for phrasings the recipe cannot match. 48/48 markers were correct; one prose restatement
  ("Measured from the 1860/1890/1920 block") matched neither the marker nor the recipe.
- **Also grep the note's blanket completeness claims** ("reports every arm it computed", "should
  not have significant undiscussed series sitting in its own artifact"). These convert a
  selective-reporting judgment call into a falsified self-claim — much harder to argue with.

**7. Count the hypothesis tests.** `df[df.unit=='trend'].p_value.notna().sum()` (or equivalent).
444 tests / 50 in the [0.004, 0.05) band / zero multiplicity disclosure — while the note's own
*proposed confirmatory design* planned a Holm correction for 3 future tests. Silence in the
exploratory half + correction in the confirmatory half is the tell.
- **The good fix to accept:** "disclose, don't correct" is defensible for an exploratory note
  that publishes its nulls, *if* it also states the resolution limit. Here 0.05/444 = 1.13e-4
  sits below the bootstrap floor 1/B = 5e-4, so no test can be *shown* to clear a family-wise
  threshold at any effect size. Verify that arithmetic (`444/0.05` → B ≳ 8,880) — it is the
  single most useful sentence in the section.
- Then check the disclosure's own two lists for the same asymmetry as #5: is the strongest
  *pro-headline* floor-level result in the "would not be excluded" table? Here it was in neither.

**8. "N of M treatment arms survive" — check the arms aren't the same data.** Genre-standardized
placed 74% of paragraph mass (93% at the endpoint eras) on the very speeches `sotu_only` uses
exclusively — `reference_genre_mix` prints the weight. The note disclosed only the endpoint case
while counting arms everywhere. Read the reference/weight vector before believing an arm count.
- Recompute per-era: the shared share is `w_sotu / sum(w_included)`, so cell-dropping can only
  *raise* it. "Three-quarters in every era" is then a conservative lower bound, not an average.

**9. The orchestrator's note-vs-parquet harness only covers parquet-backed claims.** Tables from
diagnostic functions that deliberately stay OUT of the artifact (`rarefied_effective_topics`,
`within_speech_icc`) are unverified by construction — recompute them yourself. Both were exact
here, but the rarefaction table contrasted raw ENDPOINTS while supporting a genre-controlled
BLOCK conclusion; recomputing the block contrast on the same output fixed it and made the
argument stronger. Cheapest high-value 5 minutes in the review.

**10. A two-sided empirical claim in a docstring where only ONE side is test-pinned — check the
unpinned side (highest-yield check on `issue-attention-over-time`, 2026-07-21).** `attention.py`'s
pre-registration docstring said the under-powered guard "flags NOTHING at level 2 under `raw` ...
It fires routinely under `sotu`, ... many claimed deaths cannot be checked inside annual messages
alone." A test pinned the `raw` half (`test_no_topic_is_under_powered_at_level_2_under_the_raw_
treatment`); nothing pinned the `sotu` half, which was FALSE — 1 of 50 topics, and 15 of 17 raw
deaths still die under `sotu`. Survived five gated stages including two comment-truthfulness
sweeps.
- **How to apply:** in modules with a load-bearing method/pre-registration docstring, extract every
  sentence asserting a fact about the module's own OUTPUT and recompute it from the artifact. The
  half that reads as rhetorical reinforcement ("which is the point: ...") is the half nobody pinned.
  A false claim here is a [[llm-annotation-provenance-layer]] provenance defect even though it moves
  no number, and a docs stage that is code-frozen cannot fix it — say who owns it.

**11. A published column whose aggregation across a many-to-many relation is undocumented.**
`corex_decline_ratio` is the MAX over a topic's legacy crosswalk parents (correct for the
"persists if ANY parent persists" rule, and test-pinned) but neither the function's `Returns:`
nor the module docstring says "max". Worse, the docstring states a ratio > 1 means the parent
"never tracked this topic and its verdict carries no information" — and the code then lets exactly
those parents supply the persist verdict and become the reported ratio. The report's divergence
multiples are computed off this column.
- **How to apply:** for any inverted many-to-many join, find where N parents collapse to 1 number
  and check the reduction is named in the docstring. Then check every caveat the docstring states
  about individual members is actually applied by the reduction.

**12. RE-REVIEW: verify a threshold fix is outcome-INSENSITIVE, not just outcome-correct.** When a
fix introduces a new cutoff to resolve a finding, the obvious objection is "you moved the arbitrary
threshold." Answer it empirically: print the sorted distribution of the quantity the threshold cuts.
On `combativeness-over-time` round 2, thin-stratum weights were {0, .024, .029, .042, .064, .314,
.359, .549, .744} — an empty gap from 0.064 to 0.314, so **any** cutoff in that range gives the
identical partition and the published result is provably insensitive to the 0.25 chosen. That turns
"arbitrary" into "irrelevant" in one command. Also diff old-vs-new parquets on a key join and count
changed cells — a good fix changes exactly the cells named and no point estimates.
- **Sibling check:** a new two-tier policy usually ships one tier that never fires. Run
  `--cov-branch --cov-report=term-missing` and check the inert tier is at least *tested*; it will
  be the uncovered line, and prior-stage mutation scores predate it.
- **When a fix INTERPRETS your finding rather than following it literally, re-derive the
  alternative.** M1's "a band applies if an interval existed" beat my literal "if a bound moved":
  a measured half-width of exactly 0.0 is a real component, and a `ci_lo` already at 0.0 gets
  clipped so the bound cannot move. Drive all three shapes (normal / exact-zero / absent band).
- **Stale-number sweep after a report retraction.** Withdrawing a subsection leaves orphans: a
  corrected figure in §5.7 (9.5% -> 10.1%) that §8.9 still quotes at the old value across an
  explicit cross-reference, and an ordinal heading ("A fifth contributor") stranded by a new
  "four independent lines" intro. `grep -n` the old number and every ordinal word.

**13. Accept a pass that adds evidence AGAINST its own skepticism — but check the verdict line.**
On `breadth-depth-register` round 2 the pass volunteered that the legacy labeler *agrees* with the
LLM labeler post-1860 (+0.105, p at floor) — a point for the headline it had killed. That is
genuine falsification discipline, not drift, and the test is mechanical: (a) recompute it,
(b) confirm the section's verdict still rests on the arms that were n.s. before (0.063 / 0.087
here — unchanged), (c) confirm no summary line was upgraded. All three held. Reward it; the
finding is that the disclosure did not propagate (see 5b), not that it was made.

**Adjudicating a pre-registered bail condition that fails.** The Civil War enemy-naming bail
did fail on its face and the report argued the premise was mis-specified. That is a legitimate
adjudication *here* because the rescuing hypothesis was written into `.pipeline/<task>/CONTEXT.md`
at PICKUP (before IMPLEMENT), named its own test, used an INDEPENDENT data field
(`paragraph_entities.type`, scored separately from the flag), and stated the stop condition.
- **How to apply:** read CONTEXT.md's timestamps/ordering before judging "post-hoc rescue."
  Check the rescuing evidence comes from a field the failing measure did not produce. Watch for
  a post-hoc falsifier list ("what would have fired it") being *presented* as pre-registered —
  flag the framing, not the reasoning, when the quantities really could have gone either way.
