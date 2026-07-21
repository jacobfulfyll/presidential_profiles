# Topic method comparison v1 — LLM vs CorEx vs embedding clusters

**Date:** 2026-07-21 · **Corpus:** 36,229 paragraphs / 1,057 speeches
**Artifacts:** `data/method_compositions.parquet`, `data/topic_display_names.json`,
`data/issues_meta.json` (coherence added)
**Code:** `src/presidential_profiles/topic_quality.py`, `src/presidential_profiles/triangulate.py`

---

## 0. Headline

**The pre-registered prediction failed.** It was predicted that the four anchored issues with
the lowest NPMI coherence — Immigration, Foreign policy, Infrastructure, Civil rights & race —
would show the *lowest* LLM↔CorEx agreement. Only one of the four (Infrastructure) is in the
observed bottom four. Immigration, the single **least coherent** anchored issue (NPMI −0.028),
has the **third-highest** agreement of all sixteen (κ = 0.533).

Across the 15 legacy issues, anchor coherence and cross-method agreement are **statistically
independent**:

| Predictor of agreement | vs Jaccard | vs κ | vs fill (ceiling-controlled) |
|---|---|---|---|
| Anchor NPMI coherence (**the prediction**) | ρ = −0.054, p = 0.850 | ρ = +0.118, p = 0.676 | ρ = −0.093, p = 0.742 |
| Crosswalk fan-out (topics per issue) | ρ = −0.282, p = 0.309 | ρ = −0.474, p = 0.074 ⚠ | **ρ = +0.022, p = 0.937** |
| **Base-rate divergence** D = \|log(n_LLM/n_CorEx)\| | **ρ = −0.739, p = 0.0016** ⚠ | **ρ = −0.761, p = 0.0010** ⚠ | ρ = −0.182, p = 0.516 |

⚠ **Neither uncontrolled column survives the ceiling control.** Jaccard is algebraically bounded
by the base-rate ratio (next subsection), and κ is separately depressed by the same marginal
imbalance — so the two left columns are two views of one confound, not two witnesses. The
right-hand column is the honest one. The fan-out figure ρ = −0.474 in particular is **withdrawn**
as a ceiling artifact; see §8 limitation 1 for the claim that survives in its place.

The prediction is not merely unsupported; the correlation is ~zero with the sign flipping
between the two metrics. Agreement instead **co-varies strongly with base-rate divergence** —
though, as the next subsection establishes, the two are partly linked by construction, so that
association is reported as exploratory rather than as an established instrument property.
The falsification itself is not in doubt and does not depend on any of it.

### ⚠ The base-rate finding is partly tautological — read this before citing it

**Jaccard is bounded above by the very quantity we correlate it against.** With a = n_LLM and
b = n_CorEx, since |A∩B| ≤ min(a,b) and |A∪B| ≥ max(a,b):

```
J = |A∩B| / |A∪B|  ≤  min(a,b)/max(a,b)  =  exp(−|log(a/b)|)  =  exp(−D)
```

This holds **deterministically, for every issue, before any data is examined**. A
monotone-decreasing rank association between D and J is therefore guaranteed *in part by
arithmetic*, not discovered. Verified on this data: the bound holds for all 15 issues,
`min/max` equals `exp(−D)` to 1.1e-16, and the ceiling is genuinely binding — the realised
fraction J/ceiling ranges from 0.252 (Education) to 0.626 (Money & banking), mean 0.433. **Some
of ρ = −0.739 is arithmetic, not evidence.**

**The control.** Correlating the *fill fraction* J/(min/max) — the share of the attainable
maximum actually realised — against D removes the tautological component:

| Test | ρ | p |
|---|---|---|
| D vs Jaccard (uncontrolled) | −0.739286 | 0.001636 |
| **D vs fill fraction (CONTROLLED)** | **−0.182143** | **0.515882** |

**The association does not survive the control.** |ρ| falls from 0.74 to 0.18 and p rises from
0.0016 to 0.52 — at n = 15 it is no longer distinguishable from noise. Whatever residual signal
exists is not demonstrable at this sample size.

**κ does not independently corroborate it.** The κ column above reads like a second, agreeing
witness and it is not one: κ is separately depressed by marginal imbalance — the well-known
*kappa paradox* / prevalence-and-bias effect — and that imbalance is itself a function of D. The
two columns are two views of one confound, not two pieces of evidence.

**Multiple comparisons.** The table is 3 predictors × 2 metrics = **6 tests**. Bonferroni at
0.05/6 = 0.0083. In the finding's favour: the uncontrolled p = 0.0016 **does survive** that
correction — omitting this would be the mirror-image error of overclaiming. But surviving a
correction for multiplicity is not the same as surviving the correction for tautology, and the
latter is the one that matters here.

**Status: post-hoc and exploratory.** This predictor was not pre-registered; it was found while
investigating why the pre-registered prediction failed. It **requires independent confirmation
on a held-out labeler pair or a different corpus** before being treated as a property of our
instruments. The §8 reading — that much of the apparent disagreement is a granularity mismatch
between a 50-topic and a 15-topic scheme — is the substantive claim worth carrying forward, and
it does not rest on the correlation coefficient.

*Why this warning exists:* a falsified prediction leaves a vacuum, and the first strong
correlation found afterwards is exactly the thing most likely to be over-promoted to fill it.
Flagging that is the same discipline that made publishing the falsification mandatory.

**Why the prediction was wrong (mechanism).** NPMI and agreement ask different questions.
NPMI asks *do these anchor words co-occur inside one paragraph?* Agreement asks *do the anchors
select the same paragraphs the LLM selects?* An anchor set can be dispersed yet highly
**precise**: Immigration's terms (`border`, `immigration`, `aliens`, `naturalization`) almost
never co-occur — hence NPMI −0.028 — but any *one* of them unambiguously marks an immigration
paragraph. Low coherence, high precision, high agreement. Immigration alone carries this
argument: it is the single least coherent issue and among the highest-agreeing, which is the
exact pairing the prediction forbids.

*(An earlier draft offered Education — highest-but-one NPMI, lowest agreement — as the converse
example. That is withdrawn: §2 establishes Education's agreement is a taxonomy-gap artifact, and
§8.3 forbids using it for instrument-quality readings, which is what that sentence did.)

Neither bail condition fired (§7).

---

## 1. What was built

Three labelers now cover the same 36,229 paragraphs, joined on the real `(doc_name, para_idx)`
key with `validate="one_to_one"` **and** a `_require_full_merge` row-count assertion on every
join (an inner join silently drops rows when key sets diverge; `validate` does not catch that).

| Arm | Unit | Vocabulary | Nature |
|---|---|---|---|
| **LLM** on discovered taxonomy | paragraph, multi-label | 50 level-2 topics | semantic, drift-robust |
| **CorEx** on frozen legacy 15 | paragraph, multi-label | 15 anchored + 7 free | lexical, keyword-anchored, **frozen** |
| **Embedding clusters** k=15 | paragraph, single-label | 15 clusters | unsupervised, no supplied vocabulary |

`data/method_compositions.parquet` — 85,617 rows, long format
(`doc_name, method, topic, share, n_paragraphs`). Long rather than wide because the three
methods have non-comparable vocabularies (50 / 16 / 15); a wide frame would be 81 mostly
meaningless columns. Downstream analyses get the promised one-line method switch:
`comps[comps.method == "llm_taxonomy"]`.

All three use the **same denominator** — every paragraph in the speech, including the 404 the
LLM left unlabeled — so the LLM arm is not silently advantaged by a smaller denominator.
Sanity: `embed_k15` shares sum to exactly 1.000 per speech (hard partition); `corex_legacy`
1.413 and `llm_taxonomy` 1.526 (both multi-label, correctly >1).

### Three decisions that move the numbers

**Era axis (`trends.ERAS`, 9 named eras).** Every per-era figure in this report is cut on
`trends.ERAS` — *The founding* (1789–1815), *Expansion*, *Civil War & Reconstruction*,
*The Gilded Age*, *Progressives & Depression*, *War & New Deal*, *The Cold War*, *Post-Cold War*,
*The present era* (2017–2026) — the corpus's named reporting axis, with boundaries on real events.
It is **not** `taxonomy.ERA_SPAN`'s equal-width 30-year bands: those are a *sampling* device whose
regularity is deliberate (per-era LLM taxonomy proposals must not inherit a periodization the
taxonomy was supposed to discover on its own), and as a reporting grid they fuse the Civil War with
the Gilded Age and the Depression with WWII. A per-issue × per-era table is a published axis that
`era-atlas` and `convergence-analysis` join on, so it takes the reporting grid; two competing era
axes would make every cross-task number unjoinable.

> ⚠ **Readers comparing against an earlier draft: the era grid changed.** An earlier version of
> this report cut §5 and §8.5 on the 30-year bands (1770, 1800, … 2010) and its per-era numbers are
> **not** comparable with the ones here. Everything era-*independent* — the §0/§2 falsification, the
> ceiling control and its ρ values, all coherence/NPMI figures, and `method_compositions.parquet` —
> is bit-for-bit unchanged by the switch, and was verified so rather than assumed.

**Projection (union / any-activation).** `crosswalk_v1.json` runs legacy→level-2, one-to-many,
and is **not a partition** — level-2 topics are reused across legacy issues. There is no reverse
index and inverting it is many-to-many. A paragraph counts for legacy issue *L* iff **any** of
its LLM topics is in *L*'s crosswalk row, so one paragraph can activate several issues at once,
matching CorEx's own multi-label behaviour. A different choice (e.g. argmax to a single issue)
would produce different agreement numbers.

**Name normalization.** `paragraph_annotations.topics` holds **58 distinct raw strings for 50
canonical level-2 names** — 8 title-case variants across 111 rows, concentrated in exactly the
war/foreign-policy topics this analysis cares about most. All names pass through
`taxonomy._norm_name()` and the post-normalization distinct count is asserted `== 50`. An
exact-match join would have dropped those rows and biased agreement *downward* precisely where
the prediction was being tested.

---

## 2. The prediction, tested

Agreement per issue, whole corpus, 404 LLM-unlabeled paragraphs excluded. **Bold** = one of the
four predicted-lowest issues.

| Issue | NPMI | n_LLM | n_CorEx | Jaccard | κ | Jaccard rank |
|---|---|---|---|---|---|---|
| Education | 0.372 | 3222 | 1566 | 0.123 | 0.170 | 1 (lowest) |
| Crime & justice | 0.223 | 3046 | 1940 | 0.191 | 0.272 | 2 |
| Energy & environment | 0.267 | 2568 | 1006 | 0.200 | 0.305 | 3 |
| **Infrastructure** | 0.089 | 1981 | 2989 | 0.202 | 0.289 | **4** |
| **Civil rights & race** | 0.091 | 3519 | 1620 | 0.211 | 0.305 | **5** |
| Religion & values | 0.204 | 3666 | 2169 | 0.221 | 0.310 | 6 |
| Economy & jobs | 0.192 | 3745 | 5720 | 0.267 | 0.337 | 7 |
| **Foreign policy** | 0.079 | 9456 | 5525 | 0.283 | 0.306 | **8** |
| Agriculture | 0.375 | 1134 | 1000 | 0.330 | 0.481 | 9 |
| Health care | 0.284 | 1195 | 1620 | 0.333 | 0.480 | 10 |
| Trade & tariffs | 0.183 | 2733 | 3284 | 0.342 | 0.465 | 11 |
| **Immigration** | **−0.028** | 788 | 971 | 0.374 | 0.533 | **12** |
| War & military | 0.187 | 6099 | 8041 | 0.390 | 0.456 | 13 |
| Taxes & budget | 0.263 | 2984 | 3903 | 0.428 | 0.558 | 14 |
| Money & banking | 0.300 | 1508 | 2030 | 0.465 | 0.616 | 15 (highest) |
| *Security & peace* (Discovered 5) | 0.195 | 11196 | 7389 | 0.388 | 0.414 | — |

- Predicted bottom-4: Immigration, Foreign policy, Infrastructure, Civil rights & race
- **Observed bottom-4 (Jaccard):** Education, Crime & justice, Energy & environment, Infrastructure
- **Observed bottom-4 (κ):** Education, Crime & justice, Infrastructure, Energy & environment
- **Overlap: 1 of 4** on both metrics. Mean Jaccard rank of the predicted four is 7.25 vs 8.27
  for the rest — no meaningful separation, and driven entirely by Infrastructure.

**However, "1 of 4" overstates the miss under this report's own rule.** §2's next subsection and
§8.3 both say Education's last place is a taxonomy gap that must be excluded from any
instrument-quality reading — and the prediction test *is* such a reading. Dropping Education
(n = 14):

| | bottom-4 (Jaccard) | overlap |
|---|---|---|
| All 15 issues | Education, Crime & justice, Energy & environment, Infrastructure | **1 / 4** |
| **Education excluded** (the report's own rule) | Crime & justice, Energy & environment, **Infrastructure, Civil rights & race** | **2 / 4** |

Identical on κ. ρ(NPMI, Jaccard) moves from −0.054 (p = 0.850) to **+0.130 (p = 0.659)** — still
no association, and now the *wrong sign* for the prediction. **The verdict is unchanged, but the
report's most-cited number doubles under its own exclusion rule, so both are stated.**

### The falsification survives the clean metric too

§0 argues Jaccard and κ are both contaminated by base-rate divergence, which invites the
objection: *you tested the prediction on contaminated metrics.* It survives the ceiling-controlled
metric as well:

| Metric | overlap | mean rank, predicted 4 | mean rank, rest |
|---|---|---|---|
| Jaccard | 1 / 4 | 7.25 | 8.27 |
| κ | 1 / 4 | 6.75 | 8.45 |
| **fill (ceiling-controlled)** | **1 / 4** | **8.25** | **7.91** |

ρ(NPMI, fill) = −0.093, p = 0.742. Under the clean metric the predicted four rank *above*
average — the opposite of the prediction — so the falsification does not depend on the
contaminated statistics at all.

**Verdict: the prediction does not hold.** Jaccard is reported alongside κ because Jaccard is
computed over the two paragraph *sets* (|both| / |either|) and ignores the vast agreed-negative
mass that would make any two labelers look near-identical on a rare issue; κ keeps the negatives
but corrects for chance. They rank the issues nearly identically here, so the conclusion does
not depend on the metric choice.

### Education is contaminated — an important caveat on the bottom-4

Education's last-place finish is **not** evidence about the labelers. The frozen 50-topic
taxonomy contains **no education topic at all** (no level-2 name matches
educat/school/univers/science/research). The crosswalk therefore projects `Education` from three
only-tangentially-related topics:

> `civil rights, voting rights & discrimination` · `prosperity, jobs & the middle class` ·
> `welfare, poverty & economic opportunity`

That is a **taxonomy gap**, not a disagreement — and it fully explains the 2.06× base-rate
divergence (n_LLM 3222 vs n_CorEx 1566) that drags its agreement to the floor. Recorded as a
discovered issue; `taxonomy_v1.json` is frozen and was not touched.

---

## 3. Anchored-coherence caveat table

`embed_topics.py:21-29` prohibits applying a coherence *threshold* to anchored issues. It does
not prohibit *computing* the score, and the numbers are needed for this table — so all 22 CorEx
topics are scored and **only the 7 discovered ones are ever gated**. That asymmetry is enforced
in code (`topic_quality.classify_discovered` skips anchored issues) and flagged in comments at
each site, because it is the single most inviting place for a future reader to "fix" a non-bug.

Scores below are on the **identical yardstick** as `paragraph_clusters_meta.json` —
`embed_topics._fit_vectorizer` + `embed_topics._npmi`, imported, not reimplemented. They
reproduce the previously published anchored figures to three decimals (Immigration −0.028,
Foreign policy 0.079, Infrastructure 0.089, Civil rights 0.091), confirming the yardstick is
genuinely shared.

### One prior figure did not reproduce — reconciled

`task.md`'s Verified Facts publish **Discovered 3 at −0.031**; this report gets **−0.047**.
Discovered 5 reproduced exactly (0.195), so a single counter-example in a section arguing the
yardstick is shared cannot go unmentioned.

**Cause identified and confirmed.** The prior figures predate the contraction fragments
(`ve`, `ll`, `re`, `don`, `didn`, …) being added to `topics.EXTRA_STOP`. Re-scoring with those
fragments removed from the stop list — i.e. reconstructing the older vocabulary — reproduces
**both** prior figures exactly:

| Stop list | Discovered 3 | Discovered 5 |
|---|---|---|
| Current (fragments stopped) | **−0.0468** | +0.1950 |
| Without contraction fragments (prior state) | **−0.0307** ≈ published −0.031 | +0.1950 = published 0.195 |

The divergence is confined to Discovered 3 because it is the only topic whose top terms *are*
contraction fragments: under the old vocabulary all 10 were scoreable, today only 7 are, and the
3 that dropped out were among its most distinctive. Discovered 5 contains no fragments, so it is
invariant — which is precisely why one reproduced and the other did not.

**Term-count truncation is ruled out, and the term count is JOINTLY pinned at 10.**
`issues_meta.json` stores 12 top words per topic while the shared yardstick scores the top 10, so
the competing hypothesis is "the prior figures were computed on the untruncated 12". Scanning
n = 4…12 on the reconstructed pre-fragment vocabulary, **n = 10 is the only value at which both
published figures land simultaneously** (D3 → −0.031, D5 → 0.195):

| n terms | Discovered 3 | Discovered 5 | both match? |
|---|---|---|---|
| 7 | +0.045 | +0.1954 → "0.195" | no (D3 far off) |
| 8 | −0.023 | +0.191 | no |
| **10** | **−0.0307 → "−0.031"** | **+0.1950 → "0.195"** | **yes — unique** |
| 11 | −0.022 | +0.1959 → "0.196" | no |
| 12 | −0.008 | +0.188 | no |

Neither figure pins it alone — D5 at **7** terms is 0.1954, which also publishes as "0.195", so
D5 by itself excludes only 12. The **joint** fit is what identifies 10. And n = 11, the one value
whose D3 sits nearest −0.031, is cleanly excluded by the other arm: **D5 at 11 terms is 0.1959,
which rounds to 0.196, not 0.195.** No appeal to coincidence is needed.

**Both values sit below the zero floor, so the `noise` classification is unchanged either way.**

*(An earlier draft additionally claimed this instability was "direct empirical support for gate
clause (b)". Withdrawn as a non-sequitur: clause (b) is about term **coverage**, whereas the
instability shown here is across term **counts at fixed coverage**. The pre-fragment column
refutes the link outright — at full 10/10 coverage D3 is just as unstable, +0.183 at 4 terms to
−0.031 at 10 to −0.008 at 12. Clause (b) never fired, so nothing depends on it.)*

| Anchored issue | NPMI | Reading |
|---|---|---|
| Agriculture | +0.375 | tight, single-era vocabulary |
| Education | +0.372 | tight — yet lowest agreement (§2) |
| Money & banking | +0.300 | tight |
| Health care | +0.284 | tight |
| Energy & environment | +0.267 | tight |
| Taxes & budget | +0.263 | tight |
| Crime & justice | +0.223 | tight |
| Religion & values | +0.204 | tight |
| Economy & jobs | +0.192 | moderate |
| War & military | +0.187 | moderate |
| Trade & tariffs | +0.183 | moderate |
| **Civil rights & race** | **+0.091** | spans slavery-era ↔ segregation-era wording **by design** |
| **Infrastructure** | **+0.089** | spans `railroad` ↔ `broadband` **by design** |
| **Foreign policy** | **+0.079** | spans two centuries of diplomatic vocabulary **by design** |
| **Immigration** | **−0.028** | maximally dispersed anchors — **and 3rd-highest agreement** |

The four bolded issues score low **because the taxonomy is doing its job**: anchor sets
deliberately span vocabulary that never co-occurs in one paragraph, so a president can be scored
on the same axis across 240 years. Thresholding them would delete the taxonomy's whole point.
§2 now adds a second, independent reason not to read these as quality scores: they do not
predict cross-method agreement either.

**Sanity check on the metric itself** (the real theme should beat word soup):

| Probe | NPMI |
|---|---|
| Real theme — Discovered 5 top terms | **+0.195** |
| Scrambled — 10 random vocabulary words | −0.154 |
| Scrambled — one top term from each of 10 issues | −0.015 |

Reproduction (the two scrambled probes are seeded, not illustrative):

```python
import json, random, pandas as pd
from presidential_profiles import topic_quality as tq
tw = json.loads(open("data/issues_meta.json").read())["topic_words"]
issues_ = json.loads(open("data/issues_meta.json").read())["issues"]
random.seed(0)                                   # <- the pin
vocab = sorted({w for ws in tw.values() for w in ws})
probes = {"real": tw["Discovered 5"],
          "random10": random.sample(vocab, 10),
          "one_per_issue": [tw[i][0] for i in issues_[:10]]}
tq.compute_coherence(probes, pd.read_parquet("data/paragraphs.parquet")["text"])
```

---

## 4. Discovered-cluster coherence and naming

**Gate, pre-registered before any score was computed.** A discovered topic is `noise` iff
either: (a) NPMI ≤ **0** — not a tuned cutoff but NPMI's own null, the point of exact
statistical independence, where a topic's defining terms co-occur no more than chance; or
(b) fewer than **half** its top terms survive into the shared content vocabulary — because
`topics.EXTRA_STOP` already strips contraction fragments (`ve`, `ll`, `don`, `didn`) and
honorifics (`mr`), which is exactly what defines the sludge topics. Without clause (b) those
terms would be silently dropped and the topic scored on its handful of incidental content words,
**inflating the coherence of the worst topics**.

*Transparency note:* clause (b) never fired. The worst coverage was Discovered 4 at 8/10 and
Discovered 3 at 7/10, both above the 50% bar. It was a real guard against a real hazard, and it
turned out not to be load-bearing. Reported rather than quietly removed.

| Column | NPMI | Terms scored | Status | Display name | Surfaced |
|---|---|---|---|---|---|
| Discovered 1 | +0.129 | 10/10 | coherent | Fiscal Administration & Appropriations | no |
| Discovered 2 | +0.142 | 10/10 | coherent | Executive Departments & Official Reports | no |
| **Discovered 3** | **−0.047** | 7/10 | **noise** | *(none — deliberately unnamed)* | **never** |
| Discovered 4 | +0.159 | 8/10 | coherent | Debate & Interview Exchanges | no |
| Discovered 5 | +0.195 | 10/10 | coherent | **Security & peace** | **yes** |
| Discovered 6 | +0.140 | 10/10 | coherent | Territory, Sovereignty & Constitutional Powers | no |
| Discovered 7 | +0.110 | 10/10 | coherent | Law, Duty & Public Obligation | no |

Names authored **in-session** by `claude-opus-4-8` from each topic's top c-TF-IDF terms and its
coherence score, stamped with model, date, gate parameters and the evidence used in
`data/topic_display_names.json`. **No Anthropic API call was made** — a paid batch pass for
seven short labels would have been pure waste.

### `status` and `surface` are separate on purpose

`status` is the pre-registered quantitative gate and nothing else. `surface` is an **editorial**
decision — *is this a policy issue a reader should see beside the curated 15?* — recorded with a
written rationale per topic. Keeping them apart stops editorial judgment from being laundered
through a number.

Discovered 4 is the clearest case: NPMI +0.159 is genuinely coherent (`mr`, `senator`,
`governor`, `kennedy`, `sir` really do co-occur), but it is a **televised-debate transcript
artifact**, not something a president cared about. It gets a name; it does not get surfaced.
Marking it "noise" would have been smuggling a judgment call in under a quantitative label.

**`surfaced` remains exactly `["Discovered 5"]`** — the site's behaviour is unchanged and the
wiring change is verifiably behaviour-preserving (`display_issues(issues) == issues +
["Discovered 5"]` → `True`).

**Recommendation (not acted on):** Discovered 6 — *Territory, Sovereignty & Constitutional
Powers* (`constitution`, `territory`, `powers`, `mexico`, `spain`; NPMI +0.140) is a
substantively real domain the curated 15 never named, and is the strongest candidate for
promotion. Surfacing it requires an anchor-term list and adds a page to the live site — beyond
this task's display-name-wiring boundary.

There are **two** `KeyError` sites, not one: `issues_site.py:260` and `profiles.py:397/400` both
index `anchors_all[name]` / `anchors[name]`. Because the names file is meant to be hand-edited,
`topic_quality.validate_surfaced()` now raises a named `UnanchoredTopicError` naming the fix
whenever a surfaced topic lacks anchors. It is enforced on the **write** path (`build_names`),
not on read — raising at read time would turn a hand-edit typo into a hard site-build failure,
and `display_issues`'s job is to degrade rather than crash.

---

## 5. Rename vs death — the vocabulary-drift casualties

The distinction only a two-method comparison can make: **a topic CorEx says died but the LLM
says is alive has been renamed, not abandoned.** Each arm is normalized to its own peak era,
because the two methods have different base rates and only their *trajectories* are comparable.

Flagged where CorEx fell below 25% of its own peak while the LLM held above 50% of its own.
Thresholds select what to look at; they do not create the effect, and the full unfiltered
trajectory table is available from `triangulate.rename_vs_death()`.

**Support check.** `n` below is the **era's total paragraph count**, not this issue's support
within it — and `rename_vs_death` gates only on `MIN_ERA_PARAGRAPHS` (era size), which is the
very filter §8.5 criticises. Since this section elsewhere mandates a support floor before any
per-era reading, that floor was checked separately here rather than assumed: **all 9 flagged rows
clear it**, with `min(n_LLM, n_CorEx)` ranging from 28 (Energy & environment / Expansion) to 517.
That is currently a property of the data, not of the code — see the discovered issue on
`rename_candidates` lacking an issue-support floor.

| Issue | Era | n (era) | CorEx share | LLM share | CorEx rel | LLM rel | gap |
|---|---|---|---|---|---|---|---|
| Security & peace | The founding | 922 | 0.069 | 0.319 | 0.148 | 0.822 | **+0.674** |
| Security & peace | The Gilded Age | 3576 | 0.062 | 0.288 | 0.132 | 0.742 | +0.610 |
| Security & peace | Expansion | 3703 | 0.044 | 0.258 | 0.093 | 0.664 | +0.571 |
| Agriculture | Expansion | 3703 | 0.012 | 0.045 | 0.138 | 0.654 | +0.516 |
| Energy & environment | Expansion | 3703 | 0.008 | 0.094 | 0.162 | 0.643 | +0.481 |
| Infrastructure | The Cold War | 9039 | 0.018 | 0.052 | 0.109 | 0.516 | +0.407 |
| Infrastructure | The present era | 3896 | 0.033 | 0.058 | 0.194 | 0.574 | +0.380 |
| Security & peace | Progressives & Depression | 4714 | 0.110 | 0.237 | 0.234 | 0.610 | +0.376 |
| Foreign policy | The present era | 3896 | 0.060 | 0.202 | 0.236 | 0.555 | +0.319 |

### Manually confirmed: Infrastructure, the Cold War

CorEx peaks in **The founding** (0.168 — roads and canals; note this peak is only weakly
anchor-driven, with just 10 of its 155 CorEx-positive paragraphs containing any Infrastructure
anchor word — era totals `roads` 9, `canals` 5, `highways` 1, and **zero** occurrences of
`railroad`, which postdates 1815 entirely) and collapses to 0.018
in **The Cold War**, **10.9% of its own peak**, while the LLM holds at **51.6%** of its peak
(which falls in Progressives & Depression). Reading the 429 paragraphs the LLM marks
Infrastructure and CorEx misses in that era confirms the mechanism outright — Eisenhower on
housing shortages, on "a strong Federal program in the field of resource development", on "our
continental transport system" and the **St. Lawrence Seaway**. All unmistakably infrastructure —
*that* is the informative half of the evidence, and it is what the manual read confirms. The
concern persisted; the words changed. CorEx partially recovers in The present era (0.033) as
`infrastructure` and `broadband` finally enter *presidential* vocabulary — the anchor list is
frozen and always contained both; what moved is the corpus (`infrastructure` 2 → 39 → 63 and
`broadband` 0 → 4 → 3 across The Cold War / Post-Cold War / The present era).

**One clause from an earlier draft is withdrawn as uninformative.** It noted that none of the
429 paragraphs contains `railroad`, `canals`, `highways` or `bridges`. True, but near-tautological
at `anchor_strength = 6`: corpus-wide only **16 of 773** anchor-bearing paragraphs are
CorEx-negative for Infrastructure (2.1%), and in The Cold War it is **0 of 68**. So "CorEx missed it"
implies "contains no anchor word" ~98% of the time *regardless of what vocabulary is doing* — the
clause confirms the model is working, not that drift occurred. (Anchor matching here is
prefix-based, which is a **superset** of what CorEx actually sees — its vectorizer uses
`token_pattern=r"[a-zA-Z][a-zA-Z]+"` with no stemming, so a paragraph containing only
`roadstead` — a sheltered anchorage for ships — never put `roads` in front of the model.
Prefix matching is therefore a conservative bound that makes the tautology look *weaker* than
it is; exact matching gives **3 of 754**, i.e. 99.6%. The Cold War is 0 of 68 either way, and the
conclusion holds under both. The two corpus-wide counts are era-independent and are unchanged by
the move to the named era grid; only the per-era figure was recomputed.)

### The mirror case: CorEx *over*-reports Money & banking

Money & banking runs the other way, and is the strongest reminder that neither arm is ground
truth. Jaccard by era: 0.677 (Expansion, the attention peak for both arms; the Jaccard maximum
is actually The Gilded Age at 0.797) → 0.424 (Progressives & Depression) → **0.073**
(Post-Cold War). Both arms agree attention peaked in **Expansion** (1816–1849), but by Post-Cold
War CorEx stays at 21.8% of its peak while the LLM falls to 11.9%. *(The present era reads a still
sharper 0.026, but rests on `n_LLM = 5, n_CorEx = 74` — below the support floor, and excluded here
for the reason §5's era-trend retraction gives. The Post-Cold War cell used above clears it at
`n_LLM = 87, n_CorEx = 133`.)*

**Two relative shares are symmetric between "CorEx over-reports" and "the LLM under-reports",
and both are happening.** The discriminating evidence is the paragraphs themselves, not the
ratio:

- **CorEx false positives are real.** Reading the 118 CorEx-only paragraphs in Post-Cold War turns
  up `"landed at Omaha and Gold"` (D-Day beaches), `"the search for a silver bullet"`, `"pennies on
  the dollar"`, and `World Bank` / `European Bank for Reconstruction and Development`. Modern
  speech is saturated with `gold`, `silver`, `dollar` and `bank` in paragraphs that are not about
  monetary policy — the mirror image of drift blindness, and why the uncorrelated-errors argument
  cuts both ways. (Every one of these exemplars was re-located inside the Post-Cold War band; the
  qualitative finding did not depend on the old grid.)
- **But the LLM arm is also structurally handicapped here**, exactly as Education is in §2.
  Money & banking's entire crosswalk row is two pre-modern topics — `coinage, currency & specie`
  and `national bank & banking crises` — and the frozen taxonomy has **no modern monetary-policy
  or central-banking topic at all**. The LLM labels on those same 118 paragraphs are
  `taxes, budget deficits & federal spending` (28), `prosperity, jobs & the middle class` (24) and
  `health care, medicare & social security` (23): modern economic topics that simply do not
  crosswalk to Money & banking.

So the modern collapse of the LLM arm is **partly a taxonomy gap, not only a CorEx defect** —
the same caveat §2 applies to Education and §8 instructs readers to apply to any
instrument-quality reading. The false-positive finding survives on the paragraph evidence; the
share ratio alone would not have established it.

### Caveat: some "rename" signal is crosswalk fan-out, not euphemism

**In The founding, *every one* of Security & peace's 294 LLM-positive paragraphs is reached
through just three crosswalk topics — two of them broad foreign-policy ones.** The disjoint
split: 144 via `military preparedness, armed forces & veterans` alone, 143 via
`treaties, diplomacy & international arbitration` alone, 6 via both, and 1 via
`immigration & border security` — 144 + 143 + 6 + 1 = **294**, the whole set. (Per-topic totals
are 150 / 149 / 1, which sum to 300 rather than 294 because the field is multi-label and the
6 dual-labelled paragraphs are counted twice.) These arrive via the union projection (§1); they
are not euphemisms for anything. That is the projection behaving exactly as specified.

Security & peace is the issue examined here because it takes 4 of the 9 flagged rows. **Do not
read that as fan-out predicting flagging** — across all 16 issues it does not: ρ(fan-out,
n_flagged) = +0.163, p = 0.546, and the two equal-or-wider crosswalk rows contribute 1 flagged row
(Foreign policy, fan-out 8) and 0 (War & military, fan-out 7). The existential claim above is
carried entirely by the paragraph-level composition, which is exact. Any general "fan-out inflates
LLM breadth" reading carries §8 limitation 1's caveat with it — ρ(fan-out, divergence) = +0.497 at
**p = 0.060, short of conventional significance at n = 15**, and post-hoc.
**Infrastructure (fan-out 2) is the cleaner drift case for that reason, which is why it is the one
confirmed manually above.**

*(On the previous 30-year grid this caveat was illustrated with a flagged Civil rights & race row
in the earliest band. Under the named eras that row is a **near miss** rather than a flagged
candidate — `corex_rel` 0.097 but `llm_rel` 0.469, just under the 0.50 persistence bar — so it is
no longer in the table above. The mechanism is unchanged and, if anything, starker: 139 of its 144
LLM-positive paragraphs in The founding arrive through `indian affairs, removal & allotment` alone,
via a 6-topic crosswalk row.)*

### Era trend — support-confounded; do NOT read as instrument convergence

An earlier draft of this report described mean Jaccard as rising "steadily" across eras and
attributed it to CorEx's modern anchors converging with the LLM as the corpus approaches the
anchors' own era. **That reading was wrong on three counts and is retracted.**

| Era | mean J, all 16 cells | mean J, `min(n_LLM, n_CorEx) ≥ 25` | cells surviving |
|---|---|---|---|
| The founding | 0.195 | **0.335 — highest of any era** | 8 |
| Expansion | 0.172 | 0.224 | 12 |
| Civil War & Reconstruction | 0.208 | 0.222 | 15 |
| The Gilded Age | 0.234 | 0.249 | 15 |
| Progressives & Depression | 0.273 | 0.288 | 15 |
| War & New Deal | 0.260 | 0.303 | 12 |
| The Cold War | 0.292 | 0.292 | 16 |
| Post-Cold War | 0.317 | 0.330 | 15 |
| The present era | 0.272 | 0.288 | 15 |

1. **It is not monotone.** It *falls* from The founding to Expansion (0.195 → 0.172), dips again
   at War & New Deal (0.273 → 0.260), and falls at the end (0.317 → 0.272). "Steadily" was simply
   inaccurate.
2. **The mechanism is contradicted by its own series.** If convergence tracked proximity to the
   anchors' era, The present era would be the maximum. It is Post-Cold War (0.317), and under the
   support floor it is The founding (0.335) — not the latest era on either column. *(An earlier
   draft added "and The present era sits below Progressives & Depression". That is dropped: the
   gap is 0.272032 vs 0.272999, and it reverses under the ≥25 floor this same section mandates —
   0.288436 vs 0.287546, a tie at published precision. The argmax comparison carries point 2 on
   its own.)*
3. **Decisive: the early-era dip is an averaging artifact.** 7 era cells report Jaccard exactly
   0.000, and **6 of those have an entirely empty arm.** Health care has zero LLM support in four
   consecutive early eras (The founding, Expansion, Civil War & Reconstruction, The Gilded Age)
   and Immigration in two more (Expansion, and again in War & New Deal — so 5 of the 6 empty-arm
   cells are early) — issues whose LLM topics did not exist yet. Under a support floor
   the ordering **inverts at the low end**: The founding goes from second-*lowest* (0.195) to the
   *highest*-agreement era of all nine (0.335).

The published trend is substantially "how many anachronistic issues have zero LLM support in
early eras", not instrument convergence. **No era-trend instrument property is claimed.** Any
era-comparative analysis must apply a support floor first; `data/method_agreement.parquet`
ships `min_support`, `low_support` and `empty_arm` per cell so that is a one-line filter.

---

## 6. The 404 unlabeled paragraphs — policy and sensitivity

1.12% of paragraphs (404) carry `topics == []`. **Primary analysis excludes them.** An LLM
abstention is not the same event as "the LLM judged this issue absent"; counting it as an
all-negative row would let it inflate κ's agreed-negative mass across all 16 issues at once
(and Jaccard of two empty sets is conventionally 1, which would inflate agreement outright).

Sensitivity, both ways, all 16 issues:

| | mean \|Δ Jaccard\| | mean \|Δ κ\| | max \|Δ\| |
|---|---|---|---|
| exclude vs include | **0.0006** | **0.0009** | 0.005 (Health care) |

**The choice does not matter.** No issue moves by more than 0.005 on either metric and no
ranking changes. Documented because it was a live risk, not because it turned out to be one.

---

## 7. Bail conditions — neither fired

| Condition | Result |
|---|---|
| All discovered clusters score incoherent → stop | **Did not fire.** 6 of 7 coherent; only Discovered 3 is noise. |
| LLM↔CorEx agreement near zero **everywhere** → stop (implies a crosswalk/join bug) | **Did not fire.** All 16 issues positive on both metrics: Jaccard 0.123–0.465, κ 0.170–0.616. |

The second is the one that mattered, since Corrections 9 and 10 flagged two plausible silent-bug
paths (title-case join drops, crosswalk inversion). Uniformly positive, substantial agreement
across every issue and era is evidence the join and projection are sound.

---

## 8. Limitations

1. **The union projection inflates LLM breadth** for wide-crosswalk issues (Foreign policy 8
   topics, War & military 7, Civil rights 6). An earlier draft supported this with
   ρ(fan-out, κ) = −0.474 (p = 0.074), calling it "suggestive but not dismissible."
   **That figure is withdrawn as a ceiling artifact.** Fan-out inflates `n_LLM` — this
   limitation's own premise — and `n_LLM` is half the base-rate ratio that §0 proves
   algebraically bounds Jaccard, so fan-out acts through precisely the tautological channel.
   Worse, §0 states that κ is *not* independent corroboration because it is separately depressed
   by marginal imbalance; quoting the κ figure as the load-bearing one contradicted that
   directly. Under the same control §0 applies to base-rate divergence:

   | test | ρ | p |
   |---|---|---|
   | fan-out vs Jaccard | −0.2817 | 0.309 |
   | fan-out vs κ *(the withdrawn figure)* | −0.4744 | 0.074 |
   | **fan-out vs fill (ceiling-controlled)** | **+0.0222** | **0.937** |
   | fan-out vs base-rate divergence | +0.4966 | 0.060 |

   It vanishes completely under the control. **The substantive claim survives on different
   evidence**: fan-out correlates with base-rate divergence at ρ = +0.497, p = 0.060 — like the
   figure it replaces, short of conventional significance at n = 15, but *not* ceiling-bounded,
   which is the entire point of the substitution. It directly supports "wide crosswalk rows
   inflate LLM breadth", which is limitation 1's actual claim. What does not survive is any
   claim that fan-out predicts *agreement*.
2. **Base-rate divergence co-varies with agreement** (ρ = −0.74) — but **Jaccard is
   algebraically bounded by that same divergence (J ≤ e^−D), and the association does not
   survive the ceiling control (ρ = −0.182, p = 0.516); see the ⚠ subsection in §0.** The claim
   that survives is qualitative and does not rest on the coefficient: much of what reads as
   "disagreement" is a granularity mismatch between a 50-topic and a 15-topic scheme, not a
   substantive conflict. Treat as post-hoc and exploratory pending independent confirmation.
3. **Education is a taxonomy gap, not a measurement** (§2). Its bottom-4 placement should be
   excluded from any instrument-quality reading.
4. **Coherence ≠ interest.** Discovered 4 is coherent and worthless as an issue. NPMI measures
   whether terms co-occur, nothing more.
5. **Thin cells are flagged, not handled.** κ is NaN where a labeler is constant over a slice,
   and eras with fewer than `MIN_ERA_PARAGRAPHS = 100` paragraphs are skipped. But that gate is
   on **era size, not issue support** — a 9,039-paragraph era clears it while an individual issue
   has near-zero positives inside it. Of the 144 era cells, **13 have
   `min(n_LLM, n_CorEx) < 10` and 21 have < 25**; 7 report Jaccard exactly 0.000, 6 of those
   with an entirely empty arm. Those cells are **retained and flagged**, not dropped:
   `data/method_agreement.parquet` carries `min_support`, `low_support` (< 25) and `empty_arm`.
   Any per-era reading must filter on them first — §5's retracted era trend is what happens if
   you do not.

---

## 9. Reproduction

```python
from presidential_profiles import triangulate, topic_quality, issues
topic_quality.build_names()          # coherence + names file  ($0, no API call)
issues.refresh_meta_coherence()      # coherence into issues_meta.json, no CorEx refit
r = triangulate.run()                # writes both parquets; returns every table

r["drivers"]["spearman"]             # §0 rho/p table; every predictor also has a
                                     # *_vs_fill_CONTROLLED entry, plus
                                     # fanout_vs_divergence for §8 limitation 1
r["drivers"]["drivers"]              # per-issue divergence, ceiling, fill fraction
r["drivers"]["sensitivity"]          # the §6 unlabeled-paragraph deltas
r["agreement"]                       # per issue x era; also data/method_agreement.parquet
```

Every headline number in this report is produced by
`triangulate.agreement_drivers()` — including the ceiling control — so none of §0 or §6 exists
only as prose. Artifacts written:

| Path | Contents |
|---|---|
| `data/method_compositions.parquet` | 85,617 rows — per-speech shares, all three methods |
| `data/method_agreement.parquet` | 160 rows — per-issue agreement, 16 overall + 144 per-era (AC2), each carrying `min_support` / `low_support` / `empty_arm` |
| `data/topic_display_names.json` | coherence, status, display names, provenance |
| `data/issues_meta.json` | `coherence` (22 topics) + `discovered_status` (7) added |

`issues.refresh_meta_coherence()` deliberately avoids `build_issues()`, which would refit CorEx
over the real 36k × 25k matrix; the persisted top terms are scored in place instead. Anchors and
`anchor_strength` were **not touched** — the legacy arm is the bias-independent control, and
changing it would make before/after uninterpretable.
