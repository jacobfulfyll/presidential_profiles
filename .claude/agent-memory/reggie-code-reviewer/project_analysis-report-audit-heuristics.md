---
name: analysis-report-audit-heuristics
description: How to review published analysis reports in presidential_profiles — the recurring "correct conclusion, invalid justification" defect and the four recomputations that catch it
metadata:
  type: project
---

This repo publishes findings as `notes/*-report-v1.md`. Its recurring defect class is
**a correct conclusion supported by an argument that does not establish it**. Reading
never catches these; only recomputation does. Four instances shipped past four
separate gates in `topic-method-comparison` alone (round-1 ceiling tautology, a
quality-check purity test that varied the wrong axis, a backwards §3 term-count
argument, and a non-monotone-curve coincidence).

**Why:** falsification is the user's stated hard value (`notes/convergence-investigation-direction.md` §2).
A report that states a true conclusion for a false reason is the precise failure the
value exists to prevent, and it is invisible to number-checking — every published
figure in that task reproduced exactly while four arguments were still invalid.

**How to apply — run these four checks on every analysis report, always by executing code:**

1. **Does the predictor bound the outcome by construction?** Jaccard ≤ min/max = exp(−|log base-rate ratio|).
   Anything that inflates one arm's positives (crosswalk fan-out, projection width)
   moves agreement through that same ceiling. Correlate against the *fill fraction*
   (metric ÷ ceiling), not the raw metric. Check this for **every** predictor in the
   table, not just the one the report already flagged.
2. **Is a set-membership observation a definition of the set?** e.g. "the paragraphs
   CorEx missed contain none of its anchor words" — with `anchor_strength=6`, only 3 of
   754 anchor-bearing paragraphs are CorEx-negative, so that is ~guaranteed and
   confirms nothing. The qualitative reading of the paragraphs is the real evidence.
3. **Do unweighted means over sparse cells manufacture the trend?** Per-era means
   average cells with zero positive support (an issue that did not exist yet scores
   Jaccard 0.000). Re-mean with a `min(n_llm, n_corex) >= 25` floor before believing any
   era trend — in this corpus that reversed the published direction.
4. **Is a limitation stated in one section silently dropped where the claim is reused?**
   Grep the report for its own caveats (taxonomy gap, ceiling control, fan-out
   inflation) and check each is applied at every downstream use, especially in
   §Limitations and in "mirror case" / mechanism paragraphs.

**Axis-change check (5th recomputation, learned when the era grid moved).** When a
reporting axis is re-cut, every *number* gets recomputed programmatically but every
**descriptive gloss attached to an argmax/peak cell is hand-written and does not move
with it**. Peak-era glosses are the highest-yield place to look: locate each
"X peaks in <cell> (<value> — <prose about what is in it>)" and verify the prose
against the *new* cell's actual corpus content, not the old one's. In
`topic-method-comparison` the Infrastructure CorEx peak moved from the 1800–1829 band
to *The founding* (1789–1815) and kept the gloss "canals, roads, early railroads" —
zero `railroad` tokens exist in that era, and only 10 of its 155 CorEx-positive
paragraphs contain any Infrastructure anchor word.

**Calibration, learned on the re-review:**

- **Check the direction of an error before escalating it.** When a disputed figure errs
  *against* the author's own claim (e.g. a looser match method that makes a tautology
  look weaker in a paragraph retracting that tautology), it is safe to ship — downgrade
  to a note about the stated method, do not gate on it.
- **A claim about a model's construction must use the model's own tokenizer.**
  `issues.py` builds CorEx on `token_pattern=r"[a-zA-Z][a-zA-Z]+"` with no stemming, so
  exact `\bterm\b` matching is right; `profiles.py`'s `\bterm\w*` prefix form exists for
  *highlighting word families to a reader* and is not a precedent for reproducing model
  behaviour. Prefix matching on `roads` silently pulls in `roadstead` (a nautical
  anchorage) and `roadside`.
- **Fixing a two-path divergence: change the path that is wrong, not the one that is
  right.** Coherence must be fitted on the same population as
  `paragraph_clusters_meta.json` (raw `paragraphs.parquet`, 36,229) or the
  "identical yardstick" property dies. `refresh_meta_coherence` already matches it;
  `build_issues` is the deviant. Adding the speeches merge to the correct path breaks
  synthetic-frame tests; removing it from the deviant one is two lines and touches
  nothing tested.
- **An observability field is only a substitute for a fix if something compares it.**
  A recorded count nothing asserts against is documentation, not a guard.

See [[paragraph-key-and-tests]] for how to execute against the real artifacts
(worktree `PYTHONPATH`, x86_64 venv) before trusting any recomputation.
