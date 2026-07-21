---
name: project-attention-findings-sweep
description: attention.py / issue-attention-over-time — recurring error classes found sweeping the findings note, and the read-only recompute recipe
metadata:
  type: project
---

`attention.py` + `data/attention/topic_lifecycles.parquet` + `notes/attention-findings-v1.md`
landed 2026-07-21 (task `issue-attention-over-time`). Corrected at SYNC-DOCS against a blocking
work order plus a full sweep. See [[feedback-research-note-numbers]] for the general rule.

**Error classes that recurred — check these first on any similar report:**
- **Counts of a curated set stated as counts of a measured set.** "Three lifecycle classes flip on
  treatment" was true only under a narrow reading (3 topics gain a *death* they lack under `raw`);
  14 of 50 change class at all. "Four absolute births" included one topic with 35 prior paragraphs
  — and the note's own anachronism table already listed its stray 1847 paragraph, so the report
  contradicted itself two sections apart. Cross-check a claim against other sections of the same
  document, not just against the data.
- **A window label that is off by one year** because the author read the wrong row of a curve.
- **Sorted tables that are not sorted by the quantity they claim to rank**, and that omit members.
- **Rates double-rounded** (2dp then 1dp) landing an increment high.
- **Multi-label counting ambiguity.** "26 of 35 paragraphs, rest scattered across N topics" — N
  differs depending on whether co-labels on the 26 count. State which you counted.
- **A negative result stated with the wrong mechanism.** The note said a successor candidate was
  rejected because it "does not become substantive again until much later" — the opposite of the
  truth. It was rejected because it became substantive *too early* (first year 1873 < peak 1884,
  filter 2); the two topics co-occurred for 10 of the 24 decline-window years. A rejection is a
  claim; trace **which filter actually fired** rather than narrating a plausible reason. Both of
  §6.2's rejections turned out to be the same co-occurrence failure.
- **A number reused across two different spans.** "The 43-year gap" was the candidate topic's own
  internal silence (1956–1998), not the 92-year span the sentence was about — and the note stated
  92 correctly two sections away. When the same figure appears near two relationships, check which
  one it measures.
- **Mode/peak claims are grain-dependent.** "The 1830s is the second mode" was false at decade
  grain (the 1820s is higher) though the curve is genuinely bimodal at era grain. Any modality,
  peak, or trough claim must name its grain — decade, era, and smoothed-year give different answers
  on the same curve.
- **Era-boundary arithmetic.** A trough between two eras is `next_era_start - prior_era_end - 1`;
  "70-year trough" was 68 (1878–1945). Recompute from `trends.ERAS`, do not estimate.

**Read-only recompute recipe ($0, no writes, ~2 min):**
`A.load_inputs()` then `A.attention_curves(...)` for the 6 (level, treatment) pairs; pickle both to
a scratchpad since `load_inputs` is the slow part. `A.bootstrap_era_shares(...)` reproduces every
era share + CI table. Threshold-fragility claims: monkeypatch `A.COREX_COLLAPSE_RATIO` and re-run
`A.classify_deaths` — comparing `rename_class` series shows a ~33-row NaN noise floor (non-dying
topics), so count changes *above* that floor.

To check a successor rejection, evaluate `find_successor`'s four filters one at a time against the
lifecycle row (same level-1 domain / candidate `first_year` inside `[peak_year, last_year]` /
outlives / shares an inverted-crosswalk parent) plus the `_pearson` sign gate. Substantive years
come from `A.substantive_mask(curve)`, not from the parquet.

**Gotchas:**
- `corex_decline_ratio` is a **max** over crosswalk parents, so the per-parent minimum must be read
  from the `corex_parent_ratios` string column, not from that scalar.
- `max_gap_years` is a topic's **own** internal silence. It is never the span between two topics.
- **5 of 17 level-1 domains are singletons**, so same-domain successor search is empty by
  construction for them: `Indian & Tribal Affairs`, `Ceremonial & Commemorative Address`,
  `Procedural & Administrative`, `Faith & National Values`, `Partisan & Media Combat`.

**Worktree note:** this worktree's `README.md` had no `combat.py` section even though the
`combativeness-over-time` pass added one — the worktree is frozen at branch creation. Never verify
`README.md`/`TASKS.md`/`HISTORY.md` state from inside a worktree.
