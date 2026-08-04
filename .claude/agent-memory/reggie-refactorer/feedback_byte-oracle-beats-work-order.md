---
name: byte-oracle-beats-work-order
description: When a task hands you a byte-reproducible artifact as a correctness oracle, trust it over any upstream claim of "output-neutral" — and verify neutrality across ALL treatment/slice dimensions
metadata:
  type: feedback
---

If a SIMPLIFY work order asserts a fix is "measured output-neutral" AND hands you a
byte-reproducible artifact sha as the oracle, run the oracle for **every** commissioned change
individually. Do not batch: regenerate after each edit so a moved sha names one culprit.

**Why:** On `issue-attention-over-time` (2026-07-21) the orchestrator work order commissioned
carry-forward defect #10 (`find_successor`'s filter 4 short-circuiting on an empty `legacy` set)
as a "safe change", stating the judge had measured it output-neutral and that
`topic_lifecycles.parquet` must still hash `825fbe66…` afterwards. It did not. The judge's
measurement had only covered the **raw** treatment and only the *classification* columns; the
table also carries `sotu` and `genre_standardized` rows. Applying the fix flipped
`Constitutional Union & Federalism` from `rename` to `unresolved_death` under `sotu` (losing
successor `Civil Service Reform & the Merit System`, r=−0.278) and blanked six
`cross_domain_candidate` diagnostics across all three treatments. Per the stop-and-report rule
the fix was reverted and escalated; the docstring was changed instead to record the deviation
honestly rather than keep promising "four independent filters".

**How to apply:**
- Diff artifacts *cell-by-cell across the full key* (here `level × treatment × topic`), not just
  the headline slice. A one-line diff script beats eyeballing a sha.
- "Judge measured it neutral" is a claim about a *sample* of the output space until you check it.
- When you revert a commissioned fix, still bank the value: make the code comment / docstring
  describe the ACTUAL behaviour and name the escalation, so the next stage does not re-derive it.
- Equally: `# pragma: no cover - defensive` is a claim, not proof. Same task's work order asked
  to delete an "unreachable" branch in `topic_lifecycle`; a 15-line probe proved it reachable
  (substantive years sitting at the outer edge of the window that made them substantive, so the
  evidence clamp excludes every live year and `idxmax` raises on an empty frame). Construct the
  witness before deleting; if you find one, leave the branch and rewrite the comment.
