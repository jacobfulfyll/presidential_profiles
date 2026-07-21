---
name: feedback-research-note-numbers
description: Correcting a research findings note means recomputing every number from the artifact — copying between sections and find-and-replace both produce new errors
metadata:
  type: feedback
---

When correcting a `notes/*-findings-v1.md` research report in presidential_profiles, **re-derive
every number from the artifact rather than copying it from elsewhere in the note**, and never
apply a supplied correction as a blind find-and-replace.

**Why:** On the `issue-attention-over-time` SYNC-DOCS pass (2026-07-21) seven published claims did
not reproduce. All seven were invisible to a reader of the note alone — they surfaced only by
recomputing from `topic_lifecycles.parquet` and the corpus. Two of the seven were miscounts, and
sweeping the *unlisted* numbers turned up four more errors the work order had not found (see
[[project-attention-findings-sweep]]).

**How to apply:**
- A supplied "correct" value is a hypothesis to confirm, not a fact to paste. Every value in that
  work order did confirm, but confirming cost minutes and would have caught a bad one.
- Watch for a correction that is really two conflated metrics. "Slavery is the least divergent at
  21x" was wrong as stated, yet Slavery genuinely *was* the closest call on a different quantity
  (distance to the collapse threshold). Swapping the number would have shipped a new falsehood.
  Split the metrics and state both.
- Never "fix" a class of cells uniformly. Four rate cells were double-rounded one increment high;
  a fifth cell with the same published value (1.6) was already correct. Recompute each.
- Prefer measuring a fragility claim over reasoning about it. Monkeypatching a module constant and
  re-running the classifier ($0, no writes) beat deducing the answer, and produced a sharper
  statement than the work order's.
- Include an explicit "what was and was not verified" section. This was requested and is worth
  making standard for these reports.
