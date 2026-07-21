---
name: verification-attention
description: How to verify attention.py + notes/attention-findings-v1.md ($0/4s/byte-identical; exhaustive --quotes sweep; blockquote-vs-corpus recipe and its em-dash trap)
metadata:
  type: reference
---

Verifying `presidential_profiles.attention` and its published research note. Complements
[[verification-workflow]] (Rosetta venv + worktree PYTHONPATH discipline) and
[[verification-register-note]] (same note-vs-parquet genre of task).

**Rebuild is $0, ~4s, and byte-deterministic across processes.** `-m presidential_profiles.attention`
writes `data/attention/topic_lifecycles.parquet` (201 rows x 38 cols = 3 treatments x (50 level-2 +
17 level-1)). Run it **twice in separate processes** — within-process stability is not determinism.
stdout is identical too, so `diff run1.txt run2.txt` is a second oracle. Unlike `register`, tests
DO read this parquet (`TestCommittedArtifacts`), so never leave it regenerated-but-different.

**The sha is the correctness oracle for any `attention.py` edit.** Comment-only edits must leave
`825fbe66…f0e` unchanged. Prove it by regenerating rather than by reasoning.

**Verifying `_require_full_merge` on `exemplar_quotes` — do the exhaustive sweep, not 3 spot checks.**
The guard can only fire if an assignment key is missing from `paragraphs.parquet`; both tables are
co-derived from the same 36,229 rows, so it never fires. Prove it with 50 topics x 5 year windows
in one script (250 calls, seconds) plus adversarial inputs: nonexistent topic, a level-1 name passed
where level-2 is expected, and an inverted window (`year_lo > year_hi`) — all three return `[]`
cleanly because the pool is empty and `0 == 0` satisfies the guard.

**BLOCKQUOTE-VS-CORPUS RECIPE (the highest-value check on a research note).** Parse `>`-prefixed
blocks straight out of the .md rather than retyping them, then substring-match against
`paragraphs.parquet` joined to `speeches.parquet` for `president`/`title`. Verify three things per
quote: verbatim presence, cited year == paragraph year, cited president == speech president. Also
print the paragraph's topic labels — that catches a quote filed under the wrong topic.

**Two traps that produce 100% false failures in that recipe:**
1. The note wraps quotes in literal `"`; strip them or every match fails.
2. **The corpus uses `--` where the note renders an em dash.** Normalize `--` -> `—` -> space on
   BOTH sides. Without this, 2 of 17 quotes look fabricated when they are exact.
Get 0/N or a couple of weird misses? Suspect the normalizer before accusing the note.

**Prose-attribution errors are invisible to numeric checks — group by document.** The note's
"Washington's first two annual messages … 17 of 32 paragraphs in 1790" has the count exactly right
but the attribution wrong: 14 of the 17 come from the *Talk to the Chiefs and Counselors of the
Seneca Nation*, and only 3 from the annual messages. Whenever a note explains a peak by naming a
genre or document, `groupby(title).size()` the topic's paragraphs in that year and check.

**Double-rounding recurs in this project's notes** — a second independent instance after
[[verification-register-note]]. Here 4 rate cells published one increment high (2.2456 -> "2.3",
2.3483 -> "2.4", 1.5475 -> "1.6", 1.0480 -> "1.1"): the author rounded to 2 dp then to 1 dp.
Signature: the discrepant value's second decimal is always 4 or 5. Treat as one systematic cosmetic
finding, and check it deliberately — it is now the expected defect class here, not a surprise.

**Counting claims are where notes drift, not the tabulated cells.** Every one of ~60 tabulated
share/CI cells reproduced exactly, while four *counts/ranges in prose* were wrong ("seven other
topics" = 9; "eight topics" = 10; "4.3-5.2%" = 3.06-5.21%; "least divergent" = 5th-least). Spend
verification effort on sentences containing a bare number, not on the tables.

**`run_ground_truth` returns 5 checks (2 True / 2 False / 1 not_testable), not 6.** The note's §1
table has 6 rows because the anachronism check is printed as its own block by `print_checks`, not
by `run_ground_truth`. Do not report a missing check when the tallies disagree — locate the 6th
first.
