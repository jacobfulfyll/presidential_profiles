---
name: pipeline-review-heuristics
description: Two recurring REVIEW-stage hazards in this pipeline — TASKS.md merge data-loss, and bail-conditions surfaced-but-not-enforced in run()
metadata:
  type: project
---

Two hazards worth checking on every code-workflow REVIEW in this repo. Both
found on `derive-corpus-taxonomy` (2026-07-19); neither was in the stage log.

**1. TASKS.md/HISTORY.md two-dot-diff FALSE POSITIVE (I got this wrong once — don't repeat).**
The pipeline commits "discovered-issues"/"pickup" `meta:` commits to the MAIN
repo's `TASKS.md`/`HISTORY.md` on master, out-of-band, WHILE a task branch is
open. The branch itself never touches those files. So `git diff master..HEAD --
TASKS.md` shows master's newer meta-additions as apparent branch-side DELETIONS
— a pure two-dot-diff artifact (master ref moved past the branch's base). A
3-way merge / `git merge --squash` keeps master's version because only ONE side
changed the file; nothing is deleted.
- **What I did wrong:** on `derive-corpus-taxonomy` I flagged this as a BLOCKER
  (silent loss of `annotation-pass-label-validation`). It was void — the session-
  start git-status snapshot showed a STALE master ref; master had actually
  advanced (meta commits `15a2735`, `7496513`), and `git diff master..HEAD` was
  comparing the moved master against the branch's older base.
- **How to apply (correct verification):** before flagging any TASKS.md/HISTORY.md
  deletion, run `git log --oneline master..HEAD -- TASKS.md`. EMPTY ⇒ the branch
  never committed it ⇒ NOT a finding (two-dot artifact). Only if the branch has a
  commit touching the file is it real. Confirm the merged result with
  `git merge-tree --write-tree master HEAD` then `git cat-file -p <tree>:TASKS.md`
  — if the items survive there, there is no data loss. Never trust the session-
  start git-status `master` SHA; re-run `git rev-parse master` yourself.

**2. Bail conditions are surfaced, not enforced, in run().** taxonomy.py `run()`
derives + prints + writes artifacts UNCONDITIONALLY. The plan's four bail
conditions (coverage <90% after one revision, >60 level-2, missing crosswalk
issue, anachronism) are only reported via prints/provenance — none raises or
skips the write. `_atomic_write_json` overwrites the committed artifact path
directly, so a failed re-run clobbers the good file on the working tree.
- **How to apply:** in this codebase, gate ENFORCEMENT is a manual git-commit
  step, not code. That's a deliberate pattern (all four bails behave the same),
  so flag it as MAJOR (re-run safety / plan-compliance gap) — not a per-gate
  bug, and don't expect a raise. Recommend one consolidated post-derivation
  gate-check that raises before writing.

**4. A defensive fallback that handles the TOTAL failure of a filter but not the PARTIAL one.**
Found on `issue-attention-over-time` (2026-07-21). `attention.py:topic_lifecycle` clamps a
substantive year-span to the years the topic actually has paragraphs, then guards `if live.empty:`
(the total miss — a whole task backlog item was written about proving that branch live). But the
partial miss is unguarded: when the clamp's new `first_year` lands in a HOLE of the mask while
other mask years survive, `live` is non-empty and `live.loc[live["year"] == first_year, ...]
.iloc[0]` raises `IndexError`. Reproduced end-to-end through the public API in ~15 lines
(`attention_curves` -> `extract_lifecycles`) with a sparse-early-years corpus.
- **How to apply:** whenever you see a guard for "the filter removed EVERYTHING", ask what happens
  when it removed only the boundary element the code is about to look up by value. The fix is
  usually to recompute the boundary FROM the filtered frame instead of asserting the pre-filter
  boundary survived. Prior stages had reasoned hard about this exact branch and still missed the
  neighbouring case — the presence of a well-argued guard is a signal to look harder, not less.

**3. annotate.py: resume-filtered requests but FULL persisted requests_index.json
(found on `run-llm-annotation-pass`, 2026-07-20, REVIEW).** `cmd_submit` trims
the REQUESTS list by the coverage-based resume filter (`_already_ingested`) and
the sealed filter, but calls `_write_run(run_id, requests, index)` with the
UNTRIMMED `index` (build_requests emits index entries for every speech, resume
only trims `requests`). So requests_index.json persists metadata for speeches
that were never submitted that round. Harmless for most rounds — BUT in a
`--chunk-size` round every speech's index entry carries a `chunk` key, so the
amendment-#2 provenance derivations that scan the persisted index
(`cmd_ingest`'s `chunked_docs`, `_all_chunk_escalated_docs()` for converged QA)
report EVERY speech as chunk-escalated instead of the handful of true
stragglers. Provenance-integrity defect (a HARD value here) in an
acceptance-criterion deliverable (the converged QA report names chunk-escalated
speeches). Empirically confirmed: build+resume-trim+`_write_run` then re-derive.
- **Fix:** trim `index` to the submitted cids right before `_write_run`
  (`index = {r["custom_id"]: index[r["custom_id"]] for r in requests}`). Dry-run
  is unaffected (no resume filter). Fixing at the persist site fixes both
  derivation sites.
- **How to apply / heuristic:** whenever a submit path filters a request list by
  a resume/coverage gate, check that the persisted sidecar index (and anything
  derived from it downstream — scope, provenance, amendment lists) is filtered
  to the SAME set. The tests here all build requests_index.json directly via the
  `forge_run` fixture, so no test drives real `cmd_submit` resume-skip + chunk —
  the gap slips the suite. Trace the real multi-round sequence, don't trust the
  green suite for cross-round state.
- **Resolution (round 4, fb5e795):** fixed exactly as prescribed (one-line trim
  before `_write_run`). The regression test that actually bites drives the REAL
  `cmd_submit` (not `forge_run`): `patch_anthropic(create_batch=...)` fake client
  + a written `paragraph_annotations.parquet` that makes `_already_ingested`
  resume-skip docA, submit docB with `--chunk-size 2`, then assert persisted
  index == docB-only and `_all_chunk_escalated_docs()==["docB"]`. That
  real-cmd_submit + written-parquet-resume pattern is the way to close cross-round
  state gaps the forge fixtures can't reach. Paired MAJOR (unguarded ingest
  payload shapes) fixed by wrapping the per-result body in
  `except (KeyError,TypeError,ValueError) -> "malformed_payload"`; the round-2
  quarantine `continue`s escape it cleanly (continue is control flow, not an
  exception), usage/counts accumulate before the try so billed tokens still count.
