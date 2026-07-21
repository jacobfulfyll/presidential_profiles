---
name: verification-findings-notes
description: How to re-verify a findings note (notes/*-findings-v1.md) against its parquet artifact in presidential_profiles — the error classes that survive multiple correction passes
metadata:
  type: reference
---

Re-deriving a published findings note against its artifact. Learned on
`notes/attention-findings-v1.md`, where 11 errors survived 6 gated stages and 2 more
survived the correction pass.

**Recipe** (see [[verification-workflow]] for the interpreter):
```
cd <worktree>
PYTHONPATH=$PWD/src arch -x86_64 <mainrepo>/.venv/bin/python <script>
```
Cache the expensive frames once (`load_inputs` + `attention_curves` for every
level×treatment) into a pickle in `/tmp`; every later check is then seconds, not minutes.
`bootstrap_era_shares` at 500 draws is ~1 min per level/treatment.

**Running a note's own "Reproducing" block safely.** If it says `python -m <module>` and
the module writes the artifact, `cp` the artifact to /tmp first, run the command verbatim,
then `cmp`. Byte-identical output is stronger evidence than refusing to run it, and the
backup means a non-reproducible rebuild costs nothing.

**Error classes that survive correction passes** — check these even when the work order
does not name them:

1. **Prose mechanism claims attached to a real number.** The number checks out; the
   *reason* given for it is wrong. (A rename was said to fail because of "a 43-year gap";
   it actually failed a birth-window filter, and the two topics overlapped for 10 years.)
   Re-derive the *filter that fired*, not just the quantity quoted.
2. **The same quantity stated in two sections with two different values.** Grep the note
   for every occurrence of a relationship, not every occurrence of a number.
3. **Grain mixing inside one sentence** — a decade share next to an era share next to a
   smoothed-year share, all rendered as bare percentages. A "local mode" claim can be true
   at year grain and false at the decade grain of the number quoted beside it.
4. **The note's own self-audit section.** It is written by the stage being audited and
   tends to understate the error history. Check its counts against the pipeline
   `CONTEXT.md` record.
5. **Rounding**: verify rate cells by rounding the raw fraction *once*. A 2dp-then-1dp
   path lands cells one increment high; that defect class is logged as
   `double-rounding-in-published-notes`.

**Things that reliably DO reproduce and are cheap to confirm wholesale**: parquet-backed
table cells (dump the whole table and diff against the markdown), era denominators,
bootstrap CIs, exemplar quote text. Verify quotes as *normalized substrings* — notes
legitimately elide with `…` and swap `—`/`--`, so exact equality gives false alarms.

**Backlog entries live on the base branch.** Never grep `TASKS.md` from inside a worktree;
the worktree copy is a frozen snapshot. Use `git show master:TASKS.md`.
