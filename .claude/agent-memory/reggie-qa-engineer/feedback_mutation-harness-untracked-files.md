---
name: mutation-harness-untracked-files
description: Mutation harness must snapshot file bytes, not rely on `git checkout --`, because a module written by IMPLEMENT is untracked and cannot be restored that way
metadata:
  type: feedback
---

In a WRITE-TESTS stage the module under test is usually a BRAND-NEW file that
IMPLEMENT created and nobody committed yet — `git status` shows it as `??`.
`git checkout -- <file>` then fails with *"pathspec did not match any file(s)
known to git"*, the revert silently does not happen, and the harness crashes
leaving the FIRST mutation applied in the source tree.

**Why:** the revert recipe in [[mutation-check-recurring-holes]] assumes a
tracked, committed baseline. That holds for QUALITY-CHECK on existing code, not
for WRITE-TESTS on new code.

**How to apply:** read the file's bytes once before the first mutation, keep
them in memory (and dump a copy to the scratchpad), and revert with
`SRC.write_bytes(PRISTINE)`; make the cleanliness assertion
`SRC.read_bytes() == PRISTINE` rather than an empty `git diff`. Verify with
`diff -q snapshot current` at the end, and re-run the full suite once after the
harness finishes — if a revert ever failed silently, the suite goes red and
tells you. The rest of the recipe (assert `s.count(old) == 1` before replacing,
pass each test path as a SEPARATE argv element so exit code 5 can't masquerade
as CAUGHT) still applies unchanged.
