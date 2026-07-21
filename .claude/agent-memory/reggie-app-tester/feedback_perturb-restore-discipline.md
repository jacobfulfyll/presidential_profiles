---
name: perturb-restore-discipline
description: When perturbing a committed artifact to test degradation, verify the restore by sha against a pristine backup — a chained perturb/move sequence silently restores the perturbed copy
metadata:
  type: feedback
---

When a verification step requires perturbing a committed artifact (e.g. blanking a field in
`data/topic_display_names.json` to test graceful degradation), **take a sha256 of the pristine
file first, and after restoring assert the sha matches** — do not accept "a file is back at that
path" as proof of restoration.

**Why:** on 2026-07-21 I ran two chained degradation scenarios: (A) overwrite the file with a
perturbed version, then (B) `mv` "the file" aside to test the missing-file path. Step B moved the
**already-perturbed** file, so restoring it put the perturbation back while `ls` and a successful
build both looked fine. Only the sha comparison caught it. `git checkout -- <path>` then gave a
truly clean restore. A silent bad restore on a provenance-stamped artifact is exactly the failure
mode these tests exist to prevent, and it would have been committed as "verified".

**How to apply:** before the first perturbation, `cp` the artifact to the scratchpad AND record its
sha. After every scenario, restore with `git checkout -- <path>` (authoritative) rather than moving
files back, then `diff` against the scratchpad backup and confirm `git status --short data/` is
empty. Chain scenarios only if each one re-restores from the pristine source, never from whatever
happens to be on disk. Pairs with [[verification-workflow]] and [[verification-triangulate]].
