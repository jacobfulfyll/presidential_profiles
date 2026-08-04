---
name: structural-guards-scan-the-whole-module
description: When asked to fix ONE instance of a structural code defect, write the AST guard over every function/table in the module rather than the named instance — it keeps finding more
metadata:
  type: feedback
---

When a work order names a single instance of a *structural* defect ("bind
`rolling_curve` explicitly", "add `ci_status` to `jackknife.parquet`"), write the
guard as a **sweep over the module's own registry**, not as an assertion about the
named instance.

**Why:** on `convergence-analysis` (2026-07-21) both sweeps immediately paid for
themselves.
- A loop-leak AST scan (`names READ at a function's top statement level but bound
  ONLY inside one of its loops`) was asked for on `build_convergence`. Run over
  every `FunctionDef` in the module it found a **second, unreported live
  instance** — `_selftest` read `cluster` after its loop had ended, which was
  simultaneously the mislabelled `meta.selftest.n_windows` field the work order
  listed as a separate item. One guard, two carry-forwards.
- A `ci_status` sweep driven by `C.OUTPUT_PATHS` (the module's own output-table
  registry) rather than by a hard-coded table list forced `permutation_null` into
  the contract too, and makes a future table added without a gate fail without
  anyone remembering to extend the test. Pair it with a
  `registry == set(dir.glob("*.parquet"))` test, or a table written outside the
  registry slips the sweep entirely.

**How to apply:**
- Do NOT allow-list module globals or builtins out of a leak scan: a name STORED
  inside the loop is a local regardless of what shares its spelling, so
  subtracting those sets only hides real leaks.
- Factor the check into one function the live sweep AND its mutation control both
  call, then mutate each defect shape back in (drop the column / null it / put a
  foreign word in it) and confirm exactly one violation fires.
- A gate that reads `ok` on every real row is only convincing because a
  constructed fixture proves it CAN fire. Pair the threshold table with an
  identity/counter fixture pair (a pivotal president whose deletion empties the
  trend vs a redundant one whose deletion moves nothing) — same shape as
  [[eras-test-seams]]'s LOO identity+counter pattern.

Related: [[convergence-test-seams]], [[mirror-tests-need-real-fn-anchor]].
