---
name: analysis-module-refactor-gates
description: How to gate refactors of the pure analysis modules (register/indices/issues/trends/profiles/combat) — regenerate the committed parquet and diff bytes, not just run pytest
metadata:
  type: feedback
---

The pure analysis modules (`register.py`, `combat.py`, and by the same design
`indices`/`issues`/`trends`/`profiles`) emit **committed parquets under `data/` whose figures are
quoted verbatim in a `notes/*.md` report**. A changed byte is a changed published number, so the
real gate for a SIMPLIFY pass is not the test suite — it is regenerating the artifact and
comparing bytes. Do this after EVERY individual edit, not once at the end.

**Cheapest byte-identity check: regenerate straight into the committed `data/<mod>/` and run
`git status --short data/<mod>/`.** Empty output = byte-identical across the whole directory,
including parquet metadata, in one command — no hashes to collect. Keep a *pristine* regeneration
in the scratchpad as the golden reference too, so a value-level diff can name WHICH column moved
when the byte check fails. Both writes are safe: these builds take an injectable `out_dir=`.

**Why:** these modules are float-heavy (bootstraps, entropies, ratio-of-sums). A refactor can
keep all ~530 tests green and still move a last-bit value — e.g. changing accumulation order,
swapping `np.float64` for Python `float`, or reordering a `sum()`. The hash catches what the
tests structurally cannot.

**How to apply:**
- Regeneration is cheap (`register`: ~6s for 2,000 replicates x 9 eras x 3 treatments), so
  there is no excuse to batch edits. Delete the parquet before the final run to prove it is
  rebuilt, not stale.
- Things that turned out hash-safe on `register.py`: merging three duplicated
  `merge` + `_require_full_merge` blocks into one loop over `(table_name, table, columns)`;
  replacing a hand-built `pd.DataFrame(index=...)` + per-column `to_numpy(dtype=float)` loop
  with `agg.astype(float).reset_index(drop=True)`; `1.0 + np.argsort(...)` for
  `np.argsort(...).astype(float) + 1.0`; carrying an already-computed cell total in the
  accumulator tuple instead of recomputing `paras[members].sum()` in the return comprehension
  (same accumulation ORDER, so bit-identical).
- Keep the "pure functions taking DataFrames" seams. Private helpers like `_estimate_group`,
  `_cell_plan`, `_replicate_weights` are directly pinned by tests (including exact tuple
  arity), so changing their signatures is test churn, not simplification.
- Error-message strings are pinned by `pytest.raises(match=...)` — grep the parametrize lists
  before touching any raise. On `register.py` the three speech-level merge stages are
  parametrized as `speech_annotations` / `speech_stats` / `speech_markers`.
- More byte-safe moves, confirmed on `combat.py`: a `NamedTuple.empty()` classmethod replacing
  repeated positional construction; a small local `NamedTuple` replacing a list of positional
  4-tuples unpacked as `for _, _, n, _ in sub` (7 sites); `arr / float(arr.sum())` for
  `arr / arr.sum()` (float64 -> Python float is exact); `df.reindex(columns=...)` to pin column
  ORDER; splitting `a, b = 1.0, inf if cond else 1.0` into a real if/else. Preserve any loop that
  consumes one RNG in group order — per-cell seeding depends on it.

**`path=CONSTANT` defaults are NOT a redirect seam.** A `def f(..., path: Path = SOME_PATH)`
binds at import, so `monkeypatch.setattr(mod, "SOME_PATH", tmp)` silently does nothing and tests
are forced to replace the whole FUNCTION with a stub — over-mocking that hides the real code from
the test. Use `path: Path | None = None` + `resolve at call time`, which is already this repo's
idiom (`build_combativeness`'s `out_dir=None -> COMBAT_DIR`). It also reaches `main()`/CLI entry
points, which cannot be passed a keyword argument.

**Un-stubbing over-mocked leaves exposes latent fixture weakness — expect it and fix the
fixture.** On `combat.py`, running the real lexical proxies revealed a synthetic fixture whose
`zero_sum` column was constant, so its Spearman correlation was undefined and landed as a bare
`NaN` token in the emitted JSON meta. That is a fixture bug the stub had been concealing.

**Console scripts:** `[project.scripts]` in this repo is reserved for PIPELINE entry points
(`pp-fetch`, `pp-analyze`, `pp-site`, `pp-annotate`, `pp-word-families`). One-shot analysis
modules — `taxonomy`, `register`, and the rest — have `argparse` + `main()` but NO console
script and are documented as `python -m presidential_profiles.<mod>`, which is also the command
the published note gives for reproduction. Do not add a script entry during SIMPLIFY; if a test
sets `sys.argv[0]` to an invented `pp-<name>`, fix the TEST (argv[0] is only argparse's `prog`)
rather than inventing the entry point to match it. See [[annotation-money-path-guards]] for the
sibling calibration on the paid-API module family.
