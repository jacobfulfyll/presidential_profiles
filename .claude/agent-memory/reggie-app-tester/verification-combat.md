---
name: verification-combat
description: How to verify combat.py (combativeness-over-time) end-to-end — $0, ~2s, byte-identical reruns, and the prose/parquet audit method that actually finds defects
metadata:
  type: reference
---

Verification recipe for `presidential_profiles.combat` — a **published research claim**, so the
audit is as much prose-vs-parquet as it is code.

**Invocation.** No console script by design (matches `taxonomy.py`): `python -m
presidential_profiles.combat [--quiet]`. `DATA_DIR` derives from `Path(__file__).parents[2]`, so
output location follows the **PYTHONPATH**, not the cwd — running from `/tmp` still writes the
worktree's `data/combat/`. Full run ≈ 2s over 36,229 paragraphs.

**$0 guard is verifiable, not just asserted.** `combat`'s import graph is `corpus` + `llm_annotations`
+ `trends`, none of which import anthropic. Prove it rather than grep it:
`import presidential_profiles.combat; 'anthropic' in sys.modules` → False, and stays False through a
full `build_combativeness()`. Note: importing *every* package module (pkgutil sweep) DOES load
anthropic via `annotate`/`taxonomy` — so scope the check to the module under test or the result is
meaningless.

**Determinism.** All 7 parquets are wall-clock independent and byte-identical across processes and
cwds (seeded `_cell_rng(BOOTSTRAP_SEED, group_idx, treatment_idx)` — order-independent by design).
**`combat_meta.json` is NOT**: it carries `"generated": date.today()`. `git status --short data/combat/`
is only empty on the generation date; any later rerun dirties exactly that one file. Test it by
swapping `combat.date` for a `datetime.date` subclass, not by waiting.

**The optional agreement seam** (`data/llm_annotations/agreement_v1.parquet`, owned by a parallel
task): exercise it WITHOUT creating the real path — write a synthetic band table to a temp path and
pass it via `load_agreement_bands(path=...)` → `rate_table(..., bands=)` / `ratio_table(..., bands=)`.
Caveat: `load_agreement_bands`'s `path=AGREEMENT_PATH` default **early-binds at def time**, so
monkeypatching the module constant does nothing and `build_combativeness()` cannot be redirected at
all (it calls the loader with no args). Production behaviour is still correct — the file appearing
on disk is picked up — but that asymmetry vs. the late-bound `markers_path` is a testability seam gap.

**Prose/parquet audit method that paid off.** Verifying the ~60 tabulated numbers found *zero*
defects — the tables are generated-adjacent and correct. Every defect was in **narrative asides**:
counts quoted from CONTEXT.md without recomputation, era-grain words attached to decade-grain
numbers, and internal cross-references. So spend the audit budget on prose sentences containing a
number, not on the tables. Specific traps in this repo:
- **era (`trends.ERAS`, 9 bands) vs decade** are constantly interleaved; a figure computed at one
  grain gets described in the other's vocabulary. Always recompute at BOTH grains and see which matches.
- **Counts inherited from CONTEXT.md are not verified facts.** "32 fireside chats" propagated from
  the brief into the report; the data says 32 = `public_remarks_or_address` ∧ `broadcast_radio_or_tv`
  speeches in 1933–1945 (3 of them Truman surrender broadcasts), while actual fireside-titled speeches
  are 30 era-wide / **14 in the 1930s decade** the sentence was about.
- Section cross-refs (`§5.6`) and "four/five lines of evidence" counts drift as sections are added.
