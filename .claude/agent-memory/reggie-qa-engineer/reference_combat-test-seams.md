---
name: combat-test-seams
description: Hermetic-test seams for presidential_profiles combat.py (combativeness research module) — what is injectable, what is not, and why patching its Path constants silently does nothing
metadata:
  type: reference
---

Authoring offline tests for `combat.py` (per-era flag rates + speech-clustered
bootstrap over the frozen annotation parquets). Builds on [[test-toolchain]].

**What IS injectable (use these; they cover ~everything):**
- `load_frame(annotations=, speech_annotations=, paragraphs=, speeches=, entities=)`
  — all five inputs as DataFrames. A ~12-speech synthetic corpus drives the whole
  module; 99% line coverage is reachable without the 36k corpus.
- `_adversary_counts(entities=)`, `lexical_baseline(path=)`, `lexical_correlations(path=)`,
  `load_agreement_bands(path=)`, `build_combativeness(out_dir=)`,
  `rate_table(..., bands=, fits=)`, `ratio_table(..., fits=)`.

**The trap: default args bind at import.** `def lexical_baseline(df, path=SPEECH_MARKERS_PATH)`
and `def load_agreement_bands(path=AGREEMENT_PATH)` capture the real Path when the
module is imported, so `monkeypatch.setattr(combat, "SPEECH_MARKERS_PATH", tmp)` has
**no effect** — the call still reads real data. Patch the *function* instead. Same
shape as `corpus_fingerprint`, imported into combat's namespace but looked up as a
module global at call time (so `monkeypatch.setattr(combat, "corpus_fingerprint", ...)`
DOES work).

**`build_combativeness` is therefore not hermetic on synthetic data** — it calls
`lexical_baseline(df)` / `lexical_correlations(df)` / `corpus_fingerprint()` with no
path forwarded, and `lexical_baseline`'s own row-completeness guard then (correctly)
rejects a 12-speech frame against the real 1057-row `speech_markers.parquet`. Stub
those three module globals in a fixture; each has its own direct test elsewhere.
`COMBAT_DIR` *is* read at call time (`if out_dir is None: out_dir = COMBAT_DIR`), so
`main()` is testable by patching `combat.load_frame` + `combat.COMBAT_DIR`.

**conftest's autouse `redirect_annotation_dirs` breaks real-corpus tests.** It repoints
`llm_annotations.ANNOTATIONS_DIR` at tmp_path, and `annotation_path()` reads that global
at call time — so `load_frame()` with no args cannot find the annotations. Restore with
`monkeypatch.setattr(ann, "ANNOTATIONS_DIR", combat.ANNOTATIONS_DIR)` (combat captured
the real value at import, before the fixture ran). Real `load_frame()` + `rate_table()`
over the full corpus is only ~0.5s, so ONE integration anchor test is cheap.

**Real-corpus anchors that reproduce exactly** (good integration assertions): 36,229
rows / 1,057 speeches; corpus-wide party_attack .0656 / enemy_naming .2107 /
zero_sum .0768; SOTU-only party_attack present era 0.142197, Civil War 0.029586,
War & New Deal 0.045977 (n=3, `ci_status == "suppressed_n_floor"`); entity
consistency 7393/7634.

**Never create `data/llm_annotations/agreement_v1.parquet`** — owned by the parallel
`inter-model-agreement-check` task. Build band fixtures in `tmp_path` and pass
`path=`. A test asserting `not C.AGREEMENT_PATH.exists()` doubles as a boundary guard.

**Real-corpus ratio anchors** (SOTU-only, present era as numerator), added to
the integration test at QUALITY-CHECK because only the marginal RATES were
pinned and the headline the report quotes was not: party_attack vs Civil War
`ratio 4.806243, CI [2.348699, 14.315654], low_cluster_caution`; enemy_naming vs
The founding `1.091053 [0.786503, 1.612468]`; zero_sum vs The founding
`1.761957 [0.945389, 4.878660]`; every War & New Deal ratio
`suppressed_n_floor`. Share one `_fit_all` between `rate_table` and
`ratio_table` in that test — refitting doubles its cost for nothing (per-cell
seeding makes the two routes provably identical).

Verified 2026-07-21 at the IMPLEMENT checkpoint (135 tests, 96% coverage) and
again at QUALITY-CHECK after 74 mutations (177 tests, 99% — only
`if __name__ == "__main__"` uncovered). The three surviving mutants are
EQUIVALENT, not holes: refitting instead of sharing fits (per-cell seeding makes
it identical), `to_parquet(index=True)` (RangeIndex is stored as metadata, round
trips identically), and consulting the band lookup on a decade grain (lookup
keys are `str` era labels, decade groups are `int`, so a match is impossible).
See [[deferred-seam-is-the-deliverable]] and [[same-frame-invariance-tests]] for
the two real holes this module's first test pass left.
