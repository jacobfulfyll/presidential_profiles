---
name: eras-test-seams
description: Hermetic-test seams for presidential_profiles eras.py (era-atlas: fingerprints, similarity, periodization, paid portraits) — injectable inputs, the conftest redirect trap that breaks real-corpus run(), and the LOO identity+counter pattern
metadata:
  type: reference
---

Authoring offline tests for `eras.py` (era fingerprints + raw/detrended cosine +
contiguity-constrained periodization + paid LLM portraits over the frozen
parquets). Builds on [[test-toolchain]] and [[combat-test-seams]].

**Worktree venv gotcha.** `.worktree/era-atlas/` has NO local `.venv`. Use the
MAIN repo's interpreter with the worktree on the path:
`PYTHONPATH=<worktree>/src arch -x86_64 /…/presidential_profiles/.venv/bin/python`.
Confirm `import presidential_profiles` resolves to the worktree `__file__` before
trusting anything.

**What IS injectable (covers ~everything on tiny synthetic frames):**
- `load_master_frame(annotations=, speech_annotations=, paragraphs=, speeches=, taxonomy=)`
  — pass a 3-topic/2-domain synthetic taxonomy dict `{level2:[{name,level1}]}` and
  4 tiny frames; it assembles the real master frame. Build downstream frames
  THROUGH it so their shape matches production exactly.
- `build_axis_frame(df, grain, markers=, stats=, taxonomy=)`, `periodization(fine_frames, k_range=)`,
  `combativeness_ci_check(combativeness=)`, `validate_combat_axes(era_axis, combativeness)`,
  `load_agreement_bands(path=)`, `_write_portraits(out_dir, ...)`,
  `generate_portraits(run_data=, out_dir=, dry_run=, yes=)`.

**conftest's autouse `redirect_annotation_dirs` breaks real-corpus `run()`** — same
trap as combat. It repoints `ann.ANNOTATIONS_DIR` at tmp, so the annotation loads
inside `load_master_frame()` fail. Restore with
`monkeypatch.setattr(ann, "ANNOTATIONS_DIR", eras.ANNOTATIONS_DIR)` — eras captured
the real Path at import (before the fixture ran). `run(out_dir=tmp)` is only ~1.5s
and byte-identical on rerun, so ONE integration anchor is cheap.

**Agreement seam default is bound in the BODY (`if path is None: path = AGREEMENT_PATH`),
UNLIKE combat's signature-bound default.** So `monkeypatch.setattr(eras, "AGREEMENT_PATH", tmp)`
DOES redirect a no-arg `load_agreement_bands()` call — pin that property (a
signature-bound default would make the monkeypatch silently inert). Absent → None;
present-but-missing-columns → raises; `_agreement_provenance` returns
`sampling_only` for BOTH None and non-None (never fabricates `sampling+annotator`).

**Portrait manifest ROUTING regression** (the top QA ask): `_write_portraits`
must write under `eras.MANIFESTS_DIR` (= `data/eras/manifests/`), NOT via
`llm_annotations.write_manifest` (which hardcodes the frozen paid-annotation dir).
Test: monkeypatch `eras.MANIFESTS_DIR` + `eras.corpus_fingerprint`; assert manifest
lands there AND is ABSENT from `ann.MANIFESTS_DIR` (conftest redirected that to
tmp). `texts=None` → placeholders, status `not_generated`, NO manifest (never
fabricate a portrait). Cost in the manifest is ACTUAL from usage, not the estimate.

**generate_portraits money gates:** pass `run_data={"tables":{"era_fingerprints":fp}}`
and monkeypatch `eras.load_master_frame` + `eras.PARAGRAPHS_PATH` (select_exemplars
reads the latter as a module global at call time). Use the `anthropic_construction_bomb`
fixture to prove dry-run / `--yes`-refusal / ceiling-breach all fire BEFORE any
client. For the graceful "no client → not_generated" branch, monkeypatch
`anthropic.Anthropic` to raise (don't use the bomb there).

**Real-corpus anchors (reproduce exactly):** present-era DETRENDED nearest =
`Civil War & Reconstruction`, cosine 0.315, not a temporal neighbor. 63 axes
(topic17/combat3/register3/marker19/stat11/genre10). Every similarity row
`ci_components="sampling_only"`, `agreement_source="absent:agreement_v1.parquet"`.
Periodization DISTINCT-canonical (primary k, full): bin8 6 rows / 6 distinct;
president 5 rows / **4 distinct** (1881+1882 both hit canonical 1878 — the §5
double-count fixed in 27e7387). LOO on `marker_opponents` is IDENTICAL at every k,
both grains.

**LOO / drop-machinery-is-live pattern** (generalizes [[same-frame-invariance-tests]]):
to prove a "drop column X, boundaries shouldn't move" check isn't inert, pair TWO
synthetic frames — (a) IDENTITY: make X a perfect duplicate of another axis
(redundant → dropping it moves nothing at any k); (b) COUNTER: make X carry the
ONLY clean split at a distinct position (dropping it DOES move a boundary). A
constant X is a weak identity (zscore drops it anyway); the redundant-duplicate
construction is the strong one. `.sample`-style reorders and higher-k orthogonal
wobble both leak — verify empirically, don't hand-wave.

Verified 2026-07-21 at WRITE-TESTS: 93 tests, 95% line coverage (only CLI /
paid-network loop / defensive categorical-cast uncovered); 10/10 mutation batch
caught (ddof, combat threshold+guard, fallback [1:5]→[1:], agreement sig-bind,
manifest misroute, one-sided adjacent detrend, LOO-inert, nn idxmax→idxmin,
staleness inversion). Full suite 1592 → 1685.

**Independent QUALITY-CHECK (2026-07-21), 15-mutant pass — 4 NEW gaps the 93-test
suite missed (all instances of already-known classes; suite 93→97, 1685→1689).**
Each survivor + the fixture that closes it (see git dea5ebd):
- **Contiguity constraint (`connectivity=_chain_connectivity(n)` → `None`) SURVIVED.**
  `TestBoundariesAtK` only used the clean two-block fixture `[0,0,0,10,10,10]`, where
  constrained and UNconstrained Ward agree. Kill it with an ALTERNATING fixture
  `[0,10,0,10,0,10]`: unconstrained groups all-0s vs all-10s → 5 boundaries at k=2;
  the path-graph forces contiguous runs → exactly 1. A "contiguity" test whose fixture
  is block-separated proves nothing.
- **`matches_canonical` threshold (`<= BIN_WIDTH` → `<= 2*BIN_WIDTH`) SURVIVED.** Only
  exercised by (a) a test that RE-COMPUTES the SUT formula (`== (gap.abs()<=BIN_WIDTH)`)
  and (b) frozen-parquet anchors (on-disk column, immune). Kill with a hand-derived
  two-sided pin: gap 12 → miss, gap 8 (== BIN_WIDTH) → hit. President-grain split at
  1862 gives gap +12 to canonical 1850; bin8 split at 1824 gives gap +8 to 1816.
- **`stat_self_reference` (`i/(i+we)` → `we/(i+we)`) SURVIVED** on the i=we=5 fixture
  (both 0.5). Classic symmetric-fixture trap; asymmetric i=3/we=9 → 0.25 pins the
  numerator.
- **`group_balanced_era_detrended` weight-drop (`zb = z*w` → `zb = z`) SURVIVED** the
  diagonal-only assertion (cosine self is always 1). Pin a hand-derived OFF-diagonal:
  3 eras, topic(2)+combat(1), founding↔Civil-War detrended cosine = 0.693375 (vs
  unweighted 0.577350). Derivation: with s=√(3/2), q=√3/2, vectors [-q,-2q,2s] &
  [q,-q,s] → (q²+2s²)/√((5q²+4s²)(2q²+s²)) = 3.75/√29.25.
Equivalent/near-equivalent probes that did NOT survive but note: window `<=`→`<`
(M14) needs ≥2 strict-interior peers besides the exact-boundary one to distinguish
(a 2-unit fixture's fallback re-picks the same peer); it is practically unreachable
on real float center_years but cheap to pin. Coverage-adjudication confirmed the 25
uncovered lines are EXACTLY {904 defensive categorical cast, 1095 generate_portraits
run_data=None → run() delegation, 1141-1151 paid loop, 1231-1252 main/CLI} — all
acceptable. `--cov` drops a non-gitignored `.coverage`; `rm` it before committing.
