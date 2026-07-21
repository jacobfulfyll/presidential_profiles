# Era Atlas v1 — fingerprints, similarity, and data-driven periodization

**What this is.** Every era of U.S. presidential rhetoric is reduced to a
standardized **fingerprint** — a vector over 63 measures (level-1 topic mix,
combativeness, proposal-vs-values register, style markers, readability/pronoun/modal
usage, speech-type mix). From the fingerprints fall three things: a similarity
matrix (raw and drift-detrended), a data-driven periodization held against the
historians', and LLM-written era portraits. Built by
`src/presidential_profiles/eras.py`; artifacts in `data/eras/`.

**This is description, not inference.** The fingerprint is assembled from measures
we designed, so "which era resembles which" is a statement about our measurement
apparatus, not a discovered law. The convergence *hypothesis test* is a separate
task. Read this atlas as a structured description with its uncertainty stated, not
as a causal claim.

**Number conventions.** Cosine similarities and z-scores are reported to the
precision shown (2–3 significant figures), computed once from the fraction — no
double-rounding. Recovery/agreement rates are given as explicit counts (`N of M`),
never as a rounded percentage. A "z-score" is standardized across the units being
compared (across the 9 eras for era-grain fingerprints; across the 31 8-year bins
or 45 presidencies for the fine grains), population std (ddof=0).

---

## 1. Headline — which past era does the present most resemble?

**Raw similarity gives the useless-but-instructive answer:** the present era
(2017–2026) most resembles **Post–Cold War** (1989–2016), cosine **0.54**. That is
the monotonic drift of language talking — the 2020s resemble the 2010s because
adjacent decades always resemble each other. It is exactly the trap the task exists
to defuse.

**Detrended** — after subtracting each era's adjacent-era mean so the drift is
removed — the answer changes, and changes cleanly:

| Rank | Era the present resembles (detrended) | cosine |
|---|---|---|
| 1 | **Civil War & Reconstruction** (1850–1877) | **0.315** |
| 2 | The Cold War | 0.051 |
| 3 | War & New Deal | 0.046 |
| … | The founding | 0.027 |
| 8 | Post–Cold War | −0.838 |

The present era's closest drift-removed analogue is **Civil War & Reconstruction**,
and it is *decisively* the closest: the gap from #1 to #2 is 0.264 — more than ten
times the spread among #2–#4 (0.051 down to 0.027). The match is **mutual** — Civil War &
Reconstruction's own nearest detrended neighbor is the present era (0.315). Its
immediate predecessor, Post–Cold War, sits at −0.84: the detrend defines the
present partly *against* the era it grew out of.

**What drives the rhyme** (top axis contributions to the detrended cosine): both
eras stand out from their temporal neighbors on **self-reference** ("I" over "we"),
**law/crime/justice** and **civil-rights/immigration** topic emphasis, **party
attack**, **us-vs-them** framing, and a de-emphasis of public finance. That is a
coherent *national-division* signature — polarization plus a rights/legitimacy
fight — which is a historically defensible pairing.

**Three honest caveats, all of which matter:**

1. **The resemblance is "closest," not "close."** 0.315 is a modest absolute
   cosine. In raw terms the present era is one of the most *distinctive* eras in the
   corpus — its extreme markers are hype (z = +2.71), boosters (+2.63), us-vs-them
   (+2.43), party attack (+2.33), and press-conference genre (+2.27). So the finding
   is "if forced to name its nearest historical rhyme, it is the Civil War era," not
   "the present is a rerun of the 1860s." This is nearer the task's "honest null"
   end than a strong analogue.
2. **It is an era-*aggregate* rhyme, not a president-to-president match.** At the
   noisier presidency grain the two present-era presidents resemble *different*
   individuals after detrending — Trump ↔ Truman (0.46), Biden ↔ John Adams (0.29) —
   neither pointing at a Civil-War-era president. The Civil War rhyme is a property
   of the era as a whole, not of any one president.
3. **The rhyme is robust to the two checks that could have killed it.** Re-weighting
   so each of the six axis groups contributes equally — each z-axis multiplied by
   `1/sqrt(n_group)` so every group enters the cosine as the *mean* of its per-axis
   products regardless of how many axes it holds (guarding against the topic and
   marker groups' larger axis counts) — the present era's detrended nearest neighbor
   is still Civil War & Reconstruction (cosine **0.3386**, still rank 1; #2 War & New
   Deal at 0.205). This number and the full re-weighted ranking are emitted to
   `eras_meta.json → similarity.group_balanced_robustness` so a reader can diff them.
   And the periodization it sits inside is unchanged when the availability-bounded
   `opponents` marker is dropped (§5).

**Note on a hypothesis that did *not* pan out.** The plan floated a plausible
Gilded-Age ↔ present rhyme (inequality/tariff/immigration). It did **not** appear:
the Gilded Age is only the present era's *fourth-from-last* detrended neighbor
(cosine −0.079; three eras rank below it — Progressives & Depression −0.11,
Expansion −0.20, Post-Cold War −0.84). The rhyme that did appear (Civil War) is
different, and about
partisan/rights conflict rather than political economy.

---

## 2. Method (one paragraph)

Fingerprints are assembled per unit at three grains — the 9 named `trends.ERAS`
eras (the reporting grain), 31 eight-year bins, and 45 presidencies (the plan
mandates presidency-level **and** 8-year bins; decades are never imposed). Each axis
is z-scored across the units of its grain so no scale dominates. **Raw similarity**
is cosine between era fingerprints. **Detrended similarity** subtracts each era's
adjacent-era mean (the `similarity.build_adjusted` local-mean trick, `ERA_WINDOW =
24`, adapted to the 9 ordered eras where a fixed 24-year window would capture only
the era itself); at the presidency grain the *literal* `build_adjusted` 24-year
window is used. **Periodization** is contiguity-constrained Ward clustering (a
path-graph connectivity constraint forces every cluster to be a contiguous run of
time, so a cluster boundary is a period boundary) on the *raw* z-fingerprints — the
detrend is for the rhyme question, not the boundary question.

---

## 3. Detrending sanity checks (all pass)

The plan's falsification checks for a broken fingerprint or an over-aggressive
detrend:

- **Raw similarity should track time.** It does: across the 36 era pairs, the rank
  correlation between raw cosine and temporal proximity (−|year gap|) is
  **Spearman 0.70**. In nearest-neighbor terms, **6 of 9** eras have a temporal
  neighbor as their raw nearest neighbor; the 3 exceptions still pick a
  temporally-nearby era rather than a distant one — two are the mutual
  Expansion ↔ Gilded Age tie (both 19th-century, 0.62, skipping over Civil War &
  Reconstruction), and the third is War & New Deal (1940) → Post-Cold War (2003),
  which crosses into the 21st century but skips only the Cold War between them.
  Each chosen neighbor is one intervening era away, so the fingerprint tracks the
  drift, it is not broken.
- **Detrending should break that adjacency.** It does, completely: **0 of 9** eras
  have a temporal neighbor as their *detrended* nearest neighbor.
- **Detrending should not destroy all signal.** It does not: the off-diagonal
  detrended cosines have std 0.32 and range −0.84 to +0.32 — real structure
  survives, the bail condition ("all detrended ≈ 0") is not met.
- **At least one plausible detrended pairing should appear.** The Civil War ↔ present
  national-division rhyme (§1) is that pairing.

---

## 4. Combativeness and the `ci_status` trust gate

The three combativeness axes (party_attack, enemy_naming, zero_sum) are per-unit
**raw** paragraph-flag means over the full paragraph set. At the era grain these are
**byte-identical** to `data/combat/combativeness.parquet`'s `raw` treatment (max
absolute difference 0.0 on all three flags), which carries `ci_status = ok` on all
27 of its cells. The combat table's `genre_standardized` and `sotu_only`
treatments carry thin-cell suppression: **9 cells are `suppressed_n_floor`** (Civil
War & Reconstruction and War & New Deal) and 3 are `low_cluster_caution`. **None of
these 12 non-`ok` cells seeded a fingerprint axis** — the audit is in
`data/eras/combativeness_ci_check.parquet` and the counts in `eras_meta.json`. The
raw all-genre rate does not have the suppression problem because it is computed over
each era's full paragraph set (thousands of paragraphs), not the n=3 SOTU sub-cell
that the suppression protects against.

---

## 5. Periodization — data boundaries vs the historians'

Contiguity-constrained clustering into 9 periods (matching the 9 canonical eras) on
the 8-year-bin fingerprints yields 8 internal boundaries. A discovered boundary
"matches" a canonical one if within one bin width (8 years).

| Discovered boundary | Nearest canonical | Gap (yr) | Match? | Reading |
|---|---|---|---|---|
| 1792 | 1816 | −24 | ✗ | spurious early split (founding-era data sparsity) |
| 1808 | 1816 | −8 | ✓ | founding → Expansion |
| 1848 | 1850 | −2 | ✓ | → Civil War |
| 1872 | 1878 | −6 | ✓ | → Gilded Age |
| 1912 | 1901 | +11 | ✗ | Progressive break drawn *late*, at WWI onset |
| 1936 | 1933 | +3 | ✓ | → War & New Deal |
| 1952 | 1946 | +6 | ✓ | → Cold War |
| 2016 | 2017 | −1 | ✓ | → the present era |

**The data recovers 6 of 8 *distinct* canonical boundaries** (1816, 1850, 1878,
1933, 1946, 2017) — strong validation; the bail condition ("recovers no canonical
boundary → suspect a bug") is nowhere near met. (Counts throughout are *distinct
canonicals recovered*, not matching-boundary rows — see the presidency-grain wrinkle
below.) Where it disagrees is the finding:

- **The Progressive boundary is drawn at ~1912, not 1901** (11 years late — outside
  tolerance). The data reads the rhetorical break as the WWI-era shift, not
  McKinley's assassination / T. Roosevelt's accession.
- **The end of the Cold War (1989) leaves no boundary at the 8-year-bin grain.** The
  data would rather spend a boundary on the sparse founding decades (the spurious
  1792 split) than mark 1989 — presidential *language* did not rupture at the fall
  of the Berlin Wall the way the textbooks' geopolitical periodization does. This is
  **grain-dependent**, and the disagreement is worth stating in full: at the
  **presidency** grain the clustering *does* place a boundary near 1989 (it recovers
  **4 of 8 distinct canonicals** — 1850, 1878, 1946, 1989 — with different misses).
  Note the counting subtlety this row exposes: the presidency grain has 5 *matching*
  boundaries but only 4 *distinct* canonicals, because two of its discovered
  boundaries (1881 and 1882, on either side of one short presidency) both land on the
  1878 canonical — a double-hit counted once. So "1989 is / isn't a rhetorical
  boundary" depends on whether you cut history into regular 8-year slices or into
  presidencies. Neither grain is privileged; both ship
  (`data/eras/periodization.parquet`, columns `grain`, `condition`, `k`).

---

## 6. Leave-one-out on `opponents` — no availability artifact (the mandated check)

`opponents` is availability-bounded: its regex names `democrats`, `republicans`,
`fake news`, `radical left` — words unavailable for most of the record (the marker
is 0.00 per-10k in the founding era). The amendment required rebuilding the
periodization with it dropped and reporting which boundaries are about rhetoric vs
about when the words became available.

**Result: the periodization is identical with `opponents` dropped — at every k from
2 to 12, at both the 8-year-bin and presidency grains.** Every boundary in §5
survives, byte-for-byte. In particular the single most prominent split (the k=2
boundary) is **1912 in both** the full and the dropped fingerprints — it is not near
1828/1854 (party formation) or 1990–2016 (`fake news`/`radical left`), the two
places the amendment warned an availability artifact would land by construction.

**So the headline boundary is about rhetoric, not word availability**, and the bail
condition ("headline boundary disappears when `opponents` is dropped → report as
artifact, lead with the LOO-stable boundaries") does not trigger. The boundaries in
§5 *are* the leave-one-out-stable boundaries. (`opponents` is one of 63 z-scored
axes; the contiguity-constrained boundaries are driven by the aggregate multivariate
shift, which one axis does not move.)

---

## 7. Uncertainty — the annotator-disagreement seam

Similarity intervals should widen by annotator disagreement, but
`data/llm_annotations/agreement_v1.parquet` does not exist yet
(`inter-model-agreement-check` is in flight). Following the `combat.load_agreement_bands`
precedent, this ships the **seam, not a stub**: an optional loader that returns
`None` when the file is absent, and a `ci_components` column on **every** similarity
row recording the truth — currently `sampling_only`, with `agreement_source =
absent:agreement_v1.parquet`. No disagreement magnitude is fabricated, and the
other task's file is not created. (Even when the band file lands, a cosine over the
whole fingerprint has no per-flag propagation model yet; the seam records the file
as `present_not_propagated` rather than silently claiming it was applied.)

---

## 8. Portraits

LLM era portraits (`data/eras/era_portraits.parquet`) — one Anthropic `claude-sonnet-5`
request per era, grounded in each era's z-scored fingerprint plus four
deterministically-selected exemplar quotes whose `(doc_name, para_idx, year)` triples
are stored in the `exemplar_ids` column of `era_portraits.parquet`, alongside the
portrait text. Manifest:
`data/eras/manifests/era-portraits-2026-07-21.json` (model, prompt hash, cost,
corpus fingerprint; its `notes` field points to the parquet for the exemplar IDs).
**Actual cost: $0.0595** (estimate had been $0.06; the ceiling
was $5.00). The portraits are the one atlas artifact allowed to carry a date. Spot-check:
the present-era portrait's phrases ("radical left", "fake news", hype/booster
adjectives) trace to the `opponents`/`hype` markers; the Civil War portrait's
constitutional/legalistic framing traces to its top axes (Civil Rights & Immigration
z = +2.54, self-reference +2.31, Constitutional Order & Governance +1.92) — no
external facts injected.

---

## 9. Determinism & provenance

All seven deterministic artifacts plus `eras_meta.json` are **byte-identical on
rerun** (verified: two runs, identical sha256). There is no wall-clock stamp in any
data artifact (the `combat_meta.json` precedent) — provenance identity is
`corpus_fingerprint`, and *when* it ran is git's job. A staleness guard compares the
live corpus fingerprint against `combat_meta.json`'s and warns loudly on mismatch
(currently: match). The portraits are the sole date-carrying, non-regenerated
artifact.

---

## 10. Verification boundary — which cells were checked

Per the repo's findings-note discipline, this note states what was re-derived from
the artifacts versus asserted:

**Re-derived from the parquets for this note (trust these):** every cosine in §1
(from `era_similarity.parquet`); the #1→#2 gap 0.264; the mutual-neighbor claim; the
axis-contribution drivers of the Civil War rhyme; the group-balanced robustness
(0.3386, rank 1 — emitted to `eras_meta.json → similarity.group_balanced_robustness`);
the presidency-grain Trump↔Truman / Biden↔John Adams pairings; the
Spearman 0.70 and the 6-of-9 / 0-of-9 adjacency counts (§3); the detrended off-diag
std 0.32; the combat max-abs-diff 0.0 and the 9-suppressed / 3-caution counts (§4);
the full §5 boundary table and the distinct-canonical recovery — 6 of 8 at bin grain,
4 of 8 at presidency grain (5 matching boundaries, but 1881+1882 both hit 1878);
the leave-one-out identity at every k and both grains (§6); the
present-era top raw-z markers; the actual portrait cost $0.0595.

**Asserted, not independently re-verified:** the *historical interpretation* of each
boundary (the "reading" column in §5 and the "national-division" gloss in §1) is
editorial, not a number — it is a plausible label on a measured shift, not a claim
the artifact can confirm. The portrait *texts* were spot-checked for traceability
(§8) but not exhaustively audited claim-by-claim. Readers wanting to re-derive any
number: the artifacts are `data/eras/*.parquet` and `data/eras/eras_meta.json`;
regenerate with `python -m presidential_profiles.eras`.
