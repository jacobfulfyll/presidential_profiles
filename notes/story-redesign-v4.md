# Story Page V4 expansion-era evidence and visual contract

Status: implemented for the 1816–1849 chapter on 2026-07-24. Story Page V2 and V3 remain
frozen completed records in `notes/story-redesign-v2.md` and `notes/story-redesign-v3.md`.
This note governs only the replacement expansion-era chapter.

## 2026-07-26 contextualization extension

The story era is now named **The Continental Republic**, with the subtitle
**Diplomacy, state-building, and continental conquest**. Its reusable
`Era Defined` screen is an additional graph, not a replacement for the
governed 1816–49 story matrix below the workspace.

The new graph uses four historically judged phases—1809–16, 1817–28,
1829–40, and 1841–49—and four independent, overlapping paragraph unions: war
and military survival; treaties and diplomacy; federal institutions; and
territory/continental power. They render as phase-aggregate step lines on one
shared 0–50% paragraph-share axis, with no invented interpolation between
years. Diplomacy is the thick continuous trunk; the other three threads rise
and fall around it. The territorial union publishes external acquisition and
Native removal and settlement as separate subthreads in the exact fallback.
All phase denominators include topic-free paragraphs, and the Treaty of Ghent,
Adams–Onís Treaty, Oregon Treaty, and Treaty of Guadalupe Hidalgo sit directly
on the diplomatic trunk as noncausal context markers. Every phase value and
subthread value remains available in a complete HTML table.

Story ownership now follows presidential transitions at accession boundaries:
an outgoing president's speech remains dated in its real year but belongs to
the regime that presidency closes. Jefferson's April 1809 message therefore
stays with the founding story; Madison supplies the 1809 expansion opening.
The same rule is declared for later accession boundaries. The 1850 historical
boundary receives no override.

## Historical claim and evidence boundary

The chapter follows a change in the formal presidential record after the founding: postwar
diplomacy and administration persist, federal finance and banking become occasions for domestic
combat, territorial conquest expands the agenda, and the acquired territories open directly into
the next era's slavery and sectional crisis.

The measurements are descriptive. Topic labels and the `enemy_naming` flag are AI
classifications over a curated Miller Center corpus, not historical ground truth. The named
events orient the periods and do not establish that an event caused a measured movement.
Paragraph attention does not measure policy importance, public opinion, government performance,
or consequences.

The section copy is fixed:

- Eyebrow: `1816–1849 · Expansion and federal conflict`
- Title: `The nation begins arguing with itself`
- Bridge: `The presidency began as paperwork. In this era, the paperwork became argument.`
- Movements: `Securing the postwar republic`; `Federal policy becomes domestic combat`;
  `Conquest creates a slavery question`

## Named artifacts

- Source paragraphs and text: `data/paragraphs.parquet`
- Keyed year attachment: `data/paragraph_issues.parquet`
- Speech metadata: `data/speeches.parquet`
- Frozen primary judgments:
  `data/llm_annotations/paragraph_annotations.parquet`
- Frozen secondary judgments:
  `data/llm_annotations/paragraph_annotations__opus4-8.parquet`
- Frozen primary entities:
  `data/llm_annotations/paragraph_entities.parquet`
- Frozen taxonomy: `data/llm_annotations/taxonomy_v1.json`
- Deterministic derived table:
  `data/expansion_story/period_metrics.parquet`
- Deterministic provenance sidecar: `data/expansion_story/meta.json`

`src/presidential_profiles/expansion_story.py` is the derived-layer source of truth. It is a
pure local `$0` build and never writes below `data/llm_annotations/`.

## Periods and denominator

The two aligned panels share eight fixed columns:

1. 1816–1820
2. 1821–1825
3. 1826–1830
4. 1831–1835
5. 1836–1840
6. 1841–1845
7. 1846–1849
8. 1850–1854, visibly marked as a next-era preview

Every denominator contains every unique `(doc_name, para_idx)` paragraph key in the period,
including paragraphs assigned no topic. Numerators also count unique paragraph keys. Topic lists
are normalized through the frozen taxonomy's casefold map; an unknown label raises, and repeated
case variants within one paragraph are de-duplicated. All co-derived paragraph tables are joined
on `(doc_name, para_idx)` with one-to-one validation and explicit key-set completeness checks.
No row-position join is permitted.

## Exact topic definitions

| Display row | Exact level-2 labels |
|---|---|
| Diplomacy and treaties | `Treaties, Diplomacy & International Arbitration` |
| Federal finance | `Public Debt, Revenue & Treasury Finance` |
| Tariffs | `Tariffs, Reciprocity & Navigation Laws` |
| Banking | `National Bank & Banking Crises` |
| Territorial expansion | De-duplicated union of all five labels in the two subrows below |
| ↳ External acquisition | `Relations with Spain, Mexico & Territorial Claims`; `Territorial Organization, Statehood & Insular Governance`; `Mexican War` |
| ↳ Native removal and settlement | `Indian Affairs, Removal & Allotment`; `Public Lands & Homestead Settlement` |
| Slavery and sectionalism | `Slavery, Emancipation & Sectionalism` |

The territorial parent and its two indented subrows overlap and are not additive. Composite
intervals are calculated from the paragraph-level union indicator; marginal topic intervals are
never combined.

## Graph contract

The primary Plotly figure contains two vertically aligned panels:

- An eight-row topic heatmap on one blue sequential scale fixed at 0–60% for every cell,
  including the 1850–54 coda. Every cell prints one decimal place. Text switches from dark to
  white as the cell darkens.
- A narrow bar strip showing the percentage of all period paragraphs with
  `enemy_naming=True`. Its y-axis is fixed at 0–40%; every bar prints its value; bars are not
  connected or stacked.

The coda column is outlined and shaded across both panels and labeled `Next era preview`.
Confidence uses symbols rather than color. The only adversary-strip annotations are the two
supported peaks:

- 1831–35: institutions are 53% of adversarial entity mentions; the derived recurring names are
  `Bank of the United States` and `Senate`.
- 1846–49: nations are 77% of adversarial entity mentions; the derived recurring name is
  `Mexico`.

The prose calls these episodic spikes, not a rising trend.

Each topic cell and adversary bar exposes its period, row, share, speech-clustered interval,
paragraph and speech support, `ci_status`, exact constituents, and annotator-disagreement status.
A native HTML table follows the figure and contains all 72 published cells with values,
intervals, support, and confidence status. It remains complete without Plotly, hover, color, or
JavaScript.

Below the figure's minimum readable width, only the figure and fallback table scroll locally.
The page itself must have no horizontal overflow. Printed values remain present at 390 CSS
pixels, focus styles remain visible, and reduced motion removes story entrance transitions.

## Uncertainty

Each topic union and the adversary rate uses the existing five-year uncertainty policy:

- 500 percentile-bootstrap draws;
- deterministic seed `20260721`, scoped by period start;
- whole speeches (`doc_name`) resampled within the period;
- all paragraphs belonging to each sampled speech retained;
- 2.5th and 97.5th percentiles;
- existing speech-cluster and paragraph-support `ci_status` thresholds;
- `interval_unresolvable` preserved as a separate per-cell gate.

On the paragraphs labeled by both models, the layer calculates the primary/secondary half-range
for the same period and the same paragraph-level union indicator. When at least 50 paired
paragraphs and a sampling interval are available, that half-range widens the sampling bounds.
It never narrows them. Thin paired support and a missing sampling interval remain distinct
machine-readable statuses.

The metadata records file fingerprints, periods, topic unions, denominator and normalization
rules, bootstrap settings, gates, paired-model coverage, validated receipt keys, derived
character annotations, and the complete artifact schema. It carries no wall-clock timestamp, so
the table and sidecar reproduce byte-for-byte.

## Evidence receipts

Three compact receipt groups are keyed by exact `doc_name` and `para_idx`, and every displayed
excerpt is validated as an exact substring of the frozen corpus paragraph:

- Monroe's 1817 First Annual Message: postwar diplomacy, land revenue, Native land acquisition,
  and western settlement in one administrative inventory. The 1823 Monroe Doctrine remains a
  noncausal event marker.
- Jackson in 1832: the Bank Veto and Nullification Proclamation are contrasted with the
  three-paragraph Message Regarding Indian Removal. The artifact re-derives enemy naming in
  30/56 Bank Veto paragraphs, 31/60 Nullification paragraphs, and 0/3 removal-message
  paragraphs. This is a significant, temporary, selectively combative tonal rupture; the
  administrative register does not make removal less coercive.
- Polk in 1846–48: the War Message names Mexico and war; the Message Regarding Slavery in the
  Territories supplies the documentary hinge into sectional crisis.

No sentence-length, hedging, hype, doom, zero-sum, or party-attack series enters this chapter.

## Acceptance tests

The implementation is accepted only when:

1. `tests/test_expansion_story.py` covers exact periods, complete keyed joins, taxonomy
   normalization, duplicate removal, composite union de-duplication, topic-free denominators,
   speech-cluster resampling, deterministic seed behavior, confidence and paired-model gates,
   exact receipts, Jackson character metrics, entity annotations, and the complete HTML fallback.
2. Story tests require eight topic rows and eight columns, the 0–60% scale, printed values, the
   shaded coda, aligned 0–40% adversary bars, episodic language, exact title and bridge,
   noncausal events, and the absence of the superseded three-line/founding-baseline figure.
3. Real-artifact checks re-derive:
   - 1831–35 banking: 23.3%
   - 1831–35 adversary naming: 36.5%
   - 1846–49 territorial expansion: 56.4%
   - 1846–49 external acquisition: 49.2%
   - 1846–49 adversary naming: 27.3%
   - 1850–54 slavery and sectionalism: 37.7%
4. The metric registry maps `expansion_story` to paragraph share and confidence interval, and
   the obsolete `expansion_delta` key is absent.
5. The deterministic layer reproduces byte-for-byte; the full site build validates links,
   anchors, JSON shards, and metric registration; generated JavaScript and Python compile; the
   full pytest suite and `git diff --check` pass.
6. Desktop, mobile, normal-motion, reduced-motion, hover, keyboard, fallback, coda, confidence
   symbols, local scrolling, and page-overflow behavior pass browser QA. Nothing is deployed.
