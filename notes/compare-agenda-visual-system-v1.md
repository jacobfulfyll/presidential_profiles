# Compare agenda visual system v1

Status: implemented and verified 2026-09-03. This is the durable Step 5
contract for Compare. It supersedes the dashboard-like Compare V2 presentation
without changing Profile V3, Story, Summary Topics, governed annotations, the
speaker/topic network, or legacy issue calculations.

## Public narrative and state

Compare has three ordered sections:

1. Rhetorical fingerprints.
2. Agendas in the speech record.
3. Evidence from the record.

The selector fieldset always exposes A, B, and C. A defaults to Abraham
Lincoln, B to Franklin D. Roosevelt, and C to None. A/B/C describe presentation
order only. Duplicate choices are disabled. A/B/C selection is serialized
through `a`, `b`, and optional `c` parameters
while preserving the hash. Old `agenda` and `topic` parameters are removed by
canonicalization because the focused agenda has no serialized subview. Invalid
A/B normalize deterministically; invalid or duplicate C becomes None. A lone
valid A retains the chronological adjacent-profile behavior.

The redundant Swap, Add/Remove third president, Copy link, and Reset controls
do not exist. Selection changes create browser-history entries; Back/Forward
restores the serialized state without a focus jump.

## Analytical boundaries

The corpus and AI rhetorical fingerprints retain the existing Profile V3
values, percentile support floor, axis order, and exact native-unit tables.
They remain source-document-owner records. Thin records retain exact values
but have no percentile shape.

The visible agenda copies complete-corpus Level 1 rows from the accepted
`actual-speaker-topic-network-v1` bundle. Its value is
`speaker_paragraph_share = topic_paragraph_count /
eligible_president_paragraph_count`. Topic-free eligible actual-speaker
paragraphs stay in the denominator. Multi-label shares are non-additive.
President support requires five actual-speaker appearances; edge support
retains the governed five-topic-paragraph and two-topic-appearance floor.

The Compare renderer exposes neither Level 2 topic rows nor the project's
earlier CorEx model. It also does not expose the former method bridge. These
data remain governed elsewhere in the project but are not part of the Compare
reading path.

## Focused agenda interaction

The agenda shows only the governed-order union of each selected president’s
three highest-share Level 1 AI topics. Ranking uses the existing
`speaker_paragraph_share`, with taxonomy order as the deterministic tie break;
it creates no new analytical value. Duplicate topics collapse into one row.
Each `Top 3 for …` badge names the president or presidents whose top-three set
caused that row to appear.

Every row has a shared 0–100 percent track with vertically staggered A/B/C
symbols. The stagger preserves each identity when values overlap and has no
analytical meaning. Exact value controls visibly pair the slot with a compact,
collision-safe president name (for example, `A · Lincoln` and
`B · F.D. Roosevelt`).

Selecting one president value opens a single blue detail directly beneath that
topic row. It repeats the exact share, topic-paragraph numerator, eligible
actual-speaker-paragraph denominator, topic-bearing appearance count, and
support state, followed by exactly one deterministic broad-topic example.
The example is the governed first receipt when available. The parent shard is
loaded only for that receipt; its fine-topic rows never render. Selecting the
same value or pressing Escape closes the detail.
## Compare publication

`src/presidential_profiles/compare_projection.py` builds the complete
publication in memory before the first generated-site write. The canonical
site builds it twice and requires byte identity. Every join uses canonical IDs
or declared compound keys. Publication is an atomic replacement of
`docs/data/compare/` only after validation.

The directory contains exactly 22 files:

- `agenda_index_v1.json`;
- 17 `agenda_<level-1-slug>_v1.json` parent shards;
- `evidence_index_v1.json`;
- `ai_topic_values_v1.csv` with 45 × (17 + 50) = 3,015 rows;
- `legacy_issue_values_v1.csv` with 45 × 16 = 720 rows;
- `manifest_v1.json`, which inventories the other 21 files.

This v1 publication inventory remains stable for deterministic provenance and
downstream compatibility. Compare itself renders only Level 1 AI cells, links
only the AI-topic CSV, and reads a parent shard only to retrieve the selected
Level 1 receipt. It does not render the shard's Level 2 cells or expose the
CorEx CSV.

Observed AI rows preserve Plan 3 counts, denominators, shares, appearance
support, and receipt counts. A zero is materialized only when the membership
universe is complete, the president denominator exists, and that keyed
relationship is absent. Cell states are supported, thin edge, thin president,
supported zero, thin zero, or unavailable. Unavailable quantitative fields are
null. The retained fine-topic display ceiling remains deterministic projection
metadata but has no current Compare visual consumer.

The evidence index contains source-document corpus footprint, the top three
existing adversarial entity groups, the top six existing invocation-v2 groups,
the top six existing distinctive terms with rank/z, and the top three existing
signature speeches with cosine score. The UI intentionally omits the repetitive
adversary/invocation receipt disclosures and prints vocabulary rank without its
z-score; the source values remain in the page-agnostic projection. Receipt
sampling is chronological first, lower-middle, and last after
date/document/paragraph sorting. It affects only the evidence sample.
Invocation-v2 empty records never fall back to the old regex method.

## Rendering and fallbacks

`compare-v3.js` owns selection, URL/history, lazy evidence, visible radar
mounting, and the selected Profile V3 download. `agenda-comparison-v1.js` owns
the focused broad display, parent-shard caching, exact readouts, one-example
details, support styles, and pin state. Both use element creation, fixed attributes,
`textContent`, and `replaceChildren`; data-driven raw HTML sinks are rejected
by source tests.

The initial default requests only Lincoln and Roosevelt portraits plus fixed
styles/controllers and deferred local Plotly as Rhetoric approaches. Agenda and
evidence JSON are absent from initial parsing. Requesting one topic example
loads its parent shard and caches it. The full speaker/topic
network JSON is never fetched.

Server HTML contains the default exact rhetoric tables, the focused five-topic
Lincoln–Roosevelt broad overview, evidence summaries, Profile links, and the
complete AI-topic CSV link. It contains no fine-topic or CorEx surface.
Selectors are disabled until hydration. With JavaScript disabled, noscript
styling reveals these agenda and rhetoric fallbacks, and no selector or
topic-example control promises a static-site update.

Slots use A blue circle/solid (`#2A78D6`), B dark ochre square/dashed
(`#9B6200`), and C green diamond/dotted (`#008300`). Thin states are hollow;
zero remains quantitatively positioned; unavailable is N/A. Direct exact-value
columns prevent staggered marks from becoming the only reading path. Controls
and marks are at least 44 × 44px. Reduced-motion and forced-color media rules
retain all information through symbols, outlines, text, and line patterns.

## Verification receipt

- Projection: 22 files; 3,015 unique AI CSV keys; 720 unique legacy CSV keys;
  17 Level 1, 50 Level 2, 16 legacy labels; seven exact unmapped topics; Plan 3
  field parity; legacy/Profile parity; receipt reconciliation; manifest bytes,
  gzip sizes, hashes, schemas, and deterministic two-build equality.
- Budgets: Compare HTML 155,518 bytes raw / 23,254 gzip; embedded model
  121,693 / 15,435; Compare controller 18,051 / 5,441; agenda module
  13,645 / 3,415; combined Compare CSS 12,218 bytes raw. Agenda index is
  430,187 / 35,905; largest parent shard remains within its 1.2 MB / 275 KB
  ceiling; evidence index is 653,729 / 148,131; AI CSV 460,412 / 49,332; the
  retained legacy CSV 51,448 / 10,695.
- Canonical build: 73 HTML pages and 224 JSON shards, with navigation, links,
  anchors, JSON, metrics, manifest, and downloads validated.
- Static assets: `node --check` passes both generated JavaScript files; source
  checks find no raw HTML rendering API.
- Tests: 53 focused Compare/integration tests and all 2,784 repository tests
  pass. The full suite reports 64 existing warnings.
- Interaction contracts: the default server fallback has five focused rows;
  the live Lincoln/FDR/Washington view has seven. Every selected record
  contributes exactly three memberships, duplicate rows collapse, and taxonomy
  order is stable. The old `topic` query was removed during live URL
  canonicalization. Clicking Lincoln's Foreign Relations value produced
  `7.09% · 47 topic paragraphs of 663 eligible paragraphs · 7 topic-bearing
  appearances` plus exactly one broad-topic receipt directly beneath that
  row. No fine rows, CorEx surfaces, or console errors were present.
- Live browser acceptance passed at 1280×720, 1024×768, 768×1024, and 390×844.
  Each viewport had zero guarded overflows, one visible detail, and agenda
  targets above the 44px floor. The mobile page had equal client and scroll
  widths after hydration.
- The in-app browser reports reduced motion and forced colors inactive and
  exposes no JavaScript/media emulation control. No-JavaScript, reduced-motion,
  and forced-color contracts remain covered by generated-source, DOM, and CSS
  assertions rather than live emulation.
- Captured `docs/index.html` and `docs/summary.html` hashes remained
  byte-identical across the canonical rebuild.

No deployment, staging, commit, push, frozen annotation edit, governed network
edit, Profile schema change, Summary behavior change, Story change, or
unrelated worktree cleanup is part of this implementation.
