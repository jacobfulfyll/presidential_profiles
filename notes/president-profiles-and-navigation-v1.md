# President Profiles and Navigation v1

Status: implemented and locally verified on 2026-09-07
Public contracts: `president-profile-v3` (unchanged) and `president-profile-context-v1` (new)

## Decisions

- Repair the shared navbar geometry and hover/pinned state machine in
  `src/presidential_profiles/expansion_site.py`; do not add page-specific navigation patches.
- Simplify the president directory to canonical chronology, one AI-topic preview, and
  progressive name-only search.
- Preserve `president-profile-v3`, Compare projections, and every existing analytical
  denominator. Agenda, Rhetoric, Similarity, and Compare remain document-owner based in this
  phase.
- Redesign legacy Evidence so its document-owner population, selection basis, counts, keyed
  receipts, speaker/owner attribution, and limitations are explicit; keep it as the final content
  section and collapse its detail by default.
- Add a separate actual-speaker Connections section: a field-preserving topic-ego projection
  from `SpeakerTopicNetworkBundle` and a governed
  `actual-speaker-invocation-network-v1` aggregation over the accepted speaker overlay. Render
  invocation results as one direction-switching top-five count summary rather than a relationship
  diagram.
- Replace profile Plotly bars with server-rendered HTML/CSS. The shared Plotly asset remains
  available to other pages.
- Publish new information through `president-profile-context-v1`; never add to, replace, or
  mutate the v3 JSON contract.

## Accepted baseline and protected state

Implementation began on `codex/summary-who-how-transition` with fourteen tracked,
user-owned paths already modified. Those changes must be preserved and merged around, never
cleaned, reset, discarded, reformatted, or overwritten. The accepted pre-implementation
verification was 2,820 tests with 64 existing warnings, 73 canonical HTML pages (74 physical
with `index_selfcontained.html`), 376 JSON files, 128 parsing inline scripts, clean
`git diff --check`, and Reggie Doctor at zero errors and zero warnings.

Protected surfaces are Story, Summary, Compare, Explore, the v3 profile payloads, frozen paid
annotations, and existing governed network artifacts. On protected pages, only regenerated
shared-navbar bytes may change. If the branch, accepted dirty baseline, protected projection
values, or governed source identities drift before publication, rerun preflight and refuse a
write when the new state cannot be merged safely.

## Shared navigation

The submenu bridge is an outer `.nav-submenu` at `top: 100%` with `padding-top: 7px`. The
existing white background, border, internal padding, radius, and shadow belong to an inner
`.nav-submenu-panel`. `hidden` and `aria-controls` remain on the outer wrapper.

Enhanced behavior keeps exactly one `openGroup` and one `openMode`, either `hover` or
`pinned`:

- mouse or hover-capable pen entry opens in hover mode;
- leaving the complete trigger/bridge/panel envelope closes only a hover-open group when focus
  is outside it;
- click, Enter, or Space opens a closed group pinned, promotes hover-open to pinned, and closes
  pinned;
- opening one group closes the other;
- focus within retains a group and focus leaving closes it;
- Escape closes and returns focus to the trigger;
- pointerdown outside closes; submenu anchors retain ordinary navigation;
- touch pointer entry is ignored and tap toggles pinned state.

Every transition synchronizes `hidden`, `.is-open`, and `aria-expanded`. Navigation state is
not stored in a URL, history entry, or storage. `.nav-enhanced` is added only after listeners
are installed. Before enhancement, both `:hover` and `:focus-within` expose submenu links.
Shared targets are at least 44 by 44 CSS pixels. Forced colors must preserve focus, current
links/groups, panel boundaries, and a non-color current-state underline. Existing hierarchy,
link order, routes, current state, sticky layering, reduced-motion behavior, and
`--global-nav-height` are immutable.

## President directory

Render one ordered list of all 45 presidents in canonical chronology. JavaScript never ranks
or analytically sorts the cards. Remove the redundant Dashboard, Compare, and “How AI labels
work” header links and remove the legacy issue badge.

Each card has this hierarchy:

1. 64px decorative portrait;
2. display name;
3. party and explicit `Corpus record YYYY–YYYY`;
4. speech count;
5. one `Top AI topic by source-document paragraph share` preview with percentage, or an
   explicit unavailable state;
6. percentile-rank support status.

Portrait `alt` remains empty because the linked name labels the card. Every portrait uses
`decoding="async"`; the first six are eager and the remaining 39 use `loading="lazy"`. Source
image bytes are unchanged. The grid is three columns at 960px and wider, two columns from
600–959px, and one column below 600px.

Name search is a progressive enhancement hidden until initialization. Without JavaScript all
cards remain visible. Matching is a case- and diacritic-insensitive substring comparison
against server-generated normalized display names. Chronology is preserved. Provide a 44px
Clear control, Escape-to-clear, a polite result count, and explicit no-results copy. Use safe
DOM properties and `hidden`; do not add ranking, party/era filters, URL state, or storage.

## Profile information architecture

The exact content order and sticky local-navigation order is:

1. Overview
2. Agenda
3. Rhetoric
4. Connections
5. Similarity
6. Signature speeches
7. Evidence, closed initially
8. Previous/next president navigation

Local targets are at least 44px. At phone widths they remain one contained, horizontally
scrollable row; they do not become a third sticky row and the page does not overflow.

Keep the document-owner Agenda, six-topic order, Overview values, five-speech percentile
rule, null thin-record ranks, all five Similarity instruments, signature methodology,
normalized Miller links, and corpus-record wording. Rename “Leading legacy issue” to
“Largest legacy issue share.” Permanently label the two populations:

- Core v3 profile measures: source documents assigned to the president.
- Connections: eligible paragraphs attributed to the actual speaker.

## Rhetoric

Replace both Plotly figures with semantic server-rendered bar lists. Retain the deterministic
high/low prose. Only finite, eligible-president percentiles receive a visual position; a null
thin-record rank has no position. Exact absolute values and ranks move into a closed native
disclosure titled `Exact values and eligible-president ranks`, with the semantic tables inside.
No unique value may exist only in a tooltip. Profile pages contain no Plotly script reference
or inline Plotly payload; other site pages keep the shared runtime.

## Legacy Evidence

The legacy issue model and document-owner denominators do not change:

- minimum four issue-positive paragraphs;
- eligible when share is at least 1.5 times the paragraph-weighted corpus baseline or the era
  difference is at least +0.75 percentage points;
- era peers are presidents whose first corpus year is within 24 years; when fewer than two
  peers exist, use all other presidents;
- order by the larger threshold-normalized absolute/era strength and then issue taxonomy
  order; cap at four.

Producer order is exact. Excerpt-bearing cards are never promoted ahead of a stronger
rate-only card. The visible Evidence heading and anchor precede one closed native disclosure
titled `Legacy issue evidence and audit receipts`; all Evidence population, method, card, and
receipt detail begins inside it. Once opened, show the first two cards and put any remainder in
a nested closed `More issue evidence` disclosure. This owner refinement supersedes the original
always-expanded section while leaving card and receipt production unchanged.

Every card contains:

- a plain-language claim distinguishing absolute emphasis, era-relative emphasis, both, or
  absolute-but-era-typical;
- issue paragraphs over all document-owned paragraphs, the percentage, source-document count,
  corpus baseline and multiple, and era difference;
- `Why shown` text naming the threshold or thresholds passed;
- one visible linked receipt and up to two additional receipts in a native disclosure;
- receipt key `(doc_name, para_idx)`, title/date/year, normalized Miller URL,
  source-document owner, actual speaker and eligible profile ID, cross-owner state,
  speaker eligibility/exclusion state, excerpt, and selection role;
- legacy stance and distinctive vocabulary in the audit disclosure, each labeled with its
  current method;
- a limitation visible whenever the Evidence disclosure is open: rates are document-owner
  based, speaker attribution is separate, and excerpts are deterministic audit examples rather
  than proof of representativeness, intent, influence, or policy success;
- thin-record copy when fewer than five source speeches are present.

The existing primary excerpt retains its current anchor, sentence selection, and keyed source.
Up to two additional receipts use different source documents, the same score, and deterministic
date/document/paragraph tie breakers. A rate-only card stays rate-only when no valid sentence
exists and states that no qualifying excerpt was selected. Enrichment is internal/context-v1;
the v3 serializer reproduces every existing value exactly.

## Actual-speaker invocation summaries

Build `actual-speaker-invocation-network-v1` from the accepted 2,032-row evidence and
`story_foundation.overlay_invocation_evidence()`. Validate candidates, classifications,
manifest, schema and rubric enums, classifier replay, target-status replay, source hashes, key
uniqueness, and evidence-span containment. Retain only resolved, analysis-eligible,
former-president, non-self actual-speaker rows.

The acceptance receipt is fixed: 1,447 former-president source rows, 48 unresolved keys, 156
ineligible rows, 27 actual-speaker reassignments, and 1,243 retained rows. Those rows represent
915 reference paragraphs, 405 speeches, 303 directed pairs, 38 source presidents, 44 targets,
and zero actual-speaker self edges. Valid 2025–2026 Trump-to-Biden rows remain later
corpus-speech evidence. Disclose that `target_status` follows corpus speech dates rather than an
independent legal-term calendar.

Each actual-speaker → target edge contains raw mentions, distinct reference-paragraph count,
distinct speeches, first/last speech date, complete fixed-order zero-filled function and stance
counts, evidence-row count, and evidence status. Both count maps reconcile to raw mentions.
Do not derive confidence, majority labels, normalized rates, or `mentions_per_100_speeches`.
Do not read the 341 document-owner edges, legacy regex/tone output, or old atlas.

Per direction, order edges by reference paragraphs descending, raw mentions descending,
distinct speeches descending, counterpart display order, then edge ID. Server-render both
directions as rhetoric-like semantic bar lists: `Most invoked former presidents` for outgoing rows
and `Most frequent later invokers` for incoming rows. After enhancement, a joined native
`Invoked` / `Invoked by` radio control places those alternatives in the same location; without
JavaScript the hidden switch stays absent and both lists remain visible in document order. Each
list shows the first five producer-ordered relationships. A single per-profile maximum maps
`reference_paragraph_count` to bar length across both directions, and every row prints exact
reference-paragraph, raw-mention, and distinct-speech counts.

Invocation has no SVG, ARIA tabs, hover/focus preview, `Show more relationships` control, or
pinned-edge interaction. A closed audit disclosure retains the leading relationship's complete
function/stance counts and selected receipts; a closed complete list retains every relationship,
and both CSV downloads retain every row. Empty copy says that no qualifying named
former-president invocation was found in this corpus, never that none occurred historically.
The browser never ranks, aggregates, or redraws invocation data.

## Actual-speaker topic ego network

Project only accepted all-corpus Level-1 rows from `SpeakerTopicNetworkBundle`. Use
`speaker_paragraph_share` directly for topic-node prominence and each president-topic edge
width. `topic_contribution_share` and `topic_scope_share` are text-only audit values. Never
create president-president edges, similarity or overlap scores, confidence, influence, or
summed prominence. Shared emphasis remains a bipartite president → topic → president path.

Default selection uses up to five `default_visible` focal edges ordered by speaker share, topic
paragraph count, topic appearance count, taxonomy order, and edge ID. For each topic choose up
to two default-visible supported peers with the same edge ordering plus president display order.
Select peers round-robin by focal-topic rank and peer rank; cap at ten unique peers.

Expanded selection uses up to ten supported focal edges, up to four supported peers per topic,
and the same round-robin process capped at twenty unique peers. Python precomputes all default,
expanded, and mobile node/edge IDs. Browser code never ranks or calculates relationships.

The supported-profile control is reversible: `Show expanded topic network` changes to
`Show compact topic network`, and activating it again restores the server-rendered default.
The first expansion loads the index and current shard; compacting and re-expanding reuse that
payload without another request. Hover and focus alone never change the graphic, selection, or
state. Each SVG topic node and its matching semantic-list control is an explicit 44px target:
click, Enter, or Space emphasizes only that topic's connected graph paths, activating it again
clears the state, and Escape clears it and restores focus. The earlier topic-edge audit readout and
the complete focal Level-1 topic record do not render; their governed fields remain unchanged in
the profile-context publication.

The deterministic layout fixes the focal president at center, topics on an inner ring, and peers
on an outer ring. Angle and distance have no analytical meaning. President nodes are uniform.
Topic area and edge width are direct bounded mappings of each individual
`speaker_paragraph_share`; multi-label shares are never summed. At 390px draw at most six peer
nodes in the precomputed order while retaining the complete semantic list.

The three thin presidents initially show counts and an explicit warning with no comparison.
`Show observed thin-record topics` may reveal up to five observed focal-topic edges with dashed
styling, without comparative ranking or peer nodes. It changes to `Hide observed thin-record
topics` and restores the empty thin default when activated again.

## Public profile-context contract

`docs/data/profile-context/` owns exactly 49 files:

- `index_v1.json`;
- `manifest_v1.json`;
- `presidents/<slug>_v1.json` for all 45 presidents;
- `actual_speaker_invocation_edges_v1.csv`;
- `actual_speaker_invocation_evidence_v1.csv`.

Schemas are `president-profile-context-index-v1`,
`president-profile-context-president-v1`, `president-profile-context-manifest-v1`,
`actual-speaker-invocation-edge-v1`, `actual-speaker-invocation-evidence-v1`, and
`president-profile-legacy-issue-evidence-v1`.

The index includes source identities, metric/policy definitions, 45-president and 17-topic
catalogs, support rows, all 595 supported comparison-topic edges, all 303 invocation edges,
shard receipts, counts, and a self-hash. Network records use exactly the field lists declared
below. Null identifiers are JSON null, never the string `nan`.

- President node: `president_profile_id, president_name, president_display_name, slug, party,
  display_order, node_type`.
- Topic node: `topic_id, topic_level, topic_label, topic_definition, topic_kind,
  parent_topic_id, parent_topic_label, display_order, node_type`.
- Topic edge: `edge_id, edge_type, scope_type, scope_id, topic_level,
  president_profile_id, topic_id, topic_paragraph_count, topic_appearance_count,
  eligible_president_paragraph_count, eligible_president_appearance_count,
  scope_topic_paragraph_count, scope_eligible_paragraph_count, speaker_paragraph_share,
  topic_contribution_share, topic_scope_share, president_support_status,
  edge_support_status, default_visible, evidence_receipt_count`.
- Topic receipt: `receipt_id, edge_id, doc_name, para_idx, appearance_id,
  actual_speaker_profile_id, actual_speaker, source_document_owner_profile_id,
  source_document_owner, cross_owner, story_era_id, story_era_label, speech_date,
  speech_title, source_url, evidence_excerpt, selection_role, selection_position`.
- Invocation edge: `edge_id, edge_type, source_president_profile_id,
  source_president_name, target_president_profile_id, target_president_name, raw_mentions,
  reference_paragraph_count, distinct_speeches, function_counts, stance_counts,
  first_speech_date, last_speech_date, evidence_row_count, evidence_status`.
- Invocation receipt: `candidate_id, edge_id, source_president_profile_id,
  source_president_name, target_president_profile_id, target_president_name, doc_name,
  para_idx, speech_date, story_era_id, story_era_label, speech_title, source_url,
  raw_mention, function, stance, evidence_span, rationale, quotation_status, target_status,
  rubric_version, evidence_status, annotation_speaker, source_document_owner_profile_id,
  source_document_owner, cross_owner`.

Each president shard includes every focal Level-1 edge and receipt; precomputed
default/expanded/thin topic selections; sorted incoming/outgoing invocation edge IDs and all
touching receipts; enriched legacy Evidence cards; source identity; and a shard self-hash.

Required parity:

- 45 president and 17 Level-1 topic nodes;
- 595 supported comparison edges in the index;
- all 700 focal Level-1 edges exactly once across shards;
- all 1,945 focal receipts exactly once across shards;
- 303 invocation edges exactly once in index and edge CSV;
- all 1,243 invocation rows once in the evidence CSV and twice across shards, once at source and
  once at target;
- 606 invocation edge references across shards;
- all 163 legacy cards retain scalar values and order; the 161 current primary quotations keep
  their keyed source and two cards remain honestly rate-only.

## Loading and enhancement boundaries

Server-render the default topic diagram and semantic topic controls, both invocation top-five bar
lists and complete-list disclosures, initial receipts, exact tables, limitations, thin/empty
states, and download links. Connections CSS is part of the profile stylesheet. Dynamically
import one page-agnostic `profile-connections-v1.js` module when Connections comes within 600px
of the viewport or first receives focus. Importing the module does not fetch JSON and it does not
redraw or analytically recalculate invocation summaries; it only reveals and operates the
direction radio control.

The first explicit topic expansion loads only `index_v1.json` and the current-president shard,
concurrently and once. Never automatically load either CSV, the 10.37MB full topic bundle,
Summary topic shards, or the old atlas. On module or shard failure retain all server content,
announce a polite failure, and offer one Retry. Network state is transient; only
`#connections`, `#invocations`, and `#topic-network` are stable. Do not use query parameters,
history entries, or storage.

Semantic list buttons and SVG topic nodes share the keyboard model. Topic focus and pointer hover
do not preview or select a relationship. Click, Enter, or Space emphasizes a topic's paths; Escape
clears the state and restores focus to its activating control. No topic readout is created.
Invocation uses ordinary lists, native disclosures, and a native two-option radio group, with no
chart interaction. Loading and
activation announcements are polite, and all controls and links are at least 44px with visible
3px focus.

The topic graph uses the full section width and the selected invocation direction uses one stable
panel at every viewport; at 390px the complete layout is one column. Labels wrap and
`documentElement.scrollWidth` equals viewport width. Reduced motion disables transitions and
smooth scrolling. Forced colors uses `currentColor`, system-color outlines, visible boundaries,
dashes/shapes, and no dependence on color.

## Determinism, security, and publication

Implement projection and rendering in new modules adjacent to `summary_topic_network.py`, then
wire preflight and publication through `site.py`. Build and validate the entire context twice in
memory before any `docs/` write and require byte-identical files. Use canonical sorted JSON,
stable arrays, `allow_nan=False`, deterministic CSV, source/file/self hashes, and manifest
receipts.

Validate in an isolated sibling candidate directory. Write all nonmanifest files first and the
manifest last. Atomically replace only `docs/data/profile-context/`, retaining the previous
complete directory until post-swap validation succeeds. On failure restore it. Stale cleanup is
strictly scoped to that owned directory.

Register `profile_topic_ego` against `actual_speaker_topic_presence` and register
`profile_invocation_bars` against the governed
`actual_speaker_invocation_paragraph_count` metric for invocation bar length.

Escape server output with `html.escape(..., quote=True)` as appropriate. Client rendering uses
`createElement`, `createElementNS`, `textContent`, and allowlisted attributes. Prohibit
data-driven `innerHTML`, template injection, `eval`, or unsafe URL assignment. Public source
links must normalize to the Miller presidential-speeches HTTPS prefix; reject missing, unsafe,
or mismatched sources.

Refuse public writes for stale/missing source hashes, schema/key/enum drift, classifier replay
drift, nonunique keyed joins, overlay-count drift, self/non-former rows, unknown president IDs,
topic parity/support drift, missing receipts, nonfinite values, unsafe URLs, budget failure,
nondeterministic bytes, or manifest/inventory mismatch. Never fall back to legacy invocation
edges, browser aggregation, old atlas data, or document-owner speaker substitution.

## Hard budgets

| Artifact | Raw maximum | Gzip maximum |
|---|---:|---:|
| Index | 750 KB | 90 KB |
| Largest president shard, including legacy Evidence | 500 KB | 75 KB |
| All 45 shards | 8 MB | 1.5 MB |
| Invocation edge CSV | 200 KB | 30 KB |
| Invocation evidence CSV | 1.25 MB | 225 KB |
| Enhancement module | 60 KB | 20 KB |
| Generated profile HTML | 125 KB | 30 KB |
| Directory HTML | 45 KB | 10 KB |

Initial profile loading before Connections is HTML plus the portrait only: no Plotly or data
request. The first expanded-network load, including module, index, and maximum shard, is at most
1.35MB raw and 190KB gzip.

## Acceptance and verification

Run focused source/projection/navigation/profile/security tests first with the repository's
x86_64 Python, then build the canonical site. Capture canonical owned hashes and run `--inline`;
the canonical outputs must remain byte-identical and the self-contained page must receive the
shared shell. Run generated-site/parity tests, the full suite (at least 2,820 plus new tests,
with no warnings beyond the accepted 64), inline and external JavaScript syntax checks, Python
compilation, `git diff --check`, and the portable Reggie doctor at zero errors/warnings.

Browser QA covers 1280×720, 1024×768, 768×1024, and 390×844 across representative shared-shell
pages, the directory, FDR, William Harrison, and Connections high-support/thin/empty/one-sided
fixtures. Exercise mouse bridge travel, first/second click, emulated coarse tap, keyboard,
Escape, outside dismissal, sticky overlap, current state, search/no-results/back navigation,
explicit topic pin/reset, expanded/compact and observed/hidden round trips, failure/Retry,
the direction-switching invocation top fives and their two-list no-JavaScript fallback, reduced
motion, forced colors, 200% zoom,
names/live output, long labels, request boundaries, page overflow, and console errors. Confirm
that topic focus and hover do not preview anything, graph topics are directly operable, removed
topic detail surfaces stay absent, and invocation exposes no chart or tabs.

Expected inventory is 73 canonical/74 physical HTML files, 423 JSON files, 49 files under
`profile-context`, and 129 inline scripts. Any deviation is explained and locked in tests before
acceptance. Only after browser acceptance may the durable README, TASKS, HANDOFF, HISTORY, and
generated documentation be synchronized around existing concurrent edits.

## Implementation receipt

The completed implementation preserves all 45 `president-profile-v3` payloads and the protected
Compare projection while adding the separate `president-profile-context-v1` publication. The
generated inventory is exactly 73 canonical HTML pages, 74 physical HTML files including
`index_selfcontained.html`, 423 JSON files, 49 files under `docs/data/profile-context/`, and
129 parsing inline scripts.

The measured profile-context outputs are within every accepted budget: the index is
748,641 raw / 77,842 gzip bytes; all 45 president shards total 7,705,843 / 1,293,201 bytes;
the largest shard is 360,319 / 49,401 bytes; the invocation edge CSV is
155,429 / 20,361 bytes; the invocation evidence CSV is 1,091,758 / 185,250 bytes; and the
enhancement module is 21,192 / 5,755 bytes. The largest generated profile is
111,875 / 22,482 bytes and the directory is 42,024 / 7,582 bytes. The maximum combined
module/index/shard expansion is 1,130,152 raw / 132,998 gzip bytes.

The final owner refinement's 49 changed-surface tests and the 172-test governed-profile suite
pass; the full repository suite
passes all 2,936 tests with the same 64 existing warnings. The
canonical and inline builds preserve canonical output byte-for-byte, and static JavaScript,
Python compilation, and whitespace checks pass. Targeted post-refinement in-app browser QA at
1280×720 confirms topic click/Enter/Space/Escape behavior, graph/list pressed-state parity, radio
click and arrow navigation, the unenhanced two-list fallback, exact Similarity name/value alignment,
and zero page-level overflow. The earlier complete v1 matrix remains the acceptance receipt for
compact/expanded round trips, thin show/hide, exact request boundaries, failure/Retry, and
one-sided invocation states. Reggie Doctor reports 0 errors, 0 warnings, and 4 informational
findings.
Nothing was deployed, staged, committed, pushed, switched, or cleaned.

## Explicit exclusions

- No Summary topic projection or shared-renderer changes for profiles.
- No profile-v4 or actual-speaker migration of Agenda, Rhetoric, Similarity, or Compare.
- No edits or regeneration of `data/llm_annotations`, `data/invocations_v2`, existing
  `data/networks`, or other frozen artifacts.
- No party/era directory filters, analytical browser sorting, dense all-president network,
  president-president topic edges, force layout, composite similarity, confidence, causal
  claims, or URL-persisted network state.
- No contemporary/serving-president rivalry extension beyond the accepted former-president
  contract.
- No hand edits to `docs/`; no stage, commit, merge, push, deploy, branch switch, or worktree
  cleanup without separate authorization.
