# Summary actual-speaker topic relationship migration v1

Status: implemented; interaction revised by owner direction on 2026-09-02<br>
Consumer: Summary section 05<br>
Analytical source: `actual-speaker-topic-network-v1` only<br>
Public title: **Presidents and their recurring topics**

## Owner-directed download-only ending

The section now ends immediately after the topic-gravity field with one link:
**Download the exact recurring-topic table**. The former `How to read this
topic-gravity field` disclosure, embedded 42-by-17 HTML table, complete-network
and all-edge downloads, and local Data Quality, Methods, and Feedback links no
longer render. The no-JavaScript and load-failure copy point to the CSV instead
of a removed fallback table.

The download is a deterministic 427-row CSV containing only supported,
default-visible all-corpus Level 1 relationships. It preserves all 29 Plan 3
edge fields and values without recalculation. The accepted Plan 3 bundle and
its schemas are unchanged; the Summary public projection advances from 19 JSON
files to 20 total files (the same 19 JSON files plus the CSV).

The former Extreme Speeches evidence-card appendix is also removed. Topics is
now Summary's final section, and the exact CSV link sits 8px above the existing
source/method footer on desktop and mobile. The footer divider and content are
unchanged. A legacy `index.html#records_appendix` link resolves to Summary
Topics rather than a removed fragment.

## Owner-directed gravity-field revision

This revision supersedes the original single-president/single-topic tabs, ego
diagram, and statements below that exclude every kind of force layout. The
projection source, provenance, accessibility, loading, and publication
contracts remain in force, subject to the later download-only ending above.

- Users select one to eight broad topics from the governed display order.
- Selected topics become fixed anchors around a deterministic field. There is
  no continuously running or stochastic force simulation.
- Supported presidents with a qualifying selected-topic relationship appear
  inside the anchors. Desktop retains all 42 supported presidents when the
  selection connects them; the 390px field caps only the diagram at 18 while
  the ranked list retains every qualifying president.
- A president begins at the weighted center of its selected anchors using the
  existing `speaker_paragraph_share`. Deterministic display stretch and
  fixed-iteration collision spacing keep nearby bubbles and anchors apart;
  that renderer displacement and absolute distance carry no additional
  analytical meaning.
- President bubble area encodes the sum of `topic_paragraph_count` across the
  selected topics. The UI calls this **selected-topic paragraph memberships**,
  never unique paragraphs, and visibly repeats that multi-label memberships
  overlap and are non-additive.
- Faint connections retain the original bounded line-width encoding using only
  `speaker_paragraph_share`; focusing a president, topic, or relationship
  reveals the relevant lines.
- A visible key explains portrait area and reports the largest membership total
  in the current topic selection. Pointer hover, keyboard focus, and pinning
  expose a compact live readout with exact president/topic context.
- An optional native selector lets users compare up to eight named presidents.
  No selection means all qualifying presidents; subset selection filters the
  field but preserves the full-field portrait-area scale.
- The topic buttons and president picker are horizontal, with topics above
  presidents, and the graph spans the full Summary content width.
- `Choose topics`, selection count, and Clear share one heading line. Topic
  controls use a single horizontal scroller. Compare presidents uses one line;
  after selection, Show all presidents and removable president controls occupy
  a second horizontal scroller.
- Topic sets are sorted into governed taxonomy display order before layout.
  Selection order therefore cannot alter topic anchors or president positions.
- Summary exposes only the 42 supported presidents and 427 recurring Level 1
  relationships. The lower-support reveal, thin president options, bottom exact
  relationship readout, Level 2/evidence UI, and the accessible HTML table are
  removed. The governed bundle retains every row.
- The control descriptions, recurring-view explainer, visible status sentence,
  and local taxonomy-provenance note are removed to minimize the distance to the
  chart. Loading/failure announcements remain screen-reader-only.

The governed Plan 3 bundle is byte-identical to the accepted migration. The
Summary projection adds only the field-preserving recurring CSV; this introduces
no new score, denominator, schema, or join.

## Decision

The actual-speaker topic network becomes Summary’s fifth analytical section,
replacing rather than supplementing the former Audit chapter. It asks:

> Which broad topics recur in the paragraphs actually attributed to each president?

The section is a hybrid relationship browser:

- no force-directed overview or all-network hairball;
- an initial 17-topic overview and president/topic selectors draw no edges;
- selecting a president or topic opens a stable ego view;
- a ranked relationship list and complete accessible table carry exact values;
- Level 2 topics and evidence load on demand;
- edge width encodes only `speaker_paragraph_share`;
- the Plan 3 `SpeakerTopicNetworkBundle` is the sole analytical source.

The former Summary Audit matrix is removed. The later owner revisions above
replace the original compact link set with one recurring CSV and remove the
Extreme Speeches appendix. Migrating the larger audit is reserved for the later
Data workstream.

## Placement and copy

Summary order is:

1. Voice
2. Conflict
3. Time
4. Emotional register
5. Topics
6. No post-story appendix; Topics closes Summary

The sticky Summary route becomes:

- Congress → public
- Enemies by president
- Tomorrow + yesterday
- Hope ÷ doom
- Topics by president

Section copy:

- Kicker: `05 · Topics`
- Heading: `Presidents and their recurring topics`
- Introduction: `Choose a president or broad topic to see where AI-assigned
  topics appear in paragraphs actually attributed to each president. The view
  describes this speaker-audited corpus; it does not measure importance,
  intent, influence, or policy success.`
- Population receipt: `32,531 eligible actual-president paragraphs · 45
  presidents · 17 broad topics`

The old multi-column Audit matrix must not appear in this section.

## Locked scope and filters

Summary v1 is fixed to:

- `scope_type = corpus`
- `scope_id = all-corpus`
- `topic_level = level1`
- `president_support_status = supported`
- `default_visible = true`

Story-era switching is out of scope. All 45 presidents remain discoverable:

- 42 supported presidents appear normally;
- James A. Garfield, William Harrison, and Zachary Taylor remain in the
  selector with a visible `Thin record` suffix;
- a thin president first shows the support receipt, then offers an explicit
  `Show observed thin relationships` action;
- supported presidents start with default-visible edges only;
- `Show all observed relationships` reveals `thin_edge` rows without changing
  their Plan 3 labels;
- all observed thin rows remain in the projection and exact table.

The initial state shows the 17 Level 1 topic controls in governed display order
and prompts for a president or topic. It draws no relationship edges.

## Interaction state machine

Renderer state is:

- `mode`: `president` or `topic`
- `selectedNodeId`: nullable president/topic ID
- `detailLevel`: `level1` or one selected Level 1 parent’s `level2`
- `includeThin`: false initially
- `previewEdgeId`: transient pointer/focus edge
- `pinnedEdgeId`: persistent selected edge
- `evidenceEdgeId`: nullable open evidence panel
- `returnFocusElement`: trigger that opened evidence

### Mode and selection

`By president` and `By topic` are real ARIA tabs. Arrow Left/Right, Home, and
End follow the tabs pattern. Switching mode clears preview, pin, Level 2, and
evidence state.

The president selector is native and includes all 45 presidents in governed
order. President mode centers the selected president and shows all qualifying
Level 1 relationships, up to the full 17.

Topic controls and the native topic selector activate topic mode. Topic mode
centers the topic and draws no more than the 12 highest
`speaker_paragraph_share` presidents, breaking ties by governed president
order. The ranked list retains every qualifying relationship.

Pointer hover and keyboard focus produce the same preview and exact readout.
Enter, Space, or click pins an edge. A pin remains selected after pointer or
keyboard focus moves elsewhere.

### Level 2 and evidence

Level 2 is available only after a Level 1 context exists. `Show finer topics
in [Level 1 label]` loads exactly that parent’s deterministic shard.

- President mode shows that president’s child-topic relationships.
- Topic mode first lists the parent’s child topics; selecting a child shows
  its connected presidents.
- `Back to broad topics` restores the prior Level 1 focus and focus position.

`Inspect audit examples` opens a non-modal evidence panel, moves focus to its
heading, and shows first, lower-middle, and last receipts where available.
Closing returns focus to the trigger. Evidence never opens on hover alone.

Escape acts successively:

1. close evidence;
2. clear a pinned edge;
3. clear the selected node and return to the initial overview.

Interactive state is not serialized. The only route is `#summary-topics`.

### No JavaScript

The introduction, 17 topic links, limitations, downloads, and complete exact
table remain server-rendered. Topic links jump to matching table headers. A
short `<noscript>` message explains that the focused diagram needs JavaScript.

## Layout and visual encoding

Desktop uses controls, stable SVG ego view, and exact readout/ranked list in
three columns. Evidence sits below the stage. Tablet uses controls across the
top with SVG and ranked list side by side where space permits. At about 390px,
the component stacks and the SVG shows the center plus at most six connected
nodes; the full ranked list remains immediately available.

Requirements:

- no page-level horizontal overflow;
- minimum 44×44 CSS-pixel targets;
- long president, topic, speech, and owner names wrap;
- deterministic radial/grid positions only;
- president nodes are circles and topic nodes rounded rectangles;
- Level 2 topics use a double/inset outline;
- supported edges are solid;
- thin edges/presidents use a dashed edge and visible `Thin record` text;
- selection uses a dark copper outline and increased contrast;
- keyboard focus has a separate high-contrast ring;
- party color is not used.

Edge width is a bounded linear 1.5–7 CSS-pixel scale over only
`actual_speaker_topic_presence → speaker_paragraph_share`. Exact percentages
are printed. `topic_contribution_share` and `topic_scope_share` are secondary
text values and never affect size, distance, or color. Position has no
analytical meaning and never enters Plan 3 data.

Reduced motion disables movement. No layout continuously simulates.

## Exact relationship and evidence contract

A pinned edge exposes:

- president;
- topic label and level;
- all-corpus scope;
- topic paragraph and appearance counts;
- eligible-president paragraph and appearance denominators;
- scope topic and eligible paragraph denominators;
- headline `speaker_paragraph_share`;
- secondary `topic_contribution_share` and `topic_scope_share`;
- president and edge support;
- default-visible state;
- topic-free denominator policy;
- multi-label non-additivity warning;
- up to three deterministic evidence receipts.

Receipt cards show selection role, speech date/title, actual president speaker,
source-document owner when different, Story era, excerpt, and Miller Center
source link. They are called **deterministic audit examples**, never
representative examples, validation, or proof.

The short definition and principal limitation remain visible. Formulas, full
policies, secondary measures, and receipts may use disclosures. All untrusted
strings use escaped server output or DOM `textContent`/safe attributes; no
data-driven raw `innerHTML` is allowed.

## Public projection

Summary must not fetch the 10.37 MB `network_v1.json`. It publishes:

`docs/data/summary-topic-network/`

Schemas:

- `summary-actual-speaker-topic-network-index-v1`
- `summary-actual-speaker-topic-network-topic-v1`
- `summary-actual-speaker-topic-network-manifest-v1`

Files:

- `index_v1.json`
- 17 `topics/<stable-topic-id>_v1.json` shards
- `manifest_v1.json`

Total: 19 JSON files.

The index contains Plan 3 source identity and hashes, the fixed scope, policies,
metric definitions, all 45 president nodes, all 17 Level 1 nodes, corpus
support, all 700 observed corpus Level 1 edges, and topic-shard filenames,
hashes, and sizes.

Each Level 1 shard contains its Level 2 children, corpus Level 2 support and
edges, evidence for the parent Level 1 and child Level 2 edges, source identity,
and a self-hash.

This is a field-preserving projection, not a new analytical contract. It may
filter scopes and columns, but it must not recalculate speakers, topics,
shares, support, denominators, or evidence. Validation compares every selected
row and value to the loaded Plan 3 bundle and fails before page writes for:

- missing or extra rows;
- changed counts or measures;
- unknown IDs;
- stale source hashes;
- missing evidence;
- incorrect shard ancestry;
- manifest/file mismatch;
- non-finite values.

Loading behavior:

- render the shell and exact CSV download link during the normal build;
- load `index_v1.json` within 600px of the viewport or on focus/navigation;
- fetch one Level 1 shard only for Level 2 or evidence;
- cache shards in memory;
- announce loading through `role=status`;
- leave the exact CSV download usable on failure.

Budgets:

- index: at most 550 KB raw / 80 KB gzip;
- largest shard: at most 1.2 MB raw / 275 KB gzip;
- renderer JavaScript: at most 50 KB raw;
- Summary HTML increase before Audit removal: at most 30 KB raw when feasible
  without compromising the download fallback; final page should
  not grow materially after Audit removal;
- cached updates: under 100 ms desktop / 150 ms at 390px in the test
  environment;
- no shard fetch before Level 2 or evidence.

With the 19 new JSON files and no HTML route, canonical inventory is expected
to be 73 HTML / 204 JSON after the build confirms it.

## Metrics and implementation surfaces

Plan 3 schemas remain unchanged. Existing metric definitions are sufficient:

- `actual_speaker_topic_presence`
- `actual_speaker_topic_contribution`

Add only:

`summary_topic_relationships → actual_speaker_topic_presence`

Do not add a generic network metric.

Implementation surfaces:

- this note and `.codex/knowledge/README.md`;
- reusable `summary_topic_network.py` projection, validation, manifest,
  deterministic writing, exact-table rendering, and CLI checks;
- page-agnostic `topic-relationship-browser-v1.js` and scoped CSS;
- narrow `site.py` integration that passes the already loaded Plan 3 bundle,
  builds and validates the projection before page writes, replaces Audit, and
  adds the shell/assets;
- one chart registry mapping in `metrics.py`;
- focused `tests/test_summary_topic_network.py` plus narrow Summary, metric,
  and site-validation regressions.

No analytical joins or denominator calculations belong in renderer code.

## Acceptance matrix

### Projection and provenance

- Plan 3 is loaded before generated-site writes.
- stale/invalid inputs fail before output;
- projection rows exactly match the bundle;
- scope is corpus/all-corpus only;
- index contains 45 presidents, 17 Level 1 topics, and 700 observed edges;
- default filter yields 427 edges and 42 supported presidents;
- three thin presidents remain discoverable;
- 17 shards cover all 50 Level 2 nodes exactly once;
- evidence reconciles with Plan 3;
- manifest inventory, hashes, sizes, and source identity validate;
- two builds are byte-identical;
- Plan 3 governed/public artifacts remain byte-identical.

### Interaction and wording

- initial view draws no relationship hairball;
- president and topic modes apply the approved diagram/list limits;
- thin controls reveal without relabeling;
- hover/focus parity and Enter/Space/click pinning;
- successive Escape behavior;
- mode reset, Level 2 return, evidence focus return, loading/failure fallback;
- no URL/history pollution;
- `speaker_paragraph_share` is the only edge-width measure;
- exact numerators/denominators reconcile;
- topic-free and multi-label language is visible;
- no importance, influence, confidence, similarity, intent, causality,
  policy-success, or representation claim;
- evidence is called deterministic audit examples.

### Accessibility and regression

- correct headings, tabs, labels, status regions, SVG title/description;
- stable keyboard order and visible focus;
- exact table covers the projected Level 1 edge set;
- keyboard-reachable evidence and focus restoration;
- reduced motion, no continuous layout, 44px targets, and no page overflow at
  390px;
- no unsafe data-driven `innerHTML`;
- no-JavaScript content remains complete;
- Voice, Conflict, Time, and Emotional register remain;
- Audit and `What survives` disappear;
- Extreme Speeches is absent and Topics is the final Summary section;
- Story, Compare, Profiles, Explore, Issues, Data Quality, and Methods remain
  structurally unchanged;
- Story v5/v9/v11, foundation, and Plan 3 remain unchanged;
- full repository tests and generated scripts pass;
- final test total is reported from pytest rather than predicted;
- canonical build confirms 73 HTML / 204 JSON.

## Verification order

```bash
git status --short

arch -x86_64 .venv/bin/python -m presidential_profiles.foundation_audit --check
arch -x86_64 .venv/bin/python -m presidential_profiles.story_foundation --check --site-dir docs
arch -x86_64 .venv/bin/python -m presidential_profiles.speaker_topic_network --check --site-dir docs

~/.codex/scripts/reggie/reggie_doctor.sh --repo /Users/jacobpress/Desktop/Projects/presidential_profiles

arch -x86_64 .venv/bin/python -m pytest -q tests/test_summary_topic_network.py
arch -x86_64 .venv/bin/python -m presidential_profiles.summary_topic_network --verify-reproducible
arch -x86_64 .venv/bin/python -m presidential_profiles.summary_topic_network --check --site-dir docs

arch -x86_64 .venv/bin/python -m pytest -q \
  tests/test_summary_page.py \
  tests/test_summary_topic_network.py \
  tests/test_speaker_topic_network.py \
  tests/test_story_foundation.py \
  tests/test_metric_registry.py \
  tests/test_site_validation.py

arch -x86_64 .venv/bin/python -m pytest
arch -x86_64 .venv/bin/python -m compileall -q src tests
arch -x86_64 .venv/bin/python -m presidential_profiles.site

arch -x86_64 .venv/bin/python -m presidential_profiles.story_foundation --check --site-dir docs
arch -x86_64 .venv/bin/python -m presidential_profiles.speaker_topic_network --check --site-dir docs
arch -x86_64 .venv/bin/python -m presidential_profiles.summary_topic_network --check --site-dir docs

node scripts/validate_inline_js.mjs docs
node --check docs/assets/issues-v2.js
node --check docs/assets/topic-relationship-browser-v1.js

git diff --check
git status --short

~/.codex/scripts/reggie/reggie_doctor.sh --repo /Users/jacobpress/Desktop/Projects/presidential_profiles
```

Browser QA covers desktop and 390×844, including eight-topic enforcement,
president filtering and pinning, Escape reset, CSV availability, console,
overflow, and the settled deterministic layout.

## Risks and rollback

Mitigations are structural: never draw the full network; give layout distance
no meaning; lazy-load the compact index rather than `network_v1.json`; enforce
raw/gzip budgets; retain thin records in the projection; encode only
`speaker_paragraph_share`; derive the diagram and CSV from the accepted Plan 3
bundle; escape public strings; hash Plan 3 identity; and replace Audit to
contain narrative length.

Rollback removes the projection module, 19 JSON files, recurring CSV, renderer assets,
Summary Topics section/route, metric mapping, tests, and this plan route, then
restores Audit and rebuilds. Inventory returns to 73 HTML / 185 JSON. Plan 3
governed and public artifacts remain valid.

## Out of scope and later reuse

Out of scope: Story-era switching, continuously simulated or stochastic force
layouts, president similarity,
president-president edges, party encoding, new analytical calculations or
schemas, genre scopes, Compare/Profile migration, Extreme Speeches redesign,
full Data Quality audit migration, legacy Network Atlas removal, deployment,
staging, commit, push, and unrelated cleanup.

The renderer accepts a validated projection, scope, initial node, and display
limits without knowing its host page. Later consumers may use a president ego
view on Profiles, selected-president agenda comparison, or a topic-centered
Explore view. A new score, denominator, scope, genre restriction, or era system
requires a versioned analytical extension first.
