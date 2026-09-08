# Actual-speaker topic network v1

Status: implemented reusable contract; no production-page placement<br>
Contract: `actual-speaker-topic-network-v1`<br>
Implementation: `src/presidential_profiles/speaker_topic_network.py`<br>
Governed layer: `data/speaker_topic_network/`<br>
Public downloads: `docs/data/speaker-topic-network/`

## Analytical question

Within the speaker-audited Miller Center corpus, which canonical AI-assigned
topics appear in paragraphs actually attributed to each president, and what
share of that president's eligible paragraph record do those paragraphs
represent, either corpus-wide or within one accepted Story era?

This is descriptive, non-causal corpus evidence. Topic assignment and network
proximity are not historical validation, model confidence, presidential
intent, importance, influence, causality, policy success, or representation of
the public or presidency as a whole.

The v1 contract does not appear on Story, Summary, Compare, Profiles, Explore,
or any other HTML page. It publishes governed data and downloads for later
consumers. `network-atlas-v2` remains a separate legacy document-owner network
and must not be used for named-president claims. Era Echoes remains an
invocation-conditioned subset and is not a substitute for this whole-corpus
contract.

## Terminology and population

- **Actual paragraph speaker** is the attributed president supplied by the
  accepted Story foundation. The network never infers the president from the
  source-document owner.
- **Source-document owner** is preserved separately on paragraph memberships
  and evidence receipts.
- **Actual-speaker appearance** is exactly
  `(doc_name, attributed_speaker_profile_id)`. A stable `appearance_id` hashes
  that complete identity.
- **Source-document corpus** describes document-owned chronology and other
  aggregate corpus surfaces. It is not the population for a named-president
  edge.
- **Speaker-audited paragraphs** are foundation paragraphs marked eligible for
  actual-president claims. The 2,863 excluded foundation rows remain in the
  input audit totals but produce no membership or edge.
- **Story era** is the identity already carried by the accepted foundation.
  The network neither date-bins nor invents an era definition.

At the accepted baseline, the foundation contains 35,394 retained paragraphs:
32,531 are eligible, 2,863 are excluded, and 296 eligible presidential
paragraphs are cross-owner records. The eligible population forms 1,054
actual-speaker appearances for 45 presidents across the nine existing Story
eras.

Mixed or multiple speakers, joint/shared speakers, unresolved speakers,
non-president speakers, and scaffolding are excluded by the foundation's
eligibility decision. Cross-owner presidential paragraphs remain under the
actual speaker and the corresponding actual-speaker appearance. No speaker is
re-inferred in this layer.

## Inputs and keyed join

The reader accepts one validated `StoryFoundationBundle` and resolves the
annotation ledger's active `materialized/current` pointer. It verifies the
pointer syntax, target existence, generation manifest row count, and the
materialized generation's content-addressed artifact root. The topic
projection is selected by the promotion contract:

- `promotion_channel = primary`
- `label_type = topics`
- `event_role = value`
- `value_kind = list`

The audited active generation is
`5b3f0e252c63a3b6cf3cba5d5e84e23def38eaa8edf29b89af827c92e5bce307`.
The implementation resolves the pointer rather than using that identifier as a
data path. The identifier is an acceptance guard for this pinned v1 baseline;
a later authorized input migration must update the versioned acceptance
contract rather than silently accepting drift.

Before eligibility filtering, foundation and promoted topic rows must each be
unique on `(doc_name, para_idx)`, their full 35,394-key sets must be identical,
and the join must validate one-to-one. Rows are sorted after joining, so source
row order cannot affect the result. Missing, duplicate, or extra topic keys
fail before any governed or generated-site write.

The active-generation identity must equal the annotation generation recorded
by the accepted foundation. The complete selected promoted record projection,
pointer, generation manifest, full `current_labels.parquet`, taxonomy, Story
foundation metadata, and foundation source artifacts are hashed in provenance.

## Topic normalization and multi-label semantics

The only taxonomy implementation is the repository's canonical helper set:

- `attention.canonical_label_map`
- `attention.normalize_topics`
- `attention.level1_parents`

The governed hierarchy has 17 Level 1 domains and 50 Level 2 topics. An unknown
label fails validation even if it occurs on a foundation-excluded paragraph;
v1 does not invent an `Other` node or local spelling map.

For each paragraph, each distinct normalized Level 2 topic is counted once.
Each distinct Level 1 parent derived from that Level 2 set is also counted
once. A paragraph contributes one full presence membership to every assigned
topic. It is never fractionally divided among labels. Consequently, topic
shares are non-additive and may sum above 100%; consumers must state this in
downloads, methods copy, and accessible equivalents.

Topic-free eligible paragraphs remain in every president and scope paragraph
denominator. They create no topic membership and no edge. At baseline:

| Quantity | Rows |
| --- | ---: |
| Topic-bearing eligible paragraphs | 32,394 |
| Topic-free eligible paragraphs | 137 |
| Raw Level 2 assignments | 48,211 |
| Normalized distinct Level 2 memberships | 47,549 |
| Removed duplicates/case variants | 662 |
| Distinct derived Level 1 memberships | 43,570 |
| Total Level 1 + Level 2 memberships | 91,119 |

## Network and scopes

The network is undirected and bipartite. Its 112 nodes are 45 president nodes,
17 Level 1 topic nodes, and 50 Level 2 topic nodes. Edges can only connect one
president and one topic. There are no president-president or topic-topic
edges, similarity scores, causal direction, influence measures, generic
`weight` fields, or layout coordinates.

Scopes are the complete eligible corpus (`scope_type = corpus`,
`scope_id = all-corpus`) and each of the nine accepted Story eras
(`scope_type = story_era`, with the foundation's era key as `scope_id`). An
era edge includes paragraphs actually attributed to that president in that
foundation era regardless of source-document ownership.

## Units, denominators, and measures

For president `p`, topic `t`, and scope `s`, an edge carries:

- `topic_paragraph_count`: distinct eligible speaker paragraphs attributed to
  `p` in `s` that contain `t`;
- `topic_appearance_count`: distinct actual-speaker appearances containing at
  least one qualifying paragraph;
- `eligible_president_paragraph_count`: every eligible paragraph attributed to
  `p` in `s`, including topic-free paragraphs;
- `eligible_president_appearance_count`: every eligible appearance for `p` in
  `s`;
- `scope_topic_paragraph_count`: eligible paragraphs carrying `t` across all
  presidents in `s`;
- `scope_eligible_paragraph_count`: all eligible actual-president paragraphs in
  `s`.

The three published ratios have explicit names:

1. `speaker_paragraph_share = topic_paragraph_count /
   eligible_president_paragraph_count` is the primary president/topic
   prevalence measure.
2. `topic_contribution_share = topic_paragraph_count /
   scope_topic_paragraph_count` describes the president's share of the scope's
   eligible paragraph evidence carrying that topic.
3. `topic_scope_share = scope_topic_paragraph_count /
   scope_eligible_paragraph_count` describes topic prevalence in the selected
   scope.

A zero denominator produces null, not zero. No ratio is named importance,
strength, confidence, or weight.

The metric registry names `actual_speaker_topic_presence` and
`actual_speaker_topic_contribution`. Their public definitions declare the
actual-speaker population, paragraph unit, denominators, topic-free treatment,
multi-label non-additivity, scope behavior, non-causal interpretation, and
thin-record rules. Neither is registered as a chart metric because Plan 3 adds
no chart.

## Support and publication gates

President support is computed independently in every scope:

- `supported`: at least five eligible actual-speaker appearances;
- `thin`: one through four appearances;
- `no_record`: zero appearances.

Observed edge support is:

- `supported`: supported president, at least five topic paragraphs, and at
  least two topic-bearing appearances;
- `thin_edge`: supported president but one or both edge thresholds fail;
- `thin_president`: the president has one through four total eligible
  appearances in the scope.

No edge exists for `no_record`. An edge is `default_visible` only when its
president is supported and it has at least 20 topic paragraphs across at least
five topic-bearing appearances. Thin edges remain in governed and public data;
default visibility is a future rendering default, not a deletion rule.

Baseline support and edge acceptance is:

| Quantity | Count |
| --- | ---: |
| President/scope rows | 450 |
| Supported / thin / no-record president rows | 85 / 10 / 355 |
| Topic/scope rows | 670 |
| Observed / no-record topic rows | 589 / 81 |
| Edges | 4,408 |
| Supported / thin-edge / thin-president edges | 3,340 / 864 / 204 |
| Default-visible edges | 2,104 |
| All-corpus Level 1 / Level 2 edges | 700 / 1,464 |
| Story-era Level 1 / Level 2 edges | 728 / 1,516 |

## Evidence receipts

Every edge has one to three keyed audit examples. Qualifying actual-speaker
appearances are ordered by speech date, document key, appearance identity, and
paragraph index. The selected roles are first, lower-middle, and last. When
there are one or two appearances, duplicate positions are removed. Within each
selected appearance, the lowest qualifying `para_idx` is used.

A receipt includes edge and scope identity, president and topic identity,
`(doc_name, para_idx)`, appearance identity, actual speaker, source-document
owner, cross-owner state, Story era, speech date/title/link, a whitespace-
normalized public-safe excerpt bounded to 360 characters, selection role, and
selection position. The baseline has 11,576 receipts. They are deterministic
audit examples—not representative samples or historical proof.

## Governed schemas and artifacts

Every table carries both its table schema and
`actual-speaker-topic-network-v1` contract identity.

| Artifact | Schema | Rows |
| --- | --- | ---: |
| `paragraph_topics_v1.parquet` | `actual-speaker-topic-paragraph-v1` | 32,531 |
| `memberships_v1.parquet` | `actual-speaker-topic-membership-v1` | 91,119 |
| `president_nodes_v1.parquet` | `actual-speaker-topic-president-node-v1` | 45 |
| `topic_nodes_v1.parquet` | `actual-speaker-topic-topic-node-v1` | 67 |
| `president_support_v1.parquet` | `actual-speaker-topic-president-support-v1` | 450 |
| `topic_support_v1.parquet` | `actual-speaker-topic-topic-support-v1` | 670 |
| `edges_v1.parquet` | `actual-speaker-topic-edge-v1` | 4,408 |
| `edge_evidence_v1.parquet` | `actual-speaker-topic-evidence-v1` | 11,576 |
| `acceptance_report_v1.json` | `actual-speaker-topic-network-acceptance-v1` | one report |
| `meta_v1.json` | `actual-speaker-topic-network-meta-v1` | one manifest |

`SpeakerTopicNetworkBundle` is the typed multi-consumer reader. It validates
the accepted tables, explicit schemas, stable identifiers, unique keys,
support gates, ratios, evidence reconciliation, artifact hashes, metadata
self-hash, current input identities, and every pinned acceptance value.
Consumers load this bundle; they do not join raw foundation and topic files or
recompute speakers, appearances, eras, denominators, support, or evidence.

## Public downloads and site integration

The canonical site build validates the accepted bundle before any generated-
site write and publishes:

- `network_v1.json` (`actual-speaker-topic-network-public-v1`), containing
  nodes, scopes, support rows, edges, measures, policies, and provenance;
- `edges_v1.csv`;
- `edge_evidence_v1.csv`;
- `memberships_v1.parquet`;
- `manifest_v1.json`
  (`actual-speaker-topic-network-public-manifest-v1`).

The manifest is written last and records schema versions, counts, per-file
hashes, active promoted generation, foundation identity, taxonomy hash, metric
definitions, multi-label and denominator policies, support thresholds,
evidence selection, and interpretation limits. The public validator checks
hashes, row counts, stable IDs, membership keys, and exact input-provenance
parity with the accepted bundle.

The two public JSON files add two files to the canonical JSON validation count.
At the current site baseline that changes 183 JSON files to 185 while the HTML
count remains 73. No Plan 2 schema or foundation consumer inventory changes for
this separate superseding reusable contract.

## Provenance and write safety

Governed content contains no wall-clock timestamp. Rows and JSON objects have
stable ordering. The metadata hashes the accepted foundation metadata and
artifacts, promotion pointer, resolved generation manifest and artifact root,
full promoted table, exact promoted topic projection, taxonomy, source
configuration, and every governed artifact. Metadata self-authenticates with a
hash computed without its own hash field.

All joins, input identities, taxonomy values, schemas, counts, measures, and
receipts validate in memory and in an isolated candidate directory before
accepted files are replaced. Each file is atomically replaced; rollback
restores the prior set if replacement fails; governed metadata and public
manifest are written last. Output paths resolving under paid annotations,
sealed speaker attribution, the annotation ledger, or the accepted speaker and
reference foundation are rejected.

`--check` re-resolves current inputs and rejects stale artifacts.
`--verify-reproducible` performs two independent temporary builds of all ten
governed files and five public files and requires all 15 bytes and hashes to
match. The approved rebuild never edits `data/llm_annotations/` or
`data/speaker_attribution/`.

## Future consumer and renderer requirements

Plan 4 or another authorized consumer must load the validated bundle. Broad
Level 1 domains are the default; do not render an unreadable all-node hairball.
Initial views should normally keep supported, default-visible relationships and
reveal Level 2 detail on demand.

A production renderer must:

- give hover and keyboard focus equivalent highlighting and exact values;
- support pin/select interaction, Escape-to-clear, and stable focus order;
- combine color with shape, stroke, labels, or grouping and label thin records;
- respect reduced motion and stop continuous node movement during inspection;
- provide a mobile subset, focused ego view, or locally scrollable alternative;
- provide an accessible table or structured textual equivalent;
- announce selection changes through an appropriate live region;
- keep evidence receipts keyboard reachable;
- avoid unsafe raw `innerHTML` assembly;
- never describe proximity as similarity, influence, importance, validation,
  confidence, intent, or causality.

## Known limitations

The Miller Center corpus is a source collection, not a complete record of each
presidency. Paragraph boundaries vary, topic labels are AI assignments without
calibrated numeric confidence, and multi-label shares do not form a partition.
Actual-speaker attribution has governed exclusions, and thin records remain
thin even when an observed topic count exists. Era scopes describe the nine
accepted Story eras; they do not establish historical periods as causal units.
Evidence receipts are deterministic audit handles, not representative
sampling. The contract does not estimate similarity, influence, policy
outcomes, public opinion, or presidential intent.

## Verification commands

Use the repository's x86_64 environment:

```bash
arch -x86_64 .venv/bin/python -m presidential_profiles.foundation_audit --check
arch -x86_64 .venv/bin/python -m presidential_profiles.story_foundation --check --site-dir docs
arch -x86_64 .venv/bin/python -m pytest -q tests/test_speaker_topic_network.py
arch -x86_64 .venv/bin/python -m presidential_profiles.speaker_topic_network --verify-reproducible
arch -x86_64 .venv/bin/python -m presidential_profiles.speaker_topic_network --rebuild
arch -x86_64 .venv/bin/python -m presidential_profiles.speaker_topic_network --check
arch -x86_64 .venv/bin/python -m presidential_profiles.site
arch -x86_64 .venv/bin/python -m presidential_profiles.speaker_topic_network --check --site-dir docs
arch -x86_64 .venv/bin/python -m pytest
arch -x86_64 .venv/bin/python -m compileall -q src tests
node scripts/validate_inline_js.mjs docs
node --check docs/assets/issues-v2.js
git diff --check
```

Reggie Doctor remains the portable repository check:

```bash
~/.codex/scripts/reggie/reggie_doctor.sh --repo /Users/jacobpress/Desktop/Projects/presidential_profiles
```

## Plan 4 handoff

Plan 4 may decide whether this becomes Summary's fifth section. The recommended
starting view is the all-corpus Level 1 network with supported/default-visible
relationships, focused president or topic selection, Level 2 disclosure on
demand, and inspectable evidence receipts.

Plan 4 must not independently recompute topics, speakers, appearances, eras,
denominators, support, or evidence. A genre-specific scope, different
denominator, new score, or new era definition requires a versioned contract
extension before page-level implementation.

## Completed verification receipt

- All audited input and acceptance counts match exactly; there are no
  deviations from the pinned Plan 3 baseline.
- The focused contract/metric suite passes 37 tests. The broader related
  foundation/topic/evidence/metric/site set passes 565 tests with 12 existing
  NumPy warnings.
- The complete repository suite passes 2,759 tests with 69 existing warnings.
- Two independent temporary builds produce the same ten governed and five
  public files byte-for-byte.
- The explicit rebuild and immediate stale/current check pass.
- Python compilation/import checks pass.
- The canonical build validates 73 HTML pages and 185 JSON files. Story's
  `docs/index.html` SHA-256 remains
  `b11ffde221c9abc8453e939600b3b8c6776969d065941d9fdb206943f78ea56d`,
  matching the pre-task capture, and no HTML file references the new bundle.
- Story foundation, foundation audit, network public parity, links,
  navigation, anchors, JSON, download, and metric checks pass.
- All 130 inline scripts parse; `docs/assets/issues-v2.js` passes
  `node --check`.
- The paid-annotation and sealed-speaker directories retain their pre-task
  file and byte counts, produce no Git diff/status entry, and pass the governed
  foundation protected-input audit.
- Reggie Doctor reports zero errors and zero warnings. No browser smoke was
  required because no HTML consumer or renderer was added.
