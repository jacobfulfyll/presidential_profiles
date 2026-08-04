# Annotation refresh campaign v1 — Phase 2 contract

Status: **Stage 3 pilot evaluation failed; no specification/bundle freeze or passing campaign seal**  
Contract date: 2026-07-24 (date precision; no time is inferred)  
Repository: `presidential_profiles`  
Review queue: `notes/annotation-refresh-campaign-review-queue-v1.json`  
Readable review state: `notes/annotation-refresh-campaign-review-report-v1.md`

This is the Phase 2 contract checkpoint for a provider-neutral annotation-refresh
system. It extends the approved Phase 1 ledger; it does not reopen or replace the
Phase 1 decisions. The repository user approved `ARCV1-C001` on 2026-07-24 with
date precision, authorizing Stage 1 source, registry, documentation, and test work
only. At that checkpoint, labeling, pilot execution, sealing, adjudication,
promotion, site generation, and deployment remained unauthorized. On 2026-07-24
the repository user separately
authorized Stage 2 planning only through `ARCV1-RES003`. The repository user then
approved the constructible sampling amendment and evaluation-frame boundary through
`ARCV1-RES004`. The repository user approved the exact Stage 2 plan checkpoint
`ARCV1-G002` through `ARCV1-RES006`.
An exact `gpt-5.6-sol`/`high` runtime receipt then satisfied `ARCV1-G001`
through `ARCV1-RES007`, the repository user authorized pilot execution through
`ARCV1-RES008`, and verified assignment-free initialization was recorded through
`ARCV1-RES009`.

The implemented orchestration module name is
`src/presidential_profiles/annotation_refresh.py`. That name matches the existing
`annotation_ledger.py` and `annotation_workflow.py` split: the ledger owns immutable
history, the workflow owns one offline run, and the refresh module coordinates
versioned bundles and campaigns without duplicating either.

## 1. Authority, scope, and stop condition

Phase 2 is limited to:

- an additive label-bundle registry;
- an additive campaign manifest and orchestration layer;
- configuration-driven bundle assignment and response decoding through the Phase 1
  terminal workflow;
- a combined paragraph-judgment candidate;
- speech-factual and invocation-tone bundles;
- a pilot-only constituency diagnostic bundle;
- persisted deterministic scope selection;
- exact model-identity, reasoning-effort, resume, evaluation, comparison,
  adjudication, and explicit-promotion rules;
- the one intentional typed-projection extension needed to materialize promoted
  constituency claims;
- tests, documentation, and local read-only verification.

Absent a later explicit review resolution, Phase 2 does not authorize:

- changing any raw legacy value or Phase 1 sealed artifact;
- modifying `data/llm_annotations/` or
  `data/corpus_corrections/manifest_v1.json`;
- modifying existing adjudication, promotion, content-addressed generation, or
  input-inventory artifacts;
- running a model, issuing a labeling assignment, or spending tokens;
- freezing constituency v1 or `paragraph_judgment_v2`;
- running a pilot or a full refresh;
- promoting any shadow label;
- editing or rebuilding `docs/`;
- site generation or deployment;
- adding a provider client, credential reader, network path, or Codex/Reggie
  adapter to the core pipeline.

Review item `ARCV1-C001` was approved on 2026-07-24, and Stage 1 completion is
recorded as `ARCV1-S001`/`ARCV1-RES002`. Stage 2 planning was authorized by
`ARCV1-S002`/`ARCV1-RES003`. Its first read-only selection audit reached the
contract's failure branch because the live corpus has three sparse era-by-genre
cells, not two. `ARCV1-RES004` approved `ARCV1-I001` and `ARCV1-D016`: the core is
therefore 482 paragraphs, the combined pilot is 722, and probability-weighted
inference is limited to the design-balanced evaluation frame. The exact
content-addressed selections and non-executable plan are now published. At Stage 2
publication, no runtime receipt, executable campaign, child run, assignment,
response, or later execution artifact had been created. The subsequent Stage 3
checkpoint initialized the exact approved campaign and three child runs, with
zero assignments, responses, or events.

## 2. Phase 1 foundations inherited unchanged

The following decisions are fixed inputs, not Phase 2 review questions:

1. Raw legacy values, multiplicity, spelling, case, and list order remain unchanged.
2. Independent models and versions coexist in the generic event ledger.
3. Current state changes only through explicit expected-state-guarded promotion or
   approved adjudication transitions.
4. A comparison, seal, campaign completion, or newer timestamp never promotes.
5. Opus remains an unpromoted second opinion.
6. No empty legacy entity response is synthesized.
7. Removed duplicate rows remain historical and non-current and are not remapped to
   retained copies.
8. The trusted 13-paragraph correction overlay is append-only.
9. The April 3, 1968 factual carry-forward remains the sole narrow approved
   input-mismatch exception.
10. Unknown legacy times, intermediate responses, effort, invocation prompt text,
    and invocation cost remain unknown where Phase 1 says they are unknown.
11. Paragraph membership and joins use `(doc_name, para_idx)`, never row order, with
    cardinality and key-set equality checks.
12. The Phase 1 constituency specification remains draft. The active
    `constituency_claims` projection remains schema-valid and empty.
13. Sealed runs and decision artifacts are immutable. Recovery creates a new
    append-only artifact.
14. Content-addressed materialization is derived state and must be atomically
    published and byte-reproducible.
15. No core ledger or terminal module imports or invokes a model provider.

Phase 2 may add a new specification or projection version. It may not reinterpret
an existing event under a new meaning.

## 3. Read-only Phase 1 checkpoint evidence

Read-only validation on 2026-07-24 confirmed that
`data/annotation_ledger/materialized/current` points to:

```text
5b3f0e252c63a3b6cf3cba5d5e84e23def38eaa8edf29b89af827c92e5bce307
```

The active generation re-derived as:

| Table or property | Count |
|---|---:|
| sealed runs | 16 |
| legacy backfill runs | 15 |
| terminal annotation runs | 1 |
| label events | 261,335 |
| canonical projections | 261,335 |
| promotion transitions | 196,153 |
| current labels | 206,837 |
| evaluation records | 136 |
| adjudications | 0 |
| constituency events | 0 |
| constituency projection rows | 0 |

The current layer contains one group for each of the five required paragraph
judgments on all 35,394 canonical paragraphs, the three factual labels on all
1,053 canonical speeches, 101 invocation-tone spans, and no Opus rows. The exact
current row counts by label type are:

| Label type | Current rows |
|---|---:|
| topics | 35,394 |
| party_attack | 35,394 |
| enemy_naming | 35,394 |
| zero_sum | 35,394 |
| proposal_values | 35,394 |
| entities | 26,607 rows in 15,923 groups |
| speech_type | 1,053 |
| audience | 1,053 |
| medium | 1,053 |
| invocation_tone | 101 |
| constituencies | 0 |

The required focused read-only suite passed:

```text
38 passed in 36.97s
```

The portable Reggie Doctor reported 0 errors, 0 warnings, and 5 informational
findings: optional `DECISIONS.md` and `.codex/ASSISTANT-GUIDE.md` are absent,
`.claude/stats.json` is changed tool telemetry, the worktree is extensively dirty,
and the repository uses its native/external Reggie layout without adapters.

One documentation inconsistency was found and is deliberately not repaired here.
`notes/annotation-ledger-review-report-v1.md` first records
`ALRV1-G001` as satisfied and the exact runtime receipt exists, but its later
“Provenance gaps to carry forward” list still says the receipt does not exist
because labeling has not begun. The sealed run, receipt, Phase 1 queue, active
generation, and the report's own execution-gate section agree that execution did
occur. Review item `ARCV1-X001` reports this stale sentence without modifying the
resolved Phase 1 record.

## 4. Terminology and ownership

### 4.1 Label type

A label type is one atomic registered quantity in `label_events`. Examples are
`topics`, `enemy_naming`, `entities`, `constituencies`, and `speech_type`.

Every Phase 2 event retains:

- `label_type`;
- immutable atomic label-spec version and hash;
- run provenance;
- subject type, subject identity, and trusted canonical subject key;
- target-text and assignment-input fingerprints;
- `label_group_id`;
- source response, assignment, and artifact identities;
- raw model output serialized without normalization.

One bundle response may emit many label types, but those types never collapse into
one ledger field.

### 4.2 Label bundle

A label bundle is a versioned execution contract that can emit one or more atomic
label types from one response. It owns:

- subject type and eligible-universe resolver;
- exact ordered emitted label-spec map;
- full prompt and prompt hash;
- response JSON Schema and schema hash;
- bounded-context constructor;
- deterministic batching policy;
- response-local identifier policy;
- response-to-atomic-event decoding;
- group atomicity;
- bundle validators;
- optional typed projections;
- evaluation requirements;
- campaign eligibility and publication gates.

The proposed bundle registry is:

```text
data/annotation_ledger/bundles/
  registry-v1.json
  <bundle_id>/<bundle_version>.json
```

Each bundle has `bundle_sha256`, computed over its entire canonical object with
that field omitted.

### 4.3 Refresh campaign

A refresh campaign is an immutable executable manifest coordinating one or more
bundle child runs against one corpus snapshot and persisted subject selections. A
campaign is not a run. It expands deterministically into separate Phase 1 terminal
runs, one for every `(bundle, pass)` declared by the manifest.

Each child remains independently resumable, auditable, sealable, comparable,
adjudicable, and promotable. A failed child does not mutate or invalidate a
successfully sealed sibling.

## 5. Additive registry architecture

Phase 1 label-spec files and `registry-v1.json` remain valid and unchanged. Phase 2
adds support for an additive label-spec schema v2. A v2 atomic spec has one of two
execution bindings:

- `standalone`: complete prompt and response schema live on the atomic spec; or
- `bundle`: the spec records the exact bundle ID, version, hash, response pointer,
  decoding rule, and atomic group rule.

This is a one-time registry/resolver extension. After it exists, adding a normal
label type requires only:

- one versioned atomic spec;
- a prompt/schema or a binding to a versioned bundle;
- optional named special validation;
- tests;
- optionally a typed projection.

Adding a normal bundle requires a bundle file, registry entry, tests, and eligible
status. Neither operation may require later edits to core event storage, sealing,
comparison, adjudication, materialization, or promotion.

Bundle identity is recoverable from every event through its atomic spec snapshot,
run manifest, assignment artifact, and bundle receipt. The Phase 1 `label_events`
physical schema does not need a provider-specific field.

## 6. Proposed initial bundles

### 6.1 `paragraph_judgment_v2_candidate@candidate-1`

Subject type: canonical paragraph.

Atomic emitted label types:

- `topics`;
- `party_attack`;
- `enemy_naming`;
- `zero_sum`;
- `proposal_values`;
- `entities`;
- `constituencies`.

The six already-published quantities receive new candidate atomic spec versions;
the v1 specs and current values remain intact. Constituencies receive
`constituency-v1-candidate-1`. The bundle remains `pilot_only` and is excluded from
production `all` until all gates in Sections 13–15 pass.

The proposed common context is:

- target paragraph;
- at most one preceding and one following paragraph from the same speech;
- speech decade;
- no president, party, title, historical label, Opus response, or current label.

This keeps the existing paragraph judgments masked and makes the combined and
constituency-only diagnostic contexts identical. Removing president/title metadata
from the earlier draft constituency helper is a new Phase 2 decision, not a change
to its Phase 1 historical record.

The compact response contains one object per trusted response-local integer ID.
It returns only labels and required short evidence. It does not return
`doc_name`, `para_idx`, hashes, run/campaign identity, model identity, timestamps,
assignment identity, label IDs, or chain-of-thought. The local program supplies all
of those values and maps the local ID back to `(doc_name, para_idx)`.

`enemy_naming` and `constituencies` remain independent. The schema permits:

- neither;
- constituency only;
- adversary only;
- both for different groups;
- both for the same literal group in distinct claim and adversary roles.

No decoder infers one from the other.

### 6.2 `speech_factual@v1`

Subject type: canonical speech.

Emitted label types:

- `speech_type`;
- `audience`;
- `medium`.

This bundle initially binds the frozen `speech-factual-v1` atomic specs and factual
input function. A new model may create a full shadow layer without changing the
prompt. Each assignment contains one speech: title, year, and up to the first five
canonical paragraphs.

### 6.3 `invocation_tone@candidate-1`

Subject type: `invocation_span`, exactly as established by Phase 1.

Emitted label type: `invocation_tone`.

The exact current eligible universe is the 101 registered spans. The legacy prompt
did not survive, so the legacy-partial spec cannot honestly be used as a
reproducible refresh prompt. Phase 2 must add a complete candidate rubric for the
surviving `R`/`C`/`N` quantity, bind an exact context window, and pass a predeclared
reference audit before the bundle becomes full-shadow eligible. Agreement with the
legacy values is diagnostic, not proof of accuracy.

No approved invocation audit design or thresholds existed in the Phase 1
specification, this contract, the review queue, or the routed annotation
documentation. Stage 3 therefore published the non-executable proposal
`inv_audit_98a118ea36a1e2557a7447fa83002236b01b41d2759a4c74eb7f1ee584fb58be`
under pending review item `ARCV1-D018`. It proposes:

- the full exact 101-span universe, with no sample or supplement;
- a fresh restricted medium-effort independent reference, sealed before a
  separate candidate pass can see any reference result;
- append-only post-reference adjudication of every disagreement;
- 100% structural coverage and accepted-event validity, with zero compromised
  blindness receipts;
- overall accuracy at least 0.90 and Cohen's kappa at least 0.80;
- support floor 10, supported-class macro F1 at least 0.85, and precision and
  recall at least 0.80 for every supported class;
- 2,000 speech-clustered bootstrap draws using the declared invocation seed,
  with one-sided lower bounds of at least 0.85 for accuracy and 0.75 for
  supported-class macro F1; and
- zero denominators or inconclusive bounds fail to pass.

The proposal explicitly authorizes neither labeling nor freezing. Legacy
agreement remains diagnostic only. Invocation reference labeling must stop until
the repository owner explicitly approves or revises `ARCV1-D018`.

### 6.4 `constituency_only_diagnostic@candidate-1`

Subject type: canonical paragraph.

Emitted label type: `constituencies`.

This bundle uses the exact constituency semantic block, field schema, context
constructor, local-ID rules, and model/reasoning profile of the combined bundle.
Only the absence of the other paragraph fields differs. It is permanently
`pilot_diagnostic`, never production-eligible, and may run only on the persisted
144-paragraph diagnostic subset in Section 11 without a new human approval.

## 7. Constituency v1 candidate semantics

### 7.1 Operational claim

A constituency claim exists when the presidential voice explicitly treats a group
as at least one of:

- a source of governmental or presidential authority;
- a constituency the speaker or national government represents;
- a group whose rights, security, liberty, or welfare create a duty of protection;
- an intended beneficiary of a proposed action or policy;
- a group explicitly included within the nation, citizenry, or governing people.

Audience, praise, topic mention, generic favorable language, a named actor, and a
generic unresolved `we` do not suffice. A quoted claim counts only when the
presidential voice adopts it.

### 7.2 Raw response object

Every paragraph returns:

```text
outcome: claim | none | unclear
claims: zero or more claim objects
unclear_reason: short string
```

A claim object returns:

- `group_text`: exact case-sensitive target-paragraph substring;
- `resolved_referent`: a short raw referent only when neighboring context is needed,
  otherwise empty;
- `group_type`;
- `relation`;
- `stance`;
- `evidence_span`: exact target-paragraph substring containing `group_text`;
- `certainty`: `explicit` or `context_resolved`.

No rationale or free-form historical interpretation is requested.

The candidate `group_type` enum is:

- `national_public`;
- `geographic_population`;
- `racial_ethnic_indigenous_or_legal_status`;
- `economic_class`;
- `occupation_or_industry`;
- `military_or_veteran`;
- `party_or_movement`;
- `religious_group`;
- `age_gender_or_family`;
- `foreign_population_or_nation`;
- `institution_or_organization`;
- `other`.

The `relation` enum is:

- `source_of_authority`;
- `represented_constituency`;
- `protected_group`;
- `intended_beneficiary`;
- `included_national_member`.

The `stance` enum is:

- `favorable`;
- `neutral`;
- `adversarial`;
- `mixed_or_ambivalent`.

Relation and stance are separate. A group can be explicitly protected while
described critically, and the same group can appear elsewhere as an adversary.

### 7.3 Boundary rules

- “The people” and “Americans” qualify only when the paragraph asserts one of the
  declared relations; the noun alone does not qualify.
- A pronoun may be `group_text` only when its referent is resolved from the bounded
  same-speech context and recorded in `resolved_referent`.
- Geographic populations qualify when people from the place, rather than land or
  jurisdiction alone, are represented, protected, included, or benefited.
- Demographic groups, occupations, parties, movements, economic classes, and
  religions use their specific categories when the relation is explicit.
- An institution qualifies only when its own represented interests or protected
  status are invoked. Merely assigning it an action does not make it a constituency.
- A nation qualifies only when the state or its people are explicitly placed in a
  declared constituency relation. Diplomatic mention alone does not qualify.
- Each distinct `(group_text, resolved_referent, group_type, relation, stance,
  evidence_span, certainty)` is one claim. The same group with two relations creates
  two claims. Exact duplicate claim objects are invalid.
- `none` requires an empty claim list. `unclear` requires an empty claim list and a
  reason and is ineligible for promotion until adjudicated.

### 7.4 Raw-event and projection boundary

The complete constituency response object is preserved as one raw atomic event in
one `label_group_id`. The typed projection may explode it into multiple rows, but
every projected row retains that group ID, so promotion remains atomic.

The model does not supply an authoritative `normalized_group`. A versioned derived
normalizer may populate `normalized_group`, `normalization_status`, and
`normalization_spec_sha256` later. Until then the field is null or explicitly
unresolved. Casefolding, aliases, category rollups, and entity linkage never rewrite
`raw_value_json`.

`constituency_claims` remains empty unless a qualifying constituency group is
explicitly promoted. Sealing or materializing an unpromoted pilot/full-shadow run
does not populate it.

## 8. Campaign manifest and identifiers

### 8.1 Two-step creation

`plan-campaign` creates a non-executable, content-addressed plan containing token
estimates and proposed selections. It does not initialize a run.

`init-campaign` requires:

- an approved plan hash;
- the exact current corpus artifacts and fingerprint;
- persisted exact bundle selections;
- locked bundle versions and hashes;
- an exact runtime model-identity receipt for each execution profile;
- declared reasoning effort;
- approval references required by the plan.

Only then does it atomically publish an executable campaign manifest and child work
runs. No assignment can exist before the exact identity receipt.

### 8.2 Manifest fields

The executable manifest contains:

- `campaign_manifest_version`;
- `campaign_id` and identity hash;
- source plan ID/hash;
- canonical corpus snapshot ID, fingerprint, counts, and source artifact hashes;
- exact bundle ID/version/hash set;
- bundle gate and eligibility receipts;
- exact persisted selection ID/hash and ordered key count per child;
- scope mode and any prior-campaign identity;
- child run IDs, pass roles, subject types, and completion policies;
- exact model receipt IDs/hashes and model IDs;
- provider ID if observed, otherwise null;
- declared reasoning effort per child;
- assignment batching/context policy hashes;
- evaluation-policy hash;
- promotion-policy hash;
- human checkpoint and review-resolution IDs;
- limitations and blindness receipts;
- operational timestamps outside the semantic identity payload.

`campaign_id` is `camp_` plus the SHA-256 hex of the canonical immutable semantic
manifest with `campaign_id`, operational timestamps, and derived checksums omitted.

`selection_id` is `sel_` plus the SHA-256 of the sorted complete list of subject
keys, subject input fingerprints, strata, inclusion frame, and inclusion
probability.

Child run IDs are deterministic:

```text
refresh-<first16 campaign hash>-<bundle id>-<pass role>
```

The full identity and collision check, not the shortened human-readable prefix, is
stored in every child run.

## 9. Scope modes

Exactly one scope mode is required per campaign:

- `pilot`: only the persisted stratified selection; no selector rerun at execution;
- `full-shadow`: every eligible canonical subject, producing an unpromoted layer;
- `delta`: keys absent from, or bundle-input-fingerprint-changed since, one declared
  prior campaign;
- `audit`: a persisted independent sample not used to tune the prompt;
- `explicit`: an exact supplied persisted selection.

For `delta`, the bundle input fingerprint includes the target and every bounded
context/metadata field that the model sees. A neighbor change therefore counts as
a changed bundle input. Deleted prior subjects are listed as tombstones but are not
silently deactivated or promoted.

A model-version refresh may deliberately use `full-shadow`. An ordinary corpus
update defaults to `delta`; selecting `full-shadow` requires an explicit human
override recorded in the campaign.

`all` means every registry bundle whose exact locked entry is
`production_eligible` and whose required gate receipts are satisfied. It never
means every file in the registry. `pilot_only`, `pilot_diagnostic`, `draft`, failed,
and superseded bundles are excluded.

## 10. Assignment, batching, and resume contract

### 10.1 Proposed sizes

| Bundle | Targets per assignment | Speech mixing | Context |
|---|---:|---|---|
| paragraph combined | 4 | prohibited | one neighbor each side, shared |
| constituency diagnostic | 4 | prohibited | identical to combined |
| speech factual | 1 | not applicable | title, year, first 5 paragraphs |
| invocation tone | up to 8 | prohibited | exact registered local window |

Paragraph targets are packed by canonical speech and target position. Four
contiguous targets are transmitted once with at most one non-target neighbor on
each side. The assignment may therefore contain at most six paragraph texts.

The four-target choice is a candidate, not an assumption that arrays are safe.
Phase 1 observed structured-array collapse at sizes 64, 15, 5, and 2 under a
different provider runtime. The two blind pilot passes must measure complete-array
validity. Failure supersedes the pilot/bundle version; it does not silently change
batch size inside a supposedly identical pass.

The first Stage 2 rendering audit found that the Stage 1 renderer repeated inner
target paragraphs inside each target's before/after context. Stage 2 corrected
that approved seam: model-facing paragraph targets now carry text once, followed
by one shared outer `before` and `after` context. Persisted planned batch IDs and
orders are enforced by `next`, and regression tests reject noncontiguous
same-speech mappings. `ARCV1-S003` is satisfied by this implementation evidence.

### 10.2 Trusted local identifiers

Responses echo only assignment-local integers. The local assignment artifact maps
each integer to one trusted subject ID and `(doc_name, para_idx)`. The renderer
never asks the model to return a trusted key or hash.

No paragraph or invocation assignment mixes speeches. Duplicate `para_idx` values
from different speeches therefore cannot alias even if the response schema contains
only local IDs.

### 10.3 Resume

- Assignment membership and identity are persisted before rendering.
- At most one open assignment exists per child run.
- `next` returns the existing open assignment byte-for-byte.
- A successfully ingested assignment is never selected or issued again.
- A duplicate response, including a byte-identical duplicate, fails before writes.
- Validation occurs completely in memory before the first response/event mutation.
- Interrupted campaign status is reconstructed from immutable assignments and
  ingested responses, never from row counts alone.
- A new retry assignment is permitted only after an explicit superseding decision;
  it has a new assignment ID and preserves the failed assignment as history.

## 11. Persisted combined-prompt pilot

### 11.1 Size and strata

The persisted pilot contains exactly 722 distinct canonical paragraphs:

- 482 design-balanced core paragraphs;
- 96 rare-signal enrichment paragraphs;
- 144 prompt-composition diagnostic paragraphs.

Era strata are the nine exact `trends.ERAS` values:

1. The founding, 1789–1815;
2. Expansion, 1816–1849;
3. Civil War & Reconstruction, 1850–1877;
4. The Gilded Age, 1878–1900;
5. Progressives & Depression, 1901–1932;
6. War & New Deal, 1933–1945;
7. The Cold War, 1946–1988;
8. Post-Cold War, 1989–2016;
9. The present era, 2017–2026.

Genre strata are the ten exact current `speech_type` values. An unknown future
genre causes planning to fail rather than entering an “other” cell silently.

### 11.2 Deterministic selection

The seed string is:

```text
annotation-refresh-campaign-v1|pilot|20260724
```

Selection uses SHA-256 ordering, not a library PRNG:

```text
sha256(seed || "\x1f" || doc_name || "\x1f" || first_para_idx ||
       "\x1f" || canonical_target_key_list)
```

Canonical paragraphs are tiled into non-overlapping, position-based blocks of four
within each speech. Numeric `para_idx` adjacency is not assumed.

The 482-paragraph core selects at least one four-target block from every populated
era×genre cell that supports one. A cell with fewer than four paragraphs is
take-all and marked `sparse_census`; its unused quota is reallocated by
largest-remainder allocation proportional to the square root of remaining cell
block count. Selected core rows store cell population, inclusion probability, and
post-stratification weight for the defined evaluation frame. The active generation
and canonical corpus contain 75 populated cells and three cells below four
paragraphs:

- Civil War & Reconstruction × `other`: 1 paragraph;
- The Gilded Age × `eulogy_or_commemoration`: 2 paragraphs;
- War & New Deal × `proclamation`: 3 paragraphs.

Taking all six sparse paragraphs, one full block from each of the other 72 cells,
and 47 additional full blocks produces the approved 482-row core. The repository
owner approved this amendment as `ARCV1-I001`/`ARCV1-RES004`. It preserves all
sparse cells and all four-target selections.

The same audit found 1,642 paragraphs in incomplete speech-tail blocks. They are
4.64% of the canonical paragraph universe and have zero inclusion probability
under a full-four-target-block-only frame. `ARCV1-D016`/`ARCV1-RES004` therefore
limits the core to a design-balanced evaluation frame. Its weights must not be
used to claim prevalence over all canonical paragraphs.

The 96-row enrichment selects 24 additional disjoint four-target blocks by a
deterministic maximum-coverage pass over these fixed casefolded cue families:

- national public: `people`, `citizen`, `american`;
- geographic: `state`, `territory`, `north`, `south`, `region`;
- racial/status: `black`, `negro`, `indian`, `tribe`, `immigrant`, `freed`,
  `slave`;
- economic class: `working class`, `middle class`, `poor`, `wealthy`,
  `capitalist`;
- occupation/industry: `farmer`, `laborer`, `worker`, `miner`, `manufacturer`,
  `business`;
- military/veteran: `soldier`, `sailor`, `veteran`, `armed forces`, `military`;
- party/movement: `democrat`, `republican`, `party`, `union`, `movement`;
- religious: `church`, `christian`, `jew`, `muslim`, `religious`;
- age/gender/family: `women`, `woman`, `men`, `children`, `family`, `families`;
- foreign population/nation: `nation`, `peoples`, `allies`, `refugee`;
- institution/organization: `congress`, `court`, `bank`, `school`,
  `organization`;
- relation language: `represent`, `protect`, `rights`, `welfare`, `benefit`,
  `behalf`, `authority`, `duty`.

Cue matching uses Unicode casefolding and literal phrase matches with non-word
boundaries. The greedy score is the number of previously uncovered
`(era, cue family)` pairs; the SHA-256 order breaks ties. These rows are excluded
from prevalence estimates.

The 144-row diagnostic uses 36 additional disjoint four-target blocks, four per
era, selected to maximize genre diversity and then SHA-256 order. Its exact
membership is fixed before any response exists.

Stage 2 published these immutable receipts:

- combined 722:
  `sel_17cf0d99dc497c63a0f7ee85d2d86436883df9a0eb7a6cd7f6e1234271c51515`;
- diagnostic 144:
  `sel_ae7b0e92cb8a37e03c39a6a7270cfc8c5b39698c409787a8126ec84a6bf24fd9`;
- reference 240:
  `sel_799402f09a43f76bfd1a9386b47614cf0f7fccb3c85374ffaa60103f046de9bc`.

The combined selection has 182 fixed assignments: 122 core, 24 enrichment, and
36 diagnostic. Both future blind combined passes must consume this identical
partition. The diagnostic bundle must consume the exact 36 diagnostic batches.

If the current corpus cannot produce exactly 722 distinct targets under these
rules, planning fails and creates a review item. It does not lower the sample
silently.

### 11.3 Blind passes

The main pilot creates two separate immutable combined-bundle runs with:

- the identical persisted 722 keys;
- identical assignment partitioning and local IDs;
- identical prompt, bundle version, schema, context, and selection hashes;
- the same exact runtime model ID and reasoning effort;
- different pass/annotator identities;
- no assignment exposure to current, historical, Opus, or other-pass labels.

Both manifests are initialized before either pass is rendered. Comparison is
disabled until both runs are independently audited and sealed.

The diagnostic creates one separate constituency-only run over the exact 144
diagnostic keys. It is blind to both combined responses and uses the same exact
model ID and reasoning effort. A different runtime model identity makes the
composition test invalid and requires a superseding diagnostic.

Filesystem-level blindness cannot be proven merely because the renderer omits a
path: all agents share the repository. Execution must therefore use fresh
restricted labeling sessions that receive only rendered assignments, and record a
blindness receipt. Tests prove that `next` and assignment rendering cannot open a
sibling response root. Deliberate out-of-band inspection remains a procedural
limitation and disqualifies the affected pass.

## 12. Reference construction and adjudication

Before labeling, the selection marks a 240-paragraph reference set:

- all 144 diagnostic paragraphs;
- all 96 rare-signal enrichment paragraphs.

An authorized human adjudicator or explicitly delegated subject-matter reviewer
first labels the reference from the same text/context with all model and historical
labels hidden. That independent reference is sealed before candidate responses are
revealed for adjudication. The adjudicator may then inspect the two candidates to
resolve disagreements, but the original blind reference remains immutable.

All pilot disagreements are appended to comparison artifacts. `unclear` rows and
every disagreement on the 240-row reference require append-only adjudication.
Disagreements outside the reference may be resolved by an explicitly persisted
selection, but unresolved rows cannot support freezing or promotion.

If the reference contains fewer than 60 claim-positive paragraphs or a required
aggregate metric has fewer than 40 supported positives, the gate is inconclusive.
Expansion requires a new persisted supplement and explicit human approval; it does
not mutate the pilot selection.

Pilot examples, responses, and adjudications are excluded by key from every later
blind `audit` scope. They may appear in a final frozen rubric only if separately
approved and then can never be used as an audit example.

## 13. Evaluation definitions

All metric records store numerator, denominator, excluded cases, sample frame,
weighting rule, support, seed, and source artifact hashes.

### 13.1 Structural metrics

- Coverage:
  `unique complete schema-valid subjects / exact eligible subject count`.
- First-response schema validity:
  `complete schema-valid assignment responses / submitted first responses`.
  This originally declared diagnostic is removed from the pilot decision set by
  the pre-results owner amendment `ARCV1-RES010`; it is not inferred or scored.
- Accepted-event validity is 100% by construction; invalid responses never write
  events.

### 13.2 Agreement

- Boolean/category exact agreement: exact equal pairs divided by comparable pairs.
- Cohen's kappa: pairwise non-null comparable outcomes, with prevalence displayed.
- Multi-label exact agreement: equal de-duplicated sets divided by comparable pairs.
- Jaccard: `|A ∩ B| / |A ∪ B|`, defined as 1 when both sets are empty.
- Multi-label micro precision/recall/F1: pooled label decisions.
- Macro F1: unweighted mean over categories meeting the support floor.
- Entity exact match: exact `(group_text, type, stance)` after raw string
  preservation.
- Entity partial match: deterministic maximum bipartite matching on exact span,
  then type and stance.
- Constituency outcome agreement: exact `claim`/`none`/`unclear`.
- Constituency claim exact match: exact
  `(group_text, resolved_referent, group_type, relation, stance, evidence_span,
  certainty)`.
- Constituency partial match: deterministic maximum bipartite matching requiring
  overlapping group/evidence spans and exact category/relation, with stance and
  certainty scored separately.

### 13.3 Reference accuracy

For each supported class:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
accuracy  = (TP + TN) / all comparable cases
```

Zero denominators produce `not_estimable`, not zero.

Two constituency false-negative quantities are both required:

- claim miss rate:
  `gold claim paragraphs predicted none / all gold claim paragraphs`;
- predicted-none false omission rate:
  `gold claim paragraphs predicted none / all paragraphs predicted none`.

This prevents the phrase “false negatives among none” from hiding its denominator.

The probability core reports weighted prevalence and weighted aggregate accuracy.
The enriched/reference frames report accuracy and error discovery, not corpus
prevalence.

All interval and non-inferiority calculations resample speeches, not paragraphs,
for 2,000 deterministic bootstrap draws using seed string:

```text
annotation-refresh-campaign-v1|evaluation|20260724
```

### 13.4 Historical comparisons

Candidate results are compared with:

- current primary labels;
- retained Opus labels where key coverage exists;
- the independent/adjudicated reference.

Current/legacy agreement is diagnostic, not ground truth. Opus is a second opinion,
not a tie-breaker. Differences may arise from model, prompt, context, or schema
changes and are not causally attributed to constituency composition without the
diagnostic evidence.

## 14. Proposed gates and non-inferiority margins

`ARCV1-RES010` is a pre-results owner amendment published after structural
completion but before reference labeling, candidate-label inspection,
comparison, adjudication, evaluation, or any gate decision. The owner attested
that the responses were machine-produced without owner payload edits and removed
first-response formatting validity as unnecessary to the freeze decision. The
base policy and campaign hashes remain immutable. The amendment removes only:

- the Section 14.1 first-response validity threshold;
- the Section 14.3 schema-validity non-inferiority margin; and
- the Section 14.4 schema-validity non-inferiority margin.

Accepted responses and emitted events remain required to be completely
schema-valid. Every other structural, support, accuracy, agreement,
false-negative, composition, provenance, blindness, and non-inferiority gate
remains unchanged.

The content-addressed amendment is
`amend_732e266492b6a7fb744c12b121d28a8f6b97f8837a835942018b1f3898ed4180`;
the effective evaluation-policy identity is
`sha256:43c4b5c1ff7bdb6d43c7586b1413e5db8ef6df8b08419a815b52d689b17098ae`.

### 14.1 Structural gates

- Persisted pilot key coverage after resume: exactly 100%.
- Full-shadow eligible-key coverage after resume: exactly 100%.
- First-response assignment schema validity: superseded and removed from the
  decision set by `ARCV1-RES010`.
- Unknown, duplicate, incomplete, unassigned, drifted, or cross-corpus accepted
  responses: zero.
- Sealing audit issues: zero.
- Compromised-blindness passes: zero.

### 14.2 Constituency gates

On the 240-row reference, with at least 60 positive claim paragraphs:

- claim detection precision, recall, and F1: each at least 0.85;
- balanced accuracy: at least 0.85;
- claim miss rate: at most 0.15;
- predicted-none false omission rate: at most 0.10;
- claim-level exact micro F1: at least 0.80;
- supported-category macro F1: at least 0.75;
- blind-pass outcome exact agreement: at least 0.85;
- blind-pass outcome kappa: at least 0.70;
- mean claim-set Jaccard: at least 0.70.

A category is gate-eligible at reference support at least 20. Support 10–19 is
reported as caution; support below 10 is descriptive only. Every gate-eligible
category requires precision and recall at least 0.70. Accuracy alone cannot pass a
prevalence-imbalanced gate.

### 14.3 Prompt-composition gates

The constituency-only diagnostic is the comparator. For combined minus diagnostic:

- claim-detection F1 margin: -0.05;
- claim recall margin: -0.05;
- predicted-none false-omission margin: +0.05;
- schema-validity margin: superseded and removed by `ARCV1-RES010`.

For each, the paired one-sided 95% speech-clustered bootstrap bound must remain on
the non-inferior side of the margin for both combined passes. An inconclusive bound
does not pass.

### 14.4 Existing paragraph-label gates

Against the same reference, candidate combined output must be non-inferior to
current primary labels:

- each boolean flag accuracy and F1 margin: -0.03;
- proposal-values accuracy and macro-F1 margin: -0.03;
- topic micro-F1 and mean Jaccard margin: -0.05;
- entity exact/partial micro-F1 margin: -0.05;
- schema-validity margin: superseded and removed by `ARCV1-RES010`.

The paired one-sided 95% speech-clustered lower bound must exceed the negative
margin. For a supported existing category, an absolute F1 below 0.80 also fails.
Topic/category cells with support below 20 are reported and do not independently
gate, but a concentration of unsupported failures triggers human review.

### 14.5 Freeze conditions

`constituencies@v1` may freeze only when:

- semantics and enum decisions are approved;
- both blind combined passes and the diagnostic are independently sealed;
- reference construction and required adjudications are complete;
- all structural, constituency, and composition gates pass;
- no unresolved blindness or provenance item remains;
- the final prompt/schema are byte-identical to the passing candidate.

`paragraph_judgment_v2@v2` may freeze only when all the above and all existing-label
non-inferiority gates pass. Any prompt/schema change after the pilot creates
`candidate-2` and a new pilot; it never inherits the old pass.

## 15. Full new-model shadow campaign

After the required freezes and explicit checkpoint `ARCV1-G004`, the intended first
complete campaign contains:

- one `full-shadow` `paragraph_judgment_v2` child over all 35,394 canonical
  paragraphs;
- one `full-shadow` `speech_factual` child over all 1,053 canonical speeches;
- one `full-shadow` `invocation_tone` child over all 101 registered invocation
  spans.

Constituency appears only inside the combined paragraph child. Every label coexists
with legacy, current primary, Opus, and correction-overlay history.

Recommended declared reasoning profiles, subject to exact runtime support and human
approval, are:

| Bundle | Reasoning effort |
|---|---|
| paragraph combined | high |
| constituency diagnostic | high |
| speech factual | low |
| invocation tone | medium |

If the runtime cannot expose the exact model ID and the exact requested effort, no
assignment is issued.

After sealing, every bundle and label type is evaluated independently. Campaign
completion means only that the immutable outcome manifest is complete. It does not
mean that a bundle passed, that a label passed, or that anything was promoted.

### 15.1 Constituency-failure branch

If constituency or composition gates fail:

- the failed candidate runs remain sealed historical evidence and unpromoted;
- the candidate remains ineligible for production `all`;
- speech-factual and invocation-tone children may proceed independently if their
  own approvals and gates allow;
- no full paragraph campaign may silently discard constituency fields from the
  passing candidate;
- refreshing already-published paragraph labels requires a new explicitly named
  `paragraph_judgment_v2_without_constituency_candidate`, a new prompt hash,
  targeted pilot/non-inferiority evidence, and human approval.

This branch permits useful independent modules to proceed without representing
constituency as approved.

## 16. Promotion contract

New-model labels are shadow labels by default. Promotion may be generated by the
orchestrator:

- for an entire bundle;
- for one atomic label type;
- for one persisted exact subject selection;
- for complete atomic `label_group_id` values.

Every transition delegates to the Phase 1 promotion function and includes the
complete replacement set and expected-current-state hash. Entities and the
constituency response group are indivisible. A stale expected state refuses the
entire batch.

Sealing, comparison, evaluation, adjudication, campaign completion, and
materialization do not promote. Reversal is a new append-only promotion or
adjudication transition; history is never edited.

## 17. Provider-neutral execution protocol

No core ledger, workflow, bundle, campaign, or terminal module may:

- import OpenAI, Anthropic, or another provider SDK;
- read API credentials;
- choose a model;
- invoke a model;
- make a network call.

The terminal may:

- plan and initialize manifests;
- render readable compact assignments and response templates;
- validate returned JSON;
- store immutable valid responses and events;
- report status and audits;
- seal runs/campaigns;
- materialize;
- compare;
- record adjudication;
- explicitly promote.

The Codex chat agent supplies label values only. The local program supplies keys,
hashes, IDs, receipts, timestamps, and event/group identities.

The proposed CLI is:

```text
annotation_refresh list-bundles
annotation_refresh plan-campaign
annotation_refresh init-campaign
annotation_refresh next
annotation_refresh ingest
annotation_refresh status
annotation_refresh audit
annotation_refresh seal-bundle
annotation_refresh seal-campaign
annotation_refresh materialize
annotation_refresh compare
annotation_refresh adjudicate
annotation_refresh promote
```

Every command accepts one selected bundle where applicable. Planning accepts one
bundle, an explicit set, or `all`, plus exactly one scope.

## 18. Recovery design

- Interrupted plan/campaign creation: write a complete candidate directory, validate
  and fsync it, then atomically rename. Readers ignore orphan candidates.
- Incomplete assignment: return the existing open assignment; never allocate its
  subjects again.
- Duplicate response: refuse before writes.
- Invalid schema or incomplete keys: refuse before writes; assignment remains open.
- Unknown assignment or out-of-corpus key: refuse before writes.
- Cross-speech ambiguity: prohibited assignment construction plus trusted local-ID
  map; any mismatch refuses.
- Missing/inexact model identity: stop before campaign initialization or `next`.
- Corpus drift: every stateful command rechecks corpus and selection fingerprints.
- Prompt/spec/bundle drift: every command re-resolves and hashes exact snapshots.
- Partially completed bundle: resume from persisted assignment/response state.
- Failed audit: no sealing; repair by valid completion or superseding run.
- Failed seal: no sealed directory becomes visible; work remains recoverable.
- Failed materialization: prior active pointer remains; orphan complete generation
  may be verified later.
- One module fails: sealed siblings and their evaluations remain valid.
- Compromised blindness before seal: abandon work and create new runs.
- Compromised blindness after seal: append a disqualification review resolution;
  preserve the run and supersede it.
- Failed pilot: preserve failed runs and register a new candidate/pilot identity.
- Full shadow fails gates: preserve it as historical and unpromoted; a later campaign
  supersedes it.
- Failed publication: never expose a partial content-addressed generation.
- Stale lock: never delete automatically; require operator inspection.

## 19. Deterministic token estimates

No network call or provider tokenizer was used. The exact future runtime tokenizer
is unavailable. Input text was estimated deterministically as
`ceil(UTF-8 bytes / 4)`. Static prompts use the same proxy. Existing paragraph
output uses the measured Phase 1 rate of 115 tokens per paragraph; the candidate
adds a 45-token constituency allowance, for 160. Reasoning/internal tokens and
provider caching discounts are excluded.

The canonical corpus re-derived as:

| Bundle | Eligible subjects |
|---|---:|
| paragraph combined | 35,394 |
| speech factual | 1,053 |
| invocation tone | 101 |
| constituency diagnostic | 144 pilot-only |

At four paragraph targets per assignment, the full corpus produces 9,246
same-speech assignments. Target text is 5,989,096 proxy tokens; with one shared
neighbor on each side, transmitted text is 8,759,257 proxy tokens. Stage 2
serialized every exact planned model-facing assignment with two-space indentation,
sorted keys, and UTF-8 output, then applied the same deterministic byte proxy. The
runtime tokenizer remains unavailable.

| Work | Assignments | Input tokens | Output tokens |
|---|---:|---:|---:|
| one 722-paragraph combined pilot pass | 182 | 1,631,631 | 115,520 |
| two blind combined passes | 364 | 3,263,262 | 231,040 |
| 144-paragraph constituency-only diagnostic | 36 | 95,860 | 6,480 |
| complete pilot model volume | 400 | **3,359,122** | **237,520** |
| complete pilot 10% input ceiling | — | **3,695,035** | — |
| full combined paragraph shadow | 9,246 | **82,202,611** | **5,663,040** |
| full speech-factual shadow | 1,053 | 1.71M | 0.042M |
| full invocation-tone shadow | 29 est. | 0.014M | 0.001M |
| complete full shadow campaign | 10,328 est. | **83.93M** | **5.71M** |

The small diagnostic adds 95,860 input and 6,480 output proxy tokens: roughly
0.12% of the projected full paragraph input and far below the savings from
combining labels.

For a conservative seven-separate-pass baseline, proposed compact per-label static
prompts total 7,700 tokens per assignment and the same context is transmitted
seven times. That produces an estimated 132,508,999 input tokens versus
82,202,611 for the exact combined render: a reduction of 50,306,388, or 38.0%,
without assuming prompt caching. The repeated paragraph/context payload itself
falls by 85.7%. A separate-output proxy of 181 tokens per paragraph versus 160
combined gives an additional 0.74M output-token reduction (11.6%).

These are planning estimates, not a spending authorization. The finalized complete
pilot input proxy is 24.87% above the earlier 2.69M estimate, exceeding the
contract's 10% re-review threshold. That difference principally reflects exact
serialization of the full prompt/schema/template rather than the preliminary
static-overhead estimate. `ARCV1-G002`/`ARCV1-RES006` approves the 3,359,122
estimate and 3,695,035 ceiling. If the future runtime exposes a tokenizer, its
receipt must be recomputed and a further greater-than-10% change again requires
review.

## 20. Acceptance tests

Implementation is incomplete until tests prove:

### 20.1 Atomic types and bundles

- Atomic label types remain separate when emitted by one bundle.
- Constituency and `enemy_naming` can each be present or absent independently.
- One paragraph can contain both constituency and adversary roles, including the
  same literal group in distinct roles.
- A normal label type joins a bundle without core storage/workflow edits.
- A normal bundle joins `all` through registry/configuration only.
- `all` includes only locked production-eligible bundles.
- Draft constituency cannot enter a production full campaign before its gates pass.

### 20.2 Campaign determinism and keys

- Campaign expansion is deterministic.
- Manifests lock corpus, subject universe, bundle versions, and model identity.
- Selecting one bundle creates no unrelated child.
- Paragraph joins use `(doc_name, para_idx)`, never row order.
- Join cardinality and key-set equality are both enforced.
- Same-speech batching cannot alias equal `para_idx` values across speeches.
- `pilot`, `full-shadow`, `delta`, `audit`, and `explicit` selectors persist exact
  memberships.
- `delta` selects exactly new or bundle-input-changed keys.
- `full-shadow` selects the complete eligible key set.

### 20.3 Blindness and execution identity

- Both blind passes receive identical subjects, context, bundle, schema, batching,
  and local-ID maps.
- Neither renderer can read the other pass's responses.
- Comparison refuses until both passes are sealed.
- Exact model identity is recorded before assignments.
- Inadequate identity blocks assignments.
- No provider client, credentials, or network calls exist in core execution.

### 20.4 Ingestion, resume, and sealing

- Unknown, duplicate, incomplete, drifted, unassigned, out-of-corpus, and
  schema-invalid responses fail before writes.
- Completed assignments are never reissued after resume.
- Sealed runs cannot be modified or resealed.
- One successful sealed module survives another module's failure.
- Failed pilot/full runs can be superseded without rewriting history.

### 20.5 Comparison, adjudication, and promotion

- Independent model labels coexist with legacy and current labels.
- Comparison does not promote.
- Campaign completion does not promote.
- Promotions are explicit and expected-state guarded.
- Entity and constituency groups promote atomically.
- Reversal requires a new transition.

### 20.6 Raw values and publication

- Raw events preserve model output exactly.
- Normalization occurs only in typed derived projections.
- `constituency_claims` remains empty until a qualifying group is promoted.
- Materialization is byte-reproducible.
- Failed publication leaves no partial generation or changed active pointer.

## 21. Verification commands after implementation approval

Targeted:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/presidential_profiles_mpl \
  arch -x86_64 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_annotation_refresh.py \
  tests/test_annotation_workflow.py \
  tests/test_annotation_ledger.py \
  tests/test_annotation_backfill.py \
  tests/test_constituency_labeling.py
```

Provider/network static audit:

```bash
rg -n 'anthropic|openai|requests|httpx|aiohttp|API_KEY|socket' \
  src/presidential_profiles/annotation_ledger.py \
  src/presidential_profiles/annotation_workflow.py \
  src/presidential_profiles/annotation_refresh.py
```

Full local suite:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/presidential_profiles_mpl \
  arch -x86_64 .venv/bin/python -m pytest -q -p no:cacheprovider
```

Then:

```bash
~/.codex/scripts/reggie/reggie_doctor.sh \
  --repo /Users/jacobpress/Desktop/Projects/presidential_profiles
```

No verification step in this phase builds the site.

## 22. Implementation and execution stages

### Stage 0 — contract checkpoint (current)

Creates only:

- `notes/annotation-refresh-campaign-v1.md`;
- `notes/annotation-refresh-campaign-review-queue-v1.json`;
- `notes/annotation-refresh-campaign-review-report-v1.md`.

Human checkpoint: `ARCV1-C001` was approved on 2026-07-24. Stage 1 is authorized.

### Stage 1 — source, registries, and tests

Completed under `ARCV1-RES001`; it created or changed:

- new `src/presidential_profiles/annotation_refresh.py`;
- additive bundle-mode changes in
  `src/presidential_profiles/annotation_workflow.py`;
- additive v2-spec and promoted-constituency projection support in
  `src/presidential_profiles/annotation_ledger.py`;
- narrowly required draft-spec routing changes in
  `src/presidential_profiles/constituency_labeling.py`;
- new candidate label specs under `data/annotation_ledger/specs/`;
- new bundle registry/specs under `data/annotation_ledger/bundles/`;
- new `tests/test_annotation_refresh.py`;
- focused compatibility additions to the four existing annotation test modules;
- documentation routing required for the approved system.

It does not create a campaign, selection, assignment, response, sealed run,
decision, promotion, or materialized generation.

Verification: 55 focused annotation tests and all 2,464 repository tests passed.
The bundle and label registries validate, and static inspection found no provider
client, credential, or network call in the core modules.

Human checkpoint: review implementation, exact prompts/schemas, test evidence, and
recomputed estimates before any pilot plan is initialized.

### Stage 2 — persisted pilot plan

Planning-only authority was granted by `ARCV1-RES003`. It may create:

- a content-addressed 722-row pilot selection;
- nested content-addressed 144-row diagnostic and 240-row reference selections;
- a content-addressed non-executable campaign plan;
- exact local token, batching, and corpus receipts.

Human checkpoint: inspect exact keys, strata, prompts, identities, and token ceiling
before the first assignment.

`ARCV1-RES004` approved the 482-core amendment and evaluation-frame boundary. Stage
2 then published:

- the three selection receipts listed in Section 11.2;
- plan
  `plan_3af71613624c605209c169717c7c73024235e01ba0750d046940a01d2512a17f`;
- the exact fixed 182/182/36 assignment partition; and
- the deterministic token receipt in Section 19.

The plan is explicitly non-executable. It carries no runtime model ID, campaign
ID, child run ID, or assignment ID. `ARCV1-S003` is satisfied.
`ARCV1-G002`/`ARCV1-RES006` records approval of the exact plan. `ARCV1-G001`
was later satisfied by `ARCV1-RES007`, followed by execution authorization
`ARCV1-RES008` and assignment-free initialization `ARCV1-RES009`.
The immutable plan receipt retains its publication-time `G002: pending` field;
`ARCV1-RES006` is the append-only external transition that approves it.

Verification: 57 focused annotation tests and all 2,466 repository tests passed;
an idempotent second planner run reproduced the same plan/selection IDs and file
hashes; the active Phase 1 generation and its counts remained unchanged; and the
portable Reggie Doctor reported 0 errors, 0 warnings, and 5 informational
findings.

### Stage 3 — pilot execution and freeze decision

Assignment-free initialization is complete under `ARCV1-RES009`. Campaign
`camp_7cb35b8b739bcc2fb36ebe9609e81d629fee97c2d604cad5bd64d7cc25b5b316`
contains the fixed composition-diagnostic, blind-A, and blind-B child runs with
144/722/722 subjects and zero assignments, responses, or events.

The three child runs were later completed with exact selection coverage and
passing registry-v2 structural audits. Stage 3 initially stopped at
`ARCV1-B001` because the ingest path persists accepted responses only and no
authoritative rejected-attempt log exists. Before reference labeling or any
candidate-result inspection, the repository owner issued `ARCV1-RES010`,
attested that the responses were machine-produced without owner payload edits,
and removed first-response validity and its derived margins from the decision
set. The original policy remains immutable; a content-addressed amendment
records the effective gate set. Accepted-response completeness is not
misrepresented as first-response validity.

Fresh restricted sessions completed the two combined passes, constituency
diagnostic, 240-subject independent reference, and 236-subject post-reference
adjudication. All five runs are sealed. Five hash-only comparisons were
published, and all 708 declared disputes have append-only adjudication
decisions.

Evaluation
`eval_bbb2212dea6936070b23bff11100615c602396b72d2fe6058cc707864f060b61`
used the effective policy
`sha256:43c4b5c1ff7bdb6d43c7586b1413e5db8ef6df8b08419a815b52d689b17098ae`,
the exact 240-subject reference frame, 2,000 deterministic
speech-clustered bootstrap draws, and the unchanged active Phase 1 generation.
It passed 114/150 gates and failed 36:

- structural: 13/13 passed;
- composition: 6/6 passed;
- constituency: 44/57 passed; and
- existing labels: 51/74 passed.

Because the contract requires every gate to pass, `ARCV1-RES011` rejects
`ARCV1-G003`. No `constituencies@v1` specification,
`paragraph_judgment_v2@v2` bundle, or passing campaign outcome was frozen or
published. Current labels remain unchanged. Any future candidate redesign is a
new pre-results contract and campaign, not a reinterpretation of this result.

The separate invocation-tone audit proposal remains non-executable under
pending item `ARCV1-D018`; explicit owner approval or revision is required
before any invocation labeling.

### Stage 4 — full shadow refresh

After explicit full-campaign approval, may create:

- three full-shadow child work and sealed runs;
- a sealed campaign outcome;
- comparison/evaluation records;
- a new derived content-addressed materialization.

Current labels remain unchanged.

Human checkpoint: review each bundle and label type independently.

### Stage 5 — optional promotion

Only a separate explicit promotion approval may create:

- append-only adjudications;
- append-only promotion batches;
- a new materialized generation reflecting those transitions;
- promoted typed constituency rows where approved.

Sealing, adjudication, and promotion are irreversible historical publications.
Promotion can be functionally reversed only by another append-only transition.

### Stage 6 — site work

Site generator changes, `docs/` rebuild, and deployment are separate future
authorizations. They are not implied by any campaign or promotion approval.

## 23. Reversible and irreversible later actions

Reversible before publication:

- source/registry/test edits;
- a draft campaign plan;
- unsealed work that is abandoned rather than mutated;
- recomputing derived token estimates;
- building a temporary materialization candidate.

Irreversible or economically irreversible:

- model token expenditure;
- publishing a sealed run or campaign;
- publishing an adjudication;
- publishing a promotion transition.

A content-addressed materialization is derived and can be superseded, but an
activated pointer changes what consumers see. A promotion is reversible only by a
new historical transition, never deletion.

## 24. Approval request

Approval `ARCV1-RES001` authorized Stage 1, which is complete.
`ARCV1-RES003` authorized Stage 2 planning only, and `ARCV1-RES004` approved the
sampling and evaluation-frame amendments. Stage 2 planning is complete.

`ARCV1-G002` is approved through `ARCV1-RES006`. `ARCV1-G001` is satisfied
through the exact `gpt-5.6-sol`/`high` receipt in `ARCV1-RES007`, and
`ARCV1-G006` is approved through `ARCV1-RES008`. `ARCV1-RES009` records the
assignment-free campaign initialization.

Each first or resumed assignment still requires a fresh restricted labeling
session to expose the same exact model and effort and persist a blindness receipt
before `next`. `ARCV1-RES011` rejects the paragraph/constituency freeze because
the pilot failed 36 frozen gates. No present decision authorizes a
specification freeze, full-shadow refresh, promotion, materialization, site
generation, or deployment. `ARCV1-D018` must be approved or revised before any
invocation-tone labeling.

## 25. Additive provisional corpus planning

`ARCV1-RES012` approves an additive `provisional-corpus` planning path only.
This path does not weaken or repurpose `full-shadow`: the existing full-shadow
scope continues to reject every bundle that is not `production_eligible`.
`paragraph_judgment_v2_candidate@candidate-1` remains `pilot_only` after the
failed freeze.

The planner treats existing Sonnet, existing Opus subset, evaluated Sol subset,
Sol adjudication, and a possible future Sol corpus layer as distinct evidence.
Agreement across layers may later support categorical confidence metadata, but
historical layers are not overwritten or collapsed and none is declared human
ground truth.

For the evaluated 240-paragraph selection, the immutable reuse manifest
`reuse_d5f9059121d080eab7ae31f0b4cdcf2d1be9c5bf8d3b1a6d411a4598f3ad7ec1`
contains one final value for each of seven label types per subject. It derives
972 three-way-unanimous fields and 708 adjudicated fields from the exact sealed
reference, blind-A, blind-B, and adjudication evidence. Unanimous entries select
the independent-reference group while retaining all three supporting groups and
seals. Disputed entries select the published adjudication final and retain the
original groups/hashes, resolved group, published adjudication ID, decision
artifact set, and whether the result matched a majority, the reference alone,
one non-reference source, or no source. Any missing, duplicate, unclear, or
unresolved final fails planning.

Fresh selection
`fresh_1ab0bf0f22825aa61e83e54744d90bf0b23793e748e3d71d7801f8e8b9533cd3`
derives the current complete 35,394-paragraph eligible universe from the
registered bundle input and campaign-locked corpus snapshot, then excludes the
evaluated 240 by canonical identity. The exact 35,154-subject complement
includes all other 482 pilot paragraphs and retains source-text and bundle-input
fingerprints. Reuse and fresh keys are disjoint and their union is complete.

Plan
`pcplan_541b52ab42e978227017f4d2ca5f67d7e624e5a4d76b8b256b8fb371577427bf`
locks the reuse and fresh artifacts, 9,186 deterministic same-speech
assignments, exact bundle/prompt/schema/context/render/registry hashes,
`gpt-5.6-sol` at `high`, the corpus and active Phase 1 snapshot, every
evaluation/reference/comparison/adjudication/seal identity, an 81,663,911-input
token proxy, and an 89,830,303-token ceiling.

Later assembly, if separately authorized and executed, key-joins the evaluated
reuse values and fresh run values on `(canonical_subject_id, label_type)`.
Reused and fresh provenance remain distinct; raw/source layers, current primary
labels, promotions, and the active materialization pointer remain unchanged.
The composite layer is `provisional` or `experimental`, never
`production_eligible`.

The CLI is:

```bash
arch -x86_64 .venv/bin/python -m presidential_profiles.annotation_refresh \
  plan-provisional-corpus --approval-id ARCV1-RES012
```

It publishes only the three planning artifacts. `ARCV1-G007` requires separate
repository-owner authorization before campaign/run initialization, assignment
creation, model execution, or response writing. Planning also prohibits sealing,
new adjudication, promotion, materialization, site generation, and deployment.

Verification under `ARCV1-RES013` passed 10 focused planner tests, all 61
annotation-refresh/workflow/ledger tests, Python compilation, registry and
review-queue validation, provider/network exclusion, content identities,
byte-identical idempotent replanning, side-effect counts, and whitespace checks.
The full repository suite reported 2,497 passes and one unrelated documented
story-page copy failure in
`TestProseExtraThreading::test_omitting_prose_extra_entirely_still_builds`.
Portable Reggie Doctor reported 0 errors, 0 warnings, and 5 informational
findings.
