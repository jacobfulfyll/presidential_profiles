# Annotation ledger v1 implementation contract

Status: **approved for Phase 1 implementation on 2026-07-24**  
Contract date: 2026-07-24 (date precision; no time is inferred)  
Repository: `presidential_profiles`

This document freezes the implementation contract for Phase 1 of the provider-neutral,
append-only annotation ledger. It is not an implementation report. The repository owner
explicitly approved this checkpoint on 2026-07-24. Site rebuild and deployment remain prohibited.

The machine-readable review queue is
`notes/annotation-ledger-review-queue-v1.json`. Its readable rendering is
`notes/annotation-ledger-review-report-v1.md`.

## 1. Authority, scope, and non-goals

The approved user instructions dated 2026-07-24 are the authority for this contract. When this
document is approved, Phase 1 is limited to:

- a provider-neutral label-specification registry;
- generic append-only runs and label events;
- immutable sealing;
- complete legacy backfill from all surviving frozen artifacts and all 15 manifests;
- canonical projection onto the corrected corpus;
- the approved continuity promotions;
- the trusted 13-paragraph correction overlay;
- offline terminal workflow foundations;
- deterministic materialization, comparison, adjudication, and promotion foundations;
- an append-only human-review queue and readable report;
- focused and full verification.

Phase 1 does **not** include:

- a production constituency pass, production constituency dataset, publication constituency
  specification, or constituency promotion;
- rewriting any file below `data/llm_annotations/`;
- modifying `data/corpus_corrections/manifest_v1.json`;
- hand-editing or rebuilding `docs/`;
- deploying, staging, committing, pushing, or opening a pull request;
- adding a Codex/Reggie pipeline adapter;
- choosing or invoking a model from ledger code.

Python commands use `arch -x86_64 .venv/bin/python`. Reggie remains in documentation mode and
uses the portable doctor.

## 2. Source-of-truth boundaries

### 2.1 Immutable inputs

The following are read-only inputs:

- every file under `data/llm_annotations/`;
- `data/corpus_corrections/manifest_v1.json`;
- the raw and normalized source corpus used by the correction layer;
- the committed correction mapping, exclusion, reannotation, canonical-corpus, and metadata
  artifacts under `data/corpus_corrections/`.

Before backfill, Phase 1 writes a byte inventory containing the path, byte length, and SHA-256 of
every frozen input. The final gate recomputes the inventory and requires byte equality. A mismatch
aborts before publication.

### 2.2 Ledger root

All new ledger state lives under:

```text
data/annotation_ledger/
  specs/
  work/
  sealed_runs/
  decisions/
    adjudications/
    promotions/
  materialized/
    generations/
    current
  review/
```

No ledger writer accepts an output path outside this root. Run IDs, spec names, versions, and
decision IDs are validated identifiers, never unchecked path components.

### 2.3 Logical versus physical storage

Analysts see unified logical tables:

- `annotation_runs`
- `label_events`
- `canonical_label_projection`
- `label_adjudications`
- `label_promotions`
- `current_labels`
- `evaluation_metrics`
- typed projections, initially `constituency_claims`

There is not a separate logical schema per run. Each sealed run contributes rows to the same
logical tables. The historical source of truth is the immutable sealed artifact set for each run;
Parquet generations are derived caches.

### 2.4 Implementation ownership

After approval, the Phase 1 implementation surface is:

- new `src/presidential_profiles/annotation_ledger.py` for canonical serialization, IDs, explicit
  schemas, sealed storage, decision transitions, and deterministic materialization;
- new `src/presidential_profiles/annotation_workflow.py` for the provider-neutral offline terminal
  lifecycle;
- new `src/presidential_profiles/annotation_backfill.py` for frozen legacy import and canonical
  projection;
- a narrow update to `src/presidential_profiles/constituency_labeling.py` so its existing
  assignment/validation foundation delegates to the generic workflow and exposes only the draft
  constituency spec rather than a production constituency dataset;
- new registry/spec JSON below `data/annotation_ledger/specs/`;
- focused tests in `tests/test_annotation_ledger.py`,
  `tests/test_annotation_workflow.py`, and `tests/test_annotation_backfill.py`, plus only the
  compatibility updates required in `tests/test_constituency_labeling.py`.

Backfill, sealed-run, decision, review, and materialized files below `data/annotation_ledger/` are
created only after the source/test foundations pass their focused gates. No site generator or
generated-site path is part of this implementation surface.

## 3. Canonical serialization, hashes, and identifiers

### 3.1 Canonical JSON

Canonical JSON is UTF-8 JSON with:

- object keys sorted lexicographically;
- separators `,` and `:` with no insignificant whitespace;
- Unicode emitted directly, not ASCII-escaped;
- JSON booleans and null;
- no NaN or infinity;
- list order preserved;
- string spelling, punctuation, whitespace, capitalization, and case preserved.

Canonicalization is serialization, not value normalization. Historical topic strings, entity
names, categories, case variants, list order, and duplicates remain unchanged.

Hashes are lowercase SHA-256 written as `sha256:<64 lowercase hexadecimal characters>`.

### 3.2 Identifier formulas

The implementation uses the canonical JSON of the named identity object:

| Identifier | Exact formula |
|---|---|
| `spec_sha256` | SHA-256 of the entire spec object with its `spec_sha256` field omitted |
| `subject_id` | `sub_` + SHA-256 hex of `{"subject_key": subject_key_json, "subject_type": subject_type}` |
| `assignment_id` | `asg_` + SHA-256 hex of the assignment object with checksums omitted |
| `label_group_id` | `lgrp_` + SHA-256 hex of `{run_id,label_type,spec_sha256,subject_id,source_response_id}` |
| `label_id` | `lbl_` + SHA-256 hex of `{label_group_id,event_role,item_index,raw_value_sha256,source_duplicate_ordinal}` |
| `projection_id` | `proj_` + SHA-256 hex of `{label_id,canonical_subject_id,mapping_rule_id}` |
| `adjudication_id` | `adj_` + SHA-256 hex of the immutable adjudication payload with timestamps omitted |
| `promotion_id` | `prom_` + SHA-256 hex of the immutable promotion transition with timestamps omitted |
| `promotion_batch_id` | `pbat_` + SHA-256 hex of the sorted complete promotion transition list |
| `metric_id` | `met_` + SHA-256 hex of the immutable metric or evaluation-artifact row |
| `artifact_id` | `art_` + the artifact byte SHA-256 hex |

`run_id` is a human-readable, globally unique identifier. Legacy run IDs are preserved verbatim.
New run IDs must match `[A-Za-z0-9][A-Za-z0-9._-]{0,159}`. A run ID is immutable after its work
directory is created and cannot contain a slash, backslash, whitespace, or `..`.

### 3.3 Subject keys

Core storage does not depend on a permanently closed subject enum. Phase 1 registers these shapes:

| `subject_type` | Canonical `subject_key_json` |
|---|---|
| `paragraph` | `{"doc_name":"…","para_idx":N}` |
| `speech` | `{"doc_name":"…"}` |
| `invocation_span` | `{"char_end":N,"char_start":N,"doc_name":"…"}` |

Convenience columns `doc_name`, `para_idx`, `char_start`, and `char_end` mirror these keys and are
validated against `subject_key_json`. Paragraph membership and reconciliation always use
`(doc_name, para_idx)`, never row position.

## 4. Label-specification registry

### 4.1 Registry contract

`data/annotation_ledger/specs/registry-v1.json` is a versioned registry. A registered spec file is
stored at:

```text
data/annotation_ledger/specs/<label_type>/<spec_version>.json
```

Each spec contains exactly:

- `label_type`: registered string name;
- `spec_version`: immutable version string;
- `stability`: `frozen`, `legacy_partial`, or `draft_pilot`;
- `subject_type`;
- `historical_quantity`;
- `prompt_version`, complete prompt/rubric text when it survives, and `prompt_sha256`;
- complete response JSON Schema and `response_schema_sha256`;
- `value_kind`;
- `event_extractor`: declarative JSON Pointer(s), group rule, item rule, and empty-result rule;
- `context_policy`;
- optional named `special_validator`;
- `source_spec_refs`;
- `limitations`;
- `spec_sha256`.

The storage engine and terminal lifecycle consume this metadata. `label_type` is not a closed enum
in core code. Adding an ordinary label type requires a spec, prompt/schema, optional separately
registered validator, tests, and optionally a typed projection. It does not require edits to core
event storage, run lifecycle, sealing, materialization, adjudication, or promotion.

### 4.2 Initial registry

| Label type | Spec version | Subject | Value/event shape | Stability | Frozen source identity |
|---|---|---|---|---|---|
| `topics` | `paragraph-judgment-v1` | paragraph | one list event | frozen | prompt hash `sha256:6fa617391761c6b7b0e934e3a4782aa28d259d15678e4e4e4665c8b2d93ecdfe` |
| `party_attack` | `paragraph-judgment-v1` | paragraph | one boolean event | frozen | same judgment prompt hash |
| `enemy_naming` | `paragraph-judgment-v1` | paragraph | one boolean event | frozen | same judgment prompt hash |
| `zero_sum` | `paragraph-judgment-v1` | paragraph | one boolean event | frozen | same judgment prompt hash |
| `proposal_values` | `paragraph-judgment-v1` | paragraph | one category event | frozen | same judgment prompt hash |
| `entities` | `paragraph-judgment-v1` | paragraph | zero or more object events in one atomic group | frozen | same judgment prompt hash |
| `speech_type` | `speech-factual-v1` | speech | one category event | frozen | prompt hash `sha256:450c667db92e324c1a1455914dd17528a999c88d3e32989e0d32fb3a0a7e95f1` |
| `audience` | `speech-factual-v1` | speech | one category event | frozen | same factual prompt hash |
| `medium` | `speech-factual-v1` | speech | one category event | frozen | same factual prompt hash |
| `invocation_tone` | `legacy-2026-07-12` | invocation_span | one category event | legacy_partial | surviving-method hash `sha256:5952db301a5c0e4aa1f7d567a467df2cf66b37f7744b7f35cb4595bf391d7845` |
| `constituencies` | `draft-2026-07-24` | paragraph | one structured object event; typed claims projection | draft_pilot | draft prompt hash `sha256:2bce252a681f666de8e4b404d95bf5390fa6215c0fe2177fc68a051de17c08a0`; draft schema hash `sha256:cba713949eecc2a16bbeebe395ccf74d52d7fd02cefacc59b781940abecaab30` |

The constituency spec is a pilot candidate, not publication version 1. Phase 1 creates no
production constituency run.

### 4.3 Generic value support

`value_kind` accepts `boolean`, `integer`, `number`, `string`, `category`, `list`, `object`, or
`null`. `raw_value_json` stores the canonical JSON representation. Specs impose semantic
constraints; the core storage engine only enforces valid JSON, the declared kind, hashes, keys,
and registered spec identity.

Entity responses use one `label_group_id` for all entity objects returned for one paragraph,
label type, run, and source response. Each entity is a separate value event with an `item_index`.
An observed empty entity array in a new run produces one `empty_group_marker` event with
`raw_value_json=[]`. Legacy backfill never creates that marker because the discarded legacy
response cannot be reconstructed from the absence of rows.

## 5. Exact logical table schemas

All string fields are UTF-8. Nullable fields are explicitly nullable. Timestamps are
`timestamp[us, UTC]`; dates are `date32`. JSON columns contain canonical JSON strings. Parquet
writers use explicit Arrow schemas rather than data-inferred physical types.

### 5.1 `annotation_runs`

Primary key: `run_id`.

| Column | Type | Null | Meaning |
|---|---|---:|---|
| `run_id` | string | no | Preserved legacy or validated new run ID |
| `run_kind` | string | no | `legacy_backfill`, `terminal_annotation`, or `adjudication` |
| `run_status` | string | no | Always `sealed` in a published generation |
| `scope` | string | no | `legacy_surviving`, `selection`, `pilot`, or `full` |
| `label_spec_map_json` | string | no | Sorted label type → version/hash map |
| `provider_id` | string | yes | Provider as recorded; null when unknown |
| `model_id` | string | yes | Exact stored/runtime model identity |
| `model_identity_source` | string | no | `legacy_manifest`, `runtime_metadata`, or `unknown` |
| `annotator_id` | string | yes | Assignment identity |
| `agent_id` | string | yes | Optional task/agent identity |
| `reasoning_effort` | string | no | `low`, `medium`, `mixed`, or `unknown` |
| `reasoning_effort_provenance` | string | no | `inferred_from_versioned_spec`, `observed`, or `unknown` |
| `reasoning_effort_by_spec_json` | string | yes | Exact per-spec map for mixed runs |
| `manifest_date` | date32 | yes | Legacy date as stored |
| `assigned_at` | timestamp[us, UTC] | yes | Null when unknown |
| `labeled_at` | timestamp[us, UTC] | yes | Null when unknown |
| `ingested_at` | timestamp[us, UTC] | no | New ledger/backfill ingestion time |
| `sealed_at` | timestamp[us, UTC] | no | Seal time |
| `timestamp_precision` | string | no | `date`, `microsecond`, or `unknown` |
| `source_corpus_fingerprint` | string | yes | Original corpus identity |
| `canonical_corpus_fingerprint` | string | yes | Corrected corpus identity |
| `selection_artifact_id` | string | yes | Persisted selection, if any |
| `authorization_review_item_id` | string | yes | Required for trusted/special runs |
| `human_validation_status` | string | no | `not_reviewed`, `human_validated`, or `not_applicable` |
| `independent_check_status` | string | no | `not_checked`, `independently_checked`, or `not_applicable` |
| `source_manifest_path` | string | yes | Frozen manifest path |
| `source_manifest_sha256` | string | yes | Frozen manifest byte hash |
| `artifacts_json` | string | no | Registered run artifact metadata |
| `limitations_json` | string | no | Explicit provenance limitations |
| `run_manifest_sha256` | string | no | Canonical sealed run manifest hash |
| `artifact_set_sha256` | string | no | Root hash of the sealed set |

### 5.2 `label_events`

Primary key: `label_id`. A group is atomic on `label_group_id`.

| Column | Type | Null | Meaning |
|---|---|---:|---|
| `label_id` | string | no | Immutable event ID |
| `label_group_id` | string | no | Atomic answer/label-type group |
| `run_id` | string | no | Foreign key to a sealed run |
| `label_type` | string | no | Registered name |
| `spec_version` | string | no | Registered version |
| `spec_sha256` | string | no | Exact registered spec hash |
| `subject_type` | string | no | Registered subject shape |
| `subject_id` | string | no | Historical subject ID |
| `subject_key_json` | string | no | Canonical source key |
| `doc_name` | string | yes | Validated convenience key |
| `para_idx` | int64 | yes | Validated convenience key |
| `char_start` | int64 | yes | Span key |
| `char_end` | int64 | yes | Span key |
| `source_text_sha256` | string | yes | Exact source text or transcript hash |
| `assignment_input_sha256` | string | yes | Exact assignment input when known |
| `event_role` | string | no | `value` or `empty_group_marker` |
| `item_index` | int64 | yes | Stable within-group index |
| `group_size` | int64 | yes | Observed value count; zero on an observed empty marker |
| `value_kind` | string | no | Declared generic kind |
| `raw_value_json` | string | no | Preserved raw value serialized canonically |
| `raw_value_sha256` | string | no | Hash of `raw_value_json` |
| `assignment_id` | string | yes | Terminal assignment |
| `batch_id` | string | yes | Terminal batch |
| `source_response_id` | string | no | Stable response/final-row grouping identity |
| `source_artifact_id` | string | no | Frozen or sealed source artifact |
| `source_table` | string | yes | Legacy table name |
| `source_key_json` | string | no | Key-based source locator |
| `source_duplicate_ordinal` | int64 | no | Distinguishes byte-identical duplicate source rows |
| `assigned_at` | timestamp[us, UTC] | yes | Trusted local timestamp or null |
| `labeled_at` | timestamp[us, UTC] | yes | Trusted local timestamp or null |
| `ingested_at` | timestamp[us, UTC] | no | Backfill/terminal ingestion time |
| `timestamp_precision` | string | no | Precision of source label time |
| `provenance_status_json` | string | no | Observed/inferred/unknown field-level status |

No uniqueness constraint forbids the same subject and label type across runs. Independent models
and versions coexist.

### 5.3 `canonical_label_projection`

Primary key: `projection_id`. Every event gets a projection row, including excluded history.

| Column | Type | Null |
|---|---|---:|
| `projection_id` | string | no |
| `label_id` | string | no |
| `label_group_id` | string | no |
| `source_subject_id` | string | no |
| `canonical_subject_id` | string | yes |
| `canonical_subject_key_json` | string | yes |
| `canonical_doc_name` | string | yes |
| `canonical_para_idx` | int64 | yes |
| `mapping_status` | string | no |
| `mapping_rule_id` | string | no |
| `source_text_sha256` | string | yes |
| `canonical_text_sha256` | string | yes |
| `source_input_sha256` | string | yes |
| `canonical_input_sha256` | string | yes |
| `eligible_for_promotion` | bool | no |
| `review_item_id` | string | yes |
| `reason` | string | no |

All events in one label group must have identical projection eligibility and canonical subject.

### 5.4 `label_adjudications`

Primary key: `adjudication_id`.

| Column | Type | Null |
|---|---|---:|
| `adjudication_id` | string | no |
| `canonical_subject_id` | string | no |
| `label_type` | string | no |
| `spec_sha256` | string | no |
| `candidate_label_group_ids_json` | string | no |
| `resolution_kind` | string | no |
| `selected_label_group_id` | string | yes |
| `resolved_label_group_id` | string | yes |
| `adjudicator_id` | string | no |
| `rationale` | string | no |
| `review_resolution_id` | string | yes |
| `decided_at` | timestamp[us, UTC] | no |
| `artifact_set_sha256` | string | no |

`resolution_kind` is `select_existing`, `synthesize_in_adjudication_run`, `reject_all`, or
`defer`. A synthesized resolution is stored as label events in its own sealed adjudication run.
Candidate events are never changed or deleted.

### 5.5 `label_promotions`

Primary key: `promotion_id`.

| Column | Type | Null |
|---|---|---:|
| `promotion_id` | string | no |
| `promotion_batch_id` | string | no |
| `promotion_channel` | string | no |
| `canonical_subject_id` | string | no |
| `label_type` | string | no |
| `action` | string | no |
| `label_group_id` | string | yes |
| `replaces_label_group_ids_json` | string | no |
| `expected_current_state_sha256` | string | no |
| `promotion_rule_id` | string | no |
| `adjudication_id` | string | yes |
| `review_resolution_id` | string | yes |
| `operator_id` | string | no |
| `promoted_at` | timestamp[us, UTC] | no |
| `source_run_artifact_set_sha256` | string | no |
| `artifact_set_sha256` | string | no |

`action` is `activate` or `deactivate`. A transition must state the complete set it replaces and
the expected pre-transition state hash. A stale expectation refuses the entire promotion batch.
Timestamps do not choose winners.

### 5.6 `current_labels`

`current_labels` is derived, never directly written. It contains the value events belonging to the
active group after replaying explicit promotion transitions.

Primary key: `(promotion_channel, canonical_subject_id, label_type, label_id)`.

It carries `promotion_id`, `promotion_batch_id`, `label_group_id`, `label_id`, `run_id`,
`label_type`, spec identity, canonical subject keys, event role/index, raw value, and the explicit
rule/adjudication/review IDs. There is at most one active group for
`(promotion_channel, canonical_subject_id, label_type)`. Multi-row entity groups remain atomic.

### 5.7 `evaluation_metrics`

Primary key: `metric_id`. `record_kind` is `metric` or `evaluation_artifact`.

Columns are `metric_id`, `record_kind`, `evaluation_id`, `label_type`, `left_run_id`,
`right_run_id`, `stratum_json`, `metric_name`, `metric_value` (float64 nullable), `n` (int64
nullable), `source_value_json`, `artifact_role`, `artifact_path`, `artifact_sha256`,
`source_artifact_id`, and `provenance_json`.

The 135 rows of `agreement_v1.parquet` become metric records without changing `field`, `era_bin`,
`era_label`, `metric`, `value`, or `n`; those raw cells are retained in `source_value_json`. The
persisted agreement sample becomes an `evaluation_artifact` record.

### 5.8 `constituency_claims`

This typed projection has an explicit schema even when empty:

`promotion_channel`, `label_id`, `label_group_id`, `run_id`, `doc_name`, `para_idx`, `outcome`,
`claim_index`, `group_text`, `normalized_group`, `group_type`, `relation`, `evidence_span`,
`certainty`, `notes`, spec identity, model/annotator provenance, corpus fingerprint, and promotion
identity.

Phase 1 materializes zero rows with these physical types. No production constituency dataset is
created.

## 6. Immutable run artifacts and sealing

### 6.1 Work state

`init-run` creates `data/annotation_ledger/work/<run_id>/`. Assignments and responses are
append-only files. An existing assignment or response is never overwritten. Local code supplies
keys, subject/input hashes, assignment identity, model/annotator identity receipts, and
timestamps; a chat response supplies label values only.

### 6.2 Seal contents

A sealed run contains:

- `run.json`;
- exact spec snapshots;
- `artifacts.json`;
- assignments and response artifacts, when they survive;
- canonical `label_events.jsonl`;
- `audit.json`;
- `source-inventory.json`;
- `seal.json`.

`seal.json` lists every member path, byte length, and SHA-256 and contains the root
`artifact_set_sha256`. The root is the SHA-256 of the canonical sorted member inventory, excluding
`seal.json` itself.

### 6.3 Atomic seal publication

`seal-run`:

1. acquires a ledger lock;
2. validates registry/spec hashes, corpus and selection fingerprints, assignments, complete
   responses, key sets, values, source artifacts, and the run-specific completion policy;
3. writes and fsyncs the complete candidate set below a candidate parent directory, with the
   artifact files nested at `<artifact_set_sha256>/`;
4. recomputes every member hash and the root hash;
5. refuses if `sealed_runs/<run_id>/` already exists;
6. atomically renames the complete candidate parent to `sealed_runs/<run_id>/`, making
   `sealed_runs/<run_id>/<artifact_set_sha256>/` visible as one complete set;
7. fsyncs the parent directory.

A failure before the rename leaves only recoverable work state. A successful rename exposes the
whole set at once. Every command refuses to append to or reseal a sealed run. Verification
re-hashes the set on every read. Filesystem permissions are defense in depth; hashes and refusal
logic are the invariant.

Legacy JSONL members may use deterministic gzip transport (compression level 9, filename omitted,
header mtime zero) while retaining the contract filenames. Decompression yields the canonical
UTF-8 JSONL stream exactly; hashing and sealing cover the physical compressed bytes, while
logical validation covers every decompressed row.

Legacy backfill may be partial relative to an original request only because its declared scope is
`legacy_surviving`. It must be complete relative to every surviving frozen row.

## 7. Provider-neutral terminal workflow

The generalized module exposes:

```text
init-run → next → ingest → status → audit → seal-run
         → materialize → compare → adjudicate → promote
```

- `init-run` resolves registered specs, fingerprints the exact subject selection, persists run
  identity, and refuses unknown/drifted specs or keys.
- `next` claims or resumes a stable bounded assignment and prints readable context plus a strict
  response template.
- `ingest` accepts JSON, validates the complete assigned key set and spec, then appends one
  immutable response artifact atomically.
- `status` reports assignments, coverage, open work, invalid work, and breakdowns available from
  the subject metadata.
- `audit` replays all validation without writing.
- `seal-run` performs the immutable publication above.
- `materialize` builds a full content-addressed logical generation from sealed sources.
- `compare` compares selected runs without overwriting either.
- `adjudicate` records, but never silently resolves, an explicit human decision.
- `promote` applies an explicit atomic state transition after validating sealed inputs.

No core terminal module imports a provider/model client, reads API credentials, chooses a model,
invokes a model, or makes a network call. This is tested on the real workflow by bombing model
client imports and socket connection methods.

## 8. Complete legacy backfill contract

### 8.1 All 15 manifests

Backfill creates one sealed `legacy_backfill` run for each existing manifest:

1. `2026-07-12-invocation-tone-fable5`
2. `factual-full-20260720`
3. `factual-r2-20260720`
4. `judgment-chunk1-20260720`
5. `judgment-chunk10-20260720`
6. `judgment-chunk2-20260720`
7. `judgment-chunk25-20260720`
8. `judgment-chunk5-20260720`
9. `judgment-full-20260720`
10. `judgment-r2-20260720`
11. `opus-factual-20260721`
12. `opus-judgment-20260721`
13. `opus-judgment-r2-20260722`
14. `pilot-v2-20260720`
15. `taxonomy-v1-20260719`

Every source row's existing `run_id` must resolve to its own manifest. Unknown or unused row run
IDs fail the backfill. A manifest without label rows remains a run and carries its artifacts.

### 8.2 Label extraction

- Each primary and Opus paragraph-annotation row creates events for `topics`, `party_attack`,
  `enemy_naming`, `zero_sum`, and `proposal_values`.
- Each surviving primary and Opus entity row creates one `entities` object event. Rows sharing
  `(run_id, doc_name, para_idx)` share one entity `label_group_id`.
- Each primary and Opus speech row creates `speech_type`, `audience`, and `medium` events.
- Each invocation-tone row creates one `invocation_tone` event keyed by
  `(doc_name, char_start, char_end)`.
- `taxonomy_v1.json` and `crosswalk_v1.json` are registered artifacts of
  `taxonomy-v1-20260719`, not label events.
- `agreement_v1.parquet` is imported as evaluation metrics, not label events.
- `agreement_sample_v1.json` is an evaluation artifact.

The backfill preserves the exact surviving values and multiplicity. It does not normalize topic
case, entity names, categories, list order, or duplicates. Any normalized view is a separate typed
derived projection.

No empty entity event is synthesized. Absence of a legacy entity row stays absence and is not
represented as a recovered model response.

### 8.3 Legacy provenance

- The manifest date is retained as `manifest_date` with `timestamp_precision=date`.
- `assigned_at` and `labeled_at` stay null when unknown.
- The new backfill `ingested_at` is recorded separately.
- Filesystem modification times are never provenance.
- Paragraph judgment v1 effort is `medium` with
  `reasoning_effort_provenance=inferred_from_versioned_spec` only when the stored prompt hash is
  the exact frozen judgment hash.
- Speech factual v1 effort is `low` under the same exact-hash rule.
- `pilot-v2-20260720` is run-level `mixed` with the exact per-spec map
  `{"paragraph-judgment-v1":"medium","speech-factual-v1":"low"}` and inferred provenance.
- Unsupported runs are `unknown`.
- Inferred provenance is never marked observed.
- Every relevant run limitation states: surviving final labels are preserved, but discarded
  intermediate model responses cannot be recreated.
- Invocation tone additionally records that the original prompt and cost are unknown and the
  surviving hash describes the method string, not a prompt.

### 8.4 Refusal counts

The backfill refuses unless it independently re-derives:

| Source | Required count |
|---|---:|
| primary paragraph judgments | 36,229 |
| Opus paragraph judgments | 8,570 |
| primary entity rows | 27,214 |
| Opus entity rows | 5,954 |
| primary speech rows | 1,057 |
| Opus speech rows | 266 |
| invocation-tone rows | 101 |
| manifests | 15 |
| agreement metric rows | 135 |

## 9. Canonical projection and continuity promotion

### 9.1 Paragraph reconciliation

The correction mapping is joined on source `(doc_name, para_idx)` with `many_to_one` validation.
The mapping itself must be one-to-one on old keys. In addition to merge cardinality, source and
mapping key-set equality is checked. Row order is never evidence.

Required independently re-derived projection counts:

| Source | Reusable exact-text | Changed | Internal-repeat excluded | Duplicate-document excluded |
|---|---:|---:|---:|---:|
| primary paragraph judgments | 35,381 | 13 | 751 | 84 |
| Opus paragraph judgments | 8,435 | 3 | 106 | 26 |
| primary entity rows | 26,570 | 8 | 542 | 94 |
| Opus entity rows | 5,848 | 3 | 87 | 16 |

All removed/changed legacy events remain historical. Excluded duplicate events receive projection
rows with no canonical subject and are never current. A removed duplicate is never remapped to a
retained copy.

### 9.2 Speech reconciliation

- Primary: 1,053 retained and 4 removed.
- Opus: 264 retained and 2 removed.
- The four removed duplicate speech records remain historical and unpromoted.

The April 3, 1968 press conference is the only approved input-mismatch carry-forward:

- `doc_name`:
  `/the-presidency/presidential-speeches/april-3-1968-press-conference`
- legacy factual-input SHA-256:
  `3754024e1ebd4990f5d8f4a1a7699654ff06ee3f0995803ec468f136b0ab2044`
- canonical factual-input SHA-256:
  `8cc60b9c0c3aaadab673853cc3e13f0973593a724004eb3a5030784cb2dbcd0f`
- both models:
  `speech_type=press_conference_or_interview`, `audience=press`,
  `medium=press_conference`
- mapping rule:
  `speech-input-mismatch-human-approved-april-3-1968-v1`
- review item:
  `ALRV1-D007`

All other retained speech inputs must match the registered factual input function exactly.
Unapproved mismatches are ineligible for promotion.

### 9.3 Invocation spans

All 101 invocation spans must remain in retained speeches and satisfy:

```text
canonical_transcript[char_start:char_end] == mention
```

Failure of any span aborts projection.

### 9.4 Promotion rules

The bootstrap promotion channel is `primary`.

`continuity-primary-exact-text-v1` activates every promotion-eligible reusable primary legacy
group under review resolution `ALRV1-D004`. It includes:

- all reusable primary paragraph judgment groups;
- all reusable primary entity groups that actually survive;
- all 1,053 retained primary speech rows, including the separately documented April 3 exception;
- all 101 reusable invocation-tone groups.

It does not activate Opus groups. Opus remains queryable as independent evidence.

The five required paragraph judgment label types (`topics`, three flags, and `proposal_values`)
must have one active primary group on all 35,381 exact-text paragraphs before the overlay and all
35,394 canonical paragraphs after it. Entities are sparse by their actual observed result.

## 10. Trusted 13-paragraph correction overlay

### 10.1 Exact scope

The selection is exactly the 13 keys in
`data/corpus_corrections/reannotation_required_v1.parquet`. Its key set and old/canonical text
hashes are copied into the assignment artifact and checked against correction metadata. No other
paragraph may enter the run.

### 10.2 Rubric and context

The overlay uses the frozen paragraph-judgment v1 rubric, response schema, closed taxonomy, and
prompt hash. The assignment shows:

- the corrected target paragraph;
- at most one preceding and one following paragraph from the same canonical speech;
- the speech decade;
- no president, party, or title in the judgment prompt.

The Codex chat agent supplies only `topics`, `party_attack`, `enemy_naming`, `zero_sum`,
`proposal_values`, and `entities` with stance. The local program supplies and validates all keys,
hashes, assignment/run IDs, source identity, and timestamps.

### 10.3 Model identity gate

Before the first assignment is shown, the local program persists a runtime metadata receipt
containing the exact model identifier exposed by the Codex runtime and how it was obtained. A
human-entered guess, marketing family name, or invented snapshot is prohibited. If the runtime
does not expose a stable identifier specific enough to distinguish the executing model, the run
pauses before labeling and creates a review item. It does not invent an identifier.

### 10.4 Trust status and promotion

The run is marked:

- `authorization_review_item_id=ALRV1-D006`;
- `human_validation_status=not_reviewed`;
- `independent_check_status=not_checked`;
- limitation: `user-authorized trusted single pass; not human-validated or independently checked`.

Sealing requires all 13 keys and all five mandatory judgment fields. The entity field must be
present in every observed response; an observed empty array is preserved as an empty-group marker.
The overlay promotion is one atomic batch and refuses unless it would produce 35,394 active primary
paragraph judgments. The 13 old same-key events remain historical and bind only to old text.

## 11. Deterministic materialization and recovery

### 11.1 Generation build

`materialize` reads only verified sealed run and decision artifacts. It:

1. resolves an ordered input inventory by IDs and artifact-set hashes;
2. validates every logical table and cross-table foreign key;
3. writes every Parquet plus `generation.json` to a temporary sibling directory;
4. uses explicit Arrow schemas, fixed column order, stable row sort, fixed writer options, and no
   wall-clock fields;
5. hashes the complete generation;
6. atomically renames it to
   `materialized/generations/<generation_sha256>/`;
7. atomically replaces `materialized/current` with the generation hash only after the complete
   generation exists.

The active pointer is derived state, not history. A failure before pointer replacement leaves the
previous generation active. A crash after generation rename but before pointer replacement leaves
a complete inactive generation that can be verified and activated or garbage-collected later.
There is never a partially replaced table/sidecar set.

Two materializations from the same sealed inputs and code/schema versions must produce identical
bytes and the same generation hash.

### 11.2 Decision publication

Adjudication and promotion artifacts use the same candidate-directory, full validation, hash
inventory, and atomic rename pattern. Promotion batches are all-or-nothing. Current labels are
recomputed from transitions; no command edits `current_labels` directly.

### 11.3 Recovery rules

- Open work is resumable from immutable assignments and responses.
- A stale lock is never deleted automatically; the operator inspects it.
- Orphan temporary candidates are ignored by readers and reported by `audit`.
- A corrupted sealed set is quarantined by refusal; it is never repaired in place.
- Recovery creates a new run or decision artifact. It never mutates sealed history.
- Rebuilding a materialized generation is always safe because sealed artifacts are authoritative.

## 12. Human-review system

Phase 1 implements append-only machine-readable review items and separate append-only resolutions.
A resolution references an existing item, the deciding identity, authority, date/time and
precision, decision, and rationale. Updating an item's apparent status is a derived view over
those records; neither the item nor prior resolution is overwritten.

The readable Markdown report is generated deterministically from the machine records. It shows:

- pending items;
- approved/rejected/deferred items;
- accepted provenance limitations;
- execution gates;
- the exact resolution authority;
- affected runs, rules, and artifacts.

No item is silently resolved. Code may refuse on a pending item; it may not infer approval.

The initial queue records all prompt-approved decisions, including:

- primary-label continuity promotion;
- no synthetic empty legacy entity events;
- the trusted 13-paragraph Codex overlay;
- the April 3, 1968 speech carry-forward;
- Opus retained but unpromoted;
- constituency remaining draft in Phase 1.

The only decision currently awaiting human review is approval of this complete contract
checkpoint. Runtime model identity is an execution gate, not a decision that can be pre-approved
without the receipt.

## 13. Acceptance tests

Phase 1 is not complete until tests prove all of the following.

### 13.1 Frozen inputs and backfill

- Every frozen input remains byte-identical before and after.
- All 15 manifests are present as runs.
- Every source row and source `run_id` reconciles exactly.
- Raw values, list order, spelling, case, and row multiplicity survive.
- No synthetic legacy empty entity event exists.
- Unknown timestamps and provenance stay null/unknown.
- Inferred effort is never observed effort.
- Taxonomy/crosswalk are run artifacts, agreement rows are metrics, and the sample is an
  evaluation artifact.

### 13.2 Keys and canonical projection

- Paragraph joins use `(doc_name, para_idx)`, never row order.
- One-to-one/many-to-one validation and explicit key-set equality both run.
- All required counts in Sections 8 and 9 re-derive exactly.
- Excluded duplicate events remain historical and non-current.
- No duplicate is remapped to a retained copy.
- The 13 old labels bind to old text; the 13 overlay labels bind to corrected text.
- All 35,394 canonical paragraphs have promoted primary topics, flags, and proposal/value labels.
- All 1,053 canonical speeches have promoted primary speech labels.
- A source-input mismatch cannot be promoted except through the recorded April 3 rule.
- All 101 invocation spans remain exact.

### 13.3 Append-only lifecycle

- Independent runs/models/versions coexist.
- A sealed run cannot be appended, replaced, or resealed.
- Current labels change only through a validated explicit promotion or approved adjudication
  transition.
- Timestamp order alone never affects current state.
- Entity group transitions are atomic.
- A stale expected-current-state hash refuses a promotion batch without writes.

### 13.4 Failure atomicity and reproducibility

- Materialization is byte-reproducible.
- Failure injection before generation rename leaves no visible generation.
- Failure injection after generation rename but before pointer replacement leaves the prior
  generation active.
- No table is visible with sidecars from another generation.
- Corrupt sealed artifacts refuse before materialization.

### 13.5 Offline terminal boundary

- Core terminal imports contain no model API client.
- The real init/next/ingest/audit/seal/materialize/promote path succeeds while client imports and
  socket connections are bombed.
- Unknown label types/specs, duplicate responses, drifted specs/corpora/assignments, incomplete
  batches, unassigned keys, out-of-corpus keys, invalid types/enums, and invalid evidence spans
  fail before any response/event write.

### 13.6 Extensibility test

A test creates an imaginary registered paragraph label type in a temporary registry, with a new
versioned prompt/schema and a structured-object value. Without editing or monkeypatching core
storage/workflow code, it must pass:

```text
init-run → next → ingest → audit → seal-run → materialize → promote
```

The resulting event must appear in `label_events` and `current_labels`. The test asserts that the
hashes of the core workflow source files are unchanged across the exercise. No typed projection
is required.

### 13.7 Constituency boundary

- `constituencies` is registered only as `draft_pilot`.
- `constituency_claims` exists with an explicit schema and zero rows.
- No full/published constituency run, event, promotion, or dataset exists.

## 14. Verification sequence after approval

1. Run focused schema/registry tests.
2. Run focused terminal lifecycle and refusal tests.
3. Run focused backfill/reconciliation tests.
4. Run focused projection/promotion/overlay tests.
5. Run deterministic and failure-injection tests.
6. Run the full relevant suite with:

   ```bash
   arch -x86_64 .venv/bin/python -m pytest -q
   ```

7. Re-run the portable Reggie doctor.
8. Do not build the site.

## 15. Checkpoint evidence and stop condition

Read-only re-derivation on 2026-07-24 confirmed:

- 36,229 primary and 8,570 Opus paragraph judgments;
- 27,214 primary and 5,954 Opus entity rows;
- 1,057 primary and 266 Opus speech rows;
- 101 invocation-tone rows;
- 15 manifests;
- 35,381/8,435 reusable primary/Opus judgments;
- 26,570/5,848 reusable primary/Opus entity rows;
- 1,053/264 retained primary/Opus speech rows;
- 101 exact reusable invocation spans;
- exactly 13 changed canonical paragraphs.

No unexplained reconciliation difference was found at this checkpoint.

Reggie doctor reported 0 errors and 0 warnings. Its five informational findings are the known
optional portable documents, existing `.claude/stats.json` telemetry, dirty user-owned worktree,
and native/external Reggie layout; none authorizes a change.

The contract checkpoint was explicitly approved on 2026-07-24. Phase 1 implementation may proceed
within this frozen scope. Runtime model identity remains a separate execution gate before overlay
labeling.
