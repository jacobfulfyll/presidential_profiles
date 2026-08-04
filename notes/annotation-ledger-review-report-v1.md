# Annotation ledger v1 review report

Status: **contract approved on 2026-07-24**  
Source: `notes/annotation-ledger-review-queue-v1.json`  
Contract: `notes/annotation-ledger-v1.md`  
Date: 2026-07-24 (date precision)

This is the readable view of the initial machine-readable review queue. It records the decisions
approved in the implementation brief, the explicit approval of the complete contract checkpoint,
accepted provenance limitations, and the model-identity gate that can only be evaluated at overlay
execution time. No item was silently resolved.

## Contract approval

### ALRV1-C001 — complete contract checkpoint

Resolution `ALRV1-R015`: approved by the repository owner through the explicit user message
“Approve.” on 2026-07-24.

Current status: `approved`.

## Approved decisions

| Item | Resolution | Applied decision |
|---|---|---|
| ALRV1-D001 | ALRV1-R001 | Use a provider-neutral, versioned registry and unified append-only tables. |
| ALRV1-D002 | ALRV1-R002 | Backfill all surviving legacy labels and all 15 manifests without normalizing or inventing provenance. |
| ALRV1-D003 | ALRV1-R003 | Carry forward fingerprint-identical retained labels; keep removed duplicates historical and never remap them. |
| ALRV1-D004 | ALRV1-R004 | Bootstrap-promote reusable primary labels under the explicit continuity rule. |
| ALRV1-D005 | ALRV1-R005 | Import only surviving legacy entity rows; synthesize no empty legacy entity response. |
| ALRV1-D006 | ALRV1-R006 | Run and promote the exact 13-paragraph Codex correction overlay as a trusted single pass, not human validation. |
| ALRV1-D007 | ALRV1-R007 | Carry forward the April 3, 1968 speech labels under the documented provisional exception. |
| ALRV1-D008 | ALRV1-R008 | Keep Opus as an unpromoted independent second opinion. |
| ALRV1-D009 | ALRV1-R009 | Keep constituencies draft/pilot-only; materialize an empty typed projection and no production dataset. |
| ALRV1-D010 | ALRV1-R010 | Keep model execution, credentials, provider clients, and network access outside the core terminal workflow. |
| ALRV1-D011 | ALRV1-R011 | Publish immutable sealed runs and atomic content-addressed materialization/promotion artifacts. |

All resolutions were explicitly authorized by the repository owner’s approved plan dated
2026-07-24.

## Accepted provenance limitations

| Item | Resolution | Limitation that must remain visible |
|---|---|---|
| ALRV1-P001 | ALRV1-R012 | Surviving final legacy labels can be preserved; discarded intermediate responses cannot be recreated. |
| ALRV1-P002 | ALRV1-R013 | Unknown assignment/label times remain null; date-only precision remains date-only; effort is inferred only on an exact prompt-hash match and never called observed. |
| ALRV1-P003 | ALRV1-R014 | The invocation-tone prompt and cost are unknown; its surviving hash identifies the method string, not a prompt. |

These are accepted limitations, not repaired facts. The sealed run metadata and human-review view
must keep them visible.

## Execution gate

### ALRV1-G001 — exact Codex runtime model identity

Before the first correction-overlay assignment was shown, the local runner persisted the exact
identifier exposed by `nodeRepl.requestMeta.x-codex-turn-metadata`: model `gpt-5.6-sol`,
reasoning effort `xhigh`, and the active thread/session/turn identifiers.

Resolution `ALRV1-R016`: `satisfied` on 2026-07-24. The receipt is stored at
`data/annotation_ledger/review/correction-overlay-runtime-model-receipt-v1.json`.

## Overlay recovery incident

### ALRV1-I001 — frozen-v1 mixed-speech assignment refusal

The first scheduler attempt placed 13 different speeches into one assignment. The frozen v1
response schema echoes `para_idx` but not `doc_name`, and two targets shared `para_idx=10`.
Ingest refused the response before writing any response or label event.

Resolution `ALRV1-R017`: preserve the zero-event assignment as abandoned work, constrain frozen-v1
judgment assignments to one speech, and create recovery run
`correction-overlay-13-codex-20260724-r2`. The recovery run completed and was sealed; the failed
run was not sealed or materialized.

### ALRV1-I002 — complete frozen-input inventory recovery

Final audit found that the per-run inventory included every paid annotation artifact, all 15
manifests, the correction manifest, and the raw/normalized source corpus, but omitted the other
derived correction-layer files. No sealed run was changed.

Resolution `ALRV1-R018`: validate the correction outputs against the SHA-256 values already frozen
in `meta_v1.json`, then append a complete 37-file content-addressed inventory at
`data/annotation_ledger/input_inventories/986f81a714d3b041bee718ba9e5fbc49d904da4495449b6fd6d029b9ca0cf9fd/`.

## Reconciliation evidence

Read-only checks at this checkpoint re-derived every approved target:

| Evidence | Re-derived result |
|---|---:|
| primary paragraph judgments | 36,229 |
| reusable primary paragraph judgments | 35,381 |
| changed primary paragraph judgments | 13 |
| Opus paragraph judgments | 8,570 |
| reusable Opus paragraph judgments | 8,435 |
| primary entity rows | 27,214 |
| reusable primary entity rows | 26,570 |
| Opus entity rows | 5,954 |
| reusable Opus entity rows | 5,848 |
| primary speech rows / retained | 1,057 / 1,053 |
| Opus speech rows / retained | 266 / 264 |
| exact reusable invocation spans | 101 |
| manifests | 15 |

No unexplained reconciliation difference is open.

## Provenance gaps to carry forward

- Legacy assigned/labeled timestamps do not survive.
- Legacy intermediate responses do not survive.
- Invocation-tone prompt text and cost do not survive.
- Unsupported legacy reasoning effort is unknown.
- The correction-overlay runtime model receipt does not exist yet because labeling has not begun.

## Reversible and irreversible next actions

Reversible after approval:

- implement schemas and registry in new ledger paths;
- run tests against temporary data;
- build derived content-addressed materializations from sealed inputs.

Irreversible by contract:

- sealing a run ID;
- publishing an adjudication artifact;
- publishing a promotion transition.

Those operations are append-only and cannot be repaired in place; recovery creates a new artifact.

## Exact next artifact boundary

The approved first implementation slice is limited to:

- new `src/presidential_profiles/annotation_ledger.py`;
- new `src/presidential_profiles/annotation_workflow.py`;
- new `src/presidential_profiles/annotation_backfill.py`;
- a narrow generic-workflow update to
  `src/presidential_profiles/constituency_labeling.py`;
- new `tests/test_annotation_ledger.py`,
  `tests/test_annotation_workflow.py`, and
  `tests/test_annotation_backfill.py`;
- compatibility-only changes to `tests/test_constituency_labeling.py`;
- new registry/spec JSON below `data/annotation_ledger/specs/`.

Sealed backfill, projection, review, promotion, and materialized artifacts below
`data/annotation_ledger/` follow only after the focused source/test gates pass. Nothing in
`data/llm_annotations/`, the correction manifest, or `docs/` may change.
