# President speaker-attribution remediation plan

Status: **groomed backlog plan; ready for a dedicated implementation and chat-labeling task.**
This planning document does not itself start labeling, seal a run, rebuild the site, or deploy.

Task ID: `audit-president-speaker-populations-and-version-conflict-chart-inputs`

## What is wrong

The Miller Center corpus stores a document owner in `speeches.president`. That field identifies
the presidential profile that hosts a transcript; it does not prove that the named president
spoke every paragraph in the transcript. Debates include opponents and moderators. Press
conferences include reporters. Briefings and public remarks can include other officials, audience
members, or appended questions and answers. Joint statements can have shared authorship.

The current president-level Conflict table assigns all of those paragraphs, and all adversarial
entities found in them, to the document owner. The Summary page labels that interpretation as
provisional. This task replaces that provisional interpretation with a separate, evidenced
speaker-attribution layer and a versioned Conflict producer.

The preliminary audit behind the original backlog entry found 4,562 suspect paragraphs. A fresh
read-only grooming audit showed that figure is already stale: 13 debates, the press/interview
population, and seven obvious mixed-speaker public-remarks records alone expose at least 4,827
legacy paragraphs, before joint authorship and other edge cases. These are scope clues, not a
frozen selection or acceptance count. The implementation must derive the exact population again
and must not omit a document merely to preserve either estimate.

## Which corpus is authoritative

The source inventory still accounts for all 1,057 downloaded/normalized document records so the
four excluded duplicates remain visible as provenance. The labeling projection and corrected
producer use the existing corrected canonical corpus:

- 1,053 speeches;
- 35,394 paragraphs; and
- the canonical corpus fingerprint already recorded by the correction and annotation ledgers.

The older 1,057-speech / 36,229-paragraph files remain the source census and the basis of the
current published comparison. Every source document receives a document audit row, including the
four records excluded by the correction manifest. Those excluded records do not enter production
paragraph labeling or denominators. The implementation must retain the existing
source-to-canonical mapping so every exclusion is explainable and must never combine a canonical
numerator with a legacy denominator.

The frozen paid annotations under `data/llm_annotations/` remain read-only. The corrected
Conflict producer reads the canonical projection of those judgments; it does not rewrite or
regenerate them.

## What the chats will label

There are two separate questions. Keeping them separate prevents document ownership from being
silently reused as speaker identity.

### 1. One classification for every source document

Every one of the 1,057 source speeches receives exactly one document classification, plus its
canonical-retention status. The 1,053 retained speeches form the production selection. Each row
uses exactly one of these classes:

- `single_president`: the transcript contains one presidential speaker;
- `multi_speaker`: the transcript contains turns from more than one speaker;
- `joint_authored`: the words are presented as jointly authored and cannot safely be credited to
  one person without further evidence; or
- `uncertain`: the available transcript does not support a confident classification.

The assignment shows the document owner, title, year, speech type, introduction, and the complete
paragraph sequence. One assignment must never mix documents. Very long documents may use locked
same-document parts, but the final classification must cover the complete document rather than
only its opening paragraphs.

The document record also states whether paragraph review is required and cites paragraph-indexed
evidence for any other speaker, shared authorship, transcript scaffolding, or uncertainty. A clean
`single_president` document can supply its paragraph speaker by documented inheritance. Every
other document, and every document containing standalone scaffolding or stage directions, goes to
paragraph review.

### 2. One speaker result for every paragraph requiring review

Paragraph review uses these outcomes:

- `canonical_president`: one of the corpus's 45 named presidents spoke the entire paragraph;
- `non_president`: a moderator, reporter, audience member, official, or candidate who is not one
  of those 45 presidents spoke the entire paragraph;
- `multiple_speakers`: the paragraph combines more than one speaking turn;
- `joint_or_shared`: the paragraph is jointly authored and cannot be assigned exclusively;
- `scaffolding`: the row is only a stage direction, applause marker, speaker label, or other
  non-speech transcript furniture; or
- `uncertain`: the speaker cannot be resolved confidently from the target and its context.

For `canonical_president`, the response must include the exact controlled president name. Local
code maps that name to the stable president/profile identifier used everywhere else on the site.
This can differ from the document owner and applies to former or future presidents even when the
turn occurred before or after their presidency. A non-president opponent is not turned into a
president merely because the transcript is hosted on a presidential profile.

For example, in a Carter-hosted Carter–Ford debate:

- a Carter-only paragraph is credited to Jimmy Carter;
- a Ford-only paragraph is credited to Gerald Ford;
- a moderator-only paragraph is recorded as `non_president` and excluded from president metrics;
  and
- a paragraph combining Carter, Ford, or moderator turns is `multiple_speakers` and excluded
  whole.

The same rule assigns Nixon-only turns in Kennedy-hosted 1960 debates to Richard M. Nixon and
Reagan-only turns in the Carter-hosted 1980 debate to Ronald Reagan. This is a general 45-profile
roster lookup, not a debate-specific correction list.

Each result carries an exact paragraph or neighboring-context evidence reference and a short
controlled reason. Local code, not the chat, supplies and validates `doc_name`, `para_idx`, source
hashes, assignment IDs, the 45-name vocabulary, and timestamps.

The rubric follows four fail-safe rules:

1. A quotation inside one person's speaking turn does not create a new speaker.
2. A paragraph containing more than one actual turn is excluded whole; it is not split because
   splitting would invalidate the existing paragraph-level topic, flag, and entity judgments.
3. Continuation paragraphs may use a preceding speaker cue from the same document as evidence.
4. When the evidence is insufficient, use `uncertain`; never infer a speaker from political
   position, writing style, or who would benefit from the attribution.

## How labeling runs through chats like this one

The repository already has a provider-neutral offline annotation workflow. This task extends that
workflow with additive speaker specifications, full-document/same-document context, exact speaker
evidence validation, and per-chat execution receipts.

The model boundary is explicit:

- the active, user-visible Codex chat reads one locked assignment;
- the chat supplies only the requested JSON classification values;
- local commands prepare the assignment, validate the response, and append accepted evidence;
- rejected JSON is repaired against the same open assignment; and
- the run is resumable across chat turns without reissuing completed work.

This task must not call a provider API, use an API key, run `pp-annotate`, submit a batch job,
invoke a provider SDK, run the existing headless annotation campaign worker, or launch hidden
`codex exec` labeling sessions. The active Codex task is the primary annotator. A later review
queue is also completed in a user-visible Codex chat.

Before a chat supplies its first label, persist the exact model and reasoning-effort identity
exposed by that runtime and bind it to the assignments completed in that session. Do not copy a
receipt from an earlier task and do not guess a model identifier. If the runtime cannot expose a
sufficiently exact identity, stop before labeling and report the missing provenance.

Existing `registry-v1.json` and `registry-v2.json` bytes are pinned by prior runs. Add
`registry-v3.json`, carrying the prior registrations unchanged and adding the speaker
specifications; teach the validator to accept that additive version without editing an old
registry in place. The large existing `data/annotation_ledger/` worktree is user-owned. New run
IDs and artifacts must be additive, and this task must not clean, rewrite, or globally
rematerialize that ledger.

## Calibration and review

Before the production selection is opened, run a small calibration set that includes:

- an ordinary single-president address;
- a presidential debate;
- a press conference or interview;
- a multi-official briefing;
- public remarks with appended questions and answers;
- a joint statement;
- transcript scaffolding or stage directions;
- a press-labeled Hoover or April 1968 LBJ record that is actually single-voice; and
- the 2025 joint-session paragraph that combines an audience interjection with the president's
  response.

Inspect the calibration results, repair the rubric or context packet if needed, rerun the same
fixtures, and freeze the prompt, schema, selection rules, and hashes before production labeling.
Calibration labels do not silently become production labels.

After the primary pass, build a locked review queue containing:

- every `uncertain` document and paragraph;
- every paragraph credited to a canonical president other than its document owner;
- every inclusion-changing primary/reviewer disagreement; and
- a deterministic stratified sample across the other outcomes, formats, eras, and presidents.

A fresh user-visible Codex chat reviews that queue without seeing the primary answer first.
Disagreements are adjudicated against the transcript evidence. Any unresolved disagreement that
would affect president membership fails closed and remains excluded. Chat labels are model
evidence, not human validation, and the published provenance must say so.

## Which paragraphs count

The speaker layer produces these declared populations from one set of labels:

1. `canonical_document_owner`: the corrected-corpus version of the current interpretation; kept
   only as a comparison.
2. `speaker_audited_all`: the primary population. It includes only whole paragraphs attributed
   to exactly one of the 45 canonical presidents and credits each paragraph to that actual
   speaker.
3. `debate_excluded`: a simple sensitivity that removes debate documents, making the effect of
   that common workaround visible without pretending it fixes press questions or briefings.
4. `single_president_documents`: a stricter sensitivity using only documents classified cleanly
   as one-president transcripts.
5. `annual_message_strict`: the annual-message comparison, additionally subject to the same
   speaker and scaffolding rules.

`non_president`, `multiple_speakers`, `joint_or_shared`, `scaffolding`, `uncertain`, and unresolved
review disagreements are excluded from `speaker_audited_all`. They remain present as coverage
receipts; exclusion must never erase the evidence that the corpus contained them.

The derived layer contains:

- one key-complete document table;
- one key-complete paragraph table;
- explicit document owner, attributed speaker name, and stable speaker-profile identifier
  columns;
- classification source, evidence, review, specification, run, and corpus fingerprints;
- a boolean for every declared population; and
- a metadata sidecar defining every count and exclusion.

Once the sealed chat inputs are fixed, rebuilding these tables must be deterministic and
byte-reproducible.

## Rebuilding president-level Conflict

Create `data/combat/by_president_treatments_v2.parquet` for the complete long-form treatment
contract and `data/combat/by_president_speaker_audited_v2.parquet` for the selected 45-row Summary
payload. Both declare schema `president-conflict-v2`. Do not silently change the meaning of the
current `by_president.parquet`; it remains available for historical comparison after v2 is
accepted.

For each declared population, the v2 producer:

- joins paragraphs, content flags, entities, and speaker attribution on
  `(doc_name, para_idx)` with one-to-one/key-set checks where applicable;
- credits a paragraph and its entities to the attributed canonical president, not automatically
  to the document owner;
- publishes integral flag numerators as well as paragraph denominators and rates;
- publishes included and excluded documents and paragraphs by reason;
- carries schema, treatment, corpus, speaker-specification, run, and review identities;
- retains all 45 presidents in fixed presidential chronology; and
- uses explicit missing values and support states when a treatment has no defensible estimate.

No Carter-, debate-, president-, or chart-specific correction list is allowed. Population masks
belong in the governed producer. A synthetic fixture that changes a paragraph's attributed
speaker must automatically change the affected presidents' counts, rates, category composition,
support warnings, and `N/A` states.

## What changes on Summary

After v2 passes its gates, the four president Conflict graphs use `speaker_audited_all` by
default. The page names the selected treatment and its support, retains the document-owned and
annual-message comparisons in the governed evidence, and no longer says that the speaker audit is
pending.

The visual contract remains stable:

- all 45 presidents stay in the same order;
- target composition stays denominationally separate from paragraph rates;
- thin support remains visibly different from missing support;
- unsupported values render `N/A`, never zero; and
- a producer fixture changes the rendered values without an HTML, CSS, or JavaScript exception.

The implementation may add a compact treatment control only if all four graphs switch together
from the same producer contract. It must not add independent chart-side filters that can create
four different populations.

## Safety rules

- Do not edit or regenerate anything under `data/llm_annotations/`.
- Do not modify the corpus-correction manifest or old sealed/registry artifacts.
- Inventory protected input bytes before implementation and require exact equality afterward.
- Validate response shape, keys, evidence references, and source hashes before the first write.
- Never join paragraph tables by row order.
- Do not write directly to `docs/`; change generators and rebuild only after source and data gates
  pass.
- Do not deploy, stage, commit, push, or change branches unless separately requested.
- If a chat-labeling selection materially exceeds the preliminary scope, report the new exact
  document, paragraph, assignment, and context estimates before opening the first production
  assignment. Never trim the selection to fit the estimate.

## Tests and acceptance checks

The task is complete only when all of the following are true:

1. The source document inventory has exactly 1,057 unique `doc_name` rows with an exact
   source-to-canonical disposition; exactly 1,053 are production-retained. The production
   paragraph table has exactly 35,394 unique `(doc_name, para_idx)` rows matching the governed
   canonical fingerprint.
2. Every document has one valid class, review status, evidence record, and immutable source hash.
3. Every document requiring paragraph review has complete paragraph coverage. No assignment
   crosses a document boundary, and no local response key can select the wrong document.
4. `canonical_president` requires exactly one controlled president name. Every other outcome
   forbids a credited president. Evidence references resolve to the locked transcript.
5. Mixed, shared, scaffolding, uncertain, and unresolved rows contribute zero paragraphs and zero
   entities to the primary treatment.
6. Treatment booleans and denominators are derived locally and mutually coherent; the chat never
   decides whether a row is published.
7. The v2 artifact has all 45 president rows per declared treatment, exact integral counts,
   bounded rates, complete provenance, and correct missing/support states.
8. Carter/Ford, Kennedy/Nixon, and Carter/Reagan fixtures prove that reassigning a debate
   paragraph from its document owner to another canonical president moves its flag and entity
   contributions to that president's profile automatically. Assigning it to a non-president or
   mixed outcome removes it from every president's denominator.
9. Frozen paid artifacts, corpus-correction inputs, old registries, old sealed runs, and the
   annotation-ledger active pointer are byte-identical before and after the task.
10. Static checks prove the chat workflow imports no model provider client and makes no model API
    or socket call.
11. The Summary consumer declares `president-conflict-v2`, shows the audited treatment and
    support, preserves all 45 presidents, and contains no pending-audit or president-specific
    correction path.
12. Targeted tests, the full relevant pytest suite, the canonical site build, static JavaScript
    parsing, and desktop/mobile browser checks pass using the repository's documented commands.

Implementation should add focused tests before production labeling. At minimum, test document
and paragraph key completeness, registry immutability, full-document context, per-chat receipts,
evidence validation, same-document batching, response-write atomicity/recovery, every label enum,
treatment derivation, content/entity reassignment, v2 contract validation, fixture-driven
rendering, and protected-file immutability.

## Execution order

1. Re-read `HANDOFF.md`, this plan, the annotation-ledger contract, and the canonical-corpus
   correction contract; inspect the dirty worktree and inventory protected bytes.
2. Add the speaker-attribution module, additive specifications/registry, context renderer,
   validators, session receipts, and focused refusal tests.
3. Build the deterministic 1,057-document source census, its 1,053-document canonical projection,
   and the calibration fixtures. Run calibration in the active Codex chat and freeze the accepted
   rubric/schema only after review.
4. Build and print the exact production plan: document count, paragraph-review count, assignment
   partition, context estimate, fingerprints, and expected outputs.
5. Complete document and paragraph assignments through user-visible Codex chat turns. Audit and
   seal only the new speaker runs; do not change the annotation-ledger active pointer.
6. Complete the locked review/adjudication queue in a fresh user-visible Codex chat and preserve
   unresolved rows as exclusions.
7. Build and byte-rebuild the document/paragraph attribution layer and all declared populations.
8. Build `president-conflict-v2`, compare it with the legacy document-owner result, and inspect the
   largest attribution-driven changes before publication.
9. Update the Summary generator and tests, rebuild `docs/`, validate JavaScript and links, and run
   desktop and 390-pixel browser acceptance checks.
10. Recheck protected bytes, run the full suite and Reggie Doctor, then update current handoff and
    task status without deploying.

Use `arch -x86_64 .venv/bin/python` for Python, pytest, and the site builder as required by the
repository contract.

## Out of scope

- Rejudging topics, combativeness flags, proposal/value labels, or entities.
- Splitting a mixed paragraph into new analytical paragraphs.
- Treating a model label as human validation.
- Migrating every other president-level analysis to the speaker layer; record those consumers as
  follow-up work after this first governed producer proves the contract.
- Redesigning the four Conflict graphs beyond the treatment/support changes required by v2.
- Promoting the provisional annotation-refresh composite or changing the global annotation-ledger
  materialized pointer.
- Deploying the site or performing git publication actions.

The existing `press-conference-paras-include-interviewer-words` backlog item is subsumed by this
task and should close when the v2 producer and its press-conference acceptance fixtures pass.

## Copy/paste prompt for the execution task

Open a fresh Codex task in this repository and send:

> Run `audit-president-speaker-populations-and-version-conflict-chart-inputs` according to
> `notes/president-speaker-attribution-plan-v1.md`. Perform every model judgment in user-visible
> Codex chats through the local offline assignment/ingest workflow. Do not call a model API, use
> an API key, run `pp-annotate`, invoke the headless Codex worker, or launch hidden labeling
> sessions. This message authorizes implementation, calibration, the primary and review chat
> labeling passes, sealing only the new speaker runs, derived-data generation, tests, and a local
> site rebuild. It does not authorize edits to frozen paid artifacts, changes to the global
> annotation-ledger pointer, deployment, staging, committing, pushing, or branch changes. Preserve
> all current worktree changes and follow the plan's stop conditions.
