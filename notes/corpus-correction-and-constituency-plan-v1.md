# Corpus correction and constituency-labeling plan

Status: active implementation plan. Reconstruction story design is paused until the corrected
canonical corpus and constituency labels are available.

## Implementation checkpoint — July 24, 2026

The correction and terminal-runner foundations are implemented:

- `presidential_profiles.corpus_corrections` validates the 23 confirmed manifest operations and
  deterministically builds the canonical corpus plus mapping, exclusion, and reannotation audits
  under `data/corpus_corrections/`.
- The canonical result is 1,053 speeches and 35,394 paragraphs. It excludes 751 old chunks from
  internal transcript repeats and 84 from four duplicate-document records.
- Thirteen retained boundary paragraphs have corrected text and remain the Track A blocker for a
  complete frozen-annotation projection. There are no new canonical paragraph keys.
- `presidential_profiles.constituency_labeling` implements the offline `init`, `next`, `ingest`,
  `status`, `audit`, `export`, and `disagreements` workflow. It has no model client or network
  path, and every saved label requires exact model and annotator provenance.
- The real canonical corpus has passed an initialization and two-paragraph batch smoke test in a
  temporary directory. No production constituency dataset or labeling pass has been initialized.

The next execution gate is therefore: label and independently check the 13 correction-overlay
paragraphs, freeze a persisted stratified constituency pilot, run and adjudicate that pilot, and
only then initialize the immutable full constituency dataset.

## Objective

Establish one canonical analytical speech/paragraph universe over the Miller Center source,
without altering the downloaded tarball or the frozen paid annotation artifacts, and then label
every retained paragraph for the constituencies the presidential voice explicitly claims to
represent, include, protect, or benefit.

The constituency workflow must make zero external API calls. A local command prints bounded
paragraph batches to the terminal; a Codex agent reads them and returns validated structured
labels through terminal input. Every saved row records the annotator and model identity.

## Non-negotiable invariants

- Preserve `data/raw/miller_center_speeches.tgz` as downloaded source evidence.
- Never regenerate or hand-edit `data/llm_annotations/`.
- Keep every analytical paragraph keyed by `(doc_name, para_idx)`; never join on row order.
- Retain an existing first-copy paragraph key only when the corrected paragraph text fingerprint
  is unchanged. If removing the duplicate changes a boundary chunk, textual correctness wins:
  re-chunk it, record the old-to-canonical mapping, and mark the canonical chunk as requiring an
  annotation overlay.
- Treat a frozen annotation row excluded by a declared correction differently from an unknown or
  silently missing key.
- Apply validation and refusal guards before replacing any published artifact.
- Make the correction and labeling layers resumable, reviewable, and independent of an external
  model API.
- Do not resume Reconstruction interpretation until the corrected topic evidence and
  constituency labels can be re-derived together.

## Track A: canonical corpus correction

### A1. Freeze the anomaly audit

The current high-confidence audit has two anomaly classes:

1. Nineteen documents contain a repeated transcript half. The normalized first and second halves
   agree at 99.6–100%. Removing the repeated source text before re-chunking excludes 751 old
   paragraph chunks. The earlier estimate of 748 came from rounding each already-chunked document
   in half; it retained fragments from duplicate copies in chunks that crossed the copy boundary.
2. Four pairs of records represent the same speech: Cleveland on August 8, 1893; Taft on
   January 26, 1911; Wilson's September 1916 nomination acceptance; and Hoover on July 24, 1929.

The audit must persist the metric and evidence used to classify each anomaly. Lower-confidence
partial repeats remain review candidates and must not be removed automatically.

The four canonical records are the Miller Center pages that identify the National Archives as
their source:

- Cleveland: `august-8-1893-special-session-message`;
- Taft: `january-26-1911-special-message-canadian-reciprocity`;
- Wilson: `september-2-1916-speech-acceptance`;
- Hoover: `july-24-1929-remarks-upon-proclaiming-treaty-renunciation-war`.

Their duplicate alternatives show no specified source. The Wilson alternative also titles the
speech September 3 while its own metadata dates it September 2. The manifest records these
reasons rather than relying on tar order, paragraph count, or filename order.

### A2. Add a versioned correction manifest

Each manifest entry has one of two operations.

`repeat_within_document` records:

- `doc_name`;
- retained and excluded paragraph-index bounds;
- normalized-half similarity;
- retained text/paragraph fingerprint;
- excluded text/paragraph fingerprint;
- review status and rationale.

`duplicate_document` records:

- canonical and excluded `doc_name`;
- normalized-transcript similarity/hash;
- title, date, paragraph support, and introduction/link evidence for both records;
- the reason for the canonical selection;
- review status.

The loader refuses an unknown operation, overlapping corrections, a changed retained fingerprint,
an out-of-range paragraph index, or a duplicate-document pair that no longer matches.

### A3. Project raw artifacts onto the canonical universe

The correction layer removes repeated source text before paragraph chunking and produces, without
overwriting raw provenance:

- canonical speech rows, with internally repeated transcript text and word counts corrected;
- canonical paragraph rows, retaining original `(doc_name, para_idx)` keys only where the
  corrected text is fingerprint-identical;
- a canonical key registry;
- an old-to-canonical paragraph mapping whose statuses distinguish unchanged/reusable,
  excluded-duplicate, changed/requires-reannotation, and new/requires-reannotation rows;
- an exclusions audit table containing every removed key and reason;
- a metadata sidecar carrying manifest, source, and output fingerprints.

Frozen paragraph annotations and entities are projected only onto text-fingerprint-identical
canonical paragraphs. Extras are permitted only when every extra key is declared in the
exclusions audit. A canonical paragraph with no text-matched frozen annotation remains an explicit
reannotation blocker; it may not be silently dropped, treated as topic-free, or joined merely
because its numeric key happens to match an old chunk.

### A4. Select canonical duplicate records deliberately

Do not keep the first or longest record by default. Selection must consider:

- which Miller Center page is current and stable;
- whether the introduction supplies useful provenance;
- paragraph segmentation quality;
- existing source receipts and links;
- frozen annotation coverage and integrity.

If local evidence cannot settle a pair, generation stops with an unresolved-canonical-choice
report rather than guessing.

### A5. Rebuild in dependency order

Before rebuilding, any canonical boundary chunk marked `requires_reannotation` must receive a
separate correction overlay for every frozen paragraph-level field consumed downstream. The
overlay reuses the frozen field definitions and prompt/schema versions, records its own
annotator/model provenance, and lives outside `data/llm_annotations/`. It does not rewrite the
paid artifact. Changed chunks should receive an independent check or adjudication because the
old same-key label is not evidence about the new text.

After the correction contract, overlay, and tests are locked:

1. canonical speeches and paragraphs;
2. correction annotation overlay and complete canonical annotation projection;
3. deterministic paragraph and speech features;
4. deterministic topic, issue, register, combat, agreement, network, era, band, and story layers;
5. site outputs and validation reports.

Frozen paid judgments are projected, never rerun. Every derived artifact must record the canonical
corpus fingerprint it consumed. Any layer that cannot safely consume the corrected universe is a
blocker, not an excuse to mix raw and canonical denominators.

### A6. Acceptance gates

- Exactly the reviewed internal repeats and duplicate records are excluded.
- The source-aware projection excludes 751 old chunks from internal repeats and 84 chunks from
  four duplicate documents, yielding 35,394 canonical paragraphs across 1,053 speeches.
- Thirteen retained boundary keys whose corrected text differs are marked
  `changed_requires_reannotation`; they cannot inherit same-key frozen labels.
- Peoria retains 138 paragraph chunks and the Clay eulogy retains 42.
- Correcting Peoria and Clay changes the 1850–54 slavery/sectionalism numerator and denominator
  from `341/904` to `183/724`, or 25.3%.
- No correction changes an undeclared document.
- Reapplying the manifest is idempotent.
- Unknown annotation orphans, missing canonical annotation keys, and same-key/different-text
  annotation reuse fail loudly.
- Targeted tests, full relevant tests, deterministic rebuilds, and site validation pass.

## Track B: offline constituency labeling

### B1. Historical quantity

The label answers:

> Whose authority, welfare, rights, or interests does the presidential voice explicitly invoke
> as a constituency for presidential action?

An audience, favorable mention, named subject, or generic first-person plural is not automatically
a constituency claim.

### B2. Annotation unit and context

- Unit: one retained `(doc_name, para_idx)` paragraph.
- Context: paragraph text plus bounded preceding/following paragraph text from the same speech.
- Metadata: year, president, title, and speech type are displayed separately from the text.
- The evidence span must be an exact substring of the target paragraph. Neighboring context may
  resolve a referent but may not serve as the saved evidence span.
- Every retained paragraph receives one result: `claim`, `none`, or `unclear`.

### B3. Constituency claim schema

A `claim` contains one or more entries with:

- `group_text`: literal group wording from the paragraph;
- `normalized_group`: stable human-readable group name;
- `group_type`: a frozen closed enum such as national public, regional/sectional,
  racial/ethnic/legal-status, economic/occupational, military/veteran, party/movement,
  religious, age/gender/family, foreign people, or other;
- `relation`: source of authority, represented constituency, protected group, intended
  beneficiary, or included national member;
- `evidence_span`: exact paragraph substring;
- `certainty`: explicit or context-resolved;
- `notes`: optional short disambiguation, not historical interpretation.

`none` asserts that no qualifying claim appears. `unclear` requires a short reason and enters an
adjudication queue.

The schema and instructions are piloted on an era- and genre-stratified sample, then frozen before
the full run. Later changes create a new annotation version; they never rewrite old meanings in
place.

### B4. Terminal protocol

The intended module entry point is:

```bash
arch -x86_64 .venv/bin/python -m presidential_profiles.constituency_labeling
```

Required commands:

- `init`: fingerprint the canonical corpus and create an empty run manifest/queue;
- `next`: print the next stable batch as readable text plus machine-readable JSON;
- `ingest`: read completed JSON from stdin, validate it fully, and save atomically;
- `status`: report coverage by annotator, era, genre, and shard;
- `audit`: check keys, evidence spans, enums, fingerprints, duplicates, and missing rows;
- `export`: build the consolidated long-form label table and provenance sidecar only after gates
  pass;
- `disagreements`: emit paragraphs whose independent labels differ.

No command imports an API client, reads credentials, or opens a network connection.

### B5. Provenance on every label

Each label carries:

- `(doc_name, para_idx)`;
- `annotation_version`;
- `annotator_id`;
- exact `model_id` supplied by the runner;
- optional agent/task identifier;
- prompt and schema versions plus SHA-256 hashes;
- canonical corpus fingerprint;
- stable batch and shard identifiers.

Ingestion refuses a missing model ID, corpus drift, prompt/schema drift, duplicate key within an
annotator pass, evidence mismatch, incomplete batch, or output for a key that was not assigned.

### B6. Parallel and independent operation

Two uses must remain distinct:

1. **Sharded primary pass:** agents label disjoint deterministic shards under one annotation
   version. Model and annotator provenance remain row-level even when agents differ.
2. **Independent agreement pass:** two annotator IDs receive the same paragraphs. Their rows are
   preserved separately; disagreement creates an adjudication queue rather than being averaged
   away.

Shard membership is a stable hash of `(doc_name, para_idx)`, never current row position. Atomic
claiming or explicit shards prevent two agents from accidentally completing the same primary row.

### B7. Full-run sequence

1. Finish Track A and freeze the canonical fingerprint.
2. Pilot the label schema on a persisted era/genre-stratified sample.
3. Manually adjudicate pilot disagreements and freeze annotation version 1.
4. Run a full primary pass over every canonical paragraph. Candidate heuristics may prioritize
   likely claims but may not auto-assign `none`.
5. Run an independent agreement pass on a persisted stratified sample; expand to a full second
   pass only if desired.
6. Adjudicate all `unclear` rows and chapter-defining disagreements.
7. Export a long-form table and a paragraph-level summary.
8. Measure agreement overall and by era/genre before publishing any pills or matrix.

### B8. Publication gates

- Every canonical paragraph has one valid primary result.
- Every saved evidence span matches exactly.
- Coverage and agreement are reported by era and genre.
- No excluded duplicate paragraph was labeled.
- Pills are based on de-duplicated paragraph unions and all-paragraph denominators.
- A pill exposes paragraph/speech support, whether it is episodic, and at least one keyed receipt.
- “Speaks for” remains distinct from audience, favorable mention, subject matter, and adversary.

## Integration and pause condition

The corpus correction track lands before the constituency run is initialized. The labeling
manifest fingerprints the corrected canonical universe, so a later corpus change makes the run
refuse rather than silently shifting its population.

Reconstruction design resumes only after:

1. the corrected 1850–77 topic/adversary evidence has been rebuilt;
2. constituency-label coverage gates pass;
3. agreement and disagreement are visible;
4. Johnson, Grant, and Hayes receipts have been checked against the new labels.
