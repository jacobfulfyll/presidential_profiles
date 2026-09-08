# Data Trust validation protocols — v1

Status: **planned and gated; no human labels or paid model outputs exist**
Protocol version: `data-trust-validation-protocols-v1`
Selection seed namespace: `presidential-profiles/data-trust/v1/20260908`

This note predeclares two future studies without authorizing either one. It is
not an execution script, a result, or permission to modify the frozen annotation
artifacts. Any selected keys, codebook, coder data, prompts, model outputs, or
estimates must be stored in a new versioned namespace and must never overwrite
`data/llm_annotations/`.

## Shared population and deterministic selection

The paragraph universe is the one-to-one key intersection of:

- `data/speaker_views/paragraph_view_v1.parquet`, restricted to
  `analysis_eligible == True`; and
- `data/llm_annotations/paragraph_annotations.parquet`, joined only on
  `(doc_name, para_idx)`.

The current governed universe has nine `story_era_key` strata. Before any
labeling, record source hashes, assert unique keys and exact key-set coverage,
and create a 360-paragraph selection with exactly 40 rows per Story era. A row
can occupy only one arm, assigned in this priority order:

1. **High disagreement — 10 per era.** Among paragraphs available in both
   existing model passes, rank descending by the v1 field-specific disagreement
   formula after governed topic and entity normalization: one point per
   differing binary flag, plus `1 − topic Jaccard`, plus one for a differing
   proposal/values class, plus `1 − entity-name Jaccard`. Break ties by the
   SHA-256 rank defined below.
2. **Rare positive — 10 per era.** Excluding selected disagreement rows, define
   each positive binary flag and each normalized topic with at least 20 global
   eligible paragraphs as a label. Score a paragraph by the inverse global
   prevalence of its rarest positive label; rank descending, then by SHA-256.
   Paragraphs with no positive eligible label rank last.
3. **Genre-balanced base — 20 per era.** Excluding the first two arms, take five
   paragraphs from each of four speech-type families: annual/governance
   (`state_of_the_union_or_annual_message`, `special_message_to_congress`,
   `veto_or_signing_statement`); ceremonial (`inaugural_address`,
   `eulogy_or_commemoration`, `proclamation`); interactive/political
   (`campaign_or_debate`, `press_conference_or_interview`); and public/other
   (`public_remarks_or_address`, `other`). Within each era and family, first
   independently hash-rank unique eligible documents with arm
   `genre-balanced-base-document` and `para_idx = -1`, then take up to five
   documents. Within each selected document, independently hash-rank eligible
   paragraphs with arm `genre-balanced-base-paragraph` and take the first. For a
   first-pass row, record the design inclusion probability as
   `min(5, N_documents) / N_documents × 1 / N_paragraphs_in_document`, its
   inverse sampling weight, and `weight_status = design_weight_available`.
   After all four family passes, fill any era shortfall from the remaining
   unselected rows by arm `genre-balanced-base-fallback`. Record the fallback's
   conditional stage fraction, but leave its marginal population inclusion
   probability and sampling weight null and mark it descriptive only.

The high-disagreement and rare-positive arms are purposive top-k enrichment,
not probability samples. Their population inclusion probabilities and sampling
weights are null, their `weight_status` is
`not_applicable_purposive_enrichment`, and their estimates are descriptive
only. The genre-base fallback is sequential and has
`weight_status = not_available_sequential_fallback`; its recorded conditional
stage fraction is a reproducibility receipt, not a population weight.

For every deterministic tie-break, compute SHA-256 over the UTF-8 string
`<seed namespace>|<arm>|<story_era_key>|<doc_name>|<para_idx>` and sort by the
hexadecimal digest ascending. Persist all candidate counts, inclusion arms,
selection probabilities, source hashes, selected keys, and the resulting
selection hash before exposing labels to a coder. Persist null rather than a
constructed probability or weight wherever the design does not identify a
marginal inclusion probability.

Speech-level ranking uses the explicit sentinel `para_idx = -1`, so its hash
input is `<seed namespace>|factual-speech|<story_era_key>|<doc_name>|-1`.

Create a separate 90-speech factual-label sample with exactly 10 source
documents per Story era. Its candidate frame is all 1,057 source speeches,
joined one-to-one to all 1,057 frozen factual speech annotations; paragraph-level
speaker eligibility is not an exclusion rule for this study. Assign Story eras
with the governed speech-level era function, including outgoing-president
overrides. Apply the same four genre families, targeting
3 annual/governance, 2 ceremonial, 2 interactive/political, and 3 public/other
documents per era. Use every available document in a short cell and fill from
the era remainder by SHA-256 rank. Each first-pass family draw is hash-random
and records its family-specific document inclusion probability,
`min(target, N_documents) / N_documents`, its inverse weight, and
`weight_status = design_weight_available`. A sequential fallback records its
conditional stage fraction, but its marginal population inclusion probability
and sampling weight are null and its result is descriptive only. The factual
sample is keyed by `doc_name` and cannot substitute for the 360-paragraph
sample.

## Protocol A — blinded human validity study

### Evidence shown to coders

- Paragraph task: paragraph text plus decade, matching the production judgment
  evidence policy. President, party, title, URL, existing labels, selection arm,
  and the other coder's answers remain hidden.
- Speech-factual task: title, year, and first five paragraphs, matching the
  factual evidence policy. Existing model labels and the other coder's answers
  remain hidden.
- The frozen taxonomy, field definitions, examples, and closed enum values are
  the codebook. Coders cannot introduce new topic strings or categories.

Two trained coders independently label every selected item. A third reviewer
adjudicates every field-level disagreement and a deterministic 10% audit of
agreements, selected by the same hash rule with arm `agreement-audit`. Coders
complete calibration on a separate practice set that is excluded from all
reported estimates. The codebook and calibration decisions freeze before the
first study item is opened.

### Prespecified estimands

Report descriptive results for every arm, but never present the enriched mix as
corpus prevalence. Population-weighted inference is prohibited for the
high-disagreement arm, rare-positive arm, genre-base fallback, and factual
fallback. It is permitted only for first-pass rows with
`weight_status = design_weight_available`, only for an estimand whose target
population matches the declared first-pass probability frame, and only with the
inverse of the recorded design inclusion probability. Enrichment and fallback
rows must be excluded from every population-weighted numerator and denominator;
their mutually exclusive priority does not make them probability samples.

- Human–human reliability: Cohen's κ for each binary flag and single-choice
  factual field, exact agreement for proposal/values, mean paragraph Jaccard for
  topic sets, and strict normalized-name plus stance agreement for entities.
- Model–human validity: precision, recall, and F1 for every binary flag; micro
  and macro precision/recall/F1 plus mean Jaccard for topics; exact agreement
  and macro F1 for proposal/values and factual categories; exact normalized-name
  detection plus type/stance agreement for entities. Report each production
  model against both independent coders and against adjudicated labels.
- Entity sensitivity: report strict normalized names and a separately declared
  partial-span match. Partial matching may not replace the strict primary
  result.
- Slices: overall, Story era, speech-type family, selection arm, and
  actual-speaker versus cross-owner status.

Use 2,000 deterministic whole-`doc_name` bootstrap resamples with seed namespace
`...|human-bootstrap` for 95% percentile intervals. Suppress a slice when it has
fewer than 20 evaluated units, fewer than five positive adjudicated cases for a
positive-class metric, or fewer than five source documents. Preserve the row
with status `suppressed_low_support`; do not drop it. Apply Benjamini–Hochberg
false-discovery-rate control at 0.05 only to families of formal slice-comparison
tests; do not apply it to descriptive estimates.

### Outputs and gates

The future implementation must create a versioned selection manifest, coder
codebook hash, blinded assignment files, raw immutable coder returns,
adjudication receipts, metric table, bootstrap table, qualitative error
taxonomy, and publication manifest. Human labels cannot replace production
labels without a separate migration plan and approval.

Execution requires named coders, conflict/privacy review, a frozen codebook,
successful key and blinding tests, and explicit written approval. Until all
gates pass, the public site may describe this only as a planned study.

The governed builder writes selection masters, coder task files, and the
invariance request plan beneath `data/validation_protocols/v1/restricted/`.
That directory is intentionally ignored by Git and must remain local to the
approved study team; regenerate the complete local bundle with
`arch -x86_64 .venv/bin/python -m presidential_profiles.validation_protocols --rebuild`
and validate it with `--check`. The public projection contains only the five
files declared by `PUBLIC_FILES`. This is an operational blinding control, not
cryptographic secrecy: because the selection is deterministic, someone with
the source inputs and builder can reconstruct it. Coders therefore must not
access the reconstruction inputs or repository analysis artifacts until their
independent labels are locked.

## Protocol B — decade-hidden measurement-invariance study

Use the exact same 360 paragraph keys. This is a paired sensitivity experiment,
not a refresh of production labels.

- **Current-context arm:** the frozen production judgment prompt with paragraph
  text and decade.
- **Decade-hidden arm:** byte-identical task definitions, taxonomy, examples,
  output schema, and paragraph text, with only the decade field and instructions
  referring to decade removed.

Before execution, persist both complete prompt byte strings, their SHA-256
hashes, the 360-key selection hash, request partition, random presentation order,
decoding parameters, retry policy, and exact runtime model identifier. A model
is intentionally **not assigned by this protocol**; choosing one requires a new
dated approval receipt. Both arms must use the same approved model and decoding
configuration, and no request may include president, party, title, URL, existing
labels, human labels, or selection arm.

### Prespecified comparisons

- Paired agreement between arms using the production field-specific measures.
- Paired marginal prevalence difference in percentage points for binary flags
  and topics, with whole-speech bootstrap intervals.
- Exact-switch matrices for proposal/values and entity stance.
- Overall and Story-era estimates, with the same low-support suppression rules
  as Protocol A.
- A sensitivity flag when an overall absolute prevalence shift exceeds 2
  percentage points, an era-specific shift exceeds 5 points, or paired κ falls
  below 0.80. These are diagnostic thresholds, not proof that either arm is
  correct. Human-study comparisons remain necessary.

Use 2,000 deterministic whole-`doc_name` bootstrap resamples with seed namespace
`...|invariance-bootstrap`. Preserve non-estimable and degenerate intervals with
explicit status values. Do not promote a decade-hidden result into any public
chart without a separate review of its historical meaning.

### Paid-run gate and cost receipt

Before any API submission, a dry run must validate request counts, exact key
coverage, one-to-one outputs, schema parsing, prompt hashes, and recovery from a
truncated response. The approval receipt must name the exact runtime model,
current input/output/cache token prices, projected token counts by arm, expected
cost, a hard maximum cost, and the person approving it. Cost is computed from
those token classes and rates; no placeholder model or stale price may be used.

The run must stop before submission when approval is absent, the projected cost
exceeds the cap, source or prompt hashes drift, the model identifier differs,
or any selected key is missing or duplicated. Ingestion writes a new immutable
namespace and records actual usage-derived cost. It must never overwrite the
current paid artifacts.

## Public language until execution

Permitted: “A blinded human-validity study and a decade-hidden sensitivity
experiment are preregistered but have not run.”

Prohibited: accuracy, validation, calibration, invariance, robustness, or
improvement claims based on either protocol until its evidence, intervals,
review gates, and publication manifest exist.
