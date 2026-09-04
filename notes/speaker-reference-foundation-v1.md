# Speaker attribution and reference-entity foundation v1

Status: **implemented and accepted; page-specific migrations remain separate.**

This layer is the deterministic foundation for later Story, Summary, Compare,
Explore, Profiles, Issues, and Data/Methods work. It does not change public site
JSON or generated `docs/`.

## Speaker-view contract

`data/speaker_views/paragraph_view_v1.parquet` is a key-complete join of the
corrected canonical paragraphs, canonical source-document metadata, and the
governed speaker attribution on `(doc_name, para_idx)`. It contains all 35,394
retained rows:

- 32,531 whole paragraphs attributed to exactly one canonical president are
  eligible for president-attributed analysis;
- 2,863 non-president, multiple-speaker, joint/shared, or scaffolding rows
  remain visible with their exclusion reason; and
- 296 eligible paragraphs are attributed to a president other than the source
  document owner.

`appearances_v1.parquet` has one row per
`(doc_name, attributed_speaker_profile_id)`. Its text concatenates only that
speaker's eligible paragraphs in original paragraph order. Each row preserves
the shared source document's date, title, Miller Center URL, year, speech type,
document owner, governed Story era, exact paragraph indices, word count, and a
cross-owner flag. The 32,531 eligible paragraphs reconcile to 1,054
appearances.

`coverage_v1.parquet` summarizes every inclusion and exclusion outcome.
`consumer_inventory_v1.json` records the current unit, replacement unit,
denominator, migration state, and follow-on plan for every inventoried
president-attributed analytical producer. Actual-speaker attribution is the
default for president-attributed results; explicitly document-level questions
may retain ownership only when they say so.

## Hybrid reference-entity contract

The local, pinned `spaCy 3.8.14` / `en_core_web_sm 3.8.0` NER pipeline runs over
all 35,394 retained paragraphs. `ner_mentions_v1.parquet` preserves 199,604 raw
spans with exact character offsets, NER label/display type, paragraph key,
speaker outcome, actual speaker, document owner, source metadata, and governed
Story era.

`entity_mentions_v1.parquet` conservatively matches local NER spans with the
promoted primary AI entity projection within the same paragraph. Matching uses
casefolding, whitespace collapse, and only the safe lexical equivalents in
`alias_map_v1.json`. Historically distinct entities such as Great
Britain/United Kingdom, Prussia/Germany, Soviet Union/Russia, and Ottoman
Empire/Turkey remain deliberately unmerged. The 186,093 normalized hybrid rows
retain primary-AI type and nullable stance separately. NER corroborates a name
mention; it does not validate stance, historical identity, or model confidence.

The partial second-model artifacts are used only by
`source_quality_v1.json`. The original pass declares 262 documents and 8,570
paragraphs; 260 documents and 8,438 exact paragraph keys survive unchanged in
the corrected canonical corpus and form the comparable QA population. Those
artifacts are explicitly prohibited from headline selection, era ranking, and
stance inference.

## Era-distinctive contract

`era_distinctive_v1.parquet` ranks primary-AI entity candidates against the
other eight canonical Story eras. Each normalized entity counts at most once
per paragraph. Every speaker-audited paragraph in the era is in the
denominator, including paragraphs with no entity. Ranking uses
Jeffreys-smoothed paragraph log odds (0.5 success and failure priors), requires
at least five paragraphs across two source documents, and publishes exactly
five rows per era.

Each of the 45 published rows includes exact support, the all-paragraph era and
comparison denominators, log odds and standard error, `AI + NER` versus
`AI only` source-agreement counts, nullable AI stance, and one keyed evidence
paragraph with exact AI/NER mention, speaker, document owner, title, URL, and
cross-owner state. “Source agreement” is the only permitted description; it is
not historical validation or model confidence. NER-only discoveries remain in
the hybrid data layer and never enter the headline ranking.

## Provenance and validation

Both derived directories carry timestamp-free `meta_v1.json` files with the
canonical corpus fingerprint, speaker-attribution run, active annotation
generation, exact input hashes, complete frozen-paid-artifact inventory, spaCy
versions, alias version, denominators, thresholds, and emitted artifact hashes.
The fast reader rejects source, key-set, metadata, evidence, or artifact drift.

Run the fast acceptance check without rerunning NER:

```bash
arch -x86_64 .venv/bin/python -m presidential_profiles.foundation_audit --check
```

Run the slow local rebuild or two-build byte gate separately:

```bash
arch -x86_64 .venv/bin/python -m presidential_profiles.foundation_audit --rebuild
arch -x86_64 .venv/bin/python -m presidential_profiles.foundation_audit --verify-reproducible
```

The acceptance report is
`data/reference_entities/acceptance_report_v1.json`. It includes population
receipts, cross-owner examples, every era candidate and source-agreement count,
compact evidence excerpts, and the fixed October 28, 1980 regression: all 151
paragraphs; Carter 44; Reagan 42; non-president 22; mixed-speaker 43; no
cross-contamination between the two appearances; shared source metadata intact.

No provider API, paid annotation regeneration, frozen annotation edit, sealed
speaker-input edit, public JSON migration, `docs/` generation, or deployment is
part of this foundation.
