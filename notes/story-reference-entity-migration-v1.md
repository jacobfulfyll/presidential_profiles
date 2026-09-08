# Story reference-entity migration v1

Status: implemented locally; validated generated output; not deployed.

Task ID: `migrate-story-reference-entities-v1`

## Subsequent UI migration

The governed 45-row distinctive-reference projection described below remains
accepted and downloadable, but it is no longer the visible Era Profile card.
The active `era-profile-v6` surface is the four-lane reference landscape in
`notes/story-reference-landscape-v1.md`. The v5 descriptions below are retained
as the implementation receipt for the foundation migration that preceded it.

## Owner decision

The Story page now uses `speaker-reference-foundation-v1` wherever it names a
presidential speaker or publishes a referenced entity. This supersedes, but
does not complete, `publish-era-constituency-categories`. The failed and
unpromoted constituency work remains historical evidence and is not a Story
input.

Aggregate Era Defined, Topic Life, footprint, topic, vocabulary, and style
measures remain on the source-document corpus. President rows, presidential
agendas, communication appearances, adversary edges, and Era Echoes speakers
use analysis-eligible actual-speaker evidence. The two units are intentionally
distinct and are labeled in the payloads and evidence receipts.

## Governed reader

`presidential_profiles.story_foundation` is the single Story reader for:

- `speaker_views/paragraph_view_v1.parquet`;
- `speaker_views/appearances_v1.parquet`;
- `reference_entities/entity_mentions_v1.parquet`; and
- `reference_entities/era_distinctive_v1.parquet`.

Loading first runs the portable foundation audit and then refuses metadata,
inventory, hash, schema, key, era, denominator, support, ordering, badge, or
evidence drift. The four Parquets are read once into an immutable bundle and
that same bundle is passed to all Story producers. No producer may substitute
row order for `(doc_name, para_idx)` or introduce a Story-only entity alias.

The accepted population receipt is 35,394 retained paragraphs, 32,531 eligible
actual-president paragraphs, 2,863 exclusions, 296 cross-owner paragraphs, and
1,054 actual-speaker appearances. The nine eligible denominators are 672,
3,847, 2,610, 6,721, 2,134, 1,409, 5,583, 6,144, and 3,411.

## Story contracts

Every Era Profile publishes five `Distinctive references`: positive
Jeffreys-smoothed log-odds candidates backed by the primary AI, supported by at
least five paragraphs in two source documents, and compared with the other
eight eras. The 45 rows contain 29 `AI + NER` and 16 `AI only` receipts; a
`NER only` row cannot become a headline. Source agreement is not historical
validation or model confidence, and NER never supplies stance.

Each native disclosure includes the full denominator and comparison, a bounded
HTML excerpt, nullable `Primary-AI stance`, exact AI/NER strings, actual speaker,
source document owner, Miller Center link, and canonical paragraph key. Full
artifact excerpts remain in JSON. Cross-owner evidence is always explicit.

Era Profiles are `era-profile-v5`, visualizations are
`era-visualizations-v9`, and contextualizations are
`era-contextualizations-v11`. Era Echoes retains 1,243 of 1,447 source rows,
with 48 unresolved keys, 156 speaker-ineligible rows, and 27 actual-speaker
reassignments reported rather than silently discarded.

## Public downloads and refusal boundary

The generated `docs/data/story/` directory contains the 45-row distinctive CSV,
the compact 1,054-row appearance CSV, a byte-identical copy of the governed
entity Parquet, and a hashed `story-foundation-public-v1` manifest. Public JSON,
CSV, and Parquet parity is checked both before generated-site writes and after
the build. Migration-owned JSON and download writes use atomic replacement.

Run the fast receipt with:

```bash
arch -x86_64 .venv/bin/python -m presidential_profiles.story_foundation \
  --check --site-dir docs
```

The build refuses stale foundation inputs or public projections before it can
write `docs/`. Frozen paid annotations, sealed speaker attribution, the general
annotation ledger, constituency pilots, Summary, Compare, Explore, Profiles,
Issues, and deployment remain outside this migration.

## Verification receipt

The requested focused suite passes all 99 tests; the complete repository suite
passes all 2,725 tests with 69 existing warnings. The canonical build validates
73 HTML pages and 183 JSON shards, all 130 inline scripts parse, and desktop,
mobile, keyboard/focus, reduced-motion, source-link, cross-owner, disclosure,
overflow, and console checks pass in the in-app browser. The protected paid and
speaker-attribution inventories retain their pre-implementation checksum.
