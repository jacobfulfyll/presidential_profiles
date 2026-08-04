# Era-profile constituency publication plan

Status: groomed backlog plan; blocked at the two entry gates below. This plan does not authorize
promotion, materialization, a site build, or deployment.

Task ID: `publish-era-constituency-categories`

## Objective

Replace the pending/illustrative constituency state in each of the nine shared Era Profile
screens with a comparable summary of the constituency categories explicitly invoked in that era.
Keep the result tied to promoted paragraph-level claims, exact canonical keys, declared
denominators, and inspectable evidence. Do not turn audience, subject matter, favorable mention,
or an inferred historical population into a constituency claim.

## Current evidence and entry gates

The recently completed corpus-wide source is the sealed provisional composite
`pcomp_a8580fb16926dbe523cd6a9e1ceed165b99f2f0f5828f031b49c8e6f4bc5d02b`.
Its manifest declares the layer provisional, not materialized, not promoted, and not production
eligible. The active materialized generation still publishes a schema-valid
`constituency_claims.parquet` with zero rows. The earlier candidate evaluation passed 114 of 150
gates, including 44 of 57 constituency gates, and `ARCV1-RES011` rejected the freeze checkpoint.
A read-only grooming audit also found one remaining `unclear` constituency outcome in the sealed
composite.

Implementation may start only after both gates pass:

1. **Production-readiness gate.** An explicit owner disposition must produce a production-eligible,
   promoted, materialized constituency projection with no `unclear` rows. The site task does not
   reinterpret the failed pilot, promote the provisional composite, or read the composite
   directly.
2. **Single-universe gate.** The Era Profile inputs and the constituency projection must carry the
   same canonical corpus fingerprint. The provisional labels cover the corrected 1,053-speech /
   35,394-paragraph universe, while the current site profile still derives its footprint from the
   older 1,057-speech / 36,229-paragraph frames. Do not mix canonical constituency numerators with
   legacy profile denominators. Land or reuse the canonical Story-input migration first.

At pickup, re-read the active generation metadata and these gates rather than assuming this
snapshot is still current.

## Display quantity

The profile will rank the closed `group_type` categories from the constituency candidate contract,
using short controlled display labels. It will not rank raw `group_text` values.

| `group_type` | Profile label |
|---|---|
| `national_public` | National public |
| `geographic_population` | Geographic populations |
| `racial_ethnic_indigenous_or_legal_status` | Racial, ethnic, Indigenous & legal-status groups |
| `economic_class` | Economic classes |
| `occupation_or_industry` | Occupations & industries |
| `military_or_veteran` | Military personnel & veterans |
| `party_or_movement` | Parties & movements |
| `religious_group` | Religious groups |
| `age_gender_or_family` | Age, gender & family groups |
| `foreign_population_or_nation` | Foreign populations & nations |
| `institution_or_organization` | Institutions & organizations |
| `other` | Other explicitly claimed groups |

The mapping is exhaustive. A later specification that adds, removes, or redefines a category
requires re-grooming rather than silently falling into `other`.

This choice is deliberate. The candidate contract says that no model-supplied
`normalized_group` is authoritative. The sealed composite contains 31,510 claim objects and
13,952 distinct case-folded `group_text` values; 11,581 of those literal forms occur once, and
common raw strings include pronouns. A literal-name leaderboard would therefore combine alias
fragmentation, unresolved reference, and era-specific wording with the historical quantity.
Creating a versioned normalized-name ontology is a separate annotation task.

For each era and each controlled `group_type`:

- retain only promoted rows whose outcome is `claim`;
- count a paragraph once when it contains one or more claims of that type;
- count distinct supporting speeches separately;
- divide the paragraph union by all canonical paragraphs assigned to the era;
- retain every qualifying relation and stance, including neutral, adversarial, and mixed claims;
- rank by paragraph support descending, then speech support descending, then controlled display
  label ascending;
- publish the leading five categories;
- mark a row `episodic` only when all its support comes from one speech; and
- choose the receipt with the smallest canonical `(doc_name, para_idx, claim_index)` so evidence
  selection is deterministic rather than editorial.

Categories are non-exclusive. A paragraph may contribute to more than one category, and multiple
relations or repeated claims of one category in a paragraph still contribute only one paragraph
to that category. The visible note must say that the five shares are not additive.

## Data and contract work

1. Make `era_profiles` the single loader for the active constituency projection and remove the
   duplicate loader in `site.py`.
2. Validate the active pointer, promotion/materialization receipt, corpus fingerprint, exact
   `(doc_name, para_idx)` containment, claim-index uniqueness, outcome/claim shape, and the closed
   `group_type`, `relation`, `stance`, and `certainty` enums before aggregation. Unknown keys,
   stale fingerprints, unknown enum values, duplicate claim rows, and any promoted `unclear`
   outcome fail before rendering.
3. Preserve the current behavior when no eligible projection exists: Founding retains its clearly
   marked illustrative fallback and the other eight eras remain visibly pending. An empty but
   eligible measured era uses an artifact-empty state rather than the Founding fallback.
4. Bump the public payload from `era-profile-v2` to `era-profile-v3`. Each displayed constituent
   row will contain:
   `rank`, `group_type`, controlled `label`, `paragraphs`, `speeches`,
   `paragraph_share`, `episodic`, observed `relations`, observed `stances`, and one exact receipt
   containing `doc_name`, `para_idx`, `claim_index`, `group_text`, `resolved_referent`,
   `evidence_span`, and `certainty`.
5. Add a `constituency_coverage` receipt to every profile with the canonical era denominator,
   any-claim paragraph and speech support, published-category count, source generation/promotion
   identifiers, corpus fingerprint, and specification hash. Keep raw claim text out of copied
   constants.

## Renderer work

- Change the artifact-backed card heading to `Claimed constituency types`; keep fallback wording
  honest while the artifact is unavailable.
- Show the controlled category label, exact paragraph share, paragraph count, and speech count on
  every row. Add an `Episodic` marker only for the declared one-speech case.
- Add a native, collapsed `Measure & evidence` disclosure. It states the operational claim,
  all-paragraph denominator, overlap rule, top-five selection, and the fact that constituency
  status does not imply favorable stance. It exposes the deterministic keyed receipt for each
  visible category.
- Escape every controlled label and every model/corpus-derived string at the HTML sink. Do not
  introduce an `innerHTML` path for evidence. Preserve a complete text alternative, keyboard
  access, reduced-motion behavior, content-sized expansion, and local rather than page-level
  overflow.
- Replace the Founding illustrative fallback only when the same eligible artifact supplies its
  measured rows. Never fabricate a missing era from topics, entities, audience, or neighboring
  eras.

## Tests and verification

Add synthetic contract tests before depending on the real artifact:

- exact-key and corpus-fingerprint mismatch refusals;
- duplicate claim-index, unknown enum, and promoted-`unclear` refusals;
- paragraph-union de-duplication across repeated claims and multiple relations;
- inclusion of non-favorable stances without double-counting;
- accession-year era assignment through the existing shared story rule;
- stable top-five ordering and tie breaks;
- one-speech episodic status;
- eligible-empty versus unavailable/fallback states;
- schema-v3 provenance and coverage receipts; and
- HTML escaping for controlled labels, raw group text, referents, and evidence spans.

After the entry gates pass, add a real-artifact contract asserting all nine profiles have a
measured state, every displayed row reconciles to the promoted projection, all denominators
reconcile to the canonical corpus, and no unresolved/pending row is silently omitted.

Run, in order:

```bash
arch -x86_64 .venv/bin/python -m pytest \
  tests/test_era_profiles.py \
  tests/test_founding_story.py \
  tests/test_annotation_ledger.py \
  tests/test_site_validation.py
arch -x86_64 .venv/bin/python -m pytest
arch -x86_64 .venv/bin/python -m presidential_profiles.site
```

Then run the existing static inline-JavaScript syntax check, `git diff --check`, and Reggie
Doctor. Browser QA must cover all nine profiles at desktop and 390 CSS pixels, the collapsed and
expanded evidence disclosure, keyboard operation, exact visible support, zero page-level
overflow, and an empty console. Do not deploy unless separately requested.

## Expected implementation surface

- `src/presidential_profiles/era_profiles.py` — active projection validation, aggregation, and
  `era-profile-v3` payload.
- `src/presidential_profiles/site.py` — shared loader use and accessible card rendering.
- `tests/test_era_profiles.py` — all-era data and refusal contracts.
- `tests/test_founding_story.py` — fallback replacement and rendered Founding behavior.
- `tests/test_annotation_ledger.py` — typed promoted-projection invariants, if the prerequisite
  promotion/materialization task changes them.
- `tests/test_site_validation.py` — generated payload/link/HTML checks as needed.
- `README.md` — current data source, measure, and fallback behavior.
- `docs/` — generated only after source and tests pass.

## Out of scope

- Promoting or treating the provisional composite as production data.
- Freezing a failed candidate or changing an owner decision.
- Editing `data/llm_annotations/` or any sealed raw response.
- Inventing normalized constituency names from raw literals or pronouns.
- A long-run causal claim about who presidents represented in practice.
- Redesigning other Era Profile cards, changing era boundaries, or deploying the site.
