# Story reference landscape v1

Status: implemented locally; generated output validated; not deployed.

## Decision

The Era Profile top-left card now presents four reference-type lanes instead of
publishing the five governed `Distinctive references` as a second ranked name
list. The card has the visible heading `Distinctive era references` but no
explanatory deck or comparison bars. `Distinctive` means unusually concentrated
in the selected era; it does not claim that a name occurs in no other era.
Named adversaries retain their existing data, five-bubble encoding, legend,
and interaction.

The governed `era_distinctive_v1.parquet` artifact and its public download are
retained unchanged as a foundation receipt. They are no longer rendered in the
Era Profile. The card therefore contains no visible Jeffreys-prior, log-odds,
comparison-era, ranking-policy, source-agreement-method, or download footer
copy.

## Reference landscape contract

The card uses the accepted `StoryFoundationBundle` only. For each Story era it
reports paragraph presence for four primary-AI entity types:

- people;
- institutions;
- groups or communities; and
- nations or places.

The denominator is every eligible actual-president paragraph in the era,
including paragraphs without a named entity. One paragraph counts once per
type, so the four shares are non-additive. Each lane prints the exact era share
and all-corpus share without adding a visual magnitude encoding.

Each lane may publish one compact highlight. Supported candidates must be
primary-AI backed, supported by at least five paragraphs and two source
documents, positively concentrated in the era, and at least 80 percent
favorable or neutral under the existing primary-AI stance labels. An entity
displayed in that era's Named adversaries card is excluded. Candidates are
ordered deterministically by concentration, paragraph support, document
support, and normalized entity ID.

If no supported candidate exists, the card may show a visibly labeled
`Limited record` candidate with exactly four paragraphs across at least two
documents, provided every other stance, concentration, and adversary-exclusion
gate passes. This does not relabel the candidate as supported. In 1913–1932,
that rule exposes the American Legion's four qualifying paragraphs across four
documents without weakening the five-paragraph supported-highlight floor.

NER anchors or corroborates the selected evidence span. It never supplies
stance, promotes an NER-only entity, or acts as model confidence or historical
validation. Each available highlight retains a closed native disclosure with
exact paragraph/document support, stance counts, source-agreement badge,
actual-speaker/source-owner identity, excerpt, paragraph key, and Miller Center
link. All strings are escaped by the server renderer.

## Layout and publication

The top pair remains one two-column grid row on desktop. The landscape uses a
visible heading and four compact two-column lanes: exact type percentages on the
left, a quiet vertical rule, and a flat shaded entity-name/evidence column on
the right. The right column uses no rounded container or inset accent. Source
agreement is plain metadata, and `AI only` renders once rather than repeating
`AI` as an icon and label. Each selected name is centered and modestly enlarged;
paragraph count, document count, source agreement, and limited-record status
share one smaller metadata line beneath it. This lets the names read as one
scan column while allowing the landscape and Named adversaries to settle to
the same closed height without padding the adversary content. The cards stack
without page overflow at mobile width. Opening evidence may temporarily grow
the row.

Era Profile JSON is `era-profile-v6`; `era-visualizations-v9` and
`era-contextualizations-v11` are unchanged. The v6 profile retains the v5
`distinctive_references` projection for foundation/public-download parity and
adds `reference_landscape` for the active UI. No frozen paid annotation,
speaker attribution, foundation Parquet, adversary row, topic measure, or
other Story screen changes.

## Acceptance

- All nine profiles expose four ordered lanes and three or four highlighted
  names without displayed-adversary overlap; limited records remain visibly
  distinct from supported highlights.
- Every highlight satisfies support and favorable/neutral gates.
- The visible card contains no Jeffreys, log-odds, ranking-policy, or download
  footer copy, explanatory deck, or comparison bars.
- Named adversary names, counts, types, SVG encoding, and legend are unchanged.
- The two closed desktop cards have equal measured height in every era.
- Native disclosures, escaped evidence, source links, keyboard focus, and
  mobile wrapping remain functional.
- Story foundation/publication checks, the canonical build, static validation,
  and the relevant/full test suites pass before handoff.

## Verification receipt

The 83 focused/related tests and all 2,773 repository tests pass with 69
existing warnings. The canonical builder validates 73 HTML pages and 204 JSON
files. Foundation, Story v6/v9/v11, Plan 3, Summary projection, Python
compilation, 130 inline scripts, static JavaScript, and whitespace checks pass.
Browser inspection confirms the two closed desktop cards measure 230px in
all nine eras, evidence disclosures resolve safely, and the page has no
horizontal overflow at desktop or 390px-class mobile width.
