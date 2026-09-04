# Explore interaction and evidence v1

Status: implemented and verified locally on 2026-09-03; awaiting owner release approval.

This is the durable implementation contract for Step 6 in
`notes/final-release-program-v1.md`'s proposed execution order. It supersedes
the umbrella note only where this document makes a more specific Explore
decision. Story, Summary 05, Compare, the speaker/reference foundation, and
the actual-speaker topic-network bundle remain protected regression surfaces.

## Decision summary

- Explore is one short, progressive page: **Choose series → Trends → Data and
  methods**. Selection chips, word entry, the governed catalog, uncertainty,
  context, and guided comparisons all live in the first section.
- The default is three exact lexical series—`tariff`, `freedom`, and `border`—
  with topic uncertainty shown, no historical context, and both disclosures
  closed. A user may select at most six series.
- The long flat topic picker is replaced by a native search field and governed,
  collapsible categories: 16 deterministic broad issues, 50 detailed AI topics
  nested under their 17 taxonomy parents, and three case-sensitive named
  acronyms. There is no UI-owned taxonomy and no recent-items feature.
- Word grouping is a primary radio choice beside word entry and a reversible
  per-chip action. Grouped chips name the canonical family, state `Grouping
  on`, give the form count, and expose the exact audited forms in a native
  disclosure.
- Plotly is removed from Explore. A small external renderer builds semantic SVG
  panels from published values. Pointer hover, keyboard focus, click/tap, and
  Enter/Space pinning operate on whole trajectories; arrow keys inspect exact
  periods; Escape clears a pin.
- The uncertainty control is in the main control bar and is called `Show
  uncertainty ranges for topic series`. It controls only the visible ranges;
  support and unresolved markers remain. The two governed band trust gates and
  method-specific `ci_components` pass through unchanged.
- The former chart-options disclosure is retired. Word grouping is primary,
  historical context is one primary select, combining selected words is
  removed, and the uncertainty toggle is primary.
- Exact values do not dominate the page. A prominent selected-series CSV
  download is followed by one collapsed, one-series-at-a-time exact table.
  The server HTML contains a complete default chart, 708 default exact rows,
  and three static CSV links for no-JavaScript and load-failure use.
- Existing page payloads were not sufficient as a governed browser boundary.
  Explore now has a deterministic, versioned, field-preserving 263-file public
  projection. The browser decodes transport fields and renders them; it does
  not calculate a rate, join, denominator, scope, interval, support state, or
  analytical ranking.

## Inspected baseline and selected options

The prior Explore already supported exact words and phrases, a global grouped-
word option, 66 topic series plus three acronyms, nine editable guided word
comparisons, historical era shading, URL restoration, a CSV action, and a
collapsed evidence disclosure. Those foundations were retained where they
were analytically sound.

The partial or missing pieces were structural: the governed series lived in a
single native select; grouping was buried under `Chart options`; current
selections did not explain family membership; hover did not provide a complete
keyboard/touch trajectory model; uncertainty was not in the main flow; exact
and method material were bundled into one evidence area; old unversioned shards
did not seal their full inputs or publish a complete field-preserving topic
table; and the browser owned more analytical assembly than the current project
boundary permits.

Measured against the checked-in pre-Step-6 Explore:

- The old page was 40,744 bytes and loaded the 4,851,164-byte Plotly runtime.
  Its cold default path was six successful resources, 5,560,992 raw bytes, and
  1,591,132 deterministic-gzip bytes.
- The old `docs/explorer/` contained 108 files and 25,453,332 raw bytes.
- The implemented cold default path is seven successful resources—HTML, CSS,
  JavaScript, the index, and three lexical shards—but still only four public
  data requests. It is 803,827 raw bytes / 201,243 deterministic-gzip bytes, an
  87.4% gzip reduction despite the complete server fallback.
- Final `explorer.html` is 142,384 raw / 20,978 gzip bytes; its CSS is 11,018 /
  2,828 and its JavaScript is 48,422 / 13,732. Explore makes no Plotly request.
- The complete new projection is intentionally broader: 263 files,
  78,370,786 raw bytes, and 19,312,004 deterministic-gzip bytes. It stays lazy
  and below its declared aggregate and per-shard budgets.

The options resolved as follows:

1. **Governed native catalog, not a custom combobox.** Search plus native
   `details`, `summary`, labels, and checkboxes gives predictable touch,
   keyboard, and screen-reader behavior without a bespoke ARIA listbox.
2. **No recent-items list.** A deterministic cross-session recency definition
   would require storage and add another order. Visible selected chips and
   search solve the actual retrieval problem.
3. **Grouping before entry and after selection.** A pre-entry-only control made
   correction expensive; a chip-only control hid the choice. Both paths use
   the same existing resolver and record namespaces.
4. **Native SVG, not retained Plotly.** The required whole-line focus/pin model,
   marker grammar, small initial transfer, and controlled safe-DOM construction
   are simpler with a page-specific renderer.
5. **One collapsed exact table plus CSV, not a permanent matrix or tooltip-only
   explanation.** This is the smallest coherent path that keeps visible exact
   values, provenance, structured fallback, and deep export.
6. **A v2 projection, not the old shards.** The selected projection makes
   every rate and topic cell generator-owned, seals the input set, supports
   exact/grouped unigrams and bigrams independently, preserves band fields,
   validates parity, and permits atomic stale-output replacement.

## Information architecture and default state

The document order is fixed:

1. Page title and a two-sentence scope/unit introduction.
2. `Choose series`:
   - selected count, Clear, and Reset;
   - selected chips in A–F order;
   - `Add a word or phrase` and `Browse governed series` cards;
   - the uncertainty/context control bar;
   - collapsed guided comparisons;
   - one polite status line and a dismissible/retryable failure panel.
3. `Trends`:
   - a unit/grain summary;
   - only the non-empty lexical, broad-issue, and detailed-topic panels, in that
     order;
   - one shared exact-value readout;
   - the prominent selected-series CSV action;
   - collapsed `Exact values and support`.
4. `Data and methods`, containing permanent unit/scope and band-limit language
   plus Methods and Data Quality links.
5. A concise source/provenance footer and the standard feedback link.

The default selection is ordered `lexical-exact:tariff`,
`lexical-exact:freedom`, `lexical-exact:border`. Uncertainty is on; historical
context is `None`; word-entry mode is exact; search is empty; no series is
pinned; guided and exact disclosures are closed. Clear produces a real empty
chart and `0 of 6 selected`. Reset restores the full default, uncertainty on,
no context, and no active preset.

## Series discovery and selection

### Governed sources and order

- Broad issues come from `issues_meta.json` in its governed issue order,
  extended only through `topic_quality.display_issues`. Display labels must
  equal `topic_quality.display_name`; page rendering refuses registry drift.
- Detailed topics come from frozen `taxonomy_v1.json`: all 17 Level 1 groups in
  taxonomy order and all 50 Level 2 topics in taxonomy order.
- Named acronyms come from `word_families.ACRONYMS` in governed order: `SALT`,
  `START`, and `AIDS`. These are case-sensitive catalog records; typing `aids`
  in word entry remains an ordinary lower-case lexical query.

Broad issues open initially. Detailed topics and acronyms are collapsed; each
detailed Level 1 parent is independently collapsible. Search preserves the
pre-search disclosure state, temporarily opens only matching branches, and
restores the saved state when cleared.

### Search rules

Search applies NFKC, English-US lower-casing, non-alphanumeric separation, and
AND semantics. Every query token must prefix at least one governed search-term
token. Search terms are generator-published labels, stable source labels,
definitions, anchor words, and the Level 1 parent where applicable. Results
never reorder. Zero results show: `No governed series matches every search
term. Try a shorter word or a parent topic.`

### Selection rules

- Six series is a hard maximum. At the limit, unselected catalog checkboxes are
  disabled; another word submission returns the same plain-language limit.
- Catalog checkboxes add/remove; every chip has a 44px `Remove` button. Remove
  moves focus to the next chip, then the previous chip, then word entry.
- Duplicate identity is mode-aware. The same exact query cannot be added twice;
  different spelling aliases resolving to the same family cannot create two
  grouped families. Exact and grouped forms may coexist because they answer
  different questions. Switching is refused if it would duplicate an existing
  series.
- All selection changes preserve display order. No UI sorting or ranking is
  introduced.

## Word-family workflow

The primary choice `How to count the next word` has `Exact form` and `Group
word forms` radios. Every lexical chip can switch modes afterward without
changing its slot. Grouping loads `families_v2.json` lazily; if that file fails,
exact entry remains available and grouping is disabled with a retryable error.

The source of truth is the existing checked-in `data/word_families.json`, the
existing `word_families.NORMALIZE` spelling map, and the existing tokenization
contract. Step 6 does not regenerate that artifact or add an NLP model. The
projection publishes 4,599 families, a 7,391-entry resolver, and 13 spelling
normalizations with this permanent claim boundary: algorithmic word-form
grouping is not a claim of linguistic equivalence.

Input behavior is fixed:

- trim; turn curly apostrophes into straight apostrophes; lower-case lexical
  input; punctuation and hyphens separate tokens; retain internal apostrophes;
- accept one indexed word or one adjacent two-word phrase only;
- reject letters outside English A–Z rather than silently deleting them;
- require at least 30 whole-corpus occurrences for a unigram and 15 for a
  bigram in the selected mode;
- in grouped mode, apply the existing spelling normalizations and resolver per
  token; an otherwise supported unmapped token is a singleton family;
- preserve a stable query in URL state while using the canonical family key for
  duplicate prevention and shard lookup.

A grouped chip shows its canonical display family, `Grouping on`, and a native
disclosure titled `<n> forms combined`. A unigram lists every form directly; a
bigram labels the included forms by position. Switching back shows the exact
query and `Exact form`.

## Trends, inspection, and uncertainty

The renderer creates up to three aligned panels from the current selection:

- words and named acronyms: uses per 10,000 indexed source-document words,
  centered five-year windows;
- deterministic broad issues: percent of eligible paragraphs, governed
  five-year periods;
- detailed AI topics: percent of eligible paragraphs, nine named eras.

Panels share 1785–2026 horizontally but use independent zero-based vertical
scales because their units and grains differ. Panel titles and units remain
visible. Slot letter A–F, color, dash pattern, and marker shape travel together;
color is never the only identifier.

Pointer entry on a legend or 24px transparent line hit target highlights the
entire trajectory and dims the others. Pointer movement along a line selects
the nearest published period and updates the visible exact readout. Focusing a
legend does the same for the full line and begins at its latest value. While a
legend owns focus, keyboard focus takes precedence over a stationary pointer.

Keyboard behavior is fixed:

- Left/Right: previous/next available period;
- Home/End: first/latest available period;
- Enter/Space: pin or unpin the series at the inspected period;
- Escape: clear the pin; the focused trajectory stays highlighted until blur.

Keyboard inspection uses a dedicated polite live region; pointer movement does
not flood it. Click/tap on a legend or a line pins the same state. Pins,
inspection period, hover, focus, and disclosure positions are ephemeral and do
not enter URL/history.

### Band semantics

`Show uncertainty ranges for topic series` defaults on and belongs in URL
state only when off. It applies to broad and detailed topic series, never to
lexical rates. Turning it off removes filled ranges but leaves status line
style, low-support markers, unresolved markers, exact-table language, and CSV
fields intact.

The renderer passes through both trust gates:

- period-grain `ci_status`: `ok`, `low_cluster_caution`, or
  `suppressed_n_floor`;
- cell-grain `interval_unresolvable`.

`ok` is solid. `low_cluster_caution` remains a dotted trajectory with a †
marker and cautious status. `suppressed_n_floor` keeps the point, draws ‡, and
states that the range was not published. `interval_unresolvable` keeps the
point, draws an open ◇ marker, and states that the bootstrap could not resolve
a range. Null bounds are always unknown, never zero. CorEx retains
`ci_components=sampling_only`; AI topics retain their published
sampling-plus-annotator-disagreement components, paired counts, agreement
source, and widening fields.

The single historical-context select uses the nine existing `trends.ERAS` and
adds only a quiet presentation band; it does not filter or recalculate data.

## Evidence, exact values, and guided comparisons

The selected-series download contains exactly 38 ordered fields:
selection identity/order; instrument/query/grouping/family fields; period
identity/order/bounds; displayed value/unit and numerator/denominator; point
and sampling/final bounds; paragraph, speech, and paired support; both trust
gates; disagreement provenance; value status; and source scope. Numeric cells
remain numeric, nulls remain empty, booleans are explicit, and strings starting
with spreadsheet formula characters are apostrophe-protected.

The collapsed HTML table shows only five reader-facing columns for one selected
series: period, estimate, published uncertainty, evidence/support, and status.
Its caption names the series, instrument, period grain, unit, grouping mode,
and source scope. Full component fields remain in the selected download and the
static topic download; tooltip-only text is never the sole explanation.

All nine existing guided comparisons remain because each publishes a clear
question, exact word list, rationale, and known ambiguity. Activating one
atomically replaces the selection with its two-to-six exact lexical series;
partial guide loads never replace the current selection. Guide state adds a
`preset` URL hint only while the selected sequence still equals the preset.
The cards, slots, chart, inspection, and exact paths are the same components as
ordinary selections.

## URL and history state machine

Canonical state is query-string version 2:

- `state=2` when explicit state exists;
- repeated ordered `s=` entries using `lexical-exact:<query>`,
  `lexical-family:<query>`, `corex:<stable source label>`,
  `llm:<stable taxonomy label>`, or `acronym:<token>`;
- `uncertainty=off` only when ranges are hidden;
- `context=<governed era key>` only when selected;
- `preset=<key>` only while the exact preset sequence remains active.

Add, remove, switch grouping, catalog change, clear, reset, uncertainty,
context, and guided-preset changes push history. Initial canonicalization,
legacy migration, invalid-entry pruning, duplicate pruning, and limit pruning
replace the current entry. Popstate rebuilds the selection transactionally;
request serials and selection epochs prevent a slow old response from
overwriting newer state.

Legacy `state`/`s=word|topic|entity`, `words`, `topics`, `entities`, `grouped`,
`combine`, and `periods` inputs restore where possible, then replace themselves
with v2. Combined-word state is retired into individual series. Multiple legacy
periods reduce to the first recognized context. Unknown, unsupported,
duplicate, and seventh-plus entries are skipped and announced rather than
crashing or fabricating state.

## Public projection and publication boundary

`explore_projection.py` is the only analytical producer. The canonical site
build creates the projection twice in memory and refuses non-identical bytes
before any Explore publication.

The exact inventory is 263 files:

- `manifest_v2.json` (`explore-publication-manifest-v2`), inventorying the other
  262 files with schema, rows, bytes, deterministic-gzip bytes, and SHA-256;
- `index_v2.json` (`explore-index-v2`), containing limits, the 236-period
  lexical axis, compact-record definition, governed catalogs, defaults,
  contexts, guides, shard routing, downloads, and metric IDs;
- `families_v2.json` (`explore-family-index-v2`), containing the audited
  resolver, normalization table, family display records, and algorithm claim
  boundary;
- `topic_values_v2.json` (`explore-topic-values-v2`), containing all 1,234
  field-preserving rows for 16 CorEx and 50 AI series plus declared field order
  and trust gates;
- `default_values_v2.csv`: 708 rows × 38 fields;
- `topic_values_v2.csv`: 1,234 rows × 38 fields;
- `series_catalog_v2.csv`: 69 rows × 9 fields;
- 256 `explore-lexical-shard-v2` JSON files: 64 buckets for each of exact
  unigram (`lexical/eu`), exact bigram (`lexical/eb`), family unigram
  (`lexical/fu`), and family bigram (`lexical/fb`).

Lexical records are generator-computed compact tuples in the declared order
`corpus_total_count`, delta-encoded period offsets, numerator counts, and
`rate_per_10k_x10000`. FNV-1a over UTF-8, low six bits, addresses the shards;
golden vectors are `tariff=17`, `freedom=11`, and `border=37`. Decoding fixed
point and delta offsets is transport work, not analytical derivation.

The manifest source identity hashes the speeches, audited families, bands and
band metadata, issue metadata, display-name registry, frozen taxonomy, the
projection source, and the word-family source. Build refuses missing or changed
inputs, a stale band corpus fingerprint, a stale family corpus receipt, band
row drift, schema/cardinality drift, hash drift, a key in the wrong shard, a
source change between generation and publication, or a publication that does
not exactly match the manifest.

Budgets are hard failures: each lexical shard ≤600,000 raw and ≤150,000 gzip;
the full projection ≤90,000,000 raw and ≤23,000,000 gzip. The final largest
shard is 522,404 raw / 125,031 gzip. Publication builds a validated sibling
directory, byte-checks it, swaps `docs/explorer/` atomically, rolls back a failed
swap, and deletes old unversioned shards only as part of that successful whole-
directory replacement.

## Security and browser boundary

All server HTML uses escaped output. Broad display labels are rechecked against
the centralized registry when the page renders. The browser uses only
`createElement`, `createElementNS`, `textContent`, attributes, and
`replaceChildren`; it contains no data-driven `innerHTML`, `outerHTML`,
`insertAdjacentHTML`, `document.write`, or `eval` sink. Dynamic and static CSV
paths both guard formula-leading strings.

The browser may filter and order already-published catalog records, decode
compact transport arrays, select the nearest existing period for inspection,
calculate SVG coordinates/scales, and serialize published rows. It must not
join paragraph data, count text, calculate a rate/share/interval, infer a
support label, widen a band, map a taxonomy, or change a denominator/scope.

## Accessibility, responsive behavior, and fallbacks

- Native radios, search, details/summary, labels, checkboxes, select, and
  buttons provide the control semantics. Regions and headings follow document
  order; chart SVGs have title/description references.
- Every primary target is at least 44px. Slot letters, dash patterns, and marker
  shapes supplement color. Forced-colors mode maps trajectories/grid/markers to
  system text colors, retains dashes/shapes, gives bands a system fill, and
  strengthens dimmed opacity. Reduced motion removes trajectory transitions and
  smooth scrolling.
- Selected chips use three columns wide, two at ≤899px, and one at ≤639px.
  Selector cards stack at ≤899px; guide cards follow the same two/one pattern.
  At ≤480px the word input/button stack. The catalog and exact table scroll
  internally; the page never gains horizontal overflow.
- With JavaScript unavailable, the three default selections, semantic SVG,
  three complete 236-period exact tables, unit/method/source copy, and links to
  all three static CSVs remain. Client controls are disabled and honestly say
  that interactive selection needs JavaScript.
- Before hydration, that fallback stays visible and all client controls remain
  disabled. After successful index load it is replaced atomically. On index
  failure, the fallback stays, controls stay disabled, a plain status explains
  the limitation, and Retry/Dismiss are available. Family or topic failures are
  narrower: preserve current selection, disable only the unavailable path where
  appropriate, and offer a retry.
- Native exact/guided disclosures retain focus on their summary when closed.
  Selection removal and grouping switches restore focus to a deterministic
  nearby control.

## Metrics and analytical provenance

`metrics.py` registers `explore_centered_lexical_rate` as count of exact or
audited grouped occurrences divided by all indexed source-document words in a
centered five-year window, multiplied by 10,000. The chart registry maps:

- `explore_lexical_trends` → `explore_centered_lexical_rate`;
- `explore_corex_trends` → `paragraph_share`, `confidence_interval`;
- `explore_ai_topic_trends` → `paragraph_share`, `uncertainty_envelope`.

Lexical windows publish only when the word denominator is greater than 20,000;
zero in a supported window is a real zero, while an unsupported window is
unknown. Broad and AI topic cells preserve their governed numerator,
denominator, point, interval, support, surface, agreement, and source-scope
fields. Explore remains a source-document/eligible-annotation corpus view; it
does not claim actual-president paragraph scope. Multi-label topic shares are
non-additive.

## Implementation map

- `src/presidential_profiles/explore_projection.py`: all analytical generation,
  schemas, receipts, parity validation, budgets, and atomic publication.
- `src/presidential_profiles/explorer.py`: escaped semantic HTML, default SVG,
  exact fallback tables, centralized display-name validation, and page write.
- `src/presidential_profiles/explore_assets.py`: deterministic external CSS and
  safe-DOM renderer.
- `src/presidential_profiles/site.py`: one shared corpus/bands input, two-build
  determinism gate, atomic publication, page generation, final validation.
- `src/presidential_profiles/metrics.py`: lexical definition and three chart
  registrations.
- `tests/test_explore_projection.py`, `tests/test_explorer_page.py`, and focused
  additions to conftest/security/metric tests: projection, parity, UI, URL,
  accessibility, security, budgets, and regression contracts.
- `docs/explorer.html`, `docs/assets/explorer-v2.{css,js}`, `docs/explorer/`, and
  metric pages are generated output only.

## Verification and acceptance

The required order is:

```bash
arch -x86_64 .venv/bin/python -m pytest \
  tests/test_explore_projection.py tests/test_explorer_page.py \
  tests/test_word_families.py tests/test_global_navigation.py \
  tests/test_metric_registry.py tests/test_bands.py tests/test_band_charts.py \
  tests/test_site_validation.py tests/test_display_name_security.py \
  tests/test_register_trends.py -q
arch -x86_64 .venv/bin/python -m pytest -q
arch -x86_64 .venv/bin/python -m presidential_profiles.site
node scripts/validate_inline_js.mjs docs
for file in $(rg --files docs -g '*.js'); do node --check "$file" || exit 1; done
arch -x86_64 .venv/bin/python -m compileall -q src tests
git diff --check
~/.codex/scripts/reggie/reggie_doctor.sh --repo \
  /Users/jacobpress/Desktop/Projects/presidential_profiles
```

Browser acceptance is required at 1280×720, 1024×768, 768×1024, and about
390×844. At every width: `scrollWidth == clientWidth`; controls/chips remain
legible; catalog and table overflow stay internal; 44px targets remain usable;
and the console is empty. Exercise raw and grouped unigrams/bigrams, both
grouping switch directions, issue/AI/acronym selection, six-item limit,
duplicate/empty/unsupported input, search/no results, clear/reset/remove,
legacy and v2 history, bands, context, pointer/keyboard/touch pinning, exact
tables, guides, downloads, disclosures, loading/failure, and fallback.

Final local receipt:

- 31 dedicated Explore tests and the 396-test focused/regression command pass;
- all 2,811 repository tests pass with 64 existing warnings;
- the canonical build validates 73 HTML pages and 376 JSON files/shards;
- all 128 executable inline scripts and every generated external JavaScript
  asset parse; Python compilation and whitespace checks pass;
- responsive browser widths have no horizontal overflow; pointer, keyboard,
  touch, history, exact, band, guide, loading/failure, and focus-return checks
  pass with no console warnings/errors;
- no frozen annotation, governed analytical artifact, Story, Summary, Compare,
  deployment, staging, commit, push, or unrelated worktree cleanup occurred.

## Protected surfaces, out of scope, and refusal conditions

Protected: Story and all Story contracts; Summary 05's approved recurring-only
topic-gravity field; Compare's Rhetoric → Agenda → Evidence and A/B/C contract;
Profile V3; Issues V2; speaker/reference and Plan 3 bundles; frozen paid
annotations; existing paragraph-key rules; band definitions; and all unrelated
dirty-tree changes.

Out of scope: a new taxonomy or NLP model; regeneration/hand-editing of word
families, bands, annotations, or taxonomy; actual-speaker migration of Explore;
combined word trends; topic ranking; cross-method score; adding analytical
measures in JavaScript; reorganizing Story/Summary/Compare/Profile/Issues;
deployment or git publication.

An implementing or future modifying agent must refuse publication when source
seals, expected cardinalities, band trust fields, display-name registry parity,
hashes, shard routing, budgets, two-build equality, exact inventory, metric
registration, static syntax, no-JavaScript fallback, or site validation fail.
Unknown bounds/values must never be coerced to zero, and a browser request for
new analytical work requires a new governed producer contract rather than a
renderer shortcut.
