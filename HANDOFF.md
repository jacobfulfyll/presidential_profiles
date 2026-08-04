# Handoff

## Latest addendum — GitHub Pages release candidate

- `notes/github-pages-release-plan-v1.md` is the owner-authorized publication
  contract. The existing GitHub Pages topology remains `master:/docs`; the
  release is prepared on `codex/github-pages-release` for review and merge.
- The canonical generator now writes an empty `docs/.nojekyll`, and the release
  includes independent inline-JavaScript parsing plus a fail-closed staged-tree
  audit in `scripts/audit_github_release.py`.
- The release allowlist covers the project documentation, generator, tests,
  scripts, governed small data artifacts, and complete generated `docs/` site.
  It explicitly excludes `data/annotation_ledger/`, `.codex/operators/`, and
  `.claude/stats.json`; frozen paid annotations were not regenerated or edited
  during release preparation.
- Release-candidate verification is green: 2,596 tests pass with 70 existing
  warnings; the canonical builder validates 73 HTML pages and 165 JSON shards;
  all 146 generated inline scripts parse; Python compilation and whitespace
  checks pass. The staged-tree audit, GitHub review/merge, Pages monitoring,
  and public-route verification remain before this task can be closed.
- `DECISIONS.md` records why this repository retains the native `master:/docs`
  Pages source and why the annotation control plane remains outside the public
  site release.

## Latest addendum — America in Summary V2

- `notes/summary-redesign-v2.md` supersedes V1's visible narrative. Summary now
  has five chapters: Voice, Conflict, Time, Emotional register, and Audit.
- The post–Civil War breadth graph and issue-lifecycle heat map are removed.
  Visible copy states that most topic-count expansion occurs before 1860 and
  later movement is small, uneven, and method-sensitive.
- Voice is a reduced WHO/HOW pair: Congress/general public/other and
  written/spoken/broadcast-plus-other. Both president views use six phase
  presets and nine colored era cards while retaining all 45 presidents.
- Enemy identity is displayed as nation/group/person/institution/other across
  every era. The observed annual-message combat lines no longer show confidence
  whiskers. A new president scatter compares enemy naming with partisan attack.
- The founding annual-message nostalgia audit reports 17 dictionary matches:
  10 `again`, 7 `restore*`, and 0 `back to`, with an explicit warning that this
  is not evidence of golden-age nostalgia.
- The emotional ratio is now NRC Hope divided by Doom on a log scale with the
  20,000-word and nonzero-denominator guards. The separate-axis weather map is
  no longer part of Summary. The generated page contains eight charts.
- Verification is complete: 2,591 tests pass with 70 existing warnings; the
  canonical builder validates 73 HTML pages and 165 JSON shards; all 146
  generated inline scripts parse. Browser QA at 1280×720 and 390×844 mounts all
  eight charts, exercises both phase selectors, keeps page overflow at zero,
  and reports no console warnings or errors.
- Final Reggie Doctor reports 0 errors, 0 warnings, and 5 informational
  findings: the two optional portable documents are absent, tool telemetry is
  changed, the existing worktree remains heavily dirty, and this repository
  correctly stays in native/external Reggie documentation mode without an
  adapter.
- V1's addendum below is retained as implementation history and is superseded
  where it conflicts with this section. Nothing was deployed, staged,
  committed, pushed, or placed on a new branch.

## Latest addendum — America in Summary

- `docs/summary.html` is now a six-chapter aggregate argument rather than a
  short matrix followed by three disconnected full-record charts. Its order is
  agenda breadth and Level-1 lifecycles; audience/medium and proposal/values;
  changing adversaries and combat framing; future plus nostalgia; guarded
  hype÷doom plus the separate-axis weather map; and the final
  changed/persisted/uncertain audit.
- The agenda headline carries the governed confirmatory receipt: modern
  1950/1980/2010 annual-message blocks minus the 1860/1890/1920 postbellum
  blocks, +2.17 effective topics, 95% interval −0.43 to +4.57, adjusted
  no-change surprise 4.29%, 56 messages. Visible copy states that this is not a
  monotonic trend and not a test of the 2017–2026 chapter alone.
- New exploratory views join paragraph artifacts on `(doc_name, para_idx)` and
  fail closed on key drift, missing eras/domains, category totals other than
  100%, or a missing registered receipt. Frozen paid annotations were neither
  regenerated nor edited.
- Audience and medium are aligned 100% rivers; proposal, mixed, values, and
  neither are mutually exclusive; the all-president scatter starts with all
  nine eras selected and adds All/Clear presets. Adversary succession retains
  the published alias boundary and exposes surprising raw classifications as
  an audit. Hype÷doom uses aggregate centered five-year counts, a 20,000-word
  floor, a zero-doom guard, absolute-rate tooltips, parity, and non-causal event
  guides.
- `notes/summary-redesign-v1.md` is the evidence and acceptance contract.
  Metric registrations and focused regression coverage were added. All 2,589
  tests pass with 70 existing warnings. The canonical build validates 73 HTML
  pages and 165 JSON shards; normal Story and Summary scripts parse; desktop
  and 390px browser QA mounts all ten Summary charts with no page overflow or
  console warnings/errors. Nothing was deployed, staged, committed, pushed, or
  placed on a new branch.

## Latest addendum — Al-Qaeda uses its full organization name

- The shared grouped-vocabulary layer now canonicalizes the `qaeda` token to
  `al-qaeda`, so the Post–Cold War Corpus Footprint displays `AL-QAEDA` rather
  than the fragment `QAEDA`. Both “al-Qaeda” and “al Qaeda” still contribute
  one occurrence to the same family.
- Grouped Explore resolves `qaeda` to the same `al-qaeda` node. The 68 focused
  tests pass, and the canonical build validates 73 HTML pages and 165 JSON
  shards. Frozen annotations were not changed and nothing was deployed.

## Latest addendum — Era vocabulary excludes president names and portraits link out

- Corpus Footprint now excludes every president-name token before ranking
  distinctive word families. This applies corpus-wide, not only to the visible
  examples Kennedy, Nixon, Bush, and Trump; the replacement terms are selected
  by the same grouped, saturated ranking used for all other vocabulary.
- Every presidential portrait in an Era Profile is now a keyboard-accessible
  link to that president's generated profile page. The visible name remains
  below the portrait, and hover/focus adds a clear ring without changing the
  established card layout.
- The public era-profile JSON contract is now `era-profile-v4`. The 116-test
  broader Story suite passes, the canonical build validates 73 HTML pages and
  165 JSON shards, and 146 inline scripts parse across 74 generated pages.
  Browser QA opened John F. Kennedy's profile directly from his Era Profile
  portrait and found no presidential-name leak in the Cold War vocabulary.
  Frozen paid annotations were not changed; nothing was deployed, staged,
  committed, pushed, or placed on a new branch.

## Latest addendum — Corpus Footprint shares Explore's grouped vocabulary

- Every era's distinctive vocabulary now collapses through the same audited
  word-family map and spelling normalization as grouped Explore. In 1850–1868,
  `slavery`, `slave`, and `slaves` therefore form one `SLAVERY` family with 982
  focal-era uses rather than competing for three ranks. `Viet Nam`, `Viet-Nam`,
  and `Vietnam` also normalize together.
- Rank is now the positive informative-prior log-odds effect, capped after a
  25× concentration, multiplied by `1 - exp(-era_uses / 100)`. Evidence
  therefore rises sharply at low counts but is 99.3% saturated by 500 uses;
  the displayed `× other eras` remains the uncapped raw comparison.
- Existing eligibility still requires 50 corpus uses, 10 era speeches, and two
  era presidents. The public JSON contract is now `era-profile-v3`; frozen
  paid annotations were not changed.
- Verification is green: 66 focused tests and the 115-test broader Story suite
  pass; a final 61-test post-build check passes; the canonical build validates
  73 HTML pages and 165 JSON shards; and 146 inline scripts parse across 74
  generated HTML files. At 1280×720, Corpus Footprint and Major Topics both
  measure 234.37px closed and 257.05px with the longer method open, with zero
  page overflow. Nothing was deployed, staged, committed, pushed, or placed on
  a new branch.

## Latest addendum — Footprint and Major Topics match heights again

- Corpus Footprint and Major Topics once again stretch to the same height when
  they share the two-column Era Profile row. The closed pair measures 234.37px
  on each side at the normal preview width.
- Opening “How the distinctive-word ranking works” grows the shared row, so
  both cards remain equal at 244.91px. The disclosure itself still begins
  closed.
- Each distinctive-word card now puts focal-era uses on one line and `N.N×
  other eras` on the next. Rank remains a separate line above both.
- At 390px the footprint card uses the 230px single-column minimum, all terms
  remain unwrapped, and page overflow is zero. The focused contracts and
  canonical 73-HTML/165-JSON build pass. Frozen annotations were not changed;
  nothing was deployed, staged, committed, pushed, or placed on a new branch.

## Latest addendum — Footprint cards now stop at their content

- Distinctive-word cards now contain only the requested three facts: `Rank
  #N`, focal-era uses, and `N.N× other eras`. The visible evidence label,
  evidence-weighted score, concentration wording, and speech count are gone.
  The hidden method continues to explain why the multiplier alone does not set
  rank.
- The Corpus Footprint article no longer inherits a 230px minimum or stretches
  to the adjacent Major Topics card. It aligns to the start of its grid row and
  ends immediately after the closed disclosure. Opening the disclosure grows
  the article and, when necessary, its grid row.
- Browser measurements: at the normal two-column width the card grows from
  199px closed to 245px open; at 390px it grows from 204px to 259px. In both
  cases the terms remain on one line and page overflow is zero.
- The focused card contracts and canonical 73-HTML/165-JSON build pass. Frozen
  annotations were not changed. Nothing was deployed, staged, committed,
  pushed, or placed on a new branch.

## Latest addendum — Corpus Footprint terms shrink; method starts closed

- Distinctive terms now remain on one line and shrink with their actual Corpus
  Footprint card. At the reported 1869–1912 single-column width, `COMMISSION`,
  `SERVICE`, and `DEPARTMENT` all remain intact; at 390px they shrink further
  without wrapping or page overflow.
- The abstract visible log-odds score is removed. Each card now shows its rank,
  explicitly labeled `× concentration`, and an evidence line with focal-era
  uses and speech coverage. Exact other-era uses and president coverage remain
  in the accessible hover/focus description.
- The explanation and eligibility rules now live in a native “How the
  distinctive-word ranking works” disclosure that starts closed. Era Profile
  grid rows use a 230px minimum rather than a fixed height, so at 390px opening
  the disclosure grows the footprint card from 230px to about 268px with no
  clipped content.
- Verification is green: 23 focused tests and the broader 72-test
  Story/profile/site suite pass; the canonical build validates 73 HTML pages
  and 165 JSON shards. Desktop, single-column, and 390px browser QA confirms
  one-line labels, a closed initial disclosure, contained expansion, and zero
  page overflow. Frozen annotations were not changed. Nothing was deployed,
  staged, committed, pushed, or placed on a new branch.

## Latest addendum — Adversary paths link on interaction; word rank is explicit

- Every era's Adversary Network now treats presidents, named opponents, and
  connection lines as pointer- and keyboard-accessible targets. Hover, focus,
  or click keeps the selected connected path at full opacity, enlarges the
  related lines, and dims unrelated nodes and lines.
- Corpus Footprint distinctive terms now size against their actual card using
  container units and wrap safely at narrow widths. At 390px the Story has no
  page overflow and long terms such as `CONSTITUTION` remain fully visible.
- Distinctive-word rank remains the governed informative-prior log-odds z
  score. The cards now print that evidence-weighted score separately from the
  raw rate ratio and explain that the latter does not set rank. In 1850–1868,
  `CONSTITUTION` is rank 1 because its score is 48.5 despite a 6.9× ratio;
  `SLAVERY` is more exclusive at 44.4× but ranks 2 on its lower 37.9 score.
- Verification is green: 2,582 tests pass with the same 70 existing warnings;
  the canonical build validates 73 HTML pages and 165 JSON shards; 146 inline
  scripts parse across 74 generated HTML files; and `git diff --check` passes.
  Desktop and 390px browser QA confirms connected-path emphasis, exact ranking
  copy, contained word cards, zero page overflow, and an empty console. Frozen
  annotations were not changed. Nothing was deployed, staged, committed,
  pushed, or placed on a new branch.

## Latest addendum — Story eras now use one three-part structure

- Every one of the nine chronological Story eras now renders only its title
  block, one concise historical introduction, and the shared Era Profile / Era
  Defined / Adversaries / Era Echoes workspace. Founding now reaches that
  workspace through the same generator path as the other eight eras.
- Legacy chapter-specific charts, quotations, event callouts, source-link
  paragraphs, extra evidence-boundary copy, and appended metric-takeaway prose
  no longer render in `docs/index.html`. Their governed analytical functions
  and data remain intact for testing and reuse. This supersedes older handoff
  statements that chapter-specific charts still follow each workspace.
- Because Story has no Plotly figures left, its figure payload and Plotly
  runtime tag are omitted. `docs/summary.html` is unchanged in structure and
  retains Synthesis, the Extreme Speeches appendix, and its three rendered
  full-record comparisons.
- Verification is green: 2,579 tests pass with 70 existing warnings; the
  canonical build validates 73 HTML pages and 165 JSON shards; 146 inline
  scripts parse across 74 generated HTML files; and `git diff --check` passes.
  Desktop and 390px browser QA confirms nine eras, one heading/intro/workspace
  per era, four functional tabs per workspace, no legacy chapter bodies, zero
  page overflow, three intact Summary plots, and an empty console. Nothing was
  deployed, staged, committed, pushed, or placed on a new branch.

## Latest addendum — Summary is now a standalone page

- `docs/summary.html` is now generated directly from the source pipeline. It
  contains Synthesis and the Extreme Speeches appendix; `docs/index.html`
  contains only the nine-era chronological story and ends with a visible link
  to Summary.
- The global Summary destination is `summary.html`, with its own current-page
  state, document title, introduction, and links to Synthesis and Extreme
  Speeches. Old `index.html#synthesis` and `index.html#records_appendix` URLs
  redirect to the matching anchors on the new page.
- Story output excludes the three Summary chart payloads as well as the visible
  sections. The standalone page preserves the all-president era filter, naming
  crossover, rhetorical-weather map, explanations, receipts, and appendix.
- Verification is green: 2,578 tests pass with 70 existing warnings; the
  canonical build validates 73 HTML pages and 165 JSON shards; 146 inline
  scripts parse across 74 generated HTML files. Desktop and 390px browser QA
  verifies both current navigation states, the legacy redirect, Story-to-Summary
  navigation, the era filter, contained responsive layout, zero page overflow,
  and an empty console. Nothing was deployed, staged, committed, pushed, or
  placed on a new branch.

## Latest addendum — Full-record comparison charts moved to Summary

- Summary now contains the all-president legal/procedural-versus-hype scatter,
  the 1880–1932 “United States” versus “America/American(s)” crossover, and the
  reviewed-era hype-versus-doom map in that order.
- The first two charts were removed from the 1869–1912 and 1913–1932
  chronology chapters, respectively. Their era-highlight controls, exact
  values, explanations, evidence receipts, and interactions move with them;
  the hype-versus-doom map was already in Summary and remains there once.
- Regression tests pin the three-chart Summary composition and prevent the two
  moved charts from returning to their former chapters. The full suite passes
  2,578 tests with 70 existing warnings; the canonical build validates 72 HTML
  pages and 165 JSON shards; 143 inline scripts parse. Desktop and 390px browser
  QA verifies the order, one-copy placement, era-filter interaction, contained
  horizontal chart scrolling, zero page overflow, and an empty console. Nothing
  was deployed, staged, committed, pushed, or placed on a new branch.

## Latest addendum — Every era now has the same four direct graph views

- Every chronological workspace now exposes four peer tabs directly: Era
  Profile, Era Defined, Adversaries, and Era Echoes. The former nested
  Visualize/Contextualize rails and the on-page Presidential Agendas view are
  removed; agenda data and its standalone renderer remain governed for
  compatibility.
- Era Defined now uses the Founding's combined four-line, nine-era trajectory
  grammar everywhere. Founding and the Continental Republic keep their
  historically declared topic families. The other seven eras use the first
  four Major Topics selected by their Era Profile. Each chart highlights its
  focal era and uses a zero-based ceiling at the next five-point mark above the
  largest observed value.
- Era Echoes now has a direct `Invoked by` / `Invoking` switch. The incoming
  view shows later presidents referring to focal-era presidents; the outgoing
  view shows focal-era presidents referring to earlier presidents. Each
  direction has a separately derived gravity network, so node size and
  placement reflect only the selected evidence. Founding's impossible outgoing
  direction and the present era's impossible incoming direction are disabled.
- The contextualization publication contract is
  `era-contextualizations-v10`. Frozen paid annotations were not changed.
  Existing era-specific chapter graphs still follow each shared workspace.
- Verification is green: 2,576 tests pass with 70 existing warnings; the
  canonical build validates 72 HTML pages and 165 JSON shards; Python
  compilation, scoped `git diff --check`, and static parsing of 143 inline
  scripts across 73 generated HTML files pass. Desktop and 390px browser QA
  verified all nine four-tab workspaces, four lines and 36 points per Era
  Defined chart, direct hashes, the 129-to-247 paragraph Cold War Echoes
  switch, disabled impossible edge-era directions, and responsive
  containment. The only console entry was Plotly's existing auto-margin
  warning; there were no JavaScript errors. Nothing was deployed, staged,
  committed, pushed, or placed on a new branch.

## Latest addendum — Global navigation restructured; public Network Atlas retired

- `restructure-global-navigation-and-retire-network-atlas-page` is complete on
  `master`. The primary order is now Story, Summary, Compare, Explore,
  Profiles, and Data. Profiles contains Presidents and Issues; Data contains
  Data Quality, Methods, and Era Choices. Methods remains current on
  `metrics.html` and `label-models.html`.
- At that stage, Summary remained a live `index.html#synthesis` deep link and
  no `summary.html` was created. The standalone-page addendum above supersedes
  that earlier navigation behavior.
- Profiles and Data are native-button disclosures with ordinary child anchors,
  `aria-expanded`/`aria-controls`, fine-pointer hover, click/touch,
  explicit Enter/Space toggling, Escape with focus return, normal child tab
  order, outside-click close, one-open-at-a-time behavior, visible focus, and
  responsive 390px placement. Feedback is absent from primary navigation and
  present in every generated footer.
- The public Network Atlas renderer and page-specific test are removed.
  `expansion_site.write()` explicitly prunes stale `docs/networks.html`, the
  validator rejects both the legacy output and links, and no generated page
  contains the route. `networks.py`, `data/networks/`, invocation evidence,
  copied network data, metrics, embedded era networks, and analytical tests are
  preserved.
- Verification is green: 39 focused navigation/site/story/Era Choices/network
  tests pass; the full suite passes 2,576 tests with 70 existing warnings; the
  canonical build validates 72 public HTML pages and 165 JSON shards; Python
  compilation, `git diff --check`, retired-route scans, and static parsing of
  143 inline scripts across 73 generated HTML files pass. Desktop and 390px QA
  covered Story, Washington, Economy & Jobs, Explore, Data Quality, Methods,
  and Era Choices with correct current/group state, depth-correct footer links,
  contained disclosures, zero page overflow, and empty consoles. The in-app
  browser positively exercised mouse click, mobile click, Enter, Escape/focus
  return, outside click, mutual exclusion, direct hashes, scroll, history, and
  reduced motion; its pointer-move API did not expose CSS `:hover`, so the
  fine-pointer hover path is additionally pinned by the structural contract.
- The worktree remains extensively dirty with pre-existing user-owned source,
  data, generated-site, and `.claude` changes. This task changed the shared
  navigation/page generators, the two narrow 390px containment rules found
  during QA, navigation/site/story contracts, README/TASKS/HISTORY/HANDOFF, and
  regenerated `docs/`; no unrelated changes were cleaned or reset. Nothing was
  deployed, staged, committed, pushed, or placed on a new branch. The next safe
  action is owner review of this scoped task delta before any git operation.

## Latest addendum — Era-profile constituency publication is groomed, not unblocked

- `notes/era-profile-constituencies-plan-v1.md` and the groomed
  `publish-era-constituency-categories` backlog entry define the data,
  renderer, refusal, test, and browser-acceptance contracts for all nine Era
  Profile screens.
- The plan deliberately ranks the controlled constituency `group_type`
  categories rather than raw `group_text`. The candidate declares no
  authoritative `normalized_group`; a read-only audit of the sealed composite
  found 31,510 claim objects but 13,952 case-folded literal group strings, with
  11,581 singletons and common pronoun forms. Raw-name ranking would therefore
  publish alias and reference-resolution artifacts.
- Implementation is blocked at two explicit gates. The sealed composite remains
  provisional, unmaterialized, not promoted, and not production-eligible, the
  active constituency projection remains empty, and one composite outcome is
  still `unclear`. Separately, the composite covers the corrected
  35,394-paragraph corpus while the current Era Profile uses the older
  36,229-paragraph denominator. An owner-approved production disposition and a
  single canonical Story input universe must land first.
- Only planning and routing documents changed for this grooming pass. No
  annotation, source generator, generated-site, deployment, promotion,
  materialization, active pointer, frozen paid artifact, staging, commit, or
  push was performed. Reggie Doctor was green after the planning edit with
  zero errors and zero warnings.

## Latest addendum — Two-screen era subnavigation is centered

- The shared nested navigation for both `Visualize the era` and
  `Contextualize the era` now uses two equal columns, matching the two screens
  each workspace actually contains. The 620px desktop strip and full-width
  mobile strip are centered with no reserved third column.
- Verification: desktop and 390px browser measurements show equal tab widths,
  a zero-pixel center offset, and no page overflow. All 2,561 tests pass with
  70 existing warnings; the canonical build validates 73 HTML pages and 165
  JSON shards; Python compilation, `git diff --check`, and static parsing of 71
  inline scripts across 74 generated HTML files pass. Nothing was deployed,
  staged, committed, pushed, or placed on a new branch.

## Latest addendum — ARCV1 provisional corpus sealed and verified

- Owner resolutions `ARCV1-RES026` and `ARCV1-RES027` completed the exact
  provisional campaign after its original conservative ceiling and concurrent
  generated-site baseline blocked completion. RES026 raised only the effective
  input-proxy ceiling to 93,000,000 and granted the seven named terminal
  assignments up to three additional attempts; RES027 recorded the final
  348-file docs-only protected-state overlay without modifying `docs/`.
- All 9,186 assignments / 35,154 fresh subjects / 246,078 fresh fields are
  accepted. The append-only control plane retains 1,047 invalid attempts,
  10,233 total attempts, 10,291 runtime sessions, and 91,043,250 reserved
  proxy tokens. No response was locally repaired.
- Full audit passed with zero missing or duplicate field keys and verified the
  locked 81,663,911-token partition. Sealed artifact-set digest:
  `sha256:d0bd31e73e652faa4a9e5d3c2bb89ee2c30de28b99d411ff460b10441d38d484`.
- Non-promoted composite
  `pcomp_a8580fb16926dbe523cd6a9e1ceed165b99f2f0f5828f031b49c8e6f4bc5d02b`
  independently verifies at 35,394 total subjects / 247,758 fields, combining
  the fresh layer with the locked 240-subject evaluated reuse layer. Composite
  artifact-set digest:
  `sha256:d88e08440b06d212ddea117ced1b820d69885f6b0d7e4c3b8c639bc87578f10d`.
- The composite remains provisional and non-production-eligible. No promotion,
  production materialization, active-pointer change, site generation,
  deployment, or frozen `data/llm_annotations/` modification occurred.

## Latest addendum — Agendas now read as individual presidential profiles

- Presidential Agendas now use individual vertical cards in all nine eras.
  Each card contains only that president's five leading policy domains and
  keeps the rank, stable-color bar, exact paragraph-presence percentage, and
  taxonomy definition together. `Remaining policy` and `Non-policy` no longer
  appear in the graphic or exact fallback.
- Bars normalize to the leading value inside each president's card, making the
  internal agenda shape legible without organizing the display around
  cross-president comparison. Exact percentages remain printed and are the
  comparable values across cards. Multi-label paragraphs still count once in
  every applicable domain and can make the five shares sum above 100%.
- The all-12-policy-domain paragraph/raw-word data, unassigned coverage
  receipt, and weighting-sensitivity values remain intact in the JSON contract,
  but the former on-screen text-alternative table has been removed. Measure,
  selection, and weighting prose starts collapsed; each visible taxonomy
  definition is clamped to one line while its full wording remains in the
  priority's accessible label and reveals on hover, keyboard focus, or click.
  The Measure/Evidence pair now sits eight pixels below the collapsed
  explanation instead of being bottom-anchored across the remaining panel
  height. The active Agenda screen no longer inherits the taller Era Profile
  minimum height, so its enclosing box ends directly after the receipts and
  grows naturally with the explanation. Eras with more than three supported
  presidents keep the cards in one horizontally scrollable row, with three
  visible at desktop width and one at mobile width. The fewer-than-five-speech
  disclosure remains, and thin presidents use the same card design but stay
  separate from supported records.
- The public contract is now `era-visualizations-v8`. Frozen paid annotation
  artifacts were not changed. The visualization guidance led to self-contained
  vertical agenda profiles, local rather than shared bar scales, direct labels,
  concise taxonomy descriptions beside their marks, collapsed methodology,
  three-column desktop reflow, and one-column mobile reflow.
- Verification: all 2,561 tests pass with 70 existing warnings; the canonical
  inline build validates 73 HTML pages and 165 JSON shards; Python compilation,
  `git diff --check`, and static parsing of 71 inline scripts across 74
  generated HTML files pass. Current desktop and 390px browser QA covers all
  five Founding priorities, including Washington's data-derived `#5 War &
  military`, with zero page-level overflow and an empty console. The preceding
  card-layout QA covers the dense eight-president 1869–1912 era and
  Harrison/Taylor thin support. Nothing was deployed, staged, committed,
  pushed, or placed on a new branch.

## Latest addendum — Agendas compare presidents on the same rows

- Presidential Agendas now use one compact comparison matrix in all nine eras.
  Each era's named policy rows are the union of the supported presidents'
  individual top three domains, ordered by their mean paragraph presence
  across those presidents. Every supported president receives a cell on every
  named row, so a category such as Native affairs remains directly comparable
  even when it is not one president's own top three.
- Solid bars and `#1`–`#3` receipts mark each president's individual top three;
  pale bars retain the other shared-row values. `Remaining policy` is now the
  same paragraph-level union outside the shared rows for every president, and
  `Non-policy` remains a separate union. All cells print exact percentages.
  Bar length uses the largest displayed supported value in that era without
  printing an axis, so the chart uses its available space without changing the
  underlying measure.
- Unassigned paragraphs remain separate coverage receipts in the president
  headers and exact table. The all-12-domain paragraph/raw-word fallback and
  weighting-sensitivity receipts remain intact. Presidents with fewer than five
  speeches use the same shared rows inside a Thin support disclosure but do
  not choose the headline rows or scale.
- The public contract is now `era-visualizations-v5`. Frozen paid annotation
  artifacts were not changed. The visualization guidance led to a
  row-by-president matrix with direct labels, non-color top-three ranks, sticky
  row labels on mobile, and locally contained horizontal scrolling.
- Verification: all 2,561 tests pass with 70 existing warnings; the canonical
  inline build validates 73 HTML pages and 165 JSON shards; Python compilation,
  `git diff --check`, and static parsing of 71 inline scripts across 74
  generated HTML files pass. Desktop and 390px browser QA covers Founding,
  the dense eight-president 1869–1912 matrix, the Harrison/Taylor thin-support
  disclosure, exact values, zero page-level overflow, and an empty console.
  Nothing was deployed, staged, committed, pushed, or placed on a new branch.

## Latest addendum — Agendas measure presence; Founding shows the full path

- Presidential Agendas now use one shared paragraph-presence design in all nine
  eras. Every president displays the three policy domains appearing in the
  largest shares of that president's paragraphs, the paragraph-level union of
  all remaining policy domains, and the union of all non-policy domains.
  Multi-label paragraphs receive full credit in every applicable lane, so the
  five independent values may exceed 100% in total. `Unassigned` remains a
  separate coverage receipt rather than being folded into non-policy.
- The exact agenda fallback publishes all 12 stable-color policy domains plus
  Remaining policy, Non-policy, and Unassigned. Each cell pairs the headline
  paragraph share with raw-word presence; each president also states whether
  raw-word weighting preserves or reorders the top three. Presidents with
  fewer than five speeches remain in a Thin support disclosure. Long-run
  effective-topic breadth was removed from this graph and preserved as a
  synthesis-page backlog item.
- Founding Era Defined now uses exactly the four approved topic families:
  exercising independence abroad, navigating Native nations and continental
  power, holding the union together, and financing government. One combined
  four-line graph now plots all nine eras on a zero-based 0–35% axis; 35% is
  the next five-point mark above the observed 31.5% maximum. The compact key
  now contains only the four family names, line hover or point focus isolates
  one trajectory, point hover/focus reveals its percentage, endpoint labels omit
  redundant percentages, the measure note starts collapsed, and the redundant
  exact table is removed. The active Era Defined screen is content-sized, so
  Measure/Evidence follows the collapsed note by eight pixels instead of being
  bottom-anchored below a blank band. The coverage line still distinguishes
  deduplicated Founding coverage from the summed overlapping families. No
  unvalidated reframing claim was promoted.
- Public contracts are now `era-visualizations-v4` and
  `era-contextualizations-v9`. Frozen paid annotation artifacts were not
  changed. The two completed redesign tasks are recorded in `TASKS.md`; the
  long-run breadth/depth synthesis view remains explicitly ungroomed.
- Verification: all 2,561 tests pass with 70 existing warnings; the canonical
  build validates 73 HTML pages and 165 JSON shards; Python compilation,
  `git diff --check`, and static parsing of 71 inline scripts across 74
  generated HTML files pass. Desktop and 390px browser QA covers the Founding
  agenda, the compact combined four-line trajectory graph and its interaction
  target, the dense 1869–1912 agenda, weighting reorder receipts, zero
  page-level overflow, and an empty console.
  Reggie Doctor reports zero errors, zero warnings, and five informational findings.
  Nothing was deployed, staged, committed, pushed, or placed on a new branch.

## Latest addendum — Era workspaces now focus on two graphs

- Every era's `Visualize the Era` workspace now displays exactly two graphs:
  Presidential Agendas and Adversary Network. Governing Channels is no longer
  rendered, although its existing communication-route summary remains in the
  visualization contract for compatibility and the separate synthesis still
  uses the corpus-wide communication evidence.
- Every era's `Contextualize the Era` workspace now displays exactly two
  screens: Era Defined and Era Echoes. The standalone Topic Life screen is
  removed. Its governed trajectories and annotation history remain published
  as input to the planned Founding Era Defined redesign; no frozen paid
  artifact was changed or discarded.
- Era Echoes currently answers **who invoked whom, in a paragraph assigned to
  which topic**. It retains every exact
  `speaker → assigned paragraph topic → invoked former president` path and
  excludes self-invocations. The source evidence also contains invocation
  `function` and `stance`, but the gravity graph does not encode either field,
  so it does not yet answer how or why an invocation functioned.
- `TASKS.md` now preserves separate review work for the shared Presidential
  Agendas encoding, a Topic-Life-informed Founding Era Defined replacement,
  and a possible function/stance layer for Era Echoes.
- Verification: 32 focused tests, 68 broader story/site tests, and all 2,560
  repository tests pass with 70 existing warnings; the inline canonical build
  validates 73 HTML pages and 165 JSON shards; both generated story files'
  JavaScript parses; and `git diff --check` passes. Desktop and 390px browser
  QA confirms both two-tab contracts, direct tab interaction, a second era's
  shared structure, zero page overflow, and an empty console. Reggie Doctor
  reports zero errors, zero warnings, and five informational findings. Nothing
  was deployed, staged, committed, pushed, or placed on a new branch.

## Latest addendum — Agendas now compose; diplomacy now branches

- The 1809–1849 story era is now **The Continental Republic**, subtitled
  **Diplomacy, state-building, and continental conquest**. Its authored
  `Era Defined` screen follows four historically judged phases (1809–16,
  1817–28, 1829–40, 1841–49) with four independent, overlapping step lines on
  one shared 0–50% axis. Treaties/diplomacy are the thick continuous trunk;
  war, federal institutions, and territorial power rise and fall around it.
  Territory publishes external acquisition and Native removal/settlement
  separately in the exact fallback; four treaty markers sit on the trunk and
  remain explicitly noncausal. Sixteen keyboard-focusable phase nodes and a
  complete HTML table preserve the values without hover, JavaScript, or color.
- Story ownership now follows presidential transitions at accession
  boundaries without changing raw dates. Jefferson's three-paragraph April
  1809 message belongs to the founding regime while Madison opens the new era.
  The governed rule also covers Johnson/Grant, Taft/Wilson, Hoover/FDR,
  Truman/Eisenhower, Carter/Reagan, and Obama/Trump; 1850 receives no override
  because it is a historical rather than accession boundary. The Era Choices
  page explains the rule and the current absence of a Johnson 1869 speech.
- Presidential Agendas now use the same word-weighted 100% composition in
  every era. A paragraph's words are split equally among its distinct Level-1
  policy domains; `Other policy`, `Non-policy speech`, and `Unassigned` expose
  the complete denominator. A separate effective-topic marker preserves the
  corpus-level story that policy breadth increases over time. Presidents with
  fewer than five speeches stay available in a `Thin support` disclosure but
  cannot choose the displayed policy domains or set the breadth scale. In the
  Continental Republic this keeps Madison through Polk in the main display
  while retaining Harrison (1 speech, 57 paragraphs) and Taylor (2 speeches,
  62 paragraphs) as supporting evidence.
- The public contracts are now `era-profile-v2`,
  `era-visualizations-v3`, and `era-contextualizations-v7`. The ungroomed
  backlog records a later compact story chart explaining which era choices
  were mathematical optima and which were historically reviewed tie-breaks.
  Frozen paid annotation artifacts were not modified.
- Verification: all 2,560 tests pass with 70 existing warnings; the canonical
  build validates 73 HTML pages and 165 JSON shards; Python compilation,
  `git diff --check`, and static parsing of 71 inline scripts across 74
  generated HTML files pass. Desktop and 375px browser QA confirms that all
  nine eras use the shared 100% agenda, the diplomatic tree keeps its endpoint
  labels inside the SVG, local chart scrolling causes no page overflow, all 20
  phase/event marks have keyboard-readable labels, the exact table remains
  available, and the console is empty. Reggie Doctor reports zero errors, zero
  warnings, and five informational findings. Nothing was deployed, staged,
  committed, pushed, or placed on a new branch.

## Latest addendum — Era choices now use persistent four-year regimes

- `era-boundaries-v2` replaces isolated eight-year-score rhetoric with the
  reviewed data-first method. It combines every-year four- and eight-year
  percentile ranks for abruptness, then uses dynamic programming to divide
  presidential-cycle four-year fingerprints into exactly nine contiguous
  segments of at least three units while minimizing within-era squared
  distance. The complete optimum starts at 1789, 1805, 1849, 1869, 1913,
  1937, 1953, 1981, and 2017.
- The reviewed page scheme is now 1789–1808, 1809–1849, 1850–1868,
  1869–1912, 1913–1932, 1933–1952, 1953–1980, 1981–2016, and 2017–2026.
  Grant's March 4, 1869 accession replaces the former 1878 start; the page
  explains that Reconstruction continues historically but does not form its
  own persistent presidential-speech regime. Madison's March 4, 1809
  accession resolves the near-tied 1805–09 zone, and FDR's 1933 accession
  resolves the 1933–37 zone. The reviewed partition is 0.512% above the full
  mathematical optimum. Live chronological story ranges remain unchanged
  pending explicit migration approval.
- The page now separates combined abruptness ranks from persistent-regime
  selection, publishes full and seven axis-removal nine-era reruns, explains
  why the old eight-year grid labeled the Grant transition 1872, and exposes
  the four-year unit fingerprints and consensus/segmentation CSVs. Frozen paid
  annotations were not modified.
- Verification: all 2,555 tests pass; the canonical build validates 73 HTML
  pages and 165 JSON shards; the generated era-page script and Python compile;
  Reggie Doctor reports zero errors and zero warnings. Automated localhost
  refresh was blocked by the browser safety layer, so the open page requires a
  manual refresh for visual review. Nothing was deployed, staged, or committed.

## Latest addendum — Era boundaries are now reviewable before migration

- `docs/era-boundaries.html` is a new primary-navigation methods page explaining
  the recommended 1789–1808, 1809–1849, 1850–1877, 1878–1912, 1913–1932,
  1933–1952, 1953–1980, 1981–2016, and 2017–2026 chapter scheme. The live
  chronological chapters remain on their prior ranges until the recommendation
  is reviewed and explicitly approved for migration.
- `src/presidential_profiles/era_boundaries.py` produces the governed
  `era-boundaries-v1` layer: a transition score for all 223 possible calendar
  boundaries using the same 63-axis fingerprints in symmetric eight-year
  windows, plus k=2–12 contiguous Ward reruns with every axis group omitted in
  turn. The page distinguishes historical choices from descriptive distances,
  eight-year bin starts, and word-weighted presidency center years.
- The page directly explains why 1848 was a mechanically available bin start
  while the every-year score favors 1850, and why the presidency-grain 1983
  center-year result should not be described as a 1989 break. Historical
  receipts and full non-hover tables accompany both Plotly graphs. Frozen paid
  annotations were not modified.
- Verification: 145 focused story/atlas/site tests and all 2,553 repository
  tests pass; the canonical build validates 73 HTML pages and 165 JSON shards;
  71 generated inline scripts and Python compile; Reggie Doctor reports zero
  errors and zero warnings. Desktop and 390px browser QA covers the robustness
  selector, hover receipts, local chart scrolling, zero page overflow, and an
  empty console. Nothing was deployed, staged, or committed.

## Latest addendum — Every era now opens with one shared workspace

- All nine chronological chapters now begin with the same tabbed box: Era
  Profile, Visualize the Era, and Contextualize the Era. The Founding renderer
  is now the reusable workspace renderer rather than a one-off. Every later
  era retains its existing chapter-specific graphs and evidence immediately
  below the new workspace.
- Contextualize remains a three-screen contract for every era: Era Defined,
  Topic Life, and Era Echoes. Later Era Defined screens remain explicit
  pending slots until their custom evidence-backed graphs are authored; Topic
  Life and Era Echoes are populated from the shared all-era formulas.
- Topic Life editorial provenance now lives in
  `data/contextualization/topic_life_annotations_v1.json`. It is an append-only
  event history keyed by era/topic: UTC timestamp selects the canonical entry,
  prior entries remain published, annotators are human or AI, and AI entries
  require model and reasoning-effort metadata. The consolidated contract is
  now `era-contextualizations-v5`.
- Frozen paid annotations were not modified, and no existing era-specific
  story graph was removed.
- Verification: 22 focused and 94 broader story/site/network tests pass; the
  canonical build validates 72 HTML pages and 164 JSON shards; generated
  JavaScript, Python, the v5 JSON contract, and targeted whitespace checks
  pass. Browser QA confirms the Expansion and Present workspaces, direct
  outer- and inner-tab hashes, six Topic Life rows, populated Era Echoes, and
  preserved Expansion/platform chapter graphs below the workspaces. Reggie
  Doctor reports zero errors and zero warnings. Nothing was deployed, staged,
  or committed.

## Latest addendum — Era Echoes is now a shared gravitational field

- The reusable Contextualize contract is now
  `era-contextualizations-v5`. Every era publishes its presidents as
  chronological gravity anchors plus complete external-president and topic
  satellites. Satellite weights come from exact distinct reference-paragraph
  connections; all uncollapsed speaker–topic–invoked-president paths remain
  published.
- Founding renders Washington, Adams, Jefferson, and Madison on one shared
  field. Planet area reflects 100, 18, 126, and 19 connected reference
  paragraphs respectively. All 32 later presidents and all 40 assigned topics
  are present; weighted centroids carry the meaning and deterministic
  collision spacing supplies only the offset needed to prevent overlap.
- Pointer, click, and keyboard focus reveal each node's exact anchor shares,
  highlight its connection lines, and retain a readable detail strip. The
  default graph has restrained links and only labels the leading satellites.
  Topic and invoking-president focus now work symmetrically: either endpoint
  keeps every exact counterpart at full emphasis and draws temporary dotted
  topic-to-president spokes. Founder-to-topic gravity links are also dotted,
  while founder-to-president gravity links remain solid. In the Founding graph,
  Faith & national ideals reveals 17 invoking presidents, and Ronald Reagan
  reveals 12 assigned topics. The same renderer handles all nine eras,
  including eras with incoming and outgoing references.
- Era Defined and Topic Life were not changed. Frozen paid annotations were
  not modified.
- Verification: 92 broader story/site/network tests pass; the
  canonical build validates 72 HTML pages and 164 JSON shards; generated
  JavaScript, Python syntax, the v4 JSON contract, and targeted whitespace
  checks pass. Browser QA confirms 12 dotted Ronald Reagan-to-topic spokes plus
  37 dotted Thomas Jefferson-to-topic and 29 solid
  Thomas Jefferson-to-president gravity links; zero plotted-node collisions
  and zero page overflow at 390px remain covered. Nothing was deployed, staged,
  or committed.

## Latest addendum — Contextualize now foregrounds level and cluster structure

- Era Defined uses one uninterrupted topic-colored bar for the Founding
  aggregate, retains the historical-average benchmark line, and promotes the
  percentage-point difference as the large right-edge statistic. The smaller
  cards below preserve both exact shares. A new deduplicated coverage receipt
  shows that 680 of 927 Founding paragraphs (73.4%) contain at least one of the
  four topic families; their overlapping shares sum to 80.9%. The equal-era
  historical comparisons are 39.6% deduplicated and 41.5% summed.
- Topic Life puts all six paths on the same 0–20% vertical scale and prints the
  focal-to-present percentages beside each semantic trend. Military
  preparedness therefore reads as 16.2% → 8.6%, while Native affairs reads as
  15.0% → effectively zero, rather than letting row-local scaling make those
  trajectories look comparable. Fades, Persists, Returns, Grows, and Reframes
  retain distinct, restrained colors.
  Optional reframing claims now pass a declared gate before rendering:
  Spearman rank correlation at or below -0.50, source decline of at least
  3 percentage points, and successor growth of at least 1 percentage point.
  The finance successor passes; the early-naval successor fails and is recorded
  as rejected, so Naval Wars now renders as Fades. The AI/unvalidated status is
  stated in the chart note, and the emphasized finance branch switches from
  Debt/Revenue/Treasury to Taxes/Deficits/Spending in War & New Deal, where the
  proposed successor first exceeds the named source.
- Era Echoes is one occurrence-weighted graph containing four translucent
  invoked-president clusters. Jefferson, Washington, Madison, and Adams are
  hubs; each shows its three strongest speaker and topic connections. Founding
  still covers 233 reference paragraphs, 297 mentions, and 131 speeches; the
  complete links and uncollapsed paths remain published.
- The generated chart scrollers and collapsed Measure/Evidence receipts are
  explicitly contained on narrow screens. Frozen paid annotations remain
  untouched.
- Verification: 19 focused and 91 broader story/site/network tests pass; the
  canonical build validates 72 HTML pages and 164 JSON shards; generated
  JavaScript, Python syntax, JSON contracts, and targeted whitespace checks
  pass. The current browser tab must be manually refreshed because the browser
  safety layer blocks programmatic localhost reloads. Nothing was deployed,
  staged, or committed.

## Latest addendum — Contextualize comparisons now match their visual grammar

- The reusable contract is now `era-contextualizations-v3`. Era Defined
  remains ordered by Founding-minus-history difference, but each bar now
  encodes that same quantity: the common share is gray and only the
  percentage-point gap is colored. Endpoint dots and exact values retain the
  aggregate and equal-era benchmark.
- Topic Life no longer mistakes a source-plus-successor union for a reframing
  trajectory. Both unvalidated hypotheses now use the same visual grammar:
  filled points are the named source; outlined points and a dotted line are
  the proposed successor itself. Taxes/deficits/spending begins at 0.3% in the
  Founding rather than inheriting debt/Treasury's 9.4%. Early naval wars now
  has its own outlined broader war-and-military successor. Rows use local-peak
  scaling, with exact paragraph shares on hover/focus.
- Era Echoes now renders the requested three-stage evidence path:
  president speaking → assigned paragraph topic → president invoked. For the
  Founding this covers 233 exact reference paragraphs, 297 mentions, 131
  speeches, and 232 uncollapsed president-topic-president combinations.
  Visible overflow is grouped; every exact path remains in the published JSON.
- The separate reframing-hypothesis input remains fingerprinted and
  unvalidated. Frozen paid annotation artifacts were not changed.
- Verification: 113 relevant story/site/network tests pass; the canonical
  build validates 72 HTML pages and 164 JSON shards; generated JavaScript,
  Python syntax, and targeted whitespace checks pass. Desktop browser QA found
  zero graph-node collisions, zero page overflow, and no console errors.
  Reggie Doctor reports zero errors and zero warnings. Nothing was deployed,
  staged, or committed.

## Latest addendum — Contextualize now separates measures from interpretations

- The reusable Contextualize contract is now
  `era-contextualizations-v2`. Founding Era Defined uses the unweighted mean of
  the other eight era shares as `Historical average`, so the Cold War cannot
  dominate the benchmark through corpus size. Every lane publishes and renders
  Founding aggregate, historical average, percentage-point difference, and the
  four-president mean/receipts. Exact pairs are 35.3% vs 11.3% abroad, 15.0%
  vs 1.8% Native nations/continental power, 16.4% vs 12.0% union, and 14.2%
  vs 16.5% finance.
- Topic Life now plots the literal named topic and computes only descriptive
  `Persists`, `Fades`, `Declines`, and `Persists + returns` path labels. The
  separate fingerprinted input
  `data/contextualization/topic_life_annotations_v1.json` holds two
  explicitly `unvalidated` editorial AI hypotheses: early naval wars reframe
  as broader war and military affairs, and debt/revenue/Treasury reframes as
  taxes, deficits, and spending. The latter keeps Debt/Revenue/Treasury as the
  filled trajectory and adds a keyboard-focusable, hoverable outlined point
  for the combined Public Finance union at every era.
- Era Echoes is now a directional topic network rather than an era-arc chart.
  Founding sits on the left and connects to the eight leading assigned topics
  in the exact 233 distinct later paragraphs that explicitly name a Founding
  president (297 mentions, 131 speeches). Faith/National Ideals leads at
  55/233 (23.6%). Separate actor receipts name the Founding figures invoked and
  the presidents looking back. Middle eras use the focal era in the center;
  the present era sits on the right.
- `TASKS.md` now carries the ungroomed human-validation task for the reframing
  hypotheses. The frozen paid annotations were not modified.
- Verification: 113 relevant tests pass; the canonical build validates 72
  HTML pages and 164 JSON shards; generated JavaScript and Python syntax pass.
  Browser QA covers all three Founding screens, a bidirectional later-era Echo
  network, desktop fit, contained mobile chart scrolling, keyboard-focusable
  family outlines, and an error-free console. Nothing was deployed, staged,
  or committed.

## Latest addendum — Contextualize is now a recurring three-screen system

- `src/presidential_profiles/era_contextualizations.py` is the reusable
  contract for Era Defined, Topic Life, and Era Echoes. It exposes the same
  all-era/single-era loader pattern as the profile and visualization modules
  and publishes `docs/data/era_contextualizations.json` under
  `era-contextualizations-v1`.
- Founding's authored Era Defined graph is `Getting the Republic Up and
  Running`: four horizontal topic-union lanes compare 1789–1815 with all other
  eras on the same scale. The exact pairs are independence abroad 35.3% vs
  10.3%, Native nations/continental power 15.0% vs 1.6%, union 16.4% vs 11.3%,
  and finance 14.2% vs 16.2%. The UI explicitly says overlapping labels make
  the four shares non-additive. The former next-era question is removed.
- Topic Life follows each era profile's six Major Topics across all nine eras
  on one chart scale, with a focal-era band, exact focus labels, and
  deterministic `Fades` / `Persists` / `Returns` / `Peaks later` / `Reframes`
  cues. Era Echoes uses the retained named-presidential invocation evidence:
  edge width is source-era-normalized distinct reference-paragraph share, with
  raw mentions and speeches in the detail. Its page copy states that broader
  allusions such as “the Founders,” the Constitution, Reconstruction, the New
  Deal, or Pearl Harbor are not covered by this layer.
- Contextualize now uses the approved quiet secondary rail and the same
  Era-Profile-height stage behavior as Visualize. Every later chapter receives
  the same three-screen template with unique IDs. Topic Life and Era Echoes are
  populated; later Era Defined slots are deliberately explicit pending states
  because that recurring role requires an era-specific claim and graph rather
  than a generic substitute. Legacy Founding screen hashes redirect to the new
  screens after a clean load.
- Verification: 206 relevant story/site/network tests pass; the canonical
  build validates 72 HTML pages and 164 JSON shards; generated JavaScript and
  Python syntax pass; targeted whitespace checks are clean. Browser QA covered
  all three Founding screens, a later-era instance, legacy hashes, height and
  overflow, and found no console errors. Reggie Doctor reports zero errors and
  zero warnings. The local preview remains at
  `http://localhost:8010/`; nothing was deployed, staged, or committed.
- The extensively dirty worktree remains user-owned. No unrelated source,
  generated output, data, or native Claude/Reggie files were cleaned, reset,
  or overwritten.

## Latest addendum — Visualize is now one nine-era graph system

- `src/presidential_profiles/era_visualizations.py` is the reusable input
  contract for Presidential Agendas, Governing Channels, and Adversary Network.
  `load_era_visualization("expansion")` selects one stable era key or canonical
  label; `load_all_era_visualizations()` returns all nine canonical eras from
  exact-key corpus/annotation joins.
- `site.py` now has one renderer per graph and one shared three-screen template.
  Founding retains its approved outer navigation; every later chapter receives
  the same Visualize workspace with era-local presidents, topics, channels,
  adversaries, denominators, evidence text, and unique interaction IDs.
- The build publishes `docs/data/era_visualizations.json` under the
  `era-visualizations-v1` schema. Founding’s approved eight topic rows, 0–40%
  scale, channel counts, and adversary network values are preserved by
  regression tests.
- Dense eras adapt rather than fork: seven-to-nine-president matrices use
  compact stacked portrait headers and fit the supplied desktop viewport;
  channel selectors wrap, peer pins use four deterministic lanes, and adversary
  networks cap at a readable 620px drawing height. Browser QA covered Founding,
  Expansion’s nine-president agenda/channels and 27-node adversary network, and
  a programmatic nine-era layout audit. A clean reload added no console errors.
- Verification: 161 relevant story/site tests pass; the canonical build
  validates 72 HTML pages and 163 JSON shards; generated JavaScript and Python
  syntax pass; targeted `git diff --check` is clean; Reggie Doctor reports
  zero errors and zero warnings. The monolithic 2,538-test command was
  terminated by the environment during the pre-existing annotation suites at
  8%, before reaching this site work, so no full-suite claim is made.
- The local preview remains at `http://localhost:8010/`; nothing was deployed,
  staged, or committed. The extensively dirty worktree remains user-owned.

## Latest addendum — Visualize subnavigation becomes secondary

- The Presidential Agendas / Governing Channels / Adversary Network control no
  longer repeats the full-width segmented treatment of the primary Era Profile
  / Visualize / Contextualize route. It is now a centered 620px secondary rail
  with transparent framing, smaller 650-weight type, a faint baseline, and a
  restrained active underline and tint.
- The inner rail shrank from 1,110px by 44px to 620px by 39px at the supplied
  desktop viewport. Its bottom margin is 9px, leaving more visual ownership to
  the graph and its question-led header.
- At narrow widths the same hierarchy remains a three-column rail rather than
  becoming three stacked, boxy rows. Labels may wrap locally without creating
  page-level overflow.
- Browser QA exercised the Governing Channels switch, confirmed the selected
  tab and visible panel remain synchronized with
  `#founding-screen-governance`, and measured zero horizontal page overflow.
  Verification is green: 49 focused contracts and all 2,532 repository tests
  pass; canonical and self-contained builds validate 72 HTML pages and 162
  JSON shards; both generated JavaScript bundles and Python syntax pass.
- The local preview remains at `http://localhost:8010/`; nothing was deployed,
  staged, or committed. The extensively dirty worktree remains user-owned.

## Latest addendum — Governing channels compact interaction pass

- Every muted governing-channel portrait pin now exposes an immediate custom
  hover label with the full president name, channel, exact speech share, and
  speech count. A native title remains as a fallback, while the focused bar's
  existing accessible comparison label continues to carry all peer values.
- The president controls grew to 145px by 56px with 40px portraits. The large
  empty band beneath them was traced to the global page `section` margin;
  governing-channel distribution sections now clear that margin locally, so
  the measured selector-to-graph gap is 8px without changing other sections.
- Closed Measure and Evidence receipts are single-line ellipses. Their full
  summary remains available to assistive technology and when the disclosure is
  opened; the existing mutually exclusive expansion behavior is unchanged.
- Browser QA at the supplied desktop viewport confirmed the compact spacing,
  enlarged controls, 96 labeled peer pins, and computed one-line receipt
  truncation. Verification is green: 49 focused contracts and all 2,532
  repository tests pass; canonical and self-contained builds validate 72 HTML
  pages and 162 JSON shards; both generated JavaScript bundles and Python
  syntax pass.
- The local preview remains at `http://localhost:8010/`; nothing was deployed,
  staged, or committed. The extensively dirty worktree remains user-owned.

## Latest addendum — Visualize gains room and identity

- Presidential Agenda rows now render at 49px with a 54px header, making the
  shared-scale values easier to scan at the current desktop viewport.
  Governing-channel rows now render at 54px with 42px tracks.
- Governing-channel peer marks are 20px portrait pins rather than initials.
  Three deterministic vertical lanes keep the four-president Founding view
  legible; additional peers reuse those lanes with a small horizontal stack
  offset. Each focused bar retains the full names, exact shares, and counts in
  its accessible comparison label, while portrait hover titles expose each
  individual peer.
- The former dominant-route card is now a quiet one-line note beneath the two
  distributions. It names the most common exact audience→form intersection in
  prose and retains its exact speech count and share.
- The adversary-network drawing floor increased from 280px to 420px, with
  larger vertical spacing derived from the number of named nodes.
- Each Visualize screen is a full-height flex column. Its Measure/Evidence
  receipt pair is pushed to the absolute bottom of the screen; when one
  disclosure opens, the other is removed from layout and returns when the open
  disclosure closes. Browser QA measured a zero-pixel receipt-to-panel bottom
  gap and exercised both exclusive disclosure directions.
- Verification is green: 49 focused contracts and all 2,532 repository tests
  pass. Canonical and self-contained builds validate 72 HTML pages and 162
  JSON shards; both generated JavaScript bundles and Python syntax pass;
  targeted `git diff --check` is clean.
- The local preview remains at `http://localhost:8010/`; nothing was deployed,
  staged, or committed. The extensively dirty worktree remains user-owned.

## Latest addendum — Visualize fills the workspace

- The three Visualize screen tabs now form one connected, equal-width,
  full-workspace control like the outer Era Profile / Visualize /
  Contextualize route. The active screen receives the same inset copper rule;
  mobile retains a one-column fallback.
- Every Visualize graph now opens with the same full-width editorial header:
  a small screen label, a featured question, and a compact reading or method
  note. Nested screen sections explicitly clear the global page-section margin
  and header width cap, bringing each graph directly beneath its tabs.
- Agenda topic selection now changes both axes: the selected topic moves to
  the first row while president columns rank by its paragraph shares.
  President selection continues to rank all topic rows; clicking an active
  control restores the default row and column order.
- The governing-channel dominant route is now a dark editorial callout with
  separate audience, arrow, and medium lines rather than a bordered metric
  card. Its exact count and share remain visible.
- Peer comparison markers are deliberately unchanged pending a design
  decision. Initials are not collision-safe across later eras; the next pass
  should use a stable identity treatment rather than extending the current
  one-letter marks ad hoc.
- Browser QA exercised topic and president sorts, reset behavior, all three
  equal-width screens, the question headers, and the revised route callout.
  Verification is green: 49 focused contracts and all 2,532 repository tests
  pass; canonical and self-contained builds validate 72 HTML pages and 162
  JSON shards; both generated JavaScript bundles and Python syntax pass.
- The local preview remains at `http://localhost:8010/`; nothing was deployed,
  staged, or committed. The extensively dirty worktree remains user-owned.

## Latest addendum — Visualize is sortable and president-selectable

- The Visualize workspace no longer repeats its own heading. Its three-screen
  selector is centered above the chart, the charts sit closer to it, and the
  former bottom prompt into Contextualize has been removed. The approved
  Adversary Network remains unchanged.
- Presidential Agendas keeps the eight-row, four-president shared-scale matrix.
  Selecting a topic now ranks president columns by paragraph share; selecting
  a president ranks topic rows by that president's share. Selecting the active
  control again restores chronology and the default topic order. Controls are
  pressed-state buttons with a polite status announcement. The icon vocabulary
  is now explicit: globe for Maritime rights and Native affairs; shield for
  Naval wars, Treaties/diplomacy, and Military preparedness; sparkle for
  Providence/Faith/Ideals; capitol for Union/federalism and law enforcement.
- Governing Channels is now one selected-president comparison rather than four
  simultaneous fingerprints. Four portrait buttons select a president by
  pointer or Left/Right/Home/End keys. The active president receives two
  four-bar distributions—audience and primary form—while muted initial markers
  preserve all three peers on the same 0–100% scale. A separate card reports
  the exact dominant audience→form route and speech count.
- Only the active president panel participates in layout; a specific hidden
  rule prevents the other three grid panels from inflating the screen. Desktop
  browser QA exercised both sort directions and reset behavior, pointer and
  keyboard president selection, one visible panel, eight bars, 24 comparison
  markers, and visually hidden live announcements.
- Verification is green: 49 focused Founding/V2/V3/all-era contracts and all
  2,532 repository tests pass. Canonical and self-contained builds validate 72
  HTML pages and 162 JSON shards; both generated JavaScript bundles and Python
  syntax pass; `git diff --check` is clean; Reggie doctor reports zero errors
  and zero warnings.
- The local preview remains at `http://localhost:8010/`; nothing was deployed,
  staged, or committed. The extensively dirty worktree remains user-owned.

## Latest addendum — Visualize becomes two direct comparison grids

- The Visualize heading no longer carries the explanatory subtitle. Its
  Presidential Agendas / Governing Channels / Adversary Network selector now
  sits beside the heading, leaving the chart the rest of the panel width.
- Presidential Agendas is now a four-president matrix with no aggregate Era
  column. Its eight shared rows are Maritime rights, Naval wars, Treaties &
  diplomacy, Native affairs, Military preparedness, Providence/Faith/American
  Ideals, Constitutional union & federalism, and Federal law enforcement.
  Every cell prints its exact paragraph share on one 0–40% scale; the
  row-leading president receives a quiet outline. The deterministic selection
  still preserves every president's leading topic before favoring prevalent,
  high-difference topics.
- Governing Channels is no longer a path network. Four compact presidential
  fingerprints show the same 4×4 audience-by-primary-form vocabulary:
  Congress, public, groups, and other crossed with written, spoken, radio/TV,
  and press/other performed forms. Square area encodes that exact joint route's
  share of the president's corpus speeches, the percentage is printed in every
  nonzero cell, and all 64 cells remain present for comparison across
  presidents and future eras.
- The approved Adversary Network is unchanged. Desktop browser QA measured the
  Era Profile, agenda matrix, and communication fingerprints at the same
  rendered 678px height, with zero page-level horizontal overflow. Nested tabs
  retain pointer and arrow-key operation; exact counts remain available through
  focusable cell labels.
- Verification is green: 49 focused Founding/V2/V3/all-era contracts and all
  2,530 repository tests pass. Canonical and self-contained builds validate 72
  HTML pages and 162 JSON shards; generated JavaScript and Python syntax checks
  pass; `git diff --check` is clean; and Reggie doctor reports zero errors and
  zero warnings.
- The local preview remains at `http://localhost:8010/`; nothing was deployed,
  staged, or committed. The worktree remains extensively dirty and user-owned.

## Latest addendum — Visualize now compares agendas and routes speeches

- Presidential Agendas is now one shared-scale racetrack instead of four
  isolated top-five cards. Eight topic rows use the same 0–40% paragraph-share
  axis; each row shows all four presidents, the range between them, and a
  diamond for the full-era share. The deterministic selection preserves every
  president's leading topic, then favors era-prevalent topics with large
  cross-president differences. Exact counts remain in keyboard-focusable
  point details.
- Governing Channels is now a joint speech-routing map. Every one of the 73
  Founding corpus speeches follows one exact
  `president → assigned audience → assigned primary form` path. Width encodes
  speech count, color retains the president, portraits can be selected by
  pointer or keyboard to isolate a route, and zero-count channel nodes remain
  visible so the same vocabulary can carry into later eras.
- The approved Adversary Network is unchanged. The profile-height synchronizer
  measures the rendered Era Profile and applies it to Visualize; desktop browser
  QA measured both new default screens at 674px against the 674px profile, with
  no page-level horizontal overflow. The existing adversary screen remains
  content-sized at 688px rather than being clipped.
- Verification is green: 49 focused Founding/V2/V3/all-era contracts and all
  2,530 repository tests pass. The local pointer and keyboard route filters,
  nested-tab hashes, 8 topic rows, 32 president markers, exact route totals,
  and empty channel nodes were exercised in-browser.
- The local preview remains at `http://localhost:8010/`; nothing was deployed,
  staged, or committed. The worktree remains extensively dirty and user-owned.

## Latest addendum — the Era Profile is now one nine-era template

- `src/presidential_profiles/era_profiles.py` is the single reusable profile
  data layer. `load_era_profile("expansion")` accepts either a stable key or
  canonical era label; changing that one argument returns the corresponding
  metadata, presidents, constituency state, five named adversaries, corpus
  footprint, six major topics, three distinctive words, and governing-channel
  distributions. `load_all_era_profiles()` returns the ordered nine-era map.
- The canonical site build derives all nine records once and passes each one
  through `_era_profile_html`; the Founding screen is no longer a separate
  profile implementation. Every story chapter now includes the same Era
  Profile composition. The eight later chapters use their own corpus values
  and display an explicit pending constituency state until a promoted
  constituency artifact supplies claims; the approved Founding illustrative
  fallback remains labeled as such.
- `docs/data/era_profiles.json` publishes the same switchable records under the
  `era-profile-v1` schema. Source values come from exact-key-validated corpus,
  paragraph, speech-annotation, entity, taxonomy, and promoted-constituency
  artifacts—not generated HTML or copied constants.
- All-era contracts reconcile the nine disjoint footprints exactly to 1,057
  speeches, 36,229 paragraphs, and 4,179,266 words. Desktop browser QA found
  nine visible profiles, unique IDs, equal internal cards, no clipping or
  horizontal overflow, and zero adversary-bubble collisions.
- Verification is green: all 2,530 repository tests pass; the canonical and
  self-contained builds validate 72 HTML pages and 162 JSON shards; generated
  JavaScript passes syntax checks; `git diff --check` is clean; and the portable
  Reggie doctor reports zero errors and zero warnings.
- The local preview remains at `http://localhost:8010/`; nothing was deployed,
  staged, or committed. The worktree remains extensively dirty and user-owned.

## Latest addendum — corpus footprint shows the literal distinctive-word podium

- The lower Corpus Footprint row now presents the uncurated top three
  informative-prior log-odds terms for 1789–1815 versus every other era:
  `MILITIA` (#1, 82 Founding uses / 179 elsewhere, 16.2× rate), `BRITISH`
  (#2, 114 / 692, 5.8×), and `ENSUING` (#3, 33 / 49, 23.9×).
- Rank is determined by the log-odds z-score, which weighs effect size and
  evidence. The display deliberately does not sort by raw rate multiple; this
  is why lower-count `ENSUING` remains third despite its 23.9× multiple. Exact
  ranks and counts are exposed in each tile's tooltip.
- Eligibility is now enforced before ranking: a word needs at least 50 uses in
  the combined corpus, appearances in at least 10 Founding-era speeches, and
  use by at least 2 Founding presidents. `MILITIA` appears in 31 speeches / 4
  presidents, `BRITISH` in 21 / 3, and `ENSUING` in 23 / 4. The thresholds are
  visible in the card receipt; coverage counts are present in each tooltip.
- The former words-per-sentence and `shall` tiles are no longer rendered.
  Their derived values remain in the data receipt for possible later use.
- Live browser QA confirms the three ranked tiles, no old language tiles, no
  card clipping, and zero page-level horizontal overflow. The 40 focused
  Founding/V2/V3 contracts pass; the canonical build validates 72 HTML pages
  and 161 JSON shards.
- The local preview remains at `http://localhost:8010/`; nothing was deployed,
  staged, or committed. The worktree remains extensively dirty and user-owned.

## Latest addendum — ARCV1 resumed under automatic targeted-literal policy

- The authorized provisional campaign
  `pcamp_ce0ff257543c3a342d96147b9e5e46953ca7016aab9bf29f1c6f1b573100939b`
  resumed under owner-authorized `ARCV1-RES025`. At the
  2026-07-26T02:25Z checkpoint it has 3,619 accepted assignments, 5,563
  available assignments, four claimed/started assignments, 400 preserved
  invalid attempt receipts, and 35,775,185 reserved input-proxy tokens.
- `ARCV1-RES021` added fixed, receipt-hashed constituency guidance only after a
  registered constituency-validator failure and granted assignment 41 one
  additional guided call. `ARCV1-RES023` then added separately hashed literal
  transcript guidance and recovered assignment 8912; 338 accepted responses
  now record a feedback supplement.
- Assignment 8177 exhausted three attempts because every response shortened
  the exact source sentence `It has stood for progress; it has stood for
  concern for the people's welfare.` to its second clause and capitalized that
  clause's leading `it`. A content-addressed `RES025` grant supplied the
  validator-verified full sentence as a literal hint; its fresh fourth attempt
  was accepted without editing any response.
- `RES025` allows exactly one automatic targeted fourth attempt only after
  three recorded constituency exact-substring failures and only when the local
  diagnostic proves a literal target candidate containing the claimed group.
  A failed fourth attempt remains terminal. The locked rendering, exact
  `gpt-5.6-sol`/`high` runtime, 89,830,303 token ceiling, provisional status,
  and all prohibited-effect boundaries remain unchanged.

## Latest addendum — complete adversary-type key and quieter story palette

- Named Adversaries now uses the story's established slate blue, brick, green,
  ochre, and taupe palette, softened inside the bubbles so the chart sits with
  the profile rather than reading as a separate graphic system.
- Its legend is derived from every `type` value in the full frozen entity
  artifact, not only the types present in the five displayed adversaries. It
  shows Nation/state, Person, Group, Institution, and Other. Filled marks are
  represented among the five leading Founding adversaries; hollow marks make
  the three missing categories explicit. Tooltips expose full-corpus mention
  counts. The former `Hollow = not among these five` helper line has been
  removed at the owner's request; no other legend behavior changed.
- This addendum preceded the owner's decision to replace both language-form
  tiles with the literal top-three distinctive-word ranking; see the current
  addendum above.
- Live browser QA at 1280×720 confirms five legend categories, two filled and
  three hollow states, zero bubble collisions, no clipped card content, and
  zero page-level horizontal overflow. The 40 focused Founding/V2/V3 contracts
  pass; the canonical build validates 72 HTML pages and 161 JSON shards.
- The local preview remains at `http://localhost:8010/`; nothing was deployed,
  staged, or committed. The worktree remains extensively dirty and user-owned.

## Latest addendum — founding profile becomes a flat four-card composition

- The Presidents card and visible `Presidents` label are removed. The four
  cached portraits now sit directly below the `1789–1815` / `A New State`
  heading in a centered, wrapping flex strip, without a surrounding box.
- Claimed Constituents, Named Adversaries, Corpus Footprint, and Major Topics
  now form a strict 2×2 desktop grid. Live browser measurement at 1280×720
  confirms all four cards are exactly 551×230px and none clips its contents.
- Named Adversaries is no longer drawn as a literal jar. It is a compact,
  non-overlapping five-bubble pack modeled on the supplied reference: diameter
  increases with normalized adversarial-paragraph count, color encodes the
  artifact's dominant entity type, and the rendered legend resolves to
  `Nation or state` and `Person`. Live center/radius checks find zero
  collisions.
- Corpus Footprint now identifies `MILITIA` as the era's most distinctive
  non-register content word. The existing informative-prior log-odds method
  selects it from the data: 82 uses in 1789–1815 versus 179 in every other era,
  a 16.2× higher content-word rate. The method and counts remain available in
  the visible note and tooltip.
- Live browser QA confirms four portraits, no jar element, the two-category
  legend, zero page-level horizontal overflow, and no clipped cards.
- Verification: the canonical build validates 72 HTML pages and 161 JSON
  shards; generated JavaScript syntax passes for normal and self-contained
  builds; the 40 focused Founding/V2/V3 contracts and all 2,521 repository
  tests pass. The local preview remains at `http://localhost:8010/`; nothing
  was deployed, staged, or committed.
- The worktree remains extensively dirty and user-owned. No unrelated source,
  frozen annotation artifact, generated output, or documentation was reset or
  cleaned.

## Latest addendum — founding profile becomes compact and era-adaptive

- The profile heading now pairs `1789–1815` with `A New State · 4 Presidents`;
  `A small, written, institution-facing slice of the corpus` sits directly
  below that metadata. The former standalone `A New State` overline is gone.
- Presidents is a compact, auto-fitting portrait strip instead of a two-row
  fixed-height card. At 1280px its four portraits occupy a 156px-tall half-width
  card; additional presidents wrap and increase the card height rather than
  inheriting a hard-coded era size.
- Corpus Footprint keeps the exact speech, paragraph, and word corpus shares
  without the ambiguous mini-bars. The two density measures are replaced by
  language evidence from the exact-key-joined speech statistics artifact:
  40.7 words per sentence versus 23.5 corpus-wide, and `shall` 2.0 times per
  1,000 tokens versus 1.1 corpus-wide.
- Named Adversaries now reads as five marbles inside a jar. Every marble retains
  its internal normalized name and adversarial-paragraph count, and diameter
  still increases with count. Live center/radius measurement confirms zero
  pairwise overlap.
- The Era Profile's bottom `Visualize the era` handoff bar is removed; the
  persistent three-part workspace navigation remains the transition surface.
  Its profile subtitle now reads `Start here · people, scale, topics`.
- Live browser QA at 1280×720 measured a 589px profile panel, a 156px
  Presidents card, zero marble overlap, zero horizontal overflow, no corpus
  share-bar elements, and no profile footer.
- Verification: the canonical build validates 72 HTML pages and 161 JSON
  shards; generated JavaScript syntax passes for both story builds; the 40
  focused Founding/V2/V3 contracts and all 2,521 repository tests pass.
  Portable Reggie Doctor reports 0 errors, 0 warnings, and 5 informational
  findings. The local preview remains at `http://localhost:8010/`; nothing was
  deployed, staged, or committed.
- The worktree remains extensively dirty and user-owned. No unrelated source,
  frozen annotation artifact, generated output, or documentation was reset or
  cleaned.

## Latest addendum — founding profile concentrates on people, scale, and subjects

- The Era Profile is now a five-card composition. Presidents spans both desktop
  rows with four larger local portraits; Claimed Constituents, Named
  Adversaries, Corpus Footprint, and Major Topics fill the other two columns.
  The ornamental card-header totals (`4`, `Pending`, `5`, `2.8% words`, and
  `6`) are removed.
- Governing Style is no longer duplicated in the profile. Its full treatment
  remains under Visualize the Era → Governing Channels, where all eight
  assigned audience/medium options and all four president-level channel mixes
  remain available.
- Corpus Footprint is now a compact dashboard: 73 speeches / 6.9% of the
  corpus, 927 paragraphs / 2.6%, and 117,116 words / 2.8%, plus 1,604 words per
  Founding speech against a 3,954 corpus average and 12.7 paragraphs per speech
  against a 34.3 corpus average. The three share bars use one declared 0–10%
  corpus scale.
- Named Adversaries is a deterministic overlapping five-bubble mosaic. Each
  circle contains both the normalized name and adversarial-paragraph count;
  circle size still increases with the count, while the former explanatory
  note and external labels are gone.
- The topic card is titled `Major Topics` and retains its prevalence ordering,
  exact percentages, and highlighted bars.
- Live browser QA at 1280×720 confirmed five title-only card headers, four
  portraits, five internally labeled mosaic bubbles, all five footprint
  measures, the absence of Governing Style from the profile, its complete
  eight-channel visualization under Visualize, and no page-level horizontal
  overflow.
- Verification: the canonical build validates 72 HTML pages and 161 JSON
  shards; generated JavaScript syntax passes for both story builds; the 40
  focused Founding/V2/V3 contracts and all 2,520 repository tests pass.
  Portable Reggie Doctor reports 0 errors, 0 warnings, and 5 informational
  findings. The local preview remains at `http://localhost:8010/`; nothing was
  deployed, staged, or committed.
- The worktree remains extensively dirty and user-owned. No unrelated source,
  frozen annotation artifact, generated output, or documentation was reset or
  cleaned.

## Latest addendum — founding profile becomes a visual six-card summary

- `A New State` is now the Era Profile's sole overline, directly above
  `1789–1815`; it no longer appears beside the dates. The chronological chapter
  headline remains `1789–1815 · The problems of a new state`.
- All six profile cards align their metric with the card title. The Presidents
  card uses a two-column grid of the four cached local portraits with names
  beneath them. The Governing Style card contains all four audience and all
  four medium assignments in a compact internal viewport, with overflow
  available at narrower widths.
- Named Adversaries is now a five-bubble chart. Each bubble's diameter increases
  from the normalized adversary paragraph count, and the exact count remains
  printed inside the mark.
- Topics are sorted by Founding-era prevalence and encoded as horizontal
  highlights while retaining exact percentages: Military preparedness 16.2%,
  Indian & Native affairs 15.0%, Barbary & War of 1812 11.4%, Public finance
  9.4%, Federal law enforcement 6.6%, and Executive power & courts 4.6%.
- Live browser QA at 1280×720 confirmed one visible `A New State`, four
  portraits, five adversary bubbles, all eight governing channels, the ranked
  topic bars, and no page-level horizontal overflow. The Era Profile remains a
  two-row desktop composition.
- Verification: the canonical build validates 72 HTML pages and 161 JSON
  shards; generated JavaScript syntax passes for both story builds; the 40
  focused Founding/V2/V3 contracts and all 2,520 repository tests pass.
  Portable Reggie Doctor reports 0 errors, 0 warnings, and 5 informational
  findings. The local preview remains at `http://localhost:8010/`; nothing was
  deployed, staged, or committed.
- The worktree remains extensively dirty and user-owned. No unrelated source,
  frozen annotation artifact, generated output, or documentation was reset or
  cleaned.

## Latest addendum — founding chapter refined around compact, comparable screens

- The chronological chapter headline is restored to
  `1789–1815 · The problems of a new state`. `A New State` now appears beside
  `1789–1815` inside the Era Profile, where the six shared cards occupy exactly
  two rows. Governing style is a compact two-signal summary there; all four
  assigned audience options and all four assigned medium options move to the
  dedicated Governing Channels screen.
- Governing Channels now pairs two complete four-option bar distributions with
  compact Washington/Adams/Jefferson/Madison audience and medium stacks. It
  retains zero-share channels rather than allowing the dominant written and
  congressional forms to imply that no other assigned forms existed.
- The adversary network now derives president node area from
  `sqrt(adversary-word share × distinct normalized named adversaries)`, scaled
  to the largest score. The word share is the share of each president's corpus
  words in distinct paragraphs that name at least one adversary; the name count
  uses the full normalized adversary layer even though the visible network
  limits edges to leading connections. Exact inputs are exposed in labels and
  tooltips; paragraph counts still control edge width.
- The Era Argument is reduced to one headline, four compact topic/test cards,
  the president-level ribbons, and one unrevealed next-era question. The old
  next-era metric preview is no longer rendered. The Public Finance sub-lanes
  remain derived in `founding_story_data` but are absent from the chart, exact
  table, and interaction surface.
- Topic Afterlives now opens with the Native/naval takeaway, then shows six
  rows. `Early Naval Wars: Barbary & the War of 1812` is a first-class
  afterlife row beside the five existing concerns. The label rail remains
  icon-first and the Native row remains highlighted.
- Live browser measurements at 1280×720 put Governing Channels at 434px,
  Adversaries at 516px, Era Argument at 528px, and Topic Afterlives at 536px.
  Each complete visualization fits below the sticky header and there is no
  page-level horizontal overflow.
- Verification: the canonical build validates 72 HTML pages and 161 JSON
  shards; generated JavaScript syntax passes for both story builds; the 40
  focused Founding/V2/V3 contracts and all 2,519 repository tests pass.
  Portable Reggie Doctor reports 0 errors, 0 warnings, and 5 informational
  findings. The local preview remains at `http://localhost:8010/`; nothing was
  deployed, staged, or committed.
- The worktree remains extensively dirty and user-owned. No unrelated source,
  frozen annotation artifact, generated output, or documentation was reset or
  cleaned.

## Latest addendum — provisional Sol execution stopped at frozen integrity gates

- Owner resolutions `ARCV1-RES014` and `ARCV1-RES016` authorize exact execution
  of provisional plan
  `pcplan_541b52ab42e978227017f4d2ca5f67d7e624e5a4d76b8b256b8fb371577427bf`,
  including explicit transmission of all locked rendered payload fields to
  OpenAI `gpt-5.6-sol`/`high`. They do not authorize promotion,
  materialization, site generation, deployment, or frozen paid-artifact edits.
- Campaign
  `pcamp_ce0ff257543c3a342d96147b9e5e46953ca7016aab9bf29f1c6f1b573100939b`
  contains the exact 9,186 assignments / 35,154 fresh subjects. The independent
  partition verifier reconciles every target key, source hash, bundle-input
  hash, rendered hash, and all 81,663,911 input-proxy tokens.
- The resumable executor uses SQLite atomic claims, fresh restricted runtime
  receipts, pre-invocation token reservation, a frozen maximum of three
  attempts, immutable accepted packages, deterministic gzip storage, and
  exact-model execution through the desktop-bundled Codex CLI. No response is
  consolidated into the generic ledger until the whole run passes audit.
- Current state is `retry_exhausted_blocked`: 18 assignments are accepted, one
  is terminally failed, and 9,167 remain available. Assignment index 12 used
  three base attempts plus the one owner-authorized RES018 infrastructure
  recovery attempt. The first lost durable delivery when the volume hit
  ENOSPC; attempts 2–4 independently failed the registered constituency
  validator. Attempt 4 failed because an explicit claim carried a resolved
  referent. The unchanged prompt and schema were used on every attempt. Per
  RES018, execution stopped immediately. Do not claim more assignments, repair
  a payload, repartition, reset attempt history, or increase retries without a
  new explicit owner decision.
- The protected-state audit also detects that the generated `docs/` tree
  changed after campaign initialization (185 files have newer mtimes). This
  executor did not run the site generator. The changes are user-owned and must
  not be reverted. A completion audit cannot pass against the frozen baseline
  without an explicit owner disposition.
- The earlier disk-space blocker was resolved by the repository user; 36 GiB
  was available before RES018 was applied and the database reopened cleanly.
- `ARCV1-RES017` records the original blockers, `ARCV1-RES018` the narrow
  recovery amendment
  `pexam_7ce6624c94855635ac6a0dafe5431fb1c9c92b3f86c529d42d689022160326a6`,
  and `ARCV1-RES019` the failed fourth attempt. No run seal, provisional
  composite, adjudication, promotion, materialization, active-pointer change,
  site build, or deployment was created.
- Focused executor/workflow tests passed before execution; the real partition
  verifier passed at 9,186/35,154/81,663,911. Re-run the focused and full
  relevant annotation suites after any owner-approved integrity amendment.

## Latest addendum — founding chapter becomes a guided, connected sequence

- The homepage title remains `America Through Its Presidents`; the restored
  subheadline now sits directly beneath it as `1789–2026 · A country told
  through presidential speech`. The redundant homepage row linking to
  Presidents, Explorer, Compare, and Methods is gone because those destinations
  already live in the global navigation.
- The first chapter is now introduced as `1789–1815: A New State`. The old
  numbered left story spine and vertical rule are removed across the page.
- `Era profile`, `Visualize the era`, and `Contextualize the era` are visually
  joined to one detail surface. Profile and Visualize end with explicit next
  actions that replace the panel in place and align the whole workspace below
  the sticky progress header; Contextualize ends by returning to normal
  chronology at 1816–1849. Direct hashes preserve the same alignment.
- Governing style now shows all four assigned audience groups and all four
  assigned medium groups, including zero-share options, rather than only the
  two dominant labels.
- Contextualize now has two screens, not three. `Topic afterlives` carries a
  compact two-item note showing the Founding-to-Present decline in the Native
  affairs label (15.0% → 0.02%) and the named Barbary/War of 1812 naval label
  (11.4% → 0.05%). The copy distinguishes disappearing corpus labels from
  disappearing peoples, power, policy, foreign relations, or military speech.
- The era argument now gives `Getting the Republic Up and Running`, the
  four-test question, the four test definitions, their percentages, and the
  president-by-president chart a deliberate visual hierarchy.
- Verification: the canonical build validates 72 HTML pages and 161 JSON
  shards; generated JavaScript syntax and 40 focused Founding/V2/V3 contracts
  pass. Live browser QA covered the hero, direct-hash alignment, all three
  top-level views, both Contextualize screens, collapsed finance/topic labels,
  desktop fit, and 390px containment with no page-level overflow. The full
  repository run reached 2,514 passes and five setup errors caused only by the
  full disk while writing one provisional-plan fixture; after its disposable
  temp directory was removed, that complete 10-test module passed. Thus all
  2,519 collected tests passed across the final verification runs. The local
  preview remains at `http://localhost:8010/`; nothing was deployed.
- The worktree remains extensively dirty and user-owned. No unrelated source,
  data, generated artifact, or task was reset, staged, or committed. During
  verification only disposable Python bytecode/test caches were removed to
  recover space on a nearly full volume.

## Latest addendum — founding era built as a repeatable three-part story

- The 1789–1815 chapter keeps the shared `Era profile`, `Visualize the era`,
  and `Contextualize the era` stage. Visualize now has three mutually exclusive
  screens: president-level top-five topic agendas, aggregate plus
  president-level governing channels, and a president-to-named-adversary
  network with paragraph-weighted edges and auditable alias grouping.
- Contextualize now opens with `Getting the Republic Up and Running · Four
  tests of a functioning country`: hold the union together (152/927, 16.4%),
  finance the government (132/927, 14.2%), navigate Native nations and
  continental power (139/927, 15.0%), and exercise independence abroad
  (327/927, 35.3%). The definitions share no topic labels; the largest
  paragraph overlap is Native/abroad at 21/927 (2.3%).
- The four-test bands expose Washington/Adams/Jefferson/Madison shares and a
  fixed next-era preview. National banking rises from 1.5% to 13.6%, tariffs
  from 3.9% to 9.7%, executive power/courts from 4.6% to 9.2%, while maritime
  rights fall from 17.4% to 2.2%.
- `Topic afterlives` is a compact five-row, nine-era bubble chart. Its label
  rail starts icon-only, labels can be revealed, and Public Finance still
  starts collapsed with its two subrows completely hidden until expanded.
  Phantom overflow from the hidden tooltip box was removed; the compact view
  fits the normal 756px content width without horizontal scrolling.
- `🪶 Native relations` remains the focused contextual screen with the 15.0%
  to 0.02% corpus-attention comparison, selected era bars, presidential
  receipts, and the explicit non-disappearance caveat.
- The Era profile adds an exact word footprint: 117,116/4,179,266 corpus words
  (2.8%), alongside 73/1,057 speeches and 927/36,229 paragraphs. Constituencies
  continue to use the declared fallback because the promoted projection is
  empty.
- The generic military-preparedness topic is deliberately outside the
  independence-abroad test because the current artifact cannot separate
  European, Native, domestic, and other military context. Backlog item
  `annotate-referenced-regions-and-groups` remains the prerequisite for that
  geographic/group comparison.
- Verification: 2,504 repository tests pass; the focused six-file story/site
  suite passes 60 tests; Python compilation, generated JavaScript syntax, and
  whitespace checks pass; the build validates 72 HTML pages and 161 JSON
  shards. Live browser QA covered every Founding tab, hashes, collapsed
  controls, normal-width fit, and 390px mobile containment with zero console
  errors. The local preview remains available at `http://localhost:8010/`;
  nothing was deployed.
- The worktree was already extensively dirty and remains user-owned. No
  unrelated file was cleaned, reset, staged, or committed.

## Latest addendum — provisional Sol corpus planner published

This addendum supersedes the next-action boundary in the Stage 3 addendum below.

- `ARCV1-RES012` approves planning only for an additive provisional Sol corpus
  evidence layer. Existing Sonnet, Opus, evaluated Sol, adjudicated Sol, and
  future corpus Sol remain distinct sources; none is reinterpreted as ground
  truth.
- The evaluated 240 paragraphs contribute exactly 1,680 final fields through
  reuse manifest
  `reuse_d5f9059121d080eab7ae31f0b4cdcf2d1be9c5bf8d3b1a6d411a4598f3ad7ec1`.
  The planner derived 972 three-way-unanimous and 708 adjudicated entries from
  the sealed reference/A/B/adjudication values and published decisions, with
  zero missing, duplicate, unclear, or unresolved finals.
- Fresh selection
  `fresh_1ab0bf0f22825aa61e83e54744d90bf0b23793e748e3d71d7801f8e8b9533cd3`
  excludes those 240 subjects by canonical identity and includes the other
  35,154 current canonical paragraphs, including all other 482 pilot
  paragraphs. It fixes 9,186 deterministic same-speech assignments.
- Non-executable plan
  `pcplan_541b52ab42e978227017f4d2ca5f67d7e624e5a4d76b8b256b8fb371577427bf`
  locks the candidate bundle, prompt/schema/context/render/registry hashes,
  exact `gpt-5.6-sol`/`high`, the current corpus and Phase 1 generation, every
  source seal and comparison/evaluation/adjudication identity, and a
  deterministic 81,663,911-input-token estimate with an 89,830,303 ceiling.
- The plan is `provisional`, never `production_eligible`. It preserves reused
  and fresh provenance separately, leaves current primary labels and the active
  materialization pointer unchanged, and prohibits campaign/run
  initialization, assignments, model/provider invocation, response writes,
  sealing, promotion, materialization, site generation, and deployment.
- `ARCV1-G007` was authorized by `ARCV1-RES014` and the explicit payload-egress
  boundary by `ARCV1-RES016`. Execution later stopped under `ARCV1-RES017`;
  see the latest addendum.
- Verification: 10 focused provisional-planner tests and the complete 61-test
  annotation-refresh/workflow/ledger group pass. Compilation, bundle-registry
  validation, review-queue validation, content hashes, byte-idempotent
  replanning, provider/network exclusion, side-effect counts, and whitespace
  checks pass. The full repository suite reports 2,497 passes and one unrelated
  pre-existing story-page copy failure:
  `TestProseExtraThreading::test_omitting_prose_extra_entirely_still_builds`.
  Portable Reggie Doctor reports 0 errors, 0 warnings, and 5 informational
  findings.

## Latest addendum — annotation refresh Stage 3 evaluated; freeze rejected

This addendum supersedes the Stage 3 status in the annotation-refresh addenda
below.

- The owner-approved pre-results amendment remains effective. No first-response
  provenance recovery was performed or is required; accepted payloads/events
  remain schema-valid, while first-attempt validity is not a decision metric.
- The composition diagnostic, blind A, blind B, independent reference, and
  post-reference adjudication runs are sealed and verify without drift. The
  fresh high-reasoning reference covered 240/240 subjects; adjudication covered
  236/236 disputed subjects and all 708 required disputes with zero issues or
  unresolved constituency rows.
- All 708 required adjudication decisions were published append-only. Five
  hash-only comparison artifacts were published before adjudication.
- Deterministic evaluation
  `eval_bbb2212dea6936070b23bff11100615c602396b72d2fe6058cc707864f060b61`
  applied the amended frozen policy, 2,000 speech-clustered bootstrap draws,
  the unchanged Phase 1 generation, and sealed-source/blindness checks. It
  passed 114/150 gates and failed 36: structural 13/13, composition 6/6,
  constituency 44/57, and existing-label 51/74.
- `ARCV1-RES011` therefore rejects `ARCV1-G003`. No
  `constituencies@v1` specification, `paragraph_judgment_v2@v2` bundle, or
  passing campaign outcome was frozen. No full shadow, promotion,
  materialization, site build, or deployment was performed.
- Invocation-tone proposal `ARCV1-D018` remains pending explicit owner approval
  or revision. No invocation labeling is authorized.
- Verification: 91 focused annotation tests pass; Python compilation, registry
  validation, seal/evaluation verification, static provider/network inspection,
  and whitespace checks pass. The full suite has 2,486 passes and two unrelated
  story-page copy failures in `test_band_charts.py` and
  `test_founding_story.py`. Reggie Doctor reports 0 errors, 0 warnings, and 5
  informational findings.

## Latest addendum — founding-era single-stage tab template

- The 1789–1815 chapter now has three unnumbered peer tabs in one reusable
  stage: `Era profile`, `Visualize the era`, and `Contextualize the era`. Only
  the selected panel occupies the space; the previous vertical stack and
  scroll-completion model are gone.
- `Visualize the era` has two keyboard-accessible nested screens: `What
  mattered` shows the sovereignty/security and federal-capacity displays;
  `How governance happened` shows assigned written form, congressional
  audience, and the three overlapping capacity components.
- `Contextualize the era` also has two nested screens: `Across time` retains
  the full five-topic nine-era graph and collapsed public-finance subrows;
  `Native-affairs context` isolates the 15.0% → 0.02% comparison, four selected
  era bars, Washington/Jefferson receipts, and the non-disappearance caveat.
- The profile now has a sixth `Topics` card listing all five founding concerns
  and their era paragraph shares. Presidents, constituents, adversaries,
  governing style, and corpus footprint remain alongside it.
- The fixed 210px narrative rail remains removed. Panels use short display
  figures and full-width content; the founding chapter widens safely to 1,160px
  while the topic graph retains local horizontal scrolling.
- The profile now reports both corpus footprints from keyed source frames:
  73/1,057 speeches (6.9%) and 927/36,229 paragraphs (2.6%). Presidents, style,
  and adversaries remain artifact-derived.
- The promoted `constituency_claims.parquet` projection is empty. The fallback
  therefore reads `Pending` and `Illustrative only`; it no longer lists Native
  nations or peoples. Native affairs can contribute to the distinct
  sovereignty/security analytic lens without becoming an inferred
  constituency claim.
- The external-or-Native sovereignty/security union remains 58.4%, with 47.1%
  external, 15.0% Native affairs, and 3.8% overlap displayed. Its compact
  caveat says explicitly that this is a combined lens, not a reclassification.
- Public Finance remains collapsed by default. A 27px circular down arrow
  expands the two component rows in place, and the parent row now carries the
  compact framing shift `Debt/Treasury → taxes, deficits & spending`.
- Verification: 58 focused expansion/founding/V3/V2/site-validation/
  metric-registry tests pass. Live browser QA confirms that the three top-level
  panels and four nested screens are mutually exclusive, update their hashes
  and ARIA selection state, retain the wider graph, and keep Public Finance
  closed by default. The preview remains local at `http://localhost:8010/`;
  nothing was deployed.
- The worktree was already extensively dirty and remains user-owned. No
  unrelated files were cleaned, reset, staged, or committed. The next safe
  design action is to apply this shared profile frame to later eras one at a
  time, not to fabricate their missing constituency claims.

## Latest addendum — Stage 3 first-response gate superseded pre-results

This addendum supersedes the provenance-blocker addendum immediately below.

- The repository owner attested that all completed pilot responses were
  machine-produced without owner payload edits and explicitly removed
  first-response schema validity as unnecessary.
- `ARCV1-RES010` records the decision before reference labeling,
  candidate-label inspection, comparison, adjudication, evaluation, or any gate
  decision. This timing prevents results-driven threshold selection.
- The amendment removes only the absolute 99% first-response gate and the two
  schema-validity non-inferiority margins. Accepted responses/events must remain
  completely schema-valid; every other structural and substantive gate remains
  unchanged.
- The original plan, campaign, and base evaluation-policy hashes remain
  immutable. A separate content-addressed amendment supplies the effective
  evaluation-policy identity:
  `amend_732e266492b6a7fb744c12b121d28a8f6b97f8837a835942018b1f3898ed4180`
  / `sha256:43c4b5c1ff7bdb6d43c7586b1413e5db8ef6df8b08419a815b52d689b17098ae`.
- `ARCV1-B001` is satisfied by removal rather than by pretending the missing
  attempt history exists. Stage 3 may resume at provider-neutral reference
  infrastructure and independent blind reference construction.

## Latest addendum — annotation refresh Stage 3 provenance blocker

This addendum supersedes the pilot-initialization addendum below for current
ARCV1 status.

- The three completed pilot children pass `status_run` and registry-v2
  `audit_run`: composition diagnostic 144/144 subjects in 36 accepted
  responses, blind A 722/722 in 182, and blind B 722/722 in 182. All have zero
  remaining subjects, zero open assignments, `status: ok`, and no audit issues.
- All 31 combined execution-session receipts are content-address valid and
  attest exact `gpt-5.6-sol`/`high`, a fresh restricted session, and no sibling
  response inspection: 3 diagnostic, 16 blind A, and 12 blind B.
- No temporary, partial, rejected-response, response-attempt, or ingest-audit
  artifact exists in the campaign work roots.
- `annotation_workflow.ingest_response` validates before its first write and
  persists only accepted responses. A rejected first submission leaves no
  authoritative record. The 400 accepted response files therefore prove
  accepted-response completeness but cannot prove the frozen first-response
  schema-validity numerator or denominator.
- `ARCV1-B001` records this provenance blocker. Per the explicit Stage 3 stop
  rule, no reference assignment, pilot seal, comparison, evaluation,
  adjudication, specification/bundle freeze, campaign seal, full-shadow work,
  promotion, materialization, site build, or deployment was performed.
- The campaign, approved plan, corpus snapshot, selection, prompt, schema,
  bundle, context, evaluation, and promotion identities validate without drift.
  The active Phase 1 generation remains
  `5b3f0e252c63a3b6cf3cba5d5e84e23def38eaa8edf29b89af827c92e5bce307`.

The next safe action requires authoritative evidence that reconstructs every
pilot assignment's first submitted response and validity outcome. If such
evidence does not exist, the frozen structural gate cannot be estimated and the
completed pilot must remain unsealed and unpromoted; adding attempt logging now
would protect future runs but cannot retroactively repair this campaign.

## Latest addendum — annotation refresh Phase 2 pilot initialized

This addendum supersedes the runtime/authorization and Stage 2 planning addenda
below for current Phase 2 status.

- `ARCV1-RES009` records verified initialization of campaign
  `camp_7cb35b8b739bcc2fb36ebe9609e81d629fee97c2d604cad5bd64d7cc25b5b316`
  under the exact `ARCV1-RES007` `gpt-5.6-sol`/`high` receipt and
  `ARCV1-RES008` execution authorization.
- The composition-diagnostic, blind-A, and blind-B child runs contain exactly
  144, 722, and 722 subjects. All have zero assignments, responses, and events.
- The campaign locks the approved plan, policy, bundle, prompt, schema, context,
  render, selection, corpus, model, effort, and approval receipts.
- The initializer publishes a campaign commit marker only after all three child
  runs exist. Repeating initialization is idempotent.
- Every future `next` call requires a fresh exact runtime receipt plus an explicit
  blindness receipt attesting that the labeling session is fresh and has not
  inspected sibling responses.
- The approved plan bytes remain unchanged, the active Phase 1 generation remains
  `5b3f0e252c63a3b6cf3cba5d5e84e23def38eaa8edf29b89af827c92e5bce307`,
  and no protected Phase 1, paid, generated-site, or deployment artifact changed.
- `ARCV1-G003`, `ARCV1-G004`, and `ARCV1-G005` remain closed. Initialization does
  not authorize freezing, full-shadow execution, promotion, materialization,
  site generation, or deployment.

The next safe action is a fresh restricted `gpt-5.6-sol`/`high` labeling session
for exactly one child. It must create and persist its current runtime and
blindness receipts before requesting the first assignment. Do not inspect or
label a sibling run in that session.

## Latest addendum — annotation refresh Phase 2 runtime and execution authorization

This addendum supersedes the Stage 2 planning addendum below for current Phase 2
status. The approved pilot is authorized but has not been initialized or executed
in this checkpoint conversation.

- A fresh restricted Codex turn exposed exact runtime model `gpt-5.6-sol` with
  actual reasoning effort `high`. `ARCV1-RES007` records the full runtime metadata
  receipt and satisfies `ARCV1-G001`.
- The earlier `gpt-5.6-sol`/`xhigh` observation remains historical rejected
  evidence. It was not reused or treated as satisfying the literal `high`
  requirement.
- After `ARCV1-G001` was satisfied, the repository user explicitly stated
  “Authorize ARCV1-G006 pilot execution.” `ARCV1-RES008` records approval of the
  exact pilot execution checkpoint.
- The immutable plan remains
  `plan_3af71613624c605209c169717c7c73024235e01ba0750d046940a01d2512a17f`;
  it was not modified.
- No executable campaign, child run, assignment, response, labeling, seal,
  comparison, adjudication, promotion, materialization, site build, or deployment
  was created in this conversation.
- `ARCV1-G003`, `ARCV1-G004`, and `ARCV1-G005` remain closed. Pilot authorization
  does not authorize a specification freeze, full-shadow refresh, promotion,
  materialization, site generation, or deployment.

The next safe action is a separate restricted pilot-execution conversation. It
must re-expose exact model `gpt-5.6-sol` at `high`, preserve the declared blindness
and reference ordering, and consume only the approved persisted selections and
fixed 182/182/36 assignment partitions.

## Latest addendum — annotation refresh Phase 2, Stage 2 planning

The repository user authorized Stage 2 planning through `ARCV1-RES003` and
approved `ARCV1-I001`/`ARCV1-D016` through `ARCV1-RES004`. Stage 2 planning is
complete, and the exact plan is approved as `ARCV1-G002`/`ARCV1-RES006`. No model
assignment is authorized.

- The active Phase 1 generation and canonical keyed corpus still contain 35,394
  paragraphs, 1,053 speeches, and 75 populated era×genre cells.
- Three cells, not the contract's estimated two, contain fewer than four
  paragraphs: Civil War & Reconstruction × `other` (1), The Gilded Age ×
  `eulogy_or_commemoration` (2), and War & New Deal × `proclamation` (3).
- Taking all six sparse rows plus only four-target blocks cannot produce the
  original 480-row core. `ARCV1-RES004` approves 482 core / 722 total rows.
- Full four-target speech tiling leaves 1,642 tail paragraphs (4.64%) outside
  that frame. `ARCV1-RES004` limits the core to design-balanced evaluation and
  prohibits whole-corpus prevalence claims.
- `ARCV1-S003` is satisfied: the renderer transmits each target once with shared
  outer context, and `next` honors the persisted exact batch partition.
- The combined, diagnostic, and reference selection IDs are
  `sel_17cf0d99dc497c63a0f7ee85d2d86436883df9a0eb7a6cd7f6e1234271c51515`,
  `sel_ae7b0e92cb8a37e03c39a6a7270cfc8c5b39698c409787a8126ec84a6bf24fd9`,
  and `sel_799402f09a43f76bfd1a9386b47614cf0f7fccb3c85374ffaa60103f046de9bc`.
- The non-executable plan is
  `plan_3af71613624c605209c169717c7c73024235e01ba0750d046940a01d2512a17f`.
  It fixes 182/182/36 assignments and estimates 3,359,122 pilot input tokens,
  with a 3,695,035-token ceiling. The 24.87% increase over the preliminary
  estimate triggered the now-approved `ARCV1-G002` review.
- Current orchestration metadata exposed `gpt-5.6-sol` at `xhigh`; it is not an
  execution receipt for the approved `high` labeling profile. `ARCV1-G001`
  remains pending for the later restricted labeling runtime.
- No executable campaign, runtime execution receipt, child run, assignment,
  response, seal, comparison, adjudication, promotion, materialization, site
  build, or deployment was created.
- Verification passed: 57 focused annotation tests, all 2,466 repository tests
  with 70 existing warnings, byte-identical idempotent replanning, unchanged
  active Phase 1 counts, and Reggie Doctor at 0 errors / 0 warnings / 5 infos.

The next safe action is a fresh restricted ChatGPT labeling conversation that
first exposes and persists its exact model identity and `high` effort under
`ARCV1-G001`, then stops. Only a separate repository-owner approval of
`ARCV1-G006` may permit campaign initialization or the first assignment.

## Latest addendum — annotation refresh Phase 2, Stage 1

The provider-neutral refresh contract was approved as `ARCV1-RES001`, and its
Stage 1 implementation is complete. This addendum supersedes the older Phase 1
and Story-task status below for the current handoff; those records remain intact.

- Contract and review state:
  `notes/annotation-refresh-campaign-v1.md`,
  `notes/annotation-refresh-campaign-review-queue-v1.json`, and
  `notes/annotation-refresh-campaign-review-report-v1.md`.
- `annotation_refresh.py` validates four declarative bundles, selects
  production-eligible `all` sets, derives exact full/delta selections, locks
  deterministic in-memory campaign identities, and requires exact model/effort
  receipts before a bundle run can be initialized.
- Label-spec registry v2 is additive; Phase 1 registry v1 and every Phase 1 spec
  remain unchanged. The combined paragraph, invocation, and constituency-only
  bundles remain pilot-gated. Only frozen `speech_factual@v1` is currently
  production-eligible.
- `annotation_workflow.py` now supports immutable bundle snapshots, same-speech
  batching, trusted assignment-local IDs, bounded model-facing rendering,
  schema-complete decoding into separate atomic events, special validators, and
  resume without reissuing completed work.
- `annotation_ledger.py` accepts the additive spec-v2 binding and derives
  `constituency_claims` only from explicitly promoted constituency groups.
  Raw response events remain unchanged.
- Focused annotation verification passes 55 tests. The complete repository suite
  passes 2,464 tests with 70 existing warnings. Registry validation, compilation,
  and the provider/network static audit pass. Final Reggie Doctor reports 0
  errors, 0 warnings, and 5 informational findings.
- No repository campaign, pilot selection, runtime receipt, assignment, response,
  sealed run, comparison, adjudication, promotion, materialization, site build,
  or deployment was created. The active Phase 1 generation remains unchanged.
- At Stage 1 completion, the next proposed action was a separate Stage 2
  planning approval. That approval and its subsequent refusal audit are now
  recorded in the superseding addendum above. Do not issue a model assignment.

## Latest addendum — annotation ledger Phase 1

The provider-neutral, append-only annotation ledger Phase 1 is complete. This addendum supersedes
the older Story-task status below for the current handoff; the prior record remains intact.

- The implementation contract and human-review record are
  `notes/annotation-ledger-v1.md`, `notes/annotation-ledger-review-queue-v1.json`, and
  `notes/annotation-ledger-review-report-v1.md`.
- Generic storage, sealing, materialization, comparison, adjudication, promotion, and terminal
  workflow foundations live in `annotation_ledger.py`, `annotation_backfill.py`, and
  `annotation_workflow.py`. The constituency specification remains draft/pilot-only.
- All 15 frozen manifests were backfilled into 16 sealed runs: 261,233 exact legacy events plus
  102 trusted correction-overlay events. No file under `data/llm_annotations/`, no correction
  manifest, and no generated-site file was changed for this task.
- Explicit continuity and correction-overlay promotions yield current primary judgments for all
  35,394 canonical paragraphs and primary factual labels for all 1,053 canonical speeches. Opus
  labels remain available but unpromoted.
- The active deterministic generation is
  `data/annotation_ledger/materialized/generations/5b3f0e252c63a3b6cf3cba5d5e84e23def38eaa8edf29b89af827c92e5bce307`.
  It contains 261,335 label events, 196,153 promotion transitions, 206,837 current labels, 136
  evaluation records, no adjudications, and an empty schema-valid constituency projection.
- Review decisions have no unresolved queue items. Preserved provenance gaps are limited to
  unsupported legacy details: unknown assigned/labeled timestamps, discarded intermediate model
  responses, incomplete invocation prompt/cost metadata, and unknown effort where no exact
  versioned-spec match exists.
- Focused annotation tests pass (38 tests), the complete suite passes (2,447 tests), materialization
  is byte-reproducible, and the final portable Reggie Doctor reports no errors or warnings.
- No branch, staging, commit, push, pull request, site rebuild, or deployment was performed.
  Production constituency labeling remains a separately authorized future task.

## Current status

The locked second-era Story plan is implemented in the Python generator and rebuilt static site.
The 1816–1849 chapter now uses a fixed-period topic matrix with an aligned adversary-naming strip,
artifact-derived prose, named source receipts, uncertainty status, and a 1850–1854 next-era
preview. The founding-era redesign and every later chapter remain in place. The local preview is
running at `http://127.0.0.1:8010/`. Nothing was deployed.

## Active task

No active task is claimed.

## Branch

`master`; no branch, staging, commit, push, pull request, or deployment was created for this work.

## Delivered behavior

- `src/presidential_profiles/expansion_story.py` derives the published layer over eight fixed
  periods: 1816–20 through 1846–49, plus the fixed-scale 1850–54 coda.
- The matrix carries eight declared rows: diplomacy and treaties, federal finance, tariffs,
  banking, territorial expansion, its two visibly indented subrows, and slavery/sectionalism.
  Territorial expansion is a de-duplicated union; every denominator includes topic-free
  paragraphs.
- Speech-clustered 500-draw bootstrap intervals use the repository band policy and deterministic
  seed. Paired-model disagreement only widens intervals where the paired sample supports it;
  thin, suppressed, and unresolvable statuses remain visible in symbols, hover text, and the
  native table.
- The aligned adversary strip uses separate bars on a fixed 0–40% axis. Its two annotated peaks
  are 36.5% in 1831–35 and 27.3% in 1846–49. Entity receipts identify institutions at 53% in the
  former period (Bank of the United States and Senate) and nations at 77% in the latter (Mexico).
- The matrix retains a fixed 0–60% scale. Published checkpoints include banking at 23.3% in
  1831–35; territorial expansion at 56.4% and external acquisition at 49.2% in 1846–49; and
  slavery/sectionalism at 37.7% in the 1850–54 preview.
- The chapter title is `The nation begins arguing with itself`; its bridge and takeaway are
  derived from the artifact. Three narrative movements keep chronology separate from synthesis,
  and event markers explicitly state that they orient time rather than establish causation.
- Exact keyed receipts cover Monroe's 1817 First Annual Message, Jackson's Bank Veto,
  Nullification Proclamation, and removal message, plus Polk's War Message and slavery-territory
  message. Jackson's speech-level contrast is 53.6%, 51.7%, and 0.0%, respectively.
- The Plotly surface has a local horizontal scroller, accessible hover detail, a keyboard-native
  exact table fallback, named Measure/Evidence tiles, and an outlined/shaded `Next era preview`
  coda. Page-level horizontal overflow remains clipped.
- The old three-line expansion figure and its obsolete metric registration are removed.
  `expansion_story` now registers paragraph share and confidence interval.

## Artifacts and documentation

- Published deterministic layer:
  `data/expansion_story/period_metrics.parquet` (72 rows, 27 columns) and
  `data/expansion_story/meta.json`.
- Rebuilding the layer twice produced identical SHA-256 values:
  `b82c3080c041362f99e35af43661a0d941939d4ccf1d114e015fe9a34a9f23b2` for the
  parquet and `67c76eaab384a4fb35c7273b862c0cdb5fbf5286a66ec028153de9c3339f60e5`
  for metadata.
- `notes/story-redesign-v4.md` records the locked visual, statistical, narrative, receipt, and
  acceptance contract. `README.md`, `.codex/knowledge/README.md`, `HISTORY.md`, and this handoff
  route the new layer and result.
- `docs/index.html` and `docs/index_selfcontained.html`, along with the normal generated-site
  output, were refreshed from source. Generated HTML was not hand-edited.

## Verification

- Focused expansion/founding/V3/V2/site-validation/metric-registry gate:
  `arch -x86_64 .venv/bin/python -m pytest -q tests/test_expansion_story.py
  tests/test_story_v3.py tests/test_story_v2.py tests/test_metric_registry.py
  tests/test_site_validation.py tests/test_founding_story.py` → 56 passed.
- Full gate: `arch -x86_64 .venv/bin/python -m pytest -q` → 2,394 passed, 70 existing
  warnings.
- Rebuild: both the required default build and the final
  `arch -x86_64 .venv/bin/python -m presidential_profiles.site --inline` completed; 72 HTML pages
  and 161 JSON shards validated, and both story variants were refreshed.
- Generated JavaScript syntax checks pass for the normal and self-contained stories. Python
  compilation and `git diff --check` pass.
- In-app browser QA at 1280×720 and 390×844 confirms the exact title/takeaway, matrix/bar
  alignment, eight bars, two peak annotations, entity receipts, coda shading/label, stacked
  mobile receipts, hover intervals/support/disagreement, and local horizontal scrolling.
- At mobile width the chart scroller has 785px of local range while the page itself has zero
  horizontal overflow. Normal and reduced-motion consoles have no warnings or errors.
- `?motion=reduce` sets the reduced-motion branch: chapter opacity is 1, transform is none,
  transition duration is 0s, and the chart remains rendered.
- Final Reggie Doctor reports 0 errors, 0 warnings, and 5 informational findings: optional
  `DECISIONS.md` and `.codex/ASSISTANT-GUIDE.md` are absent; `.claude/stats.json` contains existing
  tool telemetry; the worktree has 309 changed or untracked paths; and the repository remains in
  external/native Reggie docs mode without adapter scripts.

## Worktree ownership

Every modification and untracked path present at the initial scan remains user-owned, including
all pre-existing `.claude`, source, tests, data, generated-site, dependency, and documentation
changes. This task made narrow source/test/documentation edits, added the expansion-story module,
tests, V4 note, and derived layer, and regenerated the site through its builder. It did not clean,
restore, reformat, stage, or overwrite unrelated work. `.claude/stats.json` remains existing tool
telemetry and is not part of this task.

## Decisions and limitations

- Python under `src/presidential_profiles/` remains the source of truth; `docs/` is generated.
- Frozen paid annotations were read but never regenerated or hand-edited.
- Matrix and strip values describe this curated presidential-speech corpus, not public opinion,
  policy effect, reception, or causal impact.
- Topic rows overlap by design. The territorial parent and its subrows are not additive.
- The 1850–54 column is a preview on unchanged scales, not part of the chapter chronology.
- Use Reggie documentation mode. No Reggie task or generic adapter was created.
- Preview only at `http://127.0.0.1:8010/`; public deployment remains out of scope.
- No blocker remains.

## Next safe action

Review the local preview, then commit only the intended second-era changes together with any
separately reviewed pre-existing worktree changes. Do not treat the dirty worktree as this task's
ownership.

## Era-grid story migration addendum — 2026-07-26

The story page now uses the reviewed nine-era historical grid: 1789–1808, 1809–1849,
1850–1868, 1869–1912, 1913–1932, 1933–1952, 1953–1980, 1981–2016, and 2017–2026.
`era_profiles.STORY_ERAS` is the canonical story axis. The older `trends.ERAS` remains
unchanged as a legacy reporting axis for already registered and precomputed artifacts.

- Era profiles, contextualizations, custom visualizations, navigation, story progress, and
  synthesis comparisons were regenerated on the reviewed boundaries.
- The founding chapter now ends in 1808. Madison's accession and the War of 1812 open the
  1809–1849 chapter, whose historical arc runs from war through institutions and internal
  conflict to continental conquest.
- The expansion matrix now presents only its governed postwar 1816–1849 evidence. The
  1850–1854 sectional coda and `Next era preview` treatment were removed from the story and
  reserved for the 1850–1868 chapter.
- The 1869 boundary is tied to Grant's accession while explicitly stating that it does not
  declare Reconstruction historically complete. The later chapters now use the accepted 1913,
  1933, 1953, 1981, and 2017 starts, with historically scoped comparison windows and limitations.
- The story's rhetorical-weather synthesis is recomputed directly from speech markers across
  the new grid rather than reading the legacy trend-era table.

Verification is complete: the focused era/story suite passes 80 tests; the full suite passes
2,555 tests with 70 existing warnings; Python compilation, `git diff --check`, and static parsing
of 71 inline scripts across 74 generated HTML files pass. The canonical site build validated
73 HTML pages and 165 JSON shards. In-app browser QA confirmed all nine ranges, keyboard tab
navigation, the seven-column expansion matrix ending at 1846–49, absence of the sectional coda,
zero page-level horizontal overflow, and no browser console warnings or errors.

The local preview remains open at `http://127.0.0.1:8010/`. Nothing was deployed, staged,
committed, pushed, or placed on a new branch. All unrelated dirty-worktree changes remain
user-owned. Final Reggie Doctor reports 0 errors, 0 warnings, and 5 informational findings.
