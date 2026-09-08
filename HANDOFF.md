# Handoff

## Latest addendum — Mobile experience GitHub Pages release

- The owner approved publication on 2026-09-08. Content PR
  [#8](https://github.com/jacobfulfyll/presidential_profiles/pull/8) merged release commit
  `b048e4358fee34e37755178f4e47b2a20c0ca0d4` to `master` as
  `67f8ff24a1df14727a1237403ea4e4f1d590eefa` without rewriting history or deleting the reusable
  `codex/github-pages-release` branch.
- GitHub Pages run
  [34273709409](https://github.com/jacobfulfyll/presidential_profiles/actions/runs/34273709409)
  completed successfully for that exact merge SHA. Twenty representative Story, Summary,
  Compare, Explore, Profiles, Issues, Data Trust, Feedback, asset, JSON, and CSV URLs returned
  HTTPS 200 and matched the reviewed `docs/` bytes exactly. The public site is
  <https://jacobfulfyll.github.io/presidential_profiles/>.
- The release gate passed all 3,025 tests with 64 known warnings, the canonical 73-HTML/429-JSON
  build, 129 inline scripts, generated JavaScript, Python compilation, both rendered browser
  audits, the 60-state desktop lock, the 115-path staged release audit, credential scanning,
  protected-data comparison, whitespace checks, and Reggie Doctor with 0 errors and 0 warnings.
- No frozen paid annotation or governed derived-data artifact changed. Restricted annotation
  control-plane state, operator helpers, local telemetry, credentials, caches, and scratch files
  were not published.

## Latest addendum — Mobile experience v1

- All 73 canonical routes plus the self-contained Story now share a 320–760px phone contract while
  fine-pointer layouts at 768px and above retain their prior presentation and behavior. Phones use
  one 56px native-disclosure navigation row, safe-area gutters, 44px controls, readable type,
  sticky-stack anchor clearance, named local scrollers, visible focus, reduced-motion support, and
  forced-color boundaries. Short coarse-pointer landscape keeps the compact shell but makes every
  secondary rail static.
- Story/Summary cards and stages reflow; dense evidence remains readable in local viewports.
  Compare guarantees a 220px radar at 320px. Explore uses a 700px focusable graph viewport. Profile
  networks retain their reduced-node phone view through 760px on a centered 720px canvas. Issues
  and Data Trust enlarge controls and receipts; Era Choices uses a swipeable 720px matrix. Desktop
  resource loading, URLs, analytical schemas and values, navigation destinations, section order,
  and disclosure defaults did not change.
- Final acceptance is green: the inline build validates 73 HTML pages and 429 JSON shards; the
  dependency-free audit passes 74 physical pages × five viewports (370 observations) plus menu,
  no-JavaScript, rotation/landscape, touch, failure, reduced-motion, forced-color, and 200%-text
  scenarios. Sixty pre-edit desktop screenshot/geometry states pass the desktop lock after
  accounting for the existing Story/Summary reveal animation. All 3,025 repository tests pass with
  64 known warnings; 129 inline scripts, generated JavaScript, Python compilation, Data Trust's
  15-state browser audit, and whitespace checks pass.
- The durable contract and exact receipt are in `notes/mobile-experience-v1.md`. Frozen paid
  annotations retain fingerprint
  `9a8ac49ed29d63c94f4d7bdeb9f753cf0f38b45b8202ed7e7afa4de1c4297b54`; no governed data changed.
  Reggie Doctor reports 0 errors, 0 warnings, and 3 informational findings. The implementation was
  subsequently published through the release addendum above.

## Latest addendum — Data Trust GitHub Pages release

- The owner approved publication on 2026-09-08. Content PR
  [#6](https://github.com/jacobfulfyll/presidential_profiles/pull/6) merged release commit
  `a586faaa3df4af08fb8b4d8232ed021810545442` to `master` as
  `a92a1db1f2ab9274d8b9a1f8a1ef48141ec6b41b` without rewriting history or deleting the reusable
  `codex/github-pages-release` branch.
- GitHub Pages run
  [34251578173](https://github.com/jacobfulfyll/presidential_profiles/actions/runs/34251578173)
  completed successfully for that exact merge SHA. Nineteen representative Story, Summary,
  Compare, Explore, Profiles, Issues, Data Trust, asset, manifest, and CSV URLs returned HTTPS 200
  and matched the reviewed `docs/` bytes exactly. The public site is
  <https://jacobfulfyll.github.io/presidential_profiles/>.
- The release gate passed 3,018 tests with 64 known warnings, the canonical 73-HTML/429-JSON build,
  129 inline-script parsing, the five-page three-width/no-JavaScript browser audit, compilation,
  staged and unstaged whitespace checks, the 724-path staged release audit, a changed-text
  credential scan, frozen-artifact comparison, and Reggie Doctor with 0 errors and 0 warnings.
- `.claude/stats.json` was removed from Git and is ignored while the existing local copy remains
  preserved. Annotation-ledger state, `.codex/operators/`, restricted study identities/task plans,
  and other local control-plane files were not published. Frozen paid annotations remain unchanged
  at fingerprint `9a8ac49ed29d63c94f4d7bdeb9f753cf0f38b45b8202ed7e7afa4de1c4297b54`.

## Latest addendum — Data Trust redesign v1

- The five preserved Data routes now form one layered evidence system: `data-quality.html`,
  `methodology.html`, `era-boundaries.html`, `label-models.html`, and `metrics.html`. Each leads
  with a short verdict, continues through a visual explanation, and keeps technical evidence and
  downloads available. Shared local navigation covers Overview, Labeling, Evaluation,
  Populations, Model Comparison, Metric Dictionary, and Downloads.
- `src/presidential_profiles/quality_audit.py` owns the corrected deterministic audit rather than
  the page renderer. Its `data-quality-v2` public projection declares populations, units, schemas,
  semantic keys, source hashes, model/prompt receipts, validation results, and every chart
  treatment in `docs/data/quality/manifest_v2.json`. Existing compatible CSV URLs remain, while
  new population-ledger, field-by-era agreement, and chart-treatment exports make downstream
  denominators explicit.
- Methods now separates annotation completeness, taxonomy labelability, paired-evaluation
  availability, cross-model reproducibility, and the human validity that has not yet been measured.
  It includes a worked paragraph trace, a population passport, metric-family agreement panels,
  era/field evidence, and reproducibility cards. Coarse POS moved here under the source-hashed
  `grammar-pos-v2` contract with exact spaCy/model identity and stale-cache refusal.
- Era Choices now consumes `ERA_PROFILE_SPECS` as the sole public Story-era identity and anchor
  contract. `era-boundaries-v3` publishes the declared segmentation and clustering sensitivity
  conditions, support, objectives, partition similarity, signed drivers, historical receipts, and
  terminal-era right-censor status through an atomic versioned public projection while retaining
  compatible download URLs. The nine Story eras and Story boundaries themselves did not change.
- Model Comparison keeps the failed preregistered prediction prominent. The searchable Metric
  Dictionary groups measures by analytical question and supplies stable deep links used by
  substantive figures. Plotly-dependent trust charts were replaced or paired with semantic
  server-rendered evidence and accessible controls.
- `data-trust-validation-protocols-v1` persists an execution-blocked future human-validity study
  and paired decade-hidden sensitivity plan. Public output includes only the manifest,
  non-identifying sampling-cell summaries, and prompt definitions. Selected keys, coder tasks,
  and the request plan remain restricted. No human label, result, runtime model, current-price
  receipt, paid call, or production-label migration exists.
- The durable contract and verification sequence are recorded in
  `notes/data-trust-redesign-v1.md`; the full planned-study specification remains in
  `notes/data-trust-validation-protocols-v1.md`. Frozen `data/llm_annotations/` artifacts were not
  regenerated or hand-edited. The subsequently authorized publication is recorded in the release
  addendum above.
- Final acceptance is green: 106 focused trust/era/method tests and all 3,018 repository tests
  pass with 64 existing warnings; the final metric-target refinement passes 27 focused tests.
  The canonical build validates 73 HTML pages and 429 JSON shards, all 129 inline scripts and all
  generated JavaScript parse, Python compiles, every governed public projection revalidates, and
  `git diff --check` passes. The executable and independent browser reviews cover all five routes
  at 390, 768, and 1280 pixels plus JavaScript-disabled evidence, keyboard focus, contrast,
  sticky anchors, overflow, target sizing, and console diagnostics. Reggie Doctor reports 0
  errors, 0 warnings, and 4 informational findings. The frozen-annotation directory fingerprint
  remains `9a8ac49ed29d63c94f4d7bdeb9f753cf0f38b45b8202ed7e7afa4de1c4297b54`.

## Latest addendum — President Profiles and Navigation v1

- The shared Profiles/Data navbar now has a real pointer bridge, one hover/pinned state machine,
  no-JavaScript hover/focus access, 44px targets, and forced-colors treatment. The president
  directory remains in canonical chronology and now exposes one AI-topic preview plus progressive
  name-only search, with six eager and 39 lazy decorative portraits.
- All 45 profiles now follow Overview, Agenda, Rhetoric, Connections, Similarity, Signature
  speeches, and Evidence. Profile Plotly was replaced by semantic HTML/CSS bars. Legacy Evidence
  keeps its existing document-owner model and producer order while adding exact counts, selection
  bases, keyed receipts, actual-speaker/source-owner attribution, method labels, and limitations;
  it is the final content section and begins closed.
- Connections is a separately labeled actual-speaker population. Its topic ego is a
  field-preserving Level-1 projection from the accepted topic bundle with a reversible
  expanded/compact control and no focus/hover preview. Topic nodes in the SVG and their matching
  list controls now support click, Enter, Space, and Escape path emphasis; the former topic audit
  readout and complete focal Level-1 record no longer render. The governed
  `actual-speaker-invocation-network-v1` aggregation still contains 1,243 accepted rows and 303
  directed edges. One joined `Invoked` / `Invoked by` radio control now switches the same visual
  location between its two server-rendered top-five count lists; without enhancement both lists
  remain present. Browser code performs no analytical ranking or aggregation.
- `president-profile-context-v1` publishes exactly 49 files. The complete generated inventory is
  73 canonical/74 physical HTML files, 423 JSON files, and 129 inline scripts. Measured outputs
  remain within every contract budget; the largest profile is 111,875 raw / 22,482 gzip bytes,
  the enhancement module is 21,192 / 5,755 bytes, and the maximum module/index/shard expansion is
  1,130,152 / 132,998 bytes.
- The changed-surface suite passes 49 tests and the broader governed-profile suite passes 172.
  The full repository run passes 2,936 tests with
  the same 64 warnings. Canonical outputs remain byte-identical across canonical and inline
  builds; static JavaScript, Python compilation, and
  whitespace checks pass. Post-refinement browser QA on the local Lincoln profile confirms graph
  click/Enter/Space/Escape behavior, synchronized graph/list pressed state, radio click and arrow
  switching, the unenhanced two-list fallback, exact Similarity row alignment, no removed topic
  surfaces, and zero page-level overflow at 1280×720. The original four-viewport v1 matrix remains
  green for the underlying shell, expansion, failure/Retry, and request boundaries.
- `president-profile-v3`, protected Compare projections, Story, Summary, and Explore analyses
  remain unchanged apart from regenerated shared-navbar bytes. Profile Agenda, Rhetoric, and
  Similarity retain their existing document-owner denominators; this profile change introduces
  actual-speaker denominators only in Connections, while Compare retains its governed values.
  Invocations cover qualifying named former presidents in this corpus, use corpus speech dates for
  `target_status`, and do not establish historical absence, intent, influence, or causation.
- The work is uncommitted on `codex/summary-who-how-transition`. The original fourteen dirty,
  user-owned paths were preserved. Frozen paid annotations and existing governed artifacts were
  not regenerated or hand-edited. Nothing was staged, committed, pushed, deployed, switched, or
  cleaned. The next safe action is owner review of the local generated site and intended diff.

## Current status

- Last updated: 2026-09-08.
- Branch: reusable `codex/github-pages-release`, fast-forwarded through the published mobile
  content merge.
- State: Mobile experience v1 is implemented, verified, and published through GitHub Pages.

## Completed task

- Implement, verify, and publish the source-driven mobile experience refinement while preserving
  desktop behavior, protected Story contracts, frozen annotations, and governed-data provenance.

## Verification

- The mobile implementation and acceptance receipt is recorded in
  `notes/mobile-experience-v1.md`. All required tests, canonical generation, static checks,
  protected-input comparison, whole-site mobile inspection, desktop screenshot/geometry lock, and
  Reggie Doctor pass. Pages run 34273709409 succeeded for merge
  `67f8ff24a1df14727a1237403ea4e4f1d590eefa`; 20 representative public files match local bytes.

## Worktree

- The reviewed 115-path implementation and generated-output snapshot is committed and published.
  Only the post-deployment documentation receipt remains local until its follow-up pull request is
  merged. No unrelated user file was cleaned, reset, or overwritten.
- `.claude/stats.json` remains preserved as ignored user-owned local telemetry and is absent from
  Git. Frozen paid annotations and existing governed paid data were not regenerated or hand-edited.

## Decisions and blockers

- Story boundaries, `president-profile-v3`, protected Compare projections, and frozen paid
  annotations remain protected. Data Trust changes explain or project those inputs; they do not
  authorize a Story migration or label replacement.
- The owner authorized this GitHub Pages release only. Paid calls, human-label collection,
  restricted study execution, frozen-data changes, and Story-boundary changes remain unauthorized.

## Next safe action

- No mobile release action remains. Future site changes should follow
  `notes/github-pages-release-plan-v1.md` again.

## Latest addendum — Summary language boundary and divisiveness synthesis

- Summary Time's fixed America/United States boundary is now a compact three-view stage:
  `National name`, `Modal words`, and `Personal voice`. The first preserves the 1910 sustained
  crossover and its corpus examples. The second uses five disjoint surface families: Necessity
  (`must`, the complete `need / needs / needed / needing` family, and exact `have / has / had / got
  to` plus `gotta` phrases), Commitment/Intent (`will + shall` plus explicit speaker or
  administration promises, pledges, vows, guarantees, commitments, plans, resolutions, and
  statements of determination), the governed Absolute-emphasis family, Conditional (`would +
  could`), and Advice/Possibility (`should` plus the existing hedge family,
  including `may` and `might`); it explicitly refuses to recast them as a certainty/extremity
  index. The third shows the existing first-person singular and plural families and derives the
  2020s pooled rates (234 and 329 per 10,000 tokens), describing
  the recent reversal without presenting pronouns as individualism or divisiveness.
- The shared stage uses pressed-state buttons, live announcements, and Left/Right/Home/End keyboard
  movement through the existing chart-stage controller. Every line supports full-trajectory hover
  emphasis, click/tap pinning, and Escape reset; base widths are frozen before restyling so emphasis
  never compounds. Every language trajectory is now solid; distinct decade marker shapes preserve
  non-color identification for the five stance families, both pronoun families, and both national
  names. The United States line is no longer dashed, the I/me/my line is no longer dashed, and the
  national-name view now spans the complete 1789–2026 corpus timeline instead of the former
  1880–1932 context crop. Its existing five-year/20,000-word support gate yields 228 supported
  center years and 10 honest gaps; no missing value is joined or imputed. The stance families'
  238 center years are already finite, so no values are imputed or connected across missing
  observations. Modal-family and pronoun views use centered seven-year windows and a 10,000-word
  floor, preserving all 238 center years without `connectgaps`. All three views now use the same
  compact point hover: one short whole-line definition, the exact center-year rate, and the
  contributing presidents; window definitions and support floors remain in the permanent evidence
  disclosure.
  National naming retains its governed five-year crossover rule. Each view keeps its generated
  exact hover; narrow screens contain a 760px chart within the local horizontal scroller with no
  page overflow.
  Both new charts reuse the registered `rate_10k` metric and existing artifacts. Necessity's
  complete surface family and explicit commitment/intent additions are counted deterministically
  from `speeches.parquet`; the absolute-emphasis line reuses `speech_markers.parquet`. The ambiguous,
  modern-spoken-heavy `going to` construction is intentionally excluded. No browser-side measure or
  governed artifact was added.
- A derived three-card synthesis now answers “is presidential rhetoric more divisive now?” with
  the narrow verdict: more openly partisan, yes; more divisive in every broader sense, no. It
  stands behind the annual-message partisan-attack ratio versus Civil War & Reconstruction
  (4.8x, 95% speech-clustered bootstrap 2.3–14.3x), labels the pronoun result descriptive, and
  heavily caveats the modest 0.315 detrended Civil War-era nearest-neighbor analogy. The
  enemy-naming 1.09x [0.79, 1.61] and zero-sum 1.76x [0.95, 4.88] founding comparisons remain
  visible because both include one; all three ratio intervals retain the 10-present-speech
  low-cluster caution.
- Verification: 35 focused Summary/Story tests and all 2,820 repository tests pass
  under the required x86_64 environment with the existing 64 warnings. The canonical build
  validates 73 HTML pages and 376 JSON shards; all 128 generated inline scripts parse, compilation
  and `git diff --check` pass, and desktop/mobile browser QA reports no diagnostics or page-level
  overflow. Nothing was deployed, staged, committed, pushed, switched, or cleaned.

## Latest addendum — Summary Who/How/Register transition refinement

- On 2026-09-04 the owner asked for a clearer transition from Summary's WHO and
  HOW communication bars into the legal/procedural × hype president field, and
  for the dense Voice-era controls to become one dropdown.
- Voice now reads as one explicit `Who + how → register` sequence across the
  same nine governed Story eras: Congress → public, written → spoken/broadcast,
  and a less-procedural opening → procedural nineteenth-century middle → higher
  hype later. The permanent copy states that these patterns coincide in the
  corpus and do not establish that audience or delivery caused the vocabulary
  change.
- Voice now offers one two-choice `By president` / `Over time` switch. The
  default preserves the existing 45-president portrait field. `Over time`
  plots one owner-directed legal/procedural ÷ hype line across the existing
  centered five-year all-corpus rates. The logarithmic axis keeps the positive
  ratio range readable. All 238 center years remain in the generated figure and ordinary
  line hover retains both component rates; 228 supported windows render, the existing
  20,000-word support floor remains authoritative, a missing or zero hype
  denominator is suppressed, and gaps stay unknown rather than becoming zero.
- After owner review, the time view removes the six historical-event lines and
  all three administration-transition lines. Hope/Doom retains its separate
  event context. This register chart now has seven direct year labels:
  the 1827 Adams-heavy rise, 1863 Civil War-era drop, 1881 series maximum, 1944
  wartime low, 1966 policy-heavy rise, 2016 mid-2010s decline, and 2023 recent
  rebound. Direct year-label hover shows only a concise corpus-composition summary,
  left-aligned across short lines. At the owner's direction, the permanent `Selected turns in
  the line` guide and both register text-alternative tables are removed; ordinary line hover
  retains the exact ratio and component rates.
- Conflict target mix now shows nation, group, person, institution, and other together as five
  lines on one shared percentage chart instead of five vertically stacked panels. Distinct
  colors, dash patterns, and marker shapes preserve non-color identification. Native All/category
  buttons and direct line hover emphasize a complete trajectory without hiding the others;
  click/tap pins, keyboard focus previews, and Escape resets. Six direct callouts derive the
  largest adjacent-era composition explanations from the governed counts, including cases where
  the denominator rather than the category count drives the visual rise. Their hover summaries
  now name the leading raw entity labels and exact occurrences verified against frozen
  `paragraph_entities.parquet` joined to eligible `paragraph_view_v1.parquet` keys. Point hover preserves
  exact shares and counts. At the owner's direction, the target-mix text-alternative table is removed.
  The category-definition and denominator note remains immediately below the chart as a native
  disclosure that starts closed and expands without JavaScript.
- The separate zero-sum × partisan × enemy-naming portrait now uses the governed
  `annual_message_strict` treatment rather than all available speech genres. Its 42 plotted
  presidents therefore share the State of the Union/annual-message genre; William Harrison,
  James A. Garfield, and Harry S. Truman remain unsupported in the governed 45-row contract because
  the corpus has no qualifying message. The all-eligible speaker population remains the validated parity source
  for the separate target-mix chart. Every plotted portrait now has a visible border using the
  target chart's category color for the largest adversarial-entity count in that president's same
  qualifying messages. Four exact ties use a neutral border; the compact key and hover carry the
  textual category or tie label. Borders are now a uniform seven pixels. Portrait
  hover has bold labels and omits raw flag/entity counts while retaining exact rates, encoding
  explanations, support, and qualifying-message coverage. Coverage is not equal: the 42 supported
  presidents contribute 1–9 qualifying messages (median 4); the three unsupported rows contribute 0.
  Narrow-screen portrait taps keep the complete tooltip visible by adjusting only the chart's
  existing horizontal scroll position.
  At the owner's direction, the portrait exact table is removed. Its inner header now contains
  only the chart title plus two lines defining portrait area and the common paragraph denominator;
  the coverage eyebrow, enemy-naming range, position sentence, x/y shorthand, and tie note are absent.
  The subsection now uses the reader-facing `BY PRESIDENT · COMPARABLE SPEECHES` kicker,
  `How presidents frame conflict` title, and one-sentence annual-message qualifier instead of the
  former implementation-oriented shared-genre wording.
- One native `Emphasize an era` select offers All eras, Dim all, and exactly one
  of the nine chronological Story eras. Every president remains plotted and
  the choice changes emphasis only. The select is disabled in `Over time`,
  where the full chronology and nine quiet era bands remain visible; its prior
  choice is restored on return to `By president`. The view is transient and
  adds no URL state. Summary Time retains its existing multi-era controls.
- Source and generated tests lock the bridge wording, two-view switch,
  11-option selector, removal of Voice phase/card controls, preservation of
  every plotted era, guarded ratio semantics, honest gaps, exactly seven movement
  annotations, removal of the register alternatives, the one-chart conflict target layout,
  six target-line controls, six count-derived target callouts, and removal of the target-mix
  alternative. The canonical build validates 73 HTML pages and 376 JSON
  shards; 67 focused tests pass, including frozen named-entity receipts; all 128 inline scripts
  and generated external scripts parse; Python compilation and whitespace checks pass. The most
  recent full repository run, before the hover-copy-only refinement, passed all 2,814 tests with
  the same 64 existing warnings.
- Browser QA at 1280×720, 1024×768, 768×1024, and 390×844 confirms the select's
  All/Dim/single behavior, mouse and arrow-key view switching, disabled-state
  announcement, era restoration, stacked narrow bridge, visible focus,
  intended chart-local scrolling, zero page-level overflow, and an empty error console.
  The same four-width matrix confirms direct target-line hover emphasis, focus preview,
  click/tap pinning, Escape reset, 44-pixel narrow controls, callout hover summaries, and
  removal of the target-mix alternative.
  No governed data, public data
  artifact, or schema changed. The generator now owns and registers the derived
  `legal_hype_ratio`; browser code does not calculate it.
- The common-genre portrait refinement preserves the canonical 73-HTML/376-JSON build. The
  focused Summary/Story/metric/site suite now passes 67 tests; static JavaScript and Python
  compilation remain clean. Browser QA at 1280×720 and 390×844 confirms 42 rendered portraits,
  Carter's 0% annual-message partisan-attack value, three unsupported contract rows, chart-local
  narrow-screen scrolling, zero page overflow, and an empty error console. The same matrix
  confirms all 42 portraits carry category-colored rings, the six-item textual key wraps without
  overflow, tied leaders remain neutral, and hover names every border meaning.
- The work remains uncommitted on `codex/summary-who-how-transition`; no push or
  deployment occurred. The implementation contract remains
  `notes/summary-redesign-v2.md` with this owner-directed refinement recorded in
  its Voice decision and acceptance checks.

## Latest addendum — Final Release Program Steps 1–6 integration

- On 2026-09-04 the owner approved preserving the completed Story,
  actual-speaker/reference foundation, reusable topic network, Summary Topics,
  Compare, and Explore work as one cohesive local default-branch checkpoint.
- These surfaces were implemented and verified together in the repository's
  single shared worktree. They are intentionally integrated rather than
  reconstructed as separate page branches after the fact.
- The release boundary includes the governed `data/reference_entities/`,
  `data/speaker_views/`, and `data/speaker_topic_network/` artifacts plus the
  canonical generated `docs/` tree. The local `data/annotation_ledger/`,
  `.codex/operators/`, and `.claude/stats.json` telemetry remain outside the
  release commit.
- The last code-changing verification remains 2,811 passing repository tests
  with 64 existing warnings, a canonical 73-HTML/376-JSON build, parsed inline
  and external JavaScript, clean compilation and whitespace checks, and the
  completed four-viewport browser matrix recorded below. This integration step
  changes Git ownership only; it does not change application code or generated
  output.
- No remote push or deployment is authorized by this checkpoint. The next safe
  action is owner review of the integrated local site before choosing Step 7 or
  requesting publication.

## Latest addendum — Explore interaction and evidence v1

- Explore now follows `Choose series → Trends → Data and methods`. The default
  is exact `tariff`, `freedom`, and `border`; uncertainty is on, context is
  none, disclosures are closed, and the ordered selection limit is six.
- Series discovery uses one searchable governed catalog: 16 deterministic
  broad issues, 50 AI topics nested under 17 taxonomy parents, and three
  case-sensitive acronyms. Search uses published terms and AND-prefix matching;
  current chips and a persistent count replace a nondeterministic recent list.
- Exact versus audited word-family counting is a primary pre-entry choice and
  a reversible per-chip action. Grouped chips name the canonical family, state
  that grouping is on, give the form count, and disclose the exact existing
  algorithmic forms without implying linguistic equivalence.
- Explore no longer loads Plotly. The safe-DOM SVG renderer provides whole-line
  pointer/focus highlighting, A–F letter/color/dash/shape redundancy, exact
  period inspection, arrow/Home/End navigation, Enter/Space/click/tap pinning,
  Escape, live keyboard announcements, reduced motion, forced-colors support,
  and four-width responsive reflow with no page overflow.
- The main-flow uncertainty toggle preserves period `ci_status`, cell
  `interval_unresolvable`, null-as-unknown semantics, markers, exact text, and
  method-specific `ci_components` when ranges are hidden. The old combined-word
  and chart-options state is retired; one governed historical-context select
  remains presentation-only.
- Auditability is now one prominent 38-field selected CSV, one collapsed
  five-column/one-series exact table, permanent compact unit/provenance copy,
  and Methods/Data Quality links. No-JavaScript and injected load failures keep
  a semantic default SVG, 708 exact rows, three static downloads, honest
  disabled controls, and retry/dismiss messaging.
- `explore_projection.py` publishes a deterministic atomic 263-file v2
  contract: 236 lexical periods; 4,599 families / 7,391 resolver entries; four
  64-shard exact/group unigram/bigram namespaces; 1,234 field-preserving topic
  rows; and three CSVs. The manifest seals inputs, schemas, hashes, row counts,
  sizes, routing, budgets, and exact inventory. Two builds must be byte-equal;
  the final projection is 78,370,786 raw / 19,312,004 gzip bytes and its largest
  shard is 522,404 / 125,031, below every hard budget.
- Cold default loading is four public data requests and seven successful
  resources, 803,827 raw / 201,243 gzip-equivalent bytes—87.4% less compressed
  than the prior Plotly path. Final page/assets are 201,824 raw bytes combined.
- Verification is green: 31 dedicated Explore tests, 396 focused/related tests,
  and all 2,811 repository tests pass with 64 existing warnings. The canonical
  build validates 73 HTML pages and 376 JSON files/shards; all 128 executable
  inline scripts and generated external JavaScript parse; compilation,
  whitespace, display-name security, metrics, deterministic publication,
  loading/failure, exact/download, URL/history, focus-return, console, and
  1280×720 / 1024×768 / 768×1024 / 390×844 browser checks pass.
- Story, Summary 05, Compare, Profile V3, Issues V2, governed analytical and
  frozen paid artifacts, and unrelated worktree changes were not redesigned or
  cleaned. No deployment, staging, commit, or push occurred. Proposed-order
  Step 7 is next only after owner approval.
- The full contract and completion receipt are in
  `notes/explore-interaction-and-evidence-v1.md`.

## Latest addendum — Compare agenda visual system v1

- Compare retains the Rhetoric → Agenda → Evidence narrative and always-present
  A/B/C native selectors. Selection state uses only `a`, `b`, and optional
  `c`; obsolete `agenda` and `topic` parameters canonicalize away.
- Agenda now has one focused view: the governed-order union of each selected
  president’s top three Level 1 AI topics by existing
  `speaker_paragraph_share`. There is no all-topic toggle, fine-topic UI,
  CorEx disclosure/download, or method bridge.
- Each reason badge names the president or presidents whose top-three set caused
  the row to appear. Exact controls visibly pair slots with collision-safe names
  such as `A · Lincoln`, `B · F.D. Roosevelt`, and `C · Washington`.
  Vertically staggered circle/square/diamond marks preserve overlaps.
- Selecting one president value opens a blue detail directly beneath that topic
  row with the exact share, topic-paragraph numerator, eligible-paragraph
  denominator, topic-bearing appearance count, support state, and exactly one
  deterministic broad-topic example. The parent shard is used only to retrieve
  that broad receipt; Level 2 rows never render.
- The projection schema and deterministic 22-file inventory remain unchanged
  for provenance and compatibility. The visible page uses only Level 1 cells
  and links only the full AI-topic CSV. Profile V3, Story, Summary, and governed
  data remain unchanged.
- The evidence area remains comparison-wide. Adversary/invocation receipt lists
  and vocabulary z-scores remain omitted from the visible page.
- Verification: 53 focused Compare/integration tests and all 2,784 repository
  tests pass (64 existing warnings). The canonical build validates 73 HTML pages
  and 224 JSON shards; both generated JavaScript assets parse. Compare is
  155,518 bytes raw / 23,254 gzip. Live three-president browser checks passed at
  1280×720, 1024×768, 768×1024, and 390×844 with zero guarded overflow and
  targets above 44px. The tested detail contained the exact
  7.09% / 47 / 663 / 7 relationship and one receipt; no fine rows, CorEx
  surfaces, or console errors were present.
- The local preview remains available at
  `http://127.0.0.1:8010/compare.html?a=abraham-lincoln&b=franklin-d-roosevelt&c=george-washington#rhetoric`.
  No deployment, commit, push, or unrelated worktree cleanup occurred.
- The durable contract and full receipt are in
  `notes/compare-agenda-visual-system-v1.md`.
## Latest addendum — Summary topic gravity field v1

- Summary 05 now opens as an owner-directed topic field rather than the earlier
  president/topic ego browser. Users choose one to eight broad AI topics;
  selected topics stay fixed around the perimeter and president portraits
  settle inside them using the existing `speaker_paragraph_share` as their
  only gravitational weight. Deterministic display stretch and a fixed
  collision pass separate nearby portraits; absolute distance has no separate
  meaning, and there is no continuous or stochastic simulation.
- President portrait area shows the sum of `topic_paragraph_count` over the
  selected topics. The interface names this `selected-topic paragraph
  memberships`, states that it is not a unique-paragraph total, and repeats the
  multi-label non-additivity limitation. Faint line width continues to encode
  only `speaker_paragraph_share`.
- The default field includes the 42 supported presidents when connected.
  Desktop can draw all qualifying presidents; 390px caps only the diagram at 18.
  Users may optionally choose up to eight
  supported presidents to compare; the portrait-area scale remains fixed to the
  full qualifying field rather than renormalizing the subset.
- A visible size key names portrait area and its largest current membership
  total. Pointer, keyboard focus, and pinning all update a compact stage readout
  with president, membership count, strongest topic, exact share, and connected
  lines. `Choose topics`, its count, and Clear share one compact heading line;
  the topics occupy one horizontal scroller. Compare presidents is one line,
  with Show all presidents and selected-president chips in a second horizontal
  scroller only when selections exist. The graph takes the full content width,
  and the bottom exact relationship readout is removed.
- Summary now exposes only the supported recurring view: 42 selectable
  presidents and 427 president/topic relationships meeting the 20-paragraph,
  five-appearance floor. The observed/thin reveal, thin president choices,
  Level 2/evidence interaction, and the HTML exact-value table no longer render.
  The governed all-edge bundle remains unchanged. A deterministic 427-row CSV
  containing only these supported recurring relationships is now the section's
  sole download, bringing the public Summary projection to 20 files.
- Selected topics are canonicalized into frozen taxonomy display order before
  layout, so clicking the same set in any order produces identical anchors and
  president coordinates. The inline topic/president descriptions, recurring-key
  paragraph, live-status text, taxonomy-provenance card, `How to read` disclosure,
  embedded exact table, and former Plan 3/Data Quality/Methods/Feedback link row
  are removed. The graph now ends with one exact recurring-topic CSV link.
- The accepted Plan 3 bundle remains unchanged; the Summary projection is now
  20 files because its manifest owns the new filtered CSV.
- The former Extreme Speeches evidence-card appendix is removed from Summary's
  section inventory and generator path. Topics is now the final section. Its
  download sits 8px above the existing source/method footer on desktop and
  mobile instead of inheriting a full chapter gap; the footer divider remains.
  Legacy `index.html#records_appendix` links now land on Summary Topics.
  All 183 focused/related and 2,773 repository tests pass with 64 warnings.
  The canonical build validates 73 HTML / 204 JSON; source
  audits, projection reproducibility/parity, compilation, 130 inline scripts,
  the renderer asset, whitespace, desktop/390px overflow, interaction, and
  console checks pass. No deployment, commit, push, schema change, frozen-data
  edit, or unrelated cleanup occurred.

## Latest addendum — Story reference landscape v1

- The Era Profile top-left card is now a visibly headed four-lane
  `Distinctive era references` card for people, institutions, groups or
  communities, and nations or places. `Distinctive` means unusually
  concentrated here, not exclusive to this era. It shows exact era and
  all-corpus percentages without an explanatory deck or comparison bars, and
  may expose one compact, keyed evidence receipt per lane. The percentages and
  names form two scan columns;
  a quiet vertical rule separates a flat shaded name/evidence column from the
  percentages. The right side has no rounded container or inset accent, and
  `AI only` appears once rather than as a repeated icon-plus-label. Selected
  names are centered and enlarged above one small paragraph/document/source
  metadata line.
- Highlights require five paragraphs, two source documents, positive era
  concentration, and at least 80 percent favorable/neutral primary-AI stance;
  displayed Named adversaries are excluded. NER supplies only evidence-span
  agreement, never stance or NER-only promotion.
- When no supported candidate exists, a four-paragraph candidate may appear as
  a visibly labeled `Limited record` only if every other gate passes. This
  surfaces American Legion for 1913–1932 (four paragraphs across four source
  documents) without lowering the five-paragraph supported-highlight floor.
- Named adversary rows, counts, types, bubble packing, legend, and interaction
  are unchanged. The new compact card and Named adversaries measure the same
  closed desktop height in all nine eras. The visible reference card contains
  none of the former Jeffreys/log-odds, ranking-policy, source-agreement essay,
  or download-footer text.
- Corpus Footprint now assigns its spare lower-row height to the three primary
  statistic tiles and the three distinctive-word tiles in equal shares. Both
  rows now have matching box heights, with their contents vertically centered,
  so the card finishes with Major Topics instead of leaving an internal gap.
- Era Profiles advance to `era-profile-v6`; v9 visualizations and v11
  contextualizations remain unchanged. The governed 45-row distinctive
  artifact and v5 projection field remain intact for foundation and public
  download parity, but no longer render in the profile.
- Verification is green: 83 focused/related tests and all 2,773 repository
  tests pass with 69 existing warnings. The canonical build validates 73 HTML
  pages and 204 JSON files; foundation, Story v6/v9/v11, Plan 3, Summary
  projection, compilation, 130 inline scripts, static JavaScript, and
  whitespace checks pass. Browser QA confirms equal 230px closed desktop
  card heights in all nine eras, functional evidence, and zero page overflow
  at desktop and 390px-class mobile width.
- The contract is `notes/story-reference-landscape-v1.md`. No frozen paid
  artifact, foundation Parquet, speaker attribution, adversary analysis,
  deployment, commit, push, or unrelated cleanup is part of this change.

## Latest addendum — Summary actual-speaker topic relationships v1

- Summary now uses `05 · Topics — Presidents and their recurring topics` in
  place of the former Audit chapter. Its initial 17-topic overview draws no
  edges; president/topic selection opens a deterministic ego view, while the
  ranked list and complete server-rendered matrix retain exact values. Extreme
  Speeches remains the post-story appendix, and compact Plan 3 download, Data
  Quality, Methods, and Feedback links remain.
- `summary_topic_network.py` derives a field-preserving all-corpus projection
  only from the accepted `SpeakerTopicNetworkBundle`. The index contains all 45
  presidents, 17 Level 1 topics, 700 observed edges, 42 supported presidents,
  three thin presidents, and 427 default-visible edges. Seventeen shards cover
  all 50 Level 2 topics and their Plan 3 evidence exactly once. No speaker,
  topic, denominator, share, support, era, or receipt is recalculated.
- `docs/data/summary-topic-network/` contains the compact index, 17 on-demand
  topic shards, and self-hashed manifest. The index is 363,779 bytes raw / 58,553
  gzip; the largest shard is 819,063 / 209,031; the renderer is under 29 KB raw.
  All are below the locked budgets. Two independent builds produce the same 19
  files byte-for-byte, and publication validation compares every selected row
  and value back to Plan 3.
- The page-agnostic renderer uses `speaker_paragraph_share` as its only edge-
  width encoding. It supports ARIA tabs, native selectors, pointer/focus
  preview parity, Enter/Space/click pinning, three-step Escape, explicit thin
  reveal, Level 2 return, non-modal deterministic audit examples with focus
  restoration, reduced motion, failure fallbacks, and safe DOM construction
  without data-driven `innerHTML`. Topic mode caps only the diagram at 12;
  mobile caps it at six while retaining the complete list.
- Verification is green: 12 focused tests, 105 related tests, and all 2,771
  repository tests pass with 69 existing warnings. The canonical build
  validates 73 HTML pages and 204 JSON files; 130 inline scripts and both static
  renderer assets parse. Foundation, Story v5/v9/v11, and Plan 3 publication
  checks pass. Browser QA at 1280×720, 1024×768, 768×1024, 390×844, and
  `?motion=reduce` confirms stable layouts, correct interaction/focus state,
  44px targets, zero page overflow, and an empty console. The no-JavaScript
  route is covered by the server-rendered topic links, policies, downloads, and
  exact table tests.
- The accepted implementation contract is
  `notes/summary-actual-speaker-topic-network-plan-v1.md`. No deployment,
  staging, commit, push, Plan 3 schema change, frozen paid-artifact edit, or
  unrelated worktree cleanup occurred.

## Latest addendum — Actual-speaker topic network v1

- `data/speaker_topic_network/` now publishes the reusable
  `actual-speaker-topic-network-v1` contract. Its typed reader resolves the
  annotation ledger's active pointer, verifies the generation artifact root,
  requires the complete 35,394-key one-to-one paragraph join before filtering,
  and reuses actual-speaker, appearance, and Story-era identity from the
  accepted foundation. Document ownership never determines president edges.
- The governed census is 32,531 eligible paragraph rows, 91,119 distinct
  multi-label memberships, 45 president and 67 topic nodes, 450 president/scope
  support rows, 670 topic/scope support rows, 4,408 observed corpus/Story-era
  edges, and 11,576 keyed first/lower-middle/last evidence receipts. All pinned
  support, level, default-visible, topic-free, normalization, and cross-owner
  acceptances match exactly.
- Edges publish `speaker_paragraph_share`, `topic_contribution_share`, and
  `topic_scope_share` with explicit paragraph/appearance numerators and
  denominators. Topic-free eligible rows stay in denominators; multi-label
  shares are non-additive. Thin records remain downloadable. No confidence,
  similarity, influence, importance, causality, layout, or generic weight is
  manufactured.
- `docs/data/speaker-topic-network/` contains validated network JSON, edge and
  evidence CSVs, membership Parquet, and a self-hashed manifest. The canonical
  site validates the accepted bundle before writes and publishes downloads, but
  zero HTML pages consume or render the network. Story's generated hash remains
  unchanged from the pre-task capture; Plan 2 schemas remain v5/v9/v11.
- Two independent governed/public builds produce the same 15 files byte for
  byte. The 37 focused tests, 565 related tests, and all 2,759 repository tests
  pass with 69 existing warnings. Compilation, 73-HTML/185-JSON canonical
  validation, Story/foundation/network audits, 130 inline scripts, static JS,
  protected-input checks, and Reggie Doctor pass. No deployment, staging,
  commit, push, adapter installation, or unrelated cleanup was performed.
- The durable contract and Plan 4 consumer rules are in
  `notes/story-actual-speaker-topic-network-v1.md`. Plan 4 may consider an
  all-corpus Level 1 Summary view, but it must load this bundle and must not
  independently recompute topics, speakers, appearances, eras, denominators,
  support, or evidence.

## Latest addendum — Story reference-entity migration v1

- Story now loads one audited `StoryFoundationBundle` before generated-site
  writes and passes it to the era profile, visualization, contextualization,
  Founding compatibility data, and public-download producers. Public contracts
  are `era-profile-v5`, `era-visualizations-v9`, and
  `era-contextualizations-v11`.
- All nine profiles replace the blocked constituency surface with five native
  `Distinctive references` disclosures. The 45 rows preserve exact evidence,
  full speaker/owner identity, nullable primary-AI stance, 29 `AI + NER` and
  16 `AI only` source-agreement receipts, and governed Miller Center links.
  President strips use 1,054 actual-speaker source appearances and visibly
  mark cross-owner evidence.
- President agendas, appearance-based communication forms, adversary edges,
  president-level contextualization values, and 1,243 retained Era Echoes
  paths use eligible actual-speaker evidence. Aggregate chronology, topic,
  vocabulary, style, and footprint measures remain explicitly labeled
  source-document corpus quantities. The Echoes overlay reports 48 unresolved,
  156 ineligible, and 27 reassigned rows from 1,447 source rows.
- `docs/data/story/` publishes a 45-row distinctive CSV, compact 1,054-row
  appearance CSV, byte-identical full entity Parquet, and hashed manifest.
  The fast Story receipt validates public JSON/download parity. Frozen paid
  annotations and sealed speaker attribution remain unchanged; no deployment,
  staging, commit, push, or unrelated cleanup was performed.
- The requested 99-test focused suite and all 2,725 repository tests pass with
  69 existing warnings. The canonical build validates 73 HTML pages and 183
  JSON shards; 130 inline scripts parse; desktop/mobile, keyboard/focus,
  reduced-motion, no-overflow, source-link, and console browser checks pass.
- The implementation contract is
  `notes/story-reference-entity-migration-v1.md`. The former
  `publish-era-constituency-categories` task is superseded, not completed, and
  its historical evidence remains untouched.
- The pre-existing `.claude/stats.json`, `.codex/operators/`,
  `data/annotation_ledger/`, foundation inputs, and other dirty-worktree changes
  remain user-owned. This migration neither cleaned nor reclassified them.

## Latest addendum — Speaker/reference foundation v1

- `data/speaker_views/paragraph_view_v1.parquet` now joins the corrected corpus,
  source metadata, and governed attribution one-to-one on
  `(doc_name, para_idx)`. It retains all 35,394 rows, marks 32,531
  actual-president paragraphs eligible, preserves 2,863 exclusions with
  reasons, and identifies 296 cross-owner presidential paragraphs.
  `appearances_v1.parquet` assembles those eligible rows into 1,054
  `(doc_name, attributed_speaker_profile_id)` texts without speaker mixing;
  `coverage_v1.parquet` and `consumer_inventory_v1.json` provide population and
  migration receipts.
- The pinned local `spaCy 3.8.14` / `en_core_web_sm 3.8.0` pass preserves
  199,604 raw spans over every retained paragraph.
  `entity_mentions_v1.parquet` holds 186,093 paragraph-local normalized hybrid
  rows with conservative safe aliases, explicit `AI + NER` / `AI only` /
  `NER only` source state, and nullable AI stance. Historically distinct
  entities remain unmerged; NER corroborates a mention, not stance or history.
  The partial second model is restricted to Data Quality and compares its
  exact canonical overlap of 260 documents / 8,438 paragraphs.
- `era_distinctive_v1.parquet` publishes five primary-AI candidates for each of
  the nine governed Story eras. It de-duplicates by normalized entity and
  paragraph, uses every eligible era paragraph as the denominator, applies
  Jeffreys-smoothed log odds against the other eras, enforces five paragraphs
  across two documents, and carries source-agreement counts plus keyed evidence.
  NER-only discoveries stay in the data layer. Public JSON and generated
  `docs/` were not changed.
- `arch -x86_64 .venv/bin/python -m presidential_profiles.foundation_audit
  --check` passes the 35,394 / 32,531 / 2,863 / 296 populations, matching
  fingerprints and protected inventories, all 45 evidence-backed era rows,
  and the 151-paragraph Carter–Reagan split (44 Carter, 42 Reagan, 22
  non-president, 43 mixed, zero cross-contamination). Eight new focused tests
  and the 107-test broader speaker/foundation/combat/era/site-validation set
  pass. The complete repository suite passes all 2,704 tests with 69 existing
  warnings. Two isolated full rebuilds emit the same 12 files byte-for-byte.
- The compact governed contract is
  `notes/speaker-reference-foundation-v1.md`; the acceptance report is
  `data/reference_entities/acceptance_report_v1.json`. No API call, paid
  regeneration, frozen or sealed input edit, site build, deployment, staging,
  commit, push, branch switch, or unrelated worktree cleanup occurred.

## Latest addendum — GitHub Pages deployment (2026-08-22)

- Pull request [#4](https://github.com/jacobfulfyll/presidential_profiles/pull/4)
  published the current generator, governed speaker-attribution/conflict
  artifacts, and canonical `docs/` tree through the existing `master:/docs`
  source. Release commit `72b0149c76fbfe7c6af45d9bca26febbd9e0e0d9`
  merged to `master` as `5413f36824d2093a84fa7d8f3870e5d50f57d084`.
- GitHub Pages run
  [32578012035](https://github.com/jacobfulfyll/presidential_profiles/actions/runs/32578012035)
  completed its build, status, and deployment jobs successfully. The sole
  annotation is GitHub's upstream Node.js 20 action-runtime deprecation notice.
  The public site is <https://jacobfulfyll.github.io/presidential_profiles/>.
- Release verification is green: 116 focused tests and all 2,696 tests pass
  with 70 existing warnings; the canonical builder validates 73 HTML pages and
  182 JSON shards; all 130 generated inline scripts parse; Python compilation,
  whitespace, frozen-paid-artifact, staged-tree, and credential-pattern checks
  pass. The staged audit accepts 245 changed paths and a 384-file,
  86,477,738-byte Pages tree.
- Fourteen representative live Story, Summary, Compare, Explore, Profiles,
  Issues, data/methodology, asset, JSON, and CSV URLs return HTTP 200 with the
  expected content types. Live Compare HTML and Security & Peace JSON match the
  validated release byte-for-byte.
- The runtime-receipt test no longer depends on the active Codex session's
  reasoning level; a synthetic rollout fixture now preserves the exact
  model/effort parsing contract. Frozen paid annotations were not changed.
  The 53,981-file local `data/annotation_ledger/`, three `.codex/operators/`
  helpers, and live `.claude/stats.json` telemetry remain user-owned, local,
  and outside the public release.

## Latest addendum — Graph-first Compare

- Compare restores the two visual fingerprints requested by the owner: an
  eight-axis corpus-derived percentile radar and a separate six-axis
  AI-labeled percentile radar. They now occupy one fixed graph stage selected
  by accessible Corpus-derived / AI-labeled tabs above the chart. Pointer plus
  Left/Right/Home/End keyboard switching updates the visible semantic tab panel
  without moving its document position or changing its dimensions. Both use
  the bundled local Plotly runtime, preserve A/B/C color plus dash/marker
  distinctions, and keep every exact value, unit, and rank state in a native
  semantic table disclosure.
- Thin records remain selected and visible but do not draw unsupported radar
  shapes; their exact tables continue to show `N/A · Insufficient record`.
  Two- and three-president selection, swap/add/remove/reset, normalized URLs,
  hash and Back/Forward restoration, no-JS defaults, and versioned on-demand
  Profile V3 downloads are unchanged.
- Nearest-neighbor context is now five independent visual lens boards. Each
  selected-president row shows three linked portrait tiles with rank, exact
  cosine value, and a 0–1 meter; the instrument definition is collapsed behind
  `What this lens uses`, and no composite similarity score is created.
- Evidence & Data now leads with selected-record speech, word, and paragraph
  footprint bars, followed by compact adversary/invocation meters, vocabulary
  chips, signature-speech tiles, and collapsed source excerpts. The stale
  pending-audit sentence is replaced with the current population distinction:
  profile evidence follows document ownership, while the separate Conflict
  treatment uses the completed speaker-attribution audit.
- Verification: 12 focused Compare tests and 60 broader Compare/profile/site/
  security tests pass. The canonical builder validates 73 HTML pages and 182
  JSON shards; all 130 generated inline scripts parse. The full suite records
  2,695 passes, 69 existing warnings, and the known unrelated runtime-receipt
  failure because that test hard-codes `low` while this session reports
  `xhigh`.
- Browser QA covers the default pair, three presidents, a thin record, the
  ordinary app viewport, and 390×844. Pointer and keyboard tab switching render
  two traces per radar in the same document position and equal-height panel;
  both tabs remain 52px tall at 390px, and responsive reflow has zero page
  overflow or console warnings/errors. The local preview remains at
  `http://127.0.0.1:8010/compare.html`; nothing was deployed, staged, committed,
  pushed, switched, or cleaned, and frozen paid annotations were not changed.

## Latest addendum — Tomorrow and yesterday paired views

- Summary Time now pairs one all-president portrait field with a separate
  `Over time` view. The president field keeps future-family matches per 10,000
  marker words on x and nostalgia-family matches per 10,000 marker words on y,
  but every portrait is the same size. The time view draws Tomorrow and Yesterday
  as separate solid/dashed trailing four-year rolling averages of annual rates,
  plotted at each window's ending year; it does not create a ratio or calculate
  a browser-side measure. Each point requires coverage in all four years and at
  least 10,000 total marker words in the window. The current corpus has 235
  complete windows from 1789–1792 through 2023–2026; 227 pass both guards.
- `build_summary_temporal_president_contract` is now
  `summary-temporal-president-v2`. It joins `speech_markers.parquet` to the corpus
  speech table one-to-one on `doc_name`, rejects key or metadata drift, and pools
  exact future, nostalgia, and word counts before dividing. The removed
  self-reference dimension no longer loads `speech_stats.parquet` into this
  contract. Its document-owner, all-genre treatment is explicitly not
  speaker-audited.
- The interaction now matches Voice: By president / Over time buttons and one
  native All eras / Dim all / single-era selector. The selector changes emphasis
  only, disables with an explicit live status in the time view, and restores its
  prior value on return. All 45 presidents remain in fixed succession; the 42
  observed and three below-five-speech records remain plotted with ordinary and
  amber support halos. Exact values and support remain in hover. The former size
  key, self-reference copy, and text-alternative table are removed.
- The standalone founding annual-message dictionary audit has been removed from
  the reader flow at the owner's direction. No frozen annotations or governed
  derived artifacts were changed.
- Verification: the canonical build validates 73 HTML pages and 376 JSON
  shards; 68 focused Summary/Story/metric/site tests pass; all 128 generated
  inline scripts parse; compilation and `git diff --check` pass. Browser QA at
  1280×720, 1024×768, 768×1024, and 390×844 confirms the switch, disabled/live
  timeline state, restored Civil War emphasis, internal chart scrolling, and
  zero page-level overflow. Browser diagnostics are empty. Reggie Doctor reports
  0 errors, 0 warnings, and 4 informational findings. Nothing was deployed,
  staged, committed, pushed, switched, or cleaned.

## Latest addendum — Conflict target mix at era grain

- Summary Conflict now separates its two questions by grain. The target view
  uses `target_mix_by_era_speaker_audited_v1.parquet` and pools
  `speaker_audited_all` adversarial mentions within the nine canonical
  `trends.ERAS` reporting bands. It does not average president percentages or
  annual percentages.
- Nation, group, person, institution, and other each occupy one of five
  vertically aligned small-multiple trajectories. Every panel shares the same
  percentage scale; exact percentages print at all era points, and hover plus a
  collapsed era table retain counts, shares, contributing speeches, and support.
  The five supported shares for an era total 100%.
- A visible guide defines every category, describes `Other` as a heterogeneous
  residual, and states that mentions—not paragraphs—form the denominator. It
  also discloses unequal era length, corpus density, and genre mix, and says the
  connecting lines guide the eye rather than estimate intervening years.
- The standalone Enemy naming, Zero-sum framing, and Partisan attack lollipops
  are removed from generation, page markup, figure registration, metric
  registration, and tests. The joint portrait remains president-grain through
  `president-conflict-v2`: zero-sum is on x, partisan attack is on y, and enemy
  naming is encoded by portrait area.
- Final code, build, JavaScript, and browser verification belongs to the active
  implementation turn. This source-document synchronization did not regenerate
  `docs/`, deploy, stage, commit, push, switch branches, or clean unrelated
  user-owned worktree changes.

## Latest addendum — Issues Page V2

- The Issues directory and all 16 detail routes now render from one validated
  `IssuePageViewModel` contract shared by summaries, charts, exact tables,
  public JSON, and three tidy CSV exports per issue. Membership, canonical
  order, slugs, finite values, keyed evidence, and active-corpus provenance
  fail closed before any output is replaced.
- Every headline and selected highest-period excerpt uses the highest five-year
  band whose `ci_status` is `ok`. Public rates are explicitly labeled `share of
  eligible paragraphs`; sampling intervals, caution states, and unresolved
  intervals remain distinct. Each issue carries early, highest-supported, and
  recent corpus excerpts with `(doc_name, para_idx)` receipts and canonical
  Miller Center links.
- President emphasis uses an absolute 0–100% scale and the shared five-speech
  eligibility rule. Fine AI topics remain a separately labeled exploratory
  many-to-many crosswalk with 10/20-year requested and observed bounds,
  numerator, denominator, partial-period state, low-support flags, stable topic
  colors, Auto/Grouped/Heatmap modes, Select all/Clear, and lazy Plotly startup.
  Education and Money retain explicit cross-method caveats.
- Semantic index cards, one-H1 detail pages, skip/local/adjacent navigation,
  chart regions, exact-value disclosures, keyboard-visible focus, touch-sized
  controls, no-JavaScript disabled state, and live chart-failure fallbacks are
  in place. Shared `issues-v2.css` and `issues-v2.js` replace duplicated page
  assets; a manifest-scoped cleanup removes only stale outputs previously owned
  by the Issues generator.
- Verification: 161 focused tests pass; the canonical build validates 73 HTML
  pages and 182 JSON shards; all 130 inline scripts and the external Issues
  runtime parse; compilation, `git diff --check`, and frozen-artifact checks
  pass. The full run records 2,676 passes and one unrelated runtime-receipt
  failure because that test hard-codes `low` while this requested session runs
  at `ultra`. Final Reggie Doctor reports 0 errors, 0 warnings, and 4
  informational findings for the optional portable document, existing
  telemetry/worktree state, and native external-pipeline layout.
- Browser QA covered the directory plus Security, Religion, Health, Foreign,
  Education, and Money at 1280px, 768px, 390px, 320px, and a 200%-reflow
  equivalent. It verified lazy charts, chart-mode and selection controls,
  keyboard focus, internal scroll containment, zero page overflow, and both
  missing-page-script and missing-Plotly fallbacks. Nothing was deployed,
  staged, committed, or pushed, and no frozen paid artifact was changed.

## Historical addendum — Compare Page V2 baseline (superseded)

- Compare is now a server-rendered, mobile-first workspace for two or three
  presidents. The no-parameter view remains Abraham Lincoln / Franklin D.
  Roosevelt; labeled A/B/C selectors, support cards, swap, third-president,
  copy-link, reset, URL normalization, hash preservation, and Back/Forward
  restoration all use the existing canonical 45-president order.
- `compare_site.py` projects a finite compact payload from normalized
  `president-profile-v3` views rather than copying nearly complete public
  profiles. Shared catalogs define all eight corpus and six AI measures plus
  the full agenda taxonomies; per-president records carry aligned values,
  bounded evidence, support state, and the five separately governed neighbor
  instruments.
- At this checkpoint Plotly and the radar charts were removed from Compare.
  The graph-first addendum above now supersedes that presentation. Semantic
  matrices expose
  exact values, units, eligible-president percentiles, scoped headers,
  captions, non-color A/B/C markers, and `N/A · Insufficient record` for thin
  rankings. Narrow screens reflow rows into labeled cards without horizontal
  chart scrolling.
- Agenda rows are deterministic unions of the full canonical domain, fine
  topic, and legacy-issue arrays with the requested top-N caps, maximum-value
  ordering, canonical tie breaks, signed era differences, valid zeros,
  extended/show-all disclosures, and separate proposal/values and speech-type
  composition tables.
- Nearest-neighbor context kept all five instruments independent and did not
  imply scores between selected presidents. Evidence was grouped by president,
  retained source/exploratory status, and disclosed the mixed-speaker audit as
  pending at that time; the later speaker-attribution and graph-first addenda
  record the completed audit and current population distinction.
- Download now fetches only the selected canonical public president shards on
  demand and emits an ordered `president-comparison-v2` bundle with source URLs,
  source paths, positions, and a slugged filename. Preparing, success, and
  error states are announced accessibly.
- Verification: the requested focused suite passes 79 tests; the canonical
  build validates 73 HTML pages and 165 JSON shards; all 146 inline scripts
  parse; `git diff --check` passes. The full run records 2,665 passes and one
  unrelated runtime-receipt failure because that test hard-codes `low` while
  this user-requested session correctly reports `ultra`; it also reports the
  existing 70 warnings.
- Browser QA covered 1280×720, 1024px, 768px, 390×844, 320px, and a
  200%-equivalent viewport; two/three presidents, long and thin records,
  invalid/duplicate/partial URLs, controls, Back/Forward, download requests,
  live names/status, focus, contrast, reduced-motion CSS, overflow, and console
  output were checked. The browser harness isolated clipboard readback and did
  not expose the Blob download event, but the page announced both operations,
  the download fetched exactly the selected canonical shards, and the static
  contracts are covered by tests.
- At that checkpoint generated `docs/compare.html` was 346,507 bytes
  uncompressed and contained no Plotly request. Final Reggie Doctor reported 0 errors, 0 warnings, and 4
  informational findings for the optional portable document, existing
  telemetry/worktree state, and native external-pipeline layout. No frozen paid
  artifact was changed; nothing was deployed, staged, committed, or pushed.
  Work remains on `codex/github-pages-release`, and the extensive unrelated
  user-owned source, data, generated-site, telemetry, and planning changes were
  left in place.

## Latest addendum — President speaker attribution and Conflict v2 completed

- The governed speaker workflow audited all 1,057 source documents while
  retaining the canonical 1,053-speech / 35,394-paragraph production universe.
  The primary visible Codex task labeled every document and all 12,458
  paragraphs in 207 review-required documents. Its exact runtime receipt is
  `gpt-5.6-sol` / `low`.
- A separate fresh user-visible Codex task captured its own exact
  `gpt-5.6-sol` / `low` receipt before labeling and completed the locked blind
  queue: 110 assignments and 754 targets, including every cross-owner credit
  and shared/uncertain target plus deterministic marginal samples. It did not
  inspect primary labels. The comparison found 288 speaker-membership
  disagreements across 52 documents; transcript-evidence adjudication selected
  the reviewer result for all 288 and left zero unresolved.
- `data/speaker_attribution/document_attribution_v1.parquet` contains 1,057
  unique documents and `paragraph_attribution_v1.parquet` contains exactly
  35,394 unique `(doc_name, para_idx)` rows. The final outcomes are 32,531
  canonical-president, 1,678 multiple-speaker, 1,036 non-president, 135 shared,
  and 14 scaffolding paragraphs. Excluded outcomes carry no president profile
  and contribute no paragraph or entity evidence to the primary treatment.
- The additive speaker run
  `speaker-attribution-v1-primary-20260806` is the only newly sealed run. Its
  2,840-member artifact-set hash is
  `sha256:ddc5ef7a3b77b9535e8453cf6764e4ceaef40034798a613f3299d749ca255846`.
  Existing sealed runs, old registries, frozen paid annotations, correction
  inputs, and the annotation-ledger active pointer remain byte-identical.
- `data/combat/by_president_treatments_v2.parquet` publishes all five declared
  45-president treatments; `by_president_speaker_audited_v2.parquet` is the
  selected 45-row Summary payload. Both rebuild byte-identically and declare
  `president-conflict-v2`. The legacy `by_president.parquet` remains unchanged.
  Cross-owner debate turns automatically credit Gerald Ford, Richard M. Nixon,
  and Ronald Reagan; non-president and mixed turns fail closed.
- Summary now reads the v2 selected payload, names “Speaker-audited all eligible
  paragraphs,” shows support states, and contains no pending-audit language.
  The canonical build validates 73 HTML pages and 182 JSON shards. Static
  JavaScript parsing passes for 29 inline scripts and the generated external
  script; `git diff --check` passes.
- Verification: focused speaker/Summary/site tests pass 44 tests; the full suite
  passes 2,684 tests with 70 existing warnings. In-app browser QA at the default
  desktop viewport and 390×844 confirms all four charts, readable rows, zero
  horizontal overflow, and no console errors. Reggie Doctor reports 0 errors,
  0 warnings, and 4 informational findings for the optional portable guide,
  existing telemetry/dirty-worktree ownership, and native external layout.
- Nothing was deployed, staged, committed, pushed, switched, or cleaned. Every
  unrelated pre-existing worktree change remains user-owned. Python generators,
  not generated `docs/`, were the editing surface.

## Historical addendum — President speaker attribution was groomed, not executed

- `notes/president-speaker-attribution-plan-v1.md` and the groomed
  `audit-president-speaker-populations-and-version-conflict-chart-inputs`
  backlog entry now define the document census, paragraph rubric, interactive
  chat execution boundary, review pass, population treatments, v2 producer,
  refusal rules, tests, and site acceptance gates.
- The source inventory retains all 1,057 document records, but production
  attribution and `president-conflict-v2` use the corrected canonical
  1,053-speech / 35,394-paragraph universe. The stale 4,562-paragraph estimate
  is planning evidence only; the execution task must derive its exact scope.
- All model judgments must be supplied through user-visible Codex chats and the
  local offline assignment/ingest workflow. Provider APIs, API keys,
  `pp-annotate`, the headless Codex campaign worker, and hidden labeling
  sessions are explicitly out of scope.
- Clear turns by another one of the 45 profiled presidents are reassigned to
  that speaker's profile—even in a transcript hosted by someone else and even
  when the turn predates their presidency. Non-presidents are retained as
  evidence but excluded from president metrics; mixed turns fail closed.
- This checkpoint changed planning and routing documents only. No document or
  paragraph was labeled, no run was opened or sealed, no source/data/site file
  was rebuilt, and no frozen paid artifact, registry, ledger pointer,
  deployment, staging, commit, push, or branch was changed.
- The current `president-conflict-v1` payload remains document-owned and must
  continue to say `speaker attribution audit pending` until the v2 producer and
  its full verification gates pass.

## Latest addendum — Evidence-first president profiles

- All 45 individual profiles now follow one guided order: identity and corpus
  support, Overview, Agenda, Rhetoric, Evidence, five independent Similarity
  instruments, and Signature speeches. The hero carries the portrait, full
  display name, party, corpus record span, correctly pluralized speech count,
  support state, compact back link, and a president-aware Compare action.
- `profiles_site.py` now builds one normalized profile view model for HTML,
  embedded figure data, the exact `president-profile-v3` public JSON contract,
  and Compare. Production generation validates exact 45-president membership,
  keyed identity, ordered measures and units, support state, and finite JSON.
- Percentile eligibility is shared at five corpus speeches. Thin presidents are
  excluded from the reference distribution and receive null ranks in both
  legacy and AI measures; the HTML and Compare render `N/A · Insufficient
  record` rather than a midpoint. The thin warning appears before all analysis.
- The former profile radar charts are direct-label horizontal bars with exact
  accessible tables; six aligned AI-topic rows show share and era difference;
  evidence excerpts precede rate-only disclosures; and signature speeches are
  canonical Miller Center links followed by previous/next-president navigation.
- Shared formatters correct 11th/12th/13th and other ordinals, speech
  singular/plural, full display names, and Miller Center URL normalization.
  Compare consumes the real v3 producer and omits unsupported percentile shapes.
- Normal text and focus contrast meet the updated AA palette, chart names and
  textual summaries are explicit, native disclosures expose visible 3px focus,
  and mobile tables replace hidden/unmounted charts below 600px. Browser QA of
  the directory, Franklin D. Roosevelt, William Henry Harrison, and thin
  Compare at 1280×720 and 390×844 found zero page overflow or clipped primary
  labels and no console warnings or errors.
- Verification is green: the requested focused suite passes 98 tests; the full
  suite passes 2,638 tests with 70 existing warnings; the canonical build
  validates 73 HTML pages and 165 JSON shards; and all 146 inline scripts parse.
  Final Reggie Doctor reports 0 errors, 0 warnings, and 4 informational findings
  for the optional portable document, existing telemetry/worktree state, and
  native external-pipeline layout. No frozen paid annotation was changed, and
  nothing was deployed or committed.
- The repository has no governed presidential term-date artifact. To avoid
  presenting corpus years as tenure, profiles explicitly label them `Corpus
  record span` and explain that they are available speech years, not term dates.

## Historical addendum — President Conflict comparison (superseded)

- Summary Conflict now presents four full-width Plotly graphs, each containing all 45 presidents
  in the same fixed chronology. The first graph is a 100% stacked target composition—nation,
  group, person, institution, and other—so each president's adversarial mentions can be compared
  without mixing their denominator with paragraph rates.
- Enemy naming, zero-sum framing, and partisan attack each receive a separate lollipop graph.
  Those three graphs use the same president order and one shared percentage scale, making both
  within-measure peaks and across-measure magnitudes directly comparable without a combined
  legend. Exact percentages, counts, denominators, and support warnings remain available on hover.
- Low paragraph support and low adversarial-mention support remain distinct. Missing audited
  values render explicit `N/A` states rather than zero, and the target composition is suppressed
  when there is no eligible adversarial-entity denominator.
- All four figures, exact value labels, all-president medians, and support states rebuild from the validated, replaceable
  `president-conflict-v1` payload in `data/combat/by_president.parquet`. The contract preserves
  exact president membership and chronology while validating counts, rates, target shares,
  missingness, treatment metadata, and support states; no president-specific chart correction is
  required when the producer changes.
- The current payload is still document-owned and remains labeled `Document-owned corpus ·
  speaker attribution audit pending`. The groomed task and
  `notes/president-speaker-attribution-plan-v1.md` own the deterministic
  document/speaker inventory and corrected producer. Frozen paid annotations
  were not regenerated or edited, and this addendum does not claim that the
  pending audit is complete.

## Latest addendum — GitHub Pages deployment

- `notes/github-pages-release-plan-v1.md` is the executed publication contract.
  Pull request [#2](https://github.com/jacobfulfyll/presidential_profiles/pull/2)
  merged the reviewed site-content release to `master` at
  `b3c2e7567979d1ad5c72c178890fe6b0bafa43e2` without rewriting the accumulated
  local history.
- GitHub Pages run
  [30867907537](https://github.com/jacobfulfyll/presidential_profiles/actions/runs/30867907537)
  built and deployed that exact merge successfully from `master:/docs`. The
  public site is
  <https://jacobfulfyll.github.io/presidential_profiles/>. The sole workflow
  annotation is GitHub's upstream Node.js 20 action-runtime deprecation notice;
  every build, status, and deployment job passed.
- Public HTTPS checks return 200 with the expected content types for Story,
  Summary, Compare, Explore, Profiles, Data Quality, Methodology, the bundled
  Plotly asset, representative JSON, and representative CSV. Desktop and
  390×844 browser QA confirms correct active navigation, zero horizontal
  overflow, all eight Summary plots, working era-phase controls, the V2
  `Hope divided by doom` chapter, and no console warnings or errors.
- Release verification is green: 2,596 tests pass with 70 existing warnings;
  the canonical builder validates 73 HTML pages and 165 JSON shards; all 146
  generated inline scripts parse; the staged audit accepts 561 release paths,
  349 `docs/` files, and an 83,242,907-byte Pages tree.
- The deployed boundary excludes the user-owned local control plane: 51,137
  untracked files under `data/annotation_ledger/`, three files under
  `.codex/operators/`, and the live `.claude/stats.json` telemetry change.
  Frozen paid annotations were not regenerated or edited during this release.
- `DECISIONS.md` records why this repository retains the native `master:/docs`
  Pages source and why the annotation control plane remains outside the public
  site release. This documentation-only closure record does not change the
  deployed `docs/` tree identified above.

## Latest addendum — America in Summary V2

- `notes/summary-redesign-v2.md` supersedes V1's visible narrative. Summary now
  has five chapters: Voice, Conflict, Time, Emotional register, and Audit.
- The post–Civil War breadth graph and issue-lifecycle heat map are removed.
  Visible copy states that most topic-count expansion occurs before 1860 and
  later movement is small, uneven, and method-sensitive.
- Voice now uses two independent 100% stacked-bar charts, eliminating the mixed legend.
  WHO keeps Congress/general public/specific groups/other audiences; HOW keeps
  written/spoken/radio-TV/press-conference-or-debate. Each graph has nine era bars, a
  graph-specific palette and emoji-accented legend, selective in-bar labels, exact hover
  receipts, and its own Measure/Evidence disclosure. Both
  president views use six phase presets and nine colored era cards while retaining all 45
  presidents.
- Enemy identity is displayed as nation/group/person/institution/other across
  every era. The observed annual-message combat lines no longer show confidence
  whiskers. A new president scatter compares enemy naming with partisan attack.
- The founding annual-message nostalgia audit reports 17 dictionary matches:
  10 `again`, 7 `restore*`, and 0 `back to`, with an explicit warning that this
  is not evidence of golden-age nostalgia.
- The emotional ratio is now NRC Hope divided by Doom on a log scale with the
  20,000-word and nonzero-denominator guards. The separate-axis weather map is
  no longer part of Summary. The generated page contains nine Plotly charts, including the two
  separate communication charts.
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

## Historical snapshot — Second-era Story implementation

The locked second-era Story plan is implemented in the Python generator and rebuilt static site.
The 1816–1849 chapter now uses a fixed-period topic matrix with an aligned adversary-naming strip,
artifact-derived prose, named source receipts, uncertainty status, and a 1850–1854 next-era
preview. The founding-era redesign and every later chapter remain in place. The local preview is
running at `http://127.0.0.1:8010/`. Nothing was deployed.

### Active task

No active task is claimed.

### Branch

`master`; no branch, staging, commit, push, pull request, or deployment was created for this work.

### Delivered behavior

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

### Artifacts and documentation

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

### Verification

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

### Worktree ownership

Every modification and untracked path present at the initial scan remains user-owned, including
all pre-existing `.claude`, source, tests, data, generated-site, dependency, and documentation
changes. This task made narrow source/test/documentation edits, added the expansion-story module,
tests, V4 note, and derived layer, and regenerated the site through its builder. It did not clean,
restore, reformat, stage, or overwrite unrelated work. `.claude/stats.json` remains existing tool
telemetry and is not part of this task.

### Decisions and limitations

- Python under `src/presidential_profiles/` remains the source of truth; `docs/` is generated.
- Frozen paid annotations were read but never regenerated or hand-edited.
- Matrix and strip values describe this curated presidential-speech corpus, not public opinion,
  policy effect, reception, or causal impact.
- Topic rows overlap by design. The territorial parent and its subrows are not additive.
- The 1850–54 column is a preview on unchanged scales, not part of the chapter chronology.
- Use Reggie documentation mode. No Reggie task or generic adapter was created.
- Preview only at `http://127.0.0.1:8010/`; public deployment remains out of scope.
- No blocker remains.

### Next safe action

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
