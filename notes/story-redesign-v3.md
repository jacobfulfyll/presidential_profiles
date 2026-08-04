# Story Page V3 narrative and interaction contract

Status: completed and verified 2026-07-23. Story Page V2 remains the frozen completed baseline
in `notes/story-redesign-v2.md`; this note governs only the V3 refinement.

## Purpose and evidence boundary

V3 makes the nine historical chapters read as one chronological argument, gives every chapter a
data-derived takeaway, and standardizes how a reader moves between a chapter's focused evidence and
the full record. It does not reopen V2's artifact choices or registered analyses. Every historical
event is context, never a causal estimate.

Unless a registered receipt is named, results are descriptive. CorEx paragraph labels, declared
word families, Flesch–Kincaid scores, and AI-assigned primary forms remain instruments with stated
limitations. The Miller Center corpus is a curated formal record through April 2026, not an
exhaustive record of presidential communication, policy, opinion, or outcomes.

## Shared interaction contract

- Each substantive chart is followed by two compact, separately styled native disclosures:
  `Measure · <actual measure>` and `Evidence · <named artifact>`. Measure expands to definition,
  unit, calculation, example, and limitation. Evidence expands to source, statistical status,
  denominator/support, downloads, and caveats. Neither contains a nested disclosure.
- Historical time-series comparisons use the same two-button labels: `Era view` and
  `Full history`. The chapter claim remains the default. Scale and denominator changes are
  announced in visible text and an `aria-live` region.
- Events are named on the chart and repeated as always-visible accessible callouts. Hover provides
  the fuller label; keyboard and mobile readers do not need hover. The shared note says events
  orient time and do not establish causation. Numbered marker lookup guides are not used.
- The chronology rail and sticky progress both expose `date range · era name · section title`.
  Normal scrolling is preserved. Scroll reveals are subtle progressive enhancement; manual
  controls remain complete. Reduced motion removes automatic reveals and transitions.
- Every interaction is keyboard operable, visibly focused, screen-reader labeled, responsive at
  390 CSS pixels, and encoded with text/symbol/line treatment in addition to color. Charts may
  scroll locally; the page may not overflow horizontally.

## Chapter evidence and acceptance map

### 1. 1789–1815 — The written republic

- **Artifact:** `data/speeches.parquet`,
  `data/llm_annotations/speech_annotations.parquet`, `data/speech_markers.parquet`.
- **Transformation and denominator:** count the one assigned primary form over all 73 corpus
  speeches dated 1789–1815; legal/procedural matches are divided by all word tokens and scaled per
  10,000. The full-history state groups the assigned forms by decade and uses the centered
  five-year word-weighted marker rate.
- **Takeaway:** derived at render time: 62 of 73 founding-era speeches are assigned written
  messages, while all four presidents record legal/procedural rates above 22 per 10,000 words.
- **Status:** descriptive corpus composition, descriptive dictionary rate.
- **Views and events:** Era view compares the four presidents; Full history shows the later
  written-to-performed transition. No event claim is needed.
- **Hover/focus, accessibility, mobile:** speech counts, shares, word support, form limitation,
  patterned bars, direct text summary, compact local scroll.
- **Acceptance:** the 62/73 sentence is computed from keyed artifacts; the form is explicitly one
  AI-assigned primary form rather than a channel inventory.

### 2. 1816–1849 — Expansion and federal conflict

- **Artifact:** `data/paragraph_issues.parquet`.
- **Transformation and denominator:** five-year period share of all paragraphs carrying each
  CorEx issue, minus the same issue's 1789–1815 paragraph share, in percentage points.
- **Takeaway:** foreign policy is highest at the era's opening and closing periods, while
  money/banking moves from below its founding baseline in the 1820s to +22.94 points in the
  1836-centered period. The pattern changes direction rather than rising steadily.
- **Status:** descriptive CorEx labels; no inference and no causal reading.
- **Views and events:** one shared-scale issue heatmap/small-multiple view. Monroe Doctrine
  (1823), tariff/nullification context (1828–33), Bank War (1832), and territorial expansion
  (1845–48) are labeled in place.
- **Hover/focus, accessibility, mobile:** period level and baseline delta, symbols/direct labels,
  visible event callouts, table fallback.
- **Acceptance:** no numbered guide; the three issues remain tied to the founding baseline; prose
  is generated from the plotted values.

### 3. 1850–1877 — Crisis of union

- **Artifact:** `data/speech_markers.parquet`.
- **Transformation and denominator:** raw fear matches divided by raw hope matches within each
  centered five-year word-supported window. The arithmetic mean of the 19 available 1794–1815
  display windows is index 100; raw fear÷hope remains in hover/focus. Windows below the existing
  20,000-word display floor remain absent.
- **Takeaway:** the crisis-era maximum is index 135.94 in 1860, or 35.94% above the declared
  founding-window mean.
- **Status:** descriptive NRC dictionary ratio and derived index; not a stable causal percentage.
- **Views and events:** Era view is 1850–1877 with the founding baseline visible; Full history is
  1789–2026. War of 1812 and Civil War labels orient the comparison.
- **Hover/focus, accessibility, mobile:** index, raw ratio, baseline definition, event callouts,
  explicit full-history scale announcement.
- **Acceptance:** baseline definition is visible; 135.94 is re-derived in tests; no old
  early-scale/full-scale two-story staging remains.

### 4. Civil War and Reconstruction rights language

- **Artifact:** `data/paragraph_issues.parquet`, `data/paragraphs.parquet`,
  `data/bands.parquet`.
- **Transformation and denominator:** Era view uses annual labeled paragraphs divided by all
  paragraphs and carries distinct speech, paragraph, and word support. Full history uses the
  existing five-year speech-clustered CorEx band layer and its `ci_status`.
- **Takeaway:** attention is sharply uneven inside the era; the view is designed to show annual
  spikes and thin support rather than compress them into a smooth emancipation story.
- **Status:** descriptive CorEx label; full-history intervals are sampling-only. Speech attention
  does not measure rights realized.
- **Views and events:** Era view and Full history; Emancipation (1863), 13th (1865), 14th (1868),
  15th (1870), and Civil Rights Act of 1875 are labeled directly.
- **Hover/focus, accessibility, mobile:** annual share plus support counts; marker size/opacity
  indicates support; unsupported years are gaps; callouts repeat the events and full label
  definition.
- **Acceptance:** no deceptive connection across unsupported years; no numbered guide; broader
  view reads `ci_status` and does not coerce missing bounds.

### 5. 1878–1900 — The procedural presidency

- **Artifact:** `data/speech_markers.parquet`, `data/speech_stats.parquet`,
  `data/speeches.parquet`, local president portraits.
- **Transformation and denominator:** president-level declared marker matches per 10,000 words;
  presidents are assigned to one of the nine `trends.ERAS` by the midpoint of their corpus years.
- **Takeaway:** Gilded Age presidents occupy the high-procedural/low-hype side of a stable
  two-axis comparison; the vocabulary shift is not a competence, productivity, policy-depth, or
  effectiveness ranking.
- **Status:** descriptive word-family rates.
- **Interaction:** all nine named eras are native multi-select checkboxes. Every president and
  portrait remains visible; non-selected eras fade, selected eras receive an outline, and axes
  never move.
- **Hover/focus, accessibility, mobile:** name, corpus years, both rates, era, and speech support;
  labeled table alternative prevents face-only recognition.
- **Acceptance:** nine checkboxes, multi-select, 45 persistent faces, stable axes, non-color
  selection cue.

### 6. 1901–1932 — Regulation, world war, and crash

- **Artifact:** `data/paragraph_issues.parquet`.
- **Transformation and denominator:** paragraph shares for 1901–10, 1911–20, 1921–30, and
  1931–32 minus the 1878–1900 paragraph-share baseline, in percentage points.
- **Takeaway:** the four-window heatmap shows changing agenda handoffs: economy/jobs rises in
  every window, war peaks in 1911–20, and money/banking turns from below baseline to above it in
  1931–32.
- **Status:** descriptive CorEx labels.
- **Views and events:** one comparable heatmap replaces the cumulative disconnected-point reveal;
  U.S. entry into World War I (1917), the 1929 crash, and Smoot–Hawley (1930) are direct context.
- **Hover/focus, accessibility, mobile:** exact delta and period share, signed text in every cell,
  event callouts, table fallback.
- **Acceptance:** all four windows are present simultaneously on one scale; the old four-stage
  plot is absent.

### 7. “America” overtakes “the United States”

- **Artifact:** `data/speech_markers.parquet`, `data/speeches.parquet`.
- **Transformation and denominator:** `America`, `American`, and `Americans` versus the exact
  phrase `United States`, each per 10,000 words in a centered five-year rolling sum. A sustained
  crossover is five consecutive displayed windows with the America family higher; 1910 is the
  first qualifying window.
- **Takeaway:** the formal corpus begins a sustained lexical crossover in the 1910 window. This may
  suggest a change in public national naming; it cannot prove a change in national identity.
- **Status:** descriptive dictionary rates.
- **Views and examples:** 1880–1932 transition chart plus short, escaped excerpts selected from
  local 1908–1912 speeches and linked to their source pages.
- **Hover/focus, accessibility, mobile:** exact rate, term-family definition, rolling-window
  explanation, source titles/years, corpus/genre limitation.
- **Acceptance:** the crossover is derived as 1910; every quotation is found verbatim in its local
  transcript.

### 8. 1933–1945 — Recovery becomes mobilization

- **Artifact:** `data/paragraph_issues.parquet`, `data/speeches.parquet`.
- **Transformation and denominator:** topic paragraph shares for 1933–39 and 1940–45; endpoint
  labels show `(later / earlier - 1) × 100`. All six displayed earlier shares exceed 4%, so the
  percentage changes are stable enough to interpret; exact endpoint shares remain available.
- **Takeaway:** war/military rises 442.31%, while economy/jobs, agriculture, and money/banking
  fall 64.55%, 69.41%, and 81.90%. This is an agenda handoff, not program effectiveness.
- **Status:** descriptive CorEx labels.
- **Views and events:** two distinct phase markers; connector direction uses color plus arrow/
  symbol treatment. Topic-specific Plotly symbols supply secondary pictograms.
- **Hover/focus, accessibility, mobile:** both shares, signed percentage change, topic text, the two
  verified Roosevelt excerpts and source links.
- **Acceptance:** directions and percentage labels are data-derived; icons are never the only
  identifier.

### 9. 1946–1988 — From written record to performed presidency

- **Artifact:** `data/llm_annotations/speech_annotations.parquet`,
  `data/speech_stats.parquet`.
- **Transformation and denominator:** decade share of all corpus speeches assigned
  `written_message` versus any performed form; detailed assigned forms stay in hover. Supporting
  reading level is the median speech score with the existing centered rolling context.
- **Takeaway:** written messages fall from 80.6% of sampled 1900s speeches to 5.3% in the 1950s,
  0.8% in the 1960s, and 0% in both the 1970s and 1980s. The durable shift is written to
  performed, not proof that the mutually exclusive `broadcast_radio_or_tv` label dominates.
- **Status:** descriptive AI factual taxonomy and descriptive Flesch–Kincaid.
- **Views and events:** written/performed transition is primary; reading level is visually
  subordinate.
- **Hover/focus, accessibility, mobile:** decade speech counts and detailed form shares; compact
  `Measure · One assigned primary form per speech` limitation.
- **Acceptance:** chapter title changes; no visible “Taxonomy limit” audit box; copy rejects the
  inference that lower grade level means intellectually simpler communication.

### 10. 1989–2016 — The always-on presidency

- **Artifact:** `data/coverage_pressure/coverage_pressure.parquet`,
  `data/coverage_pressure/inference_receipts.parquet`,
  `data/paragraphs.parquet`, frozen paragraph annotations.
- **Transformation and denominator:** real 1894 and 2016 annual-message examples show speech words,
  effective topics, and length-biased equivalent topic-episode words together. Registered
  estimates use the preregistered annual-message sample and president→speech bootstrap.
- **Takeaway:** the illustrative 2016 message is 6,051 words versus 15,890 in 1894, with 19.37
  versus 14.43 effective topics and 191.85 versus 531.03 equivalent depth words. Confirmatory
  inference remains the registered +2.166 breadth and −303.568 depth comparison.
- **Status:** two-speech comparison is illustrative/descriptive; registered breadth and depth are
  confirmatory; sensitivities are exploratory.
- **Interaction and accessibility:** manual real-message/breadth/depth/sensitivity stages, arrow
  keys, reduced-motion branch, real paragraph strips, explicit tradeoff language.
- **Acceptance:** length, breadth, and depth appear together; breadth is not called good and depth
  is not called bad; all receipts and sensitivities remain visible.

### 11. 2017–2026 — Platform-era intensification

- **Artifact:** `data/speech_markers.parquet`.
- **Transformation and denominator:** centered five-year word-weighted rates per 10,000 in both
  views; the April 2026 endpoint also exposes the unsmoothed four-speech denominator (17,177 words).
- **Takeaway:** the chapter is a continuation from legal/procedural toward promotional and
  opponent-centered wording, not the origin of platform communication. The misleading
  `Named opponents` label becomes `Opponent-centered wording`.
- **Status:** descriptive word-family rates. The 2026 endpoint is provisional and incomplete.
- **Views and events:** Era view begins in 2001 to show the approach to the 2017 boundary; Full
  history begins in 1789. The 2017 line is an intensification boundary, not a causal estimate.
- **Hover/focus, accessibility, mobile:** raw and smoothed 2026 rates, speech/word support, direct
  endpoint label `through April 2026 · provisional`, live view announcement.
- **Acceptance:** opponent-centered wording names generic groups and media as well as opponents;
  the 2026 raw opponent rate (8.15) and smoothed rate (6.33) are re-derived in tests.

### 12. Synthesis — hype versus doom

- **Artifact:** `data/register/trends.parquet`.
- **Transformation and denominator:** raw era-level word-family rates per 10,000 words on the nine
  named `trends.ERAS`; no smoothing across era boundaries.
- **Takeaway:** the present era's hype rate is 20.309 versus the next-highest 7.433, while its doom
  rate is 8.656 versus 8.863 in War & New Deal. The present is uniquely hype-heavy, not uniquely
  doom-heavy.
- **Status:** descriptive raw-corpus era comparison.
- **View:** direct-labeled chronological trail, start/present emphasis, median-defined background
  regions that describe measured intensity without calling any region good or bad.
- **Hover/focus, accessibility, mobile:** rates, change from previous era, distinct symbols,
  selectable detail through native focus, and a complete table fallback.
- **Acceptance:** values and superlatives are derived from the artifact; chronology remains
  visually separated from synthesis and the appendix.

## Historical context receipts

V2 receipts remain valid for Monroe Doctrine, territorial expansion, tariff/nullification,
Bank War, Emancipation, the Reconstruction Amendments, and the Civil Rights Act of 1875. V3 adds
these authoritative receipts:

- [U.S. Senate, Declaration of War with Germany, WWI](https://www.senate.gov/about/images/documents/sjres1-wwi-germany.htm):
  Congress approved the declaration on April 6, 1917.
- [Federal Reserve History, Stock Market Crash of 1929](https://www.federalreservehistory.org/essays/stock-market-crash-of-1929):
  the market peaked in September and broke sharply in October 1929.
- [U.S. Senate, The Senate Passes the Smoot-Hawley Tariff](https://www.senate.gov/artandhistory/history/minute/Senate_Passes_Smoot_Hawley_Tariff.htm):
  the Senate passed the measure on June 13, 1930, and President Hoover signed it on June 17.

Event labels use years/ranges because the historical processes do not reduce honestly to a
single causal date. They orient the reader and are not causal annotations.

## Verification contract

V3 is complete only when:

1. New V3 tests re-derive every printed numeric takeaway and the V2 suite remains green.
2. The generator rebuilds the deterministic site and validates all navigation, links, anchors,
   JSON shards, and metric registrations.
3. Generated JavaScript passes `node --check`; project Python compiles; `git diff --check` passes.
4. The full pytest suite passes.
5. In-app browser QA covers every chapter at approximately 1280×720 and 390×844, normal and
   `?motion=reduce`, keyboard-only controls, chart hover, mobile callouts, view changes, multi-era
   selection, face opacity/outline, chronology progress, and page-level overflow.
6. The local preview remains at `http://127.0.0.1:8010/`. Nothing is deployed, staged, committed,
   pushed, or submitted as a pull request.

## Completion record

- New V3, retained V2, site-validation, and metric-registry tests pass; the complete suite passes
  with 2,373 tests.
- The x86_64 Rosetta build validates 72 HTML pages and 161 JSON shards and writes both the normal
  and self-contained story entry points.
- Generated story JavaScript passes `node --check`; project Python compiles; `git diff --check`
  passes.
- In-app browser QA covers all 11 story sections and 11 substantive charts at 1280×720 and
  390×844. It verifies the sticky range/era/title label, keyboard view and manual-stage controls,
  pointer hover, mobile view changes, multi-era selection, 45 persistent portraits, opacity plus
  outline emphasis, local chart scrolling, zero page-level overflow, and zero console warnings or
  errors.
- The `?motion=reduce` route exercises the same runtime branch as the operating-system preference:
  section transforms and animations are removed and transition durations resolve to zero.
- Browser QA found and fixed two defects before completion: a duplicate expansion subplot title
  and endpoint-only platform support text leaking into ordinary-year hover labels.
