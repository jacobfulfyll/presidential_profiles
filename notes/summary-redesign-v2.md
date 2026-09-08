# Summary page redesign v2

## Purpose

`summary.html` synthesizes the prepared presidential-speech corpus without forcing a strong
claim from every available measure. V2 supersedes the visible narrative and acceptance checks
in `summary-redesign-v1.md`; V1 remains as the historical record of the first implementation.

## Editorial decisions

1. **Breadth is a limitation, not the opener.** The large effective-topic increase occurs before
   the Civil War. Post-1860 movement is small, uneven, and method-sensitive. Do not publish the
   V1 breadth line or issue-lifecycle heat map on Summary; state plainly why they were removed.
2. **Make voice two separate four-category stacked-bar charts.** Show `WHO` as Congress, general
   public, specific organizations or groups, and other audiences; show `HOW` as written, spoken,
   radio/TV, and press conference/debate or other performed form. Do not combine the two
   questions in one legend. Each chart has nine 100% stacked era bars, its own palette and
   emoji-accented legend, selective in-bar labels, and exact counts and percentages on hover.
   Preserve category definitions and agreement receipts in the disclosure.
3. **Connect communication form to the president-level register comparison without claiming a
   cause.** A compact `Who + how → register` bridge carries the same nine-era chronology from
   audience and delivery into the existing 45-president portrait field. Its visible sequence is
   Congress → public, written → spoken/broadcast, and less procedural opening → procedural
   nineteenth-century middle → higher hype later. The copy must say these corpus patterns
   coincide and do not establish that audience or delivery caused the vocabulary change.
   Legal/procedural vocabulary remains on the horizontal axis and hype remains on the vertical
   axis, both as separate matches-per-10,000 dictionary rates rather than a combined score. One
   two-choice `By president` / `Over time` switch keeps the president portrait field as the
   default and adds a chronological comparison of those same two dictionaries. The time view
   uses the existing all-corpus `indices.yearly_rates` output: centered five-year pooled-word
   windows and a 20,000-word support floor. The generator divides the legal/procedural rate by
   the hype rate—which is algebraically the pooled legal/procedural count divided by the pooled
   hype count—and suppresses missing or non-positive hype denominators. One line renders on a
   logarithmic axis because the supported ratios span 0.30–75.36; unknown windows remain gaps
   rather than zero. The view does not consume the SOTU-only 30-year register artifact. Exact
   ratio and component rates remain in the line hover. At the owner's direction, the register
   section has no president or timeline text-alternative tables.
   Keep this timeline visually quiet: do not overlay the six historical event lines or the 2017,
   2021, and 2025 administration-transition lines. Hope/Doom retains its separate event context.
   Mark seven selected movements with direct year labels and concise hover summaries rather than event
   claims: the 1827-centered Adams-heavy rise (227 legal/procedural matches and 8 hype matches in
   1825–1829), the 1863-centered Civil War-era drop (164 and 28 in 1861–1865), the 1881-centered
   series maximum (829 and 11 in 1879–1883), the 1944-centered wartime low (51 and 64 in
   1942–1946), the 1966-centered policy-heavy rise (591 and 98 in 1964–1968), the 2016-centered
   mid-2010s decline (92 and 236 in 2014–2018), and the 2023-centered recent rebound (171 and 286
   in 2021–2025). Hovering a direct year label shows only its short explanatory summary,
   left-aligned and broken into compact lines; it does not repeat a title, window, ratio, or count
   receipt. Remove the permanent `Selected turns in the line` card section. Exact window values
   remain available from ordinary line hover.
   The view choice is transient rather than URL/history state. It supports click and
   Left/Right/Home/End switching with pressed-state semantics. The single line does not require
   color comparison, and quiet labeled era bands orient it chronologically. One
   native `Emphasize an era` select replaces the six historical-phase buttons and nine era cards
   for this Voice view only. Its fixed choices are All eras, Dim all, and the nine governed Story
   eras in chronological order; emphasis never changes membership or analytical values. It is
   disabled with a live explanation in the time view, where every supported window is shown, and
   restores its selected value on return. The Time portrait keeps its existing multi-era controls.
4. **Coordinate Conflict at the grain each question supports.** The target-mix view pools
   speaker-audited adversarial mentions within the nine canonical `trends.ERAS` reporting bands
   and renders five lines—nation, group, person, institution, and other—together on one shared
   percentage chart. Use color, distinct dash patterns, and distinct marker shapes rather than
   color alone. A native All/Nation/Group/Person/Institution/Other control previews on pointer or
   keyboard focus, pins on click/tap, keeps the other lines visible at low emphasis, and resets
   through All or Escape; direct line hover previews the same whole-trajectory emphasis. Point
   hover retains exact shares and mention counts. Six direct callouts explain the largest
   category-share turns from the governed current/prior era counts, including denominator-driven
   changes. Their hover summaries also name the leading raw entity labels behind each turn, using
   frozen `paragraph_entities.parquet` rows joined to eligible `paragraph_view_v1.parquet` keys;
   tests pin every printed label count. Keep the category definitions and denominator caveats in a
   native `details` disclosure immediately below the chart, closed by default and expandable without
   JavaScript. They remain descriptive rather than causal. At the owner's direction, remove the
   target-mix text-alternative table. The joint portrait view remains at president
   grain but uses only the governed `annual_message_strict` State of the Union/annual-message
   treatment so each available point represents the same broad speech genre: zero-sum framing is
   on x, partisan attack is on y, and portrait area—not diameter—is proportional to enemy-naming
   paragraph share. Give every plotted portrait a border keyed to the category with the largest
   adversarial-entity count in that president's same qualifying messages, using the target chart's
   governed Nation/Group/Person/Institution/Other colors. A tie receives the documented neutral
   border rather than an arbitrary winner; hover states the category or tie so
   color is never the sole carrier. The border is a uniform seven pixels so it remains legible
   around both small and large portraits. Portrait hover omits raw flag and entity counts, uses
   bold field labels, and retains only the exact rates, visual-encoding explanations, support
   state, and number of qualifying annual messages. On horizontally scrolling narrow layouts,
   tapping a portrait scrolls only the chart enough to keep its complete tooltip visible.
   Preserve all 45 rows in the governed contract; the figure plots the 42 supported presidents,
   while William Harrison, James A. Garfield, and Harry S. Truman have no qualifying message.
   At the owner's direction, the portrait panel has no exact-value text-alternative table. Its
   header contains only the chart title and two lines identifying portrait area and the common
   annual-message paragraph denominator; omit the redundant coverage eyebrow, observed range,
   position sentence, and x/y shorthand. Introduce the panel with the reader-facing sequence
   `BY PRESIDENT · COMPARABLE SPEECHES` / `How presidents frame conflict` / `Compared within State
   of the Union and annual-message speeches.` rather than implementation language about frames
   and genres. Remove the
   separate Enemy naming, Zero-sum, and
   Partisan lollipops. Keep mention composition and paragraph-level frame frequency
   denominationally distinct; the portrait view is not a combined score. Very narrow screens may
   scroll graph bodies internally, but must not omit eras or presidents.
5. **Use governed speaker-audited populations at both grains.** The target trajectories consume
   the validated `conflict-target-mix-v1` era payload at `speaker_audited_all`; the portrait
   consumes `annual_message_strict` rows from the validated `president-conflict-v2` treatment
   artifact. The all-speaker president rows remain the parity check for the target-mix totals;
   the portrait's common-genre restriction must not be compared to those totals. Use no copied
   values, document ownership, or chart-specific corrections. Expose the treatment labels, retain
   support states, and never coerce unavailable shares or rates to zero.
6. **Lead with computed, grain-specific findings.** Derive target-category findings from pooled
   era counts and frame-rate findings from the president contract that draws the portrait. Keep
   the language neutral enough to remain true after a producer swap and identify both summaries
   as descriptive and sensitive to genre, coverage, and speaker mix.
7. **Compare temporal appeal by president and over time.** Use one all-president portrait scatter
   built from every available corpus speech assigned to each document president, plus a switch to
   two separate trailing four-year rolling-average lines. Put future-family matches per 10,000 marker words on x
   and nostalgia-family matches per 10,000 marker words on y in the president view; use uniform
   portrait sizes and retain the below-five-speech amber halo. In the time view, show Tomorrow and
   Yesterday as separate solid/dashed lines averaging four consecutive annual rates and plotted
   at the ending year—never as a ratio or browser-computed measure. Reuse the Voice section's single All/Dim/one-era selector,
   disabling it with a polite explanation in the time view and restoring it on return. Exact values
   remain in chart hover; the former self-reference encoding, size key, and text-alternative table
   are removed. The president view remains descriptive, all-genre, document-owned, and not
   speaker-audited. Do not append a separate founding-message dictionary audit to this reader flow.
8. **Use Hope/Doom, not Hype/Doom.** Divide aggregate NRC hope matches by aggregate doom-family
   matches in centered five-year windows. Require 20,000 words and a nonzero doom count, use a
   log scale, retain both absolute rates in tooltips, and keep event lines explicitly non-causal.
   Do not show the V1 separate-axis rhetorical-weather map on Summary.

## Evidence and joins

- Communication: frozen `speech_annotations.parquet`, joined one-to-one on `doc_name`.
- Conflict target mix: governed derived
  `data/combat/target_mix_by_era_speaker_audited_v1.parquet`, validated as one row per canonical
  `trends.ERAS` band with integral mention/support counts, calendar-year bounds, balanced category
  shares, and `speaker_audited_all` treatment metadata. Each era share is a ratio of pooled
  mentions, never an average of president or annual percentages.
- President Conflict portrait: governed derived
  `data/combat/by_president_speaker_audited_v2.parquet`, validated as one row per president with
  integral support/category counts, bounded paragraph rates, schema/treatment metadata, and
  support flags.
- President Time portrait: exact one-to-one `doc_name` joins across
  `speech_markers.parquet` and `speeches.parquet`. Future and nostalgia pool family matches over
  marker words. The treatment is all available, document-owned corpus transcripts rather than a
  speaker-audited or annual-message-only population. The paired timeline reads the existing
  supported trailing four-year rolling averages generated from `speech_markers.parquet`; each
  point is placed at the ending year, requires coverage in all four years, and requires at least
  10,000 marker words across the window.
- Hope/Doom: `speech_markers.parquet`.
- Frozen paid annotations remain read-only. Event markers orient the time axis and never identify
  causes.

## Acceptance checks

- Generated Summary has five visible chapters: Voice, Conflict, Time, Emotional register, Audit.
- `summary_breadth`, `summary_lifecycle`, `summary_stance`, and `weather_map` are absent from the
  Summary figure payload and visible route.
- Voice has separate WHO and HOW stacked-bar charts, each with four categories and nine era
  bars; all 18 era compositions equal 100% and have graph-specific palettes, legends, and hover
  receipts.
- Conflict contains one five-line target-mix chart plus one portrait scatter.
  Each line carries one category over the fixed nine-band `trends.ERAS` axis; all lines
  share one percentage scale, point hover exposes exact values, and the five
  values for an era total 100%. Thin records use hollow markers and gaps remain `N/A`.
  The target view has no text-alternative table. All plus five labeled native buttons provide
  pointer, focus, click/tap, pressed-state, status, and Escape behavior without removing any line;
  direct line hover also previews the whole trajectory. Exactly six labeled callouts are derived
  from adjacent era counts and explain major compositional turns without claiming causation.
  A visible note defines
  every category, states that pooled mentions—not paragraphs, presidents, or annual shares—form
  the denominator, and describes `Other` as a heterogeneous residual of stored labels.
- The portrait scatter contains the 42 presidents with supported `annual_message_strict` rows,
  with zero-sum on x, partisan attack on y, and enemy naming encoded by portrait area. The governed
  contract retains all 45 presidents, but the owner-directed exact table is absent. Hover exposes
  exact message coverage and percentages. The compact hover uses bold labels,
  excludes raw event counts, and states the speech count because qualifying coverage varies from
  one to nine annual messages across supported presidents. Each plotted portrait has a clearly
  visible border using the target-category palette for its largest annual-message adversary count;
  ties use the neutral key and are named in hover. The three standalone framing lollipops
  are absent.
- Conflict values are contract-derived at distinct era and president grains. Low mention and
  paragraph support use separate validated states; unsupported category/rate values render `N/A`
  rather than zero.
- A fixture-level replacement of era category counts or president framing rates changes the
  relevant target trajectory or portrait position/area without an era-, president-, or
  renderer-specific correction.
- Voice places one visible, non-causal communication-to-register bridge between the separate WHO
  and HOW bars and the register comparison. `By president` is the default, places
  legal/procedural vocabulary on x and hype on y, and retains all 45 portrait markers. `Over
  time` contains exactly one line trace: pooled legal/procedural dictionary matches divided by
  pooled hype dictionary matches across the same 238 center years. Its logarithmic axis preserves
  all 228 supported centered five-year windows across the observed 0.30–75.36 range; unsupported
  or zero-denominator values stay disconnected. It identifies all nine eras, exposes the exact
  ratio and both component rates on hover. At the owner's direction, the register section has no
  president or timeline text-alternative table. The chart has no historical-event or administration-transition lines.
  Direct 1827, 1863, 1881, 1944, 1966, 2016, and 2023 year labels point to supported line values
  and expose seven concise, multiline corpus-composition summaries on hover. The permanent
  `Selected turns in the line` guide is absent; ordinary line hover retains exact values. The
  no-JavaScript fallback states that the interactive chart requires JavaScript rather than
  reproducing the removed tables. The switch has two pressed-state buttons, arrow/Home/End keyboard
  behavior, visible focus, and no URL mutation.
- The Voice register view exposes one native select with exactly All eras, Dim all, and the nine
  governed Story eras. All eras is the default; Dim all and single-era selection alter
  opacity/outline only, and every other president remains plotted at low emphasis. The control
  disables with a polite explanation in `Over time`, preserves its value, and re-applies it when
  `By president` returns. Controls have programmatic names, 44px minimum targets, visible focus,
  a single-column narrow layout, chart-local narrow-screen scrolling, and no page-level overflow.
- The Time president view contains all 45 presidents with future-family rate on x and
  nostalgia-family rate on y. It pools exact source counts rather than averaging speech rates,
  gives every portrait the same size, preserves the three below-five-speech records with a distinct
  halo, and exposes exact axis rates, marker words, speeches, record span, Story era, and support in
  hover. Its single All/Dim/one-era selector changes emphasis only and preserves positions and
  support halos. The time view uses two distinct, non-connected lines for the same families over
  supported trailing four-year rolling averages of annual rates, with exact hover and no invented ratio. Switching
  views disables and later restores the era selector. The former self-reference dimension, size
  legend, and text-alternative table are absent. Narrow layouts contain the fixed chart field inside
  the graph scroller without page-level overflow.
- The Time chapter's later lexical boundary is one three-state chart stage: National name,
  Modal words, and Personal voice. National name preserves the governed 1910 sustained-crossover
  rule; Modal words groups five disjoint surface families—Necessity (`must`, the complete
  `need / needs / needed / needing` surface family, and exact `have / has / had / got to` plus
  `gotta` phrases), Commitment/Intent (`will + shall` plus explicit speaker or administration
  promises, pledges, vows, guarantees, commitments, plans, resolutions, and statements of
  determination), governed Absolute emphasis, Conditional (`would + could`), and
  Advice/Possibility (`should` plus the existing hedge family, including `may` and
  `might`)—without presenting them as a certainty index; Personal voice restores the exact
  first-person singular and plural families and describes the 2020s reversal without treating
  pronouns as individualism or divisiveness. Arrow/Home/End keyboard behavior, pressed state,
  live announcements, internal narrow-screen scrolling, and whole-line hover/click emphasis are
  shared across the stage. Modal-family and pronoun views use centered seven-year windows with a
  10,000-word floor, retaining all 238 center years; national naming keeps its governed five-year
  rule and 20,000-word support gate while displaying the complete 1789–2026 corpus range. Missing
  national-name windows remain gaps. Every language trajectory uses a solid stroke and a distinct
  decade marker shape within its view; I/me/my and United States therefore no longer use dashes.
  Every point hover is limited to a short whole-line definition, exact center-year rate, and
  contributing presidential records; full support/method detail remains in the permanent evidence
  disclosure. The 238 modal/pronoun center-year values are finite without imputation. Necessity and
  the explicit Commitment/Intent additions are
  counted at generation time from `speeches.parquet`; Absolute emphasis reuses
  `speech_markers.parquet`. `Going to` remains excluded because it is ambiguous and heavily
  concentrated in modern spoken transcripts. No browser-side analytical measure is introduced.
- A compact three-evidence synthesis answers the broad divisiveness question without a composite
  score. It treats the annual-message partisan-attack ratio against Civil War & Reconstruction as
  the strongest result (4.8x, 95% speech-clustered bootstrap 2.3–14.3x); treats the 2020s pronoun
  reversal as descriptive rather than a division measure; and treats the present era's 0.315
  detrended Civil War nearest-neighbor result as a modest relative analogy. Enemy-naming and
  zero-sum ratios against the founding remain visibly non-separable from one because both intervals
  contain one, and all present-era ratios retain the governed 10-speech low-cluster caution.
- Hope/Doom applies the support and denominator guards, uses NRC hope as the numerator, keeps
  absolute rates in hover text, and includes non-causal event lines.
- Metric registrations, site validation, static JavaScript parsing, wide/narrow browser QA, and
  the relevant pytest suite pass before handoff.
