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
3. **Keep the president-level procedure/performance comparison.** All 45 presidents remain
   visible as portrait markers. Legal/procedural vocabulary stays on the horizontal axis and
   hype stays on the vertical axis, both as matches per 10,000 words. Historical-phase buttons
   and nine specific era cards change emphasis only.
4. **Coordinate Conflict at the grain each question supports.** The target-mix view pools
   speaker-audited adversarial mentions within the nine canonical `trends.ERAS` reporting bands
   and renders five vertically aligned small-multiple trajectories—one for nation, group, person,
   institution, and other—on one shared percentage scale. Print every era percentage at its point
   and retain exact counts and shares in an era table. The joint portrait view remains at president
   grain: zero-sum framing is on x, partisan attack is on y, and portrait area—not diameter—is
   proportional to enemy-naming paragraph share. Remove the separate Enemy naming, Zero-sum, and
   Partisan lollipops. Keep mention composition and paragraph-level frame frequency
   denominationally distinct; the portrait view is not a combined score. Very narrow screens may
   scroll graph bodies internally, but must not omit eras or presidents.
5. **Use governed speaker-audited populations at both grains.** The target trajectories consume
   the validated `conflict-target-mix-v1` era payload; the portrait consumes the validated
   `president-conflict-v2` payload. Both select `speaker_audited_all` rather than copied values,
   document ownership, or chart-specific corrections. Expose the treatment labels, retain
   support states, and never coerce unavailable shares or rates to zero.
6. **Lead with computed, grain-specific findings.** Derive target-category findings from pooled
   era counts and frame-rate findings from the president contract that draws the portrait. Keep
   the language neutral enough to remain true after a producer swap and identify both summaries
   as descriptive and sensitive to genre, coverage, and speaker mix.
7. **Audit the nostalgia dictionary in public.** The founding 1770–1799 annual-message block has
   17 declared matches: 10 `again`, 7 `restore*`, and 0 `back to`. Explain that these contexts
   usually describe recurrence, restored order, or restored relations, not a golden-age appeal.
8. **Compare temporal appeal at president grain.** Replace the era-level future/nostalgia line
   with one all-president portrait scatter built from every available corpus speech assigned to
   each document president. Put future-family matches per 10,000 marker words on x and
   nostalgia-family matches per 10,000 marker words on y. Make portrait area—not diameter—equal
   the first-person singular share among counted singular and plural first-person pronouns. Use a
   fixed theoretical 100% area reference, keep all 45 presidents in place when era controls change
   emphasis, distinguish records below five speeches with an amber halo, and retain exact rates,
   counts, denominators, support, and corpus record spans in the text alternative. This view is
   descriptive, all-genre, document-owned, and not speaker-audited; singular share is not ego or
   personality. Keep the founding annual-message dictionary audit visibly separate from this
   all-speech population.
9. **Use Hope/Doom, not Hype/Doom.** Divide aggregate NRC hope matches by aggregate doom-family
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
  `speech_markers.parquet`, `speech_stats.parquet`, and `speeches.parquet`. Future and nostalgia
  pool family matches over marker words; self-reference pools singular-family counts over
  singular plus plural first-person counts. The treatment is all available, document-owned
  corpus transcripts rather than a speaker-audited or annual-message-only population.
- Founding dictionary audit and Hope/Doom: `register/trends.parquet` and
  `speech_markers.parquet`.
- Frozen paid annotations remain read-only. Event markers orient the time axis and never identify
  causes.

## Acceptance checks

- Generated Summary has five visible chapters: Voice, Conflict, Time, Emotional register, Audit.
- `summary_breadth`, `summary_lifecycle`, `summary_stance`, and `weather_map` are absent from the
  Summary figure payload and visible route.
- Voice has separate WHO and HOW stacked-bar charts, each with four categories and nine era
  bars; all 18 era compositions equal 100% and have graph-specific palettes, legends, and hover
  receipts.
- Conflict contains five vertically aligned target-mix trajectories plus one portrait scatter.
  Each target panel carries one category over the fixed nine-band `trends.ERAS` axis; all panels
  share one percentage scale, every supported point has a direct percentage label, and the five
  values for an era total 100%. Thin records use hollow markers, gaps remain `N/A`, and an exact
  era table retains counts, shares, contributing speeches, and support. A visible note defines
  every category, states that pooled mentions—not paragraphs, presidents, or annual shares—form
  the denominator, and describes `Other` as a heterogeneous residual of stored labels.
- The portrait scatter contains all 45 complete president rows with zero-sum on x, partisan
  attack on y, and enemy naming encoded by portrait area; it exposes exact percentages on hover
  and in a text alternative. The three standalone framing lollipops are absent.
- Conflict values are contract-derived at distinct era and president grains. Low mention and
  paragraph support use separate validated states; unsupported category/rate values render `N/A`
  rather than zero.
- A fixture-level replacement of era category counts or president framing rates changes the
  relevant target trajectory or portrait position/area without an era-, president-, or
  renderer-specific correction.
- The procedure/performance president view places legal/procedural vocabulary on x and hype on y,
  retains all 45 portrait markers, nine selected era cards, six phase presets, All eras, and Dim
  all; every president remains plotted at low emphasis when unselected.
- The Time portrait contains all 45 presidents with future-family rate on x, nostalgia-family rate
  on y, and first-person singular share encoded by portrait area on a fixed 100%-reference scale.
  It pools exact source counts rather than averaging speech rates, preserves the three
  below-five-speech records with a distinct halo, never converts an unavailable pronoun
  denominator to zero, and exposes exact axis rates, match counts, singular/plural counts, marker
  words, speeches, record span, Story era, and support in hover or the complete text alternative.
  Era controls change emphasis only and preserve each portrait's position, area, and support halo.
  Narrow layouts contain the fixed president field inside the graph scroller without page-level
  overflow.
- The founding term audit prints the exact 17/10/7/0 receipt.
- Hope/Doom applies the support and denominator guards, uses NRC hope as the numerator, keeps
  absolute rates in hover text, and includes non-causal event lines.
- Metric registrations, site validation, static JavaScript parsing, wide/narrow browser QA, and
  the relevant pytest suite pass before handoff.
