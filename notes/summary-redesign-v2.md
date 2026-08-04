# Summary page redesign v2

## Purpose

`summary.html` synthesizes the prepared presidential-speech corpus without forcing a strong
claim from every available measure. V2 supersedes the visible narrative and acceptance checks
in `summary-redesign-v1.md`; V1 remains as the historical record of the first implementation.

## Editorial decisions

1. **Breadth is a limitation, not the opener.** The large effective-topic increase occurs before
   the Civil War. Post-1860 movement is small, uneven, and method-sensitive. Do not publish the
   V1 breadth line or issue-lifecycle heat map on Summary; state plainly why they were removed.
2. **Make voice a two-question transition.** Show `WHO` as Congress, general public, and other;
   show `HOW` as written, spoken, and broadcast/other performed. Each era and row sums to 100%.
   Preserve full categories and agreement receipts in the disclosure.
3. **Keep the president-level procedure/performance comparison.** All 45 presidents remain
   visible. Historical-phase buttons and nine specific era cards change emphasis only.
4. **Show enemy categories, not an entity-name parade.** For every Story era, divide adversarial
   entity mentions among nation, group, person, institution, and other. The five shares must sum
   to 100% within each era.
5. **Keep observed combat framing without confidence whiskers.** The annual-message lines show
   enemy naming, zero-sum framing, and partisan attack across all nine eras. The prose may say
   that enemies are old and sustained partisan attack is newer, while retaining the thin-cell
   caveat.
6. **Add a president-level conflict companion.** Compare each president's enemy-naming paragraph
   share with party-attack share. Reuse the exact same era interaction and fixed membership as
   the procedure/performance view.
7. **Audit the nostalgia dictionary in public.** The founding 1770–1799 annual-message block has
   17 declared matches: 10 `again`, 7 `restore*`, and 0 `back to`. Explain that these contexts
   usually describe recurrence, restored order, or restored relations, not a golden-age appeal.
8. **Use Hope/Doom, not Hype/Doom.** Divide aggregate NRC hope matches by aggregate doom-family
   matches in centered five-year windows. Require 20,000 words and a nonzero doom count, use a
   log scale, retain both absolute rates in tooltips, and keep event lines explicitly non-causal.
   Do not show the V1 separate-axis rhetorical-weather map on Summary.

## Evidence and joins

- Communication: frozen `speech_annotations.parquet`, joined one-to-one on `doc_name`.
- President conflict: frozen paragraph judgments, joined one-to-one on
  `(doc_name, para_idx)` and many-to-one to speech metadata.
- Enemy categories: governed derived `data/combat/adversary_mix.parquet` era rows.
- Combat lines: governed derived `data/combat/combativeness.parquet`, `sotu_only` treatment.
- Time and Hope/Doom: `register/trends.parquet` and `speech_markers.parquet`.
- Frozen paid annotations remain read-only. Event markers orient the time axis and never identify
  causes.

## Acceptance checks

- Generated Summary has five visible chapters: Voice, Conflict, Time, Emotional register, Audit.
- `summary_breadth`, `summary_lifecycle`, `summary_stance`, and `weather_map` are absent from the
  Summary figure payload and visible route.
- Voice has three audience and three form categories, and all era totals equal 100%.
- Enemy-category shares cover nine eras and sum to 100%; no actual entity names appear in that
  chart.
- Combat traces have no confidence error bars.
- Both president views expose nine selected era cards, six phase presets, All eras, and Dim all;
  every president remains plotted at low emphasis when unselected.
- The founding term audit prints the exact 17/10/7/0 receipt.
- Hope/Doom applies the support and denominator guards, uses NRC hope as the numerator, keeps
  absolute rates in hover text, and includes non-causal event lines.
- Metric registrations, site validation, static JavaScript parsing, wide/narrow browser QA, and
  the relevant pytest suite pass before handoff.
