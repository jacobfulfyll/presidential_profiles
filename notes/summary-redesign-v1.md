# Summary page redesign v1

## Purpose

`summary.html` synthesizes the nine chronological Story chapters. It does not add a tenth
historical chapter and it does not present an undifferentiated national character. Every claim
describes the Miller Center prepared-speech corpus and keeps confirmatory, exploratory, and
descriptive evidence visibly distinct.

## Narrative order

1. **A presidency with more to talk about.** Show the nine-era annual-message breadth series,
   then the Level-1 agenda lifecycle heat map. The registered comparison is modern
   (1950/1980/2010 blocks) minus postbellum (1860/1890/1920 blocks), not the 2017–2026 chapter
   alone. Report the registered estimate and its interval together; do not describe a monotonic
   rise.
2. **From report to performance.** Align audience and medium as two 100% era rivers; then show
   proposal, values, mixed, and neither paragraph shares. Keep the all-president
   legal/procedural-versus-hype scatter, with all eras available as keyboard-accessible presets
   and no arbitrary era selected on load.
3. **The enemy changes.** Show top named adversaries by era beside enemy naming, domestic-only
   naming, and partisan attack. Repeated names may connect visually, but raw labels and derived
   aliases remain auditable. “Partisan attack is newer” is a corpus description, not a claim that
   early presidents lacked political conflict.
4. **Selling tomorrow and yesterday.** Put future and nostalgia on the same per-10,000-word
   scale. Treat the combination as coexistence, not a psychological diagnosis.
5. **A louder emotional register.** Restore the event-annotated ratio grammar as hype divided by
   doom. Calculate the ratio from aggregate counts in centered five-year windows, suppress
   windows below 20,000 words or with zero doom, retain absolute rates in tooltips, mark parity at
   one, and label event lines as orientation rather than causes. Keep the hype-versus-doom weather
   map as the president/era inspection view.
6. **What can America say about itself?** End with the changed/persisted/uncertain matrix and
   links to data quality and corrections.

## Evidence statuses and sources

- Registered: `data/coverage_pressure/inference_receipts.parquet` and the governed coverage
  artifact. The primary breadth comparison contains 56 eligible annual messages and uses a
  hierarchical president-then-speech bootstrap.
- Exploratory: `data/register/trends.parquet`, Level-1 lifecycle classifications and heat map,
  proposal/values labels, adversary identities, and combat labels.
- Descriptive: assigned audience/medium, marker lexicons, president scatter, temporal wording,
  hype, and doom.

Frozen paid annotations under `data/llm_annotations/` remain read-only. Derived views join
paragraph artifacts on `(doc_name, para_idx)` and speech artifacts on `doc_name`, with key-set
validation. Event guides never identify causes.

## Acceptance checks

- The page renders six named internal chapters in the order above and calls the country
  “America” in visible Summary framing.
- The breadth claim names the registered comparison and displays estimate, interval, support,
  and status without claiming a steady historical rise.
- Both communication rivers sum to 100% within each era; proposal/value categories are mutually
  exclusive and sum to 100%.
- The lifecycle heat map includes all 17 Level-1 domains and all nine Story eras with a text
  alternative and lifecycle class.
- All nine era controls for the president scatter are accessible; `All eras` is the initial
  state, and keyboard focus is visible.
- The adversary display names its aliasing boundary and supplies an era-by-era text alternative.
- The hype/doom ratio applies the support and zero-denominator guards and shows absolute rates in
  tooltips; event annotations include a non-causal guide.
- New chart identifiers are registered in `metrics.py`, generated links validate, JavaScript
  parses, and the desktop browser passes wide and narrow visual checks.
