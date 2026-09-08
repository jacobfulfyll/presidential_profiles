# Final release program v1

Status: active umbrella program.

This note preserves the owner's broad release plan from the planning conversation. It is the
durable parent context for the page- and contract-specific implementation plans that follow.
Keep it available—and keep its link in the knowledge router—until the owner confirms that the
entire program is complete. A later detailed plan may refine execution, but it should explicitly
identify any decision that supersedes this note rather than silently dropping a workstream.

## Preserved program response

I’ve taken it all in. The main theme is clear: the project has plenty of analysis, but the final
version needs stronger data trust, less explanatory clutter, and more direct visual interaction.

I would organize your notes into eight major workstreams—not dozens of unrelated UI tweaks.

## 1. Population, speaker, and entity correctness

This should come first because it affects everything downstream.

### Claimed constituents

You want claimed constituents populated for every era. There is already a planned constituency
workflow, but it is blocked because the existing candidate labels are provisional and use a
different canonical population.

There is also an important distinction in your later idea:

- Named entities identify people, organizations, nations, and places.
- Claimed constituents identify groups a president presents themselves as representing: workers,
  farmers, families, taxpayers, veterans, business, and so on.
- Named-entity extraction alone will not recover many claimed constituencies.

My recommendation is a hybrid:

- Complete a governed claimed-constituent layer for the Era Profiles.
- Add free local entity extraction as a separate corpus-wide enhancement.
- Do not silently substitute named entities for claimed constituents.
- If we ultimately decide not to finish constituency labeling, rename the card to “Referenced
  groups and entities” so it does not overclaim.

### Actual speaker attribution

The data model should retain four separate facts:

1. Document owner
2. Actual paragraph speaker
3. Speech type, such as debate, press conference, or annual message
4. Presidency/era context at the time of speaking

The repository already has a speaker-attribution layer, so the first task is to audit every public
surface and confirm that it actually uses that layer.

For Carter specifically, we need to separate:

- Carter’s own partisan attacks
- Opponents attacking Carter in a Carter-owned transcript
- The effect of debates being overrepresented in his corpus record

President-level adversarial measures should use actual-speaker paragraphs. They should also show
a debate-excluded sensitivity result or at least clearly disclose genre concentration. This is a
correctness issue, not merely a presentation change.

## 2. Summary page redesign

You have several clear decisions here.

### Locked changes

- Replace “How the target mix changes” with one five-line graph.
  - All target categories present simultaneously
  - Hover or keyboard focus isolates a line
  - Exact values remain available on interaction
- Extend “America overtakes the United States” across the full corpus range.
  - Keep 1880–1932 as a shaded or annotated context window
  - Do not crop the rest of history
- Change Hope ÷ Doom to Doom ÷ Hope.
  - Rename labels, explanations, hover text, registration, and tests consistently
  - Treat a higher value as more doom relative to hope
- Remove the Audit section from Summary.
- Move the audit to Data and expand it.
- Redesign the extreme-speech material so it can be understood quickly.

### Tomorrow, yesterday, and self-reference

I agree that self-reference feels bolted on as a third dimension. Tomorrow and yesterday form a
coherent temporal orientation; self-reference answers a different question.

My recommendation:

- Keep tomorrow/future language on x.
- Keep yesterday/nostalgia language on y.
- Do not force a third rhetorical variable onto portrait size.
- Use a fixed portrait size, with perhaps a subtle support halo for thin records.
- Put first-person singular versus collective language in a separate compact comparison if it
  proves interesting enough.

This makes the main graph legible and honest.

### A fifth Summary section

The best near-term candidate is a visual redesign of the extreme-speech evidence rather than a new
analysis.

I would make it an “Unusual speeches” map:

- Each point is a speech
- Two meaningful rhetorical dimensions determine its position
- Users can filter by era or president
- Selecting a point opens one concise evidence card
- The current walls of text become on-demand details

This uses governed evidence already in the project and avoids inventing another methodology merely
to fill a fifth slot.

A more ambitious alternative is a national topic constellation, but that overlaps with the
topic-network idea and should probably come later.

## 3. Compare page simplification

Your desired shape is much cleaner.

### Controls

Remove:

- Swap A/B
- Add third president
- Copy link
- Reset

Instead:

- Always show three president selectors
- A and B have useful defaults
- C defaults to “None”
- Selecting “None” removes that president
- URL state updates automatically

### Rhetoric charts

- Change the third-president color to a distinct green.
- Fix clipping of Top word, Hope, and Partisan attack labels.
- Preserve the radar charts and exact values, but move exact tables into optional details rather
  than showing them prominently.

### Agenda comparisons

I would not use another radar for the agenda data.

For broad domains, the best fit is a shared-scale grouped horizontal bar chart:

- One row per domain
- One dot or bar per president
- Exact differences are easy to compare
- Sparse and missing values remain clear

For fine topics, use a heatmap:

- Topics as rows
- Presidents as columns
- Consistent intensity scale
- Sort by combined importance or largest between-president difference

For legacy issues, use a compact connected-dot plot or the same grouped-bar grammar as broad
domains. Consistency is more valuable than adding another chart type.

### Cross-method section

A fourth “Agenda across methods” section makes sense, but it must not imply that the three
taxonomies are interchangeable.

I would build an aligned crosswalk board.

The selected presidents would be compared within each column. The connecting structure would
communicate methodological relationships without manufacturing a combined score.

### Remove or demote

- Remove nearest-neighbor context from Compare; keep it on profiles.
- Remove proposal-versus-values unless we find a stronger comparative story.
- Remove or demote speech-type composition, particularly if most rows are unavailable.
- Keep Corpus Footprint.
- Keep adversarial entities, invocations, vocabulary, and signature speeches, but redesign them as
  a more cohesive evidence section.

## 4. Explore page interaction redesign

This page mostly needs information architecture.

### Series selection

Replace the long flat topic/acronym list with:

- Search
- Grouped categories
- Collapsible group headings
- Recently or currently selected items
- Clear selected-count feedback

### Word-family controls

“Group word forms” should be a primary choice near word entry, not buried in chart options.

When a word is added, its pill should show:

- Display family name
- Number of forms combined
- Expandable list of included forms
- Whether grouping is active

For example:

> slavery · 3 forms<br>
> slavery, slave, slaves

Grouping should be selectable before or when adding the word.

### Trend interaction

- Hovering a line should highlight the entire line and dim others.
- Hover should show the exact value.
- Keyboard focus should provide equivalent behavior.
- Error bands should be a visible toggle near the series controls.
- The expandable chart-options panel needs a clearer label and stronger visual hierarchy.

### Evidence and method

I agree that the large evidence table does not belong in the primary UI.

However, exact values should remain available for accessibility and auditability. I would:

- Remove the visible “Inspect the evidence” table from the main flow.
- Offer CSV download prominently.
- Keep exact values in a collapsed Details disclosure or accessible fallback.
- Reduce “Explain this measure” to a concise tooltip or short definition with a Methods link.

The guided comparisons can stay, but their controls and cards should use the same improved visual
system.

## 5. Navigation and presidential profiles

### Navigation

The disappearing submenu is a real interaction bug. The hover/focus region needs to bridge the gap
between the top-level item and submenu.

It should support:

- Pointer movement without collapse
- Keyboard navigation
- Click-to-open on touch devices
- Escape to close
- Adequate submenu target sizes

### President directory

Remove the three top links:

- Dashboard
- Compare
- How AI labels work

The global navigation already owns those destinations.

### Individual profiles

Remove the prominently displayed tables beneath both rhetorical fingerprints. Keep their exact
values in collapsed details or accessible markup.

The Evidence section needs a complete rethink. Your confusion about “George Washington being Crime
& Justice” is evidence that the current cards expose outputs without explaining why they matter.

Each retained evidence item should answer:

- What is being claimed?
- Why did this president receive this label?
- Is it absolute emphasis or relative to their era?
- Which speeches and excerpts support it?
- How much evidence is there?

### Networks

Two profile-level networks make sense:

1. Invocation network
   - Who this president invoked
   - Who later invoked this president
   - Direction, stance, and function available on interaction
2. Topic network
   - The president’s strongest topics
   - Other presidents connected through shared emphasis
   - Topic nodes exert stronger visual pull according to emphasis

A full all-president/all-topic graph would be too dense for every profile. I suggest:

- A corpus-wide overview constellation on Summary or Explore
- A focused “ego network” on each president profile
- Only the strongest supported edges initially
- Controls to expand the network

Similarity should remain simple on profiles; headshots are optional rather than necessary.

## 6. Issues page simplification

### Immediate fixes

- Remove the loading/placeholder copy once the chart has initialized.
- Fix the large white-space region.
- Give overlapping fine topics genuinely distinct colors.

### Fine-topic controls

- Start explanatory topic cards collapsed.
- Remove the visible Auto choice.
- Choose grouped bars or heatmap automatically as the initial mode.
- Let users switch directly between Grouped bars and Heatmap.
- Select all topics initially.
- Remove the topic-selection interface unless there is a demonstrated need for filtering.

### What stays

- Broad five-year trend
- Selected corpus excerpts
- Presidents who emphasize the issue most
- Fine-topic visualization

### What moves out

Move to Data:

- Detailed data and method disclosures
- Large exact-value/evidence tables
- Technical provenance material

Retain one concise unit/provenance statement and a link to the relevant Data section. Downloads
should remain available.

## 7. Data, Data Quality, Methods, and Era Choices

Your assessment is consistent:

- Era Choices is strongest.
- Methods is substantially better but still needs clearer metric explanations.
- Data Quality is overloaded and difficult to interpret.

### Data Quality redesign

It should be organized around user questions, not artifacts:

1. What is in the corpus?
2. What is missing or uneven?
3. How reliable are the labels?
4. Where do models disagree?
5. Which results should be treated cautiously?
6. What corrections and audits were performed?

The confusion matrix currently foregrounds class imbalance without helping the reader interpret
it. We should show:

- Counts and row-normalized percentages
- Clear labels for false positives and false negatives
- Why “mostly no” is expected or problematic
- The practical effect on published measures
- A plain-language verdict

### Agreement metrics

The Methods page should explain why different labels use different metrics:

- Cohen’s κ: binary labels such as partisan attack
- Jaccard: sets of multiple topics or entity names
- Exact match: single mutually exclusive categorical decisions
- Coverage/completion: whether a second-model judgment exists at all

Every decimal should include a plain-language interpretation and avoid implying universal “good”
thresholds.

The missing 17,618 rows, second-model completion chart, and mean entity-name Jaccard should be
either explained in direct language or demoted to technical details. A chart that cannot answer a
reader question should not lead the page.

### Larger audit

Moving Audit from Summary to Data is right. The expanded audit should cover:

- Corpus coverage and genre imbalance
- Actual-speaker attribution
- Debate and press-conference sensitivity
- Annotation coverage
- Inter-model agreement by era and field
- Entity extraction coverage
- Known taxonomy gaps
- Corrections and exclusions
- Which public charts consume each treatment

## 8. Global design rule

Across Compare, Explore, Profiles, Issues, and Data, I hear the same request:

- Fewer permanent explanations
- Fewer large exact-value tables
- More direct graphs
- Better hover/focus behavior
- Better grouping
- Clearer defaults
- Downloads for deep inspection
- Technical methods centralized on Data

I agree. I would preserve exact values and accessible fallbacks, but move them behind disclosures
or downloads instead of letting them dominate the page.

## Proposed execution order

1. Lock the population semantics: constituents, entities, actual speakers, debates, era ownership.
2. Audit which existing pages still consume document-owner rather than actual-speaker data.
3. Fix global navigation and shared interaction components.
4. Implement the concrete Summary changes.
5. Simplify Compare and add the agenda visual system.
6. Redesign Explore’s selectors, word families, bands, and hover behavior.
7. Redesign profile Evidence and build the profile networks.
8. Simplify Issues.
9. Rebuild Data Quality and consolidate audit/method material.
10. Run the full accessibility, responsive, deterministic-build, and release gates.

## Unresolved product decisions

- Whether claimed constituents remain a distinct governed concept or are replaced by a more modest
  referenced-entities view
- Whether Summary’s fifth section is the redesigned extreme-speech map or a topic constellation
- The exact scope of the corpus-wide topic network
- Which current audit charts deserve redesign versus removal

Everything else is concrete enough to turn into an implementation plan.

## Subsequent decisions already accepted

The broad response above is preserved as originally organized, but later owner decisions already
qualify two parts of it:

- The project will not complete the high-effort claimed-constituent labeling exercise for this
  release. Story instead uses the governed reference/entity foundation and labels that surface
  modestly; Plan 2 implemented `Distinctive references` across all nine eras.
- Actual-speaker attribution is a corpus-wide correctness requirement, not a Carter-only fix. The
  completed speaker/reference foundation and Story migration are the first implementation stages;
  later page plans must keep actual speaker distinct from source-document owner.

Plan 3 has now implemented the reusable actual-speaker president/topic network
as a governed, downloadable contract without production-page placement. At
that checkpoint, Plan 4 was the next staged decision: whether and how Summary
would consume that bundle.

Subsequent implementation has now completed proposed-order Steps 4–6. Summary
consumes Plan 3 only through its approved recurring-topic gravity-field
projection; Compare uses the accepted Rhetoric → Agenda → Evidence contract in
`notes/compare-agenda-visual-system-v1.md`; and Explore uses the governed v2
projection and interaction contract in
`notes/explore-interaction-and-evidence-v1.md`. The next unimplemented item in
the preserved proposed order is Step 7, profile Evidence and profile networks;
that future plan must continue to protect the completed Story, Summary,
Compare, and Explore surfaces.
