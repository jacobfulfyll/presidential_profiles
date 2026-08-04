# Story Page V2 evidence and acceptance contract

Status: implemented and verified on 2026-07-23; the evidence contract below remains frozen.

## Purpose and boundaries

The homepage tells U.S. history chronologically through presidential speech: agenda, governing
voice, medium, crisis language, conflict, and recurring questions of rights, belonging, war, and
national identity. Headline statistics enter where they become historically meaningful. A visible
boundary ends the chronology before synthesis. The corpus is the Miller Center's curated formal
record, not a complete inventory of presidential communication, public opinion, policy, or
government performance.

Every chart below is descriptive unless a registered receipt is named. Event markers orient time
and never assert that an event caused a measured rise or fall. Labels produced by CorEx or an LLM
are measurements with declared vocabularies and limitations, not historical ground truth.

## Chapter evidence map

| Chapter and historical claim | Data artifact | Visualization and annotations | Statistical status | Interaction and accessibility | Acceptance test |
|---|---|---|---|---|---|
| **1789–1815 — The written republic.** The available founding record is dominated by documents and a narrow set of formal presidential duties. | `data/speeches.parquet`, `data/llm_annotations/speech_annotations.parquet`, `data/speech_markers.parquet` | Era-only composition by president for the AI-assigned primary medium, paired with legal/procedural wording. No post-1815 point appears. | Descriptive corpus composition and dictionary rate. The medium label is one mutually exclusive primary form per speech. | Text summary exposes counts; patterns/symbols accompany colors; chart is readable at mobile width. | The x/domain ends in 1815 and contains no long-run series. Copy says “primary form assigned within this corpus.” |
| **1816–1849 — Expansion and federal conflict.** Money/banking, taxes/budgets, and foreign policy move away from the founding baseline as federal conflict and territorial reach grow. | `data/paragraph_issues.parquet` | Three issue lines expressed as percentage-point change from the 1789–1815 paragraph baseline. Context: Monroe Doctrine (1823); tariffs/nullification (1828–33); Bank veto/Bank War (1832); Texas annexation, Oregon settlement, and Mexican Cession (1845–48). | Descriptive CorEx paragraph labels; no causal inference and no cross-issue magnitude comparison beyond the shared percentage-point unit. | Manual legend/hover, numbered markers plus a readable event guide; not color-only. | Only the three claim-relevant issues and 1816–49 values appear; the all-history issue grid is absent. |
| **1850–1877 — Crisis of union.** Crisis vocabulary must first be read on its honest early scale before the whole record changes the axes; rights language rises around Civil War and Reconstruction. | `data/speech_markers.parquet`, `data/paragraph_issues.parquet`, `data/issues_meta.json` | Fear÷hope stage 1: 1789–1877 with War of 1812 and Civil War context. Stage 2: 1789–2026 with an explicit scale-change announcement and preserved early-scale ghost/inset. A separate 1850–77 civil-rights/race series marks Emancipation (1863), 13th (1865), 14th (1868), 15th (1870), and Civil Rights Act (1875). | NRC dictionary ratio and descriptive CorEx label. “Civil rights & race” is anchored on slavery, race/racial, discrimination, segregation, equality, emancipation, and the historical corpus term “negro”; it can miss paraphrase and does not measure rights realized. | Scroll stages mirror explicit buttons; `aria-live` announces axis changes; reduced motion disables automatic staging; keyboard buttons and arrow navigation work. | Early stage axes end at 1877; full stage preserves the early scale; civil-rights definition and non-causality language are rendered. |
| **1878–1900 — The procedural presidency.** Gilded Age public wording is legal/procedural; later presidential speech moves toward promotional wording, a register change rather than a competence measure. | `data/speech_markers.parquet` | Stable-axis reveal: Gilded Age presidents only → later eras through 2016 → present endpoint including Donald Trump. Scatter axes are legal/procedural words per 10k and hype words per 10k. | Descriptive declared dictionaries. Registered trend work separately finds the register shift survives genre treatments; the chapter does not turn the scatter into a causal or performance claim. | Three buttons, scroll stages, stable axes announced, symbols and direct labels distinguish cohorts. | First stage contains only 1878–1900 presidents; final stage includes Trump; copy explicitly rejects competence/productivity/policy-depth readings. |
| **1901–1932 — Regulation, world war, and crash.** Agenda change is measured against the Gilded baseline, with different subperiods kept distinct. The sustained corpus crossover from “United States” to “America/American(s)” begins in the 1910 five-year rolling window, not the 1950s. | `data/paragraph_issues.parquet`, `data/speech_markers.parquet` | Shared-scale percentage-point deltas for 1901–10, 1911–20, 1921–30, and 1931–32 revealed cumulatively. Topics: economy/jobs, money/banking, trade/tariffs, war/military, foreign policy. A separate 1880–1932 naming chart shows the measured crossover; World War I entry (1917), crash (1929), and Smoot–Hawley (1930) are context only. | Descriptive CorEx labels and dictionary rates. The naming result is corpus-specific and does not prove a change in national identity by itself. | Four manual/scroll stages, fixed shared delta scale, direct zero baseline, keyboard and reduced-motion behavior. | Civil rights is not a featured measured topic; every stage has the same axes; the 1910 sustained-crossover claim is derived in code/test. |
| **1933–1945 — Recovery becomes mobilization.** FDR's speech agenda hands the floor from domestic recovery to war and foreign affairs. | `data/paragraph_issues.parquet`, `data/speeches.parquet` | Shared-scale slope/dumbbell handoff between 1933–39 and 1940–45 for economy/jobs, agriculture, money/banking, infrastructure, war/military, and foreign policy. Corpus excerpts: “This Nation asks for action, and action now” (1933 inaugural) and “We are now in this war. We are all in it—all the way” (1941 fireside chat). | Descriptive paragraph shares. No claim about program effectiveness. | Direct endpoint labels, shapes as well as colors, source links and text excerpts. | Both excerpts match the local corpus; phases share one scale; domestic and mobilization topics are both visible. |
| **1946–1988 — The broadcast presidency.** The main finding is a change in the corpus's assigned primary communication form; reading level supplies context but does not define communicative complexity. | `data/llm_annotations/speech_annotations.parquet`, `data/speech_stats.parquet` | Stacked primary-medium composition focused on 1946–88 with pre-period context, plus a supporting reading-level line. | Descriptive AI factual taxonomy plus Flesch–Kincaid. `spoken_address` and `broadcast_radio_or_tv` are mutually exclusive primary labels even when a spoken address was broadcast; this is not a channel inventory. | Patterned medium layers, table/text fallback, tooltips, mobile scroll. | Chart title says “primary form assigned within this corpus”; limitations reject “simpler communication” from grade level alone. |
| **1989–2016 — The always-on presidency.** Cable, permanent campaigning, and early digital platforms intensify coverage pressure: modern annual messages cover more effective topics while holding a topic for fewer equivalent words. | `data/coverage_pressure/coverage_pressure.parquet`, `data/coverage_pressure/inference_receipts.parquet`, `data/llm_annotations/paragraph_annotations.parquet`, `data/paragraphs.parquet`, `data/speeches.parquet` | Real earlier and modern annual messages use topic-colored paragraph strips and source excerpts. Stages reveal example-speech breadth, the registered breadth estimate, registered depth estimate, no-change-surprise receipts, then all declared sensitivity arms. No toy A–F strips. | **Preregistered:** modern minus postbellum annual messages: effective topics +2.166 [−0.434, 4.566], Holm-adjusted no-change surprise 4.29%; depth −303.568 equivalent words [−430.880, −184.420], adjusted rate at the 0.01% floor. Joint decision: supported. Sensitivities remain visibly exploratory. | Four stages, explicit controls, readable topic legend, segment tooltips/ARIA labels, excerpts, keyboard/reduced-motion behavior. | The body contains real `doc_name` sources, both estimates/intervals/receipts, every sensitivity treatment, and the required plain-language takeaway. |
| **2017–2026 — Platform dominance.** This period intensifies trends already visible in cable, permanent campaigning, and early digital politics; it is not the birth of platform communication. | `data/speech_markers.parquet` | Era-focused 2001–26 small multiples for hype, doom, boosters, opponent naming, and legal/procedural wording, with 2017 marked as a chapter boundary rather than a causal break. | Descriptive dictionary rates. No inference about persuasion, policy, or all off-corpus platform activity. | Separate y-axes with explicit units; direct labels and dash/symbol differences; responsive layout. | Copy says “intensification,” never “birth”; the obsolete tenure-line sentence is absent; Donald Trump and Joe Biden years are present. |
| **Synthesis — change, continuity, uncertainty.** The full story must answer what changed, persisted, and remains uncertain. | All artifacts above; `data/register/trends.parquet` for the rhetorical weather map | A clearly separated change/continuity/uncertainty matrix covers agenda expansion; written→broadcast→always-on communication; procedural→promotional/partisan register; crisis language; and recurring rights, belonging, war, and identity questions. The hype-versus-doom weather map uses separate axes, era colors/symbols, a chronological trail, and named historical comparisons. | The matrix distinguishes registered results from descriptive patterns and unresolved measurement questions. The weather map is descriptive raw-era register data. | Synthesis boundary is announced in visible text and navigation; the weather map has a table/text fallback and is not color-only. | No recent-period AI-topic comparison bar remains; all five required themes appear under changed/persisted/uncertain headings. |
| **Post-story interlude — Extreme speeches reveal extreme moments.** Outliers are evidence cards, not a chronological chapter. | `data/speech_markers.parquet`, `data/speeches.parquet` | Collapsed appendix/interlude with metric definitions, top and runner-up speeches, matched word examples/counts, rates, medians, percentiles, quotations, speech context, and Miller Center links. | Descriptive dictionary rates among speeches of at least 1,500 words. Lexicon matches do not settle quotation, sarcasm, or context. | Native `<details>` disclosure works with keyboard and without JavaScript. | The interlude appears after synthesis, is expandable, and every record card exposes definition, count, percentile, runner-up, quotation, and source context. |

## Shared interaction contract

1. Sticky progress text must render **date range · era name · section title** for the active
   section; synthesis and appendix use explicit non-date labels.
2. Scroll-driven stages are progressive enhancement. Every staged chart also has visible buttons.
3. Buttons use native keyboard semantics and support left/right arrow navigation. Stage changes
   update an `aria-live` status. No stage is reachable only by scrolling or hovering.
4. `prefers-reduced-motion: reduce` removes entrance and axis transitions and disables
   scroll-initiated stage changes; manual controls remain available.
5. Color is supplemented by symbols, dash/pattern, labels, or text. Mobile views preserve readable
   copy and permit horizontal chart scrolling where necessary.
6. Scale changes are named in visible text and live announcements. Event guides state that markers
   supply context, not causes.

## Historical marker verification

Dates are cross-checked against the U.S. National Archives, the U.S. House and Senate historical
offices, and the State Department Office of the Historian. The chart labels deliberately name a
year or range rather than collapsing multi-step events into a false single date.

Primary-source receipts used in the final check:

- State Department Office of the Historian: [Monroe Doctrine, 1823](https://history.state.gov/milestones/1801-1829/monroe);
  [Texas annexation, Oregon Treaty, and Treaty of Guadalupe-Hidalgo, 1845–48](https://history.state.gov/milestones/1830-1860/texas-annexation).
- U.S. Senate and Library of Congress: [Tariff/nullification context](https://www.senate.gov/artandhistory/history/common/generic/Speeches_HaynesReply.htm);
  [Second Bank veto and Bank War](https://www.senate.gov/about/parties-leadership/censure-president-jackson.htm).
- National Archives: [Emancipation Proclamation, 1863](https://www.archives.gov/milestone-documents/emancipation-proclamation);
  [Reconstruction Amendments](https://www.archives.gov/exhibits/documented-rights/exhibit/);
  [Civil Rights Act of 1875 timeline](https://www.archives.gov/education/lessons/brown-v-board/timeline.html).

## Implementation verification

- The generator publishes the nine dated chapters in order, then a visible chronology boundary,
  synthesis, and a collapsed post-story appendix.
- The registered coverage-pressure chapter uses the real 1894 Cleveland and 2016 Obama annual
  messages, all confirmatory receipts, and every declared sensitivity treatment.
- Focused Story V2/site/registry tests pass 18/18; the complete suite passes 2,356/2,356.
- The deterministic `--inline` rebuild validates 72 HTML pages and 161 JSON shards. Generated
  JavaScript syntax, Python compilation, and `git diff --check` pass.
- In-app browser QA covers all chapters at 1280×720 and 390×844, manual and arrow-key stages,
  real hover text, responsive local chart/table scrolling, the five record cards, and the
  reduced-motion branch. No browser console warnings or errors remain.
- The preview is local only at `http://127.0.0.1:8010/`; no public deployment was performed.

## Build and acceptance commands

The completion handoff must report exact results for:

```bash
arch -x86_64 .venv/bin/python -m pytest tests/test_story_v2.py tests/test_site_validation.py tests/test_metric_registry.py
arch -x86_64 .venv/bin/python -m presidential_profiles.site
arch -x86_64 .venv/bin/python -m pytest
python3 -m http.server 8010 --directory docs
```

Also run internal-link/anchor/JSON/metric-registry validation through the site build, extract and
syntax-check generated static JavaScript, and inspect every chapter at desktop and mobile widths
in the in-app browser. Verify manual and scroll stages, reduced motion, keyboard use, tooltips,
event-marker readability, and responsive layouts. Do not deploy.
