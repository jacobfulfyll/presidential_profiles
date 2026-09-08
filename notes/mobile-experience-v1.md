# Mobile experience v1

Status: **implemented and verified locally on 2026-09-08; not deployed**

This contract improves the generated site for phones without changing its analytical content or
desktop presentation. It covers all 73 canonical HTML routes and the physical
`index_selfcontained.html` Story copy.

## Responsive boundary and desktop lock

- Phone presentation applies through `max-width: 760px`.
- A second, narrow exception applies only to coarse pointers at `max-height: 500px` and
  `max-width: 932px`; it provides the compact global navigation and makes secondary rails static
  for phone landscape.
- Fine-pointer widths at 768px and above retain the prior layout and interaction behavior.
- Shared mobile markup may be present but must be `display:none` and layout-inert on desktop.
- Before/after rendered screenshots and landmark geometry at 768, 1024, 1280, and fine-pointer
  844×390 are the desktop acceptance evidence.

## Phone interaction contract

- The global navigation is one 56px sticky row: brand mark, current section, and a native
  `details` Menu. It exposes the same destinations and current-page states as desktop, remains
  usable without JavaScript, closes after link activation or outside activation, and closes on
  Escape with focus returned to its summary.
- Actionable navigation, buttons, selects, disclosures, deep links, and label-backed choices have
  an effective activation box of at least 44×44px. Inline prose links are exempt.
- General prose remains at least 16px, control labels at least 14px, visible metadata at least
  12px, and SVG/chart text at least 10.5px or has an always-available exact readout.
- Portrait secondary navigation remains sticky and anchor offsets clear the full sticky stack.
  Secondary rails are static on short coarse-pointer landscape screens.
- The document itself never scrolls horizontally. Dense charts, tables, and networks may use a
  named, focusable local scroller with a visible phone cue and reachable horizontal endpoint.
- Reduced-motion and forced-color treatments preserve navigation state, boundaries, and focus.

## Page-family decisions

- **Story and Summary:** reflow squeezed evidence, enlarge staged controls and metadata, clear
  sticky anchors, wrap long sources, and scroll dense analytical surfaces locally.
- **Compare:** use mobile-only radar margins and height with a plot diameter of at least 220px at
  320px; redraw after rotation or width changes without changing the desktop Plotly branch.
- **Explore:** keep both the no-JavaScript and hydrated SVG inside a named 700px local viewport;
  enlarge SVG labels and hit strokes to remain readable and touchable after scaling.
- **Profiles:** retain the semantic relationship list, use the reduced-node graph through 760px,
  and place its 720px canvas in a named local viewport.
- **Issues and Data Trust:** preserve existing Plotly/table fallbacks, enlarge local controls, and
  use a 720px local viewport for the Era Choices sensitivity matrix.
- Directories and Feedback receive the shared shell improvements only unless rendered acceptance
  identifies a template-specific defect.

## Protected and excluded surfaces

- Routes, navigation destinations, data/public schemas, analytical values, section order, and
  default disclosures do not change.
- Story V2 historical/statistical boundaries and named evidence artifacts remain authoritative.
- Frozen paid annotations and governed derived layers are not regenerated or hand-edited.
- Existing resource-loading boundaries remain; this pass adds no eager dependency and does not
  split the shared static HTML payload by device.
- Publication and deployment are outside scope.

## Acceptance

- `scripts/audit_mobile_site.mjs` discovers and audits all physical HTML outputs at 320×568,
  390×844, 430×932, 760×900, and the 768×1024 desktop boundary. It also checks coarse-pointer
  844×390 landscape, root/nested native menus, JavaScript-disabled evidence, and the Explore
  request-failure fallback.
- Representative interaction routes are Story, Summary, Compare, Explore, Feedback, both
  directories, rich/thin profiles, dense/sparse issues, and all five Data Trust routes.
- Required gates are targeted tests, the full pytest suite, deterministic site generation,
  inline/external JavaScript parsing, both rendered browser audits, Python compilation, protected
  artifact comparison, whitespace review, desktop baseline comparison, and Reggie Doctor.

## Verification receipt

- The required x86_64 inline build regenerated `docs/`, including `index_selfcontained.html`, and
  validated 73 canonical HTML pages plus 429 JSON shards. All 45 profile pages and all 16 issue
  pages were generated from source.
- `scripts/audit_mobile_site.mjs` passed all 74 physical HTML files at 320×568, 390×844, 430×932,
  760×900, and the 768×1024 desktop boundary: 370 route/viewport observations with no overflow,
  target, type, scroller, clipping, anchor, focus, chart, console, hydration, link, or resource
  failure. Its separate interaction sequence passed native/no-JavaScript menus, Story stages,
  Compare's 220px radar floor, Explore search/history and request failure, profile/Explore/Era touch
  scrolling, 200% text, reduced motion, forced colors, and coarse-pointer 844×390 landscape.
- Sixty untracked pre-edit screenshots and geometry records covered 15 representative routes at
  768, 1024, 1280, and fine-pointer 844×390. Post-change screenshot comparison was exact or
  negligible for 57 states; the three Summary differences were confined to its pre-existing reveal
  animation, with static regions matching. The settled 60-state structural/geometry audit passed,
  including exact breakpoint mode, visible-element inventory, classes, fonts, display/position,
  widths, and x placement.
- The complete required pytest run passed 3,025 tests with the repository's 64 known warnings. The
  dependency-free browser wrapper passed inside that run. All 129 inline scripts and every generated
  external JavaScript file parsed, Python compilation and `git diff --check` passed, and the existing
  Data Trust browser audit passed five pages at three widths.
- `data/llm_annotations/` remains byte-identical at fingerprint
  `9a8ac49ed29d63c94f4d7bdeb9f753cf0f38b45b8202ed7e7afa4de1c4297b54`; no governed derived-data
  directory changed. Reggie Doctor reports 0 errors, 0 warnings, and 3 informational findings. No
  file was staged, committed, pushed, or deployed.
