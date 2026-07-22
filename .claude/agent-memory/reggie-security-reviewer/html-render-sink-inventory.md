---
name: html-render-sink-inventory
description: Which docs/ render sinks escape and which don't — the proven inventory for presidential_profiles, plus the two sinks that LOOK dangerous and are safe by construction
metadata:
  type: project
---

Established 2026-07-21 during `topic-method-comparison`, the first task to route a
**hand-editable data file** (`data/topic_display_names.json`) into the GitHub-Pages HTML.
This is the concrete answer to the forward-looking note in [[derived-analysis-module-recipe]].

**Why:** every future task that adds a new string source to the site needs to know which
sinks escape. Re-deriving this costs an hour; it changes rarely.

**How to apply:** when a task introduces a NEW string source reaching `docs/`, check it
against this table rather than re-tracing. Only flag the unescaped sinks.

## UNESCAPED — raw f-string interpolation into HTML (pre-dates any given task)
- `issues_site.py` `render_issue`: `<title>{label}`, `<h1>{label}`, `<p>...{label.lower()}`
- `issues_site.py` `render_issue_index`: `<div class="name">{e["label"]}`
- `profiles_site.py` `_issue_cards_html`: `<span class="i-name">{issue}`
- `profiles_site.py` `render_index`: `<div class="issue">↑ {top_issue}`
- `explorer.py` `renderChips`: `chip.innerHTML = \`...${s.label}...\`` — **DOM sink**, so the
  working payload is `<img src=x onerror=...>`, NOT `<script>` (innerHTML doesn't run script tags).
  Chain: names file -> `discovered_labels()` -> `topic_data[label]` key -> `docs/explorer/meta.json`
  -> `fetch` -> `Object.keys(META.topics)` -> `addTopic` -> `series.push({label})` -> `innerHTML`.
- `issues_site.py:89` DOES `html_mod.escape(q)` on the corpus quote — so the module already
  imports `html`; the label just isn't run through it. Makes the fix a one-liner and weakens
  any "escaping is architecturally absent here" defence.

## SAFE BY CONSTRUCTION — verified empirically, do NOT re-flag
- **Plotly `subplot_titles=` and `hovertemplate="..." + "<extra>" + title + "</extra>"`**
  (`site.py::_small_multiples`, and its `_banded_small_multiples` clone added by
  `topic-chart-upgrades`): the display name reaches plotly *config*, so it is serialized by the
  plotly encoder below and cannot break the `<script>`. Residual risk is cosmetic only — plotly
  renders a tag subset in annotation/hover text (`<b> <i> <br> <a>`, with a protocol whitelist on
  `href`), and a `%{...}` in a title would be read as a hover token. Grade INFO; the pattern
  pre-dates any given task, so a new panel grid copying it is not a regression.
- **Plotly `<script>` embeds** (`pio.to_json(fig)` inside `<script>const FIG = ...`): plotly's
  encoder emits `<` / `>` / `/`, so `</script><script>alert(1)</script>` in a
  chart title cannot break out. Verified by injecting it into a `yaxis.title`. Note plain
  `json.dumps` does NOT escape `/` — the safety is plotly's, not Python's, so a hand-rolled
  `json.dumps` into a `<script>` block would be a real finding.
- **`issues_site.issue_slug` -> filename**: delegates to `profiles.slug`, which is
  `re.sub(r"[^a-z]+", "-", s.lower()).strip("-")`. Proven: `../../../etc/passwd` -> `etc-passwd`.
  No traversal, no extension control, digits/dots/slashes all gone. Only residual risk is
  slug COLLISION (two display names -> one file, silent page loss) or an all-symbol name
  slugging to `""` -> `docs/issues/.html`. Data-integrity LOW, never a path finding.
- `docs/` is the published output; `notes/*.md` is NOT under `docs/` and is not linked from it —
  report markdown stays a human-read artifact.
