"""Generate the Explore v2 page from the governed public projection."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import explore_assets, explore_projection, topic_quality
from .figures import REPO_ROOT
from .site_style import PAGE_CSS


EXPLORER_DIR = REPO_ROOT / "docs" / "explorer"
SLOT_COLORS = ("#2A78D6", "#9B6200", "#008300")
SLOT_DASHES = ("", "9 5", "2 5")


def _validate_catalog_display_names(
    projection: explore_projection.ExploreProjectionBundle,
) -> None:
    """Refuse page rendering when CorEx labels bypass the governed registry."""
    names = topic_quality.load_names()
    for item in projection.index["catalog"]["broad_issues"]:
        source_name = str(item["source_label"])
        expected = topic_quality.display_name(source_name, names)
        if item["label"] != expected:
            raise explore_projection.ExploreProjectionError(
                "Explore broad-issue display label drifted from the registry: "
                f"{source_name!r}"
            )


def _default_models(
    projection: explore_projection.ExploreProjectionBundle,
) -> list[dict[str, Any]]:
    axis = projection.index["lexical_axis"]
    models = []
    for order, query in enumerate(("tariff", "freedom", "border"), start=1):
        filename = explore_projection.lexical_shard_path(
            "lexical-exact-unigram", query
        )
        payload = json.loads(projection.files[filename])
        record = payload["records"].get(query)
        if record is None:
            raise explore_projection.ExploreProjectionError(
                f"default series is missing from its shard: {query}"
            )
        rows = explore_projection._lexical_public_rows(
            selection_order=order,
            series_id=f"lexical-exact:{query}",
            label=query,
            kind="lexical-exact",
            instrument="Exact lexical form",
            query=query,
            grouping_active=False,
            family_label=None,
            family_forms=None,
            record=record,
            axis=axis,
        )
        models.append(
            {
                "id": f"lexical-exact:{query}",
                "label": query,
                "instrument": "Exact lexical form",
                "rows": rows,
            }
        )
    return models


def _line_path(
    rows: Sequence[Mapping[str, Any]],
    *,
    y_max: float,
    width: float = 900,
    height: float = 300,
) -> str:
    left, right, top, bottom = 58.0, 14.0, 14.0, 38.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    parts: list[str] = []
    drawing = False
    for row in rows:
        value = row.get("display_value")
        if value is None:
            drawing = False
            continue
        x = left + (float(row["x"]) - 1785) / (2026 - 1785) * plot_width
        y = top + plot_height - max(0.0, min(y_max, float(value))) / y_max * plot_height
        parts.append(f"{'L' if drawing else 'M'}{x:.2f} {y:.2f}")
        drawing = True
    return "".join(parts)


def _default_svg(models: Sequence[Mapping[str, Any]]) -> str:
    width, height = 900, 300
    left, right, top, bottom = 58, 14, 14, 38
    values = [
        float(row["display_value"])
        for model in models
        for row in model["rows"]
        if row["display_value"] is not None
    ]
    y_max = (max(values) * 1.05) if values and max(values) > 0 else 1.0
    grid = []
    for fraction in (0.0, 0.5, 1.0):
        y = top + (height - top - bottom) * (1 - fraction)
        grid.append(
            f'<line class="grid-line" x1="{left}" y1="{y:.2f}" '
            f'x2="{width-right}" y2="{y:.2f}"/>'
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end">'
            f'{y_max*fraction:.1f}</text>'
        )
    for year in (1785, 1850, 1900, 1950, 2000, 2026):
        x = left + (year - 1785) / (2026 - 1785) * (width - left - right)
        anchor = "start" if year == 1785 else "end" if year == 2026 else "middle"
        grid.append(
            f'<line class="grid-line" x1="{x:.2f}" y1="{top}" '
            f'x2="{x:.2f}" y2="{height-bottom}"/>'
            f'<text x="{x:.2f}" y="{height-12}" text-anchor="{anchor}">{year}</text>'
        )
    paths = []
    for index, model in enumerate(models):
        path = _line_path(model["rows"], y_max=y_max)
        dash = f' stroke-dasharray="{SLOT_DASHES[index]}"' if SLOT_DASHES[index] else ""
        paths.append(
            f'<path class="trajectory" d="{path}" stroke="{SLOT_COLORS[index]}"{dash}/>'
        )
    return (
        '<svg class="trend-svg" viewBox="0 0 900 300" role="img" '
        'aria-labelledby="fallback-chart-title fallback-chart-desc">'
        '<title id="fallback-chart-title">Default word trends: tariff, freedom, and border</title>'
        '<desc id="fallback-chart-desc">Exact-form rates per 10,000 indexed words in centered five-year windows. '
        'The three lines use solid, long-dashed, and dotted patterns as well as different colors.</desc>'
        + "".join(grid)
        + f'<line class="axis-line" x1="{left}" y1="{height-bottom}" '
        f'x2="{width-right}" y2="{height-bottom}"/>'
        + "".join(paths)
        + "</svg>"
    )


def _fallback_legend(models: Sequence[Mapping[str, Any]]) -> str:
    items = []
    for index, model in enumerate(models):
        dash = f' stroke-dasharray="{SLOT_DASHES[index]}"' if SLOT_DASHES[index] else ""
        items.append(
            '<span class="legend-button" aria-hidden="true">'
            '<svg class="legend-symbol" viewBox="0 0 30 14">'
            f'<line x1="1" y1="7" x2="29" y2="7" stroke="{SLOT_COLORS[index]}" '
            f'stroke-width="2.6"{dash}/></svg>'
            f'<span>{chr(65+index)} · {escape(str(model["label"]))}</span></span>'
        )
    return "".join(items)


def _format_value(value: Any) -> str:
    return "Unknown" if value is None else f"{float(value):.2f} uses per 10,000 indexed words"


def _fallback_table(model: Mapping[str, Any]) -> str:
    body = []
    for row in model["rows"]:
        period = f'{row["period_start"]}–{row["period_end"]}'
        support = (
            "Window does not clear the word floor"
            if row["denominator_count"] is None
            else f'{int(row["denominator_count"]):,} indexed words'
        )
        status = {
            "observed": "Observed",
            "observed_zero": "Observed zero",
            "unknown_low_words": "Not published: 20,000 indexed words or fewer",
        }.get(str(row["value_status"]), str(row["value_status"]))
        cells = (
            period,
            _format_value(row["display_value"]),
            "Not published for this measure",
            support,
            status,
        )
        body.append(
            "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in cells) + "</tr>"
        )
    caption = (
        f'{escape(str(model["label"]))}. Exact lexical form; centered five-year periods; '
        "uses per 10,000 indexed source-document words; complete source-document transcripts."
    )
    return (
        '<div class="table-wrap"><table class="exact-table">'
        f'<caption>{caption}</caption><thead><tr>'
        '<th scope="col">Period</th><th scope="col">Estimate</th>'
        '<th scope="col">Published uncertainty</th><th scope="col">Evidence / support</th>'
        '<th scope="col">Status</th></tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_page(projection: explore_projection.ExploreProjectionBundle) -> str:
    """Render semantic server HTML with a complete default no-JavaScript path."""
    if (
        projection.index.get("schema_version") != explore_projection.INDEX_SCHEMA
        or projection.families.get("schema_version") != explore_projection.FAMILY_SCHEMA
        or projection.topics.get("schema_version") != explore_projection.TOPIC_SCHEMA
    ):
        raise explore_projection.ExploreProjectionError(
            "Explore page received an incompatible projection"
        )
    _validate_catalog_display_names(projection)
    models = _default_models(projection)
    fallback_tables = "".join(_fallback_table(model) for model in models)
    fallback_selected = ", ".join(escape(str(model["label"])) for model in models)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Explore language and topics - Presidential Profiles</title>
<style>{PAGE_CSS}</style>
<link rel="stylesheet" href="assets/{explore_assets.CSS_FILE}">
<script defer src="assets/{explore_assets.JS_FILE}"></script>
</head>
<body>
<main class="explore-page" id="explore-app"
 data-index-url="explorer/{explore_projection.INDEX_FILE}"
 data-families-url="explorer/{explore_projection.FAMILY_FILE}"
 data-topics-url="explorer/{explore_projection.TOPIC_FILE}">
  <header>
    <h1>Explore language and topics</h1>
    <p class="explore-intro">Find exact words, audited word families, broad deterministic issues,
    detailed AI-labeled topics, and three case-sensitive named acronyms. Each measure keeps its
    own unit and published time grain.</p>
  </header>

  <section class="explore-section" aria-labelledby="choose-series-heading">
    <div class="selection-heading">
      <h2 id="choose-series-heading">Choose series</h2>
      <span class="selection-count" id="selected-count">3 of 6 selected</span>
      <div class="selection-actions">
        <button class="secondary-button" id="clear-series" type="button" data-hydration-control disabled>Clear</button>
        <button class="secondary-button" id="reset-explore" type="button" data-hydration-control disabled>Reset</button>
      </div>
    </div>
    <p class="fallback-selected server-only"><strong>Selected:</strong> {fallback_selected}.</p>
    <div class="selected-list" id="selected-list"></div>
    <div class="selection-grid">
      <section class="selector-card" aria-labelledby="word-entry-heading">
        <h3 id="word-entry-heading">Add a word or phrase</h3>
        <p>Use one word or a two-word phrase. Punctuation separates exact words; audited spelling
        normalization applies only when grouping is on.</p>
        <fieldset class="word-mode">
          <legend class="visually-hidden">How to count the next word</legend>
          <label><input id="word-mode-exact" name="word-mode" type="radio" value="exact" checked data-hydration-control disabled> Exact form</label>
          <label><input id="word-mode-family" name="word-mode" type="radio" value="family" data-hydration-control disabled> Group word forms</label>
        </fieldset>
        <div class="word-entry">
          <input id="word-query" type="text" autocomplete="off" placeholder="e.g. tariff or middle class"
            aria-describedby="word-help" data-hydration-control disabled>
          <button class="primary-button" id="add-word" type="button" data-hydration-control disabled>Add word</button>
        </div>
        <p id="word-help">Exact words need 30 corpus uses; exact phrases need 15. Grouped searches
        use the existing audited family map and do not claim that every form is linguistically equivalent.</p>
      </section>
      <section class="selector-card" aria-labelledby="catalog-heading">
        <h3 id="catalog-heading">Browse governed series</h3>
        <label class="visually-hidden" for="catalog-search">Search governed series</label>
        <input class="catalog-search" id="catalog-search" type="search" placeholder="Search labels, definitions, or anchor words"
          data-hydration-control disabled>
        <div class="catalog-groups" id="catalog-groups"></div>
        <p class="catalog-no-results" id="catalog-no-results" hidden></p>
        <p class="server-only">JavaScript enables the searchable, grouped catalog. Its complete
        accessible inventory is available in <a href="explorer/{explore_projection.SERIES_CATALOG_FILE}">series catalog CSV</a>.</p>
      </section>
    </div>
    <div class="control-bar">
      <label class="uncertainty-control"><input id="uncertainty-toggle" type="checkbox" checked
        data-hydration-control disabled> Show uncertainty ranges for topic series</label>
      <label class="context-control" for="context-select">Historical context
        <select class="context-select" id="context-select" data-hydration-control disabled><option>None</option></select>
      </label>
    </div>
    <details class="guided" id="guided">
      <summary>Guided comparisons</summary>
      <div class="guided-body">
        <div class="preset-list" id="preset-list"></div>
        <p class="preset-note" id="preset-note">Loading a guide replaces the current selection with its editable exact words.</p>
        <div class="server-only"><p>The nine guides cover crisis language, superlative politics,
        legal and procedural vocabulary, national unity, decline and restoration, war and peace,
        economic hardship, immigration, and democratic institutions. JavaScript is required to
        load a guide interactively.</p></div>
      </div>
    </details>
    <p class="status-line" id="explore-status" role="status" aria-live="polite" aria-atomic="true"></p>
    <div class="failure-panel" id="explore-failure" role="alert" hidden></div>
    <noscript><p class="failure-panel">Interactive selection requires JavaScript. The default
    chart, all default exact values, and three complete downloads remain available.</p></noscript>
  </section>

  <section class="explore-section" aria-labelledby="trends-heading">
    <h2 id="trends-heading">Trends</h2>
    <p class="chart-summary" id="chart-summary">Three exact-word series in centered five-year
    windows. The static fallback uses native rates and a zero baseline.</p>
    <div class="chart-panels" id="chart-panels">
      <section class="trend-panel server-only">
        <h3>Words and named acronyms</h3>
        <p class="panel-unit">Uses per 10,000 indexed words</p>
        <div class="series-legend">{_fallback_legend(models)}</div>
        <p class="chart-swipe-cue">Swipe chart horizontally →</p>
        <div class="trend-scroll" tabindex="0" role="region"
          aria-label="Default word trends chart; scroll horizontally on narrow screens">
          {_default_svg(models)}
        </div>
      </section>
    </div>
    <div class="chart-readout" id="chart-readout">
      <strong>Static default</strong><span>Use the exact-value disclosure below for every published period.</span>
    </div>
    <div class="visually-hidden" id="chart-live" aria-live="polite" aria-atomic="true"></div>
    <div class="download-row">
      <button class="download-selected client-only primary-button" id="download-selected" type="button">Download selected exact values (CSV)</button>
      <a class="download-selected server-only" href="explorer/{explore_projection.DEFAULT_VALUES_FILE}" download>Download default exact values (CSV)</a>
      <p>The selected download includes every period, range, support field, provenance component,
      and explicit unknown or zero state—not only what is visible in the chart.</p>
    </div>
    <details class="exact-details" id="exact-values">
      <summary>Exact values and support</summary>
      <div class="exact-body">
        <div class="exact-control client-only">
          <label for="exact-series">Series</label>
          <select class="exact-series" id="exact-series"></select>
        </div>
        <div class="server-only">{fallback_tables}</div>
        <div id="exact-table-mount"></div>
      </div>
    </details>
  </section>

  <section class="explore-section method-compact" aria-labelledby="explore-method-heading">
    <h2 id="explore-method-heading">Data and methods</h2>
    <p><strong>Measures.</strong> Word and acronym rates use complete source-document transcripts
    and source-document word totals. Broad issues use deterministic-model paragraph shares in
    five-year periods. Detailed topics use exploratory AI-labeled paragraph shares in nine named
    eras. Explore is not filtered to actual-president paragraph speakers, and overlapping topic
    labels need not sum to 100%.</p>
    <p>Topic uncertainty comes from the governed speech-clustered bands. A dotted line marked † is
    low-support; ‡ means a range was not estimated; ◇ means the estimate exists but the bootstrap
    could not resolve a range. Missing bounds are unknown, never zero. See
    <a href="methodology.html">Methods</a> and <a href="data-quality.html">Data Quality</a>.</p>
    <p class="server-only">Complete non-interactive downloads:
      <a href="explorer/{explore_projection.DEFAULT_VALUES_FILE}">default values</a> ·
      <a href="explorer/{explore_projection.TOPIC_VALUES_FILE}">all governed topic values</a> ·
      <a href="explorer/{explore_projection.SERIES_CATALOG_FILE}">series catalog</a>.
    </p>
  </section>
  <footer class="source-footer">
    <p>Source: Miller Center presidential-speech corpus, 1,057 source documents, 1789–2026.
    Lexical forms are indexed only when they meet the published corpus floor. Frozen annotations
    and governed uncertainty artifacts are read, not recalculated, by this page.</p>
  </footer>
</main>
</body>
</html>
"""


def write_page(
    projection: explore_projection.ExploreProjectionBundle,
    site_dir: Path | None = None,
) -> Path:
    """Write Explore HTML and its two external renderer assets."""
    site_dir = REPO_ROOT / "docs" if site_dir is None else site_dir
    site_dir.mkdir(parents=True, exist_ok=True)
    explore_assets.write_assets(site_dir)
    output = site_dir / "explorer.html"
    output.write_text(render_page(projection))
    display = output.relative_to(REPO_ROOT) if output.is_relative_to(REPO_ROOT) else output
    print(f"  wrote {display}")
    return output


def build_explorer_data(ai_data: dict | None = None) -> explore_projection.ExploreProjectionBundle:
    """Compatibility entry point: build and atomically publish Explore v2 data."""
    del ai_data
    projection = explore_projection.build_projection()
    explore_projection.write_public_projection(projection, REPO_ROOT / "docs")
    return projection
