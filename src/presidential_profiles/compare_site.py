"""Generate the narrative Compare V3 page from Profile V3 and Compare projections."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import html
import json
import math
from pathlib import Path
from typing import Any

from . import compare_projection, metrics, profiles_site
from .compare_assets import (
    AGENDA_COMPARISON_CSS,
    AGENDA_COMPARISON_JS,
    COMPARE_V3_CSS,
    COMPARE_V3_JS,
)
from .html_safety import json_for_script
from .profiles import president_chronology, public_display_name, slug
from .site_style import FONT, PAGE_CSS


COMPARE_PAYLOAD_VERSION = "president-comparison-page-v3"
DOWNLOAD_SCHEMA_VERSION = "president-comparison-v3"
DEFAULT_PRESIDENTS = ("Abraham Lincoln", "Franklin D. Roosevelt")
FOCUSED_TOPIC_LIMIT = 3
SLOT_LABELS = ("A", "B", "C")
SLOT_STYLES = (
    {"color": "#2A78D6", "symbol": "circle", "plotly_symbol": "circle", "dash": "solid"},
    {"color": "#9B6200", "symbol": "square", "plotly_symbol": "square", "dash": "dash"},
    {"color": "#008300", "symbol": "diamond", "plotly_symbol": "diamond", "dash": "dot"},
)
COMPARE_CHART_IDS = {
    "compare_rhetoric_corpus", "compare_rhetoric_ai", "compare_agenda_broad",
}
ASSET_FILES = {
    "compare-v3.js": COMPARE_V3_JS,
    "compare-v3.css": COMPARE_V3_CSS,
    "agenda-comparison-v1.js": AGENDA_COMPARISON_JS,
    "agenda-comparison-v1.css": AGENDA_COMPARISON_CSS,
}


def _short_president_names(display_names: Sequence[str]) -> dict[str, str]:
    """Return compact, collision-safe names for dense comparison labels."""
    surnames = {
        name: "Van Buren" if name.endswith("Van Buren") else name.rsplit(" ", 1)[-1]
        for name in display_names
    }
    counts = {
        surname: sum(candidate == surname for candidate in surnames.values())
        for surname in set(surnames.values())
    }
    result = {}
    for name, surname in surnames.items():
        if counts[surname] == 1:
            result[name] = surname
            continue
        given = name[: -(len(surname) + 1)].split()
        initials = "".join(f"{part[0].upper()}." for part in given if part)
        result[name] = f"{initials} {surname}"
    return result


def _finite_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Compare field {field} must be finite")
    return number


def _measure_rows(shared: dict, layer: str) -> list[dict]:
    ai_missing = layer == "ai_radar" and not shared.get("ai")
    rows = []
    for row in profiles_site.profile_measure_rows(shared, layer):
        percentile = _finite_number(row.get("percentile"), f"{layer}.{row['key']}.percentile")
        if percentile is None:
            if ai_missing:
                rank_display = "N/A · AI labels unavailable"
            elif shared["sample"]["thin_record"]:
                rank_display = "N/A · Insufficient record"
            else:
                rank_display = "N/A · Not ranked"
        else:
            rank_display = row["percentile_display"]
        rows.append({
            "key": row["key"], "label": row["label"], "unit": row["unit"],
            "absolute": _finite_number(row.get("absolute"), f"{layer}.{row['key']}.absolute"),
            "absolute_display": row["absolute_display"], "percentile": percentile,
            "rank_display": rank_display,
        })
    return rows


def _measure_catalogs(public_profiles: Mapping[str, dict], order: Sequence[str]) -> dict:
    if not order:
        return {"corpus": [], "ai": []}
    first = public_profiles[order[0]]
    catalogs = {
        output: [
            {"key": row["key"], "label": row["label"], "unit": row["unit"]}
            for row in _measure_rows(first, layer)
        ]
        for output, layer in (("corpus", "rhetorical_radar"), ("ai", "ai_radar"))
    }
    for president in order:
        for output, layer in (("corpus", "rhetorical_radar"), ("ai", "ai_radar")):
            observed = [row["key"] for row in _measure_rows(public_profiles[president], layer)]
            if observed != [row["key"] for row in catalogs[output]]:
                raise ValueError(f"{president} {output} measure order drifted")
    return catalogs


def compact_profile_projection(
    shared: dict,
    measure_catalogs: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
    actual_support: Mapping[str, Any] | None = None,
) -> dict:
    """Keep only identity, support, routes, and the two Profile V3 fingerprints."""
    profiles_site.validate_profile_view_model(shared)
    if measure_catalogs is None:
        measure_catalogs = _measure_catalogs({shared["president"]: shared}, [shared["president"]])
    president = str(shared["president"])
    president_id = str(shared["slug"])
    support = actual_support or {}
    measures = {}
    for output, layer in (("corpus", "rhetorical_radar"), ("ai", "ai_radar")):
        rows = _measure_rows(shared, layer)
        if [row["key"] for row in rows] != [row["key"] for row in measure_catalogs[output]]:
            raise ValueError(f"{president} {output} measure order drifted")
        measures[output] = [{
            "absolute": row["absolute"], "absolute_display": row["absolute_display"],
            "percentile": row["percentile"], "rank_display": row["rank_display"],
        } for row in rows]
    return {
        "president_id": president_id, "president": president,
        "display_name": public_display_name(president), "slug": president_id,
        "party": shared.get("party"),
        "years": {"first": int(shared["years"]["first"]), "last": int(shared["years"]["last"])},
        "source_document_speech_count": int(shared["sample"]["n_speeches"]),
        "source_document_support_state": "thin" if shared["sample"]["thin_record"] else "supported",
        "actual_speaker_appearance_count": (
            int(support["eligible_president_appearance_count"])
            if support.get("eligible_president_appearance_count") is not None else None
        ),
        "actual_speaker_support_state": support.get("president_support_status", "unavailable"),
        "measures": measures,
        "profile_url": f"presidents/{president_id}.html",
        "profile_data_url": f"data/presidents/{president_id}.json",
        "portrait_url": f"portraits/{president_id}.png",
    }


def comparison_payload(
    public_profiles: Mapping[str, dict], *, require_complete: bool = False,
    projection: compare_projection.CompareProjectionBundle | None = None,
) -> dict:
    order_names = president_chronology(public_profiles, require_complete=require_complete)
    catalogs = _measure_catalogs(public_profiles, order_names)
    support_by_id: dict[str, dict] = {}
    if projection is not None:
        compare_projection.validate_projection(projection)
        support_by_id = {row["president_profile_id"]: row for row in projection.agenda_index["president_support"]}
    presidents = {
        slug(name): compact_profile_projection(public_profiles[name], catalogs, support_by_id.get(slug(name)))
        for name in order_names
    }
    short_names = _short_president_names(
        [str(record["display_name"]) for record in presidents.values()]
    )
    for record in presidents.values():
        record["short_name"] = short_names[str(record["display_name"])]
    order = [slug(name) for name in order_names]
    defaults = [slug(name) for name in DEFAULT_PRESIDENTS if slug(name) in presidents]
    defaults.extend(item for item in order if item not in defaults)
    payload = {
        "schema_version": COMPARE_PAYLOAD_VERSION,
        "profile_schema_version": profiles_site.PROFILE_SCHEMA_VERSION,
        "order": order, "defaults": defaults[:2], "presidents": presidents,
        "measure_catalogs": catalogs,
        "slot_styles": list(SLOT_STYLES), "font": FONT,
        "urls": {
            "agenda_index": f"data/compare/{compare_projection.AGENDA_INDEX_FILE}",
            "evidence_index": f"data/compare/{compare_projection.EVIDENCE_INDEX_FILE}",
            "ai_topic_csv": f"data/compare/{compare_projection.AI_VALUES_FILE}",
            "plotly": "assets/plotly-3.0.1.min.js",
        },
    }
    json.dumps(payload, allow_nan=False)
    return payload


def build_payload(
    data: dict, display_issues: list[str], *, profile_views: dict[str, dict] | None = None,
    projection: compare_projection.CompareProjectionBundle | None = None,
) -> dict:
    views = profile_views or profiles_site.build_profile_view_models(
        data, display_issues, require_complete=len(data["scores"]) == 45
    )
    public_profiles = {president: profiles_site.profile_public_payload(view) for president, view in views.items()}
    return comparison_payload(
        public_profiles, require_complete=len(public_profiles) == 45, projection=projection
    )


def normalize_selection(
    order: Sequence[str], slug_to_name: Mapping[str, str], query: Mapping[str, str | None],
) -> tuple[list[str], bool]:
    """Python mirror of URL normalization; invalid C is omitted, never replaced."""
    order = list(order)
    if len(order) < 2:
        return order, False
    preferred = [name for name in DEFAULT_PRESIDENTS if name in order]
    preferred.extend(name for name in order if name not in preferred)
    default_a, default_b = preferred[:2]
    raw_a, raw_b, raw_c = query.get("a"), query.get("b"), query.get("c")
    requested_a = slug_to_name.get(raw_a or "")
    a = requested_a or default_a
    corrected = raw_a is not None and requested_a is None

    def adjacent(anchor: str, used: set[str]) -> str:
        position = order.index(anchor)
        candidates = order[position + 1:] + list(reversed(order[:position])) + order
        return next(name for name in candidates if name not in used)

    b = slug_to_name.get(raw_b or "")
    if b is None or b == a:
        if raw_b is None and requested_a is not None:
            b = adjacent(a, {a})
        else:
            b = default_b if default_b != a else adjacent(a, {a})
        corrected = corrected or raw_b is not None
    selected = [a, b]
    if raw_c not in (None, ""):
        c = slug_to_name.get(raw_c)
        if c is None or c in selected:
            corrected = True
        else:
            selected.append(c)
    return selected, corrected


def _fmt_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def _server_selection(payload: Mapping[str, Any], selected: Sequence[str]) -> str:
    cards = []
    for index, president_id in enumerate(selected):
        president = payload["presidents"][president_id]
        actual = president["actual_speaker_appearance_count"]
        cards.append(f'''<article class="selection-card slot-{index + 1}">
<div class="selection-card-head"><img src="{html.escape(president['portrait_url'])}" width="44" height="44" alt=""><strong>{SLOT_LABELS[index]} · {html.escape(president['display_name'])}</strong></div>
<p class="selection-meta">{html.escape(str(president['party']))} · {president['years']['first']}–{president['years']['last']}</p>
<p class="selection-support"><span>{president['source_document_speech_count']} source-document speeches</span><span>{actual if actual is not None else 'N/A'} actual-speaker appearances</span></p>
<p class="selection-links"><a href="{html.escape(president['profile_url'])}">Profile</a> · <a href="{html.escape(president['profile_data_url'])}">Profile V3 JSON</a></p></article>''')
    return "".join(cards)


def _selector_html(payload: Mapping[str, Any], slot: int, selected: Sequence[str]) -> str:
    value = selected[slot] if slot < len(selected) else ""
    options = []
    if slot == 2:
        options.append(f'<option value=""{" selected" if not value else ""}>None</option>')
    for president_id in payload["order"]:
        president = payload["presidents"][president_id]
        is_selected = president_id == value
        elsewhere = president_id in selected and not is_selected
        options.append(
            f'<option value="{html.escape(president_id)}"'
            f'{" selected" if is_selected else ""}{" disabled" if elsewhere else ""}>'
            f'{html.escape(president["display_name"])}'
            f'{" — already selected" if elsewhere else ""}</option>'
        )
    key = "abc"[slot]
    return f'<label for="compare-{key}">{SLOT_LABELS[slot]} · President<select id="compare-{key}" name="{key}" disabled>{"".join(options)}</select></label>'


def _exact_table(payload: Mapping[str, Any], selected: Sequence[str], layer: str) -> str:
    headers = "".join(
        f'<th scope="col">{html.escape(payload["presidents"][item]["display_name"])}</th>'
        for item in selected
    )
    rows = []
    for index, measure in enumerate(payload["measure_catalogs"][layer]):
        values = "".join(
            f'<td>{html.escape(payload["presidents"][item]["measures"][layer][index]["absolute_display"])} · {html.escape(payload["presidents"][item]["measures"][layer][index]["rank_display"])}</td>'
            for item in selected
        )
        rows.append(f'<tr><th scope="row">{html.escape(measure["label"])}</th>{values}</tr>')
    caption = "Corpus-derived native units" if layer == "corpus" else "AI-labeled native units"
    return f'<table class="exact-table" data-rhetoric-table="{layer}"><caption>{caption}; percentile is a relative position, not a quality rank.</caption><thead><tr><th scope="col">Measure</th>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _server_agenda_plot(
    cell: Mapping[str, Any], slot: int, value_field: str, scale_max: float,
) -> str:
    value = cell.get(value_field)
    position = 0.0 if value is None else min(100.0, float(value) * 10_000 / scale_max)
    state = html.escape(str(cell["state"]))
    return (
        f'<span class="agenda-plot-mark slot-{slot + 1} state-{state}" '
        f'style="--position:{position:.4f}%" aria-hidden="true">'
        f'{("●", "■", "◆")[slot]}</span>'
    )


def _server_agenda_value(
    cell: Mapping[str, Any], slot: int, value_field: str, president_name: str,
) -> str:
    thin = " · thin" if "thin" in str(cell["state"]) else ""
    state = html.escape(str(cell["state"]))
    return (
        f'<span class="agenda-mark agenda-static-value slot-{slot + 1} state-{state}">'
        f'<span class="mark-symbol">{("●", "■", "◆")[slot]}</span>'
        f'<span class="mark-president">{SLOT_LABELS[slot]} · {html.escape(president_name)}</span>'
        f'<span class="mark-value">{_fmt_percent(cell.get(value_field))}{thin}</span>'
        "</span>"
    )


def _server_agenda_row(
    label_html: str,
    cells: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    scale_max: float,
    president_names: Sequence[str],
    heading: str = "h3",
    row_class: str = "",
) -> str:
    plot = "".join(
        _server_agenda_plot(cell, slot, value_field, scale_max)
        for slot, cell in enumerate(cells)
    )
    values = "".join(
        _server_agenda_value(cell, slot, value_field, president_names[slot])
        for slot, cell in enumerate(cells)
    )
    classes = "agenda-row" + (f" {row_class}" if row_class else "")
    return (
        f'<section class="{classes}"><{heading} class="agenda-row-label">{label_html}</{heading}>'
        f'<div class="agenda-row-plot">{plot}</div>'
        f'<div class="agenda-row-values" style="--president-count:{len(cells)}">{values}</div>'
        "</section>"
    )


def _focused_broad_topics(
    index: Mapping[str, Any], selected: Sequence[str],
) -> list[tuple[Mapping[str, Any], tuple[int, ...]]]:
    """Return the governed-order union of each selected record's top broad topics."""
    topics = list(index["level1_topics"])
    broad_map = {
        (row["topic_id"], row["president_profile_id"]): row
        for row in index["broad_cells"]
    }
    reasons: dict[str, list[int]] = {}
    for slot, president_id in enumerate(selected):
        ranked = []
        for order, topic in enumerate(topics):
            cell = broad_map[(topic["topic_id"], president_id)]
            share = cell.get("speaker_paragraph_share")
            if share is not None:
                ranked.append((-float(share), order, str(topic["topic_id"])))
        for _, _, topic_id in sorted(ranked)[:FOCUSED_TOPIC_LIMIT]:
            reasons.setdefault(topic_id, []).append(slot)
    return [
        (topic, tuple(reasons[str(topic["topic_id"])]))
        for topic in topics
        if str(topic["topic_id"]) in reasons
    ]


def _focused_reason_html(
    slots: Sequence[int], selected: Sequence[str], payload: Mapping[str, Any],
) -> str:
    labels = " + ".join(
        str(payload["presidents"][selected[slot]]["short_name"])
        for slot in slots
    )
    text = f"Top 3 for {labels}"
    return f'<span class="agenda-reason">{html.escape(text)}</span>'


def _server_agenda(
    projection: compare_projection.CompareProjectionBundle,
    payload: Mapping[str, Any],
    selected: Sequence[str],
) -> str:
    index = projection.agenda_index
    broad_map = {
        (row["topic_id"], row["president_profile_id"]): row
        for row in index["broad_cells"]
    }
    focused_topics = _focused_broad_topics(index, selected)
    president_names = [
        str(payload["presidents"][president_id]["short_name"])
        for president_id in selected
    ]
    broad_rows = [
        _server_agenda_row(
            f'<span class="topic-label">{html.escape(str(topic["topic_label"]))}</span>'
            f'{_focused_reason_html(slots, selected, payload)}',
            [broad_map[(topic["topic_id"], president_id)] for president_id in selected],
            value_field="speaker_paragraph_share",
            scale_max=100,
            president_names=president_names,
            row_class="broad-row",
        )
        for topic, slots in focused_topics
    ]
    return f'<div class="agenda-chart broad-chart">{"".join(broad_rows)}</div>'


def _server_evidence(
    projection: compare_projection.CompareProjectionBundle,
    payload: Mapping[str, Any], selected: Sequence[str],
) -> str:
    records = {row["president_profile_id"]: row for row in projection.evidence_index["presidents"]}
    footprint = []
    for slot, president_id in enumerate(selected):
        record = records[president_id]["footprint"]
        thin = '<span class="support-pill thin">Thin record</span>' if record["support_state"] == "thin" else ""
        footprint.append(f'<article class="footprint-card slot-{slot + 1}"><h3>{SLOT_LABELS[slot]} · {html.escape(payload["presidents"][president_id]["display_name"])}</h3><p>{record["source_document_speech_count"]} speeches · {record["source_document_paragraph_count"]:,} paragraphs · {record["source_document_word_count"]:,} words</p>{thin}</article>')
    families = (
        ("adversarial_entities", "Adversarial entities", "Exploratory AI extraction · source-document paragraphs · raw mention counts"),
        ("presidential_invocations", "Presidential invocations", "Invocation v2 · source-document paragraphs · raw classified mentions"),
        ("distinctive_vocabulary", "Distinctive vocabulary", "Ranked terms that are statistically distinctive against all other president records"),
        ("signature_speeches", "Signature speeches", "Speech-level era-adjusted embedding alignment · cosine score"),
    )
    details = []
    for key, label, method in families:
        columns = []
        for slot, president_id in enumerate(selected):
            values = records[president_id][key]
            items = []
            for item in values:
                if key == "adversarial_entities":
                    text = f'{item["label"]} · {item["mention_count"]} mentions'
                elif key == "presidential_invocations":
                    text = f'{item["target"]} · {item["function"]} · {item["stance"]} · {item["mention_count"]}'
                elif key == "distinctive_vocabulary":
                    text = f'#{int(item["rank"]) + 1} {item["term"]}'
                else:
                    text = f'{item["title"]} ({item["year"]}) · {float(item["cosine_score"]):.3f}'
                if key == "signature_speeches":
                    items.append(f'<li><a href="{html.escape(item["source_url"])}" target="_blank" rel="noopener">{html.escape(text)}</a></li>')
                else:
                    items.append(f'<li>{html.escape(text)}</li>')
            if items:
                body = f'<ol class="evidence-list">{"".join(items)}</ol>'
            else:
                none = "None observed in the completed v2 artifact." if key == "presidential_invocations" else "None observed in the completed artifact."
                body = f'<p class="empty-state">{none}</p>'
            columns.append(f'<section class="evidence-president slot-{slot + 1}"><h3>{SLOT_LABELS[slot]} · {html.escape(payload["presidents"][president_id]["display_name"])}</h3>{body}</section>')
        details.append(f'<details class="evidence-family"><summary>{html.escape(label)}</summary><p class="method-label">{html.escape(method)}</p><div class="evidence-grid">{"".join(columns)}</div></details>')
    return f'<div class="footprint-grid">{"".join(footprint)}</div>{"".join(details)}'


def render_compare_page(
    payload: Mapping[str, Any], projection: compare_projection.CompareProjectionBundle | None = None,
) -> str:
    metrics.validate_charts(COMPARE_CHART_IDS)
    selected = list(payload["defaults"][:2])
    selectors = "".join(_selector_html(payload, slot, selected) for slot in range(3))
    corpus_table = _exact_table(payload, selected, "corpus")
    ai_table = _exact_table(payload, selected, "ai")
    if projection is None:
        broad = '<p class="load-error">Compare topic projection unavailable.</p>'
        evidence = '<p class="load-error">Compare evidence projection unavailable.</p>'
    else:
        compare_projection.validate_projection(projection)
        broad = _server_agenda(projection, payload, selected)
        evidence = _server_evidence(projection, payload, selected)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Compare presidents · Presidential Profiles</title>
<style>{PAGE_CSS}</style><link rel="stylesheet" href="assets/agenda-comparison-v1.css?v=4"><link rel="stylesheet" href="assets/compare-v3.css?v=5"></head>
<body class="compare-page"><header class="compare-hero"><p class="crumbs"><a href="index.html">← Presidential Speech Record</a> · <a href="presidents/index.html">Profiles</a> · <a href="methodology.html">Methods</a></p>
<p class="section-kicker">Compare · Profile V3 + governed topic network</p><h1>Three views of presidential speech records</h1><p class="sub">Compare rhetorical form, agendas, and traceable evidence without combining unlike populations or methods into one score.</p>
<fieldset class="compare-controls"><legend>Compare presidents</legend><div class="selector-grid">{selectors}</div><p class="interactive-note">President selectors require JavaScript. The Lincoln–Roosevelt comparison remains fully available below.</p><p id="compare-status" role="status" aria-live="polite"></p></fieldset>
<div id="compare-selection-strip" class="selection-strip">{_server_selection(payload, selected)}</div><nav class="compare-nav" aria-label="Compare sections"><ul><li><a href="#rhetoric">Rhetoric</a></li><li><a href="#agenda">Agenda</a></li><li><a href="#evidence">Evidence</a></li></ul></nav></header>
<main><section id="rhetoric"><p class="section-kicker">1 · Rhetorical fingerprints</p><h2>How do the records differ in rhetorical form?</h2><p class="section-intro">Profile V3 percentiles use source-document owners and exclude records under five speeches from ranking. Exact native values remain below each chart.</p>
<div class="tablist rhetoric-tabs" role="tablist" aria-label="Rhetorical fingerprint"><button type="button" role="tab" id="rhetoric-tab-corpus" aria-controls="rhetoric-panel-corpus" aria-selected="true" data-rhetoric-tab="corpus">Corpus-derived</button><button type="button" role="tab" id="rhetoric-tab-ai" aria-controls="rhetoric-panel-ai" aria-selected="false" tabindex="-1" data-rhetoric-tab="ai">AI-labeled</button></div>
<section class="rhetoric-panel" id="rhetoric-panel-corpus" role="tabpanel" aria-labelledby="rhetoric-tab-corpus"><div id="compare-radar-corpus" class="radar-chart"><p class="rhetoric-fallback">The chart loads when this section approaches the viewport. Exact values are available below.</p></div><details class="exact-disclosure"><summary>Exact corpus-derived values</summary><div class="exact-wrap">{corpus_table}</div></details></section>
<section class="rhetoric-panel" id="rhetoric-panel-ai" role="tabpanel" aria-labelledby="rhetoric-tab-ai" hidden><div id="compare-radar-ai" class="radar-chart"><p class="rhetoric-fallback">The chart loads when this tab is opened. Exact values are available below.</p></div><details class="exact-disclosure"><summary>Exact AI-labeled values</summary><div class="exact-wrap">{ai_table}</div></details></section></section>
<section id="agenda"><p class="section-kicker">2 · Agendas in the speech record</p><h2>Which topics are most prominent in the selected records?</h2><p class="section-intro">Shown are the broad topics that rank among the top three in at least one selected president’s speech record. Labels show which presidents bring each topic into view; select a value for its exact counts and one example.</p>
<div id="compare-agenda-stage"><div id="agenda-runtime-status" class="agenda-runtime-status" role="status" aria-live="polite"></div><div id="agenda-broad">{broad}</div>
<noscript><style>#rhetoric-panel-ai[hidden]{{display:block!important}}.rhetoric-tabs{{display:none}}</style><p class="method-label">JavaScript is off. The focused Lincoln–Roosevelt topic comparison, both rhetoric tables, Profile links, and the full topic CSV remain available; interactive selection and topic examples are unavailable.</p></noscript></div></section>
<section id="evidence"><p class="section-kicker">3 · Evidence from the record</p><h2>What supports this comparison?</h2><p class="section-intro">Corpus footprint stays visible. Each disclosure names its population, grain, and method before presenting comparison-wide evidence.</p><div id="compare-evidence-content">{evidence}</div>
<div class="download-panel"><button type="button" id="compare-download">Download selected Profile V3 records</button><a href="data/compare/{compare_projection.AI_VALUES_FILE}">Full actual-speaker topic CSV</a></div></section></main>
<footer><p>Data: Miller Center presidential speeches. AI topics are descriptive and non-causal.</p></footer>
<script type="application/json" id="compare-page-data">{json_for_script(payload)}</script><script defer src="assets/agenda-comparison-v1.js?v=4"></script><script defer src="assets/compare-v3.js?v=5"></script></body></html>'''


def write_renderer_assets(site_dir: Path) -> None:
    asset_dir = site_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in ASSET_FILES.items():
        (asset_dir / filename).write_text(content, encoding="utf-8")


def write_compare(
    data: dict, display_issues: list[str], *, profile_views: dict[str, dict],
    projection: compare_projection.CompareProjectionBundle,
) -> dict:
    """Require the validated Compare projection and write the V3 page/assets."""
    compare_projection.validate_projection(projection)
    payload = build_payload(data, display_issues, profile_views=profile_views, projection=projection)
    site_dir = compare_projection.REPO_ROOT / "docs"
    write_renderer_assets(site_dir)
    output = site_dir / "compare.html"
    output.write_text(render_compare_page(payload, projection), encoding="utf-8")
    return payload
