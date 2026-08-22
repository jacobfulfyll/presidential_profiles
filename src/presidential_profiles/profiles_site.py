"""Render evidence-first president profiles and their public v3 payloads."""

from __future__ import annotations

import html
import json
import math

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from . import metrics, topic_quality
from .figures import BLUE_RAMP, GRID, INK, INK2, PARTY_COLORS, SURFACE
from .html_safety import json_for_script
from .profiles import (
    FEATURE_SIMILARITY,
    MIN_ISSUE_PARAS,
    RADAR_AXES,
    RADAR_VALUE_COLUMNS,
    SPARSE_MIN_SPEECHES,
    format_percentile,
    format_speech_count,
    miller_speech_url,
    president_chronology,
    public_display_name,
    slug,
)
from .site_style import FONT, PAGE_CSS


DISCOVERED_LABELS = topic_quality.discovered_labels()
PROFILE_SCHEMA_VERSION = "president-profile-v3"
PROFILE_PUBLIC_KEYS = (
    "schema_version",
    "president",
    "slug",
    "party",
    "years",
    "sample",
    "rhetorical_radar",
    "raw_stats",
    "legacy_issue_attention",
    "issue_evidence",
    "ai",
    "distinctive_vocabulary",
    "signature_speeches",
    "legacy_invocations",
    "classified_invocations",
    "voice_neighbors",
    "agenda_neighbors",
    "feature_neighbors",
    "context_specific",
)

# The public v3 contract intentionally leaves units implicit. These ordered
# renderer specifications make those units explicit without changing schema.
LEGACY_MEASURE_SPECS = (
    {"key": "hope", "label": "Hope", "short": "Hope", "unit": "mentions per 10,000 words", "scale": 1, "digits": 1},
    {"key": "fear", "label": "Fear appeal", "short": "Fear appeal", "unit": "mentions per 10,000 words", "scale": 1, "digits": 1},
    {"key": "certainty", "label": "Certainty", "short": "Certainty", "unit": "percent of assertive and deliberative markers", "scale": 100, "digits": 1},
    {"key": "us_vs_them", "label": "Us vs them", "short": "Us vs them", "unit": "mentions per 10,000 words", "scale": 1, "digits": 1},
    {"key": "self_reference", "label": "Self-reference", "short": "Self-reference", "unit": "percent I among I + we", "scale": 100, "digits": 1},
    {"key": "formality", "label": "Formality", "short": "Formality", "unit": "Flesch–Kincaid grade level", "scale": 1, "digits": 1},
    {"key": "vocabulary", "label": "Vocabulary", "short": "Vocabulary", "unit": "windowed type-token ratio (percent)", "scale": 100, "digits": 1},
    {"key": "religiosity", "label": "Religiosity", "short": "Religiosity", "unit": "mentions per 10,000 words", "scale": 1, "digits": 1},
)

AI_MEASURE_SPECS = (
    {"key": "party_attack", "label": "Partisan attack", "short": "Partisan attack", "unit": "percent of paragraphs", "scale": 1, "digits": 2},
    {"key": "enemy_naming", "label": "Enemy naming", "short": "Enemy naming", "unit": "percent of paragraphs", "scale": 1, "digits": 2},
    {"key": "zero_sum", "label": "Zero-sum framing", "short": "Zero-sum", "unit": "percent of paragraphs", "scale": 1, "digits": 2},
    {"key": "proposal", "label": "Proposal share", "short": "Proposal share", "unit": "percent of paragraphs", "scale": 1, "digits": 2},
    {"key": "values", "label": "Values share", "short": "Values share", "unit": "percent of paragraphs", "scale": 1, "digits": 2},
    {"key": "topic_breadth", "label": "Effective topic breadth", "short": "Topic breadth", "unit": "effective topics", "scale": 1, "digits": 2},
)

PROFILE_MUTED = "#706d67"


def _json_value(value):
    """Convert pandas/numpy values into stable JSON-native values.

    Public JSON has no representation for non-finite derived values, so they
    become explicit nulls. Required chart absolutes are validated separately.
    """
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _required_finite(value, field: str) -> float:
    """Reject a non-finite value for a measure that must remain observable."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Required profile measure {field} is non-finite")
    return number


def _normalized_ai(
    ai: dict | None,
    thin_record: bool,
    n_speeches: int,
) -> dict | None:
    if ai is None:
        return None
    for spec in AI_MEASURE_SPECS:
        raw_measure = ai.get("ai_radar", {}).get(spec["key"], {})
        _required_finite(
            raw_measure.get("absolute"),
            f"ai_radar.{spec['key']}.absolute",
        )
    normalized = _json_value(ai)
    if "low_confidence" in normalized and bool(normalized["low_confidence"]) != thin_record:
        raise ValueError("AI thin-record status disagrees with the profile sample")
    if int(normalized.get("n_speeches", n_speeches)) != n_speeches:
        raise ValueError("AI speech count disagrees with the profile sample")
    normalized["low_confidence"] = thin_record
    ai_radar = normalized.get("ai_radar", {})
    expected_radar = tuple(spec["key"] for spec in AI_MEASURE_SPECS)
    if set(ai_radar) != set(expected_radar):
        raise ValueError("AI radar keys disagree with the declared measure order")
    normalized["ai_radar"] = {
        key: ai_radar[key] for key in expected_radar
    }
    if thin_record:
        for measure in normalized.get("ai_radar", {}).values():
            measure["percentile"] = None
    return normalized


def profile_view_model(
    president: str,
    data: dict,
    display_issues: list[str],
) -> dict:
    """Build the normalized model shared by HTML, figures, JSON, and Compare."""
    scores = data["scores"].loc[president]
    issue_row = data["issues"].loc[president]
    n_speeches = int(scores["n_speeches"])
    thin_record = n_speeches < SPARSE_MIN_SPEECHES

    raw_stats = scores.to_dict()
    if thin_record:
        for key in tuple(raw_stats):
            if key.startswith("pct_"):
                raw_stats[key] = None
    raw_stats = _json_value(raw_stats)

    rhetorical_radar = []
    for key, label in RADAR_AXES:
        percentile = (
            None
            if thin_record
            else _json_value(scores[f"pct_{key}"])
        )
        rhetorical_radar.append({
            "key": key,
            "label": label,
            "absolute": _required_finite(
                scores[RADAR_VALUE_COLUMNS[key]],
                f"rhetorical_radar.{key}.absolute",
            ),
            "percentile": percentile,
        })

    issue_attention = [
        {
            "key": issue,
            "label": DISCOVERED_LABELS.get(issue, issue),
            "share": round(
                _required_finite(
                    issue_row[f"share_{issue}"], f"legacy_issue.{issue}.share"
                ),
                4,
            ),
            "era_relative_difference": round(
                _required_finite(
                    issue_row[f"rel_{issue}"], f"legacy_issue.{issue}.relative"
                ),
                4,
            ),
        }
        for issue in display_issues
    ]

    issue_evidence = _json_value(data["issue_cards"].get(president, {}))
    if issue_evidence:
        if (
            "low_confidence" in issue_evidence
            and bool(issue_evidence["low_confidence"]) != thin_record
        ):
            raise ValueError("Issue-evidence thin status disagrees with the profile sample")
        if int(issue_evidence.get("n_speeches", n_speeches)) != n_speeches:
            raise ValueError("Issue-evidence speech count disagrees with the profile sample")
        expected_paragraphs = int(issue_row.get("n_paragraphs", 0))
        if int(issue_evidence.get("n_paragraphs", expected_paragraphs)) != expected_paragraphs:
            raise ValueError("Issue-evidence paragraph count disagrees with the issue table")
        issue_evidence["low_confidence"] = thin_record

    ai = _normalized_ai(
        data.get("ai", {}).get("by_president", {}).get(president),
        thin_record,
        n_speeches,
    )
    vocabulary = (
        data["distinctive"][data["distinctive"].president.eq(president)]
        .sort_values("rank")
        .head(20)
        .to_dict("records")
    )
    signatures = [
        {
            **_json_value(item),
            "url": miller_speech_url(item.get("url", "")),
        }
        for item in data["signatures"].get(president, [])
    ]

    view = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "president": president,
        "slug": slug(president),
        "party": _json_value(scores["party"]),
        "years": {
            "first": int(scores["first_year"]),
            "last": int(scores["last_year"]),
        },
        "sample": {
            "n_speeches": n_speeches,
            "n_words": int(scores["n_words"]),
            "thin_record": thin_record,
            "warning": (
                "Fewer than five corpus speeches: visible, but not ranked as a precise outlier."
                if thin_record
                else None
            ),
        },
        "rhetorical_radar": rhetorical_radar,
        "raw_stats": raw_stats,
        "legacy_issue_attention": issue_attention,
        "issue_evidence": issue_evidence,
        "ai": ai,
        "distinctive_vocabulary": _json_value(vocabulary),
        "signature_speeches": signatures,
        "legacy_invocations": {
            "invokes": _json_value(data["invokes"].get(president, [])),
            "invoked_by": _json_value(data["invoked_by"].get(president)),
        },
        "classified_invocations": _json_value(
            data.get("invocation_v2", {}).get(president, [])
        ),
        "voice_neighbors": _json_value(data["voice_neighbors"].get(president, [])),
        "agenda_neighbors": _json_value(data["agenda_neighbors"].get(president, [])),
        "feature_neighbors": _json_value(
            data.get("feature_neighbors", {}).get(president, {})
        ),
        "context_specific": {
            "profile_only": [],
            "compare_only": [],
        },
    }

    n_paragraphs = int(
        (ai or {}).get(
            "n_paragraphs",
            issue_evidence.get("n_paragraphs", issue_row.get("n_paragraphs", 0)),
        )
    )
    leading_ai = (ai or {}).get("top_topics", [None])[0] if (ai or {}).get("top_topics") else None
    leading_legacy = max(issue_attention, key=lambda item: item["share"], default=None)
    view["_view"] = {
        "display_name": public_display_name(president),
        "speech_count_label": format_speech_count(n_speeches),
        "record_span_label": f"{view['years']['first']}–{view['years']['last']}",
        "n_paragraphs": n_paragraphs,
        "support_label": (
            "Insufficient record for percentile ranking"
            if thin_record
            else "Supported for percentile ranking"
        ),
        "leading_ai_topic": leading_ai,
        "leading_legacy_issue": leading_legacy,
        "topic_count": len((ai or {}).get("top_topics", [])[:6]),
        "previous": None,
        "next": None,
        "compare_href": f"../compare.html?a={slug(president)}",
        "units": {
            "rhetorical_radar": {
                spec["key"]: spec["unit"] for spec in LEGACY_MEASURE_SPECS
            },
            "ai_radar": {
                spec["key"]: spec["unit"] for spec in AI_MEASURE_SPECS
            },
            "legacy_issue_share": "fraction",
            "legacy_issue_era_relative_difference": "percentage points",
            "ai_topic_share": "percent",
            "ai_topic_era_relative_difference": "percentage points",
            "similarity": "cosine score",
        },
    }
    return view


def build_profile_view_models(
    data: dict,
    display_issues: list[str],
    *,
    require_complete: bool = False,
) -> dict[str, dict]:
    """Build and connect every profile model exactly once."""
    order = president_chronology(
        data["scores"].index,
        require_complete=require_complete,
    )
    views = {
        president: profile_view_model(president, data, display_issues)
        for president in order
    }
    for index, president in enumerate(order):
        nav = views[president]["_view"]
        if index:
            previous = order[index - 1]
            nav["previous"] = {
                "president": previous,
                "display_name": public_display_name(previous),
                "slug": slug(previous),
            }
        if index + 1 < len(order):
            following = order[index + 1]
            nav["next"] = {
                "president": following,
                "display_name": public_display_name(following),
                "slug": slug(following),
            }
        comparison = nav["next"] or nav["previous"]
        if comparison:
            nav["compare_href"] = (
                f"../compare.html?a={slug(president)}&b={comparison['slug']}"
            )
    return views


def profile_public_payload(view: dict) -> dict:
    """Project one normalized model onto the exact public v3 contract."""
    validate_profile_view_model(view)
    return {
        key: _json_value(view[key])
        for key in PROFILE_PUBLIC_KEYS
    }


def validate_profile_view_model(view: dict) -> None:
    """Fail closed when a shared profile model drifts from its v3 identity."""
    missing = [key for key in PROFILE_PUBLIC_KEYS if key not in view]
    if missing:
        raise ValueError(f"Profile view is missing public keys: {missing}")
    if view["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError("Profile view schema must remain president-profile-v3")
    if view["slug"] != slug(view["president"]):
        raise ValueError("Profile view slug disagrees with its president key")
    thin_record = bool(view["sample"]["thin_record"])
    if thin_record != (int(view["sample"]["n_speeches"]) < SPARSE_MIN_SPEECHES):
        raise ValueError("Profile sample thin status disagrees with its speech count")
    if [item["key"] for item in view["rhetorical_radar"]] != [
        key for key, _ in RADAR_AXES
    ]:
        raise ValueError("Rhetorical measures disagree with the declared order")
    if thin_record and any(
        item["percentile"] is not None for item in view["rhetorical_radar"]
    ):
        raise ValueError("Thin profile contains a rhetorical percentile")
    ai = view.get("ai")
    if ai:
        if bool(ai["low_confidence"]) != thin_record:
            raise ValueError("AI thin status disagrees with the profile sample")
        if int(ai["n_speeches"]) != int(view["sample"]["n_speeches"]):
            raise ValueError("AI speech count disagrees with the profile sample")
        expected_ai = [spec["key"] for spec in AI_MEASURE_SPECS]
        if list(ai["ai_radar"]) != expected_ai:
            raise ValueError("AI measures disagree with the declared order")
    evidence = view.get("issue_evidence") or {}
    if evidence and bool(evidence.get("low_confidence")) != thin_record:
        raise ValueError("Issue-evidence thin status disagrees with the profile sample")


def public_profile_payload(
    president: str,
    data: dict,
    display_issues: list[str],
) -> dict:
    """Compatibility wrapper for the canonical public profile producer."""
    return profile_public_payload(
        profile_view_model(president, data, display_issues)
    )


def _absolute_display(spec: dict, value, *, compact: bool = False) -> str:
    if value is None:
        return "N/A"
    scaled = float(value) * float(spec["scale"])
    digits = int(spec["digits"])
    number = f"{scaled:,.{digits}f}"
    unit = spec["unit"]
    if "percent" in unit:
        return f"{number}%" if compact else f"{number}% · {unit}"
    if unit == "mentions per 10,000 words":
        return f"{number} / 10k" if compact else f"{number} per 10,000 words"
    if unit == "Flesch–Kincaid grade level":
        return f"grade {number}" if compact else f"{number} grade level"
    if unit == "effective topics":
        return f"{number} topics" if compact else f"{number} effective topics"
    return f"{number} {unit}"


def profile_measure_rows(view: dict, layer: str) -> list[dict]:
    """Return ordered, labeled measures used by figures and table fallbacks."""
    if layer == "rhetorical_radar":
        specs = LEGACY_MEASURE_SPECS
        source = {item["key"]: item for item in view["rhetorical_radar"]}
    elif layer == "ai_radar":
        specs = AI_MEASURE_SPECS
        source = (view.get("ai") or {}).get("ai_radar", {})
    else:
        raise KeyError(layer)

    thin_record = bool(view["sample"]["thin_record"])
    rows = []
    for spec in specs:
        item = source.get(spec["key"], {})
        absolute = item.get("absolute")
        percentile = item.get("percentile")
        unavailable = (
            "Insufficient record" if thin_record else "Not ranked"
        )
        rows.append({
            "key": spec["key"],
            "label": spec["label"],
            "short": spec["short"],
            "unit": spec["unit"],
            "absolute": absolute,
            "absolute_display": _absolute_display(spec, absolute),
            "absolute_compact": _absolute_display(spec, absolute, compact=True),
            "percentile": percentile,
            "percentile_display": format_percentile(
                percentile,
                unavailable=unavailable,
            ),
        })
    return rows


def _percentile_bars(rows: list[dict], title: str) -> go.Figure | None:
    ranked = [row for row in rows if row["percentile"] is not None]
    if not ranked:
        return None
    ranked = list(reversed(ranked))
    fig = go.Figure(go.Bar(
        x=[row["percentile"] for row in ranked],
        y=[row["short"] for row in ranked],
        orientation="h",
        marker={"color": BLUE_RAMP[4]},
        text=[
            f"{row['percentile_display']} · {row['absolute_compact']}"
            for row in ranked
        ],
        textposition="outside",
        cliponaxis=False,
        customdata=[
            [row["absolute_display"], row["percentile_display"]]
            for row in ranked
        ],
        hovertemplate=(
            "%{y}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title={"text": title, "font": {"size": 13, "color": INK}, "x": 0},
        template="simple_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT, "color": INK2, "size": 11},
        showlegend=False,
        height=max(300, 54 + 40 * len(ranked)),
        margin={"l": 116, "r": 190, "t": 42, "b": 48},
        xaxis={
            "range": [0, 100],
            "tickvals": [0, 25, 50, 75, 100],
            "ticksuffix": "th",
            "gridcolor": GRID,
            "title": "Eligible-president percentile",
            "tickfont": {"color": PROFILE_MUTED},
        },
        yaxis={"tickfont": {"color": INK, "size": 11}, "automargin": True},
    )
    return fig


def profile_figure_payload(view: dict) -> dict:
    """Generate every retained profile figure from the normalized model."""
    figures = {}
    legacy = _percentile_bars(
        profile_measure_rows(view, "rhetorical_radar"),
        "Corpus-derived rhetorical measures",
    )
    if legacy is not None:
        figures["rhetoric_legacy"] = legacy
    if view.get("ai"):
        ai_figure = _percentile_bars(
            profile_measure_rows(view, "ai_radar"),
            "AI-labeled paragraph measures",
        )
        if ai_figure is not None:
            figures["rhetoric_ai"] = ai_figure
    return {
        key: _json_value(json.loads(pio.to_json(figure)))
        for key, figure in figures.items()
    }


def _chart_controls(metric_name: str, evidence_href: str, evidence_label: str) -> str:
    return (
        metrics.lesson_html(metric_name)
        + '<details class="evidence"><summary>Inspect the evidence</summary>'
        + f'<p><a href="{html.escape(evidence_href, quote=True)}" download>'
        + f'{html.escape(evidence_label)} →</a></p></details>'
    )


def _topic_rows_html(view: dict) -> str:
    topics = (view.get("ai") or {}).get("top_topics", [])[:6]
    if not topics:
        return '<p class="unavailable">AI topic rows are unavailable.</p>'
    return '<div class="topic-rows">' + "".join(
        f'''<article class="topic-row" data-topic-row>
  <div class="topic-label"><strong>{html.escape(topic["name"])}</strong>
    <span>{html.escape(topic.get("domain", ""))} · {int(topic.get("n", 0)):,} labeled paragraphs</span></div>
  <data value="{float(topic["share"]):.2f}">{float(topic["share"]):.1f}% <small>share</small></data>
  <data class="topic-rel" value="{float(topic["rel"]):.2f}">{float(topic["rel"]):+.1f} pp <small>vs era</small></data>
</article>'''
        for topic in topics
    ) + "</div>"


def _ai_summary_grid(info: dict | None) -> str:
    if not info:
        return '<p class="unavailable">AI paragraph summaries are unavailable.</p>'
    flags = (
        ("Partisan attack", info["flags"]["party_attack"]),
        ("Named adversary", info["flags"]["enemy_naming"]),
        ("Zero-sum framing", info["flags"]["zero_sum"]),
    )
    flag_rows = "".join(
        f'''<li><span>{html.escape(label)}</span><strong>{float(value):.1f}%</strong>
  <i aria-hidden="true"><b style="width:{max(0, min(float(value), 100)):.2f}%"></b></i></li>'''
        for label, value in flags
    )
    proposal_values = info["proposal_values"]
    segments = "".join(
        f'<i class="pv-{key}" style="width:{max(0, min(float(proposal_values[key]), 100)):.3f}%"></i>'
        for key in ("proposal", "mixed", "values", "neither")
    )
    proposal_legend = "".join(
        f'<li><span class="pv-dot pv-{key}"></span>{html.escape(key.replace("_", " ").title())}'
        f'<strong>{float(proposal_values[key]):.1f}%</strong></li>'
        for key in ("proposal", "mixed", "values", "neither")
    )
    genres = "".join(
        f'<li><span>{html.escape(item["name"])}</span><strong>{float(item["value"]):.1f}%</strong></li>'
        for item in info.get("speech_types", [])
    ) or '<li><span>Unavailable</span></li>'
    adversaries = "".join(
        f'<li><span>{html.escape(item["name"])}</span><strong>{int(item["n"]):,} mentions</strong></li>'
        for item in info.get("adversaries", [])
    ) or '<li><span>No repeated adversarial entity in this record.</span></li>'
    pv_label = ", ".join(
        f'{key.replace("_", " ")} {float(proposal_values[key]):.1f}%'
        for key in ("proposal", "mixed", "values", "neither")
    )
    return f'''<div class="agenda-grid">
<article><span class="source-tag ai-source">AI-labeled</span><h3>Combat framing</h3>
  <p>Share of paragraphs carrying each closed-rubric flag.</p><ul class="meter-list">{flag_rows}</ul></article>
<article><span class="source-tag ai-source">AI-labeled</span><h3>Policy and values</h3>
  <p>One mutually exclusive paragraph class.</p>
  <div class="pv-bar" role="img" aria-label="{html.escape(pv_label, quote=True)}">{segments}</div>
  <ul class="compact-list pv-key">{proposal_legend}</ul></article>
<article><span class="source-tag ai-source">AI-labeled</span><h3>Speech genres</h3>
  <p>Primary assigned types among corpus speeches.</p><ul class="compact-list">{genres}</ul></article>
<article><span class="source-tag ai-source">AI-labeled</span><h3>Named adversaries</h3>
  <p>Most frequent entities assigned adversarial stance.</p><ul class="compact-list">{adversaries}</ul></article>
</div>'''


def _measure_table(rows: list[dict], caption: str) -> str:
    body = "".join(
        f'''<tr><th scope="row">{html.escape(row["label"])}</th>
  <td>{html.escape(row["absolute_display"])}</td>
  <td class="{'not-ranked' if row['percentile'] is None else ''}">{html.escape('N/A · ' + row['percentile_display'] if row['percentile'] is None else row['percentile_display'])}</td></tr>'''
        for row in rows
    )
    return f'''<div class="measure-table-wrap"><table class="measure-table">
  <caption>{html.escape(caption)}</caption>
  <thead><tr><th scope="col">Measure</th><th scope="col">Absolute value</th><th scope="col">Eligible-president rank</th></tr></thead>
  <tbody>{body}</tbody>
</table></div>'''


def _measure_summary(rows: list[dict], method_summary: str) -> str:
    """Add a deterministic high/low reading to a figure description."""
    ranked = [row for row in rows if row["percentile"] is not None]
    if not ranked:
        return (
            method_summary
            + " No percentile ranks are available for this thin record; "
            "absolute values remain in the structured table."
        )
    highest = max(ranked, key=lambda row: float(row["percentile"]))
    lowest = min(ranked, key=lambda row: float(row["percentile"]))
    return (
        method_summary
        + f" Highest relative position: {highest['label']}, "
        f"{highest['percentile_display']}. Lowest: {lowest['label']}, "
        f"{lowest['percentile_display']}."
    )


def _figure_block(
    view: dict,
    figure_payload: dict,
    *,
    key: str,
    title: str,
    summary: str,
    rows: list[dict],
) -> str:
    title_id = f"{key}-title"
    summary_id = f"{key}-summary"
    if key in figure_payload:
        visual = f'''<div class="profile-figure">
  <div class="chart" data-fig="{key}" role="img" aria-labelledby="{title_id}" aria-describedby="{summary_id}"></div>
</div>
{_chart_controls("president_percentile", f"../data/presidents/{view['slug']}.json", "Download absolute measures and eligible-president percentiles")}'''
    else:
        visual = (
            '<p class="rank-unavailable"><strong>N/A · Insufficient record.</strong> '
            'The absolute measures remain below, but a percentile graphic is not rendered.</p>'
        )
    return f'''<article class="rhetoric-card">
  <span class="source-tag {'ai-source' if key == 'rhetoric_ai' else 'corpus-source'}">{'AI-labeled' if key == 'rhetoric_ai' else 'Corpus-derived'}</span>
  <h3 id="{title_id}">{html.escape(title)}</h3>
  <p class="chart-summary" id="{summary_id}">{html.escape(summary)}</p>
  {visual}
  {_measure_table(rows, title + ": structured data alternative")}
</article>'''


def _issue_card_html(card: dict) -> str:
    issue = html.escape(DISCOVERED_LABELS.get(card["issue"], card["issue"]))
    words = "".join(
        f'<span class="term">{html.escape(str(word))}</span>'
        for word in card.get("words", [])
    )
    words_html = f'<div class="terms">{words}</div>' if words else ""
    quote_html = ""
    if card.get("quote"):
        # Quotes are escaped at the keyed evidence producer so their exact
        # citation pairing can be preserved here without double escaping.
        quote_html = (
            f'<blockquote><p>“{card["quote"]}”</p>'
            f'<cite>{html.escape(str(card.get("cite", "")))}</cite></blockquote>'
        )
    stance = (
        f'<span class="evidence-badge">{html.escape(str(card["stance"]))}</span>'
        if card.get("stance")
        else ""
    )
    topic_flag = (
        '<span class="evidence-badge">Topic of the day</span>'
        if card.get("topic_of_day")
        else ""
    )
    return f'''<article class="evidence-card i-card">
  <div class="evidence-head"><h3>{issue}</h3>
    <span><strong>{float(card["share"]) * 100:.1f}%</strong> of paragraphs</span>
    <span>{float(card["rel"]):+.1f} pp vs era</span>{topic_flag}{stance}</div>
  {words_html}{quote_html}
</article>'''


def _evidence_cards_from_info(info: dict) -> str:
    cards = list(info.get("cards", []))
    if cards:
        # The producer is strength-ordered. Put excerpt-bearing cards first
        # while preserving that strength order within the excerpt and
        # rate-only groups.
        ordered = (
            [card for card in cards if card.get("quote")]
            + [card for card in cards if not card.get("quote")]
        )
        leading = "".join(_issue_card_html(card) for card in ordered[:2])
        remainder = ""
        if len(ordered) > 2:
            remainder = f'''<details class="more-evidence"><summary>Additional issue evidence ({len(ordered) - 2})</summary>
  <div>{''.join(_issue_card_html(card) for card in ordered[2:])}</div></details>'''
        cards_html = leading + remainder
    else:
        cards_html = '<p class="unavailable">No legacy issue passed the declared profile-card thresholds.</p>'
    voice = info.get("voice", [])
    if voice:
        terms = "".join(
            f'<span class="term">{html.escape(str(word))}</span>' for word in voice
        )
        cards_html += f'''<details class="more-evidence"><summary>Distinctive vocabulary outside one issue</summary>
  <p>Corpus-derived words not assigned to a displayed issue card.</p><div class="terms">{terms}</div></details>'''
    return cards_html


def _issue_cards_html(president: str, data: dict) -> str:
    """Compatibility helper used by the issue-card security tests."""
    return _evidence_cards_from_info(
        data["issue_cards"].get(president, {"cards": [], "voice": []})
    )


def _invocation_details(view: dict) -> str:
    legacy = view["legacy_invocations"]
    invokes = legacy.get("invokes") or []
    invoked_by = legacy.get("invoked_by") or {}
    lines = []
    if invokes:
        lines.append(
            "Invokes: "
            + ", ".join(
                f'{html.escape(str(item["target"]))} ({int(item["n"]):,}×)'
                for item in invokes[:4]
            )
        )
    if invoked_by.get("total"):
        lines.append(
            f'Invoked {int(invoked_by["total"]):,}× by later presidents'
        )
    classified = view.get("classified_invocations", [])[:12]
    table = ""
    if classified:
        rows = "".join(
            f'<tr><td>{html.escape(str(item["target"]))}</td><td>{html.escape(str(item["function"]))}</td>'
            f'<td>{html.escape(str(item["stance"]))}</td><td>{int(item["mentions"]):,}</td></tr>'
            for item in classified
        )
        cards = "".join(
            f'''<div><dt>Target</dt><dd>{html.escape(str(item["target"]))}</dd>
  <dt>Function</dt><dd>{html.escape(str(item["function"]))}</dd>
  <dt>Stance</dt><dd>{html.escape(str(item["stance"]))}</dd>
  <dt>Mentions</dt><dd>{int(item["mentions"]):,}</dd></div>'''
            for item in classified
        )
        table = f'''<table class="invocation-table"><caption>AI-classified presidential invocations</caption>
  <thead><tr><th scope="col">Target</th><th scope="col">Function</th><th scope="col">Stance</th><th scope="col">Mentions</th></tr></thead><tbody>{rows}</tbody></table>
  <dl class="invocation-cards" aria-label="AI-classified presidential invocations">{cards}</dl>
  <p><a href="../data/networks/invocation_evidence.csv" download>Download invocation evidence →</a></p>'''
    if not lines and not table:
        return ""
    summary = " · ".join(lines) if lines else "No legacy invocation summary."
    return f'''<details class="more-evidence"><summary>Presidential invocation evidence</summary>
  <p>{summary}</p>{table}</details>'''


def _similarity_cards(view: dict) -> str:
    categories = view.get("feature_neighbors", {})
    cards = []
    for key, meta in FEATURE_SIMILARITY.items():
        neighbors = categories.get(key, [])[:3]
        if neighbors:
            rows = "".join(
                f'''<li><a href="{slug(item["president"])}.html">{html.escape(public_display_name(item["president"]))}</a>
  <data value="{float(item["similarity"]):.4f}">{float(item["similarity"]):.2f} cosine</data></li>'''
                for item in neighbors
            )
        else:
            rows = '<li class="unavailable">No adequately sampled match.</li>'
        cards.append(f'''<article class="similarity-card" data-similarity="{key}">
  <h3>{html.escape(meta["label"])}</h3><p>{html.escape(meta["description"])}</p>
  <ol>{rows}</ol></article>''')
    return '<div class="similarity-grid">' + "".join(cards) + "</div>"


def _signature_list(view: dict) -> str:
    speeches = view.get("signature_speeches", [])
    if not speeches:
        return '<p class="unavailable">No signature speech was selected.</p>'
    return '<ol class="signature-list">' + "".join(
        f'''<li><span>{int(item.get("year", 0))}</span><a href="{html.escape(item["url"], quote=True)}" target="_blank" rel="noopener">{html.escape(str(item["title"]))}</a></li>'''
        for item in speeches
    ) + "</ol>"


def _president_navigation(view: dict) -> str:
    nav = view["_view"]
    links = []
    if nav.get("previous"):
        previous = nav["previous"]
        links.append(
            f'<a class="previous" href="{previous["slug"]}.html"><span>← Previous president</span>'
            f'<strong>{html.escape(previous["display_name"])}</strong></a>'
        )
    if nav.get("next"):
        following = nav["next"]
        links.append(
            f'<a class="next" href="{following["slug"]}.html"><span>Next president →</span>'
            f'<strong>{html.escape(following["display_name"])}</strong></a>'
        )
    return '<nav class="president-nav" aria-label="Adjacent president profiles">' + "".join(links) + "</nav>"


PROFILE_CSS = r"""
  body.profile-page { overflow-x: clip; }
  .profile-page :where(header, main, section, article, nav, div) { min-width: 0; }
  .profile-hero { padding-top: 28px; }
  .hero-tools { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 22px; }
  .back-link { color: var(--ink2); font-size: .88rem; }
  .compare-action { display: inline-flex; align-items: center; min-height: 42px; padding: 8px 15px; border: 1px solid #5d350c; border-radius: 8px; background: #fffaf4; color: #5d350c; font-weight: 720; text-decoration: none; }
  .hero-body { display: grid; grid-template-columns: 118px minmax(0,1fr); gap: 24px; align-items: center; }
  .portrait { grid-row: 1 / span 2; width: 118px; height: 118px; border: 1px solid var(--border); border-radius: 50%; background: var(--surface); object-fit: cover; }
  .eyebrow { color: #6a421b; font-size: .72rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
  .profile-hero h1 { margin-top: 4px; font: 700 clamp(2rem,5vw,3.1rem)/1.04 Georgia,serif; letter-spacing: -.025em; }
  .identity-meta { grid-column: 2; display: flex; flex-wrap: wrap; gap: 7px; margin-top: -5px; }
  .identity-meta span { display: inline-flex; align-items: center; min-height: 31px; padding: 4px 10px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface); color: var(--ink2); font-size: .82rem; }
  .identity-meta .party::before { width: 9px; height: 9px; margin-right: 7px; border-radius: 50%; background: var(--party-color); content: ""; }
  .identity-meta .support { border-color: #55705f; background: #f1f7f2; color: #294c36; font-weight: 720; }
  .identity-meta .support.thin { border-color: #9b7220; background: #fff7df; color: #614b19; }
  .record-note { max-width: none; margin: 18px auto 0; padding: 15px 18px; border: 1px solid #c59a43; border-left: 5px solid #8a5d00; border-radius: 10px; background: #fff7df; color: #4e3b13; }
  .record-note strong { display: block; margin-bottom: 3px; }
  .profile-nav { position: sticky; z-index: 8; top: calc(var(--global-nav-height, 0px) + 6px); max-width: 940px; margin: 18px auto 0; padding: 0 20px; }
  .profile-nav ul { display: grid; grid-template-columns: repeat(5,minmax(0,1fr)); padding: 5px; border: 1px solid var(--border); border-radius: 12px; background: rgba(252,252,251,.96); box-shadow: 0 4px 16px rgba(11,11,11,.06); list-style: none; }
  .profile-nav a { display: grid; min-height: 38px; place-items: center; padding: 5px 4px; border-radius: 8px; color: var(--ink2); font-size: .8rem; font-weight: 700; text-align: center; text-decoration: none; }
  .profile-nav a:hover { background: var(--page); color: var(--ink); }
  .profile-page main { padding-top: 2px; }
  .profile-page section { scroll-margin-top: calc(var(--global-nav-height, 0px) + 72px); }
  .section-kicker { margin-bottom: 5px; color: #6a421b; font-size: .68rem; font-weight: 820; letter-spacing: .09em; text-transform: uppercase; }
  .profile-page section > h2 { font: 700 clamp(1.45rem,3vw,1.85rem)/1.15 Georgia,serif; }
  .profile-page section > p { max-width: 52rem; }
  .glance-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(145px,1fr)); gap: 10px; margin-top: 18px; }
  .glance-card { padding: 13px 14px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
  .glance-card span, .glance-card strong { display: block; }
  .glance-card span { color: var(--muted); font-size: .72rem; font-weight: 740; letter-spacing: .025em; }
  .glance-card strong { margin-top: 3px; font-size: .96rem; line-height: 1.3; }
  .glance-card small { display: block; margin-top: 2px; color: var(--muted); font-size: .72rem; }
  .source-key { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 10px; margin: 18px 0 0; padding: 0; list-style: none; }
  .source-key li { padding: 11px 13px; border: 1px solid var(--border); border-radius: 9px; color: var(--ink2); font-size: .78rem; }
  .source-key strong { display: block; margin-bottom: 2px; color: var(--ink); }
  .source-tag { display: inline-flex; margin-bottom: 7px; padding: 2px 7px; border: 1px solid; border-radius: 999px; font-size: .67rem; font-weight: 800; letter-spacing: .045em; text-transform: uppercase; }
  .corpus-source { border-color: #68736b; background: #f1f4f1; color: #34463a; }
  .ai-source { border-color: #6384a0; background: #eef6ff; color: #254e70; }
  .legacy-source { border-color: #9a815d; background: #f8f2e8; color: #664819; }
  .topic-rows { display: grid; gap: 7px; margin-top: 18px; }
  .topic-row { display: grid; grid-template-columns: minmax(0,1fr) minmax(90px,auto) minmax(120px,auto); align-items: center; gap: 12px; padding: 12px 14px; border: 1px solid var(--border); border-left: 4px solid #557b9c; border-radius: 8px; background: var(--surface); }
  .topic-label strong, .topic-label span { display: block; }
  .topic-label strong { line-height: 1.25; }
  .topic-label span { margin-top: 2px; color: var(--muted); font-size: .72rem; }
  .topic-row data { font-variant-numeric: tabular-nums; font-weight: 760; text-align: right; }
  .topic-row data small { display: block; color: var(--muted); font-size: .65rem; font-weight: 650; }
  .agenda-grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; margin-top: 18px; }
  .agenda-grid article { padding: 15px; border: 1px solid var(--border); border-radius: 11px; background: var(--surface); }
  .agenda-grid h3 { font-size: 1rem; }
  .agenda-grid article > p { margin: 4px 0 12px; color: var(--muted); font-size: .78rem; }
  .meter-list, .compact-list { display: grid; gap: 6px; margin: 0; padding: 0; list-style: none; }
  .meter-list li { display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 4px 10px; font-size: .8rem; }
  .meter-list i { grid-column: 1/-1; height: 6px; overflow: hidden; border-radius: 99px; background: #e5e4df; }
  .meter-list i b { display: block; height: 100%; background: #315f78; }
  .pv-bar { display: flex; height: 13px; overflow: hidden; border-radius: 99px; background: #e5e4df; }
  .pv-bar i { display: block; }
  .pv-proposal { background: #275d8c; } .pv-mixed { background: #6d8fb5; }
  .pv-values { background: #5f8b70; } .pv-neither { background: #b0ada6; }
  .compact-list { margin-top: 9px; }
  .compact-list li { display: flex; justify-content: space-between; gap: 12px; padding-top: 5px; border-top: 1px solid var(--grid); color: var(--ink2); font-size: .78rem; }
  .compact-list li:first-child { border-top: 0; }
  .pv-key { grid-template-columns: 1fr 1fr; }
  .pv-key li { justify-content: flex-start; flex-wrap: wrap; }
  .pv-key strong { margin-left: auto; }
  .pv-dot { flex: 0 0 8px; width: 8px; height: 8px; margin-top: 5px; border-radius: 50%; }
  .rhetoric-stack { display: grid; gap: 18px; margin-top: 18px; }
  .rhetoric-card { padding: 18px; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
  .rhetoric-card h3 { font-size: 1.05rem; }
  .chart-summary { margin: 5px 0 12px; color: var(--ink2); font-size: .84rem; }
  .profile-figure { width: 100%; overflow: hidden; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
  .profile-page .chart { width: 100%; min-width: 0; }
  .rank-unavailable { max-width: none; margin: 12px 0; padding: 11px 13px; border: 1px solid #c59a43; border-radius: 8px; background: #fff7df; color: #4e3b13; }
  .measure-table-wrap { margin-top: 12px; }
  .measure-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
  .measure-table caption, .invocation-table caption { padding: 0 0 7px; color: var(--muted); font-size: .72rem; text-align: left; }
  .measure-table th, .measure-table td { padding: 8px 9px; border-top: 1px solid var(--grid); text-align: left; vertical-align: top; }
  .measure-table thead th { color: var(--muted); font-size: .68rem; letter-spacing: .04em; text-transform: uppercase; }
  .measure-table tbody th { width: 24%; }
  .not-ranked { color: #6b4c0c; font-weight: 750; }
  .rhetoric-card details { margin-top: 10px; }
  .evidence-card { margin-top: 11px; padding: 16px; border: 1px solid var(--border); border-radius: 11px; background: var(--surface); }
  .evidence-head { display: flex; flex-wrap: wrap; align-items: center; gap: 7px 10px; }
  .evidence-head h3 { margin-right: auto; font-size: 1rem; }
  .evidence-head span { color: var(--muted); font-size: .73rem; }
  .evidence-head strong { color: var(--ink2); }
  .evidence-badge { padding: 2px 7px; border: 1px solid var(--border); border-radius: 999px; background: var(--page); }
  .terms { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
  .term { padding: 3px 9px; border: 1px solid var(--border); border-radius: 7px; background: var(--page); font-size: .8rem; }
  blockquote { margin: 12px 0 0; padding: 2px 0 2px 13px; border-left: 3px solid #776f64; }
  blockquote p { max-width: 50rem!important; margin: 0!important; color: var(--ink2)!important; font-size: .9rem; }
  blockquote cite { display: block; margin-top: 5px; color: var(--muted); font-size: .74rem; font-style: normal; }
  .more-evidence { margin-top: 12px; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); }
  .more-evidence > summary { padding: 12px 14px; cursor: pointer; font-weight: 720; }
  .more-evidence > div, .more-evidence > p, .more-evidence > table { margin: 0 14px 14px; }
  .invocation-table { width: calc(100% - 28px); border-collapse: collapse; font-size: .78rem; }
  .invocation-table th, .invocation-table td { padding: 7px; border-top: 1px solid var(--grid); text-align: left; }
  .invocation-cards { display: none; }
  .similarity-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(245px,1fr)); gap: 11px; margin-top: 18px; }
  .similarity-card { padding: 15px; border: 1px solid var(--border); border-radius: 11px; background: var(--surface); }
  .similarity-card h3 { font-size: .98rem; }
  .similarity-card p { min-height: 3.4em; margin: 4px 0 10px; color: var(--muted); font-size: .73rem; }
  .similarity-card ol { display: grid; gap: 4px; margin: 0; padding-left: 22px; }
  .similarity-card li { padding-left: 2px; font-size: .82rem; }
  .similarity-card data { float: right; margin-left: 8px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .signature-list { display: grid; gap: 8px; margin: 18px 0 0; padding: 0; list-style: none; counter-reset: signature; }
  .signature-list li { display: grid; grid-template-columns: 58px minmax(0,1fr); gap: 12px; align-items: baseline; padding: 11px 13px; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); }
  .signature-list span { color: var(--muted); font-size: .76rem; font-variant-numeric: tabular-nums; }
  .president-nav { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 22px; }
  .president-nav a { display: flex; flex-direction: column; min-height: 72px; justify-content: center; padding: 12px 14px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); text-decoration: none; }
  .president-nav .next { text-align: right; }
  .president-nav span { color: var(--muted); font-size: .72rem; }
  .president-nav strong { margin-top: 3px; color: var(--ink); }
  .download-row { margin-top: 14px; }
  .unavailable { color: var(--muted); font-style: italic; }
  @media (max-width: 700px) {
    .source-key, .agenda-grid { grid-template-columns: 1fr; }
    .similarity-card p { min-height: 0; }
  }
  @media (max-width: 600px) {
    .profile-hero { padding: 20px 16px 6px; }
    .hero-tools { margin-bottom: 17px; }
    .hero-body { grid-template-columns: 76px minmax(0,1fr); gap: 14px; align-items: start; }
    .portrait { width: 76px; height: 76px; }
    .profile-hero h1 { font-size: clamp(1.62rem,8vw,2.2rem); overflow-wrap: anywhere; }
    .identity-meta { grid-column: 1/-1; margin-top: 0; }
    .record-note { margin: 13px 16px 0; }
    .profile-nav { margin-top: 12px; padding: 0 8px; }
    .profile-nav ul { grid-template-columns: repeat(3,minmax(0,1fr)); padding: 3px; }
    .profile-nav a { min-height: 40px; padding: 4px 2px; font-size: .69rem; }
    .profile-page main { padding-inline: 16px; }
    .profile-page section { margin-top: 38px; scroll-margin-top: calc(var(--global-nav-height, 0px) + 112px); }
    .glance-grid { grid-template-columns: 1fr 1fr; }
    .topic-row { grid-template-columns: 1fr 1fr; gap: 7px 12px; }
    .topic-label { grid-column: 1/-1; }
    .topic-row data { text-align: left; }
    .topic-row .topic-rel { text-align: right; }
    .rhetoric-card { padding: 14px; }
    .profile-figure { display: none; }
    .measure-table caption { display: block; width: 100%; }
    .measure-table thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
    .measure-table, .measure-table tbody { display: block; }
    .measure-table tr { display: grid; grid-template-columns: 1fr 1fr; padding: 9px 0; border-top: 1px solid var(--grid); }
    .measure-table th, .measure-table td { display: block; padding: 2px 5px; border: 0; }
    .measure-table tbody th { grid-column: 1/-1; width: auto; }
    .measure-table td::before { display: block; color: var(--muted); font-size: .62rem; font-weight: 720; text-transform: uppercase; }
    .measure-table td:nth-child(2)::before { content: "Absolute"; }
    .measure-table td:nth-child(3)::before { content: "Rank"; }
    .invocation-table { display: none; }
    .invocation-cards { display: grid; gap: 8px; }
    .invocation-cards > div { display: grid; grid-template-columns: auto minmax(0,1fr); gap: 3px 9px; padding: 9px 0; border-top: 1px solid var(--grid); }
    .invocation-cards dt { color: var(--muted); font-size: .65rem; font-weight: 720; text-transform: uppercase; }
    .invocation-cards dd { min-width: 0; overflow-wrap: anywhere; }
    .president-nav { grid-template-columns: 1fr; }
    .president-nav .next { text-align: left; }
  }
  @media (max-width: 420px) {
    .hero-tools { align-items: stretch; }
    .hero-tools > * { width: 100%; overflow-wrap: anywhere; }
    .compare-action { justify-content: center; }
  }
  @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto!important; } }
"""


def render_profile(
    profile: dict | str,
    data: dict | None = None,
    display_issues: list[str] | None = None,
) -> str:
    """Render a profile from one normalized model.

    The legacy ``(president, data, issues)`` signature remains accepted for
    focused callers, but production builds pass the already-built model.
    """
    if isinstance(profile, str):
        if data is None or display_issues is None:
            raise TypeError("Legacy render_profile calls require data and display_issues")
        view = profile_view_model(profile, data, display_issues)
    else:
        view = profile
    display = view["_view"]
    president = view["president"]
    display_name = html.escape(display["display_name"])
    party = html.escape(str(view["party"]))
    party_color = PARTY_COLORS.get(view["party"], PROFILE_MUTED)
    thin = bool(view["sample"]["thin_record"])
    figure_payload = profile_figure_payload(view)
    fig_json = figure_payload

    leading_ai = display.get("leading_ai_topic")
    leading_legacy = display.get("leading_legacy_issue")
    glance_cards = [
        ("Corpus support", display["speech_count_label"], f'{display["n_paragraphs"]:,} paragraphs'),
        ("Corpus record span", display["record_span_label"], "Available speech years, not term dates"),
        ("Corpus words", f'{view["sample"]["n_words"]:,}', "Tokenized source record"),
        ("Ranking status", display["support_label"], f'{SPARSE_MIN_SPEECHES}-speech minimum'),
    ]
    if leading_ai:
        glance_cards.append((
            "Leading AI topic",
            leading_ai["name"],
            f'{float(leading_ai["share"]):.1f}% of paragraphs',
        ))
    if leading_legacy:
        glance_cards.append((
            "Leading legacy issue",
            leading_legacy["label"],
            f'{float(leading_legacy["share"]) * 100:.1f}% of paragraphs',
        ))
    glance_html = "".join(
        f'<article class="glance-card"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(str(value))}</strong><small>{html.escape(note)}</small></article>'
        for label, value, note in glance_cards
    )

    topic_count = display["topic_count"]
    topic_word = "Six" if topic_count == 6 else str(topic_count)
    thin_notice = ""
    if thin:
        thin_notice = f'''<aside class="record-note" data-thin-record="true" role="note" aria-labelledby="thin-record-title">
  <strong id="thin-record-title">Thin corpus record · {html.escape(display["speech_count_label"])}</strong>
  {html.escape(view["sample"]["warning"] or "")}
  Absolute values and excerpts remain visible. Unsupported percentiles are shown as N/A and are not plotted.
</aside>'''

    legacy_rows = profile_measure_rows(view, "rhetorical_radar")
    ai_rows = profile_measure_rows(view, "ai_radar") if view.get("ai") else []
    rhetoric_html = _figure_block(
        view,
        figure_payload,
        key="rhetoric_legacy",
        title="Rhetorical fingerprint",
        summary=_measure_summary(legacy_rows, (
            "Eight deterministic lexical and readability measures, pairing each absolute value with its percentile among presidents represented by at least five corpus speeches."
        )),
        rows=legacy_rows,
    )
    if ai_rows:
        rhetoric_html += _figure_block(
            view,
            figure_payload,
            key="rhetoric_ai",
            title="AI-labeled rhetoric",
            summary=_measure_summary(ai_rows, (
                "Six paragraph-label measures, pairing each absolute rate or breadth value with its eligible-president percentile."
            )),
            rows=ai_rows,
        )

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{display_name} - Presidential Profiles</title>
<script src="../assets/plotly-3.0.1.min.js"></script>
<style>
{PAGE_CSS}
{PROFILE_CSS}
</style>
</head>
<body class="profile-page">
<header class="profile-hero">
  <div class="hero-tools"><a class="back-link" href="index.html">← President profiles</a>
    <a class="compare-action" href="{html.escape(display["compare_href"], quote=True)}">Compare this president</a></div>
  <div class="hero-body">
    <img class="portrait" src="../portraits/{view["slug"]}.png" width="118" height="118" alt="Portrait of {display_name}">
    <div><p class="eyebrow">Evidence-first presidential profile</p><h1>{display_name}</h1></div>
    <div class="identity-meta">
      <span class="party" style="--party-color:{party_color}">{party}</span>
      <span>Corpus record {html.escape(display["record_span_label"])}</span>
      <span>{html.escape(display["speech_count_label"])}</span>
      <span class="support {'thin' if thin else ''}">{html.escape(display["support_label"])}</span>
    </div>
  </div>
</header>
{thin_notice}
<nav class="profile-nav" aria-label="On this profile"><ul>
  <li><a href="#overview">Overview</a></li><li><a href="#agenda">Agenda</a></li>
  <li><a href="#evidence">Evidence</a></li><li><a href="#similarity">Similarity</a></li>
  <li><a href="#speeches">Speeches</a></li>
</ul></nav>
<main>
<section id="overview" data-profile-section="overview">
  <p class="section-kicker">Identity and record support</p><h2>Overview</h2>
  <p>These at-a-glance facts are mechanically derived from named corpus and annotation artifacts. Corpus years describe the available speeches, not time in office.</p>
  <div class="glance-grid">{glance_html}</div>
  <ul class="source-key" aria-label="Measurement sources">
    <li><strong>Corpus-derived</strong>Deterministic counts, lexical measures, readability, and source excerpts.</li>
    <li><strong>AI-labeled</strong>Frozen paragraph topics, closed-rubric flags, genres, and entity stance labels.</li>
    <li><strong>Legacy issue model</strong>CorEx paragraph shares and era-relative differences, kept visibly separate.</li>
  </ul>
</section>
<section id="agenda" data-profile-section="agenda">
  <p class="section-kicker">AI-labeled synthesis</p><h2>Agenda</h2>
  <p><span class="source-tag ai-source">AI-labeled</span> {topic_word} leading AI topics, ordered by this president’s paragraph share. Era differences are percentage points relative to contemporary presidents.</p>
  {_topic_rows_html(view)}
  {_ai_summary_grid(view.get("ai"))}
</section>
<section id="rhetoric" data-profile-section="rhetoric">
  <p class="section-kicker">Absolute values and eligible ranks</p><h2>Rhetoric</h2>
  <p>Direct bars replace the former radar charts. Absolute values and percentiles stay together; missing ranks have no visual position.</p>
  <div class="rhetoric-stack">{rhetoric_html}</div>
</section>
<section id="evidence" data-profile-section="evidence">
  <p class="section-kicker">Legacy issue model · named artifacts</p><h2>Evidence</h2>
  <p><span class="source-tag legacy-source">Legacy issue model</span> Excerpt-bearing cards remain in the producer’s declared strength order and appear first. Rate-only evidence follows in the native disclosure.</p>
  {_evidence_cards_from_info(view.get("issue_evidence", {}))}
{_invocation_details(view)}
</section>
<section id="similarity" data-profile-section="similarity">
  <p class="section-kicker">Five independent instruments</p><h2>Similarity</h2>
  <p>Each card uses only its named feature set and reports cosine scores. The five results are not combined into an overall likeness.</p>
  {_similarity_cards(view)}
</section>
<section id="speeches" data-profile-section="speeches">
  <p class="section-kicker">Chronology kept separate from synthesis</p><h2>Signature speeches</h2>
  <p>Speeches ranked by how strongly their embedding aligns with this president’s era-adjusted distinctive direction. Years and titles come from the Miller Center corpus.</p>
  {_signature_list(view)}
  <p class="download-row"><a href="../data/presidents/{view["slug"]}.json" download>Download this profile’s complete public data →</a></p>
  {_president_navigation(view)}
</section>
</main>
<footer><p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs, University of Virginia</a>. Emotion scores: NRC Emotion Lexicon (Mohammad &amp; Turney). <a href="../methodology.html">AI labeling method</a> · <a href="https://github.com/jacobfulfyll/presidential_profiles">Code</a>.</p></footer>
<script>
const FIGS = {json_for_script(fig_json)};
const compactProfile = window.matchMedia("(max-width: 600px)");
function mountProfileFigures() {{
  if (compactProfile.matches) return;
  for (const el of document.querySelectorAll(".chart[data-fig]")) {{
    if (el.dataset.mounted === "true") continue;
    const fig = FIGS[el.dataset.fig];
    if (!fig) continue;
    Plotly.newPlot(el, fig.data, fig.layout, {{displayModeBar:false,responsive:true}});
    el.dataset.mounted = "true";
  }}
}}
mountProfileFigures();
if (compactProfile.addEventListener) compactProfile.addEventListener("change", mountProfileFigures);
</script>
</body>
</html>
'''


INDEX_CSS = r"""
  .crumbs { margin-bottom: 18px; font-size: .88rem; }
  .directory-grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(230px,1fr)); gap: 11px; margin-top: 24px; }
  .president-card { display: grid; grid-template-columns: 48px minmax(0,1fr); gap: 11px; padding: 13px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); color: var(--ink); text-decoration: none; }
  .president-card:hover { border-color: #726d64; }
  .president-card img { width: 48px; height: 48px; border: 1px solid var(--border); border-radius: 50%; object-fit: cover; }
  .president-card strong { display: block; line-height: 1.25; }
  .card-meta { margin-top: 3px; color: var(--muted); font-size: .75rem; }
  .card-measure { grid-column: 1/-1; color: var(--ink2); font-size: .8rem; }
  .card-measure span { margin-right: 4px; color: var(--muted); font-size: .64rem; font-weight: 760; letter-spacing: .045em; text-transform: uppercase; }
  .card-support { grid-column: 1/-1; font-size: .7rem; font-weight: 720; color: #34523c; }
  .card-support.thin { color: #6b4c0c; }
  @media (max-width: 420px) { .directory-grid { grid-template-columns: 1fr; } }
"""


def render_index(data: dict, display_issues: list[str]) -> str:
    scores = data["scores"]
    issues_df = data["issues"]
    ai_presidents = data.get("ai", {}).get("by_president", {})
    cards = []
    for president in president_chronology(scores.index):
        score = scores.loc[president]
        n_paragraphs = float(issues_df.loc[president, "n_paragraphs"])
        relative = {
            issue: float(issues_df.loc[president, f"rel_{issue}"])
            for issue in display_issues
            if float(issues_df.loc[president, f"share_{issue}"]) * n_paragraphs
            >= MIN_ISSUE_PARAS
        }
        if not relative:
            relative = {
                issue: float(issues_df.loc[president, f"rel_{issue}"])
                for issue in display_issues
            }
        top_issue = max(relative, key=relative.get)
        top_issue_label = html.escape(DISCOVERED_LABELS.get(top_issue, top_issue))
        ai_info = ai_presidents.get(president)
        ai_badge = ""
        if ai_info and ai_info.get("top_topics"):
            ai_badge = (
                '<div class="card-measure"><span>AI topic</span>'
                + html.escape(ai_info["top_topics"][0]["name"])
                + "</div>"
            )
        n_speeches = int(score["n_speeches"])
        thin = n_speeches < SPARSE_MIN_SPEECHES
        display_name = html.escape(public_display_name(president))
        cards.append(f'''<a class="president-card" href="{slug(president)}.html">
  <img src="../portraits/{slug(president)}.png" width="48" height="48" alt="">
  <div><strong>{display_name}</strong><div class="card-meta">{html.escape(str(score["party"]))} · {int(score["first_year"])}–{int(score["last_year"])} · {format_speech_count(n_speeches)}</div></div>
  <div class="card-measure"><span>Legacy issue</span>{top_issue_label}</div>{ai_badge}
  <div class="card-support {'thin' if thin else ''}">{'Insufficient record for ranks' if thin else 'Supported for percentile ranking'}</div>
</a>''')
    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>President Profiles</title><style>{PAGE_CSS}{INDEX_CSS}</style></head>
<body><header><p class="crumbs"><a href="../index.html">← Dashboard</a> · <a href="../compare.html">Compare</a> · <a href="../methodology.html">How the AI labels work</a></p>
<h1>President Profiles</h1><p class="sub">Evidence-first profiles for every president in the corpus. Cards keep corpus-derived, AI-labeled, and legacy issue measurements visibly separate.</p>
<div class="directory-grid">{''.join(cards)}</div></header>
<footer><p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs, University of Virginia</a>. <a href="../methodology.html">AI labeling method</a>.</p></footer></body></html>'''


def write_profiles(
    data: dict,
    site_dir,
    *,
    views: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Write all profiles and return the shared normalized model mapping."""
    display_issues = topic_quality.display_issues(data["issue_meta"]["issues"])
    complete_order = president_chronology(
        data["scores"].index,
        require_complete=True,
    )
    models = views or build_profile_view_models(
        data,
        display_issues,
        require_complete=True,
    )
    expected = set(complete_order)
    if (
        set(models) != expected
        or len({model["slug"] for model in models.values()}) != len(models)
    ):
        raise ValueError("Profile models must cover every unique president and slug")
    for president, model in models.items():
        if model.get("president") != president:
            raise ValueError("Profile model mapping key disagrees with its president")
        validate_profile_view_model(model)

    president_dir = site_dir / "presidents"
    president_dir.mkdir(parents=True, exist_ok=True)
    data_dir = site_dir / "data" / "presidents"
    data_dir.mkdir(parents=True, exist_ok=True)
    (president_dir / "index.html").write_text(
        render_index(data, display_issues), encoding="utf-8"
    )
    for president, view in models.items():
        (president_dir / f"{view['slug']}.html").write_text(
            render_profile(view), encoding="utf-8"
        )
        payload = profile_public_payload(view)
        (data_dir / f"{view['slug']}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    print(f"  wrote {len(models)} profile pages + index to docs/presidents/")
    return models
