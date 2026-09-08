"""Render corpus-based president profiles and their public v3 payloads."""

from __future__ import annotations

import gzip
import html
import json
import math
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit

from . import topic_quality
from .figures import PARTY_COLORS
from .profiles import (
    FEATURE_SIMILARITY,
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
from .site_style import PAGE_CSS


_MILLER_SPEECH_PREFIX = "https://millercenter.org/the-presidency/presidential-speeches/"
_MILLER_SPEECH_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _normalized_profile_source_url(value: object) -> str:
    """Normalize one profile source link and reject anything off-corpus."""
    supplied = str(value or "").strip()
    if not supplied:
        raise ValueError("Profile source URL is missing")
    normalized = miller_speech_url(supplied)
    parsed = urlsplit(normalized)
    path_prefix = "/the-presidency/presidential-speeches/"
    slug_value = parsed.path.removeprefix(path_prefix)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "millercenter.org"
        or not parsed.path.startswith(path_prefix)
        or not _MILLER_SPEECH_SLUG.fullmatch(slug_value)
    ):
        raise ValueError("Profile source URL is outside the normalized Miller speech corpus")
    return _MILLER_SPEECH_PREFIX + slug_value


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
PROFILE_HTML_RAW_MAX = 125_000
PROFILE_HTML_GZIP_MAX = 30_000
DIRECTORY_HTML_RAW_MAX = 48_000
DIRECTORY_HTML_GZIP_MAX = 10_000
FIRST_EXPANSION_RAW_MAX = 1_350_000
FIRST_EXPANSION_GZIP_MAX = 190_000


def _encoded_sizes(value: str | bytes) -> tuple[int, int]:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return len(raw), len(gzip.compress(raw, compresslevel=9, mtime=0))


def _require_html_budget(value: str, *, label: str, raw_max: int, gzip_max: int) -> None:
    raw_size, gzip_size = _encoded_sizes(value)
    if raw_size > raw_max or gzip_size > gzip_max:
        raise ValueError(
            f"{label} exceeds its byte budget: {raw_size}/{gzip_size} bytes"
        )


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
            "url": _normalized_profile_source_url(item.get("url")),
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


def profile_figure_payload(view: dict) -> dict:
    """Compatibility shim: profile charts are now entirely server rendered."""
    validate_profile_view_model(view)
    return {}


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
    *,
    key: str,
    title: str,
    summary: str,
    rows: list[dict],
) -> str:
    title_id = f"{key}-title"
    summary_id = f"{key}-summary"
    bars = []
    for row in rows:
        percentile = row.get("percentile")
        if percentile is None:
            bars.append(
                '<li class="percentile-row is-unranked">'
                f'<span class="bar-label">{html.escape(row["short"])}</span>'
                '<span class="bar-unranked">No eligible-president position</span></li>'
            )
            continue
        position = float(percentile)
        if not math.isfinite(position) or not 0 <= position <= 100:
            raise ValueError(f"Invalid profile percentile for {row['key']}")
        bars.append(
            '<li class="percentile-row">'
            f'<span class="bar-label">{html.escape(row["short"])}</span>'
            '<span class="bar-track" aria-hidden="true">'
            f'<i style="width:{position:.4f}%"></i></span>'
            f'<span class="bar-rank">{html.escape(row["percentile_display"])}</span></li>'
        )
    visual = (
        f'<ol class="percentile-bars" aria-labelledby="{title_id}" '
        f'aria-describedby="{summary_id}">{"".join(bars)}</ol>'
    )
    if not any(row.get("percentile") is not None for row in rows):
        visual = (
            '<p class="rank-unavailable"><strong>N/A · Insufficient record.</strong> '
            'No percentile position is rendered; exact absolute values remain available.</p>'
            + visual
        )
    exact = f'''<details class="exact-values">
  <summary>Exact values and eligible-president ranks</summary>
  {_measure_table(rows, title + ": exact values and eligible-president ranks")}
  <p><a href="../data/presidents/{view['slug']}.json" download>Download the complete profile data →</a></p>
</details>'''
    return f'''<article class="rhetoric-card">
  <span class="source-tag {'ai-source' if key == 'rhetoric_ai' else 'corpus-source'}">{'AI-labeled' if key == 'rhetoric_ai' else 'Corpus-derived'}</span>
  <h3 id="{title_id}">{html.escape(title)}</h3>
  <p class="chart-summary" id="{summary_id}">{html.escape(summary)}</p>
  {visual}
  {exact}
</article>'''


def _evidence_receipt_html(receipt: dict, *, primary: bool) -> str:
    doc_name = str(receipt.get("doc_name") or "")
    para_idx = receipt.get("para_idx")
    key = f"({doc_name}, {para_idx})" if doc_name and para_idx is not None else "Key unavailable"
    title = str(receipt.get("speech_title") or receipt.get("title") or doc_name)
    date = str(receipt.get("speech_date") or receipt.get("date") or receipt.get("year") or "Date unavailable")
    year = receipt.get("year")
    source_url = str(receipt.get("source_url") or receipt.get("url") or "")
    if source_url:
        from .actual_speaker_invocation_network import normalized_miller_url

        source_url = normalized_miller_url(doc_name, source_url)
    source = (
        f'<a href="{html.escape(source_url, quote=True)}" target="_blank" rel="noopener">'
        f'{html.escape(title)}</a>'
        if source_url
        else html.escape(title)
    )
    owner = str(receipt.get("source_document_owner") or "Unavailable")
    owner_id = receipt.get("source_document_owner_profile_id")
    owner_display = owner + (f" ({owner_id})" if owner_id else "")
    speaker = str(receipt.get("actual_speaker") or "Unavailable")
    speaker_id = receipt.get("actual_speaker_profile_id")
    speaker_display = speaker + (f" ({speaker_id})" if speaker_id else "")
    eligibility = str(
        receipt.get("speaker_eligibility_state")
        or receipt.get("speaker_state")
        or receipt.get("speaker_exclusion_state")
        or "Eligibility state unavailable"
    )
    exclusion_reason = receipt.get("speaker_exclusion_reason")
    if exclusion_reason:
        eligibility += f": {exclusion_reason}"
    if receipt.get("cross_owner") is None:
        cross_owner = "Cross-owner state unavailable"
    elif receipt.get("cross_owner"):
        cross_owner = "Cross-owner: actual speaker differs from source-document owner"
    else:
        cross_owner = "Same actual speaker and source-document owner"
    excerpt = str(receipt.get("evidence_excerpt") or receipt.get("excerpt") or "")
    quote = (
        f'<blockquote><p>“{html.escape(excerpt)}”</p></blockquote>'
        if excerpt
        else '<p class="unavailable">No qualifying excerpt was selected.</p>'
    )
    role = str(receipt.get("selection_role") or ("primary" if primary else "additional"))
    return f'''<article class="evidence-receipt {'primary-receipt' if primary else ''}">
  <p class="receipt-source">{source} <span>· {html.escape(date)}{' · Corpus year ' + html.escape(str(year)) if year is not None else ''}</span></p>
  {quote}
  <dl class="receipt-audit">
    <div><dt>Keyed receipt</dt><dd><code>{html.escape(key)}</code></dd></div>
    <div><dt>Source-document owner</dt><dd>{html.escape(owner_display)}</dd></div>
    <div><dt>Actual speaker</dt><dd>{html.escape(speaker_display)}</dd></div>
    <div><dt>Attribution state</dt><dd>{html.escape(cross_owner)}; {html.escape(eligibility)}</dd></div>
    <div><dt>Selection role</dt><dd>{html.escape(role)}</dd></div>
  </dl>
</article>'''


def _issue_card_html(card: dict) -> str:
    legacy = card.get("legacy_v3", card)
    issue_key = str(card.get("issue") or legacy.get("issue") or "Issue")
    issue = html.escape(DISCOVERED_LABELS.get(issue_key, issue_key))
    words = "".join(
        f'<span class="term">{html.escape(str(word))}</span>'
        for word in legacy.get("words", [])
    )
    words_html = f'<div class="terms">{words}</div>' if words else ""
    quote_html = ""
    if legacy.get("quote"):
        # v3 stores producer-escaped quote text. Canonicalize through an
        # unescape/escape round trip so the historical bytes render once while
        # direct or hostile callers cannot introduce markup.
        safe_quote = html.escape(html.unescape(str(legacy["quote"])))
        quote_html = (
            f'<blockquote><p>“{safe_quote}”</p>'
            f'<cite>{html.escape(str(legacy.get("cite", "")))}</cite></blockquote>'
        )
    stance = (
        f'<span class="evidence-badge">{html.escape(str(legacy["stance"]))}</span>'
        if legacy.get("stance")
        else ""
    )
    topic_flag = (
        '<span class="evidence-badge">Topic of the day</span>'
        if legacy.get("topic_of_day")
        else ""
    )
    if "legacy_v3" not in card:
        return f'''<article class="evidence-card i-card">
  <div class="evidence-head"><h3>{issue}</h3>
    <span><strong>{float(legacy["share"]) * 100:.1f}%</strong> of paragraphs</span>
    <span>{float(legacy["rel"]):+.1f} pp vs era</span>{topic_flag}{stance}</div>
  {words_html}{quote_html}
  <p class="evidence-limitation">This rate is document-owner based. Speaker attribution is reported separately; excerpts are deterministic audit examples, not proof of representativeness, intent, influence, or policy success.</p>
</article>'''

    exact = card.get("exact_evidence") or {}
    issue_paragraphs = int(exact.get("issue_paragraph_count", exact.get("issue_paragraphs", 0)))
    total_paragraphs = int(
        exact.get(
            "total_document_owned_paragraph_count",
            exact.get("total_document_owned_paragraphs", exact.get("total_paragraphs", 0)),
        )
    )
    percentage = float(exact.get("percentage", exact.get("share_percentage", float(legacy.get("share", 0)) * 100)))
    document_count = int(exact.get("source_document_count", exact.get("carrying_document_count", 0)))
    baseline = float(exact.get("corpus_baseline_percentage", exact.get("baseline_percentage", 0)))
    multiple_value = exact.get("corpus_baseline_multiple", exact.get("baseline_multiple"))
    multiple = float(multiple_value) if multiple_value is not None else None
    era_difference = float(
        exact.get(
            "era_difference_percentage_points",
            exact.get("era_difference_pp", legacy.get("rel", 0)),
        )
    )
    why = card.get("why_shown") or {}
    why_copy = str(
        (
            why.get("text", why.get("prose"))
            if isinstance(why, dict)
            else why
        )
        or "Threshold basis unavailable."
    )
    claim_value = card.get("claim")
    claim = str(
        (claim_value.get("text") if isinstance(claim_value, dict) else claim_value)
        or "This issue passed the declared evidence-card rules."
    )
    receipts = list(card.get("receipts") or [])
    if receipts:
        primary_receipt = _evidence_receipt_html(receipts[0], primary=True)
        extra_receipts = ""
        if len(receipts) > 1:
            extra_receipts = f'''<details class="receipt-more"><summary>Additional keyed receipts ({len(receipts) - 1})</summary>
  <div>{''.join(_evidence_receipt_html(receipt, primary=False) for receipt in receipts[1:3])}</div>
</details>'''
    else:
        primary_receipt = '<p class="unavailable">No qualifying excerpt was selected for this rate-only card.</p>'
        extra_receipts = ""
    method = card.get("method") or {}
    method_stance = method.get("stance", legacy.get("stance"))
    method_words = method.get("distinctive_vocabulary", legacy.get("words", []))
    method_html = "".join(
        f'<span class="term">{html.escape(str(word))}</span>' for word in method_words
    )
    method_definitions = card.get("_method_definitions") or {}
    stance_method = method_definitions.get(
        method.get("stance_method"), method.get("stance_method") or "Current legacy stance method"
    )
    vocabulary_method = method_definitions.get(
        method.get("distinctive_vocabulary_method"),
        method.get("distinctive_vocabulary_method") or "Current legacy distinctive-vocabulary method",
    )
    baseline_display = (
        f"{baseline:.1f}% · {multiple:.2f}×"
        if multiple is not None
        else f"{baseline:.1f}% · multiple unavailable"
    )
    limitation = str(card.get("limitation") or (
        "The rate is document-owner based; speaker attribution is shown separately; "
        "excerpts are deterministic audit examples rather than proof, representativeness, "
        "intent, influence, or policy success."
    ))
    return f'''<article class="evidence-card i-card">
  <div class="evidence-head"><h3>{issue}</h3>{topic_flag}</div>
  <p class="evidence-claim">{html.escape(claim)}</p>
  <dl class="evidence-counts">
    <div><dt>Document-owned paragraphs</dt><dd>{issue_paragraphs:,} / {total_paragraphs:,} ({percentage:.1f}%)</dd></div>
    <div><dt>Source documents carrying issue</dt><dd>{document_count:,}</dd></div>
    <div><dt>Paragraph-weighted corpus baseline</dt><dd>{baseline_display}</dd></div>
    <div><dt>Difference from era peers</dt><dd>{era_difference:+.2f} percentage points</dd></div>
  </dl>
  <p class="why-shown"><strong>Why shown:</strong> {html.escape(why_copy)}</p>
  {primary_receipt}{extra_receipts}
  <details class="card-audit"><summary>Methods and card audit</summary>
    <p><strong>Legacy stance:</strong> {html.escape(str(method_stance or 'No stance label'))}</p>
    <p><strong>Legacy stance method:</strong> {html.escape(str(stance_method))}</p>
    <p><strong>Distinctive-vocabulary method:</strong> {html.escape(str(vocabulary_method))}</p>
    <div class="terms">{method_html}</div>
  </details>
  <p class="evidence-limitation">{html.escape(limitation)}</p>
</article>'''


def _evidence_cards_from_info(info: dict) -> str:
    cards = list(info.get("cards", []))
    method_definitions = info.get("method_definitions") or {}
    if method_definitions:
        cards = [{**card, "_method_definitions": method_definitions} for card in cards]
    if cards:
        leading = "".join(_issue_card_html(card) for card in cards[:2])
        remainder = ""
        if len(cards) > 2:
            remainder = f'''<details class="more-evidence"><summary>More issue evidence ({len(cards) - 2})</summary>
  <div>{''.join(_issue_card_html(card) for card in cards[2:])}</div></details>'''
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
    thin_warning = info.get("thin_record_warning")
    if thin_warning:
        cards_html = (
            f'<p class="evidence-thin-warning"><strong>Thin source-document record.</strong> '
            f'{html.escape(str(thin_warning))}</p>' + cards_html
        )
    return cards_html


def _issue_cards_html(president: str, data: dict) -> str:
    """Compatibility helper used by the issue-card security tests."""
    return _evidence_cards_from_info(
        data["issue_cards"].get(president, {"cards": [], "voice": []})
    )


def _similarity_cards(view: dict) -> str:
    categories = view.get("feature_neighbors", {})
    cards = []
    for instrument_number, (key, meta) in enumerate(FEATURE_SIMILARITY.items(), start=1):
        neighbors = categories.get(key, [])[:3]
        if neighbors:
            rows = "".join(
                f'''<li><div class="similarity-match"><a href="{slug(item["president"])}.html"><span>{html.escape(public_display_name(item["president"]))}</span></a>
  <data value="{float(item["similarity"]):.4f}">{float(item["similarity"]):.2f} cosine</data></div></li>'''
                for item in neighbors
            )
        else:
            rows = '<li class="unavailable">No adequately sampled match.</li>'
        cards.append(f'''<article class="similarity-card" data-similarity="{key}">
  <p class="similarity-kicker">Instrument {instrument_number:02d}</p>
  <h3>{html.escape(meta["label"])}</h3><p class="similarity-description">{html.escape(meta["description"])}</p>
  <ol>{rows}</ol></article>''')
    return '<div class="similarity-grid">' + "".join(cards) + "</div>"


def _signature_list(view: dict) -> str:
    speeches = view.get("signature_speeches", [])
    if not speeches:
        return '<p class="unavailable">No signature speech was selected.</p>'
    return '<ol class="signature-list">' + "".join(
        f'''<li><span>{int(item.get("year", 0))}</span><a href="{html.escape(_normalized_profile_source_url(item.get("url")), quote=True)}" target="_blank" rel="noopener">{html.escape(str(item["title"]))}</a></li>'''
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


def _connections_content(
    view: dict,
    context_index: dict | None,
    context_shard: dict | None,
) -> tuple[str, str, str]:
    """Return server markup, CSS, and lazy bootstrap for Connections."""
    if context_shard is None:
        fallback = '''<section id="connections" data-profile-section="connections">
  <p class="section-kicker">Eligible actual-speaker paragraphs</p><h2>Connections</h2>
  <p class="population-label"><strong>Population:</strong> eligible paragraphs attributed to the actual speaker.</p>
  <section id="invocations"><h3>Named former-president invocations</h3><p class="unavailable">Connection context is unavailable in this rendering.</p></section>
  <section id="topic-network"><h3>Shared AI-topic emphasis</h3><p class="unavailable">Connection context is unavailable in this rendering.</p></section>
</section>'''
        return fallback, "", ""

    from .profile_connections_assets import (
        PROFILE_CONNECTIONS_CSS,
        render_profile_connections,
    )

    markup = render_profile_connections(view, context_index or {}, context_shard)
    bootstrap = r'''<script>
(() => {
  const root = document.querySelector("#connections[data-profile-connections]");
  if (!root) return;
  let modulePromise = null;
  let retried = false;
  const status = root.querySelector("[data-pc-status]");
  const retry = root.querySelector("[data-pc-retry]");
  const showFailure = () => {
    if (status) status.textContent = "Interactive Connections could not load. All server-rendered evidence remains available.";
    if (retry && !retried) retry.hidden = false;
  };
  const load = () => {
    if (modulePromise) return modulePromise;
    if (retry) retry.hidden = true;
    modulePromise = import("../assets/profile-connections-v1.js")
      .then(module => module.enhanceProfileConnections(root, {
        indexUrl: root.dataset.indexUrl,
        shardUrl: root.dataset.shardUrl,
        presidentProfileId: root.dataset.presidentProfileId
      }))
      .catch(error => { modulePromise = null; showFailure(); throw error; });
    return modulePromise;
  };
  if (retry) retry.addEventListener("click", () => {
    if (retried) return;
    retried = true;
    load().catch(() => {});
  });
  root.addEventListener("focusin", () => { load().catch(() => {}); }, {once:true});
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      observer.disconnect();
      load().catch(() => {});
    }, {rootMargin:"600px 0px"});
    observer.observe(root);
  } else {
    const maybeLoad = () => {
      const bounds = root.getBoundingClientRect();
      if (bounds.top > innerHeight + 600 || bounds.bottom < -600) return;
      removeEventListener("scroll", maybeLoad);
      removeEventListener("resize", maybeLoad);
      load().catch(() => {});
    };
    addEventListener("scroll", maybeLoad, {passive:true});
    addEventListener("resize", maybeLoad, {passive:true});
    maybeLoad();
  }
})();
</script>'''
    return markup, PROFILE_CONNECTIONS_CSS, bootstrap


PROFILE_CSS = r"""
  body.profile-page { overflow-x: clip; }
  .profile-page :where(header, main, section, article, nav, div) { min-width: 0; }
  .profile-page :where(a, button, summary) { min-width: 44px; min-height: 44px; }
  .profile-page a { display: inline-flex; align-items: center; }
  .profile-page summary { padding-block: 10px; cursor: pointer; }
  .profile-hero { padding-top: 28px; }
  .hero-tools { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 22px; }
  .back-link { color: var(--ink2); font-size: .88rem; }
  .compare-action { display: inline-flex; align-items: center; min-height: 44px; padding: 8px 15px; border: 1px solid #5d350c; border-radius: 8px; background: #fffaf4; color: #5d350c; font-weight: 720; text-decoration: none; }
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
  .profile-nav { position: sticky; z-index: 8; top: calc(var(--global-nav-height, 61px) + 6px); max-width: 940px; margin: 18px auto 0; padding: 0 20px; }
  .profile-nav { overflow-x: auto; overscroll-behavior-inline: contain; scrollbar-width: thin; }
  .profile-nav-cue { display: none; color: var(--muted); font-size: 12px; font-weight: 740; white-space: nowrap; }
  .profile-nav ul { display: grid; grid-template-columns: repeat(7,minmax(0,1fr)); min-width: 760px; padding: 5px; border: 1px solid var(--border); border-radius: 12px; background: rgba(252,252,251,.96); box-shadow: 0 4px 16px rgba(11,11,11,.06); list-style: none; }
  .profile-nav a { display: grid; min-height: 44px; place-items: center; padding: 5px 4px; border-radius: 8px; color: var(--ink2); font-size: .8rem; font-weight: 700; text-align: center; text-decoration: none; }
  .profile-nav a:hover { background: var(--page); color: var(--ink); }
  .profile-page main { padding-top: 2px; }
  .profile-page section { scroll-margin-top: calc(var(--global-nav-height, 61px) + 72px); }
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
  .percentile-bars { display: grid; gap: 10px; margin: 16px 0 0; padding: 0; list-style: none; }
  .percentile-row { display: grid; grid-template-columns: minmax(110px,1fr) minmax(120px,3fr) minmax(94px,auto); gap: 10px; align-items: center; }
  .bar-label { font-size: .8rem; font-weight: 720; }
  .bar-track { height: 14px; overflow: hidden; border: 1px solid #b9b7b0; border-radius: 99px; background: #ecebe7; }
  .bar-track i { display: block; height: 100%; background: #315f78; }
  .bar-rank, .bar-unranked { color: var(--ink2); font-size: .76rem; font-variant-numeric: tabular-nums; }
  .bar-unranked { grid-column: 2/-1; font-style: italic; }
  .rank-unavailable { max-width: none; margin: 12px 0; padding: 11px 13px; border: 1px solid #c59a43; border-radius: 8px; background: #fff7df; color: #4e3b13; }
  .measure-table-wrap { margin-top: 12px; }
  .measure-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
  .measure-table caption, .invocation-table caption { padding: 0 0 7px; color: var(--muted); font-size: .72rem; text-align: left; }
  .measure-table th, .measure-table td { padding: 8px 9px; border-top: 1px solid var(--grid); text-align: left; vertical-align: top; }
  .measure-table thead th { color: var(--muted); font-size: .68rem; letter-spacing: .04em; text-transform: uppercase; }
  .measure-table tbody th { width: 24%; }
  .not-ranked { color: #6b4c0c; font-weight: 750; }
  .rhetoric-card details { margin-top: 14px; border: 1px solid var(--border); border-radius: 9px; }
  .rhetoric-card details > summary { min-height: 44px; padding: 11px 13px; cursor: pointer; font-weight: 720; }
  .rhetoric-card details .measure-table-wrap, .rhetoric-card details > p { margin: 0 13px 13px; }
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
  .evidence-claim { max-width: 54rem; margin: 10px 0; font-weight: 680; }
  .evidence-counts, .receipt-audit { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 7px 14px; margin: 12px 0; }
  .evidence-counts div, .receipt-audit div { padding-top: 7px; border-top: 1px solid var(--grid); }
  .evidence-counts dt, .receipt-audit dt { color: var(--muted); font-size: .67rem; font-weight: 760; letter-spacing: .03em; text-transform: uppercase; }
  .evidence-counts dd, .receipt-audit dd { margin: 2px 0 0; overflow-wrap: anywhere; font-size: .8rem; }
  .why-shown { max-width: none!important; padding: 10px 12px; border-left: 3px solid #6f6454; background: var(--page); font-size: .82rem; }
  .evidence-receipt { margin-top: 12px; padding: 13px; border: 1px solid var(--border); border-radius: 9px; background: var(--page); }
  .receipt-source { max-width: none!important; margin: 0!important; font-size: .82rem; }
  .receipt-source span { color: var(--muted); }
  .receipt-more, .card-audit { margin-top: 10px; border: 1px solid var(--border); border-radius: 8px; }
  .receipt-more > summary, .card-audit > summary { min-height: 44px; padding: 11px 12px; cursor: pointer; font-weight: 720; }
  .receipt-more > div, .card-audit > p, .card-audit > div { margin: 0 12px 12px; }
  .evidence-limitation { max-width: none!important; margin: 13px 0 0!important; color: var(--ink2)!important; font-size: .78rem!important; }
  .evidence-thin-warning { max-width: none!important; padding: 11px 13px; border: 1px solid #a47b22; border-left-width: 5px; border-radius: 9px; background: #fff7df; color: #4e3b13!important; }
  .more-evidence { margin-top: 12px; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); }
  .more-evidence > summary { padding: 12px 14px; cursor: pointer; font-weight: 720; }
  .more-evidence > div, .more-evidence > p, .more-evidence > table { margin: 0 14px 14px; }
  .evidence-section-disclosure { margin-top: 14px; border: 1px solid var(--border); border-radius: 11px; background: var(--surface); }
  .evidence-section-disclosure > summary { padding: 13px 15px; font-weight: 760; }
  .evidence-section-body { padding: 0 15px 16px; }
  .evidence-section-body > .section-kicker { margin-top: 4px; }
  .invocation-table { width: calc(100% - 28px); border-collapse: collapse; font-size: .78rem; }
  .invocation-table th, .invocation-table td { padding: 7px; border-top: 1px solid var(--grid); text-align: left; }
  .invocation-cards { display: none; }
  .similarity-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(270px,1fr)); gap: 12px; margin-top: 18px; }
  .similarity-card { position: relative; overflow: hidden; padding: 17px 16px 14px; border: 1px solid var(--border); border-radius: 11px; background: var(--surface); box-shadow: 0 5px 16px rgba(47,38,30,.045); }
  .similarity-card::before { position: absolute; inset: 0 0 auto; height: 4px; background: #315f78; content: ""; }
  .similarity-card .similarity-kicker { min-height: 0; margin: 0 0 5px; color: #6a421b; font-size: .65rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
  .similarity-card h3 { margin: 0; font-size: 1.02rem; }
  .similarity-card .similarity-description { min-height: 3.4em; margin: 5px 0 12px; color: var(--muted); font-size: .73rem; }
  .similarity-card ol { display: grid; margin: 0; padding-left: 23px; }
  .similarity-card li { min-height: 48px; padding: 2px 0 2px 2px; border-top: 1px solid var(--grid); font-size: .82rem; }
  .similarity-card li::marker { color: var(--muted); font-variant-numeric: tabular-nums; }
  .similarity-match { display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 10px; align-items: center; min-height: 44px; }
  .similarity-card li a { display: flex; min-height: 44px; align-items: center; overflow-wrap: anywhere; font-weight: 690; }
  .similarity-card data { display: inline-flex; min-height: 30px; align-items: center; justify-content: center; padding: 4px 8px; border: 1px solid var(--border); border-radius: 999px; background: var(--page); color: var(--ink2); font-size: .72rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
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
  @media (max-width: 760px) {
    .profile-nav { top: calc(var(--global-nav-height, 56px) + 6px); scrollbar-gutter: stable; }
    .profile-nav-cue { display: inline-flex; min-height: 24px; align-items: center; padding-inline: 7px; }
    .profile-nav a { font-size: 14px; }
    .profile-page :where(.eyebrow, .glance-card span, .source-tag, .topic-label span,
      .evidence-head span, blockquote cite, .evidence-counts dt, .receipt-audit dt,
      .similarity-kicker, .similarity-description, .similarity-card data,
      .president-nav span, .invocation-cards dt) { font-size: 12px!important; }
    .profile-page section { scroll-margin-top: calc(var(--global-nav-height, 56px) + 96px); }
  }
  @media (max-width: 700px) {
    .source-key, .agenda-grid { grid-template-columns: 1fr; }
    .similarity-card .similarity-description { min-height: 0; }
  }
  @media (max-width: 600px) {
    .profile-hero { padding: 20px 16px 6px; }
    .hero-tools { margin-bottom: 17px; }
    .hero-body { grid-template-columns: 76px minmax(0,1fr); gap: 14px; align-items: start; }
    .portrait { width: 76px; height: 76px; }
    .profile-hero h1 { font-size: clamp(1.62rem,8vw,2.2rem); overflow-wrap: anywhere; }
    .identity-meta { grid-column: 1/-1; margin-top: 0; }
    .record-note { margin: 13px 16px 0; }
    .profile-nav { max-width: 100%; margin-top: 12px; padding: 0 8px; }
    .profile-nav ul { display: flex; width: max-content; min-width: 0; padding: 3px; }
    .profile-nav li { flex: 0 0 auto; }
    .profile-nav a { min-width: 92px; min-height: 44px; padding: 4px 8px; font-size: 14px; }
    .profile-page main { padding-inline: 16px; }
    .profile-page section { margin-top: 38px; scroll-margin-top: calc(var(--global-nav-height, 0px) + 112px); }
    .glance-grid { grid-template-columns: 1fr 1fr; }
    .topic-row { grid-template-columns: 1fr 1fr; gap: 7px 12px; }
    .topic-label { grid-column: 1/-1; }
    .topic-row data { text-align: left; }
    .topic-row .topic-rel { text-align: right; }
    .rhetoric-card { padding: 14px; }
    .percentile-row { grid-template-columns: minmax(96px,1fr) minmax(90px,2fr); gap: 6px 9px; }
    .bar-rank { grid-column: 2; }
    .bar-unranked { grid-column: 2; }
    .measure-table caption { display: block; width: 100%; }
    .measure-table thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
    .measure-table, .measure-table tbody { display: block; }
    .measure-table tr { display: grid; grid-template-columns: 1fr 1fr; padding: 9px 0; border-top: 1px solid var(--grid); }
    .measure-table th, .measure-table td { display: block; padding: 2px 5px; border: 0; }
    .measure-table tbody th { grid-column: 1/-1; width: auto; }
    .measure-table td::before { display: block; color: var(--muted); font-size: 12px; font-weight: 720; text-transform: uppercase; }
    .measure-table td:nth-child(2)::before { content: "Absolute"; }
    .measure-table td:nth-child(3)::before { content: "Rank"; }
    .invocation-table { display: none; }
    .invocation-cards { display: grid; gap: 8px; }
    .invocation-cards > div { display: grid; grid-template-columns: auto minmax(0,1fr); gap: 3px 9px; padding: 9px 0; border-top: 1px solid var(--grid); }
    .invocation-cards dt { color: var(--muted); font-size: .65rem; font-weight: 720; text-transform: uppercase; }
    .invocation-cards dd { min-width: 0; overflow-wrap: anywhere; }
    .evidence-counts, .receipt-audit { grid-template-columns: 1fr; }
    .president-nav { grid-template-columns: 1fr; }
    .president-nav .next { text-align: left; }
  }
  @media (max-width: 420px) {
    .hero-tools { align-items: stretch; }
    .hero-tools > * { width: 100%; overflow-wrap: anywhere; }
    .compare-action { justify-content: center; }
  }
  @media (forced-colors: active) {
    .profile-nav ul, .rhetoric-card, .evidence-card, .evidence-section-disclosure, .bar-track { border-color: CanvasText; }
    .bar-track i { background: Highlight; }
    :where(.profile-nav a, summary, button, a):focus-visible { outline: 3px solid Highlight; outline-offset: 2px; }
  }
  @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto!important; } }
"""


def render_profile(
    profile: dict | str,
    data: dict | None = None,
    display_issues: list[str] | None = None,
    *,
    context: dict | None = None,
    context_index: dict | None = None,
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
            "Largest legacy issue share",
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
            key="rhetoric_ai",
            title="AI-labeled rhetoric",
            summary=_measure_summary(ai_rows, (
                "Six paragraph-label measures, pairing each absolute rate or breadth value with its eligible-president percentile."
            )),
            rows=ai_rows,
        )
    connections_html, connections_css, connections_bootstrap = _connections_content(
        view, context_index, context
    )
    if context is not None:
        evidence_info = dict(context.get("legacy_issue_evidence", {}))
        method_definitions = (
            (context_index or {}).get("policies", {}).get("legacy_issue_methods", {})
        )
        if method_definitions:
            evidence_info["method_definitions"] = method_definitions
    else:
        evidence_info = view.get("issue_evidence", {})

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{display_name} - Presidential Profiles</title>
<style>
{PAGE_CSS}
{PROFILE_CSS}
{connections_css}
</style>
</head>
<body class="profile-page">
<header class="profile-hero">
  <div class="hero-tools"><a class="back-link" href="index.html">← President profiles</a>
    <a class="compare-action" href="{html.escape(display["compare_href"], quote=True)}">Compare this president</a></div>
  <div class="hero-body">
    <img class="portrait" src="../portraits/{view["slug"]}.png" width="118" height="118" alt="Portrait of {display_name}">
    <div><p class="eyebrow">Corpus-based presidential profile</p><h1>{display_name}</h1></div>
    <div class="identity-meta">
      <span class="party" style="--party-color:{party_color}">{party}</span>
      <span>Corpus record {html.escape(display["record_span_label"])}</span>
      <span>{html.escape(display["speech_count_label"])}</span>
      <span class="support {'thin' if thin else ''}">{html.escape(display["support_label"])}</span>
    </div>
  </div>
</header>
{thin_notice}
<nav class="profile-nav" aria-label="On this profile"><span class="profile-nav-cue" aria-hidden="true">More sections →</span><ul>
  <li><a href="#overview">Overview</a></li><li><a href="#agenda">Agenda</a></li>
  <li><a href="#rhetoric">Rhetoric</a></li><li><a href="#connections">Connections</a></li>
  <li><a href="#similarity">Similarity</a></li><li><a href="#speeches">Speeches</a></li>
  <li><a href="#evidence">Evidence</a></li>
</ul></nav>
<main>
<section id="overview" data-profile-section="overview">
  <p class="section-kicker">Identity and record support</p><h2>Overview</h2>
  <p>These at-a-glance facts are mechanically derived from named corpus and annotation artifacts. Corpus years describe the available speeches, not time in office.</p>
  <p class="population-label"><strong>Core v3 profile population:</strong> source documents assigned to this president.</p>
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
{connections_html}
<section id="similarity" data-profile-section="similarity">
  <p class="section-kicker">Five independent instruments</p><h2>Similarity</h2>
  <p>Each card uses only its named feature set and reports cosine scores. The five results are not combined into an overall likeness.</p>
  {_similarity_cards(view)}
</section>
<section id="speeches" data-profile-section="speeches">
  <p class="section-kicker">Chronology kept separate from synthesis</p><h2>Signature speeches</h2>
  <p>Speeches ranked by how strongly their embedding aligns with this president’s era-adjusted distinctive direction. Years and titles come from the Miller Center corpus.</p>
  {_signature_list(view)}
</section>
<section id="evidence" data-profile-section="evidence">
  <h2>Evidence</h2>
  <details class="evidence-section-disclosure">
    <summary>Legacy issue evidence and audit receipts</summary>
    <div class="evidence-section-body">
      <p class="section-kicker">Legacy issue model · named artifacts</p>
      <p class="population-label"><strong>Population:</strong> source documents assigned to this president; rates use document-owner paragraphs.</p>
      <p><span class="source-tag legacy-source">Legacy issue model</span> Cards retain the producer’s declared threshold-normalized strength order. The first two appear first and any remainder stays in the nested closed disclosure.</p>
      {_evidence_cards_from_info(evidence_info)}
    </div>
  </details>
</section>
<p class="download-row"><a href="../data/presidents/{view["slug"]}.json" download>Download this profile’s complete public data →</a></p>
{_president_navigation(view)}
</main>
<footer><p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs, University of Virginia</a>. Emotion scores: NRC Emotion Lexicon (Mohammad &amp; Turney). <a href="../methodology.html">AI labeling method</a> · <a href="https://github.com/jacobfulfyll/presidential_profiles">Code</a>.</p></footer>
{connections_bootstrap}
</body>
</html>
'''


INDEX_CSS = r"""
  .directory-search { display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 8px; max-width: 620px; margin-top: 22px; }
  .directory-search label { grid-column: 1/-1; font-size: .78rem; font-weight: 740; }
  .directory-search input { width: 100%; min-height: 44px; padding: 9px 11px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); color: var(--ink); font: inherit; }
  .directory-search button { min-width: 76px; min-height: 44px; padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); color: var(--ink); font: inherit; font-weight: 720; }
  .directory-results { grid-column: 1/-1; margin: 0; color: var(--muted); font-size: .78rem; }
  .directory-grid { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 11px; margin: 24px 0 0; padding: 0; list-style: none; }
  .directory-grid > li { min-width: 0; }
  .president-card { display: grid; grid-template-columns: 64px minmax(0,1fr); gap: 11px; height: 100%; padding: 13px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); color: var(--ink); text-decoration: none; }
  .president-card:hover { border-color: #726d64; }
  .president-card:focus-visible, .directory-search :focus-visible { outline: 3px solid #315f78; outline-offset: 2px; }
  .president-card img { width: 64px; height: 64px; border: 1px solid var(--border); border-radius: 50%; object-fit: cover; }
  .president-card strong { display: block; line-height: 1.25; }
  .card-meta { margin-top: 3px; color: var(--muted); font-size: .75rem; }
  .card-speeches { grid-column: 1/-1; color: var(--ink2); font-size: .8rem; font-weight: 700; }
  .card-measure { grid-column: 1/-1; color: var(--ink2); font-size: .8rem; }
  .card-measure span { display: block; margin-bottom: 2px; color: var(--muted); font-size: .64rem; font-weight: 760; letter-spacing: .045em; text-transform: uppercase; }
  .card-support { grid-column: 1/-1; font-size: .7rem; font-weight: 720; color: #34523c; }
  .card-support.thin { color: #6b4c0c; }
  .directory-no-results { margin-top: 20px; padding: 13px; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); }
  body > footer a { display: inline-flex; min-width: 44px; min-height: 44px; align-items: center; }
  [hidden] { display: none!important; }
  @media (max-width: 760px) {
    .card-measure span, .card-support { font-size: 12px; }
  }
  @media (max-width: 959px) { .directory-grid { grid-template-columns: repeat(2,minmax(0,1fr)); } }
  @media (max-width: 599px) { .directory-grid { grid-template-columns: 1fr; } }
  @media (forced-colors: active) {
    .president-card, .directory-search input, .directory-search button { border-color: CanvasText; }
    .president-card:focus-visible, .directory-search :focus-visible { outline-color: Highlight; }
  }
"""


def _normalized_display_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def render_index(data: dict, display_issues: list[str]) -> str:
    scores = data["scores"]
    ai_presidents = data.get("ai", {}).get("by_president", {})
    cards = []
    for card_index, president in enumerate(president_chronology(scores.index)):
        score = scores.loc[president]
        ai_info = ai_presidents.get(president)
        ai_badge = (
            '<div class="card-measure unavailable"><span>Top AI topic by source-document paragraph share</span>'
            'Unavailable for this corpus record</div>'
        )
        if ai_info and ai_info.get("top_topics"):
            topic = ai_info["top_topics"][0]
            ai_badge = (
                '<div class="card-measure"><span>Top AI topic by source-document paragraph share</span>'
                + html.escape(str(topic["name"]))
                + f' · {float(topic["share"]):.1f}%</div>'
            )
        n_speeches = int(score["n_speeches"])
        thin = n_speeches < SPARSE_MIN_SPEECHES
        raw_display_name = public_display_name(president)
        display_name = html.escape(raw_display_name)
        normalized_name = html.escape(_normalized_display_name(raw_display_name), quote=True)
        loading = "eager" if card_index < 6 else "lazy"
        cards.append(f'''<li data-president-card data-search-name="{normalized_name}"><a class="president-card" href="{slug(president)}.html">
  <img src="../portraits/{slug(president)}.png" width="64" height="64" alt="" loading="{loading}" decoding="async">
  <div><strong>{display_name}</strong><div class="card-meta">{html.escape(str(score["party"]))}<br>Corpus record {int(score["first_year"])}–{int(score["last_year"])}</div></div>
  <div class="card-speeches">{format_speech_count(n_speeches)}</div>{ai_badge}
  <div class="card-support {'thin' if thin else ''}">{'Insufficient record for percentile ranking' if thin else 'Supported for percentile ranking'}</div>
</a></li>''')
    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>President Profiles</title><style>{PAGE_CSS}{INDEX_CSS}</style></head>
<body><header><h1>President Profiles</h1><p class="sub">Every president in canonical chronology, with one AI-topic preview and explicit corpus support.</p>
<form class="directory-search" role="search" data-directory-search hidden>
  <label for="president-search">Search by president name</label>
  <input id="president-search" type="search" autocomplete="off" spellcheck="false">
  <button type="button" data-directory-clear>Clear</button>
  <p class="directory-results" data-directory-count aria-live="polite"></p>
</form>
<p class="directory-no-results" data-directory-empty hidden>No presidents match that name. Clear the search to restore the full chronology.</p>
<ol class="directory-grid">{''.join(cards)}</ol></header>
<footer><p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs, University of Virginia</a>.</p></footer>
<script>
(() => {{
  const form = document.querySelector("[data-directory-search]");
  const input = document.querySelector("#president-search");
  const clear = document.querySelector("[data-directory-clear]");
  const count = document.querySelector("[data-directory-count]");
  const empty = document.querySelector("[data-directory-empty]");
  const cards = Array.from(document.querySelectorAll("[data-president-card]"));
  if (!form || !input || !clear || !count || !empty || !cards.length) return;
  const normalize = value => value.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase("en-US");
  const apply = () => {{
    const query = normalize(input.value.trim());
    let visible = 0;
    for (const card of cards) {{
      const matches = !query || card.dataset.searchName.includes(query);
      card.hidden = !matches;
      if (matches) visible += 1;
    }}
    count.textContent = `${{visible}} president${{visible === 1 ? "" : "s"}} shown in chronological order.`;
    empty.hidden = visible !== 0;
    clear.disabled = !query;
  }};
  input.addEventListener("input", apply);
  form.addEventListener("submit", event => {{ event.preventDefault(); apply(); }});
  input.addEventListener("keydown", event => {{
    if (event.key !== "Escape") return;
    event.preventDefault();
    input.value = "";
    apply();
  }});
  clear.addEventListener("click", () => {{ input.value = ""; apply(); input.focus(); }});
  apply();
  form.hidden = false;
}})();
</script></body></html>'''


def write_profiles(
    data: dict,
    site_dir,
    *,
    views: dict[str, dict] | None = None,
    context_index: dict | None = None,
    context_shards: dict[str, dict] | None = None,
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
    directory_html = render_index(data, display_issues)
    _require_html_budget(
        directory_html,
        label="president directory HTML",
        raw_max=DIRECTORY_HTML_RAW_MAX,
        gzip_max=DIRECTORY_HTML_GZIP_MAX,
    )
    (president_dir / "index.html").write_text(directory_html, encoding="utf-8")

    index_bytes = None
    module_sizes = None
    if context_index is not None or context_shards:
        if context_index is None or not context_shards:
            raise ValueError("profile context index and shards must be supplied together")
        from .profile_connections_assets import profile_connections_asset_sizes

        index_bytes = (
            json.dumps(
                context_index,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        module_sizes = profile_connections_asset_sizes()
    for president, view in models.items():
        context = (context_shards or {}).get(view["slug"])
        if context_shards is not None and context is None:
            raise ValueError(f"profile context shard is missing for {view['slug']}")
        page = render_profile(
            view,
            context=context,
            context_index=context_index,
        )
        _require_html_budget(
            page,
            label=f"profile HTML for {view['slug']}",
            raw_max=PROFILE_HTML_RAW_MAX,
            gzip_max=PROFILE_HTML_GZIP_MAX,
        )
        if context is not None and index_bytes is not None and module_sizes is not None:
            shard_bytes = (
                json.dumps(
                    context,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            _, index_gzip = _encoded_sizes(index_bytes)
            _, shard_gzip = _encoded_sizes(shard_bytes)
            raw_total = (
                module_sizes["javascript_raw_bytes"]
                + len(index_bytes)
                + len(shard_bytes)
            )
            gzip_total = (
                module_sizes["javascript_gzip_bytes"] + index_gzip + shard_gzip
            )
            if raw_total > FIRST_EXPANSION_RAW_MAX or gzip_total > FIRST_EXPANSION_GZIP_MAX:
                raise ValueError(
                    f"first Connections expansion for {view['slug']} exceeds its "
                    f"byte budget: {raw_total}/{gzip_total} bytes"
                )
        (president_dir / f"{view['slug']}.html").write_text(
            page,
            encoding="utf-8",
        )
        payload = profile_public_payload(view)
        (data_dir / f"{view['slug']}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    print(f"  wrote {len(models)} profile pages + index to docs/presidents/")
    return models


def validate_generated_profile_budgets(site_dir) -> dict[str, int]:
    """Enforce final post-shell profile, directory, and request boundaries."""
    site_dir = Path(site_dir)
    root = site_dir / "presidents"
    directory = root / "index.html"
    if not directory.is_file():
        raise ValueError("generated president directory is missing")
    directory_bytes = directory.read_bytes()
    _require_html_budget(
        directory_bytes,
        label="generated president directory HTML",
        raw_max=DIRECTORY_HTML_RAW_MAX,
        gzip_max=DIRECTORY_HTML_GZIP_MAX,
    )
    pages = sorted(path for path in root.glob("*.html") if path.name != "index.html")
    if len(pages) != 45:
        raise ValueError("generated president profile inventory must contain 45 pages")
    maximum_raw = maximum_gzip = 0
    for path in pages:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        raw_size, gzip_size = _encoded_sizes(payload)
        maximum_raw = max(maximum_raw, raw_size)
        maximum_gzip = max(maximum_gzip, gzip_size)
        _require_html_budget(
            payload,
            label=f"generated profile HTML {path.name}",
            raw_max=PROFILE_HTML_RAW_MAX,
            gzip_max=PROFILE_HTML_GZIP_MAX,
        )
        if "plotly" in text.casefold() or "const FIGS" in text:
            raise ValueError(f"profile page retained a Plotly request/payload: {path.name}")
        if "<script src=" in text or "fetch(" in text:
            raise ValueError(f"profile page retained an initial data/script request: {path.name}")
        images = re.findall(r"<img\b[^>]*\bsrc=", text, flags=re.I)
        if len(images) != 1:
            raise ValueError(f"profile initial image boundary drift: {path.name}")

    asset = site_dir / "assets" / "profile-connections-v1.js"
    index_path = site_dir / "data" / "profile-context" / "index_v1.json"
    shard_paths = sorted(
        (site_dir / "data" / "profile-context" / "presidents").glob("*_v1.json")
    )
    if not asset.is_file() or not index_path.is_file() or len(shard_paths) != 45:
        raise ValueError("generated profile Connections assets are incomplete")
    asset_raw, asset_gzip = _encoded_sizes(asset.read_bytes())
    index_raw, index_gzip = _encoded_sizes(index_path.read_bytes())
    maximum_expansion_raw = maximum_expansion_gzip = 0
    for shard in shard_paths:
        shard_raw, shard_gzip = _encoded_sizes(shard.read_bytes())
        maximum_expansion_raw = max(
            maximum_expansion_raw, asset_raw + index_raw + shard_raw
        )
        maximum_expansion_gzip = max(
            maximum_expansion_gzip, asset_gzip + index_gzip + shard_gzip
        )
    if (
        maximum_expansion_raw > FIRST_EXPANSION_RAW_MAX
        or maximum_expansion_gzip > FIRST_EXPANSION_GZIP_MAX
    ):
        raise ValueError(
            "generated first Connections expansion exceeds its byte budget: "
            f"{maximum_expansion_raw}/{maximum_expansion_gzip} bytes"
        )
    return {
        "profiles": len(pages),
        "maximum_profile_raw_bytes": maximum_raw,
        "maximum_profile_gzip_bytes": maximum_gzip,
        "directory_raw_bytes": len(directory_bytes),
        "directory_gzip_bytes": _encoded_sizes(directory_bytes)[1],
        "maximum_expansion_raw_bytes": maximum_expansion_raw,
        "maximum_expansion_gzip_bytes": maximum_expansion_gzip,
    }
