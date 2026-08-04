"""Render the 45 president profile pages and their index."""

import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .figures import BLUE_RAMP, GRID, INK, INK2, MUTED, PARTY_COLORS, SURFACE
from .html_safety import json_for_script
from .profiles import (
    FEATURE_SIMILARITY,
    MIN_ISSUE_PARAS,
    RADAR_AXES,
    RADAR_VALUE_COLUMNS,
    slug,
)
from .site_style import FONT, PAGE_CSS
from . import metrics, topic_quality

# Issues shown on profiles: the curated taxonomy plus the one discovered
# topic that is a genuine issue (soviet/nuclear/weapons).
DISPLAY_ISSUES = None  # filled from meta at build time
# Column name -> display label for CorEx's discovered topics, sourced from the
# names file (data/topic_display_names.json) instead of being hardcoded here.
# Every read site is `.get(name, name)`, so a topic missing from the file — or a
# missing file entirely — degrades to showing the raw column name rather than
# breaking a page build.
DISCOVERED_LABELS = topic_quality.discovered_labels()


def _chart_controls(metric_name: str, evidence_href: str, evidence_label: str) -> str:
    return (
        metrics.lesson_html(metric_name)
        + '<details class="evidence"><summary>Inspect the evidence</summary>'
        + f'<p><a href="{evidence_href}" download>{html.escape(evidence_label)} →</a></p>'
        + "</details>"
    )


def fig_radar(row: pd.Series) -> go.Figure:
    theta = [label for _, label in RADAR_AXES]
    r = [float(row[f"pct_{key}"]) for key, _ in RADAR_AXES]
    fig = go.Figure(go.Scatterpolar(
        r=r + r[:1], theta=theta + theta[:1], fill="toself",
        line=dict(color=BLUE_RAMP[4], width=2),
        fillcolor="rgba(109, 167, 236, 0.30)",
        hovertemplate="%{theta}: %{r:.0f}th percentile<extra></extra>",
    ))
    fig.update_layout(
        template="simple_white", paper_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK2, size=12),
        polar=dict(
            bgcolor=SURFACE,
            radialaxis=dict(range=[0, 100], tickfont=dict(size=9, color=MUTED),
                            gridcolor=GRID, angle=90, tickangle=90),
            angularaxis=dict(tickfont=dict(size=11.5, color=INK), gridcolor=GRID),
        ),
        showlegend=False, height=380, margin=dict(l=70, r=70, t=40, b=40),
    )
    return fig


def fig_issue_bars(issue_row: pd.Series, display_issues: list[str]) -> go.Figure:
    """Two panels sharing a row per issue: raw share of paragraphs on the
    left, era-relative emphasis on the right. Selecting and ordering by raw
    share answers "what did they talk about?" first; the era panel then says
    which of it was unusual. Ordering by rel (as this once did) meant issues
    that dominated a presidency but matched its era never appeared at all."""
    share = pd.Series(
        {name: float(issue_row[f"share_{name}"]) * 100 for name in display_issues}
    ).sort_values().tail(8)
    rel = [float(issue_row[f"rel_{name}"]) for name in share.index]
    labels = [DISCOVERED_LABELS.get(n, n) for n in share.index]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        subplot_titles=("share of their paragraphs (%)",
                        "vs their era (pp)"),
    )
    fig.add_trace(go.Bar(
        y=labels, x=share.values, orientation="h",
        marker=dict(color=BLUE_RAMP[4]),
        hovertemplate="%{y}: %{x:.1f}% of their paragraphs<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=labels, x=rel, orientation="h",
        marker=dict(color=[BLUE_RAMP[4] if v >= 0 else MUTED for v in rel]),
        hovertemplate="%{y}: %{x:+.1f} pp vs their era<extra></extra>",
    ), row=1, col=2)
    fig.update_layout(
        template="simple_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=12),
        showlegend=False,
        height=380, margin=dict(l=10, r=16, t=48, b=48),
    )
    fig.update_xaxes(gridcolor=GRID, tickfont=dict(color=MUTED, size=10))
    fig.update_xaxes(rangemode="tozero", row=1, col=1)
    fig.update_xaxes(zeroline=True, zerolinecolor=INK2, zerolinewidth=1,
                     row=1, col=2)
    fig.update_yaxes(tickfont=dict(color=INK, size=12))
    for note in fig.layout.annotations:
        note.font = dict(size=10.5, color=INK2)
    return fig


def fig_ai_topic_bars(info: dict) -> go.Figure:
    """AI-labeled topic attention and contemporary-relative emphasis."""
    topics = list(reversed(info["top_topics"][:8]))
    labels = [
        "<br>".join(
            [name[:34], name[34:]]
            if len(name) > 34 and " " not in name[20:35]
            else (
                [name[:name.rfind(" ", 0, 35)], name[name.rfind(" ", 0, 35) + 1:]]
                if len(name) > 35 else [name]
            )
        )
        for name in [t["name"] for t in topics]
    ]
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        subplot_titles=("share of paragraphs (%)", "vs contemporary presidents (pp)"),
    )
    fig.add_trace(go.Bar(
        y=labels, x=[t["share"] for t in topics], orientation="h",
        marker=dict(color=BLUE_RAMP[4]),
        customdata=[t["n"] for t in topics],
        hovertemplate="%{y}: %{x:.1f}%<br>%{customdata} labeled paragraphs<extra>AI topic</extra>",
    ), row=1, col=1)
    rel = [t["rel"] for t in topics]
    fig.add_trace(go.Bar(
        y=labels, x=rel, orientation="h",
        marker=dict(color=[BLUE_RAMP[4] if v >= 0 else MUTED for v in rel]),
        hovertemplate="%{y}: %{x:+.1f} pp<extra>AI topic</extra>",
    ), row=1, col=2)
    fig.update_layout(
        template="simple_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=11), showlegend=False,
        height=500, margin=dict(l=24, r=28, t=56, b=52),
    )
    fig.update_xaxes(gridcolor=GRID, tickfont=dict(color=MUTED, size=10))
    fig.update_xaxes(rangemode="tozero", row=1, col=1)
    fig.update_xaxes(zeroline=True, zerolinecolor=INK2, row=1, col=2)
    fig.update_yaxes(tickfont=dict(color=INK, size=11), automargin=True)
    for note in fig.layout.annotations:
        note.font = dict(size=10.5, color=INK2)
    return fig


def fig_ai_radar(info: dict) -> go.Figure:
    labels = {
        "party_attack": "Partisan attack", "enemy_naming": "Enemy naming",
        "zero_sum": "Zero-sum", "proposal": "Proposal share",
        "values": "Values share", "topic_breadth": "Breadth<br>(effective topics)",
    }
    keys = list(labels)
    percentiles = [info["ai_radar"][key]["percentile"] for key in keys]
    if any(value is None for value in percentiles):
        percentiles = [50 if value is None else value for value in percentiles]
    fig = go.Figure(go.Scatterpolar(
        r=percentiles + percentiles[:1],
        theta=[labels[key] for key in keys] + [labels[keys[0]]],
        customdata=[info["ai_radar"][key]["absolute"] for key in keys] +
                   [info["ai_radar"][keys[0]]["absolute"]],
        fill="toself", line=dict(color=BLUE_RAMP[4], width=2),
        fillcolor="rgba(109,167,236,.28)",
        hovertemplate="%{theta}: %{r:.0f}th percentile<br>absolute %{customdata:.2f}<extra></extra>",
    ))
    fig.update_layout(
        template="simple_white", paper_bgcolor=SURFACE, showlegend=False, height=460,
        font=dict(family=FONT, color=INK2, size=11),
        polar=dict(bgcolor=SURFACE, radialaxis=dict(range=[0, 100], gridcolor=GRID),
                   angularaxis=dict(gridcolor=GRID)),
        margin=dict(l=90, r=90, t=42, b=42),
    )
    return fig


def _ai_radar_key_html(info: dict) -> str:
    labels = [
        ("party_attack", "Partisan attack", "% of paragraphs"),
        ("enemy_naming", "Enemy naming", "% of paragraphs"),
        ("zero_sum", "Zero-sum framing", "% of paragraphs"),
        ("proposal", "Proposal share", "% of paragraphs"),
        ("values", "Values share", "% of paragraphs"),
        ("topic_breadth", "Effective topic breadth", "effective topics"),
    ]
    cards = []
    for key, label, unit in labels:
        item = info["ai_radar"][key]
        percentile = item["percentile"]
        rank = "not ranked (thin record)" if percentile is None else f"{percentile:.0f}th percentile"
        cards.append(
            f'<div class="radar-key-item"><strong>{html.escape(label)}</strong>'
            f'<span>{item["absolute"]:.2f} {unit}</span><small>{rank}</small></div>'
        )
    return '<div class="radar-key" aria-label="Six AI radar measures">' + "".join(cards) + "</div>"


def _ai_profile_html(info: dict) -> str:
    flags = [("Partisan attack", info["flags"]["party_attack"]),
             ("Named adversary", info["flags"]["enemy_naming"]),
             ("Zero-sum framing", info["flags"]["zero_sum"])]
    flag_html = "".join(f"""<div class="ai-metric"><strong>{value:.1f}%</strong>
<span>{label}</span><div class="meter"><i style="width:{min(value, 100):.2f}%"></i></div></div>"""
                            for label, value in flags)
    pv = info["proposal_values"]
    segments = "".join(
        f'<i class="pv-{key}" style="width:{pv[key]:.3f}%" title="{key}: {pv[key]:.1f}%"></i>'
        for key in ("proposal", "mixed", "values", "neither")
    )
    legend = "".join(
        f'<span><i class="pv-dot pv-{key}"></i>{key.replace("_", " ")} {pv[key]:.1f}%</span>'
        for key in ("proposal", "mixed", "values", "neither")
    )
    genres = "".join(
        f'<li><span>{html.escape(g["name"])}</span><strong>{g["value"]:.1f}%</strong></li>'
        for g in info["speech_types"]
    )
    adversaries = "".join(
        f'<span class="term">{html.escape(a["name"])} <small>{a["n"]}×</small></span>'
        for a in info["adversaries"]
    ) or '<span class="dim">No repeated adversarial entity in this record.</span>'
    warning = (f'<p class="i-warn">AI rates are especially provisional here: only '
               f'{info["n_speeches"]} speech{"es" if info["n_speeches"] != 1 else ""} '
               f'and {info["n_paragraphs"]} paragraphs are available.</p>') \
        if info.get("low_confidence") else ""
    return f"""{warning}<div class="ai-grid">
<div class="ai-panel"><h3>Combat framing</h3><p>Share of this president's paragraphs carrying each closed rubric flag.</p>
<div class="ai-metrics">{flag_html}</div></div>
<div class="ai-panel"><h3>Policy ask or values appeal?</h3><p>Every paragraph receives exactly one of four classes.</p>
<div class="pv-bar">{segments}</div><div class="pv-legend">{legend}</div></div>
<div class="ai-panel"><h3>Speech genres</h3><p>Primary AI-classified speech types in this corpus.</p><ul class="genre-list">{genres}</ul></div>
<div class="ai-panel"><h3>Named adversaries</h3><p>Most frequent entities assigned adversarial stance; counts are mentions.</p>
<div class="terms">{adversaries}</div></div></div>"""


def _neighbor_list(pairs: list, fmt: str) -> str:
    return " · ".join(
        f'<a href="{slug(name)}.html">{name}</a> <span class="dim">({fmt.format(v)})</span>'
        for name, v in pairs
    )


def _feature_neighbors_html(president: str, data: dict) -> str:
    categories = data.get("feature_neighbors", {}).get(president, {})
    if not categories:
        return '<p class="dim">Feature-based neighbors are unavailable.</p>'
    rows = []
    for key, meta in FEATURE_SIMILARITY.items():
        neighbors = categories.get(key, [])[:4]
        links = " · ".join(
            f'<a href="{slug(item["president"])}.html">{html.escape(item["president"])}</a> '
            f'<span class="dim">({item["similarity"]:.2f})</span>'
            for item in neighbors
        ) or '<span class="dim">No adequately sampled neighbor.</span>'
        rows.append(f"""<div class="similarity-row">
  <div><strong>{html.escape(meta["label"])}</strong>
  <span>{html.escape(meta["description"])}</span></div>
  <p>{links}</p>
</div>""")
    return "".join(rows)


def _issue_cards_html(president: str, data: dict) -> str:
    info = data["issue_cards"].get(president, {"cards": [], "voice": []})
    cards = []
    for c in info["cards"]:
        issue = html.escape(DISCOVERED_LABELS.get(c["issue"], c["issue"]))
        words = "".join(f'<span class="term">{w}</span>' for w in c["words"])
        words_html = f'<div class="terms">{words}</div>' if c["words"] else ""
        quote_html = ""
        if c["quote"]:
            quote_html = (f'<blockquote><p>“{c["quote"]}”</p>'
                          f'<cite>{c["cite"]}</cite></blockquote>')
        stance = (f'<span class="i-stance">{c["stance"]}</span>'
                  if c.get("stance") else "")
        flag = ('<span class="i-flag">topic of the day</span>'
                if c.get("topic_of_day") else "")
        cards.append(f"""<div class="i-card">
  <div class="i-head"><span class="i-name">{issue}</span>
    <span class="i-badge"><strong>{c["share"] * 100:.0f}%</strong> of their paragraphs</span>
    <span class="i-badge">{c["rel"]:+.0f} pp vs their era</span>{flag}{stance}</div>
  {words_html}
  {quote_html}
</div>""")
    cards_html = "\n".join(cards) if cards else \
        '<p class="dim">No issue rises above the historical base rate or their era.</p>'

    if info.get("low_confidence"):
        n_sp = info["n_speeches"]
        cards_html = (
            f'<p class="i-warn">Thin record: {n_sp} '
            f'{"speech" if n_sp == 1 else "speeches"}, {info["n_paragraphs"]} '
            f'paragraphs. These rates carry wide error and are shown for '
            f'completeness, not precision.</p>'
        ) + cards_html

    voice_html = ""
    if info["voice"]:
        chips = "".join(f'<span class="term">{w}</span>' for w in info["voice"])
        voice_html = f"""<div class="i-voice">
  <div class="i-head"><span class="i-name">Their voice</span>
    <span class="i-badge">distinctive words that belong to no single issue</span></div>
  <div class="terms">{chips}</div>
</div>"""
    return cards_html + voice_html


def render_profile(president: str, data: dict, display_issues: list[str]) -> str:
    scores = data["scores"].loc[president]
    issue_row = data["issues"].loc[president]

    figs = {
        "radar": fig_radar(scores),
        "issues": fig_issue_bars(issue_row, display_issues),
    }
    ai_info = data.get("ai", {}).get("by_president", {}).get(president)
    if ai_info:
        figs["ai_topics"] = fig_ai_topic_bars(ai_info)
        figs["ai_radar"] = fig_ai_radar(ai_info)
    fig_json = {k: json.loads(pio.to_json(f)) for k, f in figs.items()}

    party = scores["party"]
    party_color = PARTY_COLORS.get(party, MUTED)
    chips = (
        f'<span class="chip"><span class="dot" style="background:{party_color}"></span>{party}</span>'
        f'<span class="chip">{int(scores["first_year"])}–{int(scores["last_year"])}</span>'
        f'<span class="chip">{int(scores["n_speeches"])} speeches</span>'
        f'<span class="chip">{int(scores["n_words"]):,} words</span>'
    )

    sigs = "".join(
        f'<li><a href="{s["url"]}" target="_blank" rel="noopener">{s["title"]}</a></li>'
        for s in data["signatures"][president]
    )

    # Tone percentages were tried and retracted: lexicon sentiment cannot
    # hear sarcasm ("because he was a nice guy?"), which dominates modern
    # adversarial mentions. Counts only.
    invokes = data["invokes"].get(president) or []
    invoked_by = data["invoked_by"].get(president)
    invocation_bits = []
    if invokes:
        invocation_bits.append(
            "Invokes: " + ", ".join(f"{m['target']} ({m['n']}×)"
                                    for m in invokes[:4])
        )
    if invoked_by and invoked_by["total"]:
        invocation_bits.append(
            f"Invoked {invoked_by['total']}× by later presidents")
    invocation_html = (
        f'<p class="invocations">{" &nbsp;·&nbsp; ".join(invocation_bits)}</p>'
        if invocation_bits else ""
    )
    classified_invocations = data.get("invocation_v2", {}).get(president, [])
    invocation_v2_html = ""
    if classified_invocations:
        rows = "".join(
            f"<tr><td>{html.escape(row['target'])}</td>"
            f"<td>{html.escape(row['function'])}</td>"
            f"<td>{html.escape(row['stance'])}</td><td>{int(row['mentions'])}</td></tr>"
            for row in classified_invocations[:12])
        invocation_v2_html = f"""<details><summary>AI-classified presidential invocations</summary>
<p class="dim">Exploratory AI labels; not human-validated. Historical lineage is the
default in the shared evidence layer, while contemporary rivalry remains available.</p>
<div class="chart-scroll"><table><thead><tr><th>Target</th><th>Function</th>
<th>Stance</th><th>Mentions</th></tr></thead><tbody>{rows}</tbody></table></div>
<p><a href="../data/networks/invocation_evidence.csv" download>
Download invocation evidence →</a></p></details>"""

    feature_neighbor_html = _feature_neighbors_html(president, data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{president} - Presidential Profiles</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
<style>
{PAGE_CSS}
  .crumbs {{ margin-bottom: 18px; font-size: 0.88rem; }}
  .crumbs a {{ color: var(--ink2); }}
  .id-row {{ display: flex; align-items: center; gap: 20px; }}
  .portrait {{ width: 84px; height: 84px; border-radius: 50%;
               border: 1px solid var(--border); flex: none; }}
  .i-stance {{ background: var(--page); border: 1px solid var(--border);
               border-radius: 999px; padding: 2px 10px; font-size: 0.78rem;
               color: var(--ink2); }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
  .chip {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 999px; padding: 4px 12px; font-size: 0.85rem;
           color: var(--ink2); display: inline-flex; align-items: center; gap: 7px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .duo {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .duo > div {{ min-width: 0; }}
  @media (max-width: 820px) {{ .duo {{ grid-template-columns: 1fr; }} }}
  .duo .chart {{ min-width: 380px; }}
  .terms {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .term {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 8px; padding: 5px 12px; font-size: 0.95rem; }}
  .i-card, .i-voice {{ background: var(--surface); border: 1px solid var(--border);
            border-radius: 12px; padding: 16px 18px; margin-top: 12px; }}
  .i-head {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px;
             margin-bottom: 10px; }}
  .i-name {{ font-weight: 700; font-size: 1.02rem; }}
  .i-badge {{ color: var(--muted); font-size: 0.8rem; }}
  .i-badge strong {{ color: var(--ink2); font-weight: 650; }}
  .i-flag {{ background: var(--page); border: 1px dashed var(--border);
             border-radius: 999px; padding: 2px 10px; font-size: 0.78rem;
             color: var(--muted); }}
  .i-warn {{ color: var(--ink2); font-size: 0.88rem; background: var(--surface);
             border: 1px solid var(--border); border-left: 3px solid var(--muted);
             border-radius: 8px; padding: 10px 14px; margin-top: 12px;
             max-width: none; }}
  .i-card .term, .i-voice .term {{ background: var(--page); }}
  .i-card blockquote {{ border-left: 3px solid var(--grid); margin: 12px 0 0;
                        padding: 2px 0 2px 14px; }}
  .i-card blockquote p {{ color: var(--ink2); font-size: 0.94rem; margin: 0;
                          max-width: none; }}
  .i-card cite {{ display: block; color: var(--muted); font-style: normal;
                  font-size: 0.8rem; margin-top: 6px; }}
  .dim {{ color: var(--muted); }}
  .neighbors p {{ margin: 6px 0; }}
  .similarity-row {{ display:grid;grid-template-columns:minmax(210px,.85fr) minmax(0,1.35fr);
                     gap:18px;padding:13px 0;border-top:1px solid var(--grid); }}
  .similarity-row:first-child {{ border-top:0; }}
  .similarity-row strong,.similarity-row span {{ display:block; }}
  .similarity-row span {{ color:var(--muted);font-size:.78rem;margin-top:3px; }}
  .similarity-row p {{ margin:0;max-width:none;align-self:center; }}
  @media(max-width:700px) {{ .similarity-row {{ grid-template-columns:1fr;gap:5px; }} }}
  ul.sigs {{ margin: 10px 0 0 18px; color: var(--ink2); }}
  ul.sigs li {{ margin: 5px 0; }}
  .invocations {{ color: var(--muted); font-size: 0.9rem; }}
  .method-chip {{ color: #275d8c; font-weight: 650; }}
  .ai-note {{ background:#eef6ff; border:1px solid #c9def3; border-radius:12px;
              padding:13px 16px; color:var(--ink2); font-size:.9rem; max-width:none; }}
  .ai-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:14px; }}
  @media (max-width:760px) {{ .ai-grid {{ grid-template-columns:1fr; }} }}
  .ai-panel {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px; }}
  .ai-panel h3 {{ font-size:1rem; }} .ai-panel p {{ font-size:.83rem; margin:5px 0 12px; }}
  .ai-chart-block {{ margin-top:28px;padding-top:24px;border-top:1px solid var(--grid); }}
  .ai-radar-card {{ max-width:780px;margin:20px auto 0; }}
  .radar-key {{ display:grid;grid-template-columns:repeat(3,minmax(150px,1fr));gap:8px;margin-top:10px; }}
  .radar-key-item {{ background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 12px; }}
  .radar-key-item strong,.radar-key-item span,.radar-key-item small {{ display:block; }}
  .radar-key-item strong {{ font-size:.8rem; }} .radar-key-item span {{ margin-top:3px;font-size:.82rem;color:var(--ink2); }}
  .radar-key-item small {{ color:var(--muted);font-size:.7rem;margin-top:2px; }}
  .ai-metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }}
  .ai-metric strong,.ai-metric span {{ display:block; }} .ai-metric span {{ color:var(--muted); font-size:.72rem; }}
  .meter {{ height:5px; background:var(--page); border-radius:4px; overflow:hidden; margin-top:6px; }}
  .meter i {{ display:block; height:100%; background:{BLUE_RAMP[4]}; }}
  .pv-bar {{ display:flex; height:16px; overflow:hidden; border-radius:8px; background:var(--page); }}
  .pv-bar i {{ display:block; height:100%; }} .pv-proposal {{ background:#275d8c; }}
  .pv-mixed {{ background:#6da7ec; }} .pv-values {{ background:#8bbf9f; }} .pv-neither {{ background:#d3d5d8; }}
  .pv-legend {{ display:flex; flex-wrap:wrap; gap:8px 12px; margin-top:10px; color:var(--muted); font-size:.73rem; }}
  .pv-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; }}
  .genre-list {{ list-style:none; padding:0; }} .genre-list li {{ display:flex; justify-content:space-between; gap:12px;
                  font-size:.85rem; border-top:1px solid var(--grid); padding:5px 0; }}
  .genre-list li:first-child {{ border-top:0; }} .genre-list strong {{ font-variant-numeric:tabular-nums; }}
  .term small {{ color:var(--muted); font-size:.72rem; }}
  @media (max-width:700px) {{ .radar-key {{ grid-template-columns:1fr 1fr; }} }}
</style>
</head>
<body>
<header>
  <p class="crumbs"><a href="index.html">← All presidents</a> &nbsp;·&nbsp;
     <a href="../index.html">Dashboard</a> &nbsp;·&nbsp;
     <a href="../compare.html?a={slug(president)}">Compare</a> &nbsp;·&nbsp;
     <a href="../methodology.html">How the AI labels work →</a></p>
  <div class="id-row">
    <img class="portrait" src="../portraits/{slug(president)}.png" alt="{president}">
    <div>
      <h1>{president}</h1>
      <div class="chips">{chips}</div>
    </div>
  </div>
</header>
<main>
<section>
  <div class="duo">
    <div>
      <h2>Rhetorical fingerprint</h2>
      <p>Percentile among all 45 presidents.</p>
      <div class="chart-scroll"><div class="chart" data-fig="radar" style="height:380px"></div></div>
      {_chart_controls("president_percentile", f"../data/presidents/{slug(president)}.json", "Download profile chart data")}
    </div>
    <div>
      <h2>Issues: attention and emphasis</h2>
      <p>How much of their speech each issue took (left), and how that
         compared with contemporaries (right).</p>
      <div class="chart-scroll"><div class="chart" data-fig="issues" style="height:380px"></div></div>
      {_chart_controls("era_relative", f"../data/presidents/{slug(president)}.json", "Download issue attention and baselines")}
    </div>
  </div>
</section>
{f'''<section id="ai-labels">
  <h2>AI-labeled topic and rhetoric profile</h2>
  <p class="ai-note"><span class="method-chip">Corpus-derived label layer.</span>
  A masked model read every paragraph with only its text and decade—no president, party,
  or title. Topic shares are multi-label and the right panel compares this president with
  presidents whose records begin nearby. <a href="../methodology.html">Definitions, prompts,
  agreement, and limitations →</a></p>
  <div class="ai-radar-card"><h3>Six-part AI radar</h3>
  <p>Percentiles among presidents with at least five speeches. The sixth spoke is
  <strong>effective topic breadth</strong>, which keeps the radar balanced and asks how
  many fine topics received meaningful attention. Absolute values are listed below.</p>
  <div class="chart-scroll"><div class="chart" data-fig="ai_radar" style="height:460px"></div></div>
  {_ai_radar_key_html(ai_info)}
  {_chart_controls("president_percentile", f"../data/presidents/{slug(president)}.json", "Download AI radar values")}</div>
  <div class="ai-chart-block"><h3>Detailed topic attention</h3>
  <p>Eight leading fine topics, given a full-width view so labels and era-relative
  differences remain readable.</p>
  <div class="chart-scroll"><div class="chart" data-fig="ai_topics" style="height:500px"></div></div>
  {_chart_controls("paragraph_share", f"../data/presidents/{slug(president)}.json", "Download detailed and broad topic attention")}</div>
  {_ai_profile_html(ai_info)}
  <p><a href="../data/presidents/{slug(president)}.json" download>Download this profile's complete chart data →</a></p>
</section>''' if ai_info else ''}
<section>
  <h2>Legacy issue lens — in their own words</h2>
  <p>The issues that either defined their agenda or set them apart from their era, each with
     the vocabulary that is statistically <em>theirs</em> on that issue and a verbatim sentence
     from their speeches. <span class="dim">“Topic of the day” marks a subject they gave
     unusual attention by historical standards but no more than their own contemporaries —
     the air everyone was breathing.</span></p>
  {_issue_cards_html(president, data)}
</section>
<section class="neighbors">
  <h2>Five kinds of similarity</h2>
  <p>No hidden “overall likeness” is implied. Each row compares a president in a
  small, named feature space; topic-mix scores use raw shares, while rhetoric
  scores standardize unlike units before cosine similarity.</p>
  {feature_neighbor_html}
{invocation_html}
{invocation_v2_html}
</section>
<section>
  <h2>Signature speeches</h2>
  <p>The speeches that most express what makes this president distinct.
     Links go to the Miller Center.</p>
  <ul class="sigs">{sigs}</ul>
</section>
</main>
<footer>
  <p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs,
  University of Virginia</a>. Emotion scores: NRC Emotion Lexicon (Mohammad &amp; Turney).
  <a href="../methodology.html">AI labeling method</a> ·
  <a href="https://github.com/jacobfulfyll/presidential_profiles">Code</a>.</p>
</footer>
<script>
  const FIGS = {json_for_script(fig_json)};
  for (const el of document.querySelectorAll(".chart")) {{
    const fig = FIGS[el.dataset.fig];
    Plotly.newPlot(el, fig.data, fig.layout, {{displayModeBar: false, responsive: true}});
  }}
</script>
</body>
</html>
"""


def render_index(data: dict, display_issues: list[str]) -> str:
    scores = data["scores"]
    issues_df = data["issues"]
    ai_presidents = data.get("ai", {}).get("by_president", {})
    cards = []
    for president in scores.sort_values("first_year").index:
        s = scores.loc[president]
        n_paras = float(issues_df.loc[president, "n_paragraphs"])
        # Badge only issues backed by at least MIN_ISSUE_PARAS paragraphs, so
        # one-speech presidents don't get a headline from a stray metaphor.
        # Shared with the profile-card floor so the two gates cannot diverge.
        rel = {
            n: float(issues_df.loc[president, f"rel_{n}"])
            for n in display_issues
            if float(issues_df.loc[president, f"share_{n}"]) * n_paras >= MIN_ISSUE_PARAS
        }
        if not rel:
            rel = {n: float(issues_df.loc[president, f"rel_{n}"]) for n in display_issues}
        top_issue = max(rel, key=rel.get)
        top_issue = html.escape(DISCOVERED_LABELS.get(top_issue, top_issue))
        color = PARTY_COLORS.get(s["party"], MUTED)
        ai_info = ai_presidents.get(president)
        ai_badge = (f'<div class="ai-issue"><span>AI topic</span> '
                    f'{html.escape(ai_info["top_topics"][0]["name"])}</div>') \
            if ai_info and ai_info["top_topics"] else ""
        cards.append(f"""<a class="card" href="{slug(president)}.html">
  <div class="name"><span class="dot" style="background:{color}"></span>{president}</div>
  <div class="meta">{int(s["first_year"])}–{int(s["last_year"])} · {int(s["n_speeches"])} speeches</div>
  {ai_badge}<div class="issue"><span>Legacy issue</span> ↑ {top_issue}</div>
</a>""")
    cards_html = "\n".join(cards)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>President Profiles - Presidential Profiles</title>
<style>
{PAGE_CSS}
  .crumbs {{ margin-bottom: 18px; font-size: 0.88rem; }}
  .crumbs a {{ color: var(--ink2); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
           gap: 12px; margin-top: 24px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 10px; padding: 14px 16px; text-decoration: none;
           color: var(--ink); transition: border-color 0.15s; }}
  .card:hover {{ border-color: var(--muted); }}
  .name {{ font-weight: 650; display: flex; align-items: center; gap: 8px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }}
  .meta {{ color: var(--muted); font-size: 0.8rem; margin-top: 4px; }}
  .issue {{ color: var(--ink2); font-size: 0.83rem; margin-top: 8px; }}
  .issue span,.ai-issue span {{ color:var(--muted); font-size:.68rem; text-transform:uppercase;
                               letter-spacing:.05em; margin-right:4px; }}
  .ai-issue {{ color:#275d8c; font-size:.82rem; margin-top:9px; }}
</style>
</head>
<body>
<header>
  <p class="crumbs"><a href="../index.html">← Dashboard</a> &nbsp;·&nbsp;
  <a href="../compare.html">Compare</a> &nbsp;·&nbsp;
  <a href="../methodology.html">How the AI labels work</a></p>
  <h1>President Profiles</h1>
  <p class="sub">Every president's rhetorical fingerprint, AI-labeled topics and
  paragraph judgments, legacy issue emphasis, distinctive vocabulary, and defining
  speeches. Each card shows both measurement layers rather than blending them.</p>
  <div class="grid">
{cards_html}
  </div>
</header>
<footer>
  <p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs,
  University of Virginia</a>. <a href="../methodology.html">AI labeling method</a>.</p>
</footer>
</body>
</html>
"""


def _json_value(value):
    """Convert pandas/numpy values into stable JSON-native profile values."""
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def public_profile_payload(
    president: str, data: dict, display_issues: list[str]
) -> dict:
    """One public contract consumed by both profile and comparison pages."""
    scores = data["scores"].loc[president]
    issue_row = data["issues"].loc[president]
    ai = data.get("ai", {}).get("by_president", {}).get(president)
    vocabulary = (
        data["distinctive"][data["distinctive"].president.eq(president)]
        .sort_values("rank").head(20).to_dict("records")
    )
    issue_attention = [
        {
            "key": issue,
            "label": DISCOVERED_LABELS.get(issue, issue),
            "share": round(float(issue_row[f"share_{issue}"]), 4),
            "era_relative_difference": round(float(issue_row[f"rel_{issue}"]), 4),
        }
        for issue in display_issues
    ]
    n_speeches = int(scores["n_speeches"])
    return _json_value({
        "schema_version": "president-profile-v3",
        "president": president,
        "slug": slug(president),
        "party": scores["party"],
        "years": {
            "first": int(scores["first_year"]),
            "last": int(scores["last_year"]),
        },
        "sample": {
            "n_speeches": n_speeches,
            "n_words": int(scores["n_words"]),
            "thin_record": n_speeches < 5,
            "warning": (
                "Fewer than five corpus speeches: visible, but not ranked as a precise outlier."
                if n_speeches < 5 else None
            ),
        },
        "rhetorical_radar": [
            {
                "key": key, "label": label,
                "absolute": scores[RADAR_VALUE_COLUMNS[key]],
                "percentile": scores[f"pct_{key}"],
            }
            for key, label in RADAR_AXES
        ],
        "raw_stats": scores.to_dict(),
        "legacy_issue_attention": issue_attention,
        "issue_evidence": data["issue_cards"].get(president, {}),
        "ai": ai,
        "distinctive_vocabulary": vocabulary,
        "signature_speeches": data["signatures"].get(president, []),
        "legacy_invocations": {
            "invokes": data["invokes"].get(president, []),
            "invoked_by": data["invoked_by"].get(president),
        },
        "classified_invocations": data.get("invocation_v2", {}).get(president, []),
        "voice_neighbors": data["voice_neighbors"].get(president, []),
        "agenda_neighbors": data["agenda_neighbors"].get(president, []),
        "feature_neighbors": data.get("feature_neighbors", {}).get(president, {}),
        "context_specific": {
            "profile_only": [],
            "compare_only": [],
        },
    })


def write_profiles(data: dict, site_dir) -> None:
    display_issues = topic_quality.display_issues(data["issue_meta"]["issues"])
    pres_dir = site_dir / "presidents"
    pres_dir.mkdir(parents=True, exist_ok=True)
    data_dir = site_dir / "data" / "presidents"
    data_dir.mkdir(parents=True, exist_ok=True)
    (pres_dir / "index.html").write_text(render_index(data, display_issues))
    for president in data["scores"].index:
        html = render_profile(president, data, display_issues)
        (pres_dir / f"{slug(president)}.html").write_text(html)
        payload = public_profile_payload(president, data, display_issues)
        (data_dir / f"{slug(president)}.json").write_text(json.dumps(payload, indent=2))
    print(f"  wrote {len(data['scores'])} profile pages + index to docs/presidents/")
