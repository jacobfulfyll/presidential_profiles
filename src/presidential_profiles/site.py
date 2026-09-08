"""Generate the static dashboard site: every trend as an interactive chart.

Writes docs/index.html with a bundled Plotly runtime, suitable for GitHub Pages.
With --inline, also writes a fully self-contained copy for offline sharing.
"""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re
import textwrap
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from . import (
    actual_speaker_invocation_network,
    ai_labels,
    attention,
    bands,
    combat,
    compare_projection,
    compare_site,
    corpus,
    era_boundaries_site,
    era_contextualizations,
    era_profiles,
    era_visualizations,
    eras,
    expansion_site,
    expansion_story,
    explorer,
    explore_projection,
    indices,
    issues,
    issues_site,
    methodology_site,
    metrics,
    portraits,
    profile_connections_assets,
    profile_context,
    profiles,
    profiles_site,
    rhetoric,
    similarity,
    site_validation,
    speaker_topic_network,
    story_foundation,
    summary_topic_network,
    trends,
    word_families,
)
from . import topic_quality
from .figures import (
    BASELINE,
    BLUE_RAMP,
    GRID,
    INK,
    INK2,
    MUTED,
    PARTY_COLORS,
    REPO_ROOT,
    SERIES,
    SURFACE,
)
from .html_safety import json_for_script
from .site_style import FONT, PAGE_CSS

SITE_DIR = REPO_ROOT / "docs"
PLOTLY_BUNDLE_NAME = "plotly-3.0.1.min.js"


def write_github_pages_marker(site_dir: Path) -> Path:
    """Disable Jekyll processing for the prebuilt static Pages artifact."""
    marker = site_dir / ".nojekyll"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="ascii")
    return marker


def bundle_plotly_runtime(site_dir) -> None:
    """Make every generated route work in an offline/local browser.

    Individual renderers historically pointed at the CDN. The desktop app's
    browser can reach localhost while external scripts are unavailable, which
    otherwise leaves a fully readable page with zero mounted charts.
    """
    from plotly.offline import get_plotlyjs

    asset_dir = site_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / PLOTLY_BUNDLE_NAME).write_text(get_plotlyjs())
    pattern = re.compile(
        r'<script src="https://cdn\.plot\.ly/plotly-3\.0\.1\.min\.js"'
        r'(?: charset="utf-8")?></script>'
    )
    for page in site_dir.rglob("*.html"):
        relative = page.relative_to(site_dir)
        prefix = "../" * (len(relative.parts) - 1)
        text = page.read_text()
        text, _ = pattern.subn(
            f'<script src="{prefix}assets/{PLOTLY_BUNDLE_NAME}"></script>', text
        )
        page.write_text(text)


def _layout(**overrides) -> dict:
    base = dict(
        template="simple_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=56, r=24, t=32, b=44),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        xaxis=dict(gridcolor=GRID, linecolor=BASELINE, tickcolor=MUTED,
                   tickfont=dict(color=MUTED, size=11), zeroline=False),
        yaxis=dict(gridcolor=GRID, linecolor=BASELINE, tickcolor=MUTED,
                   tickfont=dict(color=MUTED, size=11), zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=12, color=INK2)),
        height=430,
    )
    base.update(overrides)
    return base


# Derived, not restated: `issues_site` owns the window (see
# `issues_site.x_range_covering` for why the lower bound is a floor).
X_RANGE = [issues_site.X_MIN, issues_site.X_MAX]


def _stats_yearly(stats: pd.DataFrame, count_col: str, window: int = 5,
                  min_tokens: int = 20_000, raw: bool = False) -> pd.Series:
    """Rolling per-10k rate by year from the spaCy stats table. Raw mode
    returns unsmoothed yearly rates for texture markers."""
    g = stats.groupby("year")
    counts = g[count_col].sum()
    toks = g["n_tokens"].sum()
    years = pd.RangeIndex(int(stats["year"].min()), int(stats["year"].max()) + 1,
                          name="year")
    counts = counts.reindex(years, fill_value=0)
    toks = toks.reindex(years, fill_value=0)
    if raw:
        rate = counts / toks.replace(0, np.nan) * 10_000
        rate[toks < 5_000] = np.nan
        return rate
    csum = counts.rolling(window, center=True, min_periods=1).sum()
    tsum = toks.rolling(window, center=True, min_periods=1).sum()
    rate = csum / tsum * 10_000
    rate[tsum < min_tokens] = np.nan
    return rate


def _timeline_layout(**overrides) -> dict:
    out = _layout(**overrides)
    out["xaxis"] = dict(range=X_RANGE, gridcolor=GRID, linecolor=BASELINE,
                        tickcolor=MUTED, tickfont=dict(color=MUTED, size=11),
                        zeroline=False)
    return out


def kinship_pairs(adj: pd.DataFrame, min_gap: int = 30,
                  min_speeches: int = 5) -> list[tuple]:
    """Top era-adjusted similarity pairs at least min_gap years apart.
    Presidents with a handful of speeches (W. Harrison, Garfield, Taylor)
    are excluded - a few speeches are not a voice."""
    keep = adj["n_speeches"] >= min_speeches
    sub = adj[keep].reset_index(drop=True)
    vec_cols = [c for c in adj.columns if c.startswith("e")]
    V = sub[vec_cols].to_numpy(dtype=float)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    sim = V @ V.T
    names = sub["president"].tolist()
    years = sub["first_year"].to_numpy()

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            gap = abs(int(years[i] - years[j]))
            if gap >= min_gap:
                pairs.append((float(sim[i, j]), names[i], names[j], gap))
    pairs.sort(reverse=True)
    return pairs


def _kinships_html(pairs: list[tuple], faces: dict, top_n: int = 8) -> str:
    scores = np.array([pair[0] for pair in pairs], dtype=float)
    median = float(np.median(scores))
    p95 = float(np.percentile(scores, 95))
    cards = []
    for sim_val, a, b, gap in pairs[:top_n]:
        img_a = f'<img src="{faces[a]}" alt="">' if a in faces else ""
        img_b = f'<img src="{faces[b]}" alt="">' if b in faces else ""
        percentile = float((scores <= sim_val).mean() * 100)
        strength = ("moderate echo" if sim_val >= .45 else
                    "weak-to-moderate echo" if sim_val >= .30 else "weak echo")
        percentile_label = profiles.format_percentile(percentile)
        cards.append(f"""<div class="k-card">
  <div class="k-faces">{img_a}<span class="k-link"></span>{img_b}</div>
  <div class="k-names">{a} ↔ {b}</div>
  <div class="k-meta">similarity {sim_val:.2f} · {strength}</div>
  <div class="k-meta">{percentile_label} of {len(scores):,} eligible cross-era pairs · {gap} years apart</div>
</div>""")
    spectrum = f"""<div class="similarity-context">
  <strong>How to read 0.53</strong>
  <p>The median eligible cross-era pair is {median:.2f}; the 95th percentile is
  {p95:.2f}; the strongest is {scores.max():.2f}. Zero means no directional match
  in the adjusted feature pattern and 1 would mean an identical direction. The
  best observed pair is unusual in this corpus, but still far from identical.</p>
  <div class="similarity-scale" aria-label="Cosine similarity reference scale">
    <span style="left:0%">−1 opposite</span><span style="left:50%">0 unrelated</span>
    <span class="observed" style="left:{(scores.max() + 1) * 50:.1f}%">● observed max {scores.max():.2f}</span>
    <span style="left:100%">1 identical</span>
  </div>
</div>"""
    return spectrum + '<div class="kinships">' + "\n".join(cards) + "</div>"


def _feature_kinships_html(profile_data: dict, faces: dict) -> str:
    """One strongest cross-era pair for each declared similarity instrument."""
    neighbors = profile_data.get("feature_neighbors", {})
    years = profile_data["scores"]["first_year"].astype(int).to_dict()
    dimensions = {
        "ai_topics": 50,
        "ai_domains": 17,
        "ai_rhetoric": 6,
        "rhetorical_fingerprint": 8,
        "legacy_issues": 16,
    }
    cards = []
    for key, meta in profiles.FEATURE_SIMILARITY.items():
        candidates: dict[tuple[str, str], float] = {}
        for president, categories in neighbors.items():
            for item in categories.get(key, []):
                other = item["president"]
                if abs(years.get(president, 0) - years.get(other, 0)) < 30:
                    continue
                pair = tuple(sorted((president, other)))
                candidates[pair] = max(
                    candidates.get(pair, -1.0), float(item["similarity"])
                )
        if not candidates:
            continue
        (a, b), score = max(candidates.items(), key=lambda item: item[1])
        img_a = f'<img src="{faces[a]}" alt="">' if a in faces else ""
        img_b = f'<img src="{faces[b]}" alt="">' if b in faces else ""
        cards.append(f"""<article class="feature-pair" title="{html.escape(meta['description'], quote=True)}">
  <div class="k-faces">{img_a}<span class="k-link"></span>{img_b}</div>
  <div><span class="feature-label">{html.escape(meta["label"])}</span>
  <strong>{html.escape(a)} ↔ {html.escape(b)}</strong>
  <div class="feature-score">{score:.2f}</div>
  <div class="feature-score-track"><span style="width:{max(0, min(100, score * 100)):.0f}%"></span></div>
  <p>{html.escape(meta["description"])}</p></div>
</article>""")
    return f"""<div class="similarity-context">
<strong>There is no single “most similar president.”</strong>
<p>Similarity changes with the question. Topic-mix scores compare non-negative shares;
rhetoric scores first put unlike units on a common scale. The five strongest cross-era
pairs below are therefore five separate findings—not votes in an overall ranking.
Hover or focus a card for its exact definition.</p>
</div><div class="feature-pairs">{''.join(cards)}</div>
<h3 class="radar-title">Inside the closest six-measure AI-rhetoric pair</h3>
<p class="dim">The hexagon shows why a high overall score does not mean two identical
profiles: presidents can align strongly on some axes and diverge on others.</p>
<div class="chart-scroll"><div class="chart" data-fig="kinship_radar" style="height:500px"></div></div>
{metrics.lesson_html("similarity")}
<details class="evidence"><summary>Inspect the evidence</summary><p>The six absolute
values and percentile positions are included in each president profile download.
Only presidents with at least five corpus speeches can be ranked as precise neighbors.</p></details>"""


def fig_kinship_radar(profile_data: dict) -> go.Figure:
    """Six-axis portrait of the strongest eligible AI-rhetoric pairing."""
    neighbors = profile_data["feature_neighbors"]
    years = profile_data["scores"]["first_year"].astype(int).to_dict()
    candidates: dict[tuple[str, str], float] = {}
    for president, categories in neighbors.items():
        for item in categories.get("ai_rhetoric", []):
            other = item["president"]
            if abs(years.get(president, 0) - years.get(other, 0)) < 30:
                continue
            pair = tuple(sorted((president, other)))
            candidates[pair] = max(candidates.get(pair, -1), float(item["similarity"]))
    pair, _ = max(candidates.items(), key=lambda item: item[1])
    axes = [
        ("party_attack", "Partisan attack"), ("enemy_naming", "Enemy naming"),
        ("zero_sum", "Zero-sum framing"), ("proposal", "Proposals"),
        ("values", "Values"), ("topic_breadth", "Topic breadth"),
    ]
    colors = ["#275d8c", "#b16b3e"]
    fig = go.Figure()
    for president, color in zip(pair, colors):
        radar = profile_data["ai"]["by_president"][president]["ai_radar"]
        values = [float(radar[key]["percentile"]) for key, _ in axes]
        absolute = [float(radar[key]["absolute"]) for key, _ in axes]
        hover_data = [
            [absolute_value, profiles.format_percentile(percentile)]
            for absolute_value, percentile in zip(absolute, values)
        ]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=[label for _, label in axes] + [axes[0][1]],
            name=president, mode="lines+markers", fill="toself",
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=7),
            customdata=hover_data + [hover_data[0]],
            hovertemplate=(
                "<b>%{fullData.name}</b><br>%{theta}: %{customdata[1]}"
                "<br>absolute value: %{customdata[0]:.1f}<extra></extra>"
            ),
        ))
    fig.update_layout(**_layout(
        height=500,
        polar=dict(
            bgcolor=SURFACE,
            radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100],
                            ticksuffix="th", gridcolor=GRID),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(size=11, color=INK2)),
        ),
        margin=dict(l=95, r=95, t=72, b=42),
        legend=dict(orientation="h", y=1.08, x=.5, xanchor="center"),
    ))
    return fig


def _era_vocab_html(eras: list[dict]) -> str:
    cards = []
    for e in eras:
        chips = "".join(f'<span class="term">{w}</span>' for w in e["words"])
        cards.append(f"""<div class="e-card">
  <div class="e-head"><span class="e-name">{e["era"]}</span>
    <span class="e-years">{e["years"]}</span></div>
  <div class="terms">{chips}</div>
</div>""")
    return '<div class="eras">' + "\n".join(cards) + "</div>"


def _institution_growth_html() -> str:
    path = corpus.DATA_DIR / "register" / "trends.parquet"
    if not path.exists():
        return '<p class="dim">Registered trend artifact unavailable.</p>'
    table = pd.read_parquet(path)
    rows = table[(table.unit == "era") & (table.statistic == "level") &
                 (table.measure == "effective_topics") &
                 (table.taxonomy == "llm_level2") &
                 (table.genre_treatment == "raw")].sort_values("period")
    if rows.empty:
        return '<p class="dim">Effective-topic era series unavailable.</p>'
    first, civil, last = rows.iloc[0], rows.iloc[min(2, len(rows)-1)], rows.iloc[-1]
    return f"""<div class="chart-scroll"><div class="chart" data-fig="institution_grows"
style="height:430px"></div></div>
<div class="audit-card">
<p><strong>Question.</strong> Did agenda breadth rise steadily, or mostly during the institution's first century?</p>
<p><strong>Finding.</strong> Effective topics rise from {first.value:.1f} in the founding band to
{civil.value:.1f} by the Civil War-era band; the present estimate is {last.value:.1f}.</p>
<p>The dots are 30-year analysis bands, not nine equal chapters of history. The first
band starts with the available 1789 speeches. The steepest expansion is early; later
eras mostly fluctuate around a much broader presidential agenda.</p>
<p><strong>Inspect the evidence.</strong> <a href="data/inference_receipts.csv" download>Trend receipts CSV</a>.</p>
<span class="method-chip">Exploratory · corpus-native 50-topic taxonomy</span></div>"""


def fig_institution_growth() -> go.Figure:
    """Thirty-year effective-topic levels from the registered trend artifact."""
    table = pd.read_parquet(corpus.DATA_DIR / "register" / "trends.parquet")
    rows = table[(table.unit == "era") & (table.statistic == "level") &
                 (table.measure == "effective_topics") &
                 (table.taxonomy == "llm_level2") &
                 (table.genre_treatment == "raw")].sort_values("period")
    labels = [
        "1789–99", "1800–29", "1830–59", "1860–89", "1890–1919",
        "1920–49", "1950–79", "1980–2009", "2010–26",
    ][:len(rows)]
    x = np.arange(len(rows))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=rows["value"], mode="lines+markers",
        line=dict(color="#275d8c", width=3),
        marker=dict(
            size=[14 if i < 3 else 10 for i in x],
            color=["#8b5e34" if i < 3 else "#275d8c" for i in x],
            line=dict(color="white", width=2),
        ),
        customdata=np.stack([
            rows["ci_low"], rows["ci_high"], rows["n_speeches"]
        ], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>%{y:.1f} effective topics"
            "<br>95% interval %{customdata[0]:.1f}–%{customdata[1]:.1f}"
            "<br>%{customdata[2]:.0f} speeches<extra></extra>"
        ),
        text=labels, showlegend=False,
    ))
    fig.add_vrect(x0=-.45, x1=2.45, fillcolor="rgba(139,94,52,.08)",
                  line_width=0, layer="below")
    fig.add_annotation(
        x=1, y=float(rows.iloc[:3]["value"].max()) + .9,
        text="the early expansion", showarrow=False,
        font=dict(size=12, color="#8b5e34"),
    )
    fig.update_layout(**_layout(
        height=430,
        xaxis=dict(tickmode="array", tickvals=x, ticktext=labels,
                   tickangle=-30, gridcolor=GRID, linecolor=BASELINE),
        yaxis=dict(title="effective topics", rangemode="tozero",
                   gridcolor=GRID, linecolor=BASELINE),
        margin=dict(l=58, r=24, t=40, b=76),
    ))
    return fig


def _coverage_pressure_html() -> str:
    path = corpus.DATA_DIR / "coverage_pressure" / "coverage_pressure.json"
    if not path.exists():
        return '<p class="dim">Coverage-pressure artifact unavailable.</p>'
    payload = json.loads(path.read_text())
    primary = [row for row in payload["receipts"] if row.get("status") == "confirmatory"]
    cards = []
    for row in primary:
        direction = "+" if row["estimate"] >= 0 else ""
        cards.append(f"""<div class="r-card"><div class="r-label">{html.escape(row['metric'].replace('_',' '))}</div>
<div class="r-value">{direction}{row['estimate']:.2f} <span class="r-unit">{html.escape(row['unit'])}</span></div>
<p>{html.escape(row['surprise_label'])}; Holm-adjusted {row['p_adjusted']*100:.3g}%.</p></div>""")
    return f"""<p><strong>Question.</strong> Do modern annual messages cover more topics but sustain them for fewer words?</p>
<div class="records">{''.join(cards)}</div>
<div class="coverage-toy" aria-label="Illustration of breadth versus depth">
  <div class="coverage-row"><strong>Earlier message</strong>
    <div class="coverage-ribbons">
      <span style="flex:36;background:#356b92">Finance · 360 words</span>
      <span style="flex:33;background:#7f9f84">Diplomacy · 330</span>
      <span style="flex:31;background:#8d78a8">Administration · 310</span>
    </div><small>Fewer subjects, held for longer runs</small></div>
  <div class="coverage-row"><strong>Modern message</strong>
    <div class="coverage-ribbons">
      <span style="flex:16;background:#356b92">Economy</span>
      <span style="flex:15;background:#77a6c8">Health</span>
      <span style="flex:18;background:#d29b61">Security</span>
      <span style="flex:15;background:#7f9f84">Education</span>
      <span style="flex:17;background:#8d78a8">Climate</span>
      <span style="flex:19;background:#bb6b6b">Immigration</span>
    </div><small>More subjects, held for shorter runs</small></div>
</div>
<p class="toy-note">Toy illustration only: colors name topics and widths show consecutive
equivalent words. The registered estimates above come from the speeches, not this example.</p>
<p><strong>Joint decision:</strong> {html.escape(payload['decision'])}. Sensitivity arms and secondary outcomes
are published even when they weaken the headline.</p>
<p><a href="data/coverage_pressure/coverage_pressure.json" download>Chart data</a> ·
<a href="data/coverage_pressure/inference_receipts.csv" download>Statistical receipts</a>.</p>
<span class="method-chip">Preregistered · 20,000-draw president→speech bootstrap</span>"""


def _content_stage_html(
    labels: list[str], announcements: list[str], panels: list[str],
) -> str:
    assert len(labels) == len(announcements) == len(panels)
    buttons = "".join(
        f'<button type="button" data-stage-button="{index}" '
        f'aria-pressed="{"true" if index == 0 else "false"}">'
        f"{index + 1}. {html.escape(label)}</button>"
        for index, label in enumerate(labels)
    )
    rendered_panels = "".join(
        f'<div class="stage-panel" data-stage-panel="{index}"'
        f'{" hidden" if index else ""}>{panel}</div>'
        for index, panel in enumerate(panels)
    )
    steps = "".join(
        f'<div class="stage-step" data-stage-step="{index}" tabindex="0">'
        f'<strong>{index + 1}. {html.escape(label)}</strong>'
        f"<span>{html.escape(announcement)}</span></div>"
        for index, (label, announcement) in enumerate(zip(labels, announcements))
    )
    return f"""<div class="story-stage content-stage" data-stage-mode="content"
 data-stage-announcements="{html.escape(json.dumps(announcements), quote=True)}">
  <div class="stage-controls" role="group" aria-label="Evidence stages">{buttons}</div>
  <p class="stage-status" aria-live="polite">{html.escape(announcements[0])}</p>
  <div class="stage-panels">{rendered_panels}</div>
  <div class="stage-narrative" aria-label="Scroll stages">{steps}</div>
</div>"""


def _topic_parent_map() -> dict[str, str]:
    taxonomy = json.loads(
        (corpus.DATA_DIR / "llm_annotations" / "taxonomy_v1.json").read_text()
    )
    return {
        item["name"].casefold(): item["level1"]
        for item in taxonomy["level2"]
    }


def _message_strip_html(
    doc_name: str,
    speeches: pd.DataFrame,
    coverage: pd.DataFrame,
    paragraphs: pd.DataFrame,
    annotations: pd.DataFrame,
    parent_map: dict[str, str],
) -> str:
    speech = speeches.loc[speeches.doc_name.eq(doc_name)].iloc[0]
    metric = coverage.loc[coverage.doc_name.eq(doc_name)].iloc[0]
    rows = (
        paragraphs.loc[paragraphs.doc_name.eq(doc_name)]
        .merge(
            annotations.loc[annotations.doc_name.eq(doc_name), [
                "doc_name", "para_idx", "topics",
            ]],
            on=["doc_name", "para_idx"], validate="one_to_one",
        )
        .sort_values("para_idx")
    )
    domains = []
    for topics in rows.topics:
        first = topics[0] if isinstance(topics, (list, np.ndarray)) and len(topics) else ""
        domains.append(parent_map.get(str(first).casefold(), "Unlabeled / other"))
    rows["domain"] = domains
    domain_order = sorted(set(parent_map.values()) | {"Unlabeled / other"})
    domain_index = {name: index % 10 for index, name in enumerate(domain_order)}
    segments = "".join(
        f'<span class="topic-c{domain_index[row.domain]}" '
        f'style="flex-grow:{max(int(row.word_count), 1)}" '
        f'title="Paragraph {int(row.para_idx) + 1}: {html.escape(row.domain)} · '
        f'{int(row.word_count)} words" '
        f'aria-label="Paragraph {int(row.para_idx) + 1}: {html.escape(row.domain)}, '
        f'{int(row.word_count)} words"></span>'
        for row in rows.itertuples()
    )
    weights = (
        rows.groupby("domain").word_count.sum().sort_values(ascending=False).head(6)
    )
    legend = "".join(
        f'<span><i class="topic-c{domain_index[name]}"></i>{html.escape(name)}</span>'
        for name in weights.index
    )
    substantial = rows[rows.word_count.ge(40)]
    excerpt = str((substantial.iloc[0] if len(substantial) else rows.iloc[0]).text)
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if len(excerpt) > 360:
        excerpt = excerpt[:357].rsplit(" ", 1)[0] + "…"
    url = "https://millercenter.org" + doc_name
    return f"""<article class="message-evidence">
  <p class="message-kicker">{int(speech.year)} · {html.escape(speech.president)}</p>
  <h3>{html.escape(speech.title)}</h3>
  <div class="topic-strip" role="img"
       aria-label="{len(rows)} real paragraphs, colored by their first assigned broad topic">{segments}</div>
  <div class="topic-legend">{legend}</div>
  <div class="message-measures"><span><b>{int(metric.word_count):,}</b> speech words</span>
    <span><b>{metric.effective_topics:.1f}</b> effective topics</span>
    <span><b>{metric.depth_words:.0f}</b> equivalent words per topic episode</span>
    <span><b>{int(metric.n_paragraphs)}</b> paragraphs</span></div>
  <blockquote><p>“{html.escape(excerpt)}”</p></blockquote>
  <a href="{url}" target="_blank" rel="noopener">Read the source speech →</a>
</article>"""


def _coverage_pressure_v2_html(
    speeches: pd.DataFrame,
    paragraph_annotations: pd.DataFrame | None = None,
) -> str:
    coverage_path = corpus.DATA_DIR / "coverage_pressure" / "coverage_pressure.parquet"
    receipts_path = corpus.DATA_DIR / "coverage_pressure" / "inference_receipts.parquet"
    if not coverage_path.exists() or not receipts_path.exists():
        return '<p class="dim">Registered coverage-pressure artifacts unavailable.</p>'
    coverage = pd.read_parquet(coverage_path)
    receipts = pd.read_parquet(receipts_path)
    paragraphs = pd.read_parquet(corpus.DATA_DIR / "paragraphs.parquet")
    annotations = (
        ai_labels.load_paragraph_annotations("paragraph_annotations")
        if paragraph_annotations is None
        else paragraph_annotations
    )
    parent_map = _topic_parent_map()
    earlier_doc = (
        "/the-presidency/presidential-speeches/"
        "december-3-1894-second-annual-message-second-term"
    )
    modern_doc = (
        "/the-presidency/presidential-speeches/"
        "january-12-2016-2016-state-union-address"
    )
    primary = coverage[
        coverage.is_annual
        & coverage.n_paragraphs.ge(12)
        & coverage.group.isin(["postbellum", "modern"])
    ]
    descriptive = primary.groupby("group").agg(
        n_speeches=("doc_name", "size"),
        median_words=("word_count", "median"),
        mean_breadth=("effective_topics", "mean"),
        mean_depth=("depth_words", "mean"),
    )
    post = descriptive.loc["postbellum"]
    modern = descriptive.loc["modern"]
    descriptive_summary = f"""<div class="tradeoff-summary" aria-label="Descriptive primary-sample comparison">
<article><span>Median speech length</span><strong>{post.median_words:,.0f} → {modern.median_words:,.0f}</strong><small>words · descriptive</small></article>
<article><span>Mean effective breadth</span><strong>{post.mean_breadth:.2f} → {modern.mean_breadth:.2f}</strong><small>topics · descriptive</small></article>
<article><span>Mean topic depth</span><strong>{post.mean_depth:.0f} → {modern.mean_depth:.0f}</strong><small>equivalent words · descriptive</small></article>
</div><p class="toy-note">These are unadjusted descriptions of the declared {int(post.n_speeches + modern.n_speeches)}-speech primary sample. The registered estimates and uncertainty appear in the next stages.</p>"""
    examples = (
        descriptive_summary
        + '<div class="message-compare">'
        + _message_strip_html(
            earlier_doc, speeches, coverage, paragraphs, annotations, parent_map
        )
        + _message_strip_html(
            modern_doc, speeches, coverage, paragraphs, annotations, parent_map
        )
        + "</div><p class=\"toy-note\">Every colored segment is a real paragraph. "
          "Its width is that paragraph's word count; its color and pattern show the "
          "first assigned broad topic. These two speeches make the mechanism concrete; "
          "the registered result below uses the declared 56-speech comparison. The "
          "tradeoff is compression and breadth versus sustained treatment; neither side "
          "is automatically better.</p>"
    )
    confirm = receipts[receipts.status.eq("confirmatory")].set_index("metric")
    breadth = confirm.loc["effective_topics"]
    depth = confirm.loc["depth_words"]
    breadth_panel = f"""<div class="receipt-focus">
  <p class="receipt-label">Registered breadth difference · modern minus postbellum</p>
  <p class="receipt-number">+{breadth.estimate:.2f} <small>effective topics</small></p>
  <p>95% interval {breadth.interval_low:.2f} to {breadth.interval_high:.2f}.
  {html.escape(breadth.surprise_label)}; Holm-adjusted {breadth.p_adjusted * 100:.2f}%.</p>
  <p>Plain language: the modern annual messages in the declared comparison cover more
  effective topics, even though the interval still reaches slightly below zero.</p>
</div>"""
    depth_panel = f"""<div class="receipt-grid">
<div class="receipt-focus"><p class="receipt-label">Registered depth difference</p>
  <p class="receipt-number">{depth.estimate:.0f} <small>equivalent words</small></p>
  <p>95% interval {depth.interval_low:.0f} to {depth.interval_high:.0f}.
  {html.escape(depth.surprise_label)}; Holm-adjusted {depth.p_adjusted * 100:.3g}%.</p></div>
<div class="receipt-focus"><p class="receipt-label">Joint registered decision</p>
  <p class="receipt-number">Supported</p>
  <p>Modern annual messages cover more effective topics while sustaining an individual
  topic for fewer equivalent words. This is coverage pressure, not evidence of shallower
  policy or less expertise.</p></div></div>
<p><a href="data/coverage_pressure/inference_receipts.csv" download>Download every statistical receipt →</a></p>"""
    sensitivity = receipts[
        receipts.status.eq("exploratory")
        & receipts.metric.isin(["effective_topics", "depth_words"])
    ].copy()
    sensitivity_rows = "".join(
        f"<tr><td>{html.escape(row.treatment)}</td>"
        f"<td>{html.escape(row.metric.replace('_', ' '))}</td>"
        f"<td>{row.estimate:+.2f}</td>"
        f"<td>{row.interval_low:.2f} to {row.interval_high:.2f}</td></tr>"
        for row in sensitivity.itertuples()
    )
    sensitivity_panel = f"""<div class="sensitivity-panel">
<p><strong>Sensitivity does not mean optional.</strong> All declared arms are shown,
including controls for length, paragraph size, and the assigned primary medium.</p>
<div class="table-scroll"><table><thead><tr><th>Treatment</th><th>Measure</th>
<th>Estimate</th><th>95% interval</th></tr></thead><tbody>{sensitivity_rows}</tbody></table></div>
<p class="toy-note">These arms are exploratory. Their estimates test whether the result
depends on genre mix, speech length, paragraph construction, or the corpus's assigned medium.</p>
</div>"""
    staged = _content_stage_html(
        ["Real messages", "Breadth", "Depth + receipts", "Sensitivity"],
        [
            "Two real annual messages are shown as topic-colored paragraph strips.",
            "The registered breadth estimate is now visible.",
            "Depth, uncertainty, no-change-surprise receipts, and the joint decision are now visible.",
            "All declared sensitivity treatments are now visible.",
        ],
        [examples, breadth_panel, depth_panel, sensitivity_panel],
    )
    return (
        staged
        + _chart_evidence(
            "effective_topics", "coverage_pressure/coverage_pressure.parquet",
            support="56 annual messages with at least 12 paragraphs in the registered comparison.",
            caveat="The real-speech comparison is illustrative; confirmatory and exploratory statuses are labeled separately.",
        )
        + _chart_evidence(
            "depth_words", "coverage_pressure/inference_receipts.parquet",
            support="20,000-draw hierarchical president-then-speech bootstrap.",
            caveat="Breadth is not automatically good and depth is not automatically bad.",
        )
    )


def _standing_html(
    founding_data: dict | None = None,
    markers: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
    speeches: pd.DataFrame | None = None,
    rates: pd.DataFrame | None = None,
    stats: pd.DataFrame | None = None,
    summary_data: dict | None = None,
    combat_ratios: pd.DataFrame | None = None,
    era_similarity: pd.DataFrame | None = None,
    topic_network_html: str = "",
) -> str:
    long_run_inputs = (scores, speeches, rates)
    if any(value is not None for value in long_run_inputs) and not all(
        value is not None for value in long_run_inputs
    ):
        raise ValueError(
            "synthesis long-run views require scores, speeches, and rates"
        )
    long_run = (
        f"""<div class="synthesis-throughline">
{_procedural_era_html(scores, rates)}
</div>"""
        if all(value is not None for value in long_run_inputs)
        else ""
    )
    weather_table = _weather_table_html(markers) if markers is not None else ""
    if summary_data is None:
        communication_fallback = (
            _story_communication_constellation_html(founding_data)
            if founding_data is not None else ""
        )
        comparison_fallback = (
            f"""{long_run}
<div class="synthesis-throughline">
<h3>The national naming shift</h3>
{_chart_block("naming_progressive", 430, "rate_10k")}
{_naming_explanation_html(speeches, rates)}
</div>
{_chart_block("weather_map", 600, "hype_doom")}
{weather_table}"""
            if all(value is not None for value in long_run_inputs)
            else ""
        )
        return communication_fallback + comparison_fallback
    communication_detail = _summary_communication_audit_html(founding_data)
    language_inputs = (stats, combat_ratios, era_similarity)
    if any(value is None for value in language_inputs):
        raise ValueError(
            "summary language views require stats, combat ratios, and era similarity"
        )
    return f"""<div class="summary-thesis" role="note">
  <span>America, in aggregate</span>
  <p>The formal presidential record becomes more public-facing, more performed, less
  procedural, and more openly partisan. The category of enemy changes; presidents sell
  both tomorrow and yesterday; hope and doom move through the same historical shocks.
  These are changes in prepared presidential speech—not a single national personality.</p>
</div>
<nav class="summary-chapter-route" aria-label="Summary argument">
  <a href="#summary-performance">Congress → public</a><a href="#summary-enemy">Enemies by president</a>
  <a href="#summary-time">Tomorrow + yesterday</a><a href="#summary-emotion">Hope ÷ doom</a>
  <a href="#summary-topics">Topics by president</a>
</nav>

<section class="summary-chapter" id="summary-performance" aria-labelledby="summary-performance-title">
  <p class="summary-chapter-no">01 · Voice</p>
  <h2 id="summary-performance-title">Who is addressed changes. How the address is delivered changes.</h2>
  <p class="summary-lede"><strong>Who:</strong> Congress gives way to the general public.
  <strong>How:</strong> written messages give way to spoken, broadcast, and other performed forms.
  Read together across the same nine eras, the separate bars show a Congress-facing written
  record becoming more public and performed; the bridge below carries that chronology into the
  register view without treating audience or delivery as its cause.</p>
  <div class="summary-communication-bars">
    <section class="summary-communication-panel summary-communication-who"
             aria-labelledby="summary-audience-title">
      <header>
        <div><p>WHO · assigned audience</p>
        <h3 id="summary-audience-title">Who presidents addressed</h3></div>
        <strong>🏛️ Congress <span aria-hidden="true">→</span> 👥 General public</strong>
      </header>
      <div class="summary-communication-key" aria-label="WHO legend">
        <span style="--key-color:#315f78">🏛️ Congress</span>
        <span style="--key-color:#b16b3e">👥 General public</span>
        <span style="--key-color:#c9bda9">🤝 Specific groups</span>
        <span style="--key-color:#74695f">🌐 Other audiences</span>
      </div>
      {_chart_block(
          "summary_audience", 520, "assigned_audience",
          source="speech_annotations.parquet",
          measure_label="Assigned audience · percent of era speeches",
          support="Every corpus speech in each of nine Story eras; every bar sums to 100%.",
          caveat="One assigned audience cannot inventory circulation, rebroadcasting, reception, or off-corpus audiences.",
      )}
    </section>
    <section class="summary-communication-panel summary-communication-how"
             aria-labelledby="summary-medium-title">
      <header>
        <div><p>HOW · assigned delivery form</p>
        <h3 id="summary-medium-title">How presidents delivered the address</h3></div>
        <strong>✉️ Written <span aria-hidden="true">→</span> 🎙️ Spoken + 📻 broadcast</strong>
      </header>
      <div class="summary-communication-key" aria-label="HOW legend">
        <span style="--key-color:#5f568c">✉️ Written</span>
        <span style="--key-color:#4f7557">🎙️ Spoken</span>
        <span style="--key-color:#c58a3a">📻 Radio / TV</span>
        <span style="--key-color:#a24f52">🗣️ Press / debate</span>
      </div>
      {_chart_block(
          "summary_medium", 520, "primary_medium",
          source="speech_annotations.parquet",
          measure_label="Assigned delivery form · percent of era speeches",
          support="Every corpus speech in each of nine Story eras; every bar sums to 100%.",
          caveat="The assigned primary form does not inventory later broadcasts, clips, transcripts, or reception.",
      )}
    </section>
  </div>
  <details class="summary-evidence-detail"><summary>Inspect the full audience/medium categories and agreement audit</summary>
    {communication_detail}
  </details>
  {long_run}
</section>

<section class="summary-chapter" id="summary-enemy" aria-labelledby="summary-enemy-title">
  <p class="summary-chapter-no">02 · Conflict</p>
  <h2 id="summary-enemy-title">Who presidents cast as enemies—and how they frame conflict</h2>
  <p class="summary-lede">Each aligned row keeps two questions separate:
  <strong>who or what is named</strong>, and <strong>how often conflict is framed</strong> through
  enemy naming, zero-sum language, or partisan attack.</p>
  {_summary_conflict_graphs_html(
      summary_data["conflict_targets"], summary_data["conflict_presidents"]
  )}
</section>

<section class="summary-chapter" id="summary-time" aria-labelledby="summary-time-title">
  <p class="summary-chapter-no">03 · Time</p>
  <h2 id="summary-time-title">Presidents can sell tomorrow and yesterday at the same time</h2>
  <p class="summary-lede">Compare the same two independent word families in two views: each
  president’s all-corpus record, or supported four-year rolling averages. A speech can invoke
  tomorrow and yesterday at the same time; this is not one scale running from past to future.</p>
  {_summary_temporal_portrait_html(summary_data["temporal_presidents"])}
  {_summary_language_change_html(
      speeches, rates, stats, combat_ratios, era_similarity
  )}
</section>

<section class="summary-chapter" id="summary-emotion" aria-labelledby="summary-emotion-title">
  <p class="summary-chapter-no">04 · Emotional register</p>
  <h2 id="summary-emotion-title">Hope divided by doom</h2>
  <p class="summary-lede">This is Hope/Doom—not Hype/Doom. It reports broad NRC hope matches
  for each narrow doom-family match in supported centered five-year windows. The log scale keeps
  low and high ratios legible; event lines orient the timeline, and tooltips retain both rates.</p>
  {_chart_block(
      "summary_hope_doom_ratio", 570, "hope_doom_ratio",
      source="speech_markers.parquet",
      measure_label="Aggregate NRC hope count divided by aggregate doom-family count",
      support="Centered five-year windows with at least 20,000 corpus words and a nonzero doom count.",
      caveat="Hope is a much broader dictionary than doom, so the ratio is not emotional balance or parity. Event lines do not identify causes.",
  )}
  {_event_callouts(SUMMARY_RATIO_EVENTS)}
</section>

{topic_network_html}"""


SUMMARY_REGISTER_PERIODS = (1770, 1800, 1830, 1860, 1890, 1920, 1950, 1980, 2010)
SUMMARY_REGISTER_LABELS = (
    "1770–99", "1800–29", "1830–59", "1860–89", "1890–1919",
    "1920–49", "1950–79", "1980–2009", "2010–26",
)
SUMMARY_ERA_SHORT = (
    "Founding", "Continental", "Civil War", "Admin-industrial",
    "Reform/collapse", "New Deal/WWII", "Cold War", "Always-on", "Present",
)
SUMMARY_ERA_COLORS = (
    "#315f78", "#4c718a", "#9b4e50", "#8b6c42", "#a17847",
    "#6d5a91", "#4f7557", "#b16b3e", "#7b3f55",
)
SUMMARY_ERA_PRESETS = (
    ("Early republic", (0, 1)),
    ("Civil War", (2,)),
    ("Administrative", (3, 4)),
    ("Mass media", (5, 6)),
    ("Always-on", (7,)),
    ("Platform", (8,)),
)
SUMMARY_ADVERSARY_ERA_LABELS = (
    "The founding", "Expansion", "Civil War & Reconstruction",
    "The Gilded Age", "Progressives & Depression", "War & New Deal",
    "The Cold War", "Post-Cold War", "The present era",
)
SUMMARY_CONFLICT_ERA_SHORT = (
    "Founding", "Expansion", "Civil War", "Gilded Age", "Progressives",
    "War/New Deal", "Cold War", "Post-Cold War", "Present",
)
SUMMARY_RATIO_EVENTS = [
    (1812, "War of 1812"),
    (1861, "Civil War begins"),
    (1929, "Market crash"),
    (1941, "U.S. enters World War II"),
    (2001, "September 11 attacks"),
    (2020, "COVID-19 pandemic"),
]
SUMMARY_REGISTER_MOMENTS = [
    (
        1827, "Adams-heavy rise", "1825–1829 centered window", 227, 8,
        "John Quincy Adams dominates this pool: seven of ten records, 183 of "
        "227 legal/procedural matches, and only four of eight hype matches. "
        "The rise comes mainly from the sparse hype denominator.",
    ),
    (
        1863, "Civil War-era drop", "1861–1865 centered window", 164, 28,
        "Compared with 1855–1859, the legal/procedural rate falls 53% while "
        "the hype rate rises 182%. Longer annual, veto, and treaty records in "
        "the earlier pool give way to more, shorter wartime and transition records.",
    ),
    (
        1881, "Highest administrative window", "1879–1883 centered window", 829, 11,
        "Veto, appropriations, and long annual-message records dominate this "
        "19-speech pool, producing the highest supported ratio in the series.",
    ),
    (
        1944, "Wartime low", "1942–1946 centered window", 51, 64,
        "The 23-record pool mixes Roosevelt and Truman wartime, conference, and "
        "transition speeches; it is not a single-event effect.",
    ),
    (
        1966, "1960s policy-heavy rise", "1964–1968 centered window", 591, 98,
        "Policy-heavy Johnson records—including the "
        "1968 State of the Union and nuclear non-proliferation treaty remarks—"
        "lift the numerator.",
    ),
    (
        2016, "Mid-2010s decline", "2014–2018 centered window", 92, 236,
        "Obama's nine records contribute 41 legal/procedural and 36 hype matches; "
        "Trump's fourteen contribute 51 and 200. The hype denominator—not a "
        "comparable procedural increase—drives the fall.",
    ),
    (
        2023, "Recent rebound", "2021–2025 centered window", 171, 286,
        "Biden's 28 records contribute 148 legal/procedural and 132 hype matches; "
        "four joint-session or State of the Union addresses supply 96 of his "
        "148 legal/procedural matches. The rise also reflects the much larger "
        "2020 block leaving the rolling window.",
    ),
]
SUMMARY_REGISTER_HOVER_SUMMARIES = {
    1827: "Adams dominates this pool; sparse hype drives the rise.",
    1863: "The procedural rate falls as the pool shifts toward shorter wartime records.",
    1881: "Long annual, veto, and appropriations records produce the series high.",
    1944: "The wartime-transition pool has more hype than legal/procedural matches.",
    1966: "Policy-heavy Johnson records lift the legal/procedural side.",
    2016: "Trump's much larger hype count drives the ratio down.",
    2023: "Biden's major addresses lift procedural language as the large 2020 block leaves.",
}

SUMMARY_CONFLICT_CATEGORY_SPECS = (
    ("adv_n_nation", "nation", "🌍", "Nation", "#315f78"),
    ("adv_n_group", "group", "👥", "Group", "#4f7557"),
    ("adv_n_person", "person", "👤", "Person", "#a24f52"),
    ("adv_n_institution", "institution", "🏛️", "Institution", "#9a7133"),
    ("adv_n_other", "other", "•", "Other", "#74695f"),
)
SUMMARY_CONFLICT_RATE_SPECS = (
    ("enemy_naming", "n_enemy_naming", "Enemy naming", "#9b4e50"),
    ("zero_sum", "n_zero_sum", "Zero-sum framing", "#6d5a91"),
    ("party_attack", "n_party_attack", "Partisan attack", "#315f78"),
)
SUMMARY_CONFLICT_CATEGORY_LINE_STYLES = {
    "nation": ("solid", "circle"),
    "group": ("dash", "square"),
    "person": ("dot", "diamond"),
    "institution": ("dashdot", "triangle-up"),
    "other": ("longdash", "pentagon"),
}
SUMMARY_CONFLICT_PORTRAIT_TIE_COLOR = "#3f3b37"
SUMMARY_CONFLICT_TURNING_POINT_SPECS = (
    (1, "institution", "Institution spike", -10, -52),
    (2, "group", "Domestic turn", 38, -42),
    (4, "group", "Group peak", -28, -46),
    (5, "nation", "Wartime nations", 24, 38),
    (7, "group", "Group share", -28, -46),
    (8, "person", "Broader mix", -42, -44),
)
SUMMARY_CONFLICT_CATEGORY_GUIDE = (
    (
        "Nation",
        "Countries, governments, or national actors, such as the Soviet Union, Mexico, or Iran.",
    ),
    (
        "Group",
        "Collective actors, factions, movements, or militant organizations, such as Democrats, Communists, or al Qaeda.",
    ),
    (
        "Person",
        "Individuals or clearly referenced individuals, such as Saddam Hussein or “the Governor.”",
    ),
    (
        "Institution",
        "Formal governing, legal, economic, or media bodies, such as Congress, the Supreme Court, or the United Nations.",
    ),
    (
        "Other",
        "Residual ideologies, laws or policies, places, and abstractions, such as communism, the Taft–Hartley bill, or the Iron Curtain.",
    ),
)
SUMMARY_CONFLICT_PORTRAIT_FIGURE_KEY = "summary_conflict_frame_portraits"
SUMMARY_CONFLICT_TARGET_TREATMENT = "speaker_audited_all"
SUMMARY_CONFLICT_PORTRAIT_TREATMENT = "annual_message_strict"
SUMMARY_CONFLICT_FIGURE_KEYS = (
    "summary_conflict_targets",
    SUMMARY_CONFLICT_PORTRAIT_FIGURE_KEY,
)
SUMMARY_CONFLICT_COMPOSITION_MIN = 30
SUMMARY_CONFLICT_PORTRAIT_ZERO_PX = 16.0
SUMMARY_CONFLICT_PORTRAIT_MAX_PX = 64.0
SUMMARY_CONFLICT_PORTRAIT_BORDER_PX = 7.0
SUMMARY_TEMPORAL_PORTRAIT_PX = 52.0
SUMMARY_TEMPORAL_CHART_WIDTH_PX = 700.0
SUMMARY_TEMPORAL_CHART_HEIGHT_PX = 650.0
SUMMARY_TEMPORAL_PLOT_WIDTH_PX = SUMMARY_TEMPORAL_CHART_WIDTH_PX - 78.0 - 28.0
SUMMARY_TEMPORAL_PLOT_HEIGHT_PX = SUMMARY_TEMPORAL_CHART_HEIGHT_PX - 70.0 - 66.0
SUMMARY_TEMPORAL_SUPPORT_MIN_SPEECHES = indices.PRESIDENT_PERCENTILE_MIN_SPEECHES
SUMMARY_TEMPORAL_TIMELINE_YEARS = 4
SUMMARY_TEMPORAL_TIMELINE_MIN_WORDS = 10_000
SUMMARY_CONFLICT_DISPLAY_ORDER = (
    "George Washington", "John Adams", "Thomas Jefferson", "James Madison",
    "James Monroe", "John Quincy Adams", "Andrew Jackson", "Martin Van Buren",
    "William Harrison", "John Tyler", "James K. Polk", "Zachary Taylor",
    "Millard Fillmore", "Franklin Pierce", "James Buchanan", "Abraham Lincoln",
    "Andrew Johnson", "Ulysses S. Grant", "Rutherford B. Hayes",
    "James A. Garfield", "Chester A. Arthur", "Grover Cleveland",
    "Benjamin Harrison", "William McKinley", "Theodore Roosevelt",
    "William Taft", "Woodrow Wilson", "Warren G. Harding", "Calvin Coolidge",
    "Herbert Hoover", "Franklin D. Roosevelt", "Harry S. Truman",
    "Dwight D. Eisenhower", "John F. Kennedy", "Lyndon B. Johnson",
    "Richard M. Nixon", "Gerald Ford", "Jimmy Carter", "Ronald Reagan",
    "George H. W. Bush", "Bill Clinton", "George W. Bush", "Barack Obama",
    "Joe Biden", "Donald Trump",
)


def _summary_conflict_dominant_adversary(row) -> tuple[str, str, str]:
    """Return the honest label and ring color for one president portrait."""
    candidates = [
        (key, label, color, int(getattr(row, column)))
        for column, key, _emoji, label, color
        in SUMMARY_CONFLICT_CATEGORY_SPECS
    ]
    maximum = max(count for _key, _label, _color, count in candidates)
    if maximum <= 0:
        return (
            "none",
            "No adversarial entity type recorded",
            SUMMARY_CONFLICT_PORTRAIT_TIE_COLOR,
        )
    leaders = [
        (key, label, color)
        for key, label, color, count in candidates
        if count == maximum
    ]
    if len(leaders) > 1:
        labels = " + ".join(label for _key, label, _color in leaders)
        return (
            "tie",
            f"Tie · {labels}",
            SUMMARY_CONFLICT_PORTRAIT_TIE_COLOR,
        )
    key, label, color = leaders[0]
    return key, label, color


def build_summary_conflict_contract(
    frame: pd.DataFrame,
    *,
    expected_presidents: int | None = 45,
) -> dict:
    """Validate the replaceable president-level payload used by Conflict UI.

    Producers can replace the rows without changing the renderer, provided they
    preserve this explicit contract and supply treatment metadata that accurately
    describes speaker scope and provenance.
    """
    required = {
        "president", "era_first", "year_first", "year_last", "n_speeches",
        "n_paragraphs", "enemy_naming", "zero_sum", "party_attack",
        *(column for column, *_ in SUMMARY_CONFLICT_CATEGORY_SPECS),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "summary conflict president payload is missing columns: "
            + ", ".join(sorted(missing))
        )
    rows = frame.copy().reset_index(drop=True)
    if expected_presidents is not None and len(rows) != expected_presidents:
        raise ValueError(
            f"summary conflict payload needs {expected_presidents} presidents; "
            f"got {len(rows)}"
        )
    if rows["president"].isna().any() or rows["president"].duplicated().any():
        raise ValueError("summary conflict payload needs one named row per president")
    if expected_presidents == len(SUMMARY_CONFLICT_DISPLAY_ORDER):
        expected_names = set(SUMMARY_CONFLICT_DISPLAY_ORDER)
        actual_names = set(rows["president"].astype(str))
        if actual_names != expected_names:
            missing_names = sorted(expected_names - actual_names)
            extra_names = sorted(actual_names - expected_names)
            raise ValueError(
                "summary conflict president membership changed; "
                f"missing={missing_names}, extra={extra_names}"
            )
    if "display_order" not in rows:
        display_order = {
            president: index
            for index, president in enumerate(SUMMARY_CONFLICT_DISPLAY_ORDER)
        }
        rows["display_order"] = rows["president"].map(display_order)
    if rows["display_order"].isna().any() or not np.allclose(
        rows["display_order"], np.round(rows["display_order"])
    ):
        raise ValueError("summary conflict display_order must be integral and complete")
    rows["display_order"] = rows["display_order"].astype(int)
    if rows["display_order"].duplicated().any() or set(rows["display_order"]) != set(
        range(len(rows))
    ):
        raise ValueError("summary conflict display_order must cover every row exactly once")
    support_columns = ["n_speeches", "n_paragraphs"]
    category_columns = [column for column, *_ in SUMMARY_CONFLICT_CATEGORY_SPECS]
    numeric_columns = support_columns + category_columns + ["year_first", "year_last"]
    if rows[numeric_columns].isna().any().any():
        raise ValueError("summary conflict payload contains null counts or year bounds")
    if (rows[support_columns + category_columns] < 0).any().any():
        raise ValueError("summary conflict payload contains negative counts")
    if (rows["year_first"] > rows["year_last"]).any():
        raise ValueError("summary conflict payload has inverted year bounds")
    for column in support_columns + category_columns:
        if not np.allclose(rows[column], np.round(rows[column])):
            raise ValueError(f"summary conflict count {column!r} is not integral")
        rows[column] = rows[column].astype(int)
    rate_columns = [column for column, *_ in SUMMARY_CONFLICT_RATE_SPECS]
    rate_values = rows[rate_columns].to_numpy(dtype=float)
    if np.isinf(rate_values).any():
        raise ValueError("summary conflict payload contains infinite rates")
    if (((rows[rate_columns] < 0) | (rows[rate_columns] > 1))
            & rows[rate_columns].notna()).any().any():
        raise ValueError("summary conflict rates must stay within zero and one")
    no_paragraph_support = rows["n_paragraphs"].eq(0)
    if rows.loc[no_paragraph_support, rate_columns].notna().any().any():
        raise ValueError("summary conflict rows without paragraphs cannot carry rates")
    rows["n_adversarial_entities"] = rows[category_columns].sum(axis=1)
    if rows.loc[no_paragraph_support, "n_adversarial_entities"].gt(0).any():
        raise ValueError(
            "summary conflict rows without paragraphs cannot carry adversarial entities"
        )
    for column, key, *_ in SUMMARY_CONFLICT_CATEGORY_SPECS:
        rows[f"adv_share_{key}"] = np.where(
            rows["n_adversarial_entities"].gt(0),
            rows[column] / rows["n_adversarial_entities"],
            np.nan,
        )
    share_columns = [
        f"adv_share_{key}" for _, key, *_ in SUMMARY_CONFLICT_CATEGORY_SPECS
    ]
    has_adversarial_entities = rows["n_adversarial_entities"].gt(0)
    if not np.allclose(
        rows.loc[has_adversarial_entities, share_columns].sum(axis=1), 1
    ):
        raise ValueError("summary conflict category shares must sum to one")
    for rate_column, count_column, *_ in SUMMARY_CONFLICT_RATE_SPECS:
        if count_column not in rows:
            rows[count_column] = pd.Series(
                np.rint(rows[rate_column] * rows["n_paragraphs"]),
                index=rows.index,
            ).astype("Int64")
        supplied_counts = rows[count_column].dropna()
        if not np.allclose(supplied_counts, np.round(supplied_counts)):
            raise ValueError(f"summary conflict count {count_column!r} is not integral")
        rows[count_column] = rows[count_column].astype("Int64")
        missing_counts = rows[rate_column].notna() & rows[count_column].isna()
        allowed_zero_without_support = (
            no_paragraph_support
            & rows[rate_column].isna()
            & rows[count_column].eq(0)
        )
        orphan_counts = (
            rows[rate_column].isna()
            & rows[count_column].notna()
            & ~allowed_zero_without_support
        )
        invalid_counts = rows[count_column].notna() & (
            (rows[count_column] < 0)
            | (rows[count_column] > rows["n_paragraphs"])
        )
        expected_counts = np.rint(rows[rate_column] * rows["n_paragraphs"])
        inconsistent_counts = rows[rate_column].notna() & (
            rows[count_column].astype("Float64").sub(expected_counts).abs() > .5
        )
        if (
            missing_counts.any()
            or orphan_counts.any()
            or invalid_counts.any()
            or inconsistent_counts.any()
        ):
            raise ValueError(f"summary conflict count {count_column!r} is invalid")
    metadata_defaults = {
        "schema_version": "president-conflict-v1",
        "treatment": "document_owner_all_paragraphs",
        "treatment_label": "Document-owned corpus · speaker attribution audit pending",
        "speaker_scope_status": "pending_speaker_audit",
        "source_label": "combat/by_president.parquet",
    }
    metadata = {}
    for column, fallback in metadata_defaults.items():
        if column not in rows:
            rows[column] = fallback
        values = tuple(rows[column].dropna().astype(str).unique())
        if len(values) != 1:
            raise ValueError(
                f"summary conflict metadata {column!r} must have one value"
            )
        metadata[column] = values[0]
    expected_support_status = pd.Series(np.select(
        [
            rows["n_paragraphs"].eq(0),
            (rows["n_speeches"] < 5) | (rows["n_paragraphs"] < 100),
        ],
        ["not_available", "thin_record"],
        default="observed",
    ), index=rows.index)
    if "support_status" not in rows:
        rows["support_status"] = expected_support_status
    allowed_support = {"observed", "thin_record", "not_available"}
    unknown_support = set(rows["support_status"].astype(str)) - allowed_support
    if unknown_support:
        raise ValueError(
            "summary conflict payload has unknown support states: "
            + ", ".join(sorted(unknown_support))
        )
    if rows["support_status"].astype(str).ne(expected_support_status).any():
        raise ValueError(
            "summary conflict support_status contradicts its support counts"
        )
    expected_composition_status = pd.Series(np.select(
        [
            rows["n_adversarial_entities"].eq(0),
            rows["n_adversarial_entities"].lt(SUMMARY_CONFLICT_COMPOSITION_MIN),
        ],
        ["not_available", "thin_record"],
        default="observed",
    ), index=rows.index)
    if "composition_status" not in rows:
        rows["composition_status"] = expected_composition_status
    allowed_composition = {"observed", "thin_record", "not_available"}
    unknown_composition = (
        set(rows["composition_status"].astype(str)) - allowed_composition
    )
    if unknown_composition:
        raise ValueError(
            "summary conflict payload has unknown composition states: "
            + ", ".join(sorted(unknown_composition))
        )
    if rows["composition_status"].astype(str).ne(
        expected_composition_status
    ).any():
        raise ValueError(
            "summary conflict composition_status contradicts its mention counts"
        )
    era_map = dict(zip(SUMMARY_ADVERSARY_ERA_LABELS, SUMMARY_ERA_SHORT))
    color_map = dict(zip(SUMMARY_ADVERSARY_ERA_LABELS, SUMMARY_ERA_COLORS))
    unknown_eras = set(rows["era_first"].astype(str)) - set(era_map)
    if unknown_eras:
        raise ValueError(
            "summary conflict payload has unknown eras: "
            + ", ".join(sorted(unknown_eras))
        )
    rows["era_short"] = rows["era_first"].map(era_map)
    rows["era_color"] = rows["era_first"].map(color_map)
    maximum_rate = float(rows[rate_columns].max().max())
    if np.isnan(maximum_rate):
        maximum_rate = 0.0
    rate_axis_max = max(0.1, min(1.0, np.ceil(maximum_rate * 10) / 10))
    return {"rows": rows, "rate_axis_max": rate_axis_max, **metadata}


def build_summary_conflict_target_contract(frame: pd.DataFrame) -> dict:
    """Validate the speaker-audited era-grain target-composition payload."""
    count_columns = [
        column for column, *_ in SUMMARY_CONFLICT_CATEGORY_SPECS
    ]
    support_columns = [
        "n_calendar_years", "n_eligible_years", "n_active_years",
        "n_speeches", "n_adversarial_speeches", "n_paragraphs",
    ]
    metadata_defaults = {
        "schema_version": "conflict-target-mix-v1",
        "treatment": "speaker_audited_all",
        "treatment_label": "Speaker-audited all eligible paragraphs",
        "speaker_scope_status": "speaker_audit_complete",
        "source_label": "combat/target_mix_by_era_speaker_audited_v1.parquet",
        "era_scheme": "trends.ERAS",
        "year_basis": "canonical_source_speech_year",
    }
    required = {
        "era", "era_order", "era_start", "era_end", "corpus_end_date",
        *support_columns, *count_columns, *metadata_defaults,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "summary target-mix payload is missing columns: "
            + ", ".join(sorted(missing))
        )
    rows = frame.copy().reset_index(drop=True)
    if len(rows) != len(trends.ERAS) or rows["era"].isna().any():
        raise ValueError("summary target-mix payload needs all nine named eras")
    if rows["era"].duplicated().any() or rows["era_order"].duplicated().any():
        raise ValueError("summary target-mix payload needs one unique row per era")

    numeric_columns = ["era_order", "era_start", "era_end", *support_columns, *count_columns]
    if rows[numeric_columns].isna().any().any():
        raise ValueError("summary target-mix payload contains null counts or bounds")
    if (rows[support_columns + count_columns] < 0).any().any():
        raise ValueError("summary target-mix payload contains negative counts")
    for column in numeric_columns:
        if not np.allclose(rows[column], np.round(rows[column])):
            raise ValueError(f"summary target-mix field {column!r} is not integral")
        rows[column] = rows[column].astype(int)
    rows = rows.sort_values("era_order", kind="stable").reset_index(drop=True)
    expected_axis = tuple(
        (order, label, start, end)
        for order, (label, start, end) in enumerate(trends.ERAS)
    )
    actual_axis = tuple(
        rows[["era_order", "era", "era_start", "era_end"]]
        .itertuples(index=False, name=None)
    )
    if actual_axis != expected_axis:
        raise ValueError("summary target-mix era axis differs from trends.ERAS")
    expected_years = rows["era_end"] - rows["era_start"] + 1
    if rows["n_calendar_years"].ne(expected_years).any():
        raise ValueError("summary target-mix calendar-year support contradicts its bounds")
    if (rows["n_eligible_years"] > rows["n_calendar_years"]).any():
        raise ValueError("summary target-mix eligible years exceed era bounds")
    if (rows["n_active_years"] > rows["n_eligible_years"]).any():
        raise ValueError("summary target-mix active years exceed eligible years")
    if (rows["n_adversarial_speeches"] > rows["n_speeches"]).any():
        raise ValueError("summary target-mix contributing speeches exceed eligible speeches")

    rows["n_adversarial_entities"] = rows[count_columns].sum(axis=1)
    if rows.loc[rows["n_paragraphs"].eq(0), "n_adversarial_entities"].gt(0).any():
        raise ValueError("summary target-mix rows without paragraphs cannot carry mentions")
    if rows.loc[
        rows["n_adversarial_speeches"].eq(0), "n_adversarial_entities"
    ].gt(0).any():
        raise ValueError("summary target-mix rows without contributing speeches carry mentions")
    if rows.loc[
        rows["n_adversarial_entities"].eq(0),
        ["n_adversarial_speeches", "n_active_years"],
    ].gt(0).any().any():
        raise ValueError("summary target-mix zero-mention support is inconsistent")
    if rows.loc[
        rows["n_adversarial_entities"].gt(0),
        ["n_adversarial_speeches", "n_active_years"],
    ].eq(0).any().any():
        raise ValueError("summary target-mix observed mentions lack support")
    for column, key, *_ in SUMMARY_CONFLICT_CATEGORY_SPECS:
        rows[f"adv_share_{key}"] = np.where(
            rows["n_adversarial_entities"].gt(0),
            rows[column] / rows["n_adversarial_entities"],
            np.nan,
        )
    share_columns = [
        f"adv_share_{key}" for _, key, *_ in SUMMARY_CONFLICT_CATEGORY_SPECS
    ]
    supported = rows["n_adversarial_entities"].gt(0)
    if not np.allclose(rows.loc[supported, share_columns].sum(axis=1), 1):
        raise ValueError("summary target-mix category shares must sum to one")
    if not rows.loc[~supported, share_columns].isna().all().all():
        raise ValueError("summary target-mix zero-support shares must remain unavailable")
    rows["composition_status"] = np.select(
        [
            rows["n_adversarial_entities"].eq(0),
            rows["n_adversarial_entities"].lt(SUMMARY_CONFLICT_COMPOSITION_MIN),
        ],
        ["not_available", "thin_record"],
        default="observed",
    )

    metadata = {}
    for column, expected in metadata_defaults.items():
        values = tuple(rows[column].dropna().astype(str).unique())
        if values != (expected,):
            raise ValueError(
                f"summary target-mix metadata {column!r} must be {expected!r}"
            )
        metadata[column] = expected
    end_dates = tuple(rows["corpus_end_date"].dropna().astype(str).unique())
    if len(end_dates) != 1:
        raise ValueError("summary target-mix needs one corpus end date")
    try:
        corpus_end_date = pd.Timestamp(end_dates[0])
    except (TypeError, ValueError) as error:
        raise ValueError("summary target-mix corpus end date is invalid") from error
    if corpus_end_date.year != int(rows.iloc[-1]["era_end"]):
        raise ValueError("summary target-mix corpus end date contradicts the final era")

    short_map = dict(zip(SUMMARY_ADVERSARY_ERA_LABELS, SUMMARY_CONFLICT_ERA_SHORT))
    rows["era_short"] = rows["era"].map(short_map)
    maximum_share = float(rows[share_columns].max().max() * 100)
    if np.isnan(maximum_share):
        maximum_share = 0.0
    target_axis_max = max(
        60.0,
        min(100.0, float(np.ceil(maximum_share / 10.0) * 10.0)),
    )
    return {
        "rows": rows,
        "target_axis_max": target_axis_max,
        "corpus_end_date": corpus_end_date,
        **metadata,
    }


def _summary_conflict_president_row_html(row, rate_axis_max: float) -> str:
    president = str(row.president)
    president_slug = profiles.slug(president)
    total_entities = int(row.n_adversarial_entities)
    segments = []
    if total_entities:
        for count_column, key, emoji, label, color in SUMMARY_CONFLICT_CATEGORY_SPECS:
            count = int(getattr(row, count_column))
            share = float(getattr(row, f"adv_share_{key}")) * 100
            visible = (
                f'<span aria-hidden="true">{emoji}<b>{share:.0f}%</b></span>'
                if share >= 15 else ""
            )
            segments.append(
                '<span class="conflict-mix-segment" '
                + ('tabindex="0" ' if count else 'tabindex="-1" ')
                + f'style="--mix-share:{share:.6f}%;--mix-color:{color}" '
                f'title="{html.escape(label, quote=True)} · {share:.1f}% · '
                f'{count} of {total_entities} adversarial mentions" '
                f'aria-label="{html.escape(label, quote=True)}: {share:.1f} percent, '
                f'{count} of {total_entities} adversarial mentions">{visible}</span>'
            )
        mix_html = '<div class="conflict-mix-bar">' + "".join(segments) + '</div>'
    else:
        mix_html = (
            '<div class="conflict-na" title="No adversarial entities are available '
            'for this president treatment">N/A · no adversarial mentions</div>'
        )
    rates = []
    for rate_column, count_column, label, color in SUMMARY_CONFLICT_RATE_SPECS:
        short_label = {
            "Enemy naming": "Enemy",
            "Zero-sum framing": "Zero-sum",
            "Partisan attack": "Partisan",
        }.get(label, label)
        raw_rate = getattr(row, rate_column)
        if pd.isna(raw_rate):
            rates.append(
                '<div class="conflict-rate is-unavailable" '
                f'title="{html.escape(label, quote=True)} · unavailable in this treatment">'
                f'<span>{html.escape(short_label)}</span><strong>N/A</strong>'
                '<div class="conflict-rate-track" aria-hidden="true"></div></div>'
            )
            continue
        rate = float(raw_rate)
        percent = rate * 100
        width = min(100.0, rate / rate_axis_max * 100)
        count = int(getattr(row, count_column))
        rates.append(
            '<div class="conflict-rate" '
            f'title="{html.escape(label, quote=True)} · {percent:.1f}% · '
            f'{count} of {int(row.n_paragraphs)} paragraphs">'
            f'<span>{html.escape(short_label)}</span><strong>{percent:.1f}%</strong>'
            '<div class="conflict-rate-track" aria-hidden="true">'
            f'<i style="--rate-width:{width:.6f}%;--rate-color:{color}"></i>'
            '</div></div>'
        )
    support_badges = {
        "thin_record": "Thin record",
        "not_available": "No eligible record",
    }
    support_label = support_badges.get(str(row.support_status))
    thin_badge = (
        f'<span class="conflict-support-warning">{support_label}</span>'
        if support_label else ""
    )
    return (
        '<article class="conflict-president-row" data-conflict-president="'
        + html.escape(president, quote=True) + '">'
        '<header class="conflict-president-identity">'
        f'<a href="presidents/{president_slug}.html">'
        f'<img src="portraits/{president_slug}.png" alt="" loading="lazy" '
        f'width="30" height="30"><span><strong>{html.escape(president)}</strong>'
        f'<small>{int(row.year_first)}–{int(row.year_last)}</small></span></a>'
        f'<p title="{int(row.n_speeches)} speeches · {int(row.n_paragraphs):,} paragraphs · '
        f'{total_entities:,} adversarial mentions">{int(row.n_speeches)} sp · '
        f'{int(row.n_paragraphs):,} ¶ · {total_entities:,} adv {thin_badge}</p></header>'
        '<div class="conflict-mix-cell"><span class="conflict-mobile-label">'
        'Who or what is named?</span>' + mix_html + '</div>'
        '<div class="conflict-rate-cell"><span class="conflict-mobile-label">'
        'How often is conflict framed?</span><div class="conflict-rate-grid">'
        + "".join(rates) + '</div></div></article>'
    )


def _summary_conflict_atlas_html(contract: dict) -> str:
    rows = contract["rows"]
    rate_axis_max = float(contract["rate_axis_max"])
    category_key = "".join(
        '<span style="--conflict-color:' + color + '">'
        + emoji + " " + html.escape(label) + "</span>"
        for _, _, emoji, label, color in SUMMARY_CONFLICT_CATEGORY_SPECS
    )
    era_sections = []
    for era_name in rows["era_first"].drop_duplicates():
        era_rows = rows[rows["era_first"].eq(era_name)]
        era_short = str(era_rows.iloc[0]["era_short"])
        era_color = str(era_rows.iloc[0]["era_color"])
        president_rows = "".join(
            _summary_conflict_president_row_html(row, rate_axis_max)
            for row in era_rows.itertuples(index=False)
        )
        era_sections.append(
            '<details class="conflict-era" open style="--era-color:'
            + era_color + '"><summary><span>' + html.escape(era_short)
            + '</span><small>' + html.escape(str(era_name)) + '</small><strong>'
            + str(len(era_rows)) + (' president' if len(era_rows) == 1 else ' presidents')
            + '</strong></summary>'
            + president_rows + '</details>'
        )
    category_evidence = _chart_evidence(
        "enemy_category_share", str(contract["source_label"]),
        support=(
            "All president rows with at least one extracted adversarial entity; "
            "each category strip sums to 100%."
        ),
        caveat=(
            "Current rows inherit document ownership, so mixed-speaker transcripts can "
            "affect both category counts and rates until the speaker audit replaces this payload."
        ),
    )
    framing_evidence = _chart_evidence(
        "enemy_identity", str(contract["source_label"]),
        support=(
            "Enemy naming, zero-sum framing, and partisan attack divided by every keyed "
            "paragraph in the selected president treatment."
        ),
        caveat=(
            "The labels describe textual framing, not whether a target was legitimate or "
            "whether every paragraph in the provisional treatment was spoken by the named president."
        ),
    )
    return f"""<div class="conflict-atlas"
     data-conflict-schema="{html.escape(str(contract['schema_version']), quote=True)}">
  <header class="conflict-atlas-heading">
    <div><p>{len(rows)} PRESIDENTS · TWO LINKED MEASURES</p>
    <h3>President conflict atlas</h3></div>
    <div class="conflict-treatment" role="note"><span>Current input</span>
    <strong>{html.escape(str(contract['treatment_label']))}</strong></div>
  </header>
  <p class="conflict-atlas-deck">One aligned row per president. Strips divide adversarial mentions
  by entity type; meters show the share of corpus paragraphs carrying each conflict frame.
  <span>sp = speeches · ¶ = paragraphs · adv = adversarial mentions</span></p>
  <div class="conflict-category-key" aria-label="Adversary category legend">{category_key}</div>
  <div class="conflict-column-head" aria-hidden="true"><span>President + support</span>
    <span>WHO OR WHAT · each strip totals 100%</span>
    <span>HOW OFTEN · common 0–{rate_axis_max * 100:.0f}% scale</span></div>
  <div class="conflict-era-list">{"".join(era_sections)}</div>
</div>
<div class="conflict-evidence-grid">{category_evidence}{framing_evidence}</div>
<details class="summary-evidence-detail">
<summary>How this atlas will accept speaker-audited data</summary>
<p>The layout reads a validated, versioned president payload rather than copied percentages or
president-specific corrections. A later audit can replace the category counts, paragraph rates,
treatment label, support status, and missing-data states through the producer; rebuilding the
site updates every row without redesigning the atlas.</p></details>"""


def _summary_conflict_ordered_rows(contract: dict) -> pd.DataFrame:
    """Return the complete, stable president chronology shared by every chart."""
    return contract["rows"].sort_values("display_order", kind="stable").reset_index(
        drop=True
    )


def fig_summary_conflict_targets(contract: dict) -> go.Figure:
    """Show all adversarial target categories on one shared era trajectory."""
    rows = contract["rows"].sort_values("era_order", kind="stable")
    x_values = rows["era_order"].to_numpy(dtype=int)
    total_entities = rows["n_adversarial_entities"].to_numpy(dtype=int)
    tick_text = [
        f"{row.era_short}<br>{int(row.era_start)}–{int(row.era_end)}"
        for row in rows.itertuples(index=False)
    ]
    fig = go.Figure()
    target_axis_max = float(contract["target_axis_max"])
    for (
        count_column, key, _emoji, label, color
    ) in SUMMARY_CONFLICT_CATEGORY_SPECS:
        share = rows[f"adv_share_{key}"].mul(100)
        line_dash, marker_symbol = SUMMARY_CONFLICT_CATEGORY_LINE_STYLES[key]
        support_notes = [
            (
                f"<br>Thin target record (&lt;{SUMMARY_CONFLICT_COMPOSITION_MIN} mentions)"
                if status == "thin_record"
                else "<br>Target mix unavailable"
                if status == "not_available"
                else ""
            )
            for status in rows["composition_status"].astype(str)
        ]
        marker_symbols = [
            marker_symbol + "-open" if status == "thin_record" else marker_symbol
            for status in rows["composition_status"].astype(str)
        ]
        fig.add_trace(go.Scatter(
            x=x_values,
            y=share,
            mode="lines+markers",
            connectgaps=False,
            line=dict(color=color, width=2.5, dash=line_dash),
            marker=dict(
                color=color,
                size=[
                    10 if status == "thin_record" else 8
                    for status in rows["composition_status"].astype(str)
                ],
                symbol=marker_symbols,
                line=dict(color=color, width=1.4),
            ),
            cliponaxis=False,
            name=label,
            meta=key,
            customdata=list(zip(
                rows["era"].astype(str),
                rows["era_start"].astype(int),
                rows["era_end"].astype(int),
                rows[count_column].astype(int),
                total_entities,
                rows["n_adversarial_speeches"].astype(int),
                rows["n_speeches"].astype(int),
                rows["n_paragraphs"].astype(int),
                support_notes,
            )),
            hovertemplate=(
                "%{y:.1f}% · %{customdata[3]:,} of %{customdata[4]:,} mentions"
                "%{customdata[8]}<extra>%{fullData.name}</extra>"
            ),
            showlegend=False,
        ))
    indexed = rows.set_index("era_order")
    turning_point_summaries = {
        1: (
            "Institution share rises from {p_institution:.1f}% to "
            "{c_institution:.1f}% ({p_institution_n:,} to {c_institution_n:,} "
            "mentions). Bank of the United States supplies 122 of the era's "
            "235 institution mentions; the Senate supplies 37."
        ),
        2: (
            "Nation share falls from {p_nation:.1f}% to {c_nation:.1f}% as "
            "group share rises to {c_group:.1f}% and person share to "
            "{c_person:.1f}%. Repeated domestic labels include Southern people "
            "(39), Edwin M. Stanton (26), and Senator Douglas (16)."
        ),
        4: (
            "Group share reaches {c_group:.1f}% as group mentions rise from "
            "{p_group_n:,} to {c_group_n:,} and nation mentions fall from "
            "{p_nation_n:,} to {c_nation_n:,}. Democratic Party (24), opponents "
            "(11), and alien enemies (9) recur among the group labels."
        ),
        5: (
            "Nation share jumps from {p_nation:.1f}% to {c_nation:.1f}% "
            "({p_nation_n:,} to {c_nation_n:,} mentions) while the total pool "
            "is nearly unchanged ({p_total_n:,} to {c_total_n:,}). Germany and "
            "Japan supply 70 nation mentions each; Italy supplies 19."
        ),
        7: (
            "Group share rises from {p_group:.1f}% to {c_group:.1f}% even as "
            "group mentions fall from {p_group_n:,} to {c_group_n:,}; the total "
            "pool contracts from {p_total_n:,} to {c_total_n:,}. al Qaeda (73), "
            "terrorists (52), and Taliban (37) lead the raw group labels."
        ),
        8: (
            "Group share falls from {p_group:.1f}% to {c_group:.1f}% despite "
            "more group mentions ({p_group_n:,} to {c_group_n:,}), because "
            "person and institution mentions grow faster. Repeated person "
            "labels include Putin (53), Biden (43), Joe Biden (34), and Donald "
            "Trump (32)."
        ),
    }
    color_by_key = {
        key: color for _column, key, _emoji, _label, color
        in SUMMARY_CONFLICT_CATEGORY_SPECS
    }
    for era_order, key, label, ax, ay in SUMMARY_CONFLICT_TURNING_POINT_SPECS:
        current = indexed.loc[era_order]
        previous = indexed.loc[era_order - 1]
        if (
            int(current["n_adversarial_entities"]) == 0
            or int(previous["n_adversarial_entities"]) == 0
        ):
            continue
        values = {}
        for _column, category_key, _emoji, _category_label, _color in (
            SUMMARY_CONFLICT_CATEGORY_SPECS
        ):
            values[f"p_{category_key}"] = (
                float(previous[f"adv_share_{category_key}"]) * 100
            )
            values[f"c_{category_key}"] = (
                float(current[f"adv_share_{category_key}"]) * 100
            )
            values[f"p_{category_key}_n"] = int(previous[f"adv_n_{category_key}"])
            values[f"c_{category_key}_n"] = int(current[f"adv_n_{category_key}"])
        values["p_total_n"] = int(previous["n_adversarial_entities"])
        values["c_total_n"] = int(current["n_adversarial_entities"])
        summary = turning_point_summaries[era_order].format(**values)
        color = color_by_key[key]
        fig.add_annotation(
            x=era_order,
            y=float(current[f"adv_share_{key}"]) * 100,
            xref="x", yref="y",
            name=f"conflict-turn::{key}",
            text=label,
            showarrow=True,
            arrowhead=0,
            arrowwidth=1.1,
            arrowcolor=color,
            ax=ax,
            ay=ay,
            bgcolor="#fffdf9",
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
            font=dict(size=9, color=INK2),
            hovertext="<br>".join(textwrap.wrap(
                summary,
                width=34,
                break_long_words=False,
                break_on_hyphens=False,
            )),
        )
    for x_value in x_values[rows["n_adversarial_entities"].eq(0)]:
        fig.add_annotation(
            x=int(x_value), y=target_axis_max * .08,
            xref="x", yref="y",
            text="N/A", showarrow=False,
            font=dict(color=MUTED, size=9),
        )
    fig.update_layout(**_layout(
        height=520,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#fffdf9", bordercolor="#d8d0c4",
            font=dict(family=FONT, color=INK2, size=11),
        ),
        showlegend=False,
        margin=dict(l=76, r=38, t=64, b=96),
        xaxis=dict(
            range=[-.35, len(rows) - .65],
            tickmode="array", tickvals=x_values, ticktext=tick_text,
            title="fixed reporting eras · categorical order",
            showgrid=False, zeroline=False, linecolor=BASELINE,
            tickfont=dict(color=MUTED, size=9),
            fixedrange=True, showspikes=False,
        ),
        yaxis=dict(
            title="share of adversarial entity mentions",
            range=[0, target_axis_max],
            gridcolor="rgba(116,105,95,.16)", zeroline=False,
            linecolor=BASELINE, tickfont=dict(color=MUTED, size=9),
            fixedrange=True, showspikes=False,
        ),
    ))
    tick_step = 20 if target_axis_max >= 60 else 10
    tick_values = list(np.arange(0, target_axis_max + .01, tick_step))
    fig.update_yaxes(tickvals=tick_values, ticksuffix="%")
    return fig


def fig_summary_conflict_portraits(
    contract: dict,
    faces: dict[str, str],
) -> go.Figure:
    """Relate zero-sum and partisan framing; portrait area shows enemy naming."""
    rows = _summary_conflict_ordered_rows(contract)
    rate_columns = ["zero_sum", "party_attack", "enemy_naming"]
    available = rows[rate_columns].notna().all(axis=1)
    plotted = rows.loc[available].copy().sort_values(
        "enemy_naming", ascending=False, kind="stable"
    )
    if plotted.empty:
        fig = go.Figure()
        fig.add_annotation(
            x=.5, y=.5, xref="paper", yref="paper",
            text="N/A · no president has all three conflict rates available",
            showarrow=False, font=dict(size=13, color=MUTED),
        )
        fig.update_layout(**_layout(
            height=650, showlegend=False,
            title=dict(
                text="Zero-sum framing versus partisan attack",
                font=dict(size=15),
            ),
            margin=dict(l=78, r=28, t=70, b=66),
            xaxis=dict(
                title="paragraphs with zero-sum framing", range=[0, 5],
                ticksuffix="%", gridcolor=GRID, linecolor=BASELINE,
            ),
            yaxis=dict(
                title="paragraphs with partisan attack", range=[0, 5],
                ticksuffix="%", gridcolor=GRID, linecolor=BASELINE,
            ),
        ))
        return fig
    missing_faces = set(plotted["president"].astype(str)) - set(faces)
    if missing_faces:
        raise ValueError(
            "summary conflict portrait scatter is missing portraits: "
            + ", ".join(sorted(missing_faces))
        )

    x_values = plotted["zero_sum"].mul(100)
    y_values = plotted["party_attack"].mul(100)
    enemy_values = plotted["enemy_naming"].mul(100)
    enemy_reference = float(plotted["enemy_naming"].max())
    if enemy_reference > 0:
        diameters = pd.Series(
            SUMMARY_CONFLICT_PORTRAIT_MAX_PX
            * np.sqrt(plotted["enemy_naming"] / enemy_reference),
            index=plotted.index,
        ).where(
            plotted["enemy_naming"].gt(0),
            SUMMARY_CONFLICT_PORTRAIT_ZERO_PX,
        )
    else:
        diameters = pd.Series(
            SUMMARY_CONFLICT_PORTRAIT_ZERO_PX, index=plotted.index
        )
    x_ceiling = max(5.0, float(np.ceil(x_values.max() / 5) * 5))
    y_ceiling = max(5.0, float(np.ceil(y_values.max() / 5) * 5))
    x_padding = (
        x_ceiling * SUMMARY_CONFLICT_PORTRAIT_MAX_PX
        / (2 * (540 - SUMMARY_CONFLICT_PORTRAIT_MAX_PX))
    )
    y_padding = (
        y_ceiling * SUMMARY_CONFLICT_PORTRAIT_MAX_PX
        / (2 * (470 - SUMMARY_CONFLICT_PORTRAIT_MAX_PX))
    )
    x_axis_span = x_ceiling + 2 * x_padding
    y_axis_span = y_ceiling + 2 * y_padding
    support_notes = [
        "Thin paragraph record" if status == "thin_record" else "Observed record"
        for status in plotted["support_status"].astype(str)
    ]
    border_encodings = [
        _summary_conflict_dominant_adversary(row)
        for row in plotted.itertuples(index=False)
    ]
    border_notes = [label for _key, label, _color in border_encodings]
    border_colors = [color for _key, _label, color in border_encodings]
    coverage_notes = [
        f"{int(count)} annual message{'s' if int(count) != 1 else ''}"
        for count in plotted["n_speeches"]
    ]
    fig = go.Figure(go.Scatter(
        x=x_values,
        y=y_values,
        mode="markers",
        marker=dict(
            size=diameters,
            sizemode="diameter",
            color="#fff8ec",
            opacity=1,
            line=dict(
                color=border_colors,
                width=SUMMARY_CONFLICT_PORTRAIT_BORDER_PX,
            ),
        ),
        customdata=list(zip(
            plotted["president"].astype(str),
            plotted["year_first"].astype(int),
            plotted["year_last"].astype(int),
            enemy_values,
            coverage_notes,
            support_notes,
            border_notes,
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b>"
            "<br><b>Years:</b> %{customdata[1]}–%{customdata[2]}"
            "<br><b>Zero-sum framing:</b> %{x:.1f}% of paragraphs"
            "<br><b>Partisan attack:</b> %{y:.1f}% of paragraphs"
            "<br><b>Enemy naming:</b> %{customdata[3]:.1f}% of paragraphs "
            "· portrait size"
            "<br><b>Most-named adversary type:</b> %{customdata[6]} "
            "· portrait border"
            "<br><b>Speeches analyzed:</b> %{customdata[4]}"
            "<br><b>Support:</b> %{customdata[5]}<extra></extra>"
        ),
        cliponaxis=False,
        showlegend=False,
    ))
    for (_, row), diameter in zip(plotted.iterrows(), diameters):
        portrait_diameter = max(
            6.0,
            float(diameter) - SUMMARY_CONFLICT_PORTRAIT_BORDER_PX - 1.0,
        )
        fig.add_layout_image(
            source=faces[str(row.president)],
            x=float(row.zero_sum) * 100,
            y=float(row.party_attack) * 100,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
            sizex=x_axis_span * portrait_diameter / 540,
            sizey=y_axis_span * portrait_diameter / 470,
            sizing="contain",
            layer="above",
            opacity=1,
            name=f"conflict-portrait::{row.president}",
        )
    fig.update_layout(**_layout(
        height=650,
        title=dict(
            text="Zero-sum framing versus partisan attack",
            font=dict(size=15),
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#fffdf9",
            bordercolor="#d8d0c4",
            align="left",
            font=dict(family=FONT, color=INK2, size=11),
        ),
        showlegend=False,
        margin=dict(l=78, r=28, t=70, b=66),
        xaxis=dict(
            title="paragraphs with zero-sum framing",
            range=[-x_padding, x_ceiling + x_padding],
            tickvals=list(np.arange(0, x_ceiling + .01, 5)),
            ticksuffix="%",
            gridcolor=GRID,
            linecolor=BASELINE,
            showspikes=False,
        ),
        yaxis=dict(
            title="paragraphs with partisan attack",
            range=[-y_padding, y_ceiling + y_padding],
            tickvals=list(np.arange(0, y_ceiling + .01, 5)),
            ticksuffix="%",
            gridcolor=GRID,
            linecolor=BASELINE,
            showspikes=False,
        ),
    ))
    return fig


def _summary_conflict_line_controls_html() -> str:
    buttons = ['<button type="button" data-conflict-line="__all__" '
               'aria-pressed="true">All</button>']
    buttons.extend(
        '<button type="button" data-conflict-line="'
        + html.escape(key, quote=True)
        + '" data-line-style="'
        + html.escape(SUMMARY_CONFLICT_CATEGORY_LINE_STYLES[key][0], quote=True)
        + '" class="conflict-line-'
        + html.escape(key, quote=True)
        + '" aria-pressed="false"><span aria-hidden="true"></span>'
        + html.escape(label)
        + "</button>"
        for _column, key, _emoji, label, _color
        in SUMMARY_CONFLICT_CATEGORY_SPECS
    )
    return f"""<div class="conflict-line-picker" data-conflict-line-picker>
  <span class="conflict-line-picker-label">Highlight a line</span>
  <div class="conflict-line-buttons" role="group" aria-label="Highlight a target-type line">
    {"".join(buttons)}
  </div>
  <span class="conflict-line-status" role="status" aria-live="polite">All five target lines emphasized.</span>
</div>"""


def _summary_conflict_category_guide_html(contract: dict) -> str:
    definitions = "".join(
        "<div><dt>" + html.escape(label) + "</dt><dd>"
        + html.escape(description) + "</dd></div>"
        for label, description in SUMMARY_CONFLICT_CATEGORY_GUIDE
    )
    return f"""<details class="conflict-category-guide">
<summary id="conflict-category-guide-title">What each target category includes</summary>
<div class="conflict-category-guide-body">
<dl>{definitions}</dl>
<p>Each point pools speaker-audited adversarial entity mentions within one fixed reporting era;
it is not an average of presidents or years. Repeated mentions count, and one paragraph can
contribute more than one. Because the five shares total 100%, a rise in one category mechanically
lowers at least one other. Era lengths, corpus density, and genre mix vary, and connecting lines
guide the eye between aggregates without estimating values between them. Categories are frozen
AI-assigned types; “Other” is a heterogeneous residual, not a single kind of enemy. A textual
target is not a judgment that it was legitimate. The corpus currently ends in {contract['corpus_end_date'].strftime('%B %Y')}.</p>
</div>
</details>"""


def _summary_conflict_portrait_border_key_html() -> str:
    items = "".join(
        '<span style="--portrait-border-color:'
        + html.escape(color, quote=True)
        + '"><i aria-hidden="true"></i>'
        + html.escape(label)
        + "</span>"
        for _column, _key, _emoji, label, color
        in SUMMARY_CONFLICT_CATEGORY_SPECS
    )
    items += (
        '<span style="--portrait-border-color:'
        + SUMMARY_CONFLICT_PORTRAIT_TIE_COLOR
        + '"><i aria-hidden="true"></i>Tied leaders</span>'
    )
    return f"""<div class="conflict-portrait-border-key" role="note"
aria-label="Portrait border colors show each president's most-named adversary type">
<strong>Portrait border · most named in these messages</strong>
<div>{items}</div>
</div>"""


def _summary_conflict_graphs_html(
    target_contract: dict,
    president_contract: dict,
) -> str:
    speaker_audit_complete = (
        str(target_contract["speaker_scope_status"]) == "speaker_audit_complete"
        and str(president_contract["speaker_scope_status"]) == "speaker_audit_complete"
    )
    speaker_caveat = (
        "The selected treatment uses audited paragraph speakers; thin corpus records "
        "remain more sensitive to genre and coverage."
        if speaker_audit_complete
        else "Document ownership can include other speakers, so values remain "
        "provisional until the speaker audit."
    )
    category_evidence = _chart_evidence(
        "enemy_category_share", str(target_contract["source_label"]),
        support=(
            "All nine fixed reporting eras pool speaker-audited adversarial entity "
            "mentions; each era's five shares total 100%."
        ),
        caveat=(
            "The eras differ in duration and corpus composition. Lines connect discrete "
            "era aggregates as a chronological guide and do not estimate intervening years. "
            + speaker_caveat
        ),
    )
    portrait_evidence = _chart_evidence(
        "enemy_identity", str(president_contract["source_label"]),
        measure_label=(
            "Zero-sum framing × partisan attack · enemy-naming portrait area"
        ),
        support=(
            "State of the Union and annual-message paragraphs only. Presidents without "
            "a qualifying corpus speech remain explicit N/A rows; every available rate "
            "divides its labeled paragraphs by all eligible message paragraphs."
        ),
        caveat=(
            "Portrait area—not diameter—is proportional to enemy-naming share; the "
            "largest observed share uses the source image's native 64-pixel size. A zero "
            "uses a 16-pixel locator. Overlap does not merge presidents; hover and the "
            "generated figure retain each value. Annual messages provide a shared genre across "
            "eras, not identical venues, delivery modes, record sizes, or political contexts."
        ),
    )
    return f"""<div class="conflict-graph-suite"
     data-conflict-schema="{html.escape(str(president_contract['schema_version']), quote=True)}"
     data-target-schema="{html.escape(str(target_contract['schema_version']), quote=True)}"
     data-conflict-target-treatment="{html.escape(str(target_contract['treatment']), quote=True)}"
     data-conflict-president-treatment="{html.escape(str(president_contract['treatment']), quote=True)}">
  <header class="conflict-graph-heading">
    <div><p>{len(target_contract['rows'])} ERAS · {len(president_contract['rows'])} PRESIDENTS · 2 COMPARISONS</p>
    <h3>Conflict across eras and presidents</h3></div>
    <div class="conflict-treatment" role="note"><span>Governed inputs</span>
    <strong>Target mix · {html.escape(str(target_contract['treatment_label']))}</strong>
    <small>Portrait · {html.escape(str(president_contract['treatment_label']))}</small></div>
  </header>
  <p class="conflict-graph-deck">Five lines on one era chart follow who or what receives
  adversarial framing; a common-genre president portrait view shows how three paragraph-level
  frames combine. Target lines use all eligible adversarial entity mentions; portrait position
  and area use State of the Union and annual-message paragraphs only.
  <span>○ fewer than {SUMMARY_CONFLICT_COMPOSITION_MIN} adversarial mentions · hover points for exact counts · hover labeled turns for the compositional explanation</span></p>
  <section class="conflict-question-group" aria-labelledby="conflict-target-heading">
    <div class="conflict-question-heading"><span>01</span><div>
    <p>WHO OR WHAT IS NAMED?</p><h4 id="conflict-target-heading">How the target mix changes</h4>
    <small>Five target types share one scale; highlight any line, then inspect six labeled changes in the mix.</small>
    </div></div>
    <section class="conflict-measure-panel conflict-target-panel" aria-label="Target mix by historical era">
      <p class="conflict-denominator">Each era pools speaker-audited adversarial mentions ·
      across the five lines, every supported era totals 100% · lines connect aggregates only</p>
      {_summary_conflict_line_controls_html()}
      <div class="chart-scroll conflict-target-scroll"><div class="chart"
      data-fig="summary_conflict_targets" style="height:520px"
      aria-label="Five target-type lines across nine historical eras"></div></div>
      {_summary_conflict_category_guide_html(target_contract)}
    </section>
  </section>
  <section class="conflict-question-group" aria-labelledby="conflict-relationship-heading">
    <div class="conflict-question-heading"><span>02</span><div>
    <p>BY PRESIDENT · COMPARABLE SPEECHES</p>
    <h4 id="conflict-relationship-heading">How presidents frame conflict</h4>
    <small>Compared within State of the Union and annual-message speeches.</small>
    </div></div>
    <section class="conflict-measure-panel conflict-portrait-panel"
             style="--measure-color:#6d5a91"
             aria-labelledby="summary-conflict-frame-portraits-title">
      <header><div>
      <h5 id="summary-conflict-frame-portraits-title">Zero-sum × partisan × enemy naming</h5>
      <p>Portrait area = enemy naming.<br>
      All values are shares of eligible State of the Union and annual-message paragraphs.</p>
      </div></header>
      {_summary_conflict_portrait_border_key_html()}
      <div class="chart-scroll conflict-portrait-scroll"><div class="chart"
      data-fig="{SUMMARY_CONFLICT_PORTRAIT_FIGURE_KEY}" style="height:650px"></div></div>
    </section>
  </section>
</div>
<details class="summary-evidence-detail"><summary>Evidence, limitations, and speaker-audited data</summary>
<div class="conflict-evidence-grid">{category_evidence}{portrait_evidence}</div>
<p>The target and portrait figures use separate validated era- and president-grain contracts from
governed speaker-audited populations: all eligible paragraphs for the target mix and the shared
annual-message genre for the portrait comparison. Rebuilding the governed producer updates both
without averaging percentages or applying chart-specific corrections.</p></details>"""


def _summary_register_series(
    trend_table: pd.DataFrame,
    measure: str,
    taxonomy: str = "none",
) -> pd.DataFrame:
    rows = trend_table[
        trend_table["unit"].eq("era")
        & trend_table["measure"].eq(measure)
        & trend_table["taxonomy"].eq(taxonomy)
        & trend_table["genre_treatment"].eq("sotu_only")
        & trend_table["statistic"].eq("level")
    ].sort_values("period").copy()
    periods = tuple(int(value) for value in rows["period"])
    if periods != SUMMARY_REGISTER_PERIODS:
        raise ValueError(
            f"summary register {measure!r} needs the nine fixed era rows; got {periods}"
        )
    rows["display_period"] = SUMMARY_REGISTER_LABELS
    return rows


def build_summary_temporal_president_contract(
    speech_markers: pd.DataFrame,
    speeches: pd.DataFrame,
    *,
    expected_presidents: int | None = 45,
) -> dict:
    """Pool all-corpus future and nostalgia wording by president."""
    required_by_source = {
        "speech markers": {
            "doc_name", "president", "year", "n_words", "future", "nostalgia",
        },
        "speeches": {"doc_name", "president", "year"},
    }
    sources = {
        "speech markers": speech_markers,
        "speeches": speeches,
    }
    for source_name, required in required_by_source.items():
        frame = sources[source_name]
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"summary temporal {source_name} is missing columns: "
                + ", ".join(sorted(missing))
            )
        if (
            frame["doc_name"].isna().any()
            or frame["doc_name"].astype(str).str.strip().eq("").any()
            or frame["doc_name"].duplicated().any()
        ):
            raise ValueError(
                f"summary temporal {source_name} needs one named row per speech"
            )
        if (
            frame["president"].isna().any()
            or frame["president"].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(
                f"summary temporal {source_name} needs a named president per speech"
            )
        years = pd.to_numeric(frame["year"], errors="coerce")
        if (
            years.isna().any()
            or np.isinf(years.to_numpy(dtype=float)).any()
            or not np.allclose(years, np.round(years))
        ):
            raise ValueError(
                f"summary temporal {source_name} years must be finite integers"
            )

    speech_meta = speeches[["doc_name", "president", "year"]].copy()
    marker_rows = speech_markers[
        [
            "doc_name", "president", "year", "n_words", "future", "nostalgia",
        ]
    ].rename(columns={"president": "marker_president", "year": "marker_year"})
    joined = _strict_one_to_one_merge(
        speech_meta,
        marker_rows,
        ["doc_name"],
        "summary temporal speeches x markers",
    )
    if not joined["president"].astype(str).equals(
        joined["marker_president"].astype(str)
    ):
        raise ValueError("summary temporal president metadata differs across sources")
    if not np.array_equal(
        joined["year"].to_numpy(), joined["marker_year"].to_numpy()
    ):
        raise ValueError("summary temporal year metadata differs across sources")

    count_columns = ["n_words", "future", "nostalgia"]
    numeric = joined[count_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("summary temporal source counts must be finite")
    if (numeric < 0).any().any():
        raise ValueError("summary temporal source counts cannot be negative")
    for column in count_columns:
        if not np.allclose(numeric[column], np.round(numeric[column])):
            raise ValueError(f"summary temporal count {column!r} is not integral")
        joined[column] = numeric[column].astype(int)
    if (
        joined["future"].gt(joined["n_words"])
        | joined["nostalgia"].gt(joined["n_words"])
    ).any():
        raise ValueError(
            "summary temporal family matches cannot exceed the marker-word denominator"
        )

    rows = (
        joined.groupby("president", sort=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_speeches=("doc_name", "size"),
            n_rate_words=("n_words", "sum"),
            n_future_matches=("future", "sum"),
            n_nostalgia_matches=("nostalgia", "sum"),
        )
        .reset_index()
    )
    if expected_presidents is not None and len(rows) != expected_presidents:
        raise ValueError(
            f"summary temporal payload needs {expected_presidents} presidents; "
            f"got {len(rows)}"
        )
    if rows["president"].isna().any() or rows["president"].duplicated().any():
        raise ValueError("summary temporal payload needs one named row per president")
    if expected_presidents == len(SUMMARY_CONFLICT_DISPLAY_ORDER):
        expected_names = set(SUMMARY_CONFLICT_DISPLAY_ORDER)
        actual_names = set(rows["president"].astype(str))
        if actual_names != expected_names:
            raise ValueError(
                "summary temporal president membership changed; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"extra={sorted(actual_names - expected_names)}"
            )
    order = {
        president: index
        for index, president in enumerate(SUMMARY_CONFLICT_DISPLAY_ORDER)
    }
    rows["display_order"] = rows["president"].map(order)
    if expected_presidents is None and rows["display_order"].isna().any():
        rows["display_order"] = np.arange(len(rows))
    if rows["display_order"].isna().any():
        raise ValueError("summary temporal display order is incomplete")
    rows["display_order"] = rows["display_order"].astype(int)
    rows["future"] = np.where(
        rows["n_rate_words"].gt(0),
        rows["n_future_matches"] / rows["n_rate_words"] * 10_000,
        np.nan,
    )
    rows["nostalgia"] = np.where(
        rows["n_rate_words"].gt(0),
        rows["n_nostalgia_matches"] / rows["n_rate_words"] * 10_000,
        np.nan,
    )
    rows["era"] = rows.apply(_president_story_era, axis=1)
    complete = rows[["future", "nostalgia"]].notna().all(axis=1)
    rows["support_status"] = np.select(
        [
            ~complete,
            rows["n_speeches"].lt(SUMMARY_TEMPORAL_SUPPORT_MIN_SPEECHES),
        ],
        ["not_available", "thin_record"],
        default="observed",
    )
    return {
        "schema_version": "summary-temporal-president-v2",
        "treatment": "all_corpus_document_owner",
        "treatment_label": "All corpus speeches · document-owner president records",
        "speaker_scope_status": "document_owned_transcripts_not_speaker_audited",
        "source_label": "speech_markers.parquet + speeches.parquet",
        "rate_unit": "matches_per_10k_marker_words",
        "support_min_speeches": SUMMARY_TEMPORAL_SUPPORT_MIN_SPEECHES,
        "corpus_end_date": (
            pd.Timestamp(speeches["date"].max())
            if "date" in speeches and speeches["date"].notna().any()
            else None
        ),
        "rows": rows.sort_values("display_order", kind="stable").reset_index(drop=True),
    }


def build_summary_data(
    speeches: pd.DataFrame,
    paragraphs: pd.DataFrame,
    paragraph_annotations: pd.DataFrame,
    speech_annotations: pd.DataFrame,
    speech_markers: pd.DataFrame,
    taxonomy: dict,
    lifecycle_table: pd.DataFrame,
    trend_table: pd.DataFrame,
    president_conflict_treatments: pd.DataFrame,
    target_mix_table: pd.DataFrame,
    receipts: pd.DataFrame,
    founding_data: dict,
) -> dict:
    """Build one keyed, validated payload for every Summary-only aggregate."""
    temporal_presidents = build_summary_temporal_president_contract(
        speech_markers, speeches
    )
    paragraph_frame = _strict_one_to_one_merge(
        paragraphs[["doc_name", "para_idx"]],
        paragraph_annotations[
            [
                "doc_name", "para_idx", "topics", "proposal_values",
                "enemy_naming", "party_attack", "zero_sum",
            ]
        ],
        ["doc_name", "para_idx"],
        "summary paragraphs x paragraph annotations",
    )
    speech_meta = speeches[["doc_name", "president", "year"]].copy()
    if speech_meta["doc_name"].duplicated().any():
        raise ValueError("summary speech metadata has duplicate doc_name values")
    paragraph_frame = paragraph_frame.merge(
        speech_meta, on="doc_name", how="left", validate="many_to_one",
        indicator=True,
    )
    if paragraph_frame["_merge"].ne("both").any():
        missing = int(paragraph_frame["_merge"].ne("both").sum())
        raise ValueError(f"summary paragraphs x speeches left {missing} unmatched rows")
    paragraph_frame = paragraph_frame.drop(columns="_merge")
    paragraph_frame["era_key"] = era_profiles.story_era_key_series(
        paragraph_frame["year"], paragraph_frame["president"]
    )
    if paragraph_frame["era_key"].isna().any():
        raise ValueError("summary paragraphs contain years outside the Story eras")

    label_map = attention.canonical_label_map(taxonomy)
    parent_map = attention.level1_parents(taxonomy)
    paragraph_frame["domains"] = [
        frozenset(
            parent_map[topic]
            for topic in attention.normalize_topics(raw, label_map)
        )
        for raw in paragraph_frame["topics"]
    ]
    era_keys = [spec.key for spec in era_profiles.ERA_PROFILE_SPECS]
    level1_names = [row["name"] for row in taxonomy["level1"]]
    denominators = paragraph_frame.groupby("era_key", observed=True).size()
    if set(denominators.index) != set(era_keys):
        raise ValueError("summary lifecycle is missing one or more Story eras")
    lifecycle_share = pd.DataFrame(
        {
            era_key: [
                paragraph_frame.loc[
                    paragraph_frame["era_key"].eq(era_key), "domains"
                ].map(lambda domains: domain in domains).mean() * 100
                for domain in level1_names
            ]
            for era_key in era_keys
        },
        index=level1_names,
    )
    lifecycle_rows = lifecycle_table[
        lifecycle_table["level"].eq("level1")
        & lifecycle_table["treatment"].eq("raw")
    ].set_index("topic")
    if set(lifecycle_rows.index) != set(level1_names):
        raise ValueError("summary lifecycle classes do not match taxonomy Level-1")

    stance = paragraph_frame["proposal_values"].astype(str).replace({"both": "mixed"})
    expected_stance = {"proposal", "values", "mixed", "neither"}
    if set(stance.unique()) != expected_stance:
        raise ValueError(
            "summary proposal/value labels changed: "
            f"{sorted(set(stance.unique()) - expected_stance)}"
        )
    stance_frame = (
        pd.crosstab(paragraph_frame["era_key"], stance, normalize="index")
        .reindex(index=era_keys, columns=["proposal", "values", "mixed", "neither"])
        .mul(100)
    )
    if not np.allclose(stance_frame.sum(axis=1), 100):
        raise ValueError("summary proposal/value shares must sum to 100% by era")

    if "treatment" not in president_conflict_treatments:
        raise ValueError("summary conflict president treatments lack treatment labels")
    target_population_rows = president_conflict_treatments.loc[
        president_conflict_treatments["treatment"].eq(
            SUMMARY_CONFLICT_TARGET_TREATMENT
        )
    ].copy()
    portrait_rows = president_conflict_treatments.loc[
        president_conflict_treatments["treatment"].eq(
            SUMMARY_CONFLICT_PORTRAIT_TREATMENT
        )
    ].copy()
    if target_population_rows.empty or portrait_rows.empty:
        raise ValueError(
            "summary conflict president treatments lack the required all-speaker "
            "or annual-message population"
        )
    conflict_target_population = build_summary_conflict_contract(
        target_population_rows,
        expected_presidents=int(speeches["president"].nunique()),
    )
    conflict_presidents = build_summary_conflict_contract(
        portrait_rows,
        expected_presidents=int(speeches["president"].nunique()),
    )
    conflict_targets = build_summary_conflict_target_contract(target_mix_table)
    for count_column, *_ in SUMMARY_CONFLICT_CATEGORY_SPECS:
        target_total = int(conflict_targets["rows"][count_column].sum())
        president_total = int(
            conflict_target_population["rows"][count_column].sum()
        )
        if target_total != president_total:
            raise ValueError(
                f"summary conflict {count_column} differs by era and president grain"
            )
    if int(conflict_targets["rows"]["n_paragraphs"].sum()) != int(
        conflict_target_population["rows"]["n_paragraphs"].sum()
    ):
        raise ValueError(
            "summary conflict eligible paragraphs differ by era and president grain"
        )
    if conflict_targets["treatment"] != conflict_target_population["treatment"]:
        raise ValueError("summary conflict contracts use different treatments")
    if "date" in speeches:
        expected_end_date = pd.Timestamp(speeches["date"].max()).date()
        if conflict_targets["corpus_end_date"].date() != expected_end_date:
            raise ValueError("summary target-mix corpus end date differs from speeches")

    receipt_rows = receipts[
        receipts["status"].eq("confirmatory")
        & receipts["metric"].eq("effective_topics")
    ]
    if len(receipt_rows) != 1:
        raise ValueError("summary needs one confirmatory breadth receipt")

    # Validate every published register arm now, before any figure can render.
    register = {
        measure: _summary_register_series(
            trend_table, measure,
            taxonomy="llm_level2" if measure == "effective_topics" else "none",
        )
        for measure in (
            "effective_topics", "mechanism", "hedges", "boosters", "hype",
            "doom", "opponents",
        )
    }
    communications = founding_data["communications"]
    if len(communications) != 9:
        raise ValueError("summary communication payload needs nine Story eras")
    for era in communications:
        for dimension in ("audience", "medium"):
            if not np.isclose(sum(row["share"] for row in era[dimension]), 100):
                raise ValueError(
                    f"summary {dimension} shares do not sum to 100 in {era['era']}"
                )
    return {
        "paragraphs": paragraph_frame,
        "lifecycle_share": lifecycle_share,
        "lifecycle_rows": lifecycle_rows,
        "stance": stance_frame,
        "conflict_targets": conflict_targets,
        "conflict_presidents": conflict_presidents,
        "temporal_presidents": temporal_presidents,
        "register": register,
        "communications": communications,
        "breadth_receipt": receipt_rows.iloc[0],
    }


def fig_summary_breadth(summary_data: dict) -> go.Figure:
    rows = summary_data["register"]["effective_topics"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rows["display_period"], y=rows["value"], mode="lines+markers",
        line=dict(color="#315f78", width=3), marker=dict(size=10, color="#315f78"),
        customdata=np.stack([
            rows["ci_low"], rows["ci_high"], rows["n_speeches"]
        ], axis=-1),
        hovertemplate=(
            "<b>%{x}</b><br>%{y:.2f} effective topics"
            "<br>95% interval %{customdata[0]:.2f}–%{customdata[1]:.2f}"
            "<br>%{customdata[2]:.0f} annual messages<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=list(rows["display_period"]) + list(rows["display_period"])[::-1],
        y=list(rows["ci_high"]) + list(rows["ci_low"])[::-1],
        fill="toself", fillcolor="rgba(49,95,120,.12)",
        line=dict(color="rgba(49,95,120,0)"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_vrect(x0=5.5, x1=8.5, fillcolor="rgba(155,78,80,.055)", line_width=0)
    fig.add_annotation(
        x="2010–26", y=float(rows.iloc[-1]["value"]), text="modern returns near the top",
        showarrow=True, ax=-96, ay=-44, bgcolor="#fff9ef", bordercolor="#8b6c42",
    )
    fig.update_layout(**_layout(
        height=500, title=dict(
            text="Breadth rises early, then rises again in the modern block",
            font=dict(size=15),
        ),
        xaxis=dict(title="fixed 30-year block", gridcolor=GRID, linecolor=BASELINE),
        yaxis=dict(title="effective Level-2 topics", gridcolor=GRID,
                   linecolor=BASELINE, rangemode="tozero"),
        margin=dict(l=70, r=28, t=66, b=58),
    ))
    return fig


def _fig_summary_communication_bar(
    summary_data: dict,
    dimension: str,
) -> go.Figure:
    """One graph-specific 100% stacked communication chart for Summary."""
    configurations = {
        "audience": (
            (
                ("Congress", "🏛️ Congress", "#315f78", "#ffffff"),
                ("General public", "👥 General public", "#b16b3e", "#ffffff"),
                (
                    "Specific organizations or groups",
                    "🤝 Specific groups",
                    "#c9bda9",
                    "#302b26",
                ),
                ("Other audiences", "🌐 Other audiences", "#74695f", "#ffffff"),
            ),
            "Assigned audience",
        ),
        "medium": (
            (
                ("Written message", "✉️ Written", "#5f568c", "#ffffff"),
                ("Spoken address", "🎙️ Spoken", "#4f7557", "#ffffff"),
                ("Radio/TV broadcast", "📻 Radio / TV", "#c58a3a", "#302b26"),
                (
                    "Press conference/debate or other performed form",
                    "🗣️ Press / debate",
                    "#a24f52",
                    "#ffffff",
                ),
            ),
            "Assigned delivery form",
        ),
    }
    if dimension not in configurations:
        raise ValueError(f"unsupported Summary communication dimension: {dimension}")
    specs, _ = configurations[dimension]
    communications = summary_data["communications"]
    if len(communications) != len(SUMMARY_ERA_SHORT):
        raise ValueError("Summary communication charts require all nine Story eras")
    x = [
        f"{short}<br><span style='font-size:9px'>{era['years']}</span>"
        for short, era in zip(SUMMARY_ERA_SHORT, communications)
    ]
    expected = {spec[0] for spec in specs}
    fig = go.Figure()
    for source_label, legend_label, color, text_color in specs:
        values = []
        for era in communications:
            rows = {row["label"]: row for row in era[dimension]}
            if set(rows) != expected:
                raise ValueError(
                    f"Summary {dimension} bar categories drifted for {era['short']}: "
                    f"{sorted(set(rows) ^ expected)}"
                )
            values.append(float(rows[source_label]["share"]))
        fig.add_trace(go.Bar(
            x=x,
            y=values,
            name=legend_label,
            marker=dict(
                color=color,
                line=dict(color="rgba(255,255,255,.92)", width=1.2),
            ),
            text=[f"{value:.0f}%" if value >= 7.5 else "" for value in values],
            texttemplate="%{text}",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color=text_color, size=11),
            hovertemplate=(
                f"<b>{legend_label}</b><br>%{{y:.1f}}% of era speeches"
                "<extra></extra>"
            ),
        ))
    totals = np.sum([np.asarray(trace.y, dtype=float) for trace in fig.data], axis=0)
    if not np.allclose(totals, 100, atol=.05):
        raise ValueError(
            f"Summary {dimension} stacked bars do not sum to 100%: {totals.tolist()}"
        )
    fig.update_layout(**_layout(
        height=520,
        barmode="stack",
        bargap=.22,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#fffdf9", bordercolor="#d8d0c4",
            font=dict(family=FONT, color=INK2, size=12),
        ),
        uniformtext=dict(minsize=10, mode="hide"),
        showlegend=False,
        margin=dict(l=54, r=18, t=24, b=88),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0)", linecolor=BASELINE,
            tickfont=dict(color=INK2, size=10), tickangle=0, fixedrange=True,
            showspikes=False,
        ),
        yaxis=dict(
            range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticksuffix="%",
            gridcolor="rgba(116,105,95,.18)", linecolor=BASELINE,
            tickfont=dict(color=MUTED, size=10), fixedrange=True,
            showspikes=False,
        ),
    ))
    return fig


def fig_summary_audience(summary_data: dict) -> go.Figure:
    return _fig_summary_communication_bar(summary_data, "audience")


def fig_summary_medium(summary_data: dict) -> go.Figure:
    return _fig_summary_communication_bar(summary_data, "medium")


def fig_summary_enemy_categories(summary_data: dict) -> go.Figure:
    frame = summary_data["adversary_categories"]
    specs = (
        ("adv_share_nation", "Nation", "#315f78"),
        ("adv_share_group", "Group", "#4f7557"),
        ("adv_share_person", "Person", "#a24f52"),
        ("adv_share_institution", "Institution", "#9a7133"),
        ("adv_share_other", "Other", "#74695f"),
    )
    fig = go.Figure()
    for column, label, color in specs:
        fig.add_trace(go.Bar(
            x=SUMMARY_ERA_SHORT,
            y=frame[column].mul(100),
            name=label,
            marker=dict(color=color, line=dict(color="white", width=.7)),
            customdata=np.stack([
                frame["n_adversarial_entities"], frame["n_paragraphs"]
            ], axis=-1),
            hovertemplate=(
                f"<b>{label}</b><br>%{{x}}: %{{y:.1f}}% of adversarial mentions"
                "<br>%{customdata[0]:.0f} entity mentions in era"
                "<br>%{customdata[1]:.0f} total paragraphs<extra></extra>"
            ),
        ))
    fig.update_layout(**_layout(
        height=540, barmode="stack",
        title=dict(text="What category of enemy is named?", font=dict(size=15)),
        yaxis=dict(
            title="share of adversarial entity mentions", range=[0, 100],
            ticksuffix="%", gridcolor=GRID, linecolor=BASELINE,
        ),
        xaxis=dict(tickangle=-24, gridcolor=GRID, linecolor=BASELINE),
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=76, r=24, t=78, b=92),
    ))
    return fig


def fig_summary_stance(summary_data: dict) -> go.Figure:
    frame = summary_data["stance"]
    fig = go.Figure()
    specs = (
        ("proposal", "Proposal only", "#315f78"),
        ("mixed", "Proposal + values", "#7b5f91"),
        ("values", "Values only", "#b16b3e"),
        ("neither", "Neither", "#b7aea2"),
    )
    for key, label, color in specs:
        fig.add_trace(go.Bar(
            x=SUMMARY_ERA_SHORT, y=frame[key], name=label,
            marker=dict(color=color, line=dict(color="white", width=.7)),
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**_layout(
        height=500, barmode="stack",
        title=dict(text="The formal report becomes more explicitly argumentative", font=dict(size=15)),
        yaxis=dict(title="share of paragraphs", range=[0, 100], ticksuffix="%",
                   gridcolor=GRID, linecolor=BASELINE),
        xaxis=dict(tickangle=-24, gridcolor=GRID, linecolor=BASELINE),
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=64, r=20, t=76, b=90),
    ))
    return fig


def fig_summary_lifecycle(summary_data: dict) -> go.Figure:
    shares = summary_data["lifecycle_share"].copy()
    lifecycle = summary_data["lifecycle_rows"].reindex(shares.index)
    order = lifecycle.assign(_topic=lifecycle.index).sort_values(
        ["lifecycle_class", "dominant_era", "_topic"]
    ).index
    shares = shares.reindex(order)
    class_labels = lifecycle.loc[order, "lifecycle_class"].str.title()
    y_labels = [f"{topic} · {klass}" for topic, klass in zip(order, class_labels)]
    custom = np.empty(shares.shape, dtype=object)
    for row_index, klass in enumerate(class_labels):
        custom[row_index, :] = klass
    fig = go.Figure(go.Heatmap(
        z=shares.to_numpy(), x=SUMMARY_ERA_SHORT, y=y_labels,
        colorscale=[
            [0, "#f4efe7"], [.2, "#d8e7e9"], [.55, "#6f9aaa"], [1, "#264f67"],
        ],
        colorbar=dict(title="paragraph<br>share", ticksuffix="%"),
        customdata=custom,
        hovertemplate=(
            "<b>%{y}</b><br>%{x}: %{z:.1f}% of paragraphs"
            "<br>Lifecycle: %{customdata}<extra></extra>"
        ),
        hoverongaps=False,
    ))
    fig.update_layout(**_layout(
        height=760, title=dict(text="Seventeen issue domains across nine eras", font=dict(size=15)),
        xaxis=dict(side="top", tickangle=-24, linecolor=BASELINE),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10), linecolor=BASELINE),
        margin=dict(l=275, r=80, t=115, b=35),
    ))
    return fig


def _summary_lifecycle_table(summary_data: dict) -> str:
    shares = summary_data["lifecycle_share"]
    lifecycle = summary_data["lifecycle_rows"].reindex(shares.index)
    rows = "".join(
        f"<tr><th scope=\"row\">{html.escape(topic)}</th>"
        f"<td>{html.escape(str(row.lifecycle_class).title())}</td>"
        f"<td>{html.escape(str(row.dominant_era))}</td>"
        f"<td>{float(shares.loc[topic].max()):.1f}%</td></tr>"
        for topic, row in lifecycle.assign(_topic=lifecycle.index).sort_values(
            ["lifecycle_class", "_topic"]
        ).iterrows()
    )
    return f"""<details class="summary-evidence-detail"><summary>Text alternative · all 17 agenda lifecycles</summary>
<div class="table-scroll"><table><thead><tr><th>Domain</th><th>Lifecycle</th>
<th>Dominant era</th><th>Largest era share</th></tr></thead><tbody>{rows}</tbody></table></div>
</details>"""


def _summary_adversary_html(founding_data: dict) -> str:
    profiles = founding_data["era_profiles"]
    cards = []
    table_rows = []
    for short, spec in zip(SUMMARY_ERA_SHORT, era_profiles.ERA_PROFILE_SPECS):
        adversaries = profiles[spec.key]["adversaries"]
        max_count = max(row["paragraphs"] for row in adversaries)
        bubbles = "".join(
            f'<li class="adversary-{html.escape(row["type"])}" '
            f'style="--adversary-size:{18 + 26 * np.sqrt(row["paragraphs"] / max_count):.1f}px">'
            f'<span aria-hidden="true"></span><strong>{html.escape(row["name"])}</strong>'
            f'<small>{row["paragraphs"]} paragraphs · {html.escape(row["type"])}</small></li>'
            for row in adversaries
        )
        cards.append(
            f'<article class="adversary-era"><header><strong>{html.escape(short)}</strong>'
            f'<span>{spec.start_year}–{spec.end_year}</span></header><ol>{bubbles}</ol></article>'
        )
        table_rows.append(
            f'<tr><th scope="row">{html.escape(short)}</th><td>'
            + " · ".join(
                f"{html.escape(row['name'])} ({row['paragraphs']})" for row in adversaries
            )
            + "</td></tr>"
        )
    return f"""<div class="adversary-succession" aria-label="Top named adversaries by era">
  {"".join(cards)}
</div>
<p class="summary-method-note"><strong>Derived alias boundary.</strong> Obvious variants such as
Britain/British and Tripoli/Barbary are joined by the published alias map; the frozen raw entity
labels are not changed. Ranking counts distinct paragraphs carrying an adversarial entity mention.
Some contextually surprising names remain visible because this is an audit of the instrument, not
a hand-edited list of historically approved enemies.</p>
<details class="summary-evidence-detail"><summary>Text alternative · top five named adversaries in every era</summary>
<div class="table-scroll"><table><thead><tr><th>Era</th><th>Names (distinct paragraphs)</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody></table></div></details>"""


def fig_summary_combat(summary_data: dict) -> go.Figure:
    frame = summary_data["combat"]
    era_order = list(frame.sort_values("era_order")["era"].drop_duplicates())
    labels = dict(zip(era_order, SUMMARY_ERA_SHORT))
    fig = go.Figure()
    specs = (
        ("enemy_naming", "Enemy named", "#9b4e50", "circle"),
        ("zero_sum", "Zero-sum frame", "#6d5a91", "diamond"),
        ("party_attack", "Partisan attack", "#315f78", "square"),
    )
    for flag, label, color, symbol in specs:
        rows = frame[frame["flag"].eq(flag)].sort_values("era_order")
        fig.add_trace(go.Scatter(
            x=[labels[name] for name in rows["era"]], y=rows["rate"].mul(100),
            mode="lines+markers", name=label,
            line=dict(color=color, width=2.4), marker=dict(color=color, symbol=symbol, size=9),
            customdata=np.stack([rows["n_paragraphs"], rows["n_flagged"]], axis=-1),
            hovertemplate=(
                f"<b>{label}</b><br>%{{x}}: %{{y:.1f}}% of paragraphs"
                "<br>%{customdata[1]:.0f} flagged / %{customdata[0]:.0f}<extra></extra>"
            ),
        ))
    fig.update_layout(**_layout(
        height=520, title=dict(text="Enemies are old; sustained partisan attack is newer", font=dict(size=15)),
        yaxis=dict(title="share of annual-message paragraphs", ticksuffix="%",
                   rangemode="tozero", gridcolor=GRID, linecolor=BASELINE),
        xaxis=dict(tickangle=-24, gridcolor=GRID, linecolor=BASELINE),
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=72, r=24, t=78, b=92),
    ))
    return fig


def fig_summary_temporal(
    contract: dict,
    faces: dict[str, str],
) -> go.Figure:
    """Relate president-level future and nostalgia rates with uniform portraits."""
    rows = contract["rows"].copy()
    available = rows[["future", "nostalgia"]].notna().all(axis=1)
    plotted = rows.loc[available].copy().sort_values(
        "display_order", kind="stable"
    )
    if plotted.empty:
        fig = go.Figure()
        fig.add_annotation(
            x=.5, y=.5, xref="paper", yref="paper",
            text="N/A · no president has both temporal measures available",
            showarrow=False, font=dict(size=13, color=MUTED),
        )
        fig.update_layout(**_layout(
            height=SUMMARY_TEMPORAL_CHART_HEIGHT_PX, showlegend=False,
            title=dict(
                text="Tomorrow language versus yesterday language",
                font=dict(size=15),
            ),
            margin=dict(l=78, r=28, t=70, b=66),
            xaxis=dict(
                title="future-family matches per 10,000 marker words", range=[0, 5],
                gridcolor=GRID, linecolor=BASELINE, fixedrange=True,
            ),
            yaxis=dict(
                title="nostalgia-family matches per 10,000 marker words", range=[0, 5],
                gridcolor=GRID, linecolor=BASELINE, fixedrange=True,
            ),
        ))
        return fig

    missing_faces = set(plotted["president"].astype(str)) - set(faces)
    if missing_faces:
        raise ValueError(
            "summary temporal portrait scatter is missing portraits: "
            + ", ".join(sorted(missing_faces))
        )
    portrait_diameters = np.full(len(plotted), SUMMARY_TEMPORAL_PORTRAIT_PX)
    x_ceiling = max(5.0, float(np.ceil(plotted["future"].max() / 5) * 5))
    y_ceiling = max(5.0, float(np.ceil(plotted["nostalgia"].max() / 5) * 5))
    x_padding = (
        x_ceiling * SUMMARY_TEMPORAL_PORTRAIT_PX
        / (
            2
            * (
                SUMMARY_TEMPORAL_PLOT_WIDTH_PX
                - SUMMARY_TEMPORAL_PORTRAIT_PX
            )
        )
    )
    y_padding = (
        y_ceiling * SUMMARY_TEMPORAL_PORTRAIT_PX
        / (
            2
            * (
                SUMMARY_TEMPORAL_PLOT_HEIGHT_PX
                - SUMMARY_TEMPORAL_PORTRAIT_PX
            )
        )
    )
    x_axis_span = x_ceiling + 2 * x_padding
    y_axis_span = y_ceiling + 2 * y_padding
    support_notes = [
        (
            f"Thin record · fewer than {contract['support_min_speeches']} speeches"
            if status == "thin_record" else "Observed record"
        )
        for status in plotted["support_status"].astype(str)
    ]
    fig = go.Figure(go.Scatter(
        x=plotted["future"],
        y=plotted["nostalgia"],
        mode="markers",
        marker=dict(
            size=portrait_diameters.tolist(),
            sizemode="diameter",
            symbol=["circle"] * len(plotted),
            color=["#fff8ec"] * len(plotted),
            opacity=1,
            line=dict(
                color=np.where(
                    plotted["support_status"].eq("observed"),
                    "#6f4828",
                    "#b16b3e",
                ).tolist(),
                width=np.where(
                    plotted["support_status"].eq("observed"), 3.0, 4.0
                ).tolist(),
            ),
        ),
        customdata=list(zip(
            plotted["president"].astype(str),
            plotted["first_year"].astype(int),
            plotted["last_year"].astype(int),
            plotted["era"].astype(str),
            plotted["n_speeches"].astype(int),
            plotted["n_rate_words"].astype(int),
            support_notes,
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b>"
            "<br><b>Years:</b> %{customdata[1]}–%{customdata[2]}"
            "<br><b>Story era:</b> %{customdata[3]}"
            "<br><b>Tomorrow:</b> %{x:.2f} per 10,000 marker words"
            "<br><b>Yesterday:</b> %{y:.2f} per 10,000 marker words"
            "<br><b>Speeches analyzed:</b> %{customdata[4]}"
            "<br><b>Marker words:</b> %{customdata[5]:,}"
            "<br><b>Support:</b> %{customdata[6]}<extra></extra>"
        ),
        cliponaxis=False,
        showlegend=False,
    ))
    for (_, row), diameter in zip(plotted.iterrows(), portrait_diameters):
        fig.add_layout_image(
            source=faces[str(row.president)],
            x=float(row.future),
            y=float(row.nostalgia),
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
            sizex=(
                x_axis_span * float(diameter) / SUMMARY_TEMPORAL_PLOT_WIDTH_PX
            ),
            sizey=(
                y_axis_span * float(diameter) / SUMMARY_TEMPORAL_PLOT_HEIGHT_PX
            ),
            sizing="contain",
            layer="above",
            opacity=1,
            name=f"portrait::{row.president}",
        )
    fig.update_layout(**_layout(
        height=SUMMARY_TEMPORAL_CHART_HEIGHT_PX,
        title=dict(
            text="Tomorrow language versus yesterday language",
            font=dict(size=15),
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#fffdf9",
            bordercolor="#d8d0c4",
            font=dict(family=FONT, color=INK2, size=11),
        ),
        showlegend=False,
        margin=dict(l=78, r=28, t=70, b=66),
        xaxis=dict(
            title="future-family matches per 10,000 marker words",
            range=[-x_padding, x_ceiling + x_padding],
            tickvals=list(np.arange(0, x_ceiling + .01, 5)),
            gridcolor=GRID, linecolor=BASELINE,
            showspikes=False, fixedrange=True,
        ),
        yaxis=dict(
            title="nostalgia-family matches per 10,000 marker words",
            range=[-y_padding, y_ceiling + y_padding],
            tickvals=list(np.arange(0, y_ceiling + .01, 5)),
            gridcolor=GRID, linecolor=BASELINE,
            showspikes=False, fixedrange=True,
        ),
    ))
    return fig


def _summary_temporal_timeline_frame(markers: pd.DataFrame) -> pd.DataFrame:
    """Average four consecutive annual future and nostalgia rates."""
    required = {"year", "n_words", "future", "nostalgia"}
    missing = required - set(markers.columns)
    if missing:
        raise ValueError(
            "summary temporal timeline is missing marker columns: "
            + ", ".join(sorted(missing))
        )
    numeric = markers[["year", "n_words", "future", "nostalgia"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.empty:
        raise ValueError("summary temporal timeline has no supported windows")
    if numeric.isna().any().any() or np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("summary temporal timeline marker values must be finite")
    if not np.allclose(numeric["year"], np.round(numeric["year"])):
        raise ValueError("summary temporal timeline years must be integers")
    count_columns = ["n_words", "future", "nostalgia"]
    if (numeric[count_columns] < 0).any().any():
        raise ValueError("summary temporal timeline marker counts cannot be negative")
    if not np.allclose(
        numeric[count_columns], np.round(numeric[count_columns])
    ):
        raise ValueError("summary temporal timeline marker counts must be integers")
    if (
        numeric["future"].gt(numeric["n_words"])
        | numeric["nostalgia"].gt(numeric["n_words"])
    ).any():
        raise ValueError(
            "summary temporal timeline family matches cannot exceed marker words"
        )
    numeric = numeric.astype(int)
    grouped = numeric.groupby("year")[count_columns].sum()
    first_year = max(int(grouped.index.min()), era_profiles.STORY_ERAS[0][1])
    last_year = min(int(grouped.index.max()), era_profiles.STORY_ERAS[-1][2])
    years = pd.RangeIndex(first_year, last_year + 1, name="year")
    grouped = grouped.reindex(years, fill_value=0)
    annual_rates = grouped[["future", "nostalgia"]].div(
        grouped["n_words"].where(grouped["n_words"] > 0), axis=0
    ) * 10_000
    rolling_rates = annual_rates.rolling(
        SUMMARY_TEMPORAL_TIMELINE_YEARS,
        min_periods=SUMMARY_TEMPORAL_TIMELINE_YEARS,
    ).mean()
    rolling_words = grouped["n_words"].rolling(
        SUMMARY_TEMPORAL_TIMELINE_YEARS,
        min_periods=SUMMARY_TEMPORAL_TIMELINE_YEARS,
    ).sum()
    rolling_rates = rolling_rates.iloc[3:].copy()
    rolling_words = rolling_words.iloc[3:].copy()
    supported = rolling_words.ge(SUMMARY_TEMPORAL_TIMELINE_MIN_WORDS)
    frame = pd.DataFrame({"n_words": rolling_words})
    for column in ("future", "nostalgia"):
        frame[column] = rolling_rates[column].where(supported)
    frame["window_start"] = frame.index - 3
    frame["window_end"] = frame.index
    frame.index.name = "window_end_year"
    if frame[["future", "nostalgia"]].dropna(how="all").empty:
        raise ValueError("summary temporal timeline has no supported windows")
    return frame


def fig_summary_temporal_timeline(markers: pd.DataFrame) -> go.Figure:
    """Plot future and nostalgia as trailing four-year rolling averages."""
    frame = _summary_temporal_timeline_frame(markers)
    first_end_year = int(frame.index.min())
    last_end_year = int(frame.index.max())
    windows = np.asarray([
        f"{int(row.window_start)}–{int(row.window_end)} window"
        for row in frame.itertuples()
    ])
    fig = go.Figure()
    for column, label, color, dash in (
        ("future", "Tomorrow", "#315f78", "solid"),
        ("nostalgia", "Yesterday", "#a24f52", "dash"),
    ):
        fig.add_trace(go.Scatter(
            x=frame.index,
            y=frame[column],
            mode="lines",
            name=label,
            connectgaps=False,
            line=dict(color=color, width=3, dash=dash),
            customdata=windows,
            hovertemplate=(
                f"<b>{label}</b>"
                "<br>%{customdata}"
                "<br>%{y:.2f} matches per 10,000 marker words"
                "<extra></extra>"
            ),
        ))
    era_annotation_labels = {
        "Admin-industrial": "Admin-<br>industrial",
        "Reform/collapse": "Reform/<br>collapse",
        "New Deal/WWII": "New Deal/<br>WWII",
        "Always-on": "Always-<br>on",
    }
    for index, ((_, start, end), short) in enumerate(zip(
        era_profiles.STORY_ERAS, SUMMARY_ERA_SHORT
    )):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=(
                "rgba(139,94,52,.035)" if index % 2 == 0
                else "rgba(49,95,120,.035)"
            ),
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=(start + end) / 2,
            y=1.025,
            xref="x",
            yref="paper",
            text=era_annotation_labels.get(short, short),
            showarrow=False,
            font=dict(size=9, color=MUTED),
        )
    ceiling = max(
        5.0,
        float(np.ceil(frame[["future", "nostalgia"]].max().max() / 5) * 5),
    )
    fig.update_layout(**_layout(
        height=SUMMARY_TEMPORAL_CHART_HEIGHT_PX,
        title=dict(
            text="Tomorrow and yesterday over time",
            font=dict(size=15),
            x=0,
            xanchor="left",
        ),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=78, r=28, t=92, b=66),
        xaxis=dict(
            title="ending year of supported four-year rolling average",
            range=[first_end_year, last_end_year],
            gridcolor=GRID,
            linecolor=BASELINE,
        ),
        yaxis=dict(
            title="matches per 10,000 marker words",
            range=[0, ceiling],
            gridcolor=GRID,
            linecolor=BASELINE,
        ),
    ))
    return fig


def fig_summary_hope_doom_ratio(markers: pd.DataFrame) -> go.Figure:
    grouped = markers.groupby("year")[["nrc_hope", "doom", "n_words"]].sum()
    years = pd.RangeIndex(
        int(markers["year"].min()), int(markers["year"].max()) + 1, name="year"
    )
    grouped = grouped.reindex(years, fill_value=0)
    rolling = grouped.rolling(5, center=True, min_periods=1).sum()
    supported = rolling["n_words"].ge(20_000) & rolling["doom"].gt(0)
    hope_rate = rolling["nrc_hope"] / rolling["n_words"] * 10_000
    doom_rate = rolling["doom"] / rolling["n_words"] * 10_000
    ratio = (rolling["nrc_hope"] / rolling["doom"]).where(supported)
    custom = np.stack([hope_rate, doom_rate, rolling["n_words"]], axis=-1)
    fig = go.Figure(go.Scatter(
        x=ratio.index, y=ratio, mode="lines", connectgaps=False,
        line=dict(color="#4f7557", width=3), customdata=custom,
        hovertemplate=(
            "<b>%{x}</b><br>hope matches per doom match %{y:.1f}"
            "<br>NRC hope %{customdata[0]:.2f} per 10,000"
            "<br>doom %{customdata[1]:.2f} per 10,000"
            "<br>%{customdata[2]:,.0f} words in window<extra></extra>"
        ), showlegend=False,
    ))
    for index, (year, label) in enumerate(SUMMARY_RATIO_EVENTS):
        fig.add_vline(x=year, line_color="#a79d90", line_dash="dot", line_width=1)
        fig.add_annotation(
            x=year, y=.98 if index % 2 == 0 else .88, xref="x", yref="paper",
            text=label, textangle=-90, showarrow=False,
            font=dict(size=9, color="#6b6259"), xanchor="right",
        )
    fig.update_layout(**_timeline_layout(
        height=570,
        title=dict(text="Hope ÷ doom · supported centered five-year windows", font=dict(size=15)),
        yaxis=dict(
            title="NRC hope matches per doom match", type="log",
            tickvals=[50, 100, 200, 500, 1000],
            ticktext=["50", "100", "200", "500", "1,000"],
            gridcolor=GRID, linecolor=BASELINE,
        ),
        margin=dict(l=72, r=25, t=70, b=55),
    ))
    return fig


def _stats_president_year(stats: pd.DataFrame, cols: list[str],
                          min_tokens: int = 4_000) -> pd.DataFrame:
    g = stats.groupby(["president", "year"])
    toks = g["n_tokens"].sum()
    out = g[cols].sum().div(toks, axis=0) * 10_000
    out = out.reset_index()
    out["n_speeches"] = g.size().values
    out["n_tokens"] = toks.values
    return out[out["n_tokens"] >= min_tokens].reset_index(drop=True)


FACE_CHART_WIDTH = 920  # fixed so portrait circles stay circular


def _face_chart(scores: pd.DataFrame, value_col: str, ytitle: str, faces: dict,
                hover_fmt: str = "%{customdata[1]:.2f}",
                trend: pd.Series | None = None,
                trend_name: str = "corpus rolling rate",
                segments: bool = True) -> go.Figure:
    """Every president as their portrait, sitting at their score. With
    segments=True the line under each face spans their years in the corpus.
    The `face` column (if present) picks the portrait, so a president split
    into two terms can appear twice with the same face."""
    fig = go.Figure()
    if trend is not None:
        fig.add_trace(go.Scatter(
            x=trend.index, y=trend.values, mode="lines", name=trend_name,
            line=dict(color=GRID, width=2), connectgaps=False,
            hovertemplate="%{y:.2f}<extra>" + trend_name + "</extra>",
        ))
    vals = scores[value_col]
    y_span = float(vals.max() - vals.min()) or 1.0
    face_h = y_span * 1.35 * 30 / 430  # ~30px at chart height
    for pres, row in scores.iterrows():
        v = float(row[value_col])
        x0, x1 = int(row["first_year"]), int(row["last_year"])
        mid = (x0 + x1) / 2
        color = PARTY_COLORS.get(row["party"], MUTED)
        if segments:
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[v, v], mode="lines", showlegend=False,
                line=dict(color=color, width=2.5), hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=[mid], y=[v], mode="markers", showlegend=False,
            marker=dict(size=26, opacity=0),
            customdata=[[pres, v]],
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          + hover_fmt + "<extra></extra>",
        ))
        face_key = row.get("face", pres)
        if face_key in faces:
            fig.add_layout_image(
                source=faces[face_key], x=mid, y=v, xref="x", yref="y",
                sizex=8.2, sizey=face_h, xanchor="center", yanchor="middle",
                layer="above",
            )
    layout = _timeline_layout(height=560, width=FACE_CHART_WIDTH,
                              yaxis_title=ytitle, showlegend=trend is not None)
    pad = y_span * 0.09
    layout["yaxis"]["range"] = [float(vals.min()) - pad, float(vals.max()) + pad]
    fig.update_layout(**layout)
    return fig


# Presidents with two non-consecutive terms: (label suffix, year the term
# window ends/starts). Each term gets its own face on the certainty chart.
SPLIT_TERMS = {
    "Grover Cleveland": 1891,
    "Donald Trump": 2023,
}


def _certainty_scores_split(scores: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    """Per-president certainty rows, with split-term presidents appearing
    once per term."""
    rows = scores.copy()
    rows["face"] = rows.index
    for pres, cut in SPLIT_TERMS.items():
        if pres not in rows.index:
            continue
        base = rows.loc[pres]
        rows = rows.drop(pres)
        for label, mask_fn in [
            (f"{pres} (1st term)", lambda y: y < cut),
            (f"{pres} (2nd term)", lambda y: y >= cut),
        ]:
            m = markers[(markers["president"] == pres)
                        & markers["year"].map(mask_fn)]
            if not len(m):
                continue
            assertive = (m["boosters"] + m["assertive_modals"]).sum()
            deliberative = (m["hedges"] + m["concessives"]).sum()
            new = base.copy()
            new["certainty"] = assertive / max(assertive + deliberative, 1)
            new["first_year"] = int(m["year"].min())
            new["last_year"] = int(m["year"].max())
            new["face"] = pres
            rows.loc[label] = new
    return rows


def fig_certainty_faces(scores: pd.DataFrame, markers: pd.DataFrame,
                        faces: dict) -> go.Figure:
    split = _certainty_scores_split(scores, markers)
    return _face_chart(
        split, "certainty", "assertive share of stance markers", faces,
        hover_fmt="assertive share %{customdata[1]:.2f}",
        trend=indices.certainty_yearly(markers),
        segments=False,
    )


def fig_pronoun_faces(scores: pd.DataFrame, faces: dict) -> go.Figure:
    s = scores.copy()
    s["i_share"] = s["self_reference"] * 100
    return _face_chart(
        s, "i_share", "share of first-person that is “I” (%)", faces,
        hover_fmt="“I” share %{customdata[1]:.0f}%",
        segments=False,
    )


I_DELIBERATIVE = {"believe", "think", "hope", "trust", "urge", "recommend",
                  "propose", "ask", "submit", "doubt", "suppose", "wish"}
I_SELF = {"am", "was", "have", "had", "did", "know", "made", "got", "won",
          "built", "say", "do", "'m", "'ve"}
_I_NEXT_RE = re.compile(r"\bi ([a-z']+)")


def fig_after_i(df: pd.DataFrame) -> go.Figure:
    """What follows 'I': epistemic framing vs self/state assertion."""
    rows = []
    for _, sp in df.iterrows():
        nxt = _I_NEXT_RE.findall(sp["transcript"].lower())
        rows.append({"year": sp["year"], "total": len(nxt),
                     "delib": sum(1 for w in nxt if w in I_DELIBERATIVE),
                     "self": sum(1 for w in nxt if w in I_SELF)})
    t = pd.DataFrame(rows).groupby("year").sum()
    years = pd.RangeIndex(int(df["year"].min()), int(df["year"].max()) + 1)
    t = t.reindex(years, fill_value=0)
    w = t.rolling(7, center=True, min_periods=1).sum()
    fig = go.Figure()
    for i, (name, col) in enumerate([
            ("“I believe / think / hope / recommend …”", "delib"),
            ("“I am / have / did / won …”", "self")]):
        share = (w[col] / w["total"] * 100).where(w["total"] >= 150)
        fig.add_trace(go.Scatter(
            x=share.index, y=share.values, name=name, mode="lines",
            line=dict(color=SERIES[i], width=2.4), connectgaps=False,
            hovertemplate="%{y:.0f}% of “I …”<extra>" + name + "</extra>",
        ))
    fig.update_layout(**_timeline_layout(
        yaxis_title="share of everything that follows “I” (%)"))
    return fig


def fig_readability(stats: pd.DataFrame, speech_annotations: pd.DataFrame) -> go.Figure:
    """Reading difficulty beside the corpus's changing delivery media."""
    medium_labels = {
        "written_message": "Written message",
        "spoken_address": "Spoken address",
        "broadcast_radio_or_tv": "Radio / TV broadcast",
        "press_conference": "Press conference",
        "debate": "Debate",
    }
    medium_colors = {
        "Written message": "#9a7b55",
        "Spoken address": "#4e7fa1",
        "Radio / TV broadcast": "#b45f4b",
        "Press conference": "#6f9271",
        "Debate": "#8a6ca3",
    }
    s = stats.merge(
        speech_annotations[["doc_name", "medium"]],
        on="doc_name", how="left", validate="one_to_one",
    )
    s["medium_label"] = s.medium.map(medium_labels).fillna("Unclassified")
    s["fk"] = s["fk_grade"].clip(0, 30)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        row_heights=[0.70, 0.30],
    )
    for medium in medium_labels.values():
        group = s[s.medium_label.eq(medium)]
        if group.empty:
            continue
        fig.add_trace(go.Scatter(
            x=group["year"], y=group["fk"], mode="markers", name=medium,
            legendgroup=medium,
            marker=dict(color=medium_colors[medium], size=5, opacity=0.48),
            customdata=np.stack([
                group["president"], group["title"].str.slice(0, 80)
            ], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                          "<br>grade %{y:.1f}<extra>" + medium + "</extra>",
        ), row=1, col=1)
    med = (s.groupby("year")["fk"].median()
           .reindex(pd.RangeIndex(int(s["year"].min()), int(s["year"].max()) + 1))
           .rolling(7, center=True, min_periods=3).median())
    fig.add_trace(go.Scatter(
        x=med.index, y=med.values, mode="lines", name="All-speech rolling median",
        line=dict(color=INK, width=3), connectgaps=False,
        hovertemplate="median grade %{y:.1f}<extra></extra>",
    ), row=1, col=1)
    s["period"] = (s["year"] // 10) * 10
    mix = (
        pd.crosstab(s.period, s.medium_label, normalize="index") * 100
    ).reindex(columns=list(medium_labels.values()), fill_value=0)
    for medium in medium_labels.values():
        fig.add_trace(go.Bar(
            x=mix.index + 5, y=mix[medium], name=medium,
            legendgroup=medium, showlegend=False,
            marker=dict(color=medium_colors[medium]),
            hovertemplate="%{x}<br>%{y:.0f}% of corpus speeches<extra>"
                          + medium + "</extra>",
        ), row=2, col=1)
    fig.update_layout(**_timeline_layout(
        height=650, barmode="stack",
        legend=dict(orientation="h", y=1.08, x=0),
    ))
    fig.update_yaxes(
        title_text="Flesch–Kincaid grade", gridcolor=GRID,
        linecolor=BASELINE, row=1, col=1,
    )
    fig.update_yaxes(
        title_text="corpus medium mix (%)", range=[0, 100],
        gridcolor=GRID, linecolor=BASELINE, row=2, col=1,
    )
    fig.update_xaxes(
        range=X_RANGE, gridcolor=GRID, linecolor=BASELINE, row=2, col=1
    )
    return fig


def _small_multiples(panels: list[tuple[str, pd.Series]], rows: int, cols: int,
                     height: int, hovertemplate: str,
                     dots: dict | None = None,
                     dots_unit: str = "",
                     events: dict[str, list[tuple[int, str]]] | None = None) -> go.Figure:
    """dots: optional {panel_title: DataFrame(x, y, name)} overlay of
    president-attributed points on each panel."""
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True,
                        subplot_titles=[t for t, _ in panels],
                        vertical_spacing=0.09, horizontal_spacing=0.06)
    for k, (title, series) in enumerate(panels):
        r, c = divmod(k, cols)
        if dots and title in dots:
            d = dots[title]
            fig.add_trace(go.Scatter(
                x=d["x"], y=d["y"], mode="markers", showlegend=False,
                marker=dict(color=BLUE_RAMP[5], size=4, opacity=0.4),
                customdata=d["name"],
                hovertemplate="<b>%{customdata}</b>: %{y:.1f}" + dots_unit
                              + "<extra>" + title + "</extra>",
            ), row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines",
            line=dict(color=BLUE_RAMP[4], width=2),
            fill="tozeroy", fillcolor="rgba(109, 167, 236, 0.35)",
            hovertemplate=hovertemplate + "<extra>" + title + "</extra>",
            showlegend=False,
        ), row=r + 1, col=c + 1)
        axis_no = k + 1
        xref = "x" if axis_no == 1 else f"x{axis_no}"
        yref = "y domain" if axis_no == 1 else f"y{axis_no} domain"
        for event_no, (year, label) in enumerate((events or {}).get(title, [])[:2], 1):
            fig.add_shape(
                type="line", x0=year, x1=year, y0=0, y1=.93,
                xref=xref, yref=yref, layer="below",
                line=dict(color="#8b6c42", width=1, dash="dot"),
                label=dict(
                    text=str(event_no), textangle=0,
                    textposition="top right",
                    font=dict(size=8, color="#735a39"),
                ),
            )
    fig.update_layout(**_layout(height=height, margin=dict(l=40, r=16, t=48, b=36)))
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED, size=10))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED, size=10),
                     rangemode="tozero")
    fig.update_annotations(font=dict(size=12.5, color=INK))
    return fig


# One string for both branches. The dots are per (president, PERIOD) now, and a
# hover reading "12.3% of their paragraphs" invites a reader to hear a
# whole-presidency share — a different quantity from the one plotted. The issue
# pages were updated when the dots changed and the dashboard was not; sharing
# the constant is what stops the two renderings of one function drifting again.
DOTS_UNIT = "% of their paragraphs in this period"

# Context markers are deliberately sparse and non-causal. They help a reader
# orient a spike in time without claiming the event produced the measured
# language. Every panel keeps its own scale, so the annotations emphasize when,
# not how high one issue was relative to another.
ISSUE_EVENTS = {
    "Agriculture": [(1862, "Homestead & Morrill Acts"), (1933, "New Deal farm policy")],
    "Civil rights & race": [(1863, "Emancipation"), (1964, "Civil Rights Act")],
    "Crime & justice": [(1968, "Law-and-order election"), (1994, "Crime bill")],
    "Economy & jobs": [(1929, "Market crash"), (2008, "Financial crisis")],
    "Education": [(1954, "Brown v. Board"), (1965, "Federal education law")],
    "Energy & environment": [(1973, "Oil shock"), (2022, "Climate law")],
    "Foreign policy": [(1823, "Monroe Doctrine"), (1947, "Truman Doctrine")],
    "Health care": [(1965, "Medicare & Medicaid"), (2010, "Affordable Care Act")],
    "Immigration": [(1924, "Immigration Act"), (1965, "Hart–Celler Act")],
    "Infrastructure": [(1956, "Interstate highways")],
    "Money & banking": [(1832, "Bank War"), (1933, "Bank holiday")],
    "Religion & values": [(1954, "“Under God” added")],
    "Discovered 5": [(1947, "National Security Act"), (2001, "September 11")],
    "Taxes & budget": [(1913, "Federal income tax"), (1981, "Reagan tax cuts")],
    "Trade & tariffs": [(1930, "Smoot–Hawley"), (2018, "New tariff cycle")],
    "War & military": [(1861, "Civil War"), (1941, "Pearl Harbor")],
}

WORD_EVENTS = {
    "democracy": [(1917, "War message: “safe for democracy”"), (2021, "January 6")],
    "freedom / liberty": [(1863, "Emancipation"), (1941, "Four Freedoms")],
    "God": [(1954, "“Under God” added"), (2001, "September 11")],
    "economy / jobs": [(1929, "Market crash"), (2008, "Financial crisis")],
    "war": [(1812, "War of 1812"), (1941, "Pearl Harbor")],
    "immigration": [(1924, "Immigration Act"), (1965, "Hart–Celler Act")],
    "tariff / trade": [(1930, "Smoot–Hawley"), (2018, "New tariff cycle")],
    "constitution": [(1861, "Secession crisis"), (1974, "Watergate")],
    "border": [(1848, "Mexican Cession"), (2019, "Border emergency")],
}


def _event_guide_html(events: dict[str, list[tuple[int, str]]]) -> str:
    """Readable companion to numbered plot markers; markers are contextual."""
    rows = []
    for label, items in events.items():
        marks = " · ".join(
            f"<span><b>{i}</b> {year} · {html.escape(event)}</span>"
            for i, (year, event) in enumerate(items, 1)
        )
        rows.append(f"<li><strong>{html.escape(label)}</strong> — {marks}</li>")
    return (
        '<details class="event-guide"><summary>Event guide for the numbered markers</summary>'
        '<p>Markers orient the date of a rise or fall; they do not say the event caused '
        'the measured language.</p><ul>' + "".join(rows) + "</ul></details>"
    )


def _llm_events(topic: str) -> list[tuple[int, str]]:
    """Return at most two useful context markers for a fine AI topic."""
    low = topic.casefold()
    rules = [
        (("slavery", "emancipat"), (1863, "Emancipation")),
        (("civil rights", "racial"), (1964, "Civil Rights Act")),
        (("bank", "finance"), (1933, "Bank holiday")),
        (("world war", "military"), (1941, "Pearl Harbor")),
        (("cold war", "soviet", "nuclear"), (1947, "Cold War begins")),
        (("health", "medicare"), (1965, "Medicare & Medicaid")),
        (("immigration", "border"), (1965, "Hart–Celler Act")),
        (("terror", "homeland"), (2001, "September 11")),
        (("climate", "energy", "oil"), (1973, "Oil shock")),
        (("econom", "labor", "employment"), (1929, "Market crash")),
        (("trade", "tariff"), (1930, "Smoot–Hawley")),
        (("pandemic", "covid"), (2020, "COVID-19")),
    ]
    matches = [event for needles, event in rules if any(n in low for n in needles)]
    return matches[:2]


def _x_range_covering(panels: list[tuple[str, pd.DataFrame | None]]) -> list:
    """Grid-shaped adapter over `issues_site.x_range_covering`.

    The rule itself lives in `issues_site` next to `X_MIN`/`X_MAX`, so the
    single-issue figure and both dashboard grids share one implementation
    instead of two that can drift apart on the bound that matters.
    """
    return issues_site.x_range_covering([b for _, b in panels])


def _wrap_panel_title(title: str, per_line: int, max_lines: int = 2) -> str:
    """Word-wrap a subplot title so a long topic name is not clipped mid-word.

    Plotly clips a subplot title at the panel's width, which turned "Partisan
    Combat, Press Conferences & Media Attacks" into "…& Media Attac" — a string
    that reads as a rendering bug rather than as a shortened label. Titles are
    wrapped onto at most `max_lines` lines with `<br>`; anything still over
    length is truncated with a real ellipsis, which reads as deliberate.

    Only the DISPLAYED title is shortened. The hover tooltip keeps the full
    name, so nothing is actually lost.
    """
    words, lines, current = title.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > per_line:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                break
        else:
            current = candidate
    if len(lines) < max_lines:
        lines.append(current)
        current = ""
    if current:                                  # ran out of lines mid-title
        last = lines[-1]
        keep = max(0, per_line - 1)
        lines[-1] = (last[:keep].rstrip() if len(last) > keep else last) + "…"
    return "<br>".join(line for line in lines if line)


def _banded_small_multiples(panels: list[tuple[str, pd.DataFrame]], rows: int,
                            cols: int, height: int, unit: str,
                            dots: dict | None = None,
                            dots_unit: str = "",
                            x_range: list | None = None,
                            show_components: bool = False,
                            events: dict[str, list[tuple[int, str]]] | None = None) -> go.Figure:
    """Small multiples of banded trend lines.

    `panels` are `(title, band_frame)` where the frame is a `bands.parquet`
    slice (`x`, `point`, `lo`, `hi`, `ci_status`, `n_speeches`). Trace order is
    band, then line, then the unresolved-interval rings, then dots, because
    plotly paints later traces on top and a band over its own dots hides them.
    The rings go after the line so they sit on top of the dotted trend they
    annotate, and before the dots so a president marker is never hidden.
    """
    per_line = 34 if cols <= 3 else 26
    titles = [_wrap_panel_title(t, per_line) for t, _ in panels]
    # A wrapped title grows UPWARD from its panel, so the top row needs more
    # margin — but only when something actually wrapped, so a grid of short
    # titles (the 16 CorEx issues) keeps its existing layout untouched.
    top_margin = 62 if any("<br>" in t for t in titles) else 48
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True,
                        subplot_titles=titles,
                        vertical_spacing=0.09, horizontal_spacing=0.06)
    for k, (title, band) in enumerate(panels):
        r, c = divmod(k, cols)
        # `bands.series_band` returns None for a series absent from the table —
        # the seam's PARTIAL failure mode, which a stale `bands.parquet`
        # predating a CorEx refit (a renamed or added issue) produces. Whole-file
        # absence already degrades gracefully and schema drift already raises a
        # rebuild message; without this, a per-series miss instead died as an
        # opaque TypeError inside `_band_hover`. The panel keeps its dots and
        # its title, and simply carries no line.
        if band is not None and not band.empty:
            for trace in issues_site.band_traces(band):
                fig.add_trace(trace, row=r + 1, col=c + 1)
            for trace in issues_site.line_traces(band, unit=unit, width=2,
                                                 suffix=title,
                                                 show_components=show_components):
                fig.add_trace(trace, row=r + 1, col=c + 1)
            # The hollow ring for cells whose bootstrap resolved no interval.
            # A suppressed zero-width band is invisible by definition, so
            # without this the grid would render the change as no change at
            # all. Slightly smaller than on the single-issue page: these panels
            # are a quarter the width.
            for trace in issues_site.unresolved_traces(band, size=7,
                                                       unit=unit,
                                                       suffix=title):
                fig.add_trace(trace, row=r + 1, col=c + 1)
        top = issues_site.band_y_top(band)
        if dots and title in dots:
            d = dots[title]
            fig.add_trace(go.Scatter(
                x=d["x"], y=d["y"], mode="markers", showlegend=False,
                marker=dict(color=BLUE_RAMP[5], size=4, opacity=0.4),
                customdata=d["name"],
                hovertemplate="<b>%{customdata}</b>: %{y:.1f}" + dots_unit
                              + "<extra>" + title + "</extra>",
            ), row=r + 1, col=c + 1)
            if top is not None and len(d):
                top = max(top, float(d["y"].max()) * 1.05)
        # Per-panel ceiling, so one 2-speech period's enormous interval cannot
        # flatten its neighbours' 240 years of real movement. See
        # `issues_site.band_y_top` for why overflow is the intended reading.
        fig.update_yaxes(range=None if top is None else [0, top],
                         row=r + 1, col=c + 1)
        axis_no = k + 1
        xref = "x" if axis_no == 1 else f"x{axis_no}"
        yref = "y domain" if axis_no == 1 else f"y{axis_no} domain"
        for event_no, (year, label) in enumerate((events or {}).get(title, [])[:2], 1):
            fig.add_shape(
                type="line", x0=year, x1=year, y0=0, y1=.94,
                xref=xref, yref=yref, layer="below",
                line=dict(color="#8b6c42", width=1, dash="dot"),
                label=dict(
                    text=str(event_no), textangle=0,
                    textposition="top right",
                    font=dict(size=8, color="#735a39"),
                ),
            )
    fig.update_layout(**_layout(height=height, margin=dict(l=40, r=16, t=top_margin, b=36)))
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE,
                     tickfont=dict(color=MUTED, size=10), range=x_range)
    # `rangemode` is the fallback for panels with no band to scale to; it is
    # inert on the panels that got an explicit [0, top] range above.
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, rangemode="tozero",
                     tickfont=dict(color=MUTED, size=10))
    fig.update_annotations(font=dict(size=12.5, color=INK))
    return fig


def fig_issues_decade(para_labels: pd.DataFrame, issue_names: list[str],
                      band_table: pd.DataFrame | None = None) -> go.Figure:
    """Share of paragraphs touching each curated issue, in 5-year periods.

    Each panel carries a speech-clustered 95% sampling band and one dot per
    (president, period). `band_table` is `bands.parquet`; when it is absent the
    panels fall back to the unbanded line rather than to an invented interval.
    These are CorEx labels, so the intervals are sampling-only by construction —
    there is no annotator in that pipeline whose disagreement could widen them.
    """
    pl = para_labels.copy()
    # Resolved once and shared by both branches: a discovered topic's panel
    # title is its hand-edited display name, and having the two branches look
    # that up separately is how one of them silently keeps the raw column name.
    labelled = [(name, profiles_site.DISCOVERED_LABELS.get(name, name))
                for name in topic_quality.display_issues(issue_names)]

    dots = {label: issues_site.president_period_dots(pl, name)
            for name, label in labelled}

    if band_table is None:
        # The `40` stays a bare literal on purpose: this branch preserves the
        # PRE-BAND behaviour, so it must keep applying the historic hard mask
        # even if `bands.MIN_PERIOD_PARAGRAPHS` is ever retuned as a
        # de-emphasis threshold. The period grain is genuinely shared with the
        # banded branch, so that one does read the constant (as
        # `issues_site.fig_issue_timeline`'s matching fallback already does).
        pl["period"] = (pl["year"] // bands.PERIOD_YEARS) * bands.PERIOD_YEARS
        counts = pl.groupby("period").size()
        valid = counts[counts >= 40].index
        panels = []
        for name, label in labelled:
            share = pl.groupby("period")[name].mean() * 100
            panels.append((label, share.loc[share.index.isin(valid)]))
        return _small_multiples(panels, rows=4, cols=4, height=880,
                                hovertemplate="%{y:.1f}% of paragraphs",
                                dots=dots, dots_unit=DOTS_UNIT)

    panels = [(label, bands.series_band(band_table, bands.COREX_SURFACE, name))
              for name, label in labelled]
    issue_events = dict(ISSUE_EVENTS)
    for source_name, display_label in labelled:
        if source_name in ISSUE_EVENTS:
            issue_events[display_label] = ISSUE_EVENTS[source_name]
    return _banded_small_multiples(panels, rows=4, cols=4, height=880,
                                   unit="% of paragraphs", dots=dots,
                                   dots_unit=DOTS_UNIT,
                                   x_range=_x_range_covering(panels),
                                   events=issue_events)


def fig_llm_topic_bands(band_table: pd.DataFrame, top_n: int = 12,
                        rows: int = 4, cols: int = 3) -> go.Figure:
    """Surface B: the LLM topic layer, banded with BOTH uncertainty sources.

    Unlike the CorEx charts above, these labels were produced by an AI annotator
    (`paragraph_annotations.topics`, level 2 of `taxonomy_v1`), so a second
    annotator's disagreement is genuine uncertainty about the plotted quantity
    and is composed into the band on top of the speech-clustered sampling
    interval. Panels are the `top_n` topics ranked by share of ALL corpus
    paragraphs (each era's share weighted by that era's paragraph count, so a
    topic cannot rank high off one thin era), which makes the selection a
    property of the data rather than a curated list.
    """
    llm = band_table[band_table["surface"] == bands.LLM_SURFACE]
    weight = llm["point"] * llm["n_paragraphs"]
    corpus_share = (
        weight.groupby(llm["series"]).sum()
        / llm.groupby("series")["n_paragraphs"].sum()
    )
    chosen = corpus_share.nlargest(top_n).index.tolist()
    panels = [
        (topic, bands.series_band(band_table, bands.LLM_SURFACE, topic))
        for topic in chosen
    ]
    events = {topic: _llm_events(topic) for topic in chosen}
    return _banded_small_multiples(panels, rows=rows, cols=cols, height=940,
                                   unit="% of paragraphs", x_range=X_RANGE,
                                   show_components=True, events=events)


def fig_llm_topic_change(band_table: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Largest descriptive AI-topic shifts from 1989–2016 to 2017–2026.

    A ranked change view answers the synthesis question better than twelve tiny
    timelines. The source bands describe uncertainty around each era estimate;
    they are not a confidence interval for the difference, so this chart does
    not imply one.
    """
    llm = band_table[band_table["surface"] == bands.LLM_SURFACE].copy()
    prior_x, present_x = sorted(llm["x"].dropna().unique())[-2:]
    wide = llm[llm["x"].isin([prior_x, present_x])].pivot(
        index="series", columns="x", values="point"
    ).dropna()
    wide["change"] = (wide[present_x] - wide[prior_x]) * 100
    chosen = wide.reindex(wide.change.abs().nlargest(top_n).index).sort_values("change")
    colors = ["#9b4e50" if value < 0 else "#2f7397" for value in chosen["change"]]
    labels = [_wrap_panel_title(name, 31, max_lines=2) for name in chosen.index]
    fig = go.Figure(go.Bar(
        x=chosen["change"], y=labels, orientation="h",
        marker=dict(color=colors),
        customdata=np.stack([
            chosen[prior_x] * 100, chosen[present_x] * 100
        ], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>1989–2016: %{customdata[0]:.1f}%"
            "<br>2017–2026: %{customdata[1]:.1f}%"
            "<br>change: %{x:+.1f} percentage points<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_color=BASELINE, line_width=1.2)
    fig.update_layout(**_layout(
        height=560, showlegend=False,
        margin=dict(l=245, r=30, t=34, b=62),
        xaxis=dict(title="change in share of paragraphs (percentage points)",
                   gridcolor=GRID, linecolor=BASELINE, zeroline=False),
        yaxis=dict(gridcolor=SURFACE, linecolor=BASELINE,
                   tickfont=dict(size=11, color=INK2)),
    ))
    return fig


def fig_progressive_crises(para_labels: pd.DataFrame) -> go.Figure:
    """Shared-scale historical windows for the 1913–1932 chapter."""
    topics = ["Economy & jobs", "Money & banking", "Trade & tariffs",
              "War & military", "Civil rights & race"]
    sub = para_labels[para_labels.year.between(1913, 1932)].copy()
    sub["period"] = pd.cut(
        sub["year"], [1912, 1918, 1928, 1932],
        labels=["1913–18", "1919–28", "1929–32"],
        include_lowest=True,
    )
    grouped = sub.groupby("period", observed=False)[topics].mean() * 100
    fig = go.Figure()
    colors = ["#275d8c", "#7f6b9a", "#b17a43", "#9b4e50", "#52805b"]
    for topic, color in zip(topics, colors):
        fig.add_trace(go.Bar(
            name=topic, x=grouped.index.astype(str), y=grouped[topic],
            marker_color=color,
            hovertemplate="%{x}<br>%{y:.1f}% of paragraphs<extra>" + topic + "</extra>",
        ))
    fig.update_layout(**_layout(
        height=470, barmode="group",
        margin=dict(l=58, r=20, t=64, b=72),
        xaxis=dict(title="historical period", gridcolor=SURFACE,
                   linecolor=BASELINE),
        yaxis=dict(title="% of paragraphs", rangemode="tozero",
                   gridcolor=GRID, linecolor=BASELINE),
        legend=dict(orientation="h", y=1.03, x=0, font=dict(size=11)),
    ))
    return fig


def fig_new_deal_war(para_labels: pd.DataFrame) -> go.Figure:
    """Shared-scale agenda movement across recovery, war, and settlement."""
    topics = ["Economy & jobs", "Agriculture", "Money & banking",
              "Infrastructure", "War & military", "Foreign policy"]
    phases = [
        ("1933–39 · recovery", 1933, 1939),
        ("1940–45 · mobilization", 1940, 1945),
        ("1946–52 · settlement", 1946, 1952),
    ]
    grouped = pd.DataFrame({
        label: (
            para_labels[para_labels.year.between(lo, hi)][topics].mean() * 100
        )
        for label, lo, hi in phases
    }).reindex(topics)
    fig = go.Figure()
    symbols = {
        "Economy & jobs": "hexagon",
        "Agriculture": "triangle-up",
        "Money & banking": "square",
        "Infrastructure": "cross",
        "War & military": "diamond-tall",
        "Foreign policy": "star",
    }
    colors = {
        "Economy & jobs": "#315f78",
        "Agriculture": "#667342",
        "Money & banking": "#8b6c42",
        "Infrastructure": "#6d5a88",
        "War & military": "#a24f52",
        "Foreign policy": "#2f7397",
    }
    phase_labels = list(grouped.columns)
    phase_speech_counts = np.array([
        para_labels[para_labels.year.between(lo, hi)]["doc_name"].nunique()
        for _label, lo, hi in phases
    ])
    for topic in topics:
        values = grouped.loc[topic]
        fig.add_trace(go.Scatter(
            x=phase_labels,
            y=values,
            mode="lines+markers",
            name=topic,
            line=dict(color=colors[topic], width=2.8),
            marker=dict(
                color=colors[topic],
                size=11,
                symbol=symbols[topic],
                line=dict(color="white", width=1),
            ),
            customdata=np.column_stack([
                (values - float(values.iloc[0])).to_numpy(),
                phase_speech_counts,
            ]),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>%{x}"
                "<br>%{y:.2f}% of paragraphs"
                "<br>%{customdata[0]:+.2f} points from 1933–39"
                "<br>%{customdata[1]:.0f} speeches in phase<extra></extra>"
            ),
        ))
    fig.update_layout(**_layout(
        height=500,
        margin=dict(l=68, r=24, t=68, b=82),
        title=dict(
            text="Paragraph shares across recovery, mobilization, and postwar settlement",
            font=dict(size=14),
        ),
        xaxis=dict(title="historical phase", gridcolor=SURFACE,
                   linecolor=BASELINE),
        yaxis=dict(title="% of paragraphs (one shared scale)", rangemode="tozero",
                   gridcolor=GRID, linecolor=BASELINE),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
    ))
    return fig


def new_deal_takeaway(para_labels: pd.DataFrame) -> str:
    topics = ["Economy & jobs", "Agriculture", "Money & banking",
              "Infrastructure", "War & military", "Foreign policy"]
    shares = pd.DataFrame({
        "recovery": (
            para_labels[para_labels.year.between(1933, 1939)][topics].mean()
            * 100
        ),
        "mobilization": (
            para_labels[para_labels.year.between(1940, 1945)][topics].mean()
            * 100
        ),
        "settlement": (
            para_labels[para_labels.year.between(1946, 1952)][topics].mean()
            * 100
        ),
    })
    war_rise = shares.loc["War & military", "mobilization"] - shares.loc[
        "War & military", "recovery"
    ]
    war_after = shares.loc["War & military", "settlement"] - shares.loc[
        "War & military", "mobilization"
    ]
    foreign_settlement = shares.loc["Foreign policy", "settlement"]
    return (
        "The wider 1933–1952 chapter contains three rhetorical phases. "
        f"War/military rises {war_rise:+.1f} percentage points from recovery "
        f"to mobilization, then moves {war_after:+.1f} points in the 1946–52 "
        f"settlement; foreign policy accounts for {foreign_settlement:.1f}% "
        "of that postwar phase's paragraphs."
    )


def llm_coverage_sentence() -> str:
    """How much of the corpus the second annotator read, DERIVED from the artifacts.

    The section's prose said "every paragraph … then read again by a second
    model", whose subject is every paragraph — claiming a fully double-annotated
    corpus. The second reader covers a quarter of it. That single number governs
    how much weight the annotator component deserves, and this is the one
    surface a reader actually sees, so it is stated here rather than left to
    `bands.py`'s docstring and `bands_meta.json`.

    Derived, not typed: `bands.paired_coverage()` recomputes it from the
    annotation tables, so the claim cannot drift from the artifact. The transfer
    assumption is named in the same breath, because a 262-speech measurement is
    what widens all nine full-corpus era bands.
    """
    c = bands.paired_coverage()
    return (
        f"That second reading is a sample, not a re-run: it covers "
        f"{c['n_paired_paragraphs']:,} of {c['n_corpus_paragraphs']:,} "
        f"paragraphs ({c['paragraph_fraction']:.1%}), drawn as whole speeches "
        f"- {c['n_paired_speeches']} of {c['n_corpus_speeches']:,} - and the "
        f"disagreement measured there is assumed to hold for the eras it widens."
    )


def llm_divergence_sentence(band_table: pd.DataFrame) -> str:
    """The most/least-divergent era sentence, DERIVED from `bands.parquet`.

    Written at render time rather than typed into `SECTIONS`, so the claim
    cannot drift away from the artifact it describes — the failure mode
    CLAUDE.md's "published research notes are a deliverable" section documents,
    where a bare superlative in prose outlives the numbers behind it.

    The aggregation is stated in the sentence itself (a mean across all 50
    level-2 topics, in percentage points) because a superlative whose
    aggregation is unstated is not checkable: mean, max and share-weighted
    rankings need not agree, and a reader must know which one they are reading.

    **The printed number is the FULL annotator-to-annotator gap, not the
    half-width** — i.e. `2 x disagreement_half_width`, which is
    `|share_primary - share_secondary|` before it is halved. The stored column
    is the half-width, because that is what gets added to each SIDE of the band
    (`bands.py`'s composition rule). Printing the half-width and calling it "the
    gap between the two readers" would leave a reader off by a factor of two on
    the one quantity this section exists to expose, so the sentence prints the
    gap and names the halving explicitly. This is the "prose carries the wrong
    mechanism while every number is right" defect class from CLAUDE.md; the
    factor is pinned by test rather than left to the wording.
    """
    llm = band_table[band_table["surface"] == bands.LLM_SURFACE]
    # x200, not x100: x100 converts a fraction to percentage points, x2 undoes
    # the halving in `bands.disagreement_half_widths`.
    per_era = (llm.groupby("period")["disagreement_half_width"].mean() * 200)
    per_era = per_era.dropna()
    if per_era.empty:
        return ""
    top, bottom = per_era.idxmax(), per_era.idxmin()
    # Each printed value is a PER-ERA mean, so the count must be that era's
    # own. Counting the union of measured series across all nine eras still
    # claims 50 while an era that lost one cell to `no_interval_to_widen`
    # averages 49 — the same off-by-a-topic defect one level further out. When
    # the eras disagree the count is qualified rather than picking one to
    # represent both superlatives.
    counts = llm.groupby("period")["disagreement_half_width"].count()
    n_top, n_bottom = int(counts[top]), int(counts[bottom])
    # The two eras average over the same topics today, which is what lets one
    # count describe both clauses. If they ever diverge, each clause carries its
    # own — rather than one era's count silently standing in for the other's.
    tail = (f"({per_era[bottom]:.2f})" if n_top == n_bottom
            else f"({per_era[bottom]:.2f}, across {n_bottom})")
    return (
        f"Averaging across all {n_top} topics, the two readers' topic shares "
        f"diverge most in {html.escape(top)} — by {per_era[top]:.2f} percentage "
        f"points of paragraph share per topic, half of which widens each side "
        f"of the band; they came closest on {html.escape(bottom)} {tail}."
    )


def _president_dots(py: pd.DataFrame, col: str, color: str,
                    unit: str = "per 10k") -> go.Scatter:
    """One faint dot per (president, year), hoverable with attribution."""
    sub = py.dropna(subset=[col])
    return go.Scatter(
        x=sub["year"], y=sub[col], mode="markers", showlegend=False,
        marker=dict(color=color, size=4.5, opacity=0.35),
        customdata=np.stack([sub["president"], sub["n_speeches"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b>, %{x}"
                      " · %{customdata[1]} speech(es)<br>"
                      "%{y:.1f} " + unit + "<extra></extra>",
    )


def _two_line_fig(series: list, ytitle: str, dash_second: bool = False) -> go.Figure:
    """Each entry: (name, smoothed_series) or (name, smoothed, dots_trace)."""
    fig = go.Figure()
    for i, entry in enumerate(series):
        name, s = entry[0], entry[1]
        dots = entry[2] if len(entry) > 2 else None
        if dots is not None:
            fig.add_trace(dots)
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode="lines",
            line=dict(color=SERIES[i], width=2.4,
                      dash="dash" if (dash_second and i == 1) else "solid"),
            connectgaps=False,
            hovertemplate="%{y:.2f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(**_timeline_layout(yaxis_title=ytitle))
    return fig


def fig_certainty(markers: pd.DataFrame, py: pd.DataFrame) -> go.Figure:
    fig = _two_line_fig(
        [("all speeches", indices.certainty_yearly(markers),
          _president_dots(py, "certainty", SERIES[0], unit="share"))],
        ytitle="assertive share of stance markers",
    )
    # Inaugurals happen every four years — plot each as its own point
    # rather than a gap-riddled rolling line.
    inaug = markers[markers["title"].str.contains("Inaugural", case=False, na=False)].copy()
    assertive = inaug["boosters"] + inaug["assertive_modals"]
    deliberative = inaug["hedges"] + inaug["concessives"]
    inaug["share"] = assertive / (assertive + deliberative)
    inaug = inaug[(assertive + deliberative) >= 10]
    fig.add_trace(go.Scatter(
        x=inaug["year"], y=inaug["share"], mode="markers",
        name="inaugural addresses (one point each)",
        marker=dict(color=SERIES[1], size=7, symbol="diamond",
                    line=dict(color=SURFACE, width=1)),
        customdata=inaug["president"],
        hovertemplate="<b>%{customdata}</b> %{x}<br>share %{y:.2f}<extra></extra>",
    ))
    return fig


def fig_naming(rates: pd.DataFrame, py: pd.DataFrame) -> go.Figure:
    return _two_line_fig(
        [("“United States”", rates["united_states"],
          _president_dots(py, "united_states", SERIES[0])),
         ("“America / American(s)”", rates["america"],
          _president_dots(py, "america", SERIES[1]))],
        ytitle="uses per 10,000 words",
    )


def fig_orientation(rates: pd.DataFrame, py: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                        vertical_spacing=0.12,
                        subplot_titles=["future vs nostalgia words per 10,000",
                                        "the two divisions, compared: nostalgia ÷ future"
                                        " and fear ÷ hope"])
    fig.add_trace(_president_dots(py, "future", SERIES[0]), row=1, col=1)
    fig.add_trace(_president_dots(py, "nostalgia", SERIES[1]), row=1, col=1)
    for i, (name, col) in enumerate([("future (future / forward / tomorrow)", "future"),
                                     ("nostalgia (again / restore / back to)", "nostalgia")]):
        fig.add_trace(go.Scatter(
            x=rates.index, y=rates[col], name=name, mode="lines",
            line=dict(color=SERIES[i], width=2.4), connectgaps=False,
            hovertemplate="%{y:.1f} per 10k<extra>" + name + "</extra>",
        ), row=1, col=1)

    ratios = [("nostalgia ÷ future", rates["nostalgia"] / rates["future"], SERIES[2]),
              ("fear ÷ hope", rates["nrc_fear"] / rates["nrc_hope"], SERIES[4])]
    for name, r, color in ratios:
        fig.add_trace(go.Scatter(
            x=r.index, y=r.values, name=name, mode="lines",
            line=dict(color=color, width=2.4), connectgaps=False,
            hovertemplate="%{y:.2f}<extra>" + name + "</extra>",
        ), row=2, col=1)

    fig.update_layout(**_layout(height=640, margin=dict(l=56, r=24, t=40, b=44)))
    fig.update_xaxes(range=X_RANGE, gridcolor=GRID, linecolor=BASELINE,
                     tickfont=dict(color=MUTED, size=11))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE,
                     tickfont=dict(color=MUTED, size=11))
    fig.update_annotations(font=dict(size=12.5, color=INK))
    return fig


def fig_religion(rates: pd.DataFrame, py: pd.DataFrame) -> go.Figure:
    return _two_line_fig(
        [("civil religion (god / faith / pray / bless / sacred)",
          rates["religiosity"], _president_dots(py, "religiosity", SERIES[0])),
         ("“God bless”", rates["god_bless"],
          _president_dots(py, "god_bless", SERIES[1]))],
        ytitle="uses per 10,000 words",
    )


FAMILY_PANELS = [
    ("Soviet → Russia", [(r"\bsoviets?\b|\bussr\b", "Soviet / USSR"),
                         (r"\brussians?\b|\brussia\b", "Russia")]),
    ("global warming → climate change",
     [(r"\bglobal warming\b", "global warming"),
      (r"\bclimate change\b", "climate change")]),
    ("atomic → nuclear", [(r"\batomic\b", "atomic"), (r"\bnuclear\b", "nuclear")]),
    ("terror: the 2000s word that vanished", [(r"\bterror\w*\b", "terror*")]),
    ("social security", [(r"\bsocial security\b", "social security")]),
    ("middle class", [(r"\bmiddle class\b", "middle class")]),
]


def _pattern_rate(df: pd.DataFrame, pattern: str, bucket: int = 5,
                  min_words: int = 20_000) -> pd.Series:
    out = {}
    d = df.assign(p=(df["year"] // bucket) * bucket)
    for period, g in d.groupby("p"):
        text = " ".join(g["transcript"]).lower()
        n = len(re.findall(r"[a-z']+", text))
        if n >= min_words:
            out[period] = len(re.findall(pattern, text)) / n * 10_000
    return pd.Series(out)


def fig_families(df: pd.DataFrame) -> go.Figure:
    """Concepts that hide from single-word views: renamings and vanishings."""
    fig = make_subplots(rows=2, cols=3, shared_xaxes=True,
                        subplot_titles=[t for t, _ in FAMILY_PANELS],
                        vertical_spacing=0.14, horizontal_spacing=0.07)
    for k, (_, series_specs) in enumerate(FAMILY_PANELS):
        r, c = divmod(k, 3)
        for i, (pattern, name) in enumerate(series_specs):
            rate = _pattern_rate(df, pattern)
            fig.add_trace(go.Scatter(
                x=rate.index, y=rate.values, mode="lines", name=name,
                line=dict(color=SERIES[i], width=2), showlegend=False,
                hovertemplate="%{y:.1f} per 10k<extra>" + name + "</extra>",
            ), row=r + 1, col=c + 1)
    fig.update_layout(**_layout(height=560, margin=dict(l=40, r=16, t=48, b=36)))
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE,
                     tickfont=dict(color=MUTED, size=10))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, rangemode="tozero",
                     tickfont=dict(color=MUTED, size=10))
    fig.update_annotations(font=dict(size=12, color=INK))
    return fig


def fig_hype_doom(rates: pd.DataFrame, py: pd.DataFrame) -> go.Figure:
    return _two_line_fig(
        [("hype (greatest / unstoppable / historic / of all time)",
          rates["hype"], _president_dots(py, "hype", SERIES[0])),
         ("doom (worst / disaster / catastrophe / carnage)",
          rates["doom"], _president_dots(py, "doom", SERIES[1]))],
        ytitle="uses per 10,000 words",
    )


def fig_quadrant(scores: pd.DataFrame, faces: dict) -> go.Figure:
    """Unbounded language against legal and procedural vocabulary."""
    fig = go.Figure()
    x_span = float(scores["mechanism"].max()) * 1.15
    y_span = float(scores["hype"].max()) * 1.15
    for pres, row in scores.iterrows():
        x, y = float(row["mechanism"]), float(row["hype"])
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", showlegend=False,
            marker=dict(size=26, opacity=0),
            customdata=[[pres]],
            hovertemplate="<b>%{customdata[0]}</b><br>legal/procedural %{x:.0f} · "
                          "hype %{y:.1f} per 10k<extra></extra>",
        ))
        if pres in faces:
            fig.add_layout_image(
                source=faces[pres], x=x, y=y, xref="x", yref="y",
                sizex=x_span * 30 / 860, sizey=y_span * 30 / 470,
                xanchor="center", yanchor="middle", layer="above",
            )
    for x, y, text in [(x_span * 0.10, y_span * 0.95, "loud, little legal/procedural wording"),
                       (x_span * 0.85, y_span * 0.05, "quiet, heavy legal/procedural wording")]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                           font=dict(size=11, color=MUTED))
    fig.update_layout(**_layout(
        height=560, width=FACE_CHART_WIDTH,
        xaxis=dict(title=dict(text="legal and procedural vocabulary per 10k (act / bill / treaty / appropriation)",
                              font=dict(size=11, color=INK2)),
                   range=[0, x_span], gridcolor=GRID, linecolor=BASELINE,
                   tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(title=dict(text="hype per 10k", font=dict(size=11, color=INK2)),
                   range=[0, y_span], gridcolor=GRID,
                   linecolor=BASELINE, tickfont=dict(color=MUTED, size=11)),
    ))
    return fig


def fig_hope_fear(rates: pd.DataFrame, py: pd.DataFrame) -> go.Figure:
    """The balance only: fear words divided by hope words."""
    ratio = rates["nrc_fear"] / rates["nrc_hope"]
    py = py.copy()
    py["fh_ratio"] = py["nrc_fear"] / py["nrc_hope"].replace(0, np.nan)

    fig = go.Figure()
    fig.add_trace(_president_dots(py, "fh_ratio", SERIES[4], unit="fear/hope"))
    fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio.values, name="fear ÷ hope", mode="lines",
        line=dict(color=SERIES[4], width=2.6), connectgaps=False,
        hovertemplate="fear/hope %{y:.2f}<extra></extra>", showlegend=False,
    ))
    events = [(1812, "War of 1812"), (1860, "Eve of the Civil War"),
              (1893, "Panic of 1893"), (1941, "U.S. enters WWII"),
              (1983, "Cold War escalation"), (2002, "9/11 → Iraq"),
              (2026, "Highest since WWII")]
    for event_no, (year, label) in enumerate(events, 1):
        if year in ratio.index and pd.notna(ratio[year]):
            fig.add_annotation(x=year, y=float(ratio[year]),
                               text=str(event_no), showarrow=True, arrowhead=0,
                               ax=0, ay=-24 if event_no % 2 else 27,
                               bgcolor="#fff9ef", bordercolor="#8b6c42",
                               borderwidth=1, borderpad=3,
                               hovertext=f"{year} · {label}",
                               arrowcolor=MUTED, font=dict(size=9, color=INK2))
    fig.update_layout(**_timeline_layout(height=470,
                                         yaxis_title="fear words ÷ hope words"))
    return fig


def _president_keyword_rates(df: pd.DataFrame) -> pd.DataFrame:
    order = corpus.president_order(df)
    rows = []
    for p in order:
        text = " ".join(df[df["president"] == p]["transcript"]).lower()
        total = max(len(trends.tokens(text)), 1)
        rows.append({
            "president": p,
            **{term: sum(len(re.findall(rf"\b{re.escape(t)}\b", text))
                         for t in terms) / total * 10_000
               for term, terms in trends.term_groups().items()},
        })
    return pd.DataFrame(rows).set_index("president")


def fig_keywords(kw: pd.DataFrame, df: pd.DataFrame | None = None,
                 scores: pd.DataFrame | None = None) -> go.Figure:
    dots = None
    if df is not None and scores is not None:
        pk = _president_keyword_rates(df)
        mids = (scores["first_year"] + scores["last_year"]) / 2
        dots = {
            term: pd.DataFrame({
                "x": mids.values,
                "y": pk.loc[mids.index, term].values,
                "name": mids.index,
            })
            for term in trends.term_groups()
        }
    panels = [
        (term, kw[kw["term"] == term].set_index("period")["rate"])
        for term in kw["term"].unique()
    ]
    return _small_multiples(panels, rows=3, cols=3, height=680,
                            hovertemplate="%{y:.1f} per 10k words",
                            dots=dots, dots_unit=" per 10k",
                            events=WORD_EVENTS)


def fig_distinctive(scores: pd.DataFrame) -> go.Figure:
    eras = list(scores["era"].unique())
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
                        subplot_titles=[f"Distinctive of {e}" for e in eras])
    for c, era in enumerate(eras):
        sub = scores[scores["era"] == era].copy()
        sub["mag"] = sub["z"].abs()
        sub = sub.sort_values("mag")
        fig.add_trace(go.Bar(
            y=sub["term"], x=sub["mag"], orientation="h",
            marker=dict(color=BLUE_RAMP[4]), showlegend=False,
            hovertemplate="%{y}: z = %{x:.1f}<extra></extra>",
        ), row=1, col=c + 1)
    fig.update_layout(**_layout(height=560, margin=dict(l=80, r=24, t=48, b=44)))
    fig.update_xaxes(title_text="log-odds z-score", title_font=dict(size=11, color=INK2),
                     gridcolor=GRID, tickfont=dict(color=MUTED, size=11))
    fig.update_yaxes(tickfont=dict(color=INK, size=12), gridcolor=SURFACE)
    fig.update_annotations(font=dict(size=13, color=INK))
    return fig


_LABEL_ROTATION = ["top center", "bottom center", "top right",
                   "bottom left", "top left", "bottom right"]


def _label_positions(emb: pd.DataFrame, radius: float = 0.085) -> pd.Series:
    """Greedy anti-collision: each point crowded by already-placed neighbors
    takes the next position in the rotation."""
    placed: list[tuple[float, float]] = []
    out = []
    for _, row in emb.iterrows():
        crowded = sum(
            1 for x, y in placed
            if abs(row["pc1"] - x) < radius * 1.6 and abs(row["pc2"] - y) < radius
        )
        out.append(_LABEL_ROTATION[crowded % len(_LABEL_ROTATION)])
        placed.append((row["pc1"], row["pc2"]))
    return pd.Series(out, index=emb.index)


def fig_map(emb: pd.DataFrame, edges: list[tuple] | None = None) -> go.Figure:
    fig = go.Figure()
    if edges:
        coords = emb.set_index("president")[["pc1", "pc2"]]
        for sim_val, a, b, _gap in edges:
            fig.add_trace(go.Scatter(
                x=[coords.loc[a, "pc1"], coords.loc[b, "pc1"]],
                y=[coords.loc[a, "pc2"], coords.loc[b, "pc2"]],
                mode="lines", showlegend=False,
                line=dict(color=BASELINE, width=1.4, dash="dot"),
                hovertemplate=f"{a} ↔ {b}: {sim_val:.2f}<extra></extra>",
            ))
    positions = _label_positions(emb)
    for party, color in PARTY_COLORS.items():
        sub = emb[emb["party"] == party]
        fig.add_trace(go.Scatter(
            x=sub["pc1"], y=sub["pc2"], mode="markers+text",
            text=sub["president"], textposition=positions[sub.index].tolist(),
            textfont=dict(size=10, color=INK2),
            marker=dict(size=11, color=color), name=party,
            customdata=np.stack([sub["first_year"], sub["n_speeches"]], axis=-1),
            hovertemplate="<b>%{text}</b><br>" + party
                          + "<br>first speech: %{customdata[0]}"
                          + "<br>speeches in corpus: %{customdata[1]}<extra></extra>",
        ))
    layout = _layout(
        height=720,
        xaxis_title="principal component 1",
        yaxis_title="principal component 2",
    )
    # Pad the ranges so labels near the hull stay inside the plot area.
    layout["xaxis"]["range"] = [emb["pc1"].min() - 0.14, emb["pc1"].max() + 0.14]
    layout["yaxis"]["range"] = [emb["pc2"].min() - 0.07, emb["pc2"].max() + 0.07]
    fig.update_layout(**layout)
    return fig


def fig_heatmap(sim: pd.DataFrame) -> go.Figure:
    names = list(sim.index)
    m = sim.to_numpy()
    lo = np.percentile(m[~np.eye(len(m), dtype=bool)], 2)
    fig = go.Figure(go.Heatmap(
        z=m, x=names, y=names,
        colorscale=[[i / 6, c] for i, c in enumerate(BLUE_RAMP)],
        zmin=lo, zmax=1.0,
        hovertemplate="%{y} × %{x}<br>similarity: %{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text="cosine", font=dict(size=11, color=INK2)),
                      tickfont=dict(size=10, color=MUTED), thickness=12, outlinewidth=0),
    ))
    fig.update_layout(**_layout(
        height=780,
        yaxis=dict(autorange="reversed", tickfont=dict(size=9.5, color=INK2)),
        xaxis=dict(tickfont=dict(size=9.5, color=INK2), tickangle=90),
        margin=dict(l=140, r=24, t=24, b=130),
    ))
    return fig


def _chart_evidence(
    metric_name: str,
    source: str | None = None,
    *,
    measure_label: str | None = None,
    support: str | None = None,
    caveat: str | None = None,
) -> str:
    """Compact Measure/Evidence pair for one substantive Story Page chart."""
    metric = metrics.METRICS[metric_name]
    source = source or metric["source"]
    support_html = (
        f"<p><strong>Denominator / support.</strong> {html.escape(support)}</p>"
        if support else ""
    )
    caveat_html = (
        f"<p><strong>Important caveat.</strong> {html.escape(caveat)}</p>"
        if caveat else ""
    )
    evidence = (
        '<details class="inspector-tile evidence-tile">'
        '<summary><span aria-hidden="true">▤</span> Evidence · '
        f'{html.escape(Path(source).name)}</summary><div class="inspector-panel">'
        f'<p><strong>Source artifact.</strong> <code>{html.escape(source)}</code>.</p>'
        f'<p><strong>Statistical status.</strong> {html.escape(metric["status"])}.</p>'
        f"{support_html}{caveat_html}"
        '<p><a href="data-quality.html">Definitions, downloads, and data-quality '
        'receipts →</a></p></div></details>'
    )
    return (
        '<div class="inspector-row">'
        + metrics.measure_tile_html(metric_name, measure_label)
        + evidence
        + "</div>"
    )


def _chart_block(
    key: str,
    height: int,
    metric_name: str,
    *,
    source: str | None = None,
    measure_label: str | None = None,
    support: str | None = None,
    caveat: str | None = None,
) -> str:
    return (
        f'<div class="chart-scroll"><div class="chart" data-fig="{key}" '
        f'style="height:{height}px"></div></div>'
        + _chart_evidence(
            metric_name, source, measure_label=measure_label,
            support=support, caveat=caveat,
        )
    )


def _view_toggle_html(
    keys: list[str],
    announcements: list[str],
    height: int,
    metric_name: str,
    *,
    source: str | None = None,
    measure_label: str | None = None,
    support: str | None = None,
    caveat: str | None = None,
) -> str:
    """The shared, explicit Era view / Full history comparison control."""
    if len(keys) != 2 or len(announcements) != 2:
        raise ValueError("historical view toggles require exactly two states")
    labels = ["Era view", "Full history"]
    buttons = "".join(
        f'<button type="button" data-stage-button="{index}" '
        f'aria-pressed="{"true" if index == 0 else "false"}">'
        f"{html.escape(label)}</button>"
        for index, label in enumerate(labels)
    )
    return f"""<div class="story-stage view-toggle" data-stage-mode="chart"
 data-stage-figs="{html.escape(json.dumps(keys), quote=True)}"
 data-stage-announcements="{html.escape(json.dumps(announcements), quote=True)}">
  <div class="stage-controls" role="group" aria-label="Historical extent">{buttons}</div>
  <p class="stage-status" aria-live="polite">{html.escape(announcements[0])}</p>
  <div class="chart-scroll"><div class="chart staged-chart" data-fig="{keys[0]}"
       style="height:{height}px"></div></div>
</div>{_chart_evidence(
    metric_name, source, measure_label=measure_label,
    support=support, caveat=caveat,
)}"""


def _event_callouts(
    events: list[tuple[int, str]], *, numbered: bool = False,
) -> str:
    """Always-visible keyboard/mobile alternative to chart event hover."""
    items = "".join(
        f'<li><span>{f"{index} · " if numbered else ""}{year}</span>'
        f'{html.escape(label)}</li>'
        for index, (year, label) in enumerate(events, 1)
    )
    aria_label = (
        "Numbered historical event context" if numbered
        else "Historical event context"
    )
    note = (
        "Numbered events orient time; they do not establish that an event "
        "caused a measured movement."
        if numbered else
        "Events orient time; they do not establish that an event caused a "
        "measured movement."
    )
    return (
        '<div class="event-callouts" role="note" '
        f'aria-label="{aria_label}"><ul>{items}</ul><p>{note}</p></div>'
    )


def _stage_chart_html(
    keys: list[str],
    labels: list[str],
    announcements: list[str],
    height: int,
    metric_name: str,
) -> str:
    """Accessible scroll-plus-button controller for a sequence of Plotly figures."""
    assert len(keys) == len(labels) == len(announcements)
    buttons = "".join(
        f'<button type="button" data-stage-button="{index}" '
        f'aria-pressed="{"true" if index == 0 else "false"}">'
        f"{index + 1}. {html.escape(label)}</button>"
        for index, label in enumerate(labels)
    )
    steps = "".join(
        f'<div class="stage-step" data-stage-step="{index}" tabindex="0">'
        f'<strong>{index + 1}. {html.escape(label)}</strong>'
        f"<span>{html.escape(announcement)}</span></div>"
        for index, (label, announcement) in enumerate(zip(labels, announcements))
    )
    return f"""<div class="story-stage" data-stage-mode="chart"
 data-stage-figs="{html.escape(json.dumps(keys), quote=True)}"
 data-stage-announcements="{html.escape(json.dumps(announcements), quote=True)}">
  <div class="stage-controls" role="group" aria-label="Chart stages">{buttons}</div>
  <p class="stage-status" aria-live="polite">{html.escape(announcements[0])}</p>
  <div class="chart-scroll"><div class="chart staged-chart" data-fig="{keys[0]}"
       style="height:{height}px"></div></div>
  <div class="stage-narrative" aria-label="Scroll stages">{steps}</div>
</div>{_chart_evidence(metric_name)}"""


FOUNDING_EXTERNAL_TOPICS = (
    "Treaties, Diplomacy & International Arbitration",
    "Neutral Rights & Maritime Depredations",
    "Military Preparedness, Armed Forces & Veterans",
    "Early Naval Wars: Barbary & the War of 1812",
)
FOUNDING_FOREIGN_INDEPENDENCE_TOPICS = (
    "Treaties, Diplomacy & International Arbitration",
    "Neutral Rights & Maritime Depredations",
    "Early Naval Wars: Barbary & the War of 1812",
)
FOUNDING_UNION_TOPICS = (
    "Constitutional Union & Federalism",
    "Crime, Insurrection & Federal Law Enforcement",
    "Executive Power, Vetoes & the Courts",
)
FOUNDING_FISCAL_SYSTEM_TOPICS = (
    "Public Debt, Revenue & Treasury Finance",
    "Taxes, Budget Deficits & Federal Spending",
    "Coinage, Currency & Specie",
    "National Bank & Banking Crises",
    "Tariffs, Reciprocity & Navigation Laws",
)
FOUNDING_CAPACITY_TOPICS = (
    "Public Debt, Revenue & Treasury Finance",
    "Crime, Insurrection & Federal Law Enforcement",
    "Executive Power, Vetoes & the Courts",
)
FOUNDING_NATIVE_TOPIC = "Indian Affairs, Removal & Allotment"
FOUNDING_FINANCE_TOPICS = (
    "Public Debt, Revenue & Treasury Finance",
    "Taxes, Budget Deficits & Federal Spending",
)
FOUNDING_TEST_EXPLANATIONS = {
    "union": "Make constitutional authority hold at home.",
    "finance": "Establish credit, revenue, money, and banking.",
    "native": "Confront Native sovereignty, land, and continental power.",
    "abroad": "Defend treaty rights, neutral commerce, and national independence.",
}
FOUNDING_THREAD_ICONS = {
    "Executive Power, Vetoes & the Courts": "🏛️",
    "Public finance": "💰",
    "Military Preparedness, Armed Forces & Veterans": "🛡️",
    "Crime, Insurrection & Federal Law Enforcement": "⚖️",
    FOUNDING_NATIVE_TOPIC: "🪶",
    "Early Naval Wars: Barbary & the War of 1812": "⚓",
    "Public Debt, Revenue & Treasury Finance": "↳",
    "Taxes, Budget Deficits & Federal Spending": "↳",
}
FOUNDING_DISTINCTIVE_MIN_CORPUS_USES = 50
FOUNDING_DISTINCTIVE_MIN_ERA_SPEECHES = 10
FOUNDING_DISTINCTIVE_MIN_ERA_PRESIDENTS = 2
FOUNDING_AUDIENCE_GROUPS = (
    ("Congress", ("congress",)),
    ("General public", ("general_public",)),
    ("Specific organizations or groups", ("specific_organization_or_group",)),
    ("Other audiences", ("press", "foreign_or_diplomatic", "military", "other")),
)
FOUNDING_MEDIUM_GROUPS = (
    ("Written message", ("written_message",)),
    ("Spoken address", ("spoken_address",)),
    ("Radio/TV broadcast", ("broadcast_radio_or_tv",)),
    (
        "Press conference/debate or other performed form",
        ("press_conference", "debate"),
    ),
)
FOUNDING_ERA_SHORT = (
    "Establishing",
    "Continental republic",
    "Sectional crisis",
    "Admin-industrial",
    "Reform & collapse",
    "New Deal / WWII",
    "Mature Cold War",
    "Conservative / always-on",
    "Platform era",
)
FOUNDING_THREAD_SPECS = (
    (
        "Executive Power, Vetoes & the Courts",
        ("Executive Power, Vetoes & the Courts",),
        "Persistent · later Civil War-era peak.",
        "persistent",
    ),
    (
        "Public finance",
        FOUNDING_FINANCE_TOPICS,
        "Debt/Treasury → taxes, deficits & spending.",
        "persistent",
    ),
    (
        "Military Preparedness, Armed Forces & Veterans",
        ("Military Preparedness, Armed Forces & Veterans",),
        "Persistent · repeatedly presidential across the corpus.",
        "persistent",
    ),
    (
        "Crime, Insurrection & Federal Law Enforcement",
        ("Crime, Insurrection & Federal Law Enforcement",),
        "Recurrent · falls and returns in the raw corpus.",
        "recurrent",
    ),
    (
        FOUNDING_NATIVE_TOPIC,
        (FOUNDING_NATIVE_TOPIC,),
        "Founding high · Gilded-era return, then near-zero presidential attention.",
        "disappearing",
    ),
    (
        "Early Naval Wars: Barbary & the War of 1812",
        ("Early Naval Wars: Barbary & the War of 1812",),
        "Early conflict label · concentrated around the Barbary wars and War of 1812.",
        "disappearing",
    ),
)
def _strict_one_to_one_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    stage: str,
) -> pd.DataFrame:
    """Join two co-derived artifacts only when their exact key sets match."""
    joined = left.merge(
        right, on=keys, how="outer", validate="one_to_one", indicator=True,
    )
    mismatch = joined["_merge"].ne("both")
    if mismatch.any():
        counts = joined.loc[mismatch, "_merge"].value_counts().to_dict()
        raise ValueError(
            f"{stage}: key sets differ on {keys}; unmatched rows={counts}"
        )
    return joined.drop(columns="_merge")


def _topic_mask(frame: pd.DataFrame, topics: tuple[str, ...]) -> pd.Series:
    wanted = frozenset(topics)
    return frame["topic_set"].map(lambda assigned: not assigned.isdisjoint(wanted))


def _persistence_color(score: float, low: float, high: float) -> str:
    """Accessible rust-to-blue row color; text still carries the classification."""
    t = 1.0 if high == low else (score - low) / (high - low)
    t = float(np.clip(t, 0, 1))
    start = np.array([162, 79, 82], dtype=float)
    end = np.array([49, 95, 120], dtype=float)
    rgb = np.rint(start + (end - start) * t).astype(int)
    return "#" + "".join(f"{value:02x}" for value in rgb)


def _rank_founding_threads(rows: list[dict]) -> list[dict]:
    """Sort by persistence, resolving scores within .02 by normalized stability."""
    ranked = sorted(rows, key=lambda row: row["persistence_score"], reverse=True)
    out: list[dict] = []
    cursor = 0
    while cursor < len(ranked):
        anchor = ranked[cursor]["persistence_score"]
        end = cursor + 1
        while (
            end < len(ranked)
            and anchor - ranked[end]["persistence_score"] <= .02
        ):
            end += 1
        out.extend(sorted(
            ranked[cursor:end],
            key=lambda row: row["normalized_std"],
        ))
        cursor = end
    return out


def founding_story_data(
    speeches: pd.DataFrame,
    paragraphs: pd.DataFrame,
    paragraph_annotations: pd.DataFrame,
    speech_annotations: pd.DataFrame,
    speech_stats: pd.DataFrame,
    agreement: pd.DataFrame,
    lifecycles: pd.DataFrame,
    taxonomy: dict,
    foundation: story_foundation.StoryFoundationBundle,
    invocation_evidence: pd.DataFrame | None = None,
) -> dict:
    """Recompute every founding-story value from frozen, key-validated artifacts."""
    paragraph_frame = _strict_one_to_one_merge(
        paragraphs[["doc_name", "para_idx", "word_count"]],
        paragraph_annotations[["doc_name", "para_idx", "topics"]],
        ["doc_name", "para_idx"],
        "founding story paragraphs x paragraph annotations",
    )
    speech_frame = _strict_one_to_one_merge(
        speeches[["doc_name", "president", "year", "word_count"]],
        speech_annotations[["doc_name", "audience", "medium"]],
        ["doc_name"],
        "founding story speeches x speech annotations",
    )
    speech_frame["era_key"] = era_profiles.story_era_key_series(
        speech_frame["year"], speech_frame["president"]
    )
    speech_frame["era"] = speech_frame["era_key"].map(
        lambda key: era_profiles.resolve_era(key).label
    )
    language_frame = _strict_one_to_one_merge(
        speech_frame[["doc_name", "year", "era_key", "era"]],
        speech_stats[["doc_name", "n_tokens", "n_sents", "modal_shall"]],
        ["doc_name"],
        "founding story speeches x speech language statistics",
    )
    if (
        language_frame["n_tokens"].le(0).any()
        or language_frame["n_sents"].le(0).any()
    ):
        raise ValueError(
            "founding story speech language statistics require positive "
            "token and sentence counts"
        )
    paragraph_frame = paragraph_frame.merge(
        speech_frame[
            ["doc_name", "president", "year", "era_key", "era"]
        ],
        on="doc_name", how="left", validate="many_to_one",
    )
    if paragraph_frame[["president", "year"]].isna().any().any():
        missing = int(paragraph_frame["year"].isna().sum())
        raise ValueError(
            f"founding story paragraphs x speeches: {missing} paragraphs "
            "have no speech metadata"
        )

    label_map = attention.canonical_label_map(taxonomy)
    paragraph_frame["topic_set"] = [
        frozenset(attention.normalize_topics(raw, label_map))
        for raw in paragraph_frame["topics"]
    ]
    if paragraph_frame["era"].isna().any() or speech_frame["era"].isna().any():
        raise ValueError("founding story contains years outside the story eras")

    founding = paragraph_frame[
        paragraph_frame["era_key"].eq("founding")
    ].copy()
    founding_n = len(founding)
    problem_specs = (
        (
            "External sovereignty and security",
            FOUNDING_EXTERNAL_TOPICS,
            "external",
        ),
        (
            "Building federal capacity",
            FOUNDING_CAPACITY_TOPICS,
            "capacity",
        ),
        (
            "Indian and Native affairs",
            (FOUNDING_NATIVE_TOPIC,),
            "native",
        ),
    )
    problems = []
    for label, topics, key in problem_specs:
        mask = _topic_mask(founding, topics)
        components = []
        for topic in topics:
            component_mask = _topic_mask(founding, (topic,))
            components.append({
                "topic": topic,
                "count": int(component_mask.sum()),
                "share": float(component_mask.mean() * 100),
            })
        problems.append({
            "key": key,
            "label": label,
            "topics": topics,
            "count": int(mask.sum()),
            "share": float(mask.mean() * 100),
            "components": components,
        })
    any_problem = np.logical_or.reduce([
        _topic_mask(founding, topics).to_numpy()
        for _, topics, _ in problem_specs
    ])
    treaty_neutral_overlap = (
        _topic_mask(founding, (FOUNDING_EXTERNAL_TOPICS[0],))
        & _topic_mask(founding, (FOUNDING_EXTERNAL_TOPICS[1],))
    )
    external_mask = _topic_mask(founding, FOUNDING_EXTERNAL_TOPICS)
    native_mask = _topic_mask(founding, (FOUNDING_NATIVE_TOPIC,))
    sovereignty_security = external_mask | native_mask
    sovereignty_overlap = external_mask & native_mask

    founding_speeches = speech_frame[
        speech_frame["era_key"].eq("founding")
    ].copy()
    founding_language = language_frame[
        language_frame["era_key"].eq("founding")
    ].copy()
    founding_words = int(founding_speeches["word_count"].sum())
    corpus_words = int(speech_frame["word_count"].sum())
    founding_language_tokens = int(founding_language["n_tokens"].sum())
    corpus_language_tokens = int(language_frame["n_tokens"].sum())
    language_profile = {
        "words_per_sentence": float(
            founding_language_tokens / founding_language["n_sents"].sum()
        ),
        "corpus_words_per_sentence": float(
            corpus_language_tokens / language_frame["n_sents"].sum()
        ),
        "shall_per_1000": float(
            founding_language["modal_shall"].sum()
            / founding_language_tokens
            * 1_000
        ),
        "corpus_shall_per_1000": float(
            language_frame["modal_shall"].sum()
            / corpus_language_tokens
            * 1_000
        ),
    }
    story_keys = era_profiles.story_era_key_series(
        speeches["year"], speeches["president"]
    )
    founding_word_counts = trends.word_counts(
        speeches.loc[story_keys.eq("founding"), "transcript"]
    )
    other_word_counts = trends.word_counts(
        speeches.loc[~story_keys.eq("founding"), "transcript"]
    )
    distinctive_scores = trends.log_odds_scores(
        founding_word_counts,
        other_word_counts,
        min_count=FOUNDING_DISTINCTIVE_MIN_CORPUS_USES,
    )
    distinctive_scores = distinctive_scores[
        ~distinctive_scores["term"].isin(trends.REGISTER_WORDS)
    ].copy()
    candidate_terms = frozenset(distinctive_scores["term"])
    speech_counts: Counter[str] = Counter()
    president_sets = {term: set() for term in candidate_terms}
    for row in speeches.loc[
        story_keys.eq("founding"),
        ["president", "transcript"],
    ].itertuples(index=False):
        present_terms = set(trends.tokens(row.transcript)) & candidate_terms
        speech_counts.update(present_terms)
        for term in present_terms:
            president_sets[term].add(str(row.president))
    distinctive_scores["era_speech_count"] = distinctive_scores["term"].map(
        speech_counts
    )
    distinctive_scores["era_president_count"] = distinctive_scores["term"].map(
        lambda term: len(president_sets[term])
    )
    distinctive_scores = distinctive_scores[
        distinctive_scores["era_speech_count"].ge(
            FOUNDING_DISTINCTIVE_MIN_ERA_SPEECHES
        )
        & distinctive_scores["era_president_count"].ge(
            FOUNDING_DISTINCTIVE_MIN_ERA_PRESIDENTS
        )
    ]
    if len(distinctive_scores) < 3:
        raise ValueError(
            "founding story has fewer than three eligible corpus-relative "
            "distinctive words"
        )
    founding_content_words = sum(founding_word_counts.values())
    other_content_words = sum(other_word_counts.values())
    distinctive_words = []
    for rank, row in enumerate(
        distinctive_scores.tail(3).iloc[::-1].itertuples(index=False), start=1
    ):
        term = str(row.term)
        founding_term_rate = (
            founding_word_counts[term] / founding_content_words * 10_000
        )
        other_term_rate = (
            other_word_counts[term] / other_content_words * 10_000
        )
        distinctive_words.append({
            "rank": rank,
            "term": term,
            "founding_count": int(founding_word_counts[term]),
            "other_count": int(other_word_counts[term]),
            "era_speech_count": int(row.era_speech_count),
            "era_president_count": int(row.era_president_count),
            "founding_rate_per_10000": float(founding_term_rate),
            "other_rate_per_10000": float(other_term_rate),
            "rate_ratio": float(founding_term_rate / other_term_rate),
            "z": float(row.z),
        })
    language_profile["distinctive_words"] = distinctive_words
    language_profile["distinctive_word"] = distinctive_words[0]
    language_profile["distinctive_eligibility"] = {
        "min_corpus_uses": FOUNDING_DISTINCTIVE_MIN_CORPUS_USES,
        "min_era_speeches": FOUNDING_DISTINCTIVE_MIN_ERA_SPEECHES,
        "min_era_presidents": FOUNDING_DISTINCTIVE_MIN_ERA_PRESIDENTS,
    }
    audience_values = {
        raw for _, raw_values in FOUNDING_AUDIENCE_GROUPS for raw in raw_values
    }
    medium_values = {
        raw for _, raw_values in FOUNDING_MEDIUM_GROUPS for raw in raw_values
    }
    unknown_audience = sorted(set(speech_frame["audience"]) - audience_values)
    unknown_medium = sorted(set(speech_frame["medium"]) - medium_values)
    if unknown_audience or unknown_medium:
        raise ValueError(
            "founding story display grouping is incomplete: "
            f"audience={unknown_audience}, medium={unknown_medium}"
        )

    communications = []
    for index, ((era, lo, hi), short) in enumerate(
        zip(era_profiles.STORY_ERAS, FOUNDING_ERA_SHORT)
    ):
        era_speeches = speech_frame[speech_frame["era"].eq(era)]
        total = len(era_speeches)
        if not total:
            raise ValueError(f"founding story communication era {era!r} is empty")

        def grouped(groups: tuple, column: str) -> list[dict]:
            out = []
            for label, raw_values in groups:
                count = int(era_speeches[column].isin(raw_values).sum())
                out.append({
                    "label": label,
                    "raw_values": raw_values,
                    "count": count,
                    "share": count / total * 100,
                })
            return out

        if index <= 3:
            interpretation = (
                "Congress and written messages dominate this corpus era."
            )
        elif index == 4:
            interpretation = (
                "The earlier congressional, written-message pattern loosens."
            )
        elif index == 5:
            interpretation = (
                "General-public speech and assigned broadcast form dominate."
            )
        else:
            interpretation = (
                "The record remains public-facing while performed forms diversify."
            )
        communications.append({
            "key": f"founding-era-{index}",
            "era": era,
            "short": short,
            "years": f"{lo}–{hi}",
            "total": total,
            "audience": grouped(FOUNDING_AUDIENCE_GROUPS, "audience"),
            "medium": grouped(FOUNDING_MEDIUM_GROUPS, "medium"),
            "interpretation": interpretation,
        })

    founding_communication = communications[0]
    dominant_audience = max(
        founding_communication["audience"], key=lambda row: row["share"]
    )
    dominant_medium = max(
        founding_communication["medium"], key=lambda row: row["share"]
    )

    def agreement_row(field: str) -> dict:
        rows = agreement[
            agreement["field"].eq(field)
            & agreement["metric"].eq("exact_match")
            & agreement["era_bin"].eq(-1)
        ]
        if len(rows) != 1:
            raise ValueError(
                f"agreement_v1 needs one overall exact_match row for {field!r}"
            )
        row = rows.iloc[0]
        return {"rate": float(row["value"] * 100), "n": int(row["n"])}

    audience_agreement = agreement_row("audience")
    medium_agreement = agreement_row("medium")
    if audience_agreement["n"] != medium_agreement["n"]:
        raise ValueError("audience and medium agreement samples have different n")

    lifecycle_rows = lifecycles[
        lifecycles["level"].eq("level2") & lifecycles["treatment"].eq("raw")
    ].set_index("topic")
    required_topics = set(
        FOUNDING_EXTERNAL_TOPICS
        + FOUNDING_CAPACITY_TOPICS
        + (FOUNDING_NATIVE_TOPIC,)
        + FOUNDING_FINANCE_TOPICS
        + FOUNDING_UNION_TOPICS
        + FOUNDING_FISCAL_SYSTEM_TOPICS
        + FOUNDING_FOREIGN_INDEPENDENCE_TOPICS
    )
    missing_lifecycles = sorted(required_topics - set(lifecycle_rows.index))
    if missing_lifecycles:
        raise ValueError(
            "topic_lifecycles raw level-2 rows are missing: "
            f"{missing_lifecycles}"
        )

    denominators = []
    era_frames = []
    for era, _, _ in era_profiles.STORY_ERAS:
        era_frame = paragraph_frame[paragraph_frame["era"].eq(era)]
        if era_frame.empty:
            raise ValueError(f"founding story paragraph era {era!r} is empty")
        era_frames.append(era_frame)
        denominators.append(len(era_frame))

    next_era_preview = []
    for label, topic in (
        ("National banking", "National Bank & Banking Crises"),
        ("Tariffs", "Tariffs, Reciprocity & Navigation Laws"),
        ("Executive power & courts", "Executive Power, Vetoes & the Courts"),
        ("Maritime rights", "Neutral Rights & Maritime Depredations"),
    ):
        founding_count = int(_topic_mask(era_frames[0], (topic,)).sum())
        expansion_count = int(_topic_mask(era_frames[1], (topic,)).sum())
        founding_share = founding_count / denominators[0] * 100
        expansion_share = expansion_count / denominators[1] * 100
        next_era_preview.append({
            "label": label,
            "topic": topic,
            "founding_count": founding_count,
            "expansion_count": expansion_count,
            "founding_share": founding_share,
            "expansion_share": expansion_share,
            "change": expansion_share - founding_share,
        })

    attention_departures = []
    for key, icon, label, topic, note in (
        (
            "native",
            "🪶",
            "Native affairs",
            FOUNDING_NATIVE_TOPIC,
            "The corpus label collapses; Native presence, power, and policy do not.",
        ),
        (
            "naval",
            "⚓",
            "Barbary and War of 1812 naval affairs",
            "Early Naval Wars: Barbary & the War of 1812",
            "The early-war label ends; foreign and military speech continue.",
        ),
    ):
        counts = [int(_topic_mask(frame, (topic,)).sum()) for frame in era_frames]
        shares = [
            count / denominator * 100
            for count, denominator in zip(counts, denominators)
        ]
        attention_departures.append({
            "key": key,
            "icon": icon,
            "label": label,
            "topic": topic,
            "counts": counts,
            "shares": shares,
            "note": note,
        })

    threads = []
    for label, topics, descriptor, pattern in FOUNDING_THREAD_SPECS:
        counts = [int(_topic_mask(frame, topics).sum()) for frame in era_frames]
        shares = [
            count / denominator * 100
            for count, denominator in zip(counts, denominators)
        ]
        if shares[0] <= 0:
            raise ValueError(f"founding thread {label!r} has no founding baseline")
        normalized = np.array(shares, dtype=float) / shares[0]
        threads.append({
            "label": label,
            "topics": topics,
            "descriptor": descriptor,
            "pattern": pattern,
            "counts": counts,
            "denominators": denominators,
            "shares": shares,
            "normalized": normalized.tolist(),
            "persistence_score": float(normalized[1:].mean()),
            "normalized_std": float(normalized.std(ddof=0)),
            "lifecycle_classes": [
                str(lifecycle_rows.loc[topic, "lifecycle_class"])
                for topic in topics
            ],
        })
    threads = _rank_founding_threads(threads)
    scores = [row["persistence_score"] for row in threads]
    for row in threads:
        row["color"] = _persistence_color(row["persistence_score"], min(scores), max(scores))

    finance_parent = next(row for row in threads if row["label"] == "Public finance")
    finance_sublanes = []
    for topic in FOUNDING_FINANCE_TOPICS:
        counts = [int(_topic_mask(frame, (topic,)).sum()) for frame in era_frames]
        shares = [
            count / denominator * 100
            for count, denominator in zip(counts, denominators)
        ]
        finance_sublanes.append({
            "label": topic,
            "topics": (topic,),
            "descriptor": "Public-finance sub-lane.",
            "pattern": finance_parent["pattern"],
            "counts": counts,
            "denominators": denominators,
            "shares": shares,
            "color": finance_parent["color"],
            "is_sublane": True,
        })

    all_era_profile_data = era_profiles.build_era_profiles(
        speeches,
        paragraphs,
        paragraph_annotations,
        speech_annotations,
        taxonomy,
        foundation,
    )
    all_era_visualization_data = (
        era_visualizations.build_era_visualizations(
            paragraph_annotations,
            speech_annotations,
            taxonomy,
            foundation,
        )
    )
    if invocation_evidence is None:
        invocation_evidence = pd.read_parquet(
            corpus.DATA_DIR / "networks" / "invocation_evidence.parquet"
        )
    all_era_contextualization_data = (
        era_contextualizations.build_era_contextualizations(
            speeches,
            paragraphs,
            paragraph_annotations,
            taxonomy,
            all_era_profile_data,
            invocation_evidence,
            foundation,
        )
    )
    founding_visualization = all_era_visualization_data["founding"]
    founding_contextualization = all_era_contextualization_data["founding"]

    return {
        "founding_paragraphs": founding_n,
        "founding_speeches": int(founding["doc_name"].nunique()),
        "corpus_paragraphs": len(paragraph_frame),
        "corpus_speeches": len(speech_frame),
        "founding_words": founding_words,
        "corpus_words": corpus_words,
        "language_profile": language_profile,
        "founding_paragraph_corpus_share": float(
            founding_n / len(paragraph_frame) * 100
        ),
        "founding_speech_corpus_share": float(
            len(founding_speeches) / len(speech_frame) * 100
        ),
        "founding_word_corpus_share": float(
            founding_words / corpus_words * 100
        ),
        "problems": problems,
        "any_problem_count": int(any_problem.sum()),
        "any_problem_share": float(any_problem.mean() * 100),
        "treaty_neutral_overlap_count": int(treaty_neutral_overlap.sum()),
        "treaty_neutral_overlap_share": float(treaty_neutral_overlap.mean() * 100),
        "sovereignty_security_count": int(sovereignty_security.sum()),
        "sovereignty_security_share": float(
            sovereignty_security.mean() * 100
        ),
        "sovereignty_security_overlap_count": int(sovereignty_overlap.sum()),
        "sovereignty_security_overlap_share": float(
            sovereignty_overlap.mean() * 100
        ),
        "communications": communications,
        "agreement": {
            "audience": audience_agreement,
            "medium": medium_agreement,
            "n": audience_agreement["n"],
        },
        "threads": threads,
        "finance_sublanes": finance_sublanes,
        "president_communications": (
            founding_visualization["governance"]["presidents"]
        ),
        "adversary_network": founding_visualization["adversaries"]["edges"],
        "president_adversary_summary": (
            founding_visualization["adversaries"]["president_summary"]
        ),
        "next_era_preview": next_era_preview,
        "attention_departures": attention_departures,
        "era_profiles": all_era_profile_data,
        "era_profile": all_era_profile_data["founding"],
        "era_visualizations": all_era_visualization_data,
        "era_visualization": founding_visualization,
        "era_contextualizations": all_era_contextualization_data,
        "era_contextualization": founding_contextualization,
    }


ERA_ADVERSARY_BUBBLE_POSITIONS = (
    (92, 86), (187, 45), (265, 47), (193, 126), (270, 125)
)
ERA_ADVERSARY_VIEWBOX = (350, 165)


def _era_adversary_bubble_layout(adversaries: list[dict]) -> list[dict]:
    """Scale one fixed five-node pack until all current-era circles clear."""
    if not adversaries:
        return []
    max_count = max(row["paragraphs"] for row in adversaries)
    raw_radii = [
        25 + 30 * np.sqrt(row["paragraphs"] / max_count)
        for row in adversaries
    ]
    positions = ERA_ADVERSARY_BUBBLE_POSITIONS[:len(adversaries)]
    width, height = ERA_ADVERSARY_VIEWBOX
    margin = 2.0
    scale = 1.0
    for (x, y), radius in zip(positions, raw_radii):
        scale = min(
            scale,
            (x - margin) / radius,
            (width - x - margin) / radius,
            (y - margin) / radius,
            (height - y - margin) / radius,
        )
    for left_index, ((left_x, left_y), left_radius) in enumerate(
        zip(positions, raw_radii)
    ):
        for (right_x, right_y), right_radius in zip(
            positions[left_index + 1:],
            raw_radii[left_index + 1:],
        ):
            distance = float(np.hypot(
                right_x - left_x, right_y - left_y
            ))
            scale = min(
                scale,
                (distance - margin) / (left_radius + right_radius),
            )
    return [
        {"x": x, "y": y, "radius": float(radius * scale)}
        for (x, y), radius in zip(positions, raw_radii)
    ]


def _word_boundary_excerpt(value: object, max_chars: int = 360) -> str:
    """Collapse whitespace and truncate without cutting through a word."""
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    prefix = text[: max_chars - 1]
    if " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix.rstrip() + "…"


def _era_profile_html(
    profile: dict,
    panel_id: str,
    tab_id: str | None = None,
) -> str:
    """Render one era-agnostic profile from ``era_profiles`` data."""
    ranked_topics = profile["major_topics"]
    max_topic_share = max(
        (row["share"] for row in ranked_topics), default=1
    )
    topics = "".join(
        f'<li title="{html.escape(row["topic"], quote=True)}" '
        f'style="--topic-width:{row["share"] / max_topic_share * 100:.2f}%">'
        f'<i aria-hidden="true"></i><span>{html.escape(row["label"])}</span>'
        f'<strong>{row["share"]:.1f}%</strong></li>'
        for row in ranked_topics
    )
    president_cards = []
    for row in profile["presidents"]:
        president_name = html.escape(row["name"])
        president_name_attr = html.escape(row["name"], quote=True)
        president_slug = profiles.slug(row["name"])
        source_years = (
            str(row["first_source_year"])
            if row["first_source_year"] == row["last_source_year"]
            else f'{row["first_source_year"]}–{row["last_source_year"]}'
        )
        cross_owner = ""
        if row["cross_owner_appearances"]:
            qualifier = (
                "cross-owner only"
                if row["cross_owner_only"]
                else f'{row["cross_owner_appearances"]} cross-owner'
            )
            cross_owner = (
                f'<b class="era-president-cross-owner">'
                f'{html.escape(qualifier)}</b>'
            )
        support = (
            f'{row["appearances"]} source appearances · '
            f'{row["paragraphs"]:,} speaker-audited paragraphs'
        )
        president_cards.append(
            f'<li title="{html.escape(source_years, quote=True)} · '
            f'{html.escape(support, quote=True)}">'
            f'<a class="era-president-portrait-link" '
            f'href="presidents/{president_slug}.html" '
            f'aria-label="Open {president_name_attr} profile">'
            f'<img src="portraits/{president_slug}.png" alt="" '
            f'width="64" height="64"></a>'
            f'<span>{president_name}</span>'
            f'<small>{row["appearances"]} appearances</small>'
            f'{cross_owner}</li>'
        )
    presidents_html = "".join(president_cards)
    bubble_layout = _era_adversary_bubble_layout(profile["adversaries"])

    def bubble_label(name: str) -> tuple[str, ...]:
        words = name.split()
        if len(name) <= 11 or len(words) == 1:
            return (name,)
        split = min(
            range(1, len(words)),
            key=lambda index: abs(
                len(" ".join(words[:index]))
                - len(" ".join(words[index:]))
            ),
        )
        return (" ".join(words[:split]), " ".join(words[split:]))

    adversary_groups = []
    for row, layout in zip(profile["adversaries"], bubble_layout):
        legend_label, color = era_profiles.ADVERSARY_TYPE_STYLES.get(
            row["type"], ("Other", "#74695f")
        )
        radius = layout["radius"]
        x, y = layout["x"], layout["y"]
        lines = bubble_label(row["name"])
        first_y = -5 if len(lines) == 1 else -10
        text_lines = "".join(
            f'<tspan x="0" y="{first_y + line_index * 10}">'
            f'{html.escape(line)}</tspan>'
            for line_index, line in enumerate(lines)
        )
        count_y = first_y + len(lines) * 10 + 3
        adversary_groups.append(
            f'<g class="era-adversary-bubble" '
            f'transform="translate({x} {y})" '
            f'style="--adversary-color:{color}">'
            f'<title>{html.escape(row["name"])} · {legend_label} · '
            f'{row["paragraphs"]} adversarial paragraphs</title>'
            f'<circle r="{radius:.2f}"></circle>'
            f'<text text-anchor="middle">{text_lines}'
            f'<tspan class="era-adversary-count" x="0" y="{count_y}">'
            f'{row["paragraphs"]}</tspan></text></g>'
        )
    adversaries = "".join(adversary_groups) or (
        '<text class="era-adversary-empty" x="175" y="88" '
        'text-anchor="middle">Awaiting entity layer</text>'
    )
    adversary_legend = "".join(
        f'<span class="{"is-present" if row["present"] else "is-absent"}" '
        f'title="{row["corpus_mentions"]:,} entity mentions in the full corpus'
        f'{" · represented among these five" if row["present"] else " · not represented among these five"}">'
        f'<i style="--adversary-color:{row["color"]}"></i>'
        f'{html.escape(row["label"])}</span>'
        for row in profile["adversary_types"]
    )
    landscape = profile["reference_landscape"]
    reference_lanes = []
    for row in landscape["types"]:
        share = float(row["paragraph_share"])
        corpus_share = float(row["corpus_paragraph_share"])
        highlight = row["highlight"]
        if highlight is None:
            highlight_html = (
                '<span class="era-reference-highlight-empty" '
                'title="No candidate met the support, stance, and '
                'named-adversary exclusion rules">No supported highlight</span>'
            )
        else:
            evidence = highlight["evidence"]
            support_status = str(highlight.get("support_status", "supported"))
            is_limited = support_status == "limited_record"
            badge = str(highlight["source_agreement"]["badge"])
            badge_class = "ai-ner" if badge == "AI + NER" else "ai-only"
            badge_icon_html = (
                '<i aria-hidden="true">✓</i>' if badge == "AI + NER" else ""
            )
            badge_title = (
                "Primary AI and local NER found the same normalized name "
                "in this evidence paragraph."
                if badge == "AI + NER"
                else "The primary AI found this name; local NER did not "
                "find the same normalized name in this evidence paragraph."
            )
            ner_mention = (
                "not found"
                if evidence["ner_mention"] is None
                else str(evidence["ner_mention"])
            )
            if evidence["cross_owner"]:
                ownership = (
                    f'Actual speaker: {evidence["actual_speaker"]} · source '
                    f'document cataloged under {evidence["document_owner"]}'
                )
            else:
                ownership = (
                    "Speaker and source owner: "
                    f'{evidence["actual_speaker"]}'
                )
            stance_mix = highlight["stance_mix"]
            stance_text = " · ".join(
                f'{stance_mix[name]} {name}'
                for name in ("favorable", "neutral", "adversarial")
                if stance_mix[name]
            )
            key = f'({evidence["doc_name"]}, {evidence["para_idx"]})'
            support_status_html = (
                '<span class="era-reference-status">Limited record</span>'
                if is_limited else ""
            )
            limited_note = (
                '<p class="era-reference-limited-note"><strong>Limited record:</strong> '
                'this name appears across multiple documents but is one paragraph '
                'short of the five-paragraph supported-highlight floor.</p>'
                if is_limited else ""
            )
            highlight_html = (
                '<details class="era-reference-highlight" '
                f'data-support-status="{html.escape(support_status, quote=True)}">'
                '<summary>'
                f'<span class="era-reference-name"><b>{html.escape(str(highlight["label"]))}</b></span>'
                '<small class="era-reference-meta">'
                f'<span>{highlight["support"]["paragraphs"]} paragraphs</span>'
                f'<span>{highlight["support"]["source_documents"]} documents</span>'
                f'<span class="era-reference-badge is-{badge_class}" '
                f'title="{html.escape(badge_title, quote=True)}">'
                f'{badge_icon_html}'
                f'{html.escape(badge)}</span>'
                f'{support_status_html}</small></summary>'
                '<div class="era-reference-evidence">'
                f'{limited_note}'
                f'<p class="era-reference-excerpt">'
                f'{html.escape(_word_boundary_excerpt(evidence["excerpt"]))}</p>'
                f'<p><strong>AI stance across this era:</strong> '
                f'{html.escape(stance_text)}<br>'
                f'<strong>Selected mention:</strong> '
                f'{html.escape(str(evidence["ai_mention"]))} · '
                f'<strong>NER:</strong> {html.escape(ner_mention)}<br>'
                f'{html.escape(ownership)}</p>'
                f'<p><a href="{html.escape(str(evidence["source_url"]), quote=True)}" '
                'target="_blank" rel="noopener">'
                f'{html.escape(str(evidence["title"]))}</a><br>'
                f'<code>{html.escape(key)}</code></p>'
                '</div></details>'
            )
        reference_lanes.append(
            '<li class="era-reference-lane">'
            '<div class="era-reference-metric">'
            f'<span>{html.escape(str(row["label"]))}</span>'
            f'<strong>{share * 100:.1f}%</strong>'
            f'<small>all eras {corpus_share * 100:.1f}%</small></div>'
            f'{highlight_html}</li>'
        )
    reference_landscape = "".join(reference_lanes)
    distinctive_word_cards = []
    for row in profile["distinctive_words"]:
        relative_rate = (
            "only in this era"
            if row["rate_ratio"] is None
            else f'{row["rate_ratio"]:.1f}× other eras'
        )
        distinctive_word_cards.append(
            f'<p class="era-distinctive-word" '
            f'title="Explore word-family distinctiveness rank '
            f'{row["rank"]}: '
            f'{row["era_count"]} uses in {profile["years"]} and '
            f'{row["other_count"]} in every other era">'
            f'<strong>{html.escape(row["term"]).upper()}</strong>'
            f'<span>Rank #{row["rank"]}</span>'
            f'<small>{row["era_count"]:,} uses<br>{relative_rate}</small></p>'
        )
    distinctive_words = "".join(distinctive_word_cards)
    footprint = profile["footprint"]
    eligibility = profile["distinctive_eligibility"]
    panel_attrs = (
        f' role="tabpanel" aria-labelledby="{tab_id}"' if tab_id else ""
    )
    standalone_class = "" if tab_id else " era-profile-standalone"
    title_id = f"{panel_id}-title"
    reference_title_id = f"{panel_id}-reference-title"
    adversary_title_id = f"{panel_id}-adversary-title"
    return f"""<section class="founding-story-card founding-panel era-template{standalone_class}"
  id="{panel_id}"{panel_attrs}>
  <div class="era-template-heading">
    <h3 id="{title_id}">{profile['years']}</h3>
    <p><strong>{html.escape(profile['title'])}</strong>
      <span>{html.escape(profile['subtitle'])}</span></p>
  </div>
  <ul class="era-president-strip"
    aria-label="Presidential voices in these sources">{presidents_html}</ul>
  <p class="era-president-strip-label">Presidential voices in these sources ·
    years and appearances describe source evidence, not tenure.</p>
  <div class="era-template-grid">
    <article class="era-template-references" aria-labelledby="{reference_title_id}">
      <header class="era-card-heading" id="{reference_title_id}">
        <span>Distinctive era references</span></header>
      <ul class="era-reference-landscape">{reference_landscape}</ul>
    </article>
    <article class="era-template-adversaries">
      <header class="era-card-heading"><span>Named adversaries</span></header>
      <svg class="era-adversary-pack" viewBox="0 0 350 165"
        role="img" aria-labelledby="{adversary_title_id}">
        <title id="{adversary_title_id}">Named adversaries; bubble size represents
          adversarial paragraph count and color represents adversary type</title>
        {adversaries}
      </svg>
      <div class="era-adversary-legend"
        aria-label="All entity types in the corpus">
        {adversary_legend}</div>
    </article>
    <article class="era-template-footprint">
      <header class="era-card-heading"><span>Corpus footprint</span></header>
      <div class="era-footprint-core">
        <p><strong>{footprint['speech_share']:.1f}%</strong>
          <span>Speeches</span><small>{footprint['speeches']} of
          {footprint['corpus_speeches']:,}</small></p>
        <p><strong>{footprint['paragraph_share']:.1f}%</strong>
          <span>Paragraphs</span><small>{footprint['paragraphs']:,} of
          {footprint['corpus_paragraphs']:,}</small></p>
        <p><strong>{footprint['word_share']:.1f}%</strong>
          <span>Words</span><small>{footprint['words']:,} of
          {footprint['corpus_words']:,}</small></p>
      </div>
      <div class="era-footprint-language">{distinctive_words}</div>
      <details class="era-distinctive-method">
        <summary>How the distinctive-word ranking works</summary>
        <small><strong>× other eras</strong> compares the word's per-word rate
          here with its rate across all other eras. Inflected forms use the
          same families as Explore, so slavery, slave, and slaves count
          together. Rank emphasizes the smoothed rate difference; evidence
          uses 1 − exp(−era family uses ÷
          {eligibility['evidence_scale_uses']}) so it rises quickly and then
          levels off; concentration beyond
          {eligibility['effect_cap_ratio']}× receives no additional ranking
          credit. The printed multiplier remains uncapped. Presidential names
          are excluded.
          Eligibility:
          ≥{eligibility['min_corpus_uses']} corpus uses ·
          ≥{eligibility['min_era_speeches']} era speeches ·
          ≥{eligibility['min_era_presidents']} presidents.</small>
      </details>
    </article>
    <article class="era-template-topics">
      <header class="era-card-heading"><span>Major Topics</span></header>
      <ul class="era-topic-bars">{topics}</ul>
      <small>{html.escape(profile['topic_note'])}</small></article>
  </div>
</section>"""


def _founding_era_profile_html(data: dict) -> str:
    return _era_profile_html(
        data["era_profile"],
        panel_id="founding-profile",
        tab_id="founding-tab-profile",
    )


def _founding_problem_summary_html(data: dict) -> str:
    external = next(
        problem for problem in data["problems"] if problem["key"] == "external"
    )
    native = next(
        problem for problem in data["problems"] if problem["key"] == "native"
    )
    capacity = next(
        problem for problem in data["problems"] if problem["key"] == "capacity"
    )
    capacity_names = {
        "Public Debt, Revenue & Treasury Finance": "Public finance",
        "Crime, Insurrection & Federal Law Enforcement": "Federal law",
        "Executive Power, Vetoes & the Courts": "Executive / courts",
    }
    capacity_breakout = "".join(
        f"<div><dt>{capacity_names[component['topic']]}</dt>"
        f"<dd>{component['share']:.1f}%</dd></div>"
        for component in capacity["components"]
    )
    return f"""<div class="founding-problem-grid founding-problem-grid-combined">
  <article class="founding-problem founding-problem-sovereignty">
    <p class="founding-problem-value">{data['sovereignty_security_share']:.1f}%</p>
    <h4>Sovereignty &amp; security</h4>
    <p>External affairs <em>or</em> the distinct Native-affairs label.</p>
    <dl class="founding-problem-breakout">
      <div><dt>External</dt><dd>{external['share']:.1f}%</dd></div>
      <div><dt>Native affairs</dt><dd>{native['share']:.1f}%</dd></div>
      <div><dt>Overlap</dt>
        <dd>{data['sovereignty_security_overlap_share']:.1f}%</dd></div>
    </dl>
  </article>
  <article class="founding-problem founding-problem-capacity">
    <p class="founding-problem-value">{capacity['share']:.1f}%</p>
    <h4>Federal capacity</h4>
    <p>Finance, enforcement, executive power &amp; courts.</p>
    <dl class="founding-problem-breakout founding-capacity-breakout">
      {capacity_breakout}
    </dl>
  </article>
</div>
<p class="founding-overlap"><strong>Combined lens, not reclassification.</strong>
  Native sovereignty remains historically distinct within the governed topic
  taxonomy.</p>"""


def _era_presidential_agendas_html(visualization: dict) -> str:
    """Render one vertical agenda card for every president in an era."""
    agenda = visualization["agenda"]
    categories = agenda["categories"]
    compositions = agenda["compositions"]
    if not categories or not compositions:
        return "<p>Presidential topic assignments are not available.</p>"
    supported = set(agenda["supported_presidents"])
    support_by_president = {
        row["president"]: row for row in agenda["president_support"]
    }
    supported_records = [
        record
        for record in compositions
        if record["president"] in supported
    ]
    thin_records = [
        record
        for record in compositions
        if record["president"] not in supported
    ]

    def card_html(record: dict) -> str:
        president = record["president"]
        support = support_by_president[president]
        appearance_word = (
            "appearance" if support["appearances"] == 1 else "appearances"
        )
        top_share = max(
            (segment["share"] for segment in record["segments"]),
            default=0.0,
        )
        priorities = []
        for segment in record["segments"]:
            share = float(segment["share"])
            width = (
                0.0
                if top_share <= 0
                else min(share / top_share * 100, 100)
            )
            detail = (
                f"{president}'s number {segment['president_rank']} policy "
                f"domain, {segment['full_label']}: appeared in "
                f"{segment['count']:,} of "
                f"{segment['denominator_paragraphs']:,} paragraphs "
                f"({share:.1f}%). {segment['definition']} Multi-label "
                "paragraphs count once in every applicable policy domain."
            )
            priorities.append(
                '<div class="agenda-card-priority" '
                f'style="--agenda-width:{width:.8f}%;'
                f'--agenda-color:{segment["color"]}" '
                f'role="group" aria-label="{html.escape(detail, quote=True)}">'
                '<div class="agenda-card-priority-heading">'
                f'<span><b>#{segment["president_rank"]}</b> '
                f'{html.escape(segment["label"])}</span>'
                f"<strong>{share:.1f}%</strong></div>"
                '<span class="agenda-card-bar" aria-hidden="true"><i></i></span>'
                '<details class="agenda-card-description">'
                f'<summary>{html.escape(segment["definition"])}</summary>'
                f'<p>{html.escape(segment["definition"])}</p></details></div>'
            )
        return (
            '<article class="agenda-card" data-agenda-card>'
            '<header class="agenda-card-president">'
            f'<img src="portraits/{profiles.slug(president)}.png" alt="">'
            '<div>'
            f"<h5>{html.escape(president)}</h5>"
            f'<p>{support["appearances"]} source {appearance_word} · '
            f'{support["paragraphs"]:,} speaker-audited paragraphs</p>'
            "</div></header>"
            f'<div class="agenda-card-priorities">{"".join(priorities)}</div>'
            "</article>"
        )

    def cards_html(records: list[dict], *, thin: bool = False) -> str:
        if not records:
            return ""
        attribute = (
            "data-agenda-thin-cards" if thin else "data-agenda-cards"
        )
        thin_class = " agenda-card-grid-is-thin" if thin else ""
        scroll_class = (
            " agenda-card-grid-scroll" if len(records) > 3 else ""
        )
        scroll_label = (
            ' role="region" aria-label="Scrollable presidential agenda cards"'
            if len(records) > 3 else ""
        )
        return (
            f'<div class="agenda-card-grid{thin_class}{scroll_class}" '
            f'{attribute}{scroll_label}>'
            f'{"".join(card_html(record) for record in records)}</div>'
        )

    supported_cards = cards_html(supported_records)
    thin_html = ""
    if thin_records:
        thin_names = ", ".join(
            support["president"]
            for support in agenda["president_support"]
            if support["status"] == "thin"
        )
        thin_html = f"""<details class="agenda-thin-support">
  <summary>Thin support · {html.escape(thin_names)}</summary>
  <p>These records use the same presence measure, but fewer than
    {agenda['minimum_supported_appearances']} source-document appearances makes
    their apparent agenda unusually sensitive to individual sources. They use
    the same card design as the supported records but remain separate supporting
    evidence.</p>
  {cards_html(thin_records, thin=True)}
</details>"""
    sensitivity = agenda["weighting_sensitivity"]
    return f"""<header class="founding-visual-question">
  <div>
    <span>Presidential agendas</span>
    <h4>What defined each president’s agenda?</h4>
  </div>
  <p>Each card keeps a president’s five leading policy domains, their measured
    paragraph shares, and their meanings together.</p>
</header>
<p class="agenda-card-note">Within each card, the leading domain fills the bar.
  The printed percentages are the comparable values across presidents.</p>
{supported_cards}
<details class="agenda-method-note">
  <summary>What the percentages mean</summary>
  <p>{html.escape(agenda["measure"])} <strong>Selection:</strong>
    {html.escape(agenda["selection"])} <strong>Weighting check:</strong>
    Raw-word presence preserves the leading domain for
    {sensitivity['same_leading_domain']} of
    {sensitivity['supported_records']} supported records in this era and the
    complete top five for {sensitivity['same_top_five_set']}.</p>
</details>
{thin_html}"""


def _founding_presidential_agendas_html(data: dict) -> str:
    """Compatibility wrapper for the Founding-story tests and template."""
    return _era_presidential_agendas_html(data["era_visualization"])


def _era_governance_visual_html(visualization: dict) -> str:
    """Compare one era president's channels against muted peer benchmarks."""
    governance = visualization["governance"]
    presidents = governance["presidents"]
    dense = len(presidents) > 6
    benchmark_lanes = 4 if dense else 3
    prefix = profiles.slug(visualization["key"])
    color_by_president = {
        row["president"]: row["color"] for row in presidents
    }
    audience_rows = governance["audience_options"]
    medium_rows = governance["medium_options"]
    audience_labels = {
        "Congress": ("🏛️", "Congress"),
        "General public": ("👥", "Public"),
        "Specific organizations or groups": ("🤝", "Groups"),
        "Other audiences": ("🌐", "Other"),
    }
    medium_labels = {
        "Written message": ("✉️", "Written"),
        "Spoken address": ("🗣️", "Spoken"),
        "Radio/TV broadcast": ("📻", "Radio / TV"),
        "Press conference/debate or other performed form": ("🎙️", "Press +"),
    }
    controls = "".join(
        f'<button type="button" class="founding-governance-president" '
        f'id="{prefix}-governance-president-'
        f'{profiles.slug(row["president"])}" '
        f'data-governance-president="{html.escape(row["president"], quote=True)}" '
        f'aria-pressed="{"true" if index == 0 else "false"}" '
        f'style="--president-color:{color_by_president[row["president"]]}">'
        f'<img src="portraits/{profiles.slug(row["president"])}.png" alt="">'
        f'<span><strong>{html.escape(row["president"].split()[-1])}</strong>'
        f'<small>{row["appearances"]} appearances</small></span></button>'
        for index, row in enumerate(presidents)
    )

    def distribution_rows(
        selected: dict,
        channel_rows: list[dict],
        key: str,
        labels: dict[str, tuple[str, str]],
    ) -> str:
        selected_map = {row["label"]: row for row in selected[key]}
        benchmark_maps = {
            row["president"]: {item["label"]: item for item in row[key]}
            for row in presidents
            if row["president"] != selected["president"]
        }
        rendered = []
        for channel in channel_rows:
            label = channel["label"]
            value = selected_map[label]
            benchmarks = []
            comparison_parts = []
            for benchmark_index, (president, values) in enumerate(
                benchmark_maps.items()
            ):
                benchmark = values[label]
                surname = president.split()[-1]
                comparison_parts.append(
                    f"{surname} {benchmark['share']:.1f}%"
                )
                tip = (
                    f"{president} · {label}: {benchmark['share']:.1f}% · "
                    f"{benchmark['count']} appearances"
                )
                benchmarks.append(
                    f'<i class="founding-governance-benchmark" '
                    f'style="--benchmark-share:{benchmark["share"]:.4f}%;'
                    f'--benchmark-row:{benchmark_index % benchmark_lanes};'
                    f'--benchmark-stack:{benchmark_index // benchmark_lanes}" '
                    f'data-tooltip="{html.escape(tip, quote=True)}" '
                    f'aria-hidden="true" title="{html.escape(tip, quote=True)}">'
                    f'<img class="founding-governance-peer-portrait" '
                    f'src="portraits/{profiles.slug(president)}.png" alt=""></i>'
                )
            selected_tip = (
                f"{selected['president']} · {label}: {value['share']:.1f}% · "
                f"{value['count']} of {selected['appearances']} appearances; "
                f"comparison: {', '.join(comparison_parts)}"
            )
            icon, short_label = labels[label]
            rendered.append(
                f'<div class="founding-governance-bar" role="img" tabindex="0" '
                f'aria-label="{html.escape(selected_tip, quote=True)}">'
                f'<div class="founding-governance-channel" '
                f'title="{html.escape(label, quote=True)}">'
                f'<span aria-hidden="true">{icon}</span>'
                f'<strong>{html.escape(short_label)}</strong></div>'
                f'<div class="founding-governance-track" '
                f'style="--selected-share:{value["share"]:.4f}%">'
                f'<i class="founding-governance-fill" aria-hidden="true"></i>'
                f'{"".join(benchmarks)}</div>'
                f'<strong class="founding-governance-value">'
                f'{value["share"]:.1f}%</strong></div>'
            )
        return "".join(rendered)

    panels = []
    for index, row in enumerate(presidents):
        name = row["president"]
        route = row["routes"][0]
        panels.append(
            f'<article class="founding-governance-panel" '
            f'data-governance-panel="{html.escape(name, quote=True)}" '
            f'aria-labelledby="{prefix}-governance-president-'
            f'{profiles.slug(name)}" '
            f'style="--president-color:{color_by_president[name]}"'
            f'{" hidden" if index else ""}>'
            f'<div class="founding-governance-distributions">'
            f'<section><h4>Who they addressed</h4>'
            f'{distribution_rows(row, audience_rows, "audience", audience_labels)}'
            f'</section><section><h4>How they communicated</h4>'
            f'{distribution_rows(row, medium_rows, "medium", medium_labels)}'
            f'</section></div>'
            f'<aside class="founding-governance-takeaway">'
            f'<span>Most common route</span>'
            f'<p><strong>{html.escape(route["audience"])} '
            f'<b aria-hidden="true">→</b> {html.escape(route["medium"])}</strong> '
            f'was {html.escape(name.split()[-1])}’s most common combination of '
            f'intended audience and communication form: '
            f'{route["count"]} of {row["appearances"]} appearances '
            f'({route["share"]:.1f}%).</p></aside></article>'
        )
    return f"""<header class="founding-visual-question">
  <div>
    <span>Governing channels</span>
    <h4>Who did each president address—and in what form?</h4>
  </div>
  <p>Selected president = filled bars · peers = muted comparison markers</p>
</header>
<div class="founding-governance-filter" data-governance-filter
  data-governance-density="{"dense" if dense else "regular"}">
  <div class="founding-governance-selector" role="group"
    aria-label="Choose a president">{controls}</div>
  <div class="founding-governance-stage">{"".join(panels)}</div>
  <p class="sr-only" data-governance-status aria-live="polite">
    {html.escape(presidents[0]["president"])} selected.</p>
</div>
<p class="founding-chart-note">Every bar uses the same 0–100% appearance-share
  scale. Audience and primary form are separate distributions; the note records
  their most common exact intersection.</p>"""


def _founding_governance_visual_html(data: dict) -> str:
    """Compatibility wrapper for the Founding-story tests and template."""
    return _era_governance_visual_html(data["era_visualization"])


def _era_adversary_network_html(visualization: dict) -> str:
    """Render one era's compact president-to-adversary network."""
    adversary_data = visualization["adversaries"]
    edges = adversary_data["edges"]
    if not edges:
        return "<p>Named adversary annotations are not available.</p>"

    presidents = [row["name"] for row in visualization["presidents"]]
    color_by_president = {
        row["name"]: row["color"] for row in visualization["presidents"]
    }
    summary_by_president = {
        row["president"]: row
        for row in adversary_data["president_summary"]
    }
    raw_scores = {
        name: np.sqrt(
            summary_by_president[name]["word_share"]
            * summary_by_president[name]["named_adversaries"]
        )
        for name in presidents
    }
    max_score = max(raw_scores.values())
    president_radii = {
        name: (
            7.0
            if raw_scores[name] == 0
            else 26 * np.sqrt(raw_scores[name] / max_score)
        )
        for name in presidents
    }
    adversary_totals = {}
    for edge in edges:
        adversary_totals[edge["adversary"]] = (
            adversary_totals.get(edge["adversary"], 0) + edge["paragraphs"]
        )
    adversaries = sorted(
        adversary_totals,
        key=lambda name: (-adversary_totals[name], name),
    )
    president_keys = {
        name: profiles.slug(f"president {name}") for name in presidents
    }
    adversary_keys = {
        name: profiles.slug(f"adversary {index} {name}")
        for index, name in enumerate(adversaries)
    }
    height = min(
        620,
        max(420, 70 + max(len(adversaries), len(presidents)) * 24),
    )
    president_y = (
        {presidents[0]: height / 2}
        if len(presidents) == 1
        else {
            name: 42 + index * ((height - 84) / (len(presidents) - 1))
            for index, name in enumerate(presidents)
        }
    )
    adversary_y = {
        name: 24 + index * ((height - 48) / max(1, len(adversaries) - 1))
        for index, name in enumerate(adversaries)
    }
    paths = []
    for edge in edges:
        y1 = president_y[edge["president"]]
        y2 = adversary_y[edge["adversary"]]
        width = min(15.0, 1.5 + np.sqrt(edge["paragraphs"]) * 1.25)
        start_x = 198 + president_radii[edge["president"]]
        adversary_radius = min(12, 4 + np.sqrt(adversary_totals[edge["adversary"]]))
        end_x = 674 - adversary_radius
        raw = ", ".join(edge["raw_entities"])
        tip = (
            f"Actual speaker {edge['president']} → {edge['adversary']}: "
            f"{edge['paragraphs']} adversarial paragraphs; "
            f"{edge['ai_ner_paragraphs']} AI + NER and "
            f"{edge['ai_only_paragraphs']} AI only; "
            f"{edge['cross_owner_paragraphs']} cross-owner"
        )
        paths.append(
            f'<path class="founding-network-edge" '
            f'data-adversary-edge '
            f'data-president-key="{president_keys[edge["president"]]}" '
            f'data-adversary-key="{adversary_keys[edge["adversary"]]}" '
            f'tabindex="0" role="img" aria-label="{html.escape(tip, quote=True)}" '
            f'd="M {start_x:.2f} {y1:.2f} C 390 {y1:.2f}, '
            f'500 {y2:.2f}, {end_x:.2f} {y2:.2f}" '
            f'stroke="{color_by_president[edge["president"]]}" '
            f'style="--adversary-edge-width:{width:.2f}px" '
            f'stroke-width="{width:.2f}">'
            f"<title>{html.escape(tip)} · raw: {html.escape(raw)}</title></path>"
        )
    left_nodes = "".join(
        f'<g class="founding-adversary-node founding-adversary-president" '
        f'data-adversary-node data-node-kind="president" '
        f'data-president-key="{president_keys[name]}" tabindex="0" '
        f'role="img" aria-label="Actual speaker {html.escape(name, quote=True)}">'
        f'<circle class="founding-president-node" cx="198" '
        f'cy="{president_y[name]:.2f}" r="{president_radii[name]:.2f}">'
        f'<title>Actual speaker {html.escape(name)}: '
        f'{summary_by_president[name]["word_share"]:.1f}% of words in paragraphs '
        f'naming an adversary; {summary_by_president[name]["named_adversaries"]} '
        f'distinct named adversaries</title></circle>'
        f'<text x="164" y="{president_y[name] - 3:.2f}" text-anchor="end">'
        f'{html.escape(name.split()[-1])}</text>'
        f'<text class="founding-network-meta" x="164" '
        f'y="{president_y[name] + 11:.2f}" text-anchor="end">'
        f'{summary_by_president[name]["word_share"]:.1f}% · '
        f'{summary_by_president[name]["named_adversaries"]} names</text></g>'
        for name in presidents
    )
    right_nodes = "".join(
        f'<g class="founding-adversary-node founding-adversary-target" '
        f'data-adversary-node data-node-kind="adversary" '
        f'data-adversary-key="{adversary_keys[name]}" tabindex="0" '
        f'role="img" aria-label="{html.escape(name, quote=True)}">'
        f'<circle cx="674" cy="{adversary_y[name]:.2f}" '
        f'r="{min(12, 4 + np.sqrt(adversary_totals[name])):.2f}"></circle>'
        f'<text x="694" y="{adversary_y[name] + 4:.2f}">'
        f'{html.escape(name)} · {adversary_totals[name]}</text></g>'
        for name in adversaries
    )
    return f"""<header class="founding-visual-question">
  <div>
    <span>Adversary network</span>
    <h4>Who named adversaries most intensively?</h4>
  </div>
  <p>President node score = √(adversary-word share × distinct named adversaries);
  node area scales to that score. All president labels identify actual speakers.</p>
</header>
<div class="founding-network-scroll">
  <svg class="founding-adversary-network" viewBox="0 0 900 {height}"
    data-adversary-network
    role="img" aria-label="Actual presidential speakers connected to their most frequently named adversaries; president node area combines adversary-word share and distinct adversary count; line width represents adversarial paragraph count">
    {"".join(paths)}{left_nodes}{right_nodes}
  </svg>
</div>
<p class="founding-chart-note">Percentages use words in distinct paragraphs that
name at least one adversary; name counts use all normalized named adversaries.
The network shows leading connections only. Line width = distinct adversarial
paragraphs; line details report AI and NER source agreement, cross-owner support,
and exact primary-AI names.</p>"""


def _founding_adversary_network_html(data: dict) -> str:
    """Compatibility wrapper for the Founding-story tests and template."""
    return _era_adversary_network_html(data["era_visualization"])


def _founding_ribbon_path(
    x_positions: list[float],
    y_position: float,
    values: list[float],
) -> str:
    widths = [6 + value * .34 for value in values]
    top = f"M {x_positions[0]} {y_position - widths[0] / 2}"
    for index in range(1, len(x_positions)):
        midpoint = (x_positions[index - 1] + x_positions[index]) / 2
        top += (
            f" C {midpoint} {y_position - widths[index - 1] / 2},"
            f" {midpoint} {y_position - widths[index] / 2},"
            f" {x_positions[index]} {y_position - widths[index] / 2}"
        )
    bottom = f" L {x_positions[-1]} {y_position + widths[-1] / 2}"
    for index in range(len(x_positions) - 2, -1, -1):
        midpoint = (x_positions[index + 1] + x_positions[index]) / 2
        bottom += (
            f" C {midpoint} {y_position + widths[index + 1] / 2},"
            f" {midpoint} {y_position + widths[index] / 2},"
            f" {x_positions[index]} {y_position + widths[index] / 2}"
        )
    return f"{top}{bottom} Z"


def _founding_four_tests_html(data: dict) -> str:
    """Compatibility wrapper for the approved Founding Era Defined graph."""
    return _era_defined_html(data["era_contextualization"])


def _founding_attention_departures_html(data: dict) -> str:
    """Put two founding-specific disappearances beside the afterlife chart."""
    cards = []
    for row in data["attention_departures"]:
        present = row["shares"][-1]
        cards.append(
            f"""<article>
  <span class="founding-departure-icon" aria-hidden="true">{row['icon']}</span>
  <div><strong>{html.escape(row['label'])}</strong>
    <p>{html.escape(row['note'])}</p></div>
  <div class="founding-departure-change">
    <b>{row['shares'][0]:.1f}%</b><small>Founding</small>
    <i aria-hidden="true">→</i>
    <b>{present:.2f}%</b><small>Present</small>
  </div>
</article>"""
        )
    return f"""<aside class="founding-attention-departures" role="note">
  <div class="founding-departure-heading">
    <span>What drops out of presidential attention</span>
    <strong>Two founding concerns stop occupying the same share of the formal record.</strong>
  </div>
  <div class="founding-departure-grid">{"".join(cards)}</div>
</aside>"""


def _story_communication_constellation_html(data: dict) -> str:
    """Compact story-wide audience/medium summary using size-scaled symbols."""
    audience_icons = {
        "Congress": ("🏛️", "Congress"),
        "General public": ("👥", "General public"),
        "Specific organizations or groups": ("🤝", "Specific groups"),
        "Other audiences": ("🌐", "Other audiences"),
    }
    medium_icons = {
        "Written message": ("✉️", "Written message"),
        "Spoken address": ("🎙️", "Spoken address"),
        "Radio/TV broadcast": ("📻", "Radio / TV"),
        "Press conference/debate or other performed form": (
            "🗣️",
            "Press / debate",
        ),
    }

    def constellation(groups: list[dict], icons: dict, label: str) -> str:
        symbols = []
        for group in groups:
            emoji, short = icons[group["label"]]
            size = 13 + np.sqrt(group["share"] / 100) * 31
            opacity = .28 if group["share"] == 0 else .55 + group["share"] / 225
            tip = (
                f"{short}: {group['share']:.1f}% "
                f"({group['count']} speeches)"
            )
            symbols.append(
                f'<span class="communication-symbol" '
                f'style="--symbol-size:{size:.2f}px;--symbol-opacity:{opacity:.3f}" '
                f'role="img" aria-label="{html.escape(tip, quote=True)}" '
                f'title="{html.escape(tip, quote=True)}">{emoji}</span>'
            )
        return (
            f'<div class="communication-constellation" '
            f'aria-label="{html.escape(label, quote=True)}">'
            f'{"".join(symbols)}</div>'
        )

    cards = []
    for era in data["communications"]:
        cards.append(
            f"""<article class="communication-era-card">
  <header><strong>{html.escape(era['short'])}</strong><span>{era['years']}</span></header>
  <div><small>Who</small>{constellation(era['audience'], audience_icons, 'Audience shares')}</div>
  <div><small>How</small>{constellation(era['medium'], medium_icons, 'Medium shares')}</div>
</article>"""
        )
    founding = data["communications"][0]
    war = data["communications"][5]
    founding_congress = next(
        group for group in founding["audience"] if group["label"] == "Congress"
    )
    founding_written = next(
        group for group in founding["medium"] if group["label"] == "Written message"
    )
    war_public = next(
        group for group in war["audience"] if group["label"] == "General public"
    )
    war_broadcast = next(
        group for group in war["medium"] if group["label"] == "Radio/TV broadcast"
    )
    agreement = data["agreement"]
    return f"""<section class="communication-summary" aria-labelledby="communication-summary-title">
  <div class="communication-summary-heading">
    <span>Story-wide summary</span>
    <h3 id="communication-summary-title">Who presidents addressed—and how</h3>
    <p>Symbol size tracks the share of corpus speeches in each named era. The
    founding pair is 🏛️ Congress ({founding_congress['share']:.1f}%) and ✉️ written
    messages ({founding_written['share']:.1f}%); by War/New Deal it is 👥 the
    general public ({war_public['share']:.1f}%) and 📻 assigned broadcast form
    ({war_broadcast['share']:.1f}%).</p>
  </div>
  <div class="communication-legend" aria-label="Symbol legend">
    <span>Audience: 🏛️ Congress · 👥 public · 🤝 groups · 🌐 other</span>
    <span>Medium: ✉️ written · 🎙️ spoken · 📻 radio/TV · 🗣️ press/debate</span>
  </div>
  <div class="communication-summary-scroll">
    <div class="communication-era-strip">{"".join(cards)}</div>
  </div>
  <p class="communication-summary-note">AI-assigned factual classifications:
  {agreement['audience']['rate']:.1f}% audience and
  {agreement['medium']['rate']:.1f}% medium exact match in a
  {agreement['n']}-speech second-model sample. One assigned audience or medium
  is not a complete inventory of circulation or reception.</p>
</section>"""


def _summary_communication_mosaics_html(data: dict) -> str:
    """Two independent, area-proportional communication mosaics for Summary."""
    dimensions = (
        (
            "audience",
            "WHO · assigned audience",
            "Who presidents addressed",
            "Each era box divides the full speech record among four assigned audiences.",
            (
                ("Congress", "🏛️", "Congress", "#315f78", "#ffffff"),
                ("General public", "👥", "General public", "#b16b3e", "#ffffff"),
                (
                    "Specific organizations or groups",
                    "🤝",
                    "Specific groups",
                    "#c9bda9",
                    "#302b26",
                ),
                (
                    "Other audiences",
                    "🌐",
                    "Press, diplomatic, military + other",
                    "#74695f",
                    "#ffffff",
                ),
            ),
        ),
        (
            "medium",
            "HOW · assigned delivery form",
            "How presidents delivered the address",
            "Each era box divides the full speech record among four assigned forms.",
            (
                ("Written message", "✉️", "Written", "#5f568c", "#ffffff"),
                ("Spoken address", "🎙️", "Spoken", "#4f7557", "#ffffff"),
                ("Radio/TV broadcast", "📻", "Radio / TV", "#c58a3a", "#302b26"),
                (
                    "Press conference/debate or other performed form",
                    "🗣️",
                    "Press / debate",
                    "#a24f52",
                    "#ffffff",
                ),
            ),
        ),
    )

    def graph(
        dimension: str,
        eyebrow: str,
        title: str,
        description: str,
        specs: tuple[tuple[str, str, str, str, str], ...],
    ) -> str:
        graph_id = f"summary-{dimension}-mosaic-title"
        legend = "".join(
            f'<span style="--mosaic-color:{color}"><b aria-hidden="true">'
            f'{emoji}</b><span><strong>{html.escape(short)}</strong>'
            f'<small>{html.escape(label)}</small></span></span>'
            for label, emoji, short, color, _ in specs
        )
        cards = []
        for era in data["communications"]:
            values = {row["label"]: row for row in era[dimension]}
            expected = {spec[0] for spec in specs}
            if set(values) != expected:
                raise ValueError(
                    f"Summary {dimension} mosaic categories drifted: "
                    f"{sorted(set(values) ^ expected)}"
                )
            total = sum(float(values[label]["share"]) for label in expected)
            if not np.isclose(total, 100, atol=.05):
                raise ValueError(
                    f"Summary {dimension} mosaic does not sum to 100% for "
                    f"{era['short']}: {total}"
                )
            rows_html = []
            for pair in (specs[:2], specs[2:]):
                row_total = sum(float(values[spec[0]]["share"]) for spec in pair)
                if row_total <= 0:
                    continue
                cells = []
                for label, emoji, short, color, text_color in pair:
                    item = values[label]
                    share = float(item["share"])
                    if share <= 0:
                        continue
                    within_row = share / row_total * 100
                    size_class = (
                        "is-tiny" if share < 6 else
                        "is-small" if share < 18 else
                        "is-medium" if share < 28 else
                        "is-large"
                    )
                    tip = (
                        f"{short}: {share:.1f}% "
                        f"({item['count']} of {era['total']} speeches)"
                    )
                    cells.append(
                        f'<div class="communication-mosaic-cell {size_class}" '
                        f'style="--cell-share:{within_row:.6f}%;'
                        f'--mosaic-color:{color};--mosaic-text:{text_color}" '
                        f'data-share="{share:.6f}" tabindex="0" '
                        f'aria-label="{html.escape(tip, quote=True)}" '
                        f'title="{html.escape(tip, quote=True)}">'
                        f'<span aria-hidden="true" class="communication-mosaic-emoji">'
                        f'{emoji}</span><strong aria-hidden="true">{share:.1f}%</strong>'
                        f'<small aria-hidden="true">{html.escape(short)}</small></div>'
                    )
                rows_html.append(
                    f'<div class="communication-mosaic-row" '
                    f'style="--row-share:{row_total:.6f}%" '
                    f'data-row-share="{row_total:.6f}">{"".join(cells)}</div>'
                )
            breakdown = "; ".join(
                f"{short}: {float(values[label]['share']):.1f}%"
                for label, _, short, _, _ in specs
            )
            readout = "".join(
                f'<span title="{html.escape(label, quote=True)}">'
                f'<b aria-hidden="true">{emoji}</b> '
                f'{float(values[label]["share"]):.1f}%</span>'
                for label, emoji, _, _, _ in specs
            )
            cards.append(
                f'<article class="communication-mosaic-card" '
                f'data-dimension="{dimension}" data-composition-total="{total:.6f}" '
                f'aria-label="{html.escape(str(era["short"]), quote=True)}: '
                f'{html.escape(breakdown, quote=True)}">'
                f'<header><strong>{html.escape(str(era["short"]))}</strong>'
                f'<span>{html.escape(str(era["years"]))}</span>'
                f'<small>n = {era["total"]} speeches</small></header>'
                f'<div class="communication-mosaic-box">{"".join(rows_html)}</div>'
                f'<p class="communication-mosaic-readout">{readout}</p></article>'
            )
        return f"""<section class="communication-mosaic communication-mosaic-{dimension}"
    data-category-count="4" aria-labelledby="{graph_id}">
  <header class="communication-mosaic-heading">
    <p>{html.escape(eyebrow)}</p>
    <h3 id="{graph_id}">{html.escape(title)}</h3>
    <span>{html.escape(description)} Region area equals percentage; the exact values remain below each box.</span>
  </header>
  <div class="communication-mosaic-key" aria-label="{html.escape(eyebrow)} legend">{legend}</div>
  <p class="communication-mosaic-scroll-cue" aria-hidden="true">Founding → Present · scroll across eras</p>
  <div class="communication-mosaic-scroll" tabindex="0"
       aria-label="{html.escape(title)} across nine eras">
    <div class="communication-mosaic-strip">{"".join(cards)}</div>
  </div>
</section>"""

    return (
        '<div class="communication-mosaic-suite">'
        + "".join(graph(*dimension) for dimension in dimensions)
        + "</div>"
    )


def _summary_communication_audit_html(data: dict) -> str:
    """Plain-language category definitions and reliability receipt for Summary."""
    agreement = data["agreement"]
    return f"""<div class="communication-audit-grid">
  <section>
    <h4>WHO classifications</h4>
    <ul>
      <li>🏛️ <strong>Congress</strong></li>
      <li>👥 <strong>General public</strong></li>
      <li>🤝 <strong>Specific organizations or groups</strong></li>
      <li>🌐 <strong>Other audiences</strong>: press, foreign or diplomatic,
      military, and residual other assignments</li>
    </ul>
  </section>
  <section>
    <h4>HOW classifications</h4>
    <ul>
      <li>✉️ <strong>Written message</strong></li>
      <li>🎙️ <strong>Spoken address</strong></li>
      <li>📻 <strong>Radio/TV broadcast</strong></li>
      <li>🗣️ <strong>Press conference, debate, or other performed form</strong></li>
    </ul>
  </section>
</div>
<p class="communication-summary-note">AI-assigned factual classifications:
{agreement['audience']['rate']:.1f}% audience and
{agreement['medium']['rate']:.1f}% medium exact match in a
{agreement['n']}-speech second-model sample. One assigned audience or medium
is not a complete inventory of circulation or reception.</p>"""


def _founding_threads_html(data: dict) -> str:
    eras = data["communications"]
    rows = data["threads"]
    max_share = max(max(row["shares"]) for row in rows)
    headers = (
        '<div class="founding-thread-corner" title="Topic labels">'
        '<span aria-hidden="true">◉</span>'
        '<span class="founding-thread-copy">Founding concern</span></div>'
    ) + "".join(
        f'<div class="founding-thread-era"><strong>{html.escape(era["short"])}</strong>'
        f'<span>{era["years"]}</span></div>'
        for era in eras
    )

    def render_grid_row(row: dict) -> str:
        label_class = ""
        if row["label"] == FOUNDING_NATIVE_TOPIC:
            label_class += " founding-thread-highlight"
        score = f"Persistence score {row['persistence_score'] * 100:.1f}%."
        icon = FOUNDING_THREAD_ICONS[row["label"]]
        result = [
            f'<div class="founding-thread-label{label_class}" '
            f'style="--row-color:{row["color"]}" '
            f'title="{html.escape(row["label"], quote=True)}" '
            f'aria-label="{html.escape(row["label"], quote=True)}">'
            f'<strong><span class="founding-thread-icon" aria-hidden="true">'
            f'{icon}</span><span class="founding-thread-copy">'
            f'{html.escape(row["label"])}</span></strong>'
            f'<span class="founding-thread-copy">'
            f'{html.escape(row["descriptor"])}</span>'
            f'<small class="founding-thread-copy">{score}</small>'
            f'</div>'
        ]
        for era, share, count, denominator in zip(
            eras, row["shares"], row["counts"], row["denominators"]
        ):
            cell_class = "founding-bubble-cell"
            if row["label"] == FOUNDING_NATIVE_TOPIC:
                cell_class += " founding-thread-highlight-cell"
            if share <= 0:
                result.append(
                    f'<div class="{cell_class}"><span class="founding-zero" '
                    f'aria-label="{html.escape(row["label"], quote=True)}, '
                    f'{html.escape(era["short"], quote=True)}: 0 of {denominator} '
                    f'paragraphs">—</span></div>'
                )
            else:
                size = max(1.0, np.sqrt(share / max_share) * 38)
                tip = (
                    f"{row['label']} · {era['short']}: {share:.1f}% · "
                    f"{count:,} of {denominator:,} paragraphs"
                )
                result.append(
                    f'<div class="{cell_class}"><button type="button" '
                    f'class="founding-bubble bubble-{row["pattern"]}" '
                    f'style="--bubble-size:{size:.2f}px;'
                    f'--row-color:{row["color"]}" '
                    f'data-tip="{html.escape(tip, quote=True)}" '
                    f'title="{html.escape(tip, quote=True)}" '
                    f'aria-label="{html.escape(tip, quote=True)}"></button></div>'
                )
        return "".join(result)

    grid_rows = [render_grid_row(row) for row in rows]

    table_rows = []
    for row in rows:
        table_cells = []
        for share, count, denominator in zip(
            row["shares"], row["counts"], row["denominators"]
        ):
            table_cells.append(
                f"<td>{share:.1f}% <small>({count:,}/{denominator:,})</small></td>"
            )
        table_rows.append(
            f'<tr><th scope="row">'
            f'{html.escape(row["label"])}</th>{"".join(table_cells)}</tr>'
        )
    table_headers = "".join(
        f"<th scope=\"col\">{html.escape(era['short'])}</th>" for era in eras
    )
    return f"""<div class="founding-thread-intro">
  <strong>Bubble area = paragraph share</strong>
  <span>9 eras · exact values on focus</span>
  <button type="button" class="founding-label-toggle"
    data-founding-label-toggle aria-expanded="false"
    aria-controls="founding-thread-grid">Show topic labels</button>
</div>
<div class="founding-thread-scroll">
  <div class="founding-thread-grid" id="founding-thread-grid" role="group"
       aria-label="Founding topic shares across all nine eras">{headers}
    {"".join(grid_rows)}
  </div>
</div>
<details class="founding-thread-table"><summary>Text alternative · exact shares and
paragraph counts by era</summary>
  <div class="table-scroll"><table><thead><tr><th scope="col">Topic</th>
  {table_headers}</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>
</details>"""


def _expansion_era_defined_html(chart: dict) -> str:
    """Render one shared-axis step tree with diplomacy as the trunk."""
    plot_left = 66.0
    plot_right = 790.0
    plot_top = 51.0
    plot_bottom = 338.0
    year_start = chart["phases"][0]["start_year"]
    year_end = chart["phases"][-1]["end_year"] + 1

    def x_for_year(year: float) -> float:
        return plot_left + (
            (year - year_start) / (year_end - year_start)
        ) * (plot_right - plot_left)

    def y_for_share(share: float) -> float:
        return plot_bottom - (
            share / chart["scale_max"]
        ) * (plot_bottom - plot_top)

    phase_guides = []
    for index, phase in enumerate(chart["phases"]):
        x_start = x_for_year(phase["start_year"])
        x_end = x_for_year(phase["end_year"] + 1)
        center = (x_start + x_end) / 2
        phase_guides.append(
            f'<rect class="{"is-alt" if index % 2 else ""}" '
            f'x="{x_start:.2f}" y="{plot_top:.2f}" '
            f'width="{x_end - x_start:.2f}" '
            f'height="{plot_bottom - plot_top:.2f}"></rect>'
            f'<line x1="{x_start:.2f}" y1="{plot_top:.2f}" '
            f'x2="{x_start:.2f}" y2="{plot_bottom:.2f}"></line>'
            f'<text x="{center:.2f}" y="18" text-anchor="middle">'
            f'{html.escape(phase["years"])}</text>'
            f'<text class="expansion-defined-phase-name" '
            f'x="{center:.2f}" y="32" text-anchor="middle">'
            f'{html.escape(phase["label"])}</text>'
        )
    phase_guides.append(
        f'<line x1="{plot_right:.2f}" y1="{plot_top:.2f}" '
        f'x2="{plot_right:.2f}" y2="{plot_bottom:.2f}"></line>'
    )
    y_guides = []
    for tick in range(0, chart["scale_max"] + 1, 10):
        y = y_for_share(tick)
        y_guides.append(
            f'<line x1="{plot_left:.2f}" y1="{y:.2f}" '
            f'x2="{plot_right:.2f}" y2="{y:.2f}"></line>'
            f'<text x="{plot_left - 10:.2f}" y="{y + 3:.2f}" '
            f'text-anchor="end">{tick}%</text>'
        )

    def step_path(values: list[dict]) -> str:
        first_y = y_for_share(values[0]["share"])
        commands = [
            f"M {x_for_year(chart['phases'][0]['start_year']):.2f} "
            f"{first_y:.2f}"
        ]
        for index, (phase, value) in enumerate(
            zip(chart["phases"], values)
        ):
            end_x = x_for_year(phase["end_year"] + 1)
            value_y = y_for_share(value["share"])
            if index:
                commands.append(f"V {value_y:.2f}")
            commands.append(f"H {end_x:.2f}")
        return " ".join(commands)

    def node_shape(key: str, x: float, y: float) -> str:
        if key == "diplomacy":
            return (
                f'<path class="expansion-defined-node-shape" '
                f'd="M {x:.2f} {y - 6:.2f} L {x + 6:.2f} {y:.2f} '
                f'L {x:.2f} {y + 6:.2f} L {x - 6:.2f} {y:.2f} Z"></path>'
            )
        if key == "institutions":
            return (
                f'<rect class="expansion-defined-node-shape" '
                f'x="{x - 5.5:.2f}" y="{y - 5.5:.2f}" '
                f'width="11" height="11"></rect>'
            )
        if key == "territory":
            return (
                f'<path class="expansion-defined-node-shape" '
                f'd="M {x:.2f} {y - 6.5:.2f} L {x + 6.5:.2f} '
                f'{y + 5.5:.2f} L {x - 6.5:.2f} {y + 5.5:.2f} Z"></path>'
            )
        return (
            f'<circle class="expansion-defined-node-shape" '
            f'cx="{x:.2f}" cy="{y:.2f}" r="5.5"></circle>'
        )

    series_groups = []
    endpoints = []
    table_rows = []
    for series in chart["series"]:
        nodes = []
        table_cells = []
        peak_index = max(
            range(len(series["values"])),
            key=lambda index: series["values"][index]["share"],
        )
        for index, (phase, value) in enumerate(
            zip(chart["phases"], series["values"])
        ):
            phase_center = (
                phase["start_year"] + phase["end_year"] + 1
            ) / 2
            x = x_for_year(phase_center)
            y = y_for_share(value["share"])
            detail = (
                f"{series['label']} · {phase['label']} ({phase['years']}): "
                f"{value['share']:.1f}% ({value['count']:,} of "
                f"{value['denominator']:,} paragraphs)."
            )
            subseries = value.get("subseries", [])
            if subseries:
                detail += " " + "; ".join(
                    f"{row['label']} {row['share']:.1f}%"
                    for row in subseries
                ) + "."
            peak_label = ""
            if index == peak_index and series["key"] != "diplomacy":
                peak_label = (
                    f'<text class="expansion-defined-peak-value" '
                    f'x="{x:.2f}" y="{y - 11:.2f}" text-anchor="middle">'
                    f'{value["share"]:.1f}%</text>'
                )
            nodes.append(
                f'<g class="expansion-defined-node" tabindex="0" role="img" '
                f'aria-label="{html.escape(detail, quote=True)}">'
                f'{node_shape(series["key"], x, y)}{peak_label}'
                f"<title>{html.escape(detail)}</title></g>"
            )
            split = ""
            if subseries:
                split = "<br><small>" + " · ".join(
                    f"{html.escape(row['label'])} {row['share']:.1f}%"
                    for row in subseries
                ) + "</small>"
            table_cells.append(
                f'<td>{value["share"]:.1f}% '
                f'<small>({value["count"]:,}/{value["denominator"]:,})</small>'
                f"{split}</td>"
            )
        series_groups.append(
            f'<g class="expansion-defined-series expansion-defined-'
            f'{series["key"]}" style="--expansion-series:{series["color"]}">'
            f'<path class="expansion-defined-step" '
            f'd="{step_path(series["values"])}"></path>'
            f'{"".join(nodes)}</g>'
        )
        endpoints.append({
            "key": series["key"],
            "label": series["label"],
            "icon": series["icon"],
            "color": series["color"],
            "share": series["values"][-1]["share"],
            "actual_y": y_for_share(series["values"][-1]["share"]),
        })
        table_rows.append(
            f'<tr><th scope="row">{html.escape(series["label"])}</th>'
            f'{"".join(table_cells)}</tr>'
        )

    endpoint_positions = {}
    previous_y = plot_top - 18
    for endpoint in sorted(endpoints, key=lambda row: row["actual_y"]):
        label_y = max(endpoint["actual_y"], previous_y + 18)
        endpoint_positions[endpoint["key"]] = label_y
        previous_y = label_y
    overflow = max(endpoint_positions.values()) - (plot_bottom - 3)
    if overflow > 0:
        endpoint_positions = {
            key: value - overflow
            for key, value in endpoint_positions.items()
        }
    endpoint_labels = []
    for endpoint in endpoints:
        label_y = endpoint_positions[endpoint["key"]]
        endpoint_labels.append(
            f'<g class="expansion-defined-end-label" '
            f'style="--expansion-series:{endpoint["color"]}">'
            f'<path d="M {plot_right + 2:.2f} {endpoint["actual_y"]:.2f} '
            f'L 818 {label_y:.2f}"></path>'
            f'<circle cx="826" cy="{label_y:.2f}" r="3.5"></circle>'
            f'<text x="835" y="{label_y + 3.5:.2f}">'
            f'{html.escape(endpoint["label"])} · '
            f'{endpoint["share"]:.1f}%</text></g>'
        )

    diplomacy = next(
        series for series in chart["series"] if series["key"] == "diplomacy"
    )
    event_nodes = []
    events = []
    for index, marker in enumerate(chart["event_markers"]):
        phase_index = next(
            phase_index
            for phase_index, phase in enumerate(chart["phases"])
            if phase["start_year"] <= marker["year"] <= phase["end_year"]
        )
        x = x_for_year(marker["year"])
        y = y_for_share(diplomacy["values"][phase_index]["share"])
        label_y = y - 18 if index % 2 == 0 else y + 25
        detail = (
            f"{marker['year']} · {marker['label']}. {marker['detail']}"
        )
        event_nodes.append(
            f'<g class="expansion-defined-event-node" tabindex="0" role="img" '
            f'aria-label="{html.escape(detail, quote=True)}">'
            f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x:.2f}" '
            f'y2="{label_y + (-5 if index % 2 == 0 else 5):.2f}"></line>'
            f'<path d="M {x:.2f} {y - 6:.2f} L {x + 6:.2f} {y:.2f} '
            f'L {x:.2f} {y + 6:.2f} L {x - 6:.2f} {y:.2f} Z"></path>'
            f'<text x="{x:.2f}" y="{label_y:.2f}" '
            f'text-anchor="middle">{marker["year"]}</text>'
            f"<title>{html.escape(detail)}</title></g>"
        )
        events.append(
            f'<li><time>{marker["year"]}</time>'
            f'<strong>{html.escape(marker["label"])}</strong>'
            f'<span>{html.escape(marker["detail"])}</span></li>'
        )
    phase_cards = "".join(
        f'<article><span>{html.escape(phase["years"])}</span>'
        f'<strong>{html.escape(phase["label"])}</strong>'
        f'<p>{html.escape(phase["explanation"])}</p>'
        f'<small>{phase["denominator"]:,} paragraphs · '
        f'{html.escape(", ".join(phase["presidents"]))}</small></article>'
        for phase in chart["phases"]
    )
    table_headers = "".join(
        f'<th scope="col">{html.escape(phase["years"])}<br>'
        f'<small>{phase["denominator"]:,} paragraphs</small></th>'
        for phase in chart["phases"]
    )
    return f"""<div class="expansion-defined-phase-grid">{phase_cards}</div>
<div class="expansion-defined-scroll">
  <svg class="expansion-defined-chart" viewBox="0 0 1000 372"
    role="img" aria-labelledby="expansion-defined-title expansion-defined-desc">
    <title id="expansion-defined-title">A diplomatic trunk with three changing
      branches across the continental republic</title>
    <desc id="expansion-defined-desc">Treaties and diplomacy persist while war,
      federal institutions, and territorial conquest rise and fall. All four
      step lines show independent phase paragraph shares on one zero to
      {chart["scale_max"]} percent axis.</desc>
    <g class="expansion-defined-guides">{"".join(phase_guides)}</g>
    <g class="expansion-defined-y-guides">{"".join(y_guides)}</g>
    <text class="expansion-defined-axis-title" x="{plot_left:.0f}" y="45">
      Share of phase paragraphs</text>
    {"".join(
        group
        for series, group in sorted(
            zip(chart["series"], series_groups),
            key=lambda item: item[0]["key"] == "diplomacy",
        )
    )}
    <g class="expansion-defined-event-layer">{"".join(event_nodes)}</g>
    <g class="expansion-defined-end-labels">{"".join(endpoint_labels)}</g>
  </svg>
</div>
<div class="expansion-defined-key">
  <span>Step height = phase paragraph share</span>
  <span>Thick line = diplomatic trunk</span>
  <span>Independent, overlapping measures</span>
</div>
<ol class="expansion-defined-events" aria-label="Diplomatic context markers">
  {"".join(events)}
</ol>
<details class="expansion-defined-table">
  <summary>Text alternative · exact phase shares and paragraph counts</summary>
  <div class="table-scroll"><table><thead><tr><th scope="col">Thread</th>
    {table_headers}</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>
</details>
<p class="era-context-note">{html.escape(chart["measure_note"])}</p>"""


def _combined_trajectory_html(
    chart: dict,
    axis: list[dict],
) -> str:
    """Overlay four focal-era topic families on one shared line graph."""
    plot_left = 54.0
    plot_right = 720.0
    plot_top = 22.0
    plot_bottom = 238.0
    label_x = 748.0
    shared_scale = float(chart["scale_max"])
    scale_step = float(chart.get("scale_step", 5))
    focal_index = int(chart.get("focal_index", 0))
    x_positions = np.linspace(plot_left, plot_right, len(axis))

    def y_position(share: float) -> float:
        return plot_bottom - (
            share / shared_scale
        ) * (plot_bottom - plot_top)

    grid_values = np.arange(0.0, shared_scale + .001, scale_step)
    gridlines = "".join(
        f'<g><line x1="{plot_left:.1f}" y1="{y_position(value):.2f}" '
        f'x2="{plot_right:.1f}" y2="{y_position(value):.2f}"></line>'
        f'<text x="{plot_left - 9:.1f}" y="{y_position(value) + 3.5:.2f}" '
        f'text-anchor="end">{value:.0f}%</text></g>'
        for value in reversed(grid_values)
    )
    axis_labels = "".join(
        f'<text x="{x_position:.2f}" y="{plot_bottom + 23:.1f}" '
        f'transform="rotate(-34 {x_position:.2f} {plot_bottom + 23:.1f})" '
        f'text-anchor="end"><tspan>{html.escape(value["short"])}</tspan>'
        f'<tspan x="{x_position:.2f}" dy="11">'
        f'{html.escape(value["years"])}</tspan></text>'
        for x_position, value in zip(x_positions, axis)
    )

    endpoint_rows = sorted(
        (
            y_position(row["era_values"][-1]["share"]),
            row,
        )
        for row in chart["rows"]
    )
    endpoint_y = {}
    next_label_y = plot_top
    label_gap = 17.0
    for raw_y, row in endpoint_rows:
        placed_y = max(raw_y, next_label_y)
        endpoint_y[row["key"]] = placed_y
        next_label_y = placed_y + label_gap
    label_overflow = max(
        0.0,
        max(endpoint_y.values(), default=plot_bottom) - plot_bottom,
    )
    if label_overflow:
        endpoint_y = {
            key: value - label_overflow
            for key, value in endpoint_y.items()
        }

    founding_short_labels = {
        "native": "Native power",
        "abroad": "Abroad",
        "union": "Union",
        "finance": "Finance",
    }
    series = []
    legend_items = []
    chart_summary_parts = []
    for row in chart["rows"]:
        points = []
        nodes = []
        for index, (x_position, value) in enumerate(
            zip(x_positions, row["era_values"])
        ):
            point_y = y_position(value["share"])
            points.append(f"{x_position:.2f},{point_y:.2f}")
            value_label_y = (
                point_y + 15.0
                if point_y < plot_top + 16.0
                else point_y - 9.0
            )
            detail = (
                f"{row['label']} · {value['era_label']}: "
                f"{value['share']:.1f}% ({value['count']:,} of "
                f"{value['denominator']:,} paragraphs)"
            )
            nodes.append(
                f'<g class="era-defined-combined-node '
                f'{"is-focal" if index == focal_index else ""}" '
                f'tabindex="0" role="img" '
                f'aria-label="{html.escape(detail, quote=True)}">'
                f'<circle cx="{x_position:.2f}" cy="{point_y:.2f}" '
                f'r="{5.2 if index == focal_index else 3.8}"></circle>'
                f'<text class="era-defined-point-value" '
                f'x="{x_position:.2f}" y="{value_label_y:.2f}" '
                f'text-anchor="middle">{value["share"]:.1f}%</text>'
                f"<title>{html.escape(detail)}</title></g>"
            )
        last_value = row["era_values"][-1]
        last_point_y = y_position(last_value["share"])
        placed_label_y = endpoint_y[row["key"]]
        line_label = row.get(
            "short_label",
            founding_short_labels.get(row["key"], row["label"]),
        )
        line_label = textwrap.shorten(
            line_label,
            width=28,
            placeholder="…",
        )
        series.append(
            f'<g class="era-defined-combined-series" '
            f'style="--defined-color:{row["color"]}">'
            f'<polyline class="era-defined-combined-hit" '
            f'points="{" ".join(points)}" aria-hidden="true"></polyline>'
            f'<polyline class="era-defined-combined-line" '
            f'points="{" ".join(points)}"></polyline>'
            f'{"".join(nodes)}'
            f'<path class="era-defined-end-connector" '
            f'd="M {plot_right + 5:.1f} {last_point_y:.2f} '
            f'L {label_x - 6:.1f} {placed_label_y:.2f}"></path>'
            f'<text class="era-defined-end-label" x="{label_x:.1f}" '
            f'y="{placed_label_y + 3.5:.2f}">'
            f'{html.escape(line_label)}</text></g>'
        )
        legend_items.append(
            f'<span style="--defined-color:{row["color"]}">'
            '<i aria-hidden="true"></i>'
            f'<strong>{html.escape(row["label"])}</strong></span>'
        )
        chart_summary_parts.append(
            f"{row['label']} begins at {row['aggregate_share']:.1f}% "
            f"and ends at {last_value['share']:.1f}%"
        )
    chart_description = (
        "Four independent, overlapping topic-family paragraph shares across "
        f"nine eras on a shared zero to {chart['scale_max']} percent scale. "
        + "; ".join(chart_summary_parts)
        + "."
    )
    focal_band_width = (x_positions[1] - x_positions[0]) * .58
    coverage = chart["coverage"]
    focal_share = coverage.get("focal_share", coverage.get("founding_share"))
    focal_summed_share = coverage.get(
        "focal_summed_share",
        coverage.get("founding_summed_share"),
    )
    return f"""<p class="era-defined-coverage-line">
  <strong>{focal_share:.1f}%</strong> of
  {html.escape(chart.get("focal_short", "Founding"))} paragraphs
  included at least one family; the overlapping family shares sum to
  <strong>{focal_summed_share:.1f}%</strong>.
</p>
<div class="era-defined-combined-key" aria-label="Line key">
  {"".join(legend_items)}
</div>
<p class="era-defined-combined-scale">
  Shared scale: 0–{chart['scale_max']}%. The largest observed value is
  {chart['observed_max_share']:.1f}%; the ceiling is the next five-point mark.
</p>
<div class="era-defined-combined-scroll">
  <svg class="era-defined-combined-chart" viewBox="0 0 900 330"
    role="img" aria-label="{html.escape(chart_description, quote=True)}">
    <title>Four topic families defining
      {html.escape(chart.get("focal_label", "the focal era"))} across nine eras</title>
    <desc>{html.escape(chart_description)}</desc>
    <rect class="era-defined-focal-band"
      x="{x_positions[focal_index] - focal_band_width / 2:.2f}" y="{plot_top:.1f}"
      width="{focal_band_width:.2f}"
      height="{plot_bottom - plot_top:.1f}"></rect>
    <g class="era-defined-combined-grid">{gridlines}</g>
    <line class="era-defined-combined-y-axis"
      x1="{plot_left:.1f}" y1="{plot_top:.1f}"
      x2="{plot_left:.1f}" y2="{plot_bottom:.1f}"></line>
    {"".join(series)}
    <g class="era-defined-combined-axis">{axis_labels}</g>
  </svg>
</div>
<details class="era-defined-combined-method">
  <summary>How the measure works</summary>
  <p class="era-context-note">{html.escape(chart['measure_note'])}</p>
</details>
"""


def _era_defined_html(contextualization: dict) -> str:
    """Render the era-specific defining argument without a generic fallback."""
    chart = contextualization["era_defined"]
    heading = f"""<header class="era-context-question">
  <span>{html.escape(chart['title'])}</span>
  <h4>{html.escape(chart['headline'])}</h4>
  <p>{html.escape(chart['dek'])}</p>
</header>"""
    if chart["status"] != "authored":
        return heading + f"""<div class="era-defined-pending" role="note">
  <span>Era-specific graph pending</span>
  <p>{html.escape(chart['measure_note'])}</p>
</div>"""
    if chart.get("visual_kind") == "diplomatic_tree":
        return heading + _expansion_era_defined_html(chart)
    if chart.get("visual_kind") == "combined_trajectory":
        return heading + _combined_trajectory_html(
            chart,
            contextualization["era_axis"],
        )

    axis = contextualization["era_axis"]
    x_positions = np.linspace(18, 482, len(axis))
    plot_top = 9.0
    plot_bottom = 63.0
    shared_scale = float(chart["scale_max"])
    grid_values = (0.0, shared_scale / 2, shared_scale)
    gridlines = "".join(
        f'<line x1="18" y1="'
        f'{plot_bottom - (value / shared_scale) * (plot_bottom - plot_top):.2f}" '
        f'x2="482" y2="'
        f'{plot_bottom - (value / shared_scale) * (plot_bottom - plot_top):.2f}">'
        f"</line>"
        for value in grid_values
    )
    rows = []
    for row in chart["rows"]:
        aggregate_position = (
            row["aggregate_share"] / chart["scale_max"] * 100
        )
        historical_position = (
            row["historical_average_share"] / chart["scale_max"] * 100
        )
        difference = row["difference_pp"]
        difference_text = f"{difference:+.1f} pp"
        trajectory_points = []
        trajectory_nodes = []
        for index, (x_position, value) in enumerate(
            zip(x_positions, row["era_values"])
        ):
            y_position = plot_bottom - (
                value["share"] / shared_scale
            ) * (plot_bottom - plot_top)
            trajectory_points.append(
                f"{x_position:.2f},{y_position:.2f}"
            )
            detail = (
                f"{row['label']} · {value['era_label']}: "
                f"{value['share']:.1f}% ({value['count']:,} of "
                f"{value['denominator']:,} paragraphs)"
            )
            trajectory_nodes.append(
                f'<g class="era-defined-trajectory-node '
                f'{"is-founding" if index == 0 else ""}" '
                f'tabindex="0" role="img" '
                f'aria-label="{html.escape(detail, quote=True)}">'
                f'<circle cx="{x_position:.2f}" cy="{y_position:.2f}" '
                f'r="{5 if index == 0 else 3.7}"></circle>'
                f"<title>{html.escape(detail)}</title></g>"
            )
        chart_description = (
            f"{row['label']} appears in {row['aggregate_share']:.1f}% of "
            f"Founding paragraphs, compared with an equal-era average of "
            f"{row['historical_average_share']:.1f}% across the other eight "
            f"eras. The nine-era trajectory is shown on a shared zero to "
            f"{chart['scale_max']} percent scale."
        )
        rows.append(
            f"""<li class="era-defined-trajectory-row"
  style="--defined-color:{row['color']};
  --defined-fill:{aggregate_position:.4f}%;
  --defined-benchmark:{historical_position:.4f}%">
  <div class="era-defined-family">
    <span class="era-defined-icon" aria-hidden="true">{row['icon']}</span>
    <span><strong>{html.escape(row['label'])}</strong>
      <small>{html.escape(row['explanation'])}</small></span>
  </div>
  <div class="era-defined-benchmark"
    role="img" aria-label="{html.escape(chart_description, quote=True)}">
    <div class="era-defined-benchmark-track" aria-hidden="true">
      <i></i><span></span>
    </div>
    <div class="era-defined-benchmark-values">
      <b>Founding {row['aggregate_share']:.1f}%</b>
      <span>Average {row['historical_average_share']:.1f}%</span>
      <strong class="{'is-negative' if difference < 0 else ''}">
        {difference_text}</strong>
    </div>
  </div>
  <div class="era-defined-trajectory">
    <svg viewBox="0 0 500 72" role="img"
      aria-label="{html.escape(chart_description, quote=True)}">
      <title>{html.escape(row['label'])} across all nine eras</title>
      <desc>{html.escape(chart_description)}</desc>
      <g class="era-defined-trajectory-grid">{gridlines}</g>
      <polyline points="{" ".join(trajectory_points)}"></polyline>
      {"".join(trajectory_nodes)}
    </svg>
  </div>
</li>"""
        )
    coverage = chart["coverage"]
    era_axis_labels = "".join(
        f'<span title="{html.escape(value["label"], quote=True)} · '
        f'{html.escape(value["years"], quote=True)}">'
        f'{html.escape(value["short"])}</span>'
        for value in axis
    )
    table_headers = "".join(
        f'<th scope="col">{html.escape(value["short"])}</th>'
        for value in axis
    )
    table_rows = []
    for row in chart["rows"]:
        era_cells = "".join(
            f'<td>{value["share"]:.1f}% '
            f'<small>({value["count"]:,}/{value["denominator"]:,})</small></td>'
            for value in row["era_values"]
        )
        table_rows.append(
            f'<tr><th scope="row">{html.escape(row["label"])}</th>'
            f"{era_cells}<td>{row['historical_average_share']:.1f}%</td>"
            f"<td>{row['difference_pp']:+.1f} pp</td></tr>"
        )
    return heading + f"""<p class="era-defined-coverage-line">
  <strong>{coverage['founding_share']:.1f}%</strong> of Founding paragraphs
  included at least one family; the overlapping family shares sum to
  <strong>{coverage['founding_summed_share']:.1f}%</strong>.
</p>
<div class="era-defined-trajectory-key" aria-label="Chart key">
  <span><i aria-hidden="true"></i>Founding share</span>
  <span><em aria-hidden="true"></em>Equal-era average of the other eras</span>
  <span>All trajectories share a 0–{chart['scale_max']}% scale</span>
</div>
<div class="era-defined-trajectory-head" aria-hidden="true">
  <span>Topic family</span><span>Founding vs. later-era average</span>
  <span>Nine-era trajectory</span>
</div>
<ol class="era-defined-trajectory-lanes">{"".join(rows)}</ol>
<div class="era-defined-trajectory-axis" aria-hidden="true">
  <span></span><span></span><div>{era_axis_labels}</div>
</div>
<details class="era-defined-trajectory-table">
  <summary>Text alternative · exact shares and paragraph counts by era</summary>
  <div class="table-scroll"><table><thead><tr><th scope="col">Topic family</th>
    {table_headers}<th scope="col">Later-era average</th>
    <th scope="col">Difference</th></tr></thead>
    <tbody>{"".join(table_rows)}</tbody></table></div>
</details>
<p class="era-context-note">{html.escape(chart['measure_note'])}</p>"""


def _topic_life_html(contextualization: dict) -> str:
    """Render observed paths plus optional, declared editorial hypotheses."""
    chart = contextualization["topic_life"]
    axis = contextualization["era_axis"]
    focal_index = chart["focal_index"]
    axis_labels = "".join(
        f'<span class="{"is-focal" if index == focal_index else ""}" '
        f'title="{html.escape(row["label"], quote=True)} · '
        f'{html.escape(row["years"], quote=True)}">'
        f'{html.escape(row["short"])}</span>'
        for index, row in enumerate(axis)
    )
    x_positions = np.linspace(24, 676, len(axis))
    shared_scale = float(chart["scale_max"])
    scale_ticks = [
        shared_scale / 2,
        shared_scale,
    ]
    scale_guides = "".join(
        f'<line class="topic-life-gridline" x1="24" '
        f'y1="{59 - (tick / shared_scale * 47):.2f}" x2="676" '
        f'y2="{59 - (tick / shared_scale * 47):.2f}"></line>'
        for tick in scale_ticks
    )
    rows = []
    for row in chart["rows"]:
        interpretation = row["interpretation"]
        annotation = row["annotation"]
        branch_switch_index = row["reframe_switch_index"]
        points = []
        source_xy = []
        circles = []
        successor_points = []
        successor_xy = []
        successor_circles = []
        for index, (x_position, value) in enumerate(
            zip(x_positions, row["values"])
        ):
            y_position = 59 - (
                min(value["share"], shared_scale) / shared_scale * 47
            )
            points.append(f"{x_position:.2f},{y_position:.2f}")
            source_xy.append((x_position, y_position))
            tip = (
                f"{row['source_label']} · {value['era_label']}: "
                f"{value['share']:.1f}% ({value['count']:,} of "
                f"{value['denominator']:,} paragraphs)"
            )
            circles.append(
                f'<circle class="'
                f'{"is-focal " if index == focal_index else ""}'
                f'{"is-muted-branch" if branch_switch_index is not None and index >= branch_switch_index else ""}" '
                f'cx="{x_position:.2f}" cy="{y_position:.2f}" '
                f'r="{5 if index == focal_index else 3.6}" tabindex="0" '
                f'role="img" aria-label="{html.escape(tip, quote=True)}">'
                f'<title>{html.escape(tip)}</title></circle>'
            )
        if row["successor_values"]:
            for index, (x_position, value) in enumerate(
                zip(x_positions, row["successor_values"])
            ):
                y_position = 59 - (
                    min(value["share"], shared_scale) / shared_scale * 47
                )
                successor_points.append(
                    f"{x_position:.2f},{y_position:.2f}"
                )
                successor_xy.append((x_position, y_position))
                tip = (
                    f"Proposed successor · {row['successor_label']} · "
                    f"{value['era_label']}: {value['share']:.1f}% "
                    f"({value['count']:,} of "
                    f"{value['denominator']:,} paragraphs). Includes "
                    f"{', '.join(row['successor_topics'])}."
                )
                successor_circles.append(
                    f'<circle class="topic-life-family-ring '
                    f'{"is-focal " if index == focal_index else ""}'
                    f'{"is-active-successor" if branch_switch_index is not None and index >= branch_switch_index else ""}" '
                    f'cx="{x_position:.2f}" cy="{y_position:.2f}" '
                    f'r="{7 if index == focal_index else 5.9}" tabindex="0" '
                    f'role="img" aria-label="{html.escape(tip, quote=True)}">'
                    f'<title>{html.escape(tip)}</title></circle>'
                )
        focal_x = x_positions[focal_index]
        trend_map = {
            "Fades": ("Fades", "fades"),
            "Declines": ("Declines", "fades"),
            "Persists": ("Persists", "persists"),
            "Persists + returns": ("Returns", "returns"),
            "Returns": ("Returns", "returns"),
            "Peaks later": ("Grows", "grows"),
            "Grows": ("Grows", "grows"),
            "Peaks here": ("Peaks here", "grows"),
            "Changes": ("Changes", "changes"),
        }
        if interpretation:
            trend_label = (
                "Reframes → " + interpretation["successor_display"]
            )
            trend_tone = "reframes"
            status_title = (
                f"{interpretation['rationale']} This interpretation is "
                f"{interpretation['validation_status']}."
            )
        elif (
            annotation
            and annotation["label"] != "Reframes"
            and annotation["validation_status"] != "rejected"
        ):
            trend_label, trend_tone = trend_map.get(
                annotation["label"],
                (annotation["label"], "changes"),
            )
            status_title = (
                f"{annotation['rationale']} This annotation is "
                f"{annotation['validation_status']}."
            )
        else:
            trend_label, trend_tone = trend_map.get(
                row["attention_pattern"],
                (row["attention_pattern"], "changes"),
            )
            status_title = row["pattern_detail"]
        if focal_index == len(axis) - 1:
            level_label = f"{row['focal_share']:.1f}% in this era"
        else:
            level_label = (
                f"{row['focal_share']:.1f}% here → "
                f"{row['present_share']:.1f}% now"
            )
        source_line = (
            f'<polyline points="{" ".join(points)}"></polyline>'
        )
        successor_line = ""
        if successor_points and branch_switch_index is not None:
            source_before = points[:branch_switch_index]
            source_after = points[branch_switch_index - 1:]
            successor_before = successor_points[:branch_switch_index + 1]
            successor_after = successor_points[branch_switch_index:]
            source_line = (
                '<polyline class="topic-life-source-before" '
                f'points="{" ".join(source_before)}"></polyline>'
                '<polyline class="topic-life-source-after" '
                f'points="{" ".join(source_after)}"></polyline>'
            )
            source_x, source_y = source_xy[branch_switch_index - 1]
            successor_x, successor_y = successor_xy[branch_switch_index]
            successor_line = (
                '<polyline class="topic-life-successor-before" '
                f'points="{" ".join(successor_before)}"></polyline>'
                f'<line class="topic-life-branch-switch" '
                f'x1="{source_x:.2f}" y1="{source_y:.2f}" '
                f'x2="{successor_x:.2f}" y2="{successor_y:.2f}"></line>'
                '<polyline class="topic-life-successor-active" '
                f'points="{" ".join(successor_after)}"></polyline>'
            )
        elif successor_points:
            successor_line = (
                '<polyline class="topic-life-successor-line" '
                f'points="{" ".join(successor_points)}"></polyline>'
            )
        aria = (
            f"{row['source_label']}. {row['attention_pattern']}. "
            + "; ".join(
                f"{value['era_label']} {value['share']:.1f}%"
                for value in row["values"]
            )
        )
        rows.append(
            f"""<li class="topic-life-row is-{trend_tone}">
  <div class="topic-life-label"
    title="{html.escape(status_title, quote=True)}">
    <span aria-hidden="true">{row['icon']}</span>
    <strong>{html.escape(row['source_label'])}</strong>
    <span class="topic-life-trend">{html.escape(trend_label)}</span>
    <span class="topic-life-level">{html.escape(level_label)}</span>
  </div>
  <svg viewBox="0 0 700 68" role="img"
    aria-label="{html.escape(aria, quote=True)}">
    <rect class="topic-life-focal-band" x="{focal_x - 22:.2f}" y="2"
      width="44" height="64" rx="7"></rect>
    {scale_guides}
    <line class="topic-life-baseline" x1="24" y1="59" x2="676" y2="59"></line>
    {source_line}
{successor_line}
{"".join(successor_circles)}{"".join(circles)}
  </svg>
</li>"""
        )
    successor_key = ""
    if any(row["successor_values"] for row in chart["rows"]):
        successor_key = """<div class="topic-life-key" aria-label="Point key">
  <span><i aria-hidden="true"></i>Named topic</span>
  <span><b aria-hidden="true"></b>Proposed successor · hover outline</span>
</div>"""
    ai_note = ""
    if any(row["interpretation"] for row in chart["rows"]):
        ai_note = (
            " “Reframes” is AI · unvalidated; "
            "the emphasized branch switches when the proposed successor "
            "first exceeds the named source."
        )
    return f"""<header class="era-context-question">
  <span>{html.escape(chart['title'])}</span>
  <h4>{html.escape(chart['headline'])}</h4>
  <p>{html.escape(chart['dek'])}</p>
</header>
{successor_key}
<div class="topic-life-scroll">
  <div class="topic-life-chart">
    <div class="topic-life-axis" aria-hidden="true">{axis_labels}</div>
    <ol class="topic-life-rows">{"".join(rows)}</ol>
  </div>
</div>
<p class="era-context-note">Every row uses the same 0–{chart['scale_max']:.0f}%
vertical scale · exact paragraph shares on hover or focus.{ai_note}
The display cutoff additionally requires
Spearman ≤ {chart['reframe_rule']['max_spearman']:.2f}, source decline ≥
{chart['reframe_rule']['minimum_source_decline_pp']:.0f} pp, and successor
growth ≥ {chart['reframe_rule']['minimum_successor_growth_pp']:.0f} pp.</p>"""


def _era_echo_anchor_positions(
    count: int,
    *,
    width: float,
    height: float,
) -> list[tuple[float, float]]:
    """Return one chronological, shared gravity field for focal presidents."""
    center_x = width / 2
    center_y = height / 2
    if count <= 0:
        return []
    if count == 1:
        return [(center_x, center_y)]
    if count == 2:
        return [(width * .27, center_y), (width * .73, center_y)]
    if count == 3:
        return [
            (center_x, height * .20),
            (width * .25, height * .76),
            (width * .75, height * .76),
        ]
    if count == 4:
        return [
            (width * .24, height * .25),
            (width * .76, height * .25),
            (width * .76, height * .75),
            (width * .24, height * .75),
        ]
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, count, endpoint=False)
    return [
        (
            center_x + np.cos(angle) * width * .34,
            center_y + np.sin(angle) * height * .34,
        )
        for angle in angles
    ]


def _era_echo_gravity_layout(
    gravity: dict,
    *,
    width: float = 900,
    height: float = 510,
) -> dict:
    """Resolve weighted centroids, then add deterministic collision spacing."""
    raw_anchors = gravity["anchor_nodes"]
    positions = _era_echo_anchor_positions(
        len(raw_anchors),
        width=width,
        height=height,
    )
    max_anchor_count = max(
        (node["reference_paragraphs"] for node in raw_anchors),
        default=1,
    ) or 1
    max_anchor_radius = (
        48 if len(raw_anchors) <= 4
        else 38 if len(raw_anchors) <= 6
        else 29
    )
    anchors = []
    for node, (x_position, y_position) in zip(raw_anchors, positions):
        count = node["reference_paragraphs"]
        radius = (
            max(18.0, np.sqrt(count / max_anchor_count) * max_anchor_radius)
            if count
            else 14.0
        )
        anchors.append({
            **node,
            "x": float(x_position),
            "y": float(y_position),
            "radius": float(radius),
        })
    anchor_by_key = {node["key"]: node for node in anchors}

    satellites = []
    raw_groups = (
        ("president", gravity["president_satellites"], 7.5, 16.0, 6),
        ("topic", gravity["topic_satellites"], 5.5, 14.0, 4),
    )
    center_x = width / 2
    center_y = height / 2
    for kind, raw_nodes, min_radius, max_radius, label_limit in raw_groups:
        max_count = max(
            (node["reference_paragraphs"] for node in raw_nodes),
            default=1,
        ) or 1
        for rank, node in enumerate(raw_nodes):
            radius = min_radius + np.sqrt(
                node["reference_paragraphs"] / max_count
            ) * (max_radius - min_radius)
            weighted = [
                (
                    anchor_by_key[row["anchor"]],
                    row["reference_paragraphs"],
                )
                for row in node["connections"]
                if row["anchor"] in anchor_by_key
                and row["reference_paragraphs"] > 0
            ]
            total_weight = sum(weight for _, weight in weighted)
            if not weighted or total_weight <= 0:
                semantic_x, semantic_y = center_x, center_y
            else:
                semantic_x = sum(
                    anchor["x"] * weight for anchor, weight in weighted
                ) / total_weight
                semantic_y = sum(
                    anchor["y"] * weight for anchor, weight in weighted
                ) / total_weight
            seed = sum(
                (index + 1) * ord(character)
                for index, character in enumerate(f"{kind}:{node['key']}")
            )
            angle = np.deg2rad(seed % 360)
            jitter = 4.0 + seed % 7
            target_x = semantic_x + np.cos(angle) * jitter
            target_y = semantic_y + np.sin(angle) * jitter
            if len(weighted) == 1:
                anchor = weighted[0][0]
                dx = center_x - anchor["x"]
                dy = center_y - anchor["y"]
                distance = np.hypot(dx, dy)
                if distance < .01:
                    dx, dy = np.cos(angle), np.sin(angle)
                    distance = 1.0
                separation = anchor["radius"] + radius + 5
                target_x = anchor["x"] + dx / distance * separation
                target_y = anchor["y"] + dy / distance * separation
            satellites.append({
                **node,
                "kind": kind,
                "rank": rank + 1,
                "show_label": rank < label_limit,
                "radius": float(radius),
                "semantic_x": float(semantic_x),
                "semantic_y": float(semantic_y),
                "target_x": float(target_x),
                "target_y": float(target_y),
                "x": float(target_x),
                "y": float(target_y),
            })

    satellites.sort(
        key=lambda node: (
            -node["radius"],
            0 if node["kind"] == "president" else 1,
            node["label"],
        )
    )

    def keep_in_bounds(node: dict) -> None:
        margin = node["radius"] + 5
        node["x"] = float(min(max(node["x"], margin), width - margin))
        node["y"] = float(min(max(node["y"], margin), height - margin))

    def separate_nodes(*, spring: float) -> None:
        for node in satellites:
            node["x"] += (node["target_x"] - node["x"]) * spring
            node["y"] += (node["target_y"] - node["y"]) * spring
            for anchor in anchors:
                dx = node["x"] - anchor["x"]
                dy = node["y"] - anchor["y"]
                distance = np.hypot(dx, dy)
                minimum = node["radius"] + anchor["radius"] + 4
                if distance >= minimum:
                    continue
                if distance < .01:
                    angle = np.deg2rad(
                        sum(ord(character) for character in node["key"]) % 360
                    )
                    dx, dy = np.cos(angle), np.sin(angle)
                    distance = 1.0
                shift = minimum - distance
                node["x"] += dx / distance * shift
                node["y"] += dy / distance * shift
            keep_in_bounds(node)
        for left_index, left in enumerate(satellites):
            for right in satellites[left_index + 1:]:
                dx = right["x"] - left["x"]
                dy = right["y"] - left["y"]
                distance = np.hypot(dx, dy)
                minimum = left["radius"] + right["radius"] + 2.5
                if distance >= minimum:
                    continue
                if distance < .01:
                    angle = np.deg2rad(
                        (
                            sum(ord(character) for character in left["key"])
                            + sum(ord(character) for character in right["key"])
                        )
                        % 360
                    )
                    dx, dy = np.cos(angle), np.sin(angle)
                    distance = 1.0
                shift = (minimum - distance) / 2
                left["x"] -= dx / distance * shift
                left["y"] -= dy / distance * shift
                right["x"] += dx / distance * shift
                right["y"] += dy / distance * shift
                keep_in_bounds(left)
                keep_in_bounds(right)

    for _ in range(150):
        separate_nodes(spring=.035)
    for _ in range(24):
        separate_nodes(spring=0)
    return {
        "width": width,
        "height": height,
        "anchors": anchors,
        "satellites": satellites,
    }


def _era_echo_view_html(
    contextualization: dict,
    chart: dict,
    view: dict,
) -> str:
    """Render one direction of the exact echo paths as a gravity field."""
    network = view["network"]
    gravity = network["gravity"]
    layout = _era_echo_gravity_layout(gravity)
    prefix = profiles.slug(
        f"{contextualization['key']}-echo-{view['key']}"
    )
    anchors = layout["anchors"]
    satellites = layout["satellites"]
    anchor_by_key = {node["key"]: node for node in anchors}
    anchor_slug = {
        node["key"]: profiles.slug(f"{prefix}-anchor-{node['key']}")
        for node in anchors
    }

    def related_text(rows: list[dict], fallback: str) -> str:
        if not rows:
            return fallback
        return ", ".join(
            f"{row['label']} ({row['reference_paragraphs']:,})"
            for row in rows[:3]
        )

    definitions = []
    paths = []
    associations = []
    planet_nodes = []
    satellite_nodes = []
    satellite_slug = {
        (node["kind"], node["key"]): profiles.slug(
            f"{prefix}-{node['kind']}-{node['key']}"
        )
        for node in satellites
    }
    president_layout = {
        node["key"]: node
        for node in satellites
        if node["kind"] == "president"
    }
    topic_president_label = (
        "Invoking actual speakers"
        if view["key"] == "incoming"
        else "Earlier presidents invoked"
    )
    for anchor in anchors:
        node_slug = anchor_slug[anchor["key"]]
        clip_id = f"{node_slug}-clip"
        definitions.append(
            f'<clipPath id="{clip_id}"><circle '
            f'cx="{anchor["x"]:.2f}" cy="{anchor["y"]:.2f}" '
            f'r="{anchor["radius"] - 2:.2f}"></circle></clipPath>'
        )
        anchor_role = (
            "Invoked president" if view["key"] == "incoming"
            else "Actual speaker"
        )
        detail = (
            f"{anchor_role} {anchor['label']} anchors "
            f"{anchor['reference_paragraphs']:,} connected reference "
            f"paragraphs. Strongest connected presidents: "
            f"{related_text(anchor['top_presidents'], 'none')}. "
            f"Leading topics: {related_text(anchor['top_topics'], 'none')}."
        )
        count_label = (
            f"{anchor['reference_paragraphs']:,} paragraphs"
            if anchor["reference_paragraphs"]
            else "No retained references"
        )
        anchor_class = (
            "era-echo-gravity-anchor"
            + ("" if anchor["is_connected"] else " is-unconnected")
        )
        planet_nodes.append(
            f'<g class="{anchor_class}" data-echo-node '
            f'data-node-kind="anchor" data-node-key="{node_slug}" '
            f'data-anchor-keys="{node_slug}" '
            f'data-echo-detail="{html.escape(detail, quote=True)}" '
            f'tabindex="0" role="img" '
            f'aria-label="{html.escape(detail, quote=True)}">'
            f'<circle class="era-echo-gravity-well" '
            f'cx="{anchor["x"]:.2f}" cy="{anchor["y"]:.2f}" '
            f'r="{anchor["radius"] + 9:.2f}"></circle>'
            f'<image href="portraits/{profiles.slug(anchor["label"])}.png" '
            f'x="{anchor["x"] - anchor["radius"]:.2f}" '
            f'y="{anchor["y"] - anchor["radius"]:.2f}" '
            f'width="{anchor["radius"] * 2:.2f}" '
            f'height="{anchor["radius"] * 2:.2f}" '
            f'clip-path="url(#{clip_id})" '
            f'preserveAspectRatio="xMidYMid slice"></image>'
            f'<circle class="era-echo-gravity-planet" '
            f'cx="{anchor["x"]:.2f}" cy="{anchor["y"]:.2f}" '
            f'r="{anchor["radius"]:.2f}"><title>'
            f'{html.escape(detail)}</title></circle>'
            f'<text class="era-echo-gravity-anchor-label" '
            f'x="{anchor["x"]:.2f}" '
            f'y="{anchor["y"] + anchor["radius"] + 15:.2f}" '
            f'text-anchor="middle">{html.escape(anchor["label"])}'
            f'<tspan x="{anchor["x"]:.2f}" dy="10">'
            f'{html.escape(count_label)}</tspan></text></g>'
        )

    for satellite_index, node in enumerate(satellites):
        node_slug = satellite_slug[(node["kind"], node["key"])]
        connection_slugs = [
            anchor_slug[row["anchor"]]
            for row in node["connections"]
            if row["anchor"] in anchor_slug
        ]
        president_slugs = (
            [
                satellite_slug[("president", row["label"])]
                for row in node["president_connections"]
                if ("president", row["label"]) in satellite_slug
            ]
            if node["kind"] == "topic"
            else []
        )
        pull = ", ".join(
            f"{row['anchor']} {row['weight_share']:.0f}% "
            f"({row['reference_paragraphs']:,})"
            for row in sorted(
                node["connections"],
                key=lambda row: (
                    -row["reference_paragraphs"],
                    row["anchor"],
                ),
            )
        )
        if node["kind"] == "president":
            related = related_text(node["top_topics"], "no assigned topic")
            actor_role = (
                "Actual speaker" if view["key"] == "incoming"
                else "Invoked president"
            )
            detail = (
                f"{actor_role} {node['label']}: "
                f"{node['reference_paragraphs']:,} "
                f"distinct reference paragraphs in "
                f"{node['speeches']:,} source documents. Gravitational pull: "
                f"{pull}. Leading topics: {related}."
            )
        else:
            related = related_text(
                node["top_presidents"],
                "no connected president",
            )
            detail = (
                f"{node['label']}: {node['reference_paragraphs']:,} "
                f"distinct reference paragraphs. Gravitational pull: "
                f"{pull}. {topic_president_label}: {related}."
            )
        for connection in node["connections"]:
            anchor = anchor_by_key.get(connection["anchor"])
            if anchor is None:
                continue
            dx = anchor["x"] - node["x"]
            dy = anchor["y"] - node["y"]
            distance = np.hypot(dx, dy) or 1.0
            start_x = node["x"] + dx / distance * node["radius"]
            start_y = node["y"] + dy / distance * node["radius"]
            end_x = anchor["x"] - dx / distance * anchor["radius"]
            end_y = anchor["y"] - dy / distance * anchor["radius"]
            path_width = .65 + np.sqrt(
                connection["weight_share"] / 100
            ) * 3.2
            paths.append(
                f'<line class="era-echo-gravity-link is-{node["kind"]}" '
                f'x1="{start_x:.2f}" y1="{start_y:.2f}" '
                f'x2="{end_x:.2f}" y2="{end_y:.2f}" '
                f'style="--echo-width:{path_width:.2f}px'
                + (
                    f';--echo-color:{node["color"]}'
                    if node["kind"] == "topic"
                    else ""
                )
                + f'" data-echo-link data-satellite-key="{node_slug}" '
                f'data-anchor-key="{anchor_slug[anchor["key"]]}"></line>'
            )
        if node["kind"] == "topic":
            for connection in node["president_connections"]:
                president = president_layout.get(connection["label"])
                if president is None:
                    continue
                president_slug = satellite_slug[
                    ("president", president["key"])
                ]
                dx = president["x"] - node["x"]
                dy = president["y"] - node["y"]
                distance = np.hypot(dx, dy) or 1.0
                start_x = node["x"] + dx / distance * node["radius"]
                start_y = node["y"] + dy / distance * node["radius"]
                end_x = (
                    president["x"]
                    - dx / distance * president["radius"]
                )
                end_y = (
                    president["y"]
                    - dy / distance * president["radius"]
                )
                association_width = .65 + np.sqrt(
                    connection["reference_paragraphs"]
                    / max(node["reference_paragraphs"], 1)
                ) * 2.4
                associations.append(
                    f'<line class="era-echo-topic-president-link" '
                    f'x1="{start_x:.2f}" y1="{start_y:.2f}" '
                    f'x2="{end_x:.2f}" y2="{end_y:.2f}" '
                    f'style="--echo-width:{association_width:.2f}px;'
                    f'--echo-color:{node["color"]}" '
                    f'data-topic-president-link '
                    f'data-topic-key="{node_slug}" '
                    f'data-president-key="{president_slug}"></line>'
                )
        node_class = f"era-echo-gravity-satellite is-{node['kind']}"
        common = (
            f'class="{node_class}" data-echo-node '
            f'data-node-kind="{node["kind"]}" data-node-key="{node_slug}" '
            f'data-anchor-keys="{" ".join(connection_slugs)}" '
            f'data-president-keys="{" ".join(president_slugs)}" '
            f'data-echo-detail="{html.escape(detail, quote=True)}" '
            f'tabindex="0" role="img" '
            f'aria-label="{html.escape(detail, quote=True)}"'
        )
        label = ""
        if node["show_label"]:
            short_label = textwrap.shorten(
                node["label"],
                width=23 if node["kind"] == "topic" else 18,
                placeholder="…",
            )
            label = (
                f'<text class="era-echo-gravity-satellite-label" '
                f'x="{node["x"] + node["radius"] + 4:.2f}" '
                f'y="{node["y"] + 2.5:.2f}">'
                f'{html.escape(short_label)}</text>'
            )
        if node["kind"] == "president":
            clip_id = f"{node_slug}-clip-{satellite_index}"
            definitions.append(
                f'<clipPath id="{clip_id}"><circle '
                f'cx="{node["x"]:.2f}" cy="{node["y"]:.2f}" '
                f'r="{node["radius"] - 1:.2f}"></circle></clipPath>'
            )
            shape = (
                f'<image href="portraits/{profiles.slug(node["label"])}.png" '
                f'x="{node["x"] - node["radius"]:.2f}" '
                f'y="{node["y"] - node["radius"]:.2f}" '
                f'width="{node["radius"] * 2:.2f}" '
                f'height="{node["radius"] * 2:.2f}" '
                f'clip-path="url(#{clip_id})" '
                f'preserveAspectRatio="xMidYMid slice"></image>'
                f'<circle cx="{node["x"]:.2f}" cy="{node["y"]:.2f}" '
                f'r="{node["radius"]:.2f}"><title>'
                f'{html.escape(detail)}</title></circle>'
            )
        else:
            shape = (
                f'<circle cx="{node["x"]:.2f}" cy="{node["y"]:.2f}" '
                f'r="{node["radius"]:.2f}" '
                f'style="--echo-color:{node["color"]}"><title>'
                f'{html.escape(detail)}</title></circle>'
                f'<text class="era-echo-gravity-topic-icon" '
                f'x="{node["x"]:.2f}" y="{node["y"] + 3.5:.2f}" '
                f'text-anchor="middle">{html.escape(node["icon"])}</text>'
            )
        satellite_nodes.append(f"<g {common}>{shape}{label}</g>")

    summary = (
        f"{network['reference_paragraphs']:,} reference paragraphs · "
        f"{len(gravity['president_satellites'])} connected presidents · "
        f"{len(gravity['topic_satellites'])} paragraph topics"
    )
    graph_html = (
        f'<div class="era-echo-gravity-field" data-echo-gravity>'
        f'<p class="era-echo-gravity-readout" data-echo-readout '
        f'data-default-text="Hover or focus a node to inspect its pull.">'
        f'Hover or focus a node to inspect its pull.</p>'
        f'<svg class="era-echo-gravity-chart" '
        f'viewBox="0 0 {layout["width"]:.0f} {layout["height"]:.0f}" '
        f'role="img" '
        f'aria-label="{html.escape(view["description"], quote=True)}">'
        f'<defs>{"".join(definitions)}</defs>'
        f'<circle class="era-echo-gravity-center" '
        f'cx="{layout["width"] / 2:.2f}" '
        f'cy="{layout["height"] / 2:.2f}" r="5"></circle>'
        f'<text class="era-echo-gravity-center-label" '
        f'x="{layout["width"] / 2 + 9:.2f}" '
        f'y="{layout["height"] / 2 + 3:.2f}">shared pull</text>'
        f'{"".join(paths)}{"".join(associations)}{"".join(planet_nodes)}'
        f'{"".join(satellite_nodes)}</svg></div>'
        if anchors
        else (
            '<p class="era-echo-empty">No cross-era named-presidential '
            'references are available for this focal era.</p>'
        )
    )
    president_key = (
        "President looking back"
        if view["key"] == "incoming"
        else "Earlier president invoked"
    )
    return f"""<div class="era-echo-summary">{html.escape(summary)}</div>
<div class="era-echo-gravity-key" aria-label="Graph key">
  <span><i class="is-anchor"></i>{html.escape(contextualization["label"])} president</span>
  <span><i class="is-president"></i>{html.escape(president_key)}</span>
  <span><i class="is-topic"></i>Paragraph topic</span>
  <span>Distance = share of connected paragraphs</span>
</div>
<div class="era-echo-scroll">{graph_html}</div>"""


def _era_echoes_html(contextualization: dict) -> str:
    """Render directional echo views with a direct invoked-by/invoking switch."""
    chart = contextualization["era_echoes"]
    views = chart["directional_views"]
    default_direction = chart["default_direction"]
    controls = []
    panels = []
    for key in ("incoming", "outgoing"):
        view = views[key]
        selected = key == default_direction
        controls.append(
            f'<button type="button" data-echo-direction="{key}" '
            f'aria-pressed="{str(selected).lower()}" '
            f'{"disabled " if not view["available"] else ""}'
            f'aria-controls="{profiles.slug(contextualization["key"])}-'
            f'echo-view-{key}">{html.escape(view["label"])}'
            f'<small>{html.escape(view["description"])}</small></button>'
        )
        panels.append(
            f'<section id="{profiles.slug(contextualization["key"])}-'
            f'echo-view-{key}" data-echo-view="{key}" '
            f'{"hidden" if not selected else ""}>'
            + (
                _era_echo_view_html(contextualization, chart, view)
                if view["available"]
                else (
                    '<p class="era-echo-empty">No retained named-presidential '
                    'references are available in this direction.</p>'
                )
            )
            + "</section>"
        )
    scope_detail = " ".join((chart["scope"], chart["measure_note"]))
    return f"""<header class="era-context-question">
  <span>{html.escape(chart['title'])}</span>
  <h4>{html.escape(chart['headline'])}</h4>
  <p>{html.escape(chart['dek'])}</p>
</header>
<div class="era-echo-direction-switch" data-echo-direction-switch
  role="group" aria-label="Reference direction">
  {"".join(controls)}
</div>
<p class="era-echo-direction-status" data-echo-direction-status>
  {html.escape(views[default_direction]["description"])}</p>
<div class="era-echo-direction-stage">{"".join(panels)}</div>
<p class="era-context-note"><strong>Scope:</strong> One field places every
connected president and assigned topic around the focal era’s presidential
anchors. {html.escape(scope_detail)}</p>"""


def _era_contextualize_group_html(contextualization: dict) -> str:
    """Render the shared two-screen Contextualize template for one era."""
    prefix = profiles.slug(contextualization["key"])
    defined_id = f"{prefix}-screen-defined"
    echoes_id = f"{prefix}-screen-echoes"
    support = contextualization["support"]
    years = contextualization["years"]
    defined = contextualization["era_defined"]
    defined_evidence = ""
    if defined["status"] == "authored":
        if defined.get("visual_kind") == "diplomatic_tree":
            defined_measure = (
                "Independent paragraph shares for four declared topic unions "
                "on one shared axis across four historically defined phases"
            )
            defined_support = (
                f"All {support['paragraphs']:,} corpus paragraphs assigned "
                f"to the governed {years} story era; phase denominators "
                "include topic-free paragraphs."
            )
            defined_caveat = (
                "Topic unions overlap and the step lines are not stacked. "
                "Treaty markers "
                "orient the chronology without claiming that an event caused "
                "a measured change; attention is not policy importance or "
                "consequence."
            )
        else:
            defined_measure = (
                "Era aggregate paragraph share versus the equal-era mean "
                "of the other eight eras for four declared topic unions"
            )
            defined_support = (
                f"All {support['paragraphs']:,} corpus paragraphs assigned "
                f"to the governed {years} story era; each of the other eight "
                "eras receives equal weight in the historical average."
            )
            defined_caveat = (
                "Multi-label paragraphs can enter more than one lane. Textual "
                "attention does not measure actual danger, institutional "
                "success, policy consequences, or causation."
            )
        defined_evidence = _chart_evidence(
            "paragraph_share",
            (
                "paragraphs.parquet + "
                "llm_annotations/paragraph_annotations.parquet"
            ),
            measure_label=defined_measure,
            support=defined_support,
            caveat=defined_caveat,
        )
    return f"""<div class="founding-screen-group founding-contextualize-group"
  data-founding-screen-group
  data-era-contextualize="{html.escape(contextualization["key"], quote=True)}">
  <nav class="founding-screen-tabs founding-contextualize-tabs" role="tablist"
    aria-label="{html.escape(contextualization["label"], quote=True)} contextualization screens">
    <button type="button" role="tab" id="{prefix}-screen-tab-defined"
      aria-controls="{defined_id}" aria-selected="true"
      data-founding-screen-tab="{defined_id}">Era Defined</button>
    <button type="button" role="tab" id="{prefix}-screen-tab-echoes"
      aria-controls="{echoes_id}" aria-selected="false"
      data-founding-screen-tab="{echoes_id}"
      tabindex="-1">Era Echoes</button>
  </nav>
  <div class="founding-screen-stage">
    <section id="{defined_id}" role="tabpanel"
      aria-labelledby="{prefix}-screen-tab-defined" data-founding-screen
      data-era-context-screen="defined">
{_era_defined_html(contextualization)}
{defined_evidence}
    </section>
    <section id="{echoes_id}" role="tabpanel"
      aria-labelledby="{prefix}-screen-tab-echoes"
      data-founding-screen hidden data-era-context-screen="echoes">
      {_era_echoes_html(contextualization)}
      {_chart_evidence(
          "network_edge",
          "networks/invocation_evidence.parquet",
          measure_label=(
              "President speaking → assigned paragraph topic → named former "
              "president paths, separated by reference direction"
          ),
          support=(
              f"{support['invocation_rows']:,} retained invocation-evidence "
              "rows across the corpus; topic shares use distinct reference "
              "paragraphs within each direction."
          ),
          caveat=(
              "The deterministic name registry does not capture unnamed or "
              "collective historical references. Invocation classifications "
              "are exploratory and not human-validated."
          ),
      )}
    </section>
  </div>
</div>"""


def _era_contextualize_html(
    contextualization: dict,
    panel_id: str,
) -> str:
    """Render a standalone reusable Contextualize workspace for a later era."""
    return f"""<section class="era-visualize-standalone era-contextualize-standalone"
  id="{panel_id}">
  <header class="era-visualize-heading">
    <div>
      <span>Contextualize the era</span>
      <h3>{html.escape(contextualization["title"])}</h3>
    </div>
    <p>{html.escape(contextualization["years"])} · defining claim and named
      presidential references</p>
  </header>
  {_era_contextualize_group_html(contextualization)}
</section>"""


def _era_visualize_group_html(visualization: dict) -> str:
    """Render the shared two-screen visualization template for one era."""
    prefix = profiles.slug(visualization["key"])
    agenda_id = f"{prefix}-screen-agenda"
    adversaries_id = f"{prefix}-screen-adversaries"
    support = visualization["support"]
    years = visualization["years"]
    return f"""<div class="founding-screen-group founding-visualize-group"
  data-founding-screen-group data-era-visualize="{html.escape(visualization["key"], quote=True)}">
  <nav class="founding-screen-tabs founding-visualize-tabs" role="tablist"
    aria-label="{html.escape(visualization["label"], quote=True)} visualization screens">
    <button type="button" role="tab" id="{prefix}-screen-tab-agenda"
      aria-controls="{agenda_id}" aria-selected="true"
      data-founding-screen-tab="{agenda_id}">Presidential agendas</button>
    <button type="button" role="tab" id="{prefix}-screen-tab-adversaries"
      aria-controls="{adversaries_id}" aria-selected="false"
      data-founding-screen-tab="{adversaries_id}"
      tabindex="-1">Adversary network</button>
  </nav>
  <div class="founding-screen-stage">
    <section id="{agenda_id}" role="tabpanel"
      aria-labelledby="{prefix}-screen-tab-agenda" data-founding-screen
      data-era-screen="agenda">
      {_era_presidential_agendas_html(visualization)}
      {_chart_evidence(
          "paragraph_share",
          "speaker_views/paragraph_view_v1.parquet + llm_annotations/paragraph_annotations.parquet",
          measure_label="Actual-speaker assigned topic paragraph shares",
          support=(
              f"All {support['speaker_audited_paragraphs']:,} eligible "
              f"speaker-audited paragraphs assigned to the governed {years} "
              "Story era."
          ),
          caveat=(
              f"{visualization['agenda']['selection']} Multi-label shares are "
              "non-additive and do not measure importance, outcomes, or causes."
          ),
      )}
    </section>
    <section id="{adversaries_id}" role="tabpanel"
      aria-labelledby="{prefix}-screen-tab-adversaries"
      data-founding-screen hidden data-era-screen="adversaries">
      {_era_adversary_network_html(visualization)}
      {_chart_evidence(
          "network_edge",
          "reference_entities/entity_mentions_v1.parquet",
          measure_label="Actual-speaker paragraphs with adversarial named entities",
          support=(
              f"AI-backed adversarial names in the "
              f"{support['speaker_audited_paragraphs']:,} eligible "
              f"speaker-audited paragraphs assigned to the governed {years} Story era; "
              f"{visualization['adversaries']['selection'].lower()}"
          ),
          caveat=(
              "Entity stance is a primary-AI classification. AI + NER and AI "
              "only report source agreement, not historical validation or "
              "model confidence."
          ),
      )}
    </section>
  </div>
</div>"""


def _era_visualize_html(
    visualization: dict,
    panel_id: str,
) -> str:
    """Render a standalone reusable visualization workspace for a later era."""
    return f"""<section class="era-visualize-standalone" id="{panel_id}">
  <header class="era-visualize-heading">
    <div>
      <span>Visualize the era</span>
      <h3>{html.escape(visualization["title"])}</h3>
    </div>
    <p>{html.escape(visualization["years"])} · same measures, scales, and
      interactions as every era</p>
  </header>
  {_era_visualize_group_html(visualization)}
</section>"""


def _era_workspace_html(
    profile: dict,
    visualization: dict,
    contextualization: dict,
) -> str:
    """Render the same four direct views for every chronological era."""
    keys = {
        profile["key"],
        visualization["key"],
        contextualization["key"],
    }
    if len(keys) != 1:
        raise ValueError(
            "era workspace inputs must use one shared era key: "
            f"{sorted(keys)}"
        )
    prefix = profiles.slug(profile["key"])
    workspace_id = f"{prefix}-workspace"
    profile_id = f"{prefix}-profile"
    defined_id = f"{prefix}-screen-defined"
    adversaries_id = f"{prefix}-screen-adversaries"
    echoes_id = f"{prefix}-screen-echoes"
    profile_tab_id = f"{prefix}-tab-profile"
    defined_tab_id = f"{prefix}-tab-defined"
    adversaries_tab_id = f"{prefix}-tab-adversaries"
    echoes_tab_id = f"{prefix}-tab-echoes"
    label = html.escape(profile["label"], quote=True)
    support = contextualization["support"]
    years = contextualization["years"]
    return f"""<div class="founding-workspace era-workspace"
  id="{workspace_id}" data-era-workspace="{prefix}">
<nav class="founding-movement-route" role="tablist"
  aria-label="{label} era views">
  <button type="button" role="tab" id="{profile_tab_id}"
    aria-controls="{profile_id}" aria-selected="true"
    data-era-workspace-tab="{profile_id}" data-founding-tab="{profile_id}">
    <strong>Era profile</strong>
    <small>Start here · people, scale, topics</small>
  </button>
  <button type="button" role="tab" id="{defined_tab_id}"
    aria-controls="{defined_id}" aria-selected="false"
    data-era-workspace-tab="{defined_id}"
    data-founding-tab="{defined_id}" tabindex="-1">
    <strong>Era defined</strong>
    <small>Four topics across nine eras</small>
  </button>
  <button type="button" role="tab" id="{adversaries_tab_id}"
    aria-controls="{adversaries_id}" aria-selected="false"
    data-era-workspace-tab="{adversaries_id}"
    data-founding-tab="{adversaries_id}" tabindex="-1">
    <strong>Adversaries</strong>
    <small>Presidents and named opponents</small>
  </button>
  <button type="button" role="tab" id="{echoes_tab_id}"
    aria-controls="{echoes_id}" aria-selected="false"
    data-era-workspace-tab="{echoes_id}"
    data-founding-tab="{echoes_id}" tabindex="-1">
    <strong>Era echoes</strong>
    <small>Invoked by · invoking</small>
  </button>
</nav>
<div class="founding-panel-stage">
  {_era_profile_html(
      profile,
      panel_id=profile_id,
      tab_id=profile_tab_id,
  )}
  <section class="founding-story-card founding-panel" id="{defined_id}"
    role="tabpanel" aria-labelledby="{defined_tab_id}"
    data-era-panel="defined" data-era-context-screen="defined" hidden>
    {_era_defined_html(contextualization)}
    {_chart_evidence(
        "paragraph_share",
        (
            "paragraphs.parquet + "
            "llm_annotations/paragraph_annotations.parquet"
        ),
        measure_label=(
            "Four focal-era topic families traced as paragraph shares "
            "across the same nine-era axis"
        ),
        support=(
            f"All {support['paragraphs']:,} corpus paragraphs assigned to "
            f"the governed {years} story era; each point keeps every "
            "paragraph in its era denominator."
        ),
        caveat=(
            "Multi-label paragraphs can enter more than one line. Textual "
            "attention does not measure importance, policy consequences, "
            "public opinion, or causation."
        ),
    )}
  </section>
  <section class="founding-story-card founding-panel" id="{adversaries_id}"
    role="tabpanel" aria-labelledby="{adversaries_tab_id}"
    data-era-panel="adversaries" data-era-screen="adversaries" hidden>
    {_era_adversary_network_html(visualization)}
    {_chart_evidence(
        "network_edge",
        "reference_entities/entity_mentions_v1.parquet",
        measure_label="Actual-speaker paragraphs with adversarial named entities",
        support=(
            f"AI-backed adversarial names in the "
            f"{support['speaker_audited_paragraphs']:,} eligible "
            f"speaker-audited paragraphs assigned to the governed {years} Story era; "
            f"{visualization['adversaries']['selection'].lower()}"
        ),
        caveat=(
            "Entity stance is a primary-AI classification. AI + NER and AI "
            "only report source agreement, not historical validation or "
            "model confidence."
        ),
    )}
  </section>
  <section class="founding-story-card founding-panel" id="{echoes_id}"
    role="tabpanel" aria-labelledby="{echoes_tab_id}"
    data-era-panel="echoes" data-era-context-screen="echoes" hidden>
    {_era_echoes_html(contextualization)}
    {_chart_evidence(
        "network_edge",
        "networks/invocation_evidence.parquet",
        measure_label=(
            "President speaking → assigned paragraph topic → named former "
            "president paths, switchable by reference direction"
        ),
        support=(
            f"{support['invocation_rows']:,} retained invocation-evidence "
            "rows across the corpus; every view uses distinct reference "
            "paragraphs in its selected direction."
        ),
        caveat=(
            "The deterministic name registry does not capture unnamed or "
            "collective historical references. Invocation classifications "
            "are exploratory and not human-validated."
        ),
    )}
  </section>
</div>
</div>"""


def founding_story_html(data: dict) -> str:
    return f"""<div class="founding-chapter">
{_era_workspace_html(
    data["era_profile"],
    data["era_visualization"],
    data["era_contextualization"],
)}
<details class="founding-evidence-boundary">
  <summary>Evidence boundary · descriptive corpus measurements</summary>
  <p>Sources: <code>data/paragraphs.parquet</code>,
  <code>data/speeches.parquet</code>,
  <code>data/llm_annotations/paragraph_annotations.parquet</code>,
  <code>data/llm_annotations/speech_annotations.parquet</code>,
  <code>data/llm_annotations/agreement_v1.parquet</code>,
  <code>data/llm_annotations/taxonomy_v1.json</code>, and
  <code>data/attention/topic_lifecycles.parquet</code>. Topic labels are
  multi-label AI measurements of textual attention. Historical interpretation
  remains separate from these descriptive counts; no causal effect is estimated.</p>
</details>
</div>"""


def written_republic_takeaway(
    speeches: pd.DataFrame,
    markers: pd.DataFrame,
    speech_annotations: pd.DataFrame,
) -> str:
    """Derive the founding chapter's visible count claim from current artifacts."""
    early = speeches[speeches.year.between(1789, 1808)][
        ["doc_name", "president", "year"]
    ].merge(
        speech_annotations[["doc_name", "medium"]],
        on="doc_name", how="left", validate="one_to_one",
    )
    written = int(early.medium.eq("written_message").sum())
    mechanism = (
        markers[markers.year.between(1789, 1808)]
        .groupby("president")[["mechanism", "n_words"]].sum()
    )
    rates = mechanism.mechanism / mechanism.n_words * 10_000
    return (
        f"The early presidency appears first as paperwork, not performance: "
        f"{written} of {len(early)} corpus speeches are assigned written messages, "
        f"while legal/procedural wording stays above {rates.min():.0f} per 10,000 "
        f"words across all {len(rates)} presidents."
    )


def fig_written_republic(
    speeches: pd.DataFrame,
    markers: pd.DataFrame,
    speech_annotations: pd.DataFrame,
    *,
    full_history: bool = False,
) -> go.Figure:
    """Founding composition by default; an explicit full-history comparison."""
    joined = speeches[["doc_name", "president", "year"]].merge(
        speech_annotations[["doc_name", "medium"]],
        on="doc_name", how="left", validate="one_to_one",
    )
    if full_history:
        joined["decade"] = (joined.year // 10) * 10
        joined["form"] = np.where(
            joined.medium.eq("written_message"),
            "Written message",
            "Performed / spoken assigned form",
        )
        counts = pd.crosstab(joined.decade, joined.form)
        mix = counts.div(counts.sum(axis=1), axis=0) * 100
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[.58, .42],
            subplot_titles=[
                "Full history · one assigned primary form per corpus speech",
                "Legal/procedural wording · centered five-year rate",
            ],
            vertical_spacing=.14,
        )
        for label, color, pattern in [
            ("Written message", "#8b6c42", "/"),
            ("Performed / spoken assigned form", "#477e9d", "."),
        ]:
            fig.add_trace(go.Bar(
                x=mix.index + 5, y=mix[label], name=label,
                marker=dict(color=color, pattern=dict(shape=pattern)),
                customdata=np.stack(
                    [counts[label], counts.sum(axis=1)], axis=-1
                ),
                hovertemplate=(
                    "<b>%{x:.0f}s</b><br>%{y:.1f}%"
                    "<br>%{customdata[0]:.0f} of %{customdata[1]:.0f} speeches"
                    f"<extra>{label}</extra>"
                ),
            ), row=1, col=1)
        mechanism = indices.yearly_rates(markers)["mechanism"]
        fig.add_trace(go.Scatter(
            x=mechanism.index, y=mechanism, mode="lines",
            line=dict(color="#6f4828", width=2.8),
            name="Legal/procedural wording", showlegend=False,
            hovertemplate="%{x}<br>%{y:.1f} matches per 10,000 words<extra></extra>",
        ), row=2, col=1)
        fig.add_vrect(
            x0=1789, x1=1808, fillcolor="rgba(139,108,66,.10)",
            line_width=1, line_color="#8b6c42", row="all", col=1,
        )
        fig.update_layout(**_layout(
            height=590, barmode="stack",
            margin=dict(l=62, r=24, t=78, b=58),
            legend=dict(orientation="h", y=1.08, x=0),
        ))
        fig.update_xaxes(range=[1789, 2026], gridcolor=GRID, linecolor=BASELINE)
        fig.update_yaxes(title_text="% of speeches", range=[0, 100],
                         gridcolor=GRID, linecolor=BASELINE, row=1, col=1)
        fig.update_yaxes(title_text="matches per 10,000 words", rangemode="tozero",
                         gridcolor=GRID, linecolor=BASELINE, row=2, col=1)
        return fig

    early = joined[joined.year.between(1789, 1808)].copy()
    order = (
        early.groupby("president").year.min().sort_values().index.tolist()
    )
    medium_labels = {
        "written_message": "Written message",
        "spoken_address": "Spoken address",
    }
    mix = (
        pd.crosstab(early.president, early.medium, normalize="index")
        .reindex(index=order, columns=list(medium_labels), fill_value=0) * 100
    )
    fig = make_subplots(
        rows=1, cols=2, column_widths=[.62, .38],
        subplot_titles=[
            "Primary form assigned within this corpus",
            "Legal/procedural wording",
        ],
        horizontal_spacing=.13,
    )
    for medium, color, pattern in [
        ("written_message", "#8b6c42", "/"),
        ("spoken_address", "#477e9d", "."),
    ]:
        fig.add_trace(go.Bar(
            x=order, y=mix[medium], name=medium_labels[medium],
            marker=dict(color=color, pattern=dict(shape=pattern)),
            customdata=(
                pd.crosstab(early.president, early.medium)
                .reindex(index=order, columns=[medium], fill_value=0)[medium]
            ),
            hovertemplate=(
                "<b>%{x}</b><br>%{y:.1f}% · %{customdata} speeches"
                f"<extra>{medium_labels[medium]}</extra>"
            ),
        ), row=1, col=1)
    mechanism = (
        markers[markers.year.between(1789, 1808)]
        .groupby("president")[["mechanism", "n_words"]].sum()
        .reindex(order)
    )
    rate = mechanism.mechanism / mechanism.n_words * 10_000
    fig.add_trace(go.Scatter(
        x=order, y=rate, mode="lines+markers+text",
        text=[f"{value:.1f}" for value in rate],
        textposition="top center", name="Legal/procedural", showlegend=False,
        marker=dict(color="#6f4828", size=10, symbol="diamond"),
        line=dict(color="#6f4828", width=2),
        customdata=mechanism.n_words,
        hovertemplate=(
            "<b>%{x}</b><br>%{y:.1f} matches per 10,000 words"
            "<br>%{customdata:,.0f} words<extra></extra>"
        ),
    ), row=1, col=2)
    fig.update_layout(**_layout(
        height=500, barmode="stack",
        margin=dict(l=54, r=24, t=96, b=105),
        legend=dict(
            orientation="h", y=1.18, x=.27, xanchor="center",
            yanchor="bottom",
        ),
    ))
    fig.update_xaxes(tickangle=-25, gridcolor=SURFACE, linecolor=BASELINE)
    fig.update_yaxes(
        title_text="% of speeches", range=[0, 100],
        gridcolor=GRID, linecolor=BASELINE, row=1, col=1,
    )
    fig.update_yaxes(
        title_text="matches per 10,000 words", rangemode="tozero",
        gridcolor=GRID, linecolor=BASELINE, row=1, col=2,
    )
    return fig


EXPANSION_STORY_EVENTS = [
    (1823, "Monroe Doctrine"),
    (1832, "Bank Veto and Nullification Proclamation"),
    (1846, "War with Mexico; territorial acquisition follows, 1846–48"),
]


def _expansion_cell_symbol(row: pd.Series) -> str:
    """Visible confidence cue; color never carries confidence status."""
    symbols = []
    if row["ci_status"] == "low_cluster_caution":
        symbols.append("†")
    elif row["ci_status"] == "suppressed_n_floor":
        symbols.append("‡")
    if bool(row["interval_unresolvable"]):
        symbols.append("◇")
    if row["disagreement_status"] == bands.THIN_PAIRED:
        symbols.append("△")
    return "".join(symbols)


def _expansion_interval_text(row: pd.Series) -> str:
    if bool(row["interval_unresolvable"]):
        return "unresolvable"
    if pd.isna(row["lo"]) or pd.isna(row["hi"]):
        return "suppressed"
    return f"{row['lo'] * 100:.1f}–{row['hi'] * 100:.1f}%"


def _expansion_customdata(rows: pd.DataFrame) -> np.ndarray:
    return np.asarray([
        [
            row["period"],
            row["display_row"],
            f"{row['point'] * 100:.1f}%",
            _expansion_interval_text(row),
            f"{int(row['n_paragraphs']):,}",
            str(int(row["n_speeches"])),
            str(row["ci_status"]),
            " · ".join(json.loads(row["constituent_labels"])),
            str(row["disagreement_status"]),
        ]
        for _, row in rows.iterrows()
    ], dtype=object)


def fig_expansion_story(
    period_metrics: pd.DataFrame,
    meta: dict,
) -> go.Figure:
    """Aligned topic matrix and enemy-naming strip on fixed published scales."""
    period_labels = [
        label
        for label, _, _, is_preview in expansion_story.PERIODS
        if not is_preview
    ]
    display_periods = list(period_labels)
    display_for_period = dict(zip(period_labels, display_periods))
    row_labels = list(expansion_story.TOPIC_ROWS)
    topics = period_metrics[
        period_metrics["row_kind"].eq("topic_union")
        & period_metrics["period"].isin(period_labels)
    ]
    adversary = period_metrics[
        period_metrics["row_kind"].eq("adversary_rate")
        & period_metrics["period"].isin(period_labels)
    ].sort_values("period_order")

    z = np.empty((len(row_labels), len(period_labels)), dtype=float)
    custom = np.empty((len(row_labels), len(period_labels), 9), dtype=object)
    symbols: dict[tuple[str, str], str] = {}
    for row_index, row_label in enumerate(row_labels):
        ordered = topics[topics["display_row"].eq(row_label)].sort_values(
            "period_order"
        )
        if len(ordered) != len(period_labels):
            raise ValueError(
                f"expansion-story row {row_label!r} has {len(ordered)} periods"
            )
        z[row_index] = ordered["point"].to_numpy(dtype=float) * 100
        custom[row_index] = _expansion_customdata(ordered)
        for _, cell in ordered.iterrows():
            symbols[(row_label, cell["period"])] = _expansion_cell_symbol(cell)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.79, 0.21],
    )
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=display_periods,
            y=row_labels,
            zmin=0,
            zmax=60,
            colorscale=[
                [index / (len(BLUE_RAMP) - 1), color]
                for index, color in enumerate(BLUE_RAMP)
            ],
            colorbar=dict(
                title=dict(text="% of<br>paragraphs", font=dict(size=11)),
                thickness=12,
                outlinewidth=0,
                tickvals=[0, 20, 40, 60],
                ticksuffix="%",
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]} · %{customdata[1]}</b>"
                "<br>paragraph share %{customdata[2]}"
                "<br>95% interval %{customdata[3]}"
                "<br>%{customdata[4]} paragraphs · %{customdata[5]} speeches"
                "<br>confidence %{customdata[6]}"
                "<br>constituent labels %{customdata[7]}"
                "<br>annotator disagreement %{customdata[8]}"
                "<extra></extra>"
            ),
            hoverongaps=False,
            showscale=True,
        ),
        row=1,
        col=1,
    )
    # Plotly Heatmap cannot assign a different text color to each cell. Cell
    # annotations do, preserving contrast without repurposing color for status.
    for row_index, row_label in enumerate(row_labels):
        for column_index, period in enumerate(period_labels):
            value = z[row_index, column_index]
            fig.add_annotation(
                x=display_for_period[period],
                y=row_label,
                text=f"{value:.1f}%{symbols[(row_label, period)]}",
                showarrow=False,
                font=dict(
                    size=11,
                    color="white" if value >= 31 else "#26343b",
                ),
                row=1,
                col=1,
            )

    adversary_custom = _expansion_customdata(adversary)
    fig.add_trace(
        go.Bar(
            x=display_periods,
            y=adversary["point"] * 100,
            marker=dict(color="#8b6c42"),
            text=[
                f"{row['point'] * 100:.1f}%{_expansion_cell_symbol(row)}"
                for _, row in adversary.iterrows()
            ],
            textposition="outside",
            textfont=dict(color=INK2, size=10),
            cliponaxis=False,
            customdata=adversary_custom,
            hovertemplate=(
                "<b>%{customdata[0]} · adversary naming</b>"
                "<br>paragraph share %{customdata[2]}"
                "<br>95% interval %{customdata[3]}"
                "<br>%{customdata[4]} paragraphs · %{customdata[5]} speeches"
                "<br>confidence %{customdata[6]}"
                "<br>flag %{customdata[7]}"
                "<br>annotator disagreement %{customdata[8]}"
                "<extra></extra>"
            ),
            name="Enemy naming",
        ),
        row=2,
        col=1,
    )

    entity_details = meta["story_details"]["entity_annotations"]
    domestic = entity_details["1831–35"]
    foreign = entity_details["1846–49"]
    domestic_names = " / ".join(
        item["entity"] for item in domestic["recurring_names"]
    )
    foreign_names = " / ".join(
        item["entity"] for item in foreign["recurring_names"]
    )
    fig.add_annotation(
        x="1831–35",
        y=float(
            adversary.loc[adversary["period"].eq("1831–35"), "point"].iloc[0]
            * 100
        ),
        text=(
            "Domestic institutions dominate · "
            f"{domestic['dominant_type_share'] * 100:.0f}% of adversarial mentions"
            f"<br>{html.escape(domestic_names)}"
        ),
        showarrow=True,
        arrowhead=2,
        ax=-120,
        ay=-54,
        bgcolor="rgba(250,248,243,.94)",
        bordercolor="#c9b89f",
        borderpad=5,
        font=dict(size=9, color=INK2),
        row=2,
        col=1,
    )
    fig.add_annotation(
        x="1846–49",
        y=float(
            adversary.loc[adversary["period"].eq("1846–49"), "point"].iloc[0]
            * 100
        ),
        text=(
            "Foreign nations dominate · "
            f"{foreign['dominant_type_share'] * 100:.0f}% of adversarial mentions"
            f"<br>{html.escape(foreign_names)}"
        ),
        showarrow=True,
        arrowhead=2,
        ax=105,
        ay=-50,
        bgcolor="rgba(250,248,243,.94)",
        bordercolor="#c9b89f",
        borderpad=5,
        font=dict(size=9, color=INK2),
        row=2,
        col=1,
    )

    fig.update_layout(
        **_layout(
            height=790,
            showlegend=False,
            margin=dict(l=215, r=100, t=94, b=92),
            bargap=0.34,
        )
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=display_periods,
        gridcolor=GRID,
        linecolor=BASELINE,
    )
    fig.update_xaxes(
        side="top",
        tickfont=dict(size=10, color=INK2),
        showticklabels=True,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickangle=-22,
        tickfont=dict(size=10, color=INK2),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=row_labels,
        tickfont=dict(size=10.5, color=INK2),
        gridcolor=GRID,
        linecolor=BASELINE,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[0, 40],
        title_text="% of all paragraphs",
        ticksuffix="%",
        dtick=10,
        gridcolor=GRID,
        linecolor=BASELINE,
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=0,
        y=-0.16,
        xref="paper",
        yref="paper",
        xanchor="left",
        text=(
            "† thin sampling support · △ paired-model disagreement unavailable "
            "· ◇ interval unresolvable · ‡ interval suppressed"
        ),
        showarrow=False,
        font=dict(size=9, color=MUTED),
    )
    return fig


def _expansion_cell(
    period_metrics: pd.DataFrame, period: str, display_row: str
) -> pd.Series:
    cell = period_metrics[
        period_metrics["period"].eq(period)
        & period_metrics["display_row"].eq(display_row)
    ]
    if len(cell) != 1:
        raise ValueError(
            f"expected one expansion-story cell for {period} / {display_row}, "
            f"found {len(cell)}"
        )
    return cell.iloc[0]


def expansion_takeaway(period_metrics: pd.DataFrame, meta: dict) -> str:
    banking = _expansion_cell(period_metrics, "1831–35", "Banking")
    domestic_voice = _expansion_cell(
        period_metrics, "1831–35", expansion_story.ADVERSARY_ROW
    )
    expansion = _expansion_cell(
        period_metrics, "1846–49", "Territorial expansion"
    )
    foreign_voice = _expansion_cell(
        period_metrics, "1846–49", expansion_story.ADVERSARY_ROW
    )
    domestic_mix = meta["story_details"]["entity_annotations"]["1831–35"]
    foreign_mix = meta["story_details"]["entity_annotations"]["1846–49"]
    return (
        f"Banking reaches {banking['point'] * 100:.1f}% in 1831–35 as "
        f"enemy naming spikes to {domestic_voice['point'] * 100:.1f}% and "
        f"institutions supply {domestic_mix['dominant_type_share'] * 100:.0f}% "
        "of adversarial mentions. In 1846–49 territorial expansion reaches "
        f"{expansion['point'] * 100:.1f}% and enemy naming spikes to "
        f"{foreign_voice['point'] * 100:.1f}%, with nations supplying "
        f"{foreign_mix['dominant_type_share'] * 100:.0f}% of adversarial mentions. "
        "These are episodic spikes, not a rising trend."
    )


def _expansion_receipt_group(meta: dict, group: str) -> str:
    receipts = [
        receipt
        for receipt in meta["story_details"]["receipts"]
        if receipt["group"] == group
    ]
    if not receipts:
        raise ValueError(f"expansion receipt group {group!r} is empty")
    blocks = []
    for receipt in receipts:
        url = f"https://millercenter.org{receipt['doc_name']}"
        blocks.append(
            '<blockquote><p>“'
            + html.escape(receipt["excerpt"])
            + '”</p><cite><a href="'
            + html.escape(url, quote=True)
            + '" target="_blank" rel="noopener">'
            + html.escape(receipt["title"])
            + "</a> · paragraph "
            + str(receipt["para_idx"])
            + " · "
            + html.escape(receipt["reason"])
            + "</cite></blockquote>"
        )
    return "".join(blocks)


def _expansion_story_html(period_metrics: pd.DataFrame, meta: dict) -> str:
    jackson = meta["story_details"]["jackson_character"]
    bank = jackson["bank_veto"]
    nullification = jackson["nullification_proclamation"]
    removal = jackson["removal_message"]
    figure = (
        '<div class="expansion-story-figure">'
        '<p class="chart-title">1816–49 · Paragraph attention and temporary changes in voice</p>'
        '<p class="chart-subtitle">The chapter begins in 1809; this governed matrix '
        "covers its postwar 1816–49 portion. The aligned strip shows period-specific "
        "adversary naming; bars are not connected because the spikes are episodic.</p>"
        '<div class="expansion-story-scroll"><div class="chart expansion-story-chart" '
        'data-fig="expansion_story" style="height:790px"></div></div>'
        '<p class="expansion-overlap-note"><strong>Overlapping rows.</strong> '
        "Territorial expansion is the de-duplicated union of the two indented "
        "subrows. Parent and subrows overlap and are not additive.</p>"
        + expansion_story.accessible_table_html(
            period_metrics, include_preview=False
        )
        + _chart_evidence(
            "paragraph_share",
            "expansion_story/period_metrics.parquet",
            measure_label="Period paragraph share with speech-clustered interval",
            support=(
                "Every paragraph in each fixed period, including topic-free "
                "paragraphs; intervals resample whole speeches 500 times."
            ),
            caveat=(
                "Topic labels are AI measurements. Paired-model disagreement "
                "is measured on the available subsample and only widens intervals."
            ),
        )
        + "</div>"
    )
    return (
        '<p class="expansion-story-role">The topic matrix carries the continuous '
        "historical story. The adversary strip shows temporary changes in voice.</p>"
        "<h3>War tests the republic</h3>"
        "<p>Madison’s 1809 accession opens the chapter. Renewed conflict with "
        "Britain culminates in the War of 1812, testing whether the constitutional "
        "state could defend sovereignty, finance war, and survive military failure. "
        "The fixed-period matrix begins in 1816, so this opening is historical "
        "framing rather than a relabeled matrix column.</p>"
        "<h3>Securing the postwar republic</h3>"
        "<p>Monroe’s 1817 First Annual Message places diplomacy, public credit, "
        "Native land acquisition, and settlement inside one administrative "
        "inventory. The 1823 Monroe Doctrine is retained as historical context, "
        "not as a cause assigned to a matrix cell.</p>"
        '<div class="expansion-receipt-group" data-receipt-group="monroe">'
        + _expansion_receipt_group(meta, "monroe")
        + "</div>"
        "<h3>Federal policy becomes domestic combat</h3>"
        "<p>Jackson is a significant, temporary, selectively combative tonal "
        f"rupture: enemy naming appears in {bank['share'] * 100:.1f}% of Bank "
        f"Veto paragraphs and {nullification['share'] * 100:.1f}% of the "
        "Nullification Proclamation, but in "
        f"{removal['share'] * 100:.1f}% of the three-paragraph removal message. "
        "The contrast is descriptive; it does not make administrative wording "
        "less coercive.</p>"
        '<div class="expansion-receipt-group" data-receipt-group="jackson">'
        + _expansion_receipt_group(meta, "jackson")
        + "</div>"
        + figure
        + "<h3>Conquest creates a slavery question</h3>"
        "<p>Polk’s 1846 War Message names Mexico and war; the 1848 Message "
        "Regarding Slavery in the Territories supplies the documentary hinge "
        "into sectional crisis. The evidence stops at 1849; the sectional crisis "
        "is reserved for the next chapter.</p>"
        '<div class="expansion-receipt-group" data-receipt-group="polk">'
        + _expansion_receipt_group(meta, "polk")
        + "</div>"
        + _event_callouts(EXPANSION_STORY_EVENTS)
    )


def _fear_ratio(rates: pd.DataFrame) -> pd.Series:
    return (rates["nrc_fear"] / rates["nrc_hope"].replace(0, np.nan)).dropna()


def fig_fear_early(rates: pd.DataFrame) -> go.Figure:
    ratio = _fear_ratio(rates).loc[1789:1868]
    ceiling = float(ratio.max() * 1.18)
    fig = go.Figure(go.Scatter(
        x=ratio.index, y=ratio, mode="lines",
        line=dict(color="#8f4e56", width=3),
        fill="tozeroy", fillcolor="rgba(143,78,86,.16)",
        hovertemplate="%{x}<br>fear ÷ hope %{y:.2f}<extra></extra>",
    ))
    for year, label in [(1812, "War of 1812"), (1861, "Civil War begins")]:
        fig.add_vline(
            x=year, line_color="#735a39", line_dash="dot",
            annotation_text=f"{year} · {label}", annotation_position="top left",
        )
    fig.update_layout(**_layout(
        height=500, showlegend=False,
        title=dict(text="Stage 1 · Early record only; early-period axes", font=dict(size=15)),
        xaxis=dict(range=[1789, 1868], title="1789–1868",
                   gridcolor=GRID, linecolor=BASELINE),
        yaxis=dict(range=[0, ceiling], title="fear words ÷ hope words",
                   gridcolor=GRID, linecolor=BASELINE),
    ))
    return fig


def fig_fear_full(rates: pd.DataFrame) -> go.Figure:
    ratio = _fear_ratio(rates)
    early = ratio.loc[1789:1868]
    early_ceiling = float(early.max() * 1.18)
    fig = make_subplots(
        rows=2, cols=1, row_heights=[.72, .28], vertical_spacing=.14,
        subplot_titles=[
            "Stage 2 · Axis changed: the full 1789–2026 record",
            "Preserved ghost of stage 1 · same early scale as before",
        ],
    )
    fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio, mode="lines",
        line=dict(color="#8f4e56", width=2.7),
        hovertemplate="%{x}<br>fear ÷ hope %{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=early.index, y=early, mode="lines",
        line=dict(color="#8f4e56", width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(143,78,86,.10)",
        hovertemplate="%{x}<br>early-scale fear ÷ hope %{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_vrect(
        x0=1789, x1=1868, fillcolor="rgba(139,108,66,.09)",
        line_width=1, line_color="#8b6c42", row=1, col=1,
    )
    fig.add_annotation(
        x=1833, y=float(ratio.max()) * .92,
        text="the stage-1 window", showarrow=False,
        font=dict(color="#735a39", size=11), row=1, col=1,
    )
    fig.update_layout(**_layout(
        height=670, showlegend=False,
        margin=dict(l=68, r=26, t=68, b=55),
    ))
    fig.update_xaxes(range=[1789, 2026], gridcolor=GRID, linecolor=BASELINE,
                     row=1, col=1)
    fig.update_xaxes(range=[1789, 1868], gridcolor=GRID, linecolor=BASELINE,
                     row=2, col=1)
    fig.update_yaxes(title_text="fear ÷ hope", rangemode="tozero",
                     gridcolor=GRID, linecolor=BASELINE, row=1, col=1)
    fig.update_yaxes(title_text="early scale", range=[0, early_ceiling],
                     gridcolor=GRID, linecolor=BASELINE, row=2, col=1)
    return fig


def fear_index(rates: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    """Fear÷hope indexed to the mean of available founding display windows."""
    raw = _fear_ratio(rates)
    founding = raw.loc[1789:1808].dropna()
    if founding.empty:
        raise ValueError("fear index needs at least one supported founding window")
    baseline = float(founding.mean())
    return raw / baseline * 100, raw, baseline


def fear_index_takeaway(rates: pd.DataFrame) -> str:
    indexed, _raw, _baseline = fear_index(rates)
    crisis = indexed.loc[1850:1868].dropna()
    year = int(crisis.idxmax())
    value = float(crisis.max())
    return (
        f"Crisis language peaks at index {value:.1f} in {year}: "
        f"{value - 100:.1f}% above the mean of the supported 1789–1808 "
        "fear-to-hope display windows."
    )


def fig_fear_index(rates: pd.DataFrame, *, full_history: bool = False) -> go.Figure:
    indexed, raw, baseline = fear_index(rates)
    lo, hi = (1789, 2026) if full_history else (1850, 1868)
    view = indexed.loc[lo:hi]
    raw_view = raw.reindex(view.index)
    fig = go.Figure(go.Scatter(
        x=view.index, y=view, mode="lines",
        line=dict(color="#8f4e56", width=3),
        fill="tozeroy", fillcolor="rgba(143,78,86,.14)",
        customdata=raw_view,
        hovertemplate=(
            "%{x}<br>founding index %{y:.1f}"
            "<br>raw fear ÷ hope %{customdata:.3f}<extra></extra>"
        ),
    ))
    fig.add_hline(
        y=100, line_color="#6f4828", line_width=1.4, line_dash="dash",
        annotation_text=(
            f"1789–1808 mean = 100 · raw ratio {baseline:.3f}"
        ),
        annotation_position="bottom left",
    )
    events = [(1812, "War of 1812"), (1861, "Civil War begins")]
    for index, (year, label) in enumerate(events):
        if not lo <= year <= hi:
            continue
        fig.add_vline(x=year, line_color="#735a39", line_dash="dot", line_width=1)
        fig.add_annotation(
            x=year, y=.97 - index * .08, xref="x", yref="paper",
            text=label, showarrow=False, textangle=-30,
            font=dict(size=10, color="#6f4828"),
        )
    fig.update_layout(**_layout(
        height=500, showlegend=False,
        title=dict(
            text=(
                "Full history · scale expands to every supported window"
                if full_history else
                "Era view · crisis windows relative to the founding mean"
            ),
            font=dict(size=15),
        ),
        xaxis=dict(range=[lo, hi], title=f"{lo}–{hi}",
                   gridcolor=GRID, linecolor=BASELINE),
        yaxis=dict(title="fear÷hope index · founding mean = 100",
                   rangemode="tozero", gridcolor=GRID, linecolor=BASELINE),
    ))
    return fig


CIVIL_RIGHTS_EVENTS = {
    "Civil rights & race": [
        (1863, "Emancipation Proclamation takes effect"),
        (1865, "13th Amendment ratified"),
        (1868, "14th Amendment ratified"),
    ],
}


def _rights_year_support(
    para_labels: pd.DataFrame, paragraphs: pd.DataFrame,
) -> pd.DataFrame:
    joined = para_labels[
        ["doc_name", "para_idx", "year", "Civil rights & race"]
    ].merge(
        paragraphs[["doc_name", "para_idx", "word_count"]],
        on=["doc_name", "para_idx"], validate="one_to_one",
    )
    if len(joined) != len(para_labels) or len(joined) != len(paragraphs):
        raise ValueError("rights view requires identical paragraph key sets")
    annual = joined[joined.year.between(1850, 1868)].groupby("year").agg(
        share=("Civil rights & race", "mean"),
        n_paragraphs=("para_idx", "size"),
        n_speeches=("doc_name", "nunique"),
        n_words=("word_count", "sum"),
    )
    return annual.reindex(range(1850, 1869))


def fig_civil_rights_yearly(
    para_labels: pd.DataFrame, paragraphs: pd.DataFrame,
) -> go.Figure:
    annual = _rights_year_support(para_labels, paragraphs)
    share = annual.share * 100
    support = annual.n_speeches.fillna(0)
    sizes = 6 + np.minimum(support, 10) * .8
    opacity = (.28 + np.minimum(support, 5) / 5 * .72).clip(upper=1)
    custom = np.stack([
        annual.n_speeches.fillna(0),
        annual.n_paragraphs.fillna(0),
        annual.n_words.fillna(0),
    ], axis=-1)
    fig = go.Figure(go.Scatter(
        x=annual.index, y=share, mode="lines+markers",
        connectgaps=False,
        line=dict(color="#6d5a91", width=2.5),
        marker=dict(
            color="#6d5a91", symbol="diamond", size=sizes, opacity=opacity,
            line=dict(color="#3f3156", width=1),
        ),
        customdata=custom,
        hovertemplate=(
            "%{x}<br>%{y:.1f}% of paragraphs"
            "<br>%{customdata[0]:.0f} speeches"
            "<br>%{customdata[1]:,.0f} paragraphs"
            "<br>%{customdata[2]:,.0f} words<extra></extra>"
        ),
    ))
    short = ["Emancipation", "13th", "14th"]
    for index, ((year, _), label) in enumerate(
        zip(CIVIL_RIGHTS_EVENTS["Civil rights & race"], short)
    ):
        fig.add_vline(x=year, line_color="#8b6c42", line_dash="dot", line_width=1)
        fig.add_annotation(
            x=year, y=1.02 + .08 * (index % 2), xref="x", yref="paper",
            text=f"{year} · {label}", showarrow=False, textangle=-25,
            font=dict(size=9.5, color="#6f4828"),
        )
    fig.update_layout(**_layout(
        height=500, showlegend=False,
        margin=dict(l=68, r=24, t=100, b=58),
        xaxis=dict(range=[1849.5, 1868.5], title="annual observations",
                   gridcolor=GRID, linecolor=BASELINE),
        yaxis=dict(title="% of paragraphs carrying the CorEx label",
                   rangemode="tozero", gridcolor=GRID, linecolor=BASELINE),
    ))
    return fig


def fig_civil_rights_full(band_table: pd.DataFrame) -> go.Figure:
    band = band_table[
        band_table.surface.eq("corex_issues")
        & band_table.series.eq("Civil rights & race")
    ].sort_values("period_order")
    if band.empty:
        raise ValueError("full-history rights band is unavailable")
    fig = go.Figure()
    valid = band.lo.notna() & band.hi.notna()
    fig.add_trace(go.Scatter(
        x=pd.concat([band.loc[valid, "x"], band.loc[valid, "x"].iloc[::-1]]),
        y=pd.concat([
            band.loc[valid, "hi"] * 100,
            band.loc[valid, "lo"].iloc[::-1] * 100,
        ]),
        fill="toself", fillcolor="rgba(109,90,145,.14)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=band.x, y=band.point * 100, mode="lines+markers",
        line=dict(color="#6d5a91", width=2.5),
        marker=dict(
            color=np.where(band.ci_status.eq("ok"), "#6d5a91", "#b9a8cc"),
            symbol=np.where(band.ci_status.eq("ok"), "diamond", "diamond-open"),
            size=8,
        ),
        customdata=np.stack([
            band.n_speeches, band.n_paragraphs,
            band.lo.fillna(np.nan) * 100, band.hi.fillna(np.nan) * 100,
        ], axis=-1),
        hovertemplate=(
            "%{x:.0f}–%{x:.0f} bucket<br>%{y:.1f}% of paragraphs"
            "<br>%{customdata[0]:.0f} speeches · %{customdata[1]:,.0f} paragraphs"
            "<br>95% sampling interval %{customdata[2]:.1f}–%{customdata[3]:.1f}%"
            "<extra></extra>"
        ),
    ))
    for index, ((year, _), label) in enumerate(zip(
        CIVIL_RIGHTS_EVENTS["Civil rights & race"],
        ["Emancipation", "13th", "14th"],
    )):
        fig.add_vline(x=year, line_color="#8b6c42", line_dash="dot", line_width=1)
        fig.add_annotation(
            x=year, y=.97 - .08 * (index % 2), xref="x", yref="paper",
            text=label, showarrow=False, textangle=-25,
            font=dict(size=9, color="#6f4828"),
        )
    fig.update_layout(**_layout(
        height=500, showlegend=False,
        title=dict(text="Full history · five-year speech-clustered sampling bands",
                   font=dict(size=15)),
        xaxis=dict(range=[1785, 2026], gridcolor=GRID, linecolor=BASELINE),
        yaxis=dict(title="% of paragraphs carrying the CorEx label",
                   rangemode="tozero", gridcolor=GRID, linecolor=BASELINE),
    ))
    return fig


def fig_civil_rights_crisis(para_labels: pd.DataFrame) -> go.Figure:
    sub = para_labels[para_labels.year.between(1850, 1868)].copy()
    sub["period"] = ((sub.year - 1850) // 2) * 2 + 1850
    share = sub.groupby("period")["Civil rights & race"].mean() * 100
    x = share.index + 1
    fig = go.Figure(go.Scatter(
        x=x, y=share, mode="lines+markers",
        line=dict(color="#6d5a91", width=3),
        marker=dict(color="#6d5a91", size=8, symbol="diamond"),
        fill="tozeroy", fillcolor="rgba(109,90,145,.14)",
        hovertemplate="%{x}<br>%{y:.1f}% of paragraphs<extra></extra>",
    ))
    for number, (year, _label) in enumerate(CIVIL_RIGHTS_EVENTS["Civil rights & race"], 1):
        fig.add_vline(
            x=year, line_color="#8b6c42", line_dash="dot", line_width=1,
            annotation_text=str(number), annotation_position="top",
        )
    fig.update_layout(**_layout(
        height=430, showlegend=False,
        xaxis=dict(range=[1849.5, 1868.5], title="1850–1868 only",
                   gridcolor=GRID, linecolor=BASELINE),
        yaxis=dict(title="% of paragraphs carrying the CorEx label",
                   rangemode="tozero", gridcolor=GRID, linecolor=BASELINE),
    ))
    return fig


def fig_procedural_reveal(scores: pd.DataFrame, stage: int) -> go.Figure:
    """Stable-axis reveal from the administrative-industrial era onward."""
    points = scores.copy()
    points["mid"] = (points.first_year + points.last_year) / 2
    points["cohort"] = np.select(
        [
            points.first_year.between(1869, 1912),
            points.first_year.between(1913, 2016),
            points.last_year.ge(2017),
        ],
        ["1869–1912", "1913–2016", "2017–2026"],
        default="Outside view",
    )
    visible = ["1869–1912"] + (["1913–2016"] if stage >= 2 else []) + (
        ["2017–2026"] if stage >= 3 else []
    )
    fig = go.Figure()
    styles = {
        "1869–1912": ("#8b6c42", "diamond"),
        "1913–2016": ("#7e8790", "circle-open"),
        "2017–2026": ("#9b4e50", "star"),
    }
    for cohort in visible:
        group = points[points.cohort.eq(cohort)].sort_values("mid")
        color, symbol = styles[cohort]
        fig.add_trace(go.Scatter(
            x=group.mechanism, y=group.hype, mode="markers+text",
            text=group.index if cohort != "1913–2016" else None,
            textposition="top center", name=cohort,
            marker=dict(color=color, symbol=symbol,
                        size=12 if cohort != "1913–2016" else 8),
            customdata=np.stack([group.index, group.first_year, group.last_year], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]:.0f}–%{customdata[2]:.0f}"
                "<br>legal/procedural %{x:.1f} per 10k"
                "<br>hype %{y:.1f} per 10k<extra></extra>"
            ),
        ))
    if stage >= 3 and "Donald Trump" in points.index:
        trump = points.loc["Donald Trump"]
        fig.add_annotation(
            x=float(trump.mechanism), y=float(trump.hype),
            text="Trump endpoint", showarrow=True, ax=-55, ay=-40,
            bgcolor="#fff9ef", bordercolor="#9b4e50",
        )
    fig.update_layout(**_layout(
        height=540,
        title=dict(
            text=f"Stage {stage} · axes stay fixed while later presidents enter",
            font=dict(size=15),
        ),
        xaxis=dict(title="legal/procedural vocabulary per 10,000 words",
                   range=[0, float(points.mechanism.max()) * 1.10],
                   gridcolor=GRID, linecolor=BASELINE),
        yaxis=dict(title="hype vocabulary per 10,000 words",
                   range=[0, float(points.hype.max()) * 1.14],
                   gridcolor=GRID, linecolor=BASELINE),
        legend=dict(orientation="h", y=1.06, x=0),
    ))
    return fig


def _president_story_era(row: pd.Series) -> str:
    midpoint = int(np.floor(
        (float(row.first_year) + float(row.last_year)) / 2
    ))
    for name, start, end in era_profiles.STORY_ERAS:
        if start <= midpoint <= end:
            return name
    raise ValueError(f"president midpoint {midpoint} is outside story eras")


def fig_summary_enemy_presidents(
    points: pd.DataFrame, faces: dict[str, str]
) -> go.Figure:
    """President-level enemy naming and party attack with fixed membership."""
    points = points.copy().sort_values(["first_year", "last_year"])
    points["era"] = points.apply(_president_story_era, axis=1)
    x_span = float(points.enemy_naming.max() - points.enemy_naming.min()) or 1
    y_span = float(points.party_attack.max() - points.party_attack.min()) or 1
    fig = go.Figure(go.Scatter(
        x=points.enemy_naming, y=points.party_attack, mode="markers",
        marker=dict(
            size=34, color="#fff8ec", opacity=1,
            line=dict(color="#6f4828", width=3.2),
        ),
        customdata=np.stack([
            points.index, points.first_year, points.last_year, points.era,
            points.n_speeches, points.n_paragraphs, points.zero_sum,
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b> · %{customdata[1]:.0f}–%{customdata[2]:.0f}"
            "<br>%{customdata[3]}"
            "<br>enemy named in %{x:.1f}% of paragraphs"
            "<br>partisan attack in %{y:.1f}%"
            "<br>zero-sum framing in %{customdata[6]:.1f}%"
            "<br>%{customdata[5]:.0f} paragraphs · %{customdata[4]:.0f} speeches"
            "<extra></extra>"
        ),
        showlegend=False,
    ))
    for president, row in points.iterrows():
        if president not in faces:
            continue
        fig.add_layout_image(
            source=faces[president], x=float(row.enemy_naming),
            y=float(row.party_attack), xref="x", yref="y",
            xanchor="center", yanchor="middle",
            sizex=x_span * .055, sizey=y_span * .065,
            sizing="contain", layer="above", opacity=1,
            name=f"portrait::{president}",
        )
    fig.update_layout(**_layout(
        height=620,
        title=dict(
            text="Who names enemies—and who sustains partisan attack?",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="paragraphs naming an enemy", ticksuffix="%",
            range=[0, float(points.enemy_naming.max()) * 1.10],
            gridcolor=GRID, linecolor=BASELINE,
        ),
        yaxis=dict(
            title="paragraphs attacking a party", ticksuffix="%",
            range=[0, float(points.party_attack.max()) * 1.14],
            gridcolor=GRID, linecolor=BASELINE,
        ),
        margin=dict(l=72, r=28, t=70, b=62),
    ))
    return fig


def fig_procedural_eras(scores: pd.DataFrame, faces: dict[str, str]) -> go.Figure:
    """All presidents remain visible while selected named eras receive emphasis."""
    points = scores.copy().sort_values(["first_year", "last_year"])
    points["era"] = points.apply(_president_story_era, axis=1)
    x_span = float(points.mechanism.max() - points.mechanism.min()) or 1
    y_span = float(points.hype.max() - points.hype.min()) or 1
    initial = pd.Series(True, index=points.index)
    fig = go.Figure(go.Scatter(
        x=points.mechanism, y=points.hype, mode="markers",
        marker=dict(
            size=34,
            color=np.where(initial, "#fff8ec", "#eef0f1").tolist(),
            opacity=np.where(initial, 1.0, .22).tolist(),
            line=dict(
                color=np.where(initial, "#6f4828", "#8d9498").tolist(),
                width=np.where(initial, 3.2, .8).tolist(),
            ),
        ),
        customdata=np.stack([
            points.index,
            points.first_year,
            points.last_year,
            points.era,
            points.n_speeches,
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b> · %{customdata[1]:.0f}–%{customdata[2]:.0f}"
            "<br>%{customdata[3]}"
            "<br>legal/procedural %{x:.1f} per 10,000"
            "<br>hype %{y:.1f} per 10,000"
            "<br>%{customdata[4]:.0f} corpus speeches<extra></extra>"
        ),
        showlegend=False,
    ))
    for is_selected, (president, row) in zip(initial, points.iterrows()):
        if president not in faces:
            continue
        fig.add_layout_image(
            source=faces[president],
            x=float(row.mechanism), y=float(row.hype),
            xref="x", yref="y", xanchor="center", yanchor="middle",
            sizex=x_span * .055, sizey=y_span * .065,
            sizing="contain", layer="above",
            opacity=1.0 if is_selected else .18,
            name=f"portrait::{president}",
        )
    fig.update_layout(**_layout(
        height=620,
        title=dict(
            text="Legal/procedural vocabulary × hype across presidential records",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="legal/procedural vocabulary per 10,000 words",
            range=[0, float(points.mechanism.max()) * 1.10],
            gridcolor=GRID, linecolor=BASELINE,
        ),
        yaxis=dict(
            title="hype vocabulary per 10,000 words",
            range=[0, float(points.hype.max()) * 1.14],
            gridcolor=GRID, linecolor=BASELINE,
        ),
        margin=dict(l=72, r=28, t=70, b=62),
    ))
    return fig


def _summary_register_timeline_frame(rates: pd.DataFrame) -> pd.DataFrame:
    """Return supported component rates and their guarded ratio by center year."""
    required = {"mechanism", "hype"}
    missing = required - set(rates.columns)
    if missing:
        raise ValueError(
            "summary register timeline is missing rate columns: "
            + ", ".join(sorted(missing))
        )
    if not rates.index.is_unique or not rates.index.is_monotonic_increasing:
        raise ValueError("summary register timeline years must be unique and ordered")
    frame = rates.loc[:, ["mechanism", "hype"]].copy()
    frame = frame[
        (frame.index >= era_profiles.STORY_ERAS[0][1])
        & (frame.index <= era_profiles.STORY_ERAS[-1][2])
    ]
    frame["legal_hype_ratio"] = frame["mechanism"].div(
        frame["hype"].where(frame["hype"] > 0)
    )
    if frame["legal_hype_ratio"].dropna().empty:
        raise ValueError("summary register timeline has no supported windows")
    return frame


def fig_summary_register_timeline(rates: pd.DataFrame) -> go.Figure:
    """Plot legal/procedural matches per hype match over supported windows."""
    frame = _summary_register_timeline_frame(rates)

    first_year = int(frame.index.min())
    last_year = int(frame.index.max())
    windows = np.asarray([
        f"{max(first_year, int(year) - 2)}–{min(last_year, int(year) + 2)} window"
        for year in frame.index
    ])
    customdata = np.column_stack((
        windows,
        frame["mechanism"].to_numpy(),
        frame["hype"].to_numpy(),
    ))
    fig = go.Figure(go.Scatter(
        x=frame.index,
        y=frame["legal_hype_ratio"],
        mode="lines",
        name="Legal/procedural ÷ hype",
        connectgaps=False,
        line=dict(color="#315f78", width=3),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b>"
            "<br>%{y:.2f} legal/procedural matches per hype match"
            "<br>Legal/procedural: %{customdata[1]:.2f} per 10,000"
            "<br>Hype: %{customdata[2]:.2f} per 10,000"
            "<extra></extra>"
        ),
    ))
    era_annotation_labels = {
        "Admin-industrial": "Admin-<br>industrial",
        "Reform/collapse": "Reform/<br>collapse",
        "New Deal/WWII": "New Deal/<br>WWII",
        "Always-on": "Always-<br>on",
    }
    for index, ((_, start, end), short) in enumerate(zip(
        era_profiles.STORY_ERAS, SUMMARY_ERA_SHORT
    )):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=(
                "rgba(139,94,52,.035)" if index % 2 == 0
                else "rgba(49,95,120,.035)"
            ),
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=(start + end) / 2,
            y=1.025,
            xref="x",
            yref="paper",
            text=era_annotation_labels.get(short, short),
            showarrow=False,
            font=dict(size=9, color=MUTED),
        )
    annotation_offsets = {
        1827: (-28, -34),
        1863: (-26, 34),
        1881: (24, 34),
        1944: (22, -34),
        1966: (-28, -34),
        2016: (-34, 34),
        2023: (-32, -34),
    }
    for year, title, window, legal_count, hype_count, detail in SUMMARY_REGISTER_MOMENTS:
        if year not in frame.index or pd.isna(frame.at[year, "legal_hype_ratio"]):
            raise ValueError(
                f"summary register moment needs supported year {year}"
            )
        ratio = float(frame.at[year, "legal_hype_ratio"])
        ax, ay = annotation_offsets[year]
        fig.add_annotation(
            x=year,
            # Plotly annotation coordinates on a logarithmic axis use log10 units.
            y=float(np.log10(ratio)),
            xref="x",
            yref="y",
            text=str(year),
            showarrow=True,
            arrowhead=0,
            arrowwidth=1.2,
            arrowcolor="#315f78",
            ax=ax,
            ay=ay,
            bgcolor="#f5f8fa",
            bordercolor="#315f78",
            borderwidth=1,
            borderpad=4,
            font=dict(size=9, color="#213f50"),
            hovertext="<br>".join(textwrap.wrap(
                SUMMARY_REGISTER_HOVER_SUMMARIES[year],
                width=26,
                break_long_words=False,
                break_on_hyphens=False,
            )),
        )
    fig.update_layout(**_layout(
        height=620,
        title=dict(
            text="Legal/procedural ÷ hype over time",
            font=dict(size=15),
            x=0,
            xanchor="left",
            y=.98,
        ),
        showlegend=False,
        hovermode="x",
        hoverlabel=dict(
            align="left",
            bgcolor="#f5f8fa",
            bordercolor="#315f78",
            font=dict(family=FONT, size=12, color="#213f50"),
        ),
        xaxis=dict(
            title="center year of supported five-year window",
            range=[first_year, last_year],
            gridcolor=GRID,
            linecolor=BASELINE,
        ),
        yaxis=dict(
            title="legal/procedural matches per hype match · log scale",
            type="log",
            tickmode="array",
            tickvals=[.3, .5, 1, 2, 5, 10, 20, 50, 100],
            ticktext=["0.3", "0.5", "1", "2", "5", "10", "20", "50", "100"],
            gridcolor=GRID,
            linecolor=BASELINE,
        ),
        margin=dict(l=72, r=28, t=125, b=62),
    ))
    return fig


def _president_era_controls_html(
    figure_key: str,
    *,
    height: int = 620,
    extra_class: str = "",
    control_style: str = "multi",
    alternate_figure_key: str | None = None,
    comparison_label: str = "Register",
    timeline_status: str = (
        "Over time: the legal/procedural-per-hype ratio is shown for every "
        "supported centered five-year window; year labels explain seven selected "
        "corpus movements; era emphasis is unavailable in this view."
    ),
) -> str:
    """Shared dim-only Story-era controls for fixed-membership portraits."""
    if control_style not in {"multi", "single"}:
        raise ValueError(f"unknown president-era control style: {control_style}")
    chips = "".join(
        '<label class="era-chip" '
        f'style="--era-chip-color:{color}">'
        '<input type="checkbox" data-era-choice '
        f'value="{html.escape(name, quote=True)}" checked>'
        '<span class="era-chip-mark" aria-hidden="true"></span>'
        '<span class="era-chip-copy">'
        f'<strong>{html.escape(short)}</strong><small>{start}–{end}</small>'
        f'<span class="sr-only"> · {html.escape(name)}</span></span></label>'
        for (name, start, end), short, color in zip(
            era_profiles.STORY_ERAS, SUMMARY_ERA_SHORT, SUMMARY_ERA_COLORS
        )
    )
    preset_buttons = "".join(
        '<button type="button" data-era-preset="'
        + html.escape(
            "||".join(era_profiles.STORY_ERAS[index][0] for index in indexes),
            quote=True,
        )
        + f'">{html.escape(label)}</button>'
        for label, indexes in SUMMARY_ERA_PRESETS
    )
    class_names = "president-era-explorer"
    if extra_class:
        class_names += " " + extra_class
    if control_style == "single":
        select_id = f"{figure_key}-era-select"
        chart_id = f"{figure_key}-chart"
        status_id = f"{figure_key}-era-status"
        era_options = "".join(
            f'<option value="{html.escape(name, quote=True)}">'
            f'{html.escape(short)} · {start}–{end}</option>'
            for (name, start, end), short in zip(
                era_profiles.STORY_ERAS, SUMMARY_ERA_SHORT
            )
        )
        view_switch = ""
        if alternate_figure_key is not None:
            view_switch = f"""<div class="register-view-switch" role="group"
         aria-label="{html.escape(comparison_label, quote=True)} comparison view">
      <span>View</span>
      <button type="button" data-register-view="presidents" aria-pressed="true"
              aria-controls="{html.escape(chart_id, quote=True)}">By president</button>
      <button type="button" data-register-view="timeline" aria-pressed="false"
              aria-controls="{html.escape(chart_id, quote=True)}">Over time</button>
    </div>"""
        controls = f"""<div class="era-select-bar{' has-register-switch' if view_switch else ''}">
    {view_switch}
    <div class="era-select-control">
      <label for="{html.escape(select_id, quote=True)}">Emphasize an era</label>
      <select id="{html.escape(select_id, quote=True)}" data-era-select
              aria-describedby="{html.escape(status_id, quote=True)}">
        <option value="__all__" selected>All eras</option>
        <option value="__none__">Dim all</option>
        <optgroup label="Choose one era">{era_options}</optgroup>
      </select>
    </div>
    <p class="filter-status" id="{html.escape(status_id, quote=True)}"
       aria-live="polite">All nine eras emphasized.</p>
  </div>"""
    else:
        controls = f"""<p class="era-control-intro"><strong>Highlight a chapter.</strong> Start with a broad
  historical phase or refine it with the nine era cards. Unselected presidents remain visible.</p>
  <div class="era-preset-actions" aria-label="Historical phase presets">
    {preset_buttons}
    <button type="button" data-era-all>All eras</button>
    <button type="button" data-era-clear>Dim all</button>
  </div>
  <fieldset class="era-choices"><legend>Refine by specific era</legend>{chips}</fieldset>
  <p class="filter-status" aria-live="polite">All nine eras highlighted; all plotted presidents visible.</p>"""
    alternate_attribute = (
        f' data-register-timeline-figure="{html.escape(alternate_figure_key, quote=True)}"'
        if alternate_figure_key is not None else ""
    )
    timeline_status_attribute = (
        f' data-timeline-status="{html.escape(timeline_status, quote=True)}"'
        if alternate_figure_key is not None else ""
    )
    chart_id_attribute = (
        f' id="{html.escape(figure_key, quote=True)}-chart"'
        if control_style == "single" else ""
    )
    chart_scroll_class = (
        "chart-scroll register-chart-scroll"
        if control_style == "single" else "chart-scroll"
    )
    chart_scroll_attributes = (
        ' tabindex="0" aria-label="'
        + html.escape(comparison_label, quote=True)
        + ' chart; scroll horizontally on narrow screens"'
        if control_style == "single" else ""
    )
    return f"""<div class="{html.escape(class_names, quote=True)}"
     data-era-figure="{html.escape(figure_key, quote=True)}"{alternate_attribute}{timeline_status_attribute}>
  {controls}
  <div class="{chart_scroll_class}"{chart_scroll_attributes}>
    <div class="chart"{chart_id_attribute} data-fig="{html.escape(figure_key, quote=True)}"
       style="height:{height}px"></div></div>
</div>"""


def _president_era_html(
    *,
    figure_key: str,
    evidence_metric: str,
    evidence_source: str,
    evidence_support: str,
    evidence_caveat: str,
    control_style: str = "multi",
    alternate_figure_key: str | None = None,
) -> str:
    return f"""{_president_era_controls_html(
        figure_key,
        control_style=control_style,
        alternate_figure_key=alternate_figure_key,
    )}
{_chart_evidence(
    evidence_metric, evidence_source,
    support=evidence_support,
    caveat=evidence_caveat,
)}
"""


def _procedural_era_html(_scores: pd.DataFrame, _rates: pd.DataFrame) -> str:
    portrait_field = _president_era_html(
        figure_key="procedural_eras",
        evidence_metric="legal_procedural",
        evidence_source="speech_markers.parquet",
        evidence_support=(
            "By president pools every word in all 45 corpus records; Over time uses the "
            "existing centered five-year pooled-word windows above the 20,000-word floor."
        ),
        evidence_caveat=(
            "The president axes are separate dictionary rates. The time view divides the "
            "legal/procedural count by the hype count; either component can move the ratio, "
            "and the dictionaries differ in breadth. Vocabulary is not competence, "
            "productivity, policy depth, or policy effectiveness. Event lines provide "
            "historical context, administration lines mark transitions, and A/B explain "
            "window composition; none is a causal estimate."
        ),
        control_style="single",
        alternate_figure_key="summary_register_timeline",
    )
    return f"""<div class="summary-register-bridge">
  <p class="summary-register-kicker">Who + how → register</p>
  <h3>Audience, delivery, and register across the same eras</h3>
  <p>Read the two bars into the portrait field below. In this corpus, the opening record is
  less legal/procedural than its long nineteenth-century middle. In later public- and
  broadcast-heavy eras, legal/procedural wording recedes while hype rises. These patterns
  coincide in the collection; the charts do not establish that audience or delivery caused
  the vocabulary change. Use the switch below to read individual presidential records or the
  legal/procedural-per-hype ratio over time.</p>
  <ol class="summary-register-sequence" aria-label="Voice transition across the corpus">
    <li><span>Who</span><strong>Congress → public</strong></li>
    <li><span>How</span><strong>Written → spoken + broadcast</strong></li>
    <li><span>Register</span><strong>Less procedural → procedural middle → higher hype later</strong></li>
  </ol>
</div>
<noscript><p class="summary-register-noscript">The interactive register chart requires JavaScript.</p></noscript>
{portrait_field}"""


def _summary_temporal_portrait_html(contract: dict) -> str:
    rows = contract["rows"]
    has_plotted_rows = rows[["future", "nostalgia"]].notna().all(axis=1).any()
    thin_count = int(rows["support_status"].eq("thin_record").sum())
    chart_html = (
        _president_era_controls_html(
            "summary_temporal",
            height=int(SUMMARY_TEMPORAL_CHART_HEIGHT_PX),
            extra_class="summary-temporal-explorer",
            control_style="single",
            alternate_figure_key="summary_temporal_timeline",
            comparison_label="Tomorrow and yesterday",
            timeline_status=(
                "Over time: tomorrow and yesterday are separate lines for every "
                "supported four-year rolling average; era emphasis is unavailable "
                "in this view."
            ),
        )
        if has_plotted_rows
        else (
            '<div class="summary-temporal-explorer temporal-na-chart">'
            '<div class="chart-scroll"><div class="chart" '
            'data-fig="summary_temporal" '
            f'style="height:{int(SUMMARY_TEMPORAL_CHART_HEIGHT_PX)}px"></div>'
            '</div></div>'
        )
    )
    return f"""<section class="summary-temporal-portrait"
     data-temporal-schema="{html.escape(str(contract['schema_version']), quote=True)}"
     data-temporal-treatment="{html.escape(str(contract['treatment']), quote=True)}">
  <header class="temporal-portrait-heading"><div>
    <span>TOMORROW + YESTERDAY</span>
    <h3>Two temporal appeals, viewed together</h3>
    <p>Future-family and nostalgia-family matches per 10,000 marker words.</p></div>
  </header>
  <p class="temporal-denominator">x = future-family matches per 10,000 marker words ·
  y = nostalgia-family matches per 10,000 marker words · portraits use a uniform size ·
  {thin_count} records below the {contract['support_min_speeches']}-speech precision floor use an amber halo</p>
  <noscript><p class="summary-register-noscript">The interactive temporal chart requires JavaScript.</p></noscript>
  {chart_html}
</section>
{_chart_evidence(
    "temporal_portrait",
    str(contract["source_label"]),
    support=(
        "All 45 presidents aggregate every available corpus speech assigned to that "
        "document president. The time view averages four consecutive annual rates, plots each "
        "at the ending year, requires coverage in all four years, and applies a 10,000-word "
        "window floor. Both views divide exact future- and nostalgia-family matches by marker words."
    ),
    caveat=(
        "These are unadjusted all-genre, document-owned transcript aggregates rather than "
        "speaker-audited claims. They are sensitive to genre, coverage, quotations, questions, "
        "and other speakers. Each year receives equal weight in the rolling average even when "
        "its speech volume differs. Dictionary matches cannot establish intent, feasibility, "
        "or historical accuracy."
    ),
)}"""


PROGRESSIVE_PERIODS = [
    ("1913–18", 1913, 1918),
    ("1919–28", 1919, 1928),
    ("1929–32", 1929, 1932),
]


def _progressive_deltas(para_labels: pd.DataFrame) -> pd.DataFrame:
    topics = [
        "Economy & jobs", "Money & banking", "Trade & tariffs",
        "War & military", "Foreign policy",
    ]
    baseline = para_labels[
        para_labels.year.between(1869, 1912)
    ][topics].mean() * 100
    rows = {}
    for label, lo, hi in PROGRESSIVE_PERIODS:
        rows[label] = (
            para_labels[para_labels.year.between(lo, hi)][topics].mean() * 100
            - baseline
        )
    return pd.DataFrame(rows).reindex(topics)


def fig_progressive_delta(para_labels: pd.DataFrame, stage: int) -> go.Figure:
    delta = _progressive_deltas(para_labels)
    extent = float(np.nanmax(np.abs(delta.to_numpy()))) * 1.16
    fig = go.Figure()
    palette = ["#2f7397", "#7a568f", "#b16b3e", "#9b4e50"]
    symbols = ["circle", "diamond", "square", "star"]
    for index, label in enumerate(delta.columns[:stage]):
        fig.add_trace(go.Scatter(
            x=delta[label], y=delta.index, mode="markers", name=label,
            marker=dict(color=palette[index], symbol=symbols[index], size=12),
            hovertemplate=(
                f"<b>{label}</b><br>%{{y}}: %{{x:+.1f}} percentage points "
                "vs 1869–1912<extra></extra>"
            ),
        ))
    fig.add_vline(
        x=0, line_color=INK2, line_width=1.2,
        annotation_text="1869–1912 baseline", annotation_position="top",
    )
    fig.update_layout(**_layout(
        height=470,
        title=dict(
            text=f"Stage {stage} · {PROGRESSIVE_PERIODS[stage - 1][0]} added; shared scale",
            font=dict(size=15),
        ),
        margin=dict(l=145, r=25, t=66, b=55),
        xaxis=dict(
            title="percentage-point change from 1869–1912",
            range=[-extent, extent], gridcolor=GRID, linecolor=BASELINE,
        ),
        yaxis=dict(gridcolor=SURFACE, linecolor=BASELINE),
        legend=dict(orientation="h", y=1.06, x=0),
    ))
    return fig


def progressive_takeaway(para_labels: pd.DataFrame) -> str:
    delta = _progressive_deltas(para_labels)
    money = delta.loc["Money & banking"]
    war = delta.loc["War & military"]
    war_peak = str(war.idxmax())
    return (
        "The agenda changes hands rather than moving as one field. Relative to "
        f"the 1869–1912 paragraph-share baseline, war reaches its largest "
        f"departure in {war_peak} ({war[war_peak]:+.1f} points), while "
        f"money/banking moves from {money['1913–18']:+.1f} points in the "
        f"reform-and-war window to {money['1929–32']:+.1f} in the collapse window."
    )


def fig_progressive_heatmap(para_labels: pd.DataFrame) -> go.Figure:
    """Comparable three-window agenda handoff, replacing disconnected points."""
    delta = _progressive_deltas(para_labels)
    topics = list(delta.index)
    windows = list(delta.columns)
    baseline = (
        para_labels[
            para_labels.year.between(1869, 1912)
        ][topics].mean() * 100
    )
    period_levels = {}
    for label, lo, hi in PROGRESSIVE_PERIODS:
        period_levels[label] = (
            para_labels[para_labels.year.between(lo, hi)][topics].mean() * 100
        )
    levels = pd.DataFrame(period_levels).reindex(index=topics, columns=windows)
    extent = float(np.nanmax(np.abs(delta.to_numpy())))
    fig = go.Figure(go.Heatmap(
        z=delta.to_numpy(), x=windows, y=topics,
        zmin=-extent, zmax=extent, zmid=0,
        colorscale=[
            [0, "#486c87"], [.5, "#f3eee6"], [1, "#a34f3f"],
        ],
        text=np.vectorize(lambda value: f"{value:+.1f} pp")(delta.to_numpy()),
        texttemplate="%{text}",
        textfont=dict(size=12),
        customdata=np.stack([
            levels.to_numpy(),
            np.repeat(baseline.to_numpy()[:, None], len(windows), axis=1),
        ], axis=-1),
        hovertemplate=(
            "<b>%{y} · %{x}</b><br>%{z:+.1f} points vs 1869–1912"
            "<br>period share %{customdata[0]:.1f}%"
            "<br>prior-era baseline %{customdata[1]:.1f}%<extra></extra>"
        ),
        colorbar=dict(title="pp vs<br>1869–1912", thickness=12),
    ))
    for x, y, label in [
        ("1913–18", 1.12, "1917 · U.S. enters WWI"),
        ("1929–32", 1.12, "1929 · market crash"),
        ("1929–32", 1.05, "1930 · Smoot–Hawley"),
    ]:
        fig.add_annotation(
            x=x, y=y, xref="x", yref="paper", text=label,
            showarrow=False, font=dict(size=10, color="#6f4828"),
        )
    fig.update_layout(**_layout(
        height=520,
        title=dict(
            text="Issue movement from the 1869–1912 paragraph-share baseline",
            font=dict(size=15),
        ),
        margin=dict(l=145, r=40, t=105, b=68),
        xaxis=dict(title="historical window", linecolor=BASELINE),
        yaxis=dict(autorange="reversed", linecolor=BASELINE),
    ))
    return fig


def naming_crossover_year(rates: pd.DataFrame, run: int = 5) -> int:
    """First year beginning a sustained `run`-year America-over-U.S. crossover."""
    frame = rates[["united_states", "america"]].dropna()
    years = [
        int(year) for year in frame.index
        if all(
            later in frame.index
            and frame.loc[later, "america"] > frame.loc[later, "united_states"]
            for later in range(int(year), int(year) + run)
        )
    ]
    if not years:
        raise ValueError("no sustained naming crossover in the measured record")
    return years[0]


def fig_naming_progressive(
    rates: pd.DataFrame, speeches: pd.DataFrame | None = None,
) -> go.Figure:
    frame = rates[["united_states", "america"]].copy()
    crossover = naming_crossover_year(rates)
    context = (
        _centered_window_context(speeches, "word_count", window=5).reindex(frame.index)
        if speeches is not None else None
    )
    marker_sizes = [
        5 if year in (frame.index[0], frame.index[-1]) or year % 10 == 0 else 0
        for year in frame.index
    ]
    fig = go.Figure()
    for label, summary, column, color, marker_symbol in [
        (
            "“United States”", "the exact national name",
            "united_states", "#8b6c42", "square",
        ),
        (
            "“America / American(s)”", "the shorter national-name family",
            "america", "#2f7397", "circle",
        ),
    ]:
        customdata = (
            context["presidents"].to_numpy()
            if context is not None else None
        )
        hovertemplate = (
            f"<b>{label}</b> · {summary}"
            "<br><b>%{x}:</b> %{y:.1f} per 10,000 words"
            "<br><b>Presidents:</b> %{customdata}<extra></extra>"
            if context is not None else
            f"<b>{label}</b> · {summary}"
            "<br><b>%{x}:</b> %{y:.1f} per 10,000 words<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=frame.index, y=frame[column], mode="lines+markers", name=label,
            connectgaps=False,
            line=dict(color=color, width=3, dash="solid"),
            marker=dict(
                color=color,
                line=dict(color="#fcfcfb", width=.8),
                size=marker_sizes,
                symbol=marker_symbol,
            ),
            customdata=customdata,
            hovertemplate=hovertemplate,
        ))
    fig.add_vline(
        x=crossover, line_color="#6f4828", line_dash="dot",
        annotation_text=f"{crossover} · first sustained 5-year crossover",
        annotation_position="top left",
    )
    fig.update_layout(**_layout(
        height=430,
        hoverlabel=dict(
            align="left", bgcolor="rgba(255,253,249,.94)",
            font=dict(family=FONT, size=10),
        ),
        xaxis=dict(range=X_RANGE, title="center year of five-year window",
                   gridcolor=GRID, linecolor=BASELINE),
        yaxis=dict(title="uses per 10,000 words", rangemode="tozero",
                   gridcolor=GRID, linecolor=BASELINE),
    ))
    return fig


SUMMARY_LANGUAGE_WINDOW_YEARS = 7
SUMMARY_LANGUAGE_MIN_WORDS = 10_000
SUMMARY_NECESSITY_NEED_RE = re.compile(
    r"\bneed(?:s|ed|ing)?\b", re.IGNORECASE
)
SUMMARY_NECESSITY_OBLIGATION_RE = re.compile(
    r"\b(?:have|has|had|got) to\b|\bgotta\b", re.IGNORECASE
)
SUMMARY_COMMITMENT_EXPLICIT_RE = re.compile(
    r"(?:"
    r"\b(?:i|we|this administration|my administration|the united states|america)\s+"
    r"(?:"
    r"(?:hereby\s+)?(?:promise|pledge|vow|guarantee|commit)\b"
    r"|(?:intend(?:s)?|resolve(?:s)?|plan(?:s)?)\s+to\b"
    r")"
    r"|\b(?:i\s+am|i'm|we\s+are|we're|this administration\s+is|"
    r"my administration\s+is|the united states\s+is|america\s+is)\s+"
    r"(?:(?:fully|deeply|firmly|irrevocably|absolutely)\s+)?"
    r"(?:committed|determined)\s+to\b"
    r")",
    re.IGNORECASE,
)
SUMMARY_STANCE_FAMILIES = (
    (
        "Necessity",
        "must, need, and obligation phrases",
        "#a24f52",
        "circle",
    ),
    (
        "Commitment / intent",
        "will, shall, and explicit commitments",
        "#315f78",
        "square",
    ),
    (
        "Absolute emphasis",
        "absolute and emphatic word forms",
        "#986a32",
        "diamond",
    ),
    ("Conditional", "would and could", "#6d5d8f", "triangle-up"),
    (
        "Advice / possibility",
        "should and possibility terms",
        "#4f7557",
        "x",
    ),
)


def _centered_window_context(
    rows: pd.DataFrame,
    denominator: str,
    *,
    window: int = SUMMARY_LANGUAGE_WINDOW_YEARS,
) -> pd.DataFrame:
    """Human-readable support and contributing names for every center year."""
    if window % 2 != 1:
        raise ValueError("summary language context requires an odd window")
    radius = window // 2
    first_year = int(rows["year"].min())
    last_year = int(rows["year"].max())
    contexts = []
    for year in range(first_year, last_year + 1):
        start = max(first_year, year - radius)
        end = min(last_year, year + radius)
        subset = rows.loc[rows["year"].between(start, end)].sort_values(
            ["year", "date", "president", "doc_name"]
        )
        presidents = list(dict.fromkeys(subset["president"].astype(str)))
        contexts.append({
            "year": year,
            "window": f"{start}–{end}",
            "presidents": ", ".join(presidents),
            "window_words": int(subset[denominator].sum()),
            "n_speeches": int(len(subset)),
        })
    return pd.DataFrame(contexts).set_index("year")


def summary_stance_family_rates(
    stats: pd.DataFrame,
    markers: pd.DataFrame,
    speeches: pd.DataFrame,
    *,
    window: int = SUMMARY_LANGUAGE_WINDOW_YEARS,
    min_words: int = SUMMARY_LANGUAGE_MIN_WORDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Five disjoint, auditable surface families pooled over centered windows."""
    if any(
        table["doc_name"].duplicated().any()
        for table in (stats, markers, speeches)
    ):
        raise ValueError("summary stance families require one row per speech")
    key_sets = [set(table["doc_name"]) for table in (stats, markers, speeches)]
    if key_sets[0] != key_sets[1] or key_sets[0] != key_sets[2]:
        raise ValueError("speech, speech-stat, and marker key sets differ")
    if speeches["transcript"].isna().any():
        raise ValueError("summary stance families require every transcript")
    marker_cols = [
        "doc_name", "president", "date", "year", "n_words", "boosters", "hedges",
    ]
    stat_cols = [
        "doc_name", "president", "date", "year", "modal_must", "modal_will",
        "modal_shall", "modal_would", "modal_could", "modal_should",
    ]
    joined = markers[marker_cols].merge(
        stats[stat_cols], on="doc_name", how="inner", validate="one_to_one",
        suffixes=("_marker", "_stat"),
    )
    speech_counts = speeches[[
        "doc_name", "president", "date", "year", "transcript",
    ]].copy()
    speech_counts["need_forms"] = speech_counts["transcript"].str.count(
        SUMMARY_NECESSITY_NEED_RE
    )
    speech_counts["obligation_phrases"] = speech_counts["transcript"].str.count(
        SUMMARY_NECESSITY_OBLIGATION_RE
    )
    speech_counts["explicit_commitment"] = speech_counts["transcript"].map(
        lambda transcript: len(SUMMARY_COMMITMENT_EXPLICIT_RE.findall(transcript))
    )
    speech_counts = speech_counts.rename(columns={
        column: f"{column}_speech" for column in ("president", "date", "year")
    })
    joined = joined.merge(
        speech_counts.drop(columns="transcript"),
        on="doc_name", how="inner", validate="one_to_one",
    )
    for column in ("president", "date", "year"):
        left = joined[f"{column}_marker"]
        right = joined[f"{column}_stat"]
        if not left.equals(right.rename(left.name)):
            raise ValueError(f"speech-stat and marker {column} metadata differs")
        speech = joined[f"{column}_speech"]
        if not left.equals(speech.rename(left.name)):
            raise ValueError(f"speech and marker {column} metadata differs")
        joined[column] = left
    joined["necessity"] = (
        joined["modal_must"]
        + joined["need_forms"]
        + joined["obligation_phrases"]
    )
    joined["commitment"] = (
        joined["modal_will"]
        + joined["modal_shall"]
        + joined["explicit_commitment"]
    )
    joined["absolute_emphasis"] = joined["boosters"]
    joined["conditional"] = joined["modal_would"] + joined["modal_could"]
    joined["advice_possibility"] = joined["modal_should"] + joined["hedges"]
    family_columns = [
        "necessity", "commitment", "absolute_emphasis", "conditional",
        "advice_possibility",
    ]
    annual = joined.groupby("year")[[*family_columns, "n_words"]].sum()
    years = pd.RangeIndex(
        int(joined["year"].min()), int(joined["year"].max()) + 1, name="year"
    )
    annual = annual.reindex(years, fill_value=0)
    rolled = annual.rolling(window, center=True, min_periods=1).sum()
    rates = rolled[family_columns].div(rolled["n_words"], axis=0) * 10_000
    rates.loc[rolled["n_words"].lt(min_words), :] = np.nan
    context = _centered_window_context(joined, "n_words", window=window)
    if not np.array_equal(
        context["window_words"].to_numpy(), rolled["n_words"].to_numpy(dtype=int)
    ):
        raise ValueError("summary stance window support does not reconcile")
    return rates, context


def fig_summary_modals(
    stats: pd.DataFrame,
    markers: pd.DataFrame,
    speeches: pd.DataFrame,
) -> go.Figure:
    """Grouped stance words without turning them into a latent certainty index."""
    rates, context = summary_stance_family_rates(stats, markers, speeches)
    fig = go.Figure()
    marker_sizes = [
        5 if year in (rates.index[0], rates.index[-1]) or year % 10 == 0 else 0
        for year in rates.index
    ]
    for (label, summary, color, marker_symbol), column in zip(
        SUMMARY_STANCE_FAMILIES, rates.columns
    ):
        fig.add_trace(go.Scatter(
            x=rates.index,
            y=rates[column],
            mode="lines+markers",
            name=label,
            connectgaps=False,
            line=dict(color=color, width=2.7, dash="solid"),
            marker=dict(
                color=color,
                line=dict(color="#fcfcfb", width=.8),
                size=marker_sizes,
                symbol=marker_symbol,
            ),
            customdata=np.stack([
                context["window"], context["presidents"],
                context["window_words"], context["n_speeches"],
            ], axis=-1),
            hovertemplate=(
                f"<b>{label}</b> · {summary}"
                "<br><b>%{x}:</b> %{y:.1f} per 10,000 words"
                "<br><b>Presidents:</b> %{customdata[1]}"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(**_timeline_layout(
        height=470,
        margin=dict(l=62, r=22, t=58, b=58),
        title=dict(
            text="Stance and absolute-emphasis word families",
            font=dict(size=15),
        ),
        hoverlabel=dict(
            align="left", bgcolor="rgba(255,253,249,.94)",
            font=dict(family=FONT, size=10),
        ),
        xaxis=dict(
            range=X_RANGE,
            title="center year of seven-year window",
            gridcolor=GRID,
            linecolor=BASELINE,
        ),
        yaxis=dict(
            title="uses per 10,000 words",
            rangemode="tozero",
            gridcolor=GRID,
            linecolor=BASELINE,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, x=0,
            font=dict(size=11, color=INK2),
        ),
    ))
    fig.update_xaxes(title="center year of seven-year window")
    return fig


def _decade_rate(stats: pd.DataFrame, column: str, decade: int) -> float:
    rows = stats.loc[(stats["year"] // 10 * 10).eq(decade)]
    if rows.empty or int(rows["n_tokens"].sum()) <= 0:
        raise ValueError(f"no speech-stat support for the {decade}s")
    return float(rows[column].sum() / rows["n_tokens"].sum() * 10_000)


def fig_summary_pronouns(stats: pd.DataFrame) -> go.Figure:
    """First-person families over time; levels, not a latent individualism score."""
    context = _centered_window_context(
        stats, "n_tokens", window=SUMMARY_LANGUAGE_WINDOW_YEARS
    )
    fig = go.Figure()
    marker_sizes = [
        5 if year in (context.index[0], context.index[-1]) or year % 10 == 0 else 0
        for year in context.index
    ]
    for label, summary, column, color, marker_symbol in [
        (
            "we / us / our family", "collective first-person language",
            "we_count", "#315f78", "circle",
        ),
        (
            "I / me / my family", "singular first-person language",
            "i_count", "#a24f52", "square",
        ),
    ]:
        series = _stats_yearly(
            stats, column, window=SUMMARY_LANGUAGE_WINDOW_YEARS,
            min_tokens=SUMMARY_LANGUAGE_MIN_WORDS,
        )
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines+markers",
            name=label,
            connectgaps=False,
            line=dict(color=color, width=3, dash="solid"),
            marker=dict(
                color=color,
                line=dict(color="#fcfcfb", width=.8),
                size=marker_sizes,
                symbol=marker_symbol,
            ),
            customdata=np.stack([
                context["window"], context["presidents"],
                context["window_words"], context["n_speeches"],
            ], axis=-1),
            hovertemplate=(
                f"<b>{label}</b> · {summary}"
                "<br><b>%{x}:</b> %{y:.1f} per 10,000 words"
                "<br><b>Presidents:</b> %{customdata[1]}"
                "<extra></extra>"
            ),
        ))
    current_i = _decade_rate(stats, "i_count", 2020)
    current_we = _decade_rate(stats, "we_count", 2020)
    fig.add_vrect(
        x0=2020, x1=2026, fillcolor="rgba(162,79,82,.08)", line_width=0,
    )
    fig.add_annotation(
        x=2024,
        y=.92,
        yref="paper",
        text=f"<b>2020s pooled</b><br>I {current_i:.0f} · we {current_we:.0f}",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="right",
        bgcolor="rgba(255,253,249,.94)",
        bordercolor="#cbb89f",
        borderpad=4,
        font=dict(size=11, color="#6f4828"),
    )
    fig.update_layout(**_timeline_layout(
        height=470,
        margin=dict(l=62, r=22, t=58, b=58),
        title=dict(
            text="Collective language still leads; the 2020s interrupt its long rise",
            font=dict(size=15),
        ),
        hoverlabel=dict(
            align="left", bgcolor="rgba(255,253,249,.94)",
            font=dict(family=FONT, size=10),
        ),
        xaxis=dict(
            range=X_RANGE,
            title="center year of seven-year window",
            gridcolor=GRID,
            linecolor=BASELINE,
        ),
        yaxis=dict(
            title="uses per 10,000 words",
            rangemode="tozero",
            gridcolor=GRID,
            linecolor=BASELINE,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, x=0,
            font=dict(size=11, color=INK2),
        ),
    ))
    fig.update_xaxes(title="center year of seven-year window")
    return fig


def summary_divisiveness_contract(
    combat_ratios: pd.DataFrame,
    era_similarity: pd.DataFrame,
    stats: pd.DataFrame,
) -> dict:
    """Extract the three published signals and fail closed on their scopes."""
    ratio_keys = {
        "party": ("party_attack", "Civil War & Reconstruction"),
        "enemy": ("enemy_naming", "The founding"),
        "zero_sum": ("zero_sum", "The founding"),
    }
    ratio_rows = {}
    for key, (flag, denominator) in ratio_keys.items():
        rows = combat_ratios.loc[
            combat_ratios["numerator"].eq("The present era")
            & combat_ratios["denominator"].eq(denominator)
            & combat_ratios["flag"].eq(flag)
            & combat_ratios["treatment"].eq("sotu_only")
        ]
        if len(rows) != 1:
            raise ValueError(
                f"expected one present-era SOTU ratio row for {flag}/{denominator}"
            )
        row = rows.iloc[0]
        if row["ci_status"] != "low_cluster_caution":
            raise ValueError(f"unexpected CI status for {flag}: {row['ci_status']}")
        if not all(np.isfinite(float(row[column])) for column in ("ratio", "ci_lo", "ci_hi")):
            raise ValueError(f"ratio interval is not finite for {flag}")
        ratio_rows[key] = row

    present_similarity = era_similarity.loc[
        era_similarity["grain"].eq("era")
        & era_similarity["matrix"].eq("detrended")
        & era_similarity["unit_i"].eq("The present era")
        & era_similarity["unit_j"].ne("The present era")
    ].sort_values(["cosine", "unit_j"], ascending=[False, True])
    if present_similarity.empty:
        raise ValueError("present-era detrended similarity rows are missing")
    nearest = present_similarity.iloc[0]
    if nearest["unit_j"] != "Civil War & Reconstruction":
        raise ValueError("present-era detrended nearest neighbor changed")

    current_i = _decade_rate(stats, "i_count", 2020)
    current_we = _decade_rate(stats, "we_count", 2020)
    prior_i = {
        decade: _decade_rate(stats, "i_count", decade)
        for decade in sorted(set((stats["year"] // 10 * 10).astype(int)))
        if 1800 <= decade < 2020
    }
    if not prior_i or current_i <= max(prior_i.values()):
        raise ValueError("2020s first-person singular rate is no longer highest since the 1790s")

    return {
        "party": ratio_rows["party"],
        "enemy": ratio_rows["enemy"],
        "zero_sum": ratio_rows["zero_sum"],
        "nearest": nearest,
        "current_i": current_i,
        "current_we": current_we,
    }


def _summary_divisiveness_html(
    combat_ratios: pd.DataFrame,
    era_similarity: pd.DataFrame,
    stats: pd.DataFrame,
) -> str:
    contract = summary_divisiveness_contract(combat_ratios, era_similarity, stats)
    party = contract["party"]
    enemy = contract["enemy"]
    zero_sum = contract["zero_sum"]
    nearest = contract["nearest"]
    return f"""<section class="summary-divisiveness" aria-labelledby="summary-divisiveness-title">
  <p class="summary-language-kicker">THREE TESTS OF A BROAD CLAIM</p>
  <h3 id="summary-divisiveness-title">Is presidential rhetoric more divisive now?</h3>
  <p class="summary-divisiveness-verdict"><strong>More openly partisan, yes. More divisive in every broader sense, no.</strong> The three signals answer different questions, so they
  should not be combined into one index.</p>
  <div class="summary-divisiveness-grid">
    <article class="summary-divisiveness-card is-strong">
      <p>THE RESULT I STAND BEHIND</p>
      <strong>{float(party['ratio']):.1f}×</strong>
      <h4>Opposing-party attack in annual messages</h4>
      <p>The present-era rate is {float(party['ratio']):.1f} times the Civil War &amp;
      Reconstruction rate; the 95% speech-clustered bootstrap interval is
      {float(party['ci_lo']):.1f}–{float(party['ci_hi']):.1f}×. The interval excludes one
      inside the one speech genre represented in every era.</p>
    </article>
    <article class="summary-divisiveness-card">
      <p>A DESCRIPTIVE SHIFT, NOT A DIVISION SCORE</p>
      <strong>{contract['current_i']:.0f} <small>vs {contract['current_we']:.0f}</small></strong>
      <h4>First-person singular versus plural in the 2020s</h4>
      <p>I/me/my-family wording reaches its highest decade rate since the 1790s while
      we/us/our-family wording falls from its 2010s level. I stand behind those lexical rates;
      I would not call them evidence of divisiveness by themselves.</p>
    </article>
    <article class="summary-divisiveness-card">
      <p>AN ANALOGY I WOULD CAVEAT</p>
      <strong>{float(nearest['cosine']):.3f}</strong>
      <h4>Nearest drift-detrended era: Civil War &amp; Reconstruction</h4>
      <p>This is the present era’s closest relative match after removing adjacent-era drift,
      but {float(nearest['cosine']):.3f} is modest absolute similarity. It is a historical
      rhyme, not evidence that the present is a rerun of the Civil War era.</p>
    </article>
  </div>
  <p class="summary-divisiveness-null"><strong>The important limits:</strong> against the
  founding-era annual message, enemy naming is {float(enemy['ratio']):.2f}×
  ({float(enemy['ci_lo']):.2f}–{float(enemy['ci_hi']):.2f}×) and zero-sum framing is
  {float(zero_sum['ratio']):.2f}× ({float(zero_sum['ci_lo']):.2f}–{float(zero_sum['ci_hi']):.2f}×).
  Both intervals include one. The present era has only {int(party['n_speeches_numerator'])}
  annual messages, so all three ratio intervals carry the governed low-cluster caution.</p>
  <details class="summary-evidence-detail"><summary>Measures, provenance, and interpretation limits</summary>
    <p><strong>Annual-message comparison.</strong> Paragraph shares from frozen combativeness
    annotations; 95% speech-clustered bootstrap ratios from <code>combat/ratios.parquet</code>.
    The interval component is {html.escape(str(party['ci_components']).replace('_', ' '))}.</p>
    <p><strong>Pronouns.</strong> spaCy-token counts from <code>speech_stats.parquet</code>, pooled
    within the 2020s and divided by all alphabetic tokens. The displayed chart uses centered
    five-year window totals.</p>
    <p><strong>Era analogy.</strong> Detrended cosine similarity from
    <code>eras/era_similarity.parquet</code>; “nearest” is a relative rank, not a closeness
    threshold, causal claim, or president-to-president match.</p>
  </details>
</section>"""


def _summary_language_change_html(
    speeches: pd.DataFrame,
    rates: pd.DataFrame,
    stats: pd.DataFrame,
    combat_ratios: pd.DataFrame,
    era_similarity: pd.DataFrame,
) -> str:
    keys = ["naming_progressive", "summary_modals", "summary_pronouns"]
    labels = ["National name", "Modal words", "Personal voice"]
    announcements = [
        "National name: America overtakes the United States around the 1910 boundary.",
        "Stance families: necessity, commitment and intent, absolute emphasis, condition, and possibility move differently across the full record.",
        "Personal voice: collective terms still lead, but the 2020s reverse the recent direction.",
    ]
    buttons = "".join(
        f'<button type="button" data-stage-button="{index}" '
        f'aria-pressed="{"true" if index == 0 else "false"}">'
        f"{html.escape(label)}</button>"
        for index, label in enumerate(labels)
    )
    return f"""<div class="summary-language-boundary">
  <p class="summary-language-kicker">LANGUAGE ACROSS THE BOUNDARY</p>
  <h3>Three ways presidential wording changes over time</h3>
  <p>Flip between a national naming shift, grouped stance words, and first-person language.
  These are separate lexical descriptions—not one scale of certainty, identity, or division.</p>
  <div class="story-stage summary-language-stage" data-stage-mode="chart"
       data-stage-figs="{html.escape(json.dumps(keys), quote=True)}"
       data-stage-announcements="{html.escape(json.dumps(announcements), quote=True)}">
    <div class="stage-controls" role="group" aria-label="Language change chart">{buttons}</div>
    <p class="stage-status" aria-live="polite">{html.escape(announcements[0])}</p>
    <div class="chart-scroll"><div class="chart staged-chart"
         data-fig="naming_progressive" style="height:470px"></div></div>
    <p class="summary-line-status" aria-live="polite">Hover any line to highlight its full
    trajectory; click or tap to pin it, and press Escape to restore every line.</p>
    <noscript><p>The interactive chart switcher requires JavaScript. The summaries and
    provenance below preserve the main findings.</p></noscript>
  </div>
  {_chart_evidence(
      "rate_10k", "speeches.parquet + speech_markers.parquet + speech_stats.parquet",
      measure_label="Exact word and phrase uses per 10,000 words",
      support="National naming keeps its governed centered five-year/20,000-word view; grouped stance and pronoun lines use centered seven-year windows above 10,000 words, which retain every 1789–2026 center year.",
      caveat="The grouped stance view combines explicit surface forms, not a certainty or extremity index. Need includes verb, adjective, and noun uses; the commitment/intent additions require explicit speaker or administration wording and exclude going to; quotations can contribute to every family. The pronoun families do not measure individualism.",
  )}
  <div class="summary-language-takeaways">
    <article><strong>National name</strong><span>1910 is the first start of five consecutive
    displayed windows with America above the United States.</span></article>
    <article><strong>Grouped stance</strong><span>Necessity now includes the complete need word
    family and exact obligation phrases; commitment/intent joins will and shall to explicit
    speaker or administration pledge and intent phrases; absolute-emphasis wording remains
    separate. Solid lines and distinct decade symbols keep all five trajectories legible. Hover
    any line for a compact family definition, exact rate, year, and contributing
    presidents.</span></article>
    <article><strong>Personal voice</strong><span>The 2020s interrupt a century-long collective
    rise: I-family wording reaches its highest decade rate since the 1790s, but we still leads.</span></article>
  </div>
  <details class="summary-evidence-detail"><summary>Inspect the 1910 naming rule and corpus examples</summary>
    {_naming_explanation_html(speeches, rates)}
  </details>
</div>
{_summary_divisiveness_html(combat_ratios, era_similarity, stats)}"""


def _local_sentence(speech: pd.Series, fragment: str) -> str:
    text = re.sub(r"\s+", " ", str(speech.transcript)).strip()
    sentence = next(
        (
            item.strip() for item in re.split(r"(?<=[.!?])\s+", text)
            if fragment.casefold() in item.casefold()
        ),
        "",
    )
    if not sentence or sentence not in text:
        raise ValueError(f"verified naming excerpt not found: {fragment}")
    return sentence


def _naming_explanation_html(speeches: pd.DataFrame, rates: pd.DataFrame) -> str:
    crossover = naming_crossover_year(rates)
    doc_name = (
        "/the-presidency/presidential-speeches/"
        "december-6-1910-second-annual-message"
    )
    speech = speeches.loc[speeches.doc_name.eq(doc_name)].iloc[0]
    examples = [
        _local_sentence(
            speech,
            "foreign relations of the United States have continued",
        ),
        _local_sentence(
            speech,
            "exact equality between America, Great Britain, France, and Germany",
        ),
    ]
    quote_html = "".join(
        f'<blockquote><p>“{html.escape(sentence)}”</p>'
        f'<cite>{int(speech.year)} · {html.escape(speech.president)}, '
        f'<a href="https://millercenter.org{doc_name}" target="_blank" '
        f'rel="noopener">{html.escape(speech.title)}</a></cite></blockquote>'
        for sentence in examples
    )
    return f"""<div class="naming-method">
<p><strong>Exactly what is counted.</strong> The America family matches the whole words
<em>America</em>, <em>American</em>, and <em>Americans</em>; the comparison family matches
the exact phrase <em>United States</em>. Each count is divided by all words and expressed per
10,000.</p>
<p><strong>Why {crossover} qualifies.</strong> The chart sums counts and words inside a centered
five-year window. A sustained crossover requires five consecutive displayed windows in which
the America family is higher. {crossover} is the first start that passes; 1909 fails.</p>
<p><strong>What it may suggest—and cannot prove.</strong> The pattern may suggest a shift toward
a shorter public name for the nation. It cannot prove a change in national identity: the family
also captures adjectival and geographic uses such as “American commerce” and “Central America,”
the corpus mixes genres, and the Miller Center collection is not exhaustive.</p>
<div class="naming-examples"><h4>Local corpus examples at the transition</h4>{quote_html}</div>
</div>"""


def fig_broadcast_presidency(
    stats: pd.DataFrame, speech_annotations: pd.DataFrame,
) -> go.Figure:
    """Written-to-performed transition is primary; grade level stays secondary."""
    joined = stats.merge(
        speech_annotations[["doc_name", "medium"]],
        on="doc_name", how="left", validate="one_to_one",
    )
    joined = joined[joined.year.between(1900, 1980)].copy()
    joined["decade"] = (joined.year // 10) * 10
    detailed = pd.crosstab(joined.decade, joined.medium)
    totals = detailed.sum(axis=1)
    written = detailed.get(
        "written_message", pd.Series(0, index=detailed.index)
    ) / totals * 100
    performed = 100 - written
    detail_columns = [
        "spoken_address", "broadcast_radio_or_tv", "press_conference", "debate",
    ]
    for column in detail_columns:
        if column not in detailed:
            detailed[column] = 0
    custom = np.stack([
        totals,
        detailed["spoken_address"] / totals * 100,
        detailed["broadcast_radio_or_tv"] / totals * 100,
        detailed["press_conference"] / totals * 100,
        detailed["debate"] / totals * 100,
    ], axis=-1)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[.68, .32],
        subplot_titles=[
            "Primary form assigned within this corpus · written versus performed",
            "Supporting view · median reading level",
        ],
        vertical_spacing=.13,
    )
    for label, values, color, symbol in [
        ("Written message", written, "#8b6c42", "square"),
        ("Performed / spoken form", performed, "#477e9d", "circle"),
    ]:
        fig.add_trace(go.Scatter(
            x=values.index + 5, y=values, mode="lines+markers+text",
            name=label, line=dict(color=color, width=3),
            marker=dict(color=color, size=9, symbol=symbol),
            text=[
                f"{value:.0f}%" if decade in {1900, 1950, 1960, 1970, 1980} else ""
                for decade, value in values.items()
            ],
            textposition="top center",
            customdata=custom,
            hovertemplate=(
                "<b>%{x:.0f}s · %{fullData.name}</b><br>%{y:.1f}% of corpus speeches"
                "<br>%{customdata[0]:.0f} speeches in decade"
                "<br>spoken address %{customdata[1]:.1f}%"
                "<br>radio / TV %{customdata[2]:.1f}%"
                "<br>press conference %{customdata[3]:.1f}%"
                "<br>debate %{customdata[4]:.1f}%<extra></extra>"
            ),
        ), row=1, col=1)
    grade = joined.groupby("year").fk_grade.median().rolling(
        7, center=True, min_periods=3
    ).median()
    fig.add_trace(go.Scatter(
        x=grade.index, y=grade, mode="lines",
        line=dict(color=INK, width=2.7), name="Flesch–Kincaid median",
        hovertemplate="%{x}<br>grade %{y:.1f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_vrect(
        x0=1953, x1=1980, fillcolor="rgba(71,126,157,.08)",
        line_width=1, line_color="#477e9d", row="all", col=1,
    )
    fig.add_annotation(
        x=1967, y=92, text="written forms approach zero",
        showarrow=False, font=dict(color="#275d8c", size=11), row=1, col=1,
    )
    fig.update_layout(**_layout(
        height=650,
        margin=dict(l=62, r=24, t=72, b=55),
        legend=dict(orientation="h", y=1.09, x=0),
    ))
    fig.update_xaxes(range=[1900, 1980], gridcolor=GRID, linecolor=BASELINE)
    fig.update_yaxes(title_text="% of speeches", range=[0, 100],
                     gridcolor=GRID, linecolor=BASELINE, row=1, col=1)
    fig.update_yaxes(title_text="grade level", rangemode="tozero",
                     gridcolor=GRID, linecolor=BASELINE, row=2, col=1)
    return fig


def broadcast_takeaway(
    stats: pd.DataFrame, speech_annotations: pd.DataFrame,
) -> str:
    joined = stats.merge(
        speech_annotations[["doc_name", "medium"]],
        on="doc_name", validate="one_to_one",
    )
    joined["decade"] = (joined.year // 10) * 10
    written = (
        joined.assign(written=joined.medium.eq("written_message"))
        .groupby("decade").written.mean() * 100
    )
    return (
        "The durable shift is not simply that broadcast wins. Assigned written "
        f"messages fall from {written.loc[1900]:.1f}% of sampled 1900s speeches "
        f"to {written.loc[1950]:.1f}% in the 1950s and {written.loc[1960]:.1f}% "
        "in the 1960s, then disappear from the sampled 1970s as performed forms "
        "become normal across the mature Cold War chapter."
    )


def fig_platform_signals(
    rates: pd.DataFrame,
    markers: pd.DataFrame | None = None,
    *,
    full_history: bool = False,
) -> go.Figure:
    """Long transition from procedural to promotional/opponent-centered wording."""
    specs = [
        ("Legal / procedural", "mechanism", "#477e70", "diamond"),
        ("Hype / promotional", "hype", "#9b4e50", "circle"),
        ("Opponent-centered wording", "opponents", "#2f7397", "square"),
    ]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[label for label, _, _, _ in specs],
        vertical_spacing=.10,
    )
    start = 1789 if full_history else 2001
    frame = rates.loc[start:2026]
    endpoint = markers[markers.year.eq(2026)] if markers is not None else None
    for index, (label, column, color, symbol) in enumerate(specs):
        custom = np.full((len(frame), 1), "", dtype=object)
        if endpoint is not None and 2026 in frame.index:
            raw = endpoint[column].sum() / endpoint.n_words.sum() * 10_000
            position = list(frame.index).index(2026)
            custom[position, 0] = (
                f"<br>raw 2026 rate {raw:.2f}"
                f"<br>{len(endpoint)} speeches · {int(endpoint.n_words.sum()):,} words"
                "<br>through April 2026 · provisional"
            )
        fig.add_trace(go.Scatter(
            x=frame.index, y=frame[column], mode="lines+markers",
            line=dict(color=color, width=2.4),
            marker=dict(
                color=color, size=np.where(frame.index == 2026, 11, 4.5),
                symbol=np.where(frame.index == 2026, f"{symbol}-open", symbol),
                line=dict(color=color, width=1),
            ),
            customdata=custom,
            showlegend=False,
            hovertemplate=(
                "%{x}<br>centered five-year rate %{y:.2f} per 10,000"
                "%{customdata[0]}<extra>" + label + "</extra>"
            ),
        ), row=index + 1, col=1)
        fig.add_vline(
            x=2017, line_color="#6f4828", line_dash="dot", line_width=1,
            row=index + 1, col=1,
        )
    fig.add_annotation(
        x=2017, y=1.06, xref="x3", yref="paper",
        text="2017 · intensification boundary, not an origin or cause",
        showarrow=False, font=dict(size=11, color="#6f4828"),
    )
    fig.add_annotation(
        x=2026, y=.01, xref="x3", yref="paper",
        text="through Apr. 2026 · provisional", showarrow=True,
        ax=-90, ay=-26, font=dict(size=10, color="#9b4e50"),
        bgcolor="#fff9ef", bordercolor="#9b4e50",
    )
    fig.update_layout(**_layout(
        height=720, showlegend=False,
        title=dict(
            text=(
                "Full history · same rates and definitions"
                if full_history else
                "Era view · the approach to and continuation beyond 2017"
            ),
            font=dict(size=15),
        ),
        margin=dict(l=64, r=24, t=88, b=58),
    ))
    fig.update_xaxes(range=[start, 2026], gridcolor=GRID, linecolor=BASELINE)
    fig.update_xaxes(title_text=f"{start}–2026", row=3, col=1)
    fig.update_yaxes(title_text="per 10k", rangemode="tozero",
                     gridcolor=GRID, linecolor=BASELINE)
    return fig


def platform_takeaway(rates: pd.DataFrame, markers: pd.DataFrame) -> str:
    endpoint = markers[markers.year.eq(2026)]
    raw = endpoint.opponents.sum() / endpoint.n_words.sum() * 10_000
    smoothed = float(rates.loc[2026, "opponents"])
    return (
        "The 2017 boundary intensifies a longer transition from legal/procedural "
        "toward promotional and opponent-centered wording; it does not originate it. "
        f"The provisional 2026 opponent-centered endpoint is {raw:.2f} per 10,000 "
        f"words raw and {smoothed:.2f} in the centered display, based on "
        f"{len(endpoint)} speeches and {int(endpoint.n_words.sum()):,} words through April."
    )


def fig_rhetorical_weather(markers: pd.DataFrame) -> go.Figure:
    """Median-defined intensity regions with a direct chronological trail."""
    wide = _weather_rows(markers)
    labels = list(wide["era"])
    short = [
        "Establishing", "War/expansion", "Sectional crisis", "Admin-industrial",
        "Reform/collapse", "New Deal/WWII", "Mature Cold War",
        "Conservative/always-on", "Platform era",
    ]
    colors = [
        "#8b6c42", "#aa7d47", "#9b4e50", "#7f6b57", "#b16b3e",
        "#8f4e56", "#477e9d", "#6d5a91", "#c43d4d",
    ]
    symbols = [
        "circle", "square", "diamond", "triangle-up", "triangle-down",
        "star", "hexagon", "cross", "star-diamond",
    ]
    fig = go.Figure()
    x_cut = float(wide.hype.median())
    y_cut = float(wide.doom.median())
    x_max = float(wide.hype.max()) * 1.16
    y_max = float(wide.doom.max()) * 1.18
    regions = [
        (0, x_cut, 0, y_cut, "Below era medians", "rgba(91,112,104,.07)"),
        (x_cut, x_max, 0, y_cut, "Higher hype", "rgba(177,107,62,.08)"),
        (0, x_cut, y_cut, y_max, "Higher doom", "rgba(109,90,145,.08)"),
        (x_cut, x_max, y_cut, y_max, "Higher on both", "rgba(155,78,80,.07)"),
    ]
    for x0, x1, y0, y1, _label, color in regions:
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=color, line=dict(width=0), layer="below",
        )
    for x, y, label in [
        (x_cut * .45, y_cut * .25, "Below era medians"),
        ((x_cut + x_max) / 2, y_cut * .25, "Higher hype"),
        (x_cut * .45, (y_cut + y_max) / 2, "Higher doom"),
        ((x_cut + x_max) / 2, (y_cut + y_max) / 2, "Higher on both"),
    ]:
        fig.add_annotation(
            x=x, y=y, text=label, showarrow=False,
            font=dict(size=10, color="#766f68"),
        )
    hype_delta = wide.hype.diff()
    doom_delta = wide.doom.diff()
    fig.add_trace(go.Scatter(
        x=wide.hype, y=wide.doom, mode="lines+markers+text",
        line=dict(color="#777d80", width=2.2, dash="dot"),
        text=short, textposition=[
            "bottom left", "top center", "top center", "bottom center",
            "top left", "top center", "bottom center", "top left", "top center",
        ],
        textfont=dict(size=10.5, color=INK2),
        marker=dict(
            color=colors, symbol=symbols, size=[15] + [13] * 7 + [19],
            line=dict(color="white", width=[2.5] + [1] * 7 + [3]),
        ),
        customdata=np.stack([
            labels, wide["years"],
            hype_delta.fillna(0), doom_delta.fillna(0),
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>story era %{customdata[1]}"
            "<br>hype %{x:.3f} per 10,000"
            "<br>doom %{y:.3f} per 10,000"
            "<br>from previous era: hype %{customdata[2]:+.3f}, "
            "doom %{customdata[3]:+.3f}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_annotation(
        x=float(wide.iloc[0].hype), y=float(wide.iloc[0].doom),
        text="start", showarrow=True, ax=-38, ay=38,
    )
    fig.add_annotation(
        x=float(wide.iloc[-1].hype), y=float(wide.iloc[-1].doom),
        text="present · hype separates sharply", showarrow=True, ax=-115, ay=-48,
        bgcolor="#fff9ef", bordercolor="#9b4e50",
    )
    fig.add_vline(x=x_cut, line_color="#a8a29a", line_dash="dash", line_width=1)
    fig.add_hline(y=y_cut, line_color="#a8a29a", line_dash="dash", line_width=1)
    fig.update_layout(**_layout(
        height=600,
        title=dict(text="Hype and doom move separately · reviewed story eras",
                   font=dict(size=15)),
        margin=dict(l=72, r=35, t=70, b=65),
        xaxis=dict(title="hype words per 10,000",
                   range=[0, x_max], gridcolor=GRID, linecolor=BASELINE),
        yaxis=dict(title="doom words per 10,000",
                   range=[0, y_max], gridcolor=GRID, linecolor=BASELINE),
    ))
    return fig


def _weather_rows(markers: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "n_words", "hype", "doom"}
    missing = sorted(required - set(markers.columns))
    if missing:
        raise ValueError(f"rhetorical weather markers are missing: {missing}")
    rows = []
    for label, start, end in era_profiles.STORY_ERAS:
        subset = markers[markers["year"].between(start, end)]
        denominator = int(subset["n_words"].sum())
        if denominator <= 0:
            raise ValueError(f"rhetorical weather era {label!r} has no words")
        rows.append({
            "period": start,
            "years": f"{start}–{end}",
            "era": label,
            "hype": float(subset["hype"].sum() / denominator * 10_000),
            "doom": float(subset["doom"].sum() / denominator * 10_000),
        })
    return pd.DataFrame(rows).set_index("period")


def weather_takeaway(markers: pd.DataFrame) -> str:
    wide = _weather_rows(markers)
    current = wide.iloc[-1]
    prior = wide.iloc[:-1]
    doom_peak = prior.loc[prior.doom.idxmax()]
    return (
        "The present era is not uniquely doom-heavy; it is uniquely hype-heavy. "
        f"Hype reaches {current.hype:.1f} per 10,000 words versus "
        f"{prior.hype.max():.1f} in the next-highest era, while doom is "
        f"{current.doom:.1f} versus a higher {doom_peak.doom:.1f} in "
        f"{doom_peak.era}."
    )


def _weather_table_html(markers: pd.DataFrame) -> str:
    wide = _weather_rows(markers)
    rows = "".join(
        f"<tr><th scope=\"row\">{html.escape(row.era)}</th>"
        f"<td>{html.escape(row.years)}</td><td>{row.hype:.3f}</td>"
        f"<td>{row.doom:.3f}</td></tr>"
        for _period, row in wide.iterrows()
    )
    return f"""<details class="weather-table"><summary>Text alternative · all era rates</summary>
<div class="table-scroll"><table><thead><tr><th>Era</th><th>Story years</th>
<th>Hype per 10,000</th><th>Doom per 10,000</th></tr></thead>
<tbody>{rows}</tbody></table></div></details>"""


# Verified verbatim from the corpus - the certainty finding, in two sentences.
QUOTE_PAIRS = {
    "certainty": [
        ("Offensive operations have therefore been directed; to be conducted, "
         "<strong>however</strong>, as consistently as possible with the dictates of "
         "humanity.", "George Washington, Annual Message, 1791"),
        ("The American dream is unstoppable, and our country is on the verge of a "
         "comeback, the likes of which the world has <strong>never</strong> witnessed "
         "and perhaps will <strong>never</strong> witness again.",
         "Donald Trump, Address to Congress, 2025"),
    ],
    "recovery_mobilization": [
        ("This Nation asks for <strong>action, and action now</strong>.",
         "Franklin D. Roosevelt, First Inaugural Address, 1933"),
        ("We are now in this war. We are all in it—<strong>all the way</strong>.",
         "Franklin D. Roosevelt, Fireside Chat on the War with Japan, 1941"),
    ],
}

# key, chapter (eyebrow, only where a new chapter starts), title, prose
SECTIONS = [
    ("written_republic", "1789–1808 · Establishing the republic",
     "A constitutional state has to become a country",
     "The first three presidencies define sovereignty, construct federal capacity, and "
     "extend national authority across Native lands. Madison's accession starts the next "
     "calendar chapter; Jefferson's single April 1809 message remains with the presidency "
     "and story regime it closes."),
    ("expansion_conflict", "1809–1849 · The continental republic",
     "Diplomacy links war, state-building, and continental conquest",
     "Madison's accession and the War of 1812 open the chapter. Treaties and diplomacy "
     "remain connective tissue as the presidency turns toward public finance, banking, "
     "federal authority, Native removal, and continental conquest."),
    ("union_crisis", "1850–1868 · Sectional crisis and constitutional rupture",
     "Union breaks; emancipation changes the Constitution",
     "Sectional conflict overwhelms compromise, civil war preserves the Union, and "
     "emancipation turns abolition into a federal war aim. The Reconstruction "
     "Amendments begin to redefine citizenship and national authority before Grant's "
     "accession opens the next presidential speech regime."),
    ("procedural_presidency", "1869–1912 · Reconstruction, administration, and industrial power",
     "Reconstruction continues inside a longer administrative-industrial order",
     "Grant's accession supplies the boundary; it does not declare Reconstruction "
     "historically complete. Civil service, regulation, industrial growth, labor "
     "conflict, money, and overseas power widen the administrative and economic scope "
     "of presidential speech."),
    ("progressive_transform", "1913–1932 · Reform, world war, and collapse",
     "Three historical windows show an agenda changing hands",
     "Progressive reform becomes federal administration, World War I enlarges national "
     "mobilization and diplomacy, and the postwar return to conservative government is "
     "overtaken by financial collapse and depression."),
    ("recovery_mobilization", "1933–1952 · New Deal, world war, and postwar settlement",
     "Governing moves from recovery to mobilization to settlement",
     "The federal government assumes new responsibility for recovery and social "
     "security, World War II reorganizes the country for mass mobilization, and the "
     "Truman years turn wartime power toward a durable postwar international order."),
    ("broadcast_presidency", "1953–1980 · Mature Cold War and broadcast presidency",
     "Performance becomes an ordinary condition of the presidency",
     "Cold War management, nuclear danger, civil rights, Vietnam, Watergate, and "
     "economic disorder place the presidency at the center of broadcast national life. "
     "The focused era begins with Eisenhower and ends with Carter."),
    ("always_on", "1981–2016 · Conservative turn and the always-on presidency",
     "A political turn precedes a transformed communications environment",
     "Reagan's accession opens a conservative governing turn before the Cold War ends. "
     "Globalization and terrorism reorder the agenda while cable, permanent "
     "campaigning, and digital platforms intensify pressure and reshape the "
     "expectations and tempo of presidential communication."),
    ("platform_dominance", "2017–2026 · Platform-era intensification",
     "A continuation, not an origin",
     "The boundary is an intensification, not a birth or origin. Social platforms did not "
     "suddenly originate in 2017, but the presidency now operates inside a platform-shaped "
     "environment of fractured audiences, continuous conflict, and direct communication. "
     "The corpus endpoint is provisional through April 2026 and excludes most posts, rallies, "
     "interviews, and other off-record communication."),
    ("synthesis", "Synthesis · Change, continuity, and uncertainty",
     "What can presidential speech say about America in aggregate?",
     "The chronology ends and six through-lines begin: agenda breadth, communication form, "
     "governing voice, adversaries, temporal appeal, and emotional register. Confirmatory, "
     "exploratory, and descriptive evidence remain visibly separate throughout."),
]


def _quotes_html(key: str) -> str:
    pair = QUOTE_PAIRS.get(key)
    if not pair:
        return ""
    blocks = "\n".join(
        f'<blockquote><p>“{text}”</p><cite>{who}</cite></blockquote>'
        for text, who in pair
    )
    return f'<div class="quotepair">{blocks}</div>'


def build_html(
    figs: dict[str, go.Figure],
    stats_line: dict,
    bodies: dict[str, str],
    inline: bool,
    prose_extra: dict[str, str] | None = None,
    era_profile_sections: dict[str, str] | None = None,
    page_kind: str = "combined",
) -> str:
    """Build the chronological story, standalone summary, or legacy combined page."""
    from plotly.offline import get_plotlyjs

    if page_kind == "story":
        page_sections = SECTIONS[:9]
    elif page_kind == "summary":
        page_sections = SECTIONS[9:]
    elif page_kind == "combined":
        page_sections = SECTIONS
    else:
        raise ValueError(f"unknown page kind: {page_kind}")

    if not figs:
        plotly_src = ""
    elif inline:
        plotly_src = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_src = (
            '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" '
            'charset="utf-8"></script>'
        )
    fig_json = {k: json.loads(pio.to_json(f)) for k, f in figs.items()}

    # Explicit pixel heights keep plotly's percent-sized inner containers from
    # collapsing when its responsive handler re-renders after a window resize.
    parts = []
    story_index = 0
    chronology_ended = False
    for key, chapter, title, prose in page_sections:
        # A section whose figure could not be built is omitted entirely rather
        # than rendered as an empty box with prose describing a chart that is
        # not there.
        # It WARNS rather than dropping quietly: a `SECTIONS` entry that is
        # prose-only, or whose fig key is misspelled, would otherwise disappear
        # from the page with no signal.
        #
        # It does NOT catch the pre-existing dead `FINDINGS` anchors, and an
        # earlier version of this comment wrongly claimed it did. `records`,
        # `kinships` and `naming` are absent from `SECTIONS` ENTIRELY, so this
        # loop never visits them and the warning fires zero times on a real
        # build. Catching those needs a check on the rendered ids AFTER the
        # loop, which belongs to the backlog item that owns them.
        if key not in figs and key not in bodies:
            warnings.warn(
                f"section {key!r} has neither a figure nor a body and was "
                f"omitted from index.html; any FINDINGS card anchored to "
                f"#{key} is now a dead link.",
                RuntimeWarning, stacklevel=2,
            )
            continue
        story_index += 1
        if prose_extra and prose_extra.get(key):
            prose = f"{prose} {prose_extra[key]}"
        if chapter and "·" in chapter:
            story_range, story_era = (
                part.strip() for part in chapter.split("·", 1)
            )
        elif chapter:
            story_range, story_era = chapter, chapter
        else:
            story_range, story_era = "Continuity", "Continuity"
        if (
            page_kind == "combined"
            and chapter
            and chapter.startswith("Synthesis")
            and not chronology_ended
        ):
            parts.append(
                '<aside class="chronology-end" role="note">'
                '<strong>The chronological story ends here.</strong>'
                '<span>Synthesis begins: the same evidence is now read across eras.</span>'
                '</aside>'
            )
            chronology_ended = True
        if chapter:
            parts.append(f'<div class="eyebrow">{chapter}</div>')
        if key in bodies:
            body = bodies[key]
        else:
            body = (f'<div class="chart-scroll"><div class="chart" data-fig="{key}"'
                    f' style="height:{figs[key].layout.height}px"></div></div>')
        lesson = ""
        if key in metrics.CHART_METRICS and key not in bodies:
            metric_name = sorted(metrics.CHART_METRICS[key])[0]
            lesson = _chart_evidence(metric_name)
        quotes = "" if page_kind == "story" else _quotes_html(key)
        section_html = f"""<section id="{key}"
  data-story-range="{html.escape(story_range, quote=True)}"
  data-story-era="{html.escape(story_era, quote=True)}"
  data-story-title="{html.escape(title, quote=True)}">
  <h2>{title}</h2>
  <p class="section-deck">{prose}</p>
  {quotes}
  {(era_profile_sections or {}).get(key, "")}
  {body}
  {lesson}
</section>"""
        parts.append(re.sub(r"(?m)^[ \t]+$", "", section_html))
    sections_html = "\n".join(parts)

    era_route_items = []
    for key, chapter, _, _ in page_sections:
        if not chapter:
            continue
        if "·" in chapter:
            marker, label = (part.strip() for part in chapter.split("·", 1))
        else:
            marker, label = "Continuity", chapter
        era_route_items.append(
            f'<a href="#{key}"><span>{html.escape(marker)}</span>'
            f'{html.escape(label)}</a>'
        )
    era_route_html = "".join(era_route_items)

    corpus_line = (
        f"{stats_line['speeches']:,} speeches · {stats_line['words'] / 1e6:.1f}M words · "
        f"{stats_line['presidents']} presidents · {stats_line['start']} - April "
        f"{stats_line['end']}"
    )
    if page_kind == "summary":
        document_title = "America in Summary · Presidential Profiles"
        document_description = (
            "Change, continuity, and uncertainty across the presidential "
            "speech record, 1789-2026."
        )
        header_html = f"""<header class="summary-header">
  <h1>America in Summary</h1>
  <p class="hero-subheadline">What the formal presidential record says—and cannot say—across 1789–2026</p>
  <p class="corpus-line">{corpus_line}</p>
  <nav class="era-route" aria-label="Summary sections">{era_route_html}</nav>
</header>"""
        meter_html = ""
        after_sections = ""
        body_class = ' class="summary-document"'
    else:
        document_title = "America Through Its Presidents · Presidential Profiles"
        document_description = (
            "Interactive analysis of 1,057 presidential speeches, 1789-2026, "
            "from the Miller Center corpus."
        )
        header_html = f"""<header>
  <h1>America Through Its Presidents</h1>
  <p class="hero-subheadline">1789–2026 · A country told through presidential speech</p>
  <p class="corpus-line">{corpus_line}</p>
  <nav class="era-route" aria-label="Story eras">{era_route_html}</nav>
  <a class="scroll-cue" href="#written_republic">Begin the story</a>
</header>"""
        meter_html = """<div class="story-meter" aria-label="Story progress">
  <div class="story-meter-track"><span class="story-meter-fill" id="story-progress"></span></div>
  <div class="story-meter-label" id="story-progress-label">
    <strong id="story-progress-era">1789–1808 · Establishing the republic</strong>
    <span id="story-progress-title">A constitutional state has to become a country</span>
  </div>
</div>"""
        after_sections = (
            '<aside class="chronology-end story-summary-link" role="note">'
            '<strong>The chronological story ends here.</strong>'
            '<span><a href="summary.html">Continue to the Summary →</a></span>'
            '</aside>'
            if page_kind == "story"
            else ""
        )
        body_class = ""
    legacy_summary_redirect = (
        '<script>if (location.hash === "#synthesis") '
        'location.replace("summary.html" + location.hash); '
        'if (location.hash === "#records_appendix") '
        'location.replace("summary.html#summary-topics");</script>'
        if page_kind == "story"
        else ""
    )
    topic_relationship_assets = (
        f'<link rel="stylesheet" href="assets/{summary_topic_network.ASSET_CSS_NAME}">\n'
        f'<script defer src="assets/{summary_topic_network.ASSET_JS_NAME}"></script>'
        if page_kind == "summary"
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{document_title}</title>
<meta name="description" content="{document_description}">
{plotly_src}
{topic_relationship_assets}
<style>
{PAGE_CSS}
  .corpus-line {{ color: var(--muted); font-size: 0.86rem; margin-top: 14px; }}
  .findings {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
               gap: 12px; margin-top: 24px; }}
  .f-card {{ background: var(--surface); border: 1px solid var(--border);
             border-radius: 12px; padding: 16px 18px; text-decoration: none;
             color: var(--ink); transition: border-color 0.15s; }}
  .f-card:hover {{ border-color: var(--muted); }}
  .f-stat {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -0.01em; }}
  .f-text {{ color: var(--ink2); font-size: 0.9rem; margin-top: 7px; line-height: 1.45; }}
  .eyebrow {{ margin-top: 72px; color: #6f4828; font-size: clamp(.9rem,1.8vw,1.12rem);
              font-weight: 780; letter-spacing: 0.055em; text-transform: uppercase;
              padding:14px 16px;background:#f4eadc;border-left:5px solid #8b5e34;
              border-radius:0 12px 12px 0; }}
  .eyebrow + section {{ margin-top: 16px; }}
  .quotepair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
                margin: 0 0 16px; }}
  @media (max-width: 820px) {{ .quotepair {{ grid-template-columns: 1fr; }} }}
  .quotepair blockquote {{ background: var(--surface); border: 1px solid var(--border);
                           border-left: 3px solid var(--muted); border-radius: 10px;
                           padding: 14px 18px; }}
  .quotepair p {{ color: var(--ink); font-size: 0.98rem; margin: 0; max-width: none; }}
  .quotepair strong {{ font-weight: 750; }}
  .quotepair cite {{ display: block; color: var(--muted); font-style: normal;
                     font-size: 0.82rem; margin-top: 10px; }}
  .records {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
              gap: 12px; }}
  .r-card {{ background: var(--surface); border: 1px solid var(--border);
             border-radius: 12px; padding: 16px 18px; }}
  .r-label {{ color: var(--muted); font-size: 0.78rem; font-weight: 650;
              letter-spacing: 0.07em; text-transform: uppercase; }}
  .r-value {{ font-size: 1.6rem; font-weight: 700; margin-top: 6px; }}
  .r-unit {{ font-size: 0.8rem; font-weight: 500; color: var(--muted); }}
  .kinships {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
               gap: 12px; }}
  .k-card {{ background: var(--surface); border: 1px solid var(--border);
             border-radius: 12px; padding: 16px 18px; text-align: center; }}
  .k-faces {{ display: flex; align-items: center; justify-content: center; gap: 4px; }}
  .k-faces img {{ width: 52px; height: 52px; border-radius: 50%; }}
  .k-link {{ width: 26px; border-top: 2px dotted var(--muted); }}
  .k-names {{ font-weight: 650; font-size: 0.92rem; margin-top: 10px; }}
  .k-meta {{ color: var(--muted); font-size: 0.8rem; margin-top: 4px; }}
  .similarity-context {{ background:var(--surface);border:1px solid var(--border);border-radius:12px;
                         padding:16px 18px;margin-bottom:14px; }}
  .similarity-context p {{ max-width:none;margin-bottom:26px; }}
  .similarity-scale {{ position:relative;height:4px;background:linear-gradient(90deg,#9d6f72,#dedede 50%,#477e9d);
                       border-radius:99px;margin:30px 24px 22px; }}
  .similarity-scale span {{ position:absolute;top:10px;transform:translateX(-50%);font-size:.7rem;
                            color:var(--muted);white-space:nowrap; }}
  .similarity-scale .observed {{ top:-24px;color:#275d8c;font-weight:700; }}
  .feature-pairs {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:10px; }}
  .feature-pair {{ display:grid;grid-template-columns:auto 1fr;align-items:center;gap:14px;
                   background:var(--surface);border:1px solid var(--border);border-radius:12px;
                   padding:14px; }}
  .feature-pair .k-faces img {{ width:42px;height:42px; }}
  .feature-pair strong,.feature-pair .feature-label {{ display:block; }}
  .feature-label {{ color:var(--muted);font-size:.72rem;text-transform:uppercase;
                    letter-spacing:.07em;margin-bottom:3px; }}
  .feature-pair p {{ margin:3px 0 0;font-size:.78rem; }}
  .feature-score {{ font:750 1.55rem/1 Georgia,serif;color:#275d8c;margin-top:8px; }}
  .feature-score-track {{ width:100%;height:5px;background:var(--grid);border-radius:99px;
                          overflow:hidden;margin:6px 0 8px; }}
  .feature-score-track span {{ display:block;height:100%;background:linear-gradient(90deg,#b9cfde,#275d8c); }}
  .radar-title {{ margin-top:30px; }}
  .eras {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  @media (max-width: 820px) {{ .eras {{ grid-template-columns: 1fr; }} }}
  .e-card {{ background: var(--surface); border: 1px solid var(--border);
             border-radius: 12px; padding: 16px 18px; }}
  .e-head {{ display: flex; align-items: baseline; gap: 10px; margin-bottom: 10px; }}
  .e-name {{ font-weight: 700; }}
  .e-years {{ color: var(--muted); font-size: 0.8rem; }}
  .terms {{ display: flex; flex-wrap: wrap; gap: 7px; }}
  .term {{ background: var(--page); border: 1px solid var(--border);
           border-radius: 8px; padding: 4px 10px; font-size: 0.88rem; }}
  .ai-bridge {{ background:#eef6ff; border:1px solid #c9def3; border-radius:14px;
                padding:18px; display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
  .ai-bridge-stat {{ background:rgba(255,255,255,.68); border-radius:10px; padding:12px; }}
  .ai-bridge-stat strong,.ai-bridge-stat span {{ display:block; }}
  .ai-bridge-stat strong {{ font-size:1.35rem; }} .ai-bridge-stat span {{ color:var(--muted); font-size:.74rem; }}
  .ai-bridge p {{ grid-column:1/-1; margin:2px 0 0; max-width:none; }}
  .ai-bridge a {{ color:#275d8c; font-weight:700; }}
  .audit-grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px; }}
  .audit-card {{ background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px; }}
  .chronology-end {{ margin:88px 0 20px;padding:26px;background:#1f2d36;color:white;
                     border-radius:16px;display:grid;gap:7px; }}
  .chronology-end strong {{ font:700 clamp(1.35rem,3vw,2rem)/1.1 Georgia,serif; }}
  .chronology-end span {{ color:#d6e1e7; }}
  .chronology-end a {{ color:white;font-weight:750; }}
  .summary-document .eyebrow:first-child {{ margin-top:28px; }}
  .summary-document .story-section {{ margin-top:24px; }}
  .summary-document main > .eyebrow {{ display:none; }}
  .summary-document main > .story-section > h2,
  .summary-document main > .story-section > .section-deck {{ display:none; }}
  .summary-document main {{ padding-bottom:0; }}
  .summary-document .summary-topic-section {{ margin-bottom:0; }}
  .summary-document footer {{ margin-top:8px; }}
  .summary-thesis {{ margin:14px 0 20px;padding:clamp(20px,4vw,34px);
                     background:#1f2d36;color:white;border-radius:18px; }}
  .summary-thesis span {{ color:#efc58f;font-size:.72rem;font-weight:850;
                          letter-spacing:.1em;text-transform:uppercase; }}
  .summary-thesis p {{ max-width:900px;margin:10px 0 0;color:#f2f5f6;
                       font:650 clamp(1.2rem,2.8vw,2rem)/1.32 Georgia,serif; }}
  .summary-chapter-route {{ position:sticky;top:var(--global-nav-height,48px);z-index:12;
                            display:flex;gap:7px;overflow-x:auto;margin:0 0 42px;padding:9px;
                            background:color-mix(in srgb,var(--page) 94%,transparent);
                            backdrop-filter:blur(10px);border:1px solid var(--border);
                            border-radius:12px;scrollbar-width:thin; }}
  .summary-chapter-route a {{ flex:0 0 auto;padding:7px 10px;color:var(--ink2);
                              text-decoration:none;font-size:.72rem;font-weight:750;
                              border-radius:7px; }}
  .summary-chapter-route a:hover,.summary-chapter-route a:focus-visible {{ background:#e9ded0; }}
  .summary-chapter {{ margin:0 0 90px;padding-top:34px;border-top:1px solid var(--border);
                      scroll-margin-top:110px; }}
  .summary-chapter > h2 {{ max-width:910px;margin:4px 0 10px;
                           font-size:clamp(2rem,5vw,3.9rem);line-height:1.02;
                           letter-spacing:-.035em; }}
  .summary-chapter > h3 {{ margin-top:42px;font-size:clamp(1.35rem,2.6vw,2rem); }}
  .summary-chapter-no {{ margin:0;color:#8b5e34;font-size:.72rem;font-weight:850;
                         letter-spacing:.11em;text-transform:uppercase; }}
  .summary-lede {{ max-width:890px;margin:0 0 24px;color:var(--ink2);
                   font-size:clamp(1rem,1.8vw,1.22rem);line-height:1.65; }}
  .summary-receipt-grid {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
                           gap:10px;margin:22px 0 14px; }}
  .summary-receipt-grid article {{ padding:16px;background:#edf5f8;border:1px solid #c7dce2;
                                   border-radius:12px; }}
  .summary-receipt-grid span,.summary-receipt-grid strong,
  .summary-receipt-grid small {{ display:block; }}
  .summary-receipt-grid span {{ color:#315f78;font-size:.68rem;font-weight:850;
                                letter-spacing:.07em;text-transform:uppercase; }}
  .summary-receipt-grid strong {{ margin:5px 0 3px;font:750 clamp(1.35rem,3vw,2rem)/1 Georgia,serif; }}
  .summary-receipt-grid small {{ color:var(--muted);line-height:1.4; }}
  .summary-shift-grid {{ display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
                         gap:8px;margin:18px 0 16px; }}
  .summary-shift-grid article {{ padding:13px 14px;background:var(--surface);
                                 border-top:5px solid #315f78;border-radius:10px;
                                 box-shadow:0 0 0 1px var(--border); }}
  .summary-shift-grid article:nth-child(even) {{ border-top-color:#b16b3e; }}
  .summary-shift-grid span,.summary-shift-grid strong {{ display:block; }}
  .summary-shift-grid span {{ color:var(--muted);font-size:.67rem;font-weight:780;
                              text-transform:uppercase;letter-spacing:.04em; }}
  .summary-shift-grid strong {{ margin-top:5px;font:750 clamp(1.5rem,3vw,2.25rem)/1 Georgia,serif; }}
  .summary-temporal-portrait {{ margin:24px 0 16px;border:1px solid var(--border);
                                border-top:4px solid #315f78;border-radius:14px;
                                background:var(--surface);overflow:hidden; }}
  .temporal-portrait-heading {{ display:block;padding:18px 20px 12px; }}
  .temporal-portrait-heading span {{ color:#8b5e34;font-size:.59rem;font-weight:850;
                                     letter-spacing:.09em;text-transform:uppercase; }}
  .temporal-portrait-heading h3 {{ margin:4px 0 0;font:750 clamp(1.2rem,2.4vw,1.62rem)/1.08 Georgia,serif; }}
  .temporal-portrait-heading p {{ max-width:660px;margin:6px 0 0;color:var(--muted);
                                  font-size:.72rem;line-height:1.45; }}
  .temporal-denominator {{ margin:0;padding:0 20px 12px;color:var(--muted);
                           font-size:.59rem;font-weight:720;letter-spacing:.02em; }}
  .summary-temporal-explorer {{ padding:0 20px 4px; }}
  .summary-temporal-explorer .chart-scroll {{ margin-inline:-20px;padding:4px 2px 0;
                                              border-width:1px 0 0;border-radius:0;
                                              background:#fffdf9;overscroll-behavior-inline:contain; }}
  .summary-temporal-explorer .chart {{ width:{SUMMARY_TEMPORAL_CHART_WIDTH_PX:.0f}px;
                                      min-width:{SUMMARY_TEMPORAL_CHART_WIDTH_PX:.0f}px;
                                      max-width:{SUMMARY_TEMPORAL_CHART_WIDTH_PX:.0f}px;
                                      margin-inline:auto; }}
  .summary-evidence-detail {{ margin:10px 0 20px;padding:10px 12px;
                              background:var(--surface);border:1px solid var(--border);
                              border-radius:10px; }}
  .summary-evidence-detail > summary {{ cursor:pointer;color:var(--ink2);
                                       font-size:.78rem;font-weight:780; }}
  .summary-communication-bars {{ display:grid;gap:30px;margin:24px 0 16px; }}
  .summary-communication-panel {{ min-width:0;margin:0;padding:20px;
                                  border:1px solid var(--border);border-radius:16px;
                                  background:#faf7f1;box-shadow:0 10px 28px rgba(48,42,31,.045); }}
  .summary-communication-panel > header {{ display:grid;grid-template-columns:1fr auto;
                                          align-items:end;gap:12px;max-width:none;margin:0;
                                          padding:0 2px 13px;border-bottom:1px solid var(--grid); }}
  .summary-communication-panel > header p {{ margin:0;color:#8b5e34;font-size:.67rem;
                                             font-weight:850;letter-spacing:.09em;
                                             text-transform:uppercase; }}
  .summary-communication-panel > header h3 {{ margin:4px 0 0;
                                              font:680 clamp(1.35rem,2.6vw,2rem)/1.12 Georgia,serif; }}
  .summary-communication-panel > header > strong {{ align-self:center;padding:7px 10px;
                                                    border:1px solid var(--border);border-radius:999px;
                                                    background:#fffdf9;color:var(--ink2);
                                                    font-size:.67rem;white-space:nowrap; }}
  .summary-communication-panel > header > strong span {{ margin:0 4px;color:#8b5e34; }}
  .summary-communication-key {{ display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
                                gap:7px;margin:12px 2px 0; }}
  .summary-communication-key span {{ display:flex;align-items:center;gap:7px;min-width:0;
                                     padding:7px 8px;border:1px solid var(--border);
                                     border-radius:8px;background:#fffdf9;color:var(--ink2);
                                     font-size:.63rem;font-weight:760;line-height:1.2; }}
  .summary-communication-key span::before {{ content:"";flex:0 0 auto;width:8px;height:20px;
                                             border-radius:3px;background:var(--key-color); }}
  .summary-communication-panel .chart-scroll {{ margin-top:12px;padding:4px 2px 0;
                                                border:0;border-radius:10px;background:#fffdf9;
                                                overscroll-behavior-inline:contain; }}
  .summary-communication-panel .chart {{ min-width:860px; }}
  .summary-communication-panel .inspector-row {{ margin-bottom:0; }}
  .summary-register-bridge {{ margin:30px 0 14px;padding:20px 22px 18px;
                              border:1px solid #d7c7b2;border-radius:16px;
                              background:linear-gradient(135deg,#f4ede2 0%,#faf7f1 58%,#eef4f5 100%);
                              box-shadow:0 10px 28px rgba(48,42,31,.045); }}
  .summary-register-kicker {{ margin:0;color:#8b5e34;font-size:.65rem;font-weight:850;
                              letter-spacing:.1em;text-transform:uppercase; }}
  .summary-register-bridge h3 {{ margin:5px 0 8px;
                                 font:680 clamp(1.4rem,2.8vw,2.15rem)/1.1 Georgia,serif; }}
  .summary-register-bridge > p:not(.summary-register-kicker) {{ max-width:870px;margin:0;
                                                                 color:var(--ink2);font-size:.86rem;
                                                                 line-height:1.55; }}
  .summary-register-sequence {{ list-style:none;display:grid;
                                grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;
                                margin:17px 0 0;padding:0; }}
  .summary-register-sequence li {{ position:relative;display:grid;gap:4px;align-content:center;
                                   min-height:72px;padding:11px 13px;border:1px solid var(--border);
                                   border-radius:10px;background:rgba(255,253,249,.88); }}
  .summary-register-sequence li + li::before {{ content:"→";position:absolute;left:-19px;top:50%;
                                                color:#8b5e34;font-size:1.05rem;font-weight:850;
                                                transform:translateY(-50%); }}
  .summary-register-sequence span {{ color:#8b5e34;font-size:.57rem;font-weight:850;
                                     letter-spacing:.09em;text-transform:uppercase; }}
  .summary-register-sequence strong {{ color:var(--ink2);font:680 .83rem/1.28 Georgia,serif; }}
  .era-select-bar {{ display:grid;grid-template-columns:minmax(240px,auto) minmax(290px,360px) minmax(0,1fr);
                     align-items:center;gap:9px 12px;margin:0 0 10px;padding:11px 13px;
                     border:1px solid var(--border);border-radius:11px;background:var(--surface); }}
  .register-view-switch {{ display:flex;align-items:center;gap:4px;min-width:0; }}
  .register-view-switch > span {{ margin-right:4px;color:var(--ink2);font-size:.76rem;
                                  font-weight:800; }}
  .register-view-switch button {{ min-height:44px;padding:8px 12px;border:1px solid #b9aa97;
                                  background:#fffdf9;color:var(--ink2);font-size:.74rem;
                                  font-weight:760;cursor:pointer; }}
  .register-view-switch button:first-of-type {{ border-radius:8px 0 0 8px; }}
  .register-view-switch button:last-of-type {{ margin-left:-5px;border-radius:0 8px 8px 0; }}
  .register-view-switch button[aria-pressed="true"] {{ position:relative;z-index:1;
                                                        border-color:#315f78;background:#315f78;
                                                        color:#fff; }}
  .register-view-switch button:focus-visible {{ position:relative;z-index:2;
                                                 outline:3px solid #d08b45;outline-offset:2px; }}
  .era-select-control {{ display:grid;grid-template-columns:auto minmax(170px,1fr);
                         align-items:center;gap:8px;min-width:0; }}
  .era-select-control label {{ color:var(--ink2);font-size:.76rem;font-weight:800; }}
  .era-select-control select {{ width:100%;min-height:44px;padding:8px 34px 8px 10px;
                            border:1px solid #b9aa97;border-radius:8px;background:#fffdf9;
                            color:var(--ink2);font:700 .76rem/1.2 system-ui;cursor:pointer; }}
  .era-select-control select:disabled {{ cursor:not-allowed;opacity:.62; }}
  .era-select-control select:focus-visible {{ outline:3px solid #d08b45;outline-offset:2px; }}
  .era-select-bar .filter-status {{ margin:0; }}
  .president-era-explorer.is-timeline-view .chart {{ min-width:860px; }}
  .register-chart-scroll:focus-visible {{ outline:3px solid #d08b45;outline-offset:2px; }}
  .summary-register-noscript {{ margin:10px 0;padding:10px 12px;border-left:4px solid #8b5e34;
                                background:#f5efe6;color:var(--ink2);font-size:.78rem; }}
  .conflict-atlas {{ margin:20px 0 16px;border:1px solid var(--border);border-radius:16px;
                     background:#faf7f1;box-shadow:0 12px 32px rgba(48,42,31,.055);
                     overflow:clip; }}
  .conflict-atlas-heading {{ display:grid;grid-template-columns:minmax(0,1fr) auto;
                             align-items:center;gap:14px;padding:16px 18px 10px; }}
  .conflict-atlas-heading p {{ margin:0;color:#8b5e34;font-size:.61rem;font-weight:850;
                               letter-spacing:.1em;text-transform:uppercase; }}
  .conflict-atlas-heading h3 {{ margin:4px 0 0;
                                font:680 clamp(1.35rem,2.5vw,1.9rem)/1.08 Georgia,serif; }}
  .conflict-treatment {{ max-width:340px;padding:7px 9px;border:1px solid #d9c9b4;
                         border-radius:10px;background:#fff9ef; }}
  .conflict-treatment span,.conflict-treatment strong,.conflict-treatment small {{ display:block; }}
  .conflict-treatment span {{ color:#8b5e34;font-size:.58rem;font-weight:850;
                              letter-spacing:.08em;text-transform:uppercase; }}
  .conflict-treatment strong {{ margin-top:2px;color:#5d5144;font-size:.65rem;line-height:1.25; }}
  .conflict-treatment small {{ margin-top:3px;color:var(--muted);font-size:.58rem;line-height:1.3; }}
  .conflict-atlas-deck {{ max-width:960px;margin:0;padding:0 18px 10px;color:var(--ink2);
                          font-size:.75rem;line-height:1.45; }}
  .conflict-atlas-deck span {{ margin-left:5px;color:var(--muted);font-size:.61rem;white-space:nowrap; }}
  .conflict-category-key {{ display:grid;grid-template-columns:repeat(5,minmax(0,1fr));
                            gap:5px;padding:0 18px 10px; }}
  .conflict-category-key span {{ display:flex;align-items:center;gap:6px;padding:5px 7px;
                                 border:1px solid var(--border);border-radius:8px;background:#fffdf9;
                                 color:var(--ink2);font-size:.6rem;font-weight:760; }}
  .conflict-category-key span::before {{ content:"";width:7px;height:14px;flex:0 0 auto;
                                         border-radius:3px;background:var(--conflict-color); }}
  .conflict-column-head,.conflict-president-row {{ display:grid;
    grid-template-columns:minmax(180px,.85fr) minmax(230px,1.15fr) minmax(330px,1.45fr);
    column-gap:12px; }}
  .conflict-column-head {{ position:sticky;top:calc(var(--global-nav-height,48px) + 52px);
                           z-index:8;padding:7px 18px;background:#e8e1d7;color:#695d51;
                           font-size:.54rem;font-weight:850;letter-spacing:.05em;
                           text-transform:uppercase; }}
  .conflict-era {{ margin:0;border-top:1px solid var(--border);background:#fffdf9; }}
  .conflict-era > summary {{ display:grid;grid-template-columns:auto 1fr auto;align-items:baseline;
                             gap:7px;padding:7px 18px;border-left:5px solid var(--era-color);
                             background:#f4efe7;cursor:pointer;list-style:none; }}
  .conflict-era > summary::-webkit-details-marker {{ display:none; }}
  .conflict-era > summary::before {{ content:"▸";grid-column:1;grid-row:1;color:var(--era-color);
                                     font-size:.72rem;transform:rotate(0);transition:transform .15s ease; }}
  .conflict-era[open] > summary::before {{ transform:rotate(90deg); }}
  .conflict-era > summary span {{ grid-column:2;color:var(--ink);font-weight:820;font-size:.7rem; }}
  .conflict-era > summary small {{ grid-column:2;color:var(--muted);font-size:.54rem; }}
  .conflict-era > summary strong {{ grid-column:3;grid-row:1 / span 2;color:var(--muted);
                                    font-size:.55rem;white-space:nowrap; }}
  .conflict-president-row {{ align-items:center;padding:7px 18px;border-top:1px solid var(--grid); }}
  .conflict-president-row:first-of-type {{ border-top:0; }}
  .conflict-president-row:hover {{ background:#fffaf1; }}
  .conflict-president-identity {{ min-width:0;max-width:none;margin:0;padding:0;text-align:left; }}
  .conflict-president-identity a {{ display:flex;align-items:center;gap:7px;color:var(--ink);
                                    text-decoration:none; }}
  .conflict-president-identity a:hover strong {{ text-decoration:underline;text-underline-offset:3px; }}
  .conflict-president-identity img {{ width:30px;height:30px;flex:0 0 auto;border:1px solid #d7cabb;
                                     border-radius:50%;background:#eee5d7;object-fit:cover; }}
  .conflict-president-identity a span,.conflict-president-identity a strong,
  .conflict-president-identity a small {{ display:block;min-width:0; }}
  .conflict-president-identity a strong {{ overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                                           font-size:.7rem;line-height:1.15; }}
  .conflict-president-identity a small {{ margin-top:1px;color:var(--muted);font-size:.52rem; }}
  .conflict-president-identity p {{ margin:2px 0 0;color:var(--muted);font-size:.5rem;line-height:1.2;
                                    white-space:nowrap; }}
  .conflict-support-warning {{ display:inline-block;margin-left:4px;padding:1px 4px;border-radius:4px;
                               background:#f1dfc6;color:#704c27;font-size:.5rem;font-weight:820;
                               text-transform:uppercase;letter-spacing:.03em; }}
  .conflict-mobile-label {{ display:none;margin:0 0 5px;color:#6e6256;font-size:.56rem;
                            font-weight:850;letter-spacing:.055em;text-transform:uppercase; }}
  .conflict-mix-bar {{ display:flex;width:100%;height:28px;overflow:hidden;border:1px solid #d3c7b8;
                       border-radius:7px;background:#ede8e0;box-shadow:inset 0 1px 2px rgba(48,42,31,.08); }}
  .conflict-mix-segment {{ display:flex;flex:0 0 var(--mix-share);min-width:0;align-items:center;
                           justify-content:center;background:var(--mix-color);outline:0;
                           box-shadow:inset -1px 0 rgba(255,255,255,.3); }}
  .conflict-mix-segment:last-child {{ box-shadow:none; }}
  .conflict-mix-segment:focus-visible {{ position:relative;z-index:1;outline:3px solid #d08b45;
                                        outline-offset:-3px; }}
  .conflict-mix-segment span {{ display:flex;align-items:center;gap:2px;color:white;
                                font-size:.59rem;line-height:1;text-shadow:0 1px 2px rgba(0,0,0,.35); }}
  .conflict-mix-segment b {{ font-size:.51rem; }}
  .conflict-na {{ display:flex;align-items:center;justify-content:center;min-height:28px;
                  border:1px dashed #c7baaa;border-radius:9px;background:#f0ece5;color:var(--muted);
                  font-size:.56rem;font-weight:740; }}
  .conflict-rate-grid {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px; }}
  .conflict-rate {{ display:grid;grid-template-columns:minmax(0,1fr) auto;gap:2px 4px;min-width:0; }}
  .conflict-rate > span {{ overflow:hidden;color:#6b6055;font-size:.53rem;font-weight:720;
                           text-overflow:ellipsis;white-space:nowrap; }}
  .conflict-rate > strong {{ color:var(--ink);font-size:.57rem;font-variant-numeric:tabular-nums; }}
  .conflict-rate-track {{ grid-column:1 / -1;height:6px;overflow:hidden;border-radius:99px;
                          background:#e3ded6; }}
  .conflict-rate-track i {{ display:block;width:var(--rate-width);height:100%;border-radius:99px;
                            background:var(--rate-color); }}
  .conflict-rate.is-unavailable {{ opacity:.58; }}
  .conflict-rate.is-unavailable .conflict-rate-track {{ background:repeating-linear-gradient(
    -45deg,#e3ded6,#e3ded6 4px,#eeeae4 4px,#eeeae4 8px); }}
  .conflict-evidence-grid {{ display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:0 0 10px; }}
  .conflict-evidence-grid .chart-evidence {{ margin:0; }}
  .conflict-graph-suite {{ margin:20px 0 16px;border:1px solid var(--border);border-radius:16px;
                           background:#faf7f1;box-shadow:0 12px 32px rgba(48,42,31,.055);
                           overflow:hidden; }}
  .conflict-graph-heading {{ display:grid;grid-template-columns:minmax(0,1fr) auto;
                             align-items:center;gap:14px;max-width:none;margin:0;padding:16px 18px 10px; }}
  .conflict-graph-heading p {{ margin:0;color:#8b5e34;font-size:.61rem;font-weight:850;
                               letter-spacing:.1em;text-transform:uppercase; }}
  .conflict-graph-heading h3 {{ margin:4px 0 0;
                                font:680 clamp(1.35rem,2.5vw,1.9rem)/1.08 Georgia,serif; }}
  .conflict-graph-deck {{ max-width:960px;margin:0;padding:0 18px 12px;color:var(--ink2);
                          font-size:.75rem;line-height:1.45; }}
  .conflict-graph-deck span {{ margin-left:5px;color:var(--muted);font-size:.61rem;white-space:nowrap; }}
  .conflict-question-group {{ padding:18px;border-top:1px solid var(--border);background:#f7f2ea; }}
  .conflict-question-group + .conflict-question-group {{ padding-top:24px; }}
  .conflict-question-heading {{ display:grid;grid-template-columns:34px minmax(0,1fr);gap:10px;
                                align-items:start;margin:0 0 12px; }}
  .conflict-question-heading > span {{ display:grid;place-items:center;width:30px;height:30px;
                                       border-radius:50%;background:#315f78;color:#fffdf9;
                                       font-size:.58rem;font-weight:850;letter-spacing:.04em; }}
  .conflict-question-heading p {{ margin:0;color:#8b5e34;font-size:.56rem;font-weight:850;
                                  letter-spacing:.09em; }}
  .conflict-question-heading h4 {{ margin:2px 0 3px;font:680 1.08rem/1.15 Georgia,serif; }}
  .conflict-question-heading small {{ display:block;color:var(--muted);font-size:.62rem;
                                      line-height:1.38; }}
  .conflict-target-key {{ margin:0 0 10px;padding:0; }}
  .conflict-target-key .conflict-category-key {{ grid-template-columns:repeat(5,minmax(0,1fr));
                                                  gap:4px;padding:0; }}
  .conflict-target-key .conflict-category-key span {{ justify-content:center;padding:4px 3px;
                                                       border:0;background:#f4efe7;font-size:.54rem; }}
  .conflict-target-key .conflict-category-key span::before {{ width:18px;height:3px;
                                                               border-radius:99px; }}
  .conflict-measure-panel {{ min-width:0;margin:0 0 16px;overflow:hidden;border:1px solid var(--border);
                             border-top:4px solid var(--measure-color,#315f78);border-radius:12px;
                             background:#fffdf9;box-shadow:0 7px 20px rgba(48,42,31,.045); }}
  .conflict-measure-panel:last-child {{ margin-bottom:0; }}
  .conflict-measure-panel > header {{ display:grid;grid-template-columns:minmax(0,1fr) auto;
                                      align-items:start;gap:18px;max-width:none;margin:0;
                                      padding:13px 14px 7px;text-align:left; }}
  .conflict-measure-panel > header span {{ color:var(--measure-color);font-size:.5rem;font-weight:850;
                                           letter-spacing:.08em; }}
  .conflict-measure-panel > header h5 {{ margin:2px 0 3px;color:var(--ink);
                                         font:680 1.02rem/1.15 Georgia,serif; }}
  .conflict-measure-panel > header p {{ margin:0;color:var(--ink2);font-size:.62rem;line-height:1.35; }}
  .conflict-measure-panel > header > strong {{ align-self:center;padding:5px 8px;border-radius:6px;
                                               background:#f1ece4;color:var(--ink2);font-size:.58rem;
                                               font-variant-numeric:tabular-nums;white-space:nowrap; }}
  .conflict-denominator {{ margin:0;padding:0 14px 9px;color:var(--muted);font-size:.55rem;
                           font-weight:720;letter-spacing:.025em; }}
  .conflict-line-picker {{ display:flex;align-items:center;gap:8px;flex-wrap:wrap;
                           padding:0 14px 10px; }}
  .conflict-line-picker-label {{ color:var(--ink2);font-size:.63rem;font-weight:820;
                                 letter-spacing:.02em; }}
  .conflict-line-buttons {{ display:flex;align-items:center;gap:4px;flex-wrap:wrap; }}
  .conflict-line-buttons button {{ display:inline-flex;align-items:center;gap:6px;min-height:44px;
                                    padding:6px 9px;border:1px solid #cfc4b5;border-radius:7px;
                                    background:#fffdf9;color:var(--ink2);font:750 .62rem/1 system-ui,sans-serif;
                                    cursor:pointer; }}
  .conflict-line-buttons button:hover {{ border-color:#8d7d69;background:#f8f3eb; }}
  .conflict-line-buttons button:focus-visible {{ outline:3px solid #315f78;outline-offset:2px; }}
  .conflict-line-buttons button[aria-pressed="true"] {{ border-color:#274a5d;background:#e8f0f4;
                                                         color:#17384a;box-shadow:inset 0 0 0 1px #274a5d; }}
  .conflict-line-buttons button span {{ display:block;width:20px;height:3px;border-radius:99px;
                                        background:currentColor; }}
  .conflict-line-buttons [data-line-style="dash"] span {{ background:repeating-linear-gradient(90deg,currentColor 0 7px,transparent 7px 10px); }}
  .conflict-line-buttons [data-line-style="dot"] span {{ background:repeating-linear-gradient(90deg,currentColor 0 3px,transparent 3px 6px); }}
  .conflict-line-buttons [data-line-style="dashdot"] span {{ background:repeating-linear-gradient(90deg,currentColor 0 7px,transparent 7px 9px,currentColor 9px 11px,transparent 11px 14px); }}
  .conflict-line-buttons [data-line-style="longdash"] span {{ background:repeating-linear-gradient(90deg,currentColor 0 10px,transparent 10px 14px); }}
  .conflict-line-nation {{ color:#315f78 !important; }}
  .conflict-line-group {{ color:#4f7557 !important; }}
  .conflict-line-person {{ color:#a24f52 !important; }}
  .conflict-line-institution {{ color:#9a7133 !important; }}
  .conflict-line-other {{ color:#74695f !important; }}
  .conflict-line-status {{ flex:1 1 190px;color:var(--muted);font-size:.57rem;line-height:1.35; }}
  .conflict-target-panel .chart-scroll,
  .conflict-portrait-panel .chart-scroll {{ margin:0;padding:4px 2px 0;border:0;
                                             border-top:1px solid #eee8df;border-radius:0;
                                             background:#fffdf9;overscroll-behavior-inline:contain; }}
  .conflict-target-panel .chart {{ min-width:680px; }}
  .conflict-portrait-panel .chart {{ min-width:640px; }}
  .conflict-category-guide {{ margin:12px;border:1px solid #ddd1c2;border-radius:10px;
                              overflow:hidden;background:#f8f3eb;color:var(--ink2); }}
  .conflict-category-guide > summary {{ box-sizing:border-box;min-height:44px;padding:11px 12px;
                                        color:var(--ink);cursor:pointer;
                                        font:680 .88rem/1.35 Georgia,serif; }}
  .conflict-category-guide > summary::marker {{ color:#7b684f; }}
  .conflict-category-guide > summary:focus-visible {{ outline:3px solid #315f78;
                                                      outline-offset:-3px; }}
  .conflict-category-guide[open] > summary {{ border-bottom:1px solid #ddd1c2; }}
  .conflict-category-guide-body {{ padding:12px; }}
  .conflict-category-guide dl {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));
                                 gap:8px;margin:0; }}
  .conflict-category-guide dl > div {{ padding:8px;border-left:3px solid #cabdad;
                                       background:#fffdf9; }}
  .conflict-category-guide dt {{ color:var(--ink);font-size:.62rem;font-weight:850; }}
  .conflict-category-guide dd {{ margin:3px 0 0;color:var(--ink2);font-size:.58rem;line-height:1.4; }}
  .conflict-category-guide-body > p {{ margin:10px 0 0;color:var(--muted);font-size:.58rem;
                                       line-height:1.45; }}
  .conflict-portrait-border-key {{ display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;
                                   margin:10px 12px;padding:9px 10px;border:1px solid #e0d7ca;
                                   border-radius:9px;background:#fffdf9;color:var(--ink2); }}
  .conflict-portrait-border-key > strong {{ flex:0 0 auto;color:var(--ink);
                                            font-size:.61rem; }}
  .conflict-portrait-border-key > div {{ display:flex;flex:1 1 360px;flex-wrap:wrap;
                                         align-items:center;gap:5px 10px; }}
  .conflict-portrait-border-key span {{ display:inline-flex;align-items:center;gap:4px;
                                        font-size:.56rem;font-weight:760;white-space:nowrap; }}
  .conflict-portrait-border-key i {{ display:block;width:12px;height:12px;box-sizing:border-box;
                                     border:3px solid var(--portrait-border-color);
                                     border-radius:50%;background:#fff; }}
  .conflict-portrait-border-key small {{ flex:1 1 100%;color:var(--muted);
                                         font-size:.54rem;line-height:1.35; }}
  .conflict-target-panel {{ --measure-color:#315f78; }}
  .communication-mosaic-suite {{ display:grid;gap:28px;margin:24px 0 16px; }}
  .communication-mosaic {{ min-width:0;padding:20px;border:1px solid var(--border);
                           border-radius:16px;background:#faf7f1;
                           box-shadow:0 10px 28px rgba(48,42,31,.045); }}
  .communication-mosaic-heading p {{ margin:0;color:#8b5e34;font-size:.67rem;
                                      font-weight:850;letter-spacing:.09em;
                                      text-transform:uppercase; }}
  .communication-mosaic-heading h3 {{ margin:5px 0 6px;
                                       font:680 clamp(1.35rem,2.6vw,2rem)/1.12 Georgia,serif; }}
  .communication-mosaic-heading > span {{ display:block;max-width:820px;color:var(--ink2);
                                          font-size:.8rem;line-height:1.5; }}
  .communication-mosaic-key {{ display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
                                gap:7px;margin:16px 0 10px; }}
  .communication-mosaic-key > span {{ display:grid;grid-template-columns:28px minmax(0,1fr);
                                      align-items:center;gap:7px;min-width:0;padding:8px;
                                      border:1px solid color-mix(in srgb,var(--mosaic-color) 34%,#ddd);
                                      border-top:4px solid var(--mosaic-color);border-radius:9px;
                                      background:#fffdf9; }}
  .communication-mosaic-key b {{ font-size:1.18rem;line-height:1; }}
  .communication-mosaic-key strong,.communication-mosaic-key small {{ display:block; }}
  .communication-mosaic-key strong {{ color:var(--ink);font-size:.65rem;line-height:1.2; }}
  .communication-mosaic-key small {{ margin-top:2px;color:var(--muted);
                                     font-size:.51rem;line-height:1.22; }}
  .communication-mosaic-scroll-cue {{ margin:0 0 5px;color:var(--muted);
                                       font-size:.6rem;font-weight:730;
                                       letter-spacing:.03em;text-align:right; }}
  .communication-mosaic-scroll {{ width:100%;max-width:100%;overflow-x:auto;
                                   overscroll-behavior-inline:contain;padding:2px 1px 10px;
                                   scrollbar-width:thin; }}
  .communication-mosaic-scroll:focus-visible {{ outline:3px solid rgba(49,95,120,.3);
                                                 outline-offset:3px;border-radius:7px; }}
  .communication-mosaic-strip {{ display:grid;grid-template-columns:repeat(9,184px);
                                  gap:9px;width:max-content; }}
  .communication-mosaic-card {{ min-width:0;padding:10px;background:#fffdf9;
                                 border:1px solid var(--border);border-radius:12px;
                                 box-shadow:0 4px 14px rgba(48,42,31,.035); }}
  .communication-mosaic-card header {{ display:grid;align-content:start;min-height:63px;
                                        padding-bottom:8px;border-bottom:1px solid var(--grid); }}
  .communication-mosaic-card header strong {{ font-size:.73rem;line-height:1.2; }}
  .communication-mosaic-card header span {{ margin-top:2px;color:var(--muted);font-size:.59rem; }}
  .communication-mosaic-card header small {{ margin-top:4px;color:#8b5e34;
                                              font-size:.52rem;font-weight:800; }}
  .communication-mosaic-box {{ display:flex;flex-direction:column;width:100%;height:224px;
                                margin-top:9px;overflow:hidden;border-radius:9px;
                                background:#e6ded3;box-shadow:inset 0 0 0 1px rgba(48,42,31,.12); }}
  .communication-mosaic-row {{ display:flex;flex:0 0 var(--row-share);min-height:0; }}
  .communication-mosaic-cell {{ position:relative;display:flex;flex:0 0 var(--cell-share);
                                 box-sizing:border-box;min-width:0;min-height:0;overflow:hidden;
                                 align-items:center;justify-content:center;flex-direction:column;
                                 gap:3px;padding:4px;background:var(--mosaic-color);
                                 color:var(--mosaic-text);text-align:center;
                                 box-shadow:inset 0 0 0 1px rgba(255,255,255,.72);cursor:help; }}
  .communication-mosaic-cell:focus {{ z-index:2;outline:3px solid #181511;outline-offset:-4px; }}
  .communication-mosaic-emoji {{ font-size:1.35rem;line-height:1;filter:saturate(.9); }}
  .communication-mosaic-cell strong {{ font:800 .72rem/1 system-ui,sans-serif; }}
  .communication-mosaic-cell small {{ max-width:100%;font-size:.53rem;font-weight:760;
                                      line-height:1.1;overflow-wrap:anywhere; }}
  .communication-mosaic-cell.is-tiny > * {{ display:none; }}
  .communication-mosaic-cell.is-small {{ flex-direction:row;gap:2px;padding:2px; }}
  .communication-mosaic-cell.is-small small {{ display:none; }}
  .communication-mosaic-cell.is-small .communication-mosaic-emoji {{ font-size:.85rem; }}
  .communication-mosaic-cell.is-small strong {{ font-size:.55rem; }}
  .communication-mosaic-cell.is-medium small {{ display:none; }}
  .communication-mosaic-readout {{ display:grid;grid-template-columns:1fr 1fr;gap:3px 6px;
                                    margin:8px 0 0;color:var(--ink2);font-size:.54rem;
                                    font-variant-numeric:tabular-nums; }}
  .communication-mosaic-readout span {{ white-space:nowrap; }}
  .communication-mosaic-readout b {{ font-size:.7rem; }}
  .communication-mosaic-evidence {{ display:grid;grid-template-columns:1fr 1fr;gap:10px;
                                    margin-bottom:12px; }}
  .communication-mosaic-evidence .inspector-row {{ grid-template-columns:1fr;align-content:start;
                                                   margin:0; }}
  .communication-audit-grid {{ display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px; }}
  .communication-audit-grid section {{ padding:12px;background:#faf7f1;border:1px solid var(--border);
                                       border-radius:9px; }}
  .communication-audit-grid h4 {{ margin:0 0 7px;font:680 .9rem/1.2 Georgia,serif; }}
  .communication-audit-grid ul {{ display:grid;gap:5px;margin:0;padding-left:20px;
                                  color:var(--ink2);font-size:.7rem;line-height:1.4; }}
  .summary-method-note {{ max-width:none;margin:10px 0 18px;padding:13px 15px;
                          background:#f6f1e9;border-left:4px solid #8b5e34;
                          border-radius:0 9px 9px 0;color:var(--ink2);font-size:.8rem; }}
  .adversary-succession {{ display:grid;grid-auto-flow:column;
                           grid-auto-columns:minmax(178px,1fr);gap:8px;overflow-x:auto;
                           padding:4px 2px 12px;scroll-snap-type:x proximity; }}
  .adversary-era {{ min-height:345px;padding:12px;background:var(--surface);
                    border:1px solid var(--border);border-radius:12px;scroll-snap-align:start; }}
  .adversary-era header {{ min-height:45px;border-bottom:1px solid var(--grid); }}
  .adversary-era header strong,.adversary-era header span {{ display:block; }}
  .adversary-era header strong {{ font-size:.78rem; }}
  .adversary-era header span {{ margin-top:2px;color:var(--muted);font-size:.65rem; }}
  .adversary-era ol {{ list-style:none;margin:12px 0 0;padding:0;display:grid;gap:9px; }}
  .adversary-era li {{ display:grid;grid-template-columns:var(--adversary-size) 1fr;
                       align-items:center;column-gap:7px;min-height:var(--adversary-size); }}
  .adversary-era li > span {{ grid-row:1/3;width:var(--adversary-size);height:var(--adversary-size);
                              border-radius:50%;background:#74695f;border:2px solid white;
                              box-shadow:0 0 0 1px #74695f;opacity:.86; }}
  .adversary-era li strong {{ font-size:.7rem;line-height:1.15; }}
  .adversary-era li small {{ color:var(--muted);font-size:.58rem;line-height:1.2; }}
  .adversary-era .adversary-nation > span {{ background:#315f78;box-shadow:0 0 0 1px #315f78; }}
  .adversary-era .adversary-person > span {{ background:#a24f52;box-shadow:0 0 0 1px #a24f52; }}
  .adversary-era .adversary-group > span {{ background:#4f7557;box-shadow:0 0 0 1px #4f7557; }}
  .adversary-era .adversary-institution > span {{ background:#9a7133;box-shadow:0 0 0 1px #9a7133; }}
  .era-control-intro {{ max-width:820px;color:var(--ink2);font-size:.85rem; }}
  .era-preset-actions {{ display:flex;flex-wrap:wrap;gap:7px;margin:0 0 10px; }}
  .era-preset-actions button {{ appearance:none;padding:7px 10px;border:1px solid var(--border);
                                border-radius:999px;background:var(--surface);color:var(--ink2);
                                font:750 .72rem/1 system-ui;cursor:pointer; }}
  .era-preset-actions button:hover {{ border-color:#8b5e34; }}
  .era-preset-actions button:focus-visible {{ outline:3px solid #d08b45;outline-offset:2px; }}
  .story-stage {{ margin:20px 0 8px; }}
  .stage-controls {{ display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px; }}
  .stage-controls button {{ appearance:none;border:1px solid var(--border);background:var(--surface);
                            color:var(--ink2);border-radius:999px;padding:8px 12px;cursor:pointer;
                            font:650 .8rem/1.2 system-ui; }}
  .stage-controls button[aria-pressed="true"] {{ color:white;background:#315f78;border-color:#315f78; }}
  .stage-controls button:focus-visible {{ outline:3px solid #d08b45;outline-offset:2px; }}
  .stage-status {{ min-height:1.4em;color:#5d5144;font-size:.82rem;font-weight:650;margin:5px 0 10px; }}
  .stage-narrative {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
                      gap:8px;margin-top:14px; }}
  .stage-step {{ border-left:3px solid var(--border);padding:10px 12px;background:var(--surface);
                 border-radius:0 9px 9px 0;min-height:85px; }}
  .stage-step.is-active {{ border-left-color:#315f78;background:#edf5f8; }}
  .stage-step strong,.stage-step span {{ display:block; }}
  .stage-step span {{ margin-top:5px;color:var(--muted);font-size:.78rem;line-height:1.42; }}
  .stage-panels {{ min-height:240px; }}
  .message-compare {{ display:grid;grid-template-columns:1fr 1fr;gap:12px; }}
  .message-evidence {{ padding:16px;background:var(--surface);border:1px solid var(--border);
                       border-radius:12px;min-width:0; }}
  .message-evidence h3 {{ font-size:1rem;margin:4px 0 12px; }}
  .message-kicker {{ color:#6f4828;font-size:.75rem;font-weight:800;text-transform:uppercase;
                     letter-spacing:.06em;margin:0; }}
  .topic-strip {{ display:flex;height:58px;overflow:hidden;border-radius:8px;border:1px solid #b7aa98;
                  background:#eee; }}
  .topic-strip span {{ min-width:1px;border-right:1px solid rgba(255,255,255,.45); }}
  .topic-legend {{ display:flex;flex-wrap:wrap;gap:5px 10px;margin:10px 0;font-size:.68rem;color:var(--ink2); }}
  .topic-legend span {{ display:inline-flex;align-items:center;gap:4px; }}
  .topic-legend i {{ width:11px;height:11px;border:1px solid rgba(0,0,0,.25); }}
  .topic-c0 {{ background:repeating-linear-gradient(45deg,#315f78,#315f78 5px,#4d7b91 5px,#4d7b91 8px); }}
  .topic-c1 {{ background:repeating-linear-gradient(-45deg,#9b4e50,#9b4e50 5px,#b36a6c 5px,#b36a6c 8px); }}
  .topic-c2 {{ background:repeating-linear-gradient(90deg,#7b6a98,#7b6a98 5px,#9586af 5px,#9586af 8px); }}
  .topic-c3 {{ background:repeating-linear-gradient(0deg,#b17a43,#b17a43 5px,#c99867 5px,#c99867 8px); }}
  .topic-c4 {{ background:repeating-linear-gradient(45deg,#557c63,#557c63 4px,#71947b 4px,#71947b 8px); }}
  .topic-c5 {{ background:repeating-linear-gradient(-45deg,#8b6c42,#8b6c42 4px,#aa8b62 4px,#aa8b62 8px); }}
  .topic-c6 {{ background:repeating-linear-gradient(90deg,#477e9d,#477e9d 4px,#6c9bb5 4px,#6c9bb5 8px); }}
  .topic-c7 {{ background:repeating-linear-gradient(0deg,#8d6b72,#8d6b72 4px,#a8848b 4px,#a8848b 8px); }}
  .topic-c8 {{ background:repeating-linear-gradient(45deg,#6d7b83,#6d7b83 4px,#89969d 4px,#89969d 8px); }}
  .topic-c9 {{ background:repeating-linear-gradient(-45deg,#a08e72,#a08e72 4px,#b7a68c 4px,#b7a68c 8px); }}
  .message-measures {{ display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:10px 0; }}
  .message-measures span {{ font-size:.7rem;color:var(--muted); }}
  .message-measures b {{ display:block;color:var(--ink);font-size:1rem; }}
  .message-evidence blockquote {{ margin:12px 0;padding:10px 12px;background:var(--page);
                                  border-left:3px solid #8b6c42; }}
  .message-evidence blockquote p {{ margin:0;font-size:.78rem;line-height:1.5; }}
  .receipt-grid {{ display:grid;grid-template-columns:1fr 1fr;gap:12px; }}
  .receipt-focus,.sensitivity-panel {{ padding:18px;background:var(--surface);
                                      border:1px solid var(--border);border-radius:12px; }}
  .receipt-label {{ margin:0;color:var(--muted);font-size:.76rem;text-transform:uppercase;
                    letter-spacing:.06em;font-weight:750; }}
  .receipt-number {{ font:750 clamp(2rem,5vw,3.6rem)/1 Georgia,serif;color:#315f78;margin:10px 0; }}
  .receipt-number small {{ font:600 .8rem/1.2 system-ui;color:var(--muted); }}
  .tradeoff-summary {{ display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin:0 0 10px; }}
  .tradeoff-summary article {{ padding:13px;background:#f4eee5;border:1px solid #ded2c2;border-radius:10px; }}
  .tradeoff-summary span,.tradeoff-summary strong,.tradeoff-summary small {{ display:block; }}
  .tradeoff-summary span {{ color:var(--muted);font-size:.72rem;font-weight:750;text-transform:uppercase; }}
  .tradeoff-summary strong {{ margin-top:5px;font:700 1.2rem/1.15 Georgia,serif; }}
  .tradeoff-summary small {{ margin-top:5px;color:var(--muted);font-size:.68rem; }}
  .table-scroll {{ overflow-x:auto; }}
  .table-scroll table {{ width:100%;border-collapse:collapse;font-size:.8rem; }}
  .table-scroll th,.table-scroll td {{ padding:9px 10px;border-bottom:1px solid var(--border);
                                       text-align:left;vertical-align:top; }}
  .expansion-story-role {{ margin:18px 0 28px;padding:13px 15px;border-left:4px solid #315f78;
                           border-radius:0 10px 10px 0;background:#edf4f7;color:var(--ink2); }}
  .expansion-story-figure {{ margin:30px 0 42px;padding:18px;border:1px solid var(--border);
                             border-radius:14px;background:var(--surface); }}
  .expansion-story-scroll {{ max-width:100%;overflow-x:auto;overflow-y:hidden;
                              padding-bottom:10px;overscroll-behavior-inline:contain; }}
  .expansion-story-chart {{ min-width:1080px; }}
  .expansion-overlap-note {{ margin:10px 0 14px;padding:10px 12px;background:#f6f1e9;
                             border:1px solid #ded2c2;border-radius:9px;color:var(--ink2);
                             font-size:.78rem;line-height:1.5; }}
  .expansion-story-table {{ margin:10px 0 16px;border:1px solid var(--border);
                            border-radius:9px;padding:9px 11px; }}
  .expansion-story-table > summary {{ cursor:pointer;font-size:.75rem;font-weight:780; }}
  .expansion-story-table table {{ min-width:1480px; }}
  .expansion-story-table th:first-child {{ position:sticky;left:0;z-index:2;
                                           min-width:185px;background:var(--surface); }}
  .expansion-story-table td {{ min-width:155px;font-size:.69rem;line-height:1.42; }}
  .expansion-story-table .adversary-row th,.expansion-story-table .adversary-row td {{
    border-top:3px solid #8b6c42;background:#f7f0e7;
  }}
  .expansion-receipt-group {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
                              gap:9px;margin:12px 0 30px; }}
  .expansion-receipt-group blockquote {{ margin:0;padding:13px 14px;border-left:3px solid #8b6c42;
                                         border-radius:0 9px 9px 0;background:var(--surface);
                                         box-shadow:0 6px 18px rgba(48,42,31,.04); }}
  .expansion-receipt-group blockquote p {{ margin:0;font:500 .82rem/1.55 Georgia,serif; }}
  .expansion-receipt-group cite {{ display:block;margin-top:8px;color:var(--muted);
                                   font-style:normal;font-size:.68rem;line-height:1.4; }}
  .expansion-receipt-group a {{ color:#315f78; }}
  .communication-summary {{ margin:36px 0;padding:22px;border:1px solid var(--border);
                            border-radius:16px;background:#faf7f1; }}
  .communication-summary-heading > span {{ color:#8b5e34;font-size:.68rem;font-weight:800;
                                           text-transform:uppercase;letter-spacing:.08em; }}
  .communication-summary-heading h3 {{ margin:5px 0 8px;
                                       font:650 clamp(1.35rem,2.5vw,1.8rem)/1.15 Georgia,serif; }}
  .communication-summary-heading p {{ max-width:850px;margin:0;color:var(--ink2);
                                      font-size:.86rem;line-height:1.55; }}
  .communication-legend {{ display:flex;flex-wrap:wrap;gap:6px 18px;margin:18px 0 10px;
                           color:var(--muted);font-size:.69rem; }}
  .communication-summary-scroll {{ max-width:100%;overflow-x:auto;padding-bottom:8px; }}
  .communication-era-strip {{ display:grid;grid-template-columns:repeat(9,132px);gap:8px;
                              min-width:1252px; }}
  .communication-era-card {{ padding:10px;border:1px solid var(--border);border-radius:11px;
                             background:rgba(255,255,255,.72); }}
  .communication-era-card header {{ min-height:45px;border-bottom:1px solid var(--grid); }}
  .communication-era-card header strong,.communication-era-card header span {{ display:block; }}
  .communication-era-card header strong {{ font-size:.69rem;line-height:1.25; }}
  .communication-era-card header span {{ margin-top:3px;color:var(--muted);font-size:.59rem; }}
  .communication-era-card > div {{ display:grid;grid-template-columns:27px 1fr;align-items:center;
                                   min-height:72px;border-bottom:1px solid var(--grid); }}
  .communication-era-card > div:last-child {{ border-bottom:0; }}
  .communication-era-card small {{ color:#8b5e34;font-size:.58rem;font-weight:800;
                                   text-transform:uppercase; }}
  .communication-constellation {{ display:flex;align-items:center;justify-content:center;
                                  gap:1px;min-width:0; }}
  .communication-symbol {{ display:inline-grid;place-items:center;font-size:var(--symbol-size);
                           line-height:1;opacity:var(--symbol-opacity);
                           filter:saturate(.75);cursor:help; }}
  .communication-summary-note {{ margin:10px 0 0;color:var(--muted);
                                 font-size:.68rem;line-height:1.5; }}
  .coverage-toy {{ display:grid;gap:14px;margin:22px 0 10px;padding:16px;
                   background:#f6f1e9;border:1px solid #ded2c2;border-radius:12px; }}
  .coverage-row {{ display:grid;grid-template-columns:125px 1fr;gap:8px 12px;align-items:center; }}
  .coverage-row > strong {{ font-size:.82rem; }}
  .coverage-row small {{ grid-column:2;color:var(--muted); }}
  .coverage-ribbons {{ display:flex;height:50px;border-radius:8px;overflow:hidden;gap:2px; }}
  .coverage-ribbons span {{ display:flex;align-items:center;justify-content:center;min-width:0;
                            padding:4px;color:white;font-size:.72rem;font-weight:700;text-align:center; }}
  .toy-note {{ color:var(--muted);font-size:.8rem; }}
  .founding-chapter {{ width:min(1160px,calc(100vw - 48px));
                       margin-left:calc((100% - min(1160px,calc(100vw - 48px)))/2);
                       min-width:0; }}
  .founding-workspace {{ margin-bottom:24px;scroll-margin-top:100px; }}
  .founding-movement-route {{ position:relative;z-index:2;display:grid;
                              grid-template-columns:repeat(4,minmax(0,1fr));gap:0;
                              margin:18px 0 0;padding:0;
                              border:1px solid var(--border);border-bottom:0;
                              border-radius:18px 18px 0 0;overflow:hidden;
                              background:#eee7dc;
                              box-shadow:0 10px 22px rgba(48,42,31,.06); }}
  .founding-movement-route button {{ appearance:none;display:grid;gap:3px;min-width:0;
                                     padding:14px 16px;border:0;border-right:1px solid var(--border);
                                     border-radius:0;background:transparent;color:var(--muted);
                                     font:inherit;text-align:left;cursor:pointer; }}
  .founding-movement-route button:last-child {{ border-right:0; }}
  .founding-movement-route button:hover {{ background:rgba(255,255,255,.48);color:var(--ink2); }}
  .founding-movement-route button:focus-visible,.founding-bubble:focus-visible {{
    outline:3px solid rgba(208,139,69,.42);outline-offset:-3px;
  }}
  .founding-movement-route button[aria-selected="true"] {{
    background:#faf7f1;color:var(--ink);box-shadow:inset 0 4px #8b5e34;
  }}
  .founding-movement-route button[data-step-status="complete"] strong::after {{
    content:" ✓";color:#477458;
  }}
  .founding-movement-route strong {{ min-width:0;font-size:.8rem;line-height:1.2; }}
  .founding-movement-route small {{ min-width:0;color:var(--muted);font-size:.59rem;
                                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }}
  .founding-panel-stage {{ min-width:0; }}
  [data-era-panel] {{
    padding-top:14px;
  }}
  .founding-panel[hidden],[data-founding-screen][hidden] {{ display:none!important; }}
  .founding-story-card {{ min-width:0;max-width:100%;margin:0;scroll-margin-top:94px;
                           border:1px solid var(--border);border-radius:0 0 18px 18px;
                           background:#faf7f1;
                           box-shadow:0 16px 38px rgba(48,42,31,.055); }}
  .era-template {{ margin:0;padding:24px;background:#f8f4ed; }}
  .era-profile-standalone {{ margin:18px 0 24px;border-radius:18px; }}
  .era-visualize-standalone {{
    min-width:0;margin:18px 0 28px;padding:16px 24px 24px;
    border:1px solid var(--border);border-radius:18px;background:#faf7f1;
    box-shadow:0 16px 38px rgba(48,42,31,.055);
  }}
  .era-visualize-heading {{
    display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:end;
    gap:8px 24px;margin:0 0 10px;padding:0 0 10px;
    border-bottom:1px solid var(--border);
  }}
  .era-visualize-heading > div > span {{
    display:block;margin-bottom:3px;color:#8b5e34;font-size:.56rem;
    font-weight:850;letter-spacing:.09em;text-transform:uppercase;
  }}
  .era-visualize-heading h3 {{
    margin:0;font:650 clamp(1.4rem,3vw,2rem)/1 Georgia,serif;
  }}
  .era-visualize-heading p {{
    max-width:340px;margin:0;color:var(--muted);font-size:.62rem;
    line-height:1.4;text-align:right;
  }}
  .era-template-heading {{ display:grid;grid-template-columns:minmax(0,1fr) auto;
                           gap:5px 28px;align-items:end;margin-bottom:8px; }}
  .era-template-heading h3 {{ margin:0;font:650 clamp(2rem,5vw,3.5rem)/.95 Georgia,serif; }}
  .era-template-heading p {{ display:grid;justify-items:end;gap:3px;margin:0; }}
  .era-template-heading p strong {{ color:#6f4828;font:680 1.05rem/1.1 Georgia,serif; }}
  .era-template-heading p span {{ color:var(--muted);font-size:.66rem; }}
  .era-president-strip {{ display:flex;align-items:flex-start;justify-content:center;
                          flex-wrap:wrap;gap:9px 22px;list-style:none;
                          margin:0 0 13px;padding:0; }}
  .era-president-strip li {{ display:grid;justify-items:center;gap:5px;min-width:76px;
                             color:var(--ink2);text-align:center; }}
  .era-president-portrait-link {{ display:block;border-radius:50%;color:inherit;
                                  text-decoration:none; }}
  .era-president-strip img {{ display:block;width:60px;height:60px;border-radius:50%;
                              box-shadow:0 0 0 1px #cbbba8,0 4px 11px rgba(48,42,31,.13);
                              transition:box-shadow .15s ease,transform .15s ease; }}
  .era-president-portrait-link:hover img,
  .era-president-portrait-link:focus-visible img {{
    box-shadow:0 0 0 3px #315f78,0 5px 13px rgba(48,42,31,.18);
    transform:translateY(-1px);
  }}
  .era-president-portrait-link:focus-visible {{ outline:none; }}
  .era-president-strip span {{ max-width:92px;font-size:.58rem;font-weight:750;
                               line-height:1.12; }}
  .era-president-strip small {{ max-width:100px;color:var(--muted);font-size:.49rem;
                                line-height:1.2; }}
  .era-president-cross-owner {{ padding:2px 5px;border:1px solid #8b5e34;
                                border-radius:999px;color:#6f4828;font-size:.45rem;
                                line-height:1.1; }}
  .era-president-strip-label {{ margin:-7px 0 12px!important;color:var(--muted);
                                font-size:.55rem!important;text-align:center; }}
  .era-template-grid {{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
                        grid-auto-rows:auto;gap:8px; }}
  .era-template-grid article {{ display:flex;min-width:0;min-height:230px;height:auto;overflow:hidden;
                                flex-direction:column;padding:13px;
                                border:1px solid var(--border);border-radius:11px;
                                background:rgba(255,255,255,.72); }}
  .era-template-grid ul {{ list-style:none;margin:0;padding:0; }}
  .era-template-grid li {{ padding:5px 0;border-top:1px solid var(--grid); }}
  .era-template-grid li:first-child {{ border-top:0;padding-top:0; }}
  .era-template-grid li {{ color:var(--ink2);font-size:.67rem;line-height:1.3; }}
  .era-template-grid li span {{ display:inline; }}
  .era-template-grid li small {{ float:right;color:#8b5e34;font-weight:800; }}
  .era-card-heading {{ display:flex;align-items:baseline;justify-content:space-between;
                       gap:8px;width:auto;min-height:0;margin:0 0 9px!important;
                       padding:0 0 7px!important;border-bottom:1px solid var(--grid);
                       background:transparent; }}
  .era-card-heading > span {{ color:#6f4828;font-size:.62rem;font-weight:850;
                              text-transform:uppercase;letter-spacing:.07em; }}
  .era-card-heading > strong {{ flex:0 0 auto;color:var(--ink);
                                font:720 1.35rem/1 Georgia,serif; }}
  .era-card-heading > strong small {{ color:var(--muted);font:750 .52rem/1 system-ui;
                                      text-transform:uppercase;letter-spacing:.05em; }}
  .era-template-grid article > p {{ display:grid;grid-template-columns:auto 1fr;
                                    align-items:baseline;gap:4px 7px;margin:5px 0; }}
  .era-template-grid article > p strong {{ color:var(--ink);font:680 1.35rem/1 Georgia,serif; }}
  .era-template-grid article > p span {{ color:var(--muted);font-size:.6rem;line-height:1.3; }}
  .era-template-grid article > small {{ display:block;margin-top:8px;color:var(--muted);
                                        font-size:.58rem;line-height:1.4; }}
  .era-template-grid .era-template-references,
  .era-template-grid .era-template-adversaries {{ min-height:230px;align-self:stretch; }}
  .era-template-grid .era-template-references {{ overflow:visible; }}
  .era-reference-landscape {{ display:grid;flex:1;min-width:0;margin:0;padding:0;
    list-style:none; }}
  .era-reference-lane {{ display:grid;grid-template-columns:minmax(116px,.72fr)
    minmax(170px,1.28fr);align-items:stretch;gap:0;
    min-width:0;padding:0!important;border-top:1px solid var(--grid); }}
  .era-reference-lane:first-child {{ border-top:0; }}
  .era-reference-metric {{ display:grid;grid-template-columns:minmax(0,1fr) auto;
    align-items:center;align-content:center;gap:1px 5px;min-width:0;padding:6px 12px 6px 0; }}
  .era-reference-metric span {{ min-width:0;color:#6f4828;font-size:.55rem;
    font-weight:800;line-height:1.15;overflow-wrap:anywhere; }}
  .era-reference-metric strong {{ color:var(--ink);font:680 .8rem/1 Georgia,serif; }}
  .era-reference-metric small {{ grid-column:1/-1;float:none!important;color:var(--muted)!important;
    font-size:.45rem!important;font-weight:650!important; }}
  .era-reference-highlight,.era-reference-highlight-empty {{ display:block!important;min-width:0;
    align-self:stretch;border:0;border-left:1px solid #d2c5b4;border-radius:0;
    background:#f3ede4;box-shadow:none; }}
  .era-reference-highlight[data-support-status="limited_record"] {{ background:#f7f2e9; }}
  .era-reference-highlight > summary {{ display:grid;grid-template-columns:minmax(0,1fr);
    justify-items:center;align-content:center;gap:3px;min-width:0;padding:7px 8px;
    text-align:center;cursor:pointer; }}
  .era-reference-highlight > summary:focus-visible {{ outline:3px solid rgba(208,139,69,.45);
    outline-offset:2px;border-radius:5px; }}
  .era-reference-name {{ display:block!important;width:100%;min-width:0; }}
  .era-reference-highlight > summary b {{ display:block;min-width:0;color:var(--ink2);
    font:700 .74rem/1.2 Georgia,serif;overflow-wrap:anywhere; }}
  .era-template-grid .era-reference-meta {{ display:flex!important;align-items:center;
    justify-content:center;flex-wrap:wrap;min-width:0;color:var(--muted)!important;
    font-size:.42rem!important;font-weight:650!important;line-height:1.3; }}
  .era-template-grid .era-reference-meta > span {{ display:inline-flex;align-items:center; }}
  .era-template-grid .era-reference-meta > span + span::before {{ content:"·";
    margin:0 4px;color:#9b8c7d;font-weight:700; }}
  .era-reference-highlight-empty {{ padding:9px 10px;color:var(--muted);
    font-size:.5rem;font-style:italic;line-height:1.2;text-align:center; }}
  .era-reference-status {{ display:inline;color:#6e5945;font-size:.4rem;font-weight:850;
    letter-spacing:.02em;white-space:nowrap; }}
  .era-reference-badge {{ display:inline-flex!important;align-items:center;gap:3px;padding:0;
    border:0;border-radius:0;color:#6f6255;font-size:.43rem;
    font-weight:800;white-space:nowrap; }}
  .era-reference-badge i {{ color:#6f4828;font-style:normal; }}
  .era-reference-evidence {{ grid-column:1/-1;min-width:0;margin:0;padding:9px 10px;
    border-top:1px solid #d8ccbc;border-radius:0;background:#faf7f2;
    overflow-wrap:anywhere; }}
  .era-reference-evidence p {{ display:block!important;margin:0 0 7px!important;
    font-size:.58rem;line-height:1.45; }}
  .era-reference-evidence p:last-child {{ margin-bottom:0!important; }}
  .era-reference-evidence .era-reference-limited-note {{ margin-bottom:8px!important;
    color:#5f4b38;font-size:.54rem!important; }}
  .era-reference-excerpt {{ margin:7px 0;padding:8px 9px;border-left:2px solid #315f78;
    background:rgba(255,255,255,.7);font:500 .62rem/1.5 Georgia,serif; }}
  .era-reference-evidence code {{ font-size:.5rem;white-space:normal;overflow-wrap:anywhere; }}
  .era-adversary-pack {{ display:block;width:100%;height:auto;max-height:154px;
                         flex:1;overflow:visible; }}
  .era-adversary-bubble circle {{
    fill:color-mix(in srgb,var(--adversary-color) 42%,#fffaf1);
    stroke:color-mix(in srgb,var(--adversary-color) 72%,#493523);
    stroke-width:1.2;
  }}
  .era-adversary-bubble text {{ fill:#2f2922;font:730 9px/1 system-ui; }}
  .era-adversary-bubble .era-adversary-count {{
    fill:#4b4035;font:750 10px/1 Georgia,serif;
  }}
  .era-adversary-legend {{ display:flex;align-items:center;justify-content:center;
                           flex-wrap:wrap;gap:5px 10px;min-height:14px;color:var(--muted);
                           font-size:.52rem;font-weight:700;line-height:1; }}
  .era-adversary-legend span {{ display:inline-flex;align-items:center;gap:4px; }}
  .era-adversary-legend i {{ width:7px;height:7px;border-radius:50%;
                             background:var(--adversary-color); }}
  .era-adversary-legend .is-absent {{ color:#938a7f; }}
  .era-adversary-legend .is-absent i {{
    border:1px solid var(--adversary-color);background:transparent;
  }}
  .era-adversary-empty {{ fill:var(--muted);font:650 12px/1 system-ui; }}
  .era-style-scroll {{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;
                       max-height:145px;overflow-y:auto;overscroll-behavior:contain;
                       padding:1px 4px 2px 1px;scrollbar-width:thin; }}
  .era-style-scroll:focus-visible {{ outline:3px solid rgba(208,139,69,.4);
                                     outline-offset:2px; }}
  .era-style-scroll section {{ min-width:0;margin:0!important;padding:7px;border-radius:8px;
                               background:#f6f1e9; }}
  .era-style-scroll section > strong {{ display:block;margin-bottom:3px;color:#6f4828;
                                        font-size:.54rem;text-transform:uppercase;
                                        letter-spacing:.07em; }}
  .era-style-scroll li {{ display:grid;grid-template-columns:auto minmax(0,1fr) auto;
                          align-items:center;gap:4px;padding:4px 0!important; }}
  .era-style-scroll li > span {{ font-size:.72rem; }}
  .era-style-scroll li > b {{ min-width:0;font-size:.54rem;font-weight:650;line-height:1.15; }}
  .era-style-scroll li > strong {{ color:#8b5e34;font-size:.54rem; }}
  .era-footprint-core {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
                         gap:5px;flex:1 1 0; }}
  .era-footprint-core p {{ position:relative;display:flex;min-width:0;min-height:65px;
                           flex-direction:column;justify-content:center;gap:2px;
                           margin:0;padding:8px;border-radius:8px;
                           background:#f6f1e9; }}
  .era-footprint-core strong {{ color:var(--ink);font:690 1.22rem/1 Georgia,serif; }}
  .era-footprint-core span {{ color:#6f4828;font-size:.54rem;font-weight:800;
                              text-transform:uppercase;letter-spacing:.04em; }}
  .era-template-footprint {{ container-type:inline-size; }}
  .era-footprint-core small {{ min-width:0;color:var(--muted);font-size:.5rem;
                               line-height:1.15;overflow-wrap:anywhere; }}
  .era-footprint-language {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
                             gap:5px;margin-top:5px;flex:1 1 0; }}
  .era-footprint-language p {{ display:grid;grid-template-columns:minmax(0,1fr);
                               align-content:center;gap:2px;margin:0;padding:6px 7px;
                               border:1px solid var(--grid);border-radius:8px; }}
  .era-footprint-language strong {{ min-width:0;color:var(--ink);
    font:680 clamp(.54rem,3cqi,.9rem)/1.05 Georgia,serif;
    white-space:nowrap;overflow-wrap:normal;hyphens:none;letter-spacing:-.01em; }}
  .era-footprint-language span {{ min-width:0;color:var(--ink2);
                                 font-size:.48rem;font-weight:750;line-height:1.1; }}
  .era-footprint-language small {{ min-width:0;color:var(--muted);
                                  font-size:.45rem;line-height:1.1; }}
  .era-footprint-language .era-distinctive-word {{
    border-color:rgba(127,147,197,.52);
    background:color-mix(in srgb,#7f93c5 10%,#fffaf1);
  }}
  .era-footprint-language .era-distinctive-word strong {{ color:#51689f; }}
  .era-distinctive-method {{ margin-top:6px;color:var(--muted); }}
  .era-distinctive-method summary {{ cursor:pointer;color:#6f4828;
    font-size:.5rem;font-weight:780;line-height:1.2; }}
  .era-distinctive-method > small {{ display:block;margin-top:5px;
    font-size:.47rem;line-height:1.35; }}
  .era-distinctive-method > small strong {{ color:var(--ink2);font-weight:780; }}
  .era-topic-bars {{ display:grid;gap:3px; }}
  .era-topic-bars li {{ position:relative;display:grid;
                        grid-template-columns:minmax(0,1fr) auto;align-items:center;
                        gap:6px;min-height:23px;padding:3px 6px!important;overflow:hidden;
                        border:0!important;border-radius:6px;background:#f6f1e9; }}
  .era-topic-bars li i {{ position:absolute;inset:0 auto 0 0;width:var(--topic-width);
                          background:linear-gradient(90deg,rgba(49,95,120,.24),rgba(49,95,120,.09)); }}
  .era-topic-bars li span,.era-topic-bars li strong {{ position:relative;z-index:1; }}
  .era-topic-bars li span {{ min-width:0;font-size:.57rem;font-weight:650;line-height:1.15; }}
  .era-topic-bars li strong {{ color:#315f78;font-size:.57rem; }}
  .founding-panel:not(.era-template) {{ padding:24px; }}
  .founding-panel-heading {{ display:grid;grid-template-columns:minmax(0,1fr) auto;
                             gap:5px 24px;align-items:end;margin-bottom:14px;padding-bottom:14px;
                             border-bottom:1px solid var(--border); }}
  .founding-panel-heading h3 {{ max-width:850px;margin:0;
                                font:600 clamp(1.55rem,3vw,2.35rem)/1.05 Georgia,serif; }}
  .founding-panel-heading p {{ margin:0;color:var(--muted);font-size:.68rem; }}
  .founding-screen-tabs {{ display:inline-flex;gap:4px;margin:0 0 14px;padding:4px;
                           border:1px solid var(--border);border-radius:999px;background:#eee7dc; }}
  :is([data-era-visualize],[data-era-contextualize]) .founding-screen-tabs {{
    display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px;
    width:min(620px,100%);margin:0 auto 9px;padding:3px;overflow:visible;
    border:0;border-bottom:1px solid rgba(11,11,11,.12);border-radius:0;
    background:transparent;
  }}
  .founding-screen-tabs button {{ appearance:none;padding:8px 13px;border:0;border-radius:999px;
                                  background:transparent;color:var(--muted);font:750 .68rem/1 system-ui;
                                  cursor:pointer; }}
  :is([data-era-visualize],[data-era-contextualize]) .founding-screen-tabs button {{
    min-height:32px;padding:7px 13px;border:0;border-bottom:2px solid transparent;
    border-radius:6px 6px 0 0;background:transparent;color:#7f776e;
    font:650 .61rem/1 system-ui;text-align:center;white-space:nowrap;
  }}
  :is([data-era-visualize],[data-era-contextualize])
    .founding-screen-tabs button:last-child {{ border-right:0; }}
  .founding-screen-tabs button[aria-selected="true"] {{
    background:var(--surface);color:var(--ink);box-shadow:0 2px 7px rgba(48,42,31,.12);
  }}
  :is([data-era-visualize],[data-era-contextualize])
    .founding-screen-tabs button[aria-selected="true"] {{
    background:rgba(139,94,52,.055);color:var(--ink2);box-shadow:none;
    border-bottom-color:#8b5e34;
  }}
  .founding-screen-tabs button:focus-visible {{ outline:3px solid rgba(208,139,69,.4);
                                                outline-offset:2px; }}
  .founding-screen-stage {{ min-width:0; }}
  .founding-screen-stage > [data-founding-screen] {{ scroll-margin-top:92px; }}
  :is([data-era-visualize],[data-era-contextualize]).founding-screen-group {{
    display:flex;min-height:calc(var(--era-profile-height,678px) - 48px);
    flex-direction:column;
  }}
  [data-era-visualize].founding-screen-group:has(
    [data-era-screen="agenda"]:not([hidden])
  ),
  [data-era-contextualize].founding-screen-group:has(
    [data-era-context-screen="defined"]:not([hidden])
  ) {{
    min-height:0;
  }}
  :is([data-era-visualize],[data-era-contextualize])
    .founding-screen-stage {{ display:grid;flex:1; }}
  :is([data-era-visualize],[data-era-contextualize])
    .founding-screen-stage > [data-founding-screen] {{
    display:flex;width:100%;max-width:none;min-width:0;min-height:100%;
    flex-direction:column;margin:0;
  }}
  :is([data-era-visualize],[data-era-contextualize]) .inspector-row {{
    width:100%;margin-top:auto;margin-bottom:0;padding-top:12px;
  }}
  [data-era-visualize] [data-era-screen="agenda"] .inspector-row {{
    margin-top:8px;padding-top:0;
  }}
  [data-era-contextualize] [data-era-context-screen="defined"] .inspector-row {{
    margin-top:8px;padding-top:0;
  }}
  :is([data-era-visualize],[data-era-contextualize])
    .inspector-tile:not([open]) summary {{
    display:block;max-width:100%;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap;
  }}
  :is([data-era-visualize],[data-era-contextualize])
    .inspector-row:has(.inspector-tile[open])
    .inspector-tile:not([open]) {{ display:none; }}
  .era-context-question {{
    display:grid;grid-template-columns:auto minmax(0,1fr);align-items:baseline;
    gap:4px 16px;margin:0 0 10px;padding:13px 15px;
    border:1px solid #ded2c2;border-left:4px solid #8b5e34;
    border-radius:0 11px 11px 0;background:#f8f3eb;
  }}
  .era-context-question > span {{
    color:#8b5e34;font-size:.55rem;font-weight:850;letter-spacing:.09em;
    text-transform:uppercase;
  }}
  .era-context-question h4 {{
    margin:0;font:680 clamp(1.25rem,2.7vw,1.85rem)/1.03 Georgia,serif;
  }}
  .era-context-question p {{
    grid-column:1/-1;max-width:780px;margin:2px 0 0;color:var(--muted);
    font-size:.64rem;line-height:1.4;
  }}
  .era-context-note {{
    margin:8px 0 0;color:var(--muted);font-size:.57rem;line-height:1.4;
  }}
  .era-defined-coverage-line {{
    margin:0 0 8px;color:var(--muted);font-size:.57rem;line-height:1.4;
  }}
  .era-defined-coverage-line strong {{
    color:var(--ink2);font:700 .74rem/1 Georgia,serif;
  }}
  .era-defined-combined-key {{
    display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
    gap:6px 14px;margin:0 0 7px;
  }}
  .era-defined-combined-key > span {{
    display:grid;grid-template-columns:22px minmax(0,1fr);align-items:start;
    gap:7px;min-width:0;
  }}
  .era-defined-combined-key > span > i {{
    height:3px;margin-top:7px;border-radius:99px;
    background:var(--defined-color);
  }}
  .era-defined-combined-key > span > span {{
    display:grid;gap:2px;min-width:0;
  }}
  .era-defined-combined-key strong {{
    color:var(--ink2);font:650 .58rem/1.2 Georgia,serif;
  }}
  .era-defined-combined-scale {{
    margin:0 0 5px;color:var(--muted);font-size:.48rem;line-height:1.35;
  }}
  .era-defined-combined-scroll {{
    width:100%;max-width:100%;overflow-x:auto;overscroll-behavior-inline:contain;
    border:1px solid var(--border);border-radius:10px;
    background:rgba(255,255,255,.66);
  }}
  .era-defined-combined-chart {{
    display:block;width:100%;min-width:680px;height:auto;
  }}
  .era-defined-focal-band {{
    fill:#f1e8dc;
  }}
  .era-defined-combined-grid line {{
    stroke:#ddd5ca;stroke-width:1;
  }}
  .era-defined-combined-grid text {{
    fill:#756b61;font:9px system-ui;
  }}
  .era-defined-combined-y-axis {{
    stroke:#8f857a;stroke-width:1.1;
  }}
  .era-defined-combined-series {{
    color:var(--defined-color);transition:opacity .16s ease;
  }}
  .era-defined-combined-series polyline {{
    fill:none;stroke-linecap:round;stroke-linejoin:round;
  }}
  .era-defined-combined-hit {{
    stroke:transparent;stroke-width:14;pointer-events:stroke;
  }}
  .era-defined-combined-line {{
    stroke:currentColor;stroke-width:2.7;pointer-events:none;
    transition:stroke-width .16s ease;
  }}
  .era-defined-combined-chart:has(.era-defined-combined-series:hover)
    .era-defined-combined-series:not(:hover),
  .era-defined-combined-chart:has(.era-defined-combined-node:focus)
    .era-defined-combined-series:not(:has(.era-defined-combined-node:focus)) {{
    opacity:.16;
  }}
  .era-defined-combined-series:hover .era-defined-combined-line,
  .era-defined-combined-series:has(.era-defined-combined-node:focus)
    .era-defined-combined-line {{
    stroke-width:4.4;
  }}
  .era-defined-combined-node circle {{
    fill:#fffdf9;stroke:currentColor;stroke-width:2;
  }}
  .era-defined-combined-node.is-focal circle {{
    fill:currentColor;stroke:#fffdf9;stroke-width:2;
  }}
  .era-defined-combined-node:focus circle {{
    stroke:#1e1b17;stroke-width:3.2;
  }}
  .era-defined-point-value {{
    fill:currentColor;opacity:0;pointer-events:none;
    font:700 10px/1 system-ui;paint-order:stroke;
    stroke:#fffdf9;stroke-width:4px;stroke-linejoin:round;
  }}
  .era-defined-combined-node:hover .era-defined-point-value,
  .era-defined-combined-node:focus .era-defined-point-value {{
    opacity:1;
  }}
  .era-defined-end-connector {{
    fill:none;stroke:currentColor;stroke-width:1.2;
  }}
  .era-defined-end-label {{
    fill:currentColor;font:650 10px/1 system-ui;
  }}
  .era-defined-combined-axis text {{
    fill:#6f665d;font:9px/1 system-ui;
  }}
  .era-defined-combined-method {{
    margin:7px 0 0;
  }}
  .era-defined-combined-method summary {{
    color:var(--muted);cursor:pointer;font-size:.5rem;font-weight:700;
  }}
  .era-defined-combined-method .era-context-note {{
    margin-top:6px;
  }}
  .era-defined-trajectory-key {{
    display:flex;justify-content:flex-end;gap:14px;margin:0 4px 5px;
    color:var(--muted);font-size:.5rem;
  }}
  .era-defined-trajectory-key span {{
    display:inline-flex;align-items:center;gap:5px;
  }}
  .era-defined-trajectory-key i {{
    width:18px;height:6px;border-radius:99px;background:#315f78;
  }}
  .era-defined-trajectory-key em {{
    width:2px;height:13px;border-radius:2px;background:#1e1b17;
  }}
  .era-defined-trajectory-head,
  .era-defined-trajectory-row,
  .era-defined-trajectory-axis {{
    display:grid;grid-template-columns:minmax(220px,.95fr)
      minmax(205px,.8fr) minmax(340px,1.35fr);align-items:center;gap:14px;
  }}
  .era-defined-trajectory-head {{
    min-height:25px;padding:0 11px;color:#6f4828;font-size:.43rem;
    font-weight:850;letter-spacing:.055em;text-transform:uppercase;
  }}
  .era-defined-trajectory-lanes {{
    display:grid;gap:0;list-style:none;margin:0;padding:0;
    border:1px solid var(--border);border-radius:10px;
    background:rgba(255,255,255,.66);
  }}
  .era-defined-trajectory-row {{
    min-height:84px;padding:7px 11px;border-top:1px solid #e5ddd1;
  }}
  .era-defined-trajectory-row:first-child {{ border-top:0; }}
  .era-defined-family {{
    display:grid;grid-template-columns:27px minmax(0,1fr);align-items:start;
    gap:8px;min-width:0;
  }}
  .era-defined-icon {{ font-size:1.03rem;line-height:1.1;text-align:center; }}
  .era-defined-family > span {{ display:grid;gap:3px;min-width:0; }}
  .era-defined-family strong {{ font:650 .71rem/1.15 Georgia,serif; }}
  .era-defined-family small {{
    color:var(--muted);font-size:.47rem;line-height:1.3;
  }}
  .era-defined-benchmark {{ display:grid;gap:5px;min-width:0; }}
  .era-defined-benchmark-track {{
    position:relative;height:13px;border-radius:99px;background:#eee8df;
    box-shadow:inset 0 0 0 1px rgba(48,42,31,.08);
  }}
  .era-defined-benchmark-track i {{
    display:block;width:var(--defined-fill);height:100%;border-radius:99px;
    background:var(--defined-color);
  }}
  .era-defined-benchmark-track span {{
    position:absolute;z-index:2;top:-4px;bottom:-4px;
    left:var(--defined-benchmark);width:2px;border-radius:2px;
    background:#1e1b17;box-shadow:0 0 0 2px rgba(255,255,255,.72);
  }}
  .era-defined-benchmark-values {{
    display:grid;grid-template-columns:auto auto 1fr;align-items:baseline;gap:6px;
    color:var(--muted);font-size:.42rem;white-space:nowrap;
  }}
  .era-defined-benchmark-values b {{
    color:var(--ink2);font:700 .52rem/1 Georgia,serif;
  }}
  .era-defined-benchmark-values strong {{
    justify-self:end;color:#477458;font-size:.47rem;
  }}
  .era-defined-benchmark-values strong.is-negative {{ color:#a24f52; }}
  .era-defined-trajectory {{ min-width:0; }}
  .era-defined-trajectory svg {{
    display:block;width:100%;height:auto;overflow:visible;
    color:var(--defined-color);
  }}
  .era-defined-trajectory-grid line {{
    stroke:#ddd5ca;stroke-width:1;
  }}
  .era-defined-trajectory polyline {{
    fill:none;stroke:currentColor;stroke-width:2.3;stroke-linecap:round;
    stroke-linejoin:round;
  }}
  .era-defined-trajectory-node circle {{
    fill:#fffdf9;stroke:currentColor;stroke-width:2;
  }}
  .era-defined-trajectory-node.is-founding circle {{
    fill:currentColor;stroke:#fffdf9;stroke-width:2;
  }}
  .era-defined-trajectory-node:focus circle {{
    stroke:#1e1b17;stroke-width:3.2;
  }}
  .era-defined-trajectory-axis {{
    min-height:25px;padding:0 11px;
  }}
  .era-defined-trajectory-axis div {{
    display:grid;grid-template-columns:repeat(9,minmax(0,1fr));
    color:var(--muted);font-size:.38rem;text-align:center;
  }}
  .era-defined-trajectory-table {{ margin-top:5px; }}
  .era-defined-trajectory-table summary {{
    color:#665b50;font-size:.5rem;font-weight:750;cursor:pointer;
  }}
  .era-defined-trajectory-table table {{ min-width:1020px; }}
  .era-defined-trajectory-table th,
  .era-defined-trajectory-table td {{
    font-size:.45rem;line-height:1.2;text-align:right;
  }}
  .era-defined-trajectory-table th:first-child {{ text-align:left; }}
  .era-defined-pending {{
    display:grid;place-items:center;min-height:330px;padding:30px;
    border:1px dashed #cdbda9;border-radius:12px;background:#f7f2ea;
    color:var(--muted);text-align:center;
  }}
  .era-defined-pending span {{
    color:#8b5e34;font-size:.63rem;font-weight:850;letter-spacing:.08em;
    text-transform:uppercase;
  }}
  .era-defined-pending p {{
    max-width:500px;margin:8px auto 0;font-size:.66rem;line-height:1.5;
  }}
  .expansion-defined-phase-grid {{
    display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;
    margin:0 0 8px;
  }}
  .expansion-defined-phase-grid article {{
    display:grid;align-content:start;gap:3px;min-width:0;padding:9px 10px;
    border:1px solid var(--border);border-top:3px solid #4f7557;
    border-radius:9px;background:#fffdf9;
  }}
  .expansion-defined-phase-grid span {{
    color:#4f7557;font-size:.47rem;font-weight:850;letter-spacing:.06em;
    text-transform:uppercase;
  }}
  .expansion-defined-phase-grid strong {{
    font:650 .69rem/1.15 Georgia,serif;
  }}
  .expansion-defined-phase-grid p {{
    margin:0;color:var(--muted);font-size:.47rem;line-height:1.3;
  }}
  .expansion-defined-phase-grid small {{
    margin-top:2px;color:#74695f;font-size:.39rem;line-height:1.25;
  }}
  .expansion-defined-scroll {{
    width:100%;max-width:100%;overflow-x:auto;overscroll-behavior-inline:contain;
    border:1px solid var(--border);border-radius:11px;
    background:linear-gradient(180deg,#fffdf9,#f7f1e8);
  }}
  .expansion-defined-chart {{
    display:block;width:100%;min-width:820px;height:auto;
  }}
  .expansion-defined-guides rect {{
    fill:#fbf7f0;
  }}
  .expansion-defined-guides rect.is-alt {{
    fill:#f2ece3;
  }}
  .expansion-defined-guides line {{
    stroke:#d8d0c4;stroke-width:1;
  }}
  .expansion-defined-guides text {{
    fill:#5e554c;font:750 10px/1 system-ui;
  }}
  .expansion-defined-guides .expansion-defined-phase-name {{
    fill:#8b8175;font-size:7px;font-weight:650;
  }}
  .expansion-defined-y-guides line {{
    stroke:#d9d1c6;stroke-width:1;stroke-dasharray:2 5;
  }}
  .expansion-defined-y-guides text {{
    fill:#7b7166;font:650 8px/1 system-ui;
  }}
  .expansion-defined-axis-title {{
    fill:#6b6259;font:750 8px/1 system-ui;letter-spacing:.03em;
  }}
  .expansion-defined-step {{
    fill:none;stroke:var(--expansion-series);stroke-width:3;
    stroke-linecap:round;stroke-linejoin:round;opacity:.92;
  }}
  .expansion-defined-diplomacy .expansion-defined-step {{
    stroke-width:8;opacity:.78;
  }}
  .expansion-defined-institutions .expansion-defined-step {{
    stroke-dasharray:8 4;
  }}
  .expansion-defined-node-shape {{
    fill:#fffdf9;stroke:var(--expansion-series);stroke-width:3;
  }}
  .expansion-defined-diplomacy .expansion-defined-node-shape {{
    fill:var(--expansion-series);stroke:#fffdf9;stroke-width:2;
  }}
  .expansion-defined-peak-value {{
    fill:#403931;font:800 8px/1 system-ui;paint-order:stroke;
    stroke:#fffdf9;stroke-width:3px;stroke-linejoin:round;
  }}
  .expansion-defined-node:focus {{ outline:none; }}
  .expansion-defined-node:focus .expansion-defined-node-shape,
  .expansion-defined-node:hover .expansion-defined-node-shape {{
    stroke:#1e1b17;stroke-width:4;
  }}
  .expansion-defined-end-label path {{
    fill:none;stroke:var(--expansion-series);stroke-width:1.5;
  }}
  .expansion-defined-end-label circle {{
    fill:var(--expansion-series);
  }}
  .expansion-defined-end-label text {{
    fill:#302b26;font:750 8px/1 system-ui;
  }}
  .expansion-defined-event-node line {{
    stroke:#31543e;stroke-width:1;stroke-dasharray:2 2;
  }}
  .expansion-defined-event-node path {{
    fill:#edf3ee;stroke:#31543e;stroke-width:2;
  }}
  .expansion-defined-event-node text {{
    fill:#31543e;font:850 7px/1 system-ui;paint-order:stroke;
    stroke:#fffdf9;stroke-width:2px;
  }}
  .expansion-defined-event-node:focus {{ outline:none; }}
  .expansion-defined-event-node:focus path,
  .expansion-defined-event-node:hover path {{
    fill:#31543e;stroke:#1e1b17;stroke-width:3;
  }}
  .expansion-defined-key {{
    display:flex;justify-content:flex-end;flex-wrap:wrap;gap:6px 13px;
    margin:5px 2px 0;color:var(--muted);font-size:.48rem;
  }}
  .expansion-defined-key span {{
    padding-left:9px;border-left:3px solid #4f7557;
  }}
  .expansion-defined-events {{
    display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;
    list-style:none;margin:8px 0 0;padding:0;
  }}
  .expansion-defined-events li {{
    display:grid;gap:2px;padding:7px 8px;border-radius:8px;background:#edf3ee;
  }}
  .expansion-defined-events time {{
    color:#4f7557;font-size:.43rem;font-weight:850;
  }}
  .expansion-defined-events strong {{
    font:650 .58rem/1.15 Georgia,serif;
  }}
  .expansion-defined-events span {{
    color:var(--muted);font-size:.4rem;line-height:1.25;
  }}
  .expansion-defined-table {{ margin-top:7px; }}
  .expansion-defined-table summary {{
    color:#665b50;font-size:.53rem;font-weight:750;cursor:pointer;
  }}
  .expansion-defined-table table {{ min-width:780px; }}
  .expansion-defined-table th,.expansion-defined-table td {{
    font-size:.5rem;line-height:1.3;
  }}
  .expansion-defined-table small {{ color:var(--muted);font-size:.4rem; }}
  .topic-life-scroll,.era-echo-scroll {{
    width:100%;max-width:100%;min-width:0;overflow-x:auto;
    overscroll-behavior-inline:contain;
  }}
  [data-era-context-screen="topic-life"],
  [data-era-context-screen="echoes"] {{
    min-width:0;overflow-x:hidden;
  }}
  .topic-life-key {{
    display:flex;justify-content:flex-end;gap:14px;margin:0 4px 4px;
    color:var(--muted);font-size:.49rem;
  }}
  .topic-life-key span {{ display:inline-flex;align-items:center;gap:5px; }}
  .topic-life-key i {{
    width:8px;height:8px;border-radius:50%;background:#315f78;
  }}
  .topic-life-key b {{
    width:11px;height:11px;border:2px solid #9a7133;border-radius:50%;
    background:transparent;
  }}
  .topic-life-chart {{ min-width:900px; }}
  .topic-life-axis {{
    display:grid;grid-template-columns:repeat(9,minmax(0,1fr));
    margin:0 5px 3px 208px;color:var(--muted);
    font-size:.48rem;font-weight:700;text-align:center;
  }}
  .topic-life-axis span {{
    display:grid;place-items:center;min-height:20px;padding:2px 3px;
    border-radius:5px;
  }}
  .topic-life-axis span.is-focal {{ background:#f0e4d4;color:#6f4828; }}
  .topic-life-rows {{ display:grid;gap:4px;list-style:none;margin:0;padding:0; }}
  .topic-life-row {{
    --topic-life-color:#74695f;--topic-life-soft:#eee9e3;
    display:grid;grid-template-columns:195px minmax(680px,1fr);
    align-items:center;gap:8px;min-height:63px;padding:1px 5px;
    border-bottom:1px solid rgba(48,42,31,.09);
  }}
  .topic-life-row.is-fades {{
    --topic-life-color:#a24f52;--topic-life-soft:#f5e2e0;
  }}
  .topic-life-row.is-persists {{
    --topic-life-color:#4f7557;--topic-life-soft:#e4eee6;
  }}
  .topic-life-row.is-returns {{
    --topic-life-color:#6d5a88;--topic-life-soft:#ece5f2;
  }}
  .topic-life-row.is-grows {{
    --topic-life-color:#9a7133;--topic-life-soft:#f3ead9;
  }}
  .topic-life-row.is-reframes {{
    --topic-life-color:#8b5e34;--topic-life-soft:#f2e5d7;
  }}
  .topic-life-label {{
    display:grid;grid-template-columns:19px 1fr;align-items:center;gap:4px 5px;
  }}
  .topic-life-label strong {{ font-size:.61rem;line-height:1.2; }}
  .topic-life-trend {{
    grid-column:2;width:max-content;max-width:100%;padding:3px 7px;
    border-radius:99px;background:var(--topic-life-soft);
    color:var(--topic-life-color);font-size:.46rem;font-weight:850;
    line-height:1.15;
  }}
  .topic-life-level {{
    grid-column:2;color:#5f574f;font-size:.45rem;font-weight:750;
    font-variant-numeric:tabular-nums;
  }}
  .topic-life-label em {{
    grid-column:2;color:#7b4c20;font-size:.38rem;font-style:normal;
    font-weight:800;letter-spacing:.04em;text-transform:uppercase;
  }}
  .topic-life-row svg {{ display:block;width:100%;height:63px;overflow:visible; }}
  .topic-life-row polyline {{
    fill:none;stroke:var(--topic-life-color);stroke-width:2;stroke-linecap:round;
    stroke-linejoin:round;opacity:.68;
  }}
  .topic-life-row .topic-life-successor-line {{
    stroke:#9a7133;stroke-width:1.6;stroke-dasharray:4 3;opacity:.52;
  }}
  .topic-life-row .topic-life-source-after,
  .topic-life-row .topic-life-successor-before {{
    stroke-width:1.35;stroke-dasharray:3 3;opacity:.2;
  }}
  .topic-life-row .topic-life-successor-before {{
    stroke:#9a7133;
  }}
  .topic-life-row .topic-life-successor-active {{
    stroke:#9a7133;stroke-width:2.35;opacity:.86;
  }}
  .topic-life-branch-switch {{
    stroke:#8b5e34;stroke-width:2;stroke-dasharray:2 2;opacity:.72;
  }}
  .topic-life-row circle {{
    fill:#faf7f1;stroke:var(--topic-life-color);stroke-width:2;
  }}
  .topic-life-row circle.is-focal {{
    fill:var(--topic-life-color);stroke:#fffaf1;stroke-width:2.5;
  }}
  .topic-life-row circle.is-muted-branch {{ opacity:.28; }}
  .topic-life-row circle:focus {{ outline:none;stroke:#1e1b17;stroke-width:3.2; }}
  .topic-life-row .topic-life-family-ring {{
    fill:rgba(255,255,255,.01);stroke:#9a7133;stroke-width:2.2;
    stroke-dasharray:2.5 1.5;cursor:help;
  }}
  .topic-life-row .topic-life-family-ring.is-focal {{
    fill:rgba(255,255,255,.01);stroke:#9a7133;stroke-width:2.7;
  }}
  .topic-life-row .topic-life-family-ring.is-active-successor {{
    fill:#f3ead9;stroke-width:2.7;opacity:1;
  }}
  .topic-life-row .topic-life-family-ring:hover,
  .topic-life-row .topic-life-family-ring:focus {{
    stroke:#5d3a1e;stroke-width:3.4;stroke-dasharray:none;
  }}
  .topic-life-baseline {{ stroke:#d9cdbd;stroke-width:1; }}
  .topic-life-gridline {{
    stroke:#d7cec2;stroke-width:.8;stroke-dasharray:2 4;opacity:.72;
  }}
  .topic-life-focal-band {{ fill:#f0e4d4;opacity:.7; }}
  .era-echo-summary {{
    margin:0 0 2px;color:var(--muted);font-size:.49rem;text-align:center;
  }}
  .era-echo-direction-switch {{
    display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px;
    width:min(620px,100%);margin:0 auto 4px;padding:3px;
    border-bottom:1px solid rgba(11,11,11,.12);
  }}
  .era-echo-direction-switch button {{
    appearance:none;display:grid;gap:3px;min-width:0;min-height:38px;
    padding:7px 12px;border:0;border-bottom:2px solid transparent;
    border-radius:6px 6px 0 0;background:transparent;color:#7f776e;
    font:700 .62rem/1 system-ui;text-align:center;cursor:pointer;
  }}
  .era-echo-direction-switch button small {{
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
    color:var(--muted);font-size:.44rem;font-weight:600;
  }}
  .era-echo-direction-switch button[aria-pressed="true"] {{
    border-bottom-color:#8b5e34;background:rgba(139,94,52,.055);
    color:var(--ink2);
  }}
  .era-echo-direction-switch button:focus-visible {{
    outline:3px solid rgba(208,139,69,.4);outline-offset:2px;
  }}
  .era-echo-direction-switch button:disabled {{
    opacity:.4;cursor:not-allowed;
  }}
  .era-echo-direction-status {{
    margin:0 0 3px;color:var(--muted);font-size:.48rem;text-align:center;
  }}
  .era-echo-direction-stage > [data-echo-view][hidden] {{
    display:none!important;
  }}
  .era-echo-gravity-key {{
    display:flex;justify-content:center;gap:12px;margin:5px 0 1px;
    color:var(--muted);font-size:.44rem;
  }}
  .era-echo-gravity-key span {{
    display:inline-flex;align-items:center;gap:4px;white-space:nowrap;
  }}
  .era-echo-gravity-key i {{
    width:9px;height:9px;border-radius:50%;
  }}
  .era-echo-gravity-key i.is-anchor {{
    background:#c99b6a;border:1px solid #8b5e34;
  }}
  .era-echo-gravity-key i.is-president {{
    background:#dbe7ec;border:1px solid #315f78;
  }}
  .era-echo-gravity-key i.is-topic {{ background:#9a7133; }}
  .era-echo-gravity-field {{
    position:relative;min-width:840px;
  }}
  .era-echo-gravity-readout {{
    position:absolute;z-index:3;left:50%;top:5px;transform:translateX(-50%);
    width:min(620px,72%);min-height:14px;margin:0;padding:4px 9px;
    border:1px solid rgba(139,94,52,.16);border-radius:999px;
    background:rgba(255,250,241,.88);backdrop-filter:blur(4px);
    color:#665c52;font:650 7.8px/1.25 system-ui;text-align:center;
    pointer-events:none;
  }}
  .era-echo-gravity-chart {{
    display:block;width:100%;min-width:840px;height:510px;
  }}
  .era-echo-gravity-center {{
    fill:#8b5e34;opacity:.34;
  }}
  .era-echo-gravity-center-label {{
    fill:#8d8277;font:650 7px/1 system-ui;letter-spacing:.03em;
  }}
  .era-echo-gravity-link {{
    stroke-width:var(--echo-width);stroke-linecap:round;opacity:.055;
    transition:opacity .16s ease,stroke-width .16s ease;
    pointer-events:none;
  }}
  .era-echo-gravity-link.is-president {{ stroke:#315f78; }}
  .era-echo-gravity-link.is-topic {{
    stroke:var(--echo-color);stroke-dasharray:2 3;
  }}
  .era-echo-gravity-link.is-active {{ opacity:.7; }}
  .era-echo-topic-president-link {{
    stroke:var(--echo-color);stroke-width:var(--echo-width);
    stroke-linecap:round;stroke-dasharray:2 3;opacity:0;
    transition:opacity .16s ease;pointer-events:none;
  }}
  .era-echo-topic-president-link.is-active {{ opacity:.52; }}
  .era-echo-gravity-field.has-active
    [data-echo-node]:not(.is-related):not(.is-active) {{
    opacity:.17;
  }}
  .era-echo-gravity-anchor,
  .era-echo-gravity-satellite {{
    cursor:help;transition:opacity .16s ease,filter .16s ease;
  }}
  .era-echo-gravity-anchor:focus,
  .era-echo-gravity-satellite:focus {{ outline:none; }}
  .era-echo-gravity-well {{
    fill:rgba(240,228,212,.26);stroke:#c8ab89;stroke-width:.8;
    stroke-dasharray:2 4;
  }}
  .era-echo-gravity-planet {{
    fill:rgba(240,228,212,.12);stroke:#8b5e34;stroke-width:2.8;
    transition:stroke-width .16s ease,filter .16s ease;
  }}
  .era-echo-gravity-anchor:hover .era-echo-gravity-planet,
  .era-echo-gravity-anchor:focus .era-echo-gravity-planet,
  .era-echo-gravity-anchor.is-active .era-echo-gravity-planet {{
    stroke:#315f78;stroke-width:4;filter:drop-shadow(0 2px 5px rgba(49,95,120,.24));
  }}
  .era-echo-gravity-anchor.is-unconnected {{ opacity:.42; }}
  .era-echo-gravity-anchor-label {{
    fill:#402b1c;font:700 9px/1 Georgia,serif;pointer-events:none;
  }}
  .era-echo-gravity-anchor-label tspan {{
    fill:#806b59;font:650 7px/1 system-ui;
  }}
  .era-echo-gravity-satellite circle {{
    stroke-width:1.7;transition:stroke-width .16s ease,filter .16s ease;
  }}
  .era-echo-gravity-satellite.is-president circle {{
    fill:rgba(219,231,236,.18);stroke:#315f78;
  }}
  .era-echo-gravity-satellite.is-topic circle {{
    fill:color-mix(in srgb,var(--echo-color) 72%,white);
    stroke:var(--echo-color);
  }}
  .era-echo-gravity-satellite:hover circle,
  .era-echo-gravity-satellite:focus circle,
  .era-echo-gravity-satellite.is-active circle {{
    stroke:#1e1b17;stroke-width:2.8;filter:brightness(1.04);
  }}
  .era-echo-gravity-topic-icon {{
    fill:#fff;font:700 8px/1 system-ui;pointer-events:none;
  }}
  .era-echo-gravity-satellite-label {{
    fill:#4d4741;font:700 7px/1 system-ui;paint-order:stroke;
    stroke:#fffaf1;stroke-width:2.5px;stroke-linejoin:round;
    pointer-events:none;
  }}
  .era-echo-empty {{ margin:8px 0;color:var(--muted);font-size:.56rem; }}
  .founding-step-footer {{ display:flex;align-items:center;justify-content:space-between;
                            gap:14px;margin:22px -24px -24px;padding:16px 24px;
                            border-top:1px solid var(--border);border-radius:0 0 17px 17px;
                            background:#eee7dc; }}
  .era-template .founding-step-footer {{ margin:22px -24px -24px; }}
  .founding-step-footer > span {{ color:var(--muted);font-size:.68rem;line-height:1.35; }}
  .founding-step-footer button,.founding-step-footer a {{
    appearance:none;display:inline-flex;align-items:center;gap:8px;flex:0 0 auto;
    padding:10px 14px;border:1px solid #8b5e34;border-radius:999px;
    background:#fffaf1;color:#5d3a1e;text-decoration:none;
    font:800 .7rem/1 system-ui;cursor:pointer;
  }}
  .founding-step-footer button:hover,.founding-step-footer a:hover {{
    background:#8b5e34;color:white;
  }}
  .founding-step-footer button:focus-visible,.founding-step-footer a:focus-visible {{
    outline:3px solid rgba(208,139,69,.4);outline-offset:2px;
  }}
  .founding-story-continue {{ background:#e8efe9; }}
  .founding-story-continue a {{ border-color:#477458;color:#31543e;background:#f8fcf8; }}
  .founding-story-continue a:hover {{ background:#477458; }}
  .founding-chart-note {{ margin:9px 0 0;color:var(--muted);font-size:.65rem;line-height:1.45; }}
  [data-era-screen="agenda"] .founding-chart-note {{ margin-top:4px; }}
  .founding-visual-question {{
    display:grid;grid-template-columns:minmax(0,1.15fr) minmax(250px,.85fr);
    align-items:end;gap:14px 28px;width:100%;max-width:none;
    margin:0 0 10px;padding:14px 16px 13px;
    border:1px solid var(--border);border-left:5px solid #8b5e34;
    border-radius:0 11px 11px 0;
    background:linear-gradient(100deg,#f4eadc 0%,#faf7f1 62%,#f8f4ed 100%);
  }}
  .founding-visual-question > div {{ min-width:0; }}
  .founding-visual-question > div > span {{
    display:block;margin:0 0 4px;color:#8b5e34;font-size:.54rem;font-weight:850;
    letter-spacing:.1em;text-transform:uppercase;
  }}
  .founding-visual-question h4 {{
    margin:0;color:var(--ink);
    font:680 clamp(1.15rem,2.2vw,1.55rem)/1.04 Georgia,serif;
  }}
  .founding-visual-question p {{
    margin:0;color:var(--muted);font-size:.6rem;line-height:1.45;text-align:right;
  }}
  .agenda-card-note {{
    margin:0 0 8px;color:#665b50;font-size:.5rem;line-height:1.4;
  }}
  .agenda-card-grid {{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
    align-items:stretch;gap:10px;
  }}
  .agenda-card-grid-scroll {{
    grid-template-columns:none;grid-auto-flow:column;
    grid-auto-columns:calc((100% - 20px)/3);
    max-width:100%;min-width:0;overflow-x:auto;overscroll-behavior-inline:contain;
    padding-bottom:7px;scroll-snap-type:inline proximity;
  }}
  .agenda-card-grid-scroll .agenda-card {{ scroll-snap-align:start; }}
  .agenda-card {{
    display:grid;grid-template-rows:auto 1fr;min-width:0;overflow:hidden;
    border:1px solid #d8cec1;border-radius:11px;background:#fffdf9;
    box-shadow:0 4px 14px rgba(61,45,31,.055);
  }}
  .agenda-card-president {{
    display:flex;align-items:center;gap:9px;padding:10px 11px 9px;
    border-bottom:1px solid #e4dbcf;background:#f3ece2;
  }}
  .agenda-card-president img {{
    width:40px;height:40px;flex:0 0 40px;object-fit:cover;
    border:1px solid #8b8175;border-radius:50%;background:#f8f4ed;
  }}
  .agenda-card-president div {{ min-width:0; }}
  .agenda-card-president h5 {{
    margin:0;color:var(--ink2);font:700 .64rem/1.08 Georgia,serif;
  }}
  .agenda-card-president p {{
    margin:3px 0 0;color:#756b61;font-size:.42rem;font-weight:650;
  }}
  .agenda-card-priorities {{
    display:grid;grid-template-rows:repeat(5,minmax(0,1fr));
    padding:0 11px;
  }}
  .agenda-card-priority {{
    display:flex;flex-direction:column;justify-content:center;min-width:0;
    padding:9px 0 10px;border-bottom:1px solid #e8e0d6;
  }}
  .agenda-card-priority:last-child {{ border-bottom:0; }}
  .agenda-card-priority-heading {{
    display:flex;align-items:baseline;justify-content:space-between;
    gap:8px;color:var(--ink2);font-size:.5rem;line-height:1.15;
  }}
  .agenda-card-priority-heading span {{
    min-width:0;font-weight:750;
  }}
  .agenda-card-priority-heading span b {{
    margin-right:3px;color:#8b5e34;font-size:.42rem;font-weight:850;
  }}
  .agenda-card-priority-heading strong {{
    flex:0 0 auto;color:var(--ink);font:700 .57rem/1 Georgia,serif;
  }}
  .agenda-card-bar {{
    display:block;height:8px;margin:5px 0 6px;overflow:hidden;
    border-radius:99px;background:#eee8df;
    box-shadow:inset 0 0 0 1px rgba(48,42,31,.08);
  }}
  .agenda-card-bar i {{
    display:block;width:var(--agenda-width);height:100%;
    border-radius:99px;background:var(--agenda-color);
  }}
  .agenda-card-description {{
    position:relative;min-width:0;
  }}
  .agenda-card-description > summary {{
    display:block;margin:0;overflow:hidden;list-style:none;color:#665b50;
    font-size:.41rem;line-height:1.33;text-overflow:ellipsis;white-space:nowrap;
    cursor:help;
  }}
  .agenda-card-description > summary::-webkit-details-marker {{ display:none; }}
  .agenda-card-description > summary::marker {{ content:""; }}
  .agenda-card-description > p {{
    position:absolute;z-index:4;right:0;bottom:calc(100% + 4px);left:0;
    display:none;margin:0;padding:7px 8px;border:1px solid rgba(37,59,71,.2);
    border-radius:6px;background:#253b47;color:#fff;
    box-shadow:0 4px 12px rgba(37,59,71,.22);
    font-size:.48rem;line-height:1.4;white-space:normal;
  }}
  .agenda-card-priority:first-child .agenda-card-description > p {{
    top:calc(100% + 4px);bottom:auto;
  }}
  .agenda-card-description:is(:hover,:focus-within,[open]) > p {{
    display:block;
  }}
  .agenda-card:has(.agenda-card-description:is(:hover,:focus-within,[open])) {{
    z-index:2;
  }}
  .agenda-card-grid-is-thin {{ opacity:.86; }}
  .agenda-method-note {{ margin-top:7px; }}
  .agenda-method-note summary {{
    color:#665b50;font-size:.52rem;font-weight:750;cursor:pointer;
  }}
  .agenda-method-note p {{
    margin:6px 0 0;color:var(--muted);font-size:.52rem;line-height:1.45;
  }}
  .agenda-thin-support {{
    margin-top:8px;padding:8px 10px;border:1px solid #d6c7b5;
    border-radius:9px;background:#f7f2ea;
  }}
  .agenda-thin-support > summary {{
    color:#6f4828;font-size:.57rem;font-weight:800;cursor:pointer;
  }}
  .agenda-thin-support > p {{
    margin:6px 0;color:var(--muted);font-size:.49rem;line-height:1.35;
  }}
  .founding-network-scroll,.founding-test-scroll {{ max-width:100%;overflow-x:auto;
                                                    overscroll-behavior-inline:contain; }}
  .founding-adversary-network,.founding-test-chart {{ display:block;width:100%;min-width:700px;height:auto; }}
  .founding-adversary-network .founding-network-edge {{
    fill:none;stroke-linecap:round;stroke-width:var(--adversary-edge-width);
    opacity:.42;cursor:pointer;
  }}
  .founding-adversary-network .founding-adversary-node {{ cursor:pointer; }}
  .founding-adversary-network.has-active .founding-network-edge,
  .founding-adversary-network.has-active .founding-adversary-node {{ opacity:.12; }}
  .founding-adversary-network.has-active .founding-network-edge:is(.is-active,.is-related) {{
    opacity:1;stroke-width:calc(var(--adversary-edge-width) * 1.35);
    filter:drop-shadow(0 0 1.5px rgba(44,37,29,.28));
  }}
  .founding-adversary-network.has-active .founding-adversary-node:is(.is-active,.is-related) {{
    opacity:1;
  }}
  .founding-adversary-network .founding-network-edge:focus-visible,
  .founding-adversary-network .founding-adversary-node:focus-visible {{
    outline:none;
  }}
  .founding-adversary-network .founding-network-edge:focus-visible {{
    filter:drop-shadow(0 0 2px rgba(35,59,71,.7));
  }}
  .founding-adversary-network .founding-adversary-node:focus-visible circle,
  .founding-adversary-network .founding-adversary-node.is-active circle {{
    stroke:#1e1b17;stroke-width:3.5;
  }}
  .founding-adversary-network circle {{ fill:var(--surface);stroke:#5d5144;stroke-width:2; }}
  .founding-adversary-network .founding-president-node {{ fill:#f4eadc;stroke:#8b5e34; }}
  .founding-adversary-network text {{ fill:var(--ink);font:650 12px/1 system-ui; }}
  .founding-adversary-network .founding-network-meta {{ fill:var(--muted);font-size:9px; }}
  .founding-network-key {{ display:flex;align-items:baseline;justify-content:space-between;
                           gap:10px;margin-bottom:5px;padding:8px 11px;border:1px solid var(--border);
                           border-radius:9px;background:#f8f4ed; }}
  .founding-network-key strong {{ font:650 .84rem/1.2 Georgia,serif; }}
  .founding-network-key span {{ max-width:570px;color:var(--muted);font-size:.58rem;
                                line-height:1.35;text-align:right; }}
  .founding-era-argument {{ margin:0 0 8px; }}
  .founding-era-argument header {{ display:grid;grid-template-columns:auto minmax(0,1fr);
                                   align-items:baseline;gap:3px 14px;padding:12px 15px;
                                   border-radius:11px;background:#253b47;color:white; }}
  .founding-era-argument header > span {{ color:#d7b78f;font-size:.6rem;font-weight:850;
                                         text-transform:uppercase;letter-spacing:.08em; }}
  .founding-era-argument h4 {{ margin:0;font:700 clamp(1.15rem,2.4vw,1.65rem)/1 Georgia,serif; }}
  .founding-era-argument p {{ grid-column:1/-1;margin:3px 0 0;color:#dce5e8;
                              font-size:.63rem;line-height:1.4; }}
  .founding-test-legend {{ display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;
                           list-style:none;margin:0 0 5px;padding:0; }}
  .founding-test-legend li {{ display:grid;grid-template-columns:18px auto minmax(0,1fr);
                              align-items:start;gap:5px;padding:7px 8px;border:1px solid var(--border);
                              border-top:3px solid var(--test-color);
                              border-radius:9px;background:var(--surface); }}
  .founding-test-legend i {{ width:18px;height:4px;border-radius:99px;background:var(--test-color); }}
  .founding-test-legend li > span {{ line-height:1; }}
  .founding-test-legend li > div {{ display:grid;gap:3px;min-width:0; }}
  .founding-test-legend strong {{ min-width:0;font-size:.64rem;line-height:1.25; }}
  .founding-test-legend b {{ grid-column:1/-1;margin-top:2px;
                             font:700 1rem/1 Georgia,serif; }}
  .founding-test-guides line {{ stroke:var(--border);stroke-width:1; }}
  .founding-test-guides text,.founding-test-chart g > text {{ fill:var(--ink);
                                                              font:650 12px/1 system-ui; }}
  .founding-test-guides .founding-test-date {{ fill:var(--muted);font-size:10px; }}
  .founding-test-ribbon {{ opacity:.32; }}
  .founding-test-chart circle {{ fill:var(--surface);stroke-width:2; }}
  .founding-next-era {{ display:flex;align-items:baseline;gap:8px 14px;margin-top:4px;
                        padding:9px 12px;border-left:4px solid #8b5e34;
                        border-radius:0 9px 9px 0;background:#f4eadc; }}
  .founding-next-era > span {{ color:#6f4828;font-size:.6rem;font-weight:850;
                               text-transform:uppercase;letter-spacing:.07em; }}
  .founding-next-era > strong {{ font:650 .9rem/1.2 Georgia,serif; }}
  .founding-problem-grid {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px; }}
  .founding-problem-grid-combined {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  .founding-problem {{ min-width:0;padding:18px;border:1px solid var(--border);border-radius:14px;
                       background:var(--surface);box-shadow:0 10px 30px rgba(48,42,31,.05); }}
  .founding-problem-sovereignty {{ border-top:5px solid #315f78; }}
  .founding-problem-external {{ border-top:5px solid #315f78; }}
  .founding-problem-capacity {{ border-top:5px solid #9a7133; }}
  .founding-problem-native {{ border-top:5px solid #a24f52; }}
  .founding-problem-value {{ margin:0 0 7px;font:750 clamp(2.7rem,7vw,5rem)/.9 Georgia,serif; }}
  .founding-problem h4 {{ margin:0 0 8px;font-size:.9rem;line-height:1.3; }}
  .founding-problem > p:not(.founding-problem-value) {{ margin:0;color:var(--muted);
                                                        font-size:.68rem;line-height:1.35; }}
  .founding-problem-breakout {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
                                gap:7px;margin:14px 0 0; }}
  .founding-problem-breakout div {{ padding:8px;background:#edf4f7;border-radius:8px; }}
  .founding-capacity-breakout div {{ background:#f5efe5; }}
  .founding-problem-breakout dt {{ color:var(--muted);font-size:.59rem;line-height:1.25; }}
  .founding-problem-breakout dd {{ margin:4px 0 0;font:700 1.25rem/1 Georgia,serif; }}
  .founding-overlap {{ margin:10px 0 0;padding:9px 12px;border-left:4px solid #315f78;
                       background:#edf4f7;border-radius:0 9px 9px 0;
                       color:var(--ink2);font-size:.69rem; }}
  .founding-era-facts {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
                         gap:8px;margin:12px 0 0; }}
  .founding-era-facts p {{ display:grid;grid-template-columns:auto 1fr;align-items:center;
                           gap:8px;margin:0;padding:10px 12px;border:1px solid var(--border);
                           border-radius:10px;background:#fffdf9; }}
  .founding-era-facts strong {{ font:700 1.5rem/1 Georgia,serif; }}
  .founding-era-facts span {{ color:var(--muted);font-size:.62rem;line-height:1.3; }}
  .founding-governance-filter {{ display:grid;gap:8px; }}
  .founding-governance-filter .sr-only {{
    position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
    clip:rect(0,0,0,0);white-space:nowrap;border:0;
  }}
  .founding-governance-selector {{
    display:flex;justify-content:safe center;gap:7px;max-width:100%;
    padding:2px;overflow-x:auto;overscroll-behavior-inline:contain;
  }}
  .founding-governance-filter[data-governance-density="dense"]
    .founding-governance-selector {{
      flex-wrap:wrap;overflow-x:visible;
  }}
  .founding-governance-president {{
    appearance:none;display:flex;align-items:center;gap:8px;min-width:145px;
    padding:7px 12px;border:1px solid var(--border);border-radius:999px;
    background:#f8f4ed;color:var(--muted);cursor:pointer;opacity:.58;
    transition:opacity .16s ease,background .16s ease,box-shadow .16s ease;
  }}
  .founding-governance-president img {{ width:40px;height:40px;object-fit:cover;
                                        border-radius:50%;filter:grayscale(.65);
                                        border:2px solid #b9aea1; }}
  .founding-governance-filter[data-governance-density="dense"]
    .founding-governance-president {{ min-width:130px;padding:6px 10px; }}
  .founding-governance-filter[data-governance-density="dense"]
    .founding-governance-president img {{ width:34px;height:34px; }}
  .founding-governance-president span {{ display:grid;gap:2px;text-align:left; }}
  .founding-governance-president strong {{ color:var(--ink2);font-size:.62rem;line-height:1; }}
  .founding-governance-president small {{ color:var(--muted);font-size:.47rem;line-height:1; }}
  .founding-governance-president[aria-pressed="true"] {{
    background:var(--surface);color:var(--ink);opacity:1;
    border-color:var(--president-color);
    box-shadow:0 3px 10px color-mix(in srgb,var(--president-color) 20%,transparent);
  }}
  .founding-governance-president[aria-pressed="true"] img {{
    filter:none;border-color:var(--president-color);
  }}
  .founding-governance-president:focus-visible {{
    outline:3px solid rgba(208,139,69,.5);outline-offset:2px;
  }}
  .founding-governance-stage {{ min-width:0; }}
  .founding-governance-panel {{ display:grid;grid-template-columns:minmax(0,1fr);gap:9px; }}
  .founding-governance-panel[hidden] {{ display:none; }}
  .founding-governance-distributions {{ display:grid;
                                        grid-template-columns:repeat(2,minmax(0,1fr));
                                        gap:9px;min-width:0; }}
  .founding-governance-distributions section {{
    min-width:0;margin-top:0;padding:12px;border:1px solid var(--border);border-radius:10px;
    background:rgba(255,255,255,.72);
  }}
  .founding-governance-distributions h4 {{
    margin:0 0 8px;color:#6f4828;font:680 .8rem/1.1 Georgia,serif;
  }}
  .founding-governance-bar {{ display:grid;
                              grid-template-columns:94px minmax(0,1fr) 42px;
                              align-items:center;gap:7px;min-height:54px; }}
  .founding-governance-bar + .founding-governance-bar {{
    border-top:1px solid #ece4d9;
  }}
  .founding-governance-channel {{ display:flex;align-items:center;gap:5px;min-width:0; }}
  .founding-governance-channel span {{ flex:0 0 auto;font-size:.78rem; }}
  .founding-governance-channel strong {{ min-width:0;color:var(--ink2);
                                         font-size:.56rem;line-height:1.15; }}
  .founding-governance-track {{ position:relative;height:42px;overflow:visible;
                                border:1px solid #d8cec0;border-radius:6px;background:#f4eee4; }}
  .founding-governance-filter[data-governance-density="dense"]
    .founding-governance-track {{ height:50px; }}
  .founding-governance-fill {{ display:block;width:var(--selected-share);height:100%;
                               border-radius:5px;
                               background:color-mix(in srgb,var(--president-color) 48%,white);
                               transition:width .22s ease; }}
  .founding-governance-benchmark {{
    position:absolute;z-index:2;top:calc(var(--benchmark-row) * 11px);
    left:clamp(10px,var(--benchmark-share),calc(100% - 10px));
    width:20px;height:20px;border-radius:50%;
    transform:translateX(calc(-50% + var(--benchmark-stack) * 4px));
    filter:grayscale(.22);opacity:.82;
  }}
  .founding-governance-filter[data-governance-density="dense"]
    .founding-governance-benchmark {{
      top:calc(var(--benchmark-row) * 10px);width:18px;height:18px;
  }}
  .founding-governance-benchmark::after {{
    content:attr(data-tooltip);position:absolute;z-index:8;left:50%;bottom:calc(100% + 7px);
    padding:5px 7px;border:1px solid rgba(37,59,71,.2);border-radius:6px;
    background:#253b47;color:#fff;box-shadow:0 4px 12px rgba(37,59,71,.22);
    font:750 .56rem/1 system-ui;letter-spacing:.01em;white-space:nowrap;
    opacity:0;pointer-events:none;transform:translate(-50%,3px);
    transition:opacity .12s ease,transform .12s ease;
  }}
  .founding-governance-benchmark:hover {{
    z-index:6;filter:none;opacity:1;
    transform:translateX(calc(-50% + var(--benchmark-stack) * 4px)) scale(1.12);
  }}
  .founding-governance-benchmark:hover::after {{
    opacity:1;transform:translate(-50%,0);
  }}
  .founding-governance-peer-portrait {{
    display:block;width:20px;height:20px;object-fit:cover;border:2px solid #f8f4ed;
    border-radius:50%;box-shadow:0 0 0 1px #756d63,0 1px 3px rgba(37,59,71,.22);
  }}
  .founding-governance-filter[data-governance-density="dense"]
    .founding-governance-peer-portrait {{ width:18px;height:18px; }}
  .founding-governance-value {{ color:var(--president-color);
                                font:760 .7rem/1 Georgia,serif;text-align:right; }}
  .founding-governance-takeaway {{
    display:flex;align-items:baseline;gap:8px 12px;min-width:0;
    padding:8px 11px;border-left:3px solid var(--president-color);
    border-radius:0 8px 8px 0;background:rgba(244,234,220,.62);
  }}
  .founding-governance-takeaway > span {{
    flex:0 0 auto;color:#8b5e34;font-size:.5rem;font-weight:850;
    text-transform:uppercase;letter-spacing:.08em;
  }}
  .founding-governance-takeaway p {{
    min-width:0;margin:0;color:var(--muted);font-size:.59rem;line-height:1.4;
  }}
  .founding-governance-takeaway p strong {{
    color:var(--ink2);font:680 .72rem/1.2 Georgia,serif;
  }}
  .founding-governance-takeaway b {{ color:var(--president-color);font-weight:800; }}
  .founding-attention-departures {{ display:grid;grid-template-columns:minmax(180px,.55fr)
                                    minmax(0,1.45fr);gap:7px;margin:0 0 7px;padding:7px;
                                    border:1px solid #d6b8ad;border-radius:12px;background:#f8ece8; }}
  .founding-departure-heading {{ display:grid;align-content:center;gap:4px;padding:5px 7px; }}
  .founding-departure-heading span {{ color:#7f3c40;font-size:.58rem;font-weight:850;
                                      text-transform:uppercase;letter-spacing:.07em; }}
  .founding-departure-heading strong {{ font:650 .88rem/1.25 Georgia,serif; }}
  .founding-departure-grid {{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px; }}
  .founding-departure-grid article {{ display:grid;
                                      grid-template-columns:auto minmax(0,1fr) auto;
                                      align-items:center;gap:5px 7px;padding:7px;
                                      border:1px solid #ddc9c1;
                                      border-radius:9px;background:#fff9f6; }}
  .founding-departure-icon {{ font-size:1.3rem;line-height:1; }}
  .founding-departure-grid article strong {{ font-size:.65rem;line-height:1.25; }}
  .founding-departure-grid article p {{ margin:2px 0 0;color:var(--muted);
                                        font-size:.52rem;line-height:1.25; }}
  .founding-departure-change {{ display:grid;
                                grid-template-columns:auto auto 1fr auto auto;
                                align-items:baseline;gap:2px 4px; }}
  .founding-departure-change b {{ font:700 .92rem/1 Georgia,serif; }}
  .founding-departure-change small {{ color:var(--muted);font-size:.51rem; }}
  .founding-departure-change i {{ justify-self:center;color:#a24f52;font-style:normal; }}
  .chart-title {{ margin:0 0 4px;font-weight:800;font-size:.9rem; }}
  .chart-subtitle {{ margin:0 0 12px;color:var(--muted);font-size:.75rem; }}
  .founding-thread-scroll {{ max-width:100%;overflow-x:auto;overflow-y:hidden;
                             padding-bottom:3px;overscroll-behavior-inline:contain; }}
  .founding-thread-intro {{ display:flex;justify-content:space-between;gap:10px;margin:0 0 8px;
                            padding:8px 11px;border:1px solid var(--border);
                            border-radius:10px;background:#f8f4ed;color:var(--ink2);
                            font-size:.66rem;align-items:center; }}
  .founding-thread-intro span {{ color:var(--muted); }}
  .founding-label-toggle {{ appearance:none;padding:6px 9px;border:1px solid #c8a783;
                            border-radius:999px;background:#fffaf1;color:#6f4828;
                            font:750 .6rem/1 system-ui;cursor:pointer;white-space:nowrap; }}
  .founding-label-toggle:focus-visible {{ outline:3px solid rgba(208,139,69,.4);
                                         outline-offset:2px; }}
  .founding-thread-grid {{ display:grid;grid-template-columns:64px repeat(9,minmax(66px,1fr));
                           min-width:700px;align-items:stretch; }}
  .founding-thread-grid.labels-open {{ grid-template-columns:minmax(285px,330px)
                                      repeat(9,minmax(94px,1fr));min-width:1220px; }}
  .founding-thread-corner,.founding-thread-era,.founding-thread-label,.founding-bubble-cell {{
    min-width:0;border-bottom:1px solid var(--border);
  }}
  .founding-thread-corner,.founding-thread-label {{ position:sticky;left:0;z-index:3;background:var(--page); }}
  .founding-thread-corner {{ display:flex;align-items:end;justify-content:center;padding:8px 5px;
                             color:var(--muted);font-size:.68rem;font-weight:800;
                             text-transform:uppercase;letter-spacing:.06em; }}
  .founding-thread-grid.labels-open .founding-thread-corner {{ justify-content:flex-start;
                                                               padding-inline:10px; }}
  .founding-thread-era {{ padding:5px 4px;text-align:center;background:#f8f4ed; }}
  .founding-thread-era strong,.founding-thread-era span {{ display:block; }}
  .founding-thread-era strong {{ min-height:2.3em;font-size:.58rem;line-height:1.15; }}
  .founding-thread-era span {{ color:var(--muted);font-size:.53rem; }}
  .founding-thread-label {{ display:flex;flex-direction:column;justify-content:center;align-items:center;
                            padding:8px 5px;border-left:5px solid var(--row-color); }}
  .founding-thread-label strong {{ display:flex;align-items:center;justify-content:center;
                                   max-width:290px;font-size:.76rem;line-height:1.3;
                                   white-space:normal; }}
  .founding-thread-icon {{ display:inline-grid;place-items:center;margin:0!important;
                           color:var(--ink)!important;font-size:1.25rem!important;line-height:1!important; }}
  .founding-thread-copy {{ display:none; }}
  .founding-thread-grid.labels-open .founding-thread-copy {{ display:inline; }}
  .founding-thread-grid.labels-open .founding-thread-label {{ align-items:stretch;padding:10px 12px; }}
  .founding-thread-grid.labels-open .founding-thread-label strong {{ justify-content:flex-start; }}
  .founding-thread-grid.labels-open .founding-thread-icon {{ margin-right:7px!important; }}
  .founding-thread-label > span.founding-thread-copy {{ margin-top:4px;color:var(--muted);
                                                        font-size:.64rem;line-height:1.35; }}
  .founding-thread-label small {{ margin-top:3px;color:#6f4828;font-size:.59rem; }}
  .founding-thread-highlight {{ background:#f8ece8; }}
  .founding-bubble-cell {{ display:grid;place-items:center;min-height:40px;background:var(--surface); }}
  .founding-bubble-cell.founding-thread-highlight-cell {{ background:#fcf4f1; }}
  .founding-bubble {{ position:relative;width:34px;height:34px;padding:0;border:0;border-radius:50%;
                      background:transparent;cursor:help; }}
  .founding-bubble::before {{ content:"";position:absolute;left:50%;top:50%;
                              width:var(--bubble-size);height:var(--bubble-size);
                              transform:translate(-50%,-50%);border:2px solid #fff;border-radius:50%;
                              background:var(--row-color);box-shadow:0 0 0 1px #423e38; }}
  .bubble-recurrent::before {{ background:repeating-linear-gradient(45deg,var(--row-color) 0 4px,
                                  color-mix(in srgb,var(--row-color) 55%,white) 4px 7px); }}
  .bubble-disappearing::before {{ background:radial-gradient(circle at 2px 2px,#fff 1px,
                                    transparent 1.5px),var(--row-color);background-size:6px 6px; }}
  .founding-zero {{ color:var(--muted);font-size:.7rem; }}
  .founding-thread-table {{ margin-top:6px;border:1px solid var(--border);border-radius:9px;padding:7px 10px; }}
  .founding-thread-table summary {{ cursor:pointer;font-size:.75rem;font-weight:780; }}
  .founding-thread-table table {{ min-width:1120px; }}
  .founding-thread-table small {{ color:var(--muted);font-size:.6rem; }}
  .founding-evidence-boundary {{ margin:22px 0 0;padding:15px 17px;border:1px solid var(--border);
                                border-radius:12px;background:#f8f4ed; }}
  .founding-evidence-boundary summary {{ cursor:pointer;color:#8b5e34;font-size:.68rem;
                                         font-weight:800;text-transform:uppercase;
                                         letter-spacing:.06em; }}
  .founding-evidence-boundary p {{ margin:7px 0 0;color:var(--muted);font-size:.73rem;line-height:1.55; }}
  .event-callouts {{ margin:10px 0 16px;padding:10px 12px;background:#f6f1e9;
                     border:1px solid #ded2c2;border-radius:10px; }}
  .event-callouts ul {{ list-style:none;display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
                        gap:7px;margin:0;padding:0; }}
  .event-callouts li {{ color:var(--ink2);font-size:.78rem;line-height:1.35; }}
  .event-callouts li span {{ display:block;color:#6f4828;font-weight:800;font-size:.7rem; }}
  .event-callouts p {{ margin:8px 0 0;color:var(--muted);font-size:.74rem; }}
  .inspector-row {{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:9px 0 16px; }}
  .inspector-tile {{
    min-width:0;border:1px solid var(--border);border-radius:9px;
    background:var(--surface);
  }}
  .inspector-tile summary {{ cursor:pointer;list-style:none;padding:8px 10px;font-size:.76rem;
                             font-weight:780;color:var(--ink2); }}
  .inspector-tile summary::-webkit-details-marker {{ display:none; }}
  .inspector-tile summary span {{ display:inline-grid;place-items:center;width:18px;height:18px;
                                  margin-right:5px;border-radius:5px;font-size:.68rem; }}
  .measure-tile {{ border-left:4px solid #315f78; }}
  .measure-tile summary span {{ background:#e4f0f5;color:#315f78; }}
  .evidence-tile {{ border-left:4px solid #8b5e34; }}
  .evidence-tile summary span {{ background:#f4eadc;color:#6f4828; }}
  .inspector-tile[open] {{ grid-column:1/-1; }}
  .inspector-panel {{ border-top:1px solid var(--border);padding:10px 13px 12px; }}
  .inspector-panel p {{ margin:5px 0;font-size:.8rem;line-height:1.5;max-width:62rem; }}
  .inspector-panel ol {{ margin:5px 0 8px;padding-left:20px;color:var(--ink2);font-size:.78rem; }}
  summary:focus-visible {{ outline:3px solid #d08b45;outline-offset:3px; }}
  .era-choices {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;
                  border:0;padding:0;margin:0 0 10px; }}
  .era-choices legend {{ padding:0 6px;font-size:.78rem;font-weight:800;color:var(--ink2); }}
  .era-chip {{ position:relative;display:flex;align-items:center;gap:9px;min-height:54px;
               padding:9px 10px;border:1px solid var(--border);border-radius:10px;
               background:#f2f1ee;color:var(--ink2);cursor:pointer;opacity:.56;
               transition:opacity .14s ease,background .14s ease,box-shadow .14s ease; }}
  .era-chip:has(input:checked) {{ opacity:1;background:var(--surface);
               border-color:var(--era-chip-color);
               box-shadow:inset 4px 0 0 var(--era-chip-color),0 2px 7px rgba(37,34,30,.07); }}
  .era-chip:has(input:focus-visible) {{ outline:3px solid #d08b45;outline-offset:2px; }}
  .era-chip input {{ position:absolute;width:1px;height:1px;opacity:0;pointer-events:none; }}
  .era-chip-mark {{ flex:0 0 17px;width:17px;height:17px;border:2px solid var(--era-chip-color);
                    border-radius:50%;background:transparent; }}
  .era-chip:has(input:checked) .era-chip-mark {{ background:var(--era-chip-color);
                    box-shadow:inset 0 0 0 3px white; }}
  .era-chip-copy,.era-chip-copy strong,.era-chip-copy small {{ display:block; }}
  .era-chip-copy strong {{ font-size:.74rem;line-height:1.1; }}
  .era-chip-copy small {{ margin-top:3px;color:var(--muted);font-size:.61rem; }}
  .era-chip .sr-only {{ position:absolute;width:1px;height:1px;padding:0;margin:-1px;
                       overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0; }}
  .filter-status {{ color:#5d5144;font-size:.8rem;font-weight:650; }}
  .president-table,.weather-table {{ margin:9px 0 16px;border:1px solid var(--border);
                                     border-radius:9px;padding:9px 11px; }}
  .naming-method {{ display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0; }}
  .naming-method > p {{ padding:12px;background:#f6f1e9;border:1px solid #ded2c2;border-radius:10px;
                        margin:0;font-size:.82rem; }}
  .naming-examples {{ grid-column:1/-1; }}
  .naming-examples h4 {{ margin-bottom:8px; }}
  .naming-examples blockquote {{ padding:12px 14px;border-left:3px solid #8b5e34;background:var(--surface);
                                  margin:8px 0;border-radius:0 9px 9px 0; }}
  .naming-examples blockquote p {{ margin:0;font-size:.82rem; }}
  .naming-examples cite {{ display:block;margin-top:6px;color:var(--muted);font-style:normal;font-size:.72rem; }}
  .summary-language-boundary {{ margin-top:38px;padding-top:26px;border-top:1px solid var(--border); }}
  .summary-language-boundary > h3,.summary-divisiveness > h3 {{ margin:4px 0 8px; }}
  .summary-language-boundary > p:not(.summary-language-kicker) {{ max-width:780px;color:var(--ink2); }}
  .summary-language-kicker {{ margin:0;color:#8b5e34;font-size:.7rem;font-weight:850;
                              letter-spacing:.08em; }}
  .summary-language-stage {{ margin-top:16px;padding:15px;border:1px solid var(--border);
                             border-radius:14px;background:#f8f5ef; }}
  .summary-language-stage .stage-status {{ margin-bottom:4px; }}
  .summary-language-stage .chart-scroll {{ background:var(--surface);border-radius:10px; }}
  .summary-line-status {{ margin:8px 2px 0;color:var(--muted);font-size:.72rem;line-height:1.4; }}
  .summary-language-takeaways {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
                                 gap:9px;margin:12px 0 16px; }}
  .summary-language-takeaways article {{ padding:11px 12px;border:1px solid var(--border);
                                         border-radius:10px;background:var(--surface); }}
  .summary-language-takeaways strong,.summary-language-takeaways span {{ display:block; }}
  .summary-language-takeaways strong {{ color:var(--ink);font-size:.77rem; }}
  .summary-language-takeaways span {{ margin-top:4px;color:var(--muted);
                                      font-size:.72rem;line-height:1.42; }}
  .summary-divisiveness {{ margin:34px 0 0;padding:22px;border:1px solid #cbb89f;
                           border-radius:15px;background:#f4ede3; }}
  .summary-divisiveness-verdict {{ max-width:800px;margin:6px 0 17px;color:var(--ink2);
                                   font-size:1rem;line-height:1.58; }}
  .summary-divisiveness-verdict strong {{ color:#6f4828; }}
  .summary-divisiveness-grid {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
                                gap:10px; }}
  .summary-divisiveness-card {{ min-width:0;padding:14px;border:1px solid #d8cab8;
                                border-radius:11px;background:var(--surface); }}
  .summary-divisiveness-card.is-strong {{ border-top:4px solid #315f78; }}
  .summary-divisiveness-card > p:first-child {{ margin:0 0 7px;color:#8b5e34;
                                                font-size:.64rem;font-weight:850;
                                                letter-spacing:.06em; }}
  .summary-divisiveness-card > strong {{ display:block;color:var(--ink);
                                         font:700 2rem/1 Georgia,serif; }}
  .summary-divisiveness-card > strong small {{ color:var(--muted);font-size:.8rem; }}
  .summary-divisiveness-card h4 {{ margin:8px 0 6px;font-size:.86rem; }}
  .summary-divisiveness-card h4 + p {{ margin:0;color:var(--ink2);
                                      font-size:.76rem;line-height:1.5; }}
  .summary-divisiveness-null {{ margin:13px 0 0;padding:11px 13px;border-left:4px solid #8b5e34;
                                border-radius:0 8px 8px 0;background:#fffaf1;color:var(--ink2);
                                font-size:.8rem;line-height:1.5; }}
  header h1 {{ max-width:850px;font-size:clamp(2.7rem,7vw,5.7rem);line-height:.92;letter-spacing:-.055em; }}
  .hero-subheadline {{ max-width:830px;margin:.85rem 0 0;color:#6f4828;
                       font:650 clamp(1.08rem,2.2vw,1.6rem)/1.25 Georgia,serif; }}
  .era-route {{ display:flex;gap:8px;overflow-x:auto;padding:5px 0 12px;margin:22px 0 2px;
                scrollbar-width:thin; }}
  .era-route a {{ flex:0 0 auto;min-width:158px;background:var(--surface);border:1px solid var(--border);
                  border-radius:11px;padding:11px 13px;color:var(--ink2);text-decoration:none;
                  font-size:.82rem;font-weight:650;line-height:1.25; }}
  .era-route a:hover {{ border-color:#8b5e34; }}
  .era-route span {{ display:block;color:#8b5e34;font-size:.72rem;font-weight:800;
                     letter-spacing:.05em;margin-bottom:3px; }}
  .scroll-cue {{ display:inline-flex;align-items:center;gap:8px;margin-top:20px;color:var(--muted);font-size:.84rem;text-decoration:none; }}
  .scroll-cue::after {{ content:"↓";display:inline-block;animation:story-bob 1.8s ease-in-out infinite; }}
  .story-meter {{ position:sticky;top:var(--global-nav-height,48px);z-index:20;background:color-mix(in srgb,var(--page) 92%,transparent);backdrop-filter:blur(9px);border-bottom:1px solid var(--border);padding:7px max(18px,calc((100vw - 1160px)/2)); }}
  .story-meter-track {{ height:3px;background:var(--grid);overflow:hidden;border-radius:99px; }}
  .story-meter-fill {{ display:block;height:100%;width:0;background:linear-gradient(90deg,#8b5e34,#27678f);transition:width .16s linear; }}
  .story-meter-label {{ display:flex;align-items:baseline;gap:10px;min-width:0;margin-top:5px;
                        white-space:nowrap;overflow:hidden; }}
  .story-meter-label strong {{ flex:0 0 auto;color:var(--ink);
                               font:680 .96rem/1.25 Georgia,serif; }}
  .story-meter-label span {{ min-width:0;color:var(--muted);font-size:.65rem;
                             overflow:hidden;text-overflow:ellipsis; }}
  main {{ position:relative;overflow-x:visible; }}
  .story-section {{ position:relative;padding-top:26px;margin-top:34px;scroll-margin-top:calc(var(--global-nav-height,48px) + 48px); }}
  .section-deck {{ font-size:1.04rem;line-height:1.72; }}
  .js-motion .story-section {{ opacity:0;transform:translateY(28px);transition:opacity .7s ease,transform .7s cubic-bezier(.2,.75,.25,1); }}
  .js-motion .story-section.is-visible {{ opacity:1;transform:none; }}
  @keyframes story-bob {{ 0%,100%{{transform:translateY(0)}}50%{{transform:translateY(5px)}} }}
  @media (max-width:800px) {{ .story-meter{{padding-inline:16px}} }}
  @media (max-width:1050px) {{
    .founding-test-legend{{grid-template-columns:repeat(2,minmax(0,1fr))}}
  }}
  .reduced-motion .scroll-cue::after {{ animation:none; }}
  .reduced-motion .story-meter-fill {{ transition:none; }}
  .reduced-motion .founding-governance-fill,
  .reduced-motion .founding-governance-president {{ transition:none; }}
  @media (prefers-reduced-motion:reduce) {{ .scroll-cue::after{{animation:none}} .story-meter-fill{{transition:none}} .founding-governance-fill,.founding-governance-president{{transition:none}} .era-defined-combined-series,.era-defined-combined-line{{transition:none}} .js-motion .story-section{{opacity:1;transform:none;transition:none}} }}
  @media (forced-colors:active) {{
    .summary-register-bridge,.summary-register-sequence li,.era-select-bar{{
      border-color:CanvasText;background:Canvas;box-shadow:none;forced-color-adjust:auto}}
    .summary-register-kicker,.summary-register-sequence span,
    .summary-register-sequence li + li::before{{color:CanvasText}}
    .register-view-switch button{{border-color:ButtonText;background:ButtonFace;color:ButtonText}}
    .register-view-switch button[aria-pressed="true"]{{border-color:Highlight;background:Highlight;
      color:HighlightText}}
    .register-view-switch button:focus-visible,.era-select-control select:focus-visible{{
      outline-color:Highlight}}
    .conflict-line-buttons button{{border-color:ButtonText;background:ButtonFace;color:ButtonText!important}}
    .conflict-line-buttons button[aria-pressed="true"]{{border-color:Highlight;background:Highlight;
      color:HighlightText!important;box-shadow:none}}
    .conflict-line-buttons button:focus-visible{{outline-color:Highlight}}
    .conflict-category-guide,.conflict-category-guide > summary{{border-color:CanvasText;
      background:Canvas;color:CanvasText}}
    .conflict-category-guide > summary::marker{{color:CanvasText}}
    .conflict-category-guide > summary:focus-visible{{outline-color:Highlight}}
    .conflict-portrait-border-key{{border-color:CanvasText;background:Canvas;color:CanvasText}}
    .conflict-portrait-border-key i{{border-color:CanvasText;background:Canvas}}
    .summary-language-stage,.summary-language-takeaways article,.summary-divisiveness,
    .summary-divisiveness-card,.summary-divisiveness-null{{border-color:CanvasText;
      background:Canvas;color:CanvasText;box-shadow:none}}
  }}
  @media (max-width:960px) {{
    .era-select-bar.has-register-switch{{grid-template-columns:auto minmax(270px,1fr)}}
    .era-select-bar.has-register-switch .filter-status{{grid-column:1/-1}}
  }}
  @media (max-width:960px) {{
    .conflict-atlas-heading{{grid-template-columns:minmax(0,1fr) minmax(230px,290px);
                             gap:10px;padding:14px 12px 9px}}
    .conflict-treatment{{max-width:290px}}
    .conflict-atlas-deck{{padding:0 12px 9px}}
    .conflict-atlas-deck span{{display:block;margin:2px 0 0}}
    .conflict-category-key{{grid-template-columns:repeat(5,minmax(0,1fr));padding:0 12px 9px}}
    .conflict-column-head,.conflict-president-row{{
      grid-template-columns:minmax(145px,.78fr) minmax(180px,1fr) minmax(245px,1.35fr);
      column-gap:10px}}
    .conflict-column-head{{display:grid;padding:6px 12px}}
    .conflict-era > summary{{padding:6px 10px}}
    .conflict-president-row{{padding:7px 12px}}
    .conflict-evidence-grid{{grid-template-columns:1fr}}
  }}
  @media (max-width:700px) {{ .ai-bridge {{ grid-template-columns:1fr 1fr; }}
    .summary-receipt-grid{{grid-template-columns:1fr}}
    .summary-shift-grid{{grid-template-columns:1fr 1fr}}
    .summary-communication-panel{{padding:14px}}
    .summary-communication-panel > header{{grid-template-columns:1fr;align-items:start}}
    .summary-communication-panel > header > strong{{justify-self:start;white-space:normal}}
    .summary-communication-key{{grid-template-columns:1fr 1fr}}
    .summary-communication-panel .chart{{min-width:820px}}
    .summary-register-bridge{{padding:17px 15px}}
    .summary-register-sequence{{grid-template-columns:1fr;gap:20px}}
    .summary-register-sequence li{{min-height:58px}}
    .summary-register-sequence li + li::before{{content:"↓";left:50%;top:-18px;
      transform:translateX(-50%)}}
    .summary-language-stage .chart{{min-width:760px}}
    .era-select-bar,.era-select-bar.has-register-switch{{grid-template-columns:1fr;align-items:start}}
    .era-select-bar.has-register-switch .filter-status{{grid-column:auto}}
    .register-view-switch{{flex-wrap:wrap}}
    .era-select-control{{grid-template-columns:1fr}}
    .era-select-control select{{max-width:none}}
    .temporal-portrait-heading{{padding-inline:14px}}
    .temporal-denominator{{padding-inline:14px}}
    .summary-temporal-explorer{{padding-inline:14px}}
    .summary-temporal-explorer .chart-scroll{{margin-inline:-14px}}
    .conflict-graph-heading{{grid-template-columns:1fr;padding:14px 12px 9px}}
    .conflict-graph-heading .conflict-treatment{{max-width:none}}
    .conflict-graph-deck{{padding:0 12px 10px}}
    .conflict-graph-deck span{{display:block;margin:2px 0 0}}
    .conflict-question-group{{padding:14px 10px}}
    .conflict-question-group + .conflict-question-group{{padding-top:20px}}
    .conflict-measure-panel > header{{gap:10px;padding-inline:10px}}
    .conflict-denominator{{padding-inline:10px}}
    .communication-mosaic{{padding:14px}}
    .communication-mosaic-key{{grid-template-columns:1fr 1fr}}
    .communication-mosaic-strip{{grid-template-columns:repeat(9,170px)}}
    .communication-mosaic-evidence,.communication-audit-grid{{grid-template-columns:1fr}}
    .summary-chapter{{margin-bottom:62px}}
    .summary-chapter-route{{top:var(--global-nav-height,48px)}}
    .message-compare,.receipt-grid{{grid-template-columns:1fr}}
    .message-measures{{grid-template-columns:1fr 1fr}}
    .tradeoff-summary,.naming-method{{grid-template-columns:1fr}}
    .summary-language-takeaways,.summary-divisiveness-grid{{grid-template-columns:1fr}}
    .naming-examples{{grid-column:1}}
    .era-choices{{grid-template-columns:1fr 1fr}}
    .inspector-row{{grid-template-columns:minmax(0,1fr)}}
    .stage-narrative{{grid-template-columns:1fr}}
    .coverage-row{{grid-template-columns:1fr}} .coverage-row small{{grid-column:1}}
    .coverage-ribbons{{height:64px}}
    .era-template-heading,.era-template-grid{{grid-template-columns:1fr}}
    .era-template-heading p{{justify-items:start}}
    .era-visualize-heading{{grid-template-columns:1fr}}
    .era-visualize-heading p{{text-align:left}}
    .era-template-grid{{grid-auto-rows:auto}}
    .era-template-grid > article{{grid-column:1}}
    .era-reference-lane{{grid-template-columns:minmax(102px,.68fr)
      minmax(0,1.32fr)}}
    .founding-panel-heading{{grid-template-columns:1fr}}
    .founding-visual-question{{grid-template-columns:1fr;gap:7px}}
    .founding-visual-question p{{text-align:left}}
    .era-context-question{{grid-template-columns:1fr;gap:5px}}
    .agenda-card-grid{{grid-template-columns:1fr}}
    .agenda-card-grid-scroll{{grid-template-columns:none;
                              grid-auto-columns:min(100%,326px)}}
    .era-defined-combined-key{{grid-template-columns:repeat(2,minmax(0,1fr))}}
    .era-defined-trajectory-key{{justify-content:flex-start;flex-wrap:wrap}}
    .era-defined-trajectory-head{{display:none}}
    .era-defined-trajectory-row{{grid-template-columns:1fr;gap:7px}}
    .era-defined-trajectory-axis{{grid-template-columns:1fr;padding-left:11px}}
    .era-defined-trajectory-axis > span{{display:none}}
    .expansion-defined-phase-grid,
    .expansion-defined-events{{grid-template-columns:repeat(2,minmax(0,1fr))}}
    .era-echo-gravity-key{{justify-content:flex-start;flex-wrap:wrap}}
    .era-echo-receipts{{grid-template-columns:1fr}}
    .founding-next-era{{grid-template-columns:1fr}}
    .founding-network-key{{align-items:flex-start;flex-direction:column}}
    .founding-network-key span{{text-align:left}}
    .founding-problem-grid{{grid-template-columns:1fr}}
    .founding-era-facts{{grid-template-columns:1fr}}
    .founding-attention-departures{{grid-template-columns:1fr}}
    .founding-governance-selector{{justify-content:flex-start}}
    .founding-governance-president{{min-width:calc(50% - 4px)}}
    .founding-governance-distributions{{grid-template-columns:1fr}}
    .founding-governance-takeaway{{align-items:flex-start;flex-direction:column;gap:4px}}
    .expansion-story-figure{{padding:14px 8px}}
  }}
  @media (max-width:820px) {{
    .founding-chapter{{width:calc(100vw - 28px);
                       margin-left:calc((100% - (100vw - 28px))/2)}}
    .founding-movement-route button{{padding:11px 9px}}
    .founding-movement-route small{{display:none}}
    .founding-panel:not(.era-template),.era-template{{padding:17px}}
    .founding-step-footer,.era-template .founding-step-footer{{margin:18px -17px -17px;
                                                               padding:14px 17px}}
    .story-meter-label span{{display:none}}
  }}
  @media (max-width:580px) {{
    .conflict-atlas{{overflow-x:auto;overscroll-behavior-inline:contain;scrollbar-width:thin}}
    .conflict-atlas-heading{{grid-template-columns:1fr}}
    .conflict-treatment{{max-width:none}}
    .conflict-category-key,.conflict-target-key .conflict-category-key{{grid-template-columns:1fr 1fr}}
    .conflict-line-picker{{align-items:flex-start}}
    .conflict-line-picker-label,.conflict-line-status{{flex-basis:100%}}
    .conflict-graph-deck span{{white-space:normal}}
    .conflict-column-head,.conflict-era-list{{min-width:610px}}
    .conflict-column-head > span:first-child{{position:sticky;left:0;z-index:3;
                                              background:#e8e1d7}}
    .conflict-president-identity{{position:sticky;left:12px;z-index:2;
                                  background:#fffdf9;box-shadow:8px 0 10px -10px rgba(48,42,31,.55)}}
    .conflict-president-row:hover .conflict-president-identity{{background:#fffaf1}}
    .conflict-measure-panel > header{{grid-template-columns:1fr}}
    .conflict-measure-panel > header > strong{{justify-self:start}}
    .founding-movement-route strong{{font-size:.68rem}}
    .founding-screen-tabs{{display:grid;grid-template-columns:1fr;width:100%;border-radius:12px}}
    :is([data-era-visualize],[data-era-contextualize])
      .founding-screen-tabs{{display:grid;
                                               grid-template-columns:repeat(2,minmax(0,1fr));
                                               width:100%;margin-bottom:7px;gap:2px}}
    :is([data-era-visualize],[data-era-contextualize])
      .founding-screen-tabs button{{min-height:34px;padding:7px 4px;
                                                      border-right:0;
                                                      border-bottom:2px solid transparent;
                                                      text-align:center;white-space:normal}}
    :is([data-era-visualize],[data-era-contextualize])
      .founding-screen-tabs button[aria-selected="true"]{{
      border-bottom-color:#8b5e34;
    }}
    .founding-screen-tabs button{{border-radius:8px;text-align:left}}
    .founding-test-legend{{grid-template-columns:1fr}}
    .founding-thread-intro{{flex-wrap:wrap}}
    .founding-agenda-key,.founding-governance-key{{align-items:flex-start;flex-direction:column}}
    .founding-agenda-key span,.founding-governance-key span{{text-align:left}}
    .era-style-scroll,.founding-departure-grid{{grid-template-columns:1fr}}
    .founding-step-footer{{align-items:flex-start;flex-direction:column}}
  }}
</style>
</head>
<body{body_class}>
{header_html}
{meter_html}
<main>
{sections_html}
{after_sections}
</main>
<footer>
  <p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs,
  University of Virginia</a> (speeches are in the public domain; collection curated by
  Miller Center staff). Analysis &amp; code:
  <a href="https://github.com/jacobfulfyll/presidential_profiles">jacobfulfyll/presidential_profiles</a>.
  Originally a 2019 Galvanize capstone, rebuilt in 2026.</p>
  <p><a href="methodology.html">Illustrated AI labeling methodology, complete taxonomy,
  prompts, agreement audit, and limitations →</a></p>
  <p>Method note: most timelines are 5-year centered rolling rates weighted by word count,
  through April 2026, with points under 20,000 words omitted. The Tomorrow/Yesterday view
  is a trailing four-year average of annual rates with a 10,000-word window floor. The corpus
  before ~1790 is a handful of speeches, and one personal inaugural should not set a
  national trend line. Dots are per president per year: 46 years have more than one
  president speaking, and the corpus files a few famous pre-presidency speeches
  (Nixon's Checkers speech, Reagan's "A Time for Choosing") under the later president.
  This is the formal register only - major prepared addresses; rallies, debates, and
  social media are outside the corpus, so every trend here is presidents at their most
  prepared.</p>
</footer>
{legacy_summary_redirect}
<script>
  const FIGS = {json_for_script(fig_json)};
  const reduceMotion = (
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
    || new URLSearchParams(window.location.search).get("motion") === "reduce"
  );
  if (reduceMotion) document.documentElement.classList.add("reduced-motion");
  else document.documentElement.classList.add("js-motion");
  const plotted = new WeakSet();
  function setupConflictTargetHighlight(chart) {{
    if (chart.dataset.conflictHighlightReady === "true") return;
    const picker = document.querySelector("[data-conflict-line-picker]");
    if (!picker) return;
    const buttons = [...picker.querySelectorAll("[data-conflict-line]")];
    const status = picker.querySelector(".conflict-line-status");
    const figure = FIGS.summary_conflict_targets;
    const keys = figure.data.map(trace => trace.meta).filter(Boolean);
    const labels = new Map(figure.data.map(trace => [trace.meta, trace.name]));
    const callouts = (figure.layout.annotations || []).map((annotation, index) => ({{
      index,
      key: (annotation.name || "").replace("conflict-turn::", ""),
    }})).filter(item => keys.includes(item.key));
    let pinned = "__all__";
    chart.dataset.conflictHighlightReady = "true";

    function applyHighlight(key) {{
      keys.forEach((traceKey, index) => {{
        const active = key === "__all__" || key === traceKey;
        Plotly.restyle(chart, {{
          opacity: active ? 1 : .14,
          "line.width": key === traceKey ? 4.5 : key === "__all__" ? 2.5 : 1.4,
          "marker.line.width": key === traceKey ? 2 : 1.4,
        }}, [index]);
      }});
      const annotationUpdate = {{}};
      callouts.forEach(item => {{
        annotationUpdate[`annotations[${{item.index}}].opacity`] = (
          key === "__all__" || key === item.key ? 1 : .26
        );
      }});
      Plotly.relayout(chart, annotationUpdate);
    }}
    function syncPinnedState() {{
      buttons.forEach(button => {{
        button.setAttribute(
          "aria-pressed", String(button.dataset.conflictLine === pinned)
        );
      }});
      if (status) status.textContent = pinned === "__all__"
        ? "All five target lines emphasized. Hover a line or choose one to isolate it."
        : `${{labels.get(pinned)}} highlighted; the other four lines remain visible at low emphasis. Choose All or press Escape to reset.`;
      applyHighlight(pinned);
    }}
    function setPinned(key) {{
      pinned = key === pinned && key !== "__all__" ? "__all__" : key;
      syncPinnedState();
    }}
    buttons.forEach(button => {{
      const key = button.dataset.conflictLine;
      button.addEventListener("click", () => setPinned(key));
      button.addEventListener("pointerenter", () => applyHighlight(key));
      button.addEventListener("pointerleave", () => applyHighlight(pinned));
      button.addEventListener("focus", () => applyHighlight(key));
      button.addEventListener("blur", () => applyHighlight(pinned));
    }});
    picker.addEventListener("keydown", event => {{
      if (event.key !== "Escape") return;
      pinned = "__all__";
      syncPinnedState();
    }});
    chart.on("plotly_hover", event => {{
      const point = event.points.find(item => keys.includes(item.data.meta));
      if (point) applyHighlight(point.data.meta);
    }});
    chart.on("plotly_unhover", () => applyHighlight(pinned));
    chart.on("plotly_click", event => {{
      const point = event.points.find(item => keys.includes(item.data.meta));
      if (point) setPinned(point.data.meta);
    }});
    syncPinnedState();
  }}
  function setupContainedPortraitHover(chart) {{
    if (chart.dataset.containedHoverReady === "true") return;
    const scroller = chart.closest(".chart-scroll");
    if (!scroller) return;
    chart.dataset.containedHoverReady = "true";
    function revealHoverLabel(point, attempt = 0) {{
      requestAnimationFrame(() => {{
        const label = chart.querySelector(".hoverlayer .hovertext");
        if (!label) return;
        const labelBounds = label.getBoundingClientRect();
        const scrollBounds = scroller.getBoundingClientRect();
        const inset = 8;
        const rightOverflow = labelBounds.right - (scrollBounds.right - inset);
        const leftOverflow = (scrollBounds.left + inset) - labelBounds.left;
        let shifted = false;
        if (rightOverflow > 0) {{
          scroller.scrollLeft += Math.ceil(rightOverflow);
          shifted = true;
        }} else if (leftOverflow > 0) {{
          scroller.scrollLeft -= Math.ceil(leftOverflow);
          shifted = true;
        }}
        if (!shifted) return;
        Plotly.Fx.hover(chart, [{{
          curveNumber: point.curveNumber,
          pointNumber: point.pointNumber,
        }}]);
        if (attempt < 3) revealHoverLabel(point, attempt + 1);
      }});
    }}
    chart.on("plotly_click", event => {{
      const point = event.points?.[0];
      if (point) revealHoverLabel(point);
    }});
  }}
  const summaryLanguageFigures = new Set([
    "naming_progressive", "summary_modals", "summary_pronouns",
  ]);
  const summaryLanguageBaseWidths = new Map(
    [...summaryLanguageFigures].map(key => [
      key,
      Object.freeze(
        (FIGS[key]?.data || []).map(trace => Number(trace.line?.width || 2.5))
      ),
    ])
  );
  function setupSummaryLanguageHighlight(chart) {{
    if (chart.dataset.summaryLanguageHighlightReady === "true") return;
    chart.dataset.summaryLanguageHighlightReady = "true";
    chart.tabIndex = 0;
    let pinned = null;
    const status = chart.closest(".summary-language-stage")
      ?.querySelector(".summary-line-status");
    function figure() {{ return FIGS[chart.dataset.fig]; }}
    function applyHighlight(activeIndex) {{
      const traces = figure()?.data || [];
      const baseWidths = summaryLanguageBaseWidths.get(chart.dataset.fig) || [];
      traces.forEach((trace, index) => {{
        const baseWidth = baseWidths[index] || 2.5;
        const active = activeIndex == null || activeIndex === index;
        Plotly.restyle(chart, {{
          opacity: active ? 1 : .14,
          "line.width": activeIndex === index ? baseWidth + 2.2 : baseWidth,
        }}, [index]);
      }});
    }}
    function restoreAll() {{
      pinned = null;
      applyHighlight(null);
      if (status) status.textContent = (
        "Hover any line to highlight its full trajectory; click or tap to pin it, "
        + "and press Escape to restore every line."
      );
    }}
    chart.on("plotly_hover", event => {{
      const point = event.points?.[0];
      if (!point) return;
      applyHighlight(point.curveNumber);
      if (status) status.textContent = `${{point.data.name}} highlighted.`;
    }});
    chart.on("plotly_unhover", () => {{
      if (pinned == null) restoreAll();
      else applyHighlight(pinned);
    }});
    chart.on("plotly_click", event => {{
      const point = event.points?.[0];
      if (!point) return;
      pinned = pinned === point.curveNumber ? null : point.curveNumber;
      if (pinned == null) restoreAll();
      else {{
        applyHighlight(pinned);
        if (status) status.textContent = (
          `${{point.data.name}} pinned; press Escape to restore every line.`
        );
      }}
    }});
    chart.addEventListener("keydown", event => {{
      if (event.key === "Escape") restoreAll();
    }});
    chart.addEventListener("summary-language-reset", restoreAll);
    restoreAll();
  }}
  function renderChart(el) {{
    if (plotted.has(el)) return;
    const fig = FIGS[el.dataset.fig];
    Plotly.newPlot(el, fig.data, fig.layout,
                   {{displayModeBar: false, responsive: true}}).then(() => {{
      if (el.dataset.fig === "summary_conflict_targets") {{
        setupConflictTargetHighlight(el);
      }}
      if (el.dataset.fig === "summary_conflict_frame_portraits") {{
        setupContainedPortraitHover(el);
      }}
      if (summaryLanguageFigures.has(el.dataset.fig)) {{
        setupSummaryLanguageHighlight(el);
      }}
    }});
    plotted.add(el);
  }}
  const sections = [...document.querySelectorAll("main > section")];
  sections.forEach(section => {{
    section.classList.add("story-section");
  }});
  if ("IntersectionObserver" in window) {{
    const observer = new IntersectionObserver(entries => {{
      for (const entry of entries) {{
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("is-visible");
        entry.target.querySelectorAll(".chart").forEach(renderChart);
        observer.unobserve(entry.target);
      }}
    }}, {{rootMargin:"180px 0px 220px",threshold:.03}});
    sections.forEach(section => observer.observe(section));
  }} else {{
    sections.forEach(section => {{
      section.classList.add("is-visible");
      section.querySelectorAll(".chart").forEach(renderChart);
    }});
  }}
  const stageContainers = [...document.querySelectorAll(".story-stage")];
  function activateStage(container, index) {{
    const announcements = JSON.parse(container.dataset.stageAnnouncements || "[]");
    const buttons = [...container.querySelectorAll("[data-stage-button]")];
    const steps = [...container.querySelectorAll("[data-stage-step]")];
    buttons.forEach((button, position) => {{
      button.setAttribute("aria-pressed", String(position === index));
    }});
    steps.forEach((step, position) => {{
      step.classList.toggle("is-active", position === index);
    }});
    const status = container.querySelector(".stage-status");
    if (status && announcements[index]) status.textContent = announcements[index];
    if (container.dataset.stageMode === "chart") {{
      const keys = JSON.parse(container.dataset.stageFigs || "[]");
      const chart = container.querySelector(".staged-chart");
      if (!chart || !keys[index]) return;
      chart.dataset.fig = keys[index];
      if (plotted.has(chart)) {{
        const fig = FIGS[keys[index]];
        Plotly.react(chart, fig.data, fig.layout,
                     {{displayModeBar:false,responsive:true}}).then(() => {{
          if (summaryLanguageFigures.has(keys[index])) {{
            setupSummaryLanguageHighlight(chart);
            chart.dispatchEvent(new CustomEvent("summary-language-reset"));
          }}
        }});
      }}
    }} else {{
      container.querySelectorAll("[data-stage-panel]").forEach((panel, position) => {{
        panel.hidden = position !== index;
      }});
    }}
  }}
  stageContainers.forEach(container => {{
    const buttons = [...container.querySelectorAll("[data-stage-button]")];
    buttons.forEach((button, index) => {{
      button.addEventListener("click", () => activateStage(container, index));
      button.addEventListener("keydown", event => {{
        if (!["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
        event.preventDefault();
        let target = index + (event.key === "ArrowRight" ? 1 : -1);
        if (event.key === "Home") target = 0;
        if (event.key === "End") target = buttons.length - 1;
        target = (target + buttons.length) % buttons.length;
        buttons[target].focus();
        activateStage(container, target);
      }});
    }});
    container.querySelectorAll("[data-stage-step]").forEach((step, index) => {{
      step.addEventListener("click", () => activateStage(container, index));
      step.addEventListener("focus", () => activateStage(container, index));
    }});
    activateStage(container, 0);
  }});
  document.querySelectorAll(".president-era-explorer").forEach(explorer => {{
    const figureKey = explorer.dataset.eraFigure;
    const timelineFigureKey = explorer.dataset.registerTimelineFigure;
    const timelineStatus = explorer.dataset.timelineStatus;
    const chart = explorer.querySelector("[data-fig]");
    const choices = [...explorer.querySelectorAll("[data-era-choice]")];
    const eraSelect = explorer.querySelector("[data-era-select]");
    const registerViewButtons = [
      ...explorer.querySelectorAll("[data-register-view]")
    ];
    const allButton = explorer.querySelector("[data-era-all]");
    const clearButton = explorer.querySelector("[data-era-clear]");
    const presetButtons = [...explorer.querySelectorAll("[data-era-preset]")];
    const status = explorer.querySelector(".filter-status");
    const sourceFigure = FIGS[figureKey];
    const sourceTrace = sourceFigure?.data?.[0];
    const rows = sourceTrace?.customdata || [];
    const marker = sourceTrace?.marker || {{}};
    const markerLine = marker.line || {{}};
    const styleList = (value, fallback) => {{
      if (Array.isArray(value) || ArrayBuffer.isView(value)) {{
        const expanded = Array.from(value);
        if (expanded.length === rows.length) return expanded.slice();
      }}
      const scalar = value == null || typeof value !== "object"
        ? (value ?? fallback) : fallback;
      return rows.map(() => scalar);
    }};
    const baseOpacities = Object.freeze(styleList(marker.opacity, 1));
    const baseWidths = Object.freeze(styleList(markerLine.width, 3.2));
    const baseFills = Object.freeze(styleList(marker.color, "#fff8ec"));
    const baseOutlines = Object.freeze(styleList(markerLine.color, "#6f4828"));
    const baseImages = Object.freeze((sourceFigure?.layout?.images || []).map(
      item => Object.freeze({{name:item.name || "", opacity:item.opacity ?? 1}})
    ));
    function applyEraSelection() {{
      const timelineActive = registerViewButtons.some(
        button => button.dataset.registerView === "timeline"
          && button.getAttribute("aria-pressed") === "true"
      );
      if (timelineActive) {{
        if (status) status.textContent = timelineStatus;
        return;
      }}
      renderChart(chart);
      if (!rows.length) {{
        if (status) status.textContent = "No complete president points are available to highlight.";
        return;
      }}
      const allEras = new Set(rows.map(row => row[3]));
      let selected;
      if (eraSelect) {{
        selected = eraSelect.value === "__all__"
          ? allEras
          : eraSelect.value === "__none__"
            ? new Set()
            : new Set([eraSelect.value]);
      }} else {{
        selected = new Set(
          choices.filter(choice => choice.checked).map(choice => choice.value)
        );
      }}
      const opacities = rows.map((row, index) =>
        selected.has(row[3]) ? baseOpacities[index] : .22);
      const widths = rows.map((row, index) =>
        selected.has(row[3]) ? baseWidths[index] : .8);
      const fills = rows.map((row, index) =>
        selected.has(row[3]) ? baseFills[index] : "#eef0f1");
      const outlines = rows.map((row, index) =>
        selected.has(row[3]) ? baseOutlines[index] : "#8d9498");
      Plotly.restyle(chart, {{
        "marker.opacity": [opacities],
        "marker.line.width": [widths],
        "marker.color": [fills],
        "marker.line.color": [outlines],
      }}, [0]);
      const eraByPresident = new Map(rows.map(row => [row[0], row[3]]));
      const imageUpdates = {{}};
      baseImages.forEach((item, index) => {{
        const president = (item.name || "").replace("portrait::", "");
        imageUpdates[`images[${{index}}].opacity`] = selected.has(
          eraByPresident.get(president)
        ) ? item.opacity : .18;
      }});
      Plotly.relayout(chart, imageUpdates);
      if (eraSelect) {{
        const selectedLabel = eraSelect.selectedOptions[0]?.textContent || "Selected era";
        status.textContent = eraSelect.value === "__all__"
          ? "All nine eras emphasized."
          : eraSelect.value === "__none__"
            ? "All eras dimmed; every plotted president remains available for inspection."
            : `${{selectedLabel}} emphasized; all other plotted presidents remain visible at low emphasis.`;
      }} else {{
        const labels = choices.filter(choice => choice.checked)
          .map(choice => choice.value);
        status.textContent = labels.length === choices.length
          ? "All nine eras highlighted; all plotted presidents visible."
          : labels.length
            ? `${{labels.join(", ")}} highlighted; all other plotted presidents remain visible.`
          : "No era highlighted; every plotted president remains visible at low emphasis.";
      }}
    }}
    async function activateRegisterView(view) {{
      if (!timelineFigureKey || !registerViewButtons.length) return;
      const isTimeline = view === "timeline";
      registerViewButtons.forEach(button => {{
        button.setAttribute(
          "aria-pressed", String(button.dataset.registerView === view)
        );
      }});
      if (eraSelect) eraSelect.disabled = isTimeline;
      explorer.classList.toggle("is-timeline-view", isTimeline);
      chart.dataset.fig = isTimeline ? timelineFigureKey : figureKey;
      if (plotted.has(chart)) {{
        const figure = FIGS[chart.dataset.fig];
        await Plotly.react(
          chart, figure.data, figure.layout,
          {{displayModeBar:false,responsive:true}}
        );
      }} else {{
        renderChart(chart);
      }}
      applyEraSelection();
    }}
    registerViewButtons.forEach((button, index) => {{
      button.addEventListener("click", () => {{
        void activateRegisterView(button.dataset.registerView);
      }});
      button.addEventListener("keydown", event => {{
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        let target = index + (event.key === "ArrowRight" ? 1 : -1);
        if (event.key === "Home") target = 0;
        if (event.key === "End") target = registerViewButtons.length - 1;
        target = (target + registerViewButtons.length) % registerViewButtons.length;
        registerViewButtons[target].focus();
        registerViewButtons[target].click();
      }});
    }});
    eraSelect?.addEventListener("change", applyEraSelection);
    choices.forEach(choice => choice.addEventListener("change", applyEraSelection));
    presetButtons.forEach(button => button.addEventListener("click", () => {{
      const wanted = new Set(button.dataset.eraPreset.split("||"));
      choices.forEach(choice => {{ choice.checked = wanted.has(choice.value); }});
      applyEraSelection();
    }}));
    allButton?.addEventListener("click", () => {{
      choices.forEach(choice => {{ choice.checked = true; }});
      applyEraSelection();
    }});
    clearButton?.addEventListener("click", () => {{
      choices.forEach(choice => {{ choice.checked = false; }});
      applyEraSelection();
    }});
  }});
  if (!reduceMotion && "IntersectionObserver" in window) {{
    const stageObserver = new IntersectionObserver(entries => {{
      for (const entry of entries) {{
        if (!entry.isIntersecting) continue;
        const container = entry.target.closest(".story-stage");
        activateStage(container, Number(entry.target.dataset.stageStep));
      }}
    }}, {{rootMargin:"-28% 0px -48%",threshold:.2}});
    document.querySelectorAll("[data-stage-step]").forEach(step => stageObserver.observe(step));
  }}
  const progress = document.getElementById("story-progress");
  const progressEra = document.getElementById("story-progress-era");
  const progressTitle = document.getElementById("story-progress-title");
  function eraWorkspacePanels(workspace) {{
    const stage = workspace?.querySelector(
      ":scope > .founding-panel-stage"
    );
    return stage
      ? [...stage.children].filter(
          child => child.classList.contains("founding-panel")
        )
      : [];
  }}
  function activateEraWorkspacePanel(
    workspace, panelId, updateHash = false, scrollToWorkspace = false
  ) {{
    if (!workspace) return;
    const tabs = [
      ...workspace.querySelectorAll("[data-era-workspace-tab]")
    ];
    const panels = eraWorkspacePanels(workspace);
    const panelIndex = panels.findIndex(panel => panel.id === panelId);
    if (panelIndex < 0) return;
    tabs.forEach((tab, index) => {{
      const selected = tab.dataset.eraWorkspaceTab === panelId;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      tab.dataset.stepStatus = (
        selected ? "current" : index < panelIndex ? "complete" : "upcoming"
      );
    }});
    panels.forEach(panel => {{
      panel.hidden = panel.id !== panelId;
    }});
    if (updateHash) history.replaceState(null, "", `#${{panelId}}`);
    if (scrollToWorkspace) {{
      const alignWorkspace = () => workspace.scrollIntoView({{
          behavior: reduceMotion || !updateHash ? "auto" : "smooth",
          block: "start",
        }});
      if (!updateHash) {{
        const settleWorkspace = () => setTimeout(alignWorkspace, 250);
        if (document.readyState === "complete") {{
          settleWorkspace();
        }} else {{
          addEventListener("load", settleWorkspace, {{once:true}});
        }}
      }} else {{
        alignWorkspace();
      }}
    }}
  }}
  document.querySelectorAll("[data-era-workspace]").forEach(workspace => {{
    const tabs = [
      ...workspace.querySelectorAll("[data-era-workspace-tab]")
    ];
    const panels = eraWorkspacePanels(workspace);
    const profile = panels.find(
      panel => !panel.hasAttribute("data-era-panel")
    );
    function syncEraPanelHeight() {{
      if (!profile) return;
      const height = profile.getBoundingClientRect().height;
      if (height > 0) {{
        workspace.style.setProperty(
          "--era-profile-height", `${{Math.ceil(height)}}px`
        );
      }}
    }}
    syncEraPanelHeight();
    addEventListener("load", syncEraPanelHeight, {{once:true}});
    addEventListener("resize", syncEraPanelHeight);
    if ("ResizeObserver" in window && profile) {{
      new ResizeObserver(syncEraPanelHeight).observe(profile);
    }}
    tabs.forEach((tab, index) => {{
      tab.addEventListener("click", () => {{
        activateEraWorkspacePanel(
          workspace, tab.dataset.eraWorkspaceTab, true, true
        );
      }});
      tab.addEventListener("keydown", event => {{
        if (!["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
        event.preventDefault();
        let target = index + (event.key === "ArrowRight" ? 1 : -1);
        if (event.key === "Home") target = 0;
        if (event.key === "End") target = tabs.length - 1;
        target = (target + tabs.length) % tabs.length;
        tabs[target].focus();
        activateEraWorkspacePanel(
          workspace, tabs[target].dataset.eraWorkspaceTab, true, true
        );
      }});
    }});
    const requestedHash = location.hash.slice(1);
    const prefix = workspace.dataset.eraWorkspace;
    const legacyPanelAliases = {{
      [`${{prefix}}-visualize`]: `${{prefix}}-screen-adversaries`,
      [`${{prefix}}-contextualize`]: `${{prefix}}-screen-defined`,
      [`${{prefix}}-screen-agenda`]: `${{prefix}}-screen-adversaries`,
    }};
    const requestedPanelId =
      legacyPanelAliases[requestedHash] || requestedHash;
    const requestedPanel = panels.find(
      panel => panel.id === requestedPanelId
    );
    activateEraWorkspacePanel(
      workspace,
      requestedPanel?.id || profile?.id || panels[0]?.id,
    );
  }});
  document.querySelectorAll("[data-founding-screen-group]").forEach(group => {{
    const tabs = [...group.querySelectorAll("[data-founding-screen-tab]")];
    const panels = [...group.querySelectorAll("[data-founding-screen]")];
    function activateScreen(panelId, updateHash = false) {{
      if (!panels.some(panel => panel.id === panelId)) return;
      tabs.forEach(tab => {{
        const selected = tab.dataset.foundingScreenTab === panelId;
        tab.setAttribute("aria-selected", String(selected));
        tab.tabIndex = selected ? 0 : -1;
      }});
      panels.forEach(panel => {{
        panel.hidden = panel.id !== panelId;
      }});
      if (updateHash) history.replaceState(null, "", `#${{panelId}}`);
    }}
    tabs.forEach((tab, index) => {{
      tab.addEventListener("click", () => {{
        activateScreen(tab.dataset.foundingScreenTab, true);
      }});
      tab.addEventListener("keydown", event => {{
        if (!["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
        event.preventDefault();
        let target = index + (event.key === "ArrowRight" ? 1 : -1);
        if (event.key === "Home") target = 0;
        if (event.key === "End") target = tabs.length - 1;
        target = (target + tabs.length) % tabs.length;
        tabs[target].focus();
        activateScreen(tabs[target].dataset.foundingScreenTab, true);
      }});
    }});
    const legacyScreenAliases = {{
      "founding-screen-native": "founding-screen-defined",
      "founding-screen-afterlife": "founding-screen-defined",
      "founding-screen-tests": "founding-screen-defined",
    }};
    const hashScreenId = location.hash.slice(1);
    const requestedScreenId =
      legacyScreenAliases[hashScreenId] || hashScreenId;
    const requestedScreen = panels.find(
      panel => panel.id === requestedScreenId
    );
    if (requestedScreen) {{
      const eraPanel = requestedScreen.closest(".founding-panel");
      const eraWorkspace = requestedScreen.closest("[data-era-workspace]");
      if (eraPanel && eraWorkspace) {{
        activateEraWorkspacePanel(eraWorkspace, eraPanel.id);
      }}
      activateScreen(requestedScreen.id);
    }} else if (panels.length) {{
      activateScreen(panels[0].id);
    }}
  }});
  document.querySelectorAll("[data-adversary-network]").forEach(network => {{
    const edges = [...network.querySelectorAll("[data-adversary-edge]")];
    const nodes = [...network.querySelectorAll("[data-adversary-node]")];
    const interactive = [...edges, ...nodes];
    function clearAdversaryHighlight() {{
      network.classList.remove("has-active");
      interactive.forEach(item => {{
        item.classList.remove("is-active", "is-related");
      }});
    }}
    function activateAdversaryItem(active) {{
      const presidentKey = active.dataset.presidentKey || "";
      const adversaryKey = active.dataset.adversaryKey || "";
      const relatedEdges = edges.filter(edge => (
        (!presidentKey || edge.dataset.presidentKey === presidentKey)
        && (!adversaryKey || edge.dataset.adversaryKey === adversaryKey)
      ));
      const presidentKeys = new Set(
        relatedEdges.map(edge => edge.dataset.presidentKey)
      );
      const adversaryKeys = new Set(
        relatedEdges.map(edge => edge.dataset.adversaryKey)
      );
      network.classList.add("has-active");
      edges.forEach(edge => {{
        edge.classList.toggle("is-active", edge === active);
        edge.classList.toggle(
          "is-related", edge !== active && relatedEdges.includes(edge)
        );
      }});
      nodes.forEach(node => {{
        const related = (
          presidentKeys.has(node.dataset.presidentKey)
          || adversaryKeys.has(node.dataset.adversaryKey)
        );
        node.classList.toggle("is-active", node === active);
        node.classList.toggle("is-related", node !== active && related);
      }});
    }}
    interactive.forEach(item => {{
      item.addEventListener(
        "pointerenter", () => activateAdversaryItem(item)
      );
      item.addEventListener("click", () => {{
        item.focus();
        activateAdversaryItem(item);
      }});
      item.addEventListener("focus", () => activateAdversaryItem(item));
      item.addEventListener("blur", () => {{
        setTimeout(() => {{
          if (!network.contains(document.activeElement)) {{
            clearAdversaryHighlight();
          }}
        }}, 0);
      }});
    }});
    network.addEventListener("pointerleave", () => {{
      if (!network.contains(document.activeElement)) {{
        clearAdversaryHighlight();
      }}
    }});
  }});
  document.querySelectorAll("[data-echo-direction-switch]").forEach(
    switcher => {{
      const container = switcher.parentElement;
      const buttons = [
        ...switcher.querySelectorAll("[data-echo-direction]")
      ];
      const views = [...container.querySelectorAll("[data-echo-view]")];
      const status = container.querySelector("[data-echo-direction-status]");
      function activateEchoDirection(direction) {{
        const button = buttons.find(
          candidate => candidate.dataset.echoDirection === direction
        );
        if (!button || button.disabled) return;
        buttons.forEach(candidate => {{
          candidate.setAttribute(
            "aria-pressed",
            String(candidate === button),
          );
        }});
        views.forEach(view => {{
          view.hidden = view.dataset.echoView !== direction;
        }});
        if (status) {{
          status.textContent = button.querySelector("small")?.textContent || "";
        }}
      }}
      buttons.forEach((button, index) => {{
        button.addEventListener("click", () => {{
          activateEchoDirection(button.dataset.echoDirection);
        }});
        button.addEventListener("keydown", event => {{
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {{
            return;
          }}
          event.preventDefault();
          const available = buttons.filter(candidate => !candidate.disabled);
          if (!available.length) return;
          const current = available.indexOf(button);
          let target = current + (event.key === "ArrowRight" ? 1 : -1);
          if (event.key === "Home") target = 0;
          if (event.key === "End") target = available.length - 1;
          target = (target + available.length) % available.length;
          available[target].focus();
          activateEchoDirection(available[target].dataset.echoDirection);
        }});
      }});
    }}
  );
  document.querySelectorAll("[data-echo-gravity]").forEach(field => {{
    const nodes = [...field.querySelectorAll("[data-echo-node]")];
    const links = [...field.querySelectorAll("[data-echo-link]")];
    const topicPresidentLinks = [
      ...field.querySelectorAll("[data-topic-president-link]")
    ];
    const readout = field.querySelector("[data-echo-readout]");
    const defaultText = readout?.dataset.defaultText || "";
    function clearEchoGravity() {{
      field.classList.remove("has-active");
      nodes.forEach(node => node.classList.remove("is-active","is-related"));
      links.forEach(link => link.classList.remove("is-active"));
      topicPresidentLinks.forEach(
        link => link.classList.remove("is-active")
      );
      if (readout) readout.textContent = defaultText;
    }}
    function activateEchoNode(active) {{
      const kind = active.dataset.nodeKind;
      const nodeKey = active.dataset.nodeKey;
      const anchorKeys = new Set(
        (active.dataset.anchorKeys || "").split(" ").filter(Boolean)
      );
      const presidentKeys = new Set(
        (active.dataset.presidentKeys || "").split(" ").filter(Boolean)
      );
      const topicKeys = new Set(
        topicPresidentLinks
          .filter(link => link.dataset.presidentKey === nodeKey)
          .map(link => link.dataset.topicKey)
      );
      field.classList.add("has-active");
      nodes.forEach(node => {{
        const candidateAnchors = new Set(
          (node.dataset.anchorKeys || "").split(" ").filter(Boolean)
        );
        const related = (
          kind === "anchor"
          ? candidateAnchors.has(nodeKey)
          : (
              node.dataset.nodeKind === "anchor"
              && anchorKeys.has(node.dataset.nodeKey)
            )
            || (
              kind === "topic"
              && node.dataset.nodeKind === "president"
              && presidentKeys.has(node.dataset.nodeKey)
            )
            || (
              kind === "president"
              && node.dataset.nodeKind === "topic"
              && topicKeys.has(node.dataset.nodeKey)
            )
        );
        node.classList.toggle("is-active", node === active);
        node.classList.toggle("is-related", related);
      }});
      links.forEach(link => {{
        const related = kind === "anchor"
          ? link.dataset.anchorKey === nodeKey
          : link.dataset.satelliteKey === nodeKey;
        link.classList.toggle("is-active", related);
      }});
      topicPresidentLinks.forEach(link => {{
        const related = (
          (kind === "topic" && link.dataset.topicKey === nodeKey)
          || (
            kind === "president"
            && link.dataset.presidentKey === nodeKey
          )
        );
        link.classList.toggle("is-active", related);
      }});
      if (readout) readout.textContent = active.dataset.echoDetail || defaultText;
    }}
    nodes.forEach(node => {{
      node.addEventListener("pointerenter", () => activateEchoNode(node));
      node.addEventListener("click", () => {{
        node.focus();
        activateEchoNode(node);
      }});
      node.addEventListener("pointerleave", () => {{
        if (!field.contains(document.activeElement)) clearEchoGravity();
      }});
      node.addEventListener("focus", () => activateEchoNode(node));
      node.addEventListener("blur", () => {{
        setTimeout(() => {{
          if (!field.contains(document.activeElement)) clearEchoGravity();
        }}, 0);
      }});
    }});
  }});
  document.querySelectorAll("[data-founding-label-toggle]").forEach(button => {{
    const grid = document.getElementById(button.getAttribute("aria-controls"));
    if (!grid) return;
    button.addEventListener("click", () => {{
      const expanded = button.getAttribute("aria-expanded") !== "true";
      button.setAttribute("aria-expanded", String(expanded));
      button.textContent = expanded ? "Hide topic labels" : "Show topic labels";
      grid.classList.toggle("labels-open", expanded);
    }});
  }});
  const foundingAliases = {{
    "founding-problems": "founding-screen-adversaries",
    "founding-threads": "founding-screen-defined",
    "founding-visualize": "founding-screen-adversaries",
    "founding-contextualize": "founding-screen-defined",
    "founding-screen-agenda": "founding-screen-adversaries",
    "founding-screen-native": "founding-screen-defined",
    "founding-screen-afterlife": "founding-screen-defined",
    "founding-screen-tests": "founding-screen-defined",
  }};
  const legacyFoundingScreenAliases = {{
    "founding-screen-native": "founding-screen-defined",
    "founding-screen-afterlife": "founding-screen-defined",
    "founding-screen-tests": "founding-screen-defined",
  }};
  const requestedHashId = location.hash.slice(1);
  const requestedFoundingId =
    legacyFoundingScreenAliases[requestedHashId] || requestedHashId;
  const requestedFoundingScreen = [
    ...document.querySelectorAll("[data-founding-screen]")
  ].find(panel => panel.id === requestedFoundingId);
  const foundingWorkspace = document.getElementById("founding-workspace");
  const foundingPanels = eraWorkspacePanels(foundingWorkspace);
  if (requestedFoundingScreen) {{
    const foundingPanel = requestedFoundingScreen.closest(".founding-panel");
    if (foundingPanel) {{
      activateEraWorkspacePanel(
        foundingWorkspace, foundingPanel.id, false, true
      );
    }}
  }} else if (foundingAliases[requestedFoundingId]) {{
    activateEraWorkspacePanel(
      foundingWorkspace, foundingAliases[requestedFoundingId], false, true
    );
  }} else if (foundingPanels.some(panel => panel.id === requestedFoundingId)) {{
    activateEraWorkspacePanel(
      foundingWorkspace, requestedFoundingId, false, true
    );
  }} else if (foundingPanels.length) {{
    activateEraWorkspacePanel(foundingWorkspace, "founding-profile");
  }}
  function updateStoryProgress() {{
    if (!progress || !progressEra || !progressTitle) return;
    const root = document.documentElement;
    const distance = root.scrollHeight - innerHeight;
    progress.style.width = `${{Math.max(0,Math.min(100,(scrollY / Math.max(distance,1))*100))}}%`;
    const marker = innerHeight * .36;
    let active = sections[0];
    for (const section of sections) {{
      if (section.getBoundingClientRect().top <= marker) active = section;
    }}
    if (active) {{
      progressEra.textContent =
        `${{active.dataset.storyRange}} · ${{active.dataset.storyEra}}`;
      progressTitle.textContent = active.dataset.storyTitle;
    }}
  }}
  addEventListener("scroll", updateStoryProgress, {{passive:true}});
  updateStoryProgress();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the dashboard site")
    parser.add_argument("--inline", action="store_true",
                        help="also write a self-contained copy (plotly.js embedded)")
    args = parser.parse_args()

    story_bundle = story_foundation.load_story_foundation()
    story_foundation.check_story_contract(story_bundle)
    topic_network_bundle = speaker_topic_network.load_network_bundle(
        foundation=story_bundle
    )
    invocation_network_bundle = actual_speaker_invocation_network.build_network_bundle(
        story_bundle,
        topic_network_bundle,
    )
    summary_topic_projection = summary_topic_network.build_projection(
        topic_network_bundle
    )
    summary_topic_section = summary_topic_network.render_summary_section(
        summary_topic_projection
    )

    df = corpus.load()
    word_families.build_families(df)      # before keyword_trends: term_groups reads the map
    trends.term_groups.cache_clear()
    stats = rhetoric.build_stats(df)
    markers = indices.build_markers(df)
    rates = indices.yearly_rates(markers)
    _, issue_meta = issues.build_issues()
    para_labels = pd.read_parquet(issues.PARA_LABELS_PATH)

    all_scores = indices.president_scores(markers, stats, df).set_index("president")
    # Presidents with a handful of speeches distort per-president charts
    # (Garfield's one speech made him a certainty outlier); they keep their
    # profile pages but sit out the dashboard graphics.
    sparse = set(all_scores[all_scores["n_speeches"] < 5].index)
    scores = all_scores[all_scores["n_speeches"] >= 5]
    faces = portraits.data_uris(list(all_scores.index))
    issue_df, _ = issues.build_issues()
    ai_data = ai_labels.build_ai_data()
    speech_annotations = ai_labels.load_speech_annotations("speech_annotations")
    paragraphs = pd.read_parquet(corpus.DATA_DIR / "paragraphs.parquet")
    paragraph_annotations = pd.read_parquet(
        corpus.DATA_DIR / "llm_annotations" / "paragraph_annotations.parquet"
    )
    expansion_metrics = expansion_story.load_metrics()
    expansion_meta = expansion_story.load_meta()
    agreement_table = pd.read_parquet(ai_labels.AGREEMENT_PATH)
    invocation_evidence = pd.read_parquet(
        corpus.DATA_DIR / "networks" / "invocation_evidence.parquet"
    )
    lifecycle_table = pd.read_parquet(attention.LIFECYCLES_PATH)
    taxonomy = attention.load_taxonomy()
    band_table = pd.read_parquet(corpus.DATA_DIR / "bands.parquet")
    founding_data = founding_story_data(
        df,
        paragraphs,
        paragraph_annotations,
        speech_annotations,
        stats,
        agreement_table,
        lifecycle_table,
        taxonomy,
        story_bundle,
        invocation_evidence,
    )
    all_era_profile_data = founding_data["era_profiles"]
    all_era_visualization_data = founding_data["era_visualizations"]
    all_era_contextualization_data = founding_data["era_contextualizations"]
    register_trends = pd.read_parquet(
        corpus.DATA_DIR / "register" / "trends.parquet"
    )
    president_conflict_treatments = pd.read_parquet(
        combat.BY_PRESIDENT_TREATMENTS_V2_PATH
    )
    combat_ratios = pd.read_parquet(combat.RATIOS_PATH)
    era_similarity = pd.read_parquet(eras.ERA_SIMILARITY_PATH)
    target_mix_table = pd.read_parquet(
        corpus.DATA_DIR
        / "combat"
        / "target_mix_by_era_speaker_audited_v1.parquet"
    )
    coverage_receipts = pd.read_parquet(
        corpus.DATA_DIR / "coverage_pressure" / "inference_receipts.parquet"
    )
    summary_data = build_summary_data(
        df,
        paragraphs,
        paragraph_annotations,
        speech_annotations,
        markers,
        taxonomy,
        lifecycle_table,
        register_trends,
        president_conflict_treatments,
        target_mix_table,
        coverage_receipts,
        founding_data,
    )
    era_profile_sections = {
        spec.section_key: _era_workspace_html(
            all_era_profile_data[spec.key],
            all_era_visualization_data[spec.key],
            all_era_contextualization_data[spec.key],
        )
        for spec in era_profiles.ERA_PROFILE_SPECS
    }
    profile_data = profiles.build_profile_data()
    profile_data["ai"] = ai_data
    profile_data["feature_neighbors"] = profiles.feature_neighbors(
        profile_data, ai_data
    )
    display_issues = topic_quality.display_issues(issue_meta["issues"])
    profile_views = profiles_site.build_profile_view_models(
        profile_data,
        display_issues,
        require_complete=True,
    )
    profile_context_projection = profile_context.verify_reproducible(
        topic_network_bundle,
        invocation_network_bundle,
        story_bundle,
        profile_views,
    )
    profile_connection_sizes = profile_connections_assets.profile_connections_asset_sizes()
    if (
        profile_connection_sizes["javascript_raw_bytes"] > 60_000
        or profile_connection_sizes["javascript_gzip_bytes"] > 20_000
    ):
        raise ValueError(
            "profile Connections enhancement module exceeds its byte budget"
        )
    compare_projection_bundle = compare_projection.build_projection(
        topic_network_bundle,
        profile_data,
        profile_views,
        display_issues,
    )
    compare_projection_repeat = compare_projection.build_projection(
        topic_network_bundle,
        profile_data,
        profile_views,
        display_issues,
    )
    if compare_projection_repeat.files != compare_projection_bundle.files:
        raise compare_projection.CompareProjectionError(
            "two in-memory Compare projection builds were not byte-identical"
        )
    explore_projection_bundle = explore_projection.build_projection(
        speeches=df,
        band_table=band_table,
    )
    explore_projection_repeat = explore_projection.build_projection(
        speeches=df,
        band_table=band_table,
    )
    if explore_projection_repeat.files != explore_projection_bundle.files:
        raise explore_projection.ExploreProjectionError(
            "two in-memory Explore projection builds were not byte-identical"
        )
    del explore_projection_repeat
    conflict_contract = summary_data["conflict_presidents"]
    conflict_target_contract = summary_data["conflict_targets"]
    conflict_figs = {
        "summary_conflict_targets": fig_summary_conflict_targets(
            conflict_target_contract
        ),
        SUMMARY_CONFLICT_PORTRAIT_FIGURE_KEY: fig_summary_conflict_portraits(
            conflict_contract, faces
        ),
    }

    figs = {
        "expansion_story": fig_expansion_story(
            expansion_metrics, expansion_meta
        ),
        "fear_index_era": fig_fear_index(rates),
        "fear_index_full": fig_fear_index(rates, full_history=True),
        "civil_rights_yearly": fig_civil_rights_yearly(para_labels, paragraphs),
        "civil_rights_full": fig_civil_rights_full(band_table),
        "procedural_eras": fig_procedural_eras(all_scores, faces),
        "summary_register_timeline": fig_summary_register_timeline(rates),
        "summary_audience": fig_summary_audience(summary_data),
        "summary_medium": fig_summary_medium(summary_data),
        **conflict_figs,
        "summary_temporal": fig_summary_temporal(
            summary_data["temporal_presidents"], faces
        ),
        "summary_temporal_timeline": fig_summary_temporal_timeline(markers),
        "summary_hope_doom_ratio": fig_summary_hope_doom_ratio(markers),
        "progressive_heatmap": fig_progressive_heatmap(para_labels),
        "naming_progressive": fig_naming_progressive(rates, df),
        "summary_modals": fig_summary_modals(stats, markers, df),
        "summary_pronouns": fig_summary_pronouns(stats),
        "new_deal": fig_new_deal_war(para_labels),
        "broadcast": fig_broadcast_presidency(stats, speech_annotations),
        "platform_era": fig_platform_signals(rates, markers),
        "platform_full": fig_platform_signals(
            rates, markers, full_history=True,
        ),
        "weather_map": fig_rhetorical_weather(markers),
    }
    metrics.validate_charts(set(figs))
    prose_extra = {
        "expansion_conflict": expansion_takeaway(
            expansion_metrics, expansion_meta
        ),
        "union_crisis": fear_index_takeaway(rates),
        "progressive_transform": progressive_takeaway(para_labels),
        "recovery_mobilization": new_deal_takeaway(para_labels),
        "broadcast_presidency": broadcast_takeaway(stats, speech_annotations),
        "platform_dominance": platform_takeaway(rates, markers),
        "synthesis": weather_takeaway(markers),
    }
    bodies = {
        "written_republic": founding_story_html(founding_data),
        "expansion_conflict": _expansion_story_html(
            expansion_metrics, expansion_meta
        ),
        "union_crisis": (
            _view_toggle_html(
                ["fear_index_era", "fear_index_full"],
                [
                    "Era view: 1850–1868 windows are indexed to the mean of the supported 1789–1808 founding windows.",
                    "Full history: the x- and y-scales expand to every supported 1789–2026 window; the same 1789–1808 index remains 100.",
                ],
                500, "rate_10k",
                source="speech_markers.parquet",
                measure_label="Fear÷hope index · founding mean = 100",
                support="Centered five-year word-weighted windows above the 20,000-word display floor.",
                caveat="NRC word matches can reflect quotation or description; the index is descriptive.",
            )
            + _event_callouts([(1861, "Civil War begins")])
            + '<h3>Rights language through war and constitutional rupture</h3>'
            + _view_toggle_html(
                ["civil_rights_yearly", "civil_rights_full"],
                [
                    "Era view: annual points expose speech, paragraph, and word support; thinner years use smaller, lighter markers.",
                    "Full history: the grain changes to five-year speech-clustered sampling bands and reports each period's support status.",
                ],
                500, "paragraph_share",
                source="bands.parquet",
                measure_label="Civil rights & race paragraph share",
                support="Annual paragraph share in Era view; five-year speech-clustered CorEx bands in Full history.",
                caveat=(
                    "The anchored label includes slavery, race/racial, discrimination, segregation, "
                    "equality, emancipation, and the historical corpus term “negro.” It can miss "
                    "paraphrase or discussion without endorsement and does not measure rights realized."
                ),
            )
            + _event_callouts(CIVIL_RIGHTS_EVENTS["Civil rights & race"])
        ),
        "procedural_presidency": "",
        "progressive_transform": (
            _chart_block(
                "progressive_heatmap", 520, "paragraph_share",
                source="paragraph_issues.parquet",
                measure_label="Issue movement from the preceding story era",
                support="All paragraphs in three fixed windows; every cell uses the same 1869–1912 issue baseline.",
                caveat="Events orient the windows and do not identify causes.",
            )
            + _event_callouts([
                (1917, "U.S. entry into World War I"),
                (1929, "Market crash"),
                (1930, "Smoot–Hawley tariff"),
            ])
        ),
        "recovery_mobilization": (
            _chart_block(
                "new_deal", 500, "paragraph_share",
                source="paragraph_issues.parquet",
                measure_label="Agenda movement · paragraph share across three phases",
                support="Every corpus paragraph in 1933–39, 1940–45, and 1946–52.",
                caveat="Changing rhetoric does not measure program effectiveness.",
            )
            + '<p class="source-links"><a href="https://millercenter.org/the-presidency/'
              'presidential-speeches/march-4-1933-first-inaugural-address" target="_blank" '
              'rel="noopener">1933 inaugural source →</a> · '
              '<a href="https://millercenter.org/the-presidency/presidential-speeches/'
              'december-9-1941-fireside-chat-19-war-japan" target="_blank" rel="noopener">'
              '1941 fireside-chat source →</a></p>'
        ),
        "broadcast_presidency": (
            _chart_block(
                "broadcast", 650, "primary_medium",
                source="llm_annotations/speech_annotations.parquet",
                measure_label="One assigned primary form per speech",
                support="Every corpus speech in each displayed decade.",
                caveat=(
                    "Spoken address and radio/TV broadcast are mutually exclusive AI labels even "
                    "when a spoken address was broadcast; this is not a channel inventory."
                ),
            )
            + _chart_evidence(
                "reading_level", "speech_stats.parquet",
                support="Median speech-level Flesch–Kincaid scores in the displayed rolling context.",
                caveat="Lower grade level does not mean intellectually simpler communication.",
            )
        ),
        "always_on": _coverage_pressure_v2_html(df),
        "platform_dominance": (
            _view_toggle_html(
                ["platform_era", "platform_full"],
                [
                    "Era view: 2001–April 2026 shows the approach to and continuation beyond the 2017 boundary.",
                    "Full history: the time scale expands to 1789 while definitions and per-10,000-word units stay fixed.",
                ],
                720, "rate_10k",
                source="speech_markers.parquet",
                measure_label="Legal, hype, and opponent-centered wording",
                support="Centered five-year word-weighted rates; 2026 also exposes four raw speeches and 17,177 words.",
                caveat="Opponent-centered wording includes generic opponents, parties, politicians, press, and media; it does not identify named people.",
            )
            + _event_callouts([(2017, "Chapter boundary and intensification point")])
        ),
        "synthesis": _standing_html(
            founding_data,
            markers,
            scores=all_scores,
            speeches=df,
            rates=rates,
            stats=stats,
            summary_data=summary_data,
            combat_ratios=combat_ratios,
            era_similarity=era_similarity,
            topic_network_html=summary_topic_section,
        ),
    }
    stats_line = {
        "speeches": len(df),
        "words": int(df["word_count"].sum()),
        "presidents": df["president"].nunique(),
        "start": int(df["year"].min()),
        "end": int(df["year"].max()),
    }
    summary_section_keys = {"synthesis"}
    summary_figure_keys = {
        "procedural_eras", "summary_register_timeline", "naming_progressive",
        "summary_modals", "summary_pronouns",
        "summary_audience", "summary_medium",
        *SUMMARY_CONFLICT_FIGURE_KEYS,
        "summary_temporal", "summary_temporal_timeline",
        "summary_hope_doom_ratio",
    }
    story_section_keys = {key for key, *_ in SECTIONS[:9]}
    story_bodies = {key: "" for key in story_section_keys}
    summary_bodies = {
        key: value for key, value in bodies.items()
        if key in summary_section_keys
    }
    story_figs = {}
    summary_figs = {
        key: value for key, value in figs.items()
        if key in summary_figure_keys
    }

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    write_github_pages_marker(SITE_DIR)
    summary_topic_network.write_public_projection(
        topic_network_bundle,
        SITE_DIR,
        projection=summary_topic_projection,
    )
    summary_topic_network.write_renderer_assets(SITE_DIR)
    profile_context.write_public_projection(
        topic_network_bundle,
        invocation_network_bundle,
        story_bundle,
        profile_views,
        SITE_DIR,
        projection=profile_context_projection,
    )
    profile_connection_asset = SITE_DIR / "assets" / "profile-connections-v1.js"
    profile_connection_asset.parent.mkdir(parents=True, exist_ok=True)
    profile_connection_asset.write_text(
        profile_connections_assets.PROFILE_CONNECTIONS_JS,
        encoding="utf-8",
    )
    if profile_connection_asset.read_text(encoding="utf-8") != (
        profile_connections_assets.PROFILE_CONNECTIONS_JS
    ):
        raise ValueError("profile Connections enhancement asset write drift")
    compare_projection.write_public_projection(
        compare_projection_bundle,
        SITE_DIR,
        network_bundle=topic_network_bundle,
    )
    explore_projection.write_public_projection(
        explore_projection_bundle,
        SITE_DIR,
    )
    era_profiles.write_era_profiles(all_era_profile_data, SITE_DIR)
    era_visualizations.write_era_visualizations(
        all_era_visualization_data, SITE_DIR
    )
    era_contextualizations.write_era_contextualizations(
        all_era_contextualization_data, SITE_DIR
    )
    story_foundation.write_story_downloads(story_bundle, SITE_DIR)
    speaker_topic_network.write_public_downloads(topic_network_bundle, SITE_DIR)
    out = SITE_DIR / "index.html"
    out.write_text(build_html(
        story_figs,
        stats_line,
        story_bodies,
        inline=False,
        era_profile_sections=era_profile_sections,
        page_kind="story",
    ))
    print(f"wrote {out.relative_to(REPO_ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")
    summary_out = SITE_DIR / "summary.html"
    summary_out.write_text(build_html(
        summary_figs,
        stats_line,
        summary_bodies,
        inline=False,
        prose_extra=prose_extra,
        page_kind="summary",
    ))
    print(
        f"wrote {summary_out.relative_to(REPO_ROOT)} "
        f"({summary_out.stat().st_size / 1e6:.1f} MB)"
    )

    context_shards = {
        str(shard["president"]["president_profile_id"]): shard
        for shard in profile_context_projection.shards.values()
    }
    if len(context_shards) != 45:
        raise ValueError("profile context must provide 45 unique president shards")
    profile_views = profiles_site.write_profiles(
        profile_data,
        SITE_DIR,
        views=profile_views,
        context_index=profile_context_projection.index,
        context_shards=context_shards,
    )
    expansion_site.write(SITE_DIR)

    explorer.write_page(explore_projection_bundle, SITE_DIR)
    issues_site.write_issue_pages(SITE_DIR, issue_df, issue_meta, scores, faces,
                                  ai_data=ai_data)
    compare_site.write_compare(
        profile_data,
        display_issues,
        profile_views=profile_views,
        projection=compare_projection_bundle,
    )
    methodology_site.write_methodology(ai_data)
    era_boundaries_site.write_era_boundaries(SITE_DIR)
    if args.inline:
        out2 = SITE_DIR / "index_selfcontained.html"
        out2.write_text(build_html(
            story_figs,
            stats_line,
            story_bodies,
            inline=True,
            era_profile_sections=era_profile_sections,
            page_kind="story",
        ))
        print(f"wrote {out2.relative_to(REPO_ROOT)} ({out2.stat().st_size / 1e6:.1f} MB)")
    bundle_plotly_runtime(SITE_DIR)
    expansion_site.add_global_navigation(SITE_DIR)
    profile_budget_report = profiles_site.validate_generated_profile_budgets(SITE_DIR)
    report = site_validation.validate_site(SITE_DIR)
    story_foundation.check_story_contract(story_bundle, SITE_DIR)
    speaker_topic_network.validate_publication(topic_network_bundle, SITE_DIR)
    summary_topic_network.validate_publication(topic_network_bundle, SITE_DIR)
    profile_context.validate_publication(
        topic_network_bundle,
        invocation_network_bundle,
        story_bundle,
        profile_views,
        SITE_DIR,
    )
    if not profile_connection_asset.is_file():
        raise ValueError("profile Connections enhancement module is missing")
    compare_projection.validate_publication(
        compare_projection_bundle,
        SITE_DIR,
        network_bundle=topic_network_bundle,
    )
    explore_projection.validate_publication(explore_projection_bundle, SITE_DIR)
    print(
        f"  validated {report['html_pages']} HTML pages and "
        f"{report['json_shards']} JSON shards; profile max "
        f"{profile_budget_report['maximum_profile_raw_bytes']:,} raw bytes"
    )


if __name__ == "__main__":
    main()
