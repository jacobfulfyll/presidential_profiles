"""Generate the static dashboard site: every trend as an interactive chart.

Writes docs/index.html (plotly.js from CDN, suitable for GitHub Pages).
With --inline, also writes a fully self-contained copy for offline sharing.
"""

import argparse
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from . import corpus, rhetoric, similarity, topics, trends
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

SITE_DIR = REPO_ROOT / "docs"

FONT = "system-ui, -apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"


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


def _decade_rate(stats: pd.DataFrame, col: str) -> pd.Series:
    g = stats.groupby("decade")
    return g[col].sum() / g["n_tokens"].sum() * 10_000


def fig_modals(stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, m in enumerate(["shall", "will", "must", "should"]):
        rate = _decade_rate(stats, f"modal_{m}")
        fig.add_trace(go.Scatter(
            x=rate.index, y=rate.values, name=m, mode="lines",
            line=dict(color=SERIES[i], width=2.4),
            hovertemplate="%{y:.1f} per 10k<extra>" + m + "</extra>",
        ))
    fig.update_layout(**_layout(hovermode="x unified",
                                yaxis_title="uses per 10,000 words"))
    return fig


def fig_pronouns(stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, (name, col) in enumerate([("we / us / our", "we_count"),
                                     ("I / me / my", "i_count")]):
        rate = _decade_rate(stats, col)
        fig.add_trace(go.Scatter(
            x=rate.index, y=rate.values, name=name, mode="lines",
            line=dict(color=SERIES[i], width=2.4),
            hovertemplate="%{y:.0f} per 10k<extra>" + name + "</extra>",
        ))
    fig.update_layout(**_layout(hovermode="x unified",
                                yaxis_title="uses per 10,000 words"))
    return fig


def fig_readability(stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    s = stats.copy()
    s["fk"] = s["fk_grade"].clip(0, 30)
    fig.add_trace(go.Scatter(
        x=s["year"], y=s["fk"], mode="markers", name="speeches",
        marker=dict(color=BLUE_RAMP[1], size=5, opacity=0.5),
        customdata=np.stack([s["president"], s["title"].str.slice(0, 80)], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>grade %{y:.1f}<extra></extra>",
    ))
    med = s.groupby("decade")["fk"].median()
    fig.add_trace(go.Scatter(
        x=med.index + 5, y=med.values, mode="lines", name="decade median",
        line=dict(color=BLUE_RAMP[5], width=3),
        hovertemplate="median grade %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(**_layout(yaxis_title="Flesch-Kincaid grade level"))
    return fig


def _small_multiples(panels: list[tuple[str, pd.Series]], rows: int, cols: int,
                     height: int, hovertemplate: str) -> go.Figure:
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True,
                        subplot_titles=[t for t, _ in panels],
                        vertical_spacing=0.09, horizontal_spacing=0.06)
    for k, (title, series) in enumerate(panels):
        r, c = divmod(k, cols)
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines",
            line=dict(color=BLUE_RAMP[4], width=2),
            fill="tozeroy", fillcolor="rgba(109, 167, 236, 0.35)",
            hovertemplate=hovertemplate + "<extra>" + title + "</extra>",
            showlegend=False,
        ), row=r + 1, col=c + 1)
    fig.update_layout(**_layout(height=height, margin=dict(l=40, r=16, t=48, b=36)))
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED, size=10))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED, size=10),
                     rangemode="tozero")
    fig.update_annotations(font=dict(size=12.5, color=INK))
    return fig


def fig_topics(doc_topics: pd.DataFrame, topic_terms: dict) -> go.Figure:
    keys = [k for k in doc_topics.columns if k.startswith("topic_")]
    panels = [
        (" · ".join(topic_terms[k][:3]), doc_topics.groupby("decade")[k].mean() * 100)
        for k in keys
    ]
    return _small_multiples(panels, rows=4, cols=3, height=880,
                            hovertemplate="%{y:.1f}% of speech")


def fig_keywords(kw: pd.DataFrame) -> go.Figure:
    panels = [
        (term, kw[kw["term"] == term].set_index("decade")["rate"])
        for term in kw["term"].unique()
    ]
    return _small_multiples(panels, rows=3, cols=3, height=680,
                            hovertemplate="%{y:.1f} per 10k words")


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


def fig_map(emb: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
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


SECTIONS = [
    ("map", "Who sounds like whom",
     "Every speech embedded, averaged per president, projected to 2D with PCA. The model "
     "never sees a date, yet the first principal component recovers time almost perfectly - "
     "presidents drift left to right in chronological order, and the modern era forms its "
     "own cluster. Colors follow party convention."),
    ("heatmap", "Rhetorical similarity, president by president",
     "Cosine similarity between president embeddings, in chronological order. The dark block "
     "in the lower right is the modern presidency; the most similar pair in the corpus is "
     "Bill Clinton and Barack Obama (0.978). Donald Trump has the lowest average similarity "
     "to everyone else of any president with a substantial speech record."),
    ("pronouns", "The 2020s flipped the pronoun trend",
     "Presidential speech spent a century becoming more collective - then the 2020s reversed "
     "it. “We” fell for the first time in a hundred years while “I” "
     "surged to its highest rate since George Washington."),
    ("modals", "The death of “shall”",
     "The classic marker of formal obligation collapsed from 21.8 uses per 10k words in the "
     "1790s to 0.35 today. “Must” peaked in the FDR and war years; promising, "
     "future-facing “will” took over modern speech."),
    ("readability", "Speeches dropped twelve grade levels",
     "Median Flesch-Kincaid reading level fell from grade 19.9 in the 1790s to grade 7.8 in "
     "the 2020s. Hover any dot to see the speech behind it."),
    ("topics", "What presidents talk about, 1789-2026",
     "Twelve NMF topics trace the arc of American history: treaties and commerce in the "
     "early republic, the Constitution and union peaking in the 1860s, gold and silver in "
     "the 1890s, the Soviet block in the Cold War, Iraq and Afghanistan in the 2000s - and "
     "an informal-register topic that explodes in the 2020s."),
    ("keywords", "One word at a time",
     "Usage rates for key terms, per 10,000 words by decade. “Border” and "
     "“immigration” reach all-time highs in the 2020s, above the early-1900s "
     "immigration-era peaks."),
    ("distinctive", "What's new since 2019",
     "Log-odds comparison of speeches added since this project's 2019 snapshot against the "
     "1989-2019 baseline: ukraine, china, testing - and a marked shift toward informal "
     "narration."),
]


def build_html(figs: dict[str, go.Figure], stats_line: dict, inline: bool) -> str:
    from plotly.offline import get_plotlyjs

    plotly_src = (
        f"<script>{get_plotlyjs()}</script>"
        if inline
        else '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>'
    )
    fig_json = {k: json.loads(pio.to_json(f)) for k, f in figs.items()}

    # Explicit pixel heights keep plotly's percent-sized inner containers from
    # collapsing when its responsive handler re-renders after a window resize.
    sections_html = "\n".join(
        f"""<section id="{key}">
  <h2>{title}</h2>
  <p>{prose}</p>
  <div class="chart-scroll"><div class="chart" data-fig="{key}"
       style="height:{figs[key].layout.height}px"></div></div>
</section>"""
        for key, title, prose in SECTIONS
    )

    tiles = [
        (f"{stats_line['speeches']:,}", "speeches"),
        (f"{stats_line['words'] / 1e6:.1f}M", "words"),
        (str(stats_line["presidents"]), "presidents"),
        (f"{stats_line['start']}–{stats_line['end']}", "years covered"),
    ]
    tiles_html = "\n".join(
        f'<div class="tile"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>'
        for num, lbl in tiles
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Presidential Profiles - rhetoric across U.S. history</title>
<meta name="description" content="Interactive analysis of 1,057 presidential speeches, 1789-2026, from the Miller Center corpus.">
{plotly_src}
<style>
  :root {{
    --surface: {SURFACE}; --page: #f9f9f7; --ink: {INK}; --ink2: {INK2};
    --muted: {MUTED}; --grid: {GRID}; --border: rgba(11,11,11,0.10);
  }}
  * {{ box-sizing: border-box; margin: 0; }}
  body {{ background: var(--page); color: var(--ink);
         font-family: {FONT}; line-height: 1.55; }}
  header {{ max-width: 980px; margin: 0 auto; padding: 56px 20px 8px; }}
  header h1 {{ font-size: 2rem; letter-spacing: -0.02em; }}
  header p.sub {{ color: var(--ink2); margin-top: 10px; max-width: 46rem; }}
  .tiles {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 26px 0 8px; }}
  .tile {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 10px; padding: 14px 22px; min-width: 130px; }}
  .tile .num {{ font-size: 1.55rem; font-weight: 650; }}
  .tile .lbl {{ color: var(--muted); font-size: 0.82rem; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 8px 20px 40px; }}
  section {{ margin-top: 44px; }}
  section h2 {{ font-size: 1.28rem; letter-spacing: -0.01em; }}
  section p {{ color: var(--ink2); margin: 8px 0 14px; max-width: 46rem; }}
  .chart-scroll {{ background: var(--surface); border: 1px solid var(--border);
                   border-radius: 12px; padding: 10px 6px 6px; overflow-x: auto; }}
  .chart {{ min-width: 640px; }}
  footer {{ max-width: 980px; margin: 24px auto 60px; padding: 18px 20px 0;
            border-top: 1px solid var(--grid); color: var(--muted); font-size: 0.85rem; }}
  footer a {{ color: var(--ink2); }}
</style>
</head>
<body>
<header>
  <h1>Presidential Profiles</h1>
  <p class="sub">How presidential rhetoric, issues, and policies have shifted across U.S.
  history - every speech in the Miller Center corpus, George Washington's 1789 inaugural
  through April 2026. All charts are interactive: hover for detail, drag to zoom,
  double-click to reset.</p>
  <div class="tiles">
{tiles_html}
  </div>
</header>
<main>
{sections_html}
</main>
<footer>
  <p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs,
  University of Virginia</a> (speeches are in the public domain; collection curated by
  Miller Center staff). Analysis &amp; code:
  <a href="https://github.com/jacobfulfyll/presidential_profiles">jacobfulfyll/presidential_profiles</a>.
  Originally a 2019 Galvanize capstone, rebuilt in 2026.</p>
</footer>
<script>
  const FIGS = {json.dumps(fig_json)};
  for (const el of document.querySelectorAll(".chart")) {{
    const fig = FIGS[el.dataset.fig];
    Plotly.newPlot(el, fig.data, fig.layout,
                   {{displayModeBar: false, responsive: true}});
  }}
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the dashboard site")
    parser.add_argument("--inline", action="store_true",
                        help="also write a self-contained copy (plotly.js embedded)")
    args = parser.parse_args()

    df = corpus.load()
    stats = rhetoric.build_stats(df)
    doc_topics, topic_terms = topics.build_topics(df)
    emb = similarity.build_embeddings(df)
    sim = similarity.similarity_matrix(emb)
    kw = trends.keyword_trends(df)
    distinctive = trends.distinctive_terms(df)

    figs = {
        "map": fig_map(emb),
        "heatmap": fig_heatmap(sim),
        "pronouns": fig_pronouns(stats),
        "modals": fig_modals(stats),
        "readability": fig_readability(stats),
        "topics": fig_topics(doc_topics, topic_terms),
        "keywords": fig_keywords(kw),
        "distinctive": fig_distinctive(distinctive),
    }
    stats_line = {
        "speeches": len(df),
        "words": int(df["word_count"].sum()),
        "presidents": df["president"].nunique(),
        "start": int(df["year"].min()),
        "end": int(df["year"].max()),
    }

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out = SITE_DIR / "index.html"
    out.write_text(build_html(figs, stats_line, inline=False))
    print(f"wrote {out.relative_to(REPO_ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")

    if args.inline:
        out2 = SITE_DIR / "index_selfcontained.html"
        out2.write_text(build_html(figs, stats_line, inline=True))
        print(f"wrote {out2.relative_to(REPO_ROOT)} ({out2.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
