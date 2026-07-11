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

from . import corpus, indices, issues, profiles, profiles_site, rhetoric, similarity, trends
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
from .site_style import FONT, PAGE_CSS

SITE_DIR = REPO_ROOT / "docs"


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


def fig_issues_decade(para_labels: pd.DataFrame, issue_names: list[str]) -> go.Figure:
    """Share of paragraphs touching each curated issue, by decade."""
    pl = para_labels.copy()
    pl["decade"] = (pl["year"] // 10) * 10
    display = issue_names + ["Discovered 5"]
    panels = []
    for name in display:
        label = profiles_site.DISCOVERED_LABELS.get(name, name)
        panels.append((label, pl.groupby("decade")[name].mean() * 100))
    return _small_multiples(panels, rows=4, cols=4, height=880,
                            hovertemplate="%{y:.1f}% of paragraphs")


def _two_line_fig(series: list[tuple[str, pd.Series]], ytitle: str,
                  dash_second: bool = False) -> go.Figure:
    fig = go.Figure()
    for i, (name, s) in enumerate(series):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode="lines",
            line=dict(color=SERIES[i], width=2.4,
                      dash="dash" if (dash_second and i == 1) else "solid"),
            hovertemplate="%{y:.2f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(**_layout(hovermode="x unified", yaxis_title=ytitle))
    return fig


def fig_certainty(markers: pd.DataFrame) -> go.Figure:
    return _two_line_fig(
        [("all speeches", indices.certainty_by_decade(markers)),
         ("inaugural addresses only", indices.certainty_by_decade(markers, inaugural_only=True))],
        ytitle="assertive share of stance markers", dash_second=True,
    )


def fig_naming(rates: pd.DataFrame) -> go.Figure:
    r = rates.set_index("decade")
    return _two_line_fig(
        [("“United States”", r["united_states"]), ("“America / American(s)”", r["america"])],
        ytitle="uses per 10,000 words",
    )


def fig_orientation(rates: pd.DataFrame) -> go.Figure:
    r = rates.set_index("decade")
    return _two_line_fig(
        [("future (future / forward / tomorrow)", r["future"]),
         ("nostalgia (again / restore / back to)", r["nostalgia"])],
        ytitle="uses per 10,000 words",
    )


def fig_religion(rates: pd.DataFrame) -> go.Figure:
    r = rates.set_index("decade")
    return _two_line_fig(
        [("civil religion (god / faith / pray / bless / sacred)", r["religiosity"]),
         ("“God bless”", r["god_bless"])],
        ytitle="uses per 10,000 words",
    )


def fig_hope_fear(rates: pd.DataFrame) -> go.Figure:
    r = rates.set_index("decade")
    return _two_line_fig(
        [("hope words (NRC trust + anticipation + joy)", r["nrc_hope"]),
         ("fear words (NRC fear + anger)", r["nrc_fear"])],
        ytitle="uses per 10,000 words",
    )


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
    ("map", "The river of history",
     "Every speech embedded, averaged per president, projected to 2D with PCA. The model "
     "never sees a date, yet time flows left to right almost perfectly - two-thirds of raw "
     "voice similarity is simply era. Colors follow party convention."),
    ("charmap", "The character map: era removed",
     "The same embeddings with each president's era subtracted - what remains is what made "
     "them different from their contemporaries. Lincoln lands beside FDR (their adjusted "
     "similarity is the highest cross-era pair in the corpus), the great communicators "
     "cluster, and the plain-spoken fighters find each other across centuries."),
    ("heatmap", "The language of eras",
     "Raw cosine similarity, in chronological order - read it as a map of how presidential "
     "language itself changed. The dark block from FDR onward is the modern voice. "
     "For character comparisons free of this era effect, see each president's "
     "“sounds like” list on their profile page."),
    ("certainty", "Confidence replaced deliberation",
     "The assertive share of stance markers: boosters and will/must vs hedges and "
     "concessives (“however”, “although” - the grammar of trade-offs). "
     "The dashed line is inaugural addresses only - the same genre for 240 years - showing "
     "the shift is rhetorical strategy, not just the move from written to spoken messages. "
     "On inaugurals, certainty rose from 0.58 (1800s) to 0.93 (2020s)."),
    ("pronouns", "The 2020s flipped the pronoun trend",
     "Presidential speech spent a century becoming more collective - then the 2020s reversed "
     "it. “We” fell for the first time in a hundred years while “I” "
     "surged to its highest rate since George Washington."),
    ("modals", "The death of “shall”",
     "The classic marker of formal obligation collapsed from 21.8 uses per 10k words in the "
     "1790s to 0.35 today. “Must” peaked in the FDR and war years; promising, "
     "future-facing “will” took over modern speech."),
    ("naming", "From “the United States” to “America”",
     "In 1800 the country was named as a legal entity seven times more often than as an "
     "idea. The lines cross in the 1950s; by 2000 “America” leads eight to one. "
     "The republic became a brand."),
    ("orientation", "Nostalgia is catching the future",
     "Restoration language (“again / restore / back to”) vs future language. "
     "Future-talk won the entire twentieth century. The 1980s brought the first nostalgia "
     "wave; in the 2020s nostalgia surges again while future-talk falls to its lowest "
     "level since WWII."),
    ("religion", "“God bless” is a television-era invention",
     "The phrase does not occur in a single 19th-century speech in the corpus. It appears "
     "in the 1950s and becomes mandatory by Reagan. Broader civil-religion language "
     "doubled from 1800 to today - presidential speech got more religious as the country "
     "secularized."),
    ("hopefear", "Hope and fear",
     "NRC Emotion Lexicon scores. Hope language (trust, anticipation, joy) and fear "
     "language (fear, anger) per 10,000 words - the raw material of the profile pages' "
     "hope and fear scores."),
    ("readability", "Speeches dropped twelve grade levels",
     "Median Flesch-Kincaid reading level fell from grade 19.9 in the 1790s to grade 7.8 in "
     "the 2020s. Hover any dot to see the speech behind it."),
    ("issues", "What presidents actually cared about, 1789-2026",
     "Anchored topic model over 36,000 paragraph-sized chunks: a fixed issue taxonomy plus "
     "discovered topics, so every era is scored on the same axes. Money & banking dies "
     "after the gold-standard era, agriculture fades with the family farm, health care and "
     "education are late-20th-century arrivals - and immigration's 2020s spike exceeds "
     "anything before it."),
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
{PAGE_CSS}
  .tiles {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 26px 0 8px; }}
  .tile {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 10px; padding: 14px 22px; min-width: 130px; }}
  .tile .num {{ font-size: 1.55rem; font-weight: 650; }}
  .tile .lbl {{ color: var(--muted); font-size: 0.82rem; }}
  .profiles-link {{ display: inline-block; margin-top: 18px; font-size: 1rem;
                    color: var(--ink); font-weight: 600; }}
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
  <a class="profiles-link" href="presidents/index.html">Browse the 45 president profiles →</a>
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
    emb = similarity.build_embeddings(df)
    sim = similarity.similarity_matrix(emb)
    adj = similarity.build_adjusted(emb)
    kw = trends.keyword_trends(df)
    distinctive = trends.distinctive_terms(df)
    markers = indices.build_markers(df)
    rates = indices.decade_rates(markers)
    _, issue_meta = issues.build_issues()
    para_labels = pd.read_parquet(issues.PARA_LABELS_PATH)

    figs = {
        "map": fig_map(emb),
        "charmap": fig_map(adj),
        "heatmap": fig_heatmap(sim),
        "certainty": fig_certainty(markers),
        "pronouns": fig_pronouns(stats),
        "modals": fig_modals(stats),
        "naming": fig_naming(rates),
        "orientation": fig_orientation(rates),
        "religion": fig_religion(rates),
        "hopefear": fig_hope_fear(rates),
        "readability": fig_readability(stats),
        "issues": fig_issues_decade(para_labels, issue_meta["issues"]),
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

    profile_data = profiles.build_profile_data()
    profiles_site.write_profiles(profile_data, SITE_DIR)

    if args.inline:
        out2 = SITE_DIR / "index_selfcontained.html"
        out2.write_text(build_html(figs, stats_line, inline=True))
        print(f"wrote {out2.relative_to(REPO_ROOT)} ({out2.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
