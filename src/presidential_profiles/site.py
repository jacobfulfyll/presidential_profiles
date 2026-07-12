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


X_RANGE = [1786, 2029]


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


def fig_kinships(adj: pd.DataFrame, min_gap: int = 30, top_n: int = 12) -> go.Figure:
    """The exact numbers behind cross-era voice kinship: top era-adjusted
    similarity pairs at least min_gap years apart. A 2-D map necessarily
    distorts pairwise distances; this chart doesn't."""
    vec_cols = [c for c in adj.columns if c.startswith("e")]
    V = adj[vec_cols].to_numpy(dtype=float)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    sim = V @ V.T
    names = adj["president"].tolist()
    years = adj["first_year"].to_numpy()

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            gap = abs(int(years[i] - years[j]))
            if gap >= min_gap:
                pairs.append((float(sim[i, j]), names[i], names[j], gap))
    pairs.sort(reverse=True)
    top = pairs[:top_n][::-1]

    labels = [f"{a} ↔ {b}" for _, a, b, _ in top]
    fig = go.Figure(go.Bar(
        y=labels, x=[s for s, *_ in top], orientation="h",
        marker=dict(color=BLUE_RAMP[4]),
        customdata=[g for *_, g in top],
        hovertemplate="%{y}<br>adjusted similarity %{x:.2f} · "
                      "%{customdata} years apart<extra></extra>",
    ))
    fig.update_layout(**_layout(
        height=470, margin=dict(l=10, r=24, t=24, b=48),
        xaxis=dict(title=dict(text="era-adjusted voice similarity",
                              font=dict(size=11, color=INK2)),
                   gridcolor=GRID, tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(tickfont=dict(color=INK, size=12.5), gridcolor=SURFACE),
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


def fig_pronouns(stats: pd.DataFrame) -> go.Figure:
    py = _stats_president_year(stats, ["we_count", "i_count"])
    return _two_line_fig(
        [("we / us / our", _stats_yearly(stats, "we_count"),
          _president_dots(py, "we_count", SERIES[0])),
         ("I / me / my", _stats_yearly(stats, "i_count"),
          _president_dots(py, "i_count", SERIES[1]))],
        ytitle="uses per 10,000 words",
    )


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
    med = (s.groupby("year")["fk"].median()
           .reindex(pd.RangeIndex(int(s["year"].min()), int(s["year"].max()) + 1))
           .rolling(7, center=True, min_periods=3).median())
    fig.add_trace(go.Scatter(
        x=med.index, y=med.values, mode="lines", name="rolling median (7 yr)",
        line=dict(color=BLUE_RAMP[5], width=3), connectgaps=False,
        hovertemplate="median grade %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(**_timeline_layout(yaxis_title="Flesch-Kincaid grade level"))
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
    """Share of paragraphs touching each curated issue, in 5-year periods.
    Periods backed by fewer than 40 paragraphs are dropped."""
    pl = para_labels.copy()
    pl["period"] = (pl["year"] // 5) * 5
    counts = pl.groupby("period").size()
    valid = counts[counts >= 40].index
    display = issue_names + ["Discovered 5"]
    panels = []
    for name in display:
        label = profiles_site.DISCOVERED_LABELS.get(name, name)
        share = pl.groupby("period")[name].mean() * 100
        panels.append((label, share.loc[share.index.isin(valid)]))
    return _small_multiples(panels, rows=4, cols=4, height=880,
                            hovertemplate="%{y:.1f}% of paragraphs")


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
    return _two_line_fig(
        [("future (future / forward / tomorrow)", rates["future"],
          _president_dots(py, "future", SERIES[0])),
         ("nostalgia (again / restore / back to)", rates["nostalgia"],
          _president_dots(py, "nostalgia", SERIES[1]))],
        ytitle="uses per 10,000 words",
    )


def fig_religion(rates: pd.DataFrame, py: pd.DataFrame) -> go.Figure:
    return _two_line_fig(
        [("civil religion (god / faith / pray / bless / sacred)",
          rates["religiosity"], _president_dots(py, "religiosity", SERIES[0])),
         ("“God bless”", rates["god_bless"],
          _president_dots(py, "god_bless", SERIES[1]))],
        ytitle="uses per 10,000 words",
    )


def fig_hope_fear(rates: pd.DataFrame, py: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.58, 0.42],
                        vertical_spacing=0.1,
                        subplot_titles=["hope and fear words per 10,000",
                                        "the balance: fear ÷ hope"])
    fig.add_trace(_president_dots(py, "nrc_hope", SERIES[0]), row=1, col=1)
    fig.add_trace(_president_dots(py, "nrc_fear", SERIES[1]), row=1, col=1)
    for i, (name, col) in enumerate([("hope words", "nrc_hope"),
                                     ("fear words", "nrc_fear")]):
        fig.add_trace(go.Scatter(
            x=rates.index, y=rates[col], name=name, mode="lines",
            line=dict(color=SERIES[i], width=2.4), connectgaps=False,
            hovertemplate="%{y:.0f} per 10k<extra>" + name + "</extra>",
        ), row=1, col=1)

    ratio = rates["nrc_fear"] / rates["nrc_hope"]
    fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio.values, name="fear ÷ hope", mode="lines",
        line=dict(color=SERIES[4], width=2.4), connectgaps=False,
        hovertemplate="fear/hope %{y:.2f}<extra></extra>",
    ), row=2, col=1)
    for year, label in [(1860, "1860: eve of the Civil War"), (1941, "WWII"),
                        (1983, "1983"), (2026, "2026: highest since WWII")]:
        if year in ratio.index and pd.notna(ratio[year]):
            fig.add_annotation(x=year, y=float(ratio[year]), xref="x2", yref="y2",
                               text=label, showarrow=True, arrowhead=0, ax=0, ay=-26,
                               arrowcolor=MUTED,
                               font=dict(size=10.5, color=INK2))

    fig.update_layout(**_layout(height=620, margin=dict(l=56, r=24, t=40, b=44)))
    fig.update_xaxes(range=X_RANGE, gridcolor=GRID, linecolor=BASELINE,
                     tickfont=dict(color=MUTED, size=11))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE,
                     tickfont=dict(color=MUTED, size=11))
    fig.update_annotations(font=dict(size=12.5, color=INK), selector=dict(ay=0))
    return fig


def fig_issues_opponents(rates: pd.DataFrame, py: pd.DataFrame,
                         para_labels: pd.DataFrame, issue_names: list[str]) -> go.Figure:
    """Are presidents talking about policy or about opponents?"""
    pl = para_labels.copy()
    pl["period"] = (pl["year"] // 5) * 5
    counts = pl.groupby("period").size()
    display = issue_names + ["Discovered 5"]
    any_issue = pl.groupby("period")[display].apply(lambda g: g.any(axis=1).mean()) * 100
    any_issue = any_issue[counts >= 40]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.11,
                        subplot_titles=["share of paragraphs about a policy issue (%)",
                                        "opponent talk per 10,000 words"])
    fig.add_trace(go.Scatter(
        x=any_issue.index, y=any_issue.values, mode="lines", name="policy issue share",
        line=dict(color=BLUE_RAMP[4], width=2.4),
        fill="tozeroy", fillcolor="rgba(109, 167, 236, 0.25)",
        hovertemplate="%{y:.0f}% of paragraphs<extra></extra>", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(_president_dots(py, "opponents", SERIES[5]), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=rates.index, y=rates["opponents"], mode="lines", name="opponent talk",
        line=dict(color=SERIES[5], width=2.4), connectgaps=False,
        hovertemplate="%{y:.1f} per 10k<extra></extra>", showlegend=False,
    ), row=2, col=1)
    fig.update_layout(**_layout(height=560, margin=dict(l=56, r=24, t=40, b=44),
                                showlegend=False))
    fig.update_xaxes(range=X_RANGE, gridcolor=GRID, linecolor=BASELINE,
                     tickfont=dict(color=MUTED, size=11))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, rangemode="tozero",
                     tickfont=dict(color=MUTED, size=11))
    fig.update_annotations(font=dict(size=12.5, color=INK))
    return fig


def fig_keywords(kw: pd.DataFrame) -> go.Figure:
    panels = [
        (term, kw[kw["term"] == term].set_index("period")["rate"])
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


# The headline findings. Each links to the section that carries the evidence.
FINDINGS = [
    ("0.58 → 0.93", "Presidents stopped hedging. On inaugural addresses alone - the same "
     "genre for 240 years - the assertive share of stance language rose from balanced to "
     "near-total. The grammar of trade-offs is nearly extinct.", "certainty"),
    ("highest since WWII", "The fear-to-hope balance in 2025-26 is the most fearful "
     "since the Second World War - and the earlier peaks are 1983 and the eve of the "
     "Civil War. Hope itself is at its lowest level in 240 years.", "hopefear"),
    ("Truman ×2", "One president holds both emotional records: the most hopeful "
     "substantial speech ever (four days after FDR died) and the most fearful "
     "(announcing the Korea emergency).", "records"),
    ("“I” > any decade since 1790s", "The 2020s broke a century-long rise of “we”: "
     "self-reference is at its highest since George Washington spoke for himself.",
     "pronouns"),
    ("7:1 → 1:8", "“The United States” became “America”: the country stopped being named "
     "as a legal entity and became an idea. The lines cross in the 1950s.", "naming"),
    ("Lincoln ↔ FDR", "Remove each president's era from his voice, and the closest pair "
     "across any century is Lincoln and FDR - the two crisis unifiers.", "charmap"),
]

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
}

# key, chapter (eyebrow, only where a new chapter starts), title, prose
SECTIONS = [
    ("certainty", "The headline finding", "Presidents stopped hedging",
     "In 1790, a president qualified a military order with “however … the dictates "
     "of humanity” in the same sentence. In 2025, nothing is qualified. The chart is the "
     "assertive share of stance language - boosters and will/must against hedges and "
     "concessives - and each diamond is one inaugural address, the same genre across 240 "
     "years. The two most recent inaugurals sit near 0.98: almost pure assertion. "
     "Hover any dot to see whose year it was."),
    ("hopefear", "The last five years", "The most fearful balance since WWII",
     "Fear levels alone are not unprecedented - the 1940s ran hotter. The story is the "
     "balance: hope has fallen to its lowest level in 240 years while fear runs high, "
     "putting the 2025-26 fear-to-hope ratio (0.61) above everything except the Second "
     "World War itself. The other peaks on the lower panel say what company the present "
     "keeps: 1983, and 1854-1861 - the eve of the Civil War."),
    ("orientation", None, "Nostalgia is beating the future",
     "Future language won every decade of the twentieth century - often two to one over "
     "restoration language. There have been three nostalgia waves: the 1850s (Lincoln and "
     "Buchanan pleading to restore the Union as it fractured), the 1980s (Reagan's "
     "“again”), and the 2020s - the only one where future-talk is "
     "simultaneously at its lowest since WWII. Both this chart and the fear-hope balance "
     "above point at the same historical rhyme: the 1850s."),
    ("distinctive", None, "The words that mark the new era",
     "Vocabulary statistically distinctive of speeches since April 2019 against the "
     "1989-2019 baseline, register words filtered out: <em>ukraine, china, testing, "
     "border</em>. What presidents talk about changed; how they talk changed more - see "
     "the next chart."),
    ("issuesopp", None, "From issues to opponents",
     "Two measures of what a president's words are for. Top: the share of paragraphs "
     "that touch any policy issue in the taxonomy. Bottom: opponent talk - references to "
     "the other party, opponents, politicians, the media. Policy share erodes in the "
     "modern era while opponent talk climbs to all-time highs in the 2010s-2020s: "
     "presidential speech is increasingly about who's wrong rather than what to do."),
    ("records", "The record book", "The most extreme speeches ever given",
     "Substantial speeches only (1,500+ words), rated per 10,000 words. Truman holds both "
     "emotional records. The most absolutist speech in presidential history is Nixon's "
     "farewell to his staff, the morning he resigned - a man consoling himself in "
     "“always” and “never”. Below the length bar, short statements "
     "spike higher still: the most fearful short statement ever is Trump's June 1, 2020 "
     "remarks on the protests. Links go to the full transcripts."),
    ("charmap", "Who presidents are", "The character map: era removed",
     "Subtract each president's era from his voice and what remains is character. The "
     "layout preserves pairwise distances (MDS on the adjusted similarities), but any "
     "2-D map distorts - the ranked list below it carries the exact numbers. Every "
     "president's own kinships are on their profile page."),
    ("kinships", None, "The strongest kinships across the centuries, exactly",
     "The top era-adjusted voice pairs at least 30 years apart - the precise values "
     "behind the map. Lincoln ↔ FDR is the strongest cross-era kinship in the "
     "corpus; Lincoln appears three times in the top ranks."),
    ("map", None, "The river of history",
     "Why the adjustment above is necessary: raw voice similarity is two-thirds era. "
     "Project the unadjusted embeddings and time flows left to right almost perfectly, "
     "even though the model never sees a date. Both maps are true - this one is about "
     "eras, the one above is about people."),
    ("heatmap", None, "The language of eras",
     "The same era effect as a matrix: cosine similarity in chronological order. The dark "
     "block from FDR onward is the modern voice forming. The most similar pair anywhere: "
     "Bill Clinton and Barack Obama (0.978)."),
    ("pronouns", "The long arc", "The 2020s flipped a century of “we”",
     "Presidential speech spent a hundred years becoming more collective - “we” "
     "quadrupled from 1900 to the 2010s. Then the 2020s reversed it: “we” fell "
     "for the first time in a century while “I” surged to its highest rate "
     "since George Washington - who was, to be fair, mostly introducing himself."),
    ("naming", None, "From “the United States” to “America”",
     "In 1800 the country was named as a legal entity seven times more often than as an "
     "idea. The lines cross in the 1950s - television, again - and by 2000 "
     "“America” leads eight to one. The republic became a brand."),
    ("religion", None, "“God bless” is a television-era invention",
     "The phrase does not occur in a single 19th-century speech in this corpus. It appears "
     "in the 1950s and becomes mandatory by Reagan. Presidential speech got more religious "
     "as the country secularized."),
    ("readability", None, "Twelve grade levels, gone",
     "Median reading level fell from grade 19.9 in the 1790s to grade 7.8 in the 2020s. "
     "Every dot is one speech - hover to see which. The all-time floor is in the last "
     "few years."),
    ("issues", None, "What presidents actually cared about",
     "36,000 paragraph-sized chunks, scored against one issue taxonomy across all eras. "
     "Money & banking dies with the gold standard, agriculture fades with the family "
     "farm, health care and education arrive only in the late twentieth century - and "
     "immigration's 2020s spike exceeds anything in 240 years, including the Ellis "
     "Island era."),
    ("keywords", None, "One word at a time",
     "Raw rates for single terms. “Border” and “immigration” at "
     "all-time highs; “tariff” back from the dead after a century; "
     "“constitution” never recovered from the 1860s."),
]


RECORD_SPECS = [
    ("Most fearful", "nrc_fear", "fear words / 10k"),
    ("Most hopeful", "nrc_hope", "hope words / 10k"),
    ("Most absolutist", "boosters", "boosters / 10k"),
    ("Most us-vs-them", "us_them", "they / them / 10k"),
    ("Most superlative", "superlatives", "superlatives / 10k"),
]


def compute_records(markers: pd.DataFrame, min_words: int = 1_500) -> list[dict]:
    """Substantial speeches only: below ~1,500 words, short ceremonial
    statements top every emotion category on lexicon density alone."""
    m = markers[markers["n_words"] >= min_words]
    records = []
    for label, col, unit in RECORD_SPECS:
        rate = (m[col] / m["n_words"] * 10_000).sort_values(ascending=False)
        row, runner = m.loc[rate.index[0]], m.loc[rate.index[1]]
        records.append({
            "label": label,
            "value": f"{rate.iloc[0]:.0f}",
            "unit": unit,
            "president": row["president"],
            "year": int(row["year"]),
            "title": row["title"],
            "url": profiles.MILLER_URL + row["doc_name"],
            "runner": f"{runner['president']}, {runner['title'].split(':', 1)[-1].strip()}"
                      f" ({rate.iloc[1]:.0f})",
        })
    return records


def _records_html(records: list[dict]) -> str:
    cards = []
    for r in records:
        cards.append(f"""<div class="r-card">
  <div class="r-label">{r["label"]}</div>
  <div class="r-value">{r["value"]} <span class="r-unit">{r["unit"]}</span></div>
  <div class="r-who">{r["president"]}, {r["year"]}</div>
  <a class="r-title" href="{r["url"]}" target="_blank" rel="noopener">{r["title"]}</a>
  <div class="r-runner">runner-up: {r["runner"]}</div>
</div>""")
    return '<div class="records">' + "\n".join(cards) + "</div>"


def _quotes_html(key: str) -> str:
    pair = QUOTE_PAIRS.get(key)
    if not pair:
        return ""
    blocks = "\n".join(
        f'<blockquote><p>“{text}”</p><cite>{who}</cite></blockquote>'
        for text, who in pair
    )
    return f'<div class="quotepair">{blocks}</div>'


def build_html(figs: dict[str, go.Figure], stats_line: dict, records: list[dict],
               inline: bool) -> str:
    from plotly.offline import get_plotlyjs

    plotly_src = (
        f"<script>{get_plotlyjs()}</script>"
        if inline
        else '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>'
    )
    fig_json = {k: json.loads(pio.to_json(f)) for k, f in figs.items()}

    # Explicit pixel heights keep plotly's percent-sized inner containers from
    # collapsing when its responsive handler re-renders after a window resize.
    parts = []
    for key, chapter, title, prose in SECTIONS:
        if chapter:
            parts.append(f'<div class="eyebrow">{chapter}</div>')
        if key == "records":
            body = _records_html(records)
        else:
            body = (f'<div class="chart-scroll"><div class="chart" data-fig="{key}"'
                    f' style="height:{figs[key].layout.height}px"></div></div>')
        parts.append(f"""<section id="{key}">
  <h2>{title}</h2>
  <p>{prose}</p>
  {_quotes_html(key)}
  {body}
</section>""")
    sections_html = "\n".join(parts)

    findings_html = "\n".join(
        f"""<a class="f-card" href="#{anchor}">
  <div class="f-stat">{stat}</div>
  <div class="f-text">{text}</div>
</a>"""
        for stat, text, anchor in FINDINGS
    )

    corpus_line = (
        f"{stats_line['speeches']:,} speeches · {stats_line['words'] / 1e6:.1f}M words · "
        f"{stats_line['presidents']} presidents · {stats_line['start']} - April "
        f"{stats_line['end']}"
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
  .corpus-line {{ color: var(--muted); font-size: 0.86rem; margin-top: 14px; }}
  .findings {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
               gap: 12px; margin-top: 24px; }}
  .f-card {{ background: var(--surface); border: 1px solid var(--border);
             border-radius: 12px; padding: 16px 18px; text-decoration: none;
             color: var(--ink); transition: border-color 0.15s; }}
  .f-card:hover {{ border-color: var(--muted); }}
  .f-stat {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -0.01em; }}
  .f-text {{ color: var(--ink2); font-size: 0.9rem; margin-top: 7px; line-height: 1.45; }}
  .eyebrow {{ margin-top: 54px; color: var(--muted); font-size: 0.78rem;
              font-weight: 650; letter-spacing: 0.09em; text-transform: uppercase; }}
  .eyebrow + section {{ margin-top: 10px; }}
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
  .r-who {{ margin-top: 6px; font-weight: 600; font-size: 0.94rem; }}
  .r-title {{ display: block; color: var(--ink2); font-size: 0.85rem; margin-top: 4px; }}
  .r-runner {{ color: var(--muted); font-size: 0.78rem; margin-top: 8px; }}
  .profiles-link {{ display: inline-block; margin-top: 20px; font-size: 1rem;
                    color: var(--ink); font-weight: 600; }}
</style>
</head>
<body>
<header>
  <h1>Presidential Profiles</h1>
  <p class="sub">What 240 years of presidential speech actually says - and how it is
  changing right now. Every finding below is computed from the full Miller Center corpus
  and links to its evidence; all charts are interactive.</p>
  <div class="findings">
{findings_html}
  </div>
  <p class="corpus-line">{corpus_line}</p>
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
  <p>Method note: timelines are 5-year centered rolling rates weighted by word count,
  through April 2026. Points backed by under 20,000 words are not plotted - the corpus
  before ~1790 is a handful of speeches, and one personal inaugural should not set a
  national trend line. Dots are per president per year: 46 years have more than one
  president speaking, and the corpus files a few famous pre-presidency speeches
  (Nixon's Checkers speech, Reagan's "A Time for Choosing") under the later president.</p>
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
    rates = indices.yearly_rates(markers)
    py = indices.president_year_rates(markers)
    _, issue_meta = issues.build_issues()
    para_labels = pd.read_parquet(issues.PARA_LABELS_PATH)

    figs = {
        "map": fig_map(emb),
        "charmap": fig_map(adj),
        "heatmap": fig_heatmap(sim),
        "certainty": fig_certainty(markers, py),
        "kinships": fig_kinships(adj),
        "pronouns": fig_pronouns(stats),
        "naming": fig_naming(rates, py),
        "orientation": fig_orientation(rates, py),
        "religion": fig_religion(rates, py),
        "hopefear": fig_hope_fear(rates, py),
        "readability": fig_readability(stats),
        "issues": fig_issues_decade(para_labels, issue_meta["issues"]),
        "issuesopp": fig_issues_opponents(rates, py, para_labels, issue_meta["issues"]),
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

    records = compute_records(markers)

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out = SITE_DIR / "index.html"
    out.write_text(build_html(figs, stats_line, records, inline=False))
    print(f"wrote {out.relative_to(REPO_ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")

    profile_data = profiles.build_profile_data()
    profiles_site.write_profiles(profile_data, SITE_DIR)

    if args.inline:
        out2 = SITE_DIR / "index_selfcontained.html"
        out2.write_text(build_html(figs, stats_line, records, inline=True))
        print(f"wrote {out2.relative_to(REPO_ROOT)} ({out2.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
