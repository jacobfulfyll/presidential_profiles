"""Generate the static dashboard site: every trend as an interactive chart.

Writes docs/index.html (plotly.js from CDN, suitable for GitHub Pages).
With --inline, also writes a fully self-contained copy for offline sharing.
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from . import corpus, explorer, indices, issues, issues_site, portraits, profiles, profiles_site, rhetoric, similarity, trends
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
    cards = []
    for sim_val, a, b, gap in pairs[:top_n]:
        img_a = f'<img src="{faces[a]}" alt="">' if a in faces else ""
        img_b = f'<img src="{faces[b]}" alt="">' if b in faces else ""
        cards.append(f"""<div class="k-card">
  <div class="k-faces">{img_a}<span class="k-link"></span>{img_b}</div>
  <div class="k-names">{a} ↔ {b}</div>
  <div class="k-meta">similarity {sim_val:.2f} · {gap} years apart</div>
</div>""")
    return '<div class="kinships">' + "\n".join(cards) + "</div>"


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
                     height: int, hovertemplate: str,
                     dots: dict | None = None,
                     dots_unit: str = "") -> go.Figure:
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
    fig.update_layout(**_layout(height=height, margin=dict(l=40, r=16, t=48, b=36)))
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED, size=10))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED, size=10),
                     rangemode="tozero")
    fig.update_annotations(font=dict(size=12.5, color=INK))
    return fig


def fig_issues_decade(para_labels: pd.DataFrame, issue_names: list[str],
                      issue_df: pd.DataFrame | None = None,
                      scores: pd.DataFrame | None = None) -> go.Figure:
    """Share of paragraphs touching each curated issue, in 5-year periods,
    with per-president dots (each president's own share, at their term
    midpoint)."""
    pl = para_labels.copy()
    pl["period"] = (pl["year"] // 5) * 5
    counts = pl.groupby("period").size()
    valid = counts[counts >= 40].index
    display = issue_names + ["Discovered 5"]

    dots = None
    if issue_df is not None and scores is not None:
        d = issue_df.set_index("president")
        mids = (scores["first_year"] + scores["last_year"]) / 2
        dots = {}
        for name in display:
            label = profiles_site.DISCOVERED_LABELS.get(name, name)
            dots[label] = pd.DataFrame({
                "x": mids.values,
                "y": (d.loc[mids.index, f"share_{name}"] * 100).values,
                "name": mids.index,
            })

    panels = []
    for name in display:
        label = profiles_site.DISCOVERED_LABELS.get(name, name)
        share = pl.groupby("period")[name].mean() * 100
        panels.append((label, share.loc[share.index.isin(valid)]))
    return _small_multiples(panels, rows=4, cols=4, height=880,
                            hovertemplate="%{y:.1f}% of paragraphs",
                            dots=dots, dots_unit="% of their speech")


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
    """Loud vs deep: unbounded language against policy machinery."""
    fig = go.Figure()
    x_span = float(scores["mechanism"].max()) * 1.15
    y_span = float(scores["hype"].max()) * 1.15
    for pres, row in scores.iterrows():
        x, y = float(row["mechanism"]), float(row["hype"])
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", showlegend=False,
            marker=dict(size=26, opacity=0),
            customdata=[[pres]],
            hovertemplate="<b>%{customdata[0]}</b><br>machinery %{x:.0f} · "
                          "hype %{y:.1f} per 10k<extra></extra>",
        ))
        if pres in faces:
            fig.add_layout_image(
                source=faces[pres], x=x, y=y, xref="x", yref="y",
                sizex=x_span * 30 / 860, sizey=y_span * 30 / 470,
                xanchor="center", yanchor="middle", layer="above",
            )
    for x, y, text in [(x_span * 0.10, y_span * 0.95, "loud, no machinery"),
                       (x_span * 0.85, y_span * 0.05, "quiet, heavy machinery")]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                           font=dict(size=11, color=MUTED))
    fig.update_layout(**_layout(
        height=560, width=FACE_CHART_WIDTH,
        xaxis=dict(title=dict(text="policy machinery per 10k (act / bill / treaty / appropriation)",
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
    for year, label in [(1812, "War of 1812"), (1860, "1860: eve of the Civil War"),
                        (1894, "Pullman strike"), (1941, "WWII"),
                        (1983, "1983: Reagan's Cold War"), (2002, "9/11 → Iraq"),
                        (2026, "2026: highest since WWII")]:
        if year in ratio.index and pd.notna(ratio[year]):
            fig.add_annotation(x=year, y=float(ratio[year]),
                               text=label, showarrow=True, arrowhead=0, ax=0, ay=-30,
                               arrowcolor=MUTED, font=dict(size=11, color=INK2))
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
               for term, terms in trends.TERM_GROUPS.items()},
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
            for term in trends.TERM_GROUPS
        }
    panels = [
        (term, kw[kw["term"] == term].set_index("period")["rate"])
        for term in kw["term"].unique()
    ]
    return _small_multiples(panels, rows=3, cols=3, height=680,
                            hovertemplate="%{y:.1f} per 10k words",
                            dots=dots, dots_unit=" per 10k")


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
    ("7:1 → 1:8", "“The United States” became “America”: the country stopped being named "
     "as a legal entity and became an idea. The lines cross in the 1950s.", "naming"),
    ("Lincoln ↔ FDR", "Remove each president's era from his voice, and the closest pair "
     "across any century is Lincoln and FDR - the two crisis unifiers.", "kinships"),
    ("hype ×3", "Unbounded language - greatest, unstoppable, historic, of all time - "
     "tripled in the present era after two centuries of slow growth. Doom barely moved: "
     "the explosion is self-superlatives.", "hypedoom"),
    ("machinery ÷5", "The vocabulary of actually governing - act, bill, treaty, "
     "appropriation - has fallen five-fold since the 19th century. Speeches got louder "
     "and emptier at the same time.", "quadrant"),
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
     "of humanity” in the same sentence. In 2025, nothing is qualified. Every "
     "president sits at their assertive share of stance language - boosters and will/must "
     "against hedges and concessives - with the line under each face spanning their years "
     "in the corpus. The gray line is the corpus-wide rolling rate. No president before "
     "1900 sits where any president after 1950 does."),
    ("hypedoom", None, "The volume knob: hype and doom",
     "Unbounded language, split by direction. Hype - greatest, unstoppable, historic, "
     "of all time - tripled in the present era after two centuries of slow growth. Doom "
     "- worst, disaster, carnage - rose only modestly; the only era where doom beat hype "
     "was 1900-1949, the Depression and the wars. What exploded in the 2020s is "
     "self-superlatives, not catastrophizing."),
    ("quadrant", None, "Loud vs deep",
     "Two measures against each other: hype (up) and policy machinery - act, bill, "
     "treaty, section, appropriation, the vocabulary of actually governing - which has "
     "collapsed five-fold since the 19th century (right). The old presidency lives in "
     "the bottom right: quiet and procedural. The modern one has drifted up and left. "
     "One corner is empty: nobody is loud AND procedural."),
    ("hopefear", "The last five years", "The most fearful balance since WWII",
     "Fear words divided by hope words. Hope has fallen to its lowest level in 240 years "
     "while fear runs high, putting the 2025-26 ratio (0.61) above everything except the "
     "Second World War itself. The other peaks say what company the present keeps: 1983, "
     "and 1854-1861 - the eve of the Civil War. Each faint dot is one president's year."),
    ("readability", None, "Twelve grade levels, gone",
     "Median reading level fell from grade 19.9 in the 1790s to grade 7.8 in the 2020s. "
     "Every dot is one speech - hover to see which. The all-time floor is in the last "
     "few years."),
    ("issues", None, "What presidents actually cared about",
     "36,000 paragraph-sized chunks, scored against one issue taxonomy across all eras. "
     "Money & banking dies with the gold standard, agriculture fades with the family "
     "farm, health care and education arrive only in the late twentieth century - and "
     "immigration's 2020s spike exceeds anything in 240 years, including the Ellis "
     "Island era. Every issue has its own page - timeline, owners, and defining "
     "quotes - in the <a href='issues/index.html'>issue profiles</a>."),
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


def build_html(figs: dict[str, go.Figure], stats_line: dict, bodies: dict[str, str],
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
        if key in bodies:
            body = bodies[key]
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
  .kinships {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
               gap: 12px; }}
  .k-card {{ background: var(--surface); border: 1px solid var(--border);
             border-radius: 12px; padding: 16px 18px; text-align: center; }}
  .k-faces {{ display: flex; align-items: center; justify-content: center; gap: 4px; }}
  .k-faces img {{ width: 52px; height: 52px; border-radius: 50%; }}
  .k-link {{ width: 26px; border-top: 2px dotted var(--muted); }}
  .k-names {{ font-weight: 650; font-size: 0.92rem; margin-top: 10px; }}
  .k-meta {{ color: var(--muted); font-size: 0.8rem; margin-top: 4px; }}
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
  &nbsp; <a class="profiles-link" href="explorer.html">Look up any word or phrase →</a>
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

    scores = indices.president_scores(markers, stats, df).set_index("president")
    # Presidents with a handful of speeches distort per-president charts
    # (Garfield's one speech made him a certainty outlier); they keep their
    # profile pages but sit out the dashboard graphics.
    sparse = set(scores[scores["n_speeches"] < 5].index)
    scores = scores[scores["n_speeches"] >= 5]
    py = py[~py["president"].isin(sparse)].reset_index(drop=True)
    faces = portraits.data_uris(list(scores.index))
    issue_df, _ = issues.build_issues()

    figs = {
        "map": fig_map(emb),
        "heatmap": fig_heatmap(sim),
        "certainty": fig_certainty_faces(scores, markers, faces),
        "hypedoom": fig_hype_doom(rates, py),
        "quadrant": fig_quadrant(scores, faces),
        "families": fig_families(df),
        "naming": fig_naming(rates, py),
        "hopefear": fig_hope_fear(rates, py),
        "readability": fig_readability(stats),
        "issues": fig_issues_decade(para_labels, issue_meta["issues"],
                                    issue_df, scores),
        "keywords": fig_keywords(kw, df, scores),
    }
    bodies = {
        "records": _records_html(compute_records(markers)),
        "kinships": _kinships_html(kinship_pairs(adj), faces),
        "distinctive": _era_vocab_html(trends.era_vocabulary(df)),
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
    out.write_text(build_html(figs, stats_line, bodies, inline=False))
    print(f"wrote {out.relative_to(REPO_ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")

    profile_data = profiles.build_profile_data()
    profiles_site.write_profiles(profile_data, SITE_DIR)

    if not (explorer.EXPLORER_DIR / "meta.json").exists():
        explorer.build_explorer_data()
    explorer.write_page()
    issues_site.write_issue_pages(SITE_DIR, issue_df, issue_meta, scores, faces)

    if args.inline:
        out2 = SITE_DIR / "index_selfcontained.html"
        out2.write_text(build_html(figs, stats_line, bodies, inline=True))
        print(f"wrote {out2.relative_to(REPO_ROOT)} ({out2.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
