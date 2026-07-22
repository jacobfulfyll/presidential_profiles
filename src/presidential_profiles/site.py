"""Generate the static dashboard site: every trend as an interactive chart.

Writes docs/index.html (plotly.js from CDN, suitable for GitHub Pages).
With --inline, also writes a fully self-contained copy for offline sharing.
"""

import argparse
import html
import json
import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from . import bands, compare_site, corpus, explorer, indices, issues, issues_site, portraits, profiles, profiles_site, rhetoric, similarity, topic_quality, trends, word_families
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


# One string for both branches. The dots are per (president, PERIOD) now, and a
# hover reading "12.3% of their paragraphs" invites a reader to hear a
# whole-presidency share — a different quantity from the one plotted. The issue
# pages were updated when the dots changed and the dashboard was not; sharing
# the constant is what stops the two renderings of one function drifting again.
DOTS_UNIT = "% of their paragraphs in this period"


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
                            show_components: bool = False) -> go.Figure:
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
    return _banded_small_multiples(panels, rows=4, cols=4, height=880,
                                   unit="% of paragraphs", dots=dots,
                                   dots_unit=DOTS_UNIT,
                                   x_range=_x_range_covering(panels))


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
    return _banded_small_multiples(panels, rows=rows, cols=cols, height=940,
                                   unit="% of paragraphs", x_range=X_RANGE,
                                   show_components=True)


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
     "The shaded band around each line is a 95% interval from a bootstrap that resamples "
     "whole speeches - so it balloons where a 5-year period rests on a handful of them - "
     "a dotted segment marks periods too thin to read as a trend, and a hollow ring "
     "marks a point with no interval at all: every speech in that period agreed exactly, "
     "on too few speeches for the agreement to carry information. The share is plotted, "
     "the uncertainty is unknown - which is not the same as small. These labels come "
     "from a deterministic topic model with no AI annotator in the loop, so the band "
     "covers sampling error only. "
     "Money & banking dies with the gold standard, agriculture fades with the family "
     "farm, health care and education arrive only in the late twentieth century - and "
     "immigration's 2020s spike exceeds anything in 240 years, including the Ellis "
     "Island era. Every issue has its own page - timeline, owners, and defining "
     "quotes - in the <a href='issues/index.html'>issue profiles</a>."),
    ("llm_topics", None, "When the labels come from an AI, the AI's doubt is part of the answer",
     "The same corpus, labeled a second way: every paragraph read by Claude against a "
     "50-topic taxonomy built from the corpus itself. A <em>sample</em> of those speeches "
     "was then read again by a second, different model, and where the two disagreed about "
     "an era that disagreement is added to the band - so these intervals carry two things "
     "the chart above cannot, sampling error <em>and</em> annotator error. Panels are the "
     "twelve largest topics by share of all paragraphs, across the nine eras."),
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
               inline: bool, prose_extra: dict[str, str] | None = None) -> str:
    """`prose_extra` appends a data-derived sentence to a section's prose —
    see `llm_divergence_sentence` for why that claim is not typed into
    `SECTIONS` as a literal."""
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
        # A section whose figure could not be built is omitted entirely rather
        # than rendered as an empty box with prose describing a chart that is
        # not there. `llm_topics` is the live case: it needs data/bands.parquet.
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
        if prose_extra and prose_extra.get(key):
            prose = f"{prose} {prose_extra[key]}"
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
  changing right now. This corpus is the presidency's most formal register: the major
  prepared addresses curated by the Miller Center, not rallies or tweets - presidents
  with their best foot forward. Every finding below holds even there. All charts are
  interactive and every finding links to its evidence.</p>
  <div class="findings">
{findings_html}
  </div>
  <p class="corpus-line">{corpus_line}</p>
  <a class="profiles-link" href="presidents/index.html">Browse the 45 president profiles →</a>
  &nbsp; <a class="profiles-link" href="explorer.html">Look up any word or phrase →</a>
  &nbsp; <a class="profiles-link" href="compare.html">Compare two presidents →</a>
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
  (Nixon's Checkers speech, Reagan's "A Time for Choosing") under the later president.
  This is the formal register only - major prepared addresses; rallies, debates, and
  social media are outside the corpus, so every trend here is presidents at their most
  prepared.</p>
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
    word_families.build_families(df)      # before keyword_trends: term_groups reads the map
    trends.term_groups.cache_clear()
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
    band_table = bands.load_bands()

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
                                    band_table),
        "keywords": fig_keywords(kw, df, scores),
    }
    prose_extra = {}
    if band_table is not None:
        figs["llm_topics"] = fig_llm_topic_bands(band_table)
        prose_extra["llm_topics"] = " ".join(filter(None, [
            llm_coverage_sentence(),
            llm_divergence_sentence(band_table),
        ]))
    else:
        print("  data/bands.parquet absent - charts render unbanded and the "
              "LLM-topic section is omitted. Build it with "
              "`python -m presidential_profiles.bands`.")
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
    out.write_text(build_html(figs, stats_line, bodies, inline=False,
                              prose_extra=prose_extra))
    print(f"wrote {out.relative_to(REPO_ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")

    profile_data = profiles.build_profile_data()
    profiles_site.write_profiles(profile_data, SITE_DIR)

    meta = explorer.EXPLORER_DIR / "meta.json"
    # Every input the explorer payload is built FROM must be watched, not just
    # the word families. Its topic series come from issues.ISSUES_META_PATH and
    # its topic LABELS from topic_quality.NAMES_PATH, so without those two a
    # renamed or newly-surfaced topic would serve stale labels forever.
    explorer_inputs = (word_families.FAMILIES_PATH, issues.ISSUES_META_PATH,
                       topic_quality.NAMES_PATH)
    if (not meta.exists()
            or any(p.exists() and p.stat().st_mtime > meta.stat().st_mtime
                   for p in explorer_inputs)):
        explorer.build_explorer_data()      # rebuild if any input changed
    explorer.write_page()
    issues_site.write_issue_pages(SITE_DIR, issue_df, issue_meta, scores, faces)
    compare_site.write_compare(profile_data,
                               topic_quality.display_issues(issue_meta["issues"]))

    if args.inline:
        out2 = SITE_DIR / "index_selfcontained.html"
        out2.write_text(build_html(figs, stats_line, bodies, inline=True,
                                   prose_extra=prose_extra))
        print(f"wrote {out2.relative_to(REPO_ROOT)} ({out2.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
