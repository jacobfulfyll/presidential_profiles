"""Issue profile pages: the biography of each issue across 240 years."""

import html as html_mod

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .figures import BASELINE, BLUE_RAMP, GRID, INK, MUTED, SURFACE
from .profiles import MILLER_URL, _EXTRA_ANCHORS, _pick_sentence, slug
from .site_style import FONT, PAGE_CSS
from . import bands, issues, topic_quality

# Bands are drawn UNDER the trend line at low opacity: the 16-panel
# small-multiples grid on the dashboard reuses these traces, and a heavier fill
# turns it into mush. The old `fill="tozeroy"` area under the line is gone —
# two overlapping fills read as one shape and the reader cannot tell which edge
# is the estimate and which is the interval.
BAND_FILL = "rgba(109, 167, 236, 0.22)"
CAUTION_LINE = "rgba(109, 167, 236, 0.55)"

# THE definition of the chart x window, for every issue surface. `site.X_RANGE`
# is derived from it rather than restating the pair — the same two numbers in two
# modules is how the lower bound comes to be widened in one place and not the
# other, which is exactly the defect `x_range_covering` exists to fix.
X_MIN, X_MAX = 1786, 2029


def x_range_covering(bands_frames) -> list[int]:
    """`[X_MIN, X_MAX]`, widened DOWNWARD to cover every plotted x.

    The lower bound is a floor to be widened; the upper bound is a fixed crop
    and is deliberately NOT adaptive (nothing in this corpus reaches 2029, and a
    stray future year should be visibly out of range rather than silently
    restyle every panel's axis).

    Widening matters because Surface A plots the recovered 1785 bucket one year
    below `X_MIN`: a fixed lower bound clipped the exact period this whole
    change exists to stop hiding — band, marker and hover all outside the axis.

    One implementation, used by the single-issue figure and by both dashboard
    grids. It lived twice, differently, for one round; `int()` is here because
    the shipped `x` column is float64 (era rows carry midpoints) and a bare
    min() would emit `1785.0` as an axis bound.

    Args:
        bands_frames: Iterable of `bands.parquet` slices or None.
    """
    lows = [int(b["x"].min()) for b in bands_frames
            if b is not None and not b.empty]
    return [min([X_MIN, *lows]), X_MAX]

# Minimum paragraphs before one president's own share inside one period is
# worth plotting as a dot. Below this a dot is a coin flip wearing a portrait.
# Measured on the shipped corpus: of the 100 (president, period) cells, this
# drops 9 and keeps 91. No president disappears entirely — all 45 keep at least
# one dot — so the floor thins the overlay rather than silently removing anyone
# from their own chart.
MIN_DOT_PARAGRAPHS = 25


def issue_slug(name: str) -> str:
    return slug(name.replace("&", "and"))


def president_period_dots(pl: pd.DataFrame, name: str,
                          min_paragraphs: int = MIN_DOT_PARAGRAPHS,
                          period_years: int = bands.PERIOD_YEARS) -> pd.DataFrame:
    """One dot per (president, period) — their own share inside that period.

    Replaces the previous one-dot-per-president-at-term-midpoint overlay, which
    could only ever say "this president, on average, somewhere in here". A
    president spanning several periods now gets a point in each, so the dots and
    the trend line are on the same time axis instead of merely near it.

    `(president, period)` cells thinner than `min_paragraphs` are dropped rather
    than plotted, so a dot never carries less evidence than the band it sits on.
    At the default that drops 9 of 100 cells and no president entirely; the
    chart prose states the floor, because "one dot per president per period"
    would otherwise imply a completeness the overlay does not have.
    """
    d = pl.assign(period=(pl["year"] // period_years) * period_years)
    g = d.groupby(["president", "period"], sort=True)
    out = g.agg(y=(name, "mean"), n=(name, "size")).reset_index()
    out = out[out["n"] >= min_paragraphs]
    return pd.DataFrame({
        "x": out["period"].to_numpy(),
        "y": (out["y"] * 100).to_numpy(),
        "name": out["president"].to_numpy(),
        "n": out["n"].to_numpy(),
    })


def band_y_top(band: pd.DataFrame | None, headroom: float = 1.05) -> float | None:
    """Panel y-ceiling, in percent — scaled by the TRUSTWORTHY content only.

    A 2-speech period's bootstrap interval is legitimately enormous: 1785's
    upper bound on "Religion & values" is 66.7% against a series that never
    exceeds 14.3%. Letting plotly autoscale to that flattens 240 years of real
    trend into a hairline so one un-trustworthy cell can be drawn whole.

    So the axis is scaled to cover every point estimate plus the full interval
    of every `ci_status == "ok"` period, and a cautioned period's band is
    allowed to run off the top of the panel — which reads as exactly what it is
    ("this period's uncertainty exceeds the chart"). Nothing is clipped in the
    DATA: `bands.parquet` and the hover tooltip both carry the true bound.
    Across the 16 CorEx issue panels the site actually renders
    (`topic_quality.display_issues` yields 15 anchored issues plus
    `Discovered 5`), this rule lets exactly ONE overflow: Religion & values,
    whose 1785 upper bound is 66.7% against a series maximum of 14.3%.

    Returns None when there is nothing to scale to, leaving plotly's autorange.
    """
    if band is None or band.empty:
        return None
    ok = band[band["ci_status"] == "ok"]
    tops = [band["point"].max()]
    if not ok.empty and ok["hi"].notna().any():
        tops.append(ok["hi"].max())
    top = float(np.nanmax(tops))
    return top * 100 * headroom if top > 0 else None


def band_traces(band: pd.DataFrame | None) -> list[go.Scatter]:
    """The shaded interval, as one `toself` polygon per CONTIGUOUS run.

    Rows with no published interval (`ci_status == "suppressed_n_floor"`) are
    not merely dropped from the vertex list — a single `toself` polygon closes
    straight across such a gap, which draws a band over periods that have none.
    That is visual interpolation, and it is exactly what `connectgaps=False` on
    the solid line already refuses to do. So the frame is split into maximal
    runs of consecutive periods that DO have an interval, and each run gets its
    own polygon; the gap between them is left empty.

    A run of one period has no area, so it is emitted as a vertical whisker
    instead — otherwise a lone period's interval would silently render as
    nothing at all.

    Precisely: runs are maximal runs of consecutive ROWS, which are the same
    thing as consecutive periods only because `bands.series_band` returns a
    period-complete frame ordered by `period_order` — every period in the
    surface gets a row, intervals absent or not. A series-sparse table would
    make two non-adjacent periods adjacent rows and bridge a real gap. That
    invariant is asserted directly — over every series on both surfaces, on
    consecutiveness rather than on a row count — by
    `TestSeriesBandIsPeriodComplete` in `tests/test_band_charts.py`.
    """
    if band is None or band.empty:
        return []
    b = band.reset_index(drop=True)
    have = (b["lo"].notna() & b["hi"].notna()).to_numpy()
    if not have.any():
        return []
    run_id = np.cumsum(~have)          # constant within a run of True
    traces = []
    for _, run in b[have].groupby(run_id[have], sort=True):
        x = run["x"].tolist()
        lo = (run["lo"] * 100).tolist()
        hi = (run["hi"] * 100).tolist()
        if len(x) == 1:
            traces.append(go.Scatter(
                x=[x[0], x[0]], y=[lo[0], hi[0]], mode="lines",
                line=dict(color=BAND_FILL, width=6),
                hoverinfo="skip", showlegend=False,
            ))
            continue
        traces.append(go.Scatter(
            x=x + x[::-1], y=hi + lo[::-1],
            mode="lines", fill="toself", fillcolor=BAND_FILL,
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))
    return traces


def _band_hover(b: pd.DataFrame, unit: str,
                show_components: bool) -> tuple[np.ndarray, str]:
    """Customdata + template putting the interval and its trust gate in the tip.

    `ci_status` and (on the LLM surface) `ci_components` are the machine-readable
    trust gate on every band row; putting them in the tooltip is what stops a
    reader treating a `low_cluster_caution` interval as an `ok` one.
    """
    cols = [
        (b["lo"] * 100).to_numpy(),
        (b["hi"] * 100).to_numpy(),
        b["n_speeches"].to_numpy(),
        b["ci_status"].to_numpy(),
    ]
    tpl = ("%{y:.1f}" + unit
           + "<br>95% band %{customdata[0]:.1f}–%{customdata[1]:.1f}"
           + "<br>%{customdata[2]} speeches · %{customdata[3]}")
    if show_components:
        cols.append(b["ci_components"].to_numpy())
        tpl += " · %{customdata[4]}"
    return np.stack(cols, axis=-1), tpl


def line_traces(band: pd.DataFrame, unit: str = "% of paragraphs",
                width: float = 2.4, suffix: str = "",
                show_components: bool = False) -> list[go.Scatter]:
    """Trend line, split so thin periods are visibly provisional.

    The charts used to apply a hard `counts >= 40` mask, which deleted thin
    periods outright — presenting absence of data as absence of interest. The
    mask is now a rendering distinction: every period is drawn, but only
    `ci_status == "ok"` periods get the solid line. Thin ones show as a dashed
    underlay carrying an enormous band. Corpus-wide this recovers exactly one
    period (1785: 14 paragraphs, 2 speeches) — it is a correctness fix, not a
    large data recovery.
    """
    custom, tpl = _band_hover(band, unit, show_components)
    y = (band["point"] * 100).to_numpy()
    solid = np.where(band["ci_status"].to_numpy() == "ok", y, np.nan)
    return [
        go.Scatter(
            x=band["x"], y=y, mode="lines", showlegend=False,
            line=dict(color=CAUTION_LINE, width=width * 0.75, dash="dot"),
            customdata=custom, hovertemplate=tpl + "<extra>" + suffix + "</extra>",
        ),
        go.Scatter(
            x=band["x"], y=solid, mode="lines", showlegend=False,
            connectgaps=False, line=dict(color=BLUE_RAMP[4], width=width),
            hoverinfo="skip",
        ),
    ]


def fig_issue_timeline(pl: pd.DataFrame, name: str, label: str,
                       pres_dots: pd.DataFrame,
                       band: pd.DataFrame | None = None) -> go.Figure:
    """One issue's 240 years, with its speech-clustered sampling band.

    `band` is the `bands.parquet` slice for this issue, or None when the table
    has not been built — in which case the line renders unbanded rather than
    with an invented interval. CorEx labels come from a deterministic model with
    no annotator in the loop, so these intervals are `sampling_only` by
    construction and permanently: annotator disagreement is not applicable here,
    not merely absent (see `bands.py`).
    """
    fig = go.Figure()
    for trace in band_traces(band):
        fig.add_trace(trace)
    if band is None:
        d = pl.assign(period=(pl["year"] // bands.PERIOD_YEARS) * bands.PERIOD_YEARS)
        counts = d.groupby("period").size()
        share = (d.groupby("period")[name].mean() * 100)
        share = share[counts >= 40]
        fig.add_trace(go.Scatter(
            x=share.index, y=share.values, mode="lines", showlegend=False,
            line=dict(color=BLUE_RAMP[4], width=2.4),
            hovertemplate="%{y:.1f}% of paragraphs<extra></extra>",
        ))
    else:
        for trace in line_traces(band):
            fig.add_trace(trace)
    fig.add_trace(go.Scatter(
        x=pres_dots["x"], y=pres_dots["y"], mode="markers", showlegend=False,
        marker=dict(color=BLUE_RAMP[5], size=5, opacity=0.45),
        customdata=np.stack([pres_dots["name"], pres_dots["n"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b>: %{y:.1f}% of their paragraphs "
                      "in this period (%{customdata[1]})<extra></extra>",
    ))
    x_lo, x_hi = x_range_covering([band])
    top = band_y_top(band)
    if top is not None and len(pres_dots):
        # Dots are observed shares, never estimates — a president who spent 30%
        # of a period on one issue must not be cropped out of their own chart.
        top = max(top, float(pres_dots["y"].max()) * 1.05)
    fig.update_layout(
        template="simple_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=56, r=24, t=24, b=44), height=380,
        xaxis=dict(range=[x_lo, x_hi], gridcolor=GRID, linecolor=BASELINE,
                   tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(title=f"% of speech about {label.lower()}",
                   gridcolor=GRID, linecolor=BASELINE, rangemode="tozero",
                   range=None if top is None else [0, top],
                   tickfont=dict(color=MUTED, size=11)),
    )
    return fig


def _issue_quotes(merged: pd.DataFrame, name: str,
                  anchors: list[str], titles: pd.DataFrame) -> list[dict]:
    """Three verbatim moments: early, peak, and recent.

    `merged` is paragraphs joined with their issue labels on
    (doc_name, para_idx), so paragraph text and label booleans live in the
    same frame/row - no positional alignment between separate tables."""
    mask = merged[name].to_numpy()
    idx = pd.Series(range(len(merged)))[mask]
    years = merged.loc[mask, "year"]
    if not len(years):
        return []
    d = merged.assign(period=(merged["year"] // 20) * 20)
    peak_period = (d.groupby("period")[name].mean()).idxmax()
    windows = [
        ("Early", int(years.min()), int(years.min()) + 40),
        ("At its peak", int(peak_period), int(peak_period) + 20),
        ("Most recent", int(years.max()) - 15, int(years.max())),
    ]
    quotes, seen = [], set()
    for label, lo, hi in windows:
        win = idx[(years >= lo) & (years <= hi)]
        if not len(win):
            continue
        texts = merged.loc[win, "text"]
        hits = texts.str.lower().str.count(
            "|".join(rf"\b{a}\w*" for a in anchors))
        top = hits.nlargest(3).index
        q = _pick_sentence([merged.loc[i, "text"] for i in top], anchors, [])
        if not q or q in seen:
            continue
        seen.add(q)
        src = next(i for i in top if q in merged.loc[i, "text"])
        doc = merged.loc[src, "doc_name"]
        t = titles.loc[doc]
        quotes.append({
            "label": label, "quote": html_mod.escape(q),
            "cite": f"{t['president']}, {t['title'].split(':', 1)[-1].strip()}, "
                    f"{int(t['year'])}",
            "url": MILLER_URL + doc,
        })
    return quotes


def render_issue(label: str, owners, fig: go.Figure,
                 quotes: list[dict]) -> str:
    fig_json = pio.to_json(fig)
    owner_rows = []
    max_share = owners["share"].max() or 1
    for pres, row in owners.iterrows():
        img = f'<img src="../portraits/{slug(pres)}.png" alt="">'
        width = row["share"] / max_share * 100
        owner_rows.append(f"""<div class="o-row">
  {img}<span class="o-name">{pres}</span>
  <span class="o-bar"><span style="width:{width:.0f}%"></span></span>
  <span class="o-val">{row["share"]:.0f}%</span>
</div>""")
    quotes_html = "\n".join(
        f"""<blockquote><div class="q-label">{q["label"]}</div>
<p>“{q["quote"]}”</p>
<cite><a href="{q["url"]}" target="_blank" rel="noopener">{q["cite"]}</a></cite>
</blockquote>""" for q in quotes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{label} - Presidential Profiles</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
<style>
{PAGE_CSS}
  .crumbs {{ margin-bottom: 18px; font-size: 0.88rem; }}
  .crumbs a {{ color: var(--ink2); }}
  .o-row {{ display: flex; align-items: center; gap: 12px; margin: 8px 0; }}
  .o-row img {{ width: 34px; height: 34px; border-radius: 50%; flex: none; }}
  .o-name {{ width: 190px; flex: none; font-size: 0.92rem; }}
  .o-bar {{ flex: 1; height: 10px; background: var(--page); border-radius: 5px;
            border: 1px solid var(--border); overflow: hidden; }}
  .o-bar span {{ display: block; height: 100%; background: {BLUE_RAMP[4]}; }}
  .o-val {{ width: 46px; text-align: right; color: var(--ink2);
            font-size: 0.88rem; flex: none; }}
  blockquote {{ background: var(--surface); border: 1px solid var(--border);
                border-left: 3px solid var(--muted); border-radius: 10px;
                padding: 14px 18px; margin: 12px 0; }}
  blockquote p {{ color: var(--ink); margin: 4px 0 0; max-width: none; }}
  .q-label {{ color: var(--muted); font-size: 0.76rem; font-weight: 650;
              letter-spacing: 0.07em; text-transform: uppercase; }}
  blockquote cite {{ display: block; font-style: normal; font-size: 0.82rem;
                     margin-top: 8px; }}
  blockquote cite a {{ color: var(--muted); }}
</style>
</head>
<body>
<header>
  <p class="crumbs"><a href="index.html">← All issues</a> &nbsp;·&nbsp;
     <a href="../index.html">Dashboard</a> &nbsp;·&nbsp;
     <a href="../presidents/index.html">Presidents</a></p>
  <h1>{label}</h1>
</header>
<main>
<section>
  <h2>240 years of attention</h2>
  <p>Share of presidential speech about {label.lower()}. The shaded band is a
  95% interval from a bootstrap that resamples whole <em>speeches</em>, so it
  widens where a period rests on a handful of them; a dotted line marks periods
  too thin to trust. Each dot is one president's own share inside one 5-year
  period, for the {MIN_DOT_PARAGRAPHS}-paragraph-and-up cells where they said
  enough to measure. These labels come from a deterministic topic model with no
  AI annotator in the loop, so the band covers sampling error only.</p>
  <div class="chart-scroll"><div class="chart" id="chart" style="height:380px"></div></div>
</section>
<section>
  <h2>Who owned it</h2>
  <p>The presidents who gave it the largest share of their words.</p>
  {"".join(owner_rows)}
</section>
<section>
  <h2>In their words</h2>
  {quotes_html}
</section>
</main>
<footer>
  <p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs,
  University of Virginia</a>.</p>
</footer>
<script>
  const FIG = {fig_json};
  Plotly.newPlot("chart", FIG.data, FIG.layout, {{displayModeBar: false, responsive: true}});
</script>
</body>
</html>
"""


def render_issue_index(entries: list[dict]) -> str:
    cards = "\n".join(f"""<a class="card" href="{e["slug"]}.html">
  <div class="name">{e["label"]}</div>
  <div class="meta">peak: {e["peak"]}s · top voice: {e["top"]}</div>
</a>""" for e in entries)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Issues - Presidential Profiles</title>
<style>
{PAGE_CSS}
  .crumbs {{ margin-bottom: 18px; font-size: 0.88rem; }}
  .crumbs a {{ color: var(--ink2); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
           gap: 12px; margin-top: 24px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 10px; padding: 14px 16px; text-decoration: none;
           color: var(--ink); }}
  .card:hover {{ border-color: var(--muted); }}
  .name {{ font-weight: 650; }}
  .meta {{ color: var(--muted); font-size: 0.8rem; margin-top: 4px; }}
</style>
</head>
<body>
<header>
  <p class="crumbs"><a href="../index.html">← Dashboard</a></p>
  <h1>Issue profiles</h1>
  <p class="sub">The biography of every issue: 240 years of attention, the presidents
  who owned it, and their words at its defining moments.</p>
  <div class="grid">
{cards}
  </div>
</header>
</body>
</html>
"""


def write_issue_pages(site_dir, issue_df: pd.DataFrame, issue_meta: dict,
                      scores: pd.DataFrame, faces: dict) -> None:
    from .fetch import PARAGRAPHS_PATH
    from . import profiles_site

    paras = pd.read_parquet(PARAGRAPHS_PATH).reset_index(drop=True)
    pl = pd.read_parquet(issues.PARA_LABELS_PATH).reset_index(drop=True)
    # Explicit key join for anything that needs both paragraph text and
    # labels in the same row - no positional alignment between the tables.
    merged = paras.merge(
        pl, on=["doc_name", "para_idx"], how="inner", validate="one_to_one"
    )
    if len(merged) != len(paras) or len(merged) != len(pl):
        raise RuntimeError("paragraphs and labels key sets diverge - rerun the pipeline")
    merged = merged.reset_index(drop=True)
    speeches_df = pd.read_parquet(issues.DATA_DIR / "speeches.parquet")
    titles = speeches_df.set_index("doc_name")[["president", "title", "year"]]
    anchors_all = {**issues.ISSUE_ANCHORS, **_EXTRA_ANCHORS}

    d = issue_df.set_index("president")

    out_dir = site_dir / "issues"
    out_dir.mkdir(parents=True, exist_ok=True)
    display = topic_quality.display_issues(issue_meta["issues"])
    band_table = bands.load_bands()
    entries = []
    for name in display:
        label = profiles_site.DISCOVERED_LABELS.get(name, name)
        pres_dots = president_period_dots(pl, name)
        owners = pd.DataFrame({
            "share": d.loc[scores.index, f"share_{name}"] * 100,
        }).nlargest(8, "share")
        band = bands.series_band(band_table, bands.COREX_SURFACE, name)
        fig = fig_issue_timeline(pl, name, label, pres_dots, band)
        quotes = _issue_quotes(merged, name, anchors_all[name], titles)
        page = render_issue(label, owners, fig, quotes)
        (out_dir / f"{issue_slug(label)}.html").write_text(page)

        periods = pl.assign(period=(pl["year"] // 10) * 10)
        peak = int(periods.groupby("period")[name].mean().idxmax())
        entries.append({"slug": issue_slug(label), "label": label,
                        "peak": peak, "top": owners.index[0]})
    (out_dir / "index.html").write_text(render_issue_index(entries))
    print(f"  wrote {len(display)} issue pages + index to docs/issues/")
