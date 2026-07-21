"""Issue profile pages: the biography of each issue across 240 years."""

import html as html_mod
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .figures import BASELINE, BLUE_RAMP, GRID, INK, INK2, MUTED, SURFACE
from .profiles import MILLER_URL, _EXTRA_ANCHORS, _pick_sentence, slug
from .site_style import FONT, PAGE_CSS
from . import issues, topic_quality


def issue_slug(name: str) -> str:
    return slug(name.replace("&", "and"))


def fig_issue_timeline(pl: pd.DataFrame, name: str, label: str,
                       pres_dots: pd.DataFrame) -> go.Figure:
    d = pl.assign(period=(pl["year"] // 5) * 5)
    counts = d.groupby("period").size()
    share = (d.groupby("period")[name].mean() * 100)
    share = share[counts >= 40]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pres_dots["x"], y=pres_dots["y"], mode="markers", showlegend=False,
        marker=dict(color=BLUE_RAMP[5], size=5, opacity=0.45),
        customdata=pres_dots["name"],
        hovertemplate="<b>%{customdata}</b>: %{y:.1f}% of their speech"
                      "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=share.index, y=share.values, mode="lines", showlegend=False,
        line=dict(color=BLUE_RAMP[4], width=2.4),
        fill="tozeroy", fillcolor="rgba(109, 167, 236, 0.30)",
        hovertemplate="%{y:.1f}% of paragraphs<extra></extra>",
    ))
    fig.update_layout(
        template="simple_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=56, r=24, t=24, b=44), height=380,
        xaxis=dict(range=[1786, 2029], gridcolor=GRID, linecolor=BASELINE,
                   tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(title=f"% of speech about {label.lower()}",
                   gridcolor=GRID, linecolor=BASELINE, rangemode="tozero",
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
  <p>Share of presidential speech about {label.lower()}; each dot is one
  president's own share, at their term midpoint.</p>
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
    mids = (scores["first_year"] + scores["last_year"]) / 2

    out_dir = site_dir / "issues"
    out_dir.mkdir(parents=True, exist_ok=True)
    display = topic_quality.display_issues(issue_meta["issues"])
    entries = []
    for name in display:
        label = profiles_site.DISCOVERED_LABELS.get(name, name)
        pres_dots = pd.DataFrame({
            "x": mids.values,
            "y": (d.loc[mids.index, f"share_{name}"] * 100).values,
            "name": mids.index,
        })
        owners = pd.DataFrame({
            "share": d.loc[scores.index, f"share_{name}"] * 100,
        }).nlargest(8, "share")
        fig = fig_issue_timeline(pl, name, label, pres_dots)
        quotes = _issue_quotes(merged, name, anchors_all[name], titles)
        page = render_issue(label, owners, fig, quotes)
        (out_dir / f"{issue_slug(label)}.html").write_text(page)

        periods = pl.assign(period=(pl["year"] // 10) * 10)
        peak = int(periods.groupby("period")[name].mean().idxmax())
        entries.append({"slug": issue_slug(label), "label": label,
                        "peak": peak, "top": owners.index[0]})
    (out_dir / "index.html").write_text(render_issue_index(entries))
    print(f"  wrote {len(display)} issue pages + index to docs/issues/")
