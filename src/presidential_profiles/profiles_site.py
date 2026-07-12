"""Render the 45 president profile pages and their index."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .figures import BLUE_RAMP, GRID, INK, INK2, MUTED, PARTY_COLORS, SURFACE
from .profiles import RADAR_AXES, slug
from .site_style import FONT, PAGE_CSS

# Issues shown on profiles: the curated taxonomy plus the one discovered
# topic that is a genuine issue (soviet/nuclear/weapons).
DISPLAY_ISSUES = None  # filled from meta at build time
DISCOVERED_LABELS = {"Discovered 5": "Security & peace"}


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
    rel = {name: float(issue_row[f"rel_{name}"]) for name in display_issues}
    s = pd.Series(rel).sort_values()
    top = pd.concat([s.head(2), s.tail(6)])
    labels = [DISCOVERED_LABELS.get(n, n) for n in top.index]
    colors = [BLUE_RAMP[4] if v >= 0 else MUTED for v in top.values]
    fig = go.Figure(go.Bar(
        y=labels, x=top.values, orientation="h",
        marker=dict(color=colors),
        hovertemplate="%{y}: %{x:+.1f} pp vs their era<extra></extra>",
    ))
    fig.update_layout(
        template="simple_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=12),
        xaxis=dict(title=dict(text="attention vs contemporaries (percentage points)",
                              font=dict(size=10.5, color=INK2)),
                   gridcolor=GRID, tickfont=dict(color=MUTED, size=10),
                   zeroline=True, zerolinecolor=INK2, zerolinewidth=1),
        yaxis=dict(tickfont=dict(color=INK, size=12)),
        height=380, margin=dict(l=10, r=16, t=40, b=48),
    )
    return fig


def _neighbor_list(pairs: list, fmt: str) -> str:
    return " · ".join(
        f'<a href="{slug(name)}.html">{name}</a> <span class="dim">({fmt.format(v)})</span>'
        for name, v in pairs
    )


def _issue_cards_html(president: str, data: dict) -> str:
    info = data["issue_cards"].get(president, {"cards": [], "voice": []})
    cards = []
    for c in info["cards"]:
        issue = DISCOVERED_LABELS.get(c["issue"], c["issue"])
        words = "".join(f'<span class="term">{w}</span>' for w in c["words"])
        words_html = f'<div class="terms">{words}</div>' if c["words"] else ""
        quote_html = ""
        if c["quote"]:
            quote_html = (f'<blockquote><p>“{c["quote"]}”</p>'
                          f'<cite>{c["cite"]}</cite></blockquote>')
        stance = (f'<span class="i-stance">{c["stance"]}</span>'
                  if c.get("stance") else "")
        cards.append(f"""<div class="i-card">
  <div class="i-head"><span class="i-name">{issue}</span>
    <span class="i-badge">+{c["rel"]:.0f} pp vs their era</span>{stance}</div>
  {words_html}
  {quote_html}
</div>""")
    cards_html = "\n".join(cards) if cards else \
        '<p class="dim">No issue stands out above their era.</p>'

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

    voice = _neighbor_list(data["voice_neighbors"][president], "{:.2f}")
    agenda = _neighbor_list(data["agenda_neighbors"][president], "{:.2f}")

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
  .i-card .term, .i-voice .term {{ background: var(--page); }}
  .i-card blockquote {{ border-left: 3px solid var(--grid); margin: 12px 0 0;
                        padding: 2px 0 2px 14px; }}
  .i-card blockquote p {{ color: var(--ink2); font-size: 0.94rem; margin: 0;
                          max-width: none; }}
  .i-card cite {{ display: block; color: var(--muted); font-style: normal;
                  font-size: 0.8rem; margin-top: 6px; }}
  .dim {{ color: var(--muted); }}
  .neighbors p {{ margin: 6px 0; }}
  ul.sigs {{ margin: 10px 0 0 18px; color: var(--ink2); }}
  ul.sigs li {{ margin: 5px 0; }}
  .invocations {{ color: var(--muted); font-size: 0.9rem; }}
</style>
</head>
<body>
<header>
  <p class="crumbs"><a href="index.html">← All presidents</a> &nbsp;·&nbsp;
     <a href="../index.html">Dashboard</a> &nbsp;·&nbsp;
     <a href="../compare.html?a={slug(president)}">Compare →</a></p>
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
    </div>
    <div>
      <h2>Issues, relative to their era</h2>
      <p>What they talked about more — and less — than contemporaries.</p>
      <div class="chart-scroll"><div class="chart" data-fig="issues" style="height:380px"></div></div>
    </div>
  </div>
</section>
<section>
  <h2>What they cared about — in their own words</h2>
  <p>Their strongest issues relative to contemporaries, each with the vocabulary that is
     statistically <em>theirs</em> on that issue and a verbatim sentence from their speeches.</p>
  {_issue_cards_html(president, data)}
</section>
<section class="neighbors">
  <h2>Kinships</h2>
  <p>Sounds like <span class="dim">(era-adjusted voice)</span>: {voice}</p>
  <p>Shared agenda <span class="dim">(issue mix)</span>: {agenda}</p>
  {invocation_html}
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
  <a href="https://github.com/jacobfulfyll/presidential_profiles">Code &amp; method</a>.</p>
</footer>
<script>
  const FIGS = {json.dumps(fig_json)};
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
    cards = []
    for president in scores.sort_values("first_year").index:
        s = scores.loc[president]
        n_paras = float(issues_df.loc[president, "n_paragraphs"])
        # Badge only issues backed by >= 4 paragraphs, so one-speech
        # presidents don't get a headline from a stray metaphor.
        rel = {
            n: float(issues_df.loc[president, f"rel_{n}"])
            for n in display_issues
            if float(issues_df.loc[president, f"share_{n}"]) * n_paras >= 4
        }
        if not rel:
            rel = {n: float(issues_df.loc[president, f"rel_{n}"]) for n in display_issues}
        top_issue = max(rel, key=rel.get)
        top_issue = DISCOVERED_LABELS.get(top_issue, top_issue)
        color = PARTY_COLORS.get(s["party"], MUTED)
        cards.append(f"""<a class="card" href="{slug(president)}.html">
  <div class="name"><span class="dot" style="background:{color}"></span>{president}</div>
  <div class="meta">{int(s["first_year"])}–{int(s["last_year"])} · {int(s["n_speeches"])} speeches</div>
  <div class="issue">↑ {top_issue}</div>
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
</style>
</head>
<body>
<header>
  <p class="crumbs"><a href="../index.html">← Dashboard</a></p>
  <h1>President Profiles</h1>
  <p class="sub">Every president's rhetorical fingerprint, the issues they pressed
  harder than their contemporaries, their distinctive vocabulary, and the speeches
  that define them. Card badge: the issue they emphasized most vs their era.</p>
  <div class="grid">
{cards_html}
  </div>
</header>
<footer>
  <p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs,
  University of Virginia</a>.</p>
</footer>
</body>
</html>
"""


def write_profiles(data: dict, site_dir) -> None:
    display_issues = data["issue_meta"]["issues"] + ["Discovered 5"]
    pres_dir = site_dir / "presidents"
    pres_dir.mkdir(parents=True, exist_ok=True)
    (pres_dir / "index.html").write_text(render_index(data, display_issues))
    for president in data["scores"].index:
        html = render_profile(president, data, display_issues)
        (pres_dir / f"{slug(president)}.html").write_text(html)
    print(f"  wrote {len(data['scores'])} profile pages + index to docs/presidents/")
