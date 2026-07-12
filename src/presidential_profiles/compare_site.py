"""The president comparison page: two fingerprints overlaid, profiles side
by side. A single static page with all 45 payloads inlined (~small)."""

import json

from .figures import GRID, INK, INK2, MUTED, SERIES, SURFACE, REPO_ROOT
from .profiles import RADAR_AXES, slug
from . import profiles_site
from .site_style import FONT, PAGE_CSS


def build_payload(data: dict, display_issues: list[str]) -> dict:
    scores = data["scores"]
    issues_df = data["issues"]
    dist = data["distinctive"]
    out = {}
    for pres in scores.index:
        s = scores.loc[pres]
        rel = {profiles_site.DISCOVERED_LABELS.get(n, n):
               round(float(issues_df.loc[pres, f"rel_{n}"]), 1)
               for n in display_issues}
        top_issues = sorted(rel.items(), key=lambda x: -x[1])[:4]
        words = dist[dist["president"] == pres].sort_values("rank")["term"].tolist()[:8]
        out[pres] = {
            "slug": slug(pres),
            "party": s["party"],
            "years": f"{int(s['first_year'])}-{int(s['last_year'])}",
            "speeches": int(s["n_speeches"]),
            "radar": [round(float(s[f"pct_{k}"]), 0) for k, _ in RADAR_AXES],
            "issues": [[n, v] for n, v in top_issues if v > 0],
            "words": words,
            "stats": {
                "certainty": round(float(s["certainty"]), 2),
                "hype / 10k": round(float(s["hype"]), 1),
                "machinery / 10k": round(float(s["mechanism"]), 0),
                "hope / 10k": round(float(s["nrc_hope"]), 0),
                "fear / 10k": round(float(s["nrc_fear"]), 0),
                "grade level": round(float(s["fk_grade"]), 1),
            },
            "kin": [n for n, _ in data["voice_neighbors"].get(pres, [])[:3]],
        }
    return out


def write_compare(data: dict, display_issues: list[str]) -> None:
    payload = build_payload(data, display_issues)
    axes = [label for _, label in RADAR_AXES]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compare presidents - Presidential Profiles</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
<style>
{PAGE_CSS}
  .crumbs {{ margin-bottom: 18px; font-size: 0.88rem; }}
  .crumbs a {{ color: var(--ink2); }}
  .pickers {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 20px; }}
  .picker {{ display: flex; align-items: center; gap: 10px; }}
  .picker img {{ width: 46px; height: 46px; border-radius: 50%;
                 border: 2px solid var(--border); }}
  .picker.a img {{ border-color: {SERIES[0]}; }}
  .picker.b img {{ border-color: {SERIES[1]}; }}
  .picker.c img {{ border-color: {SERIES[2]}; }}
  .picker.c img.empty {{ visibility: hidden; }}
  select {{ padding: 9px 12px; border: 1px solid var(--border); border-radius: 10px;
            background: var(--surface); color: var(--ink); font-family: inherit;
            font-size: 0.95rem; }}
  .duo {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
          gap: 16px; margin-top: 10px; }}
  .col {{ background: var(--surface); border: 1px solid var(--border);
          border-radius: 12px; padding: 16px 18px; }}
  .col h3 {{ font-size: 1.02rem; display: flex; align-items: center; gap: 8px; }}
  .col h3 .sw {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .col .meta {{ color: var(--muted); font-size: 0.82rem; margin: 2px 0 12px; }}
  .col h4 {{ font-size: 0.78rem; color: var(--muted); letter-spacing: 0.07em;
             text-transform: uppercase; margin: 14px 0 6px; }}
  .terms {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .term {{ background: var(--page); border: 1px solid var(--border);
           border-radius: 8px; padding: 3px 9px; font-size: 0.85rem; }}
  .issue-line {{ font-size: 0.9rem; margin: 3px 0; color: var(--ink2); }}
  table {{ width: 100%; margin-top: 20px; border-collapse: collapse;
           background: var(--surface); border: 1px solid var(--border);
           border-radius: 12px; overflow: hidden; }}
  th, td {{ padding: 9px 14px; text-align: left; font-size: 0.92rem;
            border-top: 1px solid var(--grid); }}
  thead th {{ border-top: none; color: var(--muted); font-size: 0.8rem;
              text-transform: uppercase; letter-spacing: 0.06em; }}
  td.num {{ font-variant-numeric: tabular-nums; }}
  td.win {{ font-weight: 700; }}
  a.plink {{ font-size: 0.85rem; }}
</style>
</head>
<body>
<header>
  <p class="crumbs"><a href="index.html">← Dashboard</a> &nbsp;·&nbsp;
     <a href="presidents/index.html">All presidents</a></p>
  <h1>Compare presidents</h1>
  <p class="sub">Two rhetorical fingerprints, overlaid - and everything underneath
  them, side by side.</p>
  <div class="pickers">
    <div class="picker a"><img id="imga" src="" alt=""><select id="sela"></select></div>
    <div class="picker b"><img id="imgb" src="" alt=""><select id="selb"></select></div>
    <div class="picker c"><img id="imgc" src="" alt=""><select id="selc"></select></div>
  </div>
</header>
<main>
  <div class="chart-scroll"><div class="chart" id="radar" style="height:430px"></div></div>
  <div class="duo" id="cols"></div>
  <table id="stats"></table>
</main>
<footer>
  <p>Fingerprint axes are percentiles among all 45 presidents. Data:
  <a href="https://data.millercenter.org">Miller Center of Public Affairs,
  University of Virginia</a>.</p>
</footer>
<script>
const P = {json.dumps(payload)};
const AXES = {json.dumps(axes)};
const COLORS = ["{SERIES[0]}", "{SERIES[1]}", "{SERIES[2]}"];
const CHROME = {{surface:"{SURFACE}", ink:"{INK}", ink2:"{INK2}",
                 muted:"{MUTED}", grid:"{GRID}"}};
const names = Object.keys(P);

function fillSelect(sel, chosen, allowNone) {{
  if (allowNone) {{
    const o = document.createElement("option");
    o.value = ""; o.textContent = "add a third…";
    sel.appendChild(o);
  }}
  names.forEach(n => {{
    const o = document.createElement("option");
    o.value = n; o.textContent = n; if (n === chosen) o.selected = true;
    sel.appendChild(o);
  }});
}}
function radarTrace(name, i) {{
  const r = P[name].radar.concat([P[name].radar[0]]);
  const theta = AXES.concat([AXES[0]]);
  return {{type: "scatterpolar", r, theta, name, fill: "toself",
          line: {{color: COLORS[i], width: 2}},
          fillcolor: COLORS[i] + "30",
          hovertemplate: "%{{theta}}: %{{r:.0f}}th pct<extra>" + name + "</extra>"}};
}}
function colHTML(name, i) {{
  const p = P[name];
  const issues = p.issues.map(x =>
    `<div class="issue-line">${{x[0]}} <span style="color:${{CHROME.muted}}">+${{x[1]}} pp vs era</span></div>`).join("")
    || `<div class="issue-line">no issue above their era</div>`;
  const words = p.words.map(w => `<span class="term">${{w}}</span>`).join("");
  const kin = p.kin.join(" · ");
  return `<div class="col">
    <h3><span class="sw" style="background:${{COLORS[i]}}"></span>${{name}}</h3>
    <div class="meta">${{p.party}} · ${{p.years}} · ${{p.speeches}} speeches ·
      <a class="plink" href="presidents/${{p.slug}}.html">full profile →</a></div>
    <h4>Pressed harder than their era</h4>${{issues}}
    <h4>In their own words</h4><div class="terms">${{words}}</div>
    <h4>Sounds like</h4><div class="issue-line">${{kin}}</div>
  </div>`;
}}
function statsHTML(sel) {{
  const rows = Object.keys(P[sel[0]].stats).map(k => {{
    const vals = sel.map(n => P[n].stats[k]);
    const mx = Math.max(...vals);
    const tds = vals.map(v =>
      `<td class="num ${{v === mx ? "win" : ""}}">${{v}}</td>`).join("");
    return `<tr><td>${{k}}</td>${{tds}}</tr>`;
  }}).join("");
  const heads = sel.map(n => `<th>${{n}}</th>`).join("");
  return `<thead><tr><th>measure</th>${{heads}}</tr></thead><tbody>${{rows}}</tbody>`;
}}
function render() {{
  const a = document.getElementById("sela").value;
  const b = document.getElementById("selb").value;
  const c = document.getElementById("selc").value;
  const sel = c ? [a, b, c] : [a, b];
  document.getElementById("imga").src = "portraits/" + P[a].slug + ".png";
  document.getElementById("imgb").src = "portraits/" + P[b].slug + ".png";
  const imgc = document.getElementById("imgc");
  imgc.className = c ? "" : "empty";
  if (c) imgc.src = "portraits/" + P[c].slug + ".png";
  Plotly.react("radar", sel.map((n, i) => radarTrace(n, i)), {{
    template: "simple_white", paper_bgcolor: CHROME.surface,
    font: {{family: "{FONT}", color: CHROME.ink2, size: 12}},
    polar: {{bgcolor: CHROME.surface,
            radialaxis: {{range: [0, 100], tickfont: {{size: 9, color: CHROME.muted}},
                         gridcolor: CHROME.grid, angle: 90, tickangle: 90}},
            angularaxis: {{tickfont: {{size: 12, color: CHROME.ink}},
                          gridcolor: CHROME.grid}}}},
    legend: {{orientation: "h", x: 0, y: 1.12}},
    height: 430, margin: {{l: 70, r: 70, t: 40, b: 30}},
  }}, {{displayModeBar: false, responsive: true}});
  document.getElementById("cols").innerHTML =
    sel.map((n, i) => colHTML(n, i)).join("");
  document.getElementById("stats").innerHTML = statsHTML(sel);
  const url = new URL(location);
  url.searchParams.set("a", P[a].slug); url.searchParams.set("b", P[b].slug);
  if (c) url.searchParams.set("c", P[c].slug); else url.searchParams.delete("c");
  history.replaceState(null, "", url);
}}
const params = new URLSearchParams(location.search);
const bySlug = Object.fromEntries(names.map(n => [P[n].slug, n]));
const a0 = bySlug[params.get("a")] || "Abraham Lincoln";
const b0 = bySlug[params.get("b")] || "Franklin D. Roosevelt";
const c0 = bySlug[params.get("c")] || "";
fillSelect(document.getElementById("sela"), a0);
fillSelect(document.getElementById("selb"), b0);
fillSelect(document.getElementById("selc"), c0, true);
document.getElementById("sela").onchange = render;
document.getElementById("selb").onchange = render;
document.getElementById("selc").onchange = render;
render();
</script>
</body>
</html>
"""
    (REPO_ROOT / "docs" / "compare.html").write_text(html)
    print("  wrote docs/compare.html")
