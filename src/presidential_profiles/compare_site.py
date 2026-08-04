"""The president comparison page: two fingerprints overlaid, profiles side
by side. A single static page with all 45 payloads inlined (~small)."""

import json

from .figures import GRID, INK, INK2, MUTED, SERIES, SURFACE, REPO_ROOT
from .html_safety import json_for_script
from .profiles import FEATURE_SIMILARITY, RADAR_AXES, slug
from . import metrics, profiles_site
from .site_style import FONT, PAGE_CSS


def build_payload(data: dict, display_issues: list[str]) -> dict:
    scores = data["scores"]
    out = {}
    for pres in scores.index:
        shared = profiles_site.public_profile_payload(pres, data, display_issues)
        s = scores.loc[pres]
        rel = {
            item["label"]: round(float(item["era_relative_difference"]), 1)
            for item in shared["legacy_issue_attention"]
        }
        top_issues = sorted(rel.items(), key=lambda x: -x[1])[:4]
        words = [item["term"] for item in shared["distinctive_vocabulary"][:8]]
        ai = shared["ai"]
        stats = {
            "certainty": round(float(s["certainty"]), 2),
            "hype / 10k": round(float(s["hype"]), 1),
            "legal & procedural / 10k": round(float(s["mechanism"]), 0),
            "hope / 10k": round(float(s["nrc_hope"]), 0),
            "fear / 10k": round(float(s["nrc_fear"]), 0),
            "grade level": round(float(s["fk_grade"]), 1),
        }
        if ai:
            stats.update({
                "AI · partisan attack %": ai["flags"]["party_attack"],
                "AI · named adversary %": ai["flags"]["enemy_naming"],
                "AI · zero-sum %": ai["flags"]["zero_sum"],
                "AI · concrete proposal %": ai["proposal_values"]["proposal"],
                "AI · values appeal %": ai["proposal_values"]["values"],
                "AI · effective topic breadth": ai["ai_radar"]["topic_breadth"]["absolute"],
            })
        out[pres] = {
            **shared,
            "years_label": f"{int(s['first_year'])}-{int(s['last_year'])}",
            "speeches": shared["sample"]["n_speeches"],
            "radar": [round(float(s[f"pct_{k}"]), 0) for k, _ in RADAR_AXES],
            "issues": [[n, v] for n, v in top_issues if v > 0],
            "words": words,
            "stats": stats,
            "kin": [n for n, _ in data["voice_neighbors"].get(pres, [])[:3]],
            "agenda_kin": [n for n, _ in data["agenda_neighbors"].get(pres, [])[:3]],
            "invocations": shared["classified_invocations"],
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
  .compare-section,.column-section {{ background:var(--surface);border:1px solid var(--border);
                                      border-radius:12px;margin:12px 0;overflow:hidden; }}
  .compare-section > summary,.column-section > summary {{ cursor:pointer;list-style:none;
      padding:15px 18px;font-weight:700;color:var(--ink);display:flex;align-items:center;
      justify-content:space-between;gap:12px; }}
  .compare-section > summary::-webkit-details-marker,
  .column-section > summary::-webkit-details-marker {{ display:none; }}
  .compare-section > summary::after,.column-section > summary::after {{
      content:"＋";color:var(--muted);font-size:1.05rem; }}
  .compare-section[open] > summary::after,.column-section[open] > summary::after {{ content:"−"; }}
  .compare-section[open] > summary,.column-section[open] > summary {{
      border-bottom:1px solid var(--grid); }}
  .compare-section-body,.column-section-body {{ padding:14px 18px 18px; }}
  .column-section-body h5 {{ color:var(--muted);font-size:.72rem;letter-spacing:.07em;
                             text-transform:uppercase;margin:15px 0 6px; }}
  .column-section-body h5:first-child {{ margin-top:0; }}
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
  .method-note {{ background:#eef6ff; border:1px solid #c9def3; border-radius:12px;
                  padding:13px 16px; margin-top:20px; color:var(--ink2); font-size:.9rem; }}
  .method-note a {{ color:#275d8c; font-weight:650; }}
  .ai-topic {{ font-size:.86rem; margin:5px 0; color:var(--ink2); }}
  .ai-topic strong {{ color:var(--ink); }} .ai-topic small {{ color:var(--muted); }}
  .pvbar {{ display:flex; height:9px; border-radius:6px; overflow:hidden; margin:8px 0 5px; background:var(--page); }}
  .pvbar i {{ display:block; height:100%; }} .pv-p {{ background:#275d8c; }}
  .pv-m {{ background:#6da7ec; }} .pv-v {{ background:#8bbf9f; }} .pv-n {{ background:#d3d5d8; }}
  .pv-caption {{ color:var(--muted); font-size:.72rem; }}
  .warning {{ background:#fff7df; border:1px solid #ead49a; border-radius:8px;
              padding:8px 10px; color:#6d5318; font-size:.82rem; margin:8px 0; }}
  .evidence-quote {{ border-left:3px solid var(--grid); padding-left:10px;
                     color:var(--ink2); font-size:.84rem; margin:7px 0; }}
  .download-button {{ margin:10px 0 20px; padding:8px 12px; border:1px solid var(--border);
                      border-radius:8px; background:var(--surface); color:var(--ink);
                      cursor:pointer; }}
</style>
</head>
<body>
<header>
  <p class="crumbs"><a href="index.html">← Dashboard</a> &nbsp;·&nbsp;
     <a href="presidents/index.html">All presidents</a> &nbsp;·&nbsp;
     <a href="methodology.html">How the AI labels work</a></p>
  <h1>Compare presidents</h1>
  <p class="sub">Two rhetorical fingerprints, overlaid - and everything underneath
  them, side by side.</p>
  <div class="pickers">
    <div class="picker a"><img id="imga" src="" alt=""><select id="sela"></select></div>
    <div class="picker b"><img id="imgb" src="" alt=""><select id="selb"></select></div>
    <div class="picker c"><img id="imgc" src="" alt=""><select id="selc"></select></div>
  </div>
  <p class="method-note"><strong>Now includes the corpus-derived AI label layer.</strong>
  Compare semantic topics, partisan attacks, named adversaries, zero-sum framing,
  policy proposals, values appeals, genres, and adversarial entities alongside the
  original lexical fingerprint. The judgment pass hid president, party, and title.
  <a href="methodology.html">Read the illustrated method and reliability audit →</a></p>
</header>
<main>
  <details class="compare-section"><summary>Original rhetorical fingerprint</summary>
  <div class="compare-section-body"><p>Eight named lexical measures, shown as president
  percentiles.</p><div class="chart-scroll"><div class="chart" id="radar" style="height:430px"></div></div>
  {metrics.lesson_html("president_percentile")}<details class="evidence"><summary>Inspect the evidence</summary><p>Complete absolute values and percentiles are in each selected
  profile download below.</p></details></div></details>
  <details class="compare-section"><summary>AI-labeled paragraph signals</summary>
  <div class="compare-section-body"><p>Percent of each president's paragraphs carrying
  partisan attack, named-adversary, or zero-sum labels.</p>
  <div class="chart-scroll"><div class="chart" id="ai-rates" style="height:360px"></div></div>
  {metrics.lesson_html("paragraph_share")}<p>These are descriptive rates; thin records
  remain visible and should be read cautiously.</p><details class="evidence"><summary>Inspect the evidence</summary><p>Profile downloads include paragraph counts, rates, and
  annotation status.</p></details></div></details>
  <details class="compare-section"><summary>Six-part AI radar</summary>
  <div class="compare-section-body"><p>Percentiles for partisan attack, enemy naming,
  zero-sum framing, proposals, values, and effective topic breadth.</p>
  <div class="chart-scroll"><div class="chart" id="ai-radar" style="height:460px"></div></div>
  {metrics.lesson_html("president_percentile")}<p>Thin records remain in tables but do
  not receive a precise percentile rank.</p><details class="evidence"><summary>Inspect the evidence</summary><p>Profile downloads include every absolute AI-radar value and
  eligible-president percentile.</p></details></div></details>
  <button class="download-button" id="download-comparison">Download selected comparison data</button>
  <div class="duo" id="cols"></div>
  <details class="compare-section"><summary>Complete raw-stat table</summary>
  <div class="compare-section-body"><table id="stats"></table></div></details>
</main>
<footer>
  <p>Fingerprint axes are percentiles among all 45 presidents. Data:
  <a href="https://data.millercenter.org">Miller Center of Public Affairs,
  University of Virginia</a>. <a href="methodology.html">AI label method and audit</a>.</p>
</footer>
<script>
const P = {json_for_script(payload)};
const AXES = {json.dumps(axes)};
const FEATURE_LABELS = {json.dumps({key: value["label"] for key, value in FEATURE_SIMILARITY.items()})};
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
  const issues = p.issues.map((x, issueIndex) =>
    `<div class="issue-line"><span class="legacy-issue-label" data-slot="${{i}}" data-issue-index="${{issueIndex}}"></span> <span style="color:${{CHROME.muted}}">+${{x[1]}} pp vs era</span></div>`).join("")
    || `<div class="issue-line">no issue above their era</div>`;
  const words = p.words.map(w => `<span class="term">${{w}}</span>`).join("");
  const ai = p.ai;
  const thin = p.sample.thin_record
    ? `<div class="warning">${{p.sample.warning}}</div>` : "";
  let aiHTML = "";
  if (ai) {{
    const topics = ai.top_topics.slice(0, 4).map(t =>
      `<div class="ai-topic"><strong>${{t.name}}</strong> ${{t.share.toFixed(1)}}% ` +
      `<small>(${{t.rel >= 0 ? "+" : ""}}${{t.rel.toFixed(1)}} pp vs contemporaries)</small></div>`
    ).join("");
    const pv = ai.proposal_values;
    const adv = ai.adversaries.length
      ? ai.adversaries.map(x => `${{x.name}} (${{x.n}}×)`).join(" · ") : "none repeated";
    const inv = p.invocations.slice(0,4).map(x =>
      `${{x.target}} — ${{x.function}} (${{x.mentions}})`).join("<br>") || "none classified";
    const domains = ai.domains.slice(0,4).map(x =>
      `<div class="ai-topic"><strong>${{x.name}}</strong> ${{x.share.toFixed(1)}}%
       <small>(${{x.rel >= 0 ? "+" : ""}}${{x.rel.toFixed(1)}} pp vs era)</small></div>`).join("");
    const genres = ai.speech_types.map(x =>
      `<div class="issue-line">${{x.name}} <span style="color:${{CHROME.muted}}">${{x.value.toFixed(1)}}%</span></div>`).join("");
    aiHTML = `<details class="column-section">
      <summary>AI-labeled evidence</summary><div class="column-section-body">
      <h5>Fine topics</h5>${{topics}}
      <h5>Broad topic domains</h5>${{domains}}
      <h5>Proposal ↔ values</h5><div class="pvbar">
      <i class="pv-p" style="width:${{pv.proposal}}%"></i><i class="pv-m" style="width:${{pv.mixed}}%"></i>
      <i class="pv-v" style="width:${{pv.values}}%"></i><i class="pv-n" style="width:${{pv.neither}}%"></i></div>
      <div class="pv-caption">proposal ${{pv.proposal.toFixed(1)}}% · mixed ${{pv.mixed.toFixed(1)}}% · values ${{pv.values.toFixed(1)}}% · neither ${{pv.neither.toFixed(1)}}%</div>
      <h5>Speech-type composition</h5>${{genres}}
      <h5>Adversarial entities</h5><div class="issue-line">${{adv}}</div>
      <div class="pv-caption">Exploratory AI stance labels; not human-validated.</div>
      <h5>Presidential invocations</h5><div class="issue-line">${{inv}}</div>
      <div class="issue-line"><a href="data/networks/invocation_evidence.csv" download>Download invocation evidence →</a></div>
      </div></details>`;
  }}
  const featureRows = Object.entries(p.feature_neighbors || {{}}).map(([key, rows]) => {{
    const linked = rows.slice(0,3).map(item =>
      `${{item.president}} <span style="color:${{CHROME.muted}}">(${{item.similarity.toFixed(2)}})</span>`
    ).join(" · ") || "not available";
    return `<div class="issue-line"><strong>${{FEATURE_LABELS[key] || key}}</strong><br>${{linked}}</div>`;
  }}).join("");
  const signatures = p.signature_speeches.slice(0,3).map(x =>
    `<div class="issue-line"><a href="${{x.url}}" target="_blank" rel="noopener">${{x.title}}</a></div>`).join("");
  const evidence = (p.issue_evidence.cards || []).filter(x=>x.quote).slice(0,2).map(x =>
    `<div class="evidence-quote">“${{x.quote}}”<br><small>${{x.cite}}</small></div>`).join("");
  return `<div class="col">
    <h3><span class="sw" style="background:${{COLORS[i]}}"></span>${{name}}</h3>
    <div class="meta">${{p.party}} · ${{p.years_label}} · ${{p.speeches}} speeches ·
      <a class="plink" href="presidents/${{p.slug}}.html">full profile →</a></div>
    ${{thin}}
    <details class="column-section"><summary>Historical issue lens</summary>
      <div class="column-section-body">
      <h5>Pressed harder than their era</h5>${{issues}}
      <h5>Distinctive vocabulary</h5><div class="terms">${{words}}</div>
      </div></details>
    <details class="column-section"><summary>Similarity by feature set</summary>
      <div class="column-section-body">${{featureRows}}
      <p class="pv-caption">Each line uses only the named inputs; rhetoric measures are
      standardized before cosine similarity.</p></div></details>
    <details class="column-section"><summary>Speeches and source excerpts</summary>
      <div class="column-section-body">
      <h5>Signature speeches</h5>${{signatures}}
      <h5>Evidence excerpts</h5>${{evidence || '<div class="issue-line">No excerpt selected.</div>'}}
      </div></details>
    ${{aiHTML}}
    <p><a href="data/presidents/${{p.slug}}.json" download>Download complete profile data →</a></p>
  </div>`;
}}
function aiTrace(name, i) {{
  const a = P[name].ai;
  const keys = ["party_attack", "enemy_naming", "zero_sum"];
  const labels = ["Partisan attack", "Named adversary", "Zero-sum framing"];
  return {{type:"bar", name, x:labels, y:keys.map(k => a ? a.flags[k] : 0),
          marker:{{color:COLORS[i]}},
          hovertemplate:"%{{x}}: %{{y:.1f}}% of paragraphs<extra>" + name + "</extra>"}};
}}
const AI_AXES = ["Partisan attack","Enemy naming","Zero-sum","Proposal share","Values share","Effective topic breadth"];
const AI_KEYS = ["party_attack","enemy_naming","zero_sum","proposal","values","topic_breadth"];
function aiRadarTrace(name, i) {{
  const radar = P[name].ai.ai_radar;
  const values = AI_KEYS.map(k => radar[k].percentile);
  const absolute = AI_KEYS.map(k => radar[k].absolute);
  return {{type:"scatterpolar", name, r:values.concat([values[0]]),
    theta:AI_AXES.concat([AI_AXES[0]]), customdata:absolute.concat([absolute[0]]),
    fill:"toself", line:{{color:COLORS[i],width:2}}, fillcolor:COLORS[i]+"28",
    hovertemplate:"%{{theta}}: %{{r:.0f}}th pct<br>absolute %{{customdata:.2f}}<extra>"+name+"</extra>"}};
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
function populateLegacyIssueLabels(sel) {{
  document.querySelectorAll(".legacy-issue-label").forEach(el => {{
    const slot = Number(el.dataset.slot);
    const issueIndex = Number(el.dataset.issueIndex);
    el.textContent = P[sel[slot]].issues[issueIndex][0];
  }});
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
  Plotly.react("ai-rates", sel.map((n, i) => aiTrace(n, i)), {{
    template:"simple_white", paper_bgcolor:CHROME.surface, plot_bgcolor:CHROME.surface,
    font:{{family:"{FONT}", color:CHROME.ink2, size:12}}, barmode:"group", height:360,
    margin:{{l:55,r:20,t:25,b:70}},
    yaxis:{{title:"% of paragraphs", rangemode:"tozero", gridcolor:CHROME.grid}},
    xaxis:{{tickfont:{{color:CHROME.ink}}}}, legend:{{orientation:"h",y:1.08}}
  }}, {{displayModeBar:false,responsive:true}});
  Plotly.react("ai-radar", sel.map((n, i) => aiRadarTrace(n, i)), {{
    template:"simple_white", paper_bgcolor:CHROME.surface,
    font:{{family:"{FONT}",color:CHROME.ink2,size:12}},
    polar:{{bgcolor:CHROME.surface,radialaxis:{{range:[0,100],gridcolor:CHROME.grid}},
           angularaxis:{{gridcolor:CHROME.grid}}}},
    legend:{{orientation:"h",x:0,y:1.12}},height:460,margin:{{l:95,r:95,t:44,b:38}}
  }}, {{displayModeBar:false,responsive:true}});
  document.getElementById("cols").innerHTML =
    sel.map((n, i) => colHTML(n, i)).join("");
  populateLegacyIssueLabels(sel);
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
document.getElementById("download-comparison").onclick = () => {{
  const selected = ["sela","selb","selc"].map(id=>document.getElementById(id).value).filter(Boolean);
  const blob = new Blob([JSON.stringify({{schema_version:"president-comparison-v1",
    presidents:Object.fromEntries(selected.map(name=>[name,P[name]]))}}, null, 2)],
    {{type:"application/json"}});
  const link=document.createElement("a");link.href=URL.createObjectURL(blob);
  link.download="presidential-comparison.json";link.click();URL.revokeObjectURL(link.href);
}};
document.addEventListener("toggle", event => {{
  if (!event.target.matches("details.compare-section") || !event.target.open) return;
  event.target.querySelectorAll(".chart").forEach(chart => Plotly.Plots.resize(chart));
}}, true);
render();
</script>
</body>
</html>
"""
    (REPO_ROOT / "docs" / "compare.html").write_text(html)
    print("  wrote docs/compare.html")
