"""Build the data behind the word & phrase explorer.

Precomputes per-year counts for every unigram (>=30 total uses) and bigram
(>=15), sharded by first letter so the explorer page can lazy-load only
what the user searches for. Also ships per-year issue shares so topics can
be overlaid on the same chart.
"""

import json
import re
from collections import Counter, defaultdict

import pandas as pd

from .corpus import load
from . import issues
from .figures import REPO_ROOT

EXPLORER_DIR = REPO_ROOT / "docs" / "explorer"

WORD_RE = re.compile(r"[a-z']+")
MIN_UNIGRAM = 30
MIN_BIGRAM = 15


def build_explorer_data() -> None:
    df = load()
    EXPLORER_DIR.mkdir(parents=True, exist_ok=True)

    # Pass 1: global counts to pick the kept vocabulary.
    uni_total: Counter = Counter()
    bi_total: Counter = Counter()
    speech_tokens: list[tuple[int, list[str]]] = []
    for _, sp in df.iterrows():
        toks = WORD_RE.findall(sp["transcript"].lower())
        speech_tokens.append((int(sp["year"]), toks))
        uni_total.update(toks)
        bi_total.update(" ".join(p) for p in zip(toks, toks[1:]))
    keep_uni = {w for w, n in uni_total.items() if n >= MIN_UNIGRAM}
    keep_bi = {b for b, n in bi_total.items() if n >= MIN_BIGRAM}

    # Pass 2: per-year counts for kept terms.
    per_year_words: Counter = Counter()
    uni_years: dict[str, Counter] = defaultdict(Counter)
    bi_years: dict[str, Counter] = defaultdict(Counter)
    for year, toks in speech_tokens:
        per_year_words[year] += len(toks)
        for w in toks:
            if w in keep_uni:
                uni_years[w][year] += 1
        for b in (" ".join(p) for p in zip(toks, toks[1:])):
            if b in keep_bi:
                bi_years[b][year] += 1

    def shard(entries: dict, prefix: str) -> None:
        shards: dict[str, dict] = defaultdict(dict)
        for term, years in entries.items():
            first = term[0]
            key = first if first.isalpha() else "0"
            pairs = sorted(years.items())
            shards[key][term] = {"y": [y for y, _ in pairs],
                                 "c": [c for _, c in pairs]}
        for key, content in shards.items():
            (EXPLORER_DIR / f"{prefix}_{key}.json").write_text(
                json.dumps(content, separators=(",", ":")))

    shard(uni_years, "u")
    shard(bi_years, "b")

    # Topics: share of paragraphs per year for each display issue.
    para_labels = pd.read_parquet(issues.PARA_LABELS_PATH)
    meta_issues = json.loads(issues.ISSUES_META_PATH.read_text())
    display = meta_issues["issues"] + ["Discovered 5"]
    topic_data = {}
    year_counts = para_labels.groupby("year").size()
    for name in display:
        label = "Security & peace" if name == "Discovered 5" else name
        share = (para_labels.groupby("year")[name].mean() * 100)
        share = share[year_counts >= 10].round(2)
        topic_data[label] = {"y": [int(y) for y in share.index],
                             "v": list(share.values)}

    years = sorted(per_year_words)
    meta = {
        "years": years,
        "totals": [per_year_words[y] for y in years],
        "topics": topic_data,
    }
    (EXPLORER_DIR / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    n_files = len(list(EXPLORER_DIR.glob("*.json")))
    size = sum(f.stat().st_size for f in EXPLORER_DIR.glob("*.json")) / 1e6
    print(f"  explorer: {len(keep_uni):,} words, {len(keep_bi):,} phrases, "
          f"{n_files} files, {size:.1f} MB")


def write_page() -> None:
    from .figures import BASELINE, GRID, INK, INK2, MUTED, SERIES, SURFACE
    from .site_style import FONT, PAGE_CSS

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Word explorer - Presidential Profiles</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
<style>
{PAGE_CSS}
  .crumbs {{ margin-bottom: 18px; font-size: 0.88rem; }}
  .crumbs a {{ color: var(--ink2); }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }}
  .controls input {{ flex: 1; min-width: 220px; padding: 10px 14px; font-size: 1rem;
                     border: 1px solid var(--border); border-radius: 10px;
                     background: var(--surface); color: var(--ink);
                     font-family: inherit; }}
  .controls input:focus {{ outline: 2px solid var(--muted); }}
  .controls select {{ padding: 10px 12px; border: 1px solid var(--border);
                      border-radius: 10px; background: var(--surface);
                      color: var(--ink); font-family: inherit; font-size: 0.95rem; }}
  .controls button {{ padding: 10px 18px; border: 1px solid var(--border);
                      border-radius: 10px; background: var(--ink); color: var(--page);
                      font-family: inherit; font-size: 0.95rem; cursor: pointer; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; min-height: 34px; }}
  .chip {{ display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px;
           background: var(--surface); border: 1px solid var(--border);
           border-radius: 999px; font-size: 0.9rem; }}
  .chip .swatch {{ width: 10px; height: 10px; border-radius: 50%; }}
  .chip button {{ border: none; background: none; cursor: pointer; color: var(--muted);
                  font-size: 1rem; padding: 0; }}
  #msg {{ color: var(--muted); font-size: 0.88rem; min-height: 1.3em; }}
  #unitnote {{ color: var(--muted); font-size: 0.82rem; margin-top: 8px; }}
</style>
</head>
<body>
<header>
  <p class="crumbs"><a href="index.html">← Dashboard</a> &nbsp;·&nbsp;
     <a href="presidents/index.html">President profiles</a></p>
  <h1>Word explorer</h1>
  <p class="sub">Type any word or two-word phrase - “tariff”, “middle
  class”, “crypto”, “slavery” - and see its 240-year history in
  presidential speech. Add several to compare, or overlay an issue from the topic menu.
  Rates are 5-year rolling per 10,000 words.</p>
  <div class="controls">
    <input id="q" placeholder="type a word or two-word phrase, press Enter"
           autocomplete="off">
    <select id="topic"><option value="">add a topic…</option></select>
    <button id="add">Add</button>
  </div>
  <div class="chips" id="chips"></div>
  <div id="msg"></div>
</header>
<main>
  <div class="chart-scroll"><div class="chart" id="chart" style="height:480px"></div></div>
  <p id="unitnote"></p>
</main>
<footer>
  <p>Words with at least 30 uses and phrases with at least 15 across the corpus are
  indexed. Data: <a href="https://data.millercenter.org">Miller Center of Public
  Affairs, University of Virginia</a>.</p>
</footer>
<script>
const COLORS = {json.dumps(SERIES)};
const CHROME = {{surface:"{SURFACE}", ink:"{INK}", ink2:"{INK2}", muted:"{MUTED}",
                 grid:"{GRID}", baseline:"{BASELINE}"}};
const FONT = "{FONT}";
let META = null;
const shardCache = {{}};
let series = [];

async function loadJSON(path) {{
  const r = await fetch(path);
  if (!r.ok) throw new Error(path);
  return r.json();
}}
async function getShard(prefix, letter) {{
  const key = prefix + "_" + letter;
  if (!(key in shardCache)) {{
    try {{ shardCache[key] = await loadJSON("explorer/" + key + ".json"); }}
    catch (e) {{ shardCache[key] = {{}}; }}
  }}
  return shardCache[key];
}}
function rolling(years, values, allYears, window) {{
  const map = new Map(years.map((y, i) => [y, values[i]]));
  const half = Math.floor(window / 2);
  return allYears.map(y => {{
    let s = 0;
    for (let d = -half; d <= half; d++) s += map.get(y + d) || 0;
    return s;
  }});
}}
function seriesTrace(s, idx, indexed) {{
  let x, y;
  if (s.type === "term") {{
    const counts = rolling(s.data.y, s.data.c, META.years, 5);
    const totals = rolling(META.years, META.totals, META.years, 5);
    x = META.years;
    y = counts.map((c, i) => totals[i] > 20000 ? c / totals[i] * 10000 : null);
  }} else {{
    const m = new Map(s.data.y.map((yy, i) => [yy, s.data.v[i]]));
    x = META.years;
    const vals = META.years.map(yy => m.has(yy) ? m.get(yy) : null);
    y = x.map((yy, i) => {{
      let s2 = 0, n = 0;
      for (let d = -2; d <= 2; d++) {{
        const v = m.get(yy + d);
        if (v != null) {{ s2 += v; n++; }}
      }}
      return n ? s2 / n : null;
    }});
  }}
  if (indexed) {{
    const mx = Math.max(...y.filter(v => v != null));
    y = y.map(v => v == null ? null : v / mx * 100);
  }}
  return {{x, y, name: s.label, mode: "lines", connectgaps: false,
          line: {{color: COLORS[idx % COLORS.length], width: 2.4}},
          hovertemplate: "%{{y:.2f}}<extra>" + s.label + "</extra>"}};
}}
function redraw() {{
  const mixed = new Set(series.map(s => s.type)).size > 1;
  const traces = series.map((s, i) => seriesTrace(s, i, mixed));
  const ylabel = mixed ? "% of each series' own peak"
    : (series[0] && series[0].type === "topic" ? "% of paragraphs"
       : "uses per 10,000 words");
  Plotly.react("chart", traces, {{
    template: "simple_white", paper_bgcolor: CHROME.surface,
    plot_bgcolor: CHROME.surface,
    font: {{family: FONT, color: CHROME.ink, size: 13}},
    margin: {{l: 56, r: 24, t: 24, b: 44}},
    xaxis: {{range: [1786, 2029], gridcolor: CHROME.grid,
            linecolor: CHROME.baseline, tickfont: {{color: CHROME.muted, size: 11}}}},
    yaxis: {{title: ylabel, gridcolor: CHROME.grid, linecolor: CHROME.baseline,
            tickfont: {{color: CHROME.muted, size: 11}}, rangemode: "tozero"}},
    legend: {{orientation: "h", yanchor: "bottom", y: 1.01, x: 0}},
  }}, {{displayModeBar: false, responsive: true}});
  document.getElementById("unitnote").textContent = mixed
    ? "Mixing words and topics: each line is scaled to its own peak (=100) so shapes are comparable."
    : "";
  renderChips();
}}
function renderChips() {{
  const el = document.getElementById("chips");
  el.innerHTML = "";
  series.forEach((s, i) => {{
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<span class="swatch" style="background:${{COLORS[i % COLORS.length]}}"></span>
      ${{s.label}} <button aria-label="remove">×</button>`;
    chip.querySelector("button").onclick = () => {{ series.splice(i, 1); redraw(); }};
    el.appendChild(chip);
  }});
}}
function msg(t) {{ document.getElementById("msg").textContent = t; }}
async function addTerm(raw) {{
  const q = raw.trim().toLowerCase().replace(/[^a-z' ]/g, "").replace(/ +/g, " ");
  if (!q) return;
  if (series.some(s => s.label === q)) {{ msg("already on the chart"); return; }}
  const words = q.split(" ");
  if (words.length > 2) {{ msg("words and two-word phrases only (for now)"); return; }}
  const prefix = words.length === 1 ? "u" : "b";
  const letter = /[a-z]/.test(q[0]) ? q[0] : "0";
  msg("loading…");
  const shard = await getShard(prefix, letter);
  if (!(q in shard)) {{
    msg(`“${{q}}” ${{words.length === 1
      ? "appears fewer than 30 times in 240 years of presidential speech"
      : "appears fewer than 15 times as a phrase"}}`);
    return;
  }}
  msg("");
  series.push({{label: q, type: "term", data: shard[q]}});
  redraw();
}}
function addTopic(name) {{
  if (!name || series.some(s => s.label === name)) return;
  series.push({{label: name, type: "topic", data: META.topics[name]}});
  redraw();
}}
(async () => {{
  META = await loadJSON("explorer/meta.json");
  const sel = document.getElementById("topic");
  Object.keys(META.topics).forEach(t => {{
    const o = document.createElement("option"); o.value = t; o.textContent = t;
    sel.appendChild(o);
  }});
  sel.onchange = () => {{ addTopic(sel.value); sel.value = ""; }};
  const input = document.getElementById("q");
  input.addEventListener("keydown", e => {{
    if (e.key === "Enter") {{ addTerm(input.value); input.value = ""; }}
  }});
  document.getElementById("add").onclick = () => {{ addTerm(input.value); input.value = ""; }};
  for (const t of ["tariff", "freedom", "border"]) await addTerm(t);
}})();
</script>
</body>
</html>
"""
    (REPO_ROOT / "docs" / "explorer.html").write_text(html)
    print("  wrote docs/explorer.html")
