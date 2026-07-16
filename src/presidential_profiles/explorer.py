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
from . import issues, word_families
from .figures import REPO_ROOT

EXPLORER_DIR = REPO_ROOT / "docs" / "explorer"

WORD_RE = re.compile(r"[a-z']+")
MIN_UNIGRAM = 30
MIN_BIGRAM = 15
_ACRONYM_RES = {acr: re.compile(rf"\b{acr}\b") for acr in word_families.ACRONYMS}


def _shard(entries: dict, prefix: str, value_key: str = "c") -> None:
    """Write per-first-letter shards: {term: {y:[years], c:[counts]}}."""
    shards: dict[str, dict] = defaultdict(dict)
    for term, years in entries.items():
        first = term[0]
        key = first if first.isalpha() else "0"
        pairs = sorted(years.items())
        shards[key][term] = {"y": [y for y, _ in pairs],
                             value_key: [c for _, c in pairs]}
    for key, content in shards.items():
        (EXPLORER_DIR / f"{prefix}_{key}.json").write_text(
            json.dumps(content, separators=(",", ":")))


def build_explorer_data() -> None:
    df = load()
    EXPLORER_DIR.mkdir(parents=True, exist_ok=True)
    fam = word_families.build_families(df)          # {form: node_label}

    # Pass 1: exact tokens (byte-identical to the pre-grouping build), plus a
    # parallel grouped tokenization with case-collision acronyms peeled off, and
    # the acronym counts themselves as their own entity series.
    uni_total: Counter = Counter()
    bi_total: Counter = Counter()
    guni_total: Counter = Counter()
    gbi_total: Counter = Counter()
    speech_tokens: list[tuple[int, list[str], list[str]]] = []
    acr_years: dict[str, Counter] = defaultdict(Counter)
    for _, sp in df.iterrows():
        year = int(sp["year"])
        raw = sp["transcript"]
        toks = WORD_RE.findall(raw.lower())          # exact: unchanged
        # grouped: strip lexicalized acronyms case-sensitively, then normalize
        g_raw = raw
        for acr, label in word_families.ACRONYMS.items():
            n = len(_ACRONYM_RES[acr].findall(g_raw))
            if n:
                acr_years[label][year] += n
                g_raw = _ACRONYM_RES[acr].sub(" ", g_raw)
        gtoks = [fam.get(t, t) for t in word_families.tokenize(g_raw)]
        speech_tokens.append((year, toks, gtoks))
        uni_total.update(toks)
        bi_total.update(" ".join(p) for p in zip(toks, toks[1:]))
        guni_total.update(gtoks)
        gbi_total.update(" ".join(p) for p in zip(gtoks, gtoks[1:]))
    keep_uni = {w for w, n in uni_total.items() if n >= MIN_UNIGRAM}
    keep_bi = {b for b, n in bi_total.items() if n >= MIN_BIGRAM}
    keep_guni = {w for w, n in guni_total.items() if n >= MIN_UNIGRAM}
    keep_gbi = {b for b, n in gbi_total.items() if n >= MIN_BIGRAM}

    # Pass 2: per-year counts for kept exact and grouped terms.
    per_year_words: Counter = Counter()
    uni_years: dict[str, Counter] = defaultdict(Counter)
    bi_years: dict[str, Counter] = defaultdict(Counter)
    guni_years: dict[str, Counter] = defaultdict(Counter)
    gbi_years: dict[str, Counter] = defaultdict(Counter)
    for year, toks, gtoks in speech_tokens:
        per_year_words[year] += len(toks)
        for w in toks:
            if w in keep_uni:
                uni_years[w][year] += 1
        for b in (" ".join(p) for p in zip(toks, toks[1:])):
            if b in keep_bi:
                bi_years[b][year] += 1
        for w in gtoks:
            if w in keep_guni:
                guni_years[w][year] += 1
        for b in (" ".join(p) for p in zip(gtoks, gtoks[1:])):
            if b in keep_gbi:
                gbi_years[b][year] += 1

    _shard(uni_years, "u")           # exact  (byte-identical to before)
    _shard(bi_years, "b")
    _shard(guni_years, "gu")         # grouped, keyed by node label
    _shard(gbi_years, "gb")

    # Client-side form -> node lookup (every form whose label differs, so the
    # explorer can resolve a typed "immigrants" to the "immigration" node, and
    # resolve each half of a grouped bigram). Also map each NORMALIZE variant to
    # its canonical node, so a word folded out of the grouped vocabulary
    # ("defence" -> "defense") still resolves when the toggle is on.
    resolve = {f: lab for f, lab in fam.items() if lab != f}
    for variant, canon in word_families.NORMALIZE.items():
        if canon in fam:
            resolve[variant] = fam[canon]
    (EXPLORER_DIR / "families.json").write_text(
        json.dumps(resolve, separators=(",", ":")))

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

    # Entities: the acronyms peeled out of their colliding word, as term-like
    # per-year counts (rate per 10k words), offered from the dropdown.
    entities = {}
    for label, years in acr_years.items():
        pairs = sorted(years.items())
        entities[label] = {"y": [y for y, _ in pairs],
                           "c": [c for _, c in pairs]}

    years = sorted(per_year_words)
    meta = {
        "years": years,
        "totals": [per_year_words[y] for y in years],
        "topics": topic_data,
        "entities": entities,
    }
    (EXPLORER_DIR / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    n_files = len(list(EXPLORER_DIR.glob("*.json")))
    size = sum(f.stat().st_size for f in EXPLORER_DIR.glob("*.json")) / 1e6
    print(f"  explorer: {len(keep_uni):,} words / {len(keep_guni):,} grouped, "
          f"{len(keep_bi):,} phrases, {n_files} files, {size:.1f} MB")


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
  .chip .count {{ color: var(--muted); font-size: 0.78rem; cursor: help;
                  border-bottom: 1px dotted var(--border); }}
  #tip {{ position: fixed; z-index: 50; max-width: 340px; padding: 8px 11px;
          background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; font-size: 0.84rem; color: var(--ink);
          line-height: 1.5; pointer-events: none;
          box-shadow: 0 6px 20px rgba(0,0,0,0.14); }}
  #tip[hidden] {{ display: none; }}
  #tip .thead {{ color: var(--muted); font-size: 0.72rem; text-transform: uppercase;
                 letter-spacing: 0.04em; margin-bottom: 4px; }}
  .chip button {{ border: none; background: none; cursor: pointer; color: var(--muted);
                  font-size: 1rem; padding: 0; }}
  #msg {{ color: var(--muted); font-size: 0.88rem; min-height: 1.3em; }}
  #unitnote {{ color: var(--muted); font-size: 0.82rem; margin-top: 8px; }}
  .grouptoggle {{ display: inline-flex; align-items: center; gap: 7px;
                  font-size: 0.9rem; color: var(--ink2); cursor: pointer;
                  margin: 6px 0 2px; }}
  .grouptoggle input {{ flex: none; min-width: 0; width: auto; margin: 0;
                        accent-color: var(--ink); cursor: pointer; }}
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
  <label class="grouptoggle" for="grp"><input type="checkbox" id="grp">
    group word forms (count “immigrant”, “immigrants”, “immigration” as one)</label>
  <div class="chips" id="chips"></div>
  <div id="msg"></div>
  <div id="tip" hidden></div>
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
let FAMILIES = {{}};
const MEMBERS = {{}};        // node label -> [surface forms folded into it]
let grouped = false;
const shardCache = {{}};
let series = [];

// The surface forms counted under each position of a (grouped) query, e.g.
// ["immigration"] -> [["immigration","immigrants","immigrant"]].
function memberForms(words) {{
  return words.map(w => {{
    const lab = node(w);
    return [...new Set([lab].concat(MEMBERS[lab] || []))];
  }});
}}

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
  const extra = s.members ? s.label + "<br>" + s.members : s.label;
  return {{x, y, name: s.label, mode: "lines", connectgaps: false,
          line: {{color: COLORS[idx % COLORS.length], width: 2.4}},
          hovertemplate: "%{{y:.2f}}<extra>" + extra + "</extra>"}};
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
    const badge = s.members
      ? `<span class="count">${{s.nforms}} forms</span>` : "";
    chip.innerHTML = `<span class="swatch" style="background:${{COLORS[i % COLORS.length]}}"></span>
      ${{s.label}} ${{badge}}<button aria-label="remove">×</button>`;
    chip.querySelector("button").onclick = () => {{ series.splice(i, 1); hideTip(); redraw(); }};
    if (s.members) {{
      chip.addEventListener("mousemove", e => showTip(s, e));
      chip.addEventListener("mouseleave", hideTip);
    }}
    el.appendChild(chip);
  }});
}}
function msg(t) {{ document.getElementById("msg").textContent = t; }}
function showTip(s, e) {{
  const tip = document.getElementById("tip");
  const list = s.members.split(" / ").join(", ").split("  +  ").join("  •  ");
  tip.innerHTML = `<div class="thead">counted as “${{s.label}}” (${{s.nforms}} forms)</div>${{list}}`;
  tip.hidden = false;
  moveTip(e);
}}
function moveTip(e) {{
  const tip = document.getElementById("tip"), pad = 14, r = tip.getBoundingClientRect();
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + r.width > innerWidth) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight) y = e.clientY - r.height - pad;
  tip.style.left = Math.max(4, x) + "px"; tip.style.top = Math.max(4, y) + "px";
}}
function hideTip() {{ document.getElementById("tip").hidden = true; }}
const node = w => (grouped && FAMILIES[w]) ? FAMILIES[w] : w;
const shardLetter = w => /[a-z]/.test(w[0]) ? w[0] : "0";
async function addTerm(raw, silent) {{
  const q = raw.trim().toLowerCase().replace(/[^a-z' ]/g, "").replace(/ +/g, " ");
  if (!q) return;
  const words = q.split(" ");
  if (words.length > 2) {{ msg("words and two-word phrases only (for now)"); return; }}
  // Resolve to the family node when grouping is on: "immigrants" -> "immigration".
  const key = words.map(node).join(" ");
  const label = grouped ? key : q;
  if (series.some(s => s.label === label)) {{
    if (!silent) msg("already on the chart"); return; }}
  const shardPrefix = grouped ? (words.length === 1 ? "gu" : "gb")
                              : (words.length === 1 ? "u" : "b");
  if (!silent) msg("loading…");
  const shard = await getShard(shardPrefix, shardLetter(key));
  if (!(key in shard)) {{
    if (!silent) msg(`“${{q}}” ${{words.length === 1
      ? "appears fewer than 30 times in 240 years of presidential speech"
      : "appears fewer than 15 times as a phrase"}}`);
    return;
  }}
  msg("");
  // What's actually being counted, for the chip badge and hover tooltip.
  let members = null, nforms = 0;
  if (grouped) {{
    const perPos = memberForms(words);
    const total = perPos.reduce((a, f) => a + f.length, 0);
    if (total > words.length) {{        // at least one position folds >1 form
      members = perPos.map(f => f.join(" / ")).join("  +  ");
      nforms = words.length === 1 ? perPos[0].length : total;
    }}
  }}
  series.push({{label, q: raw, type: "term", data: shard[key], members, nforms}});
  if (!silent) redraw();
}}
function addTopic(name) {{
  if (!name || series.some(s => s.label === name)) return;
  series.push({{label: name, type: "topic", data: META.topics[name]}});
  redraw();
}}
function addEntity(name) {{
  if (!name || series.some(s => s.label === name)) return;
  series.push({{label: name, type: "term", data: META.entities[name]}});
  redraw();
}}
async function regroup() {{
  grouped = document.getElementById("grp").checked;
  const queries = series.filter(s => s.q !== undefined).map(s => s.q);
  series = series.filter(s => s.q === undefined);   // keep topics + entities
  for (const q of queries) await addTerm(q, true);
  redraw();
}}
(async () => {{
  [META, FAMILIES] = await Promise.all([
    loadJSON("explorer/meta.json"), loadJSON("explorer/families.json")]);
  for (const f in FAMILIES) (MEMBERS[FAMILIES[f]] ||= []).push(f);
  const sel = document.getElementById("topic");
  Object.keys(META.topics).forEach(t => {{
    const o = document.createElement("option"); o.value = "t:" + t; o.textContent = t;
    sel.appendChild(o);
  }});
  Object.keys(META.entities || {{}}).forEach(t => {{
    const o = document.createElement("option"); o.value = "e:" + t; o.textContent = t;
    sel.appendChild(o);
  }});
  sel.onchange = () => {{
    const v = sel.value;
    if (v.startsWith("t:")) addTopic(v.slice(2));
    else if (v.startsWith("e:")) addEntity(v.slice(2));
    sel.value = "";
  }};
  const input = document.getElementById("q");
  input.addEventListener("keydown", e => {{
    if (e.key === "Enter") {{ addTerm(input.value); input.value = ""; }}
  }});
  document.getElementById("add").onclick = () => {{ addTerm(input.value); input.value = ""; }};
  document.getElementById("grp").addEventListener("change", regroup);
  for (const t of ["tariff", "freedom", "border"]) await addTerm(t, true);
  redraw();
}})();
</script>
</body>
</html>
"""
    (REPO_ROOT / "docs" / "explorer.html").write_text(html)
    print("  wrote docs/explorer.html")
