"""Reader-facing Methods appendices shared by the generated static site.

The module keeps presentation separate from the Data Quality audit.  It never
mutates analytical sources: the model-comparison page derives its values from
governed artifacts in memory, while the metric dictionary renders the central
registry in a stable, searchable order.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import attention, corpus, metrics
from .issues import ISSUE_ANCHORS


METHODS_NAV = (
    ("Overview", "methodology.html#overview", "overview"),
    ("Labeling", "methodology.html#labeling", "labeling"),
    ("Evaluation", "methodology.html#evaluation", "evaluation"),
    ("Populations", "methodology.html#populations", "populations"),
    ("Model Comparison", "label-models.html", "model-comparison"),
    ("Metric Dictionary", "metrics.html", "metric-dictionary"),
    ("Downloads", "methodology.html#downloads", "downloads"),
)

MODEL_COMPARISON_PUBLIC_SCHEMA = "method-comparison-public-v1"
METHOD_AGREEMENT_KEY = ("issue", "era")
METHOD_COMPOSITIONS_KEY = ("doc_name", "method", "topic")

CURRENT_NAV_SCRIPT = r"""<script>
(() => {
  const nav = document.querySelector(".methods-local-nav");
  const current = nav?.querySelector("[aria-current='page']");
  if (!nav || !current) return;
  const reveal = () => {
    if (!window.matchMedia("(max-width: 780px)").matches) return;
    const left = current.offsetLeft - (nav.clientWidth - current.offsetWidth) / 2;
    nav.scrollTo({left: Math.max(0, left), behavior: "auto"});
  };
  reveal();
  window.addEventListener("resize", reveal, {passive: true});
})();
</script>"""


APPENDIX_CSS = r"""
.methods-local-nav{position:sticky;top:var(--global-nav-height,0);z-index:30;
margin:20px 0 28px;background:rgba(249,249,247,.97);border-block:1px solid var(--border);
overflow-x:auto;scrollbar-width:thin}.methods-local-nav ul{display:flex;list-style:none;
padding:6px 0;margin:0;min-width:max-content}.methods-local-nav a{display:grid;
place-items:center;min-height:40px;padding:6px 12px;white-space:nowrap;color:var(--ink2);
text-decoration:none;font-size:.78rem;font-weight:680;border-radius:8px}
.methods-scroll-cue{display:none;align-items:center;white-space:nowrap;padding:6px 10px;
color:#52616c;font-size:.7rem;font-weight:750}
.methods-local-nav a:hover{background:#ece7df}.methods-local-nav a[aria-current=page]{
background:#172a38;color:#fff}.appendix-kicker{color:#275d8c;font-size:.72rem;
font-weight:800;letter-spacing:.08em;text-transform:uppercase}.verdict-panel{display:grid;
grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:18px 0 32px}
.verdict-panel article,.receipt-card,.instrument-card{background:var(--surface);
border:1px solid var(--border);border-radius:14px;padding:17px}.verdict-panel h2,
.verdict-panel h3{font-size:1rem;margin:.35rem 0}.verdict-panel p{font-size:.86rem;
color:var(--ink2)}.appendix-section{margin-top:50px;scroll-margin-top:190px}
.appendix-section>h2{font:700 clamp(1.45rem,3vw,2rem)/1.15 Georgia,serif}
.appendix-section>p{max-width:800px;color:var(--ink2)}.trust-receipt{display:grid;
grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--border);
border:1px solid var(--border);border-radius:14px;overflow:hidden;margin:20px 0}
.trust-receipt div{background:#fff;padding:13px;min-width:0}.trust-receipt dt{
color:#52616c;font-size:.68rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase}
.trust-receipt dd{margin:.35rem 0 0;font-size:.8rem;overflow-wrap:anywhere}
.metric-tools{display:flex;flex-wrap:wrap;gap:10px;align-items:end;padding:15px;
background:var(--surface);border:1px solid var(--border);border-radius:14px;margin:18px 0}
.metric-search{display:grid;gap:5px;min-width:min(100%,260px);flex:1}.metric-search span{
font-size:.75rem;font-weight:750}.metric-search input{min-height:44px;border:1px solid #8b969e;
border-radius:9px;padding:9px 11px;background:#fff;color:var(--ink)}.metric-filters{
display:flex;flex-wrap:wrap;gap:6px}.metric-filters button{min-height:40px;border:1px solid #83909a;
border-radius:999px;background:#fff;color:var(--ink2);padding:7px 11px;cursor:pointer}
.metric-filters button[aria-pressed=true]{background:#172a38;color:#fff;border-color:#172a38}
.metric-results{min-height:1.3em;color:var(--muted);font-size:.8rem;margin-top:8px}
.metric-group{margin-top:38px;scroll-margin-top:190px}.metric-group>h2{font-size:1.35rem}
.metric-card-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px;
margin-top:14px}.metric-card{position:relative;background:#fff;border:1px solid var(--border);
border-radius:14px;padding:17px;scroll-margin-top:190px}.metric-card h3{font-size:1rem;
padding-right:28px}.metric-card .metric-question{font-weight:700;margin-top:8px}
.metric-card .metric-definition{font-size:.86rem;color:var(--ink2);margin-top:7px}
.metric-card .deep-link{position:absolute;right:12px;top:11px;color:#52616c;
font-weight:800;text-decoration:none;padding:5px;min-width:24px;box-sizing:border-box;
text-align:center}.metric-card details{margin-top:13px;background:var(--surface)}
.metric-card dl{display:grid;grid-template-columns:minmax(90px,.35fr) 1fr;gap:7px 12px;
font-size:.78rem}.metric-card dt{font-weight:750}.metric-card dd{margin:0;overflow-wrap:anywhere}
.metric-card ol{padding-left:20px}.metric-card li{margin:5px 0}.metric-empty{padding:24px;
border:1px dashed #87939c;border-radius:12px}.instrument-grid{display:grid;
grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.instrument-card h2{font-size:1.15rem}
.instrument-card .eyebrow{font-size:.69rem;letter-spacing:.06em;text-transform:uppercase;
font-weight:800}.instrument-card ul{padding-left:20px}.instrument-card li{margin:.55rem 0}
.failed-prediction{border:2px solid #82551f;border-left-width:8px;border-radius:14px;
background:#fff7e8;padding:20px;margin:22px 0}.failed-prediction .result{font:700 1.4rem Georgia,serif}
.failed-prediction p{max-width:850px}.comparison-figure{margin:24px 0;background:var(--surface);
border:1px solid var(--border);border-radius:16px;padding:18px}.comparison-figure figcaption{
margin-bottom:16px}.comparison-figure figcaption strong,.comparison-figure figcaption span{
display:block}.comparison-figure figcaption span{color:var(--muted);font-size:.82rem;margin-top:4px}
.comparison-figure ol{list-style:none;padding:0}.rate-row{display:grid;
grid-template-columns:minmax(145px,.8fr) minmax(260px,2fr);gap:14px;padding:12px 0;
border-top:1px solid var(--grid)}.rate-row h3{font-size:.86rem}.rate-pair{display:grid;gap:7px}
.rate-line{display:grid;grid-template-columns:108px minmax(90px,1fr) 52px;gap:8px;
align-items:center;font-size:.75rem}.track{height:10px;background:#e4e8ea;border-radius:999px;
overflow:hidden}.fill{display:block;height:100%;width:calc(var(--value) * 1%);min-width:1px;
background:#205f8f}.corex .fill{background:repeating-linear-gradient(135deg,#87561f 0 5px,
#c99452 5px 9px)}.agreement-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
gap:9px;list-style:none;padding:0}.agreement-row{background:#fff;border:1px solid var(--border);
border-radius:12px;padding:13px}.agreement-row h3{font-size:.88rem}.agreement-values{display:grid;
grid-template-columns:1fr 1fr;gap:8px;margin-top:9px}.agreement-values div{border-left:4px solid #205f8f;
padding-left:8px}.agreement-values div+div{border-left-color:#87561f}.agreement-values span,
.agreement-values strong{display:block}.agreement-values span{font-size:.69rem;color:var(--muted)}
.agreement-values small{display:block;color:var(--muted);margin-top:8px;font-size:.68rem}
.anchor-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;
list-style:none;padding:0}.anchor-list li{background:#fff;border:1px solid var(--border);
border-radius:11px;padding:13px}.anchor-list strong,.anchor-list span{display:block}
.anchor-list span{font-size:.78rem;color:var(--ink2);margin-top:5px}.receipt-grid{display:grid;
grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.receipt-card span{color:#275d8c;
font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;font-weight:800}.receipt-card h3{
font-size:.95rem;margin-top:6px}.receipt-card p,.receipt-card code{font-size:.76rem;
overflow-wrap:anywhere}.download-links{display:flex;flex-wrap:wrap;gap:8px;list-style:none;padding:0}
.download-links a{display:block;background:#fff;border:1px solid var(--border);border-radius:9px;
padding:9px 11px}.methods-local-nav a:focus-visible,.metric-card a:focus-visible,
.metric-tools button:focus-visible,.metric-tools input:focus-visible,.download-links a:focus-visible{
outline:3px solid #824600;outline-offset:2px}
@media(max-width:780px){.verdict-panel,.trust-receipt,.receipt-grid{grid-template-columns:1fr 1fr}
.methods-scroll-cue{display:flex}
.metric-card-list,.instrument-grid,.agreement-list{grid-template-columns:1fr}.rate-row{grid-template-columns:1fr}}
@media(max-width:430px){.verdict-panel,.trust-receipt,.receipt-grid{grid-template-columns:1fr}
.comparison-figure{padding:14px}.rate-line{grid-template-columns:92px minmax(70px,1fr) 48px}
.metric-card{padding:15px}.anchor-list{grid-template-columns:1fr}}
@media(forced-colors:active){.fill{background:Highlight}.corex .fill{background:CanvasText}}
"""


def methods_local_nav(current: str) -> str:
    links = []
    for label, href, key in METHODS_NAV:
        active = ' aria-current="page"' if key == current else ""
        links.append(f'<li><a href="{href}"{active}>{html.escape(label)}</a></li>')
    return (
        '<nav class="methods-local-nav" aria-label="Methods sections"><ul>'
        '<li class="methods-scroll-cue" aria-hidden="true">More sections →</li>'
        + "".join(links)
        + "</ul></nav>"
    )


def trust_receipt(
    *,
    claim: str,
    population: str,
    method: str,
    uncertainty: str,
    artifact: str,
    status: str,
    limitation: str,
    downstream_use: str,
) -> str:
    fields = (
        ("Claim", claim),
        ("Population", population),
        ("Method", method),
        ("Uncertainty", uncertainty),
        ("Artifact", artifact),
        ("Status", status),
        ("Limitation", limitation),
        ("Downstream use", downstream_use),
    )
    return '<dl class="trust-receipt">' + "".join(
        f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
        for label, value in fields
    ) + "</dl>"


def render_metric_dictionary() -> tuple[str, str]:
    """Return body and progressive-filter script for ``metrics.html``."""
    metrics.validate_metric_groups()
    group_buttons = ['<button type="button" data-group="all" aria-pressed="true">All</button>']
    sections: list[str] = []
    for group_key, group_label, names in metrics.METRIC_GROUPS:
        group_buttons.append(
            f'<button type="button" data-group="{html.escape(group_key)}" '
            f'aria-pressed="false">{html.escape(group_label)}</button>'
        )
        cards: list[str] = []
        for name in names:
            metric = metrics.METRICS[name]
            searchable = " ".join(
                str(metric[key])
                for key in ("label", "question", "definition", "unit", "status")
            ).casefold()
            steps = "".join(f"<li>{html.escape(step)}</li>" for step in metric["steps"])
            cards.append(
                f'<article class="metric-card" id="{html.escape(name)}" '
                f'data-group="{html.escape(group_key)}" data-search="{html.escape(searchable)}">'
                f'<a class="deep-link" href="#{html.escape(name)}" '
                f'aria-label="Copyable link to {html.escape(metric["label"])}">#</a>'
                f'<h3>{html.escape(metric["label"])}</h3>'
                f'<p class="metric-question">{html.escape(metric["question"])}</p>'
                f'<p class="metric-definition">{html.escape(metric["definition"])}</p>'
                '<details><summary>Technical evidence</summary><dl>'
                f'<dt>Unit</dt><dd>{html.escape(metric["unit"])}</dd>'
                f'<dt>Formula</dt><dd><code>{html.escape(metric["formula"])}</code></dd>'
                f'<dt>Example</dt><dd>{html.escape(metric["example"])}</dd>'
                f'<dt>Source</dt><dd><code>{html.escape(metric["source"])}</code></dd>'
                f'<dt>Status</dt><dd>{html.escape(metric["status"])}</dd>'
                f'<dt>Limitation</dt><dd>{html.escape(metric["limitations"])}</dd>'
                f'</dl><p><strong>Calculation</strong></p><ol>{steps}</ol></details></article>'
            )
        sections.append(
            f'<section class="metric-group" id="group-{html.escape(group_key)}" '
            f'data-metric-group="{html.escape(group_key)}"><h2>{html.escape(group_label)}</h2>'
            f'<div class="metric-card-list">{"".join(cards)}</div></section>'
        )
    registry_receipt = trust_receipt(
        claim="Every substantive public chart must name a registered measure and link to its stable definition.",
        population=f"All {len(metrics.METRICS)} governed public metrics",
        method="Central registry grouped by analytical question with one stable HTML anchor per metric",
        uncertainty="Declared per metric; descriptive definitions do not imply a common interval model",
        artifact="metrics.json",
        status="Validated deterministic site projection",
        limitation="A correct definition does not by itself establish construct validity",
        downstream_use="Interpret charts, audit units and denominators, and prevent silent metric drift",
    )
    body = f"""<style>{APPENDIX_CSS}</style>
{methods_local_nav('metric-dictionary')}
<p class="appendix-kicker">Methods appendix · governed measures</p>
<section class="verdict-panel" aria-label="Metric dictionary verdict">
<article><span class="appendix-kicker">30-second answer</span><h2>Every number names its question</h2>
<p>Measures are grouped by the analytical question they answer, not ranked on one false common scale.</p></article>
<article><span class="appendix-kicker">How to read</span><h2>Definition first, formula second</h2>
<p>Each card starts in plain language; unit, calculation, source, status, and limitations remain one disclosure away.</p></article>
<article><span class="appendix-kicker">Audit contract</span><h2>Stable links prevent silent drift</h2>
<p>Substantive figures link to these permanent IDs, while tests require every registered measure to appear once.</p></article></section>
{registry_receipt}
<section class="appendix-section" id="dictionary"><span class="appendix-kicker">Two-minute visual narrative</span>
<h2>Find the measure behind a chart</h2>
<p>Search by wording, analytical question, unit, or status. Filters require JavaScript; all definitions remain visible without it.</p>
<div class="metric-tools"><label class="metric-search"><span>Search the dictionary</span>
<input id="metric-search" type="search" autocomplete="off" placeholder="Try agreement, topics, speeches…"></label>
<div class="metric-filters" aria-label="Filter by analytical question">{"".join(group_buttons)}</div></div>
<p id="metric-results" class="metric-results" role="status" aria-live="polite"></p>
<noscript><p class="note">All measures are shown because filtering is optional.</p></noscript>
<div id="metric-list">{"".join(sections)}</div>
<p id="metric-empty" class="metric-empty" hidden>No measures match that search and group.</p></section>
<section class="appendix-section" id="downloads"><h2>Download the registry</h2>
<ul class="download-links"><li><a href="metrics.json" download>Metric registry · JSON</a></li>
<li><a href="methodology.html#evaluation">Evaluation design</a></li>
<li><a href="label-models.html">Model Comparison</a></li></ul></section>"""
    scripts = r"""<script>
(() => {
  const input = document.getElementById("metric-search");
  const buttons = [...document.querySelectorAll("[data-group]")].filter(x => x.tagName === "BUTTON");
  const cards = [...document.querySelectorAll(".metric-card")];
  const groups = [...document.querySelectorAll("[data-metric-group]")];
  const status = document.getElementById("metric-results");
  const empty = document.getElementById("metric-empty");
  if (!input || !cards.length) return;
  let selected = "all";
  const update = () => {
    const query = input.value.trim().toLocaleLowerCase();
    let visible = 0;
    cards.forEach(card => {
      const showGroup = selected === "all" || card.dataset.group === selected;
      const showSearch = !query || card.dataset.search.includes(query);
      card.hidden = !(showGroup && showSearch);
      if (!card.hidden) visible += 1;
    });
    groups.forEach(group => {
      group.hidden = ![...group.querySelectorAll(".metric-card")].some(card => !card.hidden);
    });
    status.textContent = `${visible} ${visible === 1 ? "measure" : "measures"} shown`;
    empty.hidden = visible !== 0;
  };
  buttons.forEach(button => button.addEventListener("click", () => {
    selected = button.dataset.group;
    buttons.forEach(candidate => candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false"));
    update();
  }));
  input.addEventListener("input", update);
  update();
})();
</script>""" + CURRENT_NAV_SCRIPT
    return body, scripts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_comparison(data_dir: Path = corpus.DATA_DIR) -> dict[str, Any]:
    agreement_path = Path(data_dir) / "method_agreement.parquet"
    compositions_path = Path(data_dir) / "method_compositions.parquet"
    primary_annotations_path = (
        Path(data_dir) / "llm_annotations" / "paragraph_annotations.parquet"
    )
    issues_path = Path(data_dir) / "issues_meta.json"
    taxonomy_path = Path(data_dir) / "llm_annotations" / "taxonomy_v1.json"
    frame = pd.read_parquet(agreement_path)
    required = {"issue", "era", "n", "n_llm", "n_corex", "n_both", "jaccard", "kappa"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"method agreement is missing columns {missing}")
    if frame.duplicated(list(METHOD_AGREEMENT_KEY)).any():
        raise ValueError(
            f"method agreement must be unique on {METHOD_AGREEMENT_KEY}"
        )
    overall = frame[frame["era"].isna()].copy()
    if overall.empty or overall["issue"].duplicated().any():
        raise ValueError("method agreement must contain one overall row per issue")
    counts = overall[["n", "n_llm", "n_corex", "n_both"]]
    count_values = counts.to_numpy(float)
    if (
        not np.isfinite(count_values).all()
        or (count_values < 0).any()
        or not np.equal(count_values, np.floor(count_values)).all()
        or (overall["n"] <= 0).any()
    ):
        raise ValueError("model-comparison counts must be finite nonnegative integers")
    if overall["n"].nunique() != 1:
        raise ValueError("public issues do not share one comparison population")
    if (
        (overall["n_llm"] > overall["n"]).any()
        or (overall["n_corex"] > overall["n"]).any()
        or (overall["n_both"] > overall[["n_llm", "n_corex"]].min(axis=1)).any()
        or (
            overall["n_both"]
            < overall["n_llm"] + overall["n_corex"] - overall["n"]
        ).any()
    ):
        raise ValueError("model-comparison count margins are inconsistent")
    overall["llm_rate"] = overall["n_llm"] / overall["n"]
    overall["corex_rate"] = overall["n_corex"] / overall["n"]
    numeric = overall[["llm_rate", "corex_rate", "jaccard", "kappa"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("model comparison contains non-finite values")
    if not overall[["llm_rate", "corex_rate", "jaccard"]].apply(
        lambda column: column.between(0, 1).all()
    ).all():
        raise ValueError("model comparison rates or overlaps fall outside 0–1")
    compositions = pd.read_parquet(compositions_path)
    composition_required = {*METHOD_COMPOSITIONS_KEY, "share", "n_paragraphs"}
    composition_missing = sorted(composition_required - set(compositions))
    if composition_missing:
        raise ValueError(f"method compositions is missing columns {composition_missing}")
    if compositions.duplicated(list(METHOD_COMPOSITIONS_KEY)).any():
        raise ValueError(
            f"method compositions must be unique on {METHOD_COMPOSITIONS_KEY}"
        )
    issues_meta = json.loads(issues_path.read_text())
    taxonomy = json.loads(taxonomy_path.read_text())
    primary_annotations = pd.read_parquet(
        primary_annotations_path, columns=["doc_name", "para_idx", "topics"]
    )
    if primary_annotations.duplicated(["doc_name", "para_idx"]).any():
        raise ValueError("primary paragraph annotations have duplicate semantic keys")
    label_map = attention.canonical_label_map(taxonomy)
    normalized_topics = primary_annotations["topics"].map(
        lambda value: attention.normalize_topics(value, label_map)
    )
    topic_bearing = int(normalized_topics.map(bool).sum())
    no_topic = int(len(primary_annotations) - topic_bearing)
    n_comparable = int(overall["n"].iloc[0])
    if topic_bearing != n_comparable:
        raise ValueError(
            "model-comparison population must equal normalized primary-LLM topic-bearing paragraphs"
        )
    coherence = issues_meta["coherence"]
    anchored = overall[overall["issue"].isin(coherence)].copy()
    anchored["npmi"] = [float(coherence[name]["npmi"]) for name in anchored["issue"]]
    rho, p_value = spearmanr(anchored["npmi"], anchored["jaccard"])
    if not math.isfinite(float(rho)) or not math.isfinite(float(p_value)):
        raise ValueError("preregistered coherence diagnostic is not estimable")
    counterexample = anchored.sort_values(["npmi", "issue"]).iloc[0]
    return {
        "overall": overall.sort_values(["llm_rate", "issue"]).reset_index(drop=True),
        "issues_meta": issues_meta,
        "taxonomy": taxonomy,
        "prediction": {
            "rho": float(rho),
            "p_value": float(p_value),
            "n_issues": int(len(anchored)),
            "counterexample": str(counterexample["issue"]),
            "counterexample_jaccard": float(counterexample["jaccard"]),
        },
        "hashes": {
            "method_agreement.parquet": _sha256(agreement_path),
            "method_compositions.parquet": _sha256(compositions_path),
            "paragraph_annotations.parquet": _sha256(primary_annotations_path),
            "issues_meta.json": _sha256(issues_path),
            "taxonomy_v1.json": _sha256(taxonomy_path),
        },
        "public_schema": {
            "schema_version": MODEL_COMPARISON_PUBLIC_SCHEMA,
            "method-agreement.csv": {
                "semantic_key": list(METHOD_AGREEMENT_KEY),
                "columns": list(frame.columns),
            },
            "method-compositions.csv": {
                "semantic_key": list(METHOD_COMPOSITIONS_KEY),
                "columns": list(compositions.columns),
            },
        },
        "population": {
            "source_paragraphs": int(len(primary_annotations)),
            "topic_bearing_paragraphs": topic_bearing,
            "no_topic_paragraphs_excluded": no_topic,
            "exclusion_rule": "drop_unlabeled=True in the preregistered comparison estimand",
        },
    }


def _rate_rows(overall: pd.DataFrame) -> str:
    rows = []
    for row in overall.itertuples(index=False):
        llm = float(row.llm_rate) * 100
        corex = float(row.corex_rate) * 100
        rows.append(
            '<li class="rate-row">'
            f'<h3>{html.escape(str(row.issue))}</h3><div class="rate-pair">'
            f'<div class="rate-line llm"><span>LLM projection</span><span class="track" aria-hidden="true"><span class="fill" style="--value:{llm:.6f}"></span></span><strong>{llm:.1f}%</strong></div>'
            f'<div class="rate-line corex"><span>CorEx</span><span class="track" aria-hidden="true"><span class="fill" style="--value:{corex:.6f}"></span></span><strong>{corex:.1f}%</strong></div>'
            f'</div></li>'
        )
    return "".join(rows)


def _agreement_rows(overall: pd.DataFrame) -> str:
    rows = []
    ranked = overall.sort_values(["kappa", "issue"], ascending=[False, True])
    for row in ranked.itertuples(index=False):
        rows.append(
            '<li class="agreement-row">'
            f'<h3>{html.escape(str(row.issue))}</h3><div class="agreement-values">'
            f'<div><span>Jaccard</span><strong>{float(row.jaccard):.3f}</strong></div>'
            f'<div><span>Cohen’s κ</span><strong>{float(row.kappa):.3f}</strong></div></div>'
            f'<small>{int(row.n_both):,} shared positive paragraphs · {int(row.n):,} topic-bearing comparison paragraphs</small></li>'
        )
    return "".join(rows)


def render_model_comparison(
    data_dir: Path = corpus.DATA_DIR,
) -> tuple[str, str]:
    evidence = load_model_comparison(data_dir)
    overall = evidence["overall"]
    prediction = evidence["prediction"]
    issues_meta = evidence["issues_meta"]
    taxonomy = evidence["taxonomy"]
    public_schema = evidence["public_schema"]
    comparison_population = evidence["population"]
    if overall["n"].nunique() != 1:
        raise ValueError("public issues do not share one comparison population")
    n_comparable = int(overall["n"].iloc[0])
    n_public_issues = int(len(overall))
    n_anchored = int(len(issues_meta["issues"]))
    n_domains = len(taxonomy["level1"])
    n_topics = len(taxonomy["level2"])
    anchor_rows = "".join(
        f'<li><strong>{html.escape(issue)}</strong><span>{html.escape(", ".join(words))}</span></li>'
        for issue, words in ISSUE_ANCHORS.items()
    )
    hash_cards = "".join(
        f'<article class="receipt-card"><span>{"Frozen paid input" if name == "taxonomy_v1.json" else "Deterministic input"}</span>'
        f'<h3>{html.escape(name)}</h3><p><code>sha256:{value}</code></p></article>'
        for name, value in evidence["hashes"].items()
    )
    receipt = trust_receipt(
        claim="The two instruments often differ without either difference being a known error.",
        population=(
            f"{n_comparable:,} primary-LLM topic-bearing paragraphs; "
            f"{comparison_population['no_topic_paragraphs_excluded']:,} no-topic paragraphs excluded"
        ),
        method=f"{n_public_issues} issue-wise prevalence, Jaccard, and Cohen’s kappa comparisons",
        uncertainty="Point-estimate comparison; per-issue support is visible",
        artifact="data/method-agreement.csv",
        status="Cross-instrument descriptive audit",
        limitation="Neither instrument is human ground truth and the taxonomies are not identical",
        downstream_use="Choose a lens, qualify model-sensitive findings, and inspect disagreements",
    )
    body = f"""<style>{APPENDIX_CSS}</style>
{methods_local_nav('model-comparison')}
<p class="appendix-kicker">Methods appendix · model comparison</p>
<section class="verdict-panel" aria-label="Model comparison verdict">
<article><span class="appendix-kicker">30-second answer</span><h2>These are different instruments</h2>
<p>CorEx follows inspectable word patterns. The LLM applies a frozen semantic rubric. Agreement supports robustness; mismatch identifies sensitivity.</p></article>
<article><span class="appendix-kicker">Shared population</span><h2>{n_comparable:,} topic-bearing paragraphs</h2>
<p>The preregistered <code>drop_unlabeled=True</code> estimand excludes
{comparison_population['no_topic_paragraphs_excluded']:,} primary-LLM no-topic paragraphs from
{comparison_population['source_paragraphs']:,} source paragraphs. Neither model is treated as a gold label.</p></article>
<article><span class="appendix-kicker">Scientific result</span><h2>A prediction failed</h2>
<p>Anchor-word coherence did not predict cross-method Jaccard. The failed preregistered expectation remains prominent.</p></article></section>
<section class="failed-prediction" aria-labelledby="failed-title"><p class="appendix-kicker">Preregistered result · failed prediction</p>
<h2 id="failed-title">Cleaner anchor co-occurrence did not imply stronger cross-method agreement</h2>
<p class="result">Spearman ρ = {prediction['rho']:+.3f} · two-sided p = {prediction['p_value']:.3f}</p>
<p>Across {prediction['n_issues']} anchored issues, the relationship is essentially absent.
{html.escape(prediction['counterexample'])} supplies the sharpest counterexample: it has the lowest
anchor coherence yet a Jaccard overlap of {prediction['counterexample_jaccard']:.3f}. Publishing this
failure is evidence of a testable workflow—not evidence that either labeling system is true.</p></section>
<section class="appendix-section" id="instruments"><h2>The two-minute visual explanation</h2>
<div class="instrument-grid"><article class="instrument-card"><p class="eyebrow">CorEx · deterministic local model</p>
<h2>{n_anchored} anchored issue factors</h2><ul><li>Binary word-presence features from the complete paragraph corpus.</li>
<li>Declared era-spanning anchors guide correlated-word factors.</li><li>Strength: repeatable vocabulary evidence.</li>
<li>Blind spot: semantic continuity when historical wording changes.</li></ul></article>
<article class="instrument-card"><p class="eyebrow">LLM · frozen paid judgments</p><h2>{n_topics} topics in {n_domains} domains</h2>
<ul><li>Corpus-derived taxonomy frozen before the complete paragraph pass.</li><li>Text and decade visible; identity and title masked for paragraph judgments.</li>
<li>Strength: meaning beyond exact anchor words.</li><li>Blind spot: rubric interpretation and shared model-family bias.</li></ul></article></div>
<p><strong>Named entities are not a CorEx output.</strong> Entity detection and stance come from a separate LLM pass and are evaluated as model-to-model reproducibility on Data Quality.</p></section>
<section class="appendix-section" id="rates"><h2>How often each instrument activates an issue</h2>
<p>Paired bars share one percentage scale and the same denominator: the {n_comparable:,}
primary-LLM topic-bearing paragraphs retained by the preregistered comparison estimand. The
{comparison_population['no_topic_paragraphs_excluded']:,} no-topic paragraphs are excluded.
Patterns distinguish the instruments; they do not decide which instrument is correct.</p>
<figure class="comparison-figure" data-substantive-chart data-metric="paragraph_share"
data-evidence="data/method-agreement.csv" aria-labelledby="rates-title"><figcaption id="rates-title">
<strong>Positive paragraph share by public issue</strong><span>Exact values are printed beside every server-rendered bar; JavaScript is not required. <a href="metrics.html#paragraph_share">Define paragraph share.</a></span></figcaption>
<ol>{_rate_rows(overall)}</ol></figure></section>
<section class="appendix-section" id="agreement"><h2>Overlap and chance-adjusted agreement answer different questions</h2>
<p>Jaccard asks how much of the positive union overlaps. κ uses the complete yes/no table and corrects for expected agreement from each instrument’s prevalence.</p>
<figure class="comparison-figure" data-substantive-chart data-metric="jaccard kappa"
data-evidence="data/method-agreement.csv" aria-labelledby="agreement-title"><figcaption id="agreement-title">
<strong>Agreement by metric family</strong><span><a href="metrics.html#jaccard">Define Jaccard.</a> <a href="metrics.html#kappa">Define Cohen’s κ.</a> Support stays visible because rare issues can be unstable.</span></figcaption>
<ol class="agreement-list">{_agreement_rows(overall)}</ol></figure>{receipt}</section>
<section class="appendix-section" id="technical-evidence"><h2>Technical evidence and reproducibility</h2>
<div class="receipt-grid">{hash_cards}</div><div class="receipt-grid">
<article class="receipt-card"><span>Output · schema identity</span><h3>Public comparison tables</h3>
<p><code>{html.escape(str(public_schema['schema_version']))}</code><br>
<code>docs/data/method-agreement.csv</code> · semantic key <code>(issue, era)</code><br>
<code>docs/data/method-compositions.csv</code> · semantic key <code>(doc_name, method, topic)</code></p>
<p>Both CSVs are deterministic projections of the hash-identified Parquet inputs above.</p></article>
<article class="receipt-card"><span>Command</span><h3>Deterministic site projection</h3><p><code>arch -x86_64 .venv/bin/python -m presidential_profiles.site</code></p></article>
<article class="receipt-card"><span>Tests</span><h3>Schema, values, rendering</h3><p><code>tests/test_trust_appendices.py</code><br><code>tests/test_ai_labels_site.py</code></p></article></div>
<details><summary>Inspect the exact CorEx anchor vocabulary</summary><p>Anchors guide factors; they are not a keyword OR rule. CorEx can learn correlated non-anchor words.</p>
<ul class="anchor-list">{anchor_rows}</ul></details></section>
<section class="appendix-section" id="downloads"><h2>Downloads and next checks</h2><ul class="download-links">
<li><a href="data/method-agreement.csv" download>Issue agreement · CSV</a></li>
<li><a href="data/method-compositions.csv" download>Speech compositions · CSV</a></li>
<li><a href="data-quality.html">Data Quality audit</a></li><li><a href="methodology.html#evaluation">Evaluation design</a></li>
<li><a href="metrics.html">Metric Dictionary</a></li></ul></section>"""
    return body, CURRENT_NAV_SCRIPT
