"""Render the reader-facing article explaining the AI annotation pipeline."""

from __future__ import annotations

import html
import json

from .figures import REPO_ROOT
from .site_style import PAGE_CSS


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _agreement(ai_data: dict, key: str) -> tuple[float | None, int | None]:
    rec = ai_data["agreement"].get(key)
    return (rec["value"], rec["n"]) if rec else (None, None)


def dashboard_summary(ai_data: dict) -> str:
    """Compact methodology bridge for the main dashboard."""
    s = ai_data["summary"]
    jaccard, n = _agreement(ai_data, "topics:jaccard")
    agreement = f"{_pct(jaccard)} mean topic overlap on {n:,} re-read paragraphs" \
        if jaccard is not None else "a separately measured second-opinion pass"
    return f"""<div class="ai-bridge">
  <div class="ai-bridge-stat"><strong>{s['n_paragraphs']:,}</strong><span>paragraphs labeled</span></div>
  <div class="ai-bridge-stat"><strong>{s['n_topics']}</strong><span>corpus-derived topics</span></div>
  <div class="ai-bridge-stat"><strong>{s['n_domains']}</strong><span>broad domains</span></div>
  <div class="ai-bridge-stat"><strong>{s['mean_topics']:.3f}</strong><span>topics per paragraph</span></div>
  <p>Names and definitions were derived from era-isolated samples of this corpus, then
  frozen before the full pass. The primary model saw only paragraph text and decade;
  president, party, and title were hidden. Reliability is reported, not assumed:
  {agreement}. <a href="methodology.html">Read the illustrated methodology →</a></p>
</div>"""


def _pipeline_svg() -> str:
    return """<svg class="diagram" viewBox="0 0 1160 430" role="img"
 aria-labelledby="pipeline-title pipeline-desc">
<title id="pipeline-title">From speeches to AI labels</title>
<desc id="pipeline-desc">A seven-stage flow from the Miller Center corpus through paragraph
segmentation, unsupervised clusters, era-isolated topic proposals, a frozen taxonomy,
masked full-corpus annotation, and a second-opinion audit.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7"
 markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#6b7280"/></marker></defs>
<g fill="none" stroke="#6b7280" stroke-width="2" marker-end="url(#arrow)">
 <path d="M150 105H205"/><path d="M350 105H405"/><path d="M550 105H605"/>
 <path d="M750 105H805"/><path d="M950 105H1005"/>
 <path d="M1080 165V245H885"/><path d="M805 315H750"/>
</g>
<g class="node"><rect x="20" y="45" width="130" height="120" rx="18"/>
 <text x="85" y="82">1,057 speeches</text><text x="85" y="108">4.2M words</text>
 <text class="small" x="85" y="138">source corpus</text></g>
<g class="node"><rect x="205" y="45" width="145" height="120" rx="18"/>
 <text x="278" y="82">36,229</text><text x="278" y="108">paragraphs</text>
 <text class="small" x="278" y="138">stable keyed units</text></g>
<g class="node"><rect x="405" y="45" width="145" height="120" rx="18"/>
 <text x="478" y="82">Embeddings</text><text x="478" y="108">+ k=40</text>
 <text class="small" x="478" y="138">no topic names yet</text></g>
<g class="node accent"><rect x="605" y="45" width="145" height="120" rx="18"/>
 <text x="678" y="82">9 isolated</text><text x="678" y="108">era proposals</text>
 <text class="small" x="678" y="138">anachronism guard</text></g>
<g class="node accent"><rect x="805" y="45" width="145" height="120" rx="18"/>
 <text x="878" y="82">17 domains</text><text x="878" y="108">50 topics</text>
 <text class="small" x="878" y="138">frozen taxonomy</text></g>
<g class="node"><rect x="1005" y="45" width="135" height="120" rx="18"/>
 <text x="1072" y="82">Held-out</text><text x="1072" y="108">coverage</text>
 <text class="small" x="1072" y="138">200 paragraphs</text></g>
<g class="node dark"><rect x="805" y="245" width="270" height="140" rx="20"/>
 <text x="940" y="285">Masked full-corpus pass</text><text x="940" y="315">topics · flags · values · entities</text>
 <text class="small" x="940" y="348">text + decade only</text></g>
<g class="node"><rect x="550" y="245" width="200" height="140" rx="20"/>
 <text x="650" y="285">Second opinion</text><text x="650" y="315">same prompts</text>
 <text class="small" x="650" y="348">different model · 25% sample</text></g>
</svg>"""


def _masking_svg() -> str:
    return """<svg class="diagram" viewBox="0 0 1160 420" role="img"
 aria-labelledby="mask-title mask-desc">
<title id="mask-title">Two passes use different evidence</title>
<desc id="mask-desc">The judgment pass hides president, party, and title while showing
paragraph text and decade. The factual speech pass shows title, year, and opening paragraphs.</desc>
<g class="lane"><rect x="25" y="35" width="535" height="340" rx="24"/>
 <text class="lane-title" x="60" y="82">Paragraph judgment · masked</text>
 <g class="pill on"><rect x="60" y="112" width="205" height="48" rx="24"/><text x="162" y="142">paragraph text</text></g>
 <g class="pill on"><rect x="285" y="112" width="205" height="48" rx="24"/><text x="387" y="142">speech decade</text></g>
 <g class="pill off"><rect x="60" y="180" width="205" height="48" rx="24"/><text x="162" y="210">president hidden</text></g>
 <g class="pill off"><rect x="285" y="180" width="205" height="48" rx="24"/><text x="387" y="210">party hidden</text></g>
 <g class="pill off"><rect x="60" y="248" width="430" height="48" rx="24"/><text x="275" y="278">speech title hidden</text></g>
 <text class="caption" x="60" y="334">Used for topics, combat flags, proposal/values, entities</text></g>
<g class="lane"><rect x="600" y="35" width="535" height="340" rx="24"/>
 <text class="lane-title" x="635" y="82">Speech facts · unmasked</text>
 <g class="pill on"><rect x="635" y="112" width="205" height="48" rx="24"/><text x="737" y="142">speech title</text></g>
 <g class="pill on"><rect x="860" y="112" width="205" height="48" rx="24"/><text x="962" y="142">year</text></g>
 <g class="pill on"><rect x="635" y="180" width="430" height="48" rx="24"/><text x="850" y="210">first five paragraphs</text></g>
 <text class="caption" x="635" y="274">Used for speech type, audience, and medium</text>
 <text class="caption" x="635" y="310">The title is evidence here—not a source of reputational bias.</text></g>
</svg>"""


def _taxonomy_atlas(ai_data: dict) -> str:
    taxonomy = ai_data["taxonomy"]
    by_domain: dict[str, list[dict]] = {}
    for topic in taxonomy["level2"]:
        by_domain.setdefault(topic["level1"], []).append(topic)
    kinds = {d["name"]: d["kind"] for d in taxonomy["level1"]}
    blocks = []
    for domain in taxonomy["level1"]:
        name = domain["name"]
        topics = by_domain.get(name, [])
        rows = "".join(
            f"""<li><strong>{html.escape(t['name'])}</strong>
  <span>{html.escape(t['definition'])}</span></li>""" for t in topics
        )
        blocks.append(f"""<details class="tax-domain">
<summary><span>{html.escape(name)}</span><span class="tax-kind">{kinds[name]} · {len(topics)} topics</span></summary>
<p>{html.escape(domain['definition'])}</p><ul>{rows}</ul></details>""")
    return "\n".join(blocks)


def _rubric_cards() -> str:
    cards = [
        ("topics", "Zero or more names from the frozen 50-topic list. No label is forced when none fits; several may apply."),
        ("party_attack", "True only when a partisan actor is identifiable and the text imputes fault, failure, or bad motive."),
        ("enemy_naming", "True only when a specific person, group, nation, or institution is framed as an adversary or threat."),
        ("zero_sum", "True only when the paragraph itself frames gains as requiring another side's loss—not merely when conflict is mentioned."),
        ("proposal", "A concrete policy, program, law, appropriation, or course of action is urged."),
        ("values", "Principles, identity, faith, gratitude, or civic virtue appear without a concrete policy ask."),
        ("mixed", "A substantial concrete ask and a substantial values appeal both appear."),
        ("neither", "Narrative, procedure, ceremony, factual reporting, or rebuttal that fits neither side."),
        ("entities", "Specific actors are extracted with person/group/nation/institution/other and adversarial/favorable/neutral stance."),
    ]
    return "".join(
        f'<div class="rubric"><code>{html.escape(name)}</code><p>{html.escape(text)}</p></div>'
        for name, text in cards
    )


def render_methodology(ai_data: dict) -> str:
    s = ai_data["summary"]
    tax = ai_data["manifests"]["taxonomy"]
    primary = ai_data["manifests"]["primary"]
    secondary = ai_data["manifests"]["secondary"]
    topics_j, topics_n = _agreement(ai_data, "topics:jaccard")
    party_k, flag_n = _agreement(ai_data, "party_attack:cohen_kappa")
    enemy_k, _ = _agreement(ai_data, "enemy_naming:cohen_kappa")
    zero_k, _ = _agreement(ai_data, "zero_sum:cohen_kappa")
    proposal_x, _ = _agreement(ai_data, "proposal_values:exact_match")
    speech_k, speech_n = _agreement(ai_data, "speech_type:cohen_kappa")
    entity_s, entity_n = _agreement(ai_data, "entities:stance_agreement")
    agreement_cards = [
        ("Topic sets", _pct(topics_j), "mean per-paragraph Jaccard", topics_n),
        ("Partisan attack", f"κ {party_k:.2f}", "chance-corrected agreement", flag_n),
        ("Named adversary", f"κ {enemy_k:.2f}", "chance-corrected agreement", flag_n),
        ("Zero-sum framing", f"κ {zero_k:.2f}", "chance-corrected agreement", flag_n),
        ("Proposal / values", _pct(proposal_x), "exact match", flag_n),
        ("Speech type", f"κ {speech_k:.2f}", "chance-corrected agreement", speech_n),
        ("Entity stance", _pct(entity_s), "agreement where names matched", entity_n),
    ]
    agreement_html = "".join(
        f"""<div class="metric"><span>{html.escape(label)}</span><strong>{value}</strong>
<small>{html.escape(note)} · n={n:,}</small></div>"""
        for label, value, note, n in agreement_cards
    )
    agreement_chart = [
        {"label": "Topic sets", "value": topics_j, "measure": "mean Jaccard"},
        {"label": "Partisan attack", "value": party_k, "measure": "Cohen’s κ"},
        {"label": "Named adversary", "value": enemy_k, "measure": "Cohen’s κ"},
        {"label": "Zero-sum", "value": zero_k, "measure": "Cohen’s κ"},
        {"label": "Proposal / values", "value": proposal_x, "measure": "exact match"},
        {"label": "Speech type", "value": speech_k, "measure": "Cohen’s κ"},
        {"label": "Entity stance", "value": entity_s, "measure": "matched-name agreement"},
    ]

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How the AI labels were made - Presidential Profiles</title>
<meta name="description" content="Illustrated methodology for the corpus-derived taxonomy and AI annotation layer behind Presidential Profiles.">
<style>
{PAGE_CSS}
  header, main, footer {{ max-width: 1040px; }}
  header {{ padding-top: 38px; }}
  .crumbs {{ margin-bottom: 22px; font-size: .88rem; }} .crumbs a {{ color:var(--ink2); }}
  .kicker {{ color:#275d8c; font-weight:750; letter-spacing:.08em; text-transform:uppercase; font-size:.76rem; }}
  header h1 {{ font-size:clamp(2.2rem,6vw,4.5rem); line-height:1.02; max-width:850px; margin-top:12px; }}
  header .sub {{ font-size:1.08rem; max-width:760px; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-top:28px; }}
  .stat {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:16px; }}
  .stat strong {{ display:block; font-size:1.65rem; }} .stat span {{ color:var(--muted); font-size:.82rem; }}
  article {{ max-width:760px; }} article h2 {{ font-size:1.7rem; margin-top:58px; letter-spacing:-.02em; }}
  article h3 {{ font-size:1.12rem; margin-top:30px; }} article p, article li {{ color:var(--ink2); }}
  article p {{ margin-top:10px; }} article ul, article ol {{ padding-left:22px; margin-top:12px; }}
  .wide {{ width:min(1040px,calc(100vw - 40px)); margin-top:24px; }}
  .diagram {{ display:block; width:100%; height:auto; background:var(--surface); border:1px solid var(--border); border-radius:18px; }}
  .diagram .node rect,.diagram .lane>rect {{ fill:#fff; stroke:#d7d9dd; stroke-width:2; }}
  .diagram .node.accent rect {{ fill:#eef6ff; stroke:#8bb9e8; }} .diagram .node.dark rect {{ fill:#17202a; stroke:#17202a; }}
  .diagram text {{ font-family:system-ui,sans-serif; font-size:16px; font-weight:700; text-anchor:middle; fill:#17202a; }}
  .diagram .small {{ font-size:12px; font-weight:500; fill:#6b7280; }} .diagram .dark text {{ fill:#fff; }}
  .diagram .dark .small {{ fill:#d1d5db; }} .diagram .lane-title {{ text-anchor:start; font-size:19px; }}
  .diagram .pill rect {{ stroke-width:1.5; }} .diagram .pill.on rect {{ fill:#eef8f2; stroke:#70aa85; }}
  .diagram .pill.off rect {{ fill:#f8f1f1; stroke:#bc8b8b; stroke-dasharray:6 5; }}
  .diagram .pill text {{ font-size:14px; }} .diagram .caption {{ text-anchor:start; font-size:13px; font-weight:500; fill:#6b7280; }}
  .callout {{ border-left:4px solid #6da7ec; background:#f1f6fc; border-radius:0 12px 12px 0; padding:16px 18px; margin:20px 0; }}
  .callout p {{ margin:0; }} .rubrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; margin-top:18px; }}
  .rubric {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px; }}
  .rubric code {{ font-weight:750; color:#275d8c; }} .rubric p {{ font-size:.88rem; margin-top:7px; }}
  .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; margin-top:18px; }}
  .metric {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px; }}
  .metric span,.metric small {{ display:block; color:var(--muted); font-size:.78rem; }} .metric strong {{ font-size:1.35rem; display:block; margin:4px 0; }}
  .tax-atlas {{ margin-top:18px; }} .tax-domain {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; margin:8px 0; padding:0 16px; }}
  .tax-domain summary {{ cursor:pointer; display:flex; justify-content:space-between; gap:20px; padding:15px 0; font-weight:700; }}
  .tax-kind {{ color:var(--muted); font-size:.76rem; font-weight:500; }} .tax-domain>p {{ margin:0 0 12px; font-size:.9rem; }}
  .tax-domain ul {{ margin:0 0 18px; }} .tax-domain li {{ margin:10px 0; }} .tax-domain li span {{ display:block; font-size:.86rem; }}
  .method-table {{ width:100%; border-collapse:collapse; margin-top:16px; background:var(--surface); }}
  .method-table th,.method-table td {{ border:1px solid var(--border); padding:10px; text-align:left; font-size:.86rem; vertical-align:top; }}
  .method-table th {{ color:var(--muted); }}
  .method-chart {{ width:min(1040px,calc(100vw - 40px));height:430px;margin:20px 0; }}
  @media(max-width:680px){{ .diagram text {{ font-size:12px; }} .diagram .small,.diagram .caption {{ font-size:9px; }} .diagram .lane-title {{ font-size:14px; }} }}
</style></head><body>
<header><p class="crumbs"><a href="index.html">← Dashboard</a> &nbsp;·&nbsp;
<a href="compare.html">Compare presidents</a> &nbsp;·&nbsp; <a href="presidents/index.html">Profiles</a></p>
<div class="kicker">Methods article · AI label layer</div><h1>How a speech becomes a label</h1>
<p class="sub">The topic names were discovered from this corpus before the full annotation
pass. The model did not receive a president's identity when making subjective judgments.
Every label is keyed back to its paragraph, and a second model measures where the first one
is uncertain.</p><div class="stat-grid">
<div class="stat"><strong>{s['n_paragraphs']:,}</strong><span>paragraphs, 100% covered</span></div>
<div class="stat"><strong>{s['n_topics']}</strong><span>fine-grained topics</span></div>
<div class="stat"><strong>{s['n_domains']}</strong><span>broad domains</span></div>
<div class="stat"><strong>{s['n_assignments']:,}</strong><span>distinct topic assignments</span></div>
<div class="stat"><strong>{s['n_entities']:,}</strong><span>named-entity rows</span></div>
</div><div id="method-coverage" class="method-chart"
aria-label="Primary and second-opinion paragraph coverage"></div></header><main><article>
<h2>First: what “AI-labeled” means here</h2>
<p>There are two measurement layers on this website. The original issue charts use a seeded,
deterministic CorEx topic model with hand-written anchor words. The newer layer uses a
corpus-derived taxonomy and Claude to read paragraphs semantically. The old layer remains
visible as an independent instrument; it is not relabeled as AI work after the fact.
<a href="label-models.html"><strong>See the complete CorEx derivation and the
side-by-side model comparison →</strong></a></p>
<div class="callout"><p><strong>The unit is the paragraph.</strong> A speech can cover several
subjects, switch from policy to ceremony, or name an adversary once. Paragraph-sized units keep
those changes visible. All joins use <code>(doc_name, para_idx)</code>, never row position.</p></div>
</article><div class="wide">{_pipeline_svg()}</div><article>
<h2>How the 50 topics were arrived at</h2>
<ol><li><strong>Find the blind spots.</strong> The 15 anchored issues left 10,381 of 36,229
paragraphs without a match. Those missing paragraphs were deliberately oversampled.</li>
<li><strong>Let language cluster before naming it.</strong> Every paragraph was embedded with
<code>minishlab/potion-base-8M</code>, normalized, and clustered with seeded MiniBatchKMeans at
k=40. c-TF-IDF terms described what distinguished each cluster; the clusters were evidence,
not final topics.</li><li><strong>Isolate eras.</strong> Nine 30-year samples were drawn with
80 paragraphs per era and roughly half from the old taxonomy's blind mass. Each era proposal
saw only its own paragraphs, dominant clusters, and terms. The 1800s proposal could not borrow
“health care” from a modern sample.</li><li><strong>Merge once, then freeze.</strong> Opus merged
the era proposals into 17 domains and 50 period-aware topics, then produced a many-to-many
crosswalk to the 16 existing issue pages.</li><li><strong>Test what was not used to build it.</strong>
A held-out set of 200 paragraphs—100 from the former blind mass—was labeled against the frozen
list. Coverage reached 100%, above the pre-registered 90% gate. The taxonomy was then frozen
before the 36,229-paragraph pass.</li></ol>
<p>The taxonomy-generation run used {html.escape(', '.join(tax['models']))}, {tax['requests']} API
requests, and recorded ${tax['cost_usd']:.4f} in actual usage-derived cost.</p>
</article><div class="wide">{_masking_svg()}</div><article>
<h2>What the annotator saw—and what it did not</h2>
<p>The subjective paragraph pass received the paragraph text and its decade. President, party,
and speech title were withheld. The decade allowed the rubric to judge combativeness against the
speech's own era instead of silently imposing 2026 norms.</p><p>A second factual pass used the
opposite policy: title, year, and the first five paragraphs were shown because those are direct
evidence for speech type, audience, and medium. Masking a title like “Fourth Annual Message”
would remove the best factual clue without reducing reputational bias.</p>
<h3>The closed rubric</h3><p>Structured output allowed only the following fields and closed
enum values. Unknown topic strings could not be invented by the model.</p>
<div class="rubrics">{_rubric_cards()}</div>
<h2>How every paragraph was labeled</h2>
<p>Claude Sonnet 5 annotated all {s['n_paragraphs']:,} paragraphs. Each request carried an exact
count contract and the paragraph keys it had to return. Some long structured arrays stopped
early, so incomplete speeches were never marked done: they were retried through seven rounds
with smaller chunks until paragraph and speech coverage both reached 100%. Thirty-two speeches
needed partial-speech chunk context; that amendment is recorded rather than hidden.</p>
<p>The final table contains {s['n_assignments']:,} distinct paragraph-topic pairs, or
{s['mean_topics']:.3f} topics per paragraph. Labels are multi-label, so shares across topics do
not sum to 100%. The denominator always includes the {s['empty_topics']:,} paragraphs
({s['empty_share']:.2f}%) assigned no topic. Case variants are mapped to the canonical 50 names,
and duplicate paragraph-topic pairs are removed before counting.</p>
<h2>Quality checks and the second opinion</h2>
<p>Coverage alone is not accuracy. A persisted, era-stratified sample of 266 speeches was read
again by Claude Opus 4.8 using byte-identical prompts. The model was the only intended variable.
Agreement is published field by field; no disagreement score is used to erase labels.</p>
<div class="metrics">{agreement_html}</div>
<div id="method-agreement" class="method-chart" aria-label="Agreement audit measures"></div>
<div class="callout"><p><strong>These bars are not one league table.</strong> Topic-set
Jaccard, binary-label κ, exact match, and matched-name stance agreement answer different
questions. The shared 0–1 axis makes their scale visible; each tooltip retains the metric
type.</p></div>
<p>Those figures are reliability evidence, not human ground truth. Topic Jaccard is set overlap
for a multi-label field; Cohen's kappa corrects categorical agreement for chance; exact match is
used for the four-way proposal/values field. Entity-name matching is particularly strict because
two models may choose different spans for the same actor; stance agreement is reported only on
matched names.</p>
<h2>How uncertainty reaches the charts</h2>
<p>Trend bands resample whole speeches, not individual paragraphs, because paragraphs inside one
speech are correlated. AI-topic trend bands combine that sampling interval with the measured
half-gap between the primary and second-opinion labelers in the same reporting era. Thin periods
are dotted; cells whose interval cannot be resolved receive a hollow ring. A point estimate may
still be real while its uncertainty is unknown.</p>
<h2>Where the labels appear</h2><ul><li><strong>Dashboard:</strong> AI topic trends carry sampling
and annotator-disagreement bands.</li><li><strong>President profiles:</strong> top topics, distinctive
topics, proposal/values mix, combat flags, genres, and named adversaries.</li><li><strong>Comparison:</strong>
the same rates are aligned across two or three presidents.</li><li><strong>Issue pages:</strong> each
broad legacy issue shows the finer AI topics in its crosswalk.</li><li><strong>Explorer:</strong> all
50 AI topics can be overlaid against any word or phrase.</li></ul>
<h2>Limitations that remain visible</h2><ul><li>The Miller Center collection is curated and not
exhaustive; this is presidential formal speech, not every rally, interview, tweet, or private
conversation.</li><li>The taxonomy is frozen for comparability and therefore retains known gaps,
including no standalone education topic and no modern monetary-policy topic.</li><li>Era tags reduce
anachronism; they do not eliminate every historically implausible assignment.</li><li>Inter-model
agreement is not human validation. Zero-sum framing is the least stable binary flag here
(κ {zero_k:.2f}), so it should be read with more caution than speech type (κ {speech_k:.2f}).</li>
<li>President-level percentages are descriptive. Thin records are marked rather than promoted as
precise estimates.</li></ul>
<h2>The complete frozen taxonomy</h2><p>Open any domain to inspect its definition and every
level-2 topic available to the annotator. Policy and non-policy domains are both first class.</p>
<div class="tax-atlas">{_taxonomy_atlas(ai_data)}</div>
<h2>Provenance</h2><table class="method-table"><thead><tr><th>Stage</th><th>Models</th><th>Requests</th><th>Actual recorded cost</th></tr></thead><tbody>
<tr><td>Taxonomy + crosswalk</td><td>{html.escape(', '.join(tax['models']))}</td><td>{tax['requests']:,}</td><td>${tax['cost_usd']:.4f}</td></tr>
<tr><td>Primary full annotation and convergence</td><td>{html.escape(', '.join(primary['models']))}</td><td>{primary['requests']:,}</td><td>${primary['cost_usd']:.4f}</td></tr>
<tr><td>Second-opinion sample</td><td>{html.escape(', '.join(secondary['models']))}</td><td>{secondary['requests']:,}</td><td>${secondary['cost_usd']:.4f}</td></tr>
</tbody></table><p>Every output row carries a <code>run_id</code> that points to a manifest with
model, prompt version and hash, batch ID, tokens, cost, and corpus fingerprint. The frozen paid
artifacts are read-only inputs to this site build.</p>
</article></main><footer><p>Data: <a href="https://data.millercenter.org">Miller Center of Public
Affairs, University of Virginia</a>. Source code and full research notes:
<a href="https://github.com/jacobfulfyll/presidential_profiles">Presidential Profiles</a>.</p></footer>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script><script>
const MA={json.dumps(agreement_chart)};
const methodBase={{template:"simple_white",paper_bgcolor:"#fff",plot_bgcolor:"#fff",
 font:{{family:"system-ui",color:"#24313d"}},margin:{{l:64,r:24,t:38,b:80}}}};
Plotly.newPlot("method-coverage",[{{type:"bar",x:["Primary label pass","Paired second opinion"],
 y:[{s['n_paragraphs']},{flag_n}],marker:{{color:["#2d6f9d","#b17a43"]}},
 text:[{s['n_paragraphs']},{flag_n}],textposition:"outside",
 hovertemplate:"%{{x}}<br>%{{y:,}} paragraphs<extra></extra>"}}],
 {{...methodBase,height:430,yaxis:{{title:"paragraphs",gridcolor:"#e7e9eb"}},
 title:{{text:"How much of the corpus each judgment pass covers",font:{{size:16}}}}}},
 {{displayModeBar:false,responsive:true}});
Plotly.newPlot("method-agreement",[{{type:"bar",x:MA.map(x=>x.label),y:MA.map(x=>x.value),
 customdata:MA.map(x=>x.measure),
 marker:{{color:MA.map(x=>x.label==="Zero-sum"?"#b17a43":"#2d6f9d")}},
 hovertemplate:"%{{x}}<br>%{{customdata}}: %{{y:.3f}}<extra></extra>"}}],
 {{...methodBase,height:430,yaxis:{{title:"reported agreement (0–1)",range:[0,1],gridcolor:"#e7e9eb"}},
 title:{{text:"Second-opinion reliability audit",font:{{size:16}}}}}},
 {{displayModeBar:false,responsive:true}});
</script>
</body></html>"""


def write_methodology(ai_data: dict) -> None:
    path = REPO_ROOT / "docs" / "methodology.html"
    path.write_text(render_methodology(ai_data))
    print("  wrote docs/methodology.html")
