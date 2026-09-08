"""Generated metric lessons, quality audit, model-comparison and feedback pages."""
from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from . import (
    corpus,
    metrics,
    portraits,
    quality_audit,
    trust_appendices,
    validation_protocols,
)
from .site_style import PAGE_CSS

NAV = [
    ("Story", "index.html"),
    ("Summary", "summary.html"),
    ("Compare", "compare.html"),
    ("Explore", "explorer.html"),
    ("Profiles", (
        ("Presidents", "presidents/index.html"),
        ("Issues", "issues/index.html"),
    )),
    ("Data", (
        ("Data Quality", "data-quality.html"),
        ("Methods", "methodology.html"),
        ("Era Choices", "era-boundaries.html"),
    )),
]
RETIRED_PUBLIC_ROUTES = ("networks.html",)

NAV_CSS = """
.global-nav{position:sticky;top:0;z-index:100;background:rgba(249,249,247,.94);
backdrop-filter:blur(14px);border-bottom:1px solid rgba(11,11,11,.12)}
.nav-shell{max-width:1120px;margin:0 auto;padding:8px 20px;display:flex;
align-items:center;gap:20px}.nav-brand{display:flex;align-items:center;gap:9px;
box-sizing:border-box;min-width:44px;min-height:44px;color:#172a38;text-decoration:none;font-weight:760;
letter-spacing:-.015em;white-space:nowrap}
.nav-mark{display:grid;place-items:center;width:31px;height:31px;border-radius:9px;
background:#172a38;color:#fff;font:700 15px Georgia,serif}.nav-links{display:flex;
align-items:center;gap:3px;margin-left:auto;min-width:0}.nav-item{position:relative}
.nav-item>a,.nav-trigger{display:flex;align-items:center;justify-content:center;gap:5px;
box-sizing:border-box;min-width:44px;min-height:44px;border:0;border-radius:8px;
padding:7px 9px;background:transparent;
color:#53616c;text-decoration:none;font:620 .79rem/1 system-ui,sans-serif;
white-space:nowrap;cursor:pointer}.nav-item>a:hover,.nav-trigger:hover,
.nav-group.is-open>.nav-trigger{background:#ece7df;color:#172a38}
.nav-item>a[aria-current=page],.nav-group.nav-group-current>.nav-trigger{
background:#172a38;color:#fff}.nav-trigger::after{content:"";width:6px;height:6px;
border-right:1.5px solid currentColor;border-bottom:1.5px solid currentColor;
transform:translateY(-2px) rotate(45deg);transition:transform .16s ease}
.nav-group.is-open>.nav-trigger::after{transform:translateY(1px) rotate(225deg)}
.nav-submenu{position:absolute;top:100%;right:0;z-index:120;padding-top:7px}
.nav-submenu-panel{display:grid;min-width:178px;padding:7px;
border:1px solid rgba(11,11,11,.14);border-radius:11px;background:#fff;
box-shadow:0 14px 34px rgba(23,42,56,.18)}
.nav-submenu[hidden]{display:none}.nav-submenu a{display:flex;align-items:center;
box-sizing:border-box;min-height:44px;border-radius:7px;padding:9px 10px;
color:#3f4d58;text-decoration:none;font-size:.8rem;font-weight:620;
white-space:nowrap}.nav-submenu a:hover{background:#f1ede7;color:#172a38}
.nav-submenu a[aria-current=page]{background:#e2ebf1;color:#172a38}
html:not(.nav-enhanced) .nav-group:hover>.nav-submenu,
html:not(.nav-enhanced) .nav-group:focus-within>.nav-submenu{display:block!important}
.nav-item>a:focus-visible,.nav-trigger:focus-visible,.nav-submenu a:focus-visible,
.nav-brand:focus-visible{outline:3px solid #7a3d00;outline-offset:2px}
.footer-feedback{margin-top:12px}.footer-feedback a{font-weight:650}
@media(max-width:760px){.nav-shell{padding:7px 10px;gap:8px;align-items:flex-start}
.nav-brand-label{display:none}.nav-links{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
gap:2px;width:100%;margin-left:0}.nav-item{min-width:0}.nav-item>a,.nav-trigger{
width:100%;min-width:44px;font-size:.72rem;padding:6px 4px}.nav-submenu{max-width:min(220px,calc(100vw - 20px))}
.nav-group:nth-last-child(2) .nav-submenu{left:50%;right:auto;transform:translateX(-50%)}}
@media(prefers-reduced-motion:reduce){.nav-trigger::after{transition:none}}
@media(forced-colors:active){.global-nav{background:Canvas;border-color:CanvasText;
backdrop-filter:none}.nav-submenu-panel{background:Canvas;color:CanvasText;
border:2px solid CanvasText;box-shadow:none}.nav-item>a:hover,.nav-trigger:hover,
.nav-group.is-open>.nav-trigger,.nav-submenu a:hover{background:Canvas;color:CanvasText;
outline:1px solid currentColor}.nav-item>a[aria-current=page],
.nav-group.nav-group-current>.nav-trigger,.nav-submenu a[aria-current=page]{
background:Canvas;color:CanvasText;text-decoration:underline;text-decoration-thickness:3px;
text-underline-offset:4px}.nav-item>a:focus-visible,.nav-trigger:focus-visible,
.nav-submenu a:focus-visible,.nav-brand:focus-visible{outline-color:Highlight}}
"""

NAV_JS = """
<script>
(() => {
  const nav = document.currentScript.previousElementSibling;
  if (!nav || !nav.matches(".global-nav")) return;
  const groups = [...nav.querySelectorAll(".nav-group")];
  const hoverPointer = matchMedia("(any-hover: hover)");
  let openGroup = null;
  let openMode = null;
  const sync = (group, isOpen) => {
    const trigger = group.querySelector(".nav-trigger");
    const submenu = group.querySelector(".nav-submenu");
    trigger.setAttribute("aria-expanded", isOpen ? "true" : "false");
    submenu.hidden = !isOpen;
    group.classList.toggle("is-open", isOpen);
  };
  const close = (group, restoreFocus = false) => {
    const trigger = group.querySelector(".nav-trigger");
    if (openGroup === group) {
      openGroup = null;
      openMode = null;
    }
    sync(group, false);
    if (restoreFocus) trigger.focus();
  };
  const open = (group, mode) => {
    if (openGroup && openGroup !== group) sync(openGroup, false);
    openGroup = group;
    openMode = mode;
    sync(group, true);
  };
  const activate = group => {
    if (openGroup !== group) {
      open(group, "pinned");
      return;
    }
    if (openMode === "hover") {
      openMode = "pinned";
      sync(group, true);
      return;
    }
    close(group);
  };
  groups.forEach(group => {
    const trigger = group.querySelector(".nav-trigger");
    trigger.addEventListener("click", () => activate(group));
    trigger.addEventListener("keydown", event => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      activate(group);
    });
    group.addEventListener("pointerenter", event => {
      if (event.pointerType === "touch") return;
      const canHover = event.pointerType === "mouse" ||
        (event.pointerType === "pen" && hoverPointer.matches);
      if (canHover && openGroup !== group) open(group, "hover");
    });
    group.addEventListener("pointerleave", event => {
      if (event.pointerType === "touch") return;
      if (openGroup === group && openMode === "hover" &&
          !group.contains(document.activeElement)) close(group);
    });
    group.addEventListener("focusout", () => setTimeout(() => {
      if (!group.contains(document.activeElement)) close(group);
    }));
    group.addEventListener("keydown", event => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      close(group, true);
    });
  });
  document.addEventListener("pointerdown", event => {
    if (!nav.contains(event.target)) groups.forEach(group => close(group));
  });

  const setNavHeight = () => {
    document.documentElement.style.setProperty("--global-nav-height", `${nav.offsetHeight}px`);
  };
  setNavHeight();
  if ("ResizeObserver" in window) new ResizeObserver(setNavHeight).observe(nav);
  else addEventListener("resize", setNavHeight);

  document.documentElement.classList.add("nav-enhanced");
})();
</script>
"""


def nav(prefix: str = "", current_href: str | None = None) -> str:
    items: list[str] = []
    for label, target in NAV:
        if isinstance(target, str):
            active = ' aria-current="page"' if target == current_href else ""
            items.append(
                f'<div class="nav-item"><a href="{prefix}{target}" '
                f'data-nav-label="{label}"{active}>{label}</a></div>'
            )
            continue
        key = label.lower()
        group_current = any(href == current_href for _, href in target)
        child_items = []
        for child, href in target:
            active = ' aria-current="page"' if href == current_href else ""
            child_items.append(f'<a href="{prefix}{href}"{active}>{child}</a>')
        children = "".join(child_items)
        items.append(
            f'<div class="nav-item nav-group{" nav-group-current" if group_current else ""}" '
            f'data-nav-label="{label}"><button class="nav-trigger" type="button" '
            f'aria-expanded="false" aria-controls="nav-submenu-{key}">{label}</button>'
            f'<div class="nav-submenu" id="nav-submenu-{key}" hidden>'
            f'<div class="nav-submenu-panel">{children}</div></div></div>'
        )
    links = "".join(items)
    return (
        '<nav class="global-nav" aria-label="Primary"><div class="nav-shell">'
        f'<a class="nav-brand" href="{prefix}index.html"><span class="nav-mark">P</span>'
        '<span class="nav-brand-label">Presidential Profiles</span></a>'
        f'<div class="nav-links">{links}</div></div></nav>{NAV_JS}'
    )


def _page(
    title: str,
    intro: str,
    body: str,
    scripts: str = "",
    current_href: str | None = None,
) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Presidential Profiles</title>
<style>{PAGE_CSS}
{NAV_CSS}
.note,.audit-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}}
.audit-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.audit-card strong{{display:block;font-size:1.5rem}} details{{margin:.8rem 0;padding:.7rem;border:1px solid var(--border);border-radius:10px}}
table{{border-collapse:collapse;width:100%;display:block;overflow-x:auto}}th,td{{padding:.5rem .65rem;border-bottom:1px solid var(--grid);text-align:left}}
th button{{border:0;background:transparent;padding:0;color:inherit;font-weight:700;cursor:pointer;text-align:left}}
button,select,input{{font:inherit}} .tabs,.filters{{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}}
.tabs button,.filters button{{padding:7px 11px;border:1px solid var(--border);border-radius:999px;background:var(--surface);cursor:pointer}}
.tabs button[aria-selected=true]{{background:var(--ink);color:var(--page)}} .panel[hidden]{{display:none}}
.network-plot{{height:620px}} .sr-note{{color:var(--muted);font-size:.85rem}}
@media(max-width:700px){{.network-plot{{height:460px}}}}
</style></head><body>{nav(current_href=current_href)}<header><h1>{html.escape(title)}</h1><p class="sub">{intro}</p></header>
<main>{body}</main><footer><p>Presidential Profiles · generated from versioned public artifacts.</p></footer>
{scripts}</body></html>"""


def build_quality_tables(data_dir: Path, public_dir: Path) -> dict:
    """Compatibility wrapper around the governed, renderer-independent audit."""
    return quality_audit.build_and_write(data_dir, public_dir)


def _metric_page() -> str:
    body, scripts = trust_appendices.render_metric_dictionary()
    return _page(
        "Metric dictionary",
        "Plain-language definitions first, then exact units, formulas, sources, and limitations.",
        body,
        scripts,
        current_href="methodology.html",
    )


def _quality_page(summary: dict) -> str:
    charts = summary["_charts"]

    def pct(value: float | None, digits: int = 1) -> str:
        return "Not estimable" if value is None else f"{float(value) * 100:.{digits}f}%"

    def ci_text(row: dict) -> str:
        if row.get("ci_low") is None or row.get("ci_high") is None:
            return "Not estimable for this slice"
        return f"{pct(row['ci_low'])}–{pct(row['ci_high'])}"

    def meter(value: float, maximum: float, label: str, tone: str = "blue") -> str:
        width = 0.0 if maximum <= 0 else max(0.0, min(100.0, value / maximum * 100))
        return (
            f'<span class="meter" aria-label="{html.escape(label, quote=True)}">'
            f'<span class="meter-fill {tone}" style="width:{width:.2f}%"></span></span>'
        )

    def excerpt(value: object, limit: int = 520) -> str:
        text = " ".join(str(value).split())
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + "…"

    zero_sum_case_cards = "".join(
        "<article class=\"case-study\">"
        f"<h3>{'Primary-only' if row['zero_sum_primary'] else 'Second-only'} example</h3>"
        f"<p class=\"case-meta\">{html.escape(str(row['president']))} · {int(row['year'])} · "
        f"{html.escape(str(row['era_label']))} · paragraph {int(row['para_idx'])}</p>"
        f"<blockquote>{html.escape(excerpt(row['text']))}</blockquote>"
        f"<p><strong>Primary:</strong> {'yes' if row['zero_sum_primary'] else 'no'} · "
        f"<strong>Second:</strong> {'yes' if row['zero_sum_second'] else 'no'}</p>"
        "</article>"
        for row in charts["zero_sum_cases"]
    )
    entity_case_labels = {
        "matched": "Matched after normalization",
        "primary_only": "Primary-only detection",
        "second_only": "Second-only detection",
    }
    entity_case_cards = "".join(
        "<article class=\"case-study\">"
        f"<h3>{html.escape(entity_case_labels[str(row['case_kind'])])}</h3>"
        f"<p class=\"case-meta\">{html.escape(str(row['president']))} · {int(row['year'])} · "
        f"{html.escape(str(row['era_label']))} · paragraph {int(row['para_idx'])}</p>"
        f"<blockquote>{html.escape(excerpt(row['text']))}</blockquote>"
        f"<dl class=\"case-fields\"><div><dt>Primary string</dt><dd>{html.escape(str(row['entity_primary']) or 'Not extracted')}</dd></div>"
        f"<div><dt>Second string</dt><dd>{html.escape(str(row['entity_second']) or 'Not extracted')}</dd></div>"
        f"<div><dt>Normalized key</dt><dd><code>{html.escape(str(row['entity_normalized']))}</code></dd></div></dl>"
        "</article>"
        for row in charts["entity_cases"]
    )

    completion_rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{html.escape(str(row['era_label']))}</th>"
        f"<td>{int(row['completed_paragraphs']):,} / {int(row['expected_paragraphs']):,}</td>"
        f"<td>{float(row['completion_rate']):.1%}</td>"
        f"<td>{meter(float(row['completion_rate']), 1.0, '{}: {:.1%} complete'.format(row['era_label'], float(row['completion_rate'])))}</td>"
        "</tr>"
        for row in charts["completion"]
    )
    era_zero_rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{html.escape(str(row['era_label']))}</th>"
        f"<td>{'Not estimable' if row['kappa'] is None else format(float(row['kappa']), '.2f')}</td>"
        f"<td>{float(row['primary_positive_rate']):.1%}</td>"
        f"<td>{float(row['second_positive_rate']):.1%}</td>"
        f"<td>{int(row['n']):,}</td>"
        "</tr>"
        for row in charts["zero_sum_era"]
        if row["era_key"] != "overall"
    )
    overall_agreement = [
        row for row in charts["agreement"]
        if row["era_key"] == "overall"
        and row["field"] in {
            "party_attack", "enemy_naming", "zero_sum", "topics",
            "proposal_values", "entities",
        }
    ]
    field_labels = {
        ("party_attack", "cohen_kappa"): "Partisan attack · Cohen’s κ",
        ("enemy_naming", "cohen_kappa"): "Enemy naming · Cohen’s κ",
        ("zero_sum", "cohen_kappa"): "Zero-sum framing · Cohen’s κ",
        ("topics", "jaccard_normalized"): "Topic sets · normalized Jaccard",
        ("proposal_values", "exact_match"): "Proposal / values · exact match",
        ("entities", "name_match_rate"): "Entity names · union match",
        ("entities", "stance_agreement"): "Matched entities · stance agreement",
    }
    agreement_rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{html.escape(field_labels[(row['field'], row['metric'])])}</th>"
        f"<td>{pct(row['value'])}</td><td>{html.escape(ci_text(row))}</td>"
        f"<td>{int(row['n_units']):,} {html.escape(str(row['support_unit']).replace('_', ' '))}</td>"
        "</tr>"
        for row in overall_agreement
    )
    taxonomy_rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{html.escape(str(row['era_label']))}</th>"
        f"<td>{int(row['n_paragraphs']):,}</td>"
        f"<td>{int(row['unlabeled_paragraphs']):,} ({int(row['unlabeled_paragraphs']) / int(row['n_paragraphs']):.1%})</td>"
        f"<td>{int(row['multi_label_paragraphs']):,} ({int(row['multi_label_paragraphs']) / int(row['n_paragraphs']):.1%})</td>"
        f"<td>{int(row['normalized_assignments']):,}</td>"
        "</tr>"
        for row in charts["taxonomy"]
    )
    coverage_rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{html.escape(str(row['era_label']))}</th>"
        f"<td>{int(row['n_speeches']):,}</td><td>{int(row['n_presidents']):,}</td>"
        f"<td>{int(row['n_words']):,}</td>"
        "</tr>"
        for row in charts["era_coverage"]
    )
    uncertainty = {row["surface"]: row for row in charts["uncertainty"]}
    corex_uncertainty = uncertainty["corex_issues"]
    llm_uncertainty = uncertainty["llm_topics"]
    manifest_hash = summary.get("manifest", {}).get("metadata_sha256", "available after publication")
    bootstrap_draws = summary.get("manifest", {}).get("agreement_bootstrap", {}).get(
        "draws", quality_audit.BOOTSTRAP_DRAWS
    )
    primary_no_total = summary["zero_sum_both_no"] + summary["zero_sum_second_only"]
    primary_yes_total = summary["zero_sum_primary_only"] + summary["zero_sum_both_yes"]
    population_receipt = trust_appendices.trust_receipt(
        claim="Public results must declare whether they use source-document ownership or the corrected actual-speaker population.",
        population=f"{summary['source_paragraphs']:,} source paragraphs; {summary['eligible_paragraphs']:,} analysis-eligible actual-speaker paragraphs",
        method="Keyed population ledger with governed speaker-attribution eligibility and exclusion rules",
        uncertainty="Descriptive census; no sampling interval",
        artifact="data/quality/population_ledger.csv",
        status="Governed data-quality-v2 projection",
        limitation="Population stages use different units and are not additive",
        downstream_use="Select and label the denominator used by every president-attributed result",
    )
    audit_receipt = trust_appendices.trust_receipt(
        claim="The second model is an independent reproducibility check, not human ground truth.",
        population=f"{summary['paired']:,} paired paragraphs from {summary['sampled_speeches']:,} sampled speeches",
        method="Field-specific normalized agreement with 95% whole-speech bootstrap intervals",
        uncertainty=f"{int(bootstrap_draws):,} deterministic speech-cluster resamples",
        artifact="data/quality/manifest_v2.json",
        status="Cross-model reproducibility measured; human validity not measured",
        limitation="Related model families can share errors and historical assumptions",
        downstream_use="Qualify model-sensitive findings and define future human validation",
    )

    body = f"""
<style>
.quality-nav{{position:sticky;top:calc(var(--global-nav-height,60px) + 8px);z-index:20;
display:flex;gap:7px;overflow-x:auto;padding:9px;margin:0 0 22px;background:rgba(249,249,247,.96);
border:1px solid var(--border);border-radius:12px;scrollbar-width:thin}}
.quality-nav a{{flex:0 0 auto;padding:7px 10px;border-radius:999px;color:var(--ink);
font-size:.78rem;font-weight:700;text-decoration:none}}.quality-nav a:hover{{background:#e8edf0}}
.verdict-grid,.quality-context,.trust-grid,.uncertainty-grid{{display:grid;
grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:16px 0}}
.verdict-card,.trust-card,.uncertainty-card{{border:1px solid var(--border);border-radius:14px;
padding:16px;background:var(--surface)}}.verdict-card strong{{display:block;font-size:1.55rem;
line-height:1.1;margin-bottom:7px}}.verdict-card span,.trust-card p,.uncertainty-card p{{
font-size:.86rem;color:var(--muted)}}.verdict-card.good{{border-top:5px solid #376c51}}
.verdict-card.caution{{border-top:5px solid #9b5f26}}.verdict-card.scope{{border-top:5px solid #245f88}}
.layer-label{{display:inline-block;margin:0 0 8px;color:#5d6870;font-size:.72rem;font-weight:800;
letter-spacing:.09em;text-transform:uppercase}}section[id]{{scroll-margin-top:190px}}
.quality-figure{{margin:18px 0;padding:17px;border:1px solid var(--border);border-radius:14px;
background:#fff}}.quality-figure figcaption{{font-weight:750;margin-bottom:12px}}
.ledger{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}
.ledger-stage{{padding:14px;border-radius:12px;background:#f2f5f6;border:1px solid #dce3e6}}
.ledger-stage b{{display:block;font-size:1.3rem;color:#172a38}}.ledger-stage small{{color:#596771}}
.funnel{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:stretch;gap:9px}}
.funnel-step{{padding:15px;border:1px solid var(--border);border-radius:12px;background:#f6f7f7}}
.funnel-step strong{{display:block;font-size:1.45rem}}.funnel-arrow{{align-self:center;font-size:1.5rem;
color:#66747d}}.meter{{display:block;width:100%;min-width:120px;height:9px;border-radius:99px;
overflow:hidden;background:#e6eaec}}.meter-fill{{display:block;height:100%;background:#245f88}}
.meter-fill.green{{background:#376c51}}.meter-fill.orange{{background:#9b5f26}}
.quality-table{{display:table;width:100%;font-size:.82rem}}.quality-table th{{font-weight:700}}
.quality-table td,.quality-table th{{vertical-align:top}}.quality-table tbody tr:hover{{background:#f4f6f7}}
.matrix td{{text-align:right}}.matrix .shared{{background:#dcebe2}}.matrix .split{{background:#f5e7d7}}
.case-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:18px}}
.case-grid.entities{{grid-template-columns:repeat(3,minmax(0,1fr))}}.case-study{{min-width:0;
padding:14px;border:1px solid #d7dfe3;border-radius:12px;background:#f8f9f9}}
.case-study h3{{font-size:.95rem;margin:.1rem 0 .25rem}}.case-meta{{color:#52616c;font-size:.75rem}}
.case-study blockquote{{margin:.7rem 0;padding:.7rem 0 .7rem 12px;border-left:3px solid #6d7d86;
font-size:.8rem;line-height:1.55}}.case-fields{{display:grid;gap:6px;margin:.7rem 0 0}}
.case-fields div{{display:grid;grid-template-columns:7rem 1fr;gap:7px}}.case-fields dt{{font-size:.7rem;
font-weight:800;color:#52616c;text-transform:uppercase}}.case-fields dd{{margin:0;font-size:.79rem;
overflow-wrap:anywhere}}
.table-scroll{{max-width:100%;overflow-x:auto}}.scroll-cue{{display:none;color:#52616c;
font-size:.7rem;font-weight:750;white-space:nowrap}}#receipts code,.case-fields code{{overflow-wrap:anywhere;
word-break:break-word}}
.receipt{{border-left:4px solid #245f88;padding:10px 13px;margin:14px 0;background:#f2f5f7;
font-size:.84rem}}.receipt b{{color:#172a38}}.status-pill{{display:inline-block;border-radius:999px;
padding:4px 8px;background:#deece3;color:#25573e;font-size:.72rem;font-weight:800}}
.trust-receipt{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;
background:var(--border);border:1px solid var(--border);border-radius:14px;overflow:hidden;
margin:20px 0}}.trust-receipt div{{background:#fff;padding:13px;min-width:0}}
.trust-receipt dt{{color:#52616c;font-size:.68rem;font-weight:800;letter-spacing:.06em;
text-transform:uppercase}}.trust-receipt dd{{margin:.35rem 0 0;font-size:.8rem;overflow-wrap:anywhere}}
.downloads{{columns:2;column-gap:28px}}.downloads li{{break-inside:avoid;margin:.35rem 0}}
.technical-note{{font-size:.85rem;color:var(--muted)}}details.exact{{margin-top:14px}}
a:focus-visible,.quality-nav a:focus-visible,summary:focus-visible,.table-scroll:focus-visible{{outline:3px solid #7a3d00;
outline-offset:3px}}
@media(max-width:820px){{.verdict-grid,.quality-context,.trust-grid,.uncertainty-grid{{
grid-template-columns:1fr}}.trust-receipt{{grid-template-columns:1fr 1fr}}.ledger{{grid-template-columns:1fr 1fr}}.funnel{{
grid-template-columns:1fr}}.funnel-arrow{{transform:rotate(90deg);justify-self:center}}
.downloads{{columns:1}}.quality-nav{{top:calc(var(--global-nav-height,90px) + 6px)}}
.scroll-cue{{display:inline-flex;align-items:center;padding:7px 8px;background:#eef2f3;border-radius:999px}}
.table-scroll{{overscroll-behavior-inline:contain}}.table-scroll::before{{content:"Scroll table horizontally →";
display:block;position:sticky;left:0;width:max-content;margin:0 0 7px;padding:4px 7px;border-radius:6px;
background:#eef2f3;color:#52616c;font-size:.68rem;font-weight:750}}.quality-table{{min-width:38rem}}}}
@media(max-width:470px){{.ledger,.trust-receipt{{grid-template-columns:1fr}}.quality-figure{{padding:12px}}
.verdict-card,.trust-card,.uncertainty-card{{padding:13px}}}}
@media(max-width:700px){{.case-grid,.case-grid.entities{{grid-template-columns:1fr}}}}
@media(forced-colors:active){{.verdict-card,.quality-figure,.ledger-stage,.funnel-step,
.trust-card,.uncertainty-card{{border:1px solid CanvasText}}.meter{{border:1px solid CanvasText}}
.meter-fill{{background:Highlight}}}}
</style>
<nav class="quality-nav" aria-label="Data quality sections">
<span class="scroll-cue" aria-hidden="true">More sections →</span>
<a href="#verdict">Verdict</a><a href="#population">Population</a>
<a href="#completion">Missingness</a><a href="#reproducibility">Reproducibility</a>
<a href="#entities">Entities</a><a href="#uncertainty">Uncertainty</a>
<a href="#coverage">Coverage</a><a href="#receipts">Receipts</a>
</nav>

<section id="verdict"><span class="layer-label">30-second verdict</span>
<h2>Useful evidence—with explicit boundaries</h2>
<p>This audit asks whether the site’s conclusions survive different instruments and
whether every denominator is visible. It does <strong>not</strong> claim that agreement
between two related language models is human-validated accuracy.</p>
<div class="verdict-grid">
<article class="verdict-card good"><strong>{summary['paired']:,}</strong>
<span>paired paragraph judgments make broad reproducibility checks possible.</span></article>
<article class="verdict-card caution"><strong>κ {summary['zero_sum_kappa']:.2f}</strong>
<span>zero-sum judgments are materially model-sensitive despite
{summary['zero_sum_raw_agreement']:.1%} raw agreement.</span></article>
<article class="verdict-card scope"><strong>{summary['eligible_paragraphs']:,}</strong>
<span>speaker-audited paragraphs are eligible for actual-speaker analysis; other pages
may use explicitly different populations.</span></article>
</div>
<p class="note"><strong>What changed in this audit:</strong> version 2 repairs the
missing-row calculation, restricts entity comparisons to genuinely paired paragraphs,
normalizes and deduplicates topics before counting, expresses interval widths in percentage
points, and orders every era chronologically.</p></section>

<section id="population"><span class="layer-label">Two-minute visual story · 1</span>
<h2>First choose the population</h2>
<p>The project has several legitimate populations because “the document filed under a
president” and “the president actually speaking this paragraph” are different questions.
The cards below are a ledger, not one additive funnel.</p>
<figure class="quality-figure" data-substantive-chart
 data-metric="annotation_population_lineage"
 data-evidence="data/quality/population_ledger.csv"
 aria-labelledby="population-caption">
<figcaption id="population-caption">Population ledger · <a href="metrics.html#annotation_population_lineage">define population lineage</a></figcaption>
<div class="ledger">
<div class="ledger-stage"><b>{summary['source_documents']:,} documents</b>
<small>{summary['source_paragraphs']:,} source paragraphs</small></div>
<div class="ledger-stage"><b>{summary['retained_documents']:,} retained</b>
<small>{summary['retained_paragraphs']:,} speaker-audited paragraphs</small></div>
<div class="ledger-stage"><b>{summary['eligible_paragraphs']:,} eligible</b>
<small>{summary['excluded_paragraphs']:,} excluded; {summary['cross_owner_paragraphs']:,}
eligible cross-owner paragraphs</small></div>
<div class="ledger-stage"><b>{summary['paired']:,} paired</b>
<small>of {summary['sampled']:,} expected second-opinion paragraphs</small></div>
</div>
<p class="technical-note">An actual-speaker appearance is a speaker within a source
document: the governed layer contains {summary['speaker_appearances']:,} appearances across
{summary['retained_documents']:,} retained documents.</p></figure>
{population_receipt}</section>

<section id="completion"><span class="layer-label">Two-minute visual story · 2</span>
<h2>The second reading is broad, not complete</h2>
<p>The persisted document-level sample contains {summary['sampled_speeches']:,} speeches
and {summary['sampled']:,} expected paragraphs. The second pass completed
<strong>{summary['paired']:,} ({summary['paired']/summary['sampled']:.1%})</strong>.
The remaining <strong>{summary['missing']:,}</strong> rows occur across
{summary['affected_speeches']} speeches; they remain missing rather than becoming negative
labels.</p>
<figure class="quality-figure" data-substantive-chart
 data-metric="annotation_processing_completion"
 data-evidence="data/quality/second_model_by_era.csv"
 aria-labelledby="completion-caption">
<figcaption id="completion-caption">Second-opinion completion by Story era · <a href="metrics.html#annotation_processing_completion">define completion</a></figcaption>
<div class="table-scroll" role="region" aria-label="Second-opinion completion by Story era table" tabindex="0">
<table class="quality-table"><thead><tr><th scope="col">Era</th><th scope="col">Completed</th>
<th scope="col">Rate</th><th scope="col">Visual rate</th></tr></thead>
<tbody>{completion_rows}</tbody></table></div></figure>
<p><a href="data/quality/second_model_by_speech.csv" download>Inspect missingness by speech</a>
· <a href="data/quality/second_model_by_era.csv" download>Download era totals</a></p></section>

<section id="reproducibility"><span class="layer-label">Technical evidence · 1</span>
<h2>{summary['zero_sum_raw_agreement']:.1%} agreement can still hide unstable positives</h2>
<p>Zero-sum framing is rare. Both models usually say “no,” so raw agreement is dominated
by easy shared negatives. Cohen’s κ adjusts for chance agreement under the observed label
prevalence and falls to <strong>{summary['zero_sum_kappa']:.2f}</strong>.</p>
<figure class="quality-figure" data-substantive-chart
 data-metric="intermodel_agreement_overall"
 data-evidence="data/quality/zero_sum_confusion.csv"
 aria-labelledby="zero-flow-caption">
<figcaption id="zero-flow-caption">Zero-sum decisions: counts and percentages within each
primary-model row · <a href="metrics.html#intermodel_agreement_overall">define overall reproducibility</a></figcaption>
<div class="table-scroll" role="region" aria-label="Zero-sum decision comparison table" tabindex="0">
<table class="quality-table matrix"><thead><tr><th scope="col">Primary decision</th>
<th scope="col">Second: no</th><th scope="col">Second: yes</th><th scope="col">Row total</th>
</tr></thead><tbody>
<tr><th scope="row">Primary: no</th><td class="shared">{summary['zero_sum_both_no']:,}
({summary['zero_sum_both_no']/primary_no_total:.1%})</td>
<td class="split">{summary['zero_sum_second_only']:,}
({summary['zero_sum_second_only']/primary_no_total:.1%})</td>
<td>{primary_no_total:,}</td></tr>
<tr><th scope="row">Primary: yes</th><td class="split">{summary['zero_sum_primary_only']:,}
({summary['zero_sum_primary_only']/primary_yes_total:.1%})</td>
<td class="shared">{summary['zero_sum_both_yes']:,}
({summary['zero_sum_both_yes']/primary_yes_total:.1%})</td>
<td>{primary_yes_total:,}</td></tr>
</tbody></table></div>
<p>The off-diagonal cells are <strong>primary-only</strong> and <strong>second-only</strong>
judgments. Without adjudicated human labels, neither cell identifies which judgment is correct.</p></figure>
<aside aria-labelledby="zero-sum-cases-heading">
<h3 id="zero-sum-cases-heading">What a disagreement looks like</h3>
<p class="technical-note"><strong>Illustrative, not adjudicated.</strong> These two
artifact-selected paragraphs show each disagreement direction; they do not establish which
model is correct. <a href="data/quality/zero_sum_disagreement_excerpts.csv" download>Inspect
all disagreement excerpts</a>.</p><div class="case-grid">{zero_sum_case_cards}</div></aside>

<figure class="quality-figure" data-substantive-chart
 data-metric="intermodel_agreement_overall kappa jaccard exact_agreement"
 data-evidence="data/quality/agreement_by_field_and_era.csv"
 aria-labelledby="agreement-caption">
<figcaption id="agreement-caption">Field-specific reproducibility with 95% whole-speech
bootstrap intervals · <a href="metrics.html#intermodel_agreement_overall">overall reproducibility</a> ·
<a href="metrics.html#kappa">κ</a> ·
<a href="metrics.html#jaccard">Jaccard</a> ·
<a href="metrics.html#exact_agreement">exact agreement</a></figcaption>
<div class="table-scroll" role="region" aria-label="Field-specific reproducibility table" tabindex="0">
<table class="quality-table"><thead><tr><th scope="col">Field and appropriate metric</th>
<th scope="col">Estimate</th><th scope="col">95% interval</th><th scope="col">Support</th>
</tr></thead><tbody>{agreement_rows}</tbody></table></div>
<p class="technical-note">These metrics answer different questions and should not be read
as a league table. κ is for flags, normalized Jaccard for topic sets, and exact match for
one-of-four proposal/value judgments. The intervals resample complete speeches
{int(bootstrap_draws):,} times.</p>
<details class="exact"><summary>Zero-sum detail by Story era</summary>
<div class="table-scroll" role="region" aria-label="Zero-sum detail by Story era table" tabindex="0">
<table class="quality-table"><thead><tr><th scope="col">Era</th><th scope="col">κ</th>
<th scope="col">Primary positive</th><th scope="col">Second positive</th>
<th scope="col">Paragraphs</th></tr></thead><tbody>{era_zero_rows}</tbody></table></div></details>
</figure>
<div class="quality-context">
<article class="verdict-card good"><strong>Stable negatives</strong><span>
{summary['zero_sum_both_no']:,} paragraphs
receive a shared negative judgment.</span></article>
<article class="verdict-card caution"><strong>Model-sensitive positives</strong><span>The
second pass confirms {summary['zero_sum_primary_confirmation']:.1%} of primary positives.</span></article>
<article class="verdict-card scope"><strong>Interpretation</strong><span>Use shared positives
as a cross-model-stable subset; treat absolute rates and rankings as model-dependent.</span></article>
</div></section>

<section id="entities"><span class="layer-label">Technical evidence · 2</span>
<h2>Finding the same name is harder than judging its stance</h2>
<p>Entity evaluation has two stages. First ask whether both models extracted the same
normalized name; only then ask whether their stance judgments agree.</p>
<figure class="quality-figure" data-substantive-chart
 data-metric="entity_name_and_stance_reproducibility"
 data-evidence="data/quality/entity_name_agreement.csv"
 aria-labelledby="entity-caption">
<figcaption id="entity-caption">Paired entity evaluation funnel ·
<a href="metrics.html#entity_name_and_stance_reproducibility">define the two-stage measure</a></figcaption>
<div class="funnel">
<div class="funnel-step"><strong>{summary['entity_union']:,}</strong> normalized entity
instances named by either model</div><div class="funnel-arrow" aria-hidden="true">→</div>
<div class="funnel-step"><strong>{summary['entity_matched']:,} ·
{summary['entity_match_rate']:.1%}</strong> named by both models</div>
<div class="funnel-arrow" aria-hidden="true">→</div>
<div class="funnel-step"><strong>{summary['entity_stance_matches']:,} ·
{summary['entity_stance_agreement']:.1%}</strong> matched names with the same stance</div>
</div>
<p>Mean paragraph-level name-set Jaccard is
<strong>{summary['entity_jaccard_bearing']:.3f}</strong> among the
{summary['entity_bearing_paragraphs']:,} entity-bearing paragraphs. Including all paired
paragraphs—and treating two empty sets as
agreement—it is <strong>{summary['entity_jaccard_all']:.3f}</strong>. Both denominators are
shown because they answer different questions.</p></figure>
<p>Case and whitespace normalization is deliberately conservative; it does not silently
collapse aliases such as a title and a full name.</p>
<aside aria-labelledby="entity-cases-heading"><h3 id="entity-cases-heading">From source
text to a normalized comparison key</h3>
<p class="technical-note"><strong>Illustrative, not adjudicated.</strong> The first row
shows two extracted strings joined by the governed normalization rule. The directional rows
show an extraction made by only one model; “not extracted” is not a human judgment that the
entity is absent. <a href="data/quality/entity_case_examples.csv" download>Download the exact
case-study rows</a>.</p><div class="case-grid entities">{entity_case_cards}</div></aside>
<p>
<a href="data/quality/entity_name_agreement.csv" download>Paragraph audit</a> ·
<a href="data/quality/entity_stance_matches.csv" download>Matched-name stance rows</a> ·
<a href="data/quality/president_alias_examples.csv" download>President alias registry</a></p>
</section>

<section id="uncertainty"><span class="layer-label">Technical evidence · 3</span>
<h2>Two surfaces, two uncertainty budgets</h2>
<p>These widths are displayed in <strong>percentage points</strong>. CorEx uses five-year
periods; LLM topics use nine broad Story eras. Their averages are shown separately because
different temporal grains and base rates make a cross-surface bar comparison misleading.</p>
<div class="uncertainty-grid">
<figure class="uncertainty-card" data-substantive-chart
 data-metric="uncertainty_envelope"
 data-evidence="data/quality/uncertainty_decomposition.csv"
 aria-labelledby="corex-uncertainty-caption">
<figcaption id="corex-uncertainty-caption"><strong>CorEx issues · five-year cells</strong> ·
<a href="metrics.html#uncertainty_envelope">define the uncertainty envelope</a></figcaption>
<strong style="font-size:1.5rem">{float(corex_uncertainty['mean_sampling_width_pp']):.2f} pp</strong>
<p>mean resolved sampling interval width across
{int(corex_uncertainty['n_resolved_sampling']):,} cells. Annotator disagreement is not
applicable because this pipeline has no LLM annotator.</p></figure>
<figure class="uncertainty-card" data-substantive-chart
 data-metric="uncertainty_envelope"
 data-evidence="data/quality/uncertainty_decomposition.csv"
 aria-labelledby="llm-uncertainty-caption">
<figcaption id="llm-uncertainty-caption"><strong>LLM topics · Story-era cells</strong> ·
<a href="metrics.html#uncertainty_envelope">define the uncertainty envelope</a></figcaption>
<strong style="font-size:1.5rem">{float(llm_uncertainty['mean_sampling_width_pp']):.2f} pp
→ {float(llm_uncertainty['mean_combined_width_pp']):.2f} pp</strong>
<p>mean sampling-only to combined width across
{int(llm_uncertainty['n_resolved_combined']):,} cells after the measured paired-model
half-gap is added.</p></figure>
</div>
<p class="technical-note">The disagreement component does not capture shared model bias,
taxonomy error, or within-speech correlation in the half-gap. It assumes disagreement in
the sampled documents transfers to full-corpus era estimates.</p></section>

<section id="coverage"><span class="layer-label">Technical evidence · 4</span>
<h2>Coverage and label behavior constrain every finding</h2>
<p>After canonical case normalization and within-paragraph topic deduplication, the primary
pass contains <strong>{summary['normalized_topic_assignments']:,}</strong> assignments and
<strong>{summary['multi_label_paragraphs']:,}</strong> multi-label paragraphs.
{summary['unlabeled_paragraphs']:,} paragraphs carry no topic and remain in denominators.
The project also marks {summary['thin_presidents']} presidents with fewer than five source
documents rather than ranking them as precise outliers.</p>
<figure class="quality-figure" data-substantive-chart
 data-metric="paragraph_share"
 data-evidence="data/quality/taxonomy_label_behavior.csv"
 aria-labelledby="taxonomy-caption">
<figcaption id="taxonomy-caption">Normalized topic-label behavior by Story era ·
<a href="metrics.html#paragraph_share">define paragraph share</a></figcaption>
<div class="table-scroll" role="region" aria-label="Normalized topic-label behavior by Story era table" tabindex="0">
<table class="quality-table"><thead><tr><th scope="col">Era</th>
<th scope="col">Paragraphs</th><th scope="col">No topic</th>
<th scope="col">Multiple topics</th><th scope="col">Assignments</th>
</tr></thead><tbody>{taxonomy_rows}</tbody></table></div></figure>
<figure class="quality-figure" data-substantive-chart
 data-metric="annotation_population_lineage"
 data-evidence="data/quality/era_coverage.csv"
 aria-labelledby="era-coverage-caption">
<figcaption id="era-coverage-caption">Source-corpus coverage by Story era ·
<a href="metrics.html#annotation_population_lineage">define population lineage</a></figcaption>
<div class="table-scroll" role="region" aria-label="Source-corpus coverage by Story era table" tabindex="0">
<table class="quality-table"><thead><tr><th scope="col">Era</th><th scope="col">Documents</th>
<th scope="col">Presidents</th><th scope="col">Words</th></tr></thead>
<tbody>{coverage_rows}</tbody></table></div>
<p class="technical-note">This is a curated formal-speech corpus, not a census of everything
presidents said or everything the public heard. Changing genre composition can resemble
historical change.</p></figure></section>

<section id="receipts"><span class="layer-label">Reproduce and challenge it</span>
<h2>Versioned receipts, explicit failure modes</h2>
<div class="trust-grid">
<article class="trust-card"><span class="status-pill">Derived locally</span>
<h3>Build contract</h3><p><code>data-quality-v2</code> performs no model calls. It validates
semantic key sets, source hashes, populations, units, and canonical agreement values before
publication.</p></article>
<article class="trust-card"><span class="status-pill">Frozen inputs</span>
<h3>Model evidence</h3><p>Each annotation row retains its run ID. The public manifest records
model, prompt version, prompt hash, source hash, and the deterministic bootstrap recipe.</p></article>
<article class="trust-card"><span class="status-pill">Not yet measured</span>
<h3>Human validity</h3><p>Cross-model reproducibility can reveal sensitivity, not truth.
A blinded, adjudicated human study is still required for model-to-human validity estimates.
The <a href="data/validation-protocols-v1/manifest_v1.json">execution-blocked protocol receipt</a>
publishes its design and gates without exposing coder assignments.</p>
</article></div>{audit_receipt}
<p><strong>Manifest identity:</strong> <code>{html.escape(str(manifest_hash))}</code></p>
<details><summary>Downloads and machine-readable evidence</summary>
<ul class="downloads">
<li><a href="data/quality/manifest_v2.json">Quality manifest v2</a></li>
<li><a href="data/quality/population_ledger.csv" download>Population ledger</a></li>
<li><a href="data/quality/agreement_by_field_and_era.csv" download>Agreement + bootstrap intervals</a></li>
<li><a href="data/quality/public_chart_treatments.csv" download>Chart treatment map</a></li>
<li><a href="data/quality/second_model_by_speech.csv" download>Completion by speech</a></li>
<li><a href="data/quality/second_model_by_era.csv" download>Completion by era</a></li>
<li><a href="data/quality/zero_sum_confusion.csv" download>Zero-sum count matrix</a></li>
<li><a href="data/quality/zero_sum_by_era.csv" download>Zero-sum by era</a></li>
<li><a href="data/quality/zero_sum_disagreement_excerpts.csv" download>Disagreement excerpts</a></li>
<li><a href="data/quality/entity_name_agreement.csv" download>Entity name audit</a></li>
<li><a href="data/quality/entity_stance_matches.csv" download>Entity stance audit</a></li>
<li><a href="data/quality/entity_case_examples.csv" download>Entity case-study rows</a></li>
<li><a href="data/quality/uncertainty_decomposition.csv" download>Uncertainty decomposition</a></li>
<li><a href="data/quality/president_coverage.csv" download>President coverage</a></li>
<li><a href="data/quality/era_coverage.csv" download>Era coverage</a></li>
<li><a href="data/quality/taxonomy_label_behavior.csv" download>Taxonomy behavior</a></li>
<li><a href="data/validation-protocols-v1/manifest_v1.json" download>Planned validation protocol</a></li>
<li><a href="data/validation-protocols-v1/paragraph_selection_cells_v1.csv" download>Planned paragraph sampling cells</a></li>
<li><a href="data/validation-protocols-v1/speech_selection_cells_v1.csv" download>Planned factual-label sampling cells</a></li>
</ul></details>
<details><summary>Processing and interpretation checklist</summary><ol>
<li>Join paragraphs only on <code>(doc_name, para_idx)</code>, with uniqueness and
key-set validation.</li><li>Normalize taxonomy labels against the frozen taxonomy and
reject unknown values.</li><li>Keep missing, empty, excluded, and thin records visible.</li>
<li>Bootstrap whole speeches rather than treating paragraphs as independent.</li>
<li>Use the metric appropriate to each output structure and show its denominator.</li>
<li>Do not translate cross-model disagreement into accuracy without human labels.</li>
</ol></details></section>"""
    return _page(
        "Can I trust the data?",
        "A layered audit of population, completeness, reproducibility, uncertainty, and known limits.",
        body,
        current_href="data-quality.html",
    )


def _feedback_page() -> str:
    body = """
<section><h2>What belongs here</h2><ul><li>Feature suggestions</li><li>Data corrections
with a source speech or paragraph</li><li>Methodology criticism</li><li>Proposed cleaning
steps that can be tested reproducibly</li></ul></section>
<p class="note">Posts are public and moderated through GitHub Discussions. Do not include
private information. Loading the embedded discussion sends normal request metadata to
GitHub/giscus.</p>
<div class="giscus"></div>
<p><a href="https://github.com/jacobfulfyll/presidential_profiles/discussions/1">Open the
Website feedback &amp; accuracy discussion directly if the embedded thread does not load
→</a></p>"""
    scripts = """<script src="https://giscus.app/client.js"
 data-repo="jacobfulfyll/presidential_profiles" data-repo-id="R_kgDOCrXyjA"
 data-category="General" data-category-id="DIC_kwDOCrXyjM4DByqZ"
 data-mapping="pathname" data-strict="1" data-reactions-enabled="1"
 data-emit-metadata="0" data-input-position="top" data-theme="preferred_color_scheme"
 data-lang="en" data-loading="lazy" crossorigin="anonymous" async></script>"""
    return _page(
        "Feedback & accuracy",
        "A public place to improve the site and its evidence.",
        body, scripts, current_href="feedback.html",
    )


def _model_comparison_page() -> str:
    body, scripts = trust_appendices.render_model_comparison()
    return _page(
        "How the label models differ",
        "One shared corpus, two measuring instruments, and a preregistered prediction that failed.",
        body,
        scripts,
        current_href="methodology.html",
    )


def _copy_public_artifacts(site_dir: Path) -> None:
    mapping = {
        corpus.DATA_DIR / "networks": site_dir / "data" / "networks",
        corpus.DATA_DIR / "coverage_pressure": site_dir / "data" / "coverage_pressure",
    }
    for source, target in mapping.items():
        if not source.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.suffix in {".json", ".csv"}:
                shutil.copyfile(path, target / path.name)
    receipt = corpus.DATA_DIR / "inference_receipts.csv"
    if receipt.exists():
        target = site_dir / "data"
        target.mkdir(exist_ok=True)
        shutil.copyfile(receipt, target / receipt.name)
    lifecycle = corpus.DATA_DIR / "attention" / "topic_lifecycles.parquet"
    if lifecycle.exists():
        pd.read_parquet(lifecycle).to_csv(site_dir / "data" / "topic_lifecycles.csv", index=False)
    for source_name, public_name in [
        ("method_agreement.parquet", "method-agreement.csv"),
        ("method_compositions.parquet", "method-compositions.csv"),
    ]:
        source = corpus.DATA_DIR / source_name
        if source.exists():
            pd.read_parquet(source).to_csv(site_dir / "data" / public_name, index=False)


def _prune_retired_routes(site_dir: Path) -> None:
    """Remove generated routes that must not survive an incremental rebuild."""
    for relative in RETIRED_PUBLIC_ROUTES:
        (site_dir / relative).unlink(missing_ok=True)


def write(site_dir: Path) -> None:
    metrics.validate_metric_names(set(metrics.METRICS))
    site_dir.mkdir(parents=True, exist_ok=True)
    _prune_retired_routes(site_dir)
    quality_dir = site_dir / "data" / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    summary = build_quality_tables(corpus.DATA_DIR, quality_dir)
    (site_dir / "metrics.html").write_text(_metric_page())
    (site_dir / "data-quality.html").write_text(_quality_page(summary))
    (site_dir / "feedback.html").write_text(_feedback_page())
    (site_dir / "label-models.html").write_text(_model_comparison_page())
    (site_dir / "metrics.json").write_text(json.dumps(metrics.METRICS, indent=2))
    _copy_public_artifacts(site_dir)
    validation_protocols.write_public_projection(
        site_dir / "data" / "validation-protocols-v1"
    )


def add_global_navigation(site_dir: Path) -> None:
    """Final generated pass so all independently-owned page renderers share navigation."""
    for page in site_dir.rglob("*.html"):
        text = page.read_text()
        if "<body" not in text:
            continue
        relative = page.relative_to(site_dir)
        prefix = "../" * (len(relative.parts) - 1)
        if relative.parts[0] == "presidents":
            current = "presidents/index.html"
        elif relative.parts[0] == "issues":
            current = "issues/index.html"
        elif relative.name == "index_selfcontained.html":
            current = "index.html"
        elif relative.name in {"methodology.html", "metrics.html", "label-models.html"}:
            current = "methodology.html"
        else:
            current = relative.as_posix()
        if 'aria-label="Primary"' not in text:
            marker = re.search(r"<body[^>]*>", text)
            if marker is None:
                continue
            style = f"<style>{NAV_CSS}</style>"
            replacement = marker.group() + style + nav(prefix, current)
            text = text[:marker.start()] + replacement + text[marker.end():]
        if 'class="footer-feedback"' not in text and "</footer>" in text:
            footer_link = (
                '<p class="footer-feedback"><a href="'
                f'{prefix}feedback.html">Feedback and accuracy →</a></p>'
            )
            text = text.replace("</footer>", footer_link + "</footer>", 1)
        page.write_text(text)
