"""Generated metric lessons, quality audit, model-comparison and feedback pages."""
from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from . import corpus, grammar, invocations, metrics, portraits, trends
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
color:#172a38;text-decoration:none;font-weight:760;letter-spacing:-.015em;white-space:nowrap}
.nav-mark{display:grid;place-items:center;width:31px;height:31px;border-radius:9px;
background:#172a38;color:#fff;font:700 15px Georgia,serif}.nav-links{display:flex;
align-items:center;gap:3px;margin-left:auto;min-width:0}.nav-item{position:relative}
.nav-item>a,.nav-trigger{display:flex;align-items:center;justify-content:center;gap:5px;
min-height:34px;border:0;border-radius:8px;padding:7px 9px;background:transparent;
color:#53616c;text-decoration:none;font:620 .79rem/1 system-ui,sans-serif;
white-space:nowrap;cursor:pointer}.nav-item>a:hover,.nav-trigger:hover,
.nav-group.is-open>.nav-trigger{background:#ece7df;color:#172a38}
.nav-item>a[aria-current=page],.nav-group.nav-group-current>.nav-trigger{
background:#172a38;color:#fff}.nav-trigger::after{content:"";width:6px;height:6px;
border-right:1.5px solid currentColor;border-bottom:1.5px solid currentColor;
transform:translateY(-2px) rotate(45deg);transition:transform .16s ease}
.nav-group.is-open>.nav-trigger::after{transform:translateY(1px) rotate(225deg)}
.nav-submenu{position:absolute;top:calc(100% + 7px);right:0;z-index:120;display:grid;
min-width:178px;padding:7px;border:1px solid rgba(11,11,11,.14);border-radius:11px;
background:#fff;box-shadow:0 14px 34px rgba(23,42,56,.18)}
.nav-submenu[hidden]{display:none}.nav-submenu a{display:block;border-radius:7px;
padding:9px 10px;color:#3f4d58;text-decoration:none;font-size:.8rem;font-weight:620;
white-space:nowrap}.nav-submenu a:hover{background:#f1ede7;color:#172a38}
.nav-submenu a[aria-current=page]{background:#e2ebf1;color:#172a38}
html:not(.nav-enhanced) .nav-group:hover>.nav-submenu{display:grid}
.nav-item>a:focus-visible,.nav-trigger:focus-visible,.nav-submenu a:focus-visible,
.nav-brand:focus-visible{outline:3px solid #d08b45;outline-offset:2px}
.footer-feedback{margin-top:12px}.footer-feedback a{font-weight:650}
@media(max-width:760px){.nav-shell{padding:7px 10px;gap:8px;align-items:flex-start}
.nav-brand-label{display:none}.nav-links{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
gap:2px;width:100%;margin-left:0}.nav-item{min-width:0}.nav-item>a,.nav-trigger{
width:100%;min-width:0;font-size:.72rem;padding:6px 4px}.nav-submenu{max-width:min(220px,calc(100vw - 20px))}
.nav-group:nth-last-child(2) .nav-submenu{left:50%;right:auto;transform:translateX(-50%)}}
@media(prefers-reduced-motion:reduce){.nav-trigger::after{transition:none}}
"""

NAV_JS = """
<script>
(() => {
  const nav = document.currentScript.previousElementSibling;
  if (!nav || !nav.matches(".global-nav")) return;
  document.documentElement.classList.add("nav-enhanced");
  const groups = [...nav.querySelectorAll(".nav-group")];
  const finePointer = matchMedia("(hover: hover) and (pointer: fine)");
  const close = (group, restoreFocus = false) => {
    const trigger = group.querySelector(".nav-trigger");
    const submenu = group.querySelector(".nav-submenu");
    trigger.setAttribute("aria-expanded", "false");
    submenu.hidden = true;
    group.classList.remove("is-open");
    if (restoreFocus) trigger.focus();
  };
  const closeOthers = active => groups.forEach(group => {
    if (group !== active) close(group);
  });
  const open = group => {
    closeOthers(group);
    group.querySelector(".nav-trigger").setAttribute("aria-expanded", "true");
    group.querySelector(".nav-submenu").hidden = false;
    group.classList.add("is-open");
  };
  groups.forEach(group => {
    const trigger = group.querySelector(".nav-trigger");
    const toggle = () => {
      if (trigger.getAttribute("aria-expanded") === "true") close(group);
      else open(group);
    };
    trigger.addEventListener("click", toggle);
    trigger.addEventListener("keydown", event => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      toggle();
    });
    group.addEventListener("pointerenter", () => {
      if (finePointer.matches) open(group);
    });
    group.addEventListener("pointerleave", () => {
      if (finePointer.matches && !group.contains(document.activeElement)) close(group);
    });
    group.addEventListener("focusout", () => setTimeout(() => {
      if (!group.contains(document.activeElement)) close(group);
    }));
    group.addEventListener("keydown", event => {
      if (event.key !== "Escape") return;
      event.preventDefault();
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
            f'<div class="nav-submenu" id="nav-submenu-{key}" hidden>{children}</div></div>'
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


def _era_for_year(year: int) -> str:
    for name, lo, hi in trends.ERAS:
        if lo <= int(year) <= hi:
            return name
    return "outside corpus"


def build_quality_tables(data_dir: Path, public_dir: Path) -> dict:
    primary = pd.read_parquet(data_dir / "llm_annotations" / "paragraph_annotations.parquet")
    second = pd.read_parquet(data_dir / "llm_annotations" / "paragraph_annotations__opus4-8.parquet")
    sample = json.loads(
        (data_dir / "llm_annotations" / "agreement_sample_v1.json").read_text()
    )
    sampled_docs = set(sample["doc_names"])
    expected = primary[primary.doc_name.isin(sampled_docs)][
        ["doc_name", "para_idx"]
    ].copy()
    completion = expected.merge(
        second[["doc_name", "para_idx"]].assign(completed=True),
        on=["doc_name", "para_idx"],
        how="left",
        validate="one_to_one",
    )
    completion["completed"] = completion.completed.fillna(False)
    paired = primary.merge(second, on=["doc_name", "para_idx"], suffixes=("_primary", "_second"),
                           validate="one_to_one")
    speeches = pd.read_parquet(data_dir / "speeches.parquet")
    completion = completion.merge(
        speeches[["doc_name", "president", "year", "title"]],
        on="doc_name",
        validate="many_to_one",
    )
    completion["era"] = completion.year.map(_era_for_year)
    missing_by_speech = (
        completion.groupby(["doc_name", "president", "year", "era", "title"])
        .completed.agg(expected_paragraphs="size", completed_paragraphs="sum")
        .reset_index()
    )
    missing_by_speech["missing_paragraphs"] = (
        missing_by_speech.expected_paragraphs -
        missing_by_speech.completed_paragraphs
    )
    missing_by_speech["completion_rate"] = (
        missing_by_speech.completed_paragraphs /
        missing_by_speech.expected_paragraphs
    )
    missing_by_speech.to_csv(public_dir / "second_model_by_speech.csv", index=False)
    missing_by_era = (
        completion.groupby("era").completed
        .agg(expected_paragraphs="size", completed_paragraphs="sum")
        .reset_index()
    )
    missing_by_era["missing_paragraphs"] = (
        missing_by_era.expected_paragraphs - missing_by_era.completed_paragraphs
    )
    missing_by_era["completion_rate"] = (
        missing_by_era.completed_paragraphs / missing_by_era.expected_paragraphs
    )
    missing_by_era.to_csv(public_dir / "second_model_by_era.csv", index=False)
    paired = paired.merge(speeches[["doc_name", "year", "president"]], on="doc_name",
                          validate="many_to_one")
    paired["era"] = paired.year.map(_era_for_year)
    confusion = pd.crosstab(paired.zero_sum_primary, paired.zero_sum_second,
                            rownames=["primary"], colnames=["second"]).reindex(
                                index=[False, True], columns=[False, True], fill_value=0)
    confusion_long = confusion.stack().rename("n").reset_index()
    confusion_long.to_csv(public_dir / "zero_sum_confusion.csv", index=False)
    kappas = []
    for era, group in [("overall", paired), *list(paired.groupby("era"))]:
        kappas.append({"era": era, "n": len(group),
                       "primary_positive_rate": float(group.zero_sum_primary.mean()),
                       "second_positive_rate": float(group.zero_sum_second.mean()),
                       "kappa": float(cohen_kappa_score(group.zero_sum_primary,
                                                       group.zero_sum_second))})
    pd.DataFrame(kappas).to_csv(public_dir / "zero_sum_by_era.csv", index=False)
    paragraphs = pd.read_parquet(data_dir / "paragraphs.parquet")
    disagreements = paired[
        paired.zero_sum_primary.ne(paired.zero_sum_second)
    ][["doc_name", "para_idx", "president", "year", "era",
       "zero_sum_primary", "zero_sum_second"]].merge(
        paragraphs[["doc_name", "para_idx", "text"]],
        on=["doc_name", "para_idx"],
        validate="one_to_one",
    )
    disagreements.to_csv(public_dir / "zero_sum_disagreement_excerpts.csv", index=False)

    e1 = pd.read_parquet(data_dir / "llm_annotations" / "paragraph_entities.parquet")
    e2 = pd.read_parquet(data_dir / "llm_annotations" / "paragraph_entities__opus4-8.parquet")
    key = ["doc_name", "para_idx"]
    sets1 = e1.groupby(key).entity.apply(lambda s: set(x.casefold() for x in s))
    sets2 = e2.groupby(key).entity.apply(lambda s: set(x.casefold() for x in s))
    keys = sets1.index.union(sets2.index)
    entity_rows = []
    for item in keys:
        left, right = sets1.get(item, set()), sets2.get(item, set())
        union = left | right
        entity_rows.append({"doc_name": item[0], "para_idx": item[1],
                            "jaccard": len(left & right) / len(union) if union else 1.0,
                            "primary_only": len(left - right), "second_only": len(right - left),
                            "matched": len(left & right)})
    entity_audit = pd.DataFrame(entity_rows)
    entity_audit.to_csv(public_dir / "entity_name_agreement.csv", index=False)
    left_entities = e1.assign(entity_normalized=e1.entity.str.casefold()).drop_duplicates(
        key + ["entity_normalized"]
    )
    right_entities = e2.assign(entity_normalized=e2.entity.str.casefold()).drop_duplicates(
        key + ["entity_normalized"]
    )
    matched_entities = left_entities.merge(
        right_entities, on=key + ["entity_normalized"],
        suffixes=("_primary", "_second"), validate="one_to_one",
    )
    matched_entities["stance_agrees"] = matched_entities.stance_primary.eq(
        matched_entities.stance_second
    )
    matched_entities.to_csv(public_dir / "entity_stance_matches.csv", index=False)
    alias_rows = [
        {"canonical_president": president, "alias": alias}
        for president, aliases in invocations.alias_registry().items()
        for alias in aliases
    ]
    pd.DataFrame(alias_rows).to_csv(public_dir / "president_alias_examples.csv", index=False)

    bands = pd.read_parquet(data_dir / "bands.parquet")
    band_widths = bands.assign(
        sampling_width=bands.hi_sampling - bands.lo_sampling,
        combined_width=bands.hi - bands.lo,
    )
    decomposition = (
        band_widths.groupby(
            ["surface", "ci_components", "disagreement_status"], dropna=False
        ).agg(
            n_cells=("series", "size"),
            n_resolved_sampling=("sampling_width", "count"),
            n_resolved_combined=("combined_width", "count"),
            mean_sampling_width=("sampling_width", "mean"),
            mean_combined_width=("combined_width", "mean"),
            mean_disagreement_half_width=("disagreement_half_width", "mean"),
        ).reset_index()
    )
    decomposition.to_csv(public_dir / "uncertainty_decomposition.csv", index=False)
    thin = speeches.groupby("president").size().rename("n_speeches").reset_index()
    thin["thin_record"] = thin.n_speeches < 5
    thin.to_csv(public_dir / "president_coverage.csv", index=False)
    speech_era = speeches.assign(era=speeches.year.map(_era_for_year))
    speech_era.groupby("era").agg(
        n_speeches=("doc_name", "nunique"),
        n_presidents=("president", "nunique"),
        n_words=("word_count", "sum"),
    ).reset_index().to_csv(public_dir / "era_coverage.csv", index=False)
    topic_behavior = primary[["doc_name", "para_idx", "topics"]].merge(
        speeches[["doc_name", "year"]], on="doc_name", validate="many_to_one"
    )
    topic_behavior["era"] = topic_behavior.year.map(_era_for_year)
    topic_behavior["n_topics"] = topic_behavior.topics.map(lambda value: len(list(value)))
    taxonomy_behavior = topic_behavior.groupby("era").n_topics.agg(
        n_paragraphs="size",
        unlabeled_paragraphs=lambda values: int(values.eq(0).sum()),
        multi_label_paragraphs=lambda values: int(values.gt(1).sum()),
        mean_labels="mean",
    ).reset_index()
    taxonomy_behavior.to_csv(public_dir / "taxonomy_label_behavior.csv", index=False)
    missing_rows = int((~completion.completed).sum())
    affected_speeches = int(
        missing_by_speech.loc[
            missing_by_speech.missing_paragraphs.gt(0), "doc_name"
        ].nunique()
    )
    entity_union = int(entity_audit[["primary_only", "second_only", "matched"]].sum().sum())
    matched_total = int(entity_audit.matched.sum())
    entity_bins = (
        pd.cut(
            entity_audit.jaccard,
            bins=[-0.001, 0.25, 0.5, 0.75, 0.999, 1.001],
            labels=["0–.25", ".25–.50", ".50–.75", ".75–<1", "1 exact"],
        ).value_counts(sort=False).rename_axis("agreement_bin").rename("paragraphs").reset_index()
    )
    era_coverage = speech_era.groupby("era").agg(
        n_speeches=("doc_name", "nunique"),
        n_presidents=("president", "nunique"),
        n_words=("word_count", "sum"),
    ).reset_index()
    both_no = int(confusion.loc[False, False])
    second_only = int(confusion.loc[False, True])
    primary_only = int(confusion.loc[True, False])
    both_yes = int(confusion.loc[True, True])
    primary_positives = primary_only + both_yes
    second_positives = second_only + both_yes
    grammar.build()
    era_pos = pd.read_csv(grammar.ERA_POS_PATH)
    era_pos.to_csv(public_dir / "era_part_of_speech.csv", index=False)
    return {
        "paired": len(paired), "sampled": len(expected),
        "missing": missing_rows, "affected_speeches": affected_speeches,
        "zero_sum_kappa": kappas[0]["kappa"],
        "zero_sum_primary": kappas[0]["primary_positive_rate"],
        "zero_sum_second": kappas[0]["second_positive_rate"],
        "zero_sum_disagreements": len(disagreements),
        "zero_sum_both_no": both_no,
        "zero_sum_second_only": second_only,
        "zero_sum_primary_only": primary_only,
        "zero_sum_both_yes": both_yes,
        "zero_sum_raw_agreement": (both_no + both_yes) / len(paired),
        "zero_sum_primary_confirmation": (
            both_yes / primary_positives if primary_positives else 1.0
        ),
        "zero_sum_second_confirmation": (
            both_yes / second_positives if second_positives else 1.0
        ),
        "entity_jaccard": float(entity_audit.jaccard.mean()),
        "entity_match_rate": matched_total / entity_union if entity_union else 1.0,
        "entity_stance_agreement": float(matched_entities.stance_agrees.mean()),
        "thin_presidents": int(thin.thin_record.sum()),
        "unlabeled_paragraphs": int(primary.topics.map(lambda x: len(list(x)) == 0).sum()),
        "_charts": {
            "completion": json.loads(missing_by_era.to_json(orient="records")),
            "confusion": confusion.astype(int).values.tolist(),
            "zero_sum_era": json.loads(pd.DataFrame(kappas).to_json(orient="records")),
            "entities": json.loads(entity_bins.to_json(orient="records")),
            "uncertainty": json.loads(decomposition.to_json(orient="records")),
            "taxonomy": json.loads(taxonomy_behavior.to_json(orient="records")),
            "era_coverage": json.loads(era_coverage.to_json(orient="records")),
            "part_of_speech": json.loads(era_pos.to_json(orient="records")),
        },
    }


def _metric_page() -> str:
    lessons = []
    for name, metric in metrics.METRICS.items():
        lessons.append(
            f'<section id="{name}"><h2>{html.escape(metric["label"])}</h2>'
            f'<p><strong>{html.escape(metric["question"])}</strong></p>'
            f'{metrics.lesson_html(name)}'
            f'<p class="sr-note">Source: {html.escape(metric["source"])} · '
            f'Statistical status: {html.escape(metric["status"])}</p></section>')
    return _page("Metric lessons",
                 "Short, high-school-level explanations first; formulas remain optional.",
                 "".join(lessons), current_href="methodology.html")


def _quality_page(summary: dict) -> str:
    body = f"""
<style>
.quality-chart{{height:410px;margin-top:14px}}.quality-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.quality-grid .quality-chart{{height:390px}}@media(max-width:820px){{.quality-grid{{grid-template-columns:1fr}}}}
.quality-context{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}}
.quality-context article{{border:1px solid var(--border);border-radius:12px;padding:14px;background:var(--surface)}}
.quality-context strong{{display:block;margin-bottom:5px}}.quality-context p{{font-size:.84rem;margin:0}}
.quality-context .good{{border-top:4px solid #4f8a68}}.quality-context .caution{{border-top:4px solid #b17a43}}
.quality-context .meaning{{border-top:4px solid #2d6f9d}}.matrix-reading{{font-size:1rem;line-height:1.65}}
@media(max-width:820px){{.quality-context{{grid-template-columns:1fr}}}}
</style>
<p class="note"><strong>This page is an argument about how much confidence the evidence
deserves—not a scorecard declaring a model “accurate.”</strong> Neither model is human
ground truth. Agreement tells us whether a conclusion is stable across instruments;
disagreement tells us where wording, rubric boundaries, or model behavior can materially
change the result.</p>
<div class="audit-grid">
<div class="audit-card"><strong>{summary['paired']:,} / {summary['sampled']:,}</strong>second-model sampled paragraphs completed</div>
<div class="audit-card"><strong>{summary['missing']:,} across {summary['affected_speeches']} speeches</strong>missing second-model rows</div>
<div class="audit-card"><strong>{summary['zero_sum_kappa']:.2f}</strong>overall zero-sum Cohen’s κ</div>
<div class="audit-card"><strong>{summary['entity_jaccard']:.2f}</strong>mean entity name-set Jaccard</div>
</div>
<section><h2>Second-model completion</h2><p>The sampled comparison expected
{summary['sampled']:,} paragraphs; {summary['paired']:,} completed and
{summary['missing']:,} did not. Missing rows remain explicit and are broken out by speech
and era. This is {summary['paired']/summary['sampled']:.1%} completion—high enough to
support a broad audit, but the {summary['missing']:,} missing rows are concentrated in
{summary['affected_speeches']} speeches, so a speech-specific comparison can still be
incomplete.</p>
<div class="quality-context"><article class="good"><strong>What is good</strong><p>Almost
all sampled paragraphs received a second reading, and every missing row remains visible
rather than silently becoming a negative label.</p></article>
<article class="caution"><strong>What is weak</strong><p>Completion alone says nothing
about whether the second judgments are correct, and concentrated missingness can matter
for individual speeches.</p></article>
<article class="meaning"><strong>What it implies</strong><p>Use the audit for corpus and
era patterns; inspect the per-speech table before making a claim about one address.</p></article></div>
<p><a href="data/quality/second_model_by_speech.csv" download>Missingness by
speech CSV</a> · <a href="data/quality/second_model_by_era.csv" download>Missingness by
era CSV</a></p><div id="quality-completion" class="quality-chart" aria-label="Second-model completion by era"></div></section>
<section><h2>Zero-sum disagreement</h2>
<p>The two model positive rates are {summary['zero_sum_primary']:.1%} and
{summary['zero_sum_second']:.1%}: the second model labels fewer than half as many
paragraphs positive. The matrix contains {summary['zero_sum_both_no']:,} shared negatives,
{summary['zero_sum_both_yes']:,} shared positives, {summary['zero_sum_primary_only']:,}
primary-only positives, and {summary['zero_sum_second_only']:,} second-only positives.</p>
<p class="matrix-reading">That produces <strong>{summary['zero_sum_raw_agreement']:.1%}
raw agreement</strong>, which sounds excellent because the label is rare and both models
usually say “no.” Cohen’s κ falls to <strong>{summary['zero_sum_kappa']:.2f}</strong> after
accounting for that easy negative agreement. More importantly, the second model confirms
only <strong>{summary['zero_sum_primary_confirmation']:.1%}</strong> of primary positives,
while the primary confirms <strong>{summary['zero_sum_second_confirmation']:.1%}</strong>
of second-model positives.</p>
<p><strong>Do not call the off-diagonal cells false positives or false negatives.</strong>
There is no human gold label here. They are primary-only and second-only judgments. The
asymmetry does show that the second model is much more conservative.</p>
<div class="quality-context"><article class="good"><strong>What is good</strong><p>The
models strongly agree on most ordinary paragraphs, and shared positives provide a
high-confidence subset.</p></article>
<article class="caution"><strong>What is weak</strong><p>Positive identification is not
stable: 437 primary positives disappear under the second model, and κ varies sharply by
era.</p></article>
<article class="meaning"><strong>What it implies</strong><p>Absolute zero-sum rates and
president rankings are model-sensitive. Combined uncertainty envelopes and disagreement
excerpts should accompany substantive claims.</p></article></div>
<p><a href="data/quality/zero_sum_confusion.csv" download>Confusion matrix CSV</a> ·
<a href="data/quality/zero_sum_by_era.csv" download>κ by era CSV</a> ·
<a href="data/quality/zero_sum_disagreement_excerpts.csv" download>Disagreement excerpts
CSV</a></p><div class="quality-grid"><div id="quality-confusion" class="quality-chart"
aria-label="Zero-sum confusion matrix"></div><div id="quality-kappa" class="quality-chart"
aria-label="Zero-sum kappa and label rates by era"></div></div>
{metrics.lesson_html("kappa")}</section>
<section><h2>Entity extraction: two LLM passes, not CorEx</h2>
<p><strong>CorEx does not create named entities.</strong> It produces topic activations
from word patterns. This audit compares the primary LLM entity extractor with the second
LLM extractor on the sampled paragraphs.</p>
<p>Name-set agreement counts matched and unmatched normalized names paragraph by
paragraph. {summary['entity_match_rate']:.1%} of the
combined name entries match after case normalization; among matches, stance agrees
{summary['entity_stance_agreement']:.1%} of the time. Mean paragraph-level name-set
Jaccard is {summary['entity_jaccard']:.2f}. These numbers separate two failure modes:
finding a different set of names and assigning a different stance to the same name.</p>
<div class="quality-context"><article class="good"><strong>What is good</strong><p>Stance
agreement is evaluated only after the name itself matches, so extraction and judgment are
not conflated.</p></article><article class="caution"><strong>What is weak</strong><p>Case
normalization is not full alias resolution; “President Roosevelt” and a full name can
still require registry logic.</p></article><article class="meaning"><strong>What it
implies</strong><p>Entity counts are useful for recurring adversaries and audit queues,
not exact rankings of every named person or institution.</p></article></div>
<p><a href="data/quality/entity_name_agreement.csv" download>Name-set audit CSV</a> ·
<a href="data/quality/entity_stance_matches.csv" download>Matched-name stance CSV</a> ·
<a href="data/quality/president_alias_examples.csv" download>President alias examples
CSV</a></p><div id="quality-entities" class="quality-chart"
aria-label="Entity name-set agreement distribution"></div></section>
<section><h2>Grammar change: a new coarse part-of-speech layer</h2>
<p>The repository now includes speech-level spaCy counts for twelve coarse parts of
speech. The chart follows noun, proper-noun, verb, auxiliary, and pronoun shares by era.
This can reveal broad shifts in the grammatical surface—for example, whether formal
documents lean more heavily on nouns while spoken genres use more pronouns—but it is not
a model of argument structure or sentence syntax.</p>
<div class="quality-context"><article class="good"><strong>What is good</strong><p>Every
speech is tagged with the same frozen tag set, and raw counts remain downloadable.</p></article>
<article class="caution"><strong>What is weak</strong><p>Historical spelling, OCR,
quotations, and changing speech genres can alter tag rates; a coarse POS tag cannot
identify clauses or rhetorical structure.</p></article>
<article class="meaning"><strong>What it implies</strong><p>Treat these curves as a new
descriptive lead. Any causal claim about broadcast media or presidential style needs a
genre-controlled follow-up.</p></article></div>
<p><a href="data/quality/era_part_of_speech.csv" download>Era POS shares CSV</a></p>
<div id="quality-pos" class="quality-chart" aria-label="Part-of-speech shares by era"></div>
</section>
<section><h2>What widens an uncertainty envelope?</h2>
<p>Sampling-only width is kept separate from the combined width after measured
annotator disagreement is added. A larger combined bar means label disagreement—not
extra speeches—accounts for the added uncertainty. This is good practice because it
prevents model disagreement from masquerading as ordinary sampling noise. It is still
not a complete error model: shared model biases and taxonomy mistakes can affect both
annotators in the same direction and therefore remain invisible.</p>
<div id="quality-uncertainty" class="quality-chart"
aria-label="Sampling and combined uncertainty widths"></div></section>
<section><h2>Coverage, taxonomy, and provenance</h2>
<ul><li>{summary['thin_presidents']} presidents have fewer than five corpus speeches and are not
ranked as precise outliers.</li><li>{summary['unlabeled_paragraphs']:,} primary-model paragraphs
carry no topic; multi-label paragraphs retain all normalized labels.</li><li>Every public
derived table has a versioned manifest and stable semantic keys.</li></ul>
<div class="quality-context"><article class="good"><strong>What is good</strong><p>Thin
records remain visible, empty topic lists stay in denominators, and multi-label paragraphs
are not forced into one box.</p></article><article class="caution"><strong>What is weak</strong>
<p>The corpus is not a census of presidential communication. Early eras and short-lived
presidents are thinner, while rallies, posts, and many informal remarks are absent.</p></article>
<article class="meaning"><strong>What it implies</strong><p>Read trends as the history of
this formal-speech corpus. Do not generalize automatically to everything a president said
or everything the public heard.</p></article></div>
<p><a href="data/quality/president_coverage.csv" download>President coverage CSV</a> ·
<a href="data/quality/era_coverage.csv" download>Era coverage CSV</a> ·
<a href="data/quality/taxonomy_label_behavior.csv" download>Taxonomy behavior CSV</a> ·
<a href="data/quality/uncertainty_decomposition.csv" download>Uncertainty decomposition CSV</a></p>
<div class="quality-grid"><div id="quality-taxonomy" class="quality-chart"
aria-label="Unlabeled and multi-label paragraphs by era"></div>
<div id="quality-era-coverage" class="quality-chart"
aria-label="Speech coverage by era"></div></div>
<details><summary>Processing and cleaning checklist</summary><ol>
<li>Join paragraph tables only on document and paragraph keys.</li><li>Normalize taxonomy
labels case-insensitively and reject unknown labels.</li><li>Preserve empty and excluded
records rather than silently dropping them.</li><li>Resample speeches, not paragraphs.</li>
<li>Keep sampling and annotator-disagreement components separately downloadable.</li></ol></details>
</section>"""
    scripts = f"""<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script><script>
const Q={json.dumps(summary["_charts"])};
const C={{blue:"#2d6f9d",orange:"#b17a43",green:"#4f8a68",muted:"#8a9299",grid:"#e7e9eb"}};
const base={{template:"simple_white",paper_bgcolor:"#fff",plot_bgcolor:"#fff",
 font:{{family:"system-ui",color:"#24313d"}},margin:{{l:62,r:22,t:42,b:82}},
 legend:{{orientation:"h",y:1.12}}}};
Plotly.newPlot("quality-completion",[{{type:"bar",x:Q.completion.map(x=>x.era),
 y:Q.completion.map(x=>x.completion_rate*100),customdata:Q.completion.map(x=>[x.completed_paragraphs,x.expected_paragraphs]),
 marker:{{color:C.blue}},hovertemplate:"%{{x}}<br>%{{y:.1f}}% complete<br>%{{customdata[0]}} of %{{customdata[1]}} paragraphs<extra></extra>"}}],
 {{...base,height:410,yaxis:{{title:"completion (%)",range:[0,101],gridcolor:C.grid}}}},
 {{displayModeBar:false,responsive:true}});
Plotly.newPlot("quality-confusion",[{{type:"heatmap",z:Q.confusion,
 x:["Second: no","Second: yes"],y:["Primary: no","Primary: yes"],text:Q.confusion,
 texttemplate:"%{{text:,}}",colorscale:[[0,"#f7f8f8"],[1,C.blue]],showscale:false,
 hovertemplate:"%{{y}} / %{{x}}<br>%{{z:,}} paragraphs<extra></extra>"}}],
 {{...base,height:390,margin:{{l:100,r:22,t:48,b:65}},title:{{text:"Full confusion matrix",font:{{size:15}}}}}},
 {{displayModeBar:false,responsive:true}});
const z=Q.zero_sum_era;
Plotly.newPlot("quality-kappa",[
 {{type:"bar",name:"Cohen’s κ",x:z.map(x=>x.era),y:z.map(x=>x.kappa),marker:{{color:C.blue}}}},
 {{type:"scatter",mode:"lines+markers",name:"Primary positive rate",x:z.map(x=>x.era),y:z.map(x=>x.primary_positive_rate),line:{{color:C.orange}}}},
 {{type:"scatter",mode:"lines+markers",name:"Second positive rate",x:z.map(x=>x.era),y:z.map(x=>x.second_positive_rate),line:{{color:C.green}}}}
],{{...base,height:390,title:{{text:"Agreement and label prevalence",font:{{size:15}}}},
 yaxis:{{title:"0–1",range:[0,1],gridcolor:C.grid}}}},{{displayModeBar:false,responsive:true}});
Plotly.newPlot("quality-entities",[{{type:"bar",x:Q.entities.map(x=>x.agreement_bin),
 y:Q.entities.map(x=>x.paragraphs),marker:{{color:C.green}},
 hovertemplate:"Name-set Jaccard %{{x}}<br>%{{y:,}} paragraphs<extra></extra>"}}],
 {{...base,height:410,xaxis:{{title:"paragraph-level name-set Jaccard"}},
 yaxis:{{title:"paragraphs",gridcolor:C.grid}}}},{{displayModeBar:false,responsive:true}});
const posOrder=[...new Set(Q.part_of_speech.map(x=>x.era))];
const posNames=["NOUN","PROPN","VERB","AUX","PRON"];
Plotly.newPlot("quality-pos",posNames.map((name,i)=>({{
 type:"scatter",mode:"lines+markers",name,
 x:posOrder,y:posOrder.map(era=>Q.part_of_speech.find(x=>x.era===era&&x.part_of_speech===name)?.share_percent),
 line:{{color:[C.blue,C.orange,C.green,"#8a6b82","#9b4e50"][i]}}
}})),{{...base,height:410,yaxis:{{title:"% of tagged alphabetic tokens",gridcolor:C.grid}}}},
{{displayModeBar:false,responsive:true}});
const u=Q.uncertainty.filter(x=>x.mean_sampling_width!=null||x.mean_combined_width!=null);
const ul=u.map(x=>`${{x.surface}} · ${{x.ci_components}}`);
Plotly.newPlot("quality-uncertainty",[
 {{type:"bar",name:"Sampling-only width",x:ul,y:u.map(x=>x.mean_sampling_width),marker:{{color:C.blue}}}},
 {{type:"bar",name:"Combined width",x:ul,y:u.map(x=>x.mean_combined_width),marker:{{color:C.orange}}}}
],{{...base,height:410,barmode:"group",yaxis:{{title:"mean interval width (percentage points)",gridcolor:C.grid}}}},
{{displayModeBar:false,responsive:true}});
const t=Q.taxonomy;
Plotly.newPlot("quality-taxonomy",[
 {{type:"bar",name:"Unlabeled",x:t.map(x=>x.era),y:t.map(x=>x.unlabeled_paragraphs/x.n_paragraphs*100),marker:{{color:C.muted}}}},
 {{type:"bar",name:"Multiple labels",x:t.map(x=>x.era),y:t.map(x=>x.multi_label_paragraphs/x.n_paragraphs*100),marker:{{color:C.blue}}}}
],{{...base,height:390,barmode:"group",title:{{text:"Label behavior by era",font:{{size:15}}}},
 yaxis:{{title:"% of paragraphs",gridcolor:C.grid}}}},{{displayModeBar:false,responsive:true}});
Plotly.newPlot("quality-era-coverage",[{{type:"bar",x:Q.era_coverage.map(x=>x.era),
 y:Q.era_coverage.map(x=>x.n_speeches),customdata:Q.era_coverage.map(x=>[x.n_presidents,x.n_words]),
 marker:{{color:C.orange}},hovertemplate:"%{{x}}<br>%{{y}} speeches<br>%{{customdata[0]}} presidents<br>%{{customdata[1]:,}} words<extra></extra>"}}],
 {{...base,height:390,title:{{text:"Corpus coverage by era",font:{{size:15}}}},
 yaxis:{{title:"speeches",gridcolor:C.grid}}}},{{displayModeBar:false,responsive:true}});
</script>"""
    return _page(
        "Data quality",
        "What is missing, where models disagree, and how thin records are handled.",
        body, scripts, current_href="data-quality.html",
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
    agreement_path = corpus.DATA_DIR / "method_agreement.parquet"
    overall = pd.read_parquet(agreement_path)
    overall = overall[overall.era.isna()].copy().sort_values("kappa", ascending=False)
    overall["llm_rate"] = overall.n_llm / overall.n * 100
    overall["corex_rate"] = overall.n_corex / overall.n * 100
    chart_rows = overall[
        ["issue", "llm_rate", "corex_rate", "jaccard", "kappa", "n_both"]
    ].round(4).to_dict("records")
    issues_meta = json.loads((corpus.DATA_DIR / "issues_meta.json").read_text())
    from .issues import ISSUE_ANCHORS
    anchors = "".join(
        f"<tr><td>{html.escape(issue)}</td><td>{html.escape(', '.join(words))}</td></tr>"
        for issue, words in ISSUE_ANCHORS.items()
    )
    body = f"""
<style>
.method-contrast{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.method-lane{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px}}
.method-lane h2{{margin-top:0}}.method-lane.corex{{border-top:5px solid #9b6b39}}
.method-lane.llm{{border-top:5px solid #2d6f9d}}.method-steps li{{margin:.65rem 0}}
.model-chart{{height:500px}}.verdict{{border-left:4px solid #8a6d3b;background:#fff8e8;padding:14px 16px}}
@media(max-width:760px){{.method-contrast{{grid-template-columns:1fr}}.model-chart{{height:420px}}}}
</style>
<section><h2>One corpus, two different measuring instruments</h2>
<p>Both systems label the same paragraph-sized units and both allow more than one label.
They differ in what they can “see.” CorEx is a deterministic, anchored word-pattern model.
The LLM layer applies a frozen semantic rubric that can recognize an idea even when its
historic wording changes. Neither is ground truth.</p>
<div class="method-contrast">
<article class="method-lane corex"><h2>CorEx: anchored word patterns</h2>
<p><strong>15 hand-defined issues + 7 free topics.</strong> Sixteen are public after the
free-topic quality gate.</p><ol class="method-steps">
<li>Turn 36,229 paragraphs into binary word-presence vectors.</li>
<li>Remove common English and speech-register words; retain words appearing in at least
20 paragraphs and at most 30% of paragraphs, capped at 25,000 features.</li>
<li>Give each of 15 issues an era-spanning anchor list, with anchor strength 6.</li>
<li>Fit 22 latent factors with seed 42: 15 anchored factors and 7 unsupplied factors.</li>
<li>Store a Boolean activation for every paragraph-topic pair.</li></ol>
<p><strong>Strength:</strong> repeatable, cheap, inspectable vocabulary. <strong>Blind spot:</strong>
renamed concepts and meanings that use none of the anchor vocabulary.</p></article>
<article class="method-lane llm"><h2>LLM: frozen semantic rubric</h2>
<p><strong>50 fine topics inside 17 broad domains.</strong> Labels were derived from
era-isolated corpus samples, frozen, then applied paragraph by paragraph with president,
party, and title hidden.</p><ol class="method-steps">
<li>Use embeddings and era-balanced samples to expose subjects the legacy list missed.</li>
<li>Normalize proposals into a closed 50-topic taxonomy with definitions and boundaries.</li>
<li>Give the model only paragraph text and decade, then allow multi-label assignments.</li>
<li>Re-label a stratified speech sample with a second model to measure disagreement.</li>
<li>Carry sampling and annotator disagreement separately, then combine them into the
default uncertainty envelope.</li></ol>
<p><strong>Strength:</strong> meaning can survive vocabulary drift. <strong>Blind spot:</strong>
subjective rubric judgments, model disagreement, and a more granular taxonomy.</p></article>
</div></section>
<section><h2>Does CorEx create named entities?</h2>
<p><strong>No.</strong> CorEx produces paragraph-topic activations from correlated word
patterns. It does not emit people, countries, organizations, stance, partisan attack,
zero-sum framing, or proposal-versus-values judgments.</p>
<p>Named entities come from a separate LLM extraction pass. The entity agreement numbers
compare that primary LLM pass with a second LLM pass on the same sampled paragraphs—they
are <em>not</em> CorEx-versus-LLM agreement. The audit separates whether the models found
the same normalized names from whether they assigned the same stance after a name
matched. <a href="data-quality.html">See the contextualized entity audit →</a></p></section>
<section><h2>Where they agree—and why a mismatch is not automatically an error</h2>
<p>The 50 fine LLM topics are projected through a declared many-to-many crosswalk onto
the 15 legacy issues. The charts below use all {int(overall.n.iloc[0]):,} comparable
paragraphs. A paragraph can activate several issues in either system.</p>
<div id="model-rates" class="model-chart" aria-label="CorEx and LLM positive rates by issue"></div>
<div id="model-agreement" class="model-chart" aria-label="CorEx and LLM agreement by issue"></div>
{metrics.lesson_html("kappa")}
<details><summary>Inspect the evidence</summary><p>
<a href="data/method-agreement.csv" download>Download paragraph-level agreement summaries</a> ·
<a href="data/method-compositions.csv" download>Download speech-level method compositions</a>.
Jaccard is overlap divided by the union; κ adjusts observed agreement for the two
systems' label rates.</p></details>
<p class="verdict"><strong>The preregistered coherence prediction failed.</strong>
Anchor-word coherence did not predict cross-method agreement (Spearman ρ = −0.054
for Jaccard, no-change surprise 85.0%). Immigration is the clearest counterexample:
its anchor words rarely co-occur, yet its CorEx and LLM labels agree relatively well.
That is evidence that word co-occurrence and label precision are different questions.</p></section>
<section><h2>The exact CorEx anchor vocabulary</h2>
<p>These words guide the factors; they are not a simple keyword OR rule. CorEx also learns
correlated non-anchor words. Low NPMI is published as a caveat, but anchored issues are not
deleted for low coherence because their lists intentionally span vocabulary across eras.</p>
<table><thead><tr><th>Anchored issue</th><th>Declared anchors</th></tr></thead>
<tbody>{anchors}</tbody></table>
<p class="sr-note">Stored model metadata reports {issues_meta["n_paragraphs"]:,} paragraphs,
anchor strength 6, seven free topics, and a fixed random seed of 42.</p></section>
<section><h2>Which view should you use?</h2>
<ul><li><strong>Use CorEx</strong> for the long-run broad issue biographies and a completely
deterministic historical lens.</li><li><strong>Use the LLM taxonomy</strong> for finer subjects,
semantic drift, proposals/values, and paragraph judgments—while keeping its uncertainty
envelopes visible.</li><li><strong>Use the comparison</strong> when a claim depends on one
instrument. Agreement supports robustness; disagreement identifies a place to inspect wording,
taxonomy boundaries, and source paragraphs.</li></ul></section>"""
    scripts = f"""<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<script>
const M={json.dumps(chart_rows)};
const common={{template:"simple_white",paper_bgcolor:"#fff",plot_bgcolor:"#fff",
 font:{{family:"system-ui",color:"#24313d"}},margin:{{l:190,r:24,t:26,b:48}},
 yaxis:{{automargin:true}},legend:{{orientation:"h",y:1.08}}}};
const ordered=[...M].sort((a,b)=>a.llm_rate-b.llm_rate);
Plotly.newPlot("model-rates",[
 {{type:"bar",orientation:"h",name:"LLM projected to issue",y:ordered.map(x=>x.issue),x:ordered.map(x=>x.llm_rate),marker:{{color:"#2d6f9d"}}}},
 {{type:"bar",orientation:"h",name:"CorEx anchored issue",y:ordered.map(x=>x.issue),x:ordered.map(x=>x.corex_rate),marker:{{color:"#b17a43"}}}}
],{{...common,barmode:"group",height:500,xaxis:{{title:"% of paragraphs",rangemode:"tozero",gridcolor:"#e8e8e8"}}}},
{{displayModeBar:false,responsive:true}});
const ranked=[...M].sort((a,b)=>a.kappa-b.kappa);
Plotly.newPlot("model-agreement",[
 {{type:"scatter",mode:"markers",name:"Cohen’s κ",y:ranked.map(x=>x.issue),x:ranked.map(x=>x.kappa),marker:{{size:10,color:"#2d6f9d"}}}},
 {{type:"scatter",mode:"markers",name:"Jaccard",y:ranked.map(x=>x.issue),x:ranked.map(x=>x.jaccard),marker:{{size:9,color:"#b17a43",symbol:"diamond"}}}}
],{{...common,height:500,xaxis:{{title:"agreement (higher = more overlap)",range:[0,1],gridcolor:"#e8e8e8"}}}},
{{displayModeBar:false,responsive:true}});
</script>"""
    return _page(
        "How the label models differ",
        "A detailed, evidence-backed comparison of the anchored CorEx issue model and the LLM semantic taxonomy.",
        body, scripts, current_href="methodology.html",
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
