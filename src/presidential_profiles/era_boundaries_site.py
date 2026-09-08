"""Render the public, artifact-driven explanation of editorial era choices."""

from __future__ import annotations

import html
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Callable

import pandas as pd

from . import era_boundaries, era_profiles, trust_appendices
from .expansion_site import NAV_CSS, nav
from .html_safety import json_for_script
from .site_style import PAGE_CSS


# Historical interpretation lives here; public identifiers, dates, titles, and
# Story anchors come exclusively from era_profiles.ERA_PROFILE_SPECS.
ERA_DECISIONS = {
    "founding": {
        "argument": (
            "The first administrations establish sovereignty, federal capacity, "
            "finance, and governing routines. Ending after Jefferson keeps that "
            "founding problem distinct from Madison's war presidency."
        ),
        "source_label": "Washington's 1789 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "april-30-1789-first-inaugural-address"
        ),
    },
    "expansion": {
        "argument": (
            "Madison's accession supplies an exact presidential boundary inside "
            "the measured transition neighborhood. Diplomacy and renewed conflict "
            "with Britain lead into postwar state-building, removal, annexation, "
            "and continental conquest."
        ),
        "source_label": "Madison's 1809 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "march-4-1809-first-inaugural-address"
        ),
    },
    "civil-war-reconstruction": {
        "argument": (
            "The Mexican-American War ends in 1848, and the Compromise of 1850 "
            "turns the status of the acquired lands into a national governing "
            "settlement—and exposes how little had been settled. The chapter "
            "continues through emancipation, Civil War, and the Johnson conflict."
        ),
        "source_label": "National Archives: Compromise of 1850",
        "source_url": "https://www.archives.gov/milestone-documents/compromise-of-1850",
    },
    "gilded-age": {
        "argument": (
            "Grant's accession is both the primary four-year-grid start and a "
            "presidential boundary. Reconstruction continues, while the measured "
            "agenda turns toward administration, trade, and industrial government."
        ),
        "source_label": "Grant's 1869 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "march-4-1869-first-inaugural-address"
        ),
    },
    "progressives-depression": {
        "argument": (
            "Wilson's accession supplies a clean institutional and presidential "
            "start for reform, World War I, the interwar order, and its collapse."
        ),
        "source_label": "Wilson's 1913 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "march-4-1913-first-inaugural-address"
        ),
    },
    "war-new-deal": {
        "argument": (
            "FDR's accession is the historically legible break: emergency "
            "government and the New Deal lead into global war, while Truman "
            "carries the wartime state into the first postwar settlement."
        ),
        "source_label": "National Archives: FDR's 1933 inaugural",
        "source_url": "https://www.archives.gov/education/lessons/fdr-inaugural",
    },
    "cold-war": {
        "argument": (
            "Eisenhower's accession and the Korean armistice provide a defensible "
            "start for the mature Cold War order. The period runs through Carter, "
            "keeping the Reagan turn visible rather than splitting at 1989."
        ),
        "source_label": "State Department: Korean armistice, 1953",
        "source_url": (
            "https://history.state.gov/historicaldocuments/frus1952-54v15p2/comp1"
        ),
    },
    "post-cold-war": {
        "argument": (
            "Reagan's accession is a clearer rhetorical and political boundary "
            "than 1989. The longer regime contains the Cold War's end, cable, "
            "permanent campaigning, and early digital communication."
        ),
        "source_label": "Reagan's 1981 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "january-20-1981-first-inaugural-address"
        ),
    },
    "present": {
        "argument": (
            "Trump's accession is the editorial start of an intensification, not "
            "the invention of platform politics. The measured transition is broad; "
            "the incomplete terminal cycle makes this final boundary provisional."
        ),
        "source_label": "Trump's 2017 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "january-20-2017-inaugural-address"
        ),
    },
}

FAMILY_LABELS = {
    "primary": "Primary specification",
    "k_sweep": "Number of segments",
    "minimum_duration": "Minimum duration",
    "grid_anchor": "Cycle-grid anchor",
    "weighting": "Axis weighting",
    "terminal": "Corpus endpoint",
    "axis_omission": "Axis-family omission",
    "common_genre": "Common-genre check",
}

PUBLIC_EXPORT_KEYS = {
    "era-boundary-sliding-scores.csv": ["window_years", "boundary_year"],
    "era-boundary-cluster-stability.csv": ["condition", "k", "boundary_year"],
    "era-boundary-consensus-scores.csv": ["boundary_year"],
    "era-boundary-cycle4-fingerprints.csv": ["grid_offset", "cycle_start"],
    "era-boundary-segmentation.csv": ["condition", "segment_index"],
    "era-boundary-sensitivity.csv": ["condition", "segment_index"],
    "era-boundary-evidence.csv": ["era_key"],
    "era-boundary-drivers.csv": ["era_key", "driver_rank"],
}


def _choices() -> list[dict]:
    if set(ERA_DECISIONS) != {spec.key for spec in era_profiles.ERA_PROFILE_SPECS}:
        raise ValueError("historical era decisions must match ERA_PROFILE_SPECS")
    return [
        {
            "key": spec.key,
            "start": spec.start_year,
            "end": spec.end_year,
            "title": spec.title,
            "label": spec.label,
            "section_key": spec.section_key,
            **ERA_DECISIONS[spec.key],
        }
        for spec in era_profiles.ERA_PROFILE_SPECS
    ]


def _choice_rows(analysis: dict) -> list[dict]:
    evidence = analysis["boundary_evidence"].set_index("era_key")
    rows: list[dict] = []
    for choice in _choices():
        row = dict(choice)
        if choice["key"] == era_profiles.ERA_PROFILE_SPECS[0].key:
            row.update({"score": None, "rank": None, "local_peak": None})
        else:
            audited = evidence.loc[choice["key"]]
            row.update(audited.to_dict())
            row["score"] = float(audited["annual_abruptness_percentile"])
            row["rank"] = int(audited["annual_abruptness_rank"])
            row["local_peak"] = int(audited["local_peak_year"])
        rows.append(row)
    return rows


def _timeline(rows: list[dict]) -> str:
    return "".join(
        f'''<a class="era-chip" href="index.html#{html.escape(row['section_key'], quote=True)}">
<strong>{row['start']}–{row['end']}</strong><span>{html.escape(row['title'])}</span></a>'''
        for row in rows
    )


def _scheme_cards(rows: list[dict]) -> str:
    cards = []
    for row in rows:
        source = (
            f'<a href="{html.escape(row["source_url"], quote=True)}">'
            f'{html.escape(row["source_label"])}</a>'
        )
        if row["start"] == era_profiles.ERA_PROFILE_SPECS[0].start_year:
            data_read = "The corpus starts here; there is no before/after boundary score."
            sensitivity_range = "Not applicable at the corpus start"
        else:
            data_read = (
                f"Annual abruptness rank {row['rank']} of "
                f"{int(row['n_annual_candidates'])}; local peak "
                f"{row['local_peak']}. The primary persistent-regime start is "
                f"{int(row['data_optimal_start'])}. Across "
                f"{int(row['n_estimable_conditions'])} estimable conditions, "
                f"{int(row['n_conditions_within_four_years'])} place a cut within "
                "four years of this grid neighborhood."
            )
            sensitivity_range = (
                f"{int(row['sensitivity_start_min'])}–"
                f"{int(row['sensitivity_start_max'])}"
            )
        provisional = (
            '<span class="provisional">Provisional · right-censored</span>'
            if row.get("right_censored") else ""
        )
        cards.append(
            f'''<article class="era-decision" id="decision-{html.escape(row['key'])}">
<div class="decision-year"><strong>{row['start']}–{row['end']}</strong>{provisional}</div>
<div><h3><a href="index.html#{html.escape(row['section_key'], quote=True)}">{html.escape(row['title'])}</a></h3>
<p>{html.escape(row['argument'])}</p><p class="receipt-line">Historical receipt: {source}</p></div>
<div><strong>What the corpus says</strong><p>{html.escape(data_read)}</p>
<span class="zone">Segmentation sensitivity range: {sensitivity_range}</span></div></article>'''
        )
    return "".join(cards)


def _sensitivity_summary(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, frame in sensitivity.groupby("condition", sort=False):
        first = frame.iloc[0]
        starts = (
            ", ".join(map(str, frame[frame["status"] == "computed"]["segment_start"].astype(int)))
            if first["status"] == "computed" else "Not estimable"
        )
        rows.append({
            "condition": condition,
            "label": first["label"],
            "family": first["analysis_family"].split(";")[0],
            "status": first["status"],
            "starts": starts,
            "similarity": (
                float(first["partition_similarity_to_primary"])
                if first["status"] == "computed" else None
            ),
            "reason": first["status_reason"],
        })
    return pd.DataFrame(rows)


def _sensitivity_fallback(summary: pd.DataFrame) -> str:
    body = []
    for row in summary.to_dict("records"):
        similarity = "—" if row["similarity"] is None else f"{row['similarity']:.3f}"
        note = row["reason"] if row["status"] != "computed" else ""
        body.append(
            f"<tr><th scope=\"row\">{html.escape(row['label'])}</th>"
            f"<td>{html.escape(FAMILY_LABELS.get(row['family'], row['family']))}</td>"
            f"<td>{html.escape(row['starts'])}</td><td>{similarity}</td>"
            f"<td>{html.escape(note)}</td></tr>"
        )
    return "".join(body)


def _annual_fallback(consensus: pd.DataFrame) -> str:
    selected = consensus.sort_values("boundary_year", kind="stable")
    return "".join(
        f"<tr><th scope=\"row\">{int(row.boundary_year)}</th>"
        f"<td>{int(row.consensus_rank)}</td><td>{row.consensus * 100:.1f}</td>"
        f"<td>{int(row.rank_4)}</td><td>{int(row.rank_8)}</td></tr>"
        for row in selected.itertuples()
    )


def _driver_label(axis: str) -> str:
    label = axis.split("__", 1)[-1].split("_", 1)[-1]
    return label.replace("_", " ").title()


def _boundary_fallback(evidence: pd.DataFrame, drivers: pd.DataFrame) -> str:
    cards = []
    for row in evidence.to_dict("records"):
        subset = drivers[drivers["era_key"] == row["era_key"]]
        decision = ERA_DECISIONS[row["era_key"]]
        receipt = (
            f'<a href="{html.escape(decision["source_url"], quote=True)}">'
            f'{html.escape(decision["source_label"])}</a>'
        )
        driver_rows = "".join(
            f"<tr><th scope=\"row\">{html.escape(_driver_label(driver.axis))}</th>"
            f"<td>{html.escape(driver.axis_group)}</td>"
            f"<td>{driver.z_delta_equal_cycle:+.2f} SD</td>"
            f"<td>{driver.left_equal_cycle_value:.3f} → {driver.right_equal_cycle_value:.3f}</td>"
            f"<td>{driver.left_story_pooled_value:.3f} → {driver.right_story_pooled_value:.3f}</td></tr>"
            for driver in subset.itertuples()
        )
        provisional = " · provisional/right-censored" if row["right_censored"] else ""
        cards.append(
            f'''<details id="boundary-{int(row['editorial_start'])}"><summary>
{int(row['editorial_start'])} · {html.escape(row['era_title'])}{provisional}</summary>
<div class="inspector-grid"><p><strong>Editorial / primary model</strong><br>
{int(row['editorial_start'])} / {int(row['data_optimal_start'])}</p>
<p><strong>Annual mountain</strong><br>rank {int(row['annual_abruptness_rank'])} of {int(row['n_annual_candidates'])};
local peak {int(row['local_peak_year'])}</p>
<p><strong>Adjacent support</strong><br>{int(row['n_left_speeches'])} / {int(row['n_right_speeches'])}
speeches; {int(row['n_left_paragraphs']):,} / {int(row['n_right_paragraphs']):,} paragraphs</p>
<p><strong>Sensitivity / fit cost</strong><br>{int(row['n_conditions_within_four_years'])} of {int(row['n_estimable_conditions'])} estimable conditions nearby;
{float(row['penalty_pct_of_optimum']):.3f}% of optimum</p></div>
<p class="receipt-line">Historical receipt: {receipt}</p>
<div class="table-wrap" role="region" aria-label="Signed drivers for this boundary" tabindex="0"><table class="driver-table"><thead><tr><th>Top signed driver</th><th>Family</th>
<th>Equal-cycle change</th><th>Equal-cycle values</th><th>Pooled Story-era values</th></tr></thead>
<tbody>{driver_rows}</tbody></table></div></details>'''
        )
    return "".join(cards)


def _matrix_payload(sensitivity: pd.DataFrame) -> list[dict]:
    records = []
    for condition, frame in sensitivity.groupby("condition", sort=False):
        first = frame.iloc[0]
        if first["status"] != "computed":
            continue
        for row in frame[frame["is_boundary"]].itertuples():
            records.append({
                "condition": condition,
                "label": first["label"],
                "family": first["analysis_family"].split(";")[0],
                "start": int(row.segment_start),
                "similarity": float(row.partition_similarity_to_primary),
            })
    return records


def render_era_boundaries(analysis: dict | None = None) -> str:
    analysis = era_boundaries.load() if analysis is None else analysis
    consensus = analysis["consensus_scores"].copy()
    sensitivity = analysis["sensitivity"].copy()
    evidence = analysis["boundary_evidence"].copy()
    drivers = analysis["boundary_drivers"].copy()
    meta = analysis["meta"]
    rows = _choice_rows(analysis)
    summary = _sensitivity_summary(sensitivity)
    computed = summary[summary["status"] == "computed"]
    unavailable = summary[summary["status"] != "computed"]
    common_genre_note = (
        html.escape(unavailable.iloc[0]["reason"])
        if not unavailable.empty else "Computed successfully."
    )
    reviewed_starts = [spec.start_year for spec in era_profiles.ERA_PROFILE_SPECS[1:]]
    primary_starts = ", ".join(map(str, meta["data_optimal_grid_starts"]))
    candidate_count = len(consensus)
    annual_start = int(consensus["boundary_year"].min())
    annual_end = int(consensus["boundary_year"].max())
    terminal_label = meta["right_censoring"]["terminal_cycle"]
    sliding_payload = consensus[[
        "boundary_year", "percentile_4", "percentile_8", "consensus",
        "rank_4", "rank_8", "consensus_rank",
    ]].to_dict("records")
    chosen_payload = evidence[[
        "editorial_start", "annual_abruptness_percentile",
        "annual_abruptness_rank", "data_optimal_start",
    ]].to_dict("records")
    terminal_rows = sensitivity[
        (sensitivity["condition"] == "complete_cycles_only")
        & sensitivity["is_boundary"]
    ]
    terminal_start = int(terminal_rows["segment_start"].max())
    rows_by_key = {row["key"]: row for row in rows}
    expansion = rows_by_key["expansion"]
    civil_war = rows_by_key["civil-war-reconstruction"]
    war_new_deal = rows_by_key["war-new-deal"]
    present = rows_by_key["present"]
    legacy_primary = analysis["cluster_stability"]
    legacy_primary = legacy_primary[
        (legacy_primary["condition"] == "full")
        & legacy_primary["is_primary_k"]
    ]
    legacy_gilded_start = int(
        legacy_primary.loc[
            (legacy_primary["boundary_year"] - rows_by_key["gilded-age"]["editorial_start"])
            .abs()
            .idxmin(),
            "boundary_year",
        ]
    )
    adjustment_summary = ", ".join(
        f"{int(item['data_start'])}→{int(item['reviewed_start'])}"
        for item in meta["reviewed_adjustments"]
    )
    era_receipt = trust_appendices.trust_receipt(
        claim="The nine Story eras are historically reviewed editorial periods tested against deterministic segmentation alternatives.",
        population=f"{meta['corpus_fingerprint']['n_speeches']:,} corpus speeches through {era_profiles.ERA_PROFILE_SPECS[-1].end_year}",
        method=f"{meta['n_axes']} governed axes; {meta['n_sensitivity_conditions']} declared sensitivity conditions",
        uncertainty="Specification sensitivity and right-censor status, not a sampling confidence interval",
        artifact="data/era-boundaries-v3/manifest.json",
        status="Current Story scheme; final era provisional/right-censored",
        limitation="The feature system and editorial constraints do not discover an inevitable historical partition",
        downstream_use="Story chronology, era-calibrated summaries, and periodized public comparisons",
    )

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>How we chose the eras · Presidential Profiles</title>
<meta name="description" content="A reproducible audit of the data, sensitivity tests, and historical judgment behind the Story's era boundaries.">
<style>{PAGE_CSS}
{NAV_CSS}
header,main,footer{{max-width:1120px}}header{{padding-top:48px}}
.kicker{{font-size:.76rem;letter-spacing:.1em;text-transform:uppercase;font-weight:780;color:#32658d}}
header h1{{font-size:clamp(2.35rem,6vw,4.8rem);line-height:1.01;max-width:900px;margin-top:12px}}
header .sub{{font-size:1.08rem;max-width:820px}}
.status,.provisional,.zone{{display:inline-block;padding:6px 10px;border-radius:999px;font-size:.75rem;font-weight:740}}
.status{{margin-top:18px;background:#e3f2e8;color:#22583a}}.provisional{{margin-top:7px;background:#fff0db;color:#754a12}}
.zone{{background:#edf4f8;color:#244e6e;padding:4px 8px}}
.local-nav{{position:sticky;top:var(--global-nav-height,0);z-index:5;display:flex;gap:8px;overflow-x:auto;padding:10px 0;background:rgba(250,249,246,.96);border-bottom:1px solid var(--border)}}
.local-nav a{{white-space:nowrap;padding:7px 10px;border-radius:999px;background:#fff;border:1px solid var(--border);font-size:.78rem;text-decoration:none}}
main section[id],details[id],.era-decision[id]{{scroll-margin-top:190px}}.local-nav .scroll-cue{{display:none;
white-space:nowrap;padding:7px 9px;border-radius:999px;background:#edf4f8;color:#244e6e;font-size:.7rem;font-weight:750}}
.answer{{background:#172a38;color:#fff;border-radius:20px;padding:24px;margin-top:28px}}
.answer h2{{font-size:1.42rem}}.answer p{{color:#d9e2e8;max-width:870px;margin-top:8px}}
.era-strip{{display:grid;grid-template-columns:repeat(9,minmax(0,1fr));gap:5px;margin-top:18px;overflow-x:auto;padding-bottom:8px}}
.era-chip{{color:#fff;text-decoration:none;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.16);border-radius:10px;padding:10px}}
.era-chip strong,.era-chip span{{display:block}}.era-chip strong{{font-size:.76rem;overflow-wrap:anywhere}}.era-chip span{{font-size:.68rem;color:#d9e2e8;margin-top:4px;line-height:1.3}}.timeline-cue{{display:none;color:#d9e2e8;font-size:.72rem;font-weight:750;margin-top:12px}}
.verdict-grid,.principles,.receipt-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}}
.verdict,.principle,.receipt{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px}}
.verdict strong,.principle strong,.receipt strong,.receipt code{{display:block}}.verdict p,.principle p,.receipt p{{font-size:.86rem;margin:6px 0 0}}
.trust-receipt{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--border);border:1px solid var(--border);border-radius:14px;overflow:hidden;margin:20px 0}}.trust-receipt div{{background:#fff;padding:13px;min-width:0}}.trust-receipt dt{{color:#52616c;font-size:.68rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase}}.trust-receipt dd{{margin:.35rem 0 0;font-size:.8rem;overflow-wrap:anywhere}}
.chart-card{{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:18px;margin-top:18px}}
.chart-head{{display:flex;justify-content:space-between;align-items:end;gap:16px;flex-wrap:wrap}}.chart-head p{{margin:5px 0 0}}
.boundary-chart{{width:100%;height:500px}}#sensitivity-chart{{height:760px}}
.chart-note,.receipt-line{{font-size:.82rem;color:var(--muted);margin-top:8px}}
.callout{{border-left:4px solid #6b9fc5;background:#edf4f8;border-radius:0 12px 12px 0;padding:15px 17px;margin:20px 0}}.callout p{{margin:0;max-width:none}}
.warning{{border-left-color:#c98636;background:#fff4e7}}code{{font-size:.88em}}
.table-wrap{{overflow-x:auto;border:1px solid var(--border);border-radius:12px;margin-top:12px;background:var(--surface)}}.table-wrap:focus-visible{{outline:3px solid #1c6a9e;outline-offset:3px}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:11px 12px;border-bottom:1px solid var(--grid);vertical-align:top;text-align:left;font-size:.84rem}}
thead th{{font-size:.71rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}}tbody tr:last-child>*{{border-bottom:0}}
.fallback-table{{min-width:700px}}.driver-table{{min-width:780px}}
details{{margin-top:12px;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px}}summary{{cursor:pointer;font-weight:720}}
.inspector-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}}.inspector-grid p{{font-size:.84rem;margin:0;background:#fff;padding:10px;border-radius:9px}}
.era-decisions{{display:grid;gap:10px;margin-top:18px}}.era-decision{{display:grid;grid-template-columns:155px 1.25fr 1fr;gap:18px;background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px}}
.era-decision h3{{font-size:1rem;margin:0}}.era-decision p{{font-size:.86rem;margin:6px 0}}.decision-year strong{{display:block}}
.method-list{{padding-left:22px;color:var(--ink2)}}.method-list li{{margin:8px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
:focus-visible{{outline:3px solid #1c6a9e;outline-offset:3px}}
@media(max-width:800px){{.verdict-grid,.principles,.receipt-grid,.inspector-grid,.trust-receipt{{grid-template-columns:1fr}}.era-decision{{grid-template-columns:1fr}}header,main{{padding-left:16px;padding-right:16px}}.boundary-chart{{height:420px}}#sensitivity-chart{{height:680px}}.local-nav .scroll-cue{{display:inline-flex}}
.era-strip{{grid-template-columns:repeat(9,minmax(108px,1fr))}}.timeline-cue{{display:block}}
.table-wrap::before{{content:"Scroll table horizontally →";display:block;position:sticky;left:0;width:max-content;
margin:7px;padding:4px 7px;border-radius:6px;background:#edf4f8;color:#244e6e;font-size:.68rem;font-weight:750}}}}
</style></head><body>{nav(current_href="era-boundaries.html")}
<header><div class="kicker">Methods · editorial periodization</div>
<h1>How we chose the eras</h1>
<p class="sub">The optimizer proposes persistent rhetorical regimes. Sensitivity tests show what
survives alternate reasonable choices. Historical evidence resolves exact dates without pretending
that an algorithm discovered the only possible history.</p>
<span class="status">Current Story scheme · reviewed and live</span></header><main>
<nav class="local-nav" aria-label="On this page"><span class="scroll-cue" aria-hidden="true">More sections →</span><a href="#verdict">Verdict</a><a href="#annual">Annual mountain</a><a href="#sensitivity">Sensitivity</a><a href="#inspector">Boundary inspector</a><a href="#decisions">Era decisions</a><a href="#reproduce">Reproduce</a></nav>

<section id="verdict"><div class="answer"><h2>The 30-second verdict</h2>
<p>Under the declared {meta['primary_cluster_count']}-era, {meta['minimum_segment_units']}-cycle-unit-minimum specification, the primary mathematical starts are
{primary_starts}. The Story makes {len(meta['reviewed_adjustments'])} modeled-start adjustments—{adjustment_summary}—and represents
the grid-labeled {int(civil_war['grid_start'])} unit with the editorial start {int(civil_war['editorial_start'])}. Together, those choices add
only {meta['reviewed_excess_objective_pct']:.3f}% to within-era variation.</p>
<p class="timeline-cue">Scroll the era timeline horizontally →</p>
<div class="era-strip" aria-label="Current Story eras; swipe horizontally on narrow screens">{_timeline(rows)}</div></div>
<div class="verdict-grid"><article class="verdict"><strong>Conditional, not inevitable</strong><p>{meta['primary_cluster_count']} eras is an editorial product constraint. The audit tests {meta['cluster_count_range'][0]}–{meta['cluster_count_range'][1]} segments, duration, grid, weighting, endpoint, and axis omissions.</p></article>
<article class="verdict"><strong>Stable center, sensitive edges</strong><p>{len(computed)} conditions are estimable. Their adjusted-Rand scores and exact starts are published—not compressed into a single confidence claim.</p></article>
<article class="verdict"><strong>The present is provisional</strong><p>The final {terminal_label} cycle is incomplete. Excluding it moves the last modeled start, so the {era_profiles.ERA_PROFILE_SPECS[-1].start_year}–{era_profiles.ERA_PROFILE_SPECS[-1].end_year} Story era is explicitly right-censored.</p></article></div></section>

<section><h2>Three questions, three measurements</h2><div class="principles">
<div class="principle"><strong>Persistent regimes</strong><p>Dynamic programming selects contiguous segments that minimize standardized within-era variation.</p></div>
<div class="principle"><strong>Annual transition mountain</strong><p>Four- and eight-year before/after ranks locate broad episodes of abrupt change without selecting the final partition.</p></div>
<div class="principle"><strong>Segmentation sensitivity range</strong><p>Alternate defensible specifications show which proposed cuts recur and which depend on modeling choices.</p></div></div></section>

<section id="annual"><h2>1 · Where does rhetoric change abruptly?</h2>
<p>The light curves rank four- and eight-year symmetric comparisons; the dark curve averages their
percentiles. High values mean sharper change in this corpus—not historical importance, causation,
or confidence in an exact boundary.</p>
<figure class="chart-card" data-substantive-chart data-metric="era_annual_abruptness"
data-evidence="data/era-boundary-consensus-scores.csv" aria-labelledby="annual-title annual-desc">
<figcaption class="chart-head"><div><strong id="annual-title">Every-year abruptness percentile</strong>
<p id="annual-desc">Reviewed starts are diamonds. Neighboring peaks form one transition mountain.
<a href="metrics.html#era_annual_abruptness">Define annual abruptness.</a></p></div></figcaption>
<div id="sliding-chart" class="boundary-chart" role="img" aria-label="Four-year, eight-year, and combined annual transition percentiles from {annual_start} through {annual_end}" hidden></div>
<p class="chart-note">Exact values remain available without JavaScript below and in the downloadable CSV.</p></figure>
<details><summary>Accessible table: all candidate-year scores</summary>
<div class="table-wrap" role="region" aria-label="All candidate-year abruptness scores" tabindex="0"><table class="fallback-table"><thead><tr><th>Candidate year</th><th>Combined rank</th><th>Combined percentile</th><th>4-year rank</th><th>8-year rank</th></tr></thead>
<tbody>{_annual_fallback(consensus)}</tbody></table></div></details></section>

<section id="sensitivity"><h2>2 · Which boundaries survive alternate choices?</h2>
<p>Each row is a complete dynamic-programming rerun. Dots are its internal starts; vertical guides
are the reviewed Story starts. The matrix makes clear that changing the number of requested eras
is a different question from testing the nine-era solution's robustness.</p>
<figure class="chart-card" data-substantive-chart data-metric="era_segmentation_sensitivity"
data-evidence="data/era-boundary-sensitivity.csv" aria-labelledby="sensitivity-title sensitivity-desc">
<figcaption class="chart-head"><div><strong id="sensitivity-title">Condition × boundary sensitivity matrix</strong>
<p id="sensitivity-desc">Marker shape identifies the test family; hover shows adjusted-Rand similarity to the primary partition.
<a href="metrics.html#era_segmentation_sensitivity">Define segmentation sensitivity.</a></p></div></figcaption>
<div id="sensitivity-chart" class="boundary-chart" role="img" aria-label="Boundary starts under every estimable segmentation sensitivity condition" hidden></div></figure>
<div class="callout warning"><p><strong>Common-genre check not estimable on this grid.</strong>
The annual-message/State-of-the-Union subset has an empty four-year unit: {common_genre_note}
The audit records that failure rather than bridging a temporal gap or claiming genre invariance.</p></div>
<details><summary>Accessible table: every condition, start, and similarity</summary><div class="table-wrap" role="region" aria-label="Every sensitivity condition, start, and similarity" tabindex="0">
<table class="fallback-table"><thead><tr><th>Condition</th><th>Family</th><th>Partition starts</th><th>Adjusted Rand vs primary</th><th>Status note</th></tr></thead>
<tbody>{_sensitivity_fallback(summary)}</tbody></table></div></details>
<p class="chart-note"><a href="data/era-boundary-cluster-stability.csv">Download the complementary k={meta['cluster_count_range'][0]}–{meta['cluster_count_range'][1]} contiguity-cluster stability audit</a>.</p></section>

<section id="inspector"><h2>3 · Inspect each editorial boundary</h2>
<p>Each record keeps the model proposal, editorial fit cost, abruptness, adjacent population, and
top signed drivers together. “Equal-cycle” matches the optimization; “pooled Story-era” uses all
speeches assigned to the adjacent live Story eras, so both estimands are shown.</p>
{_boundary_fallback(evidence, drivers)}</section>

<section><h2>How the close calls were resolved</h2><div class="verdict-grid">
<article class="verdict"><strong>{int(expansion['editorial_start'])} · Madison takes office</strong><p>The primary optimum is {int(expansion['data_optimal_start'])}. Moving the cut to the reviewed start costs {float(expansion['penalty_pct_of_optimum']):.3f}% of the optimum—an explicit historical tie-break inside the measured sensitivity range.</p></article>
<article class="verdict"><strong>{int(war_new_deal['editorial_start'])} · FDR takes office</strong><p>The primary {meta['primary_cluster_count']}-era optimum is {int(war_new_deal['data_optimal_start'])}, while the best two-segment split begins at {meta['single_split_start']}. The reviewed start costs {float(war_new_deal['penalty_pct_of_optimum']):.3f}% of the optimum.</p></article>
<article class="verdict"><strong>{int(present['editorial_start'])} · provisional endpoint</strong><p>The full primary run starts at {int(present['data_optimal_start'])}; excluding the incomplete terminal unit starts the last modeled segment at {terminal_start}. This is endpoint sensitivity, not proof against genre mix.</p></article></div>
<div class="callout"><p><strong>Why did {legacy_gilded_start} appear before?</strong> The retained bin-grain cluster audit places its closest boundary to the Gilded Age at {legacy_gilded_start}; the four-year primary partition starts at {int(rows_by_key['gilded-age']['data_optimal_start'])}, and the governed Story era starts at {int(rows_by_key['gilded-age']['editorial_start'])}. The grains answer different questions, so the legacy result remains downloadable rather than silently rewritten.</p></div></section>

<section id="decisions"><h2>The historical decision ledger</h2>
<p>Names, dates, and links to the live Story come from one governed era specification. Historical
receipts orient the editorial decision; they are not evidence that speech patterns caused events.</p>
<div class="era-decisions">{_scheme_cards(rows)}</div>
<div class="callout"><p><strong>Accession-year ownership:</strong> a speech keeps its real date,
but an outgoing president's speech belongs to the Story regime that presidency closes. This rule
changes Story organization only; it never alters the measurements or boundary analysis.</p></div></section>

<section><h2>What the model does—and does not do</h2><ol class="method-list">
<li><strong>Fingerprint:</strong> {meta['n_axes']} topic, combat, register, dictionary-marker, readability/pronoun/modal, and assigned speech-type axes.</li>
<li><strong>Primary specification:</strong> z-score each axis across four-year cycles, request exactly {meta['primary_cluster_count']} contiguous segments, and require at least {meta['minimum_segment_units']} cycles per segment.</li>
<li><strong>Weighting:</strong> the primary gives every nonconstant axis equal weight. A sensitivity run gives every axis family equal aggregate weight.</li>
<li><strong>Support:</strong> each segment carries its speech and paragraph counts; the final incomplete cycle carries a right-censor flag.</li>
<li><strong>Statistical status:</strong> descriptive periodization of this curated formal-speech corpus. These are neither confidence intervals nor causal estimates.</li></ol></section>

<section id="reproduce"><h2>Reproduce and inspect</h2>{era_receipt}<div class="receipt-grid">
<div class="receipt"><strong>Inputs</strong><code>data/eras/fine_fingerprints.parquet</code><p>Governed {meta['n_axes']}-axis fingerprints; every source hash is in the v3 receipt.</p></div>
<div class="receipt"><strong>Annual mountain</strong><code>sliding_scores.parquet + consensus_scores.parquet</code><p><a href="data/era-boundary-sliding-scores.csv">Raw windows</a> · <a href="data/era-boundary-consensus-scores.csv">combined percentiles</a></p></div>
<div class="receipt"><strong>Sensitivity</strong><code>sensitivity.parquet</code><p><a href="data/era-boundary-sensitivity.csv">All conditions, support, status, and partition similarity</a></p></div>
<div class="receipt"><strong>Boundary evidence</strong><code>boundary_evidence.parquet + boundary_drivers.parquet</code><p><a href="data/era-boundary-evidence.csv">Inspector records</a> · <a href="data/era-boundary-drivers.csv">signed drivers</a></p></div>
<div class="receipt"><strong>Primary compatibility tables</strong><code>cycle4_fingerprints.parquet + segmentation.parquet</code><p><a href="data/era-boundary-cycle4-fingerprints.csv">Four-year fingerprints</a> · <a href="data/era-boundary-segmentation.csv">primary and omission partitions</a></p></div>
<div class="receipt"><strong>Original cluster audit</strong><code>cluster_stability.parquet</code><p><a href="data/era-boundary-cluster-stability.csv">Download k={meta['cluster_count_range'][0]}–{meta['cluster_count_range'][1]} cluster stability</a></p></div>
<div class="receipt"><strong>Run locally</strong><code>arch -x86_64 .venv/bin/python -m presidential_profiles.era_boundaries</code><p>Zero API calls; staged files publish before the hash-validated v3 receipt.</p></div></div>
<details><summary>Machine-readable v3 method receipt</summary><pre>{html.escape(json.dumps(meta, indent=2))}</pre></details>
<p><a href="data/era-boundaries-v3/manifest.json">Download the public v3 projection manifest</a>.</p></section>
</main><footer><p>Presidential Profiles · Miller Center formal presidential-speech corpus ·
historical links are orientation receipts, not causal claims.</p></footer>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script><script>
const SLIDING={json_for_script(sliding_payload)};
const CHOSEN={json_for_script(chosen_payload)};
const MATRIX={json_for_script(_matrix_payload(sensitivity))};
const REVIEWED={json_for_script(reviewed_starts)};
const FAMILY_LABELS={json_for_script(FAMILY_LABELS)};
const N_ANNUAL={candidate_count};
const COLORS={{primary:"#172a38",k_sweep:"#315f78",minimum_duration:"#885d1e",grid_anchor:"#6a4a78",weighting:"#24705a",terminal:"#a24f52",axis_omission:"#596b35",common_genre:"#74695f"}};
const SYMBOLS={{primary:"diamond",k_sweep:"circle",minimum_duration:"square",grid_anchor:"triangle-up",weighting:"star",terminal:"x",axis_omission:"circle-open",common_genre:"cross"}};
const base={{template:"simple_white",paper_bgcolor:"#fff",plot_bgcolor:"#fff",font:{{family:"system-ui",color:"#24313d"}},hoverlabel:{{font:{{family:"system-ui"}}}}}};
function editorialShapes(){{return REVIEWED.map(y=>({{type:"line",xref:"x",yref:"paper",x0:y,x1:y,y0:0,y1:1,line:{{color:"rgba(173,117,63,.28)",width:1,dash:"dot"}},layer:"below"}}));}}
function renderPlot(id,traces,layout){{
 const el=document.getElementById(id); el.removeAttribute("hidden");
 try{{const pending=Plotly.newPlot(id,traces,layout,{{displayModeBar:false,responsive:true}});
  if(pending&&pending.catch){{pending.catch(()=>{{el.hidden=true;}});}}
 }}catch(error){{el.hidden=true;}}
}}
if(window.Plotly){{
 const narrow=window.matchMedia("(max-width: 800px)").matches;
 renderPlot("sliding-chart",[
  {{type:"scatter",mode:"lines",name:"4-year windows",x:SLIDING.map(r=>r.boundary_year),y:SLIDING.map(r=>r.percentile_4*100),line:{{color:"#6d96b3",width:1.5}},customdata:SLIDING.map(r=>r.rank_4),hovertemplate:"%{{x}} · %{{y:.1f}}th percentile<br>rank %{{customdata}}<extra>4-year windows</extra>"}},
  {{type:"scatter",mode:"lines",name:"8-year windows",x:SLIDING.map(r=>r.boundary_year),y:SLIDING.map(r=>r.percentile_8*100),line:{{color:"#ad753f",width:1.5,dash:"dot"}},customdata:SLIDING.map(r=>r.rank_8),hovertemplate:"%{{x}} · %{{y:.1f}}th percentile<br>rank %{{customdata}}<extra>8-year windows</extra>"}},
  {{type:"scatter",mode:"lines",name:"Combined",x:SLIDING.map(r=>r.boundary_year),y:SLIDING.map(r=>r.consensus*100),line:{{color:"#172a38",width:2.7}},customdata:SLIDING.map(r=>r.consensus_rank),hovertemplate:"%{{x}} · %{{y:.1f}}th percentile<br>rank %{{customdata}} of "+N_ANNUAL+"<extra>Combined</extra>"}},
  {{type:"scatter",mode:"markers",name:"Reviewed starts",x:CHOSEN.map(r=>r.editorial_start),y:CHOSEN.map(r=>r.annual_abruptness_percentile*100),marker:{{size:10,color:"#8b4d19",symbol:"diamond",line:{{color:"#fff",width:1}}}},customdata:CHOSEN.map(r=>[r.annual_abruptness_rank,r.data_optimal_start]),hovertemplate:"Reviewed %{{x}}<br>%{{y:.1f}}th percentile · rank %{{customdata[0]}}<br>primary model: %{{customdata[1]}}<extra></extra>"}}
 ],{{...base,height:narrow?420:500,margin:{{l:narrow?48:58,r:12,t:32,b:55}},shapes:editorialShapes(),xaxis:{{title:"candidate boundary year",dtick:narrow?60:40,gridcolor:"#e5e8ea"}},yaxis:{{title:narrow?"percentile":"abruptness percentile",range:[0,103],gridcolor:"#e5e8ea"}},legend:{{orientation:"h",y:1.13,x:0,font:{{size:narrow?11:12}}}}}});
 const order=[...new Set(MATRIX.map(r=>r.label))];
 const families=[...new Set(MATRIX.map(r=>r.family))];
 const traces=families.map(f=>{{const rows=MATRIX.filter(r=>r.family===f);return {{type:"scatter",mode:"markers",name:FAMILY_LABELS[f]||f,x:rows.map(r=>r.start),y:rows.map(r=>r.label),marker:{{size:9,color:COLORS[f],symbol:SYMBOLS[f],line:{{color:"#fff",width:.7}}}},customdata:rows.map(r=>r.similarity),hovertemplate:"%{{y}}<br>start %{{x}}<br>adjusted Rand %{{customdata:.3f}}<extra>%{{fullData.name}}</extra>"}};}});
 if(narrow){{traces.forEach(trace=>{{trace.y=trace.y.map(label=>label.replace("Primary: nine eras","Primary · 9").replace("Minimum four cycle units","Min 4 units").replace("Cycle grid ","Grid ").replace("Equal axis-family weight","Family weight").replace("Exclude incomplete 2025–26 cycle","No partial end").replace("Without ","No "));}});}}
 const shownOrder=[...new Set(traces.flatMap(trace=>trace.y))];
 renderPlot("sensitivity-chart",traces,{{...base,height:narrow?680:760,margin:{{l:narrow?112:190,r:12,t:34,b:55}},shapes:editorialShapes(),xaxis:{{title:narrow?"segment start":"modeled segment start",dtick:narrow?60:40,range:[1784,2023],gridcolor:"#e5e8ea"}},yaxis:{{categoryorder:"array",categoryarray:shownOrder.slice().reverse(),tickfont:{{size:narrow?11:12}},gridcolor:"#f0f1f2"}},legend:{{orientation:"h",y:1.08,x:0,font:{{size:narrow?11:11}}}}}});
}}
</script></body></html>'''


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n")
    return stream.getvalue().encode("utf-8")


def _public_manifest(
    analysis: dict, exports: dict[str, pd.DataFrame], payloads: dict[str, bytes]
) -> dict:
    manifest = {
        **analysis["meta"],
        "public_manifest_schema": "era-boundaries-public-manifest-v3",
        "generation_status": "deterministic_local_zero_api_calls",
        "public_projection": {
            filename: {
                "rows": int(len(exports[filename])),
                "columns": list(exports[filename].columns),
                "semantic_key": PUBLIC_EXPORT_KEYS[filename],
                "bytes": len(payloads[filename]),
                "sha256": _sha256_bytes(payloads[filename]),
            }
            for filename in sorted(exports)
        },
        "public_validations": {
            "source_artifact_hashes": "passed",
            "row_counts": "passed",
            "semantic_keys": "passed",
            "legacy_alias_byte_parity": "passed",
            "failure_atomic_publication": "passed",
        },
    }
    for filename, frame in exports.items():
        keys = PUBLIC_EXPORT_KEYS[filename]
        missing = sorted(set(keys) - set(frame.columns))
        if missing:
            raise ValueError(f"{filename} is missing public semantic keys {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{filename} duplicates its public semantic key {keys}")
    semantic = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    manifest["public_metadata_sha256"] = _sha256_bytes(semantic)
    return manifest


def validate_public_projection(site_dir: Path) -> dict:
    """Validate the installed canonical bundle and every compatibility alias."""
    versioned_dir = site_dir / "data" / "era-boundaries-v3"
    canonical_manifest = versioned_dir / "manifest.json"
    compatibility_manifest = site_dir / "data" / "era-boundaries-meta.json"
    if not canonical_manifest.is_file() or not compatibility_manifest.is_file():
        raise ValueError("era-boundaries-v3 public manifest is missing")
    canonical_bytes = canonical_manifest.read_bytes()
    if compatibility_manifest.read_bytes() != canonical_bytes:
        raise ValueError("era-boundaries public manifest aliases differ")
    manifest = json.loads(canonical_bytes)
    claimed_metadata_hash = manifest.pop("public_metadata_sha256", None)
    semantic = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if claimed_metadata_hash != _sha256_bytes(semantic):
        raise ValueError("era-boundaries public manifest metadata hash is stale")
    manifest["public_metadata_sha256"] = claimed_metadata_hash

    projections = manifest.get("public_projection", {})
    if set(projections) != set(PUBLIC_EXPORT_KEYS):
        raise ValueError("era-boundaries public projection inventory is incomplete")
    for filename, receipt in projections.items():
        canonical_path = versioned_dir / filename
        compatibility_path = site_dir / "data" / filename
        if not canonical_path.is_file() or not compatibility_path.is_file():
            raise ValueError(f"missing era-boundaries public projection: {filename}")
        payload = canonical_path.read_bytes()
        if compatibility_path.read_bytes() != payload:
            raise ValueError(f"era-boundaries compatibility alias differs: {filename}")
        if receipt.get("sha256") != _sha256_bytes(payload):
            raise ValueError(f"era-boundaries public hash mismatch: {filename}")
        if receipt.get("bytes") != len(payload):
            raise ValueError(f"era-boundaries public byte count mismatch: {filename}")
        frame = pd.read_csv(io.BytesIO(payload))
        if receipt.get("rows") != len(frame):
            raise ValueError(f"era-boundaries public row count mismatch: {filename}")
        if receipt.get("columns") != list(frame.columns):
            raise ValueError(f"era-boundaries public schema mismatch: {filename}")
        semantic_key = receipt.get("semantic_key")
        if semantic_key != PUBLIC_EXPORT_KEYS[filename]:
            raise ValueError(f"era-boundaries public semantic key drift: {filename}")
        if frame.duplicated(semantic_key).any():
            raise ValueError(f"era-boundaries public duplicate key: {filename}")
    return manifest


def _publish_public_transaction(
    site_dir: Path,
    versioned_payloads: dict[str, bytes],
    compatibility_payloads: dict[Path, bytes],
    post_validate: Callable[[Path], object] | None = None,
) -> None:
    """Failure-atomically replace the canonical bundle and compatibility files."""
    versioned_target = site_dir / "data" / "era-boundaries-v3"
    site_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="era-boundaries-public-", dir=site_dir
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        candidate = temporary_root / "candidate"
        backup = temporary_root / "backup"
        staged = temporary_root / "staged"
        candidate.mkdir()
        backup.mkdir()
        staged.mkdir()
        for filename, payload in versioned_payloads.items():
            path = candidate / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        staged_paths: dict[Path, Path] = {}
        for index, (target, payload) in enumerate(compatibility_payloads.items()):
            path = staged / f"{index:02d}-{target.name}"
            path.write_bytes(payload)
            staged_paths[target] = path

        targets = [versioned_target, *compatibility_payloads]
        backups: dict[Path, Path] = {}
        installed: list[Path] = []
        try:
            for index, target in enumerate(targets):
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    held = backup / f"{index:02d}-{target.name}"
                    os.replace(target, held)
                    backups[target] = held
            os.replace(candidate, versioned_target)
            installed.append(versioned_target)
            for target, source in staged_paths.items():
                os.replace(source, target)
                installed.append(target)
            if post_validate is not None:
                post_validate(site_dir)
        except BaseException:
            for target in reversed(installed):
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)
            for target, held in backups.items():
                if held.exists():
                    os.replace(held, target)
            raise


def write_era_boundaries(site_dir: Path, analysis: dict | None = None) -> None:
    analysis = era_boundaries.load() if analysis is None else analysis
    site_dir.mkdir(parents=True, exist_ok=True)
    public_data = site_dir / "data"
    public_data.mkdir(exist_ok=True)
    exports = {
        "era-boundary-sliding-scores.csv": analysis["sliding_scores"],
        "era-boundary-cluster-stability.csv": analysis["cluster_stability"],
        "era-boundary-consensus-scores.csv": analysis["consensus_scores"],
        "era-boundary-cycle4-fingerprints.csv": analysis["cycle_fingerprints"],
        "era-boundary-segmentation.csv": analysis["segmentation"],
        "era-boundary-sensitivity.csv": analysis["sensitivity"],
        "era-boundary-evidence.csv": analysis["boundary_evidence"],
        "era-boundary-drivers.csv": analysis["boundary_drivers"],
    }
    payloads = {filename: _csv_bytes(frame) for filename, frame in exports.items()}
    manifest = _public_manifest(analysis, exports, payloads)
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    versioned_payloads = {**payloads, "manifest.json": manifest_payload}
    compatibility_payloads = {
        **{public_data / filename: payload for filename, payload in payloads.items()},
        public_data / "era-boundaries-meta.json": manifest_payload,
        site_dir / "era-boundaries.html": render_era_boundaries(analysis).encode("utf-8"),
    }
    _publish_public_transaction(
        site_dir,
        versioned_payloads,
        compatibility_payloads,
        post_validate=validate_public_projection,
    )
    print("  wrote docs/era-boundaries.html")
