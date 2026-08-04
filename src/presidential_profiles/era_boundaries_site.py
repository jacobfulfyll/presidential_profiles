"""Render the public explanation of the site's editorial era choices."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import pandas as pd

from . import era_boundaries
from .expansion_site import NAV_CSS, nav
from .html_safety import json_for_script
from .site_style import PAGE_CSS


ERA_CHOICES = [
    {
        "start": 1789,
        "end": 1808,
        "title": "Establishing the republic",
        "argument": (
            "The first administrations establish sovereignty, federal capacity, "
            "finance, and the country's governing routines. Ending after Jefferson "
            "keeps the founding problem distinct from Madison's war presidency."
        ),
        "zone": "Corpus begins",
        "source_label": "Washington's 1789 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "april-30-1789-first-inaugural-address"
        ),
    },
    {
        "start": 1809,
        "end": 1849,
        "title": "The continental republic",
        "argument": (
            "Madison's March 4 accession supplies the exact presidential boundary "
            "inside the data-derived 1805–09 zone. Treaties and diplomacy connect "
            "renewed conflict with Britain and the War of 1812 to postwar "
            "state-building, federal-policy conflict, removal, annexation, and "
            "continental conquest."
        ),
        "zone": "1805–1809",
        "source_label": "Madison's 1809 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "march-4-1809-first-inaugural-address"
        ),
    },
    {
        "start": 1850,
        "end": 1868,
        "title": "Sectional crisis, Civil War, and constitutional rupture",
        "argument": (
            "Territorial acquisition ends in 1848, but the Compromise of 1850 "
            "turns the status of the acquired lands into a national governing "
            "settlement—and exposes how little had been settled. The chapter "
            "continues through emancipation, Civil War, and the Johnson conflict."
        ),
        "zone": "1848–1852",
        "source_label": "National Archives: Compromise of 1850",
        "source_url": "https://www.archives.gov/milestone-documents/compromise-of-1850",
    },
    {
        "start": 1869,
        "end": 1912,
        "title": "The administrative-industrial order",
        "argument": (
            "Grant's accession is both the data-optimal four-year start and a "
            "presidential boundary. Reconstruction continues, but it does not form "
            "its own presidential-speech regime: the measured agenda turns toward "
            "program administration, trade, and the developing industrial state."
        ),
        "zone": "1869–1870",
        "source_label": "Grant's 1869 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "march-4-1869-first-inaugural-address"
        ),
    },
    {
        "start": 1913,
        "end": 1932,
        "title": "Reform, world war, and the interwar order",
        "argument": (
            "Wilson's accession supplies a clean institutional and presidential "
            "start for a period spanning reform, World War I, the 1920s order, "
            "and its collapse under Hoover."
        ),
        "zone": "1913–1916",
        "source_label": "Wilson's 1913 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "march-4-1913-first-inaugural-address"
        ),
    },
    {
        "start": 1933,
        "end": 1952,
        "title": "New Deal, world war, and postwar settlement",
        "argument": (
            "FDR's accession is the historically legible break: emergency "
            "government and the New Deal lead into global war, while Truman carries "
            "the wartime state into the first postwar settlement."
        ),
        "zone": "1929–1940",
        "source_label": "National Archives: FDR's 1933 inaugural",
        "source_url": "https://www.archives.gov/education/lessons/fdr-inaugural",
    },
    {
        "start": 1953,
        "end": 1980,
        "title": "Mature Cold War and the broadcast presidency",
        "argument": (
            "Eisenhower's accession and the Korean armistice provide a defensible "
            "start for the mature Cold War order. The period runs through Carter, "
            "keeping the Reagan turn visible rather than splitting at 1989."
        ),
        "zone": "1946–1953",
        "source_label": "State Department: Korean armistice, 1953",
        "source_url": (
            "https://history.state.gov/historicaldocuments/frus1952-54v15p2/comp1"
        ),
    },
    {
        "start": 1981,
        "end": 2016,
        "title": "Conservative turn and the always-on presidency",
        "argument": (
            "Reagan's accession is a clearer rhetorical and political boundary "
            "than 1989. This longer period can contain both the Cold War's end and "
            "the post-Cold War, cable, permanent-campaign, and early digital phases."
        ),
        "zone": "1981",
        "source_label": "Reagan's 1981 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "january-20-1981-first-inaugural-address"
        ),
    },
    {
        "start": 2017,
        "end": 2026,
        "title": "Platform-era intensification",
        "argument": (
            "Trump's accession is the editorial start of an intensification, not "
            "the invention of platform politics. The measured transition is broad "
            "enough that the exact presidential boundary remains a historical choice."
        ),
        "zone": "2013–2018",
        "source_label": "Trump's 2017 inaugural",
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "january-20-2017-inaugural-address"
        ),
    },
]

CONDITION_LABELS = {
    "full": "All 63 axes",
    "drop_topic": "Without topics",
    "drop_combat": "Without combat labels",
    "drop_register": "Without proposal/value register",
    "drop_marker": "Without dictionary markers",
    "drop_stat": "Without readability and pronouns",
    "drop_genre": "Without speech type",
    "drop_opponents": "Without the date-bounded opponents marker",
}


def _choice_rows(
    consensus: pd.DataFrame,
    segmentation: pd.DataFrame,
) -> list[dict]:
    boundaries = segmentation[segmentation["is_boundary"]].copy()
    full = boundaries[boundaries["condition"] == "full"]
    conditions = boundaries["condition"].drop_duplicates().tolist()
    rows: list[dict] = []
    for choice in ERA_CHOICES:
        row = dict(choice)
        start = int(choice["start"])
        if start == 1789:
            row.update(
                {
                    "score": None,
                    "rank": None,
                    "local_peak": None,
                    "local_peak_score": None,
                    "data_start": None,
                    "zone_recurrence": None,
                    "data_zone": None,
                }
            )
            rows.append(row)
            continue

        exact = consensus[consensus["boundary_year"] == start]
        if len(exact) != 1:
            raise ValueError(f"consensus artifact has no unique score for {start}")
        exact_row = exact.iloc[0]
        local = consensus[
            consensus["boundary_year"].between(start - 4, start + 4)
        ]
        local_peak = local.loc[local["consensus"].idxmax()]

        grid_start = 1849 if start == 1850 else start
        nearest = full.loc[
            (full["segment_start"] - grid_start).abs().idxmin()
        ]
        data_start = int(nearest["segment_start"])
        matched_starts: list[int] = []
        for condition in conditions:
            candidate = boundaries[boundaries["condition"] == condition]
            candidate = candidate[
                (candidate["segment_start"] - grid_start).abs() <= 4
            ]
            if not candidate.empty:
                matched_starts.append(
                    int(
                        candidate.loc[
                            (candidate["segment_start"] - grid_start).abs().idxmin(),
                            "segment_start",
                        ]
                    )
                )
        data_zone = (
            str(matched_starts[0])
            if len(set(matched_starts)) == 1
            else f"{min(matched_starts)}–{max(matched_starts)}"
        )

        row.update(
            {
                "score": float(exact_row["consensus"]),
                "rank": int(exact_row["consensus_rank"]),
                "local_peak": int(local_peak["boundary_year"]),
                "local_peak_score": float(local_peak["consensus"]),
                "data_start": data_start,
                "zone_recurrence": len(matched_starts),
                "data_zone": data_zone,
            }
        )
        rows.append(row)
    return rows


def _timeline(rows: list[dict]) -> str:
    return "".join(
        f"""<div class="era-chip"><strong>{row['start']}–{row['end']}</strong>
<span>{html.escape(row['title'])}</span></div>"""
        for row in rows
    )


def _data_read(row: dict) -> str:
    if row["start"] == 1789:
        return (
            "The corpus starts here, so there is no before/after score. "
            "The first internal transition is tested at 1809."
        )
    peak = (
        "The exact year is also the strongest combined score within ±4 years."
        if row["local_peak"] == row["start"]
        else f"The strongest combined score within ±4 years is {row['local_peak']}."
    )
    return (
        f"Combined abruptness rank {row['rank']} of 223. {peak} "
        f"The full persistent-regime optimum starts at {row['data_start']}; "
        f"the nearby zone recurs in {row['zone_recurrence']} of 8 specifications."
    )


def _scheme_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        source = (
            f'<a href="{html.escape(row["source_url"], quote=True)}">'
            f'{html.escape(row["source_label"])}</a>'
        )
        body.append(
            f"""<tr><th scope="row"><strong>{row['start']}–{row['end']}</strong>
<span>{html.escape(row['title'])}</span></th>
<td>{html.escape(row['argument'])}<small>Historical receipt: {source}</small></td>
<td>{html.escape(_data_read(row))}<small>Transition zone: {html.escape(row['zone'])}</small></td></tr>"""
        )
    return "".join(body)


def _boundary_fallback(rows: list[dict]) -> str:
    body = []
    for row in rows[1:]:
        body.append(
            f"""<tr><th scope="row">{row['start']}</th>
<td>{row['rank']} of 223</td><td>{row['local_peak']}</td>
<td>{row['data_start']}</td><td>{row['data_zone']}</td>
<td>{row['zone_recurrence']} of 8</td></tr>"""
        )
    return "".join(body)


def _segmentation_payload(segmentation: pd.DataFrame) -> dict[str, list[dict]]:
    payload: dict[str, list[dict]] = {}
    for condition in CONDITION_LABELS:
        rows = segmentation[
            (segmentation["condition"] == condition)
            & segmentation["is_boundary"]
        ]
        payload[condition] = rows[
            ["segment_start", "segment_index", "objective"]
        ].to_dict("records")
    return payload


def render_era_boundaries(analysis: dict | None = None) -> str:
    analysis = era_boundaries.load() if analysis is None else analysis
    consensus = analysis["consensus_scores"].copy()
    segmentation = analysis["segmentation"].copy()
    meta = analysis["meta"]
    rows = _choice_rows(consensus, segmentation)

    sliding_payload = consensus[
        [
            "boundary_year",
            "percentile_4",
            "percentile_8",
            "consensus",
            "rank_4",
            "rank_8",
            "consensus_rank",
        ]
    ].to_dict("records")
    chosen_payload = []
    for row in rows[1:]:
        chosen_payload.append(
            {
                "year": row["start"],
                "score": row["score"],
                "label": str(row["start"]),
                "rank": row["rank"],
                "persistent_start": row["data_start"],
                "recurrence": row["zone_recurrence"],
            }
        )
    starts = [row["start"] for row in rows[1:]]
    condition_options = "".join(
        f'<option value="{key}">{html.escape(label)}</option>'
        for key, label in CONDITION_LABELS.items()
    )
    adjustments = {
        int(item["reviewed_start"]): item
        for item in meta["reviewed_adjustments"]
    }

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>How we chose the eras · Presidential Profiles</title>
<meta name="description" content="The historical judgment, clustering, and every-year transition analysis behind the Presidential Profiles era boundaries.">
<style>{PAGE_CSS}
{NAV_CSS}
header,main,footer{{max-width:1120px}}header{{padding-top:48px}}
.kicker{{font-size:.76rem;letter-spacing:.1em;text-transform:uppercase;font-weight:780;color:#32658d}}
header h1{{font-size:clamp(2.35rem,6vw,4.8rem);line-height:1.01;max-width:900px;margin-top:12px}}
header .sub{{font-size:1.08rem;max-width:800px}}
.status{{display:inline-block;margin-top:20px;padding:7px 11px;border-radius:999px;background:#e8f1f7;color:#244e6e;font-size:.78rem;font-weight:740}}
.answer{{background:#172a38;color:#fff;border-radius:20px;padding:24px;margin-top:28px}}
.answer h2{{font-size:1.42rem}}.answer p{{color:#d9e2e8;max-width:820px;margin-top:8px}}
.era-strip{{display:grid;grid-template-columns:repeat(9,minmax(0,1fr));gap:5px;margin-top:18px;overflow-x:auto;padding-bottom:4px}}
.era-chip{{min-width:0;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.16);border-radius:10px;padding:10px}}
.era-chip strong,.era-chip span{{display:block}}.era-chip strong{{font-size:.76rem;white-space:nowrap}}.era-chip span{{font-size:.68rem;color:#d9e2e8;margin-top:4px;line-height:1.3}}
.principles{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}}
.principle{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px}}
.principle strong{{display:block}}.principle p{{font-size:.88rem;margin:5px 0 0}}
.chart-card{{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:18px;margin-top:18px}}
.chart-head{{display:flex;justify-content:space-between;align-items:end;gap:16px;flex-wrap:wrap}}
.chart-head p{{margin:5px 0 0}}label{{font-size:.78rem;color:var(--muted);font-weight:700}}
select{{display:block;width:min(390px,100%);max-width:100%;margin-top:5px;padding:8px 30px 8px 10px;border:1px solid var(--border);border-radius:9px;background:#fff;font:inherit}}
.chart-shell{{overflow-x:auto;overflow-y:hidden;margin-top:10px}}.boundary-chart{{min-width:760px;height:520px}}
.chart-note{{font-size:.82rem;color:var(--muted);margin-top:8px}}
.callout{{border-left:4px solid #6b9fc5;background:#edf4f8;border-radius:0 12px 12px 0;padding:15px 17px;margin:20px 0}}
.callout p{{margin:0;max-width:none}}code{{font-size:.88em}}
.table-wrap{{overflow-x:auto;border:1px solid var(--border);border-radius:14px;margin-top:18px;background:var(--surface)}}
table{{width:100%;border-collapse:collapse;min-width:850px}}th,td{{padding:13px 14px;border-bottom:1px solid var(--grid);vertical-align:top;text-align:left;font-size:.88rem}}
thead th{{font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}tbody tr:last-child>*{{border-bottom:0}}
tbody th{{width:180px}}tbody th span,td small{{display:block;color:var(--muted);font-weight:450;margin-top:4px}}
td small a{{color:#275d8c}}details{{margin-top:16px;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px}}
summary{{cursor:pointer;font-weight:720}}.method-list{{padding-left:22px;margin-top:12px;color:var(--ink2)}}.method-list li{{margin:8px 0}}
.method-list+details pre,details pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
.receipt-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:16px}}
.receipt{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px}}.receipt strong,.receipt code{{display:block}}.receipt p{{font-size:.84rem;margin:5px 0 0}}
.case-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:18px}}
.case{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:17px}}
.case h3{{margin:0;font-size:1rem}}.case p{{font-size:.88rem;margin:7px 0 0}}
@media(max-width:900px){{.era-strip{{grid-template-columns:repeat(9,125px)}}}}
@media(max-width:800px){{.principles,.receipt-grid,.case-grid{{grid-template-columns:1fr}}header,main{{padding-left:16px;padding-right:16px}}}}
</style></head><body>{nav(current_href="era-boundaries.html")}
<header><div class="kicker">Methods · editorial periodization</div>
<h1>How we chose the eras</h1>
<p class="sub">The primary result divides presidential-cycle fingerprints into nine persistent
rhetorical regimes. Every-year four- and eight-year comparisons then show abrupt transition
mountains inside that complete solution. Historical judgment resolves only the few close calls.</p>
<span class="status">Recommended scheme · live story ranges not yet migrated</span>
</header><main>
<div class="answer"><h2>The data-first recommendation</h2><p>The complete mathematical optimum
starts at {", ".join(map(str, meta["data_optimal_grid_starts"]))}. The reviewed scheme moves only
1805 to Madison's 1809 accession, represents the 1849 cycle with the Compromise of 1850, and moves
1937 to FDR's 1933 accession. After adopting Grant's 1869 accession instead of 1878, its total
within-era variation is only {meta["reviewed_excess_objective_pct"]:.1f}% above the optimum.</p>
<div class="era-strip">{_timeline(rows)}</div></div>

<section><h2>Three measurements answer three different questions</h2>
<div class="principles"><div class="principle"><strong>Persistent regimes choose the cuts</strong>
<p>An exactly-nine-segment optimization minimizes variation within contiguous four-year
presidential-cycle units and prevents eras shorter than three units.</p></div>
<div class="principle"><strong>Combined ranks measure abruptness</strong><p>Every candidate year
gets four- and eight-year percentile ranks. Their average reveals broad transition mountains
without letting one window length dictate the order.</p></div>
<div class="principle"><strong>Robustness tests the zone</strong><p>The full partition is repeated
without each axis group and without the date-bounded opponents marker. Exact accessions resolve
only close zones such as 1805–09 and 1933–37.</p></div></div></section>

<section><h2>Graph 1 · Abrupt change is not the same as a new era</h2>
<p>The light lines rank the four- and eight-year before/after distances separately; the dark line
averages their percentiles. High means more abrupt measured change, not greater historical
importance or confidence. Adjacent high years often belong to one transition mountain.</p>
<div class="chart-card"><div class="chart-head"><div><strong>Every-year consensus abruptness</strong>
<p>Recommended starts are labeled. Hover reports all three ranks.</p></div></div>
<div class="chart-shell"><div id="sliding-chart" class="boundary-chart" role="img"
aria-label="Four-year, eight-year, and combined transition percentile rankings"></div></div>
<p class="chart-note">The raw combined top eight collapse into the early republic and the
Depression–war–postwar mountain. That is why annual rank diagnoses shocks but does not select the
complete partition.</p></div></section>

<section><h2>Graph 2 · Which cuts create persistent regimes?</h2>
<p>Four-year presidential-cycle fingerprints are divided into exactly nine contiguous segments,
each at least three units long. The selector below removes one measurement family at a time. A
cut that remains near the same year is more stable than an isolated annual peak.</p>
<div class="chart-card"><div class="chart-head"><div><strong>Optimal nine-era partition</strong>
<p>Diamonds are the mathematical cuts; dotted guides are the reviewed exact starts.</p></div>
<div><label for="cluster-condition">Robustness view</label><select id="cluster-condition">
{condition_options}</select></div></div><div class="chart-shell"><div id="cluster-chart"
class="boundary-chart" role="img" aria-label="Optimal persistent-regime boundaries"></div></div>
<p id="cluster-status" class="chart-note" aria-live="polite">Showing all 63 axes.</p></div>
<div class="callout"><p><strong>Why did 1872 appear before?</strong> The old eight-year grid had
one unit for 1864–71 and labeled the next unit 1872. It could not test 1869 or 1870. The four-year
partition starts the new regime at 1869, the every-year curve peaks in 1869–70, and Grant was
inaugurated on March 4, 1869. There was no presidential change in 1870.</p></div></section>

<section><h2>How the close exact-year calls were resolved</h2>
<div class="case-grid">
<article class="case"><h3>1809 · Madison takes office</h3><p>The full optimum is 1805, but moving
the cut to Madison's March 4, 1809 accession adds only
{adjustments[1809]["penalty_pct_of_optimum"]:.2f}% to the total
objective. Across the zone, founding-era ceremonial, faith, Native-affairs, and constitutional
subjects recede while banking, trade, and internal improvements rise. The data does not resolve
an exact year within that presidential term, so 1809 is the legible tie-break.</p></article>
<article class="case"><h3>1869 · Grant takes office</h3><p>This is not a tie-break. Civil-rights
and immigration topics fall from 30.7% before the cut to 6.7% after it; constitutional-order
topics fall from 34.0% to 17.3%; proposal register rises from 15.1% to 29.5%; and trade and tariff
topics rise from 3.2% to 13.6%. Reconstruction continues historically, but does not appear as a
separate persistent presidential-speech regime.</p></article>
<article class="case"><h3>1933 · FDR takes office</h3><p>The nine-era optimum is 1937, while the
single strongest division of the whole series is {meta["single_split_start"]} and dropping the
topic axes also returns
1933. The later crest captures the fuller move toward confrontation, broadcast leadership, and
war mobilization. The New Deal–war chapter therefore begins at FDR's March 4, 1933 accession,
costing {adjustments[1933]["penalty_pct_of_optimum"]:.2f}% in complete-partition fit.</p></article>
<article class="case"><h3>2017 · platform-era intensification</h3><p>Seven of eight specifications
return 2017; only removing the entire marker family moves the cut to 2013. After 2017, boosters,
hype, us-versus-them language, zero-sum framing, party attacks, and enemy naming all rise. Genre
accounts for only about 5% of the separation, so 2017 is the sustained regime break rather than
merely a change in the corpus's speech-type mix.</p></article>
</div>
<div class="callout"><p><strong>Who owns a speech in an accession year?</strong>
The speech keeps its real date, but an outgoing president’s speech belongs to the story regime
that presidency is closing. Jefferson’s April 1809 Albemarle County message therefore remains
with the founding; Madison’s March and November speeches open the continental-republic record.
The same rule assigns any Johnson speech in 1869 to the preceding crisis era (the current corpus
contains only Grant in 1869), Truman’s January 1953 address to the New Deal–war settlement,
Carter’s January 1981 address to the mature Cold War, and Obama’s January 2017 address to the
always-on era. The 1850 boundary is historical rather than an accession boundary, so it receives
no outgoing-president override. This governs story ownership only; it does not alter dates or the
boundary analysis.</p></div></section>

<section><h2>The choice for each era</h2>
<p>The right column keeps abruptness rank separate from the persistent-regime decision. “Local
peak” means the strongest combined percentile within four years; robustness recurrence counts
the full and seven axis-removal partitions. Transition zones are not confidence intervals.</p>
<div class="table-wrap"><table><thead><tr><th>Recommended era</th><th>Historical judgment</th>
<th>What the corpus says</th></tr></thead><tbody>{_scheme_table(rows)}</tbody></table></div>
<details><summary>Open the compact boundary audit table</summary><div class="table-wrap">
<table><thead><tr><th>Editorial start</th><th>Combined rank</th><th>Local peak ±4</th>
<th>Full optimum</th><th>Robust zone</th><th>Specifications near zone</th></tr></thead>
<tbody>{_boundary_fallback(rows)}</tbody></table></div><p class="chart-note">The eight
specifications are the full 63-axis model, six leave-one-axis-group-out reruns, and a rerun
without the availability-bounded opponents marker.</p></details>
</section>

<section><h2>What is inside the fingerprint</h2><ol class="method-list">
<li><strong>63 axes:</strong> 17 topic domains, 3 combat labels, 3 proposal/value register shares,
19 declared dictionary markers, 11 readability/pronoun/modal statistics, and 10 assigned speech
types.</li><li><strong>Standardization:</strong> each axis is z-scored with population standard
deviation across the units being compared. The sliding score uses the governed eight-year-bin
standard deviations so a percentage share and a per-thousand-word rate can coexist.</li>
<li><strong>Contiguity and duration:</strong> dynamic programming considers the whole sequence,
allows only adjacent four-year units in a segment, requests exactly nine segments, and requires
at least three units per segment.</li><li><strong>Robustness:</strong> the persistent-regime graph
removes each axis group and the availability-bounded <code>opponents</code> marker. The
every-year graph retains annual resolution with symmetric four- and eight-year windows.</li>
<li><strong>Statistical status:</strong> all results are descriptive over the curated Miller Center
formal-speech corpus. AI labels are measurements, not ground truth; neither graph measures policy
effect, public opinion, or historical causation.</li></ol></section>

<section><h2>Reproduce and inspect the work</h2><div class="receipt-grid">
<div class="receipt"><strong>Original fingerprints</strong><code>data/eras/fine_fingerprints.parquet</code>
<p>31 eight-year bins and 45 presidencies over the same 63 measures.</p></div>
<div class="receipt"><strong>Original clustering</strong><code>data/eras/periodization.parquet</code>
<p>Every boundary at k=2–12 for bin and presidency grains.</p></div>
<div class="receipt"><strong>Every-year windows</strong><code>data/era_boundaries/sliding_scores.parquet</code>
<p><a href="data/era-boundary-sliding-scores.csv">Download both raw window series</a>.</p></div>
<div class="receipt"><strong>Combined ranking</strong><code>data/era_boundaries/consensus_scores.parquet</code>
<p><a href="data/era-boundary-consensus-scores.csv">Download the common-year percentiles</a>.</p></div>
<div class="receipt"><strong>Four-year units</strong><code>data/era_boundaries/cycle4_fingerprints.parquet</code>
<p><a href="data/era-boundary-cycle4-fingerprints.csv">Download all 63-axis unit fingerprints</a>.</p></div>
<div class="receipt"><strong>Persistent-regime reruns</strong><code>data/era_boundaries/segmentation.parquet</code>
<p><a href="data/era-boundary-segmentation.csv">Download full and axis-removal partitions</a>.</p></div>
</div><details><summary>Machine-readable method receipt</summary><pre>{html.escape(json.dumps(meta, indent=2))}</pre></details></section>
</main><footer><p>Presidential Profiles · Miller Center formal presidential-speech corpus ·
historical event links are orientation receipts, not causal claims.</p></footer>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script><script>
const SEGMENTATION={json_for_script(_segmentation_payload(segmentation))};
const CONDITION_LABELS={json_for_script(CONDITION_LABELS)};
const SLIDING={json_for_script(sliding_payload)};
const CHOSEN={json_for_script(chosen_payload)};
const EDITORIAL_STARTS={json_for_script(starts)};
const base={{template:"simple_white",paper_bgcolor:"#fff",plot_bgcolor:"#fff",
font:{{family:"system-ui",color:"#24313d"}},margin:{{l:62,r:28,t:26,b:62}},
hoverlabel:{{font:{{family:"system-ui"}}}}}};
function editorialShapes(){{
 return EDITORIAL_STARTS.map(y=>({{type:"line",xref:"x",yref:"paper",x0:y,x1:y,y0:0,y1:1,
 line:{{color:"rgba(173,117,63,.24)",width:1,dash:"dot"}},layer:"below"}}));
}}
function drawCluster(condition){{
 const rows=SEGMENTATION[condition]||[];
 Plotly.react("cluster-chart",[
  {{type:"scatter",mode:"markers+text",x:rows.map(r=>r.segment_start),
    y:rows.map(()=>1),text:rows.map(r=>String(r.segment_start)),textposition:"top center",
    textfont:{{size:11,color:"#172a38"}},
    marker:{{size:15,color:"#172a38",symbol:"diamond",line:{{color:"#fff",width:1.5}}}},
    customdata:rows.map(r=>[r.segment_index,r.objective]),
    hovertemplate:"Persistent-regime start: %{{x}}<br>Segment %{{customdata[0]}} of 9"+
    "<br>partition objective: %{{customdata[1]:.1f}}<extra></extra>"}}
 ],{{...base,height:300,showlegend:false,shapes:editorialShapes(),
 xaxis:{{title:"four-year-unit boundary",range:[1780,2028],dtick:20,
 gridcolor:"#e8eaec",zeroline:false}},yaxis:{{visible:false,range:[.75,1.25]}}}},
 {{displayModeBar:false,responsive:true}});
 document.getElementById("cluster-status").textContent="Showing "+CONDITION_LABELS[condition]+
  ". Dotted guides are the reviewed exact starts; the 1850 guide represents the 1849–52 cycle.";
}}
drawCluster("full");
document.getElementById("cluster-condition").addEventListener("change",e=>drawCluster(e.target.value));
Plotly.newPlot("sliding-chart",[
 {{type:"scatter",mode:"lines",name:"Four-year windows",
 x:SLIDING.map(r=>r.boundary_year),y:SLIDING.map(r=>r.percentile_4*100),
 line:{{color:"#a9c3d5",width:1.4}},customdata:SLIDING.map(r=>r.rank_4),
 hovertemplate:"Boundary: %{{x}}<br>four-year percentile: %{{y:.1f}}"+
 "<br>rank %{{customdata}} of 231<extra></extra>"}},
 {{type:"scatter",mode:"lines",name:"Eight-year windows",
 x:SLIDING.map(r=>r.boundary_year),y:SLIDING.map(r=>r.percentile_8*100),
 line:{{color:"#d5b796",width:1.4}},customdata:SLIDING.map(r=>r.rank_8),
 hovertemplate:"Boundary: %{{x}}<br>eight-year percentile: %{{y:.1f}}"+
 "<br>rank %{{customdata}} of 223<extra></extra>"}},
 {{type:"scatter",mode:"lines",name:"Combined rank",
 x:SLIDING.map(r=>r.boundary_year),y:SLIDING.map(r=>r.consensus*100),
 line:{{color:"#172a38",width:2.6}},
 customdata:SLIDING.map(r=>r.consensus_rank),
 hovertemplate:"Boundary: %{{x}}<br>combined percentile: %{{y:.1f}}"+
 "<br>combined rank %{{customdata}} of 223<extra></extra>"}},
 {{type:"scatter",mode:"markers+text",name:"Reviewed starts",
 x:CHOSEN.map(r=>r.year),y:CHOSEN.map(r=>r.score*100),
 text:CHOSEN.map(r=>r.label),textposition:CHOSEN.map((r,i)=>i%2?"bottom center":"top center"),
 textfont:{{size:11,color:"#7f5730"}},marker:{{size:9,color:"#ad753f",symbol:"diamond",
 line:{{color:"#fff",width:1}}}},
 customdata:CHOSEN.map(r=>[r.rank,r.persistent_start,r.recurrence]),
 hovertemplate:"Reviewed start: %{{x}}<br>combined percentile: %{{y:.1f}}"+
 "<br>combined rank %{{customdata[0]}} of 223"+
 "<br>full persistent start: %{{customdata[1]}}"+
 "<br>nearby in %{{customdata[2]}} of 8 specifications<extra></extra>"}}
],{{...base,height:520,showlegend:true,shapes:editorialShapes(),
xaxis:{{title:"candidate boundary year",range:[1788,2021],dtick:20,gridcolor:"#e8eaec",
zeroline:false}},yaxis:{{title:"abruptness percentile (high = sharper change)",range:[0,105],
gridcolor:"#e8eaec",zeroline:false}},legend:{{orientation:"h",y:1.13,x:0}}}},
{{displayModeBar:false,responsive:true}});
</script></body></html>"""


def write_era_boundaries(site_dir: Path, analysis: dict | None = None) -> None:
    analysis = era_boundaries.load() if analysis is None else analysis
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "era-boundaries.html").write_text(render_era_boundaries(analysis))
    public_data = site_dir / "data"
    public_data.mkdir(exist_ok=True)
    analysis["sliding_scores"].to_csv(
        public_data / "era-boundary-sliding-scores.csv", index=False
    )
    analysis["cluster_stability"].to_csv(
        public_data / "era-boundary-cluster-stability.csv", index=False
    )
    analysis["consensus_scores"].to_csv(
        public_data / "era-boundary-consensus-scores.csv", index=False
    )
    analysis["cycle_fingerprints"].to_csv(
        public_data / "era-boundary-cycle4-fingerprints.csv", index=False
    )
    analysis["segmentation"].to_csv(
        public_data / "era-boundary-segmentation.csv", index=False
    )
    shutil.copyfile(
        era_boundaries.META_PATH,
        public_data / "era-boundaries-meta.json",
    )
    print("  wrote docs/era-boundaries.html")
