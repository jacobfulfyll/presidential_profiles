"""Story Page V3 evidence, interaction, and accessibility contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import (
    corpus,
    era_profiles,
    expansion_story,
    indices,
    metrics,
    portraits,
    site,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def artifacts() -> dict[str, pd.DataFrame]:
    return {
        "speeches": corpus.load(),
        "markers": pd.read_parquet(DATA / "speech_markers.parquet"),
        "stats": pd.read_parquet(DATA / "speech_stats.parquet"),
        "speech_annotations": pd.read_parquet(
            DATA / "llm_annotations" / "speech_annotations.parquet"
        ),
        "paragraph_issues": pd.read_parquet(DATA / "paragraph_issues.parquet"),
        "paragraph_annotations": pd.read_parquet(
            DATA / "llm_annotations" / "paragraph_annotations.parquet"
        ),
        "paragraphs": pd.read_parquet(DATA / "paragraphs.parquet"),
        "bands": pd.read_parquet(DATA / "bands.parquet"),
        "expansion_story": expansion_story.load_metrics(
            DATA / "expansion_story" / "period_metrics.parquet"
        ),
    }


@pytest.fixture(scope="module")
def rates(artifacts) -> pd.DataFrame:
    return indices.yearly_rates(artifacts["markers"])


@pytest.fixture(scope="module")
def scores(artifacts) -> pd.DataFrame:
    return indices.president_scores(
        artifacts["markers"], artifacts["stats"], artifacts["speeches"]
    ).set_index("president")


def test_v4_expansion_contract_is_new_and_v2_v3_remain_frozen():
    v2 = (ROOT / "notes" / "story-redesign-v2.md").read_text()
    v3 = (ROOT / "notes" / "story-redesign-v3.md").read_text()
    v4 = (ROOT / "notes" / "story-redesign-v4.md").read_text()
    assert "remains frozen" in v2
    assert "Story Page V2 remains the frozen completed" in v3
    assert "Story Page V3 narrative and interaction contract" in v3
    assert "Opponent-centered wording" in v3
    assert "Story Page V4 expansion-era evidence and visual contract" in v4
    assert "V2 and V3 remain" in v4
    assert "1831–35 banking: 23.3%" in v4
    assert "`expansion_delta` key is absent" in v4


def test_compact_measure_and_evidence_tiles_are_single_level_and_named():
    controls = site._chart_evidence(
        "legal_procedural", "speech_markers.parquet",
        support="Every word.", caveat="Not a performance score.",
    )
    assert "Measure · Legal and procedural vocabulary" in controls
    assert "Evidence · speech_markers.parquet" in controls
    assert "Definition." in controls and "Unit." in controls
    assert "Statistical status." in controls and "Denominator / support." in controls
    assert controls.count("<details") == 2
    assert "Explain this measure" not in controls
    assert "Inspect the evidence" not in controls


def test_shared_historical_toggle_uses_exact_labels_and_live_scale_announcement():
    body = site._view_toggle_html(
        ["era", "full"], ["Era denominator.", "Full-history scale changed."],
        500, "rate_10k",
    )
    assert ">Era view</button>" in body
    assert ">Full history</button>" in body
    assert "1. Era view" not in body
    assert 'aria-live="polite"' in body
    assert "Full-history scale changed." in body


def test_written_republic_claim_and_full_history_are_derived(artifacts):
    claim = site.written_republic_takeaway(
        artifacts["speeches"], artifacts["markers"],
        artifacts["speech_annotations"],
    )
    assert "45 of 53" in claim
    assert "all 3 presidents" in claim
    era = site.fig_written_republic(
        artifacts["speeches"], artifacts["markers"],
        artifacts["speech_annotations"],
    )
    full = site.fig_written_republic(
        artifacts["speeches"], artifacts["markers"],
        artifacts["speech_annotations"], full_history=True,
    )
    assert list(era.layout.yaxis.range) == [0, 100]
    assert list(full.layout.xaxis2.range) == [1789, 2026]
    assert {trace.name for trace in full.data[:2]} == {
        "Written message", "Performed / spoken assigned form",
    }


def test_expansion_matrix_strip_and_takeaway_match_the_locked_contract(artifacts):
    meta = expansion_story.load_meta(
        DATA / "expansion_story" / "meta.json"
    )
    table = artifacts["expansion_story"]
    figure = site.fig_expansion_story(table, meta)
    assert len(figure.data) == 2
    heatmap, adversary = figure.data
    assert np.asarray(heatmap.z).shape == (8, 7)
    assert list(heatmap.y) == list(expansion_story.TOPIC_ROWS)
    assert heatmap.zmin == 0 and heatmap.zmax == 60
    assert list(heatmap.x)[-1] == "1846–49"
    assert adversary.type == "bar"
    assert list(adversary.y) == pytest.approx(
        table[table.row_kind.eq("adversary_rate")]
        .loc[lambda frame: ~frame["is_preview"]]
        .sort_values("period_order")["point"]
        * 100
    )
    assert list(figure.layout.yaxis2.range) == [0, 40]
    annotation_text = " ".join(
        str(annotation.text) for annotation in figure.layout.annotations
    )
    assert "Next era preview" not in annotation_text
    assert "Domestic institutions dominate · 53%" in annotation_text
    assert "Foreign nations dominate · 77%" in annotation_text
    assert len(figure.layout.shapes) == 0
    adversary_text = " ".join(str(value) for value in adversary.text)
    for expected in ("23.3%", "36.5%", "56.4%", "49.2%", "27.3%"):
        assert expected in annotation_text or expected in adversary_text
    claim = site.expansion_takeaway(table, meta)
    assert "episodic spikes, not a rising trend" in claim
    assert "37.7%" not in claim


def test_expansion_narrative_title_bridge_movements_and_fallback(artifacts):
    meta = expansion_story.load_meta(
        DATA / "expansion_story" / "meta.json"
    )
    chapter = next(row for row in site.SECTIONS if row[0] == "expansion_conflict")
    assert chapter == (
        "expansion_conflict",
        "1809–1849 · The continental republic",
        "Diplomacy links war, state-building, and continental conquest",
        "Madison's accession and the War of 1812 open the chapter. Treaties and diplomacy "
        "remain connective tissue as the presidency turns toward public finance, banking, "
        "federal authority, Native removal, and continental conquest.",
    )
    body = site._expansion_story_html(artifacts["expansion_story"], meta)
    for movement in (
        "War tests the republic",
        "Securing the postwar republic",
        "Federal policy becomes domestic combat",
        "Conquest creates a slavery question",
    ):
        assert f"<h3>{movement}</h3>" in body
    assert body.count("data-receipt-group") == 3
    assert "Parent and subrows overlap and are not additive" in body
    assert "Text alternative · topic matrix, intervals, and support" in body
    assert "Events orient time; they do not establish" in body
    assert "founding baseline" not in body.lower()
    assert not hasattr(site, "fig_expansion_delta")


def test_crisis_index_is_100_at_declared_foundation_and_peaks_35_9_percent_above(
    rates,
):
    indexed, raw, baseline = site.fear_index(rates)
    assert baseline == pytest.approx(raw.loc[1789:1808].dropna().mean())
    assert indexed.loc[1789:1808].dropna().mean() == pytest.approx(100)
    crisis = indexed.loc[1850:1868].dropna()
    assert int(crisis.idxmax()) == 1860
    assert float(crisis.max()) == pytest.approx(145.0724269)
    era = site.fig_fear_index(rates)
    full = site.fig_fear_index(rates, full_history=True)
    assert list(era.layout.xaxis.range) == [1850, 1868]
    assert list(full.layout.xaxis.range) == [1789, 2026]
    assert "raw fear ÷ hope" in era.data[0].hovertemplate


def test_rights_era_view_exposes_annual_support_and_full_history_bands(artifacts):
    annual = site._rights_year_support(
        artifacts["paragraph_issues"], artifacts["paragraphs"]
    )
    assert list(annual.index) == list(range(1850, 1869))
    assert int(annual.loc[1862, "n_speeches"]) == 1
    era = site.fig_civil_rights_yearly(
        artifacts["paragraph_issues"], artifacts["paragraphs"]
    )
    assert "speeches" in era.data[0].hovertemplate
    assert "words" in era.data[0].hovertemplate
    assert era.data[0].connectgaps is False
    full = site.fig_civil_rights_full(artifacts["bands"])
    assert len(full.data) == 2
    assert min(full.data[1].x) == 1785
    assert "95% sampling interval" in full.data[1].hovertemplate
    labels = " ".join(str(a.text) for a in era.layout.annotations)
    assert "Emancipation" in labels and "14th" in labels
    assert "Civil Rights Act" not in labels


def test_procedural_scatter_has_all_eras_faces_and_persistent_points(scores):
    assert len(scores) == 45
    faces = portraits.data_uris(list(scores.index))
    figure = site.fig_procedural_eras(scores, faces)
    assert len(figure.data[0].x) == 45
    assert len(figure.layout.images) == 45
    ordered = scores.sort_values(["first_year", "last_year"])
    assert np.allclose(figure.data[0].x, ordered.mechanism)
    assert np.allclose(figure.data[0].y, ordered.hype)
    assert np.allclose(
        [image.x for image in figure.layout.images], ordered.mechanism
    )
    assert np.allclose(
        [image.y for image in figure.layout.images], ordered.hype
    )
    assert (
        figure.layout.xaxis.title.text
        == "legal/procedural vocabulary per 10,000 words"
    )
    assert (
        figure.layout.yaxis.title.text
        == "hype vocabulary per 10,000 words"
    )
    assert "legal/procedural %{x:.1f}" in figure.data[0].hovertemplate
    assert "hype %{y:.1f}" in figure.data[0].hovertemplate
    plotted_eras = {row[3] for row in figure.data[0].customdata}
    assert plotted_eras == {name for name, _, _ in era_profiles.STORY_ERAS}
    body = site._procedural_era_html(scores)
    assert body.count("data-era-choice") == 9
    assert body.count("data-era-preset") == 6
    assert body.count('class="era-chip"') == 9
    assert "Text alternative · all president points" in body
    assert "Unselected presidents remain visible" in body
    assert body.index("<th>Legal / procedural per 10,000</th>") < body.index(
        "<th>Hype per 10,000</th>"
    )


def test_progressive_heatmap_replaces_staged_points_and_keeps_one_scale(artifacts):
    figure = site.fig_progressive_heatmap(artifacts["paragraph_issues"])
    assert len(figure.data) == 1
    assert list(figure.data[0].x) == [label for label, _, _ in site.PROGRESSIVE_PERIODS]
    assert np.asarray(figure.data[0].z).shape == (5, 3)
    assert figure.data[0].zmin == -figure.data[0].zmax
    events = " ".join(str(a.text) for a in figure.layout.annotations)
    assert "U.S. enters WWI" in events
    assert "market crash" in events
    assert "Smoot–Hawley" in events


def test_naming_method_names_terms_window_rule_and_verified_local_quotes(
    artifacts, rates,
):
    body = site._naming_explanation_html(artifacts["speeches"], rates)
    assert "<em>America</em>, <em>American</em>, and <em>Americans</em>" in body
    assert "centered<br>" not in body
    assert "centered\nfive-year window" in body
    assert "1910 is the first start that passes; 1909 fails" in body
    assert "cannot prove a change in national identity" in body
    speech = artifacts["speeches"].loc[
        artifacts["speeches"].doc_name.str.endswith(
            "december-6-1910-second-annual-message"
        )
    ].iloc[0]
    assert "foreign relations of the United States have continued" in speech.transcript
    assert "exact equality between America, Great Britain, France, and Germany" in speech.transcript
    assert body.count("<blockquote>") == 2


def test_new_deal_handoff_covers_recovery_war_and_settlement(artifacts):
    figure = site.fig_new_deal_war(artifacts["paragraph_issues"])
    assert len(figure.data) == 6
    symbols = {trace.marker.symbol for trace in figure.data}
    assert len(symbols) == 6
    assert all(
        list(trace.x) == [
            "1933–39 · recovery",
            "1940–45 · mobilization",
            "1946–52 · settlement",
        ]
        for trace in figure.data
    )
    assert all("points from 1933–39" in trace.hovertemplate for trace in figure.data)


def test_written_to_performed_title_and_decade_values_match_taxonomy(artifacts):
    claim = site.broadcast_takeaway(
        artifacts["stats"], artifacts["speech_annotations"]
    )
    assert "80.6%" in claim and "5.3%" in claim and "0.8%" in claim
    assert "disappear from the sampled 1970s" in claim
    chapter = next(row for row in site.SECTIONS if row[0] == "broadcast_presidency")
    assert chapter[1] == "1953–1980 · Mature Cold War and broadcast presidency"
    figure = site.fig_broadcast_presidency(
        artifacts["stats"], artifacts["speech_annotations"]
    )
    assert {trace.name for trace in figure.data[:2]} == {
        "Written message", "Performed / spoken form",
    }


def test_always_on_real_speeches_show_length_breadth_and_depth_together(artifacts):
    body = site._coverage_pressure_v2_html(
        artifacts["speeches"], artifacts["paragraph_annotations"]
    )
    assert "Median speech length" in body
    assert "Mean effective breadth" in body
    assert "Mean topic depth" in body
    assert "10,068 → 5,224" in body
    assert "15,890" in body and "6,051" in body
    assert "neither side is automatically better" in body
    assert "confirmatory" in body.lower()
    assert "exploratory" in body.lower()


def test_platform_endpoint_is_accurately_named_and_provisional(artifacts, rates):
    claim = site.platform_takeaway(rates, artifacts["markers"])
    assert "opponent-centered" in claim
    assert "8.15" in claim and "6.33" in claim
    assert "4 speeches and 17,177 words through April" in claim
    era = site.fig_platform_signals(rates, artifacts["markers"])
    full = site.fig_platform_signals(
        rates, artifacts["markers"], full_history=True
    )
    titles = " ".join(str(a.text) for a in era.layout.annotations)
    assert "Opponent-centered wording" in titles
    assert "through Apr. 2026 · provisional" in titles
    for trace in era.data:
        assert trace.customdata[0, 0] == ""
        assert "raw 2026 rate" in trace.customdata[-1, 0]
        assert "4 speeches · 17,177 words" in trace.customdata[-1, 0]
    assert list(era.layout.xaxis3.range) == [2001, 2026]
    assert list(full.layout.xaxis3.range) == [1789, 2026]


def test_weather_map_supports_hype_not_doom_superlative(artifacts):
    wide = site._weather_rows(artifacts["markers"])
    current = wide.iloc[-1]
    prior = wide.iloc[:-1]
    assert current.hype == pytest.approx(25.657, abs=.001)
    assert prior.hype.max() == pytest.approx(8.487, abs=.001)
    assert current.doom == pytest.approx(8.914, abs=.001)
    assert prior.doom.max() == pytest.approx(9.312, abs=.001)
    claim = site.weather_takeaway(artifacts["markers"])
    assert "not uniquely doom-heavy; it is uniquely hype-heavy" in claim
    figure = site.fig_rhetorical_weather(artifacts["markers"])
    assert len(figure.data) == 1
    assert len(figure.layout.shapes) >= 6  # four regions plus median rules
    assert "from previous era" in figure.data[0].hovertemplate


def test_summary_collects_the_three_full_record_comparisons(
    artifacts,
    rates,
    scores,
):
    summary = site._standing_html(
        markers=artifacts["markers"],
        scores=scores,
        speeches=artifacts["speeches"],
        rates=rates,
    )
    assert summary.count('data-fig="procedural_eras"') == 1
    assert summary.count('data-fig="naming_progressive"') == 1
    assert summary.count('data-fig="weather_map"') == 1
    assert summary.index('data-fig="procedural_eras"') < summary.index(
        'data-fig="naming_progressive"'
    )
    assert summary.index('data-fig="naming_progressive"') < summary.index(
        'data-fig="weather_map"'
    )
    assert "Refine by specific era" in summary
    assert summary.count("data-era-preset") == 6
    assert "1910 is the first start that passes; 1909 fails" in summary
    with pytest.raises(ValueError, match="scores, speeches, and rates"):
        site._standing_html(scores=scores)


def test_generated_long_run_charts_appear_only_inside_summary():
    story = (ROOT / "docs" / "index.html").read_text()
    summary = (ROOT / "docs" / "summary.html").read_text()
    synthesis_start = summary.index('<section id="synthesis"')
    synthesis_end = summary.index("</main>", synthesis_start)
    synthesis = summary[synthesis_start:synthesis_end]
    assert '<section id="synthesis"' not in story
    assert '<section id="records_appendix"' not in story
    assert '<section id="records_appendix"' not in summary
    for chart in ("procedural_eras", "naming_progressive"):
        marker = f'<div class="chart" data-fig="{chart}"'
        payload_key = f'"{chart}":'
        assert marker not in story
        assert payload_key not in story
        assert summary.count(marker) == 1
        assert payload_key in summary
        assert marker in synthesis
    assert '<div class="chart" data-fig="weather_map"' not in summary
    assert '"weather_map":' not in summary


def test_generated_story_uses_only_the_shared_three_part_era_structure():
    story = (ROOT / "docs" / "index.html").read_text()
    assert story.count('class="founding-workspace era-workspace"') == 9
    assert story.count("data-era-workspace-tab=") == 36
    assert '<div class="chart" data-fig=' not in story
    assert "plotly-3.0.1.min.js" not in story

    for index, (key, *_rest) in enumerate(site.SECTIONS[:9]):
        start = story.index(f'<section id="{key}"')
        if index + 1 < 9:
            next_key = site.SECTIONS[index + 1][0]
            end = story.index(f'<section id="{next_key}"', start)
        else:
            end = story.index(
                '<aside class="chronology-end story-summary-link"', start
            )
        section = story[start:end]
        assert section.count("<h2>") == 1
        assert section.count('class="section-deck"') == 1
        assert section.count('class="founding-workspace era-workspace"') == 1
        assert "<blockquote>" not in section
        assert 'class="event-callouts"' not in section
        assert 'class="source-links"' not in section
        assert 'class="founding-evidence-boundary"' not in section

    for legacy_marker in (
        'class="expansion-story-figure"',
        'class="coverage-pressure-story"',
        'data-fig="fear_index_era"',
        'data-fig="progressive_heatmap"',
        'data-fig="new_deal"',
        'data-fig="broadcast"',
        'data-fig="platform_era"',
    ):
        assert legacy_marker not in story


def test_story_html_has_clean_sections_filters_reduced_motion_and_no_numbered_guides(
    scores,
):
    bodies = {key: "<p>Evidence body.</p>" for key, *_ in site.SECTIONS}
    bodies["written_republic"] = site._view_toggle_html(
        ["era", "full"], ["Era.", "Full scale."], 500, "primary_medium"
    )
    bodies["procedural_presidency"] = site._procedural_era_html(scores)
    page = site.build_html(
        {},
        {"speeches": 1, "words": 100, "presidents": 1,
         "start": 1789, "end": 2026},
        bodies,
        inline=False,
    )
    assert "main::before" not in page
    assert ".story-section::before" not in page
    assert ".story-section::after" not in page
    assert "data-story-index" not in page
    assert "data-era-choice" in page
    assert "Plotly.relayout(chart, imageUpdates)" in page
    assert "all other plotted presidents remain visible" in page
    assert 'get("motion") === "reduce"' in page
    assert "date range" not in page  # rendered values, not a generic placeholder
    assert "Taxonomy limit" not in page
    assert "Event guide for the numbered markers" not in page
    assert "active.dataset.storyRange" in page
    assert "active.dataset.storyEra" in page
    assert "active.dataset.storyTitle" in page


def test_new_charts_are_registered():
    expected = {
        "written_full", "expansion_story", "fear_index_era", "fear_index_full",
        "civil_rights_yearly", "civil_rights_full", "procedural_eras",
        "progressive_heatmap", "platform_era", "platform_full",
    }
    assert expected <= set(metrics.CHART_METRICS)
    metrics.validate_charts(expected)
