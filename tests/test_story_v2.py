"""Story Page V2 contracts.

These tests pin the editorial structure, honest scale changes, real-data receipts,
and accessibility hooks that are otherwise easy to regress while changing copy.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from presidential_profiles import (
    corpus,
    expansion_story,
    indices,
    issues,
    rhetoric,
    site,
)


REAL_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _story_map() -> dict[str, tuple[str, str, str]]:
    return {
        key: (chapter, title, prose)
        for key, chapter, title, prose in site.SECTIONS
    }


def test_chronology_has_nine_eras_then_synthesis_and_post_story_appendix():
    keys = [row[0] for row in site.SECTIONS]
    assert keys == [
        "written_republic",
        "expansion_conflict",
        "union_crisis",
        "procedural_presidency",
        "progressive_transform",
        "recovery_mobilization",
        "broadcast_presidency",
        "always_on",
        "platform_dominance",
        "synthesis",
        "records_appendix",
    ]
    chapters = [row[1] for row in site.SECTIONS]
    assert chapters[:9] == [
        "1789–1808 · Establishing the republic",
        "1809–1849 · The continental republic",
        "1850–1868 · Sectional crisis and constitutional rupture",
        "1869–1912 · Reconstruction, administration, and industrial power",
        "1913–1932 · Reform, world war, and collapse",
        "1933–1952 · New Deal, world war, and postwar settlement",
        "1953–1980 · Mature Cold War and broadcast presidency",
        "1981–2016 · Conservative turn and the always-on presidency",
        "2017–2026 · Platform-era intensification",
    ]
    assert chapters[9].startswith("Synthesis ·")
    assert chapters[10].startswith("Post-story appendix ·")
    assert "llm_topics" not in keys
    assert "issues" not in keys


def test_story_and_summary_render_as_separate_documents():
    keys = [row[0] for row in site.SECTIONS]
    assert keys[:9][-1] == "platform_dominance"
    assert keys[9:] == ["synthesis", "records_appendix"]
    assert not any(key == "summary" for key in keys)
    bodies = {key: "<p>Evidence body.</p>" for key in keys}
    stats = {
        "speeches": 1, "words": 100, "presidents": 1,
        "start": 1789, "end": 2026,
    }
    story = site.build_html({}, stats, bodies, inline=False, page_kind="story")
    summary = site.build_html(
        {}, stats, bodies, inline=False, page_kind="summary"
    )
    assert '<section id="platform_dominance"' in story
    assert '<section id="synthesis"' not in story
    assert '<section id="records_appendix"' not in story
    assert 'href="summary.html">Continue to the Summary' in story
    assert 'location.replace("summary.html" + location.hash)' in story
    assert '<section id="written_republic"' not in summary
    assert '<section id="synthesis"' in summary
    assert '<section id="records_appendix"' in summary
    assert "America in Summary" in summary


def test_required_limitations_are_in_reader_facing_chapter_copy():
    story = _story_map()
    all_copy = " ".join(
        value for chapter in story.values() for value in chapter
    )
    assert "old horizontal tenure lines" not in all_copy
    assert "not a birth" in all_copy
    assert "does not declare Reconstruction historically complete" in all_copy
    assert "endpoint is provisional through April 2026" in all_copy
    assert "Social platforms did not suddenly originate in 2017" in all_copy


def test_build_html_marks_the_end_of_chronology_and_progress_metadata():
    bodies = {key: "<p>Evidence body.</p>" for key, *_ in site.SECTIONS}
    bodies["union_crisis"] = site._stage_chart_html(
        ["fear_early", "fear_full"],
        ["Early", "Full"],
        ["Early axes.", "Axes changed; early scale preserved."],
        500,
        "rate_10k",
    )
    page = site.build_html(
        {},
        {"speeches": 1, "words": 100, "presidents": 1,
         "start": 1789, "end": 2026},
        bodies,
        inline=False,
        page_kind="story",
    )
    assert "The chronological story ends here." in page
    assert 'href="summary.html">Continue to the Summary' in page
    assert 'data-story-range="1789–1808"' in page
    assert 'data-story-era="Establishing the republic"' in page
    assert "active.dataset.storyRange" in page
    assert "active.dataset.storyEra" in page
    assert 'aria-live="polite"' in page
    assert "ArrowLeft" in page and "ArrowRight" in page
    assert "prefers-reduced-motion: reduce" in page
    assert "if (!reduceMotion" in page
    assert 'get("motion") === "reduce"' in page
    assert 'classList.add("reduced-motion")' in page
    assert "scroll-margin-top:calc(var(--global-nav-height,48px) + 48px)" in page
    assert "overflow-x: auto; overflow-y: hidden" in page


def test_naming_crossover_is_derived_as_1910_from_the_real_corpus():
    speeches = corpus.load()
    rates = indices.yearly_rates(indices.build_markers(speeches))
    assert site.naming_crossover_year(rates) == 1910
    for year in range(1910, 1915):
        assert rates.loc[year, "america"] > rates.loc[year, "united_states"]
    assert not all(
        rates.loc[year, "america"] > rates.loc[year, "united_states"]
        for year in range(1909, 1914)
    )


def test_early_fear_stage_and_full_reveal_publish_honest_axes():
    speeches = corpus.load()
    rates = indices.yearly_rates(indices.build_markers(speeches))
    early = site.fig_fear_early(rates)
    full = site.fig_fear_full(rates)
    assert list(early.layout.xaxis.range) == [1789, 1868]
    assert list(full.layout.xaxis.range) == [1789, 2026]
    assert list(full.layout.xaxis2.range) == [1789, 1868]
    assert list(full.layout.yaxis2.range) == list(early.layout.yaxis.range)
    assert "Axis changed" in full.layout.annotations[0].text


def test_expansion_and_progressive_views_are_era_specific():
    paragraphs = pd.read_parquet(issues.PARA_LABELS_PATH)
    expansion = site.fig_expansion_story(
        expansion_story.load_metrics(),
        expansion_story.load_meta(),
    )
    assert len(expansion.data) == 2
    assert expansion.data[0].zmin == 0
    assert expansion.data[0].zmax == 60
    assert list(expansion.data[0].y) == list(expansion_story.TOPIC_ROWS)
    assert list(expansion.data[0].x)[-1] == "1846–49"
    assert list(expansion.layout.yaxis2.range) == [0, 40]
    delta = site._progressive_deltas(paragraphs)
    assert list(delta.columns) == [row[0] for row in site.PROGRESSIVE_PERIODS]
    assert "Civil rights & race" not in delta.index
    figures = [site.fig_progressive_delta(paragraphs, stage) for stage in range(1, 4)]
    assert [len(figure.data) for figure in figures] == [1, 2, 3]
    assert len({tuple(figure.layout.xaxis.range) for figure in figures}) == 1


def test_procedural_reveal_starts_in_admin_industrial_era_and_ends_with_trump():
    speeches = corpus.load()
    markers = indices.build_markers(speeches)
    stats = rhetoric.build_stats(speeches)
    scores = indices.president_scores(markers, stats, speeches).set_index("president")
    scores = scores[scores.n_speeches >= 5]
    first = site.fig_procedural_reveal(scores, 1)
    final = site.fig_procedural_reveal(scores, 3)
    first_names = {row[0] for trace in first.data for row in trace.customdata}
    final_names = {row[0] for trace in final.data for row in trace.customdata}
    assert first_names
    assert all(
        1869 <= int(scores.loc[name, "first_year"]) <= 1912
        for name in first_names
    )
    assert "Donald Trump" not in first_names
    assert "Donald Trump" in final_names
    assert tuple(first.layout.xaxis.range) == tuple(final.layout.xaxis.range)
    assert tuple(first.layout.yaxis.range) == tuple(final.layout.yaxis.range)


def test_coverage_pressure_interactive_uses_real_messages_and_every_receipt_arm():
    annotations = pd.read_parquet(
        REAL_DATA_DIR / "llm_annotations" / "paragraph_annotations.parquet"
    )
    body = site._coverage_pressure_v2_html(corpus.load(), annotations)
    assert "December 3, 1894: Second Annual Message" in body
    assert "January 12, 2016: 2016 State of the Union Address" in body
    assert "Every colored segment is a real paragraph" in body
    assert "+2.17" in body
    assert "-304" in body
    assert "4.29%" in body
    assert "All declared sensitivity treatments are now visible." in body
    for treatment in (
        "all speeches (raw)",
        "all speeches (genre-standardized)",
        "annual messages with length/paragraph/medium controls",
    ):
        assert treatment in body
    assert "Toy illustration" not in body


def test_broadcast_title_names_primary_assigned_form():
    speeches = corpus.load()
    stats = rhetoric.build_stats(speeches)
    annotations = pd.read_parquet(
        REAL_DATA_DIR / "llm_annotations" / "speech_annotations.parquet"
    )
    figure = site.fig_broadcast_presidency(stats, annotations)
    assert "Primary form assigned within this corpus" in figure.layout.annotations[0].text
    assert list(figure.layout.xaxis2.range) == [1900, 1980]


def test_written_republic_legend_clears_the_subplot_titles():
    speeches = corpus.load()
    markers = indices.build_markers(speeches)
    annotations = pd.read_parquet(
        REAL_DATA_DIR / "llm_annotations" / "speech_annotations.parquet"
    )
    figure = site.fig_written_republic(speeches, markers, annotations)
    assert figure.layout.margin.t >= 90
    assert figure.layout.legend.y >= 1.15
    assert figure.data[2].showlegend is False


def test_historical_event_markers_are_complete_and_non_causal_copy_is_shared():
    assert site.EXPANSION_STORY_EVENTS == [
        (1823, "Monroe Doctrine"),
        (1832, "Bank Veto and Nullification Proclamation"),
        (1846, "War with Mexico; territorial acquisition follows, 1846–48"),
    ]
    assert [year for year, _ in site.CIVIL_RIGHTS_EVENTS["Civil rights & race"]] == [
        1863, 1865, 1868,
    ]
    guide = site._event_guide_html(site.CIVIL_RIGHTS_EVENTS)
    assert "do not say the event caused" in guide
